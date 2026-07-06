# SGLang Chunked Prefill与Collector对照分析

本文基于当前保留的 `formal-prefill-stage1` 运行结果做静态整理，不新增任何新的试运行。

说明：

- 先前曾做过一组通过脚本手动改写 `chunked_prefill_size / max_prefill_tokens` 的对齐实验。
- 这组实验产生了 `6144+2048`、`7168+1024`、`15360+1024` 等“非法异形”分块，不再视为遵循原生 SGLang 调度逻辑。
- 这些实验目录现已删除，本文只保留当前有效结果与原生机制分析。

目标是回答三件事：

1. 目前到底跑了哪些实例，真实执行时在 SGLang 调度下变成了哪些 `prefill` 输入形状。
2. 这些形状背后的 SGLang 源码机制是什么。
3. `collector/sglang/collect_mla_module.py` 虚拟采集的 shape 与实机执行情况能否对得上。

## 1. 当前已完成的运行实例

截至目前，`formal-prefill-stage1` 中保留了 6 个 `nsys` 运行实例。按“请求配置”分组如下：

| 请求配置 | 已完成实例数 | 代表 run |
| --- | --- | --- |
| `bs=1, isl=512` | 4 | `20260610_102158_*`, `20260610_103906_*`, `20260610_111357_*`, `20260610_111613_*` |
| `bs=4, isl=2048` | 1 | `20260610_104304_*` |
| `bs=16, isl=1024` | 1 | `20260610_104521_*` |

这里“请求配置”指的是我们喂给 Engine 的原始请求形状；但 `nsys` 中真正看到的 `self_attn.attn_mha` 输入形状，未必仍然是原始的 `bs * isl` 一次性执行。

## 2. 现有实机结果实际产生了哪些prefill形状

下文中的形状，均指 `MLA对齐解析.json` 里 `prefill` 阶段 `attn_mha` 的第一个输入维度，即：

- `Inputs = [[tokens, 128, 192], ...]`

其中 `tokens` 是本轮 `self_attn` 实际吃进去的总 token 数。

### 2.1 结果总表

| 请求配置 | 实际prefill形状 | 是否单块对齐collector | 说明 |
| --- | --- | --- | --- |
| `1 x 512` | `512` | 是 | 单请求，无切分 |
| `4 x 2048` | `8192` | 是 | 单轮完成，完全命中 collector 点 |
| `16 x 1024` | `8192 + 8192` | 否 | 原生 SGLang chunk prefill，按 8 个请求 + 8 个请求切成两轮 |

### 2.2 按实例归纳

#### `bs=1, isl=512`

- `20260610_102158_*`：`512`
  - 结构上是单块，但无 warmup，首层编译污染很重。
- `20260610_103906_*`：`512`
- `20260610_111357_*`：`512`
- `20260610_111613_*`：`512`

结论：

- 这一组始终是单块 `512`。
- 是目前最稳定的 collector-aligned 样本。

#### `bs=4, isl=2048`

- `20260610_104304_*`：`8192`

结论：

- 当前保留结果里，这一组是 clean 单块。
- 可以直接与 collector 的 `(bs=4, isl=2048)` 点对照。

#### `bs=16, isl=1024`

- `20260610_104521_*`：`8192 + 8192`

结论：

- 这是当前保留结果里最典型的“原生 chunked prefill”。
- 总请求是 `16 x 1024 = 16384` token，而 H100 默认 `chunked_prefill_size = 8192`，因此自然分成两轮 `8192 + 8192`。

这组表现出同一个特征：

- SGLang 并不是在“请求内部”切 token。
- 它更像是在“请求边界”上决定当前这一轮到底装进几个完整请求。

## 3. 背后的SGLang源码机制

### 3.1 H100上的默认chunk设置

在 `sglang v0.5.9` 中，`ServerArgs` 对 H100/A100 这一档显存容量默认设置：

- `chunked_prefill_size = 8192`
- `max_prefill_tokens = 16384`

可见：

- `server_args.py` 中 H100 分支明确把默认 `chunked_prefill_size` 设成 `8192`。
- `ServerArgs.max_prefill_tokens` 的默认值是 `16384`。

这与我们先前观察到的“H100 上 chunk 大小大致是 8192”是一致的。

### 3.2 我们当前脚本对chunk参数的影响

当前 `formal-prefill-stage1` 脚本已经恢复为默认不改写 SGLang 的 chunk 预算。

先前在开启 `align_collector_prefill` 时，会把：

- `chunked_prefill_size = batch_size * prompt_len`
- `max_prefill_tokens = batch_size * prompt_len`

直接传入 Engine。

这类 override 已被判定为偏离原生 SGLang 行为，因此相关 run 已删除。

### 3.3 真正控制切分的不是“总token超了没”，而是PrefillAdder的装箱逻辑

`Scheduler` 在进入 prefill 调度时，会构造一个 `PrefillAdder`：

- `rem_input_tokens = max_prefill_tokens`
- `rem_chunk_tokens = chunked_prefill_size`

然后依次从 `waiting_queue` 中尝试把请求加入当前 prefill batch。

关键判断在 `PrefillAdder.add_one_req(...)` 中：

- 如果当前请求的 `input_tokens >= rem_input_tokens`，并且当前 batch 已经不为空，就直接停止把它加进本轮 batch。
- 如果 `input_tokens <= rem_chunk_tokens`，才会把该请求整个加入当前 batch。
- 如果放不下，才进入“截断单个请求”的 chunk 分支。

对于当前保留的有效 stage1 实验，真正发生的主要不是“单个请求被中途截断”，而是：

- 一轮 batch 先装进若干个完整请求
- 到最后一个请求时，因为等号条件或剩余 chunk budget，用 `OTHER` 停掉
- 这个最后的完整请求被挪到下一轮 prefill

因此在当前保留结果中，会自然出现：

- `8192 + 8192 = 8 x 1024 + 8 x 1024`

这与实机 trace 中 `16 x 1024` 的分块结果是一致的。

### 3.4 为什么dynamic chunking在这里不是主因

当前 Engine 路径下我们显式传了：

- `enable_dynamic_chunking = False`

同时源码中也规定：

- `enable_dynamic_chunking` 只有在 `pp_size > 1` 时才真正启用。

而当前所有 stage1 实验都是：

- 单卡
- `pp_size = 1`

所以：

- 这里的切分现象不是 pipeline parallel 的 dynamic chunking 造成的。
- 它主要来自固定 budget 下的 prefill request packing 逻辑。

## 4. Collector的虚拟shape是怎么构造的

### 4.1 `collect_mla_module.py` 的测试shape来源

`collector/sglang/collect_mla_module.py` 中，context phase 的模块测试 shape 来自两层逻辑：

1. `_build_wideep_mla_test_cases()`
   - 负责选择模型、backend、head_num 等“大组别”
2. `run_attention_torch(...)-> _run_prefill(...)`
   - 在单个子进程里真正遍历 `(batch_size, seq_length)` 点

而 `(batch_size, seq_length)` 的 sweep 范围本身来自：

- `_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]`
- `_SEQ_LENGTHS = [1, 4, 8, ..., 16384]`

也就是说：

- batch size 只取 2 的幂
- sequence length 也只取预定义离散点

因此 collector 数据中不存在：

- `bs=3`
- `bs=7`
- `bs=15`

这点很重要，因为实机 split 后恰恰大量落到了这些“非 2 的幂 batch”上。

### 4.2 collector里的prefill并不经过真实scheduler多轮调度

在 `_run_prefill(...)` 中，collector 的做法是：

1. 创建 `batch_size` 个 `Req`
2. 每个请求都构造 `origin_input_ids`，长度为 `seq_length`
3. 显式设置：
   - `prefix_indices = empty`
   - `fill_ids = origin_input_ids`
   - `extend_input_len = len(fill_ids)`
4. `disable_radix_cache=True`
5. `ScheduleBatch.init_new(...).prepare_for_extend()`
6. 直接取：
   - `attention_module = model_runner.model.model.layers[test_layer].self_attn`
7. 手动调用一次完整的 `attention_module(...)`

因此 collector 的模块计时语义是：

- 全无 cache hit
- 单次完整 module forward
- 输入 shape 与测试点 `(batch_size, seq_length)` 一一对应
- 不经过 server/engine 层那种多轮 request packing 调度

换句话说，collector 记录的是一种“理想单块 module 调用”。

## 5. collector data与当前实机结果能否对上

### 5.1 可以直接对上的情况

`wideep_context_mla_perf.txt` 中，针对：

- `DeepSeek-V3`
- `fa3`
- `num_heads=128`

确实存在以下直接命中点：

| collector点 | 是否存在 |
| --- | --- |
| `(1, 512)` | 是 |
| `(4, 2048)` | 是 |
| `(16, 1024)` | 是 |
| `(8, 4096)` | 是 |
| `(8, 1024)` | 是 |
| `(2, 4096)` | 是 |

因此当实机执行也保持为单块时，例如：

- `1 x 512 -> 512`
- `4 x 2048 -> 8192`

就可以把实机 trace 中整块 `self_attn` 时长，和 collector 对应行做比较。

### 5.2 当前剩余结果中的无法直接对上情况

当前保留结果中，无法直接与 collector 单块对齐的只剩下一类：

| 原始请求 | 实机执行 | collector是否有直接单块行 |
| --- | --- | --- |
| `16 x 1024` | `8192 + 8192` | 否 |

原因是：

- collector 的 `mla_module` 记录的是单次完整 module forward；
- 而实机这里已经被 SGLang 原生 scheduler 切成了两次 `self_attn`；
- 因此它不能直接等价成 collector 的 `(bs=16, isl=1024)` 单块记录。

## 6. 总结结论

结论可以压缩为三条：

1. 当前保留的有效 stage1 实机 prefill 结果里，只有 `16 x 1024` 会自然触发原生 SGLang chunked prefill，表现为 `8192 + 8192` 两轮执行。
2. 这套现象与 `sglang v0.5.9` 的 `PrefillAdder` 与 `chunked_prefill_size=8192` 逻辑是对得上的。
3. `collector/sglang/collect_mla_module.py` 采到的是理想单块 module 调用，而不是 Engine 层真实 scheduler 多轮执行；因此只有“单块、collector-aligned”的实机 run，才适合直接做一对一比较。

因此，当前阶段最可靠的对比对象仍然是：

- `1 x 512` 的 clean 单块样本
- `4 x 2048` 的 clean 单块样本

而 `16 x 1024` 这类原生 split case：

- 仍然有分析 SGLang 真实调度逻辑的价值
- 但不应直接当作 collector `mla_module` 的 exact-match 对照样本
