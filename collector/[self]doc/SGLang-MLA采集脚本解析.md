# SGLang MLA 采集脚本解析

## 1. 文档目标

本文聚焦 `collector` 中与 SGLang MLA 相关的三类采集脚本：

- [collector/sglang/collect_mla.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/collector/sglang/collect_mla.py)
- [collector/sglang/collect_mla_module.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/collector/sglang/collect_mla_module.py)
- [collector/sglang/collect_mla_bmm.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/collector/sglang/collect_mla_bmm.py)

重点回答四个问题：

1. 这些脚本在 `collector` 总体工作流中的位置是什么。
2. 它们各自从 SGLang 的哪一层接口进入，到哪一层为止。
3. 输入测试配置、环境变量和输出 perf 文件分别是什么。
4. 执行时到底做了哪些步骤，哪些地方是真实调用 SGLang，哪些地方是 mock/简化。

## 2. 三个脚本的分工

| 脚本 | 采样粒度 | 进入 SGLang 的边界 | 输出文件 | 主要用途 |
| :--- | :--- | :--- | :--- | :--- |
| `collect_mla.py` | kernel/backend 级 MLA attention | 直接构造 `RadixAttention + attn backend + ForwardBatch` | `context_mla_perf.txt` / `generation_mla_perf.txt` | 测 SGLang MLA attention backend 本体的延迟 |
| `collect_mla_module.py` | module 级 self-attn | 走 `ServerArgs -> ModelConfig -> ModelRunner -> ScheduleBatch -> ForwardBatch -> self_attn` | `wideep_context_mla_perf.txt` / `wideep_generation_mla_perf.txt` / DSA 对应文件 | 测更接近真实模型执行路径的模块级开销 |
| `collect_mla_bmm.py` | MLA decode 前后两段 BMM 子核 | 基本不走完整 SGLang runtime，只复用 SGLang 的 FP8 量化 / kernel 封装 | `mla_bmm_perf.txt` | 拆分 decode 阶段两个关键矩阵乘子步骤 |

可以把它们理解成三层：

- `collect_mla_module.py` 最接近真实模型执行。
- `collect_mla.py` 再往下，只保留 attention backend 本体。
- `collect_mla_bmm.py` 最细，直接拆到 decode 前后 BMM 子步骤。

## 3. 放回 collector 共性工作流里看

SGLang MLA 相关 op 在注册表 [collector/sglang/registry.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/collector/sglang/registry.py) 中声明为：

| op 名 | 模块 | `get_func` | `run_func` | perf 文件 |
| :--- | :--- | :--- | :--- | :--- |
| `mla_context` | `collect_mla` | `get_context_mla_test_cases` | `run_mla` | `context_mla_perf.txt` |
| `mla_generation` | `collect_mla` | `get_generation_mla_test_cases` | `run_mla` | `generation_mla_perf.txt` |
| `mla_bmm_gen_pre` | `collect_mla_bmm` | `get_mla_gen_pre_test_cases` | `run_mla_gen_pre` | `mla_bmm_perf.txt` |
| `mla_bmm_gen_post` | `collect_mla_bmm` | `get_mla_gen_post_test_cases` | `run_mla_gen_post` | `mla_bmm_perf.txt` |
| `wideep_mla_context` | `collect_mla_module` | `get_wideep_mla_context_test_cases` | `run_mla_module_worker` | `wideep_context_mla_perf.txt` |
| `wideep_mla_generation` | `collect_mla_module` | `get_wideep_mla_generation_test_cases` | `run_mla_module_worker` | `wideep_generation_mla_perf.txt` |

它们统一由 [collector/collect.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/collector/collect.py) 调度：

1. `collect_sglang()` 读取 SGLang 版本。
2. `build_collections()` 按 registry 解析出要跑的 collection。
3. `collect_ops()` 动态导入 `get_func/run_func`，并用 `functools.partial` 固定 `perf_filename`。
4. `collect_module_safe()` 拉出测试用例列表。
5. worker 进程逐条执行测试用例，把结果通过 `log_perf()` 追加到对应 perf 文件。

这意味着三类脚本虽然内部复杂度不同，但外部接口都遵循同一套 collector 约定：

- 必须提供 `get_*_test_cases()`
- 必须提供 `run_*()`
- `run_*()` 要接受 `perf_filename`
- 最终都要写扁平 perf 文本

## 4. SGLang 接口边界总览

这是理解三个脚本差别的核心。

### 4.1 `collect_mla.py` 的边界

它没有创建完整 `ModelRunner`，而是自己造了一个极简 mock runtime：

- `MockModelConfig`
- `MockServerArgs`
- `MockModelRunner`
- `ReqToTokenPool`
- `MLATokenToKVPool`
- `ForwardBatch`
- `RadixAttention`
- SGLang 的 backend 实现类，例如：
  - `TRTLLMMLABackend`
  - `TritonAttnBackend`
  - `FlashAttentionBackend`

也就是说，它进入的是：

`SGLang attention backend / RadixAttention 层`

而不是：

`Engine / Scheduler / ModelRunner / 真模型 self_attn 层`

它测的是“后端 MLA attention kernel 路径本身”，不是完整模型模块。

### 4.2 `collect_mla_module.py` 的边界

它进入 SGLang 的层级明显更高，是真正走模型执行路径：

`ServerArgs -> ModelConfig -> ModelRunner -> ScheduleBatch -> ForwardBatch -> self_attn.forward`

其中关键点是：

- 通过 `load_model_runner()` 创建真实的 SGLang `ModelRunner`
- 从 `model_runner.model.model.layers[test_layer].self_attn` 取出目标 attention module
- 构造 `Req` / `ScheduleBatch` / `ForwardBatch`
- 再直接调用 module 的 `forward`

所以它不经过 HTTP server，也不经过 OpenAI API 层，但已经进入了 SGLang 内部“接近真实模型执行”的 runtime 边界。

### 4.3 `collect_mla_bmm.py` 的边界

它比 `collect_mla.py` 更低一层，主要复用：

- `sgl_kernel.bmm_fp8`
- `per_tensor_quant_mla_fp8`

不构造 `ForwardBatch`，也不走 attention module，只测 decode 前后两个 BMM 子步骤。

因此它更像“MLA 子核补充采样器”，用于给更上层模型补足细粒度数据。

## 5. `collect_mla.py` 详细解析

## 5.1 角色定位

`collect_mla.py` 是 SGLang MLA backend 的 kernel/backend 级 collector。

它的主要目标不是复现完整推理链路，而是：

- 在 SGLang backend 约束下构造合法 MLA attention 输入
- 直接调用 SGLang 的 MLA attention 实现
- 分别采 context 与 generation 两阶段延迟

这也是为什么它自己 mock 了 `ModelRunner/ServerArgs/ModelConfig`，而不是去加载真实模型。

## 5.2 输入测试配置

### Context 用例

`get_context_mla_test_cases()` 生成的单条 test case 结构为：

`[input_len, batch_size, output_len, kv_cache_dtype, num_heads, world_size, tp_size, tokens_per_block, warming_up, test_ite, is_context_phase]`

其中真正关键的是：

| 字段 | 含义 |
| :--- | :--- |
| `input_len` | prefill 序列长度 |
| `batch_size` | batch 大小 |
| `kv_cache_dtype` | `bfloat16` 或 `fp8` |
| `num_heads` | 总 attention head 数 |
| `tp_size` | 逻辑上模拟的 TP 切分 |
| `is_context_phase` | `True` |

### Generation 用例

`get_generation_mla_test_cases()` 结构相同，但：

- `input_len` 代表历史 `kv_cache` 长度
- `is_context_phase=False`

### backend 选择逻辑

`_select_default_mla_backend()` 会按 GPU 架构和 CUDA 版本模仿 SGLang 默认逻辑：

- SM100~109 且 CUDA >= 12.8: `trtllm_mla`
- SM90~99 且 CUDA >= 12.3: `fa3`
- 否则：`triton`

这说明它不是任意扫 backend，而是有意对齐“该机器上 SGLang 默认会选谁”。

## 5.3 执行过程

`run_mla()` 的执行路径可以拆成：

1. 设置 CUDA device、随机种子。
2. 创建 `MockModelRunner`，并根据 phase/backend 修正 `kv_lora_rank / v_head_dim / rope dim`。
3. 构造 `ReqToTokenPool` 与 `MLATokenToKVPool`。
4. 依据默认 backend 实例化：
   - `TRTLLMMLABackend`
   - `TritonAttnBackend`
   - `FlashAttentionBackend`
5. 构造 `RadixAttention` layer。
6. 按 context / generation 分别准备：
   - `q / k / v`
   - 可选 `q_rope / k_rope`
   - `ForwardBatch`
7. 对 decode 场景，还会把历史 KV 写入 `kv_pool.set_mla_kv_buffer(...)`。
8. 调用 `attn_backend.init_forward_metadata(forward_batch)`。
9. 通过 `benchmark_layer()` 和 `benchmark_with_power()` 做 warmup + timing + 功耗采样。
10. 用 `log_perf()` 写出结果。

## 5.4 输出边界

它输出到：

- `context_mla_perf.txt`
- `generation_mla_perf.txt`

核心字段为：

| 字段 | 含义 |
| :--- | :--- |
| `mla_dtype` | 这里固定写成 `bfloat16` |
| `kv_cache_dtype` | `bfloat16` / `fp8` |
| `num_heads` | local head 数，即 `num_heads / tp_size` |
| `batch_size` | batch |
| `isl` | context 时是输入长度；generation 时固定记为 `1` |
| `tp_size` | 逻辑 TP |
| `step` | context 为 `0`；generation 为历史长度 |
| `latency` | ms |
| `kernel_source` | `trtllm_mla` / `triton` / `flash_attention` |

## 5.5 单算子 MLA 中的 cache / 上下文配置

这里要特别区分一句：

- 它**不是**“完全不管 cache”
- 但它也**不是**在测真实 prefix hit / cache reuse 语义

更准确地说，`collect_mla.py` 是：

- **手工伪造最小可运行的 KV 上下文**
- **只为了让 SGLang MLA attention backend 能跑起来**
- **不去模拟真实请求级 prefix 命中逻辑**

### 5.5.1 它是怎么伪造 request-to-token 映射的

`create_req_to_token_pool()` 里直接创建：

- `ReqToTokenPool`
- `token_matrix`

其中 `token_matrix` 的逻辑是：

- 对每个 request 分配一段连续 token slot
- 位置从 `page_size` 之后开始排
- 然后直接写入 `pool.req_to_token[:batch_size, :total_len]`

这说明：

- request 到 token slot 的映射是人工连续构造的
- 没有真实调度器/缓存系统参与
- 只是为了满足 `ForwardBatch` 和 attn backend 对映射表的要求

### 5.5.2 Prefill/context 中的 cache 语义

在 `is_context_phase=True` 时：

- `total_len = input_len`
- `seq_lens = [input_len] * batch_size`
- `prefix_lens = 0`
- `forward_mode = EXTEND`
- `out_cache_loc = token_matrix.reshape(-1)`
- `extend_seq_lens = seq_lens`
- `extend_prefix_lens = prefix_lens`
- `extend_num_tokens = batch_size * input_len`

这组设置表达得非常明确：

- 这是一个 **纯 extend prefill**
- `prefix_lens = 0` 代表没有任何 prefix 命中
- 整段输入都需要被计算
- 同时会把结果写到 `out_cache_loc` 指定的 KV slot

所以在单算子 MLA context collector 里：

- **cache 会被写**
- 但 **没有 prefix reuse**
- 本质上仍然是“全未命中的 prefill”

### 5.5.3 Decode/generation 中的 cache 语义

在 `is_context_phase=False` 时：

- `history_len = input_len`
- `total_len = input_len + 1`
- `seq_lens = [history_len + 1] * batch_size`
- `forward_mode = DECODE`
- `out_cache_loc = token_matrix[:, history_len:]`
- `positions = history_len`

更关键的是这段：

- `history_loc = token_matrix[:, :history_len]`
- 人工生成 `cache_k / cache_k_rope`
- 调 `kv_pool.set_mla_kv_buffer(layer, history_loc, cache_k, cache_k_rope)`

这意味着 decode 不是在空 cache 上测，而是：

- 先手工往 KV pool 里塞入一段长度为 `history_len` 的历史 KV
- 然后测当前 1 token 对这段历史 KV 的 decode attention

所以单算子 MLA generation collector 的语义是：

- **历史 KV 是“人工预填充”出来的**
- **只测 decode token 对已有 KV 的访问**
- **并不关心这些 KV 来自真实 prefix 命中、prefill 写入还是别的来源**

### 5.5.4 预设的 batch / 上下文变量

单算子 MLA 的关键上下文变量是：

| 变量 | context | generation |
| :--- | :--- | :--- |
| `batch_size` | 测试 sweep 的 batch | 测试 sweep 的 batch |
| `input_len` | 真实 prefill 长度 | 真实历史 KV 长度 |
| `total_len` | `input_len` | `input_len + 1` |
| `prefix_lens` | 全 0 | 不使用 |
| `seq_lens` | `input_len` | `input_len + 1` |
| `positions` | `0..input_len-1` 展平 | 全部等于 `input_len` |
| `out_cache_loc` | 整段 token 写 cache | 只给新 decode token 分配写入位置 |

日志写出时：

- context：
  - `isl = input_len`
  - `step = 0`
- generation：
  - `isl = 1`
  - `step = input_len`

所以从 perf_database 视角看，单算子 MLA 的 decode 样本也表示：

- 当前只输入 1 个 token
- 它依赖的历史上下文长度为 `step`

### 5.5.5 和 module 级 collector 的差异

两者在 cache 语义上很像，但实现方式不同：

- `collect_mla_module.py`
  - 借助 `Req / ScheduleBatch / ForwardBatch`
  - 让 SGLang 自己把请求推进到 prefill/decode 状态

- `collect_mla.py`
  - 不走 request/batch 调度器
  - 直接手工构造：
    - `ReqToTokenPool`
    - `MLATokenToKVPool`
    - `ForwardBatch`
    - 历史 KV buffer

相同点是：

- 都不测 prefix cache 命中
- 都把 prefill 当成全未命中
- decode 都是在“已有固定历史 KV 长度”的前提下测 1 token

不同点是：

- module 级更接近真实 runtime
- 单算子级更像 backend 最小可运行 mock

## 5.6 这个脚本的边界特征

优点：

- 快，不需要加载完整模型
- 直接测 SGLang MLA backend 核心路径
- 对 perf_database 的 MLA kernel 级表格最友好

限制：

- 不是真实模型 `self_attn` 执行
- 不经过 `ScheduleBatch/Req` 等更高层调度结构
- 很多 runtime 语义是 mock 出来的

## 6. `collect_mla_module.py` 详细解析

## 6.1 角色定位

这是最重要的 MLA 模块级 collector。

它的目标是：

- 用真实的 SGLang `ModelRunner` 和真实模型结构
- 直接调用某层的 `self_attn`
- 分别测 context 与 generation 两个阶段
- 同时兼容：
  - DeepSeek-V3 的 MLA
  - DeepSeek-V3.2 / GLM-5 的 DSA

对你当前关心的“模型/模块级仿真与实机对比”来说，它比 `collect_mla.py` 更接近真实运行边界。

## 6.2 支持模型与模式

`SUPPORTED_MODELS` 定义：

| 模型 | attention 类型 |
| :--- | :--- |
| `deepseek-ai/DeepSeek-V3` | `mla` |
| `deepseek-ai/DeepSeek-V3.2` | `dsa` |
| `zai-org/GLM-5` | `dsa` |

这也解释了为什么文件名虽叫 `collect_mla_module.py`，但实际上已统一承载 MLA 和 DSA 两类 module collector。

## 6.3 输入配置与测试用例设计

### 外层 test case 不是“一个 shape 一个任务”

`_build_module_test_cases()` 很重要：它故意不把每个 `(batch_size, seq_len)` 都暴露给 `collect.py`，而是只暴露每个：

`(model, attn_type, num_heads, precision)`

组合的一条占位任务。

也就是说，registry 层看到的 test case 类似：

`[0, 0, num_heads, kv_dtype, compute_dtype, gemm_type, model_path, attn_type, attention_backend]`

这里前两个 `0` 只是占位。

真实的 `(batch_size, seq_len)` 大 sweep，是在子进程内部由：

- `get_context_test_cases()`
- `get_generation_test_cases()`

再次展开的。

这样设计的原因是：

- 每次模块采集都要加载完整 `ModelRunner`
- 如果把每个 shape 都单独交给 `collect.py`，会把模型重复加载成百上千次
- 现在改成“一个 subprocess 负责一整个 head/precision/model 组合”，大幅降低总开销

### 模块级默认只取少量精度和 head 数

`_get_module_precision_combos()` 只返回一个 baseline：

- `("bfloat16", "bfloat16", "bfloat16")`

`_MODULE_HEAD_NUMS` 也只保留：

- `128`
- `64`

原因写得很直白：完整 `ModelRunner` 加载本身很贵，所以模块采集只保留少量代表点。

## 6.4 SGLang 入口：`load_model_runner()`

这是整个脚本最关键的入口。

它真正使用的 SGLang 接口包括：

- `ServerArgs`
- `ModelConfig.from_server_args`
- `ModelRunner`
- `_set_envs_and_config`

执行过程如下：

1. 解析 `SGLANG_TEST_NUM_LAYERS` 和 `SGLANG_LOAD_FORMAT`。
2. 调 `_resolve_local_model_path()`，优先使用 AIC 本地缓存的 model config，而不是联网拉 HuggingFace。
3. 构造 `ServerArgs`：
   - `load_format`
   - `kv_cache_dtype`
   - `disable_radix_cache=True`
   - `disable_cuda_graph=True`
4. 根据 `gemm_type` 决定是否启用 `quantization="fp8"`。
5. 显式关闭 piecewise cuda graph：
   - 新版字段：`disable_piecewise_cuda_graph=True`
   - 旧版字段：`enable_piecewise_cuda_graph=False`
6. 写入 `attention_backend`。
7. 若是 dummy load，再通过 `json_model_override_args` 覆盖：
   - `num_hidden_layers`
   - `num_attention_heads`
   - `num_key_value_heads`
8. 构造 `ModelConfig`
9. 构造真实 `ModelRunner`
10. 对部分 DSA rope 连续性问题打 patch

这里有一个非常值得注意的点：

### 该 collector 主动绕开 CUDA Graph / PCG

因为在 `load_model_runner()` 里已经明确做了：

- `disable_cuda_graph=True`
- `disable_piecewise_cuda_graph=True` 或 `enable_piecewise_cuda_graph=False`

所以这个 collector 的模块采集逻辑，本来就不是为了验证 SGLang 的 cudagraph/PCG 能否工作，而是为了稳定地把 attention module 跑通并拿到 perf 数据。

这和你前面试验里碰到的 PCG 问题是完全一致的：collector 自己也在主动规避这条不稳定路径。

## 6.5 模块执行路径：`run_attention_torch()`

`run_attention_torch()` 做的是：

1. 从 `model_runner.model.model.layers[test_layer].self_attn` 取目标 module。
2. 为该 module 构造一个 `dummy_qkv_latent_func`。
3. 清空 KV pool / req pool。
4. 对每个 `(batch_size, seq_length, is_prefill)`：
   - prefill 走 `_run_prefill()`
   - decode 走 `_run_decode()`

这里的关键边界是：

- 不走 SGLang HTTP server
- 不走 Engine API 的外部 generate 请求
- 但已经完全走进了 SGLang 模型内部的 self-attn module forward

## 6.6 Prefill 执行过程：`_run_prefill()`

`_run_prefill()` 基本是在手工搭一个最小可运行的 SGLang prefill batch：

1. 构造多个 `Req`，每个请求带一段随机 token。
2. 创建：
   - `CacheInitParams`
   - `ChunkCache`
   - `ScheduleBatch`
3. 调 `batch.prepare_for_extend()`。
4. 通过 `batch.get_model_worker_batch()` 得到 worker batch。
5. 用 `ForwardBatch.init_new(model_worker_batch, model_runner)` 生成 `ForwardBatch`。
6. `model_runner.attn_backend.init_forward_metadata(forward_batch)` 初始化 backend metadata。
7. 构造：
   - `hidden_states`
   - `positions`
   - `AttentionInputs`
   - `BumpAllocator`
8. 设置 `get_attn_tp_context().set_attn_inputs(attn_inputs)`。
9. 直接调用 `attention_module(...)` 做 warmup 和 timed runs。
10. 统计均值并写 perf。

这个流程说明，它在“模块级”仍然不是完整推理，而是：

- 让 batch / cache / forward metadata 这些运行时结构尽量贴近真实
- 然后只测 attention module 本身

### 6.6.1 Prefill 中的 KV cache / prefix 上下文是怎么设的

这里有几个关键点：

1. `CacheInitParams(disable=True, ...)`
2. `req.prefix_indices = empty`
3. `req.fill_ids = req.origin_input_ids`
4. `req.extend_input_len = len(req.fill_ids)`
5. `batch.prepare_for_extend()`

这组设置的含义基本可以概括成：

- **prefix cache / chunk cache 复用被显式关闭**
- **每个请求都被当成“纯 extend、纯未命中”的 prefill**
- **不会走 prefix 命中后只补尾巴的路径**

更具体地说：

- `CacheInitParams(disable=True)` 让 `ChunkCache` 处于关闭状态，collector 不测试树形 prefix cache 命中。
- `prefix_indices` 被设为空 tensor，表示当前请求没有已有 prefix token 索引可复用。
- `fill_ids = origin_input_ids`，表示整段输入都作为本轮待处理 token。
- `extend_input_len = len(fill_ids)`，表示整段输入都属于 extend 区间。

因此，`_run_prefill()` 的 collector 语义不是“带缓存命中的 prefill”，而是：

- 一批长度为 `seq_length` 的全新请求
- 没有 prefix 命中
- 整段输入都要重新计算 attention
- 同时会为这些 token 分配 KV slot，供这次 prefill 写入 KV cache

换句话说，它更接近：

`纯 prompt prefill / 全未命中 / 不复用 prefix cache`

而不是：

`已有长 prefix，仅追加少量 suffix`

### 6.6.2 Prefill 里 collector 预设了哪些 batch 上下文

每个请求对象 `Req` 的关键字段预设如下：

| 字段 | 设定 |
| :--- | :--- |
| `origin_input_ids` | 随机生成、长度为 `seq_length` |
| `prefix_indices` | 空 |
| `fill_ids` | 等于整段 `origin_input_ids` |
| `extend_input_len` | `seq_length` |
| `logprob_start_len` | `0` |
| `sampling_params.max_new_tokens` | `1` |

批级别上：

- `batch_size` 就是本次要压测的请求数
- 所有请求长度相同，都是 `seq_length`
- `positions` 被构造成 `0..seq_length-1`，并对 batch 展平
- `hidden_states` 形状为 `batch_size * seq_length x hidden_size`

日志写出时：

- `isl = seq_length`
- `step = 0`

所以数据库里的 context/module 样本，表达的是：

- 一批 `batch_size`
- 每条请求输入长度为 `isl`
- 没有 decode step

## 6.7 Decode 执行过程：`_run_decode()`

## 6.7 Decode 执行过程：`_run_decode()`

decode 的逻辑比 prefill 多一步历史 KV 准备：

1. 同样构造 `Req / Cache / ScheduleBatch`
2. 先 `prepare_for_extend()`
3. 再写入 `batch.output_ids`
4. 然后 `prepare_for_decode()`
5. 得到 decode 用的 `ForwardBatch`
6. 构造：
   - `decode_hidden`
   - `decode_positions`
   - `AttentionInputs`
7. 定义 `kernel_func = attention_module(...)`
8. 先做数次 eager 预热，避免首次 JIT/autotune 污染后续 capture/timing
9. 再用 `benchmark_with_power()` 测量
10. 写 perf

这里的“先 eager 预热再 benchmark”非常关键，注释也写得很明确：是为了规避 DSA decode 在某些 backend 上第一次 JIT/autotune 干扰图捕获或计时的问题。

### 6.7.1 Decode 中的 KV cache 上下文是怎么设的

decode 的关键思想是：

- **不是直接在空 cache 上测 1 token decode**
- 而是**先人为构造出一个长度为 `seq_length` 的已存在历史上下文**
- 然后再测“下一 token”的 decode attention

它的步骤是：

1. 构造 `Req`，每个请求的 `origin_input_ids` 长度为 `seq_length`
2. 同样设置：
   - `prefix_indices = empty`
   - `fill_ids = origin_input_ids`
   - `extend_input_len = len(fill_ids)`
3. 额外设置：
   - `cached_tokens = 0`
   - `already_computed = 0`
4. 先调用 `batch.prepare_for_extend()`
5. 再设置 `batch.output_ids`
6. 再调用 `batch.prepare_for_decode()`

这个顺序非常重要。

它表达的是：

- 先让 SGLang runtime 走一次“把 prompt/history 放进 KV cache 的准备流程”
- 再切换到 decode 模式
- 最后测当前 1 token 对已有 KV cache 的查询

所以 decode collector 预设的是：

- **历史上下文已经存在**
- **本次只测下一 token decode**
- **测的是“带历史 KV 的 decode attention”**

但它并**不**单独区分：

- 历史 KV 是来自真实 prefix 命中
- 还是刚刚 extend 写进去

从这个 collector 的语义上，更准确地说，它是在构造：

`kv_cache_length = seq_length 的 decode 场景`

而不是在刻意测试某种 prefix hit/miss 比例。

### 6.7.2 Decode 中 `cached_tokens / already_computed` 的含义

代码里把：

- `req.cached_tokens = 0`
- `req.already_computed = 0`

都初始化为 0。

这意味着 collector **没有**预置“这个请求有一部分 token 已经命中过 cache 并完成过前向”的显式状态。随后真正把请求推到 decode 场景的，是：

- `prepare_for_extend()`
- `prepare_for_decode()`

也就是说，这个脚本不是在手工模拟复杂的 prefix-hit bookkeeping，而是依赖 SGLang 自己的 batch 准备流程，把请求推进到“可 decode”的状态。

### 6.7.3 Decode 里 collector 预设了哪些 batch 上下文

每个请求对象 `Req` 的关键字段预设如下：

| 字段 | 设定 |
| :--- | :--- |
| `origin_input_ids` | 随机生成、长度为 `seq_length`，代表已有历史上下文 |
| `prefix_indices` | 空 |
| `fill_ids` | 等于整段 `origin_input_ids` |
| `extend_input_len` | `seq_length` |
| `cached_tokens` | `0` |
| `already_computed` | `0` |
| `sampling_params.max_new_tokens` | `1` |

批级别上：

- `batch.output_ids` 会额外写入一个随机 token，作为切到 decode 的触发条件
- `decode_hidden` 形状是 `batch_size x hidden_size`
- `decode_positions` 被统一设为 `seq_length`

这表示当前测量的是：

- 已有 `seq_length` 长度的历史 KV
- 现在生成第 `seq_length + 1` 个位置上的 token

日志写出时分两种记法：

- WideEP MLA 兼容旧表：
  - `isl = seq_length`
  - `step = 0`
- DSA / 新 module 记法：
  - `isl = 1`
  - `step = seq_length`

两者本质上都在表达同一个 decode 场景：

- 当前只输入 1 个 decode token
- 它所依赖的历史上下文长度是 `seq_length`

## 6.8 Prefill / Decode 的 cache 语义总结

如果只从 `collect_mla_module.py` 的 collector 语义来看：

- **Prefill**
  - 不测 prefix 命中
  - 不测 cache reuse
  - 测的是全未命中的 prompt prefill

- **Decode**
  - 测的是已有长度为 `seq_length` 的历史 KV 上的单 token decode
  - 但这个历史 KV 是通过内部 batch 准备流程构造出来的
  - 不是显式模拟“prefix 命中多少、未命中多少”的细粒度混合场景

因此，这个 collector 更像是在测两个标准化端点：

1. `纯 prefill 端点`
2. `纯 decode 端点`

而不是测中间态的 cache hit/miss 混合请求。

## 6.9 子进程包装：`run_mla_module_worker()` 与 `_run_mla_subprocess()`

模块 collector 没有直接在主 worker 进程里跑，而是再套了一层 subprocess：

- `run_mla_module_worker()` 是 `collect.py` 看到的入口
- `_run_mla_subprocess()` 真正起一个新 Python 进程
- 新进程里调用 `run_mla_module(...)`

这样做的原因是：

- 模型加载重
- CUDA 上下文和框架内部状态容易污染后续任务
- 某些 configuration 会卡死、超时或非法访问
- 用 subprocess 容易做超时、隔离和失败恢复

`_run_mla_subprocess()` 还做了：

- `CUDA_VISIBLE_DEVICES` 隔离
- 120 秒超时
- stdout/stderr 合并回收
- 非 0 返回码时把最后 30 行日志拼进异常

这其实就是 collector 在模块级任务上的“保险丝”。

## 6.10 输出文件与字段

模块级 collector 输出主要有两类：

| attention 类型 | phase | 文件 |
| :--- | :--- | :--- |
| MLA wideep | context | `wideep_context_mla_perf.txt` |
| MLA wideep | generation | `wideep_generation_mla_perf.txt` |
| DSA | context | `dsa_context_module_perf.txt` |
| DSA | generation | `dsa_generation_module_perf.txt` |

核心字段为：

| 字段 | 含义 |
| :--- | :--- |
| `model` | HF 模型名 |
| `architecture` | 模型架构名 |
| `mla_dtype` | 记录用 dtype |
| `kv_cache_dtype` | KV dtype |
| `gemm_type` | `bfloat16` / `fp8_block` |
| `num_heads` | 当前测试 head 数 |
| `batch_size` | batch |
| `isl` | context 时是输入长度；部分 generation 记录法按兼容约定可能写 `1` 或 `seq_len` |
| `tp_size` | 这里固定记 `1` |
| `step` | context 为 `0`；generation 通常记历史长度 |
| `latency` | ms |

## 6.11 这个脚本的边界特征

优点：

- 最接近真实 SGLang 模型层执行
- 走真实 `ModelRunner`、真实 `self_attn`
- 可以采到调度、cache、module forward 的综合开销

限制：

- 仍不是完整 server/engine E2E 请求
- 用 dummy 权重和简化 batch，和真实 serving 仍有差异
- 为了稳定性主动关闭 cudagraph/PCG

## 7. `collect_mla_bmm.py` 详细解析

## 7.1 角色定位

这个脚本专门拆 MLA generation 阶段两个关键子步骤：

- `mla_gen_pre`
- `mla_gen_post`

它不是完整 attention collector，而是子 kernel collector。

## 7.2 输入配置

### `get_mla_gen_pre_test_cases()`

输出格式：

`[num_tokens, num_heads, dtype, num_warmups, num_runs]`

### `get_mla_gen_post_test_cases()`

格式相同，只是 `num_tokens` sweep 范围更长。

其中：

- `dtype` 为 `bfloat16` 或 `fp8`
- `num_heads` 从 `128` 一路降到 `1`

## 7.3 执行过程

### `run_mla_gen_pre()`

它测的是 decode 前半段：

- `q_nope x w_kc`

若是 `fp8`，则：

1. 用 `per_tensor_quant_mla_fp8()` 做输入量化
2. 调 `sgl_kernel.bmm_fp8()`

若是 `bfloat16`，则直接 `torch.bmm()`

### `run_mla_gen_post()`

它测的是 decode 后半段：

- `attn_output x w_vc`

同样：

- BF16 路径直接 `torch.bmm`
- FP8 路径先量化，再走 `bmm_fp8`

## 7.4 输出

它统一写到：

- `mla_bmm_perf.txt`

通过 `op_name` 区分：

- `mla_gen_pre`
- `mla_gen_post`

核心字段为：

| 字段 | 含义 |
| :--- | :--- |
| `bmm_dtype` | `bfloat16` / `fp8` |
| `num_tokens` | token 数 |
| `num_heads` | head 数 |
| `latency` | ms |

## 7.5 这个脚本的边界特征

优点：

- 细粒度
- 便于分析 MLA decode 分阶段开销
- 对估计模型中 attention 内部拆分项很有帮助

限制：

- 已经脱离完整 SGLang runtime
- 更像“子核补充表”，不是最终模块级真实性能

## 8. 三者之间的关系

可以把三者理解成一套递进式 MLA 采样体系：

### 第 1 层：模块级真实路径

`collect_mla_module.py`

用途：

- 最接近真实模型 attention module
- 适合给 perf_database 提供“模块级”表

### 第 2 层：backend 级 attention 路径

`collect_mla.py`

用途：

- 聚焦 SGLang MLA backend 本体
- 适合看不同 backend 的 attention 延迟差异

### 第 3 层：子核拆分

`collect_mla_bmm.py`

用途：

- 细分 decode 内部前后两段 BMM
- 适合做解释性分析和局部修正

## 9. 对 AIC 仿真的意义

从 AIC 的角度，这三类数据分别支撑不同粒度的建模：

| 层级 | 对仿真的价值 |
| :--- | :--- |
| module 级 | 最接近真实推理框架实际开销，适合直接进入 perf_database 查表 |
| backend/kernel 级 | 用于解释为什么某种 module latency 会变快或变慢 |
| 子核级 | 用于拆出 MLA 内部组成，辅助细化估计模型或 debug 偏差来源 |

如果你的目标是“DeepSeek-V3/V3.2 在 AIC 中的模块/算子级仿真与实机对比校准”，那么建议的阅读优先级是：

1. 先看 `collect_mla_module.py`
2. 再看 `collect_mla.py`
3. 最后用 `collect_mla_bmm.py` 理解 decode 内部拆分

## 10. 一句话总结

SGLang 的 MLA collector 不是单一脚本，而是一套分层采样体系：

- `collect_mla_module.py` 负责“真实 self-attn 模块级”采集
- `collect_mla.py` 负责“MLA backend 核心路径级”采集
- `collect_mla_bmm.py` 负责“decode 子 BMM 级”采集

它们共同遵循 `collector` 的注册表 + test case + run_func + log_perf 工作流，但进入 SGLang 的边界逐层下沉，因此适合分别服务于模块级数据库、后端 kernel 对比和内部算子拆解分析。
