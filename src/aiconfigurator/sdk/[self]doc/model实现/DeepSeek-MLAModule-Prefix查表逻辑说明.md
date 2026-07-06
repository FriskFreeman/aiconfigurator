# DeepSeek MLA Module Prefix 查表逻辑说明

## 1. 说明目标

本文专门解释 DeepSeek 系列模型在 AIC SDK 中，`MLAModule` 模块级查表遇到：

- `isl`：总输入前文长度
- `prefix`：已命中 KV cache 的前文长度

时，是如何完成仿真估计的。

关注重点有三点：

1. `backend` 执行层如何把 `isl/prefix` 传给模型与算子。
2. `DeepSeekModel` 在模块级表可用时，如何优先查 `mla_module`。
3. `collector` 只采了 `prefix=0` 的模块数据后，`PerfDatabase` 如何折算到 `prefix>0` 的场景，以及这会带来什么误差。

## 2. 总体结论

先给结论：

- AIC 执行层**明确区分** `isl` 和 `prefix`。
- 对 context/prefill 来说，真正参与本轮重算的 token 数是：

`effective_isl = isl - prefix`

- DeepSeek 模型优先查 `context_mla_module` / `generation_mla_module`。
- 其中 **context 模块级查表**面临一个关键不一致：
  - collector 采到的是 **`prefix=0` 全未命中** 的整模块数据
  - 但仿真执行时可能传入 **`prefix>0`**
- `PerfDatabase.query_context_mla_module()` 的处理方法是：
  - 先用 `full_s = s + prefix` 去查一条“全长、全未命中”的模块级样本
  - 再乘一个基于 attention 因果 token-pair 数的修正系数

因此：

- 这条路径**不是忽略 prefix**
- 但它对 prefix 的处理是一个**整模块级近似校正**
- 误差主要来自：**把模块内所有子项都按 attention 的二次复杂度一起缩放了**

## 3. 执行层如何处理 `isl/prefix`

在 [base_backend.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/backends/base_backend.py) 的 `_run_context_phase()` 中：

1. 先取：
   - `isl = runtime_config.isl`
   - `prefix = runtime_config.prefix`
2. 再计算：

`effective_isl = isl - prefix`

3. 然后对所有 context op 统一传入：

- `s = effective_isl`
- `prefix = prefix`

这说明：

- `isl` 代表总前文长度
- `prefix` 代表其中已经命中 KV cache 的部分
- `s` 不是总前文，而是本轮需要真正执行的 prefill token 数

如果 `effective_isl <= 0`，AIC 会直接报错，因为这意味着“没有任何未命中 token 需要计算”。

## 4. DeepSeek 模型如何组织 MLA block

在 [deepseek.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/models/deepseek.py) 中，DeepSeek V3 的 context 与 generation 都把 MLA block 组织成 `FallbackOp`：

### Context

- primary：
  - `ops.MLAModule("context_mla_module", ...)`
- fallback：
  - `context_downscale_gemm`
  - `context_q_b_proj_gemm`
  - `context_kv_b_proj_gemm`
  - `context_attention` / `context_mla`
  - `context_proj_gemm`

### Generation

- primary：
  - `ops.MLAModule("generation_mla_module", ...)`
- fallback：
  - `generation_downscale_gemm`
  - `generation_q_b_proj_gemm`
  - `generation_bmm_pre`
  - `generation_attention`
  - `generation_bmm_post`
  - `generation_proj_gemm`

所以 DeepSeek 仿真的策略是：

- 有模块级表时，优先直接查整块 MLA module
- 没有模块级表时，再退回更细粒度的 GEMM + MLA attention + BMM 组合

## 5. `MLAModule` 自己如何调用数据库

在 [operations.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/operations.py) 的 `MLAModule.query()` 中：

### Context

会调用：

- `database.query_context_mla_module(b=batch_size, s=s, prefix=prefix, ...)`

这里的 `s` 已经是 backend 传下来的 `effective_isl`。

### Generation

会调用：

- `database.query_generation_mla_module(b=batch_size, s=s, ...)`

generation 这条路径不接受 `prefix` 参数，因为 decode 模式下 `s` 本身就代表当前依赖的 KV cache 长度。

## 6. Context MLA module 的 prefix 修正逻辑

这是最关键的部分。

在 [perf_database.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/perf_database.py) 的 `query_context_mla_module()` 中：

### 6.1 输入语义

收到的参数是：

- `b`：batch size
- `s`：本轮待计算 token 数，也就是 `effective_isl`
- `prefix`：已命中 KV cache 的历史长度

因此总前文长度是：

`full_s = s + prefix`

### 6.2 实际查表方式

数据库不会直接去找：

- `(b, s, prefix)`

这种三元组的模块级表，因为 collector 根本没有采这种数据。

它做的是：

1. 先构造：

`full_s = s + prefix`

2. 再去模块表中插值查：

- `context_mla_module_data[..., num_heads][full_s][b]`

也就是把当前请求近似成：

- batch 为 `b`
- 序列长度为 `full_s`
- `prefix=0`

的一条全未命中模块样本。

### 6.3 Prefix 修正公式

查到原始模块延迟后，再乘：

`prefix_correction = (full_s^2 - prefix^2) / (full_s^2)`

最终得到：

`latency = raw_module_latency(full_s) * prefix_correction`

energy 也会乘同样的比例。

### 6.4 这个公式的含义

这个修正公式实际上来自 MLA context attention 的因果三角复杂度近似。

全未命中时，attention 计算量近似对应：

- 从 1 累加到 `full_s`

而 prefix 命中后，只需要重新计算最后 `s` 个 query 对应的那一块，因此复杂度近似对应：

- 从 `prefix+1` 累加到 `full_s`

用平方差近似，就是：

- `full_s^2 - prefix^2`

再除以原来的 `full_s^2`，就得到一个比例因子。

## 7. 为什么这会有误差

问题在于：**模块级 collector 采到的是整块 MLA module 的总延迟，不只是 attention。**

而整块 `context_mla_module` 至少还包含：

- downscale gemm
- q_b_proj gemm
- kv_b_proj gemm
- proj gemm
- attention backend 本身

其中只有 attention 这部分，才天然符合“按因果 token-pair 数二次缩放”的假设。

### 7.1 attention 部分

对 prefix 很敏感，二次修正有物理意义。

### 7.2 GEMM / 线性投影部分

更像是只和：

- `effective_isl = isl - prefix`

线性相关，而不是和：

- `full_s^2 - prefix^2`

相关。

### 7.3 整模块统一乘修正系数的问题

当数据库把整模块总延迟都乘上 `prefix_correction` 时，就等于：

- 把非 attention 部分也一起按 attention 的二次比例缩小了

这会带来系统性误差。

## 8. 误差表现会在什么场景更明显

### 情况 1：`prefix=0`

没有问题。

因为这时：

- `full_s = s`
- `prefix_correction = 1`

模块表和 collector 语义完全一致。

### 情况 2：`prefix` 较小

误差通常不大。

因为：

- `effective_isl` 仍然占大头
- 整模块里 attention 部分通常仍然主导

### 情况 3：`prefix` 很大、`effective_isl` 很小

误差最明显。

因为此时：

- attention 部分确实应该显著下降
- 但部分 GEMM / 投影项未必应该按同样比例下降

如果整模块统一乘平方差比例，就更容易低估真实延迟。

## 9. Fallback 路径为什么更细

`FallbackOp` 的意义就在这里。

如果 `context_mla_module` 表不可用：

- `FallbackOp` 会改用更细粒度的几个 op 分别查表再求和

而这条细粒度路径对 prefix 的处理更符合模块内部结构：

### GEMM

GEMM 在 backend 里收到的是：

- `x = batch_size * effective_isl`

所以只按未命中 token 数线性缩放。

### `ContextMLA`

`ContextMLA` 自己也会收到：

- `s = effective_isl`
- `prefix = prefix`

然后在 `query_context_mla()` 中，采用与模块级类似的：

- 查 `full_s = s + prefix`
- 乘 `prefix_correction`

也就是说：

- attention 子项用 prefix-aware 二次修正
- GEMM 子项只按未命中 token 线性缩放

这比“整模块统一乘 attention 比例”更细、更合理。

## 10. Generation MLA module 为什么没有这个问题

在 generation 路径：

- `query_generation_mla_module(b, s, ...)`

这里的 `s` 直接表示当前 decode token 看到的 KV cache 长度。

collector 的 generation module 本身就是在：

- 历史 KV 长度为 `s`
- 当前只解 1 个 token

这个语义上采出来的。

因此 generation 模块级查表不存在 context 那种：

- “collector 只采 `prefix=0`”
- “运行时却传 `prefix>0`”

的语义错位。

换句话说：

- **问题主要存在于 context/prefill 模块级 MLA**
- **generation MLA module 语义和 collector 更一致**

## 11. 数据库模式对这条路径的影响

在 `FallbackOp` 中，primary 的 `MLAModule` 查询会被强制切到：

- `DatabaseMode.SILICON`

目的很明确：

- 如果模块表缺失，不希望 HYBRID 模式悄悄给出一个经验估计
- 而是希望明确触发 fallback，改走细粒度真实表

这意味着：

- 模块表存在时，优先用真实模块表
- 模块表不存在时，不会先吃一个 HYBRID 的粗糙估计
- 而是直接回退到细粒度组合

这个设计本身就是在尽量压低模块路径上的误差。

## 12. 一句话总结

DeepSeek context MLA module 的 prefix 查表逻辑是：

1. backend 先把总 `isl` 拆成 `effective_isl = isl - prefix`
2. `MLAModule` 用 `(b, s=effective_isl, prefix)` 调 `PerfDatabase`
3. `PerfDatabase` 用 `full_s = s + prefix` 去查 **全未命中 collector 模块表**
4. 再乘一个基于 attention 因果复杂度的平方差修正

所以：

- 这条路径**考虑了 prefix**
- 但它是**整模块级近似校正**
- 误差主要来自：**把模块内非 attention 子项也一起按 attention 比例缩放**

而当模块表缺失时，AIC 会退回到更细的 GEMM + MLA attention 路径，这条 fallback 对 prefix 的处理会更细、更贴近真实结构。
