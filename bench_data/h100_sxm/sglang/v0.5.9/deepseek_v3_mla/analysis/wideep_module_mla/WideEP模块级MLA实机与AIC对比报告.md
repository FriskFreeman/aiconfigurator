# WideEP 模块级 MLA 实机与 AIC 对比报告

## 1. 任务边界

本报告是一个独立对比任务，不与先前的 `mla_kernel_aic_compare*` 单算子 attention 对比混用。

本轮对比关注的是：

- AIC 侧：`wideep_context_mla_perf.txt` / `wideep_generation_mla_perf.txt`
- SDK 入口：`WideEPDeepSeekModel` 中的 `WideEPContextMLA` / `WideEPGenerationMLA`
- 实机侧：复用已有 `MLA时延拆解.csv`，不新增实机运行

先前的单算子对比关注 `attn_mha` / `attn_mqa` 子模块的底层 attention kernel。本轮则关注更大的模块级 MLA 口径，包含 q/kv 后续投影、attention、o_proj 等一组子模块。

## 2. 关键代码证据

### 2.1 collector 数据确实落在 WideEP perf 文件

`collector/sglang/collect_mla_module.py` 中，`attn_type == "mla"` 的 WideEP 路径会写入：

- context: `wideep_context_mla_perf.txt`
- generation: `wideep_generation_mla_perf.txt`

其日志字段包括：

- `kernel_source`
- `mla_dtype`
- `kv_cache_dtype`
- `gemm_type`
- `num_heads`
- `batch_size`
- `isl`
- `tp_size`
- `step`
- `latency`

H100 / SGLang 0.5.9 数据文件当前存在于：

- `src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.9/wideep_context_mla_perf.txt`
- `src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.9/wideep_generation_mla_perf.txt`

当前数据规模：

- `wideep_context_mla_perf.txt`: 1000 行
- `wideep_generation_mla_perf.txt`: 1056 行
- `kernel_source`: `fa3` / `flashinfer`
- `mla_dtype`: `fp8_block`
- `kv_cache_dtype`: `fp8`

### 2.2 SDK 确实在 WideEPDeepSeekModel 中使用 WideEP MLA ops

`src/aiconfigurator/sdk/models/deepseek.py` 中，`WideEPDeepSeekModel` 的 attention 部分使用：

- context: `ops.WideEPContextMLA("context_attention", ...)`
- generation: `ops.WideEPGenerationMLA("generation_attention", ...)`

同时，代码注释明确说明：

- `qkv_a` / downscale 类投影被建模为单独 GEMM
- `WideEPContextMLA` / `WideEPGenerationMLA` 覆盖的是后续 MLA attention forward 相关部分

这也是本报告中额外提供 “WideEP 对齐子模块口径” 的原因：实机完整模块中包含 `fused_qkv_a_proj_with_mqa`，但 SDK WideEP MLA op 不应重复包含这部分。

### 2.3 PerfDatabase 查表路径

`src/aiconfigurator/sdk/perf_database.py` 中：

- `query_wideep_context_mla()`
  - 从 `_wideep_context_mla_data[kernel_source][fmha_quant_mode][kvcache_quant_mode]` 查表
  - 使用 `num_heads = 128 // tp_size`
  - 按 `(num_heads, full_s, batch)` 插值
  - 使用 prefix correction: `(full_s^2 - prefix^2) / full_s^2`
- `query_wideep_generation_mla()`
  - 从 `_wideep_generation_mla_data[kernel_source][kvcache_quant_mode]` 查表
  - 使用 `num_heads = 128 // tp_size`
  - 按 `(num_heads, batch, s)` 插值

本轮脚本使用：

- `fmha_quant_mode = fp8_block`
- `kvcache_quant_mode = fp8`
- `attention_backend = fa3` 作为主结果
- `attention_backend = flashinfer` 作为参考结果

## 3. 新增产物

新增脚本：

- `.self/task-localbench/formal-prefill-stage1/compare_wideep_mla_module_with_aic.py`

主结果目录：

- `bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/analysis/wideep_module_mla/`

主结果文件：

- `wideep_module_mla_aic_compare__comparison.csv`
- `wideep_module_mla_aic_compare__skipped.csv`
- `wideep_module_mla_aic_compare__summary.md`
- `wideep_module_mla_aic_compare__summary.json`

参考结果文件：

- `wideep_module_mla_aic_compare_flashinfer__comparison.csv`
- `wideep_module_mla_aic_compare_flashinfer__skipped.csv`
- `wideep_module_mla_aic_compare_flashinfer__summary.md`
- `wideep_module_mla_aic_compare_flashinfer__summary.json`

## 4. 对比口径

每个 chunk/layer 生成一行对比数据。

实机侧保留多个时间口径：

- `real_module_total_to_last_kernel_ms`
  - 每层 attention module host 入口到最后一个关联 GPU kernel 结束
- `real_module_host_duration_ms`
  - host 端 module 函数持续时间
- `real_module_gpu_makespan_ms`
  - module 内首个 GPU kernel 开始到最后一个 GPU kernel 结束
- `real_module_gpu_kernel_time_sum_ms`
  - module 内所有 GPU kernel 时间加总
- `real_wideep_aligned_gpu_kernel_time_sum_ms`
  - 剔除 `fused_qkv_a_proj_with_mqa` 后的子模块 GPU kernel 时间加总
- `real_wideep_aligned_gpu_makespan_ms`
  - 剔除 `fused_qkv_a_proj_with_mqa` 后的子模块 GPU makespan
- `real_wideep_aligned_host_duration_sum_ms`
  - 剔除 `fused_qkv_a_proj_with_mqa` 后的子模块 host duration 加总

WideEP 对齐子模块包含：

- Prefill
  - `q_a_layernorm`
  - `q_b_proj`
  - `kv_a_layernorm`
  - `rotary_emb`
  - `kv_b_proj`
  - `attn_mha` 或 `attn_mqa`
  - `o_proj`
- Decode
  - `q_a_layernorm`
  - `kv_a_layernorm`
  - `q_b_proj`
  - `rotary_emb`
  - `attn_mqa`
  - `o_proj`

## 5. 结果摘要

### 5.1 `attention_backend=fa3` 主结果

覆盖情况：

- 可比较行数: 190
- Prefill 行数: 90
- Decode 行数: 100
- 跳过行数: 0

MAPE 摘要：

- Prefill 对 `real_module_gpu_makespan_ms`: 15.88%
- Prefill 对 `real_wideep_aligned_gpu_makespan_ms`: 16.98%
- Prefill 对 `real_module_gpu_kernel_time_sum_ms`: 40.21%
- Prefill 对 `real_wideep_aligned_gpu_kernel_time_sum_ms`: 49.25%
- Decode 对 `real_module_gpu_kernel_time_sum_ms`: 3.02%
- Decode 对 `real_wideep_aligned_gpu_kernel_time_sum_ms`: 18.73%
- Decode 对 `real_module_gpu_makespan_ms`: 42.50%
- Decode 对 `real_wideep_aligned_gpu_makespan_ms`: 35.57%

### 5.2 `attention_backend=flashinfer` 参考结果

覆盖情况：

- 可比较行数: 190
- Prefill 行数: 90
- Decode 行数: 100
- 跳过行数: 0

MAPE 摘要：

- Prefill 对 `real_module_gpu_makespan_ms`: 17.44%
- Prefill 对 `real_wideep_aligned_gpu_makespan_ms`: 19.30%
- Prefill 对 `real_module_gpu_kernel_time_sum_ms`: 43.44%
- Prefill 对 `real_wideep_aligned_gpu_kernel_time_sum_ms`: 52.64%
- Decode 对 `real_module_gpu_kernel_time_sum_ms`: 2.74%
- Decode 对 `real_wideep_aligned_gpu_kernel_time_sum_ms`: 17.02%
- Decode 对 `real_module_gpu_makespan_ms`: 43.20%
- Decode 对 `real_wideep_aligned_gpu_makespan_ms`: 36.36%

## 6. 初步结论

### 6.1 Prefill：WideEP 模块级数据更接近 GPU makespan 口径

Prefill 阶段，WideEP MLA AIC 数据对 GPU makespan 的 MAPE 约 16% 到 19%，明显好于对 GPU kernel 加总口径的 40% 到 53%。

这说明 WideEP context MLA 表更像在捕捉一段模块执行窗口，而不是简单的 kernel duration 加总。考虑到实机 trace 中 host 下发和 GPU 执行存在异步重叠，这个结果是合理的。

### 6.2 Decode：WideEP generation MLA 更接近整块 module GPU kernel 加总

Decode 阶段，WideEP generation MLA 对 `real_module_gpu_kernel_time_sum_ms` 的 MAPE 约 3%，非常接近。

但对 makespan 的误差较大，主要因为 decode 的实机模块中存在明显的 host/GPU 异步、等待和尾部间隔。对于 decode 这种很短的单 token workload，单次 module 的 makespan 容易被调度间隔放大。

### 6.3 `fa3` 与 `flashinfer` 两套 WideEP 表差异不大

两套 backend 的整体趋势接近：

- Prefill: `fa3` 略好
- Decode: `flashinfer` 对 kernel sum 略好，但差异很小

鉴于当前实机运行并不是 WideEP runtime，且 trace backend 多为 `auto`，报告主结果采用 `fa3`，同时保留 `flashinfer` 作为参考。

## 7. 边界与风险

### 7.1 当前实机不是 WideEP 运行时

这轮复用的是普通 SGLang engine 实机 trace，而不是 WideEP / DeepEP runtime trace。

因此，本报告验证的是：

- WideEP module-level MLA 数据和普通实机模块级 MLA 形态的数值相近程度
- SDK WideEP MLA ops 与已有 trace 的可对齐程度

它不验证：

- WideEP 通信行为
- DeepEP/WideEP 调度行为
- 多卡通信重叠
- 真正 WideEP runtime 的端到端保真度

### 7.2 `qkv_a` 边界仍需谨慎

SDK `WideEPDeepSeekModel` 将 `qkv_a` / downscale 作为单独 GEMM，而 collector 的 WideEP module 数据采集边界来自完整 attention module forward。

本报告通过 `real_wideep_aligned_*` 额外剔除 `fused_qkv_a_proj_with_mqa`，但这仍是基于 trace 子模块名的近似对齐。后续若要做严格口径，需要结合 collector 侧 NVTX 或源码确认其计时边界是否完全排除了 qkv_a。

### 7.3 Prefix 大点存在插值边界告警

脚本运行时，对极大 prefix / total length 点出现若干插值边界告警，例如 `z=51200`、`z=1025`、`z=513` 的邻域不完整。

这些行仍生成了结果，但应在分析极长 prefix 点时单独检查。

## 8. 建议

1. 后续若要正式采用 WideEP module-level MLA 表，需要明确 SDK 仿真目标是对齐 `gpu_makespan` 还是 `gpu_kernel_time_sum`。
2. Prefill 更建议优先看 `real_module_gpu_makespan_ms` 或 `real_wideep_aligned_gpu_makespan_ms`。
3. Decode 更建议优先看 `real_module_gpu_kernel_time_sum_ms`。
4. 若要继续缩小偏差，应新增真正 WideEP runtime 的实机 trace，而不是继续只用普通 SGLang engine trace 外推。
