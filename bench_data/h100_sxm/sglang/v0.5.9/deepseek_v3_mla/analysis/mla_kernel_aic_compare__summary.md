# MLA Kernel AIC 对比摘要

## 范围

- 对比对象是实机 `MLA时延拆解.csv` 中 attention 子模块对应的底层 CUDA kernel 时间，而不是整个 `self_attn/attn_mha` 模块包络时间。
- 主指标是 `real_attention_kernel_time_ms`：优先取 `kernel_summary_json` 中 attention 子模块全部 kernel 时间之和；若摘要列表不完整，则回退到 `child_timing_json.gpu_kernel_time_sum_ms`。
- AIC 同时输出两套查询口径：`query_*_mla()` 与 `query_*_attention()`。
- `real_gpu_makespan_ms` 与 `real_gpu_kernel_time_sum_ms` 继续保留为辅助参考。
- 由于实机 run 目录里没有直接暴露最终 `kv_cache_dtype`，本次同时输出 `bfloat16` 和 `fp8` 两种 AIC 仿真列。

## 统计

- 可比较总行数: `90`
- Prefill 行数: `90`
- Decode 行数: `0`
- 跳过行数: `0`

## Prefill

- `MLA / bfloat16` 对 `real_attention_kernel_time_ms` 的平均绝对百分比误差: `288.20%`
- `MLA / fp8` 对 `real_attention_kernel_time_ms` 的平均绝对百分比误差: `276.39%`
- `Attention / bfloat16` 对 `real_attention_kernel_time_ms` 的平均绝对百分比误差: `26.12%`
- `Attention / fp8` 对 `real_attention_kernel_time_ms` 的平均绝对百分比误差: `12.19%`

## 解释

- `query_context_mla()` 与 `query_context_attention()` 都会先按 `full_s = fresh_len + prefix_len` 查表，再做 prefix 修正。
- `query_generation_mla()` 与 `query_generation_attention()` 都按 `s = prompt_len + 1` 查询 decode 点。
- `query_*_attention()` 中，`attn_mha` 按 `n_kv = n`，`attn_mqa` 按 `n_kv = 1` 解释。
- 因为本批正式 case 大多是“每个请求长度一致”的设置，本次可以把每个 chunk/layer 映射到一个统一的 `(batch_size, fresh_len, prefix_len)` AIC attention 查询点。
