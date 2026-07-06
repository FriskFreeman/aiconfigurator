# WideEP MLA Module AIC 对比摘要

## 范围

- 本报告独立于先前的 `mla_kernel_aic_compare*` 单算子 attention 对比。
- AIC 侧查询 `wideep_context_mla_perf.txt` / `wideep_generation_mla_perf.txt`，对应 SDK 的 `WideEPContextMLA` / `WideEPGenerationMLA`。
- 实机侧复用已有 `MLA时延拆解.csv`，不重新跑实机。
- 主表同时保留整块 module 口径与剔除 `fused_qkv_a_proj_with_mqa` 后的 WideEP 对齐子模块口径。
- 本次 AIC 查询使用 `attention_backend=fa3`、`fmha_quant_mode=fp8_block`、`kvcache_quant_mode=fp8`。

## 统计

- 可比较行数: `190`
- Prefill 行数: `90`
- Decode 行数: `100`
- 跳过行数: `0`

## Prefill

- `整块 module 端到端到尾 kernel` MAPE: `59.00%`
- `整块 module GPU makespan` MAPE: `15.88%`
- `整块 module GPU kernel 加总` MAPE: `40.21%`
- `WideEP 对齐子模块 GPU makespan` MAPE: `16.98%`
- `WideEP 对齐子模块 GPU kernel 加总` MAPE: `49.25%`
- `WideEP 对齐子模块 host duration 加总` MAPE: `602.09%`

## Decode

- `整块 module 端到端到尾 kernel` MAPE: `97.52%`
- `整块 module GPU makespan` MAPE: `42.50%`
- `整块 module GPU kernel 加总` MAPE: `3.02%`
- `WideEP 对齐子模块 GPU makespan` MAPE: `35.57%`
- `WideEP 对齐子模块 GPU kernel 加总` MAPE: `18.73%`
- `WideEP 对齐子模块 host duration 加总` MAPE: `87.25%`

## 口径说明

- `real_module_total_to_last_kernel_ms` 是 nsys 解析得到的每层 attention 模块 host 入口到最后一个关联 kernel 结束的时间。
- `real_module_gpu_kernel_time_sum_ms` 是该模块下所有 GPU kernel 时长加总。
- `real_wideep_aligned_*` 会从子模块中排除 `fused_qkv_a_proj_with_mqa`，因为 SDK 的 `WideEPDeepSeekModel` 将 qkv_a/downscale 作为单独 GEMM，而 `WideEPContextMLA` / `WideEPGenerationMLA` 本身覆盖 q/kv 后续投影、attention、o_proj 等部分。
- 当前实机 trace 来自普通 SGLang engine 路径，不是 WideEP 运行时；因此该表是模块级 MLA 数据口径对照，而不是 WideEP 通信/调度保真验证。
