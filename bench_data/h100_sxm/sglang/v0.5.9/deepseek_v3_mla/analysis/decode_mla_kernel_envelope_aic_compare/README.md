# Decode CUDA Graph MLA Kernel Envelope AIC Compare

## 口径

- 仅保留 `decode_manifest.csv` 中 `attention_backend=auto/default`、未显式指定 `flashmla`、且 CUDA Graph 开启的 decode case。
- 实机侧没有可用的 host/module NVTX 边界，因此按 kernel 语义切 self_attn/MLA module：从 `quant` 后接 `deepgemm(2112,7168)` 的位置开始，到后续 `quant -> deepgemm(7168,16384)` 的 `o_proj` 结束。
- `FusedAddRMSNorm`、MoE gate/up/down、activation 等后续 kernel 不计入 MLA module。
- 主实机指标是 `real_mla_makespan_mean_ms`：每层首 kernel start 到尾 kernel end 的包络时间；同时保留 `real_mla_kernel_sum_mean_ms` 和组件拆分。
- AIC 侧比较 `DeepSeekModel` fallback 小算子累加，以及去重后的 WideEP module 路径：一个 `2112x7168` GEMM 加 `wideep_generation_mla(fa3)`。
- 当前 SDK 原样 WideEP 路径会额外包含一次 `generation_qkv_a_proj_gemm`，该值保留为 `aic_wideep_sdk_current_*` 诊断列，但主图和 `aic_wideep_dedup_*` 采用去重对齐口径。
- 标准 `generation_mla_module` 查询结果单独放在 `decode_aic_generation_mla_module_validation.csv`；当前 0.5.9 数据目录没有对应 silicon 文件时应为 miss。

## 结果概览

- 可比较 case 数: `9`
- 跳过/告警记录数: `0`
- `generation_mla_module` 全部 miss: `True`
- DeepSeek fallback vs makespan MAPE: `14.75%`
- WideEP module dedup vs makespan MAPE: `0.98%`
- WideEP SDK-current vs makespan MAPE: `10.69%`
- DeepSeek fallback vs kernel_sum MAPE: `15.88%`
- WideEP module dedup vs kernel_sum MAPE: `2.45%`
- WideEP SDK-current vs kernel_sum MAPE: `13.70%`

## 输出文件

- `decode_real_mla_layer_segments.csv`: 每个 case 每层的实机 MLA kernel range、makespan、kernel_sum、组件拆分和 kernel 名摘要。
- `decode_compare_summary.csv`: case 级汇总，含实机均值、AIC 总时延和误差。
- `decode_aic_deepseek_fallback_stack.csv`: DeepSeek fallback 小算子查表明细。
- `decode_aic_wideep_stack.csv`: WideEP generation module 路径查表明细，含 `include_policy` 标识去重主口径与 SDK 当前重复诊断行。
- `decode_aic_generation_mla_module_validation.csv`: 标准 `generation_mla_module` 查询校验。
- `decode_mla_kernel_envelope_aic_compare.png/svg`: 主对比图。
- `decode_mla_real_kernel_components.png/svg`: 实机 MLA range 内 kernel_sum 细粒度语义组件拆分图。
- `decode_mla_real_kernel_semantic_components.png/svg`: 同上，保留一个语义明确的文件名副本。
- `artifact_manifest.json`: 产物清单。

## 边界说明

本目录采用完整 `o_proj` 结束边界。较早只截到第二个 `nvjet_tst` 的口径会漏掉最后的 `quant + deepgemm(7168,16384)`，因此不是完整 MLA module。
如果后续 SGLang kernel 名或 DeepGEMM template 参数发生变化，需要优先检查 `segment_decode_mla_layers()` 的两个 shape 匹配条件。
