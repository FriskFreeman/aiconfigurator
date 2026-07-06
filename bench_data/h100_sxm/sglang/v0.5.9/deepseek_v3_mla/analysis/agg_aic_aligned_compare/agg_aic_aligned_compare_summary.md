# AGG Mixed 实机与 AIC run_agg 对比摘要

## 输入与口径

- 实机输入：`bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/agg_csv/formal_aic_aligned_agg_summary.csv`，共 12 个 case。
- 实机主指标：每个 case 的 `MLA时延拆解.csv` 中，六层 MLA 模块的 GPU kernel 时间包络 `real_mla_gpu_envelope_all_layers_ms`；同时保留 makespan 求和、kernel 净时延求和、首层/非首层均值。
- AIC 主入口：`InferenceSession.run_agg(...)`，导出最终端到端结果字段 `ttft/tpot/request_latency/tokens/s`，并保留 run_agg 内部 `per_ops_data`。
- AIC 配置：`model=deepseek-ai/DeepSeek-V3`，`system/backend/version=h100_sxm/sglang/0.5.9`，`tp/pp/dp=1/1/1`，`layers=6`，`gemm=fp8_block`，`moe=fp8_block`，`kvcache=fp8`，`fmha=bfloat16`，`database_mode=SILICON`。

## 映射方式

- `intended_case`：使用用例设计参数，`batch_size=decode_batch_size+prefill_batch_size`，`isl=prefill_prefix_len+prefill_fresh_len`，`prefix=prefill_prefix_len`，`ctx_tokens=prefill_batch_size*isl`。
- `observed_first_batch`：使用 Nsight/meta 解析到的首个 formal mixed batch 实际组成，`batch_size=first_formal_prefill_req_count+first_formal_decode_req_count`，`ctx_tokens=prefill_req_count*observed_isl`。

## 关键结果

- AIC 成功行数：24/24。
- `observed_first_batch` 平均 `real/AIC context_mla_block`：1.792x。
- `strict_prefix0` case 数：9；`prefix_probe` case 数：3。
- 注意：AIC `ttft/tpot/request_latency` 是 run_agg 服务级端到端结果，不应直接解释为单个 mixed batch 的 MLA 模块时间；更可比的细粒度列是 `aic_mix_context_mla_block_ms` 与实机的 MLA GPU 包络。

## 产物

- `real_metrics.csv`：实机侧从每个 Nsight CSV 复算的 MLA GPU 指标。
- `aic_run_agg_results.csv`：AIC `run_agg` 结果、调度字段和 per-op JSON。
- `real_aic_joined.csv`：实机和 AIC 合并后的对比表。
- `summary_by_alignment_class.csv`：按 `strict_prefix0/prefix_probe` 聚合的摘要。
- `mla_envelope_vs_aic_mla_block_observed.png/svg`：实机 MLA 六层 GPU 包络 vs AIC run_agg MLA block。
- `real_to_aic_mla_block_ratio_observed.png/svg`：实机/AIC MLA block 比值。
- `aic_run_agg_e2e_metrics_intended.png/svg`：AIC run_agg 端到端输出指标。

## 风险与下一步

- `prefix_probe` 中，SGLang 实际首个 formal batch 可能因 prefix cache/chunked prefill 调度而与设计值不同；因此报告同时保留 intended 与 observed 两套映射。
- 当前对比先满足“run_agg 最终端到端结果”需求，但真正与 Nsight 单 batch GPU 包络最接近的是 run_agg 的 `mix_step.context_mla_block` 分解项。
- 后续若要做严谨误差归因，应固定一个 case，进一步对齐 AIC `mix_step_latency_ms` 的非 attention/MLA/MoE 子项与 Nsight 对应模块包络。
