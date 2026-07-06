# run_agg Ops Breakdown 与 WideEP Attention 字段核查

## 产物

- `run_agg_ops_breakdown_long.csv`：从 `aic_per_ops_json` 展开的逐 op 长表。
- `run_agg_ops_breakdown_by_category.csv`：按 `op_category` 聚合后的 per-step 与 weighted-total 贡献。
- `observed_mix_step_category_summary.csv`：`observed_first_batch` 映射下 mixed step 的类别平均贡献。
- `observed_mix_step_ops_breakdown.png/svg`：每个 observed mixed step 的类别堆积。
- `observed_weighted_total_ops_breakdown.png/svg`：每个 observed case 在 run_agg 总累加时延中的类别堆积。
- `intended_weighted_total_ops_breakdown.png/svg`：每个 intended case 在 run_agg 总累加时延中的类别堆积。

## observed_first_batch mixed step 平均组成

- `context_moe`: avg 41.752 ms, avg share 56.3%
- `context_mla_block`: avg 30.005 ms, avg share 36.3%
- `context_shared_ffn`: avg 3.097 ms, avg share 3.9%
- `context_other`: avg 2.662 ms, avg share 3.5%
- `attention_scaled`: avg 0.000 ms, avg share 0.0%

## WideEP 字段语义核查结论

确认存在语义风险：`SGLANGBackend.run_agg()` 通过硬编码字段名 `context_attention` / `generation_attention` 将 attention 从 `run_static` 结果中单独取出并做缩放/混合处理；但 `WideEPDeepSeekModel` 中这两个名字对应的实际 op 类并不是普通 attention 单算子，而是：

- `context_attention` -> `ops.WideEPContextMLA`
- `generation_attention` -> `ops.WideEPGenerationMLA`

因此代码运行层面通常不会 KeyError，因为字段名仍然存在；但语义层面会把 WideEP MLA module 级查询结果当作 “attention 单算子” 处理。对于当前非 WideEP 的这批对比，AIC 结果里主要表现为 `context_mla_block` 已在 first pass 中被当作 non-attention 累加，而 `context_attention (scaled)` 为 0；若切到 SGLang DeepEP/WideEP 模型，`context_attention (scaled)` 将实际代表 `wideep_context_mla_perf.txt` 的模块级 WideEP MLA，而不是普通 FA/attention kernel。

代码证据：

- `src/aiconfigurator/sdk/backends/sglang_backend.py:147-177`：first pass 只排除 `layer_name != "context_attention"`，second pass 直接读取 `latency_dict["context_attention"]` 并按 `scale_factor` 折算。
- `src/aiconfigurator/sdk/backends/sglang_backend.py:194-198`：decode attention 也直接读取 `latency_dict["generation_attention"]`。
- `src/aiconfigurator/sdk/models/deepseek.py:1078-1085`：`WideEPDeepSeekModel` 将 `ops.WideEPContextMLA` 命名为 `"context_attention"`。
- `src/aiconfigurator/sdk/models/deepseek.py:1209-1216`：`WideEPDeepSeekModel` 将 `ops.WideEPGenerationMLA` 命名为 `"generation_attention"`。
- `src/aiconfigurator/sdk/operations.py:1269-1276` 与 `src/aiconfigurator/sdk/operations.py:1315-1323`：这两个 WideEP op 实际查询的是 `query_wideep_generation_mla` / `query_wideep_context_mla`，即 WideEP MLA 表，而不是普通 attention 表。

建议后续修复方向不是简单改字段名，而是让 `run_agg` 显式从模型/op 类型判断 “需要特殊折算的 prefill attention 项”：

- 普通 attention：`ContextAttention` / `GenerationAttention`
- DeepSeek MLA 单算子或模块：`ContextMLA` / `GenerationMLA` / `MLAModule`
- WideEP module：`WideEPContextMLA` / `WideEPGenerationMLA`

然后在 `per_ops_data` 中使用语义化字段，例如 `context_prefill_attn_like_scaled`，同时保留原始 op 名和 op class，避免 `context_attention` 这个名字在不同模型中承担不同含义。
