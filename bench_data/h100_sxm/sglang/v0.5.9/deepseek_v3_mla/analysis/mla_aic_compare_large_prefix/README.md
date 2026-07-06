# Prefill Kernel Envelope AIC Compare

## 范围

- 仅包含 prefill；所有 decode 行已排除。
- 实机侧只采用 `module_gpu_makespan_ms`，即 self_attn 内首个 GPU kernel 启动到最后一个 GPU kernel 结束的包络时间。
- AIC 侧比较两个方案：普通 `DeepSeekModel` fallback 小算子 kernel 时间累加，以及 `WideEPDeepSeekModel` 的 `qkv_a_proj + WideEPContextMLA` 模块级路径。
- 每个 case/layer 为一个实机样本，并按既有规则排除 `layer_id=0`。

## 规模

- 实机 layer 样本数: `32`
- case 数: `8`
- 跳过记录数: `8`，其中包含按规则忽略的首层。
- `context_mla_module` primary miss 验证: `通过`。

## 平均绝对百分比误差（相对实机 kernel 包络时间）

- DeepSeek fallback stack: `30.61%`
- WideEP module stack: `16.44%`
- Rule(prefix=0 -> WideEP, prefix>0 -> DeepSeek): `30.61%`
- Best-of-two by absolute error: `16.05%`

## 输出文件

- `prefill_real_layer_detail.csv`: 实机每 case/layer 明细，保留原始 child/kernel JSON。
- `prefill_aic_deepseek_fallback_stack.csv`: 普通 DeepSeek fallback 小算子查表明细。
- `prefill_aic_wideep_stack.csv`: WideEP 对齐 op 查表明细。
- `prefill_compare_summary.csv`: case 级总表，含两个 AIC 方案、rule 方案和 best-of-two 的 gap。
- `prefill_kernel_envelope_aic_stacked_compare.png/svg`: 本轮主图。
- `prefill_kernel_envelope_aic_rule_compare.png/svg`: 按 prefix 规则选择最终方案的新图。
- `prefill_real_kernel_envelope_boxplot.png/svg`: 实机包络时间跨层分布箱线图。

## 图中标注

- 灰色柱: 实机 kernel 包络时间均值。
- DeepSeek 堆叠柱: 普通 DeepSeekModel fallback 路径的小算子时间累加。
- WideEP 斜线堆叠柱: `context_qkv_a_proj_gemm + wideep_context_mla`。
- `DS` 和 `WE`: 分别表示两个 AIC 方案相对实机 kernel 包络时间的误差百分比。
- `Rule`: 应用规则后最终采用方案的误差百分比；规则为 `prefix=0 -> WideEP`、`prefix>0 -> DeepSeek`。
- 红色 `Best`: 仅在 Rule 不是最优时出现，表示两个 AIC 方案中误差绝对值更小者。
