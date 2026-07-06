# MLA AIC Compare 汇总说明

## 范围

- 仅包含 prefill；所有 decode 行已排除。
- 实机侧每个 case/layer 为一条样本，并按要求排除 `layer_id=0`。
- 实机主口径保留 `module_gpu_makespan_ms` 和 `module_gpu_kernel_time_sum_ms`，同时展开 `kernel_summary_json` 记录每个子模块底层 CUDA kernel 明细。
- 普通 `DeepSeekModel` AIC 侧验证 `context_mla_module` primary miss 后，按 SDK `FallbackOp` 的 5 个 fallback op 求和。
- `WideEPDeepSeekModel` AIC 侧纳入一个 `context_qkv_a_proj_gemm` 与 `WideEPContextMLA`；源码中重复出现的 `context_downscale_gemm` 不再重复计入。

## 规模

- 实机 layer 样本数: `72`
- case 数: `18`
- 跳过记录数: `18`，其中包含按规则忽略的首层。
- `context_mla_module` primary miss 验证: `通过`。

## 平均绝对百分比误差

- 普通 DeepSeek fallback vs real GPU makespan mean: `135.18%`
- 普通 DeepSeek fallback vs real GPU kernel-sum mean: `149.62%`
- WideEP aligned stack vs real GPU makespan mean: `12.02%`
- WideEP aligned stack vs real GPU kernel-sum mean: `45.18%`

## 输出文件

- `prefill_real_layer_detail.csv`: 实机每 case/layer 明细，保留原始 child/kernel JSON。
- `prefill_real_kernel_stack.csv`: 实机每 layer 的子模块/kernel 展开明细。
- `prefill_real_kernel_stack_case_mean.csv`: 实机每 case 的子模块均值堆叠输入。
- `prefill_aic_deepseek_fallback_stack.csv`: 普通 DeepSeek fallback 小算子查表明细。
- `prefill_aic_deepseek_primary_validation.csv`: `context_mla_module` primary miss 验证。
- `prefill_aic_wideep_stack.csv`: WideEP 对齐 op 查表明细。
- `prefill_compare_summary.csv`: case 级总表，含 gap 百分比。
- `prefill_real_layer_boxplot.png/svg`: 实机跨层重复样本箱线图。
- `prefill_deepseek_fallback_stacked_compare.png/svg`: 普通 DeepSeek fallback 堆叠对比。
- `prefill_wideep_stacked_compare.png/svg`: WideEP 对齐堆叠对比。

## 注意

- AIC 普通 DeepSeek fallback 使用 `kvcache_quant_mode=fp8`、`fmha_quant_mode=bfloat16`、`gemm_quant_mode=fp8_block`，与当前 SDK `DeepSeekModel` 及现有 H100/SGLang 0.5.9 数据可查口径一致。
- WideEP 模块表使用 `kvcache_quant_mode=fp8`、`fmha_quant_mode=fp8_block`、`attention_backend=fa3`。
- 图中每个 case 使用三根柱：real GPU makespan mean、real GPU kernel-sum mean、AIC total；其中 kernel-sum 与 AIC total 继续保留组成堆叠。
