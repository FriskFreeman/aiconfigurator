# kv_b_proj prefix token sensitivity

This isolated analysis compares real H100/SGLang `kv_b_proj` timing with three AIC GEMM query semantics.

## Query Semantics

- `old_fresh_only`: current generic AIC `GEMM` behavior for context ops, `m=sum(fresh_lens)`.
- `direct_total_tokens`: direct prefix-aware fix, `m=sum(prefix_lens + fresh_lens)`.
- `sglang_chunk_policy`: SGLang 0.5.9 policy. If `sum(seq_lens) <= 128K`, MHA_ONE_SHOT queries `m=sum(prefix+fresh)` once; otherwise it queries fresh once plus prefix chunks with `prefix_chunk_len=(128K // batch_size)`.

## Summary vs real kv_b_proj kernel-sum

| source_name | estimate | case_count | mape_vs_kernel_sum_pct | mean_error_vs_kernel_sum_pct | max_abs_error_vs_kernel_sum_pct |
| --- | --- | --- | --- | --- | --- |
| mla_aic_compare_kernel_envelope_refresh | old_fresh_only | 26 | 43.91 | -38.06 | 94.87 |
| mla_aic_compare_kernel_envelope_refresh | direct_total_tokens | 26 | 5.76 | 4.73 | 21.54 |
| mla_aic_compare_kernel_envelope_refresh | sglang_chunk_policy | 26 | 5.76 | 4.73 | 21.54 |
| mla_aic_compare_large_prefix | old_fresh_only | 8 | 92.12 | -92.12 | 94.87 |
| mla_aic_compare_large_prefix | direct_total_tokens | 8 | 2.28 | 0.77 | 3.96 |
| mla_aic_compare_large_prefix | sglang_chunk_policy | 8 | 2.28 | 0.77 | 3.96 |

## Important Caveats

- Real main baseline is `kv_b_proj` child `gpu_kernel_time_sum_ms`, which includes the per-token FP8 quant kernel plus the DeepGEMM kernel.
- `real_gemm_only_mean_ms` is also exported for checking the narrower GEMM-only kernel口径.
- For TP1, `(n=32768,k=512)` is absent from `gemm_perf.txt`, so all three AIC variants still rely on 3D interpolation. Prefix-aware token counts fix semantics but not the missing-shape data gap.
- In the current real prefill cases, SGLang policy mostly equals `direct_total_tokens` because `sum(prefix+fresh)` stays below `128K` tokens, causing MHA_ONE_SHOT rather than multi-chunk prefix processing.

## Files

- `kv_b_proj_layer_detail.csv`: parsed per-layer real kv_b_proj timings.
- `kv_b_proj_case_comparison.csv`: case-level real mean and AIC estimates.
- `kv_b_proj_summary.csv`: aggregate error summary.
- `*_kv_b_proj_values.png/svg`: values by case.
- `*_kv_b_proj_error_pct.png/svg`: error percentages by case.
