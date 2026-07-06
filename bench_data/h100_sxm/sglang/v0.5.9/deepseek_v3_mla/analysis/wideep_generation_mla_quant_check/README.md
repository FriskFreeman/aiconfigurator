# WideEP Generation MLA Quantization Check

## Scope

- Existing AIC data file: `src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.9/wideep_generation_mla_perf.txt`.
- Temporary collector output: `collector_run/`.
- This checks whether a real `gemm_type=fp8_block, kv_cache_dtype=fp8` collector run differs materially from the legacy bf16 run whose rows were logged as fp8.

## Results

| kernel_source | num_heads | batch_size | effective_s | bf16_actual_legacy_logged_fp8 | existing_system_data | fp8_actual | bf16_actual_legacy_logged_fp8_gap_pct_vs_existing | fp8_actual_gap_pct_vs_existing | fp8_gap_pct_vs_bf16_actual |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fa3 | 128 | 1 | 512 | 0.1114 | 0.1128 | 54.2867 | -1.2411 | 48026.5071 | 48631.3285 |
| fa3 | 128 | 4 | 2048 | 0.1218 | 0.1216 | 54.3010 | 0.1645 | 44555.4276 | 44482.1018 |
| fa3 | 128 | 8 | 4096 | 0.1365 | 0.1381 | 54.3154 | -1.1586 | 39230.4852 | 39691.5018 |

## Findings

- Existing `wideep_generation_mla_perf.txt` rows match the temporary legacy run, where the collector actually uses `kv_cache_dtype=bfloat16` and `gemm_type=bfloat16` but logs the row as `fp8/fp8_block`. The three checked points differ by about `-1.2%`, `+0.2%`, and `-1.2%`.
- The existing data therefore should be treated as legacy bf16 module data with fp8-compatible labels, not as proof that a real fp8 collector run was used.
- The forced-fp8 collector run entered the intended fp8 setup (`kv_cache_dtype=fp8`, `gemm_type=fp8_block`, SGLang warning: `Using FP8 KV cache...`) but produced ~`54 ms` for all tested shapes. This is an abnormal collector/dummy-fp8 path result, not a plausible MLA module latency, and should not be used to overwrite the current database.
- Current decode real-run traces are nevertheless real fp8 execution: every checked default-FA3 CUDA-Graph decode run has `quant_method=fp8`, repeated `sm90_fp8_gemm_1d2d_impl` kernels, bf16-to-fp8 quant kernels, and `set_mla_kv_buffer_kernel`.
- The good AIC-vs-real module agreement is therefore a numerical coincidence/robustness observation under this workload: legacy bf16 collector module timing is close to real fp8 engine module makespan for the checked cases, even though the precision labels are not faithful.

## SGLang 0.5.9 Source Notes

- `deepseek_v2.py` stores `self.kv_cache_dtype = get_global_server_args().kv_cache_dtype`, so `kv_cache_dtype=fp8_e4m3` affects the attention module path.
- The KV write path selects `fp8_dtype` when `self.kv_cache_dtype == "fp8_e4m3"` before the fused rope/cache call.
- Output projection preparation quantizes `attn_bmm_output` with `fused_flatten_fp8_group_quant(...)` when `self.o_proj.weight.dtype == torch.float8_e4m3fn`.
- Source excerpts are saved in `sglang_0_5_9_deepseek_v2_source_excerpt.txt`.

## Output Files

- `collector_quant_experiment_raw_rows.csv`
- `collector_quant_experiment_compare.csv`
- `real_decode_fp8_trace_summary.csv`
- `real_decode_fp8_kernel_evidence.csv`
- `real_decode_fp8_trace_check.md`
- `sglang_0_5_9_deepseek_v2_source_excerpt.txt`
- `collector_run/docker_run.log`
- `collector_run/docker_run_fp8.log`
