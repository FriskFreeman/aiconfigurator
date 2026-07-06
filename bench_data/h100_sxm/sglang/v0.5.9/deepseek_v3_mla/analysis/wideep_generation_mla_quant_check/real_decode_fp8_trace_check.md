# Real Decode FP8 Trace Check

- Checked runs: `9`
- All checked runs have fp8 config + fp8 DeepGEMM + bf16-to-fp8 quant + MLA KV write kernels: `True`
- `engine_kwargs.json` does not explicitly store `kv_cache_dtype`, so KV-cache fp8 is inferred from the decode MLA source path and `set_mla_kv_buffer_kernel` fed by fp8 quantized buffers, plus SGLang source code.

## Evidence Files

- `real_decode_fp8_trace_summary.csv`
- `real_decode_fp8_kernel_evidence.csv`
