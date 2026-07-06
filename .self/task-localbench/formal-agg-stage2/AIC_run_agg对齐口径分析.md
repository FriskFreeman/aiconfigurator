# AIC run_agg 与实机 mixed 用例对齐口径分析

本文记录当前 SGLang mixed 实机用例与 AIC `SGLANGBackend.run_agg()` 的对齐判断。结论先行：**现有实机数据已经能证明 mixed batch 捕获链路可用，但只有一部分 prefix=0 且 decode KV 长度与 prompt 总长一致的 case 可直接作为 AIC `run_agg` 的近似对照；对于 prefix cache 命中场景，建议重新创建一组 AIC-aligned 用例，并明确其仍是对 AIC 当前近似模型的验证，而不是真实 scheduler 的完全等价复现。**

## 1. AIC run_agg 的参数语义

代码入口：

- `src/aiconfigurator/sdk/backends/sglang_backend.py`
- `src/aiconfigurator/sdk/backends/base_backend.py`
- `src/aiconfigurator/sdk/rust_engine_step.py`

关键语义如下：

- `runtime_config.isl` 是请求总输入长度，不是 fresh extend 长度。
- `runtime_config.prefix` 是总输入中已经 KV cache 命中的前缀长度。
- `BaseBackend._run_context_phase()` 会使用 `effective_isl = isl - prefix`，并把 `s=effective_isl, prefix=prefix` 传给各 context op。
- `run_agg()` 的 `ctx_tokens` 是 aggregate mixed step 中的 context token budget，但调度数学使用 `ceil(isl * b / ctx_tokens)` 和 `ceil(ctx_tokens / isl)`，没有把 `prefix` 从 scheduler token budget 中扣除。
- generation/decode attention 不单独接收 decode prefix；Python 路径中使用 `isl + osl // 2` 作为平均 decode KV 长度。

这意味着 AIC `run_agg` 本质是假设：

- 所有 context 请求共享同一个 `(isl, prefix)`。
- 所有 decode 请求共享同一个由 `(isl, osl)` 推导出的平均 KV 长度。
- context 请求数近似为 `ceil(ctx_tokens / isl)`，decode 请求数近似为 `b - ceil(ctx_tokens / isl)`。

## 2. 实机 mixed 用例的真实语义

当前 stage2 runner 在 SGLang 0.5.9 容器内触发真实 `ForwardMode.MIXED`，并在 `mixed_batch_meta.jsonl` 中记录：

- prefill 请求的 `prefix_lens`、`fresh_lens`、`seq_lens_after`。
- decode 请求的 `kv_lens`、`output_lens`、`seq_lens_after`。
- scheduler 的 `chunked_prefill_size`、`max_prefill_tokens`、`chunked_req_rid`。

实机侧的 mixed attention token 数基本对应：

```text
sum(prefill fresh tokens) + decode request count
```

cached prefix 不进入 chunked prefill fresh token budget，但会影响 attention 的 KV 长度和 prefix-cache 查询形态。

## 3. 现有 8 个正式 case 的可比性

下面按“是否可直接映射到 AIC 当前 Python `run_agg` 口径”分类。

### 3.1 可直接近似对齐

- `agg_uniform_small_d4_ctx512_p4_isl512_prefix0`
  - 实机：prefill `4 x fresh512 prefix0`，decode KV 约 `512`。
  - AIC 建议：`b=8, isl=512, prefix=0, ctx_tokens=2048`，`gen_tokens=4`。
  - 这是最干净的 small prefix=0 mixed 对照。

- `agg_uniform_chunk_boundary_d8_ctx8192_p1_isl8192_prefix0`
  - 实机：prefill `1 x fresh8192 prefix0`，decode KV 约 `8192`，并触达 chunk boundary。
  - AIC 建议：`b=9, isl=8192, prefix=0, ctx_tokens=8192`，`gen_tokens=8`。
  - 可以用于 chunk boundary 附近的 prefix=0 对照，但要注意真实 SGLang 仍受 chunked prefill 细节影响。

### 3.2 可作为真实 mixed 行为样本，但不适合直接对齐 AIC

- `agg_uniform_small_prefix_d8_ctx512_p4_isl512_prefix512`
  - 实机实际 prompt 总长是 `512 prefix + 512 fresh = 1024`，decode KV 约 `512`。
  - 若按 AIC `isl=512,prefix=512`，context fresh 为 0，不合法。
  - 若按 AIC `isl=1024,prefix=512`，decode KV 又明显偏短。

- `agg_uniform_prod_short_d16_ctx1024_p8_isl256_prefix1024`
  - 实机 prompt 总长是 `1280`，decode KV 约 `1024`。
  - AIC 的 `isl/prefix` 标量无法同时表示该 prefill 总长和 decode KV。

- `agg_uniform_mid_d4_ctx4096_p4_isl2048_prefix1024`
  - 实机 prompt 总长是 `3072`，decode KV 约 `4096`。
  - P/D 长度故意不同，适合观察真实 mixed 调度，不适合作为 AIC scalar case。

- `agg_uniform_long_prefix_d8_ctx2048_p2_isl4096_prefix4096`
  - 实机 prompt 总长是 `8192`，decode KV 约 `2048`。
  - P/D 长度差异很大，不能直接映射到单一 AIC `isl`。

- `agg_uniform_long_decode_d4_ctx16384_p2_isl2048_prefix2048`
  - 实机 prompt 总长是 `4096`，decode KV 约 `16384`。
  - 这是长 decode KV 压力样本，不是 AIC uniform prompt 样本。

- `agg_uniform_dense_decode_d32_ctx512_p4_isl1024_prefix0`
  - 实机 prefill 总长是 `1024`，decode KV 约 `512`。
  - prefix=0 但 P/D prompt 长度不一致，建议重跑一个 decode KV 为 `1024` 的版本。

## 4. 是否需要重新创建 AIC 假设一致的用例

需要。原因不是现有实机数据错误，而是它覆盖的是更真实、更自由的 P/D 混合形态；AIC `run_agg` 只能表达单一 scalar `(isl, prefix, osl, b, ctx_tokens)`。

建议新增两类用例。

### 4.1 严格 apples-to-apples 用例

优先使用 prefix=0，让真实 SGLang scheduler token budget 与 AIC `ctx_tokens` 数学一致。

已执行并归档的 strict prefix=0 case：

- `agg_aic_p0_small_b8_isl512_ctx2048`
  - 实机：`decode_batch_size=4, decode_prefix_len=512, prefill_batch_size=4, prefill_fresh_len=512, prefill_prefix_len=0`。

- `agg_aic_p0_decode_heavy_b12_isl512_ctx2048`
  - 实机：`decode_batch_size=8, decode_prefix_len=512, prefill_batch_size=4, prefill_fresh_len=512, prefill_prefix_len=0`。

- `agg_aic_p0_prefill_heavy_b16_isl512_ctx4096`
  - 实机：`decode_batch_size=8, decode_prefix_len=512, prefill_batch_size=8, prefill_fresh_len=512, prefill_prefix_len=0`。

- `agg_aic_p0_mid_b8_isl2048_ctx8192`
  - 实机：`decode_batch_size=4, decode_prefix_len=2048, prefill_batch_size=4, prefill_fresh_len=2048, prefill_prefix_len=0`。

- `agg_aic_p0_decode_heavy_b17_isl2048_ctx2048`
  - 实机：`decode_batch_size=16, decode_prefix_len=2048, prefill_batch_size=1, prefill_fresh_len=2048, prefill_prefix_len=0`。

- `agg_aic_p0_chunk_b9_isl8192_ctx8192`
  - 实机：`decode_batch_size=8, decode_prefix_len=8192, prefill_batch_size=1, prefill_fresh_len=8192, prefill_prefix_len=0`。

- `agg_aic_p0_dense_b36_isl1024_ctx4096`
  - 实机：`decode_batch_size=32, decode_prefix_len=1024, prefill_batch_size=4, prefill_fresh_len=1024, prefill_prefix_len=0`。

- `agg_aic_p0_dense_prefill_b40_isl512_ctx4096`
  - 实机：`decode_batch_size=32, decode_prefix_len=512, prefill_batch_size=8, prefill_fresh_len=512, prefill_prefix_len=0`。

- `agg_aic_p0_longctx_b8_isl4096_ctx8192`
  - 实机：`decode_batch_size=6, decode_prefix_len=4096, prefill_batch_size=2, prefill_fresh_len=4096, prefill_prefix_len=0`。

### 4.2 prefix-cache 误差评估用例

这类 case 不是严格等价，而是为了评估 AIC 当前 prefix 近似在真实 mixed 下的偏差。

推荐构造原则：

- 实机 prefill：`prefill_prefix_len=prefix`，`prefill_fresh_len=isl-prefix`。
- 实机 decode：`decode_prefix_len` 尽量设为 `isl`，正式 mixed 捕获时 decode KV 会约等于 `isl + 已生成少量 token`。
- AIC：`RuntimeConfig.isl=isl`，`prefix=prefix`。
- `ctx_tokens` 若按 AIC-native 口径，应设为 `prefill_batch_size * isl`，从而让 `ceil(ctx_tokens / isl)` 等于实机 prefill 请求数。
- 但真实 SGLang chunk budget 只消耗 `prefill_batch_size * (isl-prefix)` fresh tokens，因此该类对比应标注为“prefix scheduler 近似误差评估”。

已执行并归档的 prefix-cache probe case：

- `agg_aic_prefix_probe_b12_isl1024_prefix512_ctx4096`
  - 实机：`decode_batch_size=8, decode_prefix_len=1024, prefill_batch_size=4, prefill_prefix_len=512, prefill_fresh_len=512`。
  - AIC：`b=12, isl=1024, prefix=512, ctx_tokens=4096`。

- `agg_aic_prefix_probe_b24_isl1280_prefix1024_ctx10240`
  - 实机：`decode_batch_size=16, decode_prefix_len=1280, prefill_batch_size=8, prefill_prefix_len=1024, prefill_fresh_len=256`。
  - AIC：`b=24, isl=1280, prefix=1024, ctx_tokens=10240`。
  - 该 case 会明显暴露 AIC 以 total `isl` 而不是 fresh tokens 做 context 调度计数的问题。

- `agg_aic_prefix_probe_b12_isl4096_prefix2048_ctx8192`
  - 实机：`decode_batch_size=10, decode_prefix_len=4096, prefill_batch_size=2, prefill_prefix_len=2048, prefill_fresh_len=2048`。
  - AIC：`b=12, isl=4096, prefix=2048, ctx_tokens=8192`。

上述 12 个 case 已在 2026-06-30 正式运行完成，全部 `ok`。归档入口：

- manifest：`bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/manifest_agg.csv`
- 汇总：`bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/agg_csv/formal_aic_aligned_agg_summary.csv`
- 原始运行：`bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/raw_runs/*agg_aic_*`
- 解析产物：`bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/agg_csv/*agg_aic_*`

## 5. 对比范围边界

对齐时必须先选定比较对象：

- 如果比较 AIC `run_agg()` 的 `mix_step_latency_ms`，实机侧也应取整个 formal mixed step 的 GPU kernel envelope，而不是只取 `MLA时延拆解.csv` 的单层 MLA module。
- 如果比较 AIC per-op attention/MLA 子项，实机侧应使用对应 module 或 attention 子 module 的 `gpu_makespan_ms`、`module_gpu_kernel_time_sum_ms`，不要与整步 latency 混比。
- 当前用户偏好的 `gpu_makespan` 是合理的，因为它是 kernel 时间包络，更接近真实 GPU critical path；但在 CUDA stream/异步 host 下，不能用 host module duration 代替。

建议后续表格中至少保留：

- `aic_b`
- `aic_isl_total`
- `aic_prefix`
- `aic_ctx_tokens`
- `aic_expected_prefill_reqs = ceil(ctx_tokens / isl)`
- `aic_expected_decode_reqs`
- `real_prefill_fresh_lens`
- `real_prefill_prefix_lens`
- `real_decode_kv_lens`
- `real_gpu_makespan_ms`
- `comparison_scope = full_mixed_step | mla_module | attention_kernel`
- `alignment_class = strict_prefix0 | prefix_approx | scheduler_behavior_only`

## 6. 结论

现有 8 个 mixed 实机结果不需要废弃，它们对 SGLang 真实 mixed 行为很有价值；但若目标是验证 AIC 当前 `run_agg` 的 scalar 假设，应该新增一批规整用例。

最优先可直接对比的是 prefix=0 的两个已有 case，再补一个 `decode_prefix_len=1024` 的 dense case。prefix>0 的新用例应作为“误差归因实验”，因为 AIC 当前 `ctx_tokens / isl` 的调度数学没有扣除 cached prefix，而真实 SGLang chunked prefill 主要按 fresh extend tokens 计入 chunk budget。
