# AGG Mixed MLA 非 Attn 算子差异诊断

本文聚焦 `qkv_a_proj / q_b_proj / kv_b_proj / o_proj / mla_concat_k`，暂不分析 attention kernel 本身。

## 结论

- `qkv_a_proj`、`q_b_proj`、`o_proj` 的主要误差来源不是算子类型不同，也不是实机性能随机波动，而是 AIC `run_agg` 诊断拆解使用的 `query_s` 与 SGLang 实机 mixed batch 的实际 token 数不一致。
- 对这三类 projection，实机单层时延与 `MLA时延拆解.csv` 中的 `attention_token_count` 几乎完全线性相关；与 AIC 当前 `diagnostic_forced_prefix_ops` 使用的 `query_s` 相关性很弱。
- `kv_b_proj` 是另一类问题：SGLang MHA path 在存在 prefix/one-shot/chunked prefix cache 时，`kv_b_proj` 可能作用在从 KV cache 拉出的 `kv_a` token 集合上，而不是普通 hidden-state token 集合上。AIC 当前把它按普通 prefix-path GEMM 用同一个 first-pass `s` 查询，边界不匹配。
- 此外，AIC GEMM 表对 `kv_b_proj` 关键形状 `(m, n=32768, k=512)` 没有 exact 采样点，`query_gemm` 只能走插值；该插值对 `kv_b_proj` 这种大 N、小 K、DeepGEMM shape 明显低估。
- `mla_concat_k` 在 AIC 中是独立 `MLAConcatK` op；实机 trace 中没有独立 canonical child 或 kernel 名称可直接对应，当前不适合做一对一柱状图比较。

## 实机侧实际执行

SGLang 0.5.9 DeepSeek-V3 attention 源码位置：

- `/home/ai_lab/fjw/miniforge3/envs/ljc01/lib/python3.12/site-packages/sglang/srt/models/deepseek_v2.py`
- `/home/ai_lab/fjw/miniforge3/envs/ljc01/lib/python3.12/site-packages/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`

模型定义中：

- `fused_qkv_a_proj_with_mqa = ReplicatedLinear(hidden_size, q_lora_rank + kv_lora_rank + qk_rope_head_dim)`，即 DeepSeek-V3 为 `7168 -> 2112`。
- `q_b_proj = ColumnParallelLinear(q_lora_rank, num_heads * qk_head_dim)`，TP=1 时为 `1536 -> 24576`。
- `kv_b_proj = ColumnParallelLinear(kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim))`，TP=1 时为 `512 -> 32768`。
- `o_proj = RowParallelLinear(num_heads * v_head_dim, hidden_size)`，TP=1 时为 `16384 -> 7168`。

实机 trace 中 projection child 的 kernel 主要是：

- `per_token_group_quant_8bit_kernel`
- `sm90_fp8_gemm_1d2d_impl`

因此 projection 侧实机确实是 FP8 activation quant + DeepGEMM 路径；AIC `GEMM(fp8_block)` collector 只覆盖预量化输入上的 DeepGEMM microbench，本身不覆盖 runtime activation quant kernel。

## AIC 侧查询行为

SDK 普通 DeepSeek context 路径中：

- `context_downscale_gemm` 位于 `context_mla_block` 之前，是 shared op。
- `context_mla_block` 是 `PrefixConditionalOp`。
- `prefix == 0` 时选择 `_no_prefix_ops = [context_mla_module]`。
- `prefix > 0` 时选择 `_prefix_ops = [context_q_b_proj_gemm, context_kv_b_proj_gemm, context_mla_concat_k, context_attention, context_proj_gemm]`。

`sglang_backend.py::run_agg` 的 mixed step 对 context 部分做了两段 static query：

- first pass: 用 `batch_size=1, isl=ctx_tokens + gen_tokens, prefix=prefix * floor(ctx_tokens / isl)` 查询非 attention latency。
- second pass: 用 `batch_size=ceil(ctx_tokens / isl), isl=isl, prefix=prefix` 查询 `context_attention`，再按 `ceil(isl / ctx_tokens)` 做 scale。

本报告的细粒度 AIC projection 柱来自 diagnostic 拆解：按 first pass 的 `query_s` 强制查询 prefix-path 子 op。它用于定位误差来源，但不等同于原始 `run_agg` 快照直接导出的 per-op 字段。

## qkv_a / q_b / o_proj：主要是输入 token 形状不匹配

对 `qkv_a_proj`、`q_b_proj`、`o_proj`，实机时延与实际 token 数高度一致：

| operator | corr(real, actual_attention_token_count) | corr(real, AIC query_s) |
|---|---:|---:|
| qkv_a_proj | 0.999866 | 0.332747 |
| q_b_proj | 0.999877 | 0.333373 |
| o_proj | 0.998835 | 0.341816 |
| kv_b_proj | 0.338494 | 0.530894 |

典型例子：

| case | stage | real actual tokens | AIC query_s | q_b real ms | q_b AIC ms | ratio |
|---|---|---:|---:|---:|---:|---:|
| C02 | decode | 2056 | 3585 | 0.133 | 0.229 | 0.579 |
| C03 | decode | 4104 | 3585 | 0.238 | 0.229 | 1.038 |
| C05 | prefill | 2064 | 8193 | 0.132 | 0.488 | 0.270 |
| C07 | prefill | 4128 | 8193 | 0.238 | 0.488 | 0.487 |
| C09 | prefill | 8192 | 8194 | 0.447 | 0.488 | 0.914 |
| C11 | prefill | 2064 | 8961 | 0.132 | 0.533 | 0.248 |

这些例子说明：AIC 当前 first-pass `query_s` 是 `run_agg` 抽象出来的 `ctx_tokens + gen_tokens`，但 SGLang 实机 mixed step 中 projection 实际处理的 token 数由真实 scheduler batch、prefill chunk、decode 混入方式共同决定。对 projection GEMM 来说，实际 M 更接近 trace 中捕获到的 `attention_token_count` 档位，而不是 AIC 的 `query_s`。

因此，对这些 projection 的后续对齐策略应优先修正 AIC diagnostic query 的 `m`，使用实机 first formal batch 捕获到的 token 数，或在构造实机 case 时让实际 batch token 与 AIC `run_agg` 的 first-pass token 假设完全一致。

## kv_b_proj：边界和 perf 表都有问题

`kv_b_proj` 不能简单按当前 hidden-state token 数解释。

SGLang MHA path 中，在 `forward_normal_prepare`：

- 先构造 `q` 和 `latent_cache`。
- `kv_a = kv_a_layernorm(kv_a)`。
- 若 `forward_batch.mha_one_shot and sum(forward_batch.extend_prefix_lens_cpu) != 0`，会通过 `fetch_mha_one_shot_kv_indices()` 从 cache 拉取 prefix/fresh KV 对应的 `kv_a, k_pe`。
- 随后执行 `kv = self.kv_b_proj(kv_a)[0]`。

这意味着存在 prefix 的 mixed/prefill 场景下，`kv_b_proj` 的输入 M 可能是被 materialize 的 KV token 集合，而不是普通当前 hidden-state token 集合。这解释了为什么 `kv_b_proj` 与 `q_b/o_proj` 的 token 相关性完全不同。

同时，AIC 查询 `context_kv_b_proj_gemm` 时只按 prefix-path 普通 GEMM：

```text
m = query_s
n = 32768
k = 512
```

这与实机 one-shot/prefix-cache 下的 `kv_b_proj(kv_a_from_cache)` 边界不一致。

另一个独立问题是 perf database 覆盖不足。`src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.9/gemm_perf.txt` 中没有 exact row：

```text
m = 8192
n = 32768
k = 512
```

`PerfDatabase.query_gemm()` 在 exact miss 后先尝试同 `(n,k)` 下沿 `m` 插值；但 `(n=32768,k=512)` 完全无 exact sampled m，所以最终走 3D 插值。该插值会使用诸如 `(m=8192,n=16384,k=512)`、`(m=8192,n=51200,k=512)`、`(m=8192,n=65536,k=512)` 等邻近点。

实际查询结果：

| query shape | AIC GEMM latency ms |
|---|---:|
| `(8192, 32768, 512)` | 0.310 |
| `(16384, 32768, 512)` | 0.597 |
| `(32768, 32768, 512)` | 1.242 |
| `(65536, 32768, 512)` | 2.533 |

而实机 C06 的 `kv_b_proj` 单层 kernel：

```text
sm90_fp8_gemm_1d2d_impl: ~2.73-3.05 ms
per_token_group_quant_8bit_kernel: ~0.047 ms
```

因此 `kv_b_proj` 的差异不能只归因于 activation quant；主要是实机执行边界/输入 M 与 AIC query 不一致，再叠加 AIC GEMM 表对关键形状缺采样和插值低估。

## mla_concat_k：当前无实机同边界

AIC 的 `context_mla_concat_k` 来自 `MLAConcatK.query()`，按：

```text
num_tokens = batch_size * (s + prefix)
num_heads = 128 / tp
```

查询 `mla_concat_k_perf.txt`。

但当前实机 trace 的 `real_mla_kernel_long.csv` 中没有 `concat_and_cast_mha_k_triton` 或独立 `mla_concat_k` child。可见 kernel 名称只有：

```text
BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel
RMSNormKernel
device_kernel
per_token_group_quant_8bit_kernel
prepare_varlen_num_blocks_kernel
set_mla_kv_buffer_kernel
sm90_fp8_gemm_1d2d_impl
```

所以 `mla_concat_k` 当前只是 AIC 侧独立建模项，实机侧没有同名边界可比。若后续要比较 concat_k，需要增强 Nsight/NVTX 标记或改用 kernel 名称/源码路径定位其真实归属。

## 后续建议

1. 对 `qkv_a_proj/q_b_proj/o_proj`，把 AIC diagnostic 查询的 `m` 从 run_agg first-pass `query_s` 改为实机捕获的实际 hidden token 数，再重画非-attn 对比。
2. 对 `kv_b_proj`，不要复用普通 first-pass `query_s`；需要按 SGLang MHA path 判断是否 `mha_one_shot` 或 chunked prefix cache，并估算/捕获 `kv_b_proj(kv_a)` 实际 M。
3. 对 `kv_b_proj` 的 `(m,n=32768,k=512)` 系列补采 GEMM perf 点，至少覆盖 `m=2048/4096/8192/16384/32768/65536`，否则插值误差会持续污染 AIC。
4. 对 `mla_concat_k`，先解决实机边界识别问题，再做数值比较；当前直接比较会把 AIC-only op 当成误差来源，口径不成立。
