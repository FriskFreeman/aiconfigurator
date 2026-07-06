# Prefill Attention 实机与 AIC MLA 差异根因分析

## 结论摘要

本次差异的根因不是“实机 DeepSeek prefill 没有走 FA3”，也不是“普通 attention 和 MLA 在数学上必然应接近但数据异常”。更准确地说：

- 实机 DeepSeek V3 prefill 在当前配置下进入的是 SGLang 的 `attn_mha` / MHA forward 分支，底层核心 attention kernel 是 FA3 的 MHA 形态。
- AIC 的 `context_attention_perf.txt` 由 `collect_attn.py` 采集，构造的是标准 MHA/GQA `q,k,v` attention，虽然不是 DeepSeek 模块语义，但在底层 CUDA kernel 口径上更接近实机 `attn_mha`。
- AIC 的 `context_mla_perf.txt` 由 `collect_mla.py` 采集，名字叫 MLA，但在 H100 / FA3 路径下测的是 absorbed / latent MLA 形态，核心输入维度与实机 prefill `attn_mha` 不一致，特别是 value / latent 维度被放大到 `kv_lora_rank=512`，导致时延显著更高。
- 因此，prefill 单算子对比时，实机 attention kernel 更接近 `context_attention_perf` 是合理现象；把它和 `context_mla_perf` 对齐会系统性高估。

一句话：这里的 `MLA` 是模型语义名，但 prefill 实机 attention core 实际跑的是 MHA kernel 形态；AIC `context_mla_perf` 记录的是另一种 latent/absorbed MLA kernel 口径，两者不是同一个单算子边界。

## 1. 现象复盘

已有对比文件：

- `bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/analysis/mla_kernel_aic_compare__comparison.csv`
- `bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/analysis/mla_kernel_aic_compare__summary.md`

摘要统计显示，prefill 共 90 行：

- `MLA / bfloat16` 对实机 attention kernel 的 MAPE：`288.20%`
- `MLA / fp8` 对实机 attention kernel 的 MAPE：`276.39%`
- `Attention / bfloat16` 对实机 attention kernel 的 MAPE：`26.12%`
- `Attention / fp8` 对实机 attention kernel 的 MAPE：`12.19%`

几个典型 case 的均值：

| case | 实机 attention kernel | AIC context_mla bf16 | AIC context_mla fp8 | AIC context_attention bf16 | AIC context_attention fp8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `b12,f300,p0` | `0.2569 ms` | `0.6637 ms` | `0.6799 ms` | `0.3753 ms` | `0.2995 ms` |
| `b1,f4500,p0` | `1.2237 ms` | `5.0933 ms` | `4.9709 ms` | `1.4783 ms` | `1.3518 ms` |
| `b4,f2048,p0` | `1.1225 ms` | `4.3065 ms` | `4.2493 ms` | `1.4466 ms` | `1.1846 ms` |
| `b1,f8192,p0` | `3.6111 ms` | `14.8179 ms` | `14.4203 ms` | `4.4103 ms` | `4.0322 ms` |

整体比例上，prefill 中 AIC MLA 平均约为实机的 `3.7x ~ 3.8x`，而 AIC attention fp8 平均约为实机的 `1.10x`。

## 2. SGLang 实机 prefill 的真实路径

SGLang 0.5.9 容器源码位置：

- `/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py`
- `/sgl-workspace/sglang/python/sglang/srt/models/deepseek_common/attention_backend_handler.py`
- `/sgl-workspace/sglang/python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha.py`
- `/sgl-workspace/sglang/python/sglang/srt/layers/attention/flashattention_backend.py`

`DeepseekV2AttentionMLA` 同时构造了两个 attention 子模块：

- `attn_mqa`: `head_dim = kv_lora_rank + qk_rope_head_dim`，`num_kv_heads=1`，`v_head_dim=kv_lora_rank`。
- `attn_mha`: `head_dim = qk_nope_head_dim + qk_rope_head_dim`，`num_kv_heads=num_local_heads`，`v_head_dim=v_head_dim`。

对 DeepSeek V3：

- `qk_nope_head_dim = 128`
- `qk_rope_head_dim = 64`
- `qk_head_dim = 192`
- `v_head_dim = 128`
- `kv_lora_rank = 512`

### 2.1 prefill 分支选择

`attention_backend_handler.py` 的核心逻辑：

- 对 `fa3`，如果不是 deterministic inference，则走 `_handle_attention_backend(attn, forward_batch, "fa3")`。
- 当 `forward_batch.forward_mode.is_extend_without_speculative()` 且满足 prefix-cache 条件时，返回 `MHA_ONE_SHOT` 或 `MHA_CHUNKED_KV`。
- 对无 prefix 的 prefill，`sum_extend_prefix_lens == 0`，因此会进 MHA 路径。

这与实机 trace 中看到的 `attention_module=attn_mha` 对得上。

### 2.2 prefill MHA 构造的 q/k/v

`forward_mha.py::forward_normal_prepare()` 中：

- `q = q_b_proj(...).view(-1, num_local_heads, qk_head_dim)`，即 `q` 最后一维为 `192`。
- `kv_b_proj(kv_a)` 生成 `qk_nope_head_dim + v_head_dim = 128 + 128 = 256`。
- `k_nope = kv[..., :128]`。
- `v = kv[..., 128:]`，所以 `v` 最后一维为 `128`。
- `_concat_and_cast_mha_k(k_nope, k_pe, ...)` 把 `k_nope(128)` 和 `k_pe(64)` 拼成 `k`，所以 `k` 最后一维为 `192`。

`forward_normal_core()` 调用：

```python
attn_output = self.attn_mha(q, k, v, forward_batch, save_kv_cache=False)
```

因此实机 prefill attention core 是：

- `q`: `[tokens, local_heads, 192]`
- `k`: `[tokens, local_heads, 192]`
- `v`: `[tokens, local_heads, 128]`
- `num_kv_heads = local_heads`，即 MHA，不是 MQA。

在 `flashattention_backend.py::forward_extend()` 中，当 `self.use_mla` 且走 MHA chunk/prefix 分支时，会调用：

```python
flash_attn_varlen_func(
    q=q.view(-1, layer.tp_q_head_num, layer.head_dim),
    k=k.view(-1, layer.tp_k_head_num, layer.head_dim).to(q.dtype),
    v=v.view(-1, layer.tp_k_head_num, layer.v_head_dim).to(q.dtype),
    ...
)
```

这里的 `layer` 是 `attn_mha`，所以 `layer.head_dim=192`，`layer.v_head_dim=128`。

### 2.3 prefix 场景

有 prefix 时，`forward_normal_chunked_kv_core()` 先对 extend 部分做 MHA，再按 prefix chunk 取 latent cache，投影为 MHA 的 `k/v`，继续调用 `attn_mha`，最后用 `merge_state_v2` 合并。也就是说，prefix 场景依然是 MHA attention core，只是可能多次 FA3 MHA kernel 加合并，而不是直接变成 `context_mla_perf` 那种 absorbed latent MLA 单核。

## 3. `collect_attn.py` 采集的是什么

`collector/sglang/collect_attn.py` 构造普通 attention：

- `MockModelConfig.attention_arch = AttentionArch.MHA`
- KV pool 是 `MHATokenToKVPool`
- `RadixAttention(num_heads=n, head_dim=head_dim, num_kv_heads=num_key_value_heads)`
- context 阶段：
  - `q`: `[b*s, n, head_dim]`
  - `k`: `[b*s, n_kv, head_dim]`
  - `v`: `[b*s, n_kv, head_dim]`
- H100 上 backend 是 `FlashAttentionBackend`，`kernel_source=flash_attention`。

这不是 DeepSeek MLA module，但它确实测的是标准 FA3 MHA/GQA attention kernel。

和实机 `attn_mha` 的差异：

- collector 的 `head_dim` 只有 `128` 或 `256`，没有直接采 `qk=192, v=128` 的异构 head dim。
- collector 的 `v` 维度等于 `head_dim`，而实机 `q/k=192, v=128`。
- 对比脚本中 `n=128` 的 attention 查询来自 AIC 初始化后的数据补点，因为原始 `context_attention_perf.txt` 最大 `num_heads` 到 `96`，不是原始精确采样点。

即便有这些不完美，它仍比 `context_mla_perf` 更接近实机，因为二者同属 MHA/GQA FA3 kernel 形态。

## 4. `collect_mla.py` 采集的是什么

`collector/sglang/collect_mla.py` 构造的是 MLA kernel 路径：

- `MockModelConfig.attention_arch = AttentionArch.MLA`
- KV pool 是 `MLATokenToKVPool`
- `MockModelRunner.use_mla_backend = True`
- H100 默认 backend 选择为 `fa3`，但 `kernel_source` 记录为 `flash_attention`。

关键差异在 `run_mla()` 的维度构造：

```python
kv_lora_rank = 512
qk_rope_head_dim = 64
qk_nope_head_dim = 128

if selected_backend == "trtllm_mla":
    if is_context_phase:
        v_head_dim = qk_nope_head_dim      # 128
        head_dim_total = qk_nope_head_dim + qk_rope_head_dim  # 192
else:
    v_head_dim = kv_lora_rank              # 512
    head_dim_total = kv_lora_rank + qk_rope_head_dim  # 576
```

在 H100 / FA3 路径下，`selected_backend != trtllm_mla`，所以 context MLA 表构造的是：

- `v_head_dim = 512`
- `head_dim_total = 576`
- `q_nope`: `[tokens, local_heads, 512]`
- `q_rope`: `[tokens, local_heads, 64]`
- `k_nope`: `[tokens, 1, 512]`
- `k_rope`: `[tokens, 1, 64]`
- `v = k_nope`，即 value 维度 `512`

随后 `RadixAttention(num_heads=local_heads, head_dim=576, num_kv_heads=1, v_head_dim=512)`。

这对应 absorbed / latent MLA kernel 口径，不是实机 prefill `attn_mha(q/k=192, v=128, n_kv=n)` 的口径。

### 4.1 `collect_mla.py` 绕过了 DeepSeek module 的 runtime dispatch

这里有一个容易被 `kernel_source=flash_attention` 掩盖的细节：`collect_mla.py` 并不是跑完整的 `DeepseekV2AttentionMLA.forward()`，而是手工构造 `RadixAttention(attention_arch=MLA)`、`MLATokenToKVPool` 和 `ForwardBatch`，然后直接调用：

```python
layer(q, k, v, forward_batch, q_rope=q_rope_arg, k_rope=k_rope_arg)
```

因此它不会进入 DeepSeek 模型层的：

- `dispatch_attn_forward_method()`
- `forward_normal_prepare()`
- `forward_normal_core()`
- `forward_normal_chunked_kv_*()`

而这些正是真实 SGLang prefill 决定 “MHA 还是 MLA” 的关键入口。

在 SGLang 0.5.9 Docker 源码中，完整 DeepSeek module 先构造两套子 attention：

- `attn_mqa`: `head_dim=512+64=576`，`num_kv_heads=1`，`v_head_dim=512`。
- `attn_mha`: `head_dim=128+64=192`，`num_kv_heads=num_local_heads`，`v_head_dim=128`。

真实 prefill 常见路径由 `attention_backend_handler.py` 返回 `MHA` / `MHA_ONE_SHOT` / `MHA_CHUNKED_KV`，然后在 `forward_mha.py` 里执行 `kv_b_proj(kv_a)`，把 latent KV 展开成每个 head 的 `k_nope[128]` 和 `v[128]`，再拼接 `k_pe[64]` 得到 `k[192]`，最终调用 `attn_mha(q[192], k[192], v[128])`。

`collect_mla.py` 的 context 路径则直接构造 `q_nope[512] + q_rope[64]`、`k_nope[512] + k_rope[64]`、`v=k_nope[512]`，并且 `num_kv_heads=1`。这是一条 absorbed / latent MLA 单算子路径，不是完整 DeepSeek module 在当前实机 prefill 下常见的 MHA dispatch 结果。

### 4.2 同为 FA3/FlashAttention，不等于同一个算子问题

`collect_attn.py` 和 `collect_mla.py` 在 H100 上都可能写出 `kernel_source=flash_attention`，但这只说明它们最终使用了 FlashAttention backend 家族，不说明输入语义相同。

`collect_attn.py` 的关键特征是：

- `AttentionArch.MHA`
- `FlashAttentionBackend.use_mla=False`
- `MHATokenToKVPool`
- 标准 K/V cache
- `q/k/v` 共享同一个 `head_dim`

`collect_mla.py` 的关键特征是：

- `AttentionArch.MLA`
- `FlashAttentionBackend.use_mla=True`
- `MLATokenToKVPool`
- latent KV cache
- `kv_lora_rank + qk_rope_head_dim` 语义

在 SGLang 0.5.9 的 `FlashAttentionBackend.forward_extend()` 中，`use_mla` 会直接改变执行分支。`use_mla=False` 时走普通 MHA/GQA cache 路径；`use_mla=True` 且带有 DeepSeek MHA/chunked-prefix 状态时才走 MLA 模型里的 MHA 辅助路径；`use_mla=True` 且没有这些状态时，会进入 absorbed MLA latent 路径，调用带 `qv=q_nope` 的 `flash_attn_with_kvcache` 形态。

`collect_mla.py` 直接构造 standalone `RadixAttention`，不会设置真实 `forward_normal_*` 中的 `attn_attend_prefix_cache`、`mha_one_shot`、`mha_return_lse` 等状态，所以它和当前实机 `attn_mha` 的核心差异不是 “FA3 vs 非 FA3”，而是 “MHA 展开后的 Q/K/V vs latent MLA Q/KV”。

## 5. 为什么 MLA 表比 attention 表慢很多

从计算和内存形态看，AIC `context_mla_perf` H100 FA3 路径相对实机 prefill `attn_mha` 有几个系统性放大点：

- `v_head_dim`: 实机 `128`，collector MLA FA3 路径 `512`，输出和值相关读写维度约 `4x`。
- `q/k` 参与维度：实机 `192`，collector MLA FA3 路径有效 latent+rope `576`，虽然 FA3 MLA 会用 `q_rope/qv` 的特殊接口，但数据通路仍显著更重。
- KV head 形态：实机 prefill `attn_mha` 是 `num_kv_heads = num_heads`；collector MLA 是 latent MQA 形态 `num_kv_heads = 1`，语义和 kernel 参数不同。
- prefix 行为：实机 prefix prefill 是 MHA extend + prefix chunks + merge；`query_context_mla()` 只是按 full_s 查 `context_mla_perf` 再做 prefix correction，不能反映 MHA chunked-prefix 的真实 kernel 组合。

从数据上看，典型 prefill case 中 `context_mla_perf` 约为实机的 `3.7x ~ 3.8x`，这与 `v_head_dim=512` 对比 `128` 的量级非常一致。

如果直接比较 AIC 两张表在同一批实机 case 上的查询结果，差异同样非常明显：

- `context_mla_bfloat16 / context_attention_bfloat16` 的均值约 `3.07x`，中位数约 `3.41x`。
- `context_mla_fp8 / context_attention_fp8` 的均值约 `3.47x`，中位数约 `3.67x`。
- 在长 prefill 或带大 prefix 的 case 上，`context_mla` 通常是 `context_attention` 的 `3.4x ~ 4.4x`。

这说明当前现象不是某个实机 run 的偶然偏差，而是 AIC `context_mla_perf` 与 `context_attention_perf` 两套 collector 数据本身就对应不同的单算子口径。

## 6. 为什么实机更接近普通 attention

实机 trace 的 `attention_module=attn_mha` 已经说明，当前被拿来比较的“attn 单算子”其实是 `attn_mha` 子模块里的底层 FA3 kernel，而不是 `attn_mqa` / absorbed MLA kernel。

`collect_attn.py` 虽然没有 DeepSeek 特殊 `q=192,v=128` 维度，但它至少具备：

- MHA/GQA attention 语义。
- FA3 / FlashAttention backend。
- context prefill varlen attention。
- `q,k,v` 显式输入，而不是 absorbed latent `q_rope/qv + latent cache` 形态。

因此它和实机更接近。前面数据也显示，`context_attention fp8` 平均约 `1.10x` 实机，`context_attention bf16` 平均约 `1.26x` 实机，而 `context_mla` 是 `3.7x+`。

## 7. AIC SDK 当前为何会查 MLA

`src/aiconfigurator/sdk/models/deepseek.py` 中，DeepSeek V3 的 SGLang context fallback 路径使用：

```python
ops.ContextMLA("context_attention", ..., 128 // tp_size, ...)
```

`ContextMLA.query()` 调 `PerfDatabase.query_context_mla()`，后者查：

```text
context_mla_perf.txt: FMHAQuantMode -> KVCacheQuantMode -> num_heads -> full_s -> b
```

这是模型级语义上的选择：DeepSeek attention 模块是 MLA 模块，且 SDK 历史上把 attention core 抽象为 `ContextMLA`。但对 SGLang 0.5.9 的 prefill 实际执行而言，底层 core 经常是 `attn_mha`，这使得模型抽象和 CUDA kernel 对比口径发生错位。

注意：这不一定说明端到端 SDK 模型全部错误，因为完整 attention 模块还包含 `downscale/q_b/kv_b/o_proj` 等 GEMM，`ContextMLA` 在整体 fallback 栈里只是 attention core 一项。但如果单独拿 attention core 和实机 nsys kernel 对比，那么 `ContextMLA` 的表不适合作为 prefill `attn_mha` 的基准。

## 8. 量化因素是否是主因

本次差异不是主要由 fp8/bf16 量化造成。

证据：

- `context_mla_perf.txt` 里 `mla_dtype` 只有 `bfloat16`，`kv_cache_dtype` 有 `bfloat16/fp8`。
- `context_attention_perf.txt` 里 context 有 `attn_dtype=bfloat16/fp8`、`kv_cache_dtype=bfloat16/fp8`。
- 对比结果中，`MLA bf16` 和 `MLA fp8` 都远高于实机，二者差异远小于 `MLA` 与 `Attention` 之间的差异。

所以主因是输入形状和 kernel 口径，而不是 KV cache dtype。

更细地看：

- `collect_mla.py` 的 `mla_dtype` 固定写成 `bfloat16`，`kv_cache_dtype` 在 H100 上有 `bfloat16/fp8` 两类。它没有把 context MLA 的 FMHA compute dtype 扩展成独立的 `fp8` 轴。
- `collect_attn.py` 的 context attention 同时记录 `attn_dtype` 和 `kv_cache_dtype`，H100 数据里有 `attn_dtype=bfloat16/fp8` 与 `kv_cache_dtype=bfloat16/fp8`。
- `collect_attention` 的 FP8 context FMHA 会显式把 live `q/k/v` cast 到 fp8；但 `collect_mla` 的 context 路径主要通过 `kv_cache_dtype` 控制 MLATokenToKVPool 存储，并不等价于 “同一 MHA 形状下切换 bf16/fp8”。

因此当前对比里同时看 `fp8/bfloat16` 是必要的，但这不是主导误差。真正主导误差的是：实机 prefill trace 是 `attn_mha` 的 MHA 展开形态，而 AIC `context_mla_perf` 是 latent MLA 形态。

## 8.1 `context_attention` 也只是近似，不是完美对齐

虽然 `context_attention_perf` 明显更接近实机 prefill，但它仍不是完全同形状的 DeepSeek prefill MHA collector。

当前 `collect_attn.py` / `query_context_attention()` 的 schema 只有一个 `head_dim` / `head_size`，可以表达标准 MHA/GQA 的头数关系，但不能表达 DeepSeek prefill 的异构维度：

- 实机 `q/k head_dim = 192`
- 实机 `v_head_dim = 128`

H100 0.5.9 的 `context_attention_perf.txt` 当前只有 `head_dim=128` 的数据；SDK 查询也用单一 `head_size` 同时解释 Q/K/V。因此 `context_attention` 对上的主要是 “MHA/GQA FA3 语义与量级”，不是完全一致的 DeepSeek `q/k=192, v=128` 内核形状。

这解释了为什么 `context_attention` 的误差已经远小于 `context_mla`，但仍有约 `12% ~ 26%` 的平均误差：剩余误差很大一部分来自 qk/v 异维没有建模、原始表没有 `num_heads=128` 的精确点、以及 prefix/chunked-prefix 的实际 kernel 组合被查表简化。

## 9. 对后续对比的建议

1. Prefill 单算子 attention core 对比，应优先使用 `context_attention_perf` 或新增一个 DeepSeek-prefill-MHA 专用 collector。

2. 如果要完全对齐实机 `attn_mha`，collector 应新增如下 shape：

- `q_head_dim = k_head_dim = 192`
- `v_head_dim = 128`
- `num_kv_heads = num_heads`
- FA3 context varlen。

当前 `collect_attn.py` 只有 `head_dim=128/256` 且 `q/k/v` 同维，仍不是完美对齐；它只是比 `collect_mla.py` 更接近。

3. `context_mla_perf` 更适合对齐真实走 `AttnForwardMethod.MLA` / absorbed latent MLA 的路径，而不是默认 SGLang prefill `attn_mha`。

4. SDK 若要在 SGLang DeepSeek V3 prefill 上更贴近实机 attention core，可以考虑按 SGLang 的 `dispatch_attn_forward_method()` 语义，在 prefill MHA 条件下使用 `ContextAttention` 或新增 `DeepSeekContextMHA` op，而不是固定 `ContextMLA`。

5. 对 prefix 场景，不宜只用 `query_context_mla()` 的 full_s/prefix correction。SGLang 实际可能是 extend MHA + prefix chunks + `merge_state_v2`，应按实际 chunked-prefix 分解或用 trace 中的 kernel 聚合口径对齐。

## 10. 最终判断

本问题真实存在，且根因已经可以定位：

- 实机 prefill attention core：DeepSeek MLA 模块里的 MHA forward，FA3 MHA kernel，`q/k=192, v=128, n_kv=n`。
- AIC `context_attention_perf`：普通 FA3 attention，MHA/GQA 形态，虽然维度不完全相同，但 kernel 类型接近。
- AIC `context_mla_perf`：FA3/FlashAttention 的 latent MLA 形态，H100 下 `v_head_dim=512, head_dim_total=576, n_kv=1`，与实机 prefill MHA core 不同，因此显著偏慢。

所以“prefill 实机数据和普通 attention 更接近、和 MLA 差异显著”不是反常，而是暴露了 AIC 单算子数据表命名/模型语义和 SGLang prefill 实际 kernel 分支之间的口径错位。
