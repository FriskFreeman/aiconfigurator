# DeepSeek-V4 Collector 与仿真算子边界分析

本文基于 `aiconfigurator-dsv4-sjy` 代码树，梳理 DeepSeek-V4 在 SDK 仿真模型中的算子建模方式，以及 SGLang collector 对应采集组件、采集边界和性能数据消费路径。

## 结论摘要

DeepSeek-V4 的性能采集不是“逐个底层 CUDA 算子”采集，而是混合了三类粒度：

- Attention 主路径按 **完整 self-attn module forward** 采集，覆盖投影、rope/norm、cache 写入、compressor、CSA indexer/topk 和最终 FlashMLA。
- Sparse correction 按 **两个关键稀疏 kernel** 单独采集，用于修正 prefix/past-kv 对 attention module 查询的影响。
- mHC 按 **pre/post 子模块** 采集，每次构建一层 SGLang runner，但只计时 `hc_pre` 或 `hc_post` 两类调用。

SDK 模型层的 DeepSeek-V4 只显式支持 `compress_ratio in {0, 4, 128}`。其中 `0` 代表 SWA，但在 attention module latency 建模中被折叠到 HCA，即复用 `compress_ratio=128` 的 HCA perf 数据；KV cache 容量估算仍保留真实的逐层 `compress_ratios`。

## SDK 建模入口

DeepSeek-V4 模型入口位于：

- `src/aiconfigurator/sdk/models/deepseek_v4.py`
- `src/aiconfigurator/sdk/operations.py`
- `src/aiconfigurator/sdk/utils.py`
- `src/aiconfigurator/sdk/perf_database.py`

`utils.py` 在解析 HuggingFace config 时识别 `architecture == "DeepseekV4ForCausalLM"`，并将 V4 专用结构字段打包为 `DeepSeekV4Config`。关键字段包括：

- `q_lora_rank` / `o_lora_rank` / `o_groups`
- `head_dim` / `qk_rope_head_dim`
- `index_head_dim` / `index_n_heads` / `index_topk`
- `sliding_window`
- `compress_ratios`
- `hc_mult` / `hc_sinkhorn_iters`
- `n_shared_experts`

`DeepSeekV4Model` 将模型拆成 context 和 generation 两组 op。V4 专用 op 主要有：

- `ContextDeepSeekV4AttentionModule`
- `GenerationDeepSeekV4AttentionModule`
- `DeepSeekV4MHCModule`

除这些 V4 专用 op 外，DeepSeek-V4 仍复用通用的 embedding、elementwise、GEMM、MoE dispatch、MoE、logits GEMM 等仿真 op。

### Attention Ratio 建模

`DeepSeekV4Model` 内部统计每层 `compress_ratio` 的数量，并为每种 ratio 生成一个 attention module op：

- `compress_ratio=4`：CSA
- `compress_ratio=128`：HCA
- `compress_ratio=0`：SWA

需要注意的是，模型层会把 `compress_ratio=0` 的 SWA 层折叠进 HCA：

```python
ratio_counts[128] += ratio_counts.pop(0, 0)
```

这意味着仿真不会要求单独的 SWA collector 或 SWA perf 文件。SWA 层的 latency 使用 HCA module 数据近似，但 KV cache 容量计算仍按原始 ratio 逐层计算。

## Collector 注册总览

SGLang collector 注册表在 `collector/sglang/registry.py`。DeepSeek-V4 相关条目可以分为四组。

### 1. V4-Flash Attention Module

对应 `collector.sglang.collect_dsv4_flash_attn`：

| op | perf file | 采集模式 | ratio |
| --- | --- | --- | --- |
| `dsv4_flash_csa_context_module` | `dsv4_flash_csa_context_module_perf.txt` | context module | 4 |
| `dsv4_flash_hca_context_module` | `dsv4_flash_hca_context_module_perf.txt` | context module | 128 |
| `dsv4_flash_csa_generation_module` | `dsv4_flash_csa_generation_module_perf.txt` | generation module | 4 |
| `dsv4_flash_hca_generation_module` | `dsv4_flash_hca_generation_module_perf.txt` | generation module | 128 |

### 2. V4-Pro Attention Module

仍使用 `collector.sglang.collect_dsv4_flash_attn`，但模型维度和输出文件换成 Pro：

| op | perf file | 采集模式 | ratio |
| --- | --- | --- | --- |
| `dsv4_pro_csa_context_module` | `dsv4_pro_csa_context_module_perf.txt` | context module | 4 |
| `dsv4_pro_hca_context_module` | `dsv4_pro_hca_context_module_perf.txt` | context module | 128 |
| `dsv4_pro_csa_generation_module` | `dsv4_pro_csa_generation_module_perf.txt` | generation module | 4 |
| `dsv4_pro_hca_generation_module` | `dsv4_pro_hca_generation_module_perf.txt` | generation module | 128 |

V4-Pro 和 V4-Flash 共享同一套 SGLang DSV4 operator collector，差异主要来自模型 config：例如 Pro 使用更多 attention heads 和不同 `index_topk`。

### 3. Sparse Kernel Correction

对应 `collector.sglang.deepseekv4_sparse_modules`：

| op | perf file | 实际采集对象 | 用途 |
| --- | --- | --- | --- |
| `dsv4_flash_paged_mqa_logits_module` | `dsv4_flash_paged_mqa_logits_module_perf.txt` | `deep_gemm.fp8_paged_mqa_logits` | CSA indexer scoring 的 past-kv 修正 |
| `dsv4_flash_hca_attn_module` | `dsv4_flash_hca_attn_module_perf.txt` | `flash_mla.flash_mla_with_kvcache` | HCA c128 sparse FlashMLA 的 past-kv 修正 |
| `dsv4_pro_paged_mqa_logits_module` | `dsv4_pro_paged_mqa_logits_module_perf.txt` | `deep_gemm.fp8_paged_mqa_logits` | Pro 维度下的 CSA 修正 |
| `dsv4_pro_hca_attn_module` | `dsv4_pro_hca_attn_module_perf.txt` | `flash_mla.flash_mla_with_kvcache` | Pro 维度下的 HCA 修正 |

这些条目名称里带 `module`，但实际边界是 kernel-level，不是完整 module。`topk_512` 和 `csa_attn` 没有单独采集 CSV，而是在 `perf_database.py` 中解析或公式建模。

### 4. mHC Module

对应 `collector.sglang.collect_mhc_module`：

| op | perf file | 实际采集对象 |
| --- | --- | --- |
| `mhc_module` | `mhc_module_perf.txt` | DeepSeek-V4 decoder layer 的 `hc_pre` / `hc_post` |

`mhc_module` 是所有 V4 Flash/Pro 共用的 mHC 采集入口，通过 `COLLECTOR_MODEL_PATH` 或 `--model-path` 选择具体模型。

## Attention Module Collector 边界

文件：`collector/sglang/collect_dsv4_flash_attn.py`

这个 collector 的核心不是直接调某个单一 kernel，而是：

1. 构造一个只保留目标 attention kind 的临时模型目录。
2. 用 SGLang `ModelRunner` 加载一层或少量层。
3. 取出 `model_runner.model.model.layers[layer_id].self_attn`。
4. 对 `attention_module(x=hidden_states, positions=positions, forward_batch=forward_batch)` 进行 CUDA Graph 计时。

其计时边界包括：

- Q/KV 相关 projection。
- attention 内部 norm / rope。
- KV cache store。
- CSA/HCA compressor。
- CSA 的 C4 indexer/topk 路径。
- 最终 FlashMLA attention。
- 输出吸收和 attention module 内部返回路径。

其计时边界不包括：

- mHC pre/post。
- attention norm 之外的上游残差流。
- FFN / shared expert / MoE。
- layer 外部调度开销。
- tokenizer、采样、logits 后处理等服务端前后处理。

### Sweep 维度

test cases 来自 `collector/common_test_cases.py`：

- batch size：`[1, 2, 4, ..., 1024]`
- sequence length：从短序列到 `1048575`
- attention kind：`csa` / `hca`
- phase：context / generation
- TP size：Flash 默认 `[1, 2, 4, 8]`，Pro 默认 `[1, 2, 4, 8, 16]`
- KV cache dtype：当前只发出 fp8 KV
- GEMM type：默认 bfloat16；只有检测到 native FP4 capability 时才加入 `fp8_block`

context 过滤规则要求 `bs * sl <= 8192`，对应 chunked prefill 的 new-token budget。generation 过滤规则允许更长 history，但限制 `bs * sl <= 1M`，并对长上下文降低最大 batch。

### 输出 Schema

attention module CSV 行主要字段包括：

- `model`
- `architecture`
- `mla_dtype`
- `kv_cache_dtype`
- `gemm_type`
- `num_heads`
- `batch_size`
- `isl`
- `tp_size`
- `step`
- `compress_ratio`
- `latency`

context 下 `isl=seq_len` 且 `step=0`。generation 下 `isl=1` 且 `step=seq_len`，即把 decode 时的 total/history length 写入 `step`。

## Sparse Kernel Collector 边界

文件：`collector/sglang/deepseekv4_sparse_modules.py`

这个 collector 专门补足 attention module 数据在 prefix/past-kv 方向的稀疏修正。其注释中明确说明只采两个 past-kv-sensitive kernel：

- `deep_gemm.fp8_paged_mqa_logits`
- `flash_mla.flash_mla_with_kvcache`

### `paged_mqa_logits`

采集对象是 CSA indexer scoring kernel。collector 将 `M=bs*isl` 展平成 kernel batch 维度，使用 `b=M, next_n=1` 规避 SM90 kernel 对 `next_n` 的 shared memory 限制。

边界包括：

- FP8 indexer query。
- packed FP8 KV cache。
- block table / sequence length metadata。
- `deep_gemm.fp8_paged_mqa_logits` kernel 本身。

边界不包括：

- indexer projection GEMM。
- topk。
- CSA 后续 attention。
- 完整 self-attn forward。

### `hca_attn`

采集对象是 HCA 的 sparse FlashMLA kernel，即 `flash_mla.flash_mla_with_kvcache`。collector 构造 SWA window 和 c128 extra cache，对齐 V4 backend 的调用语义。

边界包括：

- token-level causal SWA page indices。
- c128 extra page indices。
- packed FP8 sparse K cache。
- FlashMLA sparse attention kernel。

边界不包括：

- Q projection。
- compressor。
- cache 写入。
- mHC。
- FFN/MoE。

### Sweep 维度

sparse kernel test cases 形状为：

```text
[bs, isl, past_kv, tp_size, kernel, model_path]
```

默认维度：

- `bs`: `[1, 2, 4, ..., 1024]`
- `isl`: `[1, 4, 8, ..., 8192]`
- `past_kv`: `[0, 1, 4, ..., 1048575]`
- `tp_size`: 当前 kernel 数据只采 `1`

过滤规则：

- `bs * isl <= 8192`
- `bs * (isl + past_kv) <= 1048576`
- `paged_mqa_logits` 要求 `full_s >= 4`
- `hca_attn` 要求 `full_s >= 64`

虽然查询侧可能传入更大的 TP，`perf_database.py` 会在 sparse kernel lookup 中回退到 `tp=1`，因为这里建模认为这两个 kernel 本身 TP-invariant。

## mHC Collector 边界

文件：`collector/sglang/collect_mhc_module.py`

mHC collector 也是 module-level 风格，但它只计时 mHC 的两个子路径：

- `pre`
- `post`

每个 worker 对单个 op 构建一次 one-layer SGLang runner，然后内部 sweep 全部 `num_tokens`。

### `pre` 边界

`pre` kernel function 会分别调用 attention-site 和 FFN-site 的 `hc_pre`：

```python
return [layer.hc_pre(residual, *args) for args in call_args]
```

这里的 `call_args` 包含：

- `layer.hc_attn_fn`
- `layer.hc_ffn_fn`

因此单条 `pre` latency 表示一个 decoder layer 内两个 mHC site 的合计，而不是单个 site。

### `post` 边界

`post` 先在计时外准备 `post_inputs = layer.hc_pre(...)`，计时内只调用：

```python
return [layer.hc_post(x, residual, post, comb) for x, post, comb, _norm_fused in post_inputs]
```

因此 `post` latency 表示 attention-site 和 FFN-site 的 `hc_post` 合计，不包含前置 `hc_pre` 的耗时。

### 输出 Schema

mHC CSV 主要字段：

- `model`
- `architecture`
- `num_tokens`
- `hc_mult`
- `hidden_size`
- `latency`

`op_name` 使用 `pre` 或 `post`。

## 通用 MoE/GEMM Collector 与 V4 的关系

DeepSeek-V4 模型中 FFN/MoE 相关部分不使用 V4 专用 collector，而是复用已有通用 collector：

- `moe`：`collector.sglang.collect_moe`
- `wideep_moe`：`collector.sglang.collect_wideep_deepep_moe`
- `gemm`：`collector.sglang.collect_gemm`

`collector/common_test_cases.py` 已将 DeepSeek-V4 Flash/Pro 的 MoE 维度加入通用 MoE 配置：

- V4-Flash: hidden size 4096, moe inter size 2048, topk 6, experts 256
- V4-Pro: hidden size 7168, moe inter size 3072, topk 6, experts 384

这些数据服务于 shared FFN、router、MoE dispatch 和 MoE 计算建模，但不属于 DeepSeek-V4 attention 专用采集链路。

## Perf Database 消费路径

`src/aiconfigurator/sdk/perf_database.py` 会加载 split perf files：

- context attention module split files
- generation attention module split files
- sparse kernel correction files
- mHC module file

attention 查询入口：

- `query_context_deepseek_v4_attention_module`
- `query_generation_deepseek_v4_attention_module`

mHC 查询入口：

- `query_mhc_module`

### Context Prefix 修正

context attention module 的主 CSV 采集通常是 `prefix=0`。当仿真查询 `prefix > 0` 时，`perf_database.py` 会优先使用 sparse kernel delta：

- CSA 使用 `paged_mqa_logits` 的 `t_with - t_without`，并额外用公式补 `topk_512` IO delta。
- HCA 使用 `hca_attn` 的 `t_with - t_without`。

如果 sparse kernel 数据不可用，则回退到 SOL ratio scaling。但在 `SILICON` 模式下，缺 sparse correction 数据会影响 prefix context 查询的真实性和完整性。

### Generation 查询

generation attention module 直接使用 generation split CSV，根据：

- quant mode
- architecture
- compress ratio
- recovered TP/head axis
- batch size
- total/history length

做 robust 3D lookup / interpolation。

## 单算子与模块边界对照表

| Collector op | 文件 | 粒度判断 | 实际边界 |
| --- | --- | --- | --- |
| `dsv4_flash_csa_context_module` | `collect_dsv4_flash_attn.py` | 完整 attention module | CSA context `self_attn(...)` |
| `dsv4_flash_hca_context_module` | `collect_dsv4_flash_attn.py` | 完整 attention module | HCA context `self_attn(...)` |
| `dsv4_flash_csa_generation_module` | `collect_dsv4_flash_attn.py` | 完整 attention module | CSA decode `self_attn(...)` |
| `dsv4_flash_hca_generation_module` | `collect_dsv4_flash_attn.py` | 完整 attention module | HCA decode `self_attn(...)` |
| `dsv4_pro_*_module` | `collect_dsv4_flash_attn.py` | 完整 attention module | Pro 维度下同上 |
| `dsv4_*_paged_mqa_logits_module` | `deepseekv4_sparse_modules.py` | 单 kernel | CSA indexer scoring |
| `dsv4_*_hca_attn_module` | `deepseekv4_sparse_modules.py` | 单 kernel | HCA sparse FlashMLA |
| `mhc_module` | `collect_mhc_module.py` | 子模块 | mHC pre/post，且每条包含 attention-site + FFN-site |
| `moe` / `wideep_moe` | 通用 collector | 通用 MoE | V4 routed experts/dispatch 相关 |
| `gemm` | 通用 collector | 通用 GEMM | shared FFN、router、projection fallback 等通用矩阵乘 |

## 采集边界的关键注意事项

1. `collect_dsv4_flash_attn.py` 的 attention module 数据已经包含若干底层 kernel 的组合，不应再与 projection GEMM、FlashMLA kernel 等重复相加。
2. `deepseekv4_sparse_modules.py` 的 kernel 数据主要用于 prefix/past-kv 修正，不代表完整 attention latency。
3. `mhc_module` 的 `pre` / `post` 均是一个 decoder layer 内两个 mHC site 的合计，与 SDK 中 `DeepSeekV4MHCModule` 的 SOL 估算边界保持一致。
4. `compress_ratio=0` 的 SWA 没有独立 collector，模型层折叠到 HCA perf 数据。
5. sparse kernel 文件名虽然带 `module`，但实际是 kernel-level CSV；这是命名兼容，不应按完整 module 解读。
6. `--sglang-version-branch` 只处理 SGLang API 兼容，例如 `ForwardBatch` 构造和 legacy forward context，不改变 registry 中 DeepSeek-V4 op 的选择，也不改变上述采集边界。

## 建议的 DeepSeek-V4 专用采集集合

若目标是覆盖 DeepSeek-V4 attention/mHC 的核心仿真数据，推荐至少采集：

```bash
python3 collect.py \
  --backend sglang \
  --model-path sgl-project/DeepSeek-V4-Pro-FP8 \
  --ops \
    dsv4_pro_csa_context_module \
    dsv4_pro_hca_context_module \
    dsv4_pro_csa_generation_module \
    dsv4_pro_hca_generation_module \
    dsv4_pro_paged_mqa_logits_module \
    dsv4_pro_hca_attn_module \
    mhc_module
```

若还需要 MoE/FFN 全链路仿真数据，需要额外采集通用 `moe`、`wideep_moe`、`gemm` 等数据，具体取决于当前 backend 和 perf database 对这些 op 的数据可用性要求。
