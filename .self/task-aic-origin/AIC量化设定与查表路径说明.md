# AIC 原生仿真量化设定与查表路径说明

本文聚焦 AIC SDK 原生仿真中与量化相关的设定，覆盖 collect 侧数据储备、SDK 执行侧参数传递与查表逻辑，以及 probe 试运行能看到的量化细节。

本次验证脚本：

- [run_aic_origin_probe.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-aic-origin/run_aic_origin_probe.py)

本次验证产物：

- [20260622_155838_aic_origin_sglang_deepseek_v3_pd_static](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-aic-origin/runs/20260622_155838_aic_origin_sglang_deepseek_v3_pd_static)
- [prefill/quant_summary.json](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-aic-origin/runs/20260622_155838_aic_origin_sglang_deepseek_v3_pd_static/prefill/quant_summary.json)
- [decode/quant_summary.json](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-aic-origin/runs/20260622_155838_aic_origin_sglang_deepseek_v3_pd_static/decode/quant_summary.json)

## 1. 量化概念边界

这里需要把三类信息拆开，不然很容易把所有 `fp8` 混成一个概念：

- **输入激活值**：模型运行时进入算子的 activation，如 GEMM 的 A 矩阵、attention 的 q/k/v。collector 中很多输入仍先用 BF16 随机张量构造，再在 kernel 内部或调用前动态量化。
- **模型参数/权重**：GEMM/MoE/MLA BMM 的权重存储格式，如 BF16、FP8 block、NVFP4。SDK 里 `GEMMQuantMode` / `MoEQuantMode` 会同时影响参数 memory 估算和查表 key。
- **计算执行/Kernel 精度**：算子实际走的 kernel 路径和数学吞吐假设，如 attention 的 `FMHAQuantMode`。对 attention 来说，它不是简单等于 KV cache dtype；例如 DeepSeek V3 在本次默认配置下是 `kvcache=fp8`，但 `fmha=bfloat16`。

SDK 的枚举定义在 [common.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/common.py)，核心含义是 `QuantMapping(memory, compute, name)`：

- `memory`：按该模式估算存储/搬运字节数。
- `compute`：按该模式估算相对 BF16 Tensor Core 的计算吞吐倍率。
- `name`：和 perf 数据文件中的 dtype 字符串基本对应。

## 2. collect 侧数据储备

### 2.1 GEMM

collector 入口为 [collect_gemm.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/collector/sglang/collect_gemm.py)。

采集结果写入 `gemm_perf.txt`，关键列：

- `gemm_dtype`
- `m, n, k`
- `latency`

H100/SGLang 0.5.9 当前数据：

- 文件：[gemm_perf.txt](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.9/gemm_perf.txt)
- 行数：`101007`
- `gemm_dtype`: `bfloat16`, `fp8`, `fp8_block`

collector 侧语义：

- `bfloat16`：输入激活和权重均为 BF16，调用 `F.linear`。
- `fp8`：激活先由 BF16 动态量化为 FP8，权重预转 FP8，调用 `fp8_scaled_mm`。
- `fp8_block`：激活由 BF16 进行 per-token group FP8 量化，权重为 FP8 block，调用 DeepGEMM `gemm_nt_f8f8bf16`，输出 BF16。

因此 `gemm_dtype=fp8_block` 不是“输入原本就是 FP8”，而是表示采集了 SGLang/DeepGEMM 的 W8A8 block FP8 执行路径，其中 activation 量化成本包含在 kernel function 计时中。

### 2.2 MoE

collector 入口为 [collect_moe.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/collector/sglang/collect_moe.py)。

采集结果写入 `moe_perf.txt`，关键列：

- `moe_dtype`
- `num_tokens, hidden_size, inter_size`
- `topk, num_experts, moe_tp_size, moe_ep_size`
- `distribution`
- `latency`

H100/SGLang 0.5.9 当前数据：

- 文件：[moe_perf.txt](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.9/moe_perf.txt)
- 行数：`76869`
- `moe_dtype`: `bfloat16`, `fp8_block`, `int4_wo`

collector 侧语义：

- `bfloat16`：专家权重和激活按 BF16 路径执行。
- `fp8_block`：构造 FP8 block MoE 路径，hidden/inter 维度需要满足 block 约束。
- `int4_wo`：weight-only int4，activation 仍可为 BF16。

这里 `moe_dtype` 同时表达专家权重量化和 fused MoE kernel 路径；dispatch/alltoall 的通信量化不是这个字段负责。

### 2.3 标准 Attention

collector 入口为 [collect_attn.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/collector/sglang/collect_attn.py)。

采集结果：

- [context_attention_perf.txt](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.9/context_attention_perf.txt)
- [generation_attention_perf.txt](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.9/generation_attention_perf.txt)

关键列：

- `attn_dtype`
- `kv_cache_dtype`
- `batch_size, isl, num_heads, num_key_value_heads, head_dim`
- `step`
- `latency`

H100/SGLang 0.5.9 当前数据：

- `context_attention_perf.txt`: `16881` 行，`attn_dtype=bfloat16/fp8`，`kv_cache_dtype=bfloat16/fp8`
- `generation_attention_perf.txt`: `10186` 行，`attn_dtype=bfloat16`，`kv_cache_dtype=bfloat16/fp8`

collector 侧语义：

- `kv_cache_dtype` 控制 `MHATokenToKVPool` 中 K/V cache 的存储 dtype。
- prefill 下，如果 `use_fp8_context_fmha=True`，collector 会把 q/k/v 显式 cast 到 FP8，因此 `attn_dtype=fp8` 表达 attention kernel 的输入/计算路径。
- decode 下，当前采集数据的 `attn_dtype` 只有 BF16；FP8 KV cache 只改变 cache 存储/读取，live q/k/v activation 仍是 BF16。

### 2.4 MLA Attention 和 MLA BMM

collector 入口：

- [collect_mla.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/collector/sglang/collect_mla.py)
- [collect_mla_bmm.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/collector/sglang/collect_mla_bmm.py)

采集结果：

- [context_mla_perf.txt](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.9/context_mla_perf.txt)
- [generation_mla_perf.txt](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.9/generation_mla_perf.txt)
- [mla_bmm_perf.txt](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.9/mla_bmm_perf.txt)

H100/SGLang 0.5.9 当前数据：

- `context_mla_perf.txt`: `3080` 行，`mla_dtype=bfloat16`，`kv_cache_dtype=bfloat16/fp8`
- `generation_mla_perf.txt`: `4648` 行，`mla_dtype=bfloat16`，`kv_cache_dtype=bfloat16/fp8`
- `mla_bmm_perf.txt`: `848` 行，`bmm_dtype=bfloat16/fp8`

collector 侧语义：

- `collect_mla.py` 构造 SGLang 的 `RadixAttention` + MLA KV pool。`kv_cache_dtype` 控制 latent KV cache 存储格式。
- 当前 H100 0.5.9 数据中 `mla_dtype` 只有 BF16；即便 KV cache 是 FP8，attention compute/table 轴仍按 BF16 MLA 查。
- `collect_mla_bmm.py` 采集 decode MLA 的前后两个 absorption BMM，`bmm_dtype=fp8` 时会对 BF16 activation 做 `per_tensor_quant_mla_fp8` 并调用 `bmm_fp8`。

这解释了一个重要现象：DeepSeek V3 decode 的 MLA 路径会同时查询 `generation_mla_perf` 和 `mla_bmm_perf`。前者由 KV cache dtype 区分，后者由 GEMM 派生出的 `mla_bmm_quant_mode` 区分。

### 2.5 MLA Module 与 WideEP MLA

collector 入口为 [collect_mla_module.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/collector/sglang/collect_mla_module.py)。

这份脚本同时服务两类“模块级/整体 attention”数据，容易混淆：

- **普通 module-level MLA/DSA**：输出到 `mla_context_module_perf.txt` / `mla_generation_module_perf.txt` 或 `dsa_*_module_perf.txt`，op 名类似 `mla_context_module`。
- **WideEP MLA 兼容旧格式**：当 `attn_type == "mla"` 时，脚本刻意输出到 `wideep_context_mla_perf.txt` / `wideep_generation_mla_perf.txt`，op 名仍是旧格式的 `mla_context` / `mla_generation`。

代码里 `_get_precision_combos()` 原本能枚举 `(compute_dtype, kv_cache_dtype, gemm_type)`：

- `compute_dtype`: 当前设计为 `bfloat16`。
- `kv_cache_dtype`: SM90+ 包含 `bfloat16` 和 `fp8`。
- `gemm_type`: SM89+ 包含 `bfloat16` 和 `fp8_block`。

但 WideEP MLA 分支为了兼容旧 `collect_wideep_attn.py` 数据格式，会把日志字段强行写为：

- `mla_dtype = fp8_block`
- `kv_cache_dtype = fp8`
- `gemm_type = fp8_block`

也就是说，WideEP MLA 表不是普通 granular `context_mla_perf/generation_mla_perf` 的 BF16-compute 表，而是面向 SGLang WideEP DeepSeek 的“整体 MLA attention forward”表，记录了 FP8-block 权重路径、FP8 KV cache 和指定 attention backend 下的端到端模块时间。

H100/SGLang 0.5.9 当前数据：

- 文件：[wideep_context_mla_perf.txt](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.9/wideep_context_mla_perf.txt)，数据行 `1000`
- 文件：[wideep_generation_mla_perf.txt](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.9/wideep_generation_mla_perf.txt)，数据行 `1056`
- `kernel_source`: `flashinfer`, `fa3`
- `model`: `deepseek-ai/DeepSeek-V3`
- `architecture`: `DeepseekV3ForCausalLM`
- `mla_dtype`: `fp8_block`
- `kv_cache_dtype`: `fp8`
- `gemm_type`: `fp8_block`
- `num_heads`: `128`, `64`, `32`, `16`

与此相对，H100/SGLang 0.5.9 当前仍缺少普通 module-level MLA 表：

- `mla_context_module_perf.txt`: 不存在
- `mla_generation_module_perf.txt`: 不存在

## 3. SDK 执行侧量化设定

### 3.1 默认化入口

SDK 入口在 [models/__init__.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/models/__init__.py)：

- `get_model()` 读取 HF config / model info。
- 调用 `_apply_model_quant_defaults()`。
- 再创建具体模型类。

默认化逻辑在 [models/helpers.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/models/helpers.py)：

- `quant_algo=fp8_block` 会推断 `gemm_quant_mode=fp8_block`、`moe_quant_mode=fp8_block`。
- `kv_cache_quant_algo=fp8` 会推断 `kvcache_quant_mode=fp8`。
- 如果存在量化算法或 FP8 KV cache，初始会把 `fmha_quant_mode` 推断为 `fp8`。
- 但 DeepSeek V3 有显式 workaround：当 architecture 为 `DeepseekV3ForCausalLM` 且 `fmha_quant_mode=fp8` 时，会改回 `bfloat16`。

本次 probe 的最终默认值：

- `gemm_quant_mode=fp8_block`
- `moe_quant_mode=fp8_block`
- `kvcache_quant_mode=fp8`
- `fmha_quant_mode=bfloat16`
- `comm_quant_mode=half`

证据见：

- [prefill/quant_summary.json](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-aic-origin/runs/20260622_155838_aic_origin_sglang_deepseek_v3_pd_static/prefill/quant_summary.json)
- [decode/quant_summary.json](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-aic-origin/runs/20260622_155838_aic_origin_sglang_deepseek_v3_pd_static/decode/quant_summary.json)

### 3.2 DeepSeek V3 ops 如何使用量化参数

DeepSeek V3 代码在 [deepseek.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/models/deepseek.py)。

Context/prefill MLA block：

- primary: `MLAModule(..., kvcache_quant_mode, fmha_quant_mode, gemm_quant_mode)`
- fallback:
  - `context_downscale_gemm`: `gemm_quant_mode`
  - `context_q_b_proj_gemm`: `gemm_quant_mode`
  - `context_kv_b_proj_gemm`: `gemm_quant_mode`
  - SGLang backend 下用 `ContextMLA(..., kvcache_quant_mode, fmha_quant_mode)`
  - `context_proj_gemm`: `gemm_quant_mode`

Decode/generation MLA block：

- primary: `MLAModule(..., kvcache_quant_mode, fmha_quant_mode, gemm_quant_mode)`
- fallback:
  - `generation_downscale_gemm`: `gemm_quant_mode`
  - `generation_q_b_proj_gemm`: `gemm_quant_mode`
  - `generation_bmm_pre`: `mla_bmm_quant_mode`
  - `generation_attention`: `GenerationMLA(..., kvcache_quant_mode)`
  - `generation_bmm_post`: `mla_bmm_quant_mode`
  - `generation_proj_gemm`: `gemm_quant_mode`

其中 `mla_bmm_quant_mode` 是派生值：

- 若 `gemm_quant_mode != bfloat16`，则 `mla_bmm_quant_mode=fp8`
- 否则为 `bfloat16`

这意味着 DeepSeek V3 decode 的 absorption BMM 不直接使用 `fp8_block` 表，而是查 `mla_bmm_perf.txt` 的 `bmm_dtype=fp8`。

### 3.3 WideEP DeepSeek 如何使用 WideEP MLA 表

DeepSeek 模型创建入口位于 [deepseek.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/models/deepseek.py)：

- `backend_name == "sglang"` 且 `model_config.moe_backend == "deepep_moe"` 时，会返回 `WideEPDeepSeekModel`。
- 普通 `backend=sglang` 且未启用 `deepep_moe` 时，仍是普通 `DeepSeekModel`，不会查询 `wideep_*_mla_perf.txt`。

`WideEPDeepSeekModel` 中的 attention ops：

- context: `ops.WideEPContextMLA("context_attention", ..., tp_size, kvcache_quant_mode, fmha_quant_mode, attn_backend)`
- generation: `ops.WideEPGenerationMLA("generation_attention", ..., tp_size, kvcache_quant_mode, fmha_quant_mode, attn_backend)`

这两类 op 分别调用：

- `PerfDatabase.query_wideep_context_mla(...)`
- `PerfDatabase.query_wideep_generation_mla(...)`

需要特别注意：

- `WideEPContextMLA` 的 SILICON 查表使用 `kernel_source -> fmha_quant_mode -> kvcache_quant_mode -> num_heads -> full_s -> batch_size`。
- `WideEPGenerationMLA` 的 SILICON 查表使用 `kernel_source -> kvcache_quant_mode -> num_heads -> batch_size -> s`，不使用 `fmha_quant_mode` 作为表索引。
- `attn_backend` 来自 `model_config.attention_backend`，支持 `flashinfer` / `fa3` 两套表。
- `num_heads` 由 `128 // tp_size` 换算得到，因此 `tp_size=8` 对应查 `num_heads=16`。

本次额外试运行也验证了一个兼容性细节：

- 直接 `--moe-backend deepep_moe` 且保持 DeepSeek V3 默认 `fmha=bfloat16` 时，WideEP context MLA 会查 `wideep_context_mla_perf` 的 `bfloat16` 轴，但 H100 0.5.9 表只有 `fp8_block`，因此 SILICON 查询失败。
- 显式使用 `--fmha-quant-mode fp8_block` 后，WideEP MLA attention 表本身可查；但完整 WideEP 端到端还会继续依赖 `wideep_deepep_normal_perf.txt` 等 DeepEP/MoE 相关表，H100/SGLang 0.5.9 当前缺少该表，完整 probe 会在后续 MoE/通信路径失败。
- 窄口径直接调用 `query_wideep_context_mla/query_wideep_generation_mla` 已验证 `flashinfer` 和 `fa3` 两个 backend 均可返回 SILICON latency。

### 3.4 operations 层如何查询

关键 op 在 [operations.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/operations.py)：

- `GEMM.query()` 调 `database.query_gemm(x, n, k, quant_mode)`。
- `MoE.query()` 调 MoE 数据库查询，`quant_mode` 参与表索引、参数 memory 和 compute 假设。
- `ContextAttention.query()` 调 `query_context_attention(..., kvcache_quant_mode, fmha_quant_mode)`。
- `ContextMLA.query()` 调 `query_context_mla(..., kvcache_quant_mode, fmha_quant_mode)`。
- `GenerationMLA.query()` 调 `query_generation_mla(..., kvcache_quant_mode)`。
- `MLABmm.query()` 调 `query_mla_bmm(..., quant_mode, if_pre)`。
- `MLAModule.query()` 调 module-level MLA 表，索引包含 `kvcache_quant_mode/fmha_quant_mode/gemm_quant_mode`。
- `WideEPContextMLA.query()` 调 `query_wideep_context_mla(..., kvcache_quant_mode, fmha_quant_mode, attention_backend)`。
- `WideEPGenerationMLA.query()` 调 `query_wideep_generation_mla(..., kvcache_quant_mode, fmha_quant_mode, attention_backend)`。

本次 H100/SGLang 0.5.9 缺少 module-level MLA 表，因此 `MLAModule` primary 查询失败后 fallback 到 granular ops。证据：

- `prefill/query_trace.json` 中 `context_mla_module` 抛 `PerfDataNotAvailableError`
- `decode/query_trace.json` 中 `generation_mla_module` 抛 `PerfDataNotAvailableError`
- `quant_summary.json` 的 `database_quant_tables` 中 `mla_context_module_perf.txt` / `mla_generation_module_perf.txt` 为 `loaded=false`

## 4. PerfDatabase 查表逻辑

查表加载逻辑在 [perf_database.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/perf_database.py)。

关键表结构：

- `gemm_perf.txt`: `GEMMQuantMode -> m -> n -> k`
- `moe_perf.txt`: `MoEQuantMode -> distribution -> topk -> num_experts -> hidden_size -> inter_size -> moe_tp -> moe_ep`
- `context_attention_perf.txt`: `FMHAQuantMode -> KVCacheQuantMode -> n_kv -> head_size -> window_size -> n -> full_s -> b`
- `generation_attention_perf.txt`: `KVCacheQuantMode -> n_kv -> head_size -> window_size -> n -> b -> s`
- `context_mla_perf.txt`: `FMHAQuantMode -> KVCacheQuantMode -> num_heads -> full_s -> b`
- `generation_mla_perf.txt`: `KVCacheQuantMode -> num_heads -> b -> s`
- `mla_bmm_perf.txt`: `GEMMQuantMode -> mla_gen_pre/post -> num_heads -> num_tokens`
- `wideep_context_mla_perf.txt`: `kernel_source -> FMHAQuantMode -> KVCacheQuantMode -> num_heads -> full_s -> b`
- `wideep_generation_mla_perf.txt`: `kernel_source -> KVCacheQuantMode -> num_heads -> b -> s`
- `mla_context_module_perf.txt`: `FMHAQuantMode -> KVCacheQuantMode -> GEMMQuantMode -> architecture -> num_heads -> full_s -> b`
- `mla_generation_module_perf.txt`: `FMHAQuantMode -> KVCacheQuantMode -> GEMMQuantMode -> architecture -> num_heads -> b -> s`

注意两个不对称点：

- `generation_attention_perf` 的 loader 读了 `attn_dtype`，但最终顶层索引只按 `kv_cache_dtype` 建表；当前数据本身也只有 `attn_dtype=bfloat16`。
- `generation_mla_perf` 同样读 `mla_dtype`，但最终顶层索引只按 `kv_cache_dtype` 建表；当前 H100 数据 `mla_dtype` 也只有 BF16。
- `wideep_generation_mla_perf` 文件包含 `mla_dtype` 和 `gemm_type` 列，但 loader 当前只把 `kernel_source/kv_cache_dtype/num_heads/b/s` 作为 SILICON 表索引；`wideep_context_mla_perf` 则会把 `mla_dtype` 映射成 `FMHAQuantMode` 参与索引。
- 普通 `mla_context_module_perf/mla_generation_module_perf` loader 会读取 `gemm_type` 并映射为 `GEMMQuantMode`，但 H100/SGLang 0.5.9 当前没有对应文件。

在 SOL / EMPIRICAL fallback 中，量化 mode 还会影响估算公式：

- GEMM/MoE 用 `quant_mode.value.memory` 估算权重/通信体积，用 `quant_mode.value.compute` 估算 Tensor Core 吞吐倍率。
- Context attention/MLA 用 `kvcache_quant_mode.value.memory` 估算 KV cache bytes，用 `fmha_quant_mode.value.compute` 估算 attention compute。
- Generation MLA 若 KV cache 为 FP8，会在 SOL 中派生 `quant_mode_gen=FMHAQuantMode.fp8`，但 SILICON 路径仍主要按 `generation_mla_perf` 的 KV cache 维度查表。
- WideEP MLA 的 SOL 公式把 q_b/kv_b/o_proj 等 attention 内部投影和 attention 计算合并估算；其中部分 memory/compute 使用 `fmha_quant_mode`，KV cache 使用 `kvcache_quant_mode`。但 SILICON 路径以 `wideep_*_mla_perf` 的实际表轴为准。

## 5. 本次原生 AIC 输出能看到什么

原生 `summary.get_summary_df()` 已能看到粗粒度量化字段：

- `gemm=fp8_block`
- `moe=fp8_block`
- `kvcache=fp8`
- `fmha=bfloat16`
- `comm=half`

本次结果：

- prefill `ttft/context_latency = 67.469 ms`
- decode `tpot = 27.683 ms`
- decode `generation_latency = 415.246 ms`
- prefill query events: `19`
- decode query events: `81`

但原生 summary 看不到：

- 每个 op 实例实际持有哪些 `_quant_mode/_kvcache_quant_mode/_fmha_quant_mode`。
- module-level MLA 失败后 fallback 子 op 的量化细节。
- 每张 perf table 的可用量化轴。
- `generation_bmm_pre/post` 为什么查 `bmm_dtype=fp8` 而不是 `fp8_block`。
- SILICON 查表时某个点是否直接命中或插值。

## 6. probe 已补充的量化信息

本次已修改 [run_aic_origin_probe.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-aic-origin/run_aic_origin_probe.py)，新增：

- `query_trace.json` 中每个事件包含 `op_quant_fields`。
- 每个 role 输出 `quant_summary.json`。
- `quant_summary.json` 汇总：
  - `backend_quant_defaults_after_model_build`
  - `operation_quant_fields`
  - `query_quant_events`
  - `database_quant_tables`

示例结论：

- `ContextMLA(context_attention)` 持有 `kvcache_quant_mode=fp8`、`fmha_quant_mode=bfloat16`、`num_heads=16`。
- `generation_bmm_pre/post` 持有 `_quant_mode=fp8`，这是由 `gemm_quant_mode=fp8_block` 派生而来。
- `context_mla_module` primary 持有 `fmha=bfloat16/kvcache=fp8/gemm=fp8_block`，但 module 表未加载，因此 fallback 生效。
- 数据库中 `context_mla_perf.txt` 顶层只有 `bfloat16`，而 `kv_cache_dtype` 二级有 `bfloat16/fp8`。
- 数据库中 `gemm_perf.txt` 顶层有 `fp8/bfloat16/fp8_block`。
- 数据库中 `wideep_context_mla_perf.txt` 顶层为 `flashinfer/fa3`，下一层为 `fp8_block`；`wideep_generation_mla_perf.txt` 顶层为 `flashinfer/fa3`，下一层为 `fp8`。

## 7. 当前对照结论

对 DeepSeek V3、H100、SGLang 0.5.9、SILICON 模式：

- GEMM 和 MoE 的默认执行量化来自模型 config 推断，最终为 `fp8_block`；collector 和数据库都具备对应数据。
- KV cache 默认为 `fp8`；attention/MLA 表均包含 `kv_cache_dtype=fp8` 数据。
- FMHA/MLA compute 在 DeepSeek V3 上被 SDK workaround 固定为 `bfloat16`；这与当前 H100 `context_mla_perf/generation_mla_perf` 只有 `mla_dtype=bfloat16` 对齐。
- module-level MLA collector 虽有脚本，但 H100/SGLang 0.5.9 系统数据目录缺少 `mla_context_module_perf.txt` / `mla_generation_module_perf.txt`，实际仿真回退到 granular ops。
- WideEP MLA 数据是另一套专用整体 attention 表：H100/SGLang 0.5.9 存在 `wideep_context_mla_perf.txt` / `wideep_generation_mla_perf.txt`，由 `collect_mla_module.py` 的 WideEP 兼容分支生成，并由 `WideEPDeepSeekModel` 使用。
- WideEP context MLA 表的量化轴是 `fmha=fp8_block, kvcache=fp8`，这与普通 DeepSeek V3 默认 workaround 后的 `fmha=bfloat16` 不一致；因此要在 SILICON 下使用 WideEP context MLA 表，需要确保模型配置/查询入参走 `FMHAQuantMode.fp8_block`，或者补采 `bfloat16` 轴数据。
- WideEP generation MLA 表当前按 `kvcache=fp8` 索引，文件中的 `mla_dtype/gemm_type=fp8_block` 主要是历史兼容记录，不进入当前 generation SILICON 查表主键。
- 原生 AIC summary 只能给出 coarse quant mode；若要解释具体误差来源，必须结合 `query_trace.json`、`quant_summary.json` 和 `interp_trace.json`。

## 8. 风险与限制

- `quant_summary.json` 是 probe 派生信息，不是 SDK 原生 API 输出；但它只读对象字段，不改变执行逻辑。
- `operation_quant_fields` 反映 op instance 持有的配置，不代表底层 CUDA kernel 一定以完全相同 dtype 计算；底层真实行为还要看 collector/SGLang kernel 实现。
- `generation_attention_perf` 和 `generation_mla_perf` 的 loader 当前没有把 `attn_dtype/mla_dtype` 作为顶层索引，这是因为现有数据只有 BF16 compute；如果未来采集 decode FP8 compute，需要同步扩展加载/查询维度。
- `wideep_context_mla_perf` 的现有 H100 数据只覆盖 `fmha=fp8_block, kvcache=fp8`；直接用 DeepSeek V3 默认 `fmha=bfloat16` 启动 WideEP SILICON 估算会查不到这张表。
- 完整 SGLang WideEP/DeepEP 仿真不仅需要 `wideep_*_mla_perf`，还需要 DeepEP/MoE/通信相关 wideep 表；本次 H100/SGLang 0.5.9 缺少 `wideep_deepep_normal_perf.txt`，因此不能把“WideEP MLA 单表可查”误解为“WideEP 端到端可完整跑通”。
- `interp_trace.json` 的直接命中/插值分类是 best-effort；SDK 原生 `_interp_3d()` 不返回邻点或 row id。
