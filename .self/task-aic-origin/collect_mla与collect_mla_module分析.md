# collect_mla 与 collect_mla_module 分析

本文聚焦 `collector/sglang/collect_mla.py` 与 `collector/sglang/collect_mla_module.py` 两个脚本，说明它们在 SGLang 中的边界、覆盖的计算单元、输入输出表项、执行过程，以及在 H100 / SGLang 0.5.9 数据中实际采到了哪些场景。

## 1. 两个脚本的定位

### 1.1 `collect_mla.py`

这是**kernel 级 MLA 采集**，关注的是单个 attention kernel / 子算子，而不是完整 attention 模块。

它主要覆盖：

- `RadixAttention` 驱动下的 MLA context / generation kernel。
- `MLATokenToKVPool` / `ReqToTokenPool` 相关 KV cache 读写。
- SGLang 不同 MLA backend 的单 kernel 计时。

它导出的核心表：

- `context_mla_perf.txt`
- `generation_mla_perf.txt`
- `mla_bmm_perf.txt`

### 1.2 `collect_mla_module.py`

这是**module 级 attention 采集**，关注的是完整 attention forward pipeline。

它覆盖：

- 普通 module 级 MLA / DSA
- WideEP 兼容的 MLA module
- 完整 forward 中的 qkv_a / downscale / attention / proj / dispatch / MoE / 通信前后链路

它导出的核心表：

- `mla_context_module_perf.txt`
- `mla_generation_module_perf.txt`
- `wideep_context_mla_perf.txt`
- `wideep_generation_mla_perf.txt`
- DSA 对应的 `dsa_*_module_perf.txt`

## 2. `collect_mla.py` 的边界与执行单元

### 2.1 SGLang 组件边界

脚本直接用到的 SGLang 组件：

- `RadixAttention`
- `ForwardBatch`
- `MLATokenToKVPool`
- `ReqToTokenPool`
- `FlashAttentionBackend` / `TritonAttnBackend` / `TRTLLMMLABackend`
- `sglang.srt.server_args`

也就是说，它不是在采集 server/router 层，而是在**单 attention kernel + cache 读写 + backend 调度**层面做测量。

### 2.2 覆盖的计算单元

context phase：

- MLA attention kernel 本体
- KV cache 写入
- q/k/v 形状准备

generation phase：

- MLA decode kernel 本体
- 已缓存历史 KV 的读取
- 单 token decode 计算

### 2.3 输入数据

脚本的 test case 轴是：

- `batch_size`
- `input_len`
- `num_heads`
- `num_key_value_heads`
- `head_dim`
- `use_fp8_kv_cache`
- `use_fp8_context_fmha`
- `is_context_phase`

对 MLA 路径还包括：

- `tp_size`
- `selected_backend`
- `kv_lora_rank`
- `qk_nope_head_dim`
- `qk_rope_head_dim`
- `v_head_dim`

### 2.4 输出表项含义

`collect_mla.py` 记录的核心字段：

- `mla_dtype`
- `kv_cache_dtype`
- `num_heads`
- `batch_size`
- `isl`
- `tp_size`
- `step`
- `latency`

这里的含义要区分开：

- `mla_dtype` 更偏 attention 计算/模块标识，不等于 KV cache。
- `kv_cache_dtype` 表示 KV cache 的存储/读写精度。
- `tp_size` 是逻辑 TP 语义下的头切分因子。
- `step` 在 context 中是 0，在 generation 中是 `input_len`。

### 2.5 执行过程

context：

1. 构造 `ReqToTokenPool`。
2. 按 batch / seq_len 构造 `MLATokenToKVPool`。
3. 根据 SM 自动选择 backend：
   - SM >= 100 且 CUDA 12.8+ -> `trtllm_mla`
   - SM >= 90 且 CUDA 12.3+ -> `fa3`
   - 否则 -> `triton`
4. 准备 `ForwardBatch(EXTEND)`。
5. 准备 q_nope / q_rope / k_nope / k_rope / v。
6. 运行 `RadixAttention(...)`。
7. warmup 后测时延并写 perf 表。

generation：

1. 构造历史 KV cache。
2. 准备 `ForwardBatch(DECODE)`。
3. 预热 JIT / autotune，特别是 Blackwell + reduced heads 场景。
4. 运行 `RadixAttention(...)`。
5. 写 perf 表。

### 2.6 量化格式处理

脚本对 MLA 的量化处理大致是：

- `kv_cache_dtype=bfloat16/fp8`
- context 下 `use_fp8_context_fmha=True` 时，q/k/v 会被显式 cast 到 FP8
- decode 下 KV cache 可为 FP8，但 live query activation 仍主要是 BF16

也就是说，这里量化既影响：

- cache 存储格式
- kernel 输入格式
- backend 选择

## 3. `collect_mla_module.py` 的边界与执行单元

### 3.1 SGLang 组件边界

这个脚本更靠近 **ModelRunner / ServerArgs / ScheduleBatch / ForwardBatch** 级别。

它直接触及：

- `ServerArgs`
- `ModelConfig.from_server_args`
- `ModelRunner`
- `ScheduleBatch`
- `ChunkCache`
- `ForwardBatch`
- `AttentionInputs`
- `RadixAttention`

因此它采集的是**完整 attention 模块前向链路**，而不是单个 kernel。

### 3.2 覆盖的计算单元

普通 module / DSA 分支中，context/generation 会覆盖：

- qkv_a / downscale GEMM
- attention module
- proj GEMM
- dispatch / MoE / 通信（DSA 时更多）

WideEP MLA 分支中，则覆盖：

- qkv_a projection
- downscale GEMM
- MLA attention module
- communication 前后衔接

### 3.3 输入数据

module 级 test case 轴比 kernel 级更少，主要是：

- `seq_len`
- `batch_size`
- `num_heads`
- `kv_cache_dtype`
- `compute_dtype`
- `gemm_type`
- `model_path`
- `attn_type`
- `attention_backend`

另外还会在内部固定：

- `_HEAD_NUMS` / `_MODULE_HEAD_NUMS`
- `MODEL_NATIVE_HEADS`
- `SUPPORTED_MODELS`

### 3.4 输出表项含义

普通 module 级（非 WideEP）：

- `attn_dtype`
- `kv_cache_dtype`
- `gemm_type`
- `num_heads`
- `batch_size`
- `isl`
- `tp_size`
- `step`
- `latency`

WideEP MLA：

- `mla_dtype`
- `kv_cache_dtype`
- `gemm_type`
- `num_heads`
- `batch_size`
- `isl`
- `tp_size`
- `step`
- `latency`

### 3.5 Attn-backend 选择

普通 DSA / module 路径：

- `attn_type == dsa` 时 backend 用 `nsa`
- 后端通过 `_get_backends()` 对齐 SGLang 默认值

WideEP MLA：

- SM >= 100 -> `trtllm_mla`
- SM >= 90 -> `flashinfer`, `fa3`
- SM < 90 -> 不采

### 3.6 量化格式处理

普通 module 级：

- `_get_module_precision_combos()` 只保留一个 baseline：`bfloat16/bfloat16/bfloat16`
- 这是为了减少 subprocess 数量

WideEP MLA：

- 兼容旧 `collect_wideep_attn.py`
- 日志字段固定成：
  - `mla_dtype=fp8_block`
  - `kv_cache_dtype=fp8`
  - `gemm_type=fp8_block`

这说明 WideEP MLA 记录的不是“可自由 sweep 的多精度模块”，而是**一套历史兼容的整体 MLA forward 表**。

## 4. H100 / SGLang 0.5.9 的实际数据对应

### 4.1 `collect_mla.py` 对应的数据

H100 / 0.5.9 目录中：

- [context_mla_perf.txt](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.9/context_mla_perf.txt)
- [generation_mla_perf.txt](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.9/generation_mla_perf.txt)
- [mla_bmm_perf.txt](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.9/mla_bmm_perf.txt)

可见特征：

- `context_mla_perf.txt`
  - `mla_dtype=bfloat16`
  - `kv_cache_dtype=bfloat16/fp8`
  - `kernel_source=flash_attention`
- `generation_mla_perf.txt`
  - `mla_dtype=bfloat16`
  - `kv_cache_dtype=bfloat16/fp8`
  - `kernel_source=flash_attention`
- `mla_bmm_perf.txt`
  - `bmm_dtype=bfloat16/fp8`

结论：

- H100 上的 kernel 级 MLA 采集，主要是 **FA / FlashAttention 路径**。
- 计算精度方面，MLA 相关 attention compute 仍以 BF16 为主。
- FP8 主要体现在 KV cache 和 absorption BMM 的一部分路径。

### 4.2 `collect_mla_module.py` 对应的数据

H100 / 0.5.9 中：

- [wideep_context_mla_perf.txt](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.9/wideep_context_mla_perf.txt)
- [wideep_generation_mla_perf.txt](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.9/wideep_generation_mla_perf.txt)

可见特征：

- `kernel_source=flashinfer/fa3`
- `model=deepseek-ai/DeepSeek-V3`
- `architecture=DeepseekV3ForCausalLM`
- `mla_dtype=fp8_block`
- `kv_cache_dtype=fp8`
- `gemm_type=fp8_block`
- `num_heads=128/64/32/16`
- `tp_size=1`

结论：

- WideEP MLA 是一套**module / overall MLA forward** 的兼容表。
- 它不是普通 `context_mla_perf` / `generation_mla_perf` 那套 BF16 compute kernel 表。
- H100 上这两张表只记录了 `fp8_block + fp8` 这条比较特殊的组合。

### 4.3 H100 的架构特性

H100 的几个明显特征：

- `collect_mla.py` 默认能走 `fa3` / `flash_attention` 路线。
- `collect_mla_module.py` 的 WideEP MLA 兼容表会记录 `flashinfer` 和 `fa3` 两种 backend。
- 普通 module-level `mla_context_module_perf.txt` / `mla_generation_module_perf.txt` 在 H100 0.5.9 数据目录里不存在。
- 也就是说，H100 上目前实际可用的是：
  - kernel 级 MLA 表
  - WideEP 兼容的整体 MLA 表
  - 但**普通 module-level MLA 表缺失**

## 5. 具体采集了哪些情景

### 5.1 `collect_mla.py`

context：

- `batch_size`: 1..256
- `seq_len`: 1..32768
- `num_heads`: 64 / 128
- `tp_size`: 1..64
- `kv_cache_dtype`: bf16 / fp8
- `backend`: 由 SM 自动选

generation：

- `batch_size`: 1..1024
- `kv_cache_len`: 1..131072
- `num_heads`: 64 / 128
- `tp_size`: 1..64
- `kv_cache_dtype`: bf16 / fp8
- `backend`: 由 SM 自动选

### 5.2 `collect_mla_module.py`

普通 module / DSA：

- `batch_size`
- `seq_len`
- `num_heads`
- `kv_cache_dtype`
- `compute_dtype=bfloat16`
- `gemm_type=bfloat16`
- `model_path`
- `attn_type=dsa`
- `attention_backend=nsa`

WideEP MLA：

- `batch_size`
- `seq_len`
- `num_heads`
- `kv_cache_dtype=fp8`
- `compute_dtype=bfloat16`
- `gemm_type=bfloat16`
- `model_path=deepseek-ai/DeepSeek-V3`
- `attn_type=mla`
- `attention_backend=flashinfer/fa3/trtllm_mla`

## 6. 总结

可以把这两个脚本简单理解为：

- `collect_mla.py` = **单 kernel / 子算子**
- `collect_mla_module.py` = **完整 attention module / forward pipeline**

在 H100 / SGLang 0.5.9 上：

- kernel 级 MLA 表是存在的，且主要是 `flash_attention` 路径。
- WideEP MLA 的兼容整体表也是存在的，且是 `fp8_block + fp8` 路径。
- 普通 module-level MLA 表缺失，因此 SDK 若要求 `MLAModule` 主表，会回退到更细粒度的 granular ops。

换句话说：

- `collect_mla.py` 更适合对齐 SDK 的 granular attention / MLA kernel 查表。
- `collect_mla_module.py` 更适合对齐完整 attention forward 的模块级时延，尤其是 WideEP / deepseek 特殊路径。

## 7. 量化细节补充

### 7.1 `collect_mla.py` 里的 FP8 不是“attention 核心全 FP8”

从脚本实现看，`collect_mla.py` 的 MLA kernel 路径里：

- q / k / v 的主输入都是 BF16 张量构造。
- `kv_cache_dtype=fp8` 只表示 KV cache 的存储 / 读取精度是 FP8。
- 脚本本身没有单独的“FP8 KV cache 反量化算子”。

因此更准确地说：

- **attention 核心计算在 collector 里仍以 BF16 路径为主**
- **FP8 主要体现在 KV cache 存储/读取语义**
- 如果 backend 内部需要把 FP8 cache 变成 BF16 参与计算，那也是 SGLang backend / kernel 内部做的，不是 collector 代码里显式写出的独立算子

这也和当前 H100 数据一致：

- `context_mla_perf.txt` / `generation_mla_perf.txt` 的 `mla_dtype` 都是 `bfloat16`
- `kv_cache_dtype` 只有 `bfloat16/fp8`

### 7.2 `collect_mla_module.py` 的 WideEP 不是“代码层面全 FP8 跑一遍”

WideEP MLA 分支里有一个很关键的实现细节：

- `_build_wideep_mla_test_cases()` 只给出一个 baseline 组合：`bfloat16, bfloat16, bfloat16`
- `load_model_runner()` 在 `gemm_type != fp8_block` 时会关闭 `server_args.quantization`
- 但 `run_attention_torch()` 在 `attn_type == "mla"` 时，会把日志字段**强行写成**
  - `mla_dtype=fp8_block`
  - `kv_cache_dtype=fp8`
  - `gemm_type=fp8_block`

所以这里要区分：

- **运行侧**：脚本更像是在跑一个 bf16 baseline 的 WideEP 模块 benchmark
- **记录侧**：它按旧 WideEP 兼容协议把结果写成 `fp8_block/fp8`

因此，`wideep_context_mla_perf.txt` / `wideep_generation_mla_perf.txt` 的字段更像“兼容型标签”，不能仅凭 CSV 值就断言 collector 当时所有 attention 核心都真的以 FP8 执行。

更稳妥的理解是：

- 这套表是给 SDK 的 WideEP MLA 路径做**旧协议对齐**
- 真正的查表轴和后端语义是 `WideEPContextMLA` / `WideEPGenerationMLA` 的 `kernel_source + kv_cache_dtype + fmha_quant_mode (+ tp_size)` 组合
- 它和普通 `context_mla_perf/generation_mla_perf` 不是同一层级的数据
