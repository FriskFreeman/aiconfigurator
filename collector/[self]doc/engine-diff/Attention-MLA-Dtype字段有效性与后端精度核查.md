# Attention / MLA dtype 字段有效性与后端精度核查

> **文档状态：历史初版，不再作为最终结论入口。**
>
> 本文包含后来明确排除的 MLA module，且没有纳入按 `stage × engine` 展开的 dtype 组合表、引擎独立图、P10/P50/P90及small/large规模分析。
>
> 最新且唯一推荐的分析入口是：[`attention-mla-dtype-analysis/Attention与基础MLA-Dtype全链路分析.md`](attention-mla-dtype-analysis/Attention与基础MLA-Dtype全链路分析.md)。工作脚本、12组引擎独立图和机器可读统计均保存在同一工作目录中。
>
> 核查日期：2026-07-27
>
> 范围：AIC 当前工作树中的 SGLang、TensorRT-LLM、vLLM CUDA/XPU attention collector，SGLang/TRT-LLM 基础 MLA collector，三端 MLA module collector，已有性能 CSV，以及 SDK loader、Operation 和 SOL/SILICON 查询逻辑。
>
> 本文区分普通 attention、基础 MLA kernel collector、完整 MLA module collector；三者的 `attn_dtype` / `mla_dtype` 不能混为一个语义。

## 1. 总结论

你的核心观察对以下范围基本成立：

- SGLang 基础 `collect_mla.py` 的 `mla_dtype` 固定记录为 BF16，不是一个传给 backend 的独立控制参数；实际变化轴是 `kv_cache_dtype` 和硬件选择出的 FA3 / TRTLLM MLA / Triton backend。
- 普通 attention 和 MLA 的 generation/decode 数据没有独立 FMHA dtype 轴；collector 输入通常是 BF16，而 FP8 cache 会触发 cache 写入量化、Q 对齐或专用 decode kernel。
- SGLang FA3/FlashMLA 的若干路径确实优先尝试把 Q（继而 K/V）向显式 KV cache dtype 对齐；不支持时，一些路径会把 cache materialize/cast 回 Q dtype。

但四项论断不能推广到所有 phase、框架和硬件。最终判定如下：

| 论断 | 判定 | 最重要的反例/限定 |
|---|---|---|
| 1. `attn_dtype` / `mla_dtype` 都是冗余字段，不影响实际算子精度 | **整体不成立，只在部分 collector 成立** | SGLang/TRT-LLM context `attn_dtype` 是真实控制项；vLLM MLA module 的 `mla_dtype=fp8` 会打开 prefill Q/K/V quantization。基础 MLA 与 generation 才基本是冗余/固定轴。 |
| 2. 字段固定 BF16 决定初始 QKV 为 BF16，并在 KV dtype 不同时于计时内量化 | **因果表述不成立，现象部分成立** | 多数 collector 是代码直接把输入硬编码为 BF16，再把同一事实写入字段；字段本身没有反向控制输入。TRT context 即使记录 `attn_dtype=fp8`，初始 `input_qkv` 仍是 BF16；SGLang显式 FP8 context 则在计时前把 Q/K/V 转为 FP8。 |
| 3. 实际框架中只有 `kv_cache_dtype` 真正决定 attention 量化格式 | **KV dtype 很重要，但不是唯一决定项** | 它确定 cache 存储格式并经常影响 backend/kernel；但 activation/model dtype、独立 context-FMHA quant config、backend 能力和硬件同样决定 Q dtype、主 kernel dtype及 fallback。vLLM FlashInfer 可接受 BF16 Q + FP8 cache，并非所有后端都把“全部 QKV”统一成 cache dtype。 |
| 4. SDK 中也只有 KV dtype 有效，FMHA/MLA dtype 完全冗余 | **明确不成立** | Context attention、context MLA 和 MLA module 的 SILICON 表都用 FMHA dtype 作为独立字典键；context SOL 也仍可能受其影响。只有普通 generation attention / 基础 generation MLA 的查询接口没有独立 FMHA dtype 轴。 |

一句话概括：**`kv_cache_dtype` 是 cache 物理格式和许多后端分派的权威字段，但它不是跨框架、跨 phase 的“attention 实际计算 dtype 唯一真相”；`attn_dtype` / `mla_dtype` 当前既有真实控制项、只读标签、历史兼容假标签，也有 SDK 独立查表键，语义并不统一。**

## 2. 必须分开的四种 dtype

当前争议的根源是同一个“attention dtype”名称混合了四层概念：

1. **模型/activation dtype**：Q/K/V 在进入 attention wrapper 前的 dtype，通常是 BF16。
2. **KV cache storage dtype**：paged cache 的物理存储格式，如 BF16、FP8 E4M3。
3. **kernel operand dtype**：核心 FMHA/XQA/MLA kernel 实际读取的 Q/K/V 类型；可能在 wrapper 内量化，也可能支持 BF16 Q + FP8 cache 的混合接口。
4. **accumulation/output dtype**：softmax、累加和输出的 dtype，通常不能仅从 Q/cache dtype 推断。

AIC CSV 的 `attn_dtype` / `mla_dtype` 有时表示第 1 层，有时意图表示第 3 层，有时只是 SDK schema key。`kv_cache_dtype` 主要准确描述第 2 层，同时可能影响第 3 层，但不自动决定第 1 和第 4 层。

建议不要再使用笼统的“实际 attention 精度”描述，而要明确是 input、cache、kernel operand 还是 accumulation。

## 3. 普通 Attention Collector

### 3.1 SGLang

`collector/sglang/collect_attn.py` 有两个独立布尔控制：

- `use_fp8_kv_cache`：选择 `MHATokenToKVPool.dtype` 和 `model_runner.kv_cache_dtype`。
- `use_fp8_context_fmha`：要求 FP8 KV，并在 context phase 将 Q/K/V 显式 `.to(float8_e4m3fn)`。

其三种 context case 是：

| CSV 标签 | Q/K/V 进入计时闭包前 | cache | 字段效力 |
|---|---|---|---|
| BF16/BF16 | BF16 | BF16 | 基线 |
| BF16/FP8 | BF16 | FP8 | `attn_dtype` 是输入模式标签；backend 内可能按 KV dtype 量化/对齐 |
| FP8/FP8 | 已在计时前转 FP8 | FP8 | `attn_dtype` 对应真实、不同的 collector 执行路径 |

因此，论断 1 对 SGLang context attention **不成立**。`attn_dtype=fp8` 不是只改日志，它由 `use_fp8_context_fmha` 产生，并对应 Q/K/V 的实际预转换。

但有一个重要计时边界：显式 `.to(FP8)` 位于 `run_iter()` 和 `benchmark_with_power()` 之前，所以该 FP8 context case **不包含显式 Q/K/V cast 成本**。反而 BF16-tag + FP8-cache case 以 BF16 进入 `layer(...)`，backend 内发生的 dtype 对齐会被计时。

Generation 只生成 `attn_dtype=bfloat16`，没有独立 FP8 FMHA case。此时 `attn_dtype` 对查分支没有贡献，KV dtype 才是变化轴。

### 3.2 TensorRT-LLM

TRT-LLM context 的 `use_fp8_context_fmha` 同样不是纯日志字段：

```text
False -> quant_algo=None
True  -> QuantConfig(quant_algo=FP8), out_scale=[1.0]
```

KV cache 则由另一项控制：

```text
use_fp8_kv_cache -> KVCacheManager(dtype=FP8)
                 -> kv_cache_quant_algo=FP8
```

所以 `attn_dtype` 与 `kv_cache_dtype` 在 TRT-LLM context 中是两个真实 runtime 配置，尽管当前 case 约束只允许 FP8 FMHA 与 FP8 KV 组合。

**着重澄清：TRT-LLM collector 无论日志最终写 BF16 还是 FP8，最初构造的 `q`、`kv` 和拼接后的 `input_qkv` 都是 BF16。** `attn_dtype=fp8` 不表示进入 timer 前的 tensor 已经是 FP8；FP8 quant config、cache update、可能的内部 quant 和 FMHA 都封装在被计时的 `attn.forward()` 中。这直接否定了“字段值决定初始 QKV dtype”的全局表述。

Generation 打开 XQA JIT，但不生成独立 FP8 compute case；只有 KV dtype 变化。TRT-LLM v1.3 的 MLA generation 在 FP8 KV 时还显式准备 `quant_q_buffer`，并把 `mla_rope_generation()` 放在 benchmark closure 内，说明 Q quant/preprocess 确实可能属于被测路径。

### 3.3 vLLM CUDA

普通 `collector/vllm/collect_attn.py` 的行为更接近你的判断：

- `dtype = torch.bfloat16` 固定。
- Q/K/V 始终以 BF16 创建并传给 `impl.forward()`。
- `attn_dtype` 固定写 `bfloat16`，没有控制分支。
- `kv_cache_dtype` 被传给 backend selector、cache spec、cache tensor 和 implementation。

所以在这个 collector 中，`attn_dtype` 确实是冗余标签；但“实际 kernel 一定全部对齐到 KV dtype”仍不能跨 backend 成立。

可用的 vLLM backend 源码显示：

- FlashAttention backend 在 FP8 cache 时，会先用 cache op 将 BF16 K/V 写入 FP8 cache，并在 forward 内对 Q 调 `scaled_fp8_quant()`，然后进入支持 FP8 的 FA kernel。这一支和 SGLang 的“Q 向 cache dtype 对齐”相似。
- Triton attention backend 的 CUDA 分支也会在 FP8 cache 时量化 Q；ROCm 分支明确跳过 Q quant，因为对应 kernel 不支持。
- FlashInfer wrapper 会把 cache reinterpret 为 FP8，但 Python wrapper 没有统一执行同样的 Q `.to(FP8)`；接口仍可以接收 BF16 query。内部 TRT-LLM/FlashInfer kernel是否再融合处理取决于版本和 shape。

现有数据也证明 KV dtype 会影响 backend 选择，而不只是 dtype 模板：

| 平台/现有 vLLM 数据 | BF16 KV | FP8 KV |
|---|---|---|
| L40S 0.14.0 | `vllm_flash_attn` | `vllm_flashinfer` |
| H100/H200 0.19.0 | `vllm_flash_attn` | `vllm_flash_attn` |
| B200/B300/GB200/GB300 0.19.0 | `vllm_flashinfer` | `vllm_flashinfer` |
| RTX PRO 6000 0.19.0 | `vllm_flash_attn` | `vllm_flashinfer` |

因此 KV dtype 有时改变 backend，有时只改变同一 backend 内的 cache/kernel dtype。

### 3.4 vLLM XPU

`collect_attn_xpu.py` 同样固定 BF16 Q/K/V 和 `attn_dtype=bfloat16`。源码注释明确指出 XPU FlashAttention 即使 KV cache 为 FP8，Query 与 Output 仍预期为 BF16；用于 CUDA FlashAttention/FlashInfer 的显式 query FP8 转换分支带有 `"xpu" not in device` 条件，在 XPU 上不会执行。

这是对“所有后端都会尽量把全部 QKV 对齐到 KV dtype”的直接硬件反例：至少 AIC 当前 vLLM XPU collector 明确保留 BF16 Q/output + FP8 cache 组合。

## 4. MLA Collector

### 4.1 基础 `context_mla_perf` / `generation_mla_perf`

基础 MLA collector 上，你的经验基本准确。

| 框架 | `mla_dtype` | 初始 activation | 真正变化轴 |
|---|---|---|---|
| SGLang `collect_mla.py` | 固定 BF16 | Q/V BF16；FP8 prefill 时 K 可在 timer 外预转 FP8 | KV dtype、SM/CUDA 选择的 backend |
| TRT-LLM `collect_mla_v1.py` | 固定 BF16 | BF16 | v1 只支持 BF16 KV |
| TRT-LLM `collect_mla_v2.py` | 固定 BF16 | `Scenario.dtype=BF16` | BF16/FP8 KV 与相应 quant config |
| 仓库已有 vLLM 基础 MLA 数据 | 固定 BF16 | 对应旧 collector 路径为 BF16 | KV dtype/backend |

仓库所有现有 `context_mla_perf.txt` 和 `generation_mla_perf.txt` 的 `mla_dtype` 唯一值都是 `bfloat16`，而多个框架同时存在 BF16/FP8 KV 行。对这两张基础表，把 `mla_dtype` 当作独立的实测轴没有数据支撑。

SGLang 基础 MLA 的硬件分派为：

- SM90 Hopper + CUDA>=12.3：FA3/FlashAttention prefill/MLA 路径。
- SM100/103 + CUDA>=12.8：TRTLLM MLA backend。
- 其他架构：Triton；当前 context collector 为避免语义不一致直接禁用 Triton context，Triton generation 只采 BF16 KV。
- SM<90：当前基础 SGLang MLA collector 不生成 case。

SGLang FA3 的已核实行为是：显式 FP8 KV 且 head dimension/backend 能力允许时，Q 向 KV dtype 对齐，K/V 再向 Q dtype 对齐，最后进入 FP8 FA；若不满足能力条件，某些 absorbed MLA 路径会把 FP8 cache `.to(q.dtype)` materialize 成 BF16。这与你描述的“先尝试 input->KV，失败后 cache->Q”一致，但它是 **SGLang 特定 backend 的条件行为**，不是 AIC 所有框架的统一规则。

### 4.2 完整 MLA module collector：关键反例

完整 module collector 不能沿用“`mla_dtype` 永远无效”的结论。

#### SGLang module

当前正式 module case 为降低模型加载成本，只采 `compute_dtype=bfloat16,kv=bfloat16,gemm=bfloat16`；完整精度组合函数虽然枚举 KV/GEMM 变化，compute dtype仍固定 BF16。这里 `mla_dtype` 基本是 schema 标签。

更严重的是旧 WideEP 兼容路径：源码明确写着“实际用 BF16 单精度运行，但日志固定记为 `mla_dtype=fp8_block,kv_cache_dtype=fp8,gemm_type=fp8_block`”。因此 WideEP 历史字段不仅可能冗余，还可能是为了旧 loader 兼容而保留的 **非真实执行标签**，绝不能用来反推 kernel dtype。

#### TensorRT-LLM module

当前 `_get_precision_combos()` 明确写“只支持 BF16 attention compute”，`compute_dtype` 固定 BF16；KV quant 由 `kv_cache_quant_algo` 控制，GEMM 精度由独立 `gemm_type` 控制。对该 collector，`mla_dtype` 是固定 schema 轴。

#### vLLM module

这是最明确的反例。vLLM v3 module collector 在 SM100+ 的 context case 会生成：

```text
(compute_dtype=fp8, kv_cache_dtype=fp8)
```

并执行：

```text
use_prefill_fp8 = compute_dtype == "fp8"
vllm_config.attention_config.use_prefill_query_quantization = True
```

源码注释明确说明这会“quantize Q/K/V to FP8 before sending to the prefill kernel”。因此这里的 `mla_dtype=fp8` 是真实控制项，不是冗余日志。现有 vLLM `mla_context_module_perf.txt` 也已经同时存在 BF16 和 FP8 `mla_dtype` 行。

Generation module 仍只生成 BF16 compute + BF16/FP8 KV，故 generation 中 `mla_dtype` 重新退化为固定维度。

## 5. `kv_cache_dtype` 在真实框架中的效力边界

`kv_cache_dtype` 在三个框架中都至少控制以下事项：

1. KV cache tensor/pool 的物理 dtype 或自定义布局。
2. 当前 K/V 写 cache 时是否需要量化和使用哪些 scale。
3. backend selector 或 backend 内部 kernel variant 的候选集合。
4. decode 的 cache 读取带宽与可能的 Q quant/dequant 路径。

但它不是唯一决定核心 attention kernel dtype的原因如下：

- **独立 compute knob**：TRT-LLM context 有 `quant_algo=FP8`；vLLM SM100+ MLA context 有 `use_prefill_query_quantization`。
- **activation dtype**：vLLM 标准 collector和 TRT-LLM collector都以 BF16 activation 进入 wrapper；是否量化由 backend 决定。
- **backend capability**：vLLM Cutlass MLA/Triton MLA 的部分版本明确拒绝 FP8 KV，而不是自动得到 FP8 MLA；SGLang Triton MLA 当前也只收 BF16 KV。
- **混合 dtype kernel**：FlashInfer 类接口可以保留 BF16 query并读取 FP8 cache；“cache 是 FP8”不等于“Q/K/V/输出/累加全为 FP8”。
- **硬件与版本**：相同配置在 L40S、Hopper、数据中心 Blackwell、SM120 会选择不同 backend，甚至直接 skip/crash filter。

所以更准确的规则是：

```text
effective kernel = f(
    phase,
    hardware/SM,
    framework version,
    selected backend,
    activation dtype,
    kv_cache_dtype,
    explicit context compute-quant config,
    shape/head_dim,
    fallback capability
)
```

不能简化成 `effective_attn_dtype = kv_cache_dtype`。

## 6. AIC SDK 中两个字段是否冗余

### 6.1 SILICON 查表

普通 context attention loader 的字典结构包含：

```text
data[attn_dtype][kv_cache_dtype][Hkv][D][window][H][S][B]
```

`query_context_attention()` 也使用调用方的 `fmha_quant_mode` 选择第一层。因此 `attn_dtype` 在 SILICON 模式下是明确的独立查表键。

基础 context MLA 同样是：

```text
data[mla_dtype][kv_cache_dtype][num_heads][S][B]
```

MLA module 的 context **和 generation** loader 都保留：

```text
data[mla_dtype][kv_cache_dtype][gemm_type][...shape...]
```

所以即使 collector 只产生 BF16，SDK schema 和查询仍把该字段当作有效轴；请求 FP8 会查另一个分支并可能报无数据。

只有两张基础 generation 表是例外：

- `generation_attention_perf` loader 读取 `attn_dtype` 但不把它放入字典 key。
- `generation_mla_perf` loader 读取 `mla_dtype` 但不把它放入字典 key。

对应 `query_generation_attention()` 和 `query_generation_mla()` 也没有 FMHA dtype 参数。对这两条基础 generation 路径，论断 4 成立。

### 6.2 SOL / EMPIRICAL

当前工作树新增了 `_effective_sglang_fmha_quant_mode()`：当 KV=FP8 且 FMHA=BF16 时，返回 FP8；否则保留传入的 FMHA mode。

这意味着：

- Context attention/context MLA：KV=FP8 时由 KV 覆盖 BF16 标签；KV=BF16 时，BF16/FP8 FMHA 仍会产生不同 compute/memory 估算。
- 普通 generation attention/MLA：计算吞吐由 KV dtype直接派生，没有 FMHA 参数。
- MLA module generation：SOL 由 KV dtype派生，但 SILICON 仍用 FMHA dtype选择表。

因此即使在当前 SOL 实现中，FMHA dtype也没有完全失效。

**需要着重指出一个设计风险：该 helper 名称和注释都写 SGLang，但它是 `staticmethod`，调用时没有检查 `self.backend`。当前代码会把同样的“FP8 KV -> FP8 core compute”规则应用到 TRT-LLM 和 vLLM PerfDatabase。** 这对 vLLM FlashInfer、XPU 或不支持 FP8 core 的 MLA backend 未必准确，应按 backend/phase/hardware 建模或从采集元数据读取 effective dtype。

### 6.3 Operation 层的额外影响

`ContextAttention.query()` 除查 attention 表外，还自行估算 RoPE 和 KV write；当前 KV write 字节数使用：

```text
k_num * fmha_quant_mode.memory
v_num * fmha_quant_mode.memory
```

而不是 `kvcache_quant_mode.memory`。所以 SDK 中 FMHA dtype甚至会额外改变 KV write latency。这与“KV cache dtype才决定 cache 写入格式”的语义不一致，属于需要修正或至少重新定义的地方：

- 若该项表示最终 cache 写流量，应使用 KV dtype。
- 若表示量化前输入读取，还需要同时计入 BF16 read、FP8 write和 quant kernel，不能只用 FMHA dtype单项替代。

模型配置层也明确保留独立 `fmha_quant_mode` 和 `kvcache_quant_mode`；部分 backend/model会把不支持的 FP8 FMHA强制回 BF16，例如 vLLM、DeepSeek V3/DSA 等。这再次说明 SDK 当前设计并没有把 FMHA dtype视为冗余字段。

## 7. 对四项论断的逐条修正版

### 论断 1 修正版

> 基础 MLA 和普通 generation 的 `mla_dtype` / `attn_dtype` 当前通常是固定 BF16 schema字段；但普通 context attention 和 vLLM SM100+ MLA context module存在真实、独立的 compute dtype控制，不能全局删除。

### 论断 2 修正版

> 多数 collector 的模型输入/hidden state硬编码为 BF16；日志字段通常反映这个输入模式，但不是输入 dtype的统一控制源。KV=FP8 时，cache写入量化一定属于 forward 语义；Q 是否在计时内量化、计时前量化或保持 BF16取决于框架/backend。

### 论断 3 修正版

> `kv_cache_dtype` 是实际推理中 cache存储、cache update和很多 decode/backend分派的关键配置；它经常间接决定核心 kernel operand dtype，但并非唯一决定因素，也不保证 Q/output/accumulation与 cache同 dtype。

### 论断 4 修正版

> SDK 的基础 generation attention/MLA只按 KV dtype查询；context和 module SILICON表仍按 FMHA dtype + KV dtype双轴查询。当前 SOL 正在向“FP8 KV驱动 effective dtype”靠拢，但实现是 SGLang假设的全局应用，尚不能代表所有 backend。

## 8. 建议的数据模型

不建议直接删除 `attn_dtype` / `mla_dtype`。更安全的是拆分并逐步迁移：

| 建议字段 | 含义 |
|---|---|
| `input_qkv_dtype` | 进入被计时 wrapper前的 Q/K/V dtype |
| `kv_cache_storage_dtype` | cache物理存储 dtype/格式 |
| `requested_compute_dtype` | collector/runtime请求的 FMHA dtype，可为空/auto |
| `effective_q_dtype` | 核心 kernel实际读取的 Q dtype |
| `effective_kv_dtype` | 核心 kernel实际读取的 K/V dtype，不等同于 storage dtype |
| `accumulation_dtype` / `output_dtype` | 累加和输出 dtype |
| `qkv_quant_in_timed_region` | Q/K/V量化是否在计时边界内 |
| `backend` / `kernel_symbol` | 实际 backend和 CUDA kernel，而非笼统 `torch_flow/default` |
| `dtype_fallback_reason` | 对齐失败、反量化或 backend fallback原因 |

迁移策略：

1. 基础 generation 表可正式废弃独立 FMHA key，但保留 legacy CSV 列用于兼容。
2. 基础 MLA 表若未来仍只有 BF16 input mode，可将 `mla_dtype` 明确重命名/定义为 `input_activation_dtype`，不要称为 effective compute dtype。
3. Context attention 保留 requested/effective compute轴，因为 SGLang/TRT-LLM已经存在真实 FP8 context路径。
4. vLLM MLA module 必须保留 compute dtype轴。
5. SDK SILICON 优先按 `effective_*` 或明确的 collector配置查表；SOL 的 effective dtype解析必须按 backend、phase、SM和 backend capability实现。
6. 修正 `ContextAttention` 的 KV write流量模型，避免用 FMHA dtype代替 cache dtype。

## 9. 证据边界

本次结论基于当前 AIC collector/SDK、仓库已有 CSV 后端标签、已有 SGLang 0.5.9/0.5.10 源码核查成果，以及本机可读取的 vLLM 0.10.1 backend源码；AIC vLLM module v3 中针对 0.19.0 的配置行为由 collector直接调用和注释确认。

当前环境没有可导入的 SGLang、TRT-LLM 或目标 vLLM 0.19.0 runtime，在线 GitHub查询也因 DNS失败而不可用。因此对于 TensorRT-LLM内部“量化是独立 kernel还是与 FMHA/cache update融合”以及 FlashInfer内部最终 operand模板，本文只陈述 Python调用边界和确定的配置效力，不虚构 CUDA symbol。跨版本最终确认仍应使用对应采集容器的 Nsight/CUPTI trace。

## 10. 关键源码索引

| 结论 | 当前工作树依据 |
|---|---|
| SGLang context 有独立 FP8 FMHA控制，并在 timer前转换 Q/K/V | `collector/sglang/collect_attn.py:293-302,491-494,530-531` |
| TRT context FP8 FMHA控制 `quant_algo`，但 input_qkv 初始为 BF16 | `collector/trtllm/collect_attn.py:91-112,225-233,274-290` |
| vLLM 普通 attention 固定 BF16 input/标签，KV dtype传给 selector/cache/backend | `collector/vllm/collect_attn.py:85,121-170,213-244,315,362-376` |
| vLLM XPU保持 BF16 Query/Output + FP8 cache | `collector/vllm/collect_attn_xpu.py:85,323-331,366-382` |
| SGLang 基础 MLA固定 `mla_dtype=bfloat16`，FP8 K可在 timer外生成 | `collector/sglang/collect_mla.py:556-585,709-735` |
| SGLang MLA backend随 SM/CUDA选择 | `collector/sglang/collect_mla.py:79-88,290-302,372-375,517-537` |
| TRT-LLM MLA v2 activation固定 BF16，FP8由 KV quant config控制 | `collector/trtllm/collect_mla_v2.py:121-137,315-355,620-641` |
| vLLM MLA module 的 FP8 compute会打开 prefill query quantization | `collector/vllm/collect_mla_module_v3.py:171-205,367-448,825-842` |
| SGLang WideEP兼容路径实际 BF16运行但写旧 FP8标签 | `collector/sglang/collect_mla_module.py:280-383,679-710` |
| Context attention/MLA loader把 FMHA dtype作为独立 key | `src/aiconfigurator/sdk/perf_database.py:895-967,1053-1111` |
| 基础 generation loader忽略 FMHA dtype key | `src/aiconfigurator/sdk/perf_database.py:975-1050,1114-1172` |
| MLA module context/generation均保留 FMHA dtype key | `src/aiconfigurator/sdk/perf_database.py:1751-1837` |
| SDK context SILICON按 FMHA+KV双轴查表 | `src/aiconfigurator/sdk/perf_database.py:4305-4458,4594-4696` |
| SDK generation基础查询只接收 KV dtype | `src/aiconfigurator/sdk/perf_database.py:4460-4591,4812-4896` |
| 当前 effective-SGLang helper没有 backend guard | `src/aiconfigurator/sdk/perf_database.py:4283-4302` |
| ContextAttention额外 KV write使用 FMHA dtype字节数 | `src/aiconfigurator/sdk/operations.py:930-948` |
| SDK配置本身保留独立 FMHA/KV轴并做 backend/model回退 | `src/aiconfigurator/sdk/config.py:13-23`、`src/aiconfigurator/sdk/models/helpers.py:128-180` |
