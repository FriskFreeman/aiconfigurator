# collector 脚本对 Triton 回退路径的静态分析

本文基于 `aiconfigurator-dsv4-sjy` 的 collector 代码，以及本地已 checkout 的 SGLang PR 分支 `/home/ai_lab/ljc/sglang`，分析 RTX PRO 6000 / SM120 上 Triton fallback 相关采集路径是否存在测量偏差风险。

## 总体结论

当前 collector 对多数 kernel/module 仍在努力排除首次 JIT、autotune、graph capture 等非稳态开销，但 DeepSeek-V4 sparse HCA / SM120 FlashMLA Triton fallback 路径存在几个静态风险：

- `deepseekv4_sparse_modules.py` 中探测和导入的 SM120 FlashMLA fallback 模块名是 `flash_mla_sm120_fallback`，而本地 SGLang PR 分支实际提供的是 `flash_mla_sm120.py` 和 `flash_mla_sm120_triton.py`。这会导致 collector 的支持性判断和实际调用路径不一致。
- `_bench_flash_mla_sparse()` 在 SM120 分支也导入 `flash_mla_sm120_fallback.flash_mla_with_kvcache_entrypoint`，与本地 PR 分支的实际入口 `flash_mla_sm120.flash_mla_with_kvcache_sm120` 不匹配。若运行环境确实是这个 PR 分支，collector 的 HCA sparse Triton 采集不是“正常测慢”，而是很可能根本没有测到该 Triton fallback。
- 若实际环境里另有 `flash_mla_sm120_fallback` 兼容模块，则 collector 会用 `benchmark_with_power(..., use_cuda_graph=True)` 包裹 Triton kernel。首次 JIT/autotune 通常会在 warmup 或 graph capture 前被触发，但 Triton `@autotune`、shape-specialized compile、以及多 kernel wrapper 的首次路径是否完全被 warmup 吸收，需要实测验证；当前代码没有显式记录 compile/cache 命中状态。
- 对 module-level attention 采集，`collect_dsv4_flash_attn.py` 测的是完整 `self_attn(...)`，不是单个 Triton kernel。若把它与 FA/FlashMLA 单 kernel数据直接比较，出现 20 倍差距并不能直接说明 Triton kernel 本身慢 20 倍。
- SM120 的 SGLang fallback 实现本身不是 FlashMLA CUDA kernel 的等价高性能替代。`flash_mla_sm120_triton.py` 的实现是 tiled sparse decode Triton kernel，并且 main cache 和 extra cache 分别运行、再做 LSE merge；这天然可能比专用 FlashMLA CUDA kernel 多 kernel launch 和额外内存读写。

因此，“Triton 算子性能比 FA 慢 20 多倍”不能只归因为 collector 把编译时间算进去了。静态分析看，至少有三类可能混在一起：

1. 采集入口没有命中预期 Triton fallback，或者命中了 PyTorch fallback。
2. 采的是完整 module 或多 kernel fallback，不是单一 FA/FlashMLA kernel。
3. Triton fallback 实现本身与 CUDA FlashMLA kernel 算法边界不同，确实可能显著慢。

## 相关 collector 路径

### 1. 通用 attention collector

文件：`collector/sglang/collect_attn.py`

该 collector 构造 mock `ModelRunner`、`RadixAttention`、KV pool 和 `ForwardBatch`，然后调用：

```python
layer(q, k, v, forward_batch)
```

在 SM120 上，它会强制选择：

```python
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
attn_backend = TritonAttnBackend(model_runner)
attn_backend_name = "triton"
```

这里测的是 SGLang 通用 MHA/GQA attention backend，不是 DeepSeek-V4 FlashMLA fallback。它使用 `benchmark_with_power()`，默认会：

- 先 eager warmup。
- 尝试 CUDA Graph capture。
- 再 replay CUDA Graph 计时。

这条路径整体比较接近“纯 GPU 稳态执行时间”，但只适用于通用 attention，不覆盖 DeepSeek-V4 的 compressed attention module。

### 2. DeepSeek-V4 attention module collector

文件：`collector/sglang/collect_dsv4_flash_attn.py`

该 collector 构建 SGLang `ModelRunner`，取出：

```python
attention_module = model_runner.model.model.layers[layer_id].self_attn
```

并计时：

```python
attention_module(x=hidden_states, positions=positions, forward_batch=forward_batch)
```

它采的是完整 self-attn module forward，边界包含：

- Q/KV projection。
- norm / rope。
- cache store。
- compressor。
- CSA indexer/topk。
- final FlashMLA 或 SM120 fallback。
- 输出相关路径。

这不是单一 Triton kernel benchmark。若该 collector 在 SM120 上走了 SGLang 的 `flash_mla_sm120`，测到的是完整 V4 attention module 中包含 Triton fallback 的端到端耗时。

另外，该文件显式设置：

```python
os.environ["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] = "0"
server_args.enable_piecewise_cuda_graph = False
```

这能避免大量不相关的模型级预编译和 piecewise CUDA graph 预热开销，但也意味着首次触发某个 Triton/JIT template 的成本要依赖 collector 自身 warmup 是否足够覆盖。

### 3. DeepSeek-V4 sparse kernel collector

文件：`collector/sglang/deepseekv4_sparse_modules.py`

这是和本次 Triton fallback 问题最相关的文件。它采两个 kernel-level 数据：

- `paged_mqa_logits`
- `hca_attn`

其中 `hca_attn` 在非 SM120 上调用：

```python
from flash_mla import flash_mla_with_kvcache, get_mla_metadata
```

而在 SM120 上，collector 当前写的是：

```python
from sglang.srt.layers.attention.flash_mla_sm120_fallback import flash_mla_with_kvcache_entrypoint
flash_mla_with_kvcache = lambda **kwargs: flash_mla_with_kvcache_entrypoint(backend="kernel", **kwargs)
```

但本地 SGLang PR 分支实际路径是：

- `python/sglang/srt/layers/attention/flash_mla_sm120.py`
- `python/sglang/srt/layers/attention/flash_mla_sm120_triton.py`

实际入口是：

```python
from sglang.srt.layers.attention.flash_mla_sm120 import flash_mla_with_kvcache_sm120
```

SGLang 的 DeepSeek-V4 backend 也是这样调用的：

```python
if _is_sm120:
    from sglang.srt.layers.attention.flash_mla_sm120 import flash_mla_with_kvcache_sm120
    o = flash_mla_with_kvcache_sm120(...)[0]
```

所以当前 collector 与本地 PR 分支存在 API 命名错配。

## SGLang PR 分支中的 SM120 Triton fallback

### FlashMLA fallback

本地 SGLang 中 `flash_mla_sm120.py` 的说明很明确：

- SM120 上常规 `flash_mla` CUDA kernel 不可用。
- 默认使用 fused Triton kernel。
- 通过 `SGLANG_SM120_TRITON_FLASHMLA=1` 开启 Triton，`0` 使用 PyTorch fallback。

入口逻辑：

```python
_sm120_default_backend = (
    "triton" if os.environ.get("SGLANG_SM120_TRITON_FLASHMLA", "1") == "1" else "torch"
)
```

Triton 实现在 `flash_mla_sm120_triton.py`，核心为：

- `_tiled_sparse_decode_kernel`：Triton JIT kernel。
- `flash_mla_sparse_decode_triton()`：分别处理 main cache 和 extra cache。
- `_merge_partial_attn()`：用 LSE 合并 main/extra 两段 attention 输出。
- `_apply_attn_sink()`：额外处理 attention sink。

这说明它不是直接复刻 FlashMLA CUDA kernel 的一个单 kernel 替代，而是一个 Python wrapper + 一个或多个 Triton kernel + torch 后处理组合。

静态上看，和原生 FlashMLA 相比它至少可能有这些额外成本：

- main cache 和 extra cache 分开执行，可能产生两次 Triton kernel launch。
- LSE merge 和 attention sink 是额外 torch op。
- KV cache 以多个 typed views 解释并进行 page-aware gather，内存访问模式与 CUDA FlashMLA 专用 kernel 不同。
- `@triton.autotune` 会按 `topk_rounded` 选择配置，首次遇到新 topk bucket 时可能触发编译或选择开销。

因此，如果将它和 H/B 卡上的专用 FlashMLA CUDA kernel做同粒度比较，性能差距显著并不必然说明 collector 错误。

### paged_mqa_logits fallback

本地 SGLang 中 `dsv4/indexer.py` 提供多个 fallback：

- `fp8_paged_mqa_logits_torch`
- `fp8_paged_mqa_logits_torch_sm120`
- 可选 `tilelang_fp8_paged_mqa_logits`

collector 中 `_import_dsv4_sm120_paged_mqa_impl()` 的逻辑是：

```python
if envs.SGLANG_OPT_USE_TILELANG_INDEXER.get():
    return ..., tilelang_fp8_paged_mqa_logits
return ..., fp8_paged_mqa_logits_torch
```

这里没有直接使用 `fp8_paged_mqa_logits_torch_sm120`。如果本地分支期望 SM120 用 `fp8_paged_mqa_logits_torch_sm120` 或 TileLang，则 collector 当前默认可能没有走到最适合 SM120 的实现。

此外，`paged_mqa_logits` 的 collector 将 `M=bs*isl` 展平成 `b=M,next_n=1`，这是为了适配 SM90 kernel shared memory 限制。这个 workload 映射是否与 SM120 fallback 的最优 batch/sequence 形态一致，需要单独验证。

## 测量方法分析

### `benchmark_with_power()` 的默认流程

文件：`collector/helper.py`

默认流程是：

1. `num_warmups` 次 eager warmup。
2. CUDA graph capture，一次 capture 中执行 `repeat_n` 次 `kernel_func()`。
3. capture 后再 replay warmup。
4. 计时阶段只 replay graph，或在 graph 不启用时执行 eager。

这对普通 CUDA kernel 很合理，可以排除大多数 launch overhead 和 Python overhead。

对于 Triton，需要注意：

- Triton 第一次调用某个 shape/config 时会 JIT compile。
- `@triton.autotune` 第一次遇到 key 时可能编译多个 config 或跑测候选 config。
- 如果首次 JIT/autotune 完全发生在 step 1 的 eager warmup，则计时不包含编译开销。
- 如果某些 lazy 初始化延迟到 graph capture 或计时阶段，collector 可能失败或污染结果。

`deepseekv4_sparse_modules.py` 的 `_bench_cuda_graph()` 传入：

```python
allow_graph_fail=False
use_cuda_graph=True
```

因此如果 Triton fallback 不支持 graph capture 或在 capture 中触发不允许的操作，应该直接报错，而不是自动退到 eager。这个设计能避免“悄悄测到 Python/eager 开销”，但也意味着一些 fallback 路径无法采集。

### module collector 的差异

`collect_mla_module.py` 的 DSA decode 路径专门加入了额外 eager pre-warm：

```python
for _ in range(5):
    kernel_func()
    torch.cuda.synchronize()
```

注释说明这是为了把 Blackwell 上 DeepGEMM / flashinfer 的 JIT 和 autotune 提前冲掉，避免进入 CUDA graph capture。

但 `deepseekv4_sparse_modules.py` 对 SM120 FlashMLA Triton fallback 没有类似的“按 fallback 特性追加 prewarm”。它只依赖 `benchmark_with_power()` 默认 warmup。对于 `@triton.autotune` 的多 config 编译路径，默认 5 次 warmup一般可能够，但没有静态保证。

## 发现的问题与风险等级

### P0：SM120 FlashMLA fallback 模块名不匹配

collector 期望：

```text
sglang.srt.layers.attention.flash_mla_sm120_fallback
```

本地 SGLang PR 分支提供：

```text
sglang.srt.layers.attention.flash_mla_sm120
sglang.srt.layers.attention.flash_mla_sm120_triton
```

影响：

- `_dsv4_sparse_kernel_support_status("hca_attn")` 在 SM120 上可能错误返回 unsupported。
- 若用户设置 `COLLECTOR_FORCE_DSV4_FLASH_SPARSE=1` 强制运行，`_bench_flash_mla_sparse()` 仍可能因 import 路径错误失败。
- 如果运行环境中另有不同命名的兼容模块，则本地 PR 静态分析结论和容器行为会分叉。

建议：

- collector 支持两个入口名，优先使用官方/当前 PR 的 `flash_mla_sm120.flash_mla_with_kvcache_sm120`。
- 保留旧 `flash_mla_sm120_fallback` 作为兼容 fallback。
- 日志中记录实际使用的 backend：`sm120_triton` / `sm120_torch` / `flash_mla_cuda`。

### P1：SM120 HCA sparse collector 可能误测 PyTorch fallback

SGLang 中 `SGLANG_SM120_TRITON_FLASHMLA=0` 会选择 PyTorch fallback。collector 当前 CSV 写入的 `kernel_source` 固定是：

```text
flash_mla_with_kvcache
```

没有记录是否实际使用 Triton 还是 torch。

影响：

- 如果环境变量被设置为 `0`，会把 PyTorch fallback 的极慢结果写成 FlashMLA/HCA 数据。
- 这类数据很容易表现为比 FA 慢十几倍或几十倍。

建议：

- collector 在 SM120 上读取 `SGLANG_SM120_TRITON_FLASHMLA` 并写入更细 kernel source，例如 `flash_mla_sm120_triton` 或 `flash_mla_sm120_torch`。
- 默认拒绝采集 torch fallback，除非显式设置 `COLLECTOR_ALLOW_TORCH_FALLBACK=1`。

### P1：Triton autotune/JIT 没有显式可观测性

当前 `_bench_cuda_graph()` 的 warmup 大概率能排除首次编译，但报告中无法证明这一点。

风险：

- 首个 shape 可能特别慢，但如果 JIT 发生在 warmup，不会污染计时。
- 若 autotune 在不同 `topk_rounded` key 下反复触发，某些 shape 的第一次采集可能失败或不稳定。
- 如果 graph capture 前的 warmup不足，capture 可能包含不允许的操作而失败。

建议：

- 对 Triton fallback 增加专门 prewarm 阶段：至少 1 次 dry-run compile，1 次 sync，若使用 autotune则按 `topk_rounded` 预热。
- 在日志中记录 `used_cuda_graph`、`num_warmup`、`repeat_n`、以及是否 SM120 Triton。
- 可增加一个 debug 模式：同一 shape 连续测两次，比较第一次和第二次结果，识别 JIT 残留。

### P1：DeepSeek-V4 module 数据不应和单 kernel FA 数据直接比较

`collect_dsv4_flash_attn.py` 计时完整 module forward，包含 projection、compressor、indexer/topk、cache store 和 attention。常规 FA/FlashMLA kernel benchmark往往只包含 attention kernel。

影响：

- 端到端 module 对单 kernel 比较可能天然大很多。
- 尤其 SM120 fallback 可能替换的是 module 内某一段，但 collector 报告的是完整 module latency。

建议：

- 比较 Triton fallback 与 FA 时，优先使用 `deepseekv4_sparse_modules.py` 的 `hca_attn` kernel-level 数据。
- module-level 数据应只和同边界的 module-level 数据比较。

### P2：paged_mqa_logits SM120 默认路径可能不是最优 fallback

collector 默认返回 `fp8_paged_mqa_logits_torch`，只有设置 `SGLANG_OPT_USE_TILELANG_INDEXER` 时才用 TileLang。SGLang PR 分支还存在 `fp8_paged_mqa_logits_torch_sm120`。

影响：

- SM120 上 CSA indexer scoring 的 sparse correction 数据可能来自非最优实现。
- 结果可能无法代表 Triton/TileLang fallback 性能。

建议：

- 明确 SM120 默认策略：torch、torch_sm120、TileLang 三者只保留一个作为默认高保真路径。
- CSV 中写入实际 kernel source，例如 `paged_mqa_logits_torch_sm120` / `tilelang_fp8_paged_mqa_logits`。

### P2：MoE Triton 配置在 SM120 上大量使用 hard-coded skip

`collect_moe.py` 中针对 SM120 fp8_block MoE 有大量 shape skip，原因是默认 Triton config 需要 144 KiB shared memory，超过 RTX PRO 6000 的约 99 KiB 限制。

这说明 collector 已经知道一些 SGLang Triton 默认配置在 SM120 上不适配。若用户看到 MoE/Triton 数据断档或只覆盖小 shape，这不是采集脚本偶然问题，而是当前运行时配置覆盖不足。

建议：

- 不要把 skipped shape 插值成大 shape 的高置信 silicon 数据。
- 对 RTX PRO 6000 建立专门 tuned config 集合，避免默认 config 导致大范围不可运行或性能异常。

## 对“慢 20 倍”的判断

静态分析无法直接证明某次测量是否包含 JIT 编译时间，但可以给出优先排查顺序：

1. 确认 CSV 的 `kernel_source` 和日志中实际 backend。若无法区分 Triton/torch fallback，则结果可信度不足。
2. 确认 collector 是否用到了本地 PR 分支的实际入口 `flash_mla_sm120.flash_mla_with_kvcache_sm120`。当前代码静态上不匹配。
3. 用同一 shape 连续运行两次 `hca_attn` collector，比较第一次和第二次。若第二次显著快，说明存在 JIT/autotune 残留。
4. 对比 kernel-level `hca_attn`，不要用完整 `dsv4_*_hca_*_module` 直接对比 FA kernel。
5. 检查环境变量 `SGLANG_SM120_TRITON_FLASHMLA` 是否为 `1`。若为 `0`，慢 20 倍是合理的 PyTorch fallback 表现。

## 修改建议

### 最小修复

1. 修正 SM120 FlashMLA fallback import：

```python
try:
    from sglang.srt.layers.attention.flash_mla_sm120 import flash_mla_with_kvcache_sm120
except ImportError:
    from sglang.srt.layers.attention.flash_mla_sm120_fallback import flash_mla_with_kvcache_entrypoint
```

2. `_has_sglang_sm120_flash_mla_impl()` 同时探测两个模块名。

3. CSV `kernel_source` 加上实际后端：

```text
flash_mla_sm120_triton
flash_mla_sm120_torch
flash_mla_cuda
```

4. SM120 下若检测到 torch fallback，默认 fail-fast，不写入 silicon 数据。

### 测量增强

1. 给 Triton fallback 增加显式 precompile/prewarm：

```python
for _ in range(extra_triton_warmup):
    kernel_fn()
    torch.cuda.synchronize()
```

2. 在 debug 模式下同 shape 重复采两轮，并输出：

- first run latency
- second run latency
- ratio
- used_cuda_graph

3. 对 `@triton.autotune` key，例如 `topk_rounded`，在正式计时前按 key 预热。

### 数据质量增强

1. 在 perf row 中增加或复用字段区分：

- `kernel_source`
- `backend_impl`
- `used_cuda_graph`
- `fallback_type`

2. 对 fallback 数据建立数据质量标签：

- `silicon`
- `triton_fallback`
- `torch_fallback_low_fidelity`

3. 在 perf database 加载时对 `torch_fallback_low_fidelity` 默认 warning 或拒绝，避免误用于高置信仿真。

## 后续验证建议

建议先做三个小实验，不需要大改代码：

1. 在 SM120 环境中运行 Python import 检查：

```python
import importlib.util
print(importlib.util.find_spec("sglang.srt.layers.attention.flash_mla_sm120"))
print(importlib.util.find_spec("sglang.srt.layers.attention.flash_mla_sm120_fallback"))
```

2. 对单个 `hca_attn` shape 连续跑两次，并确保 `SGLANG_SM120_TRITON_FLASHMLA=1`：

```bash
SGLANG_SM120_TRITON_FLASHMLA=1 \
python collector/sglang/deepseekv4_sparse_modules.py \
  --kernel hca_attn \
  --bs-list 1 \
  --isl-list 1024 \
  --past-kv-list 8192 \
  --model-path sgl-project/DeepSeek-V4-Pro-FP8
```

3. 再用 `SGLANG_SM120_TRITON_FLASHMLA=0` 跑同一 shape。若 `0` 明显慢很多，说明慢 20 倍可能是 torch fallback 或未区分 fallback source 导致。

## 最终判断

collector 的测量框架本身并不是简单地把 Triton 编译时间直接计入 latency；它默认有 warmup 和 CUDA Graph replay，理论上会排除大多数 Python launch 和首次编译开销。

但 DeepSeek-V4 SM120 Triton fallback 路径当前存在明确的静态问题：collector 的 fallback 模块名与本地 SGLang PR 分支不一致，且 perf 输出没有区分 Triton fallback 和 torch fallback。再叠加 module-level 与 kernel-level 边界混用，足以造成“看起来 Triton 比 FA 慢 20 倍”的误判。

因此，建议优先修复 backend 探测/记录和单 shape 重复采集验证，再判断 Triton kernel 本身是否真的性能异常。
