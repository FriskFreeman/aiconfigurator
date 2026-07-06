# SGLang WideEP MLA Module 量化语义分析

本文总结 `collector/sglang/collect_mla_module.py` 中 WideEP MLA module 级采集的量化语义问题，以及 SDK `PerfDatabase`/`Operation` 侧的查询行为。结论先行：当前 WideEP MLA module 表是历史兼容设计，实测执行与落盘量化标签存在明显语义偏差。

## 1. 现象与结论

当前 `collect_mla_module.py` 对 WideEP MLA module 级数据的处理是：

- 实际运行：只构造并执行 `bfloat16 / bfloat16 / bfloat16` 组合。
- 落盘记录：固定标成 `mla_dtype=fp8_block`、`kv_cache_dtype=fp8`、`gemm_type=fp8_block`。
- H100 SGLang 0.5.9 的 `wideep_context_mla_perf.txt` 和 `wideep_generation_mla_perf.txt` 也确实全部是上述标签。

这不是 SGLang 自动选择出的真实 FP8 module 采集结果，而是 collector 源码中显式写死的兼容行为。它的目的主要是复用旧版 `collect_wideep_attn.py` 与 SDK `wideep_*_mla` loader/query 的历史数据约定。

## 2. Collector 侧源码行为

`collect_mla_module.py` 中有两套精度枚举逻辑：

- `_get_precision_combos()` 会根据 SM 架构生成完整的 `(compute_dtype, kv_cache_dtype, gemm_type)` 组合，例如 Hopper 上包含 BF16 KV、FP8 KV、BF16 GEMM、FP8 block GEMM。
- `_get_module_precision_combos()` 只返回 `("bfloat16", "bfloat16", "bfloat16")`。

源码注释给出的理由是：module benchmark 需要启动完整 `ModelRunner`，每组成本较高，因此只采一个 BF16 baseline；精度相关 kernel 性能由 `collect_mla.py`、`collect_attn.py` 等 kernel-level collector 覆盖，module-level collector 主要捕获 scheduling、memory management、attention dispatch 等开销。

WideEP MLA 又有额外特殊逻辑：

```python
if is_wideep_mla:
    log_mla_dtype = "fp8_block"
    log_kv_dtype = "fp8"
    log_gemm_type = "fp8_block"
```

因此，WideEP MLA module 的实际运行 dtype 和落盘 dtype 被刻意解耦。也就是说，表项标签会让 SDK 认为这是 FP8/FP8 block 配置的数据，但 latency 本身来自 BF16 dummy module 执行。

## 3. 历史 `collect_wideep_attn.py` 证据

从 git 历史可见，旧版 `collector/sglang/collect_wideep_attn.py` 在提交 `57378df4` 中被删除，并被 `collect_mla_module.py` 吸收/替代。删除前的旧脚本已经具有同样语义：

- `hidden_states` 使用 `torch.bfloat16`。
- `decode_hidden` 使用 `torch.bfloat16`。
- `ServerArgs(dtype="auto")`，没有显式设置真实 FP8 KV cache 或 FP8 weight quantization。
- 落盘固定写 `mla_dtype="fp8_block"`、`kv_cache_dtype="fp8"`。

所以“兼容旧格式”并不是兼容一个真实的 FP8 WideEP module 采集协议，而是兼容一个从历史上就存在的“BF16 实跑、FP8 标签”的 WideEP MLA 表格式。

## 4. SDK 查询路径

SDK 中普通 `MLAModule` 与 WideEP MLA 是两条不同路径。

普通 `MLAModule`：

- 使用文件：`mla_context_module_perf.txt`、`mla_generation_module_perf.txt`。
- loader：`load_context_mla_module_data()`、`load_generation_mla_module_data()`。
- 查询 key：`fmha_quant_mode -> kv_cache_quant_mode -> gemm_quant_mode -> ...`。
- 这一路径严格按照三类 dtype 查表，没有看到自动把 BF16 映射到 FP8 的兼容逻辑。

WideEP MLA：

- 使用文件：`wideep_context_mla_perf.txt`、`wideep_generation_mla_perf.txt`。
- op：`WideEPContextMLA`、`WideEPGenerationMLA`。
- context loader 读取 `mla_dtype` 和 `kv_cache_dtype`，结构是 `kernel_source -> fmha_quant_mode -> kv_cache_dtype -> num_heads -> s -> b`。
- generation loader 只读取 `kv_cache_dtype`，结构是 `kernel_source -> kv_cache_dtype -> num_heads -> b -> s`。
- `gemm_type` 对 WideEP MLA 查询基本不生效，只是落盘字段。

因此，当前 WideEP context 如果请求 `FMHAQuantMode.fp8_block + KVCacheQuantMode.fp8` 可以命中；如果把新采集数据真实标成 `bfloat16/bfloat16`，现有 FP8 请求会无法命中。WideEP generation 更简单，主要由 `kv_cache_dtype=fp8` 决定是否命中。

## 5. 当前数据状态

以 H100 SGLang 0.5.9 为例：

- `wideep_context_mla_perf.txt` 存在，1000 行，全部为 `mla_dtype=fp8_block`、`kv_cache_dtype=fp8`、`gemm_type=fp8_block`。
- `wideep_generation_mla_perf.txt` 存在，1056 行，全部为 `mla_dtype=fp8_block`、`kv_cache_dtype=fp8`、`gemm_type=fp8_block`。
- `mla_context_module_perf.txt`、`mla_generation_module_perf.txt` 不存在。

进一步扫 `src/aiconfigurator/systems/data/*/sglang/*/wideep_*_mla_perf.txt`，已有 WideEP MLA 数据也基本沿用同一套 FP8 标签。部分 Blackwell/GB 系数据文件只有极少行，但仍属于同一命名和查询体系。

## 6. 影响面评估

如果直接把 collector 改成真实 dtype 落盘，例如 `bfloat16/bfloat16/bfloat16`，但不改 SDK：

- `WideEPDeepSeekModel` 的 `WideEPContextMLA` 和 `WideEPGenerationMLA` 会按配置请求旧的 FP8 key，导致 SILICON/HYBRID 查表失败。
- `Task` 支持矩阵检查会从 `wideep_context_mla` / `wideep_generation_mla` 推导可用 quant mode，UI/webapp 也依赖这些 key 展示可选项。
- 影响范围主要集中在 `SGLang + DeepSeek + enable_wideep + wideep_context_mla/wideep_generation_mla`。
- 普通 `context_mla/generation_mla`、普通 `mla_module`、DSA、MoE 等路径不应被直接影响。

所以这是一个局部但关键的兼容问题，不建议直接无兼容地改。

## 7. 建议迁移方案

更合理的方向是做 schema 迁移，而不是继续扩大语义误导：

1. 新 collector 真实记录实际执行 dtype，例如 BF16 baseline 就写 `bfloat16/bfloat16/bfloat16`。
2. WideEP context/generation loader 保留 legacy fallback：当发现旧表只有 `fp8_block/fp8` 但来源是历史 WideEP module 数据时，允许旧查询继续命中，并最好在 debug/warning 中标注 `legacy_bf16_labeled_fp8`。
3. WideEP generation loader 应补充 `mla_dtype/gemm_type` 维度，否则无法表达未来真实多精度 module 数据。
4. SDK 查询时优先真实 dtype 表；仅在真实表不存在时回退到 legacy 表。
5. 数据文件层面可以考虑增加显式字段，例如 `actual_mla_dtype`、`actual_kv_cache_dtype`、`actual_gemm_type` 或 `measurement_semantics`，避免同一个字段同时承担“实际执行精度”和“旧查询兼容标签”两种含义。

## 8. 最终判断

当前 WideEP MLA module 级表的 `fp8_block/fp8/fp8_block` 标签不能被解释为真实 FP8 module 实测。它更准确的含义是：

> 使用 BF16 dummy module 执行得到的 WideEP MLA module baseline latency，被挂载到历史 FP8 WideEP 查询 key 下供 SDK 使用。

这解释了为什么表项能被现有 WideEP SDK 查询命中，也解释了为什么它在量化语义上容易造成误导。后续若要提高仿真可信度，应将 collector 与 database 查询协议迁移到“真实 dtype 优先、legacy fallback 兼容”的模式。
