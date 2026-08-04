# PerfDatabase Attention 量化 SOL 估算修正说明

本文记录本次对 `src/aiconfigurator/sdk/perf_database.py` 中 attention / MLA 相关 SOL、EMPIRICAL 估算逻辑的修正背景、具体改动和边界。

## 背景结论

这次排查的核心发现是：

**SGLang 的 FA/FA3 attention 核心计算量化格式，在显式 KV cache dtype 生效时，实际主要由 `kv_cache_dtype` 主导，而不应只看 perf 表中的 `attn_dtype` 或 `mla_dtype` 字段。**

以 `collect_attn.py` 的 context/prefill 场景为例，会出现如下 perf 行：

```text
attn_dtype=bfloat16
kv_cache_dtype=fp8
```

这并不表示最终 FA 核心算子一定按 bf16 计算。它更准确表示：

```text
q/k/v 以 bf16 输入进入 RadixAttention，
但 SGLang backend 在 layer(...) 内部根据显式 fp8 KV cache dtype
将 q/k/v 对齐到 fp8，并执行 fp8 FA 路径。
```

因此，`attn_dtype` / `mla_dtype` 在 AIC 中有两个不同层面的含义：

- 在 SILICON 查表中，它们是 perf 表索引 key，必须保留原始语义。
- 在 SOL / EMPIRICAL 理论估算中，它们不一定等同于底层核心 attention kernel 的实际计算 dtype。

本次修改只调整 SOL / EMPIRICAL 的理论估算语义，不改变 SILICON 查表结构。

## 修改原则

本次采用一个保守规则：

```python
if kv_cache_dtype == fp8 and fmha_quant_mode == bfloat16:
    effective_fmha_mode = fp8
else:
    effective_fmha_mode = fmha_quant_mode
```

含义是：

- 如果 perf 表 key 显示 `attn_dtype` / `mla_dtype` 为 bf16，但 KV cache 是 fp8，则 SOL 估算按 SGLang runtime 的实际核心 attention dtype 处理为 fp8。
- 如果表 key 已经是 `fp8` / `fp8_block`，保持原值。
- 如果 KV cache 是 bf16，则不强行改写。

对应实现新增在 `PerfDatabase._effective_sglang_fmha_quant_mode()`。

## 具体改动

### 1. 新增 effective FMHA dtype helper

新增：

```python
def _effective_sglang_fmha_quant_mode(kvcache_quant_mode, fmha_quant_mode):
    ...
```

位置：`src/aiconfigurator/sdk/perf_database.py`

作用：

- 明确区分 perf 表 key 与 SOL 核心计算 dtype。
- 避免在多个 attention / MLA 查询函数中重复手写同一规则。
- 将 SGLang “显式 fp8 KV cache 会驱动 Q/K/V 对齐到 fp8”这一行为固化到理论估算路径。

### 2. 修正 `query_context_attention()`

原逻辑：

- `sol_math` 直接使用 `fmha_quant_mode.value.compute`。
- Q / output IO 固定按 2 bytes 估算。
- 对于 `attn_dtype=bfloat16, kv_cache_dtype=fp8` 的行，会把核心 FA 估算成 bf16。

新逻辑：

- 使用 `effective_fmha_mode` 计算 math throughput。
- Q / output IO bytes 使用 `effective_fmha_mode.value.memory`。
- KV read bytes 仍使用 `kvcache_quant_mode.value.memory`。

这使 context attention 的 SOL 估算和 SGLang FA3 内部 dtype 对齐行为一致。

### 3. 补充 `query_generation_attention()` 注释

decode attention 本身没有独立 `attn_dtype` 查询轴。

原代码已经按 `kvcache_quant_mode` 推导：

```python
if kvcache_quant_mode == fp8:
    quant_mode_gen = fp8
else:
    quant_mode_gen = bfloat16
```

本次未改变计算逻辑，只补充注释，说明 decode attention 的有效 FMHA mode 是从 KV cache dtype 推导而来。

### 4. 修正 `query_context_mla()`

原逻辑：

- `sol_math` 使用 `fmha_quant_mode.value.compute`。
- Q / output 相关 IO 固定用 bf16 风格的 `2 * s * (...)`。

新逻辑：

- 使用 `effective_fmha_mode` 估算核心 MLA attention 计算。
- KV cache bytes 使用 `kvcache_quant_mode`。
- Q / output bytes 使用 `effective_fmha_mode`。

这修正了 `mla_dtype=bfloat16, kv_cache_dtype=fp8` 时 SOL 仍按 bf16 attention 估算的问题。

### 5. 修正 `query_context_mla_module()`

模块级 context MLA 的 SOL 原先复用 context MLA 公式，但同样直接使用 `fmha_quant_mode`。

新逻辑：

- 使用 `effective_fmha_mode` 修正 math throughput。
- 使用 `effective_fmha_mode.value.memory` 修正 Q / output IO。
- SILICON 查询仍保持原表结构：`[fmha_quant_mode][kvcache_quant_mode][gemm_quant_mode]`。

这点很重要：module-level 表的 `mla_dtype` 仍然是查表 key，不因 SOL 估算修正而改变。

### 6. 补充 `query_generation_mla()` 与 `query_generation_mla_module()` 注释

decode MLA 已经按 KV cache dtype 推导核心 attention 量化模式：

```python
if kv_cache_dtype == fp8:
    quant_mode_gen = fp8
else:
    quant_mode_gen = bfloat16
```

本次主要补充注释，明确：

- generation MLA perf 表没有独立有效的 `mla_dtype` 查询轴。
- SOL compute 应跟随 KV cache dtype。

### 7. 修正 WideEP MLA SOL attention 部分

`query_wideep_generation_mla()` 和 `query_wideep_context_mla()` 中，WideEP MLA 的历史 perf 表会使用 `fmha_quant_mode` 作为 module schema key，例如 `fp8_block`。

原逻辑问题：

- projection / module 其他部分使用 `fmha_quant_mode` 估算。
- 但 standalone attention flop 部分固定按 bf16 吞吐估算：

```python
sol_math += attn_flop / bfloat16_tc_flops
```

新逻辑：

- 对 standalone attention flop 使用 `effective_attn_mode`。
- 当 KV cache 为 fp8 且表 key 为 bf16 时，attention 部分按 fp8 估算。
- 当 WideEP 历史 key 是 `fp8_block` 时，保持其原有 schema 语义。

这避免了 WideEP MLA SOL 中“表 key 与实际 attention kernel dtype 混用”的问题。

### 8. 修正 DSA generation SOL 注释与实现不一致

`query_generation_dsa_module()` 原注释写的是：

```text
attention group (fmha derived from kv_cache_dtype)
```

但实现固定：

```python
fmha_mode = common.FMHAQuantMode.bfloat16
```

本次修正为：

```python
fmha_mode = fp8 if kv_cache_dtype == fp8 else bfloat16
```

这使 generation DSA 的 SOL 估算与注释、KV cache dtype 语义保持一致。

### 9. 调整 DSA context sparse attention 注释与 IO 估算

`query_context_dsa_module()` 中 sparse MLA attention group 原先直接使用 `fmha_quant_mode`。

本次改为：

- 使用 `effective_fmha_mode` 估算 sparse attention throughput。
- Q activation read/write bytes 使用 `effective_fmha_mode.value.memory`。

同时保留 DSA context 表 key 语义：SILICON 查表仍按 `fmha_quant_mode` 访问原数据结构。

## 没有修改的部分

### SILICON 查表路径未改

所有 SILICON 查询仍然使用原始 perf 表 key，例如：

```python
context_attention_data[fmha_quant_mode][kvcache_quant_mode]
context_mla_data[fmha_quant_mode][kvcache_quant_mode]
context_mla_module_data[fmha_quant_mode][kvcache_quant_mode][gemm_quant_mode]
```

原因：

- `attn_dtype` / `mla_dtype` 在 perf database 中仍是历史数据索引。
- 实测数据本身已经包含 collector 边界内的 cast 和 kernel 行为。
- 修改 SILICON key 会破坏现有数据结构和兼容性。

### GEMM / MoE 等非 attention 路径未改

这次只处理 attention / MLA / DSA attention 相关 SOL 估算。

GEMM collector 中 activation quant 是否包含在计时边界内，是另一个独立问题；本次不改变 GEMM SOL 模型。

## 影响范围

本次修改影响以下 database mode：

- `SOL`
- `SOL_FULL`
- `EMPIRICAL`
- `HYBRID` 中 fallback 到 empirical 的路径

不影响：

- `SILICON` 直接查实测表的主路径。
- perf txt 文件格式。
- collector 输出字段。
- SDK operation 的入参结构。

## 为什么这样修正

之前的 SOL 模型把 `attn_dtype` / `mla_dtype` 近似当成了核心 attention kernel dtype。

但根据 SGLang 0.5.9 FA/FA3 路径分析：

- 显式 `kv_cache_dtype=fp8_e4m3` 会使 backend 进入非 `auto` dtype 分支。
- backend 会把 Q 侧张量对齐到 `model_runner.kv_cache_dtype`。
- MHA / MLA attention 调用中，K/V 或 KV cache 会继续对齐到 `q.dtype`。
- 因而最终 FA kernel 看到的核心计算 dtype 通常由 KV cache dtype 主导。

所以，在理论估算里继续用 `attn_dtype=bfloat16` 估算 fp8 KV cache 场景，会低估 fp8 路径的 tensor core throughput，同时高估 Q/output IO bytes。

这次修正就是把 SOL/EMPIRICAL 的理论口径拉回 SGLang runtime 行为。

## 验证

已执行：

```bash
python -m py_compile src/aiconfigurator/sdk/perf_database.py
git diff --check -- src/aiconfigurator/sdk/perf_database.py
```

结果：

- Python 语法编译通过。
- diff whitespace 检查通过。

有一个额外小测尝试直接 import `PerfDatabase`，但当前环境没有安装 `aiconfigurator` package metadata，触发 `PackageNotFoundError`。这属于环境安装状态问题，不是本次代码语法问题。

## 后续注意事项

后续如果继续校准 attention 相关误差，建议始终区分三类 dtype：

- `kv_cache_dtype`：SGLang runtime 中最关键的 attention dtype 对齐主轴。
- `attn_dtype` / `mla_dtype`：AIC perf 表字段，既可能是采集边界标签，也可能是历史查表 key。
- `gemm_type`：projection GEMM / module 内线性层的量化主轴，不应和 attention kernel dtype 混为一谈。

尤其是 `attn_dtype=bfloat16, kv_cache_dtype=fp8` 这类组合，不应直觉判断为“bf16 attention”。它更可能表示“bf16 live q/k/v 输入进入 collector 边界，但 SGLang 内部按 fp8 KV cache dtype 对齐并执行 fp8 FA/MLA 核心路径”。
