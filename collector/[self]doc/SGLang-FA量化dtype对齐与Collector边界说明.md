# SGLang FA量化dtype对齐与Collector边界说明

本文聚焦两个问题：

- SGLang 0.5.9 原生 `RadixAttention -> FlashAttentionBackend` 路径中，FA 算子的 Q/K/V dtype 如何由 `kv_cache_dtype` 影响。
- AIC `collector/sglang/collect_mla.py` 中，哪些量化/cast 操作在采集边界内，哪些在采集边界外。

结论先行：

- 对 SGLang 0.5.9 的 FA3 路径，若 `kv_cache_dtype_str != "auto"` 且 `layer.head_dim <= 256`，`FlashAttentionBackend` 会在调用 FA 前将 `q` 或 `q_rope` 对齐到 `self.kv_cache_dtype`。因此显式 `kv_cache_dtype=fp8` 时，FA 调用前的 Q 侧 dtype 会被主动改成 fp8。
- 在 DeepSeek MLA 的 prefill-MHA 分支中，SGLang 随后会将传入 `flash_attn_varlen_func` 的 `k/v` 转成 `q.dtype`。由于此时 `q` 已按 KV dtype 对齐，所以最终 FA varlen 调用看到的 Q/K/V dtype 通常一致，且由显式 KV dtype 主导。
- 在 absorbed MLA decode/extend 分支中，SGLang 会取出 MLA KV cache 后执行 `.to(q.dtype)`，再拆出 rope/nope 部分传入 `flash_attn_with_kvcache`。在显式 fp8 KV 条件下，前面 `q` 已被转成 fp8，所以这里不是“固定反量化到 bf16”，而是对齐到当前 `q.dtype`。
- 对 AIC `collect_mla`，构造输入阶段的 `k = k.to(kv_cache_dtype)` 在 benchmark 之外，不计入采集；但 SGLang backend 内部的 `q.to(self.kv_cache_dtype)`、`k.to(q.dtype)`、`kv_cache.to(q.dtype)` 都发生在 `layer(...)` 调用内，会计入 collector 的 latency。

## 源码依据：AIC collect 采集边界

本地文件：`collector/sglang/collect_mla.py`

`benchmark_layer()` 的计时函数只包住 `layer(...)`：

```python
def benchmark_layer(layer, forward_batch, q, k, v, q_rope, k_rope, **kwargs):
    def kernel_func():
        extra_kwargs = dict(kwargs)
        if q_rope is not None:
            extra_kwargs["q_rope"] = q_rope
        if k_rope is not None:
            extra_kwargs["k_rope"] = k_rope
        layer(q, k, v, forward_batch, **extra_kwargs)

    with benchmark_with_power(
        device=device,
        kernel_func=kernel_func,
        num_warmups=3,
        num_runs=20,
        repeat_n=1,
    ) as results:
        pass
```

对应行号：`collector/sglang/collect_mla.py:191-212`。

因此，`run_mla()` 中在调用 `benchmark_layer()` 之前完成的张量构造和预处理不计入采集。

prefill-MHA 构造阶段，collector 会先生成 bf16 的 `q_nope/q_rope/k_nope/k_rope/v`。如果 `kv_cache_dtype == torch.float8_e4m3fn`，它会在 benchmark 之前把合并后的 `k` 转成 fp8：

```python
q = torch.cat([q_nope, q_rope], dim=-1)
k = torch.cat([k_nope, k_rope.expand(-1, local_num_heads, -1)], dim=-1)
if kv_cache_dtype == torch.float8_e4m3fn:
    # forward_mha._concat_and_cast_mha_k targets the KV-pool dtype
    # for FA3 when kv_cache_dtype is not "auto"; the backend casts
    # k back to q.dtype inside flash_attn_varlen_func.
    k = k.to(kv_cache_dtype)
```

对应行号：`collector/sglang/collect_mla.py:578-585`。

这个 `k.to(kv_cache_dtype)` 在 `benchmark_layer()` 之前，因此不属于采集边界。

collector 对 prefill-MHA 还显式设置：

```python
forward_batch.set_attn_attend_prefix_cache(False)
forward_batch.mha_return_lse = False
```

对应行号：`collector/sglang/collect_mla.py:610-614`。

调用 benchmark 时，prefill-MHA 场景还会设置：

```python
save_kv_cache=not (is_context_phase and num_kv_heads == local_num_heads)
```

对应行号：`collector/sglang/collect_mla.py:711-720`。

在 prefill-MHA 的 `is_context_phase=True` 且 `num_kv_heads == local_num_heads` 场景下，`save_kv_cache=False`。也就是说，collector 的 prefill-MHA 不是在计时内写 KV cache，而是直接测 FA varlen 分支及其内部 dtype 对齐。

## kv_cache_dtype_str 的来源与默认解析

参考镜像：`booleimg.myaddr.io/lmsysorg/sglang:v0.5.9`

`FlashAttentionBackend` 中使用的两个字段来自 `ModelRunner`：

```python
self.kv_cache_dtype = model_runner.kv_cache_dtype
self.kv_cache_dtype_str = model_runner.server_args.kv_cache_dtype
```

对应行号：容器内 `flashattention_backend.py:348-349`。

这里要区分两个概念：

- `model_runner.kv_cache_dtype` 是真实 torch dtype，例如 `torch.bfloat16`、`torch.float8_e4m3fn`。
- `model_runner.server_args.kv_cache_dtype` 是配置字符串，例如 `"auto"`、`"bf16"`、`"bfloat16"`、`"fp8_e4m3"`。

FA 后端判断是否执行 Q 侧 dtype 对齐时，用的是字符串字段 `kv_cache_dtype_str`，而真正 `.to(...)` 的目标 dtype 用的是 torch dtype 字段 `kv_cache_dtype`。

### 1. ServerArgs 默认值

`ServerArgs` 的默认 KV cache dtype 是 `"auto"`：

```python
# Quantization and data type
dtype: str = "auto"
quantization: Optional[str] = None
quantization_param_path: Optional[str] = None
kv_cache_dtype: str = "auto"
```

对应行号：容器内 `server_args.py:310-314`。

CLI 参数 `--kv-cache-dtype` 也默认使用这个值：

```python
parser.add_argument(
    "--kv-cache-dtype",
    type=str,
    default=ServerArgs.kv_cache_dtype,
    choices=["auto", "fp8_e5m2", "fp8_e4m3", "bf16", "bfloat16", "fp4_e2m1"],
    help='Data type for kv cache storage. "auto" will use model data type. '
    '"bf16" or "bfloat16" for BF16 KV cache. "fp8_e5m2" and "fp8_e4m3" '
    'are supported for CUDA 11.8+. "fp4_e2m1" ...',
)
```

对应行号：容器内 `server_args.py:3079-3085`。

所以，一般运行 SGLang 时，如果没有显式传 `--kv-cache-dtype`，默认就是 `"auto"`。

### 2. ModelRunner 如何解析 auto

`ModelRunner.__init__` 中会调用：

```python
# Deduce KV cache dtype
self.configure_kv_cache_dtype()
```

对应行号：容器内 `model_runner.py:586-587`。

解析逻辑如下：

```python
def configure_kv_cache_dtype(self):
    if self.server_args.kv_cache_dtype == "auto":
        quant_config = getattr(self.model, "quant_config", None)
        kv_cache_quant_algo = getattr(quant_config, "kv_cache_quant_algo", None)
        if (
            isinstance(kv_cache_quant_algo, str)
            and kv_cache_quant_algo.upper() == "FP8"
        ):
            if _is_hip:
                self.kv_cache_dtype = fp8_dtype
                self.server_args.kv_cache_dtype = TORCH_DTYPE_TO_KV_CACHE_STR[
                    self.kv_cache_dtype
                ]
            else:
                self.kv_cache_dtype = torch.float8_e4m3fn
                self.server_args.kv_cache_dtype = TORCH_DTYPE_TO_KV_CACHE_STR[
                    self.kv_cache_dtype
                ]
        else:
            self.kv_cache_dtype = self.dtype
    elif self.server_args.kv_cache_dtype == "fp8_e5m2":
        ...
    elif self.server_args.kv_cache_dtype == "fp8_e4m3":
        self.kv_cache_dtype = torch.float8_e4m3fn
    elif self.server_args.kv_cache_dtype in ("bf16", "bfloat16"):
        self.kv_cache_dtype = torch.bfloat16
```

对应行号：容器内 `model_runner.py:1669-1700`。

这段代码有一个关键细节：

- 当 `kv_cache_dtype == "auto"` 且模型量化配置中没有 `kv_cache_quant_algo == "FP8"` 时，只设置 `self.kv_cache_dtype = self.dtype`，不改写 `self.server_args.kv_cache_dtype`。因此后续 `FlashAttentionBackend.kv_cache_dtype_str` 仍然是 `"auto"`。
- 当 `kv_cache_dtype == "auto"` 且模型量化配置声明 KV cache FP8 时，会设置 `self.kv_cache_dtype = torch.float8_e4m3fn`，并把 `self.server_args.kv_cache_dtype` 改写为 `"fp8_e4m3"`。
- 当用户显式传 `--kv-cache-dtype bf16` 或 `--kv-cache-dtype bfloat16` 时，`self.kv_cache_dtype = torch.bfloat16`，字符串字段保持非 `"auto"`。
- 当用户显式传 `--kv-cache-dtype fp8_e4m3` 时，`self.kv_cache_dtype = torch.float8_e4m3fn`，字符串字段保持非 `"auto"`。

### 3. auto、bf16、fp8 对 FA dtype 对齐的实际影响

结合 FA 后端的判断：

```python
if (
    self.kv_cache_dtype_str != "auto"
    and layer.head_dim <= 256
    and self.fa_impl_ver != 4
):
    ...
    q = q.to(self.kv_cache_dtype)
```

可以得到以下行为：

| 配置来源 | `server_args.kv_cache_dtype` 进入 FA 后端时 | `model_runner.kv_cache_dtype` | 是否触发 `kv_cache_dtype_str != "auto"` 分支 | Q 侧行为 |
| --- | --- | --- | --- | --- |
| 默认 `auto`，模型无 KV FP8 量化配置 | `"auto"` | `self.dtype`，通常 bf16/fp16 | 否 | 不在该分支内主动 `.to(kv_cache_dtype)` |
| 默认 `auto`，模型量化配置声明 `kv_cache_quant_algo == "FP8"` | `"fp8_e4m3"` | `torch.float8_e4m3fn` | 是 | Q/Q-rope 转 fp8 |
| 显式 `--kv-cache-dtype bfloat16` | `"bfloat16"` | `torch.bfloat16` | 是 | Q/Q-rope 转 bf16 |
| 显式 `--kv-cache-dtype bf16` | `"bf16"` | `torch.bfloat16` | 是 | Q/Q-rope 转 bf16 |
| 显式 `--kv-cache-dtype fp8_e4m3` | `"fp8_e4m3"` | `torch.float8_e4m3fn` | 是 | Q/Q-rope 转 fp8 |

因此，`kv_cache_dtype_str != "auto"` 不是“KV cache 实际 dtype 不是模型 dtype”的同义词。它更准确地表示：用户或模型量化配置已经让 SGLang 进入显式 KV cache dtype 模式。默认 `auto` 且无 KV FP8 配置时，FA 后端不会执行这段显式 Q 侧 dtype 对齐。

### 4. 相关兼容性与特殊模型逻辑

SGLang 对部分 backend/dtype 有额外限制。例如 FA3 不支持 `fp8_e5m2`：

```python
if self.attention_backend == "fa3" and self.kv_cache_dtype == "fp8_e5m2":
    logger.warning(
        "FlashAttention3 only supports fp8_e4m3 if using FP8; "
        "Setting attention backend to triton."
    )
```

对应行号：容器内 `server_args.py:1910-1914`。

另一个特殊分支是 DeepSeek DSA/NSA 相关逻辑。该逻辑会在 server args 校验阶段把 `"auto"` 改写成硬件相关默认值：

```python
if self.kv_cache_dtype == "auto":
    self.kv_cache_dtype = "fp8_e4m3" if major >= 10 else "bfloat16"
    logger.warning(
        f"Setting KV cache dtype to {self.kv_cache_dtype} for DeepSeek DSA on SM{major} device."
    )
if self.kv_cache_dtype == "bf16":
    self.kv_cache_dtype = "bfloat16"
```

对应行号：容器内 `server_args.py:1142-1148`。

也就是说，对普通 SGLang 运行，`--kv-cache-dtype` 默认是 `"auto"`，通常解析为模型 dtype 且字符串仍为 `"auto"`；但对部分特殊模型/后端组合，SGLang 可能在进入 `ModelRunner` 之前就把 `"auto"` 改写为 `"fp8_e4m3"` 或 `"bfloat16"`。

### 5. DeepSeek 模型配置是否会自动声明 FP8 KV cache

需要区分两类“FP8”：

- 权重量化 FP8：`quantization_config.quant_method == "fp8"`，通常描述线性层/专家权重的 FP8 存储和 GEMM 路径。
- KV cache 量化 FP8：`kv_cache_quant_algo == "FP8"` 或等价字段，才会被 `ModelRunner.configure_kv_cache_dtype()` 用来把默认 `"auto"` 改写为 `"fp8_e4m3"`。

本仓库的常规 DeepSeek-V3 / R1 配置属于第一类。以 `src/aiconfigurator/model_configs/deepseek-ai--DeepSeek-V3_config.json` 为例：

```json
"quantization_config": {
  "activation_scheme": "dynamic",
  "fmt": "e4m3",
  "quant_method": "fp8",
  "weight_block_size": [
    128,
    128
  ]
}
```

对应行号：`src/aiconfigurator/model_configs/deepseek-ai--DeepSeek-V3_config.json:37-44`。

`deepseek-ai--DeepSeek-R1_config.json` 也是同样结构：`quant_method=fp8`，但没有 `kv_cache_quant_algo`，对应行号：`src/aiconfigurator/model_configs/deepseek-ai--DeepSeek-R1_config.json:37-44`。

`deepseek-ai--DeepSeek-V3.2_config.json` 也声明了 FP8 权重量化：

```json
"quantization_config": {
  "activation_scheme": "dynamic",
  "fmt": "e4m3",
  "quant_method": "fp8",
  "scale_fmt": "ue8m0",
  "weight_block_size": [
    128,
    128
  ]
}
```

对应行号：`src/aiconfigurator/model_configs/deepseek-ai--DeepSeek-V3.2_config.json:35-43`。

但在 SGLang 0.5.9 的 `Fp8Config` 中，`from_config()` 只读取 `quant_method`、`activation_scheme`、`ignored_layers`、`weight_block_size` 等字段：

```python
class Fp8Config(QuantizationConfig):
    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
        activation_scheme: str = "dynamic",
        ignored_layers: Optional[List[str]] = None,
        weight_block_size: List[int] = None,
        use_mxfp8: bool = False,
    ) -> None:
        ...

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Fp8Config:
        quant_method = cls.get_from_keys(config, ["quant_method"])
        use_mxfp8 = "mxfp8" in quant_method
        is_checkpoint_fp8_serialized = ("fp8" in quant_method) or use_mxfp8
        activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
        ignored_layers = cls.get_from_keys_or(
            config, ["ignored_layers", "modules_to_not_convert"], None
        )
        weight_block_size = cls.get_from_keys_or(config, ["weight_block_size"], None)
        ...
```

对应行号：容器内 `fp8.py:111-190`。

这个 `Fp8Config` 没有设置 `kv_cache_quant_algo` 字段。因此，常规 DeepSeek-V3 / R1 的 `quant_method=fp8` 只会让权重/GEMM 侧使用 FP8 量化配置，不会让 `ModelRunner.configure_kv_cache_dtype()` 的：

```python
kv_cache_quant_algo = getattr(quant_config, "kv_cache_quant_algo", None)
```

拿到 `"FP8"`。在默认 `--kv-cache-dtype auto` 下，它不会因此自动变成 FP8 KV cache。

有 KV cache FP8 声明的是另一类 ModelOpt/NVIDIA 风格配置。例如本仓库 `src/aiconfigurator/model_configs/nvidia--DeepSeek-V3.1-NVFP4_hf_quant_config.json`：

```json
"quantization": {
  "quant_algo": "NVFP4",
  "kv_cache_quant_algo": "FP8",
  "group_size": 16,
  ...
}
```

对应行号：`src/aiconfigurator/model_configs/nvidia--DeepSeek-V3.1-NVFP4_hf_quant_config.json:6-9`。

这类配置进入 SGLang 的 ModelOpt quant config 后，才会使 `kv_cache_quant_algo == "FP8"` 成立，从而在 `--kv-cache-dtype auto` 时把 KV cache dtype 自动改写为 `"fp8_e4m3"`。

DeepSeek 模型代码本身也不是从 `quantization_config.quant_method=fp8` 推导 KV cache FP8，而是直接读取全局 server args：

```python
self.kv_cache_dtype = get_global_server_args().kv_cache_dtype
```

对应行号：容器内 `deepseek_v2.py:1114`。

后续 DeepSeek MHA 路径也根据这个字符串判断，例如：

```python
if (
    self.current_attention_backend == "fa3"
    and self.kv_cache_dtype != "auto"
):
    attn_dtype = forward_batch.token_to_kv_pool.dtype
else:
    attn_dtype = k_nope.dtype
```

对应行号：容器内 `forward_mha.py:488-496`。

所以，对 DeepSeek-V3/R1 常规 FP8 权重模型，默认 `--kv-cache-dtype auto` 一般仍不会触发显式 FP8 KV cache dtype 对齐；要触发 FP8 KV cache，通常需要显式传 `--kv-cache-dtype fp8_e4m3`，或使用带 `kv_cache_quant_algo=FP8` 的 ModelOpt/NVIDIA 量化配置，或进入前文提到的特殊 DSA/NSA 自动改写逻辑。

## 源码依据：SGLang 0.5.9 原生 FA 后端

参考镜像：`booleimg.myaddr.io/lmsysorg/sglang:v0.5.9`

容器源码路径：`/sgl-workspace/sglang/python/sglang/srt/layers/attention/flashattention_backend.py`

### 1. forward_extend 中 Q 侧跟随 KV dtype

`forward_extend()` 在进入具体 attention 分支前，有统一 dtype 对齐逻辑：

```python
k_descale, v_descale = None, None
# only use kv scaling if: 1) fp8 kv is explicitly enabled, 2) RadixAttention
# has corresponding quantization method so that layer.k_scale is not None,
# 3) layer.head_dim <= 256 since fa3 kernel require fp16 and bf16 data type in this case,
# 4) fa_impl_ver != 4 since fa4 does not currently support fp8 queries and keys.
if (
    self.kv_cache_dtype_str != "auto"
    and layer.head_dim <= 256
    and self.fa_impl_ver != 4
):
    if layer.k_scale is not None:
        descale_shape = (forward_batch.batch_size, layer.tp_k_head_num)
        k_descale = layer.k_scale.expand(descale_shape)
        v_descale = layer.v_scale.expand(descale_shape)
    q = q.to(self.kv_cache_dtype)
    q_rope = q_rope.to(self.kv_cache_dtype) if q_rope is not None else None
    k_rope = k_rope.to(self.kv_cache_dtype) if k_rope is not None else None
```

对应行号：容器内 `flashattention_backend.py:778-795`。

这说明：显式设置 `kv_cache_dtype` 时，且满足 head_dim/FA3 条件，Q 侧会被改成 KV cache dtype。

### 2. DeepSeek MLA prefill-MHA 分支中 K/V 跟随 Q dtype

当模型是 MLA 架构且 `forward_batch.attn_attend_prefix_cache` 被设置时，SGLang 走 MHA 分支。对于 attend prefix cache 的 chunk：

```python
output = flash_attn_varlen_func(
    q=q.view(-1, layer.tp_q_head_num, layer.head_dim),
    k=k.view(-1, layer.tp_k_head_num, layer.head_dim).to(q.dtype),
    v=v.view(-1, layer.tp_k_head_num, layer.v_head_dim).to(q.dtype),
    ...
)
```

对应行号：容器内 `flashattention_backend.py:953-965`。

对于不 attend prefix cache 的 extend part：

```python
output = flash_attn_varlen_func(
    q=q.view(-1, layer.tp_q_head_num, layer.head_dim),
    k=k.view(-1, layer.tp_k_head_num, layer.head_dim).to(q.dtype),
    v=v.view(-1, layer.tp_k_head_num, layer.v_head_dim).to(q.dtype),
    ...
)
```

对应行号：容器内 `flashattention_backend.py:978-990`。

结合上一节，若 `kv_cache_dtype=fp8` 且满足条件，进入这里前 `q` 已经被转为 fp8，所以 `k/v` 也会被转成 fp8。若 `kv_cache_dtype=bfloat16`，则最终 `q/k/v` 维持 bf16。

### 3. absorbed MLA extend 分支中 KV cache 跟随 Q dtype

当不走 prefill-MHA 分支，而走 absorbed MLA 时：

```python
kv_cache = forward_batch.token_to_kv_pool.get_key_buffer(
    layer.layer_id
).to(q.dtype)
k_rope = kv_cache[:, :, layer.v_head_dim :]
c_kv = kv_cache[:, :, : layer.v_head_dim]
...
result = flash_attn_with_kvcache(
    q=q_rope,
    k_cache=k_rope_cache,
    v_cache=c_kv_cache,
    qv=q_nope,
    ...
    k_descale=k_descale,
    v_descale=v_descale,
)
```

对应行号：容器内 `flashattention_backend.py:997-1040`。

这里不是固定反量化到 bf16，而是 `.to(q.dtype)`。如果前面的统一逻辑已把 `q` 转成 fp8，则 KV cache 也会转到 fp8。

### 4. forward_decode 中同样先让 Q 侧跟随 KV dtype

`forward_decode()` 中也有同类逻辑：

```python
k_descale, v_descale = None, None
# only use kv scaling if: 1) fp8 kv is explicitly enabled, 2) RadixAttention
# has corresponding quantization method so that layer.k_scale is not None,
# 3) layer.head_dim <= 256 since fa3 kernel require fp16 and bf16 data type in this case.
if self.kv_cache_dtype_str != "auto" and layer.head_dim <= 256:
    if layer.k_scale is not None:
        descale_shape = (forward_batch.batch_size, layer.tp_k_head_num)
        k_descale = layer.k_scale.expand(descale_shape)
        v_descale = layer.v_scale.expand(descale_shape)
    q = q.to(self.kv_cache_dtype)
    q_rope = q_rope.to(self.kv_cache_dtype) if q_rope is not None else None
    k_rope = k_rope.to(self.kv_cache_dtype) if k_rope is not None else None
```

对应行号：容器内 `flashattention_backend.py:1139-1150`。

absorbed MLA decode 分支继续使用：

```python
kv_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id).to(
    q.dtype
)
...
result = flash_attn_with_kvcache(
    q=q_rope,
    k_cache=k_rope_cache,
    v_cache=c_kv_cache,
    qv=q_nope,
    ...
    k_descale=k_descale,
    v_descale=v_descale,
)
```

对应行号：容器内 `flashattention_backend.py:1272-1317`。

所以 decode 侧也不是简单“fp8 KV cache 先反量化到 bf16”。它首先根据显式 KV dtype 改写 Q 侧 dtype，然后 KV cache 再 `.to(q.dtype)`。

## 精确回答：FA 最终执行 dtype 是否跟随 kv_dtype

可以这样精确表述：

在 SGLang 0.5.9 FA3 路径中，若满足以下条件：

- `self.kv_cache_dtype_str != "auto"`
- `layer.head_dim <= 256`
- extend 路径中 `self.fa_impl_ver != 4`

则 `FlashAttentionBackend` 会先把 Q 侧张量转到 `self.kv_cache_dtype`。之后：

- DeepSeek MLA prefill-MHA 分支会把 `k/v` 转成 `q.dtype` 后调用 `flash_attn_varlen_func`。
- absorbed MLA extend/decode 分支会把 `kv_cache` 转成 `q.dtype` 后再拆分为 `k_rope_cache/c_kv_cache`，传入 `flash_attn_with_kvcache`。
- 普通 MHA paged KV 分支会把 `q` 转到 `self.kv_cache_dtype`，同时把 `k_descale/v_descale` 传入 `flash_attn_with_kvcache`。这一路更接近 FA kernel 直接消费 KV cache dtype 和 scale 信息。

因此，不能笼统说“FA 永远直接接收 bf16 q 和 fp8 kv，并在 kernel 内部完成反量化”。对 DeepSeek MLA 相关路径，更准确的说法是：

**显式 KV dtype 会主导 SGLang backend 的 dtype 对齐；最终传给 FA 的 Q/K/V 或 Q/K-cache/V-cache 通常已经被对齐到当前路径的目标 dtype。对 AIC `collect_mla` 来说，collector 准备阶段的 cast 不计入，但 SGLang backend 内部的 dtype 对齐 cast 计入采集边界。**

## 与 AIC collect 的对应关系

`collect_mla` 使用 `RadixAttention` 并把 `forward_batch.attn_backend` 设置为 SGLang 的 attention backend。因此从 `layer(...)` 进入以后，它复用了 SGLang 的 backend dtype 对齐逻辑。

边界可以分成两类：

- 采集外：collector 在 `benchmark_layer()` 之前构造输入、拼接 q/k、以及 prefill-MHA 中的 `k = k.to(kv_cache_dtype)`。
- 采集内：`layer(...)` 内部触发的 `q.to(self.kv_cache_dtype)`、`q_rope.to(self.kv_cache_dtype)`、`k_rope.to(self.kv_cache_dtype)`、`k.to(q.dtype)`、`v.to(q.dtype)`、`kv_cache.to(q.dtype)`。

这意味着 `collect_mla` 采到的 fp8 attention latency 不是“纯 FA kernel latency”，而是包含了 SGLang backend 在 RadixAttention 调用内部执行的 dtype 对齐成本。它更接近“从 RadixAttention attention 子算子入口到 FA 输出”的端到端子路径，而不是单个底层 FA CUDA kernel 的净时延。

## 最终确认：FA 核心算子 dtype 是否跟随 KV cache dtype

可以确认，但需要带上作用条件：

**在 SGLang 0.5.9 的 FA/FA3 attention backend 中，只要进入显式 KV cache dtype 模式，且底层 kernel 支持该 dtype，attention 路径会主动把 Q 侧对齐到 `model_runner.kv_cache_dtype`，后续 K/V 或 KV cache 再对齐到 `q.dtype`，因此最终 FA 调用的量化格式由 KV cache dtype 主导。**

这里的“显式 KV cache dtype 模式”包括：

- 用户显式设置 `--kv-cache-dtype bfloat16/bf16/fp8_e4m3`。
- `--kv-cache-dtype auto` 但模型量化配置声明了 `kv_cache_quant_algo == "FP8"`，使 `ModelRunner.configure_kv_cache_dtype()` 把 `server_args.kv_cache_dtype` 改写成 `"fp8_e4m3"`。
- 特殊后端/模型逻辑在 `server_args` 校验阶段把 `"auto"` 改写为具体 dtype，例如部分 DeepSeek DSA/NSA 逻辑。

反过来，如果 `server_args.kv_cache_dtype` 字符串在进入 `FlashAttentionBackend` 时仍是 `"auto"`，则 `kv_cache_dtype_str != "auto"` 分支不会触发；此时 attention dtype 主要沿用模型/激活 dtype，通常是 bf16/fp16。也就是说，不能把“默认 auto”直接等价成“FA 一定跟随 KV cache dtype 做显式 cast”。更准确的说法是：

**一旦 SGLang 将 KV cache dtype 解析为非 auto 的具体执行 dtype，FA 路径会以这个 dtype 作为 Q/K/V 对齐目标；默认 auto 且无 KV FP8 配置时，不走这条显式对齐逻辑。**

## `mla_dtype` 字段语义：不是 SGLang runtime 控制项

`collect_mla.py` 和 `collect_mla_module.py` 中的 `mla_dtype` 容易被误读成“控制 MLA/FA 核心算子 dtype 的参数”。从代码看，它不是。

### 1. `collect_mla.py`：`mla_dtype` 只写表，不控制执行

单算子 collector 的真实执行控制量是 `kv_cache_dtype`：

- `MockModelRunner.kv_cache_dtype` 保存 torch dtype。
- `MockServerArgs.kv_cache_dtype` 根据该 torch dtype 写成 `"fp8"` 或 `"bfloat16"`。
- `run_mla()` 只接受 `torch.bfloat16` 和 `torch.float8_e4m3fn` 两种 `kv_cache_dtype`。
- prefill-MHA 校验中还要求 `k.dtype` 跟随 `kv_cache_dtype`，即 fp8 KV 时 `k` 为 `torch.float8_e4m3fn`。

但最终写 perf 行时：

```python
"mla_dtype": "bfloat16",
"kv_cache_dtype": str_type,
```

对应行号：`collector/sglang/collect_mla.py:730-735`。

这里 `mla_dtype` 被硬编码为 `"bfloat16"`，并没有参与 `MockModelRunner`、`MockServerArgs`、`RadixAttention` 或 `FlashAttentionBackend` 的构造。因此在 `collect_mla.py` 中，它是 perf schema / SDK 查询维度，不是执行开关。真正影响 attention dtype 对齐的是 `kv_cache_dtype`。

### 2. `collect_mla_module.py`：执行控制轴是 `kv_cache_dtype` 和 `gemm_type`

模块级 collector 的 precision 组合是 `(compute_dtype, kv_cache_dtype, gemm_type)`：

```python
compute_dtype:  always "bfloat16"
kv_cache_dtype: "bfloat16" always; "fp8" on SM >= 90
gemm_type:      "bfloat16" or "fp8_block"
```

对应行号：`collector/sglang/collect_mla_module.py:151-178`。

进入 SGLang runtime 时，真正传入 `ServerArgs` 的是：

```python
kv_cache_dtype=sglang_kv_dtype
```

并且当 `gemm_type == "fp8_block"` 时：

```python
server_args.quantization = "fp8"
```

对应行号：`collector/sglang/collect_mla_module.py:574-603`。

因此：

- `kv_cache_dtype` 控制 KV cache / attention backend dtype 对齐。
- `gemm_type` 控制是否启用 SGLang 的 FP8 权重/GEMM 路径。
- `compute_dtype` 主要是 collector 内部组合和写表语义，目前注释明确为 always `"bfloat16"`。
- `mla_dtype` 不是传给 SGLang 的原生参数，而是在写表前生成的日志字段。

写表前的生成逻辑如下：

```python
if is_wideep_mla:
    log_mla_dtype = "fp8_block"
    log_kv_dtype = "fp8"
    log_gemm_type = "fp8_block"
else:
    log_mla_dtype = compute_dtype
    log_kv_dtype = kv_cache_dtype
    log_gemm_type = gemm_type
```

对应行号：`collector/sglang/collect_mla_module.py:696-711`。

这里有一个历史兼容点：`attn_type == "mla"` 的 WideEP MLA 路径会强制写成 `mla_dtype="fp8_block"`、`kv_cache_dtype="fp8"`、`gemm_type="fp8_block"`，用来兼容旧的 `collect_wideep_attn.py` 和现有 perf database 读取规则。这进一步说明 `mla_dtype` 不是干净的 runtime dtype 控制量，而是历史 perf 表索引字段。

### 3. SDK 侧：`mla_dtype` 有查询意义，所以不能简单删除

虽然 `mla_dtype` 不控制 collector 执行，但它不是完全无用字段。SDK 会把它作为 `FMHAQuantMode` 的 key 来组织和查询数据。

普通 context MLA：

```python
quant_mode = common.FMHAQuantMode[quant_mode]
context_mla_data[quant_mode][kv_cache_dtype][num_heads][s][b] = ...
```

对应行号：`src/aiconfigurator/sdk/perf_database.py:1071-1106`。

模块级 MLA：

```python
fmha_mode = common.FMHAQuantMode[row["mla_dtype"]]
mla_data[fmha_mode][kv_dtype][gemm_mode][num_heads][s][b] = ...
```

对应行号：`src/aiconfigurator/sdk/perf_database.py:1773-1785`。

WideEP context MLA：

```python
quant_mode = common.FMHAQuantMode[quant_mode]
wideep_context_mla_data[kernel_source][quant_mode][kv_cache_dtype][num_heads][s][b] = ...
```

对应行号：`src/aiconfigurator/sdk/perf_database.py:2136-2175`。

同时，`FMHAQuantMode` 中的 `fp8_block` 被明确标注为 SGLang WideEP 特化：

```python
fp8_block = QuantMapping(1, 2, "fp8_block")  # FIXME: specific for sglang wideep
```

对应行号：`src/aiconfigurator/sdk/common.py:729-736`。

DeepSeek 模型实现也直接承认了这个历史约定：现有归档的 SGLang module-level DeepSeek MLA 数据族是 `wideep_*_mla_perf.txt`，其历史 `mla_dtype` key 是 `fp8_block`，因此 module 查询会强制使用 `common.FMHAQuantMode.fp8_block`。

对应行号：`src/aiconfigurator/sdk/models/deepseek.py:124-131`。

### 小结

`mla_dtype` 的定位可以概括为：

- **不是 SGLang 原生 runtime 参数。**
- **不是 `collect_mla.py` 单算子执行 dtype 的控制源。**
- **不是 `collect_mla_module.py` 模块执行 dtype 的直接控制源。**
- **是 AIC perf database 的 FMHA quant/query key。**
- **在 WideEP MLA/module 数据中还承担历史兼容标签的角色，尤其是 `fp8_block`。**

因此它不是“完全冗余干扰项”，因为 SDK 查表依赖它；但它确实不是底层执行 dtype 的第一手来源。做实机和 collector 对齐时，应优先看：

- `kv_cache_dtype`：attention / KV cache / FA dtype 对齐主轴。
- `gemm_type` 或 `server_args.quantization`：投影 GEMM / 权重量化主轴。
- `mla_dtype`：perf 表索引语义，需结合 collector 写表逻辑解释，不能直接当作底层 kernel dtype。

## `collect_attn.py` 中的 `attn_dtype`

`collect_attn.py` 的 `attn_dtype` 和 `collect_mla.py` 的 `mla_dtype` 相似，但不完全一样。

### 1. `kv_cache_dtype` 是进入 SGLang backend 的主控制量

`run_attention_torch()` 根据 `use_fp8_kv_cache` 决定真实 KV cache torch dtype：

```python
kvtype = torch.float8_e4m3fn if use_fp8_kv_cache else torch.bfloat16
```

随后构造 mock model runner 时，把 SGLang 字符串 dtype 设置为：

```python
kv_cache_dtype="fp8_e4m3" if use_fp8_kv_cache else "auto"
```

并且把 KV pool 的 dtype 设为 `kvtype`：

```python
kv_pool = MHATokenToKVPool(..., dtype=kvtype, ...)
```

对应行号：`collector/sglang/collect_attn.py:300-315`、`collector/sglang/collect_attn.py:332-341`。

因此，和 MLA collector 一样，真正进入 SGLang attention backend 的主 dtype 控制轴仍然是 `kv_cache_dtype`。

### 2. `attn_dtype` 不是 backend 参数，但 context 下不是纯冗余

context 测试用例会生成三类组合：

```python
# BF16 attention
test_cases.append([... False, False, True])

# FP8 attention related cases
test_cases.append([... True, False, True])
test_cases.append([... True, True, True])
```

对应字段是：

```python
use_fp8_kv_cache, use_fp8_context_fmha, is_context_phase
```

对应行号：`collector/sglang/collect_attn.py:188-194`。

写表时：

```python
"attn_dtype": "fp8" if use_fp8_context_fmha else "bfloat16",
"kv_cache_dtype": "fp8" if use_fp8_kv_cache else "bfloat16",
```

对应行号：`collector/sglang/collect_attn.py:521-531`。

这里的 `attn_dtype` 没有被传给 `FlashAttentionBackend` 或 `ServerArgs`。但当 `is_context_phase and use_fp8_context_fmha` 为真时，collector 会在 benchmark 之前显式把 live q/k/v 转成 `kvtype`：

```python
if is_context_phase and use_fp8_context_fmha:
    q = q.to(kvtype)
    k = k.to(kvtype)
    v = v.to(kvtype)
```

对应行号：`collector/sglang/collect_attn.py:487-494`。

这个 cast 在 `benchmark_with_power()` 之前，因此 cast 成本不计入 latency；但它确实改变了进入 `layer(q, k, v, forward_batch)` 的 q/k/v dtype。由此可见：

- `attn_dtype` 不是 SGLang runtime 参数。
- `attn_dtype` 在 context collector 中不是完全冗余，因为它标识并触发了“计时前预先把 q/k/v 转 fp8”的采集模式。
- 其语义更像“context FMHA 输入是否预先按 fp8 准备”，而不是独立于 KV cache 的底层 kernel dtype 开关。

### 3. FA3 backend 内部仍会以显式 KV dtype 对齐 Q/K/V

以当前环境的 SGLang `FlashAttentionBackend.forward_extend()` 为例，若 `kv_cache_dtype_str != "auto"`、`layer.head_dim <= 256` 且 `fa_impl_ver != 4`，backend 会在 `layer(...)` 内部执行：

```python
q = q.to(self.kv_cache_dtype)
```

对应行号：`flashattention_backend.py:585-596`。

普通 MHA prefill 的 FA varlen 调用又会执行：

```python
k=k.view(...).to(q.dtype)
v=v.view(...).to(q.dtype)
```

对应行号：`flashattention_backend.py:806-818`。

因此，对于 H100/FA3 这类路径，`kv_cache_dtype=fp8` 时，即使 `attn_dtype` 记录为 `"bfloat16"`，核心 FA 调用前也可能在 backend 内部把 Q/K/V 对齐到 fp8。此时 `attn_dtype=bfloat16` 与 `attn_dtype=fp8` 的主要差别不是“核心 kernel 一定 bf16 vs fp8”，而是：

- `attn_dtype=bfloat16, kv_cache_dtype=fp8`：q/k/v 以 bf16 进入 `layer()`，内部 dtype 对齐成本更可能计入采集。
- `attn_dtype=fp8, kv_cache_dtype=fp8`：q/k/v 在计时前已预先转成 fp8，内部 cast 多数成为 no-op 或显著变轻，采集更接近纯 fp8 FMHA 子路径。

这也是为什么 `attn_dtype` 不能直接当作唯一底层 kernel dtype 来源；底层 dtype 判断仍要结合 `kv_cache_dtype` 和 backend 内部 cast 逻辑。

### 4. SDK 查表侧：context 使用 `attn_dtype`，decode 基本只看 `kv_cache_dtype`

`load_context_attention_data()` 会把 `attn_dtype` 读成 `FMHAQuantMode`，并作为 context attention 数据的第一层 key：

```python
quant_mode = common.FMHAQuantMode[quant_mode]
context_attention_data[quant_mode][kv_cache_dtype][...]
```

对应行号：`src/aiconfigurator/sdk/perf_database.py:926-966`。

`query_context_attention()` 也显式接收 `fmha_quant_mode`，再用它查询：

```python
attention_dict = self._context_attention_data[fmha_quant_mode][kvcache_quant_mode][...]
```

对应行号：`src/aiconfigurator/sdk/perf_database.py:4283-4406`。

但 generation attention 不同。`load_generation_attention_data()` 虽然读取了 `row["attn_dtype"]`，但之后没有把它转成 `FMHAQuantMode`，最终数据结构从 `kv_cache_dtype` 开始：

```python
generation_attention_data[kv_cache_dtype][kv_n][head_size][window_size][n][b][s] = ...
```

对应行号：`src/aiconfigurator/sdk/perf_database.py:1002-1044`。

`query_generation_attention()` 也只接收 `kvcache_quant_mode`，并在 SOL 中根据 KV cache dtype 推导 FMHA 量化模式：

```python
if kvcache_quant_mode == common.KVCacheQuantMode.fp8:
    quant_mode_gen = common.FMHAQuantMode.fp8
else:
    quant_mode_gen = common.FMHAQuantMode.bfloat16
```

对应行号：`src/aiconfigurator/sdk/perf_database.py:4432-4491`。

所以，`attn_dtype` 的定位是：

- **context attention：有实际采集边界意义，也有 SDK 查表意义，但不是 SGLang 原生 runtime 参数。**
- **generation attention：基本是日志/兼容字段；实际执行和查询主要由 `kv_cache_dtype` 决定。**
- **底层 FA/FA3 kernel dtype：仍需以 `kv_cache_dtype` 和 backend 内部 cast 逻辑为准，不能只看 CSV 里的 `attn_dtype`。**
