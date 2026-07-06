# WideEP MLA qkv_a_proj 边界与重复建模分析

## 结论摘要

本轮核验的核心结论是：`qkv_a_proj` 在 SGLang 源码中并没有从 `self_attn` 模块所有权上移走；它仍然是 `DeepseekV2AttentionMLA/self_attn` 内部的 `fused_qkv_a_proj_with_mqa` 子模块。`LayerCommunicator` 只是通过 `qkv_latent_func=self.self_attn.prepare_qkv_latent` 取得一个延迟执行入口，用于调度或重叠执行该投影。真实 CUDA GEMM 仍由 `self_attn.prepare_qkv_latent()` 内部调用 `self.fused_qkv_a_proj_with_mqa(...)` 或 fused fast-path 触发。

AIC collector 的 `mla_module` 采集对这个入口做了特殊 mock：它把 `AttentionInputs` 中的 `qkv_latent_func` 替换为 `dummy_qkv_latent_func`，直接返回随机 qkv latent 张量。因此 WideEP MLA module 采集数据虽然以 `layers[test_layer].self_attn` 为模块入口，但实际绕过了真实 `qkv_a_proj` GEMM。

当前 SDK `WideEPDeepSeekModel` 中同时存在 `context_qkv_a_proj_gemm/generation_qkv_a_proj_gemm` 和 `context_downscale_gemm/generation_downscale_gemm` 两组形状相同的 `2112 x hidden_size` GEMM。结合 upstream PR 时间线看，这很可能是模型侧重复建模 bug，而不是 collector 模块边界本身错误。

## 1. SGLang 中 LayerCommunicator 与 self_attn 的关系

### self_attn 的职责：模块所有权与真实计算

在 SGLang `v0.5.9` 和 `v0.5.12` 的 `python/sglang/srt/models/deepseek_v2.py` 中，`fused_qkv_a_proj_with_mqa` 都定义在 `DeepseekV2AttentionMLA` 内部：

- `v0.5.9`: `DeepseekV2AttentionMLA` 约在 1067 行，`self.fused_qkv_a_proj_with_mqa = ReplicatedLinear(...)` 约在 1122 行。
- `v0.5.12`: `DeepseekV2AttentionMLA` 约在 1309 行，`self.fused_qkv_a_proj_with_mqa = ReplicatedLinear(...)` 约在 1371 行。

对应逻辑是：

```python
self.fused_qkv_a_proj_with_mqa = ReplicatedLinear(
    self.hidden_size,
    self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
    ...
)
```

真实执行入口也仍在 `self_attn` 内部，即 `prepare_qkv_latent()`：

```python
def prepare_qkv_latent(self, hidden_states, forward_batch):
    ...
    qkv_latent = self.fused_qkv_a_proj_with_mqa(hidden_states)[0]
    return qkv_latent
```

在小 batch / decode 等特定条件下，SGLang 还可能使用 `dsv3_fused_a_gemm(hidden_states, self.fused_qkv_a_proj_with_mqa.weight.T)` 的 fast-path。但无论走普通 `ReplicatedLinear` 还是 fused kernel，权重和执行方法仍属于 `self_attn`。

因此从模块归属和 CUDA 算子归属看，`fused_qkv_a_proj_with_mqa` 仍是 `self_attn` 的子模块。

### LayerCommunicator 的职责：调度入口与数据通路

同一份源码中，Decoder Layer 构造 `LayerCommunicator` 时把 `self_attn.prepare_qkv_latent` 作为函数指针传入：

```python
self.layer_communicator = LayerCommunicator(
    ...,
    qkv_latent_func=self.self_attn.prepare_qkv_latent,
)
```

在 `python/sglang/srt/layers/communicator.py` 中，`AttentionInputs` 保存这个函数：

```python
class AttentionInputs:
    def __init__(self, hidden_states, forward_batch, qkv_latent_func):
        self.hidden_states_local = hidden_states
        self.forward_batch = forward_batch
        self.qkv_latent_func = qkv_latent_func
        self.qkv_latent_ = None
```

随后 `fetch_qkv_latent()` 才真正调用它：

```python
def fetch_qkv_latent(self):
    if self.qkv_latent_ is not None:
        return self.qkv_latent_
    self.qkv_latent_ = self.qkv_latent_func(
        self.hidden_states_local, self.forward_batch
    )
    ...
    return self.qkv_latent_
```

这解释了看起来矛盾的现象：

- `LayerCommunicator` 参与了 `qkv_a_proj` 的执行时机和数据流调度。
- 但它不拥有 `fused_qkv_a_proj_with_mqa` 权重，也不实现 GEMM。
- 真实 CUDA GEMM 是通过 communicator 持有的函数指针回调 `self_attn.prepare_qkv_latent()` 触发的。

所以更准确的描述应是：SGLang 将 `qkv_a_proj` 的调用入口暴露给 communicator，以便在 layer pipeline / TP / overlap 逻辑中延迟或提前获取 qkv latent；但 `qkv_a_proj` 的模块所有权和 kernel 执行实现仍在 `self_attn`。

## 2. collector 中 mla_module 对 qkv_a_proj 的特殊处理

本地 `collector/sglang/collect_mla_module.py` 中，`mla_module` 采集入口取的是：

```python
attention_module = model_runner.model.model.layers[test_layer].self_attn
```

也就是说，从 Python module 入口看，它确实直接 benchmark `self_attn`。

但 collector 随后构造了一个假的 qkv latent 生成函数：

```python
qkv_latent_dim = q_lora_rank + kv_lora_rank + qk_rope_head_dim

def dummy_qkv_latent_func(h, fb):
    return torch.randn(h.shape[0], qkv_latent_dim, dtype=h.dtype, device=h.device)
```

在 prefill 和 decode 测试中，它把这个 dummy 函数塞进 `AttentionInputs`：

```python
attn_inputs = AttentionInputs(hidden_states, forward_batch, dummy_qkv_latent_func)
get_attn_tp_context().set_attn_inputs(attn_inputs)
```

于是当 `self_attn` 内部通过 `get_attn_tp_context().fetch_qkv_latent()` 获取 qkv latent 时，最终调用的不是 `self_attn.prepare_qkv_latent()`，而是 collector 提供的 `dummy_qkv_latent_func()`。

这带来的结果是：

- `fused_qkv_a_proj_with_mqa` 的真实 GEMM 没有执行。
- `q_a/kv_a/rope` 合并投影的输入张量被随机 latent mock 掉。
- WideEP `wideep_context_mla_perf.txt` / `wideep_generation_mla_perf.txt` 采到的更接近“已经拥有 qkv latent 后的 MLA attention 体内成本”，而不是完整 `self_attn` 包含 qkv_a 投影的端到端成本。

因此，collector 的边界有两层含义：

- 代码入口边界：调用 `layers[test_layer].self_attn`。
- 实际计算边界：通过 mock `qkv_latent_func` 排除了 `qkv_a_proj` GEMM。

这也解释了为什么单看 `attention_module = self_attn` 会误以为 collector 包含完整 `self_attn`，但从实际执行看 `qkv_a_proj` 已经被 mock 掉。

## 3. SDK WideEPDeepSeekModel 的重复建模风险

当前本地 `src/aiconfigurator/sdk/models/deepseek.py` 中，`WideEPDeepSeekModel` 先显式加入独立 qkv_a GEMM：

```python
ops.GEMM(
    "context_qkv_a_proj_gemm",
    self._num_layers,
    1536 + 512 + 64,
    h,
    gemm_quant_mode,
    scale_num_tokens=tp_size,
)
```

随后在 context attention op 序列中又加入：

```python
ops.GEMM("context_downscale_gemm", self._num_layers, 2112, h, gemm_quant_mode)
ops.WideEPContextMLA(...)
```

generation 路径也类似：

```python
ops.GEMM("generation_qkv_a_proj_gemm", ..., 1536 + 512 + 64, h, gemm_quant_mode)
ops.GEMM("generation_downscale_gemm", ..., 2112, h, gemm_quant_mode)
ops.WideEPGenerationMLA(...)
```

这两组 GEMM 的关键维度相同：`1536 + 512 + 64 = 2112`。结合 DeepSeek MLA 结构，`2112 x hidden_size` 正是 `fused_qkv_a_proj_with_mqa` / qkv_a fused projection 的输出维度。若二者都代表同一个 fused qkv_a 投影，则当前 WideEP model 会重复计入该 GEMM。

### upstream PR 背景

远程只读核验到的 upstream 时间线如下：

- PR [#476](https://github.com/ai-dynamo/aiconfigurator/pull/476) `fix: model qkv_a_proj as standalone GEMM in WideEP pipeline` 已合并。它加入 `context_qkv_a_proj_gemm` 和 `generation_qkv_a_proj_gemm`，理由是 SGLang `>=0.5.6` 中 `qkv_a_proj` 通过 communicator/lazy `qkv_latent_func` 在 MLA attention forward 外执行，因此不应包含在 `WideEPContextMLA` / `WideEPGenerationMLA` 内。
- PR [#657](https://github.com/ai-dynamo/aiconfigurator/pull/657) `feat: implemented ds-v32 nsa data collection for sglang and added back the missing gemm` 已合并。它后续又在 WideEPDeepSeekModel 中加入 `context_downscale_gemm` 和 `generation_downscale_gemm`。
- 当前 upstream `main` 和 `v0.9.0` 仍同时保留 `context_qkv_a_proj_gemm` 与 `context_downscale_gemm`，也同时保留 `generation_qkv_a_proj_gemm` 与 `generation_downscale_gemm`。
- PR [#1172](https://github.com/ai-dynamo/aiconfigurator/pull/1172) 的 Rust migration 虽然是 closed，不能作为已合并修复依据，但其 diff 明确称是 `WideEPDeepSeekModel` 的 apple-to-apple port，并同样包含 `context_qkv_a_proj_gemm + context_downscale_gemm`。这说明 upstream 至少没有把当前双 GEMM 行为作为已知错误反向修正。

### 对 #476 描述的修正理解

`#476` 中“qkv_a_proj was moved outside MLA attention forward via communicator”的说法容易被误读。结合 SGLang 0.5.9/0.5.12 源码，更精确的说法应是：

- `qkv_a_proj` 的调用时机被 communicator 接管或延迟获取。
- `qkv_a_proj` 的权重、模块归属、实际 GEMM 实现仍在 `self_attn.prepare_qkv_latent()` 内。
- collector 的 WideEP MLA module 数据通过 dummy latent 排除了真实 qkv_a GEMM，因此 SDK 确实需要在 module 数据之外补一个 qkv_a GEMM。
- 但在已经补了 `context_qkv_a_proj_gemm` 的前提下，再加入同形状的 `context_downscale_gemm`，就很可能重复。

## 当前判断

`WideEPContextMLA` / `WideEPGenerationMLA` 的 collector 数据边界本身并不一定错：collector 有意 mock 掉 qkv latent，采集的是不含真实 qkv_a GEMM 的 MLA 内部成本。真正的问题更可能出在 SDK model 的 op 序列：它既承认 `qkv_a_proj` 被 collector 排除，需要独立补 `context_qkv_a_proj_gemm`，又额外保留一个同形状、同语义嫌疑很高的 `context_downscale_gemm`。

建议后续若向 upstream 提 issue，可聚焦在以下最小问题上：

- `context_qkv_a_proj_gemm` 与 `context_downscale_gemm` 在 WideEPDeepSeekModel 中是否代表同一 `fused_qkv_a_proj_with_mqa`？
- 如果不是同一项，`context_downscale_gemm` 在 SGLang DeepSeek MLA/WideEP 执行路径中对应哪个真实模块、哪个 NVTX/trace 事件、哪个权重名？
- 如果是同一项，应删除或重命名其中一个，并补充单元测试验证 WideEP op list 不重复计入 `2112 x hidden_size` fused_a GEMM。

