# Decode CUDA Graph Attention 后端选择分析

生成时间：2026-06-25

## 结论

本轮正式 decode CUDA Graph 实机结果中，SGLang 实际选择的全局 attention backend 是 `fa3`，不是 `flashmla`。这不是 CUDA Graph 特殊机制导致的退化，也不是 profile 解析时遗漏了 FlashMLA，而是 SGLang v0.5.9 在 H100/Hopper + MLA 架构模型上，`attention_backend=auto` 的默认选择规则就是 `fa3`。

因此，当前正式 decode 数据应理解为：`ForwardMode.DECODE` 命中 CUDA Graph，但 DeepSeek MLA 的吸收式 decode 路径底层 attention backend 继承了默认 `fa3`。早期 CSV 中同时出现的 `flashmla` 和 `fa3`，本质上是 kernel 名启发式分类结果；其中 `nvjet_tst_*` 只是被旧规则误归为 `flashmla`，并不能直接等价为 SGLang 的 `decode_attention_backend=flashmla`。

## 实机侧证据

正式归档数据来自：

- `bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/decode_manifest.csv`
- `bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/decode_csv/*__DecodeCudaGraphKernel汇总.csv`
- `bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/raw_runs/20260625_*decode_stage1*`

抽取 9 个正式 case 的 `container.stderr.log` 中 `server_args=ServerArgs(...)`，均得到：

| 字段 | 值 |
|---|---|
| `attention_backend` | `'fa3'` |
| `decode_attention_backend` | `None` |
| `prefill_attention_backend` | `None` |

同时日志中每个 case 均有真实 decode CUDA Graph 证据：

- `Decode batch ... cuda graph: True`
- `Prefill batch ... cuda graph: True` 为 0

对应 wrapper 行为也一致：`run_decode_stage1_batch.py` 中，case 的 `attention_backend` 默认为 `auto`；当值为 `auto/default/空` 时，不会向容器内脚本传 `--attention-backend`。因此本轮没有显式要求 `flashmla`，后端选择完全交给 SGLang `auto` 规则。

## SGLang v0.5.9 源码机制

源码来自 docker 镜像：

`booleimg.myaddr.io/lmsysorg/sglang:v0.5.9`

### 1. `auto` 在 Hopper + MLA 架构默认选择 `fa3`

`/sgl-workspace/sglang/python/sglang/srt/server_args.py`

在 `_handle_attention_backend_compatibility()` 中，SGLang 对默认 attention backend 的选择逻辑写明：

- MHA 架构在 Hopper 上默认 `fa3`。
- MLA 架构在 Hopper 上也默认 `fa3`。
- Blackwell MLA 架构才默认 `flashinfer`。
- 其他平台默认 `triton`。

关键代码位置：

- `server_args.py:1791-1794`：注释说明 MLA 架构在 Hopper 使用 FA3。
- `server_args.py:1819-1823`：`use_mla_backend` 分支下，`is_hopper_with_cuda_12_3()` 时设置 `self.attention_backend = "fa3"`。

所以本轮 H100/Hopper + DeepSeek-V3 MLA 架构，在没有显式指定 backend 时，解析为 `attention_backend='fa3'` 是 SGLang 0.5.9 的预期行为。

### 2. decode 未单独指定时会继承全局 `attention_backend`

同一文件中：

- `server_args.py:5095-5106`

`get_attention_backends()` 的规则是：

- 如果 `prefill_attention_backend` 为空，则 prefill 使用 `attention_backend`。
- 如果 `decode_attention_backend` 为空，则 decode 使用 `attention_backend`。

本轮 `decode_attention_backend=None`，因此 decode backend 继承全局 `attention_backend='fa3'`。

`model_runner.py:1751-1789` 进一步把 `get_attention_backends()` 的结果写入 global server args。由于 prefill 和 decode backend 字符串相同，不会创建 hybrid backend，而是实例化同一个 `fa3` backend。

### 3. DeepSeek decode 仍走 MLA forward method，但底层 attention backend 是 FA3

DeepSeek MLA 模型不是简单按 `fa3 == MHA`、`flashmla == MLA` 二分。

`/sgl-workspace/sglang/python/sglang/srt/models/deepseek_common/attention_backend_handler.py`

`_handle_attention_backend()` 中只有 `forward_mode.is_extend_without_speculative()` 也就是 prefill/extend 条件满足时，才可能返回：

- `AttnForwardMethod.MHA_ONE_SHOT`
- `AttnForwardMethod.MHA_CHUNKED_KV`

否则返回 `_dispatch_mla_subtype()`，即 MLA forward method。

因此对于 decode：

- 当前后端名是 `fa3`。
- 但 forward method 仍是 `AttnForwardMethod.MLA`。
- `DeepseekV2AttentionMLA.forward_prepare()` 会进入 `forward_absorb_prepare()`。
- `forward_absorb_core()` 中调用的是 `self.attn_mqa(...)`。

关键代码位置：

- `attention_backend_handler.py:81-96`
- `deepseek_v2.py:1327-1343`
- `deepseek_v2.py:1406-1422`
- `deepseek_v2.py:1763-1781`

这解释了一个容易混淆的点：当前不是 prefill 的 MHA 路径，也不是 `attn_mha` 主路径；它是 DeepSeek MLA 的吸收式 decode 路径，但该路径里的 RadixAttention backend 使用 `fa3`。

## 为什么旧 CSV 中还会看到 `flashmla`

当前 `DecodeCudaGraphKernel汇总.csv` 的 `implementation` 字段不是 SGLang 原生后端字段，而是我们在 `.self/task-localbench/formal-decode-stage1/run_decode_stage1_batch.py` 中按 kernel name 做的启发式分类。旧规则里：

```python
if "nvjet" in lowered:
    return "attention", "flashmla"
if "flashattn" in name or "flash::" in lowered:
    return "attention", "fa3"
```

因此旧结果里：

- `fa3` 行来自 `flash::prepare_varlen_num_blocks_kernel`、`FlashAttnFwdSm90`、`FlashAttnFwdCombine` 等 FA3 kernel 名。
- `flashmla` 行里混入了 `nvjet_tst_*` kernel，这只是旧解析器把 `nvjet` 名称误归到 FlashMLA。

但这个字段只是“kernel 名形态”的粗粒度标签，并不是 `ServerArgs.decode_attention_backend`。当前 9 个正式 case 的 server args 已经证明全局和 decode backend 均为 FA3。

从 kernel 序列看，典型 5 层 case 的 attention 类 kernel 呈现：

- `fa3`: 15 个，约等于每层 3 个 FA3 相关 kernel。
- `nvjet_tst_*`: 11 个，通常是每层 2 个加一个尾部较大 kernel；b32/p512 case 为 13 个。

这说明 CUDA Graph 内部仍存在若干 `nvjet_tst_*` kernel，但不能据此反推 SGLang 的 decode attention backend 已经切为 `flashmla`。这些 kernel 更适合作为 DeepSeek MLA decode 中的一类 attention 子 kernel 单独观察；后续与 AIC 对齐时，不应把旧解析器里的 `implementation=flashmla` 等同于 `query_generation_mla()` 的完整语义。

## 对本轮 AIC 对比口径的影响

当前 formal decode 数据更应该与 AIC 的 `generation_attention` 或 FA3 attention 相关采集口径优先比较，而不是直接与 `generation_mla` 绑定。

这也解释了前一轮对比中的现象：

- `generation_attention / bfloat16` 对 core attention 的 MAPE 约 `9.79%`。
- `generation_attention / fp8` 对 core attention 的 MAPE 约 `14.80%`。
- `generation_mla` 口径误差显著更大。

这并不矛盾：实机 decode 是 DeepSeek MLA 模型路径，但后端选择是 FA3；AIC `generation_mla` 表更接近显式 FlashMLA/MLA backend 的采集语义，而不是本轮 `auto -> fa3` 的 decode 后端语义。

## 后续建议

如果目标是采集“显式 FlashMLA decode backend”的实机数据，需要新开一组对照 run，而不是复用当前 formal decode 数据：

- 显式传入 `--attention-backend flashmla`，或更细地扩展 wrapper 支持 `decode_attention_backend=flashmla`。
- 注意 SGLang 源码中 FlashMLA 会强制 page size 为 64；需要检查 dummy engine、KV cache、CUDA Graph capture 是否仍能跑通。
- 对比脚本也需要把 `ServerArgs` 的真实 backend 与 kernel-name 分类分开记录，避免再次把 `nvjet` 启发式标签误读为 SGLang backend。

当前 formal decode 数据本身是有效的 decode CUDA Graph 数据，但它代表的是 SGLang v0.5.9 在 H100 上默认 `auto -> fa3` 的真实行为，而不是 FlashMLA decode backend 行为。
