# KV Cache FP8 对 Prefill/Decode Attention 算子序列的影响

## 结论摘要

本轮对比使用同一套 SGLang v0.5.9 Engine 级实机脚本，比较默认 `kv_cache_dtype=auto` 与显式 `kv_cache_dtype=fp8_e4m3` 后 attention 路径的 CUDA kernel 变化。

核心结论：

- 之前未显式指定 `kv_cache_dtype` 的 run，`engine_kwargs.json` 中没有 `kv_cache_dtype` 字段；本地 DeepSeek-V3 config 为 `torch_dtype=bfloat16`，且没有 KV cache FP8 声明，因此默认 `auto` 可按 bf16 KV cache 理解。
- Prefill 走 `attn_mha` / FA3 路径。开启 FP8 KV 后，attention 子模块内新增 Q/K/V 或中间张量的 `float8_copy_kernel_cuda` 相关转换 kernel，FA 主 kernel 的模板 dtype 从 `cutlass::bfloat16_t` 切到包含 `cutlass::float_e4m3_t` 的版本。
- Decode 走 FlashMLA 路径。开启 FP8 KV 后，每层新增 `per_tensor_quant_fp8_kernel`，FlashMLA 主 kernel 签名从 bf16 `DecodingParams` 切到含 `cutlass::float_e4m3_t` / `DecodingParams_fp8` 的 FP8 版本。
- trace 中一直存在的 `per_token_group_quant_8bit_kernel` / `sm90_fp8_gemm_1d2d_impl` 主要来自 DeepSeek FP8 权重 GEMM 路径，不应被误判为 KV cache FP8 的证据。

## 数据来源

| 场景 | KV cache | 形状 | attention 路径 | run 目录 |
|---|---:|---|---|---|
| Prefill 对照 | auto/bf16 | `b1 fresh=8192 prefix=8192` | `attn_mha` / FA3 | `.self/task-localbench/formal-prefill-stage1/output/20260618_112409_prefill_stage1_req1_equal_prefix_regular_b1_f8192_p8192_b1_fvar8192_pvar8192_layers5_backend_auto_cg_off_pcg_off_tp1_marker_on_profile_nsys` |
| Prefill FP8 | `fp8_e4m3` | `b1 fresh=8192 prefix=8192` | `attn_mha` / FA3 | `.self/task-localbench/formal-prefill-stage1/output/20260708_095702_prefill_stage1_kvfp8_probe_b1_f8192_p8192_b1_fvar8192_pvar8192_layers5_backend_auto_decodebackend_auto_kv_fp8_e4m3_cg_off_pcg_off_tp1_marker_on_profile_nsys` |
| Decode 对照 | auto/bf16 | `b4 decode=1 prefix=8192` | FlashMLA，显式 `decode_attention_backend=flashmla` | `.self/task-localbench/formal-decode-stage1/output/20260626_154701_decode_stage1_decode_flashmla_b4p8192_cg_on_b4_f1_p8192_layers5_backend_auto_decodebackend_flashmla_cg_on_pcg_off_tp1_marker_on_profile_nsys` |
| Decode FP8 | `fp8_e4m3` | `b4 decode=1 prefix=8192` | FlashMLA，显式 `decode_attention_backend=flashmla` | `.self/task-localbench/formal-decode-stage1/output/20260708_101348_prefill_stage1_decode_kvfp8_flashmla_b4p8192_cg_on_b4_f1_p8192_layers5_backend_auto_decodebackend_flashmla_kv_fp8_e4m3_cg_on_pcg_off_tp1_marker_on_profile_nsys` |
| Decode 默认对照 | auto/bf16 | `b4 decode=1 prefix=8192` | 默认 decode backend，实测为 FA3 split/combine `device_kernel` | `.self/task-localbench/formal-decode-stage1/output/20260625_143152_decode_stage1_decode_b4p8192_cg_on_b4_f1_p8192_layers5_backend_auto_cg_on_pcg_off_tp1_marker_on_profile_nsys` |
| Decode 默认 FP8 | `fp8_e4m3` | `b4 decode=1 prefix=8192` | 默认 decode backend，实测仍为 FA3 split/combine `device_kernel` | `.self/task-localbench/formal-decode-stage1/output/20260708_105525_prefill_stage1_decode_kvfp8_default_b4p8192_cg_on_b4_f1_p8192_layers5_backend_auto_decodebackend_auto_kv_fp8_e4m3_cg_on_pcg_off_tp1_marker_on_profile_nsys` |

数据抽取口径：

- Prefill：使用 `MLA时延拆解.csv` 中每层 `attn_mha` 子模块的 `first_kernel_start_ns -> last_kernel_end_ns` 范围，并回查 Nsight SQLite `CUPTI_ACTIVITY_KIND_KERNEL` 的 demangled kernel 名。
- Decode：由于 decode CUDA graph 结果不总是生成 `MLA时延拆解.csv`，直接使用 `report.sqlite` 的 `CUPTI_ACTIVITY_KIND_KERNEL`，围绕每层 `flash_fwd_splitkv_mla_kernel` 抽取 attention 相关 kernel 窗口。
- 所有统计均为 5 层累计；单层平均由累计值除以 5 得到。

## Prefill: FA3 `attn_mha` 路径

### 算子序列变化

默认 bf16 KV cache 时，每层 `attn_mha` 的核心 CUDA 序列为：

| 顺序 | kernel | 说明 |
|---:|---|---|
| 1 | `prepare_varlen_num_blocks_kernel` | FA3 varlen block metadata 准备 |
| 2 | `device_kernel` | FA3 主 attention kernel，demangled 签名含 `cutlass::bfloat16_t` |

显式 `kv_cache_dtype=fp8_e4m3` 后，每层 `attn_mha` 的核心 CUDA 序列变为：

| 顺序 | kernel | 说明 |
|---:|---|---|
| 1 | `vectorized_elementwise_kernel` | `float8_copy_kernel_cuda`，bf16 到 FP8 转换/拷贝相关 |
| 2 | `vectorized_elementwise_kernel` | `float8_copy_kernel_cuda`，第二个向量化转换/拷贝 |
| 3 | `elementwise_kernel` | `float8_copy_kernel_cuda`，非向量化转换/拷贝 |
| 4 | `prepare_varlen_num_blocks_kernel` | FA3 varlen block metadata 准备 |
| 5 | `device_kernel` | FA3 主 attention kernel，demangled 签名含 `cutlass::float_e4m3_t` |

### 5 层累计耗时对比

| kernel | bf16 KV count | bf16 KV total ms | FP8 KV count | FP8 KV total ms | 变化说明 |
|---|---:|---:|---:|---:|---|
| `device_kernel` | 5 | 51.693598 | 5 | 45.073414 | FA3 主 kernel 切到 FP8 后下降约 12.8% |
| `prepare_varlen_num_blocks_kernel` | 5 | 0.017344 | 5 | 0.015840 | 基本不变 |
| `vectorized_elementwise_kernel` | 0 | 0.000000 | 10 | 3.143165 | FP8 新增转换/拷贝 |
| `elementwise_kernel` | 0 | 0.000000 | 5 | 3.529439 | FP8 新增转换/拷贝 |
| 合计 | 10 | 51.710942 | 25 | 51.761858 | 总量几乎持平 |

单层平均：

| 指标 | bf16 KV | FP8 KV | 差异 |
|---|---:|---:|---:|
| attention 子模块 GPU kernel sum | 10.342188 ms | 10.352372 ms | +0.010184 ms |
| FA3 主 kernel `device_kernel` | 10.338720 ms | 9.014683 ms | -1.324037 ms |
| FP8 转换/拷贝开销 | 0.000000 ms | 1.334521 ms | +1.334521 ms |

解释：

- FP8 KV 使 FA3 主计算 kernel 更快，但转换/拷贝 kernel 几乎抵消了收益。
- 因此在该 `b1 f8192 p8192` prefill case 上，总 attention 子模块耗时基本持平。
- demangled 签名层面，bf16 run 的 FA3 主 kernel 包含 `cutlass::bfloat16_t`；FP8 run 的 FA3 主 kernel 包含 `cutlass::float_e4m3_t`，说明核心 attention 计算路径确实发生 dtype 变化。

## Decode: FlashMLA 路径

### 算子序列变化

默认 bf16 KV cache 时，每层 FlashMLA attention 相关序列为：

| 顺序 | kernel | 说明 |
|---:|---|---|
| 1 | `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel` | Q/K RoPE，签名含 `__nv_bfloat16` |
| 2 | `CatArrayBatchedCopy` | KV/index 数据整理 |
| 3 | `CatArrayBatchedCopy` | KV/index 数据整理 |
| 4 | `index_elementwise_kernel` | index/metadata 处理 |
| 5 | `flash_fwd_splitkv_mla_kernel` | FlashMLA split-kv 主 kernel，签名含 `cutlass::bfloat16_t` |
| 6 | `flash_fwd_mla_combine_kernel` | FlashMLA combine kernel，签名含 `cutlass::bfloat16_t` |

显式 `kv_cache_dtype=fp8_e4m3` 后，每层 FlashMLA attention 相关序列为：

| 顺序 | kernel | 说明 |
|---:|---|---|
| 1 | `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel` | Q/K RoPE，仍为 bf16 输入侧处理 |
| 2 | `CatArrayBatchedCopy` | KV/index 数据整理 |
| 3 | `CatArrayBatchedCopy` | KV/index 数据整理 |
| 4 | `vectorized_elementwise_kernel` | `float8_copy_kernel_cuda`，转换/拷贝相关 |
| 5 | `index_elementwise_kernel` | index/metadata 处理 |
| 6 | `per_tensor_quant_fp8_kernel` | FP8 KV/attention 前处理新增量化 kernel |
| 7 | `flash_fwd_splitkv_mla_kernel` | FP8 FlashMLA 主 kernel，签名含 `cutlass::float_e4m3_t` 与 `DecodingParams_fp8` |
| 8 | `flash_fwd_splitkv_mla_combine_kernel` | FP8 FlashMLA combine kernel，签名含 `DecodingParams_fp8` |

### 5 层累计耗时对比

| kernel | bf16 KV count | bf16 KV total ms | FP8 KV count | FP8 KV total ms | 变化说明 |
|---|---:|---:|---:|---:|---|
| `flash_fwd_splitkv_mla_kernel` | 5 | 0.168000 | 5 | 0.117312 | 主 FlashMLA kernel 切到 FP8 后下降约 30.2% |
| combine kernel | 5 | 0.111263 | 5 | 0.023264 | kernel 名从 `flash_fwd_mla_combine_kernel` 变为 `flash_fwd_splitkv_mla_combine_kernel`，耗时明显下降 |
| `CatArrayBatchedCopy` | 10 | 0.027392 | 10 | 0.027104 | 基本不变 |
| `index_elementwise_kernel` | 5 | 0.016224 | 5 | 0.017216 | 基本不变 |
| `BatchQKApplyRotary...` | 5 | 0.009568 | 5 | 0.010240 | 基本不变 |
| `per_tensor_quant_fp8_kernel` | 0 | 0.000000 | 5 | 0.008640 | FP8 新增 |
| `vectorized_elementwise_kernel` | 1 | 0.001664 | 5 | 0.007392 | FP8 下变为每层稳定出现 |
| 合计 | 31 | 0.334111 | 40 | 0.211168 | attention 局部窗口下降约 36.8% |

单层平均：

| 指标 | bf16 KV | FP8 KV | 差异 |
|---|---:|---:|---:|
| FlashMLA attention 局部窗口 | 0.066822 ms | 0.042234 ms | -0.024588 ms |
| FlashMLA split 主 kernel | 0.033600 ms | 0.023462 ms | -0.010138 ms |
| FlashMLA combine kernel | 0.022253 ms | 0.004653 ms | -0.017600 ms |
| FP8 新增量化/转换 | 0.000333 ms | 0.003206 ms | +0.002873 ms |

解释：

- Decode FlashMLA 的 FP8 KV 收益更直接：新增 `per_tensor_quant_fp8_kernel` 与 `float8_copy_kernel_cuda` 开销很小，主 split-kv 和 combine kernel 均明显变快。
- FP8 decode 的主 kernel demangled 签名包含 `cutlass::float_e4m3_t` 与 `DecodingParams_fp8`；bf16 decode 则是 `cutlass::bfloat16_t` 与普通 `DecodingParams`。
- 由于 decode 是 CUDA graph capture/replay 下的 1-token profile，绝对耗时很小，单次 trace 的相对比例需谨慎解读；但算子序列与 dtype 变化是清晰稳定的。

## Decode: 默认后端路径

本节补充未指定 `decode_attention_backend=flashmla` 的默认 decode 路径。该路径并没有落到 FlashMLA，而是 FA3 风格的 split/combine `device_kernel` 序列。

默认 bf16 KV cache 时，每层 attention 相关序列为：

| 顺序 | kernel | 说明 |
|---:|---|---|
| 1 | `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel` | Q/K RoPE，签名含 `__nv_bfloat16` |
| 2 | `set_mla_kv_buffer_kernel` | 写入/整理 MLA KV buffer |
| 3 | `prepare_varlen_num_blocks_kernel` | FA3 varlen block metadata 准备 |
| 4 | `device_kernel` | FA3 split-kv 主 kernel，签名含 `cutlass::bfloat16_t` |
| 5 | `device_kernel` | FA3 combine kernel，签名含 `cutlass::bfloat16_t` |

显式 `kv_cache_dtype=fp8_e4m3` 后，默认 decode 路径仍然没有切到 FP8 attention 主 kernel。每层序列变为：

| 顺序 | kernel | 说明 |
|---:|---|---|
| 1 | `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel` | Q/K RoPE，签名含 `__nv_bfloat16` |
| 2 | `vectorized_elementwise_kernel` | `float8_copy_kernel_cuda`，FP8 相关转换/拷贝 |
| 3 | `elementwise_kernel` | `float8_copy_kernel_cuda`，FP8 相关转换/拷贝 |
| 4 | `set_mla_kv_buffer_kernel` | 写入/整理 MLA KV buffer |
| 5 | `unrolled_elementwise_kernel` x4 | `direct_copy_kernel_cuda`，大体量 copy/cast；每个约 2.02 ms |
| 6 | `prepare_varlen_num_blocks_kernel` | FA3 varlen block metadata 准备 |
| 7 | `device_kernel` | FA3 split-kv 主 kernel，仍为 `cutlass::bfloat16_t` |
| 8 | `device_kernel` | FA3 combine kernel，仍为 `cutlass::bfloat16_t` |

5 层累计耗时对比：

| kernel | bf16 KV count | bf16 KV total ms | FP8 KV count | FP8 KV total ms | 变化说明 |
|---|---:|---:|---:|---:|---|
| `device_kernel` | 10 | 0.190561 | 10 | 0.207680 | 主/combined FA3 kernel 仍为 bf16，未切 FP8，且略慢 |
| `prepare_varlen_num_blocks_kernel` | 5 | 0.015872 | 5 | 0.016704 | 基本不变 |
| `BatchQKApplyRotary...` | 5 | 0.009472 | 5 | 0.009120 | 基本不变 |
| `set_mla_kv_buffer_kernel` | 5 | 0.007008 | 5 | 0.005632 | 基本不变 |
| `vectorized_elementwise_kernel` | 0 | 0.000000 | 6 | 0.008960 | FP8 新增/增加 |
| `elementwise_kernel` | 0 | 0.000000 | 5 | 0.007456 | FP8 新增 |
| `unrolled_elementwise_kernel` | 0 | 0.000000 | 20 | 40.501065 | FP8 新增的大体量 copy/cast |
| 合计 | 25 | 0.222913 | 56 | 40.756617 | 默认 FA3 decode + FP8 KV 出现显著额外开销 |

单层平均：

| 指标 | bf16 KV | FP8 KV | 差异 |
|---|---:|---:|---:|
| 默认 decode attention 局部窗口 | 0.044583 ms | 8.151323 ms | +8.106740 ms |
| FA3 split/combine `device_kernel` | 0.038112 ms | 0.041536 ms | +0.003424 ms |
| FP8 相关新增 copy/cast | 0.000000 ms | 8.103496 ms | +8.103496 ms |

解释：

- 默认 decode backend 在该形状下实测为 FA3 split/combine 路径，不是 FlashMLA。
- 与显式 FlashMLA 不同，默认 FA3 decode 在 `kv_cache_dtype=fp8_e4m3` 下没有把主 attention kernel 切到 e4m3 模板；两个 `device_kernel` 的 demangled 签名仍含 `cutlass::bfloat16_t`。
- FP8 KV 在该路径下主要表现为 attention 前插入额外 copy/cast，尤其是 4 个约 2 ms 的 `unrolled_elementwise_kernel`。这更像“为 bf16 FA3 decode 准备输入”的转换路径，而不是直接使用 FP8 KV 进行核心 attention 计算。
- 因此默认 FA3 decode + FP8 KV 对这个 `b4 p8192` case 是明显不利的；如果目标是观察 FP8 KV 下的高效 decode attention，应显式使用 FlashMLA 后端，或确认 SGLang 的自动后端选择条件是否会选择 FlashMLA。

### 源码解释：为什么 Prefill FA3 能走 FP8，而 Decode 默认 FA3 会转回 bf16

该现象不是硬件层面“decode 必须 bf16”，而是 SGLang v0.5.9 中 DeepSeek MLA 在不同阶段进入了不同的 FA3 子路径。

1. 默认后端选择：`server_args.py::_handle_attention_backend_compatibility()` 中，Hopper + MLA 架构默认选择 `attention_backend="fa3"`；`get_attention_backends()` 中若未显式设置 `decode_attention_backend`，decode 会继承同一个 `attention_backend`。因此本轮“默认 decode”实际是 FA3，而不是 FlashMLA。

2. DeepSeek 调度分支：`deepseek_common/attention_backend_handler.py::_handle_attention_backend()` 对 extend/prefill 在满足 prefix/chunk 条件时返回 `MHA_ONE_SHOT` 或 `MHA_CHUNKED_KV`，而 decode 会返回 `_dispatch_mla_subtype()`，即 `AttnForwardMethod.MLA`。`deepseek_v2.py::forward_prepare()/forward_core()` 随后分别进入 MHA 或 absorbed MLA 的 core。

3. 两个 RadixAttention 的形状不同：`deepseek_v2.py` 中 `attn_mqa = RadixAttention(num_kv_heads=1, head_dim=kv_lora_rank + qk_rope_head_dim, v_head_dim=kv_lora_rank)`，DeepSeek-V3 典型为 `512 + 64 = 576`；`attn_mha = RadixAttention(num_kv_heads=num_local_heads, head_dim=qk_nope_head_dim + qk_rope_head_dim, v_head_dim=v_head_dim)`，典型为 `128 + 64 = 192`。

4. FA3 FP8 guard：`flashattention_backend.py::forward_extend()` 和 `forward_decode()` 都只有在 `kv_cache_dtype_str != "auto"` 且 `layer.head_dim <= 256` 时才执行 `q = q.to(self.kv_cache_dtype)`、`q_rope = q_rope.to(self.kv_cache_dtype)` 等转换。Prefill MHA 使用 `attn_mha.head_dim=192`，因此能把 q/k/v 转到 FP8 并进入 FP8 FA3 主 kernel。Decode absorbed MLA 使用 `attn_mqa.head_dim=576`，不满足该 guard，Q 仍保持 bf16。

5. 大 copy/cast 的直接来源：`flashattention_backend.py::forward_decode()` 的 absorbed MLA 分支中有 `kv_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id).to(q.dtype)`。在 `kv_cache_dtype=fp8_e4m3` 且 `q.dtype` 仍为 bf16 时，这行会把整块 paged latent KV cache 从 FP8 转回 bf16，然后再拆为 `k_rope_cache` 和 `c_kv_cache` 调 `flash_attn_with_kvcache(q=q_rope, qv=q_nope, ...)`。Nsight 中每层约 4 个、单个约 2 ms 的 `unrolled_elementwise_kernel/direct_copy_kernel_cuda` 正对应这种大体量 dtype copy/cast。

6. 底层 kernel 覆盖也支持这个判断：`sgl-kernel/tests/test_flash_attention.py` 对 FP8 FA3 的测试限制了 `v head dim == qk head dim`，注释写明 FP8 不支持 `v head dim != qk head dim`；而 DeepSeek absorbed MLA 正是 `q_rope` 维度 64、latent value/qv 维度 512 的组合。换言之，默认 FA3 absorbed MLA 不是为 DeepSeek decode 的 FP8 latent KV 高效路径设计的。

显式 `decode_attention_backend=flashmla` 则走另一套专用实现：`flashmla_backend.py` 会根据 `model_runner.kv_cache_dtype` 判断 `is_fp8_kvcache`，FP8 时对 Q 执行 `scaled_fp8_quant()`，再调用 `flash_mla_with_kvcache(..., descale_q=..., descale_k=...)`。这一路径支持 DeepSeek MLA decode 的 FP8 KV 语义，因此 trace 中能看到 `per_tensor_quant_fp8_kernel` 和含 `cutlass::float_e4m3_t` / `DecodingParams_fp8` 的 FlashMLA 主 kernel，且没有默认 FA3 路径里的大规模 `direct_copy_kernel_cuda`。

### `direct_copy_kernel_cuda` 的性质确认

可以把 `void at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda...>` 明确归类为 PyTorch 通用 TensorIterator copy/cast kernel，而不是 FA3、FlashMLA 或 SGLang 自定义 attention kernel。

本次 Nsight demangled 名字中包含：

- `at::native::direct_copy_kernel_cuda`：PyTorch native CUDA copy kernel。
- `at::native::memory::LoadWithCast` 与 `at::native::memory::StoreWithCast`：读写两侧带 dtype cast。
- `lambda(c10::BFloat16)`：输出侧元素类型为 bf16。

因此它表示“从一个输入 tensor 逐元素读取、必要时转换 dtype、再写入输出 tensor”的通用 PyTorch 路径。常见 Python 触发源包括 `tensor.to(dtype)`、跨 dtype 的 `copy_()`、或某些需要 materialize 新 dtype tensor 的操作。仅凭 kernel 名不能唯一反推出具体是哪一行 Python，但结合本 case 的源码和时序，可以把它归因到 `flashattention_backend.py::forward_decode()` absorbed MLA 分支的 `get_key_buffer(...).to(q.dtype)`：此时 KV cache 物理存储为 `fp8_e4m3`，而 `attn_mqa.head_dim=576` 导致 Q 未被转成 FP8、仍为 bf16，于是 `.to(q.dtype)` 会对整块 latent KV cache 做 FP8 -> bf16 materialization。

所以本 case 的结论应固定为：这些高时延 `unrolled_elementwise_kernel/direct_copy_kernel_cuda` 不是 attention 核心计算，也不是 FlashMLA 风格的专用 FP8 反量化融合路径；它们是默认 FA3 absorbed MLA decode 为了喂给 bf16 FA3 kernel 而触发的 PyTorch 通用大块 copy/cast。

## 新增/替换算子清单

| 阶段 | FP8 KV 新增/替换 | 作用判断 |
|---|---|---|
| Prefill | `vectorized_elementwise_kernel` x2/layer | `float8_copy_kernel_cuda`，bf16 -> FP8 转换/拷贝 |
| Prefill | `elementwise_kernel` x1/layer | `float8_copy_kernel_cuda`，bf16 -> FP8 转换/拷贝 |
| Prefill | `device_kernel` dtype 改变 | FA3 主 attention kernel 从 bf16 模板切到 FP8/e4m3 模板 |
| Decode | `per_tensor_quant_fp8_kernel` x1/layer | FlashMLA 前的 FP8 量化前处理 |
| Decode | `vectorized_elementwise_kernel` x1/layer | `float8_copy_kernel_cuda`，转换/拷贝相关 |
| Decode | `flash_fwd_mla_combine_kernel` -> `flash_fwd_splitkv_mla_combine_kernel` | FP8 FlashMLA combine 路径变体 |
| Decode | `flash_fwd_splitkv_mla_kernel` dtype 改变 | FlashMLA 主 kernel 从 bf16 模板切到含 e4m3 的 FP8 模板 |
| Decode 默认后端 | `unrolled_elementwise_kernel` x4/layer | 默认 FA3 decode + FP8 KV 下新增大体量 `direct_copy_kernel_cuda` copy/cast |
| Decode 默认后端 | `device_kernel` dtype 不变 | 主 attention kernel 仍为 bf16 FA3 split/combine，没有切到 FP8 模板 |

## 对 AIC/collector 对照的含义

1. 如果 AIC 数据库/collector 以 `kv_cache_dtype` 作为 attention 精度控制字段，那么实机现象与该模型一致：KV cache 为 FP8 时，prefill FA3 和 decode FlashMLA 的核心 kernel 都会切到 FP8 相关模板。
2. Prefill 的 FP8 KV 不一定带来端到端 attention 子模块收益，因为 FA 主 kernel 降低的时间会被额外转换/拷贝开销抵消。
3. Decode 的 FP8 KV 对 FlashMLA 更有利，本次 `b4 p8192` case 中主 kernel 和 combine kernel 均明显下降。
4. Decode 默认后端不一定自动选择 FlashMLA。本次未指定 `decode_attention_backend` 时，实际走 FA3 split/combine，且 FP8 KV 引入了很重的 copy/cast 开销，核心 FA3 kernel 仍是 bf16。
5. 对比实机与 AIC 时不要只用 trace 中是否出现 `fp8/quant` 字样判断 KV cache 精度；DeepSeek FP8 权重 GEMM 本身就会产生大量 `per_token_group_quant_8bit_kernel` 和 `sm90_fp8_gemm_1d2d_impl`。
