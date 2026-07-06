# MLA对齐解析摘要

源文件: `report.sqlite`

## 对齐原则

- `collector/sglang/collect_mla_module.py` 的 MLA module 计时边界是 `model.model.layers[test_layer].self_attn(...)`。
- 因此这里把 `nsys` 中每层的 `model.model.layers.X.self_attn` NVTX range 视为与 collector 对齐的 MLA-module 边界。
- 该区间内部的 `.self_attn.*` 子模块用于做 MLA 内部 breakdown。

## 运行摘要

- `prefill` 对齐成功层: `[]`
- `prefill` 被切分层: `[]`
- 若某层 `prefill` 出现多个 `self_attn` 实例，则说明 Engine 调度把一次前向切成了多块，已不再与 collector 的单次 MLA-module 采集严格一一对应。

## Layer 0 / decode / instance 1

- 执行序号: `1`
- Collector对齐边界: `{'Module': 'model.model.layers.0.self_attn'}`
- 整块 MLA-module 时长: `2.134 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.134 ms`, `module_to_last_kernel=2.134 ms`, `host_to_first_kernel_gap=0.222348`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=1.908 ms`, `gpu_kernel_sum=0.106 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.416 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.416 ms`, `host_to_first_kernel_gap=0.131593`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.252 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `18158575`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[3, 7168]]}`
- `q_a_layernorm` -> 0.052 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.052 ms`, `host_to_first_kernel_gap=0.04573`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `18658582`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[3, 1536]]}`
- `kv_a_layernorm` -> 0.034 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.034 ms`, `host_to_first_kernel_gap=0.030216`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `18736687`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[3, 512]]}`
- `q_b_proj` -> 0.251 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.251 ms`, `host_to_first_kernel_gap=0.089397`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.150 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `18812290`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[3, 1536]]}`
- `rotary_emb` -> 0.106 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.106 ms`, `host_to_first_kernel_gap=0.090983`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `19268176`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[3], [3, 128, 64], [3, 1, 64]]}`
- `attn_mqa` -> 0.327 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.327 ms`, `host_to_first_kernel_gap=0.114906`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.198 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `19424541`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[3, 128, 512], [3, 1, 512], [3, 1, 512]]}`
- `o_proj` -> 0.291 ms
  纯GPU kernel时间: `0.047 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.291 ms`, `host_to_first_kernel_gap=0.101963`, `host_end_to_last_kernel_tail=0.018 ms`, `gpu_makespan=0.206 ms`, `gpu_kernel_sum=0.047 ms`
  开始时间(ns): `19889931`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[3, 16384]]}`

## Layer 1 / decode / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `1.622 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.622 ms`, `module_to_last_kernel=1.623 ms`, `host_to_first_kernel_gap=0.148682`, `host_end_to_last_kernel_tail=0.001 ms`, `gpu_makespan=1.474 ms`, `gpu_kernel_sum=0.104 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.212 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.212 ms`, `host_to_first_kernel_gap=0.080229`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.125 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `21157327`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[3, 7168]]}`
- `q_a_layernorm` -> 0.051 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.051 ms`, `host_to_first_kernel_gap=0.046446`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `21426758`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[3, 1536]]}`
- `kv_a_layernorm` -> 0.033 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.033 ms`, `host_to_first_kernel_gap=0.029291`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `21501513`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[3, 512]]}`
- `q_b_proj` -> 0.214 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.214 ms`, `host_to_first_kernel_gap=0.08224`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.128 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `21567796`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[3, 1536]]}`
- `rotary_emb` -> 0.077 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.077 ms`, `host_to_first_kernel_gap=0.066058`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `21923081`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[3], [3, 128, 64], [3, 1, 64]]}`
- `attn_mqa` -> 0.257 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.257 ms`, `host_to_first_kernel_gap=0.090089`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.157 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `22044010`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[3, 128, 512], [3, 1, 512], [3, 1, 512]]}`
- `o_proj` -> 0.268 ms
  纯GPU kernel时间: `0.046 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.268 ms`, `host_to_first_kernel_gap=0.092614`, `host_end_to_last_kernel_tail=0.020 ms`, `gpu_makespan=0.195 ms`, `gpu_kernel_sum=0.046 ms`
  开始时间(ns): `22424333`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[3, 16384]]}`

## Layer 2 / decode / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.716 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.716 ms`, `module_to_last_kernel=1.716 ms`, `host_to_first_kernel_gap=0.149519`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=1.561 ms`, `gpu_kernel_sum=0.104 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.213 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.213 ms`, `host_to_first_kernel_gap=0.084433`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.122 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `23610752`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[3, 7168]]}`
- `q_a_layernorm` -> 0.045 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.045 ms`, `host_to_first_kernel_gap=0.03927`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `23879179`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[3, 1536]]}`
- `kv_a_layernorm` -> 0.031 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.031 ms`, `host_to_first_kernel_gap=0.027005`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `23949268`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[3, 512]]}`
- `q_b_proj` -> 0.204 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.204 ms`, `host_to_first_kernel_gap=0.075842`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.124 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `24012911`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[3, 1536]]}`
- `rotary_emb` -> 0.074 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.074 ms`, `host_to_first_kernel_gap=0.062973`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `24350771`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[3], [3, 128, 64], [3, 1, 64]]}`
- `attn_mqa` -> 0.259 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.259 ms`, `host_to_first_kernel_gap=0.084672`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.164 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `24468784`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[3, 128, 512], [3, 1, 512], [3, 1, 512]]}`
- `o_proj` -> 0.374 ms
  纯GPU kernel时间: `0.047 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.374 ms`, `host_to_first_kernel_gap=0.142559`, `host_end_to_last_kernel_tail=0.015 ms`, `gpu_makespan=0.247 ms`, `gpu_kernel_sum=0.047 ms`
  开始时间(ns): `24867281`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[3, 16384]]}`
