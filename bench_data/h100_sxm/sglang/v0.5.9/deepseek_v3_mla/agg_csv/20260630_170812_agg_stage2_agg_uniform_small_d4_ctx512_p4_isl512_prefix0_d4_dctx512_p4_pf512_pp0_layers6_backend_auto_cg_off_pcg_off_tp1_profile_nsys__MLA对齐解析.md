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
- 整块 MLA-module 时长: `2.267 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.267 ms`, `module_to_last_kernel=2.735 ms`, `host_to_first_kernel_gap=0.310482`, `host_end_to_last_kernel_tail=0.468 ms`, `gpu_makespan=2.425 ms`, `gpu_kernel_sum=1.116 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.420 ms
  纯GPU kernel时间: `0.080 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.420 ms`, `host_to_first_kernel_gap=0.148588`, `host_end_to_last_kernel_tail=0.025 ms`, `gpu_makespan=0.296 ms`, `gpu_kernel_sum=0.080 ms`
  开始时间(ns): `14864326`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2052, 7168]]}`
- `q_a_layernorm` -> 0.052 ms
  纯GPU kernel时间: `0.006 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.052 ms`, `host_to_first_kernel_gap=0.045704`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.006 ms`, `gpu_kernel_sum=0.006 ms`
  开始时间(ns): `15364170`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2052, 1536]]}`
- `kv_a_layernorm` -> 0.034 ms
  纯GPU kernel时间: `0.004 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.034 ms`, `host_to_first_kernel_gap=0.029931`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.004 ms`, `gpu_kernel_sum=0.004 ms`
  开始时间(ns): `15447911`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2052, 512]]}`
- `q_b_proj` -> 0.252 ms
  纯GPU kernel时间: `0.136 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.252 ms`, `host_to_first_kernel_gap=0.089709`, `host_end_to_last_kernel_tail=0.102 ms`, `gpu_makespan=0.265 ms`, `gpu_kernel_sum=0.136 ms`
  开始时间(ns): `15522565`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2052, 1536]]}`
- `rotary_emb` -> 0.104 ms
  纯GPU kernel时间: `0.028 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.104 ms`, `host_to_first_kernel_gap=0.088791`, `host_end_to_last_kernel_tail=0.013 ms`, `gpu_makespan=0.028 ms`, `gpu_kernel_sum=0.028 ms`
  开始时间(ns): `16001274`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2052], [2052, 128, 64], [2052, 1, 64]]}`
- `attn_mqa` -> 0.327 ms
  纯GPU kernel时间: `0.486 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.327 ms`, `host_to_first_kernel_gap=0.107671`, `host_end_to_last_kernel_tail=0.445 ms`, `gpu_makespan=0.664 ms`, `gpu_kernel_sum=0.486 ms`
  开始时间(ns): `16157690`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2052, 128, 512], [2052, 1, 512], [2052, 1, 512]]}`
- `o_proj` -> 0.303 ms
  纯GPU kernel时间: `0.376 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.303 ms`, `host_to_first_kernel_gap=0.416425`, `host_end_to_last_kernel_tail=0.490 ms`, `gpu_makespan=0.377 ms`, `gpu_kernel_sum=0.376 ms`
  开始时间(ns): `16644423`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2052, 16384]]}`

## Layer 1 / decode / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `1.709 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.709 ms`, `module_to_last_kernel=2.294 ms`, `host_to_first_kernel_gap=0.902519`, `host_end_to_last_kernel_tail=0.585 ms`, `gpu_makespan=1.391 ms`, `gpu_kernel_sum=1.115 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.217 ms
  纯GPU kernel时间: `0.081 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.217 ms`, `host_to_first_kernel_gap=0.834221`, `host_end_to_last_kernel_tail=0.699 ms`, `gpu_makespan=0.082 ms`, `gpu_kernel_sum=0.081 ms`
  开始时间(ns): `17936897`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2052, 7168]]}`
- `q_a_layernorm` -> 0.051 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.051 ms`, `host_to_first_kernel_gap=0.636638`, `host_end_to_last_kernel_tail=0.591 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `18216560`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2052, 1536]]}`
- `kv_a_layernorm` -> 0.033 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.033 ms`, `host_to_first_kernel_gap=0.567176`, `host_end_to_last_kernel_tail=0.539 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `18291302`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2052, 512]]}`
- `q_b_proj` -> 0.219 ms
  纯GPU kernel时间: `0.130 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.219 ms`, `host_to_first_kernel_gap=0.504396`, `host_end_to_last_kernel_tail=0.417 ms`, `gpu_makespan=0.132 ms`, `gpu_kernel_sum=0.130 ms`
  开始时间(ns): `18359874`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2052, 1536]]}`
- `rotary_emb` -> 0.079 ms
  纯GPU kernel时间: `0.028 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.079 ms`, `host_to_first_kernel_gap=0.39401`, `host_end_to_last_kernel_tail=0.342 ms`, `gpu_makespan=0.028 ms`, `gpu_kernel_sum=0.028 ms`
  开始时间(ns): `18736979`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2052], [2052, 128, 64], [2052, 1, 64]]}`
- `attn_mqa` -> 0.266 ms
  纯GPU kernel时间: `0.486 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.266 ms`, `host_to_first_kernel_gap=0.298055`, `host_end_to_last_kernel_tail=0.519 ms`, `gpu_makespan=0.488 ms`, `gpu_kernel_sum=0.486 ms`
  开始时间(ns): `18861862`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2052, 128, 512], [2052, 1, 512], [2052, 1, 512]]}`
- `o_proj` -> 0.294 ms
  纯GPU kernel时间: `0.380 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.294 ms`, `host_to_first_kernel_gap=0.5179`, `host_end_to_last_kernel_tail=0.605 ms`, `gpu_makespan=0.381 ms`, `gpu_kernel_sum=0.380 ms`
  开始时间(ns): `19263233`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2052, 16384]]}`

## Layer 2 / decode / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.663 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.663 ms`, `module_to_last_kernel=2.453 ms`, `host_to_first_kernel_gap=1.069429`, `host_end_to_last_kernel_tail=0.789 ms`, `gpu_makespan=1.383 ms`, `gpu_kernel_sum=1.110 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.221 ms
  纯GPU kernel时间: `0.079 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.221 ms`, `host_to_first_kernel_gap=0.998618`, `host_end_to_last_kernel_tail=0.858 ms`, `gpu_makespan=0.081 ms`, `gpu_kernel_sum=0.079 ms`
  开始时间(ns): `20494800`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2052, 7168]]}`
- `q_a_layernorm` -> 0.046 ms
  纯GPU kernel时间: `0.006 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.046 ms`, `host_to_first_kernel_gap=0.798212`, `host_end_to_last_kernel_tail=0.758 ms`, `gpu_makespan=0.006 ms`, `gpu_kernel_sum=0.006 ms`
  开始时间(ns): `20775622`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2052, 1536]]}`
- `kv_a_layernorm` -> 0.034 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.034 ms`, `host_to_first_kernel_gap=0.730471`, `host_end_to_last_kernel_tail=0.701 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `20849123`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2052, 512]]}`
- `q_b_proj` -> 0.205 ms
  纯GPU kernel时间: `0.131 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.205 ms`, `host_to_first_kernel_gap=0.668018`, `host_end_to_last_kernel_tail=0.595 ms`, `gpu_makespan=0.132 ms`, `gpu_kernel_sum=0.131 ms`
  开始时间(ns): `20917272`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2052, 1536]]}`
- `rotary_emb` -> 0.076 ms
  纯GPU kernel时间: `0.027 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.076 ms`, `host_to_first_kernel_gap=0.578936`, `host_end_to_last_kernel_tail=0.530 ms`, `gpu_makespan=0.027 ms`, `gpu_kernel_sum=0.027 ms`
  开始时间(ns): `21272466`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2052], [2052, 128, 64], [2052, 1, 64]]}`
- `attn_mqa` -> 0.260 ms
  纯GPU kernel时间: `0.484 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.260 ms`, `host_to_first_kernel_gap=0.480341`, `host_end_to_last_kernel_tail=0.706 ms`, `gpu_makespan=0.485 ms`, `gpu_kernel_sum=0.484 ms`
  开始时间(ns): `21399733`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2052, 128, 512], [2052, 1, 512], [2052, 1, 512]]}`
- `o_proj` -> 0.272 ms
  纯GPU kernel时间: `0.378 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.272 ms`, `host_to_first_kernel_gap=0.701809`, `host_end_to_last_kernel_tail=0.808 ms`, `gpu_makespan=0.379 ms`, `gpu_kernel_sum=0.378 ms`
  开始时间(ns): `21796088`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2052, 16384]]}`

## Layer 3 / decode / instance 1

- 执行序号: `4`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `1.637 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.637 ms`, `module_to_last_kernel=2.678 ms`, `host_to_first_kernel_gap=1.283922`, `host_end_to_last_kernel_tail=1.041 ms`, `gpu_makespan=1.394 ms`, `gpu_kernel_sum=1.118 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.208 ms
  纯GPU kernel时间: `0.081 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.208 ms`, `host_to_first_kernel_gap=1.219192`, `host_end_to_last_kernel_tail=1.094 ms`, `gpu_makespan=0.082 ms`, `gpu_kernel_sum=0.081 ms`
  开始时间(ns): `22983663`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2052, 7168]]}`
- `q_a_layernorm` -> 0.045 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.045 ms`, `host_to_first_kernel_gap=1.034767`, `host_end_to_last_kernel_tail=0.995 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `23250104`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2052, 1536]]}`
- `kv_a_layernorm` -> 0.035 ms
  纯GPU kernel时间: `0.004 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.035 ms`, `host_to_first_kernel_gap=0.968879`, `host_end_to_last_kernel_tail=0.938 ms`, `gpu_makespan=0.004 ms`, `gpu_kernel_sum=0.004 ms`
  开始时间(ns): `23321656`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2052, 512]]}`
- `q_b_proj` -> 0.210 ms
  纯GPU kernel时间: `0.130 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.210 ms`, `host_to_first_kernel_gap=0.9062`, `host_end_to_last_kernel_tail=0.828 ms`, `gpu_makespan=0.132 ms`, `gpu_kernel_sum=0.130 ms`
  开始时间(ns): `23391023`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2052, 1536]]}`
- `rotary_emb` -> 0.075 ms
  纯GPU kernel时间: `0.027 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.075 ms`, `host_to_first_kernel_gap=0.816318`, `host_end_to_last_kernel_tail=0.769 ms`, `gpu_makespan=0.027 ms`, `gpu_kernel_sum=0.027 ms`
  开始时间(ns): `23747593`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2052], [2052, 128, 64], [2052, 1, 64]]}`
- `attn_mqa` -> 0.257 ms
  纯GPU kernel时间: `0.482 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.257 ms`, `host_to_first_kernel_gap=0.726389`, `host_end_to_last_kernel_tail=0.953 ms`, `gpu_makespan=0.483 ms`, `gpu_kernel_sum=0.482 ms`
  开始时间(ns): `23867057`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2052, 128, 512], [2052, 1, 512], [2052, 1, 512]]}`
- `o_proj` -> 0.278 ms
  纯GPU kernel时间: `0.387 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.278 ms`, `host_to_first_kernel_gap=0.950451`, `host_end_to_last_kernel_tail=1.062 ms`, `gpu_makespan=0.389 ms`, `gpu_kernel_sum=0.387 ms`
  开始时间(ns): `24257747`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2052, 16384]]}`

## Layer 4 / decode / instance 1

- 执行序号: `5`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `1.746 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.746 ms`, `module_to_last_kernel=3.662 ms`, `host_to_first_kernel_gap=2.283465`, `host_end_to_last_kernel_tail=1.916 ms`, `gpu_makespan=1.379 ms`, `gpu_kernel_sum=1.102 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.257 ms
  纯GPU kernel时间: `0.079 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.257 ms`, `host_to_first_kernel_gap=2.216503`, `host_end_to_last_kernel_tail=2.041 ms`, `gpu_makespan=0.081 ms`, `gpu_kernel_sum=0.079 ms`
  开始时间(ns): `26025867`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2052, 7168]]}`
- `q_a_layernorm` -> 0.049 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.049 ms`, `host_to_first_kernel_gap=1.981183`, `host_end_to_last_kernel_tail=1.937 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `26342851`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2052, 1536]]}`
- `kv_a_layernorm` -> 0.042 ms
  纯GPU kernel时间: `0.004 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.042 ms`, `host_to_first_kernel_gap=1.911712`, `host_end_to_last_kernel_tail=1.875 ms`, `gpu_makespan=0.004 ms`, `gpu_kernel_sum=0.004 ms`
  开始时间(ns): `26417474`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2052, 512]]}`
- `q_b_proj` -> 0.229 ms
  纯GPU kernel时间: `0.131 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.229 ms`, `host_to_first_kernel_gap=1.839084`, `host_end_to_last_kernel_tail=1.742 ms`, `gpu_makespan=0.132 ms`, `gpu_kernel_sum=0.131 ms`
  开始时间(ns): `26495990`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2052, 1536]]}`
- `rotary_emb` -> 0.078 ms
  纯GPU kernel时间: `0.028 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.078 ms`, `host_to_first_kernel_gap=1.717179`, `host_end_to_last_kernel_tail=1.667 ms`, `gpu_makespan=0.028 ms`, `gpu_kernel_sum=0.028 ms`
  开始时间(ns): `26883238`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2052], [2052, 128, 64], [2052, 1, 64]]}`
- `attn_mqa` -> 0.260 ms
  纯GPU kernel时间: `0.483 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.260 ms`, `host_to_first_kernel_gap=1.622674`, `host_end_to_last_kernel_tail=1.847 ms`, `gpu_makespan=0.485 ms`, `gpu_kernel_sum=0.483 ms`
  开始时间(ns): `27006479`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2052, 128, 512], [2052, 1, 512], [2052, 1, 512]]}`
- `o_proj` -> 0.278 ms
  纯GPU kernel时间: `0.371 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.278 ms`, `host_to_first_kernel_gap=1.839942`, `host_end_to_last_kernel_tail=1.935 ms`, `gpu_makespan=0.373 ms`, `gpu_kernel_sum=0.371 ms`
  开始时间(ns): `27407803`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2052, 16384]]}`

## Layer 5 / decode / instance 1

- 执行序号: `6`
- Collector对齐边界: `{'Module': 'model.model.layers.5.self_attn'}`
- 整块 MLA-module 时长: `1.701 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.701 ms`, `module_to_last_kernel=4.676 ms`, `host_to_first_kernel_gap=3.279914`, `host_end_to_last_kernel_tail=2.976 ms`, `gpu_makespan=1.397 ms`, `gpu_kernel_sum=1.121 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.264 ms
  纯GPU kernel时间: `0.080 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.264 ms`, `host_to_first_kernel_gap=3.209801`, `host_end_to_last_kernel_tail=3.027 ms`, `gpu_makespan=0.081 ms`, `gpu_kernel_sum=0.080 ms`
  开始时间(ns): `29073172`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2052, 7168]]}`
- `q_a_layernorm` -> 0.054 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.054 ms`, `host_to_first_kernel_gap=2.965427`, `host_end_to_last_kernel_tail=2.917 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `29399114`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2052, 1536]]}`
- `kv_a_layernorm` -> 0.031 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.031 ms`, `host_to_first_kernel_gap=2.892001`, `host_end_to_last_kernel_tail=2.865 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `29478076`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2052, 512]]}`
- `q_b_proj` -> 0.221 ms
  纯GPU kernel时间: `0.131 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.221 ms`, `host_to_first_kernel_gap=2.828414`, `host_end_to_last_kernel_tail=2.739 ms`, `gpu_makespan=0.132 ms`, `gpu_kernel_sum=0.131 ms`
  开始时间(ns): `29547551`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2052, 1536]]}`
- `rotary_emb` -> 0.077 ms
  纯GPU kernel时间: `0.027 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.077 ms`, `host_to_first_kernel_gap=2.715515`, `host_end_to_last_kernel_tail=2.666 ms`, `gpu_makespan=0.027 ms`, `gpu_kernel_sum=0.027 ms`
  开始时间(ns): `29927969`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2052], [2052, 128, 64], [2052, 1, 64]]}`
- `attn_mqa` -> 0.246 ms
  纯GPU kernel时间: `0.488 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.246 ms`, `host_to_first_kernel_gap=2.621942`, `host_end_to_last_kernel_tail=2.865 ms`, `gpu_makespan=0.489 ms`, `gpu_kernel_sum=0.488 ms`
  开始时间(ns): `30050150`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2052, 128, 512], [2052, 1, 512], [2052, 1, 512]]}`
- `o_proj` -> 0.260 ms
  纯GPU kernel时间: `0.386 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.260 ms`, `host_to_first_kernel_gap=2.867836`, `host_end_to_last_kernel_tail=2.994 ms`, `gpu_makespan=0.387 ms`, `gpu_kernel_sum=0.386 ms`
  开始时间(ns): `30424960`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2052, 16384]]}`
