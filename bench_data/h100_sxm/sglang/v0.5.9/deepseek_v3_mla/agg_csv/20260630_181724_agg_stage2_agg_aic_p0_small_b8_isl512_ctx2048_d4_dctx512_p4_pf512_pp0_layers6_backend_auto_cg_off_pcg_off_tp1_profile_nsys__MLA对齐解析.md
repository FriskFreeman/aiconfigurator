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
- 整块 MLA-module 时长: `2.234 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.234 ms`, `module_to_last_kernel=2.694 ms`, `host_to_first_kernel_gap=0.200073`, `host_end_to_last_kernel_tail=0.460 ms`, `gpu_makespan=2.494 ms`, `gpu_kernel_sum=1.103 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.331 ms
  纯GPU kernel时间: `0.080 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.331 ms`, `host_to_first_kernel_gap=0.11464`, `host_end_to_last_kernel_tail=0.029 ms`, `gpu_makespan=0.246 ms`, `gpu_kernel_sum=0.080 ms`
  开始时间(ns): `21014173`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2052, 7168]]}`
- `q_a_layernorm` -> 0.050 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.050 ms`, `host_to_first_kernel_gap=0.043669`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `21411703`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2052, 1536]]}`
- `kv_a_layernorm` -> 0.031 ms
  纯GPU kernel时间: `0.004 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.031 ms`, `host_to_first_kernel_gap=0.027406`, `host_end_to_last_kernel_tail=0.001 ms`, `gpu_makespan=0.004 ms`, `gpu_kernel_sum=0.004 ms`
  开始时间(ns): `21485790`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2052, 512]]}`
- `q_b_proj` -> 0.243 ms
  纯GPU kernel时间: `0.135 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.243 ms`, `host_to_first_kernel_gap=0.086991`, `host_end_to_last_kernel_tail=0.103 ms`, `gpu_makespan=0.259 ms`, `gpu_kernel_sum=0.135 ms`
  开始时间(ns): `21554749`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2052, 1536]]}`
- `rotary_emb` -> 0.092 ms
  纯GPU kernel时间: `0.028 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.092 ms`, `host_to_first_kernel_gap=0.07744`, `host_end_to_last_kernel_tail=0.013 ms`, `gpu_makespan=0.028 ms`, `gpu_kernel_sum=0.028 ms`
  开始时间(ns): `22244907`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2052], [2052, 128, 64], [2052, 1, 64]]}`
- `attn_mqa` -> 0.289 ms
  纯GPU kernel时间: `0.481 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.289 ms`, `host_to_first_kernel_gap=0.100909`, `host_end_to_last_kernel_tail=0.443 ms`, `gpu_makespan=0.631 ms`, `gpu_kernel_sum=0.481 ms`
  开始时间(ns): `22388381`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2052, 128, 512], [2052, 1, 512], [2052, 1, 512]]}`
- `o_proj` -> 0.305 ms
  纯GPU kernel时间: `0.370 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.305 ms`, `host_to_first_kernel_gap=0.414612`, `host_end_to_last_kernel_tail=0.480 ms`, `gpu_makespan=0.371 ms`, `gpu_kernel_sum=0.370 ms`
  开始时间(ns): `22837301`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2052, 16384]]}`

## Layer 1 / decode / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `1.650 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.650 ms`, `module_to_last_kernel=2.322 ms`, `host_to_first_kernel_gap=0.941733`, `host_end_to_last_kernel_tail=0.672 ms`, `gpu_makespan=1.380 ms`, `gpu_kernel_sum=1.105 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.217 ms
  纯GPU kernel时间: `0.080 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.217 ms`, `host_to_first_kernel_gap=0.878367`, `host_end_to_last_kernel_tail=0.743 ms`, `gpu_makespan=0.081 ms`, `gpu_kernel_sum=0.080 ms`
  开始时间(ns): `24066792`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2052, 7168]]}`
- `q_a_layernorm` -> 0.046 ms
  纯GPU kernel时间: `0.006 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.046 ms`, `host_to_first_kernel_gap=0.683872`, `host_end_to_last_kernel_tail=0.644 ms`, `gpu_makespan=0.006 ms`, `gpu_kernel_sum=0.006 ms`
  开始时间(ns): `24342534`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2052, 1536]]}`
- `kv_a_layernorm` -> 0.032 ms
  纯GPU kernel时间: `0.004 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.032 ms`, `host_to_first_kernel_gap=0.61828`, `host_end_to_last_kernel_tail=0.591 ms`, `gpu_makespan=0.004 ms`, `gpu_kernel_sum=0.004 ms`
  开始时间(ns): `24413470`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2052, 512]]}`
- `q_b_proj` -> 0.214 ms
  纯GPU kernel时间: `0.131 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.214 ms`, `host_to_first_kernel_gap=0.559531`, `host_end_to_last_kernel_tail=0.477 ms`, `gpu_makespan=0.132 ms`, `gpu_kernel_sum=0.131 ms`
  开始时间(ns): `24477915`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2052, 1536]]}`
- `rotary_emb` -> 0.077 ms
  纯GPU kernel时间: `0.028 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.077 ms`, `host_to_first_kernel_gap=0.453161`, `host_end_to_last_kernel_tail=0.404 ms`, `gpu_makespan=0.028 ms`, `gpu_kernel_sum=0.028 ms`
  开始时间(ns): `24851293`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2052], [2052, 128, 64], [2052, 1, 64]]}`
- `attn_mqa` -> 0.266 ms
  纯GPU kernel时间: `0.474 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.266 ms`, `host_to_first_kernel_gap=0.351964`, `host_end_to_last_kernel_tail=0.562 ms`, `gpu_makespan=0.475 ms`, `gpu_kernel_sum=0.474 ms`
  开始时间(ns): `24982154`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2052, 128, 512], [2052, 1, 512], [2052, 1, 512]]}`
- `o_proj` -> 0.258 ms
  纯GPU kernel时间: `0.382 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.258 ms`, `host_to_first_kernel_gap=0.56463`, `host_end_to_last_kernel_tail=0.690 ms`, `gpu_makespan=0.383 ms`, `gpu_kernel_sum=0.382 ms`
  开始时间(ns): `25377551`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2052, 16384]]}`

## Layer 2 / decode / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.649 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.649 ms`, `module_to_last_kernel=2.585 ms`, `host_to_first_kernel_gap=1.20412`, `host_end_to_last_kernel_tail=0.936 ms`, `gpu_makespan=1.381 ms`, `gpu_kernel_sum=1.108 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.229 ms
  纯GPU kernel时间: `0.080 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.229 ms`, `host_to_first_kernel_gap=1.141413`, `host_end_to_last_kernel_tail=0.994 ms`, `gpu_makespan=0.081 ms`, `gpu_kernel_sum=0.080 ms`
  开始时间(ns): `26515677`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2052, 7168]]}`
- `q_a_layernorm` -> 0.045 ms
  纯GPU kernel时间: `0.006 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.045 ms`, `host_to_first_kernel_gap=0.934099`, `host_end_to_last_kernel_tail=0.895 ms`, `gpu_makespan=0.006 ms`, `gpu_kernel_sum=0.006 ms`
  开始时间(ns): `26803983`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2052, 1536]]}`
- `kv_a_layernorm` -> 0.034 ms
  纯GPU kernel时间: `0.004 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.034 ms`, `host_to_first_kernel_gap=0.872103`, `host_end_to_last_kernel_tail=0.842 ms`, `gpu_makespan=0.004 ms`, `gpu_kernel_sum=0.004 ms`
  开始时间(ns): `26871707`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2052, 512]]}`
- `q_b_proj` -> 0.217 ms
  纯GPU kernel时间: `0.131 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.217 ms`, `host_to_first_kernel_gap=0.811013`, `host_end_to_last_kernel_tail=0.726 ms`, `gpu_makespan=0.132 ms`, `gpu_kernel_sum=0.131 ms`
  开始时间(ns): `26938685`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2052, 1536]]}`
- `rotary_emb` -> 0.077 ms
  纯GPU kernel时间: `0.027 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.077 ms`, `host_to_first_kernel_gap=0.703163`, `host_end_to_last_kernel_tail=0.653 ms`, `gpu_makespan=0.027 ms`, `gpu_kernel_sum=0.027 ms`
  开始时间(ns): `27311815`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2052], [2052, 128, 64], [2052, 1, 64]]}`
- `attn_mqa` -> 0.266 ms
  纯GPU kernel时间: `0.485 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.266 ms`, `host_to_first_kernel_gap=0.612476`, `host_end_to_last_kernel_tail=0.834 ms`, `gpu_makespan=0.487 ms`, `gpu_kernel_sum=0.485 ms`
  开始时间(ns): `27431270`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2052, 128, 512], [2052, 1, 512], [2052, 1, 512]]}`
- `o_proj` -> 0.258 ms
  纯GPU kernel时间: `0.374 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.258 ms`, `host_to_first_kernel_gap=0.836939`, `host_end_to_last_kernel_tail=0.955 ms`, `gpu_makespan=0.375 ms`, `gpu_kernel_sum=0.374 ms`
  开始时间(ns): `27825846`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2052, 16384]]}`

## Layer 3 / decode / instance 1

- 执行序号: `4`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `1.596 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.596 ms`, `module_to_last_kernel=2.859 ms`, `host_to_first_kernel_gap=1.473152`, `host_end_to_last_kernel_tail=1.262 ms`, `gpu_makespan=1.386 ms`, `gpu_kernel_sum=1.108 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.215 ms
  纯GPU kernel时间: `0.080 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.215 ms`, `host_to_first_kernel_gap=1.413891`, `host_end_to_last_kernel_tail=1.280 ms`, `gpu_makespan=0.081 ms`, `gpu_kernel_sum=0.080 ms`
  开始时间(ns): `28943227`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2052, 7168]]}`
- `q_a_layernorm` -> 0.043 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.043 ms`, `host_to_first_kernel_gap=1.224629`, `host_end_to_last_kernel_tail=1.187 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `29213481`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2052, 1536]]}`
- `kv_a_layernorm` -> 0.030 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.030 ms`, `host_to_first_kernel_gap=1.16276`, `host_end_to_last_kernel_tail=1.137 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `29280982`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2052, 512]]}`
- `q_b_proj` -> 0.212 ms
  纯GPU kernel时间: `0.130 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.212 ms`, `host_to_first_kernel_gap=1.106371`, `host_end_to_last_kernel_tail=1.027 ms`, `gpu_makespan=0.132 ms`, `gpu_kernel_sum=0.130 ms`
  开始时间(ns): `29344027`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2052, 1536]]}`
- `rotary_emb` -> 0.075 ms
  纯GPU kernel时间: `0.028 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.075 ms`, `host_to_first_kernel_gap=1.007369`, `host_end_to_last_kernel_tail=0.961 ms`, `gpu_makespan=0.028 ms`, `gpu_kernel_sum=0.028 ms`
  开始时间(ns): `29708468`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2052], [2052, 128, 64], [2052, 1, 64]]}`
- `attn_mqa` -> 0.263 ms
  纯GPU kernel时间: `0.479 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.263 ms`, `host_to_first_kernel_gap=0.9204`, `host_end_to_last_kernel_tail=1.139 ms`, `gpu_makespan=0.481 ms`, `gpu_kernel_sum=0.479 ms`
  开始时间(ns): `29825709`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2052, 128, 512], [2052, 1, 512], [2052, 1, 512]]}`
- `o_proj` -> 0.247 ms
  纯GPU kernel时间: `0.381 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.247 ms`, `host_to_first_kernel_gap=1.144048`, `host_end_to_last_kernel_tail=1.281 ms`, `gpu_makespan=0.384 ms`, `gpu_kernel_sum=0.381 ms`
  开始时间(ns): `30214892`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2052, 16384]]}`

## Layer 4 / decode / instance 1

- 执行序号: `5`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `1.690 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.690 ms`, `module_to_last_kernel=3.975 ms`, `host_to_first_kernel_gap=2.599758`, `host_end_to_last_kernel_tail=2.285 ms`, `gpu_makespan=1.375 ms`, `gpu_kernel_sum=1.098 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.271 ms
  纯GPU kernel时间: `0.079 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.271 ms`, `host_to_first_kernel_gap=2.533524`, `host_end_to_last_kernel_tail=2.344 ms`, `gpu_makespan=0.082 ms`, `gpu_kernel_sum=0.079 ms`
  开始时间(ns): `31849764`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2052, 7168]]}`
- `q_a_layernorm` -> 0.047 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.047 ms`, `host_to_first_kernel_gap=2.282077`, `host_end_to_last_kernel_tail=2.241 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `32183354`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2052, 1536]]}`
- `kv_a_layernorm` -> 0.030 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.030 ms`, `host_to_first_kernel_gap=2.216657`, `host_end_to_last_kernel_tail=2.191 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `32253894`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2052, 512]]}`
- `q_b_proj` -> 0.224 ms
  纯GPU kernel时间: `0.130 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.224 ms`, `host_to_first_kernel_gap=2.158733`, `host_end_to_last_kernel_tail=2.066 ms`, `gpu_makespan=0.131 ms`, `gpu_kernel_sum=0.130 ms`
  开始时间(ns): `32317898`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2052, 1536]]}`
- `rotary_emb` -> 0.076 ms
  纯GPU kernel时间: `0.028 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.076 ms`, `host_to_first_kernel_gap=2.042919`, `host_end_to_last_kernel_tail=1.994 ms`, `gpu_makespan=0.028 ms`, `gpu_kernel_sum=0.028 ms`
  开始时间(ns): `32699760`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2052], [2052, 128, 64], [2052, 1, 64]]}`
- `attn_mqa` -> 0.258 ms
  纯GPU kernel时间: `0.481 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.258 ms`, `host_to_first_kernel_gap=1.953013`, `host_end_to_last_kernel_tail=2.178 ms`, `gpu_makespan=0.483 ms`, `gpu_kernel_sum=0.481 ms`
  开始时间(ns): `32818562`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2052, 128, 512], [2052, 1, 512], [2052, 1, 512]]}`
- `o_proj` -> 0.250 ms
  纯GPU kernel时间: `0.370 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.250 ms`, `host_to_first_kernel_gap=2.180839`, `host_end_to_last_kernel_tail=2.304 ms`, `gpu_makespan=0.372 ms`, `gpu_kernel_sum=0.370 ms`
  开始时间(ns): `33205391`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2052, 16384]]}`

## Layer 5 / decode / instance 1

- 执行序号: `6`
- Collector对齐边界: `{'Module': 'model.model.layers.5.self_attn'}`
- 整块 MLA-module 时长: `1.685 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.685 ms`, `module_to_last_kernel=5.099 ms`, `host_to_first_kernel_gap=3.70296`, `host_end_to_last_kernel_tail=3.414 ms`, `gpu_makespan=1.396 ms`, `gpu_kernel_sum=1.122 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.266 ms
  纯GPU kernel时间: `0.080 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.266 ms`, `host_to_first_kernel_gap=3.635422`, `host_end_to_last_kernel_tail=3.451 ms`, `gpu_makespan=0.081 ms`, `gpu_kernel_sum=0.080 ms`
  开始时间(ns): `34774291`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2052, 7168]]}`
- `q_a_layernorm` -> 0.047 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.047 ms`, `host_to_first_kernel_gap=3.389339`, `host_end_to_last_kernel_tail=3.347 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `35102230`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2052, 1536]]}`
- `kv_a_layernorm` -> 0.030 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.030 ms`, `host_to_first_kernel_gap=3.32385`, `host_end_to_last_kernel_tail=3.298 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `35172903`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2052, 512]]}`
- `q_b_proj` -> 0.225 ms
  纯GPU kernel时间: `0.130 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.225 ms`, `host_to_first_kernel_gap=3.265084`, `host_end_to_last_kernel_tail=3.171 ms`, `gpu_makespan=0.131 ms`, `gpu_kernel_sum=0.130 ms`
  开始时间(ns): `35237461`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2052, 1536]]}`
- `rotary_emb` -> 0.076 ms
  纯GPU kernel时间: `0.028 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.076 ms`, `host_to_first_kernel_gap=3.146526`, `host_end_to_last_kernel_tail=3.099 ms`, `gpu_makespan=0.028 ms`, `gpu_kernel_sum=0.028 ms`
  开始时间(ns): `35621779`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2052], [2052, 128, 64], [2052, 1, 64]]}`
- `attn_mqa` -> 0.252 ms
  纯GPU kernel时间: `0.491 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.252 ms`, `host_to_first_kernel_gap=3.055995`, `host_end_to_last_kernel_tail=3.295 ms`, `gpu_makespan=0.492 ms`, `gpu_kernel_sum=0.491 ms`
  开始时间(ns): `35741430`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2052, 128, 512], [2052, 1, 512], [2052, 1, 512]]}`
- `o_proj` -> 0.252 ms
  纯GPU kernel时间: `0.384 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.252 ms`, `host_to_first_kernel_gap=3.298389`, `host_end_to_last_kernel_tail=3.431 ms`, `gpu_makespan=0.385 ms`, `gpu_kernel_sum=0.384 ms`
  开始时间(ns): `36121947`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2052, 16384]]}`
