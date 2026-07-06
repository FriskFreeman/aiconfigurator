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
- 整块 MLA-module 时长: `1.497 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.497 ms`, `module_to_last_kernel=1.523 ms`, `host_to_first_kernel_gap=0.180635`, `host_end_to_last_kernel_tail=0.026 ms`, `gpu_makespan=1.343 ms`, `gpu_kernel_sum=0.168 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.313 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.313 ms`, `host_to_first_kernel_gap=0.103207`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.201 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `16758438`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[256, 7168]]}`
- `q_a_layernorm` -> 0.030 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.030 ms`, `host_to_first_kernel_gap=0.027568`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `17121277`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[256, 1536]]}`
- `kv_a_layernorm` -> 0.019 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.019 ms`, `host_to_first_kernel_gap=0.017762`, `host_end_to_last_kernel_tail=0.001 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `17166411`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[256, 512]]}`
- `q_b_proj` -> 0.152 ms
  纯GPU kernel时间: `0.025 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.152 ms`, `host_to_first_kernel_gap=0.056784`, `host_end_to_last_kernel_tail=0.007 ms`, `gpu_makespan=0.102 ms`, `gpu_kernel_sum=0.025 ms`
  开始时间(ns): `17212221`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[256, 1536]]}`
- `rotary_emb` -> 0.085 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.085 ms`, `host_to_first_kernel_gap=0.07415`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `17546310`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[256], [256, 128, 64], [256, 1, 64]]}`
- `attn_mqa` -> 0.237 ms
  纯GPU kernel时间: `0.059 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.237 ms`, `host_to_first_kernel_gap=0.088222`, `host_end_to_last_kernel_tail=0.036 ms`, `gpu_makespan=0.184 ms`, `gpu_kernel_sum=0.059 ms`
  开始时间(ns): `17663662`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[256, 128, 512], [256, 1, 512], [256, 1, 512]]}`
- `o_proj` -> 0.176 ms
  纯GPU kernel时间: `0.058 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.176 ms`, `host_to_first_kernel_gap=0.062681`, `host_end_to_last_kernel_tail=0.038 ms`, `gpu_makespan=0.151 ms`, `gpu_kernel_sum=0.058 ms`
  开始时间(ns): `17990643`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[256, 16384]]}`

## Layer 1 / decode / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `0.980 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=0.980 ms`, `module_to_last_kernel=1.008 ms`, `host_to_first_kernel_gap=0.09777`, `host_end_to_last_kernel_tail=0.028 ms`, `gpu_makespan=0.910 ms`, `gpu_kernel_sum=0.167 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.121 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.121 ms`, `host_to_first_kernel_gap=0.046747`, `host_end_to_last_kernel_tail=0.001 ms`, `gpu_makespan=0.075 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `18757328`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[256, 7168]]}`
- `q_a_layernorm` -> 0.027 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.027 ms`, `host_to_first_kernel_gap=0.025367`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `18913460`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[256, 1536]]}`
- `kv_a_layernorm` -> 0.018 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.018 ms`, `host_to_first_kernel_gap=0.017207`, `host_end_to_last_kernel_tail=0.001 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `18954772`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[256, 512]]}`
- `q_b_proj` -> 0.117 ms
  纯GPU kernel时间: `0.024 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.117 ms`, `host_to_first_kernel_gap=0.042163`, `host_end_to_last_kernel_tail=0.012 ms`, `gpu_makespan=0.087 ms`, `gpu_kernel_sum=0.024 ms`
  开始时间(ns): `18993944`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[256, 1536]]}`
- `rotary_emb` -> 0.049 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.049 ms`, `host_to_first_kernel_gap=0.043876`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `19207367`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[256], [256, 128, 64], [256, 1, 64]]}`
- `attn_mqa` -> 0.160 ms
  纯GPU kernel时间: `0.060 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.160 ms`, `host_to_first_kernel_gap=0.05927`, `host_end_to_last_kernel_tail=0.042 ms`, `gpu_makespan=0.143 ms`, `gpu_kernel_sum=0.060 ms`
  开始时间(ns): `19281861`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[256, 128, 512], [256, 1, 512], [256, 1, 512]]}`
- `o_proj` -> 0.158 ms
  纯GPU kernel时间: `0.058 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.158 ms`, `host_to_first_kernel_gap=0.057618`, `host_end_to_last_kernel_tail=0.038 ms`, `gpu_makespan=0.138 ms`, `gpu_kernel_sum=0.058 ms`
  开始时间(ns): `19518360`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[256, 16384]]}`

## Layer 2 / decode / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `0.961 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=0.961 ms`, `module_to_last_kernel=0.984 ms`, `host_to_first_kernel_gap=0.084555`, `host_end_to_last_kernel_tail=0.022 ms`, `gpu_makespan=0.899 ms`, `gpu_kernel_sum=0.169 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.118 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.118 ms`, `host_to_first_kernel_gap=0.045251`, `host_end_to_last_kernel_tail=0.001 ms`, `gpu_makespan=0.073 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `20228999`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[256, 7168]]}`
- `q_a_layernorm` -> 0.026 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.026 ms`, `host_to_first_kernel_gap=0.024309`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `20382101`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[256, 1536]]}`
- `kv_a_layernorm` -> 0.018 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.018 ms`, `host_to_first_kernel_gap=0.017187`, `host_end_to_last_kernel_tail=0.001 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `20422823`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[256, 512]]}`
- `q_b_proj` -> 0.113 ms
  纯GPU kernel时间: `0.025 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.113 ms`, `host_to_first_kernel_gap=0.041809`, `host_end_to_last_kernel_tail=0.013 ms`, `gpu_makespan=0.084 ms`, `gpu_kernel_sum=0.025 ms`
  开始时间(ns): `20461849`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[256, 1536]]}`
- `rotary_emb` -> 0.050 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.050 ms`, `host_to_first_kernel_gap=0.042981`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `20665572`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[256], [256, 128, 64], [256, 1, 64]]}`
- `attn_mqa` -> 0.157 ms
  纯GPU kernel时间: `0.060 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.157 ms`, `host_to_first_kernel_gap=0.056929`, `host_end_to_last_kernel_tail=0.042 ms`, `gpu_makespan=0.142 ms`, `gpu_kernel_sum=0.060 ms`
  开始时间(ns): `20746760`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[256, 128, 512], [256, 1, 512], [256, 1, 512]]}`
- `o_proj` -> 0.156 ms
  纯GPU kernel时间: `0.058 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.156 ms`, `host_to_first_kernel_gap=0.056723`, `host_end_to_last_kernel_tail=0.038 ms`, `gpu_makespan=0.138 ms`, `gpu_kernel_sum=0.058 ms`
  开始时间(ns): `20978678`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[256, 16384]]}`
