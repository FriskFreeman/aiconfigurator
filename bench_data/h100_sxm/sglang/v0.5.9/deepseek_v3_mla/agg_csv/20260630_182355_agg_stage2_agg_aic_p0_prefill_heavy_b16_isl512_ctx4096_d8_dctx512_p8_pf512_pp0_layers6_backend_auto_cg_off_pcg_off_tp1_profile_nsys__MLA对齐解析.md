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
- 整块 MLA-module 时长: `2.281 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.281 ms`, `module_to_last_kernel=3.580 ms`, `host_to_first_kernel_gap=0.229535`, `host_end_to_last_kernel_tail=1.299 ms`, `gpu_makespan=3.350 ms`, `gpu_kernel_sum=2.113 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.409 ms
  纯GPU kernel时间: `0.145 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.409 ms`, `host_to_first_kernel_gap=0.13964`, `host_end_to_last_kernel_tail=0.070 ms`, `gpu_makespan=0.340 ms`, `gpu_kernel_sum=0.145 ms`
  开始时间(ns): `23869373`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4104, 7168]]}`
- `q_a_layernorm` -> 0.057 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.057 ms`, `host_to_first_kernel_gap=0.05137`, `host_end_to_last_kernel_tail=0.003 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `24353674`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4104, 1536]]}`
- `kv_a_layernorm` -> 0.034 ms
  纯GPU kernel时间: `0.007 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.034 ms`, `host_to_first_kernel_gap=0.029667`, `host_end_to_last_kernel_tail=0.003 ms`, `gpu_makespan=0.007 ms`, `gpu_kernel_sum=0.007 ms`
  开始时间(ns): `24435889`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4104, 512]]}`
- `q_b_proj` -> 0.280 ms
  纯GPU kernel时间: `0.243 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.280 ms`, `host_to_first_kernel_gap=0.093107`, `host_end_to_last_kernel_tail=0.204 ms`, `gpu_makespan=0.391 ms`, `gpu_kernel_sum=0.243 ms`
  开始时间(ns): `24507201`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4104, 1536]]}`
- `rotary_emb` -> 0.108 ms
  纯GPU kernel时间: `0.054 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.108 ms`, `host_to_first_kernel_gap=0.236979`, `host_end_to_last_kernel_tail=0.183 ms`, `gpu_makespan=0.054 ms`, `gpu_kernel_sum=0.054 ms`
  开始时间(ns): `25016000`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4104], [4104, 128, 64], [4104, 1, 64]]}`
- `attn_mqa` -> 0.329 ms
  纯GPU kernel时间: `0.944 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.329 ms`, `host_to_first_kernel_gap=0.131621`, `host_end_to_last_kernel_tail=0.897 ms`, `gpu_makespan=1.095 ms`, `gpu_kernel_sum=0.944 ms`
  开始时间(ns): `25176622`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4104, 128, 512], [4104, 1, 512], [4104, 1, 512]]}`
- `o_proj` -> 0.343 ms
  纯GPU kernel时间: `0.711 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.343 ms`, `host_to_first_kernel_gap=0.95088`, `host_end_to_last_kernel_tail=1.321 ms`, `gpu_makespan=0.713 ms`, `gpu_kernel_sum=0.711 ms`
  开始时间(ns): `25695697`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4104, 16384]]}`

## Layer 1 / decode / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `1.914 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.914 ms`, `module_to_last_kernel=5.508 ms`, `host_to_first_kernel_gap=2.904545`, `host_end_to_last_kernel_tail=3.594 ms`, `gpu_makespan=2.603 ms`, `gpu_kernel_sum=2.088 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.221 ms
  纯GPU kernel时间: `0.144 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.221 ms`, `host_to_first_kernel_gap=2.830611`, `host_end_to_last_kernel_tail=2.755 ms`, `gpu_makespan=0.145 ms`, `gpu_kernel_sum=0.144 ms`
  开始时间(ns): `27047545`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4104, 7168]]}`
- `q_a_layernorm` -> 0.057 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.057 ms`, `host_to_first_kernel_gap=2.688436`, `host_end_to_last_kernel_tail=2.641 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `27334680`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4104, 1536]]}`
- `kv_a_layernorm` -> 0.036 ms
  纯GPU kernel时间: `0.007 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.036 ms`, `host_to_first_kernel_gap=2.616978`, `host_end_to_last_kernel_tail=2.588 ms`, `gpu_makespan=0.007 ms`, `gpu_kernel_sum=0.007 ms`
  开始时间(ns): `27415482`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4104, 512]]}`
- `q_b_proj` -> 0.225 ms
  纯GPU kernel时间: `0.235 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.225 ms`, `host_to_first_kernel_gap=2.553419`, `host_end_to_last_kernel_tail=2.565 ms`, `gpu_makespan=0.236 ms`, `gpu_kernel_sum=0.235 ms`
  开始时间(ns): `27487265`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4104, 1536]]}`
- `rotary_emb` -> 0.098 ms
  纯GPU kernel时间: `0.054 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.098 ms`, `host_to_first_kernel_gap=2.637203`, `host_end_to_last_kernel_tail=2.593 ms`, `gpu_makespan=0.054 ms`, `gpu_kernel_sum=0.054 ms`
  开始时间(ns): `27904728`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4104], [4104, 128, 64], [4104, 1, 64]]}`
- `attn_mqa` -> 0.314 ms
  纯GPU kernel时间: `0.928 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.314 ms`, `host_to_first_kernel_gap=2.545766`, `host_end_to_last_kernel_tail=3.161 ms`, `gpu_makespan=0.929 ms`, `gpu_kernel_sum=0.928 ms`
  开始时间(ns): `28050725`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4104, 128, 512], [4104, 1, 512], [4104, 1, 512]]}`
- `o_proj` -> 0.334 ms
  纯GPU kernel时间: `0.711 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.334 ms`, `host_to_first_kernel_gap=3.243699`, `host_end_to_last_kernel_tail=3.622 ms`, `gpu_makespan=0.712 ms`, `gpu_kernel_sum=0.711 ms`
  开始时间(ns): `28525526`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4104, 16384]]}`

## Layer 2 / decode / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.938 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.938 ms`, `module_to_last_kernel=7.811 ms`, `host_to_first_kernel_gap=5.196606`, `host_end_to_last_kernel_tail=5.874 ms`, `gpu_makespan=2.615 ms`, `gpu_kernel_sum=2.098 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.229 ms
  纯GPU kernel时间: `0.151 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.229 ms`, `host_to_first_kernel_gap=5.121717`, `host_end_to_last_kernel_tail=5.047 ms`, `gpu_makespan=0.153 ms`, `gpu_kernel_sum=0.151 ms`
  开始时间(ns): `29874832`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4104, 7168]]}`
- `q_a_layernorm` -> 0.054 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.054 ms`, `host_to_first_kernel_gap=4.979832`, `host_end_to_last_kernel_tail=4.935 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `30170444`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4104, 1536]]}`
- `kv_a_layernorm` -> 0.037 ms
  纯GPU kernel时间: `0.007 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.037 ms`, `host_to_first_kernel_gap=4.909873`, `host_end_to_last_kernel_tail=4.880 ms`, `gpu_makespan=0.007 ms`, `gpu_kernel_sum=0.007 ms`
  开始时间(ns): `30249587`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4104, 512]]}`
- `q_b_proj` -> 0.214 ms
  纯GPU kernel时间: `0.235 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.214 ms`, `host_to_first_kernel_gap=4.846229`, `host_end_to_last_kernel_tail=4.868 ms`, `gpu_makespan=0.236 ms`, `gpu_kernel_sum=0.235 ms`
  开始时间(ns): `30321871`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4104, 1536]]}`
- `rotary_emb` -> 0.109 ms
  纯GPU kernel时间: `0.054 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.109 ms`, `host_to_first_kernel_gap=4.924831`, `host_end_to_last_kernel_tail=4.870 ms`, `gpu_makespan=0.054 ms`, `gpu_kernel_sum=0.054 ms`
  开始时间(ns): `30740773`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4104], [4104, 128, 64], [4104, 1, 64]]}`
- `attn_mqa` -> 0.318 ms
  纯GPU kernel时间: `0.931 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.318 ms`, `host_to_first_kernel_gap=4.822136`, `host_end_to_last_kernel_tail=5.437 ms`, `gpu_makespan=0.933 ms`, `gpu_kernel_sum=0.931 ms`
  开始时间(ns): `30899083`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4104, 128, 512], [4104, 1, 512], [4104, 1, 512]]}`
- `o_proj` -> 0.339 ms
  纯GPU kernel时间: `0.711 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.339 ms`, `host_to_first_kernel_gap=5.520107`, `host_end_to_last_kernel_tail=5.895 ms`, `gpu_makespan=0.714 ms`, `gpu_kernel_sum=0.711 ms`
  开始时间(ns): `31377367`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4104, 16384]]}`

## Layer 3 / decode / instance 1

- 执行序号: `4`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `1.929 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.929 ms`, `module_to_last_kernel=10.112 ms`, `host_to_first_kernel_gap=7.474189`, `host_end_to_last_kernel_tail=8.183 ms`, `gpu_makespan=2.637 ms`, `gpu_kernel_sum=2.116 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.245 ms
  纯GPU kernel时间: `0.146 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.245 ms`, `host_to_first_kernel_gap=7.398203`, `host_end_to_last_kernel_tail=7.302 ms`, `gpu_makespan=0.148 ms`, `gpu_kernel_sum=0.146 ms`
  开始时间(ns): `32732962`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4104, 7168]]}`
- `q_a_layernorm` -> 0.054 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.054 ms`, `host_to_first_kernel_gap=7.232836`, `host_end_to_last_kernel_tail=7.188 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `33046777`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4104, 1536]]}`
- `kv_a_layernorm` -> 0.033 ms
  纯GPU kernel时间: `0.007 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.033 ms`, `host_to_first_kernel_gap=7.16228`, `host_end_to_last_kernel_tail=7.136 ms`, `gpu_makespan=0.007 ms`, `gpu_kernel_sum=0.007 ms`
  开始时间(ns): `33126357`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4104, 512]]}`
- `q_b_proj` -> 0.214 ms
  纯GPU kernel时间: `0.240 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.214 ms`, `host_to_first_kernel_gap=7.099606`, `host_end_to_last_kernel_tail=7.128 ms`, `gpu_makespan=0.242 ms`, `gpu_kernel_sum=0.240 ms`
  开始时间(ns): `33198631`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4104, 1536]]}`
- `rotary_emb` -> 0.108 ms
  纯GPU kernel时间: `0.054 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.108 ms`, `host_to_first_kernel_gap=7.203152`, `host_end_to_last_kernel_tail=7.149 ms`, `gpu_makespan=0.054 ms`, `gpu_kernel_sum=0.054 ms`
  开始时间(ns): `33599596`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4104], [4104, 128, 64], [4104, 1, 64]]}`
- `attn_mqa` -> 0.325 ms
  纯GPU kernel时间: `0.947 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.325 ms`, `host_to_first_kernel_gap=7.100812`, `host_end_to_last_kernel_tail=7.725 ms`, `gpu_makespan=0.949 ms`, `gpu_kernel_sum=0.947 ms`
  开始时间(ns): `33758416`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4104, 128, 512], [4104, 1, 512], [4104, 1, 512]]}`
- `o_proj` -> 0.321 ms
  纯GPU kernel时间: `0.713 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.321 ms`, `host_to_first_kernel_gap=7.809856`, `host_end_to_last_kernel_tail=8.203 ms`, `gpu_makespan=0.715 ms`, `gpu_kernel_sum=0.713 ms`
  开始时间(ns): `34243738`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4104, 16384]]}`

## Layer 4 / decode / instance 1

- 执行序号: `5`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `2.057 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.057 ms`, `module_to_last_kernel=14.511 ms`, `host_to_first_kernel_gap=11.909276`, `host_end_to_last_kernel_tail=12.454 ms`, `gpu_makespan=2.602 ms`, `gpu_kernel_sum=2.087 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.311 ms
  纯GPU kernel时间: `0.145 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.311 ms`, `host_to_first_kernel_gap=11.826452`, `host_end_to_last_kernel_tail=11.663 ms`, `gpu_makespan=0.147 ms`, `gpu_kernel_sum=0.145 ms`
  开始时间(ns): `36200509`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4104, 7168]]}`
- `q_a_layernorm` -> 0.058 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.058 ms`, `host_to_first_kernel_gap=11.592316`, `host_end_to_last_kernel_tail=11.544 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `36582101`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4104, 1536]]}`
- `kv_a_layernorm` -> 0.034 ms
  纯GPU kernel时间: `0.007 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.034 ms`, `host_to_first_kernel_gap=11.505445`, `host_end_to_last_kernel_tail=11.479 ms`, `gpu_makespan=0.007 ms`, `gpu_kernel_sum=0.007 ms`
  开始时间(ns): `36678092`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4104, 512]]}`
- `q_b_proj` -> 0.243 ms
  纯GPU kernel时间: `0.236 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.243 ms`, `host_to_first_kernel_gap=11.438798`, `host_end_to_last_kernel_tail=11.433 ms`, `gpu_makespan=0.237 ms`, `gpu_kernel_sum=0.236 ms`
  开始时间(ns): `36753219`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4104, 1536]]}`
- `rotary_emb` -> 0.097 ms
  纯GPU kernel时间: `0.054 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.097 ms`, `host_to_first_kernel_gap=11.495893`, `host_end_to_last_kernel_tail=11.453 ms`, `gpu_makespan=0.054 ms`, `gpu_kernel_sum=0.054 ms`
  开始时间(ns): `37196475`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4104], [4104, 128, 64], [4104, 1, 64]]}`
- `attn_mqa` -> 0.332 ms
  纯GPU kernel时间: `0.923 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.332 ms`, `host_to_first_kernel_gap=11.403556`, `host_end_to_last_kernel_tail=11.997 ms`, `gpu_makespan=0.925 ms`, `gpu_kernel_sum=0.923 ms`
  开始时间(ns): `37343852`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4104, 128, 512], [4104, 1, 512], [4104, 1, 512]]}`
- `o_proj` -> 0.320 ms
  纯GPU kernel时间: `0.712 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.320 ms`, `host_to_first_kernel_gap=12.080091`, `host_end_to_last_kernel_tail=12.474 ms`, `gpu_makespan=0.714 ms`, `gpu_kernel_sum=0.712 ms`
  开始时间(ns): `37835123`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4104, 16384]]}`

## Layer 5 / decode / instance 1

- 执行序号: `6`
- Collector对齐边界: `{'Module': 'model.model.layers.5.self_attn'}`
- 整块 MLA-module 时长: `2.045 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.045 ms`, `module_to_last_kernel=18.805 ms`, `host_to_first_kernel_gap=16.181779`, `host_end_to_last_kernel_tail=16.761 ms`, `gpu_makespan=2.623 ms`, `gpu_kernel_sum=2.108 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.312 ms
  纯GPU kernel时间: `0.142 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.312 ms`, `host_to_first_kernel_gap=16.098407`, `host_end_to_last_kernel_tail=15.929 ms`, `gpu_makespan=0.143 ms`, `gpu_kernel_sum=0.142 ms`
  开始时间(ns): `39749822`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4104, 7168]]}`
- `q_a_layernorm` -> 0.058 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.058 ms`, `host_to_first_kernel_gap=15.860222`, `host_end_to_last_kernel_tail=15.811 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `40131303`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4104, 1536]]}`
- `kv_a_layernorm` -> 0.033 ms
  纯GPU kernel时间: `0.007 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.033 ms`, `host_to_first_kernel_gap=15.784678`, `host_end_to_last_kernel_tail=15.759 ms`, `gpu_makespan=0.007 ms`, `gpu_kernel_sum=0.007 ms`
  开始时间(ns): `40216127`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4104, 512]]}`
- `q_b_proj` -> 0.247 ms
  纯GPU kernel时间: `0.237 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.247 ms`, `host_to_first_kernel_gap=15.722161`, `host_end_to_last_kernel_tail=15.712 ms`, `gpu_makespan=0.238 ms`, `gpu_kernel_sum=0.237 ms`
  开始时间(ns): `40286996`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4104, 1536]]}`
- `rotary_emb` -> 0.102 ms
  纯GPU kernel时间: `0.055 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.102 ms`, `host_to_first_kernel_gap=15.753294`, `host_end_to_last_kernel_tail=15.706 ms`, `gpu_makespan=0.055 ms`, `gpu_kernel_sum=0.055 ms`
  开始时间(ns): `40758006`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4104], [4104, 128, 64], [4104, 1, 64]]}`
- `attn_mqa` -> 0.307 ms
  纯GPU kernel时间: `0.948 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.307 ms`, `host_to_first_kernel_gap=15.657954`, `host_end_to_last_kernel_tail=16.300 ms`, `gpu_makespan=0.949 ms`, `gpu_kernel_sum=0.948 ms`
  开始时间(ns): `40908930`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4104, 128, 512], [4104, 1, 512], [4104, 1, 512]]}`
- `o_proj` -> 0.317 ms
  纯GPU kernel时间: `0.711 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.317 ms`, `host_to_first_kernel_gap=16.386418`, `host_end_to_last_kernel_tail=16.781 ms`, `gpu_makespan=0.712 ms`, `gpu_kernel_sum=0.711 ms`
  开始时间(ns): `41373360`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4104, 16384]]}`
