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
- 整块 MLA-module 时长: `2.339 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `2`
- 模块时延拆解: `host_total=2.339 ms`, `module_to_last_kernel=2.339 ms`, `host_to_first_kernel_gap=0.303333`, `host_end_to_last_kernel_tail=0.001 ms`, `gpu_makespan=2.036 ms`, `gpu_kernel_sum=0.110 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.408 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.408 ms`, `host_to_first_kernel_gap=0.141246`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.243 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `34549434`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.052 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.052 ms`, `host_to_first_kernel_gap=0.045133`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `35025067`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `q_b_proj` -> 0.275 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.275 ms`, `host_to_first_kernel_gap=0.097611`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.167 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `35114605`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.050 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.050 ms`, `host_to_first_kernel_gap=0.044088`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `35461408`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `rotary_emb` -> 0.097 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.097 ms`, `host_to_first_kernel_gap=0.082652`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `35558460`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `kv_b_proj` -> 0.269 ms
  纯GPU kernel时间: `0.011 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.269 ms`, `host_to_first_kernel_gap=0.100264`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.147 ms`, `gpu_kernel_sum=0.011 ms`
  开始时间(ns): `35818384`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[1, 512]]}`
- `attn_mha` -> 0.191 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.191 ms`, `host_to_first_kernel_gap=0.142416`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.035 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `36204904`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 192], [1, 128, 192], [1, 128, 128]]}`
- `o_proj` -> 0.262 ms
  纯GPU kernel时间: `0.046 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.262 ms`, `host_to_first_kernel_gap=0.099044`, `host_end_to_last_kernel_tail=0.017 ms`, `gpu_makespan=0.180 ms`, `gpu_kernel_sum=0.046 ms`
  开始时间(ns): `36447476`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 1 / decode / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `2.041 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `2`
- 模块时延拆解: `host_total=2.041 ms`, `module_to_last_kernel=2.041 ms`, `host_to_first_kernel_gap=0.232277`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=1.809 ms`, `gpu_kernel_sum=0.110 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.277 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.277 ms`, `host_to_first_kernel_gap=0.102981`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.160 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `37793427`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.051 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.051 ms`, `host_to_first_kernel_gap=0.044428`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `38127820`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `q_b_proj` -> 0.256 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.256 ms`, `host_to_first_kernel_gap=0.101695`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.147 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `38215225`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.047 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.047 ms`, `host_to_first_kernel_gap=0.041252`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `38541716`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `rotary_emb` -> 0.083 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.083 ms`, `host_to_first_kernel_gap=0.069926`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `38631250`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `kv_b_proj` -> 0.247 ms
  纯GPU kernel时间: `0.010 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.247 ms`, `host_to_first_kernel_gap=0.092167`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.139 ms`, `gpu_kernel_sum=0.010 ms`
  开始时间(ns): `38864753`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[1, 512]]}`
- `attn_mha` -> 0.172 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.172 ms`, `host_to_first_kernel_gap=0.125693`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.036 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `39217723`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 192], [1, 128, 192], [1, 128, 128]]}`
- `o_proj` -> 0.249 ms
  纯GPU kernel时间: `0.046 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.249 ms`, `host_to_first_kernel_gap=0.094588`, `host_end_to_last_kernel_tail=0.019 ms`, `gpu_makespan=0.173 ms`, `gpu_kernel_sum=0.046 ms`
  开始时间(ns): `39437308`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 2 / decode / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.981 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `2`
- 模块时延拆解: `host_total=1.981 ms`, `module_to_last_kernel=1.984 ms`, `host_to_first_kernel_gap=0.207556`, `host_end_to_last_kernel_tail=0.002 ms`, `gpu_makespan=1.776 ms`, `gpu_kernel_sum=0.107 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.238 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.238 ms`, `host_to_first_kernel_gap=0.090684`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.138 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `40725372`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.049 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.049 ms`, `host_to_first_kernel_gap=0.042786`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `41017238`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `q_b_proj` -> 0.264 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.264 ms`, `host_to_first_kernel_gap=0.092964`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.163 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `41100372`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.046 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.046 ms`, `host_to_first_kernel_gap=0.040765`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `41433819`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `rotary_emb` -> 0.089 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.089 ms`, `host_to_first_kernel_gap=0.076471`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `41521601`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `kv_b_proj` -> 0.247 ms
  纯GPU kernel时间: `0.010 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.247 ms`, `host_to_first_kernel_gap=0.091907`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.140 ms`, `gpu_kernel_sum=0.010 ms`
  开始时间(ns): `41756245`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[1, 512]]}`
- `attn_mha` -> 0.164 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.164 ms`, `host_to_first_kernel_gap=0.114517`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.040 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `42109635`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 192], [1, 128, 192], [1, 128, 128]]}`
- `o_proj` -> 0.255 ms
  纯GPU kernel时间: `0.044 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.255 ms`, `host_to_first_kernel_gap=0.097906`, `host_end_to_last_kernel_tail=0.017 ms`, `gpu_makespan=0.175 ms`, `gpu_kernel_sum=0.044 ms`
  开始时间(ns): `42319750`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 3 / decode / instance 1

- 执行序号: `4`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `1.989 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `2`
- 模块时延拆解: `host_total=1.989 ms`, `module_to_last_kernel=1.992 ms`, `host_to_first_kernel_gap=0.221039`, `host_end_to_last_kernel_tail=0.004 ms`, `gpu_makespan=1.771 ms`, `gpu_kernel_sum=0.108 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.266 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.266 ms`, `host_to_first_kernel_gap=0.098713`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.154 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `43604862`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.051 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.051 ms`, `host_to_first_kernel_gap=0.044773`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `43926258`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `q_b_proj` -> 0.251 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.251 ms`, `host_to_first_kernel_gap=0.088515`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.154 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `44015956`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.045 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.045 ms`, `host_to_first_kernel_gap=0.039488`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `44334743`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `rotary_emb` -> 0.083 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.083 ms`, `host_to_first_kernel_gap=0.067701`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `44421058`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `kv_b_proj` -> 0.243 ms
  纯GPU kernel时间: `0.010 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.243 ms`, `host_to_first_kernel_gap=0.089415`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.139 ms`, `gpu_kernel_sum=0.010 ms`
  开始时间(ns): `44651696`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[1, 512]]}`
- `attn_mha` -> 0.155 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.155 ms`, `host_to_first_kernel_gap=0.114443`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.031 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `44998348`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 192], [1, 128, 192], [1, 128, 128]]}`
- `o_proj` -> 0.256 ms
  纯GPU kernel时间: `0.045 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.256 ms`, `host_to_first_kernel_gap=0.099888`, `host_end_to_last_kernel_tail=0.018 ms`, `gpu_makespan=0.174 ms`, `gpu_kernel_sum=0.045 ms`
  开始时间(ns): `45201031`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 4 / decode / instance 1

- 执行序号: `5`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `2.153 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `2`
- 模块时延拆解: `host_total=2.153 ms`, `module_to_last_kernel=2.153 ms`, `host_to_first_kernel_gap=0.249946`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=1.900 ms`, `gpu_kernel_sum=0.109 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.315 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.315 ms`, `host_to_first_kernel_gap=0.110464`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.189 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `47291671`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.055 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.055 ms`, `host_to_first_kernel_gap=0.047791`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `47674024`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `q_b_proj` -> 0.253 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.253 ms`, `host_to_first_kernel_gap=0.09174`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.152 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `47765083`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.046 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.046 ms`, `host_to_first_kernel_gap=0.040762`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `48087357`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `rotary_emb` -> 0.099 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.099 ms`, `host_to_first_kernel_gap=0.081943`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `48176384`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `kv_b_proj` -> 0.260 ms
  纯GPU kernel时间: `0.011 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.260 ms`, `host_to_first_kernel_gap=0.095332`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.148 ms`, `gpu_kernel_sum=0.011 ms`
  开始时间(ns): `48420563`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[1, 512]]}`
- `attn_mha` -> 0.166 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.166 ms`, `host_to_first_kernel_gap=0.122756`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.032 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `48787635`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 192], [1, 128, 192], [1, 128, 128]]}`
- `o_proj` -> 0.286 ms
  纯GPU kernel时间: `0.045 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.286 ms`, `host_to_first_kernel_gap=0.103605`, `host_end_to_last_kernel_tail=0.013 ms`, `gpu_makespan=0.196 ms`, `gpu_kernel_sum=0.045 ms`
  开始时间(ns): `49002466`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 0 / decode / instance 2

- 执行序号: `6`
- Collector对齐边界: `{'Module': 'model.model.layers.0.self_attn'}`
- 整块 MLA-module 时长: `2.037 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `2`
- 模块时延拆解: `host_total=2.037 ms`, `module_to_last_kernel=2.037 ms`, `host_to_first_kernel_gap=0.199971`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=1.831 ms`, `gpu_kernel_sum=0.099 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.345 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.345 ms`, `host_to_first_kernel_gap=0.119312`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.208 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `53618631`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.049 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.049 ms`, `host_to_first_kernel_gap=0.042614`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `54034561`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.034 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.034 ms`, `host_to_first_kernel_gap=0.030207`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `54111736`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.285 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.285 ms`, `host_to_first_kernel_gap=0.104766`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.168 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `54188441`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.092 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.092 ms`, `host_to_first_kernel_gap=0.07721`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `54660957`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.299 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.299 ms`, `host_to_first_kernel_gap=0.104494`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.174 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `54812361`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.293 ms
  纯GPU kernel时间: `0.046 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.293 ms`, `host_to_first_kernel_gap=0.106099`, `host_end_to_last_kernel_tail=0.014 ms`, `gpu_makespan=0.201 ms`, `gpu_kernel_sum=0.046 ms`
  开始时间(ns): `55262084`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 1 / decode / instance 2

- 执行序号: `7`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `1.814 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `2`
- 模块时延拆解: `host_total=1.814 ms`, `module_to_last_kernel=1.814 ms`, `host_to_first_kernel_gap=0.160182`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=1.650 ms`, `gpu_kernel_sum=0.100 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.240 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.240 ms`, `host_to_first_kernel_gap=0.092943`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.138 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `56590376`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.049 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.049 ms`, `host_to_first_kernel_gap=0.04241`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `56894509`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.034 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.034 ms`, `host_to_first_kernel_gap=0.029605`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `56969906`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.241 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.241 ms`, `host_to_first_kernel_gap=0.083201`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.151 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `57041526`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.086 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.086 ms`, `host_to_first_kernel_gap=0.072143`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `57437544`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.295 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.295 ms`, `host_to_first_kernel_gap=0.105286`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.171 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `57574161`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.312 ms
  纯GPU kernel时间: `0.048 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.312 ms`, `host_to_first_kernel_gap=0.108723`, `host_end_to_last_kernel_tail=0.016 ms`, `gpu_makespan=0.220 ms`, `gpu_kernel_sum=0.048 ms`
  开始时间(ns): `58004899`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 2 / decode / instance 2

- 执行序号: `8`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.785 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `2`
- 模块时延拆解: `host_total=1.785 ms`, `module_to_last_kernel=1.785 ms`, `host_to_first_kernel_gap=0.161476`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=1.619 ms`, `gpu_kernel_sum=0.098 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.242 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.242 ms`, `host_to_first_kernel_gap=0.093113`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.139 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `59332541`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.050 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.050 ms`, `host_to_first_kernel_gap=0.043806`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `59638104`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.034 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.034 ms`, `host_to_first_kernel_gap=0.029917`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `59715353`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.230 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.230 ms`, `host_to_first_kernel_gap=0.084501`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.139 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `59785441`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.095 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.095 ms`, `host_to_first_kernel_gap=0.082715`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `60167963`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.277 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.277 ms`, `host_to_first_kernel_gap=0.096428`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.163 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `60311562`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.297 ms
  纯GPU kernel时间: `0.046 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.297 ms`, `host_to_first_kernel_gap=0.105887`, `host_end_to_last_kernel_tail=0.014 ms`, `gpu_makespan=0.205 ms`, `gpu_kernel_sum=0.046 ms`
  开始时间(ns): `60733207`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 3 / decode / instance 2

- 执行序号: `9`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `1.801 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `2`
- 模块时延拆解: `host_total=1.801 ms`, `module_to_last_kernel=1.801 ms`, `host_to_first_kernel_gap=0.158956`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=1.639 ms`, `gpu_kernel_sum=0.099 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.246 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.246 ms`, `host_to_first_kernel_gap=0.090677`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.145 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `62054625`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.054 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.054 ms`, `host_to_first_kernel_gap=0.047061`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `62365505`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.037 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.037 ms`, `host_to_first_kernel_gap=0.029722`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `62446972`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.226 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.226 ms`, `host_to_first_kernel_gap=0.083298`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.136 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `62520980`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.086 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.086 ms`, `host_to_first_kernel_gap=0.072737`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `62902485`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.291 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.291 ms`, `host_to_first_kernel_gap=0.100713`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.173 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `63038157`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.307 ms
  纯GPU kernel时间: `0.047 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.307 ms`, `host_to_first_kernel_gap=0.113468`, `host_end_to_last_kernel_tail=0.016 ms`, `gpu_makespan=0.209 ms`, `gpu_kernel_sum=0.047 ms`
  开始时间(ns): `63461722`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 4 / decode / instance 2

- 执行序号: `10`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `1.908 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `2`
- 模块时延拆解: `host_total=1.908 ms`, `module_to_last_kernel=1.908 ms`, `host_to_first_kernel_gap=0.182463`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=1.712 ms`, `gpu_kernel_sum=0.100 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.302 ms
  纯GPU kernel时间: `0.016 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.302 ms`, `host_to_first_kernel_gap=0.106459`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.181 ms`, `gpu_kernel_sum=0.016 ms`
  开始时间(ns): `65437819`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.058 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.058 ms`, `host_to_first_kernel_gap=0.051449`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `65809597`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.036 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.036 ms`, `host_to_first_kernel_gap=0.031356`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `65896186`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.262 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.262 ms`, `host_to_first_kernel_gap=0.089331`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.157 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `65971747`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.088 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.088 ms`, `host_to_first_kernel_gap=0.073997`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `66404041`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.276 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.276 ms`, `host_to_first_kernel_gap=0.094138`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.165 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `66545276`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.296 ms
  纯GPU kernel时间: `0.046 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.296 ms`, `host_to_first_kernel_gap=0.106358`, `host_end_to_last_kernel_tail=0.008 ms`, `gpu_makespan=0.197 ms`, `gpu_kernel_sum=0.046 ms`
  开始时间(ns): `66952704`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`
