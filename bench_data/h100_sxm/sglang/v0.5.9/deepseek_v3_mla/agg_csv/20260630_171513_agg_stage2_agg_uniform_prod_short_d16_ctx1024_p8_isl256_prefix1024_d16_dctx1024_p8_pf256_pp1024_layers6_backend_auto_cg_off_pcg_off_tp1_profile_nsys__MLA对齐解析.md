# MLA对齐解析摘要

源文件: `report.sqlite`

## 对齐原则

- `collector/sglang/collect_mla_module.py` 的 MLA module 计时边界是 `model.model.layers[test_layer].self_attn(...)`。
- 因此这里把 `nsys` 中每层的 `model.model.layers.X.self_attn` NVTX range 视为与 collector 对齐的 MLA-module 边界。
- 该区间内部的 `.self_attn.*` 子模块用于做 MLA 内部 breakdown。

## 运行摘要

- `prefill` 对齐成功层: `[0, 1, 2, 3, 4, 5]`
- `prefill` 被切分层: `[]`
- 若某层 `prefill` 出现多个 `self_attn` 实例，则说明 Engine 调度把一次前向切成了多块，已不再与 collector 的单次 MLA-module 采集严格一一对应。

## Layer 0 / prefill / instance 1

- 执行序号: `1`
- Collector对齐边界: `{'Module': 'model.model.layers.0.self_attn'}`
- 整块 MLA-module 时长: `3.728 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=3.728 ms`, `module_to_last_kernel=5.884 ms`, `host_to_first_kernel_gap=0.425014`, `host_end_to_last_kernel_tail=2.156 ms`, `gpu_makespan=5.459 ms`, `gpu_kernel_sum=2.414 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=2064`, `total_tokens=26711`, `chunked_req_prefix_len=24647`, `current_chunked_req_prefix_len=24647`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[2064, 128, 192], [26711, 128, 192], [26711, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.443 ms
  纯GPU kernel时间: `0.081 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.443 ms`, `host_to_first_kernel_gap=0.147729`, `host_end_to_last_kernel_tail=0.022 ms`, `gpu_makespan=0.317 ms`, `gpu_kernel_sum=0.081 ms`
  开始时间(ns): `24256498`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2064, 7168]]}`
- `q_a_layernorm` -> 0.058 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.058 ms`, `host_to_first_kernel_gap=0.051975`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `24765275`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2064, 1536]]}`
- `q_b_proj` -> 0.244 ms
  纯GPU kernel时间: `0.136 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.244 ms`, `host_to_first_kernel_gap=0.086199`, `host_end_to_last_kernel_tail=0.105 ms`, `gpu_makespan=0.262 ms`, `gpu_kernel_sum=0.136 ms`
  开始时间(ns): `24860555`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2064, 1536]]}`
- `kv_a_layernorm` -> 0.043 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.043 ms`, `host_to_first_kernel_gap=0.041287`, `host_end_to_last_kernel_tail=0.003 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `25167547`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2064, 512]]}`
- `rotary_emb` -> 0.102 ms
  纯GPU kernel时间: `0.026 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.102 ms`, `host_to_first_kernel_gap=0.087084`, `host_end_to_last_kernel_tail=0.011 ms`, `gpu_makespan=0.026 ms`, `gpu_kernel_sum=0.026 ms`
  开始时间(ns): `25254070`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2064], [2064, 128, 64], [2064, 1, 64]]}`
- `kv_b_proj` -> 1.060 ms
  纯GPU kernel时间: `0.939 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=1.060 ms`, `host_to_first_kernel_gap=0.093883`, `host_end_to_last_kernel_tail=0.870 ms`, `gpu_makespan=1.837 ms`, `gpu_kernel_sum=0.939 ms`
  开始时间(ns): `25932550`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[26711, 512]]}`
- `attn_mha` -> 0.200 ms
  纯GPU kernel时间: `0.857 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.200 ms`, `host_to_first_kernel_gap=1.483859`, `host_end_to_last_kernel_tail=2.140 ms`, `gpu_makespan=0.857 ms`, `gpu_kernel_sum=0.857 ms`
  开始时间(ns): `27154476`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2064, 128, 192], [26711, 128, 192], [26711, 128, 128]]}`
- `o_proj` -> 0.287 ms
  纯GPU kernel时间: `0.366 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.287 ms`, `host_to_first_kernel_gap=2.092717`, `host_end_to_last_kernel_tail=2.173 ms`, `gpu_makespan=0.367 ms`, `gpu_kernel_sum=0.366 ms`
  开始时间(ns): `27403505`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2064, 16384]]}`

## Layer 1 / prefill / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `2.041 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.041 ms`, `module_to_last_kernel=5.897 ms`, `host_to_first_kernel_gap=2.611769`, `host_end_to_last_kernel_tail=3.856 ms`, `gpu_makespan=3.285 ms`, `gpu_kernel_sum=2.458 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=2064`, `total_tokens=26711`, `chunked_req_prefix_len=24647`, `current_chunked_req_prefix_len=24647`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[2064, 128, 192], [26711, 128, 192], [26711, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.234 ms
  纯GPU kernel时间: `0.080 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.234 ms`, `host_to_first_kernel_gap=2.39388`, `host_end_to_last_kernel_tail=2.243 ms`, `gpu_makespan=0.082 ms`, `gpu_kernel_sum=0.080 ms`
  开始时间(ns): `28811301`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2064, 7168]]}`
- `q_a_layernorm` -> 0.049 ms
  纯GPU kernel时间: `0.006 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.049 ms`, `host_to_first_kernel_gap=2.193052`, `host_end_to_last_kernel_tail=2.149 ms`, `gpu_makespan=0.006 ms`, `gpu_kernel_sum=0.006 ms`
  开始时间(ns): `29094305`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2064, 1536]]}`
- `q_b_proj` -> 0.207 ms
  纯GPU kernel时间: `0.130 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.207 ms`, `host_to_first_kernel_gap=2.12079`, `host_end_to_last_kernel_tail=2.046 ms`, `gpu_makespan=0.131 ms`, `gpu_kernel_sum=0.130 ms`
  开始时间(ns): `29173671`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2064, 1536]]}`
- `kv_a_layernorm` -> 0.040 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.040 ms`, `host_to_first_kernel_gap=1.985141`, `host_end_to_last_kernel_tail=1.951 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `29440648`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2064, 512]]}`
- `rotary_emb` -> 0.088 ms
  纯GPU kernel时间: `0.027 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.088 ms`, `host_to_first_kernel_gap=1.913282`, `host_end_to_last_kernel_tail=1.852 ms`, `gpu_makespan=0.027 ms`, `gpu_kernel_sum=0.027 ms`
  开始时间(ns): `29519611`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2064], [2064, 128, 64], [2064, 1, 64]]}`
- `kv_b_proj` -> 0.243 ms
  纯GPU kernel时间: `0.987 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.243 ms`, `host_to_first_kernel_gap=1.647547`, `host_end_to_last_kernel_tail=2.394 ms`, `gpu_makespan=0.989 ms`, `gpu_kernel_sum=0.987 ms`
  开始时间(ns): `29853153`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[26711, 512]]}`
- `attn_mha` -> 0.147 ms
  纯GPU kernel时间: `0.849 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.147 ms`, `host_to_first_kernel_gap=3.067417`, `host_end_to_last_kernel_tail=3.769 ms`, `gpu_makespan=0.849 ms`, `gpu_kernel_sum=0.849 ms`
  开始时间(ns): `30196802`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2064, 128, 192], [26711, 128, 192], [26711, 128, 128]]}`
- `o_proj` -> 0.234 ms
  纯GPU kernel时间: `0.374 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.234 ms`, `host_to_first_kernel_gap=3.729776`, `host_end_to_last_kernel_tail=3.871 ms`, `gpu_makespan=0.375 ms`, `gpu_kernel_sum=0.374 ms`
  开始时间(ns): `30385482`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2064, 16384]]}`

## Layer 2 / prefill / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.947 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.947 ms`, `module_to_last_kernel=7.536 ms`, `host_to_first_kernel_gap=4.367196`, `host_end_to_last_kernel_tail=5.589 ms`, `gpu_makespan=3.169 ms`, `gpu_kernel_sum=2.337 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=2064`, `total_tokens=26711`, `chunked_req_prefix_len=24647`, `current_chunked_req_prefix_len=24647`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[2064, 128, 192], [26711, 128, 192], [26711, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.232 ms
  纯GPU kernel时间: `0.081 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.232 ms`, `host_to_first_kernel_gap=4.179351`, `host_end_to_last_kernel_tail=4.030 ms`, `gpu_makespan=0.082 ms`, `gpu_kernel_sum=0.081 ms`
  开始时间(ns): `31643426`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2064, 7168]]}`
- `q_a_layernorm` -> 0.045 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.045 ms`, `host_to_first_kernel_gap=3.980436`, `host_end_to_last_kernel_tail=3.941 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `31924165`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2064, 1536]]}`
- `q_b_proj` -> 0.202 ms
  纯GPU kernel时间: `0.131 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.202 ms`, `host_to_first_kernel_gap=3.914948`, `host_end_to_last_kernel_tail=3.846 ms`, `gpu_makespan=0.133 ms`, `gpu_kernel_sum=0.131 ms`
  开始时间(ns): `31997429`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2064, 1536]]}`
- `kv_a_layernorm` -> 0.043 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.043 ms`, `host_to_first_kernel_gap=3.785666`, `host_end_to_last_kernel_tail=3.748 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `32259095`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2064, 512]]}`
- `rotary_emb` -> 0.077 ms
  纯GPU kernel时间: `0.026 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.077 ms`, `host_to_first_kernel_gap=3.711292`, `host_end_to_last_kernel_tail=3.661 ms`, `gpu_makespan=0.026 ms`, `gpu_kernel_sum=0.026 ms`
  开始时间(ns): `32340605`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2064], [2064, 128, 64], [2064, 1, 64]]}`
- `kv_b_proj` -> 0.251 ms
  纯GPU kernel时间: `0.861 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.251 ms`, `host_to_first_kernel_gap=3.489361`, `host_end_to_last_kernel_tail=4.102 ms`, `gpu_makespan=0.863 ms`, `gpu_kernel_sum=0.861 ms`
  开始时间(ns): `32631336`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[26711, 512]]}`
- `attn_mha` -> 0.142 ms
  纯GPU kernel时间: `0.854 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.142 ms`, `host_to_first_kernel_gap=4.784219`, `host_end_to_last_kernel_tail=5.496 ms`, `gpu_makespan=0.854 ms`, `gpu_kernel_sum=0.854 ms`
  开始时间(ns): `32976924`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2064, 128, 192], [26711, 128, 192], [26711, 128, 128]]}`
- `o_proj` -> 0.228 ms
  纯GPU kernel时间: `0.373 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.228 ms`, `host_to_first_kernel_gap=5.457976`, `host_end_to_last_kernel_tail=5.605 ms`, `gpu_makespan=0.375 ms`, `gpu_kernel_sum=0.373 ms`
  开始时间(ns): `33159230`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2064, 16384]]}`

## Layer 3 / prefill / instance 1

- 执行序号: `4`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `1.893 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.893 ms`, `module_to_last_kernel=9.373 ms`, `host_to_first_kernel_gap=6.13317`, `host_end_to_last_kernel_tail=7.480 ms`, `gpu_makespan=3.240 ms`, `gpu_kernel_sum=2.411 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=2064`, `total_tokens=26711`, `chunked_req_prefix_len=24647`, `current_chunked_req_prefix_len=24647`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[2064, 128, 192], [26711, 128, 192], [26711, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.224 ms
  纯GPU kernel时间: `0.080 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.224 ms`, `host_to_first_kernel_gap=5.948255`, `host_end_to_last_kernel_tail=5.806 ms`, `gpu_makespan=0.081 ms`, `gpu_kernel_sum=0.080 ms`
  开始时间(ns): `34384662`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2064, 7168]]}`
- `q_a_layernorm` -> 0.046 ms
  纯GPU kernel时间: `0.006 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.046 ms`, `host_to_first_kernel_gap=5.756897`, `host_end_to_last_kernel_tail=5.716 ms`, `gpu_makespan=0.006 ms`, `gpu_kernel_sum=0.006 ms`
  开始时间(ns): `34657652`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2064, 1536]]}`
- `q_b_proj` -> 0.217 ms
  纯GPU kernel时间: `0.131 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.217 ms`, `host_to_first_kernel_gap=5.687432`, `host_end_to_last_kernel_tail=5.604 ms`, `gpu_makespan=0.133 ms`, `gpu_kernel_sum=0.131 ms`
  开始时间(ns): `34734701`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2064, 1536]]}`
- `kv_a_layernorm` -> 0.039 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.039 ms`, `host_to_first_kernel_gap=5.544336`, `host_end_to_last_kernel_tail=5.510 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `35010725`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2064, 512]]}`
- `rotary_emb` -> 0.075 ms
  纯GPU kernel时间: `0.026 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.075 ms`, `host_to_first_kernel_gap=5.469183`, `host_end_to_last_kernel_tail=5.420 ms`, `gpu_makespan=0.026 ms`, `gpu_kernel_sum=0.026 ms`
  开始时间(ns): `35091798`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2064], [2064, 128, 64], [2064, 1, 64]]}`
- `kv_b_proj` -> 0.215 ms
  纯GPU kernel时间: `0.938 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.215 ms`, `host_to_first_kernel_gap=5.252927`, `host_end_to_last_kernel_tail=5.977 ms`, `gpu_makespan=0.939 ms`, `gpu_kernel_sum=0.938 ms`
  开始时间(ns): `35376182`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[26711, 512]]}`
- `attn_mha` -> 0.140 ms
  纯GPU kernel时间: `0.855 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.140 ms`, `host_to_first_kernel_gap=6.661291`, `host_end_to_last_kernel_tail=7.376 ms`, `gpu_makespan=0.855 ms`, `gpu_kernel_sum=0.855 ms`
  开始时间(ns): `35682888`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2064, 128, 192], [26711, 128, 192], [26711, 128, 128]]}`
- `o_proj` -> 0.217 ms
  纯GPU kernel时间: `0.371 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.217 ms`, `host_to_first_kernel_gap=7.338164`, `host_end_to_last_kernel_tail=7.494 ms`, `gpu_makespan=0.373 ms`, `gpu_kernel_sum=0.371 ms`
  开始时间(ns): `35862078`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2064, 16384]]}`

## Layer 4 / prefill / instance 1

- 执行序号: `5`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `2.043 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.043 ms`, `module_to_last_kernel=11.912 ms`, `host_to_first_kernel_gap=8.659317`, `host_end_to_last_kernel_tail=9.869 ms`, `gpu_makespan=3.253 ms`, `gpu_kernel_sum=2.426 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=2064`, `total_tokens=26711`, `chunked_req_prefix_len=24647`, `current_chunked_req_prefix_len=24647`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[2064, 128, 192], [26711, 128, 192], [26711, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.288 ms
  纯GPU kernel时间: `0.081 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.288 ms`, `host_to_first_kernel_gap=8.459597`, `host_end_to_last_kernel_tail=8.253 ms`, `gpu_makespan=0.082 ms`, `gpu_kernel_sum=0.081 ms`
  开始时间(ns): `37760995`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2064, 7168]]}`
- `q_a_layernorm` -> 0.050 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.050 ms`, `host_to_first_kernel_gap=8.201209`, `host_end_to_last_kernel_tail=8.157 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `38101175`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2064, 1536]]}`
- `q_b_proj` -> 0.224 ms
  纯GPU kernel时间: `0.131 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.224 ms`, `host_to_first_kernel_gap=8.126215`, `host_end_to_last_kernel_tail=8.034 ms`, `gpu_makespan=0.132 ms`, `gpu_kernel_sum=0.131 ms`
  开始时间(ns): `38183593`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2064, 1536]]}`
- `kv_a_layernorm` -> 0.042 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.042 ms`, `host_to_first_kernel_gap=7.969415`, `host_end_to_last_kernel_tail=7.933 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `38472393`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2064, 512]]}`
- `rotary_emb` -> 0.075 ms
  纯GPU kernel时间: `0.026 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.075 ms`, `host_to_first_kernel_gap=7.895572`, `host_end_to_last_kernel_tail=7.846 ms`, `gpu_makespan=0.026 ms`, `gpu_kernel_sum=0.026 ms`
  开始时间(ns): `38553212`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2064], [2064, 128, 64], [2064, 1, 64]]}`
- `kv_b_proj` -> 0.225 ms
  纯GPU kernel时间: `0.947 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.225 ms`, `host_to_first_kernel_gap=7.67566`, `host_end_to_last_kernel_tail=8.400 ms`, `gpu_makespan=0.949 ms`, `gpu_kernel_sum=0.947 ms`
  开始时间(ns): `38839908`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[26711, 512]]}`
- `attn_mha` -> 0.142 ms
  纯GPU kernel时间: `0.856 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.142 ms`, `host_to_first_kernel_gap=9.079145`, `host_end_to_last_kernel_tail=9.793 ms`, `gpu_makespan=0.856 ms`, `gpu_kernel_sum=0.856 ms`
  开始时间(ns): `39159429`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2064, 128, 192], [26711, 128, 192], [26711, 128, 128]]}`
- `o_proj` -> 0.245 ms
  纯GPU kernel时间: `0.374 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.245 ms`, `host_to_first_kernel_gap=9.752128`, `host_end_to_last_kernel_tail=9.884 ms`, `gpu_makespan=0.376 ms`, `gpu_kernel_sum=0.374 ms`
  开始时间(ns): `39344685`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2064, 16384]]}`

## Layer 5 / prefill / instance 1

- 执行序号: `6`
- Collector对齐边界: `{'Module': 'model.model.layers.5.self_attn'}`
- 整块 MLA-module 时长: `2.023 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.023 ms`, `module_to_last_kernel=14.485 ms`, `host_to_first_kernel_gap=11.20687`, `host_end_to_last_kernel_tail=12.462 ms`, `gpu_makespan=3.278 ms`, `gpu_kernel_sum=2.452 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=2064`, `total_tokens=26711`, `chunked_req_prefix_len=24647`, `current_chunked_req_prefix_len=24647`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[2064, 128, 192], [26711, 128, 192], [26711, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.279 ms
  纯GPU kernel时间: `0.081 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.279 ms`, `host_to_first_kernel_gap=11.011144`, `host_end_to_last_kernel_tail=10.815 ms`, `gpu_makespan=0.083 ms`, `gpu_kernel_sum=0.081 ms`
  开始时间(ns): `41119875`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2064, 7168]]}`
- `q_a_layernorm` -> 0.050 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.050 ms`, `host_to_first_kernel_gap=10.762041`, `host_end_to_last_kernel_tail=10.718 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `41451698`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2064, 1536]]}`
- `q_b_proj` -> 0.225 ms
  纯GPU kernel时间: `0.131 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.225 ms`, `host_to_first_kernel_gap=10.688525`, `host_end_to_last_kernel_tail=10.595 ms`, `gpu_makespan=0.132 ms`, `gpu_kernel_sum=0.131 ms`
  开始时间(ns): `41531742`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2064, 1536]]}`
- `kv_a_layernorm` -> 0.041 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.041 ms`, `host_to_first_kernel_gap=10.525683`, `host_end_to_last_kernel_tail=10.490 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `41826072`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2064, 512]]}`
- `rotary_emb` -> 0.075 ms
  纯GPU kernel时间: `0.026 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.075 ms`, `host_to_first_kernel_gap=10.451004`, `host_end_to_last_kernel_tail=10.402 ms`, `gpu_makespan=0.026 ms`, `gpu_kernel_sum=0.026 ms`
  开始时间(ns): `41906991`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2064], [2064, 128, 64], [2064, 1, 64]]}`
- `kv_b_proj` -> 0.225 ms
  纯GPU kernel时间: `0.984 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.225 ms`, `host_to_first_kernel_gap=10.22843`, `host_end_to_last_kernel_tail=10.988 ms`, `gpu_makespan=0.985 ms`, `gpu_kernel_sum=0.984 ms`
  开始时间(ns): `42196669`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[26711, 512]]}`
- `attn_mha` -> 0.142 ms
  纯GPU kernel时间: `0.857 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.142 ms`, `host_to_first_kernel_gap=11.665982`, `host_end_to_last_kernel_tail=12.381 ms`, `gpu_makespan=0.857 ms`, `gpu_kernel_sum=0.857 ms`
  开始时间(ns): `42520043`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2064, 128, 192], [26711, 128, 192], [26711, 128, 128]]}`
- `o_proj` -> 0.233 ms
  纯GPU kernel时间: `0.364 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.233 ms`, `host_to_first_kernel_gap=12.344109`, `host_end_to_last_kernel_tail=12.476 ms`, `gpu_makespan=0.365 ms`, `gpu_kernel_sum=0.364 ms`
  开始时间(ns): `42699995`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2064, 16384]]}`
