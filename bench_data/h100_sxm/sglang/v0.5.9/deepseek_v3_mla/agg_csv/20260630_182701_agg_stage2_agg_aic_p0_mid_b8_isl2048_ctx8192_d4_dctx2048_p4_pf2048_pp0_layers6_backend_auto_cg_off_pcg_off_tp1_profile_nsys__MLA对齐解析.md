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
- 整块 MLA-module 时长: `1.941 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.941 ms`, `module_to_last_kernel=5.078 ms`, `host_to_first_kernel_gap=0.239207`, `host_end_to_last_kernel_tail=3.137 ms`, `gpu_makespan=4.839 ms`, `gpu_kernel_sum=3.985 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=16397`, `chunked_req_prefix_len=8205`, `current_chunked_req_prefix_len=8205`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[8192, 128, 192], [16397, 128, 192], [16397, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.308 ms
  纯GPU kernel时间: `0.274 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.308 ms`, `host_to_first_kernel_gap=0.103894`, `host_end_to_last_kernel_tail=0.176 ms`, `gpu_makespan=0.380 ms`, `gpu_kernel_sum=0.274 ms`
  开始时间(ns): `19276013`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.039 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.039 ms`, `host_to_first_kernel_gap=0.133078`, `host_end_to_last_kernel_tail=0.111 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `19627021`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.167 ms
  纯GPU kernel时间: `0.449 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.167 ms`, `host_to_first_kernel_gap=0.087169`, `host_end_to_last_kernel_tail=0.415 ms`, `gpu_makespan=0.495 ms`, `gpu_kernel_sum=0.449 ms`
  开始时间(ns): `19691586`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.028 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.028 ms`, `host_to_first_kernel_gap=0.3754`, `host_end_to_last_kernel_tail=0.360 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `19898683`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.088 ms
  纯GPU kernel时间: `0.106 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.088 ms`, `host_to_first_kernel_gap=0.332267`, `host_end_to_last_kernel_tail=0.350 ms`, `gpu_makespan=0.106 ms`, `gpu_kernel_sum=0.106 ms`
  开始时间(ns): `19955736`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.209 ms
  纯GPU kernel时间: `0.545 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.209 ms`, `host_to_first_kernel_gap=0.069553`, `host_end_to_last_kernel_tail=0.515 ms`, `gpu_makespan=0.655 ms`, `gpu_kernel_sum=0.545 ms`
  开始时间(ns): `20429074`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[16397, 512]]}`
- `attn_mha` -> 0.132 ms
  纯GPU kernel时间: `1.325 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.132 ms`, `host_to_first_kernel_gap=0.913318`, `host_end_to_last_kernel_tail=2.106 ms`, `gpu_makespan=1.325 ms`, `gpu_kernel_sum=1.325 ms`
  开始时间(ns): `20720573`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [16397, 128, 192], [16397, 128, 128]]}`
- `o_proj` -> 0.188 ms
  纯GPU kernel时间: `1.257 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.188 ms`, `host_to_first_kernel_gap=2.077854`, `host_end_to_last_kernel_tail=3.148 ms`, `gpu_makespan=1.258 ms`, `gpu_kernel_sum=1.257 ms`
  开始时间(ns): `20882693`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 1 / prefill / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `1.417 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.417 ms`, `module_to_last_kernel=11.722 ms`, `host_to_first_kernel_gap=7.18142`, `host_end_to_last_kernel_tail=10.305 ms`, `gpu_makespan=4.540 ms`, `gpu_kernel_sum=4.000 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=16397`, `chunked_req_prefix_len=8205`, `current_chunked_req_prefix_len=8205`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[8192, 128, 192], [16397, 128, 192], [16397, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.148 ms
  纯GPU kernel时间: `0.267 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.148 ms`, `host_to_first_kernel_gap=7.066866`, `host_end_to_last_kernel_tail=7.189 ms`, `gpu_makespan=0.270 ms`, `gpu_kernel_sum=0.267 ms`
  开始时间(ns): `21760527`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.047 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.047 ms`, `host_to_first_kernel_gap=7.149659`, `host_end_to_last_kernel_tail=7.121 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `21948006`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.153 ms
  纯GPU kernel时间: `0.445 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.153 ms`, `host_to_first_kernel_gap=7.102976`, `host_end_to_last_kernel_tail=7.397 ms`, `gpu_makespan=0.447 ms`, `gpu_kernel_sum=0.445 ms`
  开始时间(ns): `22014785`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.028 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.028 ms`, `host_to_first_kernel_gap=7.360219`, `host_end_to_last_kernel_tail=7.345 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `22204774`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.070 ms
  纯GPU kernel时间: `0.106 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.070 ms`, `host_to_first_kernel_gap=7.321349`, `host_end_to_last_kernel_tail=7.357 ms`, `gpu_makespan=0.106 ms`, `gpu_kernel_sum=0.106 ms`
  开始时间(ns): `22258268`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.161 ms
  纯GPU kernel时间: `0.566 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.161 ms`, `host_to_first_kernel_gap=7.228575`, `host_end_to_last_kernel_tail=7.635 ms`, `gpu_makespan=0.567 ms`, `gpu_kernel_sum=0.566 ms`
  开始时间(ns): `22503714`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[16397, 512]]}`
- `attn_mha` -> 0.119 ms
  纯GPU kernel时间: `1.327 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.119 ms`, `host_to_first_kernel_gap=8.041612`, `host_end_to_last_kernel_tail=9.250 ms`, `gpu_makespan=1.327 ms`, `gpu_kernel_sum=1.327 ms`
  开始时间(ns): `22737941`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [16397, 128, 192], [16397, 128, 128]]}`
- `o_proj` -> 0.169 ms
  纯GPU kernel时间: `1.257 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.169 ms`, `host_to_first_kernel_gap=9.225361`, `host_end_to_last_kernel_tail=10.315 ms`, `gpu_makespan=1.259 ms`, `gpu_kernel_sum=1.257 ms`
  开始时间(ns): `22883344`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 2 / prefill / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.357 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.357 ms`, `module_to_last_kernel=18.880 ms`, `host_to_first_kernel_gap=14.3695`, `host_end_to_last_kernel_tail=17.522 ms`, `gpu_makespan=4.510 ms`, `gpu_kernel_sum=3.973 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=16397`, `chunked_req_prefix_len=8205`, `current_chunked_req_prefix_len=8205`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[8192, 128, 192], [16397, 128, 192], [16397, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.153 ms
  纯GPU kernel时间: `0.269 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.153 ms`, `host_to_first_kernel_gap=14.254635`, `host_end_to_last_kernel_tail=14.372 ms`, `gpu_makespan=0.271 ms`, `gpu_kernel_sum=0.269 ms`
  开始时间(ns): `23719220`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.037 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.037 ms`, `host_to_first_kernel_gap=14.337346`, `host_end_to_last_kernel_tail=14.318 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `23907549`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.144 ms
  纯GPU kernel时间: `0.439 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.144 ms`, `host_to_first_kernel_gap=14.299844`, `host_end_to_last_kernel_tail=14.596 ms`, `gpu_makespan=0.440 ms`, `gpu_kernel_sum=0.439 ms`
  开始时间(ns): `23964059`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.028 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.028 ms`, `host_to_first_kernel_gap=14.55723`, `host_end_to_last_kernel_tail=14.543 ms`, `gpu_makespan=0.014 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `24146993`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.069 ms
  纯GPU kernel时间: `0.105 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.069 ms`, `host_to_first_kernel_gap=14.517774`, `host_end_to_last_kernel_tail=14.554 ms`, `gpu_makespan=0.105 ms`, `gpu_kernel_sum=0.105 ms`
  开始时间(ns): `24201553`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.157 ms
  纯GPU kernel时间: `0.542 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.157 ms`, `host_to_first_kernel_gap=14.434067`, `host_end_to_last_kernel_tail=14.821 ms`, `gpu_makespan=0.544 ms`, `gpu_kernel_sum=0.542 ms`
  开始时间(ns): `24436844`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[16397, 512]]}`
- `attn_mha` -> 0.117 ms
  纯GPU kernel时间: `1.328 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.117 ms`, `host_to_first_kernel_gap=15.231552`, `host_end_to_last_kernel_tail=16.442 ms`, `gpu_makespan=1.328 ms`, `gpu_kernel_sum=1.328 ms`
  开始时间(ns): `24662591`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [16397, 128, 192], [16397, 128, 128]]}`
- `o_proj` -> 0.146 ms
  纯GPU kernel时间: `1.259 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.146 ms`, `host_to_first_kernel_gap=16.417469`, `host_end_to_last_kernel_tail=17.532 ms`, `gpu_makespan=1.261 ms`, `gpu_kernel_sum=1.259 ms`
  开始时间(ns): `24805954`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 3 / prefill / instance 1

- 执行序号: `4`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `1.352 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.352 ms`, `module_to_last_kernel=26.156 ms`, `host_to_first_kernel_gap=21.601384`, `host_end_to_last_kernel_tail=24.804 ms`, `gpu_makespan=4.554 ms`, `gpu_kernel_sum=4.017 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=16397`, `chunked_req_prefix_len=8205`, `current_chunked_req_prefix_len=8205`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[8192, 128, 192], [16397, 128, 192], [16397, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.153 ms
  纯GPU kernel时间: `0.268 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.153 ms`, `host_to_first_kernel_gap=21.487684`, `host_end_to_last_kernel_tail=21.604 ms`, `gpu_makespan=0.269 ms`, `gpu_kernel_sum=0.268 ms`
  开始时间(ns): `25599065`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.038 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.038 ms`, `host_to_first_kernel_gap=21.5664`, `host_end_to_last_kernel_tail=21.546 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `25789373`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.149 ms
  纯GPU kernel时间: `0.438 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.149 ms`, `host_to_first_kernel_gap=21.530641`, `host_end_to_last_kernel_tail=21.821 ms`, `gpu_makespan=0.440 ms`, `gpu_kernel_sum=0.438 ms`
  开始时间(ns): `25845452`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.028 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.028 ms`, `host_to_first_kernel_gap=21.782826`, `host_end_to_last_kernel_tail=21.768 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `26032851`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.068 ms
  纯GPU kernel时间: `0.106 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.068 ms`, `host_to_first_kernel_gap=21.742727`, `host_end_to_last_kernel_tail=21.780 ms`, `gpu_makespan=0.106 ms`, `gpu_kernel_sum=0.106 ms`
  开始时间(ns): `26087606`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.154 ms
  纯GPU kernel时间: `0.592 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.154 ms`, `host_to_first_kernel_gap=21.661871`, `host_end_to_last_kernel_tail=22.102 ms`, `gpu_makespan=0.594 ms`, `gpu_kernel_sum=0.592 ms`
  开始时间(ns): `26321102`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[16397, 512]]}`
- `attn_mha` -> 0.115 ms
  纯GPU kernel时间: `1.326 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.115 ms`, `host_to_first_kernel_gap=22.513304`, `host_end_to_last_kernel_tail=23.725 ms`, `gpu_makespan=1.326 ms`, `gpu_kernel_sum=1.326 ms`
  开始时间(ns): `26542053`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [16397, 128, 192], [16397, 128, 128]]}`
- `o_proj` -> 0.145 ms
  纯GPU kernel时间: `1.256 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.145 ms`, `host_to_first_kernel_gap=23.701041`, `host_end_to_last_kernel_tail=24.813 ms`, `gpu_makespan=1.258 ms`, `gpu_kernel_sum=1.256 ms`
  开始时间(ns): `26682092`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 4 / prefill / instance 1

- 执行序号: `5`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `1.518 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.518 ms`, `module_to_last_kernel=38.529 ms`, `host_to_first_kernel_gap=34.02193`, `host_end_to_last_kernel_tail=37.011 ms`, `gpu_makespan=4.507 ms`, `gpu_kernel_sum=3.968 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=16397`, `chunked_req_prefix_len=8205`, `current_chunked_req_prefix_len=8205`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[8192, 128, 192], [16397, 128, 192], [16397, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.236 ms
  纯GPU kernel时间: `0.268 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.236 ms`, `host_to_first_kernel_gap=33.878057`, `host_end_to_last_kernel_tail=33.912 ms`, `gpu_makespan=0.269 ms`, `gpu_kernel_sum=0.268 ms`
  开始时间(ns): `28103345`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.042 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.042 ms`, `host_to_first_kernel_gap=33.870307`, `host_end_to_last_kernel_tail=33.846 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `28380727`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.145 ms
  纯GPU kernel时间: `0.439 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.145 ms`, `host_to_first_kernel_gap=33.828237`, `host_end_to_last_kernel_tail=34.123 ms`, `gpu_makespan=0.440 ms`, `gpu_kernel_sum=0.439 ms`
  开始时间(ns): `28442573`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.028 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.028 ms`, `host_to_first_kernel_gap=34.084571`, `host_end_to_last_kernel_tail=34.070 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `28626303`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.073 ms
  纯GPU kernel时间: `0.105 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.073 ms`, `host_to_first_kernel_gap=34.047213`, `host_end_to_last_kernel_tail=34.080 ms`, `gpu_makespan=0.105 ms`, `gpu_kernel_sum=0.105 ms`
  开始时间(ns): `28679085`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.171 ms
  纯GPU kernel时间: `0.541 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.171 ms`, `host_to_first_kernel_gap=33.965925`, `host_end_to_last_kernel_tail=34.337 ms`, `gpu_makespan=0.542 ms`, `gpu_kernel_sum=0.541 ms`
  开始时间(ns): `28912565`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[16397, 512]]}`
- `attn_mha` -> 0.119 ms
  纯GPU kernel时间: `1.327 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.119 ms`, `host_to_first_kernel_gap=34.745758`, `host_end_to_last_kernel_tail=35.954 ms`, `gpu_makespan=1.327 ms`, `gpu_kernel_sum=1.327 ms`
  开始时间(ns): `29154236`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [16397, 128, 192], [16397, 128, 128]]}`
- `o_proj` -> 0.167 ms
  纯GPU kernel时间: `1.257 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.167 ms`, `host_to_first_kernel_gap=35.928239`, `host_end_to_last_kernel_tail=37.020 ms`, `gpu_makespan=1.259 ms`, `gpu_kernel_sum=1.257 ms`
  开始时间(ns): `29301611`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 5 / prefill / instance 1

- 执行序号: `6`
- Collector对齐边界: `{'Module': 'model.model.layers.5.self_attn'}`
- 整块 MLA-module 时长: `1.484 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.484 ms`, `module_to_last_kernel=50.883 ms`, `host_to_first_kernel_gap=46.328428`, `host_end_to_last_kernel_tail=49.398 ms`, `gpu_makespan=4.554 ms`, `gpu_kernel_sum=4.016 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=16397`, `chunked_req_prefix_len=8205`, `current_chunked_req_prefix_len=8205`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[8192, 128, 192], [16397, 128, 192], [16397, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.220 ms
  纯GPU kernel时间: `0.272 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.220 ms`, `host_to_first_kernel_gap=46.204692`, `host_end_to_last_kernel_tail=46.259 ms`, `gpu_makespan=0.275 ms`, `gpu_kernel_sum=0.272 ms`
  开始时间(ns): `30649955`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.042 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.042 ms`, `host_to_first_kernel_gap=46.218327`, `host_end_to_last_kernel_tail=46.194 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `30911360`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.146 ms
  纯GPU kernel时间: `0.451 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.146 ms`, `host_to_first_kernel_gap=46.159524`, `host_end_to_last_kernel_tail=46.465 ms`, `gpu_makespan=0.452 ms`, `gpu_kernel_sum=0.451 ms`
  开始时间(ns): `30989331`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.032 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.032 ms`, `host_to_first_kernel_gap=46.426292`, `host_end_to_last_kernel_tail=46.407 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `31174211`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.069 ms
  纯GPU kernel时间: `0.106 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.069 ms`, `host_to_first_kernel_gap=46.381721`, `host_end_to_last_kernel_tail=46.419 ms`, `gpu_makespan=0.106 ms`, `gpu_kernel_sum=0.106 ms`
  开始时间(ns): `31232958`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.152 ms
  纯GPU kernel时间: `0.555 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.152 ms`, `host_to_first_kernel_gap=46.304095`, `host_end_to_last_kernel_tail=46.708 ms`, `gpu_makespan=0.556 ms`, `gpu_kernel_sum=0.555 ms`
  开始时间(ns): `31463832`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[16397, 512]]}`
- `attn_mha` -> 0.125 ms
  纯GPU kernel时间: `1.328 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.125 ms`, `host_to_first_kernel_gap=47.120029`, `host_end_to_last_kernel_tail=48.323 ms`, `gpu_makespan=1.328 ms`, `gpu_kernel_sum=1.328 ms`
  开始时间(ns): `31683802`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [16397, 128, 192], [16397, 128, 128]]}`
- `o_proj` -> 0.164 ms
  纯GPU kernel时间: `1.274 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.164 ms`, `host_to_first_kernel_gap=48.297462`, `host_end_to_last_kernel_tail=49.409 ms`, `gpu_makespan=1.275 ms`, `gpu_kernel_sum=1.274 ms`
  开始时间(ns): `31836385`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`
