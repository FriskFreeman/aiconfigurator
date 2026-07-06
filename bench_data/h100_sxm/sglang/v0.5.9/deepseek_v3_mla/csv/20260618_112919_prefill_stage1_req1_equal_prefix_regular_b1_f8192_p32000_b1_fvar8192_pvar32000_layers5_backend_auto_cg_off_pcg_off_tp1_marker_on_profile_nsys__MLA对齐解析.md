# MLA对齐解析摘要

源文件: `report.sqlite`

## 对齐原则

- `collector/sglang/collect_mla_module.py` 的 MLA module 计时边界是 `model.model.layers[test_layer].self_attn(...)`。
- 因此这里把 `nsys` 中每层的 `model.model.layers.X.self_attn` NVTX range 视为与 collector 对齐的 MLA-module 边界。
- 该区间内部的 `.self_attn.*` 子模块用于做 MLA 内部 breakdown。

## 运行摘要

- `prefill` 对齐成功层: `[0, 1, 2, 3, 4]`
- `prefill` 被切分层: `[]`
- 若某层 `prefill` 出现多个 `self_attn` 实例，则说明 Engine 调度把一次前向切成了多块，已不再与 collector 的单次 MLA-module 采集严格一一对应。

## Layer 0 / prefill / instance 1

- 执行序号: `1`
- Collector对齐边界: `{'Module': 'model.model.layers.0.self_attn'}`
- 整块 MLA-module 时长: `2.756 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.756 ms`, `module_to_last_kernel=35.565 ms`, `host_to_first_kernel_gap=0.291404`, `host_end_to_last_kernel_tail=32.809 ms`, `gpu_makespan=35.273 ms`, `gpu_kernel_sum=33.228 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=40192`, `chunked_req_prefix_len=32000`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [40192, 128, 192], [40192, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=32000`, `sum_seq_after=40192`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 8192, 'prefix_len': 32000, 'seq_len_after': 40192, 'prompt_len': 40192, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.410 ms
  纯GPU kernel时间: `0.272 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.410 ms`, `host_to_first_kernel_gap=0.141489`, `host_end_to_last_kernel_tail=0.162 ms`, `gpu_makespan=0.431 ms`, `gpu_kernel_sum=0.272 ms`
  开始时间(ns): `33797601`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.056 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.056 ms`, `host_to_first_kernel_gap=0.079881`, `host_end_to_last_kernel_tail=0.042 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `34289866`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.253 ms
  纯GPU kernel时间: `0.451 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.253 ms`, `host_to_first_kernel_gap=0.094297`, `host_end_to_last_kernel_tail=0.409 ms`, `gpu_makespan=0.568 ms`, `gpu_kernel_sum=0.451 ms`
  开始时间(ns): `34382426`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.048 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.048 ms`, `host_to_first_kernel_gap=0.345205`, `host_end_to_last_kernel_tail=0.310 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `34699519`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.103 ms
  纯GPU kernel时间: `0.107 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.103 ms`, `host_to_first_kernel_gap=0.268505`, `host_end_to_last_kernel_tail=0.273 ms`, `gpu_makespan=0.107 ms`, `gpu_kernel_sum=0.107 ms`
  开始时间(ns): `34790427`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.314 ms
  纯GPU kernel时间: `1.566 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.314 ms`, `host_to_first_kernel_gap=0.112129`, `host_end_to_last_kernel_tail=1.508 ms`, `gpu_makespan=1.711 ms`, `gpu_kernel_sum=1.566 ms`
  开始时间(ns): `35425843`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[40192, 512]]}`
- `attn_mha` -> 0.182 ms
  纯GPU kernel时间: `29.534 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.182 ms`, `host_to_first_kernel_gap=2.541216`, `host_end_to_last_kernel_tail=31.894 ms`, `gpu_makespan=29.535 ms`, `gpu_kernel_sum=29.534 ms`
  开始时间(ns): `35867768`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [40192, 128, 192], [40192, 128, 128]]}`
- `o_proj` -> 0.288 ms
  纯GPU kernel时间: `1.266 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.288 ms`, `host_to_first_kernel_gap=31.845112`, `host_end_to_last_kernel_tail=32.824 ms`, `gpu_makespan=1.267 ms`, `gpu_kernel_sum=1.266 ms`
  开始时间(ns): `36099875`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 1 / prefill / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `2.087 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.087 ms`, `module_to_last_kernel=72.647 ms`, `host_to_first_kernel_gap=36.560548`, `host_end_to_last_kernel_tail=70.560 ms`, `gpu_makespan=36.086 ms`, `gpu_kernel_sum=34.841 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=40192`, `chunked_req_prefix_len=32000`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [40192, 128, 192], [40192, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=32000`, `sum_seq_after=40192`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 8192, 'prefix_len': 32000, 'seq_len_after': 40192, 'prompt_len': 40192, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.239 ms
  纯GPU kernel时间: `0.269 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.239 ms`, `host_to_first_kernel_gap=36.417106`, `host_end_to_last_kernel_tail=36.449 ms`, `gpu_makespan=0.271 ms`, `gpu_kernel_sum=0.269 ms`
  开始时间(ns): `37442032`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.055 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.055 ms`, `host_to_first_kernel_gap=36.396451`, `host_end_to_last_kernel_tail=36.360 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `37733823`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.215 ms
  纯GPU kernel时间: `0.450 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.215 ms`, `host_to_first_kernel_gap=36.330378`, `host_end_to_last_kernel_tail=36.568 ms`, `gpu_makespan=0.452 ms`, `gpu_kernel_sum=0.450 ms`
  开始时间(ns): `37819544`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.041 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.041 ms`, `host_to_first_kernel_gap=36.503936`, `host_end_to_last_kernel_tail=36.476 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `38098243`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.106 ms
  纯GPU kernel时间: `0.107 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.106 ms`, `host_to_first_kernel_gap=36.436797`, `host_end_to_last_kernel_tail=36.438 ms`, `gpu_makespan=0.107 ms`, `gpu_kernel_sum=0.107 ms`
  开始时间(ns): `38179814`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.259 ms
  纯GPU kernel时间: `1.350 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.259 ms`, `host_to_first_kernel_gap=36.256798`, `host_end_to_last_kernel_tail=37.350 ms`, `gpu_makespan=1.351 ms`, `gpu_kernel_sum=1.350 ms`
  开始时间(ns): `38538597`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[40192, 512]]}`
- `attn_mha` -> 0.164 ms
  纯GPU kernel时间: `31.247 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.164 ms`, `host_to_first_kernel_gap=38.402429`, `host_end_to_last_kernel_tail=69.486 ms`, `gpu_makespan=31.247 ms`, `gpu_kernel_sum=31.247 ms`
  开始时间(ns): `38907241`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [40192, 128, 192], [40192, 128, 128]]}`
- `o_proj` -> 0.255 ms
  纯GPU kernel时间: `1.386 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.255 ms`, `host_to_first_kernel_gap=69.443276`, `host_end_to_last_kernel_tail=70.576 ms`, `gpu_makespan=1.388 ms`, `gpu_kernel_sum=1.386 ms`
  开始时间(ns): `39114687`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 2 / prefill / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `2.013 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.013 ms`, `module_to_last_kernel=109.980 ms`, `host_to_first_kernel_gap=74.702495`, `host_end_to_last_kernel_tail=107.968 ms`, `gpu_makespan=35.278 ms`, `gpu_kernel_sum=34.034 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=40192`, `chunked_req_prefix_len=32000`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [40192, 128, 192], [40192, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=32000`, `sum_seq_after=40192`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 8192, 'prefix_len': 32000, 'seq_len_after': 40192, 'prompt_len': 40192, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.251 ms
  纯GPU kernel时间: `0.291 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.251 ms`, `host_to_first_kernel_gap=74.555857`, `host_end_to_last_kernel_tail=74.597 ms`, `gpu_makespan=0.292 ms`, `gpu_kernel_sum=0.291 ms`
  开始时间(ns): `40380994`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.052 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.052 ms`, `host_to_first_kernel_gap=74.544686`, `host_end_to_last_kernel_tail=74.511 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `40684709`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.209 ms
  纯GPU kernel时间: `0.479 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.209 ms`, `host_to_first_kernel_gap=74.480052`, `host_end_to_last_kernel_tail=74.751 ms`, `gpu_makespan=0.480 ms`, `gpu_kernel_sum=0.479 ms`
  开始时间(ns): `40768959`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.042 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.042 ms`, `host_to_first_kernel_gap=74.691253`, `host_end_to_last_kernel_tail=74.663 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `41037663`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.099 ms
  纯GPU kernel时间: `0.107 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.099 ms`, `host_to_first_kernel_gap=74.623556`, `host_end_to_last_kernel_tail=74.632 ms`, `gpu_makespan=0.107 ms`, `gpu_kernel_sum=0.107 ms`
  开始时间(ns): `41120944`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.228 ms
  纯GPU kernel时间: `1.586 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.228 ms`, `host_to_first_kernel_gap=74.451623`, `host_end_to_last_kernel_tail=75.811 ms`, `gpu_makespan=1.587 ms`, `gpu_kernel_sum=1.586 ms`
  开始时间(ns): `41471821`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[40192, 512]]}`
- `attn_mha` -> 0.160 ms
  纯GPU kernel时间: `30.078 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.160 ms`, `host_to_first_kernel_gap=76.867387`, `host_end_to_last_kernel_tail=106.786 ms`, `gpu_makespan=30.078 ms`, `gpu_kernel_sum=30.078 ms`
  开始时间(ns): `41804156`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [40192, 128, 192], [40192, 128, 128]]}`
- `o_proj` -> 0.226 ms
  纯GPU kernel时间: `1.461 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.226 ms`, `host_to_first_kernel_gap=106.745437`, `host_end_to_last_kernel_tail=107.982 ms`, `gpu_makespan=1.463 ms`, `gpu_kernel_sum=1.461 ms`
  开始时间(ns): `42006846`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 3 / prefill / instance 1

- 执行序号: `4`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `2.032 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.032 ms`, `module_to_last_kernel=149.124 ms`, `host_to_first_kernel_gap=112.344095`, `host_end_to_last_kernel_tail=147.092 ms`, `gpu_makespan=36.780 ms`, `gpu_kernel_sum=35.529 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=40192`, `chunked_req_prefix_len=32000`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [40192, 128, 192], [40192, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=32000`, `sum_seq_after=40192`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 8192, 'prefix_len': 32000, 'seq_len_after': 40192, 'prompt_len': 40192, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.258 ms
  纯GPU kernel时间: `0.304 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.258 ms`, `host_to_first_kernel_gap=112.216778`, `host_end_to_last_kernel_tail=112.264 ms`, `gpu_makespan=0.306 ms`, `gpu_kernel_sum=0.304 ms`
  开始时间(ns): `43220345`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.051 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.051 ms`, `host_to_first_kernel_gap=112.212339`, `host_end_to_last_kernel_tail=112.181 ms`, `gpu_makespan=0.019 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `43530545`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.204 ms
  纯GPU kernel时间: `0.505 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.204 ms`, `host_to_first_kernel_gap=112.152026`, `host_end_to_last_kernel_tail=112.454 ms`, `gpu_makespan=0.506 ms`, `gpu_kernel_sum=0.505 ms`
  开始时间(ns): `43612074`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.041 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.041 ms`, `host_to_first_kernel_gap=112.395272`, `host_end_to_last_kernel_tail=112.368 ms`, `gpu_makespan=0.014 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `43874940`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.095 ms
  纯GPU kernel时间: `0.108 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.095 ms`, `host_to_first_kernel_gap=112.329712`, `host_end_to_last_kernel_tail=112.343 ms`, `gpu_makespan=0.108 ms`, `gpu_kernel_sum=0.108 ms`
  开始时间(ns): `43956340`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.230 ms
  纯GPU kernel时间: `1.375 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.230 ms`, `host_to_first_kernel_gap=112.17016`, `host_end_to_last_kernel_tail=113.316 ms`, `gpu_makespan=1.376 ms`, `gpu_kernel_sum=1.375 ms`
  开始时间(ns): `44297300`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[40192, 512]]}`
- `attn_mha` -> 0.179 ms
  纯GPU kernel时间: `31.713 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.179 ms`, `host_to_first_kernel_gap=114.372948`, `host_end_to_last_kernel_tail=145.908 ms`, `gpu_makespan=31.713 ms`, `gpu_kernel_sum=31.713 ms`
  开始时间(ns): `44634451`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [40192, 128, 192], [40192, 128, 128]]}`
- `o_proj` -> 0.251 ms
  纯GPU kernel时间: `1.491 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.251 ms`, `host_to_first_kernel_gap=145.865437`, `host_end_to_last_kernel_tail=147.107 ms`, `gpu_makespan=1.493 ms`, `gpu_kernel_sum=1.491 ms`
  开始时间(ns): `44858320`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 4 / prefill / instance 1

- 执行序号: `5`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `2.151 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.151 ms`, `module_to_last_kernel=192.811 ms`, `host_to_first_kernel_gap=156.665738`, `host_end_to_last_kernel_tail=190.660 ms`, `gpu_makespan=36.145 ms`, `gpu_kernel_sum=34.893 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=40192`, `chunked_req_prefix_len=32000`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [40192, 128, 192], [40192, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=32000`, `sum_seq_after=40192`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 8192, 'prefix_len': 32000, 'seq_len_after': 40192, 'prompt_len': 40192, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.314 ms
  纯GPU kernel时间: `0.310 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.314 ms`, `host_to_first_kernel_gap=156.521544`, `host_end_to_last_kernel_tail=156.520 ms`, `gpu_makespan=0.313 ms`, `gpu_kernel_sum=0.310 ms`
  开始时间(ns): `46809492`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.054 ms
  纯GPU kernel时间: `0.020 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.054 ms`, `host_to_first_kernel_gap=156.463791`, `host_end_to_last_kernel_tail=156.429 ms`, `gpu_makespan=0.020 ms`, `gpu_kernel_sum=0.020 ms`
  开始时间(ns): `47180462`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.234 ms
  纯GPU kernel时间: `0.511 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.234 ms`, `host_to_first_kernel_gap=156.377709`, `host_end_to_last_kernel_tail=156.656 ms`, `gpu_makespan=0.512 ms`, `gpu_kernel_sum=0.511 ms`
  开始时间(ns): `47287216`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.051 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.051 ms`, `host_to_first_kernel_gap=156.592988`, `host_end_to_last_kernel_tail=156.555 ms`, `gpu_makespan=0.014 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `47583873`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.094 ms
  纯GPU kernel时间: `0.108 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.094 ms`, `host_to_first_kernel_gap=156.51599`, `host_end_to_last_kernel_tail=156.530 ms`, `gpu_makespan=0.108 ms`, `gpu_kernel_sum=0.108 ms`
  开始时间(ns): `47676839`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.230 ms
  纯GPU kernel时间: `1.441 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.230 ms`, `host_to_first_kernel_gap=156.368564`, `host_end_to_last_kernel_tail=157.582 ms`, `gpu_makespan=1.443 ms`, `gpu_kernel_sum=1.441 ms`
  开始时间(ns): `48007370`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[40192, 512]]}`
- `attn_mha` -> 0.166 ms
  纯GPU kernel时间: `31.000 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.166 ms`, `host_to_first_kernel_gap=158.623849`, `host_end_to_last_kernel_tail=189.458 ms`, `gpu_makespan=31.000 ms`, `gpu_kernel_sum=31.000 ms`
  开始时间(ns): `48360824`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [40192, 128, 192], [40192, 128, 128]]}`
- `o_proj` -> 0.234 ms
  纯GPU kernel时间: `1.489 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.234 ms`, `host_to_first_kernel_gap=189.418521`, `host_end_to_last_kernel_tail=190.675 ms`, `gpu_makespan=1.490 ms`, `gpu_kernel_sum=1.489 ms`
  开始时间(ns): `48567821`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 0 / decode / instance 1

- 执行序号: `6`
- Collector对齐边界: `{'Module': 'model.model.layers.0.self_attn'}`
- 整块 MLA-module 时长: `2.177 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.177 ms`, `module_to_last_kernel=198.101 ms`, `host_to_first_kernel_gap=197.922155`, `host_end_to_last_kernel_tail=195.924 ms`, `gpu_makespan=0.179 ms`, `gpu_kernel_sum=0.148 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.377 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.377 ms`, `host_to_first_kernel_gap=197.835182`, `host_end_to_last_kernel_tail=197.475 ms`, `gpu_makespan=0.016 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `53386151`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.050 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.050 ms`, `host_to_first_kernel_gap=197.405546`, `host_end_to_last_kernel_tail=197.358 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `53831659`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.036 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.036 ms`, `host_to_first_kernel_gap=197.326463`, `host_end_to_last_kernel_tail=197.293 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `53912822`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.245 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.245 ms`, `host_to_first_kernel_gap=197.253072`, `host_end_to_last_kernel_tail=197.029 ms`, `gpu_makespan=0.021 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `53989381`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.102 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.102 ms`, `host_to_first_kernel_gap=196.823641`, `host_end_to_last_kernel_tail=196.724 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `54451836`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.353 ms
  纯GPU kernel时间: `0.055 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.353 ms`, `host_to_first_kernel_gap=196.672411`, `host_end_to_last_kernel_tail=196.376 ms`, `gpu_makespan=0.056 ms`, `gpu_kernel_sum=0.055 ms`
  开始时间(ns): `54606106`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.348 ms
  纯GPU kernel时间: `0.052 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.348 ms`, `host_to_first_kernel_gap=196.238057`, `host_end_to_last_kernel_tail=195.943 ms`, `gpu_makespan=0.053 ms`, `gpu_kernel_sum=0.052 ms`
  开始时间(ns): `55109069`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 1 / decode / instance 1

- 执行序号: `7`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `1.839 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.839 ms`, `module_to_last_kernel=195.355 ms`, `host_to_first_kernel_gap=195.175213`, `host_end_to_last_kernel_tail=193.516 ms`, `gpu_makespan=0.179 ms`, `gpu_kernel_sum=0.148 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.219 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.219 ms`, `host_to_first_kernel_gap=195.104127`, `host_end_to_last_kernel_tail=194.901 ms`, `gpu_makespan=0.016 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `56459511`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.050 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.050 ms`, `host_to_first_kernel_gap=194.841178`, `host_end_to_last_kernel_tail=194.793 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `56738652`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.034 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.034 ms`, `host_to_first_kernel_gap=194.768691`, `host_end_to_last_kernel_tail=194.737 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `56813379`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.204 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.204 ms`, `host_to_first_kernel_gap=194.702213`, `host_end_to_last_kernel_tail=194.518 ms`, `gpu_makespan=0.020 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `56882961`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.096 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.096 ms`, `host_to_first_kernel_gap=194.342658`, `host_end_to_last_kernel_tail=194.248 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `57275124`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.324 ms
  纯GPU kernel时间: `0.055 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.324 ms`, `host_to_first_kernel_gap=194.197632`, `host_end_to_last_kernel_tail=193.930 ms`, `gpu_makespan=0.056 ms`, `gpu_kernel_sum=0.055 ms`
  开始时间(ns): `57423126`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.316 ms
  纯GPU kernel时间: `0.052 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.316 ms`, `host_to_first_kernel_gap=193.79679`, `host_end_to_last_kernel_tail=193.535 ms`, `gpu_makespan=0.054 ms`, `gpu_kernel_sum=0.052 ms`
  开始时间(ns): `57892640`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 2 / decode / instance 1

- 执行序号: `8`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.819 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.819 ms`, `module_to_last_kernel=192.984 ms`, `host_to_first_kernel_gap=192.805081`, `host_end_to_last_kernel_tail=191.165 ms`, `gpu_makespan=0.179 ms`, `gpu_kernel_sum=0.147 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.228 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.228 ms`, `host_to_first_kernel_gap=192.734726`, `host_end_to_last_kernel_tail=192.524 ms`, `gpu_makespan=0.017 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `59174000`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.050 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.050 ms`, `host_to_first_kernel_gap=192.463039`, `host_end_to_last_kernel_tail=192.416 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `59462679`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.032 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.032 ms`, `host_to_first_kernel_gap=192.39056`, `host_end_to_last_kernel_tail=192.361 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `59537558`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.203 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.203 ms`, `host_to_first_kernel_gap=192.323687`, `host_end_to_last_kernel_tail=192.140 ms`, `gpu_makespan=0.020 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `59607567`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.095 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.095 ms`, `host_to_first_kernel_gap=191.990449`, `host_end_to_last_kernel_tail=191.897 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `59972357`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.346 ms
  纯GPU kernel时间: `0.055 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.346 ms`, `host_to_first_kernel_gap=191.849693`, `host_end_to_last_kernel_tail=191.561 ms`, `gpu_makespan=0.057 ms`, `gpu_kernel_sum=0.055 ms`
  开始时间(ns): `60116153`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.300 ms
  纯GPU kernel时间: `0.051 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.300 ms`, `host_to_first_kernel_gap=191.430856`, `host_end_to_last_kernel_tail=191.184 ms`, `gpu_makespan=0.052 ms`, `gpu_kernel_sum=0.051 ms`
  开始时间(ns): `60604430`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 3 / decode / instance 1

- 执行序号: `9`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `1.750 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.750 ms`, `module_to_last_kernel=190.636 ms`, `host_to_first_kernel_gap=190.457762`, `host_end_to_last_kernel_tail=188.886 ms`, `gpu_makespan=0.178 ms`, `gpu_kernel_sum=0.148 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.213 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.213 ms`, `host_to_first_kernel_gap=190.388271`, `host_end_to_last_kernel_tail=190.191 ms`, `gpu_makespan=0.016 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `61865640`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.050 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.050 ms`, `host_to_first_kernel_gap=190.132294`, `host_end_to_last_kernel_tail=190.085 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `62137713`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.031 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.031 ms`, `host_to_first_kernel_gap=190.06059`, `host_end_to_last_kernel_tail=190.031 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `62211753`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.207 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.207 ms`, `host_to_first_kernel_gap=189.989248`, `host_end_to_last_kernel_tail=189.802 ms`, `gpu_makespan=0.020 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `62287031`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.091 ms
  纯GPU kernel时间: `0.003 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.091 ms`, `host_to_first_kernel_gap=189.648367`, `host_end_to_last_kernel_tail=189.560 ms`, `gpu_makespan=0.003 ms`, `gpu_kernel_sum=0.003 ms`
  开始时间(ns): `62658760`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.318 ms
  纯GPU kernel时间: `0.055 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.318 ms`, `host_to_first_kernel_gap=189.511144`, `host_end_to_last_kernel_tail=189.250 ms`, `gpu_makespan=0.056 ms`, `gpu_kernel_sum=0.055 ms`
  开始时间(ns): `62799663`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.279 ms
  纯GPU kernel时间: `0.051 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.279 ms`, `host_to_first_kernel_gap=189.13002`, `host_end_to_last_kernel_tail=188.903 ms`, `gpu_makespan=0.052 ms`, `gpu_kernel_sum=0.051 ms`
  开始时间(ns): `63249235`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 4 / decode / instance 1

- 执行序号: `10`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `2.030 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.030 ms`, `module_to_last_kernel=187.657 ms`, `host_to_first_kernel_gap=187.478365`, `host_end_to_last_kernel_tail=185.627 ms`, `gpu_makespan=0.178 ms`, `gpu_kernel_sum=0.148 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.329 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.329 ms`, `host_to_first_kernel_gap=187.398487`, `host_end_to_last_kernel_tail=187.086 ms`, `gpu_makespan=0.016 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `65247264`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.056 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.056 ms`, `host_to_first_kernel_gap=187.021623`, `host_end_to_last_kernel_tail=186.968 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `65640448`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.032 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.032 ms`, `host_to_first_kernel_gap=186.943684`, `host_end_to_last_kernel_tail=186.914 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `65720723`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.233 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.233 ms`, `host_to_first_kernel_gap=186.87833`, `host_end_to_last_kernel_tail=186.665 ms`, `gpu_makespan=0.020 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `65789437`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.116 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.116 ms`, `host_to_first_kernel_gap=186.487033`, `host_end_to_last_kernel_tail=186.373 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `66211582`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.338 ms
  纯GPU kernel时间: `0.055 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.338 ms`, `host_to_first_kernel_gap=186.321377`, `host_end_to_last_kernel_tail=186.040 ms`, `gpu_makespan=0.056 ms`, `gpu_kernel_sum=0.055 ms`
  开始时间(ns): `66381078`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.305 ms
  纯GPU kernel时间: `0.052 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.305 ms`, `host_to_first_kernel_gap=185.89767`, `host_end_to_last_kernel_tail=185.646 ms`, `gpu_makespan=0.053 ms`, `gpu_kernel_sum=0.052 ms`
  开始时间(ns): `66873425`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`
