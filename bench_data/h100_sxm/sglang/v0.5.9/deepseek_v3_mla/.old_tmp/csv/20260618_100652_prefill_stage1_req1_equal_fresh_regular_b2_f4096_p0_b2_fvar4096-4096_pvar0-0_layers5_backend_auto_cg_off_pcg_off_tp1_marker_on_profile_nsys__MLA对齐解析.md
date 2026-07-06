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
- 整块 MLA-module 时长: `2.169 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.169 ms`, `module_to_last_kernel=5.281 ms`, `host_to_first_kernel_gap=0.28599`, `host_end_to_last_kernel_tail=3.112 ms`, `gpu_makespan=4.995 ms`, `gpu_kernel_sum=4.352 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=8192`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=0`, `sum_seq_after=8192`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 4096, 'prefix_len': 0, 'seq_len_after': 4096, 'prompt_len': 4096, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 4096, 'prefix_len': 0, 'seq_len_after': 4096, 'prompt_len': 4096, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.384 ms
  纯GPU kernel时间: `0.272 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.384 ms`, `host_to_first_kernel_gap=0.124047`, `host_end_to_last_kernel_tail=0.168 ms`, `gpu_makespan=0.428 ms`, `gpu_kernel_sum=0.272 ms`
  开始时间(ns): `28392892`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.047 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.047 ms`, `host_to_first_kernel_gap=0.101447`, `host_end_to_last_kernel_tail=0.072 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `28843236`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.237 ms
  纯GPU kernel时间: `0.454 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.237 ms`, `host_to_first_kernel_gap=0.087689`, `host_end_to_last_kernel_tail=0.412 ms`, `gpu_makespan=0.561 ms`, `gpu_kernel_sum=0.454 ms`
  开始时间(ns): `28924162`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.041 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.041 ms`, `host_to_first_kernel_gap=0.347569`, `host_end_to_last_kernel_tail=0.319 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `29225498`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.103 ms
  纯GPU kernel时间: `0.107 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.103 ms`, `host_to_first_kernel_gap=0.276787`, `host_end_to_last_kernel_tail=0.281 ms`, `gpu_makespan=0.107 ms`, `gpu_kernel_sum=0.107 ms`
  开始时间(ns): `29310040`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.250 ms
  纯GPU kernel时间: `0.289 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.250 ms`, `host_to_first_kernel_gap=0.159791`, `host_end_to_last_kernel_tail=0.255 ms`, `gpu_makespan=0.345 ms`, `gpu_kernel_sum=0.289 ms`
  开始时间(ns): `29563164`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[8192, 512]]}`
- `attn_mha` -> 0.180 ms
  纯GPU kernel时间: `1.950 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.180 ms`, `host_to_first_kernel_gap=0.390078`, `host_end_to_last_kernel_tail=2.160 ms`, `gpu_makespan=1.950 ms`, `gpu_kernel_sum=1.950 ms`
  开始时间(ns): `29919661`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]}`
- `o_proj` -> 0.242 ms
  纯GPU kernel时间: `1.250 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.242 ms`, `host_to_first_kernel_gap=2.116941`, `host_end_to_last_kernel_tail=3.126 ms`, `gpu_makespan=1.251 ms`, `gpu_kernel_sum=1.250 ms`
  开始时间(ns): `30144318`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 1 / prefill / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `1.764 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.764 ms`, `module_to_last_kernel=11.485 ms`, `host_to_first_kernel_gap=6.859751`, `host_end_to_last_kernel_tail=9.720 ms`, `gpu_makespan=4.625 ms`, `gpu_kernel_sum=4.340 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=8192`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=0`, `sum_seq_after=8192`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 4096, 'prefix_len': 0, 'seq_len_after': 4096, 'prompt_len': 4096, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 4096, 'prefix_len': 0, 'seq_len_after': 4096, 'prompt_len': 4096, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.232 ms
  纯GPU kernel时间: `0.267 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.232 ms`, `host_to_first_kernel_gap=6.734378`, `host_end_to_last_kernel_tail=6.772 ms`, `gpu_makespan=0.269 ms`, `gpu_kernel_sum=0.267 ms`
  开始时间(ns): `31388448`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.045 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.045 ms`, `host_to_first_kernel_gap=6.72353`, `host_end_to_last_kernel_tail=6.696 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `31668736`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.197 ms
  纯GPU kernel时间: `0.442 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.197 ms`, `host_to_first_kernel_gap=6.668482`, `host_end_to_last_kernel_tail=6.916 ms`, `gpu_makespan=0.444 ms`, `gpu_kernel_sum=0.442 ms`
  开始时间(ns): `31742536`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.039 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.039 ms`, `host_to_first_kernel_gap=6.857839`, `host_end_to_last_kernel_tail=6.832 ms`, `gpu_makespan=0.014 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `31997531`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.077 ms
  纯GPU kernel时间: `0.107 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.077 ms`, `host_to_first_kernel_gap=6.795504`, `host_end_to_last_kernel_tail=6.826 ms`, `gpu_makespan=0.107 ms`, `gpu_kernel_sum=0.107 ms`
  开始时间(ns): `32074778`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.209 ms
  纯GPU kernel时间: `0.292 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.209 ms`, `host_to_first_kernel_gap=6.723116`, `host_end_to_last_kernel_tail=6.808 ms`, `gpu_makespan=0.294 ms`, `gpu_kernel_sum=0.292 ms`
  开始时间(ns): `32283454`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[8192, 512]]}`
- `attn_mha` -> 0.145 ms
  纯GPU kernel时间: `1.951 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.145 ms`, `host_to_first_kernel_gap=6.953342`, `host_end_to_last_kernel_tail=8.759 ms`, `gpu_makespan=1.951 ms`, `gpu_kernel_sum=1.951 ms`
  开始时间(ns): `32590220`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]}`
- `o_proj` -> 0.237 ms
  纯GPU kernel时间: `1.250 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.237 ms`, `host_to_first_kernel_gap=8.719736`, `host_end_to_last_kernel_tail=9.734 ms`, `gpu_makespan=1.251 ms`, `gpu_kernel_sum=1.250 ms`
  开始时间(ns): `32776978`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 2 / prefill / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.685 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.685 ms`, `module_to_last_kernel=18.150 ms`, `host_to_first_kernel_gap=13.531029`, `host_end_to_last_kernel_tail=16.465 ms`, `gpu_makespan=4.619 ms`, `gpu_kernel_sum=4.335 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=8192`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=0`, `sum_seq_after=8192`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 4096, 'prefix_len': 0, 'seq_len_after': 4096, 'prompt_len': 4096, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 4096, 'prefix_len': 0, 'seq_len_after': 4096, 'prompt_len': 4096, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.215 ms
  纯GPU kernel时间: `0.268 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.215 ms`, `host_to_first_kernel_gap=13.416103`, `host_end_to_last_kernel_tail=13.470 ms`, `gpu_makespan=0.269 ms`, `gpu_kernel_sum=0.268 ms`
  开始时间(ns): `33955202`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.044 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.044 ms`, `host_to_first_kernel_gap=13.421568`, `host_end_to_last_kernel_tail=13.395 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `34219177`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.202 ms
  纯GPU kernel时间: `0.443 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.202 ms`, `host_to_first_kernel_gap=13.36015`, `host_end_to_last_kernel_tail=13.603 ms`, `gpu_makespan=0.444 ms`, `gpu_kernel_sum=0.443 ms`
  开始时间(ns): `34300339`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.046 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.046 ms`, `host_to_first_kernel_gap=13.54238`, `host_end_to_last_kernel_tail=13.509 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `34562397`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.073 ms
  纯GPU kernel时间: `0.108 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.073 ms`, `host_to_first_kernel_gap=13.473658`, `host_end_to_last_kernel_tail=13.509 ms`, `gpu_makespan=0.108 ms`, `gpu_kernel_sum=0.108 ms`
  开始时间(ns): `34646287`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.209 ms
  纯GPU kernel时间: `0.284 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.209 ms`, `host_to_first_kernel_gap=13.40824`, `host_end_to_last_kernel_tail=13.484 ms`, `gpu_makespan=0.285 ms`, `gpu_kernel_sum=0.284 ms`
  开始时间(ns): `34847993`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[8192, 512]]}`
- `attn_mha` -> 0.130 ms
  纯GPU kernel时间: `1.954 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.130 ms`, `host_to_first_kernel_gap=13.639545`, `host_end_to_last_kernel_tail=15.462 ms`, `gpu_makespan=1.953 ms`, `gpu_kernel_sum=1.954 ms`
  开始时间(ns): `35145968`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]}`
- `o_proj` -> 0.201 ms
  纯GPU kernel时间: `1.248 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.201 ms`, `host_to_first_kernel_gap=15.429589`, `host_end_to_last_kernel_tail=16.478 ms`, `gpu_makespan=1.249 ms`, `gpu_kernel_sum=1.248 ms`
  开始时间(ns): `35311220`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 3 / prefill / instance 1

- 执行序号: `4`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `1.751 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.751 ms`, `module_to_last_kernel=24.907 ms`, `host_to_first_kernel_gap=20.281086`, `host_end_to_last_kernel_tail=23.156 ms`, `gpu_makespan=4.626 ms`, `gpu_kernel_sum=4.344 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=8192`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=0`, `sum_seq_after=8192`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 4096, 'prefix_len': 0, 'seq_len_after': 4096, 'prompt_len': 4096, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 4096, 'prefix_len': 0, 'seq_len_after': 4096, 'prompt_len': 4096, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.220 ms
  纯GPU kernel时间: `0.267 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.220 ms`, `host_to_first_kernel_gap=20.166586`, `host_end_to_last_kernel_tail=20.215 ms`, `gpu_makespan=0.268 ms`, `gpu_kernel_sum=0.267 ms`
  开始时间(ns): `36441583`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.044 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.044 ms`, `host_to_first_kernel_gap=20.166536`, `host_end_to_last_kernel_tail=20.140 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `36710241`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.198 ms
  纯GPU kernel时间: `0.435 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.198 ms`, `host_to_first_kernel_gap=20.110903`, `host_end_to_last_kernel_tail=20.349 ms`, `gpu_makespan=0.437 ms`, `gpu_kernel_sum=0.435 ms`
  开始时间(ns): `36784786`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.037 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.037 ms`, `host_to_first_kernel_gap=20.291176`, `host_end_to_last_kernel_tail=20.267 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `37041313`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.077 ms
  纯GPU kernel时间: `0.108 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.077 ms`, `host_to_first_kernel_gap=20.22968`, `host_end_to_last_kernel_tail=20.260 ms`, `gpu_makespan=0.108 ms`, `gpu_kernel_sum=0.108 ms`
  开始时间(ns): `37117625`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.232 ms
  纯GPU kernel时间: `0.288 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.232 ms`, `host_to_first_kernel_gap=20.156118`, `host_end_to_last_kernel_tail=20.213 ms`, `gpu_makespan=0.289 ms`, `gpu_kernel_sum=0.288 ms`
  开始时间(ns): `37327347`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[8192, 512]]}`
- `attn_mha` -> 0.137 ms
  纯GPU kernel时间: `1.958 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.137 ms`, `host_to_first_kernel_gap=20.349225`, `host_end_to_last_kernel_tail=22.170 ms`, `gpu_makespan=1.958 ms`, `gpu_kernel_sum=1.958 ms`
  开始时间(ns): `37665344`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]}`
- `o_proj` -> 0.222 ms
  纯GPU kernel时间: `1.258 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.222 ms`, `host_to_first_kernel_gap=22.132897`, `host_end_to_last_kernel_tail=23.170 ms`, `gpu_makespan=1.259 ms`, `gpu_kernel_sum=1.258 ms`
  开始时间(ns): `37842119`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 4 / prefill / instance 1

- 执行序号: `5`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `1.828 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.828 ms`, `module_to_last_kernel=36.846 ms`, `host_to_first_kernel_gap=32.200806`, `host_end_to_last_kernel_tail=35.018 ms`, `gpu_makespan=4.645 ms`, `gpu_kernel_sum=4.362 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=8192`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=0`, `sum_seq_after=8192`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 4096, 'prefix_len': 0, 'seq_len_after': 4096, 'prompt_len': 4096, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 4096, 'prefix_len': 0, 'seq_len_after': 4096, 'prompt_len': 4096, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.278 ms
  纯GPU kernel时间: `0.268 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.278 ms`, `host_to_first_kernel_gap=32.070879`, `host_end_to_last_kernel_tail=32.063 ms`, `gpu_makespan=0.270 ms`, `gpu_kernel_sum=0.268 ms`
  开始时间(ns): `39647816`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.048 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.048 ms`, `host_to_first_kernel_gap=32.010924`, `host_end_to_last_kernel_tail=31.981 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `39978651`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.218 ms
  纯GPU kernel时间: `0.445 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.218 ms`, `host_to_first_kernel_gap=31.950966`, `host_end_to_last_kernel_tail=32.180 ms`, `gpu_makespan=0.447 ms`, `gpu_kernel_sum=0.445 ms`
  开始时间(ns): `40057393`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.043 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.043 ms`, `host_to_first_kernel_gap=32.120353`, `host_end_to_last_kernel_tail=32.089 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `40334854`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.076 ms
  纯GPU kernel时间: `0.108 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.076 ms`, `host_to_first_kernel_gap=32.050874`, `host_end_to_last_kernel_tail=32.082 ms`, `gpu_makespan=0.108 ms`, `gpu_kernel_sum=0.108 ms`
  开始时间(ns): `40418157`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.221 ms
  纯GPU kernel时间: `0.296 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.221 ms`, `host_to_first_kernel_gap=31.978204`, `host_end_to_last_kernel_tail=32.055 ms`, `gpu_makespan=0.298 ms`, `gpu_kernel_sum=0.296 ms`
  开始时间(ns): `40627723`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[8192, 512]]}`
- `attn_mha` -> 0.137 ms
  纯GPU kernel时间: `1.956 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.137 ms`, `host_to_first_kernel_gap=32.205116`, `host_end_to_last_kernel_tail=34.024 ms`, `gpu_makespan=1.956 ms`, `gpu_kernel_sum=1.956 ms`
  开始时间(ns): `40940395`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]}`
- `o_proj` -> 0.216 ms
  纯GPU kernel时间: `1.259 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.216 ms`, `host_to_first_kernel_gap=33.986723`, `host_end_to_last_kernel_tail=35.031 ms`, `gpu_makespan=1.260 ms`, `gpu_kernel_sum=1.259 ms`
  开始时间(ns): `41116388`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 0 / decode / instance 1

- 执行序号: `6`
- Collector对齐边界: `{'Module': 'model.model.layers.0.self_attn'}`
- 整块 MLA-module 时长: `1.789 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.789 ms`, `module_to_last_kernel=42.694 ms`, `host_to_first_kernel_gap=42.544612`, `host_end_to_last_kernel_tail=40.905 ms`, `gpu_makespan=0.149 ms`, `gpu_kernel_sum=0.117 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.298 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.298 ms`, `host_to_first_kernel_gap=42.471933`, `host_end_to_last_kernel_tail=42.190 ms`, `gpu_makespan=0.016 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `45158409`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2, 7168]]}`
- `q_a_layernorm` -> 0.047 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.047 ms`, `host_to_first_kernel_gap=42.127705`, `host_end_to_last_kernel_tail=42.083 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `45518605`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2, 1536]]}`
- `kv_a_layernorm` -> 0.032 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.032 ms`, `host_to_first_kernel_gap=42.050438`, `host_end_to_last_kernel_tail=42.020 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `45597856`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2, 512]]}`
- `q_b_proj` -> 0.226 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.226 ms`, `host_to_first_kernel_gap=41.983685`, `host_end_to_last_kernel_tail=41.778 ms`, `gpu_makespan=0.021 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `45667553`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2, 1536]]}`
- `rotary_emb` -> 0.078 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.078 ms`, `host_to_first_kernel_gap=41.629162`, `host_end_to_last_kernel_tail=41.553 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `46054972`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2], [2, 128, 64], [2, 1, 64]]}`
- `attn_mqa` -> 0.289 ms
  纯GPU kernel时间: `0.029 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.289 ms`, `host_to_first_kernel_gap=41.509328`, `host_end_to_last_kernel_tail=41.250 ms`, `gpu_makespan=0.030 ms`, `gpu_kernel_sum=0.029 ms`
  开始时间(ns): `46177654`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2, 128, 512], [2, 1, 512], [2, 1, 512]]}`
- `o_proj` -> 0.259 ms
  纯GPU kernel时间: `0.050 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.259 ms`, `host_to_first_kernel_gap=41.131343`, `host_end_to_last_kernel_tail=40.923 ms`, `gpu_makespan=0.051 ms`, `gpu_kernel_sum=0.050 ms`
  开始时间(ns): `46597047`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2, 16384]]}`

## Layer 1 / decode / instance 1

- 执行序号: `7`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `1.557 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.557 ms`, `module_to_last_kernel=40.371 ms`, `host_to_first_kernel_gap=40.22409`, `host_end_to_last_kernel_tail=38.814 ms`, `gpu_makespan=0.147 ms`, `gpu_kernel_sum=0.117 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.214 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.214 ms`, `host_to_first_kernel_gap=40.163678`, `host_end_to_last_kernel_tail=39.966 ms`, `gpu_makespan=0.016 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `47776456`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2, 7168]]}`
- `q_a_layernorm` -> 0.043 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.043 ms`, `host_to_first_kernel_gap=39.908534`, `host_end_to_last_kernel_tail=39.868 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `48047536`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2, 1536]]}`
- `kv_a_layernorm` -> 0.029 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.029 ms`, `host_to_first_kernel_gap=39.843755`, `host_end_to_last_kernel_tail=39.817 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `48114267`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2, 512]]}`
- `q_b_proj` -> 0.201 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.201 ms`, `host_to_first_kernel_gap=39.780463`, `host_end_to_last_kernel_tail=39.599 ms`, `gpu_makespan=0.019 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `48180503`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2, 1536]]}`
- `rotary_emb` -> 0.081 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.081 ms`, `host_to_first_kernel_gap=39.471378`, `host_end_to_last_kernel_tail=39.392 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `48520596`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2], [2, 128, 64], [2, 1, 64]]}`
- `attn_mqa` -> 0.265 ms
  纯GPU kernel时间: `0.029 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.265 ms`, `host_to_first_kernel_gap=39.35051`, `host_end_to_last_kernel_tail=39.116 ms`, `gpu_makespan=0.031 ms`, `gpu_kernel_sum=0.029 ms`
  开始时间(ns): `48644248`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2, 128, 512], [2, 1, 512], [2, 1, 512]]}`
- `o_proj` -> 0.232 ms
  纯GPU kernel时间: `0.049 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.232 ms`, `host_to_first_kernel_gap=39.012727`, `host_end_to_last_kernel_tail=38.830 ms`, `gpu_makespan=0.050 ms`, `gpu_kernel_sum=0.049 ms`
  开始时间(ns): `49024655`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2, 16384]]}`

## Layer 2 / decode / instance 1

- 执行序号: `8`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.558 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.558 ms`, `module_to_last_kernel=38.296 ms`, `host_to_first_kernel_gap=38.146954`, `host_end_to_last_kernel_tail=36.738 ms`, `gpu_makespan=0.149 ms`, `gpu_kernel_sum=0.118 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.205 ms
  纯GPU kernel时间: `0.017 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.205 ms`, `host_to_first_kernel_gap=38.086485`, `host_end_to_last_kernel_tail=37.900 ms`, `gpu_makespan=0.019 ms`, `gpu_kernel_sum=0.017 ms`
  开始时间(ns): `50159409`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2, 7168]]}`
- `q_a_layernorm` -> 0.043 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.043 ms`, `host_to_first_kernel_gap=37.842888`, `host_end_to_last_kernel_tail=37.802 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `50421566`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2, 1536]]}`
- `kv_a_layernorm` -> 0.030 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.030 ms`, `host_to_first_kernel_gap=37.778567`, `host_end_to_last_kernel_tail=37.751 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `50488063`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2, 512]]}`
- `q_b_proj` -> 0.207 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.207 ms`, `host_to_first_kernel_gap=37.718634`, `host_end_to_last_kernel_tail=37.531 ms`, `gpu_makespan=0.019 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `50551164`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2, 1536]]}`
- `rotary_emb` -> 0.073 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.073 ms`, `host_to_first_kernel_gap=37.405827`, `host_end_to_last_kernel_tail=37.334 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `50894787`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2], [2, 128, 64], [2, 1, 64]]}`
- `attn_mqa` -> 0.269 ms
  纯GPU kernel时间: `0.028 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.269 ms`, `host_to_first_kernel_gap=37.291722`, `host_end_to_last_kernel_tail=37.053 ms`, `gpu_makespan=0.030 ms`, `gpu_kernel_sum=0.028 ms`
  开始时间(ns): `51011932`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2, 128, 512], [2, 1, 512], [2, 1, 512]]}`
- `o_proj` -> 0.238 ms
  纯GPU kernel时间: `0.049 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.238 ms`, `host_to_first_kernel_gap=36.944517`, `host_end_to_last_kernel_tail=36.757 ms`, `gpu_makespan=0.050 ms`, `gpu_kernel_sum=0.049 ms`
  开始时间(ns): `51400609`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2, 16384]]}`

## Layer 3 / decode / instance 1

- 执行序号: `9`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `1.543 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.543 ms`, `module_to_last_kernel=36.264 ms`, `host_to_first_kernel_gap=36.11292`, `host_end_to_last_kernel_tail=34.721 ms`, `gpu_makespan=0.151 ms`, `gpu_kernel_sum=0.120 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.213 ms
  纯GPU kernel时间: `0.016 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.213 ms`, `host_to_first_kernel_gap=36.051536`, `host_end_to_last_kernel_tail=35.856 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.016 ms`
  开始时间(ns): `52506614`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2, 7168]]}`
- `q_a_layernorm` -> 0.043 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.043 ms`, `host_to_first_kernel_gap=35.79854`, `host_end_to_last_kernel_tail=35.758 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `52777402`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2, 1536]]}`
- `kv_a_layernorm` -> 0.030 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.030 ms`, `host_to_first_kernel_gap=35.734332`, `host_end_to_last_kernel_tail=35.707 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `52843594`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2, 512]]}`
- `q_b_proj` -> 0.195 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.195 ms`, `host_to_first_kernel_gap=35.675771`, `host_end_to_last_kernel_tail=35.501 ms`, `gpu_makespan=0.020 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `52906155`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2, 1536]]}`
- `rotary_emb` -> 0.074 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.074 ms`, `host_to_first_kernel_gap=35.373307`, `host_end_to_last_kernel_tail=35.301 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `53240075`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2], [2, 128, 64], [2, 1, 64]]}`
- `attn_mqa` -> 0.276 ms
  纯GPU kernel时间: `0.029 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.276 ms`, `host_to_first_kernel_gap=35.259771`, `host_end_to_last_kernel_tail=35.013 ms`, `gpu_makespan=0.030 ms`, `gpu_kernel_sum=0.029 ms`
  开始时间(ns): `53356683`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2, 128, 512], [2, 1, 512], [2, 1, 512]]}`
- `o_proj` -> 0.227 ms
  纯GPU kernel时间: `0.050 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.227 ms`, `host_to_first_kernel_gap=34.912756`, `host_end_to_last_kernel_tail=34.737 ms`, `gpu_makespan=0.051 ms`, `gpu_kernel_sum=0.050 ms`
  开始时间(ns): `53745266`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2, 16384]]}`

## Layer 4 / decode / instance 1

- 执行序号: `10`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `1.695 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.695 ms`, `module_to_last_kernel=33.710 ms`, `host_to_first_kernel_gap=33.559184`, `host_end_to_last_kernel_tail=32.015 ms`, `gpu_makespan=0.150 ms`, `gpu_kernel_sum=0.119 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.270 ms
  纯GPU kernel时间: `0.017 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.270 ms`, `host_to_first_kernel_gap=33.493198`, `host_end_to_last_kernel_tail=33.241 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.017 ms`
  开始时间(ns): `55422616`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2, 7168]]}`
- `q_a_layernorm` -> 0.045 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.045 ms`, `host_to_first_kernel_gap=33.180628`, `host_end_to_last_kernel_tail=33.137 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `55753106`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2, 1536]]}`
- `kv_a_layernorm` -> 0.029 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.029 ms`, `host_to_first_kernel_gap=33.113019`, `host_end_to_last_kernel_tail=33.086 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `55822699`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2, 512]]}`
- `q_b_proj` -> 0.214 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.214 ms`, `host_to_first_kernel_gap=33.053923`, `host_end_to_last_kernel_tail=32.860 ms`, `gpu_makespan=0.019 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `55885635`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2, 1536]]}`
- `rotary_emb` -> 0.081 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.081 ms`, `host_to_first_kernel_gap=32.722153`, `host_end_to_last_kernel_tail=32.643 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `56247517`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2], [2, 128, 64], [2, 1, 64]]}`
- `attn_mqa` -> 0.280 ms
  纯GPU kernel时间: `0.029 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.280 ms`, `host_to_first_kernel_gap=32.600931`, `host_end_to_last_kernel_tail=32.351 ms`, `gpu_makespan=0.030 ms`, `gpu_kernel_sum=0.029 ms`
  开始时间(ns): `56372643`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2, 128, 512], [2, 1, 512], [2, 1, 512]]}`
- `o_proj` -> 0.258 ms
  纯GPU kernel时间: `0.049 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.258 ms`, `host_to_first_kernel_gap=32.240779`, `host_end_to_last_kernel_tail=32.033 ms`, `gpu_makespan=0.050 ms`, `gpu_kernel_sum=0.049 ms`
  开始时间(ns): `56775419`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2, 16384]]}`
