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
- 整块 MLA-module 时长: `2.593 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.593 ms`, `module_to_last_kernel=5.949 ms`, `host_to_first_kernel_gap=0.273128`, `host_end_to_last_kernel_tail=3.356 ms`, `gpu_makespan=5.676 ms`, `gpu_kernel_sum=4.496 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=12288`, `chunked_req_prefix_len=4096`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [12288, 128, 192], [12288, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=4096`, `sum_seq_after=12288`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 2048, 'prefix_len': 1024, 'seq_len_after': 3072, 'prompt_len': 3072, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 2048, 'prefix_len': 1024, 'seq_len_after': 3072, 'prompt_len': 3072, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 2048, 'prefix_len': 1024, 'seq_len_after': 3072, 'prompt_len': 3072, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 2048, 'prefix_len': 1024, 'seq_len_after': 3072, 'prompt_len': 3072, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.369 ms
  纯GPU kernel时间: `0.274 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.369 ms`, `host_to_first_kernel_gap=0.126638`, `host_end_to_last_kernel_tail=0.167 ms`, `gpu_makespan=0.409 ms`, `gpu_kernel_sum=0.274 ms`
  开始时间(ns): `31139320`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.052 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.052 ms`, `host_to_first_kernel_gap=0.111366`, `host_end_to_last_kernel_tail=0.077 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `31563648`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.278 ms
  纯GPU kernel时间: `0.451 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.278 ms`, `host_to_first_kernel_gap=0.110812`, `host_end_to_last_kernel_tail=0.403 ms`, `gpu_makespan=0.570 ms`, `gpu_kernel_sum=0.451 ms`
  开始时间(ns): `31649866`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.044 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.044 ms`, `host_to_first_kernel_gap=0.338513`, `host_end_to_last_kernel_tail=0.307 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `31991797`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.091 ms
  纯GPU kernel时间: `0.105 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.091 ms`, `host_to_first_kernel_gap=0.266409`, `host_end_to_last_kernel_tail=0.280 ms`, `gpu_makespan=0.105 ms`, `gpu_kernel_sum=0.105 ms`
  开始时间(ns): `32077981`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.301 ms
  纯GPU kernel时间: `0.417 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.301 ms`, `host_to_first_kernel_gap=0.11445`, `host_end_to_last_kernel_tail=0.381 ms`, `gpu_makespan=0.568 ms`, `gpu_kernel_sum=0.417 ms`
  开始时间(ns): `32676500`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[12288, 512]]}`
- `attn_mha` -> 0.167 ms
  纯GPU kernel时间: `1.966 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.167 ms`, `host_to_first_kernel_gap=0.624801`, `host_end_to_last_kernel_tail=2.424 ms`, `gpu_makespan=1.966 ms`, `gpu_kernel_sum=1.966 ms`
  开始时间(ns): `33094725`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [12288, 128, 192], [12288, 128, 128]]}`
- `o_proj` -> 0.260 ms
  纯GPU kernel时间: `1.253 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.260 ms`, `host_to_first_kernel_gap=2.376612`, `host_end_to_last_kernel_tail=3.371 ms`, `gpu_makespan=1.254 ms`, `gpu_kernel_sum=1.253 ms`
  开始时间(ns): `33311009`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 1 / prefill / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `2.001 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.001 ms`, `module_to_last_kernel=12.013 ms`, `host_to_first_kernel_gap=7.120376`, `host_end_to_last_kernel_tail=10.012 ms`, `gpu_makespan=4.892 ms`, `gpu_kernel_sum=4.474 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=12288`, `chunked_req_prefix_len=4096`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [12288, 128, 192], [12288, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=4096`, `sum_seq_after=12288`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 2048, 'prefix_len': 1024, 'seq_len_after': 3072, 'prompt_len': 3072, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 2048, 'prefix_len': 1024, 'seq_len_after': 3072, 'prompt_len': 3072, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 2048, 'prefix_len': 1024, 'seq_len_after': 3072, 'prompt_len': 3072, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 2048, 'prefix_len': 1024, 'seq_len_after': 3072, 'prompt_len': 3072, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.242 ms
  纯GPU kernel时间: `0.267 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.242 ms`, `host_to_first_kernel_gap=6.989571`, `host_end_to_last_kernel_tail=7.018 ms`, `gpu_makespan=0.270 ms`, `gpu_kernel_sum=0.267 ms`
  开始时间(ns): `34574336`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.055 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.055 ms`, `host_to_first_kernel_gap=6.96746`, `host_end_to_last_kernel_tail=6.930 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `34867295`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.212 ms
  纯GPU kernel时间: `0.454 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.212 ms`, `host_to_first_kernel_gap=6.900058`, `host_end_to_last_kernel_tail=7.145 ms`, `gpu_makespan=0.457 ms`, `gpu_kernel_sum=0.454 ms`
  开始时间(ns): `34953385`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.043 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.043 ms`, `host_to_first_kernel_gap=7.083372`, `host_end_to_last_kernel_tail=7.054 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `35226998`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.085 ms
  纯GPU kernel时间: `0.105 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.085 ms`, `host_to_first_kernel_gap=7.013135`, `host_end_to_last_kernel_tail=7.033 ms`, `gpu_makespan=0.105 ms`, `gpu_kernel_sum=0.105 ms`
  开始时间(ns): `35311571`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.260 ms
  纯GPU kernel时间: `0.401 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.260 ms`, `host_to_first_kernel_gap=6.847627`, `host_end_to_last_kernel_tail=6.989 ms`, `gpu_makespan=0.402 ms`, `gpu_kernel_sum=0.401 ms`
  开始时间(ns): `35625559`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[12288, 512]]}`
- `attn_mha` -> 0.156 ms
  纯GPU kernel时间: `1.967 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.156 ms`, `host_to_first_kernel_gap=7.250469`, `host_end_to_last_kernel_tail=9.061 ms`, `gpu_makespan=1.967 ms`, `gpu_kernel_sum=1.967 ms`
  开始时间(ns): `35987933`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [12288, 128, 192], [12288, 128, 128]]}`
- `o_proj` -> 0.247 ms
  纯GPU kernel时间: `1.248 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.247 ms`, `host_to_first_kernel_gap=9.022408`, `host_end_to_last_kernel_tail=10.025 ms`, `gpu_makespan=1.249 ms`, `gpu_kernel_sum=1.248 ms`
  开始时间(ns): `36184473`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 2 / prefill / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.962 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.962 ms`, `module_to_last_kernel=18.689 ms`, `host_to_first_kernel_gap=13.810786`, `host_end_to_last_kernel_tail=16.727 ms`, `gpu_makespan=4.878 ms`, `gpu_kernel_sum=4.463 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=12288`, `chunked_req_prefix_len=4096`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [12288, 128, 192], [12288, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=4096`, `sum_seq_after=12288`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 2048, 'prefix_len': 1024, 'seq_len_after': 3072, 'prompt_len': 3072, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 2048, 'prefix_len': 1024, 'seq_len_after': 3072, 'prompt_len': 3072, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 2048, 'prefix_len': 1024, 'seq_len_after': 3072, 'prompt_len': 3072, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 2048, 'prefix_len': 1024, 'seq_len_after': 3072, 'prompt_len': 3072, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.232 ms
  纯GPU kernel时间: `0.268 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.232 ms`, `host_to_first_kernel_gap=13.676663`, `host_end_to_last_kernel_tail=13.715 ms`, `gpu_makespan=0.271 ms`, `gpu_kernel_sum=0.268 ms`
  开始时间(ns): `37406792`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.050 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.050 ms`, `host_to_first_kernel_gap=13.653571`, `host_end_to_last_kernel_tail=13.621 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `37700860`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.224 ms
  纯GPU kernel时间: `0.442 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.224 ms`, `host_to_first_kernel_gap=13.5901`, `host_end_to_last_kernel_tail=13.809 ms`, `gpu_makespan=0.443 ms`, `gpu_kernel_sum=0.442 ms`
  开始时间(ns): `37783307`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.046 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.046 ms`, `host_to_first_kernel_gap=13.748901`, `host_end_to_last_kernel_tail=13.716 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `38067706`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.085 ms
  纯GPU kernel时间: `0.105 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.085 ms`, `host_to_first_kernel_gap=13.677678`, `host_end_to_last_kernel_tail=13.698 ms`, `gpu_makespan=0.105 ms`, `gpu_kernel_sum=0.105 ms`
  开始时间(ns): `38154257`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.239 ms
  纯GPU kernel时间: `0.400 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.239 ms`, `host_to_first_kernel_gap=13.514476`, `host_end_to_last_kernel_tail=13.677 ms`, `gpu_makespan=0.402 ms`, `gpu_kernel_sum=0.400 ms`
  开始时间(ns): `38463827`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[12288, 512]]}`
- `attn_mha` -> 0.149 ms
  纯GPU kernel时间: `1.969 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.149 ms`, `host_to_first_kernel_gap=13.932184`, `host_end_to_last_kernel_tail=15.752 ms`, `gpu_makespan=1.969 ms`, `gpu_kernel_sum=1.969 ms`
  开始时间(ns): `38808423`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [12288, 128, 192], [12288, 128, 128]]}`
- `o_proj` -> 0.223 ms
  纯GPU kernel时间: `1.248 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.223 ms`, `host_to_first_kernel_gap=15.714126`, `host_end_to_last_kernel_tail=16.740 ms`, `gpu_makespan=1.249 ms`, `gpu_kernel_sum=1.248 ms`
  开始时间(ns): `38998704`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 3 / prefill / instance 1

- 执行序号: `4`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `2.004 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.004 ms`, `module_to_last_kernel=25.472 ms`, `host_to_first_kernel_gap=20.520709`, `host_end_to_last_kernel_tail=23.468 ms`, `gpu_makespan=4.952 ms`, `gpu_kernel_sum=4.533 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=12288`, `chunked_req_prefix_len=4096`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [12288, 128, 192], [12288, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=4096`, `sum_seq_after=12288`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 2048, 'prefix_len': 1024, 'seq_len_after': 3072, 'prompt_len': 3072, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 2048, 'prefix_len': 1024, 'seq_len_after': 3072, 'prompt_len': 3072, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 2048, 'prefix_len': 1024, 'seq_len_after': 3072, 'prompt_len': 3072, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 2048, 'prefix_len': 1024, 'seq_len_after': 3072, 'prompt_len': 3072, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.262 ms
  纯GPU kernel时间: `0.267 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.262 ms`, `host_to_first_kernel_gap=20.373724`, `host_end_to_last_kernel_tail=20.380 ms`, `gpu_makespan=0.269 ms`, `gpu_kernel_sum=0.267 ms`
  开始时间(ns): `40206560`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.053 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.053 ms`, `host_to_first_kernel_gap=20.32501`, `host_end_to_last_kernel_tail=20.290 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `40524074`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.243 ms
  纯GPU kernel时间: `0.445 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.243 ms`, `host_to_first_kernel_gap=20.259122`, `host_end_to_last_kernel_tail=20.462 ms`, `gpu_makespan=0.446 ms`, `gpu_kernel_sum=0.445 ms`
  开始时间(ns): `40610666`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.042 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.042 ms`, `host_to_first_kernel_gap=20.397338`, `host_end_to_last_kernel_tail=20.369 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `40918465`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.082 ms
  纯GPU kernel时间: `0.105 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.082 ms`, `host_to_first_kernel_gap=20.33081`, `host_end_to_last_kernel_tail=20.353 ms`, `gpu_makespan=0.105 ms`, `gpu_kernel_sum=0.105 ms`
  开始时间(ns): `41000449`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.222 ms
  纯GPU kernel时间: `0.465 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.222 ms`, `host_to_first_kernel_gap=20.167219`, `host_end_to_last_kernel_tail=20.412 ms`, `gpu_makespan=0.467 ms`, `gpu_kernel_sum=0.465 ms`
  开始时间(ns): `41310280`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[12288, 512]]}`
- `attn_mha` -> 0.158 ms
  纯GPU kernel时间: `1.971 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.158 ms`, `host_to_first_kernel_gap=20.673824`, `host_end_to_last_kernel_tail=22.486 ms`, `gpu_makespan=1.971 ms`, `gpu_kernel_sum=1.971 ms`
  开始时间(ns): `41631067`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [12288, 128, 192], [12288, 128, 128]]}`
- `o_proj` -> 0.221 ms
  纯GPU kernel时间: `1.250 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.221 ms`, `host_to_first_kernel_gap=22.450644`, `host_end_to_last_kernel_tail=23.482 ms`, `gpu_makespan=1.253 ms`, `gpu_kernel_sum=1.250 ms`
  开始时间(ns): `41828678`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 4 / prefill / instance 1

- 执行序号: `5`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `2.178 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.178 ms`, `module_to_last_kernel=37.344 ms`, `host_to_first_kernel_gap=32.462653`, `host_end_to_last_kernel_tail=35.167 ms`, `gpu_makespan=4.882 ms`, `gpu_kernel_sum=4.466 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=12288`, `chunked_req_prefix_len=4096`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [12288, 128, 192], [12288, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=4096`, `sum_seq_after=12288`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 2048, 'prefix_len': 1024, 'seq_len_after': 3072, 'prompt_len': 3072, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 2048, 'prefix_len': 1024, 'seq_len_after': 3072, 'prompt_len': 3072, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 2048, 'prefix_len': 1024, 'seq_len_after': 3072, 'prompt_len': 3072, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 2048, 'prefix_len': 1024, 'seq_len_after': 3072, 'prompt_len': 3072, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.318 ms
  纯GPU kernel时间: `0.268 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.318 ms`, `host_to_first_kernel_gap=32.311185`, `host_end_to_last_kernel_tail=32.264 ms`, `gpu_makespan=0.270 ms`, `gpu_kernel_sum=0.268 ms`
  开始时间(ns): `43671077`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.053 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.053 ms`, `host_to_first_kernel_gap=32.210905`, `host_end_to_last_kernel_tail=32.175 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `44042269`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.227 ms
  纯GPU kernel时间: `0.441 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.227 ms`, `host_to_first_kernel_gap=32.146657`, `host_end_to_last_kernel_tail=32.363 ms`, `gpu_makespan=0.443 ms`, `gpu_kernel_sum=0.441 ms`
  开始时间(ns): `44125461`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.044 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.044 ms`, `host_to_first_kernel_gap=32.301365`, `host_end_to_last_kernel_tail=32.270 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `44413793`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.097 ms
  纯GPU kernel时间: `0.105 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.097 ms`, `host_to_first_kernel_gap=32.232054`, `host_end_to_last_kernel_tail=32.240 ms`, `gpu_makespan=0.105 ms`, `gpu_kernel_sum=0.105 ms`
  开始时间(ns): `44497600`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.261 ms
  纯GPU kernel时间: `0.402 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.261 ms`, `host_to_first_kernel_gap=32.044646`, `host_end_to_last_kernel_tail=32.187 ms`, `gpu_makespan=0.404 ms`, `gpu_kernel_sum=0.402 ms`
  开始时间(ns): `44830832`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[12288, 512]]}`
- `attn_mha` -> 0.165 ms
  纯GPU kernel时间: `1.972 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.165 ms`, `host_to_first_kernel_gap=32.436814`, `host_end_to_last_kernel_tail=34.244 ms`, `gpu_makespan=1.972 ms`, `gpu_kernel_sum=1.972 ms`
  开始时间(ns): `45204071`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [12288, 128, 192], [12288, 128, 128]]}`
- `o_proj` -> 0.255 ms
  纯GPU kernel时间: `1.248 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.255 ms`, `host_to_first_kernel_gap=34.198764`, `host_end_to_last_kernel_tail=35.194 ms`, `gpu_makespan=1.250 ms`, `gpu_kernel_sum=1.248 ms`
  开始时间(ns): `45415593`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 0 / decode / instance 1

- 执行序号: `6`
- Collector对齐边界: `{'Module': 'model.model.layers.0.self_attn'}`
- 整块 MLA-module 时长: `2.091 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.091 ms`, `module_to_last_kernel=42.320 ms`, `host_to_first_kernel_gap=42.168498`, `host_end_to_last_kernel_tail=40.228 ms`, `gpu_makespan=0.151 ms`, `gpu_kernel_sum=0.119 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.351 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.351 ms`, `host_to_first_kernel_gap=42.088402`, `host_end_to_last_kernel_tail=41.753 ms`, `gpu_makespan=0.016 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `49941086`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4, 7168]]}`
- `q_a_layernorm` -> 0.049 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.049 ms`, `host_to_first_kernel_gap=41.684548`, `host_end_to_last_kernel_tail=41.637 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `50360300`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4, 1536]]}`
- `kv_a_layernorm` -> 0.038 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.038 ms`, `host_to_first_kernel_gap=41.612431`, `host_end_to_last_kernel_tail=41.576 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `50434401`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4, 512]]}`
- `q_b_proj` -> 0.259 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.259 ms`, `host_to_first_kernel_gap=41.536809`, `host_end_to_last_kernel_tail=41.299 ms`, `gpu_makespan=0.021 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `50513159`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4, 1536]]}`
- `rotary_emb` -> 0.094 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.094 ms`, `host_to_first_kernel_gap=41.122198`, `host_end_to_last_kernel_tail=41.030 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `50960666`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4], [4, 128, 64], [4, 1, 64]]}`
- `attn_mqa` -> 0.339 ms
  纯GPU kernel时间: `0.031 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.339 ms`, `host_to_first_kernel_gap=40.97889`, `host_end_to_last_kernel_tail=40.674 ms`, `gpu_makespan=0.033 ms`, `gpu_kernel_sum=0.031 ms`
  开始时间(ns): `51106790`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4, 128, 512], [4, 1, 512], [4, 1, 512]]}`
- `o_proj` -> 0.337 ms
  纯GPU kernel时间: `0.049 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.337 ms`, `host_to_first_kernel_gap=40.536278`, `host_end_to_last_kernel_tail=40.249 ms`, `gpu_makespan=0.050 ms`, `gpu_kernel_sum=0.049 ms`
  开始时间(ns): `51594810`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4, 16384]]}`

## Layer 1 / decode / instance 1

- 执行序号: `7`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `1.786 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.786 ms`, `module_to_last_kernel=39.637 ms`, `host_to_first_kernel_gap=39.487586`, `host_end_to_last_kernel_tail=37.851 ms`, `gpu_makespan=0.149 ms`, `gpu_kernel_sum=0.119 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.217 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.217 ms`, `host_to_first_kernel_gap=39.418839`, `host_end_to_last_kernel_tail=39.218 ms`, `gpu_makespan=0.015 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `52923065`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4, 7168]]}`
- `q_a_layernorm` -> 0.049 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.049 ms`, `host_to_first_kernel_gap=39.153949`, `host_end_to_last_kernel_tail=39.107 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `53203283`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4, 1536]]}`
- `kv_a_layernorm` -> 0.033 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.033 ms`, `host_to_first_kernel_gap=39.08025`, `host_end_to_last_kernel_tail=39.049 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `53279094`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4, 512]]}`
- `q_b_proj` -> 0.206 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.206 ms`, `host_to_first_kernel_gap=39.016484`, `host_end_to_last_kernel_tail=38.829 ms`, `gpu_makespan=0.019 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `53345964`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4, 1536]]}`
- `rotary_emb` -> 0.089 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.089 ms`, `host_to_first_kernel_gap=38.656039`, `host_end_to_last_kernel_tail=38.569 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `53736905`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4], [4, 128, 64], [4, 1, 64]]}`
- `attn_mqa` -> 0.310 ms
  纯GPU kernel时间: `0.032 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.310 ms`, `host_to_first_kernel_gap=38.522732`, `host_end_to_last_kernel_tail=38.246 ms`, `gpu_makespan=0.034 ms`, `gpu_kernel_sum=0.032 ms`
  开始时间(ns): `53872996`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4, 128, 512], [4, 1, 512], [4, 1, 512]]}`
- `o_proj` -> 0.298 ms
  纯GPU kernel时间: `0.049 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.298 ms`, `host_to_first_kernel_gap=38.11783`, `host_end_to_last_kernel_tail=37.869 ms`, `gpu_makespan=0.050 ms`, `gpu_kernel_sum=0.049 ms`
  开始时间(ns): `54323466`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4, 16384]]}`

## Layer 2 / decode / instance 1

- 执行序号: `8`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.778 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.778 ms`, `module_to_last_kernel=37.284 ms`, `host_to_first_kernel_gap=37.134034`, `host_end_to_last_kernel_tail=35.506 ms`, `gpu_makespan=0.150 ms`, `gpu_kernel_sum=0.119 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.228 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.228 ms`, `host_to_first_kernel_gap=37.065094`, `host_end_to_last_kernel_tail=36.854 ms`, `gpu_makespan=0.017 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `55586730`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4, 7168]]}`
- `q_a_layernorm` -> 0.051 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.051 ms`, `host_to_first_kernel_gap=36.788`, `host_end_to_last_kernel_tail=36.740 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `55880656`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4, 1536]]}`
- `kv_a_layernorm` -> 0.033 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.033 ms`, `host_to_first_kernel_gap=36.716207`, `host_end_to_last_kernel_tail=36.685 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `55954529`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4, 512]]}`
- `q_b_proj` -> 0.200 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.200 ms`, `host_to_first_kernel_gap=36.651974`, `host_end_to_last_kernel_tail=36.471 ms`, `gpu_makespan=0.019 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `56021930`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4, 1536]]}`
- `rotary_emb` -> 0.093 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.093 ms`, `host_to_first_kernel_gap=36.324516`, `host_end_to_last_kernel_tail=36.233 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `56380012`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4], [4, 128, 64], [4, 1, 64]]}`
- `attn_mqa` -> 0.318 ms
  纯GPU kernel时间: `0.031 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.318 ms`, `host_to_first_kernel_gap=36.186681`, `host_end_to_last_kernel_tail=35.902 ms`, `gpu_makespan=0.033 ms`, `gpu_kernel_sum=0.031 ms`
  开始时间(ns): `56520727`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4, 128, 512], [4, 1, 512], [4, 1, 512]]}`
- `o_proj` -> 0.294 ms
  纯GPU kernel时间: `0.049 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.294 ms`, `host_to_first_kernel_gap=35.768863`, `host_end_to_last_kernel_tail=35.526 ms`, `gpu_makespan=0.050 ms`, `gpu_kernel_sum=0.049 ms`
  开始时间(ns): `56982737`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4, 16384]]}`

## Layer 3 / decode / instance 1

- 执行序号: `9`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `1.774 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.774 ms`, `module_to_last_kernel=34.939 ms`, `host_to_first_kernel_gap=34.78593`, `host_end_to_last_kernel_tail=33.165 ms`, `gpu_makespan=0.153 ms`, `gpu_kernel_sum=0.121 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.213 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.213 ms`, `host_to_first_kernel_gap=34.715266`, `host_end_to_last_kernel_tail=34.518 ms`, `gpu_makespan=0.016 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `58248654`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4, 7168]]}`
- `q_a_layernorm` -> 0.051 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.051 ms`, `host_to_first_kernel_gap=34.452109`, `host_end_to_last_kernel_tail=34.403 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `58527555`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4, 1536]]}`
- `kv_a_layernorm` -> 0.033 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.033 ms`, `host_to_first_kernel_gap=34.379807`, `host_end_to_last_kernel_tail=34.349 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `58601777`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4, 512]]}`
- `q_b_proj` -> 0.218 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.218 ms`, `host_to_first_kernel_gap=34.319048`, `host_end_to_last_kernel_tail=34.120 ms`, `gpu_makespan=0.019 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `58666792`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4, 1536]]}`
- `rotary_emb` -> 0.088 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.088 ms`, `host_to_first_kernel_gap=33.970757`, `host_end_to_last_kernel_tail=33.885 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `59045323`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4], [4, 128, 64], [4, 1, 64]]}`
- `attn_mqa` -> 0.311 ms
  纯GPU kernel时间: `0.031 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.311 ms`, `host_to_first_kernel_gap=33.835134`, `host_end_to_last_kernel_tail=33.556 ms`, `gpu_makespan=0.032 ms`, `gpu_kernel_sum=0.031 ms`
  开始时间(ns): `59183890`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4, 128, 512], [4, 1, 512], [4, 1, 512]]}`
- `o_proj` -> 0.303 ms
  纯GPU kernel时间: `0.051 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.303 ms`, `host_to_first_kernel_gap=33.434422`, `host_end_to_last_kernel_tail=33.184 ms`, `gpu_makespan=0.052 ms`, `gpu_kernel_sum=0.051 ms`
  开始时间(ns): `59629978`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4, 16384]]}`

## Layer 4 / decode / instance 1

- 执行序号: `10`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `1.942 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.942 ms`, `module_to_last_kernel=31.952 ms`, `host_to_first_kernel_gap=31.803141`, `host_end_to_last_kernel_tail=30.010 ms`, `gpu_makespan=0.149 ms`, `gpu_kernel_sum=0.121 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.324 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.324 ms`, `host_to_first_kernel_gap=31.723592`, `host_end_to_last_kernel_tail=31.415 ms`, `gpu_makespan=0.015 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `61609800`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4, 7168]]}`
- `q_a_layernorm` -> 0.056 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.056 ms`, `host_to_first_kernel_gap=31.346551`, `host_end_to_last_kernel_tail=31.293 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `62002073`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4, 1536]]}`
- `kv_a_layernorm` -> 0.033 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.033 ms`, `host_to_first_kernel_gap=31.269153`, `host_end_to_last_kernel_tail=31.238 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `62081615`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4, 512]]}`
- `q_b_proj` -> 0.228 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.228 ms`, `host_to_first_kernel_gap=31.201889`, `host_end_to_last_kernel_tail=30.994 ms`, `gpu_makespan=0.020 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `62151887`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4, 1536]]}`
- `rotary_emb` -> 0.092 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.092 ms`, `host_to_first_kernel_gap=30.829263`, `host_end_to_last_kernel_tail=30.739 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `62555137`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4], [4, 128, 64], [4, 1, 64]]}`
- `attn_mqa` -> 0.316 ms
  纯GPU kernel时间: `0.031 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.316 ms`, `host_to_first_kernel_gap=30.675924`, `host_end_to_last_kernel_tail=30.393 ms`, `gpu_makespan=0.033 ms`, `gpu_kernel_sum=0.031 ms`
  开始时间(ns): `62712316`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4, 128, 512], [4, 1, 512], [4, 1, 512]]}`
- `o_proj` -> 0.283 ms
  纯GPU kernel时间: `0.050 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.283 ms`, `host_to_first_kernel_gap=30.263772`, `host_end_to_last_kernel_tail=30.032 ms`, `gpu_makespan=0.051 ms`, `gpu_kernel_sum=0.050 ms`
  开始时间(ns): `63167988`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4, 16384]]}`
