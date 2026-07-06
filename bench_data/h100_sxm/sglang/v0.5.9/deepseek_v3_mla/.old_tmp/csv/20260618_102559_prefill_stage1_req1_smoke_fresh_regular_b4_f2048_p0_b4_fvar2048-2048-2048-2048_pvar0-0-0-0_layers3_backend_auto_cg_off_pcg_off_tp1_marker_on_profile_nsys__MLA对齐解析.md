# MLA对齐解析摘要

源文件: `report.sqlite`

## 对齐原则

- `collector/sglang/collect_mla_module.py` 的 MLA module 计时边界是 `model.model.layers[test_layer].self_attn(...)`。
- 因此这里把 `nsys` 中每层的 `model.model.layers.X.self_attn` NVTX range 视为与 collector 对齐的 MLA-module 边界。
- 该区间内部的 `.self_attn.*` 子模块用于做 MLA 内部 breakdown。

## 运行摘要

- `prefill` 对齐成功层: `[0, 1, 2]`
- `prefill` 被切分层: `[]`
- 若某层 `prefill` 出现多个 `self_attn` 实例，则说明 Engine 调度把一次前向切成了多块，已不再与 collector 的单次 MLA-module 采集严格一一对应。

## Layer 0 / prefill / instance 1

- 执行序号: `1`
- Collector对齐边界: `{'Module': 'model.model.layers.0.self_attn'}`
- 整块 MLA-module 时长: `5.091 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=5.091 ms`, `module_to_last_kernel=6.659 ms`, `host_to_first_kernel_gap=0.63872`, `host_end_to_last_kernel_tail=1.568 ms`, `gpu_makespan=6.020 ms`, `gpu_kernel_sum=3.532 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=8192`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=0`, `sum_seq_after=8192`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.824 ms
  纯GPU kernel时间: `0.274 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.824 ms`, `host_to_first_kernel_gap=0.283102`, `host_end_to_last_kernel_tail=0.119 ms`, `gpu_makespan=0.660 ms`, `gpu_kernel_sum=0.274 ms`
  开始时间(ns): `34749517`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.120 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.120 ms`, `host_to_first_kernel_gap=0.10087`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `35720356`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.625 ms
  纯GPU kernel时间: `0.450 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.625 ms`, `host_to_first_kernel_gap=0.226803`, `host_end_to_last_kernel_tail=0.362 ms`, `gpu_makespan=0.760 ms`, `gpu_kernel_sum=0.450 ms`
  开始时间(ns): `35932951`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.107 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.107 ms`, `host_to_first_kernel_gap=0.193664`, `host_end_to_last_kernel_tail=0.100 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `36726058`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.203 ms
  纯GPU kernel时间: `0.105 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.203 ms`, `host_to_first_kernel_gap=0.169221`, `host_end_to_last_kernel_tail=0.071 ms`, `gpu_makespan=0.105 ms`, `gpu_kernel_sum=0.105 ms`
  开始时间(ns): `36938085`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.590 ms
  纯GPU kernel时间: `0.297 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.590 ms`, `host_to_first_kernel_gap=0.210004`, `host_end_to_last_kernel_tail=0.216 ms`, `gpu_makespan=0.596 ms`, `gpu_kernel_sum=0.297 ms`
  开始时间(ns): `37492886`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[8192, 512]]}`
- `attn_mha` -> 0.387 ms
  纯GPU kernel时间: `1.124 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.387 ms`, `host_to_first_kernel_gap=0.281594`, `host_end_to_last_kernel_tail=1.067 ms`, `gpu_makespan=1.172 ms`, `gpu_kernel_sum=1.124 ms`
  开始时间(ns): `38344752`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]}`
- `o_proj` -> 0.610 ms
  纯GPU kernel时间: `1.252 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.610 ms`, `host_to_first_kernel_gap=0.966485`, `host_end_to_last_kernel_tail=1.609 ms`, `gpu_makespan=1.253 ms`, `gpu_kernel_sum=1.252 ms`
  开始时间(ns): `38833205`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 1 / prefill / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `4.703 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=4.703 ms`, `module_to_last_kernel=6.597 ms`, `host_to_first_kernel_gap=2.815071`, `host_end_to_last_kernel_tail=1.894 ms`, `gpu_makespan=3.782 ms`, `gpu_kernel_sum=3.502 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=8192`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=0`, `sum_seq_after=8192`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.631 ms
  纯GPU kernel时间: `0.266 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.631 ms`, `host_to_first_kernel_gap=2.492238`, `host_end_to_last_kernel_tail=2.128 ms`, `gpu_makespan=0.268 ms`, `gpu_kernel_sum=0.266 ms`
  开始时间(ns): `43332762`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.120 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.120 ms`, `host_to_first_kernel_gap=1.983771`, `host_end_to_last_kernel_tail=1.881 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `44109037`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.564 ms
  纯GPU kernel时间: `0.441 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.564 ms`, `host_to_first_kernel_gap=1.803389`, `host_end_to_last_kernel_tail=1.682 ms`, `gpu_makespan=0.443 ms`, `gpu_kernel_sum=0.441 ms`
  开始时间(ns): `44308267`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.102 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.102 ms`, `host_to_first_kernel_gap=1.520023`, `host_end_to_last_kernel_tail=1.431 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `45034193`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.198 ms
  纯GPU kernel时间: `0.105 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.198 ms`, `host_to_first_kernel_gap=1.326747`, `host_end_to_last_kernel_tail=1.234 ms`, `gpu_makespan=0.105 ms`, `gpu_kernel_sum=0.105 ms`
  开始时间(ns): `45242253`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.577 ms
  纯GPU kernel时间: `0.288 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.577 ms`, `host_to_first_kernel_gap=0.914834`, `host_end_to_last_kernel_tail=0.626 ms`, `gpu_makespan=0.289 ms`, `gpu_kernel_sum=0.288 ms`
  开始时间(ns): `45787766`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[8192, 512]]}`
- `attn_mha` -> 0.369 ms
  纯GPU kernel时间: `1.121 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.369 ms`, `host_to_first_kernel_gap=0.602952`, `host_end_to_last_kernel_tail=1.355 ms`, `gpu_makespan=1.120 ms`, `gpu_kernel_sum=1.121 ms`
  开始时间(ns): `46631008`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]}`
- `o_proj` -> 0.579 ms
  纯GPU kernel时间: `1.250 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.579 ms`, `host_to_first_kernel_gap=1.253992`, `host_end_to_last_kernel_tail=1.926 ms`, `gpu_makespan=1.251 ms`, `gpu_kernel_sum=1.250 ms`
  开始时间(ns): `47101760`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 2 / prefill / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `4.423 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=4.423 ms`, `module_to_last_kernel=8.196 ms`, `host_to_first_kernel_gap=4.401614`, `host_end_to_last_kernel_tail=3.772 ms`, `gpu_makespan=3.794 ms`, `gpu_kernel_sum=3.512 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=8192`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=0`, `sum_seq_after=8192`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.584 ms
  纯GPU kernel时间: `0.266 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.584 ms`, `host_to_first_kernel_gap=4.115401`, `host_end_to_last_kernel_tail=3.799 ms`, `gpu_makespan=0.267 ms`, `gpu_kernel_sum=0.266 ms`
  开始时间(ns): `50106589`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.121 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.121 ms`, `host_to_first_kernel_gap=3.655846`, `host_end_to_last_kernel_tail=3.554 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `50833088`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.537 ms
  纯GPU kernel时间: `0.447 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.537 ms`, `host_to_first_kernel_gap=3.480744`, `host_end_to_last_kernel_tail=3.393 ms`, `gpu_makespan=0.449 ms`, `gpu_kernel_sum=0.447 ms`
  开始时间(ns): `51027774`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.105 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.105 ms`, `host_to_first_kernel_gap=3.236579`, `host_end_to_last_kernel_tail=3.145 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `51720867`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.176 ms
  纯GPU kernel时间: `0.105 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.176 ms`, `host_to_first_kernel_gap=3.043805`, `host_end_to_last_kernel_tail=2.973 ms`, `gpu_makespan=0.105 ms`, `gpu_kernel_sum=0.105 ms`
  开始时间(ns): `51928265`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.543 ms
  纯GPU kernel时间: `0.288 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.543 ms`, `host_to_first_kernel_gap=2.671232`, `host_end_to_last_kernel_tail=2.418 ms`, `gpu_makespan=0.290 ms`, `gpu_kernel_sum=0.288 ms`
  开始时间(ns): `52435014`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[8192, 512]]}`
- `attn_mha` -> 0.355 ms
  纯GPU kernel时间: `1.121 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.355 ms`, `host_to_first_kernel_gap=2.425222`, `host_end_to_last_kernel_tail=3.191 ms`, `gpu_makespan=1.121 ms`, `gpu_kernel_sum=1.121 ms`
  开始时间(ns): `53214048`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]}`
- `o_proj` -> 0.548 ms
  纯GPU kernel时间: `1.252 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.548 ms`, `host_to_first_kernel_gap=3.100136`, `host_end_to_last_kernel_tail=3.805 ms`, `gpu_makespan=1.253 ms`, `gpu_kernel_sum=1.252 ms`
  开始时间(ns): `53662974`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 0 / decode / instance 1

- 执行序号: `4`
- Collector对齐边界: `{'Module': 'model.model.layers.0.self_attn'}`
- 整块 MLA-module 时长: `3.593 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=3.593 ms`, `module_to_last_kernel=3.593 ms`, `host_to_first_kernel_gap=1.940003`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=1.610 ms`, `gpu_kernel_sum=0.119 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.597 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.597 ms`, `host_to_first_kernel_gap=1.800642`, `host_end_to_last_kernel_tail=1.220 ms`, `gpu_makespan=0.016 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `61564418`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4, 7168]]}`
- `q_a_layernorm` -> 0.103 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.103 ms`, `host_to_first_kernel_gap=1.08893`, `host_end_to_last_kernel_tail=0.988 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `62292226`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4, 1536]]}`
- `kv_a_layernorm` -> 0.063 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.063 ms`, `host_to_first_kernel_gap=0.932959`, `host_end_to_last_kernel_tail=0.872 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `62450149`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4, 512]]}`
- `q_b_proj` -> 0.483 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.483 ms`, `host_to_first_kernel_gap=0.795961`, `host_end_to_last_kernel_tail=0.333 ms`, `gpu_makespan=0.020 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `62590891`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4, 1536]]}`
- `rotary_emb` -> 0.163 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.163 ms`, `host_to_first_kernel_gap=0.135395`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `63393089`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4], [4, 128, 64], [4, 1, 64]]}`
- `attn_mqa` -> 0.563 ms
  纯GPU kernel时间: `0.029 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.563 ms`, `host_to_first_kernel_gap=0.172697`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.339 ms`, `gpu_kernel_sum=0.029 ms`
  开始时间(ns): `63648843`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4, 128, 512], [4, 1, 512], [4, 1, 512]]}`
- `o_proj` -> 0.519 ms
  纯GPU kernel时间: `0.050 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.519 ms`, `host_to_first_kernel_gap=0.183548`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.328 ms`, `gpu_kernel_sum=0.050 ms`
  开始时间(ns): `64464136`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4, 16384]]}`

## Layer 1 / decode / instance 1

- 执行序号: `5`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `3.051 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=3.051 ms`, `module_to_last_kernel=3.051 ms`, `host_to_first_kernel_gap=0.295957`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=2.729 ms`, `gpu_kernel_sum=0.117 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.439 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.439 ms`, `host_to_first_kernel_gap=0.164725`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.245 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `66902383`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4, 7168]]}`
- `q_a_layernorm` -> 0.091 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.091 ms`, `host_to_first_kernel_gap=0.076094`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `67477926`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4, 1536]]}`
- `kv_a_layernorm` -> 0.062 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.062 ms`, `host_to_first_kernel_gap=0.05152`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `67617828`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4, 512]]}`
- `q_b_proj` -> 0.396 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.396 ms`, `host_to_first_kernel_gap=0.144554`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.228 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `67743545`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4, 1536]]}`
- `rotary_emb` -> 0.148 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.148 ms`, `host_to_first_kernel_gap=0.120755`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `68408528`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4], [4, 128, 64], [4, 1, 64]]}`
- `attn_mqa` -> 0.489 ms
  纯GPU kernel时间: `0.029 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.489 ms`, `host_to_first_kernel_gap=0.154219`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.296 ms`, `gpu_kernel_sum=0.029 ms`
  开始时间(ns): `68636248`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4, 128, 512], [4, 1, 512], [4, 1, 512]]}`
- `o_proj` -> 0.454 ms
  纯GPU kernel时间: `0.049 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.454 ms`, `host_to_first_kernel_gap=0.169075`, `host_end_to_last_kernel_tail=0.004 ms`, `gpu_makespan=0.288 ms`, `gpu_kernel_sum=0.049 ms`
  开始时间(ns): `69338704`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4, 16384]]}`

## Layer 2 / decode / instance 1

- 执行序号: `6`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `2.723 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.723 ms`, `module_to_last_kernel=2.723 ms`, `host_to_first_kernel_gap=0.263745`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=2.447 ms`, `gpu_kernel_sum=0.117 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.380 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.380 ms`, `host_to_first_kernel_gap=0.146351`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.210 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `71392980`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4, 7168]]}`
- `q_a_layernorm` -> 0.079 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.079 ms`, `host_to_first_kernel_gap=0.067846`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `71876925`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4, 1536]]}`
- `kv_a_layernorm` -> 0.054 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.054 ms`, `host_to_first_kernel_gap=0.045781`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `71998542`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4, 512]]}`
- `q_b_proj` -> 0.361 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.361 ms`, `host_to_first_kernel_gap=0.130873`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.210 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `72108746`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4, 1536]]}`
- `rotary_emb` -> 0.126 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.126 ms`, `host_to_first_kernel_gap=0.105679`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `72704307`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4], [4, 128, 64], [4, 1, 64]]}`
- `attn_mqa` -> 0.474 ms
  纯GPU kernel时间: `0.029 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.474 ms`, `host_to_first_kernel_gap=0.148797`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.279 ms`, `gpu_kernel_sum=0.029 ms`
  开始时间(ns): `72904549`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4, 128, 512], [4, 1, 512], [4, 1, 512]]}`
- `o_proj` -> 0.399 ms
  纯GPU kernel时间: `0.049 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.399 ms`, `host_to_first_kernel_gap=0.147332`, `host_end_to_last_kernel_tail=0.012 ms`, `gpu_makespan=0.263 ms`, `gpu_kernel_sum=0.049 ms`
  开始时间(ns): `73576062`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4, 16384]]}`
