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
- 整块 MLA-module 时长: `3.443 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=3.443 ms`, `module_to_last_kernel=8.837 ms`, `host_to_first_kernel_gap=0.35406`, `host_end_to_last_kernel_tail=5.393 ms`, `gpu_makespan=8.483 ms`, `gpu_kernel_sum=6.862 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=9216`, `chunked_req_prefix_len=1024`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [9216, 128, 192], [9216, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=1024`, `sum_seq_after=9216`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 8192, 'prefix_len': 1024, 'seq_len_after': 9216, 'prompt_len': 9216, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.497 ms
  纯GPU kernel时间: `0.273 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.497 ms`, `host_to_first_kernel_gap=0.171916`, `host_end_to_last_kernel_tail=0.152 ms`, `gpu_makespan=0.477 ms`, `gpu_kernel_sum=0.273 ms`
  开始时间(ns): `41316138`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.071 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.071 ms`, `host_to_first_kernel_gap=0.071259`, `host_end_to_last_kernel_tail=0.018 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `41893691`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.363 ms
  纯GPU kernel时间: `0.449 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.363 ms`, `host_to_first_kernel_gap=0.126735`, `host_end_to_last_kernel_tail=0.394 ms`, `gpu_makespan=0.630 ms`, `gpu_kernel_sum=0.449 ms`
  开始时间(ns): `42018695`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.065 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.065 ms`, `host_to_first_kernel_gap=0.300988`, `host_end_to_last_kernel_tail=0.249 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `42474522`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.120 ms
  纯GPU kernel时间: `0.106 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.120 ms`, `host_to_first_kernel_gap=0.186907`, `host_end_to_last_kernel_tail=0.172 ms`, `gpu_makespan=0.106 ms`, `gpu_kernel_sum=0.106 ms`
  开始时间(ns): `42602683`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.371 ms
  纯GPU kernel时间: `0.324 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.371 ms`, `host_to_first_kernel_gap=0.132693`, `host_end_to_last_kernel_tail=0.279 ms`, `gpu_makespan=0.517 ms`, `gpu_kernel_sum=0.324 ms`
  开始时间(ns): `43366049`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[9216, 512]]}`
- `attn_mha` -> 0.217 ms
  纯GPU kernel时间: `4.430 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.217 ms`, `host_to_first_kernel_gap=0.400588`, `host_end_to_last_kernel_tail=4.614 ms`, `gpu_makespan=4.430 ms`, `gpu_kernel_sum=4.430 ms`
  开始时间(ns): `43888841`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [9216, 128, 192], [9216, 128, 128]]}`
- `o_proj` -> 0.371 ms
  纯GPU kernel时间: `1.248 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.371 ms`, `host_to_first_kernel_gap=4.540299`, `host_end_to_last_kernel_tail=5.418 ms`, `gpu_makespan=1.249 ms`, `gpu_kernel_sum=1.248 ms`
  开始时间(ns): `44180937`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 1 / prefill / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `2.707 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.707 ms`, `module_to_last_kernel=15.913 ms`, `host_to_first_kernel_gap=8.744306`, `host_end_to_last_kernel_tail=13.206 ms`, `gpu_makespan=7.169 ms`, `gpu_kernel_sum=6.845 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=9216`, `chunked_req_prefix_len=1024`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [9216, 128, 192], [9216, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=1024`, `sum_seq_after=9216`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 8192, 'prefix_len': 1024, 'seq_len_after': 9216, 'prompt_len': 9216, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.339 ms
  纯GPU kernel时间: `0.267 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.339 ms`, `host_to_first_kernel_gap=8.591107`, `host_end_to_last_kernel_tail=8.521 ms`, `gpu_makespan=0.269 ms`, `gpu_kernel_sum=0.267 ms`
  开始时间(ns): `45990704`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.067 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.067 ms`, `host_to_first_kernel_gap=8.451405`, `host_end_to_last_kernel_tail=8.402 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `46399366`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.308 ms
  纯GPU kernel时间: `0.447 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.308 ms`, `host_to_first_kernel_gap=8.360683`, `host_end_to_last_kernel_tail=8.501 ms`, `gpu_makespan=0.448 ms`, `gpu_kernel_sum=0.447 ms`
  开始时间(ns): `46509736`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.057 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.057 ms`, `host_to_first_kernel_gap=8.41243`, `host_end_to_last_kernel_tail=8.368 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `46906245`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.107 ms
  纯GPU kernel时间: `0.106 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.107 ms`, `host_to_first_kernel_gap=8.313343`, `host_end_to_last_kernel_tail=8.312 ms`, `gpu_makespan=0.106 ms`, `gpu_kernel_sum=0.106 ms`
  开始时间(ns): `47020500`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.314 ms
  纯GPU kernel时间: `0.317 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.314 ms`, `host_to_first_kernel_gap=8.037992`, `host_end_to_last_kernel_tail=8.043 ms`, `gpu_makespan=0.319 ms`, `gpu_kernel_sum=0.317 ms`
  开始时间(ns): `47440011`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[9216, 512]]}`
- `attn_mha` -> 0.208 ms
  纯GPU kernel时间: `4.429 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.208 ms`, `host_to_first_kernel_gap=8.172978`, `host_end_to_last_kernel_tail=12.394 ms`, `gpu_makespan=4.429 ms`, `gpu_kernel_sum=4.429 ms`
  开始时间(ns): `47896384`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [9216, 128, 192], [9216, 128, 128]]}`
- `o_proj` -> 0.355 ms
  纯GPU kernel时间: `1.248 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.355 ms`, `host_to_first_kernel_gap=12.331147`, `host_end_to_last_kernel_tail=13.227 ms`, `gpu_makespan=1.250 ms`, `gpu_kernel_sum=1.248 ms`
  开始时间(ns): `48169638`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 2 / prefill / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `2.661 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.661 ms`, `module_to_last_kernel=23.800 ms`, `host_to_first_kernel_gap=16.629656`, `host_end_to_last_kernel_tail=21.139 ms`, `gpu_makespan=7.170 ms`, `gpu_kernel_sum=6.849 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=9216`, `chunked_req_prefix_len=1024`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [9216, 128, 192], [9216, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=1024`, `sum_seq_after=9216`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 8192, 'prefix_len': 1024, 'seq_len_after': 9216, 'prompt_len': 9216, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.322 ms
  纯GPU kernel时间: `0.267 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.322 ms`, `host_to_first_kernel_gap=16.482351`, `host_end_to_last_kernel_tail=16.429 ms`, `gpu_makespan=0.268 ms`, `gpu_kernel_sum=0.267 ms`
  开始时间(ns): `49884001`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.070 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.070 ms`, `host_to_first_kernel_gap=16.359282`, `host_end_to_last_kernel_tail=16.307 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `50275198`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.294 ms
  纯GPU kernel时间: `0.437 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.294 ms`, `host_to_first_kernel_gap=16.265356`, `host_end_to_last_kernel_tail=16.410 ms`, `gpu_makespan=0.439 ms`, `gpu_kernel_sum=0.437 ms`
  开始时间(ns): `50388836`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.057 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.057 ms`, `host_to_first_kernel_gap=16.322628`, `host_end_to_last_kernel_tail=16.279 ms`, `gpu_makespan=0.014 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `50769996`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.108 ms
  纯GPU kernel时间: `0.106 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.108 ms`, `host_to_first_kernel_gap=16.225132`, `host_end_to_last_kernel_tail=16.222 ms`, `gpu_makespan=0.106 ms`, `gpu_kernel_sum=0.106 ms`
  开始时间(ns): `50882756`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.333 ms
  纯GPU kernel时间: `0.320 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.333 ms`, `host_to_first_kernel_gap=15.948665`, `host_end_to_last_kernel_tail=15.937 ms`, `gpu_makespan=0.321 ms`, `gpu_kernel_sum=0.320 ms`
  开始时间(ns): `51301911`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[9216, 512]]}`
- `attn_mha` -> 0.199 ms
  纯GPU kernel时间: `4.437 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.199 ms`, `host_to_first_kernel_gap=16.067369`, `host_end_to_last_kernel_tail=20.306 ms`, `gpu_makespan=4.437 ms`, `gpu_kernel_sum=4.437 ms`
  开始时间(ns): `51778631`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [9216, 128, 192], [9216, 128, 128]]}`
- `o_proj` -> 0.337 ms
  纯GPU kernel时间: `1.251 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.337 ms`, `host_to_first_kernel_gap=20.24299`, `host_end_to_last_kernel_tail=21.158 ms`, `gpu_makespan=1.252 ms`, `gpu_kernel_sum=1.251 ms`
  开始时间(ns): `52041920`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 3 / prefill / instance 1

- 执行序号: `4`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `2.646 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.646 ms`, `module_to_last_kernel=31.775 ms`, `host_to_first_kernel_gap=24.595294`, `host_end_to_last_kernel_tail=29.129 ms`, `gpu_makespan=7.180 ms`, `gpu_kernel_sum=6.858 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=9216`, `chunked_req_prefix_len=1024`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [9216, 128, 192], [9216, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=1024`, `sum_seq_after=9216`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 8192, 'prefix_len': 1024, 'seq_len_after': 9216, 'prompt_len': 9216, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.311 ms
  纯GPU kernel时间: `0.267 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.311 ms`, `host_to_first_kernel_gap=24.448245`, `host_end_to_last_kernel_tail=24.407 ms`, `gpu_makespan=0.270 ms`, `gpu_kernel_sum=0.267 ms`
  开始时间(ns): `53704888`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.071 ms
  纯GPU kernel时间: `0.017 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.071 ms`, `host_to_first_kernel_gap=24.339731`, `host_end_to_last_kernel_tail=24.286 ms`, `gpu_makespan=0.017 ms`, `gpu_kernel_sum=0.017 ms`
  开始时间(ns): `54083770`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.294 ms
  纯GPU kernel时间: `0.439 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.294 ms`, `host_to_first_kernel_gap=24.236925`, `host_end_to_last_kernel_tail=24.384 ms`, `gpu_makespan=0.441 ms`, `gpu_kernel_sum=0.439 ms`
  开始时间(ns): `54205328`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.061 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.061 ms`, `host_to_first_kernel_gap=24.294413`, `host_end_to_last_kernel_tail=24.246 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `54589120`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.103 ms
  纯GPU kernel时间: `0.106 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.103 ms`, `host_to_first_kernel_gap=24.189935`, `host_end_to_last_kernel_tail=24.193 ms`, `gpu_makespan=0.106 ms`, `gpu_kernel_sum=0.106 ms`
  开始时间(ns): `54707518`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.332 ms
  纯GPU kernel时间: `0.325 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.332 ms`, `host_to_first_kernel_gap=23.926754`, `host_end_to_last_kernel_tail=23.920 ms`, `gpu_makespan=0.326 ms`, `gpu_kernel_sum=0.325 ms`
  开始时间(ns): `55113739`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[9216, 512]]}`
- `attn_mha` -> 0.200 ms
  纯GPU kernel时间: `4.440 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.200 ms`, `host_to_first_kernel_gap=24.05193`, `host_end_to_last_kernel_tail=28.292 ms`, `gpu_makespan=4.440 ms`, `gpu_kernel_sum=4.440 ms`
  开始时间(ns): `55587859`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [9216, 128, 192], [9216, 128, 128]]}`
- `o_proj` -> 0.341 ms
  纯GPU kernel时间: `1.251 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.341 ms`, `host_to_first_kernel_gap=28.236401`, `host_end_to_last_kernel_tail=29.148 ms`, `gpu_makespan=1.252 ms`, `gpu_kernel_sum=1.251 ms`
  开始时间(ns): `55844507`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 4 / prefill / instance 1

- 执行序号: `5`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `2.792 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.792 ms`, `module_to_last_kernel=45.156 ms`, `host_to_first_kernel_gap=37.804957`, `host_end_to_last_kernel_tail=42.364 ms`, `gpu_makespan=7.351 ms`, `gpu_kernel_sum=7.025 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=9216`, `chunked_req_prefix_len=1024`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [9216, 128, 192], [9216, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=1024`, `sum_seq_after=9216`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 8192, 'prefix_len': 1024, 'seq_len_after': 9216, 'prompt_len': 9216, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.394 ms
  纯GPU kernel时间: `0.284 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.394 ms`, `host_to_first_kernel_gap=37.635139`, `host_end_to_last_kernel_tail=37.528 ms`, `gpu_makespan=0.286 ms`, `gpu_kernel_sum=0.284 ms`
  开始时间(ns): `58309798`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.070 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.070 ms`, `host_to_first_kernel_gap=37.453907`, `host_end_to_last_kernel_tail=37.402 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `58778038`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.329 ms
  纯GPU kernel时间: `0.468 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.329 ms`, `host_to_first_kernel_gap=37.358536`, `host_end_to_last_kernel_tail=37.500 ms`, `gpu_makespan=0.470 ms`, `gpu_kernel_sum=0.468 ms`
  开始时间(ns): `58893761`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.061 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.061 ms`, `host_to_first_kernel_gap=37.407066`, `host_end_to_last_kernel_tail=37.360 ms`, `gpu_makespan=0.014 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `59315630`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.110 ms
  纯GPU kernel时间: `0.106 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.110 ms`, `host_to_first_kernel_gap=37.30337`, `host_end_to_last_kernel_tail=37.300 ms`, `gpu_makespan=0.106 ms`, `gpu_kernel_sum=0.106 ms`
  开始时间(ns): `59434142`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.335 ms
  纯GPU kernel时间: `0.331 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.335 ms`, `host_to_first_kernel_gap=37.043912`, `host_end_to_last_kernel_tail=37.042 ms`, `gpu_makespan=0.333 ms`, `gpu_kernel_sum=0.331 ms`
  开始时间(ns): `59838944`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[9216, 512]]}`
- `attn_mha` -> 0.199 ms
  纯GPU kernel时间: `4.451 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.199 ms`, `host_to_first_kernel_gap=37.173081`, `host_end_to_last_kernel_tail=41.425 ms`, `gpu_makespan=4.451 ms`, `gpu_kernel_sum=4.451 ms`
  开始时间(ns): `60315855`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [9216, 128, 192], [9216, 128, 128]]}`
- `o_proj` -> 0.341 ms
  纯GPU kernel时间: `1.352 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.341 ms`, `host_to_first_kernel_gap=41.369461`, `host_end_to_last_kernel_tail=42.383 ms`, `gpu_makespan=1.354 ms`, `gpu_kernel_sum=1.352 ms`
  开始时间(ns): `60572018`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 0 / decode / instance 1

- 执行序号: `6`
- Collector对齐边界: `{'Module': 'model.model.layers.0.self_attn'}`
- 整块 MLA-module 时长: `2.700 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.700 ms`, `module_to_last_kernel=48.444 ms`, `host_to_first_kernel_gap=48.290843`, `host_end_to_last_kernel_tail=45.744 ms`, `gpu_makespan=0.154 ms`, `gpu_kernel_sum=0.124 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.439 ms
  纯GPU kernel时间: `0.016 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.439 ms`, `host_to_first_kernel_gap=48.188818`, `host_end_to_last_kernel_tail=47.766 ms`, `gpu_makespan=0.017 ms`, `gpu_kernel_sum=0.016 ms`
  开始时间(ns): `66441202`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.065 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.065 ms`, `host_to_first_kernel_gap=47.675653`, `host_end_to_last_kernel_tail=47.613 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `66970975`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.047 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.047 ms`, `host_to_first_kernel_gap=47.576276`, `host_end_to_last_kernel_tail=47.531 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `67072272`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.354 ms
  纯GPU kernel时间: `0.020 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.354 ms`, `host_to_first_kernel_gap=47.469727`, `host_end_to_last_kernel_tail=47.136 ms`, `gpu_makespan=0.021 ms`, `gpu_kernel_sum=0.020 ms`
  开始时间(ns): `67182533`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.122 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.122 ms`, `host_to_first_kernel_gap=46.896525`, `host_end_to_last_kernel_tail=46.777 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `67787223`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.448 ms
  纯GPU kernel时间: `0.032 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.448 ms`, `host_to_first_kernel_gap=46.705574`, `host_end_to_last_kernel_tail=46.291 ms`, `gpu_makespan=0.033 ms`, `gpu_kernel_sum=0.032 ms`
  开始时间(ns): `67981182`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.397 ms
  纯GPU kernel时间: `0.051 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.397 ms`, `host_to_first_kernel_gap=46.114941`, `host_end_to_last_kernel_tail=45.770 ms`, `gpu_makespan=0.052 ms`, `gpu_kernel_sum=0.051 ms`
  开始时间(ns): `68616711`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 1 / decode / instance 1

- 执行序号: `7`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `2.351 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.351 ms`, `module_to_last_kernel=44.814 ms`, `host_to_first_kernel_gap=44.660141`, `host_end_to_last_kernel_tail=42.463 ms`, `gpu_makespan=0.154 ms`, `gpu_kernel_sum=0.124 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.311 ms
  纯GPU kernel时间: `0.016 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.311 ms`, `host_to_first_kernel_gap=44.568837`, `host_end_to_last_kernel_tail=44.275 ms`, `gpu_makespan=0.017 ms`, `gpu_kernel_sum=0.016 ms`
  开始时间(ns): `70376927`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.063 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.063 ms`, `host_to_first_kernel_gap=44.192763`, `host_end_to_last_kernel_tail=44.132 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `70769513`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.045 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.045 ms`, `host_to_first_kernel_gap=44.095647`, `host_end_to_last_kernel_tail=44.053 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `70868997`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.304 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.304 ms`, `host_to_first_kernel_gap=44.003573`, `host_end_to_last_kernel_tail=43.721 ms`, `gpu_makespan=0.021 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `70964143`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.118 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.118 ms`, `host_to_first_kernel_gap=43.53536`, `host_end_to_last_kernel_tail=43.420 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `71464580`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.391 ms
  纯GPU kernel时间: `0.032 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.391 ms`, `host_to_first_kernel_gap=43.357781`, `host_end_to_last_kernel_tail=43.001 ms`, `gpu_makespan=0.034 ms`, `gpu_kernel_sum=0.032 ms`
  开始时间(ns): `71645039`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.386 ms
  纯GPU kernel时间: `0.050 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.386 ms`, `host_to_first_kernel_gap=42.823029`, `host_end_to_last_kernel_tail=42.489 ms`, `gpu_makespan=0.051 ms`, `gpu_kernel_sum=0.050 ms`
  开始时间(ns): `72225519`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 2 / decode / instance 1

- 执行序号: `8`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `2.136 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.136 ms`, `module_to_last_kernel=41.633 ms`, `host_to_first_kernel_gap=41.482601`, `host_end_to_last_kernel_tail=39.496 ms`, `gpu_makespan=0.150 ms`, `gpu_kernel_sum=0.121 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.282 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.282 ms`, `host_to_first_kernel_gap=41.39744`, `host_end_to_last_kernel_tail=41.130 ms`, `gpu_makespan=0.015 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `73863428`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.060 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.060 ms`, `host_to_first_kernel_gap=41.039784`, `host_end_to_last_kernel_tail=40.982 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `74235868`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.043 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.043 ms`, `host_to_first_kernel_gap=40.94772`, `host_end_to_last_kernel_tail=40.906 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `74330140`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.276 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.276 ms`, `host_to_first_kernel_gap=40.860302`, `host_end_to_last_kernel_tail=40.604 ms`, `gpu_makespan=0.020 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `74420598`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.094 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.094 ms`, `host_to_first_kernel_gap=40.437932`, `host_end_to_last_kernel_tail=40.346 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `74874296`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.373 ms
  纯GPU kernel时间: `0.032 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.373 ms`, `host_to_first_kernel_gap=40.287992`, `host_end_to_last_kernel_tail=39.948 ms`, `gpu_makespan=0.034 ms`, `gpu_kernel_sum=0.032 ms`
  开始时间(ns): `75027116`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.341 ms
  纯GPU kernel时间: `0.050 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.341 ms`, `host_to_first_kernel_gap=39.807798`, `host_end_to_last_kernel_tail=39.518 ms`, `gpu_makespan=0.051 ms`, `gpu_kernel_sum=0.050 ms`
  开始时间(ns): `75552014`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 3 / decode / instance 1

- 执行序号: `9`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `2.048 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.048 ms`, `module_to_last_kernel=38.759 ms`, `host_to_first_kernel_gap=38.605539`, `host_end_to_last_kernel_tail=36.711 ms`, `gpu_makespan=0.154 ms`, `gpu_kernel_sum=0.123 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.279 ms
  纯GPU kernel时间: `0.016 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.279 ms`, `host_to_first_kernel_gap=38.532269`, `host_end_to_last_kernel_tail=38.270 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.016 ms`
  开始时间(ns): `77039031`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.056 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.056 ms`, `host_to_first_kernel_gap=38.189069`, `host_end_to_last_kernel_tail=38.135 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `77399735`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.040 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.040 ms`, `host_to_first_kernel_gap=38.10368`, `host_end_to_last_kernel_tail=38.066 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `77487300`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.259 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.259 ms`, `host_to_first_kernel_gap=38.022465`, `host_end_to_last_kernel_tail=37.783 ms`, `gpu_makespan=0.019 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `77571587`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.093 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.093 ms`, `host_to_first_kernel_gap=37.622035`, `host_end_to_last_kernel_tail=37.531 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `78002065`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.352 ms
  纯GPU kernel时间: `0.033 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.352 ms`, `host_to_first_kernel_gap=37.476758`, `host_end_to_last_kernel_tail=37.160 ms`, `gpu_makespan=0.035 ms`, `gpu_kernel_sum=0.033 ms`
  开始时间(ns): `78150318`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.325 ms
  纯GPU kernel时间: `0.049 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.325 ms`, `host_to_first_kernel_gap=37.005375`, `host_end_to_last_kernel_tail=36.731 ms`, `gpu_makespan=0.051 ms`, `gpu_kernel_sum=0.049 ms`
  开始时间(ns): `78668997`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 4 / decode / instance 1

- 执行序号: `10`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `2.074 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.074 ms`, `module_to_last_kernel=35.395 ms`, `host_to_first_kernel_gap=35.24113`, `host_end_to_last_kernel_tail=33.321 ms`, `gpu_makespan=0.154 ms`, `gpu_kernel_sum=0.123 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.316 ms
  纯GPU kernel时间: `0.016 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.316 ms`, `host_to_first_kernel_gap=35.167187`, `host_end_to_last_kernel_tail=34.868 ms`, `gpu_makespan=0.017 ms`, `gpu_kernel_sum=0.016 ms`
  开始时间(ns): `80760465`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.068 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.068 ms`, `host_to_first_kernel_gap=34.79089`, `host_end_to_last_kernel_tail=34.725 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `81153114`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.040 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.040 ms`, `host_to_first_kernel_gap=34.69212`, `host_end_to_last_kernel_tail=34.655 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `81254188`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.277 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.277 ms`, `host_to_first_kernel_gap=34.609581`, `host_end_to_last_kernel_tail=34.351 ms`, `gpu_makespan=0.019 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `81339895`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.094 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.094 ms`, `host_to_first_kernel_gap=34.184499`, `host_end_to_last_kernel_tail=34.092 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `81796881`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.339 ms
  纯GPU kernel时间: `0.032 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.339 ms`, `host_to_first_kernel_gap=34.040116`, `host_end_to_last_kernel_tail=33.735 ms`, `gpu_makespan=0.034 ms`, `gpu_kernel_sum=0.032 ms`
  开始时间(ns): `81943952`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.306 ms
  纯GPU kernel时间: `0.051 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.306 ms`, `host_to_first_kernel_gap=33.593667`, `host_end_to_last_kernel_tail=33.340 ms`, `gpu_makespan=0.053 ms`, `gpu_kernel_sum=0.051 ms`
  开始时间(ns): `82435297`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`
