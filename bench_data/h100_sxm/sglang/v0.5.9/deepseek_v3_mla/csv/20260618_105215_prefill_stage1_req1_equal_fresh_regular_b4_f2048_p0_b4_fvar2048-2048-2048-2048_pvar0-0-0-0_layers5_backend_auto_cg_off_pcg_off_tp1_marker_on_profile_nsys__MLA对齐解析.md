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
- 整块 MLA-module 时长: `1.905 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.905 ms`, `module_to_last_kernel=4.334 ms`, `host_to_first_kernel_gap=0.29196`, `host_end_to_last_kernel_tail=2.429 ms`, `gpu_makespan=4.042 ms`, `gpu_kernel_sum=3.524 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=8192`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=0`, `sum_seq_after=8192`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.376 ms
  纯GPU kernel时间: `0.274 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.376 ms`, `host_to_first_kernel_gap=0.117058`, `host_end_to_last_kernel_tail=0.176 ms`, `gpu_makespan=0.435 ms`, `gpu_kernel_sum=0.274 ms`
  开始时间(ns): `30702483`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.044 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.044 ms`, `host_to_first_kernel_gap=0.124556`, `host_end_to_last_kernel_tail=0.099 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `31129961`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.189 ms
  纯GPU kernel时间: `0.447 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.189 ms`, `host_to_first_kernel_gap=0.073821`, `host_end_to_last_kernel_tail=0.413 ms`, `gpu_makespan=0.527 ms`, `gpu_kernel_sum=0.447 ms`
  开始时间(ns): `31199608`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.030 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.030 ms`, `host_to_first_kernel_gap=0.365043`, `host_end_to_last_kernel_tail=0.348 ms`, `gpu_makespan=0.014 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `31435586`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.104 ms
  纯GPU kernel时间: `0.106 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.104 ms`, `host_to_first_kernel_gap=0.317555`, `host_end_to_last_kernel_tail=0.320 ms`, `gpu_makespan=0.106 ms`, `gpu_kernel_sum=0.106 ms`
  开始时间(ns): `31497986`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.189 ms
  纯GPU kernel时间: `0.288 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.189 ms`, `host_to_first_kernel_gap=0.207464`, `host_end_to_last_kernel_tail=0.308 ms`, `gpu_makespan=0.289 ms`, `gpu_kernel_sum=0.288 ms`
  开始时间(ns): `31742765`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[8192, 512]]}`
- `attn_mha` -> 0.162 ms
  纯GPU kernel时间: `1.119 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.162 ms`, `host_to_first_kernel_gap=0.443513`, `host_end_to_last_kernel_tail=1.400 ms`, `gpu_makespan=1.119 ms`, `gpu_kernel_sum=1.119 ms`
  开始时间(ns): `32037276`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]}`
- `o_proj` -> 0.180 ms
  纯GPU kernel时间: `1.259 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.180 ms`, `host_to_first_kernel_gap=1.362698`, `host_end_to_last_kernel_tail=2.443 ms`, `gpu_makespan=1.260 ms`, `gpu_kernel_sum=1.259 ms`
  开始时间(ns): `32238699`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 1 / prefill / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `1.413 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.413 ms`, `module_to_last_kernel=10.232 ms`, `host_to_first_kernel_gap=6.421164`, `host_end_to_last_kernel_tail=8.819 ms`, `gpu_makespan=3.811 ms`, `gpu_kernel_sum=3.523 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=8192`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=0`, `sum_seq_after=8192`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.178 ms
  纯GPU kernel时间: `0.268 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.178 ms`, `host_to_first_kernel_gap=6.291233`, `host_end_to_last_kernel_tail=6.384 ms`, `gpu_makespan=0.271 ms`, `gpu_kernel_sum=0.268 ms`
  开始时间(ns): `33199795`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.040 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.040 ms`, `host_to_first_kernel_gap=6.343928`, `host_end_to_last_kernel_tail=6.322 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `33418236`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.147 ms
  纯GPU kernel时间: `0.448 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.147 ms`, `host_to_first_kernel_gap=6.300498`, `host_end_to_last_kernel_tail=6.604 ms`, `gpu_makespan=0.450 ms`, `gpu_kernel_sum=0.448 ms`
  开始时间(ns): `33480706`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.029 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.029 ms`, `host_to_first_kernel_gap=6.567393`, `host_end_to_last_kernel_tail=6.552 ms`, `gpu_makespan=0.014 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `33664243`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.076 ms
  纯GPU kernel时间: `0.106 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.076 ms`, `host_to_first_kernel_gap=6.525423`, `host_end_to_last_kernel_tail=6.556 ms`, `gpu_makespan=0.106 ms`, `gpu_kernel_sum=0.106 ms`
  开始时间(ns): `33721573`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.167 ms
  纯GPU kernel时间: `0.289 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.167 ms`, `host_to_first_kernel_gap=6.466007`, `host_end_to_last_kernel_tail=6.591 ms`, `gpu_makespan=0.292 ms`, `gpu_kernel_sum=0.289 ms`
  开始时间(ns): `33917053`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[8192, 512]]}`
- `attn_mha` -> 0.126 ms
  纯GPU kernel时间: `1.120 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.126 ms`, `host_to_first_kernel_gap=6.760397`, `host_end_to_last_kernel_tail=7.754 ms`, `gpu_makespan=1.120 ms`, `gpu_kernel_sum=1.120 ms`
  开始时间(ns): `34156838`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]}`
- `o_proj` -> 0.160 ms
  纯GPU kernel时间: `1.260 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.160 ms`, `host_to_first_kernel_gap=7.727737`, `host_end_to_last_kernel_tail=8.830 ms`, `gpu_makespan=1.262 ms`, `gpu_kernel_sum=1.260 ms`
  开始时间(ns): `34312410`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 2 / prefill / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.358 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.358 ms`, `module_to_last_kernel=16.670 ms`, `host_to_first_kernel_gap=12.881238`, `host_end_to_last_kernel_tail=15.313 ms`, `gpu_makespan=3.789 ms`, `gpu_kernel_sum=3.501 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=8192`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=0`, `sum_seq_after=8192`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.162 ms
  纯GPU kernel时间: `0.267 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.162 ms`, `host_to_first_kernel_gap=12.770558`, `host_end_to_last_kernel_tail=12.877 ms`, `gpu_makespan=0.269 ms`, `gpu_kernel_sum=0.267 ms`
  开始时间(ns): `35158708`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.037 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.037 ms`, `host_to_first_kernel_gap=12.838718`, `host_end_to_last_kernel_tail=12.820 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `35358996`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.142 ms
  纯GPU kernel时间: `0.442 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.142 ms`, `host_to_first_kernel_gap=12.796114`, `host_end_to_last_kernel_tail=13.097 ms`, `gpu_makespan=0.444 ms`, `gpu_kernel_sum=0.442 ms`
  开始时间(ns): `35422240`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.029 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.029 ms`, `host_to_first_kernel_gap=13.059964`, `host_end_to_last_kernel_tail=13.044 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `35602102`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.068 ms
  纯GPU kernel时间: `0.107 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.068 ms`, `host_to_first_kernel_gap=13.007107`, `host_end_to_last_kernel_tail=13.046 ms`, `gpu_makespan=0.107 ms`, `gpu_kernel_sum=0.107 ms`
  开始时间(ns): `35669103`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.169 ms
  纯GPU kernel时间: `0.287 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.169 ms`, `host_to_first_kernel_gap=12.959784`, `host_end_to_last_kernel_tail=13.079 ms`, `gpu_makespan=0.288 ms`, `gpu_kernel_sum=0.287 ms`
  开始时间(ns): `35851818`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[8192, 512]]}`
- `attn_mha` -> 0.119 ms
  纯GPU kernel时间: `1.122 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.119 ms`, `host_to_first_kernel_gap=13.255404`, `host_end_to_last_kernel_tail=14.259 ms`, `gpu_makespan=1.122 ms`, `gpu_kernel_sum=1.122 ms`
  开始时间(ns): `36090982`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]}`
- `o_proj` -> 0.158 ms
  纯GPU kernel时间: `1.246 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.158 ms`, `host_to_first_kernel_gap=14.232993`, `host_end_to_last_kernel_tail=15.322 ms`, `gpu_makespan=1.247 ms`, `gpu_kernel_sum=1.246 ms`
  开始时间(ns): `36237969`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 3 / prefill / instance 1

- 执行序号: `4`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `1.299 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.299 ms`, `module_to_last_kernel=23.153 ms`, `host_to_first_kernel_gap=19.370133`, `host_end_to_last_kernel_tail=21.854 ms`, `gpu_makespan=3.783 ms`, `gpu_kernel_sum=3.499 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=8192`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=0`, `sum_seq_after=8192`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.155 ms
  纯GPU kernel时间: `0.266 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.155 ms`, `host_to_first_kernel_gap=19.242324`, `host_end_to_last_kernel_tail=19.355 ms`, `gpu_makespan=0.267 ms`, `gpu_kernel_sum=0.266 ms`
  开始时间(ns): `37082525`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.041 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.041 ms`, `host_to_first_kernel_gap=19.317599`, `host_end_to_last_kernel_tail=19.294 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `37274802`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.131 ms
  纯GPU kernel时间: `0.435 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.131 ms`, `host_to_first_kernel_gap=19.274025`, `host_end_to_last_kernel_tail=19.580 ms`, `gpu_makespan=0.437 ms`, `gpu_kernel_sum=0.435 ms`
  开始时间(ns): `37337544`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.033 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.033 ms`, `host_to_first_kernel_gap=19.542257`, `host_end_to_last_kernel_tail=19.523 ms`, `gpu_makespan=0.014 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `37506304`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.066 ms
  纯GPU kernel时间: `0.106 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.066 ms`, `host_to_first_kernel_gap=19.499325`, `host_end_to_last_kernel_tail=19.539 ms`, `gpu_makespan=0.106 ms`, `gpu_kernel_sum=0.106 ms`
  开始时间(ns): `37565684`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.140 ms
  纯GPU kernel时间: `0.290 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.140 ms`, `host_to_first_kernel_gap=19.460041`, `host_end_to_last_kernel_tail=19.610 ms`, `gpu_makespan=0.291 ms`, `gpu_kernel_sum=0.290 ms`
  开始时间(ns): `37739560`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[8192, 512]]}`
- `attn_mha` -> 0.126 ms
  纯GPU kernel时间: `1.127 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.126 ms`, `host_to_first_kernel_gap=19.786433`, `host_end_to_last_kernel_tail=20.788 ms`, `gpu_makespan=1.127 ms`, `gpu_kernel_sum=1.127 ms`
  开始时间(ns): `37945967`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]}`
- `o_proj` -> 0.146 ms
  纯GPU kernel时间: `1.243 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.146 ms`, `host_to_first_kernel_gap=20.763694`, `host_end_to_last_kernel_tail=21.863 ms`, `gpu_makespan=1.245 ms`, `gpu_kernel_sum=1.243 ms`
  开始时间(ns): `38098914`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 4 / prefill / instance 1

- 执行序号: `5`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `1.407 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.407 ms`, `module_to_last_kernel=34.919 ms`, `host_to_first_kernel_gap=31.136947`, `host_end_to_last_kernel_tail=33.512 ms`, `gpu_makespan=3.782 ms`, `gpu_kernel_sum=3.495 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=8192`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=0`, `sum_seq_after=8192`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.230 ms
  纯GPU kernel时间: `0.265 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.230 ms`, `host_to_first_kernel_gap=31.012243`, `host_end_to_last_kernel_tail=31.050 ms`, `gpu_makespan=0.268 ms`, `gpu_kernel_sum=0.265 ms`
  开始时间(ns): `39579867`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.041 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.041 ms`, `host_to_first_kernel_gap=31.011884`, `host_end_to_last_kernel_tail=30.989 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `39848642`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.162 ms
  纯GPU kernel时间: `0.436 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.162 ms`, `host_to_first_kernel_gap=30.969671`, `host_end_to_last_kernel_tail=31.245 ms`, `gpu_makespan=0.437 ms`, `gpu_kernel_sum=0.436 ms`
  开始时间(ns): `39910279`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.032 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.032 ms`, `host_to_first_kernel_gap=31.206714`, `host_end_to_last_kernel_tail=31.188 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `40109972`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.073 ms
  纯GPU kernel时间: `0.107 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.073 ms`, `host_to_first_kernel_gap=31.162789`, `host_end_to_last_kernel_tail=31.197 ms`, `gpu_makespan=0.107 ms`, `gpu_kernel_sum=0.107 ms`
  开始时间(ns): `40167977`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.148 ms
  纯GPU kernel时间: `0.284 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.148 ms`, `host_to_first_kernel_gap=31.124417`, `host_end_to_last_kernel_tail=31.263 ms`, `gpu_makespan=0.286 ms`, `gpu_kernel_sum=0.284 ms`
  开始时间(ns): `40343117`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[8192, 512]]}`
- `attn_mha` -> 0.117 ms
  纯GPU kernel时间: `1.125 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.117 ms`, `host_to_first_kernel_gap=31.436027`, `host_end_to_last_kernel_tail=32.444 ms`, `gpu_makespan=1.124 ms`, `gpu_kernel_sum=1.125 ms`
  开始时间(ns): `40560851`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]}`
- `o_proj` -> 0.149 ms
  纯GPU kernel时间: `1.248 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.149 ms`, `host_to_first_kernel_gap=32.419781`, `host_end_to_last_kernel_tail=33.521 ms`, `gpu_makespan=1.250 ms`, `gpu_kernel_sum=1.248 ms`
  开始时间(ns): `40704521`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 0 / decode / instance 1

- 执行序号: `6`
- Collector对齐边界: `{'Module': 'model.model.layers.0.self_attn'}`
- 整块 MLA-module 时长: `1.584 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.584 ms`, `module_to_last_kernel=41.359 ms`, `host_to_first_kernel_gap=41.20548`, `host_end_to_last_kernel_tail=39.775 ms`, `gpu_makespan=0.154 ms`, `gpu_kernel_sum=0.120 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.276 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.276 ms`, `host_to_first_kernel_gap=41.142502`, `host_end_to_last_kernel_tail=40.884 ms`, `gpu_makespan=0.017 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `44519365`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4, 7168]]}`
- `q_a_layernorm` -> 0.038 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.038 ms`, `host_to_first_kernel_gap=40.835398`, `host_end_to_last_kernel_tail=40.800 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `44843013`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4, 1536]]}`
- `kv_a_layernorm` -> 0.022 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.022 ms`, `host_to_first_kernel_gap=40.782156`, `host_end_to_last_kernel_tail=40.762 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `44898143`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4, 512]]}`
- `q_b_proj` -> 0.180 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.180 ms`, `host_to_first_kernel_gap=40.73682`, `host_end_to_last_kernel_tail=40.578 ms`, `gpu_makespan=0.022 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `44946775`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4, 1536]]}`
- `rotary_emb` -> 0.080 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.080 ms`, `host_to_first_kernel_gap=40.436788`, `host_end_to_last_kernel_tail=40.359 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `45281687`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4], [4, 128, 64], [4, 1, 64]]}`
- `attn_mqa` -> 0.265 ms
  纯GPU kernel时间: `0.029 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.265 ms`, `host_to_first_kernel_gap=40.324096`, `host_end_to_last_kernel_tail=40.090 ms`, `gpu_makespan=0.030 ms`, `gpu_kernel_sum=0.029 ms`
  开始时间(ns): `45397547`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4, 128, 512], [4, 1, 512], [4, 1, 512]]}`
- `o_proj` -> 0.249 ms
  纯GPU kernel时间: `0.050 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.249 ms`, `host_to_first_kernel_gap=39.987573`, `host_end_to_last_kernel_tail=39.790 ms`, `gpu_makespan=0.051 ms`, `gpu_kernel_sum=0.050 ms`
  开始时间(ns): `45777014`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4, 16384]]}`

## Layer 1 / decode / instance 1

- 执行序号: `7`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `1.258 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.258 ms`, `module_to_last_kernel=39.508 ms`, `host_to_first_kernel_gap=39.35219`, `host_end_to_last_kernel_tail=38.249 ms`, `gpu_makespan=0.155 ms`, `gpu_kernel_sum=0.123 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.135 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.135 ms`, `host_to_first_kernel_gap=39.298742`, `host_end_to_last_kernel_tail=39.182 ms`, `gpu_makespan=0.019 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `46682101`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4, 7168]]}`
- `q_a_layernorm` -> 0.036 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.036 ms`, `host_to_first_kernel_gap=39.137998`, `host_end_to_last_kernel_tail=39.104 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `46861309`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4, 1536]]}`
- `kv_a_layernorm` -> 0.022 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.022 ms`, `host_to_first_kernel_gap=39.089237`, `host_end_to_last_kernel_tail=39.069 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `46912150`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4, 512]]}`
- `q_b_proj` -> 0.143 ms
  纯GPU kernel时间: `0.021 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.143 ms`, `host_to_first_kernel_gap=39.048582`, `host_end_to_last_kernel_tail=38.928 ms`, `gpu_makespan=0.022 ms`, `gpu_kernel_sum=0.021 ms`
  开始时间(ns): `46955941`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4, 1536]]}`
- `rotary_emb` -> 0.071 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.071 ms`, `host_to_first_kernel_gap=38.811603`, `host_end_to_last_kernel_tail=38.742 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `47228184`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4], [4, 128, 64], [4, 1, 64]]}`
- `attn_mqa` -> 0.230 ms
  纯GPU kernel时间: `0.029 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.230 ms`, `host_to_first_kernel_gap=38.710534`, `host_end_to_last_kernel_tail=38.511 ms`, `gpu_makespan=0.030 ms`, `gpu_kernel_sum=0.029 ms`
  开始时间(ns): `47332261`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4, 128, 512], [4, 1, 512], [4, 1, 512]]}`
- `o_proj` -> 0.207 ms
  纯GPU kernel时间: `0.049 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.207 ms`, `host_to_first_kernel_gap=38.419012`, `host_end_to_last_kernel_tail=38.263 ms`, `gpu_makespan=0.050 ms`, `gpu_kernel_sum=0.049 ms`
  开始时间(ns): `47666695`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4, 16384]]}`

## Layer 2 / decode / instance 1

- 执行序号: `8`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.239 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.239 ms`, `module_to_last_kernel=37.986 ms`, `host_to_first_kernel_gap=37.836058`, `host_end_to_last_kernel_tail=36.746 ms`, `gpu_makespan=0.150 ms`, `gpu_kernel_sum=0.117 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.127 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.127 ms`, `host_to_first_kernel_gap=37.782279`, `host_end_to_last_kernel_tail=37.672 ms`, `gpu_makespan=0.017 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `48517156`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4, 7168]]}`
- `q_a_layernorm` -> 0.035 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.035 ms`, `host_to_first_kernel_gap=37.628998`, `host_end_to_last_kernel_tail=37.596 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `48686949`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4, 1536]]}`
- `kv_a_layernorm` -> 0.022 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.022 ms`, `host_to_first_kernel_gap=37.581218`, `host_end_to_last_kernel_tail=37.561 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `48736905`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4, 512]]}`
- `q_b_proj` -> 0.126 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.126 ms`, `host_to_first_kernel_gap=37.541415`, `host_end_to_last_kernel_tail=37.435 ms`, `gpu_makespan=0.020 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `48779876`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4, 1536]]}`
- `rotary_emb` -> 0.066 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.066 ms`, `host_to_first_kernel_gap=37.315288`, `host_end_to_last_kernel_tail=37.251 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `49037587`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4], [4, 128, 64], [4, 1, 64]]}`
- `attn_mqa` -> 0.238 ms
  纯GPU kernel时间: `0.029 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.238 ms`, `host_to_first_kernel_gap=37.220333`, `host_end_to_last_kernel_tail=37.014 ms`, `gpu_makespan=0.032 ms`, `gpu_kernel_sum=0.029 ms`
  开始时间(ns): `49135646`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4, 128, 512], [4, 1, 512], [4, 1, 512]]}`
- `o_proj` -> 0.212 ms
  纯GPU kernel时间: `0.049 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.212 ms`, `host_to_first_kernel_gap=36.921341`, `host_end_to_last_kernel_tail=36.759 ms`, `gpu_makespan=0.050 ms`, `gpu_kernel_sum=0.049 ms`
  开始时间(ns): `49477582`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4, 16384]]}`

## Layer 3 / decode / instance 1

- 执行序号: `9`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `1.204 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.204 ms`, `module_to_last_kernel=36.483 ms`, `host_to_first_kernel_gap=36.334144`, `host_end_to_last_kernel_tail=35.279 ms`, `gpu_makespan=0.148 ms`, `gpu_kernel_sum=0.117 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.130 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.130 ms`, `host_to_first_kernel_gap=36.284964`, `host_end_to_last_kernel_tail=36.170 ms`, `gpu_makespan=0.015 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `50325831`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4, 7168]]}`
- `q_a_layernorm` -> 0.037 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.037 ms`, `host_to_first_kernel_gap=36.125947`, `host_end_to_last_kernel_tail=36.091 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `50499984`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4, 1536]]}`
- `kv_a_layernorm` -> 0.021 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.021 ms`, `host_to_first_kernel_gap=36.076554`, `host_end_to_last_kernel_tail=36.057 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `50551553`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4, 512]]}`
- `q_b_proj` -> 0.128 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.128 ms`, `host_to_first_kernel_gap=36.037571`, `host_end_to_last_kernel_tail=35.929 ms`, `gpu_makespan=0.020 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `50594792`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4, 1536]]}`
- `rotary_emb` -> 0.067 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.067 ms`, `host_to_first_kernel_gap=35.82496`, `host_end_to_last_kernel_tail=35.760 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `50837739`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4], [4, 128, 64], [4, 1, 64]]}`
- `attn_mqa` -> 0.244 ms
  纯GPU kernel时间: `0.030 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.244 ms`, `host_to_first_kernel_gap=35.729778`, `host_end_to_last_kernel_tail=35.517 ms`, `gpu_makespan=0.031 ms`, `gpu_kernel_sum=0.030 ms`
  开始时间(ns): `50935865`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4, 128, 512], [4, 1, 512], [4, 1, 512]]}`
- `o_proj` -> 0.193 ms
  纯GPU kernel时间: `0.049 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.193 ms`, `host_to_first_kernel_gap=35.435041`, `host_end_to_last_kernel_tail=35.292 ms`, `gpu_makespan=0.050 ms`, `gpu_kernel_sum=0.049 ms`
  开始时间(ns): `51273866`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4, 16384]]}`

## Layer 4 / decode / instance 1

- 执行序号: `10`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `1.313 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.313 ms`, `module_to_last_kernel=34.555 ms`, `host_to_first_kernel_gap=34.405543`, `host_end_to_last_kernel_tail=33.242 ms`, `gpu_makespan=0.150 ms`, `gpu_kernel_sum=0.120 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.198 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.198 ms`, `host_to_first_kernel_gap=34.351796`, `host_end_to_last_kernel_tail=34.170 ms`, `gpu_makespan=0.016 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `52625431`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4, 7168]]}`
- `q_a_layernorm` -> 0.039 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.039 ms`, `host_to_first_kernel_gap=34.124778`, `host_end_to_last_kernel_tail=34.088 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `52868481`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4, 1536]]}`
- `kv_a_layernorm` -> 0.021 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.021 ms`, `host_to_first_kernel_gap=34.072754`, `host_end_to_last_kernel_tail=34.054 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `52922585`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4, 512]]}`
- `q_b_proj` -> 0.161 ms
  纯GPU kernel时间: `0.020 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.161 ms`, `host_to_first_kernel_gap=34.033213`, `host_end_to_last_kernel_tail=33.893 ms`, `gpu_makespan=0.021 ms`, `gpu_kernel_sum=0.020 ms`
  开始时间(ns): `52965294`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4, 1536]]}`
- `rotary_emb` -> 0.067 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.067 ms`, `host_to_first_kernel_gap=33.77145`, `host_end_to_last_kernel_tail=33.706 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `53259025`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4], [4, 128, 64], [4, 1, 64]]}`
- `attn_mqa` -> 0.225 ms
  纯GPU kernel时间: `0.029 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.225 ms`, `host_to_first_kernel_gap=33.675878`, `host_end_to_last_kernel_tail=33.482 ms`, `gpu_makespan=0.030 ms`, `gpu_kernel_sum=0.029 ms`
  开始时间(ns): `53358853`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4, 128, 512], [4, 1, 512], [4, 1, 512]]}`
- `o_proj` -> 0.192 ms
  纯GPU kernel时间: `0.050 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.192 ms`, `host_to_first_kernel_gap=33.394637`, `host_end_to_last_kernel_tail=33.254 ms`, `gpu_makespan=0.051 ms`, `gpu_kernel_sum=0.050 ms`
  开始时间(ns): `53681374`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4, 16384]]}`
