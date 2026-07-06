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
- 整块 MLA-module 时长: `2.250 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.250 ms`, `module_to_last_kernel=2.541 ms`, `host_to_first_kernel_gap=0.314594`, `host_end_to_last_kernel_tail=0.291 ms`, `gpu_makespan=2.227 ms`, `gpu_kernel_sum=0.820 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=2048`, `total_tokens=2048`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[2048, 128, 192], [2048, 128, 192], [2048, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=2048`, `sum_prefix=0`, `sum_seq_after=2048`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 512, 'prefix_len': 0, 'seq_len_after': 512, 'prompt_len': 512, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 512, 'prefix_len': 0, 'seq_len_after': 512, 'prompt_len': 512, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 512, 'prefix_len': 0, 'seq_len_after': 512, 'prompt_len': 512, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 512, 'prefix_len': 0, 'seq_len_after': 512, 'prompt_len': 512, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.404 ms
  纯GPU kernel时间: `0.072 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.404 ms`, `host_to_first_kernel_gap=0.135928`, `host_end_to_last_kernel_tail=0.019 ms`, `gpu_makespan=0.288 ms`, `gpu_kernel_sum=0.072 ms`
  开始时间(ns): `28156912`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2048, 7168]]}`
- `q_a_layernorm` -> 0.049 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.049 ms`, `host_to_first_kernel_gap=0.042896`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `28622456`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2048, 1536]]}`
- `q_b_proj` -> 0.260 ms
  纯GPU kernel时间: `0.122 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.260 ms`, `host_to_first_kernel_gap=0.095184`, `host_end_to_last_kernel_tail=0.090 ms`, `gpu_makespan=0.255 ms`, `gpu_kernel_sum=0.122 ms`
  开始时间(ns): `28705144`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2048, 1536]]}`
- `kv_a_layernorm` -> 0.042 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.042 ms`, `host_to_first_kernel_gap=0.037034`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `29032062`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2048, 512]]}`
- `rotary_emb` -> 0.103 ms
  纯GPU kernel时间: `0.027 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.103 ms`, `host_to_first_kernel_gap=0.088468`, `host_end_to_last_kernel_tail=0.013 ms`, `gpu_makespan=0.027 ms`, `gpu_kernel_sum=0.027 ms`
  开始时间(ns): `29117876`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2048], [2048, 128, 64], [2048, 1, 64]]}`
- `kv_b_proj` -> 0.242 ms
  纯GPU kernel时间: `0.072 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.242 ms`, `host_to_first_kernel_gap=0.088187`, `host_end_to_last_kernel_tail=0.041 ms`, `gpu_makespan=0.195 ms`, `gpu_kernel_sum=0.072 ms`
  开始时间(ns): `29374285`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[2048, 512]]}`
- `attn_mha` -> 0.182 ms
  纯GPU kernel时间: `0.142 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.182 ms`, `host_to_first_kernel_gap=0.135072`, `host_end_to_last_kernel_tail=0.116 ms`, `gpu_makespan=0.163 ms`, `gpu_kernel_sum=0.142 ms`
  开始时间(ns): `29726632`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2048, 128, 192], [2048, 128, 192], [2048, 128, 128]]}`
- `o_proj` -> 0.260 ms
  纯GPU kernel时间: `0.374 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.260 ms`, `host_to_first_kernel_gap=0.098797`, `host_end_to_last_kernel_tail=0.305 ms`, `gpu_makespan=0.465 ms`, `gpu_kernel_sum=0.374 ms`
  开始时间(ns): `29955387`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2048, 16384]]}`

## Layer 1 / prefill / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `1.809 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.809 ms`, `module_to_last_kernel=2.105 ms`, `host_to_first_kernel_gap=0.650333`, `host_end_to_last_kernel_tail=0.297 ms`, `gpu_makespan=1.455 ms`, `gpu_kernel_sum=0.820 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=2048`, `total_tokens=2048`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[2048, 128, 192], [2048, 128, 192], [2048, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=2048`, `sum_prefix=0`, `sum_seq_after=2048`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 512, 'prefix_len': 0, 'seq_len_after': 512, 'prompt_len': 512, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 512, 'prefix_len': 0, 'seq_len_after': 512, 'prompt_len': 512, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 512, 'prefix_len': 0, 'seq_len_after': 512, 'prompt_len': 512, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 512, 'prefix_len': 0, 'seq_len_after': 512, 'prompt_len': 512, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.223 ms
  纯GPU kernel时间: `0.073 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.223 ms`, `host_to_first_kernel_gap=0.523262`, `host_end_to_last_kernel_tail=0.375 ms`, `gpu_makespan=0.075 ms`, `gpu_kernel_sum=0.073 ms`
  开始时间(ns): `31238986`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2048, 7168]]}`
- `q_a_layernorm` -> 0.050 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.050 ms`, `host_to_first_kernel_gap=0.32516`, `host_end_to_last_kernel_tail=0.281 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `31511648`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2048, 1536]]}`
- `q_b_proj` -> 0.201 ms
  纯GPU kernel时间: `0.119 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.201 ms`, `host_to_first_kernel_gap=0.253649`, `host_end_to_last_kernel_tail=0.172 ms`, `gpu_makespan=0.120 ms`, `gpu_kernel_sum=0.119 ms`
  开始时间(ns): `31589847`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2048, 1536]]}`
- `kv_a_layernorm` -> 0.042 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.042 ms`, `host_to_first_kernel_gap=0.102597`, `host_end_to_last_kernel_tail=0.066 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `31860387`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2048, 512]]}`
- `rotary_emb` -> 0.078 ms
  纯GPU kernel时间: `0.027 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.078 ms`, `host_to_first_kernel_gap=0.067256`, `host_end_to_last_kernel_tail=0.016 ms`, `gpu_makespan=0.027 ms`, `gpu_kernel_sum=0.027 ms`
  开始时间(ns): `31941392`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2048], [2048, 128, 64], [2048, 1, 64]]}`
- `kv_b_proj` -> 0.215 ms
  纯GPU kernel时间: `0.073 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.215 ms`, `host_to_first_kernel_gap=0.079386`, `host_end_to_last_kernel_tail=0.047 ms`, `gpu_makespan=0.182 ms`, `gpu_kernel_sum=0.073 ms`
  开始时间(ns): `32157422`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[2048, 512]]}`
- `attn_mha` -> 0.145 ms
  纯GPU kernel时间: `0.143 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.145 ms`, `host_to_first_kernel_gap=0.10278`, `host_end_to_last_kernel_tail=0.122 ms`, `gpu_makespan=0.164 ms`, `gpu_kernel_sum=0.143 ms`
  开始时间(ns): `32471660`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2048, 128, 192], [2048, 128, 192], [2048, 128, 128]]}`
- `o_proj` -> 0.248 ms
  纯GPU kernel时间: `0.375 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.248 ms`, `host_to_first_kernel_gap=0.090225`, `host_end_to_last_kernel_tail=0.310 ms`, `gpu_makespan=0.468 ms`, `gpu_kernel_sum=0.375 ms`
  开始时间(ns): `32658679`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2048, 16384]]}`

## Layer 2 / prefill / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.775 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.775 ms`, `module_to_last_kernel=2.072 ms`, `host_to_first_kernel_gap=0.715071`, `host_end_to_last_kernel_tail=0.297 ms`, `gpu_makespan=1.357 ms`, `gpu_kernel_sum=0.818 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=2048`, `total_tokens=2048`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[2048, 128, 192], [2048, 128, 192], [2048, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=2048`, `sum_prefix=0`, `sum_seq_after=2048`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 512, 'prefix_len': 0, 'seq_len_after': 512, 'prompt_len': 512, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 512, 'prefix_len': 0, 'seq_len_after': 512, 'prompt_len': 512, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 512, 'prefix_len': 0, 'seq_len_after': 512, 'prompt_len': 512, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 512, 'prefix_len': 0, 'seq_len_after': 512, 'prompt_len': 512, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.220 ms
  纯GPU kernel时间: `0.073 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.220 ms`, `host_to_first_kernel_gap=0.58987`, `host_end_to_last_kernel_tail=0.444 ms`, `gpu_makespan=0.074 ms`, `gpu_kernel_sum=0.073 ms`
  开始时间(ns): `33855833`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2048, 7168]]}`
- `q_a_layernorm` -> 0.049 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.049 ms`, `host_to_first_kernel_gap=0.394541`, `host_end_to_last_kernel_tail=0.351 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `34125562`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2048, 1536]]}`
- `q_b_proj` -> 0.205 ms
  纯GPU kernel时间: `0.119 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.205 ms`, `host_to_first_kernel_gap=0.323238`, `host_end_to_last_kernel_tail=0.239 ms`, `gpu_makespan=0.121 ms`, `gpu_kernel_sum=0.119 ms`
  开始时间(ns): `34203905`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2048, 1536]]}`
- `kv_a_layernorm` -> 0.049 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.049 ms`, `host_to_first_kernel_gap=0.17943`, `host_end_to_last_kernel_tail=0.136 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `34468257`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2048, 512]]}`
- `rotary_emb` -> 0.075 ms
  纯GPU kernel时间: `0.026 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.075 ms`, `host_to_first_kernel_gap=0.098566`, `host_end_to_last_kernel_tail=0.051 ms`, `gpu_makespan=0.026 ms`, `gpu_kernel_sum=0.026 ms`
  开始时间(ns): `34555617`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2048], [2048, 128, 64], [2048, 1, 64]]}`
- `kv_b_proj` -> 0.227 ms
  纯GPU kernel时间: `0.071 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.227 ms`, `host_to_first_kernel_gap=0.085172`, `host_end_to_last_kernel_tail=0.045 ms`, `gpu_makespan=0.186 ms`, `gpu_kernel_sum=0.071 ms`
  开始时间(ns): `34762195`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[2048, 512]]}`
- `attn_mha` -> 0.140 ms
  纯GPU kernel时间: `0.142 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.140 ms`, `host_to_first_kernel_gap=0.10459`, `host_end_to_last_kernel_tail=0.122 ms`, `gpu_makespan=0.158 ms`, `gpu_kernel_sum=0.142 ms`
  开始时间(ns): `35088025`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2048, 128, 192], [2048, 128, 192], [2048, 128, 128]]}`
- `o_proj` -> 0.224 ms
  纯GPU kernel时间: `0.375 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.224 ms`, `host_to_first_kernel_gap=0.086741`, `host_end_to_last_kernel_tail=0.311 ms`, `gpu_makespan=0.448 ms`, `gpu_kernel_sum=0.375 ms`
  开始时间(ns): `35268178`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2048, 16384]]}`

## Layer 3 / prefill / instance 1

- 执行序号: `4`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `1.780 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.780 ms`, `module_to_last_kernel=2.074 ms`, `host_to_first_kernel_gap=0.734679`, `host_end_to_last_kernel_tail=0.293 ms`, `gpu_makespan=1.339 ms`, `gpu_kernel_sum=0.817 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=2048`, `total_tokens=2048`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[2048, 128, 192], [2048, 128, 192], [2048, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=2048`, `sum_prefix=0`, `sum_seq_after=2048`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 512, 'prefix_len': 0, 'seq_len_after': 512, 'prompt_len': 512, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 512, 'prefix_len': 0, 'seq_len_after': 512, 'prompt_len': 512, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 512, 'prefix_len': 0, 'seq_len_after': 512, 'prompt_len': 512, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 512, 'prefix_len': 0, 'seq_len_after': 512, 'prompt_len': 512, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.209 ms
  纯GPU kernel时间: `0.074 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.209 ms`, `host_to_first_kernel_gap=0.615214`, `host_end_to_last_kernel_tail=0.482 ms`, `gpu_makespan=0.076 ms`, `gpu_kernel_sum=0.074 ms`
  开始时间(ns): `36418617`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2048, 7168]]}`
- `q_a_layernorm` -> 0.046 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.046 ms`, `host_to_first_kernel_gap=0.430732`, `host_end_to_last_kernel_tail=0.390 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `36678619`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2048, 1536]]}`
- `q_b_proj` -> 0.210 ms
  纯GPU kernel时间: `0.119 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.210 ms`, `host_to_first_kernel_gap=0.361971`, `host_end_to_last_kernel_tail=0.271 ms`, `gpu_makespan=0.120 ms`, `gpu_kernel_sum=0.119 ms`
  开始时间(ns): `36754036`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2048, 1536]]}`
- `kv_a_layernorm` -> 0.041 ms
  纯GPU kernel时间: `0.006 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.041 ms`, `host_to_first_kernel_gap=0.212094`, `host_end_to_last_kernel_tail=0.177 ms`, `gpu_makespan=0.006 ms`, `gpu_kernel_sum=0.006 ms`
  开始时间(ns): `37023657`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2048, 512]]}`
- `rotary_emb` -> 0.078 ms
  纯GPU kernel时间: `0.027 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.078 ms`, `host_to_first_kernel_gap=0.13992`, `host_end_to_last_kernel_tail=0.089 ms`, `gpu_makespan=0.027 ms`, `gpu_kernel_sum=0.027 ms`
  开始时间(ns): `37102999`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2048], [2048, 128, 64], [2048, 1, 64]]}`
- `kv_b_proj` -> 0.236 ms
  纯GPU kernel时间: `0.071 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.236 ms`, `host_to_first_kernel_gap=0.08502`, `host_end_to_last_kernel_tail=0.043 ms`, `gpu_makespan=0.193 ms`, `gpu_kernel_sum=0.071 ms`
  开始时间(ns): `37316139`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[2048, 512]]}`
- `attn_mha` -> 0.137 ms
  纯GPU kernel时间: `0.142 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.137 ms`, `host_to_first_kernel_gap=0.100914`, `host_end_to_last_kernel_tail=0.121 ms`, `gpu_makespan=0.157 ms`, `gpu_kernel_sum=0.142 ms`
  开始时间(ns): `37649973`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2048, 128, 192], [2048, 128, 192], [2048, 128, 128]]}`
- `o_proj` -> 0.232 ms
  纯GPU kernel时间: `0.374 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.232 ms`, `host_to_first_kernel_gap=0.087548`, `host_end_to_last_kernel_tail=0.307 ms`, `gpu_makespan=0.451 ms`, `gpu_kernel_sum=0.374 ms`
  开始时间(ns): `37834123`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2048, 16384]]}`

## Layer 4 / prefill / instance 1

- 执行序号: `5`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `1.855 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.855 ms`, `module_to_last_kernel=2.965 ms`, `host_to_first_kernel_gap=2.073935`, `host_end_to_last_kernel_tail=1.110 ms`, `gpu_makespan=0.891 ms`, `gpu_kernel_sum=0.807 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=2048`, `total_tokens=2048`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[2048, 128, 192], [2048, 128, 192], [2048, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=2048`, `sum_prefix=0`, `sum_seq_after=2048`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 512, 'prefix_len': 0, 'seq_len_after': 512, 'prompt_len': 512, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 512, 'prefix_len': 0, 'seq_len_after': 512, 'prompt_len': 512, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 512, 'prefix_len': 0, 'seq_len_after': 512, 'prompt_len': 512, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 512, 'prefix_len': 0, 'seq_len_after': 512, 'prompt_len': 512, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.287 ms
  纯GPU kernel时间: `0.073 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.287 ms`, `host_to_first_kernel_gap=1.943184`, `host_end_to_last_kernel_tail=1.732 ms`, `gpu_makespan=0.075 ms`, `gpu_kernel_sum=0.073 ms`
  开始时间(ns): `39698615`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2048, 7168]]}`
- `q_a_layernorm` -> 0.048 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.048 ms`, `host_to_first_kernel_gap=1.680085`, `host_end_to_last_kernel_tail=1.637 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `40037618`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2048, 1536]]}`
- `q_b_proj` -> 0.215 ms
  纯GPU kernel时间: `0.119 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.215 ms`, `host_to_first_kernel_gap=1.608178`, `host_end_to_last_kernel_tail=1.514 ms`, `gpu_makespan=0.121 ms`, `gpu_kernel_sum=0.119 ms`
  开始时间(ns): `40115765`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2048, 1536]]}`
- `kv_a_layernorm` -> 0.045 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.045 ms`, `host_to_first_kernel_gap=1.452751`, `host_end_to_last_kernel_tail=1.413 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `40392152`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2048, 512]]}`
- `rotary_emb` -> 0.077 ms
  纯GPU kernel时间: `0.027 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.077 ms`, `host_to_first_kernel_gap=1.376427`, `host_end_to_last_kernel_tail=1.326 ms`, `gpu_makespan=0.027 ms`, `gpu_kernel_sum=0.027 ms`
  开始时间(ns): `40474972`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2048], [2048, 128, 64], [2048, 1, 64]]}`
- `kv_b_proj` -> 0.232 ms
  纯GPU kernel时间: `0.071 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.232 ms`, `host_to_first_kernel_gap=1.209437`, `host_end_to_last_kernel_tail=1.050 ms`, `gpu_makespan=0.073 ms`, `gpu_kernel_sum=0.071 ms`
  开始时间(ns): `40679306`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[2048, 512]]}`
- `attn_mha` -> 0.139 ms
  纯GPU kernel时间: `0.139 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.139 ms`, `host_to_first_kernel_gap=1.01404`, `host_end_to_last_kernel_tail=1.014 ms`, `gpu_makespan=0.139 ms`, `gpu_kernel_sum=0.139 ms`
  开始时间(ns): `41008783`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2048, 128, 192], [2048, 128, 192], [2048, 128, 128]]}`
- `o_proj` -> 0.224 ms
  纯GPU kernel时间: `0.367 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.224 ms`, `host_to_first_kernel_gap=0.977445`, `host_end_to_last_kernel_tail=1.122 ms`, `gpu_makespan=0.369 ms`, `gpu_kernel_sum=0.367 ms`
  开始时间(ns): `41186754`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2048, 16384]]}`

## Layer 0 / decode / instance 1

- 执行序号: `6`
- Collector对齐边界: `{'Module': 'model.model.layers.0.self_attn'}`
- 整块 MLA-module 时长: `1.859 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.859 ms`, `module_to_last_kernel=1.859 ms`, `host_to_first_kernel_gap=0.706947`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=1.150 ms`, `gpu_kernel_sum=0.111 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.307 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.307 ms`, `host_to_first_kernel_gap=0.638043`, `host_end_to_last_kernel_tail=0.347 ms`, `gpu_makespan=0.016 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `45276267`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4, 7168]]}`
- `q_a_layernorm` -> 0.049 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.049 ms`, `host_to_first_kernel_gap=0.280483`, `host_end_to_last_kernel_tail=0.233 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `45649443`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4, 1536]]}`
- `kv_a_layernorm` -> 0.031 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.031 ms`, `host_to_first_kernel_gap=0.208633`, `host_end_to_last_kernel_tail=0.180 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `45723309`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4, 512]]}`
- `q_b_proj` -> 0.246 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.246 ms`, `host_to_first_kernel_gap=0.142803`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.096 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `45792243`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4, 1536]]}`
- `rotary_emb` -> 0.081 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.081 ms`, `host_to_first_kernel_gap=0.069244`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `46210218`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4], [4, 128, 64], [4, 1, 64]]}`
- `attn_mqa` -> 0.299 ms
  纯GPU kernel时间: `0.023 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.299 ms`, `host_to_first_kernel_gap=0.093898`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.182 ms`, `gpu_kernel_sum=0.023 ms`
  开始时间(ns): `46339932`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4, 128, 512], [4, 1, 512], [4, 1, 512]]}`
- `o_proj` -> 0.283 ms
  纯GPU kernel时间: `0.049 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.283 ms`, `host_to_first_kernel_gap=0.106838`, `host_end_to_last_kernel_tail=0.016 ms`, `gpu_makespan=0.192 ms`, `gpu_kernel_sum=0.049 ms`
  开始时间(ns): `46765488`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4, 16384]]}`

## Layer 1 / decode / instance 1

- 执行序号: `7`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `1.611 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.611 ms`, `module_to_last_kernel=1.614 ms`, `host_to_first_kernel_gap=0.143027`, `host_end_to_last_kernel_tail=0.003 ms`, `gpu_makespan=1.471 ms`, `gpu_kernel_sum=0.109 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.213 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.213 ms`, `host_to_first_kernel_gap=0.08218`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.122 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `47972482`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4, 7168]]}`
- `q_a_layernorm` -> 0.044 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.044 ms`, `host_to_first_kernel_gap=0.038308`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `48244194`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4, 1536]]}`
- `kv_a_layernorm` -> 0.030 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.030 ms`, `host_to_first_kernel_gap=0.026262`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `48312176`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4, 512]]}`
- `q_b_proj` -> 0.202 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.202 ms`, `host_to_first_kernel_gap=0.073695`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.123 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `48373415`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4, 1536]]}`
- `rotary_emb` -> 0.077 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.077 ms`, `host_to_first_kernel_gap=0.065873`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `48713141`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4], [4, 128, 64], [4, 1, 64]]}`
- `attn_mqa` -> 0.273 ms
  纯GPU kernel时间: `0.023 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.273 ms`, `host_to_first_kernel_gap=0.08616`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.170 ms`, `gpu_kernel_sum=0.023 ms`
  开始时间(ns): `48842646`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4, 128, 512], [4, 1, 512], [4, 1, 512]]}`
- `o_proj` -> 0.265 ms
  纯GPU kernel时间: `0.048 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.265 ms`, `host_to_first_kernel_gap=0.094957`, `host_end_to_last_kernel_tail=0.020 ms`, `gpu_makespan=0.189 ms`, `gpu_kernel_sum=0.048 ms`
  开始时间(ns): `49240857`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4, 16384]]}`

## Layer 2 / decode / instance 1

- 执行序号: `8`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.591 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.591 ms`, `module_to_last_kernel=1.593 ms`, `host_to_first_kernel_gap=0.142186`, `host_end_to_last_kernel_tail=0.002 ms`, `gpu_makespan=1.451 ms`, `gpu_kernel_sum=0.107 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.204 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.204 ms`, `host_to_first_kernel_gap=0.077981`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.118 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `50416137`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4, 7168]]}`
- `q_a_layernorm` -> 0.043 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.043 ms`, `host_to_first_kernel_gap=0.037696`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `50679110`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4, 1536]]}`
- `kv_a_layernorm` -> 0.031 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.031 ms`, `host_to_first_kernel_gap=0.026615`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `50746063`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4, 512]]}`
- `q_b_proj` -> 0.195 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.195 ms`, `host_to_first_kernel_gap=0.072864`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.119 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `50815014`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4, 1536]]}`
- `rotary_emb` -> 0.074 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.074 ms`, `host_to_first_kernel_gap=0.06294`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `51146186`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4], [4, 128, 64], [4, 1, 64]]}`
- `attn_mqa` -> 0.274 ms
  纯GPU kernel时间: `0.022 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.274 ms`, `host_to_first_kernel_gap=0.085378`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.171 ms`, `gpu_kernel_sum=0.022 ms`
  开始时间(ns): `51265764`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4, 128, 512], [4, 1, 512], [4, 1, 512]]}`
- `o_proj` -> 0.265 ms
  纯GPU kernel时间: `0.047 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.265 ms`, `host_to_first_kernel_gap=0.091037`, `host_end_to_last_kernel_tail=0.020 ms`, `gpu_makespan=0.194 ms`, `gpu_kernel_sum=0.047 ms`
  开始时间(ns): `51660552`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4, 16384]]}`

## Layer 3 / decode / instance 1

- 执行序号: `9`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `1.595 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.595 ms`, `module_to_last_kernel=1.597 ms`, `host_to_first_kernel_gap=0.144844`, `host_end_to_last_kernel_tail=0.002 ms`, `gpu_makespan=1.453 ms`, `gpu_kernel_sum=0.108 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.207 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.207 ms`, `host_to_first_kernel_gap=0.07979`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.120 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `52825239`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4, 7168]]}`
- `q_a_layernorm` -> 0.044 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.044 ms`, `host_to_first_kernel_gap=0.038806`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `53092079`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4, 1536]]}`
- `kv_a_layernorm` -> 0.031 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.031 ms`, `host_to_first_kernel_gap=0.027`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `53159501`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4, 512]]}`
- `q_b_proj` -> 0.193 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.193 ms`, `host_to_first_kernel_gap=0.071864`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.118 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `53222029`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4, 1536]]}`
- `rotary_emb` -> 0.075 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.075 ms`, `host_to_first_kernel_gap=0.063342`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `53549975`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4], [4, 128, 64], [4, 1, 64]]}`
- `attn_mqa` -> 0.288 ms
  纯GPU kernel时间: `0.023 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.288 ms`, `host_to_first_kernel_gap=0.09021`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.178 ms`, `gpu_kernel_sum=0.023 ms`
  开始时间(ns): `53670659`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4, 128, 512], [4, 1, 512], [4, 1, 512]]}`
- `o_proj` -> 0.258 ms
  纯GPU kernel时间: `0.047 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.258 ms`, `host_to_first_kernel_gap=0.093569`, `host_end_to_last_kernel_tail=0.020 ms`, `gpu_makespan=0.184 ms`, `gpu_kernel_sum=0.047 ms`
  开始时间(ns): `54079620`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4, 16384]]}`

## Layer 4 / decode / instance 1

- 执行序号: `10`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `1.705 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.705 ms`, `module_to_last_kernel=1.709 ms`, `host_to_first_kernel_gap=0.173096`, `host_end_to_last_kernel_tail=0.004 ms`, `gpu_makespan=1.536 ms`, `gpu_kernel_sum=0.109 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.274 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.274 ms`, `host_to_first_kernel_gap=0.099917`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.162 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `55845656`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4, 7168]]}`
- `q_a_layernorm` -> 0.047 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.047 ms`, `host_to_first_kernel_gap=0.041264`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `56184021`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4, 1536]]}`
- `kv_a_layernorm` -> 0.031 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.031 ms`, `host_to_first_kernel_gap=0.026698`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `56254907`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4, 512]]}`
- `q_b_proj` -> 0.215 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.215 ms`, `host_to_first_kernel_gap=0.078676`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.133 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `56319345`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4, 1536]]}`
- `rotary_emb` -> 0.078 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.078 ms`, `host_to_first_kernel_gap=0.066929`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `56682324`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4], [4, 128, 64], [4, 1, 64]]}`
- `attn_mqa` -> 0.282 ms
  纯GPU kernel时间: `0.022 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.282 ms`, `host_to_first_kernel_gap=0.092211`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.169 ms`, `gpu_kernel_sum=0.022 ms`
  开始时间(ns): `56811858`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4, 128, 512], [4, 1, 512], [4, 1, 512]]}`
- `o_proj` -> 0.247 ms
  纯GPU kernel时间: `0.047 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.247 ms`, `host_to_first_kernel_gap=0.089597`, `host_end_to_last_kernel_tail=0.021 ms`, `gpu_makespan=0.178 ms`, `gpu_kernel_sum=0.047 ms`
  开始时间(ns): `57213288`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4, 16384]]}`
