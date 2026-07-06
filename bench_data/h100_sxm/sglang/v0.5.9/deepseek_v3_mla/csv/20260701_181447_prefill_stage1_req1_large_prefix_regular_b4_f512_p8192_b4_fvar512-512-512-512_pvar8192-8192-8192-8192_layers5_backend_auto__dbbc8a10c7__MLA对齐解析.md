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
- 整块 MLA-module 时长: `2.762 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.762 ms`, `module_to_last_kernel=6.609 ms`, `host_to_first_kernel_gap=0.31498`, `host_end_to_last_kernel_tail=3.847 ms`, `gpu_makespan=6.294 ms`, `gpu_kernel_sum=3.757 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=2048`, `total_tokens=34816`, `chunked_req_prefix_len=32768`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[2048, 128, 192], [34816, 128, 192], [34816, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=2048`, `sum_prefix=32768`, `sum_seq_after=34816`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 512, 'prefix_len': 8192, 'seq_len_after': 8704, 'prompt_len': 8704, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 512, 'prefix_len': 8192, 'seq_len_after': 8704, 'prompt_len': 8704, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 512, 'prefix_len': 8192, 'seq_len_after': 8704, 'prompt_len': 8704, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 512, 'prefix_len': 8192, 'seq_len_after': 8704, 'prompt_len': 8704, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.417 ms
  纯GPU kernel时间: `0.074 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.417 ms`, `host_to_first_kernel_gap=0.142718`, `host_end_to_last_kernel_tail=0.017 ms`, `gpu_makespan=0.291 ms`, `gpu_kernel_sum=0.074 ms`
  开始时间(ns): `49128989`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2048, 7168]]}`
- `q_a_layernorm` -> 0.060 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.060 ms`, `host_to_first_kernel_gap=0.053337`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `49613281`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2048, 1536]]}`
- `q_b_proj` -> 0.262 ms
  纯GPU kernel时间: `0.123 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.262 ms`, `host_to_first_kernel_gap=0.093356`, `host_end_to_last_kernel_tail=0.082 ms`, `gpu_makespan=0.250 ms`, `gpu_kernel_sum=0.123 ms`
  开始时间(ns): `49709326`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2048, 1536]]}`
- `kv_a_layernorm` -> 0.046 ms
  纯GPU kernel时间: `0.006 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.046 ms`, `host_to_first_kernel_gap=0.040292`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.006 ms`, `gpu_kernel_sum=0.006 ms`
  开始时间(ns): `50038613`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2048, 512]]}`
- `rotary_emb` -> 0.105 ms
  纯GPU kernel时间: `0.027 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.105 ms`, `host_to_first_kernel_gap=0.091441`, `host_end_to_last_kernel_tail=0.013 ms`, `gpu_makespan=0.027 ms`, `gpu_kernel_sum=0.027 ms`
  开始时间(ns): `50130728`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2048], [2048, 128, 64], [2048, 1, 64]]}`
- `kv_b_proj` -> 0.287 ms
  纯GPU kernel时间: `1.227 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.287 ms`, `host_to_first_kernel_gap=0.107072`, `host_end_to_last_kernel_tail=1.176 ms`, `gpu_makespan=1.356 ms`, `gpu_kernel_sum=1.227 ms`
  开始时间(ns): `50797624`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[34816, 512]]}`
- `attn_mha` -> 0.182 ms
  纯GPU kernel时间: `1.923 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.182 ms`, `host_to_first_kernel_gap=2.0643`, `host_end_to_last_kernel_tail=3.806 ms`, `gpu_makespan=1.923 ms`, `gpu_kernel_sum=1.923 ms`
  开始时间(ns): `51203496`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2048, 128, 192], [34816, 128, 192], [34816, 128, 128]]}`
- `o_proj` -> 0.272 ms
  纯GPU kernel时间: `0.371 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.272 ms`, `host_to_first_kernel_gap=3.76213`, `host_end_to_last_kernel_tail=3.863 ms`, `gpu_makespan=0.373 ms`, `gpu_kernel_sum=0.371 ms`
  开始时间(ns): `51430494`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2048, 16384]]}`

## Layer 1 / prefill / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `2.188 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.188 ms`, `module_to_last_kernel=9.038 ms`, `host_to_first_kernel_gap=4.179723`, `host_end_to_last_kernel_tail=6.850 ms`, `gpu_makespan=4.858 ms`, `gpu_kernel_sum=3.787 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=2048`, `total_tokens=34816`, `chunked_req_prefix_len=32768`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[2048, 128, 192], [34816, 128, 192], [34816, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=2048`, `sum_prefix=32768`, `sum_seq_after=34816`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 512, 'prefix_len': 8192, 'seq_len_after': 8704, 'prompt_len': 8704, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 512, 'prefix_len': 8192, 'seq_len_after': 8704, 'prompt_len': 8704, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 512, 'prefix_len': 8192, 'seq_len_after': 8704, 'prompt_len': 8704, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 512, 'prefix_len': 8192, 'seq_len_after': 8704, 'prompt_len': 8704, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.280 ms
  纯GPU kernel时间: `0.075 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.280 ms`, `host_to_first_kernel_gap=4.026557`, `host_end_to_last_kernel_tail=3.824 ms`, `gpu_makespan=0.077 ms`, `gpu_kernel_sum=0.075 ms`
  开始时间(ns): `52783632`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2048, 7168]]}`
- `q_a_layernorm` -> 0.060 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.060 ms`, `host_to_first_kernel_gap=3.767794`, `host_end_to_last_kernel_tail=3.714 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `53120219`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2048, 1536]]}`
- `q_b_proj` -> 0.230 ms
  纯GPU kernel时间: `0.121 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.230 ms`, `host_to_first_kernel_gap=3.666124`, `host_end_to_last_kernel_tail=3.559 ms`, `gpu_makespan=0.123 ms`, `gpu_kernel_sum=0.121 ms`
  开始时间(ns): `53228257`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2048, 1536]]}`
- `kv_a_layernorm` -> 0.047 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.047 ms`, `host_to_first_kernel_gap=3.483741`, `host_end_to_last_kernel_tail=3.442 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `53533360`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2048, 512]]}`
- `rotary_emb` -> 0.094 ms
  纯GPU kernel时间: `0.027 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.094 ms`, `host_to_first_kernel_gap=3.400746`, `host_end_to_last_kernel_tail=3.333 ms`, `gpu_makespan=0.027 ms`, `gpu_kernel_sum=0.027 ms`
  开始时间(ns): `53622755`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2048], [2048, 128, 64], [2048, 1, 64]]}`
- `kv_b_proj` -> 0.237 ms
  纯GPU kernel时间: `1.259 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.237 ms`, `host_to_first_kernel_gap=3.126406`, `host_end_to_last_kernel_tail=4.149 ms`, `gpu_makespan=1.260 ms`, `gpu_kernel_sum=1.259 ms`
  开始时间(ns): `53974727`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[34816, 512]]}`
- `attn_mha` -> 0.166 ms
  纯GPU kernel时间: `1.925 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.166 ms`, `host_to_first_kernel_gap=5.053465`, `host_end_to_last_kernel_tail=6.812 ms`, `gpu_makespan=1.925 ms`, `gpu_kernel_sum=1.925 ms`
  开始时间(ns): `54317808`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2048, 128, 192], [34816, 128, 192], [34816, 128, 128]]}`
- `o_proj` -> 0.251 ms
  纯GPU kernel时间: `0.370 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.251 ms`, `host_to_first_kernel_gap=6.746859`, `host_end_to_last_kernel_tail=6.867 ms`, `gpu_makespan=0.371 ms`, `gpu_kernel_sum=0.370 ms`
  开始时间(ns): `54550650`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2048, 16384]]}`

## Layer 2 / prefill / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `2.079 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.079 ms`, `module_to_last_kernel=12.177 ms`, `host_to_first_kernel_gap=7.240498`, `host_end_to_last_kernel_tail=10.098 ms`, `gpu_makespan=4.937 ms`, `gpu_kernel_sum=3.866 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=2048`, `total_tokens=34816`, `chunked_req_prefix_len=32768`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[2048, 128, 192], [34816, 128, 192], [34816, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=2048`, `sum_prefix=32768`, `sum_seq_after=34816`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 512, 'prefix_len': 8192, 'seq_len_after': 8704, 'prompt_len': 8704, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 512, 'prefix_len': 8192, 'seq_len_after': 8704, 'prompt_len': 8704, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 512, 'prefix_len': 8192, 'seq_len_after': 8704, 'prompt_len': 8704, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 512, 'prefix_len': 8192, 'seq_len_after': 8704, 'prompt_len': 8704, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.257 ms
  纯GPU kernel时间: `0.073 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.257 ms`, `host_to_first_kernel_gap=7.090833`, `host_end_to_last_kernel_tail=6.909 ms`, `gpu_makespan=0.075 ms`, `gpu_kernel_sum=0.073 ms`
  开始时间(ns): `55822194`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2048, 7168]]}`
- `q_a_layernorm` -> 0.054 ms
  纯GPU kernel时间: `0.006 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.054 ms`, `host_to_first_kernel_gap=6.855289`, `host_end_to_last_kernel_tail=6.807 ms`, `gpu_makespan=0.006 ms`, `gpu_kernel_sum=0.006 ms`
  开始时间(ns): `56133481`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2048, 1536]]}`
- `q_b_proj` -> 0.220 ms
  纯GPU kernel时间: `0.120 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.220 ms`, `host_to_first_kernel_gap=6.774643`, `host_end_to_last_kernel_tail=6.675 ms`, `gpu_makespan=0.121 ms`, `gpu_kernel_sum=0.120 ms`
  开始时间(ns): `56221007`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2048, 1536]]}`
- `kv_a_layernorm` -> 0.052 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.052 ms`, `host_to_first_kernel_gap=6.601977`, `host_end_to_last_kernel_tail=6.555 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `56514505`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2048, 512]]}`
- `rotary_emb` -> 0.092 ms
  纯GPU kernel时间: `0.027 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.092 ms`, `host_to_first_kernel_gap=6.514192`, `host_end_to_last_kernel_tail=6.449 ms`, `gpu_makespan=0.027 ms`, `gpu_kernel_sum=0.027 ms`
  开始时间(ns): `56609874`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2048], [2048, 128, 64], [2048, 1, 64]]}`
- `kv_b_proj` -> 0.234 ms
  纯GPU kernel时间: `1.337 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.234 ms`, `host_to_first_kernel_gap=6.250921`, `host_end_to_last_kernel_tail=7.357 ms`, `gpu_makespan=1.339 ms`, `gpu_kernel_sum=1.337 ms`
  开始时间(ns): `56949081`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[34816, 512]]}`
- `attn_mha` -> 0.164 ms
  纯GPU kernel时间: `1.927 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.164 ms`, `host_to_first_kernel_gap=8.259079`, `host_end_to_last_kernel_tail=10.023 ms`, `gpu_makespan=1.927 ms`, `gpu_kernel_sum=1.927 ms`
  开始时间(ns): `57287831`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2048, 128, 192], [34816, 128, 192], [34816, 128, 128]]}`
- `o_proj` -> 0.234 ms
  纯GPU kernel时间: `0.371 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.234 ms`, `host_to_first_kernel_gap=9.974732`, `host_end_to_last_kernel_tail=10.113 ms`, `gpu_makespan=0.373 ms`, `gpu_kernel_sum=0.371 ms`
  开始时间(ns): `57502094`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2048, 16384]]}`

## Layer 3 / prefill / instance 1

- 执行序号: `4`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `2.080 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.080 ms`, `module_to_last_kernel=15.369 ms`, `host_to_first_kernel_gap=10.474721`, `host_end_to_last_kernel_tail=13.289 ms`, `gpu_makespan=4.894 ms`, `gpu_kernel_sum=3.825 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=2048`, `total_tokens=34816`, `chunked_req_prefix_len=32768`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[2048, 128, 192], [34816, 128, 192], [34816, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=2048`, `sum_prefix=32768`, `sum_seq_after=34816`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 512, 'prefix_len': 8192, 'seq_len_after': 8704, 'prompt_len': 8704, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 512, 'prefix_len': 8192, 'seq_len_after': 8704, 'prompt_len': 8704, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 512, 'prefix_len': 8192, 'seq_len_after': 8704, 'prompt_len': 8704, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 512, 'prefix_len': 8192, 'seq_len_after': 8704, 'prompt_len': 8704, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.256 ms
  纯GPU kernel时间: `0.074 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.256 ms`, `host_to_first_kernel_gap=10.326769`, `host_end_to_last_kernel_tail=10.146 ms`, `gpu_makespan=0.075 ms`, `gpu_kernel_sum=0.074 ms`
  开始时间(ns): `58763079`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2048, 7168]]}`
- `q_a_layernorm` -> 0.056 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.056 ms`, `host_to_first_kernel_gap=10.091282`, `host_end_to_last_kernel_tail=10.041 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `59073829`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2048, 1536]]}`
- `q_b_proj` -> 0.223 ms
  纯GPU kernel时间: `0.120 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.223 ms`, `host_to_first_kernel_gap=10.00993`, `host_end_to_last_kernel_tail=9.908 ms`, `gpu_makespan=0.121 ms`, `gpu_kernel_sum=0.120 ms`
  开始时间(ns): `59162029`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2048, 1536]]}`
- `kv_a_layernorm` -> 0.054 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.054 ms`, `host_to_first_kernel_gap=9.846032`, `host_end_to_last_kernel_tail=9.798 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `59446759`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2048, 512]]}`
- `rotary_emb` -> 0.090 ms
  纯GPU kernel时间: `0.027 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.090 ms`, `host_to_first_kernel_gap=9.757423`, `host_end_to_last_kernel_tail=9.694 ms`, `gpu_makespan=0.027 ms`, `gpu_kernel_sum=0.027 ms`
  开始时间(ns): `59542888`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2048], [2048, 128, 64], [2048, 1, 64]]}`
- `kv_b_proj` -> 0.237 ms
  纯GPU kernel时间: `1.298 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.237 ms`, `host_to_first_kernel_gap=9.497925`, `host_end_to_last_kernel_tail=10.560 ms`, `gpu_makespan=1.299 ms`, `gpu_kernel_sum=1.298 ms`
  开始时间(ns): `59878738`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[34816, 512]]}`
- `attn_mha` -> 0.165 ms
  纯GPU kernel时间: `1.927 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.165 ms`, `host_to_first_kernel_gap=11.462882`, `host_end_to_last_kernel_tail=13.225 ms`, `gpu_makespan=1.927 ms`, `gpu_kernel_sum=1.927 ms`
  开始时间(ns): `60220433`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2048, 128, 192], [34816, 128, 192], [34816, 128, 128]]}`
- `o_proj` -> 0.248 ms
  纯GPU kernel时间: `0.369 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.248 ms`, `host_to_first_kernel_gap=13.186746`, `host_end_to_last_kernel_tail=13.310 ms`, `gpu_makespan=0.371 ms`, `gpu_kernel_sum=0.369 ms`
  开始时间(ns): `60426646`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2048, 16384]]}`

## Layer 4 / prefill / instance 1

- 执行序号: `5`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `2.250 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.250 ms`, `module_to_last_kernel=19.190 ms`, `host_to_first_kernel_gap=14.285251`, `host_end_to_last_kernel_tail=16.940 ms`, `gpu_makespan=4.905 ms`, `gpu_kernel_sum=3.837 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=2048`, `total_tokens=34816`, `chunked_req_prefix_len=32768`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[2048, 128, 192], [34816, 128, 192], [34816, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=2048`, `sum_prefix=32768`, `sum_seq_after=34816`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 512, 'prefix_len': 8192, 'seq_len_after': 8704, 'prompt_len': 8704, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 512, 'prefix_len': 8192, 'seq_len_after': 8704, 'prompt_len': 8704, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 512, 'prefix_len': 8192, 'seq_len_after': 8704, 'prompt_len': 8704, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 512, 'prefix_len': 8192, 'seq_len_after': 8704, 'prompt_len': 8704, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.335 ms
  纯GPU kernel时间: `0.077 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.335 ms`, `host_to_first_kernel_gap=14.12551`, `host_end_to_last_kernel_tail=13.870 ms`, `gpu_makespan=0.079 ms`, `gpu_kernel_sum=0.077 ms`
  开始时间(ns): `62437316`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2048, 7168]]}`
- `q_a_layernorm` -> 0.057 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.057 ms`, `host_to_first_kernel_gap=13.813778`, `host_end_to_last_kernel_tail=13.762 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `62828792`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2048, 1536]]}`
- `q_b_proj` -> 0.233 ms
  纯GPU kernel时间: `0.119 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.233 ms`, `host_to_first_kernel_gap=13.722304`, `host_end_to_last_kernel_tail=13.611 ms`, `gpu_makespan=0.121 ms`, `gpu_kernel_sum=0.119 ms`
  开始时间(ns): `62926666`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2048, 1536]]}`
- `kv_a_layernorm` -> 0.046 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.046 ms`, `host_to_first_kernel_gap=13.546889`, `host_end_to_last_kernel_tail=13.506 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `63223169`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2048, 512]]}`
- `rotary_emb` -> 0.098 ms
  纯GPU kernel时间: `0.026 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.098 ms`, `host_to_first_kernel_gap=13.463116`, `host_end_to_last_kernel_tail=13.391 ms`, `gpu_makespan=0.026 ms`, `gpu_kernel_sum=0.026 ms`
  开始时间(ns): `63313694`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2048], [2048, 128, 64], [2048, 1, 64]]}`
- `kv_b_proj` -> 0.242 ms
  纯GPU kernel时间: `1.309 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.242 ms`, `host_to_first_kernel_gap=13.191847`, `host_end_to_last_kernel_tail=14.261 ms`, `gpu_makespan=1.311 ms`, `gpu_kernel_sum=1.309 ms`
  开始时间(ns): `63660707`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[34816, 512]]}`
- `attn_mha` -> 0.174 ms
  纯GPU kernel时间: `1.926 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.174 ms`, `host_to_first_kernel_gap=15.150947`, `host_end_to_last_kernel_tail=16.903 ms`, `gpu_makespan=1.926 ms`, `gpu_kernel_sum=1.926 ms`
  开始时间(ns): `64018979`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2048, 128, 192], [34816, 128, 192], [34816, 128, 128]]}`
- `o_proj` -> 0.266 ms
  纯GPU kernel时间: `0.368 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.266 ms`, `host_to_first_kernel_gap=16.851885`, `host_end_to_last_kernel_tail=16.955 ms`, `gpu_makespan=0.370 ms`, `gpu_kernel_sum=0.368 ms`
  开始时间(ns): `64245941`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2048, 16384]]}`

## Layer 0 / decode / instance 1

- 执行序号: `6`
- Collector对齐边界: `{'Module': 'model.model.layers.0.self_attn'}`
- 整块 MLA-module 时长: `2.284 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.284 ms`, `module_to_last_kernel=15.874 ms`, `host_to_first_kernel_gap=15.704843`, `host_end_to_last_kernel_tail=13.590 ms`, `gpu_makespan=0.169 ms`, `gpu_kernel_sum=0.135 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.439 ms
  纯GPU kernel时间: `0.016 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.439 ms`, `host_to_first_kernel_gap=15.613748`, `host_end_to_last_kernel_tail=15.192 ms`, `gpu_makespan=0.017 ms`, `gpu_kernel_sum=0.016 ms`
  开始时间(ns): `69203080`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4, 7168]]}`
- `q_a_layernorm` -> 0.058 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.058 ms`, `host_to_first_kernel_gap=15.112003`, `host_end_to_last_kernel_tail=15.056 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `69721433`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4, 1536]]}`
- `kv_a_layernorm` -> 0.045 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.045 ms`, `host_to_first_kernel_gap=15.027928`, `host_end_to_last_kernel_tail=14.985 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `69807524`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4, 512]]}`
- `q_b_proj` -> 0.257 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.257 ms`, `host_to_first_kernel_gap=14.944279`, `host_end_to_last_kernel_tail=14.709 ms`, `gpu_makespan=0.022 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `69894053`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4, 1536]]}`
- `rotary_emb` -> 0.105 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.105 ms`, `host_to_first_kernel_gap=14.510505`, `host_end_to_last_kernel_tail=14.407 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `70361491`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4], [4, 128, 64], [4, 1, 64]]}`
- `attn_mqa` -> 0.369 ms
  纯GPU kernel时间: `0.044 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.369 ms`, `host_to_first_kernel_gap=14.341364`, `host_end_to_last_kernel_tail=14.019 ms`, `gpu_makespan=0.047 ms`, `gpu_kernel_sum=0.044 ms`
  开始时间(ns): `70533288`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4, 128, 512], [4, 1, 512], [4, 1, 512]]}`
- `o_proj` -> 0.311 ms
  纯GPU kernel时间: `0.051 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.311 ms`, `host_to_first_kernel_gap=13.868505`, `host_end_to_last_kernel_tail=13.609 ms`, `gpu_makespan=0.052 ms`, `gpu_kernel_sum=0.051 ms`
  开始时间(ns): `71065186`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4, 16384]]}`

## Layer 1 / decode / instance 1

- 执行序号: `7`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `1.915 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.915 ms`, `module_to_last_kernel=12.973 ms`, `host_to_first_kernel_gap=12.804652`, `host_end_to_last_kernel_tail=11.058 ms`, `gpu_makespan=0.168 ms`, `gpu_kernel_sum=0.135 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.235 ms
  纯GPU kernel时间: `0.016 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.235 ms`, `host_to_first_kernel_gap=12.732193`, `host_end_to_last_kernel_tail=12.514 ms`, `gpu_makespan=0.017 ms`, `gpu_kernel_sum=0.016 ms`
  开始时间(ns): `72417338`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4, 7168]]}`
- `q_a_layernorm` -> 0.055 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.055 ms`, `host_to_first_kernel_gap=12.448312`, `host_end_to_last_kernel_tail=12.396 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `72718083`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4, 1536]]}`
- `kv_a_layernorm` -> 0.035 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.035 ms`, `host_to_first_kernel_gap=12.370464`, `host_end_to_last_kernel_tail=12.337 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `72798043`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4, 512]]}`
- `q_b_proj` -> 0.225 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.225 ms`, `host_to_first_kernel_gap=12.30166`, `host_end_to_last_kernel_tail=12.097 ms`, `gpu_makespan=0.020 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `72871039`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4, 1536]]}`
- `rotary_emb` -> 0.098 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.098 ms`, `host_to_first_kernel_gap=11.923771`, `host_end_to_last_kernel_tail=11.828 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `73281312`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4], [4, 128, 64], [4, 1, 64]]}`
- `attn_mqa` -> 0.349 ms
  纯GPU kernel时间: `0.043 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.349 ms`, `host_to_first_kernel_gap=11.77561`, `host_end_to_last_kernel_tail=11.471 ms`, `gpu_makespan=0.045 ms`, `gpu_kernel_sum=0.043 ms`
  开始时间(ns): `73432161`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4, 128, 512], [4, 1, 512], [4, 1, 512]]}`
- `o_proj` -> 0.301 ms
  纯GPU kernel时间: `0.052 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.301 ms`, `host_to_first_kernel_gap=11.326555`, `host_end_to_last_kernel_tail=11.078 ms`, `gpu_makespan=0.053 ms`, `gpu_kernel_sum=0.052 ms`
  开始时间(ns): `73938304`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4, 16384]]}`

## Layer 2 / decode / instance 1

- 执行序号: `8`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.885 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.885 ms`, `module_to_last_kernel=10.505 ms`, `host_to_first_kernel_gap=10.338176`, `host_end_to_last_kernel_tail=8.620 ms`, `gpu_makespan=0.167 ms`, `gpu_kernel_sum=0.133 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.211 ms
  纯GPU kernel时间: `0.016 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.211 ms`, `host_to_first_kernel_gap=10.264741`, `host_end_to_last_kernel_tail=10.071 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.016 ms`
  开始时间(ns): `75218197`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4, 7168]]}`
- `q_a_layernorm` -> 0.063 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.063 ms`, `host_to_first_kernel_gap=9.996703`, `host_end_to_last_kernel_tail=9.936 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `75503899`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4, 1536]]}`
- `kv_a_layernorm` -> 0.035 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.035 ms`, `host_to_first_kernel_gap=9.90919`, `host_end_to_last_kernel_tail=9.876 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `75593588`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4, 512]]}`
- `q_b_proj` -> 0.229 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.229 ms`, `host_to_first_kernel_gap=9.839811`, `host_end_to_last_kernel_tail=9.631 ms`, `gpu_makespan=0.020 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `75665975`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4, 1536]]}`
- `rotary_emb` -> 0.096 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.096 ms`, `host_to_first_kernel_gap=9.462827`, `host_end_to_last_kernel_tail=9.369 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `76075759`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4], [4, 128, 64], [4, 1, 64]]}`
- `attn_mqa` -> 0.351 ms
  纯GPU kernel时间: `0.043 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.351 ms`, `host_to_first_kernel_gap=9.31801`, `host_end_to_last_kernel_tail=9.012 ms`, `gpu_makespan=0.045 ms`, `gpu_kernel_sum=0.043 ms`
  开始时间(ns): `76223296`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4, 128, 512], [4, 1, 512], [4, 1, 512]]}`
- `o_proj` -> 0.290 ms
  纯GPU kernel时间: `0.051 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.290 ms`, `host_to_first_kernel_gap=8.877211`, `host_end_to_last_kernel_tail=8.638 ms`, `gpu_makespan=0.052 ms`, `gpu_kernel_sum=0.051 ms`
  开始时间(ns): `76720767`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4, 16384]]}`

## Layer 3 / decode / instance 1

- 执行序号: `9`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `1.826 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.826 ms`, `module_to_last_kernel=8.097 ms`, `host_to_first_kernel_gap=7.930067`, `host_end_to_last_kernel_tail=6.271 ms`, `gpu_makespan=0.167 ms`, `gpu_kernel_sum=0.135 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.213 ms
  纯GPU kernel时间: `0.016 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.213 ms`, `host_to_first_kernel_gap=7.858416`, `host_end_to_last_kernel_tail=7.662 ms`, `gpu_makespan=0.017 ms`, `gpu_kernel_sum=0.016 ms`
  开始时间(ns): `77953642`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4, 7168]]}`
- `q_a_layernorm` -> 0.056 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.056 ms`, `host_to_first_kernel_gap=7.595307`, `host_end_to_last_kernel_tail=7.541 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `78233519`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4, 1536]]}`
- `kv_a_layernorm` -> 0.033 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.033 ms`, `host_to_first_kernel_gap=7.516856`, `host_end_to_last_kernel_tail=7.485 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `78313858`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4, 512]]}`
- `q_b_proj` -> 0.223 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.223 ms`, `host_to_first_kernel_gap=7.452405`, `host_end_to_last_kernel_tail=7.249 ms`, `gpu_makespan=0.019 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `78382533`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4, 1536]]}`
- `rotary_emb` -> 0.093 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.093 ms`, `host_to_first_kernel_gap=7.088563`, `host_end_to_last_kernel_tail=6.998 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `78777703`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4], [4, 128, 64], [4, 1, 64]]}`
- `attn_mqa` -> 0.331 ms
  纯GPU kernel时间: `0.044 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.331 ms`, `host_to_first_kernel_gap=6.94711`, `host_end_to_last_kernel_tail=6.661 ms`, `gpu_makespan=0.045 ms`, `gpu_kernel_sum=0.044 ms`
  开始时间(ns): `78922356`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4, 128, 512], [4, 1, 512], [4, 1, 512]]}`
- `o_proj` -> 0.289 ms
  纯GPU kernel时间: `0.051 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.289 ms`, `host_to_first_kernel_gap=6.531591`, `host_end_to_last_kernel_tail=6.295 ms`, `gpu_makespan=0.052 ms`, `gpu_kernel_sum=0.051 ms`
  开始时间(ns): `79395379`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4, 16384]]}`

## Layer 4 / decode / instance 1

- 执行序号: `10`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `2.059 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.059 ms`, `module_to_last_kernel=4.981 ms`, `host_to_first_kernel_gap=4.81397`, `host_end_to_last_kernel_tail=2.922 ms`, `gpu_makespan=0.167 ms`, `gpu_kernel_sum=0.135 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.331 ms
  纯GPU kernel时间: `0.016 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.331 ms`, `host_to_first_kernel_gap=4.732635`, `host_end_to_last_kernel_tail=4.419 ms`, `gpu_makespan=0.017 ms`, `gpu_kernel_sum=0.016 ms`
  开始时间(ns): `81461630`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4, 7168]]}`
- `q_a_layernorm` -> 0.056 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.056 ms`, `host_to_first_kernel_gap=4.34741`, `host_end_to_last_kernel_tail=4.294 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `81863527`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4, 1536]]}`
- `kv_a_layernorm` -> 0.033 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.033 ms`, `host_to_first_kernel_gap=4.26399`, `host_end_to_last_kernel_tail=4.233 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `81948867`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4, 512]]}`
- `q_b_proj` -> 0.229 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.229 ms`, `host_to_first_kernel_gap=4.195703`, `host_end_to_last_kernel_tail=3.987 ms`, `gpu_makespan=0.020 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `82020258`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4, 1536]]}`
- `rotary_emb` -> 0.112 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.112 ms`, `host_to_first_kernel_gap=3.807827`, `host_end_to_last_kernel_tail=3.698 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `82439078`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4], [4, 128, 64], [4, 1, 64]]}`
- `attn_mqa` -> 0.339 ms
  纯GPU kernel时间: `0.043 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.339 ms`, `host_to_first_kernel_gap=3.649205`, `host_end_to_last_kernel_tail=3.355 ms`, `gpu_makespan=0.044 ms`, `gpu_kernel_sum=0.043 ms`
  开始时间(ns): `82602148`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4, 128, 512], [4, 1, 512], [4, 1, 512]]}`
- `o_proj` -> 0.314 ms
  纯GPU kernel时间: `0.052 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.314 ms`, `host_to_first_kernel_gap=3.210563`, `host_end_to_last_kernel_tail=2.950 ms`, `gpu_makespan=0.053 ms`, `gpu_kernel_sum=0.052 ms`
  开始时间(ns): `83097206`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4, 16384]]}`
