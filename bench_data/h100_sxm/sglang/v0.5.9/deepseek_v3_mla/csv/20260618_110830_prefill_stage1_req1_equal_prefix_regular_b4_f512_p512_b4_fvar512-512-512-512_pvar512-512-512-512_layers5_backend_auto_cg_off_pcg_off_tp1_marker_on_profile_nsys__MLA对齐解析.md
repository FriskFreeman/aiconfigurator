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
- 整块 MLA-module 时长: `2.598 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.598 ms`, `module_to_last_kernel=2.886 ms`, `host_to_first_kernel_gap=0.270206`, `host_end_to_last_kernel_tail=0.288 ms`, `gpu_makespan=2.616 ms`, `gpu_kernel_sum=1.002 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=2048`, `total_tokens=4096`, `chunked_req_prefix_len=2048`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[2048, 128, 192], [4096, 128, 192], [4096, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=2048`, `sum_prefix=2048`, `sum_seq_after=4096`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 512, 'prefix_len': 512, 'seq_len_after': 1024, 'prompt_len': 1024, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 512, 'prefix_len': 512, 'seq_len_after': 1024, 'prompt_len': 1024, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 512, 'prefix_len': 512, 'seq_len_after': 1024, 'prompt_len': 1024, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 512, 'prefix_len': 512, 'seq_len_after': 1024, 'prompt_len': 1024, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.359 ms
  纯GPU kernel时间: `0.074 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.359 ms`, `host_to_first_kernel_gap=0.124223`, `host_end_to_last_kernel_tail=0.020 ms`, `gpu_makespan=0.255 ms`, `gpu_kernel_sum=0.074 ms`
  开始时间(ns): `29050502`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2048, 7168]]}`
- `q_a_layernorm` -> 0.051 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.051 ms`, `host_to_first_kernel_gap=0.045245`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `29466760`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2048, 1536]]}`
- `q_b_proj` -> 0.292 ms
  纯GPU kernel时间: `0.123 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.292 ms`, `host_to_first_kernel_gap=0.124719`, `host_end_to_last_kernel_tail=0.086 ms`, `gpu_makespan=0.253 ms`, `gpu_kernel_sum=0.123 ms`
  开始时间(ns): `29551990`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2048, 1536]]}`
- `kv_a_layernorm` -> 0.045 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.045 ms`, `host_to_first_kernel_gap=0.041036`, `host_end_to_last_kernel_tail=0.001 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `29914425`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2048, 512]]}`
- `rotary_emb` -> 0.089 ms
  纯GPU kernel时间: `0.026 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.089 ms`, `host_to_first_kernel_gap=0.076148`, `host_end_to_last_kernel_tail=0.014 ms`, `gpu_makespan=0.026 ms`, `gpu_kernel_sum=0.026 ms`
  开始时间(ns): `30003377`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2048], [2048, 128, 64], [2048, 1, 64]]}`
- `kv_b_proj` -> 0.314 ms
  纯GPU kernel时间: `0.144 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.314 ms`, `host_to_first_kernel_gap=0.130419`, `host_end_to_last_kernel_tail=0.111 ms`, `gpu_makespan=0.295 ms`, `gpu_kernel_sum=0.144 ms`
  开始时间(ns): `30577842`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[4096, 512]]}`
- `attn_mha` -> 0.169 ms
  纯GPU kernel时间: `0.254 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.169 ms`, `host_to_first_kernel_gap=0.121368`, `host_end_to_last_kernel_tail=0.225 ms`, `gpu_makespan=0.273 ms`, `gpu_kernel_sum=0.254 ms`
  开始时间(ns): `31003149`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2048, 128, 192], [4096, 128, 192], [4096, 128, 128]]}`
- `o_proj` -> 0.254 ms
  纯GPU kernel时间: `0.370 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.254 ms`, `host_to_first_kernel_gap=0.165738`, `host_end_to_last_kernel_tail=0.303 ms`, `gpu_makespan=0.392 ms`, `gpu_kernel_sum=0.370 ms`
  开始时间(ns): `31233274`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2048, 16384]]}`

## Layer 1 / prefill / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `2.885 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.885 ms`, `module_to_last_kernel=3.174 ms`, `host_to_first_kernel_gap=0.663307`, `host_end_to_last_kernel_tail=0.289 ms`, `gpu_makespan=2.511 ms`, `gpu_kernel_sum=1.008 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=2048`, `total_tokens=4096`, `chunked_req_prefix_len=2048`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[2048, 128, 192], [4096, 128, 192], [4096, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=2048`, `sum_prefix=2048`, `sum_seq_after=4096`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 512, 'prefix_len': 512, 'seq_len_after': 1024, 'prompt_len': 1024, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 512, 'prefix_len': 512, 'seq_len_after': 1024, 'prompt_len': 1024, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 512, 'prefix_len': 512, 'seq_len_after': 1024, 'prompt_len': 1024, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 512, 'prefix_len': 512, 'seq_len_after': 1024, 'prompt_len': 1024, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.331 ms
  纯GPU kernel时间: `0.073 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.331 ms`, `host_to_first_kernel_gap=0.531303`, `host_end_to_last_kernel_tail=0.275 ms`, `gpu_makespan=0.074 ms`, `gpu_kernel_sum=0.073 ms`
  开始时间(ns): `32536861`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2048, 7168]]}`
- `q_a_layernorm` -> 0.073 ms
  纯GPU kernel时间: `0.006 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.073 ms`, `host_to_first_kernel_gap=0.20625`, `host_end_to_last_kernel_tail=0.138 ms`, `gpu_makespan=0.006 ms`, `gpu_kernel_sum=0.006 ms`
  开始时间(ns): `32936570`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2048, 1536]]}`
- `q_b_proj` -> 0.243 ms
  纯GPU kernel时间: `0.122 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.243 ms`, `host_to_first_kernel_gap=0.107998`, `host_end_to_last_kernel_tail=0.094 ms`, `gpu_makespan=0.229 ms`, `gpu_kernel_sum=0.122 ms`
  开始时间(ns): `33041446`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2048, 1536]]}`
- `kv_a_layernorm` -> 0.042 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.042 ms`, `host_to_first_kernel_gap=0.03739`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `33346005`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2048, 512]]}`
- `rotary_emb` -> 0.086 ms
  纯GPU kernel时间: `0.026 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.086 ms`, `host_to_first_kernel_gap=0.073869`, `host_end_to_last_kernel_tail=0.015 ms`, `gpu_makespan=0.026 ms`, `gpu_kernel_sum=0.026 ms`
  开始时间(ns): `33429814`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2048], [2048, 128, 64], [2048, 1, 64]]}`
- `kv_b_proj` -> 0.237 ms
  纯GPU kernel时间: `0.147 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.237 ms`, `host_to_first_kernel_gap=0.084292`, `host_end_to_last_kernel_tail=0.120 ms`, `gpu_makespan=0.273 ms`, `gpu_kernel_sum=0.147 ms`
  开始时间(ns): `33781951`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[4096, 512]]}`
- `attn_mha` -> 0.182 ms
  纯GPU kernel时间: `0.257 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.182 ms`, `host_to_first_kernel_gap=0.138363`, `host_end_to_last_kernel_tail=0.234 ms`, `gpu_makespan=0.278 ms`, `gpu_kernel_sum=0.257 ms`
  开始时间(ns): `34727400`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2048, 128, 192], [4096, 128, 192], [4096, 128, 128]]}`
- `o_proj` -> 0.320 ms
  纯GPU kernel时间: `0.372 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.320 ms`, `host_to_first_kernel_gap=0.190746`, `host_end_to_last_kernel_tail=0.304 ms`, `gpu_makespan=0.434 ms`, `gpu_kernel_sum=0.372 ms`
  开始时间(ns): `34954313`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2048, 16384]]}`

## Layer 2 / prefill / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `2.117 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.117 ms`, `module_to_last_kernel=2.423 ms`, `host_to_first_kernel_gap=0.663167`, `host_end_to_last_kernel_tail=0.306 ms`, `gpu_makespan=1.760 ms`, `gpu_kernel_sum=1.000 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=2048`, `total_tokens=4096`, `chunked_req_prefix_len=2048`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[2048, 128, 192], [4096, 128, 192], [4096, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=2048`, `sum_prefix=2048`, `sum_seq_after=4096`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 512, 'prefix_len': 512, 'seq_len_after': 1024, 'prompt_len': 1024, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 512, 'prefix_len': 512, 'seq_len_after': 1024, 'prompt_len': 1024, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 512, 'prefix_len': 512, 'seq_len_after': 1024, 'prompt_len': 1024, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 512, 'prefix_len': 512, 'seq_len_after': 1024, 'prompt_len': 1024, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.229 ms
  纯GPU kernel时间: `0.074 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.229 ms`, `host_to_first_kernel_gap=0.511733`, `host_end_to_last_kernel_tail=0.358 ms`, `gpu_makespan=0.076 ms`, `gpu_kernel_sum=0.074 ms`
  开始时间(ns): `36366221`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2048, 7168]]}`
- `q_a_layernorm` -> 0.068 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.068 ms`, `host_to_first_kernel_gap=0.28284`, `host_end_to_last_kernel_tail=0.221 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `36670602`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2048, 1536]]}`
- `q_b_proj` -> 0.228 ms
  纯GPU kernel时间: `0.121 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.228 ms`, `host_to_first_kernel_gap=0.179317`, `host_end_to_last_kernel_tail=0.096 ms`, `gpu_makespan=0.145 ms`, `gpu_kernel_sum=0.121 ms`
  开始时间(ns): `36780781`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2048, 1536]]}`
- `kv_a_layernorm` -> 0.047 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.047 ms`, `host_to_first_kernel_gap=0.038375`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `37069019`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2048, 512]]}`
- `rotary_emb` -> 0.100 ms
  纯GPU kernel时间: `0.026 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.100 ms`, `host_to_first_kernel_gap=0.088099`, `host_end_to_last_kernel_tail=0.014 ms`, `gpu_makespan=0.026 ms`, `gpu_kernel_sum=0.026 ms`
  开始时间(ns): `37157983`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2048], [2048, 128, 64], [2048, 1, 64]]}`
- `kv_b_proj` -> 0.277 ms
  纯GPU kernel时间: `0.147 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.277 ms`, `host_to_first_kernel_gap=0.084322`, `host_end_to_last_kernel_tail=0.116 ms`, `gpu_makespan=0.309 ms`, `gpu_kernel_sum=0.147 ms`
  开始时间(ns): `37495872`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[4096, 512]]}`
- `attn_mha` -> 0.158 ms
  纯GPU kernel时间: `0.252 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.158 ms`, `host_to_first_kernel_gap=0.1335`, `host_end_to_last_kernel_tail=0.230 ms`, `gpu_makespan=0.254 ms`, `gpu_kernel_sum=0.252 ms`
  开始时间(ns): `37878309`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2048, 128, 192], [4096, 128, 192], [4096, 128, 128]]}`
- `o_proj` -> 0.236 ms
  纯GPU kernel时间: `0.369 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.236 ms`, `host_to_first_kernel_gap=0.186191`, `host_end_to_last_kernel_tail=0.320 ms`, `gpu_makespan=0.370 ms`, `gpu_kernel_sum=0.369 ms`
  开始时间(ns): `38081522`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2048, 16384]]}`

## Layer 3 / prefill / instance 1

- 执行序号: `4`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `2.025 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.025 ms`, `module_to_last_kernel=2.338 ms`, `host_to_first_kernel_gap=0.680141`, `host_end_to_last_kernel_tail=0.313 ms`, `gpu_makespan=1.658 ms`, `gpu_kernel_sum=0.997 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=2048`, `total_tokens=4096`, `chunked_req_prefix_len=2048`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[2048, 128, 192], [4096, 128, 192], [4096, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=2048`, `sum_prefix=2048`, `sum_seq_after=4096`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 512, 'prefix_len': 512, 'seq_len_after': 1024, 'prompt_len': 1024, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 512, 'prefix_len': 512, 'seq_len_after': 1024, 'prompt_len': 1024, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 512, 'prefix_len': 512, 'seq_len_after': 1024, 'prompt_len': 1024, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 512, 'prefix_len': 512, 'seq_len_after': 1024, 'prompt_len': 1024, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.224 ms
  纯GPU kernel时间: `0.074 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.224 ms`, `host_to_first_kernel_gap=0.545679`, `host_end_to_last_kernel_tail=0.397 ms`, `gpu_makespan=0.075 ms`, `gpu_kernel_sum=0.074 ms`
  开始时间(ns): `39345553`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2048, 7168]]}`
- `q_a_layernorm` -> 0.062 ms
  纯GPU kernel时间: `0.006 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.062 ms`, `host_to_first_kernel_gap=0.321194`, `host_end_to_last_kernel_tail=0.264 ms`, `gpu_makespan=0.006 ms`, `gpu_kernel_sum=0.006 ms`
  开始时间(ns): `39645686`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2048, 1536]]}`
- `q_b_proj` -> 0.208 ms
  纯GPU kernel时间: `0.119 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.208 ms`, `host_to_first_kernel_gap=0.236566`, `host_end_to_last_kernel_tail=0.149 ms`, `gpu_makespan=0.120 ms`, `gpu_kernel_sum=0.119 ms`
  开始时间(ns): `39737226`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2048, 1536]]}`
- `kv_a_layernorm` -> 0.042 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.042 ms`, `host_to_first_kernel_gap=0.088622`, `host_end_to_last_kernel_tail=0.052 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `40005202`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2048, 512]]}`
- `rotary_emb` -> 0.096 ms
  纯GPU kernel时间: `0.027 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.096 ms`, `host_to_first_kernel_gap=0.085678`, `host_end_to_last_kernel_tail=0.016 ms`, `gpu_makespan=0.027 ms`, `gpu_kernel_sum=0.027 ms`
  开始时间(ns): `40087922`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2048], [2048, 128, 64], [2048, 1, 64]]}`
- `kv_b_proj` -> 0.262 ms
  纯GPU kernel时间: `0.147 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.262 ms`, `host_to_first_kernel_gap=0.084215`, `host_end_to_last_kernel_tail=0.117 ms`, `gpu_makespan=0.295 ms`, `gpu_kernel_sum=0.147 ms`
  开始时间(ns): `40423241`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[4096, 512]]}`
- `attn_mha` -> 0.155 ms
  纯GPU kernel时间: `0.252 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.155 ms`, `host_to_first_kernel_gap=0.137208`, `host_end_to_last_kernel_tail=0.234 ms`, `gpu_makespan=0.252 ms`, `gpu_kernel_sum=0.252 ms`
  开始时间(ns): `40787624`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2048, 128, 192], [4096, 128, 192], [4096, 128, 128]]}`
- `o_proj` -> 0.240 ms
  纯GPU kernel时间: `0.369 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.240 ms`, `host_to_first_kernel_gap=0.195502`, `host_end_to_last_kernel_tail=0.326 ms`, `gpu_makespan=0.371 ms`, `gpu_kernel_sum=0.369 ms`
  开始时间(ns): `40982962`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2048, 16384]]}`

## Layer 4 / prefill / instance 1

- 执行序号: `5`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `2.259 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.259 ms`, `module_to_last_kernel=3.180 ms`, `host_to_first_kernel_gap=2.029129`, `host_end_to_last_kernel_tail=0.921 ms`, `gpu_makespan=1.150 ms`, `gpu_kernel_sum=1.000 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=2048`, `total_tokens=4096`, `chunked_req_prefix_len=2048`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[2048, 128, 192], [4096, 128, 192], [4096, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=2048`, `sum_prefix=2048`, `sum_seq_after=4096`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 512, 'prefix_len': 512, 'seq_len_after': 1024, 'prompt_len': 1024, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 512, 'prefix_len': 512, 'seq_len_after': 1024, 'prompt_len': 1024, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 512, 'prefix_len': 512, 'seq_len_after': 1024, 'prompt_len': 1024, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 512, 'prefix_len': 512, 'seq_len_after': 1024, 'prompt_len': 1024, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.325 ms
  纯GPU kernel时间: `0.073 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.325 ms`, `host_to_first_kernel_gap=1.849561`, `host_end_to_last_kernel_tail=1.600 ms`, `gpu_makespan=0.075 ms`, `gpu_kernel_sum=0.073 ms`
  开始时间(ns): `42900133`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2048, 7168]]}`
- `q_a_layernorm` -> 0.052 ms
  纯GPU kernel时间: `0.006 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.052 ms`, `host_to_first_kernel_gap=1.542627`, `host_end_to_last_kernel_tail=1.496 ms`, `gpu_makespan=0.006 ms`, `gpu_kernel_sum=0.006 ms`
  开始时间(ns): `43282235`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2048, 1536]]}`
- `q_b_proj` -> 0.229 ms
  纯GPU kernel时间: `0.119 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.229 ms`, `host_to_first_kernel_gap=1.467209`, `host_end_to_last_kernel_tail=1.359 ms`, `gpu_makespan=0.120 ms`, `gpu_kernel_sum=0.119 ms`
  开始时间(ns): `43364533`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2048, 1536]]}`
- `kv_a_layernorm` -> 0.083 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.083 ms`, `host_to_first_kernel_gap=1.267048`, `host_end_to_last_kernel_tail=1.189 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `43685206`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2048, 512]]}`
- `rotary_emb` -> 0.088 ms
  纯GPU kernel时间: `0.026 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.088 ms`, `host_to_first_kernel_gap=1.145621`, `host_end_to_last_kernel_tail=1.084 ms`, `gpu_makespan=0.026 ms`, `gpu_kernel_sum=0.026 ms`
  开始时间(ns): `43813705`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2048], [2048, 128, 64], [2048, 1, 64]]}`
- `kv_b_proj` -> 0.242 ms
  纯GPU kernel时间: `0.149 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.242 ms`, `host_to_first_kernel_gap=0.877051`, `host_end_to_last_kernel_tail=0.785 ms`, `gpu_makespan=0.150 ms`, `gpu_kernel_sum=0.149 ms`
  开始时间(ns): `44125571`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[4096, 512]]}`
- `attn_mha` -> 0.174 ms
  纯GPU kernel时间: `0.253 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.174 ms`, `host_to_first_kernel_gap=0.807057`, `host_end_to_last_kernel_tail=0.886 ms`, `gpu_makespan=0.253 ms`, `gpu_kernel_sum=0.253 ms`
  开始时间(ns): `44467245`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2048, 128, 192], [4096, 128, 192], [4096, 128, 128]]}`
- `o_proj` -> 0.264 ms
  纯GPU kernel时间: `0.368 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.264 ms`, `host_to_first_kernel_gap=0.82942`, `host_end_to_last_kernel_tail=0.935 ms`, `gpu_makespan=0.370 ms`, `gpu_kernel_sum=0.368 ms`
  开始时间(ns): `44701010`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2048, 16384]]}`

## Layer 0 / decode / instance 1

- 执行序号: `6`
- Collector对齐边界: `{'Module': 'model.model.layers.0.self_attn'}`
- 整块 MLA-module 时长: `2.168 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.168 ms`, `module_to_last_kernel=2.168 ms`, `host_to_first_kernel_gap=0.195291`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=1.966 ms`, `gpu_kernel_sum=0.117 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.351 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.351 ms`, `host_to_first_kernel_gap=0.11666`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.200 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `49275944`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4, 7168]]}`
- `q_a_layernorm` -> 0.057 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.057 ms`, `host_to_first_kernel_gap=0.045894`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `49711670`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4, 1536]]}`
- `kv_a_layernorm` -> 0.033 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.033 ms`, `host_to_first_kernel_gap=0.029046`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `49807014`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4, 512]]}`
- `q_b_proj` -> 0.258 ms
  纯GPU kernel时间: `0.020 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.258 ms`, `host_to_first_kernel_gap=0.090716`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.159 ms`, `gpu_kernel_sum=0.020 ms`
  开始时间(ns): `49879200`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4, 1536]]}`
- `rotary_emb` -> 0.091 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.091 ms`, `host_to_first_kernel_gap=0.077926`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `50321526`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4], [4, 128, 64], [4, 1, 64]]}`
- `attn_mqa` -> 0.402 ms
  纯GPU kernel时间: `0.026 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.402 ms`, `host_to_first_kernel_gap=0.102953`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.269 ms`, `gpu_kernel_sum=0.026 ms`
  开始时间(ns): `50473523`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4, 128, 512], [4, 1, 512], [4, 1, 512]]}`
- `o_proj` -> 0.318 ms
  纯GPU kernel时间: `0.050 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.318 ms`, `host_to_first_kernel_gap=0.108775`, `host_end_to_last_kernel_tail=0.012 ms`, `gpu_makespan=0.222 ms`, `gpu_kernel_sum=0.050 ms`
  开始时间(ns): `51028276`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4, 16384]]}`

## Layer 1 / decode / instance 1

- 执行序号: `7`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `1.773 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.773 ms`, `module_to_last_kernel=1.777 ms`, `host_to_first_kernel_gap=0.145199`, `host_end_to_last_kernel_tail=0.005 ms`, `gpu_makespan=1.632 ms`, `gpu_kernel_sum=0.114 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.213 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.213 ms`, `host_to_first_kernel_gap=0.079853`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.126 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `52342606`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4, 7168]]}`
- `q_a_layernorm` -> 0.048 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.048 ms`, `host_to_first_kernel_gap=0.042652`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `52625631`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4, 1536]]}`
- `kv_a_layernorm` -> 0.043 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.043 ms`, `host_to_first_kernel_gap=0.039805`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `52697342`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4, 512]]}`
- `q_b_proj` -> 0.211 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.211 ms`, `host_to_first_kernel_gap=0.075267`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.130 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `52774455`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4, 1536]]}`
- `rotary_emb` -> 0.086 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.086 ms`, `host_to_first_kernel_gap=0.074227`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `53155015`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4], [4, 128, 64], [4, 1, 64]]}`
- `attn_mqa` -> 0.327 ms
  纯GPU kernel时间: `0.026 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.327 ms`, `host_to_first_kernel_gap=0.112059`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.188 ms`, `gpu_kernel_sum=0.026 ms`
  开始时间(ns): `53289567`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4, 128, 512], [4, 1, 512], [4, 1, 512]]}`
- `o_proj` -> 0.278 ms
  纯GPU kernel时间: `0.050 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.278 ms`, `host_to_first_kernel_gap=0.100046`, `host_end_to_last_kernel_tail=0.024 ms`, `gpu_makespan=0.201 ms`, `gpu_kernel_sum=0.050 ms`
  开始时间(ns): `53753484`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4, 16384]]}`

## Layer 2 / decode / instance 1

- 执行序号: `8`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.774 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.774 ms`, `module_to_last_kernel=1.775 ms`, `host_to_first_kernel_gap=0.14792`, `host_end_to_last_kernel_tail=0.001 ms`, `gpu_makespan=1.627 ms`, `gpu_kernel_sum=0.114 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.209 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.209 ms`, `host_to_first_kernel_gap=0.079273`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.123 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `54982576`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4, 7168]]}`
- `q_a_layernorm` -> 0.047 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.047 ms`, `host_to_first_kernel_gap=0.041867`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `55251790`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4, 1536]]}`
- `kv_a_layernorm` -> 0.031 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.031 ms`, `host_to_first_kernel_gap=0.027232`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `55321849`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4, 512]]}`
- `q_b_proj` -> 0.212 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.212 ms`, `host_to_first_kernel_gap=0.074425`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.124 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `55396160`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4, 1536]]}`
- `rotary_emb` -> 0.086 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.086 ms`, `host_to_first_kernel_gap=0.073893`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `55782516`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4], [4, 128, 64], [4, 1, 64]]}`
- `attn_mqa` -> 0.310 ms
  纯GPU kernel时间: `0.026 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.310 ms`, `host_to_first_kernel_gap=0.099747`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.194 ms`, `gpu_kernel_sum=0.026 ms`
  开始时间(ns): `55916918`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4, 128, 512], [4, 1, 512], [4, 1, 512]]}`
- `o_proj` -> 0.309 ms
  纯GPU kernel时间: `0.049 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.309 ms`, `host_to_first_kernel_gap=0.101289`, `host_end_to_last_kernel_tail=0.019 ms`, `gpu_makespan=0.227 ms`, `gpu_kernel_sum=0.049 ms`
  开始时间(ns): `56360624`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4, 16384]]}`

## Layer 3 / decode / instance 1

- 执行序号: `9`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `1.877 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.877 ms`, `module_to_last_kernel=1.877 ms`, `host_to_first_kernel_gap=0.194233`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=1.682 ms`, `gpu_kernel_sum=0.113 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.293 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.293 ms`, `host_to_first_kernel_gap=0.108922`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.173 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `57641214`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4, 7168]]}`
- `q_a_layernorm` -> 0.050 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.050 ms`, `host_to_first_kernel_gap=0.045121`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `58002615`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4, 1536]]}`
- `kv_a_layernorm` -> 0.031 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.031 ms`, `host_to_first_kernel_gap=0.027546`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `58076414`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4, 512]]}`
- `q_b_proj` -> 0.221 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.221 ms`, `host_to_first_kernel_gap=0.08999`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.128 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `58141618`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4, 1536]]}`
- `rotary_emb` -> 0.105 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.105 ms`, `host_to_first_kernel_gap=0.088506`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `58530238`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4], [4, 128, 64], [4, 1, 64]]}`
- `attn_mqa` -> 0.309 ms
  纯GPU kernel时间: `0.026 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.309 ms`, `host_to_first_kernel_gap=0.10129`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.185 ms`, `gpu_kernel_sum=0.026 ms`
  开始时间(ns): `58684270`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4, 128, 512], [4, 1, 512], [4, 1, 512]]}`
- `o_proj` -> 0.282 ms
  纯GPU kernel时间: `0.049 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.282 ms`, `host_to_first_kernel_gap=0.098058`, `host_end_to_last_kernel_tail=0.023 ms`, `gpu_makespan=0.207 ms`, `gpu_kernel_sum=0.049 ms`
  开始时间(ns): `59126797`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4, 16384]]}`

## Layer 4 / decode / instance 1

- 执行序号: `10`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `1.892 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.892 ms`, `module_to_last_kernel=1.898 ms`, `host_to_first_kernel_gap=0.174219`, `host_end_to_last_kernel_tail=0.005 ms`, `gpu_makespan=1.724 ms`, `gpu_kernel_sum=0.114 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.299 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.299 ms`, `host_to_first_kernel_gap=0.101401`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.186 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `61061694`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4, 7168]]}`
- `q_a_layernorm` -> 0.051 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.051 ms`, `host_to_first_kernel_gap=0.04567`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `61427088`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4, 1536]]}`
- `kv_a_layernorm` -> 0.031 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.031 ms`, `host_to_first_kernel_gap=0.027366`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `61502320`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4, 512]]}`
- `q_b_proj` -> 0.259 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.259 ms`, `host_to_first_kernel_gap=0.11016`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.145 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `61567974`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4, 1536]]}`
- `rotary_emb` -> 0.089 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.089 ms`, `host_to_first_kernel_gap=0.076212`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `62000578`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4], [4, 128, 64], [4, 1, 64]]}`
- `attn_mqa` -> 0.309 ms
  纯GPU kernel时间: `0.026 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.309 ms`, `host_to_first_kernel_gap=0.098448`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.187 ms`, `gpu_kernel_sum=0.026 ms`
  开始时间(ns): `62136806`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4, 128, 512], [4, 1, 512], [4, 1, 512]]}`
- `o_proj` -> 0.285 ms
  纯GPU kernel时间: `0.050 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.285 ms`, `host_to_first_kernel_gap=0.106486`, `host_end_to_last_kernel_tail=0.023 ms`, `gpu_makespan=0.201 ms`, `gpu_kernel_sum=0.050 ms`
  开始时间(ns): `62579008`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4, 16384]]}`
