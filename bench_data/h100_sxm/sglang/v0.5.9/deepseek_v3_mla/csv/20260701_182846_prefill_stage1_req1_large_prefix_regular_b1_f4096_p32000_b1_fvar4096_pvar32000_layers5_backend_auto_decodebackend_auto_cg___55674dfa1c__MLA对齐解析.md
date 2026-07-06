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
- 整块 MLA-module 时长: `1.857 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.857 ms`, `module_to_last_kernel=18.338 ms`, `host_to_first_kernel_gap=0.229664`, `host_end_to_last_kernel_tail=16.481 ms`, `gpu_makespan=18.108 ms`, `gpu_kernel_sum=16.326 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4096`, `total_tokens=36096`, `chunked_req_prefix_len=32000`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[4096, 128, 192], [36096, 128, 192], [36096, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=4096`, `sum_prefix=32000`, `sum_seq_after=36096`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 4096, 'prefix_len': 32000, 'seq_len_after': 36096, 'prompt_len': 36096, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.330 ms
  纯GPU kernel时间: `0.125 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.330 ms`, `host_to_first_kernel_gap=0.108049`, `host_end_to_last_kernel_tail=0.064 ms`, `gpu_makespan=0.286 ms`, `gpu_kernel_sum=0.125 ms`
  开始时间(ns): `28345726`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4096, 7168]]}`
- `q_a_layernorm` -> 0.036 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.036 ms`, `host_to_first_kernel_gap=0.03321`, `host_end_to_last_kernel_tail=0.007 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `28720980`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4096, 1536]]}`
- `q_b_proj` -> 0.178 ms
  纯GPU kernel时间: `0.240 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.178 ms`, `host_to_first_kernel_gap=0.069167`, `host_end_to_last_kernel_tail=0.215 ms`, `gpu_makespan=0.324 ms`, `gpu_kernel_sum=0.240 ms`
  开始时间(ns): `28781727`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4096, 1536]]}`
- `kv_a_layernorm` -> 0.032 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.032 ms`, `host_to_first_kernel_gap=0.167171`, `host_end_to_last_kernel_tail=0.145 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `29007339`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4096, 512]]}`
- `rotary_emb` -> 0.082 ms
  纯GPU kernel时间: `0.054 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.082 ms`, `host_to_first_kernel_gap=0.116323`, `host_end_to_last_kernel_tail=0.088 ms`, `gpu_makespan=0.054 ms`, `gpu_kernel_sum=0.054 ms`
  开始时间(ns): `29068523`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4096], [4096, 128, 64], [4096, 1, 64]]}`
- `kv_b_proj` -> 0.180 ms
  纯GPU kernel时间: `1.302 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.180 ms`, `host_to_first_kernel_gap=0.063131`, `host_end_to_last_kernel_tail=1.261 ms`, `gpu_makespan=1.378 ms`, `gpu_kernel_sum=1.302 ms`
  开始时间(ns): `29484274`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[36096, 512]]}`
- `attn_mha` -> 0.129 ms
  纯GPU kernel时间: `13.959 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.129 ms`, `host_to_first_kernel_gap=2.228864`, `host_end_to_last_kernel_tail=16.058 ms`, `gpu_makespan=13.959 ms`, `gpu_kernel_sum=13.959 ms`
  开始时间(ns): `29744137`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4096, 128, 192], [36096, 128, 192], [36096, 128, 128]]}`
- `o_proj` -> 0.166 ms
  纯GPU kernel时间: `0.627 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.166 ms`, `host_to_first_kernel_gap=16.028665`, `host_end_to_last_kernel_tail=16.491 ms`, `gpu_makespan=0.629 ms`, `gpu_kernel_sum=0.627 ms`
  开始时间(ns): `29904730`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4096, 16384]]}`

## Layer 1 / prefill / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `1.178 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.178 ms`, `module_to_last_kernel=35.723 ms`, `host_to_first_kernel_gap=18.258347`, `host_end_to_last_kernel_tail=34.546 ms`, `gpu_makespan=17.465 ms`, `gpu_kernel_sum=16.350 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4096`, `total_tokens=36096`, `chunked_req_prefix_len=32000`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[4096, 128, 192], [36096, 128, 192], [36096, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=4096`, `sum_prefix=32000`, `sum_seq_after=36096`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 4096, 'prefix_len': 32000, 'seq_len_after': 36096, 'prompt_len': 36096, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.149 ms
  纯GPU kernel时间: `0.124 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.149 ms`, `host_to_first_kernel_gap=18.17809`, `host_end_to_last_kernel_tail=18.156 ms`, `gpu_makespan=0.126 ms`, `gpu_kernel_sum=0.124 ms`
  开始时间(ns): `30706276`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4096, 7168]]}`
- `q_a_layernorm` -> 0.029 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.029 ms`, `host_to_first_kernel_gap=18.123769`, `host_end_to_last_kernel_tail=18.104 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `30887093`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4096, 1536]]}`
- `q_b_proj` -> 0.132 ms
  纯GPU kernel时间: `0.236 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.132 ms`, `host_to_first_kernel_gap=18.08537`, `host_end_to_last_kernel_tail=18.192 ms`, `gpu_makespan=0.238 ms`, `gpu_kernel_sum=0.236 ms`
  开始时间(ns): `30935988`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4096, 1536]]}`
- `kv_a_layernorm` -> 0.024 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.024 ms`, `host_to_first_kernel_gap=18.15535`, `host_end_to_last_kernel_tail=18.139 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `31103575`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4096, 512]]}`
- `rotary_emb` -> 0.052 ms
  纯GPU kernel时间: `0.054 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.052 ms`, `host_to_first_kernel_gap=18.116509`, `host_end_to_last_kernel_tail=18.119 ms`, `gpu_makespan=0.054 ms`, `gpu_kernel_sum=0.054 ms`
  开始时间(ns): `31152656`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4096], [4096, 128, 64], [4096, 1, 64]]}`
- `kv_b_proj` -> 0.131 ms
  纯GPU kernel时间: `1.276 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.131 ms`, `host_to_first_kernel_gap=18.039679`, `host_end_to_last_kernel_tail=19.185 ms`, `gpu_makespan=1.277 ms`, `gpu_kernel_sum=1.276 ms`
  开始时间(ns): `31340814`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[36096, 512]]}`
- `attn_mha` -> 0.086 ms
  纯GPU kernel时间: `13.987 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.086 ms`, `host_to_first_kernel_gap=20.173151`, `host_end_to_last_kernel_tail=34.074 ms`, `gpu_makespan=13.987 ms`, `gpu_kernel_sum=13.987 ms`
  开始时间(ns): `31529962`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4096, 128, 192], [36096, 128, 192], [36096, 128, 128]]}`
- `o_proj` -> 0.145 ms
  纯GPU kernel时间: `0.657 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.145 ms`, `host_to_first_kernel_gap=34.041506`, `host_end_to_last_kernel_tail=34.555 ms`, `gpu_makespan=0.658 ms`, `gpu_kernel_sum=0.657 ms`
  开始时间(ns): `31649649`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4096, 16384]]}`

## Layer 2 / prefill / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.121 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.121 ms`, `module_to_last_kernel=54.546 ms`, `host_to_first_kernel_gap=36.432122`, `host_end_to_last_kernel_tail=53.425 ms`, `gpu_makespan=18.114 ms`, `gpu_kernel_sum=17.000 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4096`, `total_tokens=36096`, `chunked_req_prefix_len=32000`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[4096, 128, 192], [36096, 128, 192], [36096, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=4096`, `sum_prefix=32000`, `sum_seq_after=36096`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 4096, 'prefix_len': 32000, 'seq_len_after': 36096, 'prompt_len': 36096, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.129 ms
  纯GPU kernel时间: `0.127 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.129 ms`, `host_to_first_kernel_gap=36.364389`, `host_end_to_last_kernel_tail=36.364 ms`, `gpu_makespan=0.129 ms`, `gpu_kernel_sum=0.127 ms`
  开始时间(ns): `32374377`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4096, 7168]]}`
- `q_a_layernorm` -> 0.028 ms
  纯GPU kernel时间: `0.010 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.028 ms`, `host_to_first_kernel_gap=36.333675`, `host_end_to_last_kernel_tail=36.316 ms`, `gpu_makespan=0.010 ms`, `gpu_kernel_sum=0.010 ms`
  开始时间(ns): `32534115`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4096, 1536]]}`
- `q_b_proj` -> 0.126 ms
  纯GPU kernel时间: `0.243 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.126 ms`, `host_to_first_kernel_gap=36.298236`, `host_end_to_last_kernel_tail=36.416 ms`, `gpu_makespan=0.244 ms`, `gpu_kernel_sum=0.243 ms`
  开始时间(ns): `32580978`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4096, 1536]]}`
- `kv_a_layernorm` -> 0.024 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.024 ms`, `host_to_first_kernel_gap=36.380219`, `host_end_to_last_kernel_tail=36.365 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `32743058`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4096, 512]]}`
- `rotary_emb` -> 0.048 ms
  纯GPU kernel时间: `0.054 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.048 ms`, `host_to_first_kernel_gap=36.341657`, `host_end_to_last_kernel_tail=36.347 ms`, `gpu_makespan=0.054 ms`, `gpu_kernel_sum=0.054 ms`
  开始时间(ns): `32793236`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4096], [4096, 128, 64], [4096, 1, 64]]}`
- `kv_b_proj` -> 0.141 ms
  纯GPU kernel时间: `1.424 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.141 ms`, `host_to_first_kernel_gap=36.271604`, `host_end_to_last_kernel_tail=37.557 ms`, `gpu_makespan=1.426 ms`, `gpu_kernel_sum=1.424 ms`
  开始时间(ns): `32973689`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[36096, 512]]}`
- `attn_mha` -> 0.085 ms
  纯GPU kernel时间: `14.410 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.085 ms`, `host_to_first_kernel_gap=38.540739`, `host_end_to_last_kernel_tail=52.866 ms`, `gpu_makespan=14.410 ms`, `gpu_kernel_sum=14.410 ms`
  开始时间(ns): `33175622`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4096, 128, 192], [36096, 128, 192], [36096, 128, 128]]}`
- `o_proj` -> 0.134 ms
  纯GPU kernel时间: `0.722 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.134 ms`, `host_to_first_kernel_gap=52.844157`, `host_end_to_last_kernel_tail=53.433 ms`, `gpu_makespan=0.724 ms`, `gpu_kernel_sum=0.722 ms`
  开始时间(ns): `33284629`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4096, 16384]]}`

## Layer 3 / prefill / instance 1

- 执行序号: `4`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `1.105 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.105 ms`, `module_to_last_kernel=73.820 ms`, `host_to_first_kernel_gap=55.52171`, `host_end_to_last_kernel_tail=72.715 ms`, `gpu_makespan=18.299 ms`, `gpu_kernel_sum=17.183 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4096`, `total_tokens=36096`, `chunked_req_prefix_len=32000`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[4096, 128, 192], [36096, 128, 192], [36096, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=4096`, `sum_prefix=32000`, `sum_seq_after=36096`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 4096, 'prefix_len': 32000, 'seq_len_after': 36096, 'prompt_len': 36096, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.126 ms
  纯GPU kernel时间: `0.136 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.126 ms`, `host_to_first_kernel_gap=55.456231`, `host_end_to_last_kernel_tail=55.468 ms`, `gpu_makespan=0.138 ms`, `gpu_kernel_sum=0.136 ms`
  开始时间(ns): `33982629`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4096, 7168]]}`
- `q_a_layernorm` -> 0.028 ms
  纯GPU kernel时间: `0.010 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.028 ms`, `host_to_first_kernel_gap=55.439582`, `host_end_to_last_kernel_tail=55.422 ms`, `gpu_makespan=0.010 ms`, `gpu_kernel_sum=0.010 ms`
  开始时间(ns): `34137358`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4096, 1536]]}`
- `q_b_proj` -> 0.117 ms
  纯GPU kernel时间: `0.262 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.117 ms`, `host_to_first_kernel_gap=55.406771`, `host_end_to_last_kernel_tail=55.554 ms`, `gpu_makespan=0.264 ms`, `gpu_kernel_sum=0.262 ms`
  开始时间(ns): `34182777`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4096, 1536]]}`
- `kv_a_layernorm` -> 0.026 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.026 ms`, `host_to_first_kernel_gap=55.519614`, `host_end_to_last_kernel_tail=55.503 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `34333390`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4096, 512]]}`
- `rotary_emb` -> 0.045 ms
  纯GPU kernel时间: `0.053 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.045 ms`, `host_to_first_kernel_gap=55.482882`, `host_end_to_last_kernel_tail=55.491 ms`, `gpu_makespan=0.053 ms`, `gpu_kernel_sum=0.053 ms`
  开始时间(ns): `34381674`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4096], [4096, 128, 64], [4096, 1, 64]]}`
- `kv_b_proj` -> 0.148 ms
  纯GPU kernel时间: `1.393 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.148 ms`, `host_to_first_kernel_gap=55.421765`, `host_end_to_last_kernel_tail=56.669 ms`, `gpu_makespan=1.395 ms`, `gpu_kernel_sum=1.393 ms`
  开始时间(ns): `34553543`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[36096, 512]]}`
- `attn_mha` -> 0.087 ms
  纯GPU kernel时间: `14.596 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.087 ms`, `host_to_first_kernel_gap=57.650585`, `host_end_to_last_kernel_tail=72.159 ms`, `gpu_makespan=14.596 ms`, `gpu_kernel_sum=14.596 ms`
  开始时间(ns): `34763695`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4096, 128, 192], [36096, 128, 192], [36096, 128, 128]]}`
- `o_proj` -> 0.137 ms
  纯GPU kernel时间: `0.722 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.137 ms`, `host_to_first_kernel_gap=72.136345`, `host_end_to_last_kernel_tail=72.724 ms`, `gpu_makespan=0.725 ms`, `gpu_kernel_sum=0.722 ms`
  开始时间(ns): `34876151`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4096, 16384]]}`

## Layer 4 / prefill / instance 1

- 执行序号: `5`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `1.184 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.184 ms`, `module_to_last_kernel=95.378 ms`, `host_to_first_kernel_gap=77.080926`, `host_end_to_last_kernel_tail=94.194 ms`, `gpu_makespan=18.297 ms`, `gpu_kernel_sum=17.182 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4096`, `total_tokens=36096`, `chunked_req_prefix_len=32000`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[4096, 128, 192], [36096, 128, 192], [36096, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=4096`, `sum_prefix=32000`, `sum_seq_after=36096`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 4096, 'prefix_len': 32000, 'seq_len_after': 36096, 'prompt_len': 36096, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.171 ms
  纯GPU kernel时间: `0.136 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.171 ms`, `host_to_first_kernel_gap=77.004588`, `host_end_to_last_kernel_tail=76.972 ms`, `gpu_makespan=0.139 ms`, `gpu_kernel_sum=0.136 ms`
  开始时间(ns): `36086938`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4096, 7168]]}`
- `q_a_layernorm` -> 0.030 ms
  纯GPU kernel时间: `0.010 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.030 ms`, `host_to_first_kernel_gap=76.940434`, `host_end_to_last_kernel_tail=76.920 ms`, `gpu_makespan=0.010 ms`, `gpu_kernel_sum=0.010 ms`
  开始时间(ns): `36290388`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4096, 1536]]}`
- `q_b_proj` -> 0.133 ms
  纯GPU kernel时间: `0.261 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.133 ms`, `host_to_first_kernel_gap=76.901605`, `host_end_to_last_kernel_tail=77.032 ms`, `gpu_makespan=0.263 ms`, `gpu_kernel_sum=0.261 ms`
  开始时间(ns): `36340545`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4096, 1536]]}`
- `kv_a_layernorm` -> 0.026 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.026 ms`, `host_to_first_kernel_gap=76.995606`, `host_end_to_last_kernel_tail=76.978 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `36508848`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4096, 512]]}`
- `rotary_emb` -> 0.048 ms
  纯GPU kernel时间: `0.054 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.048 ms`, `host_to_first_kernel_gap=76.957568`, `host_end_to_last_kernel_tail=76.964 ms`, `gpu_makespan=0.054 ms`, `gpu_kernel_sum=0.054 ms`
  开始时间(ns): `36558182`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4096], [4096, 128, 64], [4096, 1, 64]]}`
- `kv_b_proj` -> 0.134 ms
  纯GPU kernel时间: `1.397 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.134 ms`, `host_to_first_kernel_gap=76.883668`, `host_end_to_last_kernel_tail=78.149 ms`, `gpu_makespan=1.400 ms`, `gpu_kernel_sum=1.397 ms`
  开始时间(ns): `36743441`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[36096, 512]]}`
- `attn_mha` -> 0.088 ms
  纯GPU kernel时间: `14.592 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.088 ms`, `host_to_first_kernel_gap=79.134709`, `host_end_to_last_kernel_tail=93.639 ms`, `gpu_makespan=14.592 ms`, `gpu_kernel_sum=14.592 ms`
  开始时间(ns): `36937484`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4096, 128, 192], [36096, 128, 192], [36096, 128, 128]]}`
- `o_proj` -> 0.134 ms
  纯GPU kernel时间: `0.722 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.134 ms`, `host_to_first_kernel_gap=93.616143`, `host_end_to_last_kernel_tail=94.205 ms`, `gpu_makespan=0.723 ms`, `gpu_kernel_sum=0.722 ms`
  开始时间(ns): `37049723`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4096, 16384]]}`

## Layer 0 / decode / instance 1

- 执行序号: `6`
- Collector对齐边界: `{'Module': 'model.model.layers.0.self_attn'}`
- 整块 MLA-module 时长: `1.136 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.136 ms`, `module_to_last_kernel=97.998 ms`, `host_to_first_kernel_gap=97.823284`, `host_end_to_last_kernel_tail=96.861 ms`, `gpu_makespan=0.175 ms`, `gpu_kernel_sum=0.141 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.190 ms
  纯GPU kernel时间: `0.016 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.190 ms`, `host_to_first_kernel_gap=97.7807`, `host_end_to_last_kernel_tail=97.607 ms`, `gpu_makespan=0.017 ms`, `gpu_kernel_sum=0.016 ms`
  开始时间(ns): `39709443`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.030 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.030 ms`, `host_to_first_kernel_gap=97.571132`, `host_end_to_last_kernel_tail=97.544 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `39936099`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.020 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.020 ms`, `host_to_first_kernel_gap=97.528536`, `host_end_to_last_kernel_tail=97.511 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `39980999`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.136 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.136 ms`, `host_to_first_kernel_gap=97.489204`, `host_end_to_last_kernel_tail=97.374 ms`, `gpu_makespan=0.020 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `40023595`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.049 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.049 ms`, `host_to_first_kernel_gap=97.281967`, `host_end_to_last_kernel_tail=97.235 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `40263792`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.185 ms
  纯GPU kernel时间: `0.050 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.185 ms`, `host_to_first_kernel_gap=97.20849`, `host_end_to_last_kernel_tail=97.074 ms`, `gpu_makespan=0.051 ms`, `gpu_kernel_sum=0.050 ms`
  开始时间(ns): `40340309`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.178 ms
  纯GPU kernel时间: `0.051 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.178 ms`, `host_to_first_kernel_gap=96.998894`, `host_end_to_last_kernel_tail=96.873 ms`, `gpu_makespan=0.052 ms`, `gpu_kernel_sum=0.051 ms`
  开始时间(ns): `40613745`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 1 / decode / instance 1

- 执行序号: `7`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `0.994 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=0.994 ms`, `module_to_last_kernel=96.706 ms`, `host_to_first_kernel_gap=96.532289`, `host_end_to_last_kernel_tail=95.712 ms`, `gpu_makespan=0.173 ms`, `gpu_kernel_sum=0.142 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.122 ms
  纯GPU kernel时间: `0.017 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.122 ms`, `host_to_first_kernel_gap=96.495118`, `host_end_to_last_kernel_tail=96.392 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.017 ms`
  开始时间(ns): `41333520`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.026 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.026 ms`, `host_to_first_kernel_gap=96.358565`, `host_end_to_last_kernel_tail=96.334 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `41488377`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.018 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.018 ms`, `host_to_first_kernel_gap=96.318872`, `host_end_to_last_kernel_tail=96.303 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `41530150`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.137 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.137 ms`, `host_to_first_kernel_gap=96.283887`, `host_end_to_last_kernel_tail=96.166 ms`, `gpu_makespan=0.019 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `41568559`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.048 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.048 ms`, `host_to_first_kernel_gap=96.087684`, `host_end_to_last_kernel_tail=96.041 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `41796282`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.168 ms
  纯GPU kernel时间: `0.049 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.168 ms`, `host_to_first_kernel_gap=96.015157`, `host_end_to_last_kernel_tail=95.897 ms`, `gpu_makespan=0.050 ms`, `gpu_kernel_sum=0.049 ms`
  开始时间(ns): `41871881`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.160 ms
  纯GPU kernel时间: `0.052 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.160 ms`, `host_to_first_kernel_gap=95.832827`, `host_end_to_last_kernel_tail=95.725 ms`, `gpu_makespan=0.053 ms`, `gpu_kernel_sum=0.052 ms`
  开始时间(ns): `42116099`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 2 / decode / instance 1

- 执行序号: `8`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `0.975 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=0.975 ms`, `module_to_last_kernel=95.554 ms`, `host_to_first_kernel_gap=95.377232`, `host_end_to_last_kernel_tail=94.579 ms`, `gpu_makespan=0.177 ms`, `gpu_kernel_sum=0.145 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.125 ms
  纯GPU kernel时间: `0.017 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.125 ms`, `host_to_first_kernel_gap=95.338632`, `host_end_to_last_kernel_tail=95.233 ms`, `gpu_makespan=0.019 ms`, `gpu_kernel_sum=0.017 ms`
  开始时间(ns): `42828854`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.028 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.028 ms`, `host_to_first_kernel_gap=95.199215`, `host_end_to_last_kernel_tail=95.174 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `42987407`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.021 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.021 ms`, `host_to_first_kernel_gap=95.159323`, `host_end_to_last_kernel_tail=95.141 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `43029667`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.119 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.119 ms`, `host_to_first_kernel_gap=95.12121`, `host_end_to_last_kernel_tail=95.022 ms`, `gpu_makespan=0.020 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `43071108`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.048 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.048 ms`, `host_to_first_kernel_gap=94.948729`, `host_end_to_last_kernel_tail=94.902 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `43274629`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.165 ms
  纯GPU kernel时间: `0.050 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.165 ms`, `host_to_first_kernel_gap=94.877463`, `host_end_to_last_kernel_tail=94.765 ms`, `gpu_makespan=0.052 ms`, `gpu_kernel_sum=0.050 ms`
  开始时间(ns): `43348903`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.168 ms
  纯GPU kernel时间: `0.052 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.168 ms`, `host_to_first_kernel_gap=94.705331`, `host_end_to_last_kernel_tail=94.591 ms`, `gpu_makespan=0.053 ms`, `gpu_kernel_sum=0.052 ms`
  开始时间(ns): `43585483`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 3 / decode / instance 1

- 执行序号: `9`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `0.945 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=0.945 ms`, `module_to_last_kernel=94.432 ms`, `host_to_first_kernel_gap=94.256938`, `host_end_to_last_kernel_tail=93.487 ms`, `gpu_makespan=0.175 ms`, `gpu_kernel_sum=0.143 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.122 ms
  纯GPU kernel时间: `0.017 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.122 ms`, `host_to_first_kernel_gap=94.21902`, `host_end_to_last_kernel_tail=94.115 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.017 ms`
  开始时间(ns): `44289681`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.026 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.026 ms`, `host_to_first_kernel_gap=94.080068`, `host_end_to_last_kernel_tail=94.057 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `44446681`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.018 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.018 ms`, `host_to_first_kernel_gap=94.042755`, `host_end_to_last_kernel_tail=94.027 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `44486362`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.123 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.123 ms`, `host_to_first_kernel_gap=94.006593`, `host_end_to_last_kernel_tail=93.904 ms`, `gpu_makespan=0.020 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `44526748`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.047 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.047 ms`, `host_to_first_kernel_gap=93.831269`, `host_end_to_last_kernel_tail=93.786 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `44732632`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.166 ms
  纯GPU kernel时间: `0.049 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.166 ms`, `host_to_first_kernel_gap=93.760596`, `host_end_to_last_kernel_tail=93.645 ms`, `gpu_makespan=0.051 ms`, `gpu_kernel_sum=0.049 ms`
  开始时间(ns): `44806377`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.146 ms
  纯GPU kernel时间: `0.052 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.146 ms`, `host_to_first_kernel_gap=93.589147`, `host_end_to_last_kernel_tail=93.496 ms`, `gpu_makespan=0.053 ms`, `gpu_kernel_sum=0.052 ms`
  开始时间(ns): `45041506`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 4 / decode / instance 1

- 执行序号: `10`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `1.031 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.031 ms`, `module_to_last_kernel=92.987 ms`, `host_to_first_kernel_gap=92.813377`, `host_end_to_last_kernel_tail=91.956 ms`, `gpu_makespan=0.174 ms`, `gpu_kernel_sum=0.142 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.163 ms
  纯GPU kernel时间: `0.017 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.163 ms`, `host_to_first_kernel_gap=92.773846`, `host_end_to_last_kernel_tail=92.629 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.017 ms`
  开始时间(ns): `46118663`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.034 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.034 ms`, `host_to_first_kernel_gap=92.59468`, `host_end_to_last_kernel_tail=92.563 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `46316005`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.018 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.018 ms`, `host_to_first_kernel_gap=92.549817`, `host_end_to_last_kernel_tail=92.533 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `46363108`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.126 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.126 ms`, `host_to_first_kernel_gap=92.512624`, `host_end_to_last_kernel_tail=92.406 ms`, `gpu_makespan=0.019 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `46403501`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.055 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.055 ms`, `host_to_first_kernel_gap=92.327002`, `host_end_to_last_kernel_tail=92.274 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `46619651`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.167 ms
  纯GPU kernel时间: `0.049 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.167 ms`, `host_to_first_kernel_gap=92.250812`, `host_end_to_last_kernel_tail=92.135 ms`, `gpu_makespan=0.051 ms`, `gpu_kernel_sum=0.049 ms`
  开始时间(ns): `46700385`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.158 ms
  纯GPU kernel时间: `0.051 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.158 ms`, `host_to_first_kernel_gap=92.071489`, `host_end_to_last_kernel_tail=91.966 ms`, `gpu_makespan=0.052 ms`, `gpu_kernel_sum=0.051 ms`
  开始时间(ns): `46942139`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`
