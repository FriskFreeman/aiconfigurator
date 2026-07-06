# MLA对齐解析摘要

源文件: `report.sqlite`

## 对齐原则

- `collector/sglang/collect_mla_module.py` 的 MLA module 计时边界是 `model.model.layers[test_layer].self_attn(...)`。
- 因此这里把 `nsys` 中每层的 `model.model.layers.X.self_attn` NVTX range 视为与 collector 对齐的 MLA-module 边界。
- 该区间内部的 `.self_attn.*` 子模块用于做 MLA 内部 breakdown。

## 运行摘要

- `prefill` 对齐成功层: `[0, 1, 2, 3, 4, 5]`
- `prefill` 被切分层: `[]`
- 若某层 `prefill` 出现多个 `self_attn` 实例，则说明 Engine 调度把一次前向切成了多块，已不再与 collector 的单次 MLA-module 采集严格一一对应。

## Layer 0 / prefill / instance 1

- 执行序号: `1`
- Collector对齐边界: `{'Module': 'model.model.layers.0.self_attn'}`
- 整块 MLA-module 时长: `4.943 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=4.943 ms`, `module_to_last_kernel=8.342 ms`, `host_to_first_kernel_gap=0.483764`, `host_end_to_last_kernel_tail=3.399 ms`, `gpu_makespan=7.858 ms`, `gpu_kernel_sum=3.786 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4128`, `total_tokens=37041`, `chunked_req_prefix_len=32913`, `current_chunked_req_prefix_len=32913`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[4128, 128, 192], [37041, 128, 192], [37041, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.458 ms
  纯GPU kernel时间: `0.144 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.458 ms`, `host_to_first_kernel_gap=0.156692`, `host_end_to_last_kernel_tail=0.067 ms`, `gpu_makespan=0.368 ms`, `gpu_kernel_sum=0.144 ms`
  开始时间(ns): `24275284`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4128, 7168]]}`
- `q_a_layernorm` -> 0.078 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.078 ms`, `host_to_first_kernel_gap=0.058213`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `24799266`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4128, 1536]]}`
- `q_b_proj` -> 0.272 ms
  纯GPU kernel时间: `0.242 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.272 ms`, `host_to_first_kernel_gap=0.098389`, `host_end_to_last_kernel_tail=0.204 ms`, `gpu_makespan=0.377 ms`, `gpu_kernel_sum=0.242 ms`
  开始时间(ns): `24913138`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4128, 1536]]}`
- `kv_a_layernorm` -> 0.047 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.047 ms`, `host_to_first_kernel_gap=0.13709`, `host_end_to_last_kernel_tail=0.099 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `25251749`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4128, 512]]}`
- `rotary_emb` -> 0.119 ms
  纯GPU kernel时间: `0.053 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.119 ms`, `host_to_first_kernel_gap=0.098368`, `host_end_to_last_kernel_tail=0.033 ms`, `gpu_makespan=0.053 ms`, `gpu_kernel_sum=0.053 ms`
  开始时间(ns): `25346023`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4128], [4128, 128, 64], [4128, 1, 64]]}`
- `kv_b_proj` -> 1.812 ms
  纯GPU kernel时间: `1.289 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=1.812 ms`, `host_to_first_kernel_gap=0.113127`, `host_end_to_last_kernel_tail=1.187 ms`, `gpu_makespan=2.885 ms`, `gpu_kernel_sum=1.289 ms`
  开始时间(ns): `26176638`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[37041, 512]]}`
- `attn_mha` -> 0.227 ms
  纯GPU kernel时间: `1.324 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.227 ms`, `host_to_first_kernel_gap=2.02069`, `host_end_to_last_kernel_tail=3.118 ms`, `gpu_makespan=1.324 ms`, `gpu_kernel_sum=1.324 ms`
  开始时间(ns): `28226286`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4128, 128, 192], [37041, 128, 192], [37041, 128, 128]]}`
- `o_proj` -> 0.366 ms
  纯GPU kernel时间: `0.716 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.366 ms`, `host_to_first_kernel_gap=3.066015`, `host_end_to_last_kernel_tail=3.418 ms`, `gpu_makespan=0.717 ms`, `gpu_kernel_sum=0.716 ms`
  开始时间(ns): `28506751`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4128, 16384]]}`

## Layer 1 / prefill / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `2.404 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.404 ms`, `module_to_last_kernel=9.885 ms`, `host_to_first_kernel_gap=4.932979`, `host_end_to_last_kernel_tail=7.482 ms`, `gpu_makespan=4.952 ms`, `gpu_kernel_sum=3.810 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4128`, `total_tokens=37041`, `chunked_req_prefix_len=32913`, `current_chunked_req_prefix_len=32913`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[4128, 128, 192], [37041, 128, 192], [37041, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.284 ms
  纯GPU kernel时间: `0.143 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.284 ms`, `host_to_first_kernel_gap=4.635558`, `host_end_to_last_kernel_tail=4.497 ms`, `gpu_makespan=0.146 ms`, `gpu_kernel_sum=0.143 ms`
  开始时间(ns): `30179156`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4128, 7168]]}`
- `q_a_layernorm` -> 0.063 ms
  纯GPU kernel时间: `0.010 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.063 ms`, `host_to_first_kernel_gap=4.435834`, `host_end_to_last_kernel_tail=4.382 ms`, `gpu_makespan=0.010 ms`, `gpu_kernel_sum=0.010 ms`
  开始时间(ns): `30525088`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4128, 1536]]}`
- `q_b_proj` -> 0.247 ms
  纯GPU kernel时间: `0.237 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.247 ms`, `host_to_first_kernel_gap=4.340978`, `host_end_to_last_kernel_tail=4.333 ms`, `gpu_makespan=0.239 ms`, `gpu_kernel_sum=0.237 ms`
  开始时间(ns): `30630824`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4128, 1536]]}`
- `kv_a_layernorm` -> 0.048 ms
  纯GPU kernel时间: `0.008 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.048 ms`, `host_to_first_kernel_gap=4.268312`, `host_end_to_last_kernel_tail=4.229 ms`, `gpu_makespan=0.008 ms`, `gpu_kernel_sum=0.008 ms`
  开始时间(ns): `30942465`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4128, 512]]}`
- `rotary_emb` -> 0.111 ms
  纯GPU kernel时间: `0.053 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.111 ms`, `host_to_first_kernel_gap=4.184713`, `host_end_to_last_kernel_tail=4.127 ms`, `gpu_makespan=0.053 ms`, `gpu_kernel_sum=0.053 ms`
  开始时间(ns): `31035536`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4128], [4128, 128, 64], [4128, 1, 64]]}`
- `kv_b_proj` -> 0.251 ms
  纯GPU kernel时间: `1.327 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.251 ms`, `host_to_first_kernel_gap=3.907824`, `host_end_to_last_kernel_tail=4.985 ms`, `gpu_makespan=1.328 ms`, `gpu_kernel_sum=1.327 ms`
  开始时间(ns): `31424073`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[37041, 512]]}`
- `attn_mha` -> 0.195 ms
  纯GPU kernel时间: `1.321 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.195 ms`, `host_to_first_kernel_gap=5.949371`, `host_end_to_last_kernel_tail=7.076 ms`, `gpu_makespan=1.321 ms`, `gpu_kernel_sum=1.321 ms`
  开始时间(ns): `31782523`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4128, 128, 192], [37041, 128, 192], [37041, 128, 128]]}`
- `o_proj` -> 0.244 ms
  纯GPU kernel时间: `0.711 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.244 ms`, `host_to_first_kernel_gap=7.030346`, `host_end_to_last_kernel_tail=7.499 ms`, `gpu_makespan=0.713 ms`, `gpu_kernel_sum=0.711 ms`
  开始时间(ns): `32024010`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4128, 16384]]}`

## Layer 2 / prefill / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `2.276 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.276 ms`, `module_to_last_kernel=14.144 ms`, `host_to_first_kernel_gap=9.080015`, `host_end_to_last_kernel_tail=11.868 ms`, `gpu_makespan=5.064 ms`, `gpu_kernel_sum=3.924 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4128`, `total_tokens=37041`, `chunked_req_prefix_len=32913`, `current_chunked_req_prefix_len=32913`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[4128, 128, 192], [37041, 128, 192], [37041, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.249 ms
  纯GPU kernel时间: `0.146 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.249 ms`, `host_to_first_kernel_gap=8.814694`, `host_end_to_last_kernel_tail=8.715 ms`, `gpu_makespan=0.149 ms`, `gpu_kernel_sum=0.146 ms`
  开始时间(ns): `33474346`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4128, 7168]]}`
- `q_a_layernorm` -> 0.058 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.058 ms`, `host_to_first_kernel_gap=8.659985`, `host_end_to_last_kernel_tail=8.612 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `33777951`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4128, 1536]]}`
- `q_b_proj` -> 0.218 ms
  纯GPU kernel时间: `0.237 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.218 ms`, `host_to_first_kernel_gap=8.566966`, `host_end_to_last_kernel_tail=8.587 ms`, `gpu_makespan=0.238 ms`, `gpu_kernel_sum=0.237 ms`
  开始时间(ns): `33881946`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4128, 1536]]}`
- `kv_a_layernorm` -> 0.046 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.046 ms`, `host_to_first_kernel_gap=8.524613`, `host_end_to_last_kernel_tail=8.488 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `34161642`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4128, 512]]}`
- `rotary_emb` -> 0.099 ms
  纯GPU kernel时间: `0.053 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.099 ms`, `host_to_first_kernel_gap=8.4414`, `host_end_to_last_kernel_tail=8.395 ms`, `gpu_makespan=0.053 ms`, `gpu_kernel_sum=0.053 ms`
  开始时间(ns): `34255831`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4128], [4128, 128, 64], [4128, 1, 64]]}`
- `kv_b_proj` -> 0.266 ms
  纯GPU kernel时间: `1.426 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.266 ms`, `host_to_first_kernel_gap=8.188029`, `host_end_to_last_kernel_tail=9.351 ms`, `gpu_makespan=1.428 ms`, `gpu_kernel_sum=1.426 ms`
  开始时间(ns): `34618098`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[37041, 512]]}`
- `attn_mha` -> 0.177 ms
  纯GPU kernel时间: `1.329 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.177 ms`, `host_to_first_kernel_gap=10.312986`, `host_end_to_last_kernel_tail=11.465 ms`, `gpu_makespan=1.329 ms`, `gpu_kernel_sum=1.329 ms`
  开始时间(ns): `34992338`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4128, 128, 192], [37041, 128, 192], [37041, 128, 128]]}`
- `o_proj` -> 0.254 ms
  纯GPU kernel时间: `0.715 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.254 ms`, `host_to_first_kernel_gap=11.422474`, `host_end_to_last_kernel_tail=11.885 ms`, `gpu_makespan=0.716 ms`, `gpu_kernel_sum=0.715 ms`
  开始时间(ns): `35214464`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4128, 16384]]}`

## Layer 3 / prefill / instance 1

- 执行序号: `4`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `2.250 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.250 ms`, `module_to_last_kernel=18.485 ms`, `host_to_first_kernel_gap=13.517752`, `host_end_to_last_kernel_tail=16.235 ms`, `gpu_makespan=4.967 ms`, `gpu_kernel_sum=3.827 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4128`, `total_tokens=37041`, `chunked_req_prefix_len=32913`, `current_chunked_req_prefix_len=32913`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[4128, 128, 192], [37041, 128, 192], [37041, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.249 ms
  纯GPU kernel时间: `0.143 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.249 ms`, `host_to_first_kernel_gap=13.256521`, `host_end_to_last_kernel_tail=13.151 ms`, `gpu_makespan=0.144 ms`, `gpu_kernel_sum=0.143 ms`
  开始时间(ns): `36621853`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4128, 7168]]}`
- `q_a_layernorm` -> 0.057 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.057 ms`, `host_to_first_kernel_gap=13.095752`, `host_end_to_last_kernel_tail=13.048 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `36926173`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4128, 1536]]}`
- `q_b_proj` -> 0.222 ms
  纯GPU kernel时间: `0.236 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.222 ms`, `host_to_first_kernel_gap=13.017963`, `host_end_to_last_kernel_tail=13.033 ms`, `gpu_makespan=0.237 ms`, `gpu_kernel_sum=0.236 ms`
  开始时间(ns): `37015674`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4128, 1536]]}`
- `kv_a_layernorm` -> 0.047 ms
  纯GPU kernel时间: `0.008 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.047 ms`, `host_to_first_kernel_gap=12.972255`, `host_end_to_last_kernel_tail=12.933 ms`, `gpu_makespan=0.008 ms`, `gpu_kernel_sum=0.008 ms`
  开始时间(ns): `37298694`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4128, 512]]}`
- `rotary_emb` -> 0.094 ms
  纯GPU kernel时间: `0.053 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.094 ms`, `host_to_first_kernel_gap=12.891594`, `host_end_to_last_kernel_tail=12.851 ms`, `gpu_makespan=0.053 ms`, `gpu_kernel_sum=0.053 ms`
  开始时间(ns): `37388699`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4128], [4128, 128, 64], [4128, 1, 64]]}`
- `kv_b_proj` -> 0.267 ms
  纯GPU kernel时间: `1.340 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.267 ms`, `host_to_first_kernel_gap=12.653121`, `host_end_to_last_kernel_tail=13.728 ms`, `gpu_makespan=1.342 ms`, `gpu_kernel_sum=1.340 ms`
  开始时间(ns): `37735684`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[37041, 512]]}`
- `attn_mha` -> 0.172 ms
  纯GPU kernel时间: `1.324 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.172 ms`, `host_to_first_kernel_gap=14.688282`, `host_end_to_last_kernel_tail=15.840 ms`, `gpu_makespan=1.324 ms`, `gpu_kernel_sum=1.324 ms`
  开始时间(ns): `38114280`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4128, 128, 192], [37041, 128, 192], [37041, 128, 128]]}`
- `o_proj` -> 0.263 ms
  纯GPU kernel时间: `0.714 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.263 ms`, `host_to_first_kernel_gap=15.797471`, `host_end_to_last_kernel_tail=16.251 ms`, `gpu_makespan=0.716 ms`, `gpu_kernel_sum=0.714 ms`
  开始时间(ns): `38331713`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4128, 16384]]}`

## Layer 4 / prefill / instance 1

- 执行序号: `5`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `2.459 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.459 ms`, `module_to_last_kernel=24.649 ms`, `host_to_first_kernel_gap=19.767546`, `host_end_to_last_kernel_tail=22.190 ms`, `gpu_makespan=4.882 ms`, `gpu_kernel_sum=3.740 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4128`, `total_tokens=37041`, `chunked_req_prefix_len=32913`, `current_chunked_req_prefix_len=32913`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[4128, 128, 192], [37041, 128, 192], [37041, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.359 ms
  纯GPU kernel时间: `0.142 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.359 ms`, `host_to_first_kernel_gap=19.486318`, `host_end_to_last_kernel_tail=19.272 ms`, `gpu_makespan=0.145 ms`, `gpu_kernel_sum=0.142 ms`
  开始时间(ns): `40605706`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4128, 7168]]}`
- `q_a_layernorm` -> 0.062 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.062 ms`, `host_to_first_kernel_gap=19.211264`, `host_end_to_last_kernel_tail=19.159 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `41025976`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4128, 1536]]}`
- `q_b_proj` -> 0.251 ms
  纯GPU kernel时间: `0.237 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.251 ms`, `host_to_first_kernel_gap=19.125642`, `host_end_to_last_kernel_tail=19.113 ms`, `gpu_makespan=0.238 ms`, `gpu_kernel_sum=0.237 ms`
  开始时间(ns): `41122286`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4128, 1536]]}`
- `kv_a_layernorm` -> 0.046 ms
  纯GPU kernel时间: `0.008 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.046 ms`, `host_to_first_kernel_gap=19.04784`, `host_end_to_last_kernel_tail=19.010 ms`, `gpu_makespan=0.008 ms`, `gpu_kernel_sum=0.008 ms`
  开始时间(ns): `41438103`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4128, 512]]}`
- `rotary_emb` -> 0.100 ms
  纯GPU kernel时间: `0.053 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.100 ms`, `host_to_first_kernel_gap=18.966501`, `host_end_to_last_kernel_tail=18.919 ms`, `gpu_makespan=0.053 ms`, `gpu_kernel_sum=0.053 ms`
  开始时间(ns): `41530034`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4128], [4128, 128, 64], [4128, 1, 64]]}`
- `kv_b_proj` -> 0.256 ms
  纯GPU kernel时间: `1.249 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.256 ms`, `host_to_first_kernel_gap=18.708668`, `host_end_to_last_kernel_tail=19.703 ms`, `gpu_makespan=1.251 ms`, `gpu_kernel_sum=1.249 ms`
  开始时间(ns): `41897403`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[37041, 512]]}`
- `attn_mha` -> 0.183 ms
  纯GPU kernel时间: `1.330 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.183 ms`, `host_to_first_kernel_gap=20.664219`, `host_end_to_last_kernel_tail=21.812 ms`, `gpu_makespan=1.330 ms`, `gpu_kernel_sum=1.330 ms`
  开始时间(ns): `42263833`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4128, 128, 192], [37041, 128, 192], [37041, 128, 128]]}`
- `o_proj` -> 0.271 ms
  纯GPU kernel时间: `0.712 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.271 ms`, `host_to_first_kernel_gap=21.76651`, `host_end_to_last_kernel_tail=22.208 ms`, `gpu_makespan=0.713 ms`, `gpu_kernel_sum=0.712 ms`
  开始时间(ns): `42494148`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4128, 16384]]}`

## Layer 5 / prefill / instance 1

- 执行序号: `6`
- Collector对齐边界: `{'Module': 'model.model.layers.5.self_attn'}`
- 整块 MLA-module 时长: `2.443 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.443 ms`, `module_to_last_kernel=30.798 ms`, `host_to_first_kernel_gap=25.86503`, `host_end_to_last_kernel_tail=28.354 ms`, `gpu_makespan=4.933 ms`, `gpu_kernel_sum=3.794 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4128`, `total_tokens=37041`, `chunked_req_prefix_len=32913`, `current_chunked_req_prefix_len=32913`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[4128, 128, 192], [37041, 128, 192], [37041, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.351 ms
  纯GPU kernel时间: `0.142 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.351 ms`, `host_to_first_kernel_gap=25.584544`, `host_end_to_last_kernel_tail=25.377 ms`, `gpu_makespan=0.144 ms`, `gpu_kernel_sum=0.142 ms`
  开始时间(ns): `44677162`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4128, 7168]]}`
- `q_a_layernorm` -> 0.064 ms
  纯GPU kernel时间: `0.010 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.064 ms`, `host_to_first_kernel_gap=25.317007`, `host_end_to_last_kernel_tail=25.262 ms`, `gpu_makespan=0.010 ms`, `gpu_kernel_sum=0.010 ms`
  开始时间(ns): `45089243`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4128, 1536]]}`
- `q_b_proj` -> 0.246 ms
  纯GPU kernel时间: `0.236 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.246 ms`, `host_to_first_kernel_gap=25.221876`, `host_end_to_last_kernel_tail=25.213 ms`, `gpu_makespan=0.237 ms`, `gpu_kernel_sum=0.236 ms`
  开始时间(ns): `45195254`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4128, 1536]]}`
- `kv_a_layernorm` -> 0.047 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.047 ms`, `host_to_first_kernel_gap=25.148389`, `host_end_to_last_kernel_tail=25.110 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `45505829`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4128, 512]]}`
- `rotary_emb` -> 0.099 ms
  纯GPU kernel时间: `0.052 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.099 ms`, `host_to_first_kernel_gap=25.069032`, `host_end_to_last_kernel_tail=25.023 ms`, `gpu_makespan=0.052 ms`, `gpu_kernel_sum=0.052 ms`
  开始时间(ns): `45595458`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4128], [4128, 128, 64], [4128, 1, 64]]}`
- `kv_b_proj` -> 0.250 ms
  纯GPU kernel时间: `1.311 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.250 ms`, `host_to_first_kernel_gap=24.811875`, `host_end_to_last_kernel_tail=25.874 ms`, `gpu_makespan=1.312 ms`, `gpu_kernel_sum=1.311 ms`
  开始时间(ns): `45963014`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[37041, 512]]}`
- `attn_mha` -> 0.192 ms
  纯GPU kernel时间: `1.323 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.192 ms`, `host_to_first_kernel_gap=26.833529`, `host_end_to_last_kernel_tail=27.965 ms`, `gpu_makespan=1.323 ms`, `gpu_kernel_sum=1.323 ms`
  开始时间(ns): `46323117`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4128, 128, 192], [37041, 128, 192], [37041, 128, 128]]}`
- `o_proj` -> 0.263 ms
  纯GPU kernel时间: `0.711 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.263 ms`, `host_to_first_kernel_gap=27.919995`, `host_end_to_last_kernel_tail=28.371 ms`, `gpu_makespan=0.713 ms`, `gpu_kernel_sum=0.711 ms`
  开始时间(ns): `46560969`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4128, 16384]]}`
