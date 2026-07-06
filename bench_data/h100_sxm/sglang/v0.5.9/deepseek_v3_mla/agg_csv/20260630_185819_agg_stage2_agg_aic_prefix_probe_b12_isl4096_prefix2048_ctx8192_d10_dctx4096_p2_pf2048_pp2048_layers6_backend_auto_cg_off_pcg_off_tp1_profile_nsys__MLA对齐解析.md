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
- 整块 MLA-module 时长: `2.520 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.520 ms`, `module_to_last_kernel=8.414 ms`, `host_to_first_kernel_gap=0.309162`, `host_end_to_last_kernel_tail=5.893 ms`, `gpu_makespan=8.105 ms`, `gpu_kernel_sum=5.582 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4106`, `total_tokens=49213`, `chunked_req_prefix_len=45107`, `current_chunked_req_prefix_len=45107`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[4106, 128, 192], [49213, 128, 192], [49213, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.361 ms
  纯GPU kernel时间: `0.143 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.361 ms`, `host_to_first_kernel_gap=0.123216`, `host_end_to_last_kernel_tail=0.072 ms`, `gpu_makespan=0.310 ms`, `gpu_kernel_sum=0.143 ms`
  开始时间(ns): `21980919`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4106, 7168]]}`
- `q_a_layernorm` -> 0.049 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.049 ms`, `host_to_first_kernel_gap=0.044164`, `host_end_to_last_kernel_tail=0.004 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `22398211`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4106, 1536]]}`
- `q_b_proj` -> 0.239 ms
  纯GPU kernel时间: `0.242 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.239 ms`, `host_to_first_kernel_gap=0.086891`, `host_end_to_last_kernel_tail=0.207 ms`, `gpu_makespan=0.359 ms`, `gpu_kernel_sum=0.242 ms`
  开始时间(ns): `22482780`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4106, 1536]]}`
- `kv_a_layernorm` -> 0.042 ms
  纯GPU kernel时间: `0.008 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.042 ms`, `host_to_first_kernel_gap=0.140229`, `host_end_to_last_kernel_tail=0.106 ms`, `gpu_makespan=0.008 ms`, `gpu_kernel_sum=0.008 ms`
  开始时间(ns): `22788130`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4106, 512]]}`
- `rotary_emb` -> 0.090 ms
  纯GPU kernel时间: `0.053 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.090 ms`, `host_to_first_kernel_gap=0.075903`, `host_end_to_last_kernel_tail=0.040 ms`, `gpu_makespan=0.053 ms`, `gpu_kernel_sum=0.053 ms`
  开始时间(ns): `22878216`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4106], [4106, 128, 64], [4106, 1, 64]]}`
- `kv_b_proj` -> 0.267 ms
  纯GPU kernel时间: `1.947 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.267 ms`, `host_to_first_kernel_gap=0.095168`, `host_end_to_last_kernel_tail=1.887 ms`, `gpu_makespan=2.059 ms`, `gpu_kernel_sum=1.947 ms`
  开始时间(ns): `23453863`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[49213, 512]]}`
- `attn_mha` -> 0.175 ms
  纯GPU kernel时间: `2.468 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.175 ms`, `host_to_first_kernel_gap=3.194981`, `host_end_to_last_kernel_tail=5.489 ms`, `gpu_makespan=2.468 ms`, `gpu_kernel_sum=2.468 ms`
  开始时间(ns): `23832417`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4106, 128, 192], [49213, 128, 192], [49213, 128, 128]]}`
- `o_proj` -> 0.247 ms
  纯GPU kernel时间: `0.710 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.247 ms`, `host_to_first_kernel_gap=5.443735`, `host_end_to_last_kernel_tail=5.908 ms`, `gpu_makespan=0.712 ms`, `gpu_kernel_sum=0.710 ms`
  开始时间(ns): `24053263`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4106, 16384]]}`

## Layer 1 / prefill / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `1.918 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.918 ms`, `module_to_last_kernel=14.584 ms`, `host_to_first_kernel_gap=7.569204`, `host_end_to_last_kernel_tail=12.665 ms`, `gpu_makespan=7.015 ms`, `gpu_kernel_sum=5.515 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4106`, `total_tokens=49213`, `chunked_req_prefix_len=45107`, `current_chunked_req_prefix_len=45107`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[4106, 128, 192], [49213, 128, 192], [49213, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.222 ms
  纯GPU kernel时间: `0.151 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.222 ms`, `host_to_first_kernel_gap=7.423843`, `host_end_to_last_kernel_tail=7.354 ms`, `gpu_makespan=0.153 ms`, `gpu_kernel_sum=0.151 ms`
  开始时间(ns): `25302306`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4106, 7168]]}`
- `q_a_layernorm` -> 0.046 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.046 ms`, `host_to_first_kernel_gap=7.299455`, `host_end_to_last_kernel_tail=7.263 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `25579910`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4106, 1536]]}`
- `q_b_proj` -> 0.201 ms
  纯GPU kernel时间: `0.236 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.201 ms`, `host_to_first_kernel_gap=7.234973`, `host_end_to_last_kernel_tail=7.272 ms`, `gpu_makespan=0.238 ms`, `gpu_kernel_sum=0.236 ms`
  开始时间(ns): `25655176`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4106, 1536]]}`
- `kv_a_layernorm` -> 0.043 ms
  纯GPU kernel时间: `0.008 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.043 ms`, `host_to_first_kernel_gap=7.199216`, `host_end_to_last_kernel_tail=7.164 ms`, `gpu_makespan=0.008 ms`, `gpu_kernel_sum=0.008 ms`
  开始时间(ns): `25928981`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4106, 512]]}`
- `rotary_emb` -> 0.075 ms
  纯GPU kernel时间: `0.052 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.075 ms`, `host_to_first_kernel_gap=7.127138`, `host_end_to_last_kernel_tail=7.104 ms`, `gpu_makespan=0.052 ms`, `gpu_kernel_sum=0.052 ms`
  开始时间(ns): `26010659`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4106], [4106, 128, 64], [4106, 1, 64]]}`
- `kv_b_proj` -> 0.236 ms
  纯GPU kernel时间: `1.874 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.236 ms`, `host_to_first_kernel_gap=6.959672`, `host_end_to_last_kernel_tail=8.599 ms`, `gpu_makespan=1.875 ms`, `gpu_kernel_sum=1.874 ms`
  开始时间(ns): `26299533`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[49213, 512]]}`
- `attn_mha` -> 0.141 ms
  纯GPU kernel时间: `2.474 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.141 ms`, `host_to_first_kernel_gap=9.919013`, `host_end_to_last_kernel_tail=12.252 ms`, `gpu_makespan=2.474 ms`, `gpu_kernel_sum=2.474 ms`
  开始时间(ns): `26635359`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4106, 128, 192], [49213, 128, 192], [49213, 128, 128]]}`
- `o_proj` -> 0.244 ms
  纯GPU kernel时间: `0.710 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.244 ms`, `host_to_first_kernel_gap=12.213486`, `host_end_to_last_kernel_tail=12.680 ms`, `gpu_makespan=0.711 ms`, `gpu_kernel_sum=0.710 ms`
  开始时间(ns): `26816181`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4106, 16384]]}`

## Layer 2 / prefill / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.869 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.869 ms`, `module_to_last_kernel=21.408 ms`, `host_to_first_kernel_gap=14.386432`, `host_end_to_last_kernel_tail=19.539 ms`, `gpu_makespan=7.022 ms`, `gpu_kernel_sum=5.521 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4106`, `total_tokens=49213`, `chunked_req_prefix_len=45107`, `current_chunked_req_prefix_len=45107`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[4106, 128, 192], [49213, 128, 192], [49213, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.225 ms
  纯GPU kernel时间: `0.145 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.225 ms`, `host_to_first_kernel_gap=14.242234`, `host_end_to_last_kernel_tail=14.163 ms`, `gpu_makespan=0.147 ms`, `gpu_kernel_sum=0.145 ms`
  开始时间(ns): `28015080`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4106, 7168]]}`
- `q_a_layernorm` -> 0.044 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.044 ms`, `host_to_first_kernel_gap=14.117993`, `host_end_to_last_kernel_tail=14.083 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `28286585`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4106, 1536]]}`
- `q_b_proj` -> 0.208 ms
  纯GPU kernel时间: `0.234 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.208 ms`, `host_to_first_kernel_gap=14.056067`, `host_end_to_last_kernel_tail=14.083 ms`, `gpu_makespan=0.236 ms`, `gpu_kernel_sum=0.234 ms`
  开始时间(ns): `28358783`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4106, 1536]]}`
- `kv_a_layernorm` -> 0.042 ms
  纯GPU kernel时间: `0.008 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.042 ms`, `host_to_first_kernel_gap=14.023321`, `host_end_to_last_kernel_tail=13.989 ms`, `gpu_makespan=0.008 ms`, `gpu_kernel_sum=0.008 ms`
  开始时间(ns): `28627017`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4106, 512]]}`
- `rotary_emb` -> 0.071 ms
  纯GPU kernel时间: `0.053 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.071 ms`, `host_to_first_kernel_gap=13.954179`, `host_end_to_last_kernel_tail=13.936 ms`, `gpu_makespan=0.053 ms`, `gpu_kernel_sum=0.053 ms`
  开始时间(ns): `28706111`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4106], [4106, 128, 64], [4106, 1, 64]]}`
- `kv_b_proj` -> 0.235 ms
  纯GPU kernel时间: `1.890 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.235 ms`, `host_to_first_kernel_gap=13.788901`, `host_end_to_last_kernel_tail=15.446 ms`, `gpu_makespan=1.892 ms`, `gpu_kernel_sum=1.890 ms`
  开始时间(ns): `28991389`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[49213, 512]]}`
- `attn_mha` -> 0.139 ms
  纯GPU kernel时间: `2.472 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.139 ms`, `host_to_first_kernel_gap=16.771086`, `host_end_to_last_kernel_tail=19.104 ms`, `gpu_makespan=2.472 ms`, `gpu_kernel_sum=2.472 ms`
  开始时间(ns): `29322708`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4106, 128, 192], [49213, 128, 192], [49213, 128, 128]]}`
- `o_proj` -> 0.225 ms
  纯GPU kernel时间: `0.710 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.225 ms`, `host_to_first_kernel_gap=19.067152`, `host_end_to_last_kernel_tail=19.554 ms`, `gpu_makespan=0.711 ms`, `gpu_kernel_sum=0.710 ms`
  开始时间(ns): `29500305`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4106, 16384]]}`

## Layer 3 / prefill / instance 1

- 执行序号: `4`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `1.846 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.846 ms`, `module_to_last_kernel=28.217 ms`, `host_to_first_kernel_gap=21.274951`, `host_end_to_last_kernel_tail=26.371 ms`, `gpu_makespan=6.942 ms`, `gpu_kernel_sum=5.439 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4106`, `total_tokens=49213`, `chunked_req_prefix_len=45107`, `current_chunked_req_prefix_len=45107`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[4106, 128, 192], [49213, 128, 192], [49213, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.223 ms
  纯GPU kernel时间: `0.146 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.223 ms`, `host_to_first_kernel_gap=21.129837`, `host_end_to_last_kernel_tail=21.054 ms`, `gpu_makespan=0.147 ms`, `gpu_kernel_sum=0.146 ms`
  开始时间(ns): `30667155`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4106, 7168]]}`
- `q_a_layernorm` -> 0.043 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.043 ms`, `host_to_first_kernel_gap=21.008999`, `host_end_to_last_kernel_tail=20.976 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `30935257`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4106, 1536]]}`
- `q_b_proj` -> 0.206 ms
  纯GPU kernel时间: `0.234 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.206 ms`, `host_to_first_kernel_gap=20.948636`, `host_end_to_last_kernel_tail=20.978 ms`, `gpu_makespan=0.235 ms`, `gpu_kernel_sum=0.234 ms`
  开始时间(ns): `31006884`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4106, 1536]]}`
- `kv_a_layernorm` -> 0.040 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.040 ms`, `host_to_first_kernel_gap=20.914855`, `host_end_to_last_kernel_tail=20.884 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `31275801`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4106, 512]]}`
- `rotary_emb` -> 0.072 ms
  纯GPU kernel时间: `0.053 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.072 ms`, `host_to_first_kernel_gap=20.846873`, `host_end_to_last_kernel_tail=20.828 ms`, `gpu_makespan=0.053 ms`, `gpu_kernel_sum=0.053 ms`
  开始时间(ns): `31354215`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4106], [4106, 128, 64], [4106, 1, 64]]}`
- `kv_b_proj` -> 0.224 ms
  纯GPU kernel时间: `1.811 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.224 ms`, `host_to_first_kernel_gap=20.683447`, `host_end_to_last_kernel_tail=22.273 ms`, `gpu_makespan=1.813 ms`, `gpu_kernel_sum=1.811 ms`
  开始时间(ns): `31637865`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[49213, 512]]}`
- `attn_mha` -> 0.137 ms
  纯GPU kernel时间: `2.468 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.137 ms`, `host_to_first_kernel_gap=23.598181`, `host_end_to_last_kernel_tail=25.930 ms`, `gpu_makespan=2.468 ms`, `gpu_kernel_sum=2.468 ms`
  开始时间(ns): `31960314`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4106, 128, 192], [49213, 128, 192], [49213, 128, 128]]}`
- `o_proj` -> 0.216 ms
  纯GPU kernel时间: `0.709 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.216 ms`, `host_to_first_kernel_gap=25.892168`, `host_end_to_last_kernel_tail=26.387 ms`, `gpu_makespan=0.711 ms`, `gpu_kernel_sum=0.709 ms`
  开始时间(ns): `32135703`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4106, 16384]]}`

## Layer 4 / prefill / instance 1

- 执行序号: `5`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `1.988 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.988 ms`, `module_to_last_kernel=37.275 ms`, `host_to_first_kernel_gap=30.233428`, `host_end_to_last_kernel_tail=35.287 ms`, `gpu_makespan=7.041 ms`, `gpu_kernel_sum=5.543 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4106`, `total_tokens=49213`, `chunked_req_prefix_len=45107`, `current_chunked_req_prefix_len=45107`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[4106, 128, 192], [49213, 128, 192], [49213, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.277 ms
  纯GPU kernel时间: `0.143 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.277 ms`, `host_to_first_kernel_gap=30.073503`, `host_end_to_last_kernel_tail=29.943 ms`, `gpu_makespan=0.146 ms`, `gpu_kernel_sum=0.143 ms`
  开始时间(ns): `33923486`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4106, 7168]]}`
- `q_a_layernorm` -> 0.046 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.046 ms`, `host_to_first_kernel_gap=29.891334`, `host_end_to_last_kernel_tail=29.854 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `34251735`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4106, 1536]]}`
- `q_b_proj` -> 0.218 ms
  纯GPU kernel时间: `0.235 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.218 ms`, `host_to_first_kernel_gap=29.824984`, `host_end_to_last_kernel_tail=29.843 ms`, `gpu_makespan=0.236 ms`, `gpu_kernel_sum=0.235 ms`
  开始时间(ns): `34328901`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4106, 1536]]}`
- `kv_a_layernorm` -> 0.040 ms
  纯GPU kernel时间: `0.008 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.040 ms`, `host_to_first_kernel_gap=29.781012`, `host_end_to_last_kernel_tail=29.749 ms`, `gpu_makespan=0.008 ms`, `gpu_kernel_sum=0.008 ms`
  开始时间(ns): `34608393`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4106, 512]]}`
- `rotary_emb` -> 0.077 ms
  纯GPU kernel时间: `0.053 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.077 ms`, `host_to_first_kernel_gap=29.714056`, `host_end_to_last_kernel_tail=29.690 ms`, `gpu_makespan=0.053 ms`, `gpu_kernel_sum=0.053 ms`
  开始时间(ns): `34685557`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4106], [4106, 128, 64], [4106, 1, 64]]}`
- `kv_b_proj` -> 0.238 ms
  纯GPU kernel时间: `1.911 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.238 ms`, `host_to_first_kernel_gap=29.536885`, `host_end_to_last_kernel_tail=31.211 ms`, `gpu_makespan=1.913 ms`, `gpu_kernel_sum=1.911 ms`
  开始时间(ns): `34983176`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[49213, 512]]}`
- `attn_mha` -> 0.141 ms
  纯GPU kernel时间: `2.471 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.141 ms`, `host_to_first_kernel_gap=32.529686`, `host_end_to_last_kernel_tail=34.860 ms`, `gpu_makespan=2.471 ms`, `gpu_kernel_sum=2.471 ms`
  开始时间(ns): `35321414`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4106, 128, 192], [49213, 128, 192], [49213, 128, 128]]}`
- `o_proj` -> 0.235 ms
  纯GPU kernel时间: `0.712 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.235 ms`, `host_to_first_kernel_gap=34.821438`, `host_end_to_last_kernel_tail=35.300 ms`, `gpu_makespan=0.714 ms`, `gpu_kernel_sum=0.712 ms`
  开始时间(ns): `35503326`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4106, 16384]]}`

## Layer 5 / prefill / instance 1

- 执行序号: `6`
- Collector对齐边界: `{'Module': 'model.model.layers.5.self_attn'}`
- 整块 MLA-module 时长: `1.954 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.954 ms`, `module_to_last_kernel=46.289 ms`, `host_to_first_kernel_gap=39.233656`, `host_end_to_last_kernel_tail=44.335 ms`, `gpu_makespan=7.056 ms`, `gpu_kernel_sum=5.559 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4106`, `total_tokens=49213`, `chunked_req_prefix_len=45107`, `current_chunked_req_prefix_len=45107`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[4106, 128, 192], [49213, 128, 192], [49213, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.278 ms
  纯GPU kernel时间: `0.143 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.278 ms`, `host_to_first_kernel_gap=39.08331`, `host_end_to_last_kernel_tail=38.950 ms`, `gpu_makespan=0.145 ms`, `gpu_kernel_sum=0.143 ms`
  开始时间(ns): `37196940`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4106, 7168]]}`
- `q_a_layernorm` -> 0.048 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.048 ms`, `host_to_first_kernel_gap=38.901091`, `host_end_to_last_kernel_tail=38.863 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `37524503`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4106, 1536]]}`
- `q_b_proj` -> 0.221 ms
  纯GPU kernel时间: `0.235 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.221 ms`, `host_to_first_kernel_gap=38.832562`, `host_end_to_last_kernel_tail=38.848 ms`, `gpu_makespan=0.236 ms`, `gpu_kernel_sum=0.235 ms`
  开始时间(ns): `37603816`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4106, 1536]]}`
- `kv_a_layernorm` -> 0.044 ms
  纯GPU kernel时间: `0.008 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.044 ms`, `host_to_first_kernel_gap=38.779906`, `host_end_to_last_kernel_tail=38.744 ms`, `gpu_makespan=0.008 ms`, `gpu_kernel_sum=0.008 ms`
  开始时间(ns): `37892152`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4106, 512]]}`
- `rotary_emb` -> 0.075 ms
  纯GPU kernel时间: `0.052 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.075 ms`, `host_to_first_kernel_gap=38.708354`, `host_end_to_last_kernel_tail=38.686 ms`, `gpu_makespan=0.052 ms`, `gpu_kernel_sum=0.052 ms`
  开始时间(ns): `37973112`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4106], [4106, 128, 64], [4106, 1, 64]]}`
- `kv_b_proj` -> 0.222 ms
  纯GPU kernel时间: `1.928 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.222 ms`, `host_to_first_kernel_gap=38.544432`, `host_end_to_last_kernel_tail=40.252 ms`, `gpu_makespan=1.930 ms`, `gpu_kernel_sum=1.928 ms`
  开始时间(ns): `38257866`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[49213, 512]]}`
- `attn_mha` -> 0.142 ms
  纯GPU kernel时间: `2.474 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.142 ms`, `host_to_first_kernel_gap=41.574998`, `host_end_to_last_kernel_tail=43.907 ms`, `gpu_makespan=2.474 ms`, `gpu_kernel_sum=2.474 ms`
  开始时间(ns): `38574371`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4106, 128, 192], [49213, 128, 192], [49213, 128, 128]]}`
- `o_proj` -> 0.231 ms
  纯GPU kernel时间: `0.710 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.231 ms`, `host_to_first_kernel_gap=43.86962`, `host_end_to_last_kernel_tail=44.350 ms`, `gpu_makespan=0.711 ms`, `gpu_kernel_sum=0.710 ms`
  开始时间(ns): `38755205`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4106, 16384]]}`
