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
- 模块时延拆解: `host_total=2.520 ms`, `module_to_last_kernel=5.723 ms`, `host_to_first_kernel_gap=0.30413`, `host_end_to_last_kernel_tail=3.203 ms`, `gpu_makespan=5.419 ms`, `gpu_kernel_sum=3.074 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=2064`, `total_tokens=34905`, `chunked_req_prefix_len=32841`, `current_chunked_req_prefix_len=32841`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[2064, 128, 192], [34905, 128, 192], [34905, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.354 ms
  纯GPU kernel时间: `0.080 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.354 ms`, `host_to_first_kernel_gap=0.12662`, `host_end_to_last_kernel_tail=0.028 ms`, `gpu_makespan=0.255 ms`, `gpu_kernel_sum=0.080 ms`
  开始时间(ns): `21034207`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2064, 7168]]}`
- `q_a_layernorm` -> 0.051 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.051 ms`, `host_to_first_kernel_gap=0.04476`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `21443715`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2064, 1536]]}`
- `q_b_proj` -> 0.248 ms
  纯GPU kernel时间: `0.136 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.248 ms`, `host_to_first_kernel_gap=0.08984`, `host_end_to_last_kernel_tail=0.104 ms`, `gpu_makespan=0.261 ms`, `gpu_kernel_sum=0.136 ms`
  开始时间(ns): `21530315`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2064, 1536]]}`
- `kv_a_layernorm` -> 0.046 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.046 ms`, `host_to_first_kernel_gap=0.039819`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `21845231`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2064, 512]]}`
- `rotary_emb` -> 0.081 ms
  纯GPU kernel时间: `0.027 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.081 ms`, `host_to_first_kernel_gap=0.066817`, `host_end_to_last_kernel_tail=0.013 ms`, `gpu_makespan=0.027 ms`, `gpu_kernel_sum=0.027 ms`
  开始时间(ns): `21933369`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2064], [2064, 128, 64], [2064, 1, 64]]}`
- `kv_b_proj` -> 0.279 ms
  纯GPU kernel时间: `1.253 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.279 ms`, `host_to_first_kernel_gap=0.096547`, `host_end_to_last_kernel_tail=1.201 ms`, `gpu_makespan=1.384 ms`, `gpu_kernel_sum=1.253 ms`
  开始时间(ns): `22518806`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[34905, 512]]}`
- `attn_mha` -> 0.155 ms
  纯GPU kernel时间: `1.194 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.155 ms`, `host_to_first_kernel_gap=2.102594`, `host_end_to_last_kernel_tail=3.141 ms`, `gpu_makespan=1.194 ms`, `gpu_kernel_sum=1.194 ms`
  开始时间(ns): `22905747`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2064, 128, 192], [34905, 128, 192], [34905, 128, 128]]}`
- `o_proj` -> 0.249 ms
  纯GPU kernel时间: `0.375 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.249 ms`, `host_to_first_kernel_gap=3.090107`, `host_end_to_last_kernel_tail=3.217 ms`, `gpu_makespan=0.376 ms`, `gpu_kernel_sum=0.375 ms`
  开始时间(ns): `23113721`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2064, 16384]]}`

## Layer 1 / prefill / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `1.983 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.983 ms`, `module_to_last_kernel=7.892 ms`, `host_to_first_kernel_gap=3.69431`, `host_end_to_last_kernel_tail=5.909 ms`, `gpu_makespan=4.198 ms`, `gpu_kernel_sum=3.124 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=2064`, `total_tokens=34905`, `chunked_req_prefix_len=32841`, `current_chunked_req_prefix_len=32841`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[2064, 128, 192], [34905, 128, 192], [34905, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.250 ms
  纯GPU kernel时间: `0.081 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.250 ms`, `host_to_first_kernel_gap=3.525147`, `host_end_to_last_kernel_tail=3.358 ms`, `gpu_makespan=0.083 ms`, `gpu_kernel_sum=0.081 ms`
  开始时间(ns): `24386262`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2064, 7168]]}`
- `q_a_layernorm` -> 0.049 ms
  纯GPU kernel时间: `0.006 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.049 ms`, `host_to_first_kernel_gap=3.300981`, `host_end_to_last_kernel_tail=3.258 ms`, `gpu_makespan=0.006 ms`, `gpu_kernel_sum=0.006 ms`
  开始时间(ns): `24693596`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2064, 1536]]}`
- `q_b_proj` -> 0.216 ms
  纯GPU kernel时间: `0.132 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.216 ms`, `host_to_first_kernel_gap=3.227218`, `host_end_to_last_kernel_tail=3.144 ms`, `gpu_makespan=0.134 ms`, `gpu_kernel_sum=0.132 ms`
  开始时间(ns): `24774367`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2064, 1536]]}`
- `kv_a_layernorm` -> 0.041 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.041 ms`, `host_to_first_kernel_gap=3.083503`, `host_end_to_last_kernel_tail=3.047 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `25051522`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2064, 512]]}`
- `rotary_emb` -> 0.076 ms
  纯GPU kernel时间: `0.027 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.076 ms`, `host_to_first_kernel_gap=2.999231`, `host_end_to_last_kernel_tail=2.950 ms`, `gpu_makespan=0.027 ms`, `gpu_kernel_sum=0.027 ms`
  开始时间(ns): `25142130`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2064], [2064, 128, 64], [2064, 1, 64]]}`
- `kv_b_proj` -> 0.226 ms
  纯GPU kernel时间: `1.295 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.226 ms`, `host_to_first_kernel_gap=2.786422`, `host_end_to_last_kernel_tail=3.857 ms`, `gpu_makespan=1.296 ms`, `gpu_kernel_sum=1.295 ms`
  开始时间(ns): `25431227`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[34905, 512]]}`
- `attn_mha` -> 0.155 ms
  纯GPU kernel时间: `1.192 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.155 ms`, `host_to_first_kernel_gap=4.77091`, `host_end_to_last_kernel_tail=5.808 ms`, `gpu_makespan=1.192 ms`, `gpu_kernel_sum=1.192 ms`
  开始时间(ns): `25756143`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2064, 128, 192], [34905, 128, 192], [34905, 128, 128]]}`
- `o_proj` -> 0.235 ms
  纯GPU kernel时间: `0.387 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.235 ms`, `host_to_first_kernel_gap=5.76964`, `host_end_to_last_kernel_tail=5.922 ms`, `gpu_makespan=0.388 ms`, `gpu_kernel_sum=0.387 ms`
  开始时间(ns): `25951267`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2064, 16384]]}`

## Layer 2 / prefill / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.995 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.995 ms`, `module_to_last_kernel=10.672 ms`, `host_to_first_kernel_gap=6.453951`, `host_end_to_last_kernel_tail=8.678 ms`, `gpu_makespan=4.218 ms`, `gpu_kernel_sum=3.147 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=2064`, `total_tokens=34905`, `chunked_req_prefix_len=32841`, `current_chunked_req_prefix_len=32841`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[2064, 128, 192], [34905, 128, 192], [34905, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.263 ms
  纯GPU kernel时间: `0.081 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.263 ms`, `host_to_first_kernel_gap=6.278807`, `host_end_to_last_kernel_tail=6.099 ms`, `gpu_makespan=0.083 ms`, `gpu_kernel_sum=0.081 ms`
  开始时间(ns): `27163601`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2064, 7168]]}`
- `q_a_layernorm` -> 0.048 ms
  纯GPU kernel时间: `0.006 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.048 ms`, `host_to_first_kernel_gap=6.047881`, `host_end_to_last_kernel_tail=6.006 ms`, `gpu_makespan=0.006 ms`, `gpu_kernel_sum=0.006 ms`
  开始时间(ns): `27477407`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2064, 1536]]}`
- `q_b_proj` -> 0.217 ms
  纯GPU kernel时间: `0.131 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.217 ms`, `host_to_first_kernel_gap=5.975906`, `host_end_to_last_kernel_tail=5.891 ms`, `gpu_makespan=0.132 ms`, `gpu_kernel_sum=0.131 ms`
  开始时间(ns): `27556806`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2064, 1536]]}`
- `kv_a_layernorm` -> 0.040 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.040 ms`, `host_to_first_kernel_gap=5.830941`, `host_end_to_last_kernel_tail=5.796 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `27833323`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2064, 512]]}`
- `rotary_emb` -> 0.076 ms
  纯GPU kernel时间: `0.026 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.076 ms`, `host_to_first_kernel_gap=5.75956`, `host_end_to_last_kernel_tail=5.710 ms`, `gpu_makespan=0.026 ms`, `gpu_kernel_sum=0.026 ms`
  开始时间(ns): `27912032`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2064], [2064, 128, 64], [2064, 1, 64]]}`
- `kv_b_proj` -> 0.226 ms
  纯GPU kernel时间: `1.321 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.226 ms`, `host_to_first_kernel_gap=5.529799`, `host_end_to_last_kernel_tail=6.627 ms`, `gpu_makespan=1.323 ms`, `gpu_kernel_sum=1.321 ms`
  开始时间(ns): `28217505`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[34905, 512]]}`
- `attn_mha` -> 0.153 ms
  纯GPU kernel时间: `1.195 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.153 ms`, `host_to_first_kernel_gap=7.540463`, `host_end_to_last_kernel_tail=8.582 ms`, `gpu_makespan=1.195 ms`, `gpu_kernel_sum=1.195 ms`
  开始时间(ns): `28539445`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2064, 128, 192], [34905, 128, 192], [34905, 128, 128]]}`
- `o_proj` -> 0.236 ms
  纯GPU kernel时间: `0.382 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.236 ms`, `host_to_first_kernel_gap=8.5447`, `host_end_to_last_kernel_tail=8.692 ms`, `gpu_makespan=0.383 ms`, `gpu_kernel_sum=0.382 ms`
  开始时间(ns): `28732839`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2064, 16384]]}`

## Layer 3 / prefill / instance 1

- 执行序号: `4`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `1.955 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.955 ms`, `module_to_last_kernel=13.249 ms`, `host_to_first_kernel_gap=9.184316`, `host_end_to_last_kernel_tail=11.294 ms`, `gpu_makespan=4.065 ms`, `gpu_kernel_sum=2.993 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=2064`, `total_tokens=34905`, `chunked_req_prefix_len=32841`, `current_chunked_req_prefix_len=32841`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[2064, 128, 192], [34905, 128, 192], [34905, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.230 ms
  纯GPU kernel时间: `0.079 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.230 ms`, `host_to_first_kernel_gap=9.015764`, `host_end_to_last_kernel_tail=8.866 ms`, `gpu_makespan=0.080 ms`, `gpu_kernel_sum=0.079 ms`
  开始时间(ns): `29961420`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2064, 7168]]}`
- `q_a_layernorm` -> 0.046 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.046 ms`, `host_to_first_kernel_gap=8.816505`, `host_end_to_last_kernel_tail=8.776 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `30240775`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2064, 1536]]}`
- `q_b_proj` -> 0.219 ms
  纯GPU kernel时间: `0.131 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.219 ms`, `host_to_first_kernel_gap=8.748634`, `host_end_to_last_kernel_tail=8.662 ms`, `gpu_makespan=0.132 ms`, `gpu_kernel_sum=0.131 ms`
  开始时间(ns): `30316422`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2064, 1536]]}`
- `kv_a_layernorm` -> 0.040 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.040 ms`, `host_to_first_kernel_gap=8.602793`, `host_end_to_last_kernel_tail=8.568 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `30594455`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2064, 512]]}`
- `rotary_emb` -> 0.075 ms
  纯GPU kernel时间: `0.026 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.075 ms`, `host_to_first_kernel_gap=8.533546`, `host_end_to_last_kernel_tail=8.485 ms`, `gpu_makespan=0.026 ms`, `gpu_kernel_sum=0.026 ms`
  开始时间(ns): `30671382`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2064], [2064, 128, 64], [2064, 1, 64]]}`
- `kv_b_proj` -> 0.252 ms
  纯GPU kernel时间: `1.180 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.252 ms`, `host_to_first_kernel_gap=8.31259`, `host_end_to_last_kernel_tail=9.244 ms`, `gpu_makespan=1.183 ms`, `gpu_kernel_sum=1.180 ms`
  开始时间(ns): `30967634`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[34905, 512]]}`
- `attn_mha` -> 0.143 ms
  纯GPU kernel时间: `1.191 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.143 ms`, `host_to_first_kernel_gap=10.151618`, `host_end_to_last_kernel_tail=11.200 ms`, `gpu_makespan=1.191 ms`, `gpu_kernel_sum=1.191 ms`
  开始时间(ns): `31322138`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2064, 128, 192], [34905, 128, 192], [34905, 128, 128]]}`
- `o_proj` -> 0.228 ms
  纯GPU kernel时间: `0.374 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.228 ms`, `host_to_first_kernel_gap=11.158756`, `host_end_to_last_kernel_tail=11.308 ms`, `gpu_makespan=0.376 ms`, `gpu_kernel_sum=0.374 ms`
  开始时间(ns): `31507062`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2064, 16384]]}`

## Layer 4 / prefill / instance 1

- 执行序号: `5`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `2.038 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.038 ms`, `module_to_last_kernel=16.676 ms`, `host_to_first_kernel_gap=12.588123`, `host_end_to_last_kernel_tail=14.638 ms`, `gpu_makespan=4.088 ms`, `gpu_kernel_sum=3.015 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=2064`, `total_tokens=34905`, `chunked_req_prefix_len=32841`, `current_chunked_req_prefix_len=32841`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[2064, 128, 192], [34905, 128, 192], [34905, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.293 ms
  纯GPU kernel时间: `0.079 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.293 ms`, `host_to_first_kernel_gap=12.414586`, `host_end_to_last_kernel_tail=12.203 ms`, `gpu_makespan=0.081 ms`, `gpu_kernel_sum=0.079 ms`
  开始时间(ns): `33275356`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2064, 7168]]}`
- `q_a_layernorm` -> 0.050 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.050 ms`, `host_to_first_kernel_gap=12.148928`, `host_end_to_last_kernel_tail=12.104 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `33622550`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2064, 1536]]}`
- `q_b_proj` -> 0.225 ms
  纯GPU kernel时间: `0.131 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.225 ms`, `host_to_first_kernel_gap=12.074275`, `host_end_to_last_kernel_tail=11.982 ms`, `gpu_makespan=0.132 ms`, `gpu_kernel_sum=0.131 ms`
  开始时间(ns): `33703667`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2064, 1536]]}`
- `kv_a_layernorm` -> 0.040 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.040 ms`, `host_to_first_kernel_gap=11.920162`, `host_end_to_last_kernel_tail=11.885 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `33990035`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2064, 512]]}`
- `rotary_emb` -> 0.081 ms
  纯GPU kernel时间: `0.026 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.081 ms`, `host_to_first_kernel_gap=11.845462`, `host_end_to_last_kernel_tail=11.791 ms`, `gpu_makespan=0.026 ms`, `gpu_kernel_sum=0.026 ms`
  开始时间(ns): `34072223`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2064], [2064, 128, 64], [2064, 1, 64]]}`
- `kv_b_proj` -> 0.226 ms
  纯GPU kernel时间: `1.207 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.226 ms`, `host_to_first_kernel_gap=11.636863`, `host_end_to_last_kernel_tail=12.621 ms`, `gpu_makespan=1.210 ms`, `gpu_kernel_sum=1.207 ms`
  开始时间(ns): `34356150`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[34905, 512]]}`
- `attn_mha` -> 0.143 ms
  纯GPU kernel时间: `1.190 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.143 ms`, `host_to_first_kernel_gap=13.537341`, `host_end_to_last_kernel_tail=14.584 ms`, `gpu_makespan=1.190 ms`, `gpu_kernel_sum=1.190 ms`
  开始时间(ns): `34676949`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2064, 128, 192], [34905, 128, 192], [34905, 128, 128]]}`
- `o_proj` -> 0.263 ms
  纯GPU kernel时间: `0.370 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.263 ms`, `host_to_first_kernel_gap=14.544258`, `host_end_to_last_kernel_tail=14.652 ms`, `gpu_makespan=0.371 ms`, `gpu_kernel_sum=0.370 ms`
  开始时间(ns): `34862702`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2064, 16384]]}`

## Layer 5 / prefill / instance 1

- 执行序号: `6`
- Collector对齐边界: `{'Module': 'model.model.layers.5.self_attn'}`
- 整块 MLA-module 时长: `2.020 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.020 ms`, `module_to_last_kernel=20.157 ms`, `host_to_first_kernel_gap=15.986036`, `host_end_to_last_kernel_tail=18.136 ms`, `gpu_makespan=4.170 ms`, `gpu_kernel_sum=3.101 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=2064`, `total_tokens=34905`, `chunked_req_prefix_len=32841`, `current_chunked_req_prefix_len=32841`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[2064, 128, 192], [34905, 128, 192], [34905, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.273 ms
  纯GPU kernel时间: `0.079 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.273 ms`, `host_to_first_kernel_gap=15.813988`, `host_end_to_last_kernel_tail=15.621 ms`, `gpu_makespan=0.081 ms`, `gpu_kernel_sum=0.079 ms`
  开始时间(ns): `36593895`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2064, 7168]]}`
- `q_a_layernorm` -> 0.048 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.048 ms`, `host_to_first_kernel_gap=15.570846`, `host_end_to_last_kernel_tail=15.528 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `36918221`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2064, 1536]]}`
- `q_b_proj` -> 0.223 ms
  纯GPU kernel时间: `0.128 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.223 ms`, `host_to_first_kernel_gap=15.499092`, `host_end_to_last_kernel_tail=15.406 ms`, `gpu_makespan=0.130 ms`, `gpu_kernel_sum=0.128 ms`
  开始时间(ns): `36996375`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2064, 1536]]}`
- `kv_a_layernorm` -> 0.044 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.044 ms`, `host_to_first_kernel_gap=15.346785`, `host_end_to_last_kernel_tail=15.308 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `37278858`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2064, 512]]}`
- `rotary_emb` -> 0.075 ms
  纯GPU kernel时间: `0.027 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.075 ms`, `host_to_first_kernel_gap=15.271575`, `host_end_to_last_kernel_tail=15.224 ms`, `gpu_makespan=0.027 ms`, `gpu_kernel_sum=0.027 ms`
  开始时间(ns): `37360500`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2064], [2064, 128, 64], [2064, 1, 64]]}`
- `kv_b_proj` -> 0.242 ms
  纯GPU kernel时间: `1.290 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.242 ms`, `host_to_first_kernel_gap=15.064482`, `host_end_to_last_kernel_tail=16.114 ms`, `gpu_makespan=1.291 ms`, `gpu_kernel_sum=1.290 ms`
  开始时间(ns): `37644009`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[34905, 512]]}`
- `attn_mha` -> 0.164 ms
  纯GPU kernel时间: `1.190 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.164 ms`, `host_to_first_kernel_gap=17.023855`, `host_end_to_last_kernel_tail=18.050 ms`, `gpu_makespan=1.190 ms`, `gpu_kernel_sum=1.190 ms`
  开始时间(ns): `37984984`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2064, 128, 192], [34905, 128, 192], [34905, 128, 128]]}`
- `o_proj` -> 0.238 ms
  纯GPU kernel时间: `0.377 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.238 ms`, `host_to_first_kernel_gap=18.009718`, `host_end_to_last_kernel_tail=18.149 ms`, `gpu_makespan=0.378 ms`, `gpu_kernel_sum=0.377 ms`
  开始时间(ns): `38190672`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2064, 16384]]}`
