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
- 整块 MLA-module 时长: `4.495 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=4.495 ms`, `module_to_last_kernel=6.547 ms`, `host_to_first_kernel_gap=0.461101`, `host_end_to_last_kernel_tail=2.053 ms`, `gpu_makespan=6.086 ms`, `gpu_kernel_sum=2.813 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4128`, `total_tokens=20625`, `chunked_req_prefix_len=16497`, `current_chunked_req_prefix_len=16497`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[4128, 128, 192], [20625, 128, 192], [20625, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.431 ms
  纯GPU kernel时间: `0.142 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.431 ms`, `host_to_first_kernel_gap=0.148899`, `host_end_to_last_kernel_tail=0.066 ms`, `gpu_makespan=0.348 ms`, `gpu_kernel_sum=0.142 ms`
  开始时间(ns): `24368392`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4128, 7168]]}`
- `q_a_layernorm` -> 0.063 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.063 ms`, `host_to_first_kernel_gap=0.057362`, `host_end_to_last_kernel_tail=0.003 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `24866265`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4128, 1536]]}`
- `q_b_proj` -> 0.276 ms
  纯GPU kernel时间: `0.242 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.276 ms`, `host_to_first_kernel_gap=0.094648`, `host_end_to_last_kernel_tail=0.192 ms`, `gpu_makespan=0.373 ms`, `gpu_kernel_sum=0.242 ms`
  开始时间(ns): `24965203`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4128, 1536]]}`
- `kv_a_layernorm` -> 0.048 ms
  纯GPU kernel时间: `0.008 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.048 ms`, `host_to_first_kernel_gap=0.124192`, `host_end_to_last_kernel_tail=0.084 ms`, `gpu_makespan=0.008 ms`, `gpu_kernel_sum=0.008 ms`
  开始时间(ns): `25308203`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4128, 512]]}`
- `rotary_emb` -> 0.105 ms
  纯GPU kernel时间: `0.053 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.105 ms`, `host_to_first_kernel_gap=0.089992`, `host_end_to_last_kernel_tail=0.038 ms`, `gpu_makespan=0.053 ms`, `gpu_kernel_sum=0.053 ms`
  开始时间(ns): `25399747`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4128], [4128, 128, 64], [4128, 1, 64]]}`
- `kv_b_proj` -> 1.544 ms
  纯GPU kernel时间: `0.742 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=1.544 ms`, `host_to_first_kernel_gap=0.122059`, `host_end_to_last_kernel_tail=0.663 ms`, `gpu_makespan=2.085 ms`, `gpu_kernel_sum=0.742 ms`
  开始时间(ns): `26177344`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[20625, 512]]}`
- `attn_mha` -> 0.207 ms
  纯GPU kernel时间: `0.902 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.207 ms`, `host_to_first_kernel_gap=1.061566`, `host_end_to_last_kernel_tail=1.757 ms`, `gpu_makespan=0.902 ms`, `gpu_kernel_sum=0.902 ms`
  开始时间(ns): `27922958`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4128, 128, 192], [20625, 128, 192], [20625, 128, 128]]}`
- `o_proj` -> 0.353 ms
  纯GPU kernel时间: `0.714 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.353 ms`, `host_to_first_kernel_gap=1.70669`, `host_end_to_last_kernel_tail=2.069 ms`, `gpu_makespan=0.715 ms`, `gpu_kernel_sum=0.714 ms`
  开始时间(ns): `28181674`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4128, 16384]]}`

## Layer 1 / prefill / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `2.302 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.302 ms`, `module_to_last_kernel=7.025 ms`, `host_to_first_kernel_gap=3.60167`, `host_end_to_last_kernel_tail=4.723 ms`, `gpu_makespan=3.423 ms`, `gpu_kernel_sum=2.764 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4128`, `total_tokens=20625`, `chunked_req_prefix_len=16497`, `current_chunked_req_prefix_len=16497`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[4128, 128, 192], [20625, 128, 192], [20625, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.256 ms
  纯GPU kernel时间: `0.142 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.256 ms`, `host_to_first_kernel_gap=3.306246`, `host_end_to_last_kernel_tail=3.194 ms`, `gpu_makespan=0.143 ms`, `gpu_kernel_sum=0.142 ms`
  开始时间(ns): `29816518`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4128, 7168]]}`
- `q_a_layernorm` -> 0.060 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.060 ms`, `host_to_first_kernel_gap=3.135662`, `host_end_to_last_kernel_tail=3.086 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `30130302`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4128, 1536]]}`
- `q_b_proj` -> 0.222 ms
  纯GPU kernel时间: `0.237 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.222 ms`, `host_to_first_kernel_gap=3.047201`, `host_end_to_last_kernel_tail=3.063 ms`, `gpu_makespan=0.238 ms`, `gpu_kernel_sum=0.237 ms`
  开始时间(ns): `30230507`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4128, 1536]]}`
- `kv_a_layernorm` -> 0.044 ms
  纯GPU kernel时间: `0.008 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.044 ms`, `host_to_first_kernel_gap=3.000117`, `host_end_to_last_kernel_tail=2.964 ms`, `gpu_makespan=0.008 ms`, `gpu_kernel_sum=0.008 ms`
  开始时间(ns): `30515223`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4128, 512]]}`
- `rotary_emb` -> 0.110 ms
  纯GPU kernel时间: `0.054 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.110 ms`, `host_to_first_kernel_gap=2.922241`, `host_end_to_last_kernel_tail=2.866 ms`, `gpu_makespan=0.054 ms`, `gpu_kernel_sum=0.054 ms`
  开始时间(ns): `30603275`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4128], [4128, 128, 64], [4128, 1, 64]]}`
- `kv_b_proj` -> 0.247 ms
  纯GPU kernel时间: `0.704 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.247 ms`, `host_to_first_kernel_gap=2.649196`, `host_end_to_last_kernel_tail=3.109 ms`, `gpu_makespan=0.707 ms`, `gpu_kernel_sum=0.704 ms`
  开始时间(ns): `30971456`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[20625, 512]]}`
- `attn_mha` -> 0.183 ms
  纯GPU kernel时间: `0.899 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.183 ms`, `host_to_first_kernel_gap=3.602451`, `host_end_to_last_kernel_tail=4.319 ms`, `gpu_makespan=0.899 ms`, `gpu_kernel_sum=0.899 ms`
  开始时间(ns): `31326586`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4128, 128, 192], [20625, 128, 192], [20625, 128, 128]]}`
- `o_proj` -> 0.255 ms
  纯GPU kernel时间: `0.712 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.255 ms`, `host_to_first_kernel_gap=4.278461`, `host_end_to_last_kernel_tail=4.738 ms`, `gpu_makespan=0.715 ms`, `gpu_kernel_sum=0.712 ms`
  开始时间(ns): `31552848`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4128, 16384]]}`

## Layer 2 / prefill / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `2.263 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.263 ms`, `module_to_last_kernel=9.769 ms`, `host_to_first_kernel_gap=6.339601`, `host_end_to_last_kernel_tail=7.506 ms`, `gpu_makespan=3.429 ms`, `gpu_kernel_sum=2.770 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4128`, `total_tokens=20625`, `chunked_req_prefix_len=16497`, `current_chunked_req_prefix_len=16497`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[4128, 128, 192], [20625, 128, 192], [20625, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.252 ms
  纯GPU kernel时间: `0.142 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.252 ms`, `host_to_first_kernel_gap=6.061611`, `host_end_to_last_kernel_tail=5.952 ms`, `gpu_makespan=0.143 ms`, `gpu_kernel_sum=0.142 ms`
  开始时间(ns): `32999266`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4128, 7168]]}`
- `q_a_layernorm` -> 0.056 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.056 ms`, `host_to_first_kernel_gap=5.897295`, `host_end_to_last_kernel_tail=5.851 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `33306942`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4128, 1536]]}`
- `q_b_proj` -> 0.214 ms
  纯GPU kernel时间: `0.235 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.214 ms`, `host_to_first_kernel_gap=5.823397`, `host_end_to_last_kernel_tail=5.847 ms`, `gpu_makespan=0.238 ms`, `gpu_kernel_sum=0.235 ms`
  开始时间(ns): `33392456`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4128, 1536]]}`
- `kv_a_layernorm` -> 0.043 ms
  纯GPU kernel时间: `0.008 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.043 ms`, `host_to_first_kernel_gap=5.784396`, `host_end_to_last_kernel_tail=5.749 ms`, `gpu_makespan=0.008 ms`, `gpu_kernel_sum=0.008 ms`
  开始时间(ns): `33668961`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4128, 512]]}`
- `rotary_emb` -> 0.100 ms
  纯GPU kernel时间: `0.053 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.100 ms`, `host_to_first_kernel_gap=5.705554`, `host_end_to_last_kernel_tail=5.658 ms`, `gpu_makespan=0.053 ms`, `gpu_kernel_sum=0.053 ms`
  开始时间(ns): `33757115`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4128], [4128, 128, 64], [4128, 1, 64]]}`
- `kv_b_proj` -> 0.270 ms
  纯GPU kernel时间: `0.710 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.270 ms`, `host_to_first_kernel_gap=5.446459`, `host_end_to_last_kernel_tail=5.888 ms`, `gpu_makespan=0.711 ms`, `gpu_kernel_sum=0.710 ms`
  开始时间(ns): `34111794`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[20625, 512]]}`
- `attn_mha` -> 0.180 ms
  纯GPU kernel时间: `0.901 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.180 ms`, `host_to_first_kernel_gap=6.38189`, `host_end_to_last_kernel_tail=7.104 ms`, `gpu_makespan=0.902 ms`, `gpu_kernel_sum=0.901 ms`
  开始时间(ns): `34490827`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4128, 128, 192], [20625, 128, 192], [20625, 128, 128]]}`
- `o_proj` -> 0.252 ms
  纯GPU kernel时间: `0.712 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.252 ms`, `host_to_first_kernel_gap=7.059684`, `host_end_to_last_kernel_tail=7.523 ms`, `gpu_makespan=0.715 ms`, `gpu_kernel_sum=0.712 ms`
  开始时间(ns): `34715914`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4128, 16384]]}`

## Layer 3 / prefill / instance 1

- 执行序号: `4`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `2.278 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.278 ms`, `module_to_last_kernel=12.568 ms`, `host_to_first_kernel_gap=9.150637`, `host_end_to_last_kernel_tail=10.290 ms`, `gpu_makespan=3.417 ms`, `gpu_kernel_sum=2.762 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4128`, `total_tokens=20625`, `chunked_req_prefix_len=16497`, `current_chunked_req_prefix_len=16497`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[4128, 128, 192], [20625, 128, 192], [20625, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.261 ms
  纯GPU kernel时间: `0.142 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.261 ms`, `host_to_first_kernel_gap=8.875173`, `host_end_to_last_kernel_tail=8.759 ms`, `gpu_makespan=0.145 ms`, `gpu_kernel_sum=0.142 ms`
  开始时间(ns): `36133545`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4128, 7168]]}`
- `q_a_layernorm` -> 0.055 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.055 ms`, `host_to_first_kernel_gap=8.702783`, `host_end_to_last_kernel_tail=8.657 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `36451247`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4128, 1536]]}`
- `q_b_proj` -> 0.217 ms
  纯GPU kernel时间: `0.236 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.217 ms`, `host_to_first_kernel_gap=8.628715`, `host_end_to_last_kernel_tail=8.649 ms`, `gpu_makespan=0.238 ms`, `gpu_kernel_sum=0.236 ms`
  开始时间(ns): `36535971`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4128, 1536]]}`
- `kv_a_layernorm` -> 0.048 ms
  纯GPU kernel时间: `0.008 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.048 ms`, `host_to_first_kernel_gap=8.586599`, `host_end_to_last_kernel_tail=8.547 ms`, `gpu_makespan=0.008 ms`, `gpu_kernel_sum=0.008 ms`
  开始时间(ns): `36815687`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4128, 512]]}`
- `rotary_emb` -> 0.100 ms
  纯GPU kernel时间: `0.053 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.100 ms`, `host_to_first_kernel_gap=8.506162`, `host_end_to_last_kernel_tail=8.458 ms`, `gpu_makespan=0.053 ms`, `gpu_kernel_sum=0.053 ms`
  开始时间(ns): `36905596`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4128], [4128, 128, 64], [4128, 1, 64]]}`
- `kv_b_proj` -> 0.261 ms
  纯GPU kernel时间: `0.701 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.261 ms`, `host_to_first_kernel_gap=8.233983`, `host_end_to_last_kernel_tail=8.675 ms`, `gpu_makespan=0.702 ms`, `gpu_kernel_sum=0.701 ms`
  开始时间(ns): `37272911`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[20625, 512]]}`
- `attn_mha` -> 0.180 ms
  纯GPU kernel时间: `0.901 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.180 ms`, `host_to_first_kernel_gap=9.167631`, `host_end_to_last_kernel_tail=9.889 ms`, `gpu_makespan=0.902 ms`, `gpu_kernel_sum=0.901 ms`
  开始时间(ns): `37643391`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4128, 128, 192], [20625, 128, 192], [20625, 128, 128]]}`
- `o_proj` -> 0.253 ms
  纯GPU kernel时间: `0.711 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.253 ms`, `host_to_first_kernel_gap=9.846494`, `host_end_to_last_kernel_tail=10.306 ms`, `gpu_makespan=0.713 ms`, `gpu_kernel_sum=0.711 ms`
  开始时间(ns): `37867184`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4128, 16384]]}`

## Layer 4 / prefill / instance 1

- 执行序号: `5`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `2.409 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.409 ms`, `module_to_last_kernel=17.296 ms`, `host_to_first_kernel_gap=13.894547`, `host_end_to_last_kernel_tail=14.887 ms`, `gpu_makespan=3.401 ms`, `gpu_kernel_sum=2.745 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4128`, `total_tokens=20625`, `chunked_req_prefix_len=16497`, `current_chunked_req_prefix_len=16497`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[4128, 128, 192], [20625, 128, 192], [20625, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.368 ms
  纯GPU kernel时间: `0.145 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.368 ms`, `host_to_first_kernel_gap=13.607993`, `host_end_to_last_kernel_tail=13.386 ms`, `gpu_makespan=0.146 ms`, `gpu_kernel_sum=0.145 ms`
  开始时间(ns): `40045750`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4128, 7168]]}`
- `q_a_layernorm` -> 0.061 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.061 ms`, `host_to_first_kernel_gap=13.325489`, `host_end_to_last_kernel_tail=13.274 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `40475006`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4128, 1536]]}`
- `q_b_proj` -> 0.242 ms
  纯GPU kernel时间: `0.234 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.242 ms`, `host_to_first_kernel_gap=13.243043`, `host_end_to_last_kernel_tail=13.238 ms`, `gpu_makespan=0.237 ms`, `gpu_kernel_sum=0.234 ms`
  开始时间(ns): `40569292`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4128, 1536]]}`
- `kv_a_layernorm` -> 0.045 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.045 ms`, `host_to_first_kernel_gap=13.173397`, `host_end_to_last_kernel_tail=13.137 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `40875578`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4128, 512]]}`
- `rotary_emb` -> 0.102 ms
  纯GPU kernel时间: `0.054 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.102 ms`, `host_to_first_kernel_gap=13.096784`, `host_end_to_last_kernel_tail=13.049 ms`, `gpu_makespan=0.054 ms`, `gpu_kernel_sum=0.054 ms`
  开始时间(ns): `40962495`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4128], [4128, 128, 64], [4128, 1, 64]]}`
- `kv_b_proj` -> 0.250 ms
  纯GPU kernel时间: `0.681 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.250 ms`, `host_to_first_kernel_gap=12.836945`, `host_end_to_last_kernel_tail=13.269 ms`, `gpu_makespan=0.683 ms`, `gpu_kernel_sum=0.681 ms`
  开始时间(ns): `41316798`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[20625, 512]]}`
- `attn_mha` -> 0.183 ms
  纯GPU kernel时间: `0.902 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.183 ms`, `host_to_first_kernel_gap=13.764805`, `host_end_to_last_kernel_tail=14.484 ms`, `gpu_makespan=0.902 ms`, `gpu_kernel_sum=0.902 ms`
  开始时间(ns): `41674379`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4128, 128, 192], [20625, 128, 192], [20625, 128, 128]]}`
- `o_proj` -> 0.249 ms
  纯GPU kernel时间: `0.711 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.249 ms`, `host_to_first_kernel_gap=14.441939`, `host_end_to_last_kernel_tail=14.906 ms`, `gpu_makespan=0.713 ms`, `gpu_kernel_sum=0.711 ms`
  开始时间(ns): `41900413`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4128, 16384]]}`

## Layer 5 / prefill / instance 1

- 执行序号: `6`
- Collector对齐边界: `{'Module': 'model.model.layers.5.self_attn'}`
- 整块 MLA-module 时长: `2.392 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.392 ms`, `module_to_last_kernel=22.050 ms`, `host_to_first_kernel_gap=18.592675`, `host_end_to_last_kernel_tail=19.659 ms`, `gpu_makespan=3.458 ms`, `gpu_kernel_sum=2.803 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4128`, `total_tokens=20625`, `chunked_req_prefix_len=16497`, `current_chunked_req_prefix_len=16497`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[4128, 128, 192], [20625, 128, 192], [20625, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.337 ms
  纯GPU kernel时间: `0.143 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.337 ms`, `host_to_first_kernel_gap=18.306651`, `host_end_to_last_kernel_tail=18.114 ms`, `gpu_makespan=0.144 ms`, `gpu_kernel_sum=0.143 ms`
  开始时间(ns): `44022198`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4128, 7168]]}`
- `q_a_layernorm` -> 0.058 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.058 ms`, `host_to_first_kernel_gap=18.055606`, `host_end_to_last_kernel_tail=18.007 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `44417787`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4128, 1536]]}`
- `q_b_proj` -> 0.236 ms
  纯GPU kernel时间: `0.236 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.236 ms`, `host_to_first_kernel_gap=17.970957`, `host_end_to_last_kernel_tail=17.972 ms`, `gpu_makespan=0.237 ms`, `gpu_kernel_sum=0.236 ms`
  开始时间(ns): `44513220`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4128, 1536]]}`
- `kv_a_layernorm` -> 0.044 ms
  纯GPU kernel时间: `0.008 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.044 ms`, `host_to_first_kernel_gap=17.908382`, `host_end_to_last_kernel_tail=17.872 ms`, `gpu_makespan=0.008 ms`, `gpu_kernel_sum=0.008 ms`
  开始时间(ns): `44812979`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4128, 512]]}`
- `rotary_emb` -> 0.101 ms
  纯GPU kernel时间: `0.053 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.101 ms`, `host_to_first_kernel_gap=17.834525`, `host_end_to_last_kernel_tail=17.786 ms`, `gpu_makespan=0.053 ms`, `gpu_kernel_sum=0.053 ms`
  开始时间(ns): `44897300`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4128], [4128, 128, 64], [4128, 1, 64]]}`
- `kv_b_proj` -> 0.256 ms
  纯GPU kernel时间: `0.735 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.256 ms`, `host_to_first_kernel_gap=17.567757`, `host_end_to_last_kernel_tail=18.049 ms`, `gpu_makespan=0.737 ms`, `gpu_kernel_sum=0.735 ms`
  开始时间(ns): `45258564`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[20625, 512]]}`
- `attn_mha` -> 0.184 ms
  纯GPU kernel时间: `0.905 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.184 ms`, `host_to_first_kernel_gap=18.534992`, `host_end_to_last_kernel_tail=19.256 ms`, `gpu_makespan=0.905 ms`, `gpu_kernel_sum=0.905 ms`
  开始时间(ns): `45627425`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4128, 128, 192], [20625, 128, 192], [20625, 128, 128]]}`
- `o_proj` -> 0.252 ms
  纯GPU kernel时间: `0.714 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.252 ms`, `host_to_first_kernel_gap=19.209016`, `host_end_to_last_kernel_tail=19.674 ms`, `gpu_makespan=0.717 ms`, `gpu_kernel_sum=0.714 ms`
  开始时间(ns): `45860537`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4128, 16384]]}`
