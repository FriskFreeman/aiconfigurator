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
- 整块 MLA-module 时长: `2.756 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.756 ms`, `module_to_last_kernel=11.094 ms`, `host_to_first_kernel_gap=0.348341`, `host_end_to_last_kernel_tail=8.338 ms`, `gpu_makespan=10.746 ms`, `gpu_kernel_sum=8.936 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=32797`, `chunked_req_prefix_len=24605`, `current_chunked_req_prefix_len=24605`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[8192, 128, 192], [32797, 128, 192], [32797, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.431 ms
  纯GPU kernel时间: `0.273 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.431 ms`, `host_to_first_kernel_gap=0.141981`, `host_end_to_last_kernel_tail=0.166 ms`, `gpu_makespan=0.455 ms`, `gpu_kernel_sum=0.273 ms`
  开始时间(ns): `24653736`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.053 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.053 ms`, `host_to_first_kernel_gap=0.093876`, `host_end_to_last_kernel_tail=0.059 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `25157425`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.253 ms
  纯GPU kernel时间: `0.452 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.253 ms`, `host_to_first_kernel_gap=0.091149`, `host_end_to_last_kernel_tail=0.407 ms`, `gpu_makespan=0.569 ms`, `gpu_kernel_sum=0.452 ms`
  开始时间(ns): `25248280`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.042 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.042 ms`, `host_to_first_kernel_gap=0.339689`, `host_end_to_last_kernel_tail=0.310 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `25569051`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.103 ms
  纯GPU kernel时间: `0.107 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.103 ms`, `host_to_first_kernel_gap=0.267308`, `host_end_to_last_kernel_tail=0.270 ms`, `gpu_makespan=0.107 ms`, `gpu_kernel_sum=0.107 ms`
  开始时间(ns): `25655736`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.281 ms
  纯GPU kernel时间: `1.189 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.281 ms`, `host_to_first_kernel_gap=0.099661`, `host_end_to_last_kernel_tail=1.138 ms`, `gpu_makespan=1.320 ms`, `gpu_kernel_sum=1.189 ms`
  开始时间(ns): `26284599`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[32797, 512]]}`
- `attn_mha` -> 0.187 ms
  纯GPU kernel时间: `5.635 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.187 ms`, `host_to_first_kernel_gap=1.97115`, `host_end_to_last_kernel_tail=7.420 ms`, `gpu_makespan=5.635 ms`, `gpu_kernel_sum=5.635 ms`
  开始时间(ns): `26683380`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [32797, 128, 192], [32797, 128, 128]]}`
- `o_proj` -> 0.270 ms
  纯GPU kernel时间: `1.249 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.270 ms`, `host_to_first_kernel_gap=7.373723`, `host_end_to_last_kernel_tail=8.354 ms`, `gpu_makespan=1.250 ms`, `gpu_kernel_sum=1.249 ms`
  开始时间(ns): `26918050`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 1 / prefill / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `1.950 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.950 ms`, `module_to_last_kernel=21.916 ms`, `host_to_first_kernel_gap=12.073962`, `host_end_to_last_kernel_tail=19.966 ms`, `gpu_makespan=9.842 ms`, `gpu_kernel_sum=8.814 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=32797`, `chunked_req_prefix_len=24605`, `current_chunked_req_prefix_len=24605`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[8192, 128, 192], [32797, 128, 192], [32797, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.236 ms
  纯GPU kernel时间: `0.267 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.236 ms`, `host_to_first_kernel_gap=11.924506`, `host_end_to_last_kernel_tail=11.957 ms`, `gpu_makespan=0.269 ms`, `gpu_kernel_sum=0.267 ms`
  开始时间(ns): `28226045`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.048 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.048 ms`, `host_to_first_kernel_gap=11.904387`, `host_end_to_last_kernel_tail=11.874 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `28515572`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.218 ms
  纯GPU kernel时间: `0.442 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.218 ms`, `host_to_first_kernel_gap=11.844604`, `host_end_to_last_kernel_tail=12.070 ms`, `gpu_makespan=0.444 ms`, `gpu_kernel_sum=0.442 ms`
  开始时间(ns): `28594491`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.047 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.047 ms`, `host_to_first_kernel_gap=12.005017`, `host_end_to_last_kernel_tail=11.971 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `28877854`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.078 ms
  纯GPU kernel时间: `0.107 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.078 ms`, `host_to_first_kernel_gap=11.929373`, `host_end_to_last_kernel_tail=11.958 ms`, `gpu_makespan=0.107 ms`, `gpu_kernel_sum=0.107 ms`
  开始时间(ns): `28967834`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.229 ms
  纯GPU kernel时间: `1.076 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.229 ms`, `host_to_first_kernel_gap=11.799657`, `host_end_to_last_kernel_tail=12.648 ms`, `gpu_makespan=1.077 ms`, `gpu_kernel_sum=1.076 ms`
  开始时间(ns): `29268814`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[32797, 512]]}`
- `attn_mha` -> 0.146 ms
  纯GPU kernel时间: `5.644 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.146 ms`, `host_to_first_kernel_gap=13.497952`, `host_end_to_last_kernel_tail=18.996 ms`, `gpu_makespan=5.644 ms`, `gpu_kernel_sum=5.644 ms`
  开始时间(ns): `29599573`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [32797, 128, 192], [32797, 128, 128]]}`
- `o_proj` -> 0.226 ms
  纯GPU kernel时间: `1.248 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.226 ms`, `host_to_first_kernel_gap=18.957033`, `host_end_to_last_kernel_tail=19.980 ms`, `gpu_makespan=1.249 ms`, `gpu_kernel_sum=1.248 ms`
  开始时间(ns): `29785959`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 2 / prefill / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.901 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.901 ms`, `module_to_last_kernel=33.748 ms`, `host_to_first_kernel_gap=23.727285`, `host_end_to_last_kernel_tail=31.847 ms`, `gpu_makespan=10.021 ms`, `gpu_kernel_sum=8.994 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=32797`, `chunked_req_prefix_len=24605`, `current_chunked_req_prefix_len=24605`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[8192, 128, 192], [32797, 128, 192], [32797, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.263 ms
  纯GPU kernel时间: `0.267 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.263 ms`, `host_to_first_kernel_gap=23.582815`, `host_end_to_last_kernel_tail=23.589 ms`, `gpu_makespan=0.269 ms`, `gpu_kernel_sum=0.267 ms`
  开始时间(ns): `31014411`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.047 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.047 ms`, `host_to_first_kernel_gap=23.538143`, `host_end_to_last_kernel_tail=23.509 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `31328171`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.210 ms
  纯GPU kernel时间: `0.436 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.210 ms`, `host_to_first_kernel_gap=23.478185`, `host_end_to_last_kernel_tail=23.706 ms`, `gpu_makespan=0.437 ms`, `gpu_kernel_sum=0.436 ms`
  开始时间(ns): `31407233`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.040 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.040 ms`, `host_to_first_kernel_gap=23.646116`, `host_end_to_last_kernel_tail=23.620 ms`, `gpu_makespan=0.014 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `31676454`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.072 ms
  纯GPU kernel时间: `0.106 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.072 ms`, `host_to_first_kernel_gap=23.582733`, `host_end_to_last_kernel_tail=23.617 ms`, `gpu_makespan=0.106 ms`, `gpu_kernel_sum=0.106 ms`
  开始时间(ns): `31755165`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.229 ms
  纯GPU kernel时间: `1.182 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.229 ms`, `host_to_first_kernel_gap=23.471173`, `host_end_to_last_kernel_tail=24.427 ms`, `gpu_makespan=1.185 ms`, `gpu_kernel_sum=1.182 ms`
  开始时间(ns): `32036133`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[32797, 512]]}`
- `attn_mha` -> 0.143 ms
  纯GPU kernel时间: `5.651 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.143 ms`, `host_to_first_kernel_gap=25.283073`, `host_end_to_last_kernel_tail=30.791 ms`, `gpu_makespan=5.652 ms`, `gpu_kernel_sum=5.651 ms`
  开始时间(ns): `32360359`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [32797, 128, 192], [32797, 128, 128]]}`
- `o_proj` -> 0.214 ms
  纯GPU kernel时间: `1.320 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.214 ms`, `host_to_first_kernel_gap=30.754357`, `host_end_to_last_kernel_tail=31.861 ms`, `gpu_makespan=1.321 ms`, `gpu_kernel_sum=1.320 ms`
  开始时间(ns): `32542574`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 3 / prefill / instance 1

- 执行序号: `4`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `1.892 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.892 ms`, `module_to_last_kernel=45.953 ms`, `host_to_first_kernel_gap=35.847997`, `host_end_to_last_kernel_tail=44.061 ms`, `gpu_makespan=10.105 ms`, `gpu_kernel_sum=9.079 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=32797`, `chunked_req_prefix_len=24605`, `current_chunked_req_prefix_len=24605`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[8192, 128, 192], [32797, 128, 192], [32797, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.241 ms
  纯GPU kernel时间: `0.279 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.241 ms`, `host_to_first_kernel_gap=35.705067`, `host_end_to_last_kernel_tail=35.745 ms`, `gpu_makespan=0.281 ms`, `gpu_kernel_sum=0.279 ms`
  开始时间(ns): `33703250`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.044 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.044 ms`, `host_to_first_kernel_gap=35.695852`, `host_end_to_last_kernel_tail=35.670 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `33993137`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.230 ms
  纯GPU kernel时间: `0.458 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.230 ms`, `host_to_first_kernel_gap=35.643137`, `host_end_to_last_kernel_tail=35.872 ms`, `gpu_makespan=0.459 ms`, `gpu_kernel_sum=0.458 ms`
  开始时间(ns): `34066204`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.042 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.042 ms`, `host_to_first_kernel_gap=35.812355`, `host_end_to_last_kernel_tail=35.784 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `34355673`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.073 ms
  纯GPU kernel时间: `0.107 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.073 ms`, `host_to_first_kernel_gap=35.746357`, `host_end_to_last_kernel_tail=35.780 ms`, `gpu_makespan=0.107 ms`, `gpu_kernel_sum=0.107 ms`
  开始时间(ns): `34436391`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.222 ms
  纯GPU kernel时间: `1.228 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.222 ms`, `host_to_first_kernel_gap=35.63838`, `host_end_to_last_kernel_tail=36.648 ms`, `gpu_makespan=1.231 ms`, `gpu_kernel_sum=1.228 ms`
  开始时间(ns): `34713872`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[32797, 512]]}`
- `attn_mha` -> 0.147 ms
  纯GPU kernel时间: `5.656 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.147 ms`, `host_to_first_kernel_gap=37.505182`, `host_end_to_last_kernel_tail=43.014 ms`, `gpu_makespan=5.656 ms`, `gpu_kernel_sum=5.656 ms`
  开始时间(ns): `35028380`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [32797, 128, 192], [32797, 128, 128]]}`
- `o_proj` -> 0.223 ms
  纯GPU kernel时间: `1.320 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.223 ms`, `host_to_first_kernel_gap=42.976581`, `host_end_to_last_kernel_tail=44.075 ms`, `gpu_makespan=1.322 ms`, `gpu_kernel_sum=1.320 ms`
  开始时间(ns): `35214992`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 4 / prefill / instance 1

- 执行序号: `5`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `2.020 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.020 ms`, `module_to_last_kernel=63.247 ms`, `host_to_first_kernel_gap=53.044675`, `host_end_to_last_kernel_tail=61.227 ms`, `gpu_makespan=10.202 ms`, `gpu_kernel_sum=9.178 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=32797`, `chunked_req_prefix_len=24605`, `current_chunked_req_prefix_len=24605`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[8192, 128, 192], [32797, 128, 192], [32797, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.304 ms
  纯GPU kernel时间: `0.287 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.304 ms`, `host_to_first_kernel_gap=52.894845`, `host_end_to_last_kernel_tail=52.879 ms`, `gpu_makespan=0.288 ms`, `gpu_kernel_sum=0.287 ms`
  开始时间(ns): `37080206`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.051 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.051 ms`, `host_to_first_kernel_gap=52.826448`, `host_end_to_last_kernel_tail=52.794 ms`, `gpu_makespan=0.019 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `37437146`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.223 ms
  纯GPU kernel时间: `0.468 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.223 ms`, `host_to_first_kernel_gap=52.761974`, `host_end_to_last_kernel_tail=53.008 ms`, `gpu_makespan=0.469 ms`, `gpu_kernel_sum=0.468 ms`
  开始时间(ns): `37521748`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.042 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.042 ms`, `host_to_first_kernel_gap=52.944369`, `host_end_to_last_kernel_tail=52.916 ms`, `gpu_makespan=0.014 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `37808345`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.077 ms
  纯GPU kernel时间: `0.106 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.077 ms`, `host_to_first_kernel_gap=52.878247`, `host_end_to_last_kernel_tail=52.907 ms`, `gpu_makespan=0.106 ms`, `gpu_kernel_sum=0.106 ms`
  开始时间(ns): `37890275`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.228 ms
  纯GPU kernel时间: `1.271 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.228 ms`, `host_to_first_kernel_gap=52.753884`, `host_end_to_last_kernel_tail=53.800 ms`, `gpu_makespan=1.274 ms`, `gpu_kernel_sum=1.271 ms`
  开始时间(ns): `38182990`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[32797, 512]]}`
- `attn_mha` -> 0.140 ms
  纯GPU kernel时间: `5.660 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.140 ms`, `host_to_first_kernel_gap=54.651487`, `host_end_to_last_kernel_tail=60.171 ms`, `gpu_makespan=5.660 ms`, `gpu_kernel_sum=5.660 ms`
  开始时间(ns): `38508425`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [32797, 128, 192], [32797, 128, 128]]}`
- `o_proj` -> 0.244 ms
  纯GPU kernel时间: `1.354 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.244 ms`, `host_to_first_kernel_gap=60.129912`, `host_end_to_last_kernel_tail=61.241 ms`, `gpu_makespan=1.355 ms`, `gpu_kernel_sum=1.354 ms`
  开始时间(ns): `38692107`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 5 / prefill / instance 1

- 执行序号: `6`
- Collector对齐边界: `{'Module': 'model.model.layers.5.self_attn'}`
- 整块 MLA-module 时长: `1.976 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.976 ms`, `module_to_last_kernel=80.453 ms`, `host_to_first_kernel_gap=70.29449`, `host_end_to_last_kernel_tail=78.478 ms`, `gpu_makespan=10.159 ms`, `gpu_kernel_sum=9.130 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=32797`, `chunked_req_prefix_len=24605`, `current_chunked_req_prefix_len=24605`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[8192, 128, 192], [32797, 128, 192], [32797, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.278 ms
  纯GPU kernel时间: `0.283 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.278 ms`, `host_to_first_kernel_gap=70.146461`, `host_end_to_last_kernel_tail=70.154 ms`, `gpu_makespan=0.285 ms`, `gpu_kernel_sum=0.283 ms`
  开始时间(ns): `40428027`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.048 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.048 ms`, `host_to_first_kernel_gap=70.103765`, `host_end_to_last_kernel_tail=70.074 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `40756547`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.214 ms
  纯GPU kernel时间: `0.470 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.214 ms`, `host_to_first_kernel_gap=70.044409`, `host_end_to_last_kernel_tail=70.303 ms`, `gpu_makespan=0.472 ms`, `gpu_kernel_sum=0.470 ms`
  开始时间(ns): `40835487`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.052 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.052 ms`, `host_to_first_kernel_gap=70.241389`, `host_end_to_last_kernel_tail=70.203 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `41110826`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.075 ms
  纯GPU kernel时间: `0.107 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.075 ms`, `host_to_first_kernel_gap=70.1656`, `host_end_to_last_kernel_tail=70.197 ms`, `gpu_makespan=0.107 ms`, `gpu_kernel_sum=0.107 ms`
  开始时间(ns): `41201143`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.236 ms
  纯GPU kernel时间: `1.235 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.236 ms`, `host_to_first_kernel_gap=70.052932`, `host_end_to_last_kernel_tail=71.053 ms`, `gpu_makespan=1.236 ms`, `gpu_kernel_sum=1.235 ms`
  开始时间(ns): `41485683`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[32797, 512]]}`
- `attn_mha` -> 0.145 ms
  纯GPU kernel时间: `5.660 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.145 ms`, `host_to_first_kernel_gap=71.90843`, `host_end_to_last_kernel_tail=77.424 ms`, `gpu_makespan=5.660 ms`, `gpu_kernel_sum=5.660 ms`
  开始时间(ns): `41818023`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [32797, 128, 192], [32797, 128, 128]]}`
- `o_proj` -> 0.238 ms
  纯GPU kernel时间: `1.344 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.238 ms`, `host_to_first_kernel_gap=77.384058`, `host_end_to_last_kernel_tail=78.491 ms`, `gpu_makespan=1.346 ms`, `gpu_kernel_sum=1.344 ms`
  开始时间(ns): `42003830`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`
