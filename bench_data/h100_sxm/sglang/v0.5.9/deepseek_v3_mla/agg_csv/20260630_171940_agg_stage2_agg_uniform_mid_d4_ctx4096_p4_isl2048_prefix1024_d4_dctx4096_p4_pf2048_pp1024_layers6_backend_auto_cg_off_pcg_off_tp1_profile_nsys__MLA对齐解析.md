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
- 整块 MLA-module 时长: `2.749 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.749 ms`, `module_to_last_kernel=7.527 ms`, `host_to_first_kernel_gap=0.356145`, `host_end_to_last_kernel_tail=4.778 ms`, `gpu_makespan=7.171 ms`, `gpu_kernel_sum=5.489 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=28689`, `chunked_req_prefix_len=20497`, `current_chunked_req_prefix_len=20497`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[8192, 128, 192], [28689, 128, 192], [28689, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.447 ms
  纯GPU kernel时间: `0.272 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.447 ms`, `host_to_first_kernel_gap=0.148165`, `host_end_to_last_kernel_tail=0.165 ms`, `gpu_makespan=0.464 ms`, `gpu_kernel_sum=0.272 ms`
  开始时间(ns): `21413273`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.052 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.052 ms`, `host_to_first_kernel_gap=0.103509`, `host_end_to_last_kernel_tail=0.069 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `21921802`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.264 ms
  纯GPU kernel时间: `0.451 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.264 ms`, `host_to_first_kernel_gap=0.093075`, `host_end_to_last_kernel_tail=0.407 ms`, `gpu_makespan=0.578 ms`, `gpu_kernel_sum=0.451 ms`
  开始时间(ns): `22008524`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.046 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.046 ms`, `host_to_first_kernel_gap=0.339595`, `host_end_to_last_kernel_tail=0.307 ms`, `gpu_makespan=0.014 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `22339700`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.107 ms
  纯GPU kernel时间: `0.107 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.107 ms`, `host_to_first_kernel_gap=0.263954`, `host_end_to_last_kernel_tail=0.264 ms`, `gpu_makespan=0.107 ms`, `gpu_kernel_sum=0.107 ms`
  开始时间(ns): `22430221`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.275 ms
  纯GPU kernel时间: `1.020 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.275 ms`, `host_to_first_kernel_gap=0.092849`, `host_end_to_last_kernel_tail=0.973 ms`, `gpu_makespan=1.155 ms`, `gpu_kernel_sum=1.020 ms`
  开始时间(ns): `23043054`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[28689, 512]]}`
- `attn_mha` -> 0.193 ms
  纯GPU kernel时间: `2.356 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.193 ms`, `host_to_first_kernel_gap=1.683639`, `host_end_to_last_kernel_tail=3.846 ms`, `gpu_makespan=2.356 ms`, `gpu_kernel_sum=2.356 ms`
  开始时间(ns): `23438024`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [28689, 128, 192], [28689, 128, 128]]}`
- `o_proj` -> 0.261 ms
  纯GPU kernel时间: `1.252 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.261 ms`, `host_to_first_kernel_gap=3.801676`, `host_end_to_last_kernel_tail=4.794 ms`, `gpu_makespan=1.253 ms`, `gpu_kernel_sum=1.252 ms`
  开始时间(ns): `23677812`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 1 / prefill / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `1.876 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.876 ms`, `module_to_last_kernel=14.908 ms`, `host_to_first_kernel_gap=8.514351`, `host_end_to_last_kernel_tail=13.032 ms`, `gpu_makespan=6.394 ms`, `gpu_kernel_sum=5.490 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=28689`, `chunked_req_prefix_len=20497`, `current_chunked_req_prefix_len=20497`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[8192, 128, 192], [28689, 128, 192], [28689, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.217 ms
  纯GPU kernel时间: `0.267 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.217 ms`, `host_to_first_kernel_gap=8.377538`, `host_end_to_last_kernel_tail=8.429 ms`, `gpu_makespan=0.269 ms`, `gpu_kernel_sum=0.267 ms`
  开始时间(ns): `24951584`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.049 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.049 ms`, `host_to_first_kernel_gap=8.370701`, `host_end_to_last_kernel_tail=8.340 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `25227989`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.199 ms
  纯GPU kernel时间: `0.435 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.199 ms`, `host_to_first_kernel_gap=8.311975`, `host_end_to_last_kernel_tail=8.550 ms`, `gpu_makespan=0.437 ms`, `gpu_kernel_sum=0.435 ms`
  开始时间(ns): `25305851`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.042 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.042 ms`, `host_to_first_kernel_gap=8.490299`, `host_end_to_last_kernel_tail=8.461 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `25564455`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.077 ms
  纯GPU kernel时间: `0.106 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.077 ms`, `host_to_first_kernel_gap=8.42452`, `host_end_to_last_kernel_tail=8.454 ms`, `gpu_makespan=0.106 ms`, `gpu_kernel_sum=0.106 ms`
  开始时间(ns): `25645018`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.231 ms
  纯GPU kernel时间: `1.046 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.231 ms`, `host_to_first_kernel_gap=8.301919`, `host_end_to_last_kernel_tail=9.118 ms`, `gpu_makespan=1.047 ms`, `gpu_kernel_sum=1.046 ms`
  开始时间(ns): `25933123`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[28689, 512]]}`
- `attn_mha` -> 0.144 ms
  纯GPU kernel时间: `2.357 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.144 ms`, `host_to_first_kernel_gap=9.848154`, `host_end_to_last_kernel_tail=12.061 ms`, `gpu_makespan=2.357 ms`, `gpu_kernel_sum=2.357 ms`
  开始时间(ns): `26265992`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [28689, 128, 192], [28689, 128, 128]]}`
- `o_proj` -> 0.228 ms
  纯GPU kernel时间: `1.249 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.228 ms`, `host_to_first_kernel_gap=12.022944`, `host_end_to_last_kernel_tail=13.045 ms`, `gpu_makespan=1.250 ms`, `gpu_kernel_sum=1.249 ms`
  开始时间(ns): `26449443`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 2 / prefill / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.869 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.869 ms`, `module_to_last_kernel=23.105 ms`, `host_to_first_kernel_gap=16.826361`, `host_end_to_last_kernel_tail=21.236 ms`, `gpu_makespan=6.279 ms`, `gpu_kernel_sum=5.375 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=28689`, `chunked_req_prefix_len=20497`, `current_chunked_req_prefix_len=20497`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[8192, 128, 192], [28689, 128, 192], [28689, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.219 ms
  纯GPU kernel时间: `0.266 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.219 ms`, `host_to_first_kernel_gap=16.685638`, `host_end_to_last_kernel_tail=16.734 ms`, `gpu_makespan=0.268 ms`, `gpu_kernel_sum=0.266 ms`
  开始时间(ns): `27633215`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.046 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.046 ms`, `host_to_first_kernel_gap=16.690051`, `host_end_to_last_kernel_tail=16.662 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `27897154`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.207 ms
  纯GPU kernel时间: `0.441 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.207 ms`, `host_to_first_kernel_gap=16.63488`, `host_end_to_last_kernel_tail=16.870 ms`, `gpu_makespan=0.442 ms`, `gpu_kernel_sum=0.441 ms`
  开始时间(ns): `27971589`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.045 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.045 ms`, `host_to_first_kernel_gap=16.807642`, `host_end_to_last_kernel_tail=16.776 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `28241195`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.073 ms
  纯GPU kernel时间: `0.106 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.073 ms`, `host_to_first_kernel_gap=16.741794`, `host_end_to_last_kernel_tail=16.775 ms`, `gpu_makespan=0.106 ms`, `gpu_kernel_sum=0.106 ms`
  开始时间(ns): `28322339`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.240 ms
  纯GPU kernel时间: `0.924 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.240 ms`, `host_to_first_kernel_gap=16.627781`, `host_end_to_last_kernel_tail=17.314 ms`, `gpu_makespan=0.926 ms`, `gpu_kernel_sum=0.924 ms`
  开始时间(ns): `28600864`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[28689, 512]]}`
- `attn_mha` -> 0.140 ms
  纯GPU kernel时间: `2.357 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.140 ms`, `host_to_first_kernel_gap=18.049755`, `host_end_to_last_kernel_tail=20.267 ms`, `gpu_makespan=2.357 ms`, `gpu_kernel_sum=2.357 ms`
  开始时间(ns): `28936458`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [28689, 128, 192], [28689, 128, 128]]}`
- `o_proj` -> 0.232 ms
  纯GPU kernel时间: `1.250 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.232 ms`, `host_to_first_kernel_gap=20.230183`, `host_end_to_last_kernel_tail=21.250 ms`, `gpu_makespan=1.252 ms`, `gpu_kernel_sum=1.250 ms`
  开始时间(ns): `29115807`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 3 / prefill / instance 1

- 执行序号: `4`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `1.847 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.847 ms`, `module_to_last_kernel=31.335 ms`, `host_to_first_kernel_gap=25.053458`, `host_end_to_last_kernel_tail=29.488 ms`, `gpu_makespan=6.282 ms`, `gpu_kernel_sum=5.378 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=28689`, `chunked_req_prefix_len=20497`, `current_chunked_req_prefix_len=20497`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[8192, 128, 192], [28689, 128, 192], [28689, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.217 ms
  纯GPU kernel时间: `0.266 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.217 ms`, `host_to_first_kernel_gap=24.916105`, `host_end_to_last_kernel_tail=24.967 ms`, `gpu_makespan=0.268 ms`, `gpu_kernel_sum=0.266 ms`
  开始时间(ns): `30274271`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.042 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.042 ms`, `host_to_first_kernel_gap=24.922858`, `host_end_to_last_kernel_tail=24.898 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `30535166`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.200 ms
  纯GPU kernel时间: `0.438 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.200 ms`, `host_to_first_kernel_gap=24.87319`, `host_end_to_last_kernel_tail=25.112 ms`, `gpu_makespan=0.439 ms`, `gpu_kernel_sum=0.438 ms`
  开始时间(ns): `30604482`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.040 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.040 ms`, `host_to_first_kernel_gap=25.05347`, `host_end_to_last_kernel_tail=25.027 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `30863018`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.074 ms
  纯GPU kernel时间: `0.106 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.074 ms`, `host_to_first_kernel_gap=24.992418`, `host_end_to_last_kernel_tail=25.025 ms`, `gpu_makespan=0.106 ms`, `gpu_kernel_sum=0.106 ms`
  开始时间(ns): `30939430`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.238 ms
  纯GPU kernel时间: `0.929 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.238 ms`, `host_to_first_kernel_gap=24.860972`, `host_end_to_last_kernel_tail=25.554 ms`, `gpu_makespan=0.931 ms`, `gpu_kernel_sum=0.929 ms`
  开始时间(ns): `31234716`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[28689, 512]]}`
- `attn_mha` -> 0.135 ms
  纯GPU kernel时间: `2.359 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.135 ms`, `host_to_first_kernel_gap=26.288464`, `host_end_to_last_kernel_tail=28.512 ms`, `gpu_makespan=2.359 ms`, `gpu_kernel_sum=2.359 ms`
  开始时间(ns): `31571352`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [28689, 128, 192], [28689, 128, 128]]}`
- `o_proj` -> 0.224 ms
  纯GPU kernel时间: `1.249 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.224 ms`, `host_to_first_kernel_gap=28.475558`, `host_end_to_last_kernel_tail=29.502 ms`, `gpu_makespan=1.250 ms`, `gpu_kernel_sum=1.249 ms`
  开始时间(ns): `31746051`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 4 / prefill / instance 1

- 执行序号: `5`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `1.969 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.969 ms`, `module_to_last_kernel=44.788 ms`, `host_to_first_kernel_gap=38.346453`, `host_end_to_last_kernel_tail=42.820 ms`, `gpu_makespan=6.442 ms`, `gpu_kernel_sum=5.540 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=28689`, `chunked_req_prefix_len=20497`, `current_chunked_req_prefix_len=20497`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[8192, 128, 192], [28689, 128, 192], [28689, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.282 ms
  纯GPU kernel时间: `0.272 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.282 ms`, `host_to_first_kernel_gap=38.198437`, `host_end_to_last_kernel_tail=38.191 ms`, `gpu_makespan=0.274 ms`, `gpu_kernel_sum=0.272 ms`
  开始时间(ns): `33627975`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.048 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.048 ms`, `host_to_first_kernel_gap=38.140083`, `host_end_to_last_kernel_tail=38.110 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `33961145`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.225 ms
  纯GPU kernel时间: `0.450 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.225 ms`, `host_to_first_kernel_gap=38.07881`, `host_end_to_last_kernel_tail=38.305 ms`, `gpu_makespan=0.451 ms`, `gpu_kernel_sum=0.450 ms`
  开始时间(ns): `34041586`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.040 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.040 ms`, `host_to_first_kernel_gap=38.245267`, `host_end_to_last_kernel_tail=38.219 ms`, `gpu_makespan=0.014 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `34326169`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.073 ms
  纯GPU kernel时间: `0.106 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.073 ms`, `host_to_first_kernel_gap=38.184616`, `host_end_to_last_kernel_tail=38.218 ms`, `gpu_makespan=0.106 ms`, `gpu_kernel_sum=0.106 ms`
  开始时间(ns): `34402820`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.220 ms
  纯GPU kernel时间: `1.037 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.220 ms`, `host_to_first_kernel_gap=38.069072`, `host_end_to_last_kernel_tail=38.887 ms`, `gpu_makespan=1.039 ms`, `gpu_kernel_sum=1.037 ms`
  开始时间(ns): `34682972`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[28689, 512]]}`
- `attn_mha` -> 0.141 ms
  纯GPU kernel时间: `2.361 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.141 ms`, `host_to_first_kernel_gap=39.619809`, `host_end_to_last_kernel_tail=41.839 ms`, `gpu_makespan=2.361 ms`, `gpu_kernel_sum=2.361 ms`
  开始时间(ns): `35000972`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [28689, 128, 192], [28689, 128, 128]]}`
- `o_proj` -> 0.242 ms
  纯GPU kernel时间: `1.283 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.242 ms`, `host_to_first_kernel_gap=41.790955`, `host_end_to_last_kernel_tail=42.834 ms`, `gpu_makespan=1.285 ms`, `gpu_kernel_sum=1.283 ms`
  开始时间(ns): `35192899`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 5 / prefill / instance 1

- 执行序号: `6`
- Collector对齐边界: `{'Module': 'model.model.layers.5.self_attn'}`
- 整块 MLA-module 时长: `1.935 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.935 ms`, `module_to_last_kernel=58.164 ms`, `host_to_first_kernel_gap=51.868229`, `host_end_to_last_kernel_tail=56.229 ms`, `gpu_makespan=6.296 ms`, `gpu_kernel_sum=5.393 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=28689`, `chunked_req_prefix_len=20497`, `current_chunked_req_prefix_len=20497`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[8192, 128, 192], [28689, 128, 192], [28689, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.260 ms
  纯GPU kernel时间: `0.272 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.260 ms`, `host_to_first_kernel_gap=51.727499`, `host_end_to_last_kernel_tail=51.742 ms`, `gpu_makespan=0.275 ms`, `gpu_kernel_sum=0.272 ms`
  开始时间(ns): `36875846`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.048 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.048 ms`, `host_to_first_kernel_gap=51.68502`, `host_end_to_last_kernel_tail=51.655 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `37193589`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.212 ms
  纯GPU kernel时间: `0.450 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.212 ms`, `host_to_first_kernel_gap=51.626855`, `host_end_to_last_kernel_tail=51.866 ms`, `gpu_makespan=0.452 ms`, `gpu_kernel_sum=0.450 ms`
  开始时间(ns): `37270858`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.040 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.040 ms`, `host_to_first_kernel_gap=51.8083`, `host_end_to_last_kernel_tail=51.783 ms`, `gpu_makespan=0.014 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `37540869`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.076 ms
  纯GPU kernel时间: `0.106 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.076 ms`, `host_to_first_kernel_gap=51.748784`, `host_end_to_last_kernel_tail=51.779 ms`, `gpu_makespan=0.106 ms`, `gpu_kernel_sum=0.106 ms`
  开始时间(ns): `37616161`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.231 ms
  纯GPU kernel时间: `0.933 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.231 ms`, `host_to_first_kernel_gap=51.62652`, `host_end_to_last_kernel_tail=52.330 ms`, `gpu_makespan=0.934 ms`, `gpu_kernel_sum=0.933 ms`
  开始时间(ns): `37903929`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[28689, 512]]}`
- `attn_mha` -> 0.144 ms
  纯GPU kernel时间: `2.359 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.144 ms`, `host_to_first_kernel_gap=53.055947`, `host_end_to_last_kernel_tail=55.270 ms`, `gpu_makespan=2.359 ms`, `gpu_kernel_sum=2.359 ms`
  开始时间(ns): `38241638`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [28689, 128, 192], [28689, 128, 128]]}`
- `o_proj` -> 0.229 ms
  纯GPU kernel时间: `1.240 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.229 ms`, `host_to_first_kernel_gap=55.229779`, `host_end_to_last_kernel_tail=56.242 ms`, `gpu_makespan=1.241 ms`, `gpu_kernel_sum=1.240 ms`
  开始时间(ns): `38428095`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`
