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
- 整块 MLA-module 时长: `3.131 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=3.131 ms`, `module_to_last_kernel=4.028 ms`, `host_to_first_kernel_gap=0.382038`, `host_end_to_last_kernel_tail=0.897 ms`, `gpu_makespan=3.646 ms`, `gpu_kernel_sum=1.527 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=2056`, `total_tokens=12323`, `chunked_req_prefix_len=10267`, `current_chunked_req_prefix_len=10267`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[2056, 128, 192], [12323, 128, 192], [12323, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.451 ms
  纯GPU kernel时间: `0.080 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.451 ms`, `host_to_first_kernel_gap=0.154913`, `host_end_to_last_kernel_tail=0.021 ms`, `gpu_makespan=0.317 ms`, `gpu_kernel_sum=0.080 ms`
  开始时间(ns): `24461176`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2056, 7168]]}`
- `q_a_layernorm` -> 0.062 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.062 ms`, `host_to_first_kernel_gap=0.055754`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `24980431`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2056, 1536]]}`
- `q_b_proj` -> 0.266 ms
  纯GPU kernel时间: `0.134 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.266 ms`, `host_to_first_kernel_gap=0.098562`, `host_end_to_last_kernel_tail=0.101 ms`, `gpu_makespan=0.269 ms`, `gpu_kernel_sum=0.134 ms`
  开始时间(ns): `25076151`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2056, 1536]]}`
- `kv_a_layernorm` -> 0.047 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.047 ms`, `host_to_first_kernel_gap=0.041886`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `25410107`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2056, 512]]}`
- `rotary_emb` -> 0.103 ms
  纯GPU kernel时间: `0.027 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.103 ms`, `host_to_first_kernel_gap=0.088015`, `host_end_to_last_kernel_tail=0.012 ms`, `gpu_makespan=0.027 ms`, `gpu_kernel_sum=0.027 ms`
  开始时间(ns): `25503786`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2056], [2056, 128, 64], [2056, 1, 64]]}`
- `kv_b_proj` -> 0.347 ms
  纯GPU kernel时间: `0.414 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.347 ms`, `host_to_first_kernel_gap=0.11636`, `host_end_to_last_kernel_tail=0.372 ms`, `gpu_makespan=0.603 ms`, `gpu_kernel_sum=0.414 ms`
  开始时间(ns): `26315120`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[12323, 512]]}`
- `attn_mha` -> 0.226 ms
  纯GPU kernel时间: `0.486 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.226 ms`, `host_to_first_kernel_gap=0.601586`, `host_end_to_last_kernel_tail=0.862 ms`, `gpu_makespan=0.486 ms`, `gpu_kernel_sum=0.486 ms`
  开始时间(ns): `26797126`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2056, 128, 192], [12323, 128, 192], [12323, 128, 128]]}`
- `o_proj` -> 0.279 ms
  纯GPU kernel时间: `0.375 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.279 ms`, `host_to_first_kernel_gap=0.817204`, `host_end_to_last_kernel_tail=0.914 ms`, `gpu_makespan=0.376 ms`, `gpu_kernel_sum=0.375 ms`
  开始时间(ns): `27068996`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2056, 16384]]}`

## Layer 1 / prefill / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `2.226 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.226 ms`, `module_to_last_kernel=3.235 ms`, `host_to_first_kernel_gap=1.319473`, `host_end_to_last_kernel_tail=1.009 ms`, `gpu_makespan=1.915 ms`, `gpu_kernel_sum=1.517 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=2056`, `total_tokens=12323`, `chunked_req_prefix_len=10267`, `current_chunked_req_prefix_len=10267`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[2056, 128, 192], [12323, 128, 192], [12323, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.278 ms
  纯GPU kernel时间: `0.081 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.278 ms`, `host_to_first_kernel_gap=1.127112`, `host_end_to_last_kernel_tail=0.931 ms`, `gpu_makespan=0.082 ms`, `gpu_kernel_sum=0.081 ms`
  开始时间(ns): `28463536`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2056, 7168]]}`
- `q_a_layernorm` -> 0.060 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.060 ms`, `host_to_first_kernel_gap=0.876363`, `host_end_to_last_kernel_tail=0.822 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `28796429`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2056, 1536]]}`
- `q_b_proj` -> 0.230 ms
  纯GPU kernel时间: `0.130 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.230 ms`, `host_to_first_kernel_gap=0.793727`, `host_end_to_last_kernel_tail=0.695 ms`, `gpu_makespan=0.131 ms`, `gpu_kernel_sum=0.130 ms`
  开始时间(ns): `28885881`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2056, 1536]]}`
- `kv_a_layernorm` -> 0.046 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.046 ms`, `host_to_first_kernel_gap=0.633879`, `host_end_to_last_kernel_tail=0.592 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `29177121`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2056, 512]]}`
- `rotary_emb` -> 0.099 ms
  纯GPU kernel时间: `0.027 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.099 ms`, `host_to_first_kernel_gap=0.551922`, `host_end_to_last_kernel_tail=0.479 ms`, `gpu_makespan=0.027 ms`, `gpu_kernel_sum=0.027 ms`
  开始时间(ns): `29266278`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2056], [2056, 128, 64], [2056, 1, 64]]}`
- `kv_b_proj` -> 0.261 ms
  纯GPU kernel时间: `0.409 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.261 ms`, `host_to_first_kernel_gap=0.229357`, `host_end_to_last_kernel_tail=0.378 ms`, `gpu_makespan=0.410 ms`, `gpu_kernel_sum=0.409 ms`
  开始时间(ns): `29640587`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[12323, 512]]}`
- `attn_mha` -> 0.184 ms
  纯GPU kernel时间: `0.486 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.184 ms`, `host_to_first_kernel_gap=0.629095`, `host_end_to_last_kernel_tail=0.931 ms`, `gpu_makespan=0.486 ms`, `gpu_kernel_sum=0.486 ms`
  开始时间(ns): `30014128`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2056, 128, 192], [12323, 128, 192], [12323, 128, 128]]}`
- `o_proj` -> 0.239 ms
  纯GPU kernel时间: `0.374 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.239 ms`, `host_to_first_kernel_gap=0.889209`, `host_end_to_last_kernel_tail=1.026 ms`, `gpu_makespan=0.375 ms`, `gpu_kernel_sum=0.374 ms`
  开始时间(ns): `30241470`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2056, 16384]]}`

## Layer 2 / prefill / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `2.159 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.159 ms`, `module_to_last_kernel=3.400 ms`, `host_to_first_kernel_gap=1.472092`, `host_end_to_last_kernel_tail=1.241 ms`, `gpu_makespan=1.928 ms`, `gpu_kernel_sum=1.529 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=2056`, `total_tokens=12323`, `chunked_req_prefix_len=10267`, `current_chunked_req_prefix_len=10267`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[2056, 128, 192], [12323, 128, 192], [12323, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.269 ms
  纯GPU kernel时间: `0.080 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.269 ms`, `host_to_first_kernel_gap=1.291235`, `host_end_to_last_kernel_tail=1.103 ms`, `gpu_makespan=0.081 ms`, `gpu_kernel_sum=0.080 ms`
  开始时间(ns): `31545364`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2056, 7168]]}`
- `q_a_layernorm` -> 0.057 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.057 ms`, `host_to_first_kernel_gap=1.049438`, `host_end_to_last_kernel_tail=0.998 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `31868185`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2056, 1536]]}`
- `q_b_proj` -> 0.216 ms
  纯GPU kernel时间: `0.130 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.216 ms`, `host_to_first_kernel_gap=0.969871`, `host_end_to_last_kernel_tail=0.885 ms`, `gpu_makespan=0.131 ms`, `gpu_kernel_sum=0.130 ms`
  开始时间(ns): `31955752`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2056, 1536]]}`
- `kv_a_layernorm` -> 0.043 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.043 ms`, `host_to_first_kernel_gap=0.825279`, `host_end_to_last_kernel_tail=0.787 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `32231672`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2056, 512]]}`
- `rotary_emb` -> 0.094 ms
  纯GPU kernel时间: `0.027 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.094 ms`, `host_to_first_kernel_gap=0.747622`, `host_end_to_last_kernel_tail=0.680 ms`, `gpu_makespan=0.027 ms`, `gpu_kernel_sum=0.027 ms`
  开始时间(ns): `32316753`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2056], [2056, 128, 64], [2056, 1, 64]]}`
- `kv_b_proj` -> 0.242 ms
  纯GPU kernel时间: `0.414 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.242 ms`, `host_to_first_kernel_gap=0.439041`, `host_end_to_last_kernel_tail=0.612 ms`, `gpu_makespan=0.415 ms`, `gpu_kernel_sum=0.414 ms`
  开始时间(ns): `32678678`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[12323, 512]]}`
- `attn_mha` -> 0.179 ms
  纯GPU kernel时间: `0.487 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.179 ms`, `host_to_first_kernel_gap=0.865516`, `host_end_to_last_kernel_tail=1.173 ms`, `gpu_makespan=0.487 ms`, `gpu_kernel_sum=0.487 ms`
  开始时间(ns): `33029003`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2056, 128, 192], [12323, 128, 192], [12323, 128, 128]]}`
- `o_proj` -> 0.252 ms
  纯GPU kernel时间: `0.380 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.252 ms`, `host_to_first_kernel_gap=1.128565`, `host_end_to_last_kernel_tail=1.258 ms`, `gpu_makespan=0.381 ms`, `gpu_kernel_sum=0.380 ms`
  开始时间(ns): `33254785`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2056, 16384]]}`

## Layer 3 / prefill / instance 1

- 执行序号: `4`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `2.129 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.129 ms`, `module_to_last_kernel=3.634 ms`, `host_to_first_kernel_gap=1.712526`, `host_end_to_last_kernel_tail=1.505 ms`, `gpu_makespan=1.922 ms`, `gpu_kernel_sum=1.516 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=2056`, `total_tokens=12323`, `chunked_req_prefix_len=10267`, `current_chunked_req_prefix_len=10267`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[2056, 128, 192], [12323, 128, 192], [12323, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.254 ms
  纯GPU kernel时间: `0.080 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.254 ms`, `host_to_first_kernel_gap=1.531393`, `host_end_to_last_kernel_tail=1.359 ms`, `gpu_makespan=0.082 ms`, `gpu_kernel_sum=0.080 ms`
  开始时间(ns): `34560917`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2056, 7168]]}`
- `q_a_layernorm` -> 0.053 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.053 ms`, `host_to_first_kernel_gap=1.306843`, `host_end_to_last_kernel_tail=1.259 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `34867867`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2056, 1536]]}`
- `q_b_proj` -> 0.214 ms
  纯GPU kernel时间: `0.130 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.214 ms`, `host_to_first_kernel_gap=1.23385`, `host_end_to_last_kernel_tail=1.153 ms`, `gpu_makespan=0.133 ms`, `gpu_kernel_sum=0.130 ms`
  开始时间(ns): `34949308`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2056, 1536]]}`
- `kv_a_layernorm` -> 0.044 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.044 ms`, `host_to_first_kernel_gap=1.092874`, `host_end_to_last_kernel_tail=1.054 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `35222860`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2056, 512]]}`
- `rotary_emb` -> 0.094 ms
  纯GPU kernel时间: `0.027 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.094 ms`, `host_to_first_kernel_gap=1.013428`, `host_end_to_last_kernel_tail=0.946 ms`, `gpu_makespan=0.027 ms`, `gpu_kernel_sum=0.027 ms`
  开始时间(ns): `35308258`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2056], [2056, 128, 64], [2056, 1, 64]]}`
- `kv_b_proj` -> 0.254 ms
  纯GPU kernel时间: `0.414 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.254 ms`, `host_to_first_kernel_gap=0.700794`, `host_end_to_last_kernel_tail=0.862 ms`, `gpu_makespan=0.415 ms`, `gpu_kernel_sum=0.414 ms`
  开始时间(ns): `35675068`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[12323, 512]]}`
- `attn_mha` -> 0.172 ms
  纯GPU kernel时间: `0.486 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.172 ms`, `host_to_first_kernel_gap=1.118489`, `host_end_to_last_kernel_tail=1.433 ms`, `gpu_makespan=0.486 ms`, `gpu_kernel_sum=0.486 ms`
  开始时间(ns): `36037693`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2056, 128, 192], [12323, 128, 192], [12323, 128, 128]]}`
- `o_proj` -> 0.241 ms
  纯GPU kernel时间: `0.368 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.241 ms`, `host_to_first_kernel_gap=1.392148`, `host_end_to_last_kernel_tail=1.521 ms`, `gpu_makespan=0.370 ms`, `gpu_kernel_sum=0.368 ms`
  开始时间(ns): `36251970`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2056, 16384]]}`

## Layer 4 / prefill / instance 1

- 执行序号: `5`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `2.333 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.333 ms`, `module_to_last_kernel=4.423 ms`, `host_to_first_kernel_gap=2.496513`, `host_end_to_last_kernel_tail=2.090 ms`, `gpu_makespan=1.927 ms`, `gpu_kernel_sum=1.521 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=2056`, `total_tokens=12323`, `chunked_req_prefix_len=10267`, `current_chunked_req_prefix_len=10267`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[2056, 128, 192], [12323, 128, 192], [12323, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.344 ms
  纯GPU kernel时间: `0.080 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.344 ms`, `host_to_first_kernel_gap=2.297218`, `host_end_to_last_kernel_tail=2.034 ms`, `gpu_makespan=0.081 ms`, `gpu_kernel_sum=0.080 ms`
  开始时间(ns): `38369043`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2056, 7168]]}`
- `q_a_layernorm` -> 0.063 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.063 ms`, `host_to_first_kernel_gap=1.974571`, `host_end_to_last_kernel_tail=1.917 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `38773130`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2056, 1536]]}`
- `q_b_proj` -> 0.243 ms
  纯GPU kernel时间: `0.130 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.243 ms`, `host_to_first_kernel_gap=1.886824`, `host_end_to_last_kernel_tail=1.775 ms`, `gpu_makespan=0.131 ms`, `gpu_kernel_sum=0.130 ms`
  开始时间(ns): `38868429`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2056, 1536]]}`
- `kv_a_layernorm` -> 0.046 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.046 ms`, `host_to_first_kernel_gap=1.711086`, `host_end_to_last_kernel_tail=1.671 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `39175431`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2056, 512]]}`
- `rotary_emb` -> 0.099 ms
  纯GPU kernel时间: `0.027 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.099 ms`, `host_to_first_kernel_gap=1.628229`, `host_end_to_last_kernel_tail=1.556 ms`, `gpu_makespan=0.027 ms`, `gpu_kernel_sum=0.027 ms`
  开始时间(ns): `39265968`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2056], [2056, 128, 64], [2056, 1, 64]]}`
- `kv_b_proj` -> 0.264 ms
  纯GPU kernel时间: `0.414 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.264 ms`, `host_to_first_kernel_gap=1.337972`, `host_end_to_last_kernel_tail=1.490 ms`, `gpu_makespan=0.416 ms`, `gpu_kernel_sum=0.414 ms`
  开始时间(ns): `39609985`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[12323, 512]]}`
- `attn_mha` -> 0.183 ms
  纯GPU kernel时间: `0.485 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.183 ms`, `host_to_first_kernel_gap=1.743869`, `host_end_to_last_kernel_tail=2.046 ms`, `gpu_makespan=0.485 ms`, `gpu_kernel_sum=0.485 ms`
  开始时间(ns): `39984023`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2056, 128, 192], [12323, 128, 192], [12323, 128, 128]]}`
- `o_proj` -> 0.272 ms
  纯GPU kernel时间: `0.374 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.272 ms`, `host_to_first_kernel_gap=2.002046`, `host_end_to_last_kernel_tail=2.107 ms`, `gpu_makespan=0.377 ms`, `gpu_kernel_sum=0.374 ms`
  开始时间(ns): `40214230`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2056, 16384]]}`

## Layer 5 / prefill / instance 1

- 执行序号: `6`
- Collector对齐边界: `{'Module': 'model.model.layers.5.self_attn'}`
- 整块 MLA-module 时长: `2.286 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.286 ms`, `module_to_last_kernel=5.121 ms`, `host_to_first_kernel_gap=3.201247`, `host_end_to_last_kernel_tail=2.835 ms`, `gpu_makespan=1.920 ms`, `gpu_kernel_sum=1.518 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=2056`, `total_tokens=12323`, `chunked_req_prefix_len=10267`, `current_chunked_req_prefix_len=10267`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[2056, 128, 192], [12323, 128, 192], [12323, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.311 ms
  纯GPU kernel时间: `0.080 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.311 ms`, `host_to_first_kernel_gap=3.008493`, `host_end_to_last_kernel_tail=2.779 ms`, `gpu_makespan=0.082 ms`, `gpu_kernel_sum=0.080 ms`
  开始时间(ns): `42240359`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2056, 7168]]}`
- `q_a_layernorm` -> 0.078 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.078 ms`, `host_to_first_kernel_gap=2.720959`, `host_end_to_last_kernel_tail=2.648 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `42609717`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2056, 1536]]}`
- `q_b_proj` -> 0.249 ms
  纯GPU kernel时间: `0.130 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.249 ms`, `host_to_first_kernel_gap=2.617498`, `host_end_to_last_kernel_tail=2.500 ms`, `gpu_makespan=0.131 ms`, `gpu_kernel_sum=0.130 ms`
  开始时间(ns): `42719738`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2056, 1536]]}`
- `kv_a_layernorm` -> 0.052 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.052 ms`, `host_to_first_kernel_gap=2.437059`, `host_end_to_last_kernel_tail=2.390 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `43031152`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2056, 512]]}`
- `rotary_emb` -> 0.098 ms
  纯GPU kernel时间: `0.026 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.098 ms`, `host_to_first_kernel_gap=2.350261`, `host_end_to_last_kernel_tail=2.279 ms`, `gpu_makespan=0.026 ms`, `gpu_kernel_sum=0.026 ms`
  开始时间(ns): `43125566`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2056], [2056, 128, 64], [2056, 1, 64]]}`
- `kv_b_proj` -> 0.262 ms
  纯GPU kernel时间: `0.410 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.262 ms`, `host_to_first_kernel_gap=2.055559`, `host_end_to_last_kernel_tail=2.206 ms`, `gpu_makespan=0.413 ms`, `gpu_kernel_sum=0.410 ms`
  开始时间(ns): `43473676`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[12323, 512]]}`
- `attn_mha` -> 0.186 ms
  纯GPU kernel时间: `0.486 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.186 ms`, `host_to_first_kernel_gap=2.46007`, `host_end_to_last_kernel_tail=2.760 ms`, `gpu_makespan=0.486 ms`, `gpu_kernel_sum=0.486 ms`
  开始时间(ns): `43843917`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2056, 128, 192], [12323, 128, 192], [12323, 128, 128]]}`
- `o_proj` -> 0.244 ms
  纯GPU kernel时间: `0.376 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.244 ms`, `host_to_first_kernel_gap=2.719136`, `host_end_to_last_kernel_tail=2.852 ms`, `gpu_makespan=0.377 ms`, `gpu_kernel_sum=0.376 ms`
  开始时间(ns): `44072659`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2056, 16384]]}`
