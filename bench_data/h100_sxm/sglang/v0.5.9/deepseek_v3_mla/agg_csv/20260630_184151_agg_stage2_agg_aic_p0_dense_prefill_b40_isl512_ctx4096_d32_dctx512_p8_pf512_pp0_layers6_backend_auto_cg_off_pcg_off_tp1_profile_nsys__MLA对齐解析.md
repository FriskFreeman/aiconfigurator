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
- 整块 MLA-module 时长: `3.742 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=3.742 ms`, `module_to_last_kernel=5.857 ms`, `host_to_first_kernel_gap=0.430753`, `host_end_to_last_kernel_tail=2.115 ms`, `gpu_makespan=5.427 ms`, `gpu_kernel_sum=2.718 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4128`, `total_tokens=20633`, `chunked_req_prefix_len=16505`, `current_chunked_req_prefix_len=16505`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[4128, 128, 192], [20633, 128, 192], [20633, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.405 ms
  纯GPU kernel时间: `0.142 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.405 ms`, `host_to_first_kernel_gap=0.133952`, `host_end_to_last_kernel_tail=0.069 ms`, `gpu_makespan=0.340 ms`, `gpu_kernel_sum=0.142 ms`
  开始时间(ns): `22678546`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4128, 7168]]}`
- `q_a_layernorm` -> 0.049 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.049 ms`, `host_to_first_kernel_gap=0.043178`, `host_end_to_last_kernel_tail=0.004 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `23143368`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4128, 1536]]}`
- `q_b_proj` -> 0.245 ms
  纯GPU kernel时间: `0.242 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.245 ms`, `host_to_first_kernel_gap=0.088684`, `host_end_to_last_kernel_tail=0.207 ms`, `gpu_makespan=0.362 ms`, `gpu_kernel_sum=0.242 ms`
  开始时间(ns): `23229830`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4128, 1536]]}`
- `kv_a_layernorm` -> 0.042 ms
  纯GPU kernel时间: `0.008 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.042 ms`, `host_to_first_kernel_gap=0.143185`, `host_end_to_last_kernel_tail=0.110 ms`, `gpu_makespan=0.008 ms`, `gpu_kernel_sum=0.008 ms`
  开始时间(ns): `23537633`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4128, 512]]}`
- `rotary_emb` -> 0.099 ms
  纯GPU kernel时间: `0.053 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.099 ms`, `host_to_first_kernel_gap=0.085296`, `host_end_to_last_kernel_tail=0.039 ms`, `gpu_makespan=0.053 ms`, `gpu_kernel_sum=0.053 ms`
  开始时间(ns): `23621090`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4128], [4128, 128, 64], [4128, 1, 64]]}`
- `kv_b_proj` -> 1.111 ms
  纯GPU kernel时间: `0.747 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=1.111 ms`, `host_to_first_kernel_gap=0.095044`, `host_end_to_last_kernel_tail=0.687 ms`, `gpu_makespan=1.703 ms`, `gpu_kernel_sum=0.747 ms`
  开始时间(ns): `24319661`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[20633, 512]]}`
- `attn_mha` -> 0.180 ms
  纯GPU kernel时间: `0.805 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.180 ms`, `host_to_first_kernel_gap=1.128445`, `host_end_to_last_kernel_tail=1.753 ms`, `gpu_makespan=0.805 ms`, `gpu_kernel_sum=0.805 ms`
  开始时间(ns): `25589300`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4128, 128, 192], [20633, 128, 192], [20633, 128, 128]]}`
- `o_proj` -> 0.291 ms
  纯GPU kernel时间: `0.713 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.291 ms`, `host_to_first_kernel_gap=1.707149`, `host_end_to_last_kernel_tail=2.131 ms`, `gpu_makespan=0.714 ms`, `gpu_kernel_sum=0.713 ms`
  开始时间(ns): `25817572`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4128, 16384]]}`

## Layer 1 / prefill / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `2.044 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.044 ms`, `module_to_last_kernel=7.081 ms`, `host_to_first_kernel_gap=3.758733`, `host_end_to_last_kernel_tail=5.037 ms`, `gpu_makespan=3.322 ms`, `gpu_kernel_sum=2.669 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4128`, `total_tokens=20633`, `chunked_req_prefix_len=16505`, `current_chunked_req_prefix_len=16505`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[4128, 128, 192], [20633, 128, 192], [20633, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.252 ms
  纯GPU kernel时间: `0.141 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.252 ms`, `host_to_first_kernel_gap=3.499782`, `host_end_to_last_kernel_tail=3.392 ms`, `gpu_makespan=0.144 ms`, `gpu_kernel_sum=0.141 ms`
  开始时间(ns): `27255978`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4128, 7168]]}`
- `q_a_layernorm` -> 0.050 ms
  纯GPU kernel时间: `0.010 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.050 ms`, `host_to_first_kernel_gap=3.338137`, `host_end_to_last_kernel_tail=3.297 ms`, `gpu_makespan=0.010 ms`, `gpu_kernel_sum=0.010 ms`
  开始时间(ns): `27561399`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4128, 1536]]}`
- `q_b_proj` -> 0.213 ms
  纯GPU kernel时间: `0.235 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.213 ms`, `host_to_first_kernel_gap=3.268303`, `host_end_to_last_kernel_tail=3.291 ms`, `gpu_makespan=0.236 ms`, `gpu_kernel_sum=0.235 ms`
  开始时间(ns): `27642241`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4128, 1536]]}`
- `kv_a_layernorm` -> 0.042 ms
  纯GPU kernel时间: `0.008 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.042 ms`, `host_to_first_kernel_gap=3.232155`, `host_end_to_last_kernel_tail=3.198 ms`, `gpu_makespan=0.008 ms`, `gpu_kernel_sum=0.008 ms`
  开始时间(ns): `27914709`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4128, 512]]}`
- `rotary_emb` -> 0.092 ms
  纯GPU kernel时间: `0.053 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.092 ms`, `host_to_first_kernel_gap=3.164124`, `host_end_to_last_kernel_tail=3.125 ms`, `gpu_makespan=0.053 ms`, `gpu_kernel_sum=0.053 ms`
  开始时间(ns): `27992756`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4128], [4128, 128, 64], [4128, 1, 64]]}`
- `kv_b_proj` -> 0.221 ms
  纯GPU kernel时间: `0.713 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.221 ms`, `host_to_first_kernel_gap=2.943688`, `host_end_to_last_kernel_tail=3.437 ms`, `gpu_makespan=0.715 ms`, `gpu_kernel_sum=0.713 ms`
  开始时间(ns): `28306408`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[20633, 512]]}`
- `attn_mha` -> 0.141 ms
  纯GPU kernel时间: `0.798 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.141 ms`, `host_to_first_kernel_gap=3.942084`, `host_end_to_last_kernel_tail=4.599 ms`, `gpu_makespan=0.798 ms`, `gpu_kernel_sum=0.798 ms`
  开始时间(ns): `28622763`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4128, 128, 192], [20633, 128, 192], [20633, 128, 128]]}`
- `o_proj` -> 0.225 ms
  纯GPU kernel时间: `0.712 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.225 ms`, `host_to_first_kernel_gap=4.562797`, `host_end_to_last_kernel_tail=5.051 ms`, `gpu_makespan=0.713 ms`, `gpu_kernel_sum=0.712 ms`
  开始时间(ns): `28802242`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4128, 16384]]}`

## Layer 2 / prefill / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.944 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.944 ms`, `module_to_last_kernel=10.040 ms`, `host_to_first_kernel_gap=6.744244`, `host_end_to_last_kernel_tail=8.096 ms`, `gpu_makespan=3.296 ms`, `gpu_kernel_sum=2.638 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4128`, `total_tokens=20633`, `chunked_req_prefix_len=16505`, `current_chunked_req_prefix_len=16505`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[4128, 128, 192], [20633, 128, 192], [20633, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.231 ms
  纯GPU kernel时间: `0.148 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.231 ms`, `host_to_first_kernel_gap=6.500223`, `host_end_to_last_kernel_tail=6.419 ms`, `gpu_makespan=0.149 ms`, `gpu_kernel_sum=0.148 ms`
  开始时间(ns): `30094255`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4128, 7168]]}`
- `q_a_layernorm` -> 0.046 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.046 ms`, `host_to_first_kernel_gap=6.369796`, `host_end_to_last_kernel_tail=6.333 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `30373802`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4128, 1536]]}`
- `q_b_proj` -> 0.202 ms
  纯GPU kernel时间: `0.235 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.202 ms`, `host_to_first_kernel_gap=6.305901`, `host_end_to_last_kernel_tail=6.340 ms`, `gpu_makespan=0.236 ms`, `gpu_kernel_sum=0.235 ms`
  开始时间(ns): `30449505`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4128, 1536]]}`
- `kv_a_layernorm` -> 0.040 ms
  纯GPU kernel时间: `0.008 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.040 ms`, `host_to_first_kernel_gap=6.282204`, `host_end_to_last_kernel_tail=6.250 ms`, `gpu_makespan=0.008 ms`, `gpu_kernel_sum=0.008 ms`
  开始时间(ns): `30709458`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4128, 512]]}`
- `rotary_emb` -> 0.076 ms
  纯GPU kernel时间: `0.052 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.076 ms`, `host_to_first_kernel_gap=6.214153`, `host_end_to_last_kernel_tail=6.190 ms`, `gpu_makespan=0.052 ms`, `gpu_kernel_sum=0.052 ms`
  开始时间(ns): `30788133`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4128], [4128, 128, 64], [4128, 1, 64]]}`
- `kv_b_proj` -> 0.219 ms
  纯GPU kernel时间: `0.671 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.219 ms`, `host_to_first_kernel_gap=6.0139`, `host_end_to_last_kernel_tail=6.469 ms`, `gpu_makespan=0.673 ms`, `gpu_kernel_sum=0.671 ms`
  开始时间(ns): `31081826`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[20633, 512]]}`
- `attn_mha` -> 0.140 ms
  纯GPU kernel时间: `0.801 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.140 ms`, `host_to_first_kernel_gap=6.978923`, `host_end_to_last_kernel_tail=7.641 ms`, `gpu_makespan=0.802 ms`, `gpu_kernel_sum=0.801 ms`
  开始时间(ns): `31393731`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4128, 128, 192], [20633, 128, 192], [20633, 128, 128]]}`
- `o_proj` -> 0.212 ms
  纯GPU kernel时间: `0.712 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.212 ms`, `host_to_first_kernel_gap=7.608512`, `host_end_to_last_kernel_tail=8.110 ms`, `gpu_makespan=0.714 ms`, `gpu_kernel_sum=0.712 ms`
  开始时间(ns): `31568174`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4128, 16384]]}`

## Layer 3 / prefill / instance 1

- 执行序号: `4`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `1.926 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.926 ms`, `module_to_last_kernel=13.150 ms`, `host_to_first_kernel_gap=9.80317`, `host_end_to_last_kernel_tail=11.224 ms`, `gpu_makespan=3.347 ms`, `gpu_kernel_sum=2.690 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4128`, `total_tokens=20633`, `chunked_req_prefix_len=16505`, `current_chunked_req_prefix_len=16505`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[4128, 128, 192], [20633, 128, 192], [20633, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.224 ms
  纯GPU kernel时间: `0.141 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.224 ms`, `host_to_first_kernel_gap=9.562981`, `host_end_to_last_kernel_tail=9.481 ms`, `gpu_makespan=0.142 ms`, `gpu_kernel_sum=0.141 ms`
  开始时间(ns): `32842312`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4128, 7168]]}`
- `q_a_layernorm` -> 0.045 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.045 ms`, `host_to_first_kernel_gap=9.431299`, `host_end_to_last_kernel_tail=9.395 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `33116650`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4128, 1536]]}`
- `q_b_proj` -> 0.205 ms
  纯GPU kernel时间: `0.235 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.205 ms`, `host_to_first_kernel_gap=9.367264`, `host_end_to_last_kernel_tail=9.400 ms`, `gpu_makespan=0.238 ms`, `gpu_kernel_sum=0.235 ms`
  开始时间(ns): `33192141`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4128, 1536]]}`
- `kv_a_layernorm` -> 0.039 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.039 ms`, `host_to_first_kernel_gap=9.341075`, `host_end_to_last_kernel_tail=9.310 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `33455770`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4128, 512]]}`
- `rotary_emb` -> 0.072 ms
  纯GPU kernel时间: `0.053 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.072 ms`, `host_to_first_kernel_gap=9.27115`, `host_end_to_last_kernel_tail=9.252 ms`, `gpu_makespan=0.053 ms`, `gpu_kernel_sum=0.053 ms`
  开始时间(ns): `33535615`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4128], [4128, 128, 64], [4128, 1, 64]]}`
- `kv_b_proj` -> 0.218 ms
  纯GPU kernel时间: `0.727 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.218 ms`, `host_to_first_kernel_gap=9.085802`, `host_end_to_last_kernel_tail=9.597 ms`, `gpu_makespan=0.728 ms`, `gpu_kernel_sum=0.727 ms`
  开始时间(ns): `33815363`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[20633, 512]]}`
- `attn_mha` -> 0.135 ms
  纯GPU kernel时间: `0.805 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.135 ms`, `host_to_first_kernel_gap=10.104133`, `host_end_to_last_kernel_tail=10.774 ms`, `gpu_makespan=0.805 ms`, `gpu_kernel_sum=0.805 ms`
  开始时间(ns): `34127911`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4128, 128, 192], [20633, 128, 192], [20633, 128, 128]]}`
- `o_proj` -> 0.214 ms
  纯GPU kernel时间: `0.711 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.214 ms`, `host_to_first_kernel_gap=10.738088`, `host_end_to_last_kernel_tail=11.238 ms`, `gpu_makespan=0.714 ms`, `gpu_kernel_sum=0.711 ms`
  开始时间(ns): `34300100`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4128, 16384]]}`

## Layer 4 / prefill / instance 1

- 执行序号: `5`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `2.134 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.134 ms`, `module_to_last_kernel=18.316 ms`, `host_to_first_kernel_gap=14.991003`, `host_end_to_last_kernel_tail=16.181 ms`, `gpu_makespan=3.325 ms`, `gpu_kernel_sum=2.670 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4128`, `total_tokens=20633`, `chunked_req_prefix_len=16505`, `current_chunked_req_prefix_len=16505`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[4128, 128, 192], [20633, 128, 192], [20633, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.289 ms
  纯GPU kernel时间: `0.141 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.289 ms`, `host_to_first_kernel_gap=14.720116`, `host_end_to_last_kernel_tail=14.574 ms`, `gpu_makespan=0.143 ms`, `gpu_kernel_sum=0.141 ms`
  开始时间(ns): `36283991`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4128, 7168]]}`
- `q_a_layernorm` -> 0.049 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.049 ms`, `host_to_first_kernel_gap=14.522025`, `host_end_to_last_kernel_tail=14.482 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `36625346`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4128, 1536]]}`
- `q_b_proj` -> 0.222 ms
  纯GPU kernel时间: `0.235 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.222 ms`, `host_to_first_kernel_gap=14.451068`, `host_end_to_last_kernel_tail=14.465 ms`, `gpu_makespan=0.236 ms`, `gpu_kernel_sum=0.235 ms`
  开始时间(ns): `36707119`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4128, 1536]]}`
- `kv_a_layernorm` -> 0.047 ms
  纯GPU kernel时间: `0.008 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.047 ms`, `host_to_first_kernel_gap=14.40083`, `host_end_to_last_kernel_tail=14.362 ms`, `gpu_makespan=0.008 ms`, `gpu_kernel_sum=0.008 ms`
  开始时间(ns): `36993197`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4128, 512]]}`
- `rotary_emb` -> 0.076 ms
  纯GPU kernel时间: `0.053 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.076 ms`, `host_to_first_kernel_gap=14.325038`, `host_end_to_last_kernel_tail=14.302 ms`, `gpu_makespan=0.053 ms`, `gpu_kernel_sum=0.053 ms`
  开始时间(ns): `37079453`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4128], [4128, 128, 64], [4128, 1, 64]]}`
- `kv_b_proj` -> 0.241 ms
  纯GPU kernel时间: `0.712 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.241 ms`, `host_to_first_kernel_gap=14.131115`, `host_end_to_last_kernel_tail=14.604 ms`, `gpu_makespan=0.713 ms`, `gpu_kernel_sum=0.712 ms`
  开始时间(ns): `37367488`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[20633, 512]]}`
- `attn_mha` -> 0.143 ms
  纯GPU kernel时间: `0.800 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.143 ms`, `host_to_first_kernel_gap=15.103509`, `host_end_to_last_kernel_tail=15.761 ms`, `gpu_makespan=0.800 ms`, `gpu_kernel_sum=0.800 ms`
  开始时间(ns): `37708597`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4128, 128, 192], [20633, 128, 192], [20633, 128, 128]]}`
- `o_proj` -> 0.244 ms
  纯GPU kernel时间: `0.712 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.244 ms`, `host_to_first_kernel_gap=15.724159`, `host_end_to_last_kernel_tail=16.194 ms`, `gpu_makespan=0.714 ms`, `gpu_kernel_sum=0.712 ms`
  开始时间(ns): `37890411`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4128, 16384]]}`

## Layer 5 / prefill / instance 1

- 执行序号: `6`
- Collector对齐边界: `{'Module': 'model.model.layers.5.self_attn'}`
- 整块 MLA-module 时长: `2.047 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.047 ms`, `module_to_last_kernel=23.499 ms`, `host_to_first_kernel_gap=20.138343`, `host_end_to_last_kernel_tail=21.453 ms`, `gpu_makespan=3.361 ms`, `gpu_kernel_sum=2.709 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4128`, `total_tokens=20633`, `chunked_req_prefix_len=16505`, `current_chunked_req_prefix_len=16505`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[4128, 128, 192], [20633, 128, 192], [20633, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.260 ms
  纯GPU kernel时间: `0.144 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.260 ms`, `host_to_first_kernel_gap=19.899555`, `host_end_to_last_kernel_tail=19.786 ms`, `gpu_makespan=0.146 ms`, `gpu_kernel_sum=0.144 ms`
  开始时间(ns): `39671142`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4128, 7168]]}`
- `q_a_layernorm` -> 0.062 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.062 ms`, `host_to_first_kernel_gap=19.735295`, `host_end_to_last_kernel_tail=19.682 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `39982410`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4128, 1536]]}`
- `q_b_proj` -> 0.229 ms
  纯GPU kernel时间: `0.235 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.229 ms`, `host_to_first_kernel_gap=19.652007`, `host_end_to_last_kernel_tail=19.659 ms`, `gpu_makespan=0.236 ms`, `gpu_kernel_sum=0.235 ms`
  开始时间(ns): `40076130`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4128, 1536]]}`
- `kv_a_layernorm` -> 0.041 ms
  纯GPU kernel时间: `0.008 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.041 ms`, `host_to_first_kernel_gap=19.599513`, `host_end_to_last_kernel_tail=19.567 ms`, `gpu_makespan=0.008 ms`, `gpu_kernel_sum=0.008 ms`
  开始时间(ns): `40364783`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4128, 512]]}`
- `rotary_emb` -> 0.075 ms
  纯GPU kernel时间: `0.053 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.075 ms`, `host_to_first_kernel_gap=19.529047`, `host_end_to_last_kernel_tail=19.508 ms`, `gpu_makespan=0.053 ms`, `gpu_kernel_sum=0.053 ms`
  开始时间(ns): `40444593`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4128], [4128, 128, 64], [4128, 1, 64]]}`
- `kv_b_proj` -> 0.225 ms
  纯GPU kernel时间: `0.746 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.225 ms`, `host_to_first_kernel_gap=19.343132`, `host_end_to_last_kernel_tail=19.866 ms`, `gpu_makespan=0.748 ms`, `gpu_kernel_sum=0.746 ms`
  开始时间(ns): `40724396`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[20633, 512]]}`
- `attn_mha` -> 0.144 ms
  纯GPU kernel时间: `0.801 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.144 ms`, `host_to_first_kernel_gap=20.358753`, `host_end_to_last_kernel_tail=21.017 ms`, `gpu_makespan=0.802 ms`, `gpu_kernel_sum=0.801 ms`
  开始时间(ns): `41055719`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4128, 128, 192], [20633, 128, 192], [20633, 128, 128]]}`
- `o_proj` -> 0.229 ms
  纯GPU kernel时间: `0.712 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.229 ms`, `host_to_first_kernel_gap=20.98152`, `host_end_to_last_kernel_tail=21.466 ms`, `gpu_makespan=0.713 ms`, `gpu_kernel_sum=0.712 ms`
  开始时间(ns): `41236952`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4128, 16384]]}`
