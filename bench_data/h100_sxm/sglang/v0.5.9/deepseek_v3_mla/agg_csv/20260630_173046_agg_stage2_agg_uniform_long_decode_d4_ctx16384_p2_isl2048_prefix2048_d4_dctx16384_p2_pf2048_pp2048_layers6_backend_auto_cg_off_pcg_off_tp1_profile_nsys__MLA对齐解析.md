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
- 整块 MLA-module 时长: `3.286 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=3.286 ms`, `module_to_last_kernel=11.470 ms`, `host_to_first_kernel_gap=0.230704`, `host_end_to_last_kernel_tail=8.183 ms`, `gpu_makespan=11.239 ms`, `gpu_kernel_sum=7.162 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4100`, `total_tokens=73762`, `chunked_req_prefix_len=69662`, `current_chunked_req_prefix_len=69662`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[4100, 128, 192], [73762, 128, 192], [73762, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.311 ms
  纯GPU kernel时间: `0.142 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.311 ms`, `host_to_first_kernel_gap=0.102725`, `host_end_to_last_kernel_tail=0.080 ms`, `gpu_makespan=0.288 ms`, `gpu_kernel_sum=0.142 ms`
  开始时间(ns): `24096971`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4100, 7168]]}`
- `q_a_layernorm` -> 0.058 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.058 ms`, `host_to_first_kernel_gap=0.055179`, `host_end_to_last_kernel_tail=0.007 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `24451012`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4100, 1536]]}`
- `q_b_proj` -> 0.178 ms
  纯GPU kernel时间: `0.240 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.178 ms`, `host_to_first_kernel_gap=0.064355`, `host_end_to_last_kernel_tail=0.214 ms`, `gpu_makespan=0.328 ms`, `gpu_kernel_sum=0.240 ms`
  开始时间(ns): `24534892`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4100, 1536]]}`
- `kv_a_layernorm` -> 0.030 ms
  纯GPU kernel时间: `0.008 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.030 ms`, `host_to_first_kernel_gap=0.171394`, `host_end_to_last_kernel_tail=0.149 ms`, `gpu_makespan=0.008 ms`, `gpu_kernel_sum=0.008 ms`
  开始时间(ns): `24756141`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4100, 512]]}`
- `rotary_emb` -> 0.074 ms
  纯GPU kernel时间: `0.053 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.074 ms`, `host_to_first_kernel_gap=0.120601`, `host_end_to_last_kernel_tail=0.100 ms`, `gpu_makespan=0.053 ms`, `gpu_kernel_sum=0.053 ms`
  开始时间(ns): `24815990`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4100], [4100, 128, 64], [4100, 1, 64]]}`
- `kv_b_proj` -> 1.438 ms
  纯GPU kernel时间: `2.941 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=1.438 ms`, `host_to_first_kernel_gap=0.066747`, `host_end_to_last_kernel_tail=2.845 ms`, `gpu_makespan=4.217 ms`, `gpu_kernel_sum=2.941 ms`
  开始时间(ns): `25258868`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[73762, 512]]}`
- `attn_mha` -> 0.145 ms
  纯GPU kernel时间: `3.061 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.145 ms`, `host_to_first_kernel_gap=4.824366`, `host_end_to_last_kernel_tail=7.740 ms`, `gpu_makespan=3.061 ms`, `gpu_kernel_sum=3.061 ms`
  开始时间(ns): `26844891`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4100, 128, 192], [73762, 128, 192], [73762, 128, 128]]}`
- `o_proj` -> 0.223 ms
  纯GPU kernel时间: `0.706 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.223 ms`, `host_to_first_kernel_gap=7.710764`, `host_end_to_last_kernel_tail=8.195 ms`, `gpu_makespan=0.707 ms`, `gpu_kernel_sum=0.706 ms`
  开始时间(ns): `27021051`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4100, 16384]]}`

## Layer 1 / prefill / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `1.390 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.390 ms`, `module_to_last_kernel=19.469 ms`, `host_to_first_kernel_gap=10.074273`, `host_end_to_last_kernel_tail=18.079 ms`, `gpu_makespan=9.394 ms`, `gpu_kernel_sum=7.163 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4100`, `total_tokens=73762`, `chunked_req_prefix_len=69662`, `current_chunked_req_prefix_len=69662`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[4100, 128, 192], [73762, 128, 192], [73762, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.157 ms
  纯GPU kernel时间: `0.143 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.157 ms`, `host_to_first_kernel_gap=9.953529`, `host_end_to_last_kernel_tail=9.941 ms`, `gpu_makespan=0.144 ms`, `gpu_kernel_sum=0.143 ms`
  开始时间(ns): `27994603`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4100, 7168]]}`
- `q_a_layernorm` -> 0.039 ms
  纯GPU kernel时间: `0.010 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.039 ms`, `host_to_first_kernel_gap=9.903977`, `host_end_to_last_kernel_tail=9.874 ms`, `gpu_makespan=0.010 ms`, `gpu_kernel_sum=0.010 ms`
  开始时间(ns): `28188411`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4100, 1536]]}`
- `q_b_proj` -> 0.133 ms
  纯GPU kernel时间: `0.236 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.133 ms`, `host_to_first_kernel_gap=9.856797`, `host_end_to_last_kernel_tail=9.960 ms`, `gpu_makespan=0.237 ms`, `gpu_kernel_sum=0.236 ms`
  开始时间(ns): `28247143`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4100, 1536]]}`
- `kv_a_layernorm` -> 0.027 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.027 ms`, `host_to_first_kernel_gap=9.925419`, `host_end_to_last_kernel_tail=9.907 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `28415225`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4100, 512]]}`
- `rotary_emb` -> 0.076 ms
  纯GPU kernel时间: `0.054 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.076 ms`, `host_to_first_kernel_gap=9.872853`, `host_end_to_last_kernel_tail=9.851 ms`, `gpu_makespan=0.054 ms`, `gpu_kernel_sum=0.054 ms`
  开始时间(ns): `28478607`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4100], [4100, 128, 64], [4100, 1, 64]]}`
- `kv_b_proj` -> 0.153 ms
  纯GPU kernel时间: `2.945 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.153 ms`, `host_to_first_kernel_gap=9.764606`, `host_end_to_last_kernel_tail=12.559 ms`, `gpu_makespan=2.947 ms`, `gpu_kernel_sum=2.945 ms`
  开始时间(ns): `28731974`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[73762, 512]]}`
- `attn_mha` -> 0.119 ms
  纯GPU kernel时间: `3.061 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.119 ms`, `host_to_first_kernel_gap=14.620277`, `host_end_to_last_kernel_tail=17.563 ms`, `gpu_makespan=3.062 ms`, `gpu_kernel_sum=3.061 ms`
  开始时间(ns): `28950634`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4100, 128, 192], [73762, 128, 192], [73762, 128, 128]]}`
- `o_proj` -> 0.156 ms
  纯GPU kernel时间: `0.706 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.156 ms`, `host_to_first_kernel_gap=17.53659`, `host_end_to_last_kernel_tail=18.088 ms`, `gpu_makespan=0.708 ms`, `gpu_kernel_sum=0.706 ms`
  开始时间(ns): `29097967`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4100, 16384]]}`

## Layer 2 / prefill / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.330 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.330 ms`, `module_to_last_kernel=29.407 ms`, `host_to_first_kernel_gap=20.036289`, `host_end_to_last_kernel_tail=28.077 ms`, `gpu_makespan=9.371 ms`, `gpu_kernel_sum=7.141 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4100`, `total_tokens=73762`, `chunked_req_prefix_len=69662`, `current_chunked_req_prefix_len=69662`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[4100, 128, 192], [73762, 128, 192], [73762, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.146 ms
  纯GPU kernel时间: `0.144 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.146 ms`, `host_to_first_kernel_gap=19.92558`, `host_end_to_last_kernel_tail=19.925 ms`, `gpu_makespan=0.146 ms`, `gpu_kernel_sum=0.144 ms`
  开始时间(ns): `29921966`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4100, 7168]]}`
- `q_a_layernorm` -> 0.036 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.036 ms`, `host_to_first_kernel_gap=19.890629`, `host_end_to_last_kernel_tail=19.864 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `30102805`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4100, 1536]]}`
- `q_b_proj` -> 0.125 ms
  纯GPU kernel时间: `0.236 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.125 ms`, `host_to_first_kernel_gap=19.846725`, `host_end_to_last_kernel_tail=19.959 ms`, `gpu_makespan=0.238 ms`, `gpu_kernel_sum=0.236 ms`
  开始时间(ns): `30158293`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4100, 1536]]}`
- `kv_a_layernorm` -> 0.027 ms
  纯GPU kernel时间: `0.008 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.027 ms`, `host_to_first_kernel_gap=19.924512`, `host_end_to_last_kernel_tail=19.905 ms`, `gpu_makespan=0.008 ms`, `gpu_kernel_sum=0.008 ms`
  开始时间(ns): `30318010`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4100, 512]]}`
- `rotary_emb` -> 0.068 ms
  纯GPU kernel时间: `0.053 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.068 ms`, `host_to_first_kernel_gap=19.880831`, `host_end_to_last_kernel_tail=19.866 ms`, `gpu_makespan=0.053 ms`, `gpu_kernel_sum=0.053 ms`
  开始时间(ns): `30370459`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4100], [4100, 128, 64], [4100, 1, 64]]}`
- `kv_b_proj` -> 0.144 ms
  纯GPU kernel时间: `2.925 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.144 ms`, `host_to_first_kernel_gap=19.781274`, `host_end_to_last_kernel_tail=22.563 ms`, `gpu_makespan=2.926 ms`, `gpu_kernel_sum=2.925 ms`
  开始时间(ns): `30615295`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[73762, 512]]}`
- `attn_mha` -> 0.119 ms
  纯GPU kernel时间: `3.061 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.119 ms`, `host_to_first_kernel_gap=24.622881`, `host_end_to_last_kernel_tail=27.565 ms`, `gpu_makespan=3.061 ms`, `gpu_kernel_sum=3.061 ms`
  开始时间(ns): `30825972`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4100, 128, 192], [73762, 128, 192], [73762, 128, 128]]}`
- `o_proj` -> 0.160 ms
  纯GPU kernel时间: `0.705 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.160 ms`, `host_to_first_kernel_gap=27.538931`, `host_end_to_last_kernel_tail=28.086 ms`, `gpu_makespan=0.707 ms`, `gpu_kernel_sum=0.705 ms`
  开始时间(ns): `30972160`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4100, 16384]]}`

## Layer 3 / prefill / instance 1

- 执行序号: `4`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `1.341 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.341 ms`, `module_to_last_kernel=39.504 ms`, `host_to_first_kernel_gap=30.018524`, `host_end_to_last_kernel_tail=38.163 ms`, `gpu_makespan=9.485 ms`, `gpu_kernel_sum=7.258 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4100`, `total_tokens=73762`, `chunked_req_prefix_len=69662`, `current_chunked_req_prefix_len=69662`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[4100, 128, 192], [73762, 128, 192], [73762, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.143 ms
  纯GPU kernel时间: `0.148 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.143 ms`, `host_to_first_kernel_gap=29.907547`, `host_end_to_last_kernel_tail=29.915 ms`, `gpu_makespan=0.150 ms`, `gpu_kernel_sum=0.148 ms`
  开始时间(ns): `31812501`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4100, 7168]]}`
- `q_a_layernorm` -> 0.036 ms
  纯GPU kernel时间: `0.010 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.036 ms`, `host_to_first_kernel_gap=29.880048`, `host_end_to_last_kernel_tail=29.854 ms`, `gpu_makespan=0.010 ms`, `gpu_kernel_sum=0.010 ms`
  开始时间(ns): `31990752`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4100, 1536]]}`
- `q_b_proj` -> 0.132 ms
  纯GPU kernel时间: `0.233 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.132 ms`, `host_to_first_kernel_gap=29.836078`, `host_end_to_last_kernel_tail=29.939 ms`, `gpu_makespan=0.235 ms`, `gpu_kernel_sum=0.233 ms`
  开始时间(ns): `32045346`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4100, 1536]]}`
- `kv_a_layernorm` -> 0.026 ms
  纯GPU kernel时间: `0.008 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.026 ms`, `host_to_first_kernel_gap=29.903553`, `host_end_to_last_kernel_tail=29.885 ms`, `gpu_makespan=0.008 ms`, `gpu_kernel_sum=0.008 ms`
  开始时间(ns): `32212815`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4100, 512]]}`
- `rotary_emb` -> 0.066 ms
  纯GPU kernel时间: `0.052 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.066 ms`, `host_to_first_kernel_gap=29.858745`, `host_end_to_last_kernel_tail=29.845 ms`, `gpu_makespan=0.052 ms`, `gpu_kernel_sum=0.052 ms`
  开始时间(ns): `32267095`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4100], [4100, 128, 64], [4100, 1, 64]]}`
- `kv_b_proj` -> 0.160 ms
  纯GPU kernel时间: `3.037 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.160 ms`, `host_to_first_kernel_gap=29.759312`, `host_end_to_last_kernel_tail=32.637 ms`, `gpu_makespan=3.038 ms`, `gpu_kernel_sum=3.037 ms`
  开始时间(ns): `32510431`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[73762, 512]]}`
- `attn_mha` -> 0.112 ms
  纯GPU kernel时间: `3.063 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.112 ms`, `host_to_first_kernel_gap=34.695235`, `host_end_to_last_kernel_tail=37.646 ms`, `gpu_makespan=3.063 ms`, `gpu_kernel_sum=3.063 ms`
  开始时间(ns): `32738120`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4100, 128, 192], [73762, 128, 192], [73762, 128, 128]]}`
- `o_proj` -> 0.154 ms
  纯GPU kernel时间: `0.707 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.154 ms`, `host_to_first_kernel_gap=37.619786`, `host_end_to_last_kernel_tail=38.174 ms`, `gpu_makespan=0.709 ms`, `gpu_kernel_sum=0.707 ms`
  开始时间(ns): `32877182`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4100, 16384]]}`

## Layer 4 / prefill / instance 1

- 执行序号: `5`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `1.481 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.481 ms`, `module_to_last_kernel=51.589 ms`, `host_to_first_kernel_gap=42.318961`, `host_end_to_last_kernel_tail=50.107 ms`, `gpu_makespan=9.270 ms`, `gpu_kernel_sum=7.041 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4100`, `total_tokens=73762`, `chunked_req_prefix_len=69662`, `current_chunked_req_prefix_len=69662`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[4100, 128, 192], [73762, 128, 192], [73762, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.241 ms
  纯GPU kernel时间: `0.145 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.241 ms`, `host_to_first_kernel_gap=42.196833`, `host_end_to_last_kernel_tail=42.102 ms`, `gpu_makespan=0.146 ms`, `gpu_kernel_sum=0.145 ms`
  开始时间(ns): `34256450`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4100, 7168]]}`
- `q_a_layernorm` -> 0.040 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.040 ms`, `host_to_first_kernel_gap=42.065301`, `host_end_to_last_kernel_tail=42.035 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `34534830`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4100, 1536]]}`
- `q_b_proj` -> 0.151 ms
  纯GPU kernel时间: `0.234 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.151 ms`, `host_to_first_kernel_gap=42.015521`, `host_end_to_last_kernel_tail=42.100 ms`, `gpu_makespan=0.236 ms`, `gpu_kernel_sum=0.234 ms`
  开始时间(ns): `34596194`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4100, 1536]]}`
- `kv_a_layernorm` -> 0.029 ms
  纯GPU kernel时间: `0.008 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.029 ms`, `host_to_first_kernel_gap=42.059823`, `host_end_to_last_kernel_tail=42.039 ms`, `gpu_makespan=0.008 ms`, `gpu_kernel_sum=0.008 ms`
  开始时间(ns): `34787540`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4100, 512]]}`
- `rotary_emb` -> 0.070 ms
  纯GPU kernel时间: `0.053 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.070 ms`, `host_to_first_kernel_gap=42.013546`, `host_end_to_last_kernel_tail=41.996 ms`, `gpu_makespan=0.053 ms`, `gpu_kernel_sum=0.053 ms`
  开始时间(ns): `34842873`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4100], [4100, 128, 64], [4100, 1, 64]]}`
- `kv_b_proj` -> 0.151 ms
  纯GPU kernel时间: `2.820 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.151 ms`, `host_to_first_kernel_gap=41.924835`, `host_end_to_last_kernel_tail=44.595 ms`, `gpu_makespan=2.821 ms`, `gpu_kernel_sum=2.820 ms`
  开始时间(ns): `35074656`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[73762, 512]]}`
- `attn_mha` -> 0.117 ms
  纯GPU kernel时间: `3.066 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.117 ms`, `host_to_first_kernel_gap=46.655111`, `host_end_to_last_kernel_tail=49.604 ms`, `gpu_makespan=3.066 ms`, `gpu_kernel_sum=3.066 ms`
  开始时间(ns): `35291832`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4100, 128, 192], [73762, 128, 192], [73762, 128, 128]]}`
- `o_proj` -> 0.170 ms
  纯GPU kernel时间: `0.706 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.170 ms`, `host_to_first_kernel_gap=49.578867`, `host_end_to_last_kernel_tail=50.117 ms`, `gpu_makespan=0.708 ms`, `gpu_kernel_sum=0.706 ms`
  开始时间(ns): `35435561`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4100, 16384]]}`

## Layer 5 / prefill / instance 1

- 执行序号: `6`
- Collector对齐边界: `{'Module': 'model.model.layers.5.self_attn'}`
- 整块 MLA-module 时长: `1.435 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.435 ms`, `module_to_last_kernel=63.678 ms`, `host_to_first_kernel_gap=54.299079`, `host_end_to_last_kernel_tail=62.243 ms`, `gpu_makespan=9.379 ms`, `gpu_kernel_sum=7.151 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4100`, `total_tokens=73762`, `chunked_req_prefix_len=69662`, `current_chunked_req_prefix_len=69662`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[4100, 128, 192], [73762, 128, 192], [73762, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.214 ms
  纯GPU kernel时间: `0.143 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.214 ms`, `host_to_first_kernel_gap=54.182506`, `host_end_to_last_kernel_tail=54.113 ms`, `gpu_makespan=0.144 ms`, `gpu_kernel_sum=0.143 ms`
  开始时间(ns): `36758029`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4100, 7168]]}`
- `q_a_layernorm` -> 0.039 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.039 ms`, `host_to_first_kernel_gap=54.076848`, `host_end_to_last_kernel_tail=54.047 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `37008647`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4100, 1536]]}`
- `q_b_proj` -> 0.147 ms
  纯GPU kernel时间: `0.234 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.147 ms`, `host_to_first_kernel_gap=54.027237`, `host_end_to_last_kernel_tail=54.115 ms`, `gpu_makespan=0.235 ms`, `gpu_kernel_sum=0.234 ms`
  开始时间(ns): `37069522`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4100, 1536]]}`
- `kv_a_layernorm` -> 0.027 ms
  纯GPU kernel时间: `0.008 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.027 ms`, `host_to_first_kernel_gap=54.077862`, `host_end_to_last_kernel_tail=54.059 ms`, `gpu_makespan=0.008 ms`, `gpu_kernel_sum=0.008 ms`
  开始时间(ns): `37253681`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4100, 512]]}`
- `rotary_emb` -> 0.065 ms
  纯GPU kernel时间: `0.053 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.065 ms`, `host_to_first_kernel_gap=54.034862`, `host_end_to_last_kernel_tail=54.022 ms`, `gpu_makespan=0.053 ms`, `gpu_kernel_sum=0.053 ms`
  开始时间(ns): `37306761`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4100], [4100, 128, 64], [4100, 1, 64]]}`
- `kv_b_proj` -> 0.159 ms
  纯GPU kernel时间: `2.936 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.159 ms`, `host_to_first_kernel_gap=53.94337`, `host_end_to_last_kernel_tail=56.722 ms`, `gpu_makespan=2.937 ms`, `gpu_kernel_sum=2.936 ms`
  开始时间(ns): `37542317`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[73762, 512]]}`
- `attn_mha` -> 0.117 ms
  纯GPU kernel时间: `3.062 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.117 ms`, `host_to_first_kernel_gap=58.781139`, `host_end_to_last_kernel_tail=61.726 ms`, `gpu_makespan=3.062 ms`, `gpu_kernel_sum=3.062 ms`
  开始时间(ns): `37766527`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4100, 128, 192], [73762, 128, 192], [73762, 128, 128]]}`
- `o_proj` -> 0.157 ms
  纯GPU kernel时间: `0.706 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.157 ms`, `host_to_first_kernel_gap=61.700898`, `host_end_to_last_kernel_tail=62.252 ms`, `gpu_makespan=0.708 ms`, `gpu_kernel_sum=0.706 ms`
  开始时间(ns): `37910574`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4100, 16384]]}`
