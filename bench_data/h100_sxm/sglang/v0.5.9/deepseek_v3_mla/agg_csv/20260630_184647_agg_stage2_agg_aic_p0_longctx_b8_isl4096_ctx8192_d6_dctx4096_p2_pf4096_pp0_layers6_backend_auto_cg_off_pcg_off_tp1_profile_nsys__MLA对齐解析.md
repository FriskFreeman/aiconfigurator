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
- 整块 MLA-module 时长: `2.457 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.457 ms`, `module_to_last_kernel=7.701 ms`, `host_to_first_kernel_gap=0.264`, `host_end_to_last_kernel_tail=5.245 ms`, `gpu_makespan=7.437 ms`, `gpu_kernel_sum=5.783 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=32796`, `chunked_req_prefix_len=24604`, `current_chunked_req_prefix_len=24604`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[8192, 128, 192], [32796, 128, 192], [32796, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.340 ms
  纯GPU kernel时间: `0.275 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.340 ms`, `host_to_first_kernel_gap=0.112827`, `host_end_to_last_kernel_tail=0.159 ms`, `gpu_makespan=0.387 ms`, `gpu_kernel_sum=0.275 ms`
  开始时间(ns): `21377809`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.048 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.048 ms`, `host_to_first_kernel_gap=0.103883`, `host_end_to_last_kernel_tail=0.073 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `21773888`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.246 ms
  纯GPU kernel时间: `0.451 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.246 ms`, `host_to_first_kernel_gap=0.086821`, `host_end_to_last_kernel_tail=0.406 ms`, `gpu_makespan=0.566 ms`, `gpu_kernel_sum=0.451 ms`
  开始时间(ns): `21857190`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.041 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.041 ms`, `host_to_first_kernel_gap=0.342459`, `host_end_to_last_kernel_tail=0.314 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `22166863`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.097 ms
  纯GPU kernel时间: `0.106 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.097 ms`, `host_to_first_kernel_gap=0.272569`, `host_end_to_last_kernel_tail=0.282 ms`, `gpu_makespan=0.106 ms`, `gpu_kernel_sum=0.106 ms`
  开始时间(ns): `22250673`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.275 ms
  纯GPU kernel时间: `1.085 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.275 ms`, `host_to_first_kernel_gap=0.098413`, `host_end_to_last_kernel_tail=1.033 ms`, `gpu_makespan=1.209 ms`, `gpu_kernel_sum=1.085 ms`
  开始时间(ns): `22831709`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[32796, 512]]}`
- `attn_mha` -> 0.160 ms
  纯GPU kernel时间: `2.579 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.160 ms`, `host_to_first_kernel_gap=1.868267`, `host_end_to_last_kernel_tail=4.287 ms`, `gpu_makespan=2.579 ms`, `gpu_kernel_sum=2.579 ms`
  开始时间(ns): `23220636`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [32796, 128, 192], [32796, 128, 128]]}`
- `o_proj` -> 0.245 ms
  纯GPU kernel时间: `1.258 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.245 ms`, `host_to_first_kernel_gap=4.244306`, `host_end_to_last_kernel_tail=5.259 ms`, `gpu_makespan=1.259 ms`, `gpu_kernel_sum=1.258 ms`
  开始时间(ns): `23424337`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 1 / prefill / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `1.874 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.874 ms`, `module_to_last_kernel=15.946 ms`, `host_to_first_kernel_gap=9.023419`, `host_end_to_last_kernel_tail=14.072 ms`, `gpu_makespan=6.923 ms`, `gpu_kernel_sum=5.895 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=32796`, `chunked_req_prefix_len=24604`, `current_chunked_req_prefix_len=24604`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[8192, 128, 192], [32796, 128, 192], [32796, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.230 ms
  纯GPU kernel时间: `0.267 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.230 ms`, `host_to_first_kernel_gap=8.892382`, `host_end_to_last_kernel_tail=8.931 ms`, `gpu_makespan=0.269 ms`, `gpu_kernel_sum=0.267 ms`
  开始时间(ns): `24656862`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.048 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.048 ms`, `host_to_first_kernel_gap=8.876174`, `host_end_to_last_kernel_tail=8.846 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `24941549`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.216 ms
  纯GPU kernel时间: `0.439 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.216 ms`, `host_to_first_kernel_gap=8.817681`, `host_end_to_last_kernel_tail=9.042 ms`, `gpu_makespan=0.440 ms`, `gpu_kernel_sum=0.439 ms`
  开始时间(ns): `25020170`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.041 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.041 ms`, `host_to_first_kernel_gap=8.98026`, `host_end_to_last_kernel_tail=8.953 ms`, `gpu_makespan=0.014 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `25297847`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.075 ms
  纯GPU kernel时间: `0.106 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.075 ms`, `host_to_first_kernel_gap=8.913903`, `host_end_to_last_kernel_tail=8.945 ms`, `gpu_makespan=0.106 ms`, `gpu_kernel_sum=0.106 ms`
  开始时间(ns): `25380108`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.226 ms
  纯GPU kernel时间: `1.218 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.226 ms`, `host_to_first_kernel_gap=8.79917`, `host_end_to_last_kernel_tail=9.794 ms`, `gpu_makespan=1.220 ms`, `gpu_kernel_sum=1.218 ms`
  开始时间(ns): `25663992`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[32796, 512]]}`
- `attn_mha` -> 0.141 ms
  纯GPU kernel时间: `2.583 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.141 ms`, `host_to_first_kernel_gap=10.643665`, `host_end_to_last_kernel_tail=13.086 ms`, `gpu_makespan=2.583 ms`, `gpu_kernel_sum=2.583 ms`
  开始时间(ns): `25990566`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [32796, 128, 192], [32796, 128, 128]]}`
- `o_proj` -> 0.216 ms
  纯GPU kernel时间: `1.251 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.216 ms`, `host_to_first_kernel_gap=13.048162`, `host_end_to_last_kernel_tail=14.085 ms`, `gpu_makespan=1.253 ms`, `gpu_kernel_sum=1.251 ms`
  开始时间(ns): `26171218`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 2 / prefill / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.865 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.865 ms`, `module_to_last_kernel=24.810 ms`, `host_to_first_kernel_gap=17.901008`, `host_end_to_last_kernel_tail=22.945 ms`, `gpu_makespan=6.909 ms`, `gpu_kernel_sum=5.885 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=32796`, `chunked_req_prefix_len=24604`, `current_chunked_req_prefix_len=24604`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[8192, 128, 192], [32796, 128, 192], [32796, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.245 ms
  纯GPU kernel时间: `0.266 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.245 ms`, `host_to_first_kernel_gap=17.767365`, `host_end_to_last_kernel_tail=17.790 ms`, `gpu_makespan=0.267 ms`, `gpu_kernel_sum=0.266 ms`
  开始时间(ns): `27313415`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.045 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.045 ms`, `host_to_first_kernel_gap=17.739582`, `host_end_to_last_kernel_tail=17.712 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `27608910`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.217 ms
  纯GPU kernel时间: `0.442 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.217 ms`, `host_to_first_kernel_gap=17.684937`, `host_end_to_last_kernel_tail=17.912 ms`, `gpu_makespan=0.444 ms`, `gpu_kernel_sum=0.442 ms`
  开始时间(ns): `27683587`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.040 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.040 ms`, `host_to_first_kernel_gap=17.850883`, `host_end_to_last_kernel_tail=17.824 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `27961096`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.074 ms
  纯GPU kernel时间: `0.106 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.074 ms`, `host_to_first_kernel_gap=17.786474`, `host_end_to_last_kernel_tail=17.819 ms`, `gpu_makespan=0.106 ms`, `gpu_kernel_sum=0.106 ms`
  开始时间(ns): `28040033`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.218 ms
  纯GPU kernel时间: `1.212 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.218 ms`, `host_to_first_kernel_gap=17.67637`, `host_end_to_last_kernel_tail=18.672 ms`, `gpu_makespan=1.214 ms`, `gpu_kernel_sum=1.212 ms`
  开始时间(ns): `28319513`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[32796, 512]]}`
- `attn_mha` -> 0.146 ms
  纯GPU kernel时间: `2.579 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.146 ms`, `host_to_first_kernel_gap=19.526016`, `host_end_to_last_kernel_tail=21.959 ms`, `gpu_makespan=2.579 ms`, `gpu_kernel_sum=2.579 ms`
  开始时间(ns): `28632552`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [32796, 128, 192], [32796, 128, 128]]}`
- `o_proj` -> 0.216 ms
  纯GPU kernel时间: `1.249 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.216 ms`, `host_to_first_kernel_gap=21.923236`, `host_end_to_last_kernel_tail=22.958 ms`, `gpu_makespan=1.251 ms`, `gpu_kernel_sum=1.249 ms`
  开始时间(ns): `28815776`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 3 / prefill / instance 1

- 执行序号: `4`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `1.865 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.865 ms`, `module_to_last_kernel=33.671 ms`, `host_to_first_kernel_gap=26.833945`, `host_end_to_last_kernel_tail=31.806 ms`, `gpu_makespan=6.837 ms`, `gpu_kernel_sum=5.814 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=32796`, `chunked_req_prefix_len=24604`, `current_chunked_req_prefix_len=24604`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[8192, 128, 192], [32796, 128, 192], [32796, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.238 ms
  纯GPU kernel时间: `0.272 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.238 ms`, `host_to_first_kernel_gap=26.7004`, `host_end_to_last_kernel_tail=26.736 ms`, `gpu_makespan=0.274 ms`, `gpu_kernel_sum=0.272 ms`
  开始时间(ns): `29943404`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.045 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.045 ms`, `host_to_first_kernel_gap=26.684209`, `host_end_to_last_kernel_tail=26.657 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `30233707`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.213 ms
  纯GPU kernel时间: `0.443 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.213 ms`, `host_to_first_kernel_gap=26.628699`, `host_end_to_last_kernel_tail=26.860 ms`, `gpu_makespan=0.445 ms`, `gpu_kernel_sum=0.443 ms`
  开始时间(ns): `30308289`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.039 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.039 ms`, `host_to_first_kernel_gap=26.799253`, `host_end_to_last_kernel_tail=26.773 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `30582502`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.081 ms
  纯GPU kernel时间: `0.107 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.081 ms`, `host_to_first_kernel_gap=26.735667`, `host_end_to_last_kernel_tail=26.761 ms`, `gpu_makespan=0.107 ms`, `gpu_kernel_sum=0.107 ms`
  开始时间(ns): `30660040`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.216 ms
  纯GPU kernel时间: `1.105 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.216 ms`, `host_to_first_kernel_gap=26.615306`, `host_end_to_last_kernel_tail=27.505 ms`, `gpu_makespan=1.106 ms`, `gpu_kernel_sum=1.105 ms`
  开始时间(ns): `30951121`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[32796, 512]]}`
- `attn_mha` -> 0.137 ms
  纯GPU kernel时间: `2.582 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.137 ms`, `host_to_first_kernel_gap=28.357795`, `host_end_to_last_kernel_tail=30.803 ms`, `gpu_makespan=2.582 ms`, `gpu_kernel_sum=2.582 ms`
  开始时间(ns): `31264181`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [32796, 128, 192], [32796, 128, 128]]}`
- `o_proj` -> 0.221 ms
  纯GPU kernel时间: `1.274 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.221 ms`, `host_to_first_kernel_gap=30.766089`, `host_end_to_last_kernel_tail=31.821 ms`, `gpu_makespan=1.276 ms`, `gpu_kernel_sum=1.274 ms`
  开始时间(ns): `31439308`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 4 / prefill / instance 1

- 执行序号: `5`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `1.937 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.937 ms`, `module_to_last_kernel=47.833 ms`, `host_to_first_kernel_gap=40.832803`, `host_end_to_last_kernel_tail=45.896 ms`, `gpu_makespan=7.000 ms`, `gpu_kernel_sum=5.977 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=32796`, `chunked_req_prefix_len=24604`, `current_chunked_req_prefix_len=24604`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[8192, 128, 192], [32796, 128, 192], [32796, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.260 ms
  纯GPU kernel时间: `0.274 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.260 ms`, `host_to_first_kernel_gap=40.694095`, `host_end_to_last_kernel_tail=40.709 ms`, `gpu_makespan=0.275 ms`, `gpu_kernel_sum=0.274 ms`
  开始时间(ns): `33153430`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.046 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.046 ms`, `host_to_first_kernel_gap=40.658115`, `host_end_to_last_kernel_tail=40.630 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `33464514`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.217 ms
  纯GPU kernel时间: `0.450 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.217 ms`, `host_to_first_kernel_gap=40.600955`, `host_end_to_last_kernel_tail=40.836 ms`, `gpu_makespan=0.452 ms`, `gpu_kernel_sum=0.450 ms`
  开始时间(ns): `33541610`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.039 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.039 ms`, `host_to_first_kernel_gap=40.775754`, `host_end_to_last_kernel_tail=40.750 ms`, `gpu_makespan=0.014 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `33818906`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.079 ms
  纯GPU kernel时间: `0.107 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.079 ms`, `host_to_first_kernel_gap=40.712703`, `host_end_to_last_kernel_tail=40.741 ms`, `gpu_makespan=0.107 ms`, `gpu_kernel_sum=0.107 ms`
  开始时间(ns): `33896645`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.232 ms
  纯GPU kernel时间: `1.244 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.232 ms`, `host_to_first_kernel_gap=40.595073`, `host_end_to_last_kernel_tail=41.608 ms`, `gpu_makespan=1.245 ms`, `gpu_kernel_sum=1.244 ms`
  开始时间(ns): `34182531`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[32796, 512]]}`
- `attn_mha` -> 0.141 ms
  纯GPU kernel时间: `2.582 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.141 ms`, `host_to_first_kernel_gap=42.457079`, `host_end_to_last_kernel_tail=44.898 ms`, `gpu_makespan=2.582 ms`, `gpu_kernel_sum=2.582 ms`
  开始时间(ns): `34515946`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [32796, 128, 192], [32796, 128, 128]]}`
- `o_proj` -> 0.240 ms
  纯GPU kernel时间: `1.290 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.240 ms`, `host_to_first_kernel_gap=44.858214`, `host_end_to_last_kernel_tail=45.910 ms`, `gpu_makespan=1.291 ms`, `gpu_kernel_sum=1.290 ms`
  开始时间(ns): `34698327`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 5 / prefill / instance 1

- 执行序号: `6`
- Collector对齐边界: `{'Module': 'model.model.layers.5.self_attn'}`
- 整块 MLA-module 时长: `1.940 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.940 ms`, `module_to_last_kernel=61.978 ms`, `host_to_first_kernel_gap=54.975374`, `host_end_to_last_kernel_tail=60.038 ms`, `gpu_makespan=7.003 ms`, `gpu_kernel_sum=5.980 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=32796`, `chunked_req_prefix_len=24604`, `current_chunked_req_prefix_len=24604`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[8192, 128, 192], [32796, 128, 192], [32796, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.277 ms
  纯GPU kernel时间: `0.274 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.277 ms`, `host_to_first_kernel_gap=54.834284`, `host_end_to_last_kernel_tail=54.832 ms`, `gpu_makespan=0.275 ms`, `gpu_kernel_sum=0.274 ms`
  开始时间(ns): `36361346`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.057 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.057 ms`, `host_to_first_kernel_gap=54.779569`, `host_end_to_last_kernel_tail=54.740 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `36690908`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.224 ms
  纯GPU kernel时间: `0.451 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.224 ms`, `host_to_first_kernel_gap=54.710111`, `host_end_to_last_kernel_tail=54.938 ms`, `gpu_makespan=0.452 ms`, `gpu_kernel_sum=0.451 ms`
  开始时间(ns): `36780430`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.044 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.044 ms`, `host_to_first_kernel_gap=54.87911`, `host_end_to_last_kernel_tail=54.848 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `37063271`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.077 ms
  纯GPU kernel时间: `0.106 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.077 ms`, `host_to_first_kernel_gap=54.811454`, `host_end_to_last_kernel_tail=54.841 ms`, `gpu_makespan=0.106 ms`, `gpu_kernel_sum=0.106 ms`
  开始时间(ns): `37145999`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.217 ms
  纯GPU kernel时间: `1.250 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.217 ms`, `host_to_first_kernel_gap=54.700201`, `host_end_to_last_kernel_tail=55.735 ms`, `gpu_makespan=1.251 ms`, `gpu_kernel_sum=1.250 ms`
  开始时间(ns): `37426499`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[32796, 512]]}`
- `attn_mha` -> 0.143 ms
  纯GPU kernel时间: `2.586 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.143 ms`, `host_to_first_kernel_gap=56.575569`, `host_end_to_last_kernel_tail=59.019 ms`, `gpu_makespan=2.587 ms`, `gpu_kernel_sum=2.586 ms`
  开始时间(ns): `37749113`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [32796, 128, 192], [32796, 128, 128]]}`
- `o_proj` -> 0.217 ms
  纯GPU kernel时间: `1.282 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.217 ms`, `host_to_first_kernel_gap=58.983405`, `host_end_to_last_kernel_tail=60.051 ms`, `gpu_makespan=1.285 ms`, `gpu_kernel_sum=1.282 ms`
  开始时间(ns): `37930137`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`
