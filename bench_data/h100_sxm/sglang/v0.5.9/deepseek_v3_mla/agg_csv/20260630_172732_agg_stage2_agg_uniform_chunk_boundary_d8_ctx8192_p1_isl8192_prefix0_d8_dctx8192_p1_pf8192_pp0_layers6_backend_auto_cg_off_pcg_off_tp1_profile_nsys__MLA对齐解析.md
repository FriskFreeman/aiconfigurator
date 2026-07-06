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
- 整块 MLA-module 时长: `2.101 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.101 ms`, `module_to_last_kernel=13.216 ms`, `host_to_first_kernel_gap=0.294625`, `host_end_to_last_kernel_tail=11.115 ms`, `gpu_makespan=12.922 ms`, `gpu_kernel_sum=10.370 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=73781`, `chunked_req_prefix_len=65589`, `current_chunked_req_prefix_len=65589`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[8192, 128, 192], [73781, 128, 192], [73781, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.371 ms
  纯GPU kernel时间: `0.274 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.371 ms`, `host_to_first_kernel_gap=0.12635`, `host_end_to_last_kernel_tail=0.176 ms`, `gpu_makespan=0.421 ms`, `gpu_kernel_sum=0.274 ms`
  开始时间(ns): `22136079`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.039 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.039 ms`, `host_to_first_kernel_gap=0.12281`, `host_end_to_last_kernel_tail=0.101 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `22560899`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.193 ms
  纯GPU kernel时间: `0.448 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.193 ms`, `host_to_first_kernel_gap=0.079536`, `host_end_to_last_kernel_tail=0.413 ms`, `gpu_makespan=0.527 ms`, `gpu_kernel_sum=0.448 ms`
  开始时间(ns): `22627213`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.030 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.030 ms`, `host_to_first_kernel_gap=0.367055`, `host_end_to_last_kernel_tail=0.350 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `22866222`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.091 ms
  纯GPU kernel时间: `0.107 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.091 ms`, `host_to_first_kernel_gap=0.319254`, `host_end_to_last_kernel_tail=0.335 ms`, `gpu_makespan=0.107 ms`, `gpu_kernel_sum=0.107 ms`
  开始时间(ns): `22928199`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.190 ms
  纯GPU kernel时间: `3.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.190 ms`, `host_to_first_kernel_gap=0.06684`, `host_end_to_last_kernel_tail=2.954 ms`, `gpu_makespan=3.077 ms`, `gpu_kernel_sum=3.019 ms`
  开始时间(ns): `23420102`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[73781, 512]]}`
- `attn_mha` -> 0.149 ms
  纯GPU kernel时间: `5.236 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.149 ms`, `host_to_first_kernel_gap=4.99188`, `host_end_to_last_kernel_tail=10.078 ms`, `gpu_makespan=5.236 ms`, `gpu_kernel_sum=5.236 ms`
  开始时间(ns): `23697721`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [73781, 128, 192], [73781, 128, 128]]}`
- `o_proj` -> 0.180 ms
  纯GPU kernel时间: `1.257 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.180 ms`, `host_to_first_kernel_gap=10.048387`, `host_end_to_last_kernel_tail=11.126 ms`, `gpu_makespan=1.258 ms`, `gpu_kernel_sum=1.257 ms`
  开始时间(ns): `23878210`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 1 / prefill / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `1.347 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.347 ms`, `module_to_last_kernel=27.569 ms`, `host_to_first_kernel_gap=15.090337`, `host_end_to_last_kernel_tail=26.222 ms`, `gpu_makespan=12.479 ms`, `gpu_kernel_sum=10.240 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=73781`, `chunked_req_prefix_len=65589`, `current_chunked_req_prefix_len=65589`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[8192, 128, 192], [73781, 128, 192], [73781, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.163 ms
  纯GPU kernel时间: `0.267 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.163 ms`, `host_to_first_kernel_gap=14.984392`, `host_end_to_last_kernel_tail=15.090 ms`, `gpu_makespan=0.269 ms`, `gpu_kernel_sum=0.267 ms`
  开始时间(ns): `24809601`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.042 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.042 ms`, `host_to_first_kernel_gap=15.05444`, `host_end_to_last_kernel_tail=15.030 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `25008289`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.142 ms
  纯GPU kernel时间: `0.443 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.142 ms`, `host_to_first_kernel_gap=15.010723`, `host_end_to_last_kernel_tail=15.313 ms`, `gpu_makespan=0.445 ms`, `gpu_kernel_sum=0.443 ms`
  开始时间(ns): `25072198`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.029 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.029 ms`, `host_to_first_kernel_gap=15.264771`, `host_end_to_last_kernel_tail=15.250 ms`, `gpu_makespan=0.014 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `25262502`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.057 ms
  纯GPU kernel时间: `0.107 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.057 ms`, `host_to_first_kernel_gap=15.223712`, `host_end_to_last_kernel_tail=15.273 ms`, `gpu_makespan=0.107 ms`, `gpu_kernel_sum=0.107 ms`
  开始时间(ns): `25319433`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.152 ms
  纯GPU kernel时间: `2.893 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.152 ms`, `host_to_first_kernel_gap=15.218722`, `host_end_to_last_kernel_tail=17.962 ms`, `gpu_makespan=2.895 ms`, `gpu_kernel_sum=2.893 ms`
  开始时间(ns): `25531400`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[73781, 512]]}`
- `attn_mha` -> 0.112 ms
  纯GPU kernel时间: `5.241 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.112 ms`, `host_to_first_kernel_gap=20.019807`, `host_end_to_last_kernel_tail=25.148 ms`, `gpu_makespan=5.241 ms`, `gpu_kernel_sum=5.241 ms`
  开始时间(ns): `25750766`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [73781, 128, 192], [73781, 128, 128]]}`
- `o_proj` -> 0.149 ms
  纯GPU kernel时间: `1.257 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.149 ms`, `host_to_first_kernel_gap=25.122092`, `host_end_to_last_kernel_tail=26.232 ms`, `gpu_makespan=1.259 ms`, `gpu_kernel_sum=1.257 ms`
  开始时间(ns): `25891461`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 2 / prefill / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.388 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.388 ms`, `module_to_last_kernel=42.469 ms`, `host_to_first_kernel_gap=30.071124`, `host_end_to_last_kernel_tail=41.081 ms`, `gpu_makespan=12.397 ms`, `gpu_kernel_sum=10.161 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=73781`, `chunked_req_prefix_len=65589`, `current_chunked_req_prefix_len=65589`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[8192, 128, 192], [73781, 128, 192], [73781, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.177 ms
  纯GPU kernel时间: `0.265 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.177 ms`, `host_to_first_kernel_gap=29.961113`, `host_end_to_last_kernel_tail=30.050 ms`, `gpu_makespan=0.266 ms`, `gpu_kernel_sum=0.265 ms`
  开始时间(ns): `26894140`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.046 ms
  纯GPU kernel时间: `0.017 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.046 ms`, `host_to_first_kernel_gap=30.012807`, `host_end_to_last_kernel_tail=29.985 ms`, `gpu_makespan=0.017 ms`, `gpu_kernel_sum=0.017 ms`
  开始时间(ns): `27109294`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.156 ms
  纯GPU kernel时间: `0.434 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.156 ms`, `host_to_first_kernel_gap=29.96407`, `host_end_to_last_kernel_tail=30.244 ms`, `gpu_makespan=0.436 ms`, `gpu_kernel_sum=0.434 ms`
  开始时间(ns): `27177391`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.029 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.029 ms`, `host_to_first_kernel_gap=30.200867`, `host_end_to_last_kernel_tail=30.186 ms`, `gpu_makespan=0.014 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `27376562`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.055 ms
  纯GPU kernel时间: `0.106 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.055 ms`, `host_to_first_kernel_gap=30.15824`, `host_end_to_last_kernel_tail=30.210 ms`, `gpu_makespan=0.106 ms`, `gpu_kernel_sum=0.106 ms`
  开始时间(ns): `27434229`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.147 ms
  纯GPU kernel时间: `2.847 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.147 ms`, `host_to_first_kernel_gap=30.156524`, `host_end_to_last_kernel_tail=32.857 ms`, `gpu_makespan=2.848 ms`, `gpu_kernel_sum=2.847 ms`
  开始时间(ns): `27643625`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[73781, 512]]}`
- `attn_mha` -> 0.115 ms
  纯GPU kernel时间: `5.238 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.115 ms`, `host_to_first_kernel_gap=34.913102`, `host_end_to_last_kernel_tail=40.036 ms`, `gpu_makespan=5.238 ms`, `gpu_kernel_sum=5.238 ms`
  开始时间(ns): `27858283`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [73781, 128, 192], [73781, 128, 128]]}`
- `o_proj` -> 0.158 ms
  纯GPU kernel时间: `1.240 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.158 ms`, `host_to_first_kernel_gap=40.007517`, `host_end_to_last_kernel_tail=41.091 ms`, `gpu_makespan=1.242 ms`, `gpu_kernel_sum=1.240 ms`
  开始时间(ns): `28002912`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 3 / prefill / instance 1

- 执行序号: `4`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `1.308 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.308 ms`, `module_to_last_kernel=57.610 ms`, `host_to_first_kernel_gap=45.10714`, `host_end_to_last_kernel_tail=56.302 ms`, `gpu_makespan=12.503 ms`, `gpu_kernel_sum=10.264 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=73781`, `chunked_req_prefix_len=65589`, `current_chunked_req_prefix_len=65589`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[8192, 128, 192], [73781, 128, 192], [73781, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.157 ms
  纯GPU kernel时间: `0.267 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.157 ms`, `host_to_first_kernel_gap=45.006809`, `host_end_to_last_kernel_tail=45.119 ms`, `gpu_makespan=0.269 ms`, `gpu_kernel_sum=0.267 ms`
  开始时间(ns): `28851144`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.032 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.032 ms`, `host_to_first_kernel_gap=45.08444`, `host_end_to_last_kernel_tail=45.071 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `29043305`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.136 ms
  纯GPU kernel时间: `0.437 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.136 ms`, `host_to_first_kernel_gap=45.050742`, `host_end_to_last_kernel_tail=45.354 ms`, `gpu_makespan=0.439 ms`, `gpu_kernel_sum=0.437 ms`
  开始时间(ns): `29095723`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.027 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.027 ms`, `host_to_first_kernel_gap=45.306745`, `host_end_to_last_kernel_tail=45.293 ms`, `gpu_makespan=0.014 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `29278344`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.052 ms
  纯GPU kernel时间: `0.106 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.052 ms`, `host_to_first_kernel_gap=45.267711`, `host_end_to_last_kernel_tail=45.321 ms`, `gpu_makespan=0.106 ms`, `gpu_kernel_sum=0.106 ms`
  开始时间(ns): `29332386`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.149 ms
  纯GPU kernel时间: `2.926 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.149 ms`, `host_to_first_kernel_gap=45.2764`, `host_end_to_last_kernel_tail=48.054 ms`, `gpu_makespan=2.927 ms`, `gpu_kernel_sum=2.926 ms`
  开始时间(ns): `29530705`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[73781, 512]]}`
- `attn_mha` -> 0.100 ms
  纯GPU kernel时间: `5.240 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.100 ms`, `host_to_first_kernel_gap=50.102354`, `host_end_to_last_kernel_tail=55.242 ms`, `gpu_makespan=5.240 ms`, `gpu_kernel_sum=5.240 ms`
  开始时间(ns): `29758547`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [73781, 128, 192], [73781, 128, 128]]}`
- `o_proj` -> 0.161 ms
  纯GPU kernel时间: `1.258 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.161 ms`, `host_to_first_kernel_gap=55.214463`, `host_end_to_last_kernel_tail=56.312 ms`, `gpu_makespan=1.259 ms`, `gpu_kernel_sum=1.258 ms`
  开始时间(ns): `29887561`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 4 / prefill / instance 1

- 执行序号: `5`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `1.414 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.414 ms`, `module_to_last_kernel=78.362 ms`, `host_to_first_kernel_gap=65.530079`, `host_end_to_last_kernel_tail=76.949 ms`, `gpu_makespan=12.832 ms`, `gpu_kernel_sum=10.593 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=73781`, `chunked_req_prefix_len=65589`, `current_chunked_req_prefix_len=65589`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[8192, 128, 192], [73781, 128, 192], [73781, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.208 ms
  纯GPU kernel时间: `0.280 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.208 ms`, `host_to_first_kernel_gap=65.403403`, `host_end_to_last_kernel_tail=65.477 ms`, `gpu_makespan=0.281 ms`, `gpu_kernel_sum=0.280 ms`
  开始时间(ns): `31353861`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.034 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.034 ms`, `host_to_first_kernel_gap=65.439488`, `host_end_to_last_kernel_tail=65.423 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `31599505`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.153 ms
  纯GPU kernel时间: `0.463 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.153 ms`, `host_to_first_kernel_gap=65.403019`, `host_end_to_last_kernel_tail=65.715 ms`, `gpu_makespan=0.465 ms`, `gpu_kernel_sum=0.463 ms`
  开始时间(ns): `31655910`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.028 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.028 ms`, `host_to_first_kernel_gap=65.672949`, `host_end_to_last_kernel_tail=65.658 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `31850620`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.055 ms
  纯GPU kernel时间: `0.106 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.055 ms`, `host_to_first_kernel_gap=65.632136`, `host_end_to_last_kernel_tail=65.683 ms`, `gpu_makespan=0.106 ms`, `gpu_kernel_sum=0.106 ms`
  开始时间(ns): `31905609`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.163 ms
  纯GPU kernel时间: `3.146 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.163 ms`, `host_to_first_kernel_gap=65.625283`, `host_end_to_last_kernel_tail=68.609 ms`, `gpu_makespan=3.147 ms`, `gpu_kernel_sum=3.146 ms`
  开始时间(ns): `32119502`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[73781, 512]]}`
- `attn_mha` -> 0.101 ms
  纯GPU kernel时间: `5.245 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.101 ms`, `host_to_first_kernel_gap=70.670611`, `host_end_to_last_kernel_tail=75.815 ms`, `gpu_makespan=5.245 ms`, `gpu_kernel_sum=5.245 ms`
  开始时间(ns): `32348674`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [73781, 128, 192], [73781, 128, 128]]}`
- `o_proj` -> 0.152 ms
  纯GPU kernel时间: `1.322 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.152 ms`, `host_to_first_kernel_gap=75.786737`, `host_end_to_last_kernel_tail=76.958 ms`, `gpu_makespan=1.324 ms`, `gpu_kernel_sum=1.322 ms`
  开始时间(ns): `32479015`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 5 / prefill / instance 1

- 执行序号: `6`
- Collector对齐边界: `{'Module': 'model.model.layers.5.self_attn'}`
- 整块 MLA-module 时长: `1.427 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.427 ms`, `module_to_last_kernel=98.936 ms`, `host_to_first_kernel_gap=86.371168`, `host_end_to_last_kernel_tail=97.509 ms`, `gpu_makespan=12.565 ms`, `gpu_kernel_sum=10.325 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=73781`, `chunked_req_prefix_len=65589`, `current_chunked_req_prefix_len=65589`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[8192, 128, 192], [73781, 128, 192], [73781, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.201 ms
  纯GPU kernel时间: `0.279 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.201 ms`, `host_to_first_kernel_gap=86.264605`, `host_end_to_last_kernel_tail=86.344 ms`, `gpu_makespan=0.280 ms`, `gpu_kernel_sum=0.279 ms`
  开始时间(ns): `33726244`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.036 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.036 ms`, `host_to_first_kernel_gap=86.305272`, `host_end_to_last_kernel_tail=86.287 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `33966409`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.153 ms
  纯GPU kernel时间: `0.460 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.153 ms`, `host_to_first_kernel_gap=86.265733`, `host_end_to_last_kernel_tail=86.574 ms`, `gpu_makespan=0.461 ms`, `gpu_kernel_sum=0.460 ms`
  开始时间(ns): `34025980`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.038 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.038 ms`, `host_to_first_kernel_gap=86.531449`, `host_end_to_last_kernel_tail=86.507 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `34221544`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.065 ms
  纯GPU kernel时间: `0.106 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.065 ms`, `host_to_first_kernel_gap=86.482535`, `host_end_to_last_kernel_tail=86.524 ms`, `gpu_makespan=0.106 ms`, `gpu_kernel_sum=0.106 ms`
  开始时间(ns): `34285914`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.167 ms
  纯GPU kernel时间: `2.944 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.167 ms`, `host_to_first_kernel_gap=86.47838`, `host_end_to_last_kernel_tail=89.257 ms`, `gpu_makespan=2.946 ms`, `gpu_kernel_sum=2.944 ms`
  开始时间(ns): `34497205`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[73781, 512]]}`
- `attn_mha` -> 0.104 ms
  纯GPU kernel时间: `5.247 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.104 ms`, `host_to_first_kernel_gap=91.313852`, `host_end_to_last_kernel_tail=96.456 ms`, `gpu_makespan=5.247 ms`, `gpu_kernel_sum=5.247 ms`
  开始时间(ns): `34734153`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [73781, 128, 192], [73781, 128, 128]]}`
- `o_proj` -> 0.166 ms
  纯GPU kernel时间: `1.257 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.166 ms`, `host_to_first_kernel_gap=96.426781`, `host_end_to_last_kernel_tail=97.520 ms`, `gpu_makespan=1.259 ms`, `gpu_kernel_sum=1.257 ms`
  开始时间(ns): `34870123`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`
