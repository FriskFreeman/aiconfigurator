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
- 整块 MLA-module 时长: `1.640 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.640 ms`, `module_to_last_kernel=12.883 ms`, `host_to_first_kernel_gap=0.208106`, `host_end_to_last_kernel_tail=11.243 ms`, `gpu_makespan=12.675 ms`, `gpu_kernel_sum=10.327 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=73781`, `chunked_req_prefix_len=65589`, `current_chunked_req_prefix_len=65589`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[8192, 128, 192], [73781, 128, 192], [73781, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.268 ms
  纯GPU kernel时间: `0.274 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.268 ms`, `host_to_first_kernel_gap=0.091414`, `host_end_to_last_kernel_tail=0.180 ms`, `gpu_makespan=0.356 ms`, `gpu_kernel_sum=0.274 ms`
  开始时间(ns): `23107983`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.032 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.032 ms`, `host_to_first_kernel_gap=0.141952`, `host_end_to_last_kernel_tail=0.128 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `23413797`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.148 ms
  纯GPU kernel时间: `0.443 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.148 ms`, `host_to_first_kernel_gap=0.106991`, `host_end_to_last_kernel_tail=0.412 ms`, `gpu_makespan=0.453 ms`, `gpu_kernel_sum=0.443 ms`
  开始时间(ns): `23467798`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.026 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.026 ms`, `host_to_first_kernel_gap=0.37489`, `host_end_to_last_kernel_tail=0.362 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `23652731`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.064 ms
  纯GPU kernel时间: `0.107 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.064 ms`, `host_to_first_kernel_gap=0.337086`, `host_end_to_last_kernel_tail=0.380 ms`, `gpu_makespan=0.107 ms`, `gpu_kernel_sum=0.107 ms`
  开始时间(ns): `23705063`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.183 ms
  纯GPU kernel时间: `2.985 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.183 ms`, `host_to_first_kernel_gap=0.196911`, `host_end_to_last_kernel_tail=3.000 ms`, `gpu_makespan=2.987 ms`, `gpu_kernel_sum=2.985 ms`
  开始时间(ns): `24074710`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[73781, 512]]}`
- `attn_mha` -> 0.111 ms
  纯GPU kernel时间: `5.230 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.111 ms`, `host_to_first_kernel_gap=5.050705`, `host_end_to_last_kernel_tail=10.169 ms`, `gpu_makespan=5.230 ms`, `gpu_kernel_sum=5.230 ms`
  开始时间(ns): `24332851`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [73781, 128, 192], [73781, 128, 128]]}`
- `o_proj` -> 0.151 ms
  纯GPU kernel时间: `1.258 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.151 ms`, `host_to_first_kernel_gap=10.14329`, `host_end_to_last_kernel_tail=11.252 ms`, `gpu_makespan=1.260 ms`, `gpu_kernel_sum=1.258 ms`
  开始时间(ns): `24471338`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 1 / prefill / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `1.141 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.141 ms`, `module_to_last_kernel=27.692 ms`, `host_to_first_kernel_gap=15.330577`, `host_end_to_last_kernel_tail=26.551 ms`, `gpu_makespan=12.361 ms`, `gpu_kernel_sum=10.121 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=73781`, `chunked_req_prefix_len=65589`, `current_chunked_req_prefix_len=65589`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[8192, 128, 192], [73781, 128, 192], [73781, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.133 ms
  纯GPU kernel时间: `0.268 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.133 ms`, `host_to_first_kernel_gap=15.242577`, `host_end_to_last_kernel_tail=15.379 ms`, `gpu_makespan=0.269 ms`, `gpu_kernel_sum=0.268 ms`
  开始时间(ns): `25240690`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.028 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.028 ms`, `host_to_first_kernel_gap=15.347758`, `host_end_to_last_kernel_tail=15.337 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `25405493`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.117 ms
  纯GPU kernel时间: `0.444 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.117 ms`, `host_to_first_kernel_gap=15.321526`, `host_end_to_last_kernel_tail=15.649 ms`, `gpu_makespan=0.445 ms`, `gpu_kernel_sum=0.444 ms`
  开始时间(ns): `25450701`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.025 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.025 ms`, `host_to_first_kernel_gap=15.615292`, `host_end_to_last_kernel_tail=15.603 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `25601895`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.049 ms
  纯GPU kernel时间: `0.106 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.049 ms`, `host_to_first_kernel_gap=15.582323`, `host_end_to_last_kernel_tail=15.640 ms`, `gpu_makespan=0.106 ms`, `gpu_kernel_sum=0.106 ms`
  开始时间(ns): `25649744`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.138 ms
  纯GPU kernel时间: `2.778 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.138 ms`, `host_to_first_kernel_gap=15.614203`, `host_end_to_last_kernel_tail=18.257 ms`, `gpu_makespan=2.780 ms`, `gpu_kernel_sum=2.778 ms`
  开始时间(ns): `25825256`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[73781, 512]]}`
- `attn_mha` -> 0.102 ms
  纯GPU kernel时间: `5.236 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.102 ms`, `host_to_first_kernel_gap=20.323502`, `host_end_to_last_kernel_tail=25.457 ms`, `gpu_makespan=5.236 ms`, `gpu_kernel_sum=5.236 ms`
  开始时间(ns): `26022772`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [73781, 128, 192], [73781, 128, 128]]}`
- `o_proj` -> 0.136 ms
  纯GPU kernel时间: `1.258 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.136 ms`, `host_to_first_kernel_gap=25.435677`, `host_end_to_last_kernel_tail=26.559 ms`, `gpu_makespan=1.260 ms`, `gpu_kernel_sum=1.258 ms`
  开始时间(ns): `26148996`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 2 / prefill / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.083 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.083 ms`, `module_to_last_kernel=43.203 ms`, `host_to_first_kernel_gap=30.69906`, `host_end_to_last_kernel_tail=42.120 ms`, `gpu_makespan=12.504 ms`, `gpu_kernel_sum=10.268 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=73781`, `chunked_req_prefix_len=65589`, `current_chunked_req_prefix_len=65589`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[8192, 128, 192], [73781, 128, 192], [73781, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.133 ms
  纯GPU kernel时间: `0.266 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.133 ms`, `host_to_first_kernel_gap=30.613897`, `host_end_to_last_kernel_tail=30.748 ms`, `gpu_makespan=0.268 ms`, `gpu_kernel_sum=0.266 ms`
  开始时间(ns): `26834072`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.027 ms
  纯GPU kernel时间: `0.017 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.027 ms`, `host_to_first_kernel_gap=30.721685`, `host_end_to_last_kernel_tail=30.712 ms`, `gpu_makespan=0.017 ms`, `gpu_kernel_sum=0.017 ms`
  开始时间(ns): `26994444`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.126 ms
  纯GPU kernel时间: `0.439 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.126 ms`, `host_to_first_kernel_gap=30.695077`, `host_end_to_last_kernel_tail=31.010 ms`, `gpu_makespan=0.441 ms`, `gpu_kernel_sum=0.439 ms`
  开始时间(ns): `27039740`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.023 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.023 ms`, `host_to_first_kernel_gap=30.977341`, `host_end_to_last_kernel_tail=30.967 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `27198340`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.045 ms
  纯GPU kernel时间: `0.106 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.045 ms`, `host_to_first_kernel_gap=30.946559`, `host_end_to_last_kernel_tail=31.008 ms`, `gpu_makespan=0.106 ms`, `gpu_kernel_sum=0.106 ms`
  开始时间(ns): `27243618`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.128 ms
  纯GPU kernel时间: `2.906 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.128 ms`, `host_to_first_kernel_gap=30.986368`, `host_end_to_last_kernel_tail=33.766 ms`, `gpu_makespan=2.908 ms`, `gpu_kernel_sum=2.906 ms`
  开始时间(ns): `27411297`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[73781, 512]]}`
- `attn_mha` -> 0.089 ms
  纯GPU kernel时间: `5.235 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.089 ms`, `host_to_first_kernel_gap=35.837301`, `host_end_to_last_kernel_tail=40.984 ms`, `gpu_makespan=5.236 ms`, `gpu_kernel_sum=5.235 ms`
  开始时间(ns): `27592299`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [73781, 128, 192], [73781, 128, 128]]}`
- `o_proj` -> 0.121 ms
  纯GPU kernel时间: `1.284 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.121 ms`, `host_to_first_kernel_gap=40.963875`, `host_end_to_last_kernel_tail=42.128 ms`, `gpu_makespan=1.286 ms`, `gpu_kernel_sum=1.284 ms`
  开始时间(ns): `27702876`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 3 / prefill / instance 1

- 执行序号: `4`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `1.076 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.076 ms`, `module_to_last_kernel=59.078 ms`, `host_to_first_kernel_gap=46.353582`, `host_end_to_last_kernel_tail=58.002 ms`, `gpu_makespan=12.724 ms`, `gpu_kernel_sum=10.485 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=73781`, `chunked_req_prefix_len=65589`, `current_chunked_req_prefix_len=65589`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[8192, 128, 192], [73781, 128, 192], [73781, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.134 ms
  纯GPU kernel时间: `0.272 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.134 ms`, `host_to_first_kernel_gap=46.269327`, `host_end_to_last_kernel_tail=46.410 ms`, `gpu_makespan=0.274 ms`, `gpu_kernel_sum=0.272 ms`
  开始时间(ns): `28357263`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.028 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.028 ms`, `host_to_first_kernel_gap=46.382237`, `host_end_to_last_kernel_tail=46.372 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `28519329`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.126 ms
  纯GPU kernel时间: `0.449 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.126 ms`, `host_to_first_kernel_gap=46.356244`, `host_end_to_last_kernel_tail=46.681 ms`, `gpu_makespan=0.451 ms`, `gpu_kernel_sum=0.449 ms`
  开始时间(ns): `28565130`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.023 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.023 ms`, `host_to_first_kernel_gap=46.648011`, `host_end_to_last_kernel_tail=46.638 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `28724307`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.044 ms
  纯GPU kernel时间: `0.107 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.044 ms`, `host_to_first_kernel_gap=46.616788`, `host_end_to_last_kernel_tail=46.679 ms`, `gpu_makespan=0.107 ms`, `gpu_kernel_sum=0.107 ms`
  开始时间(ns): `28769834`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.124 ms
  纯GPU kernel时间: `3.101 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.124 ms`, `host_to_first_kernel_gap=46.661284`, `host_end_to_last_kernel_tail=49.639 ms`, `gpu_makespan=3.103 ms`, `gpu_kernel_sum=3.101 ms`
  开始时间(ns): `28934522`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[73781, 512]]}`
- `attn_mha` -> 0.085 ms
  纯GPU kernel时间: `5.241 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.085 ms`, `host_to_first_kernel_gap=51.711143`, `host_end_to_last_kernel_tail=56.867 ms`, `gpu_makespan=5.241 ms`, `gpu_kernel_sum=5.241 ms`
  开始时间(ns): `29112503`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [73781, 128, 192], [73781, 128, 128]]}`
- `o_proj` -> 0.122 ms
  纯GPU kernel时间: `1.284 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.122 ms`, `host_to_first_kernel_gap=56.847069`, `host_end_to_last_kernel_tail=58.009 ms`, `gpu_makespan=1.285 ms`, `gpu_kernel_sum=1.284 ms`
  开始时间(ns): `29219200`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 4 / prefill / instance 1

- 执行序号: `5`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `1.171 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.171 ms`, `module_to_last_kernel=80.040 ms`, `host_to_first_kernel_gap=67.514084`, `host_end_to_last_kernel_tail=78.870 ms`, `gpu_makespan=12.526 ms`, `gpu_kernel_sum=10.289 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=73781`, `chunked_req_prefix_len=65589`, `current_chunked_req_prefix_len=65589`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[8192, 128, 192], [73781, 128, 192], [73781, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.174 ms
  纯GPU kernel时间: `0.279 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.174 ms`, `host_to_first_kernel_gap=67.424762`, `host_end_to_last_kernel_tail=67.532 ms`, `gpu_makespan=0.281 ms`, `gpu_kernel_sum=0.279 ms`
  开始时间(ns): `30315585`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.029 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.029 ms`, `host_to_first_kernel_gap=67.50202`, `host_end_to_last_kernel_tail=67.491 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `30519543`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.128 ms
  纯GPU kernel时间: `0.465 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.128 ms`, `host_to_first_kernel_gap=67.475383`, `host_end_to_last_kernel_tail=67.814 ms`, `gpu_makespan=0.467 ms`, `gpu_kernel_sum=0.465 ms`
  开始时间(ns): `30566116`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.023 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.023 ms`, `host_to_first_kernel_gap=67.780622`, `host_end_to_last_kernel_tail=67.770 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `30727533`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.048 ms
  纯GPU kernel时间: `0.107 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.048 ms`, `host_to_first_kernel_gap=67.749673`, `host_end_to_last_kernel_tail=67.808 ms`, `gpu_makespan=0.107 ms`, `gpu_kernel_sum=0.107 ms`
  开始时间(ns): `30772914`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.128 ms
  纯GPU kernel时间: `2.854 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.128 ms`, `host_to_first_kernel_gap=67.788879`, `host_end_to_last_kernel_tail=70.516 ms`, `gpu_makespan=2.855 ms`, `gpu_kernel_sum=2.854 ms`
  开始时间(ns): `30940684`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[73781, 512]]}`
- `attn_mha` -> 0.083 ms
  纯GPU kernel时间: `5.241 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.083 ms`, `host_to_first_kernel_gap=72.577345`, `host_end_to_last_kernel_tail=77.735 ms`, `gpu_makespan=5.241 ms`, `gpu_kernel_sum=5.241 ms`
  开始时间(ns): `31133241`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [73781, 128, 192], [73781, 128, 128]]}`
- `o_proj` -> 0.149 ms
  纯GPU kernel时间: `1.313 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.149 ms`, `host_to_first_kernel_gap=77.712798`, `host_end_to_last_kernel_tail=78.878 ms`, `gpu_makespan=1.314 ms`, `gpu_kernel_sum=1.313 ms`
  开始时间(ns): `31240060`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 5 / prefill / instance 1

- 执行序号: `6`
- Collector对齐边界: `{'Module': 'model.model.layers.5.self_attn'}`
- 整块 MLA-module 时长: `1.147 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.147 ms`, `module_to_last_kernel=101.083 ms`, `host_to_first_kernel_gap=88.426642`, `host_end_to_last_kernel_tail=99.936 ms`, `gpu_makespan=12.656 ms`, `gpu_kernel_sum=10.419 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=73781`, `chunked_req_prefix_len=65589`, `current_chunked_req_prefix_len=65589`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[8192, 128, 192], [73781, 128, 192], [73781, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.173 ms
  纯GPU kernel时间: `0.279 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.173 ms`, `host_to_first_kernel_gap=88.338138`, `host_end_to_last_kernel_tail=88.446 ms`, `gpu_makespan=0.281 ms`, `gpu_kernel_sum=0.279 ms`
  开始时间(ns): `32268926`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.030 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.030 ms`, `host_to_first_kernel_gap=88.416913`, `host_end_to_last_kernel_tail=88.406 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `32471111`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.126 ms
  纯GPU kernel时间: `0.458 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.126 ms`, `host_to_first_kernel_gap=88.388294`, `host_end_to_last_kernel_tail=88.721 ms`, `gpu_makespan=0.459 ms`, `gpu_kernel_sum=0.458 ms`
  开始时间(ns): `32518930`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.025 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.025 ms`, `host_to_first_kernel_gap=88.687467`, `host_end_to_last_kernel_tail=88.676 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `32678797`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.046 ms
  纯GPU kernel时间: `0.106 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.046 ms`, `host_to_first_kernel_gap=88.655364`, `host_end_to_last_kernel_tail=88.716 ms`, `gpu_makespan=0.106 ms`, `gpu_kernel_sum=0.106 ms`
  开始时间(ns): `32726068`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.124 ms
  纯GPU kernel时间: `2.985 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.124 ms`, `host_to_first_kernel_gap=88.695993`, `host_end_to_last_kernel_tail=91.559 ms`, `gpu_makespan=2.987 ms`, `gpu_kernel_sum=2.985 ms`
  开始时间(ns): `32892895`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[73781, 512]]}`
- `attn_mha` -> 0.101 ms
  纯GPU kernel时间: `5.242 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.101 ms`, `host_to_first_kernel_gap=93.629164`, `host_end_to_last_kernel_tail=98.771 ms`, `gpu_makespan=5.242 ms`, `gpu_kernel_sum=5.242 ms`
  开始时间(ns): `33070635`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [73781, 128, 192], [73781, 128, 128]]}`
- `o_proj` -> 0.126 ms
  纯GPU kernel时间: `1.317 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.126 ms`, `host_to_first_kernel_gap=98.75066`, `host_end_to_last_kernel_tail=99.944 ms`, `gpu_makespan=1.319 ms`, `gpu_kernel_sum=1.317 ms`
  开始时间(ns): `33193843`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`
