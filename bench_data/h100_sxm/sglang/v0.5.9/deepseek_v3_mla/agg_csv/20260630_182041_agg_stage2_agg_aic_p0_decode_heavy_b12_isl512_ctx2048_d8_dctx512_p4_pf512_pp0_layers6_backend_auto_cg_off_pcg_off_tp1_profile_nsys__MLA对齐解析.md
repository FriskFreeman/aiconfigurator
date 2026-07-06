# MLA对齐解析摘要

源文件: `report.sqlite`

## 对齐原则

- `collector/sglang/collect_mla_module.py` 的 MLA module 计时边界是 `model.model.layers[test_layer].self_attn(...)`。
- 因此这里把 `nsys` 中每层的 `model.model.layers.X.self_attn` NVTX range 视为与 collector 对齐的 MLA-module 边界。
- 该区间内部的 `.self_attn.*` 子模块用于做 MLA 内部 breakdown。

## 运行摘要

- `prefill` 对齐成功层: `[]`
- `prefill` 被切分层: `[]`
- 若某层 `prefill` 出现多个 `self_attn` 实例，则说明 Engine 调度把一次前向切成了多块，已不再与 collector 的单次 MLA-module 采集严格一一对应。

## Layer 0 / decode / instance 1

- 执行序号: `1`
- Collector对齐边界: `{'Module': 'model.model.layers.0.self_attn'}`
- 整块 MLA-module 时长: `1.405 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.405 ms`, `module_to_last_kernel=2.035 ms`, `host_to_first_kernel_gap=0.14138`, `host_end_to_last_kernel_tail=0.629 ms`, `gpu_makespan=1.893 ms`, `gpu_kernel_sum=1.105 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.255 ms
  纯GPU kernel时间: `0.081 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.255 ms`, `host_to_first_kernel_gap=0.085556`, `host_end_to_last_kernel_tail=0.039 ms`, `gpu_makespan=0.209 ms`, `gpu_kernel_sum=0.081 ms`
  开始时间(ns): `21159622`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2056, 7168]]}`
- `q_a_layernorm` -> 0.035 ms
  纯GPU kernel时间: `0.006 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.035 ms`, `host_to_first_kernel_gap=0.032597`, `host_end_to_last_kernel_tail=0.003 ms`, `gpu_makespan=0.006 ms`, `gpu_kernel_sum=0.006 ms`
  开始时间(ns): `21461605`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2056, 1536]]}`
- `kv_a_layernorm` -> 0.030 ms
  纯GPU kernel时间: `0.004 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.030 ms`, `host_to_first_kernel_gap=0.02917`, `host_end_to_last_kernel_tail=0.003 ms`, `gpu_makespan=0.004 ms`, `gpu_kernel_sum=0.004 ms`
  开始时间(ns): `21510696`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2056, 512]]}`
- `q_b_proj` -> 0.165 ms
  纯GPU kernel时间: `0.135 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.165 ms`, `host_to_first_kernel_gap=0.058062`, `host_end_to_last_kernel_tail=0.113 ms`, `gpu_makespan=0.220 ms`, `gpu_kernel_sum=0.135 ms`
  开始时间(ns): `21576396`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2056, 1536]]}`
- `rotary_emb` -> 0.067 ms
  纯GPU kernel时间: `0.028 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.067 ms`, `host_to_first_kernel_gap=0.110705`, `host_end_to_last_kernel_tail=0.071 ms`, `gpu_makespan=0.028 ms`, `gpu_kernel_sum=0.028 ms`
  开始时间(ns): `21881865`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2056], [2056, 128, 64], [2056, 1, 64]]}`
- `attn_mqa` -> 0.200 ms
  纯GPU kernel时间: `0.474 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.200 ms`, `host_to_first_kernel_gap=0.075469`, `host_end_to_last_kernel_tail=0.447 ms`, `gpu_makespan=0.572 ms`, `gpu_kernel_sum=0.474 ms`
  开始时间(ns): `21980781`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2056, 128, 512], [2056, 1, 512], [2056, 1, 512]]}`
- `o_proj` -> 0.203 ms
  纯GPU kernel时间: `0.378 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.203 ms`, `host_to_first_kernel_gap=0.46655`, `host_end_to_last_kernel_tail=0.642 ms`, `gpu_makespan=0.379 ms`, `gpu_kernel_sum=0.378 ms`
  开始时间(ns): `22292900`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2056, 16384]]}`

## Layer 1 / decode / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `1.132 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.132 ms`, `module_to_last_kernel=2.799 ms`, `host_to_first_kernel_gap=1.402501`, `host_end_to_last_kernel_tail=1.667 ms`, `gpu_makespan=1.396 ms`, `gpu_kernel_sum=1.116 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.131 ms
  纯GPU kernel时间: `0.080 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.131 ms`, `host_to_first_kernel_gap=1.357335`, `host_end_to_last_kernel_tail=1.307 ms`, `gpu_makespan=0.081 ms`, `gpu_kernel_sum=0.080 ms`
  开始时间(ns): `23104994`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2056, 7168]]}`
- `q_a_layernorm` -> 0.032 ms
  纯GPU kernel时间: `0.006 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.032 ms`, `host_to_first_kernel_gap=1.268374`, `host_end_to_last_kernel_tail=1.242 ms`, `gpu_makespan=0.006 ms`, `gpu_kernel_sum=0.006 ms`
  开始时间(ns): `23275267`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2056, 1536]]}`
- `kv_a_layernorm` -> 0.021 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.021 ms`, `host_to_first_kernel_gap=1.22527`, `host_end_to_last_kernel_tail=1.209 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `23324035`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2056, 512]]}`
- `q_b_proj` -> 0.123 ms
  纯GPU kernel时间: `0.131 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.123 ms`, `host_to_first_kernel_gap=1.190305`, `host_end_to_last_kernel_tail=1.199 ms`, `gpu_makespan=0.132 ms`, `gpu_kernel_sum=0.131 ms`
  开始时间(ns): `23364952`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2056, 1536]]}`
- `rotary_emb` -> 0.061 ms
  纯GPU kernel时间: `0.028 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.061 ms`, `host_to_first_kernel_gap=1.216904`, `host_end_to_last_kernel_tail=1.184 ms`, `gpu_makespan=0.028 ms`, `gpu_kernel_sum=0.028 ms`
  开始时间(ns): `23611281`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2056], [2056, 128, 64], [2056, 1, 64]]}`
- `attn_mqa` -> 0.192 ms
  纯GPU kernel时间: `0.487 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.192 ms`, `host_to_first_kernel_gap=1.158482`, `host_end_to_last_kernel_tail=1.455 ms`, `gpu_makespan=0.488 ms`, `gpu_kernel_sum=0.487 ms`
  开始时间(ns): `23698887`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2056, 128, 512], [2056, 1, 512], [2056, 1, 512]]}`
- `o_proj` -> 0.189 ms
  纯GPU kernel时间: `0.379 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.189 ms`, `host_to_first_kernel_gap=1.487247`, `host_end_to_last_kernel_tail=1.679 ms`, `gpu_makespan=0.380 ms`, `gpu_kernel_sum=0.379 ms`
  开始时间(ns): `23991018`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2056, 16384]]}`

## Layer 2 / decode / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.125 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.125 ms`, `module_to_last_kernel=3.850 ms`, `host_to_first_kernel_gap=2.474506`, `host_end_to_last_kernel_tail=2.725 ms`, `gpu_makespan=1.375 ms`, `gpu_kernel_sum=1.094 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.129 ms
  纯GPU kernel时间: `0.079 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.129 ms`, `host_to_first_kernel_gap=2.429545`, `host_end_to_last_kernel_tail=2.383 ms`, `gpu_makespan=0.082 ms`, `gpu_kernel_sum=0.079 ms`
  开始时间(ns): `24764271`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2056, 7168]]}`
- `q_a_layernorm` -> 0.032 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.032 ms`, `host_to_first_kernel_gap=2.344872`, `host_end_to_last_kernel_tail=2.318 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `24931024`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2056, 1536]]}`
- `kv_a_layernorm` -> 0.021 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.021 ms`, `host_to_first_kernel_gap=2.30478`, `host_end_to_last_kernel_tail=2.289 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `24976332`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2056, 512]]}`
- `q_b_proj` -> 0.120 ms
  纯GPU kernel时间: `0.131 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.120 ms`, `host_to_first_kernel_gap=2.270158`, `host_end_to_last_kernel_tail=2.282 ms`, `gpu_makespan=0.132 ms`, `gpu_kernel_sum=0.131 ms`
  开始时间(ns): `25016938`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2056, 1536]]}`
- `rotary_emb` -> 0.057 ms
  纯GPU kernel时间: `0.029 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.057 ms`, `host_to_first_kernel_gap=2.308513`, `host_end_to_last_kernel_tail=2.280 ms`, `gpu_makespan=0.029 ms`, `gpu_kernel_sum=0.029 ms`
  开始时间(ns): `25249463`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2056], [2056, 128, 64], [2056, 1, 64]]}`
- `attn_mqa` -> 0.197 ms
  纯GPU kernel时间: `0.469 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.197 ms`, `host_to_first_kernel_gap=2.253693`, `host_end_to_last_kernel_tail=2.527 ms`, `gpu_makespan=0.471 ms`, `gpu_kernel_sum=0.469 ms`
  开始时间(ns): `25334203`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2056, 128, 512], [2056, 1, 512], [2056, 1, 512]]}`
- `o_proj` -> 0.191 ms
  纯GPU kernel时间: `0.376 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.191 ms`, `host_to_first_kernel_gap=2.549751`, `host_end_to_last_kernel_tail=2.737 ms`, `gpu_makespan=0.378 ms`, `gpu_kernel_sum=0.376 ms`
  开始时间(ns): `25640896`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2056, 16384]]}`

## Layer 3 / decode / instance 1

- 执行序号: `4`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `1.104 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.104 ms`, `module_to_last_kernel=4.930 ms`, `host_to_first_kernel_gap=3.537623`, `host_end_to_last_kernel_tail=3.826 ms`, `gpu_makespan=1.393 ms`, `gpu_kernel_sum=1.111 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.127 ms
  纯GPU kernel时间: `0.081 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.127 ms`, `host_to_first_kernel_gap=3.494727`, `host_end_to_last_kernel_tail=3.451 ms`, `gpu_makespan=0.083 ms`, `gpu_kernel_sum=0.081 ms`
  开始时间(ns): `26401935`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2056, 7168]]}`
- `q_a_layernorm` -> 0.032 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.032 ms`, `host_to_first_kernel_gap=3.406254`, `host_end_to_last_kernel_tail=3.380 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `26573736`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2056, 1536]]}`
- `kv_a_layernorm` -> 0.019 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.019 ms`, `host_to_first_kernel_gap=3.36658`, `host_end_to_last_kernel_tail=3.352 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `26618914`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2056, 512]]}`
- `q_b_proj` -> 0.125 ms
  纯GPU kernel时间: `0.131 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.125 ms`, `host_to_first_kernel_gap=3.334145`, `host_end_to_last_kernel_tail=3.342 ms`, `gpu_makespan=0.133 ms`, `gpu_kernel_sum=0.131 ms`
  开始时间(ns): `26658005`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2056, 1536]]}`
- `rotary_emb` -> 0.059 ms
  纯GPU kernel时间: `0.029 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.059 ms`, `host_to_first_kernel_gap=3.367937`, `host_end_to_last_kernel_tail=3.337 ms`, `gpu_makespan=0.029 ms`, `gpu_kernel_sum=0.029 ms`
  开始时间(ns): `26893653`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2056], [2056, 128, 64], [2056, 1, 64]]}`
- `attn_mqa` -> 0.190 ms
  纯GPU kernel时间: `0.483 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.190 ms`, `host_to_first_kernel_gap=3.312844`, `host_end_to_last_kernel_tail=3.607 ms`, `gpu_makespan=0.484 ms`, `gpu_kernel_sum=0.483 ms`
  开始时间(ns): `26979498`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2056, 128, 512], [2056, 1, 512], [2056, 1, 512]]}`
- `o_proj` -> 0.187 ms
  纯GPU kernel时间: `0.377 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.187 ms`, `host_to_first_kernel_gap=3.644466`, `host_end_to_last_kernel_tail=3.838 ms`, `gpu_makespan=0.380 ms`, `gpu_kernel_sum=0.377 ms`
  开始时间(ns): `27265252`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2056, 16384]]}`

## Layer 4 / decode / instance 1

- 执行序号: `5`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `1.203 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.203 ms`, `module_to_last_kernel=6.978 ms`, `host_to_first_kernel_gap=5.584739`, `host_end_to_last_kernel_tail=5.774 ms`, `gpu_makespan=1.393 ms`, `gpu_kernel_sum=1.115 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.192 ms
  纯GPU kernel时间: `0.080 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.192 ms`, `host_to_first_kernel_gap=5.537269`, `host_end_to_last_kernel_tail=5.427 ms`, `gpu_makespan=0.082 ms`, `gpu_kernel_sum=0.080 ms`
  开始时间(ns): `28414432`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2056, 7168]]}`
- `q_a_layernorm` -> 0.035 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.035 ms`, `host_to_first_kernel_gap=5.384097`, `host_end_to_last_kernel_tail=5.354 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `28649619`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2056, 1536]]}`
- `kv_a_layernorm` -> 0.019 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.019 ms`, `host_to_first_kernel_gap=5.339126`, `host_end_to_last_kernel_tail=5.325 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `28699934`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2056, 512]]}`
- `q_b_proj` -> 0.144 ms
  纯GPU kernel时间: `0.133 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.144 ms`, `host_to_first_kernel_gap=5.303187`, `host_end_to_last_kernel_tail=5.293 ms`, `gpu_makespan=0.134 ms`, `gpu_kernel_sum=0.133 ms`
  开始时间(ns): `28741793`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2056, 1536]]}`
- `rotary_emb` -> 0.060 ms
  纯GPU kernel时间: `0.028 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.060 ms`, `host_to_first_kernel_gap=5.310493`, `host_end_to_last_kernel_tail=5.279 ms`, `gpu_makespan=0.028 ms`, `gpu_kernel_sum=0.028 ms`
  开始时间(ns): `29005367`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2056], [2056, 128, 64], [2056, 1, 64]]}`
- `attn_mqa` -> 0.185 ms
  纯GPU kernel时间: `0.488 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.185 ms`, `host_to_first_kernel_gap=5.251116`, `host_end_to_last_kernel_tail=5.556 ms`, `gpu_makespan=0.490 ms`, `gpu_kernel_sum=0.488 ms`
  开始时间(ns): `29093736`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2056, 128, 512], [2056, 1, 512], [2056, 1, 512]]}`
- `o_proj` -> 0.174 ms
  纯GPU kernel时间: `0.377 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.174 ms`, `host_to_first_kernel_gap=5.592531`, `host_end_to_last_kernel_tail=5.796 ms`, `gpu_makespan=0.378 ms`, `gpu_kernel_sum=0.377 ms`
  开始时间(ns): `29373921`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2056, 16384]]}`

## Layer 5 / decode / instance 1

- 执行序号: `6`
- Collector对齐边界: `{'Module': 'model.model.layers.5.self_attn'}`
- 整块 MLA-module 时长: `1.193 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.193 ms`, `module_to_last_kernel=8.944 ms`, `host_to_first_kernel_gap=7.555219`, `host_end_to_last_kernel_tail=7.751 ms`, `gpu_makespan=1.389 ms`, `gpu_kernel_sum=1.109 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.199 ms
  纯GPU kernel时间: `0.080 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.199 ms`, `host_to_first_kernel_gap=7.506594`, `host_end_to_last_kernel_tail=7.389 ms`, `gpu_makespan=0.081 ms`, `gpu_kernel_sum=0.080 ms`
  开始时间(ns): `30485777`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2056, 7168]]}`
- `q_a_layernorm` -> 0.035 ms
  纯GPU kernel时间: `0.006 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.035 ms`, `host_to_first_kernel_gap=7.348225`, `host_end_to_last_kernel_tail=7.319 ms`, `gpu_makespan=0.006 ms`, `gpu_kernel_sum=0.006 ms`
  开始时间(ns): `30725106`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2056, 1536]]}`
- `kv_a_layernorm` -> 0.019 ms
  纯GPU kernel时间: `0.004 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.019 ms`, `host_to_first_kernel_gap=7.305269`, `host_end_to_last_kernel_tail=7.291 ms`, `gpu_makespan=0.004 ms`, `gpu_kernel_sum=0.004 ms`
  开始时间(ns): `30773438`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2056, 512]]}`
- `q_b_proj` -> 0.139 ms
  纯GPU kernel时间: `0.130 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.139 ms`, `host_to_first_kernel_gap=7.271287`, `host_end_to_last_kernel_tail=7.264 ms`, `gpu_makespan=0.131 ms`, `gpu_kernel_sum=0.130 ms`
  开始时间(ns): `30813308`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2056, 1536]]}`
- `rotary_emb` -> 0.059 ms
  纯GPU kernel时间: `0.028 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.059 ms`, `host_to_first_kernel_gap=7.282416`, `host_end_to_last_kernel_tail=7.251 ms`, `gpu_makespan=0.028 ms`, `gpu_kernel_sum=0.028 ms`
  开始时间(ns): `31073122`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2056], [2056, 128, 64], [2056, 1, 64]]}`
- `attn_mqa` -> 0.179 ms
  纯GPU kernel时间: `0.480 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.179 ms`, `host_to_first_kernel_gap=7.224035`, `host_end_to_last_kernel_tail=7.526 ms`, `gpu_makespan=0.481 ms`, `gpu_kernel_sum=0.480 ms`
  开始时间(ns): `31160527`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2056, 128, 512], [2056, 1, 512], [2056, 1, 512]]}`
- `o_proj` -> 0.184 ms
  纯GPU kernel时间: `0.381 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.184 ms`, `host_to_first_kernel_gap=7.564716`, `host_end_to_last_kernel_tail=7.763 ms`, `gpu_makespan=0.383 ms`, `gpu_kernel_sum=0.381 ms`
  开始时间(ns): `31433734`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2056, 16384]]}`
