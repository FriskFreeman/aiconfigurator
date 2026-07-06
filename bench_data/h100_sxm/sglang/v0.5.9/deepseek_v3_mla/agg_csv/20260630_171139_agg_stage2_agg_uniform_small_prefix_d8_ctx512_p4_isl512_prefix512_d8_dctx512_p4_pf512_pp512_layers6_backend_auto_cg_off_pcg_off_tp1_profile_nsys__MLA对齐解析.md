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
- 整块 MLA-module 时长: `2.208 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.208 ms`, `module_to_last_kernel=3.088 ms`, `host_to_first_kernel_gap=0.389685`, `host_end_to_last_kernel_tail=0.879 ms`, `gpu_makespan=2.698 ms`, `gpu_kernel_sum=1.495 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.475 ms
  纯GPU kernel时间: `0.080 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.475 ms`, `host_to_first_kernel_gap=0.217093`, `host_end_to_last_kernel_tail=0.022 ms`, `gpu_makespan=0.279 ms`, `gpu_kernel_sum=0.080 ms`
  开始时间(ns): `22905604`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2056, 7168]]}`
- `q_a_layernorm` -> 0.050 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.050 ms`, `host_to_first_kernel_gap=0.043996`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `23456972`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2056, 1536]]}`
- `kv_a_layernorm` -> 0.031 ms
  纯GPU kernel时间: `0.004 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.031 ms`, `host_to_first_kernel_gap=0.027288`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.004 ms`, `gpu_kernel_sum=0.004 ms`
  开始时间(ns): `23543088`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2056, 512]]}`
- `q_b_proj` -> 0.248 ms
  纯GPU kernel时间: `0.136 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.248 ms`, `host_to_first_kernel_gap=0.087009`, `host_end_to_last_kernel_tail=0.102 ms`, `gpu_makespan=0.263 ms`, `gpu_kernel_sum=0.136 ms`
  开始时间(ns): `23613319`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2056, 1536]]}`
- `rotary_emb` -> 0.093 ms
  纯GPU kernel时间: `0.029 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.093 ms`, `host_to_first_kernel_gap=0.086005`, `host_end_to_last_kernel_tail=0.022 ms`, `gpu_makespan=0.029 ms`, `gpu_kernel_sum=0.029 ms`
  开始时间(ns): `24060851`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2056], [2056, 128, 64], [2056, 1, 64]]}`
- `attn_mqa` -> 0.283 ms
  纯GPU kernel时间: `0.871 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.283 ms`, `host_to_first_kernel_gap=0.096244`, `host_end_to_last_kernel_tail=0.831 ms`, `gpu_makespan=1.019 ms`, `gpu_kernel_sum=0.871 ms`
  开始时间(ns): `24203892`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2056, 128, 512], [2056, 1, 512], [2056, 1, 512]]}`
- `o_proj` -> 0.275 ms
  纯GPU kernel时间: `0.369 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.275 ms`, `host_to_first_kernel_gap=0.802888`, `host_end_to_last_kernel_tail=0.899 ms`, `gpu_makespan=0.371 ms`, `gpu_kernel_sum=0.369 ms`
  开始时间(ns): `24646944`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2056, 16384]]}`

## Layer 1 / decode / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `1.676 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.676 ms`, `module_to_last_kernel=3.131 ms`, `host_to_first_kernel_gap=1.377097`, `host_end_to_last_kernel_tail=1.455 ms`, `gpu_makespan=1.754 ms`, `gpu_kernel_sum=1.477 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.240 ms
  纯GPU kernel时间: `0.080 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.240 ms`, `host_to_first_kernel_gap=1.304666`, `host_end_to_last_kernel_tail=1.146 ms`, `gpu_makespan=0.082 ms`, `gpu_kernel_sum=0.080 ms`
  开始时间(ns): `25843118`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2056, 7168]]}`
- `q_a_layernorm` -> 0.052 ms
  纯GPU kernel时间: `0.006 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.052 ms`, `host_to_first_kernel_gap=1.083386`, `host_end_to_last_kernel_tail=1.036 ms`, `gpu_makespan=0.006 ms`, `gpu_kernel_sum=0.006 ms`
  开始时间(ns): `26145902`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2056, 1536]]}`
- `kv_a_layernorm` -> 0.034 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.034 ms`, `host_to_first_kernel_gap=1.01202`, `host_end_to_last_kernel_tail=0.983 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `26223124`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2056, 512]]}`
- `q_b_proj` -> 0.232 ms
  纯GPU kernel时间: `0.131 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.232 ms`, `host_to_first_kernel_gap=0.947649`, `host_end_to_last_kernel_tail=0.848 ms`, `gpu_makespan=0.132 ms`, `gpu_kernel_sum=0.131 ms`
  开始时间(ns): `26293415`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2056, 1536]]}`
- `rotary_emb` -> 0.077 ms
  纯GPU kernel时间: `0.028 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.077 ms`, `host_to_first_kernel_gap=0.829752`, `host_end_to_last_kernel_tail=0.780 ms`, `gpu_makespan=0.028 ms`, `gpu_kernel_sum=0.028 ms`
  开始时间(ns): `26682160`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2056], [2056, 128, 64], [2056, 1, 64]]}`
- `attn_mqa` -> 0.252 ms
  纯GPU kernel时间: `0.854 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.252 ms`, `host_to_first_kernel_gap=0.738448`, `host_end_to_last_kernel_tail=1.342 ms`, `gpu_makespan=0.855 ms`, `gpu_kernel_sum=0.854 ms`
  开始时间(ns): `26802808`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2056, 128, 512], [2056, 1, 512], [2056, 1, 512]]}`
- `o_proj` -> 0.250 ms
  纯GPU kernel时间: `0.374 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.250 ms`, `host_to_first_kernel_gap=1.347893`, `host_end_to_last_kernel_tail=1.474 ms`, `gpu_makespan=0.375 ms`, `gpu_kernel_sum=0.374 ms`
  开始时间(ns): `27178803`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2056, 16384]]}`

## Layer 2 / decode / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.604 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.604 ms`, `module_to_last_kernel=3.784 ms`, `host_to_first_kernel_gap=2.013207`, `host_end_to_last_kernel_tail=2.180 ms`, `gpu_makespan=1.771 ms`, `gpu_kernel_sum=1.492 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.205 ms
  纯GPU kernel时间: `0.081 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.205 ms`, `host_to_first_kernel_gap=1.955635`, `host_end_to_last_kernel_tail=1.833 ms`, `gpu_makespan=0.083 ms`, `gpu_kernel_sum=0.081 ms`
  开始时间(ns): `28277749`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2056, 7168]]}`
- `q_a_layernorm` -> 0.042 ms
  纯GPU kernel时间: `0.006 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.042 ms`, `host_to_first_kernel_gap=1.770824`, `host_end_to_last_kernel_tail=1.734 ms`, `gpu_makespan=0.006 ms`, `gpu_kernel_sum=0.006 ms`
  开始时间(ns): `28545408`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2056, 1536]]}`
- `kv_a_layernorm` -> 0.029 ms
  纯GPU kernel时间: `0.004 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.029 ms`, `host_to_first_kernel_gap=1.70931`, `host_end_to_last_kernel_tail=1.685 ms`, `gpu_makespan=0.004 ms`, `gpu_kernel_sum=0.004 ms`
  开始时间(ns): `28612554`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2056, 512]]}`
- `q_b_proj` -> 0.195 ms
  纯GPU kernel时间: `0.130 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.195 ms`, `host_to_first_kernel_gap=1.653668`, `host_end_to_last_kernel_tail=1.590 ms`, `gpu_makespan=0.132 ms`, `gpu_kernel_sum=0.130 ms`
  开始时间(ns): `28673892`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2056, 1536]]}`
- `rotary_emb` -> 0.076 ms
  纯GPU kernel时间: `0.028 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.076 ms`, `host_to_first_kernel_gap=1.57494`, `host_end_to_last_kernel_tail=1.527 ms`, `gpu_makespan=0.028 ms`, `gpu_kernel_sum=0.028 ms`
  开始时间(ns): `29022956`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2056], [2056, 128, 64], [2056, 1, 64]]}`
- `attn_mqa` -> 0.249 ms
  纯GPU kernel时间: `0.862 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.249 ms`, `host_to_first_kernel_gap=1.476589`, `host_end_to_last_kernel_tail=2.091 ms`, `gpu_makespan=0.864 ms`, `gpu_kernel_sum=0.862 ms`
  开始时间(ns): `29150459`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2056, 128, 512], [2056, 1, 512], [2056, 1, 512]]}`
- `o_proj` -> 0.271 ms
  纯GPU kernel时间: `0.381 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.271 ms`, `host_to_first_kernel_gap=2.086061`, `host_end_to_last_kernel_tail=2.198 ms`, `gpu_makespan=0.383 ms`, `gpu_kernel_sum=0.381 ms`
  开始时间(ns): `29534875`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2056, 16384]]}`

## Layer 3 / decode / instance 1

- 执行序号: `4`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `1.637 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.637 ms`, `module_to_last_kernel=4.459 ms`, `host_to_first_kernel_gap=2.687571`, `host_end_to_last_kernel_tail=2.822 ms`, `gpu_makespan=1.772 ms`, `gpu_kernel_sum=1.489 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.208 ms
  纯GPU kernel时间: `0.080 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.208 ms`, `host_to_first_kernel_gap=2.62893`, `host_end_to_last_kernel_tail=2.503 ms`, `gpu_makespan=0.082 ms`, `gpu_kernel_sum=0.080 ms`
  开始时间(ns): `30699046`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2056, 7168]]}`
- `q_a_layernorm` -> 0.044 ms
  纯GPU kernel时间: `0.006 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.044 ms`, `host_to_first_kernel_gap=2.446555`, `host_end_to_last_kernel_tail=2.409 ms`, `gpu_makespan=0.006 ms`, `gpu_kernel_sum=0.006 ms`
  开始时间(ns): `30964013`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2056, 1536]]}`
- `kv_a_layernorm` -> 0.033 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.033 ms`, `host_to_first_kernel_gap=2.386322`, `host_end_to_last_kernel_tail=2.358 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `31029718`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2056, 512]]}`
- `q_b_proj` -> 0.211 ms
  纯GPU kernel时间: `0.131 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.211 ms`, `host_to_first_kernel_gap=2.328213`, `host_end_to_last_kernel_tail=2.249 ms`, `gpu_makespan=0.133 ms`, `gpu_kernel_sum=0.131 ms`
  开始时间(ns): `31094579`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2056, 1536]]}`
- `rotary_emb` -> 0.083 ms
  纯GPU kernel时间: `0.028 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.083 ms`, `host_to_first_kernel_gap=2.239598`, `host_end_to_last_kernel_tail=2.184 ms`, `gpu_makespan=0.028 ms`, `gpu_kernel_sum=0.028 ms`
  开始时间(ns): `31454010`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2056], [2056, 128, 64], [2056, 1, 64]]}`
- `attn_mqa` -> 0.250 ms
  纯GPU kernel时间: `0.853 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.250 ms`, `host_to_first_kernel_gap=2.143655`, `host_end_to_last_kernel_tail=2.748 ms`, `gpu_makespan=0.854 ms`, `gpu_kernel_sum=0.853 ms`
  开始时间(ns): `31579905`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2056, 128, 512], [2056, 1, 512], [2056, 1, 512]]}`
- `o_proj` -> 0.294 ms
  纯GPU kernel时间: `0.388 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.294 ms`, `host_to_first_kernel_gap=2.746301`, `host_end_to_last_kernel_tail=2.843 ms`, `gpu_makespan=0.390 ms`, `gpu_kernel_sum=0.388 ms`
  开始时间(ns): `31963115`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2056, 16384]]}`

## Layer 4 / decode / instance 1

- 执行序号: `5`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `1.676 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.676 ms`, `module_to_last_kernel=5.868 ms`, `host_to_first_kernel_gap=4.098986`, `host_end_to_last_kernel_tail=4.192 ms`, `gpu_makespan=1.769 ms`, `gpu_kernel_sum=1.490 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.255 ms
  纯GPU kernel时间: `0.080 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.255 ms`, `host_to_first_kernel_gap=4.03641`, `host_end_to_last_kernel_tail=3.863 ms`, `gpu_makespan=0.082 ms`, `gpu_kernel_sum=0.080 ms`
  开始时间(ns): `33713934`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2056, 7168]]}`
- `q_a_layernorm` -> 0.045 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.045 ms`, `host_to_first_kernel_gap=3.806168`, `host_end_to_last_kernel_tail=3.766 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `34026192`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2056, 1536]]}`
- `kv_a_layernorm` -> 0.033 ms
  纯GPU kernel时间: `0.004 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.033 ms`, `host_to_first_kernel_gap=3.74307`, `host_end_to_last_kernel_tail=3.714 ms`, `gpu_makespan=0.004 ms`, `gpu_kernel_sum=0.004 ms`
  开始时间(ns): `34094826`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2056, 512]]}`
- `q_b_proj` -> 0.220 ms
  纯GPU kernel时间: `0.130 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.220 ms`, `host_to_first_kernel_gap=3.680178`, `host_end_to_last_kernel_tail=3.592 ms`, `gpu_makespan=0.131 ms`, `gpu_kernel_sum=0.130 ms`
  开始时间(ns): `34163574`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2056, 1536]]}`
- `rotary_emb` -> 0.077 ms
  纯GPU kernel时间: `0.028 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.077 ms`, `host_to_first_kernel_gap=3.562689`, `host_end_to_last_kernel_tail=3.514 ms`, `gpu_makespan=0.028 ms`, `gpu_kernel_sum=0.028 ms`
  开始时间(ns): `34551015`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2056], [2056, 128, 64], [2056, 1, 64]]}`
- `attn_mqa` -> 0.252 ms
  纯GPU kernel时间: `0.864 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.252 ms`, `host_to_first_kernel_gap=3.4688`, `host_end_to_last_kernel_tail=4.083 ms`, `gpu_makespan=0.866 ms`, `gpu_kernel_sum=0.864 ms`
  开始时间(ns): `34674088`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2056, 128, 512], [2056, 1, 512], [2056, 1, 512]]}`
- `o_proj` -> 0.254 ms
  纯GPU kernel时间: `0.378 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.254 ms`, `host_to_first_kernel_gap=4.084258`, `host_end_to_last_kernel_tail=4.211 ms`, `gpu_makespan=0.381 ms`, `gpu_kernel_sum=0.378 ms`
  开始时间(ns): `35054694`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2056, 16384]]}`

## Layer 5 / decode / instance 1

- 执行序号: `6`
- Collector对齐边界: `{'Module': 'model.model.layers.5.self_attn'}`
- 整块 MLA-module 时长: `1.690 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.690 ms`, `module_to_last_kernel=7.309 ms`, `host_to_first_kernel_gap=5.556532`, `host_end_to_last_kernel_tail=5.620 ms`, `gpu_makespan=1.753 ms`, `gpu_kernel_sum=1.474 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.262 ms
  纯GPU kernel时间: `0.080 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.262 ms`, `host_to_first_kernel_gap=5.492833`, `host_end_to_last_kernel_tail=5.311 ms`, `gpu_makespan=0.081 ms`, `gpu_kernel_sum=0.080 ms`
  开始时间(ns): `36654342`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2056, 7168]]}`
- `q_a_layernorm` -> 0.047 ms
  纯GPU kernel时间: `0.006 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.047 ms`, `host_to_first_kernel_gap=5.252292`, `host_end_to_last_kernel_tail=5.211 ms`, `gpu_makespan=0.006 ms`, `gpu_kernel_sum=0.006 ms`
  开始时间(ns): `36975587`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2056, 1536]]}`
- `kv_a_layernorm` -> 0.031 ms
  纯GPU kernel时间: `0.004 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.031 ms`, `host_to_first_kernel_gap=5.185571`, `host_end_to_last_kernel_tail=5.159 ms`, `gpu_makespan=0.004 ms`, `gpu_kernel_sum=0.004 ms`
  开始时间(ns): `37048132`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2056, 512]]}`
- `q_b_proj` -> 0.221 ms
  纯GPU kernel时间: `0.130 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.221 ms`, `host_to_first_kernel_gap=5.121788`, `host_end_to_last_kernel_tail=5.032 ms`, `gpu_makespan=0.131 ms`, `gpu_kernel_sum=0.130 ms`
  开始时间(ns): `37117579`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2056, 1536]]}`
- `rotary_emb` -> 0.093 ms
  纯GPU kernel时间: `0.028 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.093 ms`, `host_to_first_kernel_gap=5.014117`, `host_end_to_last_kernel_tail=4.949 ms`, `gpu_makespan=0.028 ms`, `gpu_kernel_sum=0.028 ms`
  开始时间(ns): `37495170`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2056], [2056, 128, 64], [2056, 1, 64]]}`
- `attn_mqa` -> 0.248 ms
  纯GPU kernel时间: `0.851 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.248 ms`, `host_to_first_kernel_gap=4.904635`, `host_end_to_last_kernel_tail=5.509 ms`, `gpu_makespan=0.853 ms`, `gpu_kernel_sum=0.851 ms`
  开始时间(ns): `37633612`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2056, 128, 512], [2056, 1, 512], [2056, 1, 512]]}`
- `o_proj` -> 0.252 ms
  纯GPU kernel时间: `0.375 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.252 ms`, `host_to_first_kernel_gap=5.513351`, `host_end_to_last_kernel_tail=5.638 ms`, `gpu_makespan=0.376 ms`, `gpu_kernel_sum=0.375 ms`
  开始时间(ns): `38010048`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2056, 16384]]}`
