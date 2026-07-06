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
- 整块 MLA-module 时长: `1.676 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.676 ms`, `module_to_last_kernel=1.688 ms`, `host_to_first_kernel_gap=0.189168`, `host_end_to_last_kernel_tail=0.012 ms`, `gpu_makespan=1.499 ms`, `gpu_kernel_sum=0.107 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.352 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.352 ms`, `host_to_first_kernel_gap=0.11321`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.227 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `14437636`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[3, 7168]]}`
- `q_a_layernorm` -> 0.037 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.037 ms`, `host_to_first_kernel_gap=0.033782`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `14840903`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[3, 1536]]}`
- `kv_a_layernorm` -> 0.022 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.022 ms`, `host_to_first_kernel_gap=0.020305`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `14897772`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[3, 512]]}`
- `q_b_proj` -> 0.169 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.169 ms`, `host_to_first_kernel_gap=0.062316`, `host_end_to_last_kernel_tail=0.001 ms`, `gpu_makespan=0.107 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `14949105`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[3, 1536]]}`
- `rotary_emb` -> 0.095 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.095 ms`, `host_to_first_kernel_gap=0.08214`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `15307553`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[3], [3, 128, 64], [3, 1, 64]]}`
- `attn_mqa` -> 0.268 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.268 ms`, `host_to_first_kernel_gap=0.090927`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.171 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `15440654`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[3, 128, 512], [3, 1, 512], [3, 1, 512]]}`
- `o_proj` -> 0.208 ms
  纯GPU kernel时间: `0.048 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.208 ms`, `host_to_first_kernel_gap=0.073266`, `host_end_to_last_kernel_tail=0.028 ms`, `gpu_makespan=0.163 ms`, `gpu_kernel_sum=0.048 ms`
  开始时间(ns): `15813322`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[3, 16384]]}`

## Layer 1 / decode / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `1.167 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.167 ms`, `module_to_last_kernel=1.183 ms`, `host_to_first_kernel_gap=0.10747`, `host_end_to_last_kernel_tail=0.016 ms`, `gpu_makespan=1.076 ms`, `gpu_kernel_sum=0.105 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.148 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.148 ms`, `host_to_first_kernel_gap=0.055928`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.092 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `16725476`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[3, 7168]]}`
- `q_a_layernorm` -> 0.034 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.034 ms`, `host_to_first_kernel_gap=0.030948`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `16914455`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[3, 1536]]}`
- `kv_a_layernorm` -> 0.024 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.024 ms`, `host_to_first_kernel_gap=0.022177`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `16965722`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[3, 512]]}`
- `q_b_proj` -> 0.142 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.142 ms`, `host_to_first_kernel_gap=0.052211`, `host_end_to_last_kernel_tail=0.004 ms`, `gpu_makespan=0.094 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `17013928`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[3, 1536]]}`
- `rotary_emb` -> 0.066 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.066 ms`, `host_to_first_kernel_gap=0.058829`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `17260526`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[3], [3, 128, 64], [3, 1, 64]]}`
- `attn_mqa` -> 0.190 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.190 ms`, `host_to_first_kernel_gap=0.069698`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.119 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `17358425`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[3, 128, 512], [3, 1, 512], [3, 1, 512]]}`
- `o_proj` -> 0.187 ms
  纯GPU kernel时间: `0.048 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.187 ms`, `host_to_first_kernel_gap=0.065504`, `host_end_to_last_kernel_tail=0.030 ms`, `gpu_makespan=0.151 ms`, `gpu_kernel_sum=0.048 ms`
  开始时间(ns): `17640283`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[3, 16384]]}`

## Layer 2 / decode / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.138 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.138 ms`, `module_to_last_kernel=1.154 ms`, `host_to_first_kernel_gap=0.10285`, `host_end_to_last_kernel_tail=0.016 ms`, `gpu_makespan=1.051 ms`, `gpu_kernel_sum=0.103 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.148 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.148 ms`, `host_to_first_kernel_gap=0.056394`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.092 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `18486992`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[3, 7168]]}`
- `q_a_layernorm` -> 0.032 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.032 ms`, `host_to_first_kernel_gap=0.02946`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `18675206`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[3, 1536]]}`
- `kv_a_layernorm` -> 0.021 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.021 ms`, `host_to_first_kernel_gap=0.019619`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `18724983`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[3, 512]]}`
- `q_b_proj` -> 0.139 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.139 ms`, `host_to_first_kernel_gap=0.052107`, `host_end_to_last_kernel_tail=0.003 ms`, `gpu_makespan=0.090 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `18773423`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[3, 1536]]}`
- `rotary_emb` -> 0.054 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.054 ms`, `host_to_first_kernel_gap=0.047399`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `19011058`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[3], [3, 128, 64], [3, 1, 64]]}`
- `attn_mqa` -> 0.202 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.202 ms`, `host_to_first_kernel_gap=0.076071`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.117 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `19096338`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[3, 128, 512], [3, 1, 512], [3, 1, 512]]}`
- `o_proj` -> 0.180 ms
  纯GPU kernel时间: `0.047 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.180 ms`, `host_to_first_kernel_gap=0.063107`, `host_end_to_last_kernel_tail=0.029 ms`, `gpu_makespan=0.146 ms`, `gpu_kernel_sum=0.047 ms`
  开始时间(ns): `19385494`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[3, 16384]]}`
