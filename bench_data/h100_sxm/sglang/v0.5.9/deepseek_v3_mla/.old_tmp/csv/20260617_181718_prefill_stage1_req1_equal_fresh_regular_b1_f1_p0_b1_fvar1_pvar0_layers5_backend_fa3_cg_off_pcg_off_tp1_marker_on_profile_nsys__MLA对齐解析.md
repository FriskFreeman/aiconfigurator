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
- 整块 MLA-module 时长: `42408.725 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `2`
- 模块时延拆解: `host_total=42408.725 ms`, `module_to_last_kernel=42408.725 ms`, `host_to_first_kernel_gap=1.225698`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=42407.198 ms`, `gpu_kernel_sum=0.118 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 4819.291 ms
  纯GPU kernel时间: `0.017 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=4819.291 ms`, `host_to_first_kernel_gap=0.95309`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=4818.003 ms`, `gpu_kernel_sum=0.017 ms`
  开始时间(ns): `4636981711`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.262 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.262 ms`, `host_to_first_kernel_gap=0.236628`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `9456586306`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `q_b_proj` -> 6738.488 ms
  纯GPU kernel时间: `0.024 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=6738.488 ms`, `host_to_first_kernel_gap=0.488055`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=6737.761 ms`, `gpu_kernel_sum=0.024 ms`
  开始时间(ns): `9457012862`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.247 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.247 ms`, `host_to_first_kernel_gap=0.221618`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `16195822582`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `rotary_emb` -> 19770.446 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=19770.446 ms`, `host_to_first_kernel_gap=19770.250827`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `16196238585`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `kv_b_proj` -> 4662.092 ms
  纯GPU kernel时间: `0.011 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=4662.092 ms`, `host_to_first_kernel_gap=0.333837`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=4661.493 ms`, `gpu_kernel_sum=0.011 ms`
  开始时间(ns): `37255896381`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[1, 512]]}`
- `attn_mha` -> 18.777 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=18.777 ms`, `host_to_first_kernel_gap=3.000285`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=15.665 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `41919615719`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 192], [1, 128, 192], [1, 128, 128]]}`
- `o_proj` -> 5106.714 ms
  纯GPU kernel时间: `0.047 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=5106.714 ms`, `host_to_first_kernel_gap=0.476875`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=5105.999 ms`, `gpu_kernel_sum=0.047 ms`
  开始时间(ns): `41938657002`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 1 / decode / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `5.736 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `2`
- 模块时延拆解: `host_total=5.736 ms`, `module_to_last_kernel=5.736 ms`, `host_to_first_kernel_gap=0.945181`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=4.734 ms`, `gpu_kernel_sum=0.108 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.951 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.951 ms`, `host_to_first_kernel_gap=0.371109`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.520 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `57417535543`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.135 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.135 ms`, `host_to_first_kernel_gap=0.113793`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `57418652634`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `q_b_proj` -> 0.640 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.640 ms`, `host_to_first_kernel_gap=0.229446`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.365 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `57418877525`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.110 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.110 ms`, `host_to_first_kernel_gap=0.09186`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `57419696967`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `rotary_emb` -> 0.270 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.270 ms`, `host_to_first_kernel_gap=0.220895`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `57419918811`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `kv_b_proj` -> 0.631 ms
  纯GPU kernel时间: `0.010 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.631 ms`, `host_to_first_kernel_gap=0.230065`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.334 ms`, `gpu_kernel_sum=0.010 ms`
  开始时间(ns): `57420599113`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[1, 512]]}`
- `attn_mha` -> 0.416 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.416 ms`, `host_to_first_kernel_gap=0.306904`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.067 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `57421500961`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 192], [1, 128, 192], [1, 128, 128]]}`
- `o_proj` -> 0.631 ms
  纯GPU kernel时间: `0.046 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.631 ms`, `host_to_first_kernel_gap=0.237084`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.373 ms`, `gpu_kernel_sum=0.046 ms`
  开始时间(ns): `57422030877`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 2 / decode / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `4.644 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `2`
- 模块时延拆解: `host_total=4.644 ms`, `module_to_last_kernel=4.644 ms`, `host_to_first_kernel_gap=0.537325`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=4.044 ms`, `gpu_kernel_sum=0.109 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.583 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.583 ms`, `host_to_first_kernel_gap=0.222224`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.316 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `57425322726`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.132 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.132 ms`, `host_to_first_kernel_gap=0.099784`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `57426041390`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `q_b_proj` -> 0.525 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.525 ms`, `host_to_first_kernel_gap=0.193623`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.293 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `57426253470`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.103 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.103 ms`, `host_to_first_kernel_gap=0.085278`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `57426937335`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `rotary_emb` -> 0.248 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.248 ms`, `host_to_first_kernel_gap=0.215666`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `57427131779`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `kv_b_proj` -> 0.577 ms
  纯GPU kernel时间: `0.010 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.577 ms`, `host_to_first_kernel_gap=0.207356`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.323 ms`, `gpu_kernel_sum=0.010 ms`
  开始时间(ns): `57427722264`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[1, 512]]}`
- `attn_mha` -> 0.341 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.341 ms`, `host_to_first_kernel_gap=0.245627`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.055 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `57428531193`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 192], [1, 128, 192], [1, 128, 128]]}`
- `o_proj` -> 0.634 ms
  纯GPU kernel时间: `0.047 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.634 ms`, `host_to_first_kernel_gap=0.22986`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.378 ms`, `gpu_kernel_sum=0.047 ms`
  开始时间(ns): `57428980623`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 3 / decode / instance 1

- 执行序号: `4`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `4.510 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `2`
- 模块时延拆解: `host_total=4.510 ms`, `module_to_last_kernel=4.510 ms`, `host_to_first_kernel_gap=0.501847`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=3.957 ms`, `gpu_kernel_sum=0.109 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.563 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.563 ms`, `host_to_first_kernel_gap=0.212961`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.308 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `57432088912`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.110 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.110 ms`, `host_to_first_kernel_gap=0.091084`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `57432784804`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `q_b_proj` -> 0.539 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.539 ms`, `host_to_first_kernel_gap=0.192063`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.307 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `57432967985`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.103 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.103 ms`, `host_to_first_kernel_gap=0.080883`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `57433668701`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `rotary_emb` -> 0.184 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.184 ms`, `host_to_first_kernel_gap=0.151937`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `57433870926`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `kv_b_proj` -> 0.597 ms
  纯GPU kernel时间: `0.011 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.597 ms`, `host_to_first_kernel_gap=0.218653`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.326 ms`, `gpu_kernel_sum=0.011 ms`
  开始时间(ns): `57434398482`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[1, 512]]}`
- `attn_mha` -> 0.329 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.329 ms`, `host_to_first_kernel_gap=0.236743`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.055 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `57435249927`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 192], [1, 128, 192], [1, 128, 128]]}`
- `o_proj` -> 0.592 ms
  纯GPU kernel时间: `0.046 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.592 ms`, `host_to_first_kernel_gap=0.206514`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.368 ms`, `gpu_kernel_sum=0.046 ms`
  开始时间(ns): `57435684092`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 4 / decode / instance 1

- 执行序号: `5`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `2.425 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `2`
- 模块时延拆解: `host_total=2.425 ms`, `module_to_last_kernel=2.432 ms`, `host_to_first_kernel_gap=0.408648`, `host_end_to_last_kernel_tail=0.006 ms`, `gpu_makespan=2.023 ms`, `gpu_kernel_sum=0.117 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.467 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.467 ms`, `host_to_first_kernel_gap=0.173971`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.268 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `62694194573`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.060 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.060 ms`, `host_to_first_kernel_gap=0.052422`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `62694732249`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `q_b_proj` -> 0.248 ms
  纯GPU kernel时间: `0.023 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.248 ms`, `host_to_first_kernel_gap=0.093644`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.148 ms`, `gpu_kernel_sum=0.023 ms`
  开始时间(ns): `62694828723`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.050 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.050 ms`, `host_to_first_kernel_gap=0.045172`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `62695144363`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `rotary_emb` -> 0.113 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.113 ms`, `host_to_first_kernel_gap=0.096499`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `62695242380`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `kv_b_proj` -> 0.239 ms
  纯GPU kernel时间: `0.012 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.239 ms`, `host_to_first_kernel_gap=0.089236`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.138 ms`, `gpu_kernel_sum=0.012 ms`
  开始时间(ns): `62695523723`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[1, 512]]}`
- `attn_mha` -> 0.191 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.191 ms`, `host_to_first_kernel_gap=0.145744`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.035 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `62695871054`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 192], [1, 128, 192], [1, 128, 128]]}`
- `o_proj` -> 0.259 ms
  纯GPU kernel时间: `0.047 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.259 ms`, `host_to_first_kernel_gap=0.108323`, `host_end_to_last_kernel_tail=0.021 ms`, `gpu_makespan=0.172 ms`, `gpu_kernel_sum=0.047 ms`
  开始时间(ns): `62696110939`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 0 / decode / instance 2

- 执行序号: `6`
- Collector对齐边界: `{'Module': 'model.model.layers.0.self_attn'}`
- 整块 MLA-module 时长: `258.708 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `2`
- 模块时延拆解: `host_total=258.708 ms`, `module_to_last_kernel=258.708 ms`, `host_to_first_kernel_gap=0.244523`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=258.439 ms`, `gpu_kernel_sum=0.098 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.409 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.409 ms`, `host_to_first_kernel_gap=0.141787`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.248 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `63852308868`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.048 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.048 ms`, `host_to_first_kernel_gap=0.042596`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `63852799834`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.033 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.033 ms`, `host_to_first_kernel_gap=0.028896`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `63852876574`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.263 ms
  纯GPU kernel时间: `0.020 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.263 ms`, `host_to_first_kernel_gap=0.096254`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.160 ms`, `gpu_kernel_sum=0.020 ms`
  开始时间(ns): `63852956864`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.148 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.148 ms`, `host_to_first_kernel_gap=0.127462`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `63856882453`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 249.781 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=249.781 ms`, `host_to_first_kernel_gap=240.86641`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=8.815 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `63857107616`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.526 ms
  纯GPU kernel时间: `0.046 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.526 ms`, `host_to_first_kernel_gap=0.213655`, `host_end_to_last_kernel_tail=0.004 ms`, `gpu_makespan=0.317 ms`, `gpu_kernel_sum=0.046 ms`
  开始时间(ns): `64110359782`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 1 / decode / instance 2

- 执行序号: `7`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `1.895 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `2`
- 模块时延拆解: `host_total=1.895 ms`, `module_to_last_kernel=1.898 ms`, `host_to_first_kernel_gap=0.177479`, `host_end_to_last_kernel_tail=0.003 ms`, `gpu_makespan=1.721 ms`, `gpu_kernel_sum=0.100 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.258 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.258 ms`, `host_to_first_kernel_gap=0.086424`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.162 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `64112002947`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.057 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.057 ms`, `host_to_first_kernel_gap=0.050673`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `64112334634`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.031 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.031 ms`, `host_to_first_kernel_gap=0.026907`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `64112417408`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.225 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.225 ms`, `host_to_first_kernel_gap=0.081295`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.138 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `64112485772`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.114 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.114 ms`, `host_to_first_kernel_gap=0.097991`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `64112893139`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.322 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.322 ms`, `host_to_first_kernel_gap=0.125527`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.181 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `64113062371`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.272 ms
  纯GPU kernel时间: `0.048 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.272 ms`, `host_to_first_kernel_gap=0.100197`, `host_end_to_last_kernel_tail=0.022 ms`, `gpu_makespan=0.193 ms`, `gpu_kernel_sum=0.048 ms`
  开始时间(ns): `64113516821`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 2 / decode / instance 2

- 执行序号: `8`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.660 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `2`
- 模块时延拆解: `host_total=1.660 ms`, `module_to_last_kernel=1.665 ms`, `host_to_first_kernel_gap=0.164043`, `host_end_to_last_kernel_tail=0.006 ms`, `gpu_makespan=1.501 ms`, `gpu_kernel_sum=0.099 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.247 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.247 ms`, `host_to_first_kernel_gap=0.095037`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.141 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `64114753531`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.045 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.045 ms`, `host_to_first_kernel_gap=0.039362`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `64115060822`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.030 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.030 ms`, `host_to_first_kernel_gap=0.026011`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `64115130621`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.222 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.222 ms`, `host_to_first_kernel_gap=0.078259`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.138 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `64115205701`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.078 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.078 ms`, `host_to_first_kernel_gap=0.066127`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `64115568553`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.252 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.252 ms`, `host_to_first_kernel_gap=0.091053`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.147 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `64115692970`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.262 ms
  纯GPU kernel时间: `0.047 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.262 ms`, `host_to_first_kernel_gap=0.088618`, `host_end_to_last_kernel_tail=0.022 ms`, `gpu_makespan=0.195 ms`, `gpu_kernel_sum=0.047 ms`
  开始时间(ns): `64116066061`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 3 / decode / instance 2

- 执行序号: `9`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `1.603 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `2`
- 模块时延拆解: `host_total=1.603 ms`, `module_to_last_kernel=1.609 ms`, `host_to_first_kernel_gap=0.151669`, `host_end_to_last_kernel_tail=0.006 ms`, `gpu_makespan=1.457 ms`, `gpu_kernel_sum=0.101 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.207 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.207 ms`, `host_to_first_kernel_gap=0.077729`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.123 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `64117205429`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.044 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.044 ms`, `host_to_first_kernel_gap=0.038729`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `64117470317`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.034 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.034 ms`, `host_to_first_kernel_gap=0.026165`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `64117538529`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.219 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.219 ms`, `host_to_first_kernel_gap=0.078812`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.137 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `64117606105`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.074 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.074 ms`, `host_to_first_kernel_gap=0.06304`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `64117963157`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.266 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.266 ms`, `host_to_first_kernel_gap=0.101191`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.151 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `64118081486`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.248 ms
  纯GPU kernel时间: `0.048 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.248 ms`, `host_to_first_kernel_gap=0.089915`, `host_end_to_last_kernel_tail=0.024 ms`, `gpu_makespan=0.182 ms`, `gpu_kernel_sum=0.048 ms`
  开始时间(ns): `64118468889`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 4 / decode / instance 2

- 执行序号: `10`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `1.710 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `2`
- 模块时延拆解: `host_total=1.710 ms`, `module_to_last_kernel=1.712 ms`, `host_to_first_kernel_gap=0.167086`, `host_end_to_last_kernel_tail=0.002 ms`, `gpu_makespan=1.545 ms`, `gpu_kernel_sum=0.099 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.271 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.271 ms`, `host_to_first_kernel_gap=0.094269`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.164 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `64120436053`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.046 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.046 ms`, `host_to_first_kernel_gap=0.040721`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `64120768481`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.035 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.035 ms`, `host_to_first_kernel_gap=0.025638`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `64120839852`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.224 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.224 ms`, `host_to_first_kernel_gap=0.081084`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.138 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `64120910518`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.077 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.077 ms`, `host_to_first_kernel_gap=0.065667`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `64121292303`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.252 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.252 ms`, `host_to_first_kernel_gap=0.086677`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.150 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `64121416220`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.264 ms
  纯GPU kernel时间: `0.047 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.264 ms`, `host_to_first_kernel_gap=0.095933`, `host_end_to_last_kernel_tail=0.020 ms`, `gpu_makespan=0.188 ms`, `gpu_kernel_sum=0.047 ms`
  开始时间(ns): `64121791092`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`
