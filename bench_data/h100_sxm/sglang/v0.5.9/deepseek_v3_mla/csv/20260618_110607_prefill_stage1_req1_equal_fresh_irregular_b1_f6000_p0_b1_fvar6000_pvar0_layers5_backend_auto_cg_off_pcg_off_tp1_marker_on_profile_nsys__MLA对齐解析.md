# MLA对齐解析摘要

源文件: `report.sqlite`

## 对齐原则

- `collector/sglang/collect_mla_module.py` 的 MLA module 计时边界是 `model.model.layers[test_layer].self_attn(...)`。
- 因此这里把 `nsys` 中每层的 `model.model.layers.X.self_attn` NVTX range 视为与 collector 对齐的 MLA-module 边界。
- 该区间内部的 `.self_attn.*` 子模块用于做 MLA 内部 breakdown。

## 运行摘要

- `prefill` 对齐成功层: `[0, 1, 2, 3, 4]`
- `prefill` 被切分层: `[]`
- 若某层 `prefill` 出现多个 `self_attn` 实例，则说明 Engine 调度把一次前向切成了多块，已不再与 collector 的单次 MLA-module 采集严格一一对应。

## Layer 0 / prefill / instance 1

- 执行序号: `1`
- Collector对齐边界: `{'Module': 'model.model.layers.0.self_attn'}`
- 整块 MLA-module 时长: `288.167 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=288.167 ms`, `module_to_last_kernel=290.374 ms`, `host_to_first_kernel_gap=0.244159`, `host_end_to_last_kernel_tail=2.207 ms`, `gpu_makespan=290.129 ms`, `gpu_kernel_sum=3.833 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=6000`, `total_tokens=6000`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[6000, 128, 192], [6000, 128, 192], [6000, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=6000`, `sum_prefix=0`, `sum_seq_after=6000`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 6000, 'prefix_len': 0, 'seq_len_after': 6000, 'prompt_len': 6000, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.484 ms
  纯GPU kernel时间: `0.189 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.484 ms`, `host_to_first_kernel_gap=0.109835`, `host_end_to_last_kernel_tail=0.048 ms`, `gpu_makespan=0.422 ms`, `gpu_kernel_sum=0.189 ms`
  开始时间(ns): `1298286443`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[6000, 7168]]}`
- `q_a_layernorm` -> 0.144 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.144 ms`, `host_to_first_kernel_gap=0.124355`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `1298916627`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[6000, 1536]]}`
- `q_b_proj` -> 0.657 ms
  纯GPU kernel时间: `0.340 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.657 ms`, `host_to_first_kernel_gap=0.221956`, `host_end_to_last_kernel_tail=0.254 ms`, `gpu_makespan=0.689 ms`, `gpu_kernel_sum=0.340 ms`
  开始时间(ns): `1299152594`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[6000, 1536]]}`
- `kv_a_layernorm` -> 0.112 ms
  纯GPU kernel时间: `0.011 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.112 ms`, `host_to_first_kernel_gap=0.093671`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.011 ms`, `gpu_kernel_sum=0.011 ms`
  开始时间(ns): `1299994255`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[6000, 512]]}`
- `rotary_emb` -> 0.228 ms
  纯GPU kernel时间: `0.078 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.228 ms`, `host_to_first_kernel_gap=0.191482`, `host_end_to_last_kernel_tail=0.041 ms`, `gpu_makespan=0.078 ms`, `gpu_kernel_sum=0.078 ms`
  开始时间(ns): `1300215516`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[6000], [6000, 128, 64], [6000, 1, 64]]}`
- `kv_b_proj` -> 1.016 ms
  纯GPU kernel时间: `0.213 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=1.016 ms`, `host_to_first_kernel_gap=0.458963`, `host_end_to_last_kernel_tail=0.121 ms`, `gpu_makespan=0.678 ms`, `gpu_kernel_sum=0.213 ms`
  开始时间(ns): `1583887006`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[6000, 512]]}`
- `attn_mha` -> 0.379 ms
  纯GPU kernel时间: `2.005 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.379 ms`, `host_to_first_kernel_gap=0.278725`, `host_end_to_last_kernel_tail=1.950 ms`, `gpu_makespan=2.050 ms`, `gpu_kernel_sum=2.005 ms`
  开始时间(ns): `1585209963`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[6000, 128, 192], [6000, 128, 192], [6000, 128, 128]]}`
- `o_proj` -> 0.585 ms
  纯GPU kernel时间: `0.984 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.585 ms`, `host_to_first_kernel_gap=1.839661`, `host_end_to_last_kernel_tail=2.240 ms`, `gpu_makespan=0.985 ms`, `gpu_kernel_sum=0.984 ms`
  开始时间(ns): `1585700803`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[6000, 16384]]}`

## Layer 1 / prefill / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `4.076 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=4.076 ms`, `module_to_last_kernel=7.767 ms`, `host_to_first_kernel_gap=3.73486`, `host_end_to_last_kernel_tail=3.692 ms`, `gpu_makespan=4.033 ms`, `gpu_kernel_sum=3.822 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=6000`, `total_tokens=6000`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[6000, 128, 192], [6000, 128, 192], [6000, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=6000`, `sum_prefix=0`, `sum_seq_after=6000`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 6000, 'prefix_len': 0, 'seq_len_after': 6000, 'prompt_len': 6000, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.548 ms
  纯GPU kernel时间: `0.188 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.548 ms`, `host_to_first_kernel_gap=3.440115`, `host_end_to_last_kernel_tail=3.081 ms`, `gpu_makespan=0.189 ms`, `gpu_kernel_sum=0.188 ms`
  开始时间(ns): `1588603133`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[6000, 7168]]}`
- `q_a_layernorm` -> 0.114 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.114 ms`, `host_to_first_kernel_gap=2.955964`, `host_end_to_last_kernel_tail=2.855 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `1589276756`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[6000, 1536]]}`
- `q_b_proj` -> 0.450 ms
  纯GPU kernel时间: `0.335 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.450 ms`, `host_to_first_kernel_gap=2.792031`, `host_end_to_last_kernel_tail=2.679 ms`, `gpu_makespan=0.337 ms`, `gpu_kernel_sum=0.335 ms`
  开始时间(ns): `1589455153`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[6000, 1536]]}`
- `kv_a_layernorm` -> 0.092 ms
  纯GPU kernel时间: `0.010 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.092 ms`, `host_to_first_kernel_gap=2.45826`, `host_end_to_last_kernel_tail=2.377 ms`, `gpu_makespan=0.010 ms`, `gpu_kernel_sum=0.010 ms`
  开始时间(ns): `1590125852`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[6000, 512]]}`
- `rotary_emb` -> 0.190 ms
  纯GPU kernel时间: `0.077 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.190 ms`, `host_to_first_kernel_gap=2.28286`, `host_end_to_last_kernel_tail=2.171 ms`, `gpu_makespan=0.077 ms`, `gpu_kernel_sum=0.077 ms`
  开始时间(ns): `1590312707`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[6000], [6000, 128, 64], [6000, 1, 64]]}`
- `kv_b_proj` -> 0.480 ms
  纯GPU kernel时间: `0.207 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.480 ms`, `host_to_first_kernel_gap=1.892428`, `host_end_to_last_kernel_tail=1.621 ms`, `gpu_makespan=0.209 ms`, `gpu_kernel_sum=0.207 ms`
  开始时间(ns): `1590802787`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[6000, 512]]}`
- `attn_mha` -> 0.320 ms
  纯GPU kernel时间: `2.007 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.320 ms`, `host_to_first_kernel_gap=1.601421`, `host_end_to_last_kernel_tail=3.288 ms`, `gpu_makespan=2.007 ms`, `gpu_kernel_sum=2.007 ms`
  开始时间(ns): `1591480162`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[6000, 128, 192], [6000, 128, 192], [6000, 128, 128]]}`
- `o_proj` -> 0.471 ms
  纯GPU kernel时间: `0.984 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.471 ms`, `host_to_first_kernel_gap=3.205975`, `host_end_to_last_kernel_tail=3.720 ms`, `gpu_makespan=0.985 ms`, `gpu_kernel_sum=0.984 ms`
  开始时间(ns): `1591885016`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[6000, 16384]]}`

## Layer 2 / prefill / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `3.650 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=3.650 ms`, `module_to_last_kernel=9.487 ms`, `host_to_first_kernel_gap=5.448873`, `host_end_to_last_kernel_tail=5.837 ms`, `gpu_makespan=4.038 ms`, `gpu_kernel_sum=3.822 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=6000`, `total_tokens=6000`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[6000, 128, 192], [6000, 128, 192], [6000, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=6000`, `sum_prefix=0`, `sum_seq_after=6000`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 6000, 'prefix_len': 0, 'seq_len_after': 6000, 'prompt_len': 6000, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.466 ms
  纯GPU kernel时间: `0.187 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.466 ms`, `host_to_first_kernel_gap=5.21674`, `host_end_to_last_kernel_tail=4.940 ms`, `gpu_makespan=0.189 ms`, `gpu_kernel_sum=0.187 ms`
  开始时间(ns): `1594381547`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[6000, 7168]]}`
- `q_a_layernorm` -> 0.110 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.110 ms`, `host_to_first_kernel_gap=4.835134`, `host_end_to_last_kernel_tail=4.738 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `1594952305`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[6000, 1536]]}`
- `q_b_proj` -> 0.430 ms
  纯GPU kernel时间: `0.330 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.430 ms`, `host_to_first_kernel_gap=4.675964`, `host_end_to_last_kernel_tail=4.577 ms`, `gpu_makespan=0.331 ms`, `gpu_kernel_sum=0.330 ms`
  开始时间(ns): `1595126931`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[6000, 1536]]}`
- `kv_a_layernorm` -> 0.090 ms
  纯GPU kernel时间: `0.010 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.090 ms`, `host_to_first_kernel_gap=4.451368`, `host_end_to_last_kernel_tail=4.372 ms`, `gpu_makespan=0.010 ms`, `gpu_kernel_sum=0.010 ms`
  开始时间(ns): `1595682759`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[6000, 512]]}`
- `rotary_emb` -> 0.159 ms
  纯GPU kernel时间: `0.078 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.159 ms`, `host_to_first_kernel_gap=4.294602`, `host_end_to_last_kernel_tail=4.214 ms`, `gpu_makespan=0.078 ms`, `gpu_kernel_sum=0.078 ms`
  开始时间(ns): `1595852325`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[6000], [6000, 128, 64], [6000, 1, 64]]}`
- `kv_b_proj` -> 0.440 ms
  纯GPU kernel时间: `0.215 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.440 ms`, `host_to_first_kernel_gap=3.969643`, `host_end_to_last_kernel_tail=3.746 ms`, `gpu_makespan=0.216 ms`, `gpu_kernel_sum=0.215 ms`
  开始时间(ns): `1596277923`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[6000, 512]]}`
- `attn_mha` -> 0.328 ms
  纯GPU kernel时间: `2.006 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.328 ms`, `host_to_first_kernel_gap=3.728879`, `host_end_to_last_kernel_tail=5.407 ms`, `gpu_makespan=2.006 ms`, `gpu_kernel_sum=2.006 ms`
  开始时间(ns): `1596915327`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[6000, 128, 192], [6000, 128, 192], [6000, 128, 128]]}`
- `o_proj` -> 0.449 ms
  纯GPU kernel时间: `0.982 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.449 ms`, `host_to_first_kernel_gap=5.329049`, `host_end_to_last_kernel_tail=5.864 ms`, `gpu_makespan=0.984 ms`, `gpu_kernel_sum=0.982 ms`
  开始时间(ns): `1597322773`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[6000, 16384]]}`

## Layer 3 / prefill / instance 1

- 执行序号: `4`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `3.642 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=3.642 ms`, `module_to_last_kernel=11.693 ms`, `host_to_first_kernel_gap=7.661871`, `host_end_to_last_kernel_tail=8.051 ms`, `gpu_makespan=4.031 ms`, `gpu_kernel_sum=3.818 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=6000`, `total_tokens=6000`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[6000, 128, 192], [6000, 128, 192], [6000, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=6000`, `sum_prefix=0`, `sum_seq_after=6000`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 6000, 'prefix_len': 0, 'seq_len_after': 6000, 'prompt_len': 6000, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.461 ms
  纯GPU kernel时间: `0.188 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.461 ms`, `host_to_first_kernel_gap=7.446464`, `host_end_to_last_kernel_tail=7.176 ms`, `gpu_makespan=0.190 ms`, `gpu_kernel_sum=0.188 ms`
  开始时间(ns): `1599718254`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[6000, 7168]]}`
- `q_a_layernorm` -> 0.092 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.092 ms`, `host_to_first_kernel_gap=7.078381`, `host_end_to_last_kernel_tail=6.999 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `1600277217`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[6000, 1536]]}`
- `q_b_proj` -> 0.412 ms
  纯GPU kernel时间: `0.335 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.412 ms`, `host_to_first_kernel_gap=6.939263`, `host_end_to_last_kernel_tail=6.864 ms`, `gpu_makespan=0.337 ms`, `gpu_kernel_sum=0.335 ms`
  开始时间(ns): `1600430639`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[6000, 1536]]}`
- `kv_a_layernorm` -> 0.086 ms
  纯GPU kernel时间: `0.011 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.086 ms`, `host_to_first_kernel_gap=6.720347`, `host_end_to_last_kernel_tail=6.645 ms`, `gpu_makespan=0.011 ms`, `gpu_kernel_sum=0.011 ms`
  开始时间(ns): `1600986291`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[6000, 512]]}`
- `rotary_emb` -> 0.148 ms
  纯GPU kernel时间: `0.077 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.148 ms`, `host_to_first_kernel_gap=6.568656`, `host_end_to_last_kernel_tail=6.498 ms`, `gpu_makespan=0.077 ms`, `gpu_kernel_sum=0.077 ms`
  开始时间(ns): `1601149822`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[6000], [6000, 128, 64], [6000, 1, 64]]}`
- `kv_b_proj` -> 0.504 ms
  纯GPU kernel时间: `0.205 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.504 ms`, `host_to_first_kernel_gap=6.245988`, `host_end_to_last_kernel_tail=5.949 ms`, `gpu_makespan=0.207 ms`, `gpu_kernel_sum=0.205 ms`
  开始时间(ns): `1601572393`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[6000, 512]]}`
- `attn_mha` -> 0.282 ms
  纯GPU kernel时间: `2.008 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.282 ms`, `host_to_first_kernel_gap=5.924885`, `host_end_to_last_kernel_tail=7.651 ms`, `gpu_makespan=2.008 ms`, `gpu_kernel_sum=2.008 ms`
  开始时间(ns): `1602277592`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[6000, 128, 192], [6000, 128, 192], [6000, 128, 128]]}`
- `o_proj` -> 0.475 ms
  纯GPU kernel时间: `0.982 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.475 ms`, `host_to_first_kernel_gap=7.572459`, `host_end_to_last_kernel_tail=8.081 ms`, `gpu_makespan=0.983 ms`, `gpu_kernel_sum=0.982 ms`
  开始时间(ns): `1602640610`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[6000, 16384]]}`

## Layer 4 / prefill / instance 1

- 执行序号: `5`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `3.724 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=3.724 ms`, `module_to_last_kernel=16.941 ms`, `host_to_first_kernel_gap=12.899325`, `host_end_to_last_kernel_tail=13.217 ms`, `gpu_makespan=4.042 ms`, `gpu_kernel_sum=3.829 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=6000`, `total_tokens=6000`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[6000, 128, 192], [6000, 128, 192], [6000, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=6000`, `sum_prefix=0`, `sum_seq_after=6000`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 6000, 'prefix_len': 0, 'seq_len_after': 6000, 'prompt_len': 6000, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.519 ms
  纯GPU kernel时间: `0.187 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.519 ms`, `host_to_first_kernel_gap=12.667908`, `host_end_to_last_kernel_tail=12.339 ms`, `gpu_makespan=0.190 ms`, `gpu_kernel_sum=0.187 ms`
  开始时间(ns): `1606262856`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[6000, 7168]]}`
- `q_a_layernorm` -> 0.104 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.104 ms`, `host_to_first_kernel_gap=12.235803`, `host_end_to_last_kernel_tail=12.145 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `1606885329`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[6000, 1536]]}`
- `q_b_proj` -> 0.456 ms
  纯GPU kernel时间: `0.336 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.456 ms`, `host_to_first_kernel_gap=12.08088`, `host_end_to_last_kernel_tail=11.962 ms`, `gpu_makespan=0.338 ms`, `gpu_kernel_sum=0.336 ms`
  开始时间(ns): `1607054172`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[6000, 1536]]}`
- `kv_a_layernorm` -> 0.085 ms
  纯GPU kernel时间: `0.011 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.085 ms`, `host_to_first_kernel_gap=11.829609`, `host_end_to_last_kernel_tail=11.756 ms`, `gpu_makespan=0.011 ms`, `gpu_kernel_sum=0.011 ms`
  开始时间(ns): `1607643235`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[6000, 512]]}`
- `rotary_emb` -> 0.164 ms
  纯GPU kernel时间: `0.079 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.164 ms`, `host_to_first_kernel_gap=11.673942`, `host_end_to_last_kernel_tail=11.589 ms`, `gpu_makespan=0.079 ms`, `gpu_kernel_sum=0.079 ms`
  开始时间(ns): `1607810614`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[6000], [6000, 128, 64], [6000, 1, 64]]}`
- `kv_b_proj` -> 0.444 ms
  纯GPU kernel时间: `0.212 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.444 ms`, `host_to_first_kernel_gap=11.346682`, `host_end_to_last_kernel_tail=11.116 ms`, `gpu_makespan=0.214 ms`, `gpu_kernel_sum=0.212 ms`
  开始时间(ns): `1608239282`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[6000, 512]]}`
- `attn_mha` -> 0.295 ms
  纯GPU kernel时间: `2.009 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.295 ms`, `host_to_first_kernel_gap=11.095114`, `host_end_to_last_kernel_tail=12.809 ms`, `gpu_makespan=2.009 ms`, `gpu_kernel_sum=2.009 ms`
  开始时间(ns): `1608884002`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[6000, 128, 192], [6000, 128, 192], [6000, 128, 128]]}`
- `o_proj` -> 0.463 ms
  纯GPU kernel时间: `0.982 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.463 ms`, `host_to_first_kernel_gap=12.724534`, `host_end_to_last_kernel_tail=13.245 ms`, `gpu_makespan=0.983 ms`, `gpu_kernel_sum=0.982 ms`
  开始时间(ns): `1609264950`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[6000, 16384]]}`

## Layer 0 / decode / instance 1

- 执行序号: `6`
- Collector对齐边界: `{'Module': 'model.model.layers.0.self_attn'}`
- 整块 MLA-module 时长: `3.360 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=3.360 ms`, `module_to_last_kernel=14.122 ms`, `host_to_first_kernel_gap=13.970388`, `host_end_to_last_kernel_tail=10.761 ms`, `gpu_makespan=0.151 ms`, `gpu_kernel_sum=0.120 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.595 ms
  纯GPU kernel时间: `0.016 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.595 ms`, `host_to_first_kernel_gap=13.836274`, `host_end_to_last_kernel_tail=13.258 ms`, `gpu_makespan=0.017 ms`, `gpu_kernel_sum=0.016 ms`
  开始时间(ns): `1617586232`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.089 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.089 ms`, `host_to_first_kernel_gap=13.14098`, `host_end_to_last_kernel_tail=13.054 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1618298422`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.057 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.057 ms`, `host_to_first_kernel_gap=13.008042`, `host_end_to_last_kernel_tail=12.953 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1618433312`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.451 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.451 ms`, `host_to_first_kernel_gap=12.880286`, `host_end_to_last_kernel_tail=12.450 ms`, `gpu_makespan=0.021 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `1618563980`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.141 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.141 ms`, `host_to_first_kernel_gap=12.169253`, `host_end_to_last_kernel_tail=12.030 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1619307461`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.521 ms
  纯GPU kernel时间: `0.029 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.521 ms`, `host_to_first_kernel_gap=11.950479`, `host_end_to_last_kernel_tail=11.461 ms`, `gpu_makespan=0.031 ms`, `gpu_kernel_sum=0.029 ms`
  开始时间(ns): `1619529275`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.499 ms
  纯GPU kernel时间: `0.050 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.499 ms`, `host_to_first_kernel_gap=11.241262`, `host_end_to_last_kernel_tail=10.793 ms`, `gpu_makespan=0.051 ms`, `gpu_kernel_sum=0.050 ms`
  开始时间(ns): `1620281756`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 1 / decode / instance 1

- 执行序号: `7`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `2.835 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.835 ms`, `module_to_last_kernel=9.600 ms`, `host_to_first_kernel_gap=9.449007`, `host_end_to_last_kernel_tail=6.764 ms`, `gpu_makespan=0.150 ms`, `gpu_kernel_sum=0.121 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.374 ms
  纯GPU kernel时间: `0.017 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.374 ms`, `host_to_first_kernel_gap=9.343379`, `host_end_to_last_kernel_tail=8.987 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.017 ms`
  开始时间(ns): `1622390487`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.075 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.075 ms`, `host_to_first_kernel_gap=8.884577`, `host_end_to_last_kernel_tail=8.812 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1622867337`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.052 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.052 ms`, `host_to_first_kernel_gap=8.762223`, `host_end_to_last_kernel_tail=8.712 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1622991611`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.361 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.361 ms`, `host_to_first_kernel_gap=8.655645`, `host_end_to_last_kernel_tail=8.314 ms`, `gpu_makespan=0.019 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `1623101773`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.130 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.130 ms`, `host_to_first_kernel_gap=8.080401`, `host_end_to_last_kernel_tail=7.952 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1623707705`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.481 ms
  纯GPU kernel时间: `0.030 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.481 ms`, `host_to_first_kernel_gap=7.876906`, `host_end_to_last_kernel_tail=7.426 ms`, `gpu_makespan=0.030 ms`, `gpu_kernel_sum=0.030 ms`
  开始时间(ns): `1623914016`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.487 ms
  纯GPU kernel时间: `0.051 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.487 ms`, `host_to_first_kernel_gap=7.230685`, `host_end_to_last_kernel_tail=6.795 ms`, `gpu_makespan=0.052 ms`, `gpu_kernel_sum=0.051 ms`
  开始时间(ns): `1624602061`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 2 / decode / instance 1

- 执行序号: `8`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `2.492 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.492 ms`, `module_to_last_kernel=5.719 ms`, `host_to_first_kernel_gap=5.566708`, `host_end_to_last_kernel_tail=3.227 ms`, `gpu_makespan=0.152 ms`, `gpu_kernel_sum=0.118 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.329 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.329 ms`, `host_to_first_kernel_gap=5.469721`, `host_end_to_last_kernel_tail=5.157 ms`, `gpu_makespan=0.017 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `1626574129`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.077 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.077 ms`, `host_to_first_kernel_gap=5.053448`, `host_end_to_last_kernel_tail=4.978 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1627007074`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.048 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.048 ms`, `host_to_first_kernel_gap=4.942598`, `host_end_to_last_kernel_tail=4.897 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1627120100`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.309 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.309 ms`, `host_to_first_kernel_gap=4.844086`, `host_end_to_last_kernel_tail=4.554 ms`, `gpu_makespan=0.020 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `1627221428`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.111 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.111 ms`, `host_to_first_kernel_gap=4.355196`, `host_end_to_last_kernel_tail=4.246 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1627741934`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.428 ms
  纯GPU kernel时间: `0.030 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.428 ms`, `host_to_first_kernel_gap=4.184004`, `host_end_to_last_kernel_tail=3.788 ms`, `gpu_makespan=0.032 ms`, `gpu_kernel_sum=0.030 ms`
  开始时间(ns): `1627915974`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.403 ms
  纯GPU kernel时间: `0.051 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.403 ms`, `host_to_first_kernel_gap=3.615013`, `host_end_to_last_kernel_tail=3.264 ms`, `gpu_makespan=0.052 ms`, `gpu_kernel_sum=0.051 ms`
  开始时间(ns): `1628529029`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 3 / decode / instance 1

- 执行序号: `9`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `2.272 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.272 ms`, `module_to_last_kernel=2.301 ms`, `host_to_first_kernel_gap=2.148849`, `host_end_to_last_kernel_tail=0.030 ms`, `gpu_makespan=0.152 ms`, `gpu_kernel_sum=0.121 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.293 ms
  纯GPU kernel时间: `0.017 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.293 ms`, `host_to_first_kernel_gap=2.063053`, `host_end_to_last_kernel_tail=1.789 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.017 ms`
  开始时间(ns): `1630292701`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.062 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.062 ms`, `host_to_first_kernel_gap=1.711749`, `host_end_to_last_kernel_tail=1.652 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1630661989`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.043 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.043 ms`, `host_to_first_kernel_gap=1.616005`, `host_end_to_last_kernel_tail=1.575 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1630759813`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.289 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.289 ms`, `host_to_first_kernel_gap=1.531053`, `host_end_to_last_kernel_tail=1.262 ms`, `gpu_makespan=0.020 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `1630848829`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.105 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.105 ms`, `host_to_first_kernel_gap=1.079184`, `host_end_to_last_kernel_tail=0.976 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1631332602`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.391 ms
  纯GPU kernel时间: `0.030 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.391 ms`, `host_to_first_kernel_gap=0.919034`, `host_end_to_last_kernel_tail=0.558 ms`, `gpu_makespan=0.030 ms`, `gpu_kernel_sum=0.030 ms`
  开始时间(ns): `1631495408`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.376 ms
  纯GPU kernel时间: `0.050 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.376 ms`, `host_to_first_kernel_gap=0.382013`, `host_end_to_last_kernel_tail=0.056 ms`, `gpu_makespan=0.051 ms`, `gpu_kernel_sum=0.050 ms`
  开始时间(ns): `1632075341`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 4 / decode / instance 1

- 执行序号: `10`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `2.343 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.343 ms`, `module_to_last_kernel=2.343 ms`, `host_to_first_kernel_gap=0.218305`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=2.118 ms`, `gpu_kernel_sum=0.118 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.377 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.377 ms`, `host_to_first_kernel_gap=0.124987`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.219 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `1634596975`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.065 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.065 ms`, `host_to_first_kernel_gap=0.05583`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1635057172`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.047 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.047 ms`, `host_to_first_kernel_gap=0.040743`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1635153027`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.315 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.315 ms`, `host_to_first_kernel_gap=0.116483`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.183 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `1635249191`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.105 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.105 ms`, `host_to_first_kernel_gap=0.088906`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1635774048`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.381 ms
  纯GPU kernel时间: `0.029 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.381 ms`, `host_to_first_kernel_gap=0.126095`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.228 ms`, `gpu_kernel_sum=0.029 ms`
  开始时间(ns): `1635942555`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.335 ms
  纯GPU kernel时间: `0.050 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.335 ms`, `host_to_first_kernel_gap=0.122928`, `host_end_to_last_kernel_tail=0.016 ms`, `gpu_makespan=0.228 ms`, `gpu_kernel_sum=0.050 ms`
  开始时间(ns): `1636489082`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`
