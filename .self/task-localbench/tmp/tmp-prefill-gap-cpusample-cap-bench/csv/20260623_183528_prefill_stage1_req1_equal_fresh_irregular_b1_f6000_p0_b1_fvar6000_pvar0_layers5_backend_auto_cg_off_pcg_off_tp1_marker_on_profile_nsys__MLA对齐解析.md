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
- 整块 MLA-module 时长: `481.461 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=481.461 ms`, `module_to_last_kernel=484.072 ms`, `host_to_first_kernel_gap=0.307734`, `host_end_to_last_kernel_tail=2.611 ms`, `gpu_makespan=483.764 ms`, `gpu_kernel_sum=3.821 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=6000`, `total_tokens=6000`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[6000, 128, 192], [6000, 128, 192], [6000, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=6000`, `sum_prefix=0`, `sum_seq_after=6000`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 6000, 'prefix_len': 0, 'seq_len_after': 6000, 'prompt_len': 6000, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.441 ms
  纯GPU kernel时间: `0.189 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.441 ms`, `host_to_first_kernel_gap=0.13678`, `host_end_to_last_kernel_tail=0.084 ms`, `gpu_makespan=0.388 ms`, `gpu_kernel_sum=0.189 ms`
  开始时间(ns): `529528002`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[6000, 7168]]}`
- `q_a_layernorm` -> 0.053 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.053 ms`, `host_to_first_kernel_gap=0.047147`, `host_end_to_last_kernel_tail=0.007 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `530034052`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[6000, 1536]]}`
- `q_b_proj` -> 0.269 ms
  纯GPU kernel时间: `0.341 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.269 ms`, `host_to_first_kernel_gap=0.097264`, `host_end_to_last_kernel_tail=0.304 ms`, `gpu_makespan=0.475 ms`, `gpu_kernel_sum=0.341 ms`
  开始时间(ns): `530122783`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[6000, 1536]]}`
- `kv_a_layernorm` -> 0.043 ms
  纯GPU kernel时间: `0.010 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.043 ms`, `host_to_first_kernel_gap=0.234061`, `host_end_to_last_kernel_tail=0.201 ms`, `gpu_makespan=0.010 ms`, `gpu_kernel_sum=0.010 ms`
  开始时间(ns): `530460834`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[6000, 512]]}`
- `rotary_emb` -> 0.106 ms
  纯GPU kernel时间: `0.079 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.106 ms`, `host_to_first_kernel_gap=0.155159`, `host_end_to_last_kernel_tail=0.128 ms`, `gpu_makespan=0.079 ms`, `gpu_kernel_sum=0.079 ms`
  开始时间(ns): `530551608`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[6000], [6000, 128, 64], [6000, 1, 64]]}`
- `kv_b_proj` -> 0.603 ms
  纯GPU kernel时间: `0.208 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.603 ms`, `host_to_first_kernel_gap=0.267418`, `host_end_to_last_kernel_tail=0.157 ms`, `gpu_makespan=0.493 ms`, `gpu_kernel_sum=0.208 ms`
  开始时间(ns): `1009508695`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[6000, 512]]}`
- `attn_mha` -> 0.199 ms
  纯GPU kernel时间: `2.005 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.199 ms`, `host_to_first_kernel_gap=0.177955`, `host_end_to_last_kernel_tail=1.985 ms`, `gpu_makespan=2.005 ms`, `gpu_kernel_sum=2.005 ms`
  开始时间(ns): `1010267535`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[6000, 128, 192], [6000, 128, 192], [6000, 128, 128]]}`
- `o_proj` -> 0.272 ms
  纯GPU kernel时间: `0.976 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.272 ms`, `host_to_first_kernel_gap=1.922915`, `host_end_to_last_kernel_tail=2.628 ms`, `gpu_makespan=0.977 ms`, `gpu_kernel_sum=0.976 ms`
  开始时间(ns): `1010529328`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[6000, 16384]]}`

## Layer 1 / prefill / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `2.067 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.067 ms`, `module_to_last_kernel=9.198 ms`, `host_to_first_kernel_gap=5.17248`, `host_end_to_last_kernel_tail=7.132 ms`, `gpu_makespan=4.026 ms`, `gpu_kernel_sum=3.811 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=6000`, `total_tokens=6000`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[6000, 128, 192], [6000, 128, 192], [6000, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=6000`, `sum_prefix=0`, `sum_seq_after=6000`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 6000, 'prefix_len': 0, 'seq_len_after': 6000, 'prompt_len': 6000, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.249 ms
  纯GPU kernel时间: `0.187 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.249 ms`, `host_to_first_kernel_gap=4.987829`, `host_end_to_last_kernel_tail=4.929 ms`, `gpu_makespan=0.190 ms`, `gpu_kernel_sum=0.187 ms`
  开始时间(ns): `1011953249`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[6000, 7168]]}`
- `q_a_layernorm` -> 0.059 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.059 ms`, `host_to_first_kernel_gap=4.868683`, `host_end_to_last_kernel_tail=4.823 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `1012262955`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[6000, 1536]]}`
- `q_b_proj` -> 0.214 ms
  纯GPU kernel时间: `0.333 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.214 ms`, `host_to_first_kernel_gap=4.793779`, `host_end_to_last_kernel_tail=4.915 ms`, `gpu_makespan=0.335 ms`, `gpu_kernel_sum=0.333 ms`
  开始时间(ns): `1012352291`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[6000, 1536]]}`
- `kv_a_layernorm` -> 0.043 ms
  纯GPU kernel时间: `0.011 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.043 ms`, `host_to_first_kernel_gap=4.852204`, `host_end_to_last_kernel_tail=4.820 ms`, `gpu_makespan=0.011 ms`, `gpu_kernel_sum=0.011 ms`
  开始时间(ns): `1012629098`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[6000, 512]]}`
- `rotary_emb` -> 0.106 ms
  纯GPU kernel时间: `0.078 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.106 ms`, `host_to_first_kernel_gap=4.776543`, `host_end_to_last_kernel_tail=4.748 ms`, `gpu_makespan=0.078 ms`, `gpu_kernel_sum=0.078 ms`
  开始时间(ns): `1012716919`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[6000], [6000, 128, 64], [6000, 1, 64]]}`
- `kv_b_proj` -> 0.263 ms
  纯GPU kernel时间: `0.206 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.263 ms`, `host_to_first_kernel_gap=4.585478`, `host_end_to_last_kernel_tail=4.531 ms`, `gpu_makespan=0.208 ms`, `gpu_kernel_sum=0.206 ms`
  开始时间(ns): `1013008593`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[6000, 512]]}`
- `attn_mha` -> 0.160 ms
  纯GPU kernel时间: `2.005 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.160 ms`, `host_to_first_kernel_gap=4.600276`, `host_end_to_last_kernel_tail=6.446 ms`, `gpu_makespan=2.005 ms`, `gpu_kernel_sum=2.005 ms`
  开始时间(ns): `1013380547`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[6000, 128, 192], [6000, 128, 192], [6000, 128, 128]]}`
- `o_proj` -> 0.239 ms
  纯GPU kernel时间: `0.977 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.239 ms`, `host_to_first_kernel_gap=6.406166`, `host_end_to_last_kernel_tail=7.145 ms`, `gpu_makespan=0.979 ms`, `gpu_kernel_sum=0.977 ms`
  开始时间(ns): `1013582306`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[6000, 16384]]}`

## Layer 2 / prefill / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.858 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.858 ms`, `module_to_last_kernel=13.848 ms`, `host_to_first_kernel_gap=9.811995`, `host_end_to_last_kernel_tail=11.989 ms`, `gpu_makespan=4.036 ms`, `gpu_kernel_sum=3.820 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=6000`, `total_tokens=6000`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[6000, 128, 192], [6000, 128, 192], [6000, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=6000`, `sum_prefix=0`, `sum_seq_after=6000`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 6000, 'prefix_len': 0, 'seq_len_after': 6000, 'prompt_len': 6000, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.230 ms
  纯GPU kernel时间: `0.187 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.230 ms`, `host_to_first_kernel_gap=9.699022`, `host_end_to_last_kernel_tail=9.657 ms`, `gpu_makespan=0.188 ms`, `gpu_kernel_sum=0.187 ms`
  开始时间(ns): `1014786573`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[6000, 7168]]}`
- `q_a_layernorm` -> 0.048 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.048 ms`, `host_to_first_kernel_gap=9.606954`, `host_end_to_last_kernel_tail=9.572 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `1015067121`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[6000, 1536]]}`
- `q_b_proj` -> 0.204 ms
  纯GPU kernel时间: `0.335 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.204 ms`, `host_to_first_kernel_gap=9.542462`, `host_end_to_last_kernel_tail=9.675 ms`, `gpu_makespan=0.336 ms`, `gpu_kernel_sum=0.335 ms`
  开始时间(ns): `1015147069`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[6000, 1536]]}`
- `kv_a_layernorm` -> 0.044 ms
  纯GPU kernel时间: `0.011 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.044 ms`, `host_to_first_kernel_gap=9.615298`, `host_end_to_last_kernel_tail=9.582 ms`, `gpu_makespan=0.011 ms`, `gpu_kernel_sum=0.011 ms`
  开始时间(ns): `1015410393`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[6000, 512]]}`
- `rotary_emb` -> 0.074 ms
  纯GPU kernel时间: `0.078 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.074 ms`, `host_to_first_kernel_gap=9.544212`, `host_end_to_last_kernel_tail=9.549 ms`, `gpu_makespan=0.078 ms`, `gpu_kernel_sum=0.078 ms`
  开始时间(ns): `1015494631`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[6000], [6000, 128, 64], [6000, 1, 64]]}`
- `kv_b_proj` -> 0.260 ms
  纯GPU kernel时间: `0.215 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.260 ms`, `host_to_first_kernel_gap=9.428097`, `host_end_to_last_kernel_tail=9.385 ms`, `gpu_makespan=0.216 ms`, `gpu_kernel_sum=0.215 ms`
  开始时间(ns): `1015711067`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[6000, 512]]}`
- `attn_mha` -> 0.148 ms
  纯GPU kernel时间: `2.006 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.148 ms`, `host_to_first_kernel_gap=9.459387`, `host_end_to_last_kernel_tail=11.317 ms`, `gpu_makespan=2.006 ms`, `gpu_kernel_sum=2.006 ms`
  开始时间(ns): `1016077057`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[6000, 128, 192], [6000, 128, 192], [6000, 128, 128]]}`
- `o_proj` -> 0.249 ms
  纯GPU kernel时间: `0.975 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.249 ms`, `host_to_first_kernel_gap=11.273473`, `host_end_to_last_kernel_tail=12.002 ms`, `gpu_makespan=0.978 ms`, `gpu_kernel_sum=0.975 ms`
  开始时间(ns): `1016270140`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[6000, 16384]]}`

## Layer 3 / prefill / instance 1

- 执行序号: `4`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `1.834 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.834 ms`, `module_to_last_kernel=18.696 ms`, `host_to_first_kernel_gap=14.673681`, `host_end_to_last_kernel_tail=16.862 ms`, `gpu_makespan=4.023 ms`, `gpu_kernel_sum=3.809 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=6000`, `total_tokens=6000`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[6000, 128, 192], [6000, 128, 192], [6000, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=6000`, `sum_prefix=0`, `sum_seq_after=6000`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 6000, 'prefix_len': 0, 'seq_len_after': 6000, 'prompt_len': 6000, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.221 ms
  纯GPU kernel时间: `0.187 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.221 ms`, `host_to_first_kernel_gap=14.556869`, `host_end_to_last_kernel_tail=14.525 ms`, `gpu_makespan=0.189 ms`, `gpu_kernel_sum=0.187 ms`
  开始时间(ns): `1017482011`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[6000, 7168]]}`
- `q_a_layernorm` -> 0.049 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.049 ms`, `host_to_first_kernel_gap=14.477407`, `host_end_to_last_kernel_tail=14.441 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `1017751105`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[6000, 1536]]}`
- `q_b_proj` -> 0.213 ms
  纯GPU kernel时间: `0.332 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.213 ms`, `host_to_first_kernel_gap=14.411342`, `host_end_to_last_kernel_tail=14.532 ms`, `gpu_makespan=0.334 ms`, `gpu_kernel_sum=0.332 ms`
  开始时间(ns): `1017831442`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[6000, 1536]]}`
- `kv_a_layernorm` -> 0.047 ms
  纯GPU kernel时间: `0.010 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.047 ms`, `host_to_first_kernel_gap=14.461608`, `host_end_to_last_kernel_tail=14.424 ms`, `gpu_makespan=0.010 ms`, `gpu_kernel_sum=0.010 ms`
  开始时间(ns): `1018114968`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[6000, 512]]}`
- `rotary_emb` -> 0.077 ms
  纯GPU kernel时间: `0.078 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.077 ms`, `host_to_first_kernel_gap=14.384345`, `host_end_to_last_kernel_tail=14.386 ms`, `gpu_makespan=0.078 ms`, `gpu_kernel_sum=0.078 ms`
  开始时间(ns): `1018203559`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[6000], [6000, 128, 64], [6000, 1, 64]]}`
- `kv_b_proj` -> 0.241 ms
  纯GPU kernel时间: `0.207 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.241 ms`, `host_to_first_kernel_gap=14.265203`, `host_end_to_last_kernel_tail=14.233 ms`, `gpu_makespan=0.209 ms`, `gpu_kernel_sum=0.207 ms`
  开始时间(ns): `1018422382`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[6000, 512]]}`
- `attn_mha` -> 0.144 ms
  纯GPU kernel时间: `2.007 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.144 ms`, `host_to_first_kernel_gap=14.317262`, `host_end_to_last_kernel_tail=16.180 ms`, `gpu_makespan=2.006 ms`, `gpu_kernel_sum=2.007 ms`
  开始时间(ns): `1018758355`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[6000, 128, 192], [6000, 128, 192], [6000, 128, 128]]}`
- `o_proj` -> 0.231 ms
  纯GPU kernel时间: `0.975 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.231 ms`, `host_to_first_kernel_gap=16.1306`, `host_end_to_last_kernel_tail=16.876 ms`, `gpu_makespan=0.977 ms`, `gpu_kernel_sum=0.975 ms`
  开始时间(ns): `1018954042`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[6000, 16384]]}`

## Layer 4 / prefill / instance 1

- 执行序号: `5`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `1.918 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.918 ms`, `module_to_last_kernel=26.981 ms`, `host_to_first_kernel_gap=22.951741`, `host_end_to_last_kernel_tail=25.063 ms`, `gpu_makespan=4.029 ms`, `gpu_kernel_sum=3.816 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=6000`, `total_tokens=6000`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[6000, 128, 192], [6000, 128, 192], [6000, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=6000`, `sum_prefix=0`, `sum_seq_after=6000`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 6000, 'prefix_len': 0, 'seq_len_after': 6000, 'prompt_len': 6000, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.287 ms
  纯GPU kernel时间: `0.187 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.287 ms`, `host_to_first_kernel_gap=22.828144`, `host_end_to_last_kernel_tail=22.730 ms`, `gpu_makespan=0.189 ms`, `gpu_kernel_sum=0.187 ms`
  开始时间(ns): `1020866104`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[6000, 7168]]}`
- `q_a_layernorm` -> 0.046 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.046 ms`, `host_to_first_kernel_gap=22.674141`, `host_end_to_last_kernel_tail=22.641 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `1021209803`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[6000, 1536]]}`
- `q_b_proj` -> 0.226 ms
  纯GPU kernel时间: `0.333 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.226 ms`, `host_to_first_kernel_gap=22.604108`, `host_end_to_last_kernel_tail=22.714 ms`, `gpu_makespan=0.335 ms`, `gpu_kernel_sum=0.333 ms`
  开始时间(ns): `1021294204`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[6000, 1536]]}`
- `kv_a_layernorm` -> 0.041 ms
  纯GPU kernel时间: `0.011 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.041 ms`, `host_to_first_kernel_gap=22.654032`, `host_end_to_last_kernel_tail=22.624 ms`, `gpu_makespan=0.011 ms`, `gpu_kernel_sum=0.011 ms`
  开始时间(ns): `1021579640`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[6000, 512]]}`
- `rotary_emb` -> 0.078 ms
  纯GPU kernel时间: `0.078 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.078 ms`, `host_to_first_kernel_gap=22.58525`, `host_end_to_last_kernel_tail=22.586 ms`, `gpu_makespan=0.078 ms`, `gpu_kernel_sum=0.078 ms`
  开始时间(ns): `1021660742`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[6000], [6000, 128, 64], [6000, 1, 64]]}`
- `kv_b_proj` -> 0.242 ms
  纯GPU kernel时间: `0.210 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.242 ms`, `host_to_first_kernel_gap=22.471324`, `host_end_to_last_kernel_tail=22.441 ms`, `gpu_makespan=0.212 ms`, `gpu_kernel_sum=0.210 ms`
  开始时间(ns): `1021875756`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[6000, 512]]}`
- `attn_mha` -> 0.147 ms
  纯GPU kernel时间: `2.008 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.147 ms`, `host_to_first_kernel_gap=22.522048`, `host_end_to_last_kernel_tail=24.382 ms`, `gpu_makespan=2.008 ms`, `gpu_kernel_sum=2.008 ms`
  开始时间(ns): `1022215401`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[6000, 128, 192], [6000, 128, 192], [6000, 128, 128]]}`
- `o_proj` -> 0.240 ms
  纯GPU kernel时间: `0.976 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.240 ms`, `host_to_first_kernel_gap=24.339068`, `host_end_to_last_kernel_tail=25.076 ms`, `gpu_makespan=0.977 ms`, `gpu_kernel_sum=0.976 ms`
  开始时间(ns): `1022407374`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[6000, 16384]]}`

## Layer 0 / decode / instance 1

- 执行序号: `6`
- Collector对齐边界: `{'Module': 'model.model.layers.0.self_attn'}`
- 整块 MLA-module 时长: `1.940 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.940 ms`, `module_to_last_kernel=29.232 ms`, `host_to_first_kernel_gap=29.080354`, `host_end_to_last_kernel_tail=27.291 ms`, `gpu_makespan=0.151 ms`, `gpu_kernel_sum=0.119 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.322 ms
  纯GPU kernel时间: `0.016 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.322 ms`, `host_to_first_kernel_gap=29.003333`, `host_end_to_last_kernel_tail=28.699 ms`, `gpu_makespan=0.017 ms`, `gpu_kernel_sum=0.016 ms`
  开始时间(ns): `1027037707`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.052 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.052 ms`, `host_to_first_kernel_gap=28.634204`, `host_end_to_last_kernel_tail=28.585 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1027424116`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.031 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.031 ms`, `host_to_first_kernel_gap=28.558782`, `host_end_to_last_kernel_tail=28.530 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1027501298`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.250 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.250 ms`, `host_to_first_kernel_gap=28.492946`, `host_end_to_last_kernel_tail=28.264 ms`, `gpu_makespan=0.021 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `1027570718`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.088 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.088 ms`, `host_to_first_kernel_gap=28.094413`, `host_end_to_last_kernel_tail=28.008 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1028001571`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.314 ms
  纯GPU kernel时间: `0.029 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.314 ms`, `host_to_first_kernel_gap=27.948364`, `host_end_to_last_kernel_tail=27.664 ms`, `gpu_makespan=0.030 ms`, `gpu_kernel_sum=0.029 ms`
  开始时间(ns): `1028150756`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.284 ms
  纯GPU kernel时间: `0.049 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.284 ms`, `host_to_first_kernel_gap=27.543091`, `host_end_to_last_kernel_tail=27.309 ms`, `gpu_makespan=0.050 ms`, `gpu_kernel_sum=0.049 ms`
  开始时间(ns): `1028599165`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 1 / decode / instance 1

- 执行序号: `7`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `1.662 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.662 ms`, `module_to_last_kernel=26.775 ms`, `host_to_first_kernel_gap=26.62393`, `host_end_to_last_kernel_tail=25.112 ms`, `gpu_makespan=0.151 ms`, `gpu_kernel_sum=0.120 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.232 ms
  纯GPU kernel时间: `0.017 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.232 ms`, `host_to_first_kernel_gap=26.55783`, `host_end_to_last_kernel_tail=26.343 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.017 ms`
  开始时间(ns): `1029796074`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.053 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.053 ms`, `host_to_first_kernel_gap=26.285642`, `host_end_to_last_kernel_tail=26.234 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1030086310`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.031 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.031 ms`, `host_to_first_kernel_gap=26.209648`, `host_end_to_last_kernel_tail=26.180 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1030164192`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.218 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.218 ms`, `host_to_first_kernel_gap=26.148496`, `host_end_to_last_kernel_tail=25.949 ms`, `gpu_makespan=0.019 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `1030228160`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.084 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.084 ms`, `host_to_first_kernel_gap=25.818996`, `host_end_to_last_kernel_tail=25.737 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1030588988`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.282 ms
  纯GPU kernel时间: `0.030 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.282 ms`, `host_to_first_kernel_gap=25.694953`, `host_end_to_last_kernel_tail=25.443 ms`, `gpu_makespan=0.031 ms`, `gpu_kernel_sum=0.030 ms`
  开始时间(ns): `1030715943`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.252 ms
  纯GPU kernel时间: `0.050 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.252 ms`, `host_to_first_kernel_gap=25.32987`, `host_end_to_last_kernel_tail=25.129 ms`, `gpu_makespan=0.051 ms`, `gpu_kernel_sum=0.050 ms`
  开始时间(ns): `1031123682`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 2 / decode / instance 1

- 执行序号: `8`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.598 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.598 ms`, `module_to_last_kernel=24.609 ms`, `host_to_first_kernel_gap=24.462293`, `host_end_to_last_kernel_tail=23.011 ms`, `gpu_makespan=0.147 ms`, `gpu_kernel_sum=0.117 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.205 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.205 ms`, `host_to_first_kernel_gap=24.402596`, `host_end_to_last_kernel_tail=24.214 ms`, `gpu_makespan=0.017 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `1032262380`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.042 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.042 ms`, `host_to_first_kernel_gap=24.162024`, `host_end_to_last_kernel_tail=24.122 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1032519689`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.031 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.031 ms`, `host_to_first_kernel_gap=24.09517`, `host_end_to_last_kernel_tail=24.066 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1032588783`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.194 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.194 ms`, `host_to_first_kernel_gap=24.035343`, `host_end_to_last_kernel_tail=23.860 ms`, `gpu_makespan=0.019 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `1032651554`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.081 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.081 ms`, `host_to_first_kernel_gap=23.722523`, `host_end_to_last_kernel_tail=23.643 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1032993814`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.279 ms
  纯GPU kernel时间: `0.029 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.279 ms`, `host_to_first_kernel_gap=23.599771`, `host_end_to_last_kernel_tail=23.351 ms`, `gpu_makespan=0.031 ms`, `gpu_kernel_sum=0.029 ms`
  开始时间(ns): `1033119606`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.267 ms
  纯GPU kernel时间: `0.050 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.267 ms`, `host_to_first_kernel_gap=23.244294`, `host_end_to_last_kernel_tail=23.028 ms`, `gpu_makespan=0.051 ms`, `gpu_kernel_sum=0.050 ms`
  开始时间(ns): `1033516971`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 3 / decode / instance 1

- 执行序号: `9`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `1.601 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.601 ms`, `module_to_last_kernel=22.503 ms`, `host_to_first_kernel_gap=22.351422`, `host_end_to_last_kernel_tail=20.901 ms`, `gpu_makespan=0.151 ms`, `gpu_kernel_sum=0.120 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.203 ms
  纯GPU kernel时间: `0.017 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.203 ms`, `host_to_first_kernel_gap=22.291458`, `host_end_to_last_kernel_tail=22.107 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.017 ms`
  开始时间(ns): `1034681231`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.044 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.044 ms`, `host_to_first_kernel_gap=22.045541`, `host_end_to_last_kernel_tail=22.003 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1034945100`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.036 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.036 ms`, `host_to_first_kernel_gap=21.981937`, `host_end_to_last_kernel_tail=21.948 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1035010848`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.198 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.198 ms`, `host_to_first_kernel_gap=21.917266`, `host_end_to_last_kernel_tail=21.738 ms`, `gpu_makespan=0.019 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `1035079455`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.075 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.075 ms`, `host_to_first_kernel_gap=21.611963`, `host_end_to_last_kernel_tail=21.539 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1035414902`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.273 ms
  纯GPU kernel时间: `0.030 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.273 ms`, `host_to_first_kernel_gap=21.490088`, `host_end_to_last_kernel_tail=21.248 ms`, `gpu_makespan=0.031 ms`, `gpu_kernel_sum=0.030 ms`
  开始时间(ns): `1035539721`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.276 ms
  纯GPU kernel时间: `0.050 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.276 ms`, `host_to_first_kernel_gap=21.144361`, `host_end_to_last_kernel_tail=20.919 ms`, `gpu_makespan=0.051 ms`, `gpu_kernel_sum=0.050 ms`
  开始时间(ns): `1035928936`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 4 / decode / instance 1

- 执行序号: `10`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `1.778 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.778 ms`, `module_to_last_kernel=19.797 ms`, `host_to_first_kernel_gap=19.647876`, `host_end_to_last_kernel_tail=18.018 ms`, `gpu_makespan=0.149 ms`, `gpu_kernel_sum=0.119 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.288 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.288 ms`, `host_to_first_kernel_gap=19.578854`, `host_end_to_last_kernel_tail=19.307 ms`, `gpu_makespan=0.016 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `1037752043`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.052 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.052 ms`, `host_to_first_kernel_gap=19.246763`, `host_end_to_last_kernel_tail=19.197 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1038100102`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.035 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.035 ms`, `host_to_first_kernel_gap=19.172985`, `host_end_to_last_kernel_tail=19.140 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1038175800`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.240 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.240 ms`, `host_to_first_kernel_gap=19.096807`, `host_end_to_last_kernel_tail=18.875 ms`, `gpu_makespan=0.019 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `1038255210`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.077 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.077 ms`, `host_to_first_kernel_gap=18.732629`, `host_end_to_last_kernel_tail=18.657 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1038648828`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.288 ms
  纯GPU kernel时间: `0.029 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.288 ms`, `host_to_first_kernel_gap=18.615036`, `host_end_to_last_kernel_tail=18.357 ms`, `gpu_makespan=0.030 ms`, `gpu_kernel_sum=0.029 ms`
  开始时间(ns): `1038770261`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.265 ms
  纯GPU kernel时间: `0.051 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.265 ms`, `host_to_first_kernel_gap=18.247711`, `host_end_to_last_kernel_tail=18.036 ms`, `gpu_makespan=0.053 ms`, `gpu_kernel_sum=0.051 ms`
  开始时间(ns): `1039179218`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`
