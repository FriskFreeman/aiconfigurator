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
- 整块 MLA-module 时长: `512.781 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=512.781 ms`, `module_to_last_kernel=517.349 ms`, `host_to_first_kernel_gap=0.462254`, `host_end_to_last_kernel_tail=4.568 ms`, `gpu_makespan=516.887 ms`, `gpu_kernel_sum=6.004 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=8192`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=0`, `sum_seq_after=8192`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 8192, 'prefix_len': 0, 'seq_len_after': 8192, 'prompt_len': 8192, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.651 ms
  纯GPU kernel时间: `0.273 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.651 ms`, `host_to_first_kernel_gap=0.22716`, `host_end_to_last_kernel_tail=0.141 ms`, `gpu_makespan=0.566 ms`, `gpu_kernel_sum=0.273 ms`
  开始时间(ns): `580099950`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.088 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.088 ms`, `host_to_first_kernel_gap=0.075894`, `host_end_to_last_kernel_tail=0.006 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `580851888`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.404 ms
  纯GPU kernel时间: `0.446 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.404 ms`, `host_to_first_kernel_gap=0.144828`, `host_end_to_last_kernel_tail=0.379 ms`, `gpu_makespan=0.638 ms`, `gpu_kernel_sum=0.446 ms`
  开始时间(ns): `580996074`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.082 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.082 ms`, `host_to_first_kernel_gap=0.265522`, `host_end_to_last_kernel_tail=0.197 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `581512981`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.145 ms
  纯GPU kernel时间: `0.106 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.145 ms`, `host_to_first_kernel_gap=0.124275`, `host_end_to_last_kernel_tail=0.086 ms`, `gpu_makespan=0.106 ms`, `gpu_kernel_sum=0.106 ms`
  开始时间(ns): `581668948`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.658 ms
  纯GPU kernel时间: `0.290 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.658 ms`, `host_to_first_kernel_gap=0.294349`, `host_end_to_last_kernel_tail=0.234 ms`, `gpu_makespan=0.597 ms`, `gpu_kernel_sum=0.290 ms`
  开始时间(ns): `1091220808`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[8192, 512]]}`
- `attn_mha` -> 0.212 ms
  纯GPU kernel时间: `3.605 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.212 ms`, `host_to_first_kernel_gap=0.306755`, `host_end_to_last_kernel_tail=3.700 ms`, `gpu_makespan=3.605 ms`, `gpu_kernel_sum=3.605 ms`
  开始时间(ns): `1092047858`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]}`
- `o_proj` -> 0.310 ms
  纯GPU kernel时间: `1.251 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.310 ms`, `host_to_first_kernel_gap=3.642811`, `host_end_to_last_kernel_tail=4.586 ms`, `gpu_makespan=1.253 ms`, `gpu_kernel_sum=1.251 ms`
  开始时间(ns): `1092318202`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 1 / prefill / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `2.188 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.188 ms`, `module_to_last_kernel=14.430 ms`, `host_to_first_kernel_gap=8.153273`, `host_end_to_last_kernel_tail=12.242 ms`, `gpu_makespan=6.277 ms`, `gpu_kernel_sum=5.990 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=8192`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=0`, `sum_seq_after=8192`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 8192, 'prefix_len': 0, 'seq_len_after': 8192, 'prompt_len': 8192, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.268 ms
  纯GPU kernel时间: `0.266 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.268 ms`, `host_to_first_kernel_gap=7.974234`, `host_end_to_last_kernel_tail=7.976 ms`, `gpu_makespan=0.269 ms`, `gpu_kernel_sum=0.266 ms`
  开始时间(ns): `1093860669`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.064 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.064 ms`, `host_to_first_kernel_gap=7.91084`, `host_end_to_last_kernel_tail=7.864 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `1094194079`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.227 ms
  纯GPU kernel时间: `0.440 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.227 ms`, `host_to_first_kernel_gap=7.832994`, `host_end_to_last_kernel_tail=8.047 ms`, `gpu_makespan=0.442 ms`, `gpu_kernel_sum=0.440 ms`
  开始时间(ns): `1094290997`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.060 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.060 ms`, `host_to_first_kernel_gap=7.968302`, `host_end_to_last_kernel_tail=7.922 ms`, `gpu_makespan=0.014 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `1094597289`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.112 ms
  纯GPU kernel时间: `0.107 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.112 ms`, `host_to_first_kernel_gap=7.877382`, `host_end_to_last_kernel_tail=7.872 ms`, `gpu_makespan=0.107 ms`, `gpu_kernel_sum=0.107 ms`
  开始时间(ns): `1094703249`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.268 ms
  纯GPU kernel时间: `0.288 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.268 ms`, `host_to_first_kernel_gap=7.714765`, `host_end_to_last_kernel_tail=7.737 ms`, `gpu_makespan=0.290 ms`, `gpu_kernel_sum=0.288 ms`
  开始时间(ns): `1095001866`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[8192, 512]]}`
- `attn_mha` -> 0.177 ms
  纯GPU kernel时间: `3.608 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.177 ms`, `host_to_first_kernel_gap=7.874789`, `host_end_to_last_kernel_tail=11.306 ms`, `gpu_makespan=3.608 ms`, `gpu_kernel_sum=3.608 ms`
  开始时间(ns): `1095375698`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]}`
- `o_proj` -> 0.256 ms
  纯GPU kernel时间: `1.249 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.256 ms`, `host_to_first_kernel_gap=11.263632`, `host_end_to_last_kernel_tail=12.258 ms`, `gpu_makespan=1.250 ms`, `gpu_kernel_sum=1.249 ms`
  开始时间(ns): `1095597960`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 2 / prefill / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `2.054 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.054 ms`, `module_to_last_kernel=22.210 ms`, `host_to_first_kernel_gap=15.933765`, `host_end_to_last_kernel_tail=20.156 ms`, `gpu_makespan=6.276 ms`, `gpu_kernel_sum=5.992 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=8192`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=0`, `sum_seq_after=8192`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 8192, 'prefix_len': 0, 'seq_len_after': 8192, 'prompt_len': 8192, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.246 ms
  纯GPU kernel时间: `0.266 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.246 ms`, `host_to_first_kernel_gap=15.808035`, `host_end_to_last_kernel_tail=15.829 ms`, `gpu_makespan=0.267 ms`, `gpu_kernel_sum=0.266 ms`
  开始时间(ns): `1096921558`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.055 ms
  纯GPU kernel时间: `0.017 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.055 ms`, `host_to_first_kernel_gap=15.774992`, `host_end_to_last_kernel_tail=15.738 ms`, `gpu_makespan=0.017 ms`, `gpu_kernel_sum=0.017 ms`
  开始时间(ns): `1097221673`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.256 ms
  纯GPU kernel时间: `0.440 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.256 ms`, `host_to_first_kernel_gap=15.707232`, `host_end_to_last_kernel_tail=15.892 ms`, `gpu_makespan=0.441 ms`, `gpu_kernel_sum=0.440 ms`
  开始时间(ns): `1097309497`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.061 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.061 ms`, `host_to_first_kernel_gap=15.8235`, `host_end_to_last_kernel_tail=15.775 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `1097634413`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.079 ms
  纯GPU kernel时间: `0.107 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.079 ms`, `host_to_first_kernel_gap=15.733357`, `host_end_to_last_kernel_tail=15.761 ms`, `gpu_makespan=0.107 ms`, `gpu_kernel_sum=0.107 ms`
  开始时间(ns): `1097739340`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.246 ms
  纯GPU kernel时间: `0.287 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.246 ms`, `host_to_first_kernel_gap=15.638787`, `host_end_to_last_kernel_tail=15.681 ms`, `gpu_makespan=0.288 ms`, `gpu_kernel_sum=0.287 ms`
  开始时间(ns): `1097969110`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[8192, 512]]}`
- `attn_mha` -> 0.150 ms
  纯GPU kernel时间: `3.615 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.150 ms`, `host_to_first_kernel_gap=15.82405`, `host_end_to_last_kernel_tail=19.289 ms`, `gpu_makespan=3.615 ms`, `gpu_kernel_sum=3.615 ms`
  开始时间(ns): `1098316327`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]}`
- `o_proj` -> 0.322 ms
  纯GPU kernel时间: `1.247 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.322 ms`, `host_to_first_kernel_gap=19.248082`, `host_end_to_last_kernel_tail=20.175 ms`, `gpu_makespan=1.249 ms`, `gpu_kernel_sum=1.247 ms`
  开始时间(ns): `1098508936`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 3 / prefill / instance 1

- 执行序号: `4`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `2.115 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.115 ms`, `module_to_last_kernel=29.998 ms`, `host_to_first_kernel_gap=23.713534`, `host_end_to_last_kernel_tail=27.883 ms`, `gpu_makespan=6.284 ms`, `gpu_kernel_sum=6.002 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=8192`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=0`, `sum_seq_after=8192`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 8192, 'prefix_len': 0, 'seq_len_after': 8192, 'prompt_len': 8192, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.315 ms
  纯GPU kernel时间: `0.267 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.315 ms`, `host_to_first_kernel_gap=23.568968`, `host_end_to_last_kernel_tail=23.523 ms`, `gpu_makespan=0.269 ms`, `gpu_kernel_sum=0.267 ms`
  开始时间(ns): `1100060659`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.064 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.064 ms`, `host_to_first_kernel_gap=23.449423`, `host_end_to_last_kernel_tail=23.403 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `1100449292`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.247 ms
  纯GPU kernel时间: `0.437 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.247 ms`, `host_to_first_kernel_gap=23.360742`, `host_end_to_last_kernel_tail=23.554 ms`, `gpu_makespan=0.440 ms`, `gpu_kernel_sum=0.437 ms`
  开始时间(ns): `1100556661`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.055 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.055 ms`, `host_to_first_kernel_gap=23.47285`, `host_end_to_last_kernel_tail=23.431 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `1100884041`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.080 ms
  纯GPU kernel时间: `0.107 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.080 ms`, `host_to_first_kernel_gap=23.390329`, `host_end_to_last_kernel_tail=23.418 ms`, `gpu_makespan=0.107 ms`, `gpu_kernel_sum=0.107 ms`
  开始时间(ns): `1100980482`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.243 ms
  纯GPU kernel时间: `0.291 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.243 ms`, `host_to_first_kernel_gap=23.295879`, `host_end_to_last_kernel_tail=23.346 ms`, `gpu_makespan=0.293 ms`, `gpu_kernel_sum=0.291 ms`
  开始时间(ns): `1101210100`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[8192, 512]]}`
- `attn_mha` -> 0.150 ms
  纯GPU kernel时间: `3.620 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.150 ms`, `host_to_first_kernel_gap=23.472692`, `host_end_to_last_kernel_tail=26.943 ms`, `gpu_makespan=3.620 ms`, `gpu_kernel_sum=3.620 ms`
  开始时间(ns): `1101568552`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]}`
- `o_proj` -> 0.241 ms
  纯GPU kernel时间: `1.249 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.241 ms`, `host_to_first_kernel_gap=26.904651`, `host_end_to_last_kernel_tail=27.914 ms`, `gpu_makespan=1.250 ms`, `gpu_kernel_sum=1.249 ms`
  开始时间(ns): `1101759217`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 4 / prefill / instance 1

- 执行序号: `5`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `2.059 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.059 ms`, `module_to_last_kernel=42.944 ms`, `host_to_first_kernel_gap=36.640649`, `host_end_to_last_kernel_tail=40.885 ms`, `gpu_makespan=6.304 ms`, `gpu_kernel_sum=6.019 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=8192`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=0`, `sum_seq_after=8192`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 8192, 'prefix_len': 0, 'seq_len_after': 8192, 'prompt_len': 8192, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.301 ms
  纯GPU kernel时间: `0.270 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.301 ms`, `host_to_first_kernel_gap=36.517295`, `host_end_to_last_kernel_tail=36.489 ms`, `gpu_makespan=0.272 ms`, `gpu_kernel_sum=0.270 ms`
  开始时间(ns): `1103832880`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.051 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.051 ms`, `host_to_first_kernel_gap=36.43433`, `host_end_to_last_kernel_tail=36.402 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `1104188773`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.259 ms
  纯GPU kernel时间: `0.437 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.259 ms`, `host_to_first_kernel_gap=36.355083`, `host_end_to_last_kernel_tail=36.536 ms`, `gpu_makespan=0.440 ms`, `gpu_kernel_sum=0.437 ms`
  开始时间(ns): `1104287028`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.053 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.053 ms`, `host_to_first_kernel_gap=36.469091`, `host_end_to_last_kernel_tail=36.430 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `1104612508`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.080 ms
  纯GPU kernel时间: `0.107 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.080 ms`, `host_to_first_kernel_gap=36.389396`, `host_end_to_last_kernel_tail=36.417 ms`, `gpu_makespan=0.107 ms`, `gpu_kernel_sum=0.107 ms`
  开始时间(ns): `1104706667`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.241 ms
  纯GPU kernel时间: `0.295 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.241 ms`, `host_to_first_kernel_gap=36.303436`, `host_end_to_last_kernel_tail=36.360 ms`, `gpu_makespan=0.297 ms`, `gpu_kernel_sum=0.295 ms`
  开始时间(ns): `1104929331`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[8192, 512]]}`
- `attn_mha` -> 0.152 ms
  纯GPU kernel时间: `3.621 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.152 ms`, `host_to_first_kernel_gap=36.50429`, `host_end_to_last_kernel_tail=39.972 ms`, `gpu_makespan=3.620 ms`, `gpu_kernel_sum=3.621 ms`
  开始时间(ns): `1105268765`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]}`
- `o_proj` -> 0.290 ms
  纯GPU kernel时间: `1.258 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.290 ms`, `host_to_first_kernel_gap=39.931646`, `host_end_to_last_kernel_tail=40.901 ms`, `gpu_makespan=1.259 ms`, `gpu_kernel_sum=1.258 ms`
  开始时间(ns): `1105463362`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 0 / decode / instance 1

- 执行序号: `6`
- Collector对齐边界: `{'Module': 'model.model.layers.0.self_attn'}`
- 整块 MLA-module 时长: `2.019 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.019 ms`, `module_to_last_kernel=47.833 ms`, `host_to_first_kernel_gap=47.680584`, `host_end_to_last_kernel_tail=45.814 ms`, `gpu_makespan=0.153 ms`, `gpu_kernel_sum=0.121 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.355 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.355 ms`, `host_to_first_kernel_gap=47.595355`, `host_end_to_last_kernel_tail=47.257 ms`, `gpu_makespan=0.016 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `1110326247`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.060 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.060 ms`, `host_to_first_kernel_gap=47.184166`, `host_end_to_last_kernel_tail=47.126 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1110753340`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.034 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.034 ms`, `host_to_first_kernel_gap=47.098668`, `host_end_to_last_kernel_tail=47.067 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1110840758`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.252 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.252 ms`, `host_to_first_kernel_gap=47.026601`, `host_end_to_last_kernel_tail=46.795 ms`, `gpu_makespan=0.020 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `1110915769`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.084 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.084 ms`, `host_to_first_kernel_gap=46.625674`, `host_end_to_last_kernel_tail=46.544 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1111349048`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.326 ms
  纯GPU kernel时间: `0.032 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.326 ms`, `host_to_first_kernel_gap=46.495667`, `host_end_to_last_kernel_tail=46.203 ms`, `gpu_makespan=0.034 ms`, `gpu_kernel_sum=0.032 ms`
  开始时间(ns): `1111481999`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.279 ms
  纯GPU kernel时间: `0.050 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.279 ms`, `host_to_first_kernel_gap=46.060756`, `host_end_to_last_kernel_tail=45.833 ms`, `gpu_makespan=0.051 ms`, `gpu_kernel_sum=0.050 ms`
  开始时间(ns): `1111962478`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 1 / decode / instance 1

- 执行序号: `7`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `1.775 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.775 ms`, `module_to_last_kernel=45.232 ms`, `host_to_first_kernel_gap=45.081826`, `host_end_to_last_kernel_tail=43.457 ms`, `gpu_makespan=0.150 ms`, `gpu_kernel_sum=0.119 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.225 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.225 ms`, `host_to_first_kernel_gap=45.017787`, `host_end_to_last_kernel_tail=44.807 ms`, `gpu_makespan=0.015 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `1113219623`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.063 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.063 ms`, `host_to_first_kernel_gap=44.748009`, `host_end_to_last_kernel_tail=44.688 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1113504025`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.033 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.033 ms`, `host_to_first_kernel_gap=44.663213`, `host_end_to_last_kernel_tail=44.632 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1113591093`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.225 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.225 ms`, `host_to_first_kernel_gap=44.5979`, `host_end_to_last_kernel_tail=44.392 ms`, `gpu_makespan=0.019 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `1113659286`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.079 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.079 ms`, `host_to_first_kernel_gap=44.248656`, `host_end_to_last_kernel_tail=44.171 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1114039122`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.296 ms
  纯GPU kernel时间: `0.032 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.296 ms`, `host_to_first_kernel_gap=44.125348`, `host_end_to_last_kernel_tail=43.863 ms`, `gpu_makespan=0.033 ms`, `gpu_kernel_sum=0.032 ms`
  开始时间(ns): `1114165566`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.311 ms
  纯GPU kernel时间: `0.050 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.311 ms`, `host_to_first_kernel_gap=43.73731`, `host_end_to_last_kernel_tail=43.477 ms`, `gpu_makespan=0.051 ms`, `gpu_kernel_sum=0.050 ms`
  开始时间(ns): `1114599044`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 2 / decode / instance 1

- 执行序号: `8`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.719 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.719 ms`, `module_to_last_kernel=42.865 ms`, `host_to_first_kernel_gap=42.715148`, `host_end_to_last_kernel_tail=41.146 ms`, `gpu_makespan=0.150 ms`, `gpu_kernel_sum=0.118 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.221 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.221 ms`, `host_to_first_kernel_gap=42.650805`, `host_end_to_last_kernel_tail=42.446 ms`, `gpu_makespan=0.016 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `1115894669`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.063 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.063 ms`, `host_to_first_kernel_gap=42.386369`, `host_end_to_last_kernel_tail=42.325 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1116175073`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.032 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.032 ms`, `host_to_first_kernel_gap=42.301493`, `host_end_to_last_kernel_tail=42.272 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1116262125`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.225 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.225 ms`, `host_to_first_kernel_gap=42.238942`, `host_end_to_last_kernel_tail=42.033 ms`, `gpu_makespan=0.019 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `1116327684`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.082 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.082 ms`, `host_to_first_kernel_gap=41.899904`, `host_end_to_last_kernel_tail=41.820 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1116698050`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.287 ms
  纯GPU kernel时间: `0.031 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.287 ms`, `host_to_first_kernel_gap=41.777513`, `host_end_to_last_kernel_tail=41.524 ms`, `gpu_makespan=0.033 ms`, `gpu_kernel_sum=0.031 ms`
  开始时间(ns): `1116823481`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.269 ms
  纯GPU kernel时间: `0.049 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.269 ms`, `host_to_first_kernel_gap=41.407782`, `host_end_to_last_kernel_tail=41.189 ms`, `gpu_makespan=0.050 ms`, `gpu_kernel_sum=0.049 ms`
  开始时间(ns): `1117237501`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 3 / decode / instance 1

- 执行序号: `9`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `1.687 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.687 ms`, `module_to_last_kernel=40.592 ms`, `host_to_first_kernel_gap=40.440481`, `host_end_to_last_kernel_tail=38.906 ms`, `gpu_makespan=0.152 ms`, `gpu_kernel_sum=0.120 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.229 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.229 ms`, `host_to_first_kernel_gap=40.377903`, `host_end_to_last_kernel_tail=40.163 ms`, `gpu_makespan=0.015 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `1118476084`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.046 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.046 ms`, `host_to_first_kernel_gap=40.103076`, `host_end_to_last_kernel_tail=40.059 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1118766015`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.043 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.043 ms`, `host_to_first_kernel_gap=40.03479`, `host_end_to_last_kernel_tail=39.994 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1118836253`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.205 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.205 ms`, `host_to_first_kernel_gap=39.962042`, `host_end_to_last_kernel_tail=39.776 ms`, `gpu_makespan=0.019 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `1118913321`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.078 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.078 ms`, `host_to_first_kernel_gap=39.639716`, `host_end_to_last_kernel_tail=39.564 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1119267199`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.303 ms
  纯GPU kernel时间: `0.032 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.303 ms`, `host_to_first_kernel_gap=39.521142`, `host_end_to_last_kernel_tail=39.251 ms`, `gpu_makespan=0.033 ms`, `gpu_kernel_sum=0.032 ms`
  开始时间(ns): `1119388557`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.267 ms
  纯GPU kernel时间: `0.050 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.267 ms`, `host_to_first_kernel_gap=39.140407`, `host_end_to_last_kernel_tail=38.924 ms`, `gpu_makespan=0.051 ms`, `gpu_kernel_sum=0.050 ms`
  开始时间(ns): `1119814476`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 4 / decode / instance 1

- 执行序号: `10`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `1.851 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.851 ms`, `module_to_last_kernel=37.680 ms`, `host_to_first_kernel_gap=37.525969`, `host_end_to_last_kernel_tail=35.829 ms`, `gpu_makespan=0.154 ms`, `gpu_kernel_sum=0.123 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.292 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.292 ms`, `host_to_first_kernel_gap=37.456006`, `host_end_to_last_kernel_tail=37.180 ms`, `gpu_makespan=0.016 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `1121759069`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.049 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.049 ms`, `host_to_first_kernel_gap=37.107726`, `host_end_to_last_kernel_tail=37.061 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1122123477`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.044 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.044 ms`, `host_to_first_kernel_gap=37.035027`, `host_end_to_last_kernel_tail=36.993 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1122198416`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.236 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.236 ms`, `host_to_first_kernel_gap=36.956918`, `host_end_to_last_kernel_tail=36.741 ms`, `gpu_makespan=0.020 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `1122279853`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.082 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.082 ms`, `host_to_first_kernel_gap=36.5774`, `host_end_to_last_kernel_tail=36.497 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1122690635`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.301 ms
  纯GPU kernel时间: `0.031 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.301 ms`, `host_to_first_kernel_gap=36.452866`, `host_end_to_last_kernel_tail=36.184 ms`, `gpu_makespan=0.033 ms`, `gpu_kernel_sum=0.031 ms`
  开始时间(ns): `1122819265`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.261 ms
  纯GPU kernel时间: `0.051 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.261 ms`, `host_to_first_kernel_gap=36.071201`, `host_end_to_last_kernel_tail=35.862 ms`, `gpu_makespan=0.052 ms`, `gpu_kernel_sum=0.051 ms`
  开始时间(ns): `1123245762`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`
