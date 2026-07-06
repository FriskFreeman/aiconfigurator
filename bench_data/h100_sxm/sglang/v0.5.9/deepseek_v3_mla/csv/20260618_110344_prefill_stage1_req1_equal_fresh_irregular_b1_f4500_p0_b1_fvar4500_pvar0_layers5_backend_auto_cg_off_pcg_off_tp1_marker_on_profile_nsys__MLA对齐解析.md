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
- 整块 MLA-module 时长: `278.926 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=278.926 ms`, `module_to_last_kernel=280.019 ms`, `host_to_first_kernel_gap=0.563281`, `host_end_to_last_kernel_tail=1.092 ms`, `gpu_makespan=279.455 ms`, `gpu_kernel_sum=2.570 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4500`, `total_tokens=4500`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[4500, 128, 192], [4500, 128, 192], [4500, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=4500`, `sum_prefix=0`, `sum_seq_after=4500`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 4500, 'prefix_len': 0, 'seq_len_after': 4500, 'prompt_len': 4500, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.796 ms
  纯GPU kernel时间: `0.132 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.796 ms`, `host_to_first_kernel_gap=0.266266`, `host_end_to_last_kernel_tail=0.012 ms`, `gpu_makespan=0.541 ms`, `gpu_kernel_sum=0.132 ms`
  开始时间(ns): `1339792842`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4500, 7168]]}`
- `q_a_layernorm` -> 0.114 ms
  纯GPU kernel时间: `0.010 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.114 ms`, `host_to_first_kernel_gap=0.096493`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.010 ms`, `gpu_kernel_sum=0.010 ms`
  开始时间(ns): `1340723287`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4500, 1536]]}`
- `q_b_proj` -> 0.555 ms
  纯GPU kernel时间: `0.265 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.555 ms`, `host_to_first_kernel_gap=0.208449`, `host_end_to_last_kernel_tail=0.194 ms`, `gpu_makespan=0.540 ms`, `gpu_kernel_sum=0.265 ms`
  开始时间(ns): `1340913955`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4500, 1536]]}`
- `kv_a_layernorm` -> 0.099 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.099 ms`, `host_to_first_kernel_gap=0.084549`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `1341627583`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4500, 512]]}`
- `rotary_emb` -> 0.199 ms
  纯GPU kernel时间: `0.058 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.199 ms`, `host_to_first_kernel_gap=0.165384`, `host_end_to_last_kernel_tail=0.025 ms`, `gpu_makespan=0.058 ms`, `gpu_kernel_sum=0.058 ms`
  开始时间(ns): `1341822523`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4500], [4500, 128, 64], [4500, 1, 64]]}`
- `kv_b_proj` -> 1.032 ms
  纯GPU kernel时间: `0.156 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=1.032 ms`, `host_to_first_kernel_gap=0.455361`, `host_end_to_last_kernel_tail=0.060 ms`, `gpu_makespan=0.636 ms`, `gpu_kernel_sum=0.156 ms`
  开始时间(ns): `1615896273`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[4500, 512]]}`
- `attn_mha` -> 0.389 ms
  纯GPU kernel时间: `1.221 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.389 ms`, `host_to_first_kernel_gap=0.282728`, `host_end_to_last_kernel_tail=1.163 ms`, `gpu_makespan=1.269 ms`, `gpu_kernel_sum=1.221 ms`
  开始时间(ns): `1617241737`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4500, 128, 192], [4500, 128, 192], [4500, 128, 128]]}`
- `o_proj` -> 0.637 ms
  纯GPU kernel时间: `0.719 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.637 ms`, `host_to_first_kernel_gap=1.04582`, `host_end_to_last_kernel_tail=1.128 ms`, `gpu_makespan=0.720 ms`, `gpu_kernel_sum=0.719 ms`
  开始时间(ns): `1617748853`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4500, 16384]]}`

## Layer 1 / prefill / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `4.432 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=4.432 ms`, `module_to_last_kernel=5.653 ms`, `host_to_first_kernel_gap=1.657596`, `host_end_to_last_kernel_tail=1.220 ms`, `gpu_makespan=3.995 ms`, `gpu_kernel_sum=2.559 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4500`, `total_tokens=4500`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[4500, 128, 192], [4500, 128, 192], [4500, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=4500`, `sum_prefix=0`, `sum_seq_after=4500`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 4500, 'prefix_len': 0, 'seq_len_after': 4500, 'prompt_len': 4500, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.548 ms
  纯GPU kernel时间: `0.130 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.548 ms`, `host_to_first_kernel_gap=1.359762`, `host_end_to_last_kernel_tail=0.945 ms`, `gpu_makespan=0.133 ms`, `gpu_kernel_sum=0.130 ms`
  开始时间(ns): `1620781055`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4500, 7168]]}`
- `q_a_layernorm` -> 0.117 ms
  纯GPU kernel时间: `0.010 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.117 ms`, `host_to_first_kernel_gap=0.819098`, `host_end_to_last_kernel_tail=0.711 ms`, `gpu_makespan=0.010 ms`, `gpu_kernel_sum=0.010 ms`
  开始时间(ns): `1621455127`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4500, 1536]]}`
- `q_b_proj` -> 0.477 ms
  纯GPU kernel时间: `0.260 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.477 ms`, `host_to_first_kernel_gap=0.63454`, `host_end_to_last_kernel_tail=0.419 ms`, `gpu_makespan=0.261 ms`, `gpu_kernel_sum=0.260 ms`
  开始时间(ns): `1621650533`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4500, 1536]]}`
- `kv_a_layernorm` -> 0.099 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.099 ms`, `host_to_first_kernel_gap=0.274776`, `host_end_to_last_kernel_tail=0.185 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `1622271481`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4500, 512]]}`
- `rotary_emb` -> 0.195 ms
  纯GPU kernel时间: `0.059 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.195 ms`, `host_to_first_kernel_gap=0.163753`, `host_end_to_last_kernel_tail=0.028 ms`, `gpu_makespan=0.059 ms`, `gpu_kernel_sum=0.059 ms`
  开始时间(ns): `1622466504`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4500], [4500, 128, 64], [4500, 1, 64]]}`
- `kv_b_proj` -> 0.689 ms
  纯GPU kernel时间: `0.154 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.689 ms`, `host_to_first_kernel_gap=0.321591`, `host_end_to_last_kernel_tail=0.085 ms`, `gpu_makespan=0.453 ms`, `gpu_kernel_sum=0.154 ms`
  开始时间(ns): `1622989946`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[4500, 512]]}`
- `attn_mha` -> 0.337 ms
  纯GPU kernel时间: `1.221 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.337 ms`, `host_to_first_kernel_gap=0.241771`, `host_end_to_last_kernel_tail=1.173 ms`, `gpu_makespan=1.268 ms`, `gpu_kernel_sum=1.221 ms`
  开始时间(ns): `1623908198`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4500, 128, 192], [4500, 128, 192], [4500, 128, 128]]}`
- `o_proj` -> 0.541 ms
  纯GPU kernel时间: `0.715 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.541 ms`, `host_to_first_kernel_gap=1.075458`, `host_end_to_last_kernel_tail=1.251 ms`, `gpu_makespan=0.717 ms`, `gpu_kernel_sum=0.715 ms`
  开始时间(ns): `1624343663`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4500, 16384]]}`

## Layer 2 / prefill / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `4.021 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=4.021 ms`, `module_to_last_kernel=5.278 ms`, `host_to_first_kernel_gap=1.972635`, `host_end_to_last_kernel_tail=1.257 ms`, `gpu_makespan=3.305 ms`, `gpu_kernel_sum=2.569 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4500`, `total_tokens=4500`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[4500, 128, 192], [4500, 128, 192], [4500, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=4500`, `sum_prefix=0`, `sum_seq_after=4500`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 4500, 'prefix_len': 0, 'seq_len_after': 4500, 'prompt_len': 4500, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.511 ms
  纯GPU kernel时间: `0.132 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.511 ms`, `host_to_first_kernel_gap=1.734209`, `host_end_to_last_kernel_tail=1.356 ms`, `gpu_makespan=0.133 ms`, `gpu_kernel_sum=0.132 ms`
  开始时间(ns): `1627030512`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4500, 7168]]}`
- `q_a_layernorm` -> 0.112 ms
  纯GPU kernel时间: `0.010 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.112 ms`, `host_to_first_kernel_gap=1.245701`, `host_end_to_last_kernel_tail=1.144 ms`, `gpu_makespan=0.010 ms`, `gpu_kernel_sum=0.010 ms`
  开始时间(ns): `1627652268`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4500, 1536]]}`
- `q_b_proj` -> 0.509 ms
  纯GPU kernel时间: `0.259 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.509 ms`, `host_to_first_kernel_gap=1.074011`, `host_end_to_last_kernel_tail=0.825 ms`, `gpu_makespan=0.260 ms`, `gpu_kernel_sum=0.259 ms`
  开始时间(ns): `1627835574`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4500, 1536]]}`
- `kv_a_layernorm` -> 0.097 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.097 ms`, `host_to_first_kernel_gap=0.682155`, `host_end_to_last_kernel_tail=0.594 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `1628486854`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4500, 512]]}`
- `rotary_emb` -> 0.162 ms
  纯GPU kernel时间: `0.059 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.162 ms`, `host_to_first_kernel_gap=0.509028`, `host_end_to_last_kernel_tail=0.406 ms`, `gpu_makespan=0.059 ms`, `gpu_kernel_sum=0.059 ms`
  开始时间(ns): `1628670317`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4500], [4500, 128, 64], [4500, 1, 64]]}`
- `kv_b_proj` -> 0.499 ms
  纯GPU kernel时间: `0.158 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.499 ms`, `host_to_first_kernel_gap=0.179907`, `host_end_to_last_kernel_tail=0.099 ms`, `gpu_makespan=0.418 ms`, `gpu_kernel_sum=0.158 ms`
  开始时间(ns): `1629141326`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[4500, 512]]}`
- `attn_mha` -> 0.316 ms
  纯GPU kernel时间: `1.227 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.316 ms`, `host_to_first_kernel_gap=0.231825`, `host_end_to_last_kernel_tail=1.180 ms`, `gpu_makespan=1.265 ms`, `gpu_kernel_sum=1.227 ms`
  开始时间(ns): `1629855360`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4500, 128, 192], [4500, 128, 192], [4500, 128, 128]]}`
- `o_proj` -> 0.510 ms
  纯GPU kernel时间: `0.715 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.510 ms`, `host_to_first_kernel_gap=1.081495`, `host_end_to_last_kernel_tail=1.288 ms`, `gpu_makespan=0.717 ms`, `gpu_kernel_sum=0.715 ms`
  开始时间(ns): `1630271738`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4500, 16384]]}`

## Layer 3 / prefill / instance 1

- 执行序号: `4`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `3.926 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=3.926 ms`, `module_to_last_kernel=5.183 ms`, `host_to_first_kernel_gap=2.066041`, `host_end_to_last_kernel_tail=1.257 ms`, `gpu_makespan=3.117 ms`, `gpu_kernel_sum=2.560 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4500`, `total_tokens=4500`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[4500, 128, 192], [4500, 128, 192], [4500, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=4500`, `sum_prefix=0`, `sum_seq_after=4500`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 4500, 'prefix_len': 0, 'seq_len_after': 4500, 'prompt_len': 4500, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.494 ms
  纯GPU kernel时间: `0.132 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.494 ms`, `host_to_first_kernel_gap=1.847039`, `host_end_to_last_kernel_tail=1.487 ms`, `gpu_makespan=0.133 ms`, `gpu_kernel_sum=0.132 ms`
  开始时间(ns): `1632849329`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4500, 7168]]}`
- `q_a_layernorm` -> 0.110 ms
  纯GPU kernel时间: `0.010 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.110 ms`, `host_to_first_kernel_gap=1.376614`, `host_end_to_last_kernel_tail=1.277 ms`, `gpu_makespan=0.010 ms`, `gpu_kernel_sum=0.010 ms`
  开始时间(ns): `1633452810`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4500, 1536]]}`
- `q_b_proj` -> 0.484 ms
  纯GPU kernel时间: `0.258 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.484 ms`, `host_to_first_kernel_gap=1.208636`, `host_end_to_last_kernel_tail=0.984 ms`, `gpu_makespan=0.259 ms`, `gpu_kernel_sum=0.258 ms`
  开始时间(ns): `1633631924`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4500, 1536]]}`
- `kv_a_layernorm` -> 0.097 ms
  纯GPU kernel时间: `0.010 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.097 ms`, `host_to_first_kernel_gap=0.84923`, `host_end_to_last_kernel_tail=0.762 ms`, `gpu_makespan=0.010 ms`, `gpu_kernel_sum=0.010 ms`
  开始时间(ns): `1634250146`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4500, 512]]}`
- `rotary_emb` -> 0.157 ms
  纯GPU kernel时间: `0.058 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.157 ms`, `host_to_first_kernel_gap=0.677242`, `host_end_to_last_kernel_tail=0.578 ms`, `gpu_makespan=0.058 ms`, `gpu_kernel_sum=0.058 ms`
  开始时间(ns): `1634433174`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4500], [4500, 128, 64], [4500, 1, 64]]}`
- `kv_b_proj` -> 0.502 ms
  纯GPU kernel时间: `0.153 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.502 ms`, `host_to_first_kernel_gap=0.298803`, `host_end_to_last_kernel_tail=0.095 ms`, `gpu_makespan=0.298 ms`, `gpu_kernel_sum=0.153 ms`
  开始时间(ns): `1634887229`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[4500, 512]]}`
- `attn_mha` -> 0.313 ms
  纯GPU kernel时间: `1.222 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.313 ms`, `host_to_first_kernel_gap=0.229008`, `host_end_to_last_kernel_tail=1.175 ms`, `gpu_makespan=1.259 ms`, `gpu_kernel_sum=1.222 ms`
  开始时间(ns): `1635605184`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4500, 128, 192], [4500, 128, 192], [4500, 128, 128]]}`
- `o_proj` -> 0.514 ms
  纯GPU kernel时间: `0.718 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.514 ms`, `host_to_first_kernel_gap=1.080822`, `host_end_to_last_kernel_tail=1.286 ms`, `gpu_makespan=0.719 ms`, `gpu_kernel_sum=0.718 ms`
  开始时间(ns): `1636013306`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4500, 16384]]}`

## Layer 4 / prefill / instance 1

- 执行序号: `5`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `4.139 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=4.139 ms`, `module_to_last_kernel=7.266 ms`, `host_to_first_kernel_gap=4.52975`, `host_end_to_last_kernel_tail=3.127 ms`, `gpu_makespan=2.736 ms`, `gpu_kernel_sum=2.572 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4500`, `total_tokens=4500`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[4500, 128, 192], [4500, 128, 192], [4500, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=4500`, `sum_prefix=0`, `sum_seq_after=4500`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 4500, 'prefix_len': 0, 'seq_len_after': 4500, 'prompt_len': 4500, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.617 ms
  纯GPU kernel时间: `0.131 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.617 ms`, `host_to_first_kernel_gap=4.283856`, `host_end_to_last_kernel_tail=3.801 ms`, `gpu_makespan=0.134 ms`, `gpu_kernel_sum=0.131 ms`
  开始时间(ns): `1639933728`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4500, 7168]]}`
- `q_a_layernorm` -> 0.111 ms
  纯GPU kernel时间: `0.010 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.111 ms`, `host_to_first_kernel_gap=3.682045`, `host_end_to_last_kernel_tail=3.582 ms`, `gpu_makespan=0.010 ms`, `gpu_kernel_sum=0.010 ms`
  开始时间(ns): `1640669939`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4500, 1536]]}`
- `q_b_proj` -> 0.518 ms
  纯GPU kernel时间: `0.259 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.518 ms`, `host_to_first_kernel_gap=3.510118`, `host_end_to_last_kernel_tail=3.253 ms`, `gpu_makespan=0.261 ms`, `gpu_kernel_sum=0.259 ms`
  开始时间(ns): `1640853322`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4500, 1536]]}`
- `kv_a_layernorm` -> 0.093 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.093 ms`, `host_to_first_kernel_gap=3.11037`, `host_end_to_last_kernel_tail=3.026 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `1641513870`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4500, 512]]}`
- `rotary_emb` -> 0.164 ms
  纯GPU kernel时间: `0.058 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.164 ms`, `host_to_first_kernel_gap=2.941959`, `host_end_to_last_kernel_tail=2.837 ms`, `gpu_makespan=0.058 ms`, `gpu_kernel_sum=0.058 ms`
  开始时间(ns): `1641692457`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4500], [4500, 128, 64], [4500, 1, 64]]}`
- `kv_b_proj` -> 0.511 ms
  纯GPU kernel时间: `0.162 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.511 ms`, `host_to_first_kernel_gap=2.566752`, `host_end_to_last_kernel_tail=2.219 ms`, `gpu_makespan=0.164 ms`, `gpu_kernel_sum=0.162 ms`
  开始时间(ns): `1642144400`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[4500, 512]]}`
- `attn_mha` -> 0.324 ms
  纯GPU kernel时间: `1.227 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.324 ms`, `host_to_first_kernel_gap=2.133918`, `host_end_to_last_kernel_tail=3.037 ms`, `gpu_makespan=1.227 ms`, `gpu_kernel_sum=1.227 ms`
  开始时间(ns): `1642874258`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4500, 128, 192], [4500, 128, 192], [4500, 128, 128]]}`
- `o_proj` -> 0.503 ms
  纯GPU kernel时间: `0.715 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.503 ms`, `host_to_first_kernel_gap=2.943447`, `host_end_to_last_kernel_tail=3.157 ms`, `gpu_makespan=0.717 ms`, `gpu_kernel_sum=0.715 ms`
  开始时间(ns): `1643293849`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4500, 16384]]}`

## Layer 0 / decode / instance 1

- 执行序号: `6`
- Collector对齐边界: `{'Module': 'model.model.layers.0.self_attn'}`
- 整块 MLA-module 时长: `3.744 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=3.744 ms`, `module_to_last_kernel=3.744 ms`, `host_to_first_kernel_gap=1.780994`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=1.903 ms`, `gpu_kernel_sum=0.117 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.649 ms
  纯GPU kernel时间: `0.017 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.649 ms`, `host_to_first_kernel_gap=1.644684`, `host_end_to_last_kernel_tail=1.014 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.017 ms`
  开始时间(ns): `1651867395`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.100 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.100 ms`, `host_to_first_kernel_gap=0.886737`, `host_end_to_last_kernel_tail=0.788 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1652643166`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.066 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.066 ms`, `host_to_first_kernel_gap=0.737896`, `host_end_to_last_kernel_tail=0.674 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1652793895`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.496 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.496 ms`, `host_to_first_kernel_gap=0.596292`, `host_end_to_last_kernel_tail=0.120 ms`, `gpu_makespan=0.021 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `1652938379`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.160 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.160 ms`, `host_to_first_kernel_gap=0.133219`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1653760652`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.582 ms
  纯GPU kernel时间: `0.027 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.582 ms`, `host_to_first_kernel_gap=0.18612`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.340 ms`, `gpu_kernel_sum=0.027 ms`
  开始时间(ns): `1654013767`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.576 ms
  纯GPU kernel时间: `0.049 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.576 ms`, `host_to_first_kernel_gap=0.196709`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.357 ms`, `gpu_kernel_sum=0.049 ms`
  开始时间(ns): `1654861898`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 1 / decode / instance 1

- 执行序号: `7`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `2.967 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.967 ms`, `module_to_last_kernel=2.967 ms`, `host_to_first_kernel_gap=0.267476`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=2.669 ms`, `gpu_kernel_sum=0.114 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.400 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.400 ms`, `host_to_first_kernel_gap=0.150564`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.221 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `1657176587`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.082 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.082 ms`, `host_to_first_kernel_gap=0.068817`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1657681310`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.057 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.057 ms`, `host_to_first_kernel_gap=0.048286`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1657808145`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.386 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.386 ms`, `host_to_first_kernel_gap=0.142373`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.215 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `1657938250`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.141 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.141 ms`, `host_to_first_kernel_gap=0.118317`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1658578658`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.512 ms
  纯GPU kernel时间: `0.027 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.512 ms`, `host_to_first_kernel_gap=0.154765`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.312 ms`, `gpu_kernel_sum=0.027 ms`
  开始时间(ns): `1658799714`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.456 ms
  纯GPU kernel时间: `0.049 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.456 ms`, `host_to_first_kernel_gap=0.172092`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.283 ms`, `gpu_kernel_sum=0.049 ms`
  开始时间(ns): `1659541043`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 2 / decode / instance 1

- 执行序号: `8`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `2.657 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.657 ms`, `module_to_last_kernel=2.657 ms`, `host_to_first_kernel_gap=0.242642`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=2.380 ms`, `gpu_kernel_sum=0.114 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.359 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.359 ms`, `host_to_first_kernel_gap=0.135645`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.200 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `1661543922`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.084 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.084 ms`, `host_to_first_kernel_gap=0.063209`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1661999334`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.054 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.054 ms`, `host_to_first_kernel_gap=0.045554`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1662123133`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.348 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.348 ms`, `host_to_first_kernel_gap=0.126936`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.202 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `1662233335`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.116 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.116 ms`, `host_to_first_kernel_gap=0.098083`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1662797516`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.453 ms
  纯GPU kernel时间: `0.027 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.453 ms`, `host_to_first_kernel_gap=0.143549`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.275 ms`, `gpu_kernel_sum=0.027 ms`
  开始时间(ns): `1662988818`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.423 ms
  纯GPU kernel时间: `0.049 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.423 ms`, `host_to_first_kernel_gap=0.154912`, `host_end_to_last_kernel_tail=0.005 ms`, `gpu_makespan=0.273 ms`, `gpu_kernel_sum=0.049 ms`
  开始时间(ns): `1663631950`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 3 / decode / instance 1

- 执行序号: `9`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `2.443 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.443 ms`, `module_to_last_kernel=2.443 ms`, `host_to_first_kernel_gap=0.224533`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=2.203 ms`, `gpu_kernel_sum=0.114 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.341 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.341 ms`, `host_to_first_kernel_gap=0.130244`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.189 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `1665484458`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.070 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.070 ms`, `host_to_first_kernel_gap=0.059606`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1665913016`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.055 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.055 ms`, `host_to_first_kernel_gap=0.041944`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1666018486`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.322 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.322 ms`, `host_to_first_kernel_gap=0.119749`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.187 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `1666123977`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.111 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.111 ms`, `host_to_first_kernel_gap=0.092674`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1666655500`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.429 ms
  纯GPU kernel时间: `0.027 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.429 ms`, `host_to_first_kernel_gap=0.131095`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.262 ms`, `gpu_kernel_sum=0.027 ms`
  开始时间(ns): `1666831351`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.374 ms
  纯GPU kernel时间: `0.049 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.374 ms`, `host_to_first_kernel_gap=0.134773`, `host_end_to_last_kernel_tail=0.009 ms`, `gpu_makespan=0.249 ms`, `gpu_kernel_sum=0.049 ms`
  开始时间(ns): `1667434329`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 4 / decode / instance 1

- 执行序号: `10`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `2.374 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.374 ms`, `module_to_last_kernel=2.374 ms`, `host_to_first_kernel_gap=0.244283`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=2.118 ms`, `gpu_kernel_sum=0.116 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.405 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.405 ms`, `host_to_first_kernel_gap=0.151935`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.227 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `1669961999`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.070 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.070 ms`, `host_to_first_kernel_gap=0.059816`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1670457510`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.046 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.046 ms`, `host_to_first_kernel_gap=0.039624`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1670564550`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.294 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.294 ms`, `host_to_first_kernel_gap=0.105632`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.174 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `1670662446`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.105 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.105 ms`, `host_to_first_kernel_gap=0.088939`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1671159683`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.378 ms
  纯GPU kernel时间: `0.027 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.378 ms`, `host_to_first_kernel_gap=0.121203`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.228 ms`, `gpu_kernel_sum=0.027 ms`
  开始时间(ns): `1671325755`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.352 ms
  纯GPU kernel时间: `0.049 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.352 ms`, `host_to_first_kernel_gap=0.118477`, `host_end_to_last_kernel_tail=0.013 ms`, `gpu_makespan=0.247 ms`, `gpu_kernel_sum=0.049 ms`
  开始时间(ns): `1671866817`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`
