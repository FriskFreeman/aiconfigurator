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
- 整块 MLA-module 时长: `1.417 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.417 ms`, `module_to_last_kernel=2.625 ms`, `host_to_first_kernel_gap=0.201189`, `host_end_to_last_kernel_tail=1.208 ms`, `gpu_makespan=2.424 ms`, `gpu_kernel_sum=1.987 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4800`, `total_tokens=4800`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[4800, 128, 192], [4800, 128, 192], [4800, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=4800`, `sum_prefix=0`, `sum_seq_after=4800`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 1200, 'prefix_len': 0, 'seq_len_after': 1200, 'prompt_len': 1200, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 1200, 'prefix_len': 0, 'seq_len_after': 1200, 'prompt_len': 1200, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 1200, 'prefix_len': 0, 'seq_len_after': 1200, 'prompt_len': 1200, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 1200, 'prefix_len': 0, 'seq_len_after': 1200, 'prompt_len': 1200, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.255 ms
  纯GPU kernel时间: `0.165 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.255 ms`, `host_to_first_kernel_gap=0.084286`, `host_end_to_last_kernel_tail=0.103 ms`, `gpu_makespan=0.274 ms`, `gpu_kernel_sum=0.165 ms`
  开始时间(ns): `27432154`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4800, 7168]]}`
- `q_a_layernorm` -> 0.032 ms
  纯GPU kernel时间: `0.010 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.032 ms`, `host_to_first_kernel_gap=0.061023`, `host_end_to_last_kernel_tail=0.040 ms`, `gpu_makespan=0.010 ms`, `gpu_kernel_sum=0.010 ms`
  开始时间(ns): `27729145`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4800, 1536]]}`
- `q_b_proj` -> 0.147 ms
  纯GPU kernel时间: `0.275 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.147 ms`, `host_to_first_kernel_gap=0.054645`, `host_end_to_last_kernel_tail=0.251 ms`, `gpu_makespan=0.344 ms`, `gpu_kernel_sum=0.275 ms`
  开始时间(ns): `27783395`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4800, 1536]]}`
- `kv_a_layernorm` -> 0.026 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.026 ms`, `host_to_first_kernel_gap=0.212185`, `host_end_to_last_kernel_tail=0.195 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `27969631`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4800, 512]]}`
- `rotary_emb` -> 0.071 ms
  纯GPU kernel时间: `0.062 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.071 ms`, `host_to_first_kernel_gap=0.170015`, `host_end_to_last_kernel_tail=0.160 ms`, `gpu_makespan=0.062 ms`, `gpu_kernel_sum=0.062 ms`
  开始时间(ns): `28021945`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4800], [4800, 128, 64], [4800, 1, 64]]}`
- `kv_b_proj` -> 0.151 ms
  纯GPU kernel时间: `0.166 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.151 ms`, `host_to_first_kernel_gap=0.074652`, `host_end_to_last_kernel_tail=0.145 ms`, `gpu_makespan=0.222 ms`, `gpu_kernel_sum=0.166 ms`
  开始时间(ns): `28197084`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[4800, 512]]}`
- `attn_mha` -> 0.126 ms
  纯GPU kernel时间: `0.503 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.126 ms`, `host_to_first_kernel_gap=0.212448`, `host_end_to_last_kernel_tail=0.589 ms`, `gpu_makespan=0.503 ms`, `gpu_kernel_sum=0.503 ms`
  开始时间(ns): `28424696`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4800, 128, 192], [4800, 128, 192], [4800, 128, 128]]}`
- `o_proj` -> 0.144 ms
  纯GPU kernel时间: `0.797 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.144 ms`, `host_to_first_kernel_gap=0.561639`, `host_end_to_last_kernel_tail=1.216 ms`, `gpu_makespan=0.798 ms`, `gpu_kernel_sum=0.797 ms`
  开始时间(ns): `28580273`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4800, 16384]]}`

## Layer 1 / prefill / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `1.096 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.096 ms`, `module_to_last_kernel=5.658 ms`, `host_to_first_kernel_gap=3.502963`, `host_end_to_last_kernel_tail=4.562 ms`, `gpu_makespan=2.155 ms`, `gpu_kernel_sum=1.978 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4800`, `total_tokens=4800`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[4800, 128, 192], [4800, 128, 192], [4800, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=4800`, `sum_prefix=0`, `sum_seq_after=4800`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 1200, 'prefix_len': 0, 'seq_len_after': 1200, 'prompt_len': 1200, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 1200, 'prefix_len': 0, 'seq_len_after': 1200, 'prompt_len': 1200, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 1200, 'prefix_len': 0, 'seq_len_after': 1200, 'prompt_len': 1200, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 1200, 'prefix_len': 0, 'seq_len_after': 1200, 'prompt_len': 1200, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.137 ms
  纯GPU kernel时间: `0.162 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.137 ms`, `host_to_first_kernel_gap=3.419329`, `host_end_to_last_kernel_tail=3.446 ms`, `gpu_makespan=0.164 ms`, `gpu_kernel_sum=0.162 ms`
  开始时间(ns): `29345110`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4800, 7168]]}`
- `q_a_layernorm` -> 0.028 ms
  纯GPU kernel时间: `0.011 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.028 ms`, `host_to_first_kernel_gap=3.416986`, `host_end_to_last_kernel_tail=3.400 ms`, `gpu_makespan=0.011 ms`, `gpu_kernel_sum=0.011 ms`
  开始时间(ns): `29511901`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4800, 1536]]}`
- `q_b_proj` -> 0.118 ms
  纯GPU kernel时间: `0.270 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.118 ms`, `host_to_first_kernel_gap=3.382785`, `host_end_to_last_kernel_tail=3.536 ms`, `gpu_makespan=0.272 ms`, `gpu_kernel_sum=0.270 ms`
  开始时间(ns): `29558134`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4800, 1536]]}`
- `kv_a_layernorm` -> 0.024 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.024 ms`, `host_to_first_kernel_gap=3.502395`, `host_end_to_last_kernel_tail=3.487 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `29710524`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4800, 512]]}`
- `rotary_emb` -> 0.050 ms
  纯GPU kernel时间: `0.061 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.050 ms`, `host_to_first_kernel_gap=3.465955`, `host_end_to_last_kernel_tail=3.477 ms`, `gpu_makespan=0.061 ms`, `gpu_kernel_sum=0.061 ms`
  开始时间(ns): `29757364`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4800], [4800, 128, 64], [4800, 1, 64]]}`
- `kv_b_proj` -> 0.141 ms
  纯GPU kernel时间: `0.165 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.141 ms`, `host_to_first_kernel_gap=3.411055`, `host_end_to_last_kernel_tail=3.436 ms`, `gpu_makespan=0.167 ms`, `gpu_kernel_sum=0.165 ms`
  开始时间(ns): `29892712`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[4800, 512]]}`
- `attn_mha` -> 0.096 ms
  纯GPU kernel时间: `0.503 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.096 ms`, `host_to_first_kernel_gap=3.523341`, `host_end_to_last_kernel_tail=3.931 ms`, `gpu_makespan=0.503 ms`, `gpu_kernel_sum=0.503 ms`
  开始时间(ns): `30091562`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4800, 128, 192], [4800, 128, 192], [4800, 128, 128]]}`
- `o_proj` -> 0.138 ms
  纯GPU kernel时间: `0.797 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.138 ms`, `host_to_first_kernel_gap=3.909947`, `host_end_to_last_kernel_tail=4.570 ms`, `gpu_makespan=0.799 ms`, `gpu_kernel_sum=0.797 ms`
  开始时间(ns): `30210844`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4800, 16384]]}`

## Layer 2 / prefill / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.073 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.073 ms`, `module_to_last_kernel=9.061 ms`, `host_to_first_kernel_gap=6.912201`, `host_end_to_last_kernel_tail=7.988 ms`, `gpu_makespan=2.148 ms`, `gpu_kernel_sum=1.974 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4800`, `total_tokens=4800`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[4800, 128, 192], [4800, 128, 192], [4800, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=4800`, `sum_prefix=0`, `sum_seq_after=4800`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 1200, 'prefix_len': 0, 'seq_len_after': 1200, 'prompt_len': 1200, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 1200, 'prefix_len': 0, 'seq_len_after': 1200, 'prompt_len': 1200, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 1200, 'prefix_len': 0, 'seq_len_after': 1200, 'prompt_len': 1200, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 1200, 'prefix_len': 0, 'seq_len_after': 1200, 'prompt_len': 1200, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.145 ms
  纯GPU kernel时间: `0.162 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.145 ms`, `host_to_first_kernel_gap=6.836819`, `host_end_to_last_kernel_tail=6.855 ms`, `gpu_makespan=0.163 ms`, `gpu_kernel_sum=0.162 ms`
  开始时间(ns): `30911043`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4800, 7168]]}`
- `q_a_layernorm` -> 0.027 ms
  纯GPU kernel时间: `0.010 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.027 ms`, `host_to_first_kernel_gap=6.824863`, `host_end_to_last_kernel_tail=6.808 ms`, `gpu_makespan=0.010 ms`, `gpu_kernel_sum=0.010 ms`
  开始时间(ns): `31085943`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4800, 1536]]}`
- `q_b_proj` -> 0.136 ms
  纯GPU kernel时间: `0.269 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.136 ms`, `host_to_first_kernel_gap=6.791822`, `host_end_to_last_kernel_tail=6.926 ms`, `gpu_makespan=0.270 ms`, `gpu_kernel_sum=0.269 ms`
  开始时间(ns): `31131432`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4800, 1536]]}`
- `kv_a_layernorm` -> 0.025 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.025 ms`, `host_to_first_kernel_gap=6.890497`, `host_end_to_last_kernel_tail=6.874 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `31302933`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4800, 512]]}`
- `rotary_emb` -> 0.047 ms
  纯GPU kernel时间: `0.062 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.047 ms`, `host_to_first_kernel_gap=6.853643`, `host_end_to_last_kernel_tail=6.868 ms`, `gpu_makespan=0.062 ms`, `gpu_kernel_sum=0.062 ms`
  开始时间(ns): `31350795`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4800], [4800, 128, 64], [4800, 1, 64]]}`
- `kv_b_proj` -> 0.131 ms
  纯GPU kernel时间: `0.162 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.131 ms`, `host_to_first_kernel_gap=6.804855`, `host_end_to_last_kernel_tail=6.837 ms`, `gpu_makespan=0.163 ms`, `gpu_kernel_sum=0.162 ms`
  开始时间(ns): `31479711`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[4800, 512]]}`
- `attn_mha` -> 0.084 ms
  纯GPU kernel时间: `0.505 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.084 ms`, `host_to_first_kernel_gap=6.924608`, `host_end_to_last_kernel_tail=7.346 ms`, `gpu_makespan=0.505 ms`, `gpu_kernel_sum=0.505 ms`
  开始时间(ns): `31666806`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4800, 128, 192], [4800, 128, 192], [4800, 128, 128]]}`
- `o_proj` -> 0.126 ms
  纯GPU kernel时间: `0.796 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.126 ms`, `host_to_first_kernel_gap=7.324203`, `host_end_to_last_kernel_tail=7.996 ms`, `gpu_makespan=0.798 ms`, `gpu_kernel_sum=0.796 ms`
  开始时间(ns): `31774187`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4800, 16384]]}`

## Layer 3 / prefill / instance 1

- 执行序号: `4`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `1.047 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.047 ms`, `module_to_last_kernel=12.504 ms`, `host_to_first_kernel_gap=10.354648`, `host_end_to_last_kernel_tail=11.457 ms`, `gpu_makespan=2.149 ms`, `gpu_kernel_sum=1.974 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4800`, `total_tokens=4800`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[4800, 128, 192], [4800, 128, 192], [4800, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=4800`, `sum_prefix=0`, `sum_seq_after=4800`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 1200, 'prefix_len': 0, 'seq_len_after': 1200, 'prompt_len': 1200, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 1200, 'prefix_len': 0, 'seq_len_after': 1200, 'prompt_len': 1200, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 1200, 'prefix_len': 0, 'seq_len_after': 1200, 'prompt_len': 1200, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 1200, 'prefix_len': 0, 'seq_len_after': 1200, 'prompt_len': 1200, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.124 ms
  纯GPU kernel时间: `0.162 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.124 ms`, `host_to_first_kernel_gap=10.280704`, `host_end_to_last_kernel_tail=10.321 ms`, `gpu_makespan=0.165 ms`, `gpu_kernel_sum=0.162 ms`
  开始时间(ns): `32443029`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4800, 7168]]}`
- `q_a_layernorm` -> 0.027 ms
  纯GPU kernel时间: `0.010 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.027 ms`, `host_to_first_kernel_gap=10.292835`, `host_end_to_last_kernel_tail=10.276 ms`, `gpu_makespan=0.010 ms`, `gpu_kernel_sum=0.010 ms`
  开始时间(ns): `32595730`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4800, 1536]]}`
- `q_b_proj` -> 0.115 ms
  纯GPU kernel时间: `0.268 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.115 ms`, `host_to_first_kernel_gap=10.260478`, `host_end_to_last_kernel_tail=10.417 ms`, `gpu_makespan=0.271 ms`, `gpu_kernel_sum=0.268 ms`
  开始时间(ns): `32639991`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4800, 1536]]}`
- `kv_a_layernorm` -> 0.023 ms
  纯GPU kernel时间: `0.010 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.023 ms`, `host_to_first_kernel_gap=10.384368`, `host_end_to_last_kernel_tail=10.371 ms`, `gpu_makespan=0.010 ms`, `gpu_kernel_sum=0.010 ms`
  开始时间(ns): `32786629`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4800, 512]]}`
- `rotary_emb` -> 0.049 ms
  纯GPU kernel时间: `0.062 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.049 ms`, `host_to_first_kernel_gap=10.350409`, `host_end_to_last_kernel_tail=10.364 ms`, `gpu_makespan=0.062 ms`, `gpu_kernel_sum=0.062 ms`
  开始时间(ns): `32831820`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4800], [4800, 128, 64], [4800, 1, 64]]}`
- `kv_b_proj` -> 0.136 ms
  纯GPU kernel时间: `0.160 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.136 ms`, `host_to_first_kernel_gap=10.300704`, `host_end_to_last_kernel_tail=10.326 ms`, `gpu_makespan=0.161 ms`, `gpu_kernel_sum=0.160 ms`
  开始时间(ns): `32961461`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[4800, 512]]}`
- `attn_mha` -> 0.091 ms
  纯GPU kernel时间: `0.504 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.091 ms`, `host_to_first_kernel_gap=10.409542`, `host_end_to_last_kernel_tail=10.822 ms`, `gpu_makespan=0.504 ms`, `gpu_kernel_sum=0.504 ms`
  开始时间(ns): `33158159`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4800, 128, 192], [4800, 128, 192], [4800, 128, 128]]}`
- `o_proj` -> 0.134 ms
  纯GPU kernel时间: `0.797 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.134 ms`, `host_to_first_kernel_gap=10.800193`, `host_end_to_last_kernel_tail=11.465 ms`, `gpu_makespan=0.799 ms`, `gpu_kernel_sum=0.797 ms`
  开始时间(ns): `33273748`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4800, 16384]]}`

## Layer 4 / prefill / instance 1

- 执行序号: `5`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `1.128 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.128 ms`, `module_to_last_kernel=18.883 ms`, `host_to_first_kernel_gap=16.730295`, `host_end_to_last_kernel_tail=17.755 ms`, `gpu_makespan=2.153 ms`, `gpu_kernel_sum=1.977 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4800`, `total_tokens=4800`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[4800, 128, 192], [4800, 128, 192], [4800, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=4800`, `sum_prefix=0`, `sum_seq_after=4800`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 1200, 'prefix_len': 0, 'seq_len_after': 1200, 'prompt_len': 1200, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 1200, 'prefix_len': 0, 'seq_len_after': 1200, 'prompt_len': 1200, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 1200, 'prefix_len': 0, 'seq_len_after': 1200, 'prompt_len': 1200, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 1200, 'prefix_len': 0, 'seq_len_after': 1200, 'prompt_len': 1200, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.173 ms
  纯GPU kernel时间: `0.162 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.173 ms`, `host_to_first_kernel_gap=16.64868`, `host_end_to_last_kernel_tail=16.640 ms`, `gpu_makespan=0.164 ms`, `gpu_kernel_sum=0.162 ms`
  开始时间(ns): `34421355`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4800, 7168]]}`
- `q_a_layernorm` -> 0.030 ms
  纯GPU kernel时间: `0.010 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.030 ms`, `host_to_first_kernel_gap=16.609286`, `host_end_to_last_kernel_tail=16.589 ms`, `gpu_makespan=0.010 ms`, `gpu_kernel_sum=0.010 ms`
  开始时间(ns): `34625837`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4800, 1536]]}`
- `q_b_proj` -> 0.126 ms
  纯GPU kernel时间: `0.266 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.126 ms`, `host_to_first_kernel_gap=16.571594`, `host_end_to_last_kernel_tail=16.714 ms`, `gpu_makespan=0.269 ms`, `gpu_kernel_sum=0.266 ms`
  开始时间(ns): `34674857`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4800, 1536]]}`
- `kv_a_layernorm` -> 0.026 ms
  纯GPU kernel时间: `0.008 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.026 ms`, `host_to_first_kernel_gap=16.679008`, `host_end_to_last_kernel_tail=16.661 ms`, `gpu_makespan=0.008 ms`, `gpu_kernel_sum=0.008 ms`
  开始时间(ns): `34836147`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4800, 512]]}`
- `rotary_emb` -> 0.049 ms
  纯GPU kernel时间: `0.062 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.049 ms`, `host_to_first_kernel_gap=16.639057`, `host_end_to_last_kernel_tail=16.652 ms`, `gpu_makespan=0.062 ms`, `gpu_kernel_sum=0.062 ms`
  开始时间(ns): `34885890`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4800], [4800, 128, 64], [4800, 1, 64]]}`
- `kv_b_proj` -> 0.141 ms
  纯GPU kernel时间: `0.162 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.141 ms`, `host_to_first_kernel_gap=16.594497`, `host_end_to_last_kernel_tail=16.618 ms`, `gpu_makespan=0.164 ms`, `gpu_kernel_sum=0.162 ms`
  开始时间(ns): `35011858`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[4800, 512]]}`
- `attn_mha` -> 0.091 ms
  纯GPU kernel时间: `0.508 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.091 ms`, `host_to_first_kernel_gap=16.700408`, `host_end_to_last_kernel_tail=17.117 ms`, `gpu_makespan=0.508 ms`, `gpu_kernel_sum=0.508 ms`
  开始时间(ns): `35212635`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4800, 128, 192], [4800, 128, 192], [4800, 128, 128]]}`
- `o_proj` -> 0.133 ms
  纯GPU kernel时间: `0.798 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.133 ms`, `host_to_first_kernel_gap=17.096989`, `host_end_to_last_kernel_tail=17.763 ms`, `gpu_makespan=0.799 ms`, `gpu_kernel_sum=0.798 ms`
  开始时间(ns): `35326518`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4800, 16384]]}`

## Layer 0 / decode / instance 1

- 执行序号: `6`
- Collector对齐边界: `{'Module': 'model.model.layers.0.self_attn'}`
- 整块 MLA-module 时长: `1.148 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.148 ms`, `module_to_last_kernel=22.365 ms`, `host_to_first_kernel_gap=22.215578`, `host_end_to_last_kernel_tail=21.216 ms`, `gpu_makespan=0.149 ms`, `gpu_kernel_sum=0.117 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.200 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.200 ms`, `host_to_first_kernel_gap=22.172184`, `host_end_to_last_kernel_tail=21.988 ms`, `gpu_makespan=0.016 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `37950425`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4, 7168]]}`
- `q_a_layernorm` -> 0.030 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.030 ms`, `host_to_first_kernel_gap=21.949553`, `host_end_to_last_kernel_tail=21.922 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `38188672`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4, 1536]]}`
- `kv_a_layernorm` -> 0.020 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.020 ms`, `host_to_first_kernel_gap=21.90831`, `host_end_to_last_kernel_tail=21.890 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `38232315`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4, 512]]}`
- `q_b_proj` -> 0.136 ms
  纯GPU kernel时间: `0.020 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.136 ms`, `host_to_first_kernel_gap=21.869621`, `host_end_to_last_kernel_tail=21.755 ms`, `gpu_makespan=0.022 ms`, `gpu_kernel_sum=0.020 ms`
  开始时间(ns): `38274172`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4, 1536]]}`
- `rotary_emb` -> 0.052 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.052 ms`, `host_to_first_kernel_gap=21.653953`, `host_end_to_last_kernel_tail=21.603 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `38523696`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4], [4, 128, 64], [4, 1, 64]]}`
- `attn_mqa` -> 0.189 ms
  纯GPU kernel时间: `0.026 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.189 ms`, `host_to_first_kernel_gap=21.576752`, `host_end_to_last_kernel_tail=21.414 ms`, `gpu_makespan=0.027 ms`, `gpu_kernel_sum=0.026 ms`
  开始时间(ns): `38603969`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4, 128, 512], [4, 1, 512], [4, 1, 512]]}`
- `o_proj` -> 0.169 ms
  纯GPU kernel时间: `0.051 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.169 ms`, `host_to_first_kernel_gap=21.343792`, `host_end_to_last_kernel_tail=21.227 ms`, `gpu_makespan=0.052 ms`, `gpu_kernel_sum=0.051 ms`
  开始时间(ns): `38875521`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4, 16384]]}`

## Layer 1 / decode / instance 1

- 执行序号: `7`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `0.957 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=0.957 ms`, `module_to_last_kernel=21.033 ms`, `host_to_first_kernel_gap=20.88832`, `host_end_to_last_kernel_tail=20.076 ms`, `gpu_makespan=0.145 ms`, `gpu_kernel_sum=0.116 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.116 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.116 ms`, `host_to_first_kernel_gap=20.852797`, `host_end_to_last_kernel_tail=20.753 ms`, `gpu_makespan=0.016 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `39579700`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4, 7168]]}`
- `q_a_layernorm` -> 0.026 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.026 ms`, `host_to_first_kernel_gap=20.718641`, `host_end_to_last_kernel_tail=20.694 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `39730048`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4, 1536]]}`
- `kv_a_layernorm` -> 0.017 ms
  纯GPU kernel时间: `0.003 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.017 ms`, `host_to_first_kernel_gap=20.680651`, `host_end_to_last_kernel_tail=20.666 ms`, `gpu_makespan=0.003 ms`, `gpu_kernel_sum=0.003 ms`
  开始时间(ns): `39769926`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4, 512]]}`
- `q_b_proj` -> 0.118 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.118 ms`, `host_to_first_kernel_gap=20.649211`, `host_end_to_last_kernel_tail=20.551 ms`, `gpu_makespan=0.019 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `39804950`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4, 1536]]}`
- `rotary_emb` -> 0.048 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.048 ms`, `host_to_first_kernel_gap=20.476044`, `host_end_to_last_kernel_tail=20.430 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `40009221`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4], [4, 128, 64], [4, 1, 64]]}`
- `attn_mqa` -> 0.175 ms
  纯GPU kernel时间: `0.027 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.175 ms`, `host_to_first_kernel_gap=20.406094`, `host_end_to_last_kernel_tail=20.259 ms`, `gpu_makespan=0.028 ms`, `gpu_kernel_sum=0.027 ms`
  开始时间(ns): `40081987`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4, 128, 512], [4, 1, 512], [4, 1, 512]]}`
- `o_proj` -> 0.160 ms
  纯GPU kernel时间: `0.050 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.160 ms`, `host_to_first_kernel_gap=20.195804`, `host_end_to_last_kernel_tail=20.086 ms`, `gpu_makespan=0.051 ms`, `gpu_kernel_sum=0.050 ms`
  开始时间(ns): `40330997`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4, 16384]]}`

## Layer 2 / decode / instance 1

- 执行序号: `8`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `0.950 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=0.950 ms`, `module_to_last_kernel=19.917 ms`, `host_to_first_kernel_gap=19.768864`, `host_end_to_last_kernel_tail=18.967 ms`, `gpu_makespan=0.148 ms`, `gpu_kernel_sum=0.117 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.117 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.117 ms`, `host_to_first_kernel_gap=19.734001`, `host_end_to_last_kernel_tail=19.634 ms`, `gpu_makespan=0.017 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `41004928`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4, 7168]]}`
- `q_a_layernorm` -> 0.026 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.026 ms`, `host_to_first_kernel_gap=19.592146`, `host_end_to_last_kernel_tail=19.568 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `41164031`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4, 1536]]}`
- `kv_a_layernorm` -> 0.018 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.018 ms`, `host_to_first_kernel_gap=19.555521`, `host_end_to_last_kernel_tail=19.539 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `41202768`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4, 512]]}`
- `q_b_proj` -> 0.113 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.113 ms`, `host_to_first_kernel_gap=19.522464`, `host_end_to_last_kernel_tail=19.428 ms`, `gpu_makespan=0.019 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `41238769`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4, 1536]]}`
- `rotary_emb` -> 0.047 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.047 ms`, `host_to_first_kernel_gap=19.354646`, `host_end_to_last_kernel_tail=19.310 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `41437019`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4], [4, 128, 64], [4, 1, 64]]}`
- `attn_mqa` -> 0.170 ms
  纯GPU kernel时间: `0.027 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.170 ms`, `host_to_first_kernel_gap=19.284533`, `host_end_to_last_kernel_tail=19.143 ms`, `gpu_makespan=0.029 ms`, `gpu_kernel_sum=0.027 ms`
  开始时间(ns): `41510012`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4, 128, 512], [4, 1, 512], [4, 1, 512]]}`
- `o_proj` -> 0.158 ms
  纯GPU kernel时间: `0.050 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.158 ms`, `host_to_first_kernel_gap=19.082935`, `host_end_to_last_kernel_tail=18.977 ms`, `gpu_makespan=0.052 ms`, `gpu_kernel_sum=0.050 ms`
  开始时间(ns): `41752410`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4, 16384]]}`

## Layer 3 / decode / instance 1

- 执行序号: `9`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `0.945 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=0.945 ms`, `module_to_last_kernel=18.796 ms`, `host_to_first_kernel_gap=18.64814`, `host_end_to_last_kernel_tail=17.851 ms`, `gpu_makespan=0.148 ms`, `gpu_kernel_sum=0.117 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.118 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.118 ms`, `host_to_first_kernel_gap=18.612416`, `host_end_to_last_kernel_tail=18.511 ms`, `gpu_makespan=0.016 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `42436049`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4, 7168]]}`
- `q_a_layernorm` -> 0.026 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.026 ms`, `host_to_first_kernel_gap=18.47706`, `host_end_to_last_kernel_tail=18.453 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `42587629`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4, 1536]]}`
- `kv_a_layernorm` -> 0.017 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.017 ms`, `host_to_first_kernel_gap=18.439382`, `host_end_to_last_kernel_tail=18.424 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `42627227`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4, 512]]}`
- `q_b_proj` -> 0.112 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.112 ms`, `host_to_first_kernel_gap=18.407482`, `host_end_to_last_kernel_tail=18.314 ms`, `gpu_makespan=0.019 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `42663127`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4, 1536]]}`
- `rotary_emb` -> 0.047 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.047 ms`, `host_to_first_kernel_gap=18.245508`, `host_end_to_last_kernel_tail=18.200 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `42855885`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4], [4, 128, 64], [4, 1, 64]]}`
- `attn_mqa` -> 0.167 ms
  纯GPU kernel时间: `0.027 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.167 ms`, `host_to_first_kernel_gap=18.176133`, `host_end_to_last_kernel_tail=18.037 ms`, `gpu_makespan=0.028 ms`, `gpu_kernel_sum=0.027 ms`
  开始时间(ns): `42928076`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4, 128, 512], [4, 1, 512], [4, 1, 512]]}`
- `o_proj` -> 0.160 ms
  纯GPU kernel时间: `0.051 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.160 ms`, `host_to_first_kernel_gap=17.969731`, `host_end_to_last_kernel_tail=17.862 ms`, `gpu_makespan=0.052 ms`, `gpu_kernel_sum=0.051 ms`
  开始时间(ns): `43174638`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4, 16384]]}`

## Layer 4 / decode / instance 1

- 执行序号: `10`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `1.014 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.014 ms`, `module_to_last_kernel=17.365 ms`, `host_to_first_kernel_gap=17.217429`, `host_end_to_last_kernel_tail=16.351 ms`, `gpu_makespan=0.147 ms`, `gpu_kernel_sum=0.116 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.158 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.158 ms`, `host_to_first_kernel_gap=17.178795`, `host_end_to_last_kernel_tail=17.038 ms`, `gpu_makespan=0.017 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `44219238`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4, 7168]]}`
- `q_a_layernorm` -> 0.030 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.030 ms`, `host_to_first_kernel_gap=17.000816`, `host_end_to_last_kernel_tail=16.973 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `44413793`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4, 1536]]}`
- `kv_a_layernorm` -> 0.018 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.018 ms`, `host_to_first_kernel_gap=16.959343`, `host_end_to_last_kernel_tail=16.944 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `44457186`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4, 512]]}`
- `q_b_proj` -> 0.121 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.121 ms`, `host_to_first_kernel_gap=16.925584`, `host_end_to_last_kernel_tail=16.825 ms`, `gpu_makespan=0.020 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `44494241`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4, 1536]]}`
- `rotary_emb` -> 0.052 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.052 ms`, `host_to_first_kernel_gap=16.7426`, `host_end_to_last_kernel_tail=16.693 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `44707753`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4], [4, 128, 64], [4, 1, 64]]}`
- `attn_mqa` -> 0.170 ms
  纯GPU kernel时间: `0.026 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.170 ms`, `host_to_first_kernel_gap=16.667646`, `host_end_to_last_kernel_tail=16.525 ms`, `gpu_makespan=0.028 ms`, `gpu_kernel_sum=0.026 ms`
  开始时间(ns): `44786515`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4, 128, 512], [4, 1, 512], [4, 1, 512]]}`
- `o_proj` -> 0.156 ms
  纯GPU kernel时间: `0.049 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.156 ms`, `host_to_first_kernel_gap=16.46691`, `host_end_to_last_kernel_tail=16.362 ms`, `gpu_makespan=0.051 ms`, `gpu_kernel_sum=0.049 ms`
  开始时间(ns): `45027891`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4, 16384]]}`
