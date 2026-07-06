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
- 整块 MLA-module 时长: `43675.992 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=43675.992 ms`, `module_to_last_kernel=43675.992 ms`, `host_to_first_kernel_gap=1.589935`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=43674.088 ms`, `gpu_kernel_sum=0.139 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8`, `total_tokens=8`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8, 128, 192], [8, 128, 192], [8, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8`, `sum_prefix=0`, `sum_seq_after=8`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 4, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 5, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 6, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 7, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 4903.853 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=4903.853 ms`, `host_to_first_kernel_gap=1.258149`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=4902.234 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `5248286417`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8, 7168]]}`
- `q_a_layernorm` -> 0.267 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.267 ms`, `host_to_first_kernel_gap=0.241447`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `10152459631`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8, 1536]]}`
- `q_b_proj` -> 7851.659 ms
  纯GPU kernel时间: `0.023 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=7851.659 ms`, `host_to_first_kernel_gap=0.454192`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=7850.951 ms`, `gpu_kernel_sum=0.023 ms`
  开始时间(ns): `10152870982`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8, 1536]]}`
- `kv_a_layernorm` -> 0.256 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.256 ms`, `host_to_first_kernel_gap=0.228729`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `18004870373`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8, 512]]}`
- `rotary_emb` -> 19388.018 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=19388.018 ms`, `host_to_first_kernel_gap=19387.765705`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `18005297427`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8], [8, 128, 64], [8, 1, 64]]}`
- `kv_b_proj` -> 5367.675 ms
  纯GPU kernel时间: `0.011 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=5367.675 ms`, `host_to_first_kernel_gap=0.566821`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=5366.813 ms`, `gpu_kernel_sum=0.011 ms`
  开始时间(ns): `38602380463`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[8, 512]]}`
- `attn_mha` -> 18.991 ms
  纯GPU kernel时间: `0.040 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=18.991 ms`, `host_to_first_kernel_gap=3.035984`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=15.858 ms`, `gpu_kernel_sum=0.040 ms`
  开始时间(ns): `43971698188`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8, 128, 192], [8, 128, 192], [8, 128, 128]]}`
- `o_proj` -> 4932.874 ms
  纯GPU kernel时间: `0.047 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=4932.874 ms`, `host_to_first_kernel_gap=0.47122`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=4932.181 ms`, `gpu_kernel_sum=0.047 ms`
  开始时间(ns): `43990980156`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8, 16384]]}`

## Layer 1 / prefill / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `6.001 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=6.001 ms`, `module_to_last_kernel=6.001 ms`, `host_to_first_kernel_gap=1.046499`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=4.877 ms`, `gpu_kernel_sum=0.133 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8`, `total_tokens=8`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8, 128, 192], [8, 128, 192], [8, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8`, `sum_prefix=0`, `sum_seq_after=8`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 4, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 5, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 6, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 7, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.958 ms
  纯GPU kernel时间: `0.012 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.958 ms`, `host_to_first_kernel_gap=0.379296`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.510 ms`, `gpu_kernel_sum=0.012 ms`
  开始时间(ns): `59336406971`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8, 7168]]}`
- `q_a_layernorm` -> 0.145 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.145 ms`, `host_to_first_kernel_gap=0.123076`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `59337542998`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8, 1536]]}`
- `q_b_proj` -> 0.644 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.644 ms`, `host_to_first_kernel_gap=0.241077`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.356 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `59337790181`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8, 1536]]}`
- `kv_a_layernorm` -> 0.118 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.118 ms`, `host_to_first_kernel_gap=0.099545`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `59338616000`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8, 512]]}`
- `rotary_emb` -> 0.296 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.296 ms`, `host_to_first_kernel_gap=0.259359`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `59338853434`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8], [8, 128, 64], [8, 1, 64]]}`
- `kv_b_proj` -> 0.639 ms
  纯GPU kernel时间: `0.011 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.639 ms`, `host_to_first_kernel_gap=0.230787`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.351 ms`, `gpu_kernel_sum=0.011 ms`
  开始时间(ns): `59339576886`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[8, 512]]}`
- `attn_mha` -> 0.420 ms
  纯GPU kernel时间: `0.040 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.420 ms`, `host_to_first_kernel_gap=0.311705`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.092 ms`, `gpu_kernel_sum=0.040 ms`
  开始时间(ns): `59340489663`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8, 128, 192], [8, 128, 192], [8, 128, 128]]}`
- `o_proj` -> 0.631 ms
  纯GPU kernel时间: `0.046 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.631 ms`, `host_to_first_kernel_gap=0.231598`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.375 ms`, `gpu_kernel_sum=0.046 ms`
  开始时间(ns): `59341056618`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8, 16384]]}`

## Layer 2 / prefill / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `4.732 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=4.732 ms`, `module_to_last_kernel=4.732 ms`, `host_to_first_kernel_gap=0.593121`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=4.083 ms`, `gpu_kernel_sum=0.134 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8`, `total_tokens=8`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8, 128, 192], [8, 128, 192], [8, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8`, `sum_prefix=0`, `sum_seq_after=8`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 4, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 5, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 6, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 7, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.617 ms
  纯GPU kernel时间: `0.012 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.617 ms`, `host_to_first_kernel_gap=0.227013`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.335 ms`, `gpu_kernel_sum=0.012 ms`
  开始时间(ns): `59344460177`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8, 7168]]}`
- `q_a_layernorm` -> 0.117 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.117 ms`, `host_to_first_kernel_gap=0.097653`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `59345221856`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8, 1536]]}`
- `q_b_proj` -> 0.551 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.551 ms`, `host_to_first_kernel_gap=0.190425`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.312 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `59345416508`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8, 1536]]}`
- `kv_a_layernorm` -> 0.108 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.108 ms`, `host_to_first_kernel_gap=0.090912`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `59346138965`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8, 512]]}`
- `rotary_emb` -> 0.185 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.185 ms`, `host_to_first_kernel_gap=0.152414`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `59346343734`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8], [8, 128, 64], [8, 1, 64]]}`
- `kv_b_proj` -> 0.616 ms
  纯GPU kernel时间: `0.010 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.616 ms`, `host_to_first_kernel_gap=0.229095`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.332 ms`, `gpu_kernel_sum=0.010 ms`
  开始时间(ns): `59346869293`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[8, 512]]}`
- `attn_mha` -> 0.351 ms
  纯GPU kernel时间: `0.040 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.351 ms`, `host_to_first_kernel_gap=0.257433`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.081 ms`, `gpu_kernel_sum=0.040 ms`
  开始时间(ns): `59347734618`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8, 128, 192], [8, 128, 192], [8, 128, 128]]}`
- `o_proj` -> 0.599 ms
  纯GPU kernel时间: `0.046 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.599 ms`, `host_to_first_kernel_gap=0.210965`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.368 ms`, `gpu_kernel_sum=0.046 ms`
  开始时间(ns): `59348191678`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8, 16384]]}`

## Layer 3 / prefill / instance 1

- 执行序号: `4`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `4.579 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=4.579 ms`, `module_to_last_kernel=4.579 ms`, `host_to_first_kernel_gap=0.549997`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=3.977 ms`, `gpu_kernel_sum=0.136 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8`, `total_tokens=8`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8, 128, 192], [8, 128, 192], [8, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8`, `sum_prefix=0`, `sum_seq_after=8`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 4, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 5, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 6, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 7, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.554 ms
  纯GPU kernel时间: `0.012 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.554 ms`, `host_to_first_kernel_gap=0.205868`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.301 ms`, `gpu_kernel_sum=0.012 ms`
  开始时间(ns): `59351320485`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8, 7168]]}`
- `q_a_layernorm` -> 0.114 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.114 ms`, `host_to_first_kernel_gap=0.094876`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `59352019285`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8, 1536]]}`
- `q_b_proj` -> 0.540 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.540 ms`, `host_to_first_kernel_gap=0.1884`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.308 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `59352213281`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8, 1536]]}`
- `kv_a_layernorm` -> 0.121 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.121 ms`, `host_to_first_kernel_gap=0.097288`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `59352915016`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8, 512]]}`
- `rotary_emb` -> 0.179 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.179 ms`, `host_to_first_kernel_gap=0.145871`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `59353139745`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8], [8, 128, 64], [8, 1, 64]]}`
- `kv_b_proj` -> 0.625 ms
  纯GPU kernel时间: `0.010 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.625 ms`, `host_to_first_kernel_gap=0.214833`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.354 ms`, `gpu_kernel_sum=0.010 ms`
  开始时间(ns): `59353644095`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[8, 512]]}`
- `attn_mha` -> 0.331 ms
  纯GPU kernel时间: `0.041 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.331 ms`, `host_to_first_kernel_gap=0.240484`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.083 ms`, `gpu_kernel_sum=0.041 ms`
  开始时间(ns): `59354507915`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8, 128, 192], [8, 128, 192], [8, 128, 128]]}`
- `o_proj` -> 0.573 ms
  纯GPU kernel时间: `0.047 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.573 ms`, `host_to_first_kernel_gap=0.203867`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.353 ms`, `gpu_kernel_sum=0.047 ms`
  开始时间(ns): `59354946612`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8, 16384]]}`

## Layer 4 / prefill / instance 1

- 执行序号: `5`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `2.453 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.453 ms`, `module_to_last_kernel=2.458 ms`, `host_to_first_kernel_gap=0.396489`, `host_end_to_last_kernel_tail=0.005 ms`, `gpu_makespan=2.061 ms`, `gpu_kernel_sum=0.142 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8`, `total_tokens=8`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8, 128, 192], [8, 128, 192], [8, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8`, `sum_prefix=0`, `sum_seq_after=8`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 4, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 5, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 6, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 7, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.435 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.435 ms`, `host_to_first_kernel_gap=0.15533`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.255 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `64767453893`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8, 7168]]}`
- `q_a_layernorm` -> 0.064 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.064 ms`, `host_to_first_kernel_gap=0.05679`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `64767989937`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8, 1536]]}`
- `q_b_proj` -> 0.252 ms
  纯GPU kernel时间: `0.024 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.252 ms`, `host_to_first_kernel_gap=0.094664`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.155 ms`, `gpu_kernel_sum=0.024 ms`
  开始时间(ns): `64768089726`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8, 1536]]}`
- `kv_a_layernorm` -> 0.042 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.042 ms`, `host_to_first_kernel_gap=0.036527`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `64768410615`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8, 512]]}`
- `rotary_emb` -> 0.109 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.109 ms`, `host_to_first_kernel_gap=0.094631`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `64768510879`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8], [8, 128, 64], [8, 1, 64]]}`
- `kv_b_proj` -> 0.266 ms
  纯GPU kernel时间: `0.012 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.266 ms`, `host_to_first_kernel_gap=0.092099`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.158 ms`, `gpu_kernel_sum=0.012 ms`
  开始时间(ns): `64768788259`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[8, 512]]}`
- `attn_mha` -> 0.189 ms
  纯GPU kernel时间: `0.040 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.189 ms`, `host_to_first_kernel_gap=0.146989`, `host_end_to_last_kernel_tail=0.018 ms`, `gpu_makespan=0.060 ms`, `gpu_kernel_sum=0.040 ms`
  开始时间(ns): `64769166233`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8, 128, 192], [8, 128, 192], [8, 128, 128]]}`
- `o_proj` -> 0.246 ms
  纯GPU kernel时间: `0.047 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.246 ms`, `host_to_first_kernel_gap=0.093225`, `host_end_to_last_kernel_tail=0.020 ms`, `gpu_makespan=0.173 ms`, `gpu_kernel_sum=0.047 ms`
  开始时间(ns): `64769404733`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8, 16384]]}`

## Layer 0 / decode / instance 1

- 执行序号: `6`
- Collector对齐边界: `{'Module': 'model.model.layers.0.self_attn'}`
- 整块 MLA-module 时长: `16.842 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=16.842 ms`, `module_to_last_kernel=16.842 ms`, `host_to_first_kernel_gap=0.258542`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=16.569 ms`, `gpu_kernel_sum=0.099 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.429 ms
  纯GPU kernel时间: `0.012 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.429 ms`, `host_to_first_kernel_gap=0.149937`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.259 ms`, `gpu_kernel_sum=0.012 ms`
  开始时间(ns): `66156671690`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8, 7168]]}`
- `q_a_layernorm` -> 0.050 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.050 ms`, `host_to_first_kernel_gap=0.044208`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `66157184043`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8, 1536]]}`
- `kv_a_layernorm` -> 0.031 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.031 ms`, `host_to_first_kernel_gap=0.027438`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `66157264685`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8, 512]]}`
- `q_b_proj` -> 0.243 ms
  纯GPU kernel时间: `0.021 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.243 ms`, `host_to_first_kernel_gap=0.088787`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.148 ms`, `gpu_kernel_sum=0.021 ms`
  开始时间(ns): `66157341672`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8, 1536]]}`
- `rotary_emb` -> 0.151 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.151 ms`, `host_to_first_kernel_gap=0.131317`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `66161228580`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8], [8, 128, 64], [8, 1, 64]]}`
- `attn_mqa` -> 8.317 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=8.317 ms`, `host_to_first_kernel_gap=0.130992`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=8.127 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `66161451721`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8, 128, 512], [8, 1, 512], [8, 1, 512]]}`
- `o_proj` -> 0.405 ms
  纯GPU kernel时间: `0.046 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.405 ms`, `host_to_first_kernel_gap=0.173372`, `host_end_to_last_kernel_tail=0.009 ms`, `gpu_makespan=0.241 ms`, `gpu_kernel_sum=0.046 ms`
  开始时间(ns): `66172977017`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8, 16384]]}`

## Layer 1 / decode / instance 1

- 执行序号: `7`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `1.760 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.760 ms`, `module_to_last_kernel=1.761 ms`, `host_to_first_kernel_gap=0.156134`, `host_end_to_last_kernel_tail=0.001 ms`, `gpu_makespan=1.604 ms`, `gpu_kernel_sum=0.099 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.221 ms
  纯GPU kernel时间: `0.012 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.221 ms`, `host_to_first_kernel_gap=0.082793`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.126 ms`, `gpu_kernel_sum=0.012 ms`
  开始时间(ns): `66174419756`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8, 7168]]}`
- `q_a_layernorm` -> 0.049 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.049 ms`, `host_to_first_kernel_gap=0.042954`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `66174709483`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8, 1536]]}`
- `kv_a_layernorm` -> 0.030 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.030 ms`, `host_to_first_kernel_gap=0.02705`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `66174785355`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8, 512]]}`
- `q_b_proj` -> 0.237 ms
  纯GPU kernel时间: `0.020 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.237 ms`, `host_to_first_kernel_gap=0.088154`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.142 ms`, `gpu_kernel_sum=0.020 ms`
  开始时间(ns): `66174852730`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8, 1536]]}`
- `rotary_emb` -> 0.089 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.089 ms`, `host_to_first_kernel_gap=0.076188`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `66175256184`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8], [8, 128, 64], [8, 1, 64]]}`
- `attn_mqa` -> 0.282 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.282 ms`, `host_to_first_kernel_gap=0.101165`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.165 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `66175404519`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8, 128, 512], [8, 1, 512], [8, 1, 512]]}`
- `o_proj` -> 0.273 ms
  纯GPU kernel时间: `0.047 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.273 ms`, `host_to_first_kernel_gap=0.093923`, `host_end_to_last_kernel_tail=0.019 ms`, `gpu_makespan=0.198 ms`, `gpu_kernel_sum=0.047 ms`
  开始时间(ns): `66175814897`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8, 16384]]}`

## Layer 2 / decode / instance 1

- 执行序号: `8`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.627 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.627 ms`, `module_to_last_kernel=1.628 ms`, `host_to_first_kernel_gap=0.153313`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=1.474 ms`, `gpu_kernel_sum=0.096 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.219 ms
  纯GPU kernel时间: `0.012 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.219 ms`, `host_to_first_kernel_gap=0.079441`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.128 ms`, `gpu_kernel_sum=0.012 ms`
  开始时间(ns): `66176998883`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8, 7168]]}`
- `q_a_layernorm` -> 0.044 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.044 ms`, `host_to_first_kernel_gap=0.038945`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `66177278419`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8, 1536]]}`
- `kv_a_layernorm` -> 0.031 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.031 ms`, `host_to_first_kernel_gap=0.027542`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `66177347230`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8, 512]]}`
- `q_b_proj` -> 0.202 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.202 ms`, `host_to_first_kernel_gap=0.073562`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.125 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `66177411066`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8, 1536]]}`
- `rotary_emb` -> 0.080 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.080 ms`, `host_to_first_kernel_gap=0.067274`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `66177750409`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8], [8, 128, 64], [8, 1, 64]]}`
- `attn_mqa` -> 0.263 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.263 ms`, `host_to_first_kernel_gap=0.095873`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.149 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `66177881970`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8, 128, 512], [8, 1, 512], [8, 1, 512]]}`
- `o_proj` -> 0.266 ms
  纯GPU kernel时间: `0.046 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.266 ms`, `host_to_first_kernel_gap=0.095592`, `host_end_to_last_kernel_tail=0.018 ms`, `gpu_makespan=0.188 ms`, `gpu_kernel_sum=0.046 ms`
  开始时间(ns): `66178268523`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8, 16384]]}`

## Layer 3 / decode / instance 1

- 执行序号: `9`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `1.635 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.635 ms`, `module_to_last_kernel=1.635 ms`, `host_to_first_kernel_gap=0.144013`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=1.491 ms`, `gpu_kernel_sum=0.097 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.209 ms
  纯GPU kernel时间: `0.012 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.209 ms`, `host_to_first_kernel_gap=0.079235`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.119 ms`, `gpu_kernel_sum=0.012 ms`
  开始时间(ns): `66179463216`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8, 7168]]}`
- `q_a_layernorm` -> 0.048 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.048 ms`, `host_to_first_kernel_gap=0.038991`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `66179732484`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8, 1536]]}`
- `kv_a_layernorm` -> 0.032 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.032 ms`, `host_to_first_kernel_gap=0.028251`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `66179805944`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8, 512]]}`
- `q_b_proj` -> 0.226 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.226 ms`, `host_to_first_kernel_gap=0.093538`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.129 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `66179872785`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8, 1536]]}`
- `rotary_emb` -> 0.078 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.078 ms`, `host_to_first_kernel_gap=0.066151`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `66180243916`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8], [8, 128, 64], [8, 1, 64]]}`
- `attn_mqa` -> 0.251 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.251 ms`, `host_to_first_kernel_gap=0.088817`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.148 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `66180372034`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8, 128, 512], [8, 1, 512], [8, 1, 512]]}`
- `o_proj` -> 0.274 ms
  纯GPU kernel时间: `0.046 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.274 ms`, `host_to_first_kernel_gap=0.092807`, `host_end_to_last_kernel_tail=0.018 ms`, `gpu_makespan=0.199 ms`, `gpu_kernel_sum=0.046 ms`
  开始时间(ns): `66180740907`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8, 16384]]}`

## Layer 4 / decode / instance 1

- 执行序号: `10`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `1.732 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.732 ms`, `module_to_last_kernel=1.734 ms`, `host_to_first_kernel_gap=0.165669`, `host_end_to_last_kernel_tail=0.002 ms`, `gpu_makespan=1.568 ms`, `gpu_kernel_sum=0.097 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.279 ms
  纯GPU kernel时间: `0.012 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.279 ms`, `host_to_first_kernel_gap=0.096328`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.167 ms`, `gpu_kernel_sum=0.012 ms`
  开始时间(ns): `66182648778`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8, 7168]]}`
- `q_a_layernorm` -> 0.049 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.049 ms`, `host_to_first_kernel_gap=0.042915`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `66183002255`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8, 1536]]}`
- `kv_a_layernorm` -> 0.037 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.037 ms`, `host_to_first_kernel_gap=0.028029`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `66183077109`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8, 512]]}`
- `q_b_proj` -> 0.226 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.226 ms`, `host_to_first_kernel_gap=0.080342`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.141 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `66183151804`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8, 1536]]}`
- `rotary_emb` -> 0.079 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.079 ms`, `host_to_first_kernel_gap=0.067685`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `66183532876`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8], [8, 128, 64], [8, 1, 64]]}`
- `attn_mqa` -> 0.253 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.253 ms`, `host_to_first_kernel_gap=0.086766`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.151 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `66183658595`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8, 128, 512], [8, 1, 512], [8, 1, 512]]}`
- `o_proj` -> 0.250 ms
  纯GPU kernel时间: `0.047 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.250 ms`, `host_to_first_kernel_gap=0.090058`, `host_end_to_last_kernel_tail=0.020 ms`, `gpu_makespan=0.180 ms`, `gpu_kernel_sum=0.047 ms`
  开始时间(ns): `66184043431`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8, 16384]]}`
