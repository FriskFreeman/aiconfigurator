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
- 整块 MLA-module 时长: `1.264 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.264 ms`, `module_to_last_kernel=1.287 ms`, `host_to_first_kernel_gap=0.196892`, `host_end_to_last_kernel_tail=0.023 ms`, `gpu_makespan=1.090 ms`, `gpu_kernel_sum=0.134 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8`, `total_tokens=8`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8, 128, 192], [8, 128, 192], [8, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8`, `sum_prefix=0`, `sum_seq_after=8`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 4, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 5, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 6, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 7, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.214 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.214 ms`, `host_to_first_kernel_gap=0.075513`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.133 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `26404544`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8, 7168]]}`
- `q_a_layernorm` -> 0.032 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.032 ms`, `host_to_first_kernel_gap=0.029661`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `26651420`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8, 1536]]}`
- `q_b_proj` -> 0.136 ms
  纯GPU kernel时间: `0.020 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.136 ms`, `host_to_first_kernel_gap=0.051517`, `host_end_to_last_kernel_tail=0.006 ms`, `gpu_makespan=0.090 ms`, `gpu_kernel_sum=0.020 ms`
  开始时间(ns): `26703772`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8, 1536]]}`
- `kv_a_layernorm` -> 0.025 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.025 ms`, `host_to_first_kernel_gap=0.022897`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `26877832`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8, 512]]}`
- `rotary_emb` -> 0.051 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.051 ms`, `host_to_first_kernel_gap=0.045081`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `26927008`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8], [8, 128, 64], [8, 1, 64]]}`
- `kv_b_proj` -> 0.144 ms
  纯GPU kernel时间: `0.011 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.144 ms`, `host_to_first_kernel_gap=0.053388`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.085 ms`, `gpu_kernel_sum=0.011 ms`
  开始时间(ns): `27064877`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[8, 512]]}`
- `attn_mha` -> 0.104 ms
  纯GPU kernel时间: `0.039 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.104 ms`, `host_to_first_kernel_gap=0.07945`, `host_end_to_last_kernel_tail=0.026 ms`, `gpu_makespan=0.050 ms`, `gpu_kernel_sum=0.039 ms`
  开始时间(ns): `27274943`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8, 128, 192], [8, 128, 192], [8, 128, 128]]}`
- `o_proj` -> 0.134 ms
  纯GPU kernel时间: `0.045 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.134 ms`, `host_to_first_kernel_gap=0.053214`, `host_end_to_last_kernel_tail=0.031 ms`, `gpu_makespan=0.112 ms`, `gpu_kernel_sum=0.045 ms`
  开始时间(ns): `27404827`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8, 16384]]}`

## Layer 1 / prefill / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `1.036 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.036 ms`, `module_to_last_kernel=1.062 ms`, `host_to_first_kernel_gap=0.132365`, `host_end_to_last_kernel_tail=0.025 ms`, `gpu_makespan=0.929 ms`, `gpu_kernel_sum=0.131 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8`, `total_tokens=8`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8, 128, 192], [8, 128, 192], [8, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8`, `sum_prefix=0`, `sum_seq_after=8`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 4, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 5, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 6, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 7, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.124 ms
  纯GPU kernel时间: `0.012 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.124 ms`, `host_to_first_kernel_gap=0.049222`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.075 ms`, `gpu_kernel_sum=0.012 ms`
  开始时间(ns): `28106803`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8, 7168]]}`
- `q_a_layernorm` -> 0.027 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.027 ms`, `host_to_first_kernel_gap=0.024588`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `28258093`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8, 1536]]}`
- `q_b_proj` -> 0.118 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.118 ms`, `host_to_first_kernel_gap=0.047764`, `host_end_to_last_kernel_tail=0.007 ms`, `gpu_makespan=0.078 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `28302341`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8, 1536]]}`
- `kv_a_layernorm` -> 0.024 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.024 ms`, `host_to_first_kernel_gap=0.022279`, `host_end_to_last_kernel_tail=0.001 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `28451826`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8, 512]]}`
- `rotary_emb` -> 0.043 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.043 ms`, `host_to_first_kernel_gap=0.03828`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `28496753`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8], [8, 128, 64], [8, 1, 64]]}`
- `kv_b_proj` -> 0.131 ms
  纯GPU kernel时间: `0.010 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.131 ms`, `host_to_first_kernel_gap=0.048987`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.079 ms`, `gpu_kernel_sum=0.010 ms`
  开始时间(ns): `28622526`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[8, 512]]}`
- `attn_mha` -> 0.086 ms
  纯GPU kernel时间: `0.039 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.086 ms`, `host_to_first_kernel_gap=0.062749`, `host_end_to_last_kernel_tail=0.027 ms`, `gpu_makespan=0.050 ms`, `gpu_kernel_sum=0.039 ms`
  开始时间(ns): `28813788`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8, 128, 192], [8, 128, 192], [8, 128, 128]]}`
- `o_proj` -> 0.128 ms
  纯GPU kernel时间: `0.046 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.128 ms`, `host_to_first_kernel_gap=0.049986`, `host_end_to_last_kernel_tail=0.034 ms`, `gpu_makespan=0.112 ms`, `gpu_kernel_sum=0.046 ms`
  开始时间(ns): `28923735`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8, 16384]]}`

## Layer 2 / prefill / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.016 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.016 ms`, `module_to_last_kernel=1.039 ms`, `host_to_first_kernel_gap=0.129312`, `host_end_to_last_kernel_tail=0.023 ms`, `gpu_makespan=0.910 ms`, `gpu_kernel_sum=0.131 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8`, `total_tokens=8`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8, 128, 192], [8, 128, 192], [8, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8`, `sum_prefix=0`, `sum_seq_after=8`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 4, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 5, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 6, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 7, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.121 ms
  纯GPU kernel时间: `0.012 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.121 ms`, `host_to_first_kernel_gap=0.049477`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.071 ms`, `gpu_kernel_sum=0.012 ms`
  开始时间(ns): `29604340`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8, 7168]]}`
- `q_a_layernorm` -> 0.026 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.026 ms`, `host_to_first_kernel_gap=0.023605`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `29752804`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8, 1536]]}`
- `q_b_proj` -> 0.118 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.118 ms`, `host_to_first_kernel_gap=0.048367`, `host_end_to_last_kernel_tail=0.007 ms`, `gpu_makespan=0.076 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `29795946`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8, 1536]]}`
- `kv_a_layernorm` -> 0.024 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.024 ms`, `host_to_first_kernel_gap=0.022425`, `host_end_to_last_kernel_tail=0.001 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `29946688`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8, 512]]}`
- `rotary_emb` -> 0.042 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.042 ms`, `host_to_first_kernel_gap=0.037365`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `29992388`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8], [8, 128, 64], [8, 1, 64]]}`
- `kv_b_proj` -> 0.128 ms
  纯GPU kernel时间: `0.010 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.128 ms`, `host_to_first_kernel_gap=0.050843`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.075 ms`, `gpu_kernel_sum=0.010 ms`
  开始时间(ns): `30112286`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[8, 512]]}`
- `attn_mha` -> 0.086 ms
  纯GPU kernel时间: `0.040 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.086 ms`, `host_to_first_kernel_gap=0.06524`, `host_end_to_last_kernel_tail=0.027 ms`, `gpu_makespan=0.048 ms`, `gpu_kernel_sum=0.040 ms`
  开始时间(ns): `30294817`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8, 128, 192], [8, 128, 192], [8, 128, 128]]}`
- `o_proj` -> 0.128 ms
  纯GPU kernel时间: `0.044 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.128 ms`, `host_to_first_kernel_gap=0.048836`, `host_end_to_last_kernel_tail=0.031 ms`, `gpu_makespan=0.110 ms`, `gpu_kernel_sum=0.044 ms`
  开始时间(ns): `30404629`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8, 16384]]}`

## Layer 3 / prefill / instance 1

- 执行序号: `4`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `0.994 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=0.994 ms`, `module_to_last_kernel=1.019 ms`, `host_to_first_kernel_gap=0.126384`, `host_end_to_last_kernel_tail=0.026 ms`, `gpu_makespan=0.893 ms`, `gpu_kernel_sum=0.135 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8`, `total_tokens=8`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8, 128, 192], [8, 128, 192], [8, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8`, `sum_prefix=0`, `sum_seq_after=8`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 4, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 5, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 6, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 7, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.119 ms
  纯GPU kernel时间: `0.012 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.119 ms`, `host_to_first_kernel_gap=0.046651`, `host_end_to_last_kernel_tail=0.001 ms`, `gpu_makespan=0.073 ms`, `gpu_kernel_sum=0.012 ms`
  开始时间(ns): `31074878`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8, 7168]]}`
- `q_a_layernorm` -> 0.026 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.026 ms`, `host_to_first_kernel_gap=0.02372`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `31220977`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8, 1536]]}`
- `q_b_proj` -> 0.112 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.112 ms`, `host_to_first_kernel_gap=0.040287`, `host_end_to_last_kernel_tail=0.009 ms`, `gpu_makespan=0.080 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `31263898`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8, 1536]]}`
- `kv_a_layernorm` -> 0.023 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.023 ms`, `host_to_first_kernel_gap=0.021749`, `host_end_to_last_kernel_tail=0.001 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `31407844`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8, 512]]}`
- `rotary_emb` -> 0.044 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.044 ms`, `host_to_first_kernel_gap=0.036806`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `31452499`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8], [8, 128, 64], [8, 1, 64]]}`
- `kv_b_proj` -> 0.125 ms
  纯GPU kernel时间: `0.010 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.125 ms`, `host_to_first_kernel_gap=0.047203`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.076 ms`, `gpu_kernel_sum=0.010 ms`
  开始时间(ns): `31573366`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[8, 512]]}`
- `attn_mha` -> 0.081 ms
  纯GPU kernel时间: `0.041 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.081 ms`, `host_to_first_kernel_gap=0.060097`, `host_end_to_last_kernel_tail=0.028 ms`, `gpu_makespan=0.049 ms`, `gpu_kernel_sum=0.041 ms`
  开始时间(ns): `31752152`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8, 128, 192], [8, 128, 192], [8, 128, 128]]}`
- `o_proj` -> 0.124 ms
  纯GPU kernel时间: `0.046 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.124 ms`, `host_to_first_kernel_gap=0.046708`, `host_end_to_last_kernel_tail=0.033 ms`, `gpu_makespan=0.110 ms`, `gpu_kernel_sum=0.046 ms`
  开始时间(ns): `31857573`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8, 16384]]}`

## Layer 4 / prefill / instance 1

- 执行序号: `5`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `1.093 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.093 ms`, `module_to_last_kernel=1.118 ms`, `host_to_first_kernel_gap=0.1479`, `host_end_to_last_kernel_tail=0.025 ms`, `gpu_makespan=0.970 ms`, `gpu_kernel_sum=0.132 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8`, `total_tokens=8`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8, 128, 192], [8, 128, 192], [8, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8`, `sum_prefix=0`, `sum_seq_after=8`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 4, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 5, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 6, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}, {'req_id': 7, 'extend_len': 1, 'prefix_len': 0, 'seq_len_after': 1, 'prompt_len': 1, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.165 ms
  纯GPU kernel时间: `0.012 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.165 ms`, `host_to_first_kernel_gap=0.060046`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.102 ms`, `gpu_kernel_sum=0.012 ms`
  开始时间(ns): `32984491`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8, 7168]]}`
- `q_a_layernorm` -> 0.030 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.030 ms`, `host_to_first_kernel_gap=0.027466`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `33179887`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8, 1536]]}`
- `q_b_proj` -> 0.130 ms
  纯GPU kernel时间: `0.020 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.130 ms`, `host_to_first_kernel_gap=0.04609`, `host_end_to_last_kernel_tail=0.008 ms`, `gpu_makespan=0.091 ms`, `gpu_kernel_sum=0.020 ms`
  开始时间(ns): `33228431`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8, 1536]]}`
- `kv_a_layernorm` -> 0.025 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.025 ms`, `host_to_first_kernel_gap=0.023469`, `host_end_to_last_kernel_tail=0.001 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `33392780`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8, 512]]}`
- `rotary_emb` -> 0.050 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.050 ms`, `host_to_first_kernel_gap=0.043244`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `33440205`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8], [8, 128, 64], [8, 1, 64]]}`
- `kv_b_proj` -> 0.127 ms
  纯GPU kernel时间: `0.010 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.127 ms`, `host_to_first_kernel_gap=0.047632`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.078 ms`, `gpu_kernel_sum=0.010 ms`
  开始时间(ns): `33568297`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[8, 512]]}`
- `attn_mha` -> 0.083 ms
  纯GPU kernel时间: `0.039 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.083 ms`, `host_to_first_kernel_gap=0.062087`, `host_end_to_last_kernel_tail=0.027 ms`, `gpu_makespan=0.049 ms`, `gpu_kernel_sum=0.039 ms`
  开始时间(ns): `33750930`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8, 128, 192], [8, 128, 192], [8, 128, 128]]}`
- `o_proj` -> 0.125 ms
  纯GPU kernel时间: `0.045 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.125 ms`, `host_to_first_kernel_gap=0.048707`, `host_end_to_last_kernel_tail=0.033 ms`, `gpu_makespan=0.109 ms`, `gpu_kernel_sum=0.045 ms`
  开始时间(ns): `33857462`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8, 16384]]}`

## Layer 0 / decode / instance 1

- 执行序号: `6`
- Collector对齐边界: `{'Module': 'model.model.layers.0.self_attn'}`
- 整块 MLA-module 时长: `1.054 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.054 ms`, `module_to_last_kernel=1.073 ms`, `host_to_first_kernel_gap=0.106515`, `host_end_to_last_kernel_tail=0.019 ms`, `gpu_makespan=0.967 ms`, `gpu_kernel_sum=0.097 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.184 ms
  纯GPU kernel时间: `0.012 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.184 ms`, `host_to_first_kernel_gap=0.065658`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.114 ms`, `gpu_kernel_sum=0.012 ms`
  开始时间(ns): `36417343`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8, 7168]]}`
- `q_a_layernorm` -> 0.026 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.026 ms`, `host_to_first_kernel_gap=0.023914`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `36639439`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8, 1536]]}`
- `kv_a_layernorm` -> 0.018 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.018 ms`, `host_to_first_kernel_gap=0.016871`, `host_end_to_last_kernel_tail=0.001 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `36679250`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8, 512]]}`
- `q_b_proj` -> 0.131 ms
  纯GPU kernel时间: `0.020 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.131 ms`, `host_to_first_kernel_gap=0.048651`, `host_end_to_last_kernel_tail=0.007 ms`, `gpu_makespan=0.090 ms`, `gpu_kernel_sum=0.020 ms`
  开始时间(ns): `36720110`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8, 1536]]}`
- `rotary_emb` -> 0.050 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.050 ms`, `host_to_first_kernel_gap=0.043583`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `36947386`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8], [8, 128, 64], [8, 1, 64]]}`
- `attn_mqa` -> 0.155 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.155 ms`, `host_to_first_kernel_gap=0.056184`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.095 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `37023361`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8, 128, 512], [8, 1, 512], [8, 1, 512]]}`
- `o_proj` -> 0.166 ms
  纯GPU kernel时间: `0.045 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.166 ms`, `host_to_first_kernel_gap=0.056606`, `host_end_to_last_kernel_tail=0.030 ms`, `gpu_makespan=0.139 ms`, `gpu_kernel_sum=0.045 ms`
  开始时间(ns): `37253563`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8, 16384]]}`

## Layer 1 / decode / instance 1

- 执行序号: `7`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `0.944 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=0.944 ms`, `module_to_last_kernel=0.965 ms`, `host_to_first_kernel_gap=0.082706`, `host_end_to_last_kernel_tail=0.021 ms`, `gpu_makespan=0.882 ms`, `gpu_kernel_sum=0.097 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.117 ms
  纯GPU kernel时间: `0.012 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.117 ms`, `host_to_first_kernel_gap=0.046486`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.071 ms`, `gpu_kernel_sum=0.012 ms`
  开始时间(ns): `37955235`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8, 7168]]}`
- `q_a_layernorm` -> 0.032 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.032 ms`, `host_to_first_kernel_gap=0.030475`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `38107630`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8, 1536]]}`
- `kv_a_layernorm` -> 0.022 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.022 ms`, `host_to_first_kernel_gap=0.020578`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `38153431`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8, 512]]}`
- `q_b_proj` -> 0.115 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.115 ms`, `host_to_first_kernel_gap=0.044718`, `host_end_to_last_kernel_tail=0.008 ms`, `gpu_makespan=0.078 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `38194347`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8, 1536]]}`
- `rotary_emb` -> 0.046 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.046 ms`, `host_to_first_kernel_gap=0.040715`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `38400718`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8], [8, 128, 64], [8, 1, 64]]}`
- `attn_mqa` -> 0.153 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.153 ms`, `host_to_first_kernel_gap=0.054786`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.095 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `38472983`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8, 128, 512], [8, 1, 512], [8, 1, 512]]}`
- `o_proj` -> 0.154 ms
  纯GPU kernel时间: `0.046 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.154 ms`, `host_to_first_kernel_gap=0.055251`, `host_end_to_last_kernel_tail=0.030 ms`, `gpu_makespan=0.129 ms`, `gpu_kernel_sum=0.046 ms`
  开始时间(ns): `38699558`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8, 16384]]}`

## Layer 2 / decode / instance 1

- 执行序号: `8`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `0.913 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=0.913 ms`, `module_to_last_kernel=0.933 ms`, `host_to_first_kernel_gap=0.08217`, `host_end_to_last_kernel_tail=0.021 ms`, `gpu_makespan=0.851 ms`, `gpu_kernel_sum=0.096 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.118 ms
  纯GPU kernel时间: `0.012 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.118 ms`, `host_to_first_kernel_gap=0.046187`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.072 ms`, `gpu_kernel_sum=0.012 ms`
  开始时间(ns): `39370606`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8, 7168]]}`
- `q_a_layernorm` -> 0.026 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.026 ms`, `host_to_first_kernel_gap=0.023983`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `39522026`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8, 1536]]}`
- `kv_a_layernorm` -> 0.019 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.019 ms`, `host_to_first_kernel_gap=0.017693`, `host_end_to_last_kernel_tail=0.001 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `39561660`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8, 512]]}`
- `q_b_proj` -> 0.110 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.110 ms`, `host_to_first_kernel_gap=0.042109`, `host_end_to_last_kernel_tail=0.007 ms`, `gpu_makespan=0.075 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `39598844`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8, 1536]]}`
- `rotary_emb` -> 0.046 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.046 ms`, `host_to_first_kernel_gap=0.040147`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `39792198`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8], [8, 128, 64], [8, 1, 64]]}`
- `attn_mqa` -> 0.146 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.146 ms`, `host_to_first_kernel_gap=0.053868`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.090 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `39864973`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8, 128, 512], [8, 1, 512], [8, 1, 512]]}`
- `o_proj` -> 0.151 ms
  纯GPU kernel时间: `0.046 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.151 ms`, `host_to_first_kernel_gap=0.05573`, `host_end_to_last_kernel_tail=0.030 ms`, `gpu_makespan=0.126 ms`, `gpu_kernel_sum=0.046 ms`
  开始时间(ns): `40086311`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8, 16384]]}`

## Layer 3 / decode / instance 1

- 执行序号: `9`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `0.902 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=0.902 ms`, `module_to_last_kernel=0.923 ms`, `host_to_first_kernel_gap=0.080222`, `host_end_to_last_kernel_tail=0.021 ms`, `gpu_makespan=0.843 ms`, `gpu_kernel_sum=0.096 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.117 ms
  纯GPU kernel时间: `0.012 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.117 ms`, `host_to_first_kernel_gap=0.045797`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.071 ms`, `gpu_kernel_sum=0.012 ms`
  开始时间(ns): `40758836`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8, 7168]]}`
- `q_a_layernorm` -> 0.026 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.026 ms`, `host_to_first_kernel_gap=0.024138`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `40909999`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8, 1536]]}`
- `kv_a_layernorm` -> 0.020 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.020 ms`, `host_to_first_kernel_gap=0.01729`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `40949743`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8, 512]]}`
- `q_b_proj` -> 0.110 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.110 ms`, `host_to_first_kernel_gap=0.042491`, `host_end_to_last_kernel_tail=0.007 ms`, `gpu_makespan=0.075 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `40988030`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8, 1536]]}`
- `rotary_emb` -> 0.045 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.045 ms`, `host_to_first_kernel_gap=0.040479`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `41178650`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8], [8, 128, 64], [8, 1, 64]]}`
- `attn_mqa` -> 0.147 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.147 ms`, `host_to_first_kernel_gap=0.050109`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.095 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `41248764`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8, 128, 512], [8, 1, 512], [8, 1, 512]]}`
- `o_proj` -> 0.150 ms
  纯GPU kernel时间: `0.046 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.150 ms`, `host_to_first_kernel_gap=0.05477`, `host_end_to_last_kernel_tail=0.031 ms`, `gpu_makespan=0.126 ms`, `gpu_kernel_sum=0.046 ms`
  开始时间(ns): `41466727`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8, 16384]]}`

## Layer 4 / decode / instance 1

- 执行序号: `10`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `0.981 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=0.981 ms`, `module_to_last_kernel=1.003 ms`, `host_to_first_kernel_gap=0.091`, `host_end_to_last_kernel_tail=0.022 ms`, `gpu_makespan=0.912 ms`, `gpu_kernel_sum=0.096 ms`, `gpu_kernel_count=12`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.154 ms
  纯GPU kernel时间: `0.012 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.154 ms`, `host_to_first_kernel_gap=0.053482`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.098 ms`, `gpu_kernel_sum=0.012 ms`
  开始时间(ns): `42482063`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8, 7168]]}`
- `q_a_layernorm` -> 0.029 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.029 ms`, `host_to_first_kernel_gap=0.026791`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `42673394`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8, 1536]]}`
- `kv_a_layernorm` -> 0.019 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.019 ms`, `host_to_first_kernel_gap=0.017273`, `host_end_to_last_kernel_tail=0.001 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `42716512`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8, 512]]}`
- `q_b_proj` -> 0.124 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.124 ms`, `host_to_first_kernel_gap=0.046023`, `host_end_to_last_kernel_tail=0.006 ms`, `gpu_makespan=0.084 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `42755314`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8, 1536]]}`
- `rotary_emb` -> 0.049 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.049 ms`, `host_to_first_kernel_gap=0.043413`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `42970659`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8], [8, 128, 64], [8, 1, 64]]}`
- `attn_mqa` -> 0.149 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `3`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.149 ms`, `host_to_first_kernel_gap=0.053806`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.093 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `43045802`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8, 128, 512], [8, 1, 512], [8, 1, 512]]}`
- `o_proj` -> 0.149 ms
  纯GPU kernel时间: `0.046 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.149 ms`, `host_to_first_kernel_gap=0.059367`, `host_end_to_last_kernel_tail=0.032 ms`, `gpu_makespan=0.122 ms`, `gpu_kernel_sum=0.046 ms`
  开始时间(ns): `43266641`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8, 16384]]}`
