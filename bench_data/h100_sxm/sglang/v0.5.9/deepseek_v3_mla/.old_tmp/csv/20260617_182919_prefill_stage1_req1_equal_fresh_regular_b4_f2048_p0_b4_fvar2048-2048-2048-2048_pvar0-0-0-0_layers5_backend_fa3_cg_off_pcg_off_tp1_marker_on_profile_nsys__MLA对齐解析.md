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
- 整块 MLA-module 时长: `54587.132 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=54587.132 ms`, `module_to_last_kernel=54587.917 ms`, `host_to_first_kernel_gap=3.166136`, `host_end_to_last_kernel_tail=0.786 ms`, `gpu_makespan=54584.751 ms`, `gpu_kernel_sum=3.553 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=8192`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=0`, `sum_seq_after=8192`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 7329.619 ms
  纯GPU kernel时间: `0.277 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=7329.619 ms`, `host_to_first_kernel_gap=2.711145`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=7326.848 ms`, `gpu_kernel_sum=0.277 ms`
  开始时间(ns): `5671077575`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.245 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.245 ms`, `host_to_first_kernel_gap=0.219285`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `13000991898`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 5523.224 ms
  纯GPU kernel时间: `0.452 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=5523.224 ms`, `host_to_first_kernel_gap=0.37628`, `host_end_to_last_kernel_tail=0.162 ms`, `gpu_makespan=5523.010 ms`, `gpu_kernel_sum=0.452 ms`
  开始时间(ns): `13001350519`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.251 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.251 ms`, `host_to_first_kernel_gap=0.212144`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.014 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `18524908293`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 26070.513 ms
  纯GPU kernel时间: `0.105 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=26070.513 ms`, `host_to_first_kernel_gap=26070.374252`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.105 ms`, `gpu_kernel_sum=0.105 ms`
  开始时间(ns): `18525316694`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 7008.487 ms
  纯GPU kernel时间: `0.299 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=7008.487 ms`, `host_to_first_kernel_gap=0.502986`, `host_end_to_last_kernel_tail=0.011 ms`, `gpu_makespan=7007.995 ms`, `gpu_kernel_sum=0.299 ms`
  开始时间(ns): `45915088544`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[8192, 512]]}`
- `attn_mha` -> 20.743 ms
  纯GPU kernel时间: `1.121 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=20.743 ms`, `host_to_first_kernel_gap=4.934141`, `host_end_to_last_kernel_tail=0.985 ms`, `gpu_makespan=16.795 ms`, `gpu_kernel_sum=1.121 ms`
  开始时间(ns): `52930418781`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]}`
- `o_proj` -> 7306.282 ms
  纯GPU kernel时间: `1.268 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=7306.282 ms`, `host_to_first_kernel_gap=0.966749`, `host_end_to_last_kernel_tail=0.840 ms`, `gpu_makespan=7306.155 ms`, `gpu_kernel_sum=1.268 ms`
  开始时间(ns): `52951417689`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 1 / prefill / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `6.285 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=6.285 ms`, `module_to_last_kernel=7.713 ms`, `host_to_first_kernel_gap=0.919509`, `host_end_to_last_kernel_tail=1.428 ms`, `gpu_makespan=6.794 ms`, `gpu_kernel_sum=3.541 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=8192`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=0`, `sum_seq_after=8192`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 1.045 ms
  纯GPU kernel时间: `0.276 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=1.045 ms`, `host_to_first_kernel_gap=0.430152`, `host_end_to_last_kernel_tail=0.096 ms`, `gpu_makespan=0.711 ms`, `gpu_kernel_sum=0.276 ms`
  开始时间(ns): `75319028342`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.175 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.175 ms`, `host_to_first_kernel_gap=0.151525`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `75320252377`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.730 ms
  纯GPU kernel时间: `0.450 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.730 ms`, `host_to_first_kernel_gap=0.284273`, `host_end_to_last_kernel_tail=0.359 ms`, `gpu_makespan=0.804 ms`, `gpu_kernel_sum=0.450 ms`
  开始时间(ns): `75320515661`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.137 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.137 ms`, `host_to_first_kernel_gap=0.123736`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.014 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `75321481094`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.237 ms
  纯GPU kernel时间: `0.107 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.237 ms`, `host_to_first_kernel_gap=0.19478`, `host_end_to_last_kernel_tail=0.065 ms`, `gpu_makespan=0.107 ms`, `gpu_kernel_sum=0.107 ms`
  开始时间(ns): `75321738114`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.719 ms
  纯GPU kernel时间: `0.297 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.719 ms`, `host_to_first_kernel_gap=0.247444`, `host_end_to_last_kernel_tail=0.215 ms`, `gpu_makespan=0.687 ms`, `gpu_kernel_sum=0.297 ms`
  开始时间(ns): `75322449866`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[8192, 512]]}`
- `attn_mha` -> 0.424 ms
  纯GPU kernel时间: `1.120 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.424 ms`, `host_to_first_kernel_gap=0.31405`, `host_end_to_last_kernel_tail=1.059 ms`, `gpu_makespan=1.170 ms`, `gpu_kernel_sum=1.120 ms`
  开始时间(ns): `75323506620`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]}`
- `o_proj` -> 0.726 ms
  纯GPU kernel时间: `1.259 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.726 ms`, `host_to_first_kernel_gap=0.949765`, `host_end_to_last_kernel_tail=1.484 ms`, `gpu_makespan=1.260 ms`, `gpu_kernel_sum=1.259 ms`
  开始时间(ns): `75324042296`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 2 / prefill / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `5.417 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=5.417 ms`, `module_to_last_kernel=7.180 ms`, `host_to_first_kernel_gap=3.381276`, `host_end_to_last_kernel_tail=1.764 ms`, `gpu_makespan=3.799 ms`, `gpu_kernel_sum=3.518 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=8192`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=0`, `sum_seq_after=8192`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.714 ms
  纯GPU kernel时间: `0.269 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.714 ms`, `host_to_first_kernel_gap=3.009572`, `host_end_to_last_kernel_tail=2.566 ms`, `gpu_makespan=0.270 ms`, `gpu_kernel_sum=0.269 ms`
  开始时间(ns): `75327867320`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.126 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.126 ms`, `host_to_first_kernel_gap=2.427783`, `host_end_to_last_kernel_tail=2.319 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `75328719797`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.643 ms
  纯GPU kernel时间: `0.450 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.643 ms`, `host_to_first_kernel_gap=2.229955`, `host_end_to_last_kernel_tail=2.038 ms`, `gpu_makespan=0.452 ms`, `gpu_kernel_sum=0.450 ms`
  开始时间(ns): `75328936633`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.122 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.122 ms`, `host_to_first_kernel_gap=1.823109`, `host_end_to_last_kernel_tail=1.714 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `75329794903`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.213 ms
  纯GPU kernel时间: `0.107 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.213 ms`, `host_to_first_kernel_gap=1.607838`, `host_end_to_last_kernel_tail=1.502 ms`, `gpu_makespan=0.107 ms`, `gpu_kernel_sum=0.107 ms`
  开始时间(ns): `75330024510`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.671 ms
  纯GPU kernel时间: `0.285 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.671 ms`, `host_to_first_kernel_gap=1.125379`, `host_end_to_last_kernel_tail=0.740 ms`, `gpu_makespan=0.286 ms`, `gpu_kernel_sum=0.285 ms`
  开始时间(ns): `75330642905`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[8192, 512]]}`
- `attn_mha` -> 0.425 ms
  纯GPU kernel时间: `1.118 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.425 ms`, `host_to_first_kernel_gap=0.706515`, `host_end_to_last_kernel_tail=1.400 ms`, `gpu_makespan=1.118 ms`, `gpu_kernel_sum=1.118 ms`
  开始时间(ns): `75331590601`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]}`
- `o_proj` -> 0.743 ms
  纯GPU kernel时间: `1.258 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.743 ms`, `host_to_first_kernel_gap=1.284702`, `host_end_to_last_kernel_tail=1.800 ms`, `gpu_makespan=1.259 ms`, `gpu_kernel_sum=1.258 ms`
  开始时间(ns): `75332132190`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 3 / prefill / instance 1

- 执行序号: `4`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `5.458 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=5.458 ms`, `module_to_last_kernel=7.699 ms`, `host_to_first_kernel_gap=3.904895`, `host_end_to_last_kernel_tail=2.242 ms`, `gpu_makespan=3.794 ms`, `gpu_kernel_sum=3.513 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=8192`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=0`, `sum_seq_after=8192`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.683 ms
  纯GPU kernel时间: `0.268 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.683 ms`, `host_to_first_kernel_gap=3.589225`, `host_end_to_last_kernel_tail=3.175 ms`, `gpu_makespan=0.269 ms`, `gpu_kernel_sum=0.268 ms`
  开始时间(ns): `75335713682`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.179 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.179 ms`, `host_to_first_kernel_gap=3.008223`, `host_end_to_last_kernel_tail=2.847 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `75336564028`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.712 ms
  纯GPU kernel时间: `0.445 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.712 ms`, `host_to_first_kernel_gap=2.746509`, `host_end_to_last_kernel_tail=2.481 ms`, `gpu_makespan=0.447 ms`, `gpu_kernel_sum=0.445 ms`
  开始时间(ns): `75336844846`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.135 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.135 ms`, `host_to_first_kernel_gap=2.269392`, `host_end_to_last_kernel_tail=2.147 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `75337768715`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.194 ms
  纯GPU kernel时间: `0.107 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.194 ms`, `host_to_first_kernel_gap=2.035098`, `host_end_to_last_kernel_tail=1.948 ms`, `gpu_makespan=0.107 ms`, `gpu_kernel_sum=0.107 ms`
  开始时间(ns): `75338018145`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.706 ms
  纯GPU kernel时间: `0.295 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.706 ms`, `host_to_first_kernel_gap=1.623564`, `host_end_to_last_kernel_tail=1.215 ms`, `gpu_makespan=0.297 ms`, `gpu_kernel_sum=0.295 ms`
  开始时间(ns): `75338564879`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[8192, 512]]}`
- `attn_mha` -> 0.357 ms
  纯GPU kernel时间: `1.126 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.357 ms`, `host_to_first_kernel_gap=1.134163`, `host_end_to_last_kernel_tail=1.902 ms`, `gpu_makespan=1.125 ms`, `gpu_kernel_sum=1.126 ms`
  开始时间(ns): `75339593064`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]}`
- `o_proj` -> 0.720 ms
  纯GPU kernel时间: `1.241 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.720 ms`, `host_to_first_kernel_gap=1.782612`, `host_end_to_last_kernel_tail=2.305 ms`, `gpu_makespan=1.243 ms`, `gpu_kernel_sum=1.241 ms`
  开始时间(ns): `75340071846`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 4 / prefill / instance 1

- 执行序号: `5`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `2.382 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.382 ms`, `module_to_last_kernel=4.661 ms`, `host_to_first_kernel_gap=0.401813`, `host_end_to_last_kernel_tail=2.279 ms`, `gpu_makespan=4.259 ms`, `gpu_kernel_sum=3.524 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=8192`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=0`, `sum_seq_after=8192`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 2048, 'prefix_len': 0, 'seq_len_after': 2048, 'prompt_len': 2048, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.460 ms
  纯GPU kernel时间: `0.275 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.460 ms`, `host_to_first_kernel_gap=0.175249`, `host_end_to_last_kernel_tail=0.166 ms`, `gpu_makespan=0.451 ms`, `gpu_kernel_sum=0.275 ms`
  开始时间(ns): `77984898928`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.074 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.074 ms`, `host_to_first_kernel_gap=0.092728`, `host_end_to_last_kernel_tail=0.037 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `77985432233`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.248 ms
  纯GPU kernel时间: `0.449 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.248 ms`, `host_to_first_kernel_gap=0.092123`, `host_end_to_last_kernel_tail=0.407 ms`, `gpu_makespan=0.563 ms`, `gpu_kernel_sum=0.449 ms`
  开始时间(ns): `77985537734`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.043 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.043 ms`, `host_to_first_kernel_gap=0.342644`, `host_end_to_last_kernel_tail=0.313 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `77985849645`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.112 ms
  纯GPU kernel时间: `0.107 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.112 ms`, `host_to_first_kernel_gap=0.268817`, `host_end_to_last_kernel_tail=0.263 ms`, `gpu_makespan=0.107 ms`, `gpu_kernel_sum=0.107 ms`
  开始时间(ns): `77985938096`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.244 ms
  纯GPU kernel时间: `0.292 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.244 ms`, `host_to_first_kernel_gap=0.129408`, `host_end_to_last_kernel_tail=0.262 ms`, `gpu_makespan=0.376 ms`, `gpu_kernel_sum=0.292 ms`
  开始时间(ns): `77986212129`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[8192, 512]]}`
- `attn_mha` -> 0.181 ms
  纯GPU kernel时间: `1.117 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.181 ms`, `host_to_first_kernel_gap=0.395041`, `host_end_to_last_kernel_tail=1.331 ms`, `gpu_makespan=1.117 ms`, `gpu_kernel_sum=1.117 ms`
  开始时间(ns): `77986565408`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]}`
- `o_proj` -> 0.249 ms
  纯GPU kernel时间: `1.253 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.249 ms`, `host_to_first_kernel_gap=1.289284`, `host_end_to_last_kernel_tail=2.294 ms`, `gpu_makespan=1.254 ms`, `gpu_kernel_sum=1.253 ms`
  开始时间(ns): `77986790397`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 0 / decode / instance 1

- 执行序号: `6`
- Collector对齐边界: `{'Module': 'model.model.layers.0.self_attn'}`
- 整块 MLA-module 时长: `20167.914 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=20167.914 ms`, `module_to_last_kernel=20167.914 ms`, `host_to_first_kernel_gap=0.301224`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=20167.302 ms`, `gpu_kernel_sum=0.122 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 5336.140 ms
  纯GPU kernel时间: `0.016 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=5336.140 ms`, `host_to_first_kernel_gap=0.182917`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=5335.690 ms`, `gpu_kernel_sum=0.016 ms`
  开始时间(ns): `80004267611`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4, 7168]]}`
- `q_a_layernorm` -> 0.248 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.248 ms`, `host_to_first_kernel_gap=0.219223`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `85340777797`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4, 1536]]}`
- `kv_a_layernorm` -> 0.111 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.111 ms`, `host_to_first_kernel_gap=0.087488`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `85341114716`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4, 512]]}`
- `q_b_proj` -> 8102.402 ms
  纯GPU kernel时间: `0.021 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=8102.402 ms`, `host_to_first_kernel_gap=0.362881`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=8101.797 ms`, `gpu_kernel_sum=0.021 ms`
  开始时间(ns): `85341364731`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4, 1536]]}`
- `rotary_emb` -> 0.335 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.335 ms`, `host_to_first_kernel_gap=0.284749`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `93453221543`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4], [4, 128, 64], [4, 1, 64]]}`
- `attn_mqa` -> 28.757 ms
  纯GPU kernel时间: `0.029 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=28.757 ms`, `host_to_first_kernel_gap=0.304164`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=28.268 ms`, `gpu_kernel_sum=0.029 ms`
  开始时间(ns): `93453722512`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4, 128, 512], [4, 1, 512], [4, 1, 512]]}`
- `o_proj` -> 6682.245 ms
  纯GPU kernel时间: `0.051 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=6682.245 ms`, `host_to_first_kernel_gap=0.356957`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=6681.666 ms`, `gpu_kernel_sum=0.051 ms`
  开始时间(ns): `93489728946`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4, 16384]]}`

## Layer 1 / decode / instance 1

- 执行序号: `7`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `3.760 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=3.760 ms`, `module_to_last_kernel=3.760 ms`, `host_to_first_kernel_gap=0.43745`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=3.307 ms`, `gpu_kernel_sum=0.120 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.693 ms
  纯GPU kernel时间: `0.016 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.693 ms`, `host_to_first_kernel_gap=0.283969`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.366 ms`, `gpu_kernel_sum=0.016 ms`
  开始时间(ns): `113170166928`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4, 7168]]}`
- `q_a_layernorm` -> 0.115 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.115 ms`, `host_to_first_kernel_gap=0.099166`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `113171031699`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4, 1536]]}`
- `kv_a_layernorm` -> 0.060 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.060 ms`, `host_to_first_kernel_gap=0.049968`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `113171194561`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4, 512]]}`
- `q_b_proj` -> 0.470 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.470 ms`, `host_to_first_kernel_gap=0.164536`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.273 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `113171315833`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4, 1536]]}`
- `rotary_emb` -> 0.188 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.188 ms`, `host_to_first_kernel_gap=0.15141`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `113172122943`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4], [4, 128, 64], [4, 1, 64]]}`
- `attn_mqa` -> 0.595 ms
  纯GPU kernel时间: `0.029 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.595 ms`, `host_to_first_kernel_gap=0.174852`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.373 ms`, `gpu_kernel_sum=0.029 ms`
  开始时间(ns): `113172416973`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4, 128, 512], [4, 1, 512], [4, 1, 512]]}`
- `o_proj` -> 0.492 ms
  纯GPU kernel时间: `0.050 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.492 ms`, `host_to_first_kernel_gap=0.19633`, `host_end_to_last_kernel_tail=0.007 ms`, `gpu_makespan=0.303 ms`, `gpu_kernel_sum=0.050 ms`
  开始时间(ns): `113173257799`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4, 16384]]}`

## Layer 2 / decode / instance 1

- 执行序号: `8`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `2.935 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.935 ms`, `module_to_last_kernel=2.935 ms`, `host_to_first_kernel_gap=0.258896`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=2.646 ms`, `gpu_kernel_sum=0.119 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.413 ms
  纯GPU kernel时间: `0.016 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.413 ms`, `host_to_first_kernel_gap=0.137161`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.246 ms`, `gpu_kernel_sum=0.016 ms`
  开始时间(ns): `113175477767`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4, 7168]]}`
- `q_a_layernorm` -> 0.085 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.085 ms`, `host_to_first_kernel_gap=0.071763`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `113176006845`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4, 1536]]}`
- `kv_a_layernorm` -> 0.045 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.045 ms`, `host_to_first_kernel_gap=0.039662`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `113176133282`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4, 512]]}`
- `q_b_proj` -> 0.428 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.428 ms`, `host_to_first_kernel_gap=0.163211`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.241 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `113176232581`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4, 1536]]}`
- `rotary_emb` -> 0.140 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.140 ms`, `host_to_first_kernel_gap=0.116813`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `113176905795`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4], [4, 128, 64], [4, 1, 64]]}`
- `attn_mqa` -> 0.471 ms
  纯GPU kernel时间: `0.029 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.471 ms`, `host_to_first_kernel_gap=0.147932`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.280 ms`, `gpu_kernel_sum=0.029 ms`
  开始时间(ns): `113177129780`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4, 128, 512], [4, 1, 512], [4, 1, 512]]}`
- `o_proj` -> 0.438 ms
  纯GPU kernel时间: `0.050 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.438 ms`, `host_to_first_kernel_gap=0.153341`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.284 ms`, `gpu_kernel_sum=0.050 ms`
  开始时间(ns): `113177823123`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4, 16384]]}`

## Layer 3 / decode / instance 1

- 执行序号: `9`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `2.742 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.742 ms`, `module_to_last_kernel=2.742 ms`, `host_to_first_kernel_gap=0.218166`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=2.487 ms`, `gpu_kernel_sum=0.121 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.395 ms
  纯GPU kernel时间: `0.016 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.395 ms`, `host_to_first_kernel_gap=0.141394`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.230 ms`, `gpu_kernel_sum=0.016 ms`
  开始时间(ns): `113179780542`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4, 7168]]}`
- `q_a_layernorm` -> 0.067 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.067 ms`, `host_to_first_kernel_gap=0.057996`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `113180276772`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4, 1536]]}`
- `kv_a_layernorm` -> 0.057 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.057 ms`, `host_to_first_kernel_gap=0.047831`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `113180386873`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4, 512]]}`
- `q_b_proj` -> 0.400 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.400 ms`, `host_to_first_kernel_gap=0.147335`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.233 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `113180508841`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4, 1536]]}`
- `rotary_emb` -> 0.125 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.125 ms`, `host_to_first_kernel_gap=0.105576`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `113181142376`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4], [4, 128, 64], [4, 1, 64]]}`
- `attn_mqa` -> 0.433 ms
  纯GPU kernel时间: `0.029 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.433 ms`, `host_to_first_kernel_gap=0.139819`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.257 ms`, `gpu_kernel_sum=0.029 ms`
  开始时间(ns): `113181344293`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4, 128, 512], [4, 1, 512], [4, 1, 512]]}`
- `o_proj` -> 0.428 ms
  纯GPU kernel时间: `0.051 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.428 ms`, `host_to_first_kernel_gap=0.137831`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.284 ms`, `gpu_kernel_sum=0.051 ms`
  开始时间(ns): `113181986953`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4, 16384]]}`

## Layer 4 / decode / instance 1

- 执行序号: `10`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `2.244 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.244 ms`, `module_to_last_kernel=2.244 ms`, `host_to_first_kernel_gap=0.255957`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=1.988 ms`, `gpu_kernel_sum=0.124 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.416 ms
  纯GPU kernel时间: `0.016 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.416 ms`, `host_to_first_kernel_gap=0.153035`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.245 ms`, `gpu_kernel_sum=0.016 ms`
  开始时间(ns): `116826384978`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4, 7168]]}`
- `q_a_layernorm` -> 0.065 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.065 ms`, `host_to_first_kernel_gap=0.057704`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `116826885781`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4, 1536]]}`
- `kv_a_layernorm` -> 0.032 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.032 ms`, `host_to_first_kernel_gap=0.028143`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `116826976814`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4, 512]]}`
- `q_b_proj` -> 0.249 ms
  纯GPU kernel时间: `0.023 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.249 ms`, `host_to_first_kernel_gap=0.092001`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.152 ms`, `gpu_kernel_sum=0.023 ms`
  开始时间(ns): `116827051260`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4, 1536]]}`
- `rotary_emb` -> 0.113 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.113 ms`, `host_to_first_kernel_gap=0.096199`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `116827527222`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4], [4, 128, 64], [4, 1, 64]]}`
- `attn_mqa` -> 0.363 ms
  纯GPU kernel时间: `0.029 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.363 ms`, `host_to_first_kernel_gap=0.118922`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.227 ms`, `gpu_kernel_sum=0.029 ms`
  开始时间(ns): `116827696819`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4, 128, 512], [4, 1, 512], [4, 1, 512]]}`
- `o_proj` -> 0.303 ms
  纯GPU kernel时间: `0.050 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.303 ms`, `host_to_first_kernel_gap=0.104185`, `host_end_to_last_kernel_tail=0.019 ms`, `gpu_makespan=0.218 ms`, `gpu_kernel_sum=0.050 ms`
  开始时间(ns): `116828204003`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4, 16384]]}`
