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
- 整块 MLA-module 时长: `543.596 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=543.596 ms`, `module_to_last_kernel=545.103 ms`, `host_to_first_kernel_gap=0.377558`, `host_end_to_last_kernel_tail=1.508 ms`, `gpu_makespan=544.726 ms`, `gpu_kernel_sum=2.582 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4500`, `total_tokens=4500`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[4500, 128, 192], [4500, 128, 192], [4500, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=4500`, `sum_prefix=0`, `sum_seq_after=4500`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 4500, 'prefix_len': 0, 'seq_len_after': 4500, 'prompt_len': 4500, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.507 ms
  纯GPU kernel时间: `0.133 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.507 ms`, `host_to_first_kernel_gap=0.168702`, `host_end_to_last_kernel_tail=0.047 ms`, `gpu_makespan=0.385 ms`, `gpu_kernel_sum=0.133 ms`
  开始时间(ns): `480007991`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4500, 7168]]}`
- `q_a_layernorm` -> 0.064 ms
  纯GPU kernel时间: `0.010 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.064 ms`, `host_to_first_kernel_gap=0.055434`, `host_end_to_last_kernel_tail=0.001 ms`, `gpu_makespan=0.010 ms`, `gpu_kernel_sum=0.010 ms`
  开始时间(ns): `480593931`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4500, 1536]]}`
- `q_b_proj` -> 0.322 ms
  纯GPU kernel时间: `0.266 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.322 ms`, `host_to_first_kernel_gap=0.114186`, `host_end_to_last_kernel_tail=0.223 ms`, `gpu_makespan=0.430 ms`, `gpu_kernel_sum=0.266 ms`
  开始时间(ns): `480704139`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4500, 1536]]}`
- `kv_a_layernorm` -> 0.055 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.055 ms`, `host_to_first_kernel_gap=0.137501`, `host_end_to_last_kernel_tail=0.091 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `481110584`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4500, 512]]}`
- `rotary_emb` -> 0.128 ms
  纯GPU kernel时间: `0.058 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.128 ms`, `host_to_first_kernel_gap=0.104704`, `host_end_to_last_kernel_tail=0.035 ms`, `gpu_makespan=0.058 ms`, `gpu_kernel_sum=0.058 ms`
  开始时间(ns): `481228149`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4500], [4500, 128, 64], [4500, 1, 64]]}`
- `kv_b_proj` -> 0.668 ms
  纯GPU kernel时间: `0.163 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.668 ms`, `host_to_first_kernel_gap=0.312864`, `host_end_to_last_kernel_tail=0.107 ms`, `gpu_makespan=0.462 ms`, `gpu_kernel_sum=0.163 ms`
  开始时间(ns): `1021905353`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[4500, 512]]}`
- `attn_mha` -> 0.227 ms
  纯GPU kernel时间: `1.221 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.227 ms`, `host_to_first_kernel_gap=0.171729`, `host_end_to_last_kernel_tail=1.193 ms`, `gpu_makespan=1.248 ms`, `gpu_kernel_sum=1.221 ms`
  开始时间(ns): `1022758744`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4500, 128, 192], [4500, 128, 192], [4500, 128, 128]]}`
- `o_proj` -> 0.327 ms
  纯GPU kernel时间: `0.722 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.327 ms`, `host_to_first_kernel_gap=1.131897`, `host_end_to_last_kernel_tail=1.527 ms`, `gpu_makespan=0.723 ms`, `gpu_kernel_sum=0.722 ms`
  开始时间(ns): `1023047825`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4500, 16384]]}`

## Layer 1 / prefill / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `2.184 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.184 ms`, `module_to_last_kernel=5.830 ms`, `host_to_first_kernel_gap=3.096407`, `host_end_to_last_kernel_tail=3.646 ms`, `gpu_makespan=2.733 ms`, `gpu_kernel_sum=2.568 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4500`, `total_tokens=4500`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[4500, 128, 192], [4500, 128, 192], [4500, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=4500`, `sum_prefix=0`, `sum_seq_after=4500`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 4500, 'prefix_len': 0, 'seq_len_after': 4500, 'prompt_len': 4500, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.257 ms
  纯GPU kernel时间: `0.133 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.257 ms`, `host_to_first_kernel_gap=2.909696`, `host_end_to_last_kernel_tail=2.788 ms`, `gpu_makespan=0.135 ms`, `gpu_kernel_sum=0.133 ms`
  开始时间(ns): `1024628523`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4500, 7168]]}`
- `q_a_layernorm` -> 0.064 ms
  纯GPU kernel时间: `0.010 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.064 ms`, `host_to_first_kernel_gap=2.725669`, `host_end_to_last_kernel_tail=2.672 ms`, `gpu_makespan=0.010 ms`, `gpu_kernel_sum=0.010 ms`
  开始时间(ns): `1024948391`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4500, 1536]]}`
- `q_b_proj` -> 0.245 ms
  纯GPU kernel时间: `0.260 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.245 ms`, `host_to_first_kernel_gap=2.639066`, `host_end_to_last_kernel_tail=2.657 ms`, `gpu_makespan=0.262 ms`, `gpu_kernel_sum=0.260 ms`
  开始时间(ns): `1025046066`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4500, 1536]]}`
- `kv_a_layernorm` -> 0.048 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.048 ms`, `host_to_first_kernel_gap=2.589011`, `host_end_to_last_kernel_tail=2.550 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `1025358265`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4500, 512]]}`
- `rotary_emb` -> 0.120 ms
  纯GPU kernel时间: `0.058 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.120 ms`, `host_to_first_kernel_gap=2.505517`, `host_end_to_last_kernel_tail=2.444 ms`, `gpu_makespan=0.058 ms`, `gpu_kernel_sum=0.058 ms`
  开始时间(ns): `1025451583`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4500], [4500, 128, 64], [4500, 1, 64]]}`
- `kv_b_proj` -> 0.240 ms
  纯GPU kernel时间: `0.150 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.240 ms`, `host_to_first_kernel_gap=2.280989`, `host_end_to_last_kernel_tail=2.194 ms`, `gpu_makespan=0.153 ms`, `gpu_kernel_sum=0.150 ms`
  开始时间(ns): `1025752783`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[4500, 512]]}`
- `attn_mha` -> 0.175 ms
  纯GPU kernel时间: `1.226 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.175 ms`, `host_to_first_kernel_gap=2.215582`, `host_end_to_last_kernel_tail=3.267 ms`, `gpu_makespan=1.226 ms`, `gpu_kernel_sum=1.226 ms`
  开始时间(ns): `1026104494`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4500, 128, 192], [4500, 128, 192], [4500, 128, 128]]}`
- `o_proj` -> 0.279 ms
  纯GPU kernel时间: `0.722 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.279 ms`, `host_to_first_kernel_gap=3.217588`, `host_end_to_last_kernel_tail=3.662 ms`, `gpu_makespan=0.723 ms`, `gpu_kernel_sum=0.722 ms`
  开始时间(ns): `1026331192`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4500, 16384]]}`

## Layer 2 / prefill / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `2.026 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.026 ms`, `module_to_last_kernel=8.059 ms`, `host_to_first_kernel_gap=5.335441`, `host_end_to_last_kernel_tail=6.033 ms`, `gpu_makespan=2.723 ms`, `gpu_kernel_sum=2.558 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4500`, `total_tokens=4500`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[4500, 128, 192], [4500, 128, 192], [4500, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=4500`, `sum_prefix=0`, `sum_seq_after=4500`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 4500, 'prefix_len': 0, 'seq_len_after': 4500, 'prompt_len': 4500, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.240 ms
  纯GPU kernel时间: `0.132 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.240 ms`, `host_to_first_kernel_gap=5.211611`, `host_end_to_last_kernel_tail=5.104 ms`, `gpu_makespan=0.133 ms`, `gpu_kernel_sum=0.132 ms`
  开始时间(ns): `1027695667`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4500, 7168]]}`
- `q_a_layernorm` -> 0.055 ms
  纯GPU kernel时间: `0.010 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.055 ms`, `host_to_first_kernel_gap=5.044803`, `host_end_to_last_kernel_tail=5.000 ms`, `gpu_makespan=0.010 ms`, `gpu_kernel_sum=0.010 ms`
  开始时间(ns): `1027995883`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4500, 1536]]}`
- `q_b_proj` -> 0.238 ms
  纯GPU kernel时间: `0.257 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.238 ms`, `host_to_first_kernel_gap=4.969427`, `host_end_to_last_kernel_tail=4.989 ms`, `gpu_makespan=0.258 ms`, `gpu_kernel_sum=0.257 ms`
  开始时间(ns): `1028083483`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4500, 1536]]}`
- `kv_a_layernorm` -> 0.044 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.044 ms`, `host_to_first_kernel_gap=4.922953`, `host_end_to_last_kernel_tail=4.888 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `1028387525`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4500, 512]]}`
- `rotary_emb` -> 0.086 ms
  纯GPU kernel时间: `0.059 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.086 ms`, `host_to_first_kernel_gap=4.841104`, `host_end_to_last_kernel_tail=4.815 ms`, `gpu_makespan=0.059 ms`, `gpu_kernel_sum=0.059 ms`
  开始时间(ns): `1028480670`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4500], [4500, 128, 64], [4500, 1, 64]]}`
- `kv_b_proj` -> 0.268 ms
  纯GPU kernel时间: `0.153 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.268 ms`, `host_to_first_kernel_gap=4.681488`, `host_end_to_last_kernel_tail=4.568 ms`, `gpu_makespan=0.154 ms`, `gpu_kernel_sum=0.153 ms`
  开始时间(ns): `1028717374`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[4500, 512]]}`
- `attn_mha` -> 0.169 ms
  纯GPU kernel时间: `1.223 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.169 ms`, `host_to_first_kernel_gap=4.593302`, `host_end_to_last_kernel_tail=5.647 ms`, `gpu_makespan=1.223 ms`, `gpu_kernel_sum=1.223 ms`
  开始时间(ns): `1029095128`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4500, 128, 192], [4500, 128, 192], [4500, 128, 128]]}`
- `o_proj` -> 0.271 ms
  纯GPU kernel时间: `0.716 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.271 ms`, `host_to_first_kernel_gap=5.602373`, `host_end_to_last_kernel_tail=6.049 ms`, `gpu_makespan=0.718 ms`, `gpu_kernel_sum=0.716 ms`
  开始时间(ns): `1029310378`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4500, 16384]]}`

## Layer 3 / prefill / instance 1

- 执行序号: `4`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `1.981 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.981 ms`, `module_to_last_kernel=10.461 ms`, `host_to_first_kernel_gap=7.737435`, `host_end_to_last_kernel_tail=8.481 ms`, `gpu_makespan=2.724 ms`, `gpu_kernel_sum=2.559 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4500`, `total_tokens=4500`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[4500, 128, 192], [4500, 128, 192], [4500, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=4500`, `sum_prefix=0`, `sum_seq_after=4500`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 4500, 'prefix_len': 0, 'seq_len_after': 4500, 'prompt_len': 4500, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.240 ms
  纯GPU kernel时间: `0.133 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.240 ms`, `host_to_first_kernel_gap=7.617631`, `host_end_to_last_kernel_tail=7.513 ms`, `gpu_makespan=0.136 ms`, `gpu_kernel_sum=0.133 ms`
  开始时间(ns): `1030647538`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4500, 7168]]}`
- `q_a_layernorm` -> 0.050 ms
  纯GPU kernel时间: `0.010 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.050 ms`, `host_to_first_kernel_gap=7.456155`, `host_end_to_last_kernel_tail=7.416 ms`, `gpu_makespan=0.010 ms`, `gpu_kernel_sum=0.010 ms`
  开始时间(ns): `1030945110`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4500, 1536]]}`
- `q_b_proj` -> 0.231 ms
  纯GPU kernel时间: `0.258 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.231 ms`, `host_to_first_kernel_gap=7.384954`, `host_end_to_last_kernel_tail=7.414 ms`, `gpu_makespan=0.261 ms`, `gpu_kernel_sum=0.258 ms`
  开始时间(ns): `1031027319`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4500, 1536]]}`
- `kv_a_layernorm` -> 0.049 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.049 ms`, `host_to_first_kernel_gap=7.34994`, `host_end_to_last_kernel_tail=7.310 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `1031322813`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4500, 512]]}`
- `rotary_emb` -> 0.083 ms
  纯GPU kernel时间: `0.058 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.083 ms`, `host_to_first_kernel_gap=7.262039`, `host_end_to_last_kernel_tail=7.237 ms`, `gpu_makespan=0.058 ms`, `gpu_kernel_sum=0.058 ms`
  开始时间(ns): `1031420442`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4500], [4500, 128, 64], [4500, 1, 64]]}`
- `kv_b_proj` -> 0.261 ms
  纯GPU kernel时间: `0.152 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.261 ms`, `host_to_first_kernel_gap=7.11069`, `host_end_to_last_kernel_tail=7.004 ms`, `gpu_makespan=0.154 ms`, `gpu_kernel_sum=0.152 ms`
  开始时间(ns): `1031648495`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[4500, 512]]}`
- `attn_mha` -> 0.155 ms
  纯GPU kernel时间: `1.223 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.155 ms`, `host_to_first_kernel_gap=7.030119`, `host_end_to_last_kernel_tail=8.098 ms`, `gpu_makespan=1.223 ms`, `gpu_kernel_sum=1.223 ms`
  开始时间(ns): `1032017034`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4500, 128, 192], [4500, 128, 192], [4500, 128, 128]]}`
- `o_proj` -> 0.265 ms
  纯GPU kernel时间: `0.716 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.265 ms`, `host_to_first_kernel_gap=8.044777`, `host_end_to_last_kernel_tail=8.497 ms`, `gpu_makespan=0.717 ms`, `gpu_kernel_sum=0.716 ms`
  开始时间(ns): `1032227497`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4500, 16384]]}`

## Layer 4 / prefill / instance 1

- 执行序号: `5`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `2.094 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.094 ms`, `module_to_last_kernel=15.209 ms`, `host_to_first_kernel_gap=12.48825`, `host_end_to_last_kernel_tail=13.115 ms`, `gpu_makespan=2.721 ms`, `gpu_kernel_sum=2.556 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4500`, `total_tokens=4500`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[4500, 128, 192], [4500, 128, 192], [4500, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=4500`, `sum_prefix=0`, `sum_seq_after=4500`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 4500, 'prefix_len': 0, 'seq_len_after': 4500, 'prompt_len': 4500, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.307 ms
  纯GPU kernel时间: `0.133 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.307 ms`, `host_to_first_kernel_gap=12.353678`, `host_end_to_last_kernel_tail=12.182 ms`, `gpu_makespan=0.135 ms`, `gpu_kernel_sum=0.133 ms`
  开始时间(ns): `1034339559`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4500, 7168]]}`
- `q_a_layernorm` -> 0.052 ms
  纯GPU kernel时间: `0.010 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.052 ms`, `host_to_first_kernel_gap=12.125712`, `host_end_to_last_kernel_tail=12.083 ms`, `gpu_makespan=0.010 ms`, `gpu_kernel_sum=0.010 ms`
  开始时间(ns): `1034703557`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4500, 1536]]}`
- `q_b_proj` -> 0.257 ms
  纯GPU kernel时间: `0.258 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.257 ms`, `host_to_first_kernel_gap=12.04878`, `host_end_to_last_kernel_tail=12.050 ms`, `gpu_makespan=0.259 ms`, `gpu_kernel_sum=0.258 ms`
  开始时间(ns): `1034792233`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4500, 1536]]}`
- `kv_a_layernorm` -> 0.049 ms
  纯GPU kernel时间: `0.008 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.049 ms`, `host_to_first_kernel_gap=11.982037`, `host_end_to_last_kernel_tail=11.942 ms`, `gpu_makespan=0.008 ms`, `gpu_kernel_sum=0.008 ms`
  开始时间(ns): `1035117504`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4500, 512]]}`
- `rotary_emb` -> 0.094 ms
  纯GPU kernel时间: `0.059 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.094 ms`, `host_to_first_kernel_gap=11.890818`, `host_end_to_last_kernel_tail=11.855 ms`, `gpu_makespan=0.059 ms`, `gpu_kernel_sum=0.059 ms`
  开始时间(ns): `1035218323`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4500], [4500, 128, 64], [4500, 1, 64]]}`
- `kv_b_proj` -> 0.251 ms
  纯GPU kernel时间: `0.152 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.251 ms`, `host_to_first_kernel_gap=11.724607`, `host_end_to_last_kernel_tail=11.627 ms`, `gpu_makespan=0.153 ms`, `gpu_kernel_sum=0.152 ms`
  开始时间(ns): `1035461686`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[4500, 512]]}`
- `attn_mha` -> 0.156 ms
  纯GPU kernel时间: `1.221 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.156 ms`, `host_to_first_kernel_gap=11.653829`, `host_end_to_last_kernel_tail=12.719 ms`, `gpu_makespan=1.221 ms`, `gpu_kernel_sum=1.221 ms`
  开始时间(ns): `1035822384`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4500, 128, 192], [4500, 128, 192], [4500, 128, 128]]}`
- `o_proj` -> 0.255 ms
  纯GPU kernel时间: `0.715 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.255 ms`, `host_to_first_kernel_gap=12.676884`, `host_end_to_last_kernel_tail=13.137 ms`, `gpu_makespan=0.716 ms`, `gpu_kernel_sum=0.715 ms`
  开始时间(ns): `1036021666`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4500, 16384]]}`

## Layer 0 / decode / instance 1

- 执行序号: `6`
- Collector对齐边界: `{'Module': 'model.model.layers.0.self_attn'}`
- 整块 MLA-module 时长: `2.107 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.107 ms`, `module_to_last_kernel=15.162 ms`, `host_to_first_kernel_gap=15.012353`, `host_end_to_last_kernel_tail=13.055 ms`, `gpu_makespan=0.149 ms`, `gpu_kernel_sum=0.117 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.358 ms
  纯GPU kernel时间: `0.017 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.358 ms`, `host_to_first_kernel_gap=14.93201`, `host_end_to_last_kernel_tail=14.592 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.017 ms`
  开始时间(ns): `1040958607`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.055 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.055 ms`, `host_to_first_kernel_gap=14.522061`, `host_end_to_last_kernel_tail=14.469 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1041386508`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.034 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.034 ms`, `host_to_first_kernel_gap=14.438904`, `host_end_to_last_kernel_tail=14.406 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1041471553`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.270 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.270 ms`, `host_to_first_kernel_gap=14.367704`, `host_end_to_last_kernel_tail=14.118 ms`, `gpu_makespan=0.020 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `1041545697`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.092 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.092 ms`, `host_to_first_kernel_gap=13.94701`, `host_end_to_last_kernel_tail=13.857 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1041998359`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.352 ms
  纯GPU kernel时间: `0.028 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.352 ms`, `host_to_first_kernel_gap=13.804334`, `host_end_to_last_kernel_tail=13.482 ms`, `gpu_makespan=0.030 ms`, `gpu_kernel_sum=0.028 ms`
  开始时间(ns): `1042143979`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.317 ms
  纯GPU kernel时间: `0.049 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.317 ms`, `host_to_first_kernel_gap=13.340514`, `host_end_to_last_kernel_tail=13.074 ms`, `gpu_makespan=0.050 ms`, `gpu_kernel_sum=0.049 ms`
  开始时间(ns): `1042649239`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 1 / decode / instance 1

- 执行序号: `7`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `1.836 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.836 ms`, `module_to_last_kernel=12.391 ms`, `host_to_first_kernel_gap=12.244082`, `host_end_to_last_kernel_tail=10.555 ms`, `gpu_makespan=0.147 ms`, `gpu_kernel_sum=0.117 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.251 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.251 ms`, `host_to_first_kernel_gap=12.178226`, `host_end_to_last_kernel_tail=11.943 ms`, `gpu_makespan=0.016 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `1044023879`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.049 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.049 ms`, `host_to_first_kernel_gap=11.880648`, `host_end_to_last_kernel_tail=11.834 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1044337265`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.037 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.037 ms`, `host_to_first_kernel_gap=11.80823`, `host_end_to_last_kernel_tail=11.773 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1044411763`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.236 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.236 ms`, `host_to_first_kernel_gap=11.739286`, `host_end_to_last_kernel_tail=11.521 ms`, `gpu_makespan=0.019 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `1044484547`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.083 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.083 ms`, `host_to_first_kernel_gap=11.378338`, `host_end_to_last_kernel_tail=11.297 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1044875703`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.317 ms
  纯GPU kernel时间: `0.028 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.317 ms`, `host_to_first_kernel_gap=11.251341`, `host_end_to_last_kernel_tail=10.963 ms`, `gpu_makespan=0.029 ms`, `gpu_kernel_sum=0.028 ms`
  开始时间(ns): `1045005548`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.315 ms
  纯GPU kernel时间: `0.050 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.315 ms`, `host_to_first_kernel_gap=10.838555`, `host_end_to_last_kernel_tail=10.575 ms`, `gpu_makespan=0.051 ms`, `gpu_kernel_sum=0.050 ms`
  开始时间(ns): `1045458974`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 2 / decode / instance 1

- 执行序号: `8`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.837 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.837 ms`, `module_to_last_kernel=9.928 ms`, `host_to_first_kernel_gap=9.78091`, `host_end_to_last_kernel_tail=8.091 ms`, `gpu_makespan=0.147 ms`, `gpu_kernel_sum=0.115 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.251 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.251 ms`, `host_to_first_kernel_gap=9.714684`, `host_end_to_last_kernel_tail=9.480 ms`, `gpu_makespan=0.017 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `1046796157`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.050 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.050 ms`, `host_to_first_kernel_gap=9.408736`, `host_end_to_last_kernel_tail=9.361 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1047118585`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.034 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.034 ms`, `host_to_first_kernel_gap=9.328656`, `host_end_to_last_kernel_tail=9.297 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1047200841`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.224 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.224 ms`, `host_to_first_kernel_gap=9.258692`, `host_end_to_last_kernel_tail=9.055 ms`, `gpu_makespan=0.020 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `1047273717`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.084 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.084 ms`, `host_to_first_kernel_gap=8.90415`, `host_end_to_last_kernel_tail=8.822 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1047659491`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.312 ms
  纯GPU kernel时间: `0.027 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.312 ms`, `host_to_first_kernel_gap=8.772205`, `host_end_to_last_kernel_tail=8.489 ms`, `gpu_makespan=0.030 ms`, `gpu_kernel_sum=0.027 ms`
  开始时间(ns): `1047794348`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.300 ms
  纯GPU kernel时间: `0.050 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.300 ms`, `host_to_first_kernel_gap=8.360858`, `host_end_to_last_kernel_tail=8.111 ms`, `gpu_makespan=0.051 ms`, `gpu_kernel_sum=0.050 ms`
  开始时间(ns): `1048246719`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 3 / decode / instance 1

- 执行序号: `9`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `1.843 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.843 ms`, `module_to_last_kernel=7.475 ms`, `host_to_first_kernel_gap=7.329607`, `host_end_to_last_kernel_tail=5.632 ms`, `gpu_makespan=0.145 ms`, `gpu_kernel_sum=0.115 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.230 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.230 ms`, `host_to_first_kernel_gap=7.259753`, `host_end_to_last_kernel_tail=7.045 ms`, `gpu_makespan=0.015 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `1049559441`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.047 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.047 ms`, `host_to_first_kernel_gap=6.984111`, `host_end_to_last_kernel_tail=6.939 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1049850059`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.034 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.034 ms`, `host_to_first_kernel_gap=6.913853`, `host_end_to_last_kernel_tail=6.882 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1049922525`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.237 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.237 ms`, `host_to_first_kernel_gap=6.848604`, `host_end_to_last_kernel_tail=6.631 ms`, `gpu_makespan=0.019 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `1049991838`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.080 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.080 ms`, `host_to_first_kernel_gap=6.491434`, `host_end_to_last_kernel_tail=6.413 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1050378768`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.298 ms
  纯GPU kernel时间: `0.028 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.298 ms`, `host_to_first_kernel_gap=6.359058`, `host_end_to_last_kernel_tail=6.090 ms`, `gpu_makespan=0.029 ms`, `gpu_kernel_sum=0.028 ms`
  开始时间(ns): `1050513832`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.330 ms
  纯GPU kernel时间: `0.049 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.330 ms`, `host_to_first_kernel_gap=5.931377`, `host_end_to_last_kernel_tail=5.652 ms`, `gpu_makespan=0.050 ms`, `gpu_kernel_sum=0.049 ms`
  开始时间(ns): `1050982441`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 4 / decode / instance 1

- 执行序号: `10`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `1.892 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.892 ms`, `module_to_last_kernel=4.376 ms`, `host_to_first_kernel_gap=4.229922`, `host_end_to_last_kernel_tail=2.484 ms`, `gpu_makespan=0.146 ms`, `gpu_kernel_sum=0.118 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.299 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.299 ms`, `host_to_first_kernel_gap=4.157794`, `host_end_to_last_kernel_tail=3.874 ms`, `gpu_makespan=0.016 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `1053012824`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.059 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.059 ms`, `host_to_first_kernel_gap=3.805067`, `host_end_to_last_kernel_tail=3.749 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1053381231`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.034 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.034 ms`, `host_to_first_kernel_gap=3.720965`, `host_end_to_last_kernel_tail=3.689 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1053467509`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.241 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.241 ms`, `host_to_first_kernel_gap=3.649675`, `host_end_to_last_kernel_tail=3.427 ms`, `gpu_makespan=0.019 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `1053541903`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.086 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.086 ms`, `host_to_first_kernel_gap=3.264499`, `host_end_to_last_kernel_tail=3.181 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `1053956263`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.319 ms
  纯GPU kernel时间: `0.028 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.319 ms`, `host_to_first_kernel_gap=3.132667`, `host_end_to_last_kernel_tail=2.842 ms`, `gpu_makespan=0.029 ms`, `gpu_kernel_sum=0.028 ms`
  开始时间(ns): `1054091967`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.270 ms
  纯GPU kernel时间: `0.051 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.270 ms`, `host_to_first_kernel_gap=2.720387`, `host_end_to_last_kernel_tail=2.503 ms`, `gpu_makespan=0.052 ms`, `gpu_kernel_sum=0.051 ms`
  开始时间(ns): `1054543511`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`
