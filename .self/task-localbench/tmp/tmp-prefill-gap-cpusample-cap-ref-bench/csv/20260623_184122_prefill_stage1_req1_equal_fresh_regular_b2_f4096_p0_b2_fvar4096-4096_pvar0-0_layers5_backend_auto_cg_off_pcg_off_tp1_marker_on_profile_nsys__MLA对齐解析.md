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
- 整块 MLA-module 时长: `2.360 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.360 ms`, `module_to_last_kernel=5.445 ms`, `host_to_first_kernel_gap=0.330243`, `host_end_to_last_kernel_tail=3.085 ms`, `gpu_makespan=5.115 ms`, `gpu_kernel_sum=4.349 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=8192`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=0`, `sum_seq_after=8192`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 4096, 'prefix_len': 0, 'seq_len_after': 4096, 'prompt_len': 4096, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 4096, 'prefix_len': 0, 'seq_len_after': 4096, 'prompt_len': 4096, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.451 ms
  纯GPU kernel时间: `0.273 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.451 ms`, `host_to_first_kernel_gap=0.149413`, `host_end_to_last_kernel_tail=0.165 ms`, `gpu_makespan=0.467 ms`, `gpu_kernel_sum=0.273 ms`
  开始时间(ns): `29300878`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.057 ms
  纯GPU kernel时间: `0.017 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.057 ms`, `host_to_first_kernel_gap=0.101389`, `host_end_to_last_kernel_tail=0.061 ms`, `gpu_makespan=0.017 ms`, `gpu_kernel_sum=0.017 ms`
  开始时间(ns): `29815622`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.254 ms
  纯GPU kernel时间: `0.445 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.254 ms`, `host_to_first_kernel_gap=0.093942`, `host_end_to_last_kernel_tail=0.401 ms`, `gpu_makespan=0.561 ms`, `gpu_kernel_sum=0.445 ms`
  开始时间(ns): `29908733`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.044 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.044 ms`, `host_to_first_kernel_gap=0.330727`, `host_end_to_last_kernel_tail=0.299 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `30232845`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.124 ms
  纯GPU kernel时间: `0.108 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.124 ms`, `host_to_first_kernel_gap=0.249386`, `host_end_to_last_kernel_tail=0.233 ms`, `gpu_makespan=0.108 ms`, `gpu_kernel_sum=0.108 ms`
  开始时间(ns): `30327978`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.249 ms
  纯GPU kernel时间: `0.289 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.249 ms`, `host_to_first_kernel_gap=0.099015`, `host_end_to_last_kernel_tail=0.253 ms`, `gpu_makespan=0.403 ms`, `gpu_kernel_sum=0.289 ms`
  开始时间(ns): `30614189`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[8192, 512]]}`
- `attn_mha` -> 0.195 ms
  纯GPU kernel时间: `1.953 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.195 ms`, `host_to_first_kernel_gap=0.386332`, `host_end_to_last_kernel_tail=2.144 ms`, `gpu_makespan=1.953 ms`, `gpu_kernel_sum=1.953 ms`
  开始时间(ns): `30972472`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]}`
- `o_proj` -> 0.254 ms
  纯GPU kernel时间: `1.251 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.254 ms`, `host_to_first_kernel_gap=2.10342`, `host_end_to_last_kernel_tail=3.102 ms`, `gpu_makespan=1.253 ms`, `gpu_kernel_sum=1.251 ms`
  开始时间(ns): `31209241`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 1 / prefill / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `1.854 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.854 ms`, `module_to_last_kernel=11.393 ms`, `host_to_first_kernel_gap=6.777337`, `host_end_to_last_kernel_tail=9.539 ms`, `gpu_makespan=4.615 ms`, `gpu_kernel_sum=4.334 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=8192`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=0`, `sum_seq_after=8192`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 4096, 'prefix_len': 0, 'seq_len_after': 4096, 'prompt_len': 4096, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 4096, 'prefix_len': 0, 'seq_len_after': 4096, 'prompt_len': 4096, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.239 ms
  纯GPU kernel时间: `0.267 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.239 ms`, `host_to_first_kernel_gap=6.653664`, `host_end_to_last_kernel_tail=6.684 ms`, `gpu_makespan=0.269 ms`, `gpu_kernel_sum=0.267 ms`
  开始时间(ns): `32521785`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.047 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.047 ms`, `host_to_first_kernel_gap=6.633151`, `host_end_to_last_kernel_tail=6.604 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `32811962`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.208 ms
  纯GPU kernel时间: `0.441 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.208 ms`, `host_to_first_kernel_gap=6.574105`, `host_end_to_last_kernel_tail=6.809 ms`, `gpu_makespan=0.443 ms`, `gpu_kernel_sum=0.441 ms`
  开始时间(ns): `32890080`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.042 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.042 ms`, `host_to_first_kernel_gap=6.745803`, `host_end_to_last_kernel_tail=6.718 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `33161230`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.079 ms
  纯GPU kernel时间: `0.108 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.079 ms`, `host_to_first_kernel_gap=6.678741`, `host_end_to_last_kernel_tail=6.707 ms`, `gpu_makespan=0.108 ms`, `gpu_kernel_sum=0.108 ms`
  开始时间(ns): `33242756`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.222 ms
  纯GPU kernel时间: `0.286 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.222 ms`, `host_to_first_kernel_gap=6.592062`, `host_end_to_last_kernel_tail=6.658 ms`, `gpu_makespan=0.288 ms`, `gpu_kernel_sum=0.286 ms`
  开始时间(ns): `33465979`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[8192, 512]]}`
- `attn_mha` -> 0.156 ms
  纯GPU kernel时间: `1.953 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.156 ms`, `host_to_first_kernel_gap=6.797615`, `host_end_to_last_kernel_tail=8.594 ms`, `gpu_makespan=1.953 ms`, `gpu_kernel_sum=1.953 ms`
  开始时间(ns): `33789323`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]}`
- `o_proj` -> 0.249 ms
  纯GPU kernel时间: `1.248 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.249 ms`, `host_to_first_kernel_gap=8.553154`, `host_end_to_last_kernel_tail=9.553 ms`, `gpu_makespan=1.249 ms`, `gpu_kernel_sum=1.248 ms`
  开始时间(ns): `33988601`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 2 / prefill / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.831 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.831 ms`, `module_to_last_kernel=17.915 ms`, `host_to_first_kernel_gap=13.306663`, `host_end_to_last_kernel_tail=16.084 ms`, `gpu_makespan=4.608 ms`, `gpu_kernel_sum=4.328 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=8192`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=0`, `sum_seq_after=8192`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 4096, 'prefix_len': 0, 'seq_len_after': 4096, 'prompt_len': 4096, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 4096, 'prefix_len': 0, 'seq_len_after': 4096, 'prompt_len': 4096, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.239 ms
  纯GPU kernel时间: `0.268 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.239 ms`, `host_to_first_kernel_gap=13.193882`, `host_end_to_last_kernel_tail=13.224 ms`, `gpu_makespan=0.269 ms`, `gpu_kernel_sum=0.268 ms`
  开始时间(ns): `35211556`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.046 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.046 ms`, `host_to_first_kernel_gap=13.175648`, `host_end_to_last_kernel_tail=13.148 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `35499070`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.219 ms
  纯GPU kernel时间: `0.441 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.219 ms`, `host_to_first_kernel_gap=13.119509`, `host_end_to_last_kernel_tail=13.343 ms`, `gpu_makespan=0.442 ms`, `gpu_kernel_sum=0.441 ms`
  开始时间(ns): `35575081`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.040 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.040 ms`, `host_to_first_kernel_gap=13.283913`, `host_end_to_last_kernel_tail=13.257 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `35852374`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.078 ms
  纯GPU kernel时间: `0.108 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.078 ms`, `host_to_first_kernel_gap=13.215974`, `host_end_to_last_kernel_tail=13.246 ms`, `gpu_makespan=0.108 ms`, `gpu_kernel_sum=0.108 ms`
  开始时间(ns): `35934745`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.253 ms
  纯GPU kernel时间: `0.281 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.253 ms`, `host_to_first_kernel_gap=13.135374`, `host_end_to_last_kernel_tail=13.165 ms`, `gpu_makespan=0.282 ms`, `gpu_kernel_sum=0.281 ms`
  开始时间(ns): `36151121`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[8192, 512]]}`
- `attn_mha` -> 0.140 ms
  纯GPU kernel时间: `1.959 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.140 ms`, `host_to_first_kernel_gap=13.30667`, `host_end_to_last_kernel_tail=15.126 ms`, `gpu_makespan=1.959 ms`, `gpu_kernel_sum=1.959 ms`
  开始时间(ns): `36505041`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]}`
- `o_proj` -> 0.230 ms
  纯GPU kernel时间: `1.241 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.230 ms`, `host_to_first_kernel_gap=15.085818`, `host_end_to_last_kernel_tail=16.097 ms`, `gpu_makespan=1.242 ms`, `gpu_kernel_sum=1.241 ms`
  开始时间(ns): `36685606`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 3 / prefill / instance 1

- 执行序号: `4`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `1.848 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.848 ms`, `module_to_last_kernel=24.475 ms`, `host_to_first_kernel_gap=19.863444`, `host_end_to_last_kernel_tail=22.627 ms`, `gpu_makespan=4.611 ms`, `gpu_kernel_sum=4.329 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=8192`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=0`, `sum_seq_after=8192`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 4096, 'prefix_len': 0, 'seq_len_after': 4096, 'prompt_len': 4096, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 4096, 'prefix_len': 0, 'seq_len_after': 4096, 'prompt_len': 4096, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.228 ms
  纯GPU kernel时间: `0.267 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.228 ms`, `host_to_first_kernel_gap=19.740654`, `host_end_to_last_kernel_tail=19.782 ms`, `gpu_makespan=0.269 ms`, `gpu_kernel_sum=0.267 ms`
  开始时间(ns): `37881366`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.046 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.046 ms`, `host_to_first_kernel_gap=19.731245`, `host_end_to_last_kernel_tail=19.703 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `38159927`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.209 ms
  纯GPU kernel时间: `0.448 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.209 ms`, `host_to_first_kernel_gap=19.675519`, `host_end_to_last_kernel_tail=19.916 ms`, `gpu_makespan=0.450 ms`, `gpu_kernel_sum=0.448 ms`
  开始时间(ns): `38234917`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.040 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.040 ms`, `host_to_first_kernel_gap=19.858403`, `host_end_to_last_kernel_tail=19.831 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `38501793`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.081 ms
  纯GPU kernel时间: `0.107 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.081 ms`, `host_to_first_kernel_gap=19.793392`, `host_end_to_last_kernel_tail=19.820 ms`, `gpu_makespan=0.107 ms`, `gpu_kernel_sum=0.107 ms`
  开始时间(ns): `38581172`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.247 ms
  纯GPU kernel时间: `0.279 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.247 ms`, `host_to_first_kernel_gap=19.709589`, `host_end_to_last_kernel_tail=19.743 ms`, `gpu_makespan=0.281 ms`, `gpu_kernel_sum=0.279 ms`
  开始时间(ns): `38800047`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[8192, 512]]}`
- `attn_mha` -> 0.144 ms
  纯GPU kernel时间: `1.956 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.144 ms`, `host_to_first_kernel_gap=19.887876`, `host_end_to_last_kernel_tail=21.700 ms`, `gpu_makespan=1.956 ms`, `gpu_kernel_sum=1.956 ms`
  开始时间(ns): `39145249`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]}`
- `o_proj` -> 0.261 ms
  纯GPU kernel时间: `1.241 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.261 ms`, `host_to_first_kernel_gap=21.659996`, `host_end_to_last_kernel_tail=22.641 ms`, `gpu_makespan=1.242 ms`, `gpu_kernel_sum=1.241 ms`
  开始时间(ns): `39331082`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 4 / prefill / instance 1

- 执行序号: `5`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `1.940 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.940 ms`, `module_to_last_kernel=36.090 ms`, `host_to_first_kernel_gap=31.478636`, `host_end_to_last_kernel_tail=34.150 ms`, `gpu_makespan=4.612 ms`, `gpu_kernel_sum=4.329 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=8192`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=0`, `sum_seq_after=8192`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 4096, 'prefix_len': 0, 'seq_len_after': 4096, 'prompt_len': 4096, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 4096, 'prefix_len': 0, 'seq_len_after': 4096, 'prompt_len': 4096, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.310 ms
  纯GPU kernel时间: `0.266 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.310 ms`, `host_to_first_kernel_gap=31.34945`, `host_end_to_last_kernel_tail=31.308 ms`, `gpu_makespan=0.268 ms`, `gpu_kernel_sum=0.266 ms`
  开始时间(ns): `41301411`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.050 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.050 ms`, `host_to_first_kernel_gap=31.256901`, `host_end_to_last_kernel_tail=31.224 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `41662984`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.229 ms
  纯GPU kernel时间: `0.442 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.229 ms`, `host_to_first_kernel_gap=31.193028`, `host_end_to_last_kernel_tail=31.407 ms`, `gpu_makespan=0.443 ms`, `gpu_kernel_sum=0.442 ms`
  开始时间(ns): `41745577`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.046 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.046 ms`, `host_to_first_kernel_gap=31.346613`, `host_end_to_last_kernel_tail=31.314 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `42034776`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.081 ms
  纯GPU kernel时间: `0.107 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.081 ms`, `host_to_first_kernel_gap=31.275927`, `host_end_to_last_kernel_tail=31.302 ms`, `gpu_makespan=0.107 ms`, `gpu_kernel_sum=0.107 ms`
  开始时间(ns): `42119414`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.243 ms
  纯GPU kernel时间: `0.288 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.243 ms`, `host_to_first_kernel_gap=31.193784`, `host_end_to_last_kernel_tail=31.241 ms`, `gpu_makespan=0.290 ms`, `gpu_kernel_sum=0.288 ms`
  开始时间(ns): `42337429`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[8192, 512]]}`
- `attn_mha` -> 0.145 ms
  纯GPU kernel时间: `1.954 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.145 ms`, `host_to_first_kernel_gap=31.387941`, `host_end_to_last_kernel_tail=33.197 ms`, `gpu_makespan=1.954 ms`, `gpu_kernel_sum=1.954 ms`
  开始时间(ns): `42676617`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]}`
- `o_proj` -> 0.237 ms
  纯GPU kernel时间: `1.241 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.237 ms`, `host_to_first_kernel_gap=33.157277`, `host_end_to_last_kernel_tail=34.163 ms`, `gpu_makespan=1.242 ms`, `gpu_kernel_sum=1.241 ms`
  开始时间(ns): `42862610`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 0 / decode / instance 1

- 执行序号: `6`
- Collector对齐边界: `{'Module': 'model.model.layers.0.self_attn'}`
- 整块 MLA-module 时长: `1.891 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.891 ms`, `module_to_last_kernel=41.459 ms`, `host_to_first_kernel_gap=41.309989`, `host_end_to_last_kernel_tail=39.567 ms`, `gpu_makespan=0.149 ms`, `gpu_kernel_sum=0.118 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.320 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.320 ms`, `host_to_first_kernel_gap=41.238098`, `host_end_to_last_kernel_tail=40.933 ms`, `gpu_makespan=0.015 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `47206340`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2, 7168]]}`
- `q_a_layernorm` -> 0.054 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.054 ms`, `host_to_first_kernel_gap=40.869451`, `host_end_to_last_kernel_tail=40.818 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `47589707`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2, 1536]]}`
- `kv_a_layernorm` -> 0.032 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.032 ms`, `host_to_first_kernel_gap=40.793872`, `host_end_to_last_kernel_tail=40.764 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `47667302`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2, 512]]}`
- `q_b_proj` -> 0.237 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.237 ms`, `host_to_first_kernel_gap=40.723029`, `host_end_to_last_kernel_tail=40.508 ms`, `gpu_makespan=0.021 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `47741025`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2, 1536]]}`
- `rotary_emb` -> 0.084 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.084 ms`, `host_to_first_kernel_gap=40.348682`, `host_end_to_last_kernel_tail=40.266 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `48148205`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2], [2, 128, 64], [2, 1, 64]]}`
- `attn_mqa` -> 0.320 ms
  纯GPU kernel时间: `0.028 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.320 ms`, `host_to_first_kernel_gap=40.217792`, `host_end_to_last_kernel_tail=39.927 ms`, `gpu_makespan=0.030 ms`, `gpu_kernel_sum=0.028 ms`
  开始时间(ns): `48281975`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2, 128, 512], [2, 1, 512], [2, 1, 512]]}`
- `o_proj` -> 0.273 ms
  纯GPU kernel时间: `0.050 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.273 ms`, `host_to_first_kernel_gap=39.808674`, `host_end_to_last_kernel_tail=39.588 ms`, `gpu_makespan=0.052 ms`, `gpu_kernel_sum=0.050 ms`
  开始时间(ns): `48732757`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2, 16384]]}`

## Layer 1 / decode / instance 1

- 执行序号: `7`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `1.677 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.677 ms`, `module_to_last_kernel=38.969 ms`, `host_to_first_kernel_gap=38.822566`, `host_end_to_last_kernel_tail=37.292 ms`, `gpu_makespan=0.147 ms`, `gpu_kernel_sum=0.118 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.229 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.229 ms`, `host_to_first_kernel_gap=38.756601`, `host_end_to_last_kernel_tail=38.544 ms`, `gpu_makespan=0.016 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `49994174`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2, 7168]]}`
- `q_a_layernorm` -> 0.048 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.048 ms`, `host_to_first_kernel_gap=38.485867`, `host_end_to_last_kernel_tail=38.440 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `50281068`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2, 1536]]}`
- `kv_a_layernorm` -> 0.042 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.042 ms`, `host_to_first_kernel_gap=38.415556`, `host_end_to_last_kernel_tail=38.376 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `50353683`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2, 512]]}`
- `q_b_proj` -> 0.213 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.213 ms`, `host_to_first_kernel_gap=38.341508`, `host_end_to_last_kernel_tail=38.148 ms`, `gpu_makespan=0.019 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `50430739`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2, 1536]]}`
- `rotary_emb` -> 0.078 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.078 ms`, `host_to_first_kernel_gap=38.013038`, `host_end_to_last_kernel_tail=37.937 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `50789897`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2], [2, 128, 64], [2, 1, 64]]}`
- `attn_mqa` -> 0.286 ms
  纯GPU kernel时间: `0.028 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.286 ms`, `host_to_first_kernel_gap=37.892432`, `host_end_to_last_kernel_tail=37.636 ms`, `gpu_makespan=0.030 ms`, `gpu_kernel_sum=0.028 ms`
  开始时间(ns): `50913479`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2, 128, 512], [2, 1, 512], [2, 1, 512]]}`
- `o_proj` -> 0.269 ms
  纯GPU kernel时间: `0.050 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.269 ms`, `host_to_first_kernel_gap=37.527227`, `host_end_to_last_kernel_tail=37.309 ms`, `gpu_makespan=0.051 ms`, `gpu_kernel_sum=0.050 ms`
  开始时间(ns): `51319644`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2, 16384]]}`

## Layer 2 / decode / instance 1

- 执行序号: `8`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.662 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.662 ms`, `module_to_last_kernel=36.733 ms`, `host_to_first_kernel_gap=36.58413`, `host_end_to_last_kernel_tail=35.071 ms`, `gpu_makespan=0.149 ms`, `gpu_kernel_sum=0.118 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.212 ms
  纯GPU kernel时间: `0.017 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.212 ms`, `host_to_first_kernel_gap=36.520743`, `host_end_to_last_kernel_tail=36.327 ms`, `gpu_makespan=0.019 ms`, `gpu_kernel_sum=0.017 ms`
  开始时间(ns): `52535504`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2, 7168]]}`
- `q_a_layernorm` -> 0.048 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.048 ms`, `host_to_first_kernel_gap=36.270495`, `host_end_to_last_kernel_tail=36.224 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `52804184`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2, 1536]]}`
- `kv_a_layernorm` -> 0.033 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.033 ms`, `host_to_first_kernel_gap=36.200154`, `host_end_to_last_kernel_tail=36.169 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `52876701`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2, 512]]}`
- `q_b_proj` -> 0.203 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.203 ms`, `host_to_first_kernel_gap=36.133417`, `host_end_to_last_kernel_tail=35.950 ms`, `gpu_makespan=0.019 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `52946254`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2, 1536]]}`
- `rotary_emb` -> 0.076 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.076 ms`, `host_to_first_kernel_gap=35.813872`, `host_end_to_last_kernel_tail=35.739 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `53295751`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2], [2, 128, 64], [2, 1, 64]]}`
- `attn_mqa` -> 0.291 ms
  纯GPU kernel时间: `0.028 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.291 ms`, `host_to_first_kernel_gap=35.689466`, `host_end_to_last_kernel_tail=35.428 ms`, `gpu_makespan=0.030 ms`, `gpu_kernel_sum=0.028 ms`
  开始时间(ns): `53423005`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2, 128, 512], [2, 1, 512], [2, 1, 512]]}`
- `o_proj` -> 0.273 ms
  纯GPU kernel时间: `0.049 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.273 ms`, `host_to_first_kernel_gap=35.312387`, `host_end_to_last_kernel_tail=35.090 ms`, `gpu_makespan=0.050 ms`, `gpu_kernel_sum=0.049 ms`
  开始时间(ns): `53841972`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2, 16384]]}`

## Layer 3 / decode / instance 1

- 执行序号: `9`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `1.651 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.651 ms`, `module_to_last_kernel=34.519 ms`, `host_to_first_kernel_gap=34.369817`, `host_end_to_last_kernel_tail=32.868 ms`, `gpu_makespan=0.150 ms`, `gpu_kernel_sum=0.119 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.215 ms
  纯GPU kernel时间: `0.017 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.215 ms`, `host_to_first_kernel_gap=34.309549`, `host_end_to_last_kernel_tail=34.112 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.017 ms`
  开始时间(ns): `55054826`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2, 7168]]}`
- `q_a_layernorm` -> 0.047 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.047 ms`, `host_to_first_kernel_gap=34.05166`, `host_end_to_last_kernel_tail=34.006 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `55330219`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2, 1536]]}`
- `kv_a_layernorm` -> 0.033 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.033 ms`, `host_to_first_kernel_gap=33.978602`, `host_end_to_last_kernel_tail=33.948 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `55405421`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2, 512]]}`
- `q_b_proj` -> 0.207 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.207 ms`, `host_to_first_kernel_gap=33.91617`, `host_end_to_last_kernel_tail=33.728 ms`, `gpu_makespan=0.019 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `55471725`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2, 1536]]}`
- `rotary_emb` -> 0.080 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.080 ms`, `host_to_first_kernel_gap=33.602103`, `host_end_to_last_kernel_tail=33.524 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `55816192`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2], [2, 128, 64], [2, 1, 64]]}`
- `attn_mqa` -> 0.282 ms
  纯GPU kernel时间: `0.028 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.282 ms`, `host_to_first_kernel_gap=33.482774`, `host_end_to_last_kernel_tail=33.230 ms`, `gpu_makespan=0.029 ms`, `gpu_kernel_sum=0.028 ms`
  开始时间(ns): `55938337`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2, 128, 512], [2, 1, 512], [2, 1, 512]]}`
- `o_proj` -> 0.282 ms
  纯GPU kernel时间: `0.050 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.282 ms`, `host_to_first_kernel_gap=33.115814`, `host_end_to_last_kernel_tail=32.886 ms`, `gpu_makespan=0.051 ms`, `gpu_kernel_sum=0.050 ms`
  开始时间(ns): `56346673`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2, 16384]]}`

## Layer 4 / decode / instance 1

- 执行序号: `10`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `1.760 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.760 ms`, `module_to_last_kernel=31.774 ms`, `host_to_first_kernel_gap=31.623957`, `host_end_to_last_kernel_tail=30.013 ms`, `gpu_makespan=0.150 ms`, `gpu_kernel_sum=0.120 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.271 ms
  纯GPU kernel时间: `0.017 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.271 ms`, `host_to_first_kernel_gap=31.557745`, `host_end_to_last_kernel_tail=31.305 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.017 ms`
  开始时间(ns): `58160614`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2, 7168]]}`
- `q_a_layernorm` -> 0.051 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.051 ms`, `host_to_first_kernel_gap=31.241211`, `host_end_to_last_kernel_tail=31.192 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `58495036`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2, 1536]]}`
- `kv_a_layernorm` -> 0.032 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.032 ms`, `host_to_first_kernel_gap=31.16735`, `host_end_to_last_kernel_tail=31.138 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `58571041`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2, 512]]}`
- `q_b_proj` -> 0.236 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.236 ms`, `host_to_first_kernel_gap=31.102638`, `host_end_to_last_kernel_tail=30.887 ms`, `gpu_makespan=0.020 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `58638857`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2, 1536]]}`
- `rotary_emb` -> 0.083 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.083 ms`, `host_to_first_kernel_gap=30.740155`, `host_end_to_last_kernel_tail=30.659 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `59032572`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2], [2, 128, 64], [2, 1, 64]]}`
- `attn_mqa` -> 0.291 ms
  纯GPU kernel时间: `0.029 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.291 ms`, `host_to_first_kernel_gap=30.617052`, `host_end_to_last_kernel_tail=30.356 ms`, `gpu_makespan=0.030 ms`, `gpu_kernel_sum=0.029 ms`
  开始时间(ns): `59159611`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2, 128, 512], [2, 1, 512], [2, 1, 512]]}`
- `o_proj` -> 0.264 ms
  纯GPU kernel时间: `0.049 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.264 ms`, `host_to_first_kernel_gap=30.243819`, `host_end_to_last_kernel_tail=30.030 ms`, `gpu_makespan=0.050 ms`, `gpu_kernel_sum=0.049 ms`
  开始时间(ns): `59574124`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2, 16384]]}`
