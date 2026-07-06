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
- 整块 MLA-module 时长: `2.266 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.266 ms`, `module_to_last_kernel=2.817 ms`, `host_to_first_kernel_gap=0.329526`, `host_end_to_last_kernel_tail=0.551 ms`, `gpu_makespan=2.488 ms`, `gpu_kernel_sum=1.403 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=3600`, `total_tokens=3600`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[3600, 128, 192], [3600, 128, 192], [3600, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=3600`, `sum_prefix=0`, `sum_seq_after=3600`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 4, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 5, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 6, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 7, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 8, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 9, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 10, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 11, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.384 ms
  纯GPU kernel时间: `0.113 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.384 ms`, `host_to_first_kernel_gap=0.132124`, `host_end_to_last_kernel_tail=0.047 ms`, `gpu_makespan=0.299 ms`, `gpu_kernel_sum=0.113 ms`
  开始时间(ns): `29460436`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[3600, 7168]]}`
- `q_a_layernorm` -> 0.052 ms
  纯GPU kernel时间: `0.008 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.052 ms`, `host_to_first_kernel_gap=0.045181`, `host_end_to_last_kernel_tail=0.002 ms`, `gpu_makespan=0.008 ms`, `gpu_kernel_sum=0.008 ms`
  开始时间(ns): `29907059`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[3600, 1536]]}`
- `q_b_proj` -> 0.261 ms
  纯GPU kernel时间: `0.216 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.261 ms`, `host_to_first_kernel_gap=0.094601`, `host_end_to_last_kernel_tail=0.179 ms`, `gpu_makespan=0.346 ms`, `gpu_kernel_sum=0.216 ms`
  开始时间(ns): `29995687`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[3600, 1536]]}`
- `kv_a_layernorm` -> 0.046 ms
  纯GPU kernel时间: `0.008 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.046 ms`, `host_to_first_kernel_gap=0.106997`, `host_end_to_last_kernel_tail=0.069 ms`, `gpu_makespan=0.008 ms`, `gpu_kernel_sum=0.008 ms`
  开始时间(ns): `30329051`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[3600, 512]]}`
- `rotary_emb` -> 0.101 ms
  纯GPU kernel时间: `0.046 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.101 ms`, `host_to_first_kernel_gap=0.086665`, `host_end_to_last_kernel_tail=0.031 ms`, `gpu_makespan=0.046 ms`, `gpu_kernel_sum=0.046 ms`
  开始时间(ns): `30420999`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[3600], [3600, 128, 64], [3600, 1, 64]]}`
- `kv_b_proj` -> 0.256 ms
  纯GPU kernel时间: `0.134 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.256 ms`, `host_to_first_kernel_gap=0.096212`, `host_end_to_last_kernel_tail=0.100 ms`, `gpu_makespan=0.259 ms`, `gpu_kernel_sum=0.134 ms`
  开始时间(ns): `30674268`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[3600, 512]]}`
- `attn_mha` -> 0.182 ms
  纯GPU kernel时间: `0.259 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.182 ms`, `host_to_first_kernel_gap=0.137625`, `host_end_to_last_kernel_tail=0.235 ms`, `gpu_makespan=0.280 ms`, `gpu_kernel_sum=0.259 ms`
  开始时间(ns): `31041047`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[3600, 128, 192], [3600, 128, 192], [3600, 128, 128]]}`
- `o_proj` -> 0.247 ms
  纯GPU kernel时间: `0.619 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.247 ms`, `host_to_first_kernel_gap=0.192689`, `host_end_to_last_kernel_tail=0.567 ms`, `gpu_makespan=0.620 ms`, `gpu_kernel_sum=0.619 ms`
  开始时间(ns): `31267263`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[3600, 16384]]}`

## Layer 1 / prefill / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `1.839 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.839 ms`, `module_to_last_kernel=3.490 ms`, `host_to_first_kernel_gap=1.92576`, `host_end_to_last_kernel_tail=1.651 ms`, `gpu_makespan=1.564 ms`, `gpu_kernel_sum=1.431 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=3600`, `total_tokens=3600`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[3600, 128, 192], [3600, 128, 192], [3600, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=3600`, `sum_prefix=0`, `sum_seq_after=3600`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 4, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 5, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 6, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 7, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 8, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 9, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 10, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 11, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.222 ms
  纯GPU kernel时间: `0.112 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.222 ms`, `host_to_first_kernel_gap=1.771765`, `host_end_to_last_kernel_tail=1.663 ms`, `gpu_makespan=0.114 ms`, `gpu_kernel_sum=0.112 ms`
  开始时间(ns): `32532058`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[3600, 7168]]}`
- `q_a_layernorm` -> 0.046 ms
  纯GPU kernel时间: `0.008 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.046 ms`, `host_to_first_kernel_gap=1.614404`, `host_end_to_last_kernel_tail=1.576 ms`, `gpu_makespan=0.008 ms`, `gpu_kernel_sum=0.008 ms`
  开始时间(ns): `32803051`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[3600, 1536]]}`
- `q_b_proj` -> 0.201 ms
  纯GPU kernel时间: `0.209 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.201 ms`, `host_to_first_kernel_gap=1.547135`, `host_end_to_last_kernel_tail=1.558 ms`, `gpu_makespan=0.211 ms`, `gpu_kernel_sum=0.209 ms`
  开始时间(ns): `32879632`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[3600, 1536]]}`
- `kv_a_layernorm` -> 0.050 ms
  纯GPU kernel时间: `0.008 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.050 ms`, `host_to_first_kernel_gap=1.499339`, `host_end_to_last_kernel_tail=1.457 ms`, `gpu_makespan=0.008 ms`, `gpu_kernel_sum=0.008 ms`
  开始时间(ns): `33138724`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[3600, 512]]}`
- `rotary_emb` -> 0.077 ms
  纯GPU kernel时间: `0.047 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.077 ms`, `host_to_first_kernel_gap=1.419443`, `host_end_to_last_kernel_tail=1.389 ms`, `gpu_makespan=0.047 ms`, `gpu_kernel_sum=0.047 ms`
  开始时间(ns): `33227836`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[3600], [3600, 128, 64], [3600, 1, 64]]}`
- `kv_b_proj` -> 0.237 ms
  纯GPU kernel时间: `0.131 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.237 ms`, `host_to_first_kernel_gap=1.267392`, `host_end_to_last_kernel_tail=1.163 ms`, `gpu_makespan=0.133 ms`, `gpu_kernel_sum=0.131 ms`
  开始时间(ns): `33441711`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[3600, 512]]}`
- `attn_mha` -> 0.145 ms
  纯GPU kernel时间: `0.257 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.145 ms`, `host_to_first_kernel_gap=1.173127`, `host_end_to_last_kernel_tail=1.285 ms`, `gpu_makespan=0.257 ms`, `gpu_kernel_sum=0.257 ms`
  开始时间(ns): `33776168`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[3600, 128, 192], [3600, 128, 192], [3600, 128, 128]]}`
- `o_proj` -> 0.238 ms
  纯GPU kernel时间: `0.659 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.238 ms`, `host_to_first_kernel_gap=1.243044`, `host_end_to_last_kernel_tail=1.665 ms`, `gpu_makespan=0.660 ms`, `gpu_kernel_sum=0.659 ms`
  开始时间(ns): `33965003`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[3600, 16384]]}`

## Layer 2 / prefill / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.833 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.833 ms`, `module_to_last_kernel=4.631 ms`, `host_to_first_kernel_gap=3.095058`, `host_end_to_last_kernel_tail=2.798 ms`, `gpu_makespan=1.536 ms`, `gpu_kernel_sum=1.401 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=3600`, `total_tokens=3600`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[3600, 128, 192], [3600, 128, 192], [3600, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=3600`, `sum_prefix=0`, `sum_seq_after=3600`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 4, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 5, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 6, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 7, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 8, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 9, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 10, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 11, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.227 ms
  纯GPU kernel时间: `0.116 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.227 ms`, `host_to_first_kernel_gap=2.949551`, `host_end_to_last_kernel_tail=2.840 ms`, `gpu_makespan=0.117 ms`, `gpu_kernel_sum=0.116 ms`
  开始时间(ns): `35159039`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[3600, 7168]]}`
- `q_a_layernorm` -> 0.049 ms
  纯GPU kernel时间: `0.008 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.049 ms`, `host_to_first_kernel_gap=2.79141`, `host_end_to_last_kernel_tail=2.751 ms`, `gpu_makespan=0.008 ms`, `gpu_kernel_sum=0.008 ms`
  开始时间(ns): `35434460`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[3600, 1536]]}`
- `q_b_proj` -> 0.224 ms
  纯GPU kernel时间: `0.211 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.224 ms`, `host_to_first_kernel_gap=2.722876`, `host_end_to_last_kernel_tail=2.710 ms`, `gpu_makespan=0.212 ms`, `gpu_kernel_sum=0.211 ms`
  开始时间(ns): `35513298`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[3600, 1536]]}`
- `kv_a_layernorm` -> 0.049 ms
  纯GPU kernel时间: `0.007 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.049 ms`, `host_to_first_kernel_gap=2.649427`, `host_end_to_last_kernel_tail=2.607 ms`, `gpu_makespan=0.007 ms`, `gpu_kernel_sum=0.007 ms`
  开始时间(ns): `35798203`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[3600, 512]]}`
- `rotary_emb` -> 0.075 ms
  纯GPU kernel时间: `0.046 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.075 ms`, `host_to_first_kernel_gap=2.570871`, `host_end_to_last_kernel_tail=2.542 ms`, `gpu_makespan=0.046 ms`, `gpu_kernel_sum=0.046 ms`
  开始时间(ns): `35885943`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[3600], [3600, 128, 64], [3600, 1, 64]]}`
- `kv_b_proj` -> 0.233 ms
  纯GPU kernel时间: `0.137 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.233 ms`, `host_to_first_kernel_gap=2.423241`, `host_end_to_last_kernel_tail=2.328 ms`, `gpu_makespan=0.138 ms`, `gpu_kernel_sum=0.137 ms`
  开始时间(ns): `36094885`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[3600, 512]]}`
- `attn_mha` -> 0.142 ms
  纯GPU kernel时间: `0.257 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.142 ms`, `host_to_first_kernel_gap=2.341702`, `host_end_to_last_kernel_tail=2.456 ms`, `gpu_makespan=0.257 ms`, `gpu_kernel_sum=0.257 ms`
  开始时间(ns): `36423624`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[3600, 128, 192], [3600, 128, 192], [3600, 128, 128]]}`
- `o_proj` -> 0.227 ms
  纯GPU kernel时间: `0.618 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.227 ms`, `host_to_first_kernel_gap=2.419111`, `host_end_to_last_kernel_tail=2.813 ms`, `gpu_makespan=0.621 ms`, `gpu_kernel_sum=0.618 ms`
  开始时间(ns): `36604935`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[3600, 16384]]}`

## Layer 3 / prefill / instance 1

- 执行序号: `4`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `1.791 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.791 ms`, `module_to_last_kernel=5.773 ms`, `host_to_first_kernel_gap=4.247627`, `host_end_to_last_kernel_tail=3.982 ms`, `gpu_makespan=1.526 ms`, `gpu_kernel_sum=1.391 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=3600`, `total_tokens=3600`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[3600, 128, 192], [3600, 128, 192], [3600, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=3600`, `sum_prefix=0`, `sum_seq_after=3600`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 4, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 5, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 6, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 7, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 8, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 9, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 10, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 11, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.212 ms
  纯GPU kernel时间: `0.112 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.212 ms`, `host_to_first_kernel_gap=4.102383`, `host_end_to_last_kernel_tail=4.004 ms`, `gpu_makespan=0.114 ms`, `gpu_kernel_sum=0.112 ms`
  开始时间(ns): `37774238`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[3600, 7168]]}`
- `q_a_layernorm` -> 0.046 ms
  纯GPU kernel时间: `0.008 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.046 ms`, `host_to_first_kernel_gap=3.954213`, `host_end_to_last_kernel_tail=3.916 ms`, `gpu_makespan=0.008 ms`, `gpu_kernel_sum=0.008 ms`
  开始时间(ns): `38036872`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[3600, 1536]]}`
- `q_b_proj` -> 0.223 ms
  纯GPU kernel时间: `0.210 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.223 ms`, `host_to_first_kernel_gap=3.885315`, `host_end_to_last_kernel_tail=3.875 ms`, `gpu_makespan=0.212 ms`, `gpu_kernel_sum=0.210 ms`
  开始时间(ns): `38115274`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[3600, 1536]]}`
- `kv_a_layernorm` -> 0.040 ms
  纯GPU kernel时间: `0.007 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.040 ms`, `host_to_first_kernel_gap=3.81735`, `host_end_to_last_kernel_tail=3.785 ms`, `gpu_makespan=0.007 ms`, `gpu_kernel_sum=0.007 ms`
  开始时间(ns): `38395463`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[3600, 512]]}`
- `rotary_emb` -> 0.077 ms
  纯GPU kernel时间: `0.046 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.077 ms`, `host_to_first_kernel_gap=3.747204`, `host_end_to_last_kernel_tail=3.717 ms`, `gpu_makespan=0.046 ms`, `gpu_kernel_sum=0.046 ms`
  开始时间(ns): `38474121`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[3600], [3600, 128, 64], [3600, 1, 64]]}`
- `kv_b_proj` -> 0.218 ms
  纯GPU kernel时间: `0.133 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.218 ms`, `host_to_first_kernel_gap=3.598412`, `host_end_to_last_kernel_tail=3.516 ms`, `gpu_makespan=0.136 ms`, `gpu_kernel_sum=0.133 ms`
  开始时间(ns): `38684385`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[3600, 512]]}`
- `attn_mha` -> 0.138 ms
  纯GPU kernel时间: `0.256 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.138 ms`, `host_to_first_kernel_gap=3.52464`, `host_end_to_last_kernel_tail=3.642 ms`, `gpu_makespan=0.256 ms`, `gpu_kernel_sum=0.256 ms`
  开始时间(ns): `39001101`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[3600, 128, 192], [3600, 128, 192], [3600, 128, 128]]}`
- `o_proj` -> 0.222 ms
  纯GPU kernel时间: `0.618 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.222 ms`, `host_to_first_kernel_gap=3.599029`, `host_end_to_last_kernel_tail=3.996 ms`, `gpu_makespan=0.619 ms`, `gpu_kernel_sum=0.618 ms`
  开始时间(ns): `39184472`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[3600, 16384]]}`

## Layer 4 / prefill / instance 1

- 执行序号: `5`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `1.913 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.913 ms`, `module_to_last_kernel=8.754 ms`, `host_to_first_kernel_gap=7.20087`, `host_end_to_last_kernel_tail=6.841 ms`, `gpu_makespan=1.553 ms`, `gpu_kernel_sum=1.419 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=3600`, `total_tokens=3600`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[3600, 128, 192], [3600, 128, 192], [3600, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=3600`, `sum_prefix=0`, `sum_seq_after=3600`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 4, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 5, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 6, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 7, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 8, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 9, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 10, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}, {'req_id': 11, 'extend_len': 300, 'prefix_len': 0, 'seq_len_after': 300, 'prompt_len': 300, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.286 ms
  纯GPU kernel时间: `0.114 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.286 ms`, `host_to_first_kernel_gap=7.040487`, `host_end_to_last_kernel_tail=6.871 ms`, `gpu_makespan=0.117 ms`, `gpu_kernel_sum=0.114 ms`
  开始时间(ns): `41036004`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[3600, 7168]]}`
- `q_a_layernorm` -> 0.048 ms
  纯GPU kernel时间: `0.008 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.048 ms`, `host_to_first_kernel_gap=6.818159`, `host_end_to_last_kernel_tail=6.778 ms`, `gpu_makespan=0.008 ms`, `gpu_kernel_sum=0.008 ms`
  开始时间(ns): `41375644`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[3600, 1536]]}`
- `q_b_proj` -> 0.221 ms
  纯GPU kernel时间: `0.211 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.221 ms`, `host_to_first_kernel_gap=6.746657`, `host_end_to_last_kernel_tail=6.738 ms`, `gpu_makespan=0.212 ms`, `gpu_kernel_sum=0.211 ms`
  开始时间(ns): `41456298`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[3600, 1536]]}`
- `kv_a_layernorm` -> 0.046 ms
  纯GPU kernel时间: `0.007 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.046 ms`, `host_to_first_kernel_gap=6.675543`, `host_end_to_last_kernel_tail=6.637 ms`, `gpu_makespan=0.007 ms`, `gpu_kernel_sum=0.007 ms`
  开始时间(ns): `41739732`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[3600, 512]]}`
- `rotary_emb` -> 0.078 ms
  纯GPU kernel时间: `0.046 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.078 ms`, `host_to_first_kernel_gap=6.599993`, `host_end_to_last_kernel_tail=6.568 ms`, `gpu_makespan=0.046 ms`, `gpu_kernel_sum=0.046 ms`
  开始时间(ns): `41823634`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[3600], [3600, 128, 64], [3600, 1, 64]]}`
- `kv_b_proj` -> 0.230 ms
  纯GPU kernel时间: `0.133 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.230 ms`, `host_to_first_kernel_gap=6.454135`, `host_end_to_last_kernel_tail=6.359 ms`, `gpu_makespan=0.135 ms`, `gpu_kernel_sum=0.133 ms`
  开始时间(ns): `42030708`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[3600, 512]]}`
- `attn_mha` -> 0.142 ms
  纯GPU kernel时间: `0.256 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.142 ms`, `host_to_first_kernel_gap=6.371706`, `host_end_to_last_kernel_tail=6.486 ms`, `gpu_makespan=0.256 ms`, `gpu_kernel_sum=0.256 ms`
  开始时间(ns): `42355217`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[3600, 128, 192], [3600, 128, 192], [3600, 128, 128]]}`
- `o_proj` -> 0.235 ms
  纯GPU kernel时间: `0.643 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.235 ms`, `host_to_first_kernel_gap=6.447171`, `host_end_to_last_kernel_tail=6.856 ms`, `gpu_makespan=0.644 ms`, `gpu_kernel_sum=0.643 ms`
  开始时间(ns): `42537832`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[3600, 16384]]}`

## Layer 0 / decode / instance 1

- 执行序号: `6`
- Collector对齐边界: `{'Module': 'model.model.layers.0.self_attn'}`
- 整块 MLA-module 时长: `1.867 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.867 ms`, `module_to_last_kernel=8.599 ms`, `host_to_first_kernel_gap=8.44664`, `host_end_to_last_kernel_tail=6.731 ms`, `gpu_makespan=0.152 ms`, `gpu_kernel_sum=0.118 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.308 ms
  纯GPU kernel时间: `0.012 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.308 ms`, `host_to_first_kernel_gap=8.37629`, `host_end_to_last_kernel_tail=8.082 ms`, `gpu_makespan=0.014 ms`, `gpu_kernel_sum=0.012 ms`
  开始时间(ns): `46649863`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[12, 7168]]}`
- `q_a_layernorm` -> 0.046 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.046 ms`, `host_to_first_kernel_gap=8.01767`, `host_end_to_last_kernel_tail=7.974 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `47022211`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[12, 1536]]}`
- `kv_a_layernorm` -> 0.036 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.036 ms`, `host_to_first_kernel_gap=7.948352`, `host_end_to_last_kernel_tail=7.914 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `47093449`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[12, 512]]}`
- `q_b_proj` -> 0.242 ms
  纯GPU kernel时间: `0.021 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.242 ms`, `host_to_first_kernel_gap=7.870916`, `host_end_to_last_kernel_tail=7.651 ms`, `gpu_makespan=0.023 ms`, `gpu_kernel_sum=0.021 ms`
  开始时间(ns): `47173765`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[12, 1536]]}`
- `rotary_emb` -> 0.085 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.085 ms`, `host_to_first_kernel_gap=7.487813`, `host_end_to_last_kernel_tail=7.405 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `47591492`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[12], [12, 128, 64], [12, 1, 64]]}`
- `attn_mqa` -> 0.296 ms
  纯GPU kernel时间: `0.028 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.296 ms`, `host_to_first_kernel_gap=7.35863`, `host_end_to_last_kernel_tail=7.093 ms`, `gpu_makespan=0.030 ms`, `gpu_kernel_sum=0.028 ms`
  开始时间(ns): `47723555`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[12, 128, 512], [12, 1, 512], [12, 1, 512]]}`
- `o_proj` -> 0.278 ms
  纯GPU kernel时间: `0.052 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.278 ms`, `host_to_first_kernel_gap=6.975497`, `host_end_to_last_kernel_tail=6.750 ms`, `gpu_makespan=0.053 ms`, `gpu_kernel_sum=0.052 ms`
  开始时间(ns): `48150016`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[12, 16384]]}`

## Layer 1 / decode / instance 1

- 执行序号: `7`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `1.649 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.649 ms`, `module_to_last_kernel=6.179 ms`, `host_to_first_kernel_gap=6.032252`, `host_end_to_last_kernel_tail=4.530 ms`, `gpu_makespan=0.147 ms`, `gpu_kernel_sum=0.115 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.215 ms
  纯GPU kernel时间: `0.012 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.215 ms`, `host_to_first_kernel_gap=5.970994`, `host_end_to_last_kernel_tail=5.769 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.012 ms`
  开始时间(ns): `49369047`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[12, 7168]]}`
- `q_a_layernorm` -> 0.045 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.045 ms`, `host_to_first_kernel_gap=5.709177`, `host_end_to_last_kernel_tail=5.667 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `49643664`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[12, 1536]]}`
- `kv_a_layernorm` -> 0.031 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.031 ms`, `host_to_first_kernel_gap=5.643736`, `host_end_to_last_kernel_tail=5.615 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `49711089`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[12, 512]]}`
- `q_b_proj` -> 0.203 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.203 ms`, `host_to_first_kernel_gap=5.582663`, `host_end_to_last_kernel_tail=5.399 ms`, `gpu_makespan=0.020 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `49776354`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[12, 1536]]}`
- `rotary_emb` -> 0.084 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.084 ms`, `host_to_first_kernel_gap=5.261946`, `host_end_to_last_kernel_tail=5.180 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `50128367`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[12], [12, 128, 64], [12, 1, 64]]}`
- `attn_mqa` -> 0.281 ms
  纯GPU kernel时间: `0.028 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.281 ms`, `host_to_first_kernel_gap=5.136924`, `host_end_to_last_kernel_tail=4.885 ms`, `gpu_makespan=0.030 ms`, `gpu_kernel_sum=0.028 ms`
  开始时间(ns): `50256333`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[12, 128, 512], [12, 1, 512], [12, 1, 512]]}`
- `o_proj` -> 0.271 ms
  纯GPU kernel时间: `0.050 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.271 ms`, `host_to_first_kernel_gap=4.768916`, `host_end_to_last_kernel_tail=4.549 ms`, `gpu_makespan=0.051 ms`, `gpu_kernel_sum=0.050 ms`
  开始时间(ns): `50666421`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[12, 16384]]}`

## Layer 2 / decode / instance 1

- 执行序号: `8`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.616 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.616 ms`, `module_to_last_kernel=4.002 ms`, `host_to_first_kernel_gap=3.853399`, `host_end_to_last_kernel_tail=2.386 ms`, `gpu_makespan=0.149 ms`, `gpu_kernel_sum=0.115 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.210 ms
  纯GPU kernel时间: `0.012 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.210 ms`, `host_to_first_kernel_gap=3.789573`, `host_end_to_last_kernel_tail=3.594 ms`, `gpu_makespan=0.014 ms`, `gpu_kernel_sum=0.012 ms`
  开始时间(ns): `51857348`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[12, 7168]]}`
- `q_a_layernorm` -> 0.052 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.052 ms`, `host_to_first_kernel_gap=3.534132`, `host_end_to_last_kernel_tail=3.485 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `52126933`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[12, 1536]]}`
- `kv_a_layernorm` -> 0.032 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.032 ms`, `host_to_first_kernel_gap=3.461423`, `host_end_to_last_kernel_tail=3.432 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `52201850`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[12, 512]]}`
- `q_b_proj` -> 0.202 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.202 ms`, `host_to_first_kernel_gap=3.399446`, `host_end_to_last_kernel_tail=3.218 ms`, `gpu_makespan=0.020 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `52266611`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[12, 1536]]}`
- `rotary_emb` -> 0.076 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.076 ms`, `host_to_first_kernel_gap=3.08798`, `host_end_to_last_kernel_tail=3.014 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `52611229`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[12], [12, 128, 64], [12, 1, 64]]}`
- `attn_mqa` -> 0.271 ms
  纯GPU kernel时间: `0.028 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.271 ms`, `host_to_first_kernel_gap=2.967276`, `host_end_to_last_kernel_tail=2.726 ms`, `gpu_makespan=0.030 ms`, `gpu_kernel_sum=0.028 ms`
  开始时间(ns): `52734909`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[12, 128, 512], [12, 1, 512], [12, 1, 512]]}`
- `o_proj` -> 0.267 ms
  纯GPU kernel时间: `0.050 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.267 ms`, `host_to_first_kernel_gap=2.61995`, `host_end_to_last_kernel_tail=2.405 ms`, `gpu_makespan=0.052 ms`, `gpu_kernel_sum=0.050 ms`
  开始时间(ns): `53124379`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[12, 16384]]}`

## Layer 3 / decode / instance 1

- 执行序号: `9`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `1.573 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.573 ms`, `module_to_last_kernel=1.883 ms`, `host_to_first_kernel_gap=1.733418`, `host_end_to_last_kernel_tail=0.309 ms`, `gpu_makespan=0.149 ms`, `gpu_kernel_sum=0.118 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.211 ms
  纯GPU kernel时间: `0.012 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.211 ms`, `host_to_first_kernel_gap=1.675515`, `host_end_to_last_kernel_tail=1.478 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.012 ms`
  开始时间(ns): `54281486`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[12, 7168]]}`
- `q_a_layernorm` -> 0.045 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.045 ms`, `host_to_first_kernel_gap=1.418657`, `host_end_to_last_kernel_tail=1.376 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `54551176`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[12, 1536]]}`
- `kv_a_layernorm` -> 0.031 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.031 ms`, `host_to_first_kernel_gap=1.351983`, `host_end_to_last_kernel_tail=1.323 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `54619898`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[12, 512]]}`
- `q_b_proj` -> 0.197 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.197 ms`, `host_to_first_kernel_gap=1.291513`, `host_end_to_last_kernel_tail=1.115 ms`, `gpu_makespan=0.020 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `54684336`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[12, 1536]]}`
- `rotary_emb` -> 0.077 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.077 ms`, `host_to_first_kernel_gap=0.985447`, `host_end_to_last_kernel_tail=0.910 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `55022338`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[12], [12, 128, 64], [12, 1, 64]]}`
- `attn_mqa` -> 0.284 ms
  纯GPU kernel时间: `0.028 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.284 ms`, `host_to_first_kernel_gap=0.868363`, `host_end_to_last_kernel_tail=0.613 ms`, `gpu_makespan=0.029 ms`, `gpu_kernel_sum=0.028 ms`
  开始时间(ns): `55142174`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[12, 128, 512], [12, 1, 512], [12, 1, 512]]}`
- `o_proj` -> 0.237 ms
  纯GPU kernel时间: `0.053 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.237 ms`, `host_to_first_kernel_gap=0.510561`, `host_end_to_last_kernel_tail=0.328 ms`, `gpu_makespan=0.054 ms`, `gpu_kernel_sum=0.053 ms`
  开始时间(ns): `55541768`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[12, 16384]]}`

## Layer 4 / decode / instance 1

- 执行序号: `10`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `1.728 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.728 ms`, `module_to_last_kernel=1.735 ms`, `host_to_first_kernel_gap=0.160007`, `host_end_to_last_kernel_tail=0.006 ms`, `gpu_makespan=1.575 ms`, `gpu_kernel_sum=0.118 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.265 ms
  纯GPU kernel时间: `0.012 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.265 ms`, `host_to_first_kernel_gap=0.095478`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.156 ms`, `gpu_kernel_sum=0.012 ms`
  开始时间(ns): `57271539`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[12, 7168]]}`
- `q_a_layernorm` -> 0.047 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.047 ms`, `host_to_first_kernel_gap=0.040862`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `57598571`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[12, 1536]]}`
- `kv_a_layernorm` -> 0.030 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.030 ms`, `host_to_first_kernel_gap=0.02673`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `57668831`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[12, 512]]}`
- `q_b_proj` -> 0.218 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.218 ms`, `host_to_first_kernel_gap=0.080407`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.133 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `57733330`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[12, 1536]]}`
- `rotary_emb` -> 0.089 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.089 ms`, `host_to_first_kernel_gap=0.076568`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `58105393`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[12], [12, 128, 64], [12, 1, 64]]}`
- `attn_mqa` -> 0.285 ms
  纯GPU kernel时间: `0.028 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.285 ms`, `host_to_first_kernel_gap=0.092273`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.178 ms`, `gpu_kernel_sum=0.028 ms`
  开始时间(ns): `58239608`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[12, 128, 512], [12, 1, 512], [12, 1, 512]]}`
- `o_proj` -> 0.268 ms
  纯GPU kernel时间: `0.053 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.268 ms`, `host_to_first_kernel_gap=0.095104`, `host_end_to_last_kernel_tail=0.024 ms`, `gpu_makespan=0.197 ms`, `gpu_kernel_sum=0.053 ms`
  开始时间(ns): `58649800`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[12, 16384]]}`
