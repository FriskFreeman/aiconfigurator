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
- 整块 MLA-module 时长: `2.757 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.757 ms`, `module_to_last_kernel=9.607 ms`, `host_to_first_kernel_gap=0.307019`, `host_end_to_last_kernel_tail=6.850 ms`, `gpu_makespan=9.300 ms`, `gpu_kernel_sum=7.931 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=16384`, `chunked_req_prefix_len=8192`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [16384, 128, 192], [16384, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=8192`, `sum_seq_after=16384`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 4096, 'prefix_len': 4096, 'seq_len_after': 8192, 'prompt_len': 8192, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 4096, 'prefix_len': 4096, 'seq_len_after': 8192, 'prompt_len': 8192, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.418 ms
  纯GPU kernel时间: `0.275 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.418 ms`, `host_to_first_kernel_gap=0.152728`, `host_end_to_last_kernel_tail=0.165 ms`, `gpu_makespan=0.430 ms`, `gpu_kernel_sum=0.275 ms`
  开始时间(ns): `33280413`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.055 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.055 ms`, `host_to_first_kernel_gap=0.105393`, `host_end_to_last_kernel_tail=0.068 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `33758148`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.263 ms
  纯GPU kernel时间: `0.448 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.263 ms`, `host_to_first_kernel_gap=0.10028`, `host_end_to_last_kernel_tail=0.404 ms`, `gpu_makespan=0.567 ms`, `gpu_kernel_sum=0.448 ms`
  开始时间(ns): `33854557`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.050 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.050 ms`, `host_to_first_kernel_gap=0.338938`, `host_end_to_last_kernel_tail=0.302 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `34182362`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.102 ms
  纯GPU kernel时间: `0.106 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.102 ms`, `host_to_first_kernel_gap=0.258751`, `host_end_to_last_kernel_tail=0.263 ms`, `gpu_makespan=0.106 ms`, `gpu_kernel_sum=0.106 ms`
  开始时间(ns): `34277141`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.297 ms
  纯GPU kernel时间: `0.536 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.297 ms`, `host_to_first_kernel_gap=0.103715`, `host_end_to_last_kernel_tail=0.496 ms`, `gpu_makespan=0.690 ms`, `gpu_kernel_sum=0.536 ms`
  开始时间(ns): `34921553`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[16384, 512]]}`
- `attn_mha` -> 0.195 ms
  纯GPU kernel时间: `5.278 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.195 ms`, `host_to_first_kernel_gap=0.854562`, `host_end_to_last_kernel_tail=5.937 ms`, `gpu_makespan=5.278 ms`, `gpu_kernel_sum=5.278 ms`
  开始时间(ns): `35340050`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [16384, 128, 192], [16384, 128, 128]]}`
- `o_proj` -> 0.284 ms
  纯GPU kernel时间: `1.258 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.284 ms`, `host_to_first_kernel_gap=5.890474`, `host_end_to_last_kernel_tail=6.865 ms`, `gpu_makespan=1.259 ms`, `gpu_kernel_sum=1.258 ms`
  开始时间(ns): `35583528`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 1 / prefill / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `2.103 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.103 ms`, `module_to_last_kernel=19.100 ms`, `host_to_first_kernel_gap=10.56975`, `host_end_to_last_kernel_tail=16.997 ms`, `gpu_makespan=8.531 ms`, `gpu_kernel_sum=7.988 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=16384`, `chunked_req_prefix_len=8192`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [16384, 128, 192], [16384, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=8192`, `sum_seq_after=16384`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 4096, 'prefix_len': 4096, 'seq_len_after': 8192, 'prompt_len': 8192, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 4096, 'prefix_len': 4096, 'seq_len_after': 8192, 'prompt_len': 8192, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.255 ms
  纯GPU kernel时间: `0.269 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.255 ms`, `host_to_first_kernel_gap=10.430142`, `host_end_to_last_kernel_tail=10.447 ms`, `gpu_makespan=0.272 ms`, `gpu_kernel_sum=0.269 ms`
  开始时间(ns): `36929521`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.056 ms
  纯GPU kernel时间: `0.017 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.056 ms`, `host_to_first_kernel_gap=10.384857`, `host_end_to_last_kernel_tail=10.346 ms`, `gpu_makespan=0.017 ms`, `gpu_kernel_sum=0.017 ms`
  开始时间(ns): `37246902`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.238 ms
  纯GPU kernel时间: `0.447 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.238 ms`, `host_to_first_kernel_gap=10.316156`, `host_end_to_last_kernel_tail=10.529 ms`, `gpu_makespan=0.451 ms`, `gpu_kernel_sum=0.447 ms`
  开始时间(ns): `37334323`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.046 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.046 ms`, `host_to_first_kernel_gap=10.464709`, `host_end_to_last_kernel_tail=10.433 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `37636298`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.093 ms
  纯GPU kernel时间: `0.105 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.093 ms`, `host_to_first_kernel_gap=10.391809`, `host_end_to_last_kernel_tail=10.404 ms`, `gpu_makespan=0.105 ms`, `gpu_kernel_sum=0.105 ms`
  开始时间(ns): `37724206`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.236 ms
  纯GPU kernel时间: `0.590 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.236 ms`, `host_to_first_kernel_gap=10.204251`, `host_end_to_last_kernel_tail=10.559 ms`, `gpu_makespan=0.591 ms`, `gpu_kernel_sum=0.590 ms`
  开始时间(ns): `38065204`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[16384, 512]]}`
- `attn_mha` -> 0.169 ms
  纯GPU kernel时间: `5.288 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.169 ms`, `host_to_first_kernel_gap=10.915077`, `host_end_to_last_kernel_tail=16.034 ms`, `gpu_makespan=5.288 ms`, `gpu_kernel_sum=5.288 ms`
  开始时间(ns): `38424553`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [16384, 128, 192], [16384, 128, 128]]}`
- `o_proj` -> 0.242 ms
  纯GPU kernel时间: `1.258 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.242 ms`, `host_to_first_kernel_gap=15.992516`, `host_end_to_last_kernel_tail=17.011 ms`, `gpu_makespan=1.261 ms`, `gpu_kernel_sum=1.258 ms`
  开始时间(ns): `38636872`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 2 / prefill / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `2.079 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.079 ms`, `module_to_last_kernel=29.227 ms`, `host_to_first_kernel_gap=20.764511`, `host_end_to_last_kernel_tail=27.148 ms`, `gpu_makespan=8.462 ms`, `gpu_kernel_sum=7.921 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=16384`, `chunked_req_prefix_len=8192`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [16384, 128, 192], [16384, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=8192`, `sum_seq_after=16384`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 4096, 'prefix_len': 4096, 'seq_len_after': 8192, 'prompt_len': 8192, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 4096, 'prefix_len': 4096, 'seq_len_after': 8192, 'prompt_len': 8192, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.237 ms
  纯GPU kernel时间: `0.268 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.237 ms`, `host_to_first_kernel_gap=20.627518`, `host_end_to_last_kernel_tail=20.661 ms`, `gpu_makespan=0.271 ms`, `gpu_kernel_sum=0.268 ms`
  开始时间(ns): `39893292`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.053 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.053 ms`, `host_to_first_kernel_gap=20.609702`, `host_end_to_last_kernel_tail=20.574 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `40182500`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.234 ms
  纯GPU kernel时间: `0.442 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.234 ms`, `host_to_first_kernel_gap=20.544749`, `host_end_to_last_kernel_tail=20.754 ms`, `gpu_makespan=0.444 ms`, `gpu_kernel_sum=0.442 ms`
  开始时间(ns): `40266429`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.044 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.044 ms`, `host_to_first_kernel_gap=20.692877`, `host_end_to_last_kernel_tail=20.662 ms`, `gpu_makespan=0.014 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `40561853`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.096 ms
  纯GPU kernel时间: `0.106 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.096 ms`, `host_to_first_kernel_gap=20.622472`, `host_end_to_last_kernel_tail=20.632 ms`, `gpu_makespan=0.106 ms`, `gpu_kernel_sum=0.106 ms`
  开始时间(ns): `40648546`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.244 ms
  纯GPU kernel时间: `0.532 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.244 ms`, `host_to_first_kernel_gap=20.44018`, `host_end_to_last_kernel_tail=20.731 ms`, `gpu_makespan=0.535 ms`, `gpu_kernel_sum=0.532 ms`
  开始时间(ns): `40982998`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[16384, 512]]}`
- `attn_mha` -> 0.187 ms
  纯GPU kernel时间: `5.285 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.187 ms`, `host_to_first_kernel_gap=21.104093`, `host_end_to_last_kernel_tail=26.202 ms`, `gpu_makespan=5.285 ms`, `gpu_kernel_sum=5.285 ms`
  开始时间(ns): `41333068`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [16384, 128, 192], [16384, 128, 128]]}`
- `o_proj` -> 0.255 ms
  纯GPU kernel时间: `1.256 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.255 ms`, `host_to_first_kernel_gap=26.160862`, `host_end_to_last_kernel_tail=27.163 ms`, `gpu_makespan=1.258 ms`, `gpu_kernel_sum=1.256 ms`
  开始时间(ns): `41564457`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 3 / prefill / instance 1

- 执行序号: `4`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `2.098 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.098 ms`, `module_to_last_kernel=39.397 ms`, `host_to_first_kernel_gap=30.900807`, `host_end_to_last_kernel_tail=37.299 ms`, `gpu_makespan=8.496 ms`, `gpu_kernel_sum=7.955 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=16384`, `chunked_req_prefix_len=8192`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [16384, 128, 192], [16384, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=8192`, `sum_seq_after=16384`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 4096, 'prefix_len': 4096, 'seq_len_after': 8192, 'prompt_len': 8192, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 4096, 'prefix_len': 4096, 'seq_len_after': 8192, 'prompt_len': 8192, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.245 ms
  纯GPU kernel时间: `0.269 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.245 ms`, `host_to_first_kernel_gap=30.756177`, `host_end_to_last_kernel_tail=30.781 ms`, `gpu_makespan=0.271 ms`, `gpu_kernel_sum=0.269 ms`
  开始时间(ns): `42865268`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.054 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.054 ms`, `host_to_first_kernel_gap=30.727343`, `host_end_to_last_kernel_tail=30.691 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `43164598`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.234 ms
  纯GPU kernel时间: `0.448 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.234 ms`, `host_to_first_kernel_gap=30.661209`, `host_end_to_last_kernel_tail=30.877 ms`, `gpu_makespan=0.449 ms`, `gpu_kernel_sum=0.448 ms`
  开始时间(ns): `43250348`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.045 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.045 ms`, `host_to_first_kernel_gap=30.814039`, `host_end_to_last_kernel_tail=30.782 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `43546766`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.092 ms
  纯GPU kernel时间: `0.106 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.092 ms`, `host_to_first_kernel_gap=30.742274`, `host_end_to_last_kernel_tail=30.756 ms`, `gpu_makespan=0.106 ms`, `gpu_kernel_sum=0.106 ms`
  开始时间(ns): `43634371`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.239 ms
  纯GPU kernel时间: `0.530 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.239 ms`, `host_to_first_kernel_gap=30.5625`, `host_end_to_last_kernel_tail=30.856 ms`, `gpu_makespan=0.532 ms`, `gpu_kernel_sum=0.530 ms`
  开始时间(ns): `43966848`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[16384, 512]]}`
- `attn_mha` -> 0.185 ms
  纯GPU kernel时间: `5.295 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.185 ms`, `host_to_first_kernel_gap=31.232834`, `host_end_to_last_kernel_tail=36.343 ms`, `gpu_makespan=5.295 ms`, `gpu_kernel_sum=5.295 ms`
  开始时间(ns): `44308322`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [16384, 128, 192], [16384, 128, 128]]}`
- `o_proj` -> 0.264 ms
  纯GPU kernel时间: `1.276 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.264 ms`, `host_to_first_kernel_gap=36.300399`, `host_end_to_last_kernel_tail=37.315 ms`, `gpu_makespan=1.279 ms`, `gpu_kernel_sum=1.276 ms`
  开始时间(ns): `44538867`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 4 / prefill / instance 1

- 执行序号: `5`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `2.202 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.202 ms`, `module_to_last_kernel=55.064 ms`, `host_to_first_kernel_gap=46.364381`, `host_end_to_last_kernel_tail=52.862 ms`, `gpu_makespan=8.699 ms`, `gpu_kernel_sum=8.161 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=16384`, `chunked_req_prefix_len=8192`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [16384, 128, 192], [16384, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=8192`, `sum_seq_after=16384`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 4096, 'prefix_len': 4096, 'seq_len_after': 8192, 'prompt_len': 8192, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 4096, 'prefix_len': 4096, 'seq_len_after': 8192, 'prompt_len': 8192, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.312 ms
  纯GPU kernel时间: `0.287 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.312 ms`, `host_to_first_kernel_gap=46.206449`, `host_end_to_last_kernel_tail=46.183 ms`, `gpu_makespan=0.289 ms`, `gpu_kernel_sum=0.287 ms`
  开始时间(ns): `46529964`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.056 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.056 ms`, `host_to_first_kernel_gap=46.128068`, `host_end_to_last_kernel_tail=46.090 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `46898009`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.231 ms
  纯GPU kernel时间: `0.473 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.231 ms`, `host_to_first_kernel_gap=46.059386`, `host_end_to_last_kernel_tail=46.304 ms`, `gpu_makespan=0.475 ms`, `gpu_kernel_sum=0.473 ms`
  开始时间(ns): `46986371`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.046 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.046 ms`, `host_to_first_kernel_gap=46.237386`, `host_end_to_last_kernel_tail=46.205 ms`, `gpu_makespan=0.014 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `47283123`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.114 ms
  纯GPU kernel时间: `0.105 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.114 ms`, `host_to_first_kernel_gap=46.165262`, `host_end_to_last_kernel_tail=46.156 ms`, `gpu_makespan=0.105 ms`, `gpu_kernel_sum=0.105 ms`
  开始时间(ns): `47371247`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.255 ms
  纯GPU kernel时间: `0.596 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.255 ms`, `host_to_first_kernel_gap=45.959096`, `host_end_to_last_kernel_tail=46.303 ms`, `gpu_makespan=0.599 ms`, `gpu_kernel_sum=0.596 ms`
  开始时间(ns): `47730341`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[16384, 512]]}`
- `attn_mha` -> 0.163 ms
  纯GPU kernel时间: `5.306 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.163 ms`, `host_to_first_kernel_gap=46.673581`, `host_end_to_last_kernel_tail=51.816 ms`, `gpu_makespan=5.306 ms`, `gpu_kernel_sum=5.306 ms`
  开始时间(ns): `48091855`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [16384, 128, 192], [16384, 128, 128]]}`
- `o_proj` -> 0.261 ms
  纯GPU kernel时间: `1.362 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.261 ms`, `host_to_first_kernel_gap=51.774868`, `host_end_to_last_kernel_tail=52.876 ms`, `gpu_makespan=1.363 ms`, `gpu_kernel_sum=1.362 ms`
  开始时间(ns): `48298054`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 0 / decode / instance 1

- 执行序号: `6`
- Collector对齐边界: `{'Module': 'model.model.layers.0.self_attn'}`
- 整块 MLA-module 时长: `2.244 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.244 ms`, `module_to_last_kernel=59.812 ms`, `host_to_first_kernel_gap=59.651227`, `host_end_to_last_kernel_tail=57.567 ms`, `gpu_makespan=0.160 ms`, `gpu_kernel_sum=0.127 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.397 ms
  纯GPU kernel时间: `0.017 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.397 ms`, `host_to_first_kernel_gap=59.567165`, `host_end_to_last_kernel_tail=59.188 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.017 ms`
  开始时间(ns): `53226264`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2, 7168]]}`
- `q_a_layernorm` -> 0.054 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.054 ms`, `host_to_first_kernel_gap=59.111089`, `host_end_to_last_kernel_tail=59.059 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `53699876`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2, 1536]]}`
- `kv_a_layernorm` -> 0.042 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.042 ms`, `host_to_first_kernel_gap=59.033135`, `host_end_to_last_kernel_tail=58.993 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `53779846`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2, 512]]}`
- `q_b_proj` -> 0.264 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.264 ms`, `host_to_first_kernel_gap=58.952149`, `host_end_to_last_kernel_tail=58.710 ms`, `gpu_makespan=0.022 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `53863872`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2, 1536]]}`
- `rotary_emb` -> 0.122 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.122 ms`, `host_to_first_kernel_gap=58.516047`, `host_end_to_last_kernel_tail=58.396 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `54334950`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2], [2, 128, 64], [2, 1, 64]]}`
- `attn_mqa` -> 0.361 ms
  纯GPU kernel时间: `0.035 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.361 ms`, `host_to_first_kernel_gap=58.342605`, `host_end_to_last_kernel_tail=58.018 ms`, `gpu_makespan=0.036 ms`, `gpu_kernel_sum=0.035 ms`
  开始时间(ns): `54511304`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2, 128, 512], [2, 1, 512], [2, 1, 512]]}`
- `o_proj` -> 0.321 ms
  纯GPU kernel时间: `0.050 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.321 ms`, `host_to_first_kernel_gap=57.85672`, `host_end_to_last_kernel_tail=57.587 ms`, `gpu_makespan=0.051 ms`, `gpu_kernel_sum=0.050 ms`
  开始时间(ns): `55045605`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2, 16384]]}`

## Layer 1 / decode / instance 1

- 执行序号: `7`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `1.886 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.886 ms`, `module_to_last_kernel=56.951 ms`, `host_to_first_kernel_gap=56.792991`, `host_end_to_last_kernel_tail=55.064 ms`, `gpu_makespan=0.158 ms`, `gpu_kernel_sum=0.128 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.239 ms
  纯GPU kernel时间: `0.017 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.239 ms`, `host_to_first_kernel_gap=56.720849`, `host_end_to_last_kernel_tail=56.500 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.017 ms`
  开始时间(ns): `56396036`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2, 7168]]}`
- `q_a_layernorm` -> 0.054 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.054 ms`, `host_to_first_kernel_gap=56.434677`, `host_end_to_last_kernel_tail=56.383 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `56700640`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2, 1536]]}`
- `kv_a_layernorm` -> 0.034 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.034 ms`, `host_to_first_kernel_gap=56.357467`, `host_end_to_last_kernel_tail=56.325 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `56779930`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2, 512]]}`
- `q_b_proj` -> 0.224 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.224 ms`, `host_to_first_kernel_gap=56.289903`, `host_end_to_last_kernel_tail=56.086 ms`, `gpu_makespan=0.019 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `56850438`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2, 1536]]}`
- `rotary_emb` -> 0.100 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.100 ms`, `host_to_first_kernel_gap=55.917215`, `host_end_to_last_kernel_tail=55.819 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `57254518`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2], [2, 128, 64], [2, 1, 64]]}`
- `attn_mqa` -> 0.346 ms
  纯GPU kernel时间: `0.035 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.346 ms`, `host_to_first_kernel_gap=55.76747`, `host_end_to_last_kernel_tail=55.458 ms`, `gpu_makespan=0.036 ms`, `gpu_kernel_sum=0.035 ms`
  开始时间(ns): `57407143`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2, 128, 512], [2, 1, 512], [2, 1, 512]]}`
- `o_proj` -> 0.295 ms
  纯GPU kernel时间: `0.051 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.295 ms`, `host_to_first_kernel_gap=55.324917`, `host_end_to_last_kernel_tail=55.082 ms`, `gpu_makespan=0.052 ms`, `gpu_kernel_sum=0.051 ms`
  开始时间(ns): `57897408`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2, 16384]]}`

## Layer 2 / decode / instance 1

- 执行序号: `8`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.896 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.896 ms`, `module_to_last_kernel=54.483 ms`, `host_to_first_kernel_gap=54.321517`, `host_end_to_last_kernel_tail=52.587 ms`, `gpu_makespan=0.161 ms`, `gpu_kernel_sum=0.129 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.238 ms
  纯GPU kernel时间: `0.016 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.238 ms`, `host_to_first_kernel_gap=54.248593`, `host_end_to_last_kernel_tail=54.029 ms`, `gpu_makespan=0.019 ms`, `gpu_kernel_sum=0.016 ms`
  开始时间(ns): `59188100`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2, 7168]]}`
- `q_a_layernorm` -> 0.055 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.055 ms`, `host_to_first_kernel_gap=53.964383`, `host_end_to_last_kernel_tail=53.912 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `59491094`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2, 1536]]}`
- `kv_a_layernorm` -> 0.036 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.036 ms`, `host_to_first_kernel_gap=53.887141`, `host_end_to_last_kernel_tail=53.854 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `59570640`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2, 512]]}`
- `q_b_proj` -> 0.221 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.221 ms`, `host_to_first_kernel_gap=53.818759`, `host_end_to_last_kernel_tail=53.618 ms`, `gpu_makespan=0.020 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `59642190`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2, 1536]]}`
- `rotary_emb` -> 0.097 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.097 ms`, `host_to_first_kernel_gap=53.461263`, `host_end_to_last_kernel_tail=53.366 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `60031718`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2], [2, 128, 64], [2, 1, 64]]}`
- `attn_mqa` -> 0.349 ms
  纯GPU kernel时间: `0.036 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.349 ms`, `host_to_first_kernel_gap=53.317314`, `host_end_to_last_kernel_tail=53.007 ms`, `gpu_makespan=0.038 ms`, `gpu_kernel_sum=0.036 ms`
  开始时间(ns): `60178963`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2, 128, 512], [2, 1, 512], [2, 1, 512]]}`
- `o_proj` -> 0.314 ms
  纯GPU kernel时间: `0.051 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.314 ms`, `host_to_first_kernel_gap=52.867555`, `host_end_to_last_kernel_tail=52.606 ms`, `gpu_makespan=0.052 ms`, `gpu_kernel_sum=0.051 ms`
  开始时间(ns): `60678194`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2, 16384]]}`

## Layer 3 / decode / instance 1

- 执行序号: `9`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `1.876 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.876 ms`, `module_to_last_kernel=51.965 ms`, `host_to_first_kernel_gap=51.805418`, `host_end_to_last_kernel_tail=50.089 ms`, `gpu_makespan=0.160 ms`, `gpu_kernel_sum=0.129 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.239 ms
  纯GPU kernel时间: `0.017 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.239 ms`, `host_to_first_kernel_gap=51.732436`, `host_end_to_last_kernel_tail=51.511 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.017 ms`
  开始时间(ns): `62027969`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2, 7168]]}`
- `q_a_layernorm` -> 0.055 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.055 ms`, `host_to_first_kernel_gap=51.447482`, `host_end_to_last_kernel_tail=51.395 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `62331195`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2, 1536]]}`
- `kv_a_layernorm` -> 0.035 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.035 ms`, `host_to_first_kernel_gap=51.361648`, `host_end_to_last_kernel_tail=51.329 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `62419237`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2, 512]]}`
- `q_b_proj` -> 0.219 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.219 ms`, `host_to_first_kernel_gap=51.293771`, `host_end_to_last_kernel_tail=51.095 ms`, `gpu_makespan=0.020 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `62491658`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2, 1536]]}`
- `rotary_emb` -> 0.103 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.103 ms`, `host_to_first_kernel_gap=50.931671`, `host_end_to_last_kernel_tail=50.830 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `62884478`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2], [2, 128, 64], [2, 1, 64]]}`
- `attn_mqa` -> 0.333 ms
  纯GPU kernel时间: `0.036 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.333 ms`, `host_to_first_kernel_gap=50.782218`, `host_end_to_last_kernel_tail=50.486 ms`, `gpu_makespan=0.037 ms`, `gpu_kernel_sum=0.036 ms`
  开始时间(ns): `63036747`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2, 128, 512], [2, 1, 512], [2, 1, 512]]}`
- `o_proj` -> 0.289 ms
  纯GPU kernel时间: `0.051 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.289 ms`, `host_to_first_kernel_gap=50.34271`, `host_end_to_last_kernel_tail=50.106 ms`, `gpu_makespan=0.052 ms`, `gpu_kernel_sum=0.051 ms`
  开始时间(ns): `63525151`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2, 16384]]}`

## Layer 4 / decode / instance 1

- 执行序号: `10`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `2.104 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.104 ms`, `module_to_last_kernel=48.783 ms`, `host_to_first_kernel_gap=48.625304`, `host_end_to_last_kernel_tail=46.679 ms`, `gpu_makespan=0.158 ms`, `gpu_kernel_sum=0.127 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.325 ms
  纯GPU kernel时间: `0.016 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.325 ms`, `host_to_first_kernel_gap=48.543818`, `host_end_to_last_kernel_tail=48.236 ms`, `gpu_makespan=0.017 ms`, `gpu_kernel_sum=0.016 ms`
  开始时间(ns): `65585675`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2, 7168]]}`
- `q_a_layernorm` -> 0.059 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.059 ms`, `host_to_first_kernel_gap=48.17238`, `host_end_to_last_kernel_tail=48.116 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `65974457`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2, 1536]]}`
- `kv_a_layernorm` -> 0.034 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.034 ms`, `host_to_first_kernel_gap=48.08978`, `host_end_to_last_kernel_tail=48.058 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `66059137`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2, 512]]}`
- `q_b_proj` -> 0.241 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.241 ms`, `host_to_first_kernel_gap=48.021782`, `host_end_to_last_kernel_tail=47.801 ms`, `gpu_makespan=0.020 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `66130431`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2, 1536]]}`
- `rotary_emb` -> 0.111 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.111 ms`, `host_to_first_kernel_gap=47.600828`, `host_end_to_last_kernel_tail=47.492 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `66582009`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2], [2, 128, 64], [2, 1, 64]]}`
- `attn_mqa` -> 0.335 ms
  纯GPU kernel时间: `0.036 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.335 ms`, `host_to_first_kernel_gap=47.440562`, `host_end_to_last_kernel_tail=47.142 ms`, `gpu_makespan=0.037 ms`, `gpu_kernel_sum=0.036 ms`
  开始时间(ns): `66746467`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2, 128, 512], [2, 1, 512], [2, 1, 512]]}`
- `o_proj` -> 0.337 ms
  纯GPU kernel时间: `0.051 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.337 ms`, `host_to_first_kernel_gap=46.995051`, `host_end_to_last_kernel_tail=46.710 ms`, `gpu_makespan=0.052 ms`, `gpu_kernel_sum=0.051 ms`
  开始时间(ns): `67240042`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2, 16384]]}`
