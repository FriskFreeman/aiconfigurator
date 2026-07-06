# MLA对齐解析摘要

源文件: `report.sqlite`

## 对齐原则

- `collector/sglang/collect_mla_module.py` 的 MLA module 计时边界是 `model.model.layers[test_layer].self_attn(...)`。
- 因此这里把 `nsys` 中每层的 `model.model.layers.X.self_attn` NVTX range 视为与 collector 对齐的 MLA-module 边界。
- 该区间内部的 `.self_attn.*` 子模块用于做 MLA 内部 breakdown。

## 运行摘要

- `prefill` 对齐成功层: `[0, 1, 2, 3, 4, 5]`
- `prefill` 被切分层: `[]`
- 若某层 `prefill` 出现多个 `self_attn` 实例，则说明 Engine 调度把一次前向切成了多块，已不再与 collector 的单次 MLA-module 采集严格一一对应。

## Layer 0 / prefill / instance 1

- 执行序号: `1`
- Collector对齐边界: `{'Module': 'model.model.layers.0.self_attn'}`
- 整块 MLA-module 时长: `2.480 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.480 ms`, `module_to_last_kernel=5.211 ms`, `host_to_first_kernel_gap=0.33438`, `host_end_to_last_kernel_tail=2.731 ms`, `gpu_makespan=4.877 ms`, `gpu_kernel_sum=2.703 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=2064`, `total_tokens=30805`, `chunked_req_prefix_len=28741`, `current_chunked_req_prefix_len=28741`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[2064, 128, 192], [30805, 128, 192], [30805, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.331 ms
  纯GPU kernel时间: `0.081 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.331 ms`, `host_to_first_kernel_gap=0.11571`, `host_end_to_last_kernel_tail=0.028 ms`, `gpu_makespan=0.243 ms`, `gpu_kernel_sum=0.081 ms`
  开始时间(ns): `21700531`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2064, 7168]]}`
- `q_a_layernorm` -> 0.046 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.046 ms`, `host_to_first_kernel_gap=0.041386`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `22086278`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2064, 1536]]}`
- `q_b_proj` -> 0.245 ms
  纯GPU kernel时间: `0.136 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.245 ms`, `host_to_first_kernel_gap=0.085754`, `host_end_to_last_kernel_tail=0.105 ms`, `gpu_makespan=0.265 ms`, `gpu_kernel_sum=0.136 ms`
  开始时间(ns): `22167350`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2064, 1536]]}`
- `kv_a_layernorm` -> 0.041 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.041 ms`, `host_to_first_kernel_gap=0.043577`, `host_end_to_last_kernel_tail=0.007 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `22473943`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2064, 512]]}`
- `rotary_emb` -> 0.081 ms
  纯GPU kernel时间: `0.026 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.081 ms`, `host_to_first_kernel_gap=0.068873`, `host_end_to_last_kernel_tail=0.014 ms`, `gpu_makespan=0.026 ms`, `gpu_kernel_sum=0.026 ms`
  开始时间(ns): `22556839`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2064], [2064, 128, 64], [2064, 1, 64]]}`
- `kv_b_proj` -> 0.248 ms
  纯GPU kernel时间: `1.127 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.248 ms`, `host_to_first_kernel_gap=0.087339`, `host_end_to_last_kernel_tail=1.081 ms`, `gpu_makespan=1.241 ms`, `gpu_kernel_sum=1.127 ms`
  开始时间(ns): `23145476`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[30805, 512]]}`
- `attn_mha` -> 0.158 ms
  纯GPU kernel时间: `0.954 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.158 ms`, `host_to_first_kernel_gap=1.870766`, `host_end_to_last_kernel_tail=2.667 ms`, `gpu_makespan=0.954 ms`, `gpu_kernel_sum=0.954 ms`
  开始时间(ns): `23495805`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2064, 128, 192], [30805, 128, 192], [30805, 128, 128]]}`
- `o_proj` -> 0.249 ms
  纯GPU kernel时间: `0.369 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.249 ms`, `host_to_first_kernel_gap=2.624898`, `host_end_to_last_kernel_tail=2.746 ms`, `gpu_makespan=0.371 ms`, `gpu_kernel_sum=0.369 ms`
  开始时间(ns): `23697448`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2064, 16384]]}`

## Layer 1 / prefill / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `3.234 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=3.234 ms`, `module_to_last_kernel=6.965 ms`, `host_to_first_kernel_gap=3.249951`, `host_end_to_last_kernel_tail=3.731 ms`, `gpu_makespan=3.715 ms`, `gpu_kernel_sum=2.762 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=2064`, `total_tokens=30805`, `chunked_req_prefix_len=28741`, `current_chunked_req_prefix_len=28741`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[2064, 128, 192], [30805, 128, 192], [30805, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.221 ms
  纯GPU kernel时间: `0.080 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.221 ms`, `host_to_first_kernel_gap=3.067186`, `host_end_to_last_kernel_tail=2.927 ms`, `gpu_makespan=0.081 ms`, `gpu_kernel_sum=0.080 ms`
  开始时间(ns): `24961302`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2064, 7168]]}`
- `q_a_layernorm` -> 0.044 ms
  纯GPU kernel时间: `0.006 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.044 ms`, `host_to_first_kernel_gap=2.875527`, `host_end_to_last_kernel_tail=2.838 ms`, `gpu_makespan=0.006 ms`, `gpu_kernel_sum=0.006 ms`
  开始时间(ns): `25234240`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2064, 1536]]}`
- `q_b_proj` -> 0.196 ms
  纯GPU kernel时间: `0.131 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.196 ms`, `host_to_first_kernel_gap=2.812265`, `host_end_to_last_kernel_tail=2.749 ms`, `gpu_makespan=0.132 ms`, `gpu_kernel_sum=0.131 ms`
  开始时间(ns): `25305246`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2064, 1536]]}`
- `kv_a_layernorm` -> 0.038 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.038 ms`, `host_to_first_kernel_gap=2.689875`, `host_end_to_last_kernel_tail=2.657 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `25559796`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2064, 512]]}`
- `rotary_emb` -> 0.081 ms
  纯GPU kernel时间: `0.026 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.081 ms`, `host_to_first_kernel_gap=2.620363`, `host_end_to_last_kernel_tail=2.565 ms`, `gpu_makespan=0.026 ms`, `gpu_kernel_sum=0.026 ms`
  开始时间(ns): `25637020`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2064], [2064, 128, 64], [2064, 1, 64]]}`
- `kv_b_proj` -> 0.346 ms
  纯GPU kernel时间: `1.184 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.346 ms`, `host_to_first_kernel_gap=1.247016`, `host_end_to_last_kernel_tail=2.088 ms`, `gpu_makespan=1.186 ms`, `gpu_kernel_sum=1.184 ms`
  开始时间(ns): `27082239`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[30805, 512]]}`
- `attn_mha` -> 0.159 ms
  纯GPU kernel时间: `0.956 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.159 ms`, `host_to_first_kernel_gap=2.858463`, `host_end_to_last_kernel_tail=3.656 ms`, `gpu_makespan=0.956 ms`, `gpu_kernel_sum=0.956 ms`
  开始时间(ns): `27550437`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2064, 128, 192], [30805, 128, 192], [30805, 128, 128]]}`
- `o_proj` -> 0.245 ms
  纯GPU kernel时间: `0.374 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.245 ms`, `host_to_first_kernel_gap=3.614989`, `host_end_to_last_kernel_tail=3.746 ms`, `gpu_makespan=0.376 ms`, `gpu_kernel_sum=0.374 ms`
  开始时间(ns): `27752598`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2064, 16384]]}`

## Layer 2 / prefill / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.961 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.961 ms`, `module_to_last_kernel=7.888 ms`, `host_to_first_kernel_gap=4.243635`, `host_end_to_last_kernel_tail=5.927 ms`, `gpu_makespan=3.645 ms`, `gpu_kernel_sum=2.692 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=2064`, `total_tokens=30805`, `chunked_req_prefix_len=28741`, `current_chunked_req_prefix_len=28741`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[2064, 128, 192], [30805, 128, 192], [30805, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.230 ms
  纯GPU kernel时间: `0.080 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.230 ms`, `host_to_first_kernel_gap=4.051202`, `host_end_to_last_kernel_tail=3.902 ms`, `gpu_makespan=0.081 ms`, `gpu_kernel_sum=0.080 ms`
  开始时间(ns): `29024670`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2064, 7168]]}`
- `q_a_layernorm` -> 0.046 ms
  纯GPU kernel时间: `0.006 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.046 ms`, `host_to_first_kernel_gap=3.8507`, `host_end_to_last_kernel_tail=3.811 ms`, `gpu_makespan=0.006 ms`, `gpu_kernel_sum=0.006 ms`
  开始时间(ns): `29306004`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2064, 1536]]}`
- `q_b_proj` -> 0.200 ms
  纯GPU kernel时间: `0.131 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.200 ms`, `host_to_first_kernel_gap=3.783392`, `host_end_to_last_kernel_tail=3.716 ms`, `gpu_makespan=0.133 ms`, `gpu_kernel_sum=0.131 ms`
  开始时间(ns): `29381088`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2064, 1536]]}`
- `kv_a_layernorm` -> 0.042 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.042 ms`, `host_to_first_kernel_gap=3.649183`, `host_end_to_last_kernel_tail=3.612 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `29647937`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2064, 512]]}`
- `rotary_emb` -> 0.076 ms
  纯GPU kernel时间: `0.026 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.076 ms`, `host_to_first_kernel_gap=3.572402`, `host_end_to_last_kernel_tail=3.522 ms`, `gpu_makespan=0.026 ms`, `gpu_kernel_sum=0.026 ms`
  开始时间(ns): `29731470`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2064], [2064, 128, 64], [2064, 1, 64]]}`
- `kv_b_proj` -> 0.241 ms
  纯GPU kernel时间: `1.114 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.241 ms`, `host_to_first_kernel_gap=3.351038`, `host_end_to_last_kernel_tail=4.225 ms`, `gpu_makespan=1.115 ms`, `gpu_kernel_sum=1.114 ms`
  开始时间(ns): `30024802`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[30805, 512]]}`
- `attn_mha` -> 0.143 ms
  纯GPU kernel时间: `0.959 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.143 ms`, `host_to_first_kernel_gap=5.022627`, `host_end_to_last_kernel_tail=5.838 ms`, `gpu_makespan=0.959 ms`, `gpu_kernel_sum=0.959 ms`
  开始时间(ns): `30363674`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2064, 128, 192], [30805, 128, 192], [30805, 128, 128]]}`
- `o_proj` -> 0.233 ms
  纯GPU kernel时间: `0.371 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.233 ms`, `host_to_first_kernel_gap=5.800797`, `host_end_to_last_kernel_tail=5.941 ms`, `gpu_makespan=0.373 ms`, `gpu_kernel_sum=0.371 ms`
  开始时间(ns): `30546270`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2064, 16384]]}`

## Layer 3 / prefill / instance 1

- 执行序号: `4`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `1.900 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.900 ms`, `module_to_last_kernel=10.145 ms`, `host_to_first_kernel_gap=6.461395`, `host_end_to_last_kernel_tail=8.245 ms`, `gpu_makespan=3.684 ms`, `gpu_kernel_sum=2.732 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=2064`, `total_tokens=30805`, `chunked_req_prefix_len=28741`, `current_chunked_req_prefix_len=28741`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[2064, 128, 192], [30805, 128, 192], [30805, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.219 ms
  纯GPU kernel时间: `0.080 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.219 ms`, `host_to_first_kernel_gap=6.267261`, `host_end_to_last_kernel_tail=6.131 ms`, `gpu_makespan=0.083 ms`, `gpu_kernel_sum=0.080 ms`
  开始时间(ns): `31785308`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2064, 7168]]}`
- `q_a_layernorm` -> 0.043 ms
  纯GPU kernel时间: `0.006 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.043 ms`, `host_to_first_kernel_gap=6.083193`, `host_end_to_last_kernel_tail=6.047 ms`, `gpu_makespan=0.006 ms`, `gpu_kernel_sum=0.006 ms`
  开始时间(ns): `32052320`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2064, 1536]]}`
- `q_b_proj` -> 0.201 ms
  纯GPU kernel时间: `0.131 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.201 ms`, `host_to_first_kernel_gap=6.019395`, `host_end_to_last_kernel_tail=5.951 ms`, `gpu_makespan=0.133 ms`, `gpu_kernel_sum=0.131 ms`
  开始时间(ns): `32123734`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2064, 1536]]}`
- `kv_a_layernorm` -> 0.040 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.040 ms`, `host_to_first_kernel_gap=5.893407`, `host_end_to_last_kernel_tail=5.858 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `32382457`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2064, 512]]}`
- `rotary_emb` -> 0.074 ms
  纯GPU kernel时间: `0.026 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.074 ms`, `host_to_first_kernel_gap=5.821725`, `host_end_to_last_kernel_tail=5.774 ms`, `gpu_makespan=0.026 ms`, `gpu_kernel_sum=0.026 ms`
  开始时间(ns): `32460539`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2064], [2064, 128, 64], [2064, 1, 64]]}`
- `kv_b_proj` -> 0.235 ms
  纯GPU kernel时间: `1.150 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.235 ms`, `host_to_first_kernel_gap=5.605709`, `host_end_to_last_kernel_tail=6.523 ms`, `gpu_makespan=1.152 ms`, `gpu_kernel_sum=1.150 ms`
  开始时间(ns): `32749259`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[30805, 512]]}`
- `attn_mha` -> 0.137 ms
  纯GPU kernel时间: `0.957 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.137 ms`, `host_to_first_kernel_gap=7.321448`, `host_end_to_last_kernel_tail=8.141 ms`, `gpu_makespan=0.957 ms`, `gpu_kernel_sum=0.957 ms`
  开始时间(ns): `33078765`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2064, 128, 192], [30805, 128, 192], [30805, 128, 128]]}`
- `o_proj` -> 0.221 ms
  纯GPU kernel时间: `0.376 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.221 ms`, `host_to_first_kernel_gap=8.102617`, `host_end_to_last_kernel_tail=8.260 ms`, `gpu_makespan=0.378 ms`, `gpu_kernel_sum=0.376 ms`
  开始时间(ns): `33255803`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2064, 16384]]}`

## Layer 4 / prefill / instance 1

- 执行序号: `5`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `1.975 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.975 ms`, `module_to_last_kernel=13.276 ms`, `host_to_first_kernel_gap=9.599471`, `host_end_to_last_kernel_tail=11.300 ms`, `gpu_makespan=3.676 ms`, `gpu_kernel_sum=2.726 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=2064`, `total_tokens=30805`, `chunked_req_prefix_len=28741`, `current_chunked_req_prefix_len=28741`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[2064, 128, 192], [30805, 128, 192], [30805, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.278 ms
  纯GPU kernel时间: `0.080 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.278 ms`, `host_to_first_kernel_gap=9.410244`, `host_end_to_last_kernel_tail=9.214 ms`, `gpu_makespan=0.081 ms`, `gpu_kernel_sum=0.080 ms`
  开始时间(ns): `35004843`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2064, 7168]]}`
- `q_a_layernorm` -> 0.047 ms
  纯GPU kernel时间: `0.006 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.047 ms`, `host_to_first_kernel_gap=9.162387`, `host_end_to_last_kernel_tail=9.121 ms`, `gpu_makespan=0.006 ms`, `gpu_kernel_sum=0.006 ms`
  开始时间(ns): `35334364`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2064, 1536]]}`
- `q_b_proj` -> 0.215 ms
  纯GPU kernel时间: `0.130 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.215 ms`, `host_to_first_kernel_gap=9.092635`, `host_end_to_last_kernel_tail=9.010 ms`, `gpu_makespan=0.133 ms`, `gpu_kernel_sum=0.130 ms`
  开始时间(ns): `35412180`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2064, 1536]]}`
- `kv_a_layernorm` -> 0.041 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.041 ms`, `host_to_first_kernel_gap=8.944286`, `host_end_to_last_kernel_tail=8.909 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `35692977`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2064, 512]]}`
- `rotary_emb` -> 0.077 ms
  纯GPU kernel时间: `0.027 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.077 ms`, `host_to_first_kernel_gap=8.870595`, `host_end_to_last_kernel_tail=8.820 ms`, `gpu_makespan=0.027 ms`, `gpu_kernel_sum=0.027 ms`
  开始时间(ns): `35773100`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2064], [2064, 128, 64], [2064, 1, 64]]}`
- `kv_b_proj` -> 0.229 ms
  纯GPU kernel时间: `1.139 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.229 ms`, `host_to_first_kernel_gap=8.667625`, `host_end_to_last_kernel_tail=9.579 ms`, `gpu_makespan=1.140 ms`, `gpu_kernel_sum=1.139 ms`
  开始时间(ns): `36046982`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[30805, 512]]}`
- `attn_mha` -> 0.139 ms
  纯GPU kernel时间: `0.959 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.139 ms`, `host_to_first_kernel_gap=10.378702`, `host_end_to_last_kernel_tail=11.199 ms`, `gpu_makespan=0.959 ms`, `gpu_kernel_sum=0.959 ms`
  开始时间(ns): `36369406`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2064, 128, 192], [30805, 128, 192], [30805, 128, 128]]}`
- `o_proj` -> 0.229 ms
  纯GPU kernel时间: `0.380 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.229 ms`, `host_to_first_kernel_gap=11.161307`, `host_end_to_last_kernel_tail=11.315 ms`, `gpu_makespan=0.382 ms`, `gpu_kernel_sum=0.380 ms`
  开始时间(ns): `36547792`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2064, 16384]]}`

## Layer 5 / prefill / instance 1

- 执行序号: `6`
- Collector对齐边界: `{'Module': 'model.model.layers.5.self_attn'}`
- 整块 MLA-module 时长: `2.008 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.008 ms`, `module_to_last_kernel=16.388 ms`, `host_to_first_kernel_gap=12.680775`, `host_end_to_last_kernel_tail=14.380 ms`, `gpu_makespan=3.707 ms`, `gpu_kernel_sum=2.755 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=2064`, `total_tokens=30805`, `chunked_req_prefix_len=28741`, `current_chunked_req_prefix_len=28741`, `current_chunked_req_prefix_len_source=attention_inputs`, `inputs=[[2064, 128, 192], [30805, 128, 192], [30805, 128, 128]]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.276 ms
  纯GPU kernel时间: `0.081 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.276 ms`, `host_to_first_kernel_gap=12.489753`, `host_end_to_last_kernel_tail=12.296 ms`, `gpu_makespan=0.082 ms`, `gpu_kernel_sum=0.081 ms`
  开始时间(ns): `38264461`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2064, 7168]]}`
- `q_a_layernorm` -> 0.048 ms
  纯GPU kernel时间: `0.006 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.048 ms`, `host_to_first_kernel_gap=12.245709`, `host_end_to_last_kernel_tail=12.203 ms`, `gpu_makespan=0.006 ms`, `gpu_kernel_sum=0.006 ms`
  开始时间(ns): `38591193`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2064, 1536]]}`
- `q_b_proj` -> 0.218 ms
  纯GPU kernel时间: `0.131 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.218 ms`, `host_to_first_kernel_gap=12.167928`, `host_end_to_last_kernel_tail=12.082 ms`, `gpu_makespan=0.132 ms`, `gpu_kernel_sum=0.131 ms`
  开始时间(ns): `38676718`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2064, 1536]]}`
- `kv_a_layernorm` -> 0.043 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.043 ms`, `host_to_first_kernel_gap=12.022452`, `host_end_to_last_kernel_tail=11.985 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `38954066`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2064, 512]]}`
- `rotary_emb` -> 0.075 ms
  纯GPU kernel时间: `0.026 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.075 ms`, `host_to_first_kernel_gap=11.948516`, `host_end_to_last_kernel_tail=11.900 ms`, `gpu_makespan=0.026 ms`, `gpu_kernel_sum=0.026 ms`
  开始时间(ns): `39035362`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2064], [2064, 128, 64], [2064, 1, 64]]}`
- `kv_b_proj` -> 0.223 ms
  纯GPU kernel时间: `1.166 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.223 ms`, `host_to_first_kernel_gap=11.741558`, `host_end_to_last_kernel_tail=12.686 ms`, `gpu_makespan=1.168 ms`, `gpu_kernel_sum=1.166 ms`
  开始时间(ns): `39313392`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[30805, 512]]}`
- `attn_mha` -> 0.154 ms
  纯GPU kernel时间: `0.959 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.154 ms`, `host_to_first_kernel_gap=13.480947`, `host_end_to_last_kernel_tail=14.286 ms`, `gpu_makespan=0.959 ms`, `gpu_kernel_sum=0.959 ms`
  开始时间(ns): `39634160`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2064, 128, 192], [30805, 128, 192], [30805, 128, 128]]}`
- `o_proj` -> 0.234 ms
  纯GPU kernel时间: `0.383 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.234 ms`, `host_to_first_kernel_gap=14.244038`, `host_end_to_last_kernel_tail=14.395 ms`, `gpu_makespan=0.385 ms`, `gpu_kernel_sum=0.383 ms`
  开始时间(ns): `39832507`
  原始NVTX: `{'Module': 'model.model.layers.5.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2064, 16384]]}`
