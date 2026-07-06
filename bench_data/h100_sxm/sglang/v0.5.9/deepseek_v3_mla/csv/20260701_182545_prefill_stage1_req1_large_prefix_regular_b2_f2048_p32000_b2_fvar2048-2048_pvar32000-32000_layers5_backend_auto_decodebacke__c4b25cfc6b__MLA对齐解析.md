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
- 整块 MLA-module 时长: `2.556 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.556 ms`, `module_to_last_kernel=20.901 ms`, `host_to_first_kernel_gap=0.297323`, `host_end_to_last_kernel_tail=18.345 ms`, `gpu_makespan=20.603 ms`, `gpu_kernel_sum=17.470 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4096`, `total_tokens=68096`, `chunked_req_prefix_len=64000`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[4096, 128, 192], [68096, 128, 192], [68096, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=4096`, `sum_prefix=64000`, `sum_seq_after=68096`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 2048, 'prefix_len': 32000, 'seq_len_after': 34048, 'prompt_len': 34048, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 2048, 'prefix_len': 32000, 'seq_len_after': 34048, 'prompt_len': 34048, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.409 ms
  纯GPU kernel时间: `0.126 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.409 ms`, `host_to_first_kernel_gap=0.136901`, `host_end_to_last_kernel_tail=0.054 ms`, `gpu_makespan=0.325 ms`, `gpu_kernel_sum=0.126 ms`
  开始时间(ns): `33685939`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4096, 7168]]}`
- `q_a_layernorm` -> 0.050 ms
  纯GPU kernel时间: `0.010 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.050 ms`, `host_to_first_kernel_gap=0.044513`, `host_end_to_last_kernel_tail=0.005 ms`, `gpu_makespan=0.010 ms`, `gpu_kernel_sum=0.010 ms`
  开始时间(ns): `34158230`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4096, 1536]]}`
- `q_b_proj` -> 0.254 ms
  纯GPU kernel时间: `0.238 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.254 ms`, `host_to_first_kernel_gap=0.084483`, `host_end_to_last_kernel_tail=0.200 ms`, `gpu_makespan=0.370 ms`, `gpu_kernel_sum=0.238 ms`
  开始时间(ns): `34242964`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4096, 1536]]}`
- `kv_a_layernorm` -> 0.046 ms
  纯GPU kernel时间: `0.008 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.046 ms`, `host_to_first_kernel_gap=0.133041`, `host_end_to_last_kernel_tail=0.095 ms`, `gpu_makespan=0.008 ms`, `gpu_kernel_sum=0.008 ms`
  开始时间(ns): `34564006`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4096, 512]]}`
- `rotary_emb` -> 0.106 ms
  纯GPU kernel时间: `0.053 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.106 ms`, `host_to_first_kernel_gap=0.092786`, `host_end_to_last_kernel_tail=0.040 ms`, `gpu_makespan=0.053 ms`, `gpu_kernel_sum=0.053 ms`
  开始时间(ns): `34653509`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4096], [4096, 128, 64], [4096, 1, 64]]}`
- `kv_b_proj` -> 0.265 ms
  纯GPU kernel时间: `2.686 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.265 ms`, `host_to_first_kernel_gap=0.091069`, `host_end_to_last_kernel_tail=2.616 ms`, `gpu_makespan=2.790 ms`, `gpu_kernel_sum=2.686 ms`
  开始时间(ns): `35227673`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[68096, 512]]}`
- `attn_mha` -> 0.176 ms
  纯GPU kernel时间: `13.719 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.176 ms`, `host_to_first_kernel_gap=4.464752`, `host_end_to_last_kernel_tail=18.007 ms`, `gpu_makespan=13.719 ms`, `gpu_kernel_sum=13.719 ms`
  开始时间(ns): `35609119`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4096, 128, 192], [68096, 128, 192], [68096, 128, 128]]}`
- `o_proj` -> 0.238 ms
  纯GPU kernel时间: `0.631 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.238 ms`, `host_to_first_kernel_gap=17.965921`, `host_end_to_last_kernel_tail=18.360 ms`, `gpu_makespan=0.632 ms`, `gpu_kernel_sum=0.631 ms`
  开始时间(ns): `35828283`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4096, 16384]]}`

## Layer 1 / prefill / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `1.931 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.931 ms`, `module_to_last_kernel=39.574 ms`, `host_to_first_kernel_gap=19.780358`, `host_end_to_last_kernel_tail=37.643 ms`, `gpu_makespan=19.794 ms`, `gpu_kernel_sum=17.729 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4096`, `total_tokens=68096`, `chunked_req_prefix_len=64000`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[4096, 128, 192], [68096, 128, 192], [68096, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=4096`, `sum_prefix=64000`, `sum_seq_after=68096`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 2048, 'prefix_len': 32000, 'seq_len_after': 34048, 'prompt_len': 34048, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 2048, 'prefix_len': 32000, 'seq_len_after': 34048, 'prompt_len': 34048, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.237 ms
  纯GPU kernel时间: `0.124 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.237 ms`, `host_to_first_kernel_gap=19.635827`, `host_end_to_last_kernel_tail=19.525 ms`, `gpu_makespan=0.126 ms`, `gpu_kernel_sum=0.124 ms`
  开始时间(ns): `37112068`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4096, 7168]]}`
- `q_a_layernorm` -> 0.046 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.046 ms`, `host_to_first_kernel_gap=19.464797`, `host_end_to_last_kernel_tail=19.428 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `37409402`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4096, 1536]]}`
- `q_b_proj` -> 0.211 ms
  纯GPU kernel时间: `0.234 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.211 ms`, `host_to_first_kernel_gap=19.397965`, `host_end_to_last_kernel_tail=19.423 ms`, `gpu_makespan=0.236 ms`, `gpu_kernel_sum=0.234 ms`
  开始时间(ns): `37486858`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4096, 1536]]}`
- `kv_a_layernorm` -> 0.039 ms
  纯GPU kernel时间: `0.008 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.039 ms`, `host_to_first_kernel_gap=19.364326`, `host_end_to_last_kernel_tail=19.333 ms`, `gpu_makespan=0.008 ms`, `gpu_kernel_sum=0.008 ms`
  开始时间(ns): `37756209`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4096, 512]]}`
- `rotary_emb` -> 0.077 ms
  纯GPU kernel时间: `0.054 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.077 ms`, `host_to_first_kernel_gap=19.297192`, `host_end_to_last_kernel_tail=19.273 ms`, `gpu_makespan=0.054 ms`, `gpu_kernel_sum=0.054 ms`
  开始时间(ns): `37833743`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4096], [4096, 128, 64], [4096, 1, 64]]}`
- `kv_b_proj` -> 0.211 ms
  纯GPU kernel时间: `2.818 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.211 ms`, `host_to_first_kernel_gap=19.149333`, `host_end_to_last_kernel_tail=21.758 ms`, `gpu_makespan=2.820 ms`, `gpu_kernel_sum=2.818 ms`
  开始时间(ns): `38122082`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[68096, 512]]}`
- `attn_mha` -> 0.141 ms
  纯GPU kernel时间: `13.796 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.141 ms`, `host_to_first_kernel_gap=23.609972`, `host_end_to_last_kernel_tail=37.264 ms`, `gpu_makespan=13.796 ms`, `gpu_kernel_sum=13.796 ms`
  开始时间(ns): `38447132`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4096, 128, 192], [68096, 128, 192], [68096, 128, 128]]}`
- `o_proj` -> 0.251 ms
  纯GPU kernel时间: `0.686 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.251 ms`, `host_to_first_kernel_gap=37.22165`, `host_end_to_last_kernel_tail=37.658 ms`, `gpu_makespan=0.687 ms`, `gpu_kernel_sum=0.686 ms`
  开始时间(ns): `38632458`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4096, 16384]]}`

## Layer 2 / prefill / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.889 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.889 ms`, `module_to_last_kernel=59.485 ms`, `host_to_first_kernel_gap=39.300679`, `host_end_to_last_kernel_tail=57.595 ms`, `gpu_makespan=20.184 ms`, `gpu_kernel_sum=18.119 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4096`, `total_tokens=68096`, `chunked_req_prefix_len=64000`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[4096, 128, 192], [68096, 128, 192], [68096, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=4096`, `sum_prefix=64000`, `sum_seq_after=68096`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 2048, 'prefix_len': 32000, 'seq_len_after': 34048, 'prompt_len': 34048, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 2048, 'prefix_len': 32000, 'seq_len_after': 34048, 'prompt_len': 34048, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.218 ms
  纯GPU kernel时间: `0.131 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.218 ms`, `host_to_first_kernel_gap=39.18894`, `host_end_to_last_kernel_tail=39.104 ms`, `gpu_makespan=0.133 ms`, `gpu_kernel_sum=0.131 ms`
  开始时间(ns): `39831884`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4096, 7168]]}`
- `q_a_layernorm` -> 0.044 ms
  纯GPU kernel时间: `0.010 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.044 ms`, `host_to_first_kernel_gap=39.056763`, `host_end_to_last_kernel_tail=39.023 ms`, `gpu_makespan=0.010 ms`, `gpu_kernel_sum=0.010 ms`
  开始时间(ns): `40097565`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4096, 1536]]}`
- `q_b_proj` -> 0.207 ms
  纯GPU kernel时间: `0.249 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.207 ms`, `host_to_first_kernel_gap=38.993916`, `host_end_to_last_kernel_tail=39.037 ms`, `gpu_makespan=0.251 ms`, `gpu_kernel_sum=0.249 ms`
  开始时间(ns): `40171964`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4096, 1536]]}`
- `kv_a_layernorm` -> 0.042 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.042 ms`, `host_to_first_kernel_gap=38.979467`, `host_end_to_last_kernel_tail=38.946 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `40436812`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4096, 512]]}`
- `rotary_emb` -> 0.075 ms
  纯GPU kernel时间: `0.054 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.075 ms`, `host_to_first_kernel_gap=38.910856`, `host_end_to_last_kernel_tail=38.890 ms`, `gpu_makespan=0.054 ms`, `gpu_kernel_sum=0.054 ms`
  开始时间(ns): `40516335`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4096], [4096, 128, 64], [4096, 1, 64]]}`
- `kv_b_proj` -> 0.274 ms
  纯GPU kernel时间: `2.681 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.274 ms`, `host_to_first_kernel_gap=38.768045`, `host_end_to_last_kernel_tail=41.177 ms`, `gpu_makespan=2.682 ms`, `gpu_kernel_sum=2.681 ms`
  开始时间(ns): `40799786`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[68096, 512]]}`
- `attn_mha` -> 0.137 ms
  纯GPU kernel时间: `14.237 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.137 ms`, `host_to_first_kernel_gap=43.041325`, `host_end_to_last_kernel_tail=57.142 ms`, `gpu_makespan=14.237 ms`, `gpu_kernel_sum=14.237 ms`
  开始时间(ns): `41174724`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4096, 128, 192], [68096, 128, 192], [68096, 128, 128]]}`
- `o_proj` -> 0.240 ms
  纯GPU kernel时间: `0.748 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.240 ms`, `host_to_first_kernel_gap=57.101358`, `host_end_to_last_kernel_tail=57.610 ms`, `gpu_makespan=0.749 ms`, `gpu_kernel_sum=0.748 ms`
  开始时间(ns): `41354318`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4096, 16384]]}`

## Layer 3 / prefill / instance 1

- 执行序号: `4`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `1.798 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.798 ms`, `module_to_last_kernel=80.452 ms`, `host_to_first_kernel_gap=59.46529`, `host_end_to_last_kernel_tail=78.655 ms`, `gpu_makespan=20.987 ms`, `gpu_kernel_sum=18.919 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4096`, `total_tokens=68096`, `chunked_req_prefix_len=64000`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[4096, 128, 192], [68096, 128, 192], [68096, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=4096`, `sum_prefix=64000`, `sum_seq_after=68096`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 2048, 'prefix_len': 32000, 'seq_len_after': 34048, 'prompt_len': 34048, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 2048, 'prefix_len': 32000, 'seq_len_after': 34048, 'prompt_len': 34048, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.211 ms
  纯GPU kernel时间: `0.141 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.211 ms`, `host_to_first_kernel_gap=59.3522`, `host_end_to_last_kernel_tail=59.283 ms`, `gpu_makespan=0.142 ms`, `gpu_kernel_sum=0.141 ms`
  开始时间(ns): `42522768`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4096, 7168]]}`
- `q_a_layernorm` -> 0.048 ms
  纯GPU kernel时间: `0.011 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.048 ms`, `host_to_first_kernel_gap=59.23474`, `host_end_to_last_kernel_tail=59.198 ms`, `gpu_makespan=0.011 ms`, `gpu_kernel_sum=0.011 ms`
  开始时间(ns): `42782563`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4096, 1536]]}`
- `q_b_proj` -> 0.198 ms
  纯GPU kernel时间: `0.267 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.198 ms`, `host_to_first_kernel_gap=59.168974`, `host_end_to_last_kernel_tail=59.239 ms`, `gpu_makespan=0.268 ms`, `gpu_kernel_sum=0.267 ms`
  开始时间(ns): `42860969`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4096, 1536]]}`
- `kv_a_layernorm` -> 0.039 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.039 ms`, `host_to_first_kernel_gap=59.180175`, `host_end_to_last_kernel_tail=59.151 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `43117928`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4096, 512]]}`
- `rotary_emb` -> 0.071 ms
  纯GPU kernel时间: `0.053 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.071 ms`, `host_to_first_kernel_gap=59.114244`, `host_end_to_last_kernel_tail=59.096 ms`, `gpu_makespan=0.053 ms`, `gpu_kernel_sum=0.053 ms`
  开始时间(ns): `43195571`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4096], [4096, 128, 64], [4096, 1, 64]]}`
- `kv_b_proj` -> 0.222 ms
  纯GPU kernel时间: `2.529 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.222 ms`, `host_to_first_kernel_gap=58.975945`, `host_end_to_last_kernel_tail=61.285 ms`, `gpu_makespan=2.530 ms`, `gpu_kernel_sum=2.529 ms`
  开始时间(ns): `43474926`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[68096, 512]]}`
- `attn_mha` -> 0.136 ms
  纯GPU kernel时间: `15.146 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.136 ms`, `host_to_first_kernel_gap=63.154725`, `host_end_to_last_kernel_tail=78.165 ms`, `gpu_makespan=15.146 ms`, `gpu_kernel_sum=15.146 ms`
  开始时间(ns): `43793035`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4096, 128, 192], [68096, 128, 192], [68096, 128, 128]]}`
- `o_proj` -> 0.219 ms
  纯GPU kernel时间: `0.763 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.219 ms`, `host_to_first_kernel_gap=78.122248`, `host_end_to_last_kernel_tail=78.669 ms`, `gpu_makespan=0.765 ms`, `gpu_kernel_sum=0.763 ms`
  开始时间(ns): `43974803`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4096, 16384]]}`

## Layer 4 / prefill / instance 1

- 执行序号: `5`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `1.955 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.955 ms`, `module_to_last_kernel=103.899 ms`, `host_to_first_kernel_gap=82.570566`, `host_end_to_last_kernel_tail=101.944 ms`, `gpu_makespan=21.328 ms`, `gpu_kernel_sum=19.262 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4096`, `total_tokens=68096`, `chunked_req_prefix_len=64000`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[4096, 128, 192], [68096, 128, 192], [68096, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=4096`, `sum_prefix=64000`, `sum_seq_after=68096`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 2048, 'prefix_len': 32000, 'seq_len_after': 34048, 'prompt_len': 34048, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 2048, 'prefix_len': 32000, 'seq_len_after': 34048, 'prompt_len': 34048, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.274 ms
  纯GPU kernel时间: `0.143 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.274 ms`, `host_to_first_kernel_gap=82.44946`, `host_end_to_last_kernel_tail=82.321 ms`, `gpu_makespan=0.146 ms`, `gpu_kernel_sum=0.143 ms`
  开始时间(ns): `45834558`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4096, 7168]]}`
- `q_a_layernorm` -> 0.047 ms
  纯GPU kernel时间: `0.010 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.047 ms`, `host_to_first_kernel_gap=82.2641`, `host_end_to_last_kernel_tail=82.227 ms`, `gpu_makespan=0.010 ms`, `gpu_kernel_sum=0.010 ms`
  开始时间(ns): `46166254`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4096, 1536]]}`
- `q_b_proj` -> 0.230 ms
  纯GPU kernel时间: `0.271 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.230 ms`, `host_to_first_kernel_gap=82.195133`, `host_end_to_last_kernel_tail=82.237 ms`, `gpu_makespan=0.272 ms`, `gpu_kernel_sum=0.271 ms`
  开始时间(ns): `46246837`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4096, 1536]]}`
- `kv_a_layernorm` -> 0.040 ms
  纯GPU kernel时间: `0.008 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.040 ms`, `host_to_first_kernel_gap=82.175691`, `host_end_to_last_kernel_tail=82.144 ms`, `gpu_makespan=0.008 ms`, `gpu_kernel_sum=0.008 ms`
  开始时间(ns): `46538567`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4096, 512]]}`
- `rotary_emb` -> 0.081 ms
  纯GPU kernel时间: `0.054 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.081 ms`, `host_to_first_kernel_gap=82.106471`, `host_end_to_last_kernel_tail=82.079 ms`, `gpu_makespan=0.054 ms`, `gpu_kernel_sum=0.054 ms`
  开始时间(ns): `46618283`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4096], [4096, 128, 64], [4096, 1, 64]]}`
- `kv_b_proj` -> 0.221 ms
  纯GPU kernel时间: `2.940 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.221 ms`, `host_to_first_kernel_gap=81.967229`, `host_end_to_last_kernel_tail=84.689 ms`, `gpu_makespan=2.942 ms`, `gpu_kernel_sum=2.940 ms`
  开始时间(ns): `46899924`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[68096, 512]]}`
- `attn_mha` -> 0.140 ms
  纯GPU kernel时间: `15.105 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.140 ms`, `host_to_first_kernel_gap=86.553704`, `host_end_to_last_kernel_tail=101.519 ms`, `gpu_makespan=15.105 ms`, `gpu_kernel_sum=15.105 ms`
  开始时间(ns): `47220834`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4096, 128, 192], [68096, 128, 192], [68096, 128, 128]]}`
- `o_proj` -> 0.242 ms
  纯GPU kernel时间: `0.730 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.242 ms`, `host_to_first_kernel_gap=101.46982`, `host_end_to_last_kernel_tail=101.958 ms`, `gpu_makespan=0.731 ms`, `gpu_kernel_sum=0.730 ms`
  开始时间(ns): `47411513`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4096, 16384]]}`

## Layer 0 / decode / instance 1

- 执行序号: `6`
- Collector对齐边界: `{'Module': 'model.model.layers.0.self_attn'}`
- 整块 MLA-module 时长: `1.852 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.852 ms`, `module_to_last_kernel=104.434 ms`, `host_to_first_kernel_gap=104.241636`, `host_end_to_last_kernel_tail=102.582 ms`, `gpu_makespan=0.192 ms`, `gpu_kernel_sum=0.159 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.311 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.311 ms`, `host_to_first_kernel_gap=104.168482`, `host_end_to_last_kernel_tail=103.874 ms`, `gpu_makespan=0.016 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `51534089`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2, 7168]]}`
- `q_a_layernorm` -> 0.048 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.048 ms`, `host_to_first_kernel_gap=103.811596`, `host_end_to_last_kernel_tail=103.766 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `51907135`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2, 1536]]}`
- `kv_a_layernorm` -> 0.032 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.032 ms`, `host_to_first_kernel_gap=103.741185`, `host_end_to_last_kernel_tail=103.712 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `51979818`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2, 512]]}`
- `q_b_proj` -> 0.233 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.233 ms`, `host_to_first_kernel_gap=103.674409`, `host_end_to_last_kernel_tail=103.462 ms`, `gpu_makespan=0.021 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `52049922`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2, 1536]]}`
- `rotary_emb` -> 0.086 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.086 ms`, `host_to_first_kernel_gap=103.295998`, `host_end_to_last_kernel_tail=103.212 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `52461773`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2], [2, 128, 64], [2, 1, 64]]}`
- `attn_mqa` -> 0.292 ms
  纯GPU kernel时间: `0.065 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.292 ms`, `host_to_first_kernel_gap=103.161148`, `host_end_to_last_kernel_tail=102.935 ms`, `gpu_makespan=0.066 ms`, `gpu_kernel_sum=0.065 ms`
  开始时间(ns): `52599695`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2, 128, 512], [2, 1, 512], [2, 1, 512]]}`
- `o_proj` -> 0.273 ms
  纯GPU kernel时间: `0.054 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.273 ms`, `host_to_first_kernel_gap=102.818845`, `host_end_to_last_kernel_tail=102.601 ms`, `gpu_makespan=0.055 ms`, `gpu_kernel_sum=0.054 ms`
  开始时间(ns): `53020526`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2, 16384]]}`

## Layer 1 / decode / instance 1

- 执行序号: `7`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `1.643 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.643 ms`, `module_to_last_kernel=102.084 ms`, `host_to_first_kernel_gap=101.894841`, `host_end_to_last_kernel_tail=100.441 ms`, `gpu_makespan=0.189 ms`, `gpu_kernel_sum=0.159 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.222 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.222 ms`, `host_to_first_kernel_gap=101.830528`, `host_end_to_last_kernel_tail=101.625 ms`, `gpu_makespan=0.016 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `54227723`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2, 7168]]}`
- `q_a_layernorm` -> 0.043 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.043 ms`, `host_to_first_kernel_gap=101.566701`, `host_end_to_last_kernel_tail=101.526 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `54507710`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2, 1536]]}`
- `kv_a_layernorm` -> 0.030 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.030 ms`, `host_to_first_kernel_gap=101.501134`, `host_end_to_last_kernel_tail=101.473 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `54575325`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2, 512]]}`
- `q_b_proj` -> 0.206 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.206 ms`, `host_to_first_kernel_gap=101.440657`, `host_end_to_last_kernel_tail=101.255 ms`, `gpu_makespan=0.020 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `54639066`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2, 1536]]}`
- `rotary_emb` -> 0.077 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.077 ms`, `host_to_first_kernel_gap=101.128397`, `host_end_to_last_kernel_tail=101.053 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `54984094`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2], [2, 128, 64], [2, 1, 64]]}`
- `attn_mqa` -> 0.287 ms
  纯GPU kernel时间: `0.065 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.287 ms`, `host_to_first_kernel_gap=101.009437`, `host_end_to_last_kernel_tail=100.789 ms`, `gpu_makespan=0.066 ms`, `gpu_kernel_sum=0.065 ms`
  开始时间(ns): `55106094`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2, 128, 512], [2, 1, 512], [2, 1, 512]]}`
- `o_proj` -> 0.271 ms
  纯GPU kernel时间: `0.052 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.271 ms`, `host_to_first_kernel_gap=100.675529`, `host_end_to_last_kernel_tail=100.459 ms`, `gpu_makespan=0.054 ms`, `gpu_kernel_sum=0.052 ms`
  开始时间(ns): `55518562`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2, 16384]]}`

## Layer 2 / decode / instance 1

- 执行序号: `8`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.643 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.643 ms`, `module_to_last_kernel=99.975 ms`, `host_to_first_kernel_gap=99.783972`, `host_end_to_last_kernel_tail=98.332 ms`, `gpu_makespan=0.191 ms`, `gpu_kernel_sum=0.159 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.212 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.212 ms`, `host_to_first_kernel_gap=99.723442`, `host_end_to_last_kernel_tail=99.529 ms`, `gpu_makespan=0.017 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `56692952`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2, 7168]]}`
- `q_a_layernorm` -> 0.044 ms
  纯GPU kernel时间: `0.003 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.044 ms`, `host_to_first_kernel_gap=99.471262`, `host_end_to_last_kernel_tail=99.430 ms`, `gpu_makespan=0.003 ms`, `gpu_kernel_sum=0.003 ms`
  开始时间(ns): `56961676`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2, 1536]]}`
- `kv_a_layernorm` -> 0.031 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.031 ms`, `host_to_first_kernel_gap=99.405792`, `host_end_to_last_kernel_tail=99.377 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `57029610`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2, 512]]}`
- `q_b_proj` -> 0.236 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.236 ms`, `host_to_first_kernel_gap=99.343495`, `host_end_to_last_kernel_tail=99.127 ms`, `gpu_makespan=0.020 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `57095043`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2, 1536]]}`
- `rotary_emb` -> 0.075 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.075 ms`, `host_to_first_kernel_gap=98.991994`, `host_end_to_last_kernel_tail=98.919 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `57478544`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2], [2, 128, 64], [2, 1, 64]]}`
- `attn_mqa` -> 0.275 ms
  纯GPU kernel时间: `0.065 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.275 ms`, `host_to_first_kernel_gap=98.873638`, `host_end_to_last_kernel_tail=98.666 ms`, `gpu_makespan=0.067 ms`, `gpu_kernel_sum=0.065 ms`
  开始时间(ns): `57599908`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2, 128, 512], [2, 1, 512], [2, 1, 512]]}`
- `o_proj` -> 0.261 ms
  纯GPU kernel时间: `0.054 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.261 ms`, `host_to_first_kernel_gap=98.555299`, `host_end_to_last_kernel_tail=98.349 ms`, `gpu_makespan=0.055 ms`, `gpu_kernel_sum=0.054 ms`
  开始时间(ns): `57997639`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2, 16384]]}`

## Layer 3 / decode / instance 1

- 执行序号: `9`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `1.598 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.598 ms`, `module_to_last_kernel=97.859 ms`, `host_to_first_kernel_gap=97.670028`, `host_end_to_last_kernel_tail=96.261 ms`, `gpu_makespan=0.189 ms`, `gpu_kernel_sum=0.157 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.211 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.211 ms`, `host_to_first_kernel_gap=97.612679`, `host_end_to_last_kernel_tail=97.418 ms`, `gpu_makespan=0.017 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `59161475`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2, 7168]]}`
- `q_a_layernorm` -> 0.046 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.046 ms`, `host_to_first_kernel_gap=97.355868`, `host_end_to_last_kernel_tail=97.312 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `59434958`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2, 1536]]}`
- `kv_a_layernorm` -> 0.031 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.031 ms`, `host_to_first_kernel_gap=97.287737`, `host_end_to_last_kernel_tail=97.259 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `59505009`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2, 512]]}`
- `q_b_proj` -> 0.215 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.215 ms`, `host_to_first_kernel_gap=97.225026`, `host_end_to_last_kernel_tail=97.029 ms`, `gpu_makespan=0.019 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `59571912`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2, 1536]]}`
- `rotary_emb` -> 0.075 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.075 ms`, `host_to_first_kernel_gap=96.900815`, `host_end_to_last_kernel_tail=96.828 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `59927515`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2], [2, 128, 64], [2, 1, 64]]}`
- `attn_mqa` -> 0.268 ms
  纯GPU kernel时间: `0.065 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.268 ms`, `host_to_first_kernel_gap=96.783918`, `host_end_to_last_kernel_tail=96.582 ms`, `gpu_makespan=0.066 ms`, `gpu_kernel_sum=0.065 ms`
  开始时间(ns): `60047484`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2, 128, 512], [2, 1, 512], [2, 1, 512]]}`
- `o_proj` -> 0.246 ms
  纯GPU kernel时间: `0.052 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.246 ms`, `host_to_first_kernel_gap=96.470452`, `host_end_to_last_kernel_tail=96.278 ms`, `gpu_makespan=0.053 ms`, `gpu_kernel_sum=0.052 ms`
  开始时间(ns): `60439702`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2, 16384]]}`

## Layer 4 / decode / instance 1

- 执行序号: `10`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `1.727 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.727 ms`, `module_to_last_kernel=95.221 ms`, `host_to_first_kernel_gap=95.031541`, `host_end_to_last_kernel_tail=93.494 ms`, `gpu_makespan=0.190 ms`, `gpu_kernel_sum=0.158 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.291 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.291 ms`, `host_to_first_kernel_gap=94.966402`, `host_end_to_last_kernel_tail=94.691 ms`, `gpu_makespan=0.016 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `62209319`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2, 7168]]}`
- `q_a_layernorm` -> 0.052 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.052 ms`, `host_to_first_kernel_gap=94.630355`, `host_end_to_last_kernel_tail=94.580 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `62561654`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2, 1536]]}`
- `kv_a_layernorm` -> 0.031 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.031 ms`, `host_to_first_kernel_gap=94.555851`, `host_end_to_last_kernel_tail=94.527 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `62638174`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2, 512]]}`
- `q_b_proj` -> 0.220 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.220 ms`, `host_to_first_kernel_gap=94.491428`, `host_end_to_last_kernel_tail=94.291 ms`, `gpu_makespan=0.020 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `62705733`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2, 1536]]}`
- `rotary_emb` -> 0.079 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.079 ms`, `host_to_first_kernel_gap=94.154352`, `host_end_to_last_kernel_tail=94.078 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `63073657`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2], [2, 128, 64], [2, 1, 64]]}`
- `attn_mqa` -> 0.281 ms
  纯GPU kernel时间: `0.065 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.281 ms`, `host_to_first_kernel_gap=94.034495`, `host_end_to_last_kernel_tail=93.819 ms`, `gpu_makespan=0.066 ms`, `gpu_kernel_sum=0.065 ms`
  开始时间(ns): `63197610`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2, 128, 512], [2, 1, 512], [2, 1, 512]]}`
- `o_proj` -> 0.254 ms
  纯GPU kernel时间: `0.053 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.254 ms`, `host_to_first_kernel_gap=93.710327`, `host_end_to_last_kernel_tail=93.511 ms`, `gpu_makespan=0.054 ms`, `gpu_kernel_sum=0.053 ms`
  开始时间(ns): `63600530`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2, 16384]]}`
