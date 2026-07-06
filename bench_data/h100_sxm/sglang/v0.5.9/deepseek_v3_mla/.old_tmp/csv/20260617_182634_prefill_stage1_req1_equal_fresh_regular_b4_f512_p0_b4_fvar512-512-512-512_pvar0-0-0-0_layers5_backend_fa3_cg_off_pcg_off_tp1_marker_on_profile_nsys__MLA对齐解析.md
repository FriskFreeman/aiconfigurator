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
- 整块 MLA-module 时长: `45776.488 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=45776.488 ms`, `module_to_last_kernel=45776.583 ms`, `host_to_first_kernel_gap=1.409237`, `host_end_to_last_kernel_tail=0.095 ms`, `gpu_makespan=45775.174 ms`, `gpu_kernel_sum=0.832 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=2048`, `total_tokens=2048`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[2048, 128, 192], [2048, 128, 192], [2048, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=2048`, `sum_prefix=0`, `sum_seq_after=2048`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 512, 'prefix_len': 0, 'seq_len_after': 512, 'prompt_len': 512, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 512, 'prefix_len': 0, 'seq_len_after': 512, 'prompt_len': 512, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 512, 'prefix_len': 0, 'seq_len_after': 512, 'prompt_len': 512, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 512, 'prefix_len': 0, 'seq_len_after': 512, 'prompt_len': 512, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 5390.040 ms
  纯GPU kernel时间: `0.073 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=5390.040 ms`, `host_to_first_kernel_gap=1.150921`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=5388.767 ms`, `gpu_kernel_sum=0.073 ms`
  开始时间(ns): `4998388195`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2048, 7168]]}`
- `q_a_layernorm` -> 0.154 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.154 ms`, `host_to_first_kernel_gap=0.14602`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `10388613275`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2048, 1536]]}`
- `q_b_proj` -> 6018.130 ms
  纯GPU kernel时间: `0.124 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=6018.130 ms`, `host_to_first_kernel_gap=0.187895`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=6017.902 ms`, `gpu_kernel_sum=0.124 ms`
  开始时间(ns): `10388818408`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2048, 1536]]}`
- `kv_a_layernorm` -> 0.178 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.178 ms`, `host_to_first_kernel_gap=0.151988`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `16407114945`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2048, 512]]}`
- `rotary_emb` -> 20781.268 ms
  纯GPU kernel时间: `0.027 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=20781.268 ms`, `host_to_first_kernel_gap=20781.116195`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.027 ms`, `gpu_kernel_sum=0.027 ms`
  开始时间(ns): `16407437739`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2048], [2048, 128, 64], [2048, 1, 64]]}`
- `kv_b_proj` -> 5769.623 ms
  纯GPU kernel时间: `0.074 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=5769.623 ms`, `host_to_first_kernel_gap=0.536011`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=5768.964 ms`, `gpu_kernel_sum=0.074 ms`
  开始时间(ns): `38509641499`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[2048, 512]]}`
- `attn_mha` -> 11.755 ms
  纯GPU kernel时间: `0.144 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=11.755 ms`, `host_to_first_kernel_gap=2.150344`, `host_end_to_last_kernel_tail=0.057 ms`, `gpu_makespan=9.661 ms`, `gpu_kernel_sum=0.144 ms`
  开始时间(ns): `44281044972`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2048, 128, 192], [2048, 128, 192], [2048, 128, 128]]}`
- `o_proj` -> 6481.642 ms
  纯GPU kernel时间: `0.380 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=6481.642 ms`, `host_to_first_kernel_gap=0.674564`, `host_end_to_last_kernel_tail=0.128 ms`, `gpu_makespan=6481.095 ms`, `gpu_kernel_sum=0.380 ms`
  开始时间(ns): `44292943436`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2048, 16384]]}`

## Layer 1 / prefill / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `5.748 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=5.748 ms`, `module_to_last_kernel=5.948 ms`, `host_to_first_kernel_gap=0.851109`, `host_end_to_last_kernel_tail=0.200 ms`, `gpu_makespan=5.097 ms`, `gpu_kernel_sum=0.826 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=2048`, `total_tokens=2048`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[2048, 128, 192], [2048, 128, 192], [2048, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=2048`, `sum_prefix=0`, `sum_seq_after=2048`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 512, 'prefix_len': 0, 'seq_len_after': 512, 'prompt_len': 512, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 512, 'prefix_len': 0, 'seq_len_after': 512, 'prompt_len': 512, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 512, 'prefix_len': 0, 'seq_len_after': 512, 'prompt_len': 512, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 512, 'prefix_len': 0, 'seq_len_after': 512, 'prompt_len': 512, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.952 ms
  纯GPU kernel时间: `0.075 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.952 ms`, `host_to_first_kernel_gap=0.381855`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.546 ms`, `gpu_kernel_sum=0.075 ms`
  开始时间(ns): `64066417541`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2048, 7168]]}`
- `q_a_layernorm` -> 0.159 ms
  纯GPU kernel时间: `0.006 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.159 ms`, `host_to_first_kernel_gap=0.137315`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.006 ms`, `gpu_kernel_sum=0.006 ms`
  开始时间(ns): `64067530305`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2048, 1536]]}`
- `q_b_proj` -> 0.677 ms
  纯GPU kernel时间: `0.123 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.677 ms`, `host_to_first_kernel_gap=0.229402`, `host_end_to_last_kernel_tail=0.048 ms`, `gpu_makespan=0.495 ms`, `gpu_kernel_sum=0.123 ms`
  开始时间(ns): `64067775529`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2048, 1536]]}`
- `kv_a_layernorm` -> 0.122 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.122 ms`, `host_to_first_kernel_gap=0.102675`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `64068643472`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2048, 512]]}`
- `rotary_emb` -> 0.224 ms
  纯GPU kernel时间: `0.027 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.224 ms`, `host_to_first_kernel_gap=0.186783`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.027 ms`, `gpu_kernel_sum=0.027 ms`
  开始时间(ns): `64068883300`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2048], [2048, 128, 64], [2048, 1, 64]]}`
- `kv_b_proj` -> 0.695 ms
  纯GPU kernel时间: `0.072 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.695 ms`, `host_to_first_kernel_gap=0.270945`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.421 ms`, `gpu_kernel_sum=0.072 ms`
  开始时间(ns): `64069522114`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[2048, 512]]}`
- `attn_mha` -> 0.407 ms
  纯GPU kernel时间: `0.142 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.407 ms`, `host_to_first_kernel_gap=0.304358`, `host_end_to_last_kernel_tail=0.088 ms`, `gpu_makespan=0.191 ms`, `gpu_kernel_sum=0.142 ms`
  开始时间(ns): `64070494653`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2048, 128, 192], [2048, 128, 192], [2048, 128, 128]]}`
- `o_proj` -> 0.626 ms
  纯GPU kernel时间: `0.377 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.626 ms`, `host_to_first_kernel_gap=0.227367`, `host_end_to_last_kernel_tail=0.257 ms`, `gpu_makespan=0.656 ms`, `gpu_kernel_sum=0.377 ms`
  开始时间(ns): `64071012795`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2048, 16384]]}`

## Layer 2 / prefill / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `4.576 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=4.576 ms`, `module_to_last_kernel=4.817 ms`, `host_to_first_kernel_gap=0.539003`, `host_end_to_last_kernel_tail=0.240 ms`, `gpu_makespan=4.278 ms`, `gpu_kernel_sum=0.826 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=2048`, `total_tokens=2048`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[2048, 128, 192], [2048, 128, 192], [2048, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=2048`, `sum_prefix=0`, `sum_seq_after=2048`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 512, 'prefix_len': 0, 'seq_len_after': 512, 'prompt_len': 512, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 512, 'prefix_len': 0, 'seq_len_after': 512, 'prompt_len': 512, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 512, 'prefix_len': 0, 'seq_len_after': 512, 'prompt_len': 512, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 512, 'prefix_len': 0, 'seq_len_after': 512, 'prompt_len': 512, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.587 ms
  纯GPU kernel时间: `0.074 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.587 ms`, `host_to_first_kernel_gap=0.21891`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.362 ms`, `gpu_kernel_sum=0.074 ms`
  开始时间(ns): `64074259043`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2048, 7168]]}`
- `q_a_layernorm` -> 0.113 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.113 ms`, `host_to_first_kernel_gap=0.095005`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `64074968548`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2048, 1536]]}`
- `q_b_proj` -> 0.537 ms
  纯GPU kernel时间: `0.123 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.537 ms`, `host_to_first_kernel_gap=0.192457`, `host_end_to_last_kernel_tail=0.061 ms`, `gpu_makespan=0.406 ms`, `gpu_kernel_sum=0.123 ms`
  开始时间(ns): `64075159832`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2048, 1536]]}`
- `kv_a_layernorm` -> 0.104 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.104 ms`, `host_to_first_kernel_gap=0.087201`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `64075853120`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2048, 512]]}`
- `rotary_emb` -> 0.184 ms
  纯GPU kernel时间: `0.027 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.184 ms`, `host_to_first_kernel_gap=0.151771`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.027 ms`, `gpu_kernel_sum=0.027 ms`
  开始时间(ns): `64076057510`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2048], [2048, 128, 64], [2048, 1, 64]]}`
- `kv_b_proj` -> 0.608 ms
  纯GPU kernel时间: `0.072 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.608 ms`, `host_to_first_kernel_gap=0.231955`, `host_end_to_last_kernel_tail=0.008 ms`, `gpu_makespan=0.384 ms`, `gpu_kernel_sum=0.072 ms`
  开始时间(ns): `64076605710`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[2048, 512]]}`
- `attn_mha` -> 0.343 ms
  纯GPU kernel时间: `0.142 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.343 ms`, `host_to_first_kernel_gap=0.249832`, `host_end_to_last_kernel_tail=0.090 ms`, `gpu_makespan=0.183 ms`, `gpu_kernel_sum=0.142 ms`
  开始时间(ns): `64077451448`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2048, 128, 192], [2048, 128, 192], [2048, 128, 128]]}`
- `o_proj` -> 0.585 ms
  纯GPU kernel时间: `0.378 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.585 ms`, `host_to_first_kernel_gap=0.222677`, `host_end_to_last_kernel_tail=0.275 ms`, `gpu_makespan=0.637 ms`, `gpu_kernel_sum=0.378 ms`
  开始时间(ns): `64077895755`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2048, 16384]]}`

## Layer 3 / prefill / instance 1

- 执行序号: `4`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `4.562 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=4.562 ms`, `module_to_last_kernel=4.801 ms`, `host_to_first_kernel_gap=0.550564`, `host_end_to_last_kernel_tail=0.239 ms`, `gpu_makespan=4.250 ms`, `gpu_kernel_sum=0.825 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=2048`, `total_tokens=2048`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[2048, 128, 192], [2048, 128, 192], [2048, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=2048`, `sum_prefix=0`, `sum_seq_after=2048`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 512, 'prefix_len': 0, 'seq_len_after': 512, 'prompt_len': 512, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 512, 'prefix_len': 0, 'seq_len_after': 512, 'prompt_len': 512, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 512, 'prefix_len': 0, 'seq_len_after': 512, 'prompt_len': 512, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 512, 'prefix_len': 0, 'seq_len_after': 512, 'prompt_len': 512, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.633 ms
  纯GPU kernel时间: `0.073 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.633 ms`, `host_to_first_kernel_gap=0.231963`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.376 ms`, `gpu_kernel_sum=0.073 ms`
  开始时间(ns): `64081020228`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2048, 7168]]}`
- `q_a_layernorm` -> 0.120 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.120 ms`, `host_to_first_kernel_gap=0.10067`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `64081782209`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2048, 1536]]}`
- `q_b_proj` -> 0.534 ms
  纯GPU kernel时间: `0.123 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.534 ms`, `host_to_first_kernel_gap=0.190041`, `host_end_to_last_kernel_tail=0.058 ms`, `gpu_makespan=0.402 ms`, `gpu_kernel_sum=0.123 ms`
  开始时间(ns): `64081981158`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2048, 1536]]}`
- `kv_a_layernorm` -> 0.100 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.100 ms`, `host_to_first_kernel_gap=0.083226`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `64082672965`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2048, 512]]}`
- `rotary_emb` -> 0.180 ms
  纯GPU kernel时间: `0.027 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.180 ms`, `host_to_first_kernel_gap=0.14777`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.027 ms`, `gpu_kernel_sum=0.027 ms`
  开始时间(ns): `64082880997`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2048], [2048, 128, 64], [2048, 1, 64]]}`
- `kv_b_proj` -> 0.571 ms
  纯GPU kernel时间: `0.073 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.571 ms`, `host_to_first_kernel_gap=0.201644`, `host_end_to_last_kernel_tail=0.010 ms`, `gpu_makespan=0.379 ms`, `gpu_kernel_sum=0.073 ms`
  开始时间(ns): `64083390674`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[2048, 512]]}`
- `attn_mha` -> 0.338 ms
  纯GPU kernel时间: `0.144 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.338 ms`, `host_to_first_kernel_gap=0.247039`, `host_end_to_last_kernel_tail=0.092 ms`, `gpu_makespan=0.183 ms`, `gpu_kernel_sum=0.144 ms`
  开始时间(ns): `64084229183`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2048, 128, 192], [2048, 128, 192], [2048, 128, 128]]}`
- `o_proj` -> 0.550 ms
  纯GPU kernel时间: `0.376 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.550 ms`, `host_to_first_kernel_gap=0.205288`, `host_end_to_last_kernel_tail=0.274 ms`, `gpu_makespan=0.618 ms`, `gpu_kernel_sum=0.376 ms`
  开始时间(ns): `64084679126`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2048, 16384]]}`

## Layer 4 / prefill / instance 1

- 执行序号: `5`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `1.887 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.887 ms`, `module_to_last_kernel=2.191 ms`, `host_to_first_kernel_gap=0.33624`, `host_end_to_last_kernel_tail=0.304 ms`, `gpu_makespan=1.855 ms`, `gpu_kernel_sum=0.823 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=2048`, `total_tokens=2048`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[2048, 128, 192], [2048, 128, 192], [2048, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=2048`, `sum_prefix=0`, `sum_seq_after=2048`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 512, 'prefix_len': 0, 'seq_len_after': 512, 'prompt_len': 512, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 512, 'prefix_len': 0, 'seq_len_after': 512, 'prompt_len': 512, 'is_chunked_req': 0}, {'req_id': 2, 'extend_len': 512, 'prefix_len': 0, 'seq_len_after': 512, 'prompt_len': 512, 'is_chunked_req': 0}, {'req_id': 3, 'extend_len': 512, 'prefix_len': 0, 'seq_len_after': 512, 'prompt_len': 512, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.376 ms
  纯GPU kernel时间: `0.073 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.376 ms`, `host_to_first_kernel_gap=0.141131`, `host_end_to_last_kernel_tail=0.023 ms`, `gpu_makespan=0.258 ms`, `gpu_kernel_sum=0.073 ms`
  开始时间(ns): `66569643730`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2048, 7168]]}`
- `q_a_layernorm` -> 0.052 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.052 ms`, `host_to_first_kernel_gap=0.048164`, `host_end_to_last_kernel_tail=0.001 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `66570070361`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2048, 1536]]}`
- `q_b_proj` -> 0.205 ms
  纯GPU kernel时间: `0.123 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.205 ms`, `host_to_first_kernel_gap=0.066925`, `host_end_to_last_kernel_tail=0.101 ms`, `gpu_makespan=0.239 ms`, `gpu_kernel_sum=0.123 ms`
  开始时间(ns): `66570147824`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2048, 1536]]}`
- `kv_a_layernorm` -> 0.030 ms
  纯GPU kernel时间: `0.005 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.030 ms`, `host_to_first_kernel_gap=0.055296`, `host_end_to_last_kernel_tail=0.030 ms`, `gpu_makespan=0.005 ms`, `gpu_kernel_sum=0.005 ms`
  开始时间(ns): `66570397853`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2048, 512]]}`
- `rotary_emb` -> 0.096 ms
  纯GPU kernel时间: `0.027 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.096 ms`, `host_to_first_kernel_gap=0.085396`, `host_end_to_last_kernel_tail=0.016 ms`, `gpu_makespan=0.027 ms`, `gpu_kernel_sum=0.027 ms`
  开始时间(ns): `66570461033`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2048], [2048, 128, 64], [2048, 1, 64]]}`
- `kv_b_proj` -> 0.180 ms
  纯GPU kernel时间: `0.072 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.180 ms`, `host_to_first_kernel_gap=0.06859`, `host_end_to_last_kernel_tail=0.054 ms`, `gpu_makespan=0.165 ms`, `gpu_kernel_sum=0.072 ms`
  开始时间(ns): `66570700783`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[2048, 512]]}`
- `attn_mha` -> 0.151 ms
  纯GPU kernel时间: `0.144 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.151 ms`, `host_to_first_kernel_gap=0.116522`, `host_end_to_last_kernel_tail=0.127 ms`, `gpu_makespan=0.161 ms`, `gpu_kernel_sum=0.144 ms`
  开始时间(ns): `66570960243`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2048, 128, 192], [2048, 128, 192], [2048, 128, 128]]}`
- `o_proj` -> 0.177 ms
  纯GPU kernel时间: `0.374 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.177 ms`, `host_to_first_kernel_gap=0.092352`, `host_end_to_last_kernel_tail=0.315 ms`, `gpu_makespan=0.401 ms`, `gpu_kernel_sum=0.374 ms`
  开始时间(ns): `66571147101`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2048, 16384]]}`

## Layer 0 / decode / instance 1

- 执行序号: `6`
- Collector对齐边界: `{'Module': 'model.model.layers.0.self_attn'}`
- 整块 MLA-module 时长: `17309.413 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=17309.413 ms`, `module_to_last_kernel=17309.413 ms`, `host_to_first_kernel_gap=0.219302`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=17308.915 ms`, `gpu_kernel_sum=0.113 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 5362.292 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=5362.292 ms`, `host_to_first_kernel_gap=0.135234`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=5361.985 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `68910099400`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4, 7168]]}`
- `q_a_layernorm` -> 0.142 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.142 ms`, `host_to_first_kernel_gap=0.130899`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `74272604434`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4, 1536]]}`
- `kv_a_layernorm` -> 0.037 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.037 ms`, `host_to_first_kernel_gap=0.033091`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `74272779426`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4, 512]]}`
- `q_b_proj` -> 6781.784 ms
  纯GPU kernel时间: `0.020 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=6781.784 ms`, `host_to_first_kernel_gap=0.188801`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=6781.340 ms`, `gpu_kernel_sum=0.020 ms`
  开始时间(ns): `74272876388`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4, 1536]]}`
- `rotary_emb` -> 0.354 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.354 ms`, `host_to_first_kernel_gap=0.266712`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `81064103098`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4], [4, 128, 64], [4, 1, 64]]}`
- `attn_mqa` -> 28.932 ms
  纯GPU kernel时间: `0.023 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=28.932 ms`, `host_to_first_kernel_gap=0.321382`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=28.447 ms`, `gpu_kernel_sum=0.023 ms`
  开始时间(ns): `81064617259`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4, 128, 512], [4, 1, 512], [4, 1, 512]]}`
- `o_proj` -> 5119.007 ms
  纯GPU kernel时间: `0.049 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=5119.007 ms`, `host_to_first_kernel_gap=0.360696`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=5118.427 ms`, `gpu_kernel_sum=0.049 ms`
  开始时间(ns): `81100361807`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4, 16384]]}`

## Layer 1 / decode / instance 1

- 执行序号: `7`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `3.641 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=3.641 ms`, `module_to_last_kernel=3.641 ms`, `host_to_first_kernel_gap=0.433652`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=3.184 ms`, `gpu_kernel_sum=0.112 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.659 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.659 ms`, `host_to_first_kernel_gap=0.271838`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.347 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `97531628604`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4, 7168]]}`
- `q_a_layernorm` -> 0.105 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.105 ms`, `host_to_first_kernel_gap=0.090574`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `97532432172`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4, 1536]]}`
- `kv_a_layernorm` -> 0.067 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.067 ms`, `host_to_first_kernel_gap=0.057546`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `97532583664`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4, 512]]}`
- `q_b_proj` -> 0.434 ms
  纯GPU kernel时间: `0.020 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.434 ms`, `host_to_first_kernel_gap=0.138508`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.267 ms`, `gpu_kernel_sum=0.020 ms`
  开始时间(ns): `97532714350`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4, 1536]]}`
- `rotary_emb` -> 0.215 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.215 ms`, `host_to_first_kernel_gap=0.159752`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `97533476210`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4], [4, 128, 64], [4, 1, 64]]}`
- `attn_mqa` -> 0.585 ms
  纯GPU kernel时间: `0.023 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.585 ms`, `host_to_first_kernel_gap=0.182736`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.351 ms`, `gpu_kernel_sum=0.023 ms`
  开始时间(ns): `97533783914`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4, 128, 512], [4, 1, 512], [4, 1, 512]]}`
- `o_proj` -> 0.492 ms
  纯GPU kernel时间: `0.048 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.492 ms`, `host_to_first_kernel_gap=0.193581`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.299 ms`, `gpu_kernel_sum=0.048 ms`
  开始时间(ns): `97534591820`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4, 16384]]}`

## Layer 2 / decode / instance 1

- 执行序号: `8`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `2.812 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.812 ms`, `module_to_last_kernel=2.812 ms`, `host_to_first_kernel_gap=0.26027`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=2.536 ms`, `gpu_kernel_sum=0.111 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.387 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.387 ms`, `host_to_first_kernel_gap=0.147964`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.211 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `97536778589`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4, 7168]]}`
- `q_a_layernorm` -> 0.078 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.078 ms`, `host_to_first_kernel_gap=0.066771`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `97537274022`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4, 1536]]}`
- `kv_a_layernorm` -> 0.053 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.053 ms`, `host_to_first_kernel_gap=0.045296`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `97537395913`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4, 512]]}`
- `q_b_proj` -> 0.374 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.374 ms`, `host_to_first_kernel_gap=0.10885`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.241 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `97537504583`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4, 1536]]}`
- `rotary_emb` -> 0.118 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.118 ms`, `host_to_first_kernel_gap=0.103907`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `97538118998`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4], [4, 128, 64], [4, 1, 64]]}`
- `attn_mqa` -> 0.497 ms
  纯GPU kernel时间: `0.022 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.497 ms`, `host_to_first_kernel_gap=0.144061`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.309 ms`, `gpu_kernel_sum=0.022 ms`
  开始时间(ns): `97538307644`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4, 128, 512], [4, 1, 512], [4, 1, 512]]}`
- `o_proj` -> 0.446 ms
  纯GPU kernel时间: `0.049 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.446 ms`, `host_to_first_kernel_gap=0.166519`, `host_end_to_last_kernel_tail=0.006 ms`, `gpu_makespan=0.285 ms`, `gpu_kernel_sum=0.049 ms`
  开始时间(ns): `97539011489`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4, 16384]]}`

## Layer 3 / decode / instance 1

- 执行序号: `9`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `2.687 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.687 ms`, `module_to_last_kernel=2.687 ms`, `host_to_first_kernel_gap=0.247586`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=2.424 ms`, `gpu_kernel_sum=0.110 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.391 ms
  纯GPU kernel时间: `0.016 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.391 ms`, `host_to_first_kernel_gap=0.147778`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.217 ms`, `gpu_kernel_sum=0.016 ms`
  开始时间(ns): `97541025014`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4, 7168]]}`
- `q_a_layernorm` -> 0.078 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.078 ms`, `host_to_first_kernel_gap=0.067016`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `97541519728`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4, 1536]]}`
- `kv_a_layernorm` -> 0.054 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.054 ms`, `host_to_first_kernel_gap=0.044436`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `97541649956`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4, 512]]}`
- `q_b_proj` -> 0.381 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.381 ms`, `host_to_first_kernel_gap=0.120993`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.236 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `97541761623`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4, 1536]]}`
- `rotary_emb` -> 0.107 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.107 ms`, `host_to_first_kernel_gap=0.088345`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `97542366655`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4], [4, 128, 64], [4, 1, 64]]}`
- `attn_mqa` -> 0.430 ms
  纯GPU kernel时间: `0.023 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.430 ms`, `host_to_first_kernel_gap=0.140554`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.257 ms`, `gpu_kernel_sum=0.023 ms`
  开始时间(ns): `97542544270`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4, 128, 512], [4, 1, 512], [4, 1, 512]]}`
- `o_proj` -> 0.406 ms
  纯GPU kernel时间: `0.047 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.406 ms`, `host_to_first_kernel_gap=0.15461`, `host_end_to_last_kernel_tail=0.010 ms`, `gpu_makespan=0.262 ms`, `gpu_kernel_sum=0.047 ms`
  开始时间(ns): `97543179909`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4, 16384]]}`

## Layer 4 / decode / instance 1

- 执行序号: `10`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `2.513 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.513 ms`, `module_to_last_kernel=2.516 ms`, `host_to_first_kernel_gap=0.289637`, `host_end_to_last_kernel_tail=0.003 ms`, `gpu_makespan=2.227 ms`, `gpu_kernel_sum=0.115 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.467 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.467 ms`, `host_to_first_kernel_gap=0.175014`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.265 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `100925563596`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4, 7168]]}`
- `q_a_layernorm` -> 0.065 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.065 ms`, `host_to_first_kernel_gap=0.058737`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `100926133793`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4, 1536]]}`
- `kv_a_layernorm` -> 0.037 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.037 ms`, `host_to_first_kernel_gap=0.032085`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `100926222013`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4, 512]]}`
- `q_b_proj` -> 0.304 ms
  纯GPU kernel时间: `0.023 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.304 ms`, `host_to_first_kernel_gap=0.119036`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.180 ms`, `gpu_kernel_sum=0.023 ms`
  开始时间(ns): `100926310998`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4, 1536]]}`
- `rotary_emb` -> 0.124 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.124 ms`, `host_to_first_kernel_gap=0.107532`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `100926867686`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4], [4, 128, 64], [4, 1, 64]]}`
- `attn_mqa` -> 0.414 ms
  纯GPU kernel时间: `0.023 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.414 ms`, `host_to_first_kernel_gap=0.11668`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.264 ms`, `gpu_kernel_sum=0.023 ms`
  开始时间(ns): `100927047242`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4, 128, 512], [4, 1, 512], [4, 1, 512]]}`
- `o_proj` -> 0.319 ms
  纯GPU kernel时间: `0.048 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.319 ms`, `host_to_first_kernel_gap=0.124854`, `host_end_to_last_kernel_tail=0.019 ms`, `gpu_makespan=0.213 ms`, `gpu_kernel_sum=0.048 ms`
  开始时间(ns): `100927627516`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4, 16384]]}`
