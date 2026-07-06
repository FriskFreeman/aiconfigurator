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
- 整块 MLA-module 时长: `2.745 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.745 ms`, `module_to_last_kernel=12.292 ms`, `host_to_first_kernel_gap=0.317546`, `host_end_to_last_kernel_tail=9.547 ms`, `gpu_makespan=11.974 ms`, `gpu_kernel_sum=9.615 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4096`, `total_tokens=36864`, `chunked_req_prefix_len=32768`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[4096, 128, 192], [36864, 128, 192], [36864, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=4096`, `sum_prefix=32768`, `sum_seq_after=36864`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 2048, 'prefix_len': 16384, 'seq_len_after': 18432, 'prompt_len': 18432, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 2048, 'prefix_len': 16384, 'seq_len_after': 18432, 'prompt_len': 18432, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.431 ms
  纯GPU kernel时间: `0.126 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.431 ms`, `host_to_first_kernel_gap=0.158144`, `host_end_to_last_kernel_tail=0.049 ms`, `gpu_makespan=0.321 ms`, `gpu_kernel_sum=0.126 ms`
  开始时间(ns): `34040816`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4096, 7168]]}`
- `q_a_layernorm` -> 0.060 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.060 ms`, `host_to_first_kernel_gap=0.053776`, `host_end_to_last_kernel_tail=0.003 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `34533439`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4096, 1536]]}`
- `q_b_proj` -> 0.269 ms
  纯GPU kernel时间: `0.241 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.269 ms`, `host_to_first_kernel_gap=0.101142`, `host_end_to_last_kernel_tail=0.202 ms`, `gpu_makespan=0.370 ms`, `gpu_kernel_sum=0.241 ms`
  开始时间(ns): `34627001`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4096, 1536]]}`
- `kv_a_layernorm` -> 0.047 ms
  纯GPU kernel时间: `0.008 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.047 ms`, `host_to_first_kernel_gap=0.136054`, `host_end_to_last_kernel_tail=0.097 ms`, `gpu_makespan=0.008 ms`, `gpu_kernel_sum=0.008 ms`
  开始时间(ns): `34961720`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4096, 512]]}`
- `rotary_emb` -> 0.117 ms
  纯GPU kernel时间: `0.053 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.117 ms`, `host_to_first_kernel_gap=0.102481`, `host_end_to_last_kernel_tail=0.038 ms`, `gpu_makespan=0.053 ms`, `gpu_kernel_sum=0.053 ms`
  开始时间(ns): `35053437`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4096], [4096, 128, 64], [4096, 1, 64]]}`
- `kv_b_proj` -> 0.283 ms
  纯GPU kernel时间: `1.265 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.283 ms`, `host_to_first_kernel_gap=0.100401`, `host_end_to_last_kernel_tail=1.214 ms`, `gpu_makespan=1.397 ms`, `gpu_kernel_sum=1.265 ms`
  开始时间(ns): `35692572`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[36864, 512]]}`
- `attn_mha` -> 0.198 ms
  纯GPU kernel时间: `7.281 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.198 ms`, `host_to_first_kernel_gap=2.162628`, `host_end_to_last_kernel_tail=9.246 ms`, `gpu_makespan=7.281 ms`, `gpu_kernel_sum=7.281 ms`
  开始时间(ns): `36094214`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4096, 128, 192], [36864, 128, 192], [36864, 128, 128]]}`
- `o_proj` -> 0.269 ms
  纯GPU kernel时间: `0.632 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.269 ms`, `host_to_first_kernel_gap=9.19886`, `host_end_to_last_kernel_tail=9.563 ms`, `gpu_makespan=0.634 ms`, `gpu_kernel_sum=0.632 ms`
  开始时间(ns): `36340563`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4096, 16384]]}`

## Layer 1 / prefill / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `2.055 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.055 ms`, `module_to_last_kernel=21.863 ms`, `host_to_first_kernel_gap=10.967208`, `host_end_to_last_kernel_tail=19.807 ms`, `gpu_makespan=10.895 ms`, `gpu_kernel_sum=9.759 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4096`, `total_tokens=36864`, `chunked_req_prefix_len=32768`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[4096, 128, 192], [36864, 128, 192], [36864, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=4096`, `sum_prefix=32768`, `sum_seq_after=36864`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 2048, 'prefix_len': 16384, 'seq_len_after': 18432, 'prompt_len': 18432, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 2048, 'prefix_len': 16384, 'seq_len_after': 18432, 'prompt_len': 18432, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.233 ms
  纯GPU kernel时间: `0.125 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.233 ms`, `host_to_first_kernel_gap=10.822807`, `host_end_to_last_kernel_tail=10.717 ms`, `gpu_makespan=0.128 ms`, `gpu_kernel_sum=0.125 ms`
  开始时间(ns): `37682084`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4096, 7168]]}`
- `q_a_layernorm` -> 0.061 ms
  纯GPU kernel时间: `0.010 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.061 ms`, `host_to_first_kernel_gap=10.662592`, `host_end_to_last_kernel_tail=10.612 ms`, `gpu_makespan=0.010 ms`, `gpu_kernel_sum=0.010 ms`
  开始时间(ns): `37970139`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4096, 1536]]}`
- `q_b_proj` -> 0.228 ms
  纯GPU kernel时间: `0.236 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.228 ms`, `host_to_first_kernel_gap=10.582453`, `host_end_to_last_kernel_tail=10.593 ms`, `gpu_makespan=0.238 ms`, `gpu_kernel_sum=0.236 ms`
  开始时间(ns): `38061222`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4096, 1536]]}`
- `kv_a_layernorm` -> 0.043 ms
  纯GPU kernel时间: `0.008 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.043 ms`, `host_to_first_kernel_gap=10.530593`, `host_end_to_last_kernel_tail=10.496 ms`, `gpu_makespan=0.008 ms`, `gpu_kernel_sum=0.008 ms`
  开始时间(ns): `38350777`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4096, 512]]}`
- `rotary_emb` -> 0.094 ms
  纯GPU kernel时间: `0.053 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.094 ms`, `host_to_first_kernel_gap=10.456035`, `host_end_to_last_kernel_tail=10.416 ms`, `gpu_makespan=0.053 ms`, `gpu_kernel_sum=0.053 ms`
  开始时间(ns): `38434903`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4096], [4096, 128, 64], [4096, 1, 64]]}`
- `kv_b_proj` -> 0.223 ms
  纯GPU kernel时间: `1.399 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.223 ms`, `host_to_first_kernel_gap=10.220749`, `host_end_to_last_kernel_tail=11.398 ms`, `gpu_makespan=1.400 ms`, `gpu_kernel_sum=1.399 ms`
  开始时间(ns): `38782157`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[36864, 512]]}`
- `attn_mha` -> 0.186 ms
  纯GPU kernel时间: `7.294 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.186 ms`, `host_to_first_kernel_gap=12.360655`, `host_end_to_last_kernel_tail=19.469 ms`, `gpu_makespan=7.294 ms`, `gpu_kernel_sum=7.294 ms`
  开始时间(ns): `39109928`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4096, 128, 192], [36864, 128, 192], [36864, 128, 128]]}`
- `o_proj` -> 0.237 ms
  纯GPU kernel时间: `0.633 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.237 ms`, `host_to_first_kernel_gap=19.42663`, `host_end_to_last_kernel_tail=19.824 ms`, `gpu_makespan=0.634 ms`, `gpu_kernel_sum=0.633 ms`
  开始时间(ns): `39339078`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4096, 16384]]}`

## Layer 2 / prefill / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `2.059 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.059 ms`, `module_to_last_kernel=32.101 ms`, `host_to_first_kernel_gap=21.235178`, `host_end_to_last_kernel_tail=30.042 ms`, `gpu_makespan=10.866 ms`, `gpu_kernel_sum=9.729 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4096`, `total_tokens=36864`, `chunked_req_prefix_len=32768`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[4096, 128, 192], [36864, 128, 192], [36864, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=4096`, `sum_prefix=32768`, `sum_seq_after=36864`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 2048, 'prefix_len': 16384, 'seq_len_after': 18432, 'prompt_len': 18432, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 2048, 'prefix_len': 16384, 'seq_len_after': 18432, 'prompt_len': 18432, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.234 ms
  纯GPU kernel时间: `0.125 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.234 ms`, `host_to_first_kernel_gap=21.090036`, `host_end_to_last_kernel_tail=20.983 ms`, `gpu_makespan=0.127 ms`, `gpu_kernel_sum=0.125 ms`
  开始时间(ns): `40639668`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4096, 7168]]}`
- `q_a_layernorm` -> 0.052 ms
  纯GPU kernel时间: `0.010 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.052 ms`, `host_to_first_kernel_gap=20.923695`, `host_end_to_last_kernel_tail=20.881 ms`, `gpu_makespan=0.010 ms`, `gpu_kernel_sum=0.010 ms`
  开始时间(ns): `40933656`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4096, 1536]]}`
- `q_b_proj` -> 0.221 ms
  纯GPU kernel时间: `0.237 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.221 ms`, `host_to_first_kernel_gap=20.852898`, `host_end_to_last_kernel_tail=20.870 ms`, `gpu_makespan=0.238 ms`, `gpu_kernel_sum=0.237 ms`
  开始时间(ns): `41015301`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4096, 1536]]}`
- `kv_a_layernorm` -> 0.048 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.048 ms`, `host_to_first_kernel_gap=20.809583`, `host_end_to_last_kernel_tail=20.771 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `41296312`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4096, 512]]}`
- `rotary_emb` -> 0.091 ms
  纯GPU kernel时间: `0.054 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.091 ms`, `host_to_first_kernel_gap=20.730569`, `host_end_to_last_kernel_tail=20.693 ms`, `gpu_makespan=0.054 ms`, `gpu_kernel_sum=0.054 ms`
  开始时间(ns): `41386526`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4096], [4096, 128, 64], [4096, 1, 64]]}`
- `kv_b_proj` -> 0.238 ms
  纯GPU kernel时间: `1.366 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.238 ms`, `host_to_first_kernel_gap=20.504367`, `host_end_to_last_kernel_tail=21.635 ms`, `gpu_makespan=1.369 ms`, `gpu_kernel_sum=1.366 ms`
  开始时间(ns): `41722680`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[36864, 512]]}`
- `attn_mha` -> 0.177 ms
  纯GPU kernel时间: `7.296 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.177 ms`, `host_to_first_kernel_gap=22.586542`, `host_end_to_last_kernel_tail=29.705 ms`, `gpu_makespan=7.296 ms`, `gpu_kernel_sum=7.296 ms`
  开始时间(ns): `42074709`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4096, 128, 192], [36864, 128, 192], [36864, 128, 128]]}`
- `o_proj` -> 0.240 ms
  纯GPU kernel时间: `0.634 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.240 ms`, `host_to_first_kernel_gap=29.662772`, `host_end_to_last_kernel_tail=30.059 ms`, `gpu_makespan=0.636 ms`, `gpu_kernel_sum=0.634 ms`
  开始时间(ns): `42296869`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4096, 16384]]}`

## Layer 3 / prefill / instance 1

- 执行序号: `4`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `2.038 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.038 ms`, `module_to_last_kernel=42.367 ms`, `host_to_first_kernel_gap=31.528765`, `host_end_to_last_kernel_tail=40.329 ms`, `gpu_makespan=10.838 ms`, `gpu_kernel_sum=9.698 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4096`, `total_tokens=36864`, `chunked_req_prefix_len=32768`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[4096, 128, 192], [36864, 128, 192], [36864, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=4096`, `sum_prefix=32768`, `sum_seq_after=36864`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 2048, 'prefix_len': 16384, 'seq_len_after': 18432, 'prompt_len': 18432, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 2048, 'prefix_len': 16384, 'seq_len_after': 18432, 'prompt_len': 18432, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.230 ms
  纯GPU kernel时间: `0.125 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.230 ms`, `host_to_first_kernel_gap=31.391779`, `host_end_to_last_kernel_tail=31.288 ms`, `gpu_makespan=0.126 ms`, `gpu_kernel_sum=0.125 ms`
  开始时间(ns): `43534417`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4096, 7168]]}`
- `q_a_layernorm` -> 0.052 ms
  纯GPU kernel时间: `0.010 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.052 ms`, `host_to_first_kernel_gap=31.234292`, `host_end_to_last_kernel_tail=31.192 ms`, `gpu_makespan=0.010 ms`, `gpu_kernel_sum=0.010 ms`
  开始时间(ns): `43818112`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4096, 1536]]}`
- `q_b_proj` -> 0.201 ms
  纯GPU kernel时间: `0.236 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.201 ms`, `host_to_first_kernel_gap=31.165531`, `host_end_to_last_kernel_tail=31.202 ms`, `gpu_makespan=0.237 ms`, `gpu_kernel_sum=0.236 ms`
  开始时间(ns): `43899161`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4096, 1536]]}`
- `kv_a_layernorm` -> 0.044 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.044 ms`, `host_to_first_kernel_gap=31.128739`, `host_end_to_last_kernel_tail=31.093 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `44173265`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4096, 512]]}`
- `rotary_emb` -> 0.091 ms
  纯GPU kernel时间: `0.053 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.091 ms`, `host_to_first_kernel_gap=31.05387`, `host_end_to_last_kernel_tail=31.016 ms`, `gpu_makespan=0.053 ms`, `gpu_kernel_sum=0.053 ms`
  开始时间(ns): `44259366`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4096], [4096, 128, 64], [4096, 1, 64]]}`
- `kv_b_proj` -> 0.253 ms
  纯GPU kernel时间: `1.337 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.253 ms`, `host_to_first_kernel_gap=30.825552`, `host_end_to_last_kernel_tail=31.912 ms`, `gpu_makespan=1.339 ms`, `gpu_kernel_sum=1.337 ms`
  开始时间(ns): `44598596`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[36864, 512]]}`
- `attn_mha` -> 0.162 ms
  纯GPU kernel时间: `7.295 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.162 ms`, `host_to_first_kernel_gap=32.873384`, `host_end_to_last_kernel_tail=40.006 ms`, `gpu_makespan=7.295 ms`, `gpu_kernel_sum=7.295 ms`
  开始时间(ns): `44958440`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4096, 128, 192], [36864, 128, 192], [36864, 128, 128]]}`
- `o_proj` -> 0.245 ms
  纯GPU kernel时间: `0.633 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.245 ms`, `host_to_first_kernel_gap=39.95465`, `host_end_to_last_kernel_tail=40.345 ms`, `gpu_makespan=0.635 ms`, `gpu_kernel_sum=0.633 ms`
  开始时间(ns): `45174540`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4096, 16384]]}`

## Layer 4 / prefill / instance 1

- 执行序号: `5`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `2.213 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.213 ms`, `module_to_last_kernel=55.204 ms`, `host_to_first_kernel_gap=43.943074`, `host_end_to_last_kernel_tail=52.991 ms`, `gpu_makespan=11.261 ms`, `gpu_kernel_sum=10.127 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=4096`, `total_tokens=36864`, `chunked_req_prefix_len=32768`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[4096, 128, 192], [36864, 128, 192], [36864, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=4096`, `sum_prefix=32768`, `sum_seq_after=36864`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 2048, 'prefix_len': 16384, 'seq_len_after': 18432, 'prompt_len': 18432, 'is_chunked_req': 0}, {'req_id': 1, 'extend_len': 2048, 'prefix_len': 16384, 'seq_len_after': 18432, 'prompt_len': 18432, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.332 ms
  纯GPU kernel时间: `0.136 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.332 ms`, `host_to_first_kernel_gap=43.766152`, `host_end_to_last_kernel_tail=43.572 ms`, `gpu_makespan=0.139 ms`, `gpu_kernel_sum=0.136 ms`
  开始时间(ns): `47190677`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[4096, 7168]]}`
- `q_a_layernorm` -> 0.059 ms
  纯GPU kernel时间: `0.010 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.059 ms`, `host_to_first_kernel_gap=43.514256`, `host_end_to_last_kernel_tail=43.465 ms`, `gpu_makespan=0.010 ms`, `gpu_kernel_sum=0.010 ms`
  开始时间(ns): `47581773`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[4096, 1536]]}`
- `q_b_proj` -> 0.230 ms
  纯GPU kernel时间: `0.261 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.230 ms`, `host_to_first_kernel_gap=43.433713`, `host_end_to_last_kernel_tail=43.466 ms`, `gpu_makespan=0.262 ms`, `gpu_kernel_sum=0.261 ms`
  开始时间(ns): `47673164`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[4096, 1536]]}`
- `kv_a_layernorm` -> 0.046 ms
  纯GPU kernel时间: `0.009 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.046 ms`, `host_to_first_kernel_gap=43.402976`, `host_end_to_last_kernel_tail=43.366 ms`, `gpu_makespan=0.009 ms`, `gpu_kernel_sum=0.009 ms`
  开始时间(ns): `47965660`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[4096, 512]]}`
- `rotary_emb` -> 0.115 ms
  纯GPU kernel时间: `0.054 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.115 ms`, `host_to_first_kernel_gap=43.326077`, `host_end_to_last_kernel_tail=43.264 ms`, `gpu_makespan=0.054 ms`, `gpu_kernel_sum=0.054 ms`
  开始时间(ns): `48052735`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[4096], [4096, 128, 64], [4096, 1, 64]]}`
- `kv_b_proj` -> 0.235 ms
  纯GPU kernel时间: `1.360 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.235 ms`, `host_to_first_kernel_gap=43.081729`, `host_end_to_last_kernel_tail=44.209 ms`, `gpu_makespan=1.362 ms`, `gpu_kernel_sum=1.360 ms`
  开始时间(ns): `48406683`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[36864, 512]]}`
- `attn_mha` -> 0.170 ms
  纯GPU kernel时间: `7.576 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.170 ms`, `host_to_first_kernel_gap=45.170139`, `host_end_to_last_kernel_tail=52.577 ms`, `gpu_makespan=7.576 ms`, `gpu_kernel_sum=7.576 ms`
  开始时间(ns): `48747294`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[4096, 128, 192], [36864, 128, 192], [36864, 128, 128]]}`
- `o_proj` -> 0.251 ms
  纯GPU kernel时间: `0.722 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.251 ms`, `host_to_first_kernel_gap=52.53456`, `host_end_to_last_kernel_tail=53.006 ms`, `gpu_makespan=0.723 ms`, `gpu_kernel_sum=0.722 ms`
  开始时间(ns): `48960302`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[4096, 16384]]}`

## Layer 0 / decode / instance 1

- 执行序号: `6`
- Collector对齐边界: `{'Module': 'model.model.layers.0.self_attn'}`
- 整块 MLA-module 时长: `2.222 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.222 ms`, `module_to_last_kernel=54.599 ms`, `host_to_first_kernel_gap=54.427798`, `host_end_to_last_kernel_tail=52.377 ms`, `gpu_makespan=0.171 ms`, `gpu_kernel_sum=0.140 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.395 ms
  纯GPU kernel时间: `0.016 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.395 ms`, `host_to_first_kernel_gap=54.340589`, `host_end_to_last_kernel_tail=53.963 ms`, `gpu_makespan=0.017 ms`, `gpu_kernel_sum=0.016 ms`
  开始时间(ns): `53944311`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2, 7168]]}`
- `q_a_layernorm` -> 0.061 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.061 ms`, `host_to_first_kernel_gap=53.889092`, `host_end_to_last_kernel_tail=53.831 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `54412224`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2, 1536]]}`
- `kv_a_layernorm` -> 0.033 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.033 ms`, `host_to_first_kernel_gap=53.805916`, `host_end_to_last_kernel_tail=53.775 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `54497576`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2, 512]]}`
- `q_b_proj` -> 0.257 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.257 ms`, `host_to_first_kernel_gap=53.736741`, `host_end_to_last_kernel_tail=53.500 ms`, `gpu_makespan=0.021 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `54569727`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2, 1536]]}`
- `rotary_emb` -> 0.118 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.118 ms`, `host_to_first_kernel_gap=53.302139`, `host_end_to_last_kernel_tail=53.187 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `55037161`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2], [2, 128, 64], [2, 1, 64]]}`
- `attn_mqa` -> 0.365 ms
  纯GPU kernel时间: `0.047 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.365 ms`, `host_to_first_kernel_gap=53.135234`, `host_end_to_last_kernel_tail=52.818 ms`, `gpu_makespan=0.048 ms`, `gpu_kernel_sum=0.047 ms`
  开始时间(ns): `55207042`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2, 128, 512], [2, 1, 512], [2, 1, 512]]}`
- `o_proj` -> 0.325 ms
  纯GPU kernel时间: `0.052 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.325 ms`, `host_to_first_kernel_gap=52.670048`, `host_end_to_last_kernel_tail=52.398 ms`, `gpu_makespan=0.053 ms`, `gpu_kernel_sum=0.052 ms`
  开始时间(ns): `55732836`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2, 16384]]}`

## Layer 1 / decode / instance 1

- 执行序号: `7`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `1.855 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.855 ms`, `module_to_last_kernel=51.787 ms`, `host_to_first_kernel_gap=51.617716`, `host_end_to_last_kernel_tail=49.932 ms`, `gpu_makespan=0.169 ms`, `gpu_kernel_sum=0.139 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.237 ms
  纯GPU kernel时间: `0.016 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.237 ms`, `host_to_first_kernel_gap=51.541908`, `host_end_to_last_kernel_tail=51.321 ms`, `gpu_makespan=0.016 ms`, `gpu_kernel_sum=0.016 ms`
  开始时间(ns): `57079087`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2, 7168]]}`
- `q_a_layernorm` -> 0.053 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.053 ms`, `host_to_first_kernel_gap=51.256942`, `host_end_to_last_kernel_tail=51.206 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `57380373`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2, 1536]]}`
- `kv_a_layernorm` -> 0.032 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.032 ms`, `host_to_first_kernel_gap=51.18241`, `host_end_to_last_kernel_tail=51.152 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `57457113`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2, 512]]}`
- `q_b_proj` -> 0.212 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.212 ms`, `host_to_first_kernel_gap=51.118475`, `host_end_to_last_kernel_tail=50.925 ms`, `gpu_makespan=0.019 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `57524184`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2, 1536]]}`
- `rotary_emb` -> 0.096 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.096 ms`, `host_to_first_kernel_gap=50.754932`, `host_end_to_last_kernel_tail=50.661 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `57918863`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2], [2, 128, 64], [2, 1, 64]]}`
- `attn_mqa` -> 0.350 ms
  纯GPU kernel时间: `0.047 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.350 ms`, `host_to_first_kernel_gap=50.613482`, `host_end_to_last_kernel_tail=50.312 ms`, `gpu_makespan=0.048 ms`, `gpu_kernel_sum=0.047 ms`
  开始时间(ns): `58063097`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2, 128, 512], [2, 1, 512], [2, 1, 512]]}`
- `o_proj` -> 0.279 ms
  纯GPU kernel时间: `0.053 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.279 ms`, `host_to_first_kernel_gap=50.176861`, `host_end_to_last_kernel_tail=49.952 ms`, `gpu_makespan=0.054 ms`, `gpu_kernel_sum=0.053 ms`
  开始时间(ns): `58559462`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2, 16384]]}`

## Layer 2 / decode / instance 1

- 执行序号: `8`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `1.830 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.830 ms`, `module_to_last_kernel=49.386 ms`, `host_to_first_kernel_gap=49.216861`, `host_end_to_last_kernel_tail=47.556 ms`, `gpu_makespan=0.169 ms`, `gpu_kernel_sum=0.138 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.214 ms
  纯GPU kernel时间: `0.016 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.214 ms`, `host_to_first_kernel_gap=49.141801`, `host_end_to_last_kernel_tail=48.946 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.016 ms`
  开始时间(ns): `59813466`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2, 7168]]}`
- `q_a_layernorm` -> 0.067 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.067 ms`, `host_to_first_kernel_gap=48.883067`, `host_end_to_last_kernel_tail=48.818 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `60090056`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2, 1536]]}`
- `kv_a_layernorm` -> 0.034 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.034 ms`, `host_to_first_kernel_gap=48.793981`, `host_end_to_last_kernel_tail=48.763 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `60181382`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2, 512]]}`
- `q_b_proj` -> 0.209 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.209 ms`, `host_to_first_kernel_gap=48.72821`, `host_end_to_last_kernel_tail=48.539 ms`, `gpu_makespan=0.019 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `60250257`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2, 1536]]}`
- `rotary_emb` -> 0.099 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.099 ms`, `host_to_first_kernel_gap=48.378226`, `host_end_to_last_kernel_tail=48.281 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `60630481`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2], [2, 128, 64], [2, 1, 64]]}`
- `attn_mqa` -> 0.339 ms
  纯GPU kernel时间: `0.047 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.339 ms`, `host_to_first_kernel_gap=48.231067`, `host_end_to_last_kernel_tail=47.942 ms`, `gpu_makespan=0.050 ms`, `gpu_kernel_sum=0.047 ms`
  开始时间(ns): `60780520`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2, 128, 512], [2, 1, 512], [2, 1, 512]]}`
- `o_proj` -> 0.273 ms
  纯GPU kernel时间: `0.051 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.273 ms`, `host_to_first_kernel_gap=47.797469`, `host_end_to_last_kernel_tail=47.576 ms`, `gpu_makespan=0.052 ms`, `gpu_kernel_sum=0.051 ms`
  开始时间(ns): `61274950`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2, 16384]]}`

## Layer 3 / decode / instance 1

- 执行序号: `9`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `1.818 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=1.818 ms`, `module_to_last_kernel=47.004 ms`, `host_to_first_kernel_gap=46.832884`, `host_end_to_last_kernel_tail=45.186 ms`, `gpu_makespan=0.171 ms`, `gpu_kernel_sum=0.140 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.217 ms
  纯GPU kernel时间: `0.016 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.217 ms`, `host_to_first_kernel_gap=46.757924`, `host_end_to_last_kernel_tail=46.558 ms`, `gpu_makespan=0.017 ms`, `gpu_kernel_sum=0.016 ms`
  开始时间(ns): `62529822`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2, 7168]]}`
- `q_a_layernorm` -> 0.052 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.052 ms`, `host_to_first_kernel_gap=46.495054`, `host_end_to_last_kernel_tail=46.445 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `62809620`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2, 1536]]}`
- `kv_a_layernorm` -> 0.033 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.033 ms`, `host_to_first_kernel_gap=46.421088`, `host_end_to_last_kernel_tail=46.390 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `62885602`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2, 512]]}`
- `q_b_proj` -> 0.220 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.220 ms`, `host_to_first_kernel_gap=46.35929`, `host_end_to_last_kernel_tail=46.159 ms`, `gpu_makespan=0.019 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `62951688`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2, 1536]]}`
- `rotary_emb` -> 0.094 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.094 ms`, `host_to_first_kernel_gap=46.000671`, `host_end_to_last_kernel_tail=45.909 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `63340035`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2], [2, 128, 64], [2, 1, 64]]}`
- `attn_mqa` -> 0.335 ms
  纯GPU kernel时间: `0.047 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.335 ms`, `host_to_first_kernel_gap=45.856068`, `host_end_to_last_kernel_tail=45.569 ms`, `gpu_makespan=0.048 ms`, `gpu_kernel_sum=0.047 ms`
  开始时间(ns): `63487518`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2, 128, 512], [2, 1, 512], [2, 1, 512]]}`
- `o_proj` -> 0.291 ms
  纯GPU kernel时间: `0.053 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.291 ms`, `host_to_first_kernel_gap=45.44251`, `host_end_to_last_kernel_tail=45.206 ms`, `gpu_makespan=0.055 ms`, `gpu_kernel_sum=0.053 ms`
  开始时间(ns): `63961524`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2, 16384]]}`

## Layer 4 / decode / instance 1

- 执行序号: `10`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `2.052 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.052 ms`, `module_to_last_kernel=43.902 ms`, `host_to_first_kernel_gap=43.729853`, `host_end_to_last_kernel_tail=41.850 ms`, `gpu_makespan=0.172 ms`, `gpu_kernel_sum=0.140 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.341 ms
  纯GPU kernel时间: `0.016 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.341 ms`, `host_to_first_kernel_gap=43.643604`, `host_end_to_last_kernel_tail=43.319 ms`, `gpu_makespan=0.017 ms`, `gpu_kernel_sum=0.016 ms`
  开始时间(ns): `66025902`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[2, 7168]]}`
- `q_a_layernorm` -> 0.059 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.059 ms`, `host_to_first_kernel_gap=43.24985`, `host_end_to_last_kernel_tail=43.194 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `66436232`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[2, 1536]]}`
- `kv_a_layernorm` -> 0.034 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.034 ms`, `host_to_first_kernel_gap=43.168764`, `host_end_to_last_kernel_tail=43.137 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `66519462`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[2, 512]]}`
- `q_b_proj` -> 0.233 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.233 ms`, `host_to_first_kernel_gap=43.100779`, `host_end_to_last_kernel_tail=42.887 ms`, `gpu_makespan=0.019 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `66590359`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[2, 1536]]}`
- `rotary_emb` -> 0.098 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.098 ms`, `host_to_first_kernel_gap=42.706813`, `host_end_to_last_kernel_tail=42.611 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `67014501`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[2], [2, 128, 64], [2, 1, 64]]}`
- `attn_mqa` -> 0.345 ms
  纯GPU kernel时间: `0.047 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.345 ms`, `host_to_first_kernel_gap=42.54025`, `host_end_to_last_kernel_tail=42.244 ms`, `gpu_makespan=0.048 ms`, `gpu_kernel_sum=0.047 ms`
  开始时间(ns): `67185928`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[2, 128, 512], [2, 1, 512], [2, 1, 512]]}`
- `o_proj` -> 0.291 ms
  纯GPU kernel时间: `0.052 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.291 ms`, `host_to_first_kernel_gap=42.105714`, `host_end_to_last_kernel_tail=41.868 ms`, `gpu_makespan=0.053 ms`, `gpu_kernel_sum=0.052 ms`
  开始时间(ns): `67682672`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[2, 16384]]}`
