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
- 整块 MLA-module 时长: `43285.355 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=43285.355 ms`, `module_to_last_kernel=43286.140 ms`, `host_to_first_kernel_gap=1.879394`, `host_end_to_last_kernel_tail=0.786 ms`, `gpu_makespan=43284.261 ms`, `gpu_kernel_sum=6.031 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=8192`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=0`, `sum_seq_after=8192`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 8192, 'prefix_len': 0, 'seq_len_after': 8192, 'prompt_len': 8192, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 5920.008 ms
  纯GPU kernel时间: `0.276 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=5920.008 ms`, `host_to_first_kernel_gap=1.600152`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=5918.352 ms`, `gpu_kernel_sum=0.276 ms`
  开始时间(ns): `4800522792`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.272 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.272 ms`, `host_to_first_kernel_gap=0.244094`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `10720820494`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 5423.997 ms
  纯GPU kernel时间: `0.453 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=5423.997 ms`, `host_to_first_kernel_gap=0.357836`, `host_end_to_last_kernel_tail=0.171 ms`, `gpu_makespan=5423.810 ms`, `gpu_kernel_sum=0.453 ms`
  开始时间(ns): `10721210080`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.240 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.240 ms`, `host_to_first_kernel_gap=0.214337`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.014 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `16145528552`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 19234.199 ms
  纯GPU kernel时间: `0.107 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=19234.199 ms`, `host_to_first_kernel_gap=19234.069972`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.107 ms`, `gpu_kernel_sum=0.107 ms`
  开始时间(ns): `16145925369`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 5328.912 ms
  纯GPU kernel时间: `0.293 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=5328.912 ms`, `host_to_first_kernel_gap=0.50999`, `host_end_to_last_kernel_tail=0.005 ms`, `gpu_makespan=5328.407 ms`, `gpu_kernel_sum=0.293 ms`
  开始时间(ns): `36763049109`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[8192, 512]]}`
- `attn_mha` -> 19.240 ms
  纯GPU kernel时间: `3.606 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=19.240 ms`, `host_to_first_kernel_gap=3.369293`, `host_end_to_last_kernel_tail=3.470 ms`, `gpu_makespan=19.341 ms`, `gpu_kernel_sum=3.606 ms`
  开始时间(ns): `42094220824`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]}`
- `o_proj` -> 5971.840 ms
  纯GPU kernel时间: `1.264 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=5971.840 ms`, `host_to_first_kernel_gap=3.230267`, `host_end_to_last_kernel_tail=0.842 ms`, `gpu_makespan=5969.452 ms`, `gpu_kernel_sum=1.264 ms`
  开始时间(ns): `42113701927`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 1 / prefill / instance 1

- 执行序号: `2`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `5.479 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=5.479 ms`, `module_to_last_kernel=9.552 ms`, `host_to_first_kernel_gap=0.83894`, `host_end_to_last_kernel_tail=4.073 ms`, `gpu_makespan=8.713 ms`, `gpu_kernel_sum=6.019 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=8192`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=0`, `sum_seq_after=8192`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 8192, 'prefix_len': 0, 'seq_len_after': 8192, 'prompt_len': 8192, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.937 ms
  纯GPU kernel时间: `0.275 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.937 ms`, `host_to_first_kernel_gap=0.382809`, `host_end_to_last_kernel_tail=0.125 ms`, `gpu_makespan=0.679 ms`, `gpu_kernel_sum=0.275 ms`
  开始时间(ns): `60507775395`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.141 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.141 ms`, `host_to_first_kernel_gap=0.11912`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `60508877612`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.622 ms
  纯GPU kernel时间: `0.453 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.622 ms`, `host_to_first_kernel_gap=0.22351`, `host_end_to_last_kernel_tail=0.369 ms`, `gpu_makespan=0.767 ms`, `gpu_kernel_sum=0.453 ms`
  开始时间(ns): `60509105446`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.111 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.111 ms`, `host_to_first_kernel_gap=0.188123`, `host_end_to_last_kernel_tail=0.091 ms`, `gpu_makespan=0.014 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `60509908225`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.227 ms
  纯GPU kernel时间: `0.106 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.227 ms`, `host_to_first_kernel_gap=0.190385`, `host_end_to_last_kernel_tail=0.070 ms`, `gpu_makespan=0.106 ms`, `gpu_kernel_sum=0.106 ms`
  开始时间(ns): `60510135051`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.615 ms
  纯GPU kernel时间: `0.290 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.615 ms`, `host_to_first_kernel_gap=0.230879`, `host_end_to_last_kernel_tail=0.219 ms`, `gpu_makespan=0.603 ms`, `gpu_kernel_sum=0.290 ms`
  开始时间(ns): `60510764765`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[8192, 512]]}`
- `attn_mha` -> 0.427 ms
  纯GPU kernel时间: `3.605 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.427 ms`, `host_to_first_kernel_gap=0.319356`, `host_end_to_last_kernel_tail=3.545 ms`, `gpu_makespan=3.652 ms`, `gpu_kernel_sum=3.605 ms`
  开始时间(ns): `60511638112`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]}`
- `o_proj` -> 0.593 ms
  纯GPU kernel时间: `1.259 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.593 ms`, `host_to_first_kernel_gap=3.441781`, `host_end_to_last_kernel_tail=4.108 ms`, `gpu_makespan=1.260 ms`, `gpu_kernel_sum=1.259 ms`
  开始时间(ns): `60512169382`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 2 / prefill / instance 1

- 执行序号: `3`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `4.480 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=4.480 ms`, `module_to_last_kernel=12.783 ms`, `host_to_first_kernel_gap=6.491652`, `host_end_to_last_kernel_tail=8.303 ms`, `gpu_makespan=6.291 ms`, `gpu_kernel_sum=6.004 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=8192`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=0`, `sum_seq_after=8192`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 8192, 'prefix_len': 0, 'seq_len_after': 8192, 'prompt_len': 8192, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.582 ms
  纯GPU kernel时间: `0.269 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.582 ms`, `host_to_first_kernel_gap=6.212911`, `host_end_to_last_kernel_tail=5.901 ms`, `gpu_makespan=0.270 ms`, `gpu_kernel_sum=0.269 ms`
  开始时间(ns): `60515284171`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.113 ms
  纯GPU kernel时间: `0.017 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.113 ms`, `host_to_first_kernel_gap=5.772922`, `host_end_to_last_kernel_tail=5.677 ms`, `gpu_makespan=0.017 ms`, `gpu_kernel_sum=0.017 ms`
  开始时间(ns): `60515994720`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.520 ms
  纯GPU kernel时间: `0.441 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.520 ms`, `host_to_first_kernel_gap=5.602994`, `host_end_to_last_kernel_tail=5.526 ms`, `gpu_makespan=0.443 ms`, `gpu_kernel_sum=0.441 ms`
  开始时间(ns): `60516183976`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.100 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.100 ms`, `host_to_first_kernel_gap=5.367283`, `host_end_to_last_kernel_tail=5.280 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `60516862567`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.193 ms
  纯GPU kernel时间: `0.106 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.193 ms`, `host_to_first_kernel_gap=5.180892`, `host_end_to_last_kernel_tail=5.094 ms`, `gpu_makespan=0.106 ms`, `gpu_kernel_sum=0.106 ms`
  开始时间(ns): `60517062718`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.549 ms
  纯GPU kernel时间: `0.300 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.549 ms`, `host_to_first_kernel_gap=4.797111`, `host_end_to_last_kernel_tail=4.549 ms`, `gpu_makespan=0.301 ms`, `gpu_kernel_sum=0.300 ms`
  开始时间(ns): `60517582723`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[8192, 512]]}`
- `attn_mha` -> 0.356 ms
  纯GPU kernel时间: `3.605 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.356 ms`, `host_to_first_kernel_gap=4.563197`, `host_end_to_last_kernel_tail=7.812 ms`, `gpu_makespan=3.605 ms`, `gpu_kernel_sum=3.605 ms`
  开始时间(ns): `60518362397`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]}`
- `o_proj` -> 0.633 ms
  纯GPU kernel时间: `1.253 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.633 ms`, `host_to_first_kernel_gap=7.713721`, `host_end_to_last_kernel_tail=8.336 ms`, `gpu_makespan=1.256 ms`, `gpu_kernel_sum=1.253 ms`
  开始时间(ns): `60518818753`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 3 / prefill / instance 1

- 执行序号: `4`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `4.475 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=4.475 ms`, `module_to_last_kernel=17.132 ms`, `host_to_first_kernel_gap=10.873433`, `host_end_to_last_kernel_tail=12.657 ms`, `gpu_makespan=6.258 ms`, `gpu_kernel_sum=5.975 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=8192`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=0`, `sum_seq_after=8192`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 8192, 'prefix_len': 0, 'seq_len_after': 8192, 'prompt_len': 8192, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.562 ms
  纯GPU kernel时间: `0.266 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.562 ms`, `host_to_first_kernel_gap=10.611023`, `host_end_to_last_kernel_tail=10.317 ms`, `gpu_makespan=0.268 ms`, `gpu_kernel_sum=0.266 ms`
  开始时间(ns): `60521785994`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.110 ms
  纯GPU kernel时间: `0.017 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.110 ms`, `host_to_first_kernel_gap=10.191002`, `host_end_to_last_kernel_tail=10.098 ms`, `gpu_makespan=0.017 ms`, `gpu_kernel_sum=0.017 ms`
  开始时间(ns): `60522474495`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.576 ms
  纯GPU kernel时间: `0.437 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.576 ms`, `host_to_first_kernel_gap=10.020927`, `host_end_to_last_kernel_tail=9.883 ms`, `gpu_makespan=0.439 ms`, `gpu_kernel_sum=0.437 ms`
  开始时间(ns): `60522663226`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.104 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.104 ms`, `host_to_first_kernel_gap=9.724512`, `host_end_to_last_kernel_tail=9.634 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `60523398105`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.176 ms
  纯GPU kernel时间: `0.106 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.176 ms`, `host_to_first_kernel_gap=9.532384`, `host_end_to_last_kernel_tail=9.463 ms`, `gpu_makespan=0.106 ms`, `gpu_kernel_sum=0.106 ms`
  开始时间(ns): `60523605721`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.588 ms
  纯GPU kernel时间: `0.283 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.588 ms`, `host_to_first_kernel_gap=9.151672`, `host_end_to_last_kernel_tail=8.848 ms`, `gpu_makespan=0.285 ms`, `gpu_kernel_sum=0.283 ms`
  开始时间(ns): `60524120769`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[8192, 512]]}`
- `attn_mha` -> 0.334 ms
  纯GPU kernel时间: `3.611 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.334 ms`, `host_to_first_kernel_gap=8.841002`, `host_end_to_last_kernel_tail=12.117 ms`, `gpu_makespan=3.611 ms`, `gpu_kernel_sum=3.611 ms`
  开始时间(ns): `60524958671`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]}`
- `o_proj` -> 0.572 ms
  纯GPU kernel时间: `1.241 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.572 ms`, `host_to_first_kernel_gap=12.02045`, `host_end_to_last_kernel_tail=12.691 ms`, `gpu_makespan=1.243 ms`, `gpu_kernel_sum=1.241 ms`
  开始时间(ns): `60525392214`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 4 / prefill / instance 1

- 执行序号: `5`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `2.400 ms`
- 内部子模块数量: `8`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.400 ms`, `module_to_last_kernel=7.156 ms`, `host_to_first_kernel_gap=0.400955`, `host_end_to_last_kernel_tail=4.756 ms`, `gpu_makespan=6.755 ms`, `gpu_kernel_sum=6.008 ms`, `gpu_kernel_count=13`
- 是否严格对齐 collector 单块边界: `True`
- Attention形状信息: `module=attn_mha`, `token_count=8192`, `total_tokens=8192`, `chunked_req_prefix_len=0`, `current_chunked_req_prefix_len=0`, `current_chunked_req_prefix_len_source=reconstructed_scheduler`, `inputs=[[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]`
- 重建调度信息: `round=1`, `sum_extend=8192`, `sum_prefix=0`, `sum_seq_after=8192`, `chunked_req_ids=[]`, `chunked_req_prefix_lens=[]`, `items=[{'req_id': 0, 'extend_len': 8192, 'prefix_len': 0, 'seq_len_after': 8192, 'prompt_len': 8192, 'is_chunked_req': 0}]`
- SGLang调度chunk信息: `scheduler_nvtx未捕获，当前prefix由attention输入形状推断`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.456 ms
  纯GPU kernel时间: `0.275 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.456 ms`, `host_to_first_kernel_gap=0.168663`, `host_end_to_last_kernel_tail=0.166 ms`, `gpu_makespan=0.454 ms`, `gpu_kernel_sum=0.275 ms`
  开始时间(ns): `62780009987`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[8192, 7168]]}`
- `q_a_layernorm` -> 0.075 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.075 ms`, `host_to_first_kernel_gap=0.101269`, `host_end_to_last_kernel_tail=0.044 ms`, `gpu_makespan=0.018 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `62780530981`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[8192, 1536]]}`
- `q_b_proj` -> 0.254 ms
  纯GPU kernel时间: `0.448 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.254 ms`, `host_to_first_kernel_gap=0.091425`, `host_end_to_last_kernel_tail=0.405 ms`, `gpu_makespan=0.568 ms`, `gpu_kernel_sum=0.448 ms`
  开始时间(ns): `62780641529`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[8192, 1536]]}`
- `kv_a_layernorm` -> 0.043 ms
  纯GPU kernel时间: `0.013 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.043 ms`, `host_to_first_kernel_gap=0.337146`, `host_end_to_last_kernel_tail=0.307 ms`, `gpu_makespan=0.013 ms`, `gpu_kernel_sum=0.013 ms`
  开始时间(ns): `62780963776`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[8192, 512]]}`
- `rotary_emb` -> 0.114 ms
  纯GPU kernel时间: `0.106 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.114 ms`, `host_to_first_kernel_gap=0.260579`, `host_end_to_last_kernel_tail=0.252 ms`, `gpu_makespan=0.106 ms`, `gpu_kernel_sum=0.106 ms`
  开始时间(ns): `62781054391`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[8192], [8192, 128, 64], [8192, 1, 64]]}`
- `kv_b_proj` -> 0.236 ms
  纯GPU kernel时间: `0.292 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.236 ms`, `host_to_first_kernel_gap=0.116164`, `host_end_to_last_kernel_tail=0.262 ms`, `gpu_makespan=0.382 ms`, `gpu_kernel_sum=0.292 ms`
  开始时间(ns): `62781333334`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_b_proj', 'TrainableParams': {'weight': [32768, 512], 'weight_scale_inv': [256, 4]}, 'Inputs': [[8192, 512]]}`
- `attn_mha` -> 0.200 ms
  纯GPU kernel时间: `3.606 ms`, kernel数: `2`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.200 ms`, `host_to_first_kernel_gap=0.398019`, `host_end_to_last_kernel_tail=3.804 ms`, `gpu_makespan=3.606 ms`, `gpu_kernel_sum=3.606 ms`
  开始时间(ns): `62781676535`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mha', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[8192, 128, 192], [8192, 128, 192], [8192, 128, 128]]}`
- `o_proj` -> 0.241 ms
  纯GPU kernel时间: `1.250 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.241 ms`, `host_to_first_kernel_gap=3.760582`, `host_end_to_last_kernel_tail=4.770 ms`, `gpu_makespan=1.251 ms`, `gpu_kernel_sum=1.250 ms`
  开始时间(ns): `62781921524`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[8192, 16384]]}`

## Layer 0 / decode / instance 1

- 执行序号: `6`
- Collector对齐边界: `{'Module': 'model.model.layers.0.self_attn'}`
- 整块 MLA-module 时长: `17596.549 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=17596.549 ms`, `module_to_last_kernel=17596.549 ms`, `host_to_first_kernel_gap=0.334363`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=17596.016 ms`, `gpu_kernel_sum=0.122 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 5204.783 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=5204.783 ms`, `host_to_first_kernel_gap=0.205933`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=5204.388 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `66711220719`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.139 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.139 ms`, `host_to_first_kernel_gap=0.128378`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `71916207600`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.038 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.038 ms`, `host_to_first_kernel_gap=0.0332`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `71916381178`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 7270.229 ms
  纯GPU kernel时间: `0.019 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=7270.229 ms`, `host_to_first_kernel_gap=0.177095`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=7269.807 ms`, `gpu_kernel_sum=0.019 ms`
  开始时间(ns): `71916483555`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.328 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.328 ms`, `host_to_first_kernel_gap=0.277534`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `79196174644`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 28.970 ms
  纯GPU kernel时间: `0.031 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=28.970 ms`, `host_to_first_kernel_gap=0.316911`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=28.498 ms`, `gpu_kernel_sum=0.031 ms`
  开始时间(ns): `79196667299`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 5075.216 ms
  纯GPU kernel时间: `0.052 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=5075.216 ms`, `host_to_first_kernel_gap=0.327705`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=5074.753 ms`, `gpu_kernel_sum=0.052 ms`
  开始时间(ns): `79232362838`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 1 / decode / instance 1

- 执行序号: `7`
- Collector对齐边界: `{'Module': 'model.model.layers.1.self_attn'}`
- 整块 MLA-module 时长: `4.990 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=4.990 ms`, `module_to_last_kernel=4.990 ms`, `host_to_first_kernel_gap=0.595367`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=4.320 ms`, `gpu_kernel_sum=0.119 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.938 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.938 ms`, `host_to_first_kernel_gap=0.377747`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.501 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `94874071650`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.135 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.135 ms`, `host_to_first_kernel_gap=0.112948`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `94875191521`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.078 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.078 ms`, `host_to_first_kernel_gap=0.063901`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `94875395576`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.598 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.598 ms`, `host_to_first_kernel_gap=0.213522`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.340 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `94875573379`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.245 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.245 ms`, `host_to_first_kernel_gap=0.190438`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `94876633070`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.799 ms
  纯GPU kernel时间: `0.030 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.799 ms`, `host_to_first_kernel_gap=0.254237`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.470 ms`, `gpu_kernel_sum=0.030 ms`
  开始时间(ns): `94877002263`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.655 ms
  纯GPU kernel时间: `0.049 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.655 ms`, `host_to_first_kernel_gap=0.240958`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.393 ms`, `gpu_kernel_sum=0.049 ms`
  开始时间(ns): `94878135350`
  原始NVTX: `{'Module': 'model.model.layers.1.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 2 / decode / instance 1

- 执行序号: `8`
- Collector对齐边界: `{'Module': 'model.model.layers.2.self_attn'}`
- 整块 MLA-module 时长: `4.057 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=4.057 ms`, `module_to_last_kernel=4.057 ms`, `host_to_first_kernel_gap=0.362605`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=3.635 ms`, `gpu_kernel_sum=0.118 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.544 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.544 ms`, `host_to_first_kernel_gap=0.205288`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.296 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `94881247788`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.110 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.110 ms`, `host_to_first_kernel_gap=0.091467`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `94881955785`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.078 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.078 ms`, `host_to_first_kernel_gap=0.063806`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `94882128438`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.522 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.522 ms`, `host_to_first_kernel_gap=0.191966`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.290 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `94882295574`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.185 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.185 ms`, `host_to_first_kernel_gap=0.153833`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `94883172811`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.690 ms
  纯GPU kernel时间: `0.030 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.690 ms`, `host_to_first_kernel_gap=0.208946`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.419 ms`, `gpu_kernel_sum=0.030 ms`
  开始时间(ns): `94883470786`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.657 ms
  纯GPU kernel时间: `0.050 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.657 ms`, `host_to_first_kernel_gap=0.214848`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.426 ms`, `gpu_kernel_sum=0.050 ms`
  开始时间(ns): `94884446740`
  原始NVTX: `{'Module': 'model.model.layers.2.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 3 / decode / instance 1

- 执行序号: `9`
- Collector对齐边界: `{'Module': 'model.model.layers.3.self_attn'}`
- 整块 MLA-module 时长: `4.084 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=4.084 ms`, `module_to_last_kernel=4.084 ms`, `host_to_first_kernel_gap=0.372438`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=3.644 ms`, `gpu_kernel_sum=0.118 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.588 ms
  纯GPU kernel时间: `0.014 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.588 ms`, `host_to_first_kernel_gap=0.218935`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.326 ms`, `gpu_kernel_sum=0.014 ms`
  开始时间(ns): `94887373949`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.110 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.110 ms`, `host_to_first_kernel_gap=0.090631`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `94888109773`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.077 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.077 ms`, `host_to_first_kernel_gap=0.064018`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `94888282530`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.536 ms
  纯GPU kernel时间: `0.018 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.536 ms`, `host_to_first_kernel_gap=0.191216`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.307 ms`, `gpu_kernel_sum=0.018 ms`
  开始时间(ns): `94888448164`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.187 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.187 ms`, `host_to_first_kernel_gap=0.153469`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `94889320470`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.691 ms
  纯GPU kernel时间: `0.031 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.691 ms`, `host_to_first_kernel_gap=0.209175`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.407 ms`, `gpu_kernel_sum=0.031 ms`
  开始时间(ns): `94889615036`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.658 ms
  纯GPU kernel时间: `0.049 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.658 ms`, `host_to_first_kernel_gap=0.242076`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.393 ms`, `gpu_kernel_sum=0.049 ms`
  开始时间(ns): `94890601239`
  原始NVTX: `{'Module': 'model.model.layers.3.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`

## Layer 4 / decode / instance 1

- 执行序号: `10`
- Collector对齐边界: `{'Module': 'model.model.layers.4.self_attn'}`
- 整块 MLA-module 时长: `2.361 ms`
- 内部子模块数量: `7`
- 该层该阶段实例数: `1`
- 模块时延拆解: `host_total=2.361 ms`, `module_to_last_kernel=2.361 ms`, `host_to_first_kernel_gap=0.279953`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=2.070 ms`, `gpu_kernel_sum=0.124 ms`, `gpu_kernel_count=13`

### 子模块Breakdown（按执行先后）

- `fused_qkv_a_proj_with_mqa` -> 0.491 ms
  纯GPU kernel时间: `0.015 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.491 ms`, `host_to_first_kernel_gap=0.16906`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.299 ms`, `gpu_kernel_sum=0.015 ms`
  开始时间(ns): `98219600571`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.fused_qkv_a_proj_with_mqa', 'TrainableParams': {'weight': [2112, 7168], 'weight_scale_inv': [17, 56]}, 'Inputs': [[1, 7168]]}`
- `q_a_layernorm` -> 0.065 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.065 ms`, `host_to_first_kernel_gap=0.056749`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `98220171826`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_a_layernorm', 'TrainableParams': {'weight': [1536]}, 'Inputs': [[1, 1536]]}`
- `kv_a_layernorm` -> 0.033 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `RMSNormKernel`
  时延拆解: `host_total=0.033 ms`, `host_to_first_kernel_gap=0.029127`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `98220263704`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.kv_a_layernorm', 'TrainableParams': {'weight': [512]}, 'Inputs': [[1, 512]]}`
- `q_b_proj` -> 0.252 ms
  纯GPU kernel时间: `0.023 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.252 ms`, `host_to_first_kernel_gap=0.0971`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.152 ms`, `gpu_kernel_sum=0.023 ms`
  开始时间(ns): `98220345139`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.q_b_proj', 'TrainableParams': {'weight': [24576, 1536], 'weight_scale_inv': [192, 12]}, 'Inputs': [[1, 1536]]}`
- `rotary_emb` -> 0.141 ms
  纯GPU kernel时间: `0.002 ms`, kernel数: `1`, dominant kernel: `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel`
  时延拆解: `host_total=0.141 ms`, `host_to_first_kernel_gap=0.125385`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.002 ms`, `gpu_kernel_sum=0.002 ms`
  开始时间(ns): `98220825366`
  原始NVTX: `{'Module': 'model.model.layers.0.self_attn.rotary_emb', 'Inputs': [[1], [1, 128, 64], [1, 1, 64]]}`
- `attn_mqa` -> 0.372 ms
  纯GPU kernel时间: `0.031 ms`, kernel数: `4`, dominant kernel: `device_kernel`
  时延拆解: `host_total=0.372 ms`, `host_to_first_kernel_gap=0.11974`, `host_end_to_last_kernel_tail=0.000 ms`, `gpu_makespan=0.238 ms`, `gpu_kernel_sum=0.031 ms`
  开始时间(ns): `98221021507`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.attn_mqa', 'TrainableParams': {'k_scale': [], 'v_scale': []}, 'Inputs': [[1, 128, 512], [1, 1, 512], [1, 1, 512]]}`
- `o_proj` -> 0.284 ms
  纯GPU kernel时间: `0.049 ms`, kernel数: `2`, dominant kernel: `sm90_fp8_gemm_1d2d_impl`
  时延拆解: `host_total=0.284 ms`, `host_to_first_kernel_gap=0.109583`, `host_end_to_last_kernel_tail=0.020 ms`, `gpu_makespan=0.195 ms`, `gpu_kernel_sum=0.049 ms`
  开始时间(ns): `98221534832`
  原始NVTX: `{'Module': 'model.model.layers.4.self_attn.o_proj', 'TrainableParams': {'weight': [7168, 16384], 'weight_scale_inv': [56, 128]}, 'Inputs': [[1, 16384]]}`
