# GPU gap host-root-cause 深挖报告

## 核心结论

- 大 gap 的边界 kernel 与 CUDA runtime `correlationId` 能够对上；多数异常 gap 不是 GPU 上有长 kernel 占用，而是前一个 kernel 结束后 host 很久没有发起下一段相关 runtime launch。
- Prefill 的 200ms 级 gap 发生在 layer0 `rotary_emb -> kv_b_proj`，`host_silence_before_first_runtime_ms` 基本等于整个 gap，下一段 `cuModuleLoadData/cuLaunchKernel*` 都出现在 gap 尾部。这指向首层 lazy 初始化/host 调度等待，而不是 MLA/FA/GEMM kernel 本身慢。
- Decode 的 gap 绝对值较小但相对 kernel sum 很大，典型是 `attn_mqa -> o_proj` 或 `rotary_emb -> o_proj`。runtime 总时长只有几十微秒，gap 主要是短 workload 下固定 host launch 间隔吞掉 makespan。
- OSRT 长等待事件经常与 gap 窗口重叠，但它们多为后台线程等待；本报告只把它们作为旁证，不作为直接根因。

## Prefill gap 根因分类

| tag | max gap ms | mean gap ms | layer | edge | cause | child host gap ms | first runtime offset ms | next launch offset ms | runtime total ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| req1_equal_fresh_irregular_b1_f4500_p0 | 274.305 | 274.305 | 0 | rotary_emb->kv_b_proj | host_no_launch_long_prefill_first_layer_gap | 273.875 | 273.198 | 274.272 | 0.256 |
| req1_equal_fresh_irregular_b1_f6000_p0 | 283.861 | 283.861 | 0 | rotary_emb->kv_b_proj | host_no_launch_long_prefill_first_layer_gap | 283.443 | 282.708 | 283.828 | 0.283 |
| req1_equal_fresh_regular_b1_f8192_p0 | 252.021 | 252.021 | 0 | rotary_emb->kv_b_proj | host_no_launch_long_prefill_first_layer_gap | 252.005 | 251.060 | 251.993 | 0.222 |
| req1_equal_prefix_regular_b1_f8192_p0 | 222.126 | 222.126 | 0 | rotary_emb->kv_b_proj | host_no_launch_long_prefill_first_layer_gap | 221.706 | 220.913 | 222.090 | 0.281 |

## Decode gap 根因分类

| tag | max gap ms | mean gap ms | layer | edge | cause | child host gap ms | first runtime offset ms | next launch offset ms | runtime total ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| req1_equal_fresh_irregular_b12_f300_p0 | 0.235 | 0.235 | 4 | q_b_proj->rotary_emb | short_decode_fixed_launch_interval_dominates | 0.154 | 0.103 | 0.218 | 0.029 |
| req1_equal_fresh_irregular_b1_f4500_p0 | 0.519 | 0.400 | 0 | attn_mqa->o_proj | short_decode_fixed_launch_interval_dominates | 0.266 | 0.201 | 0.502 | 0.035 |
| req1_equal_fresh_irregular_b1_f6000_p0 | 0.316 | 0.316 | 4 | attn_mqa->o_proj | short_decode_fixed_launch_interval_dominates | 0.166 | 0.122 | 0.305 | 0.020 |
| req1_equal_fresh_regular_b1_f1_p0 | 0.904 | 0.554 | 0 | rotary_emb->o_proj | short_decode_fixed_launch_interval_dominates | 0.792 | 0.122 | 0.894 | 0.072 |
| req1_equal_fresh_regular_b4_f512_p0 | 0.257 | 0.237 | 0 | attn_mqa->o_proj | short_decode_fixed_launch_interval_dominates | 0.126 | 0.097 | 0.248 | 0.017 |
| req1_equal_fresh_regular_b8_f1_p0 | 0.136 | 0.132 | 0 | attn_mqa->o_proj | short_decode_fixed_launch_interval_dominates | 0.075 | 0.047 | 0.129 | 0.010 |
| req1_equal_prefix_regular_b4_f512_p512 | 0.292 | 0.268 | 0 | attn_mqa->o_proj | short_decode_fixed_launch_interval_dominates | 0.153 | 0.120 | 0.282 | 0.019 |

## 最大 gap 边界样例

| stage | tag | layer | edge | gap ms | prev kernel | next kernel | prev runtime | next runtime |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prefill | req1_equal_fresh_irregular_b1_f6000_p0 | 0 | rotary_emb->kv_b_proj | 283.861 | BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel | per_token_group_quant_8bit_kernel | cudaLaunchKernel_v7000 | cudaLaunchKernel_v7000 |
| prefill | req1_equal_fresh_irregular_b1_f4500_p0 | 0 | rotary_emb->kv_b_proj | 274.305 | BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel | per_token_group_quant_8bit_kernel | cudaLaunchKernel_v7000 | cudaLaunchKernel_v7000 |
| prefill | req1_equal_fresh_regular_b1_f8192_p0 | 0 | rotary_emb->kv_b_proj | 252.021 | BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel | per_token_group_quant_8bit_kernel | cudaLaunchKernel_v7000 | cudaLaunchKernel_v7000 |
| prefill | req1_equal_prefix_regular_b1_f8192_p0 | 0 | rotary_emb->kv_b_proj | 222.126 | BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel | per_token_group_quant_8bit_kernel | cudaLaunchKernel_v7000 | cudaLaunchKernel_v7000 |
| decode | req1_equal_fresh_regular_b1_f1_p0 | 0 | rotary_emb->o_proj | 0.904 | BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel | per_token_group_quant_8bit_kernel | cudaLaunchKernel_v7000 | cudaLaunchKernel_v7000 |
| decode | req1_equal_fresh_regular_b1_f1_p0 | 4 | rotary_emb->o_proj | 0.846 | BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel | per_token_group_quant_8bit_kernel | cudaLaunchKernel_v7000 | cudaLaunchKernel_v7000 |
| decode | req1_equal_fresh_regular_b1_f1_p0 | 1 | rotary_emb->o_proj | 0.829 | BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel | per_token_group_quant_8bit_kernel | cudaLaunchKernel_v7000 | cudaLaunchKernel_v7000 |
| decode | req1_equal_fresh_regular_b1_f1_p0 | 2 | rotary_emb->o_proj | 0.818 | BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel | per_token_group_quant_8bit_kernel | cudaLaunchKernel_v7000 | cudaLaunchKernel_v7000 |

## 字段说明

- `host_silence_before_first_runtime_ms`：gap 起点到 gap 内第一条 CUDA runtime 调用开始的时间；若接近 `gpu_gap_ms`，说明 host 在大部分 gap 中没有发起 CUDA 调用。
- `host_silence_before_next_launch_ms`：gap 起点到下一边界 kernel 对应 runtime launch 开始的时间。
- `host_gap_ms`：gap 两侧子模块 NVTX/host marker 的结束到开始间隔，用来判断 Python/module 层是否也出现同向空窗。
- `first_runtime_offset_ms/next_launch_runtime_offset_ms`：从 gap 起点算起，第一条 CUDA runtime 调用和下一边界 kernel launch 出现的位置。
- `runtime_total_ms`：gap 窗口内 CUDA runtime 调用时长总和；它远小于 gap 时，说明 runtime 自身不是主要耗时。
- `longest_runtime_idle_ms`：只看 CUDA runtime 时间线时，gap 内最长无 runtime 区间。
- `prev_runtime_json/next_runtime_json`：边界 kernel 通过 `correlationId` 对应到的 host runtime launch。

## 时间线补充核验

新增 `gap_timeline_events.csv` 将每个入选 gap 的边界 kernel、边界 kernel 对应 CUDA runtime、gap 内 CUDA runtime、NVTX module marker、OSRT/Profiler 事件统一到同一个 offset 时间轴。offset 的 0 点为前一个 GPU kernel 结束，即 gap 起点。

代表样例 `prefill / req1_equal_fresh_irregular_b1_f6000_p0 / layer0 / rotary_emb->kv_b_proj`：

| offset ms | event | 说明 |
| --- | --- | --- |
| -0.078..0.000 | `BatchQKApplyRotaryPosIdsCosSinCacheHeadParallelismKernel` | 前一个 rotary kernel 执行结束 |
| -0.041 | `rotary_emb.host_end` | host 子模块 marker 也在 gap 起点附近结束 |
| 0.000 | `gap_start_prev_kernel_end` | GPU gap 开始 |
| 77.020..102.407 | Profiler overhead | 采样中可见 TLS / profiling 初始化类事件，但不对应下一段 kernel launch |
| 282.708..282.888 | `cuModuleLoadData` | gap 尾部才出现第一条相关 CUDA runtime |
| 283.402 | `kv_b_proj.host_start` | 下一子模块 host marker 开始 |
| 283.828..283.865 | `cudaLaunchKernel_v7000` | 下一边界 kernel 的 host launch |
| 283.861 | `per_token_group_quant_8bit_kernel` start | 下一 GPU kernel 开始，gap 结束 |

这个样例说明：283.861ms 的 GPU gap 中，约 282.708ms 之前没有发起相关 CUDA runtime；OSRT 的 `epoll_wait/pthread_cond_timedwait/clock_nanosleep` 虽覆盖窗口，但属于后台等待类事件，不是下一 kernel launch 的直接原因。更可信的解释是首层进入 `kv_b_proj` 前发生 lazy module/kernel 初始化或 host 调度等待，导致 GPU 长时间没有拿到下一段工作。

代表样例 `decode / req1_equal_fresh_regular_b1_f1_p0 / layer0 / rotary_emb->o_proj`：

| offset ms | event | 说明 |
| --- | --- | --- |
| -0.002..0.000 | rotary kernel | 前一个 GPU kernel 结束 |
| 0.013 | `rotary_emb.host_end` | host marker 已结束 |
| 0.176..0.444 | `kv_b_proj` NVTX | host 端继续进入后续子模块 |
| 0.405..0.417 | `cuLaunchKernelEx` | gap 中间出现一次短 runtime |
| 0.562..0.753 | `attn_mha` NVTX | 继续执行 attention 子模块 host 逻辑 |
| 0.694..0.705 | `cudaLaunchKernel_v7000` | 又一次短 launch |
| 0.805 | `o_proj.host_start` | 下一边界子模块 host marker 开始 |
| 0.894..0.902 | `cudaLaunchKernel_v7000` | 下一边界 kernel launch |
| 0.904 | `per_token_group_quant_8bit_kernel` start | 下一 GPU kernel 开始 |

这个样例说明：decode gap 中 host 端并不是完全沉默，而是短 workload 下多个 module/NVTX 和 kernel launch 间隔本身占比较高；runtime 总时长只有约 0.072ms，但从前一 kernel 结束到下一边界 kernel 启动有约 0.904ms。对于 decode 小形状，makespan 被固定 launch/module 间隔放大，不能解读为某个 GPU kernel 慢。

## 产物

- `gap_root_cause_summary.csv`：按 case 汇总最大 gap、host silence、runtime 总时长和分类。
- `gap_root_cause_details.csv`：每个入选 gap 的边界 kernel、runtime、NVTX/OSRT/Profiler 事件摘要。
- `gap_root_cause_details.json`：完整结构化明细。
- `gap_timeline_events.csv`：按绝对时间轴展开每个 gap 的边界 kernel、runtime、NVTX、OSRT、Profiler 事件，用于定位 gap 内 host 是否发起下一段 GPU 工作。
