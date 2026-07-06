# Prefill 首层 Rope 到 set_mla_kv_buffer 大 Gap 溯源报告

## 结论

- 异常 gap 发生在 DeepSeek MHA prefill prepare 阶段的 `rotary_emb` 之后、`set_mla_kv_buffer_kernel` 之前；最终的 `set_mla_kv_buffer_kernel` 本身只有约 0.02ms，不是耗时来源。
- `b1_f4500/b1_f6000/b1_f8192` 的 gap 中可见 `ptxas`、`llvm-worker-*`、`cuModuleLoadData` 和 profiler/TLS 初始化事件；正常的 `b2_f4096/b4_f2048` 对照没有 `cuModuleLoadData`，rope 后几乎立即 launch `set_mla_kv_buffer_kernel`。
- 因此当前证据更支持：首层大 gap 是 Triton/CUDA module 的 lazy JIT 编译或模块加载污染，触发点是 SGLang 写 MLA latent KV cache 的 Triton kernel，而不是 KV cache prefix 加载、FA attention kernel 或 DeepGEMM kernel 慢。
- 这份 nsys 是 `--sample=none --cpuctxsw=none`，没有 CPU sampling/backtrace；所以还不能从 sqlite 直接证明 100% CPU 的完整函数栈。若要把“CPU 忙在 ptxas/Triton 编译”从高置信推断升级为直接证明，需要补采 `--sample=cpu` 或打开更完整 CPU stack。

## 关键数据

| group | tag | batch/fresh/prefix | warmup | self_attn host ms | rope->set_mla gap ms | set_mla kernel ms | module load in gap | first module load offset ms | compiler threads |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| target | req1_equal_fresh_regular_b1_f8192_p0 | b1 fresh=[8192] prefix=[0] | 1 | 255.489 | 251.443 | 0.025952 | 1 | 251.060 | llvm-worker-0,llvm-worker-1,ptxas |
| target | req1_equal_fresh_irregular_b1_f6000_p0 | b1 fresh=[6000] prefix=[0] | 1 | 288.167 | 283.185 | 0.019392 | 1 | 282.708 | llvm-worker-0,llvm-worker-1,ptxas |
| target | req1_equal_fresh_irregular_b1_f4500_p0 | b1 fresh=[4500] prefix=[0] | 1 | 278.926 | 273.634 | 0.014848 | 1 | 273.198 | llvm-worker-0,llvm-worker-1,ptxas |
| reference | req1_equal_fresh_regular_b2_f4096_p0 | b2 fresh=[4096,4096] prefix=[0,0] | 1 | 2.131 | 0.001 | 0.026144 | 0 |  |  |
| reference | req1_equal_fresh_regular_b4_f2048_p0 | b4 fresh=[2048,2048,2048,2048] prefix=[0,0,0,0] | 1 | 1.905 | 0.001 | 0.025984 | 0 |  |  |

## SGLang 0.5.9 执行路径对照

- `deepseek_common/attention_backend_handler.py` 中，FA3/FlashInfer 等后端在 extend prefill 且 `sum_extend_prefix_lens == 0` 时走 `MHA_ONE_SHOT`，本次 `prefix_lens=[0]` 正落在这个分支。
- 源码依据：`attention_backend_handler.py:81-94` 在 `forward_mode.is_extend_without_speculative()` 且 `(sum_extend_prefix_lens >= threshold or sum_extend_prefix_lens == 0)` 时选择 MHA one-shot 或 chunked-kv；本批次 prefix 为 0，因此进入 MHA prefill 路径。
- `deepseek_common/attention_forward_methods/forward_mha.py:209-214` 的 `forward_normal_prepare` 顺序是：`rotary_emb` -> `_set_mla_kv_buffer(...)`；随后 `forward_mha.py:235-246` 才执行 `kv_b_proj` 和拼接 MHA K。
- `_set_mla_kv_buffer` 的 CUDA 分支在 `forward_mha.py:389-400` 调用 `forward_batch.token_to_kv_pool.set_mla_kv_buffer(self.attn_mha, out_cache_loc, kv_a.unsqueeze(1), k_pe)`。
- `mem_cache/memory_pool.py:1509-1548` 的 `set_mla_kv_buffer` 最终调用 `mem_cache/utils.py` 里的 `set_mla_kv_buffer_triton(...)`。
- `mem_cache/utils.py:25-109` 定义了 `@triton.jit` 的 `set_mla_kv_buffer_kernel`；`set_mla_kv_buffer_triton` 的 grid 为 `(loc.numel(), cdiv(nope_dim + rope_dim, 128))`。DeepSeek V3 BF16 KV cache 下 `total_dim=512+64=576`，所以异常 b1_f8192 和正常 b2_f4096/b4_f2048 都看到 `grid=(8192,5,1)`。

## 为什么不是 KV cache 加载

- 这些异常 case 的 `prefix_lens=[0]`，没有 prefix KV 命中加载；当前 `_set_mla_kv_buffer` 是把本轮新 token 的 latent cache 写入 KV pool。
- 真正写入的 GPU kernel `set_mla_kv_buffer_kernel` 只有约 0.019ms 到 0.026ms；如果是 KV 写入/加载本身慢，耗时应体现在该 kernel 或 memcpy 上，而不是 kernel 启动前 200ms 级 host 空窗。
- gap 内没有 CUDA memcpy 或长 kernel，只有 gap 尾部出现 `cuModuleLoadData` 和后续 launch。

## 为什么 warmup 后仍可能出现

- 当前脚本确实在 `start_profile` 前执行了同长度 warmup；`stage.txt` 记录为 `before_warmup_generate_0 -> after_warmup_generate_0 -> before_start_profile`。
- 但正式 profile run 中仍出现 `cuModuleLoadData`，说明 warmup 没有覆盖或没有复用正式 run 的这个 CUDA module 实例。可疑因素包括：Triton specialization/module load 的 lazy 行为、profile/CUPTI attach 后触发的新模块加载、或不同 session/request 生命周期导致的首次 module load 重现。
- 目前证据能证明 `cuModuleLoadData` 发生在正式 run 的 gap 尾部，不能单靠现有 nsys 证明 warmup 未触发编译的精确原因；这需要 CPU sampling 或增加应用侧 Triton cache/compile 日志进一步确认。

## OSRT Threads 与 CPU 100% 的解释

- `b1_f8192` 的 gap 窗口内，OSRT 表能看到很长的 `epoll_wait/epoll_pwait/pthread_cond_timedwait/clock_nanosleep`，对应线程名包括 `ZMQbg/IO/*`、`python`、`pt_nccl_*`、`pt_tcpstore_uv` 等；这些事件覆盖窗口，但调用栈多是 Gloo/ZMQ/NCCL watchdog 或 TCPStore 后台等待，不是 scheduler 主线程下一 kernel launch 的直接原因。
- scheduler 主线程在 gap 起点附近只有极短的 `stat64/mmap64/munmap` 等 OSRT 调用；随后直到 gap 尾部的 `cuModuleLoadData/cuLaunchKernelEx`，OSRT 表并不记录 CPU busy 的函数栈。这与 `--sample=none --cpuctxsw=none` 的采集配置一致：CPU 忙等或编译计算不会以 sampling callchain 形式出现在 sqlite 中。
- `PROFILER_OVERHEAD` 表在异常用例 gap 中记录到 `ptxas`、`llvm-worker-0/1`、`TLS allocation`、`OS runtime libraries profiling initialization`、`In-process plugins initialization`；正常 b2/b4 对照没有同样的 `cuModuleLoadData` 和 compiler thread 组合。因此，你在 Nsight UI 中看到的 CPU 高占用，更可能对应 Triton/ptxas 编译或 module load 前的 host 侧工作，而不是 OSRT wait 事件本身。
- 因为当前 sqlite 没有 CPU sampling，报告把这一点定性为高置信推断，而不是已经精确定位到某个 Python/C++ 函数的直接栈证据。

## 为什么只在部分用例出现

- 异常集中于单请求长序列 `b1_f4500/b1_f6000/b1_f8192`；正常对照 `b2_f4096/b4_f2048` 虽然总 fresh token 同为 8192，但没有 module load，也没有大 gap。
- 这排除了“总 token 数大必然导致 gap”的解释，也排除了 `set_mla_kv_buffer_kernel` 运行时长随总 token 增大造成的解释。
- 更可能是单请求长序列路径在正式 profile run 中触发了某个 Triton/CUDA module lazy load；由于现有 nsys 没有 CPU sample，暂不能确定触发差异来自 Triton cache key、session 生命周期、CUPTI profiler attach，还是 SGLang 内部请求路径差异。

## 建议验证

- 对 `b1_f8192_p0` 重跑一次 nsys，打开 `--sample=cpu --cpuctxsw=process-tree` 或至少 CPU sampling，确认 gap 内 scheduler 线程/ptxas/llvm-worker 的调用栈。
- 显式设置并持久化 `TRITON_CACHE_DIR`，在 warmup 后检查正式 run 是否仍有 `cuModuleLoadData`。
- 增加一个 profile 前的 `torch.cuda.synchronize()`，并可选对 `set_mla_kv_buffer_triton` 做一次显式预触发，以验证是否能消除 rope->set_mla gap。

## 产物

- `prefill_first_layer_gap_summary.csv`：异常/对照 case 的首层 gap 汇总。
- `prefill_first_layer_kernel_sequence.csv`：首层 self_attn 起点到 `set_mla_kv_buffer_kernel` 附近的 kernel 序列。
