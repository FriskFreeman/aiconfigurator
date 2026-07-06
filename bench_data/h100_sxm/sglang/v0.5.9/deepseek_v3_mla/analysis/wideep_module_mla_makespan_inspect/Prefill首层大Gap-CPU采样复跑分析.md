# Prefill 首层大 Gap CPU 采样复跑分析

## 结论

- 三个单请求长序列目标用例均复现 `rotary_emb -> set_mla_kv_buffer_kernel` 前的大 gap：约 478-540ms；`set_mla_kv_buffer_kernel` 自身仍只有约 0.02ms。
- 这次打开了 Nsight CPU sampling。gap 内 CPU 样本直接落在 `sglang::scheduler` 主线程、`ptxas`、Triton `_C/libtriton.so` 的 LLVM/MLIR 编译栈上；这把上一轮“疑似 Triton/CUDA lazy JIT/module load”的判断升级为直接证据。
- `cuModuleLoadData` 只出现在 gap 尾部，耗时约 0.23-0.25ms；因此几百毫秒的大头不是 CUDA module load API 本身，而是它之前的 Triton/LLVM/ptxas 编译与 materialize 工作。
- 参考用例 `b2_f4096_p0` 同样总 fresh token 为 8192，但 gap 只有 0.001ms，且 Triton cache 中只有 warmup 已生成的 `set_mla` specialization；它没有触发第二个 formal specialization。
- 单请求长序列目标用例的 Triton cache 中均出现两个 `set_mla_kv_buffer_kernel` specialization，差异集中在 `loc_ptr` 参数是否带 `tt.divisibility = 16`。这解释了为什么 warmup 后 formal 仍可能出现 gap：warmup 编译了一个 specialization，formal 的 `out_cache_loc/loc_ptr` 触发了另一个 specialization。
- 补充对照显示，单请求有 prefix 的 `b1_f8192_p1024`、`b1_f8192_p10500` 首层 `rope -> set_mla` gap 也只有约 0.001ms；因此触发条件应收敛为“单请求、长 fresh、formal 阶段无 prefix continuation 状态导致 `loc_ptr` specialization 切换”，而不是“单请求长序列必然触发”。

## 汇总

| group | shape | gap ms | set_mla grid | samples | scheduler/ptxas/llvm samples | Triton frames | ptxas frames | module load offset ms | set_mla variants | diff hint |
| --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| target_single_long | b1 fresh=[8192] prefix=[0] | 509.180 | `[8192,5,1]` | 437 | 235/126/9 | 3443 | 186 | 508.772 | 2 | `loc_ptr_divisible16_to_unannotated` |
| target_single_long | b1 fresh=[4500] prefix=[0] | 540.371 | `[4500,5,1]` | 468 | 255/169/8 | 3289 | 288 | 539.949 | 2 | `loc_ptr_divisible16_to_unannotated` |
| target_single_long | b1 fresh=[6000] prefix=[0] | 478.590 | `[6000,5,1]` | 355 | 161/127/8 | 2475 | 180 | 478.214 | 2 | `loc_ptr_divisible16_to_unannotated` |
| reference_multi_same_total | b2 fresh=[4096,4096] prefix=[0,0] | 0.001 | `[8192,5,1]` | 0 | 0/0/0 | 0 | 0 |  | 1 | `single_loc_ptr_divisible16` |

## 为什么 warmup 后仍然有 gap

- `stage.txt` 显示每个目标 run 都执行了 `before_warmup_generate_0 -> after_warmup_generate_0 -> before_start_profile -> after_start_profile -> before_generate`，所以不是完全没有 warmup。
- 但目标 run 的 `/out/triton_cache` 中有两个 `set_mla_kv_buffer_kernel` 目录；从 mtime 和 TTIR 差异看，warmup 先生成 `loc_ptr` 带 16-byte divisibility 的 specialization，formal profile 段又生成了 `loc_ptr` 未标注 divisibility 的 specialization。
- Nsight gap 内采样显示 scheduler 主线程在 Triton `_C/libtriton.so` 的 LLVM/MLIR 路径中，同时有独立 `ptxas` 线程/进程样本。因此 formal 段不是简单读取 warmup 产物，而是在编译另一个 Triton specialization。

## 为什么主要是这些单请求长序列 shape

- 对照 `b2_f4096_p0` 和目标 `b1_f8192_p0` 的 `set_mla` grid 都是 `[8192,5,1]`，但前者 gap 为 0.001ms，后者为 509ms；所以不是总 token 或 kernel grid 本身导致。
- 更合理的触发条件是 formal run 中 `out_cache_loc/loc_ptr` 的指针属性或张量构造路径不同，导致 Triton cache key 变化。单请求长序列在 formal session 中更容易走到 `loc_ptr` 非 16-byte divisibility specialization；多请求同总 token 的参考形状复用了 warmup specialization。
- 这里仍保留一点不确定性：Nsight 数据能证明差异体现在 Triton specialization 与 `loc_ptr` 属性，不能仅从 trace 反推出 SGLang allocator 选择该 `loc_ptr` 属性的全部内部条件。若要最终闭环，需要在 SGLang `set_mla_kv_buffer_triton` 调用前打印 `out_cache_loc.data_ptr()`、`stride/storage_offset/is_contiguous` 与 Triton cache key。

## `loc_ptr` 与 `tt.divisibility = 16` 的具体含义

- `set_mla_kv_buffer_kernel` 的职责是把本轮 fresh token 计算出的 latent KV 写入全局 KV pool。`kv_buffer_ptr` 指向目标 KV cache 大池子，`cache_k_nope_ptr` 和 `cache_k_rope_ptr` 指向待写入的 KV 内容，`loc_ptr` 则是每个 fresh token 应写入哪个 KV slot 的索引表。
- 对于 `fresh_len=8192` 的 prefill，`loc_ptr` 长度通常也是 8192。kernel 逻辑可以抽象为：对每个 fresh token 读取 `loc_ptr[token_i]` 得到 cache slot，然后把该 token 的 `k_nope/k_rope` 写入 `kv_buffer[slot]`。
- `tt.divisibility = 16` 不是 SGLang 显式传入的业务参数，而是 Triton 编译器对参数做出的静态属性推断，表示该指针或张量地址在编译期被认为满足 16-byte 对齐/整除性质。这类属性可能来自 `data_ptr` 对齐、storage offset、stride、contiguity 等张量元信息。
- Triton 的 cache key 不只包含 kernel 函数名和 grid，也会包含 dtype、constexpr、指针对齐/divisibility 等 specialization 属性。因此 `loc_ptr: ptr<i64> {tt.divisibility = 16}` 和普通 `loc_ptr: ptr<i64>` 会被视作两个不同的 `set_mla_kv_buffer_kernel` 版本。
- 本次异常 run 的 TTIR diff 显示，warmup 生成的是 `loc_ptr` 带 16-byte divisibility 的 specialization，而 formal profile 段又触发了 `loc_ptr` 未标注 divisibility 的 specialization。CPU sample 同时显示 formal gap 内 scheduler 主线程在 Triton/LLVM/MLIR 编译路径中，并伴随 `ptxas` 样本；这说明 gap 是第二个 specialization 的 cold compile，而不是 `set_mla` kernel 本身慢。

## 单请求有 Prefix 为什么不触发

- 额外检查已有运行结果后，`b1_f8192_p1024` 和 `b1_f8192_p10500` 的首层 `rope -> set_mla` gap 均约 0.001ms，`set_mla` grid 仍为 `[8192,5,1]`。这说明有 prefix 后并不是因为 fresh KV 写入量变小而避免 gap，formal 阶段仍然要写入 8192 个 fresh token。
- 差异在于 session/request 生命周期。`prefix_len=0` 时，formal 前的 prefix-cache pass 只是 open session，不会先 generate prefix token；formal 请求以“空 session + 纯 fresh extend”的形态直接进入 scheduler。`prefix_len>0` 时，formal 前会先执行 prefix cache 预处理，fresh 阶段变成已有 session continuation。
- continuation 状态会改变 `prefix_indices`、KV pool 已占用位置、`out_cache_loc` 分配路径等上下文。经验上这更容易让 formal fresh 阶段的 `loc_ptr` 形态与 warmup 阶段保持一致，从而复用 warmup 已编译的 `set_mla` specialization。
- 因此单请求有 prefix 的 case 不触发，并不是 prefix 命中本身减少了 `set_mla` 工作量，而是 prefix continuation 让 formal 阶段没有踩到 warmup 未覆盖的 `loc_ptr` specialization。

## 多请求无 Prefix 为什么不触发

- 多请求无 prefix 的 `b2_f4096_p0` 与异常 `b1_f8192_p0` 有相同 total fresh token 和相同 `set_mla` grid `[8192,5,1]`，但没有大 gap。这说明触发条件不在 total token 或 grid，而在 host 侧传给 Triton 的 `loc_ptr/out_cache_loc` 参数属性。
- 多请求 batch 需要把多个请求的 fresh token 合并成一个 extend batch，例如 `[req0 4096 tokens, req1 4096 tokens]`。为了供后续 batch kernel 使用，SGLang 通常会走更标准的 batch allocation/packing 路径，把每个请求的 cache loc 拼成一个规整的一维 `out_cache_loc` tensor。
- 这种合批/拼接过程相当于对 `loc_ptr` 做了一次“规范化”：更可能得到 contiguous、storage offset 为 0、data pointer 对齐、Triton 可推断 `tt.divisibility = 16` 的 tensor。warmup 和 formal 都走同一种多请求 batch packing 路径，所以 formal 能复用 warmup 的 specialization。
- 单请求无 prefix 则可能少了这种多请求合批整理步骤，直接把某个 request/session 内部产生的 loc tensor、slice、view 或临时分配结果传给 Triton。warmup 和 formal 虽 shape 相同，但 session id、KV pool 当前分配状态、request 生命周期不同，足以让 Triton 看到不同的 `loc_ptr` 属性。
- 更精确地说，多请求无 prefix 没有污染的原因是：warmup 和 formal 都保持了同一个 `set_mla_kv_buffer_kernel(loc_ptr divisible16)` specialization；单请求无 prefix 的 formal 则切换到了 `set_mla_kv_buffer_kernel(loc_ptr generic)`，因此触发第二次 cold-specialization 编译。

## 可复现性与随机性判断

- 从当前证据看，这不是普通运行随机抖动：三个目标 shape 在独立复跑中全部稳定出现同类 gap、同类 CPU 编译栈、同类第二个 `set_mla` specialization；参考 shape 在同一采样配置下没有出现 gap。
- gap 绝对值会受机器负载、Nsight sampling、Triton/ptxas 编译耗时影响，所以 478ms/509ms/540ms 这些数值不应视作严格常数；但“是否触发第二个 `loc_ptr` specialization 编译”是当前更固定、更可复现的判据。
- 若复用持久化 Triton cache 或提前显式预触发 formal 所需 specialization，这类 gap 预期会消失或显著缩小；因此它更像 cold-specialization 编译污染，而不是模型计算本身的稳定耗时。

## 运行与环境注意

- 普通容器下 `nsys status --environment` 显示 `perf_event_open` 和 sampling trigger 失败；加 `--privileged --pid=host` 虽可打开 CPU sampling，但会暴露全部 GPU，破坏 `--gpus device=7` 隔离。
- 本次最终采用 `--cap-add SYS_ADMIN --cap-add SYS_PTRACE --security-opt seccomp=unconfined`，CPU profiling OK，且容器内仍只看到指定 GPU。

## 产物

- `gap_cpusample_rerun_summary.csv`：四个复跑用例的 gap、runtime、CPU sampling、Triton cache specialization 汇总。
- `gap_cpusample_rerun_sample_frames.csv`：gap 内截取的 Triton/LLVM/ptxas 相关 CPU sample 栈帧。
