# MLA Kernel 实机与 AIC 对比阶段性结论

## 1. 本轮工作范围

本轮只聚焦 DeepSeek V3 在 SGLang 路径下的 MLA attention kernel。

- 实机数据来源：
  - `bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/manifest.csv`
  - 每个 case 对应的 `MLA时延拆解.csv`
- AIC 仿真来源：
  - `src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.9/context_mla_perf.txt`
  - `src/aiconfigurator/systems/data/h100_sxm/sglang/0.5.9/generation_mla_perf.txt`
  - `PerfDatabase.query_context_mla()`
  - `PerfDatabase.query_generation_mla()`

已新增脚本：

- [compare_mla_kernel_with_aic.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-prefill-stage1/compare_mla_kernel_with_aic.py)

已生成结果：

- [mla_kernel_aic_compare__comparison.csv](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/analysis/mla_kernel_aic_compare__comparison.csv)
- [mla_kernel_aic_compare__summary.md](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/analysis/mla_kernel_aic_compare__summary.md)
- [mla_kernel_aic_compare_with_decode__comparison.csv](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/analysis/mla_kernel_aic_compare_with_decode__comparison.csv)
- [mla_kernel_aic_compare_with_decode__summary.md](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/analysis/mla_kernel_aic_compare_with_decode__summary.md)

## 2. 当前对比口径

### 实机侧

从 `MLA时延拆解.csv` 中取：

- prefill：`attn_mha` 子模块
- decode：`attn_mqa` 子模块

并使用其：

- `real_gpu_makespan_ms`
- `real_gpu_kernel_time_sum_ms`

作为对比参考。

当前主指标是：

- `real_gpu_makespan_ms`

因为它更接近“attention 子模块从首 GPU kernel 启动到尾 GPU kernel 结束”的整段 GPU 端时间。

### AIC 侧

严格按当前 SDK 的 DeepSeek + SGLang fallback 逻辑：

- prefill 对应 `PerfDatabase.query_context_mla()`
- decode 对应 `PerfDatabase.query_generation_mla()`

其中：

- prefill 用 `fresh_len` 作为 `s`
- prefill 用 `prefix_len` 参与 `full_s = fresh_len + prefix_len` 查表，并乘 prefix 修正
- decode 用 `total_seq_len + 1` 作为 `s`

## 3. 关键结论

### 3.1 Decode 基本在同一量级

把 decode 一并纳入后：

- decode 行数：`95`
- `bfloat16` 假设下，AIC 对 `real_gpu_makespan_ms` 的平均绝对百分比误差约为 `37.49%`

这说明：

- `query_generation_mla()` 对 `attn_mqa` 的建模口径，至少与实机 trace 的 `attn_mqa` 子模块 GPU 时间是同一量级的
- decode 侧更像是“仍有配置/插值误差，但边界大体一致”

### 3.2 Prefill 明显不是简单插值误差

prefill 结果：

- prefill 行数：`90`
- `bfloat16` 假设下，平均绝对百分比误差约为 `285.42%`
- `fp8` 假设下，平均绝对百分比误差约为 `273.74%`

这说明：

- 当前 `query_context_mla()` 与实机 trace 中 `attn_mha` 的对比口径存在系统性偏差
- 不是只靠切换 `bf16/fp8` 就能解决

### 3.3 偏差随 prefix / total_len 变大而进一步放大

最差 case 集中在：

- `req1_equal_prefix_regular_b1_f8192_p32000`

该 case 中：

- 实机 `attn_mha` GPU makespan 约 `29.5 ~ 31.7 ms`
- AIC `query_context_mla()` 给出约 `143.6 ms`

误差达到 `350% ~ 386%`。

这说明：

- 当前 AIC prefill MLA kernel 的 prefix 修正逻辑，至少在与实机 `attn_mha` 子模块对比时，缩放关系明显不匹配
- 或者说，`context_mla_perf.txt` 的采集边界并不等价于实机 trace 中当前提取的 `attn_mha` 边界

## 4. 对 collector 边界的再判断

从 [collector/sglang/collect_mla.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/collector/sglang/collect_mla.py) 看，`collect_mla` 测的是：

- 直接构造 `RadixAttention`
- 直接构造 SGLang MLA backend
- 调一次 `layer(q, k, v, forward_batch, ...)`
- 用 `benchmark_with_power()` 在 CUDA graph / eager 路径下测这整个调用的 GPU 时间

因此它采到的不是：

- 单个 `device_kernel`

而是更接近：

- 一个完整的 attention backend 调用边界

这个边界内部至少可能包含：

- `prepare_varlen_num_blocks_kernel`
- 主 `device_kernel`
- 其他前后处理 kernel

所以从定义上说，AIC `context_mla/generation_mla` 更应与实机 trace 中：

- `attn_mha / attn_mqa` 子模块的整段 GPU 时间

对齐，而不是只与单个 dominant `device_kernel` 对齐。

这一点与当前脚本选择是相符的。

## 5. 为什么 prefill 仍然明显不对

现阶段更可信的判断是：

1. decode `generation_mla` 与实机 `attn_mqa` 大体同口径
2. prefill `context_mla` 与实机 `attn_mha` 之间仍存在额外口径差异

可能的原因包括：

- `collect_mla.py` 的 context 路径与 engine 内真实 prefill `attn_mha` 执行分支并不完全一致
- formal run 中 `force_prefill_mha_for_prefix`、`attention_backend=auto`、prefix cache 条件，会让 engine 内选择的 prefill attention 执行流与 collector mock 路径有差异
- `query_context_mla()` 的 prefix 修正是“理论二次项修正”，而实机 `attn_mha` trace 的实际 GPU 时间并不按这个比例缩放
- 实机 trace 中 `attn_mha` 的 NVTX 子模块边界可能仍比 collector 的纯 backend 调用边界更窄

## 6. 当前可采信的阶段性判断

### 可以先采信的部分

- 当前 formal CSV 足以构造一套 attention-kernel 级实机数据库
- AIC 的 `generation_mla` 与实机 decode `attn_mqa` 已经能做有意义对比
- AIC 的 `context_mla` 与实机 prefill `attn_mha` 直接对比时，存在显著系统性高估

### 暂时不能直接下结论的部分

- 不能直接说 AIC 的 `context_mla_perf` 数据本身错了
- 更准确的说法应是：
  - “当前 formal trace 中选取的 prefill `attn_mha` 时延，与 AIC `query_context_mla()` 的口径尚未严格对齐”

## 7. 下一步建议

下一步应继续做两件事：

1. 进一步核对 `collect_mla.py` context 路径与 formal engine prefill `attn_mha` 的真实执行分支是否完全一致。
2. 在现有 trace 中，尝试识别一个比当前 `attn_mha gpu_makespan` 更贴近 collector 边界的时间定义，特别是确认是否还应包含某些前后相关 kernel 或同步拖尾。

在这两步完成之前，当前更稳妥的结论是：

- decode MLA-kernel 对比已经具备参考价值
- prefill MLA-kernel 对比链路已经打通，但边界仍需继续对齐
