# MLA时延拆解 CSV 字段说明

本文说明 `nsys` 拆解阶段导出的 `MLA时延拆解.csv` 各字段含义。当前对应的样例文件为：

- [MLA时延拆解.csv](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-prefill-stage1/20260615_104055_prefill_stage1_verify_var_1024_8192_4096_8192_b4_svar1024-8192-4096-8192_layers5_backend_fa3_cg_off_pcg_off_marker_on_profile_nsys/nsys/MLA时延拆解.csv)

字段由导出脚本 [export_nsys_profile.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-localbench/formal-prefill-stage1/export_nsys_profile.py) 生成。每一行表示：

- 一次 `prefill/decode stage instance`
- 其中某一层 `layer.self_attn` 模块
- 在 `nsys + NVTX + CUPTI kernel` 视角下的时延拆解结果

## 总体约定

- 后缀 `_ns` 表示纳秒。
- 后缀 `_ms` 表示毫秒。
- `module_*` 表示针对当前层 `self_attn` 整个模块聚合后的时间。
- `child_*` 表示模块内部子模块级摘要。
- `kernel_*` 表示底层 CUDA kernel 级摘要。
- `*_json` 列保存结构化 JSON 字符串，便于后续再解析。

## 一、行标识与实例定位

### `event_order_index`
- 当前 `self_attn` 事件在整份 trace 中按时间排序后的顺序编号。
- 用于恢复真实执行先后顺序。

### `run_instance_name`
- 当前运行实例的目录名，和 `formal-prefill-stage1` 下实际创建的文件夹名一致。
- 例如 `20260617_171515_prefill_stage1_smoke_nsys_tp2_rank0_noprefix_b1_f32_p0_layers3_backend_triton_cg_off_pcg_off_tp2_marker_on_profile_nsys`。

### `tp_size`
- 当前运行实例的 TP 大小。
- 例如 `1` 或 `2`。

### `layer_id`
- Transformer layer 编号。
- 例如 `0` 表示第 0 层。

### `stage`
- 当前事件所属阶段。
- 目前常见为 `prefill`，后续也可出现 `decode`。

### `stage_instance_index`
- 当前阶段内第几个实例。
- 对 prefill 来说，可理解为第几个 prefill round / chunk 实例。

### `stage_instance_count`
- 当前运行中该阶段总共有多少个实例。
- 可用于判断当前行处于该阶段的第几轮。

### `collector_prefill_aligned`
- 当前 prefill 实例是否与 collector 的 MLA module 采样边界相对齐。
- `true` 通常表示其输入形态更接近 collector 里“完整一轮 MLA module”采样目标。
- `false` 表示该轮更像是被真实调度切出来的中间 chunk，不能直接拿去与 collector 的单点表项一一对照。

## 二、模块级 Host 时间

### `module_host_start_ns`
- Python / NVTX 视角下，当前 `self_attn` 模块进入时间。

### `module_host_end_ns`
- Python / NVTX 视角下，当前 `self_attn` 模块退出时间。

### `module_host_duration_ns`
- Host 侧模块持续时间。
- 计算方式：`module_host_end_ns - module_host_start_ns`

### `module_host_duration_ms`
- `module_host_duration_ns` 的毫秒版本。

这组字段反映的是：

- 上层框架进入该模块
- 进行参数准备、kernel launch、可能的少量同步
- 到 Python 函数返回

它不是“纯 GPU 执行时间”，而是 Host 侧完整模块包络。

## 三、模块级 GPU 包络时间

### `module_first_kernel_start_ns`
- 当前模块关联到的首个 CUDA kernel 启动时间。

### `module_last_kernel_end_ns`
- 当前模块关联到的最后一个 CUDA kernel 结束时间。

### `module_total_to_last_kernel_ns`
- 从模块 Host 入口到最后一个 kernel 结束的端到端时间。
- 计算方式：`max(module_host_end_ns, module_last_kernel_end_ns) - module_host_start_ns`

### `module_total_to_last_kernel_ms`
- `module_total_to_last_kernel_ns` 的毫秒版本。

这组字段更接近“collector 若以模块头部为起点、最后一个相关 kernel 为终点”时的模块总时间。

## 四、Host 与 GPU 之间的前后间隔

### `module_host_to_first_kernel_gap_ns`
- 模块 Host 入口到首个 kernel 启动之间的间隔。
- 计算方式：`module_first_kernel_start_ns - module_host_start_ns`

### `module_host_to_first_kernel_gap_ms`
- `module_host_to_first_kernel_gap_ns` 的毫秒版本。

它反映：

- Python 进入模块后
- 到第一批 GPU 工作真正开始前
- 中间经历的 launch / 参数整理 / 框架调度开销

### `module_host_end_to_last_kernel_tail_ns`
- 模块 Host 返回后，到最后一个 kernel 真正结束之间的“拖尾时间”。
- 这是截断为非负数后的版本。
- 计算方式：`max(0, module_last_kernel_end_ns - module_host_end_ns)`

### `module_host_end_to_last_kernel_tail_ms`
- `module_host_end_to_last_kernel_tail_ns` 的毫秒版本。

### `module_host_end_to_last_kernel_tail_ns_signed`
- 与上一字段相同，但保留符号。
- 计算方式：`module_last_kernel_end_ns - module_host_end_ns`

### `module_host_end_to_last_kernel_tail_ms_signed`
- `module_host_end_to_last_kernel_tail_ns_signed` 的毫秒版本。

这四个字段里：

- `signed` 版本保留真实相对关系。
- 非 `signed` 版本更适合看“Host 返回后 GPU 仍在拖尾多久”。

解释：

- 若 `signed > 0`，说明 Host 侧函数已经返回，但 GPU 最后一个 kernel 还没跑完。
- 若 `signed < 0`，说明最后一个 kernel 在 Host 返回前就已经结束了，此时非 signed 列会记为 `0`。

## 五、纯 GPU 时间

### `module_gpu_makespan_ns`
- 当前模块所有相关 kernel 的 GPU 时间包络。
- 计算方式：`module_last_kernel_end_ns - module_first_kernel_start_ns`

### `module_gpu_makespan_ms`
- `module_gpu_makespan_ns` 的毫秒版本。

### `module_gpu_kernel_time_sum_ns`
- 当前模块下所有去重后的 kernel 时长之和。

### `module_gpu_kernel_time_sum_ms`
- `module_gpu_kernel_time_sum_ns` 的毫秒版本。

### `module_gpu_kernel_count`
- 当前模块关联到的去重后 kernel 数量。

### `module_first_kernel_name`
- 首个 kernel 的 `shortName`。

### `module_last_kernel_name`
- 最后一个 kernel 的 `shortName`。

这里要区分两个概念：

- `module_gpu_makespan_*` 是“首 kernel 到尾 kernel”的 GPU 包络时间。
- `module_gpu_kernel_time_sum_*` 是所有 kernel 时长直接累加。

二者一般不相等，因为：

- kernel 之间可能有空洞
- 不同 stream 上可能有重叠
- launch 与执行存在异步关系

## 六、Attention / MLA 形状摘要

### `attention_module`
- 当前用于形状解析的 attention 子模块名。
- 一般为 `attn_mha` 或 `attn_mqa`。

### `attention_token_count`
- 该 attention 调用本轮实际处理的新鲜 token 总量。
- 对 prefill 来说，通常对应当前 chunk 中所有请求 `fresh_len` 之和。

### `attention_total_tokens`
- 该 attention 调用在计算时看到的总 token 规模。
- 一般更接近 `prefix + fresh` 后的总工作集。

### `current_chunked_req_prefix_len`
- 当前 prefill round 中，若存在被切分的 `chunked_req`，其在本轮进入 attention 前已经累积好的 prefix 长度。
- 若本轮不存在 chunked request，则通常为 `0`。

### `current_chunked_req_prefix_len_source`
- `current_chunked_req_prefix_len` 的来源说明。
- 当前实现里主要用于标注其是：
  - 从 trace 直接可见信息推断得到
  - 还是由离线重建 scheduler 逻辑恢复得到

## 七、当前 chunk 内逐请求列表字段

这些字段是为了解决单个聚合数字过于粗糙的问题。它们描述“本轮 prefill chunk 内到底装了哪些请求，每个请求在这一轮里各是什么状态”。

### `chunk_req_ids_json`
- 当前 chunk 内各请求的 `req_id` 列表。

### `chunk_req_prefix_lens_json`
- 当前 chunk 内各请求在本轮进入 attention 前的 prefix 长度列表。
- 在当前无预设 KV 命中场景下，通常只有已被前几轮 chunk 过的那个请求会出现非零 prefix。

### `chunk_req_fresh_lens_json`
- 当前 chunk 内各请求本轮实际新处理的 token 数列表。
- 这些值加总后通常不超过本轮 chunk 窗口上限。

### `chunk_req_total_seq_lens_json`
- 当前 chunk 内各请求在本轮计算时的总序列长度列表。
- 定义为：`prefix_len + fresh_len`

### `chunk_req_prompt_lens_json`
- 每个请求原始完整 prompt 的总长度列表。
- 即请求从一开始的目标 prompt 长度，不随 chunk 切分改变。

### `chunk_req_is_chunked_json`
- 标记每个请求在当前 round 是否为“被切开的 chunked request”。
- `1` 表示本轮只吃了该请求的一部分。
- `0` 表示该请求在本轮被完整纳入。

### `chunk_req_items_json`
- 上述逐请求信息的完整对象列表版本。
- 每个元素通常包含：
  - `req_id`
  - `extend_len`
  - `prefix_len`
  - `seq_len_after`
  - `prompt_len`
  - `is_chunked_req`

其中：

- `extend_len` 等价于本轮该请求的 `fresh_len`
- `seq_len_after` 等价于 `prefix_len + extend_len`

## 八、模块内部子模块摘要

### `child_timing_json`
- 当前 `self_attn` 模块内部各子模块的时延拆解列表。
- 由脚本对每个子模块再次执行与主模块同样的 host/gpu 时间拆解得到。

每个子项通常包含：

- `canonical_name`
- `order_index`
- `host_start_ns`
- `host_end_ns`
- `host_duration_ns/ms`
- `first_kernel_start_ns`
- `last_kernel_end_ns`
- `module_total_to_last_kernel_ns/ms`
- `host_to_first_kernel_gap_ns/ms`
- `host_end_to_last_kernel_tail_ns/ms`
- `host_end_to_last_kernel_tail_ns_signed/ms_signed`
- `gpu_makespan_ns/ms`
- `gpu_kernel_time_sum_ns/ms`
- `gpu_kernel_count`
- `first_kernel_name`
- `last_kernel_name`

它适合分析如：

- `q_proj`
- `kv_b_proj`
- `attn_mha`
- `o_proj`

这些子模块各自的 host/gpu 拆解情况。

## 九、底层 kernel 摘要

### `kernel_summary_json`
- 当前模块内部各子模块对应的 CUDA kernel 摘要列表。

每个子项通常包含：

- `canonical_name`
- `top_gpu_kernels`
- `gpu_kernel_count`

其中 `top_gpu_kernels` 是按 GPU 时间排序的主要 kernel 列表，每个元素一般含有：

- `short_name`
- `gpu_time_ns`
- `gpu_time_ms`

这个字段用于回答：

- 某个子模块底层主要由哪些 kernel 构成
- 主导实现更像 `deepgemm`、`flash attention` 还是其他 fused kernel

## 十、几个最重要时间字段之间的关系

通常可按下面方式理解：

### `module_host_duration_ms`
- 纯 Host 模块持续时间。

### `module_total_to_last_kernel_ms`
- 从 Host 进入模块开始，一直到最后相关 kernel 结束的总包络时间。
- 最接近“模块级端到端完成时间”。

### `module_gpu_makespan_ms`
- 只看 GPU 侧，首 kernel 到尾 kernel 的覆盖时间。

### `module_gpu_kernel_time_sum_ms`
- 所有 kernel 自身执行时间直接求和。

经验上通常满足：

- `module_total_to_last_kernel_ms >= module_host_duration_ms`
- `module_gpu_makespan_ms >= 0`
- `module_gpu_kernel_time_sum_ms` 可能大于、等于或小于 `module_gpu_makespan_ms`

原因是：

- `kernel_time_sum` 是按 kernel 单独累计
- `gpu_makespan` 是首尾包络
- 两者受异步 launch、空隙、跨 stream 重叠影响

## 十一、适用范围说明

当前 CSV 解释基于现有 Stage 1 解析实现，默认前提包括：

- 主要针对 `prefill` 的 MLA/self-attention 事件
- prefix 信息是离线重建，不是 runtime 原生显式输出
- kernel 关联依赖 `NVTX module window -> runtime launch -> correlationId -> CUPTI kernel`

因此这份 CSV 最适合用于：

- 模块级 host/gpu 时延拆解
- collector MLA module 边界对齐分析
- chunked prefill 形状与前缀上下文分析
- 子模块与底层 kernel 主导实现分析

若后续调度策略、trace 标记方式或 profiling 采集口径变化，部分字段的解释也需要同步更新。
