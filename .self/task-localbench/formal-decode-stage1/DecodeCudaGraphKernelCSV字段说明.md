# DecodeCudaGraphKernel CSV 字段说明

本文档说明 `bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/decode_csv/*__DecodeCudaGraphKernel汇总.csv` 的字段含义。

## 采集口径

- 每行对应 Nsight Systems SQLite 中 `CUPTI_ACTIVITY_KIND_KERNEL` 的一个 GPU kernel。
- 该 CSV 用于 decode CUDA Graph 场景；因为 CUDA Graph replay 下 Python/module NVTX 层级边界不稳定，因此这里保留 kernel 级一手记录，不再强行聚合为 MLA module 时间。
- 正式 profile 窗口由 SGLang `profile_by_stage + profile_decode_only_stage` 控制，只包住真实 `ForwardMode.DECODE`。
- 每个归档 case 均通过 scheduler 日志校验：存在 `Decode batch ... cuda graph: True`，且没有 profiled prefill CUDA Graph 命中。

## 运行与形状字段

- `kernel_order_index`：按 `kernel_start_ns, kernel_end_ns` 排序后的 kernel 序号。
- `run_instance_name`：对应的实机 run 目录名。
- `stage`：当前统一为 `decode`。
- `batch_size`：正式 decode batch size。
- `kv_len`：每个请求正式 decode 前已写入 KV cache 的 prefix 长度。
- `total_kv_tokens`：`batch_size * kv_len`，用于快速对齐总 KV 规模。
- `tp_size`：本次 run 的 TP 语义切分参数；当前归档为单卡 rank0 形状口径，不包含真实 TP 通信保真。
- `cuda_graph_mode`：runner 侧配置的 CUDA Graph 开关，正式数据为 `on`。
- `cuda_graph_validated`：解析侧标记，表示该 CSV 来自已通过 decode CUDA Graph 日志校验的 run。

## 时间字段

- `kernel_start_ns`：Nsight Systems 记录的 GPU kernel 开始时间，单位 ns。
- `kernel_end_ns`：Nsight Systems 记录的 GPU kernel 结束时间，单位 ns。
- `kernel_duration_ns`：`kernel_end_ns - kernel_start_ns`。
- `kernel_duration_ms`：`kernel_duration_ns / 1e6`，单位 ms。

## CUDA 标识字段

- `device_id`：CUDA device id。
- `context_id`：CUDA context id。
- `stream_id`：CUDA stream id。
- `correlation_id`：CUPTI correlation id，可用于和 runtime/API 记录做进一步关联。
- `graph_node_id`：CUDA Graph node id；开启 `--cuda-graph-trace=node` 后，graph 内部 kernel 通常会带该字段。
- `graph_id`：CUDA Graph id；部分非 graph 或前后处理 kernel 可能为空。
- `grid_xyz`：kernel launch grid 维度，JSON list 字符串。
- `block_xyz`：kernel launch block 维度，JSON list 字符串。

## 算子分类字段

- `operator_category`：解析脚本基于 kernel name 的粗粒度分类，例如 `attention`、`gemm`、`norm`、`activation`、`quant`、`sampling`、`triton_misc` 等。
- `implementation`：解析脚本基于 kernel name 推断的实现来源，例如 `flashmla`、`fa3`、`deepgemm`、`flashinfer`、`triton`、`torch`、`sglang_mla` 等。
- `kernel_name`：Nsight SQLite 中优先使用 demangled name，其次 short/mangled name 的 kernel 原始名称。
- `kernel_short_name`：Nsight SQLite 中的 short name，用于快速阅读和分类辅助。

## 使用提醒

- `operator_category` 和 `implementation` 是启发式分类，可信的一手信息仍是 `kernel_name`、CUDA 时间戳和 graph 字段。
- Decode CUDA Graph 下不建议把这些行直接解释为 Python module 事件；如果需要和 AIC collector 对齐，应优先区分真正的 `attention` kernel 与周边 `bmm` kernel。当前部分 `nvjet_tst_*` 已按 `bmm/torch` 归类，避免再把 q 前处理和 o_proj 对应的 batch matmul 混进 attention 对齐口径。
- CSV 中可能包含少量 `unknown` 或 `torch/triton_misc` kernel，这是正式 decode graph replay 前后处理、采样或框架辅助 kernel 的正常现象。
