# Decode CUDA Graph 数据集摘要

生成时间：2026-06-25

## 采集口径

- 场景：DeepSeek-V3 decode-only，SGLang v0.5.9，H100 SXM。
- GPU：GPU5。
- 运行方式：先用 session 写入 prefix KV cache，再对 continuation 触发正式 decode。
- CUDA Graph：开启，并通过 scheduler 日志强校验 `Decode batch ... cuda graph: True`。
- Profile 窗口：通过 `profile_by_stage + profile_decode_only_stage` 只包住真实 `ForwardMode.DECODE` batch；不归档旧的 extend/prefill 伪 decode 结果。
- Nsys：使用 `--cuda-graph-trace=node` 展开 CUDA graph 内部 kernel。
- CSV 口径：decode cudagraph replay 中没有可靠的 Python/module NVTX 层级边界，因此归档 `DecodeCudaGraphKernel汇总.csv`，每行对应一个 GPU kernel，包含 kernel 时间、graph 信息、粗粒度 operator 分类与实现来源。

## 归档结果

- Manifest：`decode_manifest.csv`
- Raw runs：`raw_runs/` 是 prefill/decode 共享原始归档目录；其中本次 decode manifest 指向 9 个正式 decode run。
- CSV/JSON/MD：`decode_csv/` 下 27 个文件，9 个 case 每个 3 个文件。
- 旧 decode 归档：已移除先前未真正命中 decode CUDA graph 的旧 decode manifest、CSV 和 decode raw run 归档；历史 prefill raw run 保留。

## 正式 Case

| tag | batch_size | kv_len | 状态 |
|---|---:|---:|---|
| decode_b1p512_cg_on | 1 | 512 | ok |
| decode_b4p512_cg_on | 4 | 512 | ok |
| decode_b32p512_cg_on | 32 | 512 | ok |
| decode_b4p2048_cg_on | 4 | 2048 | ok |
| decode_b16p2048_cg_on | 16 | 2048 | ok |
| decode_b4p4096_cg_on | 4 | 4096 | ok |
| decode_b8p4096_cg_on | 8 | 4096 | ok |
| decode_b4p8192_cg_on | 4 | 8192 | ok |
| decode_b2p16384_cg_on | 2 | 16384 | ok |

## 校验摘要

- manifest 行数：9 个正式 case，状态均为 `ok`。
- 每个 case 的 stderr 均包含真实 decode CUDA graph 日志：`Decode batch ... cuda graph: True`。
- 每个 case 的 stderr 均不包含被 profile 的 prefill CUDA graph 证据：`prefill_cuda_graph_true_logs=0`。
- 每个 kernel CSV 均包含 `flashmla`、`fa3`、`deepgemm` 等实现分类。
- 已复核 9 个正式 case 的 kernel CSV 行数：`138-144` 行，且 `stage` 均为 `decode`。
