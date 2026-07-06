# PD Mixed / Agg 实机采集说明

本目录保存 SGLang v0.5.9 / H100 SXM / DeepSeek-V3 MLA 的 PD mixed 实机采集结果。

## 正式数据

- `formal_uniform_agg_summary.csv` 是正式批量的入口汇总表。
- 正式用例的 `tag` 均以 `agg_uniform_` 开头。
- 每个正式用例包含：
  - `*_MLA时延拆解.csv`：nsys 解析出的 self-attn/module 时延。
  - `*_MLA对齐解析.json`：结构化解析结果。
  - `*_MLA对齐解析.md`：可读摘要。
  - `*_mixed_batch_meta.jsonl`：SGLang scheduler 中真实 MIXED batch 的 prefill/decode 形状信息。

## 采集口径

- 正式批量使用统一 `decode_prefix_len`、`prefill_fresh_len`、`prefill_prefix_len`，对齐 AIC `run_agg` 的单一 isl/prefix 口径。
- 每个正式用例运行 6 层；首层可能仍包含冷启动/JIT 影响，后 5 层可作为重复样本参考。
- 正式批量默认未开启 DeepGEMM 预编译，因为预编译固定成本高于保留首层冷启动的代价。
- mixed warmup 使用同形状但不同 token id，避免污染 formal radix cache。

## 注意

- 目录中可能保留早期 smoke 或预编译试验文件；正式分析请优先读取 `formal_uniform_agg_summary.csv` 中列出的路径。
