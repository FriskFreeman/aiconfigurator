# SGLang MLA prefill shape修复数据对照

本目录核实`collect_mla.py` prefill形状修复后，仅刷新H100数据而其他硬件仍保留原表所造成的数据差异。

比较基准：

- 正确参考：`h100_sxm/sglang/0.5.9`
- 同卡旧表：`h100_sxm/sglang/0.5.10`
- 其他硬件原表：H200、B200/B300、GB200/GB300、RTX PRO 6000的SGLang 0.5.10
- 范围：基础MLA的`context_mla_perf.txt`和`generation_mla_perf.txt`，不包含MLA module

目录内容：

- [`SGLang-MLA-prefill形状修复跨硬件数据核查.md`](SGLang-MLA-prefill形状修复跨硬件数据核查.md)：结论与证据文档
- [`analyze_mla_shape_fix.py`](analyze_mla_shape_fix.py)：精确shape/config匹配、统计与作图脚本
- [`results/matched_shape_config_points.csv.gz`](results/matched_shape_config_points.csv.gz)：50,232个逐点匹配结果
- [`results/pair_coverage_summary.csv`](results/pair_coverage_summary.csv)：每个表的shape/config覆盖情况
- [`results/hardware_latency_summary.csv`](results/hardware_latency_summary.csv)：P10/P50/P90、量化组合、small/large和backend汇总
- `results/*.png` / `results/*.pdf`：四组可视化

复现：

```bash
python collector/'[self]doc'/debug-diff/sglang-mla-prefill-shape-fix/analyze_mla_shape_fix.py
```

严格匹配键为：

```text
mla_dtype, kv_cache_dtype, num_heads, batch_size, isl, tp_size, step
```

因此不会跨shape或跨量化配置比较。跨硬件时`kernel_source`可能不同，脚本保留并显式报告该差异，而不把它作为匹配键强行消除。
