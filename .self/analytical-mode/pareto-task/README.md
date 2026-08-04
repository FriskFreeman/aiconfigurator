# PD 分离 Pareto 最终验收

固定 workload：SGLang 0.5.10、ISL 4096、OSL 1024，只运行 PD 分离。一般使用 32 GPU；A100 DeepSeek BF16 因容量使用 64 GPU。

```bash
PYTHONPATH=src:src/aiconfigurator/sdk/kernelsim/fa \
  conda run -n ljc01 python .self/analytical-mode/pareto-task/run_pareto.py
```

结果按 `模型__硬件__模式` 写入 `results/`。`status.json` 使脚本可断点续跑；SILICON 缺表记为 skipped，EMPIRICAL/ANALYTICAL 的任何失败均记为验收失败。

DeepSeek V3 的 prefill/decode MLA、KV cache 和 FMHA 固定 BF16；A100 及 H100 的 EMPIRICAL/ANALYTICAL 因 80 GB 容量使用 WideEP，其他平台使用普通 DeepSeek 路径。H100 SILICON 因缺少 DeepEP module 表，使用普通 MoE 表及 `TP8×PP2/PP4×DP1` 容量兼容搜索。A100 使用 BF16 GEMM/MoE；其他平台使用普通 FP8 SGLang GEMM（不是 DeepGEMM）和模型推荐的 FP8-block MoE。通信默认使用无表 empirical 策略。

汇总结果并生成最高吞吐对比图及 18 张 AIC 默认语义的 `tokens/s/user`–`tokens/s/gpu_cluster` Pareto 前沿图：

```bash
PYTHONPATH=src:src/aiconfigurator/sdk/kernelsim/fa \
  conda run -n ljc01 python .self/analytical-mode/pareto-task/analyze_results.py
```

前沿图写入 `figures/pareto_frontiers/`，对应数据写入 `results/pareto_frontiers.csv`。
