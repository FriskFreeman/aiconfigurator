# ANALYTICAL 三档参数与 SILICON 对比报告

## 1. 实验范围

本批次在独立目录中重新执行，不改写先前 54 案例的结果：

- 模型：Llama 3.1 70B、Qwen3 32B、DeepSeek V3。
- 硬件：A100 SXM、H100 SXM、H200 SXM、B200 SXM、B300 SXM、RTX PRO 6000 Server。
- 服务：SGLang 0.5.10，仅 PD 分离模式。
- 工作负载：ISL=4096、OSL=1024、prefix=0、TTFT SLA=5000 ms、TPOT SLA=100 ms。
- 资源：通常 32 GPU；A100 DeepSeek 使用 64 GPU。
- DeepSeek：MLA/KV/FMHA 保持 BF16；A100 使用 BF16 GEMM/MoE，其他卡使用 SGLang FP8 GEMM 和 FP8-block MoE；不使用 DeepGEMM。

4 种模式为 `SILICON`、`ANALYTICAL-low`、`ANALYTICAL-standard`、`ANALYTICAL-high`，共 72 案例：

| 系列 | 成功 | 跳过 | 说明 |
| --- | ---: | ---: | --- |
| SILICON | 16 | 2 | A100/RTX PRO DeepSeek 缺 `wideep_context_mla_perf.txt` |
| ANALYTICAL-low | 18 | 0 | 低时延/高吞吐估计 |
| ANALYTICAL-standard | 18 | 0 | 默认标准估计 |
| ANALYTICAL-high | 18 | 0 | 高时延/低吞吐估计 |

H100 DeepSeek SILICON 继续使用先前的普通 MoE+PP2/PP4 容量兼容路径；H100 的 analytical 三档仍使用原 WideEP 配置。因此该组合的三模式差异包含执行路径差异，不能视为纯系数误差。

## 2. 图表语义

所有曲线对齐 AIC CLI 默认 Pareto 定义：

- 横轴 `tokens/s/user = 1000 / TPOT(ms)`，越高越好。
- 纵轴 `tokens/s/gpu_cluster`，是给定总 GPU 预算下可完整复制 replica 后的集群归一化单卡吞吐。
- 每张图对一个模型/硬件，叠加 4 条曲线。

`low/standard/high` 是时延参数档位，不是三个题目或形状查表。因此同一点的底层算子形状、负载和搜索空间不变，但时延评估改变会进一步改变可行候选和最终最优配置。因此图中三条曲线不仅是同一组点的纵向缩放。

## 3. 最高吞吐点汇总

下表是每条系列前沿中最高 `tokens/s/gpu_cluster`的点，单位为 tokens/s/GPU。`-` 表示 SILICON 缺表。

| 模型 | 硬件 | SILICON | analytical-low | analytical-standard | analytical-high |
| --- | --- | ---: | ---: | ---: | ---: |
| Llama 3.1 70B | A100 | 145.0 | 168.3 | 150.1 | 118.8 |
| Llama 3.1 70B | H100 | 309.5 | 319.9 | 291.3 | 245.5 |
| Llama 3.1 70B | H200 | 477.4 | 475.6 | 421.7 | 329.0 |
| Llama 3.1 70B | B200 | 732.4 | 953.8 | 831.6 | 635.0 |
| Llama 3.1 70B | B300 | 947.1 | 1085.0 | 881.6 | 646.3 |
| Llama 3.1 70B | RTX PRO 6000 | 101.8 | 139.6 | 121.3 | 105.8 |
| Qwen3 32B | A100 | 276.3 | 342.3 | 303.8 | 224.9 |
| Qwen3 32B | H100 | 639.4 | 664.9 | 564.8 | 456.0 |
| Qwen3 32B | H200 | 888.0 | 952.2 | 831.7 | 581.9 |
| Qwen3 32B | B200 | 1595.3 | 1806.9 | 1375.3 | 1016.8 |
| Qwen3 32B | B300 | 1827.7 | 1858.2 | 1542.8 | 1103.9 |
| Qwen3 32B | RTX PRO 6000 | 236.3 | 325.2 | 292.0 | 231.4 |
| DeepSeek V3 | A100 | - | 92.5 | 74.0 | 57.3 |
| DeepSeek V3 | H100 | 158.9* | 276.4 | 255.6 | 225.2 |
| DeepSeek V3 | H200 | 493.8 | 902.4 | 735.5 | 554.5 |
| DeepSeek V3 | B200 | 730.1 | 1608.1 | 1300.2 | 969.8 |
| DeepSeek V3 | B300 | 1104.6 | 1832.8 | 1445.6 | 1021.7 |
| DeepSeek V3 | RTX PRO 6000 | - | 40.5 | 29.5 | 18.9 |

`*` H100 DeepSeek SILICON 为普通 MoE+PP 路径，其他三档为 WideEP 路径。

在 16 个有 SILICON 对照的模型/硬件组合上，三档最优吞吐比的汇总为：

| 系列 | ratio 中位数 | ratio 范围 | median absolute error |
| --- | ---: | ---: | ---: |
| ANALYTICAL-low | 1.200 | 0.996~2.203 | 19.97% |
| ANALYTICAL-standard | 1.067 | 0.844~1.781 | 13.66% |
| ANALYTICAL-high | 0.817 | 0.604~1.417 | 24.68% |

这里是独立搜索后的最优点比较，不是相同并行配置的逐点误差。对 DeepSeek，无论哪一档都会受 MLA、MoE、EP 和 WideEP 执行路径差异影响，不应将这些数值解释为单纯档位校准误差。

## 4. 产物索引

- `results/run_summary.csv`：72 个案例的状态。
- `results/pareto_frontiers.csv`：70 个成功序列的前沿点。
- `results/best_frontier_points.csv`：各系列的最高吞吐点。
- `figures/pareto_frontiers/`：18 张四系列叠加曲线。
- `figures/best_throughput_by_estimate_level.png`：三个模型的最高吞吐柱状图。
