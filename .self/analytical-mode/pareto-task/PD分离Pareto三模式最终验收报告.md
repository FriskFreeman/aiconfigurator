# PD 分离 Pareto 三模式最终验收报告

## 1. 验收结论

本轮完成 3 个模型、6 种硬件、3 种 PerfDatabase mode，共 54 个 **PD 分离** Pareto 案例：

| 模式 | 成功 | 跳过 | 失败 |
| --- | ---: | ---: | ---: |
| SILICON | 16 | 2 | 0 |
| EMPIRICAL | 18 | 0 | 0 |
| ANALYTICAL | 18 | 0 | 0 |

核心验收目标达成：旧 EMPIRICAL 和新 ANALYTICAL 在全部硬件、模型上均能完成 Pareto 搜索，不依赖对应算子实测表。SILICON 仅在缺少必要表时跳过，没有把缺表伪装为估计结果。

在 16 个有 SILICON 对照的模型/硬件组合中：

- EMPIRICAL/SILICON 最优吞吐比中位数为 **1.190**，范围 `1.011~1.998`。
- ANALYTICAL/SILICON 最优吞吐比中位数为 **1.013**，范围 `0.814~1.608`。
- 新模型整体消除了旧 empirical 明显偏乐观的中心偏差，但架构/backend 外推尾部仍较大。

这里比较的是各模式独立 Pareto 搜索得到的最优点，不是相同并行配置的逐点误差。模式改变可能使 TP、EP、worker 数和 batch 一同变化，因此该指标回答的是“最终寻优决策有多大差异”，不能替代算子级准确率。

本轮运行使用 AIC 原生 `TaskRunner.run()` 链路；PD 分离路径进一步调用 `pareto_analysis.disagg_pareto()` 枚举 prefill/decode 并行配置、worker 数和 batch。每个案例的 `pareto_df.csv` 保留了这一原生搜索结果，不是本验收脚本自行构造的候选点。

本轮未显式设置 `request_latency` SLA，因此与 AIC CLI 默认功能对齐后，Pareto 横轴为 `tokens/s/user = 1000 / TPOT(ms)`，表示单用户生成速率，越高越好；纵轴为 `tokens/s/gpu_cluster`，表示按给定总 GPU 预算可完整复制 replica 后的单卡吞吐，也是越高越好。其归一化公式为 `tokens/s/gpu * floor(total_gpus / replica_gpus) * replica_gpus / total_gpus`。

汇总脚本复用 AIC `get_pareto_front()` 对上述两轴同时最大化，为每个模型/硬件组合生成三模式叠加前沿。`tokens/s/user` 是 TPOT 的倒数表达，所以这仍是用户时延与系统吞吐的折中，只是横轴方向统一为 higher-is-better。只有在 CLI 显式给定 `request_latency` 时，原生流程才会把横轴切换为端到端请求时延并执行最小化。

AIC 当前两种原生展示还有一个实现差异：命令行 ASCII Pareto 图由 `draw_pareto_to_string()` 绘制，纵轴使用 picking 阶段生成的 `tokens/s/gpu_cluster`；`report_and_save.py` 中另外保存的 PNG 却使用原始 `tokens/s/gpu`。当 replica GPU 数可以整除总预算时二者相同，不能整除时 `tokens/s/gpu_cluster` 会反映剩余 GPU 无法组成完整 replica 的损失。本验收图对齐用户所指的命令行输出，因此使用 `tokens/s/gpu_cluster`。

## 2. 固定口径

- Backend：SGLang 0.5.10。
- Serving：仅 disaggregated prefill/decode，不运行 aggregated 路径。
- Workload：ISL=4096、OSL=1024、prefix=0。
- SLA：TTFT 5000 ms、TPOT 100 ms。
- GPU 预算：一般为 32；A100 DeepSeek BF16 因容量改为 64。
- 模型：Llama 3.1 70B、Qwen3 32B、DeepSeek V3。
- Llama/Qwen3 使用模型推荐配置，本轮两个未量化模型均解析为 BF16 GEMM/attention。
- DeepSeek MLA/KV/FMHA 固定 BF16；A100 使用 BF16 GEMM/MoE，其他平台使用普通 FP8 SGLang GEMM和 FP8-block MoE。
- ANALYTICAL 使用 `standard + fa2 + sglang FP8 GEMM + empirical communication`。

## 3. 最优吞吐结果

单位为 output tokens/s/GPU；`-` 表示 SILICON 因缺表跳过。

| 模型 | 硬件 | SILICON | EMPIRICAL | ANALYTICAL | Analytical/Silicon |
| --- | --- | ---: | ---: | ---: | ---: |
| Llama 3.1 70B | A100 | 145.0 | 171.6 | 150.7 | 1.039 |
| Llama 3.1 70B | H100 | 358.0 | 371.3 | 291.3 | 0.814 |
| Llama 3.1 70B | H200 | 477.4 | 501.5 | 421.7 | 0.883 |
| Llama 3.1 70B | B200 | 843.0 | 1008.3 | 831.6 | 0.986 |
| Llama 3.1 70B | B300 | 947.1 | 1088.9 | 881.6 | 0.931 |
| Llama 3.1 70B | RTX PRO 6000 | 101.8 | 144.9 | 121.3 | 1.192 |
| Qwen3 32B | A100 | 276.3 | 311.8 | 303.8 | 1.100 |
| Qwen3 32B | H100 | 671.1 | 678.4 | 564.8 | 0.842 |
| Qwen3 32B | H200 | 888.0 | 975.1 | 831.7 | 0.937 |
| Qwen3 32B | B200 | 1595.3 | 1933.5 | 1476.5 | 0.926 |
| Qwen3 32B | B300 | 1827.7 | 1929.3 | 1542.8 | 0.844 |
| Qwen3 32B | RTX PRO 6000 | 236.3 | 343.7 | 293.5 | 1.242 |
| DeepSeek V3 | A100 | - | 90.8 | 74.0 | - |
| DeepSeek V3 | H100 | 158.9 | 259.5 | 255.6 | 1.608* |
| DeepSeek V3 | H200 | 563.9 | 840.0 | 735.5 | 1.304 |
| DeepSeek V3 | B200 | 973.5 | 1825.7 | 1402.6 | 1.441 |
| DeepSeek V3 | B300 | 1104.6 | 2206.7 | 1621.2 | 1.468 |
| DeepSeek V3 | RTX PRO 6000 | - | 19.4 | 29.5 | - |

## 4. 关键观察

### 4.1 Dense LLM

Llama/Qwen 的结果具有一致结构：

- A100 上 analytical 接近或略高于 silicon，分别为 `+3.9%/+10.0%`。
- H100/H200/B200/B300 上 analytical 多数偏保守。Llama H100 为 `-18.6%`；B200 最接近，Llama/Qwen 分别为 `-1.4%/-7.4%`；B300 分别为 `-6.9%/-15.6%`。
- RTX PRO 6000 上 analytical 偏乐观 `+19.2%/+24.2%`。该设备是 SM120，SGLang attention 使用 Triton，而模型仍套用 FA2 抽象；其 L2 容量/带宽也包含估计输入，因此不能把偏差单独归因于效率参数。
- 旧 empirical 除 H100 外普遍偏乐观，在 RTX PRO 上达到 `+42.4%/+45.4%`。这符合旧模型固定效率、弱化小 kernel/启动成本的已知问题。

### 4.2 DeepSeek V3

DeepSeek 比 dense LLM 更难外推：它同时包含 FP8 projection、MLA、BMM、MoE、EP/DP 和通信。H200/B200/B300 analytical 最优吞吐分别比 SILICON 高 `30.4%/44.1%/46.8%`，但仍比旧 empirical 的 `49.0%/87.5%/99.8%` 明显收敛。B300 结果表明 Blackwell Ultra 上的硬件峰值更新未解决 DeepSeek 路径的系统级建模偏差。

H100 的 `1.608*` 不是严格同路径准确率：EMPIRICAL/ANALYTICAL 保留 WideEP + DeepEP，SILICON 因缺少 `wideep_deepep_normal_perf.txt` 改用普通 SGLang MoE 表。普通 DeepSeek 在 PP1 下最少需约 `90 GB/GPU`，无法装入 H100 80GB，因此 silicon 搜索显式使用 `TP8×PP2/PP4×DP1`，实际候选内存约 `47~53 GB/GPU`。该结果证明普通 MoE silicon 数据链路可执行，但差值同时包含 WideEP/DeepEP、PP 和搜索空间差异。

差异不能仅解释为 MLA core：最优配置发生了变化。例如 B200 SILICON 最优 prefill 使用 `tp1-dp8-etp8-ep1`，analytical 使用 `tp1-dp8-etp1-ep8`。这说明 MoE compute、dispatch/communication和负载分布抽象会改变 EP/ETP 选择，是后续系统级校准的优先问题。

RTX PRO 6000 上 analytical 反而比 empirical 高 51.8%。该点没有 SILICON ground truth，且使用 SM120 外推的 MLA/BMM/MoE 与 PCIe 级通信配置，只能作为“路径可执行”验收，不能作为准确性结论。

## 5. SILICON 跳过原因

- A100 + DeepSeek：没有 SGLang 0.5.10 `wideep_context_mla_perf.txt`。
- RTX PRO 6000 + DeepSeek：没有 WideEP context MLA module 表。

H100 通过普通 MoE + PP 兼容路径完成 SILICON，H200 与 B200 DeepSeek SILICON 也已成功。历史 WideEP module 表把实际 BF16 MLA 记录为 FP8 dtype，本轮在 Task capability validation 和 PerfDatabase silicon key lookup 两层做兼容映射；worker config、SOL 和 analytical 计算仍保持 BF16。B200 的 SGLang module 表使用 `trtllm_mla` key，也已按表内实际 backend 选择。

## 6. 硬件输入与外推限制

H200 复用 GH100 compute/cache hierarchy，HBM 和峰值由 H200 systems YAML 覆盖。B200 使用 NVIDIA Blackwell Tuning Guide 发布的 228 KiB/SM shared memory 和 126 MiB L2。B300 在缺少公开 cache 实测的情况下复用 B200 的 148 SM、1.83 GHz、228 KiB/SM 和 126 MiB L2 基线，峰值和 HBM 由 B300 systems YAML 覆盖。RTX PRO 6000 使用 CC12.0 的 128 KiB/SM。

NVIDIA 产品资料不公开持续 L2 带宽，因此 B200/RTX PRO 的 L2 bandwidth 是按 H100 的 `SM*clock` 比例缩放的建模输入。RTX PRO 的 96 MiB L2也需要目标设备 CUDA 属性复核。完整来源和估算边界见 `HARDWARE_SOURCES.md`。

B300 已纳入功能验收，但其 cache 层级仍是 B200 迁移假设，不是 B300 实测值。因此 B300 analytical 结果可用于验证新硬件无表路径的完整性，其绝对精度结论需等待真实设备属性和算子数据校准。

## 7. 产物索引

- `results/run_summary.csv`：54 个案例状态。
- `results/best_points.csv`：52 个成功案例的最优点和完整配置。
- `results/pareto_frontiers.csv`：按 AIC CLI 默认的单用户 TPS 与集群归一化单卡吞吐提取的全部前沿点。
- `results/<case>/pareto_df.csv`：每个案例的原始 Pareto 点。
- `results/<case>/task.yaml`：可复查的最终 TaskConfig。
- `figures/best_throughput_by_mode.png`：三模型的模式/硬件吞吐柱状图。
- `figures/throughput_ratio_vs_silicon.png`：可比案例相对 SILICON 比值。
- `figures/acceptance_status.png`：54 案例通过/跳过矩阵。
- `figures/pareto_frontiers/`：18 张模型×硬件的三模式 AIC 原生语义 Pareto 前沿图。
- `run_pareto.py`、`analyze_results.py`：单入口运行与汇总脚本。

## 8. 最终判断

ANALYTICAL 已满足“没有实机算子表也能完成 Llama、Qwen3、DeepSeek V3 跨六类硬件 PD Pareto 搜索”的功能验收。纳入 H100 普通 MoE silicon 兼容基线后，在 16 个有 SILICON 结果的组合上，旧 empirical 和 analytical 的最优吞吐比中位数分别为 1.190 和 1.013。H100 DeepSeek 并非同路径对照，解读整体统计时必须保留该限制。

它尚未达到所有平台可直接替代 SILICON 的准确性验收：DeepSeek EP/MoE 系统语义和 SM120 attention backend 是两个最明确的剩余风险。生产接入时应保留结果来源与外推等级，并避免把无校准 RTX PRO/DeepSeek 结果呈现为高置信度预测。
