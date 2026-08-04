# Pareto 吞吐与 TPS 指标的推导及 TTFT/TPOT 可逆性说明

## 1. 结论先行

AIC 默认 Pareto 图的两个坐标是：

- 横轴 `tokens/s/user`：单用户生成速率，由 decode TPOT 直接得到，`tokens/s/user = 1000 / TPOT(ms)`。
- 纵轴 `tokens/s/gpu_cluster`：在给定集群 GPU 预算下，考虑完整 replica 数量后的平均单卡输出吞吐。它不能只由 TTFT/TPOT 得到，还依赖 batch、prefill/decode request throughput、worker 数、rate matching、并行配置和 GPU 预算。

因此：

1. 仅从图上的横坐标可以反推 TPOT：`TPOT(ms) = 1000 / tokens/s/user`。
2. 默认图的两个坐标都不显式包含 TTFT，所以不能仅凭曲线唯一反推 TTFT。
3. 本实验的 `results/pareto_frontiers.csv` 保留每个前沿点的 `ttft`、`tpot`、`request_latency` 和完整并行配置，因此从 CSV 而非从 PNG 可以无损回溯一手仿真结果。

## 2. 从算子时延到 worker 级 TTFT/TPOT

SILICON、EMPIRICAL 和 ANALYTICAL 的差异发生在底层算子时延来源：查实测表、旧经验式或新解析模型。进入 backend 之后，三者共用同一套系统级聚合逻辑。

对 SGLang worker，backend 先将每个模型层的 GEMM、attention/MLA、MoE、BMM、通信和其他操作时延组成 step latency，再根据 batch、ISL、OSL 和 prefill/decode 步数得到：

- `TTFT`：首 token 前的 prefill 路径时间，PD 分离聚合时取 prefill worker 的 TTFT。
- `TPOT`：decode 每个后续 token 的平均时间，PD 分离聚合时取 decode worker 的 TPOT。
- worker `seq/s`：该 worker 每秒可完成的 request/sequence 数。

对 aggregated serving，SGLang backend 还会根据 mix/gen-only steps 计算加权 TPOT 和 output throughput。本轮验收只使用 PD 分离，下文专门说明 disaggregated 公式。

### 2.1 Prefill `seq/s/worker` 如何得到

prefill worker 使用 `static_ctx` 路径。这时只有 context latency，因此 worker 刚刚完成一批 prefill 请求所需的时间就是该候选的原始 TTFT：

```text
raw_prefill_ttft = sum(context operator latencies)
```

`BaseBackend.run_static()` 中的 worker request throughput 为：

```text
prefill seq/s/worker
  = global_batch / raw_prefill_ttft(ms) * 1000 * PP

global_batch = local_batch * attention_DP
```

也就是：

```text
prefill seq/s/worker
  = local_batch * attention_DP * PP * 1000
    / raw_prefill_ttft(ms)
```

所以“prefill throughput 与 TTFT 成倒数”这个理解是对的，但需要补上三个条件：

1. 分子不是 1，而是该 worker 一批完成的 global request 数。
2. AIC 额外乘以 PP，把 pipeline stage 的稳态重叠吞吐与单批端到端时延区分开。TP 和 MoE-EP 不再作为该公式的独立乘数，因为它们是同一 worker 内协同执行一批请求，影响已体现在算子时延和 `raw_prefill_ttft` 中。
3. PD 候选选择阶段会将 prefill TTFT 额外乘以默认 `1.8` 的并发排队修正，但不会同步重算 worker `seq/s`。

因此最终 PD CSV 中显示的 TTFT 是：

```text
reported_prefill_ttft = raw_prefill_ttft * 1.8
```

而 `(p)seq/s/worker` 仍基于 `raw_prefill_ttft` 计算。若直接用最终 CSV 的 TTFT 做倒数，会少算 1.8 倍。在当前默认修正下，可写为：

```text
prefill seq/s/worker
  = global_batch * PP * 1000 * 1.8
    / reported_prefill_ttft(ms)
```

这个 `1.8` 是 AIC 对并发 prefill 排队的经验修正，不是底层算子模型直接产生的硬件时延。其目前为固定值，是系统级准确性的一个明确局限。

### 2.2 Decode `seq/s/worker` 如何得到

decode worker 使用 `static_gen` 路径。AIC 先聚合整段后续 token 的 generation latency：

```text
generation_latency = TPOT * (OSL - 1)
```

然后按一批完成的 request 数计算：

```text
decode seq/s/worker
  = global_batch / generation_latency(ms) * 1000 * PP

  = global_batch * PP * 1000
    / (TPOT(ms) * (OSL - 1))
```

与 prefill 相同，`global_batch = local_batch * attention_DP`。decode 的单用户 token rate 是：

```text
tokens/s/user = 1000 / TPOT(ms)
```

所以 decode request throughput 也可表达为：

```text
decode seq/s/worker
  = tokens/s/user * global_batch * PP / (OSL - 1)
```

这个关系很重要：TPOT 描述一个用户生成一个 token 的间隔，而 `seq/s/worker` 描述 worker 每秒完成多少条长度为 OSL 的请求。二者中间必须通过 batch 并发数和 `OSL-1` 个 decode interval 转换，不能直接等同。

### 2.3 用 H100 DeepSeek 前沿点验算

仍取后文的 H100 DeepSeek SILICON 前沿点。Prefill 字段为：

```text
reported TTFT      = 404.051 ms
prefill local bs   = 1
attention DP       = 1
PP                 = 2
```

先去掉 PD 排队修正：

```text
raw_prefill_ttft = 404.051 / 1.8 = 224.473 ms
```

再计算 worker throughput：

```text
prefill seq/s/worker
  = 1 * 1 * 2 * 1000 / 224.473
  = 8.910 seq/s
```

与 CSV 的 `(p)seq/s/worker = 8.910` 一致。Decode 字段为：

```text
TPOT               = 34.778 ms
decode local bs    = 96
attention DP       = 1
PP                 = 2
OSL                = 1024
```

因此：

```text
decode seq/s/worker
  = 96 * 1 * 2 * 1000 / (34.778 * 1023)
  = 5.397 seq/s
```

与 CSV 的 `(d)seq/s/worker = 5.397` 一致。PD 最后对两侧容量乘以 worker 数和 `0.9/0.92` 衰减，再取两者最小值：

```text
min(8.910 * 1 * 0.90,
    5.397 * 1 * 0.92)
= min(8.019, 4.965)
= 4.965 seq/s
```

该点因此是 decode-bound，最终 `request_rate = 4.965 seq/s`。

## 3. PD 分离下的吞吐聚合

设：

- `Rp`、`Rd`：单个 prefill/decode worker 的 `seq/s`。
- `Wp`、`Wd`：prefill/decode worker 数。
- `ep=0.9`、`ed=0.92`：当前默认 rate-matching 衰减，分别表示 prefill pipeline bubble 和 decode batch slot 未完全饱和。
- `Gp`、`Gd`：单个 prefill/decode worker 使用的 GPU 数，均为 `TP * PP * DP`。
- `Lout`：OSL。

AIC 先做 prefill/decode 速率匹配：

```text
request_rate = seq/s
             = min(Rp * Wp * ep,
                   Rd * Wd * ed)
```

这表示 PD 系统的请求吞吐受较慢的一侧限制。当前实现的 token 吞吐为：

```text
tokens/s = request_rate * OSL
```

注意这里在 `_build_disagg_summary_dict()` 中使用完整 `OSL`，而 worker 内部某些生成吞吐公式使用 `OSL-1`。这是 AIC 现有字段口径，在对外比较时应保持一致，不要在后处理中自行替换。

PD replica 的 GPU 数为：

```text
replica_gpus = Gp * Wp + Gd * Wd
```

所以 replica 内原始单卡吞吐是：

```text
tokens/s/gpu = tokens/s / replica_gpus
```

## 4. Pareto 两轴如何得到

### 4.1 横轴：单用户 TPS

SGLang backend 直接定义：

```text
tokens/s/user = 1000 / TPOT(ms)
```

它不是整个集群的 token throughput，而是单个正在 decode 的用户所看到的生成速率。例如 TPOT=20 ms 对应 50 tokens/s/user。用倒数作横轴后，用户体验和系统吞吐都可统一表达为 higher-is-better。

### 4.2 纵轴：集群预算归一化吞吐

设 `Gbudget` 为给定的总 GPU 预算，可以部署的完整 replica 数为：

```text
replicas = floor(Gbudget / replica_gpus)
```

CLI picking 和命令行 Pareto 图使用：

```text
tokens/s/gpu_cluster
  = tokens/s/gpu * replicas * replica_gpus / Gbudget
```

等价于：

```text
tokens/s/gpu_cluster
  = request_rate * OSL * floor(Gbudget / replica_gpus) / Gbudget
```

若 `replica_gpus` 整除 `Gbudget`，则 `tokens/s/gpu_cluster = tokens/s/gpu`。否则剩余 GPU 无法构成完整 replica，纵轴会对这部分空闲资源扣减。

## 5. TTFT、TPOT 与 request latency

PD 聚合后的端到端请求时延定义为：

```text
request_latency(ms) = TTFT + TPOT * max(OSL - 1, 0)
```

首 token 在 TTFT 结束时已经产生，因此后续只有 `OSL-1` 个 token interval。

在默认 Pareto 图中，TTFT 的作用主要是候选过滤：prefill 候选必须满足 TTFT SLA，但 TTFT 不是图上任一轴的自变量。TPOT 一方面用于 decode 候选过滤，另一方面通过倒数直接构成横轴。

若 CLI 显式指定 request-latency SLA，AIC 才会把 Pareto 横轴改为 `request_latency`、目标改为越小越好。该模式与本轮默认 `tokens/s/user` 横轴不同。

## 6. H100 DeepSeek SILICON 点的实际验算

取当前 H100 DeepSeek 普通 MoE SILICON 前沿中的一点：

```text
TTFT                    = 404.051 ms
TPOT                    = 34.778 ms
OSL                     = 1024
prefill seq/s/worker    = 8.910
decode seq/s/worker     = 5.397
prefill/decode workers  = 1 / 1
replica_gpus            = 32
Gbudget                 = 32
```

单用户生成速率：

```text
1000 / 34.778 = 28.754 tokens/s/user
```

PD rate matching：

```text
prefill capacity = 8.910 * 1 * 0.90 = 8.019 seq/s
decode capacity  = 5.397 * 1 * 0.92 = 4.965 seq/s
request_rate     = min(8.019, 4.965) = 4.965 seq/s
```

整个 replica 的 token throughput：

```text
tokens/s = 4.965 * 1024 = 5084.2 tokens/s
```

CSV 因中间字段保留三位小数而记录为 `5084.406`，差异来自上述展示值已经四舍五入。单卡与集群归一化吞吐为：

```text
tokens/s/gpu = 5084.406 / 32 = 158.888
tokens/s/gpu_cluster = 158.888
```

因为 32-GPU replica 恰好整除 32-GPU 预算，没有空闲 GPU 惩罚。端到端时延为：

```text
404.051 + 34.778 * 1023 = 35981.945 ms
```

这与 `pareto_df.csv` 和 `pareto_frontiers.csv` 中的字段一致。

## 7. 能否从 Pareto 图反推一手仿真结果

### 7.1 只有 PNG/曲线坐标

**可以反推：**

- TPOT：`TPOT = 1000 / x`。受制图像读数精度和 CSV 三位小数影响。
- 若额外已知 OSL、总 GPU 预算和 replica GPU 数，可从纵轴反推集群 request rate：

```text
request_rate
  = y * Gbudget / (OSL * floor(Gbudget / replica_gpus))
```

**不能唯一反推：**

- TTFT：默认两轴都不包含 TTFT。
- prefill/decode 各自的算子时延和 worker `seq/s`：纵轴只保留了两侧容量的 `min()` 结果。
- worker 数、TP/PP/DP/EP、batch 和内存：许多不同配置可以投影到相同或接近的二维坐标。
- silicon/empirical/analytical 的算子来源和分解：图上只显示最终投影。

### 7.2 有 request latency 时

如果已知同一点的 `request_latency`、OSL 和从横轴得到的 TPOT，可以反推：

```text
TTFT = request_latency - TPOT * (OSL - 1)
```

但默认 Pareto PNG 没有展示 request latency，因此仍需要 CSV 或图中额外的 hover/annotation 信息。

### 7.3 有当前导出 CSV 时

`results/pareto_frontiers.csv` 不是只保存 `(x,y)` 的简化表。它从 AIC 搜索返回的 `pareto_df.csv` 保留了：

- `ttft`、`tpot`、`request_latency`。
- `request_rate`、`tokens/s`、`tokens/s/gpu`、`tokens/s/user`、`tokens/s/gpu_cluster`。
- prefill/decode batch、worker 数、单 worker `seq/s`。
- TP/PP/DP/MoE-TP/MoE-EP、精度、backend、system 和内存。

因此可以将前沿上的每个点精确关联回具体仿真配置和 TTFT/TPOT。不过它只包含二次提取后的前沿点；要分析 AIC 返回但未进入最终前沿的点，应读取每个案例目录中的 `pareto_df.csv`。还需注意，`disagg_pareto()` 内部已执行 SLA 过滤、parallel category 选优和 top-k 截断，因此 `pareto_df.csv` 也不是所有枚举中间点的无损全集。

## 8. 数据层级与分析建议

建议将本轮产物按三个层级理解：

1. `results/<case>/pareto_df.csv`：AIC PD 搜索经内部筛选后的返回集，适合查找 batch、并行配置和未进入最终二维前沿的返回点。
2. `results/pareto_frontiers.csv`：按 AIC CLI 默认双高目标提取的前沿点，仍保留 TTFT/TPOT 和完整配置。
3. `figures/pareto_frontiers/*.png`：二维汇报投影，适合观察模式差异和 trade-off，不适合单独作为一手结果归档。

对后续验收，应同时报告 Pareto 坐标与点对应的 TTFT、TPOT、request latency、batch 和 parallel config。这样既保留系统寻优结论，也能回到底层仿真结果定位模型误差。

## 9. 现有结果是否全部 decode-bound

### 9.1 结论

不是。按当前代码的实际容量定义：

```text
Cp = (p)seq/s/worker * (p)workers * 0.90
Cd = (d)seq/s/worker * (d)workers * 0.92
request_rate = min(Cp, Cd)
```

对当前 52 个成功案例的 CSV 逐点检验得到：

| 数据层级 | 总点数 | Decode-bound | Prefill-bound |
| --- | ---: | ---: | ---: |
| AIC `pareto_df.csv` 返回点 | 3209 | 2130 (66.38%) | 1079 (33.62%) |
| 最终 Pareto 前沿点 | 809 | 644 (79.60%) | 165 (20.40%) |

三种模式的最终前沿中也都有反例：

| 模式 | Decode-bound | Prefill-bound | Prefill-bound 比例 |
| --- | ---: | ---: | ---: |
| ANALYTICAL | 225 | 57 | 20.21% |
| EMPIRICAL | 214 | 44 | 17.05% |
| SILICON | 205 | 64 | 23.79% |

更值得注意的是，对每条曲线取最高 `tokens/s/gpu_cluster` 端点后，52 个端点中有 30 个 prefill-bound、22 个 decode-bound。这并不表示 prefill 普遍严重不足：这些端点的 `Cp/Cd` 中位数为 `0.9965`，大部分已经非常接近理想速率平衡，只是由于离散选择而落在 1 的任意一侧。

### 9.2 对 1.8 TTFT 修正的准确理解

“最终统计又取原始 TTFT”这一点需要修正。PD 候选选择阶段执行：

```text
reported TTFT = raw worker TTFT * 1.8
```

修正后的 TTFT 会被用于 TTFT SLA 过滤，并作为最终 `ttft` 和 `request_latency` 的组成部分写入 CSV。所以最终 TTFT 并非原始值。

但 `(p)seq/s/worker` 在乘 1.8 之前已经根据 raw worker TTFT 计算完成，1.8 修正不会回写或降低 Rp。源码注释也明确说明：该修正“only TTFT will be corrected here, other latency and throughput will not be corrected”。因此它的实际效果是：

- 更严格地过滤 prefill TTFT SLA。
- 增大最终 request latency。
- 不改变 prefill capacity `Cp`。
- 不会从数学上保证 `Cp > Cd`，因此不会保证所有点 decode-bound。

这也暴露了一个建模边界：当前 1.8 被解释为并发 prefill 排队时延，而非实际服务能力降低。如果未来认为该因子也代表 prefill 服务率损失，就需要同步修正 Rp；但这会改变 worker matching 和 Pareto 结果，不应在没有实机系统数据支持时直接修改。

### 9.3 为什么会保留 prefill-bound 反例

`DisaggInferenceSession._match_workers()` 遍历 prefill/decode worker 数，其选择目标是：

```text
maximize min(Cp, Cd) / replica_gpus
```

代码中原本存在的“必须使 prefill throughput 大于 decode throughput”判断已被注释，因此 matcher 并不强制 `Cp >= Cd`。这是 prefill-bound 反例能够进入结果的直接代码原因。

从优化原理看，这种行为是合理的。batch、worker 数、TP/PP/DP/EP 和允许的 replica GPU 数都是离散值。如果当前组合为 `Cp < Cd`，再添加一个 prefill worker 虽然可能使系统改为 decode-bound，但也会增加 replica GPU 分母，最终 `min(Cp,Cd)/GPU` 反而可能下降或不再满足可选 GPU 总数。因此吞吐/GPU 最优点可以合理地停在平衡点的 prefill 一侧。

现有前沿点还有两个具体特征：

- 809 个点的 prefill batch 全部为 1。本轮 ISL=4096，prefill 候选排序又偏好高 `seq/s/gpu` 和较小 global batch，使 prefill 容量主要靠 worker 数而非 batch 连续调节。
- Decode batch 可在更大范围离散变化，并通过 TPOT 与 worker 数共同改变 `Cd`。两侧调节粒度不对称，也会使最终点落在 `Cp/Cd=1` 的两侧。

少数反例偏离平衡点较明显。例如 H100 DeepSeek ANALYTICAL 最高吞吐点为：

```text
Cp = 7.987 seq/s
Cd = 10.691 seq/s
Cp/Cd = 0.747
request_rate = 7.987 seq/s
```

该点是明显的 prefill-bound，不是小数舍入造成的假反例。RTX PRO 6000 上也有 `Cp/Cd≈0.70~0.77` 的前沿点，与该平台较强的跨架构外推不确定性同时存在，应结合底层 prefill 算子分解进一步判断。

### 9.4 数据产物

本次逐点分类由 `analyze_rate_matching.py` 可重建，输出：

- `results/rate_matching_returned_points.csv`：3209 个 AIC 返回点的 Cp/Cd、瓶颈和 request-rate 复算。
- `results/rate_matching_frontier_points.csv`：809 个最终前沿点的同类分析。
- `results/rate_matching_bottleneck_summary.csv`：按数据层级、模型、硬件和模式汇总的瓶颈计数。
