# Inference Session 解析总结

本文已按 `v0.9.0` 修订。`InferenceSession` / `DisaggInferenceSession` 的总体定位变化不大，但它们所处的 SDK 上下游环境在 0.9.0 明显增强了，因此理解时需要把新引入的 database mode、Rust estimator 和结果 provenance 一起纳入。

## 1. 快速概括

`src/aiconfigurator/sdk/inference_session.py` 仍然是 SDK 仿真阶段的门面协调层。

它最核心的作用没有变：

- 持有 `Model`
- 持有 `PerfDatabase`
- 持有 `Backend`
- 向上提供统一的单点评估、聚合评估和分离式评估接口

可以继续把它理解成：

- `Model` 负责提供算子图纸
- `PerfDatabase` 负责提供算子性能
- `Backend` 负责提供框架时序和显存规则
- `InferenceSession` 负责把三者组装成一次完整评估

## 2. 到 0.9.0 仍然成立的主线

### 2.1 `InferenceSession` 仍然主要服务同构/聚合评估

它继续负责：

- `run_static()`
- `run_agg()`
- `find_best_agg_result_under_constraints()`

### 2.2 `DisaggInferenceSession` 仍然主要服务 Prefill/Decode 分离评估

它继续负责：

- 候选 worker 生成
- P/D 双端静态评估
- 速率匹配
- 在约束下挑最优前后端组合

所以就主职责而言，0.9.0 并没有推翻 session 这一层的定位。

## 3. 0.9.0 下需要补充的新理解

### 3.1 Session 的结果对象现在“信息更厚”

在 0.8.0 的理解里，session 更像是把数值结果装进 `InferenceSummary`。

到了 0.9.0，这个 summary 里除了：

- latency
- throughput
- memory
- energy / power

还会携带：

- context/source breakdown
- generation/source breakdown
- per-op source breakdown

也就是说，session 返回的不再只是“结果值”，而是“结果值 + 来源语义”。

### 3.2 Session 所调度的 backend 现在可能走 Rust 估算路径

session 自己不直接调用 Rust estimator，但它调度的 backend 可能会。

因此在 0.9.0 中，session 所触发的评估链路实际上多了一层隐含分支：

- Python op-by-op 路径
- Rust engine-step 路径

这由 `runtime_config.engine_step_backend` 决定。

所以如果你后面发现同一个 `run_static()` 在不同配置下走出的明细结构不同，这往往不是 session 变了，而是底下 backend 的估算后端变了。

### 3.3 Session 现在更容易和 estimate-only / 非纯 silicon 模式协作

虽然这些逻辑主要落在 `task.py` 和 `perf_database.py`，但从使用视角看，0.9.0 的 session 所依赖的数据库对象不再要求一定有完整 silicon 数据。

这会影响两个理解点：

1. session 现在服务的不只是“完整实测库驱动”的评估
2. 它也可以承接 estimate-only / HYBRID / EMPIRICAL 路径下的数据库实例

因此 session 的外部语境变宽了。

### 3.4 分离式会话的价值在 0.9.0 仍然很高

这一点反而应该继续强调：

`DisaggInferenceSession` 依旧是把仿真从“单点配置预测”推进到“前后端资源组合优化”的关键层。

哪怕底下引入了 Rust estimator 或更复杂的数据来源标签，分离式 session 的核心职责仍然是：

- 生成候选
- 过滤 OOM / SLA 不合格项
- 做 rate matching
- 输出最优组合

## 4. 0.9.0 相关的外围增强，虽然不全在本文件里，但会直接影响 session 使用

### 4.1 `task.py` 对 session 的装配能力增强了

0.9.0 的任务装配层比 0.8.0 更强，新增或更明确支持了：

- `database_mode`
- `engine_step_backend`
- `allow_missing_data`
- 更稳健的 database version 选择

这些变化意味着，session 本身虽然没大改，但被创建出来的上下文比以前更复杂了。

### 4.2 `pick_default(..., strict_sla=True)` 改变了 session 结果被消费的方式

这不属于 session 文件本身的逻辑，但它影响了“session 输出如何进入最终推荐结果”的路径。

0.9.0 新增 `strict_sla` 后：

- session 产出的候选解会在 Pareto 前沿之前先经过更严格过滤

所以从整体 SDK 行为看，session 仍然负责“产出评估结果”，而更上层现在对这些结果做了更严格的二次筛选。

## 5. 对旧版文档理解的修正

### 5.1 旧说法：“Session 主要回传数值型 `InferenceSummary`”

现在应修正为：

- `InferenceSummary` 不只是数值容器
- 它还携带 per-op 来源与 breakdown 元数据

### 5.2 旧说法：“Session 绑定的 database 默认都是完整 silicon 数据库”

现在不再总是如此。

更准确的说法应该是：

- session 绑定的 database 可能是完整 silicon
- 也可能是 HYBRID / EMPIRICAL / estimate-only 语义下构造出来的对象

### 5.3 旧说法：“Session 的单点推演总是 backend 的 Python 实现”

现在也要补一句：

- backend 可以切到 Rust estimator 路径

## 6. 为什么这层在 0.9.0 仍然重要

哪怕这次最大的重构不在 `inference_session.py` 本身，这一层仍然很重要，因为它是 SDK 里真正把多组件拼在一起的门面。

可以说：

- `models/` 变复杂了
- `perf_database/` 变强了
- `backend/` 变得可追踪、可切 Rust 了

而 session 仍然是这些变化最终汇流的地方。

## 7. 建议阅读顺序

1. `src/aiconfigurator/sdk/inference_session.py`
2. `src/aiconfigurator/sdk/inference_summary.py`
3. `src/aiconfigurator/sdk/backends/base_backend.py`
4. `src/aiconfigurator/sdk/task.py`
5. `src/aiconfigurator/sdk/picking.py`

## 8. 一句话总结

0.9.0 的 `InferenceSession` 主职责没有根本变化，仍然是 SDK 仿真执行链路的协调门面；但它现在承接的是一个更强的下层体系，能够返回带来源语义的结果，并协同 estimate-only 数据库与 Rust 前向估算路径工作。
