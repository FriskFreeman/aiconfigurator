# SDK 0.8.0 到 0.9.0 主要改动总览

本文聚焦 `src/aiconfigurator/sdk/`，结合三类信息做总览分析：

- `v0.8.0..v0.9.0` 之间的实际代码 diff
- 官方 release 说明
- `sdk/[self]doc/` 中原有的 0.8.0 解析文档

参考官方 release：

- v0.8.0: https://github.com/ai-dynamo/aiconfigurator/releases/tag/v0.8.0
- v0.9.0: https://github.com/ai-dynamo/aiconfigurator/releases/tag/v0.9.0

## 1. 先给结论

如果只看 SDK，0.9.0 相比 0.8.0 的变化，不是“零散加点小功能”，而是一次比较明显的能力外扩和结构重排。

最重要的几条主线是：

1. 模型层从单文件重构成子包
2. 性能库层从“单框架查表”进一步演进到“可共享、可估算、可追溯”
3. backend / summary / operation 链路开始显式携带 per-op 来源信息
4. Rust core 前向估算器正式接进 SDK 主链
5. task / picking / database 装配层更适合新平台、新模型和非完整实测数据场景

所以 0.9.0 的 SDK 变化，本质上是在为“更复杂模型、更复杂数据来源、更复杂平台覆盖”做底座升级。

## 2. 代码层面最醒目的变化

从 `git diff --stat v0.8.0..v0.9.0 -- src/aiconfigurator/sdk` 看，几个变化最扎眼：

- `models.py` 被删除
- 新增 `models/` 整个子包
- 新增 `interpolation.py`
- 新增 `system_spec.py`
- 新增 `rust_engine_step.py`
- `perf_database.py`、`operations.py`、`task.py` 都有较大幅度改动

这说明 0.9.0 的 SDK 变化不只在数据内容上，也在代码组织方式上。

## 3. 分组件看主要变化

### 3.1 模型层：从单文件工厂变成可扩展子包

对应文件：

- 删除 `sdk/models.py`
- 新增 `sdk/models/__init__.py`
- `sdk/models/base.py`
- `sdk/models/helpers.py`
- 多个 family 文件

这是 0.9.0 最大的结构变化之一。

变化重点：

- 用 registry 代替集中式工厂分派
- 每个模型 family 独立成文件
- `create()` 成为每个 family 的构造入口
- helper 逻辑从主体实现中剥离

它带来的收益：

- 新增模型 family 更容易
- family 内 backend-aware 分支更清晰
- 复杂模型支持不再挤在一个大文件里

和 release 对应的新增模型能力主要包括：

- DeepSeek-V4
- DeepSeek-R1 相关支持扩展
- Kimi K2.5
- Qwen 3.5 的更完整支持
- MiniMax-M2.7 / NVFP4

### 3.2 性能库层：从大查表器演进成“可共享 + 可估算 + 可追溯”查询层

对应文件：

- `sdk/perf_database.py`
- `sdk/interpolation.py`
- `sdk/system_spec.py`
- `sdk/performance_result.py`

变化重点：

- `interpolation.py` 和 `system_spec.py` 从 `perf_database.py` 中拆出
- `PerformanceResult` 新增 `source`
- `get_database()` 支持 `allow_missing_data` 与 `database_mode`
- HYBRID 模式下支持跨 framework 的 shared-layer op data
- estimate-only 数据库实例化成为正式路径
- 新增 oneCCL 与 DeepSeek-V4 等更多模块级 perf 数据入口

这条线和 release 里几项表述直接对应：

- hybrid-mode op-data sharing
- per-op silicon vs empirical attribution
- estimate-only PCIe systems

### 3.3 backend 层：静态仿真底座不变，但执行语义更丰富

对应文件：

- `sdk/backends/base_backend.py`
- `sdk/backends/vllm_backend.py`
- `sdk/backends/sglang_backend.py`
- `sdk/backends/trtllm_backend.py`

变化重点：

- `BaseBackend` 开始在 context / generation phase 追踪 per-op source
- `InferenceSummary` 能保存这些 source dict
- 三个 backend 在 `run_agg` 内都开始透传 per-op latency/source breakdown
- 三个 backend 都接入 Rust engine-step 快路径

也就是说：

- 0.8.0 更像“给出预测值”
- 0.9.0 更像“给出预测值，并告诉你是怎么来的”

### 3.4 Rust core 接入：SDK 内核第一次出现双实现后端

对应文件：

- `sdk/rust_engine_step.py`
- `sdk/backends/base_backend.py`
- 三个具体 backend
- `sdk/task.py`

这是 0.9.0 很重要的一条新增主线。

它做的事是：

- 引入 Rust shared library 包装器
- 提供静态 context / generation 估算
- 提供 mixed step / decode step 估算
- 通过 `engine_step_backend` 在运行时切换 Python / Rust

这条线和 release 里的 “AIC Rust core forward-pass estimator” 对应。

它的意义不只是加速，还意味着 AIC 在把核心前向估算器从“只有 Python 实现”推进到“可切换多实现内核”。

### 3.5 Task / Picking / 装配层：更适合复杂模式和更严格筛选

对应文件：

- `sdk/task.py`
- `sdk/picking.py`
- `sdk/common.py`
- `sdk/utils.py`

变化重点：

- `task.py` 新增 `database_mode`、`engine_step_backend` 等上下文字段
- `task.py` 对 DeepSeek-V4 / Hopper 的不支持组合做前置校验
- 非 pure-silicon 模式下允许走 estimate-only 数据库
- `pick_default()` 新增 `strict_sla`
- `common.py` 里模型 family、系统集合、版本解析、默认模型列表都扩展了

这部分和 release 的对应点主要是：

- `--strict-sla`
- 新模型 / 新系统支持
- 版本与支持矩阵的扩展

### 3.6 operation 层：开始传播 provenance 语义

对应文件：

- `sdk/operations.py`

虽然 release 里不会单独高亮 `operations.py`，但 0.9.0 的 per-op attribution 能落地，离不开这里的改动。

这里的核心变化是：

- 大量 op 在返回 `PerformanceResult` 时显式保留或合并 `source`
- 一些复合 op 在组合多个子结果时会把来源折叠成 `"mixed"`

所以如果说 `PerfDatabase` 负责“产生 provenance”，那 `operations.py` 负责“在算子层保持 provenance 不丢失”。

## 4. 哪些变化是“新增能力”，哪些是“重构”

### 4.1 更偏新增能力的变化

- DeepSeek-V4 及更多新模型 family 支持
- Rust forward-pass estimator
- per-op silicon vs empirical attribution
- hybrid 模式跨 framework 共享 op data
- estimate-only 数据库加载
- `strict_sla`

这些适合在旧文档后面补充“0.9.0 新特性”。

### 4.2 更偏结构重构的变化

- `models.py` -> `models/` 子包
- `interpolation.py` / `system_spec.py` 从 `perf_database.py` 拆出
- `PerformanceResult` 的语义从“latency+energy”扩展到“latency+energy+source”

这些就不能只靠尾部补充，必须直接修正文档主体描述。

## 5. 对旧版理解最需要修正的几点

### 5.1 “模型层主要看 `models.py`”

这在 0.9.0 已经完全过时。

现在应该按子包来理解：

- `__init__.py`
- `base.py`
- `helpers.py`
- family 文件

### 5.2 “PerfDatabase 是纯单框架 silicon 查表器”

这也不够了。

0.9.0 下它还支持：

- HYBRID 共享层
- empirical fallback
- estimate-only 实例化
- provenance 输出

### 5.3 “Backend 主要负责算延迟和显存”

现在需要加上：

- 负责 per-op 来源汇总
- 可能切换到 Rust estimator 路径

### 5.4 “Session 返回的结果主要是数值”

现在也不够。

结果对象中已经开始带：

- per-op latency breakdown
- per-op source breakdown

## 6. 这次 SDK 升级的整体方向

如果把 0.9.0 的 SDK 升级总结成一句工程语言，大概是：

“把原本偏单体、偏单框架、偏单一路径的仿真 SDK，升级成更模块化、更可解释、也更能承载复杂模型和不完整数据环境的基础设施。”

这背后可以看到三个明显目标：

1. 支持更复杂的新模型
2. 支持更广的新平台和数据状态
3. 让预测结果更可解释

## 7. 这次已同步修订的旧文档

本轮已按 0.9.0 修订或补充：

- `sdk/[self]doc/Models.md`
- `sdk/[self]doc/PerfDatabase.md`
- `sdk/[self]doc/Base_backend.md`
- `sdk/[self]doc/Inference_session.md`
- `sdk/[self]doc/backend实现/Backends对比分析.md`

## 8. 后续最值得继续深入的阅读顺序

如果你下一步还想继续聚焦 SDK 仿真主干，建议顺序如下：

1. `sdk/task.py`
2. `sdk/inference_session.py`
3. `sdk/backends/base_backend.py`
4. `sdk/perf_database.py`
5. `sdk/models/`
6. `sdk/operations.py`
7. 最后再回头看具体 backend 和具体 model family

## 9. 一句话总结

0.9.0 的 SDK 不是简单地“多支持几个模型和系统”，而是在模型组织、性能库语义、结果可解释性和核心估算器实现方式上都向前迈了一大步。
