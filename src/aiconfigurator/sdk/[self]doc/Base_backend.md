# Base Backend 解析总结

本文已按 `v0.9.0` 修订。`BaseBackend` 的核心定位没有变，但 0.9.0 给它补上了两条很关键的新能力：

- 每个算子的预测来源追踪
- Rust 前向步长估算器接入

## 1. 快速概括

`src/aiconfigurator/sdk/backends/base_backend.py` 中的 `BaseBackend`，仍然是所有推理框架仿真 backend 的骨架基类。

它的职责可以概括成三件事：

1. 提供统一的静态阶段推演流程
2. 约束子类必须补齐聚合调度与显存模型
3. 把模型层、性能库层、结果汇总层串起来

所以它依然不是某个框架自己的调度实现，而是三大 backend 共享的“公共仿真底座”。

## 2. 到 0.9.0 仍然成立的主干逻辑

### 2.1 `run_static` 仍然是静态单点推演总入口

它仍然通过：

- `context` / `generation` 分阶段
- 遍历 `model.context_ops` 和 `model.generation_ops`
- 调 `op.query(database, ...)`
- 累加 latency / energy

来得到 TTFT、TPOT、tokens/s/gpu 以及显存检查所需的信息。

### 2.2 子类仍然主要补两个方向

- `run_agg` / `find_best_agg_result_under_constraints`
  负责各框架自己的聚合调度与搜索逻辑
- `_get_memory_usage`
  负责各框架自己的显存记账规则

这说明 0.9.0 并没有改变 backend 继承体系的主骨架。

## 3. 0.9.0 的关键变化

### 3.1 `BaseBackend` 开始显式追踪每个算子的来源

这是最值得注意的变化之一。

在 0.9.0 中：

- `_run_context_phase()` 不只返回 latency / energy dict，还会返回 `context_source_dict`
- `_run_generation_phase()` 也会返回 `generation_source_dict`

这些 source 来自 `PerformanceResult.source`，通常是：

- `"silicon"`
- `"empirical"`
- `"mixed"`

这意味着 `BaseBackend` 不再只负责“把各层时间加起来”，还负责把“这些时间各自来自哪里”一并传下去。

### 3.2 `InferenceSummary` 不再只是数值容器

因为有了新的 source dict，`BaseBackend` 在汇总阶段会把下面这些都塞进 summary：

- `context_latency_dict`
- `generation_latency_dict`
- `context_energy_wms_dict`
- `generation_energy_wms_dict`
- `context_source_dict`
- `generation_source_dict`

所以 `BaseBackend` 在 0.9.0 里承担了更明确的 provenance 汇总职责。

### 3.3 新增 Rust engine-step 快路径

`BaseBackend` 在 0.9.0 引入了：

- `should_use_rust_engine_step()`
- `estimate_static_latency_breakdown_with_rust()`

这表示静态阶段推演现在存在两条路径：

1. 传统 Python 路径
   遍历 op，逐层调用 `query`
2. Rust 路径
   直接通过 Rust core 估算 context / generation 的 forward-pass 时间

当 `runtime_config.engine_step_backend == "rust"` 时，`_run_static_breakdown()` 会优先走 Rust 路径。

这件事的意义不是简单“加速”，而是 SDK 里第一次明确出现了一个可切换的核心前向估算后端。

### 3.4 `_run_static_breakdown()` 的返回语义扩展了

旧理解里，它更像是：

- context latency
- context energy
- generation latency
- generation energy

现在它实际上还带上：

- context source
- generation source

也就是说，这个函数从“数值分解器”演进成了“数值 + 来源分解器”。

## 4. 和子类 backend 的关系也有新变化

虽然 `BaseBackend` 自己只负责静态路径，但 0.9.0 中的三个具体 backend：

- `vllm_backend.py`
- `sglang_backend.py`
- `trtllm_backend.py`

都开始在 `run_agg` 内：

- 透传 per-op breakdown
- 透传 per-op source
- 在需要时调用 Rust 混合步或 decode 步估算器

所以 `BaseBackend` 的新 source / Rust 语义，并没有停留在基类内部，而是被子类继续向上扩展到了聚合调度阶段。

## 5. 对旧版文档理解的修正

### 5.1 旧说法：“BaseBackend 主要负责 latency/power 累加”

现在要修正为：

- 负责 latency / energy 累加
- 同时负责 per-op source 追踪

### 5.2 旧说法：“静态路径完全是 Python op-by-op 查表”

现在也不完全对。

更准确地说：

- 默认仍是 Python 查表路径
- 但 0.9.0 新增了 Rust estimator 可选路径

### 5.3 旧说法：“BaseBackend 只对最终 summary 的数值字段负责”

现在应补充：

- 它还负责 summary 的 per-op provenance 字段

## 6. 为什么这次变化重要

如果从仿真 SDK 的演进方向看，0.9.0 的 `BaseBackend` 变化说明了两件事：

1. AIC 开始更重视“预测解释性”
   不只是给出预测，还要告诉你它来自哪里
2. AIC 开始为核心前向估算器做多实现后端铺路
   Python 和 Rust 已经并行存在

## 7. 建议阅读顺序

1. `src/aiconfigurator/sdk/backends/base_backend.py`
2. `src/aiconfigurator/sdk/performance_result.py`
3. `src/aiconfigurator/sdk/inference_summary.py`
4. 再去看 `vllm_backend.py`、`sglang_backend.py`、`trtllm_backend.py`

## 8. 一句话总结

0.9.0 的 `BaseBackend` 仍然是各推理框架共享的静态仿真骨架，但它已经从“纯数值累加器”升级为“数值累加 + 来源追踪 + 可切换 Rust 估算后端”的公共执行底座。
