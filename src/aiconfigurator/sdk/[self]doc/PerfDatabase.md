# PerfDatabase 代码解析文档

本文已按 `v0.9.0` 修订。旧版里把大量职责都归到 `perf_database.py` 一个超大文件，这个判断在“主体仍然很大”这一点上没错，但 0.9.0 已经把其中一部分基础能力拆出去了，而且查询语义也变得更强。

## 1. 快速总结

`PerfDatabase` 仍然是 SDK 中最核心的底层性能查询引擎。

它负责：

- 根据 `system + backend + version` 加载对应性能数据
- 对 GEMM / attention / communication / MoE / 特化模块数据做查表
- 在 `SILICON / HYBRID / EMPIRICAL / SOL` 这些模式下提供统一查询接口
- 在需要时做插值、外推和 fallback

但在 0.9.0 中，它不再只是“查 latency/energy 的大字典容器”，还承担了几项更关键的新能力：

- 混合模式下的跨 framework 数据共享
- estimate-only 数据库实例化
- 每个算子结果的数据来源追踪
- oneCCL / 新模型模块级数据 / DeepSeek-V4 数据支持

## 2. 0.8.0 到 0.9.0 的核心变化

### 2.1 结构层面：把通用数学和系统规格抽出去了

release 明确提到，0.9.0 把：

- `interpolation.py`
- `system_spec.py`

从 `perf_database.py` 拆了出去。

这意味着旧文档里“插值函数都在 `PerfDatabase` 内部”的说法需要修正。

现在更准确的结构是：

- `perf_database.py`
  负责数据生命周期、文件加载、查询 API、fallback 逻辑
- `interpolation.py`
  负责 1D/2D/3D 插值、外推、metric 抽取等数学细节
- `system_spec.py`
  负责系统 YAML 的轻量包装，尤其是 `get_p2p_bandwidth()`

所以 0.9.0 在这一层做的不是“功能新增”，而是“把 monolith 稍微拆成更清晰的职责边界”。

### 2.2 查询结果现在显式携带数据来源

0.9.0 一个非常关键但容易被忽略的变化是：

- `PerformanceResult` 新增 `source`
- `PerfDatabase` 在 HYBRID fallback 时会把来源打成 `"empirical"`
- 正常查到实测表时来源是 `"silicon"`
- 聚合不同来源时会折叠成 `"mixed"`

这件事很重要，因为它改变了 SDK 结果的解释能力。

以前更像是“给你一个预测值”。
现在更像是“给你一个预测值，并告诉你它究竟来自实测数据还是经验回退”。

### 2.3 HYBRID 模式下引入跨 framework 共享层

这是 0.9.0 SDK 改动里最值得重点关注的一项。

结合代码可见：

- `get_database(..., database_mode=...)`
- `PerfDatabase(..., database_mode=...)`
- `op_kernel_source_manifest.yaml`
- `_load_op_kernel_source_manifest_entries`
- `_read_filtered_rows`

这一套逻辑在做的事情是：

在 `HYBRID` 模式下，当前 backend/version 如果缺某些算子数据，不再只会在本 backend 自身数据里找，而是允许从 sibling backend/version 中继承兼容的 kernel-source 行。

可以把它理解成：

- 0.8.0：更像“每个 framework 各自守着自己的库”
- 0.9.0：在 HYBRID 模式下，开始允许“共享底层 op 数据层”

这和官方 release 里的 “op data is now shareable across frameworks in hybrid mode” 是一致的。

### 2.4 支持 estimate-only 数据库

`get_database()` 在 0.9.0 新增了：

- `allow_missing_data`
- `database_mode`

这意味着如果某个系统/后端/版本没有完整 silicon 数据，但调用方跑的是 `EMPIRICAL` / `SOL` / 非纯 silicon 模式，SDK 可以先基于 system spec 建一个“估算型数据库实例”。

这个变化的价值是：

- 让一些 PCIe / 新系统 / 数据尚未补齐的平台，至少先能做估算
- 不再把“没有完整实测 CSV”直接等同于“不能跑 SDK”

这和 release 提到的 estimate-only PCIe systems 是同一条演进方向。

### 2.5 新的数据类型与新模型模块级支持

从 `PerfDataFilename` 和大量新增 query/load 逻辑可以看到，0.9.0 又往数据库里塞进了更多“模型特化数据”：

- `oneccl_perf.txt`
- `mhc_module_perf.txt`
- `deepseek_v4_context_module_perf.txt`
- `deepseek_v4_generation_module_perf.txt`
- 多个 `dsv4_flash_*` 模块级文件

这意味着 `PerfDatabase` 的职责正在从“通用基础算子表”继续扩张到“复杂前沿模型的模块级专用性能库”。

## 3. 当前 0.9.0 的职责分层

### 3.1 文件加载与数据库实例生命周期

仍然由 `perf_database.py` 主导。

关键入口包括：

- `get_supported_databases()`
- `get_latest_database_version()`
- `get_database()`
- `get_all_databases()`

0.9.0 在这里的增强主要有：

- 更稳健的 version 选择
- 支持 suffix / rc 版本排序
- 支持 estimate-only 加载
- 支持基于 database mode 切换 shared-layer 行为

### 3.2 系统规格层

现在应从 `system_spec.py` 理解，而不再把它当作 `PerfDatabase` 的内部小工具。

`SystemSpec` 的意义是：

- 保留 dict 兼容性
- 给 `PerfDatabase` 和上层提供更清晰的系统带宽访问接口

### 3.3 插值 / 外推层

现在应从 `interpolation.py` 理解。

这一层负责：

- metric 抽取
- 1D / 2D / 3D 插值
- 数据网格外推
- 缓存部分提取后的 metric 视图

这使得 `PerfDatabase` 更聚焦在“查什么”和“去哪查”，而不是所有数学细节都埋在类内部。

## 4. `PerformanceResult` 在 0.9.0 中的重要性上升

如果只把 `PerformanceResult` 看成“float + energy”，已经不够了。

现在它至少承载三层含义：

- latency
- energy
- source

而且这个 `source` 会在加法聚合时保留或折叠成 `"mixed"`。

这意味着 `PerfDatabase` 已经从“返回数值”升级成“返回带 provenance 的数值”。

后面 `BaseBackend`、`InferenceSummary`、`operations.py` 里的改动，其实都是围绕这个新语义在传递。

## 5. 对旧文档理解的修正

### 5.1 旧说法：“PerfDatabase 同时承包插值数学与系统规格”

现在要改成：

- `PerfDatabase` 仍是查询中心
- `interpolation.py` 是数学基础层
- `system_spec.py` 是系统规格包装层

### 5.2 旧说法：“HYBRID 只是 silicon miss 时回退到经验公式”

现在也不够准确了。

更完整的说法应该是：

- `HYBRID` 仍然有 silicon -> empirical fallback
- 但 0.9.0 还增加了跨 framework 的 shared-layer 数据继承能力

### 5.3 旧说法：“查询结果主要是 latency/energy”

现在应修正为：

- 查询结果是 `PerformanceResult`
- 它还显式记录来源 provenance

## 6. 为什么这次改动很重要

如果从 AIC 的仿真精度与工程可扩展性角度看，0.9.0 对 `PerfDatabase` 的改动有三层价值：

1. 更容易接新模型
   现在复杂模型可以通过新增专用模块级 perf 文件逐步接入
2. 更容易接新平台
   estimate-only 模式让没有完整测表的平台也能先跑
3. 更容易解释结果
   per-op source 让用户知道“预测值是测出来的还是补出来的”

## 7. 建议阅读顺序

如果要继续深入 0.9.0 的性能库实现，推荐顺序如下：

1. `src/aiconfigurator/sdk/perf_database.py`
2. `src/aiconfigurator/sdk/interpolation.py`
3. `src/aiconfigurator/sdk/system_spec.py`
4. `src/aiconfigurator/sdk/performance_result.py`
5. 再结合 `operations.py` 看每类 op 如何调用 query 接口

## 8. 一句话总结

0.9.0 的 `PerfDatabase` 仍然是 SDK 的底层查表核心，但它已经从“单库单框架查表器”演进成了一个支持跨 framework 共享、estimate-only 加载、结果来源追踪、并承载前沿模型模块数据的更完整查询层。
