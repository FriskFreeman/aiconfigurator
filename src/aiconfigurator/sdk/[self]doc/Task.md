# Task 组件解析总结

本文基于 `v0.9.0` 的 `src/aiconfigurator/sdk/task.py`，从结构、功能、上下游交互和与 `InferenceSession` 的关系四个角度，对 `task` 组件做一次总览式梳理。

## 1. 一句话定位

`task` 组件不是底层算子仿真器，也不是最终执行真实推理的 runtime。

它更像 AIC SDK 里的“任务装配与实验调度层”：

- 向上承接 CLI / API / YAML / profile 的用户输入
- 向下组织 database、model config、并行搜索空间和 Pareto 分析
- 最终把一个“用户问题”翻译成一批可执行的仿真实验

如果说：

- `Model` 提供模型结构语义
- `PerfDatabase` 提供算子性能来源
- `Backend` 提供框架级时序与显存规则
- `InferenceSession` 负责一次候选配置的性能评估

那么 `task` 负责的是：

- “这次要评估什么”
- “评估哪些候选”
- “用什么 database mode / backend version / 并行空间”
- “最后把整批候选交给谁去跑”

---

## 2. 代码结构总览

`task.py` 可以大致分成 5 层。

### 2.1 轻量辅助层

文件前部有一些辅助函数和约束检查：

- `UnsupportedWideepConfigError`
- `_is_hopper_system()`
- `_validate_deepseek_v4_model_hardware_support()`
- `_deep_merge()`
- `_ensure_munch()`
- `_get_database_with_optional_missing_data()`
- `build_disagg_parallel_lists()`

这一层主要负责：

- 硬件兼容性前置检查
- 配置字典深合并
- 兼容旧测试桩的 database 获取包装
- 构造 disagg 场景的默认并行搜索空间

其中 `build_disagg_parallel_lists()` 很重要，因为它实际上把：

- backend 差异
- MoE / non-MoE 差异
- wideep / deepep / 标准通信差异
- prefill / decode 两端差异

统一编码进了默认搜索空间里。

### 2.2 配置上下文层：`ConfigLayer` 与 `TaskContext`

这一层是 `task` 比较“工程化”的地方。

- `ConfigLayer`：一层可条件生效的配置片段
- `TaskContext`：构造任务时的标准化输入上下文

`TaskContext` 里集中存放：

- serving mode
- model / backend / system 信息
- isl / osl / prefix / SLA
- wideep、chunked prefill、moe backend
- `database_mode`
- `engine_step_backend`
- profiles / yaml patch

它的价值在于：后面的默认配置生成不再直接依赖零散函数参数，而是统一围绕 `ctx` 做条件装配。

### 2.3 配置工厂层：`TaskConfigFactory`

这是 `task` 的第一大核心。

`TaskConfigFactory` 负责把 `TaskContext` 翻译成真正的任务配置树，主要机制是“layer 叠加”：

1. 先应用 `_base_layers()`
2. 再按 serving mode 应用 `_mode_layers()`
3. 再应用 profile layers
4. 最后应用 YAML patch / replace
5. 再做 agg / disagg 的 finalize

因此它不是把一堆默认值写死在一个大构造函数里，而是采用了：

- 基础层
- 模式层
- profile 层
- YAML 覆写层

这种可组合结构。

`v0.9.0` 下这层的几个重点是：

- 支持 `database_mode`
- 支持 `engine_step_backend`
- 支持 profiles 注册机制
- 支持 YAML patch / replace 两种模式
- 更清晰地区分 agg 和 disagg 默认搜索空间

### 2.4 用户级任务对象：`TaskConfig`

这是 `task` 的第二大核心，也是 SDK/CLI 最直接面对的对象。

`TaskConfig` 的职责不是“执行仿真”，而是把用户意图固化成一个经过校验的任务描述对象。它主要做 4 件事：

1. 收集构造参数、YAML patch、profiles
2. 构造 `TaskContext` 并调用 `TaskConfigFactory.create()`
3. 进行类型规范化和合法性校验
4. 生成便于日志和导出的 `task_name` / `to_yaml()`

可以把 `TaskConfig` 理解成：

- “一次实验任务的标准化描述”

而不是：

- “真正执行仿真的执行器”

### 2.5 执行调度层：`TaskRunner`

这是 `task.py` 的第三大核心。

`TaskRunner` 才真正把 `TaskConfig` 送入仿真流程。它主要负责：

- 依据任务配置创建 `RuntimeConfig`
- 加载一个或两个 `PerfDatabase`
- 组装一个或两个 `ModelConfig`
- 枚举并行配置列表
- 调用 `pareto_analysis.agg_pareto()` 或 `disagg_pareto()`
- 捕获无解、缺表、OOM 等异常并包装日志语义

因此：

- `TaskConfig` 负责“描述任务”
- `TaskRunner` 负责“执行任务”

---

## 3. `TaskConfig` 的主流程

如果从用户输入进入 `task`，主路径通常是：

1. CLI / API 收集参数
2. 构造 `TaskConfig`
3. `TaskConfigFactory.create()` 生成标准配置树
4. `TaskConfig.validate()` 校验量化模式、backend 支持性、database 支持性
5. 后续由 `TaskRunner.run()` 执行

### 3.1 默认值与 patch 的叠加顺序

`TaskConfig` 的配置装配顺序大致是：

1. 构造参数形成初始 `TaskContext`
2. `TaskConfigFactory` 注入默认结构
3. profile layers 覆盖局部字段
4. YAML patch 或 replace 最后生效
5. agg/disagg finalize 再做边界收缩

这意味着 `task` 组件本身承担了一个非常重要的职责：

- 为上层提供“声明式配置入口”
- 为下层提供“结构稳定的规范化配置”

### 3.2 `validate()` 的角色

`validate()` 不是简单做空值检查，它还承担了不少 SDK 语义层面的前置过滤：

- `fp8_static` 只允许 TRTLLM
- 根据 model family 决定应该检查哪些 perf op
- 必要时加载 database 读取 `supported_quant_mode`
- 校验 wideep / moe / attention / kv-cache 等量化模式组合是否合法
- 在非 `SILICON` 模式下允许 missing-data 路径参与验证

因此 `TaskConfig.validate()` 实际上是：

- “仿真可执行性”的第一道守门员

它能在真正进入大规模 Pareto sweep 之前，把明显不支持的任务先挡掉。

---

## 4. `TaskRunner` 的两条执行路径

### 4.1 agg 路径

`TaskRunner.run_agg()` 的流程可以概括成：

1. 从任务配置构造 `RuntimeConfig`
2. 依据 `system/backend/version/database_mode` 获取 `PerfDatabase`
3. 构造 `ModelConfig`
4. 用 `enumerate_parallel_config()` 生成并行候选
5. 调用 `pareto_analysis.agg_pareto()`
6. 返回 `pareto_df`

这里 `task` 负责的是：

- 决定要评估哪些并行配置
- 决定用哪个 database 语义
- 决定 runtime SLA 和约束

而真正对每个候选配置做静态评估、批大小搜索、吞吐筛选的，是更下游的 `agg_pareto()` 与 `InferenceSession`。

### 4.2 disagg 路径

`TaskRunner.run_disagg()` 比 agg 更复杂，因为它要同时装配两套 worker 端：

- prefill worker
- decode worker

它的流程大致是：

1. 构造 `RuntimeConfig`
2. 分别加载 prefill / decode 的 `PerfDatabase`
3. 分别构造 prefill / decode 的 `ModelConfig`
4. 分别枚举 prefill / decode 的并行配置列表
5. 处理 disagg 特有约束
6. 调用 `pareto_analysis.disagg_pareto()`
7. 返回 `pareto_df`

这里的一个关键信号是：

- `task` 层已经开始承担“系统级组合搜索”的职责

它不再只是单 worker 的参数容器，而是在 disagg 模式下直接规定：

- 两端的并行候选边界
- replica 级 GPU 上限
- worker 数量范围
- rate matching 校正因子
- autoscale 时的目标 TPOT

---

## 5. `task` 和其他仿真组件的交互

## 5.1 向上的“前端”交互：CLI / API / YAML

从上游看，`task` 是 SDK 仿真链路最重要的输入承接层之一。

主要入口包括：

- `src/aiconfigurator/cli/main.py`
- `src/aiconfigurator/cli/api.py`

CLI / API 负责：

- 解析命令行或 Python 参数
- 组织 experiment 集合
- 构造一个或多个 `TaskConfig`

而 `task` 负责把这些高层输入变成可以执行的标准任务。

所以从分层角度看：

- CLI / API 更像“用户前端”
- `task` 更像“仿真调度前端”

它是用户需求进入 SDK 内部仿真栈的第一站。

## 5.2 和 `models` 的交互

`task` 不直接构造具体 model 对象，但会大量使用模型元信息：

- `get_model_family()`
- `check_is_moe()`
- `_apply_model_quant_defaults()`
- `get_model_config_from_model_path()`

这些交互主要用于：

- 判断是不是 MoE
- 选择默认 `nextn`
- 推断量化模式默认值
- 选择该验证哪类 attention / MLA / DSA / wideep perf 表

所以 `task` 和 `models` 的关系是：

- `models` 提供模型语义
- `task` 用这些语义决定任务配置和校验逻辑

## 5.3 和 `PerfDatabase` 的交互

`task` 是 `PerfDatabase` 的重要上层入口之一。

它负责决定：

- 用哪个 `system/backend/version`
- 是否使用 `database_mode`
- 是否允许 `allow_missing_data` 间接生效
- 是否需要把数据库 default mode 改成目标 mode

尤其在 `v0.9.0` 中，`task` 是：

- `database_mode`
- estimate-only
- HYBRID shared layer

这些能力进入实际仿真任务的主要调度口。

换句话说，`PerfDatabase` 自己负责“能查什么”，而 `task` 负责“这次任务要用什么方式去加载和查询它”。

## 5.4 和 `utils` 的交互

`task` 对 `utils` 的依赖很重，尤其是：

- `enumerate_parallel_config()`
- `get_model_config_from_model_path()`
- `ListFlowDumper`

其中最关键的是 `enumerate_parallel_config()`。

`task` 并不直接穷举所有 TP/PP/DP/MoE-TP/MoE-EP 组合，而是先给出边界，再交给这个工具函数生成合法并行配置列表。

所以 `task` 更像“搜索空间定义者”，而不是“低层枚举器实现者”。

## 5.5 和 `pareto_analysis` 的交互

这是 `task` 最直接的下游。

`TaskRunner` 自己并不做 Pareto 求解，它把完整上下文交给：

- `agg_pareto()`
- `disagg_pareto()`

这说明 `task` 的职责重点是：

- 任务装配
- 候选空间组织
- 运行参数下发

而不是：

- 候选优选算法本身

Pareto 分析层负责真正对候选点做 sweep、筛选、去重和前沿抽取。

## 5.6 和 `backends` 的交互

`task` 不直接调用具体 backend 的 `run_static()` / `run_agg()` 逻辑，但会通过 `ModelConfig`、`RuntimeConfig` 和 `database_mode` 把执行语义传下去。

尤其在 `v0.9.0` 中，`task` 会把：

- `engine_step_backend`
- quant modes
- wideep / eplb / attention backend

这些 backend 敏感参数注入到配置树里。

因此它是 backend 行为的重要“参数分发层”。

---

## 6. `task` 和 `InferenceSession` 的关系

这是理解 SDK 分层时最容易混淆、但也最关键的一点。

先给结论：

- `task` 不等于 `InferenceSession`
- 两者不是替代关系，而是上下游配合关系

### 6.1 `InferenceSession` 做什么

`InferenceSession` / `DisaggInferenceSession` 负责的是：

- 持有 `Model`
- 持有 `PerfDatabase`
- 持有 `Backend`
- 对某个具体候选配置执行一次性能评估

它是“仿真执行门面”。

### 6.2 `task` 做什么

`task` 负责的是：

- 生成任务配置
- 校验任务是否合法
- 决定使用哪个 database / mode / backend version
- 生成并行搜索空间
- 决定是走 agg 还是 disagg
- 调用 Pareto 分析去批量评估候选

它是“任务装配与调度门面”。

### 6.3 两者的边界

更精确地说：

- `task` 关心“一批候选怎么定义与组织”
- `InferenceSession` 关心“一个候选怎么评估”

因此两者一个偏上层编排，一个偏下层执行。

### 6.4 两者如何配合

在 agg 路径里，真实调用链大致是：

1. CLI / API 构造 `TaskConfig`
2. `TaskRunner.run_agg()`
3. `pareto_analysis.agg_pareto()`
4. `get_model()` + `get_backend()`
5. 构造 `InferenceSession(model, database, backend)`
6. `sess.find_best_agg_result_under_constraints()`

在 disagg 路径里则是：

1. `TaskRunner.run_disagg()`
2. `pareto_analysis.disagg_pareto()`
3. `get_backend()` 两次
4. 构造 `DisaggInferenceSession(prefill_database, prefill_backend, decode_database, decode_backend)`
5. `find_best_disagg_result_under_constraints()`

所以 `task` 并不直接 new session，但它负责把 session 所需的大部分前置条件准备好。

### 6.5 一个简单类比

可以把两者类比成：

- `task`：实验总控台 / 调度器
- `InferenceSession`：单次仿真执行器

`task` 负责决定：

- 做哪些实验
- 每个实验怎么配置
- 搜哪些候选

`InferenceSession` 负责决定：

- 单个候选在当前 model + database + backend 下的性能结果是多少

### 6.6 是否存在功能重叠

有少量语义上的相邻，但总体上不重叠。

两者都会接触：

- runtime config
- model config
- database
- backend

但接触方式不同：

- `task` 是装配、选择、下发
- `InferenceSession` 是消费这些对象并执行评估

所以它们不是“两个都能做仿真”的并列层，而是前后串联的两级门面。

---

## 7. 为什么 `task` 在 0.9.0 尤其重要

`v0.9.0` 的 `task` 相比旧版本更重要，原因不只是参数变多了，而是它开始更明显地承接新能力的编排责任。

典型体现包括：

- `database_mode` 被正式纳入任务配置
- estimate-only / HYBRID 能力主要经由 `task` 进入真实任务
- `engine_step_backend` 让 backend 可切换 Python/Rust 静态估算路径
- profiles / YAML patch 让任务装配更声明式
- disagg 的 worker / replica / tuning 参数组织更完整

因此在 0.9.0 里，`task` 已经不只是一个“传参文件”，而是：

- SDK 仿真工作流里的核心编排层

---

## 8. 阅读建议

如果后面你准备继续深入 `task`，建议顺序是：

1. `src/aiconfigurator/sdk/task.py`
2. `src/aiconfigurator/sdk/pareto_analysis.py`
3. `src/aiconfigurator/sdk/inference_session.py`
4. `src/aiconfigurator/sdk/backends/base_backend.py`
5. `src/aiconfigurator/cli/main.py`

这样会比较容易形成“用户输入 -> 任务装配 -> Pareto sweep -> Session 执行”的完整链路感。

---

## 9. 一句话总结

`task` 组件在 AIC SDK 中承担的是“任务定义、配置规范化、约束校验、搜索空间组织和 Pareto 调度”的职责；而 `InferenceSession` 承担的是“针对单个候选配置执行具体仿真评估”的职责。两者前后配合，构成了 AIC 从用户任务到性能结果之间最核心的一段桥梁。
