# AIC Generator 组件总览

本文结合以下内容做总览分析：

- `docs/generator_overview.md`
- `.claude/rules/generator-development.md`
- `src/aiconfigurator/generator/` 下的核心实现代码

目标不是逐行解释，而是先回答三个问题：

1. `generator` 组件是做什么的
2. 它整体是怎么工作的
3. 代码目录里各个子模块分别负责什么

## 1. Generator 是做什么的

`src/aiconfigurator/generator` 可以理解为 AIC 里的“部署配置生成器”。

它的核心职责是：把一份相对统一的模型部署意图，转换成不同后端、不同版本、不同部署目标真正可执行的配置产物。

这里的“统一部署意图”通常包括：

- 模型名、模型路径
- backend 类型，如 `trtllm`、`vllm`、`sglang`
- 并行策略，如 TP、PP、DP、EP
- 服务模式，如 `agg` 或 `disagg`
- k8s / benchmark / sflow / SLA 等外围部署信息

最终生成的产物可能包括：

- 后端 CLI 参数
- TRT-LLM 的 `extra_engine_args.yaml`
- `run.sh`
- Dynamo 的 k8s manifest
- `llm-d` 的 Helm values
- benchmark 相关脚本和配置
- sflow 部署 YAML

所以从定位上说，`generator` 不是推理引擎本身，也不是搜索最优配置的求解器本身，而是“把配置决策落成实际部署文件”的一层。

## 2. 它解决的核心问题

如果没有 generator，上层工具就必须直接理解大量后端差异：

- 不同 backend 参数名不同
- 不同 backend 版本的参数格式不同
- `agg` / `prefill` / `decode` 三种角色的配置组合不同
- benchmark、k8s、sflow 这些外围产物又各有格式

`generator` 的价值就是把这些差异收敛到一条统一流水线里：

- 上层只描述“我要怎样部署”
- generator 负责把这份描述映射成各后端、各版本、各场景对应的文件

这也是它在 AIC 中很关键的原因：它处在“搜索结果 / 用户输入”和“实际部署输出”之间，是一个非常典型的转换层。

## 3. 总体执行链路

结合 `generator-development.md`，它的主流程基本可以概括成 6 步。

### 3.1 输入解析

入口主要在 `api.py`。

这一层负责把外部输入变成 generator 内部统一理解的参数结构。输入来源可能有：

- CLI 传入
- YAML 配置
- SDK / profiler / search 结果
- “naive” 自动推导出的初始部署参数

这一层会做的事包括：

- 解析 backend 和 backend version
- 合并 override 参数
- 准备标准化的 `params` 结构
- 调用后续渲染逻辑

### 3.2 默认值补全

默认值逻辑主要在 `rendering/schemas.py`，数据来自：

- `config/deployment_config.yaml`

这一步会把用户没写全的字段补齐，并且按 backend、部署模式、角色等条件选择默认值。

这里的意义很大，因为很多模板和规则都假设输入已经是“结构完整”的。

### 3.3 规则计算

规则执行在 `rendering/rule_engine.py`，规则文件在：

- `rule_plugin/trtllm.rule`
- `rule_plugin/vllm.rule`
- `rule_plugin/sglang.rule`
- `rule_plugin/benchmark/*.rule`

这一层不是模板渲染，而是“根据条件改写参数”。

比如：

- 某些 backend 下开启某个模式时，要自动设置额外字段
- benchmark 场景下要额外推导一些配置
- 某些角色只在 `prefill` 或 `decode` 下生效

可以把它看成模板之前的“业务规则修正层”。

### 3.4 参数映射

参数映射主要发生在 `rendering/engine.py`，并依赖：

- `config/backend_config_mapping.yaml`

这一层负责把 generator 内部的抽象字段，翻译成目标 backend 实际认识的字段名和结构。

例如同样是“并行度”或“显存相关限制”，在不同 backend 里的配置位置和表达形式可能不同，这一步负责对齐。

### 3.5 模板渲染

模板同样由 `rendering/engine.py` 组织，模板文件位于：

- `config/backend_templates/trtllm/`
- `config/backend_templates/vllm/`
- `config/backend_templates/sglang/`
- `config/backend_templates/benchmark/`
- `config/backend_templates/sflow/`

这里是 generator 真正把参数变成文本产物的阶段。

几个关键特点：

- 按 backend 区分模板
- 按 backend version 选择最合适模板
- 支持版本化模板，如 `cli_args.0.20.1.j2`
- 没有专门模板时，也会走映射或 fallback 逻辑生成 CLI 参数

对于 `dynamo-python` 这类路径，它不只是简单渲染 Jinja，还会调用 Python 逻辑去组织 k8s 相关输出。

### 3.6 产物落盘

最终输出由 `artifacts.py` 负责。

它负责：

- 把内部 artifact 名称映射成实际文件名
- 输出 YAML / JSON / shell 文件
- 对 shell 脚本加执行权限
- 统一处理产物目录结构

因此 `rendering/engine.py` 更像“生成内容”，而 `artifacts.py` 更像“写文件并规范化格式”。

## 4. 一条更直观的端到端路径

可以把 generator 理解成下面这条流水线：

```text
外部输入
  -> api.py 准备参数
  -> aggregators.py 组织统一配置结构
  -> schemas.py 补默认值
  -> rule_engine.py 按 backend/场景执行规则
  -> engine.py 选模板、做字段映射、渲染文本
  -> artifacts.py 输出为 yaml/sh 等文件
```

如果输入不是用户手写，而是来自 profiler / search 结果，则在最前面还会经过 `module_bridge.py` 或 `naive.py` / `enumerate.py`。

## 5. 子模块速览

下面按“读源码时最值得先建立的心智模型”来概括各模块作用。

### 5.1 `api.py`

这是 generator 的公共入口层。

主要职责：

- 对外暴露生成配置与生成产物的主函数
- 解析 backend、version、override
- 串起参数准备、渲染和产物生成
- 提供 help / schema / naive config 等对外接口

如果想知道“上层到底怎么调用 generator”，从这里开始最合适。

### 5.2 `aggregators.py`

这是“输入结构整理层”。

它会把零散输入组织成 generator 内部标准布局，例如：

- `ServiceConfig`
- `K8sConfig`
- `DynConfig`
- `WorkerConfig`
- `SlaConfig`
- `BenchConfig`
- `SflowConfig`
- `NodeConfig`
- `params`

它解决的是“输入很散，但后面规则和模板希望看到稳定结构”的问题。

### 5.3 `rendering/schemas.py`

这是默认值和 schema 语义层。

核心作用：

- 从 `deployment_config.yaml` 读取默认配置
- 根据 backend 或上下文应用默认值
- 让后续阶段拿到结构完整、字段补齐的参数

它不直接输出文件，但对整个渲染结果影响很大。

### 5.4 `rendering/rule_engine.py`

这是规则解释器。

它实现了一套轻量规则执行机制，用来读取 `.rule` 文件并按条件修改参数。

它的意义在于把“后端/场景特定逻辑”从 Python 主流程中拆出来，避免所有分支判断都堆在模板或主代码里。

### 5.5 `rendering/engine.py`

这是整个 generator 最核心的渲染中枢。

它负责：

- 选择 backend 对应模板
- 按版本选择最匹配模板
- 套用规则后的参数上下文
- 渲染 CLI args、engine args、k8s YAML、run.sh 等产物
- 某些情况下通过 Python 路径生成 Dynamo 相关输出

如果说 `api.py` 是入口，`engine.py` 就是 generator 真正的“发动机”。

### 5.6 `artifacts.py`

这是产物落盘层。

它关注的是：

- 文件名映射
- 目录组织
- YAML / JSON 规范输出
- shell 文件执行权限

这一层让上游不必关心“渲染出来的字符串最终以什么名字、什么格式保存”。

### 5.7 `module_bridge.py`

这是 generator 与 SDK / 搜索结果之间的桥。

它的典型场景是：

- 已经拿到某次搜索或 profiling 的最佳结果
- 现在要把结果转换成 generator 可消费的结构

也就是说，它连接的是“结果表格 / TaskConfig”到“正式部署配置”之间的鸿沟。

### 5.8 `naive.py`

这是一个“快速给出初始可用部署方案”的模块。

它更偏启发式估算，例如：

- 模型权重规模
- 显存能否放下
- 最小 TP 或其他并行策略
- dense 和 MoE 模型的不同估算方法

适合在没有完整 profiler 数据时，先得到一份基础配置。

### 5.9 `enumerate.py`

这是候选配置枚举层。

它主要用于：

- 枚举不同并行策略组合
- 给 profiling / DGD 等流程准备候选配置
- 复用 generator 的渲染能力，为每个候选方案生成实际可跑的参数

它本质上更像“配置搜索空间生成器”。

### 5.10 `rendering/translate.py`

这个模块比较专一，作用是把渲染出来的 TRT-LLM engine YAML，再翻译成动态 CLI flags。

典型形式是：

- `--trtllm.xxx.yyy value`

它服务于某些 `use_dynamo_generator=True` 的路径，尤其是需要把 YAML 形式配置压平成命令行参数的场景。

### 5.11 `sflow.py`

这是 sflow 部署的增强上下文生成器。

它不是简单模板渲染，而是提前把很多 backend-specific 逻辑算好，再注入给：

- `config/backend_templates/sflow/sflow_deploy.yaml.j2`

它会负责的内容包括：

- server launch script 片段
- benchmark script 片段
- readiness probe pattern
- 资源数、任务数、变量 profile

它的设计思想很清晰：让 sflow 模板尽量保持“编排骨架”，把复杂逻辑放在 Python 里提前算完。

### 5.12 `utils.py`

这是共享工具层。

目前比较关键的是：

- backend 名字归一化
- 字符串到布尔/整数的容错转换
- 读取 `backend_version_matrix.yaml`
- 根据 Dynamo 版本解析对应 backend 版本

这使得 generator 可以把“版本兼容关系”集中管理，而不是散落在主逻辑里。

### 5.13 `main.py`

这是命令行入口包装层，负责把 generator 能力暴露给 CLI 使用。

如果要看“用户在命令行敲一个 generator 相关命令后，最终怎么流到 `api.py`”，这里值得顺着读。

## 6. 关键数据文件速览

除了 Python 实现，generator 很大一部分行为其实由配置文件和模板驱动。

### 6.1 `config/deployment_config.yaml`

这是默认值和部署字段定义的重要来源。

很多“没显式填写时该补什么”的逻辑，都依赖它。

### 6.2 `config/backend_config_mapping.yaml`

这是 generator 抽象字段到 backend 实际字段的映射表。

它让多个 backend 可以复用统一输入模型。

### 6.3 `config/backend_version_matrix.yaml`

这是 Dynamo 版本和各 backend 版本之间的映射矩阵。

它回答的问题是：

- 目标 Dynamo 版本是什么
- 该版本下推荐或匹配的 `trtllm` / `vllm` / `sglang` 版本是什么

### 6.4 `rule_plugin/*.rule`

这些文件承载“条件化配置修正逻辑”。

相比把逻辑都硬编码在 Python 里，这种做法更方便随着 backend 演进做小步更新。

### 6.5 `config/backend_templates/**/*.j2`

这些 Jinja2 模板承载最终文本输出格式。

从目录划分上可以看出 generator 的主要输出对象：

- backend 运行参数
- k8s 部署 YAML
- `llm-d` values
- benchmark 脚本
- sflow 部署 YAML

## 7. Generator 的设计思路

从整体实现看，这个组件有几个很明显的设计特点。

### 7.1 统一输入，后端差异后置

上层尽量只表达“部署意图”，真正的 backend 差异在 rule、mapping、template 层展开。

这让上层调用更稳定，也让新增 backend 或新增版本时改动更集中。

### 7.2 规则、映射、模板三层分离

可以粗略把职责拆成：

- 规则层决定“参数应该被怎么修正”
- 映射层决定“内部字段如何翻译成 backend 字段”
- 模板层决定“最后文本长什么样”

这是它能支撑多个 backend 版本并行演进的重要原因。

### 7.3 版本敏感，但尽量数据驱动

很多模板文件本身就是按版本维护的。

这说明 generator 明确承认一个现实：不同 backend 版本的参数接口会变化，因此必须做版本选择，而不是假设一个模板永远通用。

### 7.4 既服务手工配置，也服务自动化流程

它既能接用户直接输入，也能接：

- search/profiler 结果
- naive 估算结果
- 枚举得到的候选配置

因此 generator 不只是“配模板”，也是自动化部署链路中的关键中间层。

## 8. 和其他模块的关系

把它放到整个 AIC 里看，generator 大致处在中游位置：

```text
用户输入 / SDK任务 / profiler结果 / 搜索结果
  -> generator 统一整理与渲染
  -> k8s / llm-d / run.sh / benchmark / sflow 等部署产物
  -> 实际部署与验证流程
```

换句话说：

- 搜索模块决定“什么配置可能更好”
- generator 决定“把这个配置怎样正确落地”

## 9. 建议的阅读顺序

如果接下来要继续深入源码，推荐顺序如下：

1. `src/aiconfigurator/generator/api.py`
2. `src/aiconfigurator/generator/aggregators.py`
3. `src/aiconfigurator/generator/rendering/engine.py`
4. `src/aiconfigurator/generator/rendering/rule_engine.py`
5. `src/aiconfigurator/generator/rendering/schemas.py`
6. `src/aiconfigurator/generator/config/` 下的 YAML、`.rule`、`.j2`
7. `module_bridge.py`、`naive.py`、`enumerate.py`
8. `sflow.py` 和 `translate.py`

这样会先建立主干认知，再回头看细节扩展模块。

## 10. 一句话总结

`generator` 的本质，是一个把“统一部署意图”翻译成“具体后端、具体版本、具体运行场景可执行配置产物”的生成引擎。

它靠的不是单一模板，而是一整套“输入整理 + 默认值 + 规则引擎 + 参数映射 + 版本模板 + 产物落盘”的流水线。
