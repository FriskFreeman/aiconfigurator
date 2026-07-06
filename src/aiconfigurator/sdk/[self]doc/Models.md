# Models 解析总结

本文已按 `v0.9.0` 代码结构修订，不再把模型层理解成旧版的单文件 `models.py`，而应理解为 `sdk/models/` 子包。

## 1. 快速概括

`sdk/models/` 是 AIC SDK 里的“模型结构抽象层”。

它负责把：

- HuggingFace / 本地模型配置
- backend 相关分支条件
- TP / PP / EP / attention-DP 等并行参数
- 模型家族特有的结构差异

整理成一个可供 backend 和 `PerfDatabase` 消费的统一对象。这个对象最重要的两类产出是：

- `context_ops`
- `generation_ops`

也就是说，模型层的本质任务不是“跑推理”，而是把模型翻译成一张可查表、可累加、可做显存估算的算子图纸。

## 2. 0.8.0 到 0.9.0 的核心变化

### 2.1 最大结构变化：`models.py` 被拆成 `models/` 包

0.8.0 的旧理解已经不再准确。

在 0.9.0 中：

- `src/aiconfigurator/sdk/models.py` 被删除
- 新增 `src/aiconfigurator/sdk/models/` 包
- 通过 `base.py + helpers.py + 每个 family 一个文件` 的形式组织

这次重构的意义不是“把大文件拆小”这么简单，而是把模型层改成了更明确的三层结构：

1. `base.py`
   负责 `BaseModel`、模型注册表、公共 KV cache / 公共属性逻辑
2. `helpers.py`
   负责模型信息提取、架构到 family 的映射、量化默认值推断
3. `deepseek.py` / `moe.py` / `qwen35.py` / `deepseek_v4.py` 等
   负责每个 family 的具体算子管线定义

所以现在再说“模型逻辑集中在 `models.py`”已经不对了。

### 2.2 从硬编码分派转向注册表分派

0.9.0 中的模型创建是 registry-driven 的。

核心机制是：

- 每个模型类通过 `@register_model(...)` 注册 family
- `models/__init__.py` 自动扫描并导入子模块
- `get_model()` 只负责：
  - 解析 `model_info`
  - 做 family 判定
  - 应用量化默认值
  - 从注册表拿类并调用 `create()`

这让“新增一个模型 family”从过去的集中式修改，变成了“新增一个文件并注册”的模式。

### 2.3 family 体系继续扩充，且分工更清晰

相较你之前基于 0.8.0 的理解，0.9.0 至少有这几个值得注意的变化：

- `DEEPSEEKV4` 成为新 family，并有独立的 `deepseek_v4.py`
- `KIMIK25` 不再混在旧的 DeepSeek family 名义下，而是明确成为单独 family，由 `deepseek.py` 这一个实现类兼管
- `QWEN35` 拥有独立文件 `qwen35.py`
- `HYBRIDMOE`、`NEMOTRONH`、`NEMOTRONNAS` 等复杂 family 也被拆到专门文件中

这意味着 0.9.0 的模型层不仅支持更多模型，更重要的是“把模型复杂性显式结构化”了。

### 2.4 family 内部的 backend 分派更明显

现在模型创建不只是“按 family 选类”，还会在 `create()` 内继续根据 backend 或配置做分支。

典型例子：

- `MOEModel` 会按 `backend_name == "sglang"` 以及 `deepep_moe` 等条件分派到 `SGLangEPMOEModel`
- `DeepSeekModel` 会根据 WideEP / backend 选择普通版、SGLang WideEP 版或 TRT-LLM WideEP 版
- `DeepSeekV32Model` 也有类似的 WideEP 派发

所以模型层和 backend 已经不是完全解耦的“纯结构层”，而是带有适度 backend-aware 的构造逻辑。

## 3. 当前 0.9.0 的代码结构应该怎么理解

### 3.1 `models/__init__.py`

这是模型层入口。

主要职责：

- 自动导入包内模型模块
- 暴露 `get_model()`
- 做兼容性 re-export，保持旧调用风格还能 `from ...models import XXXModel`

### 3.2 `models/base.py`

这是 `BaseModel` 和注册表中心。

它负责：

- 统一保存模型元信息
- 处理公共并行参数
- 提供 `get_kvcache_*` 等通用逻辑
- 保存 `_MODEL_REGISTRY`
- 提供 `@register_model`

### 3.3 `models/helpers.py`

这是纯函数工具层。

它把原来混在单文件中的若干基础能力抽离出来：

- `_get_model_info`
- `_architecture_to_model_family`
- `_apply_model_quant_defaults`
- `check_is_moe`
- `calc_expectation`

这样一来，模型构造逻辑和模型信息解析逻辑不再搅在一起。

### 3.4 family 文件

每个 family 文件都在做两件事：

1. 定义该 family 的 `create()` 构造路径
2. 定义该 family 的 `context_ops` / `generation_ops`

这也是你后续如果想研究某个具体模型怎么映射到仿真算子的最佳入口。

## 4. 现在仍然成立的核心职责

虽然结构重构很大，但你之前抓到的几个本质判断仍然成立。

### 4.1 模型层仍然是“算子图纸层”

backend 并不自己知道某个模型有哪些层、每层有哪些算子、KV cache 怎么算。

这些信息仍由模型层提供。

### 4.2 KV cache 估算仍然是模型层关键职责

尤其对：

- GQA / MHA
- MLA
- DeepSeek / Kimi 一类复杂架构

模型层依然是决定显存水位估算的关键来源。

### 4.3 上层仍通过 `get_model()` 统一拿实例

无论内部怎么拆包，上层拿模型对象的统一门面仍然是 `get_model()`。

## 5. 0.9.0 新增或更突出的模型能力

结合 release 和代码，模型层在 0.9.0 里比较值得关注的新增点有：

- DeepSeek-V4 独立建模支持
- Kimi K2.5 family 语义更清晰
- Qwen3.5 独立模型实现
- 模型家族拆分后更容易承载后续大型 MoE / hybrid 架构

这里有一个很重要的认识：

0.9.0 不只是“多支持几个模型”，而是在为更复杂模型结构做 SDK 层基础设施重排。

## 6. 对旧版文档理解的修正

如果继续沿用旧文档，有两处需要明确修正：

### 6.1 “模型逻辑主要在 `models.py`”这一说法已经过时

现在应改为：

- 模型入口在 `models/__init__.py`
- 注册与基类在 `models/base.py`
- family 实现在各自文件
- 工具函数在 `models/helpers.py`

### 6.2 “模型家族主要靠工厂函数里的显式 if/else 派发”也不再准确

现在更准确的说法是：

- 外层 family 选择靠 registry
- family 内部细分路径靠各类自己的 `create()`

## 7. 建议阅读顺序

如果你接下来要继续看 0.9.0 的模型层，建议这样读：

1. `src/aiconfigurator/sdk/models/README.md`
2. `src/aiconfigurator/sdk/models/__init__.py`
3. `src/aiconfigurator/sdk/models/base.py`
4. `src/aiconfigurator/sdk/models/helpers.py`
5. 你最关心的 family 文件，例如：
   - `deepseek_v32.py`
   - `deepseek_v4.py`
   - `moe.py`
   - `qwen35.py`

## 8. 一句话总结

0.9.0 的模型层不再是“一个大文件里的模型工厂”，而是一个以 registry 为中心、以 family 文件为单位、并且允许 backend-aware 构造分支的模型抽象子包。
