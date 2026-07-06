# PerfDatabase 0.9.0 细节补充

本文专门补充 `v0.9.0` 中 `PerfDatabase` 两条最值得关注的新特性：

1. HYBRID 模式下的 framework 共享层
2. estimate-only 数据库加载路径

这两部分都属于 0.9.0 相比 0.8.0 的重要能力增强，而且会直接影响你如何理解：

- 新平台为什么能“先估算再补采集”
- 不同 backend 为什么能在 HYBRID 模式下共享部分算子数据
- 为什么同一个 `PerfDatabase` 在 `SILICON` 和 `HYBRID` 语义下行为不同

---

## 1. 先给整体结论

### 1.1 HYBRID framework 共享层的本质

它不是“把 vLLM / SGLang / TRT-LLM 的数据库彻底合并”。

它真正做的是：

- 当数据库以 `HYBRID` 语义加载时
- 某个 op 文件允许跨 framework 共享
- 且 manifest 声明当前 backend 可以消费某些 `kernel_source`

那么加载器会在**当前 backend/version 的主数据文件之外**，按优先级去其他 version、甚至其他 framework 的 sibling 文件夹里，把允许继承的行补进来。

这个机制的核心目标是：

- 尽量优先使用本 backend 的高保真数据
- 在本 backend 缺失某些 shape 时，尽可能复用兼容的底层 kernel 数据
- 避免 HYBRID 模式因为局部缺表而过早退化到纯经验公式

### 1.2 estimate-only 数据库的本质

它也不是“生成一份假的完整 perf 数据库”。

它真正做的是：

- 即使 `<system>/<backend>/<version>` 对应的数据目录不存在或不完整
- 只要系统 YAML 还在
- 且调用方允许 `allow_missing_data=True`

就先构造出一个 `PerfDatabase` 对象。

这个对象并不 magically 拥有 silicon 数据，而是允许后续 query 在：

- `EMPIRICAL`
- `SOL`
- `HYBRID`

这些模式下仍然工作，因为这些模式本来就允许 fallback 到经验估算逻辑。

所以 estimate-only 更像：

- “数据库壳子 + 系统规格 + 经验公式能力还在”
- 只是“真实 silicon 表可能缺失”

---

## 2. HYBRID 模式下的 framework 共享层

## 2.1 这个特性是怎么被打开的

入口在 [perf_database.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/perf_database.py:320) 的 `get_database()`。

它新增了两个参数：

- `allow_missing_data`
- `database_mode`

其中与共享层直接相关的是：

- `shared_flag = (database_mode or "").upper() == "HYBRID"`  
  见 [perf_database.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/perf_database.py:355)

这个 `shared_flag` 会进入 cache key：

- `cache_key = (systems_root, system, shared_flag)`  
  见 [perf_database.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/perf_database.py:361)

这点很重要，因为它意味着：

- 同一个 `system/backend/version`
- 在 `SILICON` 语义加载
- 和在 `HYBRID` 语义加载

会得到两份彼此隔离的缓存实例。

换句话说，0.9.0 不是在同一个数据库实例上动态开关共享层，而是在加载阶段就把“是否开启共享层”编码进实例身份里。

## 2.2 `PerfDatabase` 初始化时如何记录这个模式

在 [perf_database.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/perf_database.py:2604) 的 `PerfDatabase.__init__()` 中：

- `self.enable_shared_layer = (database_mode or "").upper() == "HYBRID"`  
  见 [perf_database.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/perf_database.py:2626)

这说明：

- “共享层开不开”是数据库对象的加载属性
- 和后续 query 时的 `default_database_mode` 不是同一个概念

这里要特别区分两层语义：

1. **加载层语义**
   决定数据源是不是只读本 backend 主目录，还是允许读 sibling 目录
2. **查询层语义**
   决定查不到 silicon 时，是否 fallback 到 empirical

0.9.0 的一个核心设计，就是把这两层显式分开了。

## 2.3 manifest 是共享层的白名单契约

共享层不是“随便扫别的目录就拿来用”，而是严格受 manifest 控制。

manifest 文件在：

- [op_kernel_source_manifest.yaml](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/systems/op_kernel_source_manifest.yaml)

其文件头已经把机制说得很清楚：

- 这是 cross-backend / cross-version measurement reuse 的 runtime contract
- 由 `tools/perf_database/audit_kernel_source.py` 自动生成
- 不建议手工修改

加载逻辑在：

- `_load_op_kernel_source_manifest_entries()`  
  见 [perf_database.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/perf_database.py:111)

这个函数会把 manifest 组织成：

- `op_file -> tuple[entry, ...]`

每个 entry 至少包含：

- `op_file`
- `kernel_source`
- `tier`
- `frameworks`
- `systems`

因此 manifest 的作用本质上是：

- 为每个 op 文件声明哪些 kernel_source 可以共享
- 为每个 kernel_source 声明哪些 backend 有资格消费它
- 区分高保真共享和 fallback 共享

## 2.4 共享层真正发生在 `_build_op_sources()`

核心实现函数是：

- `_build_op_sources()`  
  见 [perf_database.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/perf_database.py:3482)

它的输出不是一个路径，而是：

- `list[(file_path, kernel_source_filter)]`

也就是一个**按优先级排序的数据源列表**。

默认第一项永远是主路径：

- `sources = [(primary_path, None)]`  
  见 [perf_database.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/perf_database.py:3508)

如果没有开启共享层，直接返回：

- `if not self.enable_shared_layer: return sources`  
  见 [perf_database.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/perf_database.py:3509)

同时 `nccl` / `oneccl` 这类 framework-agnostic 文件也不会走共享层：

- 见 [perf_database.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/perf_database.py:3511)

这说明共享层是“按 op 文件选择性启用”的，不是整个数据库无条件共享。

## 2.5 数据源的优先级规则

`_build_op_sources()` 的注释已经把优先级写出来了，源码见：

- [perf_database.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/perf_database.py:3488)

优先级是：

1. 当前 backend/version 主数据
2. 同 framework 的其他 version，按新到旧
3. 其他 framework，按 framework 名排序，每个 framework 内也是新到旧

这套排序的设计意图很明确：

- 先保住“当前 backend 的本地真实性”
- 再尽量用“同 framework 的历史/邻近版本”
- 最后才跨 framework 继承

所以共享层不是平均混合，而是**强主从、弱补洞**。

## 2.6 为什么需要 `kernel_source_filter`

这是共享层实现里最关键也最精巧的点。

`_build_op_sources()` 返回的不是单纯 sibling path，而是：

- `(sibling_path, ks_filter)`

原因在源码注释里写得很清楚：

- 各个 `load_*` 函数在构建内部字典时，通常会把 `kernel_source` 从最终 key 中剥离掉
- 如果不先过滤，sibling 文件中的行会在 shape 冲突时直接覆盖主 backend 行

相关说明见：

- [perf_database.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/perf_database.py:3503)

也就是说，共享层真正的安全阀不是“后面再 merge 时判断”，而是：

- **在读取 CSV 行时就先按 kernel_source 白名单过滤**

## 2.7 `_read_filtered_rows()` 如何维持 first-wins 语义

读取逻辑在：

- `_read_filtered_rows()`  
  见 [perf_database.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/perf_database.py:135)

它支持两种输入：

1. 单一路径
2. 多个 `(path, kernel_source_filter)` 元组

对多源情况，它会按顺序把所有合法行拼接起来，并保持原顺序。

这和 loader 内部的“遇到已存在 key 就跳过/保留先到者”逻辑配合起来，就能自然形成：

- 主 backend 行优先
- sibling 行只填补缺失 shape

源码注释明确写了：

- earliest source wins on every coordinate  
  见 [perf_database.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/perf_database.py:147)

## 2.8 `shared` 与 `shared_fallback` 的区别

manifest 中的 `tier` 有两类：

- `shared`
- `shared_fallback`

加载逻辑在 `_build_op_sources()` 里把它们都接纳到 `HYBRID` 模式中：

- 见 [perf_database.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/perf_database.py:3517)

其中：

- `shared`
  表示命名明确、较高保真的 kernel source
- `shared_fallback`
  表示像 `kernel_source=default` 这种更粗粒度、framework-implicit 的 fallback 行

如果某个 sibling source 包含 fallback-only kernel，代码会打 warning：

- 见 [perf_database.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/perf_database.py:3574)

这非常重要，因为它表明开发者自己也承认：

- 共享层并不保证所有继承数据都和目标 backend 的真实行为完全一致
- `shared_fallback` 明显比 `shared` 更低保真

## 2.9 共享层解决了什么问题

它主要解决三类问题：

### 2.9.1 某 backend 缺少局部 shape

比如：

- 当前 backend/version 的某个 op 文件存在
- 但只缺一部分 shape

共享层能先从同 framework 邻近 version 补。

### 2.9.2 某 backend 对某模型专用 op 的测表覆盖还不完整

尤其是在复杂模型快速迭代时，某些 kernel-source 的本质行为跨 framework 很接近，允许先复用兼容测表，而不是立刻掉到经验公式。

### 2.9.3 HYBRID 模式下尽量延后纯经验 fallback

如果不做共享层，HYBRID 的唯一救场方式就是：

- silicon miss -> empirical

有了共享层之后，HYBRID 实际上变成了：

1. 当前 backend/version
2. 同 framework 邻近 version
3. 其他 framework 兼容 kernel
4. 最后才 empirical

这是 0.9.0 很重要的增强。

## 2.10 共享层不解决什么问题

它也有非常明确的边界。

### 2.10.1 它不改变 query 时的数据库模式逻辑

共享层只影响“加载时能看到哪些行”。

真正查不到数据时要不要回退 empirical，仍然由：

- `_query_silicon_or_hybrid()`  
  见 [perf_database.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/perf_database.py:3849)

来决定。

### 2.10.2 它不保证 cross-framework 完全等价

代码自己已经用 warning 承认了：

- framework-implicit fallback rows may differ from real backend behavior  
  见 [perf_database.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/perf_database.py:3576)

### 2.10.3 它不适用于所有 op 文件

例如：

- `nccl`
- `oneccl`

就不会走这条路。

---

## 3. estimate-only 数据库

## 3.1 这个能力为什么要出现

0.8.0 的典型语义更接近：

- 没有 `<system>/<backend>/<version>` 对应数据目录
- 那么数据库基本就加载失败

0.9.0 新增 estimate-only 的核心动机是：

- 某些系统已经有系统规格 YAML
- 但还没有完整 silicon perf 表
- 或者某个 backend/version 还没做完采集

这时候如果调用方愿意接受：

- `SOL`
- `EMPIRICAL`
- `HYBRID`

这些非纯 silicon 语义，就不应该被“目录不存在”直接卡死。

## 3.2 estimate-only 是如何被触发的

入口仍然在 `get_database()`：

- [perf_database.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/perf_database.py:320)

当遍历 systems paths 时，如果主数据路径不存在：

- `elif allow_missing_data:`  
  见 [perf_database.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/perf_database.py:393)

它不会立刻失败，而是先记下一个：

- `missing_data_candidate`

当所有路径都扫完，如果找不到真实数据目录，但 `missing_data_candidate` 存在，就会进入：

- `Loading estimate-only database...`  
  见 [perf_database.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/perf_database.py:408)

然后直接调用：

- `PerfDatabase(system, backend, version, systems_root, database_mode=...)`

这一步没有要求数据目录必须存在。

## 3.3 调用方什么时候会打开 `allow_missing_data`

这不是无条件开启的。

在 `task.py` 里，调用侧明确只在**非 SILICON 模式**下打开这个能力。

例如校验路径里：

- `allow_missing_data = database_mode is not None and database_mode != SILICON`
- 见 [task.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/task.py:941)

以及统一取数据库的 helper 里：

- 同样基于 `database_mode != SILICON` 来设置
- 见 [task.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/task.py:1243)

这说明 estimate-only 的设计前提很明确：

- **它不是拿来伪装成完整 silicon 数据库的**
- 它只在调用方已经接受“非纯 silicon 估算”的前提下启用

## 3.4 estimate-only 数据库对象内部是什么状态

重点在于：

- `PerfDatabase.__init__()` 仍然会读取系统 YAML
- 仍然会构造所有 `LoadedOpData`
- 但很多 op 的 `LoadedOpData.loaded` 可能是 `False`

这里的关键包装类是：

- `LoadedOpData`
- 见 [perf_database.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/perf_database.py:2515)

如果某个文件没加载到，`loaded=False`。

一旦后面有人在纯 silicon 语义下访问它，`raise_if_not_loaded()` 会抛出：

- `PerfDataNotAvailableError`
- 并明确提示这组 model/system/backend/version 不受 SILICON mode 支持

相关实现见：

- [perf_database.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/perf_database.py:2529)

所以 estimate-only 对象并不是“装作所有 op 都有数据”，而是：

- 缺表状态仍然保留
- 只是允许 query 层根据 database mode 走 fallback

## 3.5 estimate-only 为什么还能查询成功

关键在 `_query_silicon_or_hybrid()`：

- [perf_database.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/perf_database.py:3849)

逻辑是：

1. 先尝试 `get_silicon()`
2. 如果失败：
   - 若模式是 `HYBRID`，返回 `PerformanceResult(get_empirical(), source="empirical")`
   - 若模式不是 `HYBRID`，则报错

虽然很多 query 函数还各自实现了 `SOL` / `EMPIRICAL` 路径，但这个 helper 体现了最核心的思想：

- estimate-only 能工作，不是因为它有 silicon 数据
- 而是因为 query 本身允许从经验模型出结果

## 3.6 estimate-only 适合的典型场景

### 3.6.1 新系统已有 YAML，但没完整采表

例如某个 PCIe 系统：

- 系统拓扑、带宽、显存、FLOPS 都已定义
- 但 backend/version 的 CSV 还没补齐

这时可以先让 `SOL` / `EMPIRICAL` 跑起来。

### 3.6.2 某 backend/version 还没完全支持，但想先验证任务链路

estimate-only 允许先把：

- task 装配
- backend 路径
- session 搜索

跑通，而不必等全量 silicon 表到齐。

### 3.6.3 某些 synthetic / validation 场景

比如代码里对 DeepSeek-V4 synthetic mode 的容忍逻辑，就属于这类思路的一部分。

## 3.7 estimate-only 的局限

### 3.7.1 它不是精度增强，而是可用性增强

estimate-only 的核心价值是：

- 让系统“能跑”

而不是：

- 让系统“更准”

### 3.7.2 在 `SILICON` 模式下它仍然会失败

因为 `LoadedOpData.raise_if_not_loaded()` 仍然会抛异常。

estimate-only 并没有绕过 silicon 数据缺失这个事实。

### 3.7.3 支持程度取决于各 query 函数有没有经验路径

不是所有 query 都一定有同等成熟的 empirical / SOL fallback。

所以 estimate-only 的实际可用范围，本质上还取决于：

- 当前模型会触发哪些 op
- 这些 op 的 query 有没有可用 fallback

---

## 4. 这两条功能是如何配合起来的

HYBRID 共享层和 estimate-only 不是两条孤立特性，它们组合起来才体现 0.9.0 的真正增强。

一个更准确的层次关系是：

### 第 1 层：estimate-only

解决“数据库目录缺失时还能不能先构造出对象”的问题。

### 第 2 层：HYBRID shared layer

解决“即使有数据库对象，局部 shape 缺失时，能不能先从 sibling 数据补”的问题。

### 第 3 层：empirical fallback

解决“共享层补完后仍缺数据时，能不能先给出经验估算”的问题。

因此 0.9.0 的查询韧性相对 0.8.0 其实变成了三重兜底：

1. 主 backend/version 数据
2. HYBRID 共享层数据
3. empirical / SOL 估算

这正是 0.9.0 `PerfDatabase` 最重要的进化之一。

---

## 5. 对理解 AIC 仿真结果的意义

如果你很关心“仿真结果到底来自哪”，这两条特性会直接影响解释方式。

### 5.1 同样是 HYBRID 结果，不一定都是纯 empirical

在 0.9.0 之前，你可以更粗暴地理解：

- HYBRID = 实测缺失时经验回退

但现在更准确的理解是：

- HYBRID 可能仍然来自 silicon 行
- 这些 silicon 行可能是当前 backend 的
- 也可能是 sibling version / sibling framework 共享来的
- 只有再缺，才退到 empirical

### 5.2 estimate-only 不代表“完全没有数据意义”

如果系统规格足够完整、经验模型合理，estimate-only 仍然能支撑：

- 粗粒度趋势判断
- 配置搜索初筛
- 新平台早期评估

只是不能把它误读成和完整 silicon 覆盖同等可信。

### 5.3 estimate-only 不是“特殊数据库类”，而是同一个 `PerfDatabase` 的缺表实例

这一点很关键。

从代码实现上看，estimate-only 并没有引入新的 database class，也没有额外生成一份“纯经验数据库文件”。

它仍然直接调用：

- `PerfDatabase(system, backend, version, systems_root, database_mode=...)`
- 见 [perf_database.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/perf_database.py:408)

因此 estimate-only 与正常数据库对象的根本差别不是“类型不同”，而是：

- 正常数据库：大部分 `LoadedOpData.loaded=True`
- estimate-only：很多 `LoadedOpData.loaded=False`

也就是说，estimate-only 更像：

- 一个真实的 `PerfDatabase` 外壳
- 带着系统 YAML、system spec、query 方法、经验公式路径
- 但缺少相应 backend/version 的 silicon 数据表

### 5.4 `allow_missing_data` 和 `database_mode` 不是一回事

这两个参数很容易被混为一谈，但职责其实不同。

`allow_missing_data` 的作用是：

- 当 `<system>/<backend>/<version>` 数据目录不存在或不完整时
- 是否仍允许先实例化出一个 `PerfDatabase`

对应代码见：

- [perf_database.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/perf_database.py:393)
- [perf_database.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/perf_database.py:402)

而 `database_mode` 的作用分成两层：

- 加载阶段：是否开启 `HYBRID` shared layer
- 查询阶段：后续默认按 `SILICON` / `HYBRID` / `EMPIRICAL` / `SOL` 哪种语义执行

其中上层当前策略是：

- `database_mode != SILICON` 时，自动令 `allow_missing_data=True`
- 见 [task.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/task.py:1243)

所以现在项目里的确形成了“非 `SILICON` 模式通常伴随 allow-missing”的使用习惯，但从 `PerfDatabase` 自身实现上看，它们并不是同一个开关，也不是完全冗余的字段。

### 5.5 数据库加载和数据库查询，在 0.9.0 中被更明显地解耦了

这也是这一轮阅读代码后最值得单独强调的设计点。

在 database 构建/加载阶段，`database_mode` 的直接作用其实很有限，主要只有两类：

1. 是否把 `HYBRID` 语义编码进 cache key
2. 是否在 `PerfDatabase.__init__()` 中打开 `enable_shared_layer`

对应代码见：

- `shared_flag = (database_mode or \"\").upper() == \"HYBRID\"`
- [perf_database.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/perf_database.py:355)

和：

- `self.enable_shared_layer = (database_mode or \"\").upper() == \"HYBRID\"`
- [perf_database.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/perf_database.py:2626)

除此之外，加载阶段并不会因为当前是 `EMPIRICAL` / `SOL` / `SILICON` 就构造出三四种完全不同的数据结构。

真正大量依赖 `database_mode` 的，是后续 query 逻辑：

- 是否先查 silicon
- 查不到是否 fallback 到 empirical
- 是否直接走 SOL / empirical 公式

而上层在取库后，还会额外做一次：

- 若请求 mode 与库对象当前默认 mode 不同，则 `deepcopy(db)` 后 `set_default_database_mode(mode)`
- 见 [task.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/task.py:1253)

这进一步说明：

- “怎么把数据库对象建出来”
- “之后按什么模式解释和查询这个对象”

在 0.9.0 里已经是相对独立的两个阶段。

### 5.6 estimate-only 真正生效，主要靠查询阶段而不是构造阶段

estimate-only 数据库刚被构造出来时，其默认 mode 其实仍然先是：

- `self._default_database_mode = common.DatabaseMode.SILICON`
- 见 [perf_database.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/perf_database.py:2629)

所以 estimate-only 之所以能工作，并不是因为构造出了一个“天生属于 EMPIRICAL/SOL 的特殊对象”，而是因为上层后续会把它放到合适的查询语义下使用。

也正因如此，若强行把 estimate-only 数据库拿去做 `SILICON` 查询，依然会在访问未加载 `LoadedOpData` 时抛出 `PerfDataNotAvailableError`。

换句话说，estimate-only 的真实含义更接近：

- “允许先把数据库对象和系统规格装起来”
- “然后把缺失 silicon 数据这件事推迟到 query 阶段，再按 mode 决定是报错还是 fallback”

---

## 6. 一句话总结

`PerfDatabase` 在 0.9.0 的这两条增强，本质上是在把数据库从“有完整测表才能工作”的刚性组件，升级成一个“优先用本地高保真数据、再尝试共享兼容数据、最后允许经验估算”的弹性查询层。
