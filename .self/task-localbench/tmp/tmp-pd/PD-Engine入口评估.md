# SGLang PD 分离场景下从 Engine 入口直接运行 P/D 实例的评估

## 评估问题

当前我们为了测试 DeepSeek-V3 dummy 5-layer 的 PD 分离运行，采用的是：

- `prefill server`
- `decode server`
- `router server`

这条链路在实践中暴露出较多工程性问题，例如：

- router tokenizer 注册与本地 `/model` 路径处理
- `prefill/decode` 之间的 KV transfer backend 配置
- `router -> decode` 的 500/timeout

因此需要评估：是否可以跳过 `server/router` 这一层，直接从 `Engine` 入口单独启动一个解耦的 `P` 实例或 `D` 实例，从而更干净地测试 PD 相关逻辑。

## 结论先行

结论分三层：

1. `Engine` 不是完全不支持 PD。
2. 但 `Engine` 不能天然替代完整的 `server + router` 编排体系。
3. 如果目标只是绕开 router 层干扰，值得做一个“薄编排”的 Engine 方案；如果目标是规避底层 PD/KV transfer bug，Engine 方案并不能本质绕开。

更具体地说：

- `Engine` 内部直接复用了 `ServerArgs`，且 `generate()`/`async_generate()` 已显式暴露 `bootstrap_host`、`bootstrap_port`、`bootstrap_room`、`disagg_prefill_dp_rank` 等 PD 字段。
- 这说明 SGLang 的 Python `Engine` 入口本身能理解 PD 请求，不是只能走 HTTP server。
- 但官方文档中的 PD 运行方式、启动参数、路由示例，几乎全部围绕 `launch_server`/`sglang serve` 展开，而不是“两个 Engine 直接互连”。
- 因此，若我们直接用 `Engine` 运行 PD，本质上是在自己承担原本由 `server/router` 完成的编排工作。

## 关键证据

### 1. Engine 本身吃的是 ServerArgs

在 SGLang `engine.py` 中，`Engine.__init__(**kwargs)` 的注释明确写到：

- 参数与 `ServerArgs` 相同
- `Engine` 启动后仍会拉起 tokenizer/scheduler/detokenizer 等子进程

可见：

- `Engine` 不是一个“轻量纯函数式推理对象”
- 它本质上是“无 HTTP 包装的运行时入口”

对应源码：

- `/sgl-workspace/sglang/python/sglang/srt/entrypoints/engine.py`

### 2. Engine.generate 已经带有 PD 请求字段

`Engine.generate()` 与 `Engine.async_generate()` 的参数里已经包含：

- `bootstrap_host`
- `bootstrap_port`
- `bootstrap_room`
- `disagg_prefill_dp_rank`

同时 `GenerateReqInput` 也定义了这些字段。

这说明：

- 对于 decode 端来说，它接收一个“来自 prefill 的 PD 请求”在数据结构上是成立的
- 从 Python API 层面，用户可以手动构造这类请求，而不一定非要经 HTTP router

对应源码：

- `/sgl-workspace/sglang/python/sglang/srt/entrypoints/engine.py`
- `/sgl-workspace/sglang/python/sglang/srt/managers/io_struct.py`

### 3. 底层 PD/KV transfer 代码不依赖 HTTP server 才存在

底层 PD 实现位于：

- `/sgl-workspace/sglang/python/sglang/srt/disaggregation/...`

其中可以看到：

- `common/conn.py`
- `prefill.py`
- `nixl/conn.py`
- `mori/conn.py`

这些逻辑围绕的是：

- `disaggregation_mode`
- `bootstrap_room`
- `bootstrap_port`
- transfer backend 的状态轮询与异常处理

这说明真正的 PD 难点在于：

- prefill/decode 两端的 KV 生命周期管理
- transfer backend 状态机
- bootstrap/room/rank 协调

而不在于 HTTP server 本身。

## Server 路线与 Engine 路线的真实差异

### Server 路线做了什么

`server`/`router` 方案除了拉起底层 runtime，还额外负责：

- HTTP API 封装
- `/health`、`/generate`、`/server_info`
- worker 注册
- tokenizer 注册
- PD 请求路由
- router 侧的模型/worker 元数据管理

因此 server 路线的问题来源有两类：

1. 业务外壳问题
2. 底层 PD/runtime 问题

我们目前已经实际见到两类问题都出现过：

- 第一类：router tokenizer `/model` 注册问题
- 第二类：decode 侧 `NIXL KVReceiver Exception`

### Engine 路线省掉了什么

如果改成 Python 直接驱动两个 `Engine`：

- 可以省掉 HTTP server
- 可以省掉 router worker 注册/tokenizer 注册
- 可以自己决定请求如何送到 prefill 或 decode

因此它确实有机会绕开“router 壳层”的干扰。

### Engine 路线绕不开什么

但 Engine 路线依然绕不开：

- `disaggregation_mode=prefill/decode`
- transfer backend
- bootstrap port/room
- prefill 向 decode 传 KV 的状态机
- decode 端等待/接收 KV 的逻辑

也就是说：

- 如果 bug 在 router/tokenizer/register 层，Engine 有帮助
- 如果 bug 在 `nixl/mooncake` 传输层，Engine 没有本质规避能力

这一点和我们当前实验结果高度一致：  
在修掉 router tokenizer 问题后，最终仍卡在 decode 侧的 `KVReceiver` 超时。

## 能否“单独运行一个 P 实例”？

### 可以启动，但“有意义地跑通”取决于目标

从原理上，`Engine(server_args...)` 可以带：

- `disaggregation_mode="prefill"`
- `disaggregation_transfer_backend=...`
- `disaggregation_bootstrap_port=...`

因此单独启动一个 `P Engine` 是可行的。

但这类实例是否有意义，要看我们想测试什么：

#### 场景 A：只想看 P 端能否成功启动、编译、跑 prefill

这是可行的。

我们可以验证：

- 模型是否能在 `prefill` 模式启动
- MLA/PCG 是否成功编译
- prefill 请求是否能进入 runtime

这适合做：

- P 端 warmup/PCG 静态验证
- P 端 kernel/图编译验证

#### 场景 B：想看 P 端完整完成 PD 传输

这就不成立了。

因为 prefill 在 PD 模式下不是“算完就结束”，而是还要：

- 绑定 bootstrap 元信息
- 为 decode 端生成可消费的 KV transfer

如果没有 decode 端或没有可用的 fake backend 逻辑闭环，就无法完整验证“PD prefill 完成”。

所以：

- `P Engine` 可做启动和局部计算验证
- 不足以单独完成真实 PD 链路验证

## 能否“单独运行一个 D 实例”？

### 可以启动，但单独运行几乎不具备独立业务意义

`decode` 实例在 PD 中天然依赖：

- prefill 传来的 KV
- bootstrap_room
- bootstrap_host/port

所以一个独立 `D Engine` 即便能启动成功，也只是“接收器待命”。

它不能凭空完成正常生成，因为：

- 没有 prefix KV
- 没有 prefill metadata
- 没有 transfer backend 输入

因此：

- `D Engine` 可用于验证 decode 端能否起服务、能否进入等待态
- 但不能独立完成有代表性的推理

## 是否能用两个 Engine 直接做 P/D 解耦试验？

### 理论上可行

理论上可以写一个 Python orchestrator：

1. 启动 `prefill_engine`
2. 启动 `decode_engine`
3. 人工生成 `bootstrap_room`
4. 将请求先送入 prefill
5. 再将带 `bootstrap_host/bootstrap_port/bootstrap_room` 的请求送入 decode

从数据结构与参数设计上，这条路是成立的。

### 工程上并不“免费”

我们要自己补齐原本 server/router 干的事：

- 进程生命周期管理
- readiness 检查
- bootstrap room 分配
- 请求分发顺序
- request id/routing 绑定
- 日志采集
- 错误清理

换句话说，这不是“直接调用两个函数”，而是：

- 把 HTTP server/router 的那层外壳，换成我们自己的 Python 编排层

### 最关键的一点

即使这条 Engine-PD 方案实现了，它仍会走相同的底层 PD transfer 代码。

所以：

- 若当前真正问题在 `nixl` 接收/轮询/metadata 匹配
- 那么换成 Engine 路线后，问题大概率仍会复现

只是报错位置会更靠近 Python orchestrator，而不是 HTTP router。

## 对当前任务的价值评估

### 有价值的地方

如果当前目标是：

- 去掉 router 层干扰
- 更直接地观测 prefill/decode 的交互
- 做更细粒度的日志与状态控制

那么 Engine 路线值得尝试。

它的主要收益是：

- 更少的壳层噪声
- 更容易控制请求字段
- 更容易做定制化 instrumentation

### 局限和风险

如果当前目标是：

- 证明 PD + `nixl` 在这个模型/版本/配置下能稳定跑通

那么 Engine 路线不一定更快，原因是：

- 当前核心故障已经下沉到 decode `KVReceiver`
- 这属于底层 transfer/backend 问题
- 并非 router 壳层问题

因此 Engine 路线最多能：

- 让问题更干净地暴露

但未必能：

- 真正绕开问题

## 建议结论

### 建议 1

若目标是“验证 router 是否是主要干扰源”，可以做一个最小版 `Engine-PD` 实验。

建议范围：

- 不先追求完整多请求路由
- 只做单请求、单 batch、固定 `bootstrap_room`
- 直接由 Python orchestrator 驱动两个 Engine

### 建议 2

若目标是“尽快拿到可跑通的 PD 链路”，优先级不应放在 Engine 改造，而应放在：

- 更换 transfer backend，例如 `mooncake`
- 查清 `nixl + dummy load-format + DeepSeek MLA + PD` 是否是已知限制
- 定位 decode 侧 `KVReceiver` 失败原因

### 建议 3

若目标只是“研究 prefill / decode 分离机制本身”，Engine 路线是很好的教学与调试入口。

因为它可以帮助我们把系统拆成三层看：

1. 顶层 server/router 编排
2. 中层 Engine 请求与子进程调度
3. 底层 PD transfer/runtime

## 最终判断

可以从 `Engine` 入口直接试运行解耦的 `P` 实例或 `D` 实例。  
但这更适合作为：

- 调试/分析工具
- router 去噪手段
- 最小复现实验平台

而不应被理解为：

- 一条天然更稳、更简单、能自动绕开 PD bug 的替代方案

对当前任务的最佳表述是：

- `Engine-PD` 值得做
- 但其价值主要在“更干净地暴露问题”
- 而不是“避免底层 transfer backend 问题”

