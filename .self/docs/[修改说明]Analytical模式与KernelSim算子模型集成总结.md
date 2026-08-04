# [修改说明] Analytical 模式与 KernelSim 算子模型集成总结

## 1. 修改目的

本轮将此前独立完成的 GEMM、FA/MLA、BMM 和 MoE 性能模型集成到 AIC SDK 的 `kernelsim` 组件，并在 `PerfDatabase` 中新增 `ANALYTICAL` 数据模式。

目标是在新 GPU、缺少实测数据或数据库覆盖不足时，仍能依据硬件公开规格和算子工作量快速完成模型性能评估。相较原有 SOL/EMPIRICAL：

- 不再只使用统一峰值效率或固定经验缩放；
- 区分算子类型、精度、执行阶段和主要软件实现；
- 对小算子启动成本、量化流量、attention tile/reuse、decode task service 等因素进行显式建模；
- 保留 `standard/low/high` 三档参数，用于无校准数据时给出中心、低位和高位 latency 估计。

ANALYTICAL 仍是工程理论模型，不等同于 silicon ground truth。模型参数主要源于 SGLang 算子和采集边界，迁移到其他 backend 时会发出兼容性警告。

## 2. 核心目录和调用链

新增组件位于：

```text
src/aiconfigurator/sdk/kernelsim/
├── analytical.py       # AIC 适配层和统一策略
├── gemm/               # BF16、SGLang FP8、DeepGEMM
├── fa/                 # FA2/FA3、MLA及硬件配置
├── bmm/                # 通用 BMM
└── moe/                # SGLang fused MoE
```

主调用链为：

```text
CLI / Python API / Web configuration
  -> TaskConfig
  -> PerfDatabase.set_analytical_config()
  -> Operation.query()
  -> PerfDatabase.query_*()
  -> sdk.kernelsim.analytical adapter
  -> operator-specific estimator
```

`PerfDatabase` 继续统一返回毫秒；独立 kernelsim 模型内部主要使用微秒，转换集中在适配层完成。

## 3. 四类算子模型

### 3.1 GEMM

支持：

- BF16 GEMM；
- SGLang FP8 GEMM，作为 FP8 默认 recipe；
- 显式选择 Hopper 或 Blackwell 参数的 DeepGEMM FP8。

模型综合计算、逻辑访存和启动成本。DeepGEMM 保留架构特异参数，不根据硬件名称自动启用。

前端参数：

```text
--analytical-fp8-gemm-recipe sglang
--analytical-fp8-gemm-recipe deepgemm-hopper
--analytical-fp8-gemm-recipe deepgemm-blackwell
```

### 3.2 FA 和 MLA

FA 支持 FA2/FA3 显式选择，模型覆盖 prefill、decode、GQA L2 reuse、decode split-KV/task-service 和不同 dtype 峰值。

MLA 在统一 FA 框架上增加 QK/PV 非对称维度、latent KV 单份存储、MQA decode 等语义。当前生产接入只支持 MLA BF16；FP8 MLA 数据表现和 backend 兼容性尚不足，调用时明确报错。

### 3.3 BMM

BMM 模型已解除 DeepSeek 固定 pre/post shape 限制，可接受通用矩阵形状。参数仍来自既有拟合成果，没有因接口泛化重新拟合。

BF16 内存效率参数已按 collector L2 状态风险下调；FP8 参数保留，并对其较弱的参考性给出 warning。

### 3.4 MoE

MoE analytical 模型采用聚合 Sum-3P：

```text
latency = launch + FLOPs / effective_compute + bytes / effective_bandwidth
```

模型区分 BF16 Triton、FP8 block Triton 和 NVFP4 recipe，按 TP/EP 计算本地 intermediate、assignment、active experts、GEMM FLOPs、权重/scale、量化和组合流量。

该模型以均匀 expert workload 为主要假设，不模拟完整 DeepEP pipeline、per-expert histogram、padding和底层调度。因此普通 MoE analytical 模型不能直接当作 WideEP MoE core 的严格等价模型。

## 4. ANALYTICAL 模式行为

新增 `DatabaseMode.ANALYTICAL`。主要算子由 kernelsim 估算；尚未建立专用模型的内存型小算子继续使用已有理论估算。WideEP module op 可在 analytical 下拆成已有单算子序列，避免依赖 module 级 silicon 表。

通信来源可选：

```text
--analytical-communication-mode empirical  # 默认，无需通信实测表
--analytical-communication-mode silicon    # 使用已有通信表
```

默认参数档位：

```text
--analytical-level standard
```

也可选择 `low/high`。这里的 low/high 表示预测 latency 的低位/高位，不是置信区间。

## 5. MoE 通信 dtype 配置

本轮最后聚焦修改了 SOL/EMPIRICAL 通信公式的 dtype 配置。新增四项独立参数，默认均为 `half`：

```text
--analytical-moe-dispatch-dtype half
--analytical-moe-combine-dtype half
--analytical-wideep-dispatch-dtype half
--analytical-wideep-combine-dtype half
```

支持值为 `half/fp8/int8`。配置已贯通 CLI、Python API、`TaskConfig`、`AnalyticalConfig`、`MoEDispatch` 和 `PerfDatabase`。

普通 MoE 的 pre/post 通信分别读取 dispatch/combine dtype；WideEP 理论通信分别计算两段消息字节再相加。NCCL 和 custom all-reduce 的 SOL/EMPIRICAL 公式均按 `dtype.value.memory` 计字节。

典型 DeepEP 理论通信配置为：

```text
--analytical-wideep-dispatch-dtype fp8
--analytical-wideep-combine-dtype half
```

重要边界：这些参数不改变 silicon 查表。普通 MoE silicon 数据按固定 BF16 通信采集；WideEP/DeepEP silicon 表隐含采集时的固定通信语义，通常为 FP8 dispatch 和 BF16 combine，表本身没有 dtype 查询维度。

## 6. 前端和兼容性

配置入口包括：

- CLI 默认寻优入口；
- CLI 单配置 estimate 入口；
- Python `cli_default()` / `cli_estimate()`；
- `TaskConfig` 和 `PerfDatabase.set_analytical_config()`；
- Web 配置中的 analytical mode/communication source。

ANALYTICAL 允许非 SGLang backend 使用，但会告警。原因是公式本身通常可以计算，而 kernel 选择、量化流程和采集边界可能不同。

## 7. 已知限制

- 模型不保证替代 silicon 的绝对精度，只面向无数据条件下的快速估算和方案比较。
- FA/MLA 需要 SM、时钟、L2、shared memory、HBM带宽和分精度峰值等硬件信息；新增硬件时需要补齐配置。
- MLA FP8 暂不支持。
- MoE 默认均匀路由，不处理真实 expert histogram 和完整 DeepEP overlap。
- WideEP silicon dtype 是数据表隐含属性，不能由新通信 dtype 参数改写。
- `communication_mode=silicon` 仍要求目标硬件存在通信性能表。

## 8. 合并和迁移提示

跨分支合并时至少需要同步：

```text
src/aiconfigurator/sdk/kernelsim/
src/aiconfigurator/sdk/perf_database.py
src/aiconfigurator/sdk/operations.py
src/aiconfigurator/sdk/task.py
src/aiconfigurator/sdk/common.py
src/aiconfigurator/cli/
src/aiconfigurator/webapp/
相关模型 fallback 修改
相关 unit tests
```

不应随功能提交合入：

- `.self/analytical-mode/` 下的 Pareto 结果、图和中间 CSV；
- 本地采集临时目录和容器日志；
- kernelsim 中的 `__pycache__`、`build/`、`*.egg-info`；
- 为实验生成的大规模 systems data 快照，除非目标分支另有数据更新计划。

## 9. 验证

本轮完成：

- analytical 配置、GEMM/FA/BMM/MoE adapter 和 fallback 单元测试；
- Python compileall；
- MoE communication dtype validation；
- WideEP SOL 字节比例检查：`FP8 dispatch + BF16 combine` 为 `BF16 + BF16` 的 `3/4`；
- A100/H100/H200/Blackwell 类配置的 analytical Pareto 工作流实验。

实验结果保留在本地 `.self/analytical-mode/`，不纳入核心功能提交。
