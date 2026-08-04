# AIC Analytical 模式执行问题与算子覆盖分析

> 更新状态（2026-07-31）：本文最初记录的 WideEP MLA、MLAConcatK 和通信查表阻塞已经修复。第 1、2、6、7 节中描述的“当前缺口”保留为问题发现背景；实际实现状态以本节和第 11 节为准。

## 0. 本轮实施结论

`ANALYTICAL` 的默认执行目标现在是 **不依赖算子实测表**：

- 普通 `DeepSeekModel` 与 SGLang `WideEPDeepSeekModel` 中的 `WideEPContextMLA/WideEPGenerationMLA`，在非 analytical 模式下仍查询原 module 表；在 analytical 模式下由 `AnalyticalFallbackOp` 拆成 projection GEMM、MLA core、必要的 decode BMM 与 output GEMM。
- `MLAConcatK` 在 analytical 下复用其访存 SOL/0.7 经验估计，不读取 concat-K 表。
- NCCL、CustomAllReduce、DeepEP normal/LL 和 TRT-LLM all-to-all 在 analytical 下受 `communication_mode` 控制。默认 `empirical` 使用拓扑带宽公式；显式选择 `silicon` 才查询实测通信表。
- WideEP MoE compute 不是通信，analytical 下始终使用自身 compute SOL/0.4 估计，不受通信开关影响。

这使 Llama/Qwen3 与本轮覆盖的 DeepSeek V3 主路径能够在系统 YAML 存在、但 backend 算子 CSV 缺失时运行。它不意味着所有结果都来自新校准模型：小算子与通信仍有一部分是旧经验公式。

## 1. 结论摘要

当前 `ANALYTICAL` 已经不是简单的 SOL 别名。GEMM、普通 Attention、MLA attention core、MLA BMM 和 MoE compute 会调用 `sdk/kernelsim` 中的新模型；Embedding、ElementWise、量化辅助和基础通信则继续使用 AIC 原有的经验公式。

但它还不是一个对所有 AIC `Operation` 都封闭的“纯模型数据库模式”：

1. **Llama/Qwen3 的主干路径基本完整**。GEMM、FA、embedding、norm/activation、TP/PP 通信都有可执行估计，但小算子和通信仍是旧的粗粒度公式。
2. **普通 DeepSeek V3 的默认 MLA 路径尚未真正接入新 MLA 模型**。当前 `prefix=0` prefill 使用 `WideEPContextMLA`，decode 使用 `WideEPGenerationMLA`；这两个 query 没有 `ANALYTICAL` 分支，会继续查 `wideep_*_mla_perf.txt`。在无实测数据库情景下会失败。
3. **SGLang prefix prefill 还包含 `MLAConcatK`**。MLA core 能估计 prefix attention，但 concat-K 没有 analytical 分支，仍会查表。
4. **WideEP/DeepEP 与 TRT-LLM 专用路径没有整体覆盖**。其中部分通信已有旧 SOL/empirical 公式，部分仍依赖表。
5. **FA/MLA 的硬件适配目前仅真正准备了 A100 SXM 和 H100 SXM**。其他 GPU 即使 systems YAML 有峰值算力和 HBM 带宽，也缺少 SM、时钟、共享内存、L2 容量和 L2 带宽，不能运行细粒度 FA 模型。

因此，当前模式更准确的定义是：

> 对已归档的四类核心 kernel 使用新 analytical model；对少量基础操作使用旧经验公式；对尚未适配的 module/backend 专用操作仍可能依赖 silicon table。

它已经适合做逐算子研究和 Llama 主路径试运行，但还不能宣称在 DeepSeek V3 或任意新硬件上完全替代 PerfDatabase 实测数据。

## 2. 四种执行状态

| 状态 | 含义 | 当前代表 |
| --- | --- | --- |
| 新模型覆盖 | 调用 `sdk/kernelsim`，使用新建工作量抽象和 `standard/low/high` 参数 | GEMM、FA、MLA core、BMM、MoE compute |
| 旧公式 fallback | 不读取算子 CSV，但沿用原 SOL/empirical 公式和固定效率 | ElementWise、Embedding、compute-scale、NCCL、CustomAllReduce、P2P |
| 仍依赖实测表 | 没有识别 `ANALYTICAL`，进入原 `else` 的 SILICON/HYBRID 查询路径 | WideEP MLA、MLAConcatK、module-level MLA、DeepEP 专用操作等 |
| 显式不支持 | 主动报错，避免静默套用错误模型 | FP8 MLA、GEMM/MoE/BMM 未建模 dtype、缺少 FA 微架构信息的硬件 |

需要特别注意，“旧公式 fallback”不等于新的 analytical 模型；它只是保证基本执行链路不必查询 CSV。

## 3. 已接入模型覆盖

### 3.1 GEMM

`query_gemm()` 已接入：

- BF16：`estimate_bf16_gemm()`，使用 launch + memory + compute 的 Sum-3P 模型。
- FP8 默认：`estimate_sglang_fp8()`，包含量化流量、GEMM roofline、启动和 transition 项。
- 显式 DeepGEMM：`deepgemm-hopper` 或 `deepgemm-blackwell`。
- 参数档位：`standard/low/high`。

尚不支持 INT8、INT4 weight-only、SmoothQuant、NVFP4 等 GEMM 模式。遇到这些模式会显式报错，而不是套用 BF16。

`fp8_static` 仍沿用 AIC 的特殊语义：GEMM 结果会减去 `compute_scale`，低精度输入还可能减去 `scale_matrix`。这两个辅助 query 在 ANALYTICAL 下使用旧的 HBM/SOL 除以 `0.8` 的经验公式。该处理与“动态量化 GEMM 模型包含量化成本、static 路径扣除量化成本”的意图一致，但还没有做端到端边界复核。

### 3.2 普通 Attention

`query_context_attention()` 和 `query_generation_attention()` 已接入 FA2/FA3 roofline：

- 支持 MHA/GQA/MQA 头数关系。
- 支持 prefill、prefix prefill 和 decode 的核心 attention。
- 支持 BF16/FP8 核心 dtype。
- 包含 GQA L2 reuse、decode split-KV/task-service、wave/并行度、矩阵与向量资源项。
- `standard/low/high` 可切换，FA2/FA3 显式选择。

局限包括：

- sliding-window 目前主要通过裁剪有效 KV 长度近似，没有建模窗口 kernel 的独立调度特征。
- ContextAttention 外层额外加入 RoPE、KV write、可选 QK norm；这些仍通过 `query_mem_op()` 粗略估计。
- FP8 有效 dtype 根据 SGLang 的 KV-cache 行为解释，迁移到其他 backend 只发警告，不保证 collector boundary 一致。

### 3.3 MLA Attention Core

`query_context_mla()` 和 `query_generation_mla()` 已接入 BF16 MLA 模型：

- Prefill 使用不对称 QK/PV 维度和 latent KV storage。
- Decode 使用 `Hkv=1` 的 MQA、split-KV、task waves 和 task-service。
- 支持 prefix prefill 的 `query_length < sequence_length`。
- FP8 明确报错。

这里覆盖的是 **MLA attention core**，不等于整个 MLA module。Q/KV projection、output projection 应由独立 GEMM 计入；concat-K、RMSNorm、RoPE 或 backend module 封装也不自动包含。

### 3.4 BMM

`query_mla_bmm()` 已接入通用 BMM 公式的 DeepSeek pre/post wrapper：

- BF16 与 FP8 可执行。
- FP8 会保留低可信度 warning。
- 非校准形状属于公式级外推。

普通 Llama 不使用该路径。它主要服务于某些 granular MLA decode 路径；当前普通 SGLang DeepSeek V3 已改走 module-level MLA，所以该模型并不会自动出现在默认 decode 路径中。

### 3.5 MoE Compute

`query_moe()` 已接入 SGLang fused MoE 模型：

- BF16 Triton、FP8 block Triton、NVFP4 CuTeDSL recipe。
- 包含 routing、量化、两段 GEMM、activation/combine/control 的语义流量。
- EP=1 为主要校准范围；EP>1 是理想均匀分配外推。

当前 adapter 没有使用 `workload_distribution` 调整负载不均衡，因此 power-law、热点专家、EPLB 等标签不会改变新 MoE compute 结果。MoE dispatch/combine 通信也不包含在该 compute 模型中，而由独立 `MoEDispatch` 处理。

## 4. 旧公式覆盖的算子

### 4.1 Embedding 与 ElementWise

两者都调用 `query_mem_op()`：

```text
latency = bytes / (mem_bw * 0.8) + 3 us
```

其中：

- Embedding 只计算输出 token 向量的传输量。
- ElementWise 使用 `read_bytes + write_bytes`，固定按 BF16 两字节计算。
- LayerNorm/RMSNorm、SiLU/SwiGLU gate、residual add 等不同 kernel 都被压成同一种内存操作。

该方法至少考虑了固定启动时延，明显优于纯 `bytes / mem_bw`，但没有表达 reduction 遍数、向量指令、融合程度、中间张量、occupancy 或不同 dtype。因此它是 Llama/DeepSeek 中最重要的“可运行但精度仍粗糙”部分。

### 4.2 RoPE、QK Norm 与 KV Write

这些没有独立 Operation 模型，而是在 `ContextAttention.query()` 中以若干次 `query_mem_op()` 相加，并额外乘 `1.1` 修正。优点是避免完全忽略外围 kernel；缺点是次数和流量属于手工近似，也没有区分 kernel fusion。

### 4.3 NCCL、CustomAllReduce 与 P2P

- NCCL：拓扑带宽 SOL 除以固定 `0.8`。
- CustomAllReduce：ring 式流量 SOL 除以固定 `0.8`。
- P2P：消息大小除以带宽，再加 systems YAML 中固定 `p2p_latency`。

这些公式不读取性能 CSV，但不是本轮新模型。它们缺少消息规模分段、协议/通道选择、rank 数波形、NVLink/NVSwitch 拓扑细节和拥塞。

## 5. Llama/Qwen3 家族覆盖情况

普通 `LLAMAModel` 的主要算子如下：

| 模块 | Operation | ANALYTICAL 行为 | 状态 |
| --- | --- | --- | --- |
| token embedding | `Embedding` | 旧 HBM+常数公式 | 可运行，粗略 |
| RMSNorm/residual | `ElementWise` | 旧 HBM+常数公式 | 可运行，粗略 |
| QKV/proj/FFN/logits | `GEMM` | 新 GEMM 模型 | 已覆盖 |
| prefill/decode attention | `ContextAttention` / `GenerationAttention` | 新 FA 模型 | 已覆盖 |
| RoPE/QK norm/KV write | attention 内附加 mem-op | 旧 HBM+常数公式 | 可运行，粗略 |
| SwiGLU activation/gate | `ElementWise` | 旧 HBM+常数公式 | 可运行，粗略 |
| TP collective | `CustomAllReduce` | 旧拓扑公式 | 可运行，粗略 |
| PP transfer | `P2P` | 旧带宽+时延公式 | 可运行，粗略 |

所以对 BF16/FP8 的普通 Llama/Qwen3，计算主耗时通常已由新 GEMM/FA 模型覆盖。仍值得优先改进的是 decode 小 batch 下的 RMSNorm、activation、RoPE 和 collective，因为此时固定启动开销占比高，多个小 kernel 的累计误差可能不再是次要项。

## 6. DeepSeek V3 家族覆盖情况

### 6.1 普通 SGLang 路径的实际问题

当前 `DeepSeekModel` 的 MLA 路由是：

- `prefix == 0` prefill：`WideEPContextMLA`。
- `prefix > 0` prefill：GEMM + 可选 `MLAConcatK` + `ContextMLA` + GEMM。
- decode：`WideEPGenerationMLA`。

虽然类名包含 `WideEP`，普通 SGLang DeepSeek 也复用了这些 wrapper 来读取历史 module-level MLA 表。因此：

| 路径 | 当前结果 |
| --- | --- |
| prefix=0 prefill | `query_wideep_context_mla()` 无 ANALYTICAL 分支，误入表查询 |
| prefix>0 prefill | GEMM 与 MLA core 已覆盖，但 `query_mla_concat_k()` 仍依赖表 |
| decode | `query_wideep_generation_mla()` 无 ANALYTICAL 分支，误入表查询 |

这是当前 DeepSeek V3 无实测数据运行的首要阻塞，不是精度小问题。

合理修复方式不是给 WideEP 表查询简单套一个固定系数，而是让普通 DeepSeek 的 ANALYTICAL 路由绕过 module-table wrapper，显式拆成：

```text
projection GEMM + MLA core + output GEMM + 必要的小算子
```

或让 `query_wideep_*_mla()` 在 ANALYTICAL 下调用同一 MLA core，并明确处理 module wrapper 已包含/未包含的 projection 边界，避免重复计时。

### 6.2 MoE 与通信

普通 routed MoE compute 已覆盖，但以下部分仍需注意：

- router GEMM：已由新 GEMM 覆盖。
- shared experts：由普通 GEMM/ElementWise 覆盖。
- routed experts：由新 MoE 模型覆盖。
- generation shared/routed overlap：`OverlapOp` 取两组 latency 的 `max`，调度抽象可运行。
- dispatch/combine：基础 NCCL/CustomAllReduce 可走旧公式。
- SGLang DeepEP `query_wideep_deepep_normal/ll`：没有 ANALYTICAL 分支，仍会查表或进入未实现路径。
- TRT-LLM WideEP compute/all-to-all：没有接入新 MoE 模型，仍依赖专用表；其 SOL/empirical 分支虽存在，但 ANALYTICAL 当前不会自动选择它们。

### 6.3 其他重要未建模项

DeepSeek V3 中仍有这些高价值缺口：

1. **MLA module 边界模型**：当前最直接的执行阻塞。
2. **MLAConcatK**：prefix cache 场景的独立数据整理/cast kernel。
3. **RMSNorm、RoPE 与 latent cache update 的外围 kernel**：目前仅有统一 mem-op 或由 MLA core 部分吸收。
4. **MoE dispatch/combine 与负载不均衡**：尤其 EP>1、DeepEP、EPLB 和跨节点场景。
5. **小 GEMM/量化辅助的融合边界**：DeepSeek FP8 下 projection、scale、quant 是否独立执行会影响是否重复计时。
6. **CUDA Graph 与并发 overlap 的固定开销**：当前 `OverlapOp=max(A,B)` 不表达启动、依赖同步和不完全重叠。

## 7. 其他未覆盖 query

下列接口没有专门的 ANALYTICAL 分支，不能视为新模型覆盖：

- `query_mla_concat_k`
- `query_context_mla_module` / `query_generation_mla_module`
- `query_wideep_context_mla` / `query_wideep_generation_mla`
- `query_wideep_deepep_normal` / `query_wideep_deepep_ll`
- `query_wideep_moe_compute`
- `query_trtllm_alltoall`
- Mamba2、GDN、DSA、MHC、DeepSeek V4 module 路径

最后一组不属于本轮要求的 Llama/DeepSeek V3 主范围，但说明 `ANALYTICAL` 还不能作为所有 AIC 架构的全局通用模式。

## 8. 当前执行与工程问题

### 8.1 硬件信息不完整

GEMM/BMM/MoE 只需要 systems YAML 中的峰值算力和 HBM 带宽，因此可以覆盖更多设备。FA/MLA 额外需要：

- SM 数量和时钟；
- 每 SM shared memory；
- L2 容量和有效 L2 带宽；
- vector peak；
- 各 dtype matrix peak。

当前 adapter 按系统名只映射 A100/H100 的独立 JSON。结果是：

- H200、B200/B300、L40S、国产 GPU 等会直接报缺少 FA hardware details。
- `a100_pcie`、`h100_pcie` 会因名称匹配复用 SXM 微架构 JSON，其中 SM 数/时钟/带宽语义并不严格正确；HBM 带宽虽会被 YAML 覆盖，其他字段不会。

生产接入应将这些字段纳入 systems YAML 或正式的 kernel-sim hardware 子结构，避免按名称猜测。

### 8.2 Backend 兼容只是 warning

非 SGLang backend 会记录 warning 后继续。代码实现没有硬冲突，但预测边界可能不同：

- kernel recipe 不同；
- quant/cast 是否包含在计时区间不同；
- module fusion 和 KV cache layout 不同；
- FA2/FA3/FlashInfer/TensorRT-LLM kernel 路由不同。

因此“能运行”不代表模型已对该 backend 校准。

### 8.3 配置状态与缓存

`PerfDatabase` 保存可变的 `_analytical_config`，修改档位时会清 query cache。TaskRunner 通常在切换 database mode 时 deepcopy；但当全局缓存对象已经是 ANALYTICAL 且后续任务仅改变 level/recipe 时，仍可能复用并修改同一对象。CLI API 直接加载数据库时也会修改缓存实例。

串行运行通常没有问题，并发任务或交错使用不同档位时存在配置串扰风险。更稳健的方案是：

- 将 analytical config 放入 query cache key；或
- 创建不可变的 model service/registry；或
- 每个任务持有独立 PerfDatabase facade，不修改全局缓存对象。

### 8.4 Energy 缺失

新 analytical 返回 `energy=0.0`。因此 latency 可以用于性能估计，但功耗和能耗输出没有意义；多个结果聚合时 source 可以保留，energy 不能用于比较。

### 8.5 Web 与高级参数

Web 已可选择 `ANALYTICAL` 和 `empirical/silicon` 通信来源，计算模型仍使用默认 `standard + sglang FP8 + FA2`。`low/high`、DeepGEMM recipe 和 FA2/FA3 的完整显式参数目前在 CLI/API 中可用，尚未形成独立 Gradio 高级控件。

## 8.6 无实测表执行策略与接口

### Module fallback 的边界

`AnalyticalFallbackOp(primary, analytical_ops)` 依据数据库的默认 mode 选择路径：

```text
SILICON/HYBRID/EMPIRICAL/SOL -> primary module query
ANALYTICAL                  -> sum(analytical_ops)
```

DeepSeek MLA 的 qkv-a/downscale GEMM 原本就在 module collector 边界之外，因此保持在 wrapper 外，避免重复计时。wrapper 内拆解的是 q-b projection、kv-b projection、MLA attention core、output projection；decode 还包括 pre/post BMM。该拆解恢复主要 kernel 工作量，但不会复现 SGLang module 内的融合、并行 stream 和调度收益，所以更适合无数据预测，不应与原 module 表视为严格同语义的等价替换。

### MLAConcatK

MLAConcatK 按纯访存 kernel 建模：

```text
bytes = BF16 K_nope read
      + shared K_rope read
      + BF16 concatenated-K write
latency = bytes / HBM_bandwidth / 0.7
```

它解决 prefix prefill 的执行闭合问题。固定 `0.7` 没有刻画 head 数、对齐和 Triton/torch kernel 分支差异，因此属于 fallback，不是 MLA core 精确模型的一部分。

### 通信策略

前端、CLI 和 Python API 均提供 `analytical_communication_mode`：

| 值 | 行为 | 是否依赖通信表 |
| --- | --- | --- |
| `empirical`（默认） | topology SOL 除以固定效率；P2P 仍使用带宽加固定时延 | 否 |
| `silicon` | analytical 计算算子不变，通信 query 强制转入 SILICON | 是 |

CLI 参数为 `--analytical-communication-mode empirical|silicon`。Web 的聚合系统配置有一个选择框；分离式部署的 prefill/decode 各自独立选择。`silicon` 的设计目的是在已有通信数据时保留实测精度，它不满足“完全无数据表”的目标，缺表时报错是预期行为。

当前通信经验式仍有明确局限：DeepEP 的远端 rank 数使用 `min(topk, experts, ranks-1)`，流量按 BF16 token payload；all-to-all 使用近似去重后的 remote ranks；统一效率没有表达消息大小分段、协议、拥塞或节点拓扑。因此默认模式实现的是可执行、可解释的无表基线，而不是完成了通信性能模型校准。

### 剩余主要限制

- FA/MLA 细粒度硬件参数仍只完整适配 A100/H100；新 GPU 可能在 attention 阶段报硬件信息不足。
- FP8 MLA仍显式不支持。
- Mamba2、GDN、DSA、MHC、DeepSeek V4 等架构专用 module 不在本轮闭合范围。
- ElementWise、Embedding、RoPE、norm 和 activation 仍共享粗粒度 HBM+launch 公式。
- module 拆解忽略 kernel fusion 与 overlap，通信 empirical 忽略协议分段；这些是后续无表精度工作的主要误差源。

## 9. 建议补齐顺序

### 已完成：让 DeepSeek V3 主路径闭合

1. 为 `WideEPContextMLA/WideEPGenerationMLA` 的普通 DeepSeek module wrapper 增加 ANALYTICAL 路由。
2. 明确 module collector 边界，避免 projection GEMM 与 MLA core 重复计时。
3. 给 `MLAConcatK` 增加简单但独立的 launch + bytes 模型。
4. 增加 DeepSeek V3 prefill、prefix prefill、decode 的无 CSV 端到端测试。

### P1：提高 Llama/DeepSeek 小算子可信度

1. 将 RMSNorm、SiLU/SwiGLU、RoPE、residual add 分开建模。
2. 至少区分 reduction kernel 与纯 elementwise kernel。
3. 让 dtype、融合方式、token/head geometry 进入输入，而不是统一 BF16 bytes。

### P1：完善 MoE 系统语义

1. 建立 dispatch/combine analytical model。
2. 将 workload skew、EP、EPLB 和 active-expert occupancy 纳入模型。
3. 明确 SGLang fused MoE compute 与 DeepEP 通信的计时边界。

### P2：硬件与工程化

1. 扩展 systems YAML 的 FA/MLA 微架构字段。
2. 去除按 `a100/h100` 名称加载 SXM JSON 的临时逻辑。
3. 使 analytical 配置不可变并进入 cache key。
4. 增加 energy 模型和 Gradio 高级选项。

## 10. 当前可用性判断

| 使用场景 | 判断 |
| --- | --- |
| H100/A100、SGLang、Llama/Qwen3、BF16/FP8 | 可用于主干算子和端到端趋势评估；小算子/通信仍是粗模型 |
| H100、SGLang、DeepSeek V3、逐算子 granular MLA | GEMM/MLA core/BMM/MoE 可用 |
| H100、SGLang、DeepSeek V3 默认模型执行 | 默认 empirical 通信策略下可无表运行；module 拆解不表达融合/overlap |
| DeepSeek V3 prefix prefill | MLA core 与 concat-K fallback 均可无表运行 |
| 非 SGLang backend | 可尝试，必须视为跨 backend 外推 |
| H200/Blackwell/其他 GPU 的 FA/MLA | 当前缺硬件微架构输入，不能直接运行 |
| INT8/INT4/NVFP4 GEMM 或 FP8 MLA | 当前不支持；FP8 MLA会明确报错 |
| 功耗/能耗评估 | 不支持，新模型 energy 为 0 |

总体而言，默认 `ANALYTICAL + empirical communication` 已闭合 Llama/Qwen3 和本轮 DeepSeek V3 主路径的算子数据依赖。下一阶段重点应从“能否无表执行”转向小算子、通信、module fusion/overlap 与跨硬件字段的预测质量。
