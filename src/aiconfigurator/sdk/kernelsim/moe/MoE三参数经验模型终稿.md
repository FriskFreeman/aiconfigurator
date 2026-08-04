# SGLang MoE 全语义流量三参数经验模型终稿

## 1. 成品定位

本目录归档 AIC `collect_moe.py` 的 balanced MoE 单卡算子经验模型。模型服务于目标硬件只有
公开规格、没有 MoE 实测数据时的首轮性能评估。

最终保留此前选定的硬件无关 Sum-3P 主结构：

```text
t_pred = t_launch
       + FLOPs / (PeakFLOPS(dtype) * eta_compute)
       + B_semantic / (HBM_BW * eta_mem)
```

每个 recipe 只有三个参数：`t_launch`、`eta_compute`、`eta_mem`。模型不接收设备名、架构名或
产品代际，也没有 Hopper/Blackwell 专用分支。recipe 参数不同是因为三条 collector 路径的
算子边界和 kernel backend 不同，不是硬件专用建模。

正式支持：

| recipe | dtype/backend | 直接峰值字段 | collector 计时边界 |
|---|---|---|---|
| `bf16_triton` | BF16 Triton fused MoE | `bfloat16_tc_flops` | routing + copy + fused MoE + combine |
| `fp8_block_triton` | FP8-block Triton | `fp8_tc_flops` | routing + copy + 两次量化 + fused MoE + combine |
| `nvfp4_cutedsl` | NVFP4 FlashInfer CuteDSL | `fp4_tc_flops` | 已 dispatch 输入上的 quant + GEMM1 + act/quant + GEMM2 |

INT4 Marlin 不归档：当前系统 YAML 没有与其计算语义一致的直接 INT4 峰值字段，继续用 BF16
峰值或倍率替代会重新引入本次明确取消的问题。

## 2. 硬件峰值输入

调用方必须直接提供当前 recipe 对应的 `peak_flops_s`。代码用 `required_peak_field(recipe)` 返回
所需 YAML 字段，不存在 `peak_bf16_flops_s` 或 `compute_multiplier`。

拟合阶段实际读取的例子：

| 系统 | BF16 | FP8 | NVFP4 | HBM BW |
|---|---:|---:|---:|---:|
| H100/H200 | 989 TFLOP/s | 1978 TFLOP/s | 不支持 | 3.35/4.8 TB/s |
| B200 | 2250 TFLOP/s | 4500 TFLOP/s | 9000 TFLOP/s | 8.0 TB/s |
| GB200 | 2500 TFLOP/s | 5000 TFLOP/s | 10000 TFLOP/s | 8.0 TB/s |
| GB300 | 2500 TFLOP/s | 5000 TFLOP/s | **15000 TFLOP/s** | 8.0 TB/s |

GB300 的 FP4 峰值是 15 PFLOP/s，而 `BF16*4` 只有 10 PFLOP/s，证明倍率推导会产生实质错误。

## 3. 公共工作量

定义：

```text
T       = 全局 token 数
k       = top-k
E       = 全局专家数
EP, TP  = expert/tensor parallel size
H       = hidden size
J       = inter_size / TP
L       = T*k/EP              # 理想均匀分配下的本 rank expert assignments
E_local = E/EP
A       = min(E_local, L)     # balanced 情况下活跃本地专家数
c_q(x)  = ceil(x/q)
```

上下投影 tensor-core GEMM FLOPs 为：

```text
F_gate_up = 4*L*H*J
F_down    = 2*L*H*J
F_total   = 6*L*H*J
```

计算项只使用这两次 GEMM 的 FLOPs。routing、量化和 SiLU 不能用 tensor-core 峰值合理折算为
FLOPs，因此通过完整语义流量和启动项进入三参数模型。

`B_semantic` 是“所有稳定 tensor producer/consumer 边界均完整支付一次”的 no-cache 逻辑 HBM
需求，不是 profiler 测得的 DRAM transaction。它不把权重压成“有效权重字节”，而是分别保留
value 与 scale 的真实存储字节。kernel tile padding、内部排序 scratch、allocator traffic 和
L2 residency 不属于稳定算子语义，明确不计入；它们造成的综合差异由拟合参数和残差体现。

## 4. BF16 Triton 完整流量

EP=1 collector 的 `run_op` 在计时内执行 `select_experts`，将 weights/ids/router logits 拷贝到
预分配对象，再调用 fused MoE。

```text
B_routing_logits = 2*T*E + 4*T*E = 6*T*E
B_routing_topk   = 8*L + 16*L + 8*L = 32*L

B_gemm1_input    = 2*L*H
B_gemm1_weight   = 4*A*H*J           # BF16 W1: [A,2J,H]
B_gemm1_output   = 4*L*J             # BF16 gate+up
B_silu           = 4*L*J + 2*L*J
B_gemm2_input    = 2*L*J
B_gemm2_weight   = 2*A*H*J           # BF16 W2: [A,H,J]
B_gemm2_output   = 2*L*H
```

当 `k=1` 时 down GEMM 直接写最终输出，不存在独立 combine；当 `k>1` 时：

```text
B_combine = 2*L*H + 2*T*H
```

总量是以上各项之和。这里没有 weight scale，也没有量化流量。

## 5. FP8-block Triton 完整流量

FP8 value 每元素 1 byte，block size 是 `128x128`，scale 为 FP32。routing 与 BF16 路径相同。

输入量化发生在 routing 之前的 `T x H` tensor 上，GEMM 再按 top-k 读取 `L` 个 routed row：

```text
B_input_quant = 2*T*H + T*H + 4*T*c_128(H)
B_gemm1_input = L*H + 4*L*c_128(H)
```

GEMM1：

```text
B_W1_values = 2*A*H*J
B_W1_scales = 4*A*c_128(2J)*c_128(H)
B_gemm1_out = 4*L*J
```

GEMM1 后先执行 BF16 SiLU，再量化 down 输入。这两步不能遗漏或合并成模糊字节率：

```text
B_silu_and_down_quant
  = (4*L*J + 2*L*J)                 # SiLU read/write
  + (2*L*J + L*J + 4*L*c_128(J))   # quant read/value/scale write
  = 9*L*J + 4*L*c_128(J)
```

GEMM2：

```text
B_gemm2_input = L*J + 4*L*c_128(J)
B_W2_values   = A*H*J
B_W2_scales   = 4*A*c_128(H)*c_128(J)
B_gemm2_out   = 2*L*H
```

最后按 BF16 路径的规则支付 combine。审计过程中曾漏掉 SiLU 的 `6LJ`，重拟合时
`eta_compute` 会从约 0.65 漂移到 0.73 来补偿这个定义错误。这说明参数看似稳定并不能证明
流量公式正确，必须先固定可审计的算子语义。

## 6. NVFP4 CuteDSL 完整流量

这条 collector 路径与 Triton 不同：`x_dispatched` 和 `masked_m` 在计时前已经准备好，计时内
没有 top-k routing、dispatch 或最终 combine。FP4 value 两个元素打包为 1 byte，block scale
是每 16 个 value 一个 FP8 byte；二者始终单列，不使用 `0.5625 byte/element` 之类的折算量。

输入 quant 与 GEMM1：

```text
B_input_quant = 2*L*H + 0.5*L*H + L*c_16(H)
B_gemm1_input = 0.5*L*H + L*c_16(H)
B_W1_values   = A*H*J
B_gemm1_out   = 4*L*J
```

fused SiLU + quant 与 GEMM2：

```text
B_act_quant   = 4*L*J + 0.5*L*J + L*c_16(J)
B_gemm2_input = 0.5*L*J + L*c_16(J)
B_W2_values   = 0.5*A*H*J
B_gemm2_out   = 2*L*H
```

SGLang `scaled_fp4_quant` 的 scale tensor 有 128-row 和 4-scale-column 对齐。定义：

```text
S(rows,cols) = 128*c_128(rows) * 4*c_4(c_16(cols)) bytes

B_W1_scales = A*S(2J,H)
B_W2_scales = A*S(H,J)
```

四个阶段各读取长度 `E_local` 的 `masked_m`，每个活跃 group 又读取一个 FP32 scale/alpha：

```text
B_control = 16*E_local + 16*A
```

这里只计算 masked 有效行。collector 为 kernel 安全分配的 `max_m` 对齐行、kernel 私有 scratch
和缓存命中不是稳定语义流量，未伪装成 HBM 字节加入公式。

## 7. 数据与拟合

训练数据严格限制为：

```text
SGLang 0.5.9
distribution == balanced
moe_ep_size == 1
重复配置取 latency 中位数
```

| recipe | 数据点 | 硬件数 | shape 数 |
|---|---:|---:|---:|
| BF16 Triton | 8503 | 5 | 63 |
| FP8-block Triton | 7694 | 5 | 57 |
| NVFP4 CuteDSL | 4219 | 3 | 53 |

BF16/FP8 覆盖 B200、GB200、GB300、H100、H200；NVFP4 覆盖 B200、GB200、GB300。拟合使用
system/shape 平衡权重、log latency ratio 残差和 soft-L1 loss。硬件 LOSO 每次完整留出一张卡；
shape LOSO 的 shape ID 包含 `(H,I,k,E,TP)`。

## 8. 最终参数

| recipe / 参数 | `precise` | `standard` | `low` | `high` |
|---|---:|---:|---:|---:|
| BF16 `t_launch_us` | 49.205656 | **49.0** | 30.0 | 75.0 |
| BF16 `eta_compute` | 0.274238 | **0.27** | 0.42 | 0.17 |
| BF16 `eta_mem` | 0.825475 | **0.83** | 1.00 | 0.55 |
| FP8 `t_launch_us` | 45.877790 | **46.0** | 25.0 | 75.0 |
| FP8 `eta_compute` | 0.726047 | **0.73** | 1.00 | 0.47 |
| FP8 `eta_mem` | 0.702256 | **0.70** | 1.00 | 0.43 |
| NVFP4 `t_launch_us` | 42.498345 | **42.5** | 20.0 | 75.0 |
| NVFP4 `eta_compute` | 0.559608 | **0.56** | 0.90 | 0.34 |
| NVFP4 `eta_mem` | 0.757832 | **0.76** | 1.00 | 0.45 |

`precise` 复现统一 refit，`standard` 是默认工程舍入。`low` 表示低时延/乐观性能，`high` 表示
高时延/保守性能；代码对任意合法输入保证：

```text
latency(low) <= latency(standard) <= latency(high)
```

三档不是置信区间。它们在全部历史 EP=1 点上的覆盖率约为 BF16 `85.0%`、FP8 `87.0%`、
NVFP4 `78.4%`。NVFP4 档位明显更宽，原因是简单三参数模型本身残差更大。

## 9. 验收结果

| recipe | 硬件 LOSO 平均 MAPE | 最差硬件 MAPE | 平均 P90 APE | shape LOSO 平均 MAPE |
|---|---:|---:|---:|---:|
| BF16 Triton | 21.79% | 28.92% | 51.59% | 20.42% |
| FP8-block Triton | 25.22% | 32.40% | 47.81% | 24.02% |
| NVFP4 CuteDSL | 45.36% | 46.33% | 94.18% | 48.57% |

结论：BF16/FP8 可作为无实测数据时的阶段性粗估；NVFP4 只能视为高不确定性 baseline，不宜
用于窄裕量容量承诺。三参数方案保持了简单和可解释性，但没有能力描述 grouped GEMM 的专家
集中度、tile occupancy、不同 kernel config 和 cache residency，这正是主要残差来源。

## 10. 工程接口

```python
from sglang_moe_empirical_final import estimate_sglang_moe

result = estimate_sglang_moe(
    recipe="fp8_block_triton",
    num_tokens=1024,
    hidden_size=7168,
    inter_size=2048,
    topk=8,
    num_experts=256,
    moe_tp_size=1,
    moe_ep_size=1,
    peak_flops_s=5.0e15,              # 直接 FP8 峰值
    mem_bandwidth_bytes_s=8.0e12,
    parameter_level="standard",
)

print(result.latency_us)
print(result.required_peak_field)      # fp8_tc_flops
print(result.logical_bytes)
print(result.weight_value_bytes, result.weight_scale_bytes)
```

返回值包含计算/访存/启动分解和 routing、quant、GEMM IO、weight values、weight scales、combine、
control 等流量汇总。`work_terms(...)` 可取得每个 GEMM stage 的完整明细。

## 11. EP 边界

参数只由干净 EP=1 数据识别。`EP>1` 输出明确标记为 `ideal uniform EP extrapolation`：`L`、
`E_local` 和 `A` 按均匀 rank-local 工作量缩放，但 Triton routing 仍按全局 `T,E` 支付，表示
“全局选路 + 本 rank 均匀 expert body”的理想语义。它不是当前受 remote-id clamp 污染的 EP>1
collector 数据拟合结果，也不包含通信、最慢 rank、负载不均或 dispatch/combine 通信。

若调用方只需要 dispatch 后的 rank-local expert body，不能直接把该结果当作完整 EP 层时延；应
在系统级模型中单独拆出 routing/dispatch/communication，并避免重复支付 Triton routing 项。

## 12. 可复现制品

- 成品模型：`sglang_moe_empirical_final.py`
- 单元测试：`test_sglang_moe_empirical_final.py`
- 重拟合脚本：`sglang-moe-boundary-modeling/scripts/run_moe_full_traffic_modeling.py`
- 明细数据：`sglang-moe-boundary-modeling/derived/final_model/`
- LOSO 与流量图：`sglang-moe-boundary-modeling/plots/final_model/`
