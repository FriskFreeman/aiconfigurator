# AIC 三类 GEMM 经验模型与四档参数终稿

## 1. 目的与模型选择

本文将三轮建模中选出的最佳可解释模型压缩为便于配置和后续集成的四档参数：

| 档位 | 含义 | 默认 |
|---|---|---:|
| `precise` | 原核心训练集 pooled refit 的全精度数据驱动系数 | 否 |
| `standard` | 弱硬件相关的中央工程估计，也是原 v1 取整终稿 | **是** |
| `high` | 高时延、偏保守的工程估计 | 否 |
| `low` | 低时延、偏乐观的工程估计 | 否 |

`low/standard/high` 是用于无目标卡标定时切换的工程场景，不是统计置信区间或可保证覆盖真值的上下界。`precise` 用于复现原建模结果，不参与三档单调区间的定义。

| 算子路径 | 最终结构 | 参数数 |
|---|---|---:|
| BF16 GEMM | Balanced Sum 3P | 3 |
| FP8 Block / DeepGEMM | Gated Serial Quant Max 8P | 8 |
| SGLang FP8 scaled-mm | Gated Serial Quant Max 5P | 5 |

需要特别区分历史产物：DeepGEMM 子目录的 `derived/final/deepgemm_fp8_model.json` 和 `scripts/deepgemm_fp8_model.py` 仍对应较早的 Sum 8P；后续选出的 Max 标杆位于 `derived/max_models/`。本文和中央终稿代码使用后者的 `max_gate_serial_quant_8p`，不沿用旧 Sum 默认值。

`standard` 不是重新拟合所得，而是对全核心硬件 refit 参数做有约束的两位有效数字近似。取整原则为：

1. 启动/floor 优先取整到整微秒；
2. eta 保留两位小数，避免把结构补偿项解释成过高精度的硬件常数；
3. 只有 DeepGEMM 中数据明确要求的 Hopper/Blackwell 差异才保留架构分组；
4. 用原始核心数据复算，要求取整引入的 P99 预测变化控制在约 4% 内；
5. 不为了让数字更整齐而统一三类算子的 eta，因为三者的流量公式、计时边界和 Sum/Max 语义不同。

纯标准库终稿代码位于同目录的 `gemm_empirical_final.py`，不读取 JSON，也不依赖 NumPy、Pandas 或拟合库。

终稿代码版本为 `2026-07-29.aic-gemm-empirical-v3`。

## 2. BF16 Balanced Sum 3P

### 2.1 公式

```text
F = 2*M*N*K
B = 2*(M*K + K*N + M*N)

t_mem     = B / BW
t_compute = F / PeakBF16

t = t_launch
  + t_mem / eta_mem
  + t_compute / eta_compute
```

### 2.2 参数

| 参数 | `precise` | `standard` | `high` | `low` | 解释 |
|---|---:|---:|---:|---:|---|
| `t_launch` | 2.0777778 us | **2.0 us** | 3.0 us | 1.5 us | 通用 kernel/调度启动平台 |
| `eta_mem` | 0.7682799 | **0.77** | 0.60 | 0.96 | Sum 语义下的有效显存系数 |
| `eta_compute` | 0.8953233 | **0.90** | 0.70 | 1.00 | Sum 语义下的有效 BF16 计算系数 |

最终经验式：

```text
t_bf16 = 2.0 us + t_mem/0.77 + t_compute/0.90
```

这里 `eta_compute=0.90` 较高，不能解读成“所有 BF16 GEMM 都达到 90% MFU”。Sum 同时累加 memory 与 compute，eta 是对该结构和逻辑最小流量的有效补偿参数。

## 3. FP8 Block / DeepGEMM Max 8P

### 3.1 公式

collector 计时边界是 activation block quant kernel 与 DeepGEMM kernel 的串行总和：

```text
F = 2*M*N*K

S_a = 4*M*ceil(K/128)
S_b = 4*ceil(N/128)*ceil(K/128)

B_gemm  = M*K + N*K + 2*M*N + S_a + S_b
B_quant = 3*M*K + S_a

q[a]    = (B_quant/BW) / eta_quant[a]
gemm[a] = max((B_gemm/BW)/eta_mem[a],
              (F/PeakFP8)/eta_compute)
body[a] = q[a] + gemm[a]

t = t_floor[a] + body[a]
  + rho*t_floor[a]*body[a]/(body[a]+t_floor[a])
```

`a` 取 Hopper 或 Blackwell。quant 是独立前置 kernel，只有 DeepGEMM 内部的访存和计算进入 Max envelope。

### 3.2 参数

| 参数 | `precise` | `standard` | `high` | `low` |
|---|---:|---:|---:|---:|
| `t_floor_hopper` | 3.8656665 us | **4.0 us** | 5.0 us | 3.0 us |
| `t_floor_blackwell` | 10.3787779 us | **10.0 us** | 13.0 us | 8.0 us |
| `eta_mem_hopper` | 0.8581148 | **0.86** | 0.72 | 1.00 |
| `eta_mem_blackwell` | 0.4803932 | **0.48** | 0.40 | 0.60 |
| `eta_quant_hopper` | 0.6765142 | **0.68** | 0.57 | 0.85 |
| `eta_quant_blackwell` | 0.3368849 | **0.34** | 0.28 | 0.43 |
| `eta_compute` | 0.6486840 | **0.65** | 0.54 | 0.81 |
| `rho` | 2.1264742 | **2.1** | 2.6 | 1.6 |

这组数值保留了最重要的经验事实：

- DeepGEMM 的共享有效计算系数约为 **0.65**；
- Blackwell 的逻辑访存与 quant eta 约为 Hopper 的一半；
- Blackwell 复合算子的启动 floor 约为 **10 us**，显著高于 Hopper 的 **4 us**；
- `rho` 约为 **2.1**，代表中大 body 区额外过渡税最多饱和到约 `2.1*t_floor`。

这些差异不能仅由 Peak FP8 和 HBM BW 推导，因此 DeepGEMM 接口必须显式输入 `hopper` 或 `blackwell`。四个档位都保留两套 `floor/eta_mem/eta_quant`，没有把 DeepGEMM 强行压成跨架构全局参数；共享的仍只有 `eta_compute` 和 `rho`。

该模型要求 `N/K >= 128`，但不要求 `N/K` 按 128 对齐。非整块形状仍按逻辑 `M/N/K` 计算 FLOPs 和主数据流量，block scale 元数据使用 `ceil(N/128)` 与 `ceil(K/128)`，因此可直接覆盖 DeepSeek 中 `N=2112` 等带尾块 shape。建模参数和 roofline 主体不因对齐性改变。

## 4. SGLang FP8 Scaled-MM Max 5P

### 4.1 公式

collector 计时边界为 per-token quant 与 `fp8_scaled_mm`：

```text
F = 2*M*N*K
B_gemm  = M*K + N*K + 2*M*N + 4*M + 4*N
B_quant = 3*M*K + 4*M

q    = (B_quant/BW) / eta_quant
gemm = max((B_gemm/BW)/eta_mem,
           (F/PeakFP8)/eta_compute)
body = q + gemm

t = t_floor + body
  + rho*t_floor*body/(body+t_floor)
```

### 4.2 参数

| 参数 | `precise` | `standard` | `high` | `low` |
|---|---:|---:|---:|---:|
| `t_floor` | 4.9858815 us | **5.0 us** | 6.5 us | 4.0 us |
| `eta_mem` | 0.7011994 | **0.70** | 0.58 | 0.88 |
| `eta_compute` | 0.6165924 | **0.62** | 0.52 | 0.78 |
| `eta_quant` | 0.4626841 | **0.46** | 0.38 | 0.58 |
| `rho` | 1.5129826 | **1.5** | 1.9 | 1.1 |

最终参数可直接记为：启动 **5 us**，GEMM memory/compute/quant eta 为 **0.70/0.62/0.46**，过渡 `rho=1.5`。

与 DeepGEMM 不同，这批数据在六张数据中心卡上不需要 H/B 分组。该结论只适用于当前 sgl-kernel 路径；L40S 和 RTX Pro 6000 的严格零样本误差仍然较高。

## 5. 工程档位的构造与保证

### 5.1 构造方法

`standard` 保留 v1 已审计的两位有效数字参数。`high/low` 参考 BF16、DeepGEMM 和 SGLang FP8 建模中的 LOSO 参数波动与原模型约 16%-19% 的平均误差尺度，但不复制任何一张 GPU 的单卡拟合结果。

工程档位主要围绕 `standard` 做约 20%-30% 的低精度扰动并再次取整。BF16 `high` 的 launch 扩大到 3 us，用于覆盖基础规格无法解释的小 kernel 固定时延尾部：

```text
high latency: 增大 launch/floor/rho，降低各 eta
low latency : 降低 launch/floor/rho，提高各 eta
```

这种构造保持模型公式、逻辑工作量和硬件 Peak/BW 输入不变，只改变规格表无法表达的固定开销、持续效率和过渡税假设。BF16 与 SGLang FP8 使用跨硬件通用三档；DeepGEMM 因实测稳定显示代际路径差异，继续在每一档内部保留 Hopper/Blackwell 特异参数。

### 5.2 单调性

对固定硬件、合法 shape 和同一算子路径，参数逐项保证：

```text
latency(low) <= latency(standard) <= latency(high)
```

BF16 的各加法项天然有序。两套 FP8 模型中，quant、memory、compute、floor 和 transition 均按同方向变化，因此经过 `max(memory, compute)` 后仍保持有序。测试覆盖三类模型的多种规模与 DeepGEMM 两个架构分支。

`precise` 接近 `standard`，但它是独立的 refit 复现值，不承诺位于三档预测的哪一侧，也不应被解释为对未知硬件更准确。

### 5.3 历史数据宽度抽查

下面把同一 shape 的历史实测与 `low/high` 预测比较，仅用于检查工程档位没有退化成过窄的取整扰动：

| 路径与抽查范围 | 行数 | 实测落入 `[low, high]` | `low/standard` 中位数 | `high/standard` 中位数 |
|---|---:|---:|---:|---:|
| BF16，7 张核心卡，SGLang 0.5.9 | 223692 | 70.89% | 0.81 | 1.31 |
| DeepGEMM，6 张 H/B 卡，0.5.10 压力集 | 177154 | 75.65% | 0.75 | 1.31 |
| SGLang FP8，6 张核心卡 LOSO 汇总 | 214451 | 81.46% | 0.76 | 1.31 |

这些覆盖率不是训练目标，也不能跨数据集直接排名。它们只说明三类 `high` 的典型宽度现已接近，且工程区间仍会漏掉约 19%-29% 的历史点，因此绝不能当作置信区间或硬性能边界。

## 6. 取整损失审计

下表使用同一批核心硬件全量数据，分别代入原 core-refit 参数和固定经验参数。它是“参数取整损失”检查，不是重新进行 LOSO；外推能力仍应以各项目原 LOSO 指标为准。

| 数据集 | 原 refit pooled MAPE | 经验值 pooled MAPE | 变化 | 原/经验最差硬件 MAPE | 原/经验 pooled P99 APE |
|---|---:|---:|---:|---:|---:|
| BF16 | 18.36% | 18.57% | +0.21 pp | 22.31% / 22.84% | 62.86% / 63.81% |
| FP8 DeepGEMM | 15.60% | 15.72% | +0.12 pp | 16.39% / 16.56% | 58.53% / 58.60% |
| FP8 SGLang | 16.23% | 16.20% | -0.03 pp | 17.66% / 17.65% | 64.17% / 64.11% |

经验参数相对精确 refit 预测本身的差异：

| 数据集 | 中位差异 | P99 差异 | 最大差异 |
|---|---:|---:|---:|
| BF16 | 0.81% | 3.73% | 3.74% |
| FP8 DeepGEMM | 1.82% | 3.58% | 3.65% |
| FP8 SGLang | 0.16% | 0.52% | 0.54% |

三套取整误差都远小于模型自身约 16%-19% 的跨 shape 误差，因此保留更多小数没有实际预测价值。

### 6.1 原始 LOSO 基准

经验模型的预期误差尺度仍以原模型 LOSO 为准：

| 模型 | 核心 LOSO 平均 MAPE | 最差折 | 平均 P99 APE |
|---|---:|---:|---:|
| BF16 Sum 3P | 19.14% | 22.91% | 62.22% |
| DeepGEMM Max 8P | 16.06% | 16.87% | 58.59% |
| SGLang FP8 Max 5P | 16.39% | 18.25% | 63.71% |

### 6.2 外部压力检查

- BF16 经验值在 L40S/Pro6000 上的 MAPE 为 37.82%/20.31%，原精确值为 38.50%/20.33%。
- SGLang FP8 经验值在 L40S/Pro6000 上的 MAPE 为 58.39%/34.85%，原精确值为 58.26%/34.94%。
- DeepGEMM 0.5.10 六卡中，经验值逐卡 MAPE 与原精确值的最大差异约 0.27 个百分点，没有出现版本压力失稳。

取整没有改变原有外推结论：BF16 相对稳健；DeepGEMM 只支持已知 Hopper/Blackwell 类；SGLang FP8 对 L40S/Pro6000 的 kernel recipe 差异仍无能为力。

## 7. 统一理解与使用边界

### 7.1 eta 是结构相关经验系数

三套 eta 不可横向当作硬件利用率排名：

- BF16 使用 Sum，memory 与 compute 均进入加法；
- 两套 FP8 使用 `quant + max(memory, compute)`；
- FP8 的计时还包含量化 kernel；
- 三套逻辑流量对 scale metadata 的计算不同。

例如 BF16 `eta_compute=0.90` 与 DeepGEMM `0.65` 的差异主要反映模型结构和 kernel 路径，不表示 DeepGEMM Tensor Core 只有 BF16 的 72% 性能。

### 7.2 参数优先级

用于新系统时，建议按以下顺序校准：

1. 先校准 `t_launch/t_floor`，它决定最小算子下界；
2. 对含 quant 的 FP8 路径，若有 quant-only 数据，优先校准 `eta_quant`；
3. 用明显 memory-side 与 compute-side 的大算子分别校准 `eta_mem`、`eta_compute`；
4. 只有中等规模过渡区存在稳定偏差时再调整 `rho`。

没有目标硬件数据时，默认使用 `standard`；需要服务容量保守估计或乐观性能估计时切换 `high/low`。原 LOSO MAPE/P99 仍应被视为模型误差尺度，三档不能替代统计校准，也不能把点预测变成高精度真值。

## 8. 终稿代码接口

```python
from gemm_empirical_final import (
    bf16_gemm_latency_us,
    deepgemm_fp8_latency_us,
    sglang_fp8_latency_us,
)

bf16_us = bf16_gemm_latency_us(
    4096, 4096, 4096,
    peak_bf16_flops=989e12,
    mem_bandwidth_bytes_s=3.35e12,
    parameter_level="standard",  # 默认值，可选 precise/high/low
)

deepgemm_us = deepgemm_fp8_latency_us(
    4096, 4096, 4096,
    peak_fp8_flops=1.979e15,
    mem_bandwidth_bytes_s=3.35e12,
    architecture="hopper",
    parameter_level="high",
)

sglang_fp8_us = sglang_fp8_latency_us(
    4096, 4096, 4096,
    peak_fp8_flops=1.979e15,
    mem_bandwidth_bytes_s=3.35e12,
    parameter_level="low",
)
```

对应的 `estimate_bf16_gemm`、`estimate_deepgemm_fp8`、`estimate_sglang_fp8` 会返回 `LatencyBreakdown`，包含启动、quant、有效 GEMM memory、有效 compute、过渡项和最终时延，便于后续模型集成与问题定位。

也可直接取得不可变参数对象：

```python
from gemm_empirical_final import (
    Bf16Sum3PParameters,
    get_bf16_parameters,
)

precise = get_bf16_parameters("precise")
custom = Bf16Sum3PParameters(t_launch_us=3.0, eta_mem=0.7, eta_compute=0.8)
```

显式 `params=custom` 保持向后兼容。为避免参数来源歧义，不能同时传入自定义 `params` 和非 `standard` 的 `parameter_level`。旧常量 `BF16_SUM_3P`、`DEEPGEMM_MAX_8P`、`SGLANG_FP8_MAX_5P` 均继续指向对应的 `STANDARD` 常量。

## 9. 最终默认参数清单

```text
BF16 Sum 3P
  launch_us  = 2.0
  eta_mem    = 0.77
  eta_compute= 0.90

DeepGEMM Max 8P
  floor_us   = Hopper 4.0, Blackwell 10.0
  eta_mem    = Hopper 0.86, Blackwell 0.48
  eta_quant  = Hopper 0.68, Blackwell 0.34
  eta_compute= 0.65
  rho        = 2.1

SGLang FP8 Max 5P
  floor_us   = 5.0
  eta_mem    = 0.70
  eta_compute= 0.62
  eta_quant  = 0.46
  rho        = 1.5
```

以上是默认 `standard` 参数。三套工程场景和 `precise` 均以内置不可变常量提供。若后续引入目标卡校准，应另建自定义或 calibrated profile，不覆盖本终稿常量，以保留统一对照基准。
