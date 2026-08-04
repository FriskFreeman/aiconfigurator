# SGLang 通用 BMM 经验模型终稿

## 1. 成品定位

成品模型现提供任意正整数 `batch,M,N,K` 的通用 BMM 接口：

```python
estimate_bmm(batch, m, n, k, dtype, peak_flops_s, mem_bandwidth_bytes_s)
```

模型公式和流量可以覆盖一般 batched matmul，但经验参数仍来自 AIC
`collect_mla_bmm.py` 的 DeepSeek MLA generation pre/post 数据。因此需要严格区分：

```text
通用公式能力     = 任意 B/M/N/K
参数校准范围     = DeepSeek (K,N)=(128,512)/(512,128)
其他矩阵形状     = generic_shape_extrapolation
```

原 `estimate_mla_bmm(num_tokens,num_heads,op,...)` 保留为兼容包装器。它只负责把：

```text
pre : batch=heads, M=tokens, K=128, N=512
post: batch=heads, M=tokens, K=512, N=128
```

映射到新的通用接口，不维护另一套模型或参数。

版本：`2026-07-31.aic-bmm-empirical-v3`。

## 2. 通用工作量

### BF16

```text
F = 2*B*M*N*K

B_activation = 2*B*M*(K+N)
B_weight     = 2*B*K*N
B_logical    = B_activation + B_weight
```

### FP8 compound

FP8 recipe 仍严格对应 collector 的 activation per-tensor quant 与 SGLang
`bmm_fp8` 串行边界：

```text
S_a = 4*B
S_b = 4*B*ceil(N/128)*ceil(K/128)

B_quant = 3*B*M*K + S_a
B_gemm  = B*(M*K + K*N + 2*M*N) + S_a + S_b

B_activation = 4*B*M*K + 2*B*M*N + 2*S_a
B_weight     = B*K*N + S_b
B_logical    = B_activation + B_weight
```

任意 `N/K` 均允许，scale 元数据使用向上取整。该公式不能自动代表其他 FP8 BMM
的 scale 粒度、cast、accumulator 或 kernel recipe。

## 3. 时延模型

```text
t_compute = F / PeakFLOPS(dtype) / eta_compute
t_memory  = B_logical / HBM_BW / eta_mem(dtype)

t = t_floor(dtype) + max(t_compute,t_memory)
```

原 5,088 个校准点全部位于 memory-side。`eta_compute=0.65` 是来自 GEMM 的工程先验，
不是由 BMM 数据识别的参数。通用形状若进入 compute 分支，返回结果会加入
`COMPUTE_BRANCH_WARNING`，不得宣称该分支已通过 BMM 实测验证。

## 4. BF16 L2 偏差修正

collector 采用重复 warm replay，没有在每次算子执行前严格驱逐 L2。对于固定权重和
小工作集，实测时延可能包含显著缓存收益；在“完整逻辑流量全部由 HBM 支付”的公式中，
这种收益会被拟合成偏高的 `eta_mem`。

本轮不重新拟合参数，而是只对生产 BF16 `low/standard/high` 的 memory eta 施加约
20% 下调：

| 档位 | BF16 原 eta | BF16 v3 eta | floor | compute eta |
|---|---:|---:|---:|---:|
| low | 1.00 | **0.80** | 3.0 us | 0.81 |
| standard | 0.88 | **0.70** | 4.2 us | 0.65 |
| high | 0.70 | **0.56** | 5.5 us | 0.52 |

该修正把默认语义从“尽量复现 warm collector”转为更保守的
`conservative_full_payment` 工程估计。它是基于已知采集偏差的政策修正，不是新数据
校准结果。

`precise` 保留原 pooled refit：

```text
BF16 floor   = 4.197893 us
BF16 eta_mem = 0.881009
```

因此 precise 仍可复现历史实验，但它不属于新的三档完整流量假设，也不保证位于
low/high 之间。

## 5. FP8 参数与可靠性警告

FP8 参数本轮完全不修改：

| 档位 | FP8 eta_mem | floor |
|---|---:|---:|
| low | 0.28 | 5.5 us |
| standard | 0.22 | 7.1 us |
| high | 0.18 | 9.0 us |
| precise | 0.219431 | 7.136533 us |

FP8 已经表现出非常低的复合 memory eta；继续按 BF16 相同比例降低会把 collector/backend
劣化进一步固化为硬件规律，缺乏依据。现有数据中 FP8 相比 BF16 经常没有加速甚至更慢，
可能同时受到 quant kernel、SGLang `bmm_fp8` 实现和采集 recipe 影响。

任何 FP8 `estimate_bmm()` 或兼容包装器调用都会：

1. 发出 `BmmFp8ReliabilityWarning`；
2. 在 `result.warnings` 中写入同一低可信度说明；
3. 返回 `recipe=sglang_quant_bmm_fp8`，防止被解释为通用 FP8 BMM。

警告不阻止预测，但调用方不应使用该结果证明 FP8 理论上慢于 BF16，也不应迁移到不同
FP8 scale/layout/backend。

## 6. API

### 通用接口

```python
from mla_bmm_empirical_final import estimate_bmm

result = estimate_bmm(
    batch=16,
    m=256,
    n=768,
    k=384,
    dtype="bf16",
    peak_flops_s=989e12,
    mem_bandwidth_bytes_s=3.35e12,
    parameter_level="standard",
)

print(result.latency_us)
print(result.scope)       # generic_shape_extrapolation
print(result.warnings)
```

### DeepSeek 兼容接口

```python
from mla_bmm_empirical_final import estimate_mla_bmm

result = estimate_mla_bmm(
    num_tokens=512,
    num_heads=128,
    op="pre",
    dtype="bf16",
    peak_flops_s=989e12,
    mem_bandwidth_bytes_s=3.35e12,
)
```

辅助接口：

```text
bmm_work_terms(B,M,N,K,dtype)
bmm_latency_us(...)
bmm_latency_s(...)
```

旧的 `work_terms()`、`mla_bmm_latency_us/s()`、`MlaBmmParameters` 和 `MLA_BMM_*`
名称继续可用。新代码应优先使用 `BmmParameters`、`BMM_*` 和通用接口。

## 7. Scope 与输出

当 `(K,N)` 匹配原 pre/post 时：

```text
scope = calibrated_deepseek_matrix_geometry
op    = pre/post
```

其他形状：

```text
scope = generic_shape_extrapolation
op    = generic
warnings 包含 GENERIC_SHAPE_WARNING
```

结果还包含 model/version、recipe、floor、memory/compute body、Roofline 分支、eta、
FLOPs、activation/weight/quant/GEMM 流量。Scope 只表示矩阵几何是否来自训练族；即使
K/N 匹配，超出原 token/head 范围仍需按外推谨慎使用。

## 8. 使用边界

- BF16 只适用于实际走 batched-matmul kernel 的路径；`batch=1` 也不能据此替代已经有
  独立校准的 GEMM recipe。
- FP8 只适用于相同 quant + `bmm_fp8` 边界。
- 任意形状接口没有重新拟合，不能宣称已经成为跨 shape 高精度通用模型。
- 模型不表达 L2 residency、tile、split-K、具体 cuBLAS 算法、epilogue、CUDA Graph 或
  cache eviction。
- 新 BF16 三档偏向完整流量保守估计；若目标就是复现 warm collector，应使用 precise
  并明确其数据语义。
- low/standard/high 保持单调，但不是统计置信区间。

原始拟合的统一模型 pooled MAPE `15.27%`、六折 LOSO 平均 `16.44%`，这些数字描述
warm-replay precise 参数，不是 BF16 v3 保守三档的新准确率。FP8 参数未修改，原指标仍只能
用于相同 collector/backend 边界。
