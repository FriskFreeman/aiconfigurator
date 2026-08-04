# FA2/FA3 通用参数模型工具终稿

> 版本补充（2026-07-28）：当前模型版本为 `2026-07-28.aic-fa-roofline-v3`。在不改变普通 FA 默认值的前提下，`AttentionShape` 增加 `value_head_dim` 与 `kv_storage_dim`，分别表达 PV 输出宽度和每个 KV token 的实际合计存储宽度，并增加显式 query-tile L2 reuse 端点开关。由此 DeepSeek MLA 的 prefill `192/128/320` 与 decode `576/512/576, Hkv=1` 可以由通用公式表示。H100/SGLang 0.5.9 precise 校准位于主工作区 `.self/compass-sim-mla_sglang059/`，不属于内嵌的跨硬件 standard/high/low profile。

## 1. 最终定位

本工具面向只有基础硬件数值规格、没有目标卡实测标定的数据场景。v2 的设计目标是接口含义直接、预测过程可解释、前端容易集成。

最终输入只有：

1. 参与公式计算的硬件数值；
2. 显式的 `fa2` 或 `fa3` 算法选择；
3. `standard/high/low` 三档经验参数选择。

模型不再输入或解析：

```text
name
vendor
architecture_family
backend_family
```

也不再提供自动算法识别、A100/H100 profile 路由、外推 confidence、身份告警或 engineering range。

默认行为：

```text
algorithm      = fa2
mode           = profiled
estimate_level = standard
```

---

## 2. 两种计算模式

### 2.1 Analytical

```text
mode = analytical
```

只使用算法工作量和硬件数值计算：

```text
t_resource = max(t_HBM, t_L2, t_compute)
t = t_resource
```

不加入固定启动开销、资源经验效率和 decode task-service。它是解析资源基线，不是严格数学下界，因为 tile、并行效率和 overlap 仍是模型抽象。

### 2.2 Profiled

```text
mode = profiled
```

使用通用经验参数：

```text
t_resource = max(
    t_HBM / eta_HBM,
    t_L2 / eta_L2,
    t_compute / eta_compute
)

t_task = N_task / SM_count * kv_task_cycles / clock
t = fixed_overhead + t_resource + t_task
```

Prefill 的 `N_task=0`；decode 在 `history>0 且 Lq<=decode_query_threshold` 时启用 task-service。

---

## 3. 三档通用参数

| 档位 | fixed overhead(us) | eta HBM | eta L2 | eta compute | KV task cycles |
| --- | ---: | ---: | ---: | ---: | ---: |
| `standard` | 12.5 | 0.72 | 0.80 | 0.55 | 6000 |
| `high` | 16.0 | 0.50 | 0.60 | 0.35 | 8000 |
| `low` | 10.0 | 0.92 | 0.95 | 0.80 | 3500 |

语义规定：

```text
high     = 高时延、偏保守预测
standard = 默认中部预测
low      = 低时延、偏乐观预测
```

固定硬件、shape 和算法时，参数结构保证：

```text
latency_low <= latency_standard <= latency_high
```

### 3.1 参数来源

三档借鉴 A100/H100 SGLang 0.5.9 的建模经验，但不保留硬件、架构、backend、phase 或 dtype 分组。已有中心范围大致为：

```text
fixed overhead: 11.2-14.1 us
eta HBM      : 0.54-0.97
eta L2       : 约0.78-0.80的可辨识中心
eta compute  : 0.37-0.78
KV task cycle: 3900、6125、7650
```

`standard` 取跨组中部的低精度工程值；`high/low` 在中心两侧留出更宽的性能场景。

三档不是某张卡的重新拟合结果、统计分位数、置信区间或性能上下界保证，而是生产前端可直接切换的统一假设。

### 3.2 为什么使用统一参数

硬件差异已经通过 SM、clock、HBM/L2 带宽、matrix/vector 峰值和 shared memory 进入解析层。经验参数只补充产品规格无法给出的平均损失：固定调度开销、持续资源效率和 decode 任务边界成本。

这避免在没有目标数据时，根据产品名或架构名伪造精确 profile。

---

## 4. 显式 FA2/FA3

输入方式：

```python
ModelOptions(algorithm="fa2")
ModelOptions(algorithm="fa3")
```

CLI：

```bash
--algorithm fa2
--algorithm fa3
```

默认 `fa2`。工具不会根据硬件名称、架构或 peak 自动选择。

统一计算形式：

```text
t_compute = t_matrix + t_vector_main
          - overlap_fraction * min(t_matrix, t_vector_main)
          + t_vector_boundary
```

默认：

```text
FA2 overlap_fraction = 0
FA3 overlap_fraction = 1
```

因此：

```text
FA2: t_matrix + t_vector_main + t_boundary
FA3: max(t_matrix, t_vector_main) + t_boundary
```

FA3 表达理想化的 matrix/softmax 主循环重叠，不模拟 WGMMA、TMA、warp specialization 或 pipeline stage。用户也可显式设置 `[0,1]` 的 `overlap_fraction`。

---

## 5. Shape 与 dtype

```text
B      = batch_size
Lq     = query_length
Lkv    = kv_length_total
Hq     = query_heads
Hkv    = kv_heads
d      = head_dim
```

`kv_length_total` 已包含历史和当前 query token：

```text
history_length = Lkv - Lq
```

要求 `Lkv >= Lq` 且 `Hq % Hkv == 0`。支持 `fp8/fp16/bf16/fp32`，FP8 默认输出 BF16。输入 dtype 决定 Q/K/V 字节和 matrix peak 选择。

---

## 6. Tile 模型

FA2/FA3 使用现代 query-outer 抽象：

```text
for each query tile Br:
    for each active KV tile Bc:
        S = matmul(Q, transpose(K))
        update online softmax
        O = matmul(P, V)
```

自动 tile：

```text
sram_words = shared_memory_per_sm_bytes / dtype_bytes
capacity   = floor(sram_words / (4*d))
auto_block = min(128, power_of_two_floor(capacity))
Br = Bc = auto_block
```

可通过 `br/bc` 显式覆盖。该公式没有模拟 register pressure、bank conflict 和编译器 tile search。

模型逐 query tile 计算 causal 有效 score、实际激活的 KV tiles、末尾 padding 和 online softmax 更新次数。

---

## 7. FLOPs 与 IO

设执行 score 元素数为 `E`：

```text
F_QK = 2 * E * d
F_PV = 2 * E * d
F_matrix = F_QK + F_PV
```

Vector 等价 FLOPs 包括 scale、max、subtract、exp、sum、online 状态更新、final normalize 和 split partial reduction。

HBM 主体字节：

```text
Q bytes      = splits * B * Hq * Lq * d * input_bytes
KV bytes     = 2 * B * effective_KV_heads * loaded_KV * d * input_bytes
Output bytes = B * Hq * Lq * d * output_bytes
```

默认使用理想 GQA HBM/L2 reuse，因此 KV 按 `Hkv` 计。可分别关闭复用假设。

默认还计入 collector 边界的当前 K/V source read 和 cache write：

```text
live_KV_update_bytes = 2 * (2 * B * Hkv * Lq * d) * input_bytes
```

---

## 8. Split-KV

```text
Tq = ceil(Lq / Br)
Tc = ceil(Lkv / Bc)
base_CTA = B * Hq * Tq
```

自动 split：

```text
splits_for_one_wave = ceil(SM_count / base_CTA)
max_by_work = max(1, floor(Tc / min_kv_tiles_per_split))
S = min(splits_for_one_wave, max_decode_splits, Tc, max_by_work)
```

默认只在短 query decode 启用，也可显式指定整数。`S>1` 时模型计算 FP32 partial output/LSE 的写入、读取和稳定归并。

该 split 值是解析启发式，不是实际 backend runtime 记录值。

---

## 9. Decode N_task

```text
N_task = B * Hkv * query_tiles * kv_splits
fractional_waves = N_task / SM_count
t_task = fractional_waves * kv_task_cycles / clock
```

`N_task` 是等效 KV-head-group 服务单元，不是实际 CUDA CTA 数。使用 `Hkv` 是因为 GQA 中多个 query heads 共享一个 KV head；主 matrix FLOPs 仍按 `Hq`。

不乘完整 `Tc`，因为 KV 长度对应的 FLOPs/bytes 已在资源 Roofline 中；task 项只补独立任务边界成本。

三档 task cycles 直接按目标硬件的 SM 数和 clock 缩放：

```text
tasks/s = SM_count * clock / kv_task_cycles
```

---

## 10. 硬件输入

硬件 schema v2 只包含：

| 字段 | 单位 | 作用 |
| --- | --- | --- |
| `sm_count` | 个 | 并行度和 task waves |
| `clock_hz` | Hz | task service |
| `shared_memory_per_sm_bytes` | byte | Br/Bc |
| `l2_capacity_bytes` | byte | 输出记录，v2不建容量命中模型 |
| `l2_bandwidth_bytes_s` | byte/s | L2 roof |
| `hbm_bandwidth_bytes_s` | byte/s | HBM roof |
| `matrix_peak_flops` | FLOP/s | dtype matrix roof |
| `vector_peak_flops` | FLOP/s | softmax/vector roof |
| `exp_flop_equivalent` | FLOP/exp | exp统一折算 |

产品来源、架构名和 backend 由上层系统维护，不进入公式接口。

---

## 11. 输出

主要字段：

```text
latency_us
algorithm
mode
bottleneck
reference_profile.level
```

分解位于 `tiles/work/resources_us/scheduling`。`profiled` 输出完整五参数；`analytical` 的 `reference_profile=null`。

工具不再输出自动 confidence、身份 warning 或单一 engineering range。前端展示区间时，应并列调用：

```text
optimistic   = low
central      = standard
conservative = high
```

---

## 12. 可迁移性与局限

具有较强可迁移性的部分包括 QK/PV FLOPs、causal 有效域、dtype bytes、GQA KV 存储、online softmax、split partial 工作和 `N_task` 粒度抽象。

仍与真实 kernel 强相关的部分包括 Br/Bc、CTA mapping、实际 split heuristic、GQA cache reuse、matrix/vector overlap、fixed overhead、eta 和 task cycles。

三档参数降低了对特定硬件身份的依赖，但不能消除 backend 差异。对新架构或国产 GPU，应由用户根据实际 kernel 语义选择 FA2/FA3，并同时查看三档结果，不应由工具猜测。

v2 不支持 mixed Q/KV dtype、sliding-window、sparse attention、MLA、FP4、跨卡 attention 和 backward。

---

## 13. 使用示例

默认 FA2 标准档：

```bash
python3 -m fa_roofline \
  --hardware configs/H100_SXM5_80GB.json \
  --input examples/shapes.json
```

FA3 高位估计：

```bash
python3 -m fa_roofline \
  --hardware configs/H100_SXM5_80GB.json \
  --input examples/shapes.json \
  --algorithm fa3 \
  --estimate-level high
```

解析基线：

```bash
python3 -m fa_roofline \
  --hardware configs/H100_SXM5_80GB.json \
  --input examples/shapes.json \
  --algorithm fa3 \
  --mode analytical
```

---

## 14. 验收结论

v2 测试覆盖默认 FA2/profiled/standard、显式 FA3、三档单调性、旧身份字段拒绝、FA2/FA3 历史解析快照、GQA reuse、split-KV、N_task、FP8 输出和 CLI 参数切换。
# 版本补充：MLA 非对称维度

当前模型版本为 `2026-07-28.aic-fa-roofline-v3`。在不改变普通 FA 默认值的前提下，
`AttentionShape` 增加了 `value_head_dim` 与 `kv_storage_dim`，分别表达 PV 输出宽度和
每个 KV token 的实际合计存储宽度；同时增加显式的 query-tile L2 reuse 端点开关。
这使 DeepSeek MLA 的 prefill `192/128/320` 与 decode `576/512/576, Hkv=1` 可以直接由
通用公式表示。H100/SGLang 0.5.9 的 precise 校准位于主工作区的
`.self/compass-sim-mla_sglang059/`，不属于本工具内嵌的跨硬件 standard/high/low profile。
