# MLA BF16 通用 Roofline 模型终稿

## 1. 定位与边界

MLA 生产模型作为 `fa_roofline` 的独立 namespace 归档，共享 FA2/FA3 的硬件 schema、tile、FLOPs、HBM/L2、split-KV 和 task-service 解析核心。它没有复制或分叉普通 FA 模型。

当前生产范围严格限定为：

```text
DeepSeek 默认 MLA 维度
Prefill BF16
Decode BF16
显式选择 FA2 或 FA3，默认 FA2
```

FP8 MLA 不兼容，原因见 `MLA_FP8不兼容与数据异常说明.md`。三档参数只用于目标硬件没有实测校准数据时的工程评估，不是硬件自动识别结果，也不是置信区间。

## 2. MLA 语义

Prefill 映射为：

```text
dqk = 192
dv = 128
kv_storage = 320
Hkv = local Hq
Lq = Lkv = ISL
不计 live KV cache update
```

Decode 映射为：

```text
dqk = 576
dv = 512
kv_storage = 576
Hkv = 1
Lq = 1
Lkv = total attended KV length
计入单份 latent KV cache update
```

因此：

```text
F_QK = 2 * score_elements * dqk
F_PV = 2 * score_elements * dv
```

decode 的 latent KV 每 token 只存 `512+64=576` 个 BF16 元素，不能按普通 K/V 两份流量计算。

## 3. 固定结构假设

三档参数共享相同模型结构，只有连续参数变化。

### Prefill

- Q-outer、KV-inner FA dataflow；
- query-tile 之间采用理想 KV L2 reuse 端点；
- HBM 只计一份 KV sequence，重复请求仍计入 L2 requested bytes；
- CTA 超过一波后使用 `ceil(CTA/SM)/(CTA/SM)` rounded-wave 修正；
- 不增加独立 task-service。

### Decode

- `Lq=1`，query-tile repeated/reuse 的 HBM bytes 等价；
- 使用 linear resource wave，不应用 rounded-wave；
- 增加等效 KV-head split task-service：

```text
N_task = B * Hkv * query_tiles * kv_splits
T_task = (N_task / SM_count) * task_cycles / clock_hz
```

MLA decode 中 `Hkv=1`。

## 4. 三档参数

### Prefill BF16

| 档位 | Fixed overhead | Resource efficiency | Wave |
| --- | ---: | ---: | --- |
| `low` | 10.0 us | 0.90 | rounded |
| `standard` | 12.5 us | 0.70 | rounded |
| `high` | 16.0 us | 0.50 | rounded |

### Decode BF16

| 档位 | Fixed overhead | Resource efficiency | Task cycles | Wave |
| --- | ---: | ---: | ---: | --- |
| `low` | 10.0 us | 0.95 | 4500 | linear |
| `standard` | 12.5 us | 0.85 | 7000 | linear |
| `high` | 16.0 us | 0.60 | 9000 | linear |

`low` 表示低时延、偏乐观，`high` 表示高时延、偏保守。对任意正资源工作量，参数构造保证：

```text
latency_low <= latency_standard <= latency_high
```

## 5. 最终公式

Prefill：

```text
T_raw = max(T_hbm_l2_reuse, T_l2, T_compute)
W = N_CTA / SM_count
R_wave = ceil(W)/W if W > 1 else 1

T_prefill = fixed_overhead + T_raw * R_wave / resource_efficiency
```

Decode：

```text
T_raw = max(T_hbm, T_l2, T_compute)

T_decode = fixed_overhead
         + T_raw / resource_efficiency
         + (N_task/SM_count) * task_cycles/clock_hz
```

FA2/FA3 由调用方显式选择。FA2 将 matrix 与主循环 vector 工作近似串行；FA3 默认采用理想重叠。参数档位不会根据 GPU 名称自动改变算法。

## 6. 参数来源与 H100 回放

标准档参考 H100/SGLang 0.5.9/FA3 precise 校准与普通 FA 通用档位后进行圆整，不直接携带 H100、厂商或 backend 路由。H100 precise 参数为：

```text
Prefill: floor=12.020 us, eta=0.6983
Decode : floor=12.672 us, eta=0.8568, task=7154.65 cycles
```

三档标准参数在同一 H100 BF16 物理样本上的非重新拟合回放：

| 配置 | 档位 | MAPE | MdAPE | P90 APE | Median ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| Prefill BF16 | low | 24.91% | 22.20% | 47.20% | 0.794 |
|  | standard | 21.48% | 15.84% | 51.55% | 1.016 |
|  | high | 46.18% | 42.35% | 86.15% | 1.397 |
| Decode BF16 | low | 18.76% | 17.60% | 35.38% | 0.826 |
|  | standard | 10.69% | 7.73% | 25.83% | 1.018 |
|  | high | 34.45% | 33.67% | 54.70% | 1.337 |

H100 实测落在 low/high 包络内的比例为 prefill `71.4%`、decode `88.6%`。这再次说明 high/low 是容量规划场景，不是统计覆盖保证。

## 7. API

```python
from fa_roofline import (
    HardwareSpec,
    MlaModelOptions,
    MlaRequest,
    estimate_mla,
)

hardware = HardwareSpec.from_json("configs/H100_SXM5_80GB.json")
request = MlaRequest(
    phase="decode",
    batch_size=2,
    local_query_heads=16,
    sequence_length=4096,
)
result = estimate_mla(
    hardware,
    request,
    MlaModelOptions(algorithm="fa3", estimate_level="standard"),
)
print(result.latency_us)
```

Prefix prefill 使用独立 `query_length`：

```python
request = MlaRequest(
    phase="prefill",
    batch_size=4,
    local_query_heads=32,
    sequence_length=8192,
    query_length=7168,
    dtype="bf16",
)
```

`sequence_length` 统一表示本轮 attention 的 KV total length。prefill 可选传入 `query_length`表示本轮 query token 数，默认与 `sequence_length` 相等；因此 `query_length < sequence_length` 可直接表达 prefix prefill。decode 始终使用 `query_length=1`。`local_query_heads` 是 TP 后单卡实际看到的 query head 数，模型不读取 `tp_size` 或自动换算全局 heads。

## 8. CLI

```bash
python3 -m fa_roofline.mla_cli \
  --hardware configs/H100_SXM5_80GB.json \
  --input examples/mla_shapes.json \
  --algorithm fa3 \
  --estimate-level standard
```

查看三档：

```bash
python3 -m fa_roofline.mla_cli --list-profiles
```

显式复现 H100 precise：

```bash
python3 -m fa_roofline.mla_cli \
  --hardware configs/H100_SXM5_80GB.json \
  --input examples/mla_shapes.json \
  --algorithm fa3 \
  --profile-file configs/reference/H100_SXM_SGLang059_MLA_BF16_precise.json
```

precise 文件不会被自动选择。

## 9. 跨硬件边界

可以直接迁移的部分：

- 非对称 QK/PV FLOPs；
- prefill 320、decode 576 的真实 KV storage；
- decode `Hkv=1`；
- rounded wave 与 task-service 的数值硬件缩放；
- 从 hardware JSON 获取各 BF16 peak、HBM/L2、SM、clock。

需要目标 backend 确认的部分：

- prefill 是否仍走 192/128 的 MHA 路径；
- decode 是否仍是 absorbed latent MQA；
- 实际 kernel 应选择 FA2 还是 FA3；
- split-KV、Br/Bc 和 query-tile L2 residency 是否与启发式一致。

参数本身外推能力最弱。新硬件获得少量实测后，应优先重新检查 floor、resource efficiency 和 task cycles，而不是修改 shape 查表系数。
