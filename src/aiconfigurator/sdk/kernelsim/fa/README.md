# AIC FA2/FA3 与 MLA BF16 独立 Roofline 模型工具

本目录是 FA 建模工作的最终生产工具。运行时仅依赖 Python 标准库，输入由两部分组成：

- 直接参与计算的硬件数值规格；
- 显式指定的 `fa2` 或 `fa3` 算法模型。

工具不再读取 GPU 名称、厂商、架构族或 backend 字符串，也不根据身份字段自动推断算法和经验参数。

当前版本：

```text
model  : 2026-07-28.aic-fa-roofline-v3
profile: 2026-07-24.aic-fa-generic-profile-v2
schema : 2
MLA model   : 2026-07-29.aic-mla-bf16-roofline-v2
MLA profile : 2026-07-29.aic-mla-bf16-generic-profile-v1
MLA schema  : 2
```

## 默认行为

未传入 `ModelOptions` 时使用：

```text
algorithm      = fa2
mode           = profiled
estimate_level = standard
```

三档 profiled 参数为：

| 档位 | 含义 |
| --- | --- |
| `standard` | 默认的中部工程估计 |
| `high` | 高时延、偏保守估计 |
| `low` | 低时延、偏乐观估计 |

三档不是统计置信区间。对同一硬件和 shape，参数设计保证 `low <= standard <= high`。

MLA 复用同一 FA2/FA3 解析核心，但使用独立的 request、profile 和 CLI。当前 MLA 生产接口仅支持 BF16；FP8 数据存在 backend 性能异常，工具会明确拒绝而不是返回估值。

## MLA BF16 快速使用

```bash
python3 -m fa_roofline.mla_cli \
  --hardware configs/H100_SXM5_80GB.json \
  --input examples/mla_shapes.json \
  --algorithm fa3 \
  --estimate-level standard
```

MLA 默认算法仍是 FA2；只有确认 backend 使用 FA3 时才显式传入 `--algorithm fa3`。三档参数只包含 prefill/decode BF16。H100/SGLang 0.5.9 precise 通过 `--profile-file configs/reference/H100_SXM_SGLang059_MLA_BF16_precise.json` 显式加载，不做硬件名称路由。

MLA request 中 `sequence_length` 表示 KV total length。prefill 可选传入 `query_length`；不传时等于 `sequence_length`，传入更小的值时表示 prefix prefill。例如 `query_length=7168, sequence_length=8192` 表示 1024-token prefix 与 7168-token 新 query。

## 快速使用

默认 FA2 和标准档：

```bash
python3 -m fa_roofline \
  --hardware configs/H100_SXM5_80GB.json \
  --input examples/shapes.json \
  --output results/standard_fa2.json
```

显式使用 FA3 和高时延档：

```bash
python3 -m fa_roofline \
  --hardware configs/H100_SXM5_80GB.json \
  --input examples/shapes.json \
  --algorithm fa3 \
  --estimate-level high \
  --output results/high_fa3.json
```

同时输出解析基线和标准 profiled 结果：

```bash
python3 -m fa_roofline \
  --hardware configs/H100_SXM5_80GB.json \
  --input examples/shapes.json \
  --mode both
```

查看三档内嵌参数：

```bash
python3 -m fa_roofline --list-profiles
```

## Python API

```python
from fa_roofline import AttentionShape, HardwareSpec, ModelOptions, estimate_attention

hardware = HardwareSpec.from_json("configs/H100_SXM5_80GB.json")
shape = AttentionShape(
    batch_size=1,
    query_length=1,
    kv_length_total=4096,
    query_heads=32,
    kv_heads=8,
    head_dim=128,
    dtype="bf16",
)

standard = estimate_attention(hardware, shape)
high_fa3 = estimate_attention(
    hardware,
    shape,
    ModelOptions(algorithm="fa3", estimate_level="high"),
)
```

`kv_length_total` 已包含历史和本轮 query token。FP8 默认输出 BF16。

对于 MLA 等非对称 attention，`AttentionShape` 可显式设置 `value_head_dim` 和
`kv_storage_dim`。例如 DeepSeek decode 使用 `head_dim=576`、
`value_head_dim=512`、`kv_storage_dim=576`、`kv_heads=1`；prefill 使用
`192/128/320` 且 `kv_heads=query_heads`。普通 FA 不传这两个可选字段时仍保持
对称默认值。query-tile KV 的理想 L2 驻留可通过
`ModelOptions(assume_query_tile_l2_reuse=True)` 或 CLI 的
`--assume-query-tile-l2-reuse` 显式打开；这只是固定端点假设，不是 cache simulator。

## 硬件配置

硬件 JSON 只保留参与公式的字段：

```text
sm_count
clock_hz
shared_memory_per_sm_bytes
l2_capacity_bytes
l2_bandwidth_bytes_s
hbm_bandwidth_bytes_s
matrix_peak_flops
vector_peak_flops
exp_flop_equivalent
```

产品名称、厂商、架构族、backend、来源和可信度应由上层资产管理系统维护，不进入数值模型接口。

## 输出

结果包含：

- 最终时延和瓶颈；
- Br/Bc、query tiles、KV tiles 和 split-KV；
- matrix/vector FLOPs；
- HBM/L2 requested bytes；
- raw 与 profile-adjusted 资源时延；
- fixed overhead、`N_task` 和 task-service；
- 当前显式算法和参数档位。

不再输出自动外推可信度、硬件身份告警或 engineering range。需要区间时，应分别运行 `low/standard/high`。

## 目录边界

```text
fa_roofline/                  最终模型 API 和 CLI
configs/                      数值硬件和 shape schema、示例配置
examples/                     输入 shape 样例
tests/                        API、CLI、单调性和历史快照测试
docs/                         终稿原理与使用指南
development/                  历史实验索引，不参与运行
fa_roofline_final.py          CLI 便捷入口
mla_roofline_final.py         MLA BF16 CLI 便捷入口
```

主要文档：

- [FA2_FA3零标定模型工具终稿](docs/FA2_FA3零标定模型工具终稿.md)
- [硬件配置与跨架构迁移指南](docs/硬件配置与跨架构迁移指南.md)
- [MLA BF16通用模型终稿](docs/MLA_BF16通用模型终稿.md)
- [MLA FP8不兼容与数据异常说明](docs/MLA_FP8不兼容与数据异常说明.md)
- [交付清单](DELIVERY_MANIFEST.md)

## 验收

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile fa_roofline/*.py fa_roofline_final.py mla_roofline_final.py
```
