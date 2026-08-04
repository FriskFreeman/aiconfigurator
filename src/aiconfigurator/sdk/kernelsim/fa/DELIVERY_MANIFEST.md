# FA 模型工具交付清单

## 最终运行产物

| 产物 | 作用 |
| --- | --- |
| `fa_roofline/model.py` | FA2/FA3 工作量、资源和 decode service 模型 |
| `fa_roofline/schema.py` | hardware/shape/options 严格接口 |
| `fa_roofline/profiles.py` | standard/high/low 三档通用参数 |
| `fa_roofline/cli.py` | 批量 JSON CLI |
| `fa_roofline_final.py` | 便捷启动脚本 |
| `fa_roofline/mla_schema.py` | BF16 MLA request/options 严格接口、prefix prefill 的独立 Lq/Lkv 与 FP8 fail-closed |
| `fa_roofline/mla_model.py` | MLA phase adapter、rounded-wave 与 task-service |
| `fa_roofline/mla_profiles.py` | MLA BF16 standard/high/low 三档参数 |
| `fa_roofline/mla_cli.py` | MLA BF16 批量 JSON CLI |
| `mla_roofline_final.py` | MLA CLI 便捷启动脚本 |
| `pyproject.toml` | 无依赖 package 和 console entrypoint |

## 配置与样例

| 产物 | 作用 |
| --- | --- |
| `configs/hardware_schema.json` | JSON Schema draft 2020-12 |
| `configs/shape_schema.json` | Attention shape JSON Schema |
| `configs/A100_SXM4_80GB.json` | A100 数值硬件配置，不含算法路由 |
| `configs/H100_SXM5_80GB.json` | H100 数值硬件配置，不含算法路由 |
| `configs/hardware_template.json` | 新硬件字段模板，数值仅供格式示例 |
| `examples/shapes.json` | Prefill、GQA decode、FP8短KV样例 |
| `configs/mla_shape_schema.json` | 仅接受 BF16 的 MLA request schema |
| `configs/reference/H100_SXM_SGLang059_MLA_BF16_precise.json` | 显式 H100 BF16 precise 参考，不自动路由 |
| `examples/mla_shapes.json` | MLA prefill/decode BF16 样例 |

## 终稿文档

| 产物 | 作用 |
| --- | --- |
| `README.md` | 快速使用、API 和目录边界 |
| `docs/FA2_FA3零标定模型工具终稿.md` | 公式、profile、证据、局限与输出解释 |
| `docs/硬件配置与跨架构迁移指南.md` | 数值硬件配置和生产使用规范 |
| `docs/MLA_BF16通用模型终稿.md` | MLA 公式、三档参数、API、CLI 与迁移边界 |
| `docs/MLA_FP8不兼容与数据异常说明.md` | FP8 性能异常、排除依据和恢复支持条件 |
| `development/README.md` | 历史实验索引；不参与运行 |

MLA 非对称维度与 query-tile L2 reuse 的通用扩展包含在共享 FA v3 core 中；BF16
生产 adapter 和三档 profile 已归档。完整 H100/SGLang 0.5.9 实验位于
`.self/compass-sim-mla_sglang059/`。FP8 明确不兼容，不进入生产参数。

## 验收范围

`tests/` 覆盖：

- hardware/shape/dtype 校验；
- FP8输入/BF16输出；
- split-KV 与 `N_task`；
- 大 batch短 KV task-service；
- GQA HBM/L2 reuse；
- FA2/FA3 overlap；
- 默认 FA2 和显式 FA3；
- low/standard/high 三档单调性；
- 旧硬件身份字段严格拒绝；
- CLI 多 shape、多模式、输出文件和错误路径；
- 与 LLMCompass 中性语义的 FA2/FA3 固定数值回归。
- MLA prefill/prefix prefill/decode 几何、latent KV、rounded-wave 与 `N_task`；
- MLA low/standard/high 在 A100/H100、FA2/FA3 和多形状下单调；
- H100 BF16 precise 快照；
- FP8 API/CLI fail-closed 错误路径；
- 普通 FA 历史 17 项行为保持不变。

验收命令：

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile fa_roofline/*.py fa_roofline_final.py mla_roofline_final.py
python3 -m fa_roofline \
  --hardware configs/H100_SXM5_80GB.json \
  --input examples/shapes.json \
  --mode both \
  --output results/example_predictions.json

python3 -m fa_roofline.mla_cli \
  --hardware configs/H100_SXM5_80GB.json \
  --input examples/mla_shapes.json \
  --algorithm fa3 \
  --estimate-level standard \
  --output results/example_mla_predictions.json
```
