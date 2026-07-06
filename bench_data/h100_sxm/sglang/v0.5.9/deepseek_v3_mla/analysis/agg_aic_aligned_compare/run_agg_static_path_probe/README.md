# AIC run_agg / run_static 路径探针结果

## 探针范围

本目录通过 `probe_run_agg_static_paths.py` 实际运行了 SGLang 后端 `run_agg`，并额外复现了 `run_agg` 内部的几次 `run_static` 调用，目标是确认 AGG 模式下 MLA 相关查询到底走：

- 普通 `DeepSeekModel` 的 `FallbackOp(context_mla_block / generation_mla_block)`
- 还是 `WideEPDeepSeekModel` 的 module 级 `WideEPContextMLA / WideEPGenerationMLA`

核心输出：

- `probe_summary.csv`：各 case 的模型类、run_agg 状态、per_ops keys、FallbackOp 状态摘要。
- `normal_sglang_deepseek_tp1.json` 等：逐 case 的完整 JSON，包括 `context_ops_before/after`、`generation_ops_before/after`、`per_ops_data`、复现的 `run_static` summary。
- `wideep_direct_mla_op_probe.json`：绕过 MoE，仅直接查询 WideEP MLA op 的验证结果。

## 结论

1. SGLang DeepSeek-V3 是否进入 WideEP 分支，不由 `enable_wideep=True` 单独决定，而由 `moe_backend == "deepep_moe"` 决定。
2. 普通 SGLang DeepSeek 跑 `run_agg` 时，模型类是 `DeepSeekModel`，`run_static` summary 暴露的字段名是 `context_mla_block` / `generation_mla_block`。
3. 普通 DeepSeek 的 `context_mla_block` / `generation_mla_block` 是 `FallbackOp` 外层字段；本次 H100/SGLang/0.5.9 数据下 primary `MLAModule` 不可用，实际查询落到了 fallback 子算子求和。
4. 只设置 `enable_wideep=True` 但不设置 `moe_backend="deepep_moe"` 时，仍然是 `DeepSeekModel`，不是 `WideEPDeepSeekModel`。
5. 设置 `moe_backend="deepep_moe"` 后会构造 `WideEPDeepSeekModel`；该模型没有普通 `FallbackOp(context_mla_block)`，而是直接把 `context_attention` 映射到 `WideEPContextMLA`，把 `generation_attention` 映射到 `WideEPGenerationMLA`。
6. WideEP 的完整 `run_agg` 目前在 SGLang 0.5.9 / H100 数据下跑不通：`fmha=bfloat16` 会先卡在 `wideep_context_mla` dtype/表结构不匹配；改成 `fmha=fp8_block` 后，MLA 能查，但会继续卡在缺失 `wideep_deepep_normal_perf.txt`，即 WideEP/DeepEP MoE normal 组件数据缺失。
7. 直接查询 WideEP MLA op 已验证成功：`WideEPContextMLA(context_attention)` 和 `WideEPGenerationMLA(generation_attention)` 在 `fmha=fp8_block, tp=1, attention_backend=fa3` 下均命中 SILICON wideep MLA 表。

## 关键 case 结果

| case | 状态 | 模型类 | 关键路径 |
|---|---:|---|---|
| `normal_sglang_deepseek_tp1` | ok | `DeepSeekModel` | `context_mla_block` / `generation_mla_block` 外层是 `FallbackOp`，primary `MLAModule` unavailable，实际走 fallback 子算子 |
| `enable_wideep_flag_only_tp1` | ok | `DeepSeekModel` | 与普通 case 相同；`enable_wideep=True` 对 SGLang DeepSeek-V3 分支选择无效 |
| `normal_sglang_deepseek_tp2_shape` | ok | `DeepSeekModel` | 与普通 case 相同，但子算子形状按 TP 切分 |
| `wideep_deepep_moe_tp1_adp16_ep16` | error | `WideEPDeepSeekModel` | 进入 WideEP；`fmha=bfloat16` 下 `wideep_context_mla` 查表失败 |
| `wideep_deepep_moe_tp1_grid_b1_s512_silicon_fp8block` | error | `WideEPDeepSeekModel` | WideEP MLA dtype 已对齐，但完整 run_agg 后续缺 `wideep_deepep_normal_perf.txt` |
| `wideep_deepep_moe_tp1_grid_b1_s512_sol_fp8block` | error | `WideEPDeepSeekModel` | SOL 也不能完整跑，因为 WideEP deepep normal op 未实现 SOL |

## 普通 DeepSeek run_agg 背后的 run_static

以 `normal_sglang_deepseek_tp1` 为例，`run_agg` 内部复现出的 `run_static` 调用为：

- `mix_non_attention_static_ctx`：`batch_size=1, isl=2052, prefix=0`，summary keys 包含 `context_mla_block`，但不展开其内部 fallback 子 op。
- `context_attention_static_ctx`：`batch_size=4, isl=512, prefix=0`，summary keys 同样包含 `context_mla_block`，不存在单独 `context_attention` 字段。
- `mix_generation_attention_static_gen`：`batch_size=4, isl=520, osl=2`，summary keys 包含 `generation_mla_block`。
- `genonly_static_gen`：`batch_size=8, isl=520, osl=2`，summary keys 包含 `generation_mla_block`。

因此在普通 DeepSeek 路径下，`run_agg` 中硬编码读取 `context_attention` / `generation_attention` 的逻辑与当前模型 summary 字段存在语义错位：真实 MLA 成本主要在 `context_mla_block` / `generation_mla_block`，而不是独立 attention 字段。

## 代码证据

- `src/aiconfigurator/sdk/models/deepseek.py:50`：SGLang DeepSeek 只有 `moe_backend == "deepep_moe"` 才返回 `WideEPDeepSeekModel`。
- `src/aiconfigurator/sdk/models/deepseek.py:138`：普通 `DeepSeekModel` 的 context MLA 是 `FallbackOp("context_mla_block", primary=MLAModule(...), fallback=[...])`。
- `src/aiconfigurator/sdk/models/deepseek.py:1078`：WideEP context MLA 是 `ops.WideEPContextMLA("context_attention", ...)`。
- `src/aiconfigurator/sdk/models/deepseek.py:1209`：WideEP generation MLA 是 `ops.WideEPGenerationMLA("generation_attention", ...)`。
- `src/aiconfigurator/sdk/backends/sglang_backend.py:126`、`:159`、`:182`：`run_agg` 通过三次 `run_static` 重构 mixed step。
- `src/aiconfigurator/sdk/backends/sglang_backend.py:148`、`:176`、`:196`：`run_agg` 硬编码按 `context_attention` / `generation_attention` 抽取特殊 attention 项。
- `src/aiconfigurator/sdk/operations.py:1995`：`FallbackOp` 先尝试 primary，并在 primary 不可用时求和 fallback 子 op。

## 对后续 MLA-only breakdown 的影响

普通 SGLang DeepSeek 的 AIC `run_agg` 结果不能只看字段名 `context_mla_block` 并误认为命中了 module 级 MLA 表；本次探针显示它实际是 fallback 子算子的合计。WideEP 路径恰好相反：字段名仍可能叫 `context_attention` / `generation_attention`，但 op class 是 module 级 WideEP MLA。后续做 MLA-only 对比时，应显式记录 `model_class`、`op_class`、`FallbackOp.primary_unavailable` 和数据源，而不是仅靠字段名分类。
