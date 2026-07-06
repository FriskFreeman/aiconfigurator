# AIC 原生仿真执行细节说明

本文记录一次轻量 AIC SDK 端到端仿真探针，目标是厘清：

- 使用 `backend=sglang`、`model=deepseek-ai/DeepSeek-V3` 时，SDK 实际使用的模型类和后端类。
- PD 分离语义下，分别通过 `run_static(..., mode="static_ctx")` 和 `run_static(..., mode="static_gen")` 跑 P/D。
- AIC 原生结果能给出哪些信息，哪些需要额外 probe，哪些目前拿不到。
- SILICON 数据表是否加载、op 查询是否成功、是否走 fallback、是否发生插值/直接命中等。

## 1. 运行脚本

脚本路径：

- [run_aic_origin_probe.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-aic-origin/run_aic_origin_probe.py)

脚本特点：

- 不修改 SDK 源码，只在运行时对当前进程里的 op instance 和 `PerfDatabase._interp_3d()` 做轻量包装。
- 分别执行：
  - prefill: `InferenceSession.run_static(..., mode="static_ctx")`
  - decode: `InferenceSession.run_static(..., mode="static_gen")`
  - native disagg: `DisaggInferenceSession.run_disagg(...)`
- 保留 AIC 原生 summary 产物，并额外导出 query/interpolation trace。

默认命令：

```bash
python .self/task-aic-origin/run_aic_origin_probe.py
```

常用参数：

```bash
python .self/task-aic-origin/run_aic_origin_probe.py \
  --system h100_sxm \
  --backend sglang \
  --version 0.5.9 \
  --model-path deepseek-ai/DeepSeek-V3 \
  --database-mode SILICON \
  --isl 1024 \
  --osl 16 \
  --prefix 0 \
  --prefill-batch-size 1 \
  --decode-batch-size 4 \
  --prefill-tp-size 8 \
  --decode-tp-size 8 \
  --prefill-moe-tp-size 1 \
  --decode-moe-tp-size 1 \
  --prefill-moe-ep-size 8 \
  --decode-moe-ep-size 8
```

## 2. 本次试运行结果

最终运行目录：

- [20260622_151748_aic_origin_sglang_deepseek_v3_pd_static](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/.self/task-aic-origin/runs/20260622_151748_aic_origin_sglang_deepseek_v3_pd_static)

本次配置：

- `model_path`: `deepseek-ai/DeepSeek-V3`
- `system`: `h100_sxm`
- `backend`: `sglang`
- `version`: `0.5.9`
- `database_mode`: `SILICON`
- `isl`: `1024`
- `osl`: `16`
- `prefix`: `0`
- prefill batch size: `1`
- decode batch size: `4`
- parallel: `tp8 pp1 dp1 moe_tp1 moe_ep8`

执行成功，`run_complete.json` 输出：

- prefill `ttft/context_latency`: `67.469 ms`
- decode `tpot`: `27.683 ms`
- decode `generation_latency`: `415.246 ms`
- prefill query events: `19`
- decode query events: `81`
- prefill `_interp_3d` events: `3`
- decode `_interp_3d` events: `7`

## 3. 输出文件说明

每个运行目录下包含：

- `args.json`
  - 本次脚本入参。
- `run_complete.json`
  - 脚本最终摘要。
- `prefill/`
  - `prefill_summary_df.csv/json`: AIC 原生 `summary.get_summary_df()`。
  - `prefill_static_info.txt`: AIC 原生 `summary.get_static_info()` 文本。
  - `prefill_native_summary.json`: AIC 原生 `InferenceSummary` getter 汇总。
  - `run_meta.json`: probe 汇总的模型类、后端类、ops tree、数据库加载状态。
  - `query_trace.json`: probe 包装每个 op `query()` 后得到的调用入参、返回、异常。
  - `interp_trace.json`: probe 包装 `PerfDatabase._interp_3d()` 后得到的查表坐标和启发式命中/插值分类。
- `decode/`
  - 与 `prefill/` 同结构。
- `disagg_native/`
  - 未包裹 op trace 的 `DisaggInferenceSession.run_disagg(...)` 原生结果。

## 4. SDK 实际使用的模型类和后端类

本次 `run_meta.json` 记录：

- prefill model class: `DeepSeekModel`
- decode model class: `DeepSeekModel`
- backend class: `SGLANGBackend`

源码入口：

- `models.get_model()` 位于 [models/__init__.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/models/__init__.py)
  - 读取 HF config / model info。
  - 根据 architecture 映射 model family。
  - 对 quant mode 应用默认值。
  - 通过 registry 调用对应模型类的 `create()`。
- `DeepSeekModel.create()` 位于 [deepseek.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/models/deepseek.py)
  - 对 `backend_name == "sglang"` 且 `moe_backend == "deepep_moe"` 时会走 `WideEPDeepSeekModel`。
  - 本次未启用 deepep/wideep，因此走普通 `DeepSeekModel`。
- `get_backend("sglang")` 位于 [factory.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/backends/factory.py)
  - 返回 `SGLANGBackend()`。

## 5. P/D run_static 与原生 disagg 过程

脚本显式执行：

- `InferenceSession(model, db, backend).run_static(runtime_config, mode="static_ctx")`
- `InferenceSession(model, db, backend).run_static(runtime_config, mode="static_gen")`
- `DisaggInferenceSession(...).run_disagg(...)`

源码入口：

- [inference_session.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/inference_session.py)
  - `InferenceSession.run_static()` 直接委托给 backend。
  - `DisaggInferenceSession.run_disagg()` 内部分别创建 prefill/decode model 和 `InferenceSession`，再调用 `static_ctx/static_gen`。
- [base_backend.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/backends/base_backend.py)
  - `run_static()` 调 `_run_static_breakdown()`。
  - `static_ctx` 只保留 context latency。
  - `static_gen` 只保留 generation latency。
  - `_run_context_phase()` / `_run_generation_phase()` 遍历对应 `model.context_ops` / `model.generation_ops` 并调用 op `query()`。

## 6. DeepSeek V3 的 ops 查询路径

本次普通 `DeepSeekModel` 的 op tree 中：

- context ops 数量：`13`
- generation ops 数量：`7`

Prefill query trace 统计：

- 总事件：`19`
- 成功：`18`
- 异常：`1`
- 异常 op：`context_mla_module`
- 异常类型：`PerfDataNotAvailableError`

Prefill 关键路径：

- `context_embedding`
- `context_add_norm_1`
- `context_mla_block`
  - primary: `context_mla_module`
  - 因 `mla_context_module_perf.txt` 缺失失败
  - fallback: GEMM + `ContextMLA` + GEMM
- MoE pre dispatch / MoE / post dispatch
- logits / p2p

Decode query trace 统计：

- 总事件：`81`
- 成功：`80`
- 异常：`1`
- 异常 op：`generation_mla_module`
- 异常类型：`PerfDataNotAvailableError`

Decode 关键路径：

- `generation_embedding`
- `generation_add_norm_1`
- `generation_mla_block`
  - primary: `generation_mla_module`
  - 因 `mla_generation_module_perf.txt` 缺失失败
  - fallback: GEMM + `MLABmm` + `GenerationMLA` + `MLABmm` + GEMM
- `generation_moe_overlap`
- logits / p2p

重要判断：

- H100/SGLang 0.5.9 当前没有 module-level MLA 表，因此 `FallbackOp` 确实回退到了更细粒度 ops。
- 对应证据在：
  - `prefill/query_trace.json`
  - `decode/query_trace.json`
  - `prefill/run_meta.json`
  - `decode/run_meta.json`

## 7. SILICON 数据表加载状态

本次 `PerfDatabase` 成功创建并设置为 `DatabaseMode.SILICON`。

`run_meta.json` 中 `database.loaded_op_data` 记录所有 `LoadedOpData`：

- 已加载表：`16 / 29`
- 未加载表包括：
  - `mla_context_module_perf.txt`
  - `mla_generation_module_perf.txt`
  - `mhc_module_perf.txt`
  - `deepseek_v4_context_module_perf.txt`
  - `deepseek_v4_generation_module_perf.txt`
  - `wideep_*`
  - 以及若干当前模型/后端不使用的表

这说明：

- SILICON 模式本身可用。
- 被本次 fallback 实际需要的 granular 表是可用的。
- module-level MLA 表缺失，primary module 查询失败是预期现象，随后由 `FallbackOp` 接管。

## 8. 直接命中、插值、外推信息

AIC 原生接口目前不直接暴露：

- 是否直接命中某一行。
- 使用了哪些邻点。
- 是否外推。
- 具体 CSV row id / row number。
- 插值过程中的中间权重。

源码依据：

- `PerfDatabase._interp_3d()` 只是调用 `interpolation.interp_3d(...)` 并返回 metrics dict。
- `interpolation.interp_3d(...)` 返回 `{latency, power, energy}`，不包含 provenance。
- `BaseBackend` 和 `InferenceSummary` 只保存 op 级 latency/energy/source。

为尽量补足信息，脚本额外包装了 `PerfDatabase._interp_3d()`，导出 `interp_trace.json`。该文件包含：

- `current_op`: 当前触发插值的 op 名、class、path。
- `requested`: 查询坐标 `x/y/z`。
- `method`: `cubic` / `bilinear` 等。
- `exact_hit`: 该三维点是否在当前 data dict 中精确存在。
- `classification_best_effort`:
  - `direct_hit`: 精确点存在。
  - `interpolation_or_sparse_grid`: 坐标落在全局轴范围内，但不是精确点。
  - `extrapolation_or_out_of_grid`: 坐标超出全局轴范围，或明显不在表格网格内。
- `global_axes`: 当前 data dict 全局轴范围。
- `local_axes`: 指定 `x` 或 `x/y` 下的局部轴范围。
- `result`: `_interp_3d()` 原始返回。

本次统计：

- prefill `_interp_3d` 事件：`3`
  - `direct_hit`: `1`
  - `interpolation_or_sparse_grid`: `2`
- decode `_interp_3d` 事件：`7`
  - `interpolation_or_sparse_grid`: `7`

注意：

- 这是 probe 的 best-effort 追踪，不是 AIC SDK 原生输出。
- 对稀疏网格而言，全局范围内不等于一定完成标准插值；因此分类名保守地写为 `interpolation_or_sparse_grid`。

## 9. AIC 原生结果能提供什么

通过 `InferenceSummary` 原生 getter 可以拿到：

- summary dataframe
  - `get_summary_df()`
- 文本摘要
  - `get_static_info()`
- memory breakdown
  - `get_memory()`
- context/generation latency dict
  - `get_context_latency_dict()`
  - `get_generation_latency_dict()`
- context/generation energy dict
  - `get_context_energy_wms_dict()`
  - `get_generation_energy_wms_dict()`
- context/generation source dict
  - `get_context_source_dict()`
  - `get_generation_source_dict()`
- OOM 状态
  - `check_oom()`
  - `check_kv_cache_oom()`
- 结果 dict
  - `get_result_dict()`

本次脚本将这些原生 getter 汇总到：

- `prefill/prefill_native_summary.json`
- `decode/decode_native_summary.json`
- `disagg_native/disagg_native_summary.json`

## 10. AIC 原生结果不能提供什么

当前原生 AIC 接口不能直接提供：

- op query 的完整 kwargs。
- `FallbackOp` primary 失败再 fallback 的完整调用链。
- module-level MLA 表缺失导致 fallback 的详细异常。
- 每一次 `_interp_3d()` 的坐标、命中类型、邻点、外推状态。
- 原始 perf CSV 行号。
- 插值邻点及权重。
- 数据表中 exact row 与 query point 的精确匹配关系。

这些信息需要通过本脚本这样的 probe 额外捕获。

## 11. 本次仿真的关键结论

1. `backend=sglang` + `deepseek-ai/DeepSeek-V3` 默认使用 `DeepSeekModel` 和 `SGLANGBackend`。
2. `run_static(static_ctx)` 和 `run_static(static_gen)` 都可以在 SILICON 模式下完成。
3. H100/SGLang 0.5.9 当前 granular perf 表足够支撑本次仿真。
4. module-level MLA 表缺失：
   - `mla_context_module_perf.txt`
   - `mla_generation_module_perf.txt`
5. 因 module 表缺失，`context_mla_block` / `generation_mla_block` 的 primary `MLAModule` 会失败，然后 `FallbackOp` 回退到 granular ops。
6. AIC 原生 summary 只能提供 op 级聚合结果和 coarse source，不能提供命中/插值/外推 provenance。
7. 本脚本新增的 `query_trace.json` 和 `interp_trace.json` 可以辅助分析，但属于 probe 产物，不是 SDK 原生 API。

