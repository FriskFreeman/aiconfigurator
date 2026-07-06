# Decode Single-Op Kernel AIC Error Analysis

## 结论摘要

- DeepSeek fallback 单算子累加相对实机 MLA kernel_sum 的 MAPE 为 `15.88%`，相对 makespan 的 MAPE 为 `14.75%`。
- 仅把 `generation_proj_gemm` 从 SDK 当前 `7168 x 7168/tp` 诊断性改为实机 `7168 x 16384/tp` 后，case 级带符号总和 MAPE 从 `15.88%` 变为 `25.69%`；它变大是因为原先 o_proj 低估抵消了 attention 高估。
- 按每个 op 的绝对误差加总看，修正 o_proj 后平均 L1 error 从 `0.071577ms` 降到 `0.047907ms`，说明该 shape 问题确实是单算子误差的重要来源。
- 若同时把 attention 查询从 SDK 当前 `s=kv_len+1` 诊断性改成 `s=kv_len`，case 级 kernel_sum MAPE 为 `25.68%`；这不是建议直接改 SDK，只用于分离插值/坐标误差。
- 误差主要不是 CUDA Graph module 边界问题；WideEP module 已与 makespan 高度吻合，单算子误差集中在 fallback op 的采集/查表口径。
- GEMM 对照已按 collect_gemm 的 `fp8_block` 口径包含前置 `per_token_group_quant_8bit_kernel + deepgemm`，否则会低估实机 GEMM 对齐口径。
- BMM 对照按可见 `nvjet_tst_*` kernel 进行；collector 的 fp8 BMM 口径包含 `per_tensor_quant_mla_fp8 + bmm_fp8`，而当前默认 FA3 实机 trace 中 MLA body 内可见的是 torch/nvjet BMM。
- `generation_attention` 的 AIC 查询来自 `query_generation_mla(b, s, num_heads, fp8)`，对实机侧主要对应 FA3 attention kernels；rotary 与 KV write 是 SGLang wrapper 支持 kernel，未计入主 attention 单算子口径。
- 根因排序：`generation_proj_gemm` 是明确的 SDK 普通 DeepSeekModel shape 口径错误；`generation_attention` 还叠加了 `s=kv_len+1` 插值坐标与 collect_mla/RadixAttention 计时口径差异。

## 最大误差来源

- `attention` 是 `6` 个 case 的最大绝对误差来源。
- `o_proj GEMM` 是 `3` 个 case 的最大绝对误差来源。

## Op 级均值

| op | real mean ms | AIC mean ms | AIC-real mean ms | MAPE | diagnostic AIC mean ms | diagnostic MAPE |
|---|---:|---:|---:|---:|---:|---:|
| downscale GEMM | 0.014126 | 0.015756 | +0.001630 | 11.63% | 0.015756 | 11.63% |
| q_b GEMM | 0.018702 | 0.019708 | +0.001006 | 5.72% | 0.019708 | 5.72% |
| q_w_kc BMM | 0.008274 | 0.010499 | +0.002225 | 26.50% | 0.010499 | 26.50% |
| attention | 0.033921 | 0.072962 | +0.039040 | 103.22% | 0.072962 | 103.22% |
| s_w_vc BMM | 0.008764 | 0.010474 | +0.001710 | 18.82% | 0.010474 | 18.82% |
| o_proj GEMM | 0.050408 | 0.024502 | -0.025906 | 51.34% | 0.049970 | 4.44% |

## 关键诊断

- `generation_proj_gemm` 当前 SDK 查询 `n=7168, k=7168/tp`，而实机 CUDA Graph 边界中的最终 o_proj DeepGEMM 是 `deepgemm(7168,16384)`；诊断性改成 `k=16384/tp` 后，该 op 平均 AIC 从 `0.024502ms` 变为 `0.049970ms`，MAPE 从 `51.34%` 降到 `4.44%`。
- `generation_attention` 当前 `s=kv_len+1` 查询在 `0` 个原始 case 精确命中；诊断性 `s=kv_len` 在 `9` 个 case 精确命中，平均改变 `-0.000019ms`，attention op MAPE 变为 `103.17%`。
- `attention_s_eq_kv_len` 诊断不能单独证明 SDK 应改成 `kv_len`：decode 语义里当前 token 也参与 query，`s=kv_len+1` 有语义合理性；它主要说明 generation_mla_perf 的采集网格与 decode case 坐标错开时会放大插值误差。
- 默认 FA3 实机 trace 中两段 `nvjet_tst_*` 已按 SGLang `forward_mla.py` 的 `torch.bmm` 语义归入 BMM；这修正了旧数据中将部分 `nvjet_tst_*` 错标成 flashmla attention 的问题。

## Collector 口径证据

- `collector/sglang/collect_gemm.py` 的 `fp8_block` kernel_func 先调用 `sglang_per_token_group_quant_fp8()`，再调用 `fp8_gemm_deepgemm()`；因此实机 GEMM 对照必须包含 `quant + deepgemm`。
- `collector/sglang/collect_mla.py` 用 `RadixAttention` 构造 `mla_generation`，decode 采集记录 `isl=1, step=input_len`，PerfDatabase 加载后按 `s=isl+step` 查询。
- `collector/sglang/collect_mla_bmm.py` 的 fp8 pre/post BMM 会把 `per_tensor_quant_mla_fp8()` 与 `bmm_fp8()` 放进同一次 `benchmark_with_power()`；默认 FA3 实机路径当前可见的是 torch/nvjet BMM，没有显式的 fp8 BMM quant kernel。

## AIC 查询链路

- `generation_downscale_gemm`: `ops.GEMM(..., n=2112, k=7168, fp8_block)` -> `PerfDatabase.query_gemm(m=batch, n=2112, k=7168, fp8_block)` -> `gemm_perf.txt`。
- `generation_q_b_proj_gemm`: `ops.GEMM(..., n=24576/tp, k=1536, fp8_block)` -> `query_gemm()` -> `gemm_perf.txt`。
- `generation_bmm_pre`: `ops.MLABmm(if_pre=True, fp8)` -> `query_mla_bmm(num_tokens=batch, num_heads=128/tp, fp8, pre)` -> `mla_bmm_perf.txt`。
- `generation_attention`: `ops.GenerationMLA(kvcache=fp8)` -> `query_generation_mla(b=batch, s=kv_len+1, num_heads=128/tp, fp8)` -> `generation_mla_perf.txt`。
- `generation_bmm_post`: `ops.MLABmm(if_pre=False, fp8)` -> `query_mla_bmm(..., post)` -> `mla_bmm_perf.txt`。
- `generation_proj_gemm`: `ops.GEMM(..., n=7168, k=7168/tp, fp8_block)` -> `query_gemm()` -> `gemm_perf.txt`。

## 插值与外推

- `query_gemm()` 优先精确命中 `(m,n,k)`；若同一 `n,k` 下有多个 `m` 点，则沿 `m` 做 1D 插值/外推；否则进入 3D cubic 插值。
- `query_mla_bmm()` 在固定 `num_heads/op/type` 后按 `num_tokens` 做 1D 插值/外推。
- `query_generation_mla()` 在 `(num_heads,b,s)` 上使用 3D bilinear 插值；本分析的 `s=kv_len+1`，因此很多点虽然接近采集网格，但不一定精确命中。

## 输出文件

- `decode_single_op_kernel_vs_aic.csv`: 每个 case/op 的实机 kernel 映射、AIC 查询值、误差和查询追踪。
- `decode_single_op_query_trace.csv`: AIC 查表函数、输入坐标、数据文件、命中/插值状态；包含 `__diagnostic_correct_shape` 与 `__diagnostic_s_eq_kv_len` 追踪行。
- `decode_single_op_case_error_summary.csv`: case 级总误差、修正 o_proj 诊断总误差、组合诊断总误差与最大误差来源。
- `decode_single_op_error_contribution.png/svg`: 总误差按 op 分解图。
- `decode_real_vs_aic_per_single_op.png/svg`: 每个 op 的实机 vs AIC 对比小图。
