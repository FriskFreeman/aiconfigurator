# MLA 算子级 Real vs AIC run_agg Breakdown

## 口径

- 实机侧：读取 `MLA时延拆解.csv` 的 `child_timing_json`，使用非首层平均 `gpu_makespan_ms` 作为主时延；同时在 CSV 中保留 `gpu_kernel_time_sum_ms` 和 top kernel。
- AIC 侧：保留既有 `run_agg` 快照中的 `context_mla_block` block 值，同时基于当前 SDK 的 `PrefixConditionalOp(context_mla_block)` 子结构，用同一批 `observed_first_batch` 形状诊断性重放细粒度 prefix-path 子 op 查询。
- 单层比较：AIC 子 op latency 除以 `num_layers=6`；实机取非首层均值，避免首层冷启动。
- 当前图只比较 mixed step 中的 context MLA 相关算子；不混入 MoE/FFN/embedding/logits，也不混入 genonly step。
- 注意：若 `run_agg` 快照实际只暴露 `context_mla_block`，则 q_b/kv_b/attention/o_proj 等 AIC 细粒度柱是 `diagnostic_forced_prefix_ops*` 口径，用于定位误差来源，不等价于原始快照直接导出的 per-op 字段。

## 输出

- `mla_operator_compare_long.csv`：每个 case x operator_family 的 Real/AIC 对比长表。
- `aic_mla_operator_ops_long.csv`：AIC 侧 block 快照和细粒度诊断查询长表。
- `aic_mla_operator_context_ops.csv`：AIC 侧用于画图的 context MLA 细粒度聚合表。
- `mla_operator_compare_by_operator_summary.csv`：按算子聚合的误差摘要。
- `mla_operator_compare_total_rollup.csv`：MLA total roll-up。
- `case_label_map.csv`：图中 `C01...C12` 与原始 case 的映射。
- `mla_operator_facets_real_vs_aic.png/svg`：每个算子一个子图的 Real vs AIC 对比。
- `mla_operator_ratio_heatmap.png/svg`：Real/AIC ratio 热力图。
- `per_case_operator_charts/`：每个用例一张算子级对比图。

## 最大正向误差项

- `C06` `kv_b_proj`: real 2.926 ms vs AIC 0.310 ms, ratio 9.43x
- `C12` `kv_b_proj`: real 1.885 ms vs AIC 0.310 ms, ratio 6.07x
- `C06` `attention`: real 5.239 ms vs AIC 3.732 ms, ratio 1.40x
- `C07` `kv_b_proj`: real 1.332 ms vs AIC 0.310 ms, ratio 4.29x
- `C05` `kv_b_proj`: real 1.261 ms vs AIC 0.310 ms, ratio 4.06x
- `C09` `kv_b_proj`: real 1.207 ms vs AIC 0.310 ms, ratio 3.89x
- `C11` `kv_b_proj`: real 1.152 ms vs AIC 0.337 ms, ratio 3.42x
- `C03` `attention`: real 0.937 ms vs AIC 0.376 ms, ratio 2.49x
- `C09` `attention`: real 2.583 ms vs AIC 2.120 ms, ratio 1.22x
- `C08` `kv_b_proj`: real 0.716 ms vs AIC 0.310 ms, ratio 2.31x

## 算子边界说明

- `qkv_a_proj`: Real fused_qkv_a_proj_with_mqa includes q_a/kv_a projection path and fp8 activation quant kernel. AIC context_downscale_gemm is a standalone SDK GEMM op before context_mla_block; collector fp8 GEMM uses pre-quantized inputs and usually excludes runtime activation quantization.
- `q_a_layernorm`: Real has an explicit q_a RMSNorm child range; ordinary DeepSeekModel MLA path does not model it as a separate SDK op.
- `kv_a_layernorm`: Real has an explicit kv_a RMSNorm child range; ordinary DeepSeekModel MLA path does not model it as a separate SDK op.
- `q_b_proj`: Both sides represent q_b projection, but real fp8 path includes activation quantization plus DeepGEMM kernels.
- `kv_b_proj`: Present as a separate real child mainly for attn_mha/prefill cases; MQA cases often fold or omit this child.
- `mla_concat_k`: AIC prefix/granular path models SGLang MLA K concat as a separate op; real trace usually exposes concat_mla_k_kernel inside surrounding child ranges.
- `rotary_emb`: Real has an explicit RoPE child range; AIC context MLA path does not expose a separate RoPE SDK op.
- `attention`: Real attention may be attn_mha or attn_mqa depending on SGLang scheduling/backend choice. AIC context_attention is queried with the second-pass run_agg context-attention shape and scale correction.
- `o_proj`: Both sides represent output projection, but real fp8 path includes activation quantization plus DeepGEMM kernels.

## 源码与 collector 证据

- `src/aiconfigurator/sdk/models/deepseek.py` 中普通 DeepSeek context 路径将 `context_downscale_gemm` 放在 `context_mla_block` 之前，注释说明 qkv/downscale 不在 SGLang MLA module collector boundary 内，而是 shared op 单独计一次。
- `src/aiconfigurator/sdk/operations.py` 的 `PrefixConditionalOp` 按 `prefix` 选择路径：`prefix == 0` 走 `_no_prefix_ops`，即 `context_mla_module`；`prefix > 0` 走 `_prefix_ops`，即 `context_q_b_proj_gemm/context_kv_b_proj_gemm/context_mla_concat_k/context_attention/context_proj_gemm`。
- 本批 `real_aic_joined.csv` 的 `aic_prefix` 均为 0，因此既有 `run_agg` 快照主要暴露 `context_mla_block` block 值；本报告的细粒度 AIC 子算子柱是按当前 SDK prefix-path 强制拆解出的 diagnostic attribution，用来定位误差来源，而不是声称原始快照直接导出了这些字段。
- `src/aiconfigurator/sdk/operations.py` 的 `ContextMLA.query()` 仅调用 `database.query_context_mla(...)`，源码注释写明 “Context MLA operation. now only contains MHA part.”；因此图中的 AIC `attention` 对应 context MHA/FA 类 attention 表，而不是整个 MLA 模块。
- `src/aiconfigurator/sdk/operations.py` 的 `MLAModule.query()` 调用 `query_context_mla_module/query_generation_mla_module`，其模块边界注释为 context 替代 `q_b_proj + kv_b_proj + ContextMLA + proj`，generation 替代 `MLABmm(pre) + GenerationMLA + MLABmm(post)`。
- `collector/sglang/collect_gemm.py` 的 fp8 DeepGEMM collector 直接调用 `gemm_nt_f8f8bf16((x_fp8, x_scale), (y_fp8, y_scale), out)`，输入和 scale 已预先构造。因此该表主要覆盖 DeepGEMM kernel 本身，不覆盖实机 child range 中可能出现的运行时 activation quantization kernel。
- `collector/sglang/collect_mla.py` 构造 `RadixAttention`/`ForwardBatch` 并校验 prefill 走 DeepSeek MHA 分支；它覆盖的是 attention/RadixAttention 这一类 kernel 边界，不包含 q_a/kv_a layernorm、RoPE child range 或 qkv/downscale GEMM。

## 重要解释

fp8 GEMM 的 AIC `GEMM` 查询来自 `collector/sglang/collect_gemm.py` 的 `fp8_block` DeepGEMM microbench；该 collector 预先构造 fp8 输入和 scale，并调用 `gemm_nt_f8f8bf16`，通常不包含实机 trace 中 child range 前置的 `per_token_group_quant_8bit_kernel`。因此 qkv/q_b/o_proj 一类 GEMM 对比中，实机侧常会比 AIC 侧多出激活动态量化 kernel 开销。

`q_a_layernorm`、`kv_a_layernorm`、`rotary_emb` 在实机中是显式 child range，但普通 DeepSeekModel 的 MLA path 未单独建模；这些算子若在图中只出现 Real bar，代表 AIC 当前口径没有独立对应项，而不是实机异常。

`attention` 的实机子模块可能是 `attn_mha` 或 `attn_mqa`。这与 SGLang mixed/prefix 调度相关；AIC 诊断拆解里的 `context_attention` 固定查询 `ContextMLA/query_context_mla` 并按 `run_agg` 第二段 attention pass 做 scale 修正，因此小 batch/prefix 命中 MQA 的 case 会天然存在 backend 口径差异。
