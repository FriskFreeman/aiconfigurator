# AGG Mixed 非 Attn Projection 修正形状查表分析

本文只覆盖 `qkv_a_proj / q_b_proj / o_proj` 三个 projection GEMM；`kv_b_proj` 和 `mla_concat_k` 的实机边界另有问题，不在本次修正形状口径内。

## 形状推导

SGLang DeepSeek-V3 TP=1 下三类 projection 的 GEMM 形状为：

| op | 实机 component | AIC GEMM 形状 | M 的来源 |
|---|---|---|---|
| `qkv_a_proj` | `fused_qkv_a_proj_with_mqa` | `(M, N=2112, K=7168)` | Nsight `attention_token_count` |
| `q_b_proj` | `q_b_proj` | `(M, N=24576, K=1536)` | Nsight `attention_token_count` |
| `o_proj` | `o_proj` | `(M, N=7168, K=16384)` | Nsight `attention_token_count` |

`run_agg` 外层 diagnostic 原先使用 first-pass `query_s` 查询这些 GEMM；但 mixed batch 中这些 projection 实际处理的是当前 formal batch 的真实 token 数。因此本分析把 AIC 查表的 `M` 修正为实机 trace 捕获的 `attention_token_count`。

实机侧同时给出两个口径：`real_gemm_kernel_ms` 是纯 `sm90_fp8_gemm_1d2d_impl` kernel；`real_child_ms` 是该子模块 NVTX range 的 GPU makespan，通常包含 `per_token_group_quant_8bit_kernel + DeepGEMM`。AIC `GEMM(fp8_block)` 更接近前者，不包含 runtime activation quant。

## 汇总结论

| operator_family   |   cases |   avg_original_ratio_child_to_aic |   avg_corrected_ratio_child_to_aic |   avg_corrected_ratio_gemm_to_aic |   median_corrected_ratio_gemm_to_aic |   avg_abs_gemm_minus_corrected_aic_ms |   max_abs_gemm_minus_corrected_aic_ms |   avg_quant_kernel_ms |   exact_shape_cases |
|:------------------|--------:|----------------------------------:|-----------------------------------:|----------------------------------:|-------------------------------------:|--------------------------------------:|--------------------------------------:|----------------------:|--------------------:|
| o_proj            |      12 |                          0.655585 |                           0.903086 |                          0.797014 |                             0.795554 |                              0.184100 |                              0.421359 |              0.084266 |                   3 |
| q_b_proj          |      12 |                          0.744988 |                           1.031053 |                          0.898650 |                             0.883721 |                              0.029595 |                              0.060759 |              0.008648 |                   0 |
| qkv_a_proj        |      12 |                          0.997657 |                           1.289702 |                          0.759994 |                             0.744667 |                              0.032390 |                              0.049373 |              0.039882 |                   0 |

结论：

- `q_b_proj` 在修正 `M` 后基本对齐，纯 GEMM kernel / AIC 的均值约为 `0.895`，剩余误差主要来自 perf 表插值与 collector/实机环境差异。
- `qkv_a_proj` 修正后仍偏离，纯 GEMM kernel / AIC 均值约为 `0.759`；AIC 对该形状没有同 `(N,K)` 采样族，属于 3D 插值结果。
- `o_proj` 修正后仍偏离，纯 GEMM kernel / AIC 均值约为 `0.798`；其中 `M=8192` 有 exact `fp8_block` 行但仍明显慢于实机 kernel，说明不只是插值问题，也可能包含 collector DeepGEMM microbench 与 engine 内实际 DeepGEMM 调用路径/输入布局/调度环境差异。
- 若使用 `real_child_ms` 与 AIC 比较，`qkv_a_proj` 会显得更接近甚至偏大，这是因为 child range 包含 activation quant；但 AIC GEMM 表按当前实现不覆盖该 quant kernel，不应把这部分当成 GEMM 表误差。

## 代表用例明细

| case_id   | operator_family   |   corrected_m |     n |     k |   real_gemm_kernel_ms |   corrected_aic_gemm_ms |   corrected_ratio_gemm_to_aic | corrected_shape_exact_in_gemm_perf   |
|:----------|:------------------|--------------:|------:|------:|----------------------:|------------------------:|------------------------------:|:-------------------------------------|
| C01       | o_proj            |          2052 |  7168 | 16384 |              0.335664 |                0.398569 |                      0.842171 | False                                |
| C01       | q_b_proj          |          2052 | 24576 |  1536 |              0.126325 |                0.136069 |                      0.928394 | False                                |
| C01       | qkv_a_proj        |          2052 |  2112 |  7168 |              0.059440 |                0.080613 |                      0.737350 | False                                |
| C03       | o_proj            |          4104 |  7168 | 16384 |              0.630362 |                0.791842 |                      0.796070 | False                                |
| C03       | q_b_proj          |          4104 | 24576 |  1536 |              0.229541 |                0.262453 |                      0.874598 | False                                |
| C03       | qkv_a_proj        |          4104 |  2112 |  7168 |              0.106373 |                0.140786 |                      0.755565 | False                                |
| C06       | o_proj            |          8192 |  7168 | 16384 |              1.124784 |                1.521796 |                      0.739116 | True                                 |
| C06       | q_b_proj          |          8192 | 24576 |  1536 |              0.433312 |                0.488263 |                      0.887457 | False                                |
| C06       | qkv_a_proj        |          8192 |  2112 |  7168 |              0.198923 |                0.245640 |                      0.809814 | False                                |
| C12       | o_proj            |          4106 |  7168 | 16384 |              0.629962 |                0.792367 |                      0.795038 | False                                |
| C12       | q_b_proj          |          4106 | 24576 |  1536 |              0.228384 |                0.262410 |                      0.870332 | False                                |
| C12       | qkv_a_proj        |          4106 |  2112 |  7168 |              0.106411 |                0.140721 |                      0.756184 | False                                |

完整 12 个 case 的逐算子明细见 `nonattn_proj_corrected_shape_compare.csv`。

## 产物

- `nonattn_proj_corrected_shape_compare.csv`: case/op 级明细，含修正后 GEMM 形状、实机 child/GEMM/quant 时延、AIC 原始与修正查表值。
- `nonattn_proj_corrected_shape_summary.csv`: op 级汇总。
- `nonattn_proj_corrected_shape_compare.png/svg`: 每个 op 的实机与 AIC 修正查表对比图。
