# GEMM 数据图结果解读

图中横轴为 SGLang latency，纵轴为 TensorRT-LLM latency；每个散点是一个精确匹配的 `(M,N,K)`。定义：

```text
ratio = TensorRT-LLM latency / SGLang latency
```

- `ratio < 1`：TensorRT-LLM 更快，点位于 `y=x` 下方。
- `ratio > 1`：SGLang 更快，点位于 `y=x` 上方。

所有图均使用全量匹配数据，没有抽样或聚合。

## 图和点数

| 精度 | 全部硬件的匹配点数 | 图 |
|---|---:|---|
| BF16 | 267,752 | [PNG](results/gemm_bfloat16_sglang_x_trtllm_y.png) / [PDF](results/gemm_bfloat16_sglang_x_trtllm_y.pdf) |
| FP8 | 258,927 | [PNG](results/gemm_fp8_sglang_x_trtllm_y.png) / [PDF](results/gemm_fp8_sglang_x_trtllm_y.pdf) |
| block-FP8 | 176,333 | [PNG](results/gemm_fp8_block_sglang_x_trtllm_y.png) / [PDF](results/gemm_fp8_block_sglang_x_trtllm_y.pdf) |
| NVFP4 | 147,592 | [PNG](results/gemm_nvfp4_sglang_x_trtllm_y.png) / [PDF](results/gemm_nvfp4_sglang_x_trtllm_y.pdf) |

完整的匹配数、独有 shape 数、重复数和 P10/P50/P90 见 [matched_shape_summary.csv](results/matched_shape_summary.csv)。

## BF16

| 硬件 | 匹配 shape | ratio 中位数 | TRT-LLM 更快占比 |
|---|---:|---:|---:|
| A100 | 8,800 | 0.954 | 87.0% |
| L40S | 8,800 | 0.954 | 78.5% |
| H100 | 35,742 | 0.992 | 57.4% |
| H200 | 35,742 | 0.963 | 70.9% |
| B200 | 35,742 | 0.967 | 68.4% |
| B300 | 35,742 | 0.992 | 55.3% |
| GB200 | 35,742 | 0.999 | 50.6% |
| GB300 | 35,742 | 0.997 | 52.5% |
| RTX PRO 6000 | 35,700 | 0.966 | 65.2% |

BF16 整体最接近等性能线。H100、B300、GB200、GB300 的中位差在 1% 内；A100/L40S 使用较旧 TRT-LLM 1.0.0 网格，只能匹配 8,800 个 shape，不能与其余平台的完整新网格等量比较。

## FP8

| 硬件 | 匹配 shape | ratio 中位数 | TRT-LLM 更快占比 |
|---|---:|---:|---:|
| A100 | 0 | - | - |
| L40S | 8,800 | 1.110 | 35.6% |
| H100 | 35,741 | 1.210 | 27.2% |
| H200 | 35,742 | 1.229 | 26.1% |
| B200 | 35,742 | 1.393 | 18.5% |
| B300 | 35,742 | 1.420 | 15.5% |
| GB200 | 35,742 | 1.463 | 13.4% |
| GB300 | 35,742 | 1.469 | 12.8% |
| RTX PRO 6000 | 35,676 | 0.996 | 50.7% |

除 RTX PRO 6000 外，普通 FP8 的点云明显偏向 `y=x` 上方，SGLang collector 路径占优；Blackwell B/GB 平台差距最大。RTX PRO 6000 的中位数接近 1，但点云两侧分散，不能解释为所有 shape 都等性能。

该结果包含两框架不同的 activation quantization 和 scale scheme，不应直接解释为某个纯 FP8 GEMM kernel 的性能排名。

## Block FP8

| 硬件 | 匹配 shape | ratio 中位数 | TRT-LLM 更快占比 |
|---|---:|---:|---:|
| A100 / L40S | 0 | - | - |
| H100 | 29,524 | 0.990 | 52.5% |
| H200 | 29,526 | 0.994 | 51.4% |
| B200 | 29,289 | 0.856 | 68.6% |
| B300 | 29,242 | 0.855 | 69.4% |
| GB200 | 29,370 | 0.855 | 69.0% |
| GB300 | 29,382 | 0.869 | 67.4% |
| RTX PRO 6000 | 0 | - | - |

H100/H200 基本沿等性能线分布。B200/B300/GB200/GB300 上 TRT-LLM 的中位 latency 约为 SGLang 的 85.5%~86.9%，small/medium latency 区域的下偏尤其明显。

RTX PRO 6000 没有共同点不是数据 merge 失败：SGLang 当前不生成 SM120 block-FP8，TRT-LLM 单边有 29,268 个 shape。

## NVFP4

| 硬件 | 匹配 shape | ratio 中位数 | TRT-LLM 更快占比 |
|---|---:|---:|---:|
| A100/L40S/H100/H200 | 0 | - | - |
| B200 | 29,526 | 1.258 | 12.3% |
| B300 | 29,526 | 1.349 | 8.1% |
| GB200 | 29,526 | 1.276 | 12.9% |
| GB300 | 29,526 | 1.368 | 7.1% |
| RTX PRO 6000 | 29,488 | 0.970 | 66.8% |

B/GB 平台上 SGLang NVFP4 路径明显占优，其中 B300/GB300 的中位差距最大。RTX PRO 6000 的方向相反但差距较小，TRT-LLM 中位 latency 约低 3%。这与两边不同的 FlashInfer/TRT-LLM Linear 后端和 layout 路径一致，仍不能脱离 collector 算子链解释为纯 FP4 kernel 对比。

## 使用限制

1. A100/L40S 的 TRT-LLM 版本和 shape 网格较旧。
2. 图按每个合成 shape 等权，不按实际模型调用频率加权。
3. 重复主键采用 SDK 一致的 first-wins；GB200 block-FP8 去除了 8 条后写重复，GB300 NVFP4 去除了 4 条。
4. 图展示端到端 collector closure 的 latency，量化前处理是否计入取决于框架和 dtype。
5. `kernel_source` 不能证明实际 CUDA kernel；后端归因仍需 profiler trace。
