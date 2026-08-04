# GEMM 数据作图工作目录

本目录比较仓库中 SGLang 与 TensorRT-LLM 的 GEMM 实测数据。

## 比较口径

- 横轴：SGLang latency（ms）。
- 纵轴：TensorRT-LLM latency（ms）。
- 一个点：同一硬件、同一 `gemm_dtype` 下精确匹配的一个 `(M,N,K)`。
- 不抽样、不聚合；全部匹配 shape 都绘制。
- 主键重复时采用 SDK `load_gemm_data()` 一致的 first-wins 规则。
- 每种量化精度一张大图，每个硬件平台一个子图。
- 两轴采用相同范围的对数坐标；虚线为 `y=x`。
- 红点位于等性能线下方，表示 TRT-LLM latency 更低；蓝绿点位于线上方，表示 SGLang 更低或相等。

当前比较四种两框架仍有共同数据的精度：

```text
bfloat16, fp8, fp8_block, nvfp4
```

历史 `int8_wo/int4_wo/sq` 没有足够的同硬件、同 dtype 跨框架匹配数据，因此不生成空的大图；它们仍在上一级 GEMM 分析文档中说明。

## 版本选择

| 硬件 | SGLang | TensorRT-LLM |
|---|---|---|
| A100 SXM | 0.5.10 | 1.0.0 |
| L40S | 0.5.10 | 1.0.0 |
| H100/H200 SXM | 0.5.10 | 1.3.0rc10 |
| B200/B300/GB200/GB300 | 0.5.10 | 1.3.0rc10 |
| RTX PRO 6000 | 0.5.10 | 1.3.0rc10 |

这是“各平台较新可用数据”的比较，不是相同发布日期或相同依赖栈的严格实验。图中 panel 标题会显示实际版本。

## 运行

从仓库根目录执行：

```bash
python3 'collector/[self]doc/engine-diff/gemm-data-compare/plot_gemm_engine_comparison.py'
```

默认结果写入 `results/`：

- `gemm_<dtype>_sglang_x_trtllm_y.png`：高分辨率位图；
- `gemm_<dtype>_sglang_x_trtllm_y.pdf`：文字为矢量、散点栅格化的 PDF；
- `matched_shape_summary.csv`：每个硬件和 dtype 的原始行数、重复数、独有/匹配 shape 数、TRT/SGL 比例分位数及胜率。

可指定输出位置和 DPI：

```bash
python3 'collector/[self]doc/engine-diff/gemm-data-compare/plot_gemm_engine_comparison.py' \
  --output-dir /tmp/gemm-compare \
  --dpi 260
```
