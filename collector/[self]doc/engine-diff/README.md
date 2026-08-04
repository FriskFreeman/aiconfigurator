# AIC collector 引擎差异分析索引

本目录保存不同推理引擎同名collector的源码、采集边界和数据面比较。

## Attention 与基础 MLA dtype

最新主文档：

- [`attention-mla-dtype-analysis/Attention与基础MLA-Dtype全链路分析.md`](attention-mla-dtype-analysis/Attention与基础MLA-Dtype全链路分析.md)

该版本是当前唯一推荐入口，范围严格限定为普通Attention和基础MLA，排除MLA module、WideEP MLA和DSA module。文档按`prefill/decode × SGLang/TRT-LLM/vLLM`展开，逐项列出collector设计组合、数据实存组合、硬件/backend差异、SDK效力及P10/P50/P90和small/large规模关系。

配套工作目录：

- [`attention-mla-dtype-analysis/README.md`](attention-mla-dtype-analysis/README.md)：复现方法和结果文件说明
- [`attention-mla-dtype-analysis/analyze_dtype_data.py`](attention-mla-dtype-analysis/analyze_dtype_data.py)：数据分析与作图脚本
- [`attention-mla-dtype-analysis/results/`](attention-mla-dtype-analysis/results/)：12组引擎独立图和机器可读统计

历史文档：

- [`Attention-MLA-Dtype字段有效性与后端精度核查.md`](Attention-MLA-Dtype字段有效性与后端精度核查.md)：早期全范围核查，包含后来排除的MLA module，不应代替最新主文档
- [`Attention-SGLang-vs-TRTLLM-Collector差异分析.md`](Attention-SGLang-vs-TRTLLM-Collector差异分析.md)：普通Attention的SGLang/TRT-LLM早期比较

## GEMM

- [`GEMM-SGLang-vs-TRTLLM-Collector差异分析.md`](GEMM-SGLang-vs-TRTLLM-Collector差异分析.md)：collector和后端kernel差异
- [`gemm-data-compare/comparison_findings.md`](gemm-data-compare/comparison_findings.md)：同shape数据比较结论
- [`gemm-data-compare/`](gemm-data-compare/)：脚本、图和统计结果
