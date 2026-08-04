# AIC collector数据问题核查索引

## SGLang基础MLA prefill形状修复

- [结论文档](sglang-mla-prefill-shape-fix/SGLang-MLA-prefill形状修复跨硬件数据核查.md)
- [工作目录说明](sglang-mla-prefill-shape-fix/README.md)
- [分析脚本](sglang-mla-prefill-shape-fix/analyze_mla_shape_fix.py)
- [统计与图表](sglang-mla-prefill-shape-fix/results/)

该核查以H100 SGLang 0.5.9重采数据为正确参考，严格匹配逻辑shape与量化配置后，对照H100旧0.5.10及H200、Blackwell、RTX PRO 6000原始表。范围只包含基础MLA context/generation，不包含MLA module。

## SGLang基础MLA BF16/FP8实机归因

- [结论文档](sglang-mla-bf16-fp8-nsys/SGLang-MLA-BF16-FP8差异实机分析.md)
- [工作目录说明](sglang-mla-bf16-fp8-nsys/README.md)
- [实验runner](sglang-mla-bf16-fp8-nsys/run_suite.py)
- [Nsight解析与作图](sglang-mla-bf16-fp8-nsys/analyze_nsys.py)
- [原始报告、汇总与图表](sglang-mla-bf16-fp8-nsys/results/)

该实验在H100、SGLang 0.5.9 FA3后端上复现匹配shape的CUDA Graph时延，并以逐iteration NVTX关联Nsight kernel。结论是prefill慢于Q/V边界内量化未被FP8主核收益覆盖；decode的大幅变慢来自完整FP8 MLA cache在每次调用中被物化回BF16，主attention kernel仍为BF16。
