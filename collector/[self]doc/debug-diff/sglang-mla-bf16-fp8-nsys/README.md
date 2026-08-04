# SGLang 0.5.9 MLA BF16/FP8 Nsight实验

本目录比较基础`collect_mla`在H100 FA3后端上的`BF16/BF16`与`BF16/FP8`，范围包含prefill和decode，不包含MLA module。

文件：

- `profile_collect_mla.py`：复用当前`collector/sglang/collect_mla.py`的完整参数构造，仅替换计时函数以增加逐次NVTX标记和原始时延记录。
- `run_suite.py`：在指定GPU和SGLang 0.5.9 Docker镜像内执行匹配shape的CUDA Graph时延实验与eager Nsight采集。
- `analyze_nsys.py`：导出SQLite、限定NVTX边界解析kernel并生成对照表和图。
- `SGLang-MLA-BF16-FP8差异实机分析.md`：源码和实测结论。
- `results/`：原始日志、`.nsys-rep`、SQLite、CSV、JSON和图片；该目录受仓库根`.gitignore`规则忽略。

复现命令：

```bash
python collector/'[self]doc'/debug-diff/sglang-mla-bf16-fp8-nsys/run_suite.py --gpu 4
python collector/'[self]doc'/debug-diff/sglang-mla-bf16-fp8-nsys/analyze_nsys.py
```
