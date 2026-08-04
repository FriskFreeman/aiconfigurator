# Development 索引

本目录只记录最终模型的来源，不放入运行时参数搜索、临时图或数据副本。最终工具不会读取这里的任何文件。

历史建模与证据位于工作区：

| 内容 | 位置 |
| --- | --- |
| A100/H100 SGLang 0.5.9 全量分析 | `.self/compass-sim-fa_sglang059/` |
| 六组 prefill/decode 基础校准 | `.self/compass-sim-fa_sglang059/data/processed/` |
| decode 下凹残差与候选 task 变量 | `.self/compass-sim-fa_sglang059/docs/Decode下凹误差曲线聚类分析与建模建议.md` |
| decode task-cycle 拟合与 bootstrap | `.self/compass-sim-fa-decode-granularity/` |
| H200/L40S/Blackwell 跨硬件检验 | `.self/compass-sim-fa-cross-hardware/` |
| LLMCompass 权威实现 | `LLMCompass/software_model/flashattention.py` |
| 完整模型研究指南 | `LLMCompass/.self/docs/FA_Roofline模型指南与跨硬件泛用性总结.md` |
| H100 SGLang 0.5.9 MLA 数据、OOF、bootstrap 与图表 | `.self/compass-sim-mla_sglang059/` |
| MLA 最终结构、L2 reuse 与 rounded wave | `.self/compass-sim-mla_sglang059/docs/最终MLA模型结构与FA及普通Attention差异.md` |

内嵌的 standard/high/low 是上述实验经验综合后的通用工程参数，不再保留 A100/H100、架构、backend、phase 或 dtype 路由，也不是运行时重新拟合。任何后续更新都应先在独立实验目录中完成数据审计、训练/测试验证和跨硬件压力测试，再修改 `fa_roofline/profiles.py` 并提升版本号。

MLA 使用独立 `mla_profiles.py`。当前三档仅覆盖 BF16；FP8 异常数据只作为不兼容证据，不允许用于生成生产 profile。H100 BF16 precise 文件位于 `configs/reference/`，只能显式加载。
