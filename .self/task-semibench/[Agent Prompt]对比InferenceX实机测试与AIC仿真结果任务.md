# 对比InferenceX实机测试与AIC仿真结果任务

本文档是对当前任务和意图的总体描述

## 总体任务

1. 依据semianalysis的InferenceX，获取不同模型在不同推理框架+硬件平台（以及各种运行配置，包括并行策略、batch并发数等）下的运行表现。将其与AIC仿真结果进行对比。
2. 验证AIC仿真结果能否准确反映真实实机运行情况，分析误差来源。

## InferenceX实机测试的数据

1. 在InferenceX官网上可以找到各种模型在各种硬件平台下的性能表现图
   - 每种并发数(concurrency，应该是batch)和P/D并行配置（一般有TP、EP和Workers数）为一个数据点。
   - 以及还有其他影响因素和信息，如量化设定FP4/FP8等
2. InferenceX是开源项目，有源码库：https://github.com/SemiAnalysisAI/InferenceX
3. 在github库中也可以找到某模型某推理框架+硬件平台的Run sweep信息，如：https://github.com/SemiAnalysisAI/InferenceX/actions/runs/21793201578 (Deepseek-R1, sglang, GB200, 包括FP4和FP8)

## AIC仿真执行

1. 首先要将InferenceX中已有的实机执行数据相关配置，转换为AIC仿真执行的配置。以确保二者设定的对齐和统一。
2. 先以InferenceX中已有的配置和运行结果数据（先不大规模搜索），执行AIC仿真，对比二者情况，评估AIC的仿真准确度。
3. 根据仿真结果，对AIC的结果数据进一步进行breakdown（op级别分析），尝试找出误差来源。
4. 仿真和对比维度：关心latency、吞吐、时延等核心性能，暂不关心能耗等。