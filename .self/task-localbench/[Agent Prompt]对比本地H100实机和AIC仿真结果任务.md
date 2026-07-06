# 对比本地H100实机和AIC仿真结果任务

本文档是对当前任务和意图的总体描述

## 总体任务

1. 在H100服务器上实机运行sglang模型推理任务并进行性能采集trace，在子模块和底层算子级别获得运行信息。
2. 执行AIC仿真，在子模块和底层算子级别对比实机与仿真结果。分析当前仿真准确度和改进空间。
3. 目前聚焦于Deepseek-v3/v3.2模型的MLA模块（整个模型的TransformerBlock视为由MLA+MoE组成），关注整个MLA模块的内部及入口出口处相关计算和通信算子的完整性能信息。

## 本机实机运行模型推理

1. H100硬件环境：至多使用编号4和5这两张卡。（执行前先判断是否完全空闲，如有任何其他任务则不得占用）
2. 在docker中执行推理，docker运行环境：booleimg.myaddr.io/lmsysorg/sglang:v0.5.9
3. 本地没有下载Deepseek原始模型文件，并且硬件资源也不足以跑完整模型，为此进行针对性调整：
   - 使用sglang的dummy模型进行推理
   - 手动限制模型层数为前3层、前5层之类的（对于子模块和底层算子分析而言已足够代表性）。避免跑全量的模型权重。
   - 直接使用sglang的offline batch推理，直接使用Engine入口（绕开server等复杂前端层）。
4. 将端到端的和底层细节的profile trace等结果保存，后续进行一定数据处理等（目标是和AIC进行比对分析）

## AIC仿真执行

1. 始终和实机执行的运行配置保持对齐
2. 对结果进行breakdown（op级别分析等），尝试找出误差来源
3. *可用conda环境ljc01

## 子任务设定

### 1. prefill阶段的实机运行与仿真数据结果（collector+sdk）偏差【Doing】

先分以下几个阶段从实机运行中得到csv拆解数据形成数据库。
（且以下情景都控制fresh token总数在prefill chunk窗口大小内，即在单chunk内完成推理）

1. 每个请求等长情况（fresh和prefix都等长）
   1. （全部为fresh token）多个小到中请求，未填满或刚好填满单个prefill chunk。包含规整的（如1\*1、1\*8、512\*4、2048\*4、4096\*2、8192等）和不规整的（如300\*12、1200\*4、4500、6000等）。此时单chunk可完成全部任务无遗留token，也没有prefix_len等
   2. （包含prefix token）此时每个请求的fresh token同上，为多个小到中请求并且总长在单个prefill chunk长度内。同样包含规整和不规整的。但是每个请求都有prefix token数并且或长或短（如每个请求prefix 512、1k、2k、4k、8k、16k等等）
   3. 单个请求fresh token数正好为chunk大小（8192），但包含各种长度的prefix token（规整或不规整），如0、1k、4k、8k、10500、32k等等。
2. 在1的基础上，加上若干decode请求混合进来的情况（代表PD混合推理时sglang的Continuous Batching调度过程和结果）。理论上纳入decode将混入多个fresh token数为1，但有一定prefix token的请求（这部分的fresh token应该也会消耗chunk预算，因此prefill部分请求的总token数将受到影响）
   - 此时需要配置4个变量：各prefill请求的fresh和prefix token数（各请求等长）；decode请求的数量以及prefix token数（各请求等长）。prefill的其余情景设定与情况1相同（包括全fresh小到中请求、含prefix小到中请求和单个长请求）

数据库保存与档案管理要求：

- 参考src/aiconfigurator/systems/data中的“collector”采集的项目文件夹分级结构，在bench_data文件夹中也建立类似的结构：【测试平台/推理引擎/版本号/模块文件夹】（如h100_sxm/sglnag/v0.5.9/deepseek_v3_mla）
- 将正式运行的核心结果（nsys解析的csv数据）保存到相应的文件夹中，形成实机运行的数据库。
- 原始的运行结果以及日志保存在/deepseek_v3_mla的一个子文件夹中

## 其他要求

- 实机端的执行尽量要完整执行模型推理，然后从trace中定位事件获取相应时延，而不要像collector一样重新构建算子或模块去对比。
- 实机执行过程遇到比较大的问题，及时反馈，不要随便改方案
- 对于一些比较重大的改动或步骤，执行完落一些说明文档，供人参阅修改细节或进度情况
