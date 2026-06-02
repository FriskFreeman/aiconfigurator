## 背景
1. 核心缘由是，在RTX PRO 6000硬件（SM版本：SM120）中不支持常规的FA、FlashMLA等算子。在sglang库中找到了一个民间的PR，使用Triton算子作为替代方案：
https://github.com/sgl-project/sglang/pull/24692
2. 在aiconfigurator库中，为了仿真评估PRO 6000硬件的运行性能（底层使用Triton算子），因此在collector组件中进行了一定适配，来在PRO 6000上采集Triton算子或模块的运行性能数据。
3. 但是现在遇到一个问题，发现测出来的Triton算子性能下降非常多，相较于FA算子性能差了20多倍。感觉非常不合理，怀疑是collector脚本代码存在一定缺陷：
   - Triton算子可能包含一些预编译、启动时间等
   - 在官方collector脚本实现中，采用了一套严格的工序和方法，将算子的准备、下发等时间尽可能剔除了，并大量使用cudagraph。以求在纯GPU角度捕获最核心最具代表性的算子纯粹GPU执行时间。
   - 目前怀疑是collector脚本代码存在一定的缺陷，导致对于Triton算子性能的采集未能反映真实情况（如未能忽视预编译、启动时间等）


## 需求
静态分析collector脚本代码，分析评估涉及Triton回退情况的算子相关测试脚本是否存在问题（上述问题或其他问题），输出分析文档（先不大修代码）
- 先解析一下相关collector脚本的实现方式，特别是对于sglang库调用、Triton算子调用和适配的过程和原理
- 分析Triton算子测试问题，包括是否存在问题，问题成因和修改建议等等。
- 本地包含已checkout到上述民间PR分支的sglang库，位置：/home/ai_lab/ljc/sglang。可结合该sglang库代码进行分析。