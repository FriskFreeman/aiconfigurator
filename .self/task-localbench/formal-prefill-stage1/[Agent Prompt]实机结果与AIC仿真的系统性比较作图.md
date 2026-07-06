MLA模块的实机与仿真汇总比较&作图任务

包含两个子步骤：创建CSV汇总表格、结果作图

目前仅考虑prefill部分，不纳入decode

工作文件夹和结果输出目录：bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/analysis/mla_aic_compare

## 1. 以AIC的普通DeepSeekModel作对比（kernel时间累加）

- 实机数据和比较口径：每实例每层为一行数据，考虑其GPU makespan时间（首尾kernel包络）和kernel_time_sum时间（所有kernel加总的净时间），并且【一律忽略首层】。并且详细记录每个kernel的具体持续时间。
- AIC仿真的比较口径：参照DeepSeekModel的FallbackOp，理论上会找不到context_mla_module的相关数据文档，会直接退回fallback的小算子时间累加。（执行时顺便验证一下该回退行为）。记录其每个小算子的时间，以及整个FallbackOp的时间（应该就是所有小算子时间的累加）
- 作图：
  - 实机数据侧：横轴为每个实例，实机数据侧多层结果视为重复实验作箱线图。另外对于每个kernel的平均时间，作堆积柱状图，自执行时间顺序从底到顶进行堆积（以不同颜色对其区分）
  - AIC仿真侧：以每个kernel的时间作堆积柱状图，与实机侧的柱状图类似，同含义kernel使用相近的颜色便于可视化对比。
  - 数据标注：标注实机侧的makespan和kernel_time_sum时间绝对值，AIC侧标注其时间绝对值，以及和实机侧的两个总时间的差距百分比

## 2. 以AIC的WideEPDeepSeekModel作对比（主要是WideEPContextMLA这个module级别的Ops以及目前独立出来的qkv_a_proj和norm等）

- 实机数据比较口径：与1.一致。
- AIC仿真的比较口径：参照WideEPDeepSeekModel的若干Ops，采取其中的Ops与实机的self_attn对齐即可。
- \*注意：WideEPDeepSeekModel源码中存在一个重复的qkv_a_proj GEMM，为目前bug。比较时仅纳入一个相应的gemm即可。
- 作图：
  - 实机数据侧：与1.一致。
  - AIC仿真侧：以WideEPDeepSeekModel的若干Ops的kernel时间作堆积柱状图。（此时包含的ops和1.不同，由于有模块级的Ops）
  - 数据标注：同1.一致。
