# DeepSeek-V3.2 / GLM-5 模型建模梳理

本文基于 `v0.9.0` 的 `src/aiconfigurator/sdk/models/deepseek_v32.py`、`operations.py`、`perf_database.py`，梳理 AIC 对 DeepSeek-V3.2 / GLM-5 这一族模型的建模方式、算子拆分粒度，以及对应的性能数据来源。

## 1. 快速结论

`DeepSeekV32Model` 对应的不是旧版 DeepSeek-V3 那套 MLA 图纸，而是新的 DSA 图纸。

它最重要的变化是：

- Attention 不再拆成若干个 MLA 子算子再拼起来
- 而是优先直接使用 `ContextDSAModule` / `GenerationDSAModule`
- 即以“完整 DSA attention module”为核心采样和查询粒度

与此同时，MoE 部分仍然保留比较清晰的拆分：

- shared expert GEMM/激活
- router GEMM
- MoE dispatch 通信
- MoE compute
- generation 期 shared/routed 路径并行重叠

所以可以把 DeepSeek-V3.2 在 AIC 中理解成：

- Attention 侧：更偏 module-level
- MoE 侧：仍偏 op-level / submodule-level

---

## 2. 模型类与分支

DeepSeek-V3.2 / GLM-5 对应的 family 是 `DEEPSEEKV32`，入口类见：

- [deepseek_v32.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/models/deepseek_v32.py:16)

主要有 3 条分支：

1. `DeepSeekV32Model`
2. `TrtllmWideEPDeepSeekV32Model`
3. `WideEPDeepSeekV32Model`

分派规则是：

- 默认：走 `DeepSeekV32Model`
- `backend_name == "trtllm"` 且 `enable_wideep=True`：走 TRT-LLM WideEP 变体
- `backend_name == "sglang"` 且 `enable_wideep=True`：走 SGLang WideEP 变体

这意味着同一个“模型家族”在 AIC 中实际上对应三套不同图纸：

- 普通图纸
- TRT-LLM WideEP 图纸
- SGLang WideEP 图纸

---

## 3. 建模核心特征

## 3.1 并行拓扑约束

普通版和 WideEP 版都会先检查：

- `tp_size * attention_dp_size == moe_tp_size * moe_ep_size`
- `num_experts >= moe_ep_size`

这说明 AIC 默认把：

- attention 路径的 TP/DP
- MoE 路径的 TP/EP

看成同一张并行资源版图中的两种切分表达，必须保持总宽度一致。

## 3.2 generation 期带 MTP 缩放

`DeepSeekV32Model` 里 generation 路径使用：

- `_mtp_scale_factor`

该系数综合考虑：

- nextn 的期望接受率
- 额外 MTP 层带来的开销

因此 generation 侧很多 op 的 `scale_factor` 都不是简单的 `num_layers`，而是：

- `num_layers * _mtp_scale_factor`

TRT-LLM WideEP 版还进一步引入：

- `_pdl_factor = 0.9`

所以它的 generation MoE / DSA 路径又会比普通版再多一层缩放。

## 3.3 KV cache 建模更接近 DSA/索引器真实结构

`DeepSeekV32Model.get_kvcache_bytes_per_sequence()` 没有沿用普通 GQA/MHA 的简单公式，而是把：

- `kv_lora_rank`
- `qk_rope_head_dim`
- `index_head_dim`

都计入了每层每 token 的缓存占用。

这说明 V3.2 的注意力在 AIC 里不只是“算子 latency 变了”，连 KV cache 结构模型也一起变了。

---

## 4. Attention 建模粒度

这是和 DeepSeek-V3 最大的区别。

## 4.1 普通版：直接用 DSA module 级 op

普通版 `DeepSeekV32Model` 在 context 阶段直接放入：

- `ops.ContextDSAModule("context_attention", ...)`

在 generation 阶段直接放入：

- `ops.GenerationDSAModule("generation_attention", ...)`

也就是说，AIC 不再把 V3.2 的 attention 主要表示成：

- 下采样 GEMM
- q/k/v 投影
- MLA attention kernel
- BMM pre/post
- output projection

这些细粒度核的显式顺序和求和。

而是把这些组合逻辑尽量封装在 DSA module perf 表里，让数据库直接给出该 module 的总 latency / energy。

## 4.2 对应的 ops 类含义

在 `operations.py` 中：

- `ContextDSAModule`
- `GenerationDSAModule`

它们最终分别调用：

- `database.query_context_dsa_module(...)`
- `database.query_generation_dsa_module(...)`

因此从查询路径看，DeepSeek-V3.2 的 attention 是标准的 module-level perf query。

## 4.3 为什么这是重要变化

这意味着 V3.2 的仿真精度更依赖：

- module 级 profile 是否齐全
- DSA module 的 architecture 维度是否匹配
- collector 是否真实复现了框架内部 DSA 路径

而不是像旧版 MLA 模型那样，更依赖若干通用 GEMM/BMM/attention 子表的拼接。

对于你后面做实机对比，这一点非常关键：

- V3.2 的 attention 误差更可能来自“module 采样抽象本身”
- V3 的 attention 误差更可能来自“子算子组合与 overlap 建模”

---

## 5. MoE 建模粒度

虽然 attention 提升到了 module 粒度，但 MoE 没有被完全模块化。

普通版 `DeepSeekV32Model` 的 MoE 路径仍然拆成：

- shared expert gate/up GEMM
- shared expert act/gate ElementWise
- shared expert down GEMM
- router GEMM
- `MoEDispatch` pre
- `MoE` compute
- `MoEDispatch` post

其中 generation 期最关键的是：

- shared path
- routed path

通过 `OverlapOp("generation_moe_overlap", ...)` 建模为两条流并行，延迟取 `max()` 而不是简单求和。

这说明在 AIC 中：

- attention 的抽象粒度更粗
- MoE 的抽象粒度更细

从校准角度看，MoE 路径更适合继续追到通信和计算分项。

---

## 6. 普通版结构化 Ops 图纸表

下表对应普通 `DeepSeekV32Model`。

| 阶段 | 图纸节点名 | Ops 类 | 建模含义 | 粒度说明 |
| --- | --- | --- | --- | --- |
| Context | `context_embedding` | `Embedding` | 词嵌入 | 单独 op |
| Context | `context_add_norm_1` | `ElementWise` | attention 前 add/norm | 单独 op |
| Context | `context_attention` | `ContextDSAModule` | 完整 DSA context attention block | module-level |
| Context | `context_add_norm_2` | `ElementWise` | FFN/MoE 前 add/norm | 单独 op |
| Context | `context_shared_gate_up_gemm` | `GEMM` | shared expert fused gate/up | submodule-level |
| Context | `context_shared_act_gate` | `ElementWise` | shared expert 激活/门控 | submodule-level |
| Context | `context_shared_ffn2_gemm` | `GEMM` | shared expert down proj | submodule-level |
| Context | `context_router_gemm` | `GEMM` | routed expert router | submodule-level |
| Context | `context_moe_pre_dispatch` | `MoEDispatch` | routed tokens 发出通信 | op-level comm |
| Context | `context_moe` | `MoE` | routed experts 计算 | op-level compute |
| Context | `context_moe_post_dispatch` | `MoEDispatch` | routed outputs 回收通信 | op-level comm |
| Context | `context_logits_gemm` | `GEMM` | logits 输出头 | 单独 op |
| Context | `context_p2p` | `P2P` | pipeline parallel 跨 stage 传输 | 单独 op |
| Generation | `generation_embedding` | `Embedding` | 解码 token embedding | 单独 op |
| Generation | `generation_add_norm_1` | `ElementWise` | attention 前 add/norm | 单独 op |
| Generation | `generation_attention` | `GenerationDSAModule` | 完整 DSA decode attention block | module-level |
| Generation | `generation_add_norm_2` | `ElementWise` | FFN/MoE 前 add/norm | 单独 op |
| Generation | `generation_moe_overlap` | `OverlapOp` | shared path 与 routed path 并行 | 组合级 |
| Generation | `generation_logits_gemm` | `GEMM` | 解码 logits 输出头 | 单独 op |
| Generation | `generation_p2p` | `P2P` | pipeline parallel 跨 stage 传输 | 单独 op |

其中 `generation_moe_overlap` 内部实际展开为两组：

共享支路：

- `generation_shared_gate_up_gemm`
- `generation_shared_act_gate`
- `generation_shared_ffn2_gemm`

路由支路：

- `generation_router_gemm`
- `generation_moe_pre_dispatch`
- `generation_moe`
- `generation_moe_post_dispatch`

---

## 7. WideEP 变体图纸差异

## 7.1 TRT-LLM WideEP 版

`TrtllmWideEPDeepSeekV32Model` 的 attention 仍然是 module-level DSA：

- `ContextDSAModule`
- `GenerationDSAModule`

所以 V3.2 的 attention 在普通版和 TRT-LLM WideEP 版之间没有变成细粒度展开。

真正变化主要集中在 MoE：

- 通信 op 从 `MoEDispatch` 换成 `TrtLLMWideEPMoEDispatch`
- 计算 op 从 `MoE` 换成 `TrtLLMWideEPMoE`
- 多了 `context_moe_reduce_add` / `generation_moe_reduce_add`
- generation 期继续保留 `OverlapOp`
- shared expert 维度不再按 `tp_size` 缩小，而是走 shared_tp_size=1 的建模思路

它对应的数据源也随之变化：

- MoE compute 走 `wideep_moe_compute_perf.txt`
- All2All 走 `trtllm_alltoall_perf.txt`

## 7.2 SGLang WideEP 版

`WideEPDeepSeekV32Model` 中，attention 依旧是：

- `ContextDSAModule`
- `GenerationDSAModule`

但模型图纸更加“瘦身”，很多普通版里显式的 embedding、addnorm、logits、p2p 没再完全照搬进来，重点转向：

- DSA attention
- TP all_gather / reduce_scatter
- shared expert GEMM
- DeepEP / WideEP 的 dispatch + MoE compute

因此如果你后面只关注 H100 上 SGLang 的 MLA/DSA 模块校准，要先确认你比较的是：

- 普通 `DeepSeekV32Model`
- 还是 `WideEPDeepSeekV32Model`

这两者图纸并不完全等价。

---

## 8. 性能数据文件与采集来源

## 8.1 Attention 相关 perf 表

DeepSeek-V3.2 主要命中的 attention 相关文件是：

- `dsa_context_module_perf.txt`
- `dsa_generation_module_perf.txt`

对应 loader / query：

- `load_context_dsa_module_data()`
- `load_generation_dsa_module_data()`
- `query_context_dsa_module()`
- `query_generation_dsa_module()`

这就是它和 DeepSeek-V3 的最大区别：  
V3 更偏向 `context_mla_perf.txt` / `generation_mla_perf.txt` 以及 `mla_*_module_perf.txt`，而 V3.2 首要依赖的是 DSA module 表。

## 8.2 MoE 与 WideEP 相关 perf 表

普通 MoE：

- `moe_perf.txt`

SGLang WideEP / DeepEP：

- `wideep_context_moe_perf.txt`
- `wideep_generation_moe_perf.txt`

TRT-LLM WideEP：

- `wideep_moe_compute_perf.txt`
- `trtllm_alltoall_perf.txt`

## 8.3 采集脚本

和 DeepSeek-V3.2 直接相关的 collector 主要有：

- `collector/sglang/collect_mla_module.py`
- `collector/trtllm/collect_mla_module.py`
- `collector/vllm/collect_mla_module_v3.py`

这几份脚本里都能看到对：

- `DeepseekV32ForCausalLM`
- `GlmMoeDsaForCausalLM`

的专门兼容或映射处理。

MoE / WideEP 相关采集则主要包括：

- `collector/sglang/collect_moe.py`
- `collector/sglang/collect_wideep_deepep_moe.py`
- `collector/trtllm/collect_moe_v3.py`
- `collector/trtllm/collect_wideep_moe_compute.py`
- `collector/slurm_comm_collector/collect_trtllm_alltoall.py`

---

## 9. 对“实机校准任务”的直接意义

如果你现在要做 H100 上的模块/算子级校准，DeepSeek-V3.2 这条线建议优先分成两层看。

第一层：attention module 校准

- 直接对 `ContextDSAModule` / `GenerationDSAModule`
- 看 AIC 的 module 级结果和真实 trace 的 DSA 大块时间是否对齐

第二层：MoE 路径拆分校准

- shared expert GEMM
- router GEMM
- dispatch 通信
- MoE compute
- overlap 关系

因为在 V3.2 里：

- attention 已经被 AIC 抬到 module 级
- 而 MoE 仍保留了较强的可拆性

所以两部分的误差分析方法应该不同，不能混成一类。

---

## 10. 一句话总结

DeepSeek-V3.2 / GLM-5 在 AIC 中的核心特征，是“DSA attention module-level + MoE submodule/op-level”的混合建模方式。Attention 精度主要取决于 DSA module 数据表和 collector 抽象是否贴近真实框架实现；MoE 精度则更依赖 dispatch、compute、overlap 等更细粒度路径的还原质量。
