# DeepSeek-V3 / R1 模型建模梳理

本文基于 `v0.9.0` 的 `src/aiconfigurator/sdk/models/deepseek.py`、`operations.py`、`perf_database.py`，梳理 AIC 对 DeepSeek-V3 / R1 这条模型线的建模方式、attention 与 MoE 的拆分粒度，以及对应的性能数据来源。

## 1. 快速结论

DeepSeek-V3 在 AIC 中仍然是一套“MLA 主导”的图纸，而不是 V3.2 的 DSA 图纸。

其 attention 建模的核心特点是：

- 优先尝试 module-level `MLAModule`
- 若 module perf 表缺失，则退回一组更细的 fallback ops

也就是说，DeepSeek-V3 的 attention 在 AIC 中是“module-level 优先、细粒度 fallback 兜底”的混合方案。

而 MoE 部分与 V3.2 类似，仍然保持比较细的拆分：

- shared expert
- router
- dispatch
- compute
- generation 期 overlap

因此和 DeepSeek-V3.2 相比：

- V3 的 attention 更容易退回细粒度 MLA 子算子
- V3.2 的 attention 更倾向于直接依赖 DSA module perf 表

---

## 2. 模型类与分支

DeepSeek-V3 / R1 对应的实现类是：

- `DeepSeekModel`
- 见 [deepseek.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/models/deepseek.py:16)

它同时也注册给：

- `DEEPSEEK`
- `KIMIK25`

但本文只聚焦 `DEEPSEEK` 这条线。

主要分支有 3 条：

1. `DeepSeekModel`
2. `TrtllmWideEPDeepSeekModel`
3. `WideEPDeepSeekModel`

分派逻辑是：

- 默认：普通 `DeepSeekModel`
- `backend_name == "trtllm"` 且 `enable_wideep=True`：TRT-LLM WideEP
- `backend_name == "sglang"` 且 `moe_backend == "deepep_moe"`：SGLang WideEP/DeepEP

所以和 V3.2 一样，V3 也不是只有一张固定图纸，而是会随 backend / WideEP 配置切换不同实现。

---

## 3. 建模核心特征

## 3.1 仍然是 DeepSeek 风格 MLA + MoE 拆块

DeepSeek-V3 在 AIC 中可以粗略看成一个 block 由两大部分组成：

- MLA attention block
- MoE block

和你当前本地校准任务里“把 TransformerBlock 视作 MLA + MoE”的理解是一致的。

## 3.2 attention 和 MoE 并行域有耦合约束

模型初始化会检查：

- `tp_size * attention_dp_size == moe_tp_size * moe_ep_size`
- `num_experts >= moe_ep_size`

这表明 AIC 把 DeepSeek-V3 的 attention 并行域和 MoE 并行域看成同一套资源切分关系的两种投影。

## 3.3 generation 期引入 MTP 缩放

generation 路径中的大多数 op 会乘上：

- `_mtp_scale_factor`

这个系数不是简单的“层数倍数”，而是把：

- speculative / multi-token prediction 的收益
- 额外层开销

折算进了解码阶段总计算量。

因此你后面若拿单 token decode trace 对比 AIC，必须确认：

- `nextn`
- `nextn_accept_rates`

是否和实机实验保持一致。

---

## 4. Attention 建模粒度

这是 DeepSeek-V3 最值得单独讲清楚的部分。

## 4.1 `FallbackOp` 是核心机制

在普通 `DeepSeekModel` 中：

- context attention 不是单一 op
- generation attention 也不是单一 op

它们都先被包装成一个 `FallbackOp`：

- `context_mla_block`
- `generation_mla_block`

这个 `FallbackOp` 的语义是：

1. 先强制按 `SILICON` 模式尝试 primary op
2. 如果 primary 的 perf data 缺失，才退回 fallback 序列

也就是说，AIC 对 DeepSeek-V3 的 attention 建模不是二选一静态写死，而是：

- 运行时优先 module-level
- 缺表时自动回退到细粒度拼装

## 4.2 primary：`MLAModule`

普通版 primary op 是：

- `ops.MLAModule("context_mla_module", ...)`
- `ops.MLAModule("generation_mla_module", ...)`

对应数据库查询：

- `query_context_mla_module()`
- `query_generation_mla_module()`

这代表一种“整块 MLA 模块”的 profile 粒度。

## 4.3 fallback：一串更细的 MLA 子算子

若 module 数据缺失，则会退回细粒度路径。

### Context fallback

context 侧 fallback 由这些 op 组成：

- `context_downscale_gemm`
- `context_q_b_proj_gemm`
- `context_kv_b_proj_gemm`
- `context_attention`
- `context_proj_gemm`

这里 `context_attention` 又分 backend：

- vLLM：`ContextAttention`
- 其他：`ContextMLA`

也就是说，在 vLLM 下，DeepSeek-V3 可能被当成“标准 attention 风格”去建模；
而在 TRT-LLM / SGLang 下，更接近“显式 MLA attention kernel”。

### Generation fallback

generation 侧 fallback 更复杂，通常包括：

- `generation_downscale_gemm`
- `generation_q_b_proj_gemm`
- vLLM 路径：`GenerationAttention`
- 非 vLLM 路径：`generation_bmm_pre` + `GenerationMLA` + `generation_bmm_post`
- `generation_proj_gemm`

所以 generation 期实际上比 context 期更接近真实 MLA 内部执行流。

## 4.4 这对校准意味着什么

DeepSeek-V3 attention 的误差来源可能来自两类完全不同的地方：

1. `MLAModule` 模块级 profile 是否准确
2. fallback 细粒度路径的求和、overlap 和 backend 选择是否准确

因此做对比时，一定要先确认实机配置最终走的是：

- module-level primary
- 还是 fallback 子算子路径

否则 AIC 和实机可能根本不是在比较同一个抽象层级。

---

## 5. MoE 建模粒度

DeepSeek-V3 的 MoE 路径与 V3.2 很相似，仍然是相对细粒度的拆分。

Context 阶段显式包含：

- shared gate/up GEMM
- shared act/gate
- shared ffn2 GEMM
- router GEMM
- `MoEDispatch` pre
- `MoE`
- `MoEDispatch` post

Generation 阶段则进一步用：

- `OverlapOp("generation_moe_overlap", ...)`

去表达：

- shared expert 路径
- routed expert 路径

在不同 CUDA stream 上并发执行，最终 latency 取 `max()`。

所以在 DeepSeek-V3 中：

- attention 是“可模块化、可回退”的混合建模
- MoE 是“较细粒度 + overlap 建模”

---

## 6. 普通版结构化 Ops 图纸表

下表对应普通 `DeepSeekModel`。

| 阶段 | 图纸节点名 | Ops 类 | 建模含义 | 粒度说明 |
| --- | --- | --- | --- | --- |
| Context | `context_embedding` | `Embedding` | 词嵌入 | 单独 op |
| Context | `context_add_norm_1` | `ElementWise` | attention 前 add/norm | 单独 op |
| Context | `context_mla_block` | `FallbackOp` | 优先 MLA module，失败则走细粒度 fallback | 组合级 |
| Context | `context_add_norm_2` | `ElementWise` | FFN/MoE 前 add/norm | 单独 op |
| Context | `context_shared_gate_up_gemm` | `GEMM` | shared expert fused gate/up | submodule-level |
| Context | `context_shared_act_gate` | `ElementWise` | shared expert 激活/门控 | submodule-level |
| Context | `context_shared_ffn2_gemm` | `GEMM` | shared expert down proj | submodule-level |
| Context | `context_router_gemm` | `GEMM` | router 打分 | submodule-level |
| Context | `context_moe_pre_dispatch` | `MoEDispatch` | MoE 发出通信 | op-level comm |
| Context | `context_moe` | `MoE` | MoE 计算 | op-level compute |
| Context | `context_moe_post_dispatch` | `MoEDispatch` | MoE 汇聚通信 | op-level comm |
| Context | `context_logits_gemm` | `GEMM` | logits 输出头 | 单独 op |
| Context | `context_p2p` | `P2P` | pipeline parallel 传输 | 单独 op |
| Generation | `generation_embedding` | `Embedding` | 解码 embedding | 单独 op |
| Generation | `generation_add_norm_1` | `ElementWise` | attention 前 add/norm | 单独 op |
| Generation | `generation_mla_block` | `FallbackOp` | 优先 MLA module，失败则走细粒度 fallback | 组合级 |
| Generation | `generation_add_norm_2` | `ElementWise` | FFN/MoE 前 add/norm | 单独 op |
| Generation | `generation_moe_overlap` | `OverlapOp` | shared/routed 并行 MoE | 组合级 |
| Generation | `generation_logits_gemm` | `GEMM` | 解码 logits 输出头 | 单独 op |
| Generation | `generation_p2p` | `P2P` | pipeline parallel 传输 | 单独 op |

### `context_mla_block` fallback 内部细项

| 子节点 | Ops 类 | 作用 |
| --- | --- | --- |
| `context_downscale_gemm` | `GEMM` | latent 压缩/下采样投影 |
| `context_q_b_proj_gemm` | `GEMM` | Q 路径投影 |
| `context_kv_b_proj_gemm` | `GEMM` | KV 路径投影 |
| `context_attention` | `ContextMLA` 或 `ContextAttention` | MLA attention 核心 |
| `context_proj_gemm` | `GEMM` | 输出投影 |

### `generation_mla_block` fallback 内部细项

| 子节点 | Ops 类 | 作用 |
| --- | --- | --- |
| `generation_downscale_gemm` | `GEMM` | latent 压缩/下采样投影 |
| `generation_q_b_proj_gemm` | `GEMM` | Q 路径投影 |
| `generation_bmm_pre` | `MLABmm` | MLA pre BMM |
| `generation_attention` | `GenerationMLA` 或 `GenerationAttention` | MLA decode 核心 |
| `generation_bmm_post` | `MLABmm` | MLA post BMM |
| `generation_proj_gemm` | `GEMM` | 输出投影 |

---

## 7. WideEP 变体差异

## 7.1 TRT-LLM WideEP 版

`TrtllmWideEPDeepSeekModel` 的 attention 不再使用 `FallbackOp + MLAModule` 这套优先回退逻辑，而是显式写出较长的 MLA 计算链：

- `context_downscale_gemm`
- `context_q_a_layernorm`
- `context_q_b_proj_gemm`
- `context_kv_b_proj_gemm`
- `ContextMLA`
- `context_proj_gemm`

generation 侧还显式出现：

- `generation_bmm_rope_overlap`
- `GenerationMLA`
- `generation_bmm_post`

说明 TRT-LLM WideEP 版为了配合其实际执行路径，对 attention 内部拆得更开。

同时 MoE 路径变成：

- `TrtLLMWideEPMoEDispatch`
- `TrtLLMWideEPMoE`

并引入：

- `context_moe_reduce_add`
- `generation_moe_reduce_add`

以及 `wideep_num_slots`、`enable_eplb` 等 WideEP/EPLB 约束。

## 7.2 SGLang WideEP / DeepEP 版

`WideEPDeepSeekModel` 中，attention 主要围绕：

- TP all_gather / reduce_scatter
- `WideEPContextMLA` / `WideEPGenerationMLA`

以及额外的 qkv_a 投影构造。

MoE 路径仍然使用：

- `MoEDispatch`
- `MoE`

但会把：

- `moe_backend`
- `sms`
- `enable_eplb`

这些 DeepEP / WideEP 运行时因素带入查询。

因此若你后面在 SGLang H100 上做实机对比，必须区分：

- 普通 DeepSeekV3 图纸
- SGLang WideEP / DeepEP 图纸

它们的 attention 和 MoE 通信图谱明显不同。

---

## 8. 性能数据文件与采集来源

## 8.1 Attention 相关 perf 表

DeepSeek-V3 主要会命中的文件有两层。

module-level：

- `mla_context_module_perf.txt`
- `mla_generation_module_perf.txt`

granular fallback：

- `context_mla_perf.txt`
- `generation_mla_perf.txt`
- 某些 backend 下也可能走 `context_attention_perf.txt`
- 某些 decode 路径还会用到 `mla_bmm` 相关表

这正对应了它“module 优先、细粒度 fallback 兜底”的建模逻辑。

## 8.2 MoE 相关 perf 表

普通 MoE：

- `moe_perf.txt`

SGLang WideEP / DeepEP：

- `wideep_context_mla_perf.txt`
- `wideep_generation_mla_perf.txt`
- `wideep_context_moe_perf.txt`
- `wideep_generation_moe_perf.txt`

TRT-LLM WideEP：

- `wideep_moe_compute_perf.txt`
- `trtllm_alltoall_perf.txt`

## 8.3 采集脚本

和 DeepSeek-V3/MLA 直接相关的 collector 主要包括：

- `collector/sglang/collect_mla.py`
- `collector/sglang/collect_mla_module.py`
- `collector/sglang/collect_mla_bmm.py`
- `collector/trtllm/collect_mla_v1.py`
- `collector/trtllm/collect_mla_v2.py`
- `collector/trtllm/collect_mla_module.py`
- `collector/trtllm/collect_mla_bmm.py`
- `collector/vllm/collect_mla_module_v1.py`
- `collector/vllm/collect_mla_module_v2.py`

MoE / WideEP 相关采集主要包括：

- `collector/sglang/collect_moe.py`
- `collector/sglang/collect_wideep_deepep_moe.py`
- `collector/trtllm/collect_moe_v1.py`
- `collector/trtllm/collect_moe_v2.py`
- `collector/trtllm/collect_moe_v3.py`
- `collector/trtllm/collect_wideep_moe_compute.py`
- `collector/slurm_comm_collector/collect_trtllm_alltoall.py`

---

## 9. 对“实机校准任务”的直接意义

如果你现在聚焦 H100 上 DeepSeek-V3 的 MLA 模块校准，建议优先分成三层。

第一层：确认模型到底走哪张图纸

- 普通版
- TRT-LLM WideEP 版
- SGLang WideEP / DeepEP 版

第二层：确认 attention 走的是哪条建模路径

- `MLAModule` primary
- 还是 fallback 的细粒度 MLA 子算子

第三层：把 MoE 误差和 MLA 误差分开分析

因为在 V3 中：

- attention 路径可能因为 module/fallback 切换而产生结构性差异
- MoE 路径则更多体现通信和 overlap 建模误差

所以如果你后面做 op breakdown，对 `context_mla_block` / `generation_mla_block` 这种组合节点要特别留意，最好进一步展开到 fallback 子项级别再对比实机 trace。

---

## 10. 一句话总结

DeepSeek-V3 在 AIC 中是一套“MLA module 优先、细粒度 fallback 兜底、MoE 路径继续细拆并在 generation 期做 overlap”的混合建模体系。它比 DeepSeek-V3.2 更保留 attention 内部结构，因此也更适合做 MLA 子模块级别的实机对齐与误差溯源。
