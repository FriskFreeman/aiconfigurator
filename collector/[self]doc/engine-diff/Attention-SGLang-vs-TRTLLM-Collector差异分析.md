# Attention Collector：SGLang 与 TensorRT-LLM 差异分析

> 分析日期：2026-07-27
>
> 分析对象：`collector/sglang/collect_attn.py`、`collector/trtllm/collect_attn.py`、registry、SDK attention loader/query，以及仓库现有 `context_attention_perf.txt`、`generation_attention_perf.txt`。
>
> 横轴固定为 SGLang latency，纵轴固定为 TensorRT-LLM latency；每个精确匹配 attention shape 是一个散点，不采样、不聚合。

## 1. 结论先行

1. 同名 `attention_context` / `attention_generation` 并不表示相同前端工作。SGLang 测 `RadixAttention(q,k,v,ForwardBatch)`；TRT-LLM 测 `_torch create_attention(backend_name="TRTLLM").forward(concatenated_qkv, metadata)`。后者还配置了 GPT-NeoX RoPE，SGLang collector 则直接提供已分离的 Q/K/V，没有在 collector 中配置 RoPE。
2. 两端的计时闭包都只包 attention forward，metadata、cache 分配和历史 cache 准备在计时外；但 forward 内包含当前 token/full context 的 KV cache 写入以及 attention 主干。因此测到的不是“只读 KV 的纯 FMHA”。
3. TRT-LLM 的 FP8 KV 路径以 BF16 `input_qkv` 进入被计时的 `attn.forward`，而 KV cache 物理 dtype 为 FP8，所以本步 K/V 的 FP8 转换/量化和 cache 写入必然属于 forward 语义，也在计时边界内。是否表现为独立 quant kernel，还是融合在 FMHA/XQA kernel 中，必须以 Nsight/CUPTI trace 确认。
4. 显式 FP8 context FMHA 的边界仍不对称：SGLang 在计时前把 Q/K/V `.to(float8_e4m3fn)`；TRT-LLM 保持 BF16 `input_qkv`，以 `QuantConfig(quant_algo=FP8)` 和 `out_scale` 进入计时 forward。因此相同 `attn_dtype=fp8,kv_cache_dtype=fp8` 不是完全相同输入边界的纯 kernel A/B。
5. `attn_dtype=bfloat16,kv_cache_dtype=fp8` 也不能简单理解成“BF16 FMHA + 仅 FP8 存储”。SGLang 的 FlashAttention backend 会在 `layer(...)` 内按显式 KV dtype 做 Q/K/V 对齐；这些 cast 在计时内，核心 attention kernel 也可能采用 FP8 模板。CSV 的 `attn_dtype` 描述 collector 入参模式，不足以还原实际 kernel dtype。
6. 当前 case 范围不同。SGLang 仅 head_dim 128/256、window=0；TRT-LLM 还采 head_dim=64，并在 head_dim=64 的部分 context/XQA generation case 加 window=128。TRT-LLM 从 SM89 就采 FP8，SGLang 从 SM90 开始，故 L40S 只有 TRT-LLM FP8 数据。
7. backend 随硬件变化。现有 SGLang 数据记录 A100/L40S/H100/H200 为 `flash_attention`，B200/B300/GB200/GB300 为 `trtllm_mha`，RTX PRO 6000 为 `triton`；TRT-LLM 全部只记录 `torch_flow`。这些字段是 Python backend 标签，不是 CUDA kernel symbol。
8. 数据结果不是某框架全面更快。Context 多数数据中心 GPU 上 SGLang 更快；generation 在 H100/H200 的 FP8 KV 子集上 TRT-LLM 明显更快，在 B200/B300/GB200/GB300 上则 SGLang 明显更快；RTX PRO 6000 又接近或发生反转。
9. 仓库中的同版本目录不代表同一代网格。SGLang `0.5.10` 的 H100/H200/B300/GB 数据只有 head_dim=128，B200/RTX 同目录则还有 256；TRT-LLM 的版本目录也存在部分组合缺行。做统计时必须以文件实际键集合为准。

## 2. 入口、输出与数据库语义

| 项目 | SGLang | TensorRT-LLM |
|---|---|---|
| Registry op | `attention_context` / `attention_generation` | 同名 |
| 模块 | `collector.sglang.collect_attn` | `collector.trtllm.collect_attn` |
| 核心入口 | `RadixAttention(...)` | `create_attention(backend_name="TRTLLM",...)` |
| Context 输出 | `context_attention_perf.txt` | 同名 |
| Generation 输出 | `generation_attention_perf.txt` | 同名 |
| 兼容声明 | `sglang>=0.5.10rc0` | 当前脚本无 `__compat__` |
| 计时 | warmup=3，runs=20 | forward dry-run 一次；warmup=10，runs=6 |
| CUDA Graph | `benchmark_with_power` 默认启用 | 同左 |
| kernel 日志 | `flash_attention` / `trtllm_mha` / `triton` | 固定 `torch_flow` |

SDK 的真实索引语义决定了比较键：

- Context：`attn_dtype,kv_cache_dtype,num_key_value_heads,head_dim,window_size,num_heads,isl,batch_size`。
- Generation：`kv_cache_dtype,num_key_value_heads,head_dim,window_size,num_heads,total_seq_len,batch_size`；`attn_dtype` 被读取但不进入索引，`total_seq_len=isl+step`。
- MHA 在 loader 内把 `num_key_value_heads == num_heads` 归一成内部 `kv_n=0`；精确作图仍保留原始 Hkv，以避免把不同外部 shape 合并。
- SGLang CSV 没有 `window_size`，SDK 兼容逻辑按 0 处理，本次脚本相同。
- 若查询键重复，SDK first-wins；本次选定 36 个 phase/platform 文件没有重复查询键。

所以 generation 作图不使用 `attn_dtype` 分图，只按 BF16/FP8 KV cache 分两图。这不是遗漏，而是 SDK 和当前两端 collector 的实际语义：decode 没有独立 FP8 attention compute case。

## 3. 被测执行链与计时边界

### 3.1 SGLang

Context 构造：

```text
Q [B*S,H,D] BF16
K/V [B*S,Hkv,D] BF16
ReqToTokenPool + MHATokenToKVPool + ForwardBatch(EXTEND)
metadata init                                      <- 计时外

若 explicit FP8 context FMHA: Q/K/V -> E4M3       <- 计时外

RadixAttention(Q,K,V,ForwardBatch)                 <- 计时内
  backend dtype 对齐/转换（若 backend 需要）
  写当前 K/V 到 paged KV cache
  context/prefill attention 主干
```

Generation 构造：

```text
Q/K/V [B,heads,D] BF16
历史 K/V 随机初始化并写入 pool                     <- 计时外
metadata init                                      <- 计时外

RadixAttention(Q,K,V,ForwardBatch(DECODE))         <- 计时内
  当前 K/V dtype 转换及 cache 写入
  读取历史 cache
  decode attention 主干
```

关键细节：

- `use_fp8_kv_cache=True,use_fp8_context_fmha=False` 时，live Q/K/V 仍以 BF16 进入 `layer(...)`。若 backend 在内部为 FP8 cache/kernel 做 cast，该转换在计时内。
- `use_fp8_context_fmha=True` 时，collector 在定义 `run_iter()` 前已经把 Q/K/V 转成 E4M3；显式 `.to()` 不计时。
- Context 每次 replay 会向相同 cache location 重写整段 K/V；generation 每次重写相同的“新 token”位置。结果值不校验。
- SGLang generation 的历史 cache 被真实随机数据填充；这点与 TRT-LLM collector 不同。

### 3.2 TensorRT-LLM

共同构造：

```text
create_attention(backend="TRTLLM", RoPE=GPT-NeoX, QuantConfig)
KVCacheManager(dtype=BF16/FP8, tokens_per_block=64)
TrtllmAttentionMetadata.prepare()                  <- 计时外
Q BF16 + KV BF16 -> concatenated input_qkv BF16    <- 计时外
一次 attn.forward dry-run                          <- 计时外

attn.forward(input_qkv, metadata, out_scale=...)   <- 计时内
  QKV/RoPE/runtime 前处理（具体融合方式由版本决定）
  本步 K/V 转换并写入 KV cache
  context FMHA 或 generation XQA/MHA 主干
```

Context 显式设置 `TRTLLM_ENABLE_XQA_JIT=0`；generation 设置为 1。Context 的 `num_contexts=B`，generation 的 `num_contexts=0`。

Generation 只通过 `KVCacheManager.add_dummy_requests()` 分配请求和 block，并把 `num_cached_tokens_per_seq=input_len` 写入 metadata；collector 没有像 SGLang 那样显式随机填充全部历史 K/V。一次 dry-run 和后续 warmup 会写本步位置，但历史 block 内容主要用于驱动访存/算子，不用于数值验证。

### 3.3 FP8 是否同时包含量化和 attention 主干

答案分配置看：

| 配置 | SGLang 计时边界 | TRT-LLM 计时边界 |
|---|---|---|
| Context BF16/BF16 | backend forward + KV 写入 + FMHA | `attn.forward` + KV 写入 + FMHA |
| Context BF16/FP8 | BF16 Q/K/V 进入计时；backend 内 dtype 对齐/FP8 cache 写入 + attention 均计时 | BF16 QKV 进入计时；FP8 KV 转换/写入 + attention 均属于 `attn.forward` |
| Context FP8/FP8 | Q/K/V 显式 `.to(FP8)` 在计时外；计时含 cache 写入 + FP8 attention | BF16 QKV 进入计时；`QuantAlgo.FP8`、FP8 cache 与 attention 均在 `attn.forward` 路径内 |
| Generation FP8 KV | BF16 本步 Q/K/V 进入计时；cache 转换/写入 + decode 主干 | BF16 input_qkv 进入计时；FP8 cache 转换/写入 + XQA/MHA 主干 |

因此，对问题“TRT-LLM 按 collect 脚本执行时是否会同时包含量化和 GEMM/attention 主干”，attention 的答案是：**从调用边界和 dtype 必要条件看，会同时包含本步 BF16 K/V 到 FP8 cache 所需的转换/量化语义和 attention 主干；显式 FP8 context 模式还把 FP8 quant config 放在同一个 forward 内。** 但不能仅凭 Python 调用断言量化一定是一个独立 CUDA kernel；TRT-LLM 可能将其与 QKV preprocessing、cache update 或 FMHA 融合。CSV 的 `kernel_source=torch_flow` 也无法区分，需要 profiler 给出最终 kernel 序列。

这与 SGLang 显式 FP8 context case 有实质差异：SGLang collector 的 Q/K/V `.to(FP8)` 已在 timer 外，TRT-LLM 的输入仍是 BF16。

## 4. 硬件后端与 kernel 归属

### 4.1 SGLang 的静态选择

| SM / 现有硬件 | collector backend | 预期主干类别 | 现有 CSV `kernel_source` |
|---|---|---|---|
| SM80 A100 | `FlashAttentionBackend` | FlashAttention 的 context/decode 路径 | `flash_attention` |
| SM89 L40S | `FlashAttentionBackend` | FlashAttention 路径 | `flash_attention` |
| SM90 H100/H200 | `FlashAttentionBackend` | FlashAttention/FA3 能力范围内的 FMHA | `flash_attention` |
| SM100/103 B200/B300/GB200/GB300 | 优先 `TRTLLMHAAttnBackend`，ImportError 才回退 FlashAttention | SGLang wrapper 下的 TRT-LLM MHA/FMHA 路径 | 现有数据均为 `trtllm_mha` |
| SM>=110，现有为 SM120 RTX PRO 6000 | `TritonAttnBackend` | Triton extend/decode attention | `triton` |

`kernel_source` 只能证明 Python backend 选择，不能证明每个 shape 的 CUDA kernel 模板、tile、split-K 或 fallback。尤其 `flash_attention` 没有区分 FA2/FA3，`trtllm_mha` 也没有记录内部 runner。

### 4.2 TRT-LLM 的静态选择

所有硬件都通过 `create_attention(backend_name="TRTLLM")`，并统一日志为 `torch_flow`：

- Context 由 TRT-LLM attention runtime 走 context FMHA 类路径，关闭 XQA JIT。
- Generation 打开 XQA JIT；GQA/MQA shape 会进入 XQA 能力范围，MHA 或不支持 shape 可由 runtime 选择其他 MHA/decode kernel。
- BF16/FP8 context FMHA 由 `quant_algo` 区分；KV cache dtype 由 `kv_cache_quant_algo` 和 `KVCacheManager.dtype` 区分。
- collector 不记录具体 kernel symbol、runner、tile、JIT/fallback 结果。因此本文不把 `torch_flow` 当作一个 CUDA kernel 名。

精确回答“某 shape 到底用了哪个 kernel”需要新增 Nsight Systems/Compute 或 CUPTI 抽样，并至少记录 CUDA symbol、dtype 模板、XQA JIT 状态和 fallback 原因。

## 5. 当前脚本采集范围

### 5.1 Context 公共维度

两边公共列表：

- B：`1,2,4,8,16,32,64,128,256`
- S：`1,16,32,64,128,256,512,1024,1536,2048,3072,4096,6144,8192,10240,12288,16384,262144`
- H：`1,2,4,8,12,16,24,32,40,48,64,96`
- Hkv label：`0,1,2,4,8`，其中 0 展开为 MHA，即 Hkv=H；GQA 要求 `Hkv < H` 且整除。
- MHA 限制 `B*S<=65536,B<=128`；GQA 限制 `B*S<=131072`。
- 共同避免 `B*S*Hkv*D*2 >= 2^31`。

差异：

| 维度/条件 | SGLang | TensorRT-LLM |
|---|---|---|
| head_dim | 128, 256 | 64, 128, 256 |
| window | 固定全 attention，CSV 缺列，按 0 | D=64 额外采 window=128；其他为 0 |
| FP8 最低架构 | SM90 | SM>86，即 SM89 L40S |
| SM100+ Q/Hkv filter | 无对应生成器过滤 | 若 H/Hkv>=32 且不能被 32 整除则跳过 |
| SM120 大 Q/O | `B*S*H*D >= 2^31` 跳过 | 无同一通用过滤 |
| SM120 FP8 context 崩溃区 | 无 TRT 规则 | 1.3.0rc5/rc10 按 H/Hkv/D/tokens 跳过已验证 crash 区域 |

每个有效 SGLang shape 生成：BF16/BF16；SM90+ 再生成 BF16/FP8 和 FP8/FP8。TRT-LLM 语义相同，但从 SM89 开始生成 FP8。

### 5.2 Generation 公共维度

- B 基本为 `1,2,4,8,16,32,64,128,256,512,1024,2048`。SGLang 源列表重复写了 64，但后续使用 set 汇总，不会生成重复 case。
- step 候选由 `2..131072` 的 2 倍序列减 1得到，并额外保留每个 B 的边界点。
- MHA H：`1,2,4,8,12,16,24,32,40,48,64`。
- XQA/GQA H：`1,2,4,8,16,32,64,96,128`，Hkv=`1,2,4,8` 且 Hkv<H。
- 当前 generation 只有 `attn_dtype=bfloat16`；分别采 BF16 KV 和 FP8 KV。

差异：

- SGLang D=128/256，TRT-LLM D=64/128/256。
- TRT-LLM 在 D=64 的 XQA case 同时采 window=0/128；SGLang 仅 window=0。
- SGLang 的 MHA `max_b` 按 `128/head_dim` 缩放；TRT-LLM 的初始 `max_b` 不按 D 缩放，D=256 的理论高负载范围更宽。
- TRT-LLM SM100+ 对 H/Hkv ratio 使用与 context 相同的 32-head 过滤；SGLang 无生成器侧等价过滤。

## 6. 已落盘数据范围审计

本次固定比较：

- SGLang：全部平台优先 `0.5.10`。
- TRT-LLM：A100/L40S 为仓库最新可比 `1.0.0`；其余为 `1.3.0rc10`。
- 平台：A100 SXM、L40S、H100 SXM、H200 SXM、B200 SXM、B300 SXM、GB200、GB300、RTX PRO 6000，共 9 个小图。

实际文件暴露出以下版本漂移：

1. H100/H200 的 SGLang `0.5.10` 只有 D=128；B300/GB200/GB300 也只有 D=128；B200 与 RTX PRO 6000 有 D=128/256。
2. TRT-LLM 1.3.0rc10 的数据中心 Blackwell 文件包含 D=64/128/256；H100/H200 文件只到 D=128，尽管当前脚本已经列出 256。
3. H100 TRT-LLM FP8 generation 只有 9,381 行，而 BF16 有 15,057 行；按共同键只能匹配 2,872 个 FP8 shape，明显少于 BF16 的 5,093。
4. B200 SGLang context 每类配置 11,238 个 unique shape，接近当前 D=128/256 网格但并非理论完整；RTX SM120 每类 11,174，受额外 int32 边界过滤。
5. 选定文件中没有重复 query key，SGLang generation 源码中的重复 B=64 没有落成重复 CSV 行。

因此，图中的 `SGL only` / `TRT only` 同时反映脚本维度差异、硬件过滤、崩溃 skip 和历史采集完整度，不能全部解释成某一端“不支持”。

## 7. 延迟结果

比值定义为 `TRT-LLM latency / SGLang latency`。中位数小于 1 表示 TRT-LLM 更快，大于 1 表示 SGLang 更快。

| 配置 | 总匹配点 | 主要结果 |
|---|---:|---|
| Context BF16/BF16 | 57,720 | A100 到数据中心 Blackwell 中位 TRT/SGL=1.317~1.448，SGLang 更快；RTX 为 0.898 |
| Context BF16-tag/FP8-KV | 47,988 | H100 到 GB300 中位 1.460~1.569，SGLang 更快；RTX 为 0.681，TRT-LLM 更快 |
| Context FP8/FP8 | 47,965 | H100 到 GB300 中位 1.305~1.434；RTX 反而为 1.712，SGLang 更快 |
| Generation BF16 KV | 53,282 | A100/L40S 为 0.811/0.646；H100/H200 约 1；数据中心 Blackwell 为 1.640~1.774；RTX 为 0.975 |
| Generation FP8 KV | 39,740 | H100/H200 子集为 0.631/0.620；数据中心 Blackwell 为 1.863~2.052；RTX 为 1.024 |

完整 45 行的平台统计见 `attention-data-compare/comparison_findings.md` 和 `results/matched_shape_summary.csv`。

需要特别避免以下错误解读：

- 不能用未匹配 shape 比延迟；散点只使用 exact inner join。
- shape 网格不是模型调用频率分布，中位数不是端到端模型收益。
- H100/H200 FP8 generation 是缩小子集，不能和 BF16 总体直接做“量化收益”差分。
- SGLang B200/GB200 的 `trtllm_mha` 说明两者甚至可能共享 TRT-LLM 底层能力，但 wrapper、QKV layout、RoPE、metadata、cache 初始化和版本仍不同。
- SM120 的 SGLang Triton 与 TRT-LLM runtime 是完全不同后端，BF16 趋势不能外推到显式 FP8。

## 8. 图表和复现

工作目录：`collector/[self]doc/engine-diff/attention-data-compare/`

- `plot_attention_engine_comparison.py`：读取 18 组 phase/framework 文件，按 SDK first-wins 语义去重并 exact join。
- `results/context_bfloat16_kv_bfloat16.{png,pdf}`
- `results/context_bfloat16_kv_fp8.{png,pdf}`
- `results/context_fp8_kv_fp8.{png,pdf}`
- `results/generation_kv_bfloat16.{png,pdf}`
- `results/generation_kv_fp8.{png,pdf}`
- `results/matched_shape_summary.csv`：原始行、unique shape、duplicate、matched、单边 shape、ratio P10/P50/P90 和 TRT faster 占比。

运行：

```bash
python collector/'[self]doc'/engine-diff/attention-data-compare/plot_attention_engine_comparison.py
```

脚本在每个 subplot 都断言绘制点数等于 exact matched shape 数；PDF 的 scatter 被 rasterize 以控制体积，但没有减少数据点。

## 9. 后续采集建议

1. CSV 增加 `actual_kernel_source`、CUDA kernel symbol/runner、FA2/FA3/XQA/Triton 细分、是否 fallback。
2. 增加 `input_qkv_dtype`、`effective_fmha_dtype`、KV scale dtype/粒度、RoPE enabled、window、tokens_per_block。
3. 记录 `used_cuda_graph`、warmup/runs、collector commit、容器/CUDA/PyTorch/框架版本。
4. 明确 latency 是否包含 QKV cast、RoPE、KV update；最好同时输出 `forward_total` 和 profiler 拆分。
5. 对 generation 初始化真实历史 KV，并增加结果有限值/基本正确性检查；至少区分“仅性能占位 cache”和“有效内容 cache”。
6. 统一 D=64/128/256 和 window 网格后再做覆盖率比较；对 crash skip 输出结构化 skip manifest。
7. 对 H100/H200 FP8 generation 补齐缺失 shape，再比较精度类别总体趋势。

## 10. 分析限制

当前环境没有安装 `tensorrt_llm` Python 包，且在线 GitHub 源码查询发生 DNS 解析失败；因此 TRT-LLM 内部是否将 FP8 quant/cache update 融合为一个或多个 CUDA kernel，本文没有做超出证据的断言。可以确定的是 collector 的 BF16 `input_qkv`、FP8 cache/quant config 和单一被计时 `attn.forward` 调用边界；具体 kernel 序列必须用对应 1.0.0/1.3.0rc10 容器或 Nsight trace补证。
