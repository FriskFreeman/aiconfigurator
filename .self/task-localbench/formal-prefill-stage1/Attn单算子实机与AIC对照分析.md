# Attn单算子实机与AIC对照分析

## 背景与结论

当前这轮对比已经可以先得到一个比较明确的判断：

- 对于 **SGLang 0.5.9 的 DeepSeek V3 实机执行**，当前观测到的 **prefill attention** 更接近常规 `attention` 路径，而不是 `mla` 路径。
- 对于 **decode attention**，当前观测到的则更接近 `mla` 路径，而不是常规 `attention` 路径。
- 从 AIC SDK 当前的 DeepSeek V3 组网逻辑看，**decode 侧走 `GenerationMLA` 是合理的**；但 **context/prefill 侧在非 vLLM 后端下直接走 `ContextMLA`，与当前 SGLang 0.5.9 + H100 实机行为并不一致**，很可能是现阶段的主要口径偏差来源之一。
- 进一步看 collector 代码，`collect_attn + query_attention` 当前不能精确表达 DeepSeek prefill 的 `q/k head_dim=192, v head_dim=128` 异维形状；`collect_mla` 的 prefill 虽然也可能走 FA3，但它直接进入 MLA 形态 `RadixAttention`，绕过 DeepSeek 的 MHA dispatch，采到的是 absorbed MLA latent attention，而不是常见实机 prefill 的 `attn_mha`。

这不是“实机和仿真谁错了”的简单问题，而是两边当前采用的 **attention 语义边界不同**：

- 实机侧：prefill 和 decode 会按 SGLang 的 runtime dispatch 逻辑，落到不同 attention 实现分支。
- 仿真侧：collector 和 SDK 分别维护了 `attention` 与 `mla` 两套性能数据与查表路径；但模型组网时是否选对了路径，还要结合具体框架后端与阶段判断。

## 一、实机侧：SGLang 0.5.9 的 attention 实际执行行为

本节以用户指定的 **docker 镜像 `booleimg.myaddr.io/lmsysorg/sglang:v0.5.9`** 中源码为准。

### 1. DeepSeek V3 注意力模块内部同时存在 `attn_mqa` 与 `attn_mha`

在 docker 源码文件 `sglang/srt/models/deepseek_v2.py` 中，`DeepseekV2AttentionMLA` 同时构造了两套 `RadixAttention`：

- `attn_mqa`
  - `num_kv_heads=1`
  - `head_dim = kv_lora_rank + qk_rope_head_dim`
  - `v_head_dim = kv_lora_rank`
  - 前缀名为 `attn_mqa`
- `attn_mha`
  - `num_kv_heads=self.num_local_heads`
  - `head_dim = qk_nope_head_dim + qk_rope_head_dim`
  - `v_head_dim = self.v_head_dim`
  - 前缀名为 `attn_mha`

这意味着 SGLang 在 DeepSeek V3 上本身就保留了两类 attention 表达：

- 一类是更像常规 MHA/GQA 的 `attn_mha`
- 一类是更像 DeepSeek MLA/MQA latent 路径的 `attn_mqa`

### 2. prefill / decode 的分支不是固定写死，而是 runtime dispatch

同文件后半段会先根据 `forward_batch` 和当前 backend 决定 `attn_forward_method`，然后分别进入：

- `forward_normal_prepare/core`
- `forward_normal_chunked_kv_prepare/core`
- `forward_normal_one_shot_prepare/core`
- `forward_absorb_prepare/core`

其中可以粗略理解为：

- `MHA` / `MHA_ONE_SHOT` / `MHA_CHUNKED_KV`：走常规 attention/MHA 风格路径
- `MLA`：走吸收权重后的 MLA/MQA 风格路径

也就是说，**prefill 与 decode 是否落在同一条 attention 路径上，取决于 runtime dispatch，而不是由模型结构名字直接决定**。

### 3. backend handler 明确允许 “prefill 走 MHA，decode 走 MLA”

更关键的代码在 docker 源码 `sglang/srt/models/deepseek_common/attention_backend_handler.py`。

对当前问题最重要的几个分支是：

- `handle_attention_fa3()`
  - 默认走 `_handle_attention_backend(...)`
- `_handle_attention_backend(...)`
  - 若是 `extend/prefill`，并且
    - `sum_extend_prefix_lens == 0`
    - 或者 `sum_extend_prefix_lens >= chunked_prefix_cache_threshold` 且未关闭 chunked prefix cache
  - 则返回 `MHA_ONE_SHOT` 或 `MHA_CHUNKED_KV`
  - 否则返回 MLA 子类型
- `handle_attention_triton()`
  - 若 `extend/prefill` 且 `sum(extend_prefix_lens_cpu) == 0`，则直接返回 `MHA`
  - 否则返回 MLA 子类型
- `handle_attention_trtllm_mla()`
  - `extend/prefill` 时，如果未禁用 chunked prefix cache 或前缀和为 0，则返回 `MHA_CHUNKED_KV`
  - 否则返回 MLA 子类型

这段逻辑非常关键，因为它直接说明：

- **prefill 并不必然走 MLA**
- 在常见场景下，**prefill 完全可能走 MHA 路径**
- **decode 则通常落到 MLA 子类型**

所以从 SGLang 原生设计看，“prefill 对上 attention、decode 对上 mla” 本身是合理的。

### 4. 现有实机解析结果也确实支持这个结论

现有对比结果见：

- [mla_kernel_aic_compare_with_decode__summary.md](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/analysis/mla_kernel_aic_compare_with_decode__summary.md)
- [mla_kernel_aic_compare_with_decode__comparison.csv](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/analysis/mla_kernel_aic_compare_with_decode__comparison.csv)

从 `comparison.csv` 可直接看到：

- `prefill` 行的 `attention_module` 当前都是 `attn_mha`
- `decode` 行的 `attention_module` 当前都是 `attn_mqa`

从摘要统计看：

- Prefill
  - `Attention / fp8` MAPE: `12.19%`
  - `Attention / bfloat16` MAPE: `26.12%`
  - `MLA / bfloat16` MAPE: `288.20%`
- Decode
  - `MLA / bfloat16` MAPE: `12.03%`
  - `Attention / bfloat16` MAPE: `38.04%`

这和源码 dispatch 结论是互相印证的。

## 二、AIC collector：`attention` 与 `mla` 两套采集到底在采什么

### 1. `collect_attn.py` 采的是常规 attention 路径

文件：[collect_attn.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/collector/sglang/collect_attn.py)

核心特征：

- 用的是 `MHATokenToKVPool`
- 构造的是标准 `RadixAttention(num_heads, num_kv_heads, head_dim, ...)`
- prefill 时：
  - `q` 形状是 `batch_size * input_len`
  - `k/v` 也是标准 MHA/GQA 形态
  - `extend_prefix_lens` 被设为全 0
- decode 时：
  - 先把历史 K/V 填入标准 KV cache
  - 再执行单 token generation attention

输出到 perf 文件时：

- prefill 记为 `context_attention`
- decode 记为 `generation_attention`

此外它还显式区分两类量化维度：

- `attn_dtype`
  - prefill 可选 BF16 或 FP8 FMHA
- `kv_cache_dtype`
  - KV cache 可选 BF16 / FP8

所以 `collect_attn.py` 对应的是 **常规 attention 数据表**，而不是 MLA latent attention 表。

### 2. `collect_mla.py` 采的是 MLA 路径

文件：[collect_mla.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/collector/sglang/collect_mla.py)

核心特征：

- 用的是 `MLATokenToKVPool`
- 构造的 `MockModelConfig.attention_arch = AttentionArch.MLA`
- 根据 `_select_default_mla_backend()` 自动选择 SGLang 默认 MLA backend
  - Hopper 上优先是 `fa3`
  - Blackwell 某些条件下是 `trtllm_mla`
  - 否则是 `triton`
- prefill 和 decode 两个阶段用的是两套 MLA 形状构造逻辑

其中最关键的差别在于：

- `context MLA`
  - 采的是 MLA 的 context 路径
  - 但其张量/头维构造并不是标准 MHA/GQA，而是围绕 DeepSeek latent 维度组织
- `generation MLA`
  - 采的是 decode 下的 MLA/MQA 路径
  - 会显式构造历史 latent KV，并通过 `set_mla_kv_buffer()` 写入

输出到 perf 文件时：

- prefill 记为 `mla_context`
- decode 记为 `mla_generation`

因此，AIC 这边其实本来就维护了两类单算子：

- 一类是 `attention`
- 一类是 `mla`

问题不在 collector 没有分开，而在 **SDK 组网时 context 阶段是否选对了这两条路中的哪一条**。

### 3. 更细的代码差异：`collect_attn` 当前并不能完整表达 DeepSeek prefill 的 `qk_dim != v_dim`

这里需要区分三层能力：

- **底层 SGLang attention/KV cache 实现**
- **collector 脚本暴露出来的建模方式**
- **AIC SDK 查表接口的参数化能力**

先看底层 SGLang：

- `RadixAttention` 本身支持 `v_head_dim`
- `MHATokenToKVPool` 也支持单独的 `v_head_dim`

也就是说，**SGLang 底层并不是不支持 `qk_dim != v_dim`**。例如 DeepSeek 真正的 prefill MHA 路径就是：

- `q/k` 的 head dim 是 `192`
- `v` 的 head dim 是 `128`

但 `collect_attn.py` 当前没有把这层能力暴露出来：

- `MockModelConfig` 只接受一个 `head_dim`
- `run_attention_torch()` 只接收一个 `head_dim`
- 构造 `RadixAttention(...)` 时没有传 `v_head_dim`
- 构造 `MHATokenToKVPool(...)` 时也没有传 `v_head_dim`
- `q/k/v` 张量都按同一个 `head_dim` 构造

因此，**`collect_attn` 当前实际采的是“q、k、v 同维”的标准 attention/GQA 路径**，而不是 DeepSeek prefill 中那种 `qk=192, v=128` 的异构 head 维 attention。

这点会直接传导到 SDK 查表接口：

- `ContextAttention` / `GenerationAttention` 只有一个 `head_size`
- `query_context_attention()` 和 `query_generation_attention()` 也只有一个 `head_size`
- 额外修正里同样按
  - `q_num = n * head_size`
  - `k_num = n_kv * head_size`
  - `v_num = n_kv * head_size`
 处理

所以 AIC 当前这套 `attention` 数据模型，本质上假设：

- `qk_dim == v_dim == head_size`

结论是：

- **底层 SGLang 内核能力支持 `qk_dim != v_dim`**
- **AIC 当前 `collect_attn + query_*_attention` 这条链路，尚未把这种能力完整建模出来**

### 4. `collect_mla` 虽然在 Hopper 上也可能走 FA3，但它采的不是 DeepSeek prefill 的 `attn_mha`

这是当前最容易混淆的一点。

在 Hopper 上，`collect_mla.py` 的 `_select_default_mla_backend()` 会返回 `fa3`，因此它最终也可能调用 `FlashAttentionBackend`。但这并不等价于“它采到的就是 DeepSeek 实机 prefill 的 attention”。

原因在于，`collect_mla` 和 `collect_attn` 虽然都可能走 `FlashAttentionBackend`，但它们的 **attention 语义模式** 完全不同：

- `collect_attn`
  - `MockModelConfig.attention_arch = AttentionArch.MHA`
  - `FlashAttentionBackend.use_mla = False`
  - forward 时走标准 `set_kv_buffer(...)`
  - KV cache 是标准的 `K buffer + V buffer` 双张量布局
- `collect_mla`
  - `MockModelConfig.attention_arch = AttentionArch.MLA`
  - `FlashAttentionBackend.use_mla = True`
  - forward 时走 `set_mla_kv_buffer(...)`
  - KV cache 是单个 latent KV buffer，布局是 `(kv_lora_rank + qk_rope_head_dim)`

也就是说，**两者只是共享了“FlashAttention 后端家族”这一件事，但输入张量语义、KV 存储布局、forward 分支都已经不同了**。

### 5. 在 prefill、DeepSeek 场景下，`collect_mla` 与真实 `attn_mha` 的核心差异

对照 SGLang 0.5.9 docker 中 `forward_mha.py` 可以看到，真实 DeepSeek prefill MHA 路径大致是：

1. 先得到 `q`，其 per-head `qk_head_dim = 192`
2. 对 latent KV 做 `kv_b_proj`
3. 将投影结果切成
   - `k_nope`
   - `v`
4. 把 `k_pe` 拼回去，得到
   - `k`: head dim = `192`
   - `v`: head dim = `128`
5. 最终调用 `attn_mha(q[192], k[192], v[128])`

而 `collect_mla.py` 的 prefill 路径不是这样：

- 在 Hopper/FA3 分支下，它构造的是
  - `q_nope`: `v_head_dim = kv_lora_rank = 512`
  - `q_rope`: `64`
  - `k_nope`: `512`
  - `k_rope`: `64`
  - `v = k_nope`
- 即便 `RadixAttention` 的 `head_dim_total` 记为 `576`，它本质上仍是在跑 **MLA latent 语义**
- 它没有经过真实 DeepSeek prefill MHA 中那一步关键的 `kv_b_proj -> (k_nope, v)` 展开
- 也没有形成真实 `attn_mha` 的 `q/k=192, v=128` 这一组张量

所以从代码层面讲：

- `collect_mla` prefill 更像是“用 FA/FA3 内核执行 MLA 语义的 latent attention”
- 真实 DeepSeek prefill `attn_mha` 则是“先把 latent KV 展开成按 head 的 K/V，再执行常规 MHA”

两者即便最后都落在 FlashAttention 家族 kernel，上层输入问题已经不是同一个问题了。

### 6. 因而，为什么 `mla` prefill 数据未必能与当前实机 prefill 对上

如果只看“kernel source 都是 FA3/FlashAttention”，会误以为：

- `collect_mla` 的 prefill 数据应该天然能对上实机

但从上面的代码差异看，真正决定是否能对上的不是“是不是 FA3”，而是：

- 输入张量是不是同一语义
- KV cache 是 latent 还是标准 K/V
- 有没有经过 `kv_b_proj` 展开
- `qk_dim` 和 `v_dim` 是否与真实路径一致

因此对当前 formal case，更准确的判断是：

- **实机 prefill `attn_mha` 更接近 `attention` 语义**
- **`collect_mla` 的 prefill 虽也可能使用 FA 内核，但它采到的是 MLA latent 路径，不是当前实机 prefill 的那条路径**

### 7. 进一步深挖：`collect_mla` 直接进 `RadixAttention`，绕过了 DeepSeek 的 prefill dispatch

这轮重点确认了一个比“kernel source 是不是 FA3”更关键的差别：

- `collect_mla.py` 不是跑完整 DeepSeek attention module。
- 它直接构造一个 `RadixAttention(attention_arch=MLA)`，然后调用 `layer(q, k, v, forward_batch, q_rope=..., k_rope=...)`。
- 因此它绕过了 SGLang DeepSeek 模型层中的 `dispatch_attn_forward_method()`、`forward_normal_prepare()` 和 `forward_normal_core()`。

这会影响 prefill 的真实分支选择。

在 SGLang 0.5.9 docker 源码中，DeepSeek attention 的真实入口会先构造两套 attention：

- `attn_mqa`
  - `head_dim = kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576`
  - `num_kv_heads = 1`
  - `v_head_dim = kv_lora_rank = 512`
- `attn_mha`
  - `head_dim = qk_nope_head_dim + qk_rope_head_dim = 128 + 64 = 192`
  - `num_kv_heads = num_local_heads`
  - `v_head_dim = 128`

真实 prefill 在常见无 prefix 或满足 chunked-prefix 条件时，会通过 `attention_backend_handler.py` 返回 `MHA` / `MHA_ONE_SHOT` / `MHA_CHUNKED_KV`，然后进入 `forward_mha.py`。

`forward_mha.py` 的真实数据准备是：

- 从 latent cache 中取 `kv_a` 和 `k_pe`。
- 执行 `kv_b_proj(kv_a)`。
- 将结果 reshape 成 `[tokens, local_heads, qk_nope_head_dim + v_head_dim]`。
- 切成 `k_nope[128]` 和 `v[128]`。
- 将 `k_nope[128] + k_pe[64]` 拼成 `k[192]`。
- 最后调用 `attn_mha(q[192], k[192], v[128], save_kv_cache=False)`。

而 `collect_mla.py` 的 context 路径不是这个过程。它直接构造：

- `q_nope`: `[tokens, local_heads, 512]`
- `q_rope`: `[tokens, local_heads, 64]`
- `k_nope`: `[tokens, 1, 512]`
- `k_rope`: `[tokens, 1, 64]`
- `v = k_nope`

然后调用 MLA 形态的 `RadixAttention`。因为它没有经过 DeepSeek 模型层的 MHA dispatch，所以 `ForwardBatch` 上不会像真实 `forward_normal_*` 那样设置 `attn_attend_prefix_cache`、`mha_one_shot`、`mha_return_lse` 等状态。

在 `FlashAttentionBackend.forward_extend()` 中，这一点会改变实际执行分支：

- 若 `use_mla=False`，走普通 MHA cache 路径。
- 若 `use_mla=True` 且 `forward_batch.attn_attend_prefix_cache is not None`，走 “MLA 模型中的 MHA/chunked-prefix MHA” 路径。
- 若 `use_mla=True` 且 `forward_batch.attn_attend_prefix_cache is None`，走 absorbed MLA 路径，调用 `flash_attn_with_kvcache(q=q_rope, qv=q_nope, k_cache=k_rope_cache, v_cache=c_kv_cache, ...)`。

`collect_mla.py` 属于最后一种。因此它的 context 数据虽然 `kernel_source=flash_attention`，但其语义是 **absorbed MLA latent attention**，不是 SGLang 完整模型 prefill 常见的 `attn_mha`。

### 8. `collect_attn` / `query_attention` 对 qk/v 异维的能力边界

`collect_attn.py` 当前没有完整表达 DeepSeek prefill 的 `qk_dim != v_dim`。

代码证据：

- `MockModelConfig` 只有一个 `head_dim`。
- `run_attention_torch()` 只接受一个 `head_dim`。
- `MHATokenToKVPool(...)` 只传入 `head_dim=head_dim`，没有传 `v_head_dim`。
- `RadixAttention(...)` 也没有传 `v_head_dim`，所以默认 `v_head_dim=head_dim`。
- prefill 构造的 `q/k/v` 都是 `[tokens, heads, head_dim]`。

对应到数据文件，H100 + SGLang 0.5.9 的 `context_attention_perf.txt` 当前只有：

- `head_dim=128`
- `num_heads / num_key_value_heads`
- `attn_dtype`
- `kv_cache_dtype`

没有独立的 `qk_head_dim` 和 `v_head_dim` 轴。

SDK 查询侧也同样如此：

- `ContextAttention` 只有 `head_size`。
- `query_context_attention()` 只有 `head_size`。
- SOL 公式里用同一个 `h` 同时估算 Q/K/V。
- 额外 rope/KV write 修正也按 `q_num=n*head_size`、`k_num=n_kv*head_size`、`v_num=n_kv*head_size` 处理。

所以这条链路可以处理标准 MHA/GQA，也可以通过 `num_key_value_heads` 表达 MHA/GQA/XQA 的头数差异，但不能精确表达 DeepSeek prefill 的：

- `q/k head_dim = 192`
- `v head_dim = 128`

这解释了为什么当前实机 prefill 与 `context_attention` “趋势和量级能对上”，但仍存在一层结构性近似误差：它对上的是更接近的 MHA 语义，而不是完全同形状的 DeepSeek MHA。

### 9. `collect_mla` 与 `collect_attn` 在 DeepSeek prefill 下的核心差异

两者在 H100 上都可能显示 `kernel_source=flash_attention`，但差异不在 backend 名字，而在输入语义和 cache 布局。

`collect_attn.py`：

- `attention_arch=MHA`
- `use_mla=False`
- KV pool 是 `MHATokenToKVPool`
- K/V cache 是标准 `K buffer + V buffer`
- Q/K/V 使用统一 `head_dim`
- 采集结果进入 `context_attention_perf.txt`

`collect_mla.py`：

- `attention_arch=MLA`
- `use_mla=True`
- KV pool 是 `MLATokenToKVPool`
- KV cache 是 latent cache，核心维度为 `kv_lora_rank + qk_rope_head_dim = 576`
- context 直接构造 `512 + 64` 的 MLA 输入，并绕过 DeepSeek `attn_mha` dispatch
- 采集结果进入 `context_mla_perf.txt`

因此，`context_mla_perf.txt` 的 prefill 数据理论上可以代表 “SGLang absorbed MLA prefill 分支”，但不能直接代表 “SGLang DeepSeek 常见无 prefix prefill 的 attn_mha 分支”。

换句话说：

- 如果实机 prefill trace 是 `attn_mha`，应优先与 `context_attention` 对齐，但要承认 qk/v 异维没有被完全建模。
- 如果实机 prefill trace 真的落入 absorbed `attn_mqa` / MLA 分支，才应考虑与 `context_mla` 对齐。
- 仅凭 `kernel_source=flash_attention` 不能判断二者应当对齐。

## 三、AIC SDK：ops 与 PerfDatabase 的实际查表路径

### 1. `deepseek.py` 当前的 fallback 路径

文件：[deepseek.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/models/deepseek.py)

在 context/prefill 路径中：

- 若 `backend_name == "vllm"`，则使用 `ops.ContextAttention`
- 否则使用 `ops.ContextMLA`

也就是说，**当前 SGLang/非-vLLM 路径下，DeepSeek V3 的 context attention 会直接走 `ContextMLA`**。

在 generation/decode 路径中：

- 若 `backend_name == "vllm"`，则使用 `GenerationAttention`
- 否则使用 `MLABmm + GenerationMLA + MLABmm`

这说明：

- `generation` 侧走 MLA，是当前 SDK 中的既定设计
- `context` 侧对非-vLLM 后端也走 MLA，则与当前 SGLang 0.5.9 实机行为存在可疑偏差

### 2. `operations.py` 中四条查询路径是完全分开的

文件：[operations.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/operations.py)

四类 op 的数据库查询分别是：

- `ContextAttention.query()` -> `database.query_context_attention(...)`
- `GenerationAttention.query()` -> `database.query_generation_attention(...)`
- `ContextMLA.query()` -> `database.query_context_mla(...)`
- `GenerationMLA.query()` -> `database.query_generation_mla(...)`

因此 SDK 一旦在组网时选成了 `ContextMLA`，后面就不会再自动回到 `context_attention` 数据表。

### 3. `common.py` 中 perf 文件就是四张表

文件：[common.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/common.py)

对应关系很直接：

- `generation_attention_perf.txt`
- `context_attention_perf.txt`
- `context_mla_perf.txt`
- `generation_mla_perf.txt`

同时还定义了两组重要量化枚举：

- `FMHAQuantMode`
  - `bfloat16`
  - `fp8`
- `KVCacheQuantMode`
  - `bfloat16`
  - `fp8`
  - `int8`

这也是当前比较中要同时关心：

- attention 算子本身是否是 FP8 FMHA
- KV cache 是否是 BF16 / FP8

### 4. `perf_database.py` 中四条查表逻辑也不是同一个公式

文件：[perf_database.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/perf_database.py)

#### `query_context_attention()`

特点：

- 用 `full_s = s + prefix`
- 用 `(full_s^2 - prefix^2) / full_s^2` 做 prefix 修正
- 以 `n / n_kv / head_size / kv_cache_quant_mode / fmha_quant_mode` 为主要索引

这是一套标准 attention/GQA 语义。

#### `query_generation_attention()`

特点：

- 以 `n / n_kv / head_size / kv_cache_quant_mode` 为主
- decode 查表时会对 `s` 在 `0.9x ~ 1.1x` 范围内取若干样本再平均

这表示 AIC 对 decode 常规 attention 做了一个邻域平滑。

#### `query_context_mla()`

特点：

- 同样使用 `full_s = s + prefix`
- 同样有 prefix 修正
- 但其 SOL/查表语义基于 MLA latent 维度
  - 关键常量是 `(192 + 128)`

这不是标准 MHA/GQA attention 的尺寸定义。

#### `query_generation_mla()`

特点：

- decode 公式基于 MLA latent 语义
  - 计算量核心是 `2 * b * num_heads * 1088 * s`
  - KV bytes 核心是 `576 * (s - 1) * kv_cache_quant_mode.memory`
- 没有像 `generation_attention` 那样做 decode 邻域平滑

这说明 `generation_attention` 与 `generation_mla` 不只是“不同文件”，而是连底层公式/形状解释都不同。

## 四、当前实机与仿真能对应上的点

### 1. Prefill 实机 attention 更对应 `collect_attn` / `query_context_attention`

证据链是完整的：

1. SGLang 0.5.9 runtime dispatch 允许 prefill 走 MHA。
2. 现有实机 trace 中，prefill attention 模块名是 `attn_mha`。
3. 实际比较结果里，prefill 对 `query_context_attention()` 误差显著更小。
4. `collect_attn.py` 采样的正是标准 attention/MHA-GQA 语义。

因此，**在当前这批 H100 + SGLang 0.5.9 的 formal case 下，prefill attention 应优先对齐 AIC 的 `context_attention` 数据，而不是 `context_mla`**。

### 2. Decode 实机 attention 更对应 `collect_mla` / `query_generation_mla`

证据链同样完整：

1. SGLang 0.5.9 runtime dispatch 在 decode 侧通常落入 MLA 子类型。
2. 现有实机 trace 中，decode attention 模块名是 `attn_mqa`。
3. 实际比较结果里，decode 对 `query_generation_mla()` 误差显著更小。
4. `collect_mla.py` 的 generation 路径正是围绕 latent KV + MQA 风格构造的。

因此，**decode attention 对齐 `generation_mla` 是合理的**。

## 五、当前误差来源与风险边界

### 1. 最大的结构性偏差：SDK context 路径可能选错表

当前非-vLLM 的 DeepSeek context fallback 直接走 `ContextMLA`，这会导致：

- 实机 prefill 明明走了 `attn_mha`
- 仿真却去查 `context_mla_perf.txt`

这会把两种不同 attention 语义硬对比，误差自然会很大。

### 2. Prefill 并不是任何场景都一定走 MHA

虽然当前 formal case 结论很清楚，但仍要保留边界条件：

- 对 `fa3` / `flashinfer` / `flashmla` backend
  - 若 `prefix` 非零且未达到 `chunked_prefix_cache_threshold`
  - 且未满足 one-shot/chunked-kv 条件
  - runtime 可能回落到 MLA 分支
- 对 `triton`
  - `extend_prefix_lens_cpu == 0` 时走 MHA
  - 非零 prefix 则可能切到 MLA

所以更准确的说法应当是：

- **当前 formal 实机 case 中的 prefill attention 与 `context_attention` 对得上**
- 不能无限外推为“所有 SGLang prefill 都等于 attention、不等于 mla”

### 3. 量化口径仍会带来误差

需要继续注意两类量化口径：

- FMHA 的计算量化
  - `attn_dtype` / `FMHAQuantMode`
- KV cache 的存储量化
  - `kv_cache_dtype` / `KVCacheQuantMode`

即使 attention 路径选对了，如果实机与 AIC 使用的量化口径不一致，误差仍会扩大。

### 4. Decode 两条查询路径的建模细节也不同

`query_generation_attention()` 有 decode 邻域平均，而 `query_generation_mla()` 没有。即使两边都查到了“正确大类”，仍可能因为：

- decode 批内序列长度分布
- prefix/cache 实际组织方式
- KV dtype

产生剩余误差。

## 六、对当前 SDK DeepSeek context 路径的判断

基于这轮静态分析和现有实机结果，我的判断是：

- **decode/generation 侧**
  - 当前 SDK 对非-vLLM 路径使用 `GenerationMLA`
  - 这与 SGLang 实机 decode 更一致
  - 暂时看是合理的
- **context/prefill 侧**
  - 当前 SDK 对非-vLLM 路径使用 `ContextMLA`
  - 这与当前 SGLang 0.5.9 + H100 实机 formal case 不一致
  - 很可能是当前 prefill attention 对比失真的主要原因

换句话说，当前 DeepSeek V3 在 SDK 里的 attention 口径，较像：

- context: 过度 MLA 化
- generation: 基本合理

## 七、建议的后续动作

若后续要继续推进“实机 vs AIC attention 单算子对齐”，建议按下面顺序处理：

1. 先把 DeepSeek V3 的 **context attention 查询路径** 与当前 SGLang 0.5.9 formal case 对齐，优先验证改成 `ContextAttention` 后误差变化。
2. decode 侧继续保留 `GenerationMLA` 为主线，对 `kv_cache_dtype` 和实际 backend 再做细化。
3. 若要支持更广泛的 prefix 场景，需要把 “prefill 是走 MHA 还是 MLA” 做成 **按 runtime 条件判定**，而不是仅按 backend 名字或模型名静态决定。

## 八、collector 侧进一步细分：为什么 `collect_mla` 的 prefill 也不一定对上实机 `attn_mha`

本节专门回答两个细节问题：

- `collect_attn` 以及 SDK 的 `query_context_attention()` 是否能处理 DeepSeek 这类 `q/k head_dim != v head_dim` 的情况？
- `collect_mla` 在 prefill 上也用了 FA/FA3，并且写了 DeepSeek MLA 特殊形状，为什么仍然可能和实机 prefill 对不上？

### 1. `collect_attn` 当前不能完整表达 `q/k` 与 `v` 异维

`collector/sglang/collect_attn.py` 当前是标准 MHA/GQA collector。它的数据模型只有一个 `head_dim`：

- `MockModelConfig(num_attention_heads, num_key_value_heads, head_dim)` 只记录 `head_dim`。
- `MockModelConfig.hidden_size = num_attention_heads * head_dim`。
- `MHATokenToKVPool(..., head_dim=head_dim)` 只传一个维度。
- `RadixAttention(..., head_dim=head_dim)` 没有传 `v_head_dim`，因此 `RadixAttention` 内部会默认 `v_head_dim = head_dim`。
- prefill 时 `q/k/v` 都按 `[tokens, heads, head_dim]` 构造。

所以 `collect_attn` 可以表达：

- 标准 MHA：`num_key_value_heads == num_heads`
- GQA/XQA：`num_key_value_heads < num_heads`
- 不同 `head_dim` 网格：当前主要采 `128/256`
- BF16/FP8 FMHA 与 BF16/FP8 KV cache

但它不能精确表达 DeepSeek prefill MHA 的真实形状：

- `q/k head_dim = qk_nope_head_dim + qk_rope_head_dim = 128 + 64 = 192`
- `v head_dim = 128`

这不是 SGLang 底层 `RadixAttention` 不支持。SGLang 的 `RadixAttention` 构造函数本身有 `v_head_dim` 参数，DeepSeek 真正的 `attn_mha` 也确实使用了 `head_dim=192, v_head_dim=128`。限制在于 AIC 当前的 `collect_attn.py` 和 `context_attention_perf.txt` schema 没有把 `qk_head_dim` 与 `v_head_dim` 拆成两个字段。

### 2. SDK 的 `query_context_attention()` 也只有一个 `head_size`

SDK 侧同样如此：

- `ops.ContextAttention` 只有 `head_size`。
- `PerfDatabase.load_context_attention_data()` 从表里读取的也是单一 `head_dim`。
- `PerfDatabase.query_context_attention()` 的参数也是 `head_size`。
- SOL 公式里用同一个 `h` 同时表示 Q/K/V 的维度。
- `ContextAttention.query()` 里的额外修正也按 `q_num=n*head_size`、`k_num=n_kv*head_size`、`v_num=n_kv*head_size` 计算。

因此即便把 DeepSeek SGLang context op 改成查 `ContextAttention`，当前也只是更接近实机 `attn_mha` 的大类语义，而不是完全同形状查询。它会把 DeepSeek 的 `q/k=192, v=128` 近似成某个单一 `head_size` 的普通 attention 点，通常只能在 `head_dim=128` 或 `256` 表格上插值/外推。

这解释了一个看似矛盾的现象：

- 实机 prefill 与 `context_attention` 比 `context_mla` 更接近。
- 但 `context_attention` 仍然有不可忽略误差，因为它没有 qk/v 异维这一维度。

### 3. `collect_mla` 的 DeepSeek prefill 形状不是实机常见的 `attn_mha` 形状

`collector/sglang/collect_mla.py` 确实是围绕 DeepSeek MLA 写的，并且在 H100 上默认会选择 FA3/FlashAttention backend。但它的 context 构造逻辑与完整 SGLang DeepSeek 模型的 prefill MHA 仍然不同。

`collect_mla.py` 中：

- `MockModelConfig.attention_arch = AttentionArch.MLA`
- `MockModelConfig.get_num_kv_heads()` 固定返回 `1`
- KV pool 使用 `MLATokenToKVPool`
- `RadixAttention` 使用 `num_kv_heads=1`
- H100/FA3 路径下，context 侧设定：
  - `v_head_dim = kv_lora_rank = 512`
  - `head_dim_total = kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576`
  - `q_nope`: `[tokens, local_heads, 512]`
  - `q_rope`: `[tokens, local_heads, 64]`
  - `k_nope`: `[tokens, 1, 512]`
  - `k_rope`: `[tokens, 1, 64]`
  - `v = k_nope`

而 SGLang 0.5.9 DeepSeek 实机常见 prefill MHA 路径中，`forward_mha.py` 明确写了 DeepSeek-V3 的 MHA 形状：

- `q/k head_dim = 192`
- `v_head_dim = 128`
- `num_kv_heads = num_local_heads`

真实 DeepSeek prefill MHA 会先做 `kv_b_proj(kv_a)`，把 latent KV 展开成每个 attention head 对应的 `k_nope` 和 `v`，再拼上 rope 部分形成 `k[192]`，最后调用 `attn_mha(q[192], k[192], v[128])`。

`collect_mla.py` 没有跑这个完整 DeepSeek module，也没有经过 `kv_b_proj -> k_nope/v -> attn_mha` 的路径。它直接构造 MLA latent KV 形态并调用一个 MLA 语义的 `RadixAttention`。

### 4. 两者都显示 `kernel_source=flash_attention`，但这不是同一件事

当前最容易误判的地方是：`collect_attn` 和 `collect_mla` 在 H100 上都可能输出 `kernel_source=flash_attention`。

但 `kernel_source` 只说明最终用了 FlashAttention 后端家族，不说明输入语义相同：

- `collect_attn`
  - `AttentionArch.MHA`
  - `FlashAttentionBackend.use_mla = False`
  - `MHATokenToKVPool`
  - 标准 K/V cache
  - 单一 `head_dim`
  - 表为 `context_attention_perf.txt`
- `collect_mla`
  - `AttentionArch.MLA`
  - `FlashAttentionBackend.use_mla = True`
  - `MLATokenToKVPool`
  - latent KV cache
  - `kv_lora_rank + qk_rope_head_dim` 语义
  - 表为 `context_mla_perf.txt`

在 `FlashAttentionBackend.forward_extend()` 里，`use_mla` 会直接改变执行分支：

- `use_mla=False`：走普通 MHA/GQA 路径。
- `use_mla=True` 且 `forward_batch.attn_attend_prefix_cache is not None`：走 DeepSeek MLA 模型中的 MHA/chunked-prefix MHA 辅助路径。
- `use_mla=True` 且 `forward_batch.attn_attend_prefix_cache is None`：走 absorbed MLA latent 路径。

`collect_mla.py` 直接调用 `RadixAttention`，没有经过 DeepSeek 模型层的 `dispatch_attn_forward_method()`，也没有设置真实 `forward_normal_*` 路径中的 `attn_attend_prefix_cache/mha_one_shot/mha_return_lse` 等状态。因此它更接近 absorbed MLA latent attention，而不是当前实机 prefill trace 中观测到的 `attn_mha`。

### 5. 为什么 `context_mla` 的 SOL 公式看起来像 `192+128`，但表数据仍可能不对齐

SDK `query_context_mla()` 的 SOL 公式里确实写了 DeepSeek MHA 风格常量：

- attention 计算使用 `(192 + 128)`
- 注释也写了 `q read 192 / k read 192 / v read 128`

但 silicon 查询不是用这个公式直接生成，而是优先查 `context_mla_perf.txt`。这张表来自 `collect_mla.py`，而 `collect_mla.py` 在 H100/FA3 路径下实际采样的 context shape 是 `576/512` 的 MLA latent 语义，不是 `192/128` 的 `attn_mha` 语义。

所以这里存在一个重要的内部口径不一致风险：

- `query_context_mla()` 的经验/SOL 解释像 DeepSeek prefill MHA。
- `context_mla_perf.txt` 的 collector 实测数据在 H100/FA3 下更像 MLA latent attention。
- SGLang 实机常见 prefill trace 则是 DeepSeek `attn_mha`。

这也是为什么“`context_mla` 理论上 DeepSeek 特化，所以应该对上”这个直觉目前没有成立。

### 6. 本轮结论

更精确的判断如下：

- `collect_attn + query_context_attention`：语义上更接近实机 prefill `attn_mha`，但 schema 不支持 `q/k=192, v=128` 异维，只能近似。
- `collect_mla + query_context_mla`：面向 DeepSeek MLA，但 collector 直接构造 MLA latent attention，绕过真实 DeepSeek prefill MHA dispatch；在 H100/FA3 上不等价于实机常见 `attn_mha`。
- `generation_mla`：与实机 decode `attn_mqa` 更一致，当前仍是 decode 对齐的主线。
- 后续若要把 prefill attention 单算子彻底对齐，最好新增或扩展一条 DeepSeek prefill MHA collector/schema，显式记录 `qk_head_dim`、`v_head_dim`、`num_heads`、`num_kv_heads`、`prefix`、`fmha_dtype`、`kv_cache_dtype`，并让 SDK context 查询按 SGLang runtime dispatch 选择 `context_attention` / `context_mla` / DeepSeek-MHA-specialized 表。

## 附：本轮关键证据文件

- SGLang docker 0.5.9 源码
  - `sglang/srt/models/deepseek_v2.py`
  - `sglang/srt/models/deepseek_common/attention_backend_handler.py`
- AIC SDK
  - [deepseek.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/models/deepseek.py)
  - [operations.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/operations.py)
  - [perf_database.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/perf_database.py)
  - [common.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/src/aiconfigurator/sdk/common.py)
- AIC collector
  - [collect_attn.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/collector/sglang/collect_attn.py)
  - [collect_mla.py](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/collector/sglang/collect_mla.py)
- 现有对比结果
  - [mla_kernel_aic_compare_with_decode__summary.md](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/analysis/mla_kernel_aic_compare_with_decode__summary.md)
  - [mla_kernel_aic_compare_with_decode__comparison.csv](/home/ai_lab/ljc/scale-up-sim/aiconfigurator/bench_data/h100_sxm/sglang/v0.5.9/deepseek_v3_mla/analysis/mla_kernel_aic_compare_with_decode__comparison.csv)
