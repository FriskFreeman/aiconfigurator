# Decode KVCache 共享复用时延修正分析

## 1. 结论摘要

SGLang 0.5.9 的 RadixCache 会让命中相同 Prefix 的多个请求复用同一批物理 KV pool 索引。FA3 decode 仍为每个请求执行完整 attention 计算，但会重复读取相同的物理 KV 地址，因此可能通过 L2/HBM 访问局部性降低 kernel 时延。

AIC 当前的 generation MLA 查询只使用 `batch_size`、KV 长度、local heads/TP、KV dtype 和 backend，不包含 batch 内跨请求的物理 KV 复用关系。数据库中的 silicon 点也来自彼此独立的请求映射，所以当前结果对应“无共享”基线。

基于现有 `B=32, S=32768, local_heads=32, BF16 KV, FA3` 三个实验点，不能把时延直接乘以“唯一 KV 页比例”。更合理的一阶模型是带计算下限的访存 roofline：

```text
T_attn(rho) = max(T_floor, T_overhead + K_kv * rho)
```

其中 `rho` 是“唯一物理 KV token 数 / 逻辑 KV token 访问数”。现有三点拟合为：

```text
T_floor    = 0.217810 ms
T_overhead = 0.064429 ms
K_kv       = 0.341298 ms
```

该公式恰好复现现有三个 case，且 `K_kv` 对应约 `3.54 TB/s` 的有效 KV 读取带宽，与 H100 HBM 带宽数量级吻合，因而比纯经验多项式更有物理解释。但这仍只是一个 shape 上的初步校准，暂时不应直接作为全局生产公式。

## 2. SGLang 中的真实机制

依据 Docker 镜像 `booleimg.myaddr.io/lmsysorg/sglang:v0.5.9`：

1. `schedule_batch.py::Req.init_next_round_input()` 调用 `tree_cache.match_prefix()`，将返回的 `device_indices` 保存为 `req.prefix_indices`。
2. `radix_cache.py::RadixCache.match_prefix()` 返回 Radix tree 节点中保存的物理 KV indices，而不是只返回命中长度。
3. `mem_cache/common.py::write_cache_indices()` 把 `prefix_indices` 写入新请求自己的 `req_to_token` 行；只有 fresh token 才分配新 KV 位置。
4. `flashattention_backend.py::init_forward_metadata()` 将各请求的 `req_to_token` 行直接作为 FA3 `page_table`。
5. MLA decode 分支把该 `page_table` 传给 `flash_attn_with_kvcache()`。

因此，同 Prefix 请求在原生 SGLang 中会形成如下映射：

```text
req 0 page table: [P0, P1, ..., Pn, A0, A1, ...]
req 1 page table: [P0, P1, ..., Pn, B0, B1, ...]
req 2 page table: [P0, P1, ..., Pn, C0, C1, ...]
```

FA3 不会因为前三行 Prefix 相同而省略请求级 QK/softmax/V 计算；收益来自多个请求读取相同物理地址后，实际落到 HBM 的流量可能下降。

## 3. AIC 当前处理方式

### 3.1 Prefill Prefix 修正

`src/aiconfigurator/sdk/perf_database.py` 中 context attention/MLA 的 silicon 查询使用：

```text
full_s = fresh_s + prefix
prefix_correction = (full_s^2 - prefix^2) / full_s^2
corrected_latency = collected_latency(full_s) * prefix_correction
```

它修正的是 prefill 中不需要重新计算“Prefix 内部 query-key 区域”带来的计算量变化。这个公式描述的是逻辑 attention pair 数减少，而不是跨请求复用同一物理 KV 后的缓存局部性。

因此该公式不能直接用于 decode：decode 每个请求仍只有一个新 query，并且仍需逻辑上 attend 全部历史 KV；共享 Prefix 并没有减少逻辑 FLOPs。

### 3.2 Decode MLA 查询

当前主要入口包括：

- `GenerationMLA.query()` -> `PerfDatabase.query_generation_mla(b, s, num_heads, kv_dtype)`
- `WideEPGenerationMLA.query()` -> `query_wideep_generation_mla(b, s, tp, kv_dtype, fmha_dtype, backend)`
- `MLAModule` generation -> `query_generation_mla_module(...)`

这些接口没有 Prefix 共享拓扑或唯一 KV 比例参数。silicon 查表按 `(heads/TP, batch, seq)` 插值，默认每个请求的 KV 映射独立。

现有 SOL 中的 KV 访存量也按逻辑请求数计算，例如 granular generation MLA 使用近似项：

```text
KV bytes = batch * (seq - 1) * 576 * bytes_per_kv_element
```

它没有区分逻辑访问量与唯一物理 KV 数据量。

## 4. 三个实验点的数据拆解

固定实验条件：

```text
batch=32
history sequence=32768
local_heads=32
shared_prefix=29491 (约 90%)
backend=FA3
KV dtype=BF16
单层 self_attn CUDA Graph replay
```

| 拓扑 | rho：唯一/逻辑 KV | FA3 主 kernel (ms) | 全部 kernel sum (ms) | 非 FA3 kernel sum (ms) |
|---|---:|---:|---:|---:|
| 32 个独立请求 | 1.000000 | 0.405727 | 0.454373 | 0.048647 |
| 16 组 x 2，共享且打乱 | 0.550017 | 0.252148 | 0.301832 | 0.049684 |
| 1 组 x 32 全共享 | 0.128158 | 0.217810 | 0.267473 | 0.049664 |

非 FA3 kernel 基本恒定，均值约 `0.04933 ms`。变化集中在 FA3 主 kernel，因此校正应作用在 attention 部分，而不应对整个 MLA module 等比例缩放。

### 4.1 为什么唯一比例线性缩放失败

若使用：

```text
T_attn = A + B * rho
```

并用 independent/all-shared 两点确定参数，则 pair-shared 的预测约为 `0.3087 ms`，而实测为 `0.2521 ms`，误差约 `22%`。原因是 `rho` 只描述唯一地址数量，没有描述：

- 同一地址被再次访问前的 reuse distance；
- batch 行顺序和 FA3 tile 调度；
- L2 容量及 cache line 粒度；
- HBM 与计算瓶颈之间的切换；
- page size 和 kernel 内部的 split/merge 策略。

### 4.2 带计算下限的访存模型

把 independent 和 pair-shared 视为访存分支、all-shared 视为计算/固定开销下限，可得：

```text
K_kv       = (0.4057266 - 0.2521482) / (1 - 0.5500168)
           = 0.341298 ms

T_overhead = 0.4057266 - K_kv
           = 0.064429 ms

T_floor    = 0.217810 ms
```

最终：

```text
T_attn(rho) = max(0.217810, 0.064429 + 0.341298 * rho) ms
```

| rho | 公式预测 (ms) | 实测 (ms) |
|---:|---:|---:|
| 1.000000 | 0.405727 | 0.405727 |
| 0.550017 | 0.252148 | 0.252148 |
| 0.128158 | 0.217810 | 0.217810 |

访存分支和计算下限在 `rho ~= 0.4494` 处相交。也就是说，在这个固定 shape 下，当唯一 KV 比例继续下降到约 45% 以下后，继续共享的收益会迅速饱和；这与 pair-shared 到 all-shared 的边际收益明显小于 independent 到 pair-shared 相符。

逻辑 KV 数据量约为：

```text
32 * 32769 * 576 * 2 bytes ~= 1.208 GB
```

用它除以 `K_kv=0.341298 ms`，得到约 `3.54 TB/s`。该值略高于标称 HBM 带宽，但考虑计数口径、cache 命中和拟合误差，仍说明斜率确实处于 H100 KV 访存带宽的合理数量级。

## 5. rho 的计算方式

对每个共享组 `g`，设：

- `n_g`：组内请求数；
- `P_g`：组内共同物理 Prefix 长度；
- `S_i`：请求 `i` 的总 KV 长度。

假设不同组之间不共享、同组仅共享共同 Prefix，则：

```text
logical_kv = sum_i(S_i)

unique_kv = sum_g(P_g) + sum_i(S_i - P_group(i))

rho = unique_kv / logical_kv
```

本实验中所有请求等长、共享比例为 `p`、每组大小为 `n`，可简化为：

```text
rho = (1 - p) + p / n
```

因此：

```text
independent: rho = 1
pair-shared: rho ~= 0.1 + 0.9/2  = 0.55
all-shared:  rho ~= 0.1 + 0.9/32 = 0.128125
```

实际接入时应使用精确 token/page 数，并按 backend page size 对 Prefix 向下对齐。FA3 默认 `page_size=1`；FlashMLA 在 SGLang 0.5.9 中强制 `page_size=64`。

复杂的树状 Prefix（多个组又共享更短的公共祖先）不能用上述简单分组公式重复相加，应计算 Prefix trie/radix tree 中物理 KV 区间的并集。

## 6. 建议的 AIC 修正方案

### 6.1 输入语义

建议为 decode runtime/batch 增加可选的共享信息，而不是把它混入现有单请求 `prefix` 字段：

```text
decode_kv_unique_ratio: float | None
```

或提供更上层的：

```text
decode_kv_share_groups = [
  {request_indices: [...], shared_prefix_len: ...},
  ...
]
```

由 scheduler/workload 层计算 `rho`。未提供时使用 `rho=1`，保持当前行为和向后兼容性。

`prefix` 表示单请求已有多少历史 KV；`rho` 表示 batch 内这些历史 KV 有多少是同一份物理数据，二者语义不能合并。

### 6.2 Granular generation MLA

优先在 `query_generation_mla()` 的 attention 结果上校正。建议形式为：

```text
T_corrected = max(
    T_floor(b, s, heads, dtype, backend),
    T_overhead(b, heads, dtype, backend)
      + rho * T_unique_kv(b, s, dtype, backend),
)
```

不建议直接写死本实验的三个常数。可利用已有“无共享、多个 seq 长度”的 collector 数据拟合：

- `T_unique_kv`：无共享时随 `batch * seq * kv_bytes` 增长的斜率；
- `T_overhead`：访存分支截距；
- `T_floor`：短序列平台期，或基于有效计算吞吐估计的下限。

这不要求为每种共享拓扑 collect，只需要现有独立 KV 的 `(b, s)` sweep 足够覆盖从计算受限到访存受限的区间。三种共享实验可作为模型验证点，而不是数据库的新维度。

### 6.3 Module-level MLA

module 数据包含 q/k projection、RoPE、KV write、attention 和 output projection。只允许校正随历史 KV 读取变化的 attention 部分：

```text
T_module_corrected
  = T_module_baseline
  - T_attn_baseline
  + T_attn_corrected(rho)
```

若 module 表没有 attention 子项，可采用两种方案：

1. 同时查询 granular generation MLA，使用其共享前后的 delta 修正 module：

   ```text
   T_module_corrected = T_module_baseline
                      + T_granular(rho) - T_granular(1)
   ```

2. 从 module 表在固定 `(b, heads)` 下对多个 `s` 拟合常量部分与 KV 线性部分，但其可信度低于第一种。

现有 trace 中非 attention kernel 约 `0.0493 ms` 且不随共享变化，证明整 module 等比例缩放会过度修正。

### 6.4 SOL/HYBRID 模式

SOL 的逻辑 KV bytes 可以改为：

```text
unique_kv_bytes = rho * logical_kv_bytes
```

但仍需保留计算下限：

```text
sol_time = max(sol_math_effective, sol_non_kv_mem + unique_kv_bytes / effective_bw)
```

这里应使用校准后的 `effective_bw` 和 `effective_math_floor`，不能只使用峰值规格。当前 generation MLA 理论 FLOPs 除以峰值算力得到的时间明显低于实测 all-shared 平台，因此单纯替换 `mem_bytes` 会高估共享收益。

HYBRID/SILICON 建议锚定无共享 silicon 值，再应用模型给出的 delta 或 ratio，以避免破坏当前数据库在 `rho=1` 下的准确性。

## 7. 当前证据能支持与不能支持的结论

可以支持：

- AIC 无共享查表会高估有共享 Prefix 的 FA3 decode attention 时延。
- 校正变量应描述物理 KV 唯一量/复用关系，而不是 prefill 的逻辑 Prefix 命中计算量。
- 修正应只作用于历史 KV 相关 attention 部分。
- “访存分支 + 计算下限”的模型能够解释当前三个 case，并自然表达收益饱和。

暂时不能支持：

- 将当前常数推广到其他 batch、sequence、TP/head、KV dtype 或 GPU。
- 将 FA3 的校正直接用于 FlashMLA、FlashInfer 或 Triton backend。
- 认为 `rho` 是唯一决定因素；pair-shared 的稳定多档时延说明请求顺序/reuse distance 也可能重要。
- 仅凭 Chrome trace 判断收益具体来自 L2、TLB 还是 HBM transaction。

## 8. 最小补充验证建议

在不枚举所有共享拓扑的前提下，建议补充一个很小的验证矩阵：

```text
batch:       8, 16, 32
sequence:    4096, 16384, 32768
group size:  1, 2, 4, batch
shared ratio: 50%, 90%
order:       contiguous, shuffled
KV dtype:    生产默认 dtype
backend:     分别校准 FA3 与实际使用的 decode backend
```

重点不是把这些点全部写入 PerfDatabase，而是验证以下参数化关系是否成立：

```text
K_kv ~= logical_kv_bytes / effective_bandwidth
T_floor ~= f(batch, heads, backend, dtype)
T_overhead ~= f(batch, heads, backend)
```

若 contiguous 与 shuffled 在相同 `rho` 下持续存在显著差异，则需要再引入一个局部性参数，例如平均 reuse distance 或共享请求在 batch 中的平均间距；否则 `rho + roofline` 已足够作为第一版低成本修正。

## 9. 推荐落地顺序

1. 先把 `decode_kv_unique_ratio` 作为可选输入贯通到 generation MLA 查询，默认值为 1，不改变现有结果。
2. 用现有 independent collector sweep 拟合每个 backend/dtype 的访存斜率和计算平台，不增加共享拓扑数据库维度。
3. granular MLA 使用 roofline 修正；module MLA 使用 granular delta 修正。
4. 用少量共享实验验证，不满足误差阈值时再考虑 reuse distance 参数。
5. 在模型稳定前，将该功能标记为实验性，并在结果中输出实际使用的 `rho` 和 correction factor，确保可追踪。

总体上，这个 decode 问题可以采用理论访存模型低成本修正，不必为每种 Prefix 共享组合建立 collect case；但第一版应采用“silicon 基线锚定 + attention 局部 delta + roofline 饱和”，而不是把整个 module 时延乘以唯一页比例。
