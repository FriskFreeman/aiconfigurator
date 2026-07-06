# Decode Attention Root-Cause Analysis

## 结论

- AIC `generation_attention` 与实机 pure FA3 attention 的 MAPE 为 `103.22%`，即平均约 `2.03x`；把 rotary 与 KV write support 加入实机侧后 MAPE 仍有 `85.03%`，平均约 `1.85x`。
- 最大相对差异出现在 `decode_b2p16384_cg_on`：实机 pure FA3 `0.041152ms`，AIC `0.104854ms`，gap `154.80%`。
- `s=kv_len+1` 插值不是主因：诊断性改查 `s=kv_len` 后 attention MAPE 为 `103.17%`，AIC latency 平均只改变 `0.0276%`。
- 更可信的主因是采集/实机口径不同：AIC `generation_mla_perf.txt` 来自 collector 手工构造的 `RadixAttention` decode 微基准，实机对比取的是完整 SGLang CUDA Graph 中 MLA body 内的 FA3 backend kernel 三段。
- 当前 AIC generation_mla 数据本身就是较大的数值，不是 SDK 额外放大；PerfDatabase 只是按 `(num_heads,b,s)` 读取或近邻插值这些 collector 数据。

## 实机口径

- 实机默认 FA3 decode 的 attention 被解析为 `prepare_varlen_num_blocks_kernel + device_kernel + device_kernel`。
- 这三段分别在 CSV 中记录为 `fa3_prepare`、`fa3_attention`、`fa3_combine`；rotary 与 `set_mla_kv_buffer` 单独列出为 support，不计入主 attention 单算子口径。
- SGLang 0.5.9 `forward_mla.py` 中 decode MLA 调用 `self.attn_mqa(...)` 后再做 `attn_output.view(...)` 与后续 BMM/o_proj；`flashattention_backend.py::forward_decode()` 在 `self.use_mla` 分支中调用 `flash_attn_with_kvcache(q=q_rope, k_cache=k_rope_cache, v_cache=c_kv_cache, qv=q_nope, ...)`。

## Collector 口径

- `collector/sglang/collect_mla.py` 的 generation 分支手工构造 `ForwardBatch(DECODE)`、预填 KV pool，然后调用 `RadixAttention.forward()` 并用 `benchmark_with_power()` 统计整次 layer 调用。
- 采集输出 `op_name=mla_generation`，`kernel_source=flash_attention`，`isl=1`，`step=input_len`；PerfDatabase 加载时把 `s_total=isl+step` 作为查询坐标。
- 当前相关 perf 数据的 `kernel_source` 为 `flash_attention`，说明 AIC 查询确实来自 flash_attention collector 数据。

## SDK 查询

- `ops.GenerationMLA.query()` 调用 `PerfDatabase.query_generation_mla(batch_size, s, num_heads, kvcache_quant_mode)`。
- `query_generation_mla()` 在 silicon mode 下读取 `_generation_mla_data[kvcache_quant_mode][num_heads][b][s]`，通过 `_interp_3d(..., 'bilinear')` 查询。
- 本组 case 的 `b` 与 `num_heads=128` 都精确命中；`s=kv_len+1` 位于相邻 grid 点之间，但诊断表明 +1 插值几乎不改变结果。

## 判断

- 已排除或弱化：module 边界、support kernel 漏计、`s=kv_len+1` 插值、SDK 额外缩放。
- 仍需谨慎：collector 的 `RadixAttention.forward()` 计时是否包含比实机解析的三段 FA3 kernel 更多的 backend 内部工作，或者 collector 构造的 q/k/v/KV pool metadata 是否触发了比真实 CUDA Graph replay 更慢的 FA3 路径。仅靠现有 trace 不能完全拆开 collector 内部 kernel 组成。
- 因此，若要修正 AIC 单算子 fallback，优先方案不是在 SDK query 端调参，而是重新定义/重采 `generation_mla_perf` 的 collector 口径，使其与实机 FA3 attention kernel 组件对齐，或者在 SDK 普通 DeepSeek fallback 中改用 module 级 `wideep_generation_mla` 数据。

## 输出文件

- `attention_case_compare.csv`: 每个 decode case 的实机 FA3/support/AIC attention 对照。
- `generation_mla_perf_grid_extract.csv`: 与这些 case 相关的 generation_mla collector 原始格点。
