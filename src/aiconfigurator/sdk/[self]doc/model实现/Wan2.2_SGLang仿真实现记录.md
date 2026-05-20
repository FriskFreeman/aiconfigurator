# Wan2.2 SGLang 仿真实现记录

本文记录 SDK 侧 Wan2.2 视频模型仿真实现。实现目标是使用 SGLang `0.5.10.post1-wan2.2-main0515` 单卡算子数据，近似还原 SGLang Wan2.2 主推理通路；当前覆盖 T5、CLIP、VAE、Wan DiT 主干、TP/SP 通信估算，SLA/VSA/稀疏分支仍不纳入主仿真。

## 入口与配置

| 组件 | 实现内容 | 说明 |
|---|---|---|
| `common.py` | 新增 `ModelFamily=WAN`、Wan profile/别名、Wan collector 文件枚举 | 支持 Wan2.2 TI2V 5B、T2V A14B、I2V A14B |
| `utils.py` | 读取本地 HF config 与模块 config，解析为 SDK 统一模型字段 | `in_dim/out_dim/d_model/ffn_dim/num_heads` 等不再硬编码 |
| `config.py` | `RuntimeConfig` 增加 `video_*`、`denoising_steps`、`sp_size`、`ulysses_degree`、`ring_degree`、`sp_algorithm`、`attention_backend` | 这些字段只被 Wan 路径消费，LLM 路径保持兼容 |
| `models.py` | `get_model()` 接入 `WanVideoModel` | Wan 被表示为 `context_ops` 静态视频 pipeline，使用 `run_static(..., mode="static_ctx")` |
| `operations.py` | 新增 Wan 实测 op、静态 GEMM/memory op、`WanParallelComm` | 多卡通信固定用 `DatabaseMode.EMPIRICAL` 查询 NCCL/P2P 理论估算 |
| `cli/api.py` | `cli_estimate()` 可传视频与 SP 参数；Wan 自动走 `static_ctx` | 支持程序接口单点估计 Ulysses/Ring 组合 |

## 推理阶段到算子映射

| Wan 阶段 | SDK op 表达 | 性能数据 |
|---|---|---|
| T5 文本编码 | `wan_t5_*` 实测 op + `WanStaticGEMM` + T5 parallel group AllReduce | `wan_t5_perf.txt` + Wan GEMM 表 + empirical 通信 |
| CLIP 图像编码 | `wan_clip_*` 实测 op + `WanStaticGEMM`，仅 I2V/TI2V | `wan_clip_perf.txt` + Wan GEMM 表 |
| 条件图像 VAE encode | `wan_vae`、`wan_vae_attention`、`wan_vae_elementwise`，仅 I2V/TI2V | `wan_vae*.txt` + height halo/gather empirical 通信 |
| DiT patch embed | `wan_patch_embed` | `wan_patch_embed_perf.txt` |
| DiT RoPE | `wan_rope`，使用 Ulysses×Ring 后的 self-attn local shape | `wan_rope_perf.txt` |
| DiT USPAttention compute | `wan_attention`，self/cross 分别建模 | `wan_attention_perf.txt` |
| DiT norm/residual/modulation | `wan_elementwise`，SP 后 local token shape | `wan_elementwise_perf.txt` |
| DiT Linear/FFN/condition embed | `WanStaticGEMM` 与少量 `WanStaticMemOp` | Wan GEMM 表；简单 elementwise 走 memory 模型 |
| TP/SP 通信 | `WanParallelComm` | `query_nccl/query_p2p(..., database_mode=EMPIRICAL)` |
| VAE decode | `wan_vae`、`wan_vae_attention`、`wan_vae_elementwise` | `wan_vae*.txt` + height gather/halo empirical 通信 |

## 并行维度公式

| 维度 | 公式/默认值 | 备注 |
|---|---|---|
| latent shape | `F_lat=(frames-1)//4+1`，`H_lat=height//8`，`W_lat=width//8` | 与 collector `wan_common.latent_shape()` 对齐 |
| DiT token 数 | `seq_len=F_lat*(H_lat//2)*(W_lat//2)` | patch size 为 `(1,2,2)` |
| TP 后 heads | `heads_after_tp=num_heads//tp_size` | A14B `40` heads，TI2V `24` heads |
| SP 分解 | `sp_size = ulysses_degree * ring_degree` | 若只给 `sp_size`，按 SGLang 默认解析为 `ulysses_degree=sp_size, ring_degree=1` |
| self-attn local shape | `q_seq_len=ceil(global_seq_len/ring_degree)`，`num_heads=heads_after_tp//ulysses_degree` | Ulysses 切 heads，Ring 切 sequence；`sp_algorithm` 记录为 `ulysses/ring/usp/none` |
| cross-attn local shape | `q_seq_len=ceil(global_seq_len/sp_size)`，`num_heads=heads_after_tp` | SGLang cross attention 使用 `skip_sequence_parallel=True`，不做 Ulysses head all-to-all |
| DiT GEMM/elementwise M | `ceil(global_seq_len/sp_size)` | patch 后 sequence shard 再进入 block 内非 self-attn 计算 |
| T5 parallel group | `tp_size`；若 `tp_size==1 && sp_size>1` 则为 `sp_size` | 对应 SGLang `parallel_folding_mode="sp"` |
| VAE encode height | `padded_height/(sp_size)`，`padded_height` 对齐 `sp_size*2**downsample_count` | 对应 `split_for_parallel_encode()` 的 height padding/split |
| VAE decode height | `ceil(height_or_latent_height/sp_size)` | decode 按 height 分片，末尾 all-gather |

## 通信建模

| 通信点 | SGLang 行为 | SDK 表达 |
|---|---|---|
| DiT TP RowParallel 输出 | Row parallel linear 后 AllReduce | `wan_dit_*_tp_all_reduce`，NCCL empirical |
| DiT Ulysses input | Q/K/V 在 sequence/head 维 AllToAll | `wan_dit_usp_input_{q,k,v}_alltoall`，NCCL empirical |
| DiT Ring attention | KV 沿 ring 传递 | `wan_dit_ring_attention_kv_p2p`，P2P empirical，`hop_count=ring_degree-1` |
| DiT Ulysses output | attention 输出 AllToAll 回到 sequence shard | `wan_dit_usp_output_alltoall`，NCCL empirical |
| DiT output gather | block 完成后 sequence all-gather | `wan_dit_output_sequence_all_gather`，NCCL empirical |
| T5 folding/TP | embedding/out/FFN RowParallel AllReduce | `wan_t5_*_all_reduce`，NCCL empirical |
| VAE parallel encode/decode | height split 后 gather，distributed conv halo exchange | `wan_vae_*_height_all_gather` 与 `*_height_halo_p2p`，empirical |

所有 Wan 通信均强制标记为 `source="empirical"`，即使全局数据库模式是 `SILICON/HYBRID`，也不会依赖通信实测 CSV。

## PerfDatabase 查询

| 数据文件 | 查询方法 | 匹配策略 |
|---|---|---|
| `wan_patch_embed_perf.txt` | `query_wan_patch_embed` | `model/task/batch/in_channels/hidden_size` 精确，shape 最近邻 |
| `wan_rope_perf.txt` | `query_wan_rope` | 全 key 精确；旧数据缺 `usp` 时允许回退到同 shape 的 `ulysses` 历史标签 |
| `wan_attention_perf.txt` | `query_wan_attention` | `model/task/attn_kind/backend/batch/q/k/head` 精确，`tp/sp/algorithm` 最近邻；旧 cross SP 可从 `cross_local` 回退历史 `ring` 标签 |
| `wan_elementwise_perf.txt` | `query_wan_elementwise` | `op_name/batch/seq/hidden/tp` 精确 |
| `wan_t5_perf.txt` | `query_wan_t5` | `op_name/batch` 精确，shape 最近邻 |
| `wan_clip_perf.txt` | `query_wan_clip` | `op_name/batch` 精确，shape 最近邻 |
| `wan_vae*.txt` | `query_wan_vae*` | model/task/path/stage 或 op 前缀精确，shape 最近邻 |
| `gemm_perf.txt` | `query_gemm` | 直接读取 `main0515` 同级 GEMM 数据 |

## 使用示例

```python
from aiconfigurator.cli.api import cli_estimate

result = cli_estimate(
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    "h100_sxm",
    backend_name="sglang",
    backend_version="0.5.10.post1-wan2.2-main0515",
    database_mode="HYBRID",
    isl=1,
    osl=1,
    batch_size=1,
    tp_size=1,
    video_height=720,
    video_width=1280,
    video_frames=121,
    denoising_steps=50,
    sp_size=4,
    ulysses_degree=2,
    ring_degree=2,
)
print(result.raw)
```

建议默认使用 `HYBRID`。Wan 专用算子优先查实测表；通信固定 empirical；GEMM 表外形状、少量 memory op 或缺失文件会回退到经验/SOL 模型。若目标是新硬件/新 profile 的精确评估，优先补齐 collector，再把 `SILICON` 作为验收手段。

## 当前边界

- 不考虑 `MinimalA2AAttnOp`、`UlyssesAttention_VSA`、SLA/VSA 稀疏路径。
- Scheduler 更新、CPU offload、Cache-DiT 控制逻辑未单独计时。
- Blackwell 可复用同一代码路径，但需要对应 `b200_sxm/sglang/<version>/wan_*.txt` 与 GEMM 数据；通信仍按目标 system YAML 的带宽做 empirical 估算。

## 2026-05-19 并行与 Batch 修正记录

- `run.py` 现在在调用 `cli_estimate()` 前先写入完整 Wan 输入参数；即使运行中报错，日志也会保留 `batch_size/tp_size/sp_size/ulysses_degree/ring_degree` 等关键 debug 信息。
- `run_static()` 对 Wan 结果补充 `sp/sp_size/ulysses_degree/ring_degree/sp_algorithm/wan_batch_model`，并将 `num_total_gpus` 修正为 `tp*pp*dp*sp`；`parallel` 字符串追加 `sp{sp}u{ulysses}r{ring}`。
- Wan 并行合法性增加 32 卡上限检查：`tp*pp*dp*sp <= 32`，且 DiT 要求 `num_heads % tp == 0`、`(num_heads/tp) % ulysses_degree == 0`。
- `batch_size` 当前按 `serial_single_video_collect` 处理：collector 只采单视频 kernel，SDK 将整条 Wan pipeline latency/energy 乘以 batch，以保守近似多视频请求；日志会显式 warning。若未来要模拟真实 batched video kernel，需要新增 batch>1 的 Wan collector 数据并切换为 batched shape 查询。
- `PerfDatabase` 对 Wan `rope/attention/elementwise` 放宽为同语义前缀最近邻兜底，并保留 warning 输出，避免 HYBRID 在扩展 SP=16/32 但旧数据未补采时直接失败。
- VAE `avg_down3d` 与 decode 起始 height/width 对齐 collector 的偶数空间维度逻辑，避免 SGLang `AvgDown3D` 仅 pad 时间维导致的奇偶 mismatch。
