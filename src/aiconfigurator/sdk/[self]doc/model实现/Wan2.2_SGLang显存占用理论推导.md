# Wan2.2 SGLang 显存占用理论推导

## 1. 范围与结论摘要

本文基于 SGLang `0.5.12` 容器源码和本地 `Wan2.2-TI2V-5B-Diffusers` checkpoint，推导 Wan2.2 推理时的单卡显存。数值分析以此前性能校准使用的 TI2V-5B、`704x1280`、`121` 帧、batch 1 为主；公式可通过替换模型维度和 token 数复用于 A14B。

显存应拆成以下几部分，而不能只用“参数量乘 dtype 字节数”表示：

```text
M_peak = M_resident_weights
       + max(M_text_encoder_runtime,
             M_DiT_runtime,
             M_VAE_runtime)
       + M_CUDA_context_and_libraries
       + M_allocator_fragmentation
       + M_backend_workspace
```

这里使用 `max` 是因为 text encoder、DiT denoising 和 VAE 通常分阶段执行；仅当组件或阶段异步重叠时才应对相应工作集求和。SGLang 的 offload 策略还会改变 `M_resident_weights`。

核心结论如下：

- TI2V-5B 在 SGLang 默认精度下，单卡、全部组件常驻的权重约为 `33.10 GiB`：DiT `9.31 GiB`、UMT5 `21.16 GiB`、VAE `2.63 GiB`。
- TP 会切 DiT 和 UMT5 的大矩阵，但不会切 VAE；Ulysses/Ring 不切 DiT 权重。SGLang 在 `TP=1, SP>1` 时会把 UMT5 的并行组折叠到 SP group，因此 UMT5 仍可被 SP 卡分片。
- 对 TI2V-5B DiT，精确的每卡参数量是 `4,997,800,128 / T + 1,987,584`，不是简单的 `4,999,787,712 / T`。
- SP 会缩小 residual、FFN 和块内 timestep modulation；纯 TP 不切 residual 序列维度。相同 4 卡下，TP4 与 SP4 的 QKV/FFN 局部元素数相同，但 TP4 的 residual 和块内 timestep modulation 是 SP4 的 4 倍。
- Ulysses AllToAll 前后总元素数不变，主要增加 contiguous 和通信输出临时缓冲。SGLang 0.5.12 的 Ring backend 默认不是逐跳 P2P，而是对拼接后的 KV 做一次 AllGather；其聚合 KV 缓冲与 Ring degree `R` 成正比。
- VAE 的 feature-cache 解码避免了把 121 帧所有高分辨率中间特征同时保留，但完整 FP32 视频输出仍会在每卡累积；反复 `torch.cat` 在末尾可短暂同时持有新旧两份大输出。
- SGLang 0.5.12 有组件权重统计、allocated/reserved/peak runtime 采样和 stage profiler，但没有直接根据 Wan shape、并行策略和张量生命周期静态推算峰值显存的完整组件。

## 2. 符号、配置与单位

### 2.1 通用符号

| 符号 | 含义 |
|---|---|
| `B` | 单卡参与计算的有效 batch；若 CFG 分支同卡合批，应计入有效 batch |
| `T` | tensor parallel degree |
| `U` | Ulysses degree |
| `R` | Ring degree |
| `S=U*R` | sequence parallel degree |
| `G` | UMT5 实际 tensor-parallel/folding group size |
| `D` | DiT hidden size |
| `D_ff` | DiT FFN hidden size |
| `H` | attention head 数 |
| `d=D/H` | head dimension |
| `L` | Transformer block 数 |
| `N` | 全局 DiT token 数，若不能整除 SP 则使用 padding 后的 `N_tilde` |
| `C` | text token 数，Wan pipeline 固定补齐到 512 |
| `b_x` | 对应张量 dtype 的每元素字节数 |

本文统一使用二进制单位：`1 MiB=2^20 bytes`，`1 GiB=2^30 bytes`。

### 2.2 TI2V-5B 模型与输入

本地 checkpoint 配置为：

| 项目 | 数值 |
|---|---:|
| DiT hidden `D` | 3072 |
| block 数 `L` | 30 |
| head 数 `H` | 24 |
| head dim `d` | 128 |
| FFN hidden `D_ff` | 14336 |
| patch input/output channel | 48 / 48 |
| patch size | `(1,2,2)` |
| text input dim / text length | 4096 / 512 |
| VAE latent channel | 48 |
| VAE temporal/spatial compression | `4 / 16` |

代表性请求为 `B=1, F=121, H_img=704, W_img=1280`：

```text
latent grid = (31, 44, 80)
DiT grid    = (31, 22, 40)
N           = 31 * 22 * 40 = 27,280
```

若 `N` 不能整除 `S`，源码会补齐序列，因此公式中应替换为：

```text
N_tilde = ceil(N / S) * S
n       = N_tilde / S                 # 每卡 block 外部 local sequence
```

## 3. 权重显存

### 3.1 checkpoint dtype 不等于 runtime dtype

对 safetensors header 的精确统计如下：

| 组件 | 参数量 | checkpoint dtype | checkpoint 大小 | SGLang 0.5.12 runtime dtype |
|---|---:|---|---:|---|
| DiT | 4,999,787,712 | FP32 | 18.6257 GiB | BF16 |
| UMT5 encoder | 5,680,910,336 | BF16 | 10.5815 GiB | FP32 |
| VAE | 704,688,668 | FP32 | 2.6252 GiB | FP32 |

因此磁盘文件大小不能直接当作 VRAM：DiT 从 FP32 转为 BF16，而 UMT5 默认从 BF16 转为 FP32。

### 3.2 DiT 精确参数公式

TI2V-5B 单个 `WanTransformerBlock` 在 `T=1` 时有 `163,656,704` 个参数。两组 attention 分别是 self-attention 和 cross-attention。

单个 attention 的参数量为：

```text
P_attn = Q/K/V weights + biases + output weight/bias + Q/K RMSNorm
       = 4D^2 + 6D
       = 37,767,168
```

FFN 参数量为：

```text
P_ffn = 2 * D * D_ff + D_ff + D
      = 88,097,792
```

再加上 self-attention residual norm 的 `2D` 和 block modulation table 的 `6D`：

```text
P_block = 2P_attn + P_ffn + 2D + 6D
        = 163,656,704
```

TP 下，Column/Row Parallel Linear 的大矩阵与部分 bias 被切分；output bias、RMSNorm、residual norm 和 modulation table 仍复制。每卡 block 参数量为：

```text
P_block,rank(T)
  = (8D^2 + 2D*D_ff + 6D + D_ff) / T + 15D
  = 163,610,624 / T + 46,080
```

非 block 部分包括 patch embedding、text/time embedding、time modulation、output projection 和 output modulation。将其同样按 SGLang parallel Linear 的存储方式拆分后，TI2V-5B 的 DiT 每卡精确参数量是：

```text
P_DiT,rank(T)
  = P_sharded / T + P_replicated
  = 4,997,800,128 / T + 1,987,584
```

| `T` | 每卡参数量 | BF16 权重 |
|---:|---:|---:|
| 1 | 4,999,787,712 | 9.3128 GiB |
| 2 | 2,500,887,648 | 4.6583 GiB |
| 4 | 1,251,437,616 | 2.3310 GiB |

Ulysses 和 Ring 只改变 attention 的张量布局和通信，不改变上述 DiT 权重公式；同一 TP shard 会在不同 SP rank 上复制。

### 3.3 UMT5 精确参数公式与 parallel folding

UMT5-XXL encoder 配置为 `d_model=4096`、`d_ff=10240`、64 heads、24 layers、vocab 256384。大参数均由 Vocab/QKV/MergedColumn/Row Parallel 层切分，RMSNorm 复制。

```text
P_T5,sharded
  = vocab * d_model
  + L_t5 * (4*d_model^2 + 3*d_model*d_ff + buckets*heads)
  = 5,680,709,632

P_T5,replicated
  = (2*L_t5 + 1) * d_model
  = 200,704

P_T5,rank(G)
  = 5,680,709,632 / G + 200,704
```

SGLang 0.5.12 对 `G` 的选择不是始终等于总 GPU 数：

```text
G = S,  if T == 1 and S > 1   # server_args 自动启用 parallel_folding=sp
G = T,  otherwise             # 常规 TP group
```

| `G` | 每卡参数量 | FP32 权重 |
|---:|---:|---:|
| 1 | 5,680,910,336 | 21.1630 GiB |
| 2 | 2,840,555,520 | 10.5819 GiB |
| 4 | 1,420,378,112 | 5.2913 GiB |

这里的 SP folding 是 text encoder 的额外实现行为，不代表 Ulysses/Ring 能切 DiT 权重。

### 3.4 VAE 权重

TI2V-5B VAE 同时加载 encoder 和 decoder，共 `704,688,668` 个 FP32 参数：

```text
M_VAE,weights = 704,688,668 * 4 = 2.6252 GiB / GPU
```

Wan VAE 的并行 decode/encode 按 SP group 切 height，但 Conv/Norm 权重未做 TP/SP shard，因此每个 worker 都保留完整 VAE 权重。

### 3.5 典型策略的全常驻权重

下表假设禁用 DiT layerwise offload、text encoder CPU offload 和 VAE CPU offload。

| 总卡数与策略 | DiT/卡 | UMT5/卡 | VAE/卡 | 合计/卡 |
|---|---:|---:|---:|---:|
| 1 卡：T1/S1 | 9.3128 | 21.1630 | 2.6252 | 33.1010 GiB |
| 2 卡：T2/S1 | 4.6583 | 10.5819 | 2.6252 | 17.8654 GiB |
| 2 卡：T1/S2 | 9.3128 | 10.5819 | 2.6252 | 22.5199 GiB |
| 4 卡：T4/S1 | 2.3310 | 5.2913 | 2.6252 | 10.2475 GiB |
| 4 卡：T2/S2 | 4.6583 | 10.5819 | 2.6252 | 17.8654 GiB |
| 4 卡：T1/S4，任意 U/R 分解 | 9.3128 | 5.2913 | 2.6252 | 17.2293 GiB |

纯 TP 的每卡权重更小，主要因为它切分了 DiT；纯 SP 只通过 T5 folding 切分 text encoder，DiT 和 VAE 仍完整复制。

### 3.6 A14B 双专家边界

Wan2.2 T2V/I2V A14B 使用 high-noise/low-noise 两套 DiT（checkpoint 中是 `transformer` 和 `transformer_2`），由 boundary timestep 选择其中一个执行。运行时激活不会因为两个 expert 而同时翻倍，但若两者同时常驻，DiT 权重需要对两套 expert 求和：

```text
M_A14B_DiT,resident
  = b_w * sum_e(P_sharded,e / T + P_replicated,e)
```

对每个普通 Wan block，可复用：

```text
P_block,rank(T)
  = (8D^2 + 2D*D_ff + 6D + D_ff) / T + 15D
```

A14B 的 `D=5120`、`D_ff=13824`、`L=40`。I2V 若存在 added image K/V projection，还要把对应 projection 与 norm 参数加入每 block 公式。实际部署还应确认 SGLang 的 component residency/offload 是否只让当前 expert 留在 GPU，不能只看一次 forward 中使用了一个 expert 就假定另一套权重不占显存。

## 4. 运行时激活与通信工作集

### 4.1 峰值不是所有层和 timestep 的总和

推理无反向图，30 个 block 和多个 denoising step 通常复用 allocator buffer。因此：

```text
peak activation != num_layers * per_layer activation
peak activation != denoising_steps * per_step activation
```

层数和 denoising steps 主要放大时间。只有 TeaCache/Cache-DiT、保留中间结果、异步 prefetch、CUDA Graph private pool 等机制才会让额外状态跨层或跨 step 常驻。

### 4.2 Text encoder 激活

UMT5 默认 FP32，序列长 `C=512`。每层主要张量可近似为：

```text
M_hidden      = B * C * d_model * 4
M_QKV         = 3 * B * C * d_model/G * 4
M_gated_FFN   = 2 * B * C * d_ff/G * 4
M_attn_score  = B * (heads/G) * C^2 * 4
```

softmax 的 FP32 logits、relative bias、mask 和输出可能在短时间内共存，所以 attention peak 应用 liveness factor 校准，而不能只取一个 `M_attn_score`。层间 buffer 可复用，24 层不乘到峰值上。以 `B=1,G=4` 为例，上述单项分别约 `8 MiB`、`6 MiB`、`10 MiB`、`16 MiB`，明显小于 UMT5 FP32 权重。

### 4.3 DiT 基础张量

令：

```text
X = B * N_tilde * D / S          # local residual 元素数
A = B * N_tilde * D / (T*S)      # 单个 local Q/K/V 元素数
F = B * N_tilde * D_ff / (T*S)   # local FFN intermediate 元素数
```

对应基础工作集：

```text
M_residual = X * b_dit
M_QKV      = 3A * b_dit
M_FFN      = F * b_dit
```

关键点是 residual 不按 TP 切 hidden，故只除以 `S`；QKV/FFN 同时受 TP 和 SP 影响。RowParallel 输出还可能在 AllReduce 期间额外持有约一个 `X*b_dit` 的通信输出或 staging buffer。

cross-attention 的 query 仍来自 local video sequence，K/V 则来自长度 512 的 context：

```text
M_cross_Q  = A * b_dit
M_cross_KV = 2 * B * C * D/T * b_dit
```

context 远短于视频 token，因此 self-attention 和 timestep modulation 通常更重要。

### 4.4 TI2V 展开 timestep modulation

TI2V-5B 使用 per-token timestep。SGLang 先对全局 timestep 生成：

```text
temb_global          : [B, N_tilde, D]
timestep_proj_global : [B, N_tilde, 6, D]
```

`timestep_proj_global` 在生成后才按 SP sequence 分片，TP projection 因 `gather_output=True` 也会恢复完整 `6D` 输出。因此所有策略都会经历一次全局 transient：

```text
M_timestep_proj,global = B * N_tilde * 6D * b_dit
```

进入 block 后，local timestep projection 是：

```text
M_timestep_proj,local = B * (N_tilde/S) * 6D * b_dit
```

更重要的是 `WanTransformerBlock.forward()` 中执行了 `temb.float()`，并与 FP32 `scale_shift_table` 相加。因此每个 block 内会出现近似：

```text
M_timestep_mod_fp32 = B * (N_tilde/S) * 6D * 4
```

这部分只被 SP 缩小，不被 TP 缩小。代表性请求中：

| 项目 | S1 | S4 |
|---|---:|---:|
| local BF16 timestep projection | 959.06 MiB | 239.77 MiB |
| block 内 FP32 modulation | 1918.13 MiB | 479.53 MiB |

此外，未分片的 `temb_global` 仍用于最后 output norm，约 `159.84 MiB` BF16。精确峰值取决于编译器是否融合加法/切分、旧 tensor 何时释放以及 allocator 是否复用 storage。

### 4.5 Ulysses AllToAll

进入 USP 前，单个 Q/K/V 的 shape 是：

```text
[B, N_tilde/S, H/T, d]
```

Ulysses input AllToAll 后变为：

```text
[B, N_tilde/R, H/(T*U), d]
```

两者元素数均为 `A`。所以 Ulysses 不增加稳态 Q/K/V 元素数，但 `permute -> contiguous -> AllToAll output -> permute/contiguous` 会产生临时张量。单个正在通信的张量可用以下工程包络：

```text
M_one_A2A_extra ~= (1 to 2) * A * b_dit + M_NCCL_workspace
M_Ulysses_QKV_working_set ~= (4 to 5) * A * b_dit + M_NCCL_workspace
```

Q/K/V 和 output AllToAll 的真实生命周期有先后，不应把四次通信的全部 buffer 无条件相加。

### 4.6 Ring AllGather 与 overlap

SGLang 0.5.12 最终调用 PyTorch context-parallel attention，默认 rotate method 为 `ALL_GATHER`。实现先拼接 local K/V，再执行一次 `all_gather_tensor`，不是把 KV 逐 hop P2P 旋转：

```text
local concatenated KV = 2A elements
gathered KV           = 2R*A elements
```

首个 FlashAttention 与 AllGather 异步 overlap 会影响时间线先后，但不会让 AllGather 输出缓冲消失。PyTorch 该路径的默认 `convert_to_f32=True`，`_SDPAMerger` 还会将 partial attention output 转成 FP32 后累计。先把 resident QKV、local KV concat、gathered KV 和一次 BF16 attention output 写成基础工作集：

```text
M_Ring_base ~= (6 + 2R) * A * b_dit
```

再加入 FP32 merger：

```text
M_Ring_attention
  ~= M_Ring_base
   + k_merge * A * 4
   + M_LSE
   + M_FA_workspace
   + M_NCCL_workspace

k_merge ~= 1 to 3
```

`k_merge=1` 只表示持久 FP32 output accumulator；实际 merge 表达式还可能同时保留当前 block output 的 FP32 copy 和一个或多个 elementwise intermediate，因此上界应由 memory trace 校准。

其中 merge 所需 LSE/metadata 远小于视频 QKV 主张量，可近似为：

```text
M_LSE ~= 4 * B * (N_tilde/R) * H/(T*U) bytes
```

该式是便于 AIC 建模的 working-set envelope，不是 PyTorch allocator 的逐字节恒等式。尤其 overlap 时，通信 stream 和 compute stream 的释放时点会改变真实峰值。

### 4.7 4 卡代表性数值

以下四种策略均满足 `T*S=4`，因此单个 Q/K/V 和 FFN intermediate 的局部大小相同。DiT dtype 按 BF16，block modulation 按源码中的 FP32 transient。

| 策略 | residual | QKV 合计 | FFN intermediate | block FP32 timestep modulation | Ulysses 额外临时量 | Ring gathered KV |
|---|---:|---:|---:|---:|---:|---:|
| TP4/U1/R1 | 159.84 MiB | 119.88 MiB | 186.48 MiB | 1918.13 MiB | - | - |
| TP1/U4/R1 | 39.96 MiB | 119.88 MiB | 186.48 MiB | 479.53 MiB | 39.96-79.92 MiB/活动张量 | - |
| TP1/U2/R2 | 39.96 MiB | 119.88 MiB | 186.48 MiB | 479.53 MiB | 39.96-79.92 MiB/活动张量 | 159.84 MiB |
| TP1/U1/R4 | 39.96 MiB | 119.88 MiB | 186.48 MiB | 479.53 MiB | - | 319.69 MiB |

Ring 基础工作集和计入 FP32 merger 后的工程范围约为：

| 策略 | BF16/通信基础 `(6+2R)A*b_dit` | 加 FP32 merger，`k_merge=1..3` |
|---|---:|---:|
| U2/R2 | 399.61 MiB | 479.53-639.38 MiB |
| U1/R4 | 559.45 MiB | 639.38-799.22 MiB |

这些值不能直接全部相加作为 DiT 精确峰值，因为 FFN 与 self-attention 不同时执行，且 modulation、residual、QKV 的实际 storage liveness 需由 trace 或 memory snapshot 校准。它们的用途是定位随并行策略变化的主项。

## 5. VAE 激活显存

### 5.1 配置与 feature-cache 解码

TI2V-5B VAE 配置为：

```text
base_dim         = 160
decoder_base_dim = 256
z_dim            = 48
dim_mult         = (1, 2, 4, 4)
num_res_blocks   = 2
patch_size       = 2
runtime dtype    = FP32
```

`use_feature_cache=True` 时，decode 按 31 个 latent frame 逐个调用 decoder；causal Conv3D 为每个卷积保存最后 `CACHE_T=2` 个 temporal slices。这样高分辨率 feature 的 temporal 维保持为一个小 chunk，而不是 121 帧，但会引入跨 chunk cache。

任意 VAE feature map 的基础大小为：

```text
M_feature(i) = B * C_i * F_i * H_i * W_i * 4
```

对后续 latent chunk，主要 decoder stage 的近似全局 feature 大小如下。首帧的 temporal shape 会略有不同。

| stage | 近似 shape `(C,F,H,W)` | 单张量 FP32 |
|---|---|---:|
| decoder input | `(1024,1,44,80)` | 13.75 MiB |
| up stage 0 | `(1024,2,88,160)` | 110 MiB |
| up stage 1 | `(1024,4,176,320)` | 880 MiB |
| up stage 2 | `(512,4,352,640)` | 1760 MiB |
| final hidden | `(256,4,352,640)` | 880 MiB |

残差块至少会同时保留 input、main-path output，卷积还可申请 cuDNN workspace，故不能只取表中最大单张量。可写成：

```text
M_VAE_internal
  ~= k_res * max_i(M_feature(i)/P_vae)
   + M_feature_cache/P_vae
   + M_conv_workspace
   + M_halo_and_collective
```

`k_res` 通常至少为 2，复杂 residual/upsample liveness 下可达到 3-4；应通过 runtime peak 校准。`P_vae` 是 VAE 实际 height-parallel group size，在当前 Wan 实现中对应 SP world size，而不是 TP size。

feature cache 可按每个 causal conv 的输入分辨率求和：

```text
M_feature_cache ~= sum_j(2 * B * C_j * H_j * W_j * 4)
```

分布式 conv 还会因 height padding、halo exchange 和 gather 临时增加 buffer。

### 5.2 完整输出不会被 SP 永久切分

parallel decoder 在返回前调用 `gather_and_trim_height()`，因此每个 rank 都拿到该 chunk 的完整 height。外层 decode 循环通过 `torch.cat([out, out_], dim=2)` 累积完整视频。

最终 FP32 RGB 输出大小为：

```text
M_output
  = B * 3 * 121 * 704 * 1280 * 4
  = 1247.81 MiB
  = 1.2186 GiB
```

接近最后一次 `torch.cat` 时，新 tensor 分配完成前旧累计输出仍存活，因此仅输出拼接就可能短暂接近：

```text
M_cat_peak ~= M_old + M_new_chunk + M_concatenated ~= 2 * M_output
           ~= 2.4371 GiB
```

这个主项不会按 SP 等比例缩小。VAE 峰值应在以下两类时点中取较大者：

```text
M_VAE_runtime ~= max(
    M_accumulated_output + M_VAE_internal,
    M_cat_peak + M_feature_cache + M_collective_buffer
)
```

启用 spatial/temporal tiling 可进一步限制内部 feature，但会引入 tile overlap/blend buffer；关闭 feature cache 则改变 temporal liveness，不能继续使用上述逐 latent-frame 包络。

## 6. Offload、常驻策略与其他不可忽略项

### 6.1 SGLang 0.5.12 默认策略

Wan pipeline 的 deployment config 支持自动 DiT layerwise offload，并在 CUDA 卡显存低于约 `130 GiB` 时可能自动启用；text encoder CPU offload 也会受 server args 自动调整影响。最终行为取决于显式参数、platform 和 auto tuner，因此分析前必须记录解析后的 server args，而不是只看 dataclass 初始值。

此前满载性能校准显式使用：

```text
--dit-layerwise-offload false
--text-encoder-cpu-offload false
--vae-cpu-offload false
```

这对应本文全常驻权重表。

若启用 DiT layerwise offload，每卡 DiT resident weight 可近似为：

```text
P_DiT,resident(T)
  ~= P_nonblock,rank(T) + k_active * P_block,rank(T)
```

`k_active` 是当前计算与 prefetch 同时在 GPU 的 block 数，不是 30。还必须增加 H2D staging/prefetch buffer 和 CUDA stream overlap 带来的双驻留。text encoder/VAE component offload 则只在其 active stage 将对应权重搬回 GPU。

### 6.2 allocator、workspace 与 CUDA Graph

理论 tensor bytes 之外还包括：

- CUDA context、cuBLAS/cuDNN/NCCL library state；
- PyTorch caching allocator 的 reserved-but-unused block 和碎片；
- FlashAttention、Triton、cuDNN convolution 与 collective workspace；
- torch.compile 生成代码与 autotune 临时量；
- CUDA Graph private memory pool，若未来 multimodal-gen 路径启用 graph capture；
- profiler 自身 buffer，采集 trace 时不应当作正常推理显存。

因此需要同时记录 `allocated` 和 `reserved`。OOM 更接近 reserved/可用大块约束，静态 tensor 模型更接近 allocated。

## 7. SGLang 0.5.12 中已有的显存组件

### 7.1 可直接使用的运行时能力

| 能力 | 源码位置 | 含义与限制 |
|---|---|---|
| `get_memory_usage_of_component(module)` | `runtime/loader/utils.py` | 参数和 buffer 的 footprint；不含 activation/workspace |
| `capture_memory_snapshot()` | `runtime/utils/perf_logger.py` | 采集 allocated、reserved、peak allocated、peak reserved |
| `StageProfiler(capture_memory=True)` | `runtime/utils/perf_logger.py` | stage 结束后记录 snapshot；若未在 stage 前 reset peak，不能自动得到该 stage 独占峰值 |
| `gpu_worker` before/after forward | `runtime/managers/gpu_worker.py` | forward 前 reset peak，记录 baseline、after-forward 和 `peak_memory_mb` |
| `do_mem_analysis()` | `runtime/managers/gpu_worker.py` | 对比 peak allocated/reserved，并报告 allocator pool overhead 与剩余显存 |
| offline throughput benchmark | `benchmarks/bench_offline_throughput.py` | 使用 `max_memory_allocated()` 汇总实测峰值 |
| component loader 日志 | `runtime/loader/component_loaders/*` | 可显示组件模型大小和加载时 GPU 增量 |
| component residency manager | `runtime/managers/component_manager.py` | 管理 resident/offload/prefetch 生命周期，不负责静态推导 Wan activation peak |

`gpu_worker` 对外的 `peak_memory_mb` 使用 peak reserved，而 `bench_offline_throughput` 使用 peak allocated。比较结果时必须先统一口径。

### 7.2 未发现的能力

在 SGLang 0.5.12 multimodal-gen 中未发现一个能够输入 Wan model config、视频 shape、dtype、TP/U/R 和 offload 策略后，直接输出理论峰值显存的静态 estimator。已有实现更偏向：

1. 加载时统计组件权重；
2. 请求执行时采样真实 allocator peak；
3. 根据实测剩余显存决定组件是否可以 resident。

LLM SRT 中面向 KV cache/memory pool 的估算不能直接套到 diffusion Wan：Wan 没有逐 token 增长的 decoder KV cache，主峰值来自大视频 activation、per-token modulation、USP communication 和 VAE Conv3D feature。

## 8. 对 AIC 显存建模的建议

建议在现有延迟模型之外增加独立的 memory DAG/liveness 模型，而不是给每个 op 的 bytes 直接求和：

1. **权重层**：记录每个 component 的 `P_sharded/T + P_replicated`、runtime dtype、resident/offload interval；A14B 两套 expert 分开记录。
2. **activation 层**：为 tensor 记录 shape、dtype、producer、last consumer 和 alias/view 关系；以拓扑执行过程求 live bytes 峰值。
3. **通信层**：AllToAll、AllGather、AllReduce 分别定义 source/output/staging buffer；Ring 使用 `2R*A` gathered KV，而不是旧的 P2P hop buffer。
4. **overlap 层**：compute/communication stream 并行时取两条 stream live set 的并集，不能把 overlap 误解为只取两者最大时间或只保留一个 buffer。
5. **allocator 层**：理论 live bytes 之外保留经验系数或 size-class allocator 模型，分别输出 `predicted_allocated` 与 `predicted_reserved`。
6. **校准层**：在 text encoding、单个 DiT block、完整 denoising、VAE decode 前后 reset/采样 peak，并用 `torch.cuda.memory_snapshot()` 或 memory history 验证大 allocation 的栈来源。

建议最终同时输出三个口径：

```text
exact_weight_bytes       # 可由 checkpoint/config 精确得到
activation_live_envelope # 公式和生命周期得到的可解释工作集
calibrated_peak_reserved # 加 workspace、allocator 与实测校准后的部署值
```

## 9. 源码依据

本分析以容器 `clever_jennings` 中 `lmsysorg/sglang:v0.5.12` 的以下源码为准：

- `multimodal_gen/runtime/models/dits/wanvideo.py`：Wan block、per-token timestep、sequence shard 和输出 gather；
- `multimodal_gen/runtime/layers/usp.py`：Ulysses/Ring USP 数据布局；
- PyTorch `torch.distributed.tensor.experimental._context_parallel._attention`：Ring 默认 AllGather rotater；
- `multimodal_gen/runtime/models/encoders/t5.py`：UMT5 parallel Linear、relative bias 和 RMSNorm；
- `multimodal_gen/runtime/distributed/__init__.py` 与 `runtime/server_args.py`：T5 SP parallel folding；
- `multimodal_gen/runtime/models/vaes/wanvae.py`：逐 latent-frame feature cache、parallel height decode 和输出拼接；
- `multimodal_gen/runtime/loader/utils.py`：组件 footprint；
- `multimodal_gen/runtime/utils/perf_logger.py`：memory snapshot 与 stage profiler；
- `multimodal_gen/runtime/managers/gpu_worker.py`：forward peak、reserved/allocated 分析；
- `multimodal_gen/runtime/managers/component_manager.py`：组件 resident/offload 策略。

本地参数和 shape 来源：

- `/mnt/nvme1n1/data/ljc/ai_model/Wan2.2-TI2V-5B-Diffusers/transformer/config.json`
- `/mnt/nvme1n1/data/ljc/ai_model/Wan2.2-TI2V-5B-Diffusers/text_encoder/config.json`
- `/mnt/nvme1n1/data/ljc/ai_model/Wan2.2-TI2V-5B-Diffusers/vae/config.json`
- 各组件 safetensors header 的精确参数量统计。
