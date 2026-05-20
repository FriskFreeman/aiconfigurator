# Wan 模型配置体系分析

## 目录
1. [HF 官方配置文件结构](#hf-官方配置文件结构)
2. [SGLang 配置文件体系](#sglang-配置文件体系)
3. [配置加载与模型构建流程](#配置加载与模型构建流程)
4. [配置字段对应关系](#配置字段对应关系)
5. [推理执行流程](#推理执行流程)

---

## HF 官方配置文件结构

### 概述

HF 官方提供的 Wan 模型配置包含以下几个主要部分，分别对应模型架构的不同组件：

```
Wan2.2 模型配置文件清单：
├── Wan-AI--Wan2.2-T2V-A14B_config.json          # 主 DiT 模型配置（T2V 文本到视频）
├── Wan-AI--Wan2.2-I2V-A14B_config.json          # 主 DiT 模型配置（I2V 图像到视频）
├── Wan-AI--Wan2.2-TI2V-5B_config.json           # 主 DiT 模型配置（TI2V 图像+文本到视频）
├── Wan-AI--Wan2.2-T2V-A14B-TextEncoder_config.json  # 文本编码器配置（T5 模型）
├── Wan-AI--Wan2.2-T2V-A14B-VAE_config.json     # VAE 编码器配置
└── Wan-AI--Wan2.2-T2V-A14B-Scheduler_config.json # 调度器配置
```

### 1. DiT (Diffusion Transformer) 配置

#### T2V A14B 配置示例

```json
{
  "_class_name": "WanTransformer3DModel",
  "_diffusers_version": "0.35.0.dev0",
  "num_attention_heads": 40,
  "attention_head_dim": 128,
  "in_channels": 16,
  "out_channels": 16,
  "text_dim": 4096,
  "freq_dim": 256,
  "ffn_dim": 13824,
  "num_layers": 40,
  "patch_size": [1, 2, 2],
  "rope_max_seq_len": 1024,
  "cross_attn_norm": true,
  "qk_norm": "rms_norm_across_heads",
  "eps": 1e-6
}
```

**关键参数说明：**

| 参数 | 类型 | 说明 | Wan2.2-T2V-A14B 值 |
|------|------|------|------------------|
| `num_attention_heads` | int | 多头注意力的头数 | 40 |
| `attention_head_dim` | int | 每个注意力头的维度 | 128 |
| `num_layers` | int | Transformer 层数 | 40 |
| `in_channels` | int | 输入通道数（VAE 潜在空间） | 16 |
| `out_channels` | int | 输出通道数 | 16 |
| `text_dim` | int | 文本编码维度（来自文本编码器） | 4096 |
| `freq_dim` | int | 时间步频率嵌入维度 | 256 |
| `ffn_dim` | int | FFN 隐藏层维度 | 13824 |
| `patch_size` | [int, int, int] | 3D 补丁大小 [T, H, W] | [1, 2, 2] |
| `rope_max_seq_len` | int | RoPE 最大序列长度 | 1024 |
| `cross_attn_norm` | bool | 跨注意力规范化 | true |
| `qk_norm` | str | Q/K 规范化方式 | "rms_norm_across_heads" |

**隐含参数：**

```
hidden_size = num_attention_heads * attention_head_dim 
           = 40 * 128 = 5120
```

#### I2V A14B 配置

```json
{
  "_class_name": "WanModel",
  "model_type": "i2v",
  "dim": 5120,
  "num_heads": 40,
  "num_layers": 40,
  "ffn_dim": 13824,
  "freq_dim": 256,
  "in_dim": 36,     // I2V 特有：输入维度（图像特征）
  "out_dim": 16,    // I2V 特有：输出维度
  "text_len": 512   // 文本最大长度
}
```

#### TI2V 5B 配置

```json
{
  "_class_name": "WanModel",
  "model_type": "ti2v",
  "dim": 3072,
  "num_heads": 24,
  "num_layers": 30,
  "ffn_dim": 14336,
  "freq_dim": 256,
  "in_dim": 48,
  "out_dim": 48,
  "text_len": 512
}
```

**对比表：**

| 参数 | T2V-A14B | I2V-A14B | TI2V-5B |
|------|----------|----------|---------|
| 隐藏维度 (dim) | 5120 | 5120 | 3072 |
| 注意力头数 | 40 | 40 | 24 |
| 层数 | 40 | 40 | 30 |
| FFN 维度 | 13824 | 13824 | 14336 |
| 频率维度 | 256 | 256 | 256 |

### 2. 文本编码器配置

```json
{
  "_name_or_path": "google/umt5-xxl",
  "architectures": ["UMT5EncoderModel"],
  "d_model": 4096,
  "num_heads": 64,
  "num_layers": 24,
  "d_ff": 10240,
  "d_kv": 64,
  "feed_forward_proj": "gated-gelu",
  "relative_attention_num_buckets": 32,
  "relative_attention_max_distance": 128,
  "dropout_rate": 0.1,
  "torch_dtype": "bfloat16"
}
```

**关键参数：**
- `d_model`: 4096 - 输出嵌入维度（与 DiT 的 `text_dim` 匹配）
- `num_layers`: 24 - 编码层数
- 该配置直接对应 HuggingFace 的 T5 模型架构

### 3. VAE 配置

```json
{
  "_class_name": "AutoencoderKLWan",
  "base_dim": 96,
  "dim_mult": [1, 2, 4, 4],
  "num_res_blocks": 2,
  "z_dim": 16,           // 潜在空间维度
  "latents_mean": [...],  // 16个值的规范化均值
  "latents_std": [...],   // 16个值的规范化标准差
  "temperal_downsample": [false, true, true],
  "scale_factor_temporal": 4,
  "scale_factor_spatial": 8
}
```

**关键参数：**
- `z_dim`: 16 - 与 DiT 的 `in_channels`/`out_channels` 匹配
- `scale_factor_temporal`: 4 - 时间维度压缩率
- `scale_factor_spatial`: 8 - 空间维度压缩率
- 规范化参数（latents_mean/std）用于标准化潜在表示

### 4. 调度器配置

```json
{
  "_class_name": "UniPCMultistepScheduler",
  "num_train_timesteps": 1000,
  "beta_start": 0.0001,
  "beta_end": 0.02,
  "beta_schedule": "linear",
  "prediction_type": "flow_prediction",
  "flow_shift": 3.0,
  "solver_type": "bh2",
  "solver_order": 2,
  "timestep_spacing": "linspace"
}
```

---

## SGLang 配置文件体系

### 文件位置

```
sglang/multimodal_gen/
├── configs/
│   ├── models/
│   │   ├── dits/
│   │   │   ├── base.py              # DiT 基类配置
│   │   │   └── wanvideo.py          # Wan 特定配置
│   │   ├── vaes/
│   │   │   ├── base.py              # VAE 基类配置
│   │   │   └── wanvae.py            # Wan 特定配置
│   │   └── encoders/                # 编码器配置
│   ├── pipeline_configs/
│   │   └── wan.py                   # Wan Pipeline 配置
│   ├── backend/
│   │   └── vmoba/                   # 后端优化配置
│   │       ├── wan_1.3B_77_480_832.json
│   │       └── ...
│   └── sample/
│       └── wan.py                   # 采样和推理参数
├── runtime/
│   ├── models/
│   │   ├── dits/wanvideo.py         # 模型实现
│   │   └── vaes/wanvae.py           # VAE 实现
│   └── pipelines/
│       ├── wan_pipeline.py
│       └── wan_dmd_pipeline.py
└── test/
    └── test_files/launch_wan.json   # 测试启动配置
```

### 1. DiT 配置 (wanvideo.py)

**关键类：`WanVideoArchConfig` 和 `WanVideoConfig`**

```python
@dataclass
class WanVideoArchConfig(DiTArchConfig):
    # 参数映射：从 HF 权重名称到 SGLang 内部名称
    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^patch_embedding\.(.*)$": r"patch_embedding.proj.\1",
            r"^blocks\.(\d+)\.attn1\.to_q\.(.*)$": r"blocks.\1.to_q.\2",
            # ... 其他映射
        }
    )
    
    # 架构参数
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_len: int = 512
    num_attention_heads: int = 40
    attention_head_dim: int = 128
    in_channels: int = 16
    out_channels: int = 16
    text_dim: int = 4096
    freq_dim: int = 256
    ffn_dim: int = 13824
    num_layers: int = 40
    cross_attn_norm: bool = True
    qk_norm: str = "rms_norm_across_heads"
    eps: float = 1e-6
    
    # MoE 相关参数
    boundary_ratio: float | None = None
    
    # 因果注意力参数
    local_attn_size: int = -1
    sink_size: int = 0
    num_frames_per_block: int = 3
    sliding_window_num_frames: int = 21
    attention_type: str = "original"
    sla_topk: float = 0.1
```

**配置继承：**
```
DiTArchConfig (base.py)
    ↓
WanVideoArchConfig
    ├── param_names_mapping    # 权重名映射
    ├── lora_param_names_mapping
    └── 架构参数...
```

### 2. VAE 配置 (wanvae.py)

**关键类：`WanVAEArchConfig` 和 `WanVAEConfig`**

```python
@dataclass
class WanVAEArchConfig(VAEArchConfig):
    base_dim: int = 96
    z_dim: int = 16
    dim_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    temperal_downsample: tuple[bool, ...] = (False, True, True)
    
    # 规范化参数
    latents_mean: tuple[float, ...] = (...)  # 16个值
    latents_std: tuple[float, ...] = (...)   # 16个值
    
    scale_factor_temporal: int = 4
    scale_factor_spatial: int = 8

@dataclass
class WanVAEConfig(VAEConfig):
    arch_config: WanVAEArchConfig = field(default_factory=WanVAEArchConfig)
    
    # VAE 特定参数
    use_feature_cache: bool = True
    use_tiling: bool = False
    use_temporal_tiling: bool = False
    use_parallel_tiling: bool = False
    
    use_parallel_encode: bool = True
    use_parallel_decode: bool = True
```

### 3. Pipeline 配置 (wan.py)

**关键类定义：**

```python
@dataclass
class WanT2V480PConfig(PipelineConfig):
    # DiT 配置
    dit_config: DiTConfig = field(default_factory=WanVideoConfig)
    
    # VAE 配置
    vae_config: VAEConfig = field(default_factory=WanVAEConfig)
    vae_tiling: bool = False
    vae_sp: bool = False
    
    # 去噪参数
    flow_shift: float | None = 3.0
    
    # 文本编码
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (T5Config(),)
    )
    
    # 精度设置
    precision: str = "bf16"
    vae_precision: str = "fp32"
    text_encoder_precisions: tuple[str, ...] = ("fp32",)

# 特定模型配置
@dataclass
class WanT2V720PConfig(WanT2V480PConfig):
    flow_shift: float | None = 5.0

@dataclass
class Wan2_2_T2V_A14B_Config(WanT2V480PConfig):
    """Wan 2.2 T2V A14B 配置"""
    # 继承父类配置

@dataclass
class Wan2_2_I2V_A14B_Config(WanI2V720PConfig):
    """Wan 2.2 I2V A14B 配置"""
    # 继承父类配置

@dataclass
class Wan2_2_TI2V_5B_Config(WanT2V480PConfig, WanI2VCommonConfig):
    """Wan 2.2 TI2V 5B 配置"""
    # 继承父类配置
```

### 4. 采样参数配置 (sample/wan.py)

```python
@dataclass
class WanT2V_14B_SamplingParams(SamplingParams):
    # 视频参数
    height: int = 720
    width: int = 1280
    num_frames: int = 81
    fps: int = 16
    
    # 去噪参数
    guidance_scale: float = 5.0
    num_inference_steps: int = 50
    negative_prompt: str = "..."
    
    # TeaCache 加速参数
    teacache_params: TeaCacheParams = field(
        default_factory=lambda: TeaCacheParams(
            teacache_thresh=0.08,
            use_ret_steps=True,
            coefficients_callback=_wan_14b_coefficients,
            start_skipping=5,
            end_skipping=1.0,
        )
    )
```

### 5. 后端优化配置 (backend/vmoba/*.json)

**Vmoba 注意力优化配置示例：**

```json
{
    "temporal_chunk_size": 2,
    "temporal_topk": 3,
    "spatial_chunk_size": [3, 4],
    "spatial_topk": 20,
    "st_chunk_size": [4, 6, 4],
    "st_topk": 15,
    "moba_select_mode": "threshold",
    "moba_threshold": 0.25,
    "moba_threshold_type": "query_head",
    "first_full_layer": 0,
    "first_full_step": 12,
    "temporal_layer": 1,
    "spatial_layer": 1,
    "st_layer": 1
}
```

**参数说明：**
- `temporal_chunk_size`/`spatial_chunk_size`：块大小，用于降低注意力复杂度
- `temporal_topk`/`spatial_topk`：Top-K 选择
- `moba_threshold`：注意力矩阵值的阈值
- `first_full_layer`/`first_full_step`：何时开始使用完整注意力

### 6. 启动配置 (test_files/launch_wan.json)

```json
{
    "model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "prompt": "A beautiful woman in a red dress walking down a street",
    "text_encoder_cpu_offload": true,
    "pin_cpu_memory": true,
    "save_output": true,
    "width": 720,
    "height": 720,
    "output_path": "outputs",
    "output_file_name": "Wan2.1-T2V-1.3B-Diffusers, single gpu"
}
```

---

## 配置加载与模型构建流程

### 配置加载流程

```
启动 SGLang 服务
    ↓
ServerArgs 解析命令行参数
    ↓
从 model_path 加载 HF 模型配置
    ├─ config.json (HF 官方配置)
    └─ 匹配到对应的 SGLang Pipeline 配置类
    ↓
Pipeline 初始化
    ├─ DiT 配置加载 (WanVideoConfig)
    ├─ VAE 配置加载 (WanVAEConfig)
    ├─ 文本编码器配置加载 (T5Config)
    └─ 调度器初始化 (UniPCMultistepScheduler)
    ↓
权重加载与映射
    ├─ HF 权重名映射到 SGLang 内部名称
    ├─ 应用 param_names_mapping
    └─ 加载权重到模型
    ↓
模型初始化完成
```

### 关键映射规则

#### 权重名称映射 (param_names_mapping)

```python
# HF 格式 → SGLang 格式
{
    # Patch Embedding
    r"^patch_embedding\.(.*)$": r"patch_embedding.proj.\1",
    
    # Condition Embedder
    r"^condition_embedder\.text_embedder\.linear_1\.(.*)$": 
        r"condition_embedder.text_embedder.fc_in.\1",
    r"^condition_embedder\.text_embedder\.linear_2\.(.*)$": 
        r"condition_embedder.text_embedder.fc_out.\1",
    
    # Attention
    r"^blocks\.(\d+)\.attn1\.to_q\.(.*)$": r"blocks.\1.to_q.\2",
    r"^blocks\.(\d+)\.attn1\.to_k\.(.*)$": r"blocks.\1.to_k.\2",
    r"^blocks\.(\d+)\.attn1\.to_v\.(.*)$": r"blocks.\1.to_v.\2",
    r"^blocks\.(\d+)\.attn1\.to_out\.0\.(.*)$": r"blocks.\1.to_out.\2",
    
    # FFN
    r"^blocks\.(\d+)\.ffn\.net\.0\.proj\.(.*)$": r"blocks.\1.ffn.fc_in.\2",
    r"^blocks\.(\d+)\.ffn\.net\.2\.(.*)$": r"blocks.\1.ffn.fc_out.\2",
}
```

### 配置加载优先级

1. **命令行参数** (ServerArgs) - 最高优先级
2. **Pipeline 配置类** (如 Wan2_2_T2V_A14B_Config)
3. **HF 官方 config.json** - 架构参数来源
4. **默认值** (DiTArchConfig, VAEArchConfig) - 最低优先级

---

## 配置字段对应关系

### 核心对应表

| HF config.json 字段 | HF 值 (A14B) | SGLang 配置类 | SGLang 属性 | 说明 |
|---|---|---|---|---|
| `num_attention_heads` | 40 | WanVideoArchConfig | `num_attention_heads` | 直接对应 |
| `attention_head_dim` | 128 | WanVideoArchConfig | `attention_head_dim` | 直接对应 |
| `num_layers` | 40 | WanVideoArchConfig | `num_layers` | 直接对应 |
| `in_channels` | 16 | WanVideoArchConfig | `in_channels` | VAE 潜在维度 |
| `out_channels` | 16 | WanVideoArchConfig | `out_channels` | 输出维度 |
| `text_dim` | 4096 | WanVideoArchConfig | `text_dim` | 文本编码器输出维度 |
| `freq_dim` | 256 | WanVideoArchConfig | `freq_dim` | 时间步嵌入维度 |
| `ffn_dim` | 13824 | WanVideoArchConfig | `ffn_dim` | FFN 隐藏层维度 |
| `patch_size` | [1, 2, 2] | WanVideoArchConfig | `patch_size` | 补丁大小 |
| `rope_max_seq_len` | 1024 | WanVideoArchConfig | `rope_max_seq_len` | RoPE 最大长度 |
| `cross_attn_norm` | true | WanVideoArchConfig | `cross_attn_norm` | 跨注意力规范化 |
| `qk_norm` | "rms_norm_across_heads" | WanVideoArchConfig | `qk_norm` | Q/K 规范化类型 |
| `eps` | 1e-6 | WanVideoArchConfig | `eps` | LayerNorm epsilon |

### VAE 配置对应表

| HF config.json 字段 | HF 值 | SGLang 配置类 | SGLang 属性 |
|---|---|---|---|
| `base_dim` | 96 | WanVAEArchConfig | `base_dim` |
| `z_dim` | 16 | WanVAEArchConfig | `z_dim` |
| `dim_mult` | [1, 2, 4, 4] | WanVAEArchConfig | `dim_mult` |
| `num_res_blocks` | 2 | WanVAEArchConfig | `num_res_blocks` |
| `latents_mean` | [...] | WanVAEArchConfig | `latents_mean` |
| `latents_std` | [...] | WanVAEArchConfig | `latents_std` |
| `temperal_downsample` | [F, T, T] | WanVAEArchConfig | `temperal_downsample` |
| `scale_factor_temporal` | 4 | WanVAEArchConfig | `scale_factor_temporal` |
| `scale_factor_spatial` | 8 | WanVAEArchConfig | `scale_factor_spatial` |

### 文本编码器配置对应表

| HF config.json 字段 | HF 值 (T5) | SGLang 配置 | 说明 |
|---|---|---|---|
| `d_model` | 4096 | WanVideoArchConfig.`text_dim` | 必须匹配 |
| `num_heads` | 64 | T5Config | 编码器内部参数 |
| `num_layers` | 24 | T5Config | 编码器层数 |
| `d_ff` | 10240 | T5Config | 编码器 FFN 维度 |

### Pipeline 级配置对应表

| HF 配置来源 | SGLang Pipeline 配置 | 参数 | 作用 |
|---|---|---|---|
| 全局 (不直接来自HF) | WanT2V480PConfig | `flow_shift` | 调度器参数 |
| 全局 | WanT2V480PConfig | `vae_tiling` | VAE 平铺处理 |
| 全局 | WanT2V480PConfig | `precision` | 计算精度 |
| Scheduler config | FlowUniPCMultistepScheduler | 来自 Scheduler config.json | 去噪调度 |

---

## 推理执行流程

### 完整推理路径

```
用户请求 (文本/图像输入)
    ↓
加载采样参数 (SamplingParams/WanT2V_14B_SamplingParams)
    ├─ height, width, num_frames
    ├─ guidance_scale, num_inference_steps
    └─ negative_prompt
    ↓
文本编码阶段 (Text Encoder)
    ├─ 使用 T5 编码器
    ├─ 输出维度：[batch, seq_len, text_dim(4096)]
    ├─ 可选：CPU offload（text_encoder_cpu_offload）
    └─ 应用 t5_postprocess_text 后处理
    ↓
视频去噪阶段 (DiT Forward Pass)
    ├─ 初始化随机噪声潜在张量
    │  └─ 形状：[batch, num_frames, latent_h, latent_w, in_channels(16)]
    ├─ 遍历所有去噪步骤 (num_inference_steps=50)
    │  ├─ 获取时间步嵌入 (freq_dim=256)
    │  ├─ DiT 前向传播
    │  │  ├─ Patch Embedding
    │  │  ├─ 条件嵌入（文本+时间）
    │  │  ├─ 40 个 Transformer 块
    │  │  │  ├─ 自注意力 (num_heads=40, head_dim=128)
    │  │  │  ├─ 跨注意力 (文本条件)
    │  │  │  └─ FFN (ffn_dim=13824)
    │  │  ├─ 输出投影
    │  │  └─ 输出预测 (形状与输入相同)
    │  └─ 调度器更新 (UniPCMultistepScheduler)
    ├─ 可选优化：TeaCache 加速
    └─ 可选优化：Vmoba 稀疏注意力
    ↓
VAE 解码阶段 (VAE Decoder)
    ├─ 输入：去噪后的潜在张量 [16 channels]
    ├─ 可选：特征缓存 (use_feature_cache=True)
    ├─ 可选：平铺处理 (use_temporal_tiling, use_parallel_tiling)
    ├─ VAE 解码过程
    │  ├─ z 反规范化 (使用 latents_std/mean)
    │  ├─ 解码器网络 (4 层上采样)
    │  └─ 输出视频帧 RGB 像素
    ├─ 时间上采样：4x (scale_factor_temporal=4)
    ├─ 空间上采样：8x (scale_factor_spatial=8)
    └─ 输出形状：[batch, num_frames*4, height*8, width*8, 3]
    ↓
后处理与输出
    ├─ 归一化像素值到 [0, 1]
    ├─ 转换为 uint8 或 float32
    ├─ 保存或返回结果
    └─ 可选：CPU 卸载恢复
```

### 关键配置对推理的影响

| 配置参数 | 来源 | 影响 | 关键性 |
|---|---|---|---|
| `num_layers` | HF config | 模型深度，影响参数量和速度 | ⭐⭐⭐ |
| `num_attention_heads` | HF config | 注意力并行度 | ⭐⭐⭐ |
| `attention_head_dim` | HF config | 单个头的参数量 | ⭐⭐⭐ |
| `ffn_dim` | HF config | FFN 参数量 | ⭐⭐⭐ |
| `text_dim` | HF config | 文本编码维度，必须与文本编码器匹配 | ⭐⭐⭐ |
| `num_inference_steps` | SamplingParams | 去噪质量 vs 速度权衡 | ⭐⭐⭐ |
| `guidance_scale` | SamplingParams | 文本条件强度 | ⭐⭐ |
| `flow_shift` | Pipeline config | 调度器行为 | ⭐⭐ |
| `precision` | Pipeline config | 计算精度（bf16 vs fp32） | ⭐⭐⭐ |
| `use_tiling` | VAE config | 内存使用 vs 处理速度 | ⭐⭐ |
| `teacache_thresh` | SamplingParams | 加速倍数 | ⭐ |

### 多模态输入支持

#### T2V (文本到视频)
```
输入：文本 prompt
处理：
  text → T5Encoder → [batch, seq_len, 4096]
       → Transformer blocks
       → DiT forward
  → 去噪 50 步
  → VAE decode
输出：视频帧
```

#### I2V (图像到视频)
```
输入：静止图像 + 可选文本
处理：
  image → CLIPVisionEncoder → [batch, num_tokens, embed_dim]
  text → T5Encoder → [batch, seq_len, 4096]
       → 融合到 condition embedding
       → DiT forward (I2V 特定架构)
  → 去噪 50 步
  → VAE decode
输出：视频帧
```

#### TI2V (文本+图像到视频)
```
输入：文本 prompt + 初始图像
处理：
  image → 初始图像编码 (可选 CLIP)
  text → T5Encoder → [batch, seq_len, 4096]
       → 融合
       → TI2V 特定 DiT forward
  → 去噪 50 步
  → VAE decode
输出：视频帧
```

---

## 配置参数尺寸对比

### 模型规模参数

| 参数 | T2V-A14B | I2V-A14B | TI2V-5B |
|------|----------|----------|---------|
| 隐藏维度 | 5120 | 5120 | 3072 |
| 注意力头数 | 40 | 40 | 24 |
| 每头维度 | 128 | 128 | 128 |
| 层数 | 40 | 40 | 30 |
| FFN 维度 | 13824 | 13824 | 14336 |
| 频率维度 | 256 | 256 | 256 |
| 文本维度 | 4096 | 4096 | 4096 |
| **估计参数量** | **~14B** | **~14B** | **~5B** |

### 推理配置

| 配置项 | T2V-1.3B | T2V-A14B | I2V-A14B |
|--------|----------|----------|----------|
| 推荐分辨率 | 480×832 | 720×1280 | 720×1280 |
| 默认帧数 | 81 | 81 | 81 |
| 默认去噪步数 | 50 | 50 | 50 |
| 文本引导强度 (guidance_scale) | 3.0 | 5.0 | 5.0 |
| 流移位 (flow_shift) | 8.0 | 3.0-5.0 | 5.0 |

---

## 总结

### 配置体系特点

1. **分层架构**
   - HF 官方配置定义模型架构（必需）
   - SGLang 配置扩展了优化参数和运行参数
   - Pipeline 配置整合了端到端的推理流程

2. **权重兼容性**
   - HF 预训练权重通过 `param_names_mapping` 映射到 SGLang 内部名称
   - 支持多种权重格式转换和 LoRA 适配

3. **灵活的精度和优化**
   - 支持混合精度推理 (bf16 计算, fp32 VAE)
   - 支持多种优化：Tiling、TeaCache、Vmoba 等
   - 支持分布式推理 (FSDP、张量并行)

4. **多模态统一框架**
   - 同一套配置体系支持 T2V、I2V、TI2V
   - 通过 Pipeline 配置的继承和覆盖实现差异

5. **可定制性**
   - CLI 参数可动态覆盖配置
   - 采样参数独立配置
   - 后端优化参数可选应用
