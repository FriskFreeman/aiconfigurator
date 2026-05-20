# Wan 模型配置架构和数据流详解

## 目录
1. [配置继承架构图](#配置继承架构图)
2. [完整数据流](#完整数据流)
3. [配置加载时序图](#配置加载时序图)
4. [张量形状变化追踪](#张量形状变化追踪)
5. [模块间数据传递](#模块间数据传递)

---

## 配置继承架构图

### Python 类继承关系

```
┌─────────────────────────────────────────────────────────────────┐
│                          基础配置框架                              │
├─────────────────────────────────────────────────────────────────┤
│ ArchConfig (base.py)                                             │
│   - 架构配置基类                                                   │
│   - 不包含具体参数                                                 │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
        ┌───────▼─────────┐         ┌────────▼────────┐
        │ DiTArchConfig   │         │ VAEArchConfig   │
        │ (dits/base.py)  │         │ (vaes/base.py)  │
        │                 │         │                 │
        │ - 参数映射规则   │         │ - 压缩比例      │
        │ - 注意力后端支持 │         │ - 规范化参数    │
        │ - 基础参数      │         │ - 平铺参数      │
        └────────┬────────┘         └────────┬────────┘
                 │                           │
        ┌────────▼────────────┐    ┌────────▼─────────┐
        │ WanVideoArchConfig  │    │ WanVAEArchConfig │
        │ (wanvideo.py)       │    │ (wanvae.py)      │
        │                     │    │                  │
        │ ✓ 权重映射规则(40+) │    │ ✓ 规范化参数     │
        │ ✓ DiT参数(15个)     │    │ ✓ VAE参数(10个)  │
        │ ✓ MoE参数           │    │ ✓ 精度设置       │
        │ ✓ 因果注意力参数    │    │                  │
        └────────┬────────────┘    └────────┬─────────┘
                 │                           │
        ┌────────▼────────────┐    ┌────────▼─────────┐
        │  WanVideoConfig     │    │  WanVAEConfig    │
        │  (wanvideo.py)      │    │  (wanvae.py)     │
        │                     │    │                  │
        │ - 继承 DiTConfig    │    │ - 继承 VAEConfig │
        │ - arch_config       │    │ - arch_config    │
        │ - 量化配置(可选)    │    │ - 输出配置       │
        └────────┬────────────┘    └────────┬─────────┘
                 │                           │
                 └──────────────┬────────────┘
                                │
                    ┌───────────▼──────────────┐
                    │  WanT2V480PConfig       │
                    │  (wan.py)               │
                    │                         │
                    │ ✓ dit_config           │
                    │ ✓ vae_config           │
                    │ ✓ flow_shift           │
                    │ ✓ 文本编码配置         │
                    │ ✓ 精度配置             │
                    └───────────┬──────────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
        ┌───────▼─────────┐    │    ┌──────────▼────────┐
        │ WanT2V720PConfig│    │    │ Wan2_2_T2V_A14B  │
        │ (720p variant)  │    │    │ (具体T2V模型)     │
        └─────────────────┘    │    └──────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │ Wan2_2_I2V_A14B     │
                    │ (具体I2V模型)        │
                    └─────────────────────┘
```

### 配置参数流向图

```
┌─────────────────────────────────────────────────────────────────┐
│                   用户输入（启动参数）                              │
│        model_path, height, width, num_frames, guidance_scale    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │ 配置加载器       │
                    │ ConfigLoader    │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
   ┌────▼──────┐      ┌─────▼──────┐      ┌──────▼────┐
   │ HF config │      │ Pipeline   │      │ Sampling  │
   │ .json     │      │ Config     │      │ Params    │
   │ 架构参数  │      │ 整体配置   │      │ 推理参数  │
   └────┬──────┘      └─────┬──────┘      └──────┬────┘
        │                   │                    │
        └───────────────────┼────────────────────┘
                            │
                    ┌───────▼──────────┐
                    │ 配置合并          │
                    │ merge_configs()  │
                    └───────┬──────────┘
                            │
        ┌───────────────────┼────────────────────┐
        │                   │                    │
   ┌────▼──────────┐   ┌────▼──────────┐   ┌────▼──────────┐
   │ DiT 配置       │   │ VAE 配置       │   │ Scheduler 配置│
   │ (权重映射)    │   │ (规范化参数)   │   │ (flow_shift) │
   └────┬──────────┘   └────┬──────────┘   └────┬──────────┘
        │                   │                    │
        └───────────────────┼────────────────────┘
                            │
                    ┌───────▼────────┐
                    │ 模型初始化      │
                    │ initialize()   │
                    └────────────────┘
```

---

## 完整数据流

### 推理数据流路径

```
用户输入：prompt="A cat walking down the street"
  ↓
┌─────────────────────────────────────────────┐
│ 1. 文本编码阶段 (Text Encoder - T5)          │
├─────────────────────────────────────────────┤
│ 输入:  "A cat walking down the street"      │
│ 处理:  T5 Tokenizer → Token IDs             │
│       T5 Encoder → [batch, seq_len, 4096]  │
│ 输出:  text_embeddings                      │
│        形状: [1, ~20, 4096]                 │
└──────────────┬────────────────────────────┘
               │
┌──────────────▼────────────────────────────┐
│ 2. 初始化 (Initialization)                 │
├──────────────────────────────────────────┤
│ 创建随机噪声张量                           │
│ 形状: [batch=1, frames=9, height=30,      │
│        width=40, channels=16]             │
│ (对应 720x1280 图像通过 VAE 编码)         │
│                                           │
│ 初始化去噪步骤计数器: step=0              │
└──────────────┬────────────────────────────┘
               │
┌──────────────▼────────────────────────────┐
│ 3. 去噪循环 (Denoising Loop)               │
│ for step in range(50):                    │
├──────────────────────────────────────────┤
│                                           │
│ ┌──────────────────────────────────────┐  │
│ │ 3.1 时间步嵌入编码                    │  │
│ │ timestep_embedding = encode_timestep │  │
│ │ 输入: step (标量)                     │  │
│ │ 输出: [batch=1, freq_dim=256]        │  │
│ └────────────┬─────────────────────────┘  │
│              │                             │
│ ┌────────────▼─────────────────────────┐  │
│ │ 3.2 补丁嵌入 (Patch Embedding)       │  │
│ │ x = patch_embed(noise_tensor)        │  │
│ │ 输入: [1, 9, 30, 40, 16]             │  │
│ │ 处理: 应用投影层                      │  │
│ │ 输出: [1, 2700, 5120]                │  │
│ │ (9*15*20 = 2700 patches)             │  │
│ └────────────┬─────────────────────────┘  │
│              │                             │
│ ┌────────────▼─────────────────────────┐  │
│ │ 3.3 条件嵌入 (Condition Embedding)   │  │
│ │ cond = condition_embedder(           │  │
│ │   timestep_emb, text_embeddings,    │  │
│ │   image_emb[可选]                    │  │
│ │ )                                     │  │
│ │ 输出: [1, 5120] 或 [1, seq, 5120]   │  │
│ └────────────┬─────────────────────────┘  │
│              │                             │
│ ┌────────────▼─────────────────────────┐  │
│ │ 3.4 Transformer 块处理 (40 层)        │  │
│ │ for block_idx in range(40):          │  │
│ │   x = transformer_block(             │  │
│ │     x=[1, 2700, 5120],              │  │
│ │     cond=[1, 5120]/[1, seq, 5120], │  │
│ │     context=text_embeddings          │  │
│ │   )                                   │  │
│ │                                       │  │
│ │   块内处理:                           │  │
│ │   ├─ 自注意力 (Self-Attn)            │  │
│ │   │  Q,K,V 投影                      │  │
│ │   │   多头处理 (40头, 每头128维)     │  │
│ │   │  heads=[batch*40, 2700, 128]    │  │
│ │   ├─ 跨注意力 (Cross-Attn)           │  │
│ │   │  与文本条件交互                   │  │
│ │   └─ FFN                             │  │
│ │      中间维度 13824                  │  │
│ │                                       │  │
│ │ 输出: [1, 2700, 5120] (形状不变)     │  │
│ └────────────┬─────────────────────────┘  │
│              │                             │
│ ┌────────────▼─────────────────────────┐  │
│ │ 3.5 输出投影 (Output Projection)     │  │
│ │ pred_noise = output_proj(x)          │  │
│ │ 输入: [1, 2700, 5120]                │  │
│ │ 输出: [1, 2700, 16*4]                │  │
│ │       = [1, 2700, 64]                │  │
│ │ 重塑: [1, 9, 30, 40, 16]             │  │
│ └────────────┬─────────────────────────┘  │
│              │                             │
│ ┌────────────▼─────────────────────────┐  │
│ │ 3.6 调度器更新 (Scheduler Update)    │  │
│ │ noise_tensor = scheduler.step(       │  │
│ │   pred_noise,                        │  │
│ │   timestep,                          │  │
│ │   noise_tensor                       │  │
│ │ )                                     │  │
│ │ 输入/输出: [1, 9, 30, 40, 16]       │  │
│ │ (应用去噪更新)                       │  │
│ └──────────────────────────────────────┘  │
│ (循环 50 次)                              │
└──────────────┬────────────────────────────┘
               │
┌──────────────▼────────────────────────────┐
│ 4. VAE 解码阶段 (VAE Decoder)             │
├──────────────────────────────────────────┤
│ 输入: denoise_latents [1, 9, 30, 40, 16] │
│                                           │
│ 步骤:                                     │
│ 1) 潜在反规范化 (Denormalize)            │
│    使用 latents_mean 和 latents_std      │
│                                           │
│ 2) 解码器网络 (4 层上采样)               │
│    ├─ 上采样 T: 9 → 36  (4x)            │
│    ├─ 上采样 H: 30 → 240 (8x)           │
│    └─ 上采样 W: 40 → 320 (8x)           │
│                                           │
│ 3) 最终投影到 RGB                        │
│    输出: [1, 36, 240, 320, 3]           │
│    (即 1440x2560 视频，1/4 倍速)        │
│                                           │
│ 输出: video_frames                       │
│ 范围: [0, 1] 或 [0, 255]                 │
└──────────────┬────────────────────────────┘
               │
┌──────────────▼────────────────────────────┐
│ 5. 后处理 (Post-processing)               │
├──────────────────────────────────────────┤
│ - 像素值归一化                            │
│ - 格式转换 (uint8 或 float32)            │
│ - 可选: 压缩编码 (H.264, VP9)            │
│ - 保存或返回                              │
│                                           │
│ 最终输出: video.mp4                      │
│ 分辨率: 1440×2560 (4倍于潜在空间)       │
│ 帧数: 36 (4倍于潜在空间)                │
│ 时长: ~2.25 秒 @ 16fps                   │
└──────────────────────────────────────────┘
```

---

## 配置加载时序图

### 启动到推理的时间流

```
时间轴:
│
├─ T0: 用户执行启动命令
│  │
│  └─> serverArgs = ServerArgs(model_path="Wan-AI/Wan2.2-T2V-A14B-Diffusers")
│
├─ T1: 加载 HF 模型配置
│  │
│  ├─> model_path/config.json
│  │    ├─ num_attention_heads: 40
│  │    ├─ num_layers: 40
│  │    ├─ in_channels: 16
│  │    └─ ...
│  │
│  ├─> model_path/text_encoder_config.json
│  │    ├─ d_model: 4096
│  │    └─ ...
│  │
│  ├─> model_path/vae_config.json
│  │    ├─ z_dim: 16
│  │    └─ ...
│  │
│  └─> model_path/scheduler_config.json
│       ├─ flow_shift: 3.0
│       └─ ...
│
├─ T2: 选择 SGLang Pipeline 配置
│  │
│  └─> 从模型名推断: Wan2_2_T2V_A14B_Config()
│
├─ T3: 合并配置（优先级合并）
│  │
│  ├─ 基础: DiTArchConfig 默认值
│  ├─ 覆盖: WanVideoArchConfig 定义值
│  ├─ 覆盖: HF config.json 值
│  └─ 覆盖: 命令行参数 (如有)
│
├─ T4: 加载预训练权重
│  │
│  ├─> pytorch_model.bin 或 safetensors
│  │    └─ 包含所有模型权重
│  │
│  ├─> 应用权重名映射 (param_names_mapping)
│  │    ├─ HF: patch_embedding.weight
│  │    └─ SGLang: patch_embedding.proj.weight
│  │
│  └─> 加载到模型对象
│       ├─ DiT 权重
│       ├─ VAE 权重
│       ├─ 文本编码器权重
│       └─ 调度器参数
│
├─ T5: 模型初始化完成
│  │
│  └─> pipeline_ready = True
│       ├─ DiT 在评估模式
│       ├─ 所有权重已加载
│       └─ 精度已设置 (bf16)
│
├─ T6: 等待用户请求
│  │
│  └─> 服务处于监听状态
│
├─ T7: 接收推理请求
│  │
│  └─> {
│       "prompt": "A cat ...",
│       "height": 720,
│       "width": 1280,
│       "num_frames": 81,
│       "num_inference_steps": 50,
│       "guidance_scale": 5.0
│      }
│
├─ T8-T57: 执行去噪循环 (50 步)
│  │
│  └─> 每步耗时 ~2-5 秒 (取决于 GPU 和优化)
│       总耗时 ~100-250 秒
│
├─ T58: VAE 解码
│  │
│  └─> 耗时 ~10-20 秒
│
└─ T59+: 返回结果 + 后处理
   └─> 总耗时 ~2-5 分钟
```

---

## 张量形状变化追踪

### T2V 推理中的张量形状

```
=== 输入阶段 ===
文本:          "A cat walking down the street"
  ↓ Tokenize
Token IDs:     [101, 1037, 3406, ...]  (shape: [20])
  ↓ Embedding + T5 Encoder
文本特征:      shape: [1, 20, 4096]
  ↓ 后处理 (t5_postprocess_text)
条件张量:      shape: [1, 512, 4096]   (填充到最大长度)

=== 初始化 ===
随机噪声:      shape: [1, 9, 30, 40, 16]
               (对应输入 720×1280 在 VAE 潜在空间)
               计算: frames=81→9, height=720/24→30, width=1280/32→40

=== 去噪循环第 0 步 ===

补丁嵌入前:    [1, 9, 30, 40, 16]
  ↓ Patch Embedding (patch_size=1×2×2)
补丁嵌入后:    [1, 2700, 5120]
               (9×30×20=5400 patches → 9×15×20=2700 patches)
               
时间步嵌入:    [1, 256]
  ↓ 投影到条件
条件向量:      [1, 5120]

=== Transformer Block 0-39 ===
每个块的输入:  [1, 2700, 5120]

自注意力内部:
  Q, K, V:     [1, 2700, 5120] → [1, 40, 2700, 128]
  注意力权重:   [1, 40, 2700, 2700]
  输出:         [1, 40, 2700, 128] → [1, 2700, 5120]

跨注意力内部:
  Q:           [1, 2700, 5120]
  K, V:        [1, 512, 5120]  (来自文本)
  注意力权重:   [1, 40, 2700, 512]
  输出:         [1, 2700, 5120]

FFN:
  中间:         [1, 2700, 13824]
  输出:         [1, 2700, 5120]

块输出:         [1, 2700, 5120]  (形状保持不变)

=== 所有块处理后 ===
Transformer 输出: [1, 2700, 5120]

输出投影:
  投影前:       [1, 2700, 5120]
  投影后:       [1, 2700, 64]   (16*4, 因为 patch_size=(1,2,2))
  
重塑:
  最终预测:     [1, 9, 30, 40, 16]
                (恢复原始补丁空间形状)

=== 调度器更新 ===
去噪前:         [1, 9, 30, 40, 16]
去噪后:         [1, 9, 30, 40, 16]  (形状不变)

=== VAE 解码 ===
解码输入:       [1, 9, 30, 40, 16]

解码器块 0:
  输入:         [1, 9, 30, 40, 16]
  时间上采样:   [1, 36, 30, 40, ?]   (9 → 36, 4x)
  空间上采样:   [1, 36, 30, 40, ?]
  输出:         [1, 36, 60, 80, ?]

解码器块 1-3:
  逐步上采样
  最终:         [1, 36, 240, 320, 3]

VAE 输出:       [1, 36, 240, 320, 3]
                (320×1280 视频)

后处理:
  范围调整:     [0, 1] → [0, 255]
  格式转换:     float32 → uint8
  最终形状:     [36, 320, 1280, 3]  (帧, 高, 宽, 通道)
```

### 内存占用估计

```
批次大小 = 1, 精度 = bf16 (2 字节/值)

文本特征:           1×512×4096×2 = 4.2 MB
随机噪声:           1×9×30×40×16×2 = 34 MB

补丁嵌入:           1×2700×5120×2 = 27.6 MB

Transformer 块:
  单块激活:         1×2700×5120×2 = 27.6 MB
  注意力权重:       1×40×2700×2700×2 = 720 MB (最大)
  总内存:           ~750 MB 每块

VAE 解码:
  中间激活:         1×36×240×320×3×2 = 18 MB

总峰值内存:         ~800 MB (考虑梯度和缓存)
实际 GPU 内存:      ~12-16 GB @ A14B (包括权重)
```

---

## 模块间数据传递

### 关键接口定义

```python
# 1. DiT Forward 接口
class WanVideoModel(CachableDiT):
    def forward(
        self,
        x: torch.Tensor,              # [B, T, H, W, C] 潜在表示
        timesteps: torch.Tensor,      # [B] 时间步
        context: torch.Tensor,        # [B, S, D] 文本特征
        context_lens: torch.Tensor,   # [B] 有效长度
    ) -> torch.Tensor:               # [B, T, H, W, C] 预测噪声
        pass

# 2. VAE Forward 接口
class WanVAEModel(nn.Module):
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # [B, T, H, W, 3] → [B, T, H/8, W/8, 16]
        pass
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # [B, T, H/8, W/8, 16] → [B, 4T, H, W, 3]
        pass

# 3. Pipeline 接口
class WanPipeline:
    def __call__(
        self,
        prompt: str,
        height: int = 720,
        width: int = 1280,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        negative_prompt: str = "",
    ) -> Video:
        pass
```

### 数据流关键节点

```
Pipeline Layer:
┌────────────────────────────────────┐
│ WanPipeline.__call__()             │
│ ├─ prompt, height, width, ...     │
│ └─ returns: Video                 │
└───────────┬────────────────────────┘
            │
            ├─> TextEncoder Layer
            │   ├─ Input: prompt (str)
            │   ├─ Config: T5Config
            │   └─ Output: [1, 512, 4096]
            │
            ├─> VAE Encoder (可选)
            │   ├─ Input: image (可选)
            │   └─ Output: [1, H/8, W/8, 16]
            │
            ├─> Scheduler Init
            │   ├─ Config: flow_shift, num_steps
            │   └─ State: timesteps, sigmas
            │
            ├─> Denoising Loop (50 iterations)
            │   ├─ Input: noise, timestep, context
            │   ├─ DiT.forward()
            │   │  ├─ PatchEmbed
            │   │  ├─ ConditionEmbed
            │   │  ├─ TransformerBlocks[0..39]
            │   │  └─ OutputProj
            │   ├─ Scheduler.step()
            │   └─ Output: denoised_latent
            │
            ├─> VAE Decoder
            │   ├─ Input: [1, 9, 30, 40, 16]
            │   ├─ Config: scale_factor_t=4, scale_factor_s=8
            │   └─ Output: [1, 36, 320, 1280, 3]
            │
            └─> Post-processing
                ├─ Normalization
                ├─ Format conversion
                └─ Output: Video file
```

### 配置到执行的映射

```
启动配置层:
├─ model_path: "Wan-AI/Wan2.2-T2V-A14B"
├─ precision: "bf16"
├─ guidance_scale: 5.0
└─ num_inference_steps: 50

                    ↓ 加载和选择

Pipeline 配置层:
├─ Wan2_2_T2V_A14B_Config
│  ├─ dit_config: WanVideoConfig
│  │  ├─ arch_config: WanVideoArchConfig
│  │  │  ├─ num_layers: 40
│  │  │  ├─ num_attention_heads: 40
│  │  │  ├─ attention_head_dim: 128
│  │  │  └─ param_names_mapping: {...}
│  │  └─ quant_config: None
│  ├─ vae_config: WanVAEConfig
│  │  └─ arch_config: WanVAEArchConfig
│  │     ├─ z_dim: 16
│  │     ├─ scale_factor_spatial: 8
│  │     └─ latents_mean/std: [...]
│  ├─ flow_shift: 3.0
│  └─ precision: "bf16"

                    ↓ 初始化模型

模型实例层:
├─ WanVideoModel (40 个 Transformer 块)
│  ├─ patch_embedding (1×2×2 补丁)
│  ├─ condition_embedder (时间+文本)
│  └─ transformer_blocks[0..39]
│     ├─ 每块 40 个头 × 128 维
│     ├─ 跨注意力 (文本交互)
│     └─ FFN (13824 中间维度)
├─ WanVAEModel
│  ├─ encoder (可选)
│  └─ decoder (4 层上采样)
├─ TextEncoder (T5-4096)
└─ Scheduler (UniPC)

                    ↓ 执行推理

运行时执行层:
├─ 文本 → [1, 512, 4096]
├─ 50 步循环
│  ├─ 时间步 → 嵌入
│  ├─ DiT forward
│  │  ├─ [1, 9, 30, 40, 16] → 补丁 → [1, 2700, 5120]
│  │  ├─ 40 块处理
│  │  └─ [1, 2700, 5120] → 输出 → [1, 9, 30, 40, 16]
│  └─ 调度器更新
├─ VAE 解码
│  ├─ [1, 9, 30, 40, 16] → 解码 → [1, 36, 320, 1280, 3]
└─ 返回视频
```

---

## 配置与性能的关系

```
配置参数变化的影响:

num_layers: 40 → 30
  ├─ 模型参数: 14B → 10.5B (-25%)
  ├─ 推理时间: 100s → 75s (-25%)
  ├─ 内存占用: 16GB → 12GB (-25%)
  └─ 输出质量: 略有下降 (-5%)

num_inference_steps: 50 → 30
  ├─ 推理时间: 100s → 60s (-40%)
  ├─ 内存占用: 基本不变
  └─ 输出质量: 明显下降 (-20%)

guidance_scale: 5.0 → 7.0
  ├─ 推理时间: 不变
  ├─ 内存占用: 不变
  └─ 输出质量: 更符合文本，部分过饱和

precision: bf16 → fp32
  ├─ 内存占用: 16GB → 32GB (+100%)
  ├─ 推理时间: 100s → 120s (+20%)
  └─ 输出质量: 略有提高 (+2%)

enable_teacache: false → true
  ├─ 推理时间: 100s → 50s (-50%)
  ├─ 内存占用: 16GB → 14GB (-12.5%)
  └─ 输出质量: 基本不变 (-1%)

use_tiling: false → true
  ├─ 推理时间: 100s → 120s (+20%)
  ├─ 内存占用: 16GB → 10GB (-37.5%)
  └─ 输出质量: 略有伪影 (-3%)
```

