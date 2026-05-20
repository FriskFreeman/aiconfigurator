# Wan 模型配置快速参考

## 配置文件位置速查

```
HuggingFace 官方配置（必需）:
├── config.json                          # 主 DiT 模型配置
├── text_encoder_config.json             # T5 文本编码器配置
├── vae_config.json                      # VAE 编码器配置
└── scheduler_config.json                # 调度器配置

SGLang 内部配置（运行时）:
├── configs/models/dits/wanvideo.py      # DiT 架构配置类
├── configs/models/vaes/wanvae.py        # VAE 架构配置类
├── configs/pipeline_configs/wan.py      # Pipeline 整体配置
├── configs/sample/wan.py                # 采样/推理参数配置
├── configs/backend/vmoba/*.json         # 后端优化配置（可选）
└── test_files/launch_wan.json           # 启动示例配置
```

---

## 模型变体快速对应

### 三种主要模型类型

| 类型 | HF 模型名 | SGLang 配置类 | 关键特性 |
|------|---------|-------------|--------|
| T2V | `Wan2.2-T2V-A14B-Diffusers` | `Wan2_2_T2V_A14B_Config` | 文本→视频，14B 参数 |
| I2V | `Wan2.2-I2V-A14B-Diffusers` | `Wan2_2_I2V_A14B_Config` | 图像→视频，14B 参数 |
| TI2V | `Wan2.2-TI2V-5B-Diffusers` | `Wan2_2_TI2V_5B_Config` | 文本+图像→视频，5B 参数 |

### 分辨率变体

| 变体 | 分辨率 | 帧数 | 文本编码 | 默认 guidance |
|------|--------|------|---------|-------------|
| 480p | 480×832 (9:16) 或 832×480 (16:9) | 81 | T5-4096 | 3.0 |
| 720p | 720×1280 (9:16) 或 1280×720 (16:9) | 81 | T5-4096 | 5.0 |
| 1.3B | 小型模型 | 77 | T5-2048 | 3.0 |

---

## 关键参数对应快速表

### HF config.json → SGLang 配置映射

```yaml
DiT 参数:
  num_attention_heads: 40           → WanVideoArchConfig.num_attention_heads
  attention_head_dim: 128           → WanVideoArchConfig.attention_head_dim
  num_layers: 40                    → WanVideoArchConfig.num_layers
  in_channels: 16                   → WanVideoArchConfig.in_channels
  out_channels: 16                  → WanVideoArchConfig.out_channels
  text_dim: 4096                    → WanVideoArchConfig.text_dim
  freq_dim: 256                     → WanVideoArchConfig.freq_dim
  ffn_dim: 13824                    → WanVideoArchConfig.ffn_dim
  patch_size: [1,2,2]               → WanVideoArchConfig.patch_size
  rope_max_seq_len: 1024            → WanVideoArchConfig.rope_max_seq_len
  cross_attn_norm: true             → WanVideoArchConfig.cross_attn_norm
  qk_norm: rms_norm_across_heads    → WanVideoArchConfig.qk_norm
  eps: 1e-6                         → WanVideoArchConfig.eps

VAE 参数:
  z_dim: 16                         → WanVAEArchConfig.z_dim
  base_dim: 96                      → WanVAEArchConfig.base_dim
  dim_mult: [1,2,4,4]               → WanVAEArchConfig.dim_mult
  num_res_blocks: 2                 → WanVAEArchConfig.num_res_blocks
  latents_mean: [...]               → WanVAEArchConfig.latents_mean
  latents_std: [...]                → WanVAEArchConfig.latents_std
  scale_factor_temporal: 4          → WanVAEArchConfig.scale_factor_temporal
  scale_factor_spatial: 8           → WanVAEArchConfig.scale_factor_spatial

调度器参数:
  flow_shift: 3.0/5.0               → WanT2V*PConfig.flow_shift
  num_train_timesteps: 1000         → FlowUniPCMultistepScheduler
  prediction_type: flow_prediction  → FlowUniPCMultistepScheduler
```

---

## 推理流程参数检查

### 启动前验证清单

```bash
✓ 检查 model_path 是否存在且包含 config.json
✓ 验证 HF config.json 中的 num_attention_heads * attention_head_dim == hidden_size
✓ 检查 text_dim 是否与文本编码器输出维度匹配（通常 4096）
✓ 验证 in_channels == z_dim == 16（VAE 潜在空间）
✓ 检查采样参数 num_inference_steps（通常 50）
✓ 验证 guidance_scale 是否适合模型类型
✓ 确认 precision 设置（bf16 或 fp32）
✓ 检查 vae_precision（通常为 fp32）
✓ 验证分辨率是否在支持范围内
```

### 运行时配置参数

```python
# 最小可运行配置
config = {
    "model_path": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    "height": 720,
    "width": 1280,
    "num_frames": 81,
    "num_inference_steps": 50,
    "guidance_scale": 5.0,
    "precision": "bf16",
}

# 推荐完整配置
config = {
    # 模型
    "model_path": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    
    # 输出分辨率
    "height": 720,
    "width": 1280,
    "num_frames": 81,
    "fps": 16,
    
    # 去噪参数
    "num_inference_steps": 50,
    "guidance_scale": 5.0,
    "negative_prompt": "...",
    
    # 精度
    "precision": "bf16",
    "vae_precision": "fp32",
    "text_encoder_precision": "fp32",
    
    # 优化
    "text_encoder_cpu_offload": true,
    "pin_cpu_memory": true,
    "enable_teacache": true,
    "teacache_thresh": 0.08,
    
    # 硬件
    "dtype": "bfloat16",
    "device": "cuda",
    "num_gpu_per_node": 1,
}
```

---

## 常见配置问题排查

### Q1: 文本维度不匹配
```
错误: RuntimeError: expected scalar type Double but found Float
原因: text_dim (4096) 与文本编码器输出维度不匹配
解决: 检查 HF config.json 中 text_dim 是否与文本编码器的 d_model 一致
```

### Q2: VAE 潜在维度不对应
```
错误: RuntimeError: size mismatch at input 0
原因: in_channels != z_dim（应都为 16）
解决: 验证 HF config.json 中 in_channels 和 VAE config 中 z_dim
```

### Q3: 注意力头维度错误
```
错误: RuntimeError: expected size [head_dim]
原因: num_attention_heads * attention_head_dim != hidden_size
解决: 计算: 40 * 128 = 5120（默认）
```

### Q4: 采样参数不匹配
```
错误: 生成质量差或形状错误
原因: num_frames/height/width 与模型训练尺寸不匹配
解决: 使用支持的分辨率
  - T2V-A14B: 720×1280 或 832×480
  - 帧数: 通常 81
```

### Q5: 精度配置导致的性能问题
```
问题: 内存溢出或速度很慢
原因: precision 设置不当
解决: 推荐配置
  - 主计算: bf16（节省内存，速度快）
  - VAE: fp32（保证精度）
  - 文本编码器: fp32（避免量化错误）
```

---

## 配置优先级（高到低）

```
1. 命令行参数         ← 最高优先级（会覆盖所有下层）
2. ServerArgs 设置
3. Pipeline 配置类 (WanT2V*Config)
4. HF config.json      ← 中间优先级（架构定义）
5. 基类默认值         ← 最低优先级
```

**示例：** 如果 HF config.json 中 `flow_shift=3.0`，但 Pipeline 类中定义 `flow_shift=5.0`，最终使用值为 5.0（优先级 3 > 4）。

---

## 权重加载流程速查

```
1. 从 HF 加载原始权重
   ├─ 权重名: patch_embedding.weight
   └─ 维度: [hidden_size, in_channels, patch_size[1], patch_size[2]]

2. 应用 param_names_mapping
   ├─ 模式: ^patch_embedding\.(.*)$
   ├─ 替换: patch_embedding.proj.$1
   └─ 结果: patch_embedding.proj.weight

3. 加载到模型
   └─ 模型内部: self.patch_embedding.proj.weight ← patch_embedding.proj.weight

4. 可选: 应用 lora_param_names_mapping
   └─ 用于加载 LoRA 适配器权重
```

---

## 性能优化参数

### TeaCache 加速（推荐）
```python
{
    "enable_teacache": true,
    "teacache_thresh": 0.08,        # 缓存阈值
    "start_skipping": 5,            # 何时开始加速
    "end_skipping": 1.0,            # 何时停止加速
}
# 预期加速: ~1.5-2x（精度损失 < 1%）
```

### Vmoba 稀疏注意力（可选）
```python
{
    "enable_vmoba": true,
    "temporal_chunk_size": 2,
    "spatial_chunk_size": [3, 4],
    "moba_threshold": 0.25,
}
# 预期加速: ~1.2-1.5x（对质量影响较小）
```

### VAE 平铺（低内存模式）
```python
{
    "use_tiling": true,
    "use_temporal_tiling": true,
    "use_parallel_tiling": true,
}
# 权衡: 内存使用 ↓ 50%，速度 ↓ 20%
```

---

## 模型尺寸参考

| 模型 | 参数量 | 推荐 GPU 内存 | 最小 GPU 内存 |
|------|--------|-------------|-------------|
| T2V-1.3B | 1.3B | 8GB | 6GB |
| T2V-A14B | 14B | 48GB | 24GB |
| I2V-A14B | 14B | 48GB | 24GB |
| TI2V-5B | 5B | 16GB | 12GB |

**备注：** 使用优化（TeaCache、Vmoba）可降低 VRAM 使用 20-30%。

---

## 关键参数含义速查

```
hidden_size = num_attention_heads * attention_head_dim
            = 40 * 128 = 5120
            → Transformer 所有层的通道维度

num_layers = 40  → Transformer 堆叠深度，影响模型容量和速度

ffn_dim = 13824  → FFN 的隐藏层维度，约 2.7 倍于 hidden_size

text_dim = 4096  → 文本编码器输出维度，必须与 T5 模型匹配

freq_dim = 256   → 时间步频率嵌入维度，用于扩散时间步编码

patch_size = (1,2,2)  → 补丁大小，降低序列长度
            → num_patches = frames/1 * height/2 * width/2

flow_shift = 3.0-5.0  → 调度器参数，影响去噪过程，5.0 通常质量更好

num_inference_steps = 50  → 去噪迭代次数，越多质量越好（越慢）

guidance_scale = 5.0  → 文本条件强度，4-6 较好

scale_factor_temporal = 4   → VAE 时间解码倍数（总压缩率）
scale_factor_spatial = 8    → VAE 空间解码倍数（总压缩率）

precision = "bf16"  → Brain Float 16，节省显存，保持精度

use_tiling = true  → VAE 分块处理，降低内存
```

---

## 常用命令示例

### 启动推理服务
```bash
python -m sglang.multimodal_gen.launch_server \
  --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --dtype bfloat16 \
  --num-gpu-per-node 1 \
  --enable-teacache
```

### Python API 使用
```python
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.pipelines import WanPipeline

# 初始化
server_args = ServerArgs(
    model_path="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    dtype="bfloat16",
)
pipeline = WanPipeline(server_args)

# 推理
output = pipeline(
    prompt="A cat walking on the street",
    height=720,
    width=1280,
    num_frames=81,
    num_inference_steps=50,
    guidance_scale=5.0,
)
```

### 调整配置
```python
# 修改采样参数
config.dit_config.arch_config.num_layers = 30  # 快速推理
config.pipeline_config.flow_shift = 8.0        # 提高质量
config.pipeline_config.precision = "fp32"      # 提高精度（更慢）
```

---

## 学习路径建议

### 初级：快速了解
1. 阅读本快速参考（5 分钟）
2. 查看支持的模型列表
3. 运行默认推理脚本

### 中级：理解配置
1. 阅读主分析文档（wan_model_config_analysis.md）
2. 理解 HF config → SGLang 映射
3. 学习权重加载流程

### 高级：深度优化
1. 阅读实现细节文档（wan_model_config_implementation.md）
2. 学习参数映射规则
3. 研究性能优化选项
4. 修改/扩展配置类

