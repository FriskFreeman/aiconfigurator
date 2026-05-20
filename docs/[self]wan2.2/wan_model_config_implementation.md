# Wan 模型配置实现细节与代码示例

## 目录
1. [配置加载代码示例](#配置加载代码示例)
2. [权重名映射详解](#权重名映射详解)
3. [模型初始化流程](#模型初始化流程)
4. [SGLang 配置类完整定义](#sglang-配置类完整定义)
5. [推理参数配置示例](#推理参数配置示例)

---

## 配置加载代码示例

### 1. 从 HF 模型加载配置

```python
# SGLang 推理启动时
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.configs.pipeline_configs import (
    Wan2_2_T2V_A14B_Config,
    Wan2_2_I2V_A14B_Config,
    Wan2_2_TI2V_5B_Config,
)

# 步骤 1: 解析启动参数
server_args = ServerArgs(
    model_path="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    # ... 其他参数
)

# 步骤 2: 加载 HF 的 config.json
import json
from pathlib import Path

config_path = Path(server_args.model_path) / "config.json"
hf_config = json.load(open(config_path))

# 输出示例:
# {
#   "_class_name": "WanTransformer3DModel",
#   "num_attention_heads": 40,
#   "attention_head_dim": 128,
#   "num_layers": 40,
#   "in_channels": 16,
#   ...
# }

# 步骤 3: 选择对应的 SGLang Pipeline 配置
def get_pipeline_config(model_path: str):
    """根据模型路径或配置，选择合适的 Pipeline 配置"""
    if "T2V" in model_path and "A14B" in model_path:
        return Wan2_2_T2V_A14B_Config()
    elif "I2V" in model_path and "A14B" in model_path:
        return Wan2_2_I2V_A14B_Config()
    elif "TI2V" in model_path and "5B" in model_path:
        return Wan2_2_TI2V_5B_Config()
    else:
        raise ValueError(f"Unsupported model: {model_path}")

pipeline_config = get_pipeline_config(server_args.model_path)

# 步骤 4: DiT 配置已自动初始化
dit_config = pipeline_config.dit_config
# dit_config.arch_config.num_attention_heads == 40
# dit_config.arch_config.num_layers == 40
# dit_config.arch_config.ffn_dim == 13824
```

### 2. 配置覆盖层次

```python
# 配置优先级（从高到低）

# 最高：命令行参数
server_args = ServerArgs(
    model_path="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    # CLI 参数直接覆盖下层配置
)

# 第二层：Pipeline 配置类定义
@dataclass
class Wan2_2_T2V_A14B_Config(WanT2V480PConfig):
    # 可在这里覆盖默认值
    pass

# 第三层：HF 官方 config.json
# 从 model_path/config.json 加载的参数值

# 最低：基类默认值
@dataclass
class WanVideoArchConfig(DiTArchConfig):
    num_attention_heads: int = 40  # 默认值
    num_layers: int = 40           # 默认值
```

### 3. 完整配置初始化流程

```python
from sglang.multimodal_gen.runtime.server_args import ServerArgs

class WanPipelineBuilder:
    def __init__(self, server_args: ServerArgs):
        self.server_args = server_args
        self.pipeline_config = self._load_pipeline_config()
        self.dit_config = self.pipeline_config.dit_config
        self.vae_config = self.pipeline_config.vae_config
    
    def _load_pipeline_config(self):
        """从模型路径加载合适的 Pipeline 配置"""
        # 1. 尝试从 server_args.pipeline_config 直接获取
        if hasattr(self.server_args, 'pipeline_config'):
            return self.server_args.pipeline_config
        
        # 2. 从模型名称推断
        model_name = self.server_args.model_path.split('/')[-1]
        if 'T2V' in model_name and 'A14B' in model_name:
            return Wan2_2_T2V_A14B_Config()
        # ... 其他模型类型
    
    def initialize_models(self):
        """初始化各个模型组件"""
        # DiT 模型
        dit_model = self._init_dit_model()
        
        # VAE 模型
        vae_model = self._init_vae_model()
        
        # 文本编码器
        text_encoder = self._init_text_encoder()
        
        # 调度器
        scheduler = self._init_scheduler()
        
        return dit_model, vae_model, text_encoder, scheduler
    
    def _init_dit_model(self):
        """初始化 DiT（Diffusion Transformer）"""
        from sglang.multimodal_gen.runtime.models.dits import WanVideoModel
        
        # 使用 DiT 配置
        dit_model = WanVideoModel(
            config=self.dit_config.arch_config,
            # ... 其他参数
        )
        
        # 加载预训练权重
        self._load_weights_with_mapping(
            dit_model,
            weight_mapping=self.dit_config.arch_config.param_names_mapping
        )
        
        return dit_model
    
    def _init_vae_model(self):
        """初始化 VAE（变分自编码器）"""
        from sglang.multimodal_gen.runtime.models.vaes import WanVAEModel
        
        vae_model = WanVAEModel(
            config=self.vae_config.arch_config,
            encoder_only=(not self.vae_config.load_decoder),
            decoder_only=(not self.vae_config.load_encoder),
        )
        
        # 加载预训练权重
        self._load_weights_with_mapping(
            vae_model,
            weight_mapping=self.vae_config.arch_config.param_names_mapping
        )
        
        return vae_model
    
    def _init_text_encoder(self):
        """初始化文本编码器 (T5)"""
        from transformers import AutoModel
        
        # 从配置获取文本编码器参数
        text_config = self.pipeline_config.text_encoder_configs[0]
        
        text_encoder = AutoModel.from_pretrained(
            self.server_args.model_path,  # 父目录包含 text_encoder config
            trust_remote_code=True,
        )
        
        return text_encoder
    
    def _init_scheduler(self):
        """初始化去噪调度器"""
        from sglang.multimodal_gen.runtime.models.schedulers import (
            FlowUniPCMultistepScheduler,
        )
        
        scheduler = FlowUniPCMultistepScheduler(
            shift=self.pipeline_config.flow_shift,
            # ... 其他参数从 config.json 或 scheduler config 获取
        )
        
        return scheduler
    
    def _load_weights_with_mapping(self, model, weight_mapping):
        """使用参数名映射加载权重"""
        import re
        
        # 加载 HF 权重
        pretrained_dict = self._load_hf_weights()
        
        # 创建新的权重字典，应用映射
        state_dict = {}
        for hf_name, weight in pretrained_dict.items():
            # 尝试应用映射规则
            sglang_name = self._apply_mapping(hf_name, weight_mapping)
            state_dict[sglang_name] = weight
        
        # 加载到模型
        model.load_state_dict(state_dict, strict=False)
    
    def _apply_mapping(self, hf_name, mapping_rules):
        """应用正则表达式映射规则"""
        import re
        
        for pattern, replacement in mapping_rules.items():
            if re.match(pattern, hf_name):
                return re.sub(pattern, replacement, hf_name)
        
        # 如果没有映射规则匹配，使用原始名称
        return hf_name
```

---

## 权重名映射详解

### 映射规则完整列表

```python
# 来自 WanVideoArchConfig.param_names_mapping
PARAM_NAMES_MAPPING = {
    # === Patch Embedding ===
    r"^patch_embedding\.(.*)$": r"patch_embedding.proj.\1",
    
    # === Condition Embedder - Text ===
    r"^condition_embedder\.text_embedder\.linear_1\.(.*)$": 
        r"condition_embedder.text_embedder.fc_in.\1",
    r"^condition_embedder\.text_embedder\.linear_2\.(.*)$": 
        r"condition_embedder.text_embedder.fc_out.\1",
    
    # === Condition Embedder - Time ===
    r"^condition_embedder\.time_embedder\.linear_1\.(.*)$": 
        r"condition_embedder.time_embedder.mlp.fc_in.\1",
    r"^condition_embedder\.time_embedder\.linear_2\.(.*)$": 
        r"condition_embedder.time_embedder.mlp.fc_out.\1",
    r"^condition_embedder\.time_proj\.(.*)$": 
        r"condition_embedder.time_modulation.linear.\1",
    
    # === Condition Embedder - Image (for I2V) ===
    r"^condition_embedder\.image_embedder\.ff\.net\.0\.proj\.(.*)$": 
        r"condition_embedder.image_embedder.ff.fc_in.\1",
    r"^condition_embedder\.image_embedder\.ff\.net\.2\.(.*)$": 
        r"condition_embedder.image_embedder.ff.fc_out.\1",
    
    # === Attention Layers ===
    # 自注意力 (Self-Attention)
    r"^blocks\.(\d+)\.attn1\.to_q\.(.*)$": r"blocks.\1.to_q.\2",
    r"^blocks\.(\d+)\.attn1\.to_k\.(.*)$": r"blocks.\1.to_k.\2",
    r"^blocks\.(\d+)\.attn1\.to_v\.(.*)$": r"blocks.\1.to_v.\2",
    r"^blocks\.(\d+)\.attn1\.to_out\.0\.(.*)$": r"blocks.\1.to_out.\2",
    
    # 注意力规范化
    r"^blocks\.(\d+)\.attn1\.norm_q\.(.*)$": r"blocks.\1.norm_q.\2",
    r"^blocks\.(\d+)\.attn1\.norm_k\.(.*)$": r"blocks.\1.norm_k.\2",
    
    # 局部注意力投影
    r"^blocks\.(\d+)\.attn1\.attn_op\.local_attn\.proj_l\.(.*)$": 
        r"blocks.\1.attn1.local_attn.proj_l.\2",
    
    # 跨注意力 (Cross-Attention)
    r"^blocks\.(\d+)\.attn2\.to_out\.0\.(.*)$": r"blocks.\1.attn2.to_out.\2",
    
    # === FFN Layers ===
    r"^blocks\.(\d+)\.ffn\.net\.0\.proj\.(.*)$": r"blocks.\1.ffn.fc_in.\2",
    r"^blocks\.(\d+)\.ffn\.net\.2\.(.*)$": r"blocks.\1.ffn.fc_out.\2",
    
    # === Layer Normalization ===
    r"^blocks\.(\d+)\.norm2\.(.*)$": r"blocks.\1.self_attn_residual_norm.norm.\2",
}
```

### 映射示例

```
原始 HF 权重名称:
├── patch_embedding.weight
├── patch_embedding.bias
├── condition_embedder.text_embedder.linear_1.weight
├── condition_embedder.text_embedder.linear_2.weight
├── blocks.0.attn1.to_q.weight
├── blocks.0.attn1.to_k.weight
├── blocks.0.attn1.to_v.weight
├── blocks.0.attn1.to_out.0.weight
├── blocks.0.attn1.norm_q.weight
├── blocks.0.attn1.norm_k.weight
├── blocks.0.ffn.net.0.proj.weight
├── blocks.0.ffn.net.2.weight
└── blocks.0.norm2.weight

↓ 应用映射规则

SGLang 内部名称:
├── patch_embedding.proj.weight
├── patch_embedding.proj.bias
├── condition_embedder.text_embedder.fc_in.weight
├── condition_embedder.text_embedder.fc_out.weight
├── blocks.0.to_q.weight
├── blocks.0.to_k.weight
├── blocks.0.to_v.weight
├── blocks.0.to_out.weight
├── blocks.0.norm_q.weight
├── blocks.0.norm_k.weight
├── blocks.0.ffn.fc_in.weight
├── blocks.0.ffn.fc_out.weight
└── blocks.0.self_attn_residual_norm.norm.weight
```

### LoRA 权重映射

```python
LORA_PARAM_NAMES_MAPPING = {
    # 原始官方 repo 中的 LoRA 参数名 → HF 格式
    r"^blocks\.(\d+)\.self_attn\.q\.(.*)$": r"blocks.\1.attn1.to_q.\2",
    r"^blocks\.(\d+)\.self_attn\.k\.(.*)$": r"blocks.\1.attn1.to_k.\2",
    r"^blocks\.(\d+)\.self_attn\.v\.(.*)$": r"blocks.\1.attn1.to_v.\2",
    r"^blocks\.(\d+)\.self_attn\.o\.(.*)$": r"blocks.\1.attn1.to_out.0.\2",
    r"^blocks\.(\d+)\.cross_attn\.q\.(.*)$": r"blocks.\1.attn2.to_q.\2",
    r"^blocks\.(\d+)\.cross_attn\.k\.(.*)$": r"blocks.\1.attn2.to_k.\2",
    r"^blocks\.(\d+)\.cross_attn\.v\.(.*)$": r"blocks.\1.attn2.to_v.\2",
    r"^blocks\.(\d+)\.cross_attn\.o\.(.*)$": r"blocks.\1.attn2.to_out.0.\2",
    r"^blocks\.(\d+)\.ffn\.0\.(.*)$": r"blocks.\1.ffn.fc_in.\2",
    r"^blocks\.(\d+)\.ffn\.2\.(.*)$": r"blocks.\1.ffn.fc_out.\2",
}
```

---

## 模型初始化流程

### 完整初始化伪代码

```python
class WanVideoModel(CachableDiT):
    """Wan 视频 DiT 实现"""
    
    def __init__(self, config: WanVideoArchConfig):
        super().__init__()
        self.config = config
        
        # === 1. 嵌入层初始化 ===
        
        # Patch Embedding
        self.patch_embedding = PatchEmbed(
            in_channels=config.in_channels,      # 16 (VAE 潜在空间)
            embed_dim=config.hidden_size,        # 5120 = 40 * 128
            patch_size=config.patch_size,        # (1, 2, 2)
        )
        
        # 条件嵌入 (Conditioning Embeddings)
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=config.hidden_size,              # 5120
            time_freq_dim=config.freq_dim,       # 256
            text_embed_dim=config.text_dim,      # 4096
        )
        
        # === 2. Transformer 块初始化 ===
        
        self.blocks = nn.ModuleList([
            WanTransformerBlock(
                dim=config.hidden_size,          # 5120
                num_heads=config.num_attention_heads,  # 40
                attention_head_dim=config.attention_head_dim,  # 128
                ffn_dim=config.ffn_dim,          # 13824
                cross_attn_norm=config.cross_attn_norm,
                qk_norm=config.qk_norm,
                eps=config.eps,
                # ... 其他参数
            )
            for _ in range(config.num_layers)     # 40 层
        ])
        
        # === 3. 输出投影 ===
        
        self.out_proj = nn.Linear(
            config.hidden_size,                  # 5120
            config.out_channels * np.prod(config.patch_size),  # 16 * 4
        )
    
    def forward(
        self,
        x: torch.Tensor,              # [batch, seq_len, hidden]
        timesteps: torch.Tensor,      # [batch]
        context: torch.Tensor,        # [batch, context_len, text_dim]
        context_lens: torch.Tensor,   # [batch]
    ) -> torch.Tensor:
        """
        Args:
            x: 输入潜在表示 [batch, frames, height, width, 16]
            timesteps: 去噪步骤 [batch]
            context: 文本嵌入 [batch, seq_len, 4096]
            context_lens: 有效文本长度 [batch]
        
        Returns:
            out: 预测噪声 (与 x 形状相同)
        """
        
        # 1. 补丁嵌入 (Patch Embedding)
        x = self.patch_embedding(x)              # → [batch, num_patches, hidden_size]
        
        # 2. 条件嵌入 (Conditioning)
        cond = self.condition_embedder(
            timesteps=timesteps,
            context_text=context,
            context_lens=context_lens,
        )                                         # → [batch, hidden_size]
        
        # 3. Transformer 块处理
        for block in self.blocks:
            x = block(
                x=x,
                cond=cond,
                context=context,
                context_lens=context_lens,
            )
        
        # 4. 输出投影
        out = self.out_proj(x)                    # → [batch, num_patches, out_channels * patch_prod]
        
        # 5. 重塑为原始形状
        out = out.reshape(
            batch_size,
            num_frames,
            height,
            width,
            self.config.out_channels,            # 16
        )
        
        return out
```

### 关键计算维度

```
输入：[batch=2, frames=9, height=30, width=40, channels=16]  (VAE 潜在)

补丁嵌入：
  patch_size = (1, 2, 2)
  num_patches = frames/1 * height/2 * width/2 = 9 * 15 * 20 = 2700

Transformer 块处理：
  [batch=2, num_patches=2700, hidden_size=5120]
  
  每个块的内部计算：
  - 自注意力: heads=40, head_dim=128
    query: [batch*heads, num_patches, head_dim] = [80, 2700, 128]
  - 跨注意力: 与文本交互
    context: [batch*heads, text_len, head_dim]
  - FFN: hidden → ffn_dim(13824) → hidden

输出：[batch=2, num_patches=2700, hidden_size=5120]

重塑：[batch=2, frames=9, height=30, width=40, channels=16]
```

---

## SGLang 配置类完整定义

### 基类继承链

```
ArchConfig (base.py)
    ↓
DiTArchConfig (dits/base.py)
    ├── param_names_mapping: dict
    ├── lora_param_names_mapping: dict
    ├── supported_attention_backends: set
    └── ...
    
    ↓
WanVideoArchConfig (wanvideo.py)
    ├── 参数映射规则
    ├── 架构参数 (40 个字段)
    └── MoE/因果注意力参数

---

ModelConfig (base.py)
    ↓
DiTConfig (dits/base.py)
    
    ↓
WanVideoConfig
    └── arch_config: WanVideoArchConfig

---

PipelineConfig (base.py)
    ↓
WanT2V480PConfig
    ├── dit_config: WanVideoConfig
    ├── vae_config: WanVAEConfig
    ├── 调度器参数
    └── 精度配置
    
    ↓
├── WanT2V720PConfig (720p 变体)
│   └── 修改分辨率和流移位参数
│
├── Wan2_2_T2V_A14B_Config (A14B 变体)
├── Wan2_2_I2V_A14B_Config (I2V 变体)
└── Wan2_2_TI2V_5B_Config (TI2V 变体)
```

### 完整类定义

```python
# ============ 基类 ============

@dataclass
class DiTArchConfig(ArchConfig):
    """DiT 架构基类配置"""
    _fsdp_shard_conditions: list = field(default_factory=list)
    _compile_conditions: list = field(default_factory=list)
    param_names_mapping: dict = field(default_factory=dict)
    lora_param_names_mapping: dict = field(default_factory=dict)
    reverse_param_names_mapping: dict = field(default_factory=dict)
    _supported_attention_backends: set[AttentionBackendEnum] = field(
        default_factory=lambda: {
            AttentionBackendEnum.SLIDING_TILE_ATTN,
            AttentionBackendEnum.SAGE_ATTN,
            AttentionBackendEnum.FA,
            AttentionBackendEnum.TORCH_SDPA,
            AttentionBackendEnum.VMOBA_ATTN,
        }
    )


@dataclass
class DiTConfig(ModelConfig):
    """DiT 配置基类"""
    arch_config: DiTArchConfig = field(default_factory=DiTArchConfig)
    prefix: str = ""
    quant_config: QuantizationConfig | None = None


@dataclass
class VAEArchConfig(ArchConfig):
    """VAE 架构基类配置"""
    scaling_factor: float | torch.Tensor = 0
    temporal_compression_ratio: int = 4
    spatial_compression_ratio: int = 8


@dataclass
class VAEConfig(ModelConfig):
    """VAE 配置基类"""
    arch_config: VAEArchConfig = field(default_factory=VAEArchConfig)
    load_encoder: bool = True
    load_decoder: bool = True
    tile_sample_min_height: int = 256
    tile_sample_min_width: int = 256
    tile_sample_min_num_frames: int = 16
    tile_sample_stride_height: int = 192
    tile_sample_stride_width: int = 192
    tile_sample_stride_num_frames: int = 12
    use_tiling: bool = True
    use_temporal_tiling: bool = True
    use_parallel_tiling: bool = True
    use_temporal_scaling_frames: bool = True


# ============ Wan 特定类 ============

@dataclass
class WanVideoArchConfig(DiTArchConfig):
    """Wan 视频 DiT 架构配置"""
    
    # 权重映射规则
    param_names_mapping: dict = field(
        default_factory=lambda: {
            r"^patch_embedding\.(.*)$": r"patch_embedding.proj.\1",
            r"^condition_embedder\.text_embedder\.linear_1\.(.*)$": 
                r"condition_embedder.text_embedder.fc_in.\1",
            # ... 完整列表见上面
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
    
    # MoE 参数
    boundary_ratio: float | None = None
    
    # 因果注意力参数
    local_attn_size: int = -1
    sink_size: int = 0
    num_frames_per_block: int = 3
    sliding_window_num_frames: int = 21
    attention_type: str = "original"
    sla_topk: float = 0.1
    
    def __post_init__(self):
        # 计算派生参数
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        self.num_channels_latents = self.out_channels


@dataclass
class WanVideoConfig(DiTConfig):
    """Wan 视频 DiT 完整配置"""
    arch_config: DiTArchConfig = field(default_factory=WanVideoArchConfig)
    prefix: str = "Wan"


@dataclass
class WanVAEArchConfig(VAEArchConfig):
    """Wan VAE 架构配置"""
    
    base_dim: int = 96
    decoder_base_dim: int | None = None
    z_dim: int = 16
    dim_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    attn_scales: tuple[float, ...] = ()
    temperal_downsample: tuple[bool, ...] = (False, True, True)
    dropout: float = 0.0
    
    # 规范化参数
    latents_mean: tuple[float, ...] = (-0.7571, -0.7089, ...)
    latents_std: tuple[float, ...] = (2.8184, 1.4541, ...)
    
    scale_factor_temporal: int = 4
    scale_factor_spatial: int = 8
    
    def __post_init__(self):
        self.scaling_factor = 1.0 / torch.tensor(self.latents_std)
        self.shift_factor = torch.tensor(self.latents_mean)


@dataclass
class WanVAEConfig(VAEConfig):
    """Wan VAE 完整配置"""
    arch_config: WanVAEArchConfig = field(default_factory=WanVAEArchConfig)
    use_feature_cache: bool = True
    use_tiling: bool = False
    use_temporal_tiling: bool = False
    use_parallel_tiling: bool = False
    use_parallel_encode: bool = True
    use_parallel_decode: bool = True


@dataclass
class WanT2V480PConfig(PipelineConfig):
    """Wan T2V 480p Pipeline 配置"""
    
    # 模型配置
    dit_config: DiTConfig = field(default_factory=WanVideoConfig)
    vae_config: VAEConfig = field(default_factory=WanVAEConfig)
    
    # VAE 处理
    vae_tiling: bool = False
    vae_sp: bool = False
    
    # 去噪
    flow_shift: float | None = 3.0
    
    # 文本编码
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (T5Config(),)
    )
    postprocess_text_funcs: tuple[Callable, ...] = field(
        default_factory=lambda: (t5_postprocess_text,)
    )
    
    # 精度
    precision: str = "bf16"
    vae_precision: str = "fp32"
    text_encoder_precisions: tuple[str, ...] = ("fp32",)


@dataclass
class Wan2_2_T2V_A14B_Config(WanT2V480PConfig):
    """Wan 2.2 T2V A14B 具体配置"""
    # 继承所有父类配置
    # 可在此覆盖具体参数
    pass


@dataclass
class Wan2_2_I2V_A14B_Config(WanI2V720PConfig):
    """Wan 2.2 I2V A14B 具体配置"""
    pass


@dataclass
class Wan2_2_TI2V_5B_Config(WanT2V480PConfig, WanI2VCommonConfig):
    """Wan 2.2 TI2V 5B 具体配置"""
    pass
```

---

## 推理参数配置示例

### 1. 采样参数配置

```python
@dataclass
class WanT2V_14B_SamplingParams(SamplingParams):
    """Wan T2V 14B 采样参数"""
    
    # 视频分辨率
    height: int = 720
    width: int = 1280
    num_frames: int = 81
    fps: int = 16
    
    # 去噪参数
    guidance_scale: float = 5.0
    negative_prompt: str = "..."
    num_inference_steps: int = 50
    
    # 支持的分辨率
    supported_resolutions: list[tuple[int, int]] | None = field(
        default_factory=lambda: [
            (1280, 720),  # 16:9
            (720, 1280),  # 9:16
            (832, 480),   # 16:9
            (480, 832),   # 9:16
        ]
    )
    
    # TeaCache 加速
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

### 2. 完整启动参数示例

```json
{
  "model_path": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
  "model_name": "wan2.2-t2v-a14b",
  "dtype": "bfloat16",
  "device": "cuda",
  "num_gpu_per_node": 1,
  "parallelization": "single",
  
  "text_encoder_cpu_offload": true,
  "pin_cpu_memory": true,
  
  "height": 720,
  "width": 1280,
  "num_frames": 81,
  "fps": 16,
  
  "guidance_scale": 5.0,
  "num_inference_steps": 50,
  
  "flow_shift": 3.0,
  
  "enable_teacache": true,
  "teacache_thresh": 0.08,
  
  "enable_vmoba": false,
  
  "precision": "bf16",
  "vae_precision": "fp32",
  "text_encoder_precision": "fp32"
}
```

### 3. TeaCache 参数详解

```python
@dataclass
class TeaCacheParams:
    """Token Economy Aware Cache 加速参数"""
    
    # 缓存阈值 (0-1)
    teacache_thresh: float = 0.08
    
    # 是否使用重新时间化步数
    use_ret_steps: bool = True
    
    # 系数计算回调
    coefficients_callback: Callable[[TeaCacheParams], list[float]] = None
    
    # 开始跳过的步数
    start_skipping: int = 5
    
    # 停止跳过的步数（相对结束）
    end_skipping: float = 1.0

# A14B 模型的系数（来自官方 TeaCache 实现）
def _wan_14b_coefficients(p: TeaCacheParams) -> list[float]:
    if p.use_ret_steps:
        # 从官方 TeaCache repo 的 Wan2.1 配置
        return [
            -3.03318725e05,
            4.90537029e04,
            -2.65530556e03,
            5.87365115e01,
            -3.15583525e-01,
        ]
    return [-5784.54975374, 5449.50911966, -1811.16591783, 256.27178429, -13.02252404]
```

### 4. Vmoba 注意力优化参数

```python
@dataclass
class VmobaAttentionConfig:
    """Vmoba (Variational Motion-Aware Bottom-up Approximation) 注意力优化"""
    
    # 时间维度参数
    temporal_chunk_size: int = 2        # 时间块大小
    temporal_topk: int = 3              # 时间 Top-K
    
    # 空间维度参数
    spatial_chunk_size: list[int] = [3, 4]  # 空间块大小 [H, W]
    spatial_topk: int = 20              # 空间 Top-K
    
    # 时空联合参数
    st_chunk_size: list[int] = [4, 6, 4]    # 时空块大小 [T, H, W]
    st_topk: int = 15                   # 时空 Top-K
    
    # 选择策略
    moba_select_mode: str = "threshold"  # "threshold" 或 "topk"
    moba_threshold: float = 0.25         # 注意力值阈值
    moba_threshold_type: str = "query_head"  # 按 query-head 应用阈值
    
    # 完整注意力层配置
    first_full_layer: int = 0            # 第一个全注意力层
    first_full_step: int = 12            # 第一个全注意力步数
    
    # 按维度分别优化的层
    temporal_layer: int = 1
    spatial_layer: int = 1
    st_layer: int = 1
```

---

## 配置验证检查清单

### 初始化验证

```python
def verify_config_consistency(pipeline_config):
    """验证配置一致性"""
    
    checks = []
    
    # 1. 文本维度一致性
    expected_text_dim = 4096  # T5 输出维度
    actual_text_dim = pipeline_config.dit_config.arch_config.text_dim
    checks.append(f"Text dim match: {actual_text_dim} == {expected_text_dim}")
    
    # 2. VAE 潜在维度一致性
    expected_z_dim = 16
    actual_z_dim = pipeline_config.vae_config.arch_config.z_dim
    actual_in_channels = pipeline_config.dit_config.arch_config.in_channels
    checks.append(f"Z-dim match: {actual_z_dim} == {actual_in_channels}")
    
    # 3. 隐藏维度计算
    num_heads = pipeline_config.dit_config.arch_config.num_attention_heads
    head_dim = pipeline_config.dit_config.arch_config.attention_head_dim
    expected_hidden = num_heads * head_dim
    actual_hidden = pipeline_config.dit_config.arch_config.hidden_size
    checks.append(f"Hidden size: {actual_hidden} == {expected_hidden}")
    
    # 4. 规范化参数长度
    latents_mean_len = len(pipeline_config.vae_config.arch_config.latents_mean)
    latents_std_len = len(pipeline_config.vae_config.arch_config.latents_std)
    checks.append(f"Latent params length: mean={latents_mean_len}, std={latents_std_len}")
    
    # 5. 精度设置
    main_precision = pipeline_config.precision
    vae_precision = pipeline_config.vae_precision
    checks.append(f"Precision: main={main_precision}, vae={vae_precision}")
    
    return checks
```

---

## 总结表格

### 配置参数快速查询

| 参数类型 | 参数名 | 来源 | 数据类型 | 默认值 |
|--------|-------|------|--------|--------|
| DiT 架构 | `num_attention_heads` | HF config | int | 40 |
| DiT 架构 | `attention_head_dim` | HF config | int | 128 |
| DiT 架构 | `num_layers` | HF config | int | 40 |
| DiT 架构 | `ffn_dim` | HF config | int | 13824 |
| DiT 架构 | `text_dim` | HF config | int | 4096 |
| DiT 架构 | `freq_dim` | HF config | int | 256 |
| VAE 架构 | `z_dim` | HF config | int | 16 |
| VAE 架构 | `scale_factor_spatial` | HF config | int | 8 |
| VAE 架构 | `scale_factor_temporal` | HF config | int | 4 |
| Pipeline | `flow_shift` | Pipeline config | float | 3.0/5.0 |
| Pipeline | `precision` | Pipeline config | str | "bf16" |
| Sampling | `num_inference_steps` | SamplingParams | int | 50 |
| Sampling | `guidance_scale` | SamplingParams | float | 3.0-5.0 |
| Optimization | `teacache_thresh` | SamplingParams | float | 0.08 |
| Backend | `temporal_chunk_size` | Backend config | int | 2 |

