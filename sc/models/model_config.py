from transformers import ViTConfig
from typing import Optional


class MCViTConfig(ViTConfig):
    """
    Configuration class for Multi-Channel Vision Transformer (MCViT).
    
    This model extends standard ViT to handle multi-channel inputs (e.g., 6-channel
    satellite imagery) with a fixed number of channels defined at initialization.
    
    Args:
        in_channels (`int`, *optional*, defaults to 6):
            Number of input channels. Unlike standard ViT (3 channels for RGB),
            MCViT supports arbitrary channel counts.
        use_instance_norm (`bool`, *optional*, defaults to `True`):
            Whether to apply instance normalization to input images before
            patch embedding. Recommended for multi-channel data with varying
            intensity distributions.
        pixel_scale (`float`, *optional*, defaults to 255.0):
            Scale factor for normalizing pixel values. Input pixels are divided
            by this value. Set to 1.0 if inputs are already normalized.
        use_cls_token (`bool`, *optional*, defaults to `False`):
            Whether to use a [CLS] token for pooling. If False, global average
            pooling is used instead.
        use_flash_attention (`bool`, *optional*, defaults to `True`):
            Whether to use Flash Attention 2 for efficient attention computation.
            Falls back to standard attention if flash_attn is not installed.
            
    Example:
```python
        from src.models import MCViTConfig, MCViT
        
        # 6-channel input (e.g., satellite imagery)
        config = MCViTConfig(
            in_channels=6,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            image_size=224,
            patch_size=16,
        )
        model = MCViT(config)
```
    """
    
    model_type = "multichannel_vit"
    
    def __init__(
        self,
        in_channels: int = 6,
        use_instance_norm: bool = True,
        pixel_scale: float = 255.0,
        use_cls_token: bool = False,
        use_flash_attention: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.use_instance_norm = use_instance_norm
        self.pixel_scale = pixel_scale
        self.use_flash_attention = use_flash_attention
        self.use_cls_token = use_cls_token


class CAViTConfig(ViTConfig):
    """
    Configuration class for Channel-Agnostic Vision Transformer (CAViT).
    
    Unlike MCViT which has a fixed number of channels, CAViT can handle
    variable numbers of input channels at inference time. This is achieved
    through channel-agnostic tokenization where each channel is processed
    independently with shared weights.
    
    Key differences from MCViT:
    - Single-channel patch embedding shared across all channels
    - Position embeddings are spatial only (shared across channels)
    - Total sequence length = num_channels * num_patches
    
    Args:
        in_channels (`int`, *optional*, defaults to 6):
            Default number of input channels. Can be changed at inference.
        use_instance_norm (`bool`, *optional*, defaults to `True`):
            Whether to apply instance normalization per channel.
        pixel_scale (`float`, *optional*, defaults to 255.0):
            Scale factor for normalizing pixel values.
        use_cls_token (`bool`, *optional*, defaults to `False`):
            Whether to use a [CLS] token for pooling.
        use_flash_attention (`bool`, *optional*, defaults to `True`):
            Whether to use Flash Attention 2.
        use_sincos_pos_embed (`bool`, *optional*, defaults to `True`):
            Whether to use fixed sinusoidal position embeddings instead of
            learnable embeddings. Recommended for better generalization to
            different image sizes.
            
    Example:
```python
        from src.models import CAViTConfig, CAViT
        
        config = CAViTConfig(
            in_channels=6,  # Default, but can vary at inference
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
        )
        model = CAViT(config)
        
        # Inference with different channel counts
        out_6ch = model(torch.randn(1, 6, 224, 224))  # 6 channels
        out_4ch = model(torch.randn(1, 4, 224, 224))  # 4 channels (works!)
```
    """
    
    model_type = "channel_agnostic_vit"
    
    def __init__(
        self,
        in_channels: int = 6,
        use_instance_norm: bool = True,
        pixel_scale: float = 255.0,
        use_cls_token: bool = False,
        use_flash_attention: bool = True,
        use_sincos_pos_embed: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.use_instance_norm = use_instance_norm
        self.pixel_scale = pixel_scale
        self.use_flash_attention = use_flash_attention
        self.use_cls_token = use_cls_token
        self.use_sincos_pos_embed = use_sincos_pos_embed