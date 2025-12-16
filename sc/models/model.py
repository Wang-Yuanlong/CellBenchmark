import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from transformers import PreTrainedModel, ViTConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling
from typing import Optional, Literal, Union, Tuple, Any
import math
import warnings
from flash_attn import flash_attn_qkvpacked_func as fa_qkv
from flash_attn import flash_attn_func
import re
import timm
import time


class MCViTConfig(ViTConfig):
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
    Configuration for Channel-Agnostic ViT.
    
    Key differences from standard ViT:
    - Supports variable number of input channels
    - Channel-agnostic tokenization
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
        

class ScaledDotProductAttention(nn.Module):
    """
    Unified attention module with Flash Attention 2 (packed QKV) support.
    
    Uses flash_attn_qkvpacked_func for optimal performance when available.
    """
    
    def __init__(self, use_flash_attention: bool = True):
        super().__init__()
        self.use_flash_attention = use_flash_attention
        self._flash_available = self._check_flash_available()
        
        if self.use_flash_attention and not self._flash_available:
            warnings.warn(
                "Flash Attention requested but not available. "
                "Falling back to standard attention."
            )
    
    @staticmethod
    def _check_flash_available() -> bool:
        """Check if Flash Attention 2 is available."""
        try:
            from flash_attn import flash_attn_qkvpacked_func
            return True
        except ImportError:
            return False
    
    @property
    def using_flash(self) -> bool:
        """Whether Flash Attention will actually be used."""
        return self.use_flash_attention and self._flash_available
    
    def forward(
        self,
        qkv: torch.Tensor,
        dropout_p: float = 0.0,
        softmax_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute attention with packed QKV.
        
        Args:
            qkv: [batch, seqlen, 3, num_heads, head_dim]
            dropout_p: Dropout probability
            softmax_scale: Optional scale factor (default: 1/sqrt(head_dim))
            
        Returns:
            output: [batch, seqlen, num_heads, head_dim]
        """
        head_dim = qkv.shape[-1]
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(head_dim)
        
                                                       
        can_use_flash = (
            self.using_flash
            and qkv.is_cuda
            and qkv.dtype in (torch.float16, torch.bfloat16)
        )
        
        if can_use_flash:
            return self._flash_attention_packed(qkv, dropout_p, softmax_scale)
        else:
            return self._standard_attention_packed(qkv, dropout_p, softmax_scale)
    
    def _flash_attention_packed(
        self,
        qkv: torch.Tensor,
        dropout_p: float,
        softmax_scale: float,
    ) -> torch.Tensor:
        """Flash Attention 2 with packed QKV."""
        from flash_attn import flash_attn_qkvpacked_func
        
        qkv = qkv.contiguous()
        return flash_attn_qkvpacked_func(
            qkv,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=False
        )
    
    def _standard_attention_packed(
        self,
        qkv: torch.Tensor,
        dropout_p: float,
        softmax_scale: float,
    ) -> torch.Tensor:
        """Standard attention with packed QKV input."""
                                                         
        q, k, v = qkv.unbind(dim=2)
        
        batch_size, seq_len, num_heads, head_dim = q.shape
        
                                          
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
                            
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        if dropout_p > 0.0 and self.training:
            attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)
        
        output = torch.matmul(attn_weights, v)
        
                                          
        return output.transpose(1, 2)
    
    
    
class ViTAttention(nn.Module):
    """Vision Transformer attention layer."""
    
    def __init__(self, config: MCViTConfig):
        super().__init__()
        
        assert config.hidden_size % config.num_attention_heads == 0, (
            f"hidden_size ({config.hidden_size}) must be divisible by "
            f"num_attention_heads ({config.num_attention_heads})"
        )
        
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.softmax_scale = 1.0 / math.sqrt(self.head_dim)
        
                        
        self.qkv = nn.Linear(config.hidden_size, 3 * self.all_head_size)
        self.proj_out = nn.Linear(self.all_head_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
                           
        self.attention = ScaledDotProductAttention(
            use_flash_attention=config.use_flash_attention
        )
        
                                               
        if config.use_flash_attention and self.attention.using_flash:
            assert self.head_dim % 8 == 0 and self.head_dim <= 256, (
                f"Flash Attention requires head_dim % 8 == 0 and head_dim <= 256, "
                f"got head_dim={self.head_dim}"
            )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, S, H]
            
        Returns:
            output: [B, S, H]
        """
        B, S, _ = hidden_states.shape
        dropout_p = self.dropout.p if self.training else 0.0
        
                                                              
        qkv = self.qkv(hidden_states)
        qkv = qkv.view(B, S, 3, self.num_heads, self.head_dim)
        
                                   
        context = self.attention(qkv, dropout_p, self.softmax_scale)                  
        
                             
        context = context.reshape(B, S, self.all_head_size)
        output = self.proj_out(context)
        
        return output
    
    
    
class ViTLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.attention = ViTAttention(config)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act = nn.GELU()
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layernorm_before(hidden_states)
        attention_output = self.attention(hidden_states)
        attention_output = self.attention_dropout(attention_output)
        hidden_states = residual + attention_output
        
        residual = hidden_states
        hidden_states = self.layernorm_after(hidden_states)
        hidden_states = self.intermediate(hidden_states)
        hidden_states = self.intermediate_act(hidden_states)
        hidden_states = self.output(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

class MCPatchEmbedding(nn.Module):
    def __init__(self, config: MCViTConfig):
        super().__init__()
        self.config = config
        self.in_channels = config.in_channels
        self.hidden_size = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.num_patches = (config.image_size // config.patch_size) ** 2
        
        self.pixel_scale = config.pixel_scale

        self.use_instance_norm = config.use_instance_norm
        if self.use_instance_norm:
            self.instance_norm = nn.InstanceNorm2d(
                self.in_channels,
                affine=False,
                track_running_stats=False
            )
        
        self.projection = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, C, H, W] where C=6
        Returns:
            embeddings: [B, num_patches, hidden_size]
        """
        if self.pixel_scale != 1.0:
            pixel_values = pixel_values / self.pixel_scale
        
        if self.use_instance_norm:
            pixel_values = self.instance_norm(pixel_values)
        
        embeddings = self.projection(pixel_values)
        
        batch_size = embeddings.shape[0]
        embeddings = embeddings.flatten(2).transpose(1, 2)
        
        return embeddings

class MCViT(PreTrainedModel):
    config_class = MCViTConfig
    base_model_prefix = "vit"
    supports_gradient_checkpointing = True
    
    def __init__(self, config: MCViTConfig):
        super().__init__(config)
        
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_patches = (config.image_size // config.patch_size) ** 2
        
                         
        self.patch_embedding = MCPatchEmbedding(config)
        
                   
        if config.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
       
            self.position_embeddings = nn.Parameter(
                torch.zeros(1, self.num_patches + 1, config.hidden_size)
            )
        else:
            self.position_embeddings = nn.Parameter(
                torch.zeros(1, self.num_patches, config.hidden_size)
            )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.layers = nn.ModuleList([
            ViTLayer(config) for _ in range(config.num_hidden_layers)
        ])
        
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        if config.use_flash_attention:
            try:
                import flash_attn
                print(f"Flash Attention 2 enabled (version {flash_attn.__version__})")
            except ImportError:
                warnings.warn("flash_attn not available, using standard attention")
                config.use_flash_attention = False
        
        self.post_init()
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Parameter):
            module.data.normal_(mean=0.0, std=self.config.initializer_range)
            
    def _interpolate_pos_embed(self, num_patches_new: int) -> torch.Tensor:
        pe = self.position_embeddings                             
        D = pe.shape[-1]

        use_cls = getattr(self.config, "use_cls_token", True)
        if use_cls:
            cls_pos = pe[:, :1, :]                    
            patch_pos = pe[:, 1:, :]                   
        else:
            cls_pos = None
            patch_pos = pe                             

        N0 = patch_pos.shape[1]
        if N0 == num_patches_new:
            return pe if use_cls else patch_pos

        gs0 = int(math.sqrt(N0))
        gs1 = int(math.sqrt(num_patches_new))
        assert gs0 * gs0 == N0 and gs1 * gs1 == num_patches_new, "num_patches should be a perfect square"

                                   
        patch_pos = patch_pos.reshape(1, gs0, gs0, D).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(gs1, gs1), mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, gs1 * gs1, D)

        if use_cls:
            return torch.cat([cls_pos, patch_pos], dim=1)  
        else:
            return patch_pos                              

    def _add_pos_and_cls(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        B, N, D = patch_embeddings.shape
        pos = self._interpolate_pos_embed(N).to(patch_embeddings.dtype)

        if getattr(self.config, "use_cls_token", True):
            cls_tok = self.cls_token.to(patch_embeddings.dtype).expand(B, 1, D)
            tokens = torch.cat([cls_tok, patch_embeddings], dim=1)             
        else:
            tokens = patch_embeddings                                         

        tokens = tokens + pos                                        
        return tokens

    
    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        """
        Args:
            pixel_values: [B, 6, H, W]
        Returns:
            BaseModelOutputWithPooling
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        batch_size = pixel_values.shape[0]
        
        embeddings = self.patch_embedding(pixel_values)

        embeddings = self._add_pos_and_cls(embeddings)
                                       
                                                                   
                                                                    

                                      
                                                            
        
        embeddings = self.dropout(embeddings)
        
                               
        hidden_states = embeddings
        all_hidden_states = () if output_hidden_states else None
        
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states = layer(hidden_states)
        
                            
        hidden_states = self.layernorm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        pooled_output = hidden_states[:, 0] if self.config.use_cls_token else hidden_states.mean(dim=1)
        
        if not return_dict:
            return (hidden_states, pooled_output, all_hidden_states)
        
        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=pooled_output,
            hidden_states=all_hidden_states,
        )

def mcvit_small(
    in_channels: int = 6,
    image_size: int = 224,
    use_cls_token: bool = False,
    use_flash_attention: bool = True,
    use_instance_norm: bool = True,
    pixel_scale: float = 255.0,
    **kwargs: Any
) -> MCViT:
    config = MCViTConfig(
        in_channels=in_channels,
        image_size=image_size,
        patch_size=16,
        hidden_size=384,
        num_hidden_layers=12,
        num_attention_heads=6,
        intermediate_size=1536,
        use_cls_token=use_cls_token,
        use_flash_attention=use_flash_attention,
        use_instance_norm=use_instance_norm,
        pixel_scale=pixel_scale
    )
    return MCViT(config)

def mcvit_base(
    in_channels: int = 6,
    image_size: int = 224,
    use_cls_token: bool = False,
    use_flash_attention: bool = True,
    use_instance_norm: bool = True,
    pixel_scale: float = 255.0,
    **kwargs: Any
) -> MCViT:
    config = MCViTConfig(
        in_channels=in_channels,
        image_size=image_size,
        patch_size=16,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        use_cls_token=use_cls_token,
        use_flash_attention=use_flash_attention,
        use_instance_norm=use_instance_norm,
        pixel_scale=pixel_scale
    )
    return MCViT(config)


def mcvit_large(
    in_channels: int = 6,
    image_size: int = 224,
    use_flash_attention: bool = True,
    use_instance_norm: bool = True,
    pixel_scale: float = 255.0
) -> MCViT:
    config = MCViTConfig(
        in_channels=in_channels,
        image_size=image_size,
        patch_size=16,
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        use_flash_attention=use_flash_attention,
        use_instance_norm=use_instance_norm,
        pixel_scale=pixel_scale
    )
    return MCViT(config)

def load_timm_vit_into_mcvit(model, timm_name='vit_small_patch16_224', pretrained=True, strict=False):
    ref = timm.create_model(timm_name, pretrained=pretrained)
    sd = ref.state_dict()

    drop_keys = [
        'patch_embed.proj.weight', 'patch_embed.proj.bias',
        'pos_embed', 'cls_token',
        'fc_norm.weight', 'fc_norm.bias',  
        'head.weight', 'head.bias'
    ]
    for k in drop_keys:
        if k in sd:
            sd.pop(k)

    renamed = {}
    for k, v in sd.items():
        new_k = k
        new_k = re.sub(r'^blocks\.(\d+)\.norm1\.(weight|bias)$', r'layers.\1.layernorm_before.\2', new_k)
        new_k = re.sub(r'^blocks\.(\d+)\.attn\.qkv\.(weight|bias)$', r'layers.\1.attention.qkv.\2', new_k)
        new_k = re.sub(r'^blocks\.(\d+)\.attn\.proj\.(weight|bias)$', r'layers.\1.attention.proj_out.\2', new_k)
        new_k = re.sub(r'^blocks\.(\d+)\.norm2\.(weight|bias)$', r'layers.\1.layernorm_after.\2', new_k)
        new_k = re.sub(r'^blocks\.(\d+)\.mlp\.fc1\.(weight|bias)$', r'layers.\1.intermediate.\2', new_k)
        new_k = re.sub(r'^blocks\.(\d+)\.mlp\.fc2\.(weight|bias)$', r'layers.\1.output.\2', new_k)
        new_k = re.sub(r'^norm\.(weight|bias)$', r'layernorm.\1', new_k)

        renamed[new_k] = v

    missing, unexpected = model.load_state_dict(renamed, strict=strict)
    print(f'Loaded weights from {timm_name}')
    return model


class CAPatchEmbedding(nn.Module):
    """
    Channel-Agnostic Patch Embedding (Optimized).
    
    Key difference: Uses single-channel Conv2d projection shared across all channels.
    This allows the model to handle variable numbers of channels at inference time.
    
    Optimization: Uses vectorized operations instead of Python loops for speed.
    """
    def __init__(self, config: CAViTConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.num_patches = (config.image_size // config.patch_size) ** 2
        
        self.pixel_scale = config.pixel_scale
        self.use_instance_norm = config.use_instance_norm
        
        if self.use_instance_norm:
                                                  
            self.instance_norm = nn.InstanceNorm2d(
                1,                                        
                affine=False,
                track_running_stats=False
            )

        self.projection = nn.Conv2d(
            in_channels=1,                         
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=True,
        )
    
    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Optimized forward pass using vectorized operations.
        
        Args:
            pixel_values: [B, C, H, W] where C can be any number
        
        Returns:
            embeddings: [B, C*N, D] where:
                - C = num_channels (variable, from input)
                - N = num_patches per channel = (H/patch_size) × (W/patch_size)
                - D = hidden_size
            num_channels: C (number of input channels)
        
        Example:
            Input:  [4, 6, 224, 224]
            Output: [4, 1176, 768], 6
                    where 1176 = 6 × 196 (6 channels × 14×14 patches)
        """
        B, C, H, W = pixel_values.shape
        
                                
        if self.pixel_scale != 1.0:
            pixel_values = pixel_values / self.pixel_scale
        x = pixel_values.view(B * C, 1, H, W)                  
        
                                                                          
        if self.use_instance_norm:
            x = self.instance_norm(x)                  
        
                                                                        
        x = self.projection(x)                                         
        
                               
        _, D, H_out, W_out = x.shape                                         
        
                                                             
        x = x.view(B, C, D, H_out, W_out)
        
                                            
                                                                                      
        x = x.permute(0, 2, 1, 3, 4)                     
        x = x.flatten(2)                                
        x = x.transpose(1, 2)                           
        
        return x, C
    

class CAViT(PreTrainedModel):
    """
    Channel-Agnostic Vision Transformer.
    
    This model can handle variable numbers of input channels by treating each
    channel as a separate modality with shared spatial position embeddings.
    
    Key concepts:
    - num_patches (N): Number of spatial patches PER CHANNEL
    - num_tokens: Total tokens = C × N (varies with input channels C)
    - Position embeddings: Only N embeddings (shared across all channels)
    """
    config_class = CAViTConfig
    base_model_prefix = "ca_vit"
    supports_gradient_checkpointing = True
    def __init__(self, config: CAViTConfig):
        super().__init__(config)
        
        self.config = config
        self.hidden_size = config.hidden_size
        
                                                   
                                                                                      
        self.num_patches = (config.image_size // config.patch_size) ** 2
        
                                          
        self.patch_embedding = CAPatchEmbedding(config)
        
                             
        if config.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
            
        if config.use_sincos_pos_embed:
                                                   
            pos_embed = self._get_sincos_pos_embed(
                config.hidden_size,
                int(self.num_patches ** 0.5)
            )
            self.register_buffer('position_embeddings', pos_embed)
            self._pos_cache = {}  

        else:
                                           
            self.position_embeddings = nn.Parameter(
                torch.zeros(1, self.num_patches, config.hidden_size)
            )
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layers = nn.ModuleList([
            ViTLayer(config) for _ in range(config.num_hidden_layers)
        ])
        
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.post_init()

    def _get_sincos_pos_embed(self, embed_dim: int, grid_size: int) -> torch.Tensor:
        """
        Generate 2D sinusoidal position embeddings (pure PyTorch implementation).
        
        Args:
            embed_dim: Embedding dimension
            grid_size: Grid size (e.g., 14 for 14x14 patches)
        Returns:
            pos_embed: [1, grid_size*grid_size, embed_dim]
        """
                                 
        grid_h = torch.arange(grid_size, dtype=torch.float32)
        grid_w = torch.arange(grid_size, dtype=torch.float32)
        grid = torch.meshgrid(grid_w, grid_h, indexing='xy')                               
        grid = torch.stack(grid, dim=0)                             
        
                                                                                           
        grid = grid.reshape(2, -1)            
        
                                                
        pos_embed = self._get_2d_sincos_pos_embed_from_grid(embed_dim, grid)                    
        
        return pos_embed.unsqueeze(0)                       

    @staticmethod
    def _get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: torch.Tensor) -> torch.Tensor:
        """
        Generate 2D sinusoidal position embeddings from grid coordinates.
        
        Args:
            embed_dim: Embedding dimension (must be even)
            grid: [2, H*W] grid coordinates
        Returns:
            pos_embed: [H*W, embed_dim]
        """
        assert embed_dim % 2 == 0, "embed_dim must be even"
        
                                                                  
        emb_h = CAViT._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])              
        emb_w = CAViT._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])              
        
                     
        pos_embed = torch.cat([emb_h, emb_w], dim=1)            
        return pos_embed


    @staticmethod
    def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
        """
        Generate 1D sinusoidal position embeddings.
        
        Args:
            embed_dim: Output dimension (must be even)
            pos: [H*W] positions
        Returns:
            emb: [H*W, embed_dim]
        """
        assert embed_dim % 2 == 0, "embed_dim must be even"
        
                                  
        omega = torch.arange(embed_dim // 2, dtype=torch.float32)
        omega /= embed_dim / 2.
        omega = 1. / (10000 ** omega)         
        
                                           
        pos = pos.reshape(-1)         
        out = torch.einsum('m,d->md', pos, omega)              
        
                           
        emb_sin = torch.sin(out)              
        emb_cos = torch.cos(out)              
        
                                
        emb = torch.cat([emb_sin, emb_cos], dim=1)            
        return emb
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Parameter):
            module.data.normal_(mean=0.0, std=self.config.initializer_range)
    
    def _interpolate_pos_embed(self, num_patches_new: int) -> torch.Tensor:
        """
        Interpolate position embeddings for different image sizes.
        
        Args:
            num_patches_new: New number of patches
        Returns:
            Interpolated position embeddings
        """
        pe = self.position_embeddings             
        N, D = pe.shape[1], pe.shape[2]
        
        if N == num_patches_new:
            return pe
        
                                              
        gs_old = int(math.sqrt(N))
        gs_new = int(math.sqrt(num_patches_new))
        assert gs_old * gs_old == N and gs_new * gs_new == num_patches_new
        
        pe = pe.reshape(1, gs_old, gs_old, D).permute(0, 3, 1, 2)
        pe = F.interpolate(pe, size=(gs_new, gs_new), mode="bicubic", align_corners=False)
        pe = pe.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, D)
        
        return pe
    
    def _add_channel_agnostic_pos_embed(
        self,
        patch_embeddings: torch.Tensor,
        num_channels: int
    ) -> torch.Tensor:
        """
        Add position embeddings in a channel-agnostic manner.
        
        Key: All channels share the same spatial position embeddings.
        
        Args:
            patch_embeddings: [B, C*N, D] where:
                - C = num_channels (variable, detected from input)
                - N = self.num_patches (fixed, spatial patches per channel)
                - D = hidden_size
            num_channels: C (number of input channels)
        
        Returns:
            embeddings with position: [B, C*N, D] or [B, 1+C*N, D] with cls token
        
        Process:
            1. Get spatial position embeddings: [1, N, D]
            2. Repeat for all channels: [1, C*N, D]
            3. Add to patch embeddings
            4. Optionally prepend CLS token
        """
        B, CN, D = patch_embeddings.shape
        N_in = CN // num_channels                                 
        if self.config.use_sincos_pos_embed:
            if N_in == self.num_patches:
                pos_embed = self.position_embeddings
            else:
                pos_embed_cpu = self._pos_cache.get(N_in)
                if pos_embed_cpu is None:
                    gs = int(math.sqrt(N_in))
                    assert gs * gs == N_in, "Input patches must form a square grid"
                    pos_embed_cpu = self._get_sincos_pos_embed(self.hidden_size, gs).cpu()
                    self._pos_cache[N_in] = pos_embed_cpu
                pos_embed = pos_embed_cpu
            pos_embed = pos_embed.to(patch_embeddings.device, patch_embeddings.dtype)                
        else:
            pos_embed = self._interpolate_pos_embed(N_in).to(
                patch_embeddings.device, patch_embeddings.dtype
            )                


                      
                                                                                                     
        
                                                                               
                                                                                            
        
                                                     
                                                                
        pos_embed = pos_embed.repeat(1, num_channels, 1)               
        
                                 
        patch_embeddings = patch_embeddings + pos_embed
        
                                  
        if self.config.use_cls_token:
            cls_token = self.cls_token.expand(B, 1, D)
            embeddings = torch.cat([cls_token, patch_embeddings], dim=1)                 
        else:
            embeddings = patch_embeddings               
        
        return embeddings
    
   
    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        """
        Forward pass supporting variable number of channels.
        
        Args:
            pixel_values: [B, C, H, W] where C can be any number
        Returns:
            BaseModelOutputWithPooling with:
                - last_hidden_state: [B, C*N, D] where N=num_patches (spatial patches per channel)
                - pooler_output: [B, D]
        
        Note:
            - N = self.num_patches = (image_size // patch_size)^2  (spatial patches per channel)
            - Total tokens = C × N (varies with input channel count C)
            - Position embeddings are shared across channels (only N embeddings)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        B = pixel_values.shape[0]
        
                                             
                                                            
        embeddings, num_channels = self.patch_embedding(pixel_values)               
        
                                                                              
        embeddings = self._add_channel_agnostic_pos_embed(embeddings, num_channels)
        
        embeddings = self.dropout(embeddings)
        
                                                      
        hidden_states = embeddings
        all_hidden_states = () if output_hidden_states else None
        
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states = layer(hidden_states)
        
                            
        hidden_states = self.layernorm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
                        
        if self.config.use_cls_token:
            pooled_output = hidden_states[:, 0]                 
        else:
                                                                
            pooled_output = hidden_states.mean(dim=1)
            

        if not return_dict:
            return (hidden_states, pooled_output, all_hidden_states)
        
        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=pooled_output,
            hidden_states=all_hidden_states,
        )
    
    
def test_mcvit_6channel():
    print("Testing MCViT-6Channel with Flash Attention 2...")

    model = mcvit_small(
        image_size=224,
        use_flash_attention=True,
        pixel_scale=255.0
    )
                  
    
    print(f"✓ Model created")
    print(f"✓ Flash Attention enabled: {model.config.use_flash_attention}")
    
          
    
    x = torch.randn(160, 6, 224, 224) * 255
    x = x.clamp(0, 255)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    x = x.to(device)
                                                             
    start_time = time.time()
                           
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model(x)

    print(f"✓ Forward pass successful")
    print(f"  - CLS token shape: {output.pooler_output.shape}")
    print(f"  - All tokens shape: {output.last_hidden_state.shape}")

    end_time = time.time()
    print(f"✓ Inference time: {end_time - start_time:.4f} seconds")

    model_baseline = vit_small_6ch(
        image_size=224,
        use_flash_attention=False,
        pixel_scale=255.0
    )
    model_baseline.to(device)
    start_time = time.time()
                           
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output_baseline = model_baseline(x)
    end_time = time.time()
    print(f"✓ Baseline Inference time: {end_time - start_time:.4f} seconds")

    return True

def load_timm_vit_into_cavit(
    model: nn.Module,
    timm_name: str = 'vit_base_patch16_224',
    pretrained: bool = True,
    strict: bool = False,
    verbose: bool = True
) -> nn.Module:
    """
    Load pretrained Transformer weights from timm ViT into CA-ViT.
    
    Only Transformer layers are loaded; embedding layers need to be retrained
    due to the channel-agnostic architecture.
    
    Args:
        model: CA-ViT model
        timm_name: Name of timm model to load from
        pretrained: Whether to load pretrained weights
        strict: Whether to strictly match keys (should be False for partial loading)
        verbose: Whether to print detailed information
    
    Returns:
        model: Model with loaded Transformer weights
    
    Example:
        >>> from ca_vit import ca_vit_base
        >>> model = ca_vit_base()
        >>> model = load_timm_vit_into_cavit(model, 'vit_base_patch16_224')
        ✓ Loaded Transformer weights from vit_base_patch16_224
          - Dropped 7 embedding-related keys
          - Loaded 146 Transformer layer keys
          - Missing keys (expected): 4 (embedding layers)
          - Unexpected keys: 0
          - Loaded 12 transformer layers
    """
    if verbose:
        print(f"Loading reference model: {timm_name}")
    
    ref = timm.create_model(timm_name, pretrained=pretrained)
    sd = ref.state_dict()
    
                                                        
    drop_keys = [
        'patch_embed.proj.weight',
        'patch_embed.proj.bias',
        'pos_embed',
        'cls_token',
        'fc_norm.weight',
        'fc_norm.bias',
        'head.weight',
        'head.bias',
        'head_dist.weight',                          
        'head_dist.bias',
    ]
    
    dropped_count = 0
    for k in drop_keys:
        if k in sd:
            sd.pop(k)
            dropped_count += 1
    
                                                   
    renamed = {}
    for k, v in sd.items():
        new_k = k
        
                                    
        new_k = re.sub(r'^blocks\.(\d+)\.norm1\.(weight|bias)$', 
                       r'layers.\1.layernorm_before.\2', new_k)
        new_k = re.sub(r'^blocks\.(\d+)\.attn\.qkv\.(weight|bias)$', 
                       r'layers.\1.attention.qkv.\2', new_k)
        new_k = re.sub(r'^blocks\.(\d+)\.attn\.proj\.(weight|bias)$', 
                       r'layers.\1.attention.proj_out.\2', new_k)
        new_k = re.sub(r'^blocks\.(\d+)\.norm2\.(weight|bias)$', 
                       r'layers.\1.layernorm_after.\2', new_k)
        new_k = re.sub(r'^blocks\.(\d+)\.mlp\.fc1\.(weight|bias)$', 
                       r'layers.\1.intermediate.\2', new_k)
        new_k = re.sub(r'^blocks\.(\d+)\.mlp\.fc2\.(weight|bias)$', 
                       r'layers.\1.output.\2', new_k)
        
                          
        new_k = re.sub(r'^norm\.(weight|bias)$', r'layernorm.\1', new_k)
        
        renamed[new_k] = v
    
                             
    missing, unexpected = model.load_state_dict(renamed, strict=strict)
    
    print(missing, unexpected)
    
    return model


def ca_vit_small(
    in_channels: int = 6,
    image_size: int = 224,
    use_flash_attention: bool = True,
    use_instance_norm: bool = True,
    pixel_scale: float = 255.0,
    use_cls_token: bool = False,
    use_sincos_pos_embed: bool = True,
) -> CAViT:
    """Create a small CA-ViT model"""
    config = CAViTConfig(
        in_channels=in_channels,
        image_size=image_size,
        patch_size=16,
        hidden_size=384,
        num_hidden_layers=12,
        num_attention_heads=6,
        intermediate_size=1536,
        use_flash_attention=use_flash_attention,
        use_instance_norm=use_instance_norm,
        pixel_scale=pixel_scale,
        use_cls_token=use_cls_token,
        use_sincos_pos_embed=use_sincos_pos_embed,
    )
    return CAViT(config)


def ca_vit_base(
    in_channels: int = 6,
    image_size: int = 224,
    use_flash_attention: bool = True,
    use_instance_norm: bool = True,
    pixel_scale: float = 255.0,
    use_cls_token: bool = False,
    use_sincos_pos_embed: bool = True,
) -> CAViT:
    """Create a base CA-ViT model"""
    config = CAViTConfig(
        in_channels=in_channels,
        image_size=image_size,
        patch_size=16,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        use_flash_attention=use_flash_attention,
        use_instance_norm=use_instance_norm,
        pixel_scale=pixel_scale,
        use_cls_token=use_cls_token,
        use_sincos_pos_embed=use_sincos_pos_embed,
    )
    return CAViT(config)


if __name__ == '__main__':
    model = ca_vit_small(use_flash_attention=False).to(torch.device('cuda'))
    load_timm_vit_into_cavit(model, 'vit_small_patch16_224', pretrained=True, strict=False)
    inputs = torch.randn(2, 6, 224, 224) * 255
                                       
    outputs = model(inputs.to(torch.device('cuda')))
    print(f"Output shape: {outputs.last_hidden_state.shape}")
                                             

                                     
                                                
                                          
                                                                           