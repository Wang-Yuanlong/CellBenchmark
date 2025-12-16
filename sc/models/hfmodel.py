import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
from typing import Optional, Union, Tuple

from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling

from models.model_config import MCViTConfig, CAViTConfig

class ScaledDotProductAttention(nn.Module):
    """
    Unified attention module with optional Flash Attention 2 support.
    
    Uses flash_attn_qkvpacked_func for optimal performance when available,
    with automatic fallback to standard attention.
    
    Args:
        use_flash_attention: Whether to attempt using Flash Attention.
            Falls back gracefully if not available.
    """
    
    def __init__(self, use_flash_attention: bool = True):
        super().__init__()
        self.use_flash_attention = use_flash_attention
        self._flash_available: Optional[bool] = None
    
    @property
    def flash_available(self) -> bool:
        """Lazily check if Flash Attention 2 is available."""
        if self._flash_available is None:
            try:
                from flash_attn import flash_attn_qkvpacked_func
                self._flash_available = True
            except ImportError:
                self._flash_available = False
                if self.use_flash_attention:
                    warnings.warn(
                        "Flash Attention requested but flash_attn not installed. "
                        "Falling back to standard attention. "
                        "Install with: pip install flash-attn --no-build-isolation",
                        UserWarning,
                    )
        return self._flash_available
    
    @property
    def using_flash(self) -> bool:
        """Whether Flash Attention will actually be used."""
        return self.use_flash_attention and self.flash_available
    
    def forward(
        self,
        qkv: torch.Tensor,
        dropout_p: float = 0.0,
        softmax_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute attention with packed QKV.
        
        Args:
            qkv: Packed query, key, value tensor of shape [B, S, 3, H, D]
                where B=batch, S=sequence, H=heads, D=head_dim
            dropout_p: Dropout probability (only applied during training)
            softmax_scale: Optional scale factor (default: 1/sqrt(head_dim))
            
        Returns:
            Output tensor of shape [B, S, H, D]
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
            causal=False,
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
    """
    Multi-head self-attention layer for Vision Transformer.
    
    Uses packed QKV projection for efficiency and supports both
    Flash Attention and standard attention backends.
    """
    
    def __init__(self, config: Union[MCViTConfig, CAViTConfig]):
        super().__init__()
        
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
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
        
                                               
        if config.use_flash_attention:
            if self.head_dim % 8 != 0 or self.head_dim > 256:
                warnings.warn(
                    f"Flash Attention works best with head_dim % 8 == 0 and "
                    f"head_dim <= 256, got head_dim={self.head_dim}. "
                    f"May fall back to standard attention.",
                    UserWarning,
                )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: Input tensor of shape [B, S, H]
            
        Returns:
            Output tensor of shape [B, S, H]
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
    
    def __init__(self, config: Union[MCViTConfig, CAViTConfig]):
        super().__init__()
        
                         
        self.attention = ViTAttention(config)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        
                                             
        self.layernorm_before = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.layernorm_after = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        
                   
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act = nn.GELU()
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Input tensor of shape [B, S, H]
            attention_mask: Optional attention mask (not currently used)
            
        Returns:
            Output tensor of shape [B, S, H]
        """
                                       
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
    """
    Patch embedding for Multi-Channel ViT.
    
    Converts multi-channel images into patch tokens using a single
    convolutional layer that handles all channels together.
    
    Args:
        config: MCViTConfig with model parameters
    """
    
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
                track_running_stats=False,
            )
        
                          
        self.projection = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: Input images of shape [B, C, H, W]
                where C must equal self.in_channels
                
        Returns:
            Patch embeddings of shape [B, num_patches, hidden_size]
        """
                                
        if self.pixel_scale != 1.0:
            pixel_values = pixel_values / self.pixel_scale
        
                                      
        if self.use_instance_norm:
            pixel_values = self.instance_norm(pixel_values)
        
                                                           
        embeddings = self.projection(pixel_values)
        
                                                                             
        batch_size = embeddings.shape[0]
        embeddings = embeddings.flatten(2).transpose(1, 2)
        
        return embeddings


class CAPatchEmbedding(nn.Module):
    """
    Channel-Agnostic Patch Embedding.
    
    Key difference from MCPatchEmbedding: Uses single-channel Conv2d projection
    shared across all channels. This allows the model to handle variable
    numbers of channels at inference time.
    
    Each channel is processed independently with shared weights, then all
    channel tokens are concatenated into the sequence.
    
    Args:
        config: CAViTConfig with model parameters
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
                track_running_stats=False,
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
            pixel_values: Input images of shape [B, C, H, W]
                where C can be any number of channels
        
        Returns:
            embeddings: Patch embeddings of shape [B, C*N, D] where:
                - C = num_channels (variable, from input)
                - N = num_patches per channel = (H/P) * (W/P)
                - D = hidden_size
            num_channels: Number of input channels (C)
            
        Example:
            Input:  [4, 6, 224, 224]
            Output: [4, 1176, 768], 6
                    where 1176 = 6 * 196 (6 channels * 14 * 14 patches)
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
                                                           

class MCViT(PreTrainedModel):
    """
    Multi-Channel Vision Transformer.
    
    A ViT variant that handles multi-channel inputs (e.g., satellite imagery
    with 6+ channels) through a modified patch embedding layer.
    
    This model is compatible with HuggingFace Transformers:
```python
    model = MCViT.from_pretrained("your-username/mcvit-base", trust_remote_code=True)
```
    
    Args:
        config: MCViTConfig with model parameters
    """
    
    config_class = MCViTConfig
    base_model_prefix = "mcvit"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    
    def __init__(self, config: MCViTConfig):
        super().__init__(config)
        
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_patches = (config.image_size // config.patch_size) ** 2
        
                         
        self.patch_embedding = MCPatchEmbedding(config)
        
                              
        if config.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
            num_positions = self.num_patches + 1
        else:
            num_positions = self.num_patches
        
                             
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, num_positions, config.hidden_size)
        )
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
                             
        self.layers = nn.ModuleList([
            ViTLayer(config) for _ in range(config.num_hidden_layers)
        ])
        
                          
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
                            
        self.post_init()
    
    def _init_weights(self, module: nn.Module):
        """Initialize weights following ViT paper."""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def _interpolate_pos_embed(self, num_patches_new: int) -> torch.Tensor:
        """
        Interpolate position embeddings for different image sizes.
        
        Args:
            num_patches_new: New number of patches
            
        Returns:
            Interpolated position embeddings
        """
        pe = self.position_embeddings
        D = pe.shape[-1]
        
        use_cls = self.config.use_cls_token
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
        
        if gs0 * gs0 != N0 or gs1 * gs1 != num_patches_new:
            raise ValueError("num_patches should be a perfect square")
        
        patch_pos = patch_pos.reshape(1, gs0, gs0, D).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos, size=(gs1, gs1), mode="bicubic", align_corners=False
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, gs1 * gs1, D)
        
        if use_cls:
            return torch.cat([cls_pos, patch_pos], dim=1)
        return patch_pos
    
    def _add_pos_and_cls(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        """Add position embeddings and optional CLS token."""
        B, N, D = patch_embeddings.shape
        pos = self._interpolate_pos_embed(N).to(patch_embeddings.dtype)
        
        if self.config.use_cls_token:
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
        Forward pass.
        
        Args:
            pixel_values: Input images of shape [B, C, H, W]
                where C must equal config.in_channels
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return a ModelOutput or tuple
            
        Returns:
            BaseModelOutputWithPooling with:
                - last_hidden_state: [B, num_tokens, hidden_size]
                - pooler_output: [B, hidden_size]
                - hidden_states: Optional tuple of all hidden states
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
                            
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


                                                                               
             
                                                                               

class CAViT(PreTrainedModel):
    """
    Channel-Agnostic Vision Transformer.
    
    This model can handle variable numbers of input channels by treating each
    channel as a separate modality with shared spatial position embeddings.
    
    Key concepts:
    - num_patches (N): Number of spatial patches PER CHANNEL
    - num_tokens: Total tokens = C × N (varies with input channels C)
    - Position embeddings: Only N embeddings (shared across all channels)
    
    This model is compatible with HuggingFace Transformers:
```python
    model = CAViT.from_pretrained("your-username/cavit-base", trust_remote_code=True)
    
    # Works with different channel counts!
    out1 = model(torch.randn(1, 6, 224, 224))  # 6 channels
    out2 = model(torch.randn(1, 4, 224, 224))  # 4 channels
```
    
    Args:
        config: CAViTConfig with model parameters
    """
    
    config_class = CAViTConfig
    base_model_prefix = "cavit"
    main_input_name = "pixel_values"
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
                int(self.num_patches ** 0.5),
            )
            self.register_buffer('position_embeddings', pos_embed)
            self._pos_cache: dict = {}
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
    
    def _get_sincos_pos_embed(
        self,
        embed_dim: int,
        grid_size: int,
    ) -> torch.Tensor:
        """
        Generate 2D sinusoidal position embeddings.
        
        Args:
            embed_dim: Embedding dimension
            grid_size: Grid size (e.g., 14 for 14×14 patches)
            
        Returns:
            Position embeddings of shape [1, grid_size*grid_size, embed_dim]
        """
        grid_h = torch.arange(grid_size, dtype=torch.float32)
        grid_w = torch.arange(grid_size, dtype=torch.float32)
        grid = torch.meshgrid(grid_w, grid_h, indexing='xy')
        grid = torch.stack(grid, dim=0).reshape(2, -1)
        
        pos_embed = self._get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
        return pos_embed.unsqueeze(0)
    
    @staticmethod
    def _get_2d_sincos_pos_embed_from_grid(
        embed_dim: int,
        grid: torch.Tensor,
    ) -> torch.Tensor:
        """Generate 2D sinusoidal embeddings from grid coordinates."""
        assert embed_dim % 2 == 0
        
        emb_h = CAViT._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
        emb_w = CAViT._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
        
        return torch.cat([emb_h, emb_w], dim=1)
    
    @staticmethod
    def _get_1d_sincos_pos_embed_from_grid(
        embed_dim: int,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        """Generate 1D sinusoidal position embeddings."""
        assert embed_dim % 2 == 0
        
        omega = torch.arange(embed_dim // 2, dtype=torch.float32)
        omega /= embed_dim / 2.0
        omega = 1.0 / (10000 ** omega)
        
        pos = pos.reshape(-1)
        out = torch.einsum('m,d->md', pos, omega)
        
        emb_sin = torch.sin(out)
        emb_cos = torch.cos(out)
        
        return torch.cat([emb_sin, emb_cos], dim=1)
    
    def _init_weights(self, module: nn.Module):
        """Initialize weights following ViT paper."""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def _interpolate_pos_embed(self, num_patches_new: int) -> torch.Tensor:
        """Interpolate position embeddings for different image sizes."""
        pe = self.position_embeddings
        N, D = pe.shape[1], pe.shape[2]
        
        if N == num_patches_new:
            return pe
        
        gs_old = int(math.sqrt(N))
        gs_new = int(math.sqrt(num_patches_new))
        
        if gs_old * gs_old != N or gs_new * gs_new != num_patches_new:
            raise ValueError("num_patches should be a perfect square")
        
        pe = pe.reshape(1, gs_old, gs_old, D).permute(0, 3, 1, 2)
        pe = F.interpolate(pe, size=(gs_new, gs_new), mode="bicubic", align_corners=False)
        pe = pe.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, D)
        
        return pe
    
    def _add_channel_agnostic_pos_embed(
        self,
        patch_embeddings: torch.Tensor,
        num_channels: int,
    ) -> torch.Tensor:
        """
        Add position embeddings in a channel-agnostic manner.
        
        All channels share the same spatial position embeddings.
        
        Args:
            patch_embeddings: [B, C*N, D]
            num_channels: Number of input channels (C)
            
        Returns:
            Embeddings with position: [B, C*N, D] or [B, 1+C*N, D] with CLS
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
                    if gs * gs != N_in:
                        raise ValueError("Input patches must form a square grid")
                    pos_embed_cpu = self._get_sincos_pos_embed(self.hidden_size, gs).cpu()
                    self._pos_cache[N_in] = pos_embed_cpu
                pos_embed = pos_embed_cpu
            pos_embed = pos_embed.to(
                patch_embeddings.device, patch_embeddings.dtype
            )
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
            pixel_values: Input images of shape [B, C, H, W]
                where C can be any number of channels
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return a ModelOutput or tuple
            
        Returns:
            BaseModelOutputWithPooling with:
                - last_hidden_state: [B, C*N, D] where N=num_patches per channel
                - pooler_output: [B, D]
                - hidden_states: Optional tuple of all hidden states
                
        Note:
            Total tokens = C × N varies with input channel count C
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
                                             
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