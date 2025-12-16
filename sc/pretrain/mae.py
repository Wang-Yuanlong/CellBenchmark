import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from typing import Optional, Dict, Any, Tuple
import math

import os
import logging
from dataclasses import dataclass, field

from torch.utils.data import DataLoader
from transformers import (
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)

from data.datasets import RxRx3Dataset
from data.transformations import MAETrainTransform, SSLDataCollator
from utils.helper_func import load_timm_vit_into_mcvit
from pretrain.base import setup_logging


class MAEConfig(PretrainedConfig):
    """Configuration for MAE model."""
    
    model_type = "mae"
    
    def __init__(
        self,
        mask_ratio: float = 0.75,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        norm_pix_loss: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mask_ratio = mask_ratio
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads
        self.norm_pix_loss = norm_pix_loss


class MAEDecoder(nn.Module):
    """
    Lightweight Transformer decoder for MAE.
    
    Reconstructs masked patches from encoder output.
    """
    
    def __init__(
        self,
        embed_dim: int,
        decoder_embed_dim: int,
        decoder_depth: int,
        decoder_num_heads: int,
        mlp_ratio: float = 4.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_num_heads = decoder_num_heads
        
                                                 
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        
                    
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
                        
        self.decoder_blocks = nn.ModuleList([
            MAEDecoderBlock(
                decoder_embed_dim,
                decoder_num_heads,
                int(decoder_embed_dim * mlp_ratio),
            )
            for _ in range(decoder_depth)
        ])
        
        self.decoder_norm = norm_layer(decoder_embed_dim)
        
                            
        nn.init.normal_(self.mask_token, std=0.02)
    
    def forward(
        self,
        x: torch.Tensor,
        ids_restore: torch.Tensor,
        decoder_pos_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Encoded visible patches [B, L_visible, D_encoder]
            ids_restore: Indices to restore original order [B, L_total]
            decoder_pos_embed: Position embeddings [1, L_total, D_decoder]
            
        Returns:
            Reconstructed patches [B, L_total, D_decoder]
        """
                                      
        x = self.decoder_embed(x)                             
        
        B, L_visible, D = x.shape
        L_total = ids_restore.shape[1]
        
                            
        mask_tokens = self.mask_token.repeat(B, L_total - L_visible, 1)
        x = torch.cat([x, mask_tokens], dim=1)                           
        
                                           
        x = torch.gather(
            x, 
            dim=1, 
            index=ids_restore.unsqueeze(-1).expand(-1, -1, D)
        )                           
        
                                 
        x = x + decoder_pos_embed
        
                              
        for block in self.decoder_blocks:
            x = block(x)
        
        x = self.decoder_norm(x)
        
        return x


class MAEDecoderBlock(nn.Module):
    """Single Transformer block for MAE decoder."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
                                      
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
                           
        x = x + self.mlp(self.norm2(x))
        return x


class MAE(PreTrainedModel):
    """
    Masked Autoencoder (MAE) for self-supervised pretraining.
    
    Randomly masks patches and reconstructs them using a lightweight decoder.
    """
    
    config_class = MAEConfig
    
    def __init__(
        self,
        config: MAEConfig,
        encoder: PreTrainedModel,
    ):
        super().__init__(config)
        
        self.encoder = encoder
        self.mask_ratio = config.mask_ratio
        self.norm_pix_loss = config.norm_pix_loss
        
                                
        encoder_dim = getattr(
            encoder.config,
            'hidden_size',
            getattr(encoder.config, 'embed_dim', 768)
        )
        
                                          
        if hasattr(encoder, 'num_patches'):
            self.num_patches = encoder.num_patches
        else:
                                   
            img_size = encoder.config.image_size
            patch_size = encoder.config.patch_size
            self.num_patches = (img_size // patch_size) ** 2
        
                 
        self.decoder = MAEDecoder(
            embed_dim=encoder_dim,
            decoder_embed_dim=config.decoder_embed_dim,
            decoder_depth=config.decoder_depth,
            decoder_num_heads=config.decoder_num_heads,
        )
        
                                     
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, config.decoder_embed_dim)
        )
        
                                                      
        patch_size = encoder.config.patch_size
        in_channels = encoder.config.in_channels
        self.patch_dim = patch_size * patch_size * in_channels
        
        self.decoder_pred = nn.Linear(
            config.decoder_embed_dim, 
            self.patch_dim, 
            bias=True
        )
        
                            
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize decoder weights."""
        nn.init.normal_(self.decoder_pos_embed, std=0.02)
        
                            
        self.apply(self._init_decoder_weights)
    
    def _init_decoder_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Convert images to patches.
        
        Args:
            imgs: [B, C, H, W]
            
        Returns:
            patches: [B, L, patch_dim] where L = (H/P) * (W/P)
        """
        p = self.encoder.config.patch_size
        c = imgs.shape[1]
        h = w = imgs.shape[2] // p
        
                                                                  
        x = imgs.reshape(imgs.shape[0], c, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1)
        
                                               
        patches = x.reshape(imgs.shape[0], h * w, p * p * c)
        
        return patches
    
    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert patches back to images.
        
        Args:
            x: [B, L, patch_dim]
            
        Returns:
            imgs: [B, C, H, W]
        """
        p = self.encoder.config.patch_size
        c = self.encoder.config.in_channels
        h = w = int(x.shape[1] ** 0.5)
        
                                               
        x = x.reshape(x.shape[0], h, w, p, p, c)
        
                                                                  
        x = x.permute(0, 5, 1, 3, 2, 4)
        imgs = x.reshape(x.shape[0], c, h * p, w * p)
        
        return imgs
    
    def random_masking(
        self,
        x: torch.Tensor,
        mask_ratio: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Random masking following MAE paper.
        
        Args:
            x: Input sequence [B, L, D]
            mask_ratio: Fraction of patches to mask
            
        Returns:
            x_masked: Visible patches [B, L_visible, D]
            mask: Binary mask [B, L], 0 is keep, 1 is remove
            ids_restore: Indices to restore original order [B, L]
        """
        B, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        
                                    
        noise = torch.rand(B, L, device=x.device)
        
                                           
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
                                     
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )
        
                                                      
        mask = torch.ones([B, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(
        self,
        pixel_values: torch.Tensor,
        mask_ratio: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode visible patches.
        
        Args:
            pixel_values: [B, C, H, W]
            mask_ratio: Masking ratio
            
        Returns:
            latent: Encoded visible patches
            mask: Binary mask
            ids_restore: Restore indices
        """
                       
        x = self.encoder.patch_embedding(pixel_values)             
        
                                                             
        if hasattr(self.encoder, 'position_embeddings'):
            pos_embed = self.encoder.position_embeddings
            if self.encoder.config.use_cls_token:
                                               
                pos_embed = pos_embed[:, 1:, :]
            x = x + pos_embed
        
                        
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
                                   
        for layer in self.encoder.layers:
            x = layer(x)
        
        x = self.encoder.layernorm(x)
        
        return x, mask, ids_restore
    
    def forward_decoder(
        self,
        latent: torch.Tensor,
        ids_restore: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode latent representations to reconstruct patches.
        
        Args:
            latent: Encoded visible patches [B, L_visible, D_encoder]
            ids_restore: Indices to restore order [B, L_total]
            
        Returns:
            Reconstructed patches [B, L_total, patch_dim]
        """
                       
        x = self.decoder(latent, ids_restore, self.decoder_pos_embed)
        
                              
        x = self.decoder_pred(x)                     
        
        return x
    
    def forward_loss(
        self,
        imgs: torch.Tensor,
        pred: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute reconstruction loss on masked patches.
        
        Args:
            imgs: Original images [B, C, H, W]
            pred: Predicted patches [B, L, patch_dim]
            mask: Binary mask [B, L], 0 is keep, 1 is remove
            
        Returns:
            Scalar loss
        """
        if self.encoder.config.use_instance_norm:
            imgs = self.encoder.patch_embedding.instance_norm(imgs)
            
        target = self.patchify(imgs)                     
        
        if self.norm_pix_loss:
                                        
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** 0.5
        
                  
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)                               
        
                                             
        loss = (loss * mask).sum() / mask.sum()
        
        return loss
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        mask_ratio: Optional[float] = None,
        return_reconstructions: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Forward pass for MAE.
        
        Args:
            pixel_values: Input images [B, C, H, W]
            mask_ratio: Masking ratio (uses config default if None)
            return_reconstructions: Whether to return reconstructed images
            
        Returns:
            Dictionary with 'loss' and optionally reconstructions
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        
                
                                                          
        latent, mask, ids_restore = self.forward_encoder(
            pixel_values, mask_ratio
        )
        
                
        pred = self.forward_decoder(latent, ids_restore)
        
                      
                                    
        loss = self.forward_loss(pixel_values, pred, mask)
        
        
        output = {'loss': loss}
        
        if return_reconstructions:
                                                  
            pred_imgs = self.unpatchify(pred)
            output['reconstructions'] = pred_imgs
            output['mask'] = mask
        
        return output
    
    def get_encoder(self) -> PreTrainedModel:
        """Return the encoder for downstream tasks."""
        return self.encoder


@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    
    encoder_type: str = field(
        default="mcvit",
        metadata={"help": "Encoder type: mcvit or cavit"},
    )
    in_channels: int = field(
        default=6,
        metadata={"help": "Number of input channels"},
    )
    hidden_size: int = field(
        default=384,
        metadata={"help": "Hidden size (384 for ViT-S, 768 for ViT-B)"},
    )
    num_hidden_layers: int = field(
        default=12,
        metadata={"help": "Number of transformer layers"},
    )
    num_attention_heads: int = field(
        default=6,
        metadata={"help": "Number of attention heads (6 for ViT-S, 12 for ViT-B)"},
    )
    image_size: int = field(
        default=224,
        metadata={"help": "Input image size"},
    )
    patch_size: int = field(
        default=16,
        metadata={"help": "Patch size"},
    )
    use_flash_attention: bool = field(
        default=True,
        metadata={"help": "Use Flash Attention if available"},
    )
    use_instance_norm: bool = field(
        default=True,
        metadata={"help": "Whether to use InstanceNorm for images"},
    )
    
                  
    mask_ratio: float = field(
        default=0.75,
        metadata={"help": "Ratio of patches to mask (0.75 = 75%)"},
    )
    decoder_embed_dim: int = field(
        default=512,
        metadata={"help": "Decoder embedding dimension"},
    )
    decoder_depth: int = field(
        default=8,
        metadata={"help": "Number of decoder layers"},
    )
    decoder_num_heads: int = field(
        default=16,
        metadata={"help": "Number of decoder attention heads"},
    )
    norm_pix_loss: bool = field(
        default=True,
        metadata={"help": "Use normalized pixel loss"},
    )
    
    pretrained_vit: bool = field(
        default=False,
        metadata={"help": "Whether to load pretrained ViT weights"},
    )


@dataclass
class DataArguments:
    """Arguments for data configuration."""
    
    dataset: str = field(
        default="rxrx3",
        metadata={"help": "Dataset name: rxrx3, cpg12, cpp"},
    )
    csv_path: str = field(
        default=None,
        metadata={"help": "Path to metadata CSV file"},
    )
    img_folder: str = field(
        default=None,
        metadata={"help": "Path to image folder"},
    )
    train_split: str = field(
        default="train",
        metadata={"help": "Training split name"},
    )


class MAETrainer(Trainer):
    """Custom Trainer for MAE."""
    
    def compute_loss(
        self,
        model: MAE,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs,
    ):
        """
        Compute MAE reconstruction loss.
        
        The inputs dict should contain 'pixel_values'.
        """
        pixel_values = inputs.get('pixel_values')
        
        if pixel_values is None:
            raise ValueError(
                "MAETrainer expects 'pixel_values' in inputs. "
                f"Got keys: {list(inputs.keys())}"
            )
        
        outputs = model(pixel_values=pixel_values)
        loss = outputs['loss']
        
        return (loss, outputs) if return_outputs else loss


def create_model(model_args: ModelArguments) -> MAE:
    """Create MAE model with MCViT/CAViT encoder."""
    
    if model_args.encoder_type == "mcvit":
        from models.hfmodel import MCViT
        from models.model_config import MCViTConfig
        
        encoder_config = MCViTConfig(
            in_channels=model_args.in_channels,
            hidden_size=model_args.hidden_size,
            num_hidden_layers=model_args.num_hidden_layers,
            num_attention_heads=model_args.num_attention_heads,
            intermediate_size=model_args.hidden_size * 4,
            image_size=model_args.image_size,
            patch_size=model_args.patch_size,
            use_instance_norm=model_args.use_instance_norm,
            use_flash_attention=model_args.use_flash_attention,
            use_cls_token=False,                             
        )
        
        encoder = MCViT(encoder_config)
        

    
                                          
    if model_args.pretrained_vit and model_args.encoder_type == "mcvit":
        encoder = load_timm_vit_into_mcvit(encoder)
    
                       
    mae_config = MAEConfig(
        mask_ratio=model_args.mask_ratio,
        decoder_embed_dim=model_args.decoder_embed_dim,
        decoder_depth=model_args.decoder_depth,
        decoder_num_heads=model_args.decoder_num_heads,
        norm_pix_loss=model_args.norm_pix_loss,
    )
    
                      
    model = MAE(mae_config, encoder)
    
    return model


def train_mae(config_path: str = None):
    """Main training function."""
    
                     
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_yaml_file(config_path)
    
    base_batch_size = 256                            
    total_batch_size = (
        training_args.per_device_train_batch_size 
        * training_args.world_size 
        * training_args.gradient_accumulation_steps
    )
    scaled_lr = float(training_args.learning_rate) * (total_batch_size / base_batch_size)
    training_args.learning_rate = scaled_lr
    
                    
    logger = setup_logging(training_args.output_dir)
    logger.info(f"Model args: {model_args}")
    logger.info(f"Data args: {data_args}")
    logger.info(f"Training args: {training_args}")
    logger.info(f"Scaled learning rate: {scaled_lr}")
    
                  
    model = create_model(model_args)
    
                    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Encoder parameters: {encoder_params:,}")
    logger.info(f"Decoder parameters: {decoder_params:,}")
    
                    
    train_dataset = RxRx3Dataset(
        csv_path=data_args.csv_path,
        img_folder=data_args.img_folder,
        transform=MAETrainTransform(image_size=model_args.image_size),
                                      
    )
    
                          
    data_collator = SSLDataCollator('mae')
    
                    
    trainer = MAETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
           
    logger.info("*** Starting training ***")
    train_result = trainer.train()
    
                
    trainer.save_model()
    
                 
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
                                                  
    if trainer.is_world_process_zero():
        encoder_path = os.path.join(training_args.output_dir, "encoder")
        model.get_encoder().save_pretrained(encoder_path)
        logger.info(f"Encoder saved to {encoder_path}")
        logger.info("*** Training complete ***")

