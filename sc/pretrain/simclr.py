import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from typing import Optional, Dict, Any

import os
import sys
import logging
import argparse
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union

from torch.utils.data import DataLoader

from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)
from transformers.trainer_utils import EvalPrediction
import wandb
from data.datasets import RxRx3Dataset
from data.transformations import SimCLRTrainTransform, SSLDataCollator
from utils.helper_func import load_timm_vit_into_mcvit
from pretrain.base import setup_logging



class SimCLRConfig(PretrainedConfig):
    """Configuration for SimCLR model."""
    
    model_type = "simclr"
    def __init__(
        self,
        temperature: float = 0.07,
        projection_dim: int = 256,
        hidden_dim: int = 2048,
        use_bn: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.use_bn = use_bn


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning."""
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 2048,
        output_dim: int = 256,
        use_bn: bool = True,
    ):
        super().__init__()
        
        layers = [nn.Linear(input_dim, hidden_dim)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.extend([
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        ])
        
        self.projection = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class SimCLR(PreTrainedModel):
    """
    SimCLR: A Simple Framework for Contrastive Learning of Visual Representations.
    
    Reference: https://arxiv.org/abs/2002.05709
    """
    
    config_class = SimCLRConfig
    
    def __init__(
        self,
        config: SimCLRConfig,
        encoder: PreTrainedModel,
    ):
        super().__init__(config)
        
        self.encoder = encoder
        self.temperature = config.temperature
        
        encoder_dim = getattr(
            encoder.config,
            'hidden_size',
            getattr(encoder.config, 'embed_dim', 768)
        )
        
        self.projection_head = ProjectionHead(
            input_dim=encoder_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.projection_dim,
            use_bn=config.use_bn,
        )
    
    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images to feature vectors."""
        outputs = self.encoder(pixel_values)
                            
        if hasattr(outputs, 'pooler_output'):
            return outputs.pooler_output
        elif isinstance(outputs, torch.Tensor):
            return outputs
        else:
            raise ValueError(f"Unsupported encoder output type: {type(outputs)}")
    
    def forward(
        self,
        view1: torch.Tensor,
        view2: Optional[torch.Tensor] = None,
        return_features: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Forward pass for SimCLR.
        
        Args:
            pixel_values_1: First view [B, C, H, W]
            pixel_values_2: Second view [B, C, H, W]
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary with 'loss' and optionally features
        """
        if view2 is None:
            raise ValueError("SimCLR requires two views. Got view2=None")
        
        batch_size = view1.shape[0]
        
                                                   
        all_images = torch.cat([view1, view2], dim=0)
        features = self.encode(all_images)           
        projections = self.projection_head(features)                        
        
                    
        z1, z2 = projections.chunk(2, dim=0)                            
        
                      
        loss = self.nt_xent_loss(z1, z2)
        
        output = {'loss': loss}
        
        if return_features:
            f1, f2 = features.chunk(2, dim=0)
            output.update({
                'features_1': f1,
                'features_2': f2,
                'projections_1': z1,
                'projections_2': z2,
            })
        
        return output
    
    def nt_xent_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).
        
        Args:
            z1: Projections from view 1, shape [B, D]
            z2: Projections from view 2, shape [B, D]
            
        Returns:
            Scalar loss
        """
        batch_size = z1.shape[0]
        device = z1.device
        
                      
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
                                          
        z = torch.cat([z1, z2], dim=0)
        
                                     
        sim_matrix = torch.mm(z, z.t()) / self.temperature
        
                                                                 
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),                  
            torch.arange(batch_size),                                   
        ], dim=0).to(device)
        
                                             
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))
        
                            
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss
    
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
        metadata={"help": "Whether to use InstanceNorm for images."}
    )
                     
    temperature: float = field(
        default=0.07,
        metadata={"help": "Temperature for NT-Xent loss"},
    )
    projection_dim: int = field(
        default=128,
        metadata={"help": "Projection head output dimension"},
    )
    projection_hidden_dim: int = field(
        default=2048,
        metadata={"help": "Projection head hidden dimension"},
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

    
    
class SimCLRTrainer(Trainer):
    """
    Custom Trainer for SimCLR that handles two-view inputs.
    """
    
    def compute_loss(
        self,
        model: SimCLR,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs,
    ):
        """
        Compute SimCLR loss.
        
        The inputs dict should contain 'view1' and 'view2'.
        """
        view1 = inputs.get('view1')
        view2 = inputs.get('view2')
        
        if view1 is None or view2 is None:
            raise ValueError(
                "SimCLRTrainer expects 'view1' and 'view2' in inputs. "
                f"Got keys: {list(inputs.keys())}"
            )
        
        outputs = model(
            view1=view1,
            view2=view2,
        )
        
        loss = outputs['loss']
        
        return (loss, outputs) if return_outputs else loss
    
def create_model(model_args: ModelArguments) -> SimCLR:
    """Create SimCLR model with MCViT encoder."""
    
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
        use_instance_norm = model_args.use_instance_norm,
        use_flash_attention=model_args.use_flash_attention,
        use_cls_token=False,
    )
    
    encoder = MCViT(encoder_config)
    if model_args.pretrained_vit:
        encoder = load_timm_vit_into_mcvit(encoder)
    
    simclr_config = SimCLRConfig(
        temperature=model_args.temperature,
        projection_dim=model_args.projection_dim,
        hidden_dim=model_args.projection_hidden_dim,
    )
    
    model = SimCLR(simclr_config, encoder)
    
    return model
    
def train_simclr(config_path: str = None):
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    
    model_args, data_args, training_args = parser.parse_yaml_file(config_path)
    base_batch_size = 256                                       
    total_batch_size = training_args.per_device_train_batch_size * training_args.world_size * training_args.gradient_accumulation_steps
    
    scaled_lr = float(training_args.learning_rate) * (total_batch_size / base_batch_size)
    training_args.learning_rate = scaled_lr
    
    logger = setup_logging(training_args.output_dir)

    logger.info(f"Model args: {model_args}")
    logger.info(f"Data args: {data_args}")
    logger.info(f"Training args: {training_args}")
    
    model = create_model(model_args)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    train_dataset = RxRx3Dataset(
        csv_path=data_args.csv_path,
        img_folder=data_args.img_folder,
        transform = SimCLRTrainTransform(image_size=model_args.image_size),
        split=data_args.train_split,
    )
                 
    data_collator = SSLDataCollator('simclr')
    
    trainer = SimCLRTrainer(
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
