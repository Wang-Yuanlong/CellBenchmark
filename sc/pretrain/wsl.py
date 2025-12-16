import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from typing import Optional, Dict, Any

import os
import logging
from dataclasses import dataclass, field

from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)
from transformers.trainer_utils import EvalPrediction
import numpy as np
from data.datasets import WSLDataset
from data.transformations import WSLTrainTransform, SSLDataCollator

from utils.helper_func import load_timm_vit_into_mcvit
from pretrain.base import setup_logging


class WSLConfig(PretrainedConfig):
    """Configuration for classification model."""
    model_type = "classification"
    
    def __init__(
        self,
        num_classes: int = 1000,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.dropout = dropout


class WSLHead(nn.Module):
    """Classification head."""
    
    def __init__(
        self,
        input_dim: int = 768,
        num_classes: int = 1000,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(input_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        return self.classifier(x)


class WSLModel(PreTrainedModel):
    """
    Classification model with encoder + head.
    """
    config_class = WSLConfig
    def __init__(
        self,
        config: WSLConfig,
        encoder: PreTrainedModel,
    ):
        super().__init__(config)
        
        self.encoder = encoder
        
        encoder_dim = getattr(
            encoder.config,
            'hidden_size',
            getattr(encoder.config, 'embed_dim', 768)
        )
        self.head = WSLHead(
            input_dim=encoder_dim,
            num_classes=config.num_classes,
            dropout=config.dropout,
        )
        
        self.num_classes = config.num_classes
    
    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images to feature vectors."""
        outputs = self.encoder(pixel_values, return_dict=True)
        
        if hasattr(outputs, 'pooler_output'):
            return outputs.pooler_output
        elif isinstance(outputs, torch.Tensor):
            return outputs
        else:
            raise ValueError(f"Unsupported encoder output type: {type(outputs)}")
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Forward pass.
        
        Args:
            pixel_values: [B, C, H, W]
            labels: [B] (optional)
            
        Returns:
            Dictionary with 'loss' (if labels provided) and 'logits'
        """
        features = self.encode(pixel_values)          
        logits = self.head(features)                    
        
        output = {'logits': logits}
        
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            output['loss'] = loss
        
        return output
    
    def freeze_encoder(self):
        """Freeze encoder parameters."""
        for p in self.encoder.parameters():
            p.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for p in self.encoder.parameters():
            p.requires_grad = True
    
    def get_encoder(self) -> PreTrainedModel:
        """Return the encoder."""
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
    
                             
    num_classes: int = field(
        default=1000,
        metadata={"help": "Number of classes"},
    )
    dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout rate in classification head"},
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


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """Compute classification metrics."""
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    preds = np.argmax(logits, axis=-1)
    
    accuracy = (preds == labels).mean()
    
    return {
        'accuracy': accuracy,
    }


def create_model(model_args: ModelArguments):
    """Create classification model."""
    
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
    

    if model_args.pretrained_vit:
        encoder = load_timm_vit_into_mcvit(encoder)
    
                                  
    clf_config = WSLConfig(
        num_classes=model_args.num_classes,
        dropout=model_args.dropout,
    )
    
                                 
    model = WSLModel(clf_config, encoder)
    

    
    return model


def train_wsl(config_path: str = None):
    """Main training function."""
    
                     
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    
    if config_path is not None:
        model_args, data_args, training_args = parser.parse_yaml_file(config_path)
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
                     
    train_dataset = WSLDataset(
        csv_path=data_args.csv_path,
        img_folder=data_args.img_folder,
        transform = WSLTrainTransform(
            image_size=model_args.image_size,
        )
    )
    model_args.num_classes = train_dataset.num_classes
    
    
                    
    logger = setup_logging(training_args.output_dir)
    
    logger.info(f"Model args: {model_args}")
    logger.info(f"Data args: {data_args}")
    logger.info(f"Training args: {training_args}")
    
                  
    model = create_model(model_args)
    
    
    
                    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    

    
    data_collator = SSLDataCollator('wsl')
    
                    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
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

