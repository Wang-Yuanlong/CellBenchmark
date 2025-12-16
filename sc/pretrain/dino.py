import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from transformers import PreTrainedModel, PretrainedConfig
from typing import Optional, Dict, Any, List
import copy

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
    TrainerCallback,
)
import wandb
from data.datasets import RxRx3Dataset
from data.transformations import DINOTrainTransform, SSLDataCollator
from utils.helper_func import load_timm_vit_into_mcvit
from pretrain.base import setup_logging


class DINOConfig(PretrainedConfig):
    """Configuration for DINO model."""
    
    model_type = "dino"
    
    def __init__(
        self,
        out_dim: int = 65536,
        use_bn: bool = False,
        norm_last_layer: bool = True,
        hidden_dim: int = 1024,
        bottleneck_dim: int = 256,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        momentum_teacher: float = 0.996,
        center_momentum: float = 0.9,
        teacher_temp_final: float = 0.07,
        teacher_temp_warmup_epochs: int = 20,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.out_dim = out_dim
        self.use_bn = use_bn
        self.norm_last_layer = norm_last_layer
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.momentum_teacher = momentum_teacher
        self.center_momentum = center_momentum
        self.teacher_temp_final = teacher_temp_final
        self.teacher_temp_warmup_epochs = teacher_temp_warmup_epochs


class DINO(PreTrainedModel):
    """
    DINO: Self-Distillation with No Labels.
    
    Reference: https://arxiv.org/abs/2104.14294
    """
    
    config_class = DINOConfig
    
    def __init__(
        self,
        config: DINOConfig,
        encoder: PreTrainedModel,
    ):
        super().__init__(config)
        
                               
        encoder_dim = getattr(
            encoder.config,
            'hidden_size',
            getattr(encoder.config, 'embed_dim', 768)
        )
        
                         
        self.student = encoder
        
                      
        self.student_projector = self._build_projector(
            encoder_dim, config.hidden_dim, config.bottleneck_dim, 
            nlayers=3, use_bn=config.use_bn
        )
        self.student_last_layer = weight_norm(nn.Linear(config.bottleneck_dim, config.out_dim, bias=False))
        self.student_last_layer.weight_g.data.fill_(1.0)
        if config.norm_last_layer:
            self.student_last_layer.weight_g.requires_grad = False
        
                                    
        self.teacher = copy.deepcopy(self.student)
        for p in self.teacher.parameters():
            p.requires_grad = False
        
                                                           
        self.teacher_projector = copy.deepcopy(self.student_projector)
        for p in self.teacher_projector.parameters():
            p.requires_grad = False
        
        self.teacher_last_layer = weight_norm(nn.Linear(config.bottleneck_dim, config.out_dim, bias=False))
        self.teacher_last_layer.weight_g.data.fill_(1.0)
        for p in self.teacher_last_layer.parameters():
            p.requires_grad = False
        
                              
        self.student_temp = config.student_temp
        self.teacher_temp = config.teacher_temp
        self.teacher_temp_final = config.teacher_temp_final
        self.teacher_temp_warmup_epochs = config.teacher_temp_warmup_epochs
        self.momentum_teacher = config.momentum_teacher
        self.center_momentum = config.center_momentum
        
                                   
        self.register_buffer("center", torch.zeros(1, config.out_dim))
        
        self._last_student_features = None
    
    @staticmethod
    def _build_projector(in_dim, hidden_dim, bottleneck_dim, nlayers=3, use_bn=False):
        """Build MLP projector."""
        layers = []
        for i in range(nlayers - 1):
            dim1 = in_dim if i == 0 else hidden_dim
            dim2 = hidden_dim
            layers.append(nn.Linear(dim1, dim2, bias=not use_bn))
            if use_bn:
                layers.append(nn.BatchNorm1d(dim2))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim if nlayers > 1 else in_dim, bottleneck_dim, bias=False))
        return nn.Sequential(*layers)
    
    def encode(self, pixel_values: torch.Tensor, use_teacher: bool = False) -> torch.Tensor:
        """Encode images to feature vectors."""
        encoder = self.teacher if use_teacher else self.student
        outputs = encoder(pixel_values, return_dict=True)
        
        if hasattr(outputs, 'pooler_output'):
            return outputs.pooler_output
        elif isinstance(outputs, torch.Tensor):
            return outputs
        else:
            raise ValueError(f"Unsupported encoder output type: {type(outputs)}")
    
    def forward(
        self,
        student_crops: List[torch.Tensor],
        teacher_crops: List[torch.Tensor],
        return_features: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Forward pass for DINO.
        
        Args:
            student_crops: [global1, global2, local1, ..., localN]
            teacher_crops: [global1, global2]
        """
        if not student_crops or not teacher_crops:
            raise ValueError("Both student_crops and teacher_crops must be non-empty lists")
        
                                                     
        n_global = 2
        global_crops = student_crops[:n_global]
        local_crops = student_crops[n_global:]
        
                              
        global_batch = torch.cat(global_crops, dim=0)
        global_features = self.encode(global_batch, use_teacher=False)
        global_z = self.student_projector(global_features)
        global_logits = self.student_last_layer(global_z)
        global_logits_list = list(global_logits.chunk(n_global, dim=0))
        
                             
        if local_crops:
            local_batch = torch.cat(local_crops, dim=0)
            local_features = self.encode(local_batch, use_teacher=False)
            local_z = self.student_projector(local_features)
            local_logits = self.student_last_layer(local_z)
            local_logits_list = list(local_logits.chunk(len(local_crops), dim=0))
        else:
            local_logits_list = []
        
        student_logits_list = global_logits_list + local_logits_list
        
        if return_features:
            global_features_list = list(global_features.chunk(n_global, dim=0))
            local_features_list = list(local_features.chunk(len(local_crops), dim=0)) if local_crops else []
            self._last_student_features = global_features_list + local_features_list
        
                 
        with torch.no_grad():
            self._ema_update_teacher()
            
            teacher_batch = torch.cat(teacher_crops, dim=0)
            teacher_features = self.encode(teacher_batch, use_teacher=True)
            teacher_z = self.teacher_projector(teacher_features)
            teacher_logits = self.teacher_last_layer(teacher_z)
            teacher_logits_list = list(teacher_logits.chunk(len(teacher_crops), dim=0))
        
        output = {
            'student_logits': student_logits_list,
            'teacher_logits': teacher_logits_list,
        }
        
        if return_features:
            output['student_features'] = self._last_student_features
        
        return output
    
    def loss_fn(
        self,
        student_logits_list: List[torch.Tensor],
        teacher_logits_list: List[torch.Tensor],
        epoch: Optional[int] = None,
    ) -> torch.Tensor:
        """DINO loss."""
        n_teacher = len(teacher_logits_list)
        n_student = len(student_logits_list)
        
        Tt = float(self._get_teacher_temp(epoch))
        Ts = float(self.student_temp)
        
                       
        with torch.no_grad():
            t_probs = []
            center32 = self.center.float()
            
            for t in teacher_logits_list:
                t32 = t.float()
                x = t32 - center32
                x = x - x.max(dim=-1, keepdim=True)[0]
                q = F.softmax(x / Tt, dim=-1)
                t_probs.append(q)
        
                           
        s_logp = []
        for s in student_logits_list:
            s32 = s.float()
            y = s32 - s32.max(dim=-1, keepdim=True)[0]
            logp = F.log_softmax(y / Ts, dim=-1)
            s_logp.append(logp)
        
                       
        total_loss = 0.0
        n_terms = 0
        
        for iq in range(n_teacher):
            q = t_probs[iq].detach()
            
            for v in range(n_student):
                if v == iq and v < n_teacher:
                    continue
                
                ce = (-q * s_logp[v]).sum(dim=-1).mean()
                total_loss += ce
                n_terms += 1
        
        loss = total_loss / max(n_terms, 1)
        
                       
        with torch.no_grad():
            batch_center = torch.cat(
                [t.float() for t in teacher_logits_list],
                dim=0
            ).mean(dim=0, keepdim=True)
            
            self.center.mul_(self.center_momentum).add_(
                batch_center,
                alpha=1.0 - self.center_momentum
            )
        
        return loss
    
    @torch.no_grad()
    def _ema_update_teacher(self):
        """Update teacher parameters via EMA."""
        m = self.momentum_teacher
        
        for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
            pt.data.mul_(m).add_(ps.data, alpha=1.0 - m)
        
        for ps, pt in zip(self.student_projector.parameters(), self.teacher_projector.parameters()):
            pt.data.mul_(m).add_(ps.data, alpha=1.0 - m)
        
        for ps, pt in zip(self.student_last_layer.parameters(), self.teacher_last_layer.parameters()):
            pt.data.mul_(m).add_(ps.data, alpha=1.0 - m)
    
    def _get_teacher_temp(self, epoch: Optional[int] = None) -> float:
        """Get teacher temperature with linear warmup."""
        if epoch is None or self.teacher_temp_warmup_epochs <= 0:
            return float(self.teacher_temp_final)
        
        if epoch >= self.teacher_temp_warmup_epochs:
            return float(self.teacher_temp_final)
        
        t0 = float(self.teacher_temp)
        t1 = float(self.teacher_temp_final)
        r = float(epoch) / float(self.teacher_temp_warmup_epochs)
        
        return t0 + (t1 - t0) * r
    
    def get_encoder(self) -> PreTrainedModel:
        """Return the student encoder for downstream tasks."""
        return self.student


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
    
                   
    out_dim: int = field(
        default=65536,
        metadata={"help": "Output dimension of DINO head"},
    )
    hidden_dim: int = field(
        default=2048,
        metadata={"help": "Hidden dimension in projection head"},
    )
    bottleneck_dim: int = field(
        default=256,
        metadata={"help": "Bottleneck dimension in projection head"},
    )
    teacher_temp: float = field(
        default=0.04,
        metadata={"help": "Initial teacher temperature"},
    )
    teacher_temp_final: float = field(
        default=0.07,
        metadata={"help": "Final teacher temperature after warmup"},
    )
    teacher_temp_warmup_epochs: int = field(
        default=30,
        metadata={"help": "Number of epochs for teacher temperature warmup"},
    )
    student_temp: float = field(
        default=0.1,
        metadata={"help": "Student temperature"},
    )
    momentum_teacher: float = field(
        default=0.996,
        metadata={"help": "Base momentum for teacher EMA (will be scheduled)"},
    )
    momentum_teacher_final: float = field(
        default=1.0,
        metadata={"help": "Final momentum for teacher EMA"},
    )
    center_momentum: float = field(
        default=0.9,
        metadata={"help": "Momentum for teacher output centering"},
    )
    use_bn: bool = field(
        default=False,
        metadata={"help": "Use batch norm in projection head"},
    )
    norm_last_layer: bool = field(
        default=True,
        metadata={"help": "Normalize last layer of projection head"},
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
    n_local_crops: int = field(
        default=8,
        metadata={"help": "Number of local crops for DINO"},
    )
    local_crop_size: int = field(
        default=96,
        metadata={"help": "Size of local crops"},
    )


class DINOTrainer(Trainer):
    """
    Custom Trainer for DINO that handles multi-crop inputs.
    """
    
    def _current_epoch_float(self) -> Optional[float]:
        """Get current epoch as float (with fractional progress)."""
        if self.state is not None and self.state.epoch is not None:
            return float(self.state.epoch)
        
                                             
        if self.args.max_steps > 0 and self.state is not None:
            prog = self.state.global_step / max(self.args.max_steps, 1)
            return prog * float(self.args.num_train_epochs)
        
        return None
    
    def compute_loss(
        self,
        model: DINO,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        """
        Compute DINO loss.
        
        The inputs dict should contain 'student_crops' and 'teacher_crops'.
        """
        student_crops: List[torch.Tensor] = inputs.get('student_crops')
        teacher_crops: List[torch.Tensor] = inputs.get('teacher_crops')
        
        if student_crops is None or teacher_crops is None:
            raise ValueError(
                "DINOTrainer expects 'student_crops' and 'teacher_crops' in inputs. "
                f"Got keys: {list(inputs.keys())}"
            )
        
                      
        outputs = model(
            student_crops=student_crops,
            teacher_crops=teacher_crops,
        )
        
                                                  
        epoch_float = self._current_epoch_float()
        epoch_int = int(epoch_float) if epoch_float is not None else None
        
                      
        loss = model.loss_fn(
            outputs['student_logits'],
            outputs['teacher_logits'],
            epoch=epoch_int,
        )
        
        if not return_outputs:
            return loss
        
                          
        with torch.no_grad():
            extras = {
                "teacher_temp": model._get_teacher_temp(epoch_int),
                "center_norm": model.center.float().norm(p=2).item(),
                "n_student_views": len(outputs['student_logits']),
                "n_teacher_views": len(outputs['teacher_logits']),
            }
        
        outputs.update(extras)
        
        return loss, outputs


class TeacherMomentumSchedulerCallback(TrainerCallback):
    """
    Linearly increase momentum_teacher from base_m to final_m.
    Could also use cosine schedule as in the paper.
    """
    
    def __init__(self, base_m: float = 0.996, final_m: float = 1.0):
        self.base_m = base_m
        self.final_m = final_m
    
    def on_step_begin(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None or not hasattr(model, "momentum_teacher"):
            return
        
                            
        if state.max_steps and state.max_steps > 0:
            progress = state.global_step / max(state.max_steps, 1)
        elif state.epoch is not None:
            progress = float(state.epoch) / max(args.num_train_epochs, 1)
        else:
            progress = 0.0
        
                       
        m = self.base_m + (self.final_m - self.base_m) * min(max(progress, 0.0), 1.0)
        model.momentum_teacher = float(m)


def create_model(model_args: ModelArguments) -> DINO:
    """Create DINO model with MCViT encoder."""
    
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
    
                        
    dino_config = DINOConfig(
        out_dim=model_args.out_dim,
        use_bn=model_args.use_bn,
        norm_last_layer=model_args.norm_last_layer,
        hidden_dim=model_args.hidden_dim,
        bottleneck_dim=model_args.bottleneck_dim,
        teacher_temp=model_args.teacher_temp,
        student_temp=model_args.student_temp,
        momentum_teacher=model_args.momentum_teacher,
        center_momentum=model_args.center_momentum,
        teacher_temp_final=model_args.teacher_temp_final,
        teacher_temp_warmup_epochs=model_args.teacher_temp_warmup_epochs,
    )
    
                       
    model = DINO(dino_config, encoder)
    
    return model


def train_dino(config_path: str = None):
    """Main training function for DINO."""
    
                     
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    
    if config_path is not None:
        model_args, data_args, training_args = parser.parse_yaml_file(config_path)
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
                                                            
    base_batch_size = 256                        
    total_batch_size = (
        training_args.per_device_train_batch_size *
        training_args.world_size *
        training_args.gradient_accumulation_steps
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
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
                    
    train_dataset = RxRx3Dataset(
        csv_path=data_args.csv_path,
        img_folder=data_args.img_folder,
        transform=DINOTrainTransform(
            global_crop_size=model_args.image_size,
            local_crop_size=data_args.local_crop_size,
            n_local_crops=data_args.n_local_crops,
        ),
                                      
    )
    
                          
    data_collator = SSLDataCollator('dino')
    
                      
    callbacks = [
        TeacherMomentumSchedulerCallback(
            base_m=model_args.momentum_teacher,
            final_m=model_args.momentum_teacher_final,
        )
    ]
    
                    
    trainer = DINOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    
           
    logger.info("*** Starting DINO training ***")
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
        
        logger.info("*** DINO training complete ***")


                            
                
    
                                                             
                                 
           
                      