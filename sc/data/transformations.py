
import cv2
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn.functional as F


class TransformBase:
    def __init__(self, image_size: int = 224):
        self.image_size = image_size
    
    def __call__(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
    

class ChannelIntensityJitter(A.ImageOnlyTransform):
    def __init__(
        self,
        scale_range=(0.8, 1.2),
        shift_range=(-0.1, 0.1),
        per_channel=True,
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply, p)
        self.scale_range = scale_range
        self.shift_range = shift_range
        self.per_channel = per_channel

    def apply(self, img, **params):
                        
        orig_dtype = img.dtype
        img = img.astype(np.float32)

        img_min = img.min()
        img_max = img.max()
        dynamic = max(img_max - img_min, 1e-6)

        h, w, c = img.shape

        if self.per_channel:
            scales = np.random.uniform(
                self.scale_range[0], self.scale_range[1], size=(1, 1, c)
            ).astype(np.float32)
            shifts = np.random.uniform(
                self.shift_range[0], self.shift_range[1], size=(1, 1, c)
            ).astype(np.float32)
        else:
            s = np.random.uniform(*self.scale_range)
            t = np.random.uniform(*self.shift_range)
            scales = np.full((1, 1, c), s, dtype=np.float32)
            shifts = np.full((1, 1, c), t, dtype=np.float32)

        shifts = shifts * dynamic

        img = img * scales + shifts
        img = np.clip(img, img_min, img_max)

        return img

    def get_transform_init_args_names(self):
        return ("scale_range", "shift_range", "per_channel")
    
    
class RandomGammaContrast(A.ImageOnlyTransform):
    def __init__(
        self,
        gamma_range=(0.7, 1.4),
        contrast_range=(0.7, 1.3),
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply, p)
        self.gamma_range = gamma_range
        self.contrast_range = contrast_range

    def apply(self, img, gamma=1.0, contrast=1.0, **params):
        orig_dtype = img.dtype
        img = img.astype(np.float32)

        img_min = img.min()
        img_max = img.max()
        dynamic = max(img_max - img_min, 1e-6)

        x = (img - img_min) / dynamic

        x = x ** gamma

        x = 0.5 + (x - 0.5) * contrast
        x = np.clip(x, 0.0, 1.0)

        img = x * dynamic + img_min

        return img

    def get_params(self):
        gamma = np.random.uniform(*self.gamma_range)
        contrast = np.random.uniform(*self.contrast_range)
        return {"gamma": gamma, "contrast": contrast}

    def get_transform_init_args_names(self):
        return ("gamma_range", "contrast_range")


class SimCLRTrainTransform(TransformBase):
    def __init__(self, image_size: int = 224, mean=None, std=None):
        super().__init__(image_size=image_size)

        self.aug = A.Compose([
            A.RandomResizedCrop(
                size=(self.image_size, self.image_size),
                scale=(0.08, 1.0),
                ratio=(0.75, 1.3333),
                interpolation=cv2.INTER_CUBIC,
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),

            RandomGammaContrast(
                gamma_range=(0.7, 1.4),
                contrast_range=(0.7, 1.3),
                p=0.5,
            ),
            
            A.GaussianBlur(
                blur_limit=(23, 23),
                sigma_limit=(0.1, 2.0),
                p=0.5,
            ),

            ToTensorV2(),
        ])

    def __call__(self, image):
        out1 = self.aug(image=image)["image"]
        out2 = self.aug(image=image)["image"]
        return {
            "view1": out1,  
            "view2": out2,
        }
        
class DINOTrainTransform(TransformBase):
    def __init__(
        self,
        global_crop_size: int = 224,
        local_crop_size: int = 96,
        n_local_crops: int = 8,
        global_crop_scale: tuple = (0.4, 1.0),
        local_crop_scale: tuple = (0.05, 0.4),
    ):
        super().__init__(global_crop_size)
        self.local_crop_size = local_crop_size
        self.n_local_crops = n_local_crops

        self.global_aug = A.Compose([
            A.RandomResizedCrop(
                size=(global_crop_size, global_crop_size),
                scale=global_crop_scale,        
                ratio=(0.75, 1.3333),
                interpolation=cv2.INTER_CUBIC,
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            
            ChannelIntensityJitter(
                scale_range=(0.9, 1.1),
                shift_range=(-0.05, 0.05),
                per_channel=True,
                p=0.8,
            ),
            
            A.GaussianBlur(
                blur_limit=(23, 23),
                sigma_limit=(0.1, 2.0),
                p=0.5,
            ),
            ToTensorV2(),   
        ])

        self.local_aug = A.Compose([
            A.RandomResizedCrop(
                size=(local_crop_size, local_crop_size),
                scale=local_crop_scale,
                ratio=(0.75, 1.3333),
                interpolation=cv2.INTER_CUBIC,
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.GaussianBlur(
                blur_limit=(3, 7),
                sigma_limit=(0.1, 2.0),
                p=0.2,
            ),
            ToTensorV2(),
        ])

    def _to_hwc_numpy(self, image: torch.Tensor | np.ndarray) -> np.ndarray:
        """
        把 [C, H, W] 的 torch.Tensor 或 [H, W, C] 的 numpy 统一成 HWC numpy,
        以便 Albumentations 使用。
        """
        if isinstance(image, torch.Tensor):
                             
            image_np = image.detach().cpu().numpy()
                                
            image_np = np.moveaxis(image_np, 0, 2)
            return image_np
        else:
                                    
            return image

    def __call__(self, image) -> Dict[str, List[torch.Tensor]]:
        """
        Args:
            image: torch.Tensor of [C, H, W]

        Returns:
            dict: {
                'student_crops': List of (2 global + n local) crops,
                'teacher_crops': List of 2 global crops,
            }
        """
        image_np = image                           

                          
        global_crops: List[torch.Tensor] = []
        for _ in range(2):
            out = self.global_aug(image=image_np)["image"] 
            global_crops.append(out)

                         
        local_crops: List[torch.Tensor] = []
        for _ in range(self.n_local_crops):
            out = self.local_aug(image=image_np)["image"]
            local_crops.append(out)

        return {
            "student_crops": global_crops + local_crops,
            "teacher_crops": global_crops,
        }
        
        
class WSLTrainTransform(TransformBase):
    def __init__(self, image_size: int = 224):
        super().__init__(image_size=image_size)

        self.aug = A.Compose([
            A.RandomResizedCrop(
                size=(self.image_size, self.image_size),
                scale=(0.08, 1.0),
                ratio=(0.75, 1.3333),
                interpolation=cv2.INTER_CUBIC,
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5), 
            
                                     
                                         
                                            
                                   
                        
                
            
                                  
                                         
                                            
                        
                 
            
            A.GaussianBlur(
                blur_limit=(23, 23),
                sigma_limit=(0.1, 2.0),
                p=0.5,
            ),
            ToTensorV2(),
        ])

    def __call__(self, image):
        out = self.aug(image=image)["image"]
        return out
    
    
class MAETrainTransform(TransformBase):
    def __init__(self, image_size: int = 224):
        super().__init__(image_size=image_size)

        self.aug = A.Compose([
            A.RandomResizedCrop(
                size=(self.image_size, self.image_size),
                scale=(0.2, 1.0),
                ratio=(0.75, 1.3333),
                interpolation=cv2.INTER_CUBIC,
                p=1.0,
            ),
            
            ChannelIntensityJitter(
                scale_range=(0.9, 1.1),
                shift_range=(-0.05, 0.05),
                per_channel=True,
                p=0.5,
            ),
            
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5), 
            ToTensorV2(),
        ])

    def __call__(self, image):
        out = self.aug(image=image)["image"].float()
        return out
    
    
    
class SSLDataCollator:
    def __init__(self, method: str):
        assert method in ['wsl', 'simclr', 'dino', 'mae'],            f"Unknown method: {method}"
        self.method = method
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate function for batching
        
        Args:
            features: List of dicts from dataset __getitem__
        
        Returns:
            Batched dict ready for model input
        """
        if self.method in ['simsiam', 'simclr']:
            return self._collate_two_views(features)
        elif self.method == 'dino':
            return self._collate_multi_crop(features)
        elif self.method in ['mae']:
            return self._collate_single_view(features)
        elif self.method == 'wsl':
            return self._collate_single_view(features, with_labels=True)
    
    def _collate_two_views(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        view1 = torch.stack([f['views']['view1'] for f in features])
        view2 = torch.stack([f['views']['view2'] for f in features])
        
        return {
            'view1': view1,
            'view2': view2
        }
    
    def _collate_multi_crop(self, features: List[Dict]) -> Dict[str, Any]:
        """
        Collate for DINO with multi-crop
        
        Input: [{
            'student_crops': [10 tensors of [6, H, W]],
            'teacher_crops': [2 tensors of [6, H, W]],
        }, ...]
        
        Output: {
            'student_crops': [10 tensors of [B, 6, H, W]],
            'teacher_crops': [2 tensors of [B, 6, H, W]],
        }
        """
        batch_size = len(features)
        num_student_crops = len(features[0]['views']['student_crops'])
        num_teacher_crops = len(features[0]['views']['teacher_crops'])
        
        student_crops = [
            torch.stack([features[i]['views']['student_crops'][j] for i in range(batch_size)])
            for j in range(num_student_crops)
        ]
        
                             
        teacher_crops = [
            torch.stack([features[i]['views']['teacher_crops'][j] for i in range(batch_size)])
            for j in range(num_teacher_crops)
        ]
        
        return {
            'student_crops': student_crops,
            'teacher_crops': teacher_crops,
        }
    
    def _collate_single_view(self, features: List[Dict], with_labels: bool = False) -> Dict[str, torch.Tensor]:
        """
        Collate for MAE
        
        Input: [{'image': [6, 224, 224]}, ...]
        Output: {'image': [B, 6, 224, 224]}
        """
        images = torch.stack([f['views'] for f in features])
        if with_labels:
            labels = torch.tensor([f['label'] for f in features], dtype=torch.long)
            return {
                'pixel_values': images,
                'labels': labels,
            }
        else:
            return {'pixel_values': images}
