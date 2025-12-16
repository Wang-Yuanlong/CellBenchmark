import timm
import re
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  
from models.hfmodel import MCViT

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



def get_model(pretrained_path, encoder_config, encoder_ckpt=False):
    encoder = MCViT(encoder_config)
    if pretrained_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        ckpt = load_file(pretrained_path, device='cpu')
        print(f'Loaded pretrained weights from {pretrained_path}')
        if not encoder_ckpt:
            new_ckpt = {}
            for k, v in ckpt.items():
                if k.startswith('encoder.'):
                    new_k = k[len('encoder.'):]
                elif k.startswith('student.'):
                    new_k = k[len('student.'):]
                else:
                    continue
                new_ckpt[new_k] = v
        else:
            new_ckpt = ckpt

    encoder.load_state_dict(new_ckpt, strict=True)
    return encoder
