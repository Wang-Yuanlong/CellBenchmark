# Benchmarking Existing SSL Methods Algorithms on Cellular Image Data

> Jiayuan Chen, Yuanlong Wang, Xianhui Chen

## Running Guide

### Prerequisites

First, install python package requirements:

```
transformers              4.40.0
timm                      0.9.16
torch                     2.3.1
numpy                     1.26.4
```

Next, download necessary dependencies.

**Baseline Method:**

```bash
cd resources
git clone https://github.com/recursionpharma/maes_microscopy.git
```
**Baseline Method - pretrained weights:**

```bash
cd resources
git clone https://huggingface.co/recursionpharma/OpenPhenom
```

### Model Training

First replace the placeholder of wandb entity at the Line 24 of `sc/scripts/train.py`.

```bash
cd sc/scripts
python train.py --method <method> --config_path ../configs/<method>.yaml
```

Here the `<method>` could be `dino`, `mae`, `simclr`, or `wsl`.

### Model Evaluation

```bash
python sc/eval/ggi.py \
        --model_type <method> \
        --model_path resources/OpenPhenom \
        --batch_key experiment_name \
        --batch_size 256 \
        --device cuda \
        --num_workers 16 \
        --pin_memory \
        --use_pca_cs \
        --use_four_quad_crops
```

Here the `<method>` could be `dino`, `mae`, `simclr`, `wsl`, or `baseline`. The argument `model_path` only works when evaluating the `baseline` method.