import argparse
import os
import sys
import wandb
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
from pretrain.simclr import train_simclr


def main():
    parser = argparse.ArgumentParser(description="Pretraining script for self-supervised learning methods")
    parser.add_argument("--method", type=str, default="simclr", 
                        help="Method type: 'dino' or 'mae' or 'simclr'")

    parser.add_argument("--wandb_project", type=str, default="cellbench-pretrain", 
                        help="WandB project name")

    parser.add_argument("--config_path", type=str, default=None,
                        help="Path to config yaml file")

    args = parser.parse_args()
    wandb.init(project=args.wandb_project,
               entity = '<your_wandb_entity>',)
    


    if args.method == "simclr":
        train_simclr(args.config_path)
    elif args.method == "dino":
        from pretrain.dino import train_dino
        train_dino(args.config_path)
        
    elif args.method == "wsl":
        from pretrain.wsl import train_wsl
        train_wsl(args.config_path)
        
    elif args.method == "mae":
        from pretrain.mae import train_mae
        train_mae(args.config_path)
if __name__ == "__main__":
    
    main()