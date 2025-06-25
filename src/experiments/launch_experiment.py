#!/usr/bin/env python3
import os
import sys
import argparse
import json

# Add config to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.training_configs import get_config
from experiments.train_distributed import main as train_main

def main():
    parser = argparse.ArgumentParser(description='Launch Spider Silk RL Training')
    parser.add_argument('--config', default='default', help='Configuration name')
    parser.add_argument('--dataset-path', help='Override dataset path')
    parser.add_argument('--episodes', type=int, help='Override number of episodes')
    parser.add_argument('--gpus', type=int, help='Override number of GPUs')
    parser.add_argument('--run-name', help='Override run name')
    parser.add_argument('--wandb-project', help='Override wandb project name')
    
    args = parser.parse_args()
    
    # Get base configuration
    config = get_config(args.config)
    
    # Apply overrides
    if args.dataset_path:
        config.dataset_path = args.dataset_path
    if args.episodes:
        config.n_episodes = args.episodes
    if args.gpus:
        config.world_size = args.gpus
    if args.run_name:
        config.run_name = args.run_name
    if args.wandb_project:
        config.project_name = args.wandb_project
    
    # Convert to dict
    config_dict = config.to_dict()
    
    print(f"🚀 Launching: {config_dict['run_name']}")
    print(f"📊 Strategy: {config_dict['curriculum_strategy']}")
    print(f"🖥️  GPUs: {config_dict['world_size']}")
    print(f"📈 Episodes: {config_dict['n_episodes']}")
    
    # Save config
    os.makedirs(config_dict['save_dir'], exist_ok=True)
    config_path = os.path.join(config_dict['save_dir'], f"{config_dict['run_name']}_config.json")
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Launch training
    train_main(config_dict)

if __name__ == "__main__":
    main()
