import torch
import os
import json
from typing import Dict, Any

def save_checkpoint(path: str, episode: int, policy: torch.nn.Module, 
                   optimizer: torch.optim.Optimizer, config: Dict[str, Any]):
    """Save training checkpoint"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Handle DDP models
    if hasattr(policy, 'module'):
        model_state = policy.module.state_dict()
    else:
        model_state = policy.state_dict()
    
    checkpoint = {
        'episode': episode,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config
    }
    
    torch.save(checkpoint, path)

def load_checkpoint(path: str, policy: torch.nn.Module, 
                   optimizer: torch.optim.Optimizer = None):
    """Load training checkpoint"""
    checkpoint = torch.load(path, map_location='cpu')
    
    # Handle DDP models
    if hasattr(policy, 'module'):
        policy.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        policy.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['episode'], checkpoint['config']