"""
Distributed Training Utilities

This module provides utilities for setting up and managing distributed
training across multiple GPUs and nodes.
"""

import os
import sys
import logging
import socket
import subprocess
from typing import Optional, Dict, Any, List
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from contextlib import contextmanager
import time
import psutil
from datetime import timedelta

logger = logging.getLogger(__name__)

class DistributedConfig:
    """Configuration for distributed training"""
    
    def __init__(self,
                 world_size: int,
                 rank: int = 0,
                 local_rank: int = 0,
                 backend: str = 'nccl',
                 init_method: str = 'env://',
                 master_addr: str = 'localhost',
                 master_port: str = '12355'):
        
        self.world_size = world_size
        self.rank = rank
        self.local_rank = local_rank
        self.backend = backend
        self.init_method = init_method
        self.master_addr = master_addr
        self.master_port = master_port
    
    def to_env_vars(self) -> Dict[str, str]:
        """Convert config to environment variables"""
        return {
            'WORLD_SIZE': str(self.world_size),
            'RANK': str(self.rank),
            'LOCAL_RANK': str(self.local_rank),
            'MASTER_ADDR': self.master_addr,
            'MASTER_PORT': self.master_port
        }


def find_free_port() -> int:
    """Find a free port for distributed training"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def setup_distributed_training(rank: int, 
                              world_size: int, 
                              backend: str = 'nccl',
                              master_addr: str = 'localhost',
                              master_port: Optional[str] = None,
                              timeout_minutes: int = 30) -> DistributedConfig:
    """
    Setup distributed training environment
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        backend: Communication backend ('nccl', 'gloo', 'mpi')
        master_addr: Master node address
        master_port: Master node port (auto-selected if None)
        timeout_minutes: Timeout for initialization
        
    Returns:
        DistributedConfig object
    """
    
    # Auto-select port if not provided
    if master_port is None:
        if rank == 0:
            master_port = str(find_free_port())
        else:
            master_port = '12355'  # Default fallback
    
    # Set environment variables
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    
    # Set CUDA device
    if torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())
        local_rank = rank % torch.cuda.device_count()
    else:
        local_rank = rank
    
    # Initialize process group
    timeout = torch.distributed.default_pg_timeout
    if timeout_minutes > 0:
        timeout = timedelta(minutes=timeout_minutes)
   
    try:
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            timeout=timeout
        )
        
        logger.info(f"Distributed training initialized: rank={rank}/{world_size}, "
                   f"backend={backend}, device=cuda:{local_rank}")
        
        return DistributedConfig(
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
            backend=backend,
            master_addr=master_addr,
            master_port=master_port
        )
        
    except Exception as e:
        logger.error(f"Failed to initialize distributed training: {e}")
        raise


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed training cleaned up")


@contextmanager
def distributed_context(rank: int, world_size: int, **kwargs):
    """Context manager for distributed training setup/cleanup"""
    config = None
    try:
        config = setup_distributed_training(rank, world_size, **kwargs)
        yield config
    finally:
        cleanup_distributed()


def is_main_process(rank: Optional[int] = None) -> bool:
    """Check if current process is the main process"""
    if rank is not None:
        return rank == 0
    
    if dist.is_initialized():
        return dist.get_rank() == 0
    
    return True


def get_world_size() -> int:
    """Get world size for distributed training"""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """Get current process rank"""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def barrier():
    """Synchronization barrier for all processes"""
    if dist.is_initialized():
        dist.barrier()


def all_reduce_tensor(tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
    """All-reduce operation on tensor across all processes"""
    if dist.is_initialized():
        dist.all_reduce(tensor, op=op)
    return tensor


def all_gather_tensor(tensor: torch.Tensor) -> List[torch.Tensor]:
    """All-gather operation on tensor across all processes"""
    if not dist.is_initialized():
        return [tensor]
    
    world_size = dist.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return gathered_tensors


def broadcast_tensor(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast tensor from source rank to all other ranks"""
    if dist.is_initialized():
        dist.broadcast(tensor, src=src)
    return tensor


class DistributedMetrics:
    """Utility for aggregating metrics across distributed processes"""
    
    def __init__(self):
        self.metrics = {}
    
    def update(self, metrics_dict: Dict[str, float]):
        """Update metrics dictionary"""
        for key, value in metrics_dict.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
    
    def aggregate(self, method: str = 'mean') -> Dict[str, float]:
        """Aggregate metrics across all processes"""
        if not dist.is_initialized():
            # Single process - just compute local aggregation
            return self._local_aggregate(method)
        
        aggregated = {}
        for key, values in self.metrics.items():
            if not values:
                continue
                
            # Convert to tensor
            local_tensor = torch.tensor(values, dtype=torch.float32)
            
            if method == 'mean':
                local_mean = local_tensor.mean()
                local_count = torch.tensor(len(values), dtype=torch.float32)
                
                # All-reduce across processes
                global_sum = all_reduce_tensor(local_mean * local_count)
                global_count = all_reduce_tensor(local_count)
                
                aggregated[key] = (global_sum / global_count).item()
                
            elif method == 'sum':
                local_sum = local_tensor.sum()
                global_sum = all_reduce_tensor(local_sum)
                aggregated[key] = global_sum.item()
                
            elif method == 'max':
                local_max = local_tensor.max()
                global_max = all_reduce_tensor(local_max, op=dist.ReduceOp.MAX)
                aggregated[key] = global_max.item()
                
            elif method == 'min':
                local_min = local_tensor.min()
                global_min = all_reduce_tensor(local_min, op=dist.ReduceOp.MIN)
                aggregated[key] = global_min.item()
        
        return aggregated
    
    def _local_aggregate(self, method: str) -> Dict[str, float]:
        """Local aggregation for single process"""
        aggregated = {}
        for key, values in self.metrics.items():
            if not values:
                continue
                
            if method == 'mean':
                aggregated[key] = sum(values) / len(values)
            elif method == 'sum':
                aggregated[key] = sum(values)
            elif method == 'max':
                aggregated[key] = max(values)
            elif method == 'min':
                aggregated[key] = min(values)
        
        return aggregated
    
    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()


def check_gpu_availability() -> Dict[str, Any]:
    """Check GPU availability and return system info"""
    info = {
        'cuda_available': torch.cuda.is_available(),
        'gpu_count': 0,
        'gpu_memory': [],
        'gpu_names': [],
        'driver_version': None
    }
    
    if torch.cuda.is_available():
        info['gpu_count'] = torch.cuda.device_count()
        
        for i in range(info['gpu_count']):
            props = torch.cuda.get_device_properties(i)
            info['gpu_names'].append(props.name)
            info['gpu_memory'].append(props.total_memory // (1024**3))  # GB
        
        try:
            # Get driver version
            result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', 
                                   '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                info['driver_version'] = result.stdout.strip().split('\n')[0]
        except:
            pass
    
    return info


def print_distributed_info(rank: int, world_size: int):
    """Print distributed training information"""
    if is_main_process(rank):
        gpu_info = check_gpu_availability()
        
        print(f"🚀 Distributed Training Information")
        print(f"=" * 50)
        print(f"World Size: {world_size}")
        print(f"Backend: {dist.get_backend() if dist.is_initialized() else 'None'}")
        print(f"CUDA Available: {gpu_info['cuda_available']}")
        print(f"GPU Count: {gpu_info['gpu_count']}")
        
        if gpu_info['gpu_names']:
            print(f"GPU Models:")
            for i, (name, memory) in enumerate(zip(gpu_info['gpu_names'], gpu_info['gpu_memory'])):
                print(f"  GPU {i}: {name} ({memory} GB)")
        
        if gpu_info['driver_version']:
            print(f"Driver Version: {gpu_info['driver_version']}")
        
        # System info
        print(f"CPU Count: {psutil.cpu_count()}")
        print(f"RAM: {psutil.virtual_memory().total // (1024**3)} GB")
        print(f"=" * 50)


def save_distributed_checkpoint(model, optimizer, epoch, rank, save_path):
    """Save checkpoint in distributed training"""
    if is_main_process(rank):
        # Only main process saves
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'rank': rank,
            'world_size': get_world_size()
        }
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved to {save_path}")
    
    # Synchronize all processes
    barrier()


def load_distributed_checkpoint(model, optimizer, checkpoint_path, rank):
    """Load checkpoint in distributed training"""
    # All processes load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{rank}')
    
    # Load model state
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    
    if is_main_process(rank):
        logger.info(f"Checkpoint loaded from {checkpoint_path}, epoch {epoch}")
    
    return epoch


class DistributedSampler:
    """Simple distributed sampler for sequences"""
    
    def __init__(self, sequences: List[str], rank: int, world_size: int, shuffle: bool = True):
        self.sequences = sequences
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.epoch = 0
    
    def __iter__(self):
        # Shuffle sequences if requested
        if self.shuffle:
            indices = torch.randperm(len(self.sequences), generator=torch.Generator().manual_seed(self.epoch))
            sequences = [self.sequences[i] for i in indices]
        else:
            sequences = self.sequences
        
        # Distribute sequences across ranks
        per_rank = len(sequences) // self.world_size
        start_idx = self.rank * per_rank
        
        if self.rank == self.world_size - 1:
            # Last rank gets remaining sequences
            end_idx = len(sequences)
        else:
            end_idx = start_idx + per_rank
        
        return iter(sequences[start_idx:end_idx])
    
    def __len__(self):
        per_rank = len(self.sequences) // self.world_size
        if self.rank == self.world_size - 1:
            return len(self.sequences) - (self.world_size - 1) * per_rank
        return per_rank
    
    def set_epoch(self, epoch: int):
        """Set epoch for shuffling"""
        self.epoch = epoch


def monitor_gpu_memory(rank: int, log_interval: int = 100):
    """Monitor GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(rank) / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved(rank) / (1024**3)    # GB
        
        if is_main_process(rank):
            logger.info(f"GPU {rank} Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
        
        return {'allocated_gb': allocated, 'reserved_gb': reserved}
    
    return {'allocated_gb': 0, 'reserved_gb': 0}


def optimize_distributed_training():
    """Apply optimizations for distributed training"""
    # Set environment variables for optimal performance
    optimizations = {
        'NCCL_DEBUG': 'WARN',  # Reduce NCCL logging
        'NCCL_TREE_THRESHOLD': '0',  # Use tree algorithm
        'CUDA_LAUNCH_BLOCKING': '0',  # Async CUDA operations
        'TORCH_NCCL_BLOCKING_WAIT': '1',  # Blocking wait for stability
    }
    
    for key, value in optimizations.items():
        if key not in os.environ:
            os.environ[key] = value
    
    # Set CUDA optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # Optimize for fixed input sizes
        torch.cuda.empty_cache()  # Clear cache
    
    logger.info("Applied distributed training optimizations")