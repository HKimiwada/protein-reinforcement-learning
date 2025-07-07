#!/usr/bin/env python3
"""
check_gpu_availability.py - GPU Memory and Availability Checker

This script helps you check which GPUs are available and plan your multi-seed runs.

Usage:
    python check_gpu_availability.py
    python check_gpu_availability.py --suggest-plan --seeds 5
"""

import subprocess
import torch
import argparse
import psutil
from typing import List, Dict

def get_gpu_info() -> List[Dict]:
    """Get detailed GPU information"""
    gpu_info = []
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return gpu_info
    
    for i in range(torch.cuda.device_count()):
        try:
            # Get basic properties
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / (1024**3)  # Convert to GB
            
            # Get current usage
            torch.cuda.set_device(i)
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            free_memory = total_memory - allocated
            
            # Estimate if GPU is "busy" (>50% memory used)
            usage_percent = (allocated / total_memory) * 100
            is_busy = usage_percent > 50
            
            gpu_info.append({
                'id': i,
                'name': props.name,
                'total_memory_gb': total_memory,
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'free_gb': free_memory,
                'usage_percent': usage_percent,
                'is_busy': is_busy,
                'recommended': not is_busy and free_memory > 8.0  # At least 8GB free
            })
            
        except Exception as e:
            gpu_info.append({
                'id': i,
                'name': 'Unknown',
                'error': str(e),
                'recommended': False
            })
    
    return gpu_info

def get_process_info():
    """Get information about running processes using GPU"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,gpu_uuid,used_memory', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            processes = []
            for line in lines:
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 4:
                        processes.append({
                            'pid': parts[0],
                            'name': parts[1],
                            'memory_mb': parts[3]
                        })
            return processes
    except:
        pass
    return []

def suggest_gpu_plan(gpu_info: List[Dict], num_seeds: int) -> Dict:
    """Suggest optimal GPU allocation plan"""
    
    available_gpus = [gpu for gpu in gpu_info if gpu.get('recommended', False)]
    busy_gpus = [gpu for gpu in gpu_info if gpu.get('is_busy', False)]
    
    if not available_gpus:
        # Find least busy GPUs
        available_gpus = sorted([gpu for gpu in gpu_info if not gpu.get('error')], 
                              key=lambda x: x.get('usage_percent', 100))[:num_seeds]
    
    plan = {
        'recommended_gpus': [gpu['id'] for gpu in available_gpus],
        'start_gpu': available_gpus[0]['id'] if available_gpus else 0,
        'max_workers': min(len(available_gpus), num_seeds),
        'parallel_seeds': min(len(available_gpus), num_seeds),
        'sequential_batches': (num_seeds + len(available_gpus) - 1) // len(available_gpus) if available_gpus else 1,
        'busy_gpus': [gpu['id'] for gpu in busy_gpus]
    }
    
    return plan

def print_gpu_status(gpu_info: List[Dict]):
    """Print formatted GPU status"""
    print("\n🖥️  GPU Status Report")
    print("=" * 80)
    
    if not gpu_info:
        print("No GPUs detected")
        return
    
    print(f"{'GPU':<4} {'Name':<25} {'Memory':<20} {'Usage':<10} {'Status':<12} {'Recommended'}")
    print("-" * 80)
    
    for gpu in gpu_info:
        if 'error' in gpu:
            print(f"{gpu['id']:<4} {'Error':<25} {gpu['error']:<20} {'N/A':<10} {'Error':<12} ❌")
        else:
            memory_str = f"{gpu['free_gb']:.1f}GB free / {gpu['total_memory_gb']:.1f}GB"
            usage_str = f"{gpu['usage_percent']:.1f}%"
            status = "🔴 Busy" if gpu['is_busy'] else "🟢 Free"
            recommended = "✅" if gpu['recommended'] else "❌"
            
            print(f"{gpu['id']:<4} {gpu['name'][:24]:<25} {memory_str:<20} {usage_str:<10} {status:<12} {recommended}")

def print_process_info():
    """Print GPU process information"""
    processes = get_process_info()
    
    if processes:
        print("\n🔄 Running GPU Processes")
        print("=" * 50)
        for proc in processes:
            print(f"PID {proc['pid']}: {proc['name']} ({proc['memory_mb']} MB)")
    else:
        print("\n🔄 No GPU processes detected (or nvidia-smi unavailable)")

def main():
    parser = argparse.ArgumentParser(description='Check GPU availability for multi-seed runs')
    parser.add_argument('--suggest-plan', action='store_true',
                       help='Suggest optimal GPU allocation plan')
    parser.add_argument('--seeds', type=int, default=5,
                       help='Number of seeds for planning (default: 5)')
    
    args = parser.parse_args()
    
    print("🕷️  Spider Silk RL - GPU Availability Checker")
    
    # Get GPU information
    gpu_info = get_gpu_info()
    
    # Print status
    print_gpu_status(gpu_info)
    print_process_info()
    
    # Suggest plan if requested
    if args.suggest_plan and gpu_info:
        plan = suggest_gpu_plan(gpu_info, args.seeds)
        
        print(f"\n📋 Suggested Plan for {args.seeds} Seeds")
        print("=" * 50)
        
        if plan['busy_gpus']:
            print(f"🔴 Busy GPUs to avoid: {plan['busy_gpus']}")
        
        print(f"✅ Recommended GPUs: {plan['recommended_gpus']}")
        print(f"🚀 Start GPU: {plan['start_gpu']}")
        print(f"👥 Max workers: {plan['max_workers']}")
        print(f"⚡ Parallel seeds: {plan['parallel_seeds']}")
        
        if plan['sequential_batches'] > 1:
            print(f"📦 Sequential batches needed: {plan['sequential_batches']}")
        
        print(f"\n💻 Recommended command:")
        print(f"python src/experiments/run_stable_v4.py \\")
        print(f"    --config stable \\")
        print(f"    --episodes 2000 \\")
        print(f"    --multi-seed \\")
        print(f"    --seeds 42,123,456,789,999 \\")  # Adjust based on args.seeds
        print(f"    --research-mode \\")
        print(f"    --start-gpu {plan['start_gpu']} \\")
        print(f"    --max-workers {plan['max_workers']}")
        
        # Memory estimate
        estimated_memory_per_seed = 2.5  # GB
        total_estimated = plan['parallel_seeds'] * estimated_memory_per_seed
        print(f"\n📊 Estimated memory usage: {total_estimated:.1f}GB total ({estimated_memory_per_seed:.1f}GB per seed)")
        
        # Check if plan is feasible
        available_gpus = [gpu for gpu in gpu_info if gpu['id'] in plan['recommended_gpus']]
        min_free_memory = min([gpu['free_gb'] for gpu in available_gpus]) if available_gpus else 0
        
        if min_free_memory < estimated_memory_per_seed:
            print(f"⚠️  Warning: Minimum free memory ({min_free_memory:.1f}GB) may be insufficient")
            print(f"    Consider reducing --max-workers or freeing up GPU memory")
    
    # Additional recommendations
    if gpu_info:
        print(f"\n💡 Tips:")
        print(f"    • Each RL experiment needs ~2.5GB GPU memory")
        print(f"    • Use --start-gpu to skip busy GPUs")
        print(f"    • Monitor with: nvidia-smi -l 1")
        print(f"    • Check progress with: tail -f research_evaluation.log")

if __name__ == "__main__":
    main()