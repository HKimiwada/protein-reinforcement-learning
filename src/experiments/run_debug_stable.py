#!/usr/bin/env python3
"""
Debug-focused integration script to identify and fix the 60% failure rate
"""
import os
import sys
import torch
import argparse
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.config.stable_configs_v2 import get_stable_config_v2
from src.models.improved_policy_v2 import ImprovedSequenceEditPolicyV2
from src.models.stable_reward_function_v2 import StableSpiderSilkRewardFunctionV2
from src.training.debug_stable_trainer import DebugStableTrainer
from src.environment.enhanced_protein_env import EnhancedProteinEditEnvironment
from src.data.dataset import SpiderSilkDataset
from src.utils.spider_silk_utils import SpiderSilkUtils

# Import model loading utilities
from src.debug.debug import fix_both_warnings
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoTokenizer

def setup_debug_system(config_dict, device):
    """Setup system with enhanced debugging components"""
    print("🔧 Setting up Debug System...")
    
    # Load dataset
    print("  📊 Loading dataset...")
    dataset = SpiderSilkDataset(
        config_dict['dataset_path'],
        test_size=config_dict['test_size'],
        n_difficulty_levels=config_dict['n_difficulty_levels'],
        random_state=config_dict['seed']
    )
    print(f"     ✓ Loaded {len(dataset.sequences)} sequences")
    
    # Load models (same as before)
    print("  🧬 Loading ESM-C model...")
    esmc_checkpoint = "src/models/checkpoint-1452"
    if not os.path.exists(esmc_checkpoint):
        print(f"     ❌ ESM-C checkpoint not found at {esmc_checkpoint}")
        return None
        
    esmc_model = AutoModelForMaskedLM.from_pretrained(esmc_checkpoint, trust_remote_code=True)
    esmc_tokenizer = esmc_model.tokenizer
    esmc_tokenizer, esmc_model = fix_both_warnings(esmc_tokenizer, esmc_model)
    print("     ✓ ESM-C model loaded")
    
    print("  🕷️  Loading SilkomeGPT model...")
    trained_model_name = 'lamm-mit/SilkomeGPT'
    try:
        silkomegpt_tokenizer = AutoTokenizer.from_pretrained(trained_model_name, trust_remote_code=True)
        silkomegpt_tokenizer.pad_token = silkomegpt_tokenizer.eos_token
        silkomegpt_model = AutoModelForCausalLM.from_pretrained(trained_model_name, trust_remote_code=True)
        silkomegpt_model.config.use_cache = False
        print("     ✓ SilkomeGPT model loaded")
    except Exception as e:
        print(f"     ❌ Failed to load SilkomeGPT: {e}")
        return None
    
    # Build components with enhanced versions
    print("  🔧 Building enhanced components...")
    utils = SpiderSilkUtils(esmc_model, esmc_tokenizer)
    reward_fn = StableSpiderSilkRewardFunctionV2(silkomegpt_model, silkomegpt_tokenizer, esmc_model)
    
    # Move models to device
    reward_fn.silkomegpt.to(device)
    reward_fn.esmc.to(device)
    
    # Use ENHANCED environment with better error handling
    env = EnhancedProteinEditEnvironment(utils, reward_fn, max_steps=config_dict['max_steps'])
    
    # Use improved policy with bounds fixes
    policy = ImprovedSequenceEditPolicyV2().to(device)
    
    print("     ✓ All debug components created successfully")
    
    return policy, env, dataset, utils, reward_fn

def run_debug_test(episodes=100):
    """Run focused debug test to identify failure modes"""
    print("🚨 DEBUGGING MODE: Identifying Failure Patterns")
    print("="*60)
    
    # Use test config
    config = get_stable_config_v2('stable_test')
    config_dict = config.to_dict()
    config_dict['max_steps'] = 15  # Shorter episodes for faster debugging
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup debug system
    setup_result = setup_debug_system(config_dict, device)
    if setup_result is None:
        print("❌ Setup failed")
        return
    
    policy, env, dataset, utils, reward_fn = setup_result
    
    # Create debug trainer
    trainer = DebugStableTrainer(policy, env, lr=1e-4, device=device)
    
    print(f"\n🔍 Running {episodes} debug episodes...")
    
    # Track failure patterns
    total_failures = 0
    major_failures = 0
    successes = 0
    
    for episode in range(episodes):
        # Get a test sequence
        seq_idx = episode % len(dataset.train_sequences)
        test_seq = dataset.train_sequences[seq_idx]
        
        # Train episode and capture detailed info
        result = trainer.train_episode(test_seq, episode)
        
        # Analyze the result
        episode_reward = result['episode_reward']
        actual_improvement = result.get('actual_improvement', 0)
        
        if episode_reward < -5.0:
            major_failures += 1
            print(f"Episode {episode}: MAJOR FAILURE (reward: {episode_reward:.3f})")
        elif episode_reward < -1.0:
            total_failures += 1
            print(f"Episode {episode}: failure (reward: {episode_reward:.3f})")
        elif actual_improvement > 0.001:
            successes += 1
            print(f"Episode {episode}: SUCCESS! (reward: {episode_reward:.3f}, improvement: {actual_improvement:.4f})")
        
        # Print progress every 20 episodes
        if (episode + 1) % 20 == 0:
            print(f"\n📊 Progress after {episode + 1} episodes:")
            print(f"  Major failures: {major_failures} ({major_failures/(episode+1)*100:.1f}%)")
            print(f"  Total failures: {total_failures} ({total_failures/(episode+1)*100:.1f}%)")
            print(f"  Successes: {successes} ({successes/(episode+1)*100:.1f}%)")
            print()
    
    # Final summary
    print(f"\n🎯 FINAL DEBUGGING SUMMARY")
    print("="*60)
    print(trainer.get_debug_summary())
    
    # Save debug data
    debug_file = f"debug_results_{episodes}_episodes.json"
    trainer.save_debug_data(debug_file)
    print(f"\n💾 Debug data saved to {debug_file}")
    
    return trainer

def run_quick_test():
    """Run a very quick test to check basic functionality"""
    print("⚡ QUICK TEST: Basic Functionality Check")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = get_stable_config_v2('stable_test')
    config_dict = config.to_dict()
    config_dict['max_steps'] = 5  # Very short episodes
    
    # Setup
    setup_result = setup_debug_system(config_dict, device)
    if setup_result is None:
        print("❌ Quick test failed - setup issues")
        return False
    
    policy, env, dataset, utils, reward_fn = setup_result
    trainer = DebugStableTrainer(policy, env, lr=1e-4, device=device)
    
    # Test 5 episodes quickly
    print("Running 5 quick test episodes...")
    
    for episode in range(5):
        test_seq = dataset.train_sequences[episode % len(dataset.train_sequences)]
        result = trainer.train_episode(test_seq, episode)
        
        reward = result['episode_reward']
        improvement = result.get('actual_improvement', 0)
        
        print(f"Episode {episode}: reward={reward:.3f}, improvement={improvement:.4f}")
    
    print("\n✅ Quick test completed successfully!")
    print("System appears to be working. Run full debug with:")
    print("python src/experiments/run_debug_stable.py --episodes 100")
    
    return True

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Debug Stable Spider Silk RL System')
    parser.add_argument('--episodes', type=int, default=100, 
                       help='Number of episodes to run for debugging')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick functionality test (5 episodes)')
    parser.add_argument('--device', default='auto',
                       help='Device to use (auto/cpu/cuda)')
    
    args = parser.parse_args()
    
    print("🕷️  Spider Silk RL Debug System")
    print("="*60)
    
    if args.quick:
        success = run_quick_test()
        if success:
            print("\n🎉 Quick test passed! System is functional.")
        else:
            print("\n❌ Quick test failed! Check setup.")
        return
    
    # Run full debug test
    try:
        trainer = run_debug_test(args.episodes)
        
        if trainer:
            print("\n🎉 Debug test completed successfully!")
            print("Check the debug summary above for failure analysis.")
        else:
            print("\n❌ Debug test failed!")
            
    except KeyboardInterrupt:
        print("\n⏹️  Debug test interrupted by user")
    except Exception as e:
        print(f"\n❌ Debug test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()