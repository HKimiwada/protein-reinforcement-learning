#!/usr/bin/env python3
"""
Integration script to run the stable V2 system
Uses new components while keeping original code intact
"""
import os
import sys
import argparse
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.config.stable_configs_v2 import get_stable_config_v2
from src.models.improved_policy_v2 import ImprovedSequenceEditPolicyV2
from src.models.stable_reward_function_v2 import StableSpiderSilkRewardFunctionV2
from src.training.simple_stable_trainer_v2 import SimpleStableTrainerV2
from src.data.dataset import SpiderSilkDataset
from src.utils.spider_silk_utils import SpiderSilkUtils
from src.environment.protein_env import ProteinEditEnvironment

# Import the model loading from your working debug script
from src.debug.debug import fix_both_warnings
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoTokenizer

def setup_stable_v2_system(config_dict, device):
    """Setup the stable V2 system with new components"""
    print("🔧 Setting up Stable V2 system...")
    
    # Load dataset
    print("  📊 Loading dataset...")
    dataset = SpiderSilkDataset(
        config_dict['dataset_path'],
        test_size=config_dict['test_size'],
        n_difficulty_levels=config_dict['n_difficulty_levels'],
        random_state=config_dict['seed']
    )
    print(f"     ✓ Loaded {len(dataset.sequences)} sequences")
    
    # Initialize ESM-C model (same as original)
    print("  🧬 Loading ESM-C model...")
    esmc_checkpoint = "src/models/checkpoint-1452"
    if not os.path.exists(esmc_checkpoint):
        print(f"     ❌ ESM-C checkpoint not found at {esmc_checkpoint}")
        return None
        
    esmc_model = AutoModelForMaskedLM.from_pretrained(esmc_checkpoint, trust_remote_code=True)
    esmc_tokenizer = esmc_model.tokenizer
    esmc_tokenizer, esmc_model = fix_both_warnings(esmc_tokenizer, esmc_model)
    print("     ✓ ESM-C model loaded")
    
    # Initialize SilkomeGPT model (same as original)
    print("  🕷️  Loading SilkomeGPT model...")
    trained_model_name = 'lamm-mit/SilkomeGPT'
    try:
        silkomegpt_tokenizer = AutoTokenizer.from_pretrained(trained_model_name, trust_remote_code=True)
        silkomegpt_tokenizer.pad_token = silkomegpt_tokenizer.eos_token
        silkomegpt_model = AutoModelForCausalLM.from_pretrained(
            trained_model_name,
            trust_remote_code=True
        )
        silkomegpt_model.config.use_cache = False
        print("     ✓ SilkomeGPT model loaded")
    except Exception as e:
        print(f"     ❌ Failed to load SilkomeGPT: {e}")
        return None
    
    # Build utility components (same as original)
    print("  🔧 Building components...")
    utils = SpiderSilkUtils(esmc_model, esmc_tokenizer)
    
    # Use NEW stable reward function V2
    reward_fn = StableSpiderSilkRewardFunctionV2(
        silkomegpt_model, silkomegpt_tokenizer, esmc_model
    )
    
    # Move models to correct device
    reward_fn.silkomegpt.to(device)
    reward_fn.esmc.to(device)
    
    # Create environment (same as original)
    env = ProteinEditEnvironment(utils, reward_fn, max_steps=config_dict['max_steps'])
    
    # Use NEW improved policy V2
    policy = ImprovedSequenceEditPolicyV2().to(device)
    
    print("     ✓ All stable V2 components created successfully")
    
    return policy, env, dataset, utils, reward_fn

def main():
    """Main function to run stable V2 training"""
    parser = argparse.ArgumentParser(description='Run Stable Spider Silk RL V2')
    parser.add_argument('--config', default='stable', 
                       choices=['stable', 'stable_conservative', 'stable_aggressive', 'stable_test'],
                       help='Configuration variant to use')
    parser.add_argument('--episodes', type=int, help='Override number of episodes')
    parser.add_argument('--device', default='auto', help='Device to use (auto/cpu/cuda)')
    
    args = parser.parse_args()
    
    print("🕷️  Spider Silk RL Stable V2 Training")
    print("="*60)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"🖥️  Using device: {device}")
    
    # Get configuration
    config = get_stable_config_v2(args.config)
    config_dict = config.to_dict()
    
    # Override episodes if provided
    if args.episodes:
        config_dict['n_episodes'] = args.episodes
    
    print(f"📋 Using config: {args.config}")
    print(f"   Episodes: {config_dict['n_episodes']}")
    print(f"   Learning rate: {config_dict['learning_rate']}")
    print(f"   Max steps: {config_dict['max_steps']}")
    print(f"   Curriculum: {config_dict['curriculum_strategy']}")
    
    # Setup the stable V2 system
    setup_result = setup_stable_v2_system(config_dict, device)
    if setup_result is None:
        print("❌ Failed to setup stable V2 system")
        return 1
    
    policy, env, dataset, utils, reward_fn = setup_result
    
    # Create NEW stable trainer V2
    print("🚀 Creating Simple Stable V2 Trainer...")
    trainer = SimpleStableTrainerV2(
        policy, env,
        lr=config_dict['learning_rate'],
        device=device
    )
    
    # Training loop with enhanced monitoring
    print(f"\n🏃 Starting training for {config_dict['n_episodes']} episodes...")
    
    best_reward = -float('inf')
    best_improvement = 0.0
    
    try:
        for episode in range(config_dict['n_episodes']):
            # Get curriculum-based sequence
            start_seq, difficulty_level = dataset.get_curriculum_sequence(
                episode, config_dict['n_episodes'], config_dict['curriculum_strategy']
            )
            
            # Train episode with stable V2 trainer
            episode_data = trainer.train_episode(start_seq, episode, difficulty_level)
            
            # Track best performance
            if episode_data['episode_reward'] > best_reward:
                best_reward = episode_data['episode_reward']
            
            if episode_data.get('actual_improvement', 0) > best_improvement:
                best_improvement = episode_data['actual_improvement']
            
            # Enhanced logging every N episodes
            if episode % config_dict['log_interval'] == 0 and episode > 0:
                avg_reward_ma = episode_data.get('avg_reward_ma', 0)
                avg_improvement_ma = episode_data.get('avg_improvement_ma', 0)
                reward_std = episode_data.get('reward_std', 0)
                reward_trend = episode_data.get('reward_trend', 0)
                current_lr = episode_data.get('current_lr', config_dict['learning_rate'])
                
                print(f"Episode {episode}:")
                print(f"  Current: reward={episode_data['episode_reward']:.3f}, improvement={episode_data.get('actual_improvement', 0):.4f}")
                print(f"  Trends:  avg_reward={avg_reward_ma:.3f}, avg_improvement={avg_improvement_ma:.4f}")
                print(f"  Stats:   std={reward_std:.3f}, trend={reward_trend:.3f}, lr={current_lr:.6f}")
                print(f"  Best:    reward={best_reward:.3f}, improvement={best_improvement:.4f}")
            
            # Test evaluation every N episodes
            if episode % config_dict['test_interval'] == 0 and episode > 0:
                print(f"\n📊 Testing at episode {episode}...")
                
                # Test on a few sequences
                test_sequences = dataset.get_test_sequences(5)
                test_improvements = []
                
                for test_seq in test_sequences:
                    # Get original toughness
                    orig_tough, _ = reward_fn.predict_toughness(test_seq)
                    
                    # Run trained policy
                    state = env.reset(test_seq).to(device)
                    policy.eval()
                    with torch.no_grad():
                        for _ in range(config_dict['max_steps']):
                            action = policy.get_action(state, deterministic=True)
                            state, reward, done, info = env.step(action)
                            state = state.to(device)
                            if done:
                                break
                    
                    # Get final toughness
                    final_tough, _ = reward_fn.predict_toughness(env.current_sequence)
                    improvement = final_tough - orig_tough
                    test_improvements.append(improvement)
                
                avg_test_improvement = sum(test_improvements) / len(test_improvements)
                print(f"  Test avg improvement: {avg_test_improvement:.4f}")
                print(f"  Individual improvements: {[f'{imp:.4f}' for imp in test_improvements]}")
                
                policy.train()  # Back to training mode
            
            # Save checkpoint periodically
            if episode % config_dict['checkpoint_interval'] == 0 and episode > 0:
                save_dir = os.path.join(config_dict['save_dir'], config_dict['run_name'])
                os.makedirs(save_dir, exist_ok=True)
                checkpoint_path = os.path.join(save_dir, f"checkpoint_stable_v2_ep_{episode}.pt")
                
                torch.save({
                    'episode': episode,
                    'model_state_dict': policy.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'config': config_dict,
                    'best_reward': best_reward,
                    'best_improvement': best_improvement
                }, checkpoint_path)
                
                print(f"💾 Checkpoint saved: {checkpoint_path}")
    
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Final summary
    print(f"\n🎯 Training Complete!")
    print(f"Best reward: {best_reward:.3f}")
    print(f"Best improvement: {best_improvement:.4f}")
    print(f"Final moving averages:")
    print(f"  Reward: {episode_data.get('avg_reward_ma', 0):.3f}")
    print(f"  Improvement: {episode_data.get('avg_improvement_ma', 0):.4f}")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)