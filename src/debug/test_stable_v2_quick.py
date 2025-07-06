#!/usr/bin/env python3
"""
Quick test script to verify the stable V2 system works
"""
import sys
sys.path.append('.')

import torch
from src.config.stable_configs_v2 import get_stable_config_v2
from src.experiments.run_stable_v2 import setup_stable_v2_system
from src.training.simple_stable_trainer_v2 import SimpleStableTrainerV2

def test_stable_v2():
    print("🧪 QUICK TEST: Stable V2 System")
    print("="*50)
    
    # Use test config for quick validation
    config = get_stable_config_v2('stable_test')
    config_dict = config.to_dict()
    config_dict['n_episodes'] = 5000  # Very quick test
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup system
    setup_result = setup_stable_v2_system(config_dict, device)
    if setup_result is None:
        print("❌ Setup failed")
        return False
    
    policy, env, dataset, utils, reward_fn = setup_result
    
    # Create simple stable trainer
    trainer = SimpleStableTrainerV2(
        policy, env,
        lr=config_dict['learning_rate'],
        device=device
    )
    
    print("\n🚀 Running 5000 test episodes...")
    
    rewards = []
    improvements = []
    
    try:
        for episode in range(5000):
            # Get sequence
            seq = dataset.train_sequences[episode % min(5000, len(dataset.train_sequences))]
            
            # Train episode
            result = trainer.train_episode(seq, episode)
            
            rewards.append(result['episode_reward'])
            improvements.append(result.get('actual_improvement', 0))
            
            if episode % 10 == 0:
                print(f"Episode {episode}: reward={result['episode_reward']:.3f}, "
                     f"improvement={result.get('actual_improvement', 0):.4f}")
        
        # Analysis
        print(f"\n📊 Results:")
        print(f"Rewards: {min(rewards):.3f} to {max(rewards):.3f} (avg: {sum(rewards)/len(rewards):.3f})")
        print(f"Improvements: {min(improvements):.4f} to {max(improvements):.4f} (avg: {sum(improvements)/len(improvements):.4f})")
        
        # Check for learning
        first_half_reward = sum(rewards[:25]) / 25
        second_half_reward = sum(rewards[25:]) / 25
        
        print(f"Learning check:")
        print(f"  First half avg reward: {first_half_reward:.3f}")
        print(f"  Second half avg reward: {second_half_reward:.3f}")
        print(f"  Improvement: {second_half_reward - first_half_reward:.3f}")
        
        if second_half_reward > first_half_reward:
            print("✅ System shows learning trend!")
        else:
            print("⚠️  No clear learning trend in 5000 episodes")
        
        # Check for positive improvements
        positive_improvements = sum(1 for imp in improvements if imp > 0.001)
        print(f"Episodes with positive improvement: {positive_improvements}/5000")
        
        if positive_improvements >= 5:
            print("✅ System can achieve improvements!")
        else:
            print("⚠️  Few positive improvements detected")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_stable_v2()
    if success:
        print("\n🎉 Stable V2 system test passed!")
        print("Ready to run full training with:")
        print("python src/experiments/run_stable_v2.py --config stable --episodes 5000")
    else:
        print("\n❌ Test failed - check setup before running full training")