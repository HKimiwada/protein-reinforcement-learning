#!/usr/bin/env python3
"""
Break the plateau with aggressive exploration
"""
import sys; sys.path.append('.')
import torch
from src.config.training_configs import get_config
from src.debug.debug import setup_models_and_environment
from src.training.ppo_trainer import PPOTrainer

def plateau_breaker():
    print("🚀 PLATEAU BREAKING STRATEGY")
    print("="*50)
    
    device = torch.device('cuda')
    config = get_config('default').to_dict()  # Use full dataset
    policy, env, dataset, utils, reward_fn = setup_models_and_environment(config, device)
    
    # Create aggressive trainer
    trainer = PPOTrainer(policy, env, lr=1e-3, device=device)  # 3x higher LR
    trainer.entropy_coeff = 0.5  # Much higher exploration
    trainer.clip_epsilon = 0.3   # Larger policy updates
    
    print("🔥 Aggressive learning parameters:")
    print(f"  Learning rate: {trainer.lr}")
    print(f"  Entropy coeff: {trainer.entropy_coeff}")
    print(f"  Clip epsilon: {trainer.clip_epsilon}")
    
    # Train with aggressive settings
    best_reward = 5.583  # Current best
    improvement_count = 0
    
    for ep in range(100):  # 100 episodes with aggressive settings
        # Random sequence from full dataset
        seq_idx = ep % len(dataset.train_sequences)
        seq = dataset.train_sequences[seq_idx]
        
        result = trainer.train_episode(seq, ep + 400)  # Continue from episode 400
        current_reward = result['episode_reward']
        
        if current_reward > best_reward:
            best_reward = current_reward
            improvement_count += 1
            print(f"🎉 NEW BEST! Episode {ep + 400}: {current_reward:.3f}")
        
        if ep % 10 == 0:
            print(f"Episode {ep + 400}: reward={current_reward:.3f}, best={best_reward:.3f}")
    
    print(f"\n📊 Results:")
    print(f"  Starting best: 5.583")
    print(f"  Final best: {best_reward:.3f}")
    print(f"  Improvements: {improvement_count}")
    
    if best_reward > 5.583:
        print("✅ Plateau broken! Continue with these settings.")
    else:
        print("❌ Plateau persists. Consider different approach.")

if __name__ == "__main__":
    plateau_breaker()