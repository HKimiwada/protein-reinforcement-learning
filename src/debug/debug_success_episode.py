#!/usr/bin/env python3
"""
Debug what's happening in the "success" episodes (5.0 rewards)
"""
import sys; sys.path.append('.')
import torch
from src.config.training_configs import get_config
from src.debug.debug import setup_models_and_environment
from src.training.ppo_trainer import PPOTrainer

def debug_success_episodes():
    print("🔍 DEBUGGING SUCCESS EPISODES")
    print("="*50)
    
    device = torch.device('cuda')
    config = get_config('phase1').to_dict()
    policy, env, dataset, utils, reward_fn = setup_models_and_environment(config, device)
    trainer = PPOTrainer(policy, env, lr=1e-4, device=device)
    
    # Train briefly to get a somewhat trained policy
    print("Training 50 episodes quickly...")
    for ep in range(50):
        seq = dataset.sequences[ep % 6]
        trainer.train_episode(seq, ep)
    
    print("\n🔍 Now testing what 'success' looks like...")
    
    # Test multiple times to see success vs failure patterns
    for test_run in range(10):
        seq = dataset.sequences[test_run % 6]
        
        # Before
        orig_tough, _ = reward_fn.predict_toughness(seq)
        
        # Run episode with detailed tracking
        state = env.reset(seq).to(device)
        total_reward = 0
        actual_edits = []
        
        for step in range(20):
            action = policy.get_action(state, deterministic=False)
            old_seq = env.current_sequence
            state, reward, done, info = env.step(action)
            state = state.to(device)
            total_reward += reward
            
            if old_seq != env.current_sequence:
                # Record actual edit
                actual_edits.append({
                    'step': step,
                    'type': action['type'],
                    'pos': action.get('position', 'N/A'),
                    'aa': action.get('amino_acid', 'N/A'),
                    'reward': reward
                })
            
            if done:
                break
        
        # After
        final_tough, _ = reward_fn.predict_toughness(env.current_sequence)
        actual_improvement = final_tough - orig_tough
        
        # Analyze what happened
        print(f"\n--- Test Run {test_run+1} ---")
        print(f"Total reward: {total_reward:.3f}")
        print(f"Actual edits made: {len(actual_edits)}")
        print(f"Original toughness: {orig_tough:.4f}")
        print(f"Final toughness: {final_tough:.4f}")
        print(f"REAL improvement: {actual_improvement:.4f}")
        
        # Check edit history
        total_claimed = sum(edit.get('toughness_improvement', 0) for edit in env.edit_history)
        print(f"Claimed improvement: {total_claimed:.4f}")
        
        if total_reward > 4.0:  # "Success" episode
            print("🎯 SUCCESS EPISODE - What did it do?")
            for edit in actual_edits:
                print(f"  {edit['type']} at pos {edit['pos']} -> {edit['aa']} (reward: {edit['reward']:.3f})")
            
            if actual_improvement < 0.001:
                print("🚨 EXPLOITATION DETECTED: High reward but no real improvement!")
            else:
                print("✅ LEGITIMATE SUCCESS: Real improvement achieved!")
        
        elif total_reward < -5.0:  # Major failure
            print("💥 FAILURE EPISODE")
            print(f"  Made {len(actual_edits)} edits but got penalized heavily")

if __name__ == "__main__":
    debug_success_episodes()