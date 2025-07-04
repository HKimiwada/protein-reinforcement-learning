#!/usr/bin/env python3
"""
Test if RL can LEARN to improve sequences (with actual training)
"""
import sys; sys.path.append('.')
import torch
from src.config.training_configs import get_config
from src.debug.debug import setup_models_and_environment
from src.training.ppo_trainer import PPOTrainer

def test_rl_learning():
    print("🧪 TESTING RL LEARNING ON 6 SEQUENCES")
    print("="*100)
    
    # Setup
    device = torch.device('cuda')
    config = get_config('phase1').to_dict()
    policy, env, dataset, utils, reward_fn = setup_models_and_environment(config, device)
    
    # Create trainer (this will do actual learning)
    trainer = PPOTrainer(policy, env, lr=1e-4, device=device)
    
    print("📊 Testing before training...")
    
    # Test before training
    before_results = test_all_sequences(trainer.policy, env, dataset, device)
    avg_before = sum(r['improvement'] for r in before_results) / len(before_results)
    print(f"Average improvement before training: {avg_before:.4f}")
    
    print("\n🚀 Training for 100 episodes...")
    
    # Train for 100 episodes
    for episode in range(100):
        # Get random sequence
        seq = dataset.sequences[episode % len(dataset.sequences)]
        
        # Train one episode (this does actual learning!)
        episode_data = trainer.train_episode(seq, episode)
        
        if episode % 10 == 0:
            print(f"Episode {episode}: reward={episode_data['episode_reward']:.3f}")
    
    print("\n📊 Testing after training...")
    
    # Test after training
    after_results = test_all_sequences(trainer.policy, env, dataset, device)
    avg_after = sum(r['improvement'] for r in after_results) / len(after_results)
    print(f"Average improvement after training: {avg_after:.4f}")
    
    # Compare
    print(f"\n🎯 LEARNING RESULTS:")
    print(f"Before training: {avg_before:.4f}")
    print(f"After training:  {avg_after:.4f}")
    print(f"Learning improvement: {avg_after - avg_before:.4f}")
    
    if avg_after > avg_before + 0.001:
        print("✅ Policy learned to improve sequences!")
    else:
        print("❌ No significant learning detected")

def test_all_sequences(policy, env, dataset, device):
    """Test policy on all sequences"""
    results = []
    
    for seq in dataset.sequences:
        # Before
        orig_tough, _ = env.reward_fn.predict_toughness(seq)
        
        # Run policy
        state = env.reset(seq).to(device)
        policy.eval()
        with torch.no_grad():
            for _ in range(10):
                action = policy.get_action(state, deterministic=True)
                state, reward, done, info = env.step(action)
                state = state.to(device)
                if done: break
        
        # After
        final_tough, _ = env.reward_fn.predict_toughness(env.current_sequence)
        improvement = final_tough - orig_tough
        
        results.append({'improvement': improvement})
    
    return results

if __name__ == "__main__":
    test_rl_learning()