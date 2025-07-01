import torch
import numpy as np
from typing import Dict, List, Any
import random

def test_on_training_sequences(policy, env, dataset, device, n_sequences=10):
    """Test if model works on training sequences to check for overfitting"""
    if len(dataset.train_sequences) < n_sequences:
        train_sequences = dataset.train_sequences
    else:
        train_sequences = random.sample(dataset.train_sequences, n_sequences)
    
    # Use the same evaluate_policy function but with training sequences
    return evaluate_policy(policy, env, train_sequences, device)

def evaluate_policy(policy: torch.nn.Module, env, test_sequences: List[str], 
                   device: torch.device, deterministic: bool = True) -> Dict[str, Any]:
    """Evaluate policy on test sequences - FIXED to match training logic"""
    policy.eval()
    results = []
    
    for seq in test_sequences:
        # Get original toughness
        original_tough, _ = env.reward_fn.predict_toughness(seq)
        
        # Run policy
        state = env.reset(seq).to(device)
        total_reward = 0
        steps = 0
        
        with torch.no_grad():
            while not env.done and steps < env.max_steps:
                action = policy.get_action(state, deterministic=deterministic)
                state, reward, done, _ = env.step(action)
                state = state.to(device)
                total_reward += reward
                steps += 1
        
        # 🚀 FIXED: Use CUMULATIVE improvement from edit_history (same as training)
        cumulative_improvement = sum(
            edit.get('toughness_improvement', 0.0) 
            for edit in env.edit_history
        )
        
        # Also calculate direct toughness difference for comparison
        final_tough, _ = env.reward_fn.predict_toughness(env.current_sequence)
        direct_improvement = final_tough - original_tough
        
        results.append({
            'reward': total_reward,
            'improvement': cumulative_improvement,  # 🚀 Now using cumulative (matches training)
            'direct_improvement': direct_improvement,  # Keep for debugging
            'steps': steps,
            'final_sequence': env.current_sequence,
            'edit_count': len(env.edit_history)
        })
    
    policy.train()
    
    # Aggregate results
    avg_reward = np.mean([r['reward'] for r in results])
    avg_improvement = np.mean([r['improvement'] for r in results])
    avg_direct_improvement = np.mean([r['direct_improvement'] for r in results])
    
    # 🚀 FIXED: Use cumulative improvement for success rate
    success_rate = np.mean([r['improvement'] > 0.0005 for r in results])  # Match training target
    
    # Also calculate success rate for any positive improvement
    any_improvement_rate = np.mean([r['improvement'] > 0 for r in results])
    
    # Find best sequence
    best_idx = np.argmax([r['improvement'] for r in results])
    best_sequence = results[best_idx]['final_sequence']
    
    return {
        'results': results,
        'avg_reward': avg_reward,
        'avg_improvement': avg_improvement,  # Cumulative improvement
        'avg_direct_improvement': avg_direct_improvement,  # Direct SilkomeGPT difference
        'success_rate': success_rate,
        'any_improvement_rate': any_improvement_rate,
        'best_sequence': best_sequence
    }