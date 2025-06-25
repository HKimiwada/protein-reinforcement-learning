import torch
import numpy as np
from typing import Dict, List, Any

def evaluate_policy(policy: torch.nn.Module, env, test_sequences: List[str], 
                   device: torch.device, deterministic: bool = True) -> Dict[str, Any]:
    """Evaluate policy on test sequences"""
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
        
        # Get final toughness
        final_tough, _ = env.reward_fn.predict_toughness(env.current_sequence)
        improvement = final_tough - original_tough
        
        results.append({
            'reward': total_reward,
            'improvement': improvement,
            'steps': steps,
            'final_sequence': env.current_sequence
        })
    
    policy.train()
    
    # Aggregate results
    avg_reward = np.mean([r['reward'] for r in results])
    avg_improvement = np.mean([r['improvement'] for r in results])
    success_rate = np.mean([r['improvement'] > 0 for r in results])
    
    # Find best sequence
    best_idx = np.argmax([r['improvement'] for r in results])
    best_sequence = results[best_idx]['final_sequence']
    
    return {
        'results': results,
        'avg_reward': avg_reward,
        'avg_improvement': avg_improvement,
        'success_rate': success_rate,
        'best_sequence': best_sequence
    }