#!/usr/bin/env python3
"""
Simple test to see if RL can improve toughness of 6 sequences
Uses a randomly initialized policy and see if random baseline can actually improve toughness.
"""
import sys
import os
sys.path.append('.')

import torch
import pandas as pd

def test_rl_on_six_sequences():
    """Test if RL can improve your 6 sequences"""
    print("🧪 TESTING RL ON 6 SEQUENCES")
    print("="*50)
    
    # Import after path setup
    from src.config.training_configs import get_config
    from src.debug.debug import setup_models_and_environment
    
    # Setup
    device = torch.device('cuda')
    config = get_config('phase1').to_dict()
    
    # Load models
    setup_result = setup_models_and_environment(config, device)
    if setup_result is None:
        print("❌ Failed to setup models")
        return
    
    policy, env, dataset, utils, reward_fn = setup_result
    
    # Load your 6 sequences
    df = pd.read_csv('src/data/raw/Testing_Sequences.csv')
    sequences = df['sequence'].tolist()
    original_toughness = df['toughness'].tolist()
    
    print(f"📊 Testing {len(sequences)} sequences...")
    
    results = []
    
    for i, (seq, orig_tough) in enumerate(zip(sequences, original_toughness)):
        print(f"\n🔬 SEQUENCE {i+1}")
        print(f"Original: {seq[:50]}...")
        print(f"Original toughness: {orig_tough:.4f}")
        
        # Get SilkomeGPT prediction for original
        pred_orig, _ = reward_fn.predict_toughness(seq)
        print(f"SilkomeGPT original: {pred_orig:.4f}")
        
        # Reset environment and run RL agent
        state = env.reset(seq).to(device)
        total_reward = 0
        edits_made = []
        
        print(f"🤖 Running RL agent...")
        
        step = 0
        while not env.done and step < 20:  # Max 20 steps
            # Get action from policy
            action = policy.get_action(state, deterministic=False)
            
            # Take step
            old_seq = env.current_sequence
            state, reward, done, info = env.step(action)
            state = state.to(device)
            total_reward += reward
            
            if old_seq != env.current_sequence:
                edits_made.append({
                    'step': step,
                    'action': action['type'],
                    'position': action.get('position', 'N/A'),
                    'amino_acid': action.get('amino_acid', 'N/A'),
                    'reward': reward
                })
                print(f"  Step {step}: {action['type']} at pos {action.get('position', 'N/A')} -> reward {reward:.3f}")
            
            step += 1
        
        # Final results
        final_seq = env.current_sequence
        pred_final, _ = reward_fn.predict_toughness(final_seq)
        
        # Calculate improvements
        silkomegpt_improvement = pred_final - pred_orig
        cumulative_claimed = sum(edit.get('toughness_improvement', 0) for edit in env.edit_history)
        
        print(f"\n📊 RESULTS:")
        print(f"  Edits made: {len(edits_made)}")
        print(f"  Total reward: {total_reward:.3f}")
        print(f"  Final sequence: {final_seq[:50]}...")
        print(f"  SilkomeGPT improvement: {silkomegpt_improvement:.4f}")
        print(f"  Cumulative claimed: {cumulative_claimed:.4f}")
        print(f"  Success: {'✅' if silkomegpt_improvement > 0.005 else '❌'}")
        
        results.append({
            'sequence_id': i+1,
            'original_toughness': orig_tough,
            'silkomegpt_original': pred_orig,
            'silkomegpt_final': pred_final,
            'silkomegpt_improvement': silkomegpt_improvement,
            'cumulative_claimed': cumulative_claimed,
            'total_reward': total_reward,
            'edits_made': len(edits_made),
            'success': silkomegpt_improvement > 0.005
        })
    
    # Summary
    print(f"\n🎯 FINAL SUMMARY")
    print("="*50)
    
    successes = sum(1 for r in results if r['success'])
    avg_improvement = sum(r['silkomegpt_improvement'] for r in results) / len(results)
    best_improvement = max(r['silkomegpt_improvement'] for r in results)
    
    print(f"Sequences improved: {successes}/{len(results)}")
    print(f"Average improvement: {avg_improvement:.4f}")
    print(f"Best improvement: {best_improvement:.4f}")
    print(f"Total edits across all sequences: {sum(r['edits_made'] for r in results)}")
    
    if successes > 0:
        print(f"🎉 SUCCESS! RL improved {successes} sequence(s)")
        best_seq = max(results, key=lambda x: x['silkomegpt_improvement'])
        print(f"Best result: Sequence {best_seq['sequence_id']} improved by {best_seq['silkomegpt_improvement']:.4f}")
    else:
        print(f"❌ No sequences significantly improved")
        print(f"Note: Improvements > 0.005 considered significant")
    
    return results

if __name__ == "__main__":
    results = test_rl_on_six_sequences()