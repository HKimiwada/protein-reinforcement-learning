#!/usr/bin/env python3
"""
src/debug.py - Debug utilities for spider silk RL training

This module provides debugging functions to diagnose training issues,
reward function problems, and model behavior.

Usage:
    from src.debug import debug_episode_details, diagnose_training_issues
    
    # In your training loop:
    if episode == 80:
        diagnose_training_issues(trainer, env, dataset, device)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


def debug_episode_details(env, policy, sequence: str, device: torch.device, 
                         max_steps: int = 10, verbose: bool = True) -> Dict[str, Any]:
    """
    Debug what the agent is actually doing during an episode
    
    Args:
        env: Environment instance
        policy: Policy network
        sequence: Starting protein sequence
        device: Device for computation
        max_steps: Maximum steps to debug (default 10)
        verbose: Whether to print detailed output
        
    Returns:
        Dictionary with episode statistics
    """
    if verbose:
        print(f"\n=== DEBUGGING EPISODE ===")
        print(f"Starting sequence: {sequence[:50]}...")
    
    # Get original toughness
    try:
        original_tough, original_std = env.reward_fn.predict_toughness(sequence)
        if verbose:
            print(f"Original toughness: {original_tough:.4f} ± {original_std:.4f}")
    except Exception as e:
        if verbose:
            print(f"Error getting original toughness: {e}")
        original_tough = 0.0
    
    # Reset environment
    state = env.reset(sequence).to(device)
    total_reward = 0
    step = 0
    actions_taken = []
    rewards_received = []
    improvements_claimed = []
    
    policy.eval()
    with torch.no_grad():
        while not env.done and step < max_steps:
            # Get policy output
            try:
                policy_output = policy(state.unsqueeze(0))
                action = policy.get_action(state, deterministic=False)
                
                if verbose:
                    print(f"Step {step}: Action = {action['type']}", end="")
                    
                    if action['type'] != 'stop':
                        print(f" at pos {action['position']}", end="")
                        if action['amino_acid']:
                            print(f" with {action['amino_acid']}")
                        else:
                            print()
                    else:
                        print()
                
                actions_taken.append(action['type'])
                
            except Exception as e:
                if verbose:
                    print(f"Error in action selection: {e}")
                break
            
            # Take environment step
            try:
                old_sequence = env.current_sequence
                state, reward, done, info = env.step(action)
                state = state.to(device)
                total_reward += reward
                rewards_received.append(reward)
                
                if verbose:
                    print(f"  Reward: {reward:.3f}, Total: {total_reward:.3f}")
                    print(f"  Sequence length: {len(env.current_sequence)}")
                
                # Check for toughness improvement
                if info.get('edit_info', {}).get('toughness_improvement'):
                    imp = info['edit_info']['toughness_improvement']
                    improvements_claimed.append(imp)
                    if verbose:
                        print(f"  Claimed toughness improvement: {imp:.4f}")
                else:
                    improvements_claimed.append(0.0)
                
                # Check if sequence actually changed
                if old_sequence != env.current_sequence and verbose:
                    print(f"  Sequence changed: {old_sequence != env.current_sequence}")
                
            except Exception as e:
                if verbose:
                    print(f"Error in environment step: {e}")
                break
            
            step += 1
    
    # Get final toughness
    try:
        final_tough, final_std = env.reward_fn.predict_toughness(env.current_sequence)
        actual_improvement = final_tough - original_tough
        if verbose:
            print(f"Final toughness: {final_tough:.4f} ± {final_std:.4f}")
            print(f"Actual improvement: {actual_improvement:.4f}")
    except Exception as e:
        if verbose:
            print(f"Error getting final toughness: {e}")
        final_tough = original_tough
        actual_improvement = 0.0
    
    total_claimed_improvement = sum(improvements_claimed)
    
    if verbose:
        print(f"Total reward: {total_reward:.3f}")
        print(f"Total claimed improvement: {total_claimed_improvement:.4f}")
        print(f"Edit history length: {len(env.edit_history)}")
        
        # Check for exploitation
        if total_claimed_improvement > 0.01 and actual_improvement < 0.001:
            print("🚨 POTENTIAL EXPLOITATION DETECTED!")
        elif actual_improvement > 0.001:
            print("✓ Real improvement achieved")
        
        print("=== END DEBUG ===\n")
    
    return {
        'original_toughness': original_tough,
        'final_toughness': final_tough,
        'actual_improvement': actual_improvement,
        'total_claimed_improvement': total_claimed_improvement,
        'total_reward': total_reward,
        'steps': step,
        'actions_taken': actions_taken,
        'rewards_received': rewards_received,
        'improvements_claimed': improvements_claimed,
        'final_sequence': env.current_sequence,
        'edit_history': env.edit_history.copy()
    }


def test_silkomegpt_predictions(reward_fn, verbose: bool = True) -> bool:
    """
    Test if SilkomeGPT is making reasonable predictions
    
    Args:
        reward_fn: Reward function instance
        verbose: Whether to print detailed output
        
    Returns:
        True if SilkomeGPT appears to be working correctly
    """
    if verbose:
        print("\n=== TESTING SILKOMEGPT ===")
    
    # Test sequences with expected different toughness
    test_sequences = [
        ("Simple spider silk", "GPGGQGPYGPGGQGPGGQGPYGPQAAAAAAAAAAAGPGGQGPYGPGGQ"),
        ("Longer spider silk", "GPGGQGPYGPGGQGPGGQGPYGPQAAAAAAAAAAAGPGGQGPYGPGGQGPGGQGPYGPGGQ"),
        ("All amino acids", "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY"),
        ("All glycine", "GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG"),
        ("All alanine", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"),
        ("Mixed sequence", "GPGGAAGGPGGYYGPGGSSGPGGKKGPGGPPGPGGLLGPGG")
    ]
    
    predictions = []
    for name, seq in test_sequences:
        try:
            tough, std = reward_fn.predict_toughness(seq)
            predictions.append((name, tough, std))
            if verbose:
                print(f"{name:20s}: toughness={tough:.4f}, std={std:.4f}")
        except Exception as e:
            if verbose:
                print(f"{name:20s}: ERROR - {e}")
            predictions.append((name, None, None))
    
    # Check if predictions are reasonable
    valid_predictions = [(n, t, s) for n, t, s in predictions if t is not None]
    if len(valid_predictions) < 3:
        if verbose:
            print("🚨 SilkomeGPT is failing on most sequences!")
        return False
    
    toughness_values = [t for _, t, _ in valid_predictions]
    toughness_range = max(toughness_values) - min(toughness_values)
    
    if verbose:
        print(f"Toughness range: {toughness_range:.4f}")
        print(f"Average toughness: {np.mean(toughness_values):.4f}")
    
    if toughness_range < 0.001:
        if verbose:
            print("🚨 SilkomeGPT predictions have no variance!")
        return False
    
    # Check for reasonable values (spider silk toughness should be 0.005-0.39 range)
    reasonable_values = [t for t in toughness_values if 0.001 <= t <= 1.0]
    if len(reasonable_values) < len(toughness_values) * 0.5:
        if verbose:
            print("🚨 Many SilkomeGPT predictions are out of reasonable range!")
        return False
    
    if verbose:
        print("✓ SilkomeGPT appears to be working")
    return True


def test_improvement_detection(reward_fn, verbose: bool = True) -> Dict[str, float]:
    """
    Test if small changes in sequences are detected by SilkomeGPT
    
    Args:
        reward_fn: Reward function instance
        verbose: Whether to print detailed output
        
    Returns:
        Dictionary with improvement statistics
    """
    if verbose:
        print("\n=== TESTING IMPROVEMENT DETECTION ===")
    
    base_seq = "GPGGQGPYGPGGQGPGGQGPYGPQAAAAAAAAAAAGPGGQGPYGPGGQ"
    
    # Make small changes
    variants = [
        ("Original", base_seq),
        ("G->A substitution", base_seq.replace('G', 'A', 1)),
        ("P->S substitution", base_seq.replace('P', 'S', 1)),
        ("Q->K substitution", base_seq.replace('Q', 'K', 1)),
        ("Add GPG motif", base_seq + 'GPG'),
        ("Remove 3 chars", base_seq[:-3]),
        ("Insert A", base_seq[:20] + 'A' + base_seq[20:]),
        ("More glycine", base_seq.replace('A', 'G', 2))
    ]
    
    try:
        base_tough, _ = reward_fn.predict_toughness(base_seq)
        if verbose:
            print(f"Base toughness: {base_tough:.4f}")
    except Exception as e:
        if verbose:
            print(f"Error getting base toughness: {e}")
        return {}
    
    improvements = []
    for name, variant in variants[1:]:  # Skip original
        try:
            var_tough, _ = reward_fn.predict_toughness(variant)
            improvement = var_tough - base_tough
            improvements.append(improvement)
            if verbose:
                print(f"{name:20s}: {improvement:+.4f} change")
        except Exception as e:
            if verbose:
                print(f"{name:20s}: ERROR - {e}")
            improvements.append(0.0)
    
    # Calculate statistics
    stats = {
        'mean_improvement': np.mean(improvements),
        'std_improvement': np.std(improvements),
        'max_improvement': np.max(improvements),
        'min_improvement': np.min(improvements),
        'num_positive': sum(1 for x in improvements if x > 0),
        'num_negative': sum(1 for x in improvements if x < 0)
    }
    
    if verbose:
        print(f"Improvement statistics:")
        print(f"  Mean: {stats['mean_improvement']:+.4f}")
        print(f"  Std:  {stats['std_improvement']:.4f}")
        print(f"  Range: {stats['min_improvement']:+.4f} to {stats['max_improvement']:+.4f}")
        print(f"  Positive changes: {stats['num_positive']}/{len(improvements)}")
    
    return stats


def analyze_policy_behavior(policy, env, sequence: str, device: torch.device, 
                          verbose: bool = True) -> Dict[str, Any]:
    """
    Analyze policy behavior and outputs
    
    Args:
        policy: Policy network
        env: Environment instance
        sequence: Test sequence
        device: Computation device
        verbose: Whether to print detailed output
        
    Returns:
        Dictionary with policy analysis
    """
    if verbose:
        print("\n=== POLICY BEHAVIOR ANALYSIS ===")
    
    state = env.reset(sequence).to(device)
    
    policy.eval()
    with torch.no_grad():
        # Get policy output
        output = policy(state.unsqueeze(0))
        
        # Analyze edit type distribution
        edit_probs = output['edit_type'][0].cpu().numpy()
        edit_types = ['substitution', 'insertion', 'deletion', 'stop']
        
        if verbose:
            print("Edit type probabilities:")
            for edit_type, prob in zip(edit_types, edit_probs):
                print(f"  {edit_type:12s}: {prob:.3f}")
        
        # Calculate entropy
        edit_entropy = -(output['edit_type'] * torch.log(output['edit_type'] + 1e-8)).sum().item()
        pos_entropy = -(output['position'] * torch.log(output['position'] + 1e-8)).sum().item()
        aa_entropy = -(output['amino_acid'] * torch.log(output['amino_acid'] + 1e-8)).sum().item()
        
        if verbose:
            print(f"Entropy analysis:")
            print(f"  Edit type entropy: {edit_entropy:.3f}")
            print(f"  Position entropy:  {pos_entropy:.3f}")
            print(f"  Amino acid entropy: {aa_entropy:.3f}")
        
        # Value estimate
        value = output['value'][0].item()
        if verbose:
            print(f"Value estimate: {value:.3f}")
        
        # Check for pathological behaviors
        issues = []
        if edit_entropy < 0.1:
            issues.append("Policy too deterministic (poor exploration)")
        elif edit_entropy > 1.0:
            issues.append("Policy too random (poor learning)")
        
        if edit_probs[3] > 0.9:  # Stop action
            issues.append("Policy always wants to stop")
        
        if np.max(edit_probs[:3]) > 0.95:  # One edit type dominates
            issues.append("Policy overly biased to one edit type")
        
        if abs(value) > 10:
            issues.append("Value estimates are extreme")
        
        if verbose:
            if issues:
                print("⚠️  Issues detected:")
                for issue in issues:
                    print(f"    - {issue}")
            else:
                print("✓ Policy behavior looks reasonable")
    
    return {
        'edit_probabilities': edit_probs,
        'edit_entropy': edit_entropy,
        'position_entropy': pos_entropy,
        'amino_acid_entropy': aa_entropy,
        'value_estimate': value,
        'issues': issues
    }


def check_reward_components(env, old_seq: str, new_seq: str, 
                          verbose: bool = True) -> Dict[str, float]:
    """
    Analyze individual reward components
    
    Args:
        env: Environment instance
        old_seq: Original sequence
        new_seq: Modified sequence
        verbose: Whether to print detailed output
        
    Returns:
        Dictionary with reward component analysis
    """
    if verbose:
        print(f"\n=== REWARD COMPONENT ANALYSIS ===")
    
    try:
        # Individual components
        r_tough = env.reward_fn.toughness_reward(old_seq, new_seq)
        r_real = env.reward_fn.realism_reward(new_seq, old_seq)
        r_explore = env.reward_fn.exploration_reward(env.edit_history, 50)
        r_eff = env.reward_fn.efficiency_reward(len(env.edit_history), 0.01, 0.02)
        
        if verbose:
            print(f"Reward components:")
            print(f"  Toughness:  {r_tough:.3f}")
            print(f"  Realism:    {r_real:.3f}")
            print(f"  Exploration:{r_explore:.3f}")
            print(f"  Efficiency: {r_eff:.3f}")
        
        # Check for exploitation
        total_imp = sum(e.get('toughness_improvement', 0) for e in env.edit_history)
        if verbose:
            print(f"  Total claimed improvement: {total_imp:.4f}")
        
        # Actual improvement
        old_t, _ = env.reward_fn.predict_toughness(old_seq)
        new_t, _ = env.reward_fn.predict_toughness(new_seq)
        actual_imp = new_t - old_t
        
        if verbose:
            print(f"  Actual improvement: {actual_imp:.4f}")
        
        exploitation_detected = False
        if total_imp > 0.01 and actual_imp < 0.001:
            exploitation_detected = True
            if verbose:
                print("  🚨 EXPLOITATION DETECTED!")
        elif actual_imp > 0.001:
            if verbose:
                print("  ✓ Real improvement detected")
        
        return {
            'toughness': r_tough,
            'realism': r_real,
            'exploration': r_explore,
            'efficiency': r_eff,
            'total_claimed': total_imp,
            'actual_improvement': actual_imp,
            'exploitation_detected': exploitation_detected
        }
        
    except Exception as e:
        if verbose:
            print(f"Error in reward analysis: {e}")
        return {}


def diagnose_training_issues(trainer, env, dataset, device: torch.device, 
                           n_test_sequences: int = 3) -> Dict[str, Any]:
    """
    Comprehensive diagnosis of training issues
    
    Args:
        trainer: PPO trainer instance
        env: Environment instance
        dataset: Dataset instance
        device: Computation device
        n_test_sequences: Number of sequences to test
        
    Returns:
        Dictionary with comprehensive diagnostics
    """
    print("\n" + "="*60)
    print("COMPREHENSIVE TRAINING DIAGNOSIS")
    print("="*60)
    
    diagnostics = {}
    
    # 1. Test SilkomeGPT
    print("\n1. Testing SilkomeGPT...")
    silkomegpt_working = test_silkomegpt_predictions(env.reward_fn)
    diagnostics['silkomegpt_working'] = silkomegpt_working
    
    if not silkomegpt_working:
        print("❌ SilkomeGPT has issues - this explains poor performance!")
        return diagnostics
    
    # 2. Test improvement detection
    print("\n2. Testing improvement detection...")
    improvement_stats = test_improvement_detection(env.reward_fn)
    diagnostics['improvement_stats'] = improvement_stats
    
    # 3. Test policy behavior
    print("\n3. Analyzing policy behavior...")
    test_seq = dataset.get_test_sequences(1)[0]
    policy_analysis = analyze_policy_behavior(trainer.policy, env, test_seq, device)
    diagnostics['policy_analysis'] = policy_analysis
    
    # 4. Debug actual episodes
    print("\n4. Debugging episodes...")
    test_sequences = dataset.get_test_sequences(n_test_sequences)
    episode_results = []
    
    for i, seq in enumerate(test_sequences):
        print(f"\n--- Episode {i+1} ---")
        result = debug_episode_details(env, trainer.policy, seq, device)
        episode_results.append(result)
        
        # Check reward components for this episode
        if len(env.edit_history) > 0:
            reward_analysis = check_reward_components(env, seq, env.current_sequence)
            result['reward_analysis'] = reward_analysis
    
    diagnostics['episode_results'] = episode_results
    
    # 5. Training statistics summary
    print(f"\n5. Training statistics...")
    if hasattr(trainer, 'episode_rewards') and trainer.episode_rewards:
        print(f"  Episodes completed: {len(trainer.episode_rewards)}")
        print(f"  Average reward: {np.mean(trainer.episode_rewards):.3f}")
        print(f"  Recent average (last 20): {np.mean(trainer.episode_rewards[-20:]):.3f}")
        print(f"  Best reward: {np.max(trainer.episode_rewards):.3f}")
        print(f"  Reward std: {np.std(trainer.episode_rewards):.3f}")
        
        diagnostics['training_stats'] = {
            'episodes_completed': len(trainer.episode_rewards),
            'average_reward': np.mean(trainer.episode_rewards),
            'recent_average': np.mean(trainer.episode_rewards[-20:]),
            'best_reward': np.max(trainer.episode_rewards),
            'reward_std': np.std(trainer.episode_rewards)
        }
    
    # 6. Dataset verification
    print(f"\n6. Dataset verification...")
    print(f"  Total sequences: {len(dataset.sequences)}")
    print(f"  Train sequences: {len(dataset.train_sequences)}")
    print(f"  Test sequences: {len(dataset.test_sequences)}")
    
    if dataset.toughness_values:
        print(f"  Toughness range: {min(dataset.toughness_values):.4f} - {max(dataset.toughness_values):.4f}")
        print(f"  Average toughness: {np.mean(dataset.toughness_values):.4f}")
    
    # Test curriculum sampling
    sample_difficulties = []
    for _ in range(10):
        seq, level = dataset.get_curriculum_sequence(50, 200, 'mixed')
        sample_difficulties.append(level)
    
    print(f"  Sample difficulty levels: {sample_difficulties}")
    print(f"  Average sampled difficulty: {np.mean(sample_difficulties):.1f}")
    
    diagnostics['dataset_stats'] = {
        'total_sequences': len(dataset.sequences),
        'train_sequences': len(dataset.train_sequences),
        'test_sequences': len(dataset.test_sequences),
        'sample_difficulties': sample_difficulties
    }
    
    # 7. Summary and recommendations
    print(f"\n" + "="*60)
    print("SUMMARY AND RECOMMENDATIONS")
    print("="*60)
    
    # Detect major issues
    major_issues = []
    if not silkomegpt_working:
        major_issues.append("SilkomeGPT predictions are broken")
    
    if improvement_stats and improvement_stats.get('std_improvement', 0) < 0.001:
        major_issues.append("SilkomeGPT shows no sensitivity to sequence changes")
    
    exploit_count = sum(1 for r in episode_results 
                       if r.get('reward_analysis', {}).get('exploitation_detected', False))
    if exploit_count > len(episode_results) * 0.5:
        major_issues.append("Agent is exploiting the reward function")
    
    if policy_analysis.get('issues'):
        major_issues.extend(policy_analysis['issues'])
    
    if major_issues:
        print("❌ MAJOR ISSUES DETECTED:")
        for issue in major_issues:
            print(f"   - {issue}")
    else:
        print("✅ No major issues detected - training should be working")
    
    # Specific recommendations
    print(f"\nRECOMMENDATIONS:")
    
    if not silkomegpt_working:
        print("  1. Check SilkomeGPT model loading and tokenizer setup")
        print("  2. Verify model weights and configuration files")
    
    if exploit_count > 0:
        print("  3. Increase anti-exploitation measures in reward function")
        print("  4. Reduce learning rate to prevent rapid exploitation learning")
    
    if policy_analysis.get('edit_entropy', 1.0) < 0.1:
        print("  5. Increase entropy regularization coefficient")
        print("  6. Add exploration bonuses")
    
    avg_actual_improvement = np.mean([r.get('actual_improvement', 0) 
                                    for r in episode_results])
    if avg_actual_improvement < 0.001:
        print("  7. Consider easier curriculum or longer training")
        print("  8. Check if sequences in dataset are actually improvable")
    
    diagnostics['major_issues'] = major_issues
    diagnostics['average_actual_improvement'] = avg_actual_improvement
    
    print(f"\n" + "="*60)
    
    return diagnostics


def save_debug_report(diagnostics: Dict[str, Any], filepath: str = "debug_report.txt"):
    """Save debugging results to a file"""
    with open(filepath, 'w') as f:
        f.write("SPIDER SILK RL TRAINING DEBUG REPORT\n")
        f.write("="*50 + "\n\n")
        
        f.write("SUMMARY:\n")
        f.write(f"SilkomeGPT Working: {diagnostics.get('silkomegpt_working', 'Unknown')}\n")
        f.write(f"Major Issues: {len(diagnostics.get('major_issues', []))}\n")
        f.write(f"Average Actual Improvement: {diagnostics.get('average_actual_improvement', 0):.4f}\n\n")
        
        if 'major_issues' in diagnostics:
            f.write("MAJOR ISSUES:\n")
            for issue in diagnostics['major_issues']:
                f.write(f"  - {issue}\n")
            f.write("\n")
        
        if 'training_stats' in diagnostics:
            stats = diagnostics['training_stats']
            f.write("TRAINING STATISTICS:\n")
            f.write(f"  Episodes: {stats['episodes_completed']}\n")
            f.write(f"  Average Reward: {stats['average_reward']:.3f}\n")
            f.write(f"  Best Reward: {stats['best_reward']:.3f}\n")
            f.write(f"  Reward Std: {stats['reward_std']:.3f}\n\n")
        
        # Add more details as needed
        f.write("Full diagnostics available in the returned dictionary.\n")
    
    print(f"Debug report saved to {filepath}")

# Add this test to your debug_tools.py to check SilkomeGPT consistency

def test_silkomegpt_consistency(reward_fn, verbose=True):
    """Test if SilkomeGPT gives consistent predictions for the same sequence"""
    if verbose:
        print("\n=== TESTING SILKOMEGPT CONSISTENCY ===")
    
    test_seq = "AAAAGGAGGQGGYGGLGSQGAGQGGYGAGQGAGAAAAAAAAGGAGGQGGR"
    
    # Test same sequence multiple times
    predictions = []
    for i in range(5):
        tough, std = reward_fn.predict_toughness(test_seq)
        predictions.append((tough, std))
        if verbose:
            print(f"Prediction {i+1}: toughness={tough:.4f}, std={std:.4f}")
    
    # Check consistency
    toughness_values = [p[0] for p in predictions]
    std_values = [p[1] for p in predictions]
    
    toughness_std = np.std(toughness_values)
    std_std = np.std(std_values)
    
    if verbose:
        print(f"Toughness consistency (std): {toughness_std:.6f}")
        print(f"Std consistency (std): {std_std:.6f}")
    
    if toughness_std > 0.001:
        if verbose:
            print("🚨 SilkomeGPT predictions are inconsistent!")
        return False
    
    # Test if predictions change with sequence modifications
    modified_seq = test_seq.replace('A', 'G', 1)  # Change one A to G
    
    orig_tough, _ = reward_fn.predict_toughness(test_seq)
    mod_tough, _ = reward_fn.predict_toughness(modified_seq)
    
    change = abs(mod_tough - orig_tough)
    if verbose:
        print(f"Original: {orig_tough:.4f}")
        print(f"Modified: {mod_tough:.4f}")
        print(f"Change magnitude: {change:.4f}")
    
    if change < 0.0001:
        if verbose:
            print("🚨 SilkomeGPT doesn't respond to sequence changes!")
        return False
    
    if verbose:
        print("✓ SilkomeGPT consistency looks good")
    
    return True

def test_reward_calculation_bug(env, verbose=True):
    """Test if reward calculation has bugs"""
    if verbose:
        print("\n=== TESTING REWARD CALCULATION ===")
    
    # Use a simple sequence
    test_seq = "GPGGQGPYGPGGQGPGGQGPYGPQAAAAAAAAAAAGPGGQGPYGPGGQ"
    
    # Reset environment
    env.reset(test_seq)
    original_seq = env.current_sequence
    
    # Get original toughness
    orig_tough, _ = env.reward_fn.predict_toughness(original_seq)
    if verbose:
        print(f"Original toughness: {orig_tough:.4f}")
    
    # Make a simple substitution
    action = {
        'type': 'substitution',
        'position': 0,
        'amino_acid': 'A',
        'log_prob': torch.tensor(0.0)
    }
    
    state, reward, done, info = env.step(action)
    new_seq = env.current_sequence
    
    # Get new toughness
    new_tough, _ = env.reward_fn.predict_toughness(new_seq)
    actual_improvement = new_tough - orig_tough
    
    if verbose:
        print(f"New toughness: {new_tough:.4f}")
        print(f"Actual improvement: {actual_improvement:.4f}")
        print(f"Reward received: {reward:.3f}")
        
        if 'edit_info' in info and 'toughness_improvement' in info['edit_info']:
            claimed_imp = info['edit_info']['toughness_improvement']
            print(f"Claimed improvement: {claimed_imp:.4f}")
            
            if abs(claimed_imp - actual_improvement) > 0.001:
                print("🚨 CLAIMED vs ACTUAL improvement mismatch!")
                return False
    
    if verbose:
        print("✓ Reward calculation looks consistent")
    
    return True

# Add this to your comprehensive debugging function
def debug_specific_issues(policy, env, dataset, device):
    """Debug the specific issues we found"""
    print("\n🔬 DEBUGGING SPECIFIC ISSUES")
    print("="*50)
    
    # Test 1: SilkomeGPT consistency
    silkomegpt_consistent = test_silkomegpt_consistency(env.reward_fn)
    
    # Test 2: Reward calculation
    reward_calc_ok = test_reward_calculation_bug(env)
    
    # Test 3: Check if policy is actually learning
    print("\n=== POLICY LEARNING TEST ===")
    
    # Create two identical states
    test_seq = dataset.get_test_sequences(1)[0]
    state1 = env.reset(test_seq).to(device)
    state2 = env.reset(test_seq).to(device)
    
    policy.eval()
    with torch.no_grad():
        output1 = policy(state1.unsqueeze(0))
        output2 = policy(state2.unsqueeze(0))
        
        # Check if outputs are identical for identical inputs
        prob_diff = torch.abs(output1['edit_type'] - output2['edit_type']).max().item()
        value_diff = torch.abs(output1['value'] - output2['value']).item()
        
        print(f"Policy output consistency:")
        print(f"  Max probability difference: {prob_diff:.6f}")
        print(f"  Value difference: {value_diff:.6f}")
        
        if prob_diff > 0.001 or value_diff > 0.001:
            print("🚨 Policy outputs are inconsistent for same input!")
        else:
            print("✓ Policy is deterministic for same input")
    
    return {
        'silkomegpt_consistent': silkomegpt_consistent,
        'reward_calc_ok': reward_calc_ok
    }

# Add this function to your debug_tools.py file

def test_step_by_step_toughness(env, verbose=True):
    """Test toughness predictions step by step during environment execution"""
    if verbose:
        print("\n=== TESTING STEP-BY-STEP TOUGHNESS ===")
    
    # Use a simple sequence
    test_seq = "GPGGQGPYGPGGQGPGGQGPYGPQAAAAAAAAAAAGPGGQGPYGPGGQ"
    
    # Reset environment and get initial prediction
    env.reset(test_seq)
    initial_tough, _ = env.reward_fn.predict_toughness(test_seq)
    
    if verbose:
        print(f"Initial sequence: {test_seq[:30]}...")
        print(f"Initial toughness: {initial_tough:.4f}")
    
    # Make a substitution action
    action = {
        'type': 'substitution',
        'position': 0,  # Change first character
        'amino_acid': 'A',
        'log_prob': torch.tensor(0.0)
    }
    
    # Capture sequence before and after
    seq_before = env.current_sequence
    tough_before, _ = env.reward_fn.predict_toughness(seq_before)
    
    if verbose:
        print(f"\nBefore action:")
        print(f"  Sequence: {seq_before[:30]}...")
        print(f"  Toughness: {tough_before:.4f}")
    
    # Execute the action
    state, reward, done, info = env.step(action)
    seq_after = env.current_sequence
    tough_after, _ = env.reward_fn.predict_toughness(seq_after)
    
    if verbose:
        print(f"\nAfter action:")
        print(f"  Sequence: {seq_after[:30]}...")
        print(f"  Toughness: {tough_after:.4f}")
        print(f"  Actual change: {tough_after - tough_before:.4f}")
        print(f"  Reward received: {reward:.3f}")
    
    # Check edit info
    if 'edit_info' in info:
        edit_info = info['edit_info']
        if verbose:
            print(f"\nEdit info:")
            for key, value in edit_info.items():
                print(f"  {key}: {value}")
        
        # Check if claimed improvement matches actual
        if 'toughness_improvement' in edit_info:
            claimed = edit_info['toughness_improvement']
            actual = tough_after - tough_before
            diff = abs(claimed - actual)
            
            if verbose:
                print(f"\nImprovement comparison:")
                print(f"  Claimed: {claimed:.4f}")
                print(f"  Actual:  {actual:.4f}")
                print(f"  Difference: {diff:.4f}")
            
            if diff > 0.001:
                if verbose:
                    print("🚨 MAJOR DISCREPANCY between claimed and actual improvement!")
                return False
    
    # Test if sequences are actually different
    if seq_before == seq_after:
        if verbose:
            print("🚨 Sequence didn't change despite successful action!")
        return False
    
    # Test toughness prediction consistency
    tough_recheck, _ = env.reward_fn.predict_toughness(seq_after)
    if abs(tough_after - tough_recheck) > 0.001:
        if verbose:
            print(f"🚨 Toughness prediction inconsistent: {tough_after:.4f} vs {tough_recheck:.4f}")
        return False
    
    if verbose:
        print("✓ Step-by-step toughness tracking looks consistent")
    
    return True


def test_multiple_sequences_toughness(env, dataset, verbose=True):
    """Test if toughness predictions vary across different sequences"""
    if verbose:
        print("\n=== TESTING TOUGHNESS VARIATION ACROSS SEQUENCES ===")
    
    # Get several test sequences
    test_sequences = dataset.get_test_sequences(5)
    predictions = []
    
    for i, seq in enumerate(test_sequences):
        tough, std = env.reward_fn.predict_toughness(seq)
        predictions.append(tough)
        
        if verbose:
            print(f"Seq {i+1} (len={len(seq):3d}): {tough:.4f} ± {std:.4f}")
    
    # Check variance
    toughness_std = np.std(predictions)
    toughness_range = max(predictions) - min(predictions)
    
    if verbose:
        print(f"\nVariation statistics:")
        print(f"  Standard deviation: {toughness_std:.4f}")
        print(f"  Range: {toughness_range:.4f}")
        print(f"  Mean: {np.mean(predictions):.4f}")
    
    if toughness_std < 0.001:
        if verbose:
            print("🚨 Very low variance - all sequences getting similar predictions!")
        return False
    
    if toughness_range < 0.005:
        if verbose:
            print("🚨 Very small range - predictions not varying much!")
        return False
    
    if verbose:
        print("✓ Good toughness variation across different sequences")
    
    return True


# Update the debug_specific_issues function to include these new tests
def debug_specific_issues_enhanced(policy, env, dataset, device):
    """Enhanced debugging of specific issues we found"""
    print("\n🔬 DEBUGGING SPECIFIC ISSUES (ENHANCED)")
    print("="*50)
    
    # Test 1: SilkomeGPT consistency
    print("\n1. Testing SilkomeGPT consistency...")
    silkomegpt_consistent = test_silkomegpt_consistency(env.reward_fn)
    
    # Test 2: Reward calculation
    print("\n2. Testing reward calculation...")
    reward_calc_ok = test_reward_calculation_bug(env)
    
    # Test 3: Step-by-step toughness tracking
    print("\n3. Testing step-by-step toughness...")
    step_by_step_ok = test_step_by_step_toughness(env)
    
    # Test 4: Toughness variation across sequences
    print("\n4. Testing toughness variation...")
    toughness_variation_ok = test_multiple_sequences_toughness(env, dataset)
    
    # Test 5: Policy learning check
    print("\n5. Testing policy learning...")
    test_seq = dataset.get_test_sequences(1)[0]
    state1 = env.reset(test_seq).to(device)
    state2 = env.reset(test_seq).to(device)
    
    policy.eval()
    with torch.no_grad():
        output1 = policy(state1.unsqueeze(0))
        output2 = policy(state2.unsqueeze(0))
        
        # Check if outputs are identical for identical inputs
        prob_diff = torch.abs(output1['edit_type'] - output2['edit_type']).max().item()
        value_diff = torch.abs(output1['value'] - output2['value']).item()
        
        print(f"Policy output consistency:")
        print(f"  Max probability difference: {prob_diff:.6f}")
        print(f"  Value difference: {value_diff:.6f}")
        
        policy_consistent = prob_diff <= 0.001 and value_diff <= 0.001
        if policy_consistent:
            print("✓ Policy is deterministic for same input")
        else:
            print("🚨 Policy outputs are inconsistent for same input!")
    
    # Summary
    print(f"\n🎯 SPECIFIC ISSUES SUMMARY:")
    print(f"  SilkomeGPT consistency: {'✅' if silkomegpt_consistent else '❌'}")
    print(f"  Reward calculation: {'✅' if reward_calc_ok else '❌'}")
    print(f"  Step-by-step tracking: {'✅' if step_by_step_ok else '❌'}")
    print(f"  Toughness variation: {'✅' if toughness_variation_ok else '❌'}")
    print(f"  Policy consistency: {'✅' if policy_consistent else '❌'}")
    
    all_good = all([silkomegpt_consistent, reward_calc_ok, step_by_step_ok, 
                   toughness_variation_ok, policy_consistent])
    
    if not all_good:
        print(f"\n🚨 CRITICAL ISSUES FOUND!")
        if not silkomegpt_consistent:
            print("  - SilkomeGPT giving inconsistent predictions")
        if not reward_calc_ok:
            print("  - Reward calculation has bugs")
        if not step_by_step_ok:
            print("  - Step-by-step toughness tracking broken")
        if not toughness_variation_ok:
            print("  - Toughness predictions too similar across sequences")
        if not policy_consistent:
            print("  - Policy network has consistency issues")
    else:
        print(f"\n✅ All specific tests passed!")
    
    return {
        'silkomegpt_consistent': silkomegpt_consistent,
        'reward_calc_ok': reward_calc_ok,
        'step_by_step_ok': step_by_step_ok,
        'toughness_variation_ok': toughness_variation_ok,
        'policy_consistent': policy_consistent,
        'all_tests_passed': all_good
    }

def run_full_diagnosis(trainer, env, dataset, device: torch.device, 
                      save_report: bool = True) -> Dict[str, Any]:
    """
    Run complete diagnosis and optionally save report
    
    Args:
        trainer: PPO trainer instance
        env: Environment instance
        dataset: Dataset instance  
        device: Computation device
        save_report: Whether to save a text report
        
    Returns:
        Complete diagnostics dictionary
    """
    diagnostics = diagnose_training_issues(trainer, env, dataset, device)
    
    if save_report:
        save_debug_report(diagnostics)
    
    return diagnostics

def test_perplexity_distribution(env, dataset, verbose=True):
    """Test perplexity distribution of real sequences and small edits"""
    if verbose:
        print("\n=== TESTING PERPLEXITY DISTRIBUTION ===")
    
    # Test original sequences
    sequences = dataset.get_test_sequences(10)
    original_perplexities = []
    
    for seq in sequences:
        ppl = env.utils.calculate_perplexity(seq)
        original_perplexities.append(ppl)
    
    if verbose:
        print(f"Original sequences perplexity:")
        print(f"  Mean: {np.mean(original_perplexities):.3f}")
        print(f"  Std:  {np.std(original_perplexities):.3f}")
        print(f"  Min:  {np.min(original_perplexities):.3f}")
        print(f"  Max:  {np.max(original_perplexities):.3f}")
        print(f"  95th percentile: {np.percentile(original_perplexities, 95):.3f}")
    
    # Test small conservative edits
    test_seq = "GPGGQGPYGPGGQGPGGQGPYGPQAAAAAAAAAAAGPGGQGPYGPGGQ"
    edit_perplexities = []
    
    conservative_edits = [
        # Conservative substitutions (similar amino acids)
        test_seq.replace('G', 'A', 1),  # G->A (both small)
        test_seq.replace('A', 'G', 1),  # A->G 
        test_seq.replace('P', 'S', 1),  # P->S (both small)
        test_seq.replace('Q', 'N', 1),  # Q->N (both polar)
        # Small insertions
        test_seq[:10] + 'G' + test_seq[10:],  # Insert G
        test_seq[:20] + 'A' + test_seq[20:],  # Insert A
    ]
    
    for edit in conservative_edits:
        ppl = env.utils.calculate_perplexity(edit)
        edit_perplexities.append(ppl)
    
    if verbose:
        print(f"\nConservative edits perplexity:")
        print(f"  Mean: {np.mean(edit_perplexities):.3f}")
        print(f"  Std:  {np.std(edit_perplexities):.3f}")
        print(f"  Min:  {np.min(edit_perplexities):.3f}")
        print(f"  Max:  {np.max(edit_perplexities):.3f}")
    
    # Test aggressive edits
    aggressive_edits = [
        test_seq.replace('G', 'W', 5),  # G->W (very different)
        test_seq.replace('A', 'F', 3),  # A->F (very different)
        test_seq + 'WWWWW',             # Add hydrophobic tail
        test_seq[:-10],                 # Remove end
    ]
    
    aggressive_perplexities = []
    for edit in aggressive_edits:
        ppl = env.utils.calculate_perplexity(edit)
        aggressive_perplexities.append(ppl)
    
    if verbose:
        print(f"\nAggressive edits perplexity:")
        print(f"  Mean: {np.mean(aggressive_perplexities):.3f}")
        print(f"  Std:  {np.std(aggressive_perplexities):.3f}")
        print(f"  Min:  {np.min(aggressive_perplexities):.3f}")
        print(f"  Max:  {np.max(aggressive_perplexities):.3f}")
    
    # Recommend threshold
    original_95th = np.percentile(original_perplexities, 95)
    conservative_max = np.max(edit_perplexities)
    
    recommended_threshold = max(original_95th, conservative_max) * 1.1
    
    if verbose:
        print(f"\nRecommended perplexity threshold: {recommended_threshold:.3f}")
        print(f"  Current threshold: 2.5")
        print(f"  95th percentile of originals: {original_95th:.3f}")
        print(f"  Max conservative edit: {conservative_max:.3f}")
    
    return {
        'original_mean': np.mean(original_perplexities),
        'original_95th': original_95th,
        'conservative_max': conservative_max,
        'aggressive_min': np.min(aggressive_perplexities),
        'recommended_threshold': recommended_threshold
    }

# Quick test to run this
def quick_perplexity_test():
    """Quick test you can run right now"""
    import sys
    sys.path.append('.')
    
    # Setup (same as debug script)
    device = torch.device('cuda')
    config = get_config('quick_test').to_dict()
    
    # You'll need to run the setup function
    print("Run this to test perplexity distribution:")
    print("python -c \"")
    print("import sys; sys.path.append('.')") 
    print("from src.debug.debug import setup_models_and_environment")
    print("from src.config.training_configs import get_config")
    print("from src.debug.debug_tools import test_perplexity_distribution")
    print("import torch")
    print("device = torch.device('cuda')")
    print("config = get_config('quick_test').to_dict()")
    print("policy, env, dataset, utils, reward_fn = setup_models_and_environment(config, device)")
    print("results = test_perplexity_distribution(env, dataset)")
    print("print('Recommended threshold:', results['recommended_threshold'])")
    print("\"")

if __name__ == "__main__":
    print("Debug utilities for Spider Silk RL Training")