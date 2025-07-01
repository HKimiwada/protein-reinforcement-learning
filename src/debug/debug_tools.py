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


if __name__ == "__main__":
    print("Debug utilities for Spider Silk RL Training")