#!/usr/bin/env python3
"""
debug.py - Enhanced standalone debugging script for Spider Silk RL Training

This script loads all models and components to run comprehensive debugging
without needing an active training session.

Usage (from project root):
    python src/debug/debug.py [--config CONFIG_NAME] [--checkpoint PATH]

Examples:
    python src/debug/debug.py --config quick_test
    python src/debug/debug.py --config quick_test --enhanced
    python src/debug/debug.py --checkpoint results/runs/my_run/checkpoint_ep_100.pt
"""

import os
import sys
import argparse
import torch
import random
import numpy as np
from typing import Dict, Any, Optional

# Add src to path to match your working setup
# Get the project root directory (two levels up from this file)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

try:
    # Import using the exact same pattern as your working launch_experiment
    from src.config.training_configs import get_config
    from src.models.policy import SequenceEditPolicy
    from src.models.reward_function import SpiderSilkRewardFunction
    from src.environment.protein_env import ProteinEditEnvironment
    from src.utils.spider_silk_utils import SpiderSilkUtils
    from src.data.dataset import SpiderSilkDataset
    from src.training.ppo_trainer import PPOTrainer
    from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoTokenizer
    
    # Import debug functions from the same directory
    from src.debug.debug_tools import (
        test_silkomegpt_predictions, 
        test_improvement_detection,
        analyze_policy_behavior,
        debug_episode_details,
        diagnose_training_issues,
        run_full_diagnosis,
        save_debug_report,
        test_silkomegpt_consistency,
        test_reward_calculation_bug
    )
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you:")
    print("  1. Are running from the project root directory OR using correct path")
    print("  2. Have your src/debug/debug_tools.py file with the debug functions")
    print("  3. Have all dependencies installed in your conda environment")
    print(f"\nProject root detected as: {project_root}")
    print(f"Current working directory: {os.getcwd()}")
    print("\nTry running from project root:")
    print("  cd protein-reinforcement-learning")
    print("  python src/debug/debug.py --config quick_test --basic-only")
    sys.exit(1)


def fix_both_warnings(tokenizer, model):
    """Fix tokenization warnings (copied from your working code)"""
    print("=== FIXING TOKENIZATION WARNINGS ===")
    
    # Fix Warning 1: max_length issue
    if not hasattr(tokenizer, 'model_max_length') or tokenizer.model_max_length > 1000000:
        tokenizer.model_max_length = 1024
        print(f"Set tokenizer.model_max_length = {tokenizer.model_max_length}")
    
    # Fix Warning 2: attention mask issue  
    print(f"Current tokens:")
    print(f"  eos_token: '{tokenizer.eos_token}' (id: {tokenizer.eos_token_id})")
    print(f"  pad_token: '{tokenizer.pad_token}' (id: {tokenizer.pad_token_id})")
    
    if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        if hasattr(tokenizer, 'unk_token') and tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
            print(f"Set pad_token to unk_token: '{tokenizer.pad_token}'")
        else:
            special_tokens_dict = {'pad_token': '<PAD>'}
            num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
            print(f"Added {num_added_tokens} new tokens")
            model.resize_token_embeddings(len(tokenizer))
            print(f"Resized model embeddings to {len(tokenizer)} tokens")
    
    print(f"Fixed tokens:")
    print(f"  eos_token: '{tokenizer.eos_token}' (id: {tokenizer.eos_token_id})")
    print(f"  pad_token: '{tokenizer.pad_token}' (id: {tokenizer.pad_token_id})")
    print("=== WARNINGS FIXED ===")
    
    return tokenizer, model


def setup_models_and_environment(config: Dict[str, Any], device: torch.device):
    """
    Setup all models and environment components (copied from your working code)
    
    Args:
        config: Configuration dictionary
        device: PyTorch device
        
    Returns:
        Tuple of (policy, env, dataset, utils, reward_fn)
    """
    print("🔧 Setting up models and environment...")
    
    try:
        # Load dataset
        print("  📊 Loading dataset...")
        dataset = SpiderSilkDataset(
            config['dataset_path'],
            test_size=config['test_size'],
            n_difficulty_levels=config['n_difficulty_levels'],
            random_state=config['seed']
        )
        print(f"     ✓ Loaded {len(dataset.sequences)} sequences")
        
        # Initialize ESM-C model (exact copy from your working code)
        print("  🧬 Loading ESM-C model...")
        esmc_checkpoint = "src/models/checkpoint-1452"
        if not os.path.exists(esmc_checkpoint):
            print(f"     ❌ ESM-C checkpoint not found at {esmc_checkpoint}")
            print("     Please check the path or download the model")
            return None
            
        esmc_model = AutoModelForMaskedLM.from_pretrained(esmc_checkpoint, trust_remote_code=True)
        esmc_tokenizer = esmc_model.tokenizer
        esmc_tokenizer, esmc_model = fix_both_warnings(esmc_tokenizer, esmc_model)
        print("     ✓ ESM-C model loaded")
        
        # Initialize SilkomeGPT model (exact copy from your working code)
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
            print("     This will cause issues with toughness prediction!")
            return None
        
        # Build utility components (exact copy from your working code)
        print("  🔧 Building components...")
        utils = SpiderSilkUtils(esmc_model, esmc_tokenizer)
        reward_fn = SpiderSilkRewardFunction(
            silkomegpt_model, silkomegpt_tokenizer, esmc_model
        )
        
        # Move models to correct device
        reward_fn.silkomegpt.to(device)
        reward_fn.esmc.to(device)
        
        # Create environment
        env = ProteinEditEnvironment(utils, reward_fn, max_steps=config['max_steps'])
        
        # Create policy
        policy = SequenceEditPolicy().to(device)
        
        print("     ✓ All components created successfully")
        
        return policy, env, dataset, utils, reward_fn
        
    except Exception as e:
        print(f"❌ Error setting up models: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_checkpoint_if_provided(policy, checkpoint_path: Optional[str], device: torch.device):
    """
    Load checkpoint if provided
    
    Args:
        policy: Policy network
        checkpoint_path: Path to checkpoint file
        device: PyTorch device
        
    Returns:
        Episode number from checkpoint (0 if no checkpoint)
    """
    if checkpoint_path is None:
        print("  📝 No checkpoint provided, using randomly initialized policy")
        return 0
    
    if not os.path.exists(checkpoint_path):
        print(f"  ❌ Checkpoint not found: {checkpoint_path}")
        return 0
    
    try:
        print(f"  📁 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle both DDP and regular model states
        if 'model_state_dict' in checkpoint:
            if hasattr(policy, 'module'):
                policy.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                policy.load_state_dict(checkpoint['model_state_dict'])
        
        episode = checkpoint.get('episode', 0)
        print(f"     ✓ Checkpoint loaded from episode {episode}")
        return episode
        
    except Exception as e:
        print(f"  ❌ Failed to load checkpoint: {e}")
        return 0


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


def run_enhanced_specific_tests(policy, env, dataset, device):
    """Run enhanced specific debugging tests"""
    print("\n🔬 RUNNING ENHANCED SPECIFIC TESTS")
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
    print(f"\n🎯 ENHANCED TESTS SUMMARY:")
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
        print(f"\n✅ All enhanced tests passed!")
    
    return {
        'silkomegpt_consistent': silkomegpt_consistent,
        'reward_calc_ok': reward_calc_ok,
        'step_by_step_ok': step_by_step_ok,
        'toughness_variation_ok': toughness_variation_ok,
        'policy_consistent': policy_consistent,
        'all_tests_passed': all_good
    }


def run_basic_model_tests(policy, env, dataset, device: torch.device):
    """
    Run basic tests to ensure models are working
    
    Args:
        policy: Policy network
        env: Environment
        dataset: Dataset
        device: PyTorch device
        
    Returns:
        Dictionary with test results
    """
    print("\n🧪 Running basic model tests...")
    
    test_results = {}
    
    # Test 1: SilkomeGPT predictions
    print("  1️⃣  Testing SilkomeGPT predictions...")
    silkomegpt_working = test_silkomegpt_predictions(env.reward_fn, verbose=False)
    test_results['silkomegpt_working'] = silkomegpt_working
    
    if silkomegpt_working:
        print("     ✅ SilkomeGPT is working correctly")
    else:
        print("     ❌ SilkomeGPT has issues!")
    
    # Test 2: Improvement detection
    print("  2️⃣  Testing improvement detection...")
    try:
        improvement_stats = test_improvement_detection(env.reward_fn, verbose=False)
        test_results['improvement_stats'] = improvement_stats
        
        if improvement_stats and improvement_stats.get('std_improvement', 0) > 0.001:
            print("     ✅ SilkomeGPT detects sequence changes")
        else:
            print("     ❌ SilkomeGPT shows no sensitivity to changes")
    except Exception as e:
        print(f"     ❌ Error testing improvement detection: {e}")
        test_results['improvement_stats'] = {}
    
    # Test 3: Policy forward pass
    print("  3️⃣  Testing policy forward pass...")
    try:
        test_seq = dataset.get_test_sequences(1)[0]
        state = env.reset(test_seq).to(device)
        
        policy.eval()
        with torch.no_grad():
            output = policy(state.unsqueeze(0))
            action = policy.get_action(state, deterministic=False)
        
        print("     ✅ Policy forward pass working")
        test_results['policy_working'] = True
    except Exception as e:
        print(f"     ❌ Policy forward pass failed: {e}")
        test_results['policy_working'] = False
    
    # Test 4: Environment step
    print("  4️⃣  Testing environment step...")
    try:
        test_seq = dataset.get_test_sequences(1)[0]
        state = env.reset(test_seq).to(device)
        
        # Create a simple action
        action = {
            'type': 'substitution',
            'position': 0,
            'amino_acid': 'A',
            'log_prob': torch.tensor(0.0, device=device)
        }
        
        new_state, reward, done, info = env.step(action)
        
        print("     ✅ Environment step working")
        print(f"        Reward: {reward:.3f}")
        test_results['environment_working'] = True
    except Exception as e:
        print(f"     ❌ Environment step failed: {e}")
        test_results['environment_working'] = False
    
    return test_results


def run_comprehensive_debugging(policy, env, dataset, device: torch.device):
    """
    Run comprehensive debugging analysis
    
    Args:
        policy: Policy network
        env: Environment
        dataset: Dataset
        device: PyTorch device
        
    Returns:
        Complete diagnostics dictionary
    """
    print("\n🔍 Running comprehensive debugging analysis...")
    
    # Create a minimal trainer for the diagnosis
    # (We don't need the full training setup, just the policy)
    class MinimalTrainer:
        def __init__(self, policy):
            self.policy = policy
            self.episode_rewards = [0.5, 1.2, 0.8, 2.1, 1.5]  # Dummy data for testing
    
    trainer = MinimalTrainer(policy)
    
    try:
        diagnostics = diagnose_training_issues(trainer, env, dataset, device, n_test_sequences=3)
        return diagnostics
    except Exception as e:
        print(f"❌ Error in comprehensive debugging: {e}")
        import traceback
        traceback.print_exc()
        return {}


def main():
    """Main debugging function"""
    parser = argparse.ArgumentParser(description='Enhanced Spider Silk RL Debugging')
    parser.add_argument('--config', default='quick_test', help='Configuration name')
    parser.add_argument('--checkpoint', help='Path to checkpoint file')
    parser.add_argument('--device', default='auto', help='Device to use (auto/cpu/cuda)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--basic-only', action='store_true', help='Run only basic tests')
    parser.add_argument('--enhanced', action='store_true', help='Run enhanced specific tests')
    parser.add_argument('--save-report', action='store_true', help='Save detailed report')
    
    args = parser.parse_args()
    
    print("🕷️  Spider Silk RL Enhanced Debugging")
    print("="*50)
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"🖥️  Using device: {device}")
    
    # Get configuration
    try:
        config = get_config(args.config)
        config_dict = config.to_dict()
        print(f"📋 Using config: {args.config}")
        print(f"   Dataset: {config_dict['dataset_path']}")
        print(f"   Max steps: {config_dict['max_steps']}")
        print(f"   Curriculum: {config_dict['curriculum_strategy']}")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return 1
    
    # Setup models and environment
    setup_result = setup_models_and_environment(config_dict, device)
    if setup_result is None:
        print("❌ Failed to setup models and environment")
        return 1
    
    policy, env, dataset, utils, reward_fn = setup_result
    
    # Load checkpoint if provided
    episode = load_checkpoint_if_provided(policy, args.checkpoint, device)
    
    # Run basic tests
    print("\n" + "="*50)
    test_results = run_basic_model_tests(policy, env, dataset, device)
    
    # Check if basic tests passed
    critical_failures = []
    if not test_results.get('silkomegpt_working', False):
        critical_failures.append("SilkomeGPT not working")
    if not test_results.get('policy_working', False):
        critical_failures.append("Policy not working")
    if not test_results.get('environment_working', False):
        critical_failures.append("Environment not working")
    
    if critical_failures:
        print(f"\n❌ CRITICAL FAILURES DETECTED:")
        for failure in critical_failures:
            print(f"   - {failure}")
        print("\nCannot proceed with comprehensive debugging until these are fixed.")
        return 1
    
    print("\n✅ Basic tests passed! All core components are working.")
    
    # Run enhanced tests if requested
    enhanced_results = None
    if args.enhanced:
        enhanced_results = run_enhanced_specific_tests(policy, env, dataset, device)
    
    # Run comprehensive debugging if requested
    if not args.basic_only and not args.enhanced:
        diagnostics = run_comprehensive_debugging(policy, env, dataset, device)
        
        if diagnostics and args.save_report:
            save_debug_report(diagnostics, "standalone_debug_report.txt")
            print(f"\n📄 Detailed report saved to standalone_debug_report.txt")
    
    # Summary
    print("\n" + "="*50)
    print("🎯 DEBUGGING SUMMARY")
    print("="*50)
    
    if test_results.get('silkomegpt_working'):
        print("✅ SilkomeGPT: Working correctly")
    else:
        print("❌ SilkomeGPT: Has issues - check model loading")
    
    if test_results.get('policy_working'):
        print("✅ Policy: Forward pass working")
    else:
        print("❌ Policy: Forward pass failed")
    
    if test_results.get('environment_working'):
        print("✅ Environment: Step function working")
    else:
        print("❌ Environment: Step function failed")
    
    improvement_stats = test_results.get('improvement_stats', {})
    if improvement_stats and improvement_stats.get('std_improvement', 0) > 0.001:
        print("✅ Sensitivity: SilkomeGPT detects sequence changes")
    else:
        print("❌ Sensitivity: SilkomeGPT shows no response to changes")
    
    # Enhanced results summary
    if enhanced_results:
        print("\n🔬 ENHANCED TESTS RESULTS:")
        if enhanced_results.get('all_tests_passed'):
            print("✅ All enhanced tests passed - no critical issues found")
        else:
            print("❌ Enhanced tests found critical issues:")
            if not enhanced_results.get('step_by_step_ok'):
                print("   🚨 Step-by-step toughness tracking is broken!")
            if not enhanced_results.get('reward_calc_ok'):
                print("   🚨 Reward calculation has bugs!")
            if not enhanced_results.get('silkomegpt_consistent'):
                print("   🚨 SilkomeGPT predictions are inconsistent!")
    
    if critical_failures:
        print(f"\n🚨 Fix these critical issues before training:")
        for failure in critical_failures:
            print(f"   - {failure}")
        return 1
    elif enhanced_results and not enhanced_results.get('all_tests_passed'):
        print(f"\n🚨 Enhanced tests found issues - check the detailed output above")
        return 1
    else:
        print(f"\n🎉 All systems operational! Training should work.")
        return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)