#!/usr/bin/env python3
"""
Debug-focused integration script to identify and fix the 60% failure rate
"""
import os
import sys
import torch
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.config.stable_configs_v2 import get_stable_config_v2
from src.models.improved_policy_v2 import ImprovedSequenceEditPolicyV2
from src.models.stable_reward_function_v2 import StableSpiderSilkRewardFunctionV2
from src.training.debug_stable_trainer import DebugStableTrainer
from src.environment.enhanced_protein_env import EnhancedProteinEditEnvironment
from src.data.dataset import SpiderSilkDataset
from src.utils.spider_silk_utils import SpiderSilkUtils

# Import model loading utilities
from src.debug.debug import fix_both_warnings
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoTokenizer

def setup_debug_system(config_dict, device):
    """Setup system with enhanced debugging components"""
    print("🔧 Setting up Debug System...")
    
    # Load dataset
    print("  📊 Loading dataset...")
    dataset = SpiderSilkDataset(
        config_dict['dataset_path'],
        test_size=config_dict['test_size'],
        n_difficulty_levels=config_dict['n_difficulty_levels'],
        random_state=config_dict['seed']
    )
    print(f"     ✓ Loaded {len(dataset.sequences)} sequences")
    
    # Load models (same as before)
    print("  🧬 Loading ESM-C model...")
    esmc_checkpoint = "src/models/checkpoint-1452"
    esmc_model = AutoModelForMaskedLM.from_pretrained(esmc_checkpoint, trust_remote_code=True)
    esmc_tokenizer = esmc_model.tokenizer
    esmc_tokenizer, esmc_model = fix_both_warnings(esmc_tokenizer, esmc_model)
    print("     ✓ ESM-C model loaded")
    
    print("  🕷️  Loading SilkomeGPT model...")
    trained_model_name = 'lamm-mit/SilkomeGPT'
    silkomegpt_tokenizer = AutoTokenizer.from_pretrained(trained_model_name, trust_remote_code=True)
    silkomegpt_tokenizer.pad_token = silkomegpt_tokenizer.eos_token
    silkomegpt_model = AutoModelForCausalLM.from_pretrained(trained_model_name, trust_remote_code=True)
    silkomegpt_model.config.use_cache = False
    print("     ✓ SilkomeGPT model loaded")
    
    # Build components with enhanced versions
    print("  🔧 Building enhanced components...")
    utils = SpiderSilkUtils(esmc_model, esmc_tokenizer)
    reward_fn = StableSpiderSilkRewardFunctionV2(silkomegpt_model, silkomegpt_tokenizer, esmc_model)
    
    # Move models to device
    reward_fn.silkomegpt.to(device)
    reward_fn.esmc.to(device)
    
    # Use ENHANCED environment with better error handling
    env = EnhancedProteinEditEnvironment(utils, reward_fn, max_steps=config_dict['max_steps'])
    
    # Use improved policy with bounds fixes
    policy = ImprovedSequenceEditPolicyV2().to(device)
    
    print("     ✓ All debug components created successfully")
    
    return policy, env, dataset, utils, reward_fn

def run_debug_test(episodes=100):
    """Run focused debug test to identify failure modes"""
    print("🚨 DEBUGGING MODE: Identifying Failure Patterns")
    print("="*60)
    
    # Use test config
    config = get_stable_config_v2('stable_test')
    config_dict = config.to_dict()
    config_dict['max_steps'] = 15  # Shorter episodes for faster debugging
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Setup debug system
    setup_result = setup_debug_system(config_dict, device)
    if setup_result is None:
        print("❌ Setup failed")
        return
    
    policy, env, dataset, utils, reward_fn = setup_result
    
    # Create debug trainer
    trainer = DebugStableTrainer(policy, env, lr=1e-4, device=device)
    
    print(f"\n🔍 Running {episodes} debug episodes...")