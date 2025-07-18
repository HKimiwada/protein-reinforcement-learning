import os
import sys
import random
import numpy as np
import torch
import torch.multiprocessing as mp
from typing import Dict, Any
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoTokenizer

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from training.distributed_utils import setup_distributed_training, cleanup_distributed
from training.ppo_trainer import DistributedPPOTrainer
from models.policy import SequenceEditPolicy
from models.reward_function import SpiderSilkRewardFunction
from environment.protein_env import ProteinEditEnvironment
from utils.spider_silk_utils import SpiderSilkUtils
from utils.logging_utils import WandBLogger
from utils.checkpoint_utils import save_checkpoint
from utils.evaluation_utils import evaluate_policy
from data.dataset import SpiderSilkDataset

# SPECIFIC FIXES FOR THE TWO WARNINGS

## WARNING 1 FIX: "Asking to truncate to max_length but no maximum length is provided..."

def fix_max_length_warning(tokenizer):
    """Fix the max_length truncation warning"""
    
    # Set explicit model_max_length if not set or too large
    if not hasattr(tokenizer, 'model_max_length') or tokenizer.model_max_length > 1000000:
        tokenizer.model_max_length = 1024
        print(f"Set tokenizer.model_max_length = {tokenizer.model_max_length}")
    
    return tokenizer

## WARNING 2 FIX: "The attention mask is not set and cannot be inferred..."

def fix_attention_mask_warning(tokenizer, model):
    """Fix the attention mask warning by ensuring different pad and eos tokens"""
    
    print(f"Current tokens:")
    print(f"  eos_token: '{tokenizer.eos_token}' (id: {tokenizer.eos_token_id})")
    print(f"  pad_token: '{tokenizer.pad_token}' (id: {tokenizer.pad_token_id})")
    
    # Check if pad_token is None or same as eos_token
    if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        
        # Option 1: Use existing special token
        if hasattr(tokenizer, 'unk_token') and tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
            print(f"Set pad_token to unk_token: '{tokenizer.pad_token}'")
        
        # Option 2: Add new pad token if no unk_token available
        else:
            special_tokens_dict = {'pad_token': '<PAD>'}
            num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
            print(f"Added {num_added_tokens} new tokens")
            
            # IMPORTANT: Resize model embeddings when adding new tokens
            model.resize_token_embeddings(len(tokenizer))
            print(f"Resized model embeddings to {len(tokenizer)} tokens")
    
    print(f"Fixed tokens:")
    print(f"  eos_token: '{tokenizer.eos_token}' (id: {tokenizer.eos_token_id})")
    print(f"  pad_token: '{tokenizer.pad_token}' (id: {tokenizer.pad_token_id})")
    
    return tokenizer, model

## COMPLETE FIX FUNCTION

def fix_both_warnings(tokenizer, model):
    """Fix both warnings in one function"""
    
    print("=== FIXING TOKENIZATION WARNINGS ===")
    
    # Fix Warning 1: max_length issue
    tokenizer = fix_max_length_warning(tokenizer)
    
    # Fix Warning 2: attention mask issue  
    tokenizer, model = fix_attention_mask_warning(tokenizer, model)
    
    print("=== WARNINGS FIXED ===")
    
    return tokenizer, model

def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_worker(rank: int, world_size: int, config: Dict[str, Any]):
    """Main training worker function"""
    
    # Setup distributed training ONLY if multi-GPU
    distributed_setup = False
    if world_size > 1:
        setup_distributed_training(rank, world_size)
        distributed_setup = True
    
    set_seed(config['seed'] + rank)
    device = torch.device(f'cuda:{rank}')
    
    try:
        # Initialize logger (only rank 0)
        logger = WandBLogger(config, rank)
        
        # Load dataset
        dataset = SpiderSilkDataset(
            config['dataset_path'],
            test_size=config['test_size'],
            n_difficulty_levels=config['n_difficulty_levels'],
            random_state=config['seed']
        )
        
        # Initialize models
        esmc_checkpoint = "src/models/checkpoint-1452"
        esmc_model = AutoModelForMaskedLM.from_pretrained(esmc_checkpoint, trust_remote_code=True)
        esmc_tokenizer = esmc_model.tokenizer
        esmc_tokenizer, esmc_model = fix_both_warnings(esmc_tokenizer, esmc_model)
   
        trained_model_name='lamm-mit/SilkomeGPT'
        silkomegpt_tokenizer = AutoTokenizer.from_pretrained(trained_model_name, trust_remote_code=True)
        silkomegpt_tokenizer.pad_token = silkomegpt_tokenizer.eos_token
        silkomegpt_model = AutoModelForCausalLM.from_pretrained(
            trained_model_name,
            trust_remote_code=True
        ).to(device)
        silkomegpt_model.config.use_cache = False

        # Build components
        utils = SpiderSilkUtils(esmc_model, esmc_tokenizer)
        reward_fn = SpiderSilkRewardFunction(
            silkomegpt_model, silkomegpt_tokenizer, esmc_model
        )
        
        # Move models to correct device
        reward_fn.silkomegpt.to(device)
        reward_fn.esmc.to(device)
        
        env = ProteinEditEnvironment(utils, reward_fn, max_steps=config['max_steps'])
        policy = SequenceEditPolicy().to(device)
        
        # Choose correct trainer based on world_size
        if world_size > 1:
            trainer = DistributedPPOTrainer(
                policy, env,
                lr=config['learning_rate'],
                clip_epsilon=config['clip_epsilon'],
                value_coeff=config['value_coeff'],
                entropy_coeff=config['entropy_coeff'],
                rank=rank,
                world_size=world_size
            )
        else:
            from training.ppo_trainer import PPOTrainer
            trainer = PPOTrainer(
                policy, env,
                lr=config['learning_rate'],
                clip_epsilon=config['clip_epsilon'],
                value_coeff=config['value_coeff'],
                entropy_coeff=config['entropy_coeff'],
                device=device
            )
        
        # Training loop
        best_reward = -float('inf')
        difficulty_counts = {i: 0 for i in range(config['n_difficulty_levels'])}
        
        for episode in range(config['n_episodes']):
            # Get curriculum-based sequence
            start_seq, difficulty_level = dataset.get_curriculum_sequence(
                episode, config['n_episodes'], config['curriculum_strategy']
            )
            difficulty_counts[difficulty_level] += 1
            
            # Train episode
            episode_data = trainer.train_episode(start_seq, episode, difficulty_level)
            
            # Logging (only rank 0)
            if rank == 0:
                # Log episode metrics
                trainer_metrics = {
                    'policy_loss': trainer.policy_losses[-1],
                    'value_loss': trainer.value_losses[-1]
                }
                
                logger.log_episode(episode, episode_data, difficulty_level, trainer_metrics)
                
                # Track best reward
                if episode_data['episode_reward'] > best_reward:
                    best_reward = episode_data['episode_reward']
                    logger.log({'best_reward': best_reward})
                
                # Periodic detailed logging
                if episode % config['log_interval'] == 0 and episode > 0:
                    recent_difficulties = trainer.difficulty_levels[-config['log_interval']:]
                    recent_diff_counts = {i: recent_difficulties.count(i) 
                                        for i in range(config['n_difficulty_levels'])}
                    
                    logger.log_curriculum_progress(episode, config['n_episodes'], recent_diff_counts)
                    
                    avg_reward = np.mean(trainer.episode_rewards[-config['log_interval']:])
                    print(f"Episode {episode}: avg_reward={avg_reward:.3f}, best={best_reward:.3f}")
                
                # Test evaluation - both training and test sequences
                if episode % config['test_interval'] == 0 and episode > 0:
                    # Import the new function
                    from utils.evaluation_utils import test_on_training_sequences
                    
                    # Test on TRAINING sequences (should work if model is learning)
                    train_results = test_on_training_sequences(
                        trainer.policy if world_size == 1 else trainer.policy.module,
                        env, dataset, device, config['test_sequences_per_eval']
                    )
                    
                    # Test on TEST sequences (generalization check)
                    test_sequences = dataset.get_test_sequences(config['test_sequences_per_eval'])
                    test_results = evaluate_policy(
                        trainer.policy if world_size == 1 else trainer.policy.module,
                        env, test_sequences, device
                    )
                    
                    # Log both results
                    logger.log({
                        'train_avg_improvement': train_results['avg_improvement'],
                        'train_avg_reward': train_results['avg_reward'],
                        'train_success_rate': train_results['success_rate']
                    })
                    logger.log_test_results(test_results)
                    
                    print(f"Test @ {episode}:")
                    print(f"  TRAIN: reward={train_results['avg_reward']:.3f}, improvement={train_results['avg_improvement']:.4f}")
                    print(f"  TEST:  reward={test_results['avg_reward']:.3f}, improvement={test_results['avg_improvement']:.4f}")
                                
                # Save checkpoint
                if episode % config['checkpoint_interval'] == 0 and episode > 0:
                    save_dir = os.path.join(config['save_dir'], config['run_name'])
                    checkpoint_path = os.path.join(save_dir, f"checkpoint_ep_{episode}.pt")
                    
                    save_checkpoint(
                        checkpoint_path, episode, trainer.policy, 
                        trainer.optimizer, config
                    )
                    
                    if hasattr(logger, 'is_main_process') and logger.is_main_process:
                        import wandb
                        wandb.save(checkpoint_path)
        
        if rank == 0:
            print("Training completed!")
            logger.finish()
    
    finally:
        # Only cleanup distributed if it was set up
        if distributed_setup:
            cleanup_distributed()

def main(config: Dict[str, Any]):
    """Main entry point for distributed training"""
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Launch distributed training
    mp.spawn(
        train_worker,
        args=(config['world_size'], config),
        nprocs=config['world_size'],
        join=True
    )