"""
python src/experiments/run_stable_v5.py --config stable --episodes 600 --seeds 42,123,456,789,999,1337,2024,7777
"""
import os
import sys
import argparse
import torch
import torch.multiprocessing as mp
import numpy as np
import pandas as pd
import json
import time
import logging
import random
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.config.stable_configs_v2 import get_stable_config_v2
from src.models.improved_policy_v2 import ImprovedSequenceEditPolicyV2
from src.models.stable_reward_function_v2 import StableSpiderSilkRewardFunctionV2
from src.data.dataset import SpiderSilkDataset
from src.utils.spider_silk_utils import SpiderSilkUtils
from src.environment.protein_env import ProteinEditEnvironment
from src.debug.debug import fix_both_warnings
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoTokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleMetricsLogger:
    """Lightweight metrics logger for raw data collection"""
    
    def __init__(self, experiment_id: str, save_dir: str):
        self.experiment_id = experiment_id
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Raw episode data
        self.episodes = []
        self.test_evaluations = []
        
        # Experiment metadata
        self.metadata = {
            'experiment_id': experiment_id,
            'start_time': datetime.now().isoformat(),
            'episodes_completed': 0
        }

    def log_episode(self, episode_num: int, result: Dict[str, Any]):
        """Log raw episode data"""
        
        episode_data = {
            'episode': episode_num,
            'timestamp': time.time(),
            'episode_reward': result.get('episode_reward', 0.0),
            'episode_length': result.get('episode_length', 0),
            'actual_improvement': result.get('actual_improvement', 0.0),
            'cumulative_improvement': result.get('cumulative_improvement', 0.0),
            'original_toughness': result.get('original_toughness', 0.0),
            'final_toughness': result.get('final_toughness', 0.0),
            'episode_time': result.get('episode_time', 0.0),
            'difficulty_level': result.get('difficulty_level', 'unknown'),
            
            # Success metrics
            'success': 1 if result.get('episode_reward', 0) > 0 else 0,
            'significant_improvement': 1 if result.get('actual_improvement', 0) > 0.002 else 0,
            'improvement_per_edit': result.get('actual_improvement', 0) / max(1, result.get('episode_length', 1)),
            'reward_improvement_correlation': result.get('episode_reward', 0) * result.get('actual_improvement', 0),
            
            # Edit metrics
            'successful_edits': result.get('episode_summary', {}).get('successful_edits', 0),
            'substitutions': result.get('episode_summary', {}).get('edit_type_counts', {}).get('substitution', 0),
            'insertions': result.get('episode_summary', {}).get('edit_type_counts', {}).get('insertion', 0),
            'deletions': result.get('episode_summary', {}).get('edit_type_counts', {}).get('deletion', 0),
            
            # Training metrics
            'policy_loss': result.get('policy_loss', 0.0),
            'value_loss': result.get('value_loss', 0.0),
            'entropy_loss': result.get('entropy_loss', 0.0),
            'learning_rate': result.get('current_lr', 0.0),
            
            # Sequence info
            'original_length': len(result.get('starting_sequence', '')),
            'final_length': len(result.get('final_sequence', '')),
        }
        
        self.episodes.append(episode_data)
        self.metadata['episodes_completed'] = episode_num

    def log_test_evaluation(self, episode_num: int, test_results: Dict[str, Any]):
        """Log test set evaluation"""
        
        eval_data = {
            'episode': episode_num,
            'timestamp': time.time(),
            'avg_reward': test_results.get('avg_reward', 0.0),
            'avg_improvement': test_results.get('avg_improvement', 0.0),
            'success_rate': test_results.get('success_rate', 0.0),
            'num_sequences': len(test_results.get('results', [])),
            'std_improvement': np.std([r['improvement'] for r in test_results.get('results', [])]),
            'max_improvement': max([r['improvement'] for r in test_results.get('results', [])], default=0),
            'min_improvement': min([r['improvement'] for r in test_results.get('results', [])], default=0)
        }
        
        self.test_evaluations.append(eval_data)

    def save_all(self):
        """Save all collected data"""
        
        # Save episode data
        episodes_df = pd.DataFrame(self.episodes)
        episodes_path = self.save_dir / f"{self.experiment_id}_episodes.csv"
        episodes_df.to_csv(episodes_path, index=False)
        
        # Save test evaluations
        if self.test_evaluations:
            test_df = pd.DataFrame(self.test_evaluations)
            test_path = self.save_dir / f"{self.experiment_id}_test_evals.csv"
            test_df.to_csv(test_path, index=False)
        
        # Save metadata
        self.metadata['end_time'] = datetime.now().isoformat()
        self.metadata['total_episodes'] = len(self.episodes)
        self.metadata['total_time'] = sum(ep['episode_time'] for ep in self.episodes)
        
        metadata_path = self.save_dir / f"{self.experiment_id}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info(f"Data saved: {episodes_path}")
        
        return {
            'episodes_file': str(episodes_path),
            'test_evals_file': str(test_path) if self.test_evaluations else None,
            'metadata_file': str(metadata_path),
            'total_episodes': len(self.episodes)
        }


class SimplifiedTrainer:
    """Streamlined trainer for data collection"""
    
    def __init__(self, policy, environment, lr=1e-4, device=None):
        self.policy = policy
        self.environment = environment
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.policy.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=lr,
            weight_decay=0.01,
            betas=(0.9, 0.95)
        )
        
        # PPO parameters
        self.clip_epsilon = 0.15
        self.value_coeff = 0.8
        self.entropy_coeff = 0.08
        self.max_grad_norm = 0.3
        self.gamma = 0.99
        self.gae_lambda = 0.95

    def train_episode(self, starting_sequence: str, episode_number: int):
        """Train single episode and return metrics"""
        
        episode_start = time.time()
        
        # Collect experience
        episode_data = self._collect_episode(starting_sequence, episode_number)
        
        # Update policy
        if len(episode_data['states']) > 1:
            training_metrics = self._update_policy(episode_data)
        else:
            training_metrics = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy_loss': 0.0}
        
        # Calculate improvements
        actual_improvement = self._calculate_improvement(starting_sequence, episode_data['final_sequence'])
        
        # Return comprehensive metrics
        return {
            'episode_reward': episode_data['episode_reward'],
            'episode_length': episode_data['episode_length'],
            'actual_improvement': actual_improvement,
            'final_sequence': episode_data['final_sequence'],
            'starting_sequence': starting_sequence,
            'episode_time': time.time() - episode_start,
            'current_lr': self.optimizer.param_groups[0]['lr'],
            'episode_summary': self.environment.get_episode_summary(),
            **training_metrics
        }

    def _collect_episode(self, starting_sequence: str, episode_number: int):
        """Collect episode experience"""
        
        state = self.environment.reset(starting_sequence).to(self.device)
        self.environment.set_episode_number(episode_number)
        
        states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []
        episode_reward = 0
        
        self.policy.eval()
        
        while not self.environment.done and len(states) < 25:
            with torch.no_grad():
                policy_output = self.policy(state)
                current_seq_len = len(self.environment.current_sequence)
                action = self.policy.get_action(state, deterministic=False, sequence_length=current_seq_len)
                
                log_prob = action.get('log_prob', torch.tensor(0.0))
                if not isinstance(log_prob, torch.Tensor):
                    log_prob = torch.tensor(float(log_prob), device=self.device)
                log_prob = log_prob.to(self.device).reshape([])
                
                states.append(state.clone())
                actions.append(action)
                values.append(policy_output['value'].item())
                log_probs.append(log_prob)
                
                next_state, reward, done, info = self.environment.step(action)
                
                if np.isnan(reward) or np.isinf(reward):
                    reward = -0.1
                
                rewards.append(reward)
                dones.append(done)
                episode_reward += reward
                
                state = next_state.to(self.device)
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'values': values,
            'log_probs': log_probs,
            'dones': dones,
            'episode_reward': episode_reward,
            'episode_length': len(states),
            'final_sequence': self.environment.current_sequence
        }

    def _update_policy(self, episode_data):
        """PPO policy update"""
        
        self.policy.train()
        
        states = episode_data['states']
        actions = episode_data['actions']
        rewards = episode_data['rewards']
        values = episode_data['values']
        log_probs = episode_data['log_probs']
        dones = episode_data['dones']
        
        # Stack tensors
        if len(states) > 1:
            states_tensor = torch.stack(states)
        else:
            states_tensor = states[0].unsqueeze(0)
        
        old_log_probs = torch.stack(log_probs)
        rewards_tensor = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        values_tensor = torch.tensor(values, device=self.device, dtype=torch.float32)
        dones_tensor = torch.tensor(dones, device=self.device, dtype=torch.bool)
        
        # GAE
        advantages, returns = self._calculate_gae(rewards_tensor, values_tensor, dones_tensor)
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # New policy outputs
        new_log_probs = []
        new_values = []
        entropies = []
        
        for i, (state, action) in enumerate(zip(states, actions)):
            policy_output = self.policy(state.unsqueeze(0))
            
            # Handle dimensions
            if policy_output['edit_type'].dim() == 1:
                action_probs = {
                    'edit_type': policy_output['edit_type'],
                    'position': policy_output['position'], 
                    'amino_acid': policy_output['amino_acid']
                }
            else:
                action_probs = {
                    'edit_type': policy_output['edit_type'][0],
                    'position': policy_output['position'][0], 
                    'amino_acid': policy_output['amino_acid'][0]
                }
            
            # Calculate log prob
            if action['type'] == 'stop':
                log_prob = torch.log(action_probs['edit_type'][3] + 1e-8)
            else:
                edit_type_idx = ['substitution', 'insertion', 'deletion'].index(action['type'])
                log_prob = torch.log(action_probs['edit_type'][edit_type_idx] + 1e-8)
                
                position = action['position']
                if position < action_probs['position'].shape[0]:
                    log_prob += torch.log(action_probs['position'][position] + 1e-8)
                
                if action['type'] in ['substitution', 'insertion'] and action['amino_acid']:
                    aa_idx = list('ACDEFGHIKLMNPQRSTVWY').index(action['amino_acid'])
                    if aa_idx < action_probs['amino_acid'].shape[0]:
                        log_prob += torch.log(action_probs['amino_acid'][aa_idx] + 1e-8)
            
            # Entropy
            edit_type_entropy = -(action_probs['edit_type'] * torch.log(action_probs['edit_type'] + 1e-8)).sum()
            pos_entropy = -(action_probs['position'] * torch.log(action_probs['position'] + 1e-8)).sum()
            aa_entropy = -(action_probs['amino_acid'] * torch.log(action_probs['amino_acid'] + 1e-8)).sum()
            total_entropy = edit_type_entropy + pos_entropy + aa_entropy
            
            # Value
            if policy_output['value'].dim() == 0:
                value = policy_output['value']
            elif policy_output['value'].dim() == 1:
                value = policy_output['value'][0] if policy_output['value'].shape[0] > 0 else policy_output['value']
            else:
                value = policy_output['value'][0, 0]
            
            new_log_probs.append(log_prob.reshape([]))
            new_values.append(value.reshape([]))
            entropies.append(total_entropy.reshape([]))
        
        new_log_probs = torch.stack(new_log_probs)
        new_values = torch.stack(new_values)
        entropies = torch.stack(entropies)
        
        # PPO losses
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        value_loss = torch.nn.functional.mse_loss(new_values, returns)
        entropy_loss = -entropies.mean()
        
        total_loss = policy_loss + self.value_coeff * value_loss + self.entropy_coeff * entropy_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item()
        }

    def _calculate_gae(self, rewards, values, dones):
        """GAE calculation"""
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        next_value = 0.0 if dones[-1] else values[-1].item()
        gae = 0
        
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[step].float()
                next_value_step = next_value
            else:
                next_non_terminal = 1.0 - dones[step].float()
                next_value_step = values[step + 1]
            
            delta = rewards[step] + self.gamma * next_value_step * next_non_terminal - values[step]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[step] = gae
            returns[step] = gae + values[step]
        
        return advantages, returns

    def _calculate_improvement(self, original_sequence, final_sequence):
        """Calculate toughness improvement"""
        try:
            original_toughness, _ = self.environment.reward_fn.predict_toughness(original_sequence)
            final_toughness, _ = self.environment.reward_fn.predict_toughness(final_sequence)
            return final_toughness - original_toughness
        except:
            return 0.0

    def evaluate_test_set(self, dataset, n_sequences=50):
        """Simple test set evaluation"""
        test_sequences = dataset.get_test_sequences(n_sequences)
        results = []
        
        self.policy.eval()
        
        for seq in test_sequences:
            original_tough, _ = self.environment.reward_fn.predict_toughness(seq)
            
            state = self.environment.reset(seq).to(self.device)
            total_reward = 0
            
            with torch.no_grad():
                while not self.environment.done and len(self.environment.edit_history) < 15:
                    action = self.policy.get_action(state, deterministic=True)
                    state, reward, done, info = self.environment.step(action)
                    state = state.to(self.device)
                    total_reward += reward
            
            final_tough, _ = self.environment.reward_fn.predict_toughness(self.environment.current_sequence)
            improvement = final_tough - original_tough
            
            results.append({
                'reward': total_reward,
                'improvement': improvement
            })
        
        self.policy.train()
        
        return {
            'results': results,
            'avg_reward': np.mean([r['reward'] for r in results]),
            'avg_improvement': np.mean([r['improvement'] for r in results]),
            'success_rate': np.mean([r['improvement'] > 0.001 for r in results])
        }


def setup_models(config_dict, device):
    """Setup models on specific device"""
    
    # CRITICAL: Verify GPU isolation is working
    if device.type == 'cuda':
        # Check that we can only see one GPU (the assigned one)
        visible_gpus = torch.cuda.device_count()
        current_gpu = torch.cuda.current_device()
        logger.info(f"Process {os.getpid()} sees {visible_gpus} GPU(s), using GPU {current_gpu}")
        
        if visible_gpus != 1:
            logger.warning(f"Expected to see 1 GPU, but seeing {visible_gpus}. GPU isolation may not be working!")
        
        torch.cuda.set_device(0)  # Use the only visible GPU (index 0)
        torch.cuda.empty_cache()
    
    # Load dataset
    dataset = SpiderSilkDataset(
        config_dict['dataset_path'],
        test_size=config_dict['test_size'],
        n_difficulty_levels=config_dict['n_difficulty_levels'],
        random_state=config_dict['seed']
    )
    
    # Load models with explicit device placement
    esmc_checkpoint = "src/models/checkpoint-1452"
    esmc_model = AutoModelForMaskedLM.from_pretrained(
        esmc_checkpoint, 
        trust_remote_code=True,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32
    )
    esmc_tokenizer = esmc_model.tokenizer
    esmc_tokenizer, esmc_model = fix_both_warnings(esmc_tokenizer, esmc_model)
    esmc_model = esmc_model.to(device)
    
    # Load SilkomeGPT
    silkomegpt_tokenizer = AutoTokenizer.from_pretrained('lamm-mit/SilkomeGPT', trust_remote_code=True)
    silkomegpt_tokenizer.pad_token = silkomegpt_tokenizer.eos_token
    silkomegpt_model = AutoModelForCausalLM.from_pretrained(
        'lamm-mit/SilkomeGPT', 
        trust_remote_code=True,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32
    )
    silkomegpt_model.config.use_cache = False
    silkomegpt_model = silkomegpt_model.to(device)
    
    # Create components
    utils = SpiderSilkUtils(esmc_model, esmc_tokenizer)
    reward_fn = StableSpiderSilkRewardFunctionV2(silkomegpt_model, silkomegpt_tokenizer, esmc_model)
    reward_fn.silkomegpt = reward_fn.silkomegpt.to(device)
    reward_fn.esmc = reward_fn.esmc.to(device)
    
    environment = ProteinEditEnvironment(utils, reward_fn, max_steps=config_dict['max_steps'])
    policy = ImprovedSequenceEditPolicyV2().to(device)
    
    return policy, environment, dataset


def run_single_experiment(seed: int, config_name: str, episodes: int, gpu_id: int, save_dir: str):
    """Run single experiment and save raw data"""
    
    # CRITICAL: Set CUDA_VISIBLE_DEVICES at the very start before any torch operations
    if gpu_id >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        logger.info(f"Process {os.getpid()} set CUDA_VISIBLE_DEVICES={gpu_id}")
    
    # Import torch AFTER setting CUDA_VISIBLE_DEVICES
    import torch
    import torch.cuda
    
    # Now setup device
    if torch.cuda.is_available() and gpu_id >= 0:
        device = torch.device('cuda:0')  # Always cuda:0 since we set CUDA_VISIBLE_DEVICES
        torch.cuda.empty_cache()
        
        # Verify isolation
        visible_gpus = torch.cuda.device_count()
        if visible_gpus == 1:
            logger.info(f"✅ Process {os.getpid()} successfully isolated to GPU {gpu_id}")
        else:
            logger.warning(f"⚠️ Process {os.getpid()} sees {visible_gpus} GPUs instead of 1!")
    else:
        device = torch.device('cpu')
        logger.info(f"Process {os.getpid()} using CPU")
    
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Get config
    config = get_stable_config_v2(config_name)
    config_dict = config.to_dict()
    config_dict['seed'] = seed
    
    # Create experiment ID
    experiment_id = f"{config_name}_seed_{seed}_gpu_{gpu_id}"
    
    logger.info(f"Starting {experiment_id} on {device}")
    
    try:
        # Setup models
        policy, environment, dataset = setup_models(config_dict, device)
        
        # Create trainer and metrics logger
        trainer = SimplifiedTrainer(policy, environment, lr=config_dict['learning_rate'], device=device)
        metrics_logger = SimpleMetricsLogger(experiment_id, save_dir)
        
        # Training loop
        for episode in range(1, episodes + 1):
            # Get sequence
            starting_sequence, difficulty_level = dataset.get_curriculum_sequence(
                episode, episodes, config_dict['curriculum_strategy']
            )
            
            # Train episode
            result = trainer.train_episode(starting_sequence, episode)
            
            # Add toughness metrics
            if result['final_sequence'] != starting_sequence:
                try:
                    original_toughness, _ = environment.reward_fn.predict_toughness(starting_sequence)
                    final_toughness, _ = environment.reward_fn.predict_toughness(result['final_sequence'])
                    result['original_toughness'] = original_toughness
                    result['final_toughness'] = final_toughness
                    result['cumulative_improvement'] = final_toughness - original_toughness
                except:
                    result['original_toughness'] = 0.0
                    result['final_toughness'] = 0.0
                    result['cumulative_improvement'] = 0.0
            
            # Add GPU info to result
            result['gpu_id'] = gpu_id
            result['device'] = str(device)
            
            # Log metrics
            metrics_logger.log_episode(episode, result)
            
            # Progress logging
            if episode % 100 == 0:
                logger.info(f"{experiment_id} - Episode {episode}: "
                           f"reward={result['episode_reward']:.3f}, "
                           f"improvement={result['actual_improvement']:.4f}")
            
            # Test evaluation
            if episode % 500 == 0:
                test_results = trainer.evaluate_test_set(dataset, n_sequences=50)
                metrics_logger.log_test_evaluation(episode, test_results)
                logger.info(f"{experiment_id} - Test eval: "
                           f"improvement={test_results['avg_improvement']:.4f}, "
                           f"success_rate={test_results['success_rate']:.3f}")
        
        # Final test evaluation
        final_test = trainer.evaluate_test_set(dataset, n_sequences=100)
        metrics_logger.log_test_evaluation(episodes, final_test)
        
        # Save all data
        saved_files = metrics_logger.save_all()
        
        # Cleanup
        if device.type == 'cuda':
            del policy, environment, trainer
            torch.cuda.empty_cache()
        
        logger.info(f"{experiment_id} completed successfully")
        
        return {
            'experiment_id': experiment_id,
            'seed': seed,
            'gpu_id': gpu_id,
            'success': True,
            'saved_files': saved_files,
            'final_test': final_test
        }
        
    except Exception as e:
        logger.error(f"{experiment_id} failed: {e}")
        return {
            'experiment_id': experiment_id,
            'seed': seed,
            'gpu_id': gpu_id,
            'success': False,
            'error': str(e)
        }


def get_available_gpus():
    """Get list of available GPUs"""
    if not torch.cuda.is_available():
        return []
    
    available_gpus = []
    for i in range(torch.cuda.device_count()):
        try:
            # Test if GPU is accessible
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
            available_gpus.append(i)
        except:
            pass
    
    return available_gpus


def main():
    """Main function for simplified multi-seed training"""
    
    parser = argparse.ArgumentParser(description='Simplified Multi-Seed RL Training')
    parser.add_argument('--config', type=str, default='stable',
                       choices=['stable', 'stable_conservative', 'stable_aggressive', 'stable_test'],
                       help='Configuration to use')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of episodes per experiment')
    parser.add_argument('--seeds', type=str, default='42,123,456,789,999',
                       help='Comma-separated list of seeds')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='Max parallel workers (default: num GPUs)')
    parser.add_argument('--start-gpu', type=int, default=0,
                       help='Starting GPU ID')
    parser.add_argument('--gpus', type=str, default=None,
                       help='Comma-separated list of GPU IDs to use (e.g., "0,1,2,3,4,5,6,7")')
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Save directory (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(',')]
    
    # Create save directory
    if args.save_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_dir = f"results/multi_seed_{args.config}_{timestamp}"
    else:
        save_dir = args.save_dir
    
    os.makedirs(save_dir, exist_ok=True)
    
    # GPU setup
    if args.gpus:
        # Use specified GPUs
        available_gpus = [int(g.strip()) for g in args.gpus.split(',')]
    else:
        # Auto-detect available GPUs
        available_gpus = get_available_gpus()
        if available_gpus:
            available_gpus = [g for g in available_gpus if g >= args.start_gpu]
        else:
            available_gpus = list(range(8))  # Default to 8 GPUs
    
    if not available_gpus:
        logger.warning("No GPUs specified, using GPUs 0-7")
        available_gpus = list(range(8))
    
    # Set max workers to number of available GPUs for 1:1 mapping
    max_workers = len(available_gpus)
    if args.max_workers:
        max_workers = min(args.max_workers, len(available_gpus))
    
    logger.info(f"Running {len(seeds)} experiments on {max_workers} workers")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Available GPUs: {available_gpus}")
    logger.info(f"Save directory: {save_dir}")
    
    # Create experiment assignments - one GPU per experiment
    experiments = []
    for i, seed in enumerate(seeds):
        gpu_id = available_gpus[i % len(available_gpus)]
        experiments.append((seed, gpu_id))
    
    logger.info("Experiment assignments:")
    for seed, gpu_id in experiments:
        logger.info(f"  Seed {seed} -> GPU {gpu_id}")
    
    # Set environment variable for memory management
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Run experiments in parallel
    all_results = []
    
    # Process experiments in batches equal to number of GPUs
    batch_size = len(available_gpus)
    
    for batch_start in range(0, len(experiments), batch_size):
        batch_experiments = experiments[batch_start:batch_start + batch_size]
        logger.info(f"Running batch {batch_start//batch_size + 1}: {len(batch_experiments)} experiments")
        
        with ProcessPoolExecutor(max_workers=len(batch_experiments)) as executor:
            future_to_experiment = {}
            
            for seed, gpu_id in batch_experiments:
                future = executor.submit(run_single_experiment, seed, args.config, args.episodes, gpu_id, save_dir)
                future_to_experiment[future] = (seed, gpu_id)
            
            for future in as_completed(future_to_experiment):
                seed, gpu_id = future_to_experiment[future]
                result = future.result()
                all_results.append(result)
                
                if result['success']:
                    logger.info(f"✅ Seed {seed} (GPU {gpu_id}) completed")
                else:
                    logger.error(f"❌ Seed {seed} (GPU {gpu_id}) failed: {result.get('error', 'Unknown')}")
        
        # Clean up between batches
        if batch_start + batch_size < len(experiments):
            logger.info("Cleaning up before next batch...")
            time.sleep(5)  # Give GPUs time to clean up
            seed, gpu_id = future_to_experiment[future]
            result = future.result()
            all_results.append(result)
            
            if result['success']:
                logger.info(f"✅ Seed {seed} (GPU {gpu_id}) completed")
            else:
                logger.error(f"❌ Seed {seed} (GPU {gpu_id}) failed: {result.get('error', 'Unknown')}")
    
    # Save summary
    successful_results = [r for r in all_results if r['success']]
    summary = {
        'config': args.config,
        'episodes': args.episodes,
        'seeds': seeds,
        'available_gpus': available_gpus,
        'max_workers': max_workers,
        'total_experiments': len(seeds),
        'successful_experiments': len(successful_results),
        'save_directory': save_dir,
        'timestamp': datetime.now().isoformat(),
        'experiment_assignments': {str(seed): gpu_id for seed, gpu_id in experiments},
        'results': all_results
    }
    
    summary_path = os.path.join(save_dir, 'experiment_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print GPU utilization summary
    gpu_usage = {}
    for result in successful_results:
        gpu_id = result.get('gpu_id', -1)
        if gpu_id not in gpu_usage:
            gpu_usage[gpu_id] = 0
        gpu_usage[gpu_id] += 1
    
    logger.info(f"🎉 Multi-seed training completed!")
    logger.info(f"✅ Successful: {len(successful_results)}/{len(seeds)} experiments")
    logger.info(f"🔧 GPU utilization: {gpu_usage}")
    logger.info(f"📁 Data saved to: {save_dir}")
    logger.info(f"📋 Summary: {summary_path}")
    
    return 0 if len(successful_results) == len(seeds) else 1


if __name__ == "__main__":
    # Set multiprocessing start method to spawn for CUDA compatibility
    mp.set_start_method('spawn', force=True)
    exit_code = main()
    sys.exit(exit_code)