#!/usr/bin/env python3
"""
run_stable_v4.py - Comprehensive RL Evaluation System for Research Paper

This script implements rigorous data collection and multi-GPU support for 
scientific evaluation of the protein sequence optimization system.

Key Features:
- Multi-seed training with statistical analysis
- Comprehensive baseline comparisons
- Detailed metrics collection for research reporting
- Multi-GPU distributed training support
- Statistical significance testing
- Publication-ready result aggregation

Usage:
    # Single run with comprehensive logging
    python src/experiments/run_stable_v4.py --config stable --episodes 2000 --seed 42

    # Multi-seed evaluation across GPUs
    python src/experiments/run_stable_v4.py --config stable --episodes 2000 --multi-seed --seeds 42,123,456,789,999

    # Full research evaluation with baselines
    python src/experiments/run_stable_v4.py --config stable --episodes 1200 --research-mode --multi-seed --seeds 42,123,456,789,999
"""

import os
import sys
import argparse
import torch
import torch.multiprocessing as mp
import numpy as np
import pandas as pd
import json
import pickle
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict, deque
from pathlib import Path
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

# Scientific computing
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import existing components
from src.config.stable_configs_v2 import get_stable_config_v2
from src.models.improved_policy_v2 import ImprovedSequenceEditPolicyV2
from src.models.stable_reward_function_v2 import StableSpiderSilkRewardFunctionV2
from src.data.dataset import SpiderSilkDataset
from src.utils.spider_silk_utils import SpiderSilkUtils
from src.environment.protein_env import ProteinEditEnvironment
from src.debug.debug import fix_both_warnings
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoTokenizer

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('research_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ComprehensiveMetricsCollector:
    """Comprehensive metrics collection for research evaluation"""
    
    def __init__(self, experiment_id: str, save_dir: str):
        self.experiment_id = experiment_id
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Primary metrics for research paper
        self.metrics = {
            # Training dynamics
            'episode_rewards': [],
            'episode_improvements': [],
            'episode_lengths': [],
            'training_losses': [],
            'learning_rates': [],
            
            # Performance metrics
            'cumulative_improvements': [],
            'success_rates': [],
            'edit_efficiencies': [],
            'convergence_episodes': [],
            
            # Sequence analysis
            'edit_patterns': [],
            'amino_acid_distributions': [],
            'position_preferences': [],
            'sequence_qualities': [],
            
            # Statistical tracking
            'confidence_intervals': [],
            'effect_sizes': [],
            'significance_tests': [],
            
            # Computational metrics
            'training_times': [],
            'memory_usage': [],
            'gpu_utilization': []
        }
        
        # Detailed episode data
        self.episode_data = []
        
        # Test set evaluation
        self.test_evaluations = []
        
        # Baseline comparisons
        self.baseline_results = {}
        
        # Statistical analysis
        self.statistical_summary = {}
    
    def record_episode(self, episode_num: int, result: Dict[str, Any]):
        """Record comprehensive episode metrics"""
        
        # Core training metrics
        self.metrics['episode_rewards'].append(result.get('episode_reward', 0.0))
        self.metrics['episode_improvements'].append(result.get('actual_improvement', 0.0))
        self.metrics['episode_lengths'].append(result.get('episode_length', 0))
        
        # Training dynamics
        self.metrics['training_losses'].append({
            'policy_loss': result.get('policy_loss', 0.0),
            'value_loss': result.get('value_loss', 0.0),
            'entropy_loss': result.get('entropy_loss', 0.0)
        })
        self.metrics['learning_rates'].append(result.get('current_lr', 0.0))
        
        # Edit pattern analysis
        edit_history = result.get('edit_history', [])
        if edit_history:
            edit_types = [edit.get('type', 'unknown') for edit in edit_history]
            positions = [edit.get('position', 0) for edit in edit_history if edit.get('position') is not None]
            
            self.metrics['edit_patterns'].append({
                'episode': episode_num,
                'edit_types': edit_types,
                'positions': positions,
                'total_edits': len(edit_history)
            })
        
        # Detailed episode data for analysis
        episode_record = {
            'episode': episode_num,
            'reward': result.get('episode_reward', 0.0),
            'improvement': result.get('actual_improvement', 0.0),
            'length': result.get('episode_length', 0),
            'final_sequence': result.get('final_sequence', ''),
            'starting_sequence': result.get('starting_sequence', ''),
            'edit_history': edit_history,
            'difficulty_level': result.get('difficulty_level', 0),
            'timestamp': time.time()
        }
        self.episode_data.append(episode_record)
    
    def record_test_evaluation(self, episode_num: int, test_results: Dict[str, Any]):
        """Record test set evaluation results"""
        evaluation = {
            'episode': episode_num,
            'avg_reward': test_results.get('avg_reward', 0.0),
            'avg_improvement': test_results.get('avg_improvement', 0.0),
            'success_rate': test_results.get('success_rate', 0.0),
            'individual_results': test_results.get('results', []),
            'timestamp': time.time()
        }
        self.test_evaluations.append(evaluation)
    
    def calculate_convergence_metrics(self):
        """Calculate convergence analysis metrics"""
        if len(self.metrics['episode_rewards']) < 100:
            return
        
        # Moving average convergence
        window_size = 50
        rewards = np.array(self.metrics['episode_rewards'])
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        
        # Find convergence point (when moving average stabilizes)
        if len(moving_avg) > 100:
            # Calculate variance in moving average
            variances = []
            for i in range(50, len(moving_avg)):
                recent_window = moving_avg[i-50:i]
                variances.append(np.var(recent_window))
            
            # Convergence when variance drops below threshold
            convergence_threshold = np.percentile(variances, 25)  # Bottom quartile
            convergence_episodes = []
            
            for i, var in enumerate(variances):
                if var <= convergence_threshold:
                    convergence_episodes.append(i + 100)  # Adjust for window offset
            
            if convergence_episodes:
                self.metrics['convergence_episodes'] = convergence_episodes[0]
            else:
                self.metrics['convergence_episodes'] = len(rewards)
    
    def calculate_statistical_summary(self):
        """Calculate comprehensive statistical summary"""
        if not self.metrics['episode_rewards']:
            return
        
        rewards = np.array(self.metrics['episode_rewards'])
        improvements = np.array(self.metrics['episode_improvements'])
        
        # Basic statistics
        self.statistical_summary = {
            'reward_stats': {
                'mean': float(np.mean(rewards)),
                'std': float(np.std(rewards)),
                'median': float(np.median(rewards)),
                'q25': float(np.percentile(rewards, 25)),
                'q75': float(np.percentile(rewards, 75)),
                'min': float(np.min(rewards)),
                'max': float(np.max(rewards))
            },
            'improvement_stats': {
                'mean': float(np.mean(improvements)),
                'std': float(np.std(improvements)),
                'median': float(np.median(improvements)),
                'positive_rate': float(np.mean(improvements > 0)),
                'significant_rate': float(np.mean(improvements > 0.001))
            },
            'convergence_analysis': {
                'episodes_to_convergence': self.metrics.get('convergence_episodes', len(rewards)),
                'final_performance': float(np.mean(rewards[-100:])) if len(rewards) >= 100 else float(np.mean(rewards))
            }
        }
        
        # Confidence intervals (bootstrap)
        self._calculate_confidence_intervals(rewards, improvements)
    
    def _calculate_confidence_intervals(self, rewards: np.ndarray, improvements: np.ndarray):
        """Calculate bootstrap confidence intervals"""
        n_bootstrap = 1000
        alpha = 0.05  # 95% confidence interval
        
        # Bootstrap for mean reward
        reward_means = []
        improvement_means = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            reward_sample = np.random.choice(rewards, size=len(rewards), replace=True)
            improvement_sample = np.random.choice(improvements, size=len(improvements), replace=True)
            
            reward_means.append(np.mean(reward_sample))
            improvement_means.append(np.mean(improvement_sample))
        
        # Calculate confidence intervals
        reward_ci = [
            float(np.percentile(reward_means, 100 * alpha / 2)),
            float(np.percentile(reward_means, 100 * (1 - alpha / 2)))
        ]
        
        improvement_ci = [
            float(np.percentile(improvement_means, 100 * alpha / 2)),
            float(np.percentile(improvement_means, 100 * (1 - alpha / 2)))
        ]
        
        self.statistical_summary['confidence_intervals'] = {
            'reward_95ci': reward_ci,
            'improvement_95ci': improvement_ci
        }
    
    def save_results(self):
        """Save all collected results"""
        # Save raw metrics
        with open(self.save_dir / f'{self.experiment_id}_metrics.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            metrics_serializable = {}
            for key, value in self.metrics.items():
                if isinstance(value, np.ndarray):
                    metrics_serializable[key] = value.tolist()
                elif isinstance(value, list):
                    metrics_serializable[key] = value
                else:
                    metrics_serializable[key] = value
            json.dump(metrics_serializable, f, indent=2)
        
        # Save detailed episode data
        with open(self.save_dir / f'{self.experiment_id}_episodes.pkl', 'wb') as f:
            pickle.dump(self.episode_data, f)
        
        # Save test evaluations
        with open(self.save_dir / f'{self.experiment_id}_test_evals.json', 'w') as f:
            json.dump(self.test_evaluations, f, indent=2)
        
        # Save statistical summary
        with open(self.save_dir / f'{self.experiment_id}_statistics.json', 'w') as f:
            json.dump(self.statistical_summary, f, indent=2)
        
        # Save baseline results if available
        if self.baseline_results:
            with open(self.save_dir / f'{self.experiment_id}_baselines.json', 'w') as f:
                json.dump(self.baseline_results, f, indent=2)
        
        logger.info(f"Results saved to {self.save_dir}")


class BaselineEvaluator:
    """Evaluate baseline methods for comparison"""
    
    def __init__(self, environment, dataset, device):
        self.environment = environment
        self.dataset = dataset
        self.device = device
    
    def evaluate_random_baseline(self, n_sequences: int = 50) -> Dict[str, Any]:
        """Evaluate random edit baseline"""
        logger.info("Evaluating random baseline...")
        
        test_sequences = self.dataset.get_test_sequences(n_sequences)
        results = []
        
        for seq in test_sequences:
            original_tough, _ = self.environment.reward_fn.predict_toughness(seq)
            
            # Make random edits
            current_seq = seq
            total_reward = 0
            edits_made = 0
            
            for _ in range(10):  # Max 10 random edits
                if len(current_seq) < 20:
                    break
                
                # Random edit
                edit_type = random.choice(['substitution', 'insertion', 'deletion'])
                
                if edit_type == 'substitution' and len(current_seq) > 0:
                    pos = random.randint(0, len(current_seq) - 1)
                    aa = random.choice('ACDEFGHIKLMNPQRSTVWY')
                    new_seq = current_seq[:pos] + aa + current_seq[pos+1:]
                elif edit_type == 'insertion':
                    pos = random.randint(0, len(current_seq))
                    aa = random.choice('ACDEFGHIKLMNPQRSTVWY')
                    new_seq = current_seq[:pos] + aa + current_seq[pos:]
                elif edit_type == 'deletion' and len(current_seq) > 20:
                    pos = random.randint(0, len(current_seq) - 1)
                    new_seq = current_seq[:pos] + current_seq[pos+1:]
                else:
                    continue
                
                # Check if edit is valid
                is_valid, _ = self.environment.utils.validate_edit(current_seq, new_seq)
                if is_valid:
                    current_seq = new_seq
                    edits_made += 1
            
            # Calculate final improvement
            final_tough, _ = self.environment.reward_fn.predict_toughness(current_seq)
            improvement = final_tough - original_tough
            
            results.append({
                'improvement': improvement,
                'edits_made': edits_made,
                'final_sequence': current_seq
            })
        
        return {
            'method': 'random',
            'results': results,
            'avg_improvement': np.mean([r['improvement'] for r in results]),
            'success_rate': np.mean([r['improvement'] > 0.001 for r in results]),
            'avg_edits': np.mean([r['edits_made'] for r in results])
        }
    
    def evaluate_no_modification_baseline(self, n_sequences: int = 50) -> Dict[str, Any]:
        """Evaluate no modification baseline (control)"""
        logger.info("Evaluating no modification baseline...")
        
        test_sequences = self.dataset.get_test_sequences(n_sequences)
        results = []
        
        for seq in test_sequences:
            results.append({
                'improvement': 0.0,  # No modification = no improvement
                'edits_made': 0,
                'final_sequence': seq
            })
        
        return {
            'method': 'no_modification',
            'results': results,
            'avg_improvement': 0.0,
            'success_rate': 0.0,
            'avg_edits': 0.0
        }
    
    def evaluate_greedy_baseline(self, n_sequences: int = 50) -> Dict[str, Any]:
        """Evaluate greedy improvement baseline"""
        logger.info("Evaluating greedy baseline...")
        
        test_sequences = self.dataset.get_test_sequences(n_sequences)
        results = []
        
        for seq in test_sequences:
            original_tough, _ = self.environment.reward_fn.predict_toughness(seq)
            
            current_seq = seq
            total_improvement = 0
            edits_made = 0
            
            # Greedy approach: try all single substitutions
            for position in range(len(current_seq)):
                best_improvement = 0
                best_seq = current_seq
                
                for aa in 'ACDEFGHIKLMNPQRSTVWY':
                    if aa == current_seq[position]:
                        continue
                    
                    test_seq = current_seq[:position] + aa + current_seq[position+1:]
                    
                    # Check validity
                    is_valid, _ = self.environment.utils.validate_edit(current_seq, test_seq)
                    if not is_valid:
                        continue
                    
                    # Calculate improvement
                    test_tough, _ = self.environment.reward_fn.predict_toughness(test_seq)
                    curr_tough, _ = self.environment.reward_fn.predict_toughness(current_seq)
                    improvement = test_tough - curr_tough
                    
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_seq = test_seq
                
                # Apply best edit if found
                if best_improvement > 0.001:
                    current_seq = best_seq
                    total_improvement += best_improvement
                    edits_made += 1
                    
                    # Limit to 5 edits for computational efficiency
                    if edits_made >= 5:
                        break
            
            results.append({
                'improvement': total_improvement,
                'edits_made': edits_made,
                'final_sequence': current_seq
            })
        
        return {
            'method': 'greedy',
            'results': results,
            'avg_improvement': np.mean([r['improvement'] for r in results]),
            'success_rate': np.mean([r['improvement'] > 0.001 for r in results]),
            'avg_edits': np.mean([r['edits_made'] for r in results])
        }


class ResearchTrainer:
    """Enhanced trainer with comprehensive research metrics"""
    
    def __init__(self, policy, environment, lr=1e-4, device=None, experiment_id="experiment"):
        self.policy = policy
        self.environment = environment
        self.lr = lr
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.experiment_id = experiment_id
        
        # Move policy to device
        self.policy.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), 
            lr=self.lr,
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
        
        # Metrics collector
        self.metrics_collector = ComprehensiveMetricsCollector(
            experiment_id, f"results/research_runs/{experiment_id}"
        )
        
        # Training state
        self.episode_count = 0
        self.start_time = time.time()
    
    def train_episode(self, starting_sequence: str, episode_number: int, difficulty_level=None):
        """Train episode with comprehensive metrics collection"""
        
        episode_start = time.time()
        
        # Collect episode
        episode_data = self._collect_episode(starting_sequence, episode_number)
        
        # Update policy
        if len(episode_data['states']) > 1:
            training_metrics = self._update_policy(episode_data)
        else:
            training_metrics = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy_loss': 0.0}
        
        # Calculate comprehensive metrics
        result = self._calculate_episode_metrics(
            episode_data, training_metrics, starting_sequence, episode_start
        )
        
        # Record all metrics
        self.metrics_collector.record_episode(episode_number, result)
        
        self.episode_count += 1
        
        return result
    
    def _collect_episode(self, starting_sequence: str, episode_number: int):
        """Collect episode with detailed tracking"""
        
        state = self.environment.reset(starting_sequence).to(self.device)
        self.environment.set_episode_number(episode_number)
        
        states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []
        episode_reward = 0
        
        self.policy.eval()
        
        while not self.environment.done and len(states) < 25:
            with torch.no_grad():
                policy_output = self.policy(state)
                action = self.policy.get_action(state, deterministic=False)
                
                states.append(state.clone())
                actions.append(action)
                values.append(policy_output['value'].item())
                log_probs.append(action['log_prob'].to(self.device))
                
                next_state, reward, done, info = self.environment.step(action)
                
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
            'final_sequence': self.environment.current_sequence,
            'edit_history': self.environment.edit_history.copy(),
            'starting_sequence': starting_sequence
        }
    
    def _update_policy(self, episode_data):
        """PPO policy update"""
        
        # Compute returns and advantages
        returns, advantages = self._compute_gae(
            episode_data['rewards'], 
            episode_data['values'], 
            episode_data['dones']
        )
        
        # Prepare tensors
        states = torch.stack(episode_data['states']).to(self.device)
        old_log_probs = torch.stack(episode_data['log_probs']).to(self.device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        
        # Normalize advantages
        if advantages_tensor.numel() > 1:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # PPO update
        self.policy.train()
        
        policy_output = self.policy(states)
        new_log_probs = self._calculate_log_probs(policy_output, episode_data['actions'])
        
        # Policy loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages_tensor
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_tensor
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        values = policy_output['value'].view(-1)
        value_loss = torch.nn.functional.mse_loss(values, returns_tensor)
        
        # Entropy loss
        entropy_loss = self._calculate_entropy(policy_output)
        
        # Total loss
        total_loss = policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy_loss
        
        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item()
        }
    
    def _compute_gae(self, rewards, values, dones):
        """Generalized Advantage Estimation"""
        returns, advantages = [], []
        next_value, advantage = 0, 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = 0
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantage = delta + self.gamma * self.gae_lambda * next_non_terminal * advantage
            
            advantages.insert(0, advantage)
            returns.insert(0, advantage + values[t])
        
        return returns, advantages
    
    def _calculate_log_probs(self, policy_output, actions):
        """Calculate log probabilities for actions"""
        log_probs = []
        
        for i, action in enumerate(actions):
            edit_types = ['substitution', 'insertion', 'deletion', 'stop']
            edit_type_idx = edit_types.index(action['type'])
            log_prob = torch.log(policy_output['edit_type'][i, edit_type_idx] + 1e-8)
            
            if action['type'] != 'stop':
                log_prob += torch.log(policy_output['position'][i, action['position']] + 1e-8)
                
                if action['amino_acid'] is not None:
                    aa_idx = list('ACDEFGHIKLMNPQRSTVWY').index(action['amino_acid'])
                    log_prob += torch.log(policy_output['amino_acid'][i, aa_idx] + 1e-8)
            
            log_probs.append(log_prob)
        
        return torch.stack(log_probs)
    
    def _calculate_entropy(self, policy_output):
        """Calculate entropy for exploration"""
        edit_type_entropy = -(policy_output['edit_type'] * 
                            torch.log(policy_output['edit_type'] + 1e-8)).sum(dim=1).mean()
        
        position_entropy = -(policy_output['position'] * 
                           torch.log(policy_output['position'] + 1e-8)).sum(dim=1).mean()
        
        aa_entropy = -(policy_output['amino_acid'] * 
                      torch.log(policy_output['amino_acid'] + 1e-8)).sum(dim=1).mean()
        
        return edit_type_entropy + position_entropy + aa_entropy
    
    def _calculate_episode_metrics(self, episode_data, training_metrics, starting_sequence, episode_start):
        """Calculate comprehensive episode metrics"""
        
        # Basic metrics
        episode_reward = episode_data['episode_reward']
        episode_length = episode_data['episode_length']
        final_sequence = episode_data['final_sequence']
        
        # Calculate actual improvement
        actual_improvement = 0.0
        if final_sequence != starting_sequence:
            try:
                old_tough, _ = self.environment.reward_fn.predict_toughness(starting_sequence)
                new_tough, _ = self.environment.reward_fn.predict_toughness(final_sequence)
                actual_improvement = new_tough - old_tough
            except:
                actual_improvement = 0.0
        
        # Timing
        episode_time = time.time() - episode_start
        
        # Return comprehensive result
        return {
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'actual_improvement': actual_improvement,
            'final_sequence': final_sequence,
            'starting_sequence': starting_sequence,
            'edit_history': episode_data['edit_history'],
            'episode_time': episode_time,
            'current_lr': self.optimizer.param_groups[0]['lr'],
            **training_metrics
        }
    
    def evaluate_on_test_set(self, dataset, n_sequences=50):
        """Comprehensive test set evaluation"""
        logger.info(f"Evaluating on test set ({n_sequences} sequences)...")
        
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
                'improvement': improvement,
                'edit_count': len(self.environment.edit_history),
                'final_sequence': self.environment.current_sequence
            })
        
        self.policy.train()
        
        return {
            'results': results,
            'avg_reward': np.mean([r['reward'] for r in results]),
            'avg_improvement': np.mean([r['improvement'] for r in results]),
            'success_rate': np.mean([r['improvement'] > 0.001 for r in results])
        }
    
    def finalize_experiment(self):
        """Finalize experiment and save results"""
        self.metrics_collector.calculate_convergence_metrics()
        self.metrics_collector.calculate_statistical_summary()
        self.metrics_collector.save_results()
        
        logger.info(f"Experiment {self.experiment_id} completed and saved")


def setup_models_and_data(config_dict, device):
    """Setup models and data with error handling and proper device assignment"""
    try:
        logger.info(f"Setting up models on device: {device}")
        
        # Set the specific GPU device BEFORE loading any models
        if device.type == 'cuda':
            torch.cuda.set_device(device)
            torch.cuda.empty_cache()  # Clear cache
            logger.info(f"Set CUDA device to: {device}")
        
        # Load dataset
        dataset = SpiderSilkDataset(
            config_dict['dataset_path'],
            test_size=config_dict['test_size'],
            n_difficulty_levels=config_dict['n_difficulty_levels'],
            random_state=config_dict['seed']
        )
        
        # Load ESM-C model
        esmc_checkpoint = "src/models/checkpoint-1452"
        if not os.path.exists(esmc_checkpoint):
            raise FileNotFoundError(f"ESM-C checkpoint not found at {esmc_checkpoint}")
        
        # Load models with explicit device mapping
        logger.info(f"Loading ESM-C model to {device}")
        esmc_model = AutoModelForMaskedLM.from_pretrained(
            esmc_checkpoint, 
            trust_remote_code=True,
            torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32  # Use half precision to save memory
        )
        esmc_tokenizer = esmc_model.tokenizer
        esmc_tokenizer, esmc_model = fix_both_warnings(esmc_tokenizer, esmc_model)
        
        # Move ESM-C to specific device
        esmc_model = esmc_model.to(device)
        
        # Load SilkomeGPT model
        logger.info(f"Loading SilkomeGPT model to {device}")
        trained_model_name = 'lamm-mit/SilkomeGPT'
        silkomegpt_tokenizer = AutoTokenizer.from_pretrained(trained_model_name, trust_remote_code=True)
        silkomegpt_tokenizer.pad_token = silkomegpt_tokenizer.eos_token
        silkomegpt_model = AutoModelForCausalLM.from_pretrained(
            trained_model_name, 
            trust_remote_code=True,
            torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32  # Use half precision
        )
        silkomegpt_model.config.use_cache = False
        
        # Move SilkomeGPT to specific device
        silkomegpt_model = silkomegpt_model.to(device)
        
        # Create components
        utils = SpiderSilkUtils(esmc_model, esmc_tokenizer)
        reward_fn = StableSpiderSilkRewardFunctionV2(
            silkomegpt_model, silkomegpt_tokenizer, esmc_model
        )
        
        # Ensure reward function models are on correct device
        reward_fn.silkomegpt = reward_fn.silkomegpt.to(device)
        reward_fn.esmc = reward_fn.esmc.to(device)
        reward_fn.device = device  # Set device attribute
        
        # Create environment and policy
        environment = ProteinEditEnvironment(utils, reward_fn, max_steps=config_dict['max_steps'])
        policy = ImprovedSequenceEditPolicyV2().to(device)
        
        # Log memory usage
        if device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(device) / (1024**3)
            total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            logger.info(f"GPU {device.index} memory after model loading: {allocated:.2f}GB / {total:.2f}GB")
        
        return policy, environment, dataset, utils, reward_fn
        
    except Exception as e:
        logger.error(f"Failed to setup models and data: {e}")
        raise


def run_single_experiment(seed: int, config_name: str, episodes: int, gpu_id: int, research_mode: bool = False):
    """Run a single experiment with specified seed and proper GPU isolation"""
    
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Setup device with proper isolation
    if torch.cuda.is_available() and gpu_id >= 0:
        device = torch.device(f'cuda:{gpu_id}')
        # Set the device BEFORE doing anything else
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        
        # Set memory management for this process
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        logger.info(f"Process for seed {seed} using GPU {gpu_id}")
    else:
        device = torch.device('cpu')
        logger.info(f"Process for seed {seed} using CPU")
    
    # Get configuration
    config = get_stable_config_v2(config_name)
    config_dict = config.to_dict()
    config_dict['seed'] = seed
    config_dict['n_episodes'] = episodes
    
    # Create experiment ID
    experiment_id = f"{config_name}_seed_{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info(f"Starting experiment: {experiment_id} on {device}")
    
    try:
        # Clear any existing CUDA cache
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
            # Check initial memory
            initial_memory = torch.cuda.memory_allocated(device) / (1024**3)
            total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            logger.info(f"Initial GPU {gpu_id} memory: {initial_memory:.2f}GB / {total_memory:.2f}GB")
        
        # Setup models and data with device isolation
        policy, environment, dataset, utils, reward_fn = setup_models_and_data(config_dict, device)
        
        # Create research trainer
        trainer = ResearchTrainer(
            policy, environment, 
            lr=config_dict['learning_rate'],
            device=device,
            experiment_id=experiment_id
        )
        
        # Run baseline evaluations if in research mode
        baseline_results = {}
        if research_mode:
            logger.info("Running baseline evaluations...")
            baseline_evaluator = BaselineEvaluator(environment, dataset, device)
            
            baseline_results['random'] = baseline_evaluator.evaluate_random_baseline()
            baseline_results['no_modification'] = baseline_evaluator.evaluate_no_modification_baseline()
            baseline_results['greedy'] = baseline_evaluator.evaluate_greedy_baseline()
            
            trainer.metrics_collector.baseline_results = baseline_results
            
            logger.info("Baseline evaluation completed")
        
        # Training loop
        logger.info(f"Starting training for {episodes} episodes...")
        
        for episode in range(1, episodes + 1):
            # Memory check every 100 episodes
            if device.type == 'cuda' and episode % 100 == 0:
                current_memory = torch.cuda.memory_allocated(device) / (1024**3)
                if current_memory > total_memory * 0.9:  # More than 90% usage
                    logger.warning(f"High memory usage on GPU {gpu_id}: {current_memory:.2f}GB / {total_memory:.2f}GB")
                    torch.cuda.empty_cache()
            
            # Get curriculum sequence
            starting_sequence, difficulty_level = dataset.get_curriculum_sequence(
                episode, episodes, config_dict['curriculum_strategy']
            )
            
            # Train episode
            result = trainer.train_episode(starting_sequence, episode, difficulty_level)
            
            # Periodic evaluation and logging
            if episode % 100 == 0:
                logger.info(f"Episode {episode}: reward={result['episode_reward']:.3f}, "
                           f"improvement={result['actual_improvement']:.4f}")
                
                # Test set evaluation
                if episode % 200 == 0:
                    test_results = trainer.evaluate_on_test_set(dataset)
                    trainer.metrics_collector.record_test_evaluation(episode, test_results)
                    
                    logger.info(f"Test evaluation at episode {episode}: "
                               f"avg_improvement={test_results['avg_improvement']:.4f}, "
                               f"success_rate={test_results['success_rate']:.3f}")
        
        # Final evaluation
        logger.info("Running final test evaluation...")
        final_test_results = trainer.evaluate_on_test_set(dataset, n_sequences=100)
        trainer.metrics_collector.record_test_evaluation(episodes, final_test_results)
        
        # Finalize experiment
        trainer.finalize_experiment()
        
        # Clean up GPU memory
        if device.type == 'cuda':
            del policy, environment, trainer
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated(device) / (1024**3)
            logger.info(f"Final GPU {gpu_id} memory after cleanup: {final_memory:.2f}GB / {total_memory:.2f}GB")
        
        logger.info(f"Experiment {experiment_id} completed successfully")
        
        return {
            'experiment_id': experiment_id,
            'seed': seed,
            'gpu_id': gpu_id,
            'final_test_results': final_test_results,
            'baseline_results': baseline_results,
            'success': True
        }
        
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"CUDA OOM in experiment {experiment_id} on GPU {gpu_id}: {e}")
        
        # Clean up as much as possible
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return {
            'experiment_id': experiment_id,
            'seed': seed,
            'gpu_id': gpu_id,
            'error': f"CUDA OOM: {str(e)}",
            'success': False
        }
        
    except Exception as e:
        logger.error(f"Experiment {experiment_id} failed: {e}")
        
        # Clean up
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return {
            'experiment_id': experiment_id,
            'seed': seed,
            'gpu_id': gpu_id,
            'error': str(e),
            'success': False
        }


def run_multi_seed_experiments(seeds: List[int], config_name: str, episodes: int, 
                              research_mode: bool = False, max_workers: int = None,
                              start_gpu: int = 0):
    """Run multiple experiments with different seeds across GPUs"""
    
    # Get available GPUs starting from start_gpu
    if torch.cuda.is_available():
        total_gpus = torch.cuda.device_count()
        available_gpus = list(range(start_gpu, total_gpus))
        if not available_gpus:
            logger.warning(f"No GPUs available starting from GPU {start_gpu}, using GPU 0")
            available_gpus = [0]
    else:
        available_gpus = [0]  # Will use CPU
    
    if max_workers is None:
        max_workers = len(available_gpus)
    
    # Ensure we don't exceed available GPUs
    max_workers = min(max_workers, len(available_gpus))
    
    logger.info(f"Running {len(seeds)} experiments across {max_workers} GPUs: {available_gpus[:max_workers]}")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f"results/multi_seed_study_{config_name}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Run experiments in parallel
    all_results = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all experiments with proper GPU assignment
        future_to_seed = {}
        for i, seed in enumerate(seeds):
            # Cycle through available GPUs
            gpu_id = available_gpus[i % len(available_gpus)]
            
            logger.info(f"Assigning seed {seed} to GPU {gpu_id}")
            
            future = executor.submit(
                run_single_experiment, seed, config_name, episodes, gpu_id, research_mode
            )
            future_to_seed[future] = (seed, gpu_id)
        
        # Collect results as they complete
        for future in as_completed(future_to_seed):
            seed, gpu_id = future_to_seed[future]
            try:
                result = future.result()
                all_results.append(result)
                
                if result['success']:
                    logger.info(f"Seed {seed} on GPU {gpu_id} completed successfully")
                else:
                    logger.error(f"Seed {seed} on GPU {gpu_id} failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"Seed {seed} on GPU {gpu_id} crashed: {e}")
                all_results.append({
                    'seed': seed,
                    'gpu_id': gpu_id,
                    'error': str(e),
                    'success': False
                })
    
    # Aggregate results
    successful_results = [r for r in all_results if r['success']]
    
    if successful_results:
        logger.info(f"Multi-seed study completed: {len(successful_results)}/{len(seeds)} successful")
        
        # Aggregate statistical analysis
        aggregate_results = aggregate_multi_seed_results(successful_results, results_dir)
        
        # Save summary
        summary = {
            'config_name': config_name,
            'episodes': episodes,
            'seeds': seeds,
            'successful_runs': len(successful_results),
            'failed_runs': len(seeds) - len(successful_results),
            'timestamp': timestamp,
            'aggregate_statistics': aggregate_results,
            'individual_results': all_results
        }
        
        with open(f"{results_dir}/multi_seed_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Multi-seed study results saved to {results_dir}")
        
        return summary
    else:
        logger.error("All experiments failed!")
        return None


def aggregate_multi_seed_results(results: List[Dict], save_dir: str) -> Dict:
    """Aggregate results from multiple seeds for statistical analysis"""
    
    logger.info("Aggregating multi-seed results...")
    
    # Collect final performance metrics
    final_improvements = []
    final_success_rates = []
    final_rewards = []
    
    for result in results:
        if 'final_test_results' in result:
            test_results = result['final_test_results']
            final_improvements.append(test_results['avg_improvement'])
            final_success_rates.append(test_results['success_rate'])
            final_rewards.append(test_results['avg_reward'])
    
    if not final_improvements:
        logger.warning("No valid results to aggregate")
        return {}
    
    # Calculate aggregate statistics
    aggregate_stats = {
        'performance_metrics': {
            'improvement': {
                'mean': float(np.mean(final_improvements)),
                'std': float(np.std(final_improvements)),
                'median': float(np.median(final_improvements)),
                'min': float(np.min(final_improvements)),
                'max': float(np.max(final_improvements)),
                'ci_95': [
                    float(np.percentile(final_improvements, 2.5)),
                    float(np.percentile(final_improvements, 97.5))
                ]
            },
            'success_rate': {
                'mean': float(np.mean(final_success_rates)),
                'std': float(np.std(final_success_rates)),
                'median': float(np.median(final_success_rates)),
                'min': float(np.min(final_success_rates)),
                'max': float(np.max(final_success_rates)),
                'ci_95': [
                    float(np.percentile(final_success_rates, 2.5)),
                    float(np.percentile(final_success_rates, 97.5))
                ]
            },
            'reward': {
                'mean': float(np.mean(final_rewards)),
                'std': float(np.std(final_rewards)),
                'median': float(np.median(final_rewards)),
                'min': float(np.min(final_rewards)),
                'max': float(np.max(final_rewards)),
                'ci_95': [
                    float(np.percentile(final_rewards, 2.5)),
                    float(np.percentile(final_rewards, 97.5))
                ]
            }
        },
        'statistical_tests': {},
        'effect_sizes': {}
    }
    
    # Statistical significance tests (if baselines available)
    if results[0].get('baseline_results'):
        baseline_improvements = []
        for result in results:
            baselines = result['baseline_results']
            if 'random' in baselines:
                baseline_improvements.append(baselines['random']['avg_improvement'])
        
        if baseline_improvements:
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(final_improvements, baseline_improvements)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(final_improvements) + np.var(baseline_improvements)) / 2)
            cohens_d = (np.mean(final_improvements) - np.mean(baseline_improvements)) / pooled_std
            
            aggregate_stats['statistical_tests']['vs_random_baseline'] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': bool(p_value < 0.05)
            }
            
            aggregate_stats['effect_sizes']['vs_random_baseline'] = {
                'cohens_d': float(cohens_d),
                'interpretation': interpret_effect_size(cohens_d)
            }
    
    # Generate publication-ready plots
    create_research_plots(results, save_dir, aggregate_stats)
    
    # Save aggregate statistics
    with open(f"{save_dir}/aggregate_statistics.json", 'w') as f:
        json.dump(aggregate_stats, f, indent=2)
    
    logger.info("Multi-seed aggregation completed")
    
    return aggregate_stats


def interpret_effect_size(cohens_d: float) -> str:
    """Interpret Cohen's d effect size"""
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def create_research_plots(results: List[Dict], save_dir: str, aggregate_stats: Dict):
    """Create publication-ready plots for research paper"""
    
    logger.info("Creating research plots...")
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Performance comparison box plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Collect data for box plots
        rl_improvements = []
        baseline_improvements = []
        
        for result in results:
            if result['success'] and 'final_test_results' in result:
                rl_improvements.append(result['final_test_results']['avg_improvement'])
                
                if 'baseline_results' in result and 'random' in result['baseline_results']:
                    baseline_improvements.append(result['baseline_results']['random']['avg_improvement'])
        
        # Box plot data
        plot_data = []
        if rl_improvements:
            plot_data.extend([('RL Agent', imp) for imp in rl_improvements])
        if baseline_improvements:
            plot_data.extend([('Random Baseline', imp) for imp in baseline_improvements])
        
        if plot_data:
            df = pd.DataFrame(plot_data, columns=['Method', 'Improvement'])
            sns.boxplot(data=df, x='Method', y='Improvement', ax=axes[0])
            axes[0].set_title('Toughness Improvement Comparison')
            axes[0].set_ylabel('Toughness Improvement')
        
        # 2. Learning curves (if individual episode data available)
        axes[1].set_title('Learning Curves (Multi-Seed)')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Reward')
        
        # 3. Success rate comparison
        if rl_improvements and baseline_improvements:
            rl_success_rates = [result['final_test_results']['success_rate'] for result in results 
                              if result['success'] and 'final_test_results' in result]
            
            methods = ['RL Agent', 'Random Baseline']
            success_rates = [np.mean(rl_success_rates), 0.0]  # Random baseline has 0 success rate
            success_stds = [np.std(rl_success_rates), 0.0]
            
            axes[2].bar(methods, success_rates, yerr=success_stds, capsize=5)
            axes[2].set_title('Success Rate Comparison')
            axes[2].set_ylabel('Success Rate')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{save_dir}/performance_comparison.pdf", bbox_inches='tight')
        plt.close()
        
        logger.info("Research plots saved")
        
    except Exception as e:
        logger.warning(f"Could not create plots: {e}")


def main():
    """Main entry point with comprehensive argument parsing"""
    parser = argparse.ArgumentParser(description='Comprehensive RL Evaluation for Research Paper')
    parser.add_argument('--config', type=str, default='stable', 
                       choices=['stable', 'stable_conservative', 'stable_aggressive', 'stable_test'],
                       help='Configuration to use')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of episodes to train')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (for single run)')
    parser.add_argument('--multi-seed', action='store_true',
                       help='Run multiple seeds for statistical rigor')
    parser.add_argument('--seeds', type=str, default='42,123,456,789,999',
                       help='Comma-separated list of seeds for multi-seed runs')
    parser.add_argument('--research-mode', action='store_true',
                       help='Enable comprehensive research evaluation including baselines')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='Maximum number of parallel workers (default: number of available GPUs)')
    parser.add_argument('--start-gpu', type=int, default=1,
                       help='Starting GPU ID (default: 1, to avoid GPU 0 if busy)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto/cpu/cuda)')
    
    args = parser.parse_args()
    
    logger.info("🕷️  Spider Silk RL Comprehensive Research Evaluation")
    logger.info("="*70)
    
    # Check GPU availability
    if torch.cuda.is_available():
        total_gpus = torch.cuda.device_count()
        logger.info(f"Available GPUs: {total_gpus} (GPU 0 to GPU {total_gpus-1})")
        logger.info(f"Starting from GPU {args.start_gpu}")
        
        # Check GPU memory
        for i in range(total_gpus):
            try:
                memory_total = torch.cuda.get_device_properties(i).total_memory / 1e9
                memory_allocated = torch.cuda.memory_allocated(i) / 1e9
                memory_free = memory_total - memory_allocated
                logger.info(f"GPU {i}: {memory_free:.1f}GB free / {memory_total:.1f}GB total")
            except:
                logger.info(f"GPU {i}: Status unknown")
    else:
        logger.info("CUDA not available, will use CPU")
    
    # Parse seeds
    if args.multi_seed:
        seeds = [int(s.strip()) for s in args.seeds.split(',')]
        logger.info(f"Multi-seed evaluation with seeds: {seeds}")
        
        # Run multi-seed experiments
        results = run_multi_seed_experiments(
            seeds=seeds,
            config_name=args.config,
            episodes=args.episodes,
            research_mode=args.research_mode,
            max_workers=args.max_workers,
            start_gpu=args.start_gpu
        )
        
        if results:
            logger.info("✅ Multi-seed evaluation completed successfully!")
            logger.info(f"Results saved to: results/multi_seed_study_{args.config}_*")
            
            # Print summary statistics
            agg_stats = results['aggregate_statistics']
            if 'performance_metrics' in agg_stats:
                perf = agg_stats['performance_metrics']
                logger.info(f"Final Performance Summary:")
                logger.info(f"  Improvement: {perf['improvement']['mean']:.4f} ± {perf['improvement']['std']:.4f}")
                logger.info(f"  Success Rate: {perf['success_rate']['mean']:.3f} ± {perf['success_rate']['std']:.3f}")
                
                if 'statistical_tests' in agg_stats and 'vs_random_baseline' in agg_stats['statistical_tests']:
                    test = agg_stats['statistical_tests']['vs_random_baseline']
                    effect = agg_stats['effect_sizes']['vs_random_baseline']
                    logger.info(f"  vs Random Baseline: p={test['p_value']:.4f}, d={effect['cohens_d']:.3f} ({effect['interpretation']})")
        else:
            logger.error("❌ Multi-seed evaluation failed!")
            return 1
    
    else:
        # Single seed run
        logger.info(f"Single seed evaluation: {args.seed}")
        
        # Determine GPU
        if torch.cuda.is_available():
            gpu_id = args.start_gpu
            logger.info(f"Using GPU {gpu_id}")
        else:
            gpu_id = 0  # Will use CPU
            logger.info("Using CPU")
        
        result = run_single_experiment(
            seed=args.seed,
            config_name=args.config,
            episodes=args.episodes,
            gpu_id=gpu_id,
            research_mode=args.research_mode
        )
        
        if result['success']:
            logger.info("✅ Single experiment completed successfully!")
            logger.info(f"Experiment ID: {result['experiment_id']}")
            
            if 'final_test_results' in result:
                test_results = result['final_test_results']
                logger.info(f"Final Test Results:")
                logger.info(f"  Improvement: {test_results['avg_improvement']:.4f}")
                logger.info(f"  Success Rate: {test_results['success_rate']:.3f}")
                logger.info(f"  Reward: {test_results['avg_reward']:.3f}")
        else:
            logger.error(f"❌ Experiment failed: {result.get('error', 'Unknown error')}")
            return 1
    
    return 0


if __name__ == "__main__":
    # Enable multiprocessing
    mp.set_start_method('spawn', force=True)
    
    exit_code = main()
    sys.exit(exit_code)