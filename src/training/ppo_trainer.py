"""
Proximal Policy Optimization (PPO) Trainer

This module implements PPO training for protein sequence editing tasks,
supporting both single-GPU and distributed multi-GPU training.
"""

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from abc import ABC, abstractmethod

from .distributed_utils import (
    is_main_process, get_rank, get_world_size, 
    all_reduce_tensor, DistributedMetrics
)

logger = logging.getLogger(__name__)


class BasePPOTrainer(ABC):
    """Base class for PPO trainers"""
    
    def __init__(self, 
                 policy,
                 environment,
                 lr: float = 3e-4,
                 clip_epsilon: float = 0.2,
                 value_coeff: float = 0.5,
                 entropy_coeff: float = 0.01,
                 max_grad_norm: float = 0.5,
                 device: Optional[torch.device] = None):
        
        self.environment = environment
        self.lr = lr
        self.clip_epsilon = clip_epsilon
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        
        # Device setup
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        # Policy setup (to be implemented by subclasses)
        self.policy = None
        self.optimizer = None
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.kl_divergences = []
        
        # Episode tracking
        self.total_episodes = 0
        self.total_steps = 0
    
    @abstractmethod
    def _setup_policy(self, policy):
        """Setup policy and optimizer (to be implemented by subclasses)"""
        pass
    
    def collect_episode(self, starting_sequence: str, episode_number: int, 
                       difficulty_level: Optional[int] = None) -> Dict[str, Any]:
        """
        Collect a complete episode of experience
        
        Args:
            starting_sequence: Initial protein sequence
            episode_number: Current episode number
            difficulty_level: Difficulty level of the sequence (for curriculum)
            
        Returns:
            Dictionary containing episode data
        """
        # Reset environment
        raw_state = self.environment.reset(starting_sequence)
        state = raw_state.to(self.device)
        self.environment.set_episode_number(episode_number)
        
        # Episode buffers
        states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []
        episode_reward = 0
        
        while not self.environment.done:
            # Get policy output
            with torch.no_grad():
                policy_output = self._forward_policy(state)
            
            # Sample action
            action = self.environment.action_space.sample_action(
                policy_output, len(self.environment.current_sequence)
            )
            
            # Store experience
            states.append(state.clone())
            actions.append(action)
            values.append(policy_output['value'].item())
            log_probs.append(action['log_prob'].to(self.device))
            
            # Take environment step
            next_state, reward, done, info = self.environment.step(action)
            rewards.append(reward)
            dones.append(done)
            episode_reward += reward
            
            # Update state
            state = next_state.to(self.device)
            
            # Safety check for infinite episodes
            if len(states) > 200:
                logger.warning("Episode exceeded 200 steps, terminating")
                break
        
        episode_data = {
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
            'difficulty_level': difficulty_level,
            'starting_sequence': starting_sequence
        }
        
        return episode_data
    
    def compute_returns_and_advantages(self, 
                                     rewards: List[float], 
                                     values: List[float], 
                                     dones: List[bool],
                                     gamma: float = 0.99, 
                                     gae_lambda: float = 0.95) -> Tuple[List[float], List[float]]:
        """
        Compute discounted returns and GAE advantages
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            
        Returns:
            Tuple of (returns, advantages)
        """
        returns = []
        advantages = []
        
        # Bootstrap value (0 for terminal states)
        next_value = 0
        advantage = 0
        
        # Compute advantages using GAE
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = 0
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_value = values[t + 1]
            
            # TD error
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            
            # GAE advantage
            advantage = delta + gamma * gae_lambda * next_non_terminal * advantage
            advantages.insert(0, advantage)
            
            # Return
            return_val = advantage + values[t]
            returns.insert(0, return_val)
        
        return returns, advantages
    
    def update_policy(self, batch: Dict[str, Any], epochs: int = 4) -> Dict[str, float]:
        """
        Update policy using PPO
        
        Args:
            batch: Batch of episode data
            epochs: Number of optimization epochs
            
        Returns:
            Dictionary of training metrics
        """
        # Prepare batch data
        states = torch.stack(batch['states']).to(self.device)
        old_log_probs = torch.stack(batch['log_probs']).to(self.device).detach()
        returns = torch.tensor(batch['returns'], dtype=torch.float32, device=self.device)
        advantages = torch.tensor(batch['advantages'], dtype=torch.float32, device=self.device)
        
        # Normalize advantages
        if advantages.numel() > 1:
            std = advantages.std(unbiased=False)
            if std > 1e-6:
                advantages = (advantages - advantages.mean()) / std
            else:
                advantages = advantages - advantages.mean()
        
        # Training metrics
        metrics = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy_loss': 0.0,
            'kl_divergence': 0.0,
            'clipfrac': 0.0
        }
        
        # PPO epochs
        for epoch in range(epochs):
            # Forward pass
            policy_output = self._forward_policy(states)
            
            # Recalculate log probabilities
            new_log_probs = self._calculate_log_probs(policy_output, batch['actions'])
            
            # PPO loss components
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Policy loss (clipped)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            values = policy_output['value'].view(-1)
            value_loss = F.mse_loss(values, returns.view_as(values))
            
            # Entropy loss (for exploration)
            entropy_loss = self._calculate_entropy(policy_output)
            
            # Total loss
            total_loss = (policy_loss + 
                         self.value_coeff * value_loss - 
                         self.entropy_coeff * entropy_loss)
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            
            # Update metrics
            with torch.no_grad():
                kl_div = torch.mean(old_log_probs - new_log_probs).item()
                clipfrac = torch.mean((torch.abs(ratio - 1) > self.clip_epsilon).float()).item()
                
                metrics['policy_loss'] += policy_loss.item()
                metrics['value_loss'] += value_loss.item()
                metrics['entropy_loss'] += entropy_loss.item()
                metrics['kl_divergence'] += kl_div
                metrics['clipfrac'] += clipfrac
        
        # Average over epochs
        for key in metrics:
            metrics[key] /= epochs
        
        # Store metrics
        self.policy_losses.append(metrics['policy_loss'])
        self.value_losses.append(metrics['value_loss'])
        self.entropy_losses.append(metrics['entropy_loss'])
        self.kl_divergences.append(metrics['kl_divergence'])
        
        return metrics
    
    def train_episode(self, starting_sequence: str, episode_number: int,
                     difficulty_level: Optional[int] = None) -> Dict[str, Any]:
        """
        Train one complete episode
        
        Args:
            starting_sequence: Initial protein sequence
            episode_number: Current episode number
            difficulty_level: Difficulty level (for curriculum learning)
            
        Returns:
            Episode data and training metrics
        """
        # Set policy to training mode
        self.policy.train()
        
        # Collect episode
        episode_data = self.collect_episode(starting_sequence, episode_number, difficulty_level)
        
        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages(
            episode_data['rewards'],
            episode_data['values'],
            episode_data['dones']
        )
        
        # Add to batch
        batch = episode_data.copy()
        batch['returns'] = returns
        batch['advantages'] = advantages
        
        # Update policy
        training_metrics = self.update_policy(batch, epochs=1)
        
        # Update episode tracking
        self.episode_rewards.append(episode_data['episode_reward'])
        self.episode_lengths.append(episode_data['episode_length'])
        self.total_episodes += 1
        self.total_steps += episode_data['episode_length']
        
        # Combine episode data with training metrics
        result = episode_data.copy()
        result.update(training_metrics)
        
        return result
    
    @abstractmethod
    def _forward_policy(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through policy (to be implemented by subclasses)"""
        pass
    
    def _calculate_log_probs(self, policy_output: Dict[str, torch.Tensor], 
                           actions: List[Dict[str, Any]]) -> torch.Tensor:
        """Calculate log probabilities for taken actions"""
        log_probs = []
        
        for i, action in enumerate(actions):
            # Edit type log probability
            edit_type_idx = ['substitution', 'insertion', 'deletion', 'stop'].index(action['type'])
            log_prob = torch.log(policy_output['edit_type'][i, edit_type_idx] + 1e-8)
            
            if action['type'] != 'stop':
                # Position log probability
                log_prob += torch.log(policy_output['position'][i, action['position']] + 1e-8)
                
                # Amino acid log probability (if applicable)
                if action['amino_acid'] is not None:
                    aa_idx = list('ACDEFGHIKLMNPQRSTVWY').index(action['amino_acid'])
                    log_prob += torch.log(policy_output['amino_acid'][i, aa_idx] + 1e-8)
            
            log_probs.append(log_prob)
        
        return torch.stack(log_probs)
    
    def _calculate_entropy(self, policy_output: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate entropy for exploration regularization"""
        edit_type_entropy = -(policy_output['edit_type'] * 
                             torch.log(policy_output['edit_type'] + 1e-8)).sum(dim=1).mean()
        
        position_entropy = -(policy_output['position'] * 
                           torch.log(policy_output['position'] + 1e-8)).sum(dim=1).mean()
        
        aa_entropy = -(policy_output['amino_acid'] * 
                      torch.log(policy_output['amino_acid'] + 1e-8)).sum(dim=1).mean()
        
        return edit_type_entropy + position_entropy + aa_entropy
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics"""
        if not self.episode_rewards:
            return {}
        
        return {
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'avg_episode_reward': np.mean(self.episode_rewards),
            'avg_episode_length': np.mean(self.episode_lengths),
            'recent_avg_reward': np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards),
            'best_episode_reward': max(self.episode_rewards),
            'avg_policy_loss': np.mean(self.policy_losses) if self.policy_losses else 0.0,
            'avg_value_loss': np.mean(self.value_losses) if self.value_losses else 0.0,
            'avg_kl_divergence': np.mean(self.kl_divergences) if self.kl_divergences else 0.0
        }


class PPOTrainer(BasePPOTrainer):
    """Single-GPU PPO Trainer"""
    
    def __init__(self, policy, environment, **kwargs):
        super().__init__(policy, environment, **kwargs)
        self._setup_policy(policy)
    
    def _setup_policy(self, policy):
        """Setup policy and optimizer for single-GPU training"""
        self.policy = policy.to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        
        logger.info(f"Single-GPU PPO trainer initialized on {self.device}")
    
    def _forward_policy(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through policy"""
        return self.policy(state)


class DistributedPPOTrainer(BasePPOTrainer):
    """Multi-GPU Distributed PPO Trainer"""
    
    def __init__(self, policy, environment, rank: int = 0, world_size: int = 1, **kwargs):
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = is_main_process(rank)
        
        # Override device with rank-specific GPU
        kwargs['device'] = torch.device(f'cuda:{rank}')
        
        super().__init__(policy, environment, **kwargs)
        self._setup_policy(policy)
        
        # Distributed metrics aggregation
        self.distributed_metrics = DistributedMetrics()
    
    def _setup_policy(self, policy):
        """Setup policy and optimizer for distributed training"""
        # Move policy to correct GPU
        self.policy = policy.to(self.device)
        
        # Wrap with DistributedDataParallel
        if self.world_size > 1:
            self.policy = DDP(self.policy, device_ids=[self.rank])
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        
        if self.is_main_process:
            logger.info(f"Distributed PPO trainer initialized: rank={self.rank}/{self.world_size}")
    
    def _forward_policy(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through policy (handling DDP wrapper)"""
        if self.world_size > 1:
            return self.policy.module(state)
        else:
            return self.policy(state)
    
    def update_policy(self, batch: Dict[str, Any], epochs: int = 4) -> Dict[str, float]:
        """Update policy with distributed synchronization"""
        # Call parent update method
        metrics = super().update_policy(batch, epochs)
        
        # Aggregate metrics across all processes
        self.distributed_metrics.update(metrics)
        aggregated_metrics = self.distributed_metrics.aggregate('mean')
        self.distributed_metrics.reset()
        
        return aggregated_metrics
    
    def train_episode(self, starting_sequence: str, episode_number: int,
                     difficulty_level: Optional[int] = None) -> Dict[str, Any]:
        """Train episode with distributed logging"""
        result = super().train_episode(starting_sequence, episode_number, difficulty_level)
        
        # Only main process logs detailed results
        if not self.is_main_process:
            # Other processes return minimal info
            return {
                'episode_reward': result['episode_reward'],
                'episode_length': result['episode_length'],
                'rank': self.rank
            }
        
        return result
    
    def save_checkpoint(self, path: str, episode: int, additional_data: Optional[Dict] = None):
        """Save checkpoint (only main process)"""
        if not self.is_main_process:
            return
        
        checkpoint = {
            'episode': episode,
            'model_state_dict': self.policy.module.state_dict() if hasattr(self.policy, 'module') else self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.get_training_stats(),
            'rank': self.rank,
            'world_size': self.world_size
        }
        
        if additional_data:
            checkpoint.update(additional_data)
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> int:
        """Load checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model state
        if hasattr(self.policy, 'module'):
            self.policy.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.policy.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training stats
        if 'training_stats' in checkpoint:
            stats = checkpoint['training_stats']
            self.total_episodes = stats.get('total_episodes', 0)
            self.total_steps = stats.get('total_steps', 0)
        
        episode = checkpoint.get('episode', 0)
        
        if self.is_main_process:
            logger.info(f"Checkpoint loaded from {path}, episode {episode}")
        
        return episode


def create_trainer(policy, environment, distributed: bool = False, 
                  rank: int = 0, world_size: int = 1, **kwargs) -> BasePPOTrainer:
    """
    Factory function to create appropriate trainer
    
    Args:
        policy: Policy network
        environment: Training environment
        distributed: Whether to use distributed training
        rank: Process rank (for distributed)
        world_size: Total number of processes (for distributed)
        **kwargs: Additional trainer arguments
        
    Returns:
        PPO trainer instance
    """
    if distributed and world_size > 1:
        return DistributedPPOTrainer(
            policy, environment, rank=rank, world_size=world_size, **kwargs
        )
    else:
        return PPOTrainer(policy, environment, **kwargs)