"""
Proximal Policy Optimization (PPO) Trainer

This module implements PPO training for protein sequence editing tasks,
supporting both single-GPU and distributed multi-GPU training.
"""

"""
Proximal Policy Optimization (PPO) Trainer

This module implements PPO training for protein sequence editing tasks,
supporting both single-GPU and distributed multi-GPU training with robust
numerical stability checks.
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
    """Base class for PPO trainers with numerical stability"""
    
    def __init__(self, 
                 policy,
                 environment,
                 lr: float = 3e-4,
                 clip_epsilon: float = 0.2,
                 value_coeff: float = 0.5,
                 entropy_coeff: float = 0.1,
                 max_grad_norm: float = 0.4,
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
        self.difficulty_levels = []
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.kl_divergences = []
        self.grad_norms = []
        
        # Episode tracking
        self.total_episodes = 0
        self.total_steps = 0
        self.nan_count = 0  # Track NaN occurrences
    
    @abstractmethod
    def _setup_policy(self, policy):
        """Setup policy and optimizer (to be implemented by subclasses)"""
        pass
    
    def _validate_tensor(self, tensor: torch.Tensor, name: str) -> bool:
        """Validate tensor for NaN/Inf values"""
        if torch.isnan(tensor).any():
            logger.warning(f"NaN detected in {name}")
            self.nan_count += 1
            return False
        if torch.isinf(tensor).any():
            logger.warning(f"Inf detected in {name}")
            self.nan_count += 1
            return False
        return True
    
    def _safe_mean(self, values: List[float]) -> float:
        """Safely compute mean, handling NaN values"""
        valid_values = [v for v in values if not (np.isnan(v) or np.isinf(v))]
        if not valid_values:
            return 0.0
        return np.mean(valid_values)
    
    def collect_episode(self, starting_sequence: str, episode_number: int, 
                       difficulty_level: Optional[int] = None) -> Dict[str, Any]:
        """
        Collect a complete episode of experience with safety checks
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
                
                # Validate policy output
                for key, tensor in policy_output.items():
                    if isinstance(tensor, torch.Tensor):
                        if not self._validate_tensor(tensor, f"policy_output_{key}"):
                            logger.error(f"Invalid policy output in episode {episode_number}, terminating episode")
                            break
                else:
                    # This else belongs to the for loop - executes if no break occurred
                    
                    # Sample action
                    try:
                        action = self.environment.action_space.sample_action(
                            policy_output, len(self.environment.current_sequence)
                        )
                        
                        # Validate action log_prob
                        if 'log_prob' in action and isinstance(action['log_prob'], torch.Tensor):
                            if not self._validate_tensor(action['log_prob'], "action_log_prob"):
                                logger.warning("Invalid action log_prob, setting to zero")
                                action['log_prob'] = torch.tensor(0.0, device=self.device)
                        
                    except Exception as e:
                        logger.error(f"Action sampling failed: {e}")
                        break
                    
                    # Store experience
                    states.append(state.clone())
                    actions.append(action)
                    
                    # Safe value extraction
                    value_item = policy_output['value'].item()
                    if np.isnan(value_item) or np.isinf(value_item):
                        logger.warning("Invalid value prediction, using 0.0")
                        value_item = 0.0
                    values.append(value_item)
                    
                    log_probs.append(action['log_prob'].to(self.device))
                    
                    # Take environment step
                    next_state, reward, done, info = self.environment.step(action)
                    
                    # Validate reward
                    if np.isnan(reward) or np.isinf(reward):
                        logger.warning(f"Invalid reward {reward}, using -0.1")
                        reward = -0.1
                    
                    rewards.append(reward)
                    dones.append(done)
                    episode_reward += reward
                    
                    # Update state
                    state = next_state.to(self.device)
                    
                    # Safety check for infinite episodes
                    if len(states) > 200:
                        logger.warning("Episode exceeded 200 steps, terminating")
                        break
                
                # If we broke out of the while loop due to invalid policy output
                if not states:
                    logger.error("No valid states collected, creating minimal episode")
                    # Create a minimal valid episode
                    states = [raw_state.to(self.device)]
                    actions = [{'type': 'stop', 'position': 0, 'amino_acid': None, 'log_prob': torch.tensor(0.0, device=self.device)}]
                    rewards = [-1.0]
                    values = [0.0]
                    log_probs = [torch.tensor(0.0, device=self.device)]
                    dones = [True]
                    episode_reward = -1.0
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
        Compute discounted returns and GAE advantages with safety checks
        """
        returns = []
        advantages = []
        
        # Validate inputs
        for i, (r, v) in enumerate(zip(rewards, values)):
            if np.isnan(r) or np.isinf(r):
                logger.warning(f"Invalid reward at step {i}: {r}")
                rewards[i] = -0.1
            if np.isnan(v) or np.isinf(v):
                logger.warning(f"Invalid value at step {i}: {v}")
                values[i] = 0.0
        
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
            
            # Validate computed values
            if np.isnan(advantage) or np.isinf(advantage):
                logger.warning(f"Invalid advantage at step {t}: {advantage}")
                advantage = 0.0
            
            advantages.insert(0, advantage)
            
            # Return
            return_val = advantage + values[t]
            if np.isnan(return_val) or np.isinf(return_val):
                logger.warning(f"Invalid return at step {t}: {return_val}")
                return_val = values[t]
            
            returns.insert(0, return_val)
        
        return returns, advantages
    
    def update_policy(self, batch: Dict[str, Any], epochs: int = 4) -> Dict[str, float]:
        """
        Update policy using PPO with comprehensive NaN protection
        """
        # Prepare batch data
        states = torch.stack(batch['states']).to(self.device)
        old_log_probs = torch.stack(batch['log_probs']).to(self.device).detach()
        returns = torch.tensor(batch['returns'], dtype=torch.float32, device=self.device)
        advantages = torch.tensor(batch['advantages'], dtype=torch.float32, device=self.device)
        
        # Validate inputs
        if not all(self._validate_tensor(t, name) for t, name in [
            (states, "states"), (old_log_probs, "old_log_probs"), 
            (returns, "returns"), (advantages, "advantages")
        ]):
            logger.error("Invalid input tensors, skipping policy update")
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy_loss': 0.0, 'kl_divergence': 0.0, 'clipfrac': 0.0}
        
        # Normalize advantages
        if advantages.numel() > 1:
            std = advantages.std(unbiased=False)
            if std > 1e-6 and not torch.isnan(std):
                advantages = (advantages - advantages.mean()) / std
            else:
                advantages = advantages - advantages.mean()
        
        # Validate normalized advantages
        if not self._validate_tensor(advantages, "normalized_advantages"):
            logger.error("Invalid normalized advantages, skipping update")
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy_loss': 0.0, 'kl_divergence': 0.0, 'clipfrac': 0.0}
        
        # Training metrics
        metrics = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy_loss': 0.0,
            'kl_divergence': 0.0,
            'clipfrac': 0.0,
            'grad_norm': 0.0
        }
        
        successful_epochs = 0
        
        # PPO epochs
        for epoch in range(epochs):
            # Forward pass
            try:
                policy_output = self._forward_policy(states)
                
                # Validate policy outputs
                valid_outputs = True
                for key, tensor in policy_output.items():
                    if isinstance(tensor, torch.Tensor):
                        if not self._validate_tensor(tensor, f"policy_output_{key}_epoch_{epoch}"):
                            valid_outputs = False
                            break
                
                if not valid_outputs:
                    logger.warning(f"Invalid policy outputs in epoch {epoch}, skipping")
                    continue
                
                # Recalculate log probabilities
                new_log_probs = self._calculate_log_probs(policy_output, batch['actions'])
                
                if not self._validate_tensor(new_log_probs, f"new_log_probs_epoch_{epoch}"):
                    logger.warning(f"Invalid new log probs in epoch {epoch}, skipping")
                    continue
                
                # PPO loss components
                ratio = torch.exp(new_log_probs - old_log_probs)
                
                # Clamp ratio to prevent extreme values
                ratio = torch.clamp(ratio, 0.1, 10.0)
                
                if not self._validate_tensor(ratio, f"ratio_epoch_{epoch}"):
                    logger.warning(f"Invalid ratio in epoch {epoch}, skipping")
                    continue
                
                # Policy loss (clipped)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                values = policy_output['value'].view(-1)
                value_loss = F.mse_loss(values, returns.view_as(values))
                
                # Entropy loss (for exploration)
                entropy_loss = self._calculate_entropy(policy_output)
                
                # Validate individual losses
                if not all(self._validate_tensor(loss, f"{name}_epoch_{epoch}") for loss, name in [
                    (policy_loss, "policy_loss"), (value_loss, "value_loss"), (entropy_loss, "entropy_loss")
                ]):
                    logger.warning(f"Invalid losses in epoch {epoch}, skipping")
                    continue
                
                # Total loss
                total_loss = (policy_loss + 
                             self.value_coeff * value_loss - 
                             self.entropy_coeff * entropy_loss)
                
                if not self._validate_tensor(total_loss, f"total_loss_epoch_{epoch}"):
                    logger.warning(f"Invalid total loss in epoch {epoch}, skipping")
                    continue
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Enhanced gradient clipping with NaN check
                grad_norm = 0.0
                if self.max_grad_norm > 0:
                    total_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    grad_norm = total_norm.item()
                    
                    if np.isnan(grad_norm) or np.isinf(grad_norm):
                        logger.warning(f"Invalid gradient norm: {grad_norm}, skipping update")
                        continue
                
                # Check if gradients are valid before stepping
                valid_gradients = True
                for name, param in self.policy.named_parameters():
                    if param.grad is not None and not self._validate_tensor(param.grad, f"grad_{name}"):
                        valid_gradients = False
                        break
                
                if not valid_gradients:
                    logger.warning(f"Invalid gradients in epoch {epoch}, skipping optimizer step")
                    continue
                
                self.optimizer.step()
                successful_epochs += 1
                
                # Update metrics
                with torch.no_grad():
                    kl_div = torch.mean(old_log_probs - new_log_probs).item()
                    clipfrac = torch.mean((torch.abs(ratio - 1) > self.clip_epsilon).float()).item()
                    
                    # Validate metrics
                    if not all(np.isfinite(x) for x in [kl_div, clipfrac]):
                        logger.warning(f"Invalid metrics in epoch {epoch}")
                        continue
                    
                    metrics['policy_loss'] += policy_loss.item()
                    metrics['value_loss'] += value_loss.item()
                    metrics['entropy_loss'] += entropy_loss.item()
                    metrics['kl_divergence'] += kl_div
                    metrics['clipfrac'] += clipfrac
                    metrics['grad_norm'] += grad_norm
            
            except Exception as e:
                logger.error(f"Exception in epoch {epoch}: {e}")
                continue
        
        # Average over successful epochs
        if successful_epochs > 0:
            for key in metrics:
                metrics[key] /= successful_epochs
        else:
            logger.error("No successful epochs in policy update!")
        
        # Store metrics
        self.policy_losses.append(metrics['policy_loss'])
        self.value_losses.append(metrics['value_loss'])
        self.entropy_losses.append(metrics['entropy_loss'])
        self.kl_divergences.append(metrics['kl_divergence'])
        self.grad_norms.append(metrics['grad_norm'])
        
        return metrics
    
    def train_episode(self, starting_sequence: str, episode_number: int,
                     difficulty_level: Optional[int] = None) -> Dict[str, Any]:
        """
        Train one complete episode with safety checks
        """
        # Set policy to training mode
        self.policy.train()
        
        try:
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
            self.difficulty_levels.append(difficulty_level if difficulty_level is not None else 0)
            self.total_episodes += 1
            self.total_steps += episode_data['episode_length']
            
            # Combine episode data with training metrics
            result = episode_data.copy()
            result.update(training_metrics)
            
            return result
        
        except Exception as e:
            logger.error(f"Exception in train_episode {episode_number}: {e}")
            # Return safe fallback result
            return {
                'episode_reward': -1.0,
                'episode_length': 1,
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy_loss': 0.0,
                'kl_divergence': 0.0,
                'clipfrac': 0.0,
                'difficulty_level': difficulty_level if difficulty_level is not None else 0,
                'final_sequence': starting_sequence,
                'edit_history': []
            }
    
    @abstractmethod
    def _forward_policy(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through policy (to be implemented by subclasses)"""
        pass
    
    def _calculate_log_probs(self, policy_output: Dict[str, torch.Tensor], 
                           actions: List[Dict[str, Any]]) -> torch.Tensor:
        """Calculate log probabilities for taken actions with safety checks"""
        log_probs = []
        
        for i, action in enumerate(actions):
            try:
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
                
                # Validate log probability
                if torch.isnan(log_prob) or torch.isinf(log_prob):
                    logger.warning(f"Invalid log prob for action {i}, using fallback")
                    log_prob = torch.tensor(-10.0, device=log_prob.device)  # Large negative value
                
                log_probs.append(log_prob)
            
            except Exception as e:
                logger.warning(f"Error calculating log prob for action {i}: {e}")
                log_probs.append(torch.tensor(-10.0, device=policy_output['edit_type'].device))
        
        return torch.stack(log_probs)
    
    def _calculate_entropy(self, policy_output: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate entropy for exploration regularization with safety checks"""
        try:
            edit_type_entropy = -(policy_output['edit_type'] * 
                                 torch.log(policy_output['edit_type'] + 1e-8)).sum(dim=1).mean()
            
            position_entropy = -(policy_output['position'] * 
                               torch.log(policy_output['position'] + 1e-8)).sum(dim=1).mean()
            
            aa_entropy = -(policy_output['amino_acid'] * 
                          torch.log(policy_output['amino_acid'] + 1e-8)).sum(dim=1).mean()
            
            total_entropy = edit_type_entropy + position_entropy + aa_entropy
            
            # Validate entropy
            if torch.isnan(total_entropy) or torch.isinf(total_entropy):
                logger.warning("Invalid entropy calculated, using fallback")
                return torch.tensor(0.1, device=total_entropy.device)
            
            return total_entropy
        
        except Exception as e:
            logger.warning(f"Error calculating entropy: {e}")
            return torch.tensor(0.1, device=policy_output['edit_type'].device)
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics"""
        if not self.episode_rewards:
            return {}
        
        stats = {
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'avg_episode_reward': self._safe_mean(self.episode_rewards),
            'avg_episode_length': self._safe_mean(self.episode_lengths),
            'recent_avg_reward': self._safe_mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else self._safe_mean(self.episode_rewards),
            'best_episode_reward': max(self.episode_rewards) if self.episode_rewards else 0.0,
            'avg_policy_loss': self._safe_mean(self.policy_losses),
            'avg_value_loss': self._safe_mean(self.value_losses),
            'avg_kl_divergence': self._safe_mean(self.kl_divergences),
            'avg_grad_norm': self._safe_mean(self.grad_norms),
            'nan_count': self.nan_count
        }
        
        # Add difficulty level statistics if available
        if self.difficulty_levels:
            stats.update({
                'avg_difficulty_level': self._safe_mean(self.difficulty_levels),
                'current_difficulty_level': self.difficulty_levels[-1] if self.difficulty_levels else 0,
                'recent_avg_difficulty': self._safe_mean(self.difficulty_levels[-100:]) if len(self.difficulty_levels) >= 100 else self._safe_mean(self.difficulty_levels)
            })
        
        return stats


class PPOTrainer(BasePPOTrainer):
    """Single-GPU PPO Trainer with NaN protection"""
    
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
    """Multi-GPU Distributed PPO Trainer with NaN protection"""
    
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
        
        # Only aggregate if we have valid metrics and distributed training
        if self.world_size > 1 and any(np.isfinite(v) for v in metrics.values()):
            try:
                self.distributed_metrics.update(metrics)
                aggregated_metrics = self.distributed_metrics.aggregate('mean')
                self.distributed_metrics.reset()
                return aggregated_metrics
            except Exception as e:
                logger.warning(f"Error in distributed metrics aggregation: {e}")
                return metrics
        
        return metrics
    
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
                'difficulty_level': result.get('difficulty_level', 0),
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
            'difficulty_levels': self.difficulty_levels.copy(),
            'nan_count': self.nan_count,
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
            self.nan_count = checkpoint.get('nan_count', 0)
        
        # Load difficulty levels if available
        if 'difficulty_levels' in checkpoint:
            self.difficulty_levels = checkpoint['difficulty_levels']
        
        episode = checkpoint.get('episode', 0)
        
        if self.is_main_process:
            logger.info(f"Checkpoint loaded from {path}, episode {episode}")
        
        return episode


def create_trainer(policy, environment, distributed: bool = False, 
                  rank: int = 0, world_size: int = 1, **kwargs) -> BasePPOTrainer:
    """
    Factory function to create appropriate trainer
    """
    if distributed and world_size > 1:
        return DistributedPPOTrainer(
            policy, environment, rank=rank, world_size=world_size, **kwargs
        )
    else:
        return PPOTrainer(policy, environment, **kwargs)