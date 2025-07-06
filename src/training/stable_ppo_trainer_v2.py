# src/training/stable_ppo_trainer_v2.py
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional
import logging

# Import base trainer
from src.training.ppo_trainer import BasePPOTrainer

logger = logging.getLogger(__name__)

class StablePPOTrainerV2(BasePPOTrainer):
    """More stable PPO trainer with better consistency"""
    
    def __init__(self, policy, environment, **kwargs):
        super().__init__(policy, environment, **kwargs)
        
        # Better optimizer with momentum and weight decay
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), 
            lr=self.lr,
            weight_decay=0.01,  # L2 regularization
            betas=(0.9, 0.95)   # Better momentum parameters
        )
        
        # Learning rate scheduler for long-term stability
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=400,  # Reduce LR every 400 episodes
            gamma=0.85      # Reduce by 15%
        )
        
        # Experience buffer for more stable learning
        self.experience_buffer = []
        self.buffer_size = 32
        
        # Moving averages for trend monitoring
        self.avg_reward = 0.0
        self.avg_improvement = 0.0
        self.avg_policy_loss = 0.0
        self.momentum = 0.95
        
        # Consistency tracking
        self.reward_history = []
        self.improvement_history = []
        self.consistency_window = 50

    def _setup_policy(self, policy):
        """Setup policy (implementation required by base class)"""
        self.policy = policy.to(self.device)

    def _forward_policy(self, state):
        """Forward pass through policy"""
        return self.policy(state)

    def update_policy(self, batch: Dict[str, Any], epochs: int = 3) -> Dict[str, float]:
        """More stable policy update with experience replay and better regularization"""
        
        # Add current batch to experience buffer
        self.experience_buffer.append(batch)
        if len(self.experience_buffer) > self.buffer_size:
            self.experience_buffer.pop(0)
        
        # Use multiple recent batches for more stable updates
        if len(self.experience_buffer) >= 3:
            # Combine last 3 batches for stability
            combined_batch = self._combine_batches(self.experience_buffer[-3:])
        else:
            combined_batch = batch
        
        # Call parent update with combined batch
        metrics = super().update_policy(combined_batch, epochs)
        
        # Update moving average of policy loss
        if 'policy_loss' in metrics:
            self.avg_policy_loss = (self.momentum * self.avg_policy_loss + 
                                  (1 - self.momentum) * metrics['policy_loss'])
        
        return metrics
    
    def _combine_batches(self, batches):
        """Combine multiple experience batches for more stable learning"""
        combined = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'returns': [],
            'advantages': []
        }
        
        for batch in batches:
            for key in combined:
                if key in batch:
                    if isinstance(batch[key], list):
                        combined[key].extend(batch[key])
                    else:
                        combined[key].append(batch[key])
        
        return combined
    
    def train_episode(self, starting_sequence: str, episode_number: int, 
                     difficulty_level=None) -> Dict[str, Any]:
        """Train episode with enhanced monitoring and stability"""
        
        result = super().train_episode(starting_sequence, episode_number, difficulty_level)
        
        # Update moving averages for trend tracking
        episode_reward = result.get('episode_reward', 0.0)
        self.avg_reward = self.momentum * self.avg_reward + (1 - self.momentum) * episode_reward
        
        # Track actual improvement if available
        actual_improvement = 0.0
        if 'edit_history' in result:
            # Calculate actual improvement from edit history or environment
            try:
                final_seq = result.get('final_sequence', starting_sequence)
                old_tough, _ = self.environment.reward_fn.predict_toughness(starting_sequence)
                new_tough, _ = self.environment.reward_fn.predict_toughness(final_seq)
                actual_improvement = new_tough - old_tough
            except Exception as e:
                logger.warning(f"Could not calculate actual improvement: {e}")
                actual_improvement = 0.0
        
        self.avg_improvement = (self.momentum * self.avg_improvement + 
                              (1 - self.momentum) * actual_improvement)
        
        # Update histories for consistency tracking
        self.reward_history.append(episode_reward)
        self.improvement_history.append(actual_improvement)
        
        # Keep only recent history
        if len(self.reward_history) > self.consistency_window:
            self.reward_history.pop(0)
            self.improvement_history.pop(0)
        
        # Step learning rate scheduler periodically
        if episode_number % 100 == 0 and episode_number > 0:
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"Learning rate updated to: {current_lr:.6f}")
        
        # Calculate consistency metrics
        consistency_metrics = self._calculate_consistency_metrics()
        
        # Add enhanced monitoring info
        result.update({
            'avg_reward_ma': self.avg_reward,
            'avg_improvement_ma': self.avg_improvement,
            'avg_policy_loss_ma': self.avg_policy_loss,
            'current_lr': self.optimizer.param_groups[0]['lr'],
            'actual_improvement': actual_improvement,
            **consistency_metrics
        })
        
        return result
    
    def _calculate_consistency_metrics(self) -> Dict[str, float]:
        """Calculate metrics to track learning consistency"""
        if len(self.reward_history) < 10:
            return {
                'reward_std': 0.0,
                'improvement_std': 0.0,
                'reward_trend': 0.0,
                'improvement_trend': 0.0
            }
        
        # Calculate standard deviations (lower = more consistent)
        reward_std = np.std(self.reward_history[-20:]) if len(self.reward_history) >= 20 else np.std(self.reward_history)
        improvement_std = np.std(self.improvement_history[-20:]) if len(self.improvement_history) >= 20 else np.std(self.improvement_history)
        
        # Calculate trends (positive = improving)
        if len(self.reward_history) >= 20:
            # Compare first half to second half of recent history
            first_half = np.mean(self.reward_history[-20:-10])
            second_half = np.mean(self.reward_history[-10:])
            reward_trend = second_half - first_half
            
            first_half_imp = np.mean(self.improvement_history[-20:-10])
            second_half_imp = np.mean(self.improvement_history[-10:])
            improvement_trend = second_half_imp - first_half_imp
        else:
            reward_trend = 0.0
            improvement_trend = 0.0
        
        return {
            'reward_std': float(reward_std),
            'improvement_std': float(improvement_std),
            'reward_trend': float(reward_trend),
            'improvement_trend': float(improvement_trend)
        }