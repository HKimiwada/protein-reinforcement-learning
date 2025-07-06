# src/training/simple_stable_trainer_v2.py
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class SimpleStableTrainerV2:
    """Simplified stable trainer without complex inheritance"""
    
    def __init__(self, policy, environment, lr=1e-4, device=None):
        self.policy = policy
        self.environment = environment
        self.lr = lr
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move policy to device
        self.policy.to(self.device)
        
        # Stable optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), 
            lr=self.lr,
            weight_decay=0.01,
            betas=(0.9, 0.95)
        )
        
        # PPO hyperparameters
        self.clip_epsilon = 0.15
        self.value_coeff = 0.8
        self.entropy_coeff = 0.08
        self.max_grad_norm = 0.3
        self.gamma = 0.99
        self.gae_lambda = 0.95
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=200,
            gamma=0.9
        )
        
        # Tracking
        self.episode_rewards = []
        self.avg_reward = 0.0
        self.avg_improvement = 0.0
        self.momentum = 0.95
        
        # Experience buffer for stability
        self.experience_buffer = []
        self.buffer_size = 16

    def train_episode(self, starting_sequence: str, episode_number: int, difficulty_level=None):
        """Train one episode with stable methods"""
        
        # Collect episode experience
        episode_data = self._collect_episode(starting_sequence, episode_number)
        
        # Compute returns and advantages
        returns, advantages = self._compute_gae(
            episode_data['rewards'], 
            episode_data['values'], 
            episode_data['dones']
        )
        
        # Create batch
        batch = {
            'states': episode_data['states'],
            'actions': episode_data['actions'],
            'old_log_probs': episode_data['log_probs'],
            'returns': returns,
            'advantages': advantages,
            'values': episode_data['values']
        }
        
        # Update policy
        training_metrics = self._update_policy(batch)
        
        # Calculate actual improvement
        actual_improvement = 0.0
        if episode_data['final_sequence'] != starting_sequence:
            try:
                old_tough, _ = self.environment.reward_fn.predict_toughness(starting_sequence)
                new_tough, _ = self.environment.reward_fn.predict_toughness(episode_data['final_sequence'])
                actual_improvement = new_tough - old_tough
            except:
                actual_improvement = 0.0
        
        # Update tracking
        episode_reward = episode_data['episode_reward']
        self.episode_rewards.append(episode_reward)
        self.avg_reward = self.momentum * self.avg_reward + (1 - self.momentum) * episode_reward
        self.avg_improvement = self.momentum * self.avg_improvement + (1 - self.momentum) * actual_improvement
        
        # Step scheduler periodically
        if episode_number % 100 == 0 and episode_number > 0:
            self.scheduler.step()
        
        # Return comprehensive result
        result = episode_data.copy()
        result.update(training_metrics)
        result.update({
            'actual_improvement': actual_improvement,
            'avg_reward_ma': self.avg_reward,
            'avg_improvement_ma': self.avg_improvement,
            'current_lr': self.optimizer.param_groups[0]['lr']
        })
        
        return result

    def _collect_episode(self, starting_sequence: str, episode_number: int):
        """Collect episode experience"""
        
        # Reset environment
        state = self.environment.reset(starting_sequence).to(self.device)
        self.environment.set_episode_number(episode_number)
        
        # Episode buffers
        states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []
        episode_reward = 0
        
        self.policy.eval()
        
        while not self.environment.done:
            with torch.no_grad():
                # Get policy output
                policy_output = self.policy(state)
                
                # Get action
                action = self.policy.get_action(state, deterministic=False)
                
                # Store experience
                states.append(state.clone())
                actions.append(action)
                values.append(policy_output['value'].item())
                log_probs.append(action['log_prob'].to(self.device))
                
                # Environment step
                next_state, reward, done, info = self.environment.step(action)
                
                rewards.append(reward)
                dones.append(done)
                episode_reward += reward
                
                state = next_state.to(self.device)
                
                # Safety check
                if len(states) > 50:
                    break
        
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

    def _compute_gae(self, rewards: List[float], values: List[float], dones: List[bool]) -> Tuple[List[float], List[float]]:
        """Compute Generalized Advantage Estimation"""
        
        returns = []
        advantages = []
        
        # Bootstrap value
        next_value = 0
        advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = 0
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_value = values[t + 1]
            
            # TD error
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            
            # GAE advantage
            advantage = delta + self.gamma * self.gae_lambda * next_non_terminal * advantage
            
            advantages.insert(0, advantage)
            returns.insert(0, advantage + values[t])
        
        return returns, advantages

    def _update_policy(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Update policy using PPO"""
        
        # Add to experience buffer
        self.experience_buffer.append(batch)
        if len(self.experience_buffer) > self.buffer_size:
            self.experience_buffer.pop(0)
        
        # Use recent batches for more stable updates
        if len(self.experience_buffer) >= 3:
            combined_batch = self._combine_batches(self.experience_buffer[-3:])
        else:
            combined_batch = batch
        
        # Prepare tensors
        states = torch.stack(combined_batch['states']).to(self.device)
        old_log_probs = torch.stack(combined_batch['old_log_probs']).to(self.device)
        returns = torch.tensor(combined_batch['returns'], dtype=torch.float32, device=self.device)
        advantages = torch.tensor(combined_batch['advantages'], dtype=torch.float32, device=self.device)
        
        # Normalize advantages
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        self.policy.train()
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        
        # Multiple epochs for stability
        for epoch in range(2):
            # Forward pass
            policy_output = self.policy(states)
            
            # Calculate new log probs
            new_log_probs = self._calculate_log_probs(policy_output, combined_batch['actions'])
            
            # PPO loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            values = policy_output['value'].view(-1)
            value_loss = F.mse_loss(values, returns)
            
            # Entropy loss
            entropy_loss = self._calculate_entropy(policy_output)
            
            # Total loss
            total_loss = policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy_loss
            
            # Backpropagation
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            
            # Accumulate losses
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy_loss += entropy_loss.item()
        
        return {
            'policy_loss': total_policy_loss / 2,
            'value_loss': total_value_loss / 2,
            'entropy_loss': total_entropy_loss / 2
        }

    def _combine_batches(self, batches):
        """Combine multiple batches for stability"""
        combined = {
            'states': [],
            'actions': [],
            'old_log_probs': [],
            'returns': [],
            'advantages': [],
            'values': []
        }
        
        for batch in batches:
            for key in combined:
                if key in batch:
                    combined[key].extend(batch[key])
        
        return combined

    def _calculate_log_probs(self, policy_output, actions):
        """Calculate log probabilities for actions"""
        log_probs = []
        
        for i, action in enumerate(actions):
            try:
                # Edit type log prob
                edit_types = ['substitution', 'insertion', 'deletion', 'stop']
                edit_type_idx = edit_types.index(action['type'])
                log_prob = torch.log(policy_output['edit_type'][i, edit_type_idx] + 1e-8)
                
                if action['type'] != 'stop':
                    # Position log prob
                    log_prob += torch.log(policy_output['position'][i, action['position']] + 1e-8)
                    
                    # Amino acid log prob
                    if action['amino_acid'] is not None:
                        aa_idx = list('ACDEFGHIKLMNPQRSTVWY').index(action['amino_acid'])
                        log_prob += torch.log(policy_output['amino_acid'][i, aa_idx] + 1e-8)
                
                log_probs.append(log_prob)
            except:
                log_probs.append(torch.tensor(-10.0, device=self.device))
        
        return torch.stack(log_probs)

    def _calculate_entropy(self, policy_output):
        """Calculate entropy for exploration"""
        try:
            edit_type_entropy = -(policy_output['edit_type'] * 
                                torch.log(policy_output['edit_type'] + 1e-8)).sum(dim=1).mean()
            
            position_entropy = -(policy_output['position'] * 
                               torch.log(policy_output['position'] + 1e-8)).sum(dim=1).mean()
            
            aa_entropy = -(policy_output['amino_acid'] * 
                          torch.log(policy_output['amino_acid'] + 1e-8)).sum(dim=1).mean()
            
            return edit_type_entropy + position_entropy + aa_entropy
        except:
            return torch.tensor(0.1, device=self.device)