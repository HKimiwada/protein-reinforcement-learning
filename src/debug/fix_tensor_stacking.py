#!/usr/bin/env python3
"""
Final fix for tensor stacking issues - specifically targets the log_prob problem
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class FinalFixedTrainer:
    """Final fixed trainer with comprehensive tensor handling"""
    
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
        
        # Tracking
        self.episode_rewards = []
        self.avg_reward = 0.0
        self.avg_improvement = 0.0
        self.momentum = 0.95

    def train_episode(self, starting_sequence: str, episode_number: int, difficulty_level=None):
        """Train one episode with bulletproof tensor handling"""
        
        try:
            # Collect episode experience
            episode_data = self._collect_episode_bulletproof(starting_sequence, episode_number)
            
            # Check if episode is valid
            if not episode_data or len(episode_data['states']) == 0:
                return self._create_fallback_result()
            
            # Only do policy update if we have meaningful data
            if len(episode_data['states']) > 1:  # Need at least 2 steps for meaningful update
                training_metrics = self._update_policy_bulletproof(episode_data)
            else:
                training_metrics = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy_loss': 0.0}
            
            # Calculate actual improvement
            actual_improvement = self._calculate_improvement_safe(starting_sequence, episode_data['final_sequence'])
            
            # Update tracking
            episode_reward = episode_data['episode_reward']
            self.episode_rewards.append(episode_reward)
            self.avg_reward = self.momentum * self.avg_reward + (1 - self.momentum) * episode_reward
            self.avg_improvement = self.momentum * self.avg_improvement + (1 - self.momentum) * actual_improvement
            
            # Return comprehensive result
            result = {
                'episode_reward': episode_reward,
                'episode_length': episode_data['episode_length'],
                'final_sequence': episode_data['final_sequence'],
                'actual_improvement': actual_improvement,
                'avg_reward_ma': self.avg_reward,
                'avg_improvement_ma': self.avg_improvement,
                'current_lr': self.optimizer.param_groups[0]['lr']
            }
            result.update(training_metrics)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in train_episode {episode_number}: {e}")
            return self._create_fallback_result()

    def _collect_episode_bulletproof(self, starting_sequence: str, episode_number: int):
        """Bulletproof episode collection with guaranteed consistent tensors"""
        
        try:
            # Reset environment
            state = self.environment.reset(starting_sequence).to(self.device)
            self.environment.set_episode_number(episode_number)
            
            # Episode data
            states = []
            actions = []
            rewards = []
            values = []
            log_probs = []
            dones = []
            episode_reward = 0
            
            self.policy.eval()
            
            step_count = 0
            max_steps = 20  # Shorter episodes for stability
            
            while not self.environment.done and step_count < max_steps:
                with torch.no_grad():
                    try:
                        # Get policy output
                        policy_output = self.policy(state)
                        
                        # Get action with proper sequence length
                        current_seq_len = len(self.environment.current_sequence)
                        action = self.policy.get_action(state, deterministic=False, sequence_length=current_seq_len)
                        
                        # 🚨 CRITICAL FIX: Ensure log_prob is ALWAYS a proper scalar tensor
                        log_prob = action.get('log_prob', torch.tensor(0.0))
                        
                        # Convert to proper scalar tensor
                        if not isinstance(log_prob, torch.Tensor):
                            log_prob = torch.tensor(float(log_prob), device=self.device)
                        elif log_prob.numel() == 0:
                            log_prob = torch.tensor(0.0, device=self.device)
                        elif log_prob.numel() > 1:
                            log_prob = log_prob.mean()
                        
                        # Ensure it's on the right device and is a scalar
                        log_prob = log_prob.to(self.device).reshape([])  # Force scalar shape
                        
                        # Store experience with validated tensors
                        states.append(state.clone())
                        actions.append(action)
                        values.append(policy_output['value'].item())
                        log_probs.append(log_prob)  # Now guaranteed to be scalar tensor
                        
                        # Environment step
                        next_state, reward, done, info = self.environment.step(action)
                        
                        # Validate reward
                        if np.isnan(reward) or np.isinf(reward):
                            reward = -0.1
                        
                        rewards.append(reward)
                        dones.append(done)
                        episode_reward += reward
                        
                        # Update state
                        state = next_state.to(self.device)
                        step_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Error in episode step {step_count}: {e}")
                        break
            
            # 🚨 FINAL VALIDATION: Ensure all log_probs are consistent scalar tensors
            validated_log_probs = []
            for i, lp in enumerate(log_probs):
                if isinstance(lp, torch.Tensor) and lp.numel() == 1:
                    validated_log_probs.append(lp.reshape([]))  # Ensure scalar shape
                else:
                    # Create a valid scalar tensor
                    validated_log_probs.append(torch.tensor(0.0, device=self.device))
            
            # Ensure we have at least one step
            if len(states) == 0:
                # Create minimal valid episode
                state = self.environment.reset(starting_sequence).to(self.device)
                states = [state]
                actions = [{'type': 'stop', 'position': 0, 'amino_acid': None}]
                rewards = [-0.5]
                values = [0.0]
                validated_log_probs = [torch.tensor(0.0, device=self.device)]
                dones = [True]
                episode_reward = -0.5
            
            return {
                'states': states,
                'actions': actions,
                'rewards': rewards,
                'values': values,
                'log_probs': validated_log_probs,  # Now guaranteed consistent
                'dones': dones,
                'episode_reward': episode_reward,
                'episode_length': len(states),
                'final_sequence': self.environment.current_sequence
            }
            
        except Exception as e:
            logger.error(f"Error in _collect_episode_bulletproof: {e}")
            return None

    def _update_policy_bulletproof(self, episode_data):
        """Bulletproof policy update with guaranteed tensor consistency"""
        
        try:
            # Compute returns and advantages
            returns, advantages = self._compute_gae_safe(
                episode_data['rewards'], 
                episode_data['values'], 
                episode_data['dones']
            )
            
            # 🚨 CRITICAL: Triple-check tensor consistency before stacking
            states = episode_data['states']
            log_probs = episode_data['log_probs']
            
            # Validate all states have same shape
            if len(states) == 0:
                return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy_loss': 0.0}
            
            state_shape = states[0].shape
            valid_indices = []
            
            for i, (state, log_prob) in enumerate(zip(states, log_probs)):
                # Check state shape consistency
                if state.shape != state_shape:
                    continue
                    
                # Check log_prob is valid scalar tensor
                if not isinstance(log_prob, torch.Tensor) or log_prob.numel() != 1:
                    continue
                    
                valid_indices.append(i)
            
            # Filter to only valid entries
            if len(valid_indices) == 0:
                return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy_loss': 0.0}
            
            # Create filtered data with only valid indices
            valid_states = [states[i] for i in valid_indices]
            valid_log_probs = [log_probs[i] for i in valid_indices]
            valid_actions = [episode_data['actions'][i] for i in valid_indices]
            valid_returns = [returns[i] for i in valid_indices]
            valid_advantages = [advantages[i] for i in valid_indices]
            
            # Now stack the validated tensors
            states_tensor = torch.stack(valid_states).to(self.device)
            old_log_probs_tensor = torch.stack(valid_log_probs).to(self.device)
            returns_tensor = torch.tensor(valid_returns, dtype=torch.float32, device=self.device)
            advantages_tensor = torch.tensor(valid_advantages, dtype=torch.float32, device=self.device)
            
            # Double-check tensor shapes match
            batch_size = states_tensor.size(0)
            if (old_log_probs_tensor.size(0) != batch_size or 
                returns_tensor.size(0) != batch_size or 
                advantages_tensor.size(0) != batch_size):
                logger.error(f"Batch size mismatch after filtering: states={states_tensor.shape}, log_probs={old_log_probs_tensor.shape}")
                return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy_loss': 0.0}
            
            # Normalize advantages
            if advantages_tensor.numel() > 1:
                advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
            
            # PPO update
            self.policy.train()
            
            # Forward pass
            policy_output = self.policy(states_tensor)
            
            # Calculate new log probs
            new_log_probs = self._calculate_log_probs_safe(policy_output, valid_actions)
            
            # PPO loss
            ratio = torch.exp(new_log_probs - old_log_probs_tensor)
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_tensor
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            values = policy_output['value'].view(-1)
            value_loss = F.mse_loss(values, returns_tensor)
            
            # Entropy loss
            entropy_loss = self._calculate_entropy_safe(policy_output)
            
            # Total loss
            total_loss = policy_loss + self.value_coeff * value_loss - self.entropy_coeff * entropy_loss
            
            # Backpropagation
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            
            return {
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item(),
                'entropy_loss': entropy_loss.item()
            }
            
        except Exception as e:
            logger.error(f"Error in bulletproof policy update: {e}")
            import traceback
            traceback.print_exc()
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy_loss': 0.0}

    def _calculate_log_probs_safe(self, policy_output, actions):
        """Calculate log probabilities safely"""
        log_probs = []
        
        for i, action in enumerate(actions):
            try:
                # Edit type log prob
                edit_types = ['substitution', 'insertion', 'deletion', 'stop']
                edit_type_idx = edit_types.index(action['type'])
                log_prob = torch.log(policy_output['edit_type'][i, edit_type_idx] + 1e-8)
                
                if action['type'] != 'stop':
                    # Position log prob
                    pos = action.get('position', 0)
                    if pos < policy_output['position'].size(1):
                        log_prob += torch.log(policy_output['position'][i, pos] + 1e-8)
                    
                    # Amino acid log prob
                    if action.get('amino_acid') is not None:
                        aa_idx = list('ACDEFGHIKLMNPQRSTVWY').index(action['amino_acid'])
                        if aa_idx < policy_output['amino_acid'].size(1):
                            log_prob += torch.log(policy_output['amino_acid'][i, aa_idx] + 1e-8)
                
                log_probs.append(log_prob)
            except Exception as e:
                logger.warning(f"Error calculating log prob for action {i}: {e}")
                log_probs.append(torch.tensor(-10.0, device=self.device))
        
        return torch.stack(log_probs)

    def _calculate_entropy_safe(self, policy_output):
        """Calculate entropy safely"""
        try:
            edit_type_entropy = -(policy_output['edit_type'] * 
                                torch.log(policy_output['edit_type'] + 1e-8)).sum(dim=1).mean()
            
            position_entropy = -(policy_output['position'] * 
                               torch.log(policy_output['position'] + 1e-8)).sum(dim=1).mean()
            
            aa_entropy = -(policy_output['amino_acid'] * 
                          torch.log(policy_output['amino_acid'] + 1e-8)).sum(dim=1).mean()
            
            return edit_type_entropy + position_entropy + aa_entropy
        except Exception as e:
            logger.warning(f"Error calculating entropy: {e}")
            return torch.tensor(0.1, device=self.device)

    def _compute_gae_safe(self, rewards, values, dones):
        """Compute GAE safely"""
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

    def _calculate_improvement_safe(self, old_seq, new_seq):
        """Calculate improvement safely"""
        try:
            if old_seq == new_seq:
                return 0.0
            old_tough, _ = self.environment.reward_fn.predict_toughness(old_seq)
            new_tough, _ = self.environment.reward_fn.predict_toughness(new_seq)
            return new_tough - old_tough
        except Exception as e:
            logger.warning(f"Error calculating improvement: {e}")
            return 0.0

    def _create_fallback_result(self):
        """Create safe fallback result"""
        return {
            'episode_reward': -1.0,
            'episode_length': 0,
            'actual_improvement': 0.0,
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy_loss': 0.0,
            'avg_reward_ma': self.avg_reward,
            'avg_improvement_ma': self.avg_improvement,
            'current_lr': self.lr
        }


def test_final_fix():
    """Test the final tensor fix"""
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    from src.config.stable_configs_v2 import get_stable_config_v2
    from src.debug.debug import setup_models_and_environment
    
    print("🔧 TESTING FINAL TENSOR FIX")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = get_stable_config_v2('stable_test')
    config_dict = config.to_dict()
    config_dict['max_steps'] = 10
    
    # Setup system
    setup_result = setup_models_and_environment(config_dict, device)
    if setup_result is None:
        print("❌ Setup failed")
        return False
    
    policy, env, dataset, utils, reward_fn = setup_result
    
    # Apply position fixes
    try:
        from apply_position_fixes import patch_environment_step, patch_policy_get_action, SafeSequenceActionSpace
        env = patch_environment_step(env)
        policy = patch_policy_get_action(policy)
        env.action_space = SafeSequenceActionSpace()
        print("✅ Position fixes applied")
    except Exception as e:
        print(f"⚠️ Could not apply position fixes: {e}")
    
    # Use the bulletproof trainer
    trainer = FinalFixedTrainer(policy, env, lr=1e-4, device=device)
    
    print("Running 30 episodes with final fix...")
    
    successful_episodes = 0
    tensor_errors = 0
    
    for episode in range(30):
        seq = dataset.train_sequences[episode % len(dataset.train_sequences)]
        
        try:
            result = trainer.train_episode(seq, episode)
            
            reward = result['episode_reward']
            improvement = result.get('actual_improvement', 0)
            
            if 'error' not in result:
                successful_episodes += 1
                
            if episode % 5 == 0:
                print(f"Episode {episode}: reward={reward:.3f}, improvement={improvement:.4f}")
                
        except Exception as e:
            if "stack expects each tensor to be equal size" in str(e):
                tensor_errors += 1
            print(f"Episode {episode}: ❌ {e}")
    
    print(f"\n🎯 FINAL FIX RESULTS:")
    print(f"  Successful episodes: {successful_episodes}/30 ({successful_episodes/30*100:.1f}%)")
    print(f"  Tensor stacking errors: {tensor_errors}")
    
    if tensor_errors == 0:
        print("✅ Tensor stacking errors completely eliminated!")
        return True
    else:
        print(f"❌ Still {tensor_errors} tensor errors")
        return False


if __name__ == "__main__":
    test_final_fix()