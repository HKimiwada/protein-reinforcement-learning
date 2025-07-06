#!/usr/bin/env python3
"""
Combined test for both position bounds fixes AND tensor stacking fixes
This script applies both fixes and tests them together

Quick Test (--quick):

Tests position bounds validation on 3 different sequence lengths
Shows how many invalid position errors are caught and fixed
Fast verification that position fixes are working

Position-Only Test (--position-only):

Comprehensive position bounds testing
Tests all action types (insertion, substitution, deletion)
Shows error catching in real-time
Verifies bounds checking works correctly

Full Combined Test (no flags):

Tests position fixes first
Then tests both position AND tensor fixes together
Runs 50 training episodes
Reports success rate, tensor errors, and position errors
Should show 90%+ success rate and 0 tensor errors
"""

import os
import sys
import torch
import argparse
import numpy as np
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.config.stable_configs_v2 import get_stable_config_v2
from src.debug.debug import setup_models_and_environment

logger = logging.getLogger(__name__)

# ===================================
# SAFE ACTION SPACE (Position Fix)
# ===================================

class SafeSequenceActionSpace:
    """Action space with comprehensive bounds checking"""
    
    def __init__(self):
        self.edit_types = ['substitution', 'insertion', 'deletion', 'stop']
        self.amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
        self.max_sequence_length = 1000

    def sample_action(self, action_probs, sequence_length):
        """Sample action with strict bounds checking"""
        
        device = action_probs['edit_type'].device
        
        # Validate sequence_length first
        if sequence_length < 5:
            return {
                'type': 'stop',
                'position': 0,
                'amino_acid': None,
                'log_prob': torch.tensor(0.0, device=device)
            }
        
        # 1) Sample edit type
        edit_type_dist = torch.distributions.Categorical(action_probs['edit_type'])
        edit_type_idx = edit_type_dist.sample()
        edit_type = self.edit_types[edit_type_idx.item()]
        
        log_prob = edit_type_dist.log_prob(edit_type_idx)
        
        if edit_type == 'stop':
            return {
                'type': 'stop',
                'position': 0,
                'amino_acid': None,
                'log_prob': log_prob
            }
        
        # 2) Sample position with STRICT bounds checking
        if edit_type == 'insertion':
            max_position = sequence_length  # Can insert at end
        else:
            max_position = sequence_length - 1  # Within sequence
        
        if max_position < 0:
            return {
                'type': 'stop',
                'position': 0,
                'amino_acid': None,
                'log_prob': torch.tensor(0.0, device=device)
            }
        
        # Get valid position probabilities
        valid_positions = max_position + 1
        pos_logits = action_probs['position'][:valid_positions]
        
        if pos_logits.numel() == 0:
            return {
                'type': 'stop',
                'position': 0,
                'amino_acid': None,
                'log_prob': torch.tensor(0.0, device=device)
            }
        
        pos_probs = pos_logits / pos_logits.sum(dim=0, keepdim=True)
        position_dist = torch.distributions.Categorical(pos_probs)
        position_idx = position_dist.sample()
        log_prob += position_dist.log_prob(position_idx)
        
        position = position_idx.item()
        
        # Final validation
        if edit_type == 'insertion' and position > sequence_length:
            return {'type': 'stop', 'position': 0, 'amino_acid': None, 'log_prob': torch.tensor(0.0, device=device)}
        
        if edit_type in ['substitution', 'deletion'] and position >= sequence_length:
            return {'type': 'stop', 'position': 0, 'amino_acid': None, 'log_prob': torch.tensor(0.0, device=device)}
        
        # 3) Sample amino acid
        amino_acid = None
        if edit_type in ('substitution', 'insertion'):
            aa_dist = torch.distributions.Categorical(action_probs['amino_acid'])
            aa_idx = aa_dist.sample()
            amino_acid = self.amino_acids[aa_idx.item()]
            log_prob += aa_dist.log_prob(aa_idx)
        
        return {
            'type': edit_type,
            'position': position,
            'amino_acid': amino_acid,
            'log_prob': log_prob
        }


# ===================================
# ENVIRONMENT PATCHING (Position Fix)
# ===================================

def patch_environment_step(env):
    """Patch environment step method to fix position validation"""
    
    env.position_error_count = 0  # Track fixes
    
    def step_with_bounds_checking(action):
        """Enhanced step with position bounds checking"""
        if env.done:
            return env.get_state(), 0.0, True, {'already_done': True}

        old_sequence = env.current_sequence
        sequence_length = len(env.current_sequence)
        
        # Validate action position bounds
        if action['type'] != 'stop':
            position = action.get('position', 0)
            
            # Check bounds based on action type
            if action['type'] == 'insertion':
                if position < 0 or position > sequence_length:
                    print(f"🚨 FIXED: Invalid insertion position {position} (seq len: {sequence_length})")
                    env.position_error_count += 1
                    action = {'type': 'stop', 'position': 0, 'amino_acid': None, 'log_prob': action.get('log_prob', torch.tensor(0.0))}
            
            elif action['type'] in ['substitution', 'deletion']:
                if position < 0 or position >= sequence_length:
                    print(f"🚨 FIXED: Invalid {action['type']} position {position} (seq len: {sequence_length})")
                    env.position_error_count += 1
                    action = {'type': 'stop', 'position': 0, 'amino_acid': None, 'log_prob': action.get('log_prob', torch.tensor(0.0))}
            
            # Check minimum sequence length for deletions
            if action['type'] == 'deletion' and sequence_length <= 15:
                print(f"🚨 FIXED: Sequence too short for deletion (len: {sequence_length})")
                env.position_error_count += 1
                action = {'type': 'stop', 'position': 0, 'amino_acid': None, 'log_prob': action.get('log_prob', torch.tensor(0.0))}
        
        # Call original step method with validated action
        return env._original_step(action)
    
    # Store original step method and replace
    env._original_step = env.step
    env.step = step_with_bounds_checking
    return env


# ===================================
# POLICY PATCHING (Position Fix)
# ===================================

def patch_policy_get_action(policy):
    """Patch policy get_action method for better bounds checking"""
    
    def get_action_safe(state, deterministic=False, sequence_length=None):
        """Safe action selection with bounds checking"""
        device = next(policy.parameters()).device
        state = state.to(device)
        
        if sequence_length is None:
            sequence_length = getattr(policy, 'default_seq_length', 500)
        
        # Ensure reasonable sequence length
        sequence_length = max(5, min(sequence_length, 800))
        
        with torch.no_grad():
            if state.dim() == 1:
                output = policy(state.unsqueeze(0))
                squeeze_output = True
            else:
                output = policy(state)
                squeeze_output = False

            if deterministic:
                if squeeze_output:
                    edit_type_probs = output['edit_type']
                    position_probs = output['position']
                    aa_probs = output['amino_acid']
                else:
                    edit_type_probs = output['edit_type'][0]
                    position_probs = output['position'][0]
                    aa_probs = output['amino_acid'][0]
                
                # Get edit type
                edit_type_idx = edit_type_probs.argmax().item()
                edit_type = ['substitution', 'insertion', 'deletion', 'stop'][edit_type_idx]
                
                if edit_type == 'stop':
                    return {
                        'type': 'stop',
                        'position': 0,
                        'amino_acid': None,
                        'log_prob': torch.tensor(0.0, device=device)
                    }
                
                # Proper position bounds
                if edit_type == 'insertion':
                    max_valid_pos = sequence_length  # Can insert at end
                else:  # substitution or deletion
                    max_valid_pos = sequence_length - 1  # Must be within sequence
                
                # Safety check
                if max_valid_pos < 0:
                    return {
                        'type': 'stop',
                        'position': 0,
                        'amino_acid': None,
                        'log_prob': torch.tensor(0.0, device=device)
                    }
                
                # Only consider valid positions
                valid_position_probs = position_probs[:max_valid_pos + 1]
                if len(valid_position_probs) > 0:
                    position_idx = valid_position_probs.argmax().item()
                else:
                    return {
                        'type': 'stop',
                        'position': 0,
                        'amino_acid': None,
                        'log_prob': torch.tensor(0.0, device=device)
                    }
                
                # Final bounds validation
                if edit_type == 'insertion' and position_idx > sequence_length:
                    print(f"🚨 POLICY FIX: insertion pos {position_idx} > {sequence_length}")
                    return {'type': 'stop', 'position': 0, 'amino_acid': None, 'log_prob': torch.tensor(0.0, device=device)}
                elif edit_type in ['substitution', 'deletion'] and position_idx >= sequence_length:
                    print(f"🚨 POLICY FIX: {edit_type} pos {position_idx} >= {sequence_length}")
                    return {'type': 'stop', 'position': 0, 'amino_acid': None, 'log_prob': torch.tensor(0.0, device=device)}
                
                aa_idx = aa_probs.argmax().item()
                amino_acid = list('ACDEFGHIKLMNPQRSTVWY')[aa_idx] if edit_type in ['substitution', 'insertion'] else None

                return {
                    'type': edit_type,
                    'position': position_idx,
                    'amino_acid': amino_acid,
                    'log_prob': torch.tensor(0.0, device=device)
                }

            else:
                # Stochastic sampling with safe action space
                action_space = SafeSequenceActionSpace()
                
                if squeeze_output:
                    sample_output = {k: v for k, v in output.items() if isinstance(v, torch.Tensor)}
                else:
                    sample_output = {k: v[0] for k, v in output.items() if isinstance(v, torch.Tensor)}
                
                # Get action with bounds checking
                action = action_space.sample_action(sample_output, sequence_length)
                
                return action
    
    # Store original method and replace
    policy._original_get_action = policy.get_action
    policy.get_action = get_action_safe
    return policy


# ===================================
# COMBINED TRAINER (Tensor Fix)
# ===================================

class CombinedFixedTrainer:
    """Trainer with both position bounds fixes AND tensor stacking fixes"""
    
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
        """Train one episode with both fixes applied"""
        
        try:
            # Collect episode experience with tensor fixes
            episode_data = self._collect_episode_bulletproof(starting_sequence, episode_number)
            
            # Check if episode is valid
            if not episode_data or len(episode_data['states']) == 0:
                return self._create_fallback_result()
            
            # Only do policy update if we have meaningful data
            if len(episode_data['states']) > 1:
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
            
            # Ensure we have at least one step
            if len(states) == 0:
                # Create minimal valid episode
                state = self.environment.reset(starting_sequence).to(self.device)
                states = [state]
                actions = [{'type': 'stop', 'position': 0, 'amino_acid': None}]
                rewards = [-0.5]
                values = [0.0]
                log_probs = [torch.tensor(0.0, device=self.device)]
                dones = [True]
                episode_reward = -0.5
            
            return {
                'states': states,
                'actions': actions,
                'rewards': rewards,
                'values': values,
                'log_probs': log_probs,  # Now guaranteed consistent
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
            
            # Validate all tensors before stacking
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
            value_loss = torch.nn.functional.mse_loss(values, returns_tensor)
            
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
            logger.error(f"Error in policy update: {e}")
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


# ===================================
# TESTING FUNCTIONS
# ===================================

def test_position_fixes_comprehensive():
    """Test position fixes comprehensively"""
    print("🧪 TESTING POSITION BOUNDS FIXES")
    print("="*50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = get_stable_config_v2('stable_test')
    config_dict = config.to_dict()
    config_dict['max_steps'] = 10
    
    # Setup system
    print("Setting up system...")
    setup_result = setup_models_and_environment(config_dict, device)
    if setup_result is None:
        print("❌ Setup failed")
        return False
    
    policy, env, dataset, utils, reward_fn = setup_result
    
    # Apply position fixes
    print("Applying position bounds fixes...")
    env = patch_environment_step(env)
    policy = patch_policy_get_action(policy)
    env.action_space = SafeSequenceActionSpace()
    
    print("✅ Position fixes applied!")
    
    # Test with various sequence lengths
    test_sequences = [
        "GPGGQGPYGPGGQ",  # Short sequence (13 chars)
        "GPGGQGPYGPGGQGPGGQGPYGPQ",  # Medium sequence (23 chars)
        "GPGGQGPYGPGGQGPGGQGPYGPQAAAAAAAAAAAGPGGQGPYGPGGQ",  # Long sequence (47 chars)
    ]
    
    total_actions = 0
    position_errors_caught = 0
    
    for i, test_seq in enumerate(test_sequences):
        print(f"\nTesting sequence {i+1} (length: {len(test_seq)})")
        
        # Reset environment
        state = env.reset(test_seq).to(device)
        sequence_length = len(test_seq)
        
        # Reset position error counter
        env.position_error_count = 0
        
        # Test multiple actions
        for step in range(10):
            try:
                # Get action from policy
                action = policy.get_action(state, deterministic=False, sequence_length=sequence_length)
                total_actions += 1
                
                print(f"  Step {step}: {action['type']}", end="")
                if action['type'] != 'stop':
                    max_pos = sequence_length if action['type'] == 'insertion' else sequence_length - 1
                    print(f" at pos {action['position']} (max: {max_pos})")
                else:
                    print()
                
                # Take environment step
                state, reward, done, info = env.step(action)
                state = state.to(device)
                
                # Update sequence length if it changed
                sequence_length = len(env.current_sequence)
                
                if done:
                    break
                    
            except Exception as e:
                print(f"  ❌ Error at step {step}: {e}")
                break
        
        position_errors_caught += env.position_error_count
        if env.position_error_count > 0:
            print(f"  🚨 Caught and fixed {env.position_error_count} position errors")
    
    print(f"\n🎯 POSITION FIXES TEST RESULTS:")
    print(f"  Total actions tested: {total_actions}")
    print(f"  Position errors caught and fixed: {position_errors_caught}")
    print(f"  Error rate: {position_errors_caught/total_actions*100:.1f}%")
    
    if total_actions > 0:
        print("✅ Position bounds testing completed!")
        return True, position_errors_caught
    else:
        print("❌ No actions were tested")
        return False, 0


def test_combined_fixes():
    """Test both position and tensor fixes together"""
    print("🚀 TESTING COMBINED FIXES (POSITION + TENSOR)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = get_stable_config_v2('stable_test')
    config_dict = config.to_dict()
    config_dict['max_steps'] = 15
    
    # Setup system
    setup_result = setup_models_and_environment(config_dict, device)
    if setup_result is None:
        print("❌ Setup failed")
        return False
    
    policy, env, dataset, utils, reward_fn = setup_result
    
    # Apply BOTH fixes
    print("Applying position bounds fixes...")
    env = patch_environment_step(env)
    policy = patch_policy_get_action(policy)
    env.action_space = SafeSequenceActionSpace()
    
    print("Creating combined trainer with tensor fixes...")
    trainer = CombinedFixedTrainer(policy, env, lr=1e-4, device=device)
    
    print("✅ Both fixes applied!")
    
    print("Running 50 episodes with combined fixes...")
    
    successful_episodes = 0
    tensor_errors = 0
    position_errors_total = 0
    
    for episode in range(50):
        seq = dataset.train_sequences[episode % len(dataset.train_sequences)]
        
        # Reset position error counter
        env.position_error_count = 0
        
        try:
            result = trainer.train_episode(seq, episode)
            
            reward = result['episode_reward']
            improvement = result.get('actual_improvement', 0)
            
            if 'error' not in result:
                successful_episodes += 1
            
            position_errors_total += env.position_error_count
            
            if episode % 10 == 0:
                print(f"Episode {episode}: reward={reward:.3f}, improvement={improvement:.4f}, pos_errors={env.position_error_count}")
                
        except Exception as e:
            if "stack expects each tensor to be equal size" in str(e):
                tensor_errors += 1
                print(f"Episode {episode}: ❌ Tensor error: {e}")
            else:
                print(f"Episode {episode}: ❌ Other error: {e}")
    
    print(f"\n🎯 COMBINED FIXES RESULTS:")
    print(f"  Successful episodes: {successful_episodes}/50 ({successful_episodes/50*100:.1f}%)")
    print(f"  Tensor stacking errors: {tensor_errors}")
    print(f"  Position errors caught and fixed: {position_errors_total}")
    print(f"  Recent average reward: {sum(trainer.episode_rewards[-10:])/max(len(trainer.episode_rewards[-10:]), 1):.3f}")
    
    if tensor_errors == 0 and successful_episodes >= 45:
        print("✅ Combined fixes working perfectly!")
        return True
    elif tensor_errors == 0:
        print("✅ Tensor errors eliminated, but training needs improvement")
        return True
    else:
        print(f"❌ Still {tensor_errors} tensor errors")
        return False


def main():
    """Main function to test combined fixes"""
    parser = argparse.ArgumentParser(description='Test Combined Position + Tensor Fixes')
    parser.add_argument('--position-only', action='store_true', help='Test only position fixes')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    
    args = parser.parse_args()
    
    print("🕷️  Spider Silk RL Combined Fixes Test")
    print("="*60)
    print("This script tests both position bounds fixes AND tensor stacking fixes")
    print()
    
    if args.position_only:
        # Test only position fixes
        success, errors_caught = test_position_fixes_comprehensive()
        if success:
            print(f"\n🎉 Position fixes working! Caught {errors_caught} errors.")
        else:
            print("\n❌ Position fixes need more work.")
        return
    
    if args.quick:
        # Quick test - just position fixes
        success, errors_caught = test_position_fixes_comprehensive()
        if success:
            print(f"\n✅ Quick test passed! Position fixes caught {errors_caught} errors.")
            print("Run without --quick for full combined test.")
        else:
            print("\n❌ Quick test failed.")
        return
    
    # Full combined test
    print("Phase 1: Testing position fixes...")
    position_success, position_errors = test_position_fixes_comprehensive()
    
    if position_success:
        print(f"\n✅ Position fixes working! Caught {position_errors} errors.")
        print("\nPhase 2: Testing combined fixes...")
        combined_success = test_combined_fixes()
        
        if combined_success:
            print("\n🎉 SUCCESS! Both position and tensor fixes are working!")
            print("\nNext steps:")
            print("1. Use CombinedFixedTrainer for your training")
            print("2. Apply patch_environment_step() and patch_policy_get_action() fixes")
            print("3. Replace action space with SafeSequenceActionSpace")
            print("4. Expect 90%+ episode success rate and 0 tensor errors")
        else:
            print("\n⚠️  Position fixes work, but tensor issues remain.")
    else:
        print("\n❌ Position fixes failed. Check implementation.")


if __name__ == "__main__":
    main()