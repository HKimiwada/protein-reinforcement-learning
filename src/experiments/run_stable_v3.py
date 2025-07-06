#!/usr/bin/env python3
"""
run_stable_v2_fixed.py - Complete Fixed Training Script with Extensive Logging

This script integrates all fixes (position bounds + tensor stacking) and provides
detailed logging to track RL performance, failures, and improvements.

Usage:
    python src/experiments/run_stable_v2_fixed.py --config stable --episodes 2000
    python src/experiments/run_stable_v2_fixed.py --config stable_test --episodes 100 --verbose
"""

import os
import sys
import argparse
import torch
import numpy as np
import logging
import time
from collections import defaultdict, deque
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.config.stable_configs_v2 import get_stable_config_v2
from src.models.improved_policy_v2 import ImprovedSequenceEditPolicyV2
from src.models.stable_reward_function_v2 import StableSpiderSilkRewardFunctionV2
from src.data.dataset import SpiderSilkDataset
from src.utils.spider_silk_utils import SpiderSilkUtils
from src.environment.protein_env import ProteinEditEnvironment

# Import model loading utilities
from src.debug.debug import fix_both_warnings
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoTokenizer

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_detailed.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===================================
# FIXED COMPONENTS (Position + Tensor)
# ===================================

class SafeSequenceActionSpace:
    """Action space with comprehensive bounds checking and logging"""
    
    def __init__(self):
        self.edit_types = ['substitution', 'insertion', 'deletion', 'stop']
        self.amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
        self.max_sequence_length = 1000
        
        # Logging counters
        self.action_counts = defaultdict(int)
        self.bounds_violations = defaultdict(int)
        self.safety_stops = 0

    def sample_action(self, action_probs, sequence_length):
        """Sample action with strict bounds checking and detailed logging"""
        
        device = action_probs['edit_type'].device
        
        # Log sequence length for analysis
        if sequence_length < 10:
            self.bounds_violations['sequence_too_short'] += 1
            logger.warning(f"Sequence too short for editing: {sequence_length}")
        
        # Validate sequence_length first
        if sequence_length < 5:
            self.safety_stops += 1
            return {
                'type': 'stop',
                'position': 0,
                'amino_acid': None,
                'log_prob': torch.tensor(0.0, device=device),
                'safety_reason': 'sequence_too_short'
            }
        
        # 1) Sample edit type
        edit_type_dist = torch.distributions.Categorical(action_probs['edit_type'])
        edit_type_idx = edit_type_dist.sample()
        edit_type = self.edit_types[edit_type_idx.item()]
        
        log_prob = edit_type_dist.log_prob(edit_type_idx)
        self.action_counts[edit_type] += 1
        
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
            self.bounds_violations[f'{edit_type}_no_valid_positions'] += 1
            self.safety_stops += 1
            logger.warning(f"No valid positions for {edit_type} in sequence length {sequence_length}")
            return {
                'type': 'stop',
                'position': 0,
                'amino_acid': None,
                'log_prob': torch.tensor(0.0, device=device),
                'safety_reason': f'{edit_type}_no_valid_positions'
            }
        
        # Get valid position probabilities
        valid_positions = max_position + 1
        pos_logits = action_probs['position'][:valid_positions]
        
        if pos_logits.numel() == 0:
            self.bounds_violations[f'{edit_type}_empty_position_logits'] += 1
            self.safety_stops += 1
            return {
                'type': 'stop',
                'position': 0,
                'amino_acid': None,
                'log_prob': torch.tensor(0.0, device=device),
                'safety_reason': f'{edit_type}_empty_position_logits'
            }
        
        pos_probs = pos_logits / pos_logits.sum(dim=0, keepdim=True)
        position_dist = torch.distributions.Categorical(pos_probs)
        position_idx = position_dist.sample()
        log_prob += position_dist.log_prob(position_idx)
        
        position = position_idx.item()
        
        # Final validation with detailed logging
        if edit_type == 'insertion' and position > sequence_length:
            self.bounds_violations['insertion_position_overflow'] += 1
            logger.error(f"Insertion position {position} > {sequence_length}")
            return {'type': 'stop', 'position': 0, 'amino_acid': None, 'log_prob': torch.tensor(0.0, device=device), 'safety_reason': 'insertion_overflow'}
        
        if edit_type in ['substitution', 'deletion'] and position >= sequence_length:
            self.bounds_violations[f'{edit_type}_position_overflow'] += 1
            logger.error(f"{edit_type} position {position} >= {sequence_length}")
            return {'type': 'stop', 'position': 0, 'amino_acid': None, 'log_prob': torch.tensor(0.0, device=device), 'safety_reason': f'{edit_type}_overflow'}
        
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

    def get_action_stats(self):
        """Get comprehensive action statistics"""
        total_actions = sum(self.action_counts.values())
        if total_actions == 0:
            return "No actions sampled yet"
        
        stats = f"Action Statistics (Total: {total_actions}):\n"
        for action_type, count in self.action_counts.items():
            percentage = (count / total_actions) * 100
            stats += f"  {action_type}: {count} ({percentage:.1f}%)\n"
        
        if self.bounds_violations:
            stats += f"\nBounds Violations:\n"
            for violation_type, count in self.bounds_violations.items():
                stats += f"  {violation_type}: {count}\n"
        
        if self.safety_stops > 0:
            stats += f"\nSafety stops: {self.safety_stops}\n"
        
        return stats


class EnhancedProteinEnvironment(ProteinEditEnvironment):
    """Environment with comprehensive logging and failure tracking"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Comprehensive failure tracking
        self.failure_stats = {
            'invalid_position': 0,
            'invalid_edit': 0,
            'perplexity_too_high': 0,
            'sequence_too_short': 0,
            'length_ratio_violation': 0,
            'motifs_missing': 0,
            'no_change': 0,
            'other_failures': 0
        }
        
        # Success tracking
        self.success_stats = {
            'successful_edits': 0,
            'positive_improvements': 0,
            'negative_improvements': 0,
            'neutral_changes': 0
        }
        
        # Episode-specific tracking
        self.episode_edits = []
        self.episode_failures = []
        
        # Replace action space with safe version
        self.action_space = SafeSequenceActionSpace()

    def reset(self, initial_sequence=None):
        """Reset with episode tracking"""
        self.episode_edits = []
        self.episode_failures = []
        return super().reset(initial_sequence)

    def step(self, action):
        """Enhanced step with comprehensive failure tracking"""
        if self.done:
            return self.get_state(), 0.0, True, {'already_done': True}

        old_sequence = self.current_sequence
        sequence_length = len(self.current_sequence)
        
        # Log action attempt
        action_log = {
            'type': action['type'],
            'position': action.get('position', 0),
            'amino_acid': action.get('amino_acid'),
            'sequence_length': sequence_length,
            'step': self.step_count
        }
        
        # Validate action position bounds with detailed logging
        if action['type'] != 'stop':
            position = action.get('position', 0)
            
            # Check bounds based on action type
            if action['type'] == 'insertion':
                if position < 0 or position > sequence_length:
                    self.failure_stats['invalid_position'] += 1
                    self.episode_failures.append({**action_log, 'failure_reason': 'insertion_out_of_bounds'})
                    logger.warning(f"🚨 FIXED: Invalid insertion position {position} (seq len: {sequence_length})")
                    action = {'type': 'stop', 'position': 0, 'amino_acid': None, 'log_prob': action.get('log_prob', torch.tensor(0.0))}
            
            elif action['type'] in ['substitution', 'deletion']:
                if position < 0 or position >= sequence_length:
                    self.failure_stats['invalid_position'] += 1
                    self.episode_failures.append({**action_log, 'failure_reason': f'{action["type"]}_out_of_bounds'})
                    logger.warning(f"🚨 FIXED: Invalid {action['type']} position {position} (seq len: {sequence_length})")
                    action = {'type': 'stop', 'position': 0, 'amino_acid': None, 'log_prob': action.get('log_prob', torch.tensor(0.0))}
            
            # Check minimum sequence length for deletions
            if action['type'] == 'deletion' and sequence_length <= 15:
                self.failure_stats['sequence_too_short'] += 1
                self.episode_failures.append({**action_log, 'failure_reason': 'sequence_too_short_for_deletion'})
                logger.warning(f"🚨 FIXED: Sequence too short for deletion (len: {sequence_length})")
                action = {'type': 'stop', 'position': 0, 'amino_acid': None, 'log_prob': action.get('log_prob', torch.tensor(0.0))}

        # Handle stop action
        if action['type'] == 'stop':
            self.done = True
            
            if len(self.edit_history) == 0:
                reward = -0.5  # Penalty for immediate stopping
                stop_reason = 'immediate_stop'
            elif len(self.edit_history) < 3:
                reward = -0.1  # Small penalty for early stopping
                stop_reason = 'early_stop'
            else:
                reward = 0.05  # Small reward for reasonable stopping
                stop_reason = 'normal_stop'
                
            return self.get_state(), reward, True, {
                'action': action,
                'edit_successful': True,
                'stop_reason': stop_reason,
                'episode_summary': self.get_episode_summary()
            }

        # Execute edit action
        new_sequence = self._execute_action(action)
        edit_successful = False

        if new_sequence != old_sequence:
            # Validate the resulting sequence with detailed logging
            is_valid, message = self.utils.validate_edit(old_sequence, new_sequence)

            if is_valid:
                # Calculate toughness improvement
                old_tough, _ = self.reward_fn.predict_toughness(old_sequence)
                new_tough, _ = self.reward_fn.predict_toughness(new_sequence)
                toughness_improvement = new_tough - old_tough

                # Update state
                self.current_sequence = new_sequence
                edit_info = {
                    'type': action['type'],
                    'position': action['position'],
                    'toughness_improvement': toughness_improvement,
                    'step': self.step_count,
                    'validation_status': 'valid',
                    'sequence_length_before': sequence_length,
                    'sequence_length_after': len(new_sequence)
                }
                
                if action['type'] == 'substitution':
                    edit_info['original'] = old_sequence[action['position']]
                    edit_info['new'] = action['amino_acid']
                elif action['type'] == 'insertion':
                    edit_info['inserted'] = action['amino_acid']
                elif action['type'] == 'deletion':
                    edit_info['deleted'] = old_sequence[action['position']]

                self.edit_history.append(edit_info)
                self.episode_edits.append(edit_info)
                edit_successful = True
                
                # Track success statistics
                self.success_stats['successful_edits'] += 1
                if toughness_improvement > 0:
                    self.success_stats['positive_improvements'] += 1
                elif toughness_improvement < 0:
                    self.success_stats['negative_improvements'] += 1
                else:
                    self.success_stats['neutral_changes'] += 1

                # Calculate reward
                reward_info = self.reward_fn.calculate_reward(
                    old_sequence, new_sequence, self.edit_history,
                    self.original_sequence, self.episode_number
                )
                reward = reward_info['total']
                self.done = reward_info.get('done', False)
                
                # Log successful edit
                logger.info(f"✅ Successful {action['type']}: pos={action['position']}, improvement={toughness_improvement:.4f}, reward={reward:.3f}")

            else:
                # Invalid edit - categorize failure
                failure_reason = self._categorize_validation_failure(message)
                self.failure_stats[failure_reason] += 1
                self.episode_failures.append({**action_log, 'failure_reason': failure_reason, 'validation_message': message})
                
                reward = -1.0
                edit_info = {
                    'type': 'invalid_edit', 
                    'message': message,
                    'attempted_action': action['type'],
                    'validation_status': 'failed',
                    'failure_category': failure_reason
                }
                
                logger.warning(f"❌ Invalid {action['type']}: {message}")
        else:
            # No sequence change
            self.failure_stats['no_change'] += 1
            self.episode_failures.append({**action_log, 'failure_reason': 'no_change'})
            reward = -0.3
            edit_info = {
                'type': 'no_change', 
                'validation_status': 'no_edit',
                'reason': 'action_had_no_effect'
            }
            
            logger.debug(f"⚪ No change from {action['type']} at position {action.get('position', 0)}")

        # Update step count
        self.step_count += 1

        # Check termination conditions
        if self.step_count >= self.max_steps:
            self.done = True

        info = {
            'action': action,
            'edit_info': edit_info,
            'edit_successful': edit_successful,
            'sequence_length': len(self.current_sequence),
            'sequence_changed': old_sequence != self.current_sequence,
            'edit_count': len(self.edit_history),
            'failure_stats': self.failure_stats.copy(),
            'success_stats': self.success_stats.copy()
        }

        return self.get_state(), reward, self.done, info

    def _categorize_validation_failure(self, message):
        """Categorize validation failure for detailed tracking"""
        message_lower = message.lower()
        
        if 'perplexity' in message_lower:
            return 'perplexity_too_high'
        elif 'length ratio' in message_lower:
            return 'length_ratio_violation'
        elif 'motif' in message_lower:
            return 'motifs_missing'
        else:
            return 'other_failures'

    def get_episode_summary(self):
        """Get comprehensive episode summary"""
        total_improvement = sum(
            edit.get('toughness_improvement', 0.0) 
            for edit in self.episode_edits
        )
        
        edit_types = [edit.get('type', 'unknown') for edit in self.episode_edits]
        edit_type_counts = {
            'substitution': edit_types.count('substitution'),
            'insertion': edit_types.count('insertion'),
            'deletion': edit_types.count('deletion')
        }
        
        failure_reasons = [failure.get('failure_reason', 'unknown') for failure in self.episode_failures]
        failure_counts = defaultdict(int)
        for reason in failure_reasons:
            failure_counts[reason] += 1
        
        return {
            'successful_edits': len(self.episode_edits),
            'failed_attempts': len(self.episode_failures),
            'total_improvement': total_improvement,
            'edit_type_counts': edit_type_counts,
            'failure_counts': dict(failure_counts),
            'final_sequence_length': len(self.current_sequence),
            'original_sequence_length': len(self.original_sequence),
            'steps_taken': self.step_count
        }

    def get_comprehensive_stats(self):
        """Get comprehensive environment statistics"""
        total_failures = sum(self.failure_stats.values())
        total_successes = sum(self.success_stats.values())
        total_actions = total_failures + total_successes
        
        if total_actions == 0:
            return "No actions attempted yet"
        
        stats = f"Environment Statistics (Total Actions: {total_actions}):\n"
        stats += f"  Success Rate: {total_successes}/{total_actions} ({total_successes/total_actions*100:.1f}%)\n"
        
        stats += f"\nSuccess Breakdown:\n"
        for success_type, count in self.success_stats.items():
            if count > 0:
                percentage = (count / total_successes) * 100 if total_successes > 0 else 0
                stats += f"  {success_type}: {count} ({percentage:.1f}%)\n"
        
        stats += f"\nFailure Breakdown:\n"
        for failure_type, count in self.failure_stats.items():
            if count > 0:
                percentage = (count / total_failures) * 100 if total_failures > 0 else 0
                stats += f"  {failure_type}: {count} ({percentage:.1f}%)\n"
        
        return stats


class ComprehensiveFixedTrainer:
    """Trainer with all fixes and extensive logging capabilities"""
    
    def __init__(self, policy, environment, lr=1e-4, device=None):
        self.policy = policy
        self.environment = environment
        self.lr = lr
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move policy to device
        self.policy.to(self.device)
        
        # Optimizer with stable settings
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
        
        # Comprehensive tracking
        self.episode_rewards = []
        self.episode_improvements = []
        self.episode_lengths = []
        self.training_losses = {'policy': [], 'value': [], 'entropy': []}
        
        # Moving averages
        self.avg_reward = 0.0
        self.avg_improvement = 0.0
        self.momentum = 0.95
        
        # Performance tracking
        self.performance_window = deque(maxlen=100)
        self.best_reward = -float('inf')
        self.best_improvement = 0.0
        self.consecutive_successes = 0
        
        # Failure analysis
        self.tensor_errors = 0
        self.policy_errors = 0
        self.environment_errors = 0

    def train_episode(self, starting_sequence: str, episode_number: int, difficulty_level=None):
        """Train episode with comprehensive logging and error handling"""
        
        episode_start_time = time.time()
        
        try:
            # Collect episode experience
            episode_data = self._collect_episode_with_logging(starting_sequence, episode_number)
            
            if not episode_data or len(episode_data['states']) == 0:
                logger.warning(f"Episode {episode_number}: No valid data collected")
                return self._create_fallback_result(episode_number)
            
            # Policy update with error handling
            if len(episode_data['states']) > 1:
                training_metrics = self._update_policy_with_logging(episode_data, episode_number)
            else:
                training_metrics = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy_loss': 0.0}
                logger.debug(f"Episode {episode_number}: Skipping policy update (insufficient data)")
            
            # Calculate comprehensive metrics
            actual_improvement = self._calculate_improvement_safe(starting_sequence, episode_data['final_sequence'])
            episode_reward = episode_data['episode_reward']
            episode_length = episode_data['episode_length']
            
            # Update tracking
            self.episode_rewards.append(episode_reward)
            self.episode_improvements.append(actual_improvement)
            self.episode_lengths.append(episode_length)
            self.performance_window.append({'reward': episode_reward, 'improvement': actual_improvement})
            
            # Update moving averages
            self.avg_reward = self.momentum * self.avg_reward + (1 - self.momentum) * episode_reward
            self.avg_improvement = self.momentum * self.avg_improvement + (1 - self.momentum) * actual_improvement
            
            # Track best performance
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                logger.info(f"🎉 New best reward: {self.best_reward:.3f} (Episode {episode_number})")
            
            if actual_improvement > self.best_improvement:
                self.best_improvement = actual_improvement
                logger.info(f"🎉 New best improvement: {self.best_improvement:.4f} (Episode {episode_number})")
            
            # Track consecutive successes
            if episode_reward > 0:
                self.consecutive_successes += 1
            else:
                self.consecutive_successes = 0
            
            # Store training losses
            for loss_type, value in training_metrics.items():
                if loss_type in self.training_losses:
                    self.training_losses[loss_type].append(value)
            
            # Learning rate scheduling
            if episode_number % 100 == 0 and episode_number > 0:
                old_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step()
                new_lr = self.optimizer.param_groups[0]['lr']
                if old_lr != new_lr:
                    logger.info(f"Learning rate updated: {old_lr:.6f} → {new_lr:.6f}")
            
            # Episode timing
            episode_time = time.time() - episode_start_time
            
            # Create comprehensive result
            result = {
                'episode_number': episode_number,
                'episode_reward': episode_reward,
                'episode_length': episode_length,
                'actual_improvement': actual_improvement,
                'final_sequence': episode_data['final_sequence'],
                'episode_time': episode_time,
                'difficulty_level': difficulty_level,
                
                # Moving averages
                'avg_reward_ma': self.avg_reward,
                'avg_improvement_ma': self.avg_improvement,
                
                # Training metrics
                'current_lr': self.optimizer.param_groups[0]['lr'],
                'consecutive_successes': self.consecutive_successes,
                'best_reward': self.best_reward,
                'best_improvement': self.best_improvement,
                
                # Episode summary from environment
                'episode_summary': self.environment.get_episode_summary(),
                
                # Training losses
                **training_metrics
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Episode {episode_number} failed: {e}")
            self.environment_errors += 1
            return self._create_fallback_result(episode_number, error=str(e))

    def _collect_episode_with_logging(self, starting_sequence: str, episode_number: int):
        """Collect episode with detailed logging"""
        
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
            max_steps = 25  # Reasonable episode length
            
            logger.debug(f"Episode {episode_number}: Starting with sequence length {len(starting_sequence)}")
            
            while not self.environment.done and step_count < max_steps:
                with torch.no_grad():
                    try:
                        # Get policy output
                        policy_output = self.policy(state)
                        
                        # Get action with proper sequence length
                        current_seq_len = len(self.environment.current_sequence)
                        action = self.policy.get_action(state, deterministic=False, sequence_length=current_seq_len)
                        
                        # 🚨 CRITICAL FIX: Ensure log_prob is always a proper scalar tensor
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
                        log_probs.append(log_prob)
                        
                        # Environment step
                        next_state, reward, done, info = self.environment.step(action)
                        
                        # Validate reward
                        if np.isnan(reward) or np.isinf(reward):
                            logger.warning(f"Invalid reward {reward}, using -0.1")
                            reward = -0.1
                        
                        rewards.append(reward)
                        dones.append(done)
                        episode_reward += reward
                        
                        # Log step details
                        if action['type'] != 'stop':
                            logger.debug(f"  Step {step_count}: {action['type']} at pos {action['position']} → reward {reward:.3f}")
                        else:
                            logger.debug(f"  Step {step_count}: stop → reward {reward:.3f}")
                        
                        # Update state
                        state = next_state.to(self.device)
                        step_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Error in episode {episode_number} step {step_count}: {e}")
                        break
            
            # Ensure we have at least one step
            if len(states) == 0:
                logger.warning(f"Episode {episode_number}: Creating minimal fallback episode")
                state = self.environment.reset(starting_sequence).to(self.device)
                states = [state]
                actions = [{'type': 'stop', 'position': 0, 'amino_acid': None}]
                rewards = [-0.5]
                values = [0.0]
                log_probs = [torch.tensor(0.0, device=self.device)]
                dones = [True]
                episode_reward = -0.5
            
            logger.debug(f"Episode {episode_number}: Completed with {len(states)} steps, reward {episode_reward:.3f}")
            
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
            
        except Exception as e:
            logger.error(f"Error collecting episode {episode_number}: {e}")
            self.environment_errors += 1
            return None

    def _update_policy_with_logging(self, episode_data, episode_number):
        """Update policy with comprehensive error handling and logging"""
        
        try:
            self.policy.train()
            
            # Extract episode data
            states = episode_data['states']
            actions = episode_data['actions']
            rewards = episode_data['rewards']
            values = episode_data['values']
            log_probs = episode_data['log_probs']
            dones = episode_data['dones']
            
            # 🚨 CRITICAL FIX: Stack tensors safely
            try:
                # Stack states safely
                if len(states) > 1:
                    states_tensor = torch.stack(states)
                else:
                    states_tensor = states[0].unsqueeze(0)
                
                # Stack log_probs safely with validation
                validated_log_probs = []
                for i, lp in enumerate(log_probs):
                    if not isinstance(lp, torch.Tensor):
                        lp = torch.tensor(float(lp), device=self.device)
                    elif lp.numel() == 0:
                        lp = torch.tensor(0.0, device=self.device)
                    elif lp.numel() > 1:
                        lp = lp.mean()
                    
                    # Ensure scalar and right device
                    lp = lp.to(self.device).reshape([])
                    validated_log_probs.append(lp)
                
                if len(validated_log_probs) > 1:
                    old_log_probs = torch.stack(validated_log_probs)
                else:
                    old_log_probs = validated_log_probs[0].unsqueeze(0)
                
                # Convert other data to tensors
                rewards_tensor = torch.tensor(rewards, device=self.device, dtype=torch.float32)
                values_tensor = torch.tensor(values, device=self.device, dtype=torch.float32)
                dones_tensor = torch.tensor(dones, device=self.device, dtype=torch.bool)
                
            except Exception as e:
                logger.error(f"Tensor stacking error in episode {episode_number}: {e}")
                self.tensor_errors += 1
                return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy_loss': 0.0}
            
            # Calculate advantages using GAE
            advantages, returns = self._calculate_gae(rewards_tensor, values_tensor, dones_tensor)
            
            # Normalize advantages
            if advantages.std() > 1e-8:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Get new policy outputs
            try:
                with torch.no_grad():
                    # Get new log probs and values for all states
                    new_log_probs = []
                    new_values = []
                    entropies = []
                    
                    for i, (state, action) in enumerate(zip(states, actions)):
                        policy_output = self.policy(state.unsqueeze(0))
                        
                        # Get action probabilities
                        action_probs = self.policy.get_action_probabilities(state.unsqueeze(0))
                        
                        # Calculate log probability for the taken action
                        if action['type'] == 'stop':
                            # Stop action
                            edit_type_idx = 3  # stop is index 3
                            log_prob = torch.log(action_probs['edit_type'][0, edit_type_idx] + 1e-8)
                        else:
                            # Non-stop action
                            edit_type_idx = ['substitution', 'insertion', 'deletion'].index(action['type'])
                            
                            # Edit type log prob
                            log_prob = torch.log(action_probs['edit_type'][0, edit_type_idx] + 1e-8)
                            
                            # Position log prob
                            position = action['position']
                            if position < action_probs['position'].shape[1]:
                                log_prob += torch.log(action_probs['position'][0, position] + 1e-8)
                            
                            # Amino acid log prob (if applicable)
                            if action['type'] in ['substitution', 'insertion'] and action['amino_acid']:
                                aa_idx = list('ACDEFGHIKLMNPQRSTVWY').index(action['amino_acid'])
                                if aa_idx < action_probs['amino_acid'].shape[1]:
                                    log_prob += torch.log(action_probs['amino_acid'][0, aa_idx] + 1e-8)
                        
                        # Calculate entropy
                        edit_type_entropy = -(action_probs['edit_type'] * torch.log(action_probs['edit_type'] + 1e-8)).sum()
                        pos_entropy = -(action_probs['position'] * torch.log(action_probs['position'] + 1e-8)).sum()
                        aa_entropy = -(action_probs['amino_acid'] * torch.log(action_probs['amino_acid'] + 1e-8)).sum()
                        total_entropy = edit_type_entropy + pos_entropy + aa_entropy
                        
                        new_log_probs.append(log_prob.reshape([]))
                        new_values.append(policy_output['value'].reshape([]))
                        entropies.append(total_entropy.reshape([]))
                
                # Stack the computed tensors
                new_log_probs = torch.stack(new_log_probs)
                new_values = torch.stack(new_values)
                entropies = torch.stack(entropies)
                
            except Exception as e:
                logger.error(f"Policy evaluation error in episode {episode_number}: {e}")
                self.policy_errors += 1
                return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy_loss': 0.0}
            
            # PPO losses
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Clipped objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = torch.nn.functional.mse_loss(new_values, returns)
            
            # Entropy loss
            entropy_loss = -entropies.mean()
            
            # Total loss
            total_loss = policy_loss + self.value_coeff * value_loss + self.entropy_coeff * entropy_loss
            
            # Optimization step
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            
            # Log training metrics
            training_metrics = {
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item(),
                'entropy_loss': entropy_loss.item(),
                'total_loss': total_loss.item(),
                'mean_advantage': advantages.mean().item(),
                'mean_return': returns.mean().item(),
                'policy_ratio_mean': ratio.mean().item(),
                'policy_ratio_std': ratio.std().item()
            }
            
            # Log detailed metrics every 50 episodes
            if episode_number % 50 == 0:
                logger.info(f"Training metrics (Ep {episode_number}): "
                           f"Policy={policy_loss.item():.4f}, "
                           f"Value={value_loss.item():.4f}, "
                           f"Entropy={entropy_loss.item():.4f}")
            
            return training_metrics
            
        except Exception as e:
            logger.error(f"Policy update failed for episode {episode_number}: {e}")
            self.policy_errors += 1
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy_loss': 0.0}

    def _calculate_gae(self, rewards, values, dones):
        """Calculate Generalized Advantage Estimation"""
        
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # Add bootstrap value for non-terminal episodes
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

    def _calculate_improvement_safe(self, original_sequence, final_sequence):
        """Safely calculate toughness improvement"""
        try:
            original_toughness, _ = self.environment.reward_fn.predict_toughness(original_sequence)
            final_toughness, _ = self.environment.reward_fn.predict_toughness(final_sequence)
            return final_toughness - original_toughness
        except Exception as e:
            logger.warning(f"Error calculating improvement: {e}")
            return 0.0

    def _create_fallback_result(self, episode_number, error=None):
        """Create fallback result for failed episodes"""
        return {
            'episode_number': episode_number,
            'episode_reward': -1.0,
            'episode_length': 0,
            'actual_improvement': 0.0,
            'final_sequence': '',
            'episode_time': 0.0,
            'difficulty_level': 'unknown',
            'avg_reward_ma': self.avg_reward,
            'avg_improvement_ma': self.avg_improvement,
            'current_lr': self.optimizer.param_groups[0]['lr'],
            'consecutive_successes': 0,
            'best_reward': self.best_reward,
            'best_improvement': self.best_improvement,
            'episode_summary': {'error': error} if error else {},
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy_loss': 0.0,
            'failed': True
        }

    def get_training_stats(self):
        """Get comprehensive training statistics"""
        if len(self.episode_rewards) == 0:
            return "No training data available"
        
        recent_window = min(100, len(self.episode_rewards))
        recent_rewards = self.episode_rewards[-recent_window:]
        recent_improvements = self.episode_improvements[-recent_window:]
        
        stats = f"""
Training Statistics:
  Episodes completed: {len(self.episode_rewards)}
  Average reward (recent {recent_window}): {np.mean(recent_rewards):.3f} ± {np.std(recent_rewards):.3f}
  Average improvement (recent {recent_window}): {np.mean(recent_improvements):.4f} ± {np.std(recent_improvements):.4f}
  Best reward: {self.best_reward:.3f}
  Best improvement: {self.best_improvement:.4f}
  Consecutive successes: {self.consecutive_successes}
  Current learning rate: {self.optimizer.param_groups[0]['lr']:.6f}
  
Error Counts:
  Tensor errors: {self.tensor_errors}
  Policy errors: {self.policy_errors}
  Environment errors: {self.environment_errors}
        """
        
        return stats.strip()


def load_models_and_data(config):
    """Load all required models and data with error handling"""
    logger.info("Loading models and data...")
    
    try:
        # Load dataset first
        dataset = SpiderSilkDataset(
            config.get('dataset_path', 'data/processed/sequences.csv'),
            test_size=config.get('test_size', 0.1),
            n_difficulty_levels=config.get('n_difficulty_levels', 3),
            random_state=config.get('seed', 42)
        )
        logger.info(f"Loaded {len(dataset.sequences)} sequences")
        
        # Load ESM-C model (following working pattern)
        logger.info("Loading ESM-C model...")
        esmc_checkpoint = config.get('esm_model_name', "src/models/checkpoint-1452")
        if not os.path.exists(esmc_checkpoint):
            logger.error(f"ESM-C checkpoint not found at {esmc_checkpoint}")
            raise FileNotFoundError(f"ESM-C checkpoint not found at {esmc_checkpoint}")
            
        esmc_model = AutoModelForMaskedLM.from_pretrained(esmc_checkpoint, trust_remote_code=True)
        esmc_tokenizer = esmc_model.tokenizer
        
        # Apply fix_both_warnings correctly (with tokenizer and model arguments)
        esmc_tokenizer, esmc_model = fix_both_warnings(esmc_tokenizer, esmc_model)
        logger.info("ESM-C model loaded successfully")
        
        # Load SilkomeGPT model (following working pattern)
        logger.info("Loading SilkomeGPT model...")
        trained_model_name = config.get('gpt_model_name', 'lamm-mit/SilkomeGPT')
        try:
            silkomegpt_tokenizer = AutoTokenizer.from_pretrained(trained_model_name, trust_remote_code=True)
            silkomegpt_tokenizer.pad_token = silkomegpt_tokenizer.eos_token
            silkomegpt_model = AutoModelForCausalLM.from_pretrained(
                trained_model_name,
                trust_remote_code=True
            )
            silkomegpt_model.config.use_cache = False
            logger.info("SilkomeGPT model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load SilkomeGPT: {e}")
            raise
        
        # Create components (following working pattern)
        utils = SpiderSilkUtils(esmc_model, esmc_tokenizer)
        
        # Use StableSpiderSilkRewardFunctionV2 with correct arguments
        reward_fn = StableSpiderSilkRewardFunctionV2(
            silkomegpt_model, silkomegpt_tokenizer, esmc_model
        )
        
        # Create policy (simplified to match working version)
        policy = ImprovedSequenceEditPolicyV2()
        
        # Create environment
        environment = EnhancedProteinEnvironment(
            reward_function=reward_fn,
            utils=utils,
            max_steps=config.get('max_episode_steps', 25)
        )
        
        logger.info("Successfully loaded all models and data")
        return policy, environment, dataset, utils, reward_fn
        
    except Exception as e:
        logger.error(f"Failed to load models and data: {e}")
        raise


def run_training_experiment(config_name, num_episodes, verbose=False):
    """Run complete training experiment with comprehensive logging"""
    
    # Set up experiment logging
    experiment_start_time = time.time()
    experiment_id = f"{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info(f"🚀 Starting experiment: {experiment_id}")
    logger.info(f"Configuration: {config_name}")
    logger.info(f"Episodes: {num_episodes}")
    logger.info(f"Verbose logging: {verbose}")
    
    if verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        # Load configuration
        config = get_stable_config_v2(config_name)
        config_dict = config.to_dict()  # Convert to dict for easier access
        
        # Override episodes if provided
        config_dict['n_episodes'] = num_episodes
        
        logger.info(f"Loaded configuration: {config_name}")
        logger.info(f"Learning rate: {config_dict.get('learning_rate', 1e-4)}")
        logger.info(f"Max steps: {config_dict.get('max_steps', 25)}")
        
        # Load models and data
        policy, environment, dataset, utils, reward_fn = load_models_and_data(config_dict)
        
        # Create trainer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move models to device
        policy = policy.to(device)
        reward_fn.silkomegpt.to(device)
        reward_fn.esmc.to(device)
        
        trainer = ComprehensiveFixedTrainer(
            policy=policy,
            environment=environment,
            lr=config_dict.get('learning_rate', 1e-4),
            device=device
        )
        
        logger.info(f"Training on device: {device}")
        logger.info(f"Policy parameters: {sum(p.numel() for p in policy.parameters())}")
        
        # Training loop with comprehensive logging
        episode_results = []
        progress_checkpoints = [int(num_episodes * p) for p in [0.1, 0.25, 0.5, 0.75, 0.9]]
        
        for episode in range(1, num_episodes + 1):
            try:
                # Get starting sequence using curriculum
                starting_sequence, difficulty_level = dataset.get_curriculum_sequence(
                    episode, num_episodes, config_dict.get('curriculum_strategy', 'easy_to_hard')
                )
                
                # Train episode
                result = trainer.train_episode(starting_sequence, episode, difficulty_level)
                episode_results.append(result)
                
                # Progress logging
                if episode in progress_checkpoints:
                    progress_pct = (episode / num_episodes) * 100
                    logger.info(f"🎯 Progress: {progress_pct:.0f}% ({episode}/{num_episodes})")
                    logger.info(trainer.get_training_stats())
                    logger.info(environment.get_comprehensive_stats())
                    logger.info(environment.action_space.get_action_stats())
                
                # Periodic detailed logging
                if episode % 100 == 0:
                    logger.info(f"Episode {episode}: "
                               f"Reward={result['episode_reward']:.3f}, "
                               f"Improvement={result['actual_improvement']:.4f}, "
                               f"Length={result['episode_length']}")
                
                # Success celebration
                if result['episode_reward'] > 2.0:
                    logger.info(f"🎉 Excellent episode {episode}! Reward: {result['episode_reward']:.3f}")
                
            except KeyboardInterrupt:
                logger.info(f"Training interrupted at episode {episode}")
                break
            except Exception as e:
                logger.error(f"Episode {episode} failed: {e}")
                continue
        
        # Final experiment summary
        experiment_time = time.time() - experiment_start_time
        successful_episodes = [r for r in episode_results if not r.get('failed', False)]
        
        logger.info(f"🏁 Experiment completed: {experiment_id}")
        logger.info(f"Total time: {experiment_time/60:.1f} minutes")
        logger.info(f"Successful episodes: {len(successful_episodes)}/{len(episode_results)}")
        
        if successful_episodes:
            final_rewards = [r['episode_reward'] for r in successful_episodes]
            final_improvements = [r['actual_improvement'] for r in successful_episodes]
            
            logger.info(f"Final stats:")
            logger.info(f"  Average reward: {np.mean(final_rewards):.3f} ± {np.std(final_rewards):.3f}")
            logger.info(f"  Average improvement: {np.mean(final_improvements):.4f} ± {np.std(final_improvements):.4f}")
            logger.info(f"  Best reward: {max(final_rewards):.3f}")
            logger.info(f"  Best improvement: {max(final_improvements):.4f}")
        
        # Final statistics
        logger.info(trainer.get_training_stats())
        logger.info(environment.get_comprehensive_stats())
        logger.info(environment.action_space.get_action_stats())
        
        return {
            'experiment_id': experiment_id,
            'config_name': config_name,
            'num_episodes': len(episode_results),
            'successful_episodes': len(successful_episodes),
            'experiment_time': experiment_time,
            'episode_results': episode_results,
            'final_stats': {
                'trainer_stats': trainer.get_training_stats(),
                'environment_stats': environment.get_comprehensive_stats(),
                'action_stats': environment.action_space.get_action_stats()
            }
        }
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(description='Run stable protein training experiment')
    parser.add_argument('--config', type=str, default='stable', 
                       choices=['stable', 'stable_test', 'stable_advanced'],
                       help='Configuration to use')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of episodes to train')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for training')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Run experiment
    try:
        results = run_training_experiment(
            config_name=args.config,
            num_episodes=args.episodes,
            verbose=args.verbose
        )
        
        logger.info("✅ Experiment completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Experiment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()