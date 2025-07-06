# src/environment/safe_protein_env.py
"""
Safe Protein Environment with comprehensive position bounds checking
This fixes the invalid_position errors that cause inconsistent performance
"""

from src.environment.protein_env import ProteinEditEnvironment
import torch
import logging

logger = logging.getLogger(__name__)

class SafeProteinEditEnvironment(ProteinEditEnvironment):
    """Environment with comprehensive position bounds checking and validation"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.position_errors = {
            'insertion_out_of_bounds': 0,
            'deletion_out_of_bounds': 0,
            'substitution_out_of_bounds': 0,
            'negative_positions': 0,
            'sequence_too_short': 0
        }
        self.debug_mode = True
    
    def step(self, action):
        """Enhanced step with comprehensive position validation"""
        if self.done:
            return self.get_state(), 0.0, True, {'already_done': True}

        old_sequence = self.current_sequence
        sequence_length = len(self.current_sequence)
        
        # 🚨 CRITICAL: Validate action before any processing
        validation_result = self._validate_action_completely(action, sequence_length)
        if not validation_result['valid']:
            # Log and fix the invalid action
            failure_type = validation_result['reason']
            self.position_errors[failure_type] = self.position_errors.get(failure_type, 0) + 1
            
            if self.debug_mode:
                logger.warning(f"Invalid action detected: {failure_type}")
                logger.warning(f"Action: {action}")
                logger.warning(f"Sequence length: {sequence_length}")
            
            # 🚨 FIX: Convert invalid action to valid stop action
            fixed_action = {
                'type': 'stop',
                'position': 0,
                'amino_acid': None,
                'log_prob': action.get('log_prob', torch.tensor(0.0))
            }
            
            # Return with penalty but don't crash
            self.done = True
            return self.get_state(), -2.0, True, {
                'original_action': action,
                'fixed_action': fixed_action,
                'failure_reason': failure_type,
                'position_errors': self.position_errors.copy()
            }

        # Handle stop action (always safe)
        if action['type'] == 'stop':
            self.done = True
            
            if len(self.edit_history) == 0:
                reward = -0.5  # Penalty for immediate stopping
            elif len(self.edit_history) < 3:
                reward = -0.1  # Small penalty for early stopping
            else:
                reward = 0.05  # Small reward for reasonable stopping
                
            return self.get_state(), reward, True, {
                'action': action,
                'edit_successful': True,
                'stop_reason': 'agent_choice'
            }

        # Execute edit action with safe bounds
        new_sequence = self._execute_action_safely(action, sequence_length)
        edit_successful = False

        if new_sequence != old_sequence:
            # Validate the resulting sequence
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
                edit_successful = True

                # Calculate reward
                reward_info = self.reward_fn.calculate_reward(
                    old_sequence, new_sequence, self.edit_history,
                    self.original_sequence, self.episode_number
                )
                reward = reward_info['total']
                self.done = reward_info.get('done', False)

            else:
                # Invalid edit - stronger penalty
                reward = -1.0
                edit_info = {
                    'type': 'invalid_edit', 
                    'message': message,
                    'attempted_action': action['type'],
                    'validation_status': 'failed'
                }
        else:
            # No sequence change - penalty for ineffective action
            reward = -0.3
            edit_info = {
                'type': 'no_change', 
                'validation_status': 'no_edit',
                'reason': 'action_had_no_effect'
            }

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
            'position_errors': self.position_errors.copy()
        }

        return self.get_state(), reward, self.done, info
    
    def _validate_action_completely(self, action, sequence_length):
        """Comprehensive action validation with specific error categorization"""
        
        # Basic structure validation
        if not isinstance(action, dict) or 'type' not in action:
            return {'valid': False, 'reason': 'invalid_action_format'}
        
        action_type = action['type']
        
        # Stop action is always valid
        if action_type == 'stop':
            return {'valid': True, 'reason': 'stop_action'}
        
        # Check valid action types
        if action_type not in ['substitution', 'insertion', 'deletion']:
            return {'valid': False, 'reason': 'invalid_action_type'}
        
        # Position validation
        position = action.get('position')
        if position is None:
            return {'valid': False, 'reason': 'missing_position'}
        
        if not isinstance(position, (int, float)) or isinstance(position, bool):
            return {'valid': False, 'reason': 'invalid_position_type'}
        
        position = int(position)  # Ensure integer
        
        # Check for negative positions
        if position < 0:
            return {'valid': False, 'reason': 'negative_positions'}
        
        # Check sequence length constraints
        if sequence_length < 10:
            return {'valid': False, 'reason': 'sequence_too_short'}
        
        # 🚨 CRITICAL: Action-specific position bounds checking
        if action_type == 'insertion':
            # Insertions can be at positions 0 to sequence_length (inclusive)
            if position > sequence_length:
                return {'valid': False, 'reason': 'insertion_out_of_bounds'}
        
        elif action_type == 'substitution':
            # Substitutions must be at positions 0 to sequence_length-1
            if position >= sequence_length:
                return {'valid': False, 'reason': 'substitution_out_of_bounds'}
        
        elif action_type == 'deletion':
            # Deletions must be at positions 0 to sequence_length-1
            if position >= sequence_length:
                return {'valid': False, 'reason': 'deletion_out_of_bounds'}
            
            # Additional check: don't delete if sequence is too short
            if sequence_length <= 15:
                return {'valid': False, 'reason': 'sequence_too_short'}
        
        # Amino acid validation for substitution/insertion
        if action_type in ['substitution', 'insertion']:
            amino_acid = action.get('amino_acid')
            if not amino_acid or amino_acid not in 'ACDEFGHIKLMNPQRSTVWY':
                return {'valid': False, 'reason': 'invalid_amino_acid'}
        
        return {'valid': True, 'reason': 'valid_action'}
    
    def _execute_action_safely(self, action, sequence_length):
        """Execute action with additional safety checks"""
        
        action_type = action['type']
        position = int(action['position'])  # Ensure integer
        
        try:
            if action_type == 'substitution':
                # Double-check bounds before substitution
                if 0 <= position < sequence_length:
                    new_seq = (self.current_sequence[:position] +
                              action['amino_acid'] +
                              self.current_sequence[position+1:])
                    return new_seq
                else:
                    logger.warning(f"Substitution bounds check failed: pos={position}, len={sequence_length}")
                    return self.current_sequence

            elif action_type == 'insertion':
                # Double-check bounds before insertion
                if 0 <= position <= sequence_length:
                    new_seq = (self.current_sequence[:position] +
                              action['amino_acid'] +
                              self.current_sequence[position:])
                    return new_seq
                else:
                    logger.warning(f"Insertion bounds check failed: pos={position}, len={sequence_length}")
                    return self.current_sequence

            elif action_type == 'deletion':
                # Double-check bounds and length before deletion
                if (0 <= position < sequence_length and 
                    sequence_length > 15):
                    new_seq = self.current_sequence[:position] + self.current_sequence[position+1:]
                    return new_seq
                else:
                    logger.warning(f"Deletion bounds check failed: pos={position}, len={sequence_length}")
                    return self.current_sequence

        except Exception as e:
            logger.error(f"Exception in action execution: {e}")
            return self.current_sequence
        
        # If we get here, something went wrong
        logger.warning(f"Action execution failed for {action_type} at position {position}")
        return self.current_sequence
    
    def get_position_error_summary(self):
        """Get summary of position errors for debugging"""
        total_errors = sum(self.position_errors.values())
        if total_errors == 0:
            return "No position errors recorded"
        
        summary = f"Total position errors: {total_errors}\n"
        for error_type, count in self.position_errors.items():
            if count > 0:
                percentage = (count / total_errors) * 100
                summary += f"  {error_type}: {count} ({percentage:.1f}%)\n"
        
        return summary
    
    def reset_error_tracking(self):
        """Reset error tracking for new training session"""
        self.position_errors = {
            'insertion_out_of_bounds': 0,
            'deletion_out_of_bounds': 0,
            'substitution_out_of_bounds': 0,
            'negative_positions': 0,
            'sequence_too_short': 0
        }


# ===================================
# FIXED ACTION SPACE WITH BOUNDS CHECKING
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
        
        # 🚨 CRITICAL: Validate sequence_length first
        if sequence_length < 5:
            # Sequence too short, force stop
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
            # Insertions: position can be 0 to sequence_length (inclusive)
            max_position = sequence_length
        else:
            # Substitutions and deletions: position can be 0 to sequence_length-1
            max_position = sequence_length - 1
        
        # Ensure we have valid positions
        if max_position < 0:
            # No valid positions, force stop
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
            # No valid positions, force stop
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
        
        # 🚨 FINAL VALIDATION: Double-check position bounds
        if edit_type == 'insertion' and position > sequence_length:
            logger.warning(f"Insertion position {position} > {sequence_length}, forcing stop")
            return {'type': 'stop', 'position': 0, 'amino_acid': None, 'log_prob': torch.tensor(0.0, device=device)}
        
        if edit_type in ['substitution', 'deletion'] and position >= sequence_length:
            logger.warning(f"{edit_type} position {position} >= {sequence_length}, forcing stop")
            return {'type': 'stop', 'position': 0, 'amino_acid': None, 'log_prob': torch.tensor(0.0, device=device)}
        
        # 3) Sample amino acid (if needed)
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
# UPDATED POLICY WITH BETTER BOUNDS HANDLING
# ===================================

def update_policy_get_action_method():
    """
    This function shows how to update the policy's get_action method
    to handle bounds checking properly
    """
    
    def get_action_safe(self, state, deterministic=False, sequence_length=None):
        """SAFE action selection with comprehensive bounds checking"""
        device = next(self.parameters()).device
        state = state.to(device)
        
        if sequence_length is None:
            sequence_length = self.default_seq_length
        
        # 🚨 CRITICAL: Ensure sequence_length is reasonable
        sequence_length = max(5, min(sequence_length, 800))
        
        with torch.no_grad():
            if state.dim() == 1:
                output = self.forward(state.unsqueeze(0))
                squeeze_output = True
            else:
                output = self.forward(state)
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
                
                # 🚨 CRITICAL FIX: Proper position bounds with validation
                if edit_type == 'insertion':
                    max_valid_pos = sequence_length  # Can insert at end
                else:  # substitution or deletion
                    max_valid_pos = sequence_length - 1  # Must be within sequence
                
                # Safety check for valid positions
                if max_valid_pos < 0:
                    # No valid positions, return stop
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
                    # No valid positions, return stop
                    return {
                        'type': 'stop',
                        'position': 0,
                        'amino_acid': None,
                        'log_prob': torch.tensor(0.0, device=device)
                    }
                
                # 🚨 TRIPLE CHECK: Final bounds validation
                if edit_type == 'insertion' and position_idx > sequence_length:
                    logger.error(f"BOUNDS ERROR: insertion pos {position_idx} > {sequence_length}")
                    return {'type': 'stop', 'position': 0, 'amino_acid': None, 'log_prob': torch.tensor(0.0, device=device)}
                elif edit_type in ['substitution', 'deletion'] and position_idx >= sequence_length:
                    logger.error(f"BOUNDS ERROR: {edit_type} pos {position_idx} >= {sequence_length}")
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
    
    return get_action_safe


# ===================================
# USAGE INSTRUCTIONS
# ===================================

def create_safe_environment(utils, reward_fn, max_steps=50):
    """Create a safe environment with position bounds checking"""
    env = SafeProteinEditEnvironment(utils, reward_fn, max_steps)
    # Replace the action space with the safe version
    env.action_space = SafeSequenceActionSpace()
    return env

def update_existing_environment(env):
    """Update existing environment with safety features"""
    # Replace methods with safe versions
    original_step = env.step
    safe_env = SafeProteinEditEnvironment(env.utils, env.reward_fn, env.max_steps)
    env.step = safe_env.step
    env._validate_action_completely = safe_env._validate_action_completely
    env._execute_action_safely = safe_env._execute_action_safely
    env.position_errors = safe_env.position_errors
    env.action_space = SafeSequenceActionSpace()
    return env

# Example usage:
"""
# In your training script, replace:
env = ProteinEditEnvironment(utils, reward_fn, max_steps=config['max_steps'])

# With:
env = create_safe_environment(utils, reward_fn, max_steps=config['max_steps'])

# And update your policy's get_action method:
policy.get_action = update_policy_get_action_method().__get__(policy, type(policy))
"""