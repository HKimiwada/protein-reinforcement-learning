# src/environment/enhanced_protein_env.py
# Enhanced environment with better error handling and logging

from src.environment.protein_env import ProteinEditEnvironment
import logging

logger = logging.getLogger(__name__)

class EnhancedProteinEditEnvironment(ProteinEditEnvironment):
    """Enhanced environment with better error handling and failure logging"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.failure_counts = {
            'invalid_position': 0,
            'invalid_edit': 0,
            'perplexity_too_high': 0,
            'sequence_too_short': 0,
            'other_failures': 0
        }
        self.debug_mode = True  # Enable detailed logging
    
    def step(self, action):
        """Enhanced step with detailed failure tracking"""
        if self.done:
            return self.get_state(), 0.0, True, {'failure_reason': 'already_done'}

        old_sequence = self.current_sequence
        
        # 🚨 ENHANCED: Validate action thoroughly before execution
        validation_result = self._validate_action_thoroughly(action)
        if not validation_result['valid']:
            # Log the failure
            failure_reason = validation_result['reason']
            self.failure_counts[failure_reason] = self.failure_counts.get(failure_reason, 0) + 1
            
            if self.debug_mode:
                logger.warning(f"Invalid action: {failure_reason}")
                logger.warning(f"Action: {action}")
                logger.warning(f"Sequence length: {len(self.current_sequence)}")
            
            # Return heavy penalty for invalid actions
            return self.get_state(), -5.0, False, {
                'failure_reason': failure_reason,
                'action_attempted': action,
                'edit_successful': False
            }

        # Handle stop action
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

        # Execute edit action with enhanced error handling
        try:
            new_sequence = self._execute_action_safely(action)
            edit_successful = False

            if new_sequence != old_sequence:
                # Validate the resulting sequence
                validation_result = self._validate_sequence_result(old_sequence, new_sequence)
                
                if validation_result['valid']:
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
                        'validation_status': 'valid'
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
                    # Sequence validation failed
                    failure_reason = validation_result['reason']
                    self.failure_counts[failure_reason] = self.failure_counts.get(failure_reason, 0) + 1
                    
                    if self.debug_mode:
                        logger.warning(f"Sequence validation failed: {failure_reason}")
                    
                    reward = -2.0  # Heavy penalty for invalid sequences
                    edit_info = {
                        'type': 'invalid_sequence',
                        'reason': failure_reason,
                        'validation_status': 'failed'
                    }
            else:
                # No sequence change
                reward = -0.3  # Penalty for ineffective actions
                edit_info = {'type': 'no_change', 'validation_status': 'no_edit'}
                
        except Exception as e:
            # Catch any unexpected errors
            self.failure_counts['other_failures'] += 1
            if self.debug_mode:
                logger.error(f"Unexpected error in step: {e}")
            
            reward = -5.0
            edit_info = {
                'type': 'error',
                'error': str(e),
                'validation_status': 'error'
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
            'failure_counts': self.failure_counts.copy()
        }

        return self.get_state(), reward, self.done, info
    
    def _validate_action_thoroughly(self, action):
        """Thorough action validation before execution"""
        if not isinstance(action, dict) or 'type' not in action:
            return {'valid': False, 'reason': 'invalid_action_format'}
        
        action_type = action['type']
        
        if action_type == 'stop':
            return {'valid': True, 'reason': 'stop_action'}
        
        if action_type not in ['substitution', 'insertion', 'deletion']:
            return {'valid': False, 'reason': 'invalid_action_type'}
        
        position = action.get('position')
        if position is None or not isinstance(position, int):
            return {'valid': False, 'reason': 'invalid_position_type'}
        
        seq_len = len(self.current_sequence)
        
        # Position bounds checking
        if action_type == 'insertion':
            if position < 0 or position > seq_len:
                return {'valid': False, 'reason': 'invalid_position'}
        else:  # substitution or deletion
            if position < 0 or position >= seq_len:
                return {'valid': False, 'reason': 'invalid_position'}
        
        # Amino acid validation for substitution/insertion
        if action_type in ['substitution', 'insertion']:
            amino_acid = action.get('amino_acid')
            if not amino_acid or amino_acid not in 'ACDEFGHIKLMNPQRSTVWY':
                return {'valid': False, 'reason': 'invalid_amino_acid'}
        
        # Sequence length constraints
        if action_type == 'deletion' and seq_len <= 10:
            return {'valid': False, 'reason': 'sequence_too_short'}
        
        return {'valid': True, 'reason': 'valid_action'}
    
    def _execute_action_safely(self, action):
        """Safely execute action with additional bounds checking"""
        try:
            if action['type'] == 'substitution':
                pos = action['position']
                if 0 <= pos < len(self.current_sequence):
                    new_seq = (self.current_sequence[:pos] +
                              action['amino_acid'] +
                              self.current_sequence[pos+1:])
                    return new_seq

            elif action['type'] == 'insertion':
                pos = action['position']
                if 0 <= pos <= len(self.current_sequence):
                    new_seq = (self.current_sequence[:pos] +
                              action['amino_acid'] +
                              self.current_sequence[pos:])
                    return new_seq

            elif action['type'] == 'deletion':
                pos = action['position']
                if 0 <= pos < len(self.current_sequence) and len(self.current_sequence) > 15:
                    new_seq = self.current_sequence[:pos] + self.current_sequence[pos+1:]
                    return new_seq

        except Exception as e:
            logger.error(f"Action execution error: {e}")
        
        # If we get here, action couldn't be executed safely
        return self.current_sequence
    
    def _validate_sequence_result(self, old_sequence, new_sequence):
        """Validate the resulting sequence is reasonable"""
        try:
            # Basic length check
            if len(new_sequence) < 10:
                return {'valid': False, 'reason': 'sequence_too_short'}
            
            if len(new_sequence) > 1000:
                return {'valid': False, 'reason': 'sequence_too_long'}
            
            # Check for valid amino acids only
            valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
            if not all(aa in valid_aas for aa in new_sequence):
                return {'valid': False, 'reason': 'invalid_amino_acids'}
            
            # Length ratio check
            length_ratio = len(new_sequence) / len(old_sequence)
            if not (0.7 <= length_ratio <= 1.3):
                return {'valid': False, 'reason': 'length_ratio_violation'}
            
            # Quick perplexity check (if it fails, the sequence is probably broken)
            try:
                perplexity = self.utils.calculate_perplexity(new_sequence)
                if perplexity > 20.0:
                    return {'valid': False, 'reason': 'perplexity_too_high'}
            except:
                return {'valid': False, 'reason': 'perplexity_calculation_failed'}
            
            return {'valid': True, 'reason': 'sequence_valid'}
            
        except Exception as e:
            return {'valid': False, 'reason': f'validation_error_{str(e)[:20]}'}
    
    def get_failure_summary(self):
        """Get summary of all failures for debugging"""
        total_failures = sum(self.failure_counts.values())
        if total_failures == 0:
            return "No failures recorded"
        
        summary = f"Total failures: {total_failures}\n"
        for failure_type, count in self.failure_counts.items():
            if count > 0:
                percentage = (count / total_failures) * 100
                summary += f"  {failure_type}: {count} ({percentage:.1f}%)\n"
        
        return summary