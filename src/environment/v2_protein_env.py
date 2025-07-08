from src.environment.action_space import SequenceActionSpace
from src.environment.state_encoder import SequenceStateEncoder
import logging

logger = logging.getLogger(__name__)

class ProteinEditEnvironmentV2:
    def __init__(self, utils, reward_function, max_steps=50):
        self.utils = utils
        self.reward_fn = reward_function
        self.max_steps = max_steps
        self.action_space = SequenceActionSpace()
        self.state_encoder = SequenceStateEncoder(utils)

        # Episode state
        self.reset()
        
        # FIXED: Add minimum episode length enforcement
        self.min_episode_length = 8  # Must match reward function

    def reset(self, initial_sequence=None):
        """Reset environment with new sequence"""
        if initial_sequence is None:
            # Default sequence if none provided
            initial_sequence = "GPGGQGPYGPGGQGPGGQGPYGPQAAAAAAAAAAAGPGGQGPYGPGGQ"

        self.original_sequence = initial_sequence
        self.current_sequence = initial_sequence
        self.edit_history = []
        self.step_count = 0
        self.done = False
        self.episode_number = 0

        return self.get_state()

    def step(self, action):
        """Execute action with FIXED minimum episode length enforcement"""
        if self.done:
            return self.get_state(), 0.0, True, {}

        old_sequence = self.current_sequence

        # 🚀 ENHANCED: Selective stop action blocking based on cumulative improvement
        if action['type'] == 'stop':
            if self.step_count < self.min_episode_length:
                # Always block stop before minimum length
                print(f"🛑 BLOCKED: Stop action at step {self.step_count} < {self.min_episode_length}, converting to substitution")
                action = {
                    'type': 'substitution',
                    'position': min(self.step_count, len(self.current_sequence) - 1),
                    'amino_acid': 'A',  # Safe default
                    'log_prob': action.get('log_prob', 0.0)
                }
            else:
                # Check cumulative improvement to decide if stop is allowed
                try:
                    original_toughness, _ = self.reward_fn.predict_toughness(self.original_sequence)
                    current_toughness, _ = self.reward_fn.predict_toughness(self.current_sequence)
                    cumulative_improvement = current_toughness - original_toughness
                except:
                    cumulative_improvement = 0.0
                
                # SELECTIVE BLOCKING: Only block stop if there's zero improvement
                if cumulative_improvement <= 0.001:  # No meaningful improvement
                    print(f"🚫 BLOCKED: Stop action with zero improvement ({cumulative_improvement:.6f}), converting to substitution")
                    print(f"   Forcing exploration via reward function logic...")
                    action = {
                        'type': 'substitution',
                        'position': min(self.step_count % len(self.current_sequence), len(self.current_sequence) - 1),
                        'amino_acid': 'G',  # Safe amino acid
                        'log_prob': action.get('log_prob', 0.0)
                    }
                else:
                    # Allow stop after meaningful improvement
                    print(f"✅ ALLOWING: Stop action with improvement ({cumulative_improvement:.6f})")
                    self.done = True
                    
                    # Reward based on improvement achieved
                    if cumulative_improvement > 0.01:
                        reward = 0.3  # Good improvement
                    elif cumulative_improvement > 0.005:
                        reward = 0.2  # Decent improvement
                    else:
                        reward = 0.1  # Small but meaningful improvement
                    
                    # LOG EPISODE COMPLETION FOR STOP ACTION
                    self._log_episode_completion("stop_action", reward)
                        
                    return self.get_state(), reward, True, {'action': action, 'stop_allowed': True}

        # Execute edit action
        new_sequence = self._execute_action(action)
        edit_successful = False

        # Validate edit
        if new_sequence != old_sequence:
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

                # Calculate reward using reward function (which has proper termination logic)
                reward_info = self.reward_fn.calculate_reward(
                    old_sequence, new_sequence, self.edit_history,
                    self.original_sequence, self.episode_number
                )
                reward = reward_info['total']
                
                # 🚀 FIXED: Only allow termination from reward function AFTER minimum length
                if self.step_count >= self.min_episode_length:
                    self.done = reward_info.get('done', False)
                else:
                    self.done = False  # Force continuation regardless of reward function

            else:
                # 🚀 FIXED: For invalid edits, try to recover instead of heavy penalty
                if self.step_count < self.min_episode_length:
                    # Convert invalid action to a safe substitution
                    print(f"🔧 RECOVERING: Invalid action at step {self.step_count}, converting to safe substitution")
                    safe_pos = min(self.step_count, len(self.current_sequence) - 1)
                    safe_action = {
                        'type': 'substitution',
                        'position': safe_pos,
                        'amino_acid': 'G',  # Glycine is usually safe
                        'log_prob': action.get('log_prob', 0.0)
                    }
                    # Recursively call step with safe action
                    return self.step(safe_action)
                else:
                    # After minimum length, allow normal invalid action penalty
                    reward = -0.5
                    edit_info = {
                        'type': 'invalid', 
                        'message': message,
                        'attempted_action': action['type'],
                        'validation_status': 'failed'
                    }
        else:
            # 🚀 FIXED: For no-change actions, try alternative instead of penalty if before min length
            if self.step_count < self.min_episode_length:
                print(f"🔄 RETRY: No-change action at step {self.step_count}, trying alternative")
                # Try a different position
                alt_pos = (action.get('position', 0) + 1) % len(self.current_sequence)
                alt_action = {
                    'type': 'substitution',
                    'position': alt_pos,
                    'amino_acid': 'A',
                    'log_prob': action.get('log_prob', 0.0)
                }
                return self.step(alt_action)
            else:
                # After minimum length, allow normal no-change penalty
                reward = -0.2
                edit_info = {'type': 'no_change', 'validation_status': 'no_edit'}

        # Update step count
        self.step_count += 1

        # 🚀 ENHANCED: Check max_steps termination with logging
        if self.step_count >= self.max_steps:
            self.done = True
            print(f"⏰ FORCED TERMINATION: Reached max_steps {self.max_steps}")
            
            # LOG EPISODE COMPLETION FOR MAX STEPS TERMINATION
            self._log_episode_completion("max_steps", reward)

        info = {
            'action': action,
            'edit_info': edit_info,
            'edit_successful': edit_successful,
            'sequence_length': len(self.current_sequence),
            'sequence_changed': old_sequence != self.current_sequence,
            'edit_count': len(self.edit_history),
            'step_count': self.step_count,
            'min_length_reached': self.step_count >= self.min_episode_length
        }

        return self.get_state(), reward, self.done, info

    def _log_episode_completion(self, termination_reason, final_reward):
        """Log episode completion when terminated by environment (not reward function)"""
        
        # Calculate toughness changes
        try:
            original_toughness, _ = self.reward_fn.predict_toughness(self.original_sequence)
            final_toughness, _ = self.reward_fn.predict_toughness(self.current_sequence)
            cumulative_improvement = final_toughness - original_toughness
        except:
            original_toughness = 0.0
            final_toughness = 0.0
            cumulative_improvement = 0.0
        
        # Get last step improvement
        last_improvement = 0.0
        if self.edit_history:
            last_edit = self.edit_history[-1]
            last_improvement = last_edit.get('toughness_improvement', 0.0)
        
        edit_count = len(self.edit_history)
        
        # LOG EPISODE COMPLETION (matching reward function format)
        logger.info(f"🏁 EPISODE {self.episode_number} COMPLETE (Steps: {edit_count}) 🏁")
        logger.info(f"   📊 TOUGHNESS CHANGE:")
        logger.info(f"      Original: {original_toughness:.6f}")
        logger.info(f"      Final:    {final_toughness:.6f}")
        logger.info(f"      Total Δ:  {cumulative_improvement:+.6f} ({cumulative_improvement*100:+.3f}%)")
        logger.info(f"      Last Δ:   {last_improvement:+.6f}")
        logger.info(f"   🎯 PERFORMANCE:")
        logger.info(f"      Final Reward: {final_reward:.3f}")
        logger.info(f"      Edits Made:   {edit_count}")
        logger.info(f"      Success:      {'✅ YES' if cumulative_improvement > 0.001 else '❌ NO'}")
        logger.info(f"   🛑 TERMINATION: {termination_reason}")
        
        # Special logging for different termination types
        if termination_reason == "max_steps":
            logger.info(f"   ⏰ Reached environment maximum steps ({self.max_steps})")
        elif termination_reason == "stop_action":
            logger.info(f"   🛑 Agent chose to stop after {edit_count} edits")
        
        logger.info(f"   " + "="*60)

    def _execute_action(self, action):
        """Execute the action and return new sequence with better bounds checking"""
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
            if (0 <= pos < len(self.current_sequence) and 
                len(self.current_sequence) > 20):
                new_seq = self.current_sequence[:pos] + self.current_sequence[pos+1:]
                return new_seq

        # If action couldn't be executed, return unchanged sequence
        return self.current_sequence

    def get_state(self):
        """Get current state representation"""
        return self.state_encoder.encode_state(
            self.current_sequence,
            self.edit_history,
            self.step_count,
            self.original_sequence
        )

    def set_episode_number(self, episode_num):
        """Set episode number for reward calculation"""
        self.episode_number = episode_num

    def get_episode_summary(self):
        """Get summary of current episode for debugging"""
        total_improvement = sum(
            edit.get('toughness_improvement', 0.0) 
            for edit in self.edit_history
        )
        
        edit_types = [edit.get('type', 'unknown') for edit in self.edit_history]
        edit_type_counts = {
            'substitution': edit_types.count('substitution'),
            'insertion': edit_types.count('insertion'),
            'deletion': edit_types.count('deletion'),
            'invalid': edit_types.count('invalid')
        }
        
        return {
            'total_edits': len(self.edit_history),
            'total_improvement': total_improvement,
            'edit_type_counts': edit_type_counts,
            'final_sequence_length': len(self.current_sequence),
            'original_sequence_length': len(self.original_sequence),
            'steps_taken': self.step_count,
            'done': self.done
        }