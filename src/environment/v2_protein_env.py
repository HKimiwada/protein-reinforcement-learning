# v2_protein_env.py
from src.environment.action_space import SequenceActionSpace
from src.environment.state_encoder import SequenceStateEncoder

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

        # 🚀 FIXED: Prevent premature stopping before minimum episode length
        if action['type'] == 'stop':
            if self.step_count < self.min_episode_length:
                # FORCE CONVERSION: Convert stop to a random valid edit action
                print(f"🛑 BLOCKED: Stop action at step {self.step_count} < {self.min_episode_length}, converting to substitution")
                action = {
                    'type': 'substitution',
                    'position': min(self.step_count, len(self.current_sequence) - 1),
                    'amino_acid': 'A',  # Safe default
                    'log_prob': action.get('log_prob', 0.0)
                }
            else:
                # Allow stop after minimum length
                self.done = True
                
                # Stop reward depends on whether any progress was made
                if len(self.edit_history) == 0:
                    reward = -0.5
                elif len(self.edit_history) < 3:
                    reward = -0.1
                else:
                    reward = 0.05
                    
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

        # 🚀 FIXED: Only check max_steps termination, let reward function handle other termination
        if self.step_count >= self.max_steps:
            self.done = True
            print(f"⏰ FORCED TERMINATION: Reached max_steps {self.max_steps}")

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