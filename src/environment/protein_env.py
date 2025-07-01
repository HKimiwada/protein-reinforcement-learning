from src.environment.action_space import SequenceActionSpace
from src.environment.state_encoder import SequenceStateEncoder

class ProteinEditEnvironment:
    def __init__(self, utils, reward_function, max_steps=50):
        self.utils = utils
        self.reward_fn = reward_function
        self.max_steps = max_steps
        self.action_space = SequenceActionSpace()
        self.state_encoder = SequenceStateEncoder(utils)

        # Episode state
        self.reset()

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
        """Execute action with fixed stop incentives"""
        if self.done:
            return self.get_state(), 0.0, True, {}

        old_sequence = self.current_sequence

        # 🚀 FIXED: Handle stop action with conditional reward based on effort
        if action['type'] == 'stop':
            self.done = True
            
            # Stop reward depends on whether any progress was made
            if len(self.edit_history) == 0:
                # No edits made - big penalty for giving up immediately
                reward = -0.5
            elif len(self.edit_history) < 3:
                # Few edits made - small penalty for giving up early
                reward = -0.1
            else:
                # Many edits made - small reward for reasonable stopping
                reward = 0.05
                
            return self.get_state(), reward, True, {'action': action}

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

                # Calculate reward
                reward_info = self.reward_fn.calculate_reward(
                    old_sequence, new_sequence, self.edit_history,
                    self.original_sequence, self.episode_number
                )
                reward = reward_info['total']
                self.done = reward_info.get('done', False)

            else:
                # 🚀 FIXED: Stronger penalty for invalid edits to discourage them
                reward = -0.5  # Increased from -0.2
                edit_info = {
                    'type': 'invalid', 
                    'message': message,
                    'attempted_action': action['type'],
                    'validation_status': 'failed'
                }
                
                # Add specific validation failure info for learning
                if 'Perplexity too high' in message:
                    edit_info['failure_reason'] = 'perplexity'
                    edit_info['suggested_fix'] = 'try_conservative_substitution'
                elif 'Length ratio' in message:
                    edit_info['failure_reason'] = 'length'
                    edit_info['suggested_fix'] = 'avoid_insertions_deletions'
                elif 'motifs missing' in message:
                    edit_info['failure_reason'] = 'motifs'
                    edit_info['suggested_fix'] = 'preserve_structure'
        else:
            # 🚀 FIXED: Stronger penalty for no-change to discourage ineffective actions
            reward = -0.2  # Increased from -0.1
            edit_info = {'type': 'no_change', 'validation_status': 'no_edit'}

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
            'edit_count': len(self.edit_history)
        }

        return self.get_state(), reward, self.done, info

    def _execute_action(self, action):
        """Execute the action and return new sequence"""
        if action['type'] == 'substitution':
            pos = action['position']
            if pos < len(self.current_sequence):
                new_seq = (self.current_sequence[:pos] +
                          action['amino_acid'] +
                          self.current_sequence[pos+1:])
                return new_seq

        elif action['type'] == 'insertion':
            pos = action['position']
            # 🚀 FIXED: Better bounds checking for insertions
            if 0 <= pos <= len(self.current_sequence):
                new_seq = (self.current_sequence[:pos] +
                          action['amino_acid'] +
                          self.current_sequence[pos:])
                return new_seq

        elif action['type'] == 'deletion':
            pos = action['position']
            # 🚀 FIXED: More conservative deletion requirements
            if (0 <= pos < len(self.current_sequence) and 
                len(self.current_sequence) > 20):  # Increased from 10 to 20
                new_seq = self.current_sequence[:pos] + self.current_sequence[pos+1:]
                return new_seq

        # If we get here, action couldn't be executed
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