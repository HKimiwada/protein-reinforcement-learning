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
        """Execute action and return (state, reward, done, info)"""
        if self.done:
            return self.get_state(), 0.0, True, {}

        old_sequence = self.current_sequence

        # Handle stop action
        if action['type'] == 'stop':
            self.done = True
            # Small reward for choosing to stop
            reward = 0.1
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
                    'step': self.step_count
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
                # Invalid edit - penalty
                reward = -0.2
                edit_info = {'type': 'invalid', 'message': message}
        else:
            # No change - small penalty
            reward = -0.1
            edit_info = {'type': 'no_change'}

        # Update step count
        self.step_count += 1

        # Check termination conditions
        if self.step_count >= self.max_steps:
            self.done = True

        info = {
            'action': action,
            'edit_info': edit_info,
            'edit_successful': edit_successful,
            'sequence_length': len(self.current_sequence)
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
            new_seq = (self.current_sequence[:pos] +
                      action['amino_acid'] +
                      self.current_sequence[pos:])
            return new_seq

        elif action['type'] == 'deletion':
            pos = action['position']
            if pos < len(self.current_sequence) and len(self.current_sequence) > 10:
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