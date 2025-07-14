from src.environment.action_space import SequenceActionSpace
from src.environment.state_encoder import SequenceStateEncoder
import logging

logger = logging.getLogger(__name__)

class ProteinEditEnvironmentV2:
    def __init__(self, utils, reward_function, max_steps=50):
        self.utils = utils
        self.reward_fn = reward_function
        self.max_steps = max_steps  # This should be action attempts, not successful edits
        self.action_space = SequenceActionSpace()
        self.state_encoder = SequenceStateEncoder(utils)

        # Episode state
        self.reset()
        
        # FIXED: Add minimum episode length enforcement
        self.min_episode_length = 8  # Must match reward function

    def reset(self, initial_sequence=None):
        """Reset environment with new sequence"""
        if initial_sequence is None:
            initial_sequence = "GPGGQGPYGPGGQGPGGQGPYGPQAAAAAAAAAAAGPGGQGPYGPGGQ"

        self.original_sequence = initial_sequence
        self.current_sequence = initial_sequence
        self.edit_history = []
        self.step_count = 0  # CLARIFICATION: This counts action attempts, not successful edits
        self.done = False
        self.episode_number = 0

        return self.get_state()

    def step(self, action):
        """Execute action with FIXED step counting and better max_steps handling"""
        if self.done:
            return self.get_state(), 0.0, True, {}

        old_sequence = self.current_sequence
        
        # 🚀 ENHANCED: Track action attempts vs successful edits separately
        action_attempt = True
        edit_successful = False

        # ENHANCED: Selective stop action blocking based on cumulative improvement
        if action['type'] == 'stop':
            if self.step_count < self.min_episode_length:
                # Always block stop before minimum action attempts
                print(f"🛑 BLOCKED: Stop action at step {self.step_count} < {self.min_episode_length}, converting to substitution")
                action = {
                    'type': 'substitution',
                    'position': min(self.step_count, len(self.current_sequence) - 1),
                    'amino_acid': 'A',
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
                if cumulative_improvement <= 0.001:
                    print(f"🚫 BLOCKED: Stop action with zero improvement ({cumulative_improvement:.6f}), converting to substitution")
                    print(f"   Forcing exploration via reward function logic...")
                    action = {
                        'type': 'substitution',
                        'position': min(self.step_count % len(self.current_sequence), len(self.current_sequence) - 1),
                        'amino_acid': 'G',
                        'log_prob': action.get('log_prob', 0.0)
                    }
                else:
                    # Allow stop after meaningful improvement
                    print(f"✅ ALLOWING: Stop action with improvement ({cumulative_improvement:.6f})")
                    self.done = True
                    
                    # Reward based on improvement achieved
                    if cumulative_improvement > 0.01:
                        reward = 0.3
                    elif cumulative_improvement > 0.005:
                        reward = 0.2
                    else:
                        reward = 0.1
                    
                    # Update step count for this final action
                    self.step_count += 1
                    
                    # LOG EPISODE COMPLETION FOR STOP ACTION
                    self._log_episode_completion("stop_action", reward)
                        
                    return self.get_state(), reward, True, {
                        'action': action, 
                        'stop_allowed': True,
                        'edit_successful': False,
                        'action_attempts': self.step_count,
                        'successful_edits': len(self.edit_history)
                    }

        # Execute edit action
        new_sequence = self._execute_action(action)

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
                    'step': self.step_count,  # Action attempt number
                    'edit_number': len(self.edit_history),  # Successful edit number
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

                # Calculate reward using reward function
                reward_info = self.reward_fn.calculate_reward(
                    old_sequence, new_sequence, self.edit_history,
                    self.original_sequence, self.episode_number
                )
                reward = reward_info['total']
                
                # Only allow termination from reward function AFTER minimum attempts
                if self.step_count >= self.min_episode_length:
                    self.done = reward_info.get('done', False)
                else:
                    self.done = False

            else:
                # Invalid edit - don't add to edit_history but still count as action attempt
                if self.step_count < self.min_episode_length:
                    print(f"🔧 INVALID ACTION at step {self.step_count}, giving penalty but continuing")
                    reward = -0.3  # Penalty for invalid action
                    edit_info = {
                        'type': 'invalid', 
                        'message': message,
                        'attempted_action': action['type'],
                        'validation_status': 'failed',
                        'step': self.step_count
                    }
                else:
                    reward = -0.5
                    edit_info = {
                        'type': 'invalid', 
                        'message': message,
                        'attempted_action': action['type'],
                        'validation_status': 'failed',
                        'step': self.step_count
                    }
        else:
            # No change - count as action attempt but not successful edit
            if self.step_count < self.min_episode_length:
                print(f"🔄 NO-CHANGE action at step {self.step_count}, giving small penalty")
                reward = -0.1
                edit_info = {
                    'type': 'no_change', 
                    'validation_status': 'no_edit',
                    'step': self.step_count
                }
            else:
                reward = -0.2
                edit_info = {
                    'type': 'no_change', 
                    'validation_status': 'no_edit',
                    'step': self.step_count
                }

        # 🚀 CRITICAL FIX: Update step count AFTER processing
        self.step_count += 1

        # 🚀 ENHANCED: Check max_steps termination with better logging
        if self.step_count >= self.max_steps:
            self.done = True
            successful_edits = len(self.edit_history)
            print(f"⏰ MAX STEPS REACHED: {self.step_count} action attempts, {successful_edits} successful edits")
            
            # LOG EPISODE COMPLETION FOR MAX STEPS TERMINATION
            self._log_episode_completion("max_steps", reward)

        info = {
            'action': action,
            'edit_info': edit_info,
            'edit_successful': edit_successful,
            'sequence_length': len(self.current_sequence),
            'sequence_changed': old_sequence != self.current_sequence,
            'edit_count': len(self.edit_history),  # Successful edits
            'step_count': self.step_count,  # Action attempts
            'action_attempts': self.step_count,  # Explicit for clarity
            'successful_edits': len(self.edit_history),  # Explicit for clarity
            'min_length_reached': self.step_count >= self.min_episode_length
        }

        return self.get_state(), reward, self.done, info

    def _log_episode_completion(self, termination_reason, final_reward):
        """Enhanced episode completion logging with action vs edit distinction"""
        
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
        
        successful_edits = len(self.edit_history)
        action_attempts = self.step_count
        
        # LOG EPISODE COMPLETION with enhanced action/edit info
        logger.info(f"🏁 EPISODE {self.episode_number} COMPLETE 🏁")
        logger.info(f"   🔢 ACTION SUMMARY:")
        logger.info(f"      Action Attempts: {action_attempts}")
        logger.info(f"      Successful Edits: {successful_edits}")
        logger.info(f"      Edit Success Rate: {successful_edits/max(1,action_attempts)*100:.1f}%")
        logger.info(f"   📊 TOUGHNESS CHANGE:")
        logger.info(f"      Original: {original_toughness:.6f}")
        logger.info(f"      Final:    {final_toughness:.6f}")
        logger.info(f"      Total Δ:  {cumulative_improvement:+.6f} ({cumulative_improvement*100:+.3f}%)")
        logger.info(f"      Last Δ:   {last_improvement:+.6f}")
        logger.info(f"   🎯 PERFORMANCE:")
        logger.info(f"      Final Reward: {final_reward:.3f}")
        logger.info(f"      Success:      {'✅ YES' if cumulative_improvement > 0.001 else '❌ NO'}")
        logger.info(f"   🛑 TERMINATION: {termination_reason}")
        
        # Special logging for different termination types
        if termination_reason == "max_steps":
            logger.info(f"   ⏰ Reached environment maximum action attempts ({self.max_steps})")
            logger.info(f"   💡 Consider increasing max_steps if edit success rate is high")
        elif termination_reason == "stop_action":
            logger.info(f"   🛑 Agent chose to stop after {successful_edits} successful edits")
        
        logger.info(f"   " + "="*60)

    # Keep other methods unchanged
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
        """Enhanced episode summary with action vs edit distinction"""
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
            'total_edits': len(self.edit_history),  # Successful edits
            'action_attempts': self.step_count,  # Total action attempts
            'edit_success_rate': len(self.edit_history) / max(1, self.step_count),
            'total_improvement': total_improvement,
            'edit_type_counts': edit_type_counts,
            'final_sequence_length': len(self.current_sequence),
            'original_sequence_length': len(self.original_sequence),
            'steps_taken': self.step_count,
            'done': self.done
        }