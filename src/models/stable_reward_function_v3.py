import math
import torch
import numpy as np
import re
import logging
from typing import Tuple, Dict, Any, List

# Import the base class from your existing reward function
from src.models.reward_function import SpiderSilkRewardFunction

logger = logging.getLogger(__name__)

class StableSpiderSilkRewardFunctionV3(SpiderSilkRewardFunction):
    """
    Improved reward function with better consistency, reduced variance, and no motif requirements
    Focuses on toughness improvement while allowing longer episodes for better learning
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Stability parameters
        self.reward_smoothing = 0.9
        self.recent_rewards = []
        self.reward_history_size = 20
        
        # Episode management parameters - FIXED to allow longer episodes
        self.min_episode_length = 8  # Increased from 3 to 8
        self.max_episode_length = 25  # Clear maximum
        self.consecutive_bad_steps_threshold = 8  # Allow more bad steps before termination
        self.target_improvement_per_episode = 0.005

    def calculate_reward(self, old_seq, new_seq, edit_history, original_seq, episode_number):
        """Simplified reward aligned with actual improvement with FIXED termination logic"""
        
        # Get actual step improvement (ONLY source of positive reward)
        old_tough, _ = self.predict_toughness(old_seq)
        new_tough, _ = self.predict_toughness(new_seq)
        actual_improvement = new_tough - old_tough
        
        # ALIGNED REWARD: Positive only if actual improvement
        if actual_improvement > 0.003:
            reward = min(3.0, actual_improvement * 800)  # Excellent improvement
        elif actual_improvement > 0.001:
            reward = min(2.0, actual_improvement * 600)  # Good improvement  
        elif actual_improvement > 0:
            reward = actual_improvement * 400  # Any improvement
        else:
            reward = max(-1.0, actual_improvement * 200)  # Penalty for degradation
        
        # FIXED TERMINATION LOGIC - Much less aggressive
        edit_count = len(edit_history)
        done = False
        
        # Calculate cumulative improvement for logging and termination
        cumulative_improvement = 0.0
        original_toughness = 0.0
        final_toughness = new_tough
        
        if original_seq and original_seq != new_seq:
            try:
                original_toughness, _ = self.predict_toughness(original_seq)
                cumulative_improvement = new_tough - original_toughness
            except Exception as e:
                logger.warning(f"Failed to calculate cumulative improvement: {e}")
                cumulative_improvement = actual_improvement
        else:
            cumulative_improvement = actual_improvement
            original_toughness = old_tough
        
        # Only consider termination after minimum episode length
        if edit_count >= self.min_episode_length:
            
            # Check for consecutive bad steps (no improvement)
            # BUT only if we have enough steps to actually check for consecutive failures
            recent_improvements = []
            for h in edit_history[-self.consecutive_bad_steps_threshold:]:
                # Get improvement from edit history - check different possible keys
                step_improvement = h.get('toughness_improvement', 
                                        h.get('improvement', 
                                             h.get('actual_improvement', 0)))
                recent_improvements.append(step_improvement)
            
            # Terminate only if:
            # 1. We've had many consecutive steps with no improvement, AND we have enough history, OR
            # 2. We've reached maximum episode length, OR  
            # 3. We've achieved exceptional cumulative improvement
            
            # FIXED: Only check consecutive no improvement if we have enough steps beyond minimum
            has_enough_history_for_consecutive_check = edit_count >= (self.min_episode_length + self.consecutive_bad_steps_threshold - 1)
            consecutive_no_improvement = (has_enough_history_for_consecutive_check and 
                                        len(recent_improvements) >= self.consecutive_bad_steps_threshold and 
                                        all(imp <= 0.0001 for imp in recent_improvements))
            
            reached_max_length = edit_count >= self.max_episode_length
            exceptional_performance = cumulative_improvement > 0.08  # Very high threshold
            
            # Termination decision
            if consecutive_no_improvement:
                done = True
                termination_reason = "no_improvement"
            elif reached_max_length:
                done = True
                termination_reason = "max_length"
            elif exceptional_performance:
                done = True
                termination_reason = "exceptional_performance"
        
        # ENHANCED EPISODE END LOGGING
        if done:
            # Comprehensive episode summary with toughness changes
            logger.info(f"🏁 EPISODE {episode_number} COMPLETE (Steps: {edit_count}) 🏁")
            logger.info(f"   📊 TOUGHNESS CHANGE:")
            logger.info(f"      Original: {original_toughness:.6f}")
            logger.info(f"      Final:    {final_toughness:.6f}")
            logger.info(f"      Total Δ:  {cumulative_improvement:+.6f} ({cumulative_improvement*100:+.3f}%)")
            logger.info(f"      Last Δ:   {actual_improvement:+.6f}")
            logger.info(f"   🎯 PERFORMANCE:")
            logger.info(f"      Final Reward: {reward:.3f}")
            logger.info(f"      Edits Made:   {edit_count}")
            logger.info(f"      Success:      {'✅ YES' if cumulative_improvement > 0.001 else '❌ NO'}")
            logger.info(f"   🛑 TERMINATION: {termination_reason}")
            
            # Special logging for exceptional performance
            if termination_reason == "exceptional_performance":
                logger.info(f"   🎉 EXCEPTIONAL PERFORMANCE! Improvement: {cumulative_improvement:.6f}")
            elif termination_reason == "no_improvement":
                logger.info(f"   ⏹️  Stopped due to {len(recent_improvements)} consecutive steps with no improvement")
            elif termination_reason == "max_length":
                logger.info(f"   ⏱️  Reached maximum episode length ({self.max_episode_length} steps)")
            
            logger.info(f"   " + "="*60)
        
        # Additional logging for debugging episode length
        elif edit_count <= 5:
            logger.debug(f"Episode {episode_number}, Step {edit_count}: improvement={actual_improvement:.6f}, reward={reward:.3f}, cumulative={cumulative_improvement:.6f}")
        
        return {
            'total': float(reward),
            'done': done,
            'actual_improvement': actual_improvement,
            'cumulative_improvement': cumulative_improvement,
            'original_toughness': original_toughness,
            'final_toughness': final_toughness,
            'edit_count': edit_count,
            'termination_reason': self._get_termination_reason(done, edit_count, exceptional_performance if 'exceptional_performance' in locals() else False)
        }
    
    def _get_termination_reason(self, done, edit_count, exceptional_performance):
        """Helper to track why episodes terminate"""
        if not done:
            return "continuing"
        elif exceptional_performance:
            return "exceptional_performance"
        elif edit_count >= self.max_episode_length:
            return "max_length"
        elif edit_count >= self.min_episode_length:
            return "no_improvement"
        else:
            return "unknown"
    
    def _has_reasonable_composition(self, sequence: str) -> bool:
        """
        Check if sequence has reasonable amino acid composition (no motif requirements)
        More lenient than motif checking
        """
        if len(sequence) < 10:
            return False
        
        sequence_upper = sequence.upper()
        length = len(sequence_upper)
        
        # Check for reasonable diversity (not too repetitive)
        unique_aa = len(set(sequence_upper))
        if unique_aa < 5:  # At least 5 different amino acids
            return False
        
        # Check for reasonable composition of key amino acids (very lenient)
        alanine_ratio = sequence_upper.count('A') / length
        glycine_ratio = sequence_upper.count('G') / length
        
        # Very lenient bounds - just avoid extreme cases
        if alanine_ratio > 0.7 or glycine_ratio > 0.7:  # Not more than 70% of any single AA
            return False
        
        # Check for presence of some structure-forming amino acids
        structure_aa = sum(sequence_upper.count(aa) for aa in 'AGPYFWH')
        if structure_aa / length < 0.3:  # At least 30% structure-forming amino acids
            return False
        
        return True
    
    def _smooth_reward(self, reward: float) -> float:
        """Apply smoothing to reduce reward variance"""
        self.recent_rewards.append(reward)
        
        # Keep only recent rewards
        if len(self.recent_rewards) > self.reward_history_size:
            self.recent_rewards.pop(0)
        
        # If we have enough history, apply light smoothing
        if len(self.recent_rewards) >= 5:
            # Weighted average: 80% current reward, 20% recent average
            recent_avg = np.mean(self.recent_rewards[:-1])  # Exclude current reward
            smoothed = 0.8 * reward + 0.2 * recent_avg
            return smoothed
        else:
            return reward
    
    def _validate_number(self, value: float, name: str, default: float = 0.0) -> float:
        """Validate that a number is finite and reasonable"""
        if np.isnan(value) or np.isinf(value):
            logger.warning(f"Invalid {name}: {value}, using default {default}")
            return default
        return float(value)