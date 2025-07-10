# stable_reward_function_v3.py
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
        
        # Episode management parameters - ENHANCED for different improvement types
        self.min_episode_length = 8  # Minimum for any episode
        self.max_episode_length = 35  # Increased to allow for full exploration
        self.consecutive_bad_steps_threshold = 6  # For negative improvement
        self.zero_improvement_min_length = 20   # Must reach 20 steps before checking zero improvement
        self.zero_improvement_patience = 10     # Allow 10 consecutive zero steps after step 20
        self.target_improvement_per_episode = 0.005

    def calculate_reward(self, old_seq, new_seq, edit_history, original_seq, episode_number):
        """Simplified reward aligned with actual improvement with FIXED termination logic"""
        
        # FIXED: Define all variables FIRST
        edit_count = len(edit_history)
        done = False
        termination_reason = None
        
        # Get actual step improvement (ONLY source of positive reward)
        old_tough, _ = self.predict_toughness(old_seq)
        new_tough, _ = self.predict_toughness(new_seq)
        actual_improvement = new_tough - old_tough
        
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
        
        # ENHANCED REWARD: Actively punish zero improvement to encourage exploration
        if actual_improvement > 0.003:
            reward = min(3.0, actual_improvement * 800)  # Excellent improvement
        elif actual_improvement > 0.001:
            reward = min(2.0, actual_improvement * 600)  # Good improvement  
        elif actual_improvement > 0:
            reward = actual_improvement * 400  # Any positive improvement
        elif actual_improvement == 0.0:
            # ACTIVELY PUNISH zero improvement - escalating penalty based on episode progress
            if edit_count <= 5:
                reward = -0.1  # Early episode: small penalty
            elif edit_count <= 15:
                reward = -0.2  # Mid episode: medium penalty
            else:
                reward = -0.3  # Late episode: larger penalty for still no progress
            
            # ADDITIONAL PUNISHMENT: Check for repeated zero improvements
            recent_zeros = 0
            for h in edit_history[-5:]:  # Look at last 5 steps
                step_improvement = h.get('toughness_improvement', h.get('improvement', h.get('actual_improvement', 0)))
                if abs(step_improvement) < 0.0001:  # Essentially zero
                    recent_zeros += 1
            
            # Escalating penalty for consecutive zero improvements
            if recent_zeros >= 3:
                consecutive_penalty = -0.1 * recent_zeros  # -0.3 for 3 zeros, -0.5 for 5 zeros
                reward += consecutive_penalty
                
        else:
            reward = max(-1.0, actual_improvement * 200)  # Penalty for degradation
        
        # CUMULATIVE PROGRESS PUNISHMENT: Penalize inefficient episodes
        if edit_count >= 10:
            efficiency = cumulative_improvement / edit_count  # Improvement per step
            if efficiency < 0.0001:  # Very low efficiency
                inefficiency_penalty = -0.2
                reward += inefficiency_penalty
        
        # Only consider termination after minimum episode length
        if edit_count >= self.min_episode_length:
            
            # CORRECTED LOGIC: Different counting windows for different termination types
            
            # 1. NEGATIVE IMPROVEMENT: Check recent steps (can start early)
            recent_negative_improvements = []
            for h in edit_history[-self.consecutive_bad_steps_threshold:]:
                step_improvement = h.get('toughness_improvement', 
                                        h.get('improvement', 
                                             h.get('actual_improvement', 0)))
                recent_negative_improvements.append(step_improvement)
            
            consecutive_negative_improvement = (len(recent_negative_improvements) >= self.consecutive_bad_steps_threshold and
                                              all(imp < -0.0001 for imp in recent_negative_improvements))
            
            # 2. ZERO IMPROVEMENT: Only check AFTER step 30 (20 + 10)
            consecutive_zero_improvement = False
            min_steps_for_zero_check = self.zero_improvement_min_length + self.zero_improvement_patience  # 20 + 10 = 30
            
            if edit_count >= min_steps_for_zero_check:
                # Look at steps 21-30 (the last 10 steps after the first 20)
                recent_zero_check_steps = edit_history[-self.zero_improvement_patience:]
                
                # Double-check we have exactly the patience window
                if len(recent_zero_check_steps) == self.zero_improvement_patience:
                    recent_zero_improvements = []
                    for h in recent_zero_check_steps:
                        step_improvement = h.get('toughness_improvement', 
                                                h.get('improvement', 
                                                     h.get('actual_improvement', 0)))
                        recent_zero_improvements.append(step_improvement)
                    
                    # Check if ALL of the last 10 steps were zero improvement
                    consecutive_zero_improvement = all(-0.0001 <= imp <= 0.0001 for imp in recent_zero_improvements)
                    
                    # DEBUG LOGGING
                    if edit_count >= min_steps_for_zero_check and edit_count <= min_steps_for_zero_check + 2:
                        logger.debug(f"Episode {episode_number}, Step {edit_count}: Zero check - last {len(recent_zero_improvements)} steps: {[f'{imp:.6f}' for imp in recent_zero_improvements]}")
                        logger.debug(f"  All zero? {consecutive_zero_improvement}")
            
            # TERMINATION RULES:
            reached_max_length = edit_count >= self.max_episode_length
            exceptional_performance = cumulative_improvement > 0.08
            
            # Termination decision with CORRECTED logic
            if consecutive_negative_improvement:
                done = True
                termination_reason = "consecutive_negative"
            elif consecutive_zero_improvement:
                done = True  
                termination_reason = "zero_after_exploration"
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
            
            # Special logging for different termination types
            if termination_reason == "exceptional_performance":
                logger.info(f"   🎉 EXCEPTIONAL PERFORMANCE! Improvement: {cumulative_improvement:.6f}")
            elif termination_reason == "consecutive_negative":
                logger.info(f"   🚫 Terminated due to {self.consecutive_bad_steps_threshold} consecutive NEGATIVE improvements")
            elif termination_reason == "zero_after_exploration":
                logger.info(f"   ⚪ Terminated at step {edit_count}: {self.zero_improvement_patience} consecutive ZERO improvements after step {self.zero_improvement_min_length}")
                logger.info(f"   📈 Required minimum steps before zero-check: {self.zero_improvement_min_length + self.zero_improvement_patience}")
            elif termination_reason == "max_length":
                logger.info(f"   ⏱️  Reached maximum episode length ({self.max_episode_length} steps)")
            
            logger.info(f"   " + "="*60)
        
        # Additional logging for debugging episode length - Enhanced with punishment info
        elif edit_count <= 5 or edit_count % 5 == 0:
            punishment_info = ""
            if actual_improvement == 0.0:
                punishment_info = f" [ZERO PUNISH: {reward:.3f}]"
            elif actual_improvement < 0:
                punishment_info = f" [NEG PUNISH: {reward:.3f}]"
            
            logger.debug(f"Episode {episode_number}, Step {edit_count}: improvement={actual_improvement:.6f}, reward={reward:.3f}, cumulative={cumulative_improvement:.6f}{punishment_info}")
            
            # Special logging around the critical step 30
            if edit_count >= 25 and edit_count <= 32:
                min_steps_for_zero_check = self.zero_improvement_min_length + self.zero_improvement_patience
                logger.debug(f"  Zero termination check will activate at step {min_steps_for_zero_check}")
                
                # Log recent improvement pattern
                if len(edit_history) >= 5:
                    recent_improvements = [h.get('toughness_improvement', 0) for h in edit_history[-5:]]
                    zeros_count = sum(1 for imp in recent_improvements if abs(imp) < 0.0001)
                    logger.debug(f"  Recent pattern: {zeros_count}/5 zero improvements in last 5 steps")
        
        return {
            'total': float(reward),
            'done': done,
            'actual_improvement': actual_improvement,
            'cumulative_improvement': cumulative_improvement,
            'original_toughness': original_toughness,
            'final_toughness': final_toughness,
            'edit_count': edit_count,
            'termination_reason': self._get_termination_reason(done, edit_count, exceptional_performance if 'exceptional_performance' in locals() else False, termination_reason)
        }
    
    def _get_termination_reason(self, done, edit_count, exceptional_performance, termination_reason=None):
        """Helper to track why episodes terminate"""
        if not done:
            return "continuing"
        elif termination_reason:
            return termination_reason  # Use the detailed reason from calculate_reward
        elif exceptional_performance:
            return "exceptional_performance"
        elif edit_count >= self.max_episode_length:
            return "max_length"
        elif edit_count >= self.min_episode_length:
            return "no_improvement"  # Fallback
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