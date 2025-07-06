import math
import torch
import numpy as np
import re
import logging
from typing import Tuple, Dict, Any, List

# Import the base class from your existing reward function
from src.models.reward_function import SpiderSilkRewardFunction

logger = logging.getLogger(__name__)

class StableSpiderSilkRewardFunctionV2(SpiderSilkRewardFunction):
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
        
        # Episode management parameters
        self.min_episode_length = 3  # Minimum steps before allowing termination
        self.max_cumulative_improvement = 0.02  # Higher threshold for early termination
        self.target_improvement_per_episode = 0.005  # More ambitious target

    def calculate_reward(self,
                        old_seq,
                        new_seq,
                        edit_history,
                        original_seq,
                        episode_number,
                        target_improvement=None):
        """
        Stable reward calculation with better episode management and no motif requirements
        """
        
        try:
            # Use instance target if not provided
            if target_improvement is None:
                target_improvement = self.target_improvement_per_episode
            
            # Get actual toughness improvement (primary signal)
            old_tough, _ = self.predict_toughness(old_seq)
            new_tough, _ = self.predict_toughness(new_seq)
            actual_improvement = new_tough - old_tough
            
            # Validate improvement
            actual_improvement = self._validate_number(actual_improvement, "actual_improvement", 0.0)
            
            edit_count = len(edit_history) if edit_history else 0
            
            # Calculate cumulative improvement from original sequence
            if original_seq and original_seq != new_seq:
                orig_tough, _ = self.predict_toughness(original_seq)
                cumulative_improvement = new_tough - orig_tough
            else:
                cumulative_improvement = actual_improvement
            
            cumulative_improvement = self._validate_number(cumulative_improvement, "cumulative_improvement", 0.0)
            
            # REWARD CALCULATION (simplified and more stable)
            
            # 1. Primary reward: Actual step improvement (60% of total reward)
            if actual_improvement > 0.003:  # Excellent improvement
                toughness_reward = min(2.0, actual_improvement * 500)
            elif actual_improvement > 0.001:  # Good improvement
                toughness_reward = min(1.2, actual_improvement * 400)
            elif actual_improvement > 0.0002:  # Small improvement
                toughness_reward = actual_improvement * 300
            elif actual_improvement > 0:  # Tiny improvement
                toughness_reward = actual_improvement * 200
            else:  # No improvement or degradation
                toughness_reward = max(-0.8, actual_improvement * 100)  # Gentler penalty
            
            # 2. Cumulative progress reward (20% of total reward)
            if cumulative_improvement > 0.01:  # Substantial total progress
                cumulative_reward = min(1.0, cumulative_improvement * 50)
            elif cumulative_improvement > 0.005:  # Good total progress
                cumulative_reward = min(0.6, cumulative_improvement * 80)
            elif cumulative_improvement > 0.001:  # Some progress
                cumulative_reward = cumulative_improvement * 100
            else:  # No cumulative progress
                cumulative_reward = max(-0.3, cumulative_improvement * 30)
            
            # 3. Exploration and efficiency (15% of total reward)
            if edit_count == 0:
                exploration_reward = -0.3  # Penalty for no action
            elif edit_count <= 5:
                exploration_reward = 0.1  # Bonus for efficient exploration
            elif edit_count <= 10:
                exploration_reward = 0.05  # Small bonus
            else:
                exploration_reward = max(-0.2, -0.01 * (edit_count - 10))  # Gentle penalty for excessive edits
            
            # 4. Sequence quality check (5% of total reward)
            quality_reward = 0.0
            try:
                # Basic length check (no motif requirements)
                if len(new_seq) < 20:
                    quality_reward = -0.3  # Too short
                elif len(new_seq) > 2000:
                    quality_reward = -0.2  # Too long
                
                # Basic amino acid composition check
                if self._has_reasonable_composition(new_seq):
                    quality_reward += 0.1
                else:
                    quality_reward -= 0.1
                
                # Perplexity check (optional)
                perplexity = self.calculate_perplexity(new_seq)
                if perplexity > 20.0:
                    quality_reward -= 0.2
                elif perplexity > 10.0:
                    quality_reward -= 0.1
                    
            except Exception as e:
                logger.debug(f"Quality check failed: {e}")
                perplexity = 0.0
            
            # TERMINATION LOGIC (key change - much less aggressive)
            done = False
            termination_bonus = 0.0
            
            # Only consider termination after minimum episode length
            if edit_count >= self.min_episode_length:
                
                # Exceptional performance - allow early termination
                if (cumulative_improvement > self.max_cumulative_improvement and 
                    actual_improvement > 0.002):
                    done = True
                    termination_bonus = 2.0  # Big bonus for exceptional performance
                    logger.info(f"🎉 EXCEPTIONAL SUCCESS! Cumulative: {cumulative_improvement:.4f}, Step: {actual_improvement:.4f}")
                
                # Good performance but continue episode to learn more
                elif cumulative_improvement > target_improvement:
                    # Don't terminate, but give progress bonus
                    termination_bonus = 0.5
                    logger.debug(f"Good progress: {cumulative_improvement:.4f}, continuing episode...")
                
                # Long episode with some progress - natural termination
                elif edit_count >= 15 and cumulative_improvement > 0.001:
                    done = True
                    termination_bonus = 0.3
                    logger.debug(f"Natural termination after {edit_count} edits")
                
                # Very long episode - force termination
                elif edit_count >= 25:
                    done = True
                    termination_bonus = 0.0
                    logger.debug(f"Forced termination after {edit_count} edits")
            
            # Calculate final reward
            total_reward = (toughness_reward + cumulative_reward + 
                          exploration_reward + quality_reward + termination_bonus)
            
            # Apply smoothing to reduce variance
            total_reward = self._smooth_reward(total_reward)
            
            # Safety clipping
            total_reward = float(np.clip(total_reward, -2.0, 5.0))
            
            # Validation
            if np.isnan(total_reward) or np.isinf(total_reward):
                total_reward = -0.1
                logger.warning(f"Invalid reward detected, using fallback")
            
            # Detailed logging for debugging
            if episode_number % 50 == 0 or done:
                logger.info(f"Episode {episode_number}: step_improvement={actual_improvement:.4f}, "
                          f"cumulative={cumulative_improvement:.4f}, reward={total_reward:.3f}, "
                          f"edits={edit_count}, done={done}")
            
            return {
                'total': total_reward,
                'done': done,
                'components': {
                    'toughness': toughness_reward,
                    'cumulative': cumulative_reward,
                    'exploration': exploration_reward,
                    'quality': quality_reward,
                    'termination_bonus': termination_bonus
                },
                'actual_improvement': actual_improvement,
                'cumulative_improvement': cumulative_improvement,
                'edit_count': edit_count,
                'perplexity': perplexity if 'perplexity' in locals() else 0.0
            }

        except Exception as e:
            logger.error(f"Error in stable reward calculation: {e}")
            return {
                'total': -0.1,
                'done': False,
                'components': {},
                'actual_improvement': 0.0,
                'cumulative_improvement': 0.0
            }
    
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