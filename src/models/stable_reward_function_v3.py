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
        
        # Episode management parameters
        self.min_episode_length = 3  # Minimum steps before allowing termination
        self.max_cumulative_improvement = 0.02  # Higher threshold for early termination
        self.target_improvement_per_episode = 0.005  # More ambitious target

    def calculate_reward(self, old_seq, new_seq, edit_history, original_seq, episode_number):
        """Simplified reward aligned with actual improvement"""
        
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
        
        # Simple termination: stop if no improvement for several steps
        edit_count = len(edit_history)
        recent_improvements = [h.get('improvement', 0) for h in edit_history[-3:]]
        done = (edit_count >= 3 and all(imp <= 0 for imp in recent_improvements)) or edit_count >= 20
        
        return {
            'total': float(reward),
            'done': done,
            'actual_improvement': actual_improvement
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