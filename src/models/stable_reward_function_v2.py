# src/models/stable_reward_function_v2.py
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
    Improved reward function with better consistency and reduced variance
    Inherits all the base functionality but overrides calculate_reward
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Additional stability parameters
        self.reward_smoothing = 0.9  # For moving averages
        self.recent_rewards = []
        self.reward_history_size = 20

    def calculate_reward(self,
                 old_seq,
                 new_seq,
                 edit_history,
                 original_seq,
                 episode_number,
                 target_improvement=0.0005):
        """
        Stable reward calculation with better consistency and reduced variance
        """
        
        try:
            # Get actual toughness improvement (most important signal)
            old_tough, _ = self.predict_toughness(old_seq)
            new_tough, _ = self.predict_toughness(new_seq)
            actual_improvement = new_tough - old_tough
            
            # Validate improvement
            actual_improvement = self._validate_number(actual_improvement, "actual_improvement", 0.0)
            
            edit_count = len(edit_history) if edit_history else 0
            
            # SIMPLIFIED REWARD STRUCTURE (much less variance)
            
            # 1. Primary reward: Actual toughness improvement (70% of reward)
            if actual_improvement > 0.002:  # Significant improvement
                toughness_reward = min(2.5, actual_improvement * 400)  # Scale and cap
            elif actual_improvement > 0.0005:  # Moderate improvement  
                toughness_reward = min(1.5, actual_improvement * 300)
            elif actual_improvement > 0:     # Small improvement
                toughness_reward = actual_improvement * 200
            else:                           # No improvement or negative
                toughness_reward = max(-1.0, actual_improvement * 50)  # Smaller penalty
            
            # 2. Success bonus (if target reached)
            if actual_improvement >= target_improvement:
                success_bonus = 1.5  # Moderate bonus, not game-breaking
                done = True
                logger.info(f"🎉 SUCCESS! Actual improvement: {actual_improvement:.4f}")
            else:
                success_bonus = 0.0
                done = False
            
            # 3. Small consistency bonuses (20% of reward)
            
            # Edit efficiency (encourage fewer, better edits)
            if edit_count == 0:
                efficiency_reward = -0.2  # Small penalty for no exploration
            elif edit_count <= 3:
                efficiency_reward = 0.1   # Bonus for very concise editing
            elif edit_count <= 8:
                efficiency_reward = 0.05  # Small bonus for reasonable editing
            else:
                efficiency_reward = max(-0.3, -0.02 * (edit_count - 8))  # Linear penalty
            
            # 4. Realism check (prevent completely unrealistic sequences) (10% of reward)
            try:
                perplexity = self.calculate_perplexity(new_seq)
                if perplexity > 15.0:  # Very high perplexity
                    realism_penalty = -0.4
                elif perplexity > 8.0:  # High perplexity
                    realism_penalty = -0.2
                elif perplexity > 5.0:  # Moderate perplexity
                    realism_penalty = -0.1
                else:
                    realism_penalty = 0.0
            except:
                realism_penalty = 0.0
            
            # 5. Exploration bonus (encourage trying different approaches)
            exploration_bonus = 0.0
            if edit_count > 0:
                # Small bonus for making edits
                exploration_bonus = min(0.1, edit_count * 0.02)
            
            # Final reward calculation (much more stable)
            total_reward = (toughness_reward + success_bonus + 
                          efficiency_reward + realism_penalty + exploration_bonus)
            
            # Apply smoothing to reduce variance
            total_reward = self._smooth_reward(total_reward)
            
            # Safety clipping (tighter bounds for consistency)
            total_reward = float(np.clip(total_reward, -1.5, 4.0))
            
            # Validation
            if np.isnan(total_reward) or np.isinf(total_reward):
                total_reward = -0.1
            
            # Debug logging every 50 episodes
            if episode_number % 50 == 0:
                logger.info(f"Episode {episode_number}: improvement={actual_improvement:.4f}, "
                          f"reward={total_reward:.3f}, edits={edit_count}")
            
            return {
                'total': total_reward,
                'done': done,
                'components': {
                    'toughness': toughness_reward,
                    'success_bonus': success_bonus,
                    'efficiency': efficiency_reward,
                    'realism': realism_penalty,
                    'exploration': exploration_bonus
                },
                'actual_improvement': actual_improvement,
                'edit_count': edit_count,
                'perplexity': perplexity if 'perplexity' in locals() else 0.0
            }

        except Exception as e:
            logger.error(f"Error in stable reward calculation: {e}")
            return {
                'total': -0.1,
                'done': False,
                'components': {},
                'actual_improvement': 0.0
            }
    
    def _smooth_reward(self, reward: float) -> float:
        """Apply smoothing to reduce reward variance"""
        self.recent_rewards.append(reward)
        
        # Keep only recent rewards
        if len(self.recent_rewards) > self.reward_history_size:
            self.recent_rewards.pop(0)
        
        # If we have enough history, apply smoothing
        if len(self.recent_rewards) >= 5:
            # Weighted average: 70% current reward, 30% recent average
            recent_avg = np.mean(self.recent_rewards[:-1])  # Exclude current reward
            smoothed = self.reward_smoothing * reward + (1 - self.reward_smoothing) * recent_avg
            return smoothed
        else:
            return reward