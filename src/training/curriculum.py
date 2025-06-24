"""
Curriculum Learning for Spider Silk Protein Optimization

This module implements various curriculum learning strategies for training
RL agents on protein sequence editing tasks.
"""

import random
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class CurriculumStrategy(ABC):
    """Abstract base class for curriculum learning strategies"""
    
    @abstractmethod
    def get_available_levels(self, episode: int, max_episodes: int, n_levels: int) -> List[int]:
        """Get available difficulty levels for current episode"""
        pass
    
    @abstractmethod
    def get_level_weights(self, available_levels: List[int]) -> List[float]:
        """Get sampling weights for available levels"""
        pass


class LinearCurriculumStrategy(CurriculumStrategy):
    """Linear progression from easy to hard"""
    
    def get_available_levels(self, episode: int, max_episodes: int, n_levels: int) -> List[int]:
        progress = min(episode / max_episodes, 1.0)
        max_difficulty = int(progress * (n_levels - 1))
        return list(range(max_difficulty + 1))
    
    def get_level_weights(self, available_levels: List[int]) -> List[float]:
        # Equal weights for all available levels
        return [1.0] * len(available_levels)


class ExponentialCurriculumStrategy(CurriculumStrategy):
    """Exponential progression - stays on easy levels longer"""
    
    def __init__(self, exponent: float = 2.0):
        self.exponent = exponent
    
    def get_available_levels(self, episode: int, max_episodes: int, n_levels: int) -> List[int]:
        progress = min(episode / max_episodes, 1.0)
        max_difficulty = int((progress ** self.exponent) * (n_levels - 1))
        return list(range(max_difficulty + 1))
    
    def get_level_weights(self, available_levels: List[int]) -> List[float]:
        # Exponential bias toward easier levels
        weights = [2.0 ** (len(available_levels) - i - 1) for i in range(len(available_levels))]
        return weights


class MixedCurriculumStrategy(CurriculumStrategy):
    """Mixed strategy with staged introduction"""
    
    def __init__(self, thresholds: Optional[List[float]] = None):
        self.thresholds = thresholds or [0.3, 0.6, 0.8]
    
    def get_available_levels(self, episode: int, max_episodes: int, n_levels: int) -> List[int]:
        progress = min(episode / max_episodes, 1.0)
        
        if progress < self.thresholds[0]:
            return [0, 1] if n_levels > 1 else [0]
        elif progress < self.thresholds[1]:
            return [0, 1, 2] if n_levels > 2 else list(range(min(n_levels, 3)))
        elif progress < self.thresholds[2]:
            return list(range(min(n_levels, 4)))
        else:
            return list(range(n_levels))
    
    def get_level_weights(self, available_levels: List[int]) -> List[float]:
        # Bias toward easier levels with gradual shift
        if len(available_levels) == 1:
            return [1.0]
        
        weights = [2.0 ** (len(available_levels) - i - 1) for i in available_levels]
        return weights


class NoCurriculumStrategy(CurriculumStrategy):
    """No curriculum - uniform sampling from all levels"""
    
    def get_available_levels(self, episode: int, max_episodes: int, n_levels: int) -> List[int]:
        return list(range(n_levels))
    
    def get_level_weights(self, available_levels: List[int]) -> List[float]:
        return [1.0] * len(available_levels)


class CurriculumManager:
    """
    Manages curriculum learning for protein sequence datasets
    
    Handles difficulty stratification and sequence sampling based on
    curriculum learning strategies.
    """
    
    def __init__(self, 
                 sequences: List[str],
                 toughness_values: List[float],
                 n_difficulty_levels: int = 5,
                 strategy: str = "mixed"):
        """
        Initialize curriculum manager
        
        Args:
            sequences: List of protein sequences
            toughness_values: Corresponding toughness values
            n_difficulty_levels: Number of difficulty levels to create
            strategy: Curriculum strategy ('linear', 'exponential', 'mixed', 'none')
        """
        self.sequences = sequences
        self.toughness_values = toughness_values
        self.n_difficulty_levels = n_difficulty_levels
        
        # Create difficulty stratification
        self._create_difficulty_stratification()
        
        # Initialize strategy
        self.strategy = self._create_strategy(strategy)
        
        # Statistics tracking
        self.level_sampling_counts = defaultdict(int)
        self.episode_count = 0
        
        logger.info(f"Curriculum manager initialized with {len(sequences)} sequences")
        logger.info(f"Strategy: {strategy}, Difficulty levels: {n_difficulty_levels}")
        self._log_difficulty_distribution()
    
    def _create_strategy(self, strategy_name: str) -> CurriculumStrategy:
        """Create curriculum strategy instance"""
        strategies = {
            'linear': LinearCurriculumStrategy(),
            'exponential': ExponentialCurriculumStrategy(),
            'mixed': MixedCurriculumStrategy(),
            'none': NoCurriculumStrategy(),
            'all': NoCurriculumStrategy(),  # Alias
        }
        
        if strategy_name not in strategies:
            logger.warning(f"Unknown strategy '{strategy_name}', using 'mixed'")
            strategy_name = 'mixed'
        
        return strategies[strategy_name]
    
    def _create_difficulty_stratification(self):
        """Create difficulty levels based on toughness quantiles"""
        # Lower toughness = easier to improve = easier level
        toughness_array = np.array(self.toughness_values)
        
        # Create quantile-based difficulty levels
        quantiles = np.linspace(0, 1, self.n_difficulty_levels + 1)
        self.difficulty_thresholds = np.quantile(toughness_array, quantiles)
        
        # Assign difficulty level to each sequence
        self.difficulty_levels = []
        for tough in self.toughness_values:
            level = np.digitize(tough, self.difficulty_thresholds) - 1
            level = max(0, min(level, self.n_difficulty_levels - 1))
            self.difficulty_levels.append(level)
        
        # Group sequences by difficulty
        self.sequences_by_difficulty = defaultdict(list)
        self.indices_by_difficulty = defaultdict(list)
        
        for i, (seq, level) in enumerate(zip(self.sequences, self.difficulty_levels)):
            self.sequences_by_difficulty[level].append(seq)
            self.indices_by_difficulty[level].append(i)
    
    def _log_difficulty_distribution(self):
        """Log the distribution of sequences across difficulty levels"""
        logger.info("Difficulty level distribution:")
        for level in range(self.n_difficulty_levels):
            count = len(self.sequences_by_difficulty[level])
            if count > 0:
                avg_tough = np.mean([self.toughness_values[i] 
                                   for i in self.indices_by_difficulty[level]])
                logger.info(f"  Level {level}: {count} sequences, avg_toughness={avg_tough:.3f}")
    
    def sample_sequence(self, episode: int, max_episodes: int) -> Tuple[str, int]:
        """
        Sample a sequence based on curriculum strategy
        
        Args:
            episode: Current episode number
            max_episodes: Total number of episodes planned
            
        Returns:
            Tuple of (sequence, difficulty_level)
        """
        self.episode_count = episode
        
        # Get available levels from strategy
        available_levels = self.strategy.get_available_levels(
            episode, max_episodes, self.n_difficulty_levels
        )
        
        # Get sampling weights
        weights = self.strategy.get_level_weights(available_levels)
        
        # Sample difficulty level
        chosen_level = random.choices(available_levels, weights=weights)[0]
        
        # Get sequence from chosen level
        level_sequences = self.sequences_by_difficulty[chosen_level]
        
        if not level_sequences:
            # Fallback to easiest level if chosen level is empty
            chosen_level = 0
            level_sequences = self.sequences_by_difficulty[chosen_level]
            
            if not level_sequences:
                # Ultimate fallback - random sequence
                chosen_level = random.choice(list(self.sequences_by_difficulty.keys()))
                level_sequences = self.sequences_by_difficulty[chosen_level]
        
        sequence = random.choice(level_sequences)
        
        # Update statistics
        self.level_sampling_counts[chosen_level] += 1
        
        return sequence, chosen_level
    
    def get_curriculum_progress(self, episode: int, max_episodes: int) -> Dict:
        """Get current curriculum progress information"""
        available_levels = self.strategy.get_available_levels(
            episode, max_episodes, self.n_difficulty_levels
        )
        
        progress = min(episode / max_episodes, 1.0)
        
        return {
            'episode': episode,
            'progress': progress,
            'available_levels': available_levels,
            'max_available_level': max(available_levels) if available_levels else 0,
            'level_sampling_counts': dict(self.level_sampling_counts),
            'total_samples': sum(self.level_sampling_counts.values())
        }
    
    def get_level_statistics(self) -> Dict:
        """Get detailed statistics about difficulty levels"""
        stats = {}
        
        for level in range(self.n_difficulty_levels):
            indices = self.indices_by_difficulty[level]
            if indices:
                toughness_vals = [self.toughness_values[i] for i in indices]
                stats[level] = {
                    'count': len(indices),
                    'avg_toughness': np.mean(toughness_vals),
                    'std_toughness': np.std(toughness_vals),
                    'min_toughness': np.min(toughness_vals),
                    'max_toughness': np.max(toughness_vals),
                    'sampling_count': self.level_sampling_counts[level]
                }
            else:
                stats[level] = {
                    'count': 0,
                    'sampling_count': 0
                }
        
        return stats
    
    def reset_sampling_counts(self):
        """Reset sampling statistics"""
        self.level_sampling_counts.clear()
        self.episode_count = 0
    
    def get_sequences_by_level(self, level: int) -> List[str]:
        """Get all sequences for a specific difficulty level"""
        return self.sequences_by_difficulty[level].copy()
    
    def get_difficulty_level(self, sequence: str) -> Optional[int]:
        """Get difficulty level for a specific sequence"""
        try:
            seq_idx = self.sequences.index(sequence)
            return self.difficulty_levels[seq_idx]
        except ValueError:
            return None


class AdaptiveCurriculumManager(CurriculumManager):
    """
    Adaptive curriculum manager that adjusts based on performance
    
    Modifies curriculum progression based on agent performance,
    slowing down or speeding up difficulty progression as needed.
    """
    
    def __init__(self, *args, performance_window: int = 100, 
                 success_threshold: float = 0.6, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.performance_window = performance_window
        self.success_threshold = success_threshold
        self.performance_history = []
        self.adaptation_factor = 1.0
        
        logger.info(f"Adaptive curriculum initialized with window={performance_window}")
    
    def update_performance(self, success: bool):
        """Update performance tracking"""
        self.performance_history.append(float(success))
        
        # Keep only recent history
        if len(self.performance_history) > self.performance_window:
            self.performance_history.pop(0)
        
        # Update adaptation factor
        if len(self.performance_history) >= self.performance_window:
            recent_success_rate = np.mean(self.performance_history)
            
            if recent_success_rate > self.success_threshold:
                # Performing well - can progress faster
                self.adaptation_factor = min(2.0, self.adaptation_factor * 1.05)
            else:
                # Struggling - slow down progression
                self.adaptation_factor = max(0.5, self.adaptation_factor * 0.95)
    
    def sample_sequence(self, episode: int, max_episodes: int) -> Tuple[str, int]:
        """Sample sequence with adaptive curriculum"""
        # Adjust effective episode based on performance
        effective_episode = int(episode * self.adaptation_factor)
        effective_episode = min(effective_episode, max_episodes)
        
        return super().sample_sequence(effective_episode, max_episodes)
    
    def get_curriculum_progress(self, episode: int, max_episodes: int) -> Dict:
        """Get progress info including adaptation"""
        progress = super().get_curriculum_progress(episode, max_episodes)
        
        progress.update({
            'adaptation_factor': self.adaptation_factor,
            'recent_success_rate': np.mean(self.performance_history) if self.performance_history else 0.0,
            'performance_samples': len(self.performance_history)
        })
        
        return progress