# src/data/dataset.py
import pandas as pd
import numpy as np
import random
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from collections import defaultdict

class SpiderSilkDataset:
    def __init__(self, csv_path: str, test_size: float = 0.2, 
                 n_difficulty_levels: int = 5, random_state: int = 42):
        self.df = pd.read_csv(csv_path)
        self.test_size = test_size
        self.n_difficulty_levels = n_difficulty_levels
        
        # Assume columns: 'sequence', 'toughness'
        self.sequences = self.df['sequence'].tolist()
        self.toughness_values = self.df['toughness'].tolist()
        
        self._create_difficulty_stratification()
        self._stratified_train_test_split(random_state)
        
    def _create_difficulty_stratification(self):
        """Create difficulty levels based on toughness quantiles"""
        toughness_array = np.array(self.toughness_values)
        quantiles = np.linspace(0, 1, self.n_difficulty_levels + 1)
        self.difficulty_thresholds = np.quantile(toughness_array, quantiles)
        
        # Assign difficulty levels (lower toughness = easier to improve)
        self.difficulty_levels = []
        for tough in self.toughness_values:
            level = np.digitize(tough, self.difficulty_thresholds) - 1
            level = max(0, min(level, self.n_difficulty_levels - 1))
            self.difficulty_levels.append(level)
        
        # Group by difficulty
        self.sequences_by_difficulty = defaultdict(list)
        for seq, level in zip(self.sequences, self.difficulty_levels):
            self.sequences_by_difficulty[level].append(seq)
    
    def _stratified_train_test_split(self, random_state: int):
        """Split preserving difficulty distribution"""
        self.train_sequences = []
        self.test_sequences = []
        self.train_difficulty_levels = []
        self.test_difficulty_levels = []
        
        for level in range(self.n_difficulty_levels):
            level_sequences = self.sequences_by_difficulty[level]
            if len(level_sequences) > 0:
                train_seqs, test_seqs = train_test_split(
                    level_sequences, test_size=self.test_size, 
                    random_state=random_state + level
                )
                
                self.train_sequences.extend(train_seqs)
                self.test_sequences.extend(test_seqs)
                self.train_difficulty_levels.extend([level] * len(train_seqs))
                self.test_difficulty_levels.extend([level] * len(test_seqs))
    
    def get_curriculum_sequence(self, episode: int, max_episodes: int, 
                               strategy: str = 'mixed') -> Tuple[str, int]:
        """Get sequence based on curriculum strategy"""
        progress = min(episode / max_episodes, 1.0)
        
        if strategy == 'linear':
            max_difficulty = int(progress * (self.n_difficulty_levels - 1))
            available_levels = list(range(max_difficulty + 1))
        elif strategy == 'exponential':
            max_difficulty = int((progress ** 2) * (self.n_difficulty_levels - 1))
            available_levels = list(range(max_difficulty + 1))
        elif strategy == 'mixed':
            if progress < 0.3:
                available_levels = [0, 1]
            elif progress < 0.6:
                available_levels = [0, 1, 2]
            elif progress < 0.8:
                available_levels = [0, 1, 2, 3]
            else:
                available_levels = list(range(self.n_difficulty_levels))
        else:  # 'all'
            available_levels = list(range(self.n_difficulty_levels))
        
        # Choose level with bias toward easier ones
        if len(available_levels) == 1:
            chosen_level = available_levels[0]
        else:
            weights = [2.0 ** (len(available_levels) - i - 1) for i in available_levels]
            chosen_level = random.choices(available_levels, weights=weights)[0]
        
        # Get random sequence from chosen level
        level_sequences = [seq for seq, level in zip(self.train_sequences, self.train_difficulty_levels) 
                          if level == chosen_level]
        
        if not level_sequences:
            level_sequences = [seq for seq, level in zip(self.train_sequences, self.train_difficulty_levels) 
                              if level == 0]
        
        return random.choice(level_sequences), chosen_level
    
    def get_test_sequences(self, n: int = None) -> List[str]:
        """Get test sequences"""
        if n is None:
            return self.test_sequences
        return random.sample(self.test_sequences, min(n, len(self.test_sequences)))