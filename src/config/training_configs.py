# config/training_configs.py
from dataclasses import dataclass
from typing import Dict, Any
import time

@dataclass
class TrainingConfig:
    # Dataset
    dataset_path: str = 'src/data/raw/V5_MaSp.csv'
    test_size: float = 0.2
    n_difficulty_levels: int = 5
    
    # Training
    n_episodes: int = 5000
    max_steps: int = 50
    learning_rate: float = 3e-5
    curriculum_strategy: str = 'mixed'  # 'linear', 'exponential', 'mixed', 'all'
    
    # PPO hyperparameters
    clip_epsilon: float = 0.4
    value_coeff: float = 0.5
    entropy_coeff: float = 0.1
    
    # Logging
    project_name: str = 'spider-silk-rl'
    run_name: str = f'mixed-curriculum-{int(time.time())}'
    log_interval: int = 100
    test_interval: int = 200
    checkpoint_interval: int = 500
    test_sequences_per_eval: int = 20
    
    # Infrastructure
    save_dir: str = 'results/runs'
    seed: int = 42
    world_size: int = 8
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

def get_config(name: str = 'default') -> TrainingConfig:
    """Get predefined configurations"""
    configs = {
        'default': TrainingConfig(),
        'quick_test': TrainingConfig(
            n_episodes=500,
            max_steps=20,
            learning_rate=3e-4,
            entropy_coeff=0.3,
            world_size=2,
            log_interval=10,
            test_interval=20,
            run_name='quick-test'
        ),
        'phase1': TrainingConfig(
            n_episodes=500,           # Very short
            max_steps=20,             # Limited steps  
            learning_rate=3e-4,       # Conservative
            curriculum_strategy='all', # No curriculum needed
            entropy_coeff=0.3,
            world_size=1,             # Single GPU
            log_interval=10,
            test_interval=20,
            run_name='phase1-proof-of-concept'
        ),
        'baseline': TrainingConfig(
            curriculum_strategy='all',
            run_name='baseline-no-curriculum'
        ),
        'linear': TrainingConfig(
            curriculum_strategy='linear',
            run_name='linear-curriculum'
        )
    }
    return configs.get(name, configs['default'])
