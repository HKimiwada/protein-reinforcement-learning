# src/config/stable_configs_v3.py
from dataclasses import dataclass
from typing import Dict, Any
import time

@dataclass
class StableTrainingConfigV3:
    """Stable training configuration with higher max_steps for better exploration"""
    
    # Dataset
    dataset_path: str = 'src/data/raw/V6_MaSp.csv'
    test_size: float = 0.2
    n_difficulty_levels: int = 5
    
    # Training - More stable hyperparameters with higher action limits
    n_episodes: int = 1500
    max_steps: int = 60  # INCREASED from 40 to 60 to handle action attempts vs successful edits
    learning_rate: float = 8e-5  # Conservative learning rate
    curriculum_strategy: str = 'mixed'  # Gradual difficulty increase
    
    # PPO - More conservative and stable
    clip_epsilon: float = 0.15   # Smaller clips for stability
    value_coeff: float = 0.8     # Balanced value learning
    entropy_coeff: float = 0.08  # Lower but consistent exploration
    max_grad_norm: float = 0.3   # Stricter gradient clipping
    
    # New stability parameters
    ppo_epochs: int = 3          # More updates per episode
    batch_size: int = 32         # Larger batches for stability
    use_gae: bool = True         # Generalized Advantage Estimation
    gae_lambda: float = 0.95     # GAE parameter
    gamma: float = 0.99          # Discount factor
    
    # Logging and evaluation
    project_name: str = 'spider-silk-rl-stable-v2'
    run_name: str = f'stable-consistent-v2-{int(time.time())}'
    log_interval: int = 25
    test_interval: int = 75
    checkpoint_interval: int = 200
    test_sequences_per_eval: int = 15
    
    # Infrastructure
    save_dir: str = 'results/stable_runs_v2'
    seed: int = 42
    world_size: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

def get_stable_config_v3(name: str = 'stable') -> StableTrainingConfigV3:
    """Get stable configurations with higher max_steps to prevent premature termination"""
    configs = {
        'stable': StableTrainingConfigV3(),
        
        'stable_conservative': StableTrainingConfigV3(
            n_episodes=1000,
            learning_rate=5e-5,      # Even more conservative
            entropy_coeff=0.05,      # Less exploration
            clip_epsilon=0.1,        # Smaller clips
            max_steps=50,            # Conservative but higher than before
            run_name='stable-conservative-v2'
        ),
        
        'stable_aggressive': StableTrainingConfigV3(
            n_episodes=2000,
            learning_rate=1.5e-4,    # Slightly higher LR
            entropy_coeff=0.12,      # More exploration
            max_steps=80,            # MUCH higher for aggressive exploration
            run_name='stable-aggressive-v2'
        ),
        
        'stable_test': StableTrainingConfigV3(
            n_episodes=300,          # Quick test
            test_interval=25,        # More frequent testing
            log_interval=10,         # More frequent logging
            max_steps=40,            # Lower for quick testing
            run_name='stable-test-v2'
        ),
        
        # ENHANCED: Configuration specifically for sequence analysis
        'stable_long': StableTrainingConfigV3(
            n_episodes=1800,
            learning_rate=6e-5,      # Slightly lower LR for stability
            entropy_coeff=0.10,      # Moderate exploration
            max_steps=100,           # VERY high for deep exploration
            clip_epsilon=0.12,       # Slightly smaller clips
            run_name='stable-long-episodes-v2'
        ),
        
        # NEW: Configuration for sequence validation runs
        'stable_sequence_analysis': StableTrainingConfigV3(
            n_episodes=500,          # Shorter for analysis
            learning_rate=8e-5,      
            entropy_coeff=0.08,      
            max_steps=60,            # Standard higher limit
            test_interval=50,        # More frequent evaluation
            log_interval=10,         # Detailed logging
            run_name='stable-sequence-analysis-v2'
        )
    }
    
    return configs.get(name, configs['stable'])