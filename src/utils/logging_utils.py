import wandb
import torch
import numpy as np
from typing import Dict, Any, Optional

class WandBLogger:
    def __init__(self, config: Dict[str, Any], rank: int = 0):
        self.rank = rank
        self.is_main_process = (rank == 0)
        
        if self.is_main_process:
            wandb.init(
                project=config['project_name'],
                name=config['run_name'],
                config=config,
                tags=['distributed', 'curriculum', 'spider-silk']
            )
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to wandb"""
        if self.is_main_process:
            wandb.log(metrics, step=step)
    
    def log_episode(self, episode: int, episode_data: Dict[str, Any], 
                   difficulty_level: int, trainer_metrics: Dict[str, Any]):
        """Log episode-specific metrics"""
        if self.is_main_process:
            metrics = {
                'episode': episode,
                'reward': episode_data['episode_reward'],
                'episode_length': episode_data['episode_length'],
                'difficulty_level': difficulty_level,
                'total_edits': len(episode_data['edit_history']),
                'policy_loss': trainer_metrics['policy_loss'],
                'value_loss': trainer_metrics['value_loss'],
            }
            self.log(metrics)
    
    def log_test_results(self, test_results: Dict[str, Any]):
        """Log test evaluation results"""
        if self.is_main_process:
            metrics = {
                'test_avg_reward': test_results['avg_reward'],
                'test_avg_improvement': test_results['avg_improvement'],
                'test_success_rate': test_results['success_rate']
            }
            self.log(metrics)
    
    def log_curriculum_progress(self, episode: int, max_episodes: int, 
                               difficulty_counts: Dict[int, int]):
        """Log curriculum learning progress"""
        if self.is_main_process:
            metrics = {
                'curriculum_progress': episode / max_episodes,
                **{f'difficulty_{i}': count for i, count in difficulty_counts.items()}
            }
            self.log(metrics)
    
    def finish(self):
        """Finish wandb run"""
        if self.is_main_process:
            wandb.finish()