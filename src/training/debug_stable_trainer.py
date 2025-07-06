# src/training/debug_stable_trainer.py
import torch
import numpy as np
from typing import Dict, List, Any
import logging
from src.training.simple_stable_trainer_v2 import SimpleStableTrainerV2

logger = logging.getLogger(__name__)

class DebugStableTrainer(SimpleStableTrainerV2):
    """Enhanced trainer with detailed debugging and failure analysis"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Debug tracking
        self.episode_failures = []
        self.success_episodes = []
        self.failure_patterns = {
            'invalid_position': 0,
            'no_improvement': 0,
            'negative_reward': 0,
            'immediate_stop': 0
        }
        
        # Success analysis
        self.best_improvements = []
        self.successful_sequences = []

    def train_episode(self, starting_sequence: str, episode_number: int, difficulty_level=None):
        """Enhanced training with comprehensive debugging"""
        
        # Store sequence length for bounds checking
        sequence_length = len(starting_sequence)
        
        # Train the episode
        result = super().train_episode(starting_sequence, episode_number, difficulty_level)
        
        # Comprehensive failure analysis
        self._analyze_episode_result(result, episode_number, starting_sequence, sequence_length)
        
        # Log detailed info for problematic episodes
        if result['episode_reward'] < -2.0 or episode_number % 100 == 0:
            self._log_detailed_episode_info(result, episode_number, starting_sequence)
        
        return result
    
    def _analyze_episode_result(self, result, episode_number, starting_sequence, sequence_length):
        """Analyze episode result for failure patterns"""
        
        episode_reward = result['episode_reward']
        actual_improvement = result.get('actual_improvement', 0)
        edit_count = len(result.get('edit_history', []))
        
        # Categorize the episode
        if episode_reward < -5.0:
            # Major failure
            self.episode_failures.append({
                'episode': episode_number,
                'reward': episode_reward,
                'improvement': actual_improvement,
                'edit_count': edit_count,
                'sequence_length': sequence_length,
                'type': 'major_failure'
            })
            self.failure_patterns['negative_reward'] += 1
            
        elif actual_improvement <= 0 and edit_count == 0:
            # Immediate stop without trying
            self.failure_patterns['immediate_stop'] += 1
            
        elif actual_improvement <= 0 and edit_count > 0:
            # Tried but failed to improve
            self.failure_patterns['no_improvement'] += 1
            
        elif actual_improvement > 0.001:
            # Success!
            self.success_episodes.append({
                'episode': episode_number,
                'reward': episode_reward,
                'improvement': actual_improvement,
                'edit_count': edit_count,
                'sequence_length': sequence_length,
                'final_sequence': result.get('final_sequence', starting_sequence)
            })
            
            if actual_improvement > 0.01:
                self.best_improvements.append({
                    'episode': episode_number,
                    'improvement': actual_improvement,
                    'sequence': starting_sequence[:50] + "..."
                })
    
    def _log_detailed_episode_info(self, result, episode_number, starting_sequence):
        """Log detailed information for analysis"""
        
        print(f"\n🔍 Episode {episode_number} Analysis:")
        print(f"  Sequence length: {len(starting_sequence)}")
        print(f"  Episode reward: {result['episode_reward']:.3f}")
        print(f"  Actual improvement: {result.get('actual_improvement', 0):.4f}")
        print(f"  Edit count: {len(result.get('edit_history', []))}")
        print(f"  Final sequence length: {len(result.get('final_sequence', starting_sequence))}")
        
        # Check for specific failure modes
        if result['episode_reward'] < -5.0:
            print(f"  🚨 MAJOR FAILURE DETECTED")
            # Look for position errors in edit history
            if 'edit_history' in result:
                for i, edit in enumerate(result['edit_history'][:3]):
                    print(f"    Edit {i}: {edit}")
        
        # Log environment failure info if available
        if 'failure_counts' in result:
            failure_counts = result['failure_counts']
            total_failures = sum(failure_counts.values())
            if total_failures > 0:
                print(f"  Environment failures: {total_failures}")
                for failure_type, count in failure_counts.items():
                    if count > 0:
                        print(f"    {failure_type}: {count}")
    
    def get_debug_summary(self):
        """Get comprehensive debugging summary"""
        total_episodes = len(self.episode_rewards)
        if total_episodes == 0:
            return "No episodes completed yet"
        
        success_count = len(self.success_episodes)
        failure_count = len(self.episode_failures)
        success_rate = (success_count / total_episodes) * 100
        
        summary = f"\n🔍 DEBUG SUMMARY (Episodes: {total_episodes})\n"
        summary += "="*50 + "\n"
        
        # Success/Failure Analysis
        summary += f"Success Rate: {success_rate:.1f}% ({success_count}/{total_episodes})\n"
        summary += f"Major Failures: {failure_count} ({(failure_count/total_episodes)*100:.1f}%)\n"
        
        # Failure Pattern Analysis
        summary += f"\nFailure Patterns:\n"
        for pattern, count in self.failure_patterns.items():
            if count > 0:
                percentage = (count / total_episodes) * 100
                summary += f"  {pattern}: {count} ({percentage:.1f}%)\n"
        
        # Best Improvements
        if self.best_improvements:
            summary += f"\nBest Improvements:\n"
            sorted_improvements = sorted(self.best_improvements, 
                                       key=lambda x: x['improvement'], reverse=True)[:5]
            for imp in sorted_improvements:
                summary += f"  Episode {imp['episode']}: {imp['improvement']:.4f}\n"
        
        # Recent Performance
        if total_episodes >= 100:
            recent_rewards = self.episode_rewards[-100:]
            recent_avg = np.mean(recent_rewards)
            summary += f"\nRecent Performance (last 100 episodes):\n"
            summary += f"  Average reward: {recent_avg:.3f}\n"
            summary += f"  Success rate: {len([r for r in recent_rewards if r > 2.0])}%\n"
        
        # Recommendations
        summary += f"\nRecommendations:\n"
        if self.failure_patterns['invalid_position'] > total_episodes * 0.1:
            summary += "  🚨 Fix position bounds checking (>10% invalid positions)\n"
        if self.failure_patterns['immediate_stop'] > total_episodes * 0.3:
            summary += "  🚨 Increase exploration (>30% immediate stops)\n"
        if success_rate < 20:
            summary += "  🚨 System needs major debugging (success rate <20%)\n"
        elif success_rate < 40:
            summary += "  ⚠️  Improve learning algorithm (success rate <40%)\n"
        else:
            summary += "  ✅ System performing reasonably well\n"
        
        return summary
    
    def save_debug_data(self, filepath):
        """Save debugging data for analysis"""
        debug_data = {
            'episode_rewards': self.episode_rewards,
            'success_episodes': self.success_episodes,
            'episode_failures': self.episode_failures,
            'failure_patterns': self.failure_patterns,
            'best_improvements': self.best_improvements
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(debug_data, f, indent=2)
        
        print(f"Debug data saved to {filepath}")