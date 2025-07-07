#!/usr/bin/env python3
"""
research_analysis.py - Statistical Analysis and Publication Tools

This script provides comprehensive statistical analysis and publication-ready
outputs for the protein sequence optimization research results.

Usage:
    # Analyze multi-seed results
    python research_analysis.py --results-dir results/multi_seed_study_stable_20241207_143022

    # Generate publication tables and figures
    python research_analysis.py --results-dir results/multi_seed_study_stable_20241207_143022 --publication-mode

    # Compare multiple configurations
    python research_analysis.py --compare-configs --config-dirs dir1,dir2,dir3
"""

import os
import sys
import argparse
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ResearchAnalyzer:
    """Comprehensive analysis for research publication"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.data = {}
        self.statistics = {}
        self.figures_dir = self.results_dir / "publication_figures"
        self.tables_dir = self.results_dir / "publication_tables"
        
        # Create output directories
        self.figures_dir.mkdir(exist_ok=True)
        self.tables_dir.mkdir(exist_ok=True)
        
        # Load data
        self._load_results()
    
    def _load_results(self):
        """Load all results from the directory"""
        logger.info(f"Loading results from {self.results_dir}")
        
        # Load multi-seed summary
        summary_file = self.results_dir / "multi_seed_summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                self.data['summary'] = json.load(f)
        
        # Load aggregate statistics
        agg_file = self.results_dir / "aggregate_statistics.json"
        if agg_file.exists():
            with open(agg_file) as f:
                self.data['aggregate'] = json.load(f)
        
        # Load individual experiment results
        self.data['individual_experiments'] = []
        
        for seed_dir in self.results_dir.glob("*_seed_*"):
            if seed_dir.is_dir():
                exp_data = self._load_individual_experiment(seed_dir)
                if exp_data:
                    self.data['individual_experiments'].append(exp_data)
        
        logger.info(f"Loaded {len(self.data['individual_experiments'])} individual experiments")
    
    def _load_individual_experiment(self, exp_dir: Path) -> Dict:
        """Load data from individual experiment"""
        try:
            exp_data = {'experiment_dir': str(exp_dir)}
            
            # Load metrics
            metrics_file = list(exp_dir.glob("*_metrics.json"))
            if metrics_file:
                with open(metrics_file[0]) as f:
                    exp_data['metrics'] = json.load(f)
            
            # Load episodes
            episodes_file = list(exp_dir.glob("*_episodes.pkl"))
            if episodes_file:
                with open(episodes_file[0], 'rb') as f:
                    exp_data['episodes'] = pickle.load(f)
            
            # Load statistics
            stats_file = list(exp_dir.glob("*_statistics.json"))
            if stats_file:
                with open(stats_file[0]) as f:
                    exp_data['statistics'] = json.load(f)
            
            # Load test evaluations
            test_file = list(exp_dir.glob("*_test_evals.json"))
            if test_file:
                with open(test_file[0]) as f:
                    exp_data['test_evaluations'] = json.load(f)
            
            # Load baselines
            baseline_file = list(exp_dir.glob("*_baselines.json"))
            if baseline_file:
                with open(baseline_file[0]) as f:
                    exp_data['baselines'] = json.load(f)
            
            return exp_data
            
        except Exception as e:
            logger.warning(f"Could not load experiment from {exp_dir}: {e}")
            return None
    
    def generate_performance_summary_table(self) -> pd.DataFrame:
        """Generate performance summary table for publication"""
        logger.info("Generating performance summary table...")
        
        # Collect performance metrics across all runs
        performance_data = []
        
        for exp in self.data['individual_experiments']:
            if 'test_evaluations' in exp and exp['test_evaluations']:
                # Get final test evaluation
                final_eval = exp['test_evaluations'][-1]
                
                # Get experiment info
                exp_id = Path(exp['experiment_dir']).name
                seed = int(exp_id.split('_seed_')[1].split('_')[0])
                
                performance_data.append({
                    'Seed': seed,
                    'Avg_Improvement': final_eval['avg_improvement'],
                    'Success_Rate': final_eval['success_rate'],
                    'Avg_Reward': final_eval['avg_reward']
                })
                
                # Add baseline comparisons if available
                if 'baselines' in exp:
                    baselines = exp['baselines']
                    for baseline_name, baseline_data in baselines.items():
                        performance_data.append({
                            'Seed': seed,
                            'Method': baseline_name,
                            'Avg_Improvement': baseline_data['avg_improvement'],
                            'Success_Rate': baseline_data['success_rate'],
                            'Avg_Reward': baseline_data.get('avg_reward', 0.0)
                        })
        
        if not performance_data:
            logger.warning("No performance data found")
            return pd.DataFrame()
        
        df = pd.DataFrame(performance_data)
        
        # Calculate summary statistics
        summary_stats = []
        
        # RL Agent statistics
        rl_data = df[~df.get('Method', '').notna()]  # Rows without Method are RL agent
        if not rl_data.empty:
            summary_stats.append({
                'Method': 'RL Agent',
                'Mean_Improvement': f"{rl_data['Avg_Improvement'].mean():.4f} ± {rl_data['Avg_Improvement'].std():.4f}",
                'Mean_Success_Rate': f"{rl_data['Success_Rate'].mean():.3f} ± {rl_data['Success_Rate'].std():.3f}",
                'Mean_Reward': f"{rl_data['Avg_Reward'].mean():.3f} ± {rl_data['Avg_Reward'].std():.3f}",
                'N_Runs': len(rl_data)
            })
        
        # Baseline statistics
        for method in df[df.get('Method', '').notna()]['Method'].unique():
            method_data = df[df['Method'] == method]
            summary_stats.append({
                'Method': method.replace('_', ' ').title(),
                'Mean_Improvement': f"{method_data['Avg_Improvement'].mean():.4f} ± {method_data['Avg_Improvement'].std():.4f}",
                'Mean_Success_Rate': f"{method_data['Success_Rate'].mean():.3f} ± {method_data['Success_Rate'].std():.3f}",
                'Mean_Reward': f"{method_data['Avg_Reward'].mean():.3f} ± {method_data['Avg_Reward'].std():.3f}",
                'N_Runs': len(method_data)
            })
        
        summary_df = pd.DataFrame(summary_stats)
        
        # Save table
        summary_df.to_csv(self.tables_dir / "performance_summary.csv", index=False)
        
        # Create LaTeX table
        latex_table = summary_df.to_latex(
            index=False,
            caption="Performance comparison across methods and seeds",
            label="tab:performance_summary",
            escape=False
        )
        
        with open(self.tables_dir / "performance_summary.tex", 'w') as f:
            f.write(latex_table)
        
        logger.info("Performance summary table saved")
        return summary_df
    
    def perform_statistical_tests(self) -> Dict[str, Any]:
        """Perform comprehensive statistical tests"""
        logger.info("Performing statistical significance tests...")
        
        # Collect data for statistical tests
        rl_improvements = []
        rl_success_rates = []
        baseline_improvements = {'random': [], 'greedy': [], 'no_modification': []}
        
        for exp in self.data['individual_experiments']:
            if 'test_evaluations' in exp and exp['test_evaluations']:
                final_eval = exp['test_evaluations'][-1]
                rl_improvements.append(final_eval['avg_improvement'])
                rl_success_rates.append(final_eval['success_rate'])
                
                # Collect baseline data
                if 'baselines' in exp:
                    baselines = exp['baselines']
                    for baseline_name in baseline_improvements.keys():
                        if baseline_name in baselines:
                            baseline_improvements[baseline_name].append(
                                baselines[baseline_name]['avg_improvement']
                            )
        
        statistical_results = {
            'sample_sizes': {
                'rl_agent': len(rl_improvements),
                'baselines': {k: len(v) for k, v in baseline_improvements.items()}
            },
            'normality_tests': {},
            'significance_tests': {},
            'effect_sizes': {},
            'confidence_intervals': {}
        }
        
        if not rl_improvements:
            logger.warning("No RL data for statistical tests")
            return statistical_results
        
        # Normality tests (Shapiro-Wilk)
        if len(rl_improvements) >= 3:
            shapiro_stat, shapiro_p = stats.shapiro(rl_improvements)
            statistical_results['normality_tests']['rl_improvements'] = {
                'statistic': float(shapiro_stat),
                'p_value': float(shapiro_p),
                'is_normal': bool(shapiro_p > 0.05)
            }
        
        # Bootstrap confidence intervals for RL agent
        ci_95 = self._bootstrap_ci(rl_improvements, confidence_level=0.95)
        ci_99 = self._bootstrap_ci(rl_improvements, confidence_level=0.99)
        
        statistical_results['confidence_intervals']['rl_improvements'] = {
            '95%': [float(ci_95[0]), float(ci_95[1])],
            '99%': [float(ci_99[0]), float(ci_99[1])]
        }
        
        # Statistical tests vs baselines
        for baseline_name, baseline_data in baseline_improvements.items():
            if len(baseline_data) == len(rl_improvements) and len(baseline_data) > 0:
                
                # Paired tests (same sequences)
                t_stat, t_p = ttest_rel(rl_improvements, baseline_data)
                wilcoxon_stat, wilcoxon_p = wilcoxon(rl_improvements, baseline_data, 
                                                   zero_method='wilcox', alternative='two-sided')
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt((np.var(rl_improvements, ddof=1) + np.var(baseline_data, ddof=1)) / 2)
                cohens_d = (np.mean(rl_improvements) - np.mean(baseline_data)) / pooled_std
                
                # Cliff's delta (non-parametric effect size)
                cliffs_delta = self._calculate_cliffs_delta(rl_improvements, baseline_data)
                
                statistical_results['significance_tests'][baseline_name] = {
                    'paired_t_test': {
                        'statistic': float(t_stat),
                        'p_value': float(t_p),
                        'significant_05': bool(t_p < 0.05),
                        'significant_01': bool(t_p < 0.01),
                        'significant_001': bool(t_p < 0.001)
                    },
                    'wilcoxon_test': {
                        'statistic': float(wilcoxon_stat),
                        'p_value': float(wilcoxon_p),
                        'significant_05': bool(wilcoxon_p < 0.05),
                        'significant_01': bool(wilcoxon_p < 0.01)
                    }
                }
                
                statistical_results['effect_sizes'][baseline_name] = {
                    'cohens_d': float(cohens_d),
                    'cohens_d_interpretation': self._interpret_cohens_d(cohens_d),
                    'cliffs_delta': float(cliffs_delta),
                    'cliffs_delta_interpretation': self._interpret_cliffs_delta(cliffs_delta)
                }
        
        # One-sample test vs zero (no improvement)
        t_stat_zero, t_p_zero = stats.ttest_1samp(rl_improvements, 0)
        statistical_results['significance_tests']['vs_zero'] = {
            't_statistic': float(t_stat_zero),
            'p_value': float(t_p_zero),
            'significant': bool(t_p_zero < 0.05)
        }
        
        # Save statistical results
        with open(self.tables_dir / "statistical_tests.json", 'w') as f:
            json.dump(statistical_results, f, indent=2)
        
        # Create statistical summary table
        self._create_statistical_summary_table(statistical_results)
        
        logger.info("Statistical tests completed")
        return statistical_results
    
    def _bootstrap_ci(self, data: List[float], confidence_level: float = 0.95, n_bootstrap: int = 10000) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval"""
        bootstrap_means = []
        data_array = np.array(data)
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data_array, size=len(data_array), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        alpha = 1 - confidence_level
        lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
        
        return lower, upper
    
    def _calculate_cliffs_delta(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cliff's delta effect size"""
        n1, n2 = len(group1), len(group2)
        greater = sum(1 for x1 in group1 for x2 in group2 if x1 > x2)
        return (greater - (n1 * n2 - greater)) / (n1 * n2)
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _interpret_cliffs_delta(self, delta: float) -> str:
        """Interpret Cliff's delta effect size"""
        abs_delta = abs(delta)
        if abs_delta < 0.147:
            return "negligible"
        elif abs_delta < 0.33:
            return "small"
        elif abs_delta < 0.474:
            return "medium"
        else:
            return "large"
    
    def _create_statistical_summary_table(self, statistical_results: Dict):
        """Create publication-ready statistical summary table"""
        
        # Create summary table
        summary_data = []
        
        for baseline, tests in statistical_results['significance_tests'].items():
            if baseline == 'vs_zero':
                continue
                
            effect_sizes = statistical_results['effect_sizes'].get(baseline, {})
            
            summary_data.append({
                'Comparison': f"RL vs {baseline.replace('_', ' ').title()}",
                'T-test p-value': f"{tests['paired_t_test']['p_value']:.6f}",
                'Wilcoxon p-value': f"{tests['wilcoxon_test']['p_value']:.6f}",
                'Cohen\'s d': f"{effect_sizes.get('cohens_d', 0):.3f} ({effect_sizes.get('cohens_d_interpretation', 'N/A')})",
                'Cliff\'s δ': f"{effect_sizes.get('cliffs_delta', 0):.3f} ({effect_sizes.get('cliffs_delta_interpretation', 'N/A')})",
                'Significant (α=0.05)': '✓' if tests['paired_t_test']['significant_05'] else '✗'
            })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(self.tables_dir / "statistical_summary.csv", index=False)
            
            # LaTeX table
            latex_table = summary_df.to_latex(
                index=False,
                caption="Statistical significance tests and effect sizes",
                label="tab:statistical_tests",
                escape=False
            )
            
            with open(self.tables_dir / "statistical_summary.tex", 'w') as f:
                f.write(latex_table)
    
    def create_publication_figures(self):
        """Create all publication-ready figures"""
        logger.info("Creating publication figures...")
        
        # Set publication style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16
        })
        
        # 1. Performance comparison box plots
        self._create_performance_boxplots()
        
        # 2. Learning curves
        self._create_learning_curves()
        
        # 3. Edit pattern analysis
        self._create_edit_analysis_plots()
        
        # 4. Convergence analysis
        self._create_convergence_plots()
        
        # 5. Confidence intervals plot
        self._create_confidence_intervals_plot()
        
        logger.info("Publication figures created")
    
    def _create_performance_boxplots(self):
        """Create performance comparison box plots"""
        
        # Collect data
        plot_data = []
        
        for exp in self.data['individual_experiments']:
            if 'test_evaluations' in exp and exp['test_evaluations']:
                final_eval = exp['test_evaluations'][-1]
                plot_data.append({
                    'Method': 'RL Agent',
                    'Improvement': final_eval['avg_improvement'],
                    'Success Rate': final_eval['success_rate'],
                    'Reward': final_eval['avg_reward']
                })
                
                if 'baselines' in exp:
                    for baseline_name, baseline_data in exp['baselines'].items():
                        plot_data.append({
                            'Method': baseline_name.replace('_', ' ').title(),
                            'Improvement': baseline_data['avg_improvement'],
                            'Success Rate': baseline_data['success_rate'],
                            'Reward': baseline_data.get('avg_reward', 0.0)
                        })
        
        if not plot_data:
            logger.warning("No data for performance box plots")
            return
        
        df = pd.DataFrame(plot_data)
        
        # Create subplot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Improvement box plot
        sns.boxplot(data=df, x='Method', y='Improvement', ax=axes[0])
        axes[0].set_title('Toughness Improvement')
        axes[0].set_ylabel('Toughness Improvement')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Success rate box plot
        sns.boxplot(data=df, x='Method', y='Success Rate', ax=axes[1])
        axes[1].set_title('Success Rate')
        axes[1].set_ylabel('Success Rate')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Reward box plot
        sns.boxplot(data=df, x='Method', y='Reward', ax=axes[2])
        axes[2].set_title('Episode Reward')
        axes[2].set_ylabel('Average Reward')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / "performance_comparison.pdf", bbox_inches='tight')
        plt.close()
    
    def _create_learning_curves(self):
        """Create learning curves with confidence bands"""
        
        # Collect episode rewards across all runs
        all_rewards = []
        max_episodes = 0
        
        for exp in self.data['individual_experiments']:
            if 'metrics' in exp and 'episode_rewards' in exp['metrics']:
                rewards = exp['metrics']['episode_rewards']
                all_rewards.append(rewards)
                max_episodes = max(max_episodes, len(rewards))
        
        if not all_rewards:
            logger.warning("No learning curve data available")
            return
        
        # Pad sequences to same length
        padded_rewards = []
        for rewards in all_rewards:
            if len(rewards) < max_episodes:
                # Pad with last value
                padded = rewards + [rewards[-1]] * (max_episodes - len(rewards))
            else:
                padded = rewards[:max_episodes]
            padded_rewards.append(padded)
        
        # Calculate statistics
        rewards_array = np.array(padded_rewards)
        mean_rewards = np.mean(rewards_array, axis=0)
        std_rewards = np.std(rewards_array, axis=0)
        episodes = np.arange(1, max_episodes + 1)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Plot individual runs (faded)
        for rewards in padded_rewards:
            plt.plot(episodes, rewards, alpha=0.2, color='gray', linewidth=0.5)
        
        # Plot mean with confidence band
        plt.plot(episodes, mean_rewards, color='blue', linewidth=2, label='Mean')
        plt.fill_between(episodes, 
                        mean_rewards - std_rewards, 
                        mean_rewards + std_rewards,
                        alpha=0.3, color='blue', label='±1 SD')
        
        plt.xlabel('Episode')
        plt.ylabel('Episode Reward')
        plt.title('Learning Curves Across Multiple Seeds')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / "learning_curves.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / "learning_curves.pdf", bbox_inches='tight')
        plt.close()
    
    def _create_edit_analysis_plots(self):
        """Create edit pattern analysis plots"""
        
        # Collect edit patterns
        edit_types_data = []
        position_data = []
        
        for exp in self.data['individual_experiments']:
            if 'metrics' in exp and 'edit_patterns' in exp['metrics']:
                patterns = exp['metrics']['edit_patterns']
                
                for pattern in patterns:
                    for edit_type in pattern.get('edit_types', []):
                        edit_types_data.append(edit_type)
                    
                    for position in pattern.get('positions', []):
                        position_data.append(position)
        
        if not edit_types_data:
            logger.warning("No edit pattern data available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Edit type distribution
        edit_counts = pd.Series(edit_types_data).value_counts()
        edit_counts.plot(kind='bar', ax=axes[0])
        axes[0].set_title('Distribution of Edit Types')
        axes[0].set_ylabel('Frequency')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Position distribution (if available)
        if position_data:
            axes[1].hist(position_data, bins=20, alpha=0.7, edgecolor='black')
            axes[1].set_title('Distribution of Edit Positions')
            axes[1].set_xlabel('Position in Sequence')
            axes[1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / "edit_analysis.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / "edit_analysis.pdf", bbox_inches='tight')
        plt.close()
    
    def _create_convergence_plots(self):
        """Create convergence analysis plots"""
        
        convergence_episodes = []
        
        for exp in self.data['individual_experiments']:
            if 'statistics' in exp and 'convergence_analysis' in exp['statistics']:
                conv_data = exp['statistics']['convergence_analysis']
                if 'episodes_to_convergence' in conv_data:
                    convergence_episodes.append(conv_data['episodes_to_convergence'])
        
        if not convergence_episodes:
            logger.warning("No convergence data available")
            return
        
        plt.figure(figsize=(8, 6))
        
        plt.hist(convergence_episodes, bins=10, alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(convergence_episodes), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(convergence_episodes):.0f}')
        plt.axvline(np.median(convergence_episodes), color='orange', linestyle='--',
                   label=f'Median: {np.median(convergence_episodes):.0f}')
        
        plt.xlabel('Episodes to Convergence')
        plt.ylabel('Frequency')
        plt.title('Convergence Time Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / "convergence_analysis.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / "convergence_analysis.pdf", bbox_inches='tight')
        plt.close()
    
    def _create_confidence_intervals_plot(self):
        """Create confidence intervals visualization"""
        
        # Get data from statistical results
        if 'aggregate' not in self.data or 'performance_metrics' not in self.data['aggregate']:
            logger.warning("No aggregate statistics for CI plot")
            return
        
        perf_metrics = self.data['aggregate']['performance_metrics']
        
        metrics = ['improvement', 'success_rate', 'reward']
        means = []
        cis_lower = []
        cis_upper = []
        
        for metric in metrics:
            if metric in perf_metrics:
                means.append(perf_metrics[metric]['mean'])
                ci = perf_metrics[metric]['ci_95']
                cis_lower.append(ci[0])
                cis_upper.append(ci[1])
        
        if not means:
            logger.warning("No CI data available")
            return
        
        # Create error bar plot
        plt.figure(figsize=(10, 6))
        
        x_pos = np.arange(len(metrics))
        errors_lower = [means[i] - cis_lower[i] for i in range(len(means))]
        errors_upper = [cis_upper[i] - means[i] for i in range(len(means))]
        
        plt.errorbar(x_pos, means, yerr=[errors_lower, errors_upper], 
                    fmt='o', capsize=5, capthick=2, elinewidth=2, markersize=8)
        
        plt.xticks(x_pos, [m.replace('_', ' ').title() for m in metrics])
        plt.ylabel('Value')
        plt.title('Performance Metrics with 95% Confidence Intervals')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / "confidence_intervals.png", dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / "confidence_intervals.pdf", bbox_inches='tight')
        plt.close()
    
    def generate_research_report(self):
        """Generate comprehensive research report"""
        logger.info("Generating research report...")
        
        # Perform all analyses
        performance_table = self.generate_performance_summary_table()
        statistical_results = self.perform_statistical_tests()
        self.create_publication_figures()
        
        # Generate written report
        report_lines = []
        report_lines.append("# Protein Sequence Optimization Research Results")
        report_lines.append("")
        
        # Executive Summary
        if 'aggregate' in self.data and 'performance_metrics' in self.data['aggregate']:
            perf = self.data['aggregate']['performance_metrics']
            report_lines.append("## Executive Summary")
            report_lines.append("")
            report_lines.append(f"- **Mean Toughness Improvement**: {perf['improvement']['mean']:.4f} ± {perf['improvement']['std']:.4f}")
            report_lines.append(f"- **Success Rate**: {perf['success_rate']['mean']:.1%} ± {perf['success_rate']['std']:.1%}")
            report_lines.append(f"- **Number of Experiments**: {len(self.data['individual_experiments'])}")
            report_lines.append("")
        
        # Statistical Significance
        if statistical_results and 'significance_tests' in statistical_results:
            report_lines.append("## Statistical Significance")
            report_lines.append("")
            
            for baseline, test_results in statistical_results['significance_tests'].items():
                if baseline == 'vs_zero':
                    continue
                    
                p_val = test_results['paired_t_test']['p_value']
                significant = "significant" if p_val < 0.05 else "not significant"
                
                effect_size = statistical_results['effect_sizes'].get(baseline, {})
                cohens_d = effect_size.get('cohens_d', 0)
                interpretation = effect_size.get('cohens_d_interpretation', 'unknown')
                
                report_lines.append(f"- **vs {baseline.title()}**: p = {p_val:.4f} ({significant}), "
                                  f"Cohen's d = {cohens_d:.3f} ({interpretation})")
            
            report_lines.append("")
        
        # Methodology
        report_lines.append("## Methodology")
        report_lines.append("")
        report_lines.append("- **Multiple random seeds** for statistical rigor")
        report_lines.append("- **Baseline comparisons** against random and greedy methods")
        report_lines.append("- **Paired statistical tests** (t-test and Wilcoxon)")
        report_lines.append("- **Effect size calculations** (Cohen's d and Cliff's δ)")
        report_lines.append("- **Bootstrap confidence intervals**")
        report_lines.append("")
        
        # Files Generated
        report_lines.append("## Files Generated")
        report_lines.append("")
        report_lines.append("### Tables")
        report_lines.append("- `performance_summary.csv/tex` - Performance comparison table")
        report_lines.append("- `statistical_summary.csv/tex` - Statistical test results")
        report_lines.append("")
        report_lines.append("### Figures")
        report_lines.append("- `performance_comparison.png/pdf` - Box plots comparing methods")
        report_lines.append("- `learning_curves.png/pdf` - Training dynamics")
        report_lines.append("- `edit_analysis.png/pdf` - Edit pattern analysis")
        report_lines.append("- `convergence_analysis.png/pdf` - Convergence time distribution")
        report_lines.append("- `confidence_intervals.png/pdf` - CI visualization")
        report_lines.append("")
        
        # Save report
        with open(self.results_dir / "research_report.md", 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info("Research report generated")


def compare_multiple_configs(config_dirs: List[str], output_dir: str):
    """Compare results across multiple configurations"""
    logger.info(f"Comparing {len(config_dirs)} configurations...")
    
    all_results = {}
    
    # Load results from each configuration
    for config_dir in config_dirs:
        config_path = Path(config_dir)
        if config_path.exists():
            analyzer = ResearchAnalyzer(str(config_path))
            config_name = config_path.name
            all_results[config_name] = analyzer.data
    
    # Create comparison plots and tables
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Comparison performance table
    comparison_data = []
    
    for config_name, data in all_results.items():
        if 'aggregate' in data and 'performance_metrics' in data['aggregate']:
            perf = data['aggregate']['performance_metrics']
            comparison_data.append({
                'Configuration': config_name,
                'Mean_Improvement': perf['improvement']['mean'],
                'Std_Improvement': perf['improvement']['std'],
                'Mean_Success_Rate': perf['success_rate']['mean'],
                'Std_Success_Rate': perf['success_rate']['std'],
                'N_Experiments': len(data.get('individual_experiments', []))
            })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(output_path / "configuration_comparison.csv", index=False)
        
        # Create comparison plot
        plt.figure(figsize=(12, 6))
        
        x_pos = np.arange(len(comparison_df))
        plt.errorbar(x_pos, comparison_df['Mean_Improvement'], 
                    yerr=comparison_df['Std_Improvement'],
                    fmt='o', capsize=5, capthick=2, elinewidth=2, markersize=8)
        
        plt.xticks(x_pos, comparison_df['Configuration'], rotation=45)
        plt.ylabel('Mean Toughness Improvement')
        plt.title('Configuration Comparison')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(output_path / "configuration_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Configuration comparison saved to {output_dir}")


def main():
    """Main analysis entry point"""
    parser = argparse.ArgumentParser(description='Research Analysis and Publication Tools')
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Directory containing results to analyze')
    parser.add_argument('--publication-mode', action='store_true',
                       help='Generate publication-ready outputs')
    parser.add_argument('--compare-configs', action='store_true',
                       help='Compare multiple configurations')
    parser.add_argument('--config-dirs', type=str,
                       help='Comma-separated list of config directories to compare')
    parser.add_argument('--output-dir', type=str, default='comparison_results',
                       help='Output directory for comparison results')
    
    args = parser.parse_args()
    
    if args.compare_configs:
        if not args.config_dirs:
            logger.error("--config-dirs required for comparison mode")
            return 1
        
        config_dirs = [d.strip() for d in args.config_dirs.split(',')]
        compare_multiple_configs(config_dirs, args.output_dir)
    
    else:
        # Single analysis
        if not os.path.exists(args.results_dir):
            logger.error(f"Results directory not found: {args.results_dir}")
            return 1
        
        analyzer = ResearchAnalyzer(args.results_dir)
        
        if args.publication_mode:
            analyzer.generate_research_report()
            logger.info("✅ Publication-ready analysis completed!")
            logger.info(f"Results saved to: {analyzer.results_dir}")
            logger.info("Files generated:")
            logger.info("  - research_report.md (comprehensive report)")
            logger.info("  - publication_tables/ (LaTeX and CSV tables)")
            logger.info("  - publication_figures/ (high-resolution figures)")
        else:
            # Quick analysis
            analyzer.generate_performance_summary_table()
            analyzer.perform_statistical_tests()
            analyzer.create_publication_figures()
            logger.info("✅ Analysis completed!")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)