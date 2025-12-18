#!/usr/bin/env python3
"""
Ablation Study Script for Uncertainty-Aware Causal Temporal GNN.

This script systematically evaluates the contribution of each component:
1. Base GNN (LightGCN-style, no causal, no uncertainty)
2. + Temporal Granger edges
3. + Uncertainty embeddings
4. + MC Dropout
5. + Calibration (Full model)

Run this to generate the ablation table for your paper.

Usage:
    python run_ablation_study.py --dataset ml-100k
    python run_ablation_study.py --dataset ml-1m --quick  # Fewer epochs for testing
"""

import os
import sys
import json
import argparse
from datetime import datetime
import numpy as np

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

import torch

from causal_gnn.config import Config
from causal_gnn.training import RecommendationSystem, UncertaintyAwareRecommendationSystem


def get_ablation_configs(dataset='ml-100k', quick=False):
    """
    Generate configurations for each ablation variant.

    Returns dict mapping variant name to config.
    """
    # Base settings
    base_kwargs = {
        'embedding_dim': 64,
        'num_layers': 2,
        'batch_size': 2048,
        'num_epochs': 50 if not quick else 10,
        'learning_rate': 0.001,
        'early_stopping_patience': 10 if not quick else 3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'use_tensorboard': False,
        'use_wandb': False,
    }

    variants = {}

    # 1. Base GNN (no causal discovery, no uncertainty)
    variants['Base GNN'] = Config(
        **base_kwargs,
        causal_method='simple',  # Minimal causal
        causal_strength=0.0,     # Disable causal
        use_uncertainty=False,
    )

    # 2. + Temporal edges (Granger-inspired)
    variants['+ Temporal'] = Config(
        **base_kwargs,
        causal_method='advanced',
        causal_strength=0.3,
        use_uncertainty=False,
    )

    # 3. + Uncertainty embeddings (learnable variance)
    variants['+ Uncertainty'] = Config(
        **base_kwargs,
        causal_method='advanced',
        causal_strength=0.3,
        use_uncertainty=True,
        mc_dropout_samples=1,  # No MC dropout yet
        uncertainty_weight=0.05,
    )

    # 4. + MC Dropout (epistemic uncertainty)
    variants['+ MC Dropout'] = Config(
        **base_kwargs,
        causal_method='advanced',
        causal_strength=0.3,
        use_uncertainty=True,
        mc_dropout_samples=10,
        uncertainty_weight=0.1,
    )

    # 5. Full model (+ calibration)
    variants['Full Model'] = Config(
        **base_kwargs,
        causal_method='advanced',
        causal_strength=0.5,
        use_uncertainty=True,
        mc_dropout_samples=10,
        uncertainty_weight=0.1,
        # Calibration is automatic in UncertaintyAwareRecommendationSystem
    )

    return variants


def run_single_variant(name: str, config: Config, data_path: str) -> dict:
    """
    Run training and evaluation for a single ablation variant.

    Returns dict of metrics.
    """
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")

    # Choose system based on whether uncertainty is enabled
    if config.use_uncertainty:
        system = UncertaintyAwareRecommendationSystem(config)
    else:
        system = RecommendationSystem(config)

    # Load and preprocess data
    system.load_data(data_path)
    system.preprocess_data()
    system.split_data()
    system.create_graph()
    system.initialize_model()

    # Train
    system.train()

    # Evaluate
    test_metrics = system.evaluate('test', k_values=[5, 10, 20])

    # Get uncertainty metrics if available
    results = {
        'NDCG@5': test_metrics['ndcg'].get(5, 0),
        'NDCG@10': test_metrics['ndcg'].get(10, 0),
        'NDCG@20': test_metrics['ndcg'].get(20, 0),
        'Recall@10': test_metrics['recall'].get(10, 0),
        'Precision@10': test_metrics['precision'].get(10, 0),
    }

    # Add uncertainty metrics if available
    if hasattr(system, 'evaluate_with_uncertainty'):
        unc_metrics = system.evaluate_with_uncertainty('test', k_values=[10])
        results['ECE'] = unc_metrics.get('ece', 0)
        results['MCE'] = unc_metrics.get('mce', 0)
        results['Unc_Corr'] = unc_metrics.get('uncertainty_correlation', 0)
    else:
        results['ECE'] = float('nan')
        results['MCE'] = float('nan')
        results['Unc_Corr'] = float('nan')

    return results


def run_ablation_study(args):
    """Run complete ablation study."""
    print("\n" + "="*70)
    print("ABLATION STUDY: Uncertainty-Aware Causal Temporal GNN")
    print("="*70)

    # Setup
    results_dir = os.path.join(args.results_dir, f'ablation_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(results_dir, exist_ok=True)

    # Get data path
    from train_movielens import download_movielens
    data_path = download_movielens(args.dataset, args.data_dir)

    # Get ablation configs
    variants = get_ablation_configs(args.dataset, args.quick)

    # Run each variant
    all_results = {}
    for name, config in variants.items():
        config.checkpoint_dir = os.path.join(results_dir, 'checkpoints', name.replace(' ', '_'))
        config.log_dir = os.path.join(results_dir, 'logs', name.replace(' ', '_'))

        try:
            results = run_single_variant(name, config, data_path)
            all_results[name] = results
            print(f"\n{name} Results:")
            for metric, value in results.items():
                if not np.isnan(value):
                    print(f"  {metric}: {value:.4f}")
        except Exception as e:
            print(f"Error running {name}: {e}")
            all_results[name] = {'error': str(e)}

    # Save results
    results_path = os.path.join(results_dir, 'ablation_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")

    # Generate comparison table
    print("\n" + "="*70)
    print("ABLATION STUDY RESULTS")
    print("="*70)

    # Header
    metrics = ['NDCG@10', 'Recall@10', 'ECE', 'Unc_Corr']
    header = f"{'Variant':<20} | " + " | ".join(f"{m:>10}" for m in metrics)
    print(header)
    print("-" * len(header))

    # Data rows
    for name, results in all_results.items():
        if 'error' not in results:
            row = f"{name:<20} | "
            row += " | ".join(
                f"{results.get(m, float('nan')):>10.4f}" if not np.isnan(results.get(m, float('nan')))
                else f"{'N/A':>10}"
                for m in metrics
            )
            print(row)

    # Generate LaTeX table
    latex_path = os.path.join(results_dir, 'ablation_table.tex')
    with open(latex_path, 'w') as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Ablation Study Results}\n")
        f.write("\\begin{tabular}{l" + "c" * len(metrics) + "}\n")
        f.write("\\toprule\n")
        f.write("Model & " + " & ".join(metrics) + " \\\\\n")
        f.write("\\midrule\n")

        for name, results in all_results.items():
            if 'error' not in results:
                row = name + " & "
                row += " & ".join(
                    f"{results.get(m, float('nan')):.4f}" if not np.isnan(results.get(m, float('nan')))
                    else "N/A"
                    for m in metrics
                )
                row += " \\\\\n"
                f.write(row)

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\label{tab:ablation}\n")
        f.write("\\end{table}\n")

    print(f"\nLaTeX table saved to: {latex_path}")

    # Generate visualization
    try:
        from causal_gnn.evaluation import plot_ablation_study

        # Prepare data for plotting
        plot_results = {
            name: {
                'NDCG@10': results.get('NDCG@10', 0),
                'ECE': results.get('ECE', 0) if not np.isnan(results.get('ECE', float('nan'))) else 0,
                'Recall@10': results.get('Recall@10', 0),
            }
            for name, results in all_results.items()
            if 'error' not in results
        }

        plot_ablation_study(
            plot_results,
            save_path=os.path.join(results_dir, 'ablation_plot.png'),
            metrics_to_show=['NDCG@10', 'Recall@10', 'ECE'],
            title='Ablation Study: Component Contributions'
        )
    except Exception as e:
        print(f"Could not generate plot: {e}")

    print(f"\n{'='*70}")
    print(f"Ablation study complete! Results in: {results_dir}")
    print(f"{'='*70}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Run ablation study')

    parser.add_argument(
        '--dataset',
        type=str,
        default='ml-100k',
        choices=['ml-100k', 'ml-1m'],
        help='Dataset to use'
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data',
        help='Data directory'
    )

    parser.add_argument(
        '--results_dir',
        type=str,
        default='./results',
        help='Results directory'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick run with fewer epochs (for testing)'
    )

    args = parser.parse_args()
    run_ablation_study(args)


if __name__ == '__main__':
    main()
