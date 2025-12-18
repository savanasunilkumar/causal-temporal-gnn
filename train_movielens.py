#!/usr/bin/env python3
"""
Comprehensive MovieLens Training Script for Maximum Performance.

This script is optimized for best results with:
- Hyperparameter optimization
- Proper validation strategy
- Comprehensive result storage
- Detailed logging and metrics

Usage:
    python train_movielens.py --dataset ml-100k
    python train_movielens.py --dataset ml-1m --use_uncertainty
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Check optional dependencies
MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib not available. Training curves will not be plotted.")
    print("Install with: pip install matplotlib")

REQUESTS_AVAILABLE = False
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    print("Warning: requests not available. Cannot auto-download datasets.")
    print("Install with: pip install requests")

from causal_gnn.config import Config
from causal_gnn.training import RecommendationSystem, UncertaintyAwareRecommendationSystem

# Import publication metrics (optional)
PUBLICATION_METRICS_AVAILABLE = False
try:
    from causal_gnn.evaluation import (
        generate_publication_metrics,
        generate_all_publication_figures,
        format_latex_table,
    )
    PUBLICATION_METRICS_AVAILABLE = True
except ImportError:
    pass


def download_movielens(dataset='ml-100k', data_dir='./data'):
    """Download and prepare MovieLens dataset."""
    if not REQUESTS_AVAILABLE:
        raise ImportError(
            "The 'requests' package is required to download datasets. "
            "Install with: pip install requests\n"
            "Or manually download MovieLens from https://grouplens.org/datasets/movielens/"
        )

    import zipfile

    os.makedirs(data_dir, exist_ok=True)

    urls = {
        'ml-100k': 'https://files.grouplens.org/datasets/movielens/ml-100k.zip',
        'ml-1m': 'https://files.grouplens.org/datasets/movielens/ml-1m.zip',
        'ml-10m': 'https://files.grouplens.org/datasets/movielens/ml-10m.zip',
    }

    if dataset not in urls:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from {list(urls.keys())}")

    zip_path = os.path.join(data_dir, f'{dataset}.zip')
    extract_path = os.path.join(data_dir, dataset)

    # Download if not exists
    if not os.path.exists(extract_path):
        print(f"Downloading {dataset}...")
        response = requests.get(urls[dataset], stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(zip_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                pct = (downloaded / total_size) * 100 if total_size > 0 else 0
                print(f"\rDownloading: {pct:.1f}%", end='')
        print("\nDownload complete!")

        # Extract
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        os.remove(zip_path)

    # Convert to standard CSV format
    csv_path = os.path.join(data_dir, f'{dataset}_ratings.csv')

    if not os.path.exists(csv_path):
        print("Converting to CSV format...")

        if dataset == 'ml-100k':
            data = pd.read_csv(
                os.path.join(extract_path, 'u.data'),
                sep='\t',
                names=['userId', 'movieId', 'rating', 'timestamp']
            )
        elif dataset == 'ml-1m':
            data = pd.read_csv(
                os.path.join(extract_path, 'ratings.dat'),
                sep='::',
                names=['userId', 'movieId', 'rating', 'timestamp'],
                engine='python'
            )
        elif dataset == 'ml-10m':
            data = pd.read_csv(
                os.path.join(extract_path, 'ml-10M100K', 'ratings.dat'),
                sep='::',
                names=['userId', 'movieId', 'rating', 'timestamp'],
                engine='python'
            )

        data.to_csv(csv_path, index=False)
        print(f"Saved {len(data)} interactions to {csv_path}")

    return csv_path


def get_optimal_config(dataset='ml-100k', use_uncertainty=False):
    """Get optimal hyperparameters for best results."""

    # Base config optimized through experimentation
    base_config = {
        # Model Architecture - larger for better representation
        'embedding_dim': 128,          # Larger embeddings
        'num_layers': 3,               # More layers for deeper learning
        'time_steps': 20,              # More temporal granularity
        'dropout': 0.2,                # Moderate dropout
        'causal_strength': 0.5,        # Balanced causal influence

        # Causal Discovery - use advanced method
        'causal_method': 'advanced',
        'significance_level': 0.05,
        'max_lag': 10,
        'min_causal_strength': 0.05,

        # Training - optimized for convergence
        'learning_rate': 0.001,
        'batch_size': 2048,
        'num_epochs': 100,             # More epochs for convergence
        'neg_samples': 4,              # More negatives for better discrimination
        'weight_decay': 0.0001,
        'contrastive_weight': 0.1,
        'early_stopping_patience': 15,  # Patient early stopping

        # Data Processing - keep more data
        'min_user_interactions': 5,
        'min_item_interactions': 5,
        'test_size': 0.1,              # 10% test
        'val_size': 0.1,               # 10% validation
        'rating_threshold': 3.5,       # Only positive ratings

        # System
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'seed': 42,

        # Advanced features
        'use_amp': torch.cuda.is_available(),  # Mixed precision on GPU
        'use_gradient_checkpointing': False,   # Enable for memory saving
        'use_tensorboard': True,
        'use_wandb': False,

        # Checkpointing
        'save_every_n_epochs': 5,
        'keep_best_k_models': 3,
    }

    # Dataset-specific adjustments
    if dataset == 'ml-1m':
        base_config['batch_size'] = 4096
        base_config['num_epochs'] = 50
        base_config['embedding_dim'] = 128
    elif dataset == 'ml-10m':
        base_config['batch_size'] = 8192
        base_config['num_epochs'] = 30
        base_config['embedding_dim'] = 128
        base_config['use_gradient_checkpointing'] = True
        base_config['use_neighbor_sampling'] = True

    # Uncertainty-specific settings
    if use_uncertainty:
        base_config.update({
            'use_uncertainty': True,
            'mc_dropout_samples': 20,      # More samples for better uncertainty
            'uncertainty_weight': 0.1,
            'n_bootstrap_samples': 200,
            'causal_prior_precision': 1.0,
            'confidence_threshold': 0.6,
            'calibration_bins': 15,
            'abstention_threshold': 0.7,
        })

    return Config(**base_config)


def create_results_directory(base_dir='./results'):
    """Create organized results directory structure."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(base_dir, f'run_{timestamp}')

    # Create subdirectories
    subdirs = ['models', 'metrics', 'plots', 'logs', 'predictions']
    for subdir in subdirs:
        os.makedirs(os.path.join(results_dir, subdir), exist_ok=True)

    return results_dir


def save_results(results_dir, metrics, config, train_history, recommendations=None):
    """Save all results in organized format."""

    # 1. Save metrics as JSON
    metrics_path = os.path.join(results_dir, 'metrics', 'final_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"Metrics saved to: {metrics_path}")

    # 2. Save config
    config_path = os.path.join(results_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2, default=str)
    print(f"Config saved to: {config_path}")

    # 3. Save training history
    history_path = os.path.join(results_dir, 'metrics', 'training_history.json')
    # Convert numpy types to python types
    history_serializable = {
        'train_loss': [float(x) for x in train_history.get('train_loss', [])],
        'val_metrics': {}
    }
    for key, values in train_history.get('val_metrics', {}).items():
        if isinstance(values, list):
            history_serializable['val_metrics'][key] = [float(x) for x in values]
        else:
            history_serializable['val_metrics'][key] = float(values)

    with open(history_path, 'w') as f:
        json.dump(history_serializable, f, indent=2)
    print(f"Training history saved to: {history_path}")

    # 4. Save training curves as CSV for easy plotting
    curves_path = os.path.join(results_dir, 'metrics', 'training_curves.csv')
    curves_data = {'epoch': list(range(1, len(train_history.get('train_loss', [])) + 1))}
    curves_data['train_loss'] = train_history.get('train_loss', [])
    for key, values in train_history.get('val_metrics', {}).items():
        if isinstance(values, list):
            curves_data[f'val_{key}'] = values

    if curves_data.get('train_loss'):
        pd.DataFrame(curves_data).to_csv(curves_path, index=False)
        print(f"Training curves saved to: {curves_path}")

    # 5. Save recommendations if provided
    if recommendations:
        recs_path = os.path.join(results_dir, 'predictions', 'sample_recommendations.json')
        with open(recs_path, 'w') as f:
            json.dump(recommendations, f, indent=2, default=str)
        print(f"Recommendations saved to: {recs_path}")

    # 6. Create summary report
    summary_path = os.path.join(results_dir, 'RESULTS_SUMMARY.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("TRAINING RESULTS SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {config.device}\n")
        f.write(f"Embedding Dim: {config.embedding_dim}\n")
        f.write(f"Num Layers: {config.num_layers}\n")
        f.write(f"Epochs Trained: {len(train_history.get('train_loss', []))}\n\n")

        f.write("-" * 40 + "\n")
        f.write("TEST METRICS\n")
        f.write("-" * 40 + "\n")

        if isinstance(metrics, dict):
            for metric_name, values in metrics.items():
                if isinstance(values, dict):
                    for k, v in values.items():
                        f.write(f"  {metric_name.upper()}@{k}: {v:.4f}\n")
                else:
                    f.write(f"  {metric_name}: {values:.4f}\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"\nSummary saved to: {summary_path}")
    return summary_path


def generate_publication_outputs(results_dir, system, args):
    """Generate publication-ready metrics and figures."""
    if not PUBLICATION_METRICS_AVAILABLE:
        print("Skipping publication outputs (evaluation module not available)")
        return None

    if not args.use_uncertainty:
        print("Skipping publication outputs (uncertainty not enabled)")
        return None

    print("\nGenerating publication-ready outputs...")

    # Create publication subdirectory
    pub_dir = os.path.join(results_dir, 'publication')
    os.makedirs(pub_dir, exist_ok=True)

    try:
        # Get uncertainty predictions on test set
        if hasattr(system, 'evaluate_with_uncertainty'):
            unc_results = system.evaluate_with_uncertainty('test', k_values=[10])

            # Generate publication metrics
            confidences = np.array(unc_results.get('confidences', []))
            predictions = np.array(unc_results.get('predictions', []))
            labels = np.array(unc_results.get('labels', []))

            if len(confidences) > 0 and len(labels) > 0:
                pub_metrics = generate_publication_metrics(
                    predictions=predictions,
                    labels=labels,
                    uncertainties=1 - confidences,  # Convert confidence to uncertainty
                    n_bins=15
                )

                # Generate all publication figures
                generate_all_publication_figures(
                    pub_metrics,
                    output_dir=pub_dir,
                    dataset_name=args.dataset.upper(),
                    model_name='UA-CTGNN'
                )

                # Generate LaTeX table
                latex_table = format_latex_table(
                    {args.dataset.upper(): pub_metrics},
                    metrics=['ece', 'mce', 'brier_score', 'auroc_uncertainty', 'coverage_at_90_accuracy']
                )

                # Save LaTeX table
                latex_path = os.path.join(pub_dir, 'metrics_table.tex')
                with open(latex_path, 'w') as f:
                    f.write(latex_table)
                print(f"LaTeX metrics table saved to: {latex_path}")

                # Save publication metrics as JSON
                metrics_path = os.path.join(pub_dir, 'publication_metrics.json')
                pub_metrics_dict = {
                    'ece': pub_metrics.expected_calibration_error,
                    'mce': pub_metrics.maximum_calibration_error,
                    'brier_score': pub_metrics.brier_score,
                    'auroc_uncertainty': pub_metrics.auroc_uncertainty,
                    'coverage_at_90_accuracy': pub_metrics.coverage_at_90_accuracy,
                    'auc_accuracy_coverage': pub_metrics.auc_accuracy_coverage,
                    'uncertainty_error_correlation': pub_metrics.uncertainty_error_correlation,
                }
                with open(metrics_path, 'w') as f:
                    json.dump(pub_metrics_dict, f, indent=2)
                print(f"Publication metrics saved to: {metrics_path}")

                return pub_metrics

    except Exception as e:
        print(f"Warning: Could not generate publication outputs: {e}")
        import traceback
        traceback.print_exc()

    return None


def plot_training_curves(results_dir, train_history):
    """Generate training curve plots."""
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping plots (matplotlib not installed)")
        print("To enable plots: pip install matplotlib")
        return

    # Plot training loss
    if train_history.get('train_loss'):
        plt.figure(figsize=(12, 5))

        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(train_history['train_loss'], 'b-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True, alpha=0.3)

        # Validation metrics plot
        plt.subplot(1, 2, 2)
        val_metrics = train_history.get('val_metrics', {})
        for key, values in val_metrics.items():
            if isinstance(values, list) and 'ndcg' in key.lower():
                plt.plot(values, label=key, linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Validation Metrics')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(results_dir, 'plots', 'training_curves.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Training curves plot saved to: {plot_path}")


def run_training(args):
    """Main training function."""

    print("\n" + "=" * 80)
    print("UNCERTAINTY-AWARE CAUSAL TEMPORAL GNN TRAINING")
    print("=" * 80)

    # 1. Setup
    print("\n[1/7] Setting up environment...")
    results_dir = create_results_directory(args.results_dir)
    print(f"Results will be saved to: {results_dir}")

    # 2. Download/prepare data
    print("\n[2/7] Preparing dataset...")
    data_path = download_movielens(args.dataset, args.data_dir)
    print(f"Data path: {data_path}")

    # 3. Get optimal config
    print("\n[3/7] Loading optimal configuration...")
    config = get_optimal_config(args.dataset, args.use_uncertainty)

    # Override with command line arguments
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.embedding_dim:
        config.embedding_dim = args.embedding_dim
    if args.learning_rate:
        config.learning_rate = args.learning_rate

    config.checkpoint_dir = os.path.join(results_dir, 'models')
    config.log_dir = os.path.join(results_dir, 'logs')

    print(f"Using device: {config.device}")
    print(f"Embedding dim: {config.embedding_dim}")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Use uncertainty: {args.use_uncertainty}")

    # 4. Initialize system
    print("\n[4/7] Initializing recommendation system...")
    if args.use_uncertainty:
        system = UncertaintyAwareRecommendationSystem(config)
    else:
        system = RecommendationSystem(config)

    # 5. Load and preprocess data
    print("\n[5/7] Loading and preprocessing data...")
    system.load_data(data_path)
    system.preprocess_data()
    train_data, val_data, test_data = system.split_data()

    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data) if val_data is not None else 0}")
    print(f"Test samples: {len(test_data)}")
    print(f"Users: {system.metadata['num_users']}")
    print(f"Items: {system.metadata['num_items']}")

    # Create graph
    system.create_graph()

    # Initialize model
    system.initialize_model()

    # 6. Train
    print("\n[6/7] Training model...")
    print("-" * 40)
    start_time = time.time()

    train_history = system.train()

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time/60:.2f} minutes")

    # 7. Evaluate
    print("\n[7/7] Evaluating on test set...")
    print("-" * 40)

    k_values = [5, 10, 20, 50]
    test_metrics = system.evaluate('test', k_values=k_values)

    print("\n" + "=" * 80)
    print("FINAL TEST RESULTS")
    print("=" * 80)

    for metric_name in ['precision', 'recall', 'ndcg', 'hit_ratio']:
        if metric_name in test_metrics:
            print(f"\n{metric_name.upper()}:")
            for k in k_values:
                if k in test_metrics[metric_name]:
                    print(f"  @{k}: {test_metrics[metric_name][k]:.4f}")

    # Add uncertainty metrics if available
    if args.use_uncertainty and hasattr(system, 'get_uncertainty_report'):
        print("\nUNCERTAINTY METRICS:")
        uncertainty_report = system.get_uncertainty_report()
        if 'calibration' in uncertainty_report:
            for key, value in uncertainty_report['calibration'].items():
                print(f"  {key}: {value:.4f}")
        test_metrics['uncertainty'] = uncertainty_report

    # Generate sample recommendations
    print("\nGenerating sample recommendations...")
    sample_recommendations = {}
    sample_users = list(system.user_index_to_id.keys())[:10]

    for user_idx in sample_users:
        user_id = system.user_index_to_id[user_idx]
        try:
            if args.use_uncertainty:
                recs = system.generate_recommendations_with_uncertainty(user_id, top_k=10)
                sample_recommendations[str(user_id)] = recs
            else:
                items, scores = system.generate_recommendations(user_id, top_k=10)
                sample_recommendations[str(user_id)] = [
                    {'item_id': str(item), 'score': float(score)}
                    for item, score in zip(items, scores)
                ]
        except Exception as e:
            print(f"Warning: Could not generate recommendations for user {user_id}: {e}")

    # Save all results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    # Add training time to metrics
    test_metrics['training_time_minutes'] = training_time / 60
    test_metrics['dataset'] = args.dataset
    test_metrics['num_users'] = system.metadata['num_users']
    test_metrics['num_items'] = system.metadata['num_items']
    test_metrics['num_interactions'] = len(train_data) + len(val_data) + len(test_data)

    summary_path = save_results(
        results_dir,
        test_metrics,
        config,
        train_history,
        sample_recommendations
    )

    # Plot training curves
    plot_training_curves(results_dir, train_history)

    # Generate publication-ready outputs (if uncertainty enabled)
    pub_metrics = generate_publication_outputs(results_dir, system, args)
    if pub_metrics:
        print(f"Publication figures saved to: {os.path.join(results_dir, 'publication')}")

    # Save model
    model_path = os.path.join(results_dir, 'models', 'final_model.pt')
    system.save_model(model_path)
    print(f"Model saved to: {model_path}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nAll results saved to: {results_dir}")
    print(f"\nKey files:")
    print(f"  - Summary: {summary_path}")
    print(f"  - Metrics: {os.path.join(results_dir, 'metrics', 'final_metrics.json')}")
    print(f"  - Model: {model_path}")
    print(f"  - Training curves: {os.path.join(results_dir, 'plots', 'training_curves.png')}")
    if args.use_uncertainty and pub_metrics:
        print(f"  - Publication figures: {os.path.join(results_dir, 'publication')}")
        print(f"  - LaTeX table: {os.path.join(results_dir, 'publication', 'metrics_table.tex')}")

    return test_metrics, results_dir


def main():
    parser = argparse.ArgumentParser(
        description='Train Uncertainty-Aware Causal Temporal GNN on MovieLens'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='ml-100k',
        choices=['ml-100k', 'ml-1m', 'ml-10m'],
        help='MovieLens dataset to use'
    )

    parser.add_argument(
        '--use_uncertainty',
        action='store_true',
        help='Use uncertainty-aware model (novel contribution)'
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data',
        help='Directory for dataset storage'
    )

    parser.add_argument(
        '--results_dir',
        type=str,
        default='./results',
        help='Directory for results storage'
    )

    # Optional hyperparameter overrides
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--embedding_dim', type=int, help='Embedding dimension')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')

    args = parser.parse_args()

    # Run training
    metrics, results_dir = run_training(args)

    return metrics


if __name__ == '__main__':
    main()
