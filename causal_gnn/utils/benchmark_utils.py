"""Utility functions for benchmarking recommendation models."""

import os
import json
import time
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def save_metrics_json(results, path):
    """Save benchmark results to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Convert numpy types to native Python types
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj
    
    results = convert_types(results)
    
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved metrics to {path}")


def save_training_log_csv(history, path):
    """Save training history to CSV file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    df = pd.DataFrame(history)
    df.to_csv(path, index=False)
    
    print(f"Saved training log to {path}")


def generate_benchmark_report(results, path):
    """Generate human-readable benchmark report."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(" " * 20 + "BENCHMARK REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Dataset info
        f.write("DATASET INFORMATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Dataset: {results.get('dataset', 'N/A')}\n")
        f.write(f"Users: {results.get('num_users', 'N/A'):,}\n")
        f.write(f"Items: {results.get('num_items', 'N/A'):,}\n")
        f.write(f"Interactions: {results.get('num_interactions', 'N/A'):,}\n")
        f.write(f"Sparsity: {results.get('sparsity', 0):.4f}%\n\n")
        
        # Hardware info
        f.write("HARDWARE INFORMATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Device: {results.get('hardware', 'N/A')}\n")
        f.write(f"Peak Memory: {results.get('peak_memory_mb', 0):.2f} MB\n\n")
        
        # Model comparisons
        f.write("MODEL COMPARISON\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Model':<20} {'Precision@10':<15} {'Recall@10':<15} {'NDCG@10':<15} {'Training Time':<15}\n")
        f.write("-" * 80 + "\n")
        
        for model_name, model_results in results.get('models', {}).items():
            metrics = model_results.get('metrics', {})
            training_time = model_results.get('training_time', 0)
            
            f.write(
                f"{model_name:<20} "
                f"{metrics.get('precision@10', 0):<15.4f} "
                f"{metrics.get('recall@10', 0):<15.4f} "
                f"{metrics.get('ndcg@10', 0):<15.4f} "
                f"{training_time/60:<15.2f}min\n"
            )
        
        f.write("\n")
        
        # Best model
        best_model = results.get('best_model', 'N/A')
        f.write(f"Best Model: {best_model}\n\n")
        
        # Timestamp
        f.write("=" * 80 + "\n")
        f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n")
    
    print(f"Saved benchmark report to {path}")


def plot_training_curves(history_dict, save_dir):
    """
    Plot training curves for all models.
    
    Args:
        history_dict: Dict mapping model_name -> training history
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot loss curves
    plt.figure(figsize=(10, 6))
    for model_name, history in history_dict.items():
        if 'loss' in history and len(history['loss']) > 0:
            epochs = range(1, len(history['loss']) + 1)
            plt.plot(epochs, history['loss'], marker='o', label=model_name)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'training_loss.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training loss plot")


def plot_metrics_comparison(results, save_dir):
    """Plot metrics comparison across models."""
    os.makedirs(save_dir, exist_ok=True)
    
    models = list(results.get('models', {}).keys())
    
    metrics_to_plot = ['precision@10', 'recall@10', 'ndcg@10', 'mrr']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        values = []
        for model in models:
            model_metrics = results['models'][model].get('metrics', {})
            values.append(model_metrics.get(metric, 0))
        
        bars = ax.bar(range(len(models)), values, alpha=0.7)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} Comparison')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved metrics comparison plot")


def plot_memory_usage(memory_history, save_dir):
    """Plot memory usage over time."""
    os.makedirs(save_dir, exist_ok=True)
    
    if not memory_history:
        return
    
    plt.figure(figsize=(10, 6))
    timestamps = [item['timestamp'] for item in memory_history]
    memory_mb = [item['memory_mb'] for item in memory_history]
    
    # Convert timestamps to relative seconds
    start_time = timestamps[0]
    relative_times = [(t - start_time) for t in timestamps]
    
    plt.plot(relative_times, memory_mb, marker='o', markersize=3)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage During Training')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'memory_usage.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved memory usage plot")


def download_movielens_100k(data_dir='./data/movielens_100k'):
    """
    Download and extract MovieLens 100K dataset.
    
    Returns:
        Path to the ratings file
    """
    import urllib.request
    import zipfile
    
    os.makedirs(data_dir, exist_ok=True)
    
    ratings_file = os.path.join(data_dir, 'u.data')
    
    # Check if already downloaded
    if os.path.exists(ratings_file):
        print(f"MovieLens 100K already exists at {data_dir}")
        return ratings_file
    
    # Download
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    zip_path = os.path.join(data_dir, 'ml-100k.zip')
    
    print(f"Downloading MovieLens 100K from {url}...")
    urllib.request.urlretrieve(url, zip_path)
    
    # Extract
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    # Move files to data_dir
    import shutil
    ml_dir = os.path.join(data_dir, 'ml-100k')
    for file in os.listdir(ml_dir):
        shutil.move(os.path.join(ml_dir, file), os.path.join(data_dir, file))
    
    # Clean up
    os.remove(zip_path)
    os.rmdir(ml_dir)
    
    print(f"MovieLens 100K downloaded to {data_dir}")
    return ratings_file


def load_movielens_100k(data_dir='./data/movielens_100k'):
    """
    Load MovieLens 100K dataset.
    
    Returns:
        DataFrame with columns: user_idx, item_idx, rating, timestamp
    """
    ratings_file = os.path.join(data_dir, 'u.data')
    
    if not os.path.exists(ratings_file):
        ratings_file = download_movielens_100k(data_dir)
    
    # Load data (tab-separated: user_id, item_id, rating, timestamp)
    df = pd.read_csv(ratings_file, sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
    
    # Convert to 0-indexed
    df['user_idx'] = df['user_id'] - 1
    df['item_idx'] = df['item_id'] - 1
    
    print(f"Loaded MovieLens 100K:")
    print(f"  Users: {df['user_idx'].nunique()}")
    print(f"  Items: {df['item_idx'].nunique()}")
    print(f"  Interactions: {len(df)}")
    
    return df


def create_comparison_table(results):
    """Create a formatted comparison table for console output."""
    models = results.get('models', {})
    
    print("\n" + "=" * 100)
    print(" " * 35 + "MODEL COMPARISON")
    print("=" * 100)
    
    # Header
    header = f"{'Model':<20} {'P@10':<10} {'R@10':<10} {'NDCG@10':<10} {'MRR':<10} {'Hit@10':<10} {'Time(min)':<12}"
    print(header)
    print("-" * 100)
    
    # Rows
    for model_name, model_results in models.items():
        metrics = model_results.get('metrics', {})
        training_time = model_results.get('training_time', 0) / 60  # Convert to minutes
        
        row = (
            f"{model_name:<20} "
            f"{metrics.get('precision@10', 0):<10.4f} "
            f"{metrics.get('recall@10', 0):<10.4f} "
            f"{metrics.get('ndcg@10', 0):<10.4f} "
            f"{metrics.get('mrr', 0):<10.4f} "
            f"{metrics.get('hit_rate@10', 0):<10.4f} "
            f"{training_time:<12.2f}"
        )
        print(row)
    
    print("=" * 100)
    
    # Find best model
    best_model = None
    best_ndcg = 0
    for model_name, model_results in models.items():
        ndcg = model_results.get('metrics', {}).get('ndcg@10', 0)
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_model = model_name
    
    if best_model:
        print(f"\n🏆 Best Model: {best_model} (NDCG@10: {best_ndcg:.4f})")
    
    print()

