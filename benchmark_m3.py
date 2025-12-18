"""
M3 Benchmark Suite for Causal GNN Recommendation System.

This script benchmarks the Causal Temporal GNN against standard baselines
on the MovieLens 100K dataset, optimized for Apple M3 with 8GB memory.

Usage:
    python benchmark_m3.py
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from causal_gnn.config import Config
from causal_gnn.training import RecommendationSystem
from causal_gnn.baselines import PopularItems, BPR, NCF, LightGCN
from causal_gnn.utils.benchmark_utils import (
    get_memory_usage,
    save_metrics_json,
    save_training_log_csv,
    generate_benchmark_report,
    plot_training_curves,
    plot_metrics_comparison,
    plot_memory_usage,
    load_movielens_100k,
    create_comparison_table
)

warnings.filterwarnings('ignore')


def compute_metrics(predictions_dict, test_data, k=10):
    """
    Compute evaluation metrics.
    
    Args:
        predictions_dict: Dict mapping user_idx -> list of (item_idx, score)
        test_data: DataFrame with ground truth interactions
        k: Top-k for metrics
        
    Returns:
        Dict of metrics
    """
    # Create ground truth dict
    ground_truth = {}
    for _, row in test_data.iterrows():
        user_idx = int(row['user_idx'])
        item_idx = int(row['item_idx'])
        if user_idx not in ground_truth:
            ground_truth[user_idx] = set()
        ground_truth[user_idx].add(item_idx)
    
    # Compute metrics
    precision_list = []
    recall_list = []
    ndcg_list = []
    mrr_list = []
    hit_list = []
    
    for user_idx, pred_items in predictions_dict.items():
        if user_idx not in ground_truth:
            continue
        
        true_items = ground_truth[user_idx]
        pred_items_top_k = [item for item, score in pred_items[:k]]
        
        # Hits
        hits = set(pred_items_top_k) & true_items
        num_hits = len(hits)
        
        # Precision
        precision = num_hits / k if k > 0 else 0
        precision_list.append(precision)
        
        # Recall
        recall = num_hits / len(true_items) if len(true_items) > 0 else 0
        recall_list.append(recall)
        
        # Hit Rate
        hit_list.append(1.0 if num_hits > 0 else 0.0)
        
        # NDCG
        dcg = 0
        idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(true_items), k))])
        for i, item in enumerate(pred_items_top_k):
            if item in true_items:
                dcg += 1.0 / np.log2(i + 2)
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_list.append(ndcg)
        
        # MRR
        mrr = 0
        for i, item in enumerate(pred_items_top_k):
            if item in true_items:
                mrr = 1.0 / (i + 1)
                break
        mrr_list.append(mrr)
    
    return {
        'precision@10': np.mean(precision_list),
        'recall@10': np.mean(recall_list),
        'ndcg@10': np.mean(ndcg_list),
        'mrr': np.mean(mrr_list),
        'hit_rate@10': np.mean(hit_list),
    }


def benchmark_popular(train_data, test_data, num_users, num_items):
    """Benchmark Popular Items baseline."""
    print("\n" + "="*80)
    print("BENCHMARKING: Popular Items")
    print("="*80)
    
    start_time = time.time()
    start_memory = get_memory_usage()
    
    model = PopularItems(num_users, num_items)
    model.fit(train_data)
    
    # Get training users for exclusion
    train_user_items = {}
    for _, row in train_data.iterrows():
        user_idx = int(row['user_idx'])
        item_idx = int(row['item_idx'])
        if user_idx not in train_user_items:
            train_user_items[user_idx] = set()
        train_user_items[user_idx].add(item_idx)
    
    # Predict
    test_users = test_data['user_idx'].unique()
    predictions = model.predict_batch(test_users, top_k=10, exclude_items_dict=train_user_items)
    
    # Evaluate
    metrics = compute_metrics(predictions, test_data, k=10)
    
    training_time = time.time() - start_time
    peak_memory = get_memory_usage() - start_memory
    
    print(f"Training Time: {training_time:.2f}s")
    print(f"Memory Usage: {peak_memory:.2f} MB")
    print(f"Metrics: {metrics}")
    
    return {
        'metrics': metrics,
        'training_time': training_time,
        'memory_mb': peak_memory,
        'history': {'loss': []}
    }


def benchmark_bpr(train_data, test_data, num_users, num_items, device, num_epochs=20):
    """Benchmark BPR baseline."""
    print("\n" + "="*80)
    print("BENCHMARKING: BPR")
    print("="*80)
    
    start_time = time.time()
    start_memory = get_memory_usage()
    
    model = BPR(num_users, num_items, embedding_dim=32, learning_rate=0.01, reg_lambda=0.01)
    model.fit(train_data, num_epochs=num_epochs, batch_size=256, device=device)
    
    # Get training users for exclusion
    train_user_items = {}
    for _, row in train_data.iterrows():
        user_idx = int(row['user_idx'])
        item_idx = int(row['item_idx'])
        if user_idx not in train_user_items:
            train_user_items[user_idx] = set()
        train_user_items[user_idx].add(item_idx)
    
    # Predict
    test_users = test_data['user_idx'].unique()
    predictions = model.predict_batch(test_users, top_k=10, exclude_items_dict=train_user_items, device=device)
    
    # Evaluate
    metrics = compute_metrics(predictions, test_data, k=10)
    
    training_time = time.time() - start_time
    peak_memory = get_memory_usage() - start_memory
    
    print(f"Training Time: {training_time:.2f}s")
    print(f"Memory Usage: {peak_memory:.2f} MB")
    print(f"Metrics: {metrics}")
    
    return {
        'metrics': metrics,
        'training_time': training_time,
        'memory_mb': peak_memory,
        'history': {'loss': []}
    }


def benchmark_ncf(train_data, test_data, num_users, num_items, device, num_epochs=20):
    """Benchmark NCF baseline."""
    print("\n" + "="*80)
    print("BENCHMARKING: NCF")
    print("="*80)
    
    start_time = time.time()
    start_memory = get_memory_usage()
    
    model = NCF(num_users, num_items, embedding_dim=32, hidden_dims=[64, 32, 16], learning_rate=0.001)
    model.fit(train_data, num_epochs=num_epochs, batch_size=256, device=device)
    
    # Get training users for exclusion
    train_user_items = {}
    for _, row in train_data.iterrows():
        user_idx = int(row['user_idx'])
        item_idx = int(row['item_idx'])
        if user_idx not in train_user_items:
            train_user_items[user_idx] = set()
        train_user_items[user_idx].add(item_idx)
    
    # Predict
    test_users = test_data['user_idx'].unique()
    predictions = model.predict_batch(test_users, top_k=10, exclude_items_dict=train_user_items, device=device)
    
    # Evaluate
    metrics = compute_metrics(predictions, test_data, k=10)
    
    training_time = time.time() - start_time
    peak_memory = get_memory_usage() - start_memory
    
    print(f"Training Time: {training_time:.2f}s")
    print(f"Memory Usage: {peak_memory:.2f} MB")
    print(f"Metrics: {metrics}")
    
    return {
        'metrics': metrics,
        'training_time': training_time,
        'memory_mb': peak_memory,
        'history': {'loss': []}
    }


def benchmark_lightgcn(train_data, test_data, num_users, num_items, device, num_epochs=20):
    """Benchmark LightGCN baseline."""
    print("\n" + "="*80)
    print("BENCHMARKING: LightGCN")
    print("="*80)
    
    start_time = time.time()
    start_memory = get_memory_usage()
    
    model = LightGCN(num_users, num_items, embedding_dim=32, num_layers=2, learning_rate=0.001, reg_lambda=1e-4)
    model.fit(train_data, num_epochs=num_epochs, batch_size=256, device=device)
    
    # Get training users for exclusion
    train_user_items = {}
    for _, row in train_data.iterrows():
        user_idx = int(row['user_idx'])
        item_idx = int(row['item_idx'])
        if user_idx not in train_user_items:
            train_user_items[user_idx] = set()
        train_user_items[user_idx].add(item_idx)
    
    # Predict
    test_users = test_data['user_idx'].unique()
    predictions = model.predict_batch(test_users, top_k=10, exclude_items_dict=train_user_items, device=device)
    
    # Evaluate
    metrics = compute_metrics(predictions, test_data, k=10)
    
    training_time = time.time() - start_time
    peak_memory = get_memory_usage() - start_memory
    
    print(f"Training Time: {training_time:.2f}s")
    print(f"Memory Usage: {peak_memory:.2f} MB")
    print(f"Metrics: {metrics}")
    
    return {
        'metrics': metrics,
        'training_time': training_time,
        'memory_mb': peak_memory,
        'history': {'loss': []}
    }


def benchmark_causal_gnn(train_data, val_data, test_data, num_users, num_items, device, num_epochs=20):
    """Benchmark Causal Temporal GNN (our model)."""
    print("\n" + "="*80)
    print("BENCHMARKING: Causal Temporal GNN (Ours)")
    print("="*80)

    start_time = time.time()
    start_memory = get_memory_usage()

    # M3-optimized config
    config = Config(
        device=device,
        embedding_dim=32,
        num_layers=2,
        batch_size=256,
        num_epochs=num_epochs,
        learning_rate=0.001,
        causal_method='simple',  # Use simple method for faster benchmark
        use_amp=False,
        use_gradient_checkpointing=False,
        use_neighbor_sampling=False,
        data_dir='./data/movielens_100k',
        output_dir='./benchmark_results/movielens_100k',
        checkpoint_dir='./benchmark_results/checkpoints',
        log_dir='./benchmark_results/logs',
        early_stopping_patience=3,  # Faster early stopping for benchmark
        save_every_n_epochs=num_epochs + 1,  # Don't save during benchmark
    )

    try:
        # Initialize the recommendation system
        rec_system = RecommendationSystem(config)

        # Prepare data in expected format - create combined data with proper schema
        all_data = pd.concat([train_data, val_data, test_data], ignore_index=True)

        # Since data is already split, we'll work directly with indices
        rec_system.data['preprocessed_data'] = all_data
        rec_system.data['schema'] = {
            'user_columns': ['user_idx'],
            'item_columns': ['item_idx'],
            'interaction_columns': ['rating'] if 'rating' in all_data.columns else [],
            'temporal_columns': ['timestamp'] if 'timestamp' in all_data.columns else [],
            'text_columns': [],
            'numeric_columns': [],
            'categorical_columns': [],
            'image_columns': [],
            'context_columns': [],
        }

        # Set metadata
        rec_system.metadata = {
            'num_users': num_users,
            'num_items': num_items,
            'num_interactions': len(all_data),
        }

        # Create ID mappings (identity since already indexed)
        rec_system.user_index_to_id = {i: i for i in range(num_users)}
        rec_system.item_index_to_id = {i: i for i in range(num_items)}
        rec_system.data['user_id_map'] = {i: i for i in range(num_users)}
        rec_system.data['item_id_map'] = {i: i for i in range(num_items)}

        # Set train/val/test data
        rec_system.data['train_data'] = train_data.copy()
        rec_system.data['val_data'] = val_data.copy()
        rec_system.data['test_data'] = test_data.copy()

        # Build user interactions from training data
        from collections import defaultdict
        rec_system.user_interactions = defaultdict(set)
        for _, row in train_data.iterrows():
            rec_system.user_interactions[int(row['user_idx'])].add(int(row['item_idx']))

        # Create graph
        rec_system.create_graph()

        # Initialize model
        rec_system.initialize_model()

        # Train the model
        train_history = rec_system.train()

        # Evaluate on test set
        test_metrics = rec_system.evaluate('test', k_values=[10])

        training_time = time.time() - start_time
        peak_memory = get_memory_usage() - start_memory

        # Format metrics
        metrics = {
            'precision@10': test_metrics['precision'][10],
            'recall@10': test_metrics['recall'][10],
            'ndcg@10': test_metrics['ndcg'][10],
            'mrr': test_metrics.get('mrr', {}).get(10, 0.0),
            'hit_rate@10': test_metrics['hit_ratio'][10],
        }

        print(f"Training Time: {training_time:.2f}s")
        print(f"Memory Usage: {peak_memory:.2f} MB")
        print(f"Metrics: {metrics}")

        return {
            'metrics': metrics,
            'training_time': training_time,
            'memory_mb': peak_memory,
            'history': train_history
        }

    except Exception as e:
        print(f"Error during Causal GNN benchmark: {e}")
        import traceback
        traceback.print_exc()

        training_time = time.time() - start_time
        peak_memory = get_memory_usage() - start_memory

        # Return fallback metrics indicating failure
        return {
            'metrics': {
                'precision@10': 0.0,
                'recall@10': 0.0,
                'ndcg@10': 0.0,
                'mrr': 0.0,
                'hit_rate@10': 0.0,
            },
            'training_time': training_time,
            'memory_mb': peak_memory,
            'history': {'loss': []},
            'error': str(e)
        }


def main():
    """Main benchmark execution."""
    print("=" * 100)
    print(" " * 30 + "M3 BENCHMARK SUITE")
    print(" " * 20 + "Causal Temporal GNN vs Baselines")
    print("=" * 100)
    
    # Setup
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    print(f"Initial Memory: {get_memory_usage():.2f} MB")
    
    # Load data
    print("\n" + "="*80)
    print("LOADING MOVIELENS 100K DATASET")
    print("="*80)
    
    data = load_movielens_100k()
    
    # Use only ratings >= 4 as positive feedback (implicit feedback)
    data = data[data['rating'] >= 4].copy()
    
    # CRITICAL FIX: Remap user and item indices to be contiguous (0 to n-1)
    # This is necessary after filtering, as some users/items may be removed
    unique_users = sorted(data['user_idx'].unique())
    unique_items = sorted(data['item_idx'].unique())
    
    user_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_users)}
    item_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_items)}
    
    data['user_idx'] = data['user_idx'].map(user_id_map)
    data['item_idx'] = data['item_idx'].map(item_id_map)
    
    num_users = data['user_idx'].max() + 1  # Now contiguous, so max + 1 = count
    num_items = data['item_idx'].max() + 1
    num_interactions = len(data)
    sparsity = 100 * (1 - num_interactions / (num_users * num_items))
    
    print(f"\nFiltered Dataset (ratings >= 4):")
    print(f"  Users: {num_users}")
    print(f"  Items: {num_items}")
    print(f"  Interactions: {num_interactions}")
    print(f"  Sparsity: {sparsity:.2f}%")
    
    # Split data
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)
    
    print(f"\nData Split:")
    print(f"  Train: {len(train_data)} interactions")
    print(f"  Val: {len(val_data)} interactions")
    print(f"  Test: {len(test_data)} interactions")
    
    # Benchmark configuration
    num_epochs = 10  # Reduced for faster benchmarking on M3
    
    # Initialize results
    results = {
        'dataset': 'MovieLens-100K',
        'hardware': 'Apple M3 8GB',
        'num_users': num_users,
        'num_items': num_items,
        'num_interactions': num_interactions,
        'sparsity': sparsity,
        'models': {},
        'timestamp': time.time()
    }
    
    # Run benchmarks
    print("\n" + "="*100)
    print(" " * 35 + "RUNNING BENCHMARKS")
    print("="*100)
    
    # 1. Popular Items
    try:
        results['models']['Popular'] = benchmark_popular(train_data, test_data, num_users, num_items)
    except Exception as e:
        print(f"Error in Popular benchmark: {e}")
    
    # 2. BPR
    try:
        results['models']['BPR'] = benchmark_bpr(train_data, test_data, num_users, num_items, device, num_epochs)
    except Exception as e:
        print(f"Error in BPR benchmark: {e}")
    
    # 3. NCF
    try:
        results['models']['NCF'] = benchmark_ncf(train_data, test_data, num_users, num_items, device, num_epochs)
    except Exception as e:
        print(f"Error in NCF benchmark: {e}")
    
    # 4. LightGCN
    try:
        results['models']['LightGCN'] = benchmark_lightgcn(train_data, test_data, num_users, num_items, device, num_epochs)
    except Exception as e:
        print(f"Error in LightGCN benchmark: {e}")
    
    # 5. Causal Temporal GNN
    try:
        results['models']['CausalGNN'] = benchmark_causal_gnn(train_data, val_data, test_data, num_users, num_items, device, num_epochs)
    except Exception as e:
        print(f"Error in CausalGNN benchmark: {e}")
    
    # Find best model
    best_model = None
    best_ndcg = 0
    for model_name, model_results in results['models'].items():
        ndcg = model_results['metrics'].get('ndcg@10', 0)
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_model = model_name
    
    results['best_model'] = best_model
    results['peak_memory_mb'] = get_memory_usage()
    
    # Save results
    output_dir = './benchmark_results/movielens_100k'
    os.makedirs(output_dir, exist_ok=True)
    
    save_metrics_json(results, os.path.join(output_dir, 'metrics.json'))
    generate_benchmark_report(results, os.path.join(output_dir, 'benchmark_report.txt'))
    
    # Plot results
    plots_dir = os.path.join(output_dir, 'plots')
    plot_metrics_comparison(results, plots_dir)
    
    # Display comparison table
    create_comparison_table(results)
    
    print("\n" + "="*100)
    print(" " * 30 + "BENCHMARK COMPLETE!")
    print("="*100)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - metrics.json")
    print(f"  - benchmark_report.txt")
    print(f"  - plots/metrics_comparison.png")
    print("\n" + "="*100)


if __name__ == '__main__':
    main()

