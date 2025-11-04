"""Evaluation script for trained UACT-GNN models."""

import os
import sys
import argparse
import torch

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from causal_gnn.config import Config
from causal_gnn.training.trainer import RecommendationSystem


def evaluate_model(model_path, data_path, config):
    """Evaluate a trained UACT-GNN model."""
    print("="*80)
    print("Evaluating Enhanced UACT-GNN Model")
    print("="*80)
    
    # Initialize recommendation system
    recommendation_system = RecommendationSystem(config)
    
    # Load and preprocess data
    print("\n1. Loading and processing data...")
    recommendation_system.load_data(data_path)
    recommendation_system.preprocess_data()
    recommendation_system.split_data()
    
    # Create graph
    print("\n2. Creating temporal interaction graph...")
    recommendation_system.create_graph()
    
    # Initialize model
    print("\n3. Initializing the Enhanced UACT-GNN model...")
    recommendation_system.initialize_model()
    
    # Load trained model
    print(f"\n4. Loading trained model from {model_path}...")
    recommendation_system.load_model(model_path)
    
    # Evaluate on test set
    print("\n5. Evaluating on test set...")
    test_metrics = recommendation_system.evaluate('test', k_values=[5, 10, 20, 50])
    
    print("\n" + "="*80)
    print("Test Set Evaluation Results:")
    print("="*80)
    for metric in test_metrics:
        print(f"\n{metric.upper()}:")
        for k, value in test_metrics[metric].items():
            print(f"  @{k:2d}: {value:.4f}")
    
    # Compute additional metrics
    print("\n" + "="*80)
    print("Additional Metrics:")
    print("="*80)
    
    # You can add additional evaluations here, such as:
    # - Diversity of recommendations
    # - Coverage of item catalog
    # - Performance on different user/item segments
    
    print("\nEvaluation complete!")
    
    return test_metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained UACT-GNN model')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--data_path', type=str, default='./data/interactions.csv',
                        help='Path to the input data file')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for evaluation')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Create config
    config = Config(
        device=args.device,
        output_dir=args.output_dir
    )
    
    # Evaluate model
    metrics = evaluate_model(args.model_path, args.data_path, config)
    
    return metrics


if __name__ == '__main__':
    main()

