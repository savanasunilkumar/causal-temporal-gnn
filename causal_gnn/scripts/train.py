"""Training script for the Enhanced UACT-GNN recommendation system."""

import os
import sys
import argparse
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from causal_gnn.config import Config
from causal_gnn.training.trainer import RecommendationSystem


def train_model(config, data_path):
    """Train the Enhanced UACT-GNN model."""
    print("="*80)
    print("Enhanced Universal Adaptive Causal Temporal GNN (UACT-GNN)")
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
    
    # Train model
    print("\n4. Training the model...")
    train_history = recommendation_system.train()
    
    # Evaluate on test set
    print("\n5. Evaluating the model on test set...")
    test_metrics = recommendation_system.evaluate('test', k_values=[5, 10, 20])
    
    print("\n" + "="*80)
    print("Test Metrics:")
    print("="*80)
    for metric in test_metrics:
        for k, value in test_metrics[metric].items():
            print(f"  {metric.upper()}@{k}: {value:.4f}")
    
    # Generate sample recommendations
    print("\n6. Generating recommendations for sample users...")
    sample_user_indices = np.random.choice(
        list(recommendation_system.user_index_to_id.keys()),
        size=min(5, recommendation_system.metadata['num_users']),
        replace=False
    )
    sample_user_ids = [recommendation_system.user_index_to_id[idx] for idx in sample_user_indices]
    
    for user_id in sample_user_ids:
        print(f"\nRecommendations for User {user_id}:")
        rec_items, rec_scores = recommendation_system.generate_recommendations(user_id, top_k=5)
        for i, (item_id, score) in enumerate(zip(rec_items, rec_scores)):
            print(f"  {i+1}. Item {item_id} (Score: {score:.4f})")
    
    # Test cold start
    print("\n7. Testing cold start recommendations...")
    cold_start_user = {
        'age': 25,
        'gender': 'F',
        'occupation': 'student',
        'genre': 'Sci-Fi'
    }
    
    cold_start_rec_items, cold_start_rec_scores = recommendation_system.generate_cold_start_recommendations(
        cold_start_user, top_k=5
    )
    
    print("\nCold Start Recommendations:")
    for i, (item_id, score) in enumerate(zip(cold_start_rec_items, cold_start_rec_scores)):
        print(f"  {i+1}. Item {item_id} (Score: {score:.4f})")
    
    # Save final model
    model_path = os.path.join(config.output_dir, 'final_model.pt')
    recommendation_system.save_model(model_path)
    print(f"\nFinal model saved to {model_path}")
    
    print("\n" + "="*80)
    print("Training completed successfully!")
    print("="*80)
    
    return {
        'recommendation_system': recommendation_system,
        'test_metrics': test_metrics,
        'train_history': train_history
    }


def main():
    parser = argparse.ArgumentParser(description='Train UACT-GNN recommendation system')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='./data/interactions.csv',
                        help='Path to the input data file')
    
    # Model arguments
    parser.add_argument('--embedding_dim', type=int, default=64,
                        help='Dimension of embeddings')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('--time_steps', type=int, default=10,
                        help='Number of time steps')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--causal_strength', type=float, default=0.5,
                        help='Strength of causal connections')
    parser.add_argument('--causal_method', type=str, default='advanced',
                        choices=['simple', 'advanced'],
                        help='Causal discovery method')
    
    # Training arguments
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--neg_samples', type=int, default=1,
                        help='Number of negative samples')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay for regularization')
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                        help='Patience for early stopping')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use mixed precision training')
    
    # Logging arguments
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases logging')
    parser.add_argument('--use_tensorboard', action='store_true',
                        help='Use TensorBoard logging')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Log directory')
    
    args = parser.parse_args()
    
    # Create config from arguments
    config = Config(
        # Model
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        time_steps=args.time_steps,
        dropout=args.dropout,
        causal_strength=args.causal_strength,
        causal_method=args.causal_method,
        
        # Training
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        neg_samples=args.neg_samples,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
        
        # System
        device=args.device,
        seed=args.seed,
        use_amp=args.use_amp,
        
        # Logging
        use_wandb=args.use_wandb,
        use_tensorboard=args.use_tensorboard,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )
    
    # Train model
    results = train_model(config, args.data_path)
    
    return results


if __name__ == '__main__':
    main()

