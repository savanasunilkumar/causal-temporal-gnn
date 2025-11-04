"""Example usage of the Enhanced UACT-GNN recommendation system."""

import os
import numpy as np
import pandas as pd

from causal_gnn.config import Config
from causal_gnn.training.trainer import RecommendationSystem


def create_sample_dataset(data_dir='./data'):
    """Create a sample dataset for demonstration."""
    os.makedirs(data_dir, exist_ok=True)
    
    n_users = 500
    n_items = 200
    n_interactions = 5000
    
    user_ids = np.random.randint(1, n_users + 1, n_interactions)
    item_ids = np.random.randint(1, n_items + 1, n_interactions)
    ratings = np.random.randint(1, 6, n_interactions)
    timestamps = np.random.randint(1000000000, 1600000000, n_interactions)
    
    interactions = pd.DataFrame({
        'userId': user_ids,
        'movieId': item_ids,
        'rating': ratings,
        'timestamp': timestamps
    })
    
    interactions.to_csv(os.path.join(data_dir, 'interactions.csv'), index=False)
    
    print(f"Sample dataset created at {data_dir}")
    return os.path.join(data_dir, 'interactions.csv')


def run_enhanced_uact_gnn_system(data_path):
    """Run the Enhanced UACT-GNN-based recommendation system."""
    
    # Create configuration
    config = Config(
        embedding_dim=64,
        num_layers=3,
        time_steps=10,
        learning_rate=0.001,
        batch_size=1024,
        num_epochs=20,
        neg_samples=1,
        weight_decay=0.0001,
        causal_strength=0.5,
        early_stopping_patience=5,
        causal_method='advanced',  # Use advanced causal discovery
        use_amp=False,  # Set to True if you have a GPU
        use_tensorboard=False,  # Set to True for TensorBoard logging
        use_wandb=False  # Set to True for Weights & Biases logging
    )
    
    # Initialize recommendation system
    recommendation_system = RecommendationSystem(config)
    
    print("\n" + "="*80)
    print("Enhanced Universal Adaptive Causal Temporal GNN (UACT-GNN)")
    print("="*80)
    
    # Step 1: Load and process data
    print("\n1. Loading and processing data...")
    recommendation_system.load_data(data_path)
    recommendation_system.preprocess_data()
    recommendation_system.split_data()
    
    # Step 2: Create temporal interaction graph
    print("\n2. Creating temporal interaction graph...")
    recommendation_system.create_graph()
    
    # Step 3: Initialize the model
    print("\n3. Initializing the Enhanced UACT-GNN model...")
    recommendation_system.initialize_model()
    
    # Step 4: Train the model
    print("\n4. Training the model...")
    train_history = recommendation_system.train()
    
    # Step 5: Evaluate on test set
    print("\n5. Evaluating the model on test set...")
    test_metrics = recommendation_system.evaluate('test', k_values=[5, 10, 20])
    
    print("\n" + "="*80)
    print("Test Metrics:")
    print("="*80)
    for metric in test_metrics:
        for k, value in test_metrics[metric].items():
            print(f"  {metric.upper()}@{k}: {value:.4f}")
    
    # Step 6: Generate recommendations for sample users
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
    
    # Step 7: Test cold start recommendations
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
    
    print("\n" + "="*80)
    print("Enhanced UACT-GNN recommendation system completed successfully!")
    print("="*80)
    
    return {
        'recommendation_system': recommendation_system,
        'test_metrics': test_metrics,
        'train_history': train_history
    }


if __name__ == "__main__":
    # Create sample dataset
    sample_data_path = './data/interactions.csv'
    if not os.path.exists(sample_data_path):
        sample_data_path = create_sample_dataset()
    
    # Run the system
    results = run_enhanced_uact_gnn_system(sample_data_path)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("Model: Enhanced Universal Adaptive Causal Temporal GNN")
    print(f"Embedding dimension: {results['recommendation_system'].config.embedding_dim}")
    print(f"Causal Method: {results['recommendation_system'].config.causal_method}")
    print("\nTest Metrics:")
    for metric in results['test_metrics']:
        for k, value in results['test_metrics'][metric].items():
            print(f"  {metric.upper()}@{k}: {value:.4f}")

