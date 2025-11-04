"""Preprocessing script for causal graph computation and feature extraction."""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import pickle
import torch

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from causal_gnn.config import Config
from causal_gnn.data.processor import DataProcessor
from causal_gnn.causal.discovery import CausalGraphConstructor


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
    
    user_ages = np.random.randint(18, 65, n_users)
    user_genders = np.random.choice(['M', 'F'], n_users)
    user_occupations = np.random.choice(['student', 'engineer', 'teacher', 'doctor'], n_users)
    
    item_genres = np.random.choice(['Action', 'Comedy', 'Drama', 'Sci-Fi'], n_items)
    item_years = np.random.randint(1980, 2023, n_items)
    
    interactions = pd.DataFrame({
        'userId': user_ids,
        'movieId': item_ids,
        'rating': ratings,
        'timestamp': timestamps
    })
    
    users = pd.DataFrame({
        'userId': range(1, n_users + 1),
        'age': user_ages,
        'gender': user_genders,
        'occupation': user_occupations
    })
    
    movies = pd.DataFrame({
        'movieId': range(1, n_items + 1),
        'title': [f'Movie {i}' for i in range(1, n_items + 1)],
        'genre': item_genres,
        'year': item_years
    })
    
    interactions.to_csv(os.path.join(data_dir, 'interactions.csv'), index=False)
    users.to_csv(os.path.join(data_dir, 'users.csv'), index=False)
    movies.to_csv(os.path.join(data_dir, 'movies.csv'), index=False)
    
    print(f"Sample dataset created at {data_dir}")
    return os.path.join(data_dir, 'interactions.csv')


def preprocess_data(data_path, config):
    """Preprocess data and compute causal graphs."""
    print("Starting preprocessing...")
    
    # Initialize data processor
    data_processor = DataProcessor(config)
    
    # Load and process data
    print(f"Loading data from {data_path}")
    processed_data, schema = data_processor.process_data(data_path)
    
    # Save processed data
    output_path = os.path.join(config.output_dir, 'processed_data.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump({
            'processed_data': processed_data,
            'schema': schema,
            'metadata': data_processor.metadata
        }, f)
    print(f"Saved processed data to {output_path}")
    
    # Compute causal graph if enabled
    if config.precompute_causal_graph:
        print("Computing causal graph...")
        causal_constructor = CausalGraphConstructor(config)
        
        # This is a placeholder - in a real implementation, you would:
        # 1. Extract interaction data and timestamps
        # 2. Build node features
        # 3. Compute causal graph
        # 4. Save the causal graph
        
        causal_output_path = os.path.join(config.causal_graph_cache_dir, 'causal_graph.pkl')
        os.makedirs(config.causal_graph_cache_dir, exist_ok=True)
        
        print(f"Causal graph computation would be saved to {causal_output_path}")
        print("Note: Full causal graph computation is done during training in the current implementation")
    
    print("Preprocessing complete!")
    return processed_data, schema


def main():
    parser = argparse.ArgumentParser(description='Preprocess data for UACT-GNN')
    parser.add_argument('--data_path', type=str, default='./data/interactions.csv',
                        help='Path to the input data file')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save processed data')
    parser.add_argument('--create_sample', action='store_true',
                        help='Create a sample dataset for testing')
    parser.add_argument('--precompute_causal', action='store_true', default=True,
                        help='Precompute causal graphs')
    
    args = parser.parse_args()
    
    # Create sample dataset if requested
    if args.create_sample:
        args.data_path = create_sample_dataset()
    
    # Create config
    config = Config(
        output_dir=args.output_dir,
        precompute_causal_graph=args.precompute_causal
    )
    
    # Preprocess data
    preprocess_data(args.data_path, config)


if __name__ == '__main__':
    main()

