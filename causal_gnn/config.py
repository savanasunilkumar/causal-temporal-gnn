"""Configuration module for the UACT-GNN system."""

import os
import random
import numpy as np
import torch


class Config:
    """Configuration class for the UACT-GNN system."""
    
    def __init__(self, **kwargs):
        # Data Configuration
        self.data_dir = kwargs.get('data_dir', './data')
        self.output_dir = kwargs.get('output_dir', './output')
        self.model_type = kwargs.get('model_type', 'enhanced_uact_gnn')
        
        # Model Architecture
        self.embedding_dim = kwargs.get('embedding_dim', 64)
        self.num_layers = kwargs.get('num_layers', 3)
        self.time_steps = kwargs.get('time_steps', 10)
        self.dropout = kwargs.get('dropout', 0.1)
        self.causal_strength = kwargs.get('causal_strength', 0.5)
        
        # Causal Discovery
        self.causal_method = kwargs.get('causal_method', 'advanced')  # 'simple', 'advanced'
        self.significance_level = kwargs.get('significance_level', 0.05)
        self.max_lag = kwargs.get('max_lag', 5)
        self.min_causal_strength = kwargs.get('min_causal_strength', 0.1)
        
        # Training Configuration
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.batch_size = kwargs.get('batch_size', 1024)
        self.num_epochs = kwargs.get('num_epochs', 20)
        self.neg_samples = kwargs.get('neg_samples', 1)
        self.weight_decay = kwargs.get('weight_decay', 0.0001)
        self.contrastive_weight = kwargs.get('contrastive_weight', 0.1)
        self.early_stopping_patience = kwargs.get('early_stopping_patience', 5)
        
        # Data Processing
        self.min_user_interactions = kwargs.get('min_user_interactions', 5)
        self.min_item_interactions = kwargs.get('min_item_interactions', 5)
        self.test_size = kwargs.get('test_size', 0.2)
        self.val_size = kwargs.get('val_size', 0.1)
        self.rating_threshold = kwargs.get('rating_threshold', 3.5)
        
        # System Configuration
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = kwargs.get('seed', 42)
        
        # Distributed Training Configuration
        self.distributed = kwargs.get('distributed', False)
        self.world_size = kwargs.get('world_size', 1)
        self.rank = kwargs.get('rank', 0)
        self.local_rank = kwargs.get('local_rank', 0)
        
        # Mixed Precision Training
        self.use_amp = kwargs.get('use_amp', False)
        
        # Gradient Checkpointing (50-70% memory reduction)
        self.use_gradient_checkpointing = kwargs.get('use_gradient_checkpointing', False)
        
        # Gradient Accumulation
        self.gradient_accumulation_steps = kwargs.get('gradient_accumulation_steps', 1)
        
        # Checkpointing
        self.checkpoint_dir = kwargs.get('checkpoint_dir', './checkpoints')
        self.save_every_n_epochs = kwargs.get('save_every_n_epochs', 1)
        self.keep_best_k_models = kwargs.get('keep_best_k_models', 3)
        
        # Logging
        self.use_wandb = kwargs.get('use_wandb', False)
        self.use_tensorboard = kwargs.get('use_tensorboard', False)
        self.log_dir = kwargs.get('log_dir', './logs')
        self.log_every_n_steps = kwargs.get('log_every_n_steps', 100)
        
        # Preprocessing
        self.precompute_causal_graph = kwargs.get('precompute_causal_graph', True)
        self.causal_graph_cache_dir = kwargs.get('causal_graph_cache_dir', './cache/causal_graphs')
        self.use_cached_causal_graph = kwargs.get('use_cached_causal_graph', True)
        
        # Neighbor Sampling (for large graphs)
        self.use_neighbor_sampling = kwargs.get('use_neighbor_sampling', False)
        self.num_neighbors = kwargs.get('num_neighbors', [10, 5])  # 2-hop sampling
        
        # Sparse Tensor Support
        self.use_sparse_tensors = kwargs.get('use_sparse_tensors', True)
        
        # Create necessary directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        if self.precompute_causal_graph:
            os.makedirs(self.causal_graph_cache_dir, exist_ok=True)
        
        # Set random seeds for reproducibility
        self._set_seed()
    
    def _set_seed(self):
        """Set random seeds for reproducibility."""
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
    
    def to_dict(self):
        """Convert config to dictionary for serialization."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def __repr__(self):
        """String representation of the config."""
        config_str = "Config(\n"
        for key, value in self.to_dict().items():
            config_str += f"  {key}={value},\n"
        config_str += ")"
        return config_str

