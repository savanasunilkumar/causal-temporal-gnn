"""Main training module for the UACT-GNN recommendation system."""

import os
import random
import numpy as np
import torch
import torch.optim as optim
from collections import defaultdict
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ..config import Config
from ..data.processor import DataProcessor
from ..data.samplers import NegativeSampler
from ..models.uact_gnn import CausalTemporalGNN
from ..utils.cold_start import ColdStartSolver
from ..utils.checkpointing import ModelCheckpointer
from ..utils.logging import ExperimentLogger, setup_logging
from .evaluator import Evaluator


class RecommendationSystem:
    """Recommendation System based on Causal Temporal GNN."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        
        self.data_processor = DataProcessor(config)
        self.cold_start_solver = ColdStartSolver(config)
        
        self.data = {}
        self.model = None
        self.edge_index = None
        self.edge_timestamps = None
        self.time_indices = None
        self.train_history = {'train_loss': [], 'val_metrics': {}}
        
        # Initialize components
        self.negative_sampler = None
        self.checkpointer = None
        self.experiment_logger = None
        self.evaluator = None
        
        # Load pretrained models for cold start
        self.cold_start_solver.load_pretrained_models()
        
        # Setup logging
        self.logger = setup_logging(config.log_dir)
    
    def load_data(self, data_path):
        """Load and process data automatically, regardless of its format."""
        self.logger.info(f"Loading data from {data_path}")
        processed_data, schema = self.data_processor.process_data(data_path)
        self.data['processed_data'] = processed_data
        self.data['schema'] = schema
        self.metadata = self.data_processor.metadata
        
        print(f"Loaded and processed {len(processed_data)} interactions")
        print(f"Detected schema: {schema}")
        print(f"Data characteristics: {self.metadata}")
        
        return processed_data, schema
    
    def preprocess_data(self, min_user_interactions=None, min_item_interactions=None):
        """Preprocess the data by filtering inactive users and items."""
        if min_user_interactions is None:
            min_user_interactions = self.config.min_user_interactions
        if min_item_interactions is None:
            min_item_interactions = self.config.min_item_interactions
        
        processed_data = self.data['processed_data']
        schema = self.data['schema']
        
        user_col = schema['user_columns'][0] if schema['user_columns'] else None
        item_col = schema['item_columns'][0] if schema['item_columns'] else None
        
        if user_col is None or item_col is None:
            raise ValueError("Could not identify user or item columns in the data")
        
        interaction_col = schema['interaction_columns'][0] if schema['interaction_columns'] else None
        
        # Apply rating threshold if specified
        if hasattr(self.config, 'rating_threshold') and interaction_col is not None:
            threshold = self.config.rating_threshold
            processed_data = processed_data[processed_data[interaction_col] >= threshold]
        
        # Filter inactive users and items
        user_counts = processed_data[user_col].value_counts()
        item_counts = processed_data[item_col].value_counts()
        
        active_users = user_counts[user_counts >= min_user_interactions].index
        active_items = item_counts[item_counts >= min_item_interactions].index
        
        filtered_data = processed_data[
            processed_data[user_col].isin(active_users) & 
            processed_data[item_col].isin(active_items)
        ]
        
        # Create ID mappings
        unique_users = filtered_data[user_col].unique()
        unique_items = filtered_data[item_col].unique()
        
        user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        item_id_map = {iid: idx for idx, iid in enumerate(unique_items)}
        
        self.user_index_to_id = {idx: uid for uid, idx in user_id_map.items()}
        self.item_index_to_id = {idx: iid for iid, idx in item_id_map.items()}
        
        filtered_data = filtered_data.copy()
        filtered_data['user_idx'] = filtered_data[user_col].map(user_id_map)
        filtered_data['item_idx'] = filtered_data[item_col].map(item_id_map)
        
        self.data['preprocessed_data'] = filtered_data
        self.data['user_id_map'] = user_id_map
        self.data['item_id_map'] = item_id_map
        
        self.metadata['num_users'] = len(unique_users)
        self.metadata['num_items'] = len(unique_items)
        
        print(f"After preprocessing: {len(filtered_data)} interactions, "
              f"{self.metadata['num_users']} users, {self.metadata['num_items']} items")
        
        return filtered_data
    
    def split_data(self, test_size=None, val_size=None):
        """Split data into training, validation, and test sets."""
        if test_size is None:
            test_size = self.config.test_size
        if val_size is None:
            val_size = self.config.val_size
        
        data = self.data['preprocessed_data']
        temporal_cols = [col for col in self.data['schema']['temporal_columns'] if col in data.columns]
        
        # Use temporal splitting if timestamps available
        if temporal_cols:
            data = data.sort_values(temporal_cols[0])
            n = len(data)
            test_start = int(n * (1 - test_size))
            val_start = int(n * (1 - test_size - val_size))
            
            train_data = data.iloc[:val_start]
            val_data = data.iloc[val_start:test_start]
            test_data = data.iloc[test_start:]
        else:
            # Random splitting
            train_val_data, test_data = train_test_split(
                data, test_size=test_size, random_state=self.config.seed
            )
            
            if val_size > 0:
                train_data, val_data = train_test_split(
                    train_val_data, test_size=val_size / (1 - test_size), random_state=self.config.seed
                )
            else:
                train_data = train_val_data
                val_data = None
        
        self.data['train_data'] = train_data
        self.data['val_data'] = val_data
        self.data['test_data'] = test_data
        
        # Build user interaction sets for negative sampling
        self.user_interactions = defaultdict(set)
        for _, row in train_data.iterrows():
            self.user_interactions[row['user_idx']].add(row['item_idx'])
        
        print(f"Data split: {len(train_data)} train, "
              f"{len(val_data) if val_data is not None else 0} validation, "
              f"{len(test_data)} test")
        
        return train_data, val_data, test_data
    
    def create_graph(self):
        """Create a temporal bipartite graph from user-item interactions."""
        train_data = self.data['train_data']
        
        user_nodes = train_data['user_idx'].values
        item_nodes = train_data['item_idx'].values + self.metadata['num_users']
        
        # Create bidirectional edges
        edge_index = torch.zeros((2, len(user_nodes) * 2), dtype=torch.long, device=self.device)
        edge_index[0, :len(user_nodes)] = torch.tensor(user_nodes, dtype=torch.long, device=self.device)
        edge_index[1, :len(user_nodes)] = torch.tensor(item_nodes, dtype=torch.long, device=self.device)
        edge_index[0, len(user_nodes):] = torch.tensor(item_nodes, dtype=torch.long, device=self.device)
        edge_index[1, len(user_nodes):] = torch.tensor(user_nodes, dtype=torch.long, device=self.device)
        
        # Handle timestamps
        temporal_cols = [col for col in self.data['schema']['temporal_columns'] if col in train_data.columns]
        if temporal_cols:
            timestamps = train_data[temporal_cols[0]].values
            edge_timestamps = torch.zeros(len(user_nodes) * 2, dtype=torch.long, device=self.device)
            edge_timestamps[:len(user_nodes)] = torch.tensor(timestamps, dtype=torch.long, device=self.device)
            edge_timestamps[len(user_nodes):] = torch.tensor(timestamps, dtype=torch.long, device=self.device)
        else:
            edge_timestamps = torch.arange(len(user_nodes) * 2, dtype=torch.long, device=self.device)
        
        # Create time indices for nodes
        time_indices = torch.zeros(self.metadata['num_users'] + self.metadata['num_items'], dtype=torch.long, device=self.device)
        
        min_time = edge_timestamps.min()
        max_time = edge_timestamps.max()
        time_range = max_time - min_time
        if time_range == 0:
            time_range = 1
        time_step_size = time_range / self.config.time_steps
        
        for idx, (_, row) in enumerate(train_data.iterrows()):
            user_idx = int(row['user_idx'])
            item_idx = int(row['item_idx']) + self.metadata['num_users']
            
            if temporal_cols:
                timestamp = timestamps[idx]  # Use enumeration index, not row.name
            else:
                timestamp = idx  # Use enumeration index
            
            time_step = int((timestamp - min_time) / time_step_size)
            time_step = min(time_step, self.config.time_steps - 1)
            
            time_indices[user_idx] = max(time_indices[user_idx], time_step)
            time_indices[item_idx] = max(time_indices[item_idx], time_step)
        
        self.edge_index = edge_index
        self.edge_timestamps = edge_timestamps
        self.time_indices = time_indices
        
        print(f"Created temporal graph with {self.edge_index.size(1)} edges")
        return self.edge_index, self.edge_timestamps, self.time_indices
    
    def initialize_model(self):
        """Initialize the Causal Temporal GNN model and supporting components."""
        self.model = CausalTemporalGNN(
            self.config, self.metadata
        ).to(self.device)
        
        self.model.edge_index = self.edge_index
        self.model.edge_timestamps = self.edge_timestamps
        self.model.time_indices = self.time_indices
        
        # Initialize negative sampler (use mixed sampler for better NDCG if enabled)
        if getattr(self.config, 'use_hard_negatives', False):
            from ..data.samplers import MixedNegativeSampler
            self.negative_sampler = MixedNegativeSampler(
                self.metadata['num_items'],
                self.user_interactions,
                device=self.device,
                hard_ratio=getattr(self.config, 'hard_negative_ratio', 0.5),
                pool_size=getattr(self.config, 'hard_negative_pool_size', 100),
            )
            self.logger.info(f"Using MixedNegativeSampler with {self.config.hard_negative_ratio*100:.0f}% hard negatives")
        else:
            self.negative_sampler = NegativeSampler(
                self.metadata['num_items'],
                self.user_interactions,
                device=self.device
            )
        
        # Initialize evaluator
        self.evaluator = Evaluator(self.model, device=self.device)
        
        # Initialize checkpointer
        self.checkpointer = ModelCheckpointer(
            self.config.checkpoint_dir,
            keep_best_k=self.config.keep_best_k_models
        )
        
        # Initialize experiment logger
        if self.config.use_wandb or self.config.use_tensorboard:
            self.experiment_logger = ExperimentLogger(
                self.config,
                project_name='uact-gnn',
                experiment_name=f'run_{self.config.seed}'
            )
        
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Initialized Causal Temporal GNN model with {num_params:,} parameters")
        self.logger.info(f"Model initialized with {num_params:,} parameters")
    
    def sample_negative_items(self, user_idx, n_neg=1):
        """Sample negative items for a user."""
        if self.negative_sampler is not None:
            return self.negative_sampler.sample(user_idx, n_neg)

        # Fallback to simple sampling with max attempts to prevent infinite loops
        positive_items = self.user_interactions.get(user_idx, set())
        neg_items = []
        max_attempts = n_neg * 20  # Reasonable upper bound

        attempts = 0
        while len(neg_items) < n_neg and attempts < max_attempts:
            neg_item = random.randint(0, self.metadata['num_items'] - 1)
            if neg_item not in positive_items and neg_item not in neg_items:
                neg_items.append(neg_item)
            attempts += 1

        # If still not enough negatives, fill with any non-duplicate items
        while len(neg_items) < n_neg:
            neg_item = random.randint(0, self.metadata['num_items'] - 1)
            if neg_item not in neg_items:
                neg_items.append(neg_item)

        return neg_items
    
    def bpr_loss(self, pos_scores, neg_scores):
        """Compute the Bayesian Personalized Ranking (BPR) loss."""
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))
        return loss
    
    def train_epoch(self, optimizer, neg_samples=1, batch_size=1024, scaler=None):
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0.0
        train_data = self.data['train_data']
        
        n_batches = (len(train_data) + batch_size - 1) // batch_size
        batches = np.array_split(train_data.index, n_batches)
        
        for batch_idx, batch_indices in enumerate(tqdm(batches, desc="Training", leave=False)):
            optimizer.zero_grad()
            batch = train_data.loc[batch_indices]
            
            user_indices = torch.tensor(batch['user_idx'].values, dtype=torch.long, device=self.device)
            pos_item_indices = torch.tensor(batch['item_idx'].values, dtype=torch.long, device=self.device)
            
            # Sample negative items
            neg_item_indices_list = []
            for user_idx in batch['user_idx'].values:
                neg_items = self.sample_negative_items(int(user_idx), neg_samples)
                neg_item_indices_list.extend(neg_items)
            
            neg_item_indices = torch.tensor(neg_item_indices_list, dtype=torch.long, device=self.device)
            
            if neg_samples > 1:
                user_indices = user_indices.repeat_interleave(neg_samples)
                pos_item_indices = pos_item_indices.repeat_interleave(neg_samples)
            
            # Forward pass with mixed precision if enabled
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    _, user_embeddings, item_embeddings = self.model.forward(
                        self.edge_index, self.edge_timestamps, self.time_indices
                    )
                    
                    users_emb = user_embeddings[user_indices]
                    pos_items_emb = item_embeddings[pos_item_indices]
                    neg_items_emb = item_embeddings[neg_item_indices]
                    
                    pos_scores = torch.sum(users_emb * pos_items_emb, dim=1)
                    neg_scores = torch.sum(users_emb * neg_items_emb, dim=1)
                    
                    bpr_loss = self.bpr_loss(pos_scores, neg_scores)
                    
                    # Add L2 regularization
                    if self.config.weight_decay > 0:
                        l2_reg = sum(torch.norm(param, 2) for param in self.model.parameters())
                        bpr_loss = bpr_loss + self.config.weight_decay * l2_reg
                
                scaler.scale(bpr_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                _, user_embeddings, item_embeddings = self.model.forward(
                    self.edge_index, self.edge_timestamps, self.time_indices
                )
                
                users_emb = user_embeddings[user_indices]
                pos_items_emb = item_embeddings[pos_item_indices]
                neg_items_emb = item_embeddings[neg_item_indices]
                
                pos_scores = torch.sum(users_emb * pos_items_emb, dim=1)
                neg_scores = torch.sum(users_emb * neg_items_emb, dim=1)
                
                bpr_loss = self.bpr_loss(pos_scores, neg_scores)
                
                # Add L2 regularization
                if self.config.weight_decay > 0:
                    l2_reg = sum(torch.norm(param, 2) for param in self.model.parameters())
                    bpr_loss = bpr_loss + self.config.weight_decay * l2_reg
                
                bpr_loss.backward()
                optimizer.step()
            
            total_loss += bpr_loss.item()
            
            # Log batch metrics
            if self.experiment_logger and batch_idx % self.config.log_every_n_steps == 0:
                self.experiment_logger.log_metrics({
                    'batch_loss': bpr_loss.item(),
                    'batch': batch_idx
                })
        
        avg_loss = total_loss / n_batches
        return avg_loss
    
    def evaluate(self, data_split='val', k_values=[5, 10, 20], batch_size=1024):
        """Evaluate the model on validation or test data."""
        if data_split == 'val':
            eval_data = self.data['val_data']
        elif data_split == 'test':
            eval_data = self.data['test_data']
        else:
            raise ValueError(f"Invalid data split: {data_split}")
        
        if eval_data is None or len(eval_data) == 0:
            return None
        
        return self.evaluator.evaluate(eval_data, self.user_interactions, k_values, batch_size)
    
    def train(self):
        """Train the model using BPR loss with early stopping and LR scheduling."""
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0  # We handle L2 regularization manually
        )
        
        # Learning rate scheduler (Cosine Annealing with Warm Restarts)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,  # Restart every 10 epochs
            T_mult=2,  # Double the period after each restart
            eta_min=self.config.learning_rate * 0.01  # Min LR is 1% of initial
        )
        
        # Mixed precision training
        scaler = None
        if self.config.use_amp and torch.cuda.is_available():
            scaler = torch.cuda.amp.GradScaler()
            self.logger.info("Using mixed precision training")
        
        neg_samples = self.config.neg_samples
        batch_size = self.config.batch_size
        num_epochs = self.config.num_epochs
        patience = self.config.early_stopping_patience
        
        best_metric = 0.0
        best_epoch = 0
        no_improvement = 0
        
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(optimizer, neg_samples, batch_size, scaler)
            self.train_history['train_loss'].append(train_loss)
            
            # Log training loss
            if self.experiment_logger:
                self.experiment_logger.log_metrics({'train_loss': train_loss, 'epoch': epoch})
            
            # Validation
            if self.data['val_data'] is not None and len(self.data['val_data']) > 0:
                val_metrics = self.evaluate('val', k_values=[10])
                
                # Track validation metrics
                for metric, values in val_metrics.items():
                    if metric not in self.train_history['val_metrics']:
                        self.train_history['val_metrics'][metric] = {}
                    for k, value in values.items():
                        key = f"{metric}@{k}"
                        if key not in self.train_history['val_metrics']:
                            self.train_history['val_metrics'][key] = []
                        self.train_history['val_metrics'][key].append(value)
                
                current_metric = val_metrics['ndcg'][10]
                
                # Log validation metrics
                if self.experiment_logger:
                    log_dict = {'epoch': epoch}
                    for metric, values in val_metrics.items():
                        for k, value in values.items():
                            log_dict[f'val_{metric}@{k}'] = value
                    self.experiment_logger.log_metrics(log_dict)
                
                print(f"Epoch {epoch}/{num_epochs}, Loss: {train_loss:.4f}, "
                      f"NDCG@10: {current_metric:.4f}, "
                      f"Precision@10: {val_metrics['precision'][10]:.4f}, "
                      f"Recall@10: {val_metrics['recall'][10]:.4f}")
                
                # Early stopping and checkpointing
                is_best = current_metric > best_metric
                if is_best:
                    best_metric = current_metric
                    best_epoch = epoch
                    no_improvement = 0
                else:
                    no_improvement += 1
                    print(f"No improvement for {no_improvement} epochs "
                          f"(best NDCG@10: {best_metric:.4f} at epoch {best_epoch})")
                
                # Save checkpoint
                if epoch % self.config.save_every_n_epochs == 0 or is_best:
                    self.checkpointer.save_checkpoint(
                        self.model, optimizer, epoch, val_metrics, self.config,
                        is_best=is_best, metric_value=current_metric
                    )
                
                # Step the learning rate scheduler
                scheduler.step()
                
                if no_improvement >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            else:
                print(f"Epoch {epoch}/{num_epochs}, Loss: {train_loss:.4f}")
                
                # Step the learning rate scheduler
                scheduler.step()
                
                # Save checkpoint periodically
                if epoch % self.config.save_every_n_epochs == 0:
                    self.checkpointer.save_checkpoint(
                        self.model, optimizer, epoch, {}, self.config
                    )
        
        # Load best model
        if self.data['val_data'] is not None:
            best_checkpoint = self.checkpointer.load_best_checkpoint(
                self.model, optimizer, self.device
            )
            if best_checkpoint:
                print(f"Loaded best model from epoch {best_checkpoint['epoch']}")
                self.logger.info(f"Loaded best model from epoch {best_checkpoint['epoch']}")
        
        # Finish logging
        if self.experiment_logger:
            self.experiment_logger.finish()
        
        return self.train_history
    
    def generate_recommendations(self, user_id, top_k=10):
        """Generate recommendations for a specific user."""
        self.model.eval()
        
        if user_id not in self.data['user_id_map']:
            raise ValueError(f"User ID {user_id} not found in the dataset")
        
        user_idx = self.data['user_id_map'][user_id]
        excluded_items = {user_idx: list(self.user_interactions[user_idx])}
        
        with torch.no_grad():
            user_indices = torch.tensor([user_idx], dtype=torch.long, device=self.device)
            top_indices, top_scores = self.model.recommend_items(
                user_indices, top_k=top_k, excluded_items=excluded_items
            )
        
        top_indices = top_indices.cpu().numpy()[0]
        top_scores = top_scores.cpu().numpy()[0]
        
        recommended_items = [self.item_index_to_id[idx] for idx in top_indices]
        
        return recommended_items, top_scores
    
    def generate_cold_start_recommendations(self, user_data, top_k=10):
        """Generate recommendations for a cold start user using zero-shot learning."""
        user_profile = self.cold_start_solver.generate_user_profile(user_data, self.data['schema'])
        
        item_profiles = []
        for item_idx in range(self.metadata['num_items']):
            item_id = self.item_index_to_id[item_idx]
            
            item_data_row = self.data['preprocessed_data'][
                self.data['preprocessed_data']['item_idx'] == item_idx
            ]
            if not item_data_row.empty:
                item_data = item_data_row.iloc[0].to_dict()
                item_profile = self.cold_start_solver.generate_item_profile(item_data, self.data['schema'])
                item_profiles.append(item_profile)
            else:
                # Create a dummy profile if no data found
                item_profiles.append({
                    'text_features': np.zeros(self.config.embedding_dim),
                    'image_features': np.zeros(self.config.embedding_dim),
                    'numeric_features': np.zeros(10),
                    'categorical_features': np.zeros(10)
                })
        
        predictions = self.cold_start_solver.predict_cold_start_interactions(user_profile, item_profiles)
        
        top_indices = np.argsort(predictions)[::-1][:top_k]
        top_scores = predictions[top_indices]
        
        recommended_items = [self.item_index_to_id[idx] for idx in top_indices]
        
        return recommended_items, top_scores
    
    def save_model(self, path):
        """Save the model to a file."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config.to_dict(),
            'metadata': self.metadata
        }, path)
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load the model from a file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.info(f"Model loaded from {path}")

