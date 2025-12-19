"""Enhanced Universal Adaptive Causal Temporal GNN model with PyTorch Geometric."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.checkpoint import checkpoint

from ..causal.discovery import CausalGraphConstructor
from .layers import CausalGNNLayer, TemporalAttentionLayer, GraphSAGELayer


class CausalTemporalGNN(nn.Module):
    """Causal Temporal Graph Neural Network for recommendations with PyTorch Geometric."""
    
    def __init__(self, config, metadata):
        super().__init__()
        self.config = config
        self.metadata = metadata
        self.num_users = metadata['num_users']
        self.num_items = metadata['num_items']
        
        self.embedding_dim = config.embedding_dim
        self.num_layers = config.num_layers
        self.time_steps = config.time_steps
        self.dropout = config.dropout
        self.causal_strength = config.causal_strength
        
        # User and item embeddings
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_dim)
        
        # Attribute embeddings for zero-shot learning
        self.user_attribute_embedding = nn.Linear(self.embedding_dim * 4, self.embedding_dim)  # 4 modalities
        self.item_attribute_embedding = nn.Linear(self.embedding_dim * 4, self.embedding_dim)
        
        # Temporal embeddings
        self.temporal_embedding = nn.Embedding(self.time_steps, self.embedding_dim)
        
        # Causal embeddings
        self.causal_embedding = nn.Embedding(self.num_users + self.num_items, self.embedding_dim)
        
        # Context embeddings
        self.context_embedding = nn.Linear(20, self.embedding_dim)
        
        # Multi-head attention for temporal dynamics
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim, 
            num_heads=4, 
            dropout=self.dropout
        )
        
        # Transformer encoder for sequence modeling
        encoder_layers = TransformerEncoderLayer(
            d_model=self.embedding_dim, 
            nhead=4, 
            dropout=self.dropout
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=2)
        
        # PyG Causal graph layers (replacing manual operations)
        self.causal_layers = nn.ModuleList([
            CausalGNNLayer(
                self.embedding_dim, 
                self.embedding_dim,
                dropout=self.dropout,
                use_edge_weight=True
            ) for _ in range(self.num_layers)
        ])
        
        # Temporal attention layers using PyG
        self.temporal_gnn_layers = nn.ModuleList([
            TemporalAttentionLayer(
                self.embedding_dim,
                self.embedding_dim,
                heads=4,
                dropout=self.dropout
            ) for _ in range(2)  # 2 temporal layers
        ])
        
        # GraphSAGE layers for neighbor sampling (large graphs)
        self.sage_layers = nn.ModuleList([
            GraphSAGELayer(
                self.embedding_dim,
                self.embedding_dim,
                normalize=True
            ) for _ in range(self.num_layers)
        ])
        
        # Gradient checkpointing flag
        self.use_gradient_checkpointing = getattr(config, 'use_gradient_checkpointing', False)
        
        # Cached causal graph
        self.cached_causal_edge_index = None
        self.cached_causal_edge_weights = None
        
        # Contrastive learning layers
        self.contrastive_projection = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, 128)
        )
        
        # Output layers
        self.output_layer = nn.Linear(self.embedding_dim, 1)
        
        # Initialize causal constructor
        self.causal_constructor = CausalGraphConstructor(config)
        
        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        nn.init.normal_(self.temporal_embedding.weight, std=0.1)
        nn.init.normal_(self.causal_embedding.weight, std=0.1)
        
        # Device
        self.device = torch.device(config.device)
        
    def _get_ego_embeddings(self):
        """Get the initial embeddings for all users and items."""
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings
    
    def _get_temporal_embeddings(self, time_indices):
        """Get temporal embeddings for the given time indices."""
        return self.temporal_embedding(time_indices)
    
    def _get_causal_embeddings(self):
        """Get causal embeddings for all nodes."""
        return self.causal_embedding.weight
    
    def _compute_causal_graph(self, edge_index, edge_timestamps, node_features):
        """Compute causal relationships using advanced techniques."""
        if self.config.causal_method == 'advanced':
            interaction_data = edge_index.detach().cpu().numpy()
            node_features_np = node_features.detach().cpu().numpy()
            edge_timestamps_np = edge_timestamps.detach().cpu().numpy()
            
            causal_edges, edge_weights = self.causal_constructor.compute_hybrid_causal_graph(
                interaction_data, node_features_np, edge_timestamps_np
            )
            
            if causal_edges:
                causal_edge_index = torch.tensor(causal_edges, dtype=torch.long).t().contiguous().to(self.device)
                causal_edge_weights = torch.tensor(edge_weights, dtype=torch.float).to(self.device)
            else:
                causal_edge_index = edge_index
                causal_edge_weights = torch.ones(edge_index.size(1), dtype=torch.float).to(self.device)
            
            return causal_edge_index, causal_edge_weights
        else:
            # Fallback to original simplified method
            causal_edges = []
            min_time = edge_timestamps.min()
            max_time = edge_timestamps.max()
            time_range = max_time - min_time
            time_step_size = time_range / self.time_steps
            
            normalized_times = ((edge_timestamps - min_time) / time_step_size).long()
            normalized_times = torch.clamp(normalized_times, 0, self.time_steps - 1)
            
            for i in range(len(edge_index[0])):
                src = edge_index[0, i]
                dst = edge_index[1, i]
                time = normalized_times[i]
                
                for j in range(len(edge_index[0])):
                    if i != j and edge_index[0, j] == src and normalized_times[j] > time:
                        causal_edges.append((src, edge_index[1, j]))
                    if i != j and edge_index[1, j] == dst and normalized_times[j] > time:
                        causal_edges.append((edge_index[0, j], dst))
            
            if causal_edges:
                causal_edge_index = torch.tensor(causal_edges, dtype=torch.long).t().contiguous().to(self.device)
            else:
                causal_edge_index = edge_index
            
            return causal_edge_index, None
    
    def _propagate_causal_information(self, node_features, causal_edge_index, causal_edge_weights=None):
        """
        Propagate information through the causal graph using PyG layers.
        
        This is now MUCH more efficient with PyG's optimized message passing!
        """
        causal_embeddings = self._get_causal_embeddings()
        combined_features = node_features + self.causal_strength * causal_embeddings
        
        # Use PyG layers for efficient message passing
        for i, layer in enumerate(self.causal_layers):
            if self.use_gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory (50-70% reduction)
                combined_features = checkpoint(
                    layer,
                    combined_features,
                    causal_edge_index,
                    causal_edge_weights,
                    use_reentrant=False
                )
            else:
                # Standard forward pass with PyG
                combined_features = layer(
                    combined_features,
                    causal_edge_index,
                    edge_weight=causal_edge_weights
                )
            
            # Apply activation
            combined_features = F.relu(combined_features)
        
        return combined_features
    
    def _apply_temporal_attention(self, node_features, time_indices):
        """Apply temporal attention to capture temporal dynamics."""
        time_emb = self._get_temporal_embeddings(time_indices)
        batch_size = node_features.size(0)
        seq_length = self.time_steps
        
        temporal_features = []
        for t in range(seq_length):
            time_mask = (time_indices == t).unsqueeze(1).float()
            time_step_features = node_features + time_emb[t]
            temporal_features.append(time_step_features * time_mask)
        
        temporal_features = torch.stack(temporal_features, dim=1)
        attended_features, _ = self.temporal_attention(
            temporal_features, temporal_features, temporal_features
        )
        aggregated_features = torch.mean(attended_features, dim=1)
        
        return aggregated_features
    
    def _contrastive_loss(self, embeddings, positive_pairs, negative_pairs, temperature=0.1):
        """Compute contrastive loss for self-supervised learning."""
        embeddings = F.normalize(embeddings, dim=1)
        projections = self.contrastive_projection(embeddings)
        projections = F.normalize(projections, dim=1)
        
        similarity_matrix = torch.matmul(projections, projections.t()) / temperature
        labels = torch.arange(embeddings.size(0), device=embeddings.device)
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss
    
    def forward(self, edge_index, edge_timestamps, time_indices=None, user_attributes=None, item_attributes=None, context_features=None):
        """Forward pass of the Enhanced UACT-GNN model."""
        all_embeddings = self._get_ego_embeddings()
        
        causal_edge_index, causal_edge_weights = self._compute_causal_graph(edge_index, edge_timestamps, all_embeddings)
        
        if time_indices is None:
            time_indices = torch.zeros(all_embeddings.size(0), dtype=torch.long, device=edge_index.device)
        
        temporal_features = self._apply_temporal_attention(all_embeddings, time_indices)
        all_embeddings = all_embeddings + temporal_features
        
        if user_attributes is not None:
            user_attr_emb = self.user_attribute_embedding(user_attributes)
            all_embeddings[:self.num_users] = all_embeddings[:self.num_users] + user_attr_emb
        
        if item_attributes is not None:
            item_attr_emb = self.item_attribute_embedding(item_attributes)
            all_embeddings[self.num_users:] = all_embeddings[self.num_users:] + item_attr_emb
        
        if context_features is not None:
            context_emb = self.context_embedding(context_features)
            all_embeddings = all_embeddings + context_emb
        
        causal_features = self._propagate_causal_information(all_embeddings, causal_edge_index, causal_edge_weights)
        all_embeddings = all_embeddings + causal_features
        
        user_embeddings = all_embeddings[:self.num_users]
        item_embeddings = all_embeddings[self.num_users:]
        
        return all_embeddings, user_embeddings, item_embeddings
    
    def get_user_embeddings(self):
        """Get the final user embeddings after propagation."""
        all_embeddings, user_embeddings, _ = self.forward(self.edge_index, self.edge_timestamps, self.time_indices)
        return user_embeddings
    
    def get_item_embeddings(self):
        """Get the final item embeddings after propagation."""
        all_embeddings, _, item_embeddings = self.forward(self.edge_index, self.edge_timestamps, self.time_indices)
        return item_embeddings
    
    def predict(self, user_indices, item_indices):
        """Predict the scores for user-item pairs."""
        _, user_embeddings, item_embeddings = self.forward(self.edge_index, self.edge_timestamps, self.time_indices)
        user_emb = user_embeddings[user_indices]
        item_emb = item_embeddings[item_indices]
        scores = torch.sum(user_emb * item_emb, dim=1)
        return scores
    
    def recommend_items(self, user_indices, top_k=10, excluded_items=None):
        """Generate top-k recommendations for users."""
        _, user_embeddings, item_embeddings = self.forward(self.edge_index, self.edge_timestamps, self.time_indices)
        user_emb = user_embeddings[user_indices]
        scores = torch.matmul(user_emb, item_embeddings.t())
        
        if excluded_items is not None:
            for i, user_idx in enumerate(user_indices.tolist()):
                if user_idx in excluded_items:
                    scores[i, excluded_items[user_idx]] = -float('inf')
        
        top_scores, top_indices = torch.topk(scores, k=top_k, dim=1)
        return top_indices, top_scores

