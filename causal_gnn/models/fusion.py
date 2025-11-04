"""Learnable multi-modal fusion components."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableMultiModalFusion(nn.Module):
    """Learnable fusion of multi-modal similarities using attention."""
    
    def __init__(self, modalities, embedding_dim):
        super().__init__()
        self.modalities = modalities
        self.embedding_dim = embedding_dim
        
        # Learnable query vector for attention
        self.query = nn.Parameter(torch.randn(1, embedding_dim))
        
        # Linear layers to project each modality to a common space
        self.modality_projections = nn.ModuleDict({
            modality: nn.Linear(1, embedding_dim) for modality in modalities
        })
        
        # Output layer to produce final similarity score
        self.output_layer = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1)
        )
        
    def forward(self, similarities):
        """
        Fuse similarities from different modalities.
        
        Args:
            similarities: Dictionary of similarities {modality: tensor}
            
        Returns:
            Fused similarity score
        """
        # Project each similarity to the embedding space
        projected_similarities = []
        for modality, similarity in similarities.items():
            if modality in self.modality_projections:
                projected = self.modality_projections[modality](similarity.unsqueeze(-1))
                projected_similarities.append(projected)
        
        if not projected_similarities:
            return torch.tensor(0.0)
        
        # Stack projected similarities
        stacked_similarities = torch.stack(projected_similarities, dim=1)  # [batch_size, n_modalities, embedding_dim]
        
        # Compute attention weights
        query_expanded = self.query.expand(stacked_similarities.size(0), -1, -1)  # [batch_size, 1, embedding_dim]
        attention_scores = torch.sum(stacked_similarities * query_expanded, dim=2)  # [batch_size, n_modalities]
        attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(-1)  # [batch_size, n_modalities, 1]
        
        # Apply attention weights
        weighted_similarities = torch.sum(stacked_similarities * attention_weights, dim=1)  # [batch_size, embedding_dim]
        
        # Produce final similarity score
        fused_similarity = self.output_layer(weighted_similarities).squeeze(-1)
        
        return fused_similarity

