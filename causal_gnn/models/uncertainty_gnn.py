"""Uncertainty-Aware Causal Temporal GNN for Recommendations.

This module implements a novel GNN architecture that provides uncertainty
quantification for recommendations by:
1. Propagating uncertainty through causal graph layers
2. Using Monte Carlo dropout for epistemic uncertainty
3. Learning aleatoric uncertainty from data
4. Providing confidence scores with each recommendation

Novel contribution: First GNN-based recommender with full uncertainty
quantification through both causal structure and neural network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from typing import Tuple, Optional, Dict, List
import numpy as np

from ..causal.bayesian_discovery import (
    UncertaintyAwareCausalLayer,
    compute_recommendation_confidence,
    get_confident_recommendations,
)


class UncertaintyAwareCausalTemporalGNN(nn.Module):
    """
    Uncertainty-Aware Causal Temporal Graph Neural Network.

    This model extends the standard CausalTemporalGNN by adding:
    1. Bayesian layers for epistemic uncertainty
    2. Heteroscedastic outputs for aleatoric uncertainty
    3. Uncertainty propagation through causal message passing
    4. Confidence-calibrated recommendations

    Architecture:
    - User/Item embeddings with uncertainty (mean + variance)
    - Temporal embeddings with position uncertainty
    - Bayesian causal graph layers
    - Monte Carlo dropout for uncertainty estimation
    - Dual-head output: predictions + confidence
    """

    def __init__(self, config, metadata):
        super().__init__()
        self.config = config
        self.metadata = metadata

        # Dimensions
        self.embedding_dim = config.embedding_dim
        self.num_users = metadata['num_users']
        self.num_items = metadata['num_items']
        self.num_nodes = self.num_users + self.num_items
        self.num_layers = config.num_layers
        self.time_steps = config.time_steps
        self.dropout = config.dropout

        # Uncertainty parameters
        self.mc_dropout_samples = getattr(config, 'mc_dropout_samples', 10)
        self.uncertainty_weight = getattr(config, 'uncertainty_weight', 0.1)
        self.min_variance = getattr(config, 'min_variance', 1e-6)

        # User and Item embeddings (mean)
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_dim)

        # User and Item embedding variance (learnable aleatoric uncertainty)
        self.user_embedding_log_var = nn.Embedding(self.num_users, self.embedding_dim)
        self.item_embedding_log_var = nn.Embedding(self.num_items, self.embedding_dim)

        # Temporal embeddings
        self.temporal_embedding = nn.Embedding(self.time_steps, self.embedding_dim)
        self.temporal_embedding_log_var = nn.Embedding(self.time_steps, self.embedding_dim)

        # Causal embeddings for discovered causal relationships
        self.causal_embedding = nn.Embedding(self.num_nodes, self.embedding_dim)

        # Uncertainty-aware causal layers
        self.causal_layers = nn.ModuleList([
            UncertaintyAwareCausalLayer(
                self.embedding_dim,
                self.embedding_dim,
                dropout=self.dropout
            ) for _ in range(self.num_layers)
        ])

        # Temporal attention with uncertainty
        self.temporal_attention = UncertainTemporalAttention(
            self.embedding_dim,
            num_heads=4,
            dropout=self.dropout
        )

        # Output projection
        self.output_mean = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.output_log_var = nn.Linear(self.embedding_dim, self.embedding_dim)

        # Confidence calibration layer
        self.confidence_calibrator = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_dim, 1),
            nn.Sigmoid()
        )

        # Initialize embeddings
        self._init_embeddings()

        # Store graph structure
        self.edge_index = None
        self.edge_timestamps = None
        self.time_indices = None
        self.edge_weight_mean = None
        self.edge_weight_var = None

    def _init_embeddings(self):
        """Initialize embeddings with appropriate distributions."""
        nn.init.normal_(self.user_embedding.weight, mean=0, std=0.1)
        nn.init.normal_(self.item_embedding.weight, mean=0, std=0.1)
        nn.init.normal_(self.temporal_embedding.weight, mean=0, std=0.1)
        nn.init.normal_(self.causal_embedding.weight, mean=0, std=0.1)

        # Initialize log-variances to small values (low initial uncertainty)
        initial_log_var = getattr(self.config, 'initial_log_variance', -2.0)
        nn.init.constant_(self.user_embedding_log_var.weight, initial_log_var)
        nn.init.constant_(self.item_embedding_log_var.weight, initial_log_var)
        nn.init.constant_(self.temporal_embedding_log_var.weight, initial_log_var)

    def set_causal_graph(
        self,
        edge_weight_mean: torch.Tensor,
        edge_weight_var: torch.Tensor
    ):
        """Set the uncertain causal graph weights."""
        self.edge_weight_mean = edge_weight_mean
        self.edge_weight_var = edge_weight_var

    def forward(
        self,
        edge_index: torch.Tensor,
        edge_timestamps: torch.Tensor,
        time_indices: torch.Tensor,
        return_uncertainty: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass with uncertainty propagation.

        Args:
            edge_index: Graph edges [2, num_edges]
            edge_timestamps: Edge timestamps [num_edges]
            time_indices: Node time indices [num_nodes]
            return_uncertainty: Whether to return uncertainty estimates

        Returns:
            Tuple of:
            - all_embeddings_mean: Mean node embeddings [num_nodes, dim]
            - user_embeddings_mean: Mean user embeddings [num_users, dim]
            - item_embeddings_mean: Mean item embeddings [num_items, dim]
            - all_embeddings_var: Variance of embeddings (if return_uncertainty)
            - confidence: Calibrated confidence scores (if return_uncertainty)
        """
        # Get embedding means
        user_emb_mean = self.user_embedding.weight
        item_emb_mean = self.item_embedding.weight
        all_emb_mean = torch.cat([user_emb_mean, item_emb_mean], dim=0)

        # Get embedding variances
        user_emb_var = torch.exp(self.user_embedding_log_var.weight) + self.min_variance
        item_emb_var = torch.exp(self.item_embedding_log_var.weight) + self.min_variance
        all_emb_var = torch.cat([user_emb_var, item_emb_var], dim=0)

        # Add temporal embeddings
        temporal_emb_mean = self.temporal_embedding(time_indices)
        temporal_emb_var = torch.exp(self.temporal_embedding_log_var(time_indices)) + self.min_variance

        all_emb_mean = all_emb_mean + temporal_emb_mean
        all_emb_var = all_emb_var + temporal_emb_var  # Variances add for independent variables

        # Add causal embeddings
        causal_emb = self.causal_embedding.weight
        all_emb_mean = all_emb_mean + causal_emb

        # Prepare edge weights
        if self.edge_weight_mean is None:
            # Default uniform weights with low uncertainty
            num_edges = edge_index.size(1)
            default_var = getattr(self.config, 'default_edge_weight_var', 0.01)
            edge_weight_mean = torch.ones(num_edges, device=edge_index.device)
            edge_weight_var = torch.ones(num_edges, device=edge_index.device) * default_var
        else:
            edge_weight_mean = self.edge_weight_mean
            edge_weight_var = self.edge_weight_var

        # Apply uncertainty-aware causal layers
        h_mean, h_var = all_emb_mean, all_emb_var
        for layer in self.causal_layers:
            h_mean, h_var = layer(h_mean, edge_index, edge_weight_mean, edge_weight_var)

        # Apply temporal attention with uncertainty
        h_mean, h_var = self.temporal_attention(
            h_mean, h_var, edge_index, edge_timestamps
        )

        # Output projection
        out_mean = self.output_mean(h_mean)
        out_log_var = self.output_log_var(h_mean)
        out_var = torch.exp(out_log_var) + self.min_variance

        # Split into user and item embeddings
        user_emb_mean = out_mean[:self.num_users]
        item_emb_mean = out_mean[self.num_users:]
        user_emb_var = out_var[:self.num_users]
        item_emb_var = out_var[self.num_users:]

        if return_uncertainty:
            # Compute calibrated confidence
            confidence_input = torch.cat([out_mean, torch.sqrt(out_var)], dim=-1)
            confidence = self.confidence_calibrator(confidence_input)

            return out_mean, user_emb_mean, item_emb_mean, out_var, confidence
        else:
            return out_mean, user_emb_mean, item_emb_mean, None, None

    def forward_with_mc_dropout(
        self,
        edge_index: torch.Tensor,
        edge_timestamps: torch.Tensor,
        time_indices: torch.Tensor,
        n_samples: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with Monte Carlo dropout for epistemic uncertainty.

        Runs multiple forward passes with dropout enabled to estimate
        model uncertainty (epistemic uncertainty).

        Args:
            edge_index: Graph edges
            edge_timestamps: Edge timestamps
            time_indices: Node time indices
            n_samples: Number of MC samples

        Returns:
            Tuple of (mean_prediction, epistemic_var, aleatoric_var, total_var)
        """
        if n_samples is None:
            n_samples = self.mc_dropout_samples

        # Enable dropout during inference
        self.train()
        samples = []

        for _ in range(n_samples):
            out_mean, _, _, out_var, _ = self.forward(
                edge_index, edge_timestamps, time_indices, return_uncertainty=True
            )
            samples.append(out_mean.unsqueeze(0))

        # Stack samples
        samples = torch.cat(samples, dim=0)  # [n_samples, num_nodes, dim]

        # Epistemic uncertainty: variance of means across samples
        epistemic_mean = samples.mean(dim=0)
        epistemic_var = samples.var(dim=0)

        # Aleatoric uncertainty: from the model's variance output
        self.eval()
        _, _, _, aleatoric_var, _ = self.forward(
            edge_index, edge_timestamps, time_indices, return_uncertainty=True
        )

        # Total uncertainty: sum of epistemic and aleatoric
        total_var = epistemic_var + aleatoric_var

        return epistemic_mean, epistemic_var, aleatoric_var, total_var

    def recommend_items_with_uncertainty(
        self,
        user_indices: torch.Tensor,
        top_k: int = 10,
        excluded_items: Optional[Dict[int, List[int]]] = None,
        confidence_threshold: float = 0.5,
        use_mc_dropout: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate recommendations with uncertainty estimates.

        Args:
            user_indices: User indices to recommend for
            top_k: Number of recommendations
            excluded_items: Items to exclude per user
            confidence_threshold: Minimum confidence for recommendations
            use_mc_dropout: Whether to use MC dropout for epistemic uncertainty

        Returns:
            Tuple of:
            - top_indices: Top-k item indices [batch, k]
            - top_scores: Prediction scores [batch, k]
            - top_confidence: Confidence scores [batch, k]
            - uncertain_flags: Flags for uncertain recommendations [batch, k]
        """
        self.eval()

        with torch.no_grad():
            if use_mc_dropout:
                # Use MC dropout for uncertainty
                mean_emb, epistemic_var, aleatoric_var, total_var = self.forward_with_mc_dropout(
                    self.edge_index, self.edge_timestamps, self.time_indices
                )
                user_emb_mean = mean_emb[:self.num_users]
                item_emb_mean = mean_emb[self.num_users:]
                user_emb_var = total_var[:self.num_users]
                item_emb_var = total_var[self.num_users:]
            else:
                # Use single forward pass
                _, user_emb_mean, item_emb_mean, out_var, _ = self.forward(
                    self.edge_index, self.edge_timestamps, self.time_indices
                )
                user_emb_var = out_var[:self.num_users]
                item_emb_var = out_var[self.num_users:]

            # Get embeddings for requested users
            batch_user_mean = user_emb_mean[user_indices]
            batch_user_var = user_emb_var[user_indices]

            # Compute scores with uncertainty
            scores_mean, scores_var = compute_recommendation_confidence(
                batch_user_mean, batch_user_var,
                item_emb_mean, item_emb_var
            )

            # Mask excluded items
            if excluded_items is not None:
                for i, user_idx in enumerate(user_indices.cpu().tolist()):
                    if user_idx in excluded_items:
                        for item_idx in excluded_items[user_idx]:
                            if item_idx < scores_mean.size(1):
                                scores_mean[i, item_idx] = float('-inf')

            # Get confident recommendations
            top_indices, top_scores, confident_flags = get_confident_recommendations(
                scores_mean, scores_var, top_k, confidence_threshold
            )

            # Compute confidence for top items
            batch_size = user_indices.size(0)
            scores_std = torch.sqrt(scores_var + 1e-6)
            confidence = scores_mean / (scores_std + scores_mean.abs() + 1e-6)
            confidence = torch.sigmoid(confidence * 2)  # Scale and normalize
            top_confidence = torch.gather(confidence, 1, top_indices)

            # Flag uncertain recommendations
            uncertain_flags = ~confident_flags

        return top_indices, top_scores, top_confidence, uncertain_flags

    def predict_with_uncertainty(
        self,
        user_indices: torch.Tensor,
        item_indices: torch.Tensor,
        edge_index: torch.Tensor,
        edge_timestamps: torch.Tensor,
        time_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict scores with uncertainty for specific user-item pairs.

        Args:
            user_indices: User indices
            item_indices: Item indices
            edge_index: Graph edges
            edge_timestamps: Edge timestamps
            time_indices: Node time indices

        Returns:
            Tuple of (scores, uncertainties)
        """
        self.eval()
        with torch.no_grad():
            _, user_emb_mean, item_emb_mean, out_var, _ = self.forward(
                edge_index, edge_timestamps, time_indices
            )
            user_emb_var = out_var[:self.num_users]
            item_emb_var = out_var[self.num_users:]

            # Get embeddings for specific pairs
            u_mean = user_emb_mean[user_indices]
            u_var = user_emb_var[user_indices]
            i_mean = item_emb_mean[item_indices]
            i_var = item_emb_var[item_indices]

            # Compute scores
            scores = torch.sum(u_mean * i_mean, dim=1)

            # Compute uncertainty (variance of dot product)
            score_var = torch.sum(
                u_var * i_mean**2 + i_var * u_mean**2 + u_var * i_var,
                dim=1
            )
            uncertainties = torch.sqrt(score_var + self.min_variance)

        return scores, uncertainties

    def compute_uncertainty_loss(
        self,
        pos_scores_mean: torch.Tensor,
        pos_scores_var: torch.Tensor,
        neg_scores_mean: torch.Tensor,
        neg_scores_var: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute uncertainty-aware loss that encourages:
        1. Higher scores for positive items
        2. Lower uncertainty for positive items
        3. Higher uncertainty for negative items (less confident about negatives)

        This implements a novel uncertainty-regularized BPR loss.
        """
        # Standard BPR component
        score_diff = pos_scores_mean - neg_scores_mean
        bpr_loss = -F.logsigmoid(score_diff).mean()

        # Uncertainty regularization
        # Penalize high variance for positive items
        pos_var_penalty = pos_scores_var.mean()

        # Encourage some variance for negative items (epistemic humility)
        neg_var_bonus = -torch.log(neg_scores_var + 1e-6).mean() * 0.1

        # Combined loss
        total_loss = bpr_loss + self.uncertainty_weight * (pos_var_penalty + neg_var_bonus)

        return total_loss


class UncertainTemporalAttention(nn.Module):
    """
    Temporal attention layer with uncertainty propagation.

    Extends standard temporal attention by:
    1. Computing attention weights with uncertainty
    2. Propagating variance through attention mechanism
    3. Outputting uncertainty estimates
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Variance projection
        self.var_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x_mean: torch.Tensor,
        x_var: torch.Tensor,
        edge_index: torch.Tensor,
        edge_timestamps: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with uncertainty propagation.

        Args:
            x_mean: Mean node features [num_nodes, dim]
            x_var: Variance of node features [num_nodes, dim]
            edge_index: Edge indices
            edge_timestamps: Edge timestamps

        Returns:
            Tuple of (output_mean, output_var)
        """
        # Project to Q, K, V
        q = self.q_proj(x_mean)
        k = self.k_proj(x_mean)
        v = self.v_proj(x_mean)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out_mean = torch.matmul(attn_weights, v)
        out_mean = self.out_proj(out_mean)

        # Propagate variance through attention
        # Var(sum of weighted vars) = sum of squared_weights * vars
        attn_weights_sq = attn_weights ** 2
        v_var = self.var_proj(x_var)
        out_var = torch.matmul(attn_weights_sq, v_var)

        return out_mean, out_var


class UncertaintyCalibrator(nn.Module):
    """
    Calibrates uncertainty estimates to be well-calibrated.

    Uses temperature scaling and learned recalibration to ensure
    that confidence scores match empirical accuracy.
    """

    def __init__(self, initial_temperature: float = 1.0, embed_dim: int = None):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor([initial_temperature]))
        self.recalibrator = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def calibrate(self, scores: torch.Tensor, labels: torch.Tensor, lr: float = 0.01, max_iter: int = 50):
        """
        Learn optimal temperature from validation data.

        Args:
            scores: Prediction scores
            labels: Ground truth labels
            lr: Learning rate for optimization
            max_iter: Maximum iterations
        """
        self.temperature.requires_grad = True
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            calibrated = scores / self.temperature
            loss = F.binary_cross_entropy_with_logits(calibrated, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature.requires_grad = False

    def forward(
        self,
        scores_mean: torch.Tensor,
        scores_var: torch.Tensor
    ) -> torch.Tensor:
        """
        Calibrate confidence scores.

        Args:
            scores_mean: Mean prediction scores
            scores_var: Variance of prediction scores

        Returns:
            Calibrated confidence scores in [0, 1]
        """
        # Temperature scaling
        calibrated_mean = scores_mean / self.temperature

        # Compute raw confidence
        scores_std = torch.sqrt(scores_var + 1e-6)
        raw_confidence = calibrated_mean / (scores_std + 1e-6)

        # Recalibrate
        features = torch.stack([calibrated_mean, scores_std], dim=-1)
        calibrated_confidence = self.recalibrator(features).squeeze(-1)

        return calibrated_confidence
