"""
Causal Temporal Graph Neural Network (UACT-GNN) for recommendation systems.

A production-ready, scalable recommendation system that combines:
- Causal discovery (Granger causality, PC algorithm)
- Temporal modeling with transformers and attention
- Multi-modal learning (text, images, numeric, categorical)
- Graph neural networks for relationship modeling
- Zero-shot cold start handling
"""

__version__ = "1.0.0"

from .config import Config
from .models.uact_gnn import CausalTemporalGNN
from .training.trainer import RecommendationSystem

__all__ = [
    'Config',
    'CausalTemporalGNN',
    'RecommendationSystem',
]

