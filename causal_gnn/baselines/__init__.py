"""Baseline recommendation models for benchmarking."""

from .popular import PopularItems
from .bpr import BPR
from .ncf import NCF
from .lightgcn import LightGCN

__all__ = [
    'PopularItems',
    'BPR',
    'NCF',
    'LightGCN',
]

