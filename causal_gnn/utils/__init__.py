"""Utility modules for the UACT-GNN system."""

from .cold_start import ColdStartSolver
from .checkpointing import ModelCheckpointer
from .logging import setup_logging, get_logger
from .uncertainty import (
    UncertaintyMetrics,
    compute_calibration_metrics,
    temperature_scaling_calibration,
    compute_prediction_intervals,
    selective_prediction,
    uncertainty_decomposition,
    compute_recommendation_uncertainty_metrics,
    UncertaintyAwareEvaluator,
    should_abstain,
)

__all__ = [
    'ColdStartSolver',
    'ModelCheckpointer',
    'setup_logging',
    'get_logger',
    # Uncertainty utilities
    'UncertaintyMetrics',
    'compute_calibration_metrics',
    'temperature_scaling_calibration',
    'compute_prediction_intervals',
    'selective_prediction',
    'uncertainty_decomposition',
    'compute_recommendation_uncertainty_metrics',
    'UncertaintyAwareEvaluator',
    'should_abstain',
]

