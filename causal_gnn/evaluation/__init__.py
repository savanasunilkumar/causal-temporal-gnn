"""Evaluation module for publication-ready metrics and visualizations."""

from .publication_metrics import (
    PublicationMetrics,
    compute_calibration_curve,
    compute_ece,
    compute_mce,
    compute_brier_score,
    compute_selective_prediction_curve,
    compute_coverage_at_accuracy,
    compute_auc_accuracy_coverage,
    compute_uncertainty_error_correlation,
    compute_auroc_uncertainty,
    generate_publication_metrics,
    format_latex_table,
)

from .visualizations import (
    plot_reliability_diagram,
    plot_selective_prediction_curve,
    plot_ablation_study,
    plot_uncertainty_distribution,
    generate_all_publication_figures,
)

__all__ = [
    # Metrics
    'PublicationMetrics',
    'compute_calibration_curve',
    'compute_ece',
    'compute_mce',
    'compute_brier_score',
    'compute_selective_prediction_curve',
    'compute_coverage_at_accuracy',
    'compute_auc_accuracy_coverage',
    'compute_uncertainty_error_correlation',
    'compute_auroc_uncertainty',
    'generate_publication_metrics',
    'format_latex_table',
    # Visualizations
    'plot_reliability_diagram',
    'plot_selective_prediction_curve',
    'plot_ablation_study',
    'plot_uncertainty_distribution',
    'generate_all_publication_figures',
]
