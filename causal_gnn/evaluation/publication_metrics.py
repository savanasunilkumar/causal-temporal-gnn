"""
Publication-Ready Evaluation Metrics for Uncertainty-Aware Recommendations.

This module generates all metrics and visualizations needed for a research paper:
1. Calibration plots (reliability diagrams)
2. Selective prediction curves (accuracy vs coverage)
3. Uncertainty-error correlation analysis
4. Ablation study metrics

These are the metrics reviewers expect to see for uncertainty quantification papers.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class PublicationMetrics:
    """Container for all publication-ready metrics."""
    # Standard recommendation metrics
    ndcg: Dict[int, float] = field(default_factory=dict)
    recall: Dict[int, float] = field(default_factory=dict)
    precision: Dict[int, float] = field(default_factory=dict)
    hit_ratio: Dict[int, float] = field(default_factory=dict)

    # Calibration metrics
    expected_calibration_error: float = 0.0
    maximum_calibration_error: float = 0.0
    brier_score: float = 0.0

    # Selective prediction metrics
    coverage_at_90_accuracy: float = 0.0
    coverage_at_95_accuracy: float = 0.0
    auc_accuracy_coverage: float = 0.0

    # Uncertainty quality metrics
    uncertainty_error_correlation: float = 0.0
    auroc_uncertainty: float = 0.0

    # Calibration curve data (for plotting)
    calibration_bins: List[float] = field(default_factory=list)
    calibration_accuracy: List[float] = field(default_factory=list)
    calibration_confidence: List[float] = field(default_factory=list)
    calibration_counts: List[int] = field(default_factory=list)

    # Selective prediction curve data (for plotting)
    coverage_thresholds: List[float] = field(default_factory=list)
    accuracy_at_coverage: List[float] = field(default_factory=list)


def compute_calibration_curve(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute calibration curve data for reliability diagram.

    A well-calibrated model has confidence ≈ accuracy for each bin.

    Args:
        confidences: Model confidence scores [0, 1]
        accuracies: Binary accuracy (1 if correct, 0 if wrong)
        n_bins: Number of bins for calibration

    Returns:
        Tuple of (bin_centers, bin_accuracies, bin_confidences, bin_counts)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences >= low) & (confidences < high)

        if mask.sum() > 0:
            bin_accuracies.append(accuracies[mask].mean())
            bin_confidences.append(confidences[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(bin_centers[i])
            bin_counts.append(0)

    return (
        np.array(bin_centers),
        np.array(bin_accuracies),
        np.array(bin_confidences),
        np.array(bin_counts)
    )


def compute_ece(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE = Σ (|bin_count| / total) * |accuracy - confidence|

    Lower is better. Perfect calibration = 0.
    """
    _, bin_accuracies, bin_confidences, bin_counts = compute_calibration_curve(
        confidences, accuracies, n_bins
    )

    total = sum(bin_counts)
    if total == 0:
        return 0.0

    ece = 0.0
    for acc, conf, count in zip(bin_accuracies, bin_confidences, bin_counts):
        ece += (count / total) * abs(acc - conf)

    return ece


def compute_mce(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Maximum Calibration Error (MCE).

    MCE = max over bins of |accuracy - confidence|

    Lower is better.
    """
    _, bin_accuracies, bin_confidences, bin_counts = compute_calibration_curve(
        confidences, accuracies, n_bins
    )

    mce = 0.0
    for acc, conf, count in zip(bin_accuracies, bin_confidences, bin_counts):
        if count > 0:  # Only consider non-empty bins
            mce = max(mce, abs(acc - conf))

    return mce


def compute_brier_score(
    confidences: np.ndarray,
    accuracies: np.ndarray
) -> float:
    """
    Compute Brier Score.

    Brier = mean((confidence - accuracy)^2)

    Lower is better. Range [0, 1].
    """
    return np.mean((confidences - accuracies) ** 2)


def compute_selective_prediction_curve(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute selective prediction curve (accuracy vs coverage).

    Shows: If we only make predictions when confidence > threshold,
    what accuracy do we achieve and what fraction of data do we cover?

    Args:
        confidences: Model confidence scores
        accuracies: Binary accuracy values
        n_points: Number of points on the curve

    Returns:
        Tuple of (coverage_values, accuracy_values)
    """
    # Sort by confidence (descending)
    sorted_indices = np.argsort(-confidences)
    sorted_accuracies = accuracies[sorted_indices]

    coverages = []
    accuracies_at_coverage = []

    n = len(confidences)
    for i in range(1, n + 1, max(1, n // n_points)):
        coverage = i / n
        acc = sorted_accuracies[:i].mean()
        coverages.append(coverage)
        accuracies_at_coverage.append(acc)

    # Ensure we include 100% coverage
    if coverages[-1] != 1.0:
        coverages.append(1.0)
        accuracies_at_coverage.append(sorted_accuracies.mean())

    return np.array(coverages), np.array(accuracies_at_coverage)


def compute_coverage_at_accuracy(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    target_accuracy: float = 0.9
) -> float:
    """
    Compute coverage when maintaining target accuracy.

    "If we want 90% accuracy, what fraction of users can we serve?"

    Higher is better.
    """
    coverages, accs = compute_selective_prediction_curve(confidences, accuracies)

    # Find largest coverage where accuracy >= target
    for i in range(len(coverages) - 1, -1, -1):
        if accs[i] >= target_accuracy:
            return coverages[i]

    return 0.0  # Can't achieve target accuracy


def compute_auc_accuracy_coverage(
    confidences: np.ndarray,
    accuracies: np.ndarray
) -> float:
    """
    Compute Area Under the Accuracy-Coverage Curve.

    Higher is better. Perfect model = 1.0 (100% accuracy at all coverage levels).
    Random baseline ≈ overall accuracy.
    """
    coverages, accs = compute_selective_prediction_curve(confidences, accuracies)

    # Trapezoidal integration
    auc = np.trapz(accs, coverages)
    return auc


def compute_uncertainty_error_correlation(
    uncertainties: np.ndarray,
    errors: np.ndarray
) -> float:
    """
    Compute correlation between uncertainty and prediction error.

    A good uncertainty estimate should correlate positively with errors.
    Higher is better (up to ~0.7-0.8 is excellent).
    """
    if len(uncertainties) < 2:
        return 0.0

    # Pearson correlation
    corr = np.corrcoef(uncertainties, errors)[0, 1]

    if np.isnan(corr):
        return 0.0

    return corr


def compute_auroc_uncertainty(
    uncertainties: np.ndarray,
    errors: np.ndarray
) -> float:
    """
    Compute AUROC for uncertainty as a predictor of errors.

    "Can uncertainty scores distinguish correct from incorrect predictions?"

    Higher is better. Random = 0.5, Perfect = 1.0.
    """
    # Binary: is this prediction an error?
    is_error = (errors > 0).astype(int)

    if is_error.sum() == 0 or is_error.sum() == len(is_error):
        return 0.5  # All same class

    # Sort by uncertainty (descending)
    sorted_indices = np.argsort(-uncertainties)
    sorted_errors = is_error[sorted_indices]

    # Compute AUROC via ranking
    n_pos = is_error.sum()
    n_neg = len(is_error) - n_pos

    # Wilcoxon-Mann-Whitney statistic
    rank_sum = 0
    for i, is_err in enumerate(sorted_errors):
        if is_err:
            rank_sum += i + 1

    auroc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    return auroc


def generate_publication_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    confidences: np.ndarray,
    uncertainties: np.ndarray,
    recommendation_metrics: Optional[Dict] = None,
    n_calibration_bins: int = 10
) -> PublicationMetrics:
    """
    Generate all publication-ready metrics.

    Args:
        predictions: Model predictions (binary: 1 if recommended item is relevant)
        labels: Ground truth labels
        confidences: Model confidence scores [0, 1]
        uncertainties: Model uncertainty scores (higher = more uncertain)
        recommendation_metrics: Dict with NDCG, Recall, etc. from standard evaluation
        n_calibration_bins: Number of bins for calibration

    Returns:
        PublicationMetrics with all computed values
    """
    metrics = PublicationMetrics()

    # Copy standard recommendation metrics if provided
    if recommendation_metrics:
        metrics.ndcg = recommendation_metrics.get('ndcg', {})
        metrics.recall = recommendation_metrics.get('recall', {})
        metrics.precision = recommendation_metrics.get('precision', {})
        metrics.hit_ratio = recommendation_metrics.get('hit_ratio', {})

    # Compute accuracies (1 if prediction matches label)
    accuracies = (predictions == labels).astype(float)
    errors = 1 - accuracies

    # Calibration metrics
    bin_centers, bin_accs, bin_confs, bin_counts = compute_calibration_curve(
        confidences, accuracies, n_calibration_bins
    )

    metrics.calibration_bins = bin_centers.tolist()
    metrics.calibration_accuracy = bin_accs.tolist()
    metrics.calibration_confidence = bin_confs.tolist()
    metrics.calibration_counts = [int(c) for c in bin_counts]

    metrics.expected_calibration_error = compute_ece(confidences, accuracies, n_calibration_bins)
    metrics.maximum_calibration_error = compute_mce(confidences, accuracies, n_calibration_bins)
    metrics.brier_score = compute_brier_score(confidences, accuracies)

    # Selective prediction metrics
    coverages, accs_at_cov = compute_selective_prediction_curve(confidences, accuracies)
    metrics.coverage_thresholds = coverages.tolist()
    metrics.accuracy_at_coverage = accs_at_cov.tolist()

    metrics.coverage_at_90_accuracy = compute_coverage_at_accuracy(confidences, accuracies, 0.9)
    metrics.coverage_at_95_accuracy = compute_coverage_at_accuracy(confidences, accuracies, 0.95)
    metrics.auc_accuracy_coverage = compute_auc_accuracy_coverage(confidences, accuracies)

    # Uncertainty quality metrics
    metrics.uncertainty_error_correlation = compute_uncertainty_error_correlation(uncertainties, errors)
    metrics.auroc_uncertainty = compute_auroc_uncertainty(uncertainties, errors)

    return metrics


def format_latex_table(metrics: PublicationMetrics, model_name: str = "Ours") -> str:
    """
    Format metrics as a LaTeX table row for paper.
    """
    latex = f"""
% Add to your LaTeX table:
{model_name} & {metrics.ndcg.get(10, 0):.4f} & {metrics.recall.get(10, 0):.4f} & """
    latex += f"{metrics.expected_calibration_error:.4f} & {metrics.coverage_at_90_accuracy:.2%} & "
    latex += f"{metrics.uncertainty_error_correlation:.3f} \\\\"

    return latex
