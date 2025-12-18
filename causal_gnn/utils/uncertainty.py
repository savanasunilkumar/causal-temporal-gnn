"""Uncertainty quantification utilities for recommendations.

This module provides tools for:
1. Computing and interpreting uncertainty metrics
2. Calibrating confidence scores
3. Visualizing uncertainty
4. Uncertainty-based decision making

Novel contribution: Comprehensive uncertainty toolkit for
GNN-based recommendation systems.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats


@dataclass
class UncertaintyMetrics:
    """Container for uncertainty evaluation metrics."""
    expected_calibration_error: float
    maximum_calibration_error: float
    brier_score: float
    negative_log_likelihood: float
    mean_confidence: float
    mean_uncertainty: float
    uncertainty_correlation: float  # Correlation between uncertainty and error


def compute_calibration_metrics(
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    ground_truth: np.ndarray,
    n_bins: int = 10
) -> UncertaintyMetrics:
    """
    Compute calibration metrics for uncertainty estimates.

    A well-calibrated model should have confidence scores that match
    the empirical accuracy: if confidence is 0.8, the model should
    be correct 80% of the time.

    Args:
        predictions: Predicted labels or scores
        uncertainties: Uncertainty estimates (higher = less confident)
        ground_truth: True labels
        n_bins: Number of bins for calibration

    Returns:
        UncertaintyMetrics with calibration statistics
    """
    # Convert uncertainties to confidence scores
    confidences = 1 / (1 + uncertainties)

    # Compute accuracy
    if predictions.ndim > 1:
        correct = (predictions.argmax(axis=1) == ground_truth).astype(float)
    else:
        correct = (predictions > 0.5) == ground_truth
        correct = correct.astype(float)

    # Bin by confidence
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bin_boundaries[1:-1])

    ece = 0.0
    mce = 0.0

    for i in range(n_bins):
        in_bin = bin_indices == i
        if in_bin.sum() > 0:
            bin_confidence = confidences[in_bin].mean()
            bin_accuracy = correct[in_bin].mean()
            bin_size = in_bin.sum() / len(confidences)

            calibration_error = abs(bin_accuracy - bin_confidence)
            ece += bin_size * calibration_error
            mce = max(mce, calibration_error)

    # Brier score
    if predictions.ndim > 1:
        brier = np.mean((predictions.max(axis=1) - correct) ** 2)
    else:
        brier = np.mean((predictions - correct) ** 2)

    # Negative log likelihood
    eps = 1e-7
    if predictions.ndim > 1:
        probs = predictions[np.arange(len(ground_truth)), ground_truth]
    else:
        probs = np.where(ground_truth == 1, predictions, 1 - predictions)
    nll = -np.mean(np.log(np.clip(probs, eps, 1 - eps)))

    # Uncertainty-error correlation (should be positive for good uncertainty)
    errors = 1 - correct
    correlation = np.corrcoef(uncertainties, errors)[0, 1]

    return UncertaintyMetrics(
        expected_calibration_error=ece,
        maximum_calibration_error=mce,
        brier_score=brier,
        negative_log_likelihood=nll,
        mean_confidence=float(confidences.mean()),
        mean_uncertainty=float(uncertainties.mean()),
        uncertainty_correlation=correlation if not np.isnan(correlation) else 0.0
    )


def temperature_scaling_calibration(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_iterations: int = 100,
    lr: float = 0.01
) -> float:
    """
    Learn optimal temperature for calibration using temperature scaling.

    Temperature scaling is a simple but effective post-hoc calibration method
    that scales the logits by a learned temperature parameter.

    Args:
        logits: Model output logits [n_samples, n_classes] or [n_samples]
        labels: True labels [n_samples]
        n_iterations: Number of optimization iterations
        lr: Learning rate

    Returns:
        Optimal temperature value
    """
    temperature = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=n_iterations)
    criterion = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        scaled_logits = logits / temperature
        loss = criterion(scaled_logits, labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    return temperature.item()


def compute_prediction_intervals(
    mean: np.ndarray,
    variance: np.ndarray,
    confidence_level: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute prediction intervals from mean and variance.

    Args:
        mean: Predicted mean values
        variance: Predicted variances
        confidence_level: Confidence level for intervals

    Returns:
        Tuple of (lower_bounds, upper_bounds)
    """
    std = np.sqrt(variance)
    z_score = stats.norm.ppf((1 + confidence_level) / 2)

    lower = mean - z_score * std
    upper = mean + z_score * std

    return lower, upper


def selective_prediction(
    predictions: np.ndarray,
    uncertainties: np.ndarray,
    threshold: float
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Make selective predictions by abstaining on high-uncertainty samples.

    This is useful when it's better to say "I don't know" than to make
    a potentially wrong prediction.

    Args:
        predictions: Model predictions
        uncertainties: Uncertainty estimates
        threshold: Maximum uncertainty for making a prediction

    Returns:
        Tuple of (selected_predictions, selection_mask, coverage)
    """
    selection_mask = uncertainties <= threshold
    coverage = selection_mask.mean()

    selected_predictions = predictions.copy()
    selected_predictions[~selection_mask] = np.nan

    return selected_predictions, selection_mask, coverage


def uncertainty_decomposition(
    mc_samples: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Decompose total uncertainty into epistemic and aleatoric components.

    Uses the law of total variance:
    Var(Y) = E[Var(Y|X)] + Var(E[Y|X])
           = Aleatoric   + Epistemic

    For MC dropout samples, epistemic uncertainty is the variance of the means.

    Args:
        mc_samples: Monte Carlo samples [n_samples, n_data, n_output]

    Returns:
        Dictionary with 'epistemic', 'aleatoric', and 'total' uncertainty
    """
    # Mean prediction across MC samples
    mean_prediction = mc_samples.mean(axis=0)

    # Epistemic: variance of means (model uncertainty)
    epistemic = mc_samples.var(axis=0)

    # For classification, aleatoric can be estimated from predictive entropy
    # For regression, we'd need to model it explicitly
    if mc_samples.shape[-1] > 1:  # Classification
        # Predictive entropy as total uncertainty
        probs = np.exp(mc_samples) / np.exp(mc_samples).sum(axis=-1, keepdims=True)
        mean_probs = probs.mean(axis=0)
        total = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=-1)

        # Aleatoric: expected entropy
        entropies = -np.sum(probs * np.log(probs + 1e-10), axis=-1)
        aleatoric = entropies.mean(axis=0)
    else:  # Regression
        total = epistemic
        aleatoric = np.zeros_like(epistemic)

    return {
        'epistemic': epistemic,
        'aleatoric': aleatoric,
        'total': total,
        'mean': mean_prediction
    }


def compute_recommendation_uncertainty_metrics(
    user_embeddings_mean: torch.Tensor,
    user_embeddings_var: torch.Tensor,
    item_embeddings_mean: torch.Tensor,
    item_embeddings_var: torch.Tensor,
    interactions: List[Tuple[int, int, float]]
) -> Dict[str, float]:
    """
    Compute uncertainty metrics specific to recommendations.

    Args:
        user_embeddings_mean: Mean user embeddings
        user_embeddings_var: Variance of user embeddings
        item_embeddings_mean: Mean item embeddings
        item_embeddings_var: Variance of item embeddings
        interactions: List of (user_idx, item_idx, rating) tuples

    Returns:
        Dictionary of uncertainty metrics
    """
    metrics = {}

    # Average user uncertainty
    user_uncertainty = user_embeddings_var.mean(dim=-1)
    metrics['avg_user_uncertainty'] = user_uncertainty.mean().item()
    metrics['max_user_uncertainty'] = user_uncertainty.max().item()

    # Average item uncertainty
    item_uncertainty = item_embeddings_var.mean(dim=-1)
    metrics['avg_item_uncertainty'] = item_uncertainty.mean().item()
    metrics['max_item_uncertainty'] = item_uncertainty.max().item()

    # Compute score uncertainties for interactions
    if interactions:
        score_uncertainties = []
        for user_idx, item_idx, _ in interactions:
            # Score variance approximation
            u_mean = user_embeddings_mean[user_idx]
            u_var = user_embeddings_var[user_idx]
            i_mean = item_embeddings_mean[item_idx]
            i_var = item_embeddings_var[item_idx]

            # Var(dot product) ≈ sum of element-wise variance products
            score_var = (u_var * i_mean**2 + i_var * u_mean**2 + u_var * i_var).sum()
            score_uncertainties.append(score_var.item())

        metrics['avg_interaction_uncertainty'] = np.mean(score_uncertainties)
        metrics['std_interaction_uncertainty'] = np.std(score_uncertainties)

    return metrics


class UncertaintyAwareEvaluator:
    """
    Evaluator that considers uncertainty in recommendation metrics.

    Novel metrics:
    1. Uncertainty-weighted precision/recall
    2. Confident coverage (what fraction of catalog can we recommend confidently)
    3. Risk-adjusted performance
    """

    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold

    def evaluate_with_uncertainty(
        self,
        recommendations: Dict[int, List[Tuple[int, float, float]]],  # user -> [(item, score, confidence)]
        ground_truth: Dict[int, set],  # user -> set of relevant items
        k: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate recommendations considering uncertainty.

        Args:
            recommendations: Dict mapping user to list of (item, score, confidence)
            ground_truth: Dict mapping user to set of relevant items
            k: Number of recommendations to consider

        Returns:
            Dictionary of uncertainty-aware metrics
        """
        metrics = {
            'precision': [],
            'recall': [],
            'confident_precision': [],
            'confident_coverage': [],
            'uncertainty_correlation': [],
        }

        for user, recs in recommendations.items():
            if user not in ground_truth:
                continue

            true_items = ground_truth[user]
            top_k = recs[:k]

            # Standard metrics
            rec_items = [r[0] for r in top_k]
            hits = len(set(rec_items) & true_items)
            metrics['precision'].append(hits / k)
            metrics['recall'].append(hits / len(true_items) if true_items else 0)

            # Confident precision (only count confident predictions)
            confident_recs = [r for r in top_k if r[2] >= self.confidence_threshold]
            if confident_recs:
                confident_items = [r[0] for r in confident_recs]
                confident_hits = len(set(confident_items) & true_items)
                metrics['confident_precision'].append(confident_hits / len(confident_recs))
            else:
                metrics['confident_precision'].append(0.0)

            # Confident coverage
            metrics['confident_coverage'].append(len(confident_recs) / k)

            # Correlation between confidence and correctness
            if len(top_k) > 1:
                correct = [1 if r[0] in true_items else 0 for r in top_k]
                confidence = [r[2] for r in top_k]
                if np.std(correct) > 0 and np.std(confidence) > 0:
                    corr = np.corrcoef(confidence, correct)[0, 1]
                    if not np.isnan(corr):
                        metrics['uncertainty_correlation'].append(corr)

        return {
            'precision@k': np.mean(metrics['precision']),
            'recall@k': np.mean(metrics['recall']),
            'confident_precision@k': np.mean(metrics['confident_precision']),
            'confident_coverage': np.mean(metrics['confident_coverage']),
            'uncertainty_calibration': np.mean(metrics['uncertainty_correlation']) if metrics['uncertainty_correlation'] else 0.0,
        }


def should_abstain(
    uncertainty: float,
    threshold: float,
    min_confidence: float = 0.3
) -> Tuple[bool, str]:
    """
    Decide whether to abstain from making a recommendation.

    Returns a decision and explanation.

    Args:
        uncertainty: Predicted uncertainty
        threshold: Uncertainty threshold
        min_confidence: Minimum confidence to recommend

    Returns:
        Tuple of (should_abstain, explanation)
    """
    confidence = 1 / (1 + uncertainty)

    if uncertainty > threshold:
        return True, f"High uncertainty ({uncertainty:.3f}). Need more data."
    elif confidence < min_confidence:
        return True, f"Low confidence ({confidence:.3f}). Prediction unreliable."
    else:
        return False, f"Confident prediction (confidence: {confidence:.3f})"
