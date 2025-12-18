"""
Publication-Ready Visualizations for Uncertainty-Aware Recommendations.

Generates figures that are commonly expected in uncertainty quantification papers:
1. Reliability Diagram (Calibration Plot)
2. Accuracy vs Coverage Curve (Selective Prediction)
3. Uncertainty Distribution Analysis
4. Ablation Study Bar Charts
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import os

# Check for matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from .publication_metrics import PublicationMetrics


def set_publication_style():
    """Set matplotlib style for publication-quality figures."""
    if not MATPLOTLIB_AVAILABLE:
        return

    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.figsize': (6, 5),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
    })


def plot_reliability_diagram(
    metrics: PublicationMetrics,
    save_path: Optional[str] = None,
    title: str = "Reliability Diagram"
) -> Optional[str]:
    """
    Plot reliability diagram (calibration plot).

    This is THE key figure for uncertainty papers. Shows:
    - X-axis: Model confidence
    - Y-axis: Actual accuracy
    - Perfect calibration = diagonal line

    Args:
        metrics: PublicationMetrics with calibration data
        save_path: Where to save the figure
        title: Plot title

    Returns:
        Path to saved figure or None
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping reliability diagram")
        return None

    set_publication_style()

    fig, ax = plt.subplots(figsize=(6, 6))

    bins = np.array(metrics.calibration_bins)
    accuracies = np.array(metrics.calibration_accuracy)
    confidences = np.array(metrics.calibration_confidence)
    counts = np.array(metrics.calibration_counts)

    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration', alpha=0.7)

    # Plot calibration bars
    bar_width = 0.08
    valid_mask = counts > 0

    # Bar colors based on gap from perfect calibration
    gaps = np.abs(accuracies - confidences)
    colors = plt.cm.RdYlGn(1 - gaps)  # Green = good, Red = bad

    bars = ax.bar(
        bins[valid_mask],
        accuracies[valid_mask],
        width=bar_width,
        color=colors[valid_mask],
        edgecolor='black',
        linewidth=1,
        alpha=0.8,
        label='Model'
    )

    # Add gap visualization
    for i, (b, acc, conf) in enumerate(zip(bins[valid_mask], accuracies[valid_mask], confidences[valid_mask])):
        if abs(acc - conf) > 0.05:  # Only show significant gaps
            ax.plot([b, b], [acc, conf], 'r-', linewidth=2, alpha=0.5)

    # Add ECE annotation
    ax.text(
        0.05, 0.95,
        f'ECE = {metrics.expected_calibration_error:.3f}',
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.legend(loc='lower right')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Reliability diagram saved to: {save_path}")
        plt.close()
        return save_path
    else:
        plt.show()
        return None


def plot_selective_prediction_curve(
    metrics: PublicationMetrics,
    save_path: Optional[str] = None,
    baseline_accuracy: Optional[float] = None,
    title: str = "Selective Prediction Curve"
) -> Optional[str]:
    """
    Plot accuracy vs coverage curve.

    Shows the tradeoff: "If we reject uncertain predictions, how much does accuracy improve?"

    Args:
        metrics: PublicationMetrics with selective prediction data
        save_path: Where to save
        baseline_accuracy: Overall accuracy for reference line
        title: Plot title

    Returns:
        Path to saved figure or None
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping selective prediction curve")
        return None

    set_publication_style()

    fig, ax = plt.subplots(figsize=(7, 5))

    coverages = np.array(metrics.coverage_thresholds)
    accuracies = np.array(metrics.accuracy_at_coverage)

    # Main curve
    ax.plot(
        coverages * 100,
        accuracies * 100,
        'b-',
        linewidth=2.5,
        label='Uncertainty-Aware',
        marker='o',
        markersize=4,
        markevery=10
    )

    # Baseline (random ordering)
    if baseline_accuracy is not None:
        ax.axhline(
            y=baseline_accuracy * 100,
            color='gray',
            linestyle='--',
            linewidth=2,
            label=f'Baseline ({baseline_accuracy*100:.1f}%)'
        )

    # Mark key points
    # Coverage at 90% accuracy
    if metrics.coverage_at_90_accuracy > 0:
        ax.axvline(
            x=metrics.coverage_at_90_accuracy * 100,
            color='green',
            linestyle=':',
            linewidth=1.5,
            alpha=0.7
        )
        ax.scatter(
            [metrics.coverage_at_90_accuracy * 100],
            [90],
            color='green',
            s=100,
            zorder=5,
            marker='*',
            label=f'90% Acc @ {metrics.coverage_at_90_accuracy*100:.1f}% Cov'
        )

    # Fill area under curve
    ax.fill_between(coverages * 100, accuracies * 100, alpha=0.2, color='blue')

    # AUC annotation
    ax.text(
        0.95, 0.05,
        f'AUC = {metrics.auc_accuracy_coverage:.3f}',
        transform=ax.transAxes,
        fontsize=11,
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    ax.set_xlabel('Coverage (%)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(title)
    ax.set_xlim(0, 100)
    ax.set_ylim(max(0, min(accuracies) * 100 - 10), 100)
    ax.legend(loc='lower left')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Selective prediction curve saved to: {save_path}")
        plt.close()
        return save_path
    else:
        plt.show()
        return None


def plot_ablation_study(
    ablation_results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    metrics_to_show: List[str] = ['NDCG@10', 'ECE', 'Coverage@90%'],
    title: str = "Ablation Study"
) -> Optional[str]:
    """
    Plot ablation study results as grouped bar chart.

    Args:
        ablation_results: Dict mapping model variant name to metrics dict
        save_path: Where to save
        metrics_to_show: Which metrics to display
        title: Plot title

    Returns:
        Path to saved figure or None
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping ablation plot")
        return None

    set_publication_style()

    n_variants = len(ablation_results)
    n_metrics = len(metrics_to_show)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(n_variants)
    width = 0.8 / n_metrics

    colors = plt.cm.Set2(np.linspace(0, 1, n_metrics))

    for i, metric in enumerate(metrics_to_show):
        values = []
        for variant_name in ablation_results.keys():
            val = ablation_results[variant_name].get(metric, 0)
            values.append(val)

        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=metric, color=colors[i], edgecolor='black')

        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(
                f'{val:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center',
                va='bottom',
                fontsize=9
            )

    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(ablation_results.keys(), rotation=45, ha='right')
    ax.legend(loc='upper right')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Ablation study plot saved to: {save_path}")
        plt.close()
        return save_path
    else:
        plt.show()
        return None


def plot_uncertainty_distribution(
    uncertainties_correct: np.ndarray,
    uncertainties_incorrect: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Uncertainty Distribution"
) -> Optional[str]:
    """
    Plot uncertainty distributions for correct vs incorrect predictions.

    Good uncertainty: incorrect predictions should have higher uncertainty.

    Args:
        uncertainties_correct: Uncertainties for correct predictions
        uncertainties_incorrect: Uncertainties for incorrect predictions
        save_path: Where to save
        title: Plot title

    Returns:
        Path to saved figure or None
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping uncertainty distribution")
        return None

    set_publication_style()

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot histograms
    bins = np.linspace(0, max(uncertainties_correct.max(), uncertainties_incorrect.max()), 30)

    ax.hist(
        uncertainties_correct,
        bins=bins,
        alpha=0.6,
        label=f'Correct (μ={uncertainties_correct.mean():.3f})',
        color='green',
        edgecolor='black'
    )
    ax.hist(
        uncertainties_incorrect,
        bins=bins,
        alpha=0.6,
        label=f'Incorrect (μ={uncertainties_incorrect.mean():.3f})',
        color='red',
        edgecolor='black'
    )

    # Add mean lines
    ax.axvline(uncertainties_correct.mean(), color='green', linestyle='--', linewidth=2)
    ax.axvline(uncertainties_incorrect.mean(), color='red', linestyle='--', linewidth=2)

    ax.set_xlabel('Uncertainty')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Uncertainty distribution saved to: {save_path}")
        plt.close()
        return save_path
    else:
        plt.show()
        return None


def generate_all_publication_figures(
    metrics: PublicationMetrics,
    output_dir: str,
    uncertainties_correct: Optional[np.ndarray] = None,
    uncertainties_incorrect: Optional[np.ndarray] = None,
    ablation_results: Optional[Dict] = None,
    baseline_accuracy: Optional[float] = None
) -> Dict[str, str]:
    """
    Generate all publication figures at once.

    Args:
        metrics: PublicationMetrics object
        output_dir: Directory to save figures
        uncertainties_correct: Uncertainties for correct predictions
        uncertainties_incorrect: Uncertainties for incorrect predictions
        ablation_results: Ablation study results
        baseline_accuracy: Baseline accuracy for reference

    Returns:
        Dict mapping figure name to file path
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_figures = {}

    # 1. Reliability Diagram
    path = plot_reliability_diagram(
        metrics,
        save_path=os.path.join(output_dir, 'reliability_diagram.png'),
        title='Calibration (Reliability Diagram)'
    )
    if path:
        saved_figures['reliability_diagram'] = path

    # 2. Selective Prediction Curve
    path = plot_selective_prediction_curve(
        metrics,
        save_path=os.path.join(output_dir, 'selective_prediction.png'),
        baseline_accuracy=baseline_accuracy,
        title='Accuracy vs Coverage'
    )
    if path:
        saved_figures['selective_prediction'] = path

    # 3. Uncertainty Distribution (if data provided)
    if uncertainties_correct is not None and uncertainties_incorrect is not None:
        path = plot_uncertainty_distribution(
            uncertainties_correct,
            uncertainties_incorrect,
            save_path=os.path.join(output_dir, 'uncertainty_distribution.png'),
            title='Uncertainty by Prediction Correctness'
        )
        if path:
            saved_figures['uncertainty_distribution'] = path

    # 4. Ablation Study (if data provided)
    if ablation_results:
        path = plot_ablation_study(
            ablation_results,
            save_path=os.path.join(output_dir, 'ablation_study.png'),
            title='Ablation Study Results'
        )
        if path:
            saved_figures['ablation_study'] = path

    return saved_figures
