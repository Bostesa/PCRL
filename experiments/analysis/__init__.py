"""Analysis and visualization utilities for PCRL experiments.

This module provides plotting utilities for:
- CVR (Compliance Violation Rate) analysis
- Representation visualizations (t-SNE/UMAP)
- Training curves and metrics
- Purpose switching heatmaps
"""

from .plot_cvr import (
    plot_cvr_by_purpose,
    plot_cvr_per_attribute,
    plot_cvr_comparison,
    plot_task_vs_fairness,
)
from .plot_training import (
    plot_loss_curves,
    plot_auditor_accuracy,
    plot_task_accuracy,
    plot_combined_metrics,
)
from .plot_purpose_switching import (
    plot_attribute_recoverability_heatmap,
    plot_cvr_heatmap,
    plot_purpose_switching_demo,
)

__all__ = [
    # CVR plots
    "plot_cvr_by_purpose",
    "plot_cvr_per_attribute",
    "plot_cvr_comparison",
    "plot_task_vs_fairness",
    # Training plots
    "plot_loss_curves",
    "plot_auditor_accuracy",
    "plot_task_accuracy",
    "plot_combined_metrics",
    # Purpose switching plots
    "plot_attribute_recoverability_heatmap",
    "plot_cvr_heatmap",
    "plot_purpose_switching_demo",
]
