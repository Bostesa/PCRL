"""Visualization utilities for PCRL evaluation.

Provides:
1. plot_purpose_separation - t-SNE per purpose, colored by sensitive attr
2. plot_cvr_comparison - bar chart comparing CVR across methods
3. plot_certificate_heatmap - heatmap of linear R² (purposes x attrs)
4. plot_task_privacy_tradeoff - task accuracy vs CVR Pareto frontier
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from pcrl.evaluation.certificates import ComplianceReport
from pcrl.models.encoder import PurposeConditionedEncoder


def plot_purpose_separation(
    encoder: PurposeConditionedEncoder,
    features: torch.Tensor,
    sensitive_attrs: dict[str, torch.Tensor],
    purpose_names: list[str],
    save_path: str | Path | None = None,
    perplexity: float = 30.0,
    n_samples: int = 2000,
) -> Any:
    """t-SNE of representations, one subplot per purpose.

    Color by sensitive attribute — good representations show NO clustering
    by sensitive attr.

    Args:
        encoder: Trained encoder.
        features: Input features (N, D).
        sensitive_attrs: Dict mapping attr name to labels (N,).
        purpose_names: Names for each purpose (length = num_purposes).
        save_path: Where to save the figure. Shows interactively if None.
        perplexity: t-SNE perplexity.
        n_samples: Max samples to use (for speed).

    Returns:
        matplotlib figure.
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    encoder.eval()
    device = next(encoder.parameters()).device

    # Subsample if needed
    n = min(n_samples, features.shape[0])
    idx = torch.randperm(features.shape[0])[:n]
    x = features[idx].to(device)

    # Pick the first sensitive attr for coloring
    attr_name = list(sensitive_attrs.keys())[0]
    labels = sensitive_attrs[attr_name][idx].numpy()

    num_purposes = len(purpose_names)
    fig, axes = plt.subplots(
        1, num_purposes, figsize=(5 * num_purposes, 4.5), squeeze=False
    )

    with torch.no_grad():
        for p_idx, p_name in enumerate(purpose_names):
            h = encoder(x, p_idx).cpu().numpy()

            tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                random_state=42,
                max_iter=500,
            )
            emb = tsne.fit_transform(h)

            ax = axes[0, p_idx]
            scatter = ax.scatter(
                emb[:, 0],
                emb[:, 1],
                c=labels,
                cmap="Set1",
                alpha=0.5,
                s=8,
            )
            ax.set_title(f"Purpose: {p_name}\n(colored by {attr_name})")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(scatter, ax=ax, fraction=0.046)

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_cvr_comparison(
    results_dict: dict[str, dict[str, float]],
    save_path: str | Path | None = None,
) -> Any:
    """Bar chart comparing CVR across methods / purposes.

    Args:
        results_dict: Nested dict: method_name -> purpose_name -> CVR value.
            For single-method use, set method_name = "PCRL".
        save_path: Where to save the figure.

    Returns:
        matplotlib figure.
    """
    import matplotlib.pyplot as plt

    methods = list(results_dict.keys())
    # Collect all purpose names
    all_purposes: list[str] = []
    for m in methods:
        for p in results_dict[m]:
            if p not in all_purposes:
                all_purposes.append(p)

    x = np.arange(len(all_purposes))
    width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(max(8, len(all_purposes) * 2), 5))

    for i, method in enumerate(methods):
        vals = [results_dict[method].get(p, 0.0) for p in all_purposes]
        offset = (i - len(methods) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=method, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel("Purpose")
    ax.set_ylabel("CVR")
    ax.set_title("Compliance Violation Rate Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(all_purposes, rotation=15, ha="right")
    ax.set_ylim(0, min(1.0, max(max(results_dict[m].get(p, 0) for p in all_purposes) for m in methods) + 0.15))
    ax.legend()
    ax.axhline(y=0.05, color="red", linestyle="--", alpha=0.4, label="margin=0.05")

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_certificate_heatmap(
    reports: list[ComplianceReport],
    save_path: str | Path | None = None,
) -> Any:
    """Heatmap of linear R² — rows = purposes, columns = disallowed attrs.

    Green = low R² (certified), Red = high R² (violation).
    This is the key figure for the paper.

    Args:
        reports: List of ComplianceReport objects.
        save_path: Where to save the figure.

    Returns:
        matplotlib figure.
    """
    import matplotlib.pyplot as plt

    # Collect unique purposes and attrs
    purposes: list[str] = []
    attrs: list[str] = []
    for r in reports:
        if r.purpose_name not in purposes:
            purposes.append(r.purpose_name)
        if r.attr_name not in attrs:
            attrs.append(r.attr_name)

    # Build matrix
    matrix = np.full((len(purposes), len(attrs)), np.nan)
    cert_matrix = np.full((len(purposes), len(attrs)), False)

    for r in reports:
        i = purposes.index(r.purpose_name)
        j = attrs.index(r.attr_name)
        matrix[i, j] = r.linear_r2
        cert_matrix[i, j] = r.certified

    fig, ax = plt.subplots(figsize=(max(6, len(attrs) * 1.5), max(4, len(purposes) * 0.8)))

    # Use RdYlGn_r: green=low, red=high
    im = ax.imshow(
        matrix,
        cmap="RdYlGn_r",
        vmin=0,
        vmax=max(0.5, np.nanmax(matrix) if not np.all(np.isnan(matrix)) else 0.5),
        aspect="auto",
    )

    # Add text annotations
    for i in range(len(purposes)):
        for j in range(len(attrs)):
            if not np.isnan(matrix[i, j]):
                status = "PASS" if cert_matrix[i, j] else "FAIL"
                color = "white" if matrix[i, j] > 0.25 else "black"
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.3f}\n{status}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=color,
                    fontweight="bold",
                )
            else:
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=9, color="gray")

    ax.set_xticks(range(len(attrs)))
    ax.set_xticklabels(attrs, rotation=30, ha="right")
    ax.set_yticks(range(len(purposes)))
    ax.set_yticklabels(purposes)
    ax.set_xlabel("Disallowed Attribute")
    ax.set_ylabel("Purpose")
    ax.set_title("Compliance Certificate Heatmap (Linear R²)")

    fig.colorbar(im, ax=ax, label="Linear R²", fraction=0.046)

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_task_privacy_tradeoff(
    results_at_different_lambdas: dict[float, dict[str, tuple[float, float]]],
    save_path: str | Path | None = None,
) -> Any:
    """Task accuracy vs CVR as lambda_adv varies.

    Shows the Pareto frontier — the tradeoff between task performance and
    privacy protection.

    Args:
        results_at_different_lambdas: Dict mapping lambda_adv to
            {purpose_name: (task_accuracy, cvr)}.
        save_path: Where to save the figure.

    Returns:
        matplotlib figure.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))

    # Collect per-purpose traces
    purposes: set[str] = set()
    for lam_results in results_at_different_lambdas.values():
        purposes.update(lam_results.keys())

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(purposes), 1)))

    for color, purpose_name in zip(colors, sorted(purposes)):
        task_accs: list[float] = []
        cvrs: list[float] = []
        lambdas: list[float] = []

        for lam in sorted(results_at_different_lambdas.keys()):
            if purpose_name in results_at_different_lambdas[lam]:
                ta, cv = results_at_different_lambdas[lam][purpose_name]
                task_accs.append(ta)
                cvrs.append(cv)
                lambdas.append(lam)

        ax.plot(
            task_accs,
            cvrs,
            "o-",
            color=color,
            label=purpose_name,
            markersize=6,
        )

        # Annotate lambda values
        for ta, cv, lam in zip(task_accs, cvrs, lambdas):
            ax.annotate(
                f"λ={lam}",
                (ta, cv),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
                alpha=0.7,
            )

    ax.set_xlabel("Task Accuracy")
    ax.set_ylabel("CVR (lower = better privacy)")
    ax.set_title("Task–Privacy Tradeoff (Pareto Frontier)")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0.05, color="red", linestyle="--", alpha=0.3, label="CVR margin")

    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
