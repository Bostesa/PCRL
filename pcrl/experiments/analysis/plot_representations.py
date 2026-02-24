#!/usr/bin/env python3
"""Plot representation visualizations using t-SNE and UMAP.

Creates visualizations showing:
1. Same inputs, different purpose tokens -> different representation clusters
2. Representation structure by sensitive attributes
3. Purpose-conditioned representation separation
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Optional imports for dimensionality reduction
try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except ImportError:
    HAS_TSNE = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def compute_representations(
    encoder: nn.Module,
    features: torch.Tensor,
    purpose_indices: list[int],
    device: str = "cpu",
) -> dict[int, np.ndarray]:
    """Compute representations for same features under different purposes.

    Args:
        encoder: Purpose-conditioned encoder.
        features: Input features tensor.
        purpose_indices: List of purpose indices to compute representations for.
        device: Device to use.

    Returns:
        Dictionary mapping purpose index to representations array.
    """
    encoder.eval()
    encoder.to(device)
    features = features.to(device)

    representations = {}
    with torch.no_grad():
        for purpose_idx in purpose_indices:
            reps = encoder(features, purpose_idx=purpose_idx)
            representations[purpose_idx] = reps.cpu().numpy()

    return representations


def reduce_dimensions(
    representations: np.ndarray,
    method: str = "tsne",
    n_components: int = 2,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> np.ndarray:
    """Reduce representation dimensions for visualization.

    Args:
        representations: High-dimensional representations.
        method: Reduction method ('tsne' or 'umap').
        n_components: Target dimensions.
        perplexity: t-SNE perplexity (only for tsne).
        random_state: Random seed.

    Returns:
        Reduced representations.
    """
    if method == "tsne":
        if not HAS_TSNE:
            raise ImportError("scikit-learn required for t-SNE. Install with: pip install scikit-learn")
        reducer = TSNE(
            n_components=n_components,
            perplexity=min(perplexity, len(representations) - 1),
            random_state=random_state,
            n_iter=1000,
        )
    elif method == "umap":
        if not HAS_UMAP:
            raise ImportError("umap-learn required for UMAP. Install with: pip install umap-learn")
        reducer = umap.UMAP(
            n_components=n_components,
            random_state=random_state,
            n_neighbors=min(15, len(representations) - 1),
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'tsne' or 'umap'.")

    return reducer.fit_transform(representations)


def plot_purpose_clusters(
    representations: dict[int, np.ndarray],
    purpose_names: list[str],
    output_path: str | Path,
    method: str = "tsne",
    title: str = "Representations by Purpose",
    max_samples: int = 1000,
) -> None:
    """Plot representations colored by purpose.

    Shows how the same inputs produce different clusters based on purpose token.

    Args:
        representations: Dict mapping purpose_idx to representations.
        purpose_names: Names of purposes.
        output_path: Path to save the plot.
        method: Dimensionality reduction method.
        title: Plot title.
        max_samples: Maximum samples to plot (for performance).
    """
    # Subsample if needed
    n_samples = min(max_samples, min(len(r) for r in representations.values()))

    # Stack all representations
    all_reps = []
    all_purposes = []
    indices = np.random.choice(len(list(representations.values())[0]), n_samples, replace=False)

    for purpose_idx, reps in representations.items():
        all_reps.append(reps[indices])
        all_purposes.extend([purpose_idx] * n_samples)

    all_reps = np.vstack(all_reps)
    all_purposes = np.array(all_purposes)

    # Reduce dimensions
    print(f"Reducing dimensions with {method}...")
    reduced = reduce_dimensions(all_reps, method=method)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(representations)))

    for i, purpose_idx in enumerate(representations.keys()):
        mask = all_purposes == purpose_idx
        name = purpose_names[purpose_idx] if purpose_idx < len(purpose_names) else f"Purpose {purpose_idx}"
        ax.scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            c=[colors[i]],
            label=name,
            alpha=0.6,
            s=20,
            edgecolors='none',
        )

    ax.set_xlabel(f'{method.upper()} 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(title='Purpose', loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved purpose cluster plot to {output_path}")


def plot_sensitive_attr_structure(
    representations: np.ndarray,
    sensitive_attrs: dict[str, np.ndarray],
    output_path: str | Path,
    method: str = "tsne",
    title: str = "Representation Structure by Sensitive Attributes",
    max_samples: int = 1000,
) -> None:
    """Plot representations colored by sensitive attributes.

    Shows whether sensitive information is encoded in representations.

    Args:
        representations: Representations array.
        sensitive_attrs: Dict mapping attr name to labels.
        output_path: Path to save the plot.
        method: Dimensionality reduction method.
        title: Plot title.
        max_samples: Maximum samples to plot.
    """
    n_samples = min(max_samples, len(representations))
    indices = np.random.choice(len(representations), n_samples, replace=False)

    reps_subset = representations[indices]

    # Reduce dimensions
    print(f"Reducing dimensions with {method}...")
    reduced = reduce_dimensions(reps_subset, method=method)

    n_attrs = len(sensitive_attrs)
    fig, axes = plt.subplots(1, n_attrs, figsize=(5 * n_attrs, 5))

    if n_attrs == 1:
        axes = [axes]

    for ax, (attr_name, attr_labels) in zip(axes, sensitive_attrs.items()):
        labels_subset = attr_labels[indices]
        unique_labels = np.unique(labels_subset)

        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            mask = labels_subset == label
            ax.scatter(
                reduced[mask, 0],
                reduced[mask, 1],
                c=[colors[i]],
                label=f'{attr_name}={label}',
                alpha=0.6,
                s=20,
                edgecolors='none',
            )

        ax.set_xlabel(f'{method.upper()} 1', fontsize=10)
        ax.set_ylabel(f'{method.upper()} 2', fontsize=10)
        ax.set_title(f'Colored by {attr_name}', fontsize=12)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved sensitive attribute structure plot to {output_path}")


def plot_purpose_comparison(
    representations: dict[int, np.ndarray],
    sensitive_attrs: dict[str, np.ndarray],
    purpose_names: list[str],
    output_path: str | Path,
    method: str = "tsne",
    attr_to_show: str | None = None,
    max_samples: int = 500,
) -> None:
    """Plot side-by-side comparison of representations under different purposes.

    Each subplot shows representations for one purpose, colored by a sensitive attribute.

    Args:
        representations: Dict mapping purpose_idx to representations.
        sensitive_attrs: Dict mapping attr name to labels.
        purpose_names: Names of purposes.
        output_path: Path to save the plot.
        method: Dimensionality reduction method.
        attr_to_show: Sensitive attribute to color by. If None, uses first.
        max_samples: Maximum samples per purpose.
    """
    if attr_to_show is None:
        attr_to_show = list(sensitive_attrs.keys())[0]

    attr_labels = sensitive_attrs[attr_to_show]
    n_purposes = len(representations)
    n_samples = min(max_samples, min(len(r) for r in representations.values()), len(attr_labels))
    indices = np.random.choice(len(attr_labels), n_samples, replace=False)

    # Create subplots
    fig, axes = plt.subplots(1, n_purposes, figsize=(5 * n_purposes, 5))
    if n_purposes == 1:
        axes = [axes]

    unique_labels = np.unique(attr_labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))

    for ax, (purpose_idx, reps) in zip(axes, representations.items()):
        reps_subset = reps[indices]
        labels_subset = attr_labels[indices]

        # Reduce dimensions for this purpose
        reduced = reduce_dimensions(reps_subset, method=method)

        for i, label in enumerate(unique_labels):
            mask = labels_subset == label
            ax.scatter(
                reduced[mask, 0],
                reduced[mask, 1],
                c=[colors[i]],
                label=f'{attr_to_show}={label}',
                alpha=0.6,
                s=30,
                edgecolors='none',
            )

        name = purpose_names[purpose_idx] if purpose_idx < len(purpose_names) else f"Purpose {purpose_idx}"
        ax.set_xlabel(f'{method.upper()} 1', fontsize=10)
        ax.set_ylabel(f'{method.upper()} 2', fontsize=10)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Purpose-Specific Representations (colored by {attr_to_show})',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved purpose comparison plot to {output_path}")


def plot_representation_distances(
    representations: dict[int, np.ndarray],
    purpose_names: list[str],
    output_path: str | Path,
    title: str = "Representation Distance Matrix Between Purposes",
) -> None:
    """Plot heatmap of average distances between purpose representations.

    Args:
        representations: Dict mapping purpose_idx to representations.
        purpose_names: Names of purposes.
        output_path: Path to save the plot.
        title: Plot title.
    """
    purpose_indices = list(representations.keys())
    n_purposes = len(purpose_indices)

    # Compute pairwise distances
    distance_matrix = np.zeros((n_purposes, n_purposes))

    for i, idx_i in enumerate(purpose_indices):
        for j, idx_j in enumerate(purpose_indices):
            reps_i = representations[idx_i]
            reps_j = representations[idx_j]
            # Mean squared difference per sample, then average
            distances = np.mean((reps_i - reps_j) ** 2, axis=1)
            distance_matrix[i, j] = np.mean(distances)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, 6))

    names = [purpose_names[idx] if idx < len(purpose_names) else f"Purpose {idx}"
             for idx in purpose_indices]

    im = ax.imshow(distance_matrix, cmap='viridis', aspect='auto')

    ax.set_xticks(range(n_purposes))
    ax.set_yticks(range(n_purposes))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_yticklabels(names)

    # Add value annotations
    for i in range(n_purposes):
        for j in range(n_purposes):
            text = ax.text(j, i, f'{distance_matrix[i, j]:.3f}',
                          ha='center', va='center', color='white' if distance_matrix[i, j] > distance_matrix.max() / 2 else 'black',
                          fontsize=10, fontweight='bold')

    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Mean Squared Distance')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved representation distance matrix to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot representation visualizations")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--output-dir", type=str, default="figures", help="Output directory")
    parser.add_argument("--method", type=str, default="tsne", choices=["tsne", "umap"])
    parser.add_argument("--max-samples", type=int, default=1000, help="Max samples to plot")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {args.model}")
    checkpoint = torch.load(args.model, map_location=args.device)

    # This would need to be adapted to your specific model loading
    # For demonstration, we'll create placeholder instructions
    print("\nNote: This script requires loading your trained PCRL model.")
    print("Adapt the model loading section to your specific checkpoint format.")
    print("\nExpected usage pattern:")
    print("  1. Load encoder from checkpoint")
    print("  2. Load test data")
    print("  3. Compute representations for each purpose")
    print("  4. Generate visualizations")

    print(f"\nOutput directory: {output_dir}")
    print(f"Dimensionality reduction method: {args.method}")


if __name__ == "__main__":
    main()
