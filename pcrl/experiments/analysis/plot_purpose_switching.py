#!/usr/bin/env python3
"""Plot purpose switching heatmaps.

Creates visualizations showing:
1. Heatmap of attribute recoverability under each purpose
2. Diagonal = allowed tasks (high accuracy expected)
3. Off-diagonal = disallowed attributes (low accuracy expected)
4. Purpose-attribute compliance matrix
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def load_results(results_path: str | Path) -> dict[str, Any]:
    """Load results from JSON file."""
    with open(results_path) as f:
        return json.load(f)


def plot_attribute_recoverability_heatmap(
    probe_results: dict[str, dict[str, Any]],
    output_path: str | Path,
    title: str = "Attribute Recoverability by Purpose",
) -> None:
    """Plot heatmap showing which attributes are recoverable under which purpose.

    Diagonal elements (allowed attributes) should have high accuracy.
    Off-diagonal elements (disallowed attributes) should have low accuracy (near chance).

    Args:
        probe_results: Dict mapping purpose name to probe results.
        output_path: Path to save the plot.
        title: Plot title.
    """
    purposes = list(probe_results.keys())

    # Collect all probed attributes across purposes
    all_attrs = set()
    for purpose_result in probe_results.values():
        sensitive_probes = purpose_result.get("sensitive_probes", {})
        all_attrs.update(sensitive_probes.keys())
        task_probes = purpose_result.get("task_probes", {})
        all_attrs.update(task_probes.keys())
    all_attrs = sorted(all_attrs)

    if not all_attrs:
        print("No probe results found")
        return

    # Build accuracy matrix: rows = purposes, cols = attributes
    accuracy_matrix = np.zeros((len(purposes), len(all_attrs)))

    for i, purpose in enumerate(purposes):
        result = probe_results[purpose]
        sensitive_probes = result.get("sensitive_probes", {})
        task_probes = result.get("task_probes", {})

        for j, attr in enumerate(all_attrs):
            if attr in sensitive_probes:
                accuracy_matrix[i, j] = sensitive_probes[attr]
            elif attr in task_probes:
                accuracy_matrix[i, j] = task_probes[attr]
            else:
                accuracy_matrix[i, j] = np.nan  # Not probed

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(max(10, len(all_attrs) * 0.8), max(6, len(purposes) * 0.6)))

    # Use diverging colormap centered at 0.5 (chance level for binary)
    im = ax.imshow(accuracy_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0.4, vmax=1.0)

    ax.set_xticks(range(len(all_attrs)))
    ax.set_yticks(range(len(purposes)))
    ax.set_xticklabels(all_attrs, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(purposes, fontsize=10)

    # Add value annotations
    for i in range(len(purposes)):
        for j in range(len(all_attrs)):
            val = accuracy_matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val > 0.7 or val < 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                       color=color, fontsize=8, fontweight='bold')

    ax.set_xlabel('Attribute / Task', fontsize=12)
    ax.set_ylabel('Purpose', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Probe Accuracy')
    cbar.ax.axhline(y=0.5, color='black', linestyle='--', linewidth=2)
    cbar.ax.text(1.5, 0.5, 'Chance', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved attribute recoverability heatmap to {output_path}")


def plot_compliance_matrix(
    purpose_specs: dict[str, dict[str, Any]],
    probe_results: dict[str, dict[str, Any]],
    output_path: str | Path,
    title: str = "Purpose Compliance Matrix",
) -> None:
    """Plot compliance matrix showing allowed vs disallowed attributes.

    Green = allowed and achieves high accuracy (good)
    Blue = disallowed and achieves low accuracy (good - protected)
    Red = disallowed but high accuracy (violation!)
    Yellow = allowed but low accuracy (poor utility)

    Args:
        purpose_specs: Dict mapping purpose name to spec with allowed_attrs/disallowed_attrs.
        probe_results: Dict mapping purpose name to probe results.
        output_path: Path to save the plot.
        title: Plot title.
    """
    purposes = list(probe_results.keys())

    # Collect all attributes
    all_attrs = set()
    for result in probe_results.values():
        all_attrs.update(result.get("sensitive_probes", {}).keys())
    all_attrs = sorted(all_attrs)

    if not all_attrs:
        print("No sensitive probe results found")
        return

    # Build matrices for accuracy and compliance
    accuracy_matrix = np.zeros((len(purposes), len(all_attrs)))
    compliance_matrix = np.zeros((len(purposes), len(all_attrs)))  # 1=allowed, 0=disallowed

    for i, purpose in enumerate(purposes):
        result = probe_results[purpose]
        spec = purpose_specs.get(purpose, {})

        allowed_attrs = set(spec.get("allowed_attrs", []))
        disallowed_attrs = set(spec.get("disallowed_attrs", []))

        for j, attr in enumerate(all_attrs):
            accuracy_matrix[i, j] = result.get("sensitive_probes", {}).get(attr, np.nan)

            if attr in allowed_attrs:
                compliance_matrix[i, j] = 1  # Allowed
            elif attr in disallowed_attrs:
                compliance_matrix[i, j] = 0  # Disallowed
            else:
                compliance_matrix[i, j] = 0.5  # Unknown

    # Create custom colormap based on compliance
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(purposes) * 0.6)))

    # Left: Accuracy heatmap
    ax = axes[0]
    im = ax.imshow(accuracy_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0.4, vmax=1.0)
    ax.set_xticks(range(len(all_attrs)))
    ax.set_yticks(range(len(purposes)))
    ax.set_xticklabels(all_attrs, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(purposes, fontsize=10)

    for i in range(len(purposes)):
        for j in range(len(all_attrs)):
            val = accuracy_matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val > 0.7 or val < 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                       color=color, fontsize=8, fontweight='bold')

    ax.set_xlabel('Sensitive Attribute', fontsize=12)
    ax.set_ylabel('Purpose', fontsize=12)
    ax.set_title('Probe Accuracy', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Accuracy')

    # Right: Compliance status
    ax = axes[1]

    # Custom colors: 0=disallowed (red/green based on protection), 1=allowed (blue)
    status_colors = np.zeros((*accuracy_matrix.shape, 3))
    status_labels = []

    for i in range(len(purposes)):
        row_labels = []
        for j in range(len(all_attrs)):
            acc = accuracy_matrix[i, j]
            is_allowed = compliance_matrix[i, j] == 1

            if is_allowed:
                if acc > 0.6:
                    status_colors[i, j] = [0.2, 0.7, 0.2]  # Green - good utility
                    row_labels.append('✓')
                else:
                    status_colors[i, j] = [0.9, 0.7, 0.2]  # Yellow - poor utility
                    row_labels.append('⚠')
            else:  # Disallowed
                if acc < 0.6:
                    status_colors[i, j] = [0.2, 0.4, 0.8]  # Blue - protected
                    row_labels.append('🛡')
                else:
                    status_colors[i, j] = [0.9, 0.2, 0.2]  # Red - violation
                    row_labels.append('✗')
        status_labels.append(row_labels)

    ax.imshow(status_colors, aspect='auto')
    ax.set_xticks(range(len(all_attrs)))
    ax.set_yticks(range(len(purposes)))
    ax.set_xticklabels(all_attrs, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(purposes, fontsize=10)

    for i in range(len(purposes)):
        for j in range(len(all_attrs)):
            ax.text(j, i, status_labels[i][j], ha='center', va='center', fontsize=12)

    ax.set_xlabel('Sensitive Attribute', fontsize=12)
    ax.set_title('Compliance Status', fontsize=12, fontweight='bold')

    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=[0.2, 0.7, 0.2], label='Allowed & High Acc (✓)'),
        plt.Rectangle((0, 0), 1, 1, facecolor=[0.9, 0.7, 0.2], label='Allowed & Low Acc (⚠)'),
        plt.Rectangle((0, 0), 1, 1, facecolor=[0.2, 0.4, 0.8], label='Disallowed & Protected (🛡)'),
        plt.Rectangle((0, 0), 1, 1, facecolor=[0.9, 0.2, 0.2], label='Disallowed & Violated (✗)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved compliance matrix to {output_path}")


def plot_cvr_heatmap(
    cvr_results: dict[str, dict[str, Any]],
    output_path: str | Path,
    title: str = "CVR Heatmap by Purpose and Attribute",
) -> None:
    """Plot CVR (Compliance Violation Rate) heatmap.

    Lower CVR is better (less sensitive information leaked).

    Args:
        cvr_results: Dict mapping purpose name to CVR results.
        output_path: Path to save the plot.
        title: Plot title.
    """
    purposes = list(cvr_results.keys())

    # Collect all attributes
    all_attrs = set()
    for result in cvr_results.values():
        per_attr = result.get("per_attr_cvr", {})
        all_attrs.update(per_attr.keys())
    all_attrs = sorted(all_attrs)

    if not all_attrs:
        print("No per-attribute CVR found")
        return

    # Build CVR matrix
    cvr_matrix = np.zeros((len(purposes), len(all_attrs)))

    for i, purpose in enumerate(purposes):
        per_attr = cvr_results[purpose].get("per_attr_cvr", {})
        for j, attr in enumerate(all_attrs):
            cvr_matrix[i, j] = per_attr.get(attr, 0)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(max(10, len(all_attrs) * 0.8), max(6, len(purposes) * 0.6)))

    # Green = low CVR (good), Red = high CVR (bad)
    im = ax.imshow(cvr_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=0.5)

    ax.set_xticks(range(len(all_attrs)))
    ax.set_yticks(range(len(purposes)))
    ax.set_xticklabels(all_attrs, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(purposes, fontsize=10)

    # Add value annotations
    for i in range(len(purposes)):
        for j in range(len(all_attrs)):
            val = cvr_matrix[i, j]
            color = 'white' if val > 0.3 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                   color=color, fontsize=8, fontweight='bold')

    ax.set_xlabel('Sensitive Attribute', fontsize=12)
    ax.set_ylabel('Purpose', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, label='CVR (lower is better)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved CVR heatmap to {output_path}")


def plot_purpose_switching_demo(
    results: dict[str, Any],
    output_path: str | Path,
    title: str = "Purpose Switching Effect on Information Leakage",
) -> None:
    """Plot demonstration of purpose switching effect.

    Shows how changing the purpose token affects what information is recoverable.

    Args:
        results: Results with probe_results per purpose.
        output_path: Path to save the plot.
        title: Plot title.
    """
    probe_results = results.get("probe_results", {})

    if not probe_results:
        print("No probe results found")
        return

    purposes = list(probe_results.keys())
    n_purposes = len(purposes)

    # Collect sensitive attributes
    all_attrs = set()
    for result in probe_results.values():
        all_attrs.update(result.get("sensitive_probes", {}).keys())
    all_attrs = sorted(all_attrs)

    if not all_attrs:
        print("No sensitive probe results found")
        return

    fig, axes = plt.subplots(1, n_purposes, figsize=(5 * n_purposes, 5), sharey=True)
    if n_purposes == 1:
        axes = [axes]

    x = np.arange(len(all_attrs))
    width = 0.6

    for ax, purpose in zip(axes, purposes):
        result = probe_results[purpose]
        sensitive_probes = result.get("sensitive_probes", {})

        accuracies = [sensitive_probes.get(attr, 0.5) for attr in all_attrs]

        # Color bars based on whether they exceed chance
        colors = ['red' if acc > 0.55 else 'green' for acc in accuracies]

        bars = ax.bar(x, accuracies, width, color=colors, edgecolor='black', alpha=0.8)

        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Chance Level')
        ax.set_xlabel('Sensitive Attribute', fontsize=11)
        ax.set_ylabel('Probe Accuracy', fontsize=11)
        ax.set_title(f'Purpose: {purpose}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(all_attrs, rotation=45, ha='right', fontsize=9)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.annotate(f'{acc:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', edgecolor='black', label='Protected (≤chance)'),
        Patch(facecolor='red', edgecolor='black', label='Leaking (>chance)'),
    ]
    axes[-1].legend(handles=legend_elements, loc='upper right', fontsize=9)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved purpose switching demo to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot purpose switching heatmaps")
    parser.add_argument("--results", type=str, required=True, help="Path to PCRL results JSON")
    parser.add_argument("--config", type=str, help="Path to config YAML (for purpose specs)")
    parser.add_argument("--output-dir", type=str, default="figures", help="Output directory")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    results = load_results(args.results)

    # Plot attribute recoverability
    if "probe_results" in results:
        plot_attribute_recoverability_heatmap(
            results["probe_results"],
            output_dir / f"attribute_recoverability.{args.format}",
        )

        plot_purpose_switching_demo(
            results,
            output_dir / f"purpose_switching_demo.{args.format}",
        )

    # Plot CVR heatmap
    if "cvr_results" in results:
        plot_cvr_heatmap(
            results["cvr_results"],
            output_dir / f"cvr_heatmap.{args.format}",
        )

    # Plot compliance matrix if config provided
    if args.config and "probe_results" in results:
        try:
            import yaml
            with open(args.config) as f:
                config = yaml.safe_load(f)
            purpose_specs = {p["name"]: p for p in config.get("purposes", [])}

            plot_compliance_matrix(
                purpose_specs,
                results["probe_results"],
                output_dir / f"compliance_matrix.{args.format}",
            )
        except Exception as e:
            print(f"Could not load config for compliance matrix: {e}")

    print(f"\nAll purpose switching plots saved to {output_dir}")


if __name__ == "__main__":
    main()
