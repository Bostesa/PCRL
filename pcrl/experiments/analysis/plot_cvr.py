#!/usr/bin/env python3
"""Plot Compliance Violation Rate (CVR) comparisons.

Creates:
1. Bar charts: CVR per attribute per purpose
2. Comparison: PCRL vs baselines
3. Summary table with statistical significance
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


def plot_cvr_by_purpose(
    results: dict[str, Any],
    output_path: str | Path,
    title: str = "CVR by Purpose",
) -> None:
    """Plot CVR bar chart for each purpose.

    Args:
        results: Results dictionary with cvr_results.
        output_path: Path to save the plot.
        title: Plot title.
    """
    cvr_results = results.get("cvr_results", {})

    if not cvr_results:
        print("No CVR results found in results file")
        return

    purposes = list(cvr_results.keys())
    overall_cvr = [cvr_results[p]["overall_cvr"] for p in purposes]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(purposes))
    bars = ax.bar(x, overall_cvr, color='steelblue', edgecolor='black', alpha=0.8)

    # Add value labels on bars
    for bar, val in zip(bars, overall_cvr):
        height = bar.get_height()
        ax.annotate(
            f'{val:.3f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=10, fontweight='bold'
        )

    ax.set_xlabel('Purpose', fontsize=12)
    ax.set_ylabel('Compliance Violation Rate (CVR)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(purposes, rotation=45, ha='right')
    ax.set_ylim(0, max(overall_cvr) * 1.2 if overall_cvr else 1.0)

    # Add horizontal line at chance level (for reference)
    ax.axhline(y=0.0, color='gray', linestyle='--', alpha=0.5, label='Perfect (CVR=0)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved CVR by purpose plot to {output_path}")


def plot_cvr_per_attribute(
    results: dict[str, Any],
    output_path: str | Path,
    title: str = "CVR per Attribute by Purpose",
) -> None:
    """Plot detailed CVR for each sensitive attribute within each purpose.

    Args:
        results: Results dictionary with cvr_results.
        output_path: Path to save the plot.
        title: Plot title.
    """
    cvr_results = results.get("cvr_results", {})

    if not cvr_results:
        print("No CVR results found")
        return

    purposes = list(cvr_results.keys())

    # Collect all unique attributes
    all_attrs = set()
    for p in purposes:
        per_attr = cvr_results[p].get("per_attr_cvr", {})
        all_attrs.update(per_attr.keys())
    all_attrs = sorted(all_attrs)

    if not all_attrs:
        print("No per-attribute CVR data found")
        return

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(purposes))
    width = 0.8 / len(all_attrs)
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_attrs)))

    for i, attr in enumerate(all_attrs):
        attr_cvr = []
        for p in purposes:
            per_attr = cvr_results[p].get("per_attr_cvr", {})
            attr_cvr.append(per_attr.get(attr, 0.0))

        offset = (i - len(all_attrs) / 2 + 0.5) * width
        bars = ax.bar(x + offset, attr_cvr, width, label=attr, color=colors[i], edgecolor='black', alpha=0.8)

    ax.set_xlabel('Purpose', fontsize=12)
    ax.set_ylabel('CVR', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(purposes, rotation=45, ha='right')
    ax.legend(title='Sensitive Attribute', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved CVR per attribute plot to {output_path}")


def plot_cvr_comparison(
    pcrl_results: dict[str, Any],
    baseline_results: dict[str, dict[str, Any]],
    output_path: str | Path,
    title: str = "CVR Comparison: PCRL vs Baselines",
) -> None:
    """Plot CVR comparison between PCRL and baselines.

    Args:
        pcrl_results: PCRL results dictionary.
        baseline_results: Dictionary mapping baseline name to results.
        output_path: Path to save the plot.
        title: Plot title.
    """
    # Extract overall CVR from PCRL (average across purposes)
    pcrl_cvr_results = pcrl_results.get("cvr_results", {})
    pcrl_overall = np.mean([r["overall_cvr"] for r in pcrl_cvr_results.values()]) if pcrl_cvr_results else 0

    # Extract CVR from baselines
    methods = ["PCRL"]
    cvr_values = [pcrl_overall]

    for name, results in baseline_results.items():
        methods.append(name)
        if "cvr_results" in results:
            # Baseline with CVR per purpose
            baseline_cvr = results["cvr_results"]
            if isinstance(baseline_cvr, dict):
                if "overall" in baseline_cvr:
                    cvr_values.append(baseline_cvr["overall"])
                elif "overall_cvr" in baseline_cvr:
                    cvr_values.append(baseline_cvr["overall_cvr"])
                else:
                    # Average across purposes
                    cvrs = [v.get("overall_cvr", v.get("cvr", 0)) for v in baseline_cvr.values() if isinstance(v, dict)]
                    cvr_values.append(np.mean(cvrs) if cvrs else 0)
            else:
                cvr_values.append(baseline_cvr)
        elif "protected_cvr" in results:
            cvr_values.append(results["protected_cvr"].get("overall", 0))
        else:
            cvr_values.append(0)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(methods))
    colors = ['steelblue'] + ['coral'] * (len(methods) - 1)

    bars = ax.bar(x, cvr_values, color=colors, edgecolor='black', alpha=0.8)

    # Add value labels
    for bar, val in zip(bars, cvr_values):
        height = bar.get_height()
        ax.annotate(
            f'{val:.3f}',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=10, fontweight='bold'
        )

    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Average CVR', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.set_ylim(0, max(cvr_values) * 1.3 if cvr_values else 1.0)

    # Add annotation
    ax.axhline(y=0.0, color='green', linestyle='--', alpha=0.5, label='Perfect fairness')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved CVR comparison plot to {output_path}")


def plot_task_vs_fairness(
    results: dict[str, Any],
    output_path: str | Path,
    title: str = "Task Performance vs Fairness Trade-off",
) -> None:
    """Plot task accuracy vs sensitive attribute leakage scatter plot.

    Args:
        results: Results dictionary with probe_results.
        output_path: Path to save the plot.
        title: Plot title.
    """
    probe_results = results.get("probe_results", {})

    if not probe_results:
        print("No probe results found")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(probe_results)))

    for i, (purpose_name, result) in enumerate(probe_results.items()):
        task_probes = result.get("task_probes", {})
        sensitive_probes = result.get("sensitive_probes", {})

        # Average task accuracy
        task_acc = np.mean(list(task_probes.values())) if task_probes else 0
        # Average sensitive probe accuracy
        sens_acc = np.mean(list(sensitive_probes.values())) if sensitive_probes else 0.5

        ax.scatter(task_acc, sens_acc, s=200, c=[colors[i]], label=purpose_name,
                   edgecolor='black', linewidth=1.5, alpha=0.8)

        # Add label
        ax.annotate(
            purpose_name,
            (task_acc, sens_acc),
            xytext=(10, 5),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold'
        )

    # Add reference lines
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance level (binary)')
    ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.3)

    # Ideal region annotation
    ax.fill_between([0.7, 1.0], [0.0, 0.0], [0.6, 0.6], alpha=0.1, color='green')
    ax.text(0.85, 0.55, 'Ideal\nRegion', ha='center', va='top', fontsize=9, style='italic', color='green')

    ax.set_xlabel('Task Accuracy (higher is better)', fontsize=12)
    ax.set_ylabel('Sensitive Attribute Probe Accuracy (lower is better)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left', title='Purpose')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved task vs fairness plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CVR analysis")
    parser.add_argument("--results", type=str, required=True, help="Path to PCRL results JSON")
    parser.add_argument("--baselines", type=str, nargs="*", default=[], help="Paths to baseline results JSONs")
    parser.add_argument("--output-dir", type=str, default="figures", help="Output directory for figures")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load PCRL results
    pcrl_results = load_results(args.results)

    # Plot CVR by purpose
    plot_cvr_by_purpose(
        pcrl_results,
        output_dir / f"cvr_by_purpose.{args.format}",
    )

    # Plot CVR per attribute
    plot_cvr_per_attribute(
        pcrl_results,
        output_dir / f"cvr_per_attribute.{args.format}",
    )

    # Plot task vs fairness trade-off
    plot_task_vs_fairness(
        pcrl_results,
        output_dir / f"task_vs_fairness.{args.format}",
    )

    # Load and plot baseline comparisons if provided
    if args.baselines:
        baseline_results = {}
        for path in args.baselines:
            name = Path(path).stem.replace("results_", "")
            try:
                baseline_results[name] = load_results(path)
            except Exception as e:
                print(f"Warning: Could not load baseline {path}: {e}")

        if baseline_results:
            plot_cvr_comparison(
                pcrl_results,
                baseline_results,
                output_dir / f"cvr_comparison.{args.format}",
            )

    print(f"\nAll figures saved to {output_dir}")


if __name__ == "__main__":
    main()
