#!/usr/bin/env python3
"""Plot training curves and metrics.

Creates visualizations showing:
1. Loss curves (task loss, adversarial loss, total loss)
2. Auditor accuracy over time (should decrease for encoder success)
3. Task accuracy over training
4. Learning dynamics analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def load_training_history(history_path: str | Path) -> dict[str, Any]:
    """Load training history from JSON file."""
    with open(history_path) as f:
        return json.load(f)


def smooth_curve(values: list[float], weight: float = 0.9) -> list[float]:
    """Exponential moving average smoothing.

    Args:
        values: List of values to smooth.
        weight: Smoothing weight (higher = smoother).

    Returns:
        Smoothed values.
    """
    smoothed = []
    last = values[0] if values else 0
    for v in values:
        smoothed_val = last * weight + v * (1 - weight)
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_loss_curves(
    history: dict[str, Any],
    output_path: str | Path,
    title: str = "Training Loss Curves",
    smooth: bool = True,
) -> None:
    """Plot training loss curves.

    Args:
        history: Training history dictionary.
        output_path: Path to save the plot.
        title: Plot title.
        smooth: Whether to apply smoothing.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Task loss
    if "task_loss" in history:
        task_loss = history["task_loss"]
        epochs = range(1, len(task_loss) + 1)
        axes[0].plot(epochs, task_loss, alpha=0.3, color='blue', label='Raw')
        if smooth:
            axes[0].plot(epochs, smooth_curve(task_loss), color='blue', linewidth=2, label='Smoothed')
        axes[0].set_xlabel('Epoch', fontsize=11)
        axes[0].set_ylabel('Task Loss', fontsize=11)
        axes[0].set_title('Task Loss', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    # Adversarial loss
    if "adv_loss" in history:
        adv_loss = history["adv_loss"]
        epochs = range(1, len(adv_loss) + 1)
        axes[1].plot(epochs, adv_loss, alpha=0.3, color='red', label='Raw')
        if smooth:
            axes[1].plot(epochs, smooth_curve(adv_loss), color='red', linewidth=2, label='Smoothed')
        axes[1].set_xlabel('Epoch', fontsize=11)
        axes[1].set_ylabel('Adversarial Loss', fontsize=11)
        axes[1].set_title('Adversarial Loss', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    # Total loss
    if "total_loss" in history:
        total_loss = history["total_loss"]
        epochs = range(1, len(total_loss) + 1)
        axes[2].plot(epochs, total_loss, alpha=0.3, color='green', label='Raw')
        if smooth:
            axes[2].plot(epochs, smooth_curve(total_loss), color='green', linewidth=2, label='Smoothed')
        axes[2].set_xlabel('Epoch', fontsize=11)
        axes[2].set_ylabel('Total Loss', fontsize=11)
        axes[2].set_title('Total Loss (Task + λ·Adv)', fontsize=12, fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved loss curves to {output_path}")


def plot_auditor_accuracy(
    history: dict[str, Any],
    output_path: str | Path,
    title: str = "Auditor Accuracy Over Training",
    smooth: bool = True,
) -> None:
    """Plot auditor accuracy over training.

    For successful PCRL training, auditor accuracy should decrease toward chance level.

    Args:
        history: Training history dictionary.
        output_path: Path to save the plot.
        title: Plot title.
        smooth: Whether to apply smoothing.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Check for auditor accuracy in history
    auditor_keys = [k for k in history.keys() if "auditor" in k.lower() and "acc" in k.lower()]

    if not auditor_keys:
        # Try alternative names
        auditor_keys = [k for k in history.keys() if "probe" in k.lower() or "sensitive" in k.lower()]

    if not auditor_keys:
        print("No auditor accuracy found in history")
        return

    colors = plt.cm.tab10(np.linspace(0, 1, len(auditor_keys)))

    for color, key in zip(colors, auditor_keys):
        values = history[key]
        epochs = range(1, len(values) + 1)
        ax.plot(epochs, values, alpha=0.3, color=color)
        if smooth:
            ax.plot(epochs, smooth_curve(values), color=color, linewidth=2, label=key)
        else:
            ax.plot(epochs, values, color=color, linewidth=2, label=key)

    # Add chance level reference line
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Chance Level (binary)')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Add annotation
    ax.annotate(
        'Lower is better\n(less sensitive info leaked)',
        xy=(0.95, 0.95), xycoords='axes fraction',
        ha='right', va='top',
        fontsize=9, style='italic',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved auditor accuracy plot to {output_path}")


def plot_task_accuracy(
    history: dict[str, Any],
    output_path: str | Path,
    title: str = "Task Accuracy Over Training",
    smooth: bool = True,
) -> None:
    """Plot task accuracy over training.

    Args:
        history: Training history dictionary.
        output_path: Path to save the plot.
        title: Plot title.
        smooth: Whether to apply smoothing.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Find task accuracy keys
    task_keys = [k for k in history.keys() if "task" in k.lower() and "acc" in k.lower()]

    if not task_keys:
        # Try validation accuracy
        task_keys = [k for k in history.keys() if "val" in k.lower() and "acc" in k.lower()]

    if not task_keys:
        print("No task accuracy found in history")
        return

    colors = plt.cm.tab10(np.linspace(0, 1, len(task_keys)))

    for color, key in zip(colors, task_keys):
        values = history[key]
        epochs = range(1, len(values) + 1)
        ax.plot(epochs, values, alpha=0.3, color=color)
        if smooth:
            ax.plot(epochs, smooth_curve(values), color=color, linewidth=2, label=key)
        else:
            ax.plot(epochs, values, color=color, linewidth=2, label=key)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved task accuracy plot to {output_path}")


def plot_combined_metrics(
    history: dict[str, Any],
    output_path: str | Path,
    title: str = "Training Dynamics Overview",
    smooth: bool = True,
) -> None:
    """Plot combined training metrics in a single figure.

    Shows task loss, adversarial loss, task accuracy, and auditor accuracy together.

    Args:
        history: Training history dictionary.
        output_path: Path to save the plot.
        title: Plot title.
        smooth: Whether to apply smoothing.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Task loss (top left)
    ax = axes[0, 0]
    if "task_loss" in history:
        values = history["task_loss"]
        epochs = range(1, len(values) + 1)
        ax.plot(epochs, values, alpha=0.3, color='blue')
        if smooth:
            ax.plot(epochs, smooth_curve(values), color='blue', linewidth=2)
        ax.set_ylabel('Task Loss', fontsize=11)
        ax.set_title('Task Loss', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # Adversarial loss (top right)
    ax = axes[0, 1]
    if "adv_loss" in history:
        values = history["adv_loss"]
        epochs = range(1, len(values) + 1)
        ax.plot(epochs, values, alpha=0.3, color='red')
        if smooth:
            ax.plot(epochs, smooth_curve(values), color='red', linewidth=2)
        ax.set_ylabel('Adversarial Loss', fontsize=11)
        ax.set_title('Adversarial Loss', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # Task accuracy (bottom left)
    ax = axes[1, 0]
    task_keys = [k for k in history.keys() if "task" in k.lower() and "acc" in k.lower()]
    if task_keys:
        for key in task_keys[:3]:  # Limit to 3 tasks
            values = history[key]
            epochs = range(1, len(values) + 1)
            ax.plot(epochs, smooth_curve(values) if smooth else values, linewidth=2, label=key)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Accuracy', fontsize=11)
        ax.set_title('Task Accuracy', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    # Auditor accuracy (bottom right)
    ax = axes[1, 1]
    auditor_keys = [k for k in history.keys() if "auditor" in k.lower() and "acc" in k.lower()]
    if auditor_keys:
        for key in auditor_keys[:3]:  # Limit to 3 auditors
            values = history[key]
            epochs = range(1, len(values) + 1)
            ax.plot(epochs, smooth_curve(values) if smooth else values, linewidth=2, label=key)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Chance')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Accuracy', fontsize=11)
        ax.set_title('Auditor Accuracy (lower = better privacy)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved combined metrics plot to {output_path}")


def plot_learning_rate_schedule(
    history: dict[str, Any],
    output_path: str | Path,
    title: str = "Learning Rate Schedule",
) -> None:
    """Plot learning rate schedule if available.

    Args:
        history: Training history dictionary.
        output_path: Path to save the plot.
        title: Plot title.
    """
    lr_keys = [k for k in history.keys() if "lr" in k.lower() or "learning_rate" in k.lower()]

    if not lr_keys:
        print("No learning rate schedule found in history")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    for key in lr_keys:
        values = history[key]
        epochs = range(1, len(values) + 1)
        ax.plot(epochs, values, linewidth=2, label=key)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved learning rate schedule to {output_path}")


def plot_per_purpose_metrics(
    history: dict[str, Any],
    output_path: str | Path,
    title: str = "Per-Purpose Training Metrics",
    smooth: bool = True,
) -> None:
    """Plot metrics broken down by purpose.

    Args:
        history: Training history dictionary.
        output_path: Path to save the plot.
        title: Plot title.
        smooth: Whether to apply smoothing.
    """
    # Find purpose-specific keys
    purpose_keys = {}
    for key in history.keys():
        if "_purpose_" in key:
            parts = key.split("_purpose_")
            metric = parts[0]
            purpose = parts[1]
            if metric not in purpose_keys:
                purpose_keys[metric] = {}
            purpose_keys[metric][purpose] = key

    if not purpose_keys:
        print("No per-purpose metrics found in history")
        return

    n_metrics = len(purpose_keys)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))

    if n_metrics == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for ax, (metric, purposes) in zip(axes, purpose_keys.items()):
        for i, (purpose, key) in enumerate(purposes.items()):
            values = history[key]
            epochs = range(1, len(values) + 1)
            ax.plot(epochs, values, alpha=0.3, color=colors[i % 10])
            if smooth:
                ax.plot(epochs, smooth_curve(values), color=colors[i % 10], linewidth=2, label=purpose)
            else:
                ax.plot(epochs, values, color=colors[i % 10], linewidth=2, label=purpose)

        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.legend(title='Purpose', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved per-purpose metrics to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training curves")
    parser.add_argument("--history", type=str, required=True, help="Path to training history JSON")
    parser.add_argument("--output-dir", type=str, default="figures", help="Output directory")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"])
    parser.add_argument("--no-smooth", action="store_true", help="Disable smoothing")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load history
    history = load_training_history(args.history)

    # Handle nested structure (if history is under a key)
    if "training_history" in history:
        history = history["training_history"]

    smooth = not args.no_smooth

    # Generate all plots
    plot_loss_curves(history, output_dir / f"loss_curves.{args.format}", smooth=smooth)
    plot_auditor_accuracy(history, output_dir / f"auditor_accuracy.{args.format}", smooth=smooth)
    plot_task_accuracy(history, output_dir / f"task_accuracy.{args.format}", smooth=smooth)
    plot_combined_metrics(history, output_dir / f"training_overview.{args.format}", smooth=smooth)
    plot_learning_rate_schedule(history, output_dir / f"lr_schedule.{args.format}")
    plot_per_purpose_metrics(history, output_dir / f"per_purpose_metrics.{args.format}", smooth=smooth)

    print(f"\nAll training plots saved to {output_dir}")


if __name__ == "__main__":
    main()
