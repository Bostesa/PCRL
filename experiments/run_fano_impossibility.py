#!/usr/bin/env python3
"""Fano-based impossibility analysis across all datasets.

Uses classifier accuracy (already measured) to compute MI lower bounds
via Fano's inequality, then derives single-representation impossibility
bounds. This avoids MINE estimation issues on high-dimensional representations.

For each conflicting attribute (allowed for one purpose, forbidden for another):
  1. Task accuracy from the purpose that NEEDS the attribute → MI lower bound
  2. MI lower bound → impossibility bound (minimum leakage any single-rep must have)
  3. LAFTR's actual leakage (from baseline comparison CSV)
  4. PCRL's actual leakage for the forbidding purpose (from baseline comparison CSV)

Runs on saved results — no retraining needed.

Saves to results/{dataset}/impossibility_fano.csv.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pcrl.purposes.verification import fano_mi_lower_bound, impossibility_bound


def load_baseline_csv(path: Path) -> list[dict]:
    """Load a baseline_comparison.csv file."""
    with open(path) as f:
        return list(csv.DictReader(f))


def get_task_accuracy(
    rows: list[dict], method: str, task_attr: str, purpose_name: str,
) -> float | None:
    """Get the task accuracy for a specific attribute from baseline results.

    Handles two CSV formats:
    - CelebA/HAR style: per-attribute columns like Smiling_acc, activity_acc
    - Adult style: single 'task_acc' column (use purpose to find the right row)
    """
    for r in rows:
        if r["method"] == method:
            # Try per-attribute column (CelebA/HAR format)
            col = f"{task_attr}_acc"
            if col in r and r[col]:
                return float(r[col])
            # Try case-insensitive
            for k in r:
                if k.lower() == f"{task_attr.lower()}_acc" and r[k]:
                    return float(r[k])

    # Fallback: Adult format — single 'task_acc' column, look for the purpose
    # that has this attribute as its task
    for r in rows:
        if r["method"] == method and r["purpose"] == purpose_name and "task_acc" in r and r["task_acc"]:
            return float(r["task_acc"])

    return None


def get_auditor_accuracy(
    rows: list[dict], method: str, purpose: str, attribute: str,
) -> tuple[float, float]:
    """Get best empirical auditor accuracy and majority baseline for a (purpose, attribute) pair."""
    for r in rows:
        if r["method"] == method and r["purpose"] == purpose and r["attribute"] == attribute:
            return float(r["best_emp_acc"]), float(r["majority_baseline"])
    return 0.0, 0.0


def compute_entropy(majority_proportion: float, num_classes: int) -> float:
    """Estimate entropy H(A) from majority proportion and num_classes.

    For binary: H = -p*log(p) - (1-p)*log(1-p)
    For multi-class: approximate using majority + uniform remainder.
    """
    if num_classes == 2:
        p = majority_proportion
        q = 1.0 - p
        if p < 1e-10 or q < 1e-10:
            return 0.0
        return -p * math.log(p) - q * math.log(q)
    else:
        # For multi-class, use uniform approximation (conservative — gives higher entropy)
        return math.log(num_classes)


def analyze_dataset(
    name: str,
    baseline_csv: Path,
    purposes_fn,
    output_csv: Path,
) -> list[dict]:
    """Run Fano impossibility analysis for one dataset."""
    print(f"\n{'=' * 90}")
    print(f"Fano Impossibility Analysis: {name}")
    print(f"{'=' * 90}")

    rows = load_baseline_csv(baseline_csv)
    purposes = purposes_fn()

    # Find conflicting attributes
    conflicts = []
    for p_need in purposes:
        for task_attr in p_need.allowed_tasks:
            for p_forbid in purposes:
                if p_need.name != p_forbid.name and task_attr in p_forbid.disallowed_attrs:
                    conflicts.append((task_attr, p_need.name, p_forbid.name))

    if not conflicts:
        print("  No conflicting attributes found.")
        return []

    # Deduplicate: for each (attr, needed_by), pick one forbidden_by for the table
    # but keep all for CSV
    print(f"  Found {len(conflicts)} conflicting (attr, need, forbid) triples")

    # Get attribute metadata from purposes
    attr_dims = {}
    for p in purposes:
        for attr, dim in p.disallowed_attr_dims.items():
            attr_dims[attr] = dim
        for attr, dim in p.allowed_task_dims.items():
            attr_dims[attr] = dim

    results = []
    pcrl_method = "PCRL (ours)"
    laftr_method = "Adversarial-only (LAFTR)"

    for attr, need, forbid in conflicts:
        num_classes = attr_dims.get(attr, 2)

        # Get PCRL task accuracy for the purpose that NEEDS this attribute
        task_acc = get_task_accuracy(rows, pcrl_method, attr, need)
        if task_acc is None:
            continue

        # Get majority baseline for this attribute (from any row that audits it)
        _, majority = get_auditor_accuracy(rows, pcrl_method, forbid, attr)
        if majority == 0.0:
            # Try other purposes
            for p in purposes:
                _, majority = get_auditor_accuracy(rows, pcrl_method, p.name, attr)
                if majority > 0.0:
                    break

        # Compute entropy
        entropy_a = compute_entropy(majority, num_classes)

        # Fano MI lower bound from task accuracy
        mi_lower = fano_mi_lower_bound(task_acc, num_classes, entropy_a)

        # Impossibility bound: minimum leakage any single-rep method must have
        sr_min_leak = impossibility_bound(mi_lower, num_classes, entropy_a)

        # LAFTR actual leakage
        laftr_acc, _ = get_auditor_accuracy(rows, laftr_method, forbid, attr)

        # PCRL actual leakage (for the FORBIDDING purpose)
        pcrl_leak, _ = get_auditor_accuracy(rows, pcrl_method, forbid, attr)

        result = {
            "attribute": attr,
            "needed_by": need,
            "forbidden_by": forbid,
            "num_classes": num_classes,
            "task_acc": task_acc,
            "entropy_a_bits": entropy_a / math.log(2),
            "mi_lower_bits": mi_lower / math.log(2),
            "single_rep_min_leak": sr_min_leak,
            "laftr_actual_leak": laftr_acc,
            "pcrl_leak": pcrl_leak,
            "majority_baseline": majority,
        }
        results.append(result)

    # Print table
    print(f"\n{'Attribute':<16} {'Needed by':<28} {'Task Acc':>8} {'MI≥':>8} "
          f"{'SR Min':>8} {'LAFTR':>8} {'PCRL':>8}")
    print("-" * 90)

    # Deduplicate for display (show each (attr, need) once, pick worst forbid)
    seen = set()
    for r in results:
        key = (r["attribute"], r["needed_by"], r["forbidden_by"])
        if key in seen:
            continue
        seen.add(key)
        print(f"{r['attribute']:<16} {r['needed_by']:<28} "
              f"{r['task_acc']:>7.1%} {r['mi_lower_bits']:>7.3f}b "
              f"{r['single_rep_min_leak']:>7.1%} "
              f"{r['laftr_actual_leak']:>7.1%} "
              f"{r['pcrl_leak']:>7.1%}")

    # Summary
    if results:
        print("-" * 90)
        # Show key conflicts where SR bound > PCRL leak
        key_wins = [r for r in results if r["single_rep_min_leak"] > r["pcrl_leak"] + 0.01]
        if key_wins:
            print(f"\n  PCRL circumvents impossibility in {len(key_wins)}/{len(results)} cases:")
            for r in key_wins:
                gap = r["single_rep_min_leak"] - r["pcrl_leak"]
                print(f"    {r['attribute']} ({r['needed_by']} -> {r['forbidden_by']}): "
                      f"SR must leak {r['single_rep_min_leak']:.1%}, "
                      f"PCRL only leaks {r['pcrl_leak']:.1%} "
                      f"(gap: {gap:.1%})")

        # Show cases where LAFTR collapsed
        laftr_collapses = [r for r in results
                          if r["laftr_actual_leak"] <= r["majority_baseline"] + 0.02]
        if laftr_collapses:
            print(f"\n  LAFTR collapsed to majority in {len(laftr_collapses)}/{len(results)} cases "
                  f"(destroyed utility to avoid leakage)")

    # Save CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "attribute", "needed_by", "forbidden_by", "num_classes",
        "task_acc", "entropy_a_bits", "mi_lower_bits", "single_rep_min_leak",
        "laftr_actual_leak", "pcrl_leak", "majority_baseline",
    ]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                k: (round(v, 6) if isinstance(v, float) else v)
                for k, v in r.items()
            })
    print(f"\nSaved {output_csv}")

    return results


def main() -> None:
    from pcrl.data.adult import get_adult_purposes
    from pcrl.data.celeba import get_celeba_purposes
    from pcrl.data.har import get_har_purposes

    results_dir = project_root / "results"

    # CelebA
    celeba_csv = results_dir / "celeba" / "baseline_comparison.csv"
    if celeba_csv.exists():
        analyze_dataset(
            "CelebA", celeba_csv, get_celeba_purposes,
            results_dir / "celeba" / "impossibility_fano.csv",
        )
    else:
        print(f"Skipping CelebA: {celeba_csv} not found")

    # Adult
    adult_csv = results_dir / "adult" / "baseline_comparison.csv"
    if adult_csv.exists():
        analyze_dataset(
            "Adult", adult_csv, get_adult_purposes,
            results_dir / "adult" / "impossibility_fano.csv",
        )
    else:
        print(f"Skipping Adult: {adult_csv} not found")

    # HAR
    har_csv = results_dir / "har_real" / "baseline_comparison.csv"
    if har_csv.exists():
        analyze_dataset(
            "HAR", har_csv, get_har_purposes,
            results_dir / "har_real" / "impossibility_fano.csv",
        )
    else:
        print(f"Skipping HAR: {har_csv} not found")

    print(f"\n{'=' * 90}")
    print("DONE — Fano impossibility analysis complete")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
