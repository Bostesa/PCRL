"""Fixed stronger auditors for the exact frozen ACS protection releases.

This adapter leaves the historical ACS head implementations unchanged. It fits
one logistic model, two 120-epoch MLP trajectories and two fixed histogram-tree
configurations. MLP epochs are chosen by validation log loss. Primary log-loss
and diagnostic AUROC selection then compare those five resulting candidates;
the AUROC diagnostic does not select different MLP epochs. No final-test arrays
are accepted, and none of these fits updates a released representation.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import time

import numpy as np

from experiments.acs_transfer_heads import HEAD_BUDGET, fit_candidates


AUDIT_BUDGET = copy.deepcopy(HEAD_BUDGET)
AUDIT_BUDGET["mlp"]["epochs"] = 120
MLP_RESTART_OFFSETS = (0, 10000)
CANDIDATE_IDS = ("logistic", "mlp_0", "mlp_1", "hist_gb_20", "hist_gb_5")
_SPECIFICATIONS = (
    ("logistic", "logistic", 0, None),
    ("mlp_0", "mlp", MLP_RESTART_OFFSETS[0], None),
    ("mlp_1", "mlp", MLP_RESTART_OFFSETS[1], None),
    ("hist_gb_20", "histgb", 0, 20),
    ("hist_gb_5", "histgb", 0, 5),
)


def _resolve_budget(budget):
    resolved = copy.deepcopy(AUDIT_BUDGET)
    for section, updates in (budget or {}).items():
        if (section not in resolved or not isinstance(updates, dict)
                or set(updates)-set(resolved[section])):
            raise ValueError(f"Unknown or malformed audit budget: {section}")
        resolved[section].update(updates)
    if resolved["histgb"]["min_samples_leaf"] != 20:
        raise ValueError("The declared tree minimum-leaf candidates are fixed at 20 and 5")
    return resolved


def select_candidates(validation_scores):
    """Select from already log-loss-fitted candidates using validation only."""
    if not validation_scores:
        raise ValueError("At least one validation candidate is required")
    finite_loss = {key: value["log_loss"] for key, value in validation_scores.items()
                   if value.get("log_loss") is not None and np.isfinite(value["log_loss"])}
    finite_auc = {key: value["auroc"] for key, value in validation_scores.items()
                  if value.get("auroc") is not None and np.isfinite(value["auroc"])}
    selected_loss = min(finite_loss, key=lambda key: (finite_loss[key], key)) if finite_loss else None
    selected_auc = min(finite_auc, key=lambda key: (-finite_auc[key], key)) if finite_auc else None
    return {
        "selected_log_loss": selected_loss, "selected_family": selected_loss,
        "selected_auroc": selected_auc,
        "selection_split": "attacker_validation",
        "primary_rule": "minimum validation log loss, then lexicographic candidate ID",
        "auroc_rule": "maximum defined validation AUROC, then lexicographic candidate ID",
        "auroc_scope": "diagnostic selection among the same five candidates; every MLP checkpoint was selected by log loss, not AUROC",
        "undefined_log_loss_candidates": sorted(set(validation_scores)-set(finite_loss)),
        "undefined_auroc_candidates": sorted(set(validation_scores)-set(finite_auc)),
    }


def fit_primary_auditors(xfit, yfit, xval, yval, n_classes, seed, *, budget=None):
    """Fit the five predeclared auditors with identical release/split inputs.

    The optional budget is for small artificial integration fixtures; scientific
    runs use AUDIT_BUDGET unchanged. The two restarts use seed and seed+10000,
    including the inherited fitting-only shuffled minibatch schedules. Passing
    the same base seed and fitting rows across releases matches those schedules.
    Each returned candidate supports predict_proba and the historical save API.
    """
    started = time.perf_counter()
    resolved = _resolve_budget(budget)
    candidates, validation = {}, {}
    common = None
    for candidate_id, family, offset, minimum_leaf in _SPECIFICATIONS:
        candidate_budget = copy.deepcopy(resolved)
        if minimum_leaf is not None:
            candidate_budget["histgb"]["min_samples_leaf"] = minimum_leaf
        fitted = fit_candidates(xfit, yfit, xval, yval, n_classes, int(seed)+offset,
                                families=(family,), budget=candidate_budget)
        candidate = fitted["candidates"][family]
        candidate.metadata["candidate_id"] = candidate_id
        candidate.metadata["base_seed"] = int(seed)
        candidate.metadata["restart_index"] = MLP_RESTART_OFFSETS.index(offset) if family == "mlp" else None
        candidate.metadata["restart_seed_offset"] = offset if family == "mlp" else None
        candidate.metadata["checkpoint_criterion"] = "validation_log_loss" if family == "mlp" else "fixed_configuration"
        candidates[candidate_id] = candidate
        validation[candidate_id] = candidate.metadata["validation_scores"]
        identity = {key: fitted["metadata"][key] for key in (
            "n_classes", "class_schema", "fit_rows", "validation_rows", "input_dim",
            "fit_support", "fit_coverage_complete", "fit_weighted", "fit_hashes", "validation_hashes")}
        if common is None:
            common = identity
        elif common != identity:
            raise AssertionError("Candidate auditors did not receive identical frozen fitting/validation data")
    selections = select_candidates(validation)
    fallbacks = {key: candidate.metadata["fallback_reason"] for key, candidate in candidates.items()
                 if "fallback_reason" in candidate.metadata}
    metadata = {
        **common, "base_seed": int(seed), "budget": resolved,
        "candidate_ids": list(CANDIDATE_IDS), "mlp_restart_seed_offsets": list(MLP_RESTART_OFFSETS),
        "histogram_minimum_leaf_candidates": [20, 5],
        "validation_scores": validation, "candidates": {key: value.metadata for key, value in candidates.items()},
        **selections,
        "fallbacks": fallbacks,
        "all_candidates_fit_coverage_complete": all(value.metadata["fit_coverage_complete"] for value in candidates.values()),
        "validation_coverage_complete": all(value["coverage_complete"] for value in validation.values()),
        "all_candidates_numerically_fit": not fallbacks,
        "total_mlp_optimizer_steps": sum(value.metadata["optimizer_steps"] for key, value in candidates.items() if key.startswith("mlp_")),
        "total_mlp_training_row_exposures": sum(value.metadata["training_row_exposures"] for key, value in candidates.items() if key.startswith("mlp_")),
        "fit_runtime_seconds": time.perf_counter()-started,
        "scope": "empirical independently fitted auditors; no representation training or universal privacy guarantee",
    }
    return {"candidates": candidates, **selections, "metadata": metadata}


def save_auditors(result, directory):
    """Persist all selected candidate checkpoints and their selection record."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    for key, candidate in result["candidates"].items():
        candidate.save(directory/key)
    (directory/"selection.json").write_text(json.dumps(result["metadata"], indent=2, allow_nan=False)+"\n")
