"""One fixed 360-epoch audit extension with nested 120-epoch selections.

Historical helpers and their fixed budgets remain unchanged. These trajectories
use their exact initialization, preprocessing, Adam defaults, minibatch order,
and validation rule. The first budget is a prefix of the second; a selected
checkpoint is never restored into the ongoing optimization trajectory. APIs
accept fitting and validation data only, including for exposed-label controls.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments.acs_bottleneck_catchup import CATCHUP_CONFIG, fit_catchup
from experiments.acs_protection_audits import (
    AUDIT_BUDGET, MLP_RESTART_OFFSETS, _resolve_budget, select_candidates,
)
from experiments.acs_transfer_heads import (
    FittedCandidate, InputStandardizer, _features, _hash_array, _labels,
    _network, _state_hash, fit_candidates, metrics,
)


EXTENDED_AUDIT_CONFIG = {
    "epochs": 360, "nested_epochs": 120,
    "fresh_candidates": ["mlp_0", "mlp_1"],
    "reused_candidates": ["logistic", "hist_gb_20", "hist_gb_5"],
    "initialization": "refit prescribed 360-epoch trajectory from original start",
    "selection": "minimum attacker-validation log loss, then earliest epoch",
    "budget_scope": "additional fitting budget only; unchanged rows and families",
}


def _validate_epochs(epochs, nested_epochs):
    if (not isinstance(epochs, (int, np.integer))
            or not isinstance(nested_epochs, (int, np.integer))
            or not 0 < nested_epochs < epochs):
        raise ValueError("Require integer 0 < nested_epochs < epochs")


def _trajectory(model, standardizer, xf, yf, xv, yv, n_classes, seed,
                parameters, common, nested_epochs, *, catchup=False):
    """Run once, retaining actual last state/Adam/RNG at both budget boundaries."""
    started = time.perf_counter()
    epochs = parameters["epochs"]
    model = copy.deepcopy(model).train().requires_grad_(True)
    candidate = FittedCandidate("mlp", model, standardizer, n_classes, {})
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters["lr"],
                                 weight_decay=parameters["weight_decay"])
    assert not optimizer.state, "Extension must use the declared Adam reset"
    xt = torch.as_tensor(standardizer.transform(xf), dtype=torch.float32)
    yt = torch.as_tensor(yf, dtype=torch.long)
    initial_hash = _state_hash(model)
    rng = np.random.default_rng(int(seed)+700000)
    digest = hashlib.sha256()
    exposures = np.zeros(len(yf), dtype=np.int64)
    steps, best_steps = 0, 0
    curve, best_state, best_key = [], None, (float("inf"), -1)
    candidates, checkpoints = {}, {}

    def evaluate(epoch):
        nonlocal best_state, best_key, best_steps
        scores = metrics(yv, candidate.predict_proba(xv), n_classes)
        row = {"epoch": epoch, "optimizer_steps": steps,
               "validation_log_loss": scores["log_loss"]}
        if catchup:
            row.update(validation_auroc=scores["auroc"],
                       validation_coverage_complete=scores["coverage_complete"])
        curve.append(row)
        key = scores["log_loss"], epoch
        if key < best_key:
            best_key, best_steps = key, steps
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def freeze_boundary(epoch):
        # This copy does not alter the live model, optimizer, or schedule RNG.
        frozen = copy.deepcopy(model)
        frozen.load_state_dict(best_state)
        frozen.eval().requires_grad_(False)
        picked = FittedCandidate("mlp", frozen, copy.deepcopy(standardizer), n_classes, {})
        scores = metrics(yv, picked.predict_proba(xv), n_classes)
        assert scores["log_loss"] == best_key[0], "Selected checkpoint failed replay"
        optimizer_steps = [int(v["step"].item()) for v in optimizer.state.values()]
        assert all(v == steps for v in optimizer_steps)
        picked.metadata = {
            **copy.deepcopy(common), "family": "mlp",
            "parameters": {**copy.deepcopy(parameters), "epochs": epoch},
            "optimizer": {"name": "Adam", "lr": parameters["lr"],
                          "betas": [.9, .999], "eps": 1e-8,
                          "weight_decay": parameters["weight_decay"],
                          "restored_state": False, "initial_state_entries": 0,
                          "final_state_entries": len(optimizer.state),
                          "final_state_steps": optimizer_steps},
            "initial_state_hash": initial_hash, "final_state_hash": _state_hash(model),
            "selected_state_hash": _state_hash(frozen), "initialization_seed": int(seed),
            "schedule_seed": int(seed)+700000, "schedule_hash": digest.hexdigest(),
            "restarts": 1, "optimizer_steps": steps,
            "selected_optimizer_steps": best_steps, "selected_epoch": best_key[1],
            "training_row_exposures": int(exposures.sum()),
            "row_exposure_min": int(exposures.min()), "row_exposure_max": int(exposures.max()),
            "parameter_count": sum(p.numel() for p in frozen.parameters()),
            "validation_curve": copy.deepcopy(curve), "validation_scores": scores,
            "selection": EXTENDED_AUDIT_CONFIG["selection"],
            "trajectory_epochs": epochs, "nested_budget_epochs": epoch,
            "trajectory_continues_from": "actual last iterate and Adam state; no best-checkpoint rewind",
            "fit_runtime_seconds": time.perf_counter()-started,
        }
        if catchup:
            picked.metadata.update(audit_kind="catchup", candidate_id="catchup")
        candidates[epoch] = picked
        checkpoints[epoch] = {
            "epoch": epoch, "optimizer_steps": steps,
            "model_state": copy.deepcopy(model.state_dict()),
            "optimizer_state": copy.deepcopy(optimizer.state_dict()),
            "schedule_rng_state": copy.deepcopy(rng.bit_generator.state),
            "schedule_hash": digest.hexdigest(), "state_hash": _state_hash(model),
            "selected_state_hash": _state_hash(frozen),
            "checkpoint_kind": "actual last training state, not validation-selected state",
        }

    evaluate(0)
    for epoch in range(1, epochs+1):
        order = rng.permutation(len(yf))
        digest.update(order.tobytes())
        for start in range(0, len(order), parameters["batch_size"]):
            rows = order[start:start+parameters["batch_size"]]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(xt[rows]), yt[rows])
            if not torch.isfinite(loss):
                error = FloatingPointError("Nonfinite extended-audit fitting loss")
                error.fit_accounting = {"optimizer_steps": steps,
                                        "training_row_exposures": int(exposures.sum())}
                raise error
            loss.backward()
            optimizer.step()
            steps += 1
            exposures[rows] += 1
        if epoch % parameters["validation_interval"] == 0 or epoch in (nested_epochs, epochs):
            evaluate(epoch)
        if epoch in (nested_epochs, epochs):
            freeze_boundary(epoch)
    assert candidates[epochs].metadata["validation_scores"]["log_loss"] <= candidates[nested_epochs].metadata["validation_scores"]["log_loss"]
    return candidates, checkpoints


def fit_extended_auditors(xfit, yfit, xval, yval, n_classes, seed, *,
                          static_candidates=None, epochs=360, nested_epochs=120,
                          budget=None):
    """Fit two fresh nested MLPs; reuse supplied matching logistic/tree objects.

    Smaller epoch/budget arguments serve artificial fixtures only. The same API
    extends the exposed-label controls by passing their original one-hot inputs.
    The nested120/nested360 key names denote lower/upper budgets in fixtures too.
    """
    started = time.perf_counter()
    _validate_epochs(epochs, nested_epochs)
    cfg = _resolve_budget(budget)
    cfg["mlp"]["epochs"] = int(epochs)
    xf, xv = _features(xfit), _features(xval)
    yf, yv = _labels(yfit, n_classes), _labels(yval, n_classes)
    if (len(xf) != len(yf) or len(xv) != len(yv) or not len(yf) or not len(yv)
            or xf.shape[1] != xv.shape[1]):
        raise ValueError("Aligned nonempty fitting/validation rows and feature schemas required")
    support = np.bincount(yf, minlength=n_classes)
    common = {"n_classes": int(n_classes), "class_schema": list(range(n_classes)),
              "fit_rows": len(yf), "validation_rows": len(yv), "input_dim": xf.shape[1],
              "fit_support": support.tolist(), "fit_coverage_complete": bool((support > 0).all()),
              "fit_weighted": False, "seed": int(seed), "base_seed": int(seed), "device": "cpu",
              "fit_hashes": {"x": _hash_array(xf), "y": _hash_array(yf)},
              "validation_hashes": {"x": _hash_array(xv), "y": _hash_array(yv)}}
    statics = dict(static_candidates or {})
    if set(statics)-set(EXTENDED_AUDIT_CONFIG["reused_candidates"]):
        raise ValueError("Only original logistic/tree candidates may be reused")
    for cid, candidate in statics.items():
        for key in ("fit_hashes", "validation_hashes", "n_classes", "seed"):
            if candidate.metadata[key] != common[key]:
                raise AssertionError("Static auditor recipe/input mismatch: "+cid+"/"+key)
        if candidate.metadata["parameters"] != cfg[candidate.metadata["family"]]:
            expected = copy.deepcopy(cfg[candidate.metadata["family"]])
            if cid == "hist_gb_5":
                expected["min_samples_leaf"] = 5
            if candidate.metadata["parameters"] != expected:
                raise AssertionError("Static auditor configuration changed: "+cid)
        if candidate.metadata["validation_scores"] != metrics(yv, candidate.predict_proba(xv), n_classes):
            raise AssertionError("Static auditor validation replay failed: "+cid)
    outputs = {nested_epochs: dict(statics), epochs: dict(statics)}
    checkpoints = {}
    fallback = not np.isfinite(xf).all() or not np.isfinite(xv).all() or (support > 0).sum() < 2
    for index, offset in enumerate(MLP_RESTART_OFFSETS):
        cid, candidate_seed = "mlp_"+str(index), int(seed)+offset
        extra = {**common, "seed": candidate_seed, "candidate_id": cid,
                 "restart_index": index, "restart_seed_offset": offset,
                 "checkpoint_criterion": "validation_log_loss"}
        if fallback:
            old = fit_candidates(xf, yf, xv, yv, n_classes, candidate_seed,
                                 families=("mlp",), budget=cfg)["candidates"]["mlp"]
            assert "fallback_reason" in old.metadata
            for boundary in outputs:
                picked = copy.deepcopy(old)
                picked.metadata.update(extra)
                picked.metadata["parameters"]["epochs"] = boundary
                picked.metadata["nested_budget_epochs"] = boundary
                outputs[boundary][cid] = picked
            checkpoints[cid] = {}
            continue
        standardizer = InputStandardizer.fit(xf, cfg["preprocessing"]["std_floor"])
        if not np.isfinite(standardizer.mean).all() or not np.isfinite(standardizer.scale).all():
            raise FloatingPointError("Nonfinite extended-audit fitting standardizer")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(candidate_seed)
            model = _network(xf.shape[1], cfg["mlp"]["hidden"], n_classes).float()
        fitted, checkpoints[cid] = _trajectory(model, standardizer, xf, yf, xv, yv,
            n_classes, candidate_seed, cfg["mlp"], extra, int(nested_epochs))
        for boundary in outputs:
            outputs[boundary][cid] = fitted[boundary]
    result = {}
    for key, boundary in (("nested120", nested_epochs), ("nested360", epochs)):
        candidates = outputs[boundary]
        scores = {cid: c.metadata["validation_scores"] for cid, c in candidates.items()}
        selections = select_candidates(scores)
        metadata = {**common, "budget": {**copy.deepcopy(cfg), "mlp": {**copy.deepcopy(cfg["mlp"]), "epochs": boundary}},
            "candidate_ids": list(candidates), "validation_scores": scores, **selections,
            "candidates": {cid: c.metadata for cid, c in candidates.items()},
            "reused_candidate_ids": list(statics), "nested_budget_epochs": boundary,
            "fallbacks": {cid: c.metadata["fallback_reason"] for cid, c in candidates.items() if "fallback_reason" in c.metadata},
            "total_mlp_optimizer_steps": sum(c.metadata["optimizer_steps"] for cid, c in candidates.items() if cid.startswith("mlp_")),
            "total_mlp_training_row_exposures": sum(c.metadata["training_row_exposures"] for cid, c in candidates.items() if cid.startswith("mlp_"))}
        result[key] = {"candidates": candidates, **selections, "metadata": metadata}
    result["metadata"] = {"config": {**copy.deepcopy(EXTENDED_AUDIT_CONFIG), "epochs": epochs, "nested_epochs": nested_epochs},
        "nested120": result["nested120"]["metadata"], "nested360": result["nested360"]["metadata"],
        "actual_new_optimizer_steps": result["nested360"]["metadata"]["total_mlp_optimizer_steps"],
        "actual_new_training_row_exposures": result["nested360"]["metadata"]["total_mlp_training_row_exposures"],
        "refit_from_original_start": True, "first_budget_does_not_add_separate_fitting": True,
        "fit_runtime_seconds": time.perf_counter()-started}
    result["training_checkpoints"] = checkpoints
    return result


def fit_extended_catchup(saved_model, xfit, yfit, xval, yval, n_classes, seed, *,
                         epochs=360, nested_epochs=120, inherited_exposure=None):
    """Reset Adam at the exact saved adversary, then train one nested trajectory."""
    started = time.perf_counter()
    _validate_epochs(epochs, nested_epochs)
    # Historical epoch-zero helper verifies architecture, direct predictions,
    # incoming tensor/mode immutability, and the declared fresh Adam start.
    original = fit_catchup(saved_model, xfit, yfit, xval, yval, n_classes, seed,
                           epochs=0, inherited_exposure=inherited_exposure)
    initial_hash = _state_hash(saved_model)
    xf, xv = _features(xfit), _features(xval)
    yf, yv = _labels(yfit, n_classes), _labels(yval, n_classes)
    common = copy.deepcopy(original["catchup"].metadata)
    parameters = {**copy.deepcopy(CATCHUP_CONFIG), "epochs": int(epochs)}
    candidates, checkpoints = _trajectory(saved_model, original["saved"].preprocessing,
        xf, yf, xv, yv, n_classes, int(seed), parameters, common, int(nested_epochs), catchup=True)
    assert _state_hash(saved_model) == initial_hash, "Incoming adversary changed"
    for candidate in candidates.values():
        assert candidate.metadata["validation_curve"][0]["validation_log_loss"] == original["saved"].metadata["validation_scores"]["log_loss"]
    metadata = {"config": {**parameters, "nested_epochs": nested_epochs},
                "source_state_hash": initial_hash, "source_unchanged": True,
                "initial_fidelity_exact": True, "identity_coordinates_preserved": True,
                "catchup_optimizer_reset": True, "inherited_exposure": copy.deepcopy(inherited_exposure),
                "refit_from_original_start": True, "first_budget_does_not_add_separate_fitting": True,
                "candidates": {"saved": original["saved"].metadata,
                               "nested120": candidates[nested_epochs].metadata,
                               "nested360": candidates[epochs].metadata},
                "actual_new_optimizer_steps": candidates[epochs].metadata["optimizer_steps"],
                "actual_new_training_row_exposures": candidates[epochs].metadata["training_row_exposures"],
                "fit_runtime_seconds": time.perf_counter()-started}
    return {"saved": original["saved"], "nested120": candidates[nested_epochs],
            "nested360": candidates[epochs], "metadata": metadata,
            "training_checkpoints": {"catchup": checkpoints}}


def save_extended_audits(result, directory):
    """Persist all selected candidates, curves and actual terminal checkpoints."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    for budget in ("nested120", "nested360"):
        value = result[budget]
        candidates = value["candidates"] if isinstance(value, dict) else {"catchup": value}
        for cid, candidate in candidates.items():
            candidate.save(directory/budget/cid)
    if "saved" in result:
        result["saved"].save(directory/"saved")
    checkpoint_files = []
    for cid, boundaries in result["training_checkpoints"].items():
        for epoch, checkpoint in boundaries.items():
            path = directory/f"last_training_{cid}_epoch{epoch}.pt"
            torch.save(checkpoint, path)
            checkpoint_files.append({"path": str(path), "epoch": epoch, "candidate_id": cid,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "state_hash": checkpoint["state_hash"], "optimizer_steps": checkpoint["optimizer_steps"]})
    (directory/"selection.json").write_text(json.dumps(result["metadata"], indent=2, allow_nan=False)+"\n")
    (directory/"last_training_checkpoints.json").write_text(json.dumps(checkpoint_files, indent=2, allow_nan=False)+"\n")
    return checkpoint_files
