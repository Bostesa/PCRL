"""Queue-only bounded independent HGB refit on a frozen coalition view.

The subset is fixed in r5/REFIT_SUBSET_RULE.json before attack results. This
worker imports no R4 reader fitting code and runs both configurations on the
same ATT train people and cached INNER release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .r5_independent import independent_auc
from .r5_replay_attack import (_archive, _role_labels, _tune_positions,
                               replay as replay_frozen_attack, sha)


def fit_fixed_hgb(x_train: np.ndarray, y_train: np.ndarray,
                  x_inner: np.ndarray, y_inner: np.ndarray, *,
                  max_leaf_nodes: int, max_iter: int, seed: int) -> dict:
    x0, x1 = np.asarray(x_train), np.asarray(x_inner)
    y0, y1 = np.asarray(y_train), np.asarray(y_inner)
    if (x0.dtype != np.float64 or x1.dtype != np.float64 or
            x0.ndim != 2 or x1.ndim != 2 or x0.shape[1] != x1.shape[1] or
            y0.dtype != np.int64 or y1.dtype != np.int64 or
            y0.shape != (len(x0),) or y1.shape != (len(x1),) or
            not np.isfinite(x0).all() or not np.isfinite(x1).all() or
            not np.isin(y0, (0, 1)).all() or not np.isin(y1, (0, 1)).all()):
        raise ValueError("complete-view HGB training/evaluation schema differs")
    model = HistGradientBoostingClassifier(
        learning_rate=.05, max_leaf_nodes=max_leaf_nodes,
        max_iter=max_iter, early_stopping=False, random_state=seed)
    model.fit(x0, y0)
    prediction = np.asarray(model.predict_proba(x1)[:, 1], dtype=np.float64)
    auc = independent_auc(y1, prediction)
    return {"model": model, "prediction": prediction,
            "positive_auc": auc, "alias_max_auc": max(auc, 1 - auc)}


def run(spec_path: Path, spec_sha: str, receipt_path: Path,
        receipt_sha: str, rule_path: Path, rule_sha: str,
        private_out: Path) -> dict:
    if sha(rule_path) != rule_sha:
        raise PermissionError("prefit R5 refit subset rule bytes changed")
    rule = json.loads(rule_path.read_text())
    if (rule.get("schema") != "BAR1-r5-prefit-reader-refit-subset-rule-v1" or
            rule.get("status") != "FIXED_BEFORE_FRESH_ATTACK_RESULTS" or
            rule.get("units", {}).get("view") != "pair" or
            rule.get("units", {}).get("cached_realizations") != [0]):
        raise PermissionError("R5 fixed refit scope differs")
    frozen = replay_frozen_attack(spec_path, spec_sha, receipt_path, receipt_sha)
    spec = json.loads(spec_path.read_text())
    public = json.loads(receipt_path.read_text())
    if spec["realization"] != 0:
        raise PermissionError("R5 refit cannot select best stochastic realization")
    reader_dir = Path(spec["private_output_path"]) / "pair" / "readers"
    att_view = _archive(reader_dir / "view_att.npz")
    inner_view = _archive(reader_dir / "view_inner.npz")
    manifest = Path(spec["role_manifest_path"])
    sex_att = _role_labels(manifest, spec["role_manifest_sha256"], "ATTACK_FIT_NEW")
    sex_inner = _role_labels(manifest, spec["role_manifest_sha256"], "INNER_SELECTION")
    for view, labels in ((att_view, sex_att), (inner_view, sex_inner)):
        if (not np.array_equal(view["row_id"], labels["row_id"]) or
                not np.array_equal(view["unit"], labels["unit"])):
            raise PermissionError("fresh refit view/label people differ")
    tune_pos, _ = _tune_positions(sex_att["unit"])
    train_mask = np.ones(len(sex_att["unit"]), dtype=bool)
    train_mask[tune_pos] = False
    if not np.unique(sex_att["sex"][train_mask]).size == 2:
        raise ValueError("fresh reader ATT train labels lack both classes")
    xtrain, ytrain = att_view["x"][train_mask], sex_att["sex"][train_mask]
    xinner, yinner = inner_view["x"], sex_inner["sex"]
    replayed = fit_fixed_hgb(xtrain, ytrain, xinner, yinner,
                             max_leaf_nodes=15, max_iter=200,
                             seed=spec["attacker_seed"])
    frozen_predictions = _archive(reader_dir / "predictions_inner.npz")
    if "HGB_15x200" not in frozen_predictions:
        raise PermissionError("frozen common nonlinear reader missing")
    gap = float(np.max(np.abs(replayed["prediction"] -
                              frozen_predictions["HGB_15x200"])))
    reported = public["attack"]["scores"]["pair"]["readers"][
        "HGB_15x200"]["fixed_alias_max_auc"]
    if gap > 1e-12 or abs(replayed["alias_max_auc"] - float(reported)) > 1e-11:
        raise ValueError("independent common HGB refit differs from frozen reader")
    diagnostic = fit_fixed_hgb(xtrain, ytrain, xinner, yinner,
                               max_leaf_nodes=31, max_iter=300,
                               seed=20261010 + spec["source_seed"])
    expected_root = Path(__file__).resolve().parents[1] / \
        "results/pcrl_bank_anchored_release_v1/r5/.private"
    expected_root.mkdir(mode=0o700, exist_ok=True)
    if expected_root.is_symlink() or expected_root.stat().st_mode & 0o077:
        raise PermissionError("R5 private refit parent custody differs")
    if (private_out.parent.resolve(strict=True) != expected_root.resolve(strict=True)
            or private_out.exists() or private_out.is_symlink()):
        raise PermissionError("private R5 refit artifact directory differs")
    private_out.mkdir(mode=0o700)
    model_hash = {}
    for name, result in (("replay", replayed), ("diagnostic", diagnostic)):
        path = private_out / f"model_{name}.joblib"
        joblib.dump(result["model"], path, compress=3)
        path.chmod(0o600)
        model_hash[name] = sha(path)
    return {"schema": "BAR1-r5-bounded-fresh-reader-refit-v1",
            "status": "PASS", "arm_id": spec["arm_id"],
            "source_seed": spec["source_seed"], "realization": 0,
            "view": "pair", "r4_unit_id": spec["unit_id"],
            "r4_receipt_sha256": receipt_sha,
            "prefit_subset_rule_sha256": rule_sha,
            "frozen_attack_replay_status": frozen["status"],
            "att_train_rows": int(np.count_nonzero(train_mask)),
            "att_tune_rows": len(tune_pos),
            "inner_rows": len(yinner),
            "common_hgb_alias_auc": replayed["alias_max_auc"],
            "common_hgb_max_prediction_gap": gap,
            "stronger_hgb_alias_auc": diagnostic["alias_max_auc"],
            "private_model_sha256": model_hash,
            "scope": "fixed realization-zero coalition development diagnostic only"}


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for name in ("r4-spec", "r4-spec-sha256", "r4-receipt",
                 "r4-receipt-sha256", "rule", "rule-sha256",
                 "private-output-dir", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    from results.pcrl_bank_anchored_release_v1.r0.compute_queue import assert_active_from_env
    active = assert_active_from_env(stage="verification")
    result = run(Path(args.r4_spec), args.r4_spec_sha256,
                 Path(args.r4_receipt), args.r4_receipt_sha256,
                 Path(args.rule), args.rule_sha256,
                 Path(args.private_output_dir))
    result["unit_id"] = active["unit_id"]
    output = Path(args.output).resolve(strict=False)
    expected = Path(__file__).resolve().parents[1] / "results/pcrl_bank_anchored_release_v1/r5"
    if output.parent != expected or output.exists():
        raise PermissionError("R5 refit public receipt output path differs")
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
