"""Queue-only bounded refit of one frozen realization-zero pair reader view."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from zipfile import ZipFile

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from results.pcrl_incumbent_view_risk_v1.r0.compute_queue import assert_active_from_env
from .r5_attack_math import auc_by_group_weights
from .r5_attack_auc_replay import row_unit_sha, sha


SPLIT_SALT = "pcrl-method-portfolio-v1|attack-tune|20261008|"


def private_arrays(path: Path, required: set[str], allowed: set[str]) -> dict[str, np.ndarray]:
    with ZipFile(path) as archive:
        members = [x.filename for x in archive.infolist()]
        names = {x[:-4] for x in members if x.endswith(".npy")}
        if len(names) != len(members) or not required.issubset(names) or not names.issubset(allowed):
            raise PermissionError("role/view archive member allowlist differs")
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in required}


def independent_split(unit: np.ndarray) -> np.ndarray:
    unique = np.unique(unit)
    if unique.size != 4061:
        raise PermissionError("ATTACK_FIT_NEW duplicate-group count differs")
    ranked = sorted((hashlib.sha256((SPLIT_SALT + str(int(g))).encode()).hexdigest(), int(g))
                    for g in unique)
    tune = np.asarray([g for _, g in ranked[:1015]], dtype=np.int64)
    return ~np.isin(unit, tune)


def point_auc(y: np.ndarray, score: np.ndarray, unit: np.ndarray) -> float:
    _, inverse = np.unique(unit, return_inverse=True)
    return float(auc_by_group_weights(y, np.asarray(score, dtype=np.float64),
                                      inverse.astype(np.int64),
                                      np.ones((1, len(np.unique(unit))), dtype=np.int64))[0])


def fit_hgb(xfit: np.ndarray, yfit: np.ndarray, xinner: np.ndarray,
            *, leaves: int, iterations: int, seed: int):
    model = HistGradientBoostingClassifier(learning_rate=0.05,
        max_leaf_nodes=leaves, max_iter=iterations, early_stopping=False,
        random_state=seed)
    model.fit(xfit, yfit)
    score = np.asarray(model.predict_proba(xinner)[:, 1], dtype=np.float64)
    return model, score


def run(config_path: str | Path) -> dict:
    active = assert_active_from_env(stage="verification")
    state = json.loads((Path(os.environ["PCRL_QUEUE_ROOT"]) / "state.json").read_text())
    sources = {str(Path(x["path"]).resolve(strict=True)): x["sha256"]
               for x in state["units"][active["unit_id"]]["spec"]["sources"]}

    def pinned(item: dict) -> Path:
        raw = Path(item["path"])
        if raw.is_symlink():
            raise PermissionError("symlinked source refused")
        path = raw.resolve(strict=True)
        if sources.get(str(path)) != item["sha256"] or sha(path) != item["sha256"]:
            raise PermissionError("R5 refit source differs from active queue pin")
        return path

    config = Path(config_path).resolve(strict=True)
    if sources.get(str(config)) != sha(config):
        raise PermissionError("R5 refit config unpinned")
    cfg = json.loads(config.read_text())
    if (cfg.get("schema") != "IVR1-r5-bounded-refit-config-v1" or
            cfg.get("unit_id") != active["unit_id"] or
            cfg.get("realization") != 0 or cfg.get("view") != "pair"):
        raise PermissionError("bounded refit scope differs")
    if cfg["queue_seal"]["sha256"] != active["seal_sha256"]:
        raise PermissionError("active queue admission seal differs")
    pinned(cfg["queue_seal"])
    science = cfg["campaign_seal"]["sha256"]
    pinned(cfg["campaign_seal"])
    if science not in {a["sha256"] for a in state["amendments"] if "protection" in a["stages"]}:
        raise PermissionError("scientific campaign seal unregistered")
    verification = [a for a in state["amendments"] if "verification" in a["stages"]]
    if (not verification or
            state["units"][active["unit_id"]]["spec"].get("amendment_sha256") != verification[-1]["sha256"] or
            json.loads(pinned(cfg["verification_amendment"]).read_text()).get("campaign_lock_sha256") != science):
        raise PermissionError("refit not bound to latest verification amendment")
    rule = json.loads(pinned(cfg["subset_rule"]).read_text())
    if (rule.get("schema") != "IVR1-r5-bounded-refit-subset-v2" or
            rule.get("status") != "FROZEN_BEFORE_FULL_FINAL_AGGREGATE" or
            rule.get("draws") != [0] or rule.get("view") != "pair" or
            rule.get("source_seeds") != [0, 1] or rule.get("maximum_units") != 6):
        raise PermissionError("fixed reader refit subset rule differs")
    selected = json.loads(pinned(cfg["r3_claims"]).read_text())
    if (selected.get("schema") != "pcrl-incumbent-r3-fixed-claim-accounting-v1" or
            selected.get("nominee") != rule["nominee"] or
            cfg["arm_id"] not in {"P0", rule["nominee"],
                                  selected["strongest_control"].get("selected")} or
            cfg["source_seed"] not in rule["source_seeds"]):
        raise PermissionError("refit arm/seed differs from predeclared subset rule")
    receipt = json.loads(pinned(cfg["r4_receipt"]).read_text())
    if (receipt.get("status") != "COMPLETE_PRIVATE_DEVELOPMENT_ONLY" or
            receipt.get("arm_id") != cfg["arm_id"] or
            receipt.get("source_seed") != cfg["source_seed"] or
            receipt.get("realization") != 0 or
            receipt.get("campaign_seal_sha256") != science or
            receipt.get("queue_seal_sha256") != active["seal_sha256"]):
        raise PermissionError("selected R4 refit release differs")
    frozen_path = pinned(cfg["frozen"])
    frozen = json.loads(frozen_path.read_text())
    if (frozen.get("status") != "FROZEN_COMPLETE" or
            frozen.get("effective_lock_sha256") != science or
            receipt["attack"]["freeze_receipt_sha256"] != sha(frozen_path)):
        raise PermissionError("reader slate not frozen before label access")
    pair = cfg["pair_view"]
    reader_path = pinned(pair["reader_receipt"])
    reader = json.loads(reader_path.read_text())
    att_path = pinned(pair["att_view"])
    inner_path = pinned(pair["inner_view"])
    predictions_path = pinned(pair["predictions"])
    if (reader.get("status") != "FROZEN" or
            reader["files"]["view_att.npz"] != sha(att_path) or
            reader["files"]["view_inner.npz"] != sha(inner_path) or
            reader["files"]["predictions_inner.npz"] != sha(predictions_path) or
            frozen["reader_receipt_sha256"]["pair"] != sha(reader_path) or
            "HGB_15x200" not in reader["reader_order"]):
        raise PermissionError("frozen pair reader/view custody differs")
    manifest = json.loads(pinned(cfg["role_manifest"]).read_text())
    for role, field in (("ATTACK_FIT_NEW", "att_sex"),
                        ("INNER_SELECTION", "inner_sex")):
        if cfg[field]["sha256"] != manifest["roles"][role]["source_files"]["sex"]["sha256"]:
            raise PermissionError("SEX role source outside allowlist")
    # Only now load ATT/INNER SEX and public-derived matrices.
    att_sex = private_arrays(pinned(cfg["att_sex"]),
                             {"row_id", "unit", "sex"}, {"row_id", "unit", "sex"})
    inner_sex = private_arrays(pinned(cfg["inner_sex"]),
                               {"row_id", "unit", "sex"}, {"row_id", "unit", "sex"})
    att = private_arrays(att_path, {"row_id", "unit", "x"},
                         {"row_id", "unit", "x", "x_defense_aware", "cell_key",
                          "x_metadata", "x_code_bits", "x_raw_only"})
    inner = private_arrays(inner_path, {"row_id", "unit", "x"},
                           {"row_id", "unit", "x", "x_defense_aware", "cell_key",
                            "x_metadata", "x_code_bits", "x_raw_only"})
    for role, labels, view in (("ATTACK_FIT_NEW", att_sex, att),
                               ("INNER_SELECTION", inner_sex, inner)):
        meta = manifest["roles"][role]
        if (not np.array_equal(labels["row_id"], view["row_id"]) or
                not np.array_equal(labels["unit"], view["unit"]) or
                labels["row_id"].dtype != np.int64 or labels["unit"].dtype != np.int64 or
                len(labels["row_id"]) != meta["rows"] or
                len(np.unique(labels["unit"])) != meta["groups"] or
                row_unit_sha(labels["row_id"], labels["unit"]) != meta["row_unit_pair_sha256"] or
                not np.isin(labels["sex"], (0, 1)).all()):
            raise PermissionError("fresh pair view / SEX row alignment differs")
    xfit, xinner = np.asarray(att["x"]), np.asarray(inner["x"])
    if (xfit.dtype != np.float64 or xinner.dtype != np.float64 or
            xfit.ndim != 2 or xinner.ndim != 2 or xfit.shape[1] != xinner.shape[1] or
            not np.isfinite(xfit).all() or not np.isfinite(xinner).all()):
        raise PermissionError("complete pair feature matrix differs")
    train = independent_split(att["unit"])
    yfit = att_sex["sex"].astype(np.int64)
    yinner = inner_sex["sex"].astype(np.int64)
    if np.unique(yfit[train]).size != 2:
        raise ValueError("fixed ATT train split lacks a SEX class")
    assert_active_from_env(unit_id=active["unit_id"], stage="verification")
    common_model, common_score = fit_hgb(xfit[train], yfit[train], xinner,
        leaves=15, iterations=200, seed=receipt["attacker_seed"])
    with np.load(predictions_path, allow_pickle=False) as predictions:
        if "HGB_15x200" not in predictions.files:
            raise PermissionError("frozen common HGB prediction is absent")
        frozen_score = np.asarray(predictions["HGB_15x200"])
    common_gap = float(np.max(np.abs(common_score - frozen_score)))
    if common_gap > 1e-12:
        raise ArithmeticError("independently refitted common reader differs")
    assert_active_from_env(unit_id=active["unit_id"], stage="verification")
    stronger_model, stronger_score = fit_hgb(xfit[train], yfit[train], xinner,
        leaves=31, iterations=300, seed=20261010 + cfg["source_seed"])
    common_auc = point_auc(yinner, common_score, inner["unit"])
    stronger_auc = point_auc(yinner, stronger_score, inner["unit"])
    private = Path(cfg["output_private_dir"]).resolve()
    public = Path(cfg["output_public"]).resolve()
    owner = Path(__file__).resolve().parents[1] / "results/pcrl_incumbent_view_risk_v1/r5"
    if (private.parent != owner / ".private" or public.parent != owner or
            private.exists() or public.exists()):
        raise PermissionError("write-once R5 refit output differs")
    private.mkdir(mode=0o700)
    files = {}
    for name, model in (("common", common_model), ("stronger", stronger_model)):
        path = private / f"{name}.joblib"
        joblib.dump(model, path, compress=3)
        path.chmod(0o600)
        files[name] = sha(path)
    predpath = private / "predictions.npz"
    with os.fdopen(os.open(predpath, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
        np.savez_compressed(stream, common=common_score, stronger=stronger_score)
    files["predictions"] = sha(predpath)
    output = {"schema": "IVR1-r5-bounded-reader-refit-v2", "status": "PASS",
              "unit_id": active["unit_id"], "arm_id": cfg["arm_id"],
              "source_seed": cfg["source_seed"], "realization": 0, "view": "pair",
              "att_train_rows": int(train.sum()), "att_tune_rows": int((~train).sum()),
              "inner_rows": len(yinner), "common_hgb_max_prediction_gap": common_gap,
              "common_hgb_auc": max(common_auc, 1 - common_auc),
              "stronger_hgb_auc": max(stronger_auc, 1 - stronger_auc),
              "private_files_sha256": files, "r4_receipt_sha256": sha(pinned(cfg["r4_receipt"])),
              "subset_rule_sha256": sha(pinned(cfg["subset_rule"])),
              "not_confirmation": True, "queue": active}
    with os.fdopen(os.open(public, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644), "w") as stream:
        json.dump(output, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    record = run(args.config)
    print(json.dumps({"unit_id": record["unit_id"], "status": record["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
