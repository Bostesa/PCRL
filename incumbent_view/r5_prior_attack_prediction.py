"""Independent replay of the frozen historical P* pair reader on actual P0 wires.

No sensitive labels are read and no attacker is fitted. This is a source
custody/deployment check, not a new privacy result.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from zipfile import ZipFile

import joblib
import numpy as np

from results.pcrl_incumbent_view_risk_v1.r0.compute_queue import assert_active_from_env

WIRE = {"row_id", "unit", "code_r1", "prob_r1", "decision_r1",
        "code_r2", "prob_r2", "decision_r2"}
VIEW = {"row_id", "unit", "x", "x_defense_aware", "cell_key",
        "x_metadata", "x_code_bits"}
READER = "PROB_HGB_15x200"


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load_member(path, allowed, key):
    with ZipFile(path) as z:
        names = [x.filename for x in z.infolist()]
        if len(names) != len(allowed) or set(names) != {x+".npy" for x in allowed}:
            raise PermissionError("private NPZ member allowlist differs before load")
    with np.load(path, allow_pickle=False) as a:
        return np.array(a[key], copy=True)


def run(config_path):
    active = assert_active_from_env(stage="verification")
    state = json.loads((Path(os.environ["PCRL_QUEUE_ROOT"])/"state.json").read_text())
    registered = {x["path"]: x["sha256"] for x in state["units"][active["unit_id"]]["spec"]["sources"]}

    def pinned(path):
        p = Path(path)
        if registered.get(str(p)) != sha(p):
            raise PermissionError("source not byte-pinned in active queue")
        return p

    cfg = json.loads(pinned(config_path).read_text())
    if (cfg.get("schema") != "IVR1-r5-historical-pstar-prediction-config-v1"
            or cfg.get("unit_id") != active["unit_id"]
            or cfg.get("source_seed") not in (0, 1)):
        raise PermissionError("historical replay configuration differs")
    seed = cfg["source_seed"]
    idx = json.loads(pinned(cfg["historical_index"]).read_text())["seeds"][str(seed)]
    old = idx["historical"]
    if old["validation_winner"] != READER or old["orientation"] != "positive":
        raise PermissionError("historical selected reader identity differs")
    for name in ("model", "model_custody_alias", "predictions_inner", "view_inner",
                 "reader_receipt", "public_attack_receipt"):
        if sha(pinned(cfg[name])) != old[name]["sha256"]:
            raise PermissionError("historical reader/prediction source bytes differ")
    if sha(cfg["model"]) != sha(cfg["model_custody_alias"]):
        raise PermissionError("historical reader custody alias differs")
    if (sha(pinned(cfg["p0_inner_release"])) !=
            idx["p0_role_only"]["INNER_SELECTION"]["release"]["sha256"]
            or sha(pinned(cfg["p0_inner_receipt"])) !=
            idx["p0_role_only"]["INNER_SELECTION"]["receipt"]["sha256"]):
        raise PermissionError("actual P0 INNER role binding differs")
    reader_receipt = json.loads(Path(cfg["reader_receipt"]).read_text())
    if (reader_receipt.get("validation_winner") != READER or
            reader_receipt.get("validation_winner_orientation") != "positive" or
            reader_receipt["files"]["predictions_inner.npz"] != sha(cfg["predictions_inner"]) or
            reader_receipt["files"]["view_inner.npz"] != sha(cfg["view_inner"])):
        raise PermissionError("historical frozen reader receipt differs")
    public = json.loads(Path(cfg["public_attack_receipt"]).read_text())
    if (public.get("arm_id") != "P_STAR" or public.get("source_seed") != seed or
            public["attack"]["scores"]["pair"]["validation_winner"] != READER):
        raise PermissionError("public historical selected-reader receipt differs")

    wire = {k: load_member(cfg["p0_inner_release"], WIRE, k)
            for k in ("row_id", "unit", "prob_r1", "prob_r2")}
    old_row = load_member(cfg["view_inner"], VIEW, "row_id")
    old_unit = load_member(cfg["view_inner"], VIEW, "unit")
    if (not np.array_equal(wire["row_id"], old_row)
            or not np.array_equal(wire["unit"], old_unit)):
        raise PermissionError("historical reader and new exact P0 people differ")
    feature = np.concatenate((wire["prob_r1"], wire["prob_r2"]), axis=1)
    if feature.shape != (2235, 8) or not np.isfinite(feature).all():
        raise PermissionError("P0 reader probability interface differs")
    model = joblib.load(cfg["model"])
    actual = np.asarray(model.predict_proba(feature)[:, 1], dtype=np.float64)
    saved = load_member(cfg["predictions_inner"], set(reader_receipt["reader_order"]), READER)
    if (actual.shape != saved.shape or not np.isfinite(actual).all()
            or not np.all((actual >= 0) & (actual <= 1))):
        raise ValueError("frozen P* model produced invalid predictions")
    gap = float(np.max(np.abs(actual-saved)))
    if gap > 1e-12:
        raise ValueError("actual P0 wire reader predictions differ from prior frozen predictions")
    result = {"schema": "IVR1-r5-historical-pstar-frozen-prediction-replay-v1",
              "status": "PASS", "unit_id": active["unit_id"], "source_seed": seed,
              "rows": len(actual), "reader": READER, "orientation": "positive",
              "max_prediction_gap": gap, "source_index_sha256": sha(cfg["historical_index"]),
              "model_sha256": sha(cfg["model"]),
              "original_predictions_sha256": sha(cfg["predictions_inner"]),
              "actual_p0_inner_release_sha256": sha(cfg["p0_inner_release"]),
              "label_fields_read": [], "new_reader_fits": 0,
              "timing": "AFTER_FIRST_NEW_FIT_STARTED; retroactive source check only"}
    out = Path(cfg["output"])
    with os.fdopen(os.open(out, os.O_CREAT|os.O_EXCL|os.O_WRONLY, 0o644), "w") as f:
        json.dump(result, f, sort_keys=True, indent=2); f.write("\n")
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument("--config", required=True)
    print(json.dumps(run(p.parse_args().config), sort_keys=True))


if __name__ == "__main__":
    main()
