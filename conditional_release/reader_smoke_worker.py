"""Queued data-free weighted architecture smoke, from sealed fixture laws."""
import json
from pathlib import Path
import sys
import time

import numpy as np

from .attacks import fit_complete_readers, score_frozen_readers, write_private_json
from .scheduler import assert_active_from_env


def main():
    active = assert_active_from_env(stage="admission")
    base = Path(__file__).resolve().parents[1]
    f = json.loads((base / "results/pcrl_conditional_budgeted_release_v1/r5/WIRE_READER_FIXTURES.json").read_text())
    s = json.loads((base / "results/pcrl_conditional_budgeted_release_v1/r5/WEIGHTED_READER_SMOKE_LAW.json").read_text())
    bank = {"release_id":"fixture-public-bank", "source_binding_sha256":"0" * 64,
        "recipients":{str(r):{"probability":f[f"bank_r{r}"], "bank_sha256":str(r) * 64} for r in (1, 2)}}
    def law(n, offset):
        people = np.arange(n)
        sex = (people % 5 >= 2).astype(np.int64)
        record = {"row_id":people.astype(np.int64) + offset, "unit":people.astype(np.int64) + offset,
            "code_r1":np.column_stack((2*sex, 2*sex+1)), "code_r2":np.column_stack((people%6, people%6)),
            "mass":np.tile(np.asarray(s["conditional_mass"]), (n, 1))}
        return record, {"row_id":record["row_id"], "unit":record["unit"], "sex":sex}
    train, train_y = law(s["fit_people"], 1000)
    report, report_y = law(s["report_people"], 10000)
    root = base / ".private/conditional_release/r5" / active["unit_id"]
    t0 = time.perf_counter()
    frozen = fit_complete_readers(bank, train, train_y, uniform_public_pair_mass=np.full((4, 6), 1/24),
        seed=0, private_root=root, queue_guard=lambda:assert_active_from_env(unit_id=active["unit_id"], stage="admission"),
        require_adult=False, source_bindings={"scope":"locked synthetic smoke"})
    with np.load(root / "FROZEN_SCORE_TABLES.npz", allow_pickle=False) as archive:
        scores = {key:archive[key].copy() for key in archive.files}
    result, _ = score_frozen_readers(bank, report, report_y, scores)
    if frozen["numerical_fit_count"] != s["required_models"] or not result["all_controls_pass"] or not result["ignore_other_monotonicity"]:
        raise AssertionError("weighted reader smoke controls/interface failed")
    public = {"schema":"pcrl-conditional-weighted-reader-smoke-receipt-v1", "status":"PASS", "unit_id":active["unit_id"],
        "scope":"synthetic engineering only", "real_people":0, "fits":frozen["numerical_fit_count"],
        "wall_seconds":time.perf_counter()-t0, "views":result["views"], "training_receipts":frozen["training_receipts"],
        "no_mean_outputs":True, "all_weighted_training":True}
    path = base / "results/pcrl_conditional_budgeted_release_v1/r5" / (active["unit_id"] + ".json")
    with path.open("x") as stream:
        json.dump(public, stream, sort_keys=True, indent=2);stream.write("\n")
    print(json.dumps({"status":"PASS","scope":"synthetic only","fits":30,"elapsed":public["wall_seconds"]}))


if __name__ == "__main__":
    main()
