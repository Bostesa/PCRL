"""A4: frozen forward pass (eval mode, no grad, CPU, OMP_NUM_THREADS=1) of every admitted Round-4 final.pt over
ALL regenerated train + test rows. No fitting.

Uses stored_model_eval/forward.py (the pilot's forward; imported read-only). Each split is run separately with
batch_size 512 (the pilot's call), so test batches are byte-identical to the pilot's.
Output ~/PCRL_eval_cache_private/bench_v1/inputs/<ds>_s<k>_forward.npz:
  row_id, split, rep_p<i> (float64 copies of the float32 forward), logits_<purpose> (float64 copies).
Checks: (1) two independent forward runs (fresh checkpoint load each) are bitwise identical; (2) an existing cache
file, if present, is bitwise identical to the recomputation; (3) Adult s0 test reps/logits equal the pilot cache
~/PCRL_eval_cache_private/pilot_adult_s0/cache/adult_s0_test.npz for matching row_ids.
Writes notes/admission/forward.json.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from admission_common import CKPT_DIR, DATASETS, INPUTS, NOTES, PILOT, SEEDS, WT, sha256_file, write_json  # noqa

sys.path.insert(0, str(WT))
from stored_model_eval.forward import FrozenPCRLv2, load_checkpoint, run_forward  # noqa: E402


def forward_all(ck_path, sha, X, split):
    ck, prov = load_checkpoint(ck_path, sha)
    m = FrozenPCRLv2(ck)
    parts = {}
    for sp in ("test", "train"):
        parts[sp] = run_forward(m, X[split == sp], batch_size=512)
    out = {}
    for k in parts["test"]:
        a = np.empty((len(X), parts["test"][k].shape[1]), dtype=np.float32)
        for sp in ("test", "train"):
            a[split == sp] = parts[sp][k]
        out[k] = a
    return out, m.describe(), prov


def main():
    import torch
    lin = json.loads((NOTES / "lineage.json").read_text())
    rec = {"omp_num_threads": os.environ.get("OMP_NUM_THREADS"), "torch": torch.__version__,
           "torch_num_threads": torch.get_num_threads(), "device": "cpu", "batch_size": 512, "fits_performed": 0,
           "runs": {}}
    for ds in DATASETS:
        f = np.load(INPUTS / f"{ds}_features.npz")
        X, split, row_id = f["features"], f["split"], f["row_id"]
        for s in SEEDS:
            L = lin["datasets"][ds]["seeds"][str(s)]["final"]
            if not L["admitted"]:
                rec["runs"][f"{ds}_s{s}"] = {"status": "missing", "reason": "lineage not admitted"}
                continue
            ck_path = CKPT_DIR / f"v2_{ds}_s{s}_final.pt"
            a, desc, prov = forward_all(ck_path, L["sha256"], X, split)
            b, _, _ = forward_all(ck_path, L["sha256"], X, split)
            det = all(np.array_equal(a[k], b[k]) for k in a)
            if not det:
                raise SystemExit(f"{ds} s{s}: forward not deterministic")
            target = INPUTS / f"{ds}_s{s}_forward.npz"
            prev_equal = None
            if target.exists():
                z = np.load(target)
                prev_equal = all(np.array_equal(z[k], a[k].astype(np.float64)) for k in a) and \
                    np.array_equal(z["row_id"], row_id)
            if prev_equal is not True:
                np.savez(target, row_id=row_id, split=split, **{k: v.astype(np.float64) for k, v in a.items()})
            r = {"checkpoint": str(ck_path).replace(str(Path.home()), "~"), "checkpoint_sha256": L["sha256"],
                 "load_mode": prov["load_mode"], "architecture": desc, "n_rows": int(len(row_id)),
                 "n_test": int((split == "test").sum()), "n_train": int((split == "train").sum()),
                 "arrays": {k: list(v.shape) for k, v in a.items()},
                 "deterministic_two_runs_bitwise": det, "existing_file_bitwise_equal": prev_equal,
                 "output": str(target).replace(str(Path.home()), "~"), "sha256": sha256_file(target),
                 "nonfinite": int(sum((~np.isfinite(v)).sum() for v in a.values()))}
            if ds == "adult" and s == 0:
                pz = np.load(PILOT / "cache" / "adult_s0_test.npz")
                pid = pz["row_id"]
                pos = {int(x): i for i, x in enumerate(row_id)}
                ix = np.array([pos[int(x)] for x in pid])
                cmp = {}
                for k in pz.files:
                    if k == "row_id":
                        continue
                    cmp[k] = {"bitwise_equal_float32": bool(np.array_equal(a[k][ix], pz[k])),
                              "max_abs_diff": float(np.max(np.abs(a[k][ix].astype(np.float64) - pz[k])))}
                r["pilot_cache_comparison"] = {"pilot_file_sha256": sha256_file(PILOT / "cache" / "adult_s0_test.npz"),
                                               "n_matching_row_ids": int(len(ix)),
                                               "pilot_row_ids_all_test": bool(np.all(split[ix] == "test")),
                                               "arrays": cmp,
                                               "all_bitwise_equal": all(v["bitwise_equal_float32"]
                                                                        for v in cmp.values())}
            rec["runs"][f"{ds}_s{s}"] = r
            print(ds, s, "det", det, "prev", prev_equal, r.get("pilot_cache_comparison", {}).get("all_bitwise_equal"))
    write_json(NOTES / "forward.json", rec)


if __name__ == "__main__":
    main()
