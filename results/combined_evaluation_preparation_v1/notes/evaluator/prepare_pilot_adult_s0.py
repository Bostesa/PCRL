"""PREPARED (not executed) scientific pilot: attacker slate on the stored PCRL Round-4 Adult seed-0 release.

Default = --dry-run: loads the local Adult test split through the archived PCRL b96c412 pipeline, prints row
counts / roles / class support, writes NOTHING and fits NOTHING.
--write-inputs (allowed preparation, not run in this session): writes features/labels/roles/units/record-key
arrays + manifests to ~/PCRL_eval_cache_private/pilot_adult_s0/ (outside git) and prints the commands:
  1. forward  (frozen, eval mode, no grad) of v2_adult_s0_final.pt on all test rows
  2. per (purpose, attribute) pair: admit, then fit-attackers --execute-scientific-fits, infer, report
Roles are assigned per RECORD KEY (sha256 of the raw test row incl. labels) so exact duplicate records share a
unit and a role; Adult has no household/linkage id, so the unit is the de-duplicated record (flagged).
Run: /Users/nathansamson/PCRL/.venv/bin/python prepare_pilot_adult_s0.py [--write-inputs]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

WT = Path("/Users/nathansamson/PCRL/.worktrees/combined-evaluation-preparation-v1")
SCR = Path("/private/tmp/claude-501/-Users-nathansamson-PCRL/f1ff337a-0f10-4ee1-bd1d-5817210be5ea/scratchpad")
EXPORT = SCR / "pcrl_b96c412"  # created by forward_smoke.py (git archive b96c412 pcrl)
OUT = Path.home() / "PCRL_eval_cache_private" / "pilot_adult_s0"
CKPT = Path.home() / "PCRL_eval_cache_private" / "checkpoints" / "v2_adult_s0_final.pt"
CKPT_SHA = "1cfc2fefa8929bb4cb9e9680205987e2f3bf3cf37e17a1ba7cf7135cf494c061"
PAIRS = [("income_prediction", 0, "race"), ("income_prediction", 0, "sex"), ("employment_analysis", 1, "race"),
         ("employment_analysis", 1, "age_group"), ("employment_analysis", 1, "marital_status"),
         ("education_assessment", 2, "sex"), ("education_assessment", 2, "race"),
         ("education_assessment", 2, "income")]
# shares and role names from notes/methodology/protocol_config.json (roles: 0.50 / 0.15 / 0.35, "assessment");
# deviation: assignment is a deterministic hash of the record key, NOT stratified by (s, y) as that config asks
SHARES = (("attacker_fit", 0.50), ("attacker_val", 0.15), ("assessment", 0.35))

sys.path.insert(0, str(WT))
from stored_model_eval.admission import sha256_file  # noqa: E402
from stored_model_eval.guards import install_network_guard  # noqa: E402

install_network_guard()


def role_of(key: str) -> str:
    u = int(hashlib.sha256(("pilot-roles-v1|" + key).encode()).hexdigest()[:8], 16) / 2 ** 32
    acc = 0.0
    for name, share in SHARES:
        acc += share
        if u < acc:
            return name
    return SHARES[-1][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-inputs", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, str(EXPORT))
    from pcrl.data.adult import AdultDataset, get_adult_purposes
    root = Path("/Users/nathansamson/PCRL/data")
    for f in ("adult.data", "adult.test"):
        if not (root / "adult" / f).exists():
            raise SystemExit(f"missing {f}: the historical loader would silently synthesise data; refusing")
    purposes = get_adult_purposes()
    train = AdultDataset(purposes=purposes, root=str(root), split="train", download=False)
    test = AdultDataset(purposes=purposes, root=str(root), split="test", download=False, norm_stats=train.norm_stats)
    n = len(test.features)
    rec = np.array([hashlib.sha256("|".join(map(str, r)).encode()).hexdigest()[:20]
                    for r in test.raw_df.itertuples(index=False)])
    _, unit = np.unique(rec, return_inverse=True)
    roles = np.array([role_of(k) for k in rec])
    row_id = np.arange(n, dtype=np.int64)
    attrs = {k: v.numpy().astype(np.int64) for k, v in test.sensitive_attrs.items()}
    summary = {"n_rows": n, "n_record_units": int(unit.max()) + 1, "n_duplicate_rows": int(n - (unit.max() + 1)),
               "rows_per_role": {r: int((roles == r).sum()) for r, _ in SHARES},
               "class_counts_assessment": {k: np.bincount(v[roles == "assessment"]).tolist() for k, v in attrs.items()},
               "pairs": [list(p) for p in PAIRS], "fits_performed": 0}
    if not a.write_inputs:
        summary["dry_run"] = True
        print(json.dumps(summary, indent=1))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "features.npz", row_id=row_id, features=test.features.numpy().astype(np.float32))
    np.savez(OUT / "labels.npz", row_id=row_id, unit=unit, role=roles, record_key=rec, **attrs)
    fman = {"schema": "stored_model_eval.manifest/v1", "synthetic": False, "required_arrays": ["features"],
            "files": {"checkpoint": {"path": str(CKPT), "sha256": CKPT_SHA},
                      "features": {"path": "features.npz", "sha256": sha256_file(OUT / "features.npz")}},
            "arrays": {"features": {"file": "features", "key": "features", "ids": "row_id"}}}
    (OUT / "forward_manifest.json").write_text(json.dumps(fman, indent=1))
    print(json.dumps(summary, indent=1))
    print("# then: forward, and write per-pair manifests with --pair-manifests after the forward cache exists")


def pair_manifests():
    fwd = OUT / "cache" / "adult_s0_test.npz"
    for purpose, p, attr in PAIRS:
        man = {"schema": "stored_model_eval.manifest/v1", "synthetic": False, "min_class_support": 100,
               "allowed_roles": ["defense_fit", "attacker_fit", "attacker_val", "assessment"],
               "files": {"fwd": {"path": str(fwd), "sha256": sha256_file(fwd)},
                         "lab": {"path": str(OUT / "labels.npz"), "sha256": sha256_file(OUT / "labels.npz")}},
               "arrays": {"representations": {"file": "fwd", "key": f"rep_p{p}", "ids": "row_id"},
                          "outputs": {"file": "fwd", "key": f"logits_{purpose}", "ids": "row_id"},
                          "labels": {"file": "lab", "key": attr, "ids": "row_id"},
                          "units": {"file": "lab", "key": "unit", "ids": "row_id"},
                          "roles": {"file": "lab", "key": "role", "ids": "row_id"},
                          "record_keys": {"file": "lab", "key": "record_key", "ids": "row_id"}}}
        (OUT / f"manifest_{purpose}__{attr}.json").write_text(json.dumps(man, indent=1))


NOISE_SEEDS = (0, 1, 2)  # one release per row per seed (historical convention); NOT repeated queries


def noise_manifests():
    """Owner addition 2026-10-02: documented noise arms (untreated + sigma in DOCUMENTED_ADULT_SIGMAS).

    Fit-free: releases = frozen representation + seeded Gaussian noise (stored_model_eval.releases). The task-output
    surface stays the clean model's output (the historical noise channel released the noisy representation only), so
    the rep+output surface for a noise arm combines a noisy representation with clean outputs; report it as such.
    """
    from stored_model_eval.releases import DOCUMENTED_ADULT_SIGMAS, gaussian_release
    fwd = OUT / "cache" / "adult_s0_test.npz"
    if not fwd.exists():
        raise SystemExit("run the frozen forward pass first")
    z = np.load(fwd)
    rel_dir = OUT / "releases"
    rel_dir.mkdir(parents=True, exist_ok=True)
    for purpose, p, attr in PAIRS:
        H = z[f"rep_p{p}"]
        for sigma in DOCUMENTED_ADULT_SIGMAS:
            for seed in NOISE_SEEDS:
                tag = f"p{p}_sigma{sigma:g}_seed{seed}"
                rf = rel_dir / f"{tag}.npz"
                if not rf.exists():
                    np.savez(rf, row_id=z["row_id"], rep=gaussian_release(H, sigma, seed).astype(np.float32))
                man = {"schema": "stored_model_eval.manifest/v1", "synthetic": False, "min_class_support": 100,
                       "release": {"kind": "gaussian_noise", "sigma_abs": sigma, "seed": seed,
                                   "release_count": "one", "outputs_surface": "clean model output"},
                       "allowed_roles": ["defense_fit", "attacker_fit", "attacker_val", "assessment"],
                       "files": {"rel": {"path": str(rf), "sha256": sha256_file(rf)},
                                 "fwd": {"path": str(fwd), "sha256": sha256_file(fwd)},
                                 "lab": {"path": str(OUT / "labels.npz"), "sha256": sha256_file(OUT / "labels.npz")}},
                       "arrays": {"representations": {"file": "rel", "key": "rep", "ids": "row_id"},
                                  "outputs": {"file": "fwd", "key": f"logits_{purpose}", "ids": "row_id"},
                                  "labels": {"file": "lab", "key": attr, "ids": "row_id"},
                                  "units": {"file": "lab", "key": "unit", "ids": "row_id"},
                                  "roles": {"file": "lab", "key": "role", "ids": "row_id"},
                                  "record_keys": {"file": "lab", "key": "record_key", "ids": "row_id"}}}
                (OUT / f"manifest_{purpose}__{attr}__{tag}.json").write_text(json.dumps(man, indent=1))


if __name__ == "__main__":
    if "--pair-manifests" in sys.argv:
        pair_manifests()
    elif "--noise-manifests" in sys.argv:
        noise_manifests()
    else:
        main()
