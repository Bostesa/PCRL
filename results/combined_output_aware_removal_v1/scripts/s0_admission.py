"""Stage 0: admit benchmark inputs by hash and inventory record overlap across all roles (no fits)."""
import hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np

HOME = Path.home()
BENCH = HOME / "PCRL_eval_cache_private" / "bench_v1"
PKG = Path(__file__).resolve().parents[1]
WT = PKG.parents[1]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def rel(p):
    s = str(p)
    return "~" + s[len(str(HOME)):] if s.startswith(str(HOME)) else s


def git(*a):
    return subprocess.run(["git", "-C", str(WT), *a], capture_output=True, text=True).stdout.strip()


idx = json.loads((BENCH / "inputs" / "INPUTS_INDEX.json").read_text())
out = {"schema": "oar_input_admission/v1", "stage": 0, "fits_performed": 0,
       "source_pins": {
           "base_branch": "research/combined-matched-removal-benchmark-v1",
           "base_commit": git("rev-parse", "70f978f"),
           "benchmark_lock_commit": git("rev-parse", "3274fe1"),
           "pilot_commit": git("rev-parse", "661db8d"),
           "reconciliation_commit": "07a9ca3ffaf27133b6cf955b3b8d526682961e8d",
           "evaluation_preparation_commit": "031860fbfd910b9aaed19b7600d288d15858dbc5",
           "empirical_preparation_commit": git("rev-parse", "f381bc2"),
           "origin_main": git("rev-parse", "origin/main"),
           "durable_guarantees_remote": "https://github.com/Bostesa/durable-guarantees.git (from BOTH_REPOS_INDEX.json)"},
       "private_root": "~/PCRL_eval_cache_private/bench_v1 (admitted in place, read-only)",
       "inputs_index_sha256": sha(BENCH / "inputs" / "INPUTS_INDEX.json"),
       "files": {}, "datasets": {}}
# every file referenced by the index, hash-checked
mism = []
for ds, d in idx["datasets"].items():
    refs = [(d["labels_npz"], d["labels_sha256"]), (d["roles_npz"], d["roles_sha256"])]
    for s, info in d.get("encoders", {}).items():
        if info.get("forward_npz"):
            refs.append((info["forward_npz"], info["forward_sha256"]))
        if info.get("checkpoint"):
            refs.append((info["checkpoint"], info["checkpoint_sha256"]))
    for f in ("features_npz", "rows_npz"):
        if d.get(f):
            refs.append((d[f], d[f.replace("_npz", "_sha256")]))
    for p, h in refs:
        got = sha(p)
        out["files"][rel(p)] = {"expected": h, "actual": got, "ok": got == h}
        if got != h:
            mism.append(rel(p))
out["hash_mismatches"] = mism
# MAP pins + saved predictions are covered by the benchmark backup SHA256SUMS; check the units/defenses we will reuse
sums = {}
import os
DRIVE_SUMS = Path(os.environ.get("PCRL_DRIVE", "/Volumes/DRIVE/relocated")) / "private_bench_v1_20261003" / "SHA256SUMS"
for line in (DRIVE_SUMS.read_text().splitlines() if DRIVE_SUMS.exists() else []):
    h, r = line.split("  ", 1)
    sums[r] = h
reuse = [r for r in sums if r.startswith(("bench_v1/defenses/", "bench_v1/units/", "bench_v1/shared/", "bench_v1/infer/SIGMA_STAR.json"))]
bad = [r for r in reuse if sha(HOME / "PCRL_eval_cache_private" / r) != sums[r]]
out["benchmark_outputs_vs_drive_sha256sums"] = {"checked": len(reuse), "mismatches": bad,
                                                 "drive_index": "<drive>/private_bench_v1_20261003/SHA256SUMS"}
for ds in ("adult", "hmda"):
    L = np.load(BENCH / "inputs" / f"{ds}_labels.npz")
    role, split, key = L["role"], L["split"], L["canon_key"]
    train_keys = set(key[split == "train"])
    rows = {}
    for r in ("defense_fit", "attacker_fit", "attacker_val", "assessment", "excluded_dup"):
        m = role == r
        rk = key[m]
        rows[r] = {"rows": int(m.sum()), "units": int(len(np.unique(L["unit"][m]))),
                   "rows_whose_record_occurs_in_encoder_train_split": int(np.isin(rk, list(train_keys)).sum()) if r != "defense_fit" else None}
    # pairwise record-key overlap between scored roles and defense_fit
    pair = {}
    names = ["defense_fit", "attacker_fit", "attacker_val", "assessment"]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ka, kb = set(key[role == a]), set(key[role == b])
            pair[f"{a}|{b}"] = len(ka & kb)
    test_overlap = (split == "test") & np.isin(key, list(train_keys))
    out["datasets"][ds] = {
        "role_counts": rows, "pairwise_shared_record_keys": pair,
        "test_rows_with_record_in_encoder_train": int(test_overlap.sum()),
        "by_role": {r: int((test_overlap & (role == r)).sum()) for r in ("attacker_fit", "attacker_val", "assessment")},
        "test_units_with_record_in_encoder_train": int(len(np.unique(L["unit"][test_overlap]))),
        "encoder_train_split_rows": int((split == "train").sum()),
        "identity_status": ("record-equality only (canon_key: Adult raw row hash with adult.test trailing '.' removed; "
                            "HMDA hash over 17 kept LAR columns). Equal records are repeated records, NOT proven same people; "
                            "HMDA has no applicant identifier."),
        "seed_dependence": "the data split is one seeded permutation shared by all encoder seeds (seeds differ only in training randomness); overlap is identical for seeds 0, 1, 2",
        "limitations": ("the encoder's validation split rows are not in the admitted arrays, so overlap of test roles with the "
                        "validation split is not measured; Round-4 uses final.pt (epoch 204), and the validation split did not train weights"),
    }
json.dump(out, open(PKG / "INPUT_ADMISSION.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("hash_mismatches", "benchmark_outputs_vs_drive_sha256sums")}, indent=1)[:600])
for ds in ("adult", "hmda"):
    print(ds, out["datasets"][ds]["by_role"], out["datasets"][ds]["test_units_with_record_in_encoder_train"], out["datasets"][ds]["pairwise_shared_record_keys"])
