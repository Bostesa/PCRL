"""Pre-fit lock for the focused no-erasure penalty study (home-relative private paths only).
    ~/PCRL/.venv/bin/python -m pnx.lock write|verify results/pcrl_penalty_no_erasure_v1/LOCK.json
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

WT = Path(__file__).resolve().parents[1]
HOME = Path.home()
PKG = WT / "results" / "pcrl_penalty_no_erasure_v1"
GLOBS = ["pnx/*.py", "pnx/tests/*.py", "jcv/*.py", "stored_model_eval/defenses.py", "stored_model_eval/bench_infer.py",
         "stored_model_eval/pilot_infer.py", "stored_model_eval/guards.py", "oar/fare_official.py", "oar/study.py",
         "pcrl/data/adult.py"]


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def code_files():
    return {str(p.relative_to(WT)): sha_file(p) for g in GLOBS for p in sorted(WT.glob(g))}


def deps():
    import joblib, numpy, scipy, sklearn, torch, concept_erasure
    return {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__, "torch": torch.__version__, "joblib": joblib.__version__,
            "concept-erasure": getattr(concept_erasure, "__version__", "0.2.4"), "machine": platform.machine(),
            "torch_threads": 1}


def admitted():
    from jcv import data as DA
    pred = HOME / "PCRL_eval_cache_private" / "jcv_v1" / "run" / "units"
    reused = {}
    for k in (0, 1, 2):
        for nm in [f"warm__s{k}", f"nn__s{k}__U", f"nn__s{k}__E"] + [f"nn__s{k}__JP__b{b}" for b in ("0.1", "1", "10")] + \
                [f"fare__s{k}__p{i}__c{c}" for i in (0, 1) for c in range(1, 7)] + [f"fare__s{k}__p{i}__Z1" for i in (0, 1)]:
            for x in (nm, f"inner__{nm}"):
                p = pred / x / "COMPLETE.json"
                if p.exists():
                    reused["~/" + str(p.relative_to(HOME))] = sha_file(p)
        for a in ("F", "F0"):
            p = pred / f"inner__pair__s{k}__{a}" / "COMPLETE.json"
            reused["~/" + str(p.relative_to(HOME))] = sha_file(p)
    return {"raw_files_sha256": DA.RAW_SHA, "label_manifest_sha256": DA.LABELS_SHA,
            "private_inputs_npz": "~/" + str(DA.INPUTS.relative_to(HOME)), "private_inputs_sha256": sha_file(DA.INPUTS),
            "predecessor_DATA_ADMISSION.json_sha256": sha_file(WT / "results/pcrl_joint_complete_view_method_v1/DATA_ADMISSION.json"),
            "reused_predecessor_units": reused}


def families():
    from pnx import family as F
    return {"primary_ids": [e["id"] for e in F.PRIMARY], "primary_size": F.PRIMARY_SIZE,
            "secondary_ids": [e["id"] for e in F.SECONDARY], "secondary_size": F.SECONDARY_SIZE,
            "z_primary": round(F.Z_PRIMARY, 6), "z_secondary": round(F.Z_SECONDARY, 6), "B": F.B, "boot_seed": F.BOOT_SEED,
            "primary": F.PRIMARY, "secondary": F.SECONDARY}


def schedule():
    from jcv import train as T, finalize as FN
    from pnx import run as R
    return {"HP": T.HP, "head_C": FN.HEAD_C, "new_arms": R.NEW_ARMS, "betas": R.BETAS, "seeds": [0, 1, 2],
            "references": ["U", "E", "JP (erased, existing)", "F", "F0 (frozen Z1)"],
            "parity_tolerance": "model bitwise; release <= 1e-12; hard identical; no map",
            "rescue": "one half-learning-rate retry for nonfinite training only"}


def git_head():
    return subprocess.run(["git", "-C", str(WT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def write_lock(path):
    lock = {"schema": "pnx-lock-v1", "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "parent_commit": git_head(), "code_files": code_files(), "dependencies": deps(), "admitted": admitted(),
            "families": families(), "schedule": schedule(),
            "documents_sha256": {d: sha_file(PKG / d) for d in ("PROTOCOL.md", "METHOD_DELTA.md", "UNIT_MANIFEST_PLAN.csv")},
            "predecessor_pin": "568f970683d11373d717ca112b619fbdde40d0b3",
            "budget": {"elapsed_h": 8, "cpu_h": 12, "heavy_workers": 2, "memory_gib": 8, "free_disk_gib_min": 5,
                       "reserve_final_h": 2, "cloud": "none ($0)"},
            "statement": "This method change was motivated by the completed, already-opened Adult development study. All "
                         "new results use previously exposed data and are EXPLORATORY DEVELOPMENT evidence. A pre-fit lock "
                         "improves reproducibility and limits outcome-driven changes; it does not restore fresh confirmation."}
    Path(path).write_text(json.dumps(lock, indent=1, sort_keys=True, default=str) + "\n")
    return lock


def verify_lock(path) -> dict:
    lock = json.loads(Path(path).read_text())
    mm = []
    cf = code_files()
    for f in sorted(set(cf) | set(lock["code_files"])):
        if cf.get(f) != lock["code_files"].get(f):
            mm.append(f"code changed/added/removed: {f}")
    if deps() != lock["dependencies"]:
        mm.append("dependencies changed")
    if admitted() != lock["admitted"]:
        mm.append("admitted inputs changed")
    fam = families()
    for k in ("primary_ids", "secondary_ids", "z_primary", "z_secondary"):
        if fam[k] != lock["families"][k]:
            mm.append(f"family {k} changed")
    return {"ok": not mm, "mismatches": mm}


if __name__ == "__main__":
    cmd, p = sys.argv[1], Path(sys.argv[2])
    if cmd == "write":
        L = write_lock(p)
        print("lock written:", len(L["code_files"]), "code files")
    else:
        v = verify_lock(p)
        print(json.dumps(v, indent=1))
        sys.exit(0 if v["ok"] else 1)
