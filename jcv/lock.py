"""Pre-fit lock for the joint complete-view study (home-relative private paths only).
    ~/PCRL/.venv/bin/python -m jcv.lock write|verify results/pcrl_joint_complete_view_method_v1/LOCK.json
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
PKG = WT / "results" / "pcrl_joint_complete_view_method_v1"
GLOBS = ["jcv/*.py", "jcv/tests/*.py", "stored_model_eval/defenses.py", "stored_model_eval/bench_infer.py",
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
    return {"raw_files_sha256": DA.RAW_SHA, "label_manifest_sha256": DA.LABELS_SHA,
            "private_inputs_npz": "~/" + str(DA.INPUTS.relative_to(HOME)), "private_inputs_sha256": sha_file(DA.INPUTS),
            "DATA_ADMISSION.json_sha256": sha_file(PKG / "DATA_ADMISSION.json")}


def families():
    from jcv import family as F
    return {"primary_ids": [e["id"] for e in F.PRIMARY], "primary_size": F.PRIMARY_SIZE,
            "secondary_ids": [e["id"] for e in F.SECONDARY], "secondary_size": F.SECONDARY_SIZE,
            "z_primary": round(F.Z_PRIMARY, 6), "z_secondary": round(F.Z_SECONDARY, 6), "B": F.B, "boot_seed": F.BOOT_SEED,
            "primary": F.PRIMARY, "secondary": F.SECONDARY}


def schedule():
    from jcv import train as T, run as R, finalize as FN
    return {"HP": T.HP, "fare_grid": R.FARE_GRID, "fare_seed_base": R.FARE_SEED_BASE, "head_C": FN.HEAD_C,
            "arms": ["U", "E", "L", "J", "JP", "S12", "S21", "F", "F0"], "seeds": [0, 1, 2]}


def git_head():
    return subprocess.run(["git", "-C", str(WT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def write_lock(path):
    lock = {"schema": "jcv-lock-v1", "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "parent_commit": git_head(), "code_files": code_files(), "dependencies": deps(), "admitted": admitted(),
            "families": families(), "schedule": schedule(),
            "documents_sha256": {d: sha_file(PKG / d) for d in ("PROTOCOL.md", "METHOD_CARD.md", "PRIOR_WORK_DELTA.md",
                                                               "THEORY_AND_LIMITS.md", "PERMISSIONS_AND_VIEWS.md",
                                                               "UNIT_MANIFEST.csv")},
            "budget": {"elapsed_h": 12, "cpu_h": 20, "heavy_workers": 2, "memory_gib": 8, "free_disk_gib_min": 5,
                       "reserve_final_h": 2, "cloud": "none planned (local compute sufficient)"},
            "statement": "Reused development data; no fresh confirmation; every outer result is development evidence."}
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
