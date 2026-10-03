"""Pre-fit lock for the output-leak diagnosis study (home-relative private paths)."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

HOME = Path.home()
WT = Path(__file__).resolve().parents[1]
GLOBS = ["odx/*.py", "oar/*.py", "stored_model_eval/*.py", "results/combined_output_diagnosis_v1/scripts/*.py"]


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def code_files():
    return {str(p.relative_to(WT)): sha_file(p) for g in GLOBS for p in sorted(WT.glob(g))}


def deps():
    import joblib, numpy, scipy, sklearn
    return {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__, "joblib": joblib.__version__, "machine": platform.machine()}


def inputs_state():
    B = HOME / "PCRL_eval_cache_private" / "bench_v1" / "inputs"
    out = {"~/PCRL_eval_cache_private/bench_v1/inputs/" + p.name: sha_file(p)
           for p in sorted(B.glob("*.npz")) + [B / "INPUTS_INDEX.json"]}
    U = HOME / "PCRL_eval_cache_private" / "oar_v1" / "run" / "units"
    reuse = {}
    for d in sorted(U.iterdir()):
        if (d / "COMPLETE.json").exists() and any(t in d.name for t in ("__O_full", "__O_prob", "__O_hard", "__REF",
                                                                         "__HEAD__A", "__U2__A")):
            reuse["~/PCRL_eval_cache_private/oar_v1/run/units/" + d.name + "/COMPLETE.json"] = sha_file(d / "COMPLETE.json")
    return {"inputs": out, "reused_units": reuse}


def verify_lock(path: Path) -> dict:
    lock = json.loads(Path(path).read_text())
    mm = []
    cf = code_files()
    for f in sorted(set(cf) | set(lock["code_files"])):
        if cf.get(f) != lock["code_files"].get(f):
            mm.append(f"code changed/added/removed: {f}")
    if deps() != lock["dependencies"]:
        mm.append("dependencies changed")
    st = inputs_state()
    for k in ("inputs", "reused_units"):
        if st[k] != lock["admitted"][k]:
            mm.append(f"admitted {k} changed")
    from odx import family as F
    if [e["id"] for e in F.PRIMARY] != lock["families"]["primary_ids"] or F.PRIMARY_SIZE != lock["families"]["primary_size"]:
        mm.append("primary family changed")
    return {"ok": not mm, "mismatches": mm}


def git_head():
    return subprocess.run(["git", "-C", str(WT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
