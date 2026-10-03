"""Execution lock for the output-aware removal study: pins code, dependencies, the official FARE tree, admitted
inputs, LEACE maps, roles, support, the FARE grid, the primary family and runtime ceilings. Private paths are
recorded home-relative."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

HOME = Path.home()
WT = Path(__file__).resolve().parents[1]
BENCH = HOME / "PCRL_eval_cache_private" / "bench_v1"
CODE_GLOBS = ["oar/*.py", "stored_model_eval/*.py", "results/combined_output_aware_removal_v1/scripts/*.py"]


def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def rel(p) -> str:
    s = str(p)
    return "~" + s[len(str(HOME)):] if s.startswith(str(HOME)) else s


def code_files() -> dict:
    out = {}
    for g in CODE_GLOBS:
        for p in sorted(WT.glob(g)):
            out[str(p.relative_to(WT))] = sha_file(p)
    return out


def deps() -> dict:
    import joblib, numpy, scipy, sklearn
    from stored_model_eval.bench_lock import concept_erasure_tree
    return {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__, "joblib": joblib.__version__,
            "concept-erasure": {k: v for k, v in concept_erasure_tree().items() if k != "path"},
            "machine": platform.machine()}


def inputs_state() -> dict:
    files = {}
    for p in sorted((BENCH / "inputs").glob("*.npz")) + [BENCH / "inputs" / "INPUTS_INDEX.json"]:
        files[rel(p)] = sha_file(p)
    maps = {}
    for d in sorted((BENCH / "defenses").iterdir()):
        if d.is_dir() and ("__income_prediction__" in d.name or "__underwriting__" in d.name):
            for f in sorted((d / "map").glob("*")):
                maps[rel(f)] = sha_file(f)
    return {"files": files, "leace_maps": maps,
            "sigma_star_json": {rel(BENCH / "infer" / "SIGMA_STAR.json"): sha_file(BENCH / "infer" / "SIGMA_STAR.json")}}


def verify_lock(lock_path: Path) -> dict:
    lock = json.loads(Path(lock_path).read_text())
    mm = []
    cf = code_files()
    for f in sorted(set(cf) | set(lock["code_files"])):
        if cf.get(f) != lock["code_files"].get(f):
            mm.append(f"code file changed/added/removed: {f}")
    d = deps()
    if d != lock["dependencies"]:
        mm.append(f"dependencies changed: {d} != {lock['dependencies']}")
    st = inputs_state()
    for k in ("files", "leace_maps", "sigma_star_json"):
        if st[k] != lock["inputs"][k]:
            mm.append(f"inputs.{k} changed")
    fare_tree = lock["fare"]["official"].get("tree_sha256")
    if fare_tree:
        from oar.fare_official import official_tree_sha256
        if official_tree_sha256() != fare_tree:
            mm.append("official FARE tree changed")
    from oar.family import FAMILY, FAMILY_SIZE
    if [list(r) for r in FAMILY] != lock["primary_family"]["rows"] or FAMILY_SIZE != lock["primary_family"]["size"]:
        mm.append("primary family changed")
    roles = json.loads((WT / "results/combined_output_aware_removal_v1/ROLES_AND_SUPPORT.json").read_text())
    if hashlib.sha256(json.dumps(roles, sort_keys=True).encode()).hexdigest() != lock["roles_and_support_sha256"]:
        mm.append("roles/support changed")
    return {"ok": not mm, "mismatches": mm}


def git_head() -> str:
    return subprocess.run(["git", "-C", str(WT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
