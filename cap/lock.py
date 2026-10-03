"""Pre-fit lock for the useful-head comparison (home-relative private paths only).

    python -m cap.lock write   results/combined_analysis_paper_v1/LOCK.json
    python -m cap.lock verify  results/combined_analysis_paper_v1/LOCK.json
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

HOME = Path.home()
WT = Path(__file__).resolve().parents[1]
GLOBS = ["cap/*.py", "oar/*.py", "odx/*.py", "stored_model_eval/*.py", "results/combined_analysis_paper_v1/scripts/*.py"]
PRIV = HOME / "PCRL_eval_cache_private"
OAR_UNITS = PRIV / "oar_v1" / "run" / "units"
REUSED_SUFFIXES = ("HEAD__A", "HEAD__B", "HEAD__F", "HEAD__FZ", "O_headA", "O_headB", "O_headF", "O_headFZ",
                   "A__rep", "B__rep", "FZ__rep", "A__rep+head", "B__rep+head", "F__rep+head", "FZ__rep+head",
                   "A__rep+clean", "B__rep+clean", "F__rep+clean", "FZ__rep+clean", "FAREFIT_Z",
                   "U2__A", "U2__B", "U2__FZ", "REF")


def rel(p: Path) -> str:
    return "~/" + str(Path(p).relative_to(HOME))


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


def reused_unit_names():
    names = []
    for k in (0, 1, 2):
        sel = json.loads((PRIV / "oar_v1" / "run" / "selection" / f"adult__s{k}.json").read_text())
        j = sel["nominee_unit_source"]
        names += [f"adult__s{k}__{s}" for s in REUSED_SUFFIXES]
        names += [f"adult__s{k}__FAREFIT_c{j}", f"adult__s{k}__Fc{j}__rep", f"adult__s{k}__U2__Fc{j}"]
    return names


def inputs_state():
    B = PRIV / "bench_v1" / "inputs"
    inputs = {rel(p): sha_file(p) for p in sorted(B.glob("adult_s*_forward.npz")) + [B / "INPUTS_INDEX.json"]}
    maps = {}
    for k in (0, 1, 2):
        d = PRIV / "bench_v1" / "defenses" / f"adult__s{k}__income_prediction__B_sex" / "map"
        for p in sorted(d.rglob("*")):
            if p.is_file():
                maps[rel(p)] = sha_file(p)
    sel = {rel(p): sha_file(p) for p in sorted((PRIV / "oar_v1" / "run" / "selection").glob("adult__s*.json"))}
    reuse = {}
    for n in reused_unit_names():
        c = OAR_UNITS / n / "COMPLETE.json"
        reuse[rel(c)] = sha_file(c) if c.exists() else "MISSING"
    return {"inputs": inputs, "leace_maps": maps, "fare_selection": sel, "reused_units": reuse}


def families():
    from cap import family as F
    return {"primary_ids": [e["id"] for e in F.PRIMARY], "primary_size": F.PRIMARY_SIZE,
            "secondary_ids": [e["id"] for e in F.SECONDARY], "secondary_size": F.SECONDARY_SIZE,
            "z_primary": round(F.Z_PRIMARY, 6), "z_secondary": round(F.Z_SECONDARY, 6), "alpha": F.ALPHA,
            "bootstrap_B": F.B_SE, "bootstrap_seed": F.SEED_SE, "primary": F.PRIMARY, "secondary": F.SECONDARY}


def git_head():
    return subprocess.run(["git", "-C", str(WT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def write_lock(path: Path):
    st = inputs_state()
    missing = [k for k, v in st["reused_units"].items() if v == "MISSING"]
    if missing:
        raise SystemExit(f"REFUSED: reused units missing: {missing[:5]}")
    lock = {"schema": "cap-lock-v1", "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "parent_commit": git_head(), "code_files": code_files(), "dependencies": deps(), "admitted": st,
            "families": families(),
            "units_planned": json.loads((WT / "results/combined_analysis_paper_v1/UNITS_PLANNED.json").read_text()),
            "budget": {"elapsed_h": 10, "cpu_h": 12, "heavy_workers": 2, "peak_rss_gib": 6, "omp_threads": 1,
                       "reserve_min_for_verification_build_backup": 90, "free_disk_gib_min": 5},
            "statement": "Reused development assessment rows; this lock does not make them fresh confirmation."}
    Path(path).write_text(json.dumps(lock, indent=1, sort_keys=True) + "\n")
    return lock


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
    for k in st:
        if st[k] != lock["admitted"][k]:
            mm.append(f"admitted {k} changed")
    fam = families()
    for k in ("primary_ids", "primary_size", "secondary_ids", "secondary_size", "z_primary", "z_secondary"):
        if fam[k] != lock["families"][k]:
            mm.append(f"family field {k} changed")
    return {"ok": not mm, "mismatches": mm}


if __name__ == "__main__":
    cmd, p = sys.argv[1], Path(sys.argv[2])
    if cmd == "write":
        L = write_lock(p)
        print("lock written:", len(L["code_files"]), "code files;", len(L["admitted"]["reused_units"]), "reused units")
    else:
        v = verify_lock(p)
        print(json.dumps(v, indent=1))
        sys.exit(0 if v["ok"] else 1)
