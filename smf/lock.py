"""Staged code/input locks for the strength-matched feedback study (placeholders only; no local paths).

    PYTHONPATH=. ~/PCRL/.venv/bin/python -m smf.lock write <NAME> [--exclude files...]
    PYTHONPATH=. ~/PCRL/.venv/bin/python -m smf.lock amend <A-name> <files...> --reason "..."
    PYTHONPATH=. ~/PCRL/.venv/bin/python -m smf.lock verify <lock file> [--stage s]

Named locks (results/pcrl_strength_matched_feedback_v1/<NAME>.json), each pushed before the stages it governs:
  DATA_AND_ENGINEERING_LOCK  roles, architecture, fixtures, ceilings  -> warm, parity, taskline, raw
  PHASE_A_PROTOCOL_LOCK      strength grid, schedules, matching, Phase A selection rule  -> phaseA, inner, selectA
  PHASE_B_PROTOCOL_LOCK      controller (after preflight), Phase B arms, nomination/C* rules, families  -> phaseB, ...
A stage runs only against the latest named lock plus dated amendments (<A-name>.json, e.g. AMENDMENT_A1.json); every
locked file must be unchanged; an unlocked file may exist only if it is a declared later-locked file, and a stage that
runs it requires it to be locked.
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
PKG = WT / "results" / "pcrl_strength_matched_feedback_v1"
GLOBS = ["smf/*.py", "smf/tests/*.py", "rgj/*.py", "jcv/*.py", "stored_model_eval/defenses.py",
         "stored_model_eval/bench_infer.py", "stored_model_eval/pilot_infer.py", "stored_model_eval/guards.py",
         "oar/fare_official.py", "oar/study.py", "pcrl/data/adult.py"]
LATER = {f"smf/{x}.py" for x in ("audit", "assess", "baselines", "select", "preflight", "track", "family", "infer",
                                  "eval_lock", "deploy", "report", "closeout")} | \
        {f"smf/tests/{x}.py" for x in ("test_audit", "test_math_review", "test_late", "test_select")}
STAGE_REQUIRES = {"inner": ["smf/audit.py"], "selectA": ["smf/select.py", "smf/audit.py"],
                  "preflight": ["smf/preflight.py"], "phaseB": ["smf/select.py"], "baselines": ["smf/baselines.py"],
                  "selectB": ["smf/select.py", "smf/baselines.py"], "tracking": ["smf/track.py"]}
DOCS = ["PROTOCOL.md", "METHOD_CARD.md", "METHOD_DELTA.md", "ROLE_MANIFEST.json", "EXPOSURE_LEDGER.md",
        "FIT_MANIFEST.json", "PRIMARY_FAMILY.json"]
ORDER = ["DATA_AND_ENGINEERING_LOCK", "PHASE_A_PROTOCOL_LOCK", "PHASE_B_PROTOCOL_LOCK"]


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def code_files():
    return {str(p.relative_to(WT)): sha_file(p) for g in GLOBS for p in sorted(WT.glob(g))}


def deps():
    import joblib, numpy, scipy, sklearn, torch
    return {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__, "torch": torch.__version__, "joblib": joblib.__version__,
            "machine": platform.machine(), "torch_threads": 1, "OMP_NUM_THREADS": "1"}


def git(*a):
    return subprocess.run(["git", "-C", str(WT), *a], capture_output=True, text=True).stdout.strip()


def inputs():
    from smf import data as DA
    return {"source_npz": "<PRIVATE_CACHE>/jcv_v1/inputs/adult_jcv.npz", "source_npz_sha256": DA.SRC_SHA,
            "admission_numeric_norm_sha256": sha_file(DA.ADMISSION),
            "role_rule": {"seed": DA.SEED, "assess_share": DA.ASSESS_SHARE, "critic_split": list(DA.CRITIC_SPLIT)}}


def write_lock(name, exclude=()):
    assert all(f in LATER for f in exclude), [f for f in exclude if f not in LATER]
    from smf import train as T
    lock = {"schema": "smf-lock-v1", "name": name, "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "parent_commit": git("rev-parse", "HEAD"),
            "code_files": {f: h for f, h in code_files().items() if f not in exclude},
            "excluded_in_progress": list(exclude), "later_locked": sorted(LATER), "dependencies": deps(),
            "inputs": inputs(), "HP": {k: (list(v) if isinstance(v, tuple) else v) for k, v in T.HP.items()},
            "documents_sha256": {d: sha_file(PKG / d) for d in DOCS if (PKG / d).exists()},
            "budget": {"elapsed_h": 10, "cpu_h": 20, "heavy_workers": 2, "memory_gib": 8, "free_disk_gib_min": 5,
                       "reserve_final_h": 2, "cloud": "none ($0)",
                       "start": (HOME / "PCRL_eval_cache_private" / "smf_v1" / "START.txt").read_text().strip()},
            "statement": "All rows are previously exposed Adult rows; NEW_DEVELOPMENT_ASSESSMENT is withheld from this "
                         "procedure until EVALUATION_LOCK.json, not fresh confirmation."}
    p = PKG / f"{name}.json"
    p.write_text(json.dumps(lock, indent=1, sort_keys=True, default=str) + "\n")
    return lock


def amendments():
    return [json.loads(p.read_text()) for p in sorted(PKG.glob("AMENDMENT_A*.json"))]


def locked_files(lock):
    out = dict(lock["code_files"])
    for a in amendments():
        if a["written_at"] >= lock["written_at"]:
            out.update(a["code_files"])
    return out


def amend(name, files, reason):
    base = latest()
    have = locked_files(base)
    rec = {"schema": "smf-amendment-v1", "name": name, "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "parent_commit": git("rev-parse", "HEAD"), "base_lock": base["name"], "reason": reason, "code_files": {},
           "changes_previously_locked": []}
    for f in files:
        rec["code_files"][f] = sha_file(WT / f)
        if f in have and have[f] != rec["code_files"][f]:
            rec["changes_previously_locked"].append(f)
        elif f not in have and f not in LATER:
            raise SystemExit(f"REFUSED: {f} is neither locked nor a declared later-locked file")
    (PKG / f"{name}.json").write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec


def latest():
    for n in reversed(ORDER):
        p = PKG / f"{n}.json"
        if p.exists():
            return json.loads(p.read_text())
    raise SystemExit("no lock written")


def verify_lock(path, stage=None) -> dict:
    lock = json.loads(Path(path).read_text())
    if lock["name"] != latest()["name"]:
        return {"ok": False, "mismatches": [f"{lock['name']} is not the latest named lock ({latest()['name']})"]}
    have = locked_files(lock)
    cf = code_files()
    mm = [f"locked file changed/removed: {f}" for f, h in have.items() if cf.get(f) != h]
    mm += [f"unlocked file added: {f}" for f in cf if f not in have and f not in LATER]
    mm += [f"stage {stage} requires locked {f}" for f in STAGE_REQUIRES.get(stage, []) if f not in have]
    if deps() != lock["dependencies"]:
        mm.append("dependencies changed")
    if inputs() != lock["inputs"]:
        mm.append("inputs changed")
    return {"ok": not mm, "mismatches": mm}


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "write":
        ex = sys.argv[sys.argv.index("--exclude") + 1:] if "--exclude" in sys.argv else []
        L = write_lock(sys.argv[2], ex)
        print("lock written:", L["name"], len(L["code_files"]), "code files")
    elif cmd == "amend":
        args = sys.argv[3:]
        print(json.dumps(amend(sys.argv[2], args[:args.index("--reason")], args[args.index("--reason") + 1]), indent=1))
    else:
        st = sys.argv[sys.argv.index("--stage") + 1] if "--stage" in sys.argv else None
        print(json.dumps(verify_lock(Path(sys.argv[2]), st), indent=1))
