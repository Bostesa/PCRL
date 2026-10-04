"""Code/input lock for the refreshed guarded joint study (home-relative private paths only).

    ~/PCRL/.venv/bin/python -m rgj.lock write results/pcrl_refreshed_guarded_joint_v1/CODE_LOCK.json
    ~/PCRL/.venv/bin/python -m rgj.lock amend results/pcrl_refreshed_guarded_joint_v1/CODE_LOCK_A1.json <files...> --reason "..."
    ~/PCRL/.venv/bin/python -m rgj.lock verify results/pcrl_refreshed_guarded_joint_v1/CODE_LOCK.json

CODE_LOCK.json hashes every file that runs before it is pushed. Files written later by the declared owners (LATER) may
only be added through a dated, pushed amendment CODE_LOCK_A<n>.json before the stage that runs them; any other added,
changed or removed file fails verification. Amendments never change a file already locked unless they state a
concrete defect (reason recorded).
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
PKG = WT / "results" / "pcrl_refreshed_guarded_joint_v1"
GLOBS = ["rgj/*.py", "rgj/tests/*.py", "jcv/*.py", "stored_model_eval/defenses.py", "stored_model_eval/bench_infer.py",
         "stored_model_eval/pilot_infer.py", "stored_model_eval/guards.py", "oar/fare_official.py", "oar/study.py",
         "pcrl/data/adult.py"]
LATER = {"rgj/baselines.py": "audit/baseline owner (before stage baselines)",
         "rgj/assess.py": "audit/baseline owner (before EVALUATION_LOCK)",
         "rgj/tests/test_audit.py": "audit/baseline owner",
         "rgj/critic_track.py": "lead (before stage tracking)", "rgj/whiten_diag.py": "lead (before stage whiten)",
         "rgj/infer.py": "lead (before EVALUATION_LOCK)", "rgj/deploy.py": "lead (release packaging)",
         "rgj/eval_lock.py": "lead (before EVALUATION_LOCK)", "rgj/report.py": "lead (reporting only)",
         "rgj/closeout.py": "lead (closeout only)", "rgj/tests/test_late.py": "lead (tests of later modules)",
         "rgj/tests/test_math_review.py": "math/design reviewer (pre-fit fixtures)"}
DOCS = ["PROTOCOL.md", "METHOD_CARD.md", "METHOD_DELTA.md", "FIT_MANIFEST.json", "PRIMARY_FAMILY.json",
        "ROLE_MANIFEST.json", "EXPOSURE_LEDGER.md"]


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


def inputs():
    from rgj import data as DA
    pred = HOME / "PCRL_eval_cache_private" / "jcv_v1" / "run" / "units"
    reused = {}
    for k in (0, 1, 2):
        names = [f"warm__s{k}", f"nn__s{k}__U"] + [f"fare__s{k}__p{i}__{t}{c}" for i in (0, 1) for t in ("c", "Z")
                                                   for c in range(1, 7)]
        for nm in names:
            p = pred / nm / "COMPLETE.json"
            if p.exists():
                reused["~/" + str(p.relative_to(HOME))] = sha_file(p)
    return {"source_npz": "~/" + str(DA.SRC.relative_to(HOME)), "source_npz_sha256": DA.SRC_SHA,
            "role_rule": {"seed": DA.SEED, "head_share": DA.HEAD_SHARE, "critic_split": list(DA.CRITIC_SPLIT)},
            "reused_predecessor_units_COMPLETE_sha256": reused}


def families():
    from rgj import family as F
    return {"primary_ids": [e["id"] for e in F.PRIMARY], "primary_size": F.PRIMARY_SIZE,
            "secondary_ids": [e["id"] for e in F.SECONDARY], "secondary_size": F.SECONDARY_SIZE,
            "z_primary": round(F.Z_PRIMARY, 6), "z_secondary": round(F.Z_SECONDARY, 6), "B": F.B,
            "boot_seed": F.BOOT_SEED, "scored_labels": F.SCORED}


def schedule():
    from rgj import train as T, finalize as FN, run as R
    return {"HP": {k: (list(v) if isinstance(v, tuple) else v) for k, v in T.HP.items()}, "arms": T.ARMS,
            "head_C": FN.HEAD_C, "seeds": list(R.SEEDS), "betas": list(R.BETAS), "stage_B_arms": list(R.STAGE_B_ARMS),
            "stage_C_arms": list(R.STAGE_C_ARMS), "taskline_epochs": R.TASKLINE_EPOCHS,
            "rescue": "one half-learning-rate retry for nonfinite training only (identical for every arm)",
            "optional_ablation": "J-G, capped transform, beta 0.1, seeds 0-2; diagnostic, never a nominee"}


def git(*a):
    return subprocess.run(["git", "-C", str(WT), *a], capture_output=True, text=True).stdout.strip()


def write_lock(path, exclude=()):
    """exclude: declared later-locked files still being written by their owners (locked by a later amendment)."""
    assert all(f in LATER for f in exclude)
    cf = {f: h for f, h in code_files().items() if f not in exclude}
    lock = {"schema": "rgj-code-lock-v1", "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "parent_commit": git("rev-parse", "HEAD"), "code_files": cf, "later_locked": LATER,
            "excluded_in_progress": list(exclude),
            "dependencies": deps(), "inputs": inputs(), "families": families(), "schedule": schedule(),
            "documents_sha256": {d: sha_file(PKG / d) for d in DOCS if (PKG / d).exists()},
            "budget": {"elapsed_h": 10, "cpu_h": 20, "heavy_workers": 2, "memory_gib": 8, "free_disk_gib_min": 5,
                       "reserve_final_h": 2, "cloud": "none ($0)", "start": (HOME / "PCRL_eval_cache_private" /
                                                                           "rgj_v1" / "START.txt").read_text().strip()},
            "statement": "All rows are previously exposed Adult rows; DEVELOPMENT_ASSESSMENT is a new development "
                         "partition, not fresh confirmation. The lock limits outcome-driven changes; it does not restore "
                         "independence."}
    Path(path).write_text(json.dumps(lock, indent=1, sort_keys=True, default=str) + "\n")
    return lock


def amendments():
    return [json.loads(p.read_text()) for p in sorted(PKG.glob("CODE_LOCK_A*.json"))]


def locked_files(lock):
    out = dict(lock["code_files"])
    for a in amendments():
        out.update(a["code_files"])
    return out


def amend(path, files, reason):
    lock = json.loads((PKG / "CODE_LOCK.json").read_text())
    have = locked_files(lock)
    rec = {"schema": "rgj-code-lock-amendment-v1", "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "parent_commit": git("rev-parse", "HEAD"), "reason": reason, "code_files": {},
           "changes_previously_locked": []}
    for f in files:
        rec["code_files"][f] = sha_file(WT / f)
        if f in have and have[f] != rec["code_files"][f]:
            rec["changes_previously_locked"].append(f)
        elif f not in have and f not in LATER:
            raise SystemExit(f"REFUSED: {f} is not a declared later-locked file")
    Path(path).write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec


def verify_lock(path, require=()) -> dict:
    lock = json.loads(Path(path).read_text())
    have = locked_files(lock)
    mm = []
    cf = code_files()
    for f, h in have.items():
        if cf.get(f) != h:
            mm.append(f"locked file changed/removed: {f}")
    for f in cf:
        if f not in have and f not in LATER:
            mm.append(f"unlocked file added: {f}")
    for f in require:
        if f not in have:
            mm.append(f"stage requires a locked {f} (amendment missing)")
    if deps() != lock["dependencies"]:
        mm.append("dependencies changed")
    if inputs() != lock["inputs"]:
        mm.append("inputs changed")
    fam = families()
    for k in ("primary_ids", "secondary_ids", "z_primary", "z_secondary"):
        if fam[k] != lock["families"][k]:
            mm.append(f"family {k} changed")
    return {"ok": not mm, "mismatches": mm}


if __name__ == "__main__":
    cmd, p = sys.argv[1], Path(sys.argv[2])
    if cmd == "write":
        ex = sys.argv[sys.argv.index("--exclude") + 1:] if "--exclude" in sys.argv else []
        L = write_lock(p, ex)
        print("lock written:", len(L["code_files"]), "code files")
    elif cmd == "amend":
        args = sys.argv[3:]
        reason = args[args.index("--reason") + 1]
        files = args[:args.index("--reason")]
        print(json.dumps(amend(p, files, reason), indent=1))
    else:
        print(json.dumps(verify_lock(p), indent=1))
