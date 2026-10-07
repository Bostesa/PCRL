"""Staged code/input locks of the held-out calibration study (hcal; placeholders only, no local paths).

A copy of lra/lock.py at 9762025 with the hcal names. Named locks (results/pcrl_heldout_calibration_v1/<NAME>.json), each
committed AND pushed (remote-verified) before the stages it governs:
  SOURCE_ADMISSION_LOCK  source pins, role split rule, admission code (verified copies, parity, frozen priors) -> admit
  ENGINEERING_LOCK       calibrators, allowlist, common-reader closure, controls, selection, 23 slots, truth table,
                         synthetic correctness checks                                                 -> engineering
  SCIENCE_LOCK           every Adult definition and all scientific code (calibrate, utility, audit, compose,
                         controls, select)                                                            -> science stages
  EVALUATION_LOCK (hcal.eval_lock; checked by hcal.assess) -> the single assessment opening.

    PYTHONPATH=. <python> -m hcal.lock write <NAME> [--protocol protocol.json] [--changes f=reason ...]
    PYTHONPATH=. <python> -m hcal.lock amend <AMENDMENT_An> <files...> --reason "..."
    PYTHONPATH=. <python> -m hcal.lock verify <lock file> [--stage s]

A stage runs only against the LATEST named lock (at least its governing lock) plus dated amendments written after it:
every locked file unchanged; every file a stage REQUIRES locked; every worktree module loaded by the stage process
locked with its hash (check_loaded_modules); the lock and its amendments byte-identical on origin/<study branch>.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

WT = Path(__file__).resolve().parents[1]
HOME = Path.home()
REL = "results/pcrl_heldout_calibration_v1"
PKG = WT / REL
BRANCH = "research/pcrl-heldout-calibration-v1"
SOURCE_SHA = "1dee33253af03bb5efcaa652774cb222df642aa3"   # lra evidence commit (full SHA resolved from 1dee332)
SOURCE_TIP = "976202546a100f2b89a863cb3785c3465cf76c6a"   # lra final tip (worktree base)
LCR_TIP = "091afc2007164fd928d4b792593d6f9eaf75b17c"
CBP_TIP = "7f3ec67b2ecd86d474e2ff27167091af9923f572"
GLOBS = ["hcal/*.py", "tests/pcrl_heldout_calibration_v1/*.py", "lra/*.py", "lra/tests/*.py", "lcr/*.py", "cbp/*.py", "cbp/tests/*.py", "qpc/*.py", "qpc/tests/*.py", "dpc/*.py", "osf/*.py", "smf/*.py", "rgj/*.py", "jcv/*.py",
         "stored_model_eval/*.py", "oar/*.py", "pcrl/data/adult.py", f"{REL}/provenance/*.py"]
ORDER = ["SOURCE_ADMISSION_LOCK", "ENGINEERING_LOCK", "SCIENCE_LOCK"]
STAGE_MIN_LOCK = {"admit": 0, "engineering": 1, "calibrate": 2, "utility": 2, "audit": 2, "compose": 2,
                  "controls": 2, "select": 2, "replay": 2}
STAGE_REQUIRES = {"admit": ["hcal/data.py", "hcal/admit.py", "hcal/run.py", "hcal/ids.py"],
                  "engineering": ["hcal/calib.py", "hcal/bank.py", "hcal/select.py", "hcal/family.py", "hcal/run.py",
                                  "hcal/engineering.py"],
                  "calibrate": ["hcal/calib.py", "hcal/run.py"],
                  "utility": ["hcal/calib.py", "hcal/run.py"],
                  "audit": ["hcal/bank.py", "hcal/run.py"],
                  "compose": ["hcal/bank.py", "hcal/run.py"],
                  "controls": ["hcal/controls.py", "hcal/bank.py", "hcal/run.py"],
                  "select": ["hcal/select.py", "hcal/family.py", "hcal/run.py"],
                  "replay": ["hcal/bank.py", "hcal/run.py"]}
DOCS = ["PROTOCOL.md", "EXPOSURE_LEDGER.md", "ROLE_MANIFEST.json", "SOURCE_INDEX.json", "SOURCE_ADMISSION.json",
        "MODEL_MANIFEST.json", "CALIBRATION_RULES.json", "CONTROL_PLAN.json", "QUEUE_MANIFEST.json", "SELECTION_RULES.json",
        "PRIMARY_FAMILY.json", "LABEL_TRUTH_TABLE.json", "PREDICTIONS.json", "TIMING.json", "ENGINEERING_CHECKS.json",
        "ENGINEERING_RESULT.json", "REVIEW_FINDINGS_DISPOSITION.json", "PROVENANCE_REPORT.md"]
STATEMENT = ("This study is motivated by observed outcomes on repeatedly used Adult development data. Frozen maps, "
             "teachers and earlier decoder outcomes are known. New calibration roles, fitting rules and selection rules "
             "are chosen after those outcomes. All new assessment results are exploratory development evidence, not "
             "fresh confirmation. Their intervals do not account for the full history of research selection. "
             "Assessment masking controls this execution's selection process; it cannot undo previous exposure. No new "
             "population is opened.")
TOP = ("hcal", "lra", "lcr", "cbp", "qpc", "dpc", "osf", "smf", "rgj", "jcv", "stored_model_eval", "oar", "pcrl")


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def code_files():
    return {str(p.relative_to(WT)): sha_file(p) for gl in GLOBS for p in sorted(WT.glob(gl))}


def deps():
    import joblib, numpy, scipy, sklearn, torch
    return {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__, "torch": torch.__version__, "joblib": joblib.__version__,
            "machine": platform.machine(), "torch_threads": 1, "OMP_NUM_THREADS": "1"}


def git(*a):
    return subprocess.run(["git", "-C", str(WT), *a], capture_output=True, text=True).stdout.strip()


def inputs():
    return {"source_npz": "<PRIVATE_CACHE>/jcv_v1/inputs/adult_jcv.npz",
            "source_npz_sha256": "e0d9e54af780f30788ee29cfe6795ec82cbdcadc127b1978c69a3891485d2f12",
            "role_rule": "osf.data via dpc.data via qpc.data via lra.data, pinned unchanged; hcal.data adds the hash-ranked calibration split",
            "source_evidence_commit": SOURCE_SHA,
            "source_tip": SOURCE_TIP, "teacher_provenance_commit": "925e0fddfcb666116c6179575339728a324ed78e",
            "private_store": "<PRIVATE_CACHE>/hcal_v1 (PCRL_HCAL_PRIVATE_CACHE)"}


def amendments():
    return [json.loads(p.read_text()) for p in sorted(PKG.glob("AMENDMENT_A*.json"))]


def latest():
    for n in reversed(ORDER):
        p = PKG / f"{n}.json"
        if p.exists():
            return json.loads(p.read_text())
    raise SystemExit("no lock written")


def locked_files(lock):
    out = dict(lock["code_files"])
    for a in amendments():
        if a["written_at"] >= lock["written_at"]:
            out.update(a["code_files"])
    return out


def start_time():
    from hcal import ids as I
    return (I.PRIV / "START.txt").read_text().strip()


def write_lock(name, protocol=None, changes=None, exclude=()):
    """Lock every present code file except explicitly excluded later files (recorded as unlocked_present)."""
    assert name in ORDER
    prev = None
    if ORDER.index(name) > 0:
        prev = json.loads((PKG / f"{ORDER[ORDER.index(name) - 1]}.json").read_text())
    allf = code_files()
    have = locked_files(prev) if prev is not None else {}
    bad = [f for f in exclude if f in have]
    if bad:
        raise SystemExit(f"REFUSED: cannot exclude already locked files: {bad}")
    cf = {f: h for f, h in allf.items() if f not in set(exclude)}
    changed = sorted(f for f, h in have.items() if f in cf and cf[f] != h)
    removed = sorted(f for f in have if f not in allf)
    missing = [f for f in changed + removed if f not in (changes or {})]
    if missing:
        raise SystemExit(f"REFUSED: previously locked files changed/removed without a stated reason: {missing}")
    lock = {"schema": "hcal-lock-v1", "name": name, "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "parent_commit": git("rev-parse", "HEAD"), "code_files": cf,
            "unlocked_present": {f: h for f, h in allf.items() if f not in cf},
            "changes_previously_locked": {f: (changes or {})[f] for f in changed + removed},
            "dependencies": deps(), "inputs": inputs(),
            "documents_sha256": {d: sha_file(PKG / d) for d in DOCS if (PKG / d).exists()},
            "protocol": json.loads(Path(protocol).read_text()) if protocol else (prev or {}).get("protocol"),
            "budget": {"elapsed_h": 10, "cpu_h": 20, "heavy_processes_total": 2, "memory_gib": 8,
                       "free_disk_gib_min": 5, "reserve_final_h": 2, "reserve_final_cpu_h": 4, "cloud": "none ($0)",
                       "start": start_time()},
            "statement": STATEMENT}
    (PKG / f"{name}.json").write_text(json.dumps(lock, indent=1, sort_keys=True, default=str, allow_nan=False) + "\n")
    return lock


def amend(name, files, reason):
    base = latest()
    have = locked_files(base)
    a = {"schema": "hcal-amendment-v1", "name": name, "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "parent_commit": git("rev-parse", "HEAD"), "base_lock": base["name"], "reason": reason, "code_files": {},
         "changes_previously_locked": [], "new_files": []}
    for f in files:
        a["code_files"][f] = sha_file(WT / f)
        if f in have and have[f] != a["code_files"][f]:
            a["changes_previously_locked"].append(f)
        elif f not in have:
            a["new_files"].append(f)
    (PKG / f"{name}.json").write_text(json.dumps(a, indent=1, sort_keys=True) + "\n")
    return a


def on_origin(rel_path):
    """True iff the committed file at origin/<BRANCH> is byte-identical to the local file."""
    subprocess.run(["git", "-C", str(WT), "fetch", "-q", "origin", BRANCH], capture_output=True)
    r = subprocess.run(["git", "-C", str(WT), "show", f"origin/{BRANCH}:{rel_path}"], capture_output=True)
    return r.returncode == 0 and r.stdout == (WT / rel_path).read_bytes()


def verify_lock(path, stage=None, require_pushed=True) -> dict:
    path = Path(path)
    lock = json.loads(path.read_text())
    lat = latest()
    mm = []
    if lock["name"] != lat["name"]:
        mm.append(f"{lock['name']} is not the latest named lock ({lat['name']})")
    if stage is not None:
        if stage not in STAGE_MIN_LOCK:
            mm.append(f"unknown stage {stage}")
        elif ORDER.index(lock["name"]) < STAGE_MIN_LOCK[stage]:
            mm.append(f"stage {stage} needs {ORDER[STAGE_MIN_LOCK[stage]]} or later")
    have = locked_files(lock)
    cf = code_files()
    mm += [f"locked file changed/removed: {f}" for f, h in have.items() if cf.get(f) != h]
    mm += [f"stage {stage} requires locked {f}" for f in STAGE_REQUIRES.get(stage, []) if f not in have]
    if deps() != lock["dependencies"]:
        mm.append("dependencies changed")
    if inputs() != lock["inputs"]:
        mm.append("inputs changed")
    pushed = None
    if stage is not None and (os.environ.get("CBP_LOCAL_ONLY") or os.environ.get("HCAL_LOCAL_ONLY")):
        mm.append("a local-only flag is set: every stage requires its pushed lock (no local-only bypass)")
    if require_pushed:
        rels = [f"{REL}/{lock['name']}.json"] + [f"{REL}/{a['name']}.json" for a in amendments()
                                                  if a["written_at"] >= lock["written_at"]]
        pushed = {r: on_origin(r) for r in rels}
        mm += [f"not on origin (push before running): {r}" for r, ok in pushed.items() if not ok]
    return {"ok": not mm, "mismatches": mm, "lock": lock["name"], "pushed": pushed,
            "local_only": bool(os.environ.get("CBP_LOCAL_ONLY") or os.environ.get("HCAL_LOCAL_ONLY")), "locked_files": have}


def check_loaded_modules(locked):
    """Every loaded module whose file lives in this worktree under a study package must be locked with that hash."""
    bad = []
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if not f or name.split(".")[0] not in TOP:
            continue
        p = Path(f).resolve()
        try:
            rel = str(p.relative_to(WT))
        except ValueError:
            continue
        if not rel.endswith(".py"):
            continue
        h = locked.get(rel)
        if h is None:
            bad.append(f"loaded but not locked: {rel}")
        elif sha_file(p) != h:
            bad.append(f"loaded with changed hash: {rel}")
    return bad


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "write":
        args = sys.argv[3:]
        prot = args[args.index("--protocol") + 1] if "--protocol" in args else None

        def _list(flag):
            out = []
            if flag in args:
                for x in args[args.index(flag) + 1:]:
                    if x.startswith("--"):
                        break
                    out.append(x)
            return out
        ch = dict(x.split("=", 1) for x in _list("--changes"))
        L = write_lock(sys.argv[2], prot, ch, _list("--exclude"))
        print("lock written:", L["name"], len(L["code_files"]), "code files; unlocked present:",
              sorted(L["unlocked_present"]), "changed:", L["changes_previously_locked"])
    elif cmd == "amend":
        args = sys.argv[3:]
        print(json.dumps(amend(sys.argv[2], args[:args.index("--reason")], args[args.index("--reason") + 1]), indent=1))
    else:
        st = sys.argv[sys.argv.index("--stage") + 1] if "--stage" in sys.argv else None
        v = verify_lock(Path(sys.argv[2]), st)
        v.pop("locked_files")
        print(json.dumps(v, indent=1))
