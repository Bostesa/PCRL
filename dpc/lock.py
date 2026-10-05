"""Staged code/input locks for the decision-preserving compression study (placeholders only; no local paths).

    PYTHONPATH=. ~/PCRL/.venv/bin/python -m dpc.lock write <NAME> [--protocol protocol.json] [--changes f=reason ...]
    PYTHONPATH=. ~/PCRL/.venv/bin/python -m dpc.lock amend <AMENDMENT_An> <files...> --reason "..."
    PYTHONPATH=. ~/PCRL/.venv/bin/python -m dpc.lock verify <lock file> [--stage s]

Named locks (results/pcrl_decision_preserving_compression_v1/<NAME>.json), each committed AND pushed before its stages
run:
  ENGINEERING_LOCK            protocol, exposure statement, roles, source pins, teacher/reference admission, method code
                              (synthetic-checked), budget                             -> admit
  TRAINING_LOCK               bank (full/reduced from SYNTHETIC timing), nominal counts, predictions, every pre-fit
                              repair                                                  -> partition, fit
  SELECTION_AND_AUDIT_LOCK    attackers, utility metrics, gates, selection rules, families, inference, assessment code
                                                                                      -> inner, controls, select
  EVALUATION_LOCK (dpc.eval_lock; checked by dpc.assess) -> the single assessment.
A stage runs only against the LATEST named lock (which must be at least its governing lock) plus dated amendments
(AMENDMENT_A*.json written after it); every locked file must be unchanged; an unlocked file may exist only if it is a
declared later-locked file; a stage that runs a later file requires it to be locked. Verification refuses unless the
lock file and every amendment it relies on are byte-identical on origin/<study branch> (prospective registration).
Later-locked files (LATER) are locked only when named (--include-later) or already locked by the previous lock.
A later named lock that re-hashes a previously locked file with a different hash must name a reason for every such
file (--changes file=reason); it is recorded as "changes_previously_locked" in the lock (no silent re-lock).
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
PKG = WT / "results" / "pcrl_decision_preserving_compression_v1"
REL = "results/pcrl_decision_preserving_compression_v1"
BRANCH = "research/pcrl-decision-preserving-compression-v1"
GLOBS = ["dpc/*.py", "dpc/tests/*.py", "osf/*.py", "smf/*.py", "rgj/*.py", "jcv/*.py", "stored_model_eval/defenses.py",
         "stored_model_eval/bench_infer.py", "stored_model_eval/pilot_infer.py", "stored_model_eval/guards.py",
         "oar/fare_official.py", "oar/study.py", "pcrl/data/adult.py", f"{REL}/provenance/*.py"]
LATER = {f"dpc/{x}.py" for x in ("audit", "utility", "assess", "baselines", "select", "family", "infer", "eval_lock",
                                  "report", "closeout", "controls")} | \
        {f"dpc/tests/{x}.py" for x in ("test_audit", "test_math_review", "test_late", "test_closeout")}
ORDER = ["ENGINEERING_LOCK", "TRAINING_LOCK", "SELECTION_AND_AUDIT_LOCK"]
STAGE_MIN_LOCK = {"admit": 0, "partition": 1, "fit": 1, "inner": 2, "controls": 2, "select": 2}
STAGE_REQUIRES = {"inner": ["dpc/audit.py", "dpc/utility.py", "dpc/baselines.py"],
                  "controls": ["dpc/audit.py", "dpc/controls.py"],
                  "select": ["dpc/select.py", "dpc/audit.py", "dpc/utility.py", "dpc/baselines.py"]}
DOCS = ["PROTOCOL.md", "METHOD_CARD.md", "ROLE_MANIFEST.json", "EXPOSURE_LEDGER.md", "SOURCE_INDEX.json",
        "FIT_MANIFEST.json", "ADMISSION.json", "PRIMARY_FAMILY.json", "PREDICTIONS.json"]


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


def cert_verdict():
    """Review A5: the CERT pool's inclusion is bound to the custody owner's recorded verdict (ROLE_MANIFEST.json)."""
    from osf import data as DA
    src = WT / "results" / "pcrl_online_strength_frontier_v1" / "ROLE_MANIFEST.json"     # pinned source verdict
    c = json.loads(src.read_text())["cert_eligibility"]
    assert c["CERT_ELIGIBLE"] == DA.CERT_ELIGIBLE and (c["verdict"] == "ESTABLISHED") == DA.CERT_ELIGIBLE, \
        "osf.data.CERT_ELIGIBLE disagrees with the custody verdict"
    return c["verdict"]


def inputs():
    from osf import data as DA
    return {"cert_eligibility_verdict": cert_verdict(), "source_npz": "<PRIVATE_CACHE>/jcv_v1/inputs/adult_jcv.npz", "source_npz_sha256": DA.SRC_SHA,
            "role_rule": "osf.data, pinned unchanged (OSF roles; consolidated assessment = four admitted pools)",
            "cert_eligible": DA.CERT_ELIGIBLE, "pinned_source_commit": "925e0fddfcb666116c6179575339728a324ed78e"}


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


def write_lock(name, protocol=None, changes=None, include_later=()):
    assert name in ORDER
    prev = None
    if ORDER.index(name) > 0:
        prev_name = ORDER[ORDER.index(name) - 1]
        prev = json.loads((PKG / f"{prev_name}.json").read_text())
    allf = code_files()
    bad = [f for f in include_later if f not in LATER or f not in allf]
    if bad:
        raise SystemExit(f"REFUSED: not a present later-locked file: {bad}")
    keep = set(include_later) | (set(locked_files(prev)) if prev is not None else set())
    cf = {f: h for f, h in allf.items() if f not in LATER or f in keep}
    changed = []
    if prev is not None:
        have = locked_files(prev)
        changed = sorted(f for f, h in have.items() if f in cf and cf[f] != h)
        missing = [f for f in changed if f not in (changes or {})]
        if missing:
            raise SystemExit(f"REFUSED: previously locked files changed without a stated reason: {missing}")
    lock = {"schema": "dpc-lock-v1", "name": name, "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "parent_commit": git("rev-parse", "HEAD"), "code_files": cf, "later_locked": sorted(LATER),
            "unlocked_later_files_present": {f: h for f, h in allf.items() if f not in cf},
            "changes_previously_locked": {f: (changes or {})[f] for f in changed},
            "dependencies": deps(), "inputs": inputs(),
            "documents_sha256": {d: sha_file(PKG / d) for d in DOCS if (PKG / d).exists()},
            "protocol": json.loads(Path(protocol).read_text()) if protocol else (prev or {}).get("protocol"),
            "budget": {"elapsed_h": 10, "cpu_h": 20, "heavy_workers": 2, "memory_gib": 8, "free_disk_gib_min": 5,
                       "reserve_final_h": 2, "cloud": "none ($0)",
                       "reserve_final_cpu_h": 4,
                       "start": (HOME / "PCRL_eval_cache_private" / "dpc_v1" / "START.txt").read_text().strip()},
            "statement": "The design is motivated by previously opened Adult development results. Every assessment row "
                         "has been used historically. This is an exploratory, locked benchmark comparison. Its nominal "
                         "intervals condition on the fitted artifacts and do not correct for the adaptive research "
                         "history. It is not fresh confirmation or a prospective population guarantee."}
    (PKG / f"{name}.json").write_text(json.dumps(lock, indent=1, sort_keys=True, default=str) + "\n")
    return lock


def amend(name, files, reason):
    base = latest()
    have = locked_files(base)
    a = {"schema": "dpc-amendment-v1", "name": name, "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "parent_commit": git("rev-parse", "HEAD"), "base_lock": base["name"], "reason": reason, "code_files": {},
         "changes_previously_locked": []}
    for f in files:
        a["code_files"][f] = sha_file(WT / f)
        if f in have and have[f] != a["code_files"][f]:
            a["changes_previously_locked"].append(f)
        elif f not in have and f not in LATER and not (WT / f).name.startswith("test_"):
            raise SystemExit(f"REFUSED: {f} is neither locked nor a declared later-locked file")
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
    mm += [f"unlocked file added: {f}" for f in cf if f not in have and f not in LATER]
    mm += [f"stage {stage} requires locked {f}" for f in STAGE_REQUIRES.get(stage, []) if f not in have]
    if deps() != lock["dependencies"]:
        mm.append("dependencies changed")
    if inputs() != lock["inputs"]:
        mm.append("inputs changed")
    pushed = None
    if require_pushed and not os.environ.get("DPC_LOCAL_ONLY"):
        rels = [f"{REL}/{lock['name']}.json"] + [f"{REL}/{a['name']}.json" for a in amendments()
                                                  if a["written_at"] >= lock["written_at"]]
        pushed = {r: on_origin(r) for r in rels}
        mm += [f"not on origin (push before running): {r}" for r, ok in pushed.items() if not ok]
    return {"ok": not mm, "mismatches": mm, "lock": lock["name"], "pushed": pushed,
            "local_only": bool(os.environ.get("DPC_LOCAL_ONLY"))}


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "write":
        args = sys.argv[3:]
        prot = args[args.index("--protocol") + 1] if "--protocol" in args else None
        ch = {}
        if "--changes" in args:
            for x in args[args.index("--changes") + 1:]:
                if x.startswith("--"):
                    break
                f, r = x.split("=", 1)
                ch[f] = r
        inc = []
        if "--include-later" in args:
            for x in args[args.index("--include-later") + 1:]:
                if x.startswith("--"):
                    break
                inc.append(x)
        L = write_lock(sys.argv[2], prot, ch, inc)
        print("lock written:", L["name"], len(L["code_files"]), "code files; changed:", L["changes_previously_locked"])
    elif cmd == "amend":
        args = sys.argv[3:]
        print(json.dumps(amend(sys.argv[2], args[:args.index("--reason")], args[args.index("--reason") + 1]), indent=1))
    else:
        st = sys.argv[sys.argv.index("--stage") + 1] if "--stage" in sys.argv else None
        print(json.dumps(verify_lock(Path(sys.argv[2]), st), indent=1))
