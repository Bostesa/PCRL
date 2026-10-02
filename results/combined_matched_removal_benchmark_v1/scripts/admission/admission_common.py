"""Shared constants and helpers for the ROLE-1 (artifact and admission) scripts of the matched removal benchmark.

Nothing here fits a model. Private, row-level outputs go only under ~/PCRL_eval_cache_private/ (outside git);
the repo receives counts, hashes and match flags only.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WT = HERE.parents[3]  # worktree root
PKG = WT / "results" / "combined_matched_removal_benchmark_v1"
NOTES = PKG / "notes" / "admission"
PCRL_REPO = Path(os.environ.get("PCRL_REPO", str(Path.home() / "PCRL")))
PCRL_DATA = PCRL_REPO / "data"
B96 = "b96c41256daeed6e644aba1443a47b16e28d089a"
EXPORT = Path(os.environ.get("PCRL_B96_EXPORT", str(Path(tempfile.gettempdir()) / "pcrl_b96c412")))  # scratch export of b96c412
PRIV = Path.home() / "PCRL_eval_cache_private"
CKPT_DIR = PRIV / "checkpoints"
INPUTS = PRIV / "bench_v1" / "inputs"
STAGING = INPUTS / "_staging"
PROV_CKPT = INPUTS / "provenance_ckpt"
PILOT = PRIV / "pilot_adult_s0"
DRIVE = Path(os.environ.get("PCRL_DRIVE", "/Volumes/DRIVE/relocated"))  # external archive drive (set PCRL_DRIVE)
CKPT_TAR = DRIVE / "archives" / "fl-PCRL-main-checkpoints.tar"
INVENTORY = Path.home() / "storage-relocation-20260930" / "inventories" / "fl-PCRL-main-checkpoints.json.gz"
ORIGIN_MAIN = "55e4cb1d1b603a52e5ae16fd5c71d41ebb9b3827"

DATASETS = ("adult", "hmda")
SEEDS = (0, 1, 2)
SIGMAS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
RELEASE_SEEDS = (0, 1, 2)
ROLE_SHARES = (("attacker_fit", 0.50), ("attacker_val", 0.15), ("assessment", 0.35))
ROLE_SALT = {"adult": "pilot-roles-v1|", "hmda": "bench-roles-hmda-v1|"}
FALLBACK_SALT = "bench-defense-fallback-v1|"
SUPPORT_MIN = {"defense_fit": 100, "attacker_fit": 100, "assessment": 100, "attacker_val": 30}

TIER1 = {"adult": {"purpose": "income_prediction", "task": "income", "target": "sex", "policy": ["race", "sex"]},
         "hmda": {"purpose": "underwriting", "task": "loan_decision", "target": "race",
                  "policy": ["race", "ethnicity"]}}
TIER2_E1 = {"adult": [("income_prediction", "race"), ("employment_analysis", "race"),
                      ("employment_analysis", "age_group"), ("employment_analysis", "marital_status"),
                      ("education_assessment", "sex"), ("education_assessment", "race"),
                      ("education_assessment", "income")],
            "hmda": [("underwriting", "ethnicity"), ("pricing_analysis", "race"), ("pricing_analysis", "sex"),
                     ("fair_lending_audit", "race"), ("fair_lending_audit", "sex")]}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha256_array(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    if a.dtype.kind in "US":
        return hashlib.sha256("\n".join(map(str, a.tolist())).encode()).hexdigest()
    return hashlib.sha256(a.tobytes()).hexdigest()


def record_key(row) -> str:
    """Pilot definition (prepare_pilot_adult_s0.py): sha256('|'.join(map(str,row)))[:20] over a raw row."""
    return hashlib.sha256("|".join(map(str, row)).encode()).hexdigest()[:20]


def u01(salted: str) -> float:
    return int(hashlib.sha256(salted.encode()).hexdigest()[:8], 16) / 2 ** 32


def role_of(key: str, salt: str) -> str:
    """Identical mechanics to the pilot's role_of (first 8 hex digits / 2^32 against cumulative shares)."""
    u = u01(salt + key)
    acc = 0.0
    for name, share in ROLE_SHARES:
        acc += share
        if u < acc:
            return name
    return ROLE_SHARES[-1][0]


def ensure_export() -> Path:
    """The b96c412 `pcrl` export; verified byte-identical to `git archive b96c412 pcrl` before use."""
    import tempfile
    import filecmp
    if not (EXPORT / "pcrl" / "data" / "adult.py").exists():
        EXPORT.mkdir(parents=True, exist_ok=True)
        arc = subprocess.run(["git", "-C", str(PCRL_REPO), "archive", B96, "pcrl"], check=True,
                             capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", str(EXPORT)], input=arc, check=True)
    with tempfile.TemporaryDirectory() as td:
        arc = subprocess.run(["git", "-C", str(PCRL_REPO), "archive", B96, "pcrl", "experiments/prepare_hmda.py"],
                             check=True, capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", td], input=arc, check=True)
        bad = []
        for p in Path(td, "pcrl").rglob("*.py"):
            q = EXPORT / p.relative_to(td)
            if not q.exists() or not filecmp.cmp(p, q, shallow=False):
                bad.append(str(p.relative_to(td)))
        if bad:
            raise SystemExit(f"b96c412 export differs from git archive: {bad[:5]}")
        prep = Path(td, "experiments", "prepare_hmda.py").read_bytes()
    exp_dir = STAGING / "b96c412_code" / "experiments"  # private copy; the shared export's pcrl/ stays untouched
    exp_dir.mkdir(parents=True, exist_ok=True)
    tgt = exp_dir / "prepare_hmda.py"
    if not tgt.exists() or tgt.read_bytes() != prep:
        tgt.write_bytes(prep)
    if str(EXPORT) not in sys.path:
        sys.path.insert(0, str(EXPORT))
    if str(exp_dir) not in sys.path:
        sys.path.insert(1, str(exp_dir))
    return EXPORT


def inventory() -> dict:
    return json.load(gzip.open(INVENTORY))


def drive_retry(fn, what: str, tries: int = 12, sleep: float = 5.0):
    last = None
    for i in range(tries):
        try:
            if not DRIVE.exists():
                raise FileNotFoundError(f"drive not mounted: {DRIVE}")
            return fn()
        except (FileNotFoundError, OSError, subprocess.CalledProcessError) as e:  # transient exFAT disappearance
            last = e
            print(f"[retry {i + 1}/{tries}] {what}: {e}", flush=True)
            time.sleep(sleep)
    raise RuntimeError(f"{what} failed after {tries} tries: {last}")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
