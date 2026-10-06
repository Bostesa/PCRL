#!/usr/bin/env python3
"""Independent verifier for the confidence-capacity study (qpc).

Owner: role F (independent verifier and scientific report review; prompt section 15). Exclusive files: this script and
results/pcrl_confidence_capacity_v1/INDEPENDENT_VERIFICATION.json.

Provenance of the code. Adapted in STRUCTURE from the predecessor's independent verifier
(results/pcrl_decision_preserving_compression_v1/verification/replay_dpc.py, itself independent code), as authorised by
the lead: the import guard, the data / role reconstruction, the forward application and the lock-chronology helpers.
Everything else (KL k-means and assignment replay, policies, metrics, attacker refits, selection, inference) is written
here from the registered definitions: the study prompt, PROTOCOL.md, METHOD_CARD.md (role B), the dpc fixed rules
(dpc/partition.py and dpc/release.py docstrings, READ for formats and rules only) and the named locks.

Independence:
  * a sys.meta_path guard refuses every import under qpc, dpc, osf, smf, rgj, jcv, pnx, oar, stored_model_eval and
    pcrl, including imports triggered while unpickling joblib heads; torch.load always uses weights_only=True; the run
    asserts at the end that none of these packages was loaded;
  * the worktree root is removed from sys.path; the shared semaphore is invoked BY PATH as the parent process
    (OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python <WORKTREE>/qpc/sema.py --label F:<what> -- <this command>), never
    imported;
  * libraries: numpy, scipy, scikit-learn, torch, joblib and the standard library.

Phases (status PASS / FAIL / WARN / INFO / PENDING):
  PHASE_0  scaffold + synthetic self-tests only (no real data is read)
  PHASE_1  roles, teachers (own forward pass), A1 historical reproduction + 200-round run, A2 restart winners,
           assignments, decoded vectors, class preservation, inner utilities, capacity gate and rate decision,
           chronology (locks pushed before stages, predictions before Stage A)
  PHASE_2  Stage B fine partitions and search receipts, inner audits (selected attackers, composition), selection,
           nominees, controls
  PHASE_3  outer units, the 37 endpoints, capacity curve, deployment / restore parity, decision documents

Usage (from the worktree root; every real-data phase under the semaphore):
    OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python <WORKTREE>/qpc/sema.py --label F:phase1 -- \\
        ~/PCRL/.venv/bin/python results/pcrl_confidence_capacity_v1/verification/replay_qpc.py --phase 1
    ~/PCRL/.venv/bin/python results/pcrl_confidence_capacity_v1/verification/replay_qpc.py --phase 0   (self-tests)
Writes results/pcrl_confidence_capacity_v1/INDEPENDENT_VERIFICATION.json (aggregates, hashes and placeholders only)
unless --no-write.
"""
from __future__ import annotations

import importlib
import importlib.abc
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

# ------------------------------------------------------------------------------------------------ import guard
_FORBIDDEN_TOP = ("qpc", "dpc", "osf", "smf", "rgj", "jcv", "pnx", "oar", "stored_model_eval", "pcrl")
_PRELOADED = sorted(m for m in sys.modules if m.split(".")[0] in _FORBIDDEN_TOP)
_BLOCKED: list = []
_SELFTEST_BLOCKED: list = []
_IN_SELFTEST = [False]


class _ImportGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in _FORBIDDEN_TOP:
            (_SELFTEST_BLOCKED if _IN_SELFTEST[0] else _BLOCKED).append(fullname)
            raise ImportError(f"independent verifier refuses to import {fullname}")
        return None


sys.meta_path.insert(0, _ImportGuard())
_WT_GUESS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path[:] = [p for p in sys.path if p and os.path.abspath(p) != _WT_GUESS]

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import pickle  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from fractions import Fraction  # noqa: E402
from pathlib import Path  # noqa: E402
from statistics import NormalDist  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as TF  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import log_loss as sk_log_loss  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.pipeline import Pipeline, make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

torch.set_num_threads(1)

# ------------------------------------------------------------------------------------------------ constants
HERE = Path(__file__).resolve().parent
RES = HERE.parent
WT = RES.parents[1]
REL_RES = "results/pcrl_confidence_capacity_v1"
OUT = RES / "INDEPENDENT_VERIFICATION.json"
CACHE = Path.home() / "PCRL_eval_cache_private"
SRC = CACHE / "jcv_v1" / "inputs" / "adult_jcv.npz"
SRC_SHA = "e0d9e54af780f30788ee29cfe6795ec82cbdcadc127b1978c69a3891485d2f12"
PRIV = CACHE / "qpc_v1"
RUN = PRIV / "run"
UNITS = RUN / "units"
DPC_PRIV = CACHE / "dpc_v1"
DPC_UNITS = DPC_PRIV / "run" / "units"
DPC_ADMITTED = DPC_PRIV / "admitted"
ADMISSION_NORM = WT / "results" / "pcrl_joint_complete_view_method_v1" / "DATA_ADMISSION.json"
BRANCH = "research/pcrl-confidence-capacity-v1"
SOURCE_SHA = "0a7b05a52746544213742f50efd0a48167efffb1"          # dpc evidence
TEACHER_PIN = "925e0fddfcb666116c6179575339728a324ed78e"         # osf teacher provenance
DPC_REL = "results/pcrl_decision_preserving_compression_v1"
OSF_REL = "results/pcrl_online_strength_frontier_v1"

SEEDS = (0, 1, 2)
TEACHERS = ("U", "RAW-J_b0.3")
TEACHER_LABEL = {"U": "U", "RAW-J_b0.3": "RAW-J|b0.3"}
REFS = ("E", "F", "F0")
ROLES = ("OSF_DEFENSE_FIT", "OSF_DEVELOPMENT_ASSESSMENT", "HEAD_VALIDATION", "AUDIT_FIT", "INNER_SELECTION")
SUBROLES = ("CRITIC_FIT", "CRITIC_VAL", "DIAGNOSTIC_CALIB")
FIT_ROLES = ("OSF_DEFENSE_FIT", "HEAD_VALIDATION", "AUDIT_FIT", "INNER_SELECTION")
POOLS = ("ORIG_ASSESSMENT", "RGJ_DEV", "SMF_DEV", "CERT")
EXCLUSIONS = ("excluded_exposure", "excluded_dup")
FIT, ASSESS, INNER = "OSF_DEFENSE_FIT", "OSF_DEVELOPMENT_ASSESSMENT", "INNER_SELECTION"
NUMERIC = ("age", "education-num", "capital-gain", "capital-loss", "hours-per-week")
LABEL_KEYS = {"sex": "sex", "race": "race", "y_income": "y_income", "y_occ": "y_occupation_group"}
KS = (2, 6)
HEAD_C = (0.01, 0.1, 1.0, 10.0, 100.0)
EXPECTED_COUNTS = {"OSF_DEFENSE_FIT": (15434, 15428), "HEAD_VALIDATION": (1500, 1499), "AUDIT_FIT": (6065, 6061),
                   "INNER_SELECTION": (2235, 2234), ASSESS: (13936, 13929)}
ROWS_KEPT = 39170

# fixed release / method rules (prompt section 6; dpc partition + release rules; qpc METHOD_CARD)
EPS = 1e-12                  # smoothing
LL_CLIP = 1e-12              # log-loss clip
SUM_TOL = 1e-12              # prototype normalisation
KL_ROUNDOFF = 1e-12          # |negative KL| below this is roundoff (k-means++ weights); larger negatives are errors
SPARSE_N = 5
A1_ROUNDS = 20               # historical source rule (dpc.partition ROUNDS)
A2_ROUNDS = 200              # registered cap
PATIENCE = 3                 # relative-tolerance rounds in a row
RTOL = 1e-9                  # registered relative-objective tolerance (kmeans rule text / METHOD_CARD section 3)
KPP_SEEDS = (20261006, 20261007)   # A2 starts 2 and 3
START_ORDER = ("source", f"kpp:{KPP_SEEDS[0]}", f"kpp:{KPP_SEEDS[1]}")
RATES_INCOME = (4, 8)
RATES_OCC = (8, 16, 32, 64)
A1_RATE = (8, 8)
FINE_CAPS = (32, 128)        # Stage B fine partitions (income, occupation) per predicted class
LAMS = (0.01, 0.1, 1.0)
PRIV_FAMS = ("LOCAL", "SEQ-12", "SEQ-21", "JOINT")
SWEEPS = 5

# utility contract (prompt section 7, A3) and preferred headroom (rate selection only)
GATE = {"acc_drop": 0.01, "ll_excess": 0.01, "brier_excess": 0.005, "retain": 0.8, "gain": 0.03}
HEADROOM = {"ll_excess": 0.0075, "brier_excess": 0.0035}
MAX_SELECTED_RATES = 2

# inference (prompt section 12)
N_ENDPOINTS = 37
Z_PRIMARY = 3.2048452050105634
B_BOOT, BOOT_SEED = 1999, 20261007
MARGIN_PAIR, MARGIN_LOCAL, BUFFER = 0.02, 0.01, 0.005

STATUS_RANK = {"FAIL": 4, "PENDING": 3, "WARN": 2, "PASS": 1, "INFO": 0, "NOT_APPLICABLE": 0}

# own full-row releases of every fitted code: OWN_REL[(seed, cid)] = {tok1, q1, hard1, tok2, q2, hard2, alpha1, alpha2}
OWN_REL: dict = {}
# every correction made to this verifier's own code (kept; disclosed in the report)
VERIFIER_CORRECTIONS: list = [
    {"at": "2026-10-06T04:20Z", "where": "init_kmeanspp / lloyd_qpc / fit_class",
     "what": "first written provisionally (row-level k-means++, generic tolerance) before the registered rule text was "
             "available; rewritten from the registered rule text (qpc/kmeans.py docstring = METHOD_CARD section 3) before "
             "any real-data comparison", "effect": "none on any report (no real Stage A data had been read)"},
    {"at": "2026-10-06T06:55Z", "where": "check_refits (SRC|U composed winners)",
     "what": "the inner reference of a composed winner was taken from the SRC|U own-bank selection arrays; it is the "
             "winning code's inner unit (the assessment predictions already matched bitwise)",
     "effect": "6 spurious refit mismatches in a PHASE_3 development run (not published)"},
    {"at": "2026-10-06T06:35Z", "where": "check_inner (composed SRC|U)",
     "what": "the composed record lists policies by configuration ID, not unit name, so the first composition resolved "
             "no code; every SRC|U family is composed and is now compared after composition",
     "effect": "PHASE_3 development runs only (not published)"},
    {"at": "2026-10-06T05:18Z", "where": "check_chronology / mutation_power_stage_a",
     "what": "the lead's 'inner_src' stage was missing from the stage -> lock map (it is governed by "
             "AUDIT_AND_SELECTION_LOCK, pushed 04:51:09Z before it started at 05:12:33Z); an acquisition without a "
             "release record (C:mutation, slot 1) was counted as open forever and inflated the concurrency peak to 3: "
             "flock exclusivity means its wrapper had exited before slot 1 was re-acquired, so it is now closed there and "
             "reported separately as a WARN; the trajectory mutation is now 1e-8 relative (above the new 1e-10 receipt "
             "tolerance)", "effect": "second PHASE_2A development run (not published)"},
    {"at": "2026-10-06T05:13Z", "where": "receipts_ok (Stage B fine partitions)",
     "what": "per-class objectives J = A_c + sum_j h_j were compared at 1e-12 relative; at 128 cells per class the "
             "lead's and the verifier's summation orders differ by 1.3e-12 to 3.5e-12 relative (cancellation of large "
             "terms), with bitwise-identical partitions, assignments, start fingerprints, stop reasons, passes and init "
             "draws; the objective tolerance is now 1e-10 relative (the bitwise checks are unchanged)",
     "effect": "fine_partitions FAIL -> PASS in the first PHASE_2A development run (scratch output, not published)"},
    {"at": "2026-10-06T05:13Z", "where": "check_stage_b sparsity / permutation alphabet",
     "what": "the alphabet was taken as max(emitted token)+1; the class-5 fallback token is never emitted (U never "
             "predicts occupation class 5), so r2 and pair alphabets were one token / 16 pairs short; now the policy "
             "alphabet", "effect": "42 sparsity-receipt WARNs in the first PHASE_2A development run (not published)"},
    {"at": "2026-10-06T04:22Z", "where": "compare_start_receipts",
     "what": "the A1 per-class summaries carry no 'init' block; init comparison now applies only where the receipt "
             "records it (every A2 start)", "effect": "development run only; no published report"}]
# the verifier's own process ledger (single-threaded processes)
VERIFIER_RUNS: list = [
    {"run": "synthetic self-tests during scaffold development (about 12 runs)", "end": "2026-10-06T04:2xZ",
     "wall_s": 6.0, "cpu_s": 6.0, "heavy": False},
    {"run": "first semaphore attempt by path (stdlib select shadowed by qpc/select.py; failed before acquiring a slot)",
     "end": "2026-10-06T04:16:06Z", "wall_s": 0.1, "cpu_s": 0.1, "heavy": False}]


def g(x) -> str:
    return f"{x:g}"


def direct_id(m1, m2):
    return f"U|DIRECT-TASK|i{m1}o{m2}"


def rate_bank():
    return [(m1, m2) for m1 in RATES_INCOME for m2 in RATES_OCC]


# ------------------------------------------------------------------------------------------------ helpers
def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha_bytes(*arrs) -> str:
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def jload(p):
    return json.loads(Path(p).read_text())


def rowid_hash(r) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.sort(np.asarray(r)).astype(np.int64)).tobytes()).hexdigest()


def unit_set_hash(u) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.unique(np.asarray(u)).astype(np.int64)).tobytes()).hexdigest()


def res(status, **kw):
    return {"status": status, **kw}


def worst(*statuses):
    s = [x for x in statuses if x]
    return max(s, key=lambda x: STATUS_RANK.get(x, 0)) if s else "PENDING"


def utc(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, timezone.utc)


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ") if dt else None


def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [jsonable(v) for v in (sorted(o) if isinstance(o, set) else o)]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        x = float(o)
        return x if math.isfinite(x) else str(x)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, Path):
        return scrub_path(o)
    if isinstance(o, datetime):
        return iso(o)
    return o


def scrub_path(p) -> str:
    s = str(p)
    for root, ph in ((str(PRIV), "<PRIVATE_CACHE>/qpc_v1"), (str(DPC_PRIV), "<PRIVATE_CACHE>/dpc_v1"),
                     (str(CACHE), "<PRIVATE_CACHE>"), (str(WT), "<WORKTREE>"), ("/Volumes", "<DRIVE_ROOT>")):
        s = s.replace(root, ph)
    return s


def git(*args, timeout=60):
    try:
        r = subprocess.run(["git", "-C", str(WT), *args], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def git_ok(*args) -> bool:
    return subprocess.run(["git", "-C", str(WT), *args], capture_output=True, text=True).returncode == 0


def git_show_bytes(rev, rel):
    r = subprocess.run(["git", "-C", str(WT), "show", f"{rev}:{rel}"], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def tload(p):
    """torch.load restricted to tensors / containers / primitives (no code objects)."""
    return torch.load(p, weights_only=True, map_location="cpu")


def maxdiff(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape:
        return float("inf")
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b)))


def bitwise(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)


def heavy_processes():
    """Other heavy study processes visible now (python processes whose command line names qpc / the semaphore), for the
    compute ledger. The semaphore is the binding rule; this is only an observation."""
    r = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True)
    me = os.getpid()
    out = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        pid, _, cmd = line.partition(" ")
        if int(pid) in (me, os.getppid()):
            continue
        if "python" in cmd and ("-m qpc" in cmd or "qpc/sema.py" in cmd or "replay_qpc" in cmd):
            out.append({"pid": int(pid), "kind": "sema" if "sema" in cmd else ("verifier" if "replay_qpc" in cmd
                                                                               else "qpc")})
    return out


def sema_status():
    """Read-only view of the semaphore from its append-only SEMA_LOG.jsonl (acquires without a matching release).
    The lead's slot lock files are never opened by the verifier."""
    open_, closed_by_reacquire = {}, []
    for e in jsonl(RUN / "SEMA_LOG.jsonl"):
        key = (e.get("wrapper_pid"), e.get("slot"))
        if e.get("event") == "acquire":
            for k2 in [k2 for k2 in open_ if k2[1] == e.get("slot")]:      # flock: the earlier holder had exited
                closed_by_reacquire.append(open_.pop(k2))
            open_[key] = e.get("label")
        elif e.get("event") == "release":
            open_.pop(key, None)
    return {"open_acquisitions": [{"slot": k[1], "label": v} for k, v in sorted(open_.items(), key=lambda x: str(x))],
            "unlogged_releases_closed_by_slot_reacquire": closed_by_reacquire}


# ------------------------------------------------------------------------------------------------ data and roles
def _u_int(seed, salt, gid) -> int:
    return int(hashlib.sha256(f"{seed}|{salt}|{int(gid)}".encode()).hexdigest()[:16], 16)


def _split(units, seed, salt, cuts, names):
    """Group-hash split: exact rational rule (authoritative) and float rule, with disagreement count."""
    m, dis = {}, 0
    for gid in np.unique(units).tolist():
        n = _u_int(seed, salt, gid)
        x, f = Fraction(n, 2 ** 64), n / 2.0 ** 64
        kx = next((names[i] for i, c in enumerate(cuts) if x < c), names[-1])
        kf = next((names[i] for i, c in enumerate(cuts) if f < float(c)), names[-1])
        dis += int(kx != kf)
        m[gid] = kx
    return np.array([m[gid] for gid in units.tolist()], dtype=object), dis


class Data:
    """Independent reconstruction of the osf roles (reused unchanged by dpc and qpc), subroles, pools and the 83-column
    inputs (structure adapted from the predecessor verifier)."""

    def __init__(self):
        self.src_sha = sha_file(SRC)
        z = np.load(SRC, allow_pickle=False)
        self.arrays_present = sorted(z.files)
        old = z["role"].astype(str)
        unit = z["unit"].astype(np.int64)
        rid = z["row_id"].astype(np.int64)
        X = np.asarray(z["X"])
        self.feature_names = [str(s) for s in z["feature_names"]]
        n = len(old)
        rgj = np.array([""] * n, dtype=object)
        for o, r in (("defense_train", "DEFENSE_FIT"), ("attacker_fit", "AUDIT_FIT"), ("attacker_val", INNER)):
            rgj[old == o] = r
        dv = old == "defense_val"
        lab, d1 = _split(unit[dv], 20261004, "dev", [Fraction(3, 10)], ["HEAD_VALIDATION", "DEVELOPMENT_ASSESSMENT"])
        rgj[dv] = lab
        df = rgj == "DEFENSE_FIT"
        smf = np.array([""] * n, dtype=object)
        lab, d3 = _split(unit[df], 20261005, "assess", [Fraction(1, 5)], ["NEW_DEVELOPMENT_ASSESSMENT", "NEW_DEFENSE_FIT"])
        smf[df] = lab
        for r in ("HEAD_VALIDATION", "AUDIT_FIT", INNER):
            smf[rgj == r] = r
        ssub = np.array([""] * n, dtype=object)
        nf = smf == "NEW_DEFENSE_FIT"
        lab, d4 = _split(unit[nf], 20261005, "critic", [Fraction(7, 10), Fraction(17, 20)],
                         ["CRITIC_FIT", "CRITIC_VAL", "CONTROLLER_CALIB"])
        ssub[nf] = lab
        self.disagree = {"rgj_dev_split": d1, "smf_assess_split": d3, "smf_critic_split": d4}
        role = np.array([""] * n, dtype=object)
        role[smf == "NEW_DEFENSE_FIT"] = FIT
        for r in ("HEAD_VALIDATION", "AUDIT_FIT", INNER):
            role[smf == r] = r
        sub = np.array([""] * n, dtype=object)
        sub[ssub == "CRITIC_FIT"] = "CRITIC_FIT"
        sub[ssub == "CRITIC_VAL"] = "CRITIC_VAL"
        sub[ssub == "CONTROLLER_CALIB"] = "DIAGNOSTIC_CALIB"
        pool = np.array([""] * n, dtype=object)
        pool[old == "assessment"] = "ORIG_ASSESSMENT"
        pool[rgj == "DEVELOPMENT_ASSESSMENT"] = "RGJ_DEV"
        pool[smf == "NEW_DEVELOPMENT_ASSESSMENT"] = "SMF_DEV"
        pool[old == "cert"] = "CERT"
        self.pool_overlap_with_role = int(((pool != "") & (role != "")).sum())
        nominal = pool.copy()
        blocked = set(np.unique(unit[np.isin(role.astype(str), FIT_ROLES) | np.isin(old, EXCLUSIONS)]).tolist())
        bad = (pool != "") & np.isin(unit, list(blocked))
        self.pool_excluded_rows = {p: int((bad & (pool == p)).sum()) for p in POOLS}
        pool[bad] = ""
        role[pool != ""] = ASSESS
        self.full = {"old": old, "role": role.astype(str), "sub": sub.astype(str), "pool": pool.astype(str),
                     "nominal_pool": nominal.astype(str), "unit": unit, "row_id": rid}
        keep = np.flatnonzero(role != "")
        self.keep = keep
        self.row_id = rid[keep]
        self.unit = unit[keep]
        self.role = role[keep].astype(str)
        self.sub = sub[keep].astype(str)
        self.pool = pool[keep].astype(str)
        self.mask = {r: self.role == r for r in ROLES}
        self.mask.update({r: self.sub == r for r in SUBROLES})
        adm = jload(ADMISSION_NORM)
        Xk = X[keep].astype(np.float64)
        self.numeric = {}
        fm = self.mask[FIT]
        for c in NUMERIC:
            j = self.feature_names.index(c)
            mu, sd = adm["numeric_norm"][c]
            inv = X[:, j].astype(np.float64) * sd + mu
            r = np.round(inv)
            rk = r[keep]
            m, s = float(rk[fm].mean()), float(rk[fm].std())
            Xk[:, j] = (rk - m) / s
            self.numeric[c] = {"inversion_max_abs_err_kept_rows": float(np.abs(inv - r)[keep].max()),
                               "mean_osf_fit": m, "sd_osf_fit": s}
        self.X = np.ascontiguousarray(Xk.astype(np.float32))
        self.idx = {r: np.flatnonzero(self.mask[r]) for r in ROLES}
        self.fit_idx = self.idx[FIT]
        self._sealed = None

    def labels(self):
        """Labels on kept rows, OSF_DEVELOPMENT_ASSESSMENT masked to -1 (unsealing is a separate gated function)."""
        if self._sealed is None:
            z = np.load(SRC, allow_pickle=False)
            out = {}
            for k, v in LABEL_KEYS.items():
                a = z[v].astype(np.int64)[self.keep].copy()
                a[self.mask[ASSESS]] = -1
                out[k] = a
            self._sealed = out
        return self._sealed


def check_roles(D: Data, manifests):
    """Own roles vs every available manifest: qpc ROLE_MANIFEST.json (role E) when present, the dpc manifest at the
    source evidence SHA, and the pinned osf role-partition fingerprint."""
    f = D.full
    out = {"source_sha256_ok": D.src_sha == SRC_SHA, "float_vs_exact_disagreements": D.disagree,
           "pool_rows_with_a_fitting_role_before_exclusion": D.pool_overlap_with_role}

    def rec(ix):
        return {"rows": int(len(ix)), "groups": int(len(np.unique(f["unit"][ix]))),
                "row_id_sha256": rowid_hash(f["row_id"][ix]), "group_id_set_sha256": unit_set_hash(f["unit"][ix])}

    keys4 = ("rows", "groups", "row_id_sha256", "group_id_set_sha256")
    mine = {r: rec(np.flatnonzero(f["role"] == r)) for r in ROLES}
    sub = {r: rec(np.flatnonzero(f["sub"] == r)) for r in SUBROLES}
    out["expected_counts_hold"] = {r: (mine[r]["rows"], mine[r]["groups"]) == EXPECTED_COUNTS[r] for r in ROLES}
    out["counts"] = {r: {"rows": mine[r]["rows"], "groups": mine[r]["groups"]} for r in ROLES}
    out["role_hashes"] = {r: {k: mine[r][k] for k in ("row_id_sha256", "group_id_set_sha256")} for r in ROLES}
    man_ok = {}
    for name, man in manifests.items():
        if man is None:
            man_ok[name] = None
            continue
        roles = man.get("roles", {})
        rm = {r: (r in roles and all(mine[r][k] == roles[r].get(k) for k in keys4)) for r in ROLES}
        sm = man.get("defense_fit_subroles")
        rs = {r: all(sub[r][k] == sm[r][k] for k in ("rows", "groups", "row_id_sha256")) for r in SUBROLES} if sm else None
        man_ok[name] = {"roles_match": rm, "subroles_match": rs}
    out["manifests"] = man_ok
    osf_man = json.loads(git_show_bytes(TEACHER_PIN, f"{OSF_REL}/ROLE_MANIFEST.json") or b"{}")
    allr = dict(mine)
    allr.update(sub)
    fp = hashlib.sha256("|".join(f"{r}:{allr[r]['row_id_sha256']}" for r in ROLES + SUBROLES).encode()).hexdigest()
    out["pinned_osf_role_partition_fingerprint_match"] = fp == osf_man.get("role_partition_fingerprint_sha256")
    out["role_partition_fingerprint_sha256"] = fp
    groups = {r: set(np.unique(f["unit"][f["role"] == r]).tolist()) for r in ROLES}
    out["roles_group_disjoint"] = all(not (groups[a] & groups[b]) for i, a in enumerate(ROLES) for b in ROLES[i + 1:])
    out["defense_fit_groups_shared_with_other_roles"] = {r: len(groups[FIT] & groups[r]) for r in ROLES if r != FIT}
    out["subroles_partition_fit"] = bool(np.array_equal(np.sort(np.flatnonzero(np.isin(f["sub"], SUBROLES))),
                                                        np.flatnonzero(f["role"] == FIT)))
    excl = np.isin(f["old"], EXCLUSIONS)
    out["no_exclusion_row_kept"] = bool((f["role"][excl] == "").all())
    kept_groups = set(np.unique(f["unit"][f["role"] != ""]).tolist())
    dropped_groups = set(np.unique(f["unit"][f["role"] == ""]).tolist())
    out["no_kept_group_in_a_dropped_row"] = not (kept_groups & dropped_groups)
    out["dropped_rows"] = int((f["role"] == "").sum())
    out["feature_names_sha256"] = hashlib.sha256("\n".join(D.feature_names).encode()).hexdigest()
    low = [s.lower().split("=")[0] for s in D.feature_names]
    out["n_columns"] = len(D.feature_names)
    out["forbidden_columns_present"] = sorted({s for s in low if s in ("sex", "race", "income", "occupation", "fnlwgt")})
    out["numeric_inversion_ok"] = {c: D.numeric[c]["inversion_max_abs_err_kept_rows"] < 0.05 for c in NUMERIC}
    out["rows_kept"] = int(len(D.keep))
    present = [v for v in man_ok.values() if v is not None]
    ok = (out["source_sha256_ok"] and all(out["expected_counts_hold"].values()) and present
          and all(all(v["roles_match"].values()) and (v["subroles_match"] is None or all(v["subroles_match"].values()))
                  for v in present)
          and out["pinned_osf_role_partition_fingerprint_match"] and out["roles_group_disjoint"]
          and out["subroles_partition_fit"] and out["no_exclusion_row_kept"] and out["no_kept_group_in_a_dropped_row"]
          and out["n_columns"] == 83 and not out["forbidden_columns_present"] and sum(D.disagree.values()) == 0
          and D.pool_overlap_with_role == 0 and out["rows_kept"] == ROWS_KEPT and all(out["numeric_inversion_ok"].values()))
    st = "PASS" if ok else "FAIL"
    if ok and manifests.get("qpc_ROLE_MANIFEST") is None:
        st = "WARN"
        out["note"] = "qpc ROLE_MANIFEST.json not yet present; checked against the dpc manifest at the source SHA"
    return res(st, **out)


# ------------------------------------------------------------------------------------------------ own forward pass
def encode(sd, i, X):
    """Recipient i encoder: Linear(83,64)-ReLU-Linear(64,64)-ReLU-Linear(64,16), float32 as stored -> float64."""
    h = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    with torch.no_grad():
        for j, act in ((0, True), (2, True), (4, False)):
            h = TF.linear(h, sd[f"enc.{i}.{j}.weight"], sd[f"enc.{i}.{j}.bias"])
            if act:
                h = torch.relu(h)
    return h.double().numpy()


def head_outputs(head, r):
    """Deployed-head outputs: centred logits (binary margin d -> (0, d)), probabilities, first-index argmax."""
    r = np.asarray(r, dtype=np.float64)
    d = np.asarray(head.decision_function(r), dtype=np.float64)
    if d.ndim == 1:
        d = np.stack([np.zeros_like(d), d], 1)
    P = head.predict_proba(r)
    return d - d.mean(1, keepdims=True), P, P.argmax(1)


def my_fit_head(R, y, tr, va, K):
    """StandardScaler + LogisticRegression(C, max_iter 3000) on OSF_DEFENSE_FIT, C by HEAD_VALIDATION log loss; a later
    C replaces the incumbent only on strict improvement > 1e-12 (historical head rule; descriptive refit)."""
    best, table = None, []
    for C in HEAD_C:
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
        m.fit(R[tr], y[tr])
        ll = float(sk_log_loss(y[va], m.predict_proba(R[va]), labels=list(range(K))))
        table.append({"C": C, "log_loss": ll})
        if best is None or ll < best[0] - 1e-12:
            best = (ll, C, m)
    return best[2], best[1], table


def own_teacher(model_pt, head_paths, X):
    """Own forward application of one admitted teacher: encoders from model.pt, deployed joblib heads."""
    sd = tload(model_pt)
    out = {"state_keys": sorted(sd)}
    for i in (0, 1):
        j = i + 1
        r = encode(sd, i, X)
        head = joblib.load(head_paths[i])
        c, p, d = head_outputs(head, r)
        out.update({f"r{j}": r, f"c{j}": c, f"p{j}": p, f"d{j}": d.astype(np.int64), f"head{j}": head})
    return out


# ------------------------------------------------------------------------------------------------ own method math
def smooth(M, c):
    """(mean + eps*1 + eps*e_c) / (1 + (K+1) eps), row-wise; c scalar or per row."""
    M = np.atleast_2d(np.asarray(M, dtype=np.float64))
    K = M.shape[1]
    c = np.broadcast_to(np.asarray(c, dtype=np.int64), (M.shape[0],))
    E = np.zeros_like(M)
    E[np.arange(M.shape[0]), c] = EPS
    return (M + EPS * np.ones_like(M) + E) / (1.0 + (K + 1) * EPS)


def strict_argmax_ok(Q, c):
    Q = np.atleast_2d(np.asarray(Q, dtype=np.float64))
    c = np.broadcast_to(np.asarray(c, dtype=np.int64), (Q.shape[0],))
    if Q.shape[0] == 0:
        return True
    qc = Q[np.arange(Q.shape[0]), c]
    other = Q.copy()
    other[np.arange(Q.shape[0]), c] = -np.inf
    return bool(np.all(qc > other.max(1)))


def kl_to_many(P, Q):
    """KL(p_r || q_j) for all r, j: sum over k in increasing order of p_k (log p_k - log q_jk) where p_k > 0 (the
    registered summation rule; 0 log 0 = 0). Q must be strictly positive."""
    P = np.asarray(P, dtype=np.float64)
    Q = np.atleast_2d(np.asarray(Q, dtype=np.float64))
    if np.any(Q <= 0):
        raise ValueError("KL reference must be strictly positive")
    LQ = np.log(Q)
    acc = np.zeros((P.shape[0], Q.shape[0]))
    for k in range(P.shape[1]):
        pk = P[:, k][:, None]
        pos = pk > 0
        lp = np.log(np.where(pos, pk, 1.0))
        acc = acc + np.where(pos, pk * (lp - LQ[None, :, k]), 0.0)
    return acc


def kl_paired(P, Q):
    """KL(p_r || q_r) row-wise with the same summation rule."""
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    acc = np.zeros(P.shape[0])
    for k in range(P.shape[1]):
        pk = P[:, k]
        pos = pk > 0
        acc = acc + np.where(pos, pk * (np.log(np.where(pos, pk, 1.0)) - np.log(Q[:, k])), 0.0)
    return acc


def negent_rows(P):
    """sum_k p_k log p_k (0 log 0 = 0), increasing k."""
    P = np.asarray(P, dtype=np.float64)
    acc = np.zeros(P.shape[0])
    for k in range(P.shape[1]):
        pk = P[:, k]
        pos = pk > 0
        acc = acc + np.where(pos, pk * np.log(np.where(pos, pk, 1.0)), 0.0)
    return acc


def group_sums(cell, W, F):
    """Sequential (row-order) per-cell sums via unbuffered np.add.at; W (n,) or (n, K)."""
    cell = np.asarray(cell, dtype=np.int64)
    W = np.asarray(W, dtype=np.float64)
    out = np.zeros((F,) + W.shape[1:])
    np.add.at(out, cell, W)
    return out


def stats_of(P, cell, F):
    n = np.zeros(F, dtype=np.int64)
    np.add.at(n, np.asarray(cell, dtype=np.int64), 1)
    return n, group_sums(cell, P, F), group_sums(cell, negent_rows(P), F)


def distinct_lex(Pc):
    """Distinct rows, ascending lexicographic (own: lexsort with the first column as primary key)."""
    if Pc.shape[0] == 0:
        return Pc.copy()
    order = np.lexsort(Pc.T[::-1])
    S = Pc[order]
    keep = np.ones(S.shape[0], dtype=bool)
    keep[1:] = np.any(S[1:] != S[:-1], axis=1)
    return S[keep]


def guard_nonneg(Dv, tol=KL_ROUNDOFF):
    """k-means++ weights: KL values in [-tol, 0) are roundoff and set to 0 (counted); anything below -tol is an error,
    never silently clipped (prompt A2)."""
    Dv = np.asarray(Dv, dtype=np.float64)
    if np.any(Dv < -tol):
        raise ValueError(f"KL weight below -{tol}: {float(Dv.min())}")
    neg = Dv < 0
    return np.where(neg, 0.0, Dv), int(neg.sum())


# ------------------------------------------------------------------------------------------------ own fine partitions
class Fine:
    """An assignment partition: cells in (class, within-class index) order, smoothed assignment centroids, unsmoothed
    member means, sufficient statistics, fallback flags."""

    def __init__(self, K, cls, cen, mean, n, S, A, fb, receipt=None):
        self.K, self.cls, self.cen, self.mean, self.n, self.S, self.A, self.fb = K, cls, cen, mean, n, S, A, fb
        self.receipt = receipt or {}

    @property
    def F(self):
        return int(len(self.cls))

    @classmethod
    def from_json(cls, z):
        K = int(z["K"])
        f = cls(K, np.asarray(z["cell_class"], dtype=np.int64), np.asarray(z["centroid"], dtype=np.float64).reshape(-1, K),
                np.asarray(z["mean"], dtype=np.float64).reshape(-1, K), np.asarray(z["n"], dtype=np.int64),
                np.asarray(z["S"], dtype=np.float64).reshape(-1, K), np.asarray(z["A"], dtype=np.float64),
                np.asarray(z["fallback"], dtype=bool), z.get("receipt", {}))
        f.stored_fp = z.get("fingerprint")
        f.kind = z.get("kind")
        return f

    def fingerprint_dpc(self):
        """dpc FinePartition.fingerprint convention re-implemented: K, classes, centroids, fallback, n, S, A."""
        h = hashlib.sha256()
        for a in (np.int64(self.K), self.cls.astype("<i8"), self.cen.astype("<f8"), self.fb.astype(np.uint8),
                  self.n.astype("<i8"), self.S.astype("<f8"), self.A.astype("<f8")):
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()

    def equal(self, o):
        return {"K": self.K == o.K, "cell_class": bitwise(self.cls, o.cls), "centroid": bitwise(self.cen, o.cen),
                "mean": bitwise(self.mean, o.mean), "n": bitwise(self.n, o.n), "S": bitwise(self.S, o.S),
                "A": bitwise(self.A, o.A), "fallback": bitwise(self.fb, o.fb)}


def class_objective(Ac, n, S, c):
    """Registered k-means objective of a coherent iterate: J(a) = A_c + sum_{j: n_j > 0} h_j with
    h_j = -S_j . log smooth(S_j / n_j, c) (the class total of KL(p || decoded prototype)); A_c = sum over the class's
    rows of sum_k p_k log p_k, computed once."""
    h = 0.0
    for j in np.flatnonzero(n > 0):
        q = smooth(S[j] / n[j], c)[0]
        h += -float(np.dot(S[j], np.log(q)))
    return Ac + h


def distinct_counts(Pc):
    """Distinct rows ascending lexicographic with multiplicities (own; equals np.unique(axis=0, return_counts))."""
    order = np.lexsort(Pc.T[::-1])
    S = Pc[order]
    new = np.ones(S.shape[0], dtype=bool)
    new[1:] = np.any(S[1:] != S[:-1], axis=1)
    starts = np.flatnonzero(new)
    w = np.diff(np.r_[starts, S.shape[0]]).astype(np.int64)
    return S[starts], w


def init_source(Pc, c, k):
    """Source deterministic initialisation (dpc rule): distinct vectors ascending lexicographic, stably re-sorted by
    p_c descending, block-midpoint order statistics floor((2j+1) nU / (2k))."""
    U = distinct_lex(Pc)
    order = np.argsort(-U[:, c], kind="stable")
    U = U[order]
    nU = U.shape[0]
    pos = [((2 * j + 1) * nU) // (2 * k) for j in range(k)]
    return U[pos].copy(), {"init": "source", "positions": [int(x) for x in pos], "distinct_vectors": int(nU)}


def init_kmeanspp(Pc, c, k, seed):
    """Registered KL k-means++ (METHOD_CARD section 3 / kmeans rule text): distinct vectors U (ascending lexicographic)
    with multiplicities w; rng = default_rng([seed, K, c]); exactly one rng.random() per centre; draw: cum = cumsum(v),
    t = u * cum[-1], i = searchsorted(cum, t, 'right'), i == |U| -> last index with v > 0. Centre 1: v = w. Centre j>1:
    v = w * min over chosen of KL(u || smooth(chosen, c)), chosen set to exactly 0; if every unchosen v is 0, v = w on
    the unchosen vectors (degenerate draw). KL values in [-1e-12, 0) are clipped to 0 (counted); below raises."""
    K = Pc.shape[1]
    U, w = distinct_counts(Pc)
    nU = U.shape[0]
    rng = np.random.default_rng([int(seed), int(K), int(c)])
    wf = w.astype(np.float64)

    def draw(v):
        cum = np.cumsum(v)
        t = rng.random() * cum[-1]
        i = int(np.searchsorted(cum, t, side="right"))
        if i == nU:
            i = int(np.flatnonzero(v > 0)[-1])
        return i

    chosen = [draw(wf)]
    dmin, clipped, degenerate = None, 0, 0
    while len(chosen) < k:
        kl, ncl = guard_nonneg(kl_to_many(U, smooth(U[chosen[-1]], c))[:, 0])
        clipped += ncl
        dmin = kl if dmin is None else np.minimum(dmin, kl)
        v = wf * dmin
        v[chosen] = 0.0
        un = np.ones(nU, dtype=bool)
        un[chosen] = False
        if not np.any(v[un] > 0):
            v = np.where(un, wf, 0.0)
            degenerate += 1
        chosen.append(draw(v))
    return U[chosen].copy(), {"init": f"kpp:{seed}", "chosen_distinct_index": [int(x) for x in chosen],
                              "distinct_vectors": int(nU), "kl_clipped": int(clipped), "degenerate_draws": degenerate}


def lloyd_dpc(Pc, c, C0, rounds):
    """The historical dpc rule exactly (A1 reproduction): at most `rounds` rounds; round r assigns with the smoothed
    centroids; stop when r > 1 and the assignment equals round r-1's; else update to member means (empty cells keep
    their centroid); if the cap is reached a final assignment with the last centroids is made; LAST iterate returned."""
    k = C0.shape[0]
    C = C0.copy()
    Q = smooth(C, np.full(k, c))
    prev, converged, used, changed = None, False, 0, []
    for r in range(1, rounds + 1):
        a = kl_to_many(Pc, Q).argmin(1)
        used = r
        if prev is not None and np.array_equal(a, prev):
            converged = True
            break
        changed.append(int(Pc.shape[0] if prev is None else np.sum(a != prev)))
        n, S, _ = stats_of(Pc, a, k)
        for j in range(k):
            if n[j] > 0:
                C[j] = S[j] / n[j]
        Q = smooth(C, np.full(k, c))
        prev = a
    if not converged:
        a = kl_to_many(Pc, Q).argmin(1)
        converged = bool(prev is not None and np.array_equal(a, prev))
    return Q, a, {"rule": "dpc", "rounds_used": int(used), "converged": bool(converged), "changed_per_round": changed,
                  "stop": "assignment_fixed_point" if converged else "cap"}


def lloyd_qpc(Pc, c, C0, rounds=A2_ROUNDS, rtol=1e-9, patience=PATIENCE):
    """Registered qpc rule. Pass r assigns with the smoothed centroids (coherent iterate (Q, a)); J by class_objective.
    Stop rules in order: (1) r > 1 and a_r == a_{r-1} -> converged 'assignment_fixed_point'; (2) rel_r =
    |J_{r-1} - J_r| / max(|J_{r-1}|, 1e-300) < rtol for `patience` successive passes -> one final update + assignment
    pass, converged 'relative_tolerance'; (3) r == rounds -> one final update + assignment pass, converged
    ('assignment_fixed_point') only if it equals pass r, else 'cap'. Otherwise update to member means (empty cells keep
    their centroid). Returns the best coherent iterate: lowest J over every assignment pass, later passes win ties."""
    k = C0.shape[0]
    C = C0.copy()
    cc = np.full(k, c)
    Ac = float(np.sum(negent_rows(Pc)))
    Q = smooth(C, cc)
    prev, Jprev, run_small = None, None, 0
    best, Js, changed, empties = None, [], [], []
    reason, converged = None, False

    def update(a, n, S):
        for j in range(k):
            if n[j] > 0:
                C[j] = S[j] / n[j]
        empties.append(int(np.sum(n == 0)))
        return smooth(C, cc)

    def one_pass(Q_, prev_):
        a_ = kl_to_many(Pc, Q_).argmin(1)
        n_, S_, _ = stats_of(Pc, a_, k)
        J_ = class_objective(Ac, n_, S_, c)
        Js.append(J_)
        changed.append(int(Pc.shape[0] if prev_ is None else np.sum(a_ != prev_)))
        return a_, n_, S_, J_

    def keep_best(J_, Q_, a_, r_):
        nonlocal best
        if best is None or J_ <= best[0]:
            best = (J_, r_, Q_.copy(), a_.copy())

    r = 0
    while True:
        r += 1
        a, n, S, J = one_pass(Q, prev)
        keep_best(J, Q, a, r)
        if prev is not None and np.array_equal(a, prev):
            reason, converged = "assignment_fixed_point", True
            break
        if Jprev is not None:
            rel = abs(Jprev - J) / max(abs(Jprev), 1e-300)
            run_small = run_small + 1 if rel < rtol else 0
        if run_small >= patience or r == rounds:
            tol_stop = run_small >= patience
            Q = update(a, n, S)
            a2, n2, S2, J2 = one_pass(Q, a)
            keep_best(J2, Q, a2, r + 1)
            if tol_stop:
                reason, converged = "relative_tolerance", True
            elif np.array_equal(a2, a):
                reason, converged = "assignment_fixed_point", True
            else:
                reason, converged = "cap", False
            r += 1
            break
        Q = update(a, n, S)
        prev, Jprev = a, J
    Jb, rb, Qb, ab = best
    return Qb, ab, {"rule": "qpc", "assignment_passes": int(r), "stop": reason, "converged": bool(converged),
                    "best_pass": int(rb), "J_best": Jb, "J_last": Js[-1], "J_trace": Js, "changed_per_pass": changed,
                    "empty_cells_per_update": empties}


def fit_class(Pc, c, m, starts, rule, rounds=None):
    """All registered starts for one predicted class (fixed order = tie order). rule 'dpc': source start, 20 rounds,
    last iterate. rule 'qpc': best coherent iterate; a later start replaces the incumbent only if
    J_new < J_inc - 1e-12 max(|J_inc|, 1)."""
    nrows = Pc.shape[0]
    nU = distinct_lex(Pc).shape[0]
    k = int(min(m, nU, nrows))
    Ac = float(np.sum(negent_rows(Pc)))
    cand = []
    for st in starts:
        if st == "source":
            C0, irec = init_source(Pc, c, k)
        else:
            C0, irec = init_kmeanspp(Pc, c, k, int(st.split(":")[1]))
        if rule == "dpc":
            Q, a, rec = lloyd_dpc(Pc, c, C0, rounds or A1_ROUNDS)
        else:
            Q, a, rec = lloyd_qpc(Pc, c, C0, rounds or A2_ROUNDS)
        n, S, _ = stats_of(Pc, a, Q.shape[0])
        J = class_objective(Ac, n, S, c)
        cand.append({"start": st, "Q": Q, "a": a, "J": J, "init": irec, "rec": rec, "k": int(C0.shape[0])})
    w = 0
    for j in range(1, len(cand)):
        Ji = cand[w]["J"]
        if cand[j]["J"] < Ji - 1e-12 * max(abs(Ji), 1.0):
            w = j
    return cand[w], cand, {"rows": int(nrows), "distinct_vectors": int(nU), "requested_cells": int(k)}


def fit_partition(P, d, K, m, starts=("source",), rule="qpc", rounds=None):
    """Own class-preserving task-only partition on fitting rows: per predicted class the registered starts and winner;
    cells empty in the returned assignment pruned (order kept); absent class -> one reserved fallback cell."""
    P = np.asarray(P, dtype=np.float64) + 0.0
    d = np.asarray(d, dtype=np.int64)
    names = ["winner"] + list(starts)
    parts = {nm: {k_: [] for k_ in ("cls", "cen", "mean", "n", "S", "A", "fb")} for nm in names}

    def put(nm, cls_, cen, mean, n_, S_, A_, fb):
        for k_, v in (("cls", cls_), ("cen", cen), ("mean", mean), ("n", n_), ("S", S_), ("A", A_), ("fb", fb)):
            parts[nm][k_].append(np.asarray(v))

    per = []
    for c in range(K):
        rows = np.flatnonzero(d == c)
        if rows.size == 0:
            u = np.full(K, 1.0 / K)
            for nm in names:
                put(nm, [c], smooth(u, c), u[None], [0], np.zeros((1, K)), np.zeros(1), [True])
            per.append({"class": c, "rows": 0, "fallback": True, "cells": 1})
            continue
        Pc = P[rows]
        win, cand, info = fit_class(Pc, c, m, starts, rule, rounds)
        for nm, x in [("winner", win)] + [(x["start"], x) for x in cand]:
            Q, a = x["Q"], x["a"]
            n, S, A = stats_of(Pc, a, Q.shape[0])
            keep = np.flatnonzero(n > 0)
            put(nm, np.full(keep.size, c), Q[keep], S[keep] / n[keep, None], n[keep], S[keep], A[keep],
                np.zeros(keep.size, dtype=bool))
            x["final_cell_counts"] = [int(v) for v in n]
            x["cells"] = int(keep.size)
            if nm == "winner":
                n_w, keep_w, Q_w = n, keep, Q
        per.append({"class": c, **info, "fallback": False, "winner": win["start"], "cells": int(keep_w.size),
                    "removed_empty_cells": int(Q_w.shape[0] - keep_w.size), "cell_counts": [int(x) for x in n_w[keep_w]],
                    "starts": [{"start": x["start"], "J": x["J"], "k": x["k"], "init": x["init"],
                                "final_cell_counts": x["final_cell_counts"], "cells": x["cells"],
                                **{kk: v for kk, v in x["rec"].items()}} for x in cand]})

    def build(nm):
        q = parts[nm]
        return Fine(K, np.concatenate(q["cls"]).astype(np.int64), np.concatenate(q["cen"]),
                    np.concatenate(q["mean"]), np.concatenate(q["n"]).astype(np.int64), np.concatenate(q["S"]),
                    np.concatenate(q["A"]).astype(np.float64), np.concatenate(q["fb"]).astype(bool))

    fine = build("winner")
    fine.receipt = {"per_class": per, "by_start": {nm: build(nm) for nm in starts}}
    return fine


def assign(P, d, fine: Fine):
    """Deployment: nearest cell (KL to the stored smoothed centroid) within the row's predicted class; ties -> lowest
    index; a class with one cell (incl. a fallback) maps to it. Never updates anything."""
    P = np.asarray(P, dtype=np.float64)
    d = np.asarray(d, dtype=np.int64)
    out = np.full(P.shape[0], -1, dtype=np.int64)
    for c in range(fine.K):
        rows = np.flatnonzero(d == c)
        if rows.size == 0:
            continue
        idx = np.flatnonzero(fine.cls == c)
        if idx.size == 1:
            out[rows] = idx[0]
        elif idx.size > 1:
            out[rows] = idx[kl_to_many(P[rows], fine.cen[idx]).argmin(1)]
    return out


# ------------------------------------------------------------------------------------------------ own policies
class Pol:
    """One recipient's public policy: assignment partition + cell->token map; prototypes recomputed from the cell sums
    (sequential accumulation in increasing cell index), fallback tokens decode to smooth(uniform, class)."""

    def __init__(self, fine: Fine, cell_token, stored=None):
        self.fine = fine
        self.cell_token = np.asarray(cell_token, dtype=np.int64)
        T = int(self.cell_token.max()) + 1
        K = fine.K
        tc, tn, tS, tfb = np.full(T, -1, dtype=np.int64), np.zeros(T, dtype=np.int64), np.zeros((T, K)), np.zeros(T, bool)
        self.mixes_classes = False
        for f in range(fine.F):
            t = int(self.cell_token[f])
            c = int(fine.cls[f])
            if tc[t] == -1:
                tc[t] = c
            elif tc[t] != c:
                self.mixes_classes = True
            tn[t] += fine.n[f]
            tS[t] = tS[t] + fine.S[f]
            tfb[t] = tfb[t] or bool(fine.fb[f])
        self.unused_ids = int(np.sum(tc < 0))
        proto = np.zeros((T, K))
        for t in range(T):
            proto[t] = smooth(tS[t] / tn[t], tc[t])[0] if tn[t] > 0 else smooth(np.full(K, 1.0 / K), max(tc[t], 0))[0]
        self.T, self.token_class, self.token_n, self.token_S, self.token_fb, self.proto = T, tc, tn, tS, tfb, proto
        seen, canon = {}, np.empty(fine.F, dtype=np.int64)
        for f in range(fine.F):
            canon[f] = seen.setdefault(int(self.cell_token[f]), len(seen))
        self.canonical = bool(np.array_equal(canon, self.cell_token))
        self.stored = stored or {}

    @classmethod
    def from_json(cls, z):
        return cls(Fine.from_json(z["fine"]), z["cell_token"],
                   {"proto": np.asarray(z["token_proto"], dtype=np.float64), "class": np.asarray(z["token_class"], np.int64),
                    "n": np.asarray(z["token_n"], dtype=np.int64), "fingerprint": z.get("fingerprint")})

    def fingerprint_dpc(self):
        """dpc Policy.fingerprint convention re-implemented from own token classes and prototypes."""
        h = hashlib.sha256()
        for a in (np.int64(self.fine.K), self.fine.cls.astype("<i8"), self.fine.cen.astype("<f8"),
                  self.cell_token.astype("<i8"), self.token_class.astype("<i8"), self.proto.astype("<f8")):
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()

    def release(self, P, d):
        f = assign(P, d, self.fine)
        tok = self.cell_token[f]
        return tok, self.proto[tok], self.token_class[tok], f

    def structure(self):
        P = self.proto
        sums = np.abs(P.sum(1) - 1.0)
        out = {"no_class_mixing": not self.mixes_classes, "all_ids_used": self.unused_ids == 0,
               "canonical_ids": self.canonical,
               "every_class_has_a_token": all(np.any(self.token_class == c) for c in range(self.fine.K)),
               "fallback_tokens_have_no_fitting_rows": bool(not np.any(self.token_fb & (self.token_n > 0))),
               "prototypes_finite_strictly_positive": bool(np.isfinite(P).all() and (P > 0).all()),
               "prototype_sums_within_tol": bool(sums.max() <= SUM_TOL),
               "prototype_strict_argmax_is_class": strict_argmax_ok(P, self.token_class)}
        if self.stored:
            out.update({"prototypes_bitwise_equal_stored": bitwise(P, self.stored["proto"]),
                        "token_class_equal_stored": bitwise(self.token_class, self.stored["class"]),
                        "token_n_equal_stored": bitwise(self.token_n, self.stored["n"])})
        return out

    def tokens_per_class(self):
        return [int(np.sum(self.token_class == c)) for c in range(self.fine.K)]

    def effective_per_class(self):
        return [int(np.sum((self.token_class == c) & (self.token_n > 0))) for c in range(self.fine.K)]


def direct_policy(fine: Fine):
    """DIRECT-TASK: identity map (each assignment cell is its own token; cells are already in canonical order)."""
    return Pol(fine, np.arange(fine.F, dtype=np.int64))


def class_only_policy(K, P_fit, d_fit):
    """CLASS-ONLY: one token per predicted class decoding to smooth(class mean) (fallback uniform)."""
    n, S, A = stats_of(P_fit, d_fit, K)
    fb = n == 0
    mean = np.where(fb[:, None], 1.0 / K, S / np.maximum(n, 1)[:, None])
    fine = Fine(K, np.arange(K, dtype=np.int64), smooth(mean, np.arange(K)), mean, n, S, A, fb)
    return Pol(fine, np.arange(K, dtype=np.int64))


# ------------------------------------------------------------------------------------------------ own MI / objectives
def mi_counts(tab):
    """Plug-in I(S;C) in nats from a (2, C) count table; empty cells contribute 0."""
    tab = np.asarray(tab, dtype=np.float64)
    N = tab.sum()
    if N == 0:
        return 0.0
    rs = tab.sum(1, keepdims=True)
    cs = tab.sum(0, keepdims=True)
    tot = 0.0
    for s in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            v = tab[s, j]
            if v > 0:
                tot += v / N * math.log(v * N / (rs[s, 0] * cs[0, j]))
    return float(tot)


def mi_of(s, *codes):
    """Plug-in I(S; full code identity) from per-row labels (exact tuple of every code)."""
    keys = {}
    ids = np.empty(len(s), dtype=np.int64)
    cols = [np.asarray(c, dtype=np.int64).tolist() for c in codes]
    for i, tup in enumerate(zip(*cols)):
        ids[i] = keys.setdefault(tup, len(keys))
    tab = np.zeros((2, len(keys)))
    np.add.at(tab, (np.asarray(s, dtype=np.int64), ids), 1.0)
    return mi_counts(tab)


def entropy_mi(s, code):
    """H(S) - H(S | C), an alternative formula (self-test)."""
    s = np.asarray(s)
    N = len(s)
    ps = np.bincount(s, minlength=2) / N
    hs = -sum(p * math.log(p) for p in ps if p > 0)
    hc = 0.0
    for v in np.unique(code):
        mk = code == v
        q = np.bincount(s[mk], minlength=2) / mk.sum()
        hc += mk.sum() / N * -sum(p * math.log(p) for p in q if p > 0)
    return hs - hc


def objectives(P1, t1, q1, P2, t2, q2, s, lam):
    o = {"D1": float(np.mean(kl_paired(P1, q1))), "D2": float(np.mean(kl_paired(P2, q2))),
         "I1": mi_of(s, t1), "I2": mi_of(s, t2), "I12": mi_of(s, t1, t2)}
    o["F_task"] = o["D1"] + o["D2"]
    if lam is not None:
        o["F_local"] = o["D1"] + o["D2"] + lam * (o["I1"] + o["I2"]) / 2
        o["F_joint"] = o["D1"] + o["D2"] + lam * ((o["I1"] + o["I2"]) / 2 + o["I12"])
    return o


class Engine:
    """From-scratch objective of any pair of fine-cell labelings on the fitting rows (own arithmetic): D_r from the
    merged sufficient statistics, I_r and I12 from the (S, f1, f2) contingency table."""

    def __init__(self, fine1: Fine, fine2: Fine, f1, f2, s):
        self.fine = {1: fine1, 2: fine2}
        self.N = int(len(s))
        self.F = {1: fine1.F, 2: fine2.F}
        T = np.zeros((2, fine1.F, fine2.F))
        np.add.at(T, (np.asarray(s, np.int64), np.asarray(f1, np.int64), np.asarray(f2, np.int64)), 1.0)
        self.T = T
        self.Atot = {r: float(np.sum(self.fine[r].A)) for r in (1, 2)}

    def D(self, r, lab):
        fn = self.fine[r]
        G = int(lab.max()) + 1
        n = group_sums(lab, fn.n.astype(np.float64), G)
        S = group_sums(lab, fn.S, G)
        cls = np.zeros(G, dtype=np.int64)
        cls[lab] = fn.cls
        u = n > 0
        q = smooth(S[u] / n[u, None], cls[u])
        return (self.Atot[r] - float(np.sum(S[u] * np.log(q)))) / self.N

    def I(self, r, lab):
        M = self.T.sum(2) if r == 1 else self.T.sum(1)        # (2, F_r)
        G = int(lab.max()) + 1
        return mi_counts(np.stack([group_sums(lab, M[v], G) for v in (0, 1)]))

    def I12(self, lab1, lab2):
        G1, G2 = int(lab1.max()) + 1, int(lab2.max()) + 1
        out = np.zeros((2, G1, G2))
        for v in (0, 1):
            tmp = np.zeros((G1, self.F[2]))
            np.add.at(tmp, lab1, self.T[v])
            np.add.at(out[v].T, lab2, tmp.T)
        return mi_counts(out.reshape(2, -1))

    def terms(self, lab):
        return {"D1": self.D(1, lab[1]), "D2": self.D(2, lab[2]), "I1": self.I(1, lab[1]), "I2": self.I(2, lab[2]),
                "I12": self.I12(lab[1], lab[2])}


def F_of(t, lam, kind):
    if kind == "task":
        return t["D1"] + t["D2"]
    if kind == "local":
        return t["D1"] + t["D2"] + lam * (t["I1"] + t["I2"]) / 2
    return t["D1"] + t["D2"] + lam * ((t["I1"] + t["I2"]) / 2 + t["I12"])


def stage1_objective(eng: Engine, a, lab_a, lam):
    """Corrected sequential stage one (prompt section 8): F_joint with the other recipient at its CLASS-ONLY labels."""
    b = 2 if a == 1 else 1
    lab = {a: lab_a, b: eng.fine[b].cls.copy()}
    return F_of(eng.terms(lab), lam, "joint")


# ------------------------------------------------------------------------------------------------ own Stage B search
TOL_B, TIE_TOL_B, SWEEPS_B = 1e-12, 1e-12, 5
WITNESS_FAMS = ("FINE-TASK", "LOCAL", "SEQ-12", "SEQ-21")
JOINT_START_ORDER = ("JOINT-GREEDY",) + WITNESS_FAMS


class Wt:
    """Objective weights (wD1, wD2, wI1, wI2, w12)."""

    def __init__(self, wD1=0.0, wD2=0.0, wI1=0.0, wI2=0.0, w12=0.0):
        self.v = {"wD1": wD1, "wD2": wD2, "wI1": wI1, "wI2": wI2, "w12": w12}

    def D(self, r):
        return self.v[f"wD{r}"]

    def I(self, r):
        return self.v[f"wI{r}"]

    @property
    def J(self):
        return self.v["w12"]


def wt_task(r):
    return Wt(**{f"wD{r}": 1.0})


def wt_local(lam, r):
    return Wt(**{f"wD{r}": 1.0, f"wI{r}": lam / 2})


def wt_joint(lam):
    return Wt(1.0, 1.0, lam / 2, lam / 2, lam)


def wt_old_stage1(lam, r):
    return Wt(**{f"wD{r}": 1.0, f"wI{r}": 1.5 * lam})


class MyState:
    """Own coarse-state engine over two fine partitions: persistent coarse labels (a label is a member fine index;
    merges keep the lower label; moves keep labels), per-label n / S (members accumulated in increasing fine index) /
    cost h = -S . log smooth(S / n, c), label SEX columns and the label pair table, all exact integer counts."""

    def __init__(self, fine1: Fine, fine2: Fine, Tf, lab1, lab2):
        self.fine = {1: fine1, 2: fine2}
        self.Tf = np.asarray(Tf, dtype=np.int64)
        self.N = int(self.Tf.sum())
        x = np.arange(self.N + 1, dtype=np.float64)
        self.xl = x * np.log(np.where(x > 0, x, 1.0))
        self.cf = {1: self.Tf.sum(2), 2: self.Tf.sum(1)}
        ns = self.Tf.sum((1, 2))
        self.const = float(self.xl[self.N] - self.xl[ns].sum())
        self.Asum = {r: float(np.sum(self.fine[r].A)) for r in (1, 2)}
        self.lab, self.mem, self.n, self.S, self.h = {}, {}, {}, {}, {}
        for r, lab in ((1, lab1), (2, lab2)):
            self._set(r, lab)
        self._tables()

    def _set(self, r, lab):
        fine = self.fine[r]
        self.lab[r] = np.asarray(lab, dtype=np.int64).copy()
        mem = {}
        for f, l_ in enumerate(self.lab[r].tolist()):
            mem.setdefault(l_, []).append(f)
        for l_, m_ in mem.items():
            assert l_ in m_ and len(set(fine.cls[m_].tolist())) == 1
        self.mem[r] = mem
        self.n[r] = np.zeros(fine.F, dtype=np.int64)
        self.S[r] = np.zeros((fine.F, fine.K))
        self.h[r] = np.zeros(fine.F)
        for l_ in mem:
            self._stat(r, l_)

    def _hc(self, S, n, c):
        S = np.atleast_2d(S)
        n = np.atleast_1d(n)
        out = np.zeros(S.shape[0])
        u = n > 0
        if u.any():
            q = smooth(S[u] / n[u, None], c)
            out[u] = -np.sum(S[u] * np.log(q), 1)
        return out

    def _stat(self, r, l_):
        fine = self.fine[r]
        S = np.zeros(fine.K)
        n = 0
        for f in self.mem[r][l_]:
            S = S + fine.S[f]
            n += int(fine.n[f])
        self.S[r][l_], self.n[r][l_] = S, n
        self.h[r][l_] = self._hc(S, n, int(fine.cls[l_]))[0]

    def _tables(self):
        F1, F2 = self.fine[1].F, self.fine[2].F
        A = np.zeros((2, F1, F2), dtype=np.int64)
        for s_ in (0, 1):
            np.add.at(A[s_], self.lab[1], self.Tf[s_])
        P = np.zeros((2, F1, F2), dtype=np.int64)
        for s_ in (0, 1):
            np.add.at(P[s_].T, self.lab[2], A[s_].T)
        self.P = P
        self.col = {1: P.sum(2), 2: P.sum(1)}

    def other(self, r):
        return 2 if r == 1 else 1

    def Po(self, r):
        return self.P if r == 1 else self.P.transpose(0, 2, 1)

    def phi(self, X):
        return self.xl[X[0]] + self.xl[X[1]] - self.xl[X[0] + X[1]]

    def alive(self, r):
        return np.array(sorted(self.mem[r]), dtype=np.int64)

    def labels_of_class(self, r, c):
        cls = self.fine[r].cls
        return np.array(sorted(l_ for l_ in self.mem[r] if cls[l_] == c), dtype=np.int64)

    def canonical(self, r):
        out = np.empty(self.fine[r].F, dtype=np.int64)
        for m_ in self.mem[r].values():
            out[m_] = min(m_)
        return out

    def terms(self):
        t = {}
        for r in (1, 2):
            al = self.alive(r)
            t[f"D{r}"] = float((self.Asum[r] + float(np.sum(self.h[r][al]))) / self.N)
            t[f"I{r}"] = float((self.phi(self.col[r][:, al]).sum() + self.const) / self.N)
        t["I12"] = float((self.phi(self.P).sum() + self.const) / self.N)
        return t

    def value(self, W: Wt):
        t = self.terms()
        return W.D(1) * t["D1"] + W.D(2) * t["D2"] + W.I(1) * t["I1"] + W.I(2) * t["I2"] + W.J * t["I12"]

    def merge_cands(self, r, c, W: Wt):
        L = self.labels_of_class(r, c)
        ia, ib = np.triu_indices(L.size, 1)
        a, b = L[ia], L[ib]
        N = self.N
        dD = (self._hc(self.S[r][a] + self.S[r][b], self.n[r][a] + self.n[r][b], c) - self.h[r][a] - self.h[r][b]) / N
        C = self.col[r][:, L]
        pc = self.phi(C)
        dI = (self.phi(C[:, ia] + C[:, ib]) - pc[ia] - pc[ib]) / N
        delta = W.D(r) * dD + W.I(r) * dI
        dJ = None
        if W.J != 0.0:
            X = self.Po(r)[:, L][:, :, self.alive(self.other(r))]
            rp = self.phi(X).sum(-1)
            dJ = (self.phi(X[:, ia] + X[:, ib]).sum(-1) - rp[ia] - rp[ib]) / N
            delta = delta + W.J * dJ
        return L, ia, ib, delta, dD, dI, dJ

    def merge(self, r, a, b):
        cls = self.fine[r].cls
        assert a < b and cls[a] == cls[b]
        m_ = sorted(self.mem[r][a] + self.mem[r].pop(b))
        self.mem[r][a] = m_
        self.lab[r][m_] = a
        self._stat(r, a)
        self.n[r][b], self.S[r][b], self.h[r][b] = 0, 0.0, 0.0
        if r == 1:
            self.P[:, a, :] += self.P[:, b, :]
            self.P[:, b, :] = 0
        else:
            self.P[:, :, a] += self.P[:, :, b]
            self.P[:, :, b] = 0
        self.col[r][:, a] += self.col[r][:, b]
        self.col[r][:, b] = 0

    def G(self, r):
        o = self.other(r)
        Tf = self.Tf if r == 1 else self.Tf.transpose(0, 2, 1)
        out = np.zeros(Tf.shape, dtype=np.int64)
        for s_ in (0, 1):
            np.add.at(out[s_].T, self.lab[o], Tf[s_].T)
        return out

    def move_cands(self, r, f, W: Wt, G):
        a = int(self.lab[r][f])
        m_ = self.mem[r][a]
        if len(m_) <= 1:
            return None
        fine = self.fine[r]
        c = int(fine.cls[f])
        B = np.array([l_ for l_ in self.labels_of_class(r, c).tolist() if l_ != a], dtype=np.int64)
        if B.size == 0:
            return None
        N = self.N
        Sa = np.zeros(fine.K)
        for g_ in m_:
            if g_ != f:
                Sa = Sa + fine.S[g_]
        ha = self._hc(Sa, int(self.n[r][a] - fine.n[f]), c)[0]
        hB = self._hc(self.S[r][B] + fine.S[f], self.n[r][B] + fine.n[f], c)
        dD = (ha + hB - self.h[r][a] - self.h[r][B]) / N
        C = self.col[r]
        cf = self.cf[r][:, f]
        dI = (self.phi((C[:, a] - cf)[:, None]) + self.phi(C[:, B] + cf[:, None]) - self.phi(C[:, [a]]) -
              self.phi(C[:, B])) / N
        delta = W.D(r) * dD + W.I(r) * dI
        dJ = None
        if W.J != 0.0:
            ao = self.alive(self.other(r))
            gv = G[:, f, ao]
            X = self.Po(r)
            Xa, XB = X[:, a, ao], X[:, B][:, :, ao]
            dJ = (self.phi(Xa - gv).sum() + self.phi(XB + gv[:, None, :]).sum(-1) - self.phi(Xa).sum() -
                  self.phi(XB).sum(-1)) / N
            delta = delta + W.J * dJ
        return B, delta, dD, dI, dJ

    def move(self, r, f, b, G):
        a = int(self.lab[r][f])
        self.mem[r][a] = [g_ for g_ in self.mem[r][a] if g_ != f]
        self.mem[r][b] = sorted(self.mem[r][b] + [f])
        self.lab[r][f] = b
        self._stat(r, a)
        self._stat(r, b)
        cf = self.cf[r][:, f]
        self.col[r][:, a] -= cf
        self.col[r][:, b] += cf
        gv = G[:, f, :]
        if r == 1:
            self.P[:, a, :] -= gv
            self.P[:, b, :] += gv
        else:
            self.P[:, :, a] -= gv
            self.P[:, :, b] += gv


def my_merge_loop(st: MyState, recips, W: Wt, caps, log, improving, sweep=None):
    """Registered merge loop: to the caps (positive increments allowed) or objective-improving (< -TOL); smallest
    increment, ties within TIE_TOL to the first in (recipient, class, a, b) order. Own per-(recipient, class) cache with
    exact invalidation (own class; every class of the other recipient when the I12 weight is nonzero)."""
    cache, steps = {}, 0
    while True:
        keys = [(r, c) for r in recips for c in range(st.fine[r].K)
                if (st.labels_of_class(r, c).size >= 2 if improving else st.labels_of_class(r, c).size > caps[r])]
        if not keys:
            return steps
        for k_ in keys:
            if k_ not in cache:
                cache[k_] = st.merge_cands(k_[0], k_[1], W)
        gmin = min(float(cache[k_][3].min()) for k_ in keys)
        if improving and not gmin < -TOL_B:
            return steps
        tied = sum(int(np.sum(cache[k_][3] <= gmin + TIE_TOL_B)) for k_ in keys)
        for k_ in keys:
            L, ia, ib, delta, dD, dI, dJ = cache[k_]
            ok = delta <= gmin + TIE_TOL_B
            if improving:
                ok &= delta < -TOL_B
            hit = np.flatnonzero(ok)
            if hit.size:
                j = int(hit[0])
                r, c = k_
                rec = {"kind": "extra" if improving else "to_cap", "recipient": r, "class": c, "a": int(L[ia[j]]),
                       "b": int(L[ib[j]]), "increment": float(delta[j]), "dD": float(dD[j]), "dI_own": float(dI[j]),
                       "dI12": None if dJ is None else float(dJ[j]), "tied_candidates": tied}
                if sweep is not None:
                    rec["sweep"] = sweep
                log.append(rec)
                st.merge(r, rec["a"], rec["b"])
                steps += 1
                cache.pop(k_, None)
                if W.J != 0.0:
                    for kk in [kk for kk in cache if kk[0] == st.other(r)]:
                        cache.pop(kk)
                break


def my_refine(st: MyState, recips, W: Wt, log, sweeps=SWEEPS_B):
    for sweep in range(1, sweeps + 1):
        moved = 0
        for r in recips:
            G = st.G(r)
            for f in range(st.fine[r].F):
                out = st.move_cands(r, f, W, G)
                if out is None:
                    continue
                B, delta, dD, dI, dJ = out
                j = int(np.argmin(delta))
                if delta[j] < -TOL_B:
                    a = int(st.lab[r][f])
                    st.move(r, f, int(B[j]), G)
                    moved += 1
                    log.append({"kind": "move", "sweep": sweep, "recipient": r, "fine_cell": f, "from": a,
                                "to": int(B[j]), "delta": float(delta[j]), "dD": float(dD[j]), "dI_own": float(dI[j]),
                                "dI12": None if dJ is None else float(dJ[j])})
        merged = my_merge_loop(st, recips, W, None, log, True, sweep=sweep)
        if moved == 0 and merged == 0:
            return sweep, True
    return sweeps, False


def F3(t, lam):
    out = {"F_task": t["D1"] + t["D2"]}
    if lam is not None:
        out["F_local"] = t["D1"] + t["D2"] + lam * (t["I1"] + t["I2"]) / 2
        out["F_joint"] = t["D1"] + t["D2"] + lam * ((t["I1"] + t["I2"]) / 2 + t["I12"])
    return out


def my_optimise(st, recips, W, caps, lam, name):
    merges = []
    a = my_merge_loop(st, recips, W, caps, merges, False)
    b = my_merge_loop(st, recips, W, caps, merges, True)
    tg = st.terms()
    g_val = st.value(W)
    moves = []
    sw, cv = my_refine(st, recips, W, moves)
    tr = st.terms()
    return {"stage": name, "merges_to_cap": a, "extra_merges_greedy": b, "merges": merges, "moves": moves,
            "sweeps": sw, "converged": cv, "after_greedy": {**tg, **F3(tg, lam)}, "stage_objective_after_greedy": g_val,
            "stage_objective_after_refine": st.value(W), "after_refine": {**tr, **F3(tr, lam)}}


def class_lab(fine: Fine):
    first = {}
    for f in range(fine.F):
        first.setdefault(int(fine.cls[f]), f)
    return np.array([first[int(c)] for c in fine.cls], dtype=np.int64)


def lab_from_tokens(tok):
    first = {}
    for f, t in enumerate(np.asarray(tok).tolist()):
        first.setdefault(t, f)
    return np.array([first[t] for t in np.asarray(tok).tolist()], dtype=np.int64)


def tokens_from_lab(lab):
    seen, out = {}, np.empty(len(lab), dtype=np.int64)
    for f, l_ in enumerate(np.asarray(lab).tolist()):
        out[f] = seen.setdefault(l_, len(seen))
    return out


def my_family(fam, fines, Tf, m1, m2, lam, witness_labels=None):
    """Own deterministic family run (METHOD_CARD 6.4). Returns (final canonical labels {1, 2}, log)."""
    f1, f2 = fines[1], fines[2]
    ident = {r: np.arange(fines[r].F, dtype=np.int64) for r in (1, 2)}
    caps = {1: m1, 2: m2}
    if fam == "CLASS":
        st = MyState(f1, f2, Tf, class_lab(f1), class_lab(f2))
        return {r: st.canonical(r) for r in (1, 2)}, {"stages": []}, st
    if fam in ("FINE-TASK", "LOCAL"):
        st = MyState(f1, f2, Tf, ident[1], ident[2])
        Ws = {r: (wt_task(r) if fam == "FINE-TASK" else wt_local(lam, r)) for r in (1, 2)}
        stages = []
        for r in (1, 2):
            merges = []
            a = my_merge_loop(st, (r,), Ws[r], caps, merges, False)
            b = my_merge_loop(st, (r,), Ws[r], caps, merges, True)
            stages.append({"stage": f"r{r}", "merges_to_cap": a, "extra_merges_greedy": b, "merges": merges,
                           "stage_objective_after_greedy": st.value(Ws[r])})
        tg = st.terms()
        after_greedy = {**tg, **F3(tg, lam)}
        for r, s_ in zip((1, 2), stages):
            moves = []
            sw, cv = my_refine(st, (r,), Ws[r], moves)
            s_.update({"moves": moves, "sweeps": sw, "converged": cv, "stage_objective_after_refine": st.value(Ws[r])})
        return {r: st.canonical(r) for r in (1, 2)}, {"stages": stages, "after_greedy": after_greedy}, st
    if fam in ("SEQ-12", "SEQ-21"):
        a, b = (1, 2) if fam == "SEQ-12" else (2, 1)
        W = wt_joint(lam)
        lab = {a: ident[a], b: class_lab(fines[b])}
        st1 = MyState(f1, f2, Tf, lab[1], lab[2])
        s1 = my_optimise(st1, (a,), W, caps, lam, f"stage1_r{a}_vs_class_r{b}")
        frozen = st1.canonical(a)
        t1 = st1.terms()
        corr = {"stage1_F_joint_with_class_counterpart": F3(t1, lam)["F_joint"], f"D{a}": t1[f"D{a}"],
                f"I{a}": t1[f"I{a}"], f"D{b}_class": t1[f"D{b}"], f"I{b}_class": t1[f"I{b}"],
                "I12_with_class_counterpart": t1["I12"], "conditional_I_S_decision_given_code": t1["I12"] - t1[f"I{a}"],
                "old_surrogate_D_plus_1p5_lam_I_at_this_map": t1[f"D{a}"] + 1.5 * lam * t1[f"I{a}"]}
        so = MyState(f1, f2, Tf, *([ident[1], class_lab(f2)] if a == 1 else [class_lab(f1), ident[2]]))
        my_optimise(so, (a,), wt_old_stage1(lam, a), caps, lam, "old_stage1_diagnostic")
        to = so.terms()
        corr["old_rule_stage1"] = {"same_map_as_corrected": bool(np.array_equal(so.canonical(a), frozen)),
                                   "F_joint_with_class_counterpart": F3(to, lam)["F_joint"], f"D{a}": to[f"D{a}"],
                                   f"I{a}": to[f"I{a}"], "I12_with_class_counterpart": to["I12"],
                                   "corrected_minus_old_F_joint": corr["stage1_F_joint_with_class_counterpart"] -
                                   F3(to, lam)["F_joint"]}
        lab2 = {a: frozen, b: ident[b]}
        st = MyState(f1, f2, Tf, lab2[1], lab2[2])
        s2 = my_optimise(st, (b,), W, caps, lam, f"stage2_r{b}")
        corr["first_map_never_revised"] = bool(np.array_equal(st.canonical(a), frozen))
        return {r: st.canonical(r) for r in (1, 2)}, {"stages": [s1, s2], "baseline_correction": corr,
                                                      "after_greedy": s2["after_greedy"]}, st
    if fam == "JOINT":
        W = wt_joint(lam)
        starts, unchanged = {}, {}
        st = MyState(f1, f2, Tf, ident[1], ident[2])
        rec = my_optimise(st, (1, 2), W, caps, lam, "JOINT-GREEDY")
        rec["initial_F_joint"] = rec["after_greedy"]["F_joint"]
        starts["JOINT-GREEDY"] = (st, rec)
        for wf in WITNESS_FAMS:
            l1, l2 = witness_labels[wf]
            ws = MyState(f1, f2, Tf, l1, l2)
            ut = ws.terms()
            unchanged[wf] = {"F_joint": F3(ut, lam)["F_joint"], "labels": (ws.canonical(1), ws.canonical(2))}
            moves = []
            sw, cv = my_refine(ws, (1, 2), W, moves)
            starts[wf] = (ws, {"stage": f"refine_{wf}", "initial_F_joint": unchanged[wf]["F_joint"], "moves": moves,
                               "sweeps": sw, "converged": cv})

        def canon_F(labs):
            return F3(MyState(f1, f2, Tf, labs[0], labs[1]).terms(), lam)["F_joint"]

        cand = [(n, "refined", canon_F((starts[n][0].canonical(1), starts[n][0].canonical(2))),
                 (starts[n][0].canonical(1), starts[n][0].canonical(2)), starts[n][0].value(W) if False else
                 F3(starts[n][0].terms(), lam)["F_joint"]) for n in JOINT_START_ORDER]
        cand += [(n, "unchanged", canon_F(unchanged[n]["labels"]), unchanged[n]["labels"], unchanged[n]["F_joint"])
                 for n in WITNESS_FAMS]
        vals = [x[2] for x in cand]
        kb = vals.index(min(vals))
        win_lab = cand[kb][3]
        near = [j for j, v in enumerate(vals) if v <= vals[kb] + 1e-15]
        same_near = all(np.array_equal(cand[j][3][0], win_lab[0]) and np.array_equal(cand[j][3][1], win_lab[1])
                        for j in near)
        for n in JOINT_START_ORDER:
            ss, rr = starts[n]
            rr["refined_F_joint"] = F3(ss.terms(), lam)["F_joint"]
            rr["same_map_as_winner"] = bool(np.array_equal(ss.canonical(1), win_lab[0]) and
                                            np.array_equal(ss.canonical(2), win_lab[1]))
        best = vals[kb]
        unresolved = sorted(n for n in JOINT_START_ORDER
                            if not starts[n][1]["same_map_as_winner"] and starts[n][1]["refined_F_joint"] - best > 1e-12)
        log = {"starts": {n: starts[n][1] for n in JOINT_START_ORDER},
               "candidates": [(x[0], x[1], x[4]) for x in cand], "winner": (cand[kb][0], cand[kb][1]),
               "near_tied_candidates": [f"{cand[j][0]}:{cand[j][1]}" for j in near],
               "near_tied_share_map": bool(same_near), "unresolved_local_optima": unresolved,
               "starts_not_converged": [n for n in JOINT_START_ORDER if not starts[n][1]["converged"]],
               "witness_F_joint": {n: unchanged[n]["F_joint"] for n in WITNESS_FAMS}, "best_canonical_F_joint": best}
        final = MyState(f1, f2, Tf, win_lab[0], win_lab[1])
        return {1: win_lab[0], 2: win_lab[1]}, log, final
    raise ValueError(fam)


# ------------------------------------------------------------------------------------------------ own utility / gate
def fitting_constant(y_fit, K):
    """Fitting-prior constant: majority class of the DEFENSE_FIT labels (first index on ties)."""
    return int(np.argmax(np.bincount(np.asarray(y_fit, np.int64), minlength=K)))


def my_utility(P, hard, y, K, const):
    P = np.asarray(P, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    hard = np.asarray(hard, dtype=np.int64)
    ll = -np.log(np.clip(P[np.arange(len(y)), y], LL_CLIP, 1.0))
    Y = np.zeros_like(P)
    Y[np.arange(len(y)), y] = 1.0
    br = np.sum((P - Y) ** 2, 1)
    return {"acc": float(np.mean(hard == y)), "logloss": float(np.mean(ll)), "brier": float(np.mean(br)),
            "const_acc": float(np.mean(y == const)), "ll_rows": ll, "br_rows": br}


def gate_task(u, uU):
    """One task on one seed: the unchanged eligibility contract (direct comparisons) + margins + headroom."""
    c = u["const_acc"]
    checks = {"acc": u["acc"] >= uU["acc"] - GATE["acc_drop"],
              "logloss": u["logloss"] <= uU["logloss"] + GATE["ll_excess"],
              "brier": u["brier"] <= uU["brier"] + GATE["brier_excess"],
              "retain": (u["acc"] - c) >= GATE["retain"] * (uU["acc"] - c),
              "gain": (u["acc"] - c) >= GATE["gain"]}
    head = {"logloss": u["logloss"] <= uU["logloss"] + HEADROOM["ll_excess"],
            "brier": u["brier"] <= uU["brier"] + HEADROOM["brier_excess"]}
    excess = {"logloss": u["logloss"] - uU["logloss"], "brier": u["brier"] - uU["brier"]}
    return {"checks": checks, "headroom": head, "excess": excess, "eligible": all(checks.values()),
            "headroom_ok": all(head.values()),
            "normalized_excess": max(excess["logloss"] / GATE["ll_excess"], excess["brier"] / GATE["brier_excess"])}


def gate_config(per_seed):
    """per_seed[k] = {"tasks": {0: gate_task, 1: gate_task}, "decisions_preserved": bool, "states": float,
    "ll": {0: .., 1: ..}}: eligible iff every task on every seed passes and decisions are preserved exactly."""
    elig = all(all(per_seed[k]["tasks"][i]["eligible"] for i in (0, 1)) and per_seed[k]["decisions_preserved"]
               for k in per_seed)
    head = elig and all(per_seed[k]["tasks"][i]["headroom_ok"] for k in per_seed for i in (0, 1))
    return {"eligible": bool(elig), "headroom_all_seeds": bool(head),
            "mean_states": float(np.mean([per_seed[k]["states"] for k in per_seed])),
            "worst_occ_ll": max(per_seed[k]["ll"][1] for k in per_seed),
            "worst_income_ll": max(per_seed[k]["ll"][0] for k in per_seed),
            "mean_income_ll": float(np.mean([per_seed[k]["ll"][0] for k in per_seed])),
            "worst_normalized_excess": max(per_seed[k]["tasks"][i]["normalized_excess"] for k in per_seed for i in (0, 1))}


def select_rates(cfg_rows, income_ll_key="worst_income_ll"):
    """Registered rate selection (prompt A4): eligible configurations only; headroom configurations first; order by
    fewer mean actual states, lower worst-seed occupation log loss, lower income log loss, configuration ID; at most
    two. income_ll_key names the income statistic (to be reconciled with the STAGE_A_LOCK rule text)."""
    el = {c: r for c, r in cfg_rows.items() if r["eligible"]}
    key = (lambda c: (el[c]["mean_states"], el[c]["worst_occ_ll"], el[c][income_ll_key], c))
    head = sorted([c for c in el if el[c]["headroom_all_seeds"]], key=key)
    rest = sorted([c for c in el if not el[c]["headroom_all_seeds"]], key=key)
    chosen = (head + rest)[:MAX_SELECTED_RATES]
    return {"decision": "CAPACITY_GATE_PASSED" if chosen else "CAPACITY_GATE_NOT_MET", "selected": chosen,
            "eligible": sorted(el), "headroom": head}


def q_star(cfg_rows):
    """Q*: eligible DIRECT-TASK configuration with the smallest worst-seed normalized excess, then fewer states, id."""
    el = [c for c, r in cfg_rows.items() if r["eligible"]]
    if not el:
        return None
    return min(el, key=lambda c: (cfg_rows[c]["worst_normalized_excess"], cfg_rows[c]["mean_states"], c))


# ------------------------------------------------------------------------------------------------ own inference
def z_primary():
    return NormalDist().inv_cdf(1 - 0.05 / (2 * N_ENDPOINTS))


def my_auc(y, s):
    """Mann-Whitney AUC of score s for y == 1 (average ranks for ties); fixed orientation, never flipped."""
    y = np.asarray(y) == 1
    r = rankdata(np.asarray(s, dtype=np.float64))
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


class GroupBoot:
    """Paired exact-record-group multinomial bootstrap: groups = sorted unique assessment group ids;
    rng = default_rng(seed); replicate b = rng.multinomial(G, uniform(G)); row weight = its group's count. The same
    draw sequence serves every arm and seed (convention to be reconciled with the qpc infer rule text)."""

    def __init__(self, groups, B=B_BOOT, seed=BOOT_SEED):
        self.u, self.inv = np.unique(np.asarray(groups), return_inverse=True)
        G = len(self.u)
        rng = np.random.default_rng(seed)
        p = np.full(G, 1.0 / G)
        self.counts = np.empty((B, G), dtype=np.int64)
        for b in range(B):
            self.counts[b] = rng.multinomial(G, p)
        self.B, self.G = B, G

    def row_weights(self, b):
        return self.counts[b][self.inv].astype(np.float64)


def weighted_auc(score, pos, w):
    """Weighted Mann-Whitney AUC (ties count 1/2) for nonnegative row weights."""
    score = np.asarray(score, dtype=np.float64)
    pos = np.asarray(pos, dtype=bool)
    w = np.asarray(w, dtype=np.float64)
    order = np.argsort(score, kind="stable")
    ss, pp, ww = score[order], pos[order], w[order]
    starts = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])
    gp = np.add.reduceat(np.where(pp, ww, 0.0), starts)
    gn = np.add.reduceat(np.where(pp, 0.0, ww), starts)
    below = np.cumsum(gn) - gn
    den = gp.sum() * gn.sum()
    return float((gp * (below + 0.5 * gn)).sum() / den) if den > 0 else float("nan")


# ------------------------------------------------------------------------------------------------ chronology helpers
def remote_reflog():
    out = git("reflog", "show", "--date=iso-strict", "--format=%H|%gd|%gs", f"refs/remotes/origin/{BRANCH}") or ""
    ent = []
    for line in out.splitlines():
        h, gd, gs = line.split("|", 2)
        m = re.search(r"@\{(.+)\}", gd)
        if m:
            ent.append((parse_iso(m.group(1)), h, gs))
    return sorted(ent)


def first_remote(commit, entries):
    for t, h, gs in entries:
        if h == commit or git_ok("merge-base", "--is-ancestor", commit, h):
            return t
    return None


def jsonl(p):
    ev = []
    if Path(p).exists():
        for line in Path(p).read_text().splitlines():
            try:
                ev.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return ev


LOCK_ORDER = ["SOURCE_ADMISSION_LOCK", "STAGE_A_LOCK", "STAGE_B_LOCK", "AUDIT_AND_SELECTION_LOCK", "EVALUATION_LOCK"]
STAGE_LOCK = {"admit": "SOURCE_ADMISSION_LOCK", "stagea": "STAGE_A_LOCK", "gate": "STAGE_A_LOCK",
              "partition": "STAGE_B_LOCK", "fit": "STAGE_B_LOCK", "inner": "AUDIT_AND_SELECTION_LOCK",
              "inner_src": "AUDIT_AND_SELECTION_LOCK",
              "controls": "AUDIT_AND_SELECTION_LOCK", "select": "AUDIT_AND_SELECTION_LOCK",
              "assess": "EVALUATION_LOCK", "outer": "EVALUATION_LOCK", "infer": "EVALUATION_LOCK"}


def lock_files():
    out = {}
    for name in LOCK_ORDER + sorted(p.stem for p in RES.glob("AMENDMENT*.json")):
        p = RES / f"{name}.json"
        log = git("log", "--format=%H|%cI", "--", f"{REL_RES}/{name}.json") or ""
        out[name] = {"exists": p.exists(), "commits": [l_.split("|") for l_ in log.splitlines() if l_]}
    return out


def evaluation_lock_gate(fetch=True):
    """Assessment labels are read only after EVALUATION_LOCK.json is verified committed and on origin: git fetch, then
    origin's blob (git show origin/<branch>:<lock>) must equal the working-tree file, the lock commit must be an
    ancestor of the ls-remote head, and the lock must have one version. The record of this check is kept."""
    rel = f"{REL_RES}/EVALUATION_LOCK.json"
    p = RES / "EVALUATION_LOCK.json"
    out = {"checked_at": iso(datetime.now(timezone.utc)), "exists": p.exists()}
    if fetch:
        out["git_fetch_ok"] = git_ok("fetch", "origin", BRANCH)
    if not p.exists():
        return {**out, "ok": False}
    log = git("log", "--format=%H|%cI", "--", rel) or ""
    commits = [l_.split("|") for l_ in log.splitlines() if l_]
    first = commits[-1] if commits else None
    ls = git("ls-remote", "origin", f"refs/heads/{BRANCH}", timeout=60)
    remote = ls.split()[0] if ls else None
    shown = git_show_bytes(f"origin/{BRANCH}", rel)
    out.update({"commits": len(commits), "commit": first[0] if first else None,
                "commit_time": first[1] if first else None,
                "first_push_time": first_remote(first[0], remote_reflog()) if first else None,
                "remote_head": remote,
                "origin_show_equals_worktree": shown is not None and shown == p.read_bytes(),
                "commit_on_origin": bool(first and remote and git_ok("merge-base", "--is-ancestor", first[0], remote)),
                "sha256": sha_file(p)})
    out["ok"] = bool(first and out["origin_show_equals_worktree"] and out["commit_on_origin"] and out["first_push_time"]
                     and len(commits) == 1)
    return out


# ------------------------------------------------------------------------------------------------ PHASE 1 checks
def phase1_pending(name, why="PHASE 1 (after the lead reports Stage A complete)"):
    return res("PENDING", reason=why, check=name)


QADM = PRIV / "admitted"


def unit_dir(name):
    return UNITS / name


def unit_done(name) -> bool:
    return (UNITS / name / "COMPLETE.json").exists()


def complete_ok(d: Path):
    """COMPLETE.json id equals the directory, every listed file re-hashes, nothing unlisted."""
    c = jload(d / "COMPLETE.json")
    files = c.get("files", {})
    rehash = {f_: (d / f_).exists() and sha_file(d / f_) == h for f_, h in files.items()}
    present = {str(q.relative_to(d)) for q in d.rglob("*") if q.is_file()} - {"COMPLETE.json"}
    return {"id_ok": c.get("id") == d.name, "rehash_ok": all(rehash.values()) and bool(files),
            "unlisted": sorted(present - set(files))}, files


def pinned_osf_manifest():
    return json.loads(git_show_bytes(TEACHER_PIN, f"{OSF_REL}/MODEL_MANIFEST.json") or b"{}")


def check_teachers(D: Data, L, refit=True):
    """Own forward application of every admitted teacher (qpc admitted copy) on own X vs the qpc tea__ units, the dpc
    tea__ units and the admitted release; custody of the copy vs the pinned osf MODEL_MANIFEST and the dpc admitted
    original; deployed-head structure and scaler moments; optional own head refit (DEFENSE_FIT / HEAD_VALIDATION task
    labels only)."""
    man = pinned_osf_manifest()
    entries = {(u["label"], u["seed"], u["unit"]): u for u in man.get("units", [])}
    out, T = {}, {}
    tr, va = D.idx[FIT], D.idx["HEAD_VALIDATION"]
    for k in SEEDS:
        for t in TEACHERS:
            name = f"rel__s{k}__{t}"
            qd, dd = QADM / name, DPC_ADMITTED / name
            info, fails = {"unit": name}, []
            entry = entries.get((TEACHER_LABEL[t], k, name))
            cq, fq = complete_ok(qd)
            cd_, fd = complete_ok(dd)
            info["custody"] = {"qpc_copy_complete": cq, "dpc_original_complete": cd_,
                               "qpc_copy_files_equal_dpc_original": fq == fd,
                               "qpc_copy_files_equal_pinned_osf_manifest": bool(entry) and
                               fq == entry.get("complete_files_sha256")}
            if not (cq["id_ok"] and cq["rehash_ok"] and not cq["unlisted"] and cd_["rehash_ok"] and fq == fd and
                    info["custody"]["qpc_copy_files_equal_pinned_osf_manifest"]):
                fails.append("custody")
            mine = own_teacher(qd / "model.pt", [qd / "head_0.joblib", qd / "head_1.joblib"], D.X)
            sd = tload(qd / "model.pt")
            exp = {f"enc.{i}.{j}.{w}" for i in (0, 1) for j in (0, 2, 4) for w in ("weight", "bias")} | \
                  {f"head.{i}.{w}" for i in (0, 1) for w in ("weight", "bias")}
            info["state_keys_ok"] = set(sd) == exp and tuple(sd["enc.0.0.weight"].shape) == (64, 83) and \
                all(v.dtype == torch.float32 for v in sd.values())
            if not info["state_keys_ok"]:
                fails.append("state dict")
            qu = UNITS / f"tea__s{k}__{t}"
            cu, _ = complete_ok(qu)
            info["qpc_unit_complete"] = cu
            if not (cu["id_ok"] and cu["rehash_ok"] and not cu["unlisted"]):
                fails.append("qpc teacher unit custody")
            tq = np.load(qu / "teacher.npz", allow_pickle=False)
            td = np.load(DPC_UNITS / f"tea__s{k}__{t}" / "teacher.npz", allow_pickle=False)
            rel = np.load(qd / "release.npz", allow_pickle=False)
            info["teacher_unit_keys"] = sorted(tq.files)
            info["row_order_equals_own"] = all(bool(np.array_equal(z["row_id"], D.row_id)) for z in (tq, td, rel))
            if not info["row_order_equals_own"]:
                fails.append("row order")
            par = {}
            for key in ("r1", "r2", "c1", "c2", "p1", "p2", "d1", "d2"):
                a = mine[key]
                rk = key if key[0] != "d" else f"hard{key[1]}"
                par[key] = {"qpc_unit": bitwise(a.astype(tq[key].dtype), tq[key]) and a.dtype == tq[key].dtype,
                            "dpc_unit": bitwise(a.astype(td[key].dtype), td[key]),
                            "admitted_release": bitwise(a.astype(rel[rk].dtype), rel[rk]),
                            "max_abs_diff_qpc_unit": maxdiff(a, tq[key])}
            info["parity"] = par
            if not all(v["qpc_unit"] and v["dpc_unit"] and v["admitted_release"] for v in par.values()):
                fails.append("forward parity")
            heads = {}
            for i in (0, 1):
                j = i + 1
                head = mine[f"head{j}"]
                hi = {"pipeline_scaler_logreg": isinstance(head, Pipeline) and len(head.steps) == 2 and
                      isinstance(head.steps[0][1], StandardScaler) and isinstance(head.steps[1][1], LogisticRegression),
                      "classes_0_to_K_minus_1": bool(np.array_equal(head[-1].classes_, np.arange(KS[i])))}
                ref = StandardScaler().fit(mine[f"r{j}"][tr])
                hi["scaler_moments_bitwise_OSF_DEFENSE_FIT"] = bool(np.array_equal(head[0].mean_, ref.mean_) and
                                                                    np.array_equal(head[0].var_, ref.var_))
                P = mine[f"p{j}"]
                hi.update({"rows_with_exact_max_ties": int(np.sum((P == P.max(1, keepdims=True)).sum(1) > 1)),
                           "exact_zero_components": int(np.sum(P == 0)), "min_probability": float(P.min()),
                           "finite": bool(np.isfinite(P).all()),
                           "predicted_class_counts_by_role": {r_: np.bincount(mine[f"d{j}"][D.mask[r_]],
                                                                              minlength=KS[i]).tolist() for r_ in ROLES}})
                if refit:
                    y = L["y_income"] if i == 0 else L["y_occ"]
                    assert (y[tr] >= 0).all() and (y[va] >= 0).all()
                    mh, C, table = my_fit_head(mine[f"r{j}"], y, tr, va, KS[i])
                    _, mp, md = head_outputs(mh, mine[f"r{j}"])
                    hi.update({"refit_C": C, "saved_head_C": float(head[-1].C),
                               "refit_p_max_abs_diff": maxdiff(mp, P), "refit_decision_mismatches": int((md != mine[f"d{j}"]).sum())})
                    if C != float(head[-1].C) or hi["refit_p_max_abs_diff"] > 1e-8 or hi["refit_decision_mismatches"]:
                        fails.append(f"recipient {j} head refit")
                if not all(v for v in hi.values() if isinstance(v, bool)):
                    fails.append(f"recipient {j} head")
                heads[f"recipient_{j}"] = hi
            info["heads"] = heads
            info["model_pt_sha256"] = fq.get("model.pt")
            info["failures"] = fails
            T[(k, t)] = {kk: mine[kk] for kk in ("p1", "p2", "d1", "d2", "r1", "r2", "c1", "c2")}
            T[(k, t)]["model_sha"] = fq.get("model.pt")
            out[f"{t}|s{k}"] = res("FAIL" if fails else "PASS", **info)
    st = worst(*[v["status"] for v in out.values()]) if len(out) == 6 else "FAIL"
    return res(st, units=out, note="own forward pass on own X (83 permitted columns) from the qpc admitted copies; "
                                   "head refits read OSF_DEFENSE_FIT and HEAD_VALIDATION task labels only"), T


def leace_apply(npz_path, H):
    """Own LEACE application x - ((x - mu) R^T) L^T from the official map arrays (float64)."""
    z = np.load(npz_path, allow_pickle=False)
    x = np.asarray(H, dtype=np.float64)
    mu, pl, pr = (np.asarray(z[a], dtype=np.float64) for a in ("mean_x", "proj_left", "proj_right"))
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a))  # noqa: E731
    with torch.no_grad():
        return (t(x) - ((t(x) - t(mu)) @ t(pr).T) @ t(pl).T).numpy().astype(np.float64, copy=True)


def tree_leaves(model_json, X):
    """Own traversal of the official FARE tree arrays (x <= thr goes left; float32-cast features as the tree saw them)."""
    m = jload(model_json)
    Xf = np.asarray(X).astype(np.float32).astype(np.float64)
    left, right = np.asarray(m["children_left"]), np.asarray(m["children_right"])
    feat, thr = np.asarray(m["feature"]), np.asarray(m["threshold"], dtype=np.float64)
    node = np.zeros(Xf.shape[0], dtype=np.int64)
    for _ in range(10_000):
        live = np.flatnonzero(left[node] != -1)
        if live.size == 0:
            break
        nd = node[live]
        node[live] = np.where(Xf[live, feat[nd]] <= thr[nd], left[nd], right[nd])
    pos = {int(v): j for j, v in enumerate(m["leaf_node_ids"])}
    return np.array([pos.get(int(v), -1) for v in node], dtype=np.int64)


def check_references(D: Data, T, keep=None):
    """References E (official LEACE on the U features), F / F0 (official FARE trees): own application from the dpc
    admitted artifacts (read-only) and the deployed heads vs the qpc ref__ units and their admitted copies."""
    out = {}
    keep = {} if keep is None else keep
    for k in SEEDS:
        for lab in REFS:
            info, fails = {}, []
            qu = UNITS / f"ref__s{k}__{lab}"
            cu, fu = complete_ok(qu)
            ca, fa_ = complete_ok(QADM / f"ref__s{k}__{lab}")
            info["custody"] = {"qpc_unit": cu, "qpc_admitted_copy": ca, "unit_equals_admitted_copy": fu == fa_}
            if not (cu["rehash_ok"] and ca["rehash_ok"] and fu == fa_ and not cu["unlisted"]):
                fails.append("custody")
            ref = np.load(qu / "reference.npz", allow_pickle=False)
            dref = np.load(DPC_UNITS / f"ref__s{k}__{lab}" / "reference.npz", allow_pickle=False)
            info["qpc_unit_equals_dpc_unit"] = sorted(ref.files) == sorted(dref.files) and \
                all(bitwise(ref[a], dref[a]) for a in ref.files)
            own = {}
            if lab == "E":
                ud = DPC_ADMITTED / f"lc__s{k}__E"
                sd = tload(ud / "model.pt")
                for i in (0, 1):
                    j = i + 1
                    R = leace_apply(ud / f"leace_{i}" / "leace_map.npz", encode(sd, i, D.X))
                    c, p, d = head_outputs(joblib.load(ud / f"head_{i}.joblib"), R)
                    own.update({f"r{j}": R, f"c{j}": c, f"p{j}": p, f"d{j}": d.astype(np.int64)})
                info["model_pt_equals_U_teacher"] = sha_file(ud / "model.pt") == T[(k, "U")]["model_sha"]
                if not info["model_pt_equals_U_teacher"]:
                    fails.append("LEACE base model")
            else:
                tag = "c1" if lab == "F" else "Z1"
                for i in (0, 1):
                    j = i + 1
                    ud = DPC_ADMITTED / f"fare__s{k}__p{i}__{tag}"
                    rec = jload(ud / "record.json")
                    cells = tree_leaves(DPC_ADMITTED / "fare_cache" / rec["provenance"]["admission"]["source"] / "model" /
                                        "model.json", D.X)
                    ncell = int(rec["n_cells"])
                    R = np.eye(ncell)[cells]
                    c, p, d = head_outputs(joblib.load(ud / "head.joblib"), R)
                    own.update({f"r{j}": R, f"c{j}": c, f"p{j}": p, f"d{j}": d.astype(np.int64), f"cells{j}": cells})
                    info[f"recipient_{j}_purpose_ok"] = rec["purpose"] == i and bool((cells >= 0).all())
                    if not info[f"recipient_{j}_purpose_ok"]:
                        fails.append(f"FARE recipient {j}")
            par = {a: bitwise(own[a].astype(ref[a].dtype), ref[a]) for a in own}
            info["own_application_bitwise_vs_qpc_unit"] = par
            info["max_abs_diff_p"] = max(maxdiff(own["p1"], ref["p1"]), maxdiff(own["p2"], ref["p2"]))
            info["row_order_ok"] = bool(np.array_equal(ref["row_id"], D.row_id))
            if not (all(par.values()) and info["qpc_unit_equals_dpc_unit"] and info["row_order_ok"]):
                fails.append("application parity")
            keep[(lab, k)] = own
            info["failures"] = fails
            out[f"{lab}|s{k}"] = res("FAIL" if fails else "PASS", **info)
    return res(worst(*[v["status"] for v in out.values()]) if len(out) == 9 else "FAIL", units=out,
               note="label-free: own LEACE formula on own U features; own traversal of the official FARE trees; "
                    "deployed heads; qpc ref__ units must equal the dpc ref__ units and the qpc admitted copies")


def check_a1_source_replay(D: Data, T):
    """The historical source code, replayed by the verifier: own dpc-rule k-means (source start, 20 rounds, m = 8)
    on each U seed's OSF_DEFENSE_FIT probabilities vs the admitted dpc pol__s{k}__U_DIRECT-TASK_m8 policy (assignment
    partitions bitwise, identity token map) and its release (tokens, decoded vectors, decisions, every row). This is
    the verifier's own parity target for the lead's A1 reproduction; it reads no label."""
    fit = D.fit_idx
    out, keep = {}, {}
    for k in SEEDS:
        tt = T[(k, "U")]
        ud = DPC_UNITS / f"pol__s{k}__U_DIRECT-TASK_m8"
        pj = jload(ud / "policy.json")
        rel = np.load(ud / "release.npz", allow_pickle=False)
        info, fails = {}, []
        for r, (P, d) in ((1, (tt["p1"], tt["d1"])), (2, (tt["p2"], tt["d2"]))):
            fine = fit_partition(P[fit], d[fit], KS[r - 1], 8, starts=("source",), rule="dpc")
            stored = Pol.from_json(pj[f"p{r}"])
            eq = fine.equal(stored.fine)
            pol = direct_policy(fine)
            tok, q, hard, _ = pol.release(P, d)
            cc = {"partition_bitwise": eq, "identity_map": bool(np.array_equal(stored.cell_token, np.arange(stored.fine.F))),
                  "fine_fingerprint_dpc_convention": fine.fingerprint_dpc() == pj[f"p{r}"]["fine"].get("fingerprint"),
                  "policy_fingerprint_dpc_convention": pol.fingerprint_dpc() == pj[f"p{r}"].get("fingerprint"),
                  "tokens_bitwise": bitwise(tok, rel[f"tok{r}"]), "decoded_bitwise": bitwise(q, rel[f"q{r}"]),
                  "decisions_bitwise": bitwise(hard, rel[f"hard{r}"]), "decisions_equal_teacher": bool(np.array_equal(hard, d)),
                  "cells": int(fine.F),
                  "classes_converged": [x.get("starts", [{}])[0].get("converged") for x in fine.receipt["per_class"]
                                        if not x.get("fallback")],
                  "rounds_used": [x.get("starts", [{}])[0].get("rounds_used") for x in fine.receipt["per_class"]
                                  if not x.get("fallback")],
                  "fallback_classes": [int(c) for c in fine.cls[fine.fb]]}
            recs = stored.fine.receipt.get("per_class", [])
            mine_pc = fine.receipt["per_class"]
            rc_ok = len(recs) == len(mine_pc)
            for mm, rr in zip(mine_pc, recs):
                if mm.get("fallback"):
                    rc_ok &= bool(rr.get("fallback"))
                    continue
                s0 = mm["starts"][0]
                rc_ok &= (rr.get("rows") == mm["rows"] and rr.get("distinct_vectors") == mm["distinct_vectors"] and
                          rr.get("initial_cells") == mm["requested_cells"] and
                          rr.get("init_positions") == s0["init"]["positions"] and rr.get("rounds_used") == s0["rounds_used"]
                          and rr.get("converged") == s0["converged"] and rr.get("changed_per_round") == s0["changed_per_round"]
                          and rr.get("cell_counts") == mm["cell_counts"] and rr.get("cells") == mm["cells"])
            cc["per_class_receipts_equal_dpc"] = bool(rc_ok)
            info[f"recipient_{r}"] = cc
            if not (all(eq.values()) and cc["identity_map"] and cc["tokens_bitwise"] and cc["decoded_bitwise"] and
                    cc["decisions_bitwise"] and cc["decisions_equal_teacher"] and cc["fine_fingerprint_dpc_convention"]
                    and cc["policy_fingerprint_dpc_convention"] and rc_ok):
                fails.append(f"recipient {r}")
            keep[(k, r)] = fine
        info["failures"] = fails
        out[f"s{k}"] = res("FAIL" if fails else "PASS", **info)
    return res(worst(*[v["status"] for v in out.values()]), seeds=out), keep


TASKS = ("income", "occupation")
STAGEA_TOL = 1e-12          # relative tolerance for recorded mean distortions vs own (bitwise partitions)
RECEIPT_REL_TOL = 1e-10     # relative tolerance for recorded per-class k-means objectives J (cancellation in A_c + sum h)


def rel_diff(a, b):
    a, b = float(a), float(b)
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


def compare_policy_to_own(stored: Pol, fine: Fine, P, d, rel=None, r=None):
    """Stored policy vs own partition (identity map), and (optionally) its release arrays vs the own release."""
    eq = fine.equal(stored.fine)
    pol = direct_policy(fine)
    out = {"partition_bitwise": all(eq.values()), "partition_fields": eq,
           "identity_map": bool(np.array_equal(stored.cell_token, np.arange(stored.fine.F))),
           "fine_fingerprint_equal": fine.fingerprint_dpc() == getattr(stored.fine, "stored_fp", None),
           "policy_fingerprint_equal": pol.fingerprint_dpc() == stored.stored.get("fingerprint")}
    st = pol.structure()
    sst = stored.structure()
    out["own_structure_ok"] = all(v for v in st.values() if isinstance(v, bool))
    out["stored_structure_ok"] = all(v for v in sst.values() if isinstance(v, bool))
    tok, q, hard, _ = pol.release(P, d)
    if rel is not None:
        out.update({"tokens_bitwise": bitwise(tok, rel[f"tok{r}"]), "decoded_bitwise": bitwise(q, rel[f"q{r}"]),
                    "decisions_bitwise": bitwise(hard, rel[f"hard{r}"]),
                    "alpha_equal": int(rel[f"alpha{r}"]) == pol.T})
    out["decisions_equal_teacher_all_rows"] = bool(np.array_equal(hard, d) and np.array_equal(q.argmax(1), d))
    return out, pol, tok, q, hard


def flags_ok(dct, keys=None):
    return all(v for k_, v in dct.items() if isinstance(v, bool) and (keys is None or k_ in keys))


def compare_start_receipts(own_pc, rec_pc):
    """Own per-class start results vs the recorded per-class receipts (qpc kmeans receipt layout)."""
    out = {"classes": 0, "starts": 0, "winner_equal": True, "stop_equal": True, "converged_equal": True,
           "returned_pass_equal": True, "objective_max_rel_diff": 0.0, "trajectory_max_rel_diff": 0.0,
           "trajectory_length_equal": True, "init_equal": True, "cell_counts_equal": True, "changed_equal": True,
           "rounds_used_minus_own_passes": set(), "missing": []}
    if len(own_pc) != len(rec_pc):
        out["missing"].append("class count")
        return out
    for mm, rr in zip(own_pc, rec_pc):
        if mm.get("fallback"):
            out["winner_equal"] &= bool(rr.get("fallback"))
            continue
        out["classes"] += 1
        out["winner_equal"] &= rr.get("winner") == mm["winner"]
        rs = {x["start"]: x for x in rr.get("starts", [])}
        for x in mm["starts"]:
            y = rs.get(x["start"])
            if y is None:
                out["missing"].append(f"class {mm['class']} start {x['start']}")
                continue
            out["starts"] += 1
            out["stop_equal"] &= y.get("stop_reason") == x["stop"]
            out["converged_equal"] &= y.get("converged") == x["converged"]
            if "returned_pass" in y and "best_pass" in x:
                out["returned_pass_equal"] &= int(y["returned_pass"]) == int(x["best_pass"])
            out["objective_max_rel_diff"] = max(out["objective_max_rel_diff"], rel_diff(y.get("objective", np.nan), x["J"]))
            tr_ = y.get("objective_trajectory")
            if tr_ is not None and "J_trace" in x:
                out["trajectory_length_equal"] &= len(tr_) == len(x["J_trace"])
                if len(tr_) == len(x["J_trace"]):
                    out["trajectory_max_rel_diff"] = max([out["trajectory_max_rel_diff"]] +
                                                         [rel_diff(a_, b_) for a_, b_ in zip(tr_, x["J_trace"])])
            if y.get("changed_per_pass") is not None and x.get("changed_per_pass") is not None:
                out["changed_equal"] &= list(y["changed_per_pass"]) == list(x["changed_per_pass"])
            ii = y.get("init")
            if isinstance(ii, dict):
                out["init_compared"] = out.get("init_compared", 0) + 1
                if x["start"] == "source":
                    out["init_equal"] &= ii.get("positions") == x["init"]["positions"]
                else:
                    out["init_equal"] &= (ii.get("chosen_distinct_index") == x["init"]["chosen_distinct_index"] and
                                          int(ii.get("kl_clipped", -1)) == x["init"]["kl_clipped"] and
                                          int(ii.get("degenerate_draws", -1)) == x["init"]["degenerate_draws"])
                    out["kl_clipped_total"] = out.get("kl_clipped_total", 0) + x["init"]["kl_clipped"]
                    out["degenerate_draws_total"] = out.get("degenerate_draws_total", 0) + x["init"]["degenerate_draws"]
            if y.get("final_cell_counts") is not None:
                out["cell_counts_equal"] &= list(y["final_cell_counts"]) == x["final_cell_counts"]
            if y.get("cell_counts") is not None:
                out["cell_counts_equal"] &= list(y["cell_counts"]) == [v for v in x["final_cell_counts"] if v > 0]
            if y.get("rounds_used") is not None:
                out["rounds_used_minus_own_passes"].add(int(y["rounds_used"]) - int(x.get("assignment_passes",
                                                                                          x.get("rounds_used", 0))))
    out["rounds_used_minus_own_passes"] = sorted(out["rounds_used_minus_own_passes"])
    return out


def receipts_ok(c):
    """Decisive: identical stop / pass / init / count / winner records (the partitions themselves are compared
    bitwise elsewhere). Objective values J = A_c + sum h are compared at RECEIPT_REL_TOL: the class total is a
    difference of large terms, so float summation order alone moves it by a few 1e-12 relative (VERIFIER_CORRECTIONS)."""
    return (not c["missing"] and c["winner_equal"] and c["stop_equal"] and c["converged_equal"] and
            c["returned_pass_equal"] and c["trajectory_length_equal"] and c["init_equal"] and c["cell_counts_equal"]
            and c["changed_equal"] and c["objective_max_rel_diff"] <= RECEIPT_REL_TOL and
            c["trajectory_max_rel_diff"] <= RECEIPT_REL_TOL and len(c["rounds_used_minus_own_passes"]) <= 1)


def token_entropy(tok):
    """Empirical entropy (nats) of the emitted token distribution."""
    cnt = np.bincount(np.asarray(tok, dtype=np.int64))
    pr = cnt[cnt > 0] / cnt.sum()
    return float(-np.sum(pr * np.log(pr)))


def own_fit_distortion(P_fit, q_fit):
    return float(np.mean(kl_paired(P_fit, q_fit)))


def check_stage_a(D: Data, L, T, a1_src):
    """Stage A replay: A1 (src20 = own dpc rule; r200 = own qpc rule, source start), A2 per-recipient three-start fits
    (every start's partition fingerprint, per-class winners and receipts), the 24 mapping-pair releases (all rows),
    class preservation, own INNER_SELECTION utility, the gate and the rate decision. Reads DEFENSE_FIT and
    INNER_SELECTION task labels only (A3)."""
    fit, sel = D.fit_idx, D.idx[INNER]
    const = {1: fitting_constant(L["y_income"][fit], 2), 2: fitting_constant(L["y_occ"][fit], 6)}
    y_sel = {1: L["y_income"][sel], 2: L["y_occ"][sel]}
    assert (y_sel[1] >= 0).all() and (y_sel[2] >= 0).all()
    a1_out, a2_out, pair_out, util, cp_fail, adv_fail = {}, {}, {}, {}, [], []
    own_fits, own_pol = {}, {}
    cp_rows = {r_: 0 for r_ in ROLES}
    advP = {K: adversarial_rows(K) for K in KS}
    t_fit = 0.0
    for k in SEEDS:
        tt = T[(k, "U")]
        P, d = {1: tt["p1"], 2: tt["p2"]}, {1: tt["d1"], 2: tt["d2"]}
        uU = {r: my_utility(P[r][sel], d[r][sel], y_sel[r], KS[r - 1], const[r]) for r in (1, 2)}
        util[("SRC|U", k)] = uU
        # ---------------- A1
        name = f"a1__s{k}"
        info, fails = {}, []
        if not unit_done(name):
            a1_out[f"s{k}"] = res("FAIL", failures=["unit missing"])
        else:
            cu, _ = complete_ok(UNITS / name)
            rec = jload(UNITS / name / "record.json")
            info["complete"] = cu
            if not (cu["id_ok"] and cu["rehash_ok"] and not cu["unlisted"]):
                fails.append("custody")
            info["lead_parity_with_admitted_release_ok"] = bool(rec.get("parity_with_admitted_release", {}).get("ok"))
            info["no_engineering_blocker"] = "ENGINEERING_BLOCKER" not in rec
            if not (info["lead_parity_with_admitted_release_ok"] and info["no_engineering_blocker"]):
                fails.append("lead A1 parity flag")
            t0 = time.process_time()
            r200 = {r: fit_partition(P[r][fit], d[r][fit], KS[r - 1], 8, starts=("source",), rule="qpc") for r in (1, 2)}
            t_fit += time.process_time() - t0
            own_fits[(k, "A1r200")] = r200
            for ver, fines in (("src20", {r: a1_src[(k, r)] for r in (1, 2)}), ("r200", r200)):
                pj = jload(UNITS / name / f"policy_{ver}.json")
                relz = np.load(UNITS / name / f"release_{ver}.npz", allow_pickle=False)
                vi = {"row_order_ok": bool(np.array_equal(relz["row_id"], D.row_id))}
                rv = rec.get(ver, {})
                for r in (1, 2):
                    stored = Pol.from_json(pj[f"p{r}"])
                    cmp_, pol, tok, q, hard = compare_policy_to_own(stored, fines[r], P[r], d[r], relz, r)
                    Dr = own_fit_distortion(P[r][fit], q[fit])
                    cmp_["fit_distortion_own"] = Dr
                    cmp_["fit_distortion_rel_diff"] = rel_diff(Dr, rv.get("fit_distortion", {}).get(f"D{r}", np.nan))
                    pr = (rv.get("per_recipient", {}) or {})
                    pr = pr.get(str(r), pr.get(r, {}))
                    if ver == "r200":
                        rc = compare_start_receipts(fines[r].receipt["per_class"], pr.get("per_class", []))
                        cmp_["receipts"] = rc
                        cmp_["receipts_ok"] = receipts_ok(rc)
                        cmp_["objective_total_rel_diff"] = rel_diff(
                            sum(pc["starts"][0]["J"] for pc in fines[r].receipt["per_class"] if not pc.get("fallback")),
                            pr.get("objective_total", np.nan))
                        cmp_["classes_converged_own"] = [pc["starts"][0]["converged"] for pc in fines[r].receipt["per_class"]
                                                         if not pc.get("fallback")]
                        cmp_["stops_own"] = [pc["starts"][0]["stop"] for pc in fines[r].receipt["per_class"]
                                             if not pc.get("fallback")]
                        cmp_["passes_own"] = [pc["starts"][0]["assignment_passes"] for pc in fines[r].receipt["per_class"]
                                              if not pc.get("fallback")]
                    vi[f"recipient_{r}"] = cmp_
                    bad = not (cmp_["partition_bitwise"] and cmp_["identity_map"] and cmp_["policy_fingerprint_equal"] and
                               cmp_["tokens_bitwise"] and cmp_["decoded_bitwise"] and cmp_["decisions_bitwise"] and
                               cmp_["decisions_equal_teacher_all_rows"] and cmp_["fit_distortion_rel_diff"] <= STAGEA_TOL
                               and cmp_.get("receipts_ok", True) and cmp_.get("objective_total_rel_diff", 0.0) <= STAGEA_TOL)
                    if bad:
                        fails.append(f"{ver} recipient {r}")
                    own_pol[(k, f"A1|{ver}", r)] = (pol, tok, q, hard)
                if ver == "src20":
                    dz = np.load(DPC_UNITS / f"pol__s{k}__U_DIRECT-TASK_m8" / "release.npz", allow_pickle=False)
                    vi["equals_admitted_dpc_release_bitwise"] = all(bitwise(relz[x], dz[x]) for x in dz.files)
                    if not vi["equals_admitted_dpc_release_bitwise"]:
                        fails.append("src20 vs admitted dpc release")
                info[ver] = vi
            info["fit_distortion_contrast_own"] = {
                f"D{r}": {"src20": info["src20"][f"recipient_{r}"]["fit_distortion_own"],
                          "r200": info["r200"][f"recipient_{r}"]["fit_distortion_own"],
                          "r200_minus_src20": info["r200"][f"recipient_{r}"]["fit_distortion_own"] -
                          info["src20"][f"recipient_{r}"]["fit_distortion_own"]} for r in (1, 2)}
            for ver, fines in (("src20", {r: a1_src[(k, r)] for r in (1, 2)}), ("r200", r200)):
                for r in (1, 2):
                    pcs = [pc for pc in fines[r].receipt["per_class"] if not pc.get("fallback")]
                    w_ = [next(x for x in pc["starts"] if x["start"] == pc["winner"]) for pc in pcs]
                    tot_ = sum(x["J"] for x in w_)
                    util.setdefault(("A1sum", ver, k), {})[r] = {
                        "fit_objective_total": tot_, "fit_mean_kl": tot_ / len(fit), "classes_fitted": len(pcs),
                        "classes_converged": sum(bool(x["converged"]) for x in w_),
                        "all_converged": all(bool(x["converged"]) for x in w_),
                        "rounds_used_max": max(int(x.get("assignment_passes", x.get("rounds_used"))) for x in w_),
                        "rounds_used_sum": sum(int(x.get("assignment_passes", x.get("rounds_used"))) for x in w_),
                        "stop_reasons": ";".join(sorted({x["stop"] for x in w_}))}
            for ver in ("src20", "r200"):
                q1, q2 = own_pol[(k, f"A1|{ver}", 1)][2], own_pol[(k, f"A1|{ver}", 2)][2]
                h1, h2 = own_pol[(k, f"A1|{ver}", 1)][3], own_pol[(k, f"A1|{ver}", 2)][3]
                util[(f"A1|{ver}", k)] = {1: my_utility(q1[sel], h1[sel], y_sel[1], 2, const[1]),
                                          2: my_utility(q2[sel], h2[sel], y_sel[2], 6, const[2])}
            info["failures"] = fails
            a1_out[f"s{k}"] = res("FAIL" if fails else "PASS", **info)
        # ---------------- A2 per-recipient fits
        for r, rates in ((1, RATES_INCOME), (2, RATES_OCC)):
            for m in rates:
                name = f"dir__s{k}__r{r}__m{m}"
                info, fails = {}, []
                if not unit_done(name):
                    a2_out[name] = res("FAIL", failures=["unit missing"])
                    continue
                cu, _ = complete_ok(UNITS / name)
                if not (cu["id_ok"] and cu["rehash_ok"] and not cu["unlisted"]):
                    fails.append("custody")
                t0 = time.process_time()
                fine = fit_partition(P[r][fit], d[r][fit], KS[r - 1], m, starts=START_ORDER, rule="qpc")
                t_fit += time.process_time() - t0
                own_fits[(k, r, m)] = fine
                stored = Pol.from_json(jload(UNITS / name / "policy.json"))
                rec = jload(UNITS / name / "record.json")
                rr = rec.get("receipts", {})
                cmp_, pol, tok, q, hard = compare_policy_to_own(stored, fine, P[r], d[r])
                rc = compare_start_receipts(fine.receipt["per_class"], rr.get("per_class", []))
                fps = rr.get("fingerprints", {})
                own_fps = {"winner": fine.fingerprint_dpc(), **{st: fine.receipt["by_start"][st].fingerprint_dpc()
                                                                 for st in START_ORDER}}
                cmp_["start_partition_fingerprints_equal"] = {x: fps.get(x) == own_fps[x] for x in own_fps}
                Dr = own_fit_distortion(P[r][fit], q[fit])
                cmp_.update({"receipts": rc, "receipts_ok": receipts_ok(rc), "fit_distortion_own": Dr,
                             "fit_distortion_rel_diff": rel_diff(Dr, rr.get("fit_distortion", np.nan)),
                             "alphabet_equal": rr.get("alphabet") == pol.T,
                             "tokens_per_class_equal": rr.get("tokens_per_class") == pol.tokens_per_class(),
                             "alphabet": pol.T, "tokens_per_class": pol.tokens_per_class(),
                             "effective_states_per_class": pol.effective_per_class(),
                             "winners": [pc.get("winner") for pc in fine.receipt["per_class"]],
                             "winner_converged": [next(x["converged"] for x in pc["starts"] if x["start"] == pc["winner"])
                                                  for pc in fine.receipt["per_class"] if not pc.get("fallback")],
                             "cells_requested_vs_actual": [(pc.get("requested_cells"), pc["cells"])
                                                           for pc in fine.receipt["per_class"]],
                             "fallback_classes": [int(c) for c in fine.cls[fine.fb]]})
                if not (cmp_["partition_bitwise"] and cmp_["identity_map"] and cmp_["policy_fingerprint_equal"] and
                        cmp_["receipts_ok"] and all(cmp_["start_partition_fingerprints_equal"].values()) and
                        cmp_["fit_distortion_rel_diff"] <= STAGEA_TOL and cmp_["alphabet_equal"] and
                        cmp_["tokens_per_class_equal"] and cmp_["decisions_equal_teacher_all_rows"] and
                        cmp_["own_structure_ok"] and cmp_["stored_structure_ok"]):
                    fails.append("replay mismatch")
                if m == 8 and (k, "A1r200") in own_fits:
                    a1s = own_fits[(k, "A1r200")][r]
                    cmp_["A1_r200_equals_A2_source_start"] = all(a1s.equal(fine.receipt["by_start"]["source"]).values())
                    if not cmp_["A1_r200_equals_A2_source_start"]:
                        fails.append("A1 r200 is not the A2 source-start receipt")
                own_pol[(k, r, m)] = (pol, tok, q, hard)
                cmp_.pop("partition_fields", None)
                cmp_["failures"] = fails
                a2_out[name] = res("FAIL" if fails else "PASS", **cmp_)
        # ---------------- mapping pairs
        for m1, m2 in rate_bank():
            cid = direct_id(m1, m2)
            name = f"pol__s{k}__{cid.replace('|', '_')}"
            info, fails = {}, []
            if not unit_done(name) or (k, 1, m1) not in own_pol or (k, 2, m2) not in own_pol:
                pair_out[f"{cid}#s{k}"] = res("FAIL", failures=["unit or per-recipient fit missing"])
                continue
            cu, _ = complete_ok(UNITS / name)
            if not (cu["id_ok"] and cu["rehash_ok"] and not cu["unlisted"]):
                fails.append("custody")
            pj = jload(UNITS / name / "policy.json")
            rec = jload(UNITS / name / "record.json")
            relz = np.load(UNITS / name / "release.npz", allow_pickle=False)
            info["row_order_ok"] = bool(np.array_equal(relz["row_id"], D.row_id))
            info["binding"] = {"config": (pj.get("config") or {}).get("config") == cid,
                               "teacher_model_sha256": (pj.get("config") or {}).get("teacher_model_sha256") == tt["model_sha"]}
            for r, m in ((1, m1), (2, m2)):
                pol, tok, q, hard = own_pol[(k, r, m)]
                stored = Pol.from_json(pj[f"p{r}"])
                eq = all(own_fits[(k, r, m)].equal(stored.fine).values())
                ts_, qs_, hs_, _ = stored.release(P[r], d[r])
                ri = {"policy_equals_own_recipient_fit": eq and bitwise(stored.cell_token, pol.cell_token),
                      "stored_policy_deployment_equals_release": bitwise(ts_, relz[f"tok{r}"]) and
                      bitwise(qs_, relz[f"q{r}"]) and bitwise(hs_, relz[f"hard{r}"]),
                      "stored_prototypes_equal_recomputed": bitwise(stored.proto, stored.stored["proto"]),
                      "tokens_bitwise": bitwise(tok, relz[f"tok{r}"]), "decoded_bitwise": bitwise(q, relz[f"q{r}"]),
                      "decisions_bitwise": bitwise(hard, relz[f"hard{r}"]), "alpha_equal": int(relz[f"alpha{r}"]) == pol.T,
                      "fit_distortion_rel_diff": rel_diff(own_fit_distortion(P[r][fit], q[fit]),
                                                          (rec.get("fit_distortion") or {}).get(f"D{r}", np.nan)),
                      "fit_token_counts_equal_stored_n": bool(np.array_equal(np.bincount(tok[fit], minlength=pol.T),
                                                                             pol.token_n))}
                rr_ = rec.get(f"r{r}") or {}
                own_pop = [[int(x) for x in pol.token_n[pol.token_class == c]] for c in range(pol.fine.K)]
                ri["cell_population_per_class_equal_record"] = rr_.get("cell_population_per_class") == own_pop
                ri["support"] = {"tokens_with_n_lt_5": int(np.sum((pol.token_n < SPARSE_N) & ~pol.token_fb)),
                                 "singleton_tokens": int(np.sum(pol.token_n == 1)), "min_n_nonfallback":
                                 int(pol.token_n[~pol.token_fb].min()), "fallback_tokens": int(pol.token_fb.sum()),
                                 "fallback_rows_by_role": {ro: int((pol.token_fb[tok] & D.mask[ro]).sum()) for ro in ROLES}}
                ok_cp = (hard == d[r]) & (q.argmax(1) == d[r]) & (relz[f"hard{r}"] == d[r])
                ri["class_preservation_failures_by_role"] = {ro: int((~ok_cp & D.mask[ro]).sum()) for ro in ROLES}
                for ro in ROLES:
                    cp_rows[ro] += int((ok_cp & D.mask[ro]).sum())
                if not ok_cp.all():
                    cp_fail.append(f"{cid}#s{k} r{r}")
                Pa = advP[KS[r - 1]]
                ta, qa, ha, _ = pol.release(Pa, Pa.argmax(1))
                if not (np.array_equal(ha, Pa.argmax(1)) and np.array_equal(qa.argmax(1), Pa.argmax(1))):
                    adv_fail.append(f"{cid}#s{k} r{r}")
                info[f"recipient_{r}"] = ri
                if not (flags_ok(ri) and ri["fit_distortion_rel_diff"] <= STAGEA_TOL):
                    fails.append(f"recipient {r}")
            p1, p2 = own_pol[(k, 1, m1)][0], own_pol[(k, 2, m2)][0]
            info["pair_fingerprint_dpc_convention"] = hashlib.sha256(
                (p1.fingerprint_dpc() + p2.fingerprint_dpc()).encode()).hexdigest() == pj.get("fingerprint")
            if not info["pair_fingerprint_dpc_convention"]:
                fails.append("pair fingerprint")
            info["alphabet"] = [p1.T, p2.T]
            info["states_emitted_fit_own"] = [int(sum(p1.effective_per_class())), int(sum(p2.effective_per_class()))]
            info["total_states_alpha_own"] = p1.T + p2.T
            info["record_total_states"] = rec.get("total_states")
            info["record_states_equal_own_emitted"] = rec.get("total_states") == sum(info["states_emitted_fit_own"])
            if not (info["row_order_ok"] and all(info["binding"].values()) and info["record_states_equal_own_emitted"]):
                fails.append("binding / states")
            q1, h1 = own_pol[(k, 1, m1)][2], own_pol[(k, 1, m1)][3]
            q2, h2 = own_pol[(k, 2, m2)][2], own_pol[(k, 2, m2)][3]
            util[(cid, k)] = {1: my_utility(q1[sel], h1[sel], y_sel[1], 2, const[1]),
                              2: my_utility(q2[sel], h2[sel], y_sel[2], 6, const[2])}
            util[(cid, k)]["preserved"] = bool(np.array_equal(h1, d[1]) and np.array_equal(h2, d[2]))
            OWN_REL[(k, cid)] = {"tok1": own_pol[(k, 1, m1)][1], "q1": q1, "hard1": h1, "tok2": own_pol[(k, 2, m2)][1],
                                 "q2": q2, "hard2": h2, "alpha1": p1.T, "alpha2": p2.T}
            util[(cid, k)]["fit"] = {r: {"alphabet": own_pol[(k, r, m)][0].T,
                                         "emitted": int(sum(own_pol[(k, r, m)][0].effective_per_class())),
                                         "entropy_nats": token_entropy(own_pol[(k, r, m)][1][fit]),
                                         "fit_kl": own_fit_distortion(P[r][fit], own_pol[(k, r, m)][2][fit])}
                                     for r, m in ((1, m1), (2, m2))}
            util[(cid, k)]["states_alpha"] = p1.T + p2.T
            util[(cid, k)]["states_emitted"] = sum(info["states_emitted_fit_own"])
            info["failures"] = fails
            pair_out[f"{cid}#s{k}"] = res("FAIL" if fails else "PASS", **info)
    a1 = res(worst(*[v["status"] for v in a1_out.values()]) if len(a1_out) == 3 else "FAIL", seeds=a1_out)
    a2 = res(worst(*[v["status"] for v in a2_out.values()]) if len(a2_out) == 18 else "FAIL", units=a2_out)
    pairs = res(worst(*[v["status"] for v in pair_out.values()]) if len(pair_out) == 24 else "FAIL", units=pair_out)
    cpres = res("PASS" if not cp_fail and not adv_fail and len(pair_out) == 24 else "FAIL",
                row_checks_passed_by_role=cp_rows, failures=cp_fail, adversarial_failures=adv_fail,
                rule="released decision == own teacher argmax == argmax of the decoded prototype on every row of every "
                     "role, plus adversarial rows (ties, one-hot, underflow, tiny, uniform) through every policy")
    return {"a1_historical": a1, "a2_restarts": a2, "a2_assignments": pairs, "class_preservation": cpres}, util, \
        {"own_fit_cpu_s": t_fit}


def summarize_a1(a1, a2):
    """Descriptive convergence contrast (A1) and convergence counts of every A2 start (own replay)."""
    out = {"a1": {}, "a2_starts": {"total": 0, "converged": 0, "by_stop": {}}}
    for sk, v in a1.get("seeds", {}).items():
        if v.get("status") != "PASS":
            continue
        out["a1"][sk] = {"fit_distortion_contrast": v["fit_distortion_contrast_own"],
                         "src20_classes_converged": {r: v["src20"][f"recipient_{r}"].get("decisions_equal_teacher_all_rows")
                                                     for r in (1, 2)},
                         "r200_stops": {r: v["r200"][f"recipient_{r}"].get("stops_own") for r in (1, 2)},
                         "r200_passes": {r: v["r200"][f"recipient_{r}"].get("passes_own") for r in (1, 2)}}
    for name, v in a2.get("units", {}).items():
        rc = v.get("receipts") or {}
        out["a2_starts"]["total"] += rc.get("starts", 0)
    st = "INFO" if out["a1"] else "PENDING"
    return res(st, **out, note="descriptive: the old code is not called incorrect for stopping at its registered cap")


def check_gate(util, D, pub=None):
    """Own A3 gate per configuration and seed and own A4 decision vs gate.json and CAPACITY_GATE.json."""
    rows, per = {}, {}
    for m1, m2 in rate_bank():
        cid = direct_id(m1, m2)
        ps = {}
        for k in SEEDS:
            u, uU = util.get((cid, k)), util[("SRC|U", k)]
            if u is None:
                continue
            ps[k] = {"tasks": {i: gate_task(u[i + 1], uU[i + 1]) for i in (0, 1)}, "decisions_preserved": u["preserved"],
                     "states": u["states_alpha"], "ll": {i: u[i + 1]["logloss"] for i in (0, 1)}}
        if len(ps) != 3:
            continue
        per[cid] = ps
        rows[cid] = gate_config(ps)
    mine = select_rates(rows)
    short = min(rows, key=lambda c: (rows[c]["worst_normalized_excess"], rows[c]["mean_states"], c)) if rows else None
    qs = q_star(rows)
    gp = RUN / "gate.json"
    out = {"own_decision": mine["decision"], "own_selected": mine["selected"], "own_eligible": mine["eligible"],
           "own_headroom": mine["headroom"], "own_q_star": qs, "own_best_shortfall": None if mine["selected"] else short,
           "configs": {c: {**rows[c], "per_seed": {k: {"eligible": all(per[c][k]["tasks"][i]["eligible"] for i in (0, 1))
                                                        and per[c][k]["decisions_preserved"],
                                                        "excess": {TASKS[i]: per[c][k]["tasks"][i]["excess"] for i in (0, 1)},
                                                        "failing": {TASKS[i]: [g_ for g_, v in per[c][k]["tasks"][i]["checks"].items()
                                                                               if not v] for i in (0, 1)}}
                                                    for k in SEEDS}} for c in rows}}
    fails = []
    if not gp.exists():
        return res("PENDING", reason="gate.json not yet written", **out)
    G = jload(gp)
    dec = G.get("decision", {})
    lead_met = dec.get("gate") == "CAPACITY_GATE_MET"
    out["lead_decision"] = dec.get("gate")
    out["lead_selected"] = dec.get("selected_rates")
    out["decision_equal"] = lead_met == (mine["decision"] == "CAPACITY_GATE_PASSED")
    out["selected_equal"] = dec.get("selected_rates") == mine["selected"]
    out["eligible_equal"] = sorted(dec.get("eligible", [])) == sorted(mine["eligible"])
    out["best_shortfall_equal"] = dec.get("best_shortfall_config") == (None if mine["selected"] else short)
    mx, bad_flags = 0.0, []
    for c, pcs in (G.get("per_config_seed") or {}).items():
        for k_, v in pcs.items():
            k = int(k_)
            u = util.get((c, k))
            if u is None:
                bad_flags.append(f"{c}#s{k}: no own replay")
                continue
            for i, t in ((1, "income"), (2, "occupation")):
                lu = (v.get("util") or {}).get(t, {})
                for q_ in ("acc", "logloss", "brier"):
                    mx = max(mx, abs(float(lu.get(q_, np.nan)) - u[i][q_]))
            ge = v.get("gate", {})
            own_e = per[c][k]["decisions_preserved"] and all(per[c][k]["tasks"][i]["eligible"] for i in (0, 1))
            if bool(ge.get("eligible")) != own_e:
                bad_flags.append(f"{c}#s{k}: eligible {ge.get('eligible')} vs own {own_e}")
            own_h = all(per[c][k]["tasks"][i]["headroom_ok"] for i in (0, 1))
            if "headroom" in ge and bool(ge["headroom"]) != own_h:
                bad_flags.append(f"{c}#s{k}: headroom")
    out["utility_max_abs_diff_vs_gate_json"] = mx
    out["per_seed_flag_mismatches"] = bad_flags
    near = []
    for c, ps in per.items():
        for k, v in ps.items():
            for i in (0, 1):
                for nm, mg in (("ll", GATE["ll_excess"] - v["tasks"][i]["excess"]["logloss"]),
                               ("brier", GATE["brier_excess"] - v["tasks"][i]["excess"]["brier"])):
                    if abs(mg) < 1e-9:
                        near.append(f"{c}#s{k} {TASKS[i]} {nm} margin {mg:.3g}")
    out["borderline_margins_within_1e-9"] = near
    if not (out["decision_equal"] and out["selected_equal"] and out["eligible_equal"] and out["best_shortfall_equal"]
            and mx <= 1e-12 and not bad_flags):
        fails.append("gate replay differs")
    pub = RES / "CAPACITY_GATE.json" if pub is None else pub
    if pub.exists():
        pj = jload(pub)
        out["public_decision_equals_private"] = pj.get("decision") == dec
        if not out["public_decision_equals_private"]:
            fails.append("CAPACITY_GATE.json decision differs from gate.json")
    st = "FAIL" if fails else ("WARN" if near else "PASS")
    return res(st, failures=fails, **out)


def check_inner_utility(util):
    """Own INNER_SELECTION utility of U and of every Stage A code (incl. the A1 versions) and the public
    CONVERGENCE_DIAGNOSTIC.csv: every numeric cell must equal the own value printed with the CSV's 8 significant
    digits; flags and stop reasons must be equal."""
    tab = {}
    for key, u in util.items():
        if key[0] == "A1sum":
            continue
        cid, k = key
        tab.setdefault(cid, {})[f"s{k}"] = {TASKS[i - 1]: {q_: u[i][q_] for q_ in ("acc", "logloss", "brier", "const_acc")}
                                             for i in (1, 2)}
    cd = RES / "CONVERGENCE_DIAGNOSTIC.csv"
    out = {"table": tab}
    if not cd.exists():
        return res("PENDING", reason="CONVERGENCE_DIAGNOSTIC.csv not yet written", **out)
    import csv
    rows = list(csv.DictReader(cd.open()))
    cells, bad = 0, []
    for r_ in rows:
        ver, k, task = r_["version"], int(r_["seed"]), r_["recipient"]
        i = 1 if task == "income" else 2
        u, uU = util.get((f"A1|{ver}", k)), util[("SRC|U", k)]
        sm = util.get(("A1sum", ver, k), {}).get(i)
        if u is None or sm is None:
            bad.append(f"{ver} s{k} {task}: no own replay")
            continue
        own = {"inner_logloss": u[i]["logloss"], "inner_brier": u[i]["brier"],
               "inner_ll_excess": u[i]["logloss"] - uU[i]["logloss"], "inner_brier_excess": u[i]["brier"] - uU[i]["brier"],
               "fit_mean_kl": sm["fit_mean_kl"], "fit_objective_total": sm["fit_objective_total"]}
        for col, v in own.items():
            if col in r_:
                cells += 1
                if r_[col] != f"{v:.8g}":
                    bad.append(f"{ver} s{k} {task} {col}: csv {r_[col]} vs own {v:.8g}")
        for col in ("classes_fitted", "classes_converged", "rounds_used_max", "rounds_used_sum"):
            if col in r_:
                cells += 1
                if int(r_[col]) != int(sm[col]):
                    bad.append(f"{ver} s{k} {task} {col}: csv {r_[col]} vs own {sm[col]}")
        for col, v in (("all_converged", str(sm["all_converged"])), ("stop_reasons", sm["stop_reasons"])):
            if col in r_:
                cells += 1
                if (set(r_[col].split(";")) if col == "stop_reasons" else r_[col]) != \
                        (set(v.split(";")) if col == "stop_reasons" else v):
                    bad.append(f"{ver} s{k} {task} {col}: csv {r_[col]} vs own {v}")
    out.update({"convergence_diagnostic_rows": len(rows), "cells_checked": cells, "mismatches": bad})
    return res("PASS" if not bad and len(rows) == 12 and cells else "FAIL", **out)


def check_capacity_curve(util, gate_check):
    """Public CAPACITY_CURVE.csv (Stage A, inner) vs own per-seed aggregates (6-decimal print: |diff| <= 5e-7)."""
    cc = RES / "CAPACITY_CURVE.csv"
    if not cc.exists():
        return res("PENDING", reason="CAPACITY_CURVE.csv not yet written")
    import csv
    rows = list(csv.DictReader(cc.open()))
    confs = gate_check.get("configs", {})
    sel = set(gate_check.get("own_selected") or [])
    bad, cells, mx = [], 0, 0.0
    for r_ in rows:
        cid = r_["config"]
        us = {k: util.get((cid, k)) for k in SEEDS}
        if any(v is None for v in us.values()) or cid not in confs:
            bad.append(f"{cid}: no own replay")
            continue
        own = {"mean_total_states": confs[cid]["mean_states"], "worst_seed_norm_excess": confs[cid]["worst_normalized_excess"]}
        for i, t in ((1, "income"), (2, "occupation")):
            lle = [us[k][i]["logloss"] - util[("SRC|U", k)][i]["logloss"] for k in SEEDS]
            bre = [us[k][i]["brier"] - util[("SRC|U", k)][i]["brier"] for k in SEEDS]
            own.update({f"inner_ll_excess_{t}_mean": float(np.mean(lle)), f"inner_ll_excess_{t}_worst": max(lle),
                        f"inner_brier_excess_{t}_mean": float(np.mean(bre)), f"inner_brier_excess_{t}_worst": max(bre),
                        f"alphabet_{t}_mean": float(np.mean([us[k]["fit"][i]["alphabet"] for k in SEEDS])),
                        f"emitted_fit_{t}_mean": float(np.mean([us[k]["fit"][i]["emitted"] for k in SEEDS])),
                        f"token_entropy_fit_{t}_mean": float(np.mean([us[k]["fit"][i]["entropy_nats"] for k in SEEDS])),
                        f"fit_kl_{t}_mean": float(np.mean([us[k]["fit"][i]["fit_kl"] for k in SEEDS]))})
        for col, v in own.items():
            if col not in r_:
                bad.append(f"{cid}: column {col} absent")
                continue
            cells += 1
            dlt = abs(float(r_[col]) - v)
            mx = max(mx, dlt)
            if dlt > 5e-7 + 1e-12:
                bad.append(f"{cid} {col}: csv {r_[col]} vs own {v:.7f}")
        for col, v in (("eligible_all_seeds", confs[cid]["eligible"]), ("headroom_all_seeds", confs[cid]["headroom_all_seeds"]),
                       ("selected", cid in sel)):
            cells += 1
            if r_.get(col) != str(bool(v)):
                bad.append(f"{cid} {col}: csv {r_.get(col)} vs own {v}")
    return res("PASS" if not bad and len(rows) == 8 else "FAIL", rows=len(rows), cells_checked=cells,
               max_abs_diff=mx, mismatches=bad, note="token entropy in nats over OSF_DEFENSE_FIT emitted tokens")


STAGE_MODULES = {"admit": ["qpc/data.py", "qpc/admit.py", "qpc/run.py"],
                 "stagea": ["qpc/kmeans.py", "qpc/stagea.py", "qpc/utility.py", "qpc/run.py", "qpc/release.py",
                            "dpc/partition.py", "dpc/release.py", "dpc/utility.py"],
                 "gate": ["qpc/gate.py", "qpc/stagea.py", "qpc/utility.py", "qpc/run.py"],
                 "partition": ["qpc/partition.py", "qpc/kmeans.py", "qpc/run.py", "dpc/partition.py"],
                 "fit": ["qpc/compress.py", "qpc/release.py", "qpc/run.py", "dpc/release.py"]}


def check_code_hashes():
    """Every named lock's code hashes equal the files in its own commit; amendments / later locks that re-hash a file
    declare it; every module a completed stage depends on (own list, from the import structure) is locked and the
    worktree copy still has the governing hash."""
    lf = lock_files()
    order = []
    for name, d in lf.items():
        if d["exists"] and d["commits"]:
            order.append((jload(RES / f"{name}.json").get("written_at", ""), name))
    order.sort()
    current, at_commit, relocks = {}, {}, []
    for _, name in order:
        L_ = jload(RES / f"{name}.json")
        cf = L_.get("code_files") or {}
        commit = lf[name]["commits"][-1][0]
        mism = [f_ for f_, h in cf.items() if (lambda b: b is None or hashlib.sha256(b).hexdigest() != h)(
            git_show_bytes(commit, f_))]
        declared = set((L_.get("changes_previously_locked") or {}).keys()) if isinstance(
            L_.get("changes_previously_locked"), dict) else set(L_.get("changes_previously_locked") or [])
        for f_, h in cf.items():
            if f_ in current and current[f_][0] != h and f_ not in declared:
                relocks.append(f"{name} re-hashes {f_} (previously {current[f_][1]}) without a declared change")
            current[f_] = (h, name)
        docs = {doc: (sha_file(RES / doc) == h) if (RES / doc).exists() else None
                for doc, h in (L_.get("documents_sha256") or {}).items()}
        at_commit[name] = {"files": len(cf), "files_differing_from_the_lock_commit": mism,
                           "declared_changes": sorted(declared), "documents_equal_worktree": docs}
    stage_mod = {}
    wt_mism = []
    for stage, mods in STAGE_MODULES.items():
        gov = STAGE_LOCK[stage]
        L_ = jload(RES / f"{gov}.json") if (RES / f"{gov}.json").exists() else {}
        cf = L_.get("code_files") or {}
        stage_mod[stage] = {m: (m in cf) for m in mods}
        for m in mods:
            if m in current and (WT / m).exists() and sha_file(WT / m) != current[m][0]:
                wt_mism.append(f"{m} (governed by {current[m][1]})")
    missing = [f"{st}:{m}" for st, v in stage_mod.items() for m, ok in v.items() if not ok and
               (RES / f"{STAGE_LOCK[st]}.json").exists()]
    bad = any(v["files_differing_from_the_lock_commit"] for v in at_commit.values()) or relocks or missing or wt_mism
    latest_doc = {}
    for _, name in order:
        for doc, h in (jload(RES / f"{name}.json").get("documents_sha256") or {}).items():
            latest_doc[doc] = (h, name)
    doc_changed = sorted(f"{doc} (locked by {nm})" for doc, (h, nm) in latest_doc.items()
                         if (RES / doc).exists() and sha_file(RES / doc) != h)
    post_lock = {}
    for doc, (h, nm) in latest_doc.items():
        lc = lf[nm]["commits"][-1][0]
        log = git("log", "--format=%h %cI %s", f"{lc}..HEAD", "--", f"{REL_RES}/{doc}") or ""
        committed = [l_ for l_ in log.splitlines() if l_]
        dirty = bool(git("status", "--short", "--", f"{REL_RES}/{doc}"))
        if committed or dirty:
            post_lock[doc] = {"governing_lock": nm, "commits_after_lock": [c_[:60] for c_ in committed],
                              "uncommitted_edit": dirty}
    st = "FAIL" if bad else ("WARN" if doc_changed else "PASS")
    return res(st, locked_documents_changed_since_their_latest_lock=doc_changed, post_lock_document_changes=post_lock,
               reason=("locked documents modified after their latest governing lock (custody; needs a declared change "
                       "in the next lock or an amendment)") if doc_changed else None,
               locks_in_order=[n for _, n in order], files_governed=len(current),
               lock_vs_its_commit=at_commit, undeclared_relocks=relocks, stage_modules_locked=stage_mod,
               stage_modules_missing_from_lock=missing, stage_modules_changed_in_worktree=wt_mism)


def mutation_power_stage_a(D: Data, T, L):
    """In-memory mutations of real Stage A records (seed 0) that each comparator must detect; nothing is written."""
    import copy
    fit = D.fit_idx
    tt = T[(0, "U")]
    out = {}
    name = "dir__s0__r2__m16"
    pj = jload(UNITS / name / "policy.json")
    rr = jload(UNITS / name / "record.json")["receipts"]
    fine = fit_partition(tt["p2"][fit], tt["d2"][fit], 6, 16, starts=START_ORDER, rule="qpc")
    base, *_ = compare_policy_to_own(Pol.from_json(pj), fine, tt["p2"], tt["d2"])
    out["baseline_clean"] = bool(base["partition_bitwise"] and base["policy_fingerprint_equal"])
    z = copy.deepcopy(pj)
    z["fine"]["centroid"][0][0] += 1e-15
    c1, *_ = compare_policy_to_own(Pol.from_json(z), fine, tt["p2"], tt["d2"])
    out["centroid_edit_1e-15_detected"] = not c1["partition_bitwise"]
    z = copy.deepcopy(pj)
    z["fine"]["n"][0] += 1
    c2, *_ = compare_policy_to_own(Pol.from_json(z), fine, tt["p2"], tt["d2"])
    out["support_count_edit_detected"] = not c2["partition_bitwise"]
    ok_r = receipts_ok(compare_start_receipts(fine.receipt["per_class"], rr["per_class"]))
    z = copy.deepcopy(rr)
    z["per_class"][0]["starts"][1]["objective_trajectory"][-1] *= 1 + 1e-8
    out["trajectory_edit_1e-8_detected"] = ok_r and not receipts_ok(compare_start_receipts(fine.receipt["per_class"],
                                                                                          z["per_class"]))
    z = copy.deepcopy(rr)
    ci = z["per_class"][0]["starts"][2]["init"]["chosen_distinct_index"]
    ci[0], ci[1] = ci[1], ci[0]
    out["kpp_draw_order_edit_detected"] = not receipts_ok(compare_start_receipts(fine.receipt["per_class"], z["per_class"]))
    z = copy.deepcopy(rr)
    z["per_class"][1]["winner"] = [s_ for s_ in START_ORDER if s_ != z["per_class"][1]["winner"]][0]
    out["winner_label_edit_detected"] = not receipts_ok(compare_start_receipts(fine.receipt["per_class"], z["per_class"]))
    z = copy.deepcopy(rr)
    z["per_class"][2]["starts"][0]["stop_reason"] = "cap"
    out["stop_reason_edit_detected"] = not receipts_ok(compare_start_receipts(fine.receipt["per_class"], z["per_class"]))
    # a same-class token swap in a pair release: decisions still preserved, token check must fail
    relz = dict(np.load(UNITS / "pol__s0__U_DIRECT-TASK_i8o16" / "release.npz", allow_pickle=False))
    pol = direct_policy(fine)
    tok = relz["tok2"].copy()
    j = int(np.flatnonzero(pol.token_class[tok] == 1)[0])
    tok[j] = [t_ for t_ in np.flatnonzero(pol.token_class == 1) if t_ != tok[j]][0]
    own_tok, _, own_hard, _ = pol.release(tt["p2"], tt["d2"])
    out["same_class_token_swap_detected_by_token_check_only"] = (not bitwise(own_tok, tok)) and \
        bool(np.array_equal(pol.token_class[tok], tt["d2"]))
    # a planted SEX-keyed partition is not a nearest-centroid partition of the teacher scores
    s_fit = L["sex"][fit]
    lab = tt["d1"][fit] * 2 + s_fit
    n_, S_, A_ = stats_of(tt["p1"][fit], lab, 4)
    planted = Fine(2, np.array([0, 0, 1, 1]), smooth(S_ / n_[:, None], [0, 0, 1, 1]), S_ / n_[:, None], n_, S_, A_,
                   np.zeros(4, bool))
    n2, S2, _ = stats_of(tt["p1"][fit], assign(tt["p1"][fit], tt["d1"][fit], planted), 4)
    out["planted_sex_partition_detected"] = not (np.array_equal(n2, n_) and np.array_equal(S2, S_))
    ok = all(v for v in out.values())
    return res("PASS" if ok else "FAIL", unit=name, **out,
               note="SEX of OSF_DEFENSE_FIT read only for the planted-partition mutation (as in the predecessor)")


LOG_TOL_B = 1e-10       # recorded increments / deltas vs own
OBJ_TOL_B = 1e-12       # recorded objectives vs own


def mi_vec(s, code, ncode):
    """Vectorised plug-in I(S; code) (nats) for integer codes in [0, ncode)."""
    tab = np.bincount(np.asarray(s, np.int64) * ncode + np.asarray(code, np.int64), minlength=2 * ncode).reshape(2, ncode)
    tab = tab.astype(np.float64)
    N = tab.sum()
    rs = tab.sum(1, keepdims=True)
    cs = tab.sum(0, keepdims=True)
    nz = tab > 0
    return float(np.sum((tab / N * np.log(np.where(nz, tab * N, 1.0) / np.where(nz, rs * cs, 1.0)))[nz]))


def b_config(fam, lam=None):
    if fam == "CLASS":
        return "U|CLASS|i1o1"
    return f"U|{fam}|i8o64" + (f"|l{lam:g}" if fam in PRIV_FAMS else "")


def b_bank():
    out = [("CLASS", None), ("FINE-TASK", None)]
    for lam in LAMS:
        out += [(f, lam) for f in ("LOCAL", "SEQ-12", "SEQ-21")]
    out += [("JOINT", lam) for lam in LAMS]          # witnesses first
    return out


def cmp_list(mine, rec, keys, vals):
    """Sequence equality on `keys`, max |diff| on `vals` (None-aware) and tied-candidate counts."""
    out = {"n_recorded": len(rec), "n_own": len(mine),
           "sequence_equal": [tuple(x.get(k_) for k_ in keys) for x in mine] == [tuple(y.get(k_) for k_ in keys)
                                                                                 for y in rec], "max_abs_diff": 0.0,
           "tied_equal": True}
    if out["sequence_equal"]:
        for x, y in zip(mine, rec):
            for v in vals:
                if (x.get(v) is None) != (y.get(v) is None):
                    out["max_abs_diff"] = float("inf")
                elif x.get(v) is not None:
                    out["max_abs_diff"] = max(out["max_abs_diff"], abs(float(x[v]) - float(y[v])))
            if "tied_candidates" in y:
                out["tied_equal"] &= x.get("tied_candidates") == y.get("tied_candidates")
        if out["sequence_equal"] is True and len(mine) == 0:
            out["max_abs_diff"] = 0.0
    else:
        k0 = next((i for i, (x, y) in enumerate(zip(mine, rec)) if tuple(x.get(k_) for k_ in keys) !=
                   tuple(y.get(k_) for k_ in keys)), min(len(mine), len(rec)))
        out["first_difference_index"] = k0
    return out


MERGE_KEYS = ("kind", "recipient", "class", "a", "b")
MERGE_VALS = ("increment", "dD", "dI_own", "dI12")
MOVE_KEYS = ("kind", "sweep", "recipient", "fine_cell", "from", "to", "a", "b", "class")
MOVE_VALS = ("delta", "increment", "dD", "dI_own", "dI12")


def terms_diff(a, b, keys=("D1", "D2", "I1", "I2", "I12")):
    return max(abs(float(a[k_]) - float(b[k_])) for k_ in keys if k_ in a and k_ in b)


def compare_stage(m, r_, extra_keys=()):
    out = {"merges": cmp_list(m.get("merges", []), r_.get("merges", []), MERGE_KEYS, MERGE_VALS),
           "moves": cmp_list(m.get("moves", []), r_.get("moves", []), MOVE_KEYS, MOVE_VALS)}
    for k_ in ("sweeps", "converged", "merges_to_cap", "extra_merges_greedy") + tuple(extra_keys):
        if k_ in r_ and k_ in m:
            out[f"{k_}_equal"] = r_[k_] == m[k_]
    for k_ in ("stage_objective_after_greedy", "stage_objective_after_refine", "initial_F_joint", "refined_F_joint"):
        if k_ in r_ and k_ in m:
            out[f"{k_}_abs_diff"] = abs(float(r_[k_]) - float(m[k_]))
    for k_ in ("after_greedy", "after_refine"):
        if k_ in r_ and k_ in m:
            out[f"{k_}_max_abs_diff"] = terms_diff(m[k_], r_[k_])
    return out


def stage_ok(c):
    ok = all(c[x]["sequence_equal"] and c[x]["max_abs_diff"] <= LOG_TOL_B and c[x]["tied_equal"] for x in ("merges", "moves"))
    ok &= all(v for k_, v in c.items() if k_.endswith("_equal") and isinstance(v, bool))
    ok &= all(v <= OBJ_TOL_B for k_, v in c.items() if k_.endswith("_abs_diff"))
    return bool(ok)


def check_stage_b(D: Data, L, T):
    """Stage B replay at the locked rate (8, 64) and lambda bank: own fine partitions (three starts, <= 200 rounds),
    own assignment of every row, the fine table, every family's merge / extra-merge / move / sweep log, the sequential
    class-only first stage and its old-rule diagnostic, JOINT starts / candidates / winner / witness dominance /
    unresolved optima, every release (all rows), class preservation, final D / I terms from the released tokens and
    decoded vectors of the fitting rows, the permutation-null MI and the sparsity receipts. Reads OSF_DEFENSE_FIT SEX."""
    fit = D.fit_idx
    s_fit = L["sex"][fit]
    assert set(np.unique(s_fit).tolist()) <= {0, 1}
    N = len(fit)
    prng = np.random.default_rng(20261006)
    perms = [prng.permutation(N) for _ in range(100)]
    fine_out, unit_out, cp_fail, adv_fail = {}, {}, [], []
    cp_rows = {r_: 0 for r_ in ROLES}
    perm_rule = {"gather": 0, "scatter": 0, "neither": 0}
    advP = {K: adversarial_rows(K) for K in KS}
    own_maps, own_terms, cpu = {}, {}, {"fine": 0.0, "search": 0.0}
    for k in SEEDS:
        tt = T[(k, "U")]
        P, d = {1: tt["p1"], 2: tt["p2"]}, {1: tt["d1"], 2: tt["d2"]}
        name = f"fine__s{k}"
        fails = []
        if not unit_done(name):
            fine_out[name] = res("FAIL", failures=["unit missing"])
            continue
        cu, _ = complete_ok(UNITS / name)
        if not (cu["id_ok"] and cu["rehash_ok"] and not cu["unlisted"]):
            fails.append("custody")
        fj = jload(UNITS / name / "fine.json")
        rec = jload(UNITS / name / "record.json")
        az = np.load(UNITS / name / "assign.npz", allow_pickle=False)
        fines, fa, info = {}, {}, {}
        t0 = time.process_time()
        for r, cap in ((1, FINE_CAPS[0]), (2, FINE_CAPS[1])):
            mf = fit_partition(P[r][fit], d[r][fit], KS[r - 1], cap, starts=START_ORDER, rule="qpc")
            stored = Fine.from_json(fj[f"fine{r}"])
            eq = mf.equal(stored)
            fa[r] = assign(P[r], d[r], mf)
            n_, S_, A_ = stats_of(P[r][fit], fa[r][fit], mf.F)
            rr = rec.get(f"r{r}", {})
            rc = compare_start_receipts(mf.receipt["per_class"], rr.get("per_class", []))
            fps = rr.get("fingerprints", {})
            own_fps = {"winner": mf.fingerprint_dpc(), **{st_: mf.receipt["by_start"][st_].fingerprint_dpc()
                                                          for st_ in START_ORDER}}
            ri = {"partition_bitwise": all(eq.values()), "fingerprint_equal": mf.fingerprint_dpc() == stored.stored_fp,
                  "start_fingerprints_equal": {x: fps.get(x) == own_fps[x] for x in own_fps},
                  "assignment_all_rows_equal": bool(np.array_equal(fa[r], az[f"f{r}"])),
                  "deployment_on_fit_reproduces_stats": bool(np.array_equal(n_, mf.n) and np.array_equal(S_, mf.S) and
                                                             np.array_equal(A_, mf.A)),
                  "receipts": rc, "receipts_ok": receipts_ok(rc), "F": mf.F,
                  "cells_per_class": [int(np.sum(mf.cls == c)) for c in range(mf.K)],
                  "fallback_classes": [int(c) for c in mf.cls[mf.fb]],
                  "winners": [pc.get("winner") for pc in mf.receipt["per_class"]],
                  "winner_stops": [next(x["stop"] for x in pc["starts"] if x["start"] == pc["winner"])
                                   for pc in mf.receipt["per_class"] if not pc.get("fallback")],
                  "min_cell_n": int(mf.n[~mf.fb].min()), "cells_n_lt_5": int(np.sum((mf.n < SPARSE_N) & ~mf.fb)),
                  "singleton_cells": int(np.sum(mf.n == 1))}
            if not (ri["partition_bitwise"] and ri["fingerprint_equal"] and all(ri["start_fingerprints_equal"].values())
                    and ri["assignment_all_rows_equal"] and ri["deployment_on_fit_reproduces_stats"] and ri["receipts_ok"]):
                fails.append(f"recipient {r}")
            fines[r] = mf
            info[f"recipient_{r}"] = ri
        cpu["fine"] += time.process_time() - t0
        info["row_order_ok"] = bool(np.array_equal(az["row_id"], D.row_id))
        info["failures"] = fails
        fine_out[name] = res("FAIL" if fails or not info["row_order_ok"] else "PASS", **info)
        F1, F2 = fines[1].F, fines[2].F
        Tf = np.zeros((2, F1, F2), dtype=np.int64)
        np.add.at(Tf, (s_fit.astype(np.int64), fa[1][fit], fa[2][fit]), 1)
        # ---------------- families
        mine_lab = {}
        for fam, lam in b_bank():
            cid = b_config(fam, lam)
            uname = f"pol__s{k}__{cid.replace('|', '_')}"
            info, fails = {"config": cid, "seed": k}, []
            if not unit_done(uname):
                unit_out[f"{cid}#s{k}"] = res("FAIL", failures=["unit missing"])
                continue
            cu, _ = complete_ok(UNITS / uname)
            if not (cu["id_ok"] and cu["rehash_ok"] and not cu["unlisted"]):
                fails.append("custody")
            rec = jload(UNITS / uname / "record.json")
            pj = jload(UNITS / uname / "policy.json")
            relz = np.load(UNITS / uname / "release.npz", allow_pickle=False)
            wl = None
            if fam == "JOINT":
                wl = {wf: mine_lab[b_config(wf, None if wf == "FINE-TASK" else lam)] for wf in WITNESS_FAMS}
            t0 = time.process_time()
            labs, log, st_final = my_family(fam, fines, Tf, 8, 64, lam, wl)
            cpu["search"] += time.process_time() - t0
            mine_lab[cid] = (labs[1], labs[2])
            # search log comparison
            if fam in ("FINE-TASK", "LOCAL", "SEQ-12", "SEQ-21"):
                rs = {x["stage"]: x for x in rec.get("stages", [])}
                cmpd = {}
                for ms in log["stages"]:
                    rr = rs.get(ms["stage"])
                    cmpd[ms["stage"]] = compare_stage(ms, rr) if rr else {"missing": True}
                info["stages"] = cmpd
                info["stage_names_equal"] = sorted(rs) == sorted(x["stage"] for x in log["stages"])
                ok_s = info["stage_names_equal"] and all(stage_ok(c) for c in cmpd.values() if not c.get("missing"))
                if "after_greedy" in rec:
                    info["after_greedy_max_abs_diff"] = terms_diff(log["after_greedy"], rec["after_greedy"])
                    ok_s &= info["after_greedy_max_abs_diff"] <= OBJ_TOL_B
                if fam.startswith("SEQ"):
                    bc, mb = rec.get("baseline_correction", {}), log["baseline_correction"]
                    keys_ = [k_ for k_ in mb if isinstance(mb[k_], float) and k_ in bc]
                    info["baseline_correction_max_abs_diff"] = max(abs(mb[k_] - float(bc[k_])) for k_ in keys_)
                    ob, mo = bc.get("old_rule_stage1", {}), mb["old_rule_stage1"]
                    info["old_rule_same_map_equal"] = ob.get("same_map_as_corrected") == mo["same_map_as_corrected"]
                    info["old_rule_values_max_abs_diff"] = max(abs(mo[k_] - float(ob[k_])) for k_ in mo
                                                               if isinstance(mo[k_], float) and k_ in ob)
                    info["counterpart_is_class_only"] = bc.get("counterpart") == "CLASS-ONLY"
                    info["first_map_never_revised"] = mb["first_map_never_revised"]
                    info["old_rule_map_differs"] = not mo["same_map_as_corrected"]
                    info["corrected_minus_old_F_joint"] = mo["corrected_minus_old_F_joint"]
                    ok_s &= (info["baseline_correction_max_abs_diff"] <= OBJ_TOL_B and info["old_rule_same_map_equal"]
                             and info["old_rule_values_max_abs_diff"] <= OBJ_TOL_B and info["counterpart_is_class_only"]
                             and info["first_map_never_revised"])
                info["search_ok"] = bool(ok_s)
            elif fam == "JOINT":
                rst = rec.get("starts", {})
                cmpd = {n: compare_stage(log["starts"][n], rst.get(n, {})) for n in JOINT_START_ORDER}
                info["starts"] = cmpd
                rc_ = [(c_["start"], c_["kind"], c_["F_joint"]) for c_ in rec.get("candidates", [])]
                info["candidate_order_equal"] = [(a_, b_) for a_, b_, _ in rc_] == [(a_, b_) for a_, b_, _ in log["candidates"]]
                info["candidate_F_joint_max_abs_diff"] = max(abs(x[2] - y[2]) for x, y in zip(rc_, log["candidates"])) \
                    if info["candidate_order_equal"] else float("inf")
                rw = f"{rec['winner']['start']}:{rec['winner']['kind']}"
                ow = "%s:%s" % log["winner"]
                info["recorded_winner"], info["own_winner"] = rw, ow
                info["winner_map_equal"] = rw == ow or (rw in log["near_tied_candidates"] and log["near_tied_share_map"])
                info["winner_label"] = "exact" if rw == ow else ("tie-equivalent (same map)" if info["winner_map_equal"]
                                                                 else "MISMATCH")
                info["same_map_as_winner_equal"] = {n: rst.get(n, {}).get("same_map_as_winner") ==
                                                    log["starts"][n]["same_map_as_winner"] for n in JOINT_START_ORDER}
                info["witness_sources"] = {n: rst.get(n, {}).get("source") for n in WITNESS_FAMS}
                wd = rec.get("witness_dominance", {})
                fj_final = F3(st_final.terms(), lam)["F_joint"]
                info["dominance_own"] = {n: fj_final - log["witness_F_joint"][n] for n in WITNESS_FAMS}
                info["dominates_all_witnesses"] = all(v <= 1e-12 for v in info["dominance_own"].values())
                info["dominance_max_abs_diff_vs_record"] = max(abs(info["dominance_own"][n] -
                                                                   float(wd.get(n, {}).get("final_minus_witness", np.nan)))
                                                               for n in WITNESS_FAMS)
                info["unresolved_equal"] = sorted(x["start"] for x in rec.get("unresolved_local_optima", [])) == \
                    log["unresolved_local_optima"]
                info["starts_not_converged_equal"] = sorted(rec.get("starts_not_converged", [])) == \
                    sorted(log["starts_not_converged"])
                info["unresolved_local_optima"] = log["unresolved_local_optima"]
                info["starts_not_converged"] = log["starts_not_converged"]
                info["gap_to_winner_own"] = {n: log["starts"][n]["refined_F_joint"] - log["best_canonical_F_joint"]
                                             for n in JOINT_START_ORDER}
                ok_s = (all(stage_ok(c) for c in cmpd.values()) and info["candidate_order_equal"] and
                        info["candidate_F_joint_max_abs_diff"] <= OBJ_TOL_B and info["winner_map_equal"] and
                        all(info["same_map_as_winner_equal"].values()) and info["dominates_all_witnesses"] and
                        info["dominance_max_abs_diff_vs_record"] <= OBJ_TOL_B and info["unresolved_equal"] and
                        info["starts_not_converged_equal"] and
                        all(v == "passed_in" for v in info["witness_sources"].values()))
                info["search_ok"] = bool(ok_s)
            else:
                info["search_ok"] = rec.get("stages", []) == []
            if not info["search_ok"]:
                fails.append("search replay")
            # policies and releases
            toks, qs, hards = {}, {}, {}
            for r in (1, 2):
                stored = Pol.from_json(pj[f"p{r}"])
                own = Pol(fines[r], tokens_from_lab(labs[r]))
                tok, q, hard, _ = own.release(P[r], d[r])
                ts_, qs_, hs_, _ = stored.release(P[r], d[r])
                ri = {"assignment_partition_equals_own_fine": all(fines[r].equal(stored.fine).values()),
                      "cell_token_equals_own": bitwise(stored.cell_token, own.cell_token),
                      "prototypes_equal_own": bitwise(stored.proto, own.proto) and bitwise(stored.stored["proto"], own.proto),
                      "policy_fingerprint_equal": own.fingerprint_dpc() == stored.stored.get("fingerprint"),
                      "tokens_bitwise": bitwise(tok, relz[f"tok{r}"]), "decoded_bitwise": bitwise(q, relz[f"q{r}"]),
                      "decisions_bitwise": bitwise(hard, relz[f"hard{r}"]), "alpha_equal": int(relz[f"alpha{r}"]) == own.T,
                      "stored_deployment_equals_release": bitwise(ts_, relz[f"tok{r}"]) and bitwise(qs_, relz[f"q{r}"])}
                st_ = own.structure()
                ri["structure_ok"] = all(v for v in st_.values() if isinstance(v, bool))
                ok_cp = (hard == d[r]) & (q.argmax(1) == d[r]) & (relz[f"hard{r}"] == d[r])
                for ro in ROLES:
                    cp_rows[ro] += int((ok_cp & D.mask[ro]).sum())
                if not ok_cp.all():
                    cp_fail.append(f"{cid}#s{k} r{r}")
                Pa = advP[KS[r - 1]]
                _, qa, ha, _ = own.release(Pa, Pa.argmax(1))
                if not (np.array_equal(ha, Pa.argmax(1)) and np.array_equal(qa.argmax(1), Pa.argmax(1))):
                    adv_fail.append(f"{cid}#s{k} r{r}")
                ri["alphabet"], ri["tokens_per_class"] = own.T, own.tokens_per_class()
                ri["effective_states"] = int(sum(own.effective_per_class()))
                info[f"recipient_{r}"] = ri
                if not all(v for v in ri.values() if isinstance(v, bool)):
                    fails.append(f"release r{r}")
                toks[r], qs[r], hards[r] = tok, q, hard
            info["pair_fingerprint_equal"] = hashlib.sha256((Pol(fines[1], tokens_from_lab(labs[1])).fingerprint_dpc() +
                                                             Pol(fines[2], tokens_from_lab(labs[2])).fingerprint_dpc())
                                                            .encode()).hexdigest() == pj.get("fingerprint")
            info["binding_ok"] = (pj.get("config") or {}).get("config") == cid and \
                (pj.get("config") or {}).get("teacher_model_sha256") == tt["model_sha"]
            if not (info["pair_fingerprint_equal"] and info["binding_ok"]):
                fails.append("fingerprint / binding")
            # final terms from released tokens / decoded vectors on the fitting rows (row level, own MI)
            o = objectives(P[1][fit], toks[1][fit], qs[1][fit], P[2][fit], toks[2][fit], qs[2][fit], s_fit, lam)
            eng = st_final.terms()
            info["final_own_row_level"] = {k_: o[k_] for k_ in ("D1", "D2", "I1", "I2", "I12")}
            info["final_max_abs_diff_vs_record"] = terms_diff(o, rec.get("final", {}))
            info["engine_vs_row_level_max_abs_diff"] = terms_diff(o, eng)
            info["row_level_check_max_abs_diff_vs_record"] = terms_diff(o, rec.get("row_level_check", {}))
            if max(info["final_max_abs_diff_vs_record"], info["engine_vs_row_level_max_abs_diff"],
                   info["row_level_check_max_abs_diff_vs_record"]) > OBJ_TOL_B:
                fails.append("final terms")
            own_terms[(k, cid)] = o
            # permutation-null MI (diagnostic): 100 fixed permutations of DEFENSE_FIT SEX
            T1 = Pol(fines[1], tokens_from_lab(labs[1])).T          # the policy alphabet (incl. never-emitted fallback)
            T2 = Pol(fines[2], tokens_from_lab(labs[2])).T
            c1, c2 = toks[1][fit], toks[2][fit]
            c12 = c1 * T2 + c2
            pn = rec.get("perm_null_mi_fit", {})
            best = None
            for rule in ("gather", "scatter"):
                vals = {"I1": [], "I2": [], "I12": []}
                for pm in perms:
                    if rule == "gather":
                        sp = s_fit[pm]
                    else:
                        sp = np.empty_like(s_fit)
                        sp[pm] = s_fit
                    vals["I1"].append(mi_vec(sp, c1, T1))
                    vals["I2"].append(mi_vec(sp, c2, T2))
                    vals["I12"].append(mi_vec(sp, c12, T1 * T2))
                summ = {x: {"mean": float(np.mean(v)), "q95": float(np.quantile(v, 0.95)), "max": float(np.max(v))}
                        for x, v in vals.items()}
                dmax = max(abs(summ[x][st_] - float(pn.get(x, {}).get(st_, np.nan))) for x in summ for st_ in summ[x])
                if best is None or dmax < best[1]:
                    best = (rule, dmax, summ)
                if dmax <= 1e-12:
                    break
            info["perm_null_rule"], info["perm_null_max_abs_diff"] = best[0], best[1]
            info["perm_null_own"] = best[2]
            perm_rule[best[0] if best[1] <= 1e-12 else "neither"] += 1
            # sparsity receipts (diagnostic): occupied-only and alphabet-wide conventions both computed
            sp_ = rec.get("sparsity_fit", {})
            _, pinv = np.unique(c12, return_inverse=True)
            pcnt = np.bincount(pinv.ravel())

            def spr(cnt_occ, alphabet):
                return {"alphabet": int(alphabet), "occupied": int(len(cnt_occ)),
                        "singleton_cells": int(np.sum(cnt_occ == 1)),
                        "lt5_cells_occupied": int(np.sum(cnt_occ < 5)),
                        "lt5_cells_with_unseen": int(np.sum(cnt_occ < 5) + (alphabet - len(cnt_occ))),
                        "entropy_nats": float(-np.sum(cnt_occ / cnt_occ.sum() * np.log(cnt_occ / cnt_occ.sum())))}
            b1, b2 = np.bincount(c1, minlength=T1), np.bincount(c2, minlength=T2)
            own_sp = {"r1": spr(b1[b1 > 0], T1), "r2": spr(b2[b2 > 0], T2), "pair": spr(pcnt, T1 * T2)}
            sp_eq, conv_used = True, set()
            for v in own_sp:
                rv = sp_.get(v, {})
                for k_ in ("alphabet", "occupied", "singleton_cells"):
                    sp_eq &= rv.get(k_) == own_sp[v][k_]
                if rv.get("lt5_cells") == own_sp[v]["lt5_cells_occupied"]:
                    conv_used.add("occupied")
                elif rv.get("lt5_cells") == own_sp[v]["lt5_cells_with_unseen"]:
                    conv_used.add("with_unseen")
                else:
                    sp_eq = False
                sp_eq &= abs(float(rv.get("entropy_nats", np.nan)) - own_sp[v]["entropy_nats"]) <= 1e-12
            info["sparsity_own"] = own_sp
            info["sparsity_equal"] = bool(sp_eq)
            info["sparsity_lt5_convention"] = sorted(conv_used)
            info["diagnostic_warnings"] = [x for x, ok_ in (("sparsity receipt", sp_eq),
                                                             ("perm-null MI", best[1] <= 1e-12)) if not ok_]
            info["failures"] = fails
            compact = {kk: vv for kk, vv in info.items() if kk not in ("stages", "starts")} if not fails else info
            if fam == "JOINT" or fam.startswith("SEQ") or fam in ("FINE-TASK", "LOCAL"):
                compact["search_summary"] = {
                    "merges_checked": sum(c["merges"]["n_recorded"] for c in (info.get("stages") or info.get("starts") or {}).values()
                                          if "merges" in c),
                    "moves_checked": sum(c["moves"]["n_recorded"] for c in (info.get("stages") or info.get("starts") or {}).values()
                                         if "moves" in c),
                    "max_abs_diff": max([0.0] + [max(c["merges"]["max_abs_diff"], c["moves"]["max_abs_diff"])
                                                 for c in (info.get("stages") or info.get("starts") or {}).values()
                                                 if "merges" in c])}
            unit_out[f"{cid}#s{k}"] = res("FAIL" if fails else ("WARN" if info["diagnostic_warnings"] else "PASS"),
                                          **compact)
            own_maps[(k, cid)] = {"tok1": toks[1], "tok2": toks[2], "q1": qs[1], "q2": qs[2], "hard1": hards[1],
                                  "hard2": hards[2], "alpha1": int(T1), "alpha2": int(T2)}
            OWN_REL[(k, cid)] = own_maps[(k, cid)]
    n_units = len(unit_out)
    fails_u = sorted(k_ for k_, v in unit_out.items() if v["status"] == "FAIL")
    warns_u = sorted(k_ for k_, v in unit_out.items() if v["status"] == "WARN")
    joint = {k_: v for k_, v in unit_out.items() if "|JOINT|" in k_}
    summary = {"units": n_units, "failing": fails_u, "diagnostic_warnings": warns_u,
               "merges_checked": sum(v.get("search_summary", {}).get("merges_checked", 0) for v in unit_out.values()),
               "moves_and_sweep_merges_checked": sum(v.get("search_summary", {}).get("moves_checked", 0)
                                                    for v in unit_out.values()),
               "max_abs_diff_recorded_vs_own_increments": max([0.0] + [v.get("search_summary", {}).get("max_abs_diff", 0.0)
                                                                     for v in unit_out.values()]),
               "joint_winner_labels": {k_: v.get("winner_label") for k_, v in joint.items()},
               "joint_unresolved_local_optima": {k_: v.get("unresolved_local_optima") for k_, v in joint.items()},
               "joint_starts_not_converged": {k_: v.get("starts_not_converged") for k_, v in joint.items()},
               "perm_null_rule_counts": perm_rule, "own_cpu_s": cpu}
    return {"stage_b_fine_partitions": res(worst(*[v["status"] for v in fine_out.values()]) if len(fine_out) == 3
                                           else "FAIL", units=fine_out),
            "stage_b_search": res("FAIL" if fails_u or n_units != 42 else ("WARN" if warns_u else "PASS"), **summary,
                                  per_unit=unit_out),
            "stage_b_class_preservation": res("PASS" if not cp_fail and not adv_fail and n_units == 42 else "FAIL",
                                              row_checks_passed_by_role=cp_rows, failures=cp_fail,
                                              adversarial_failures=adv_fail)}, own_maps, own_terms


# ================================================================================================ PHASE 2B / 3
VIEWS = ("v1", "v2", "pair")
SLATE_15 = [f"LR_C{C}" for C in (0.01, 0.1, 1.0, 10.0, 100.0)] + ["MLP_64", "MLP_128", "MLP_64x64", "MLP_128x128"] + \
    [f"HGB_{lr}_{lv}" for lr in (0.05, 0.1) for lv in (15, 31)] + ["DA_LR", "DA_MLP"]
CC_LOCAL = [f"CC_alpha{a}" for a in (0.5, 1.0, 5.0)]
CC_PAIR = [f"CCpair_alpha{a}" for a in (0.5, 1.0, 5.0)]
NONJOINT = ("DIRECT-TASK", "FINE-TASK", "LOCAL", "SEQ-12", "SEQ-21")


def all_cids():
    out = [direct_id(m1, m2) for m1, m2 in rate_bank()] + ["U|FINE-TASK|i8o64"]
    for lam in LAMS:
        out += [b_config(f, lam) for f in ("LOCAL", "SEQ-12", "SEQ-21", "JOINT")]
    return out + ["U|CLASS|i1o1", "SRC|U", "SRC|RAW-J_b0.3", "REF|E", "REF|F", "REF|F0"]


def cid_family(cid):
    if cid.startswith("SRC|"):
        return "SRC"
    if cid.startswith("REF|"):
        return "REF"
    return cid.split("|")[1]


def release_unit(k, cid):
    if cid.startswith("SRC|"):
        return f"tea__s{k}__{cid.split('|')[1]}"
    if cid.startswith("REF|"):
        return f"ref__s{k}__{cid.split('|')[1]}"
    return f"pol__s{k}__{cid.replace('|', '_')}"


def safe_label(label):
    return label.replace("|", "_").replace("*", "star").replace("/", "_").replace(" ", "_")


def my_ce(y, p1):
    y = np.asarray(y)
    pt = np.where(y == 1, np.asarray(p1, dtype=np.float64), 1.0 - np.asarray(p1, dtype=np.float64))
    return float(-np.mean(np.log(np.clip(pt, LL_CLIP, 1.0))))


def pick_first(vals, maximize):
    """Bank-order selection: a later candidate replaces the incumbent only when better by more than 1e-12."""
    best = None
    for j, v in enumerate(vals):
        if best is None or (v > vals[best] + 1e-12 if maximize else v < vals[best] - 1e-12):
            best = j
    return best


def bank_rows(keys, w):
    v = lambda view: [(k_.split(":", 1)[1], k_) for k_ in keys if k_.split(":", 1)[0] == view]  # noqa: E731
    if w in ("v1", "v2"):
        return [("own", w, a, k_) for a, k_ in v(w)]
    return [("coalition", "pair", a, k_) for a, k_ in v("pair")] + \
        [("ignore_recipient_2", "v1", a, k_) for a, k_ in v("v1")] + [("ignore_recipient_1", "v2", a, k_) for a, k_ in v("v2")]


def own_release(k, cid, T, refs):
    if cid.startswith("SRC|"):
        t = T[(k, cid.split("|")[1])]
        return {"p1": t["p1"], "p2": t["p2"], "hard1": t["d1"], "hard2": t["d2"], "finite": False}
    if cid.startswith("REF|"):
        r = refs[(cid.split("|")[1], k)]
        return {"p1": r["p1"], "p2": r["p2"], "hard1": r["d1"], "hard2": r["d2"], "finite": False}
    o = OWN_REL[(k, cid)]
    return {"p1": o["q1"], "p2": o["q2"], "hard1": o["hard1"], "hard2": o["hard2"], "tok1": o["tok1"], "tok2": o["tok2"],
            "alpha1": o["alpha1"], "alpha2": o["alpha2"], "finite": True}


def check_inner(D: Data, L, T, refs):
    """Every inner unit: own AUC / CE of every stored candidate on INNER_SELECTION SEX, own banks and dual selection,
    the slate, the selected attacker's per-seed recovery (seed-0 refit identical to the bank member), own INNER
    utility from own releases, decision preservation, token states, coverage; then the composed SRC|U bank over every
    fitted code. Returns (check, own table {(k, cid): {...}})."""
    sel, fa = D.idx[INNER], D.idx["AUDIT_FIT"]
    ys = L["sex"][sel]
    assert (ys >= 0).all()
    fit = D.fit_idx
    const = {1: fitting_constant(L["y_income"][fit], 2), 2: fitting_constant(L["y_occ"][fit], 6)}
    y_sel = {1: L["y_income"][sel], 2: L["y_occ"][sel]}
    own, fails, n_cand = {}, [], 0
    mx = {"selected_value": 0.0, "table": 0.0, "recovery": 0.0, "utility": 0.0}
    slate_bad, sel0_bad, cov_bad = [], [], []
    for k in SEEDS:
        uU = {i: my_utility(T[(k, "U")][f"p{i}"][sel], T[(k, "U")][f"d{i}"][sel], y_sel[i], KS[i - 1], const[i]) for i in (1, 2)}
        for cid in all_cids():
            name = f"inner__{release_unit(k, cid)}"
            ud = UNITS / name
            if not unit_done(name):
                fails.append(f"{name}: missing")
                continue
            cu, _ = complete_ok(ud)
            if not (cu["id_ok"] and cu["rehash_ok"] and not cu["unlisted"]):
                fails.append(f"{name}: custody")
            rec = jload(ud / "record.json")
            z = np.load(ud / "inner_preds.npz", allow_pickle=False)
            if not np.array_equal(z["sel_row_id"], D.row_id[sel]):
                fails.append(f"{name}: INNER rows differ")
            rel = own_release(k, cid, T, refs)
            fams = [rec["primary_family"]] + [f_ for f_ in rec.get("families", {})]
            mine = {}
            for fam in fams:
                fr = rec["recovery"] if fam == rec["primary_family"] else rec["families"][fam]
                keys = [str(x) for x in z[f"keys_{fam}"]]
                P = z[f"P_{fam}"]
                finite = bool(fr.get("finite", fam in ("code", "decisions", "cells")))
                for w in VIEWS:
                    want = SLATE_15 + ((CC_PAIR if w == "pair" else CC_LOCAL) if finite else [])
                    got = [k_.split(":", 1)[1] for k_ in keys if k_.split(":", 1)[0] == w]
                    if got != want:
                        slate_bad.append(f"{name}/{fam}/{w}")
                aucs = {k_: my_auc(ys, P[j]) for j, k_ in enumerate(keys)}
                ces = {k_: my_ce(ys, P[j]) for j, k_ in enumerate(keys)}
                n_cand += len(keys)
                mf = {"auc0": {}, "ce0": {}, "sel": {}, "ce_sel": {}, "auc": {}, "ce": {}, "auc_per_seed": {}}
                for w in VIEWS:
                    rows = bank_rows(keys, w)
                    va, vc = [aucs[r_[3]] for r_ in rows], [ces[r_[3]] for r_ in rows]
                    ia, ic = pick_first(va, True), pick_first(vc, False)
                    la, lc = "%s:%s:%s" % rows[ia][:3], "%s:%s:%s" % rows[ic][:3]
                    mf["auc0"][w], mf["ce0"][w], mf["sel"][w], mf["ce_sel"][w] = va[ia], vc[ic], la, lc
                    rs = fr.get("selection", {}).get(w, {})
                    if rs.get("auc", {}).get("label") != la or rs.get("ce", {}).get("label") != lc:
                        fails.append(f"{name}/{fam}/{w}: selected {la}/{lc} vs {rs.get('auc', {}).get('label')}/"
                                     f"{rs.get('ce', {}).get('label')}")
                    mx["selected_value"] = max(mx["selected_value"], abs(va[ia] - float(rs.get("auc", {}).get("inner_auc", np.nan))),
                                               abs(vc[ic] - float(rs.get("ce", {}).get("inner_ce", np.nan))))
                    tab = {r_["pred_key"] if "pred_key" in r_ else r_["view"] + ":" + r_["attacker"]: r_ for r_ in fr.get("tables", {}).get(w, [])}
                    for r_ in rows:
                        t_ = tab.get(r_[3])
                        if t_ is None:
                            fails.append(f"{name}/{fam}/{w}: table row {r_[3]} absent")
                            continue
                        mx["table"] = max(mx["table"], abs(t_["inner_auc"] - aucs[r_[3]]), abs(t_["inner_ce"] - ces[r_[3]]))
                    # recovery of the selected attackers (seed-0 = the bank member)
                    SA, SC = z[f"SEL_{fam}_auc_{w}"], z[f"SEL_{fam}_ce_{w}"]
                    if not (bitwise(SA[0], P[keys.index(rows[ia][3])]) and bitwise(SC[0], P[keys.index(rows[ic][3])])):
                        sel0_bad.append(f"{name}/{fam}/{w}")
                    aps = [my_auc(ys, SA[s_]) for s_ in range(3)]
                    cps = [my_ce(ys, SC[s_]) for s_ in range(3)]
                    mf["auc"][w], mf["ce"][w], mf["auc_per_seed"][w] = float(np.mean(aps)), float(np.mean(cps)), aps
                    if cid != "SRC|U":                         # every SRC|U family is composed (checked below)
                        mx["recovery"] = max(mx["recovery"], abs(mf["auc"][w] - fr["auc"][w]), abs(mf["ce"][w] - fr["ce"][w]))
                mine[fam] = mf
            # utility, preservation, token states, coverage
            ut = {i: my_utility(rel[f"p{i}"][sel], rel[f"hard{i}"][sel], y_sel[i], KS[i - 1], const[i]) for i in (1, 2)}
            for i, t_ in ((1, "income"), (2, "occupation")):
                for q_ in ("acc", "logloss", "brier", "const_acc"):
                    mx["utility"] = max(mx["utility"], abs(ut[i][q_] - rec["utility"][t_][q_]))
            pres = {i: bool(np.array_equal(rel[f"hard{i}"], (T[(k, "U")] if not cid.startswith(("SRC|", "REF|")) else
                                                              {"d1": rel["hard1"], "d2": rel["hard2"]})[f"d{i}"]))
                    for i in (1, 2)}
            if {str(i): pres[i] for i in (1, 2)} != {kk: bool(v) for kk, v in rec.get("preserved", {}).items()}:
                fails.append(f"{name}: preserved flags")
            states = (rel["alpha1"] + rel["alpha2"]) if rel["finite"] else None
            if (rec.get("token_states") is None) != (states is None) or (states is not None and int(rec["token_states"]) != states):
                fails.append(f"{name}: token states")
            if rel["finite"]:
                cov = rec["recovery"].get("coverage", {})
                for i, w in ((1, "v1"), (2, "v2")):
                    seen = np.isin(rel[f"tok{i}"][sel], np.unique(rel[f"tok{i}"][fa]))
                    if cov.get(w, {}).get("inner_rows_fallback") != int((~seen).sum()) or \
                            cov.get(w, {}).get("tokens_in_fit") != int(len(np.unique(rel[f"tok{i}"][fa]))):
                        cov_bad.append(f"{name}/{w}")
            own[(k, cid)] = {"fams": mine, "primary": rec["primary_family"], "utility": ut, "U": uU, "preserved": pres,
                             "states": states, "rec": rec}
    # composed SRC|U bank: own composition over every fitted code of the same seed (bank order = lock order)
    EL = jload(RES / "EVALUATION_LOCK.json") if (RES / "EVALUATION_LOCK.json").exists() else None
    comp_bad, comp_mx, comp_out = [], 0.0, {}
    for k in SEEDS:
        o = own.get((k, "SRC|U"))
        if o is None:
            continue
        units = (o["rec"].get("composed") or {}).get("policies") or (EL["seeds"][str(k)]["composed_policies"] if EL else [])
        pol_cids = []
        for u_ in units:
            u_ = u_ if isinstance(u_, str) else (u_.get("cid") or u_.get("unit"))
            cc_ = u_ if u_ in all_cids() else next((c_ for c_ in all_cids() if release_unit(k, c_) == u_), None)
            pol_cids.append(cc_)
        rc = o["rec"]
        per_family = {}
        for fam in [o["primary"]] + [f_ for f_ in rc.get("families", {})]:
            mf = o["fams"][fam]
            cands = [c_ for c_ in pol_cids if c_ and (fam != "decisions" or c_ == "U|CLASS|i1o1")]
            res_f = {"auc": {}, "ce": {}, "winner": {}, "ce_winner": {}}
            for w in VIEWS:
                best_a, win_a, rec_a = mf["auc0"][w], "source", mf["auc"][w]
                best_c, win_c, rec_c = mf["ce0"][w], "source", mf["ce"][w]
                for c_ in cands:
                    pm = own[(k, c_)]["fams"]["code"]
                    if pm["auc0"][w] > best_a + 1e-12:
                        best_a, win_a, rec_a = pm["auc0"][w], c_, pm["auc"][w]
                    if pm["ce0"][w] < best_c - 1e-12:
                        best_c, win_c, rec_c = pm["ce0"][w], c_, pm["ce"][w]
                res_f["auc"][w], res_f["ce"][w], res_f["winner"][w], res_f["ce_winner"][w] = rec_a, rec_c, win_a, win_c
            per_family[fam] = res_f
        prim = per_family[o["primary"]]
        recd = rc.get("composed", {})
        comp_out[f"s{k}"] = {"winner": prim["winner"], "ce_winner": prim["ce_winner"], "auc": prim["auc"]}
        for w in VIEWS:
            comp_mx = max(comp_mx, abs(prim["auc"][w] - recd.get("auc", {}).get(w, np.nan)),
                          abs(prim["ce"][w] - recd.get("ce", {}).get(w, np.nan)),
                          abs(prim["auc"][w] - rc["recovery"]["auc"][w]))
            if recd.get("winner", {}).get(w) != prim["winner"][w] or recd.get("ce_winner", {}).get(w) != prim["ce_winner"][w]:
                comp_bad.append(f"SRC|U#s{k}/{w}: winner {prim['winner'][w]} vs {recd.get('winner', {}).get(w)}")
        rpf = recd.get("per_family", {})
        for fam, res_f in per_family.items():
            fr = rc["recovery"] if fam == o["primary"] else rc["families"][fam]
            for w in VIEWS:
                if rpf.get(fam, {}).get("winner", {}).get(w) != res_f["winner"][w] or \
                        rpf.get(fam, {}).get("ce_winner", {}).get(w) != res_f["ce_winner"][w]:
                    comp_bad.append(f"SRC|U#s{k}/{fam}/{w}: family winner {res_f['winner'][w]} vs "
                                    f"{rpf.get(fam, {}).get('winner', {}).get(w)}")
                comp_mx = max(comp_mx, abs(res_f["auc"][w] - fr["auc"][w]), abs(res_f["ce"][w] - fr["ce"][w]))
            if fam != o["primary"]:
                o["fams"][fam]["auc"], o["fams"][fam]["ce"] = dict(res_f["auc"]), dict(res_f["ce"])
        o["composed"] = per_family
        o["fams"][o["primary"]]["auc"] = dict(prim["auc"])          # the composed recovery is the U row's recovery
        o["fams"][o["primary"]]["ce"] = dict(prim["ce"])
        n_pol = len([c_ for c_ in pol_cids if c_])
        comp_out[f"s{k}"]["policies_composed"] = n_pol
        comp_out[f"s{k}"]["per_family_winners"] = {fam: v["winner"] for fam, v in per_family.items()}
        own_freeze = sorted({v_ for f_ in per_family.values() for crit in ("winner", "ce_winner")
                             for v_ in f_[crit].values() if v_ != "source"})
        rf = recd.get("freeze") or []
        rf = sorted({(x if isinstance(x, str) else (x.get("cid") or x.get("unit"))) for x in rf})
        rf = sorted({(x if x in all_cids() else next((c_ for c_ in all_cids() if release_unit(k, c_) == x), x)) for x in rf})
        comp_out[f"s{k}"]["freeze_own"] = own_freeze
        if rf != own_freeze:
            comp_bad.append(f"SRC|U#s{k}: freeze {rf} vs own {own_freeze}")
        expected = 22
        if n_pol != expected or sorted(c_ for c_ in pol_cids if c_) != sorted(
                [c_ for c_ in all_cids() if not c_.startswith(("SRC|", "REF|"))]):
            comp_bad.append(f"SRC|U#s{k}: composed over {n_pol} codes (closure)")
    tol_ok = mx["selected_value"] <= 1e-12 and mx["table"] <= 1e-12 and mx["recovery"] <= 1e-12 and mx["utility"] <= 1e-12
    bad = fails or slate_bad or sel0_bad or cov_bad or not tol_ok
    inner = res("FAIL" if bad else "PASS", units=len(own), candidates_recomputed=n_cand,
                max_abs_diff=mx, slate_incomplete=slate_bad[:20], seed0_selected_not_bank_member=sel0_bad[:20],
                coverage_mismatches=cov_bad[:20], failures=fails[:30],
                rule="own Mann-Whitney AUC and 1e-12-clipped CE of every stored candidate; v_i bank = own slate (+ CC on "
                     "finite views); pair bank = coalition (+ CCpair) + ignore-recipient-2 (v1) + ignore-recipient-1 (v2); "
                     "first maximum / minimum beyond 1e-12; recovery = mean over attacker seeds 0-2 of the selected "
                     "attacker; utility from own releases")
    comp = res("FAIL" if comp_bad or comp_mx > 1e-12 else "PASS", per_seed=comp_out, max_abs_diff=comp_mx,
               failures=comp_bad[:20], rule="per family and view the winner is the first bank (own, then every fitted "
               "code of the seed in lock order) whose seed-0 selected value beats the running best by > 1e-12; "
               "decisions compose only with the class-only code; recovery = the winner's seed 0-2 mean")
    return inner, comp, own


def my_selection(own):
    """Own implementation of PROTOCOL section 11 on own inner values."""
    rows = {}
    for cid in all_cids():
        seeds = {}
        for k in SEEDS:
            o = own[(k, cid)]
            a = o["fams"][o["primary"]]["auc"]
            g_ = {i: gate_task(o["utility"][i], o["U"][i]) for i in (1, 2)}
            el = all(g_[i]["eligible"] for i in (1, 2)) and all(o["preserved"].values())
            seeds[k] = {"auc": a, "eligible": el, "norm_excess": max(g_[i]["normalized_excess"] for i in (1, 2)),
                        "sum_ll": o["utility"][1]["logloss"] + o["utility"][2]["logloss"],
                        "states": float("inf") if o["states"] is None else float(o["states"])}
        rows[cid] = {"config": cid, "family": cid_family(cid), "seeds": seeds,
                     "mean_pair": float(np.mean([seeds[k]["auc"]["pair"] for k in SEEDS])),
                     "mean_sum_logloss": float(np.mean([seeds[k]["sum_ll"] for k in SEEDS])),
                     "mean_states": float(np.mean([seeds[k]["states"] for k in SEEDS])),
                     "eligible": all(seeds[k]["eligible"] for k in SEEDS),
                     "worst_norm_excess": max(seeds[k]["norm_excess"] for k in SEEDS)}

    def okey(r):
        return (round(r["mean_pair"], 12), round(r["mean_sum_logloss"], 12), r["mean_states"], r["config"])

    def guard_excess(r, guards):
        ex = [r["seeds"][k]["auc"][w] - (gr["seeds"][k]["auc"][w] + BUFFER) for gr in guards for k in SEEDS
              for w in ("v1", "v2")]
        return max(ex) if ex else float("-inf")

    def pick(cands, guards=None, nominee_role=True, guard_missing=False):
        ev = []
        for c_ in cands:
            r = rows[c_]
            ge = guard_excess(r, guards or [])
            ok = r["eligible"] and ge <= 0 and not guard_missing
            fk = (0.0 if r["eligible"] else round(r["worst_norm_excess"], 12), round(max(ge, 0.0), 12)) + okey(r)
            ev.append({"config": c_, "eligible": r["eligible"], "guard_ok": ge <= 0, "nominable": ok, "fallback_key": fk})
        nom = [e for e in ev if e["nominable"]]
        if nom:
            return {"status": "NOMINEE", "config": min(nom, key=lambda e: okey(rows[e["config"]]))["config"], "evaluated": ev}
        if guard_missing and any(e["eligible"] and e["guard_ok"] for e in ev):
            return {"status": "INVALID_NOMINEE", "config": None, "evaluated": ev,
                    "descriptive_config": min(ev, key=lambda e: e["fallback_key"])["config"]}
        st = "NO_ELIGIBLE_NOMINEE" if nominee_role else "NO_ELIGIBLE_COMPARATOR"
        return {"status": st, "config": None, "evaluated": ev,
                "descriptive_config": min(ev, key=lambda e: e["fallback_key"])["config"] if ev else None}

    cids = all_cids()
    direct = [c_ for c_ in cids if cid_family(c_) == "DIRECT-TASK"]
    el_direct = [c_ for c_ in direct if rows[c_]["eligible"]]
    if el_direct:
        Q = {"status": "NOMINEE", "config": min(el_direct, key=lambda c_: (rows[c_]["worst_norm_excess"],
                                                                           rows[c_]["mean_states"], c_))}
    else:
        Q = {"status": "NO_ELIGIBLE_NOMINEE", "config": None}
    nonjoint = [c_ for c_ in cids if cid_family(c_) in NONJOINT]
    C_global = pick(nonjoint + [c_ for c_ in cids if cid_family(c_) in ("CLASS", "SRC", "REF")], nominee_role=False)
    T_star = pick(direct + ["U|FINE-TASK|i8o64", "SRC|U", "SRC|RAW-J_b0.3", "U|CLASS|i1o1", "REF|F0"], nominee_role=False)
    joint = [c_ for c_ in cids if cid_family(c_) == "JOINT"]
    cell = "i8o64"
    C_rate = pick([c_ for c_ in nonjoint if c_.split("|")[2] == cell], nominee_role=False)
    C_rate["cell"] = cell
    gm = [x for x in (C_rate, C_global) if x["status"] != "NOMINEE"]
    J = pick(joint, guards=[rows[x["config"]] for x in (C_rate, C_global) if x["status"] == "NOMINEE"],
             guard_missing=bool(gm))
    priv = [c_ for c_ in cids if cid_family(c_) in PRIV_FAMS]
    P = pick(priv, guards=[rows[T_star["config"]]] if T_star["status"] == "NOMINEE" else [],
             guard_missing=T_star["status"] != "NOMINEE")
    if P["status"] == "NOMINEE":
        P["winning_family"] = cid_family(P["config"])
    st = {"Q*": Q, "C_global": C_global, "T*": T_star, "J*": J, "C_rate": C_rate, "P*": P}
    resolved = {kk: v.get("config") or v.get("descriptive_config") for kk, v in st.items()}
    return {"rows": rows, "statuses": st, "resolved": resolved}


def check_selection(own):
    mine = my_selection(own)
    S = jload(RUN / "selection.json")
    pub = jload(RES / "SELECTION.json") if (RES / "SELECTION.json").exists() else {}
    diffs, row_mx, elig = [], 0.0, []
    for x, v in S["statuses"].items():
        m = mine["statuses"][x]
        for f_ in ("status", "config", "descriptive_config", "winning_family", "cell"):
            if (v.get(f_) or None) != (m.get(f_) or None):
                diffs.append(f"{x}.{f_}: {v.get(f_)} vs own {m.get(f_)}")
        ev_r = {e["config"]: e for e in v.get("evaluated", [])}
        for e in m.get("evaluated", []):
            r_ = ev_r.get(e["config"])
            if r_ is None or r_["eligible"] != e["eligible"] or r_["guard_ok"] != e["guard_ok"] or \
                    r_["nominable"] != e["nominable"]:
                diffs.append(f"{x} evaluated {e['config']}")
    if S.get("resolved") != mine["resolved"]:
        diffs.append("resolved")
    for cid, r_ in S["rows"].items():
        m = mine["rows"][cid]
        for f_ in ("mean_pair", "mean_v1", "mean_v2", "mean_sum_logloss", "worst_norm_excess"):
            if f_ in r_ and f_ in m:
                row_mx = max(row_mx, abs(float(r_[f_]) - m[f_]))
        if bool(r_["eligible"]) != m["eligible"]:
            elig.append(cid)
        for k in SEEDS:
            for w in VIEWS:
                row_mx = max(row_mx, abs(r_["seeds"][str(k)]["auc"][w] - m["seeds"][k]["auc"][w]))
    pub_ok = (not pub) or all(pub.get("statuses", {}).get(x, {}).get("status") == S["statuses"][x]["status"] and
                              pub.get("statuses", {}).get(x, {}).get("config") == S["statuses"][x].get("config")
                              for x in S["statuses"])
    ok = not diffs and not elig and row_mx <= 1e-12 and pub_ok
    st = {x: {kk: v.get(kk) for kk in ("status", "config", "descriptive_config", "winning_family", "cell")}
          for x, v in mine["statuses"].items()}
    return res("PASS" if ok else "FAIL", own_statuses=st, own_resolved=mine["resolved"], differences=diffs[:20],
               eligibility_differences=elig, max_abs_diff_rows=row_mx, public_SELECTION_json_consistent=pub_ok,
               eligible_candidates=sorted(c_ for c_, r_ in mine["rows"].items() if r_["eligible"]),
               rule="per seed gate vs U of the same seed (unchanged contract) and exact decision preservation; "
                    "eligible = every seed; order (mean pair AUC, mean summed log loss, mean states, id; 12-decimal "
                    "rounding); guards local AUC <= guard + 0.005 per recipient and seed; fallback (0 / worst normalised "
                    "excess, guard shortfall, order)"), mine


def unsealed_labels(D: Data, gate):
    """Assessment labels, read ONLY after evaluation_lock_gate() verified the committed lock on origin."""
    if not gate.get("ok"):
        raise RuntimeError("REFUSED: EVALUATION_LOCK not verified on origin; assessment labels stay sealed")
    z = np.load(SRC, allow_pickle=False)
    return {k: z[v].astype(np.int64)[D.keep].copy() for k, v in LABEL_KEYS.items()}


def check_eval_lock(D: Data, L, gate, sel_mine):
    """EVALUATION_LOCK content vs own selection, the lock / amendment / selection hashes, the scored list rule, the
    unit file hashes, the fitting SEX prior and the assessment role."""
    EL = jload(RES / "EVALUATION_LOCK.json")
    out, bad = {"gate": gate}, []
    st_m = sel_mine["statuses"]
    out["statuses_equal_own"] = all((EL["statuses"][x].get("status"), EL["statuses"][x].get("config"),
                                     EL["statuses"][x].get("descriptive_config")) ==
                                    (st_m[x]["status"], st_m[x].get("config"), st_m[x].get("descriptive_config"))
                                    for x in st_m)
    out["resolved_equal_own"] = EL["resolved"] == sel_mine["resolved"]
    # scored list (PROTOCOL 13): nominees/comparators (or fallbacks), Q*, U, RAW-J, CLASS, all 8 DIRECT-TASK rates,
    # FINE-TASK / LOCAL / SEQ-12 / SEQ-21 / JOINT at the J* cell (J* fallback), FARE, F0, LEACE
    want = set(sel_mine["resolved"].values()) | {"SRC|U", "SRC|RAW-J_b0.3", "U|CLASS|i1o1", "REF|E", "REF|F", "REF|F0"}
    want |= {direct_id(m1, m2) for m1, m2 in rate_bank()}
    jc = sel_mine["resolved"]["J*"]
    lam = jc.split("|l")[1] if jc and "|l" in jc else None
    want |= {"U|FINE-TASK|i8o64"} | ({b_config(f_, float(lam)) for f_ in PRIV_FAMS} if lam else set())
    out["scored_labels_equal_own_rule"] = sorted(EL["scored_labels"]) == sorted(want)
    out["scored_labels"] = len(EL["scored_labels"])
    out["locks_sha256_ok"] = all((RES / n).exists() and sha_file(RES / n) == h for n, h in EL["locks_sha256"].items()) and \
        sorted(EL["locks_sha256"]) == sorted(f"{n}.json" for n in LOCK_ORDER[:-1])
    out["amendments_sha256_ok"] = all(sha_file(RES / n) == h for n, h in EL["amendments_sha256"].items()) and \
        sorted(EL["amendments_sha256"]) == sorted(q.name for q in RES.glob("AMENDMENT*.json"))
    out["selection_sha256_ok"] = EL["selection_sha256"] == sha_file(RUN / "selection.json") and \
        EL.get("selection_public_sha256") == sha_file(RES / "SELECTION.json")
    out["capacity_gate_sha256_ok"] = EL.get("capacity_gate_sha256") in (sha_file(RES / "CAPACITY_GATE.json"),
                                                                       sha_file(RUN / "gate.json"))
    lc = EL["locked_code_files"]
    out["locked_code_equals_worktree"] = [f_ for f_, h in lc.items() if not ((WT / f_).exists() and sha_file(WT / f_) == h)]
    fit = D.fit_idx
    ph = hashlib.sha256(np.bincount(L["sex"][fit], minlength=2).astype(np.int64).tobytes()).hexdigest()
    out["sex_prior_hash_ok"] = ph == EL["sex_prior_defense_fit_sha256"]
    am = D.mask[ASSESS]
    ar = EL["assessment_role"]
    out["assessment_role_ok"] = ar["rows"] == int(am.sum()) and ar["groups"] == len(np.unique(D.unit[am])) and \
        ar["row_id_sha256"] == rowid_hash(D.row_id[am])
    files_ok, n_units = True, 0
    for k in SEEDS:
        se = EL["seeds"][str(k)]
        if list(se["score"]) != list(EL["scored_labels"]) or se["u_label"] != "SRC|U":
            files_ok = False
        for u_, h in se["unit_file_sha256"].items():
            n_units += 1
            cj = jload(UNITS / u_ / "COMPLETE.json")["files"]
            if h != cj or not all(sha_file(UNITS / u_ / f_) == hh for f_, hh in cj.items()):
                files_ok = False
        exp_comp = [release_unit(k, c_) for c_ in all_cids() if not c_.startswith(("SRC|", "REF|"))]
        if sorted(se["composed_policies"]) != sorted(exp_comp):
            files_ok = False
    out.update({"unit_file_hashes_ok": files_ok, "units_hashed": n_units,
                "endpoints_ok": EL["endpoints"]["z"] == Z_PRIMARY and EL["endpoints"]["B"] == B_BOOT and
                EL["endpoints"]["boot_seed"] == BOOT_SEED and EL["endpoints"]["size"] == N_ENDPOINTS and
                EL["endpoints"]["primary"] == [f"P{i:02d}" for i in range(1, 38)],
                "stage_a_valid": EL.get("stage_a_valid") is True, "gate_met": EL.get("gate_met") is True,
                "technical_validity_ok": (EL.get("technical_validity") or {}).get("ok") is True,
                "refits_0_1_2": EL["attackers"]["refits"] == [0, 1, 2]})
    for kk, v in out.items():
        if isinstance(v, bool) and not v:
            bad.append(kk)
    if out["locked_code_equals_worktree"]:
        bad.append("locked code changed in worktree")
    if not gate.get("ok"):
        bad.append("gate")
    return res("FAIL" if bad else "PASS", failures=bad, **out), EL


def check_outer(D: Data, LU, T, refs, own, EL, gate):
    """Every outer unit: completed after the EVALUATION_LOCK push and bound to the lock commit / hash; rows, groups and
    labels equal the own unsealed labels; own releases (codes, U, RAW-J, references) bitwise on the assessment rows;
    own row losses; own AUC of every stored attacker prediction; the scored attacker is the inner-selected one (the
    composed winner for SRC|U); own coverage counts."""
    a_idx, f_idx = D.idx[ASSESS], D.idx["AUDIT_FIT"]
    fit = D.fit_idx
    const = {0: fitting_constant(LU["y_income"][fit], 2), 1: fitting_constant(LU["y_occ"][fit], 6)}
    push = gate.get("first_push_time")
    fails, preds, mx_l, mx_a, cov_bad, sel_bad = [], {}, 0.0, 0.0, [], []
    for k in SEEDS:
        U_rel = own_release(k, "SRC|U", T, refs)
        for lab in EL["scored_labels"]:
            name = f"outer__s{k}__{safe_label(lab)}"
            ud = UNITS / name
            f_ = []
            if not unit_done(name):
                fails.append(f"{name}: missing")
                continue
            cu, _ = complete_ok(ud)
            if not (cu["id_ok"] and cu["rehash_ok"] and not cu["unlisted"]):
                f_.append("custody")
            t_c = utc((ud / "COMPLETE.json").stat().st_mtime)
            if push is None or t_c < push:
                f_.append("completed before the EVALUATION_LOCK push")
            rec = jload(ud / "record.json")
            if rec["evaluation_lock"]["commit"] != gate.get("commit") or rec["evaluation_lock"]["sha256"] != gate.get("sha256"):
                f_.append("lock commit / sha")
            z = np.load(ud / "preds.npz", allow_pickle=False)
            p = {x: z[x] for x in z.files}
            chk = {"rows": bool(np.array_equal(p["assess_row_id"], D.row_id[a_idx])),
                   "groups": bool(np.array_equal(p["assess_unit"], D.unit[a_idx])),
                   "labels": all(np.array_equal(p[x], LU[y][a_idx]) for x, y in (("sex", "sex"), ("race", "race"),
                                                                              ("y_income", "y_income"), ("y_occ", "y_occ"))),
                   "const": [int(x) for x in p["const_class"]] == [const[0], const[1]]}
            o = own_release(k, lab, T, refs)
            for i, K in ((1, 2), (2, 6)):
                y = LU["y_income" if i == 1 else "y_occ"][a_idx]
                chk[f"release_{i}_bitwise"] = bitwise(np.asarray(o[f"p{i}"], np.float64)[a_idx], p[f"prob{i}"]) and \
                    bool(np.array_equal(o[f"hard{i}"][a_idx], p[f"hard{i}"]))
                chk[f"U_{i}_bitwise"] = bitwise(np.asarray(U_rel[f"p{i}"], np.float64)[a_idx], p[f"u_prob{i}"]) and \
                    bool(np.array_equal(U_rel[f"hard{i}"][a_idx], p[f"u_hard{i}"]))
                m_ = my_utility(p[f"prob{i}"], p[f"hard{i}"], y, K, const[i - 1])
                mx_l = max(mx_l, maxdiff(m_["ll_rows"], p[f"ll{i}"]), maxdiff(m_["br_rows"], p[f"br{i}"]))
            # recorded scored AUCs and the scored attacker = inner-selected attacker
            inner_o = own[(k, lab)]
            P_ok = True
            for fam, fr in rec["families"].items():
                for w in VIEWS:
                    key = f"P_auc_{w}" if fam == rec["primary_family"] else f"P_auc_{fam}_{w}"
                    P3 = p.get(key)
                    if P3 is None or P3.shape != (3, len(a_idx), 2) or not np.isfinite(P3).all() or \
                            np.abs(P3.sum(2) - 1).max() > 1e-12:
                        P_ok = False
                        continue
                    mine_auc = [my_auc(p["sex"], P3[s_][:, 1]) for s_ in range(3)]
                    sc = fr.get("scored", {}).get(w, {}).get("auc", {})
                    mx_a = max(mx_a, max(abs(x - y) for x, y in zip(mine_auc, sc.get("auc_per_seed", [np.nan] * 3))))
                    if lab == "SRC|U" and fam == rec["primary_family"]:
                        win = inner_o["composed"][fam]["winner"][w]
                        want = inner_o["fams"][fam]["sel"][w] if win == "source" else \
                            f"composed[{release_unit(k, win)}]:" + own[(k, win)]["fams"]["code"]["sel"][w]
                    elif lab == "SRC|U":
                        win = inner_o["composed"][fam]["winner"][w]
                        want = inner_o["fams"][fam]["sel"][w] if win == "source" else \
                            f"composed[{release_unit(k, win)}]:" + own[(k, win)]["fams"]["code"]["sel"][w]
                    else:
                        want = inner_o["fams"][fam]["sel"][w]
                    if sc.get("label") != want:
                        sel_bad.append(f"{name}/{fam}/{w}: scored {sc.get('label')} vs inner-selected {want}")
            chk["attacker_prediction_arrays_ok"] = P_ok
            if o.get("finite"):
                cv = rec["families"]["code"].get("coverage", {}).get("own", {})
                for i, w in ((1, "v1"), (2, "v2")):
                    seen = np.isin(o[f"tok{i}"][a_idx], np.unique(o[f"tok{i}"][f_idx]))
                    if cv.get(w, {}).get("scored_rows_fallback") != int((~seen).sum()):
                        cov_bad.append(f"{name}/{w}")
                pairs_fit = set(zip(o["tok1"][f_idx].tolist(), o["tok2"][f_idx].tolist()))
                if cv.get("pair", {}).get("pairs_in_fit") != len(pairs_fit):
                    cov_bad.append(f"{name}/pair")
            f_ += [kk for kk, v in chk.items() if isinstance(v, bool) and not v]
            if f_:
                fails.append(f"{name}: {f_}")
            preds[(k, lab)] = p
    ok = not fails and not sel_bad and not cov_bad and mx_l <= 1e-15 and mx_a <= 1e-12 and \
        len(preds) == 3 * len(EL["scored_labels"])
    return res("PASS" if ok else "FAIL", units=len(preds), failures=fails[:30], scored_attacker_mismatches=sel_bad[:20],
               coverage_mismatches=cov_bad[:20], max_abs_diff_row_losses=mx_l, max_abs_diff_scored_auc=mx_a), preds


class OwnBoot:
    """Paired exact-record-group bootstrap (registered convention): groups = sorted unique assessment group ids;
    rng = default_rng(20261007); replicate b = rng.multinomial(G, uniform), drawn sequentially; row weight = count of
    its group; evaluated in chunks of 250 replicates."""

    def __init__(self, groups, B=B_BOOT, seed=BOOT_SEED, chunk=250):
        self.u, self.inv = np.unique(np.asarray(groups), return_inverse=True)
        G_ = len(self.u)
        rng = np.random.default_rng(seed)
        p = np.full(G_, 1.0 / G_)
        self.counts = np.empty((G_, B), dtype=np.int32)
        for b in range(B):
            self.counts[:, b] = rng.multinomial(G_, p)
        self.B, self.chunk, self.n, self.G = B, chunk, len(self.inv), G_

    def chunks(self):
        for s0 in range(0, self.B, self.chunk):
            yield self.counts[self.inv, s0:s0 + self.chunk].astype(np.float64)


class WAuc:
    """Weighted Mann-Whitney AUC over many weight columns (ties 1/2)."""

    def __init__(self, score, pos):
        score = np.asarray(score, dtype=np.float64)
        self.order = np.argsort(score, kind="stable")
        ss = score[self.order]
        self.starts = np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])
        self.pos = np.asarray(pos, bool)[self.order][:, None]

    def __call__(self, W):
        Ws = W[self.order]
        gp = np.add.reduceat(np.where(self.pos, Ws, 0.0), self.starts, axis=0)
        gn = np.add.reduceat(np.where(self.pos, 0.0, Ws), self.starts, axis=0)
        below = np.cumsum(gn, 0) - gn
        den = gp.sum(0) * gn.sum(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(den > 0, (gp * (below + 0.5 * gn)).sum(0) / den, np.nan)


class WMean:
    def __init__(self, v):
        self.v = np.asarray(v, dtype=np.float64)

    def __call__(self, W):
        return (self.v @ W) / W.sum(0)


class Stats:
    """Own statistic graph: base statistics evaluated at unit weights and on every replicate (cached)."""

    def __init__(self, boot):
        self.boot, self.fn, self.val = boot, {}, {}

    def add(self, key, fn):
        self.fn.setdefault(key, fn)
        return key

    def evaluate(self, keys):
        todo = [k_ for k_ in dict.fromkeys(keys) if k_ not in self.val]
        pts = {k_: float(self.fn[k_](np.ones((self.boot.n, 1)))[0]) for k_ in todo}
        reps = {k_: [] for k_ in todo}
        for W in self.boot.chunks():
            for k_ in todo:
                reps[k_].append(self.fn[k_](W))
        for k_ in todo:
            self.val[k_] = (pts[k_], np.concatenate(reps[k_]))
        return {k_: self.val[k_] for k_ in keys}


def own_endpoints(preds, EL, levels=True):
    """Own 37 endpoints from the stored assessment predictions and the lock (PROTOCOL 12; PRIMARY_FAMILY slots are
    re-derived from the prompt definitions and only compared with the published slot text)."""
    labels = EL["scored_labels"]
    p0 = preds[(0, labels[0])]
    sex, units = p0["sex"], p0["assess_unit"]
    y = {0: p0["y_income"], 1: p0["y_occ"]}
    const = {j: int(p0["const_class"][j]) for j in (0, 1)}
    boot = OwnBoot(units)
    S_ = Stats(boot)

    def R(k, lab, w, fam=None):
        key = f"P_auc_{w}" if fam is None else f"P_auc_{fam}_{w}"
        P3 = preds[(k, lab)][key]
        return [S_.add(f"auc#{k}#{lab}#{fam}#{w}#{s_}", WAuc(P3[s_][:, 1], sex == 1)) for s_ in range(3)]

    def acc(k, lab, j):
        return S_.add(f"acc#{k}#{lab}#{j}", WMean(preds[(k, lab)][f"hard{j + 1}"] == y[j]))

    def loss(k, lab, j, kind):
        pr = np.asarray(preds[(k, lab)][f"prob{j + 1}"], dtype=np.float64)
        ll = -np.log(np.clip(pr[np.arange(len(y[j])), y[j]], LL_CLIP, 1.0))
        br = np.sum((pr - np.eye(pr.shape[1])[y[j]]) ** 2, 1)
        return S_.add(f"{kind}#{k}#{lab}#{j}", WMean(ll if kind == "ll" else br))

    def constacc(j):
        return S_.add(f"const#{j}", WMean(y[j] == const[j]))

    res_ = EL["resolved"]
    roles = {"A": ("J*", "C_rate"), "B": ("J*", "C_global"), "C": ("P*", "T*")}
    slots = []
    i_ = 1
    for claim in ("A", "B", "C"):
        nom, ref = roles[claim]
        slots.append({"id": f"P{i_:02d}", "claim": claim, "kind": "coalition", "nominee": nom, "ref": ref,
                      "target": MARGIN_PAIR, "side": "lower>"})
        i_ += 1
        for w in ("v1", "v2"):
            slots.append({"id": f"P{i_:02d}", "claim": claim, "kind": "local", "view": w, "nominee": nom, "ref": ref,
                          "target": MARGIN_LOCAL, "side": "upper<"})
            i_ += 1
        for kind, tgt, side in (("acc", -0.01, "lower>"), ("logloss", 0.01, "upper<"), ("brier", 0.005, "upper<"),
                                ("retain", 0.0, "lower>")):
            for j in (0, 1):
                slots.append({"id": f"P{i_:02d}", "claim": claim, "kind": kind, "task": j, "nominee": nom,
                              "target": tgt, "side": side})
                i_ += 1
    for kind, tgt in (("logloss", 0.01), ("brier", 0.005)):
        for j in (0, 1):
            slots.append({"id": f"P{i_:02d}", "claim": "Q", "kind": kind, "task": j, "nominee": "Q*", "target": tgt,
                          "side": "upper<"})
            i_ += 1
    assert len(slots) == N_ENDPOINTS
    U = "SRC|U"
    combos = {}
    for e in slots:
        nom, ref = res_.get(e["nominee"]), res_.get(e.get("ref")) if e.get("ref") else None
        per = []
        for k in SEEDS:
            j = e.get("task")
            if e["kind"] == "coalition":
                per.append(("diff", R(k, ref, "pair"), R(k, nom, "pair")))
            elif e["kind"] == "local":
                per.append(("diff", R(k, nom, e["view"]), R(k, ref, e["view"])))
            elif e["kind"] == "acc":
                per.append(("diff", [acc(k, nom, j)], [acc(k, U, j)]))
            elif e["kind"] in ("logloss", "brier"):
                kd = "ll" if e["kind"] == "logloss" else "br"
                per.append(("diff", [loss(k, nom, j, kd)], [loss(k, U, j, kd)]))
            else:
                per.append(("lin", [acc(k, nom, j)], [acc(k, U, j)], [constacc(j)]))
        combos[e["id"]] = per
    keys = [x for per in combos.values() for c_ in per for part in c_[1:] for x in part]
    vals = S_.evaluate(keys)

    def m3(ks, which):
        return np.mean([S_.val[x][which] for x in ks], axis=0)

    out = []
    for e in slots:
        pts, reps = [], []
        for c_ in combos[e["id"]]:
            if c_[0] == "diff":
                pts.append(m3(c_[1], 0) - m3(c_[2], 0))
                reps.append(m3(c_[1], 1) - m3(c_[2], 1))
            else:
                pts.append(m3(c_[1], 0) - 0.8 * m3(c_[2], 0) - 0.2 * m3(c_[3], 0))
                reps.append(m3(c_[1], 1) - 0.8 * m3(c_[2], 1) - 0.2 * m3(c_[3], 1))
        pt = float(np.mean(pts))
        rp = np.mean(np.stack(reps), axis=0)
        nonfin = int((~np.isfinite(rp)).sum())
        se = float(np.std(rp, ddof=1))
        lo, hi = pt - Z_PRIMARY * se, pt + Z_PRIMARY * se
        dec = "INVALID" if nonfin or not np.isfinite(pt) else \
            (("PASS" if lo > e["target"] else "NOT_ESTABLISHED") if e["side"] == "lower>" else
             ("PASS" if hi < e["target"] else "NOT_ESTABLISHED"))
        rl = [e["nominee"]] + ([e["ref"]] if e.get("ref") else [])
        descriptive = any(EL["statuses"].get(x, {}).get("status") != "NOMINEE" for x in rl)
        out.append({**e, "point": pt, "se": se, "lower": lo, "upper": hi, "decision_numeric": dec,
                    "decision": "DESCRIPTIVE_ONLY" if descriptive and dec != "INVALID" else dec, "nonfinite": nonfin,
                    "nominee_config": res_.get(e["nominee"]), "ref_config": res_.get(e.get("ref")) if e.get("ref") else None})
    lev = {}
    if levels:
        fams_all = {}
        for (k, lab), p in preds.items():
            for key in p:
                if key.startswith("P_auc_"):
                    rest = key[len("P_auc_"):]
                    fam = None if rest in VIEWS else rest.rsplit("_", 1)[0]
                    w = rest if rest in VIEWS else rest.rsplit("_", 1)[1]
                    fams_all.setdefault((k, lab), []).append((fam, w))
        lk = []
        for (k, lab), fl in fams_all.items():
            for fam, w in fl:
                lk += R(k, lab, w, fam)
            for j in (0, 1):
                lk += [acc(k, lab, j), loss(k, lab, j, "ll"), loss(k, lab, j, "br")]
        lk += [constacc(0), constacc(1)]
        lv = S_.evaluate(lk)
        for (k, lab), fl in fams_all.items():
            for fam, w in fl:
                ks = R(k, lab, w, fam)
                lev[f"R#{k}#{lab}#{fam or 'primary'}#{w}"] = (m3(ks, 0), m3(ks, 1))
            for j in (0, 1):
                for kd, key in (("acc", acc(k, lab, j)), ("ll", loss(k, lab, j, "ll")), ("br", loss(k, lab, j, "br"))):
                    lev[f"{kd}#{k}#{lab}#{j}"] = lv[key]
                if lab != U:
                    for kd, kk in (("ll", "llx"), ("br", "brx")):
                        a_, b_ = lv[loss(k, lab, j, kd)], lv[loss(k, U, j, kd)]
                        lev[f"{kk}#{k}#{lab}#{j}"] = (a_[0] - b_[0], a_[1] - b_[1])
        for lab in labels:
            for fam, w in fams_all.get((0, lab), []):
                if all((fam, w) in fams_all.get((k, lab), []) for k in SEEDS):
                    xs = [lev[f"R#{k}#{lab}#{fam or 'primary'}#{w}"] for k in SEEDS]
                    lev[f"Rmean#{lab}#{fam or 'primary'}#{w}"] = (float(np.mean([x[0] for x in xs])),
                                                                 np.mean([x[1] for x in xs], axis=0))
            for j in (0, 1):
                for kd in ("acc", "ll", "br") + (("llx", "brx") if lab != U else ()):
                    xs = [lev[f"{kd}#{k}#{lab}#{j}"] for k in SEEDS]
                    lev[f"{kd}mean#{lab}#{j}"] = (float(np.mean([x[0] for x in xs])), np.mean([x[1] for x in xs], axis=0))
        for j in (0, 1):
            lev[f"const#{j}"] = lv[constacc(j)]
    return out, lev, boot


def claim_status_own(nom_state, cmp_state, clauses):
    """LABEL_TRUTH_TABLE claim rules, in precedence order."""
    if cmp_state == "TECHNICAL_FAILURE":
        return "INVALID_COMPARATOR"
    if nom_state == "TECHNICAL_FAILURE":
        return "INVALID_NOMINEE"
    if nom_state == "NO_ELIGIBLE":
        return "NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE"
    if cmp_state == "NO_ELIGIBLE":
        return "NOT_APPLICABLE_NO_ELIGIBLE_COMPARATOR"
    if any(c_ == "INVALID" for c_ in clauses):
        return "INVALID"
    if all(c_ == "PASS" for c_ in clauses):
        return "PASS"
    return "NOT_ESTABLISHED"


def role_state(st):
    s_ = (st or {}).get("status")
    if s_ == "NOMINEE":
        return "ELIGIBLE"
    if s_ in ("NO_ELIGIBLE_NOMINEE", "NO_ELIGIBLE_COMPARATOR"):
        return "NO_ELIGIBLE"
    return "TECHNICAL_FAILURE"


COVERAGE_BAD = ("NOT_APPLICABLE_NO_ELIGIBLE_COMPARATOR", "INVALID_COMPARATOR", "INVALID_NOMINEE", "INVALID")


def labels_own(EL, eps):
    st = EL["statuses"]
    claims = {}
    for claim, (nom, ref) in (("A", ("J*", "C_rate")), ("B", ("J*", "C_global")), ("C", ("P*", "T*"))):
        cl = [e["decision_numeric"] for e in eps if e["claim"] == claim]
        claims[claim] = claim_status_own(role_state(st.get(nom)), role_state(st.get(ref)), cl)
    qcl = [e["decision_numeric"] for e in eps if e["claim"] == "Q"]
    if role_state(st.get("Q*")) != "ELIGIBLE":
        q = "NOT_APPLICABLE_NO_Q"
    elif any(c_ == "INVALID" for c_ in qcl):
        q = "INVALID"
    elif all(c_ == "PASS" for c_ in qcl):
        q = "PASS"
    else:
        q = "NOT_ESTABLISHED"
    sa, gm = bool(EL.get("stage_a_valid")), bool(EL.get("gate_met"))
    tv = bool((EL.get("technical_validity") or {}).get("ok"))
    fam = (st.get("P*") or {}).get("winning_family")
    missing = sorted(c_ for c_, v in claims.items() if v in COVERAGE_BAD) + (["Q*"] if q in ("INVALID", "NOT_APPLICABLE_NO_Q")
                                                                            else [])

    def amended():
        if not sa:
            return "INCOMPLETE_OR_INVALID"
        if not gm:
            return "CAPACITY_GATE_NOT_MET"
        if not tv:
            return "INCOMPLETE_OR_INVALID"
        if claims["A"] == "PASS" and claims["B"] == "PASS":
            return "JOINT_DEVELOPMENT_CRITERION_MET" + (f" + PRIVACY_COMPRESSION_DEVELOPMENT_CRITERION_MET ({fam})"
                                                        if claims["C"] == "PASS" else "")
        if claims["C"] == "PASS":
            return f"PRIVACY_COMPRESSION_DEVELOPMENT_CRITERION_MET ({fam})"
        if missing:
            return "INCOMPLETE_OR_INVALID"
        return "CONFIDENCE_FEASIBILITY_ESTABLISHED" if q == "PASS" else "EXPERIMENTAL_NO_ADVANTAGE"

    def stage_a_rule():
        if not sa:
            return "INCOMPLETE_OR_INVALID"
        if not gm:
            return "CAPACITY_GATE_NOT_MET"
        if not tv or missing:
            return "INCOMPLETE_OR_INVALID"
        if claims["A"] == "PASS" and claims["B"] == "PASS":
            return "JOINT_DEVELOPMENT_CRITERION_MET" + (" + PRIVACY_COMPRESSION_DEVELOPMENT_CRITERION_MET"
                                                        if claims["C"] == "PASS" else "")
        if claims["C"] == "PASS":
            return "PRIVACY_COMPRESSION_DEVELOPMENT_CRITERION_MET"
        return "CONFIDENCE_FEASIBILITY_ESTABLISHED" if q == "PASS" else "EXPERIMENTAL_NO_ADVANTAGE"

    return {"claims": claims, "q": q, "missing": missing, "label": amended(), "label_stage_a_rule": stage_a_rule()}


def check_endpoints(preds, EL):
    eps, lev, boot = own_endpoints(preds, EL)
    inf = jload(RUN / "inference.json")
    pub = {}
    import csv
    pe = RES / "PRIMARY_ENDPOINTS.csv"
    if pe.exists():
        pub = {r_["id"]: r_ for r_ in csv.DictReader(pe.open())}
    fam_pub = {e["id"]: e for e in jload(RES / "PRIMARY_FAMILY.json")["slots"]}
    rec = {e["id"]: e for e in inf["primary"]}
    mx = {"point": 0.0, "se": 0.0, "bounds": 0.0}
    dec_bad, csv_bad, slot_bad = [], [], []
    for e in eps:
        r_ = rec.get(e["id"], {})
        fp = fam_pub.get(e["id"], {})
        if (fp.get("kind"), fp.get("nominee"), fp.get("ref"), fp.get("view"), fp.get("task"), fp.get("target"), fp.get("side")) != \
                (e["kind"], e["nominee"], e.get("ref"), e.get("view"), e.get("task"), e["target"], e["side"]):
            slot_bad.append(e["id"])
        if r_.get("point") is None:
            dec_bad.append(f"{e['id']}: no recorded point")
            continue
        mx["point"] = max(mx["point"], abs(e["point"] - r_["point"]))
        mx["se"] = max(mx["se"], abs(e["se"] - r_["se"]))
        mx["bounds"] = max(mx["bounds"], abs(e["lower"] - r_["lower"]), abs(e["upper"] - r_["upper"]))
        if e["decision"] != r_.get("decision") or e["decision_numeric"] != r_.get("decision_numeric", r_.get("decision")):
            dec_bad.append(f"{e['id']}: {e['decision']}/{e['decision_numeric']} vs {r_.get('decision')}/{r_.get('decision_numeric')}")
        c_ = pub.get(e["id"])
        if c_ is None or c_["decision"] != r_.get("decision") or abs(float(c_["point"]) - e["point"]) > 5e-7 or \
                abs(float(c_["lower"]) - e["lower"]) > 5e-7 or abs(float(c_["upper"]) - e["upper"]) > 5e-7:
            csv_bad.append(e["id"])
    lab = labels_own(EL, eps)
    lab_ok = (lab["label"] == inf.get("label") and lab["claims"] == inf.get("claim_status") and lab["q"] == inf.get("q_status")
              and lab["label_stage_a_rule"] == (inf.get("label_under_stage_a_lock_rule") or {}).get("label"))
    # all levels (points and SE)
    lv_mx, lv_se_mx, lv_missing = 0.0, 0.0, []
    rl = inf.get("levels", {})
    for nm, (pt, rp) in lev.items():
        nm_r = nm.replace("#None#", "#primary#")
        r_ = rl.get(nm_r) or rl.get(nm)
        if r_ is None:
            lv_missing.append(nm)
            continue
        lv_mx = max(lv_mx, abs(pt - r_["point"]))
        lv_se_mx = max(lv_se_mx, abs(float(np.std(rp[np.isfinite(rp)], ddof=1)) - r_["se"]))
    extra = sorted(set(rl) - {n.replace("#None#", "#primary#") for n in lev})
    ok = (not dec_bad and not csv_bad and not slot_bad and lab_ok and mx["point"] <= 1e-12 and mx["se"] <= 1e-12 and
          mx["bounds"] <= 1e-11 and lv_mx <= 1e-12 and lv_se_mx <= 1e-12 and not lv_missing and not extra)
    table = [{k_: e[k_] for k_ in ("id", "claim", "kind", "nominee_config", "ref_config", "point", "se", "lower", "upper",
                                   "target", "side", "decision", "decision_numeric")} for e in eps]
    return res("PASS" if ok else "FAIL", endpoints=table, max_abs_diff=mx, decision_mismatches=dec_bad,
               csv_mismatches=csv_bad, slot_definition_mismatches=slot_bad, own_labels=lab,
               recorded_label=inf.get("label"), recorded_label_stage_a_rule=inf.get("label_under_stage_a_lock_rule"),
               label_equal=lab_ok, levels_checked=len(lev), levels_point_max_abs_diff=lv_mx,
               levels_se_max_abs_diff=lv_se_mx, levels_missing_in_record=lv_missing[:10], levels_only_in_record=extra[:10],
               bootstrap={"B": boot.B, "groups": boot.G, "rows": boot.n, "seed": BOOT_SEED, "z": Z_PRIMARY,
                          "counts_sha256": hashlib.sha256(boot.counts.tobytes()).hexdigest()}), eps, lev


CONTROL_SEED, SPLIT_SEED, NULL_Z, PLANT_MIN = 20261021, 20261022, 3.5, 0.75


def own_null_split(D: Data):
    ix = D.idx[INNER]
    u = np.array([int(hashlib.sha256(f"{SPLIT_SEED}|null-split|{int(g_)}".encode()).hexdigest()[:16], 16) / 2.0 ** 64
                  for g_ in D.unit[ix]])
    return np.flatnonzero(u < 0.5), np.flatnonzero(u >= 0.5)


def own_frozen_permutation(S, D: Data, halves, seed):
    rng = np.random.default_rng(seed)
    Sp = S.copy()
    sel = D.idx[INNER]
    for ix in [D.idx["AUDIT_FIT"]] + [sel[h] for h in halves]:
        Sp[ix] = S[ix][rng.permutation(len(ix))]
    used = np.concatenate([D.idx["AUDIT_FIT"], sel])
    return Sp, hashlib.sha256(np.ascontiguousarray(Sp[used]).astype(np.int64).tobytes()).hexdigest()


def null_sd(y):
    n1 = int((np.asarray(y) == 1).sum())
    n0 = int(len(y) - n1)
    return math.sqrt((n0 + n1 + 1) / (12.0 * n0 * n1))


def check_controls(D: Data, L, EL):
    """Real-data controls: own half split and frozen permutation (hash), own null threshold from half-B SEX counts,
    every pass rule re-applied to the recorded held-out AUCs, the verdict, the null calibration permutations, and the
    lock pin of the verdict file. The held-out AUC values themselves come from attacker fits that this check does not
    repeat (stated)."""
    a = jload(RES / "AUDIT_PRELOCK_CHECKS.json")
    S = L["sex"].copy()
    sel = D.idx[INNER]
    A_, B_ = own_null_split(D)
    Sp, sha = own_frozen_permutation(S, D, (A_, B_), CONTROL_SEED)
    sd0 = null_sd(Sp[sel[B_]])
    thr = 0.5 + NULL_Z * sd0
    bad, vac, rules = [], [], {}
    for lab, r in a["releases"].items():
        if r.get("permutation_sha256") != sha:
            bad.append(f"{lab}: permutation hash")
        if abs(r["null_threshold"] - thr) > 1e-15 or abs(r["null_sd_B"] - sd0) > 1e-15:
            bad.append(f"{lab}: null threshold")
        for ck, v in r["checks"].items():
            hb = lambda w: v[w]["heldout_auc_B"]  # noqa: E731
            if ck == "NULL":
                own_ok = all(hb(w) <= thr for w in VIEWS)
            elif ck.startswith(("CONF_", "COLL_")):
                own_ok = hb(v["target_view"]) > PLANT_MIN and bool(v.get("serialisation_exact"))
                if ck.startswith("COLL_") and not v.get("decoded_probability_only_audit"):
                    vac.append(f"{lab}/{ck}")
            elif ck == "XOR":
                own_ok = hb("pair") > PLANT_MIN and hb("v1") <= thr and hb("v2") <= thr and bool(v.get("serialisation_exact"))
            elif ck.startswith("ROT_"):
                own_ok = hb(v.get("target_view", "v" + ck[-1])) > PLANT_MIN and hb("pair") > PLANT_MIN and \
                    bool(v.get("serialisation_exact", True))
            else:
                own_ok = bool(v.get("ok"))
            rules[f"{lab}/{ck}"] = own_ok
            if own_ok != bool(v.get("ok")):
                bad.append(f"{lab}/{ck}: own rule {own_ok} vs recorded {v.get('ok')}")
    nc = a.get("null_calibration", {})
    nc_bad = []
    for row in nc.get("rows", []):
        Spr, shr = own_frozen_permutation(S, D, (A_, B_), CONTROL_SEED + 100 + int(row["rep"]))
        sdr = null_sd(Spr[sel[B_]])
        if row["permutation_sha256"] != shr or abs(row["sd0"] - sdr) > 1e-15 or \
                bool(row["exceeds"]) != (row["heldout_auc_B"] > 0.5 + NULL_Z * sdr):
            nc_bad.append(f"rep {row['rep']} {row['view']}")
    exceed = sum(1 for row in nc.get("rows", []) if row["heldout_auc_B"] > row["threshold"])
    verdict = a.get("verdict", {})
    all_ok_own = not bad and all(rules.values()) and exceed == 0
    out = {"own_permutation_sha256": sha, "own_null_threshold": thr, "own_null_sd_B": sd0,
           "half_sizes": [int(len(A_)), int(len(B_))], "checks_replayed": len(rules), "checks_passing": sum(rules.values()),
           "null_calibration_rows": len(nc.get("rows", [])), "null_calibration_exceedances": exceed,
           "null_calibration_mismatches": nc_bad, "verdict_all_ok": verdict.get("all_ok"),
           "verdict_equals_own": bool(verdict.get("all_ok")) == all_ok_own,
           "verdict_file_sha256_equals_lock_pin": sha_file(RES / "AUDIT_PRELOCK_CHECKS.json") ==
           (EL.get("technical_validity") or {}).get("controls_verdict_sha256"),
           "decoded_probability_only_audits_empty": vac, "failures": bad[:20],
           "not_replayed": "the held-out AUCs come from attacker fits on planted / permuted labels that this check does "
                           "not refit; the split, permutation, thresholds and every pass rule are own"}
    st = "FAIL" if bad or nc_bad or not out["verdict_equals_own"] or not out["verdict_file_sha256_equals_lock_pin"] else \
        ("WARN" if vac else "PASS")
    if vac:
        out["reason"] = (f"{len(vac)} COLL checks record decoded_probability_only_misses_it = true with an EMPTY decoded-"
                         f"probability-only audit: the stated complement is not evidenced (pass rules unaffected)")
    return res(st, **out)


def check_tables(own, eps, lev, sel_mine, preds, EL):
    """Public COALITION_AND_RATE_RESULTS.csv, ACTUAL_TASK_UTILITY.csv, CAPACITY_CURVE.csv (assessment columns) and
    INNER_SELECTION_TABLE.csv vs own numbers (6-decimal prints: |diff| <= 5e-7)."""
    import csv
    out, bad = {}, []
    tol = 5e-7 + 1e-12

    def cmp(tag, a_, b_):
        if a_ in ("", None):
            return
        out.setdefault(tag, [0, 0.0])
        out[tag][0] += 1
        d_ = abs(float(a_) - float(b_))
        out[tag][1] = max(out[tag][1], d_)
        if d_ > tol:
            bad.append(f"{tag}: {a_} vs own {b_:.7f}")

    def L_(nm):
        return lev[nm]

    def se(nm):
        rp = L_(nm)[1]
        return float(np.std(rp[np.isfinite(rp)], ddof=1))
    f_ = RES / "COALITION_AND_RATE_RESULTS.csv"
    if f_.exists():
        for r_ in csv.DictReader(f_.open()):
            lab = r_["label"]
            for fam in ("primary", "complete", "decisions", "probs", "scores", "cells"):
                for w in VIEWS:
                    nm = f"Rmean#{lab}#{fam}#{w}"
                    if nm in lev:
                        cmp("coalition_auc", r_.get(f"{fam}_{w}_auc"), L_(nm)[0])
                        cmp("coalition_auc_se", r_.get(f"{fam}_{w}_se"), se(nm))
            for j, t_ in ((0, "income"), (1, "occ")):
                cmp("coalition_task", r_.get(f"acc_{t_}"), L_(f"accmean#{lab}#{j}")[0])
                cmp("coalition_task", r_.get(f"ll_{t_}"), L_(f"llmean#{lab}#{j}")[0])
                cmp("coalition_task", r_.get(f"br_{t_}"), L_(f"brmean#{lab}#{j}")[0])
                if lab != "SRC|U":
                    for kk in ("llx", "brx"):
                        cmp("coalition_excess", r_.get(f"{kk}_{t_}"), L_(f"{kk}mean#{lab}#{j}")[0])
                        cmp("coalition_excess_se", r_.get(f"{kk}_{t_}_se"), se(f"{kk}mean#{lab}#{j}"))
    f_ = RES / "ACTUAL_TASK_UTILITY.csv"
    if f_.exists():
        y = {0: preds[(0, EL["scored_labels"][0])]["y_income"], 1: preds[(0, EL["scored_labels"][0])]["y_occ"]}
        cc = [int(x) for x in preds[(0, EL["scored_labels"][0])]["const_class"]]
        for r_ in csv.DictReader(f_.open()):
            k, lab, j = int(r_["seed"]), r_["label"], 0 if r_["task"] == "income" else 1
            p = preds[(k, lab)]
            m_ = my_utility(p[f"prob{j + 1}"], p[f"hard{j + 1}"], y[j], KS[j], cc[j])
            for q_ in ("acc", "logloss", "brier", "const_acc"):
                cmp("actual_task_utility", r_[q_], m_[q_])
            cmp("actual_task_utility", r_["gain"], m_["acc"] - m_["const_acc"])
    f_ = RES / "CAPACITY_CURVE.csv"
    if f_.exists():
        for r_ in csv.DictReader(f_.open()):
            cid = r_["config"]
            for col, nm in (("assess_llx_occ", f"llxmean#{cid}#1"), ("assess_brx_occ", f"brxmean#{cid}#1"),
                            ("assess_llx_income", f"llxmean#{cid}#0"), ("assess_brx_income", f"brxmean#{cid}#0"),
                            ("assess_primary_pair_auc", f"Rmean#{cid}#primary#pair"),
                            ("assess_primary_v1_auc", f"Rmean#{cid}#primary#v1"),
                            ("assess_primary_v2_auc", f"Rmean#{cid}#primary#v2")):
                if col in r_:
                    cmp("capacity_curve_assessment", r_[col], L_(nm)[0])
    f_ = RES / "INNER_SELECTION_TABLE.csv"
    if f_.exists():
        for r_ in csv.DictReader(f_.open()):
            cid, k = r_["config"], int(r_["seed"])
            o = own[(k, cid)]
            a_ = o["fams"][o["primary"]]["auc"]
            for w in VIEWS:
                cmp("inner_selection_table", r_[f"auc_{w}"], a_[w])
            cmp("inner_selection_table", r_["ll_income"], o["utility"][1]["logloss"])
            cmp("inner_selection_table", r_["ll_occupation"], o["utility"][2]["logloss"])
            cmp("inner_selection_table", r_["brier_income"], o["utility"][1]["brier"])
            cmp("inner_selection_table", r_["brier_occupation"], o["utility"][2]["brier"])
            el_ = sel_mine["rows"][cid]["seeds"][k]["eligible"]
            if r_["seed_eligible"] != str(el_):
                bad.append(f"inner_selection_table {cid}#s{k} eligible")
    return res("FAIL" if bad else "PASS", cells_checked_and_max_abs_diff=out, mismatches=bad[:20])


def check_late_chronology(gate, EL):
    """Assessment chronology: every assess / outer / infer event after the first EVALUATION_LOCK push; the lock has one
    version; no non-outer unit completed after the push; AMENDMENT_A1 declares its file and was pushed before select;
    stage refusals and post-assessment no-op stages are reported; unlocked presentation modules are listed."""
    ev = jsonl(RUN / "ACTIVITY_LOG.jsonl")
    sema = jsonl(RUN / "SEMA_LOG.jsonl")
    push = gate.get("first_push_time")
    out, bad, warn = {}, [], []
    a_ev = [(e.get("event"), parse_iso(e["at"])) for e in ev if re.search(r"assess|unseal|outer|infer", str(e.get("event", "")))]
    out["assessment_events_before_push"] = [f"{n} at {iso(t)}" for n, t in a_ev if push is None or t < push]
    out["assessment_events"] = len(a_ev)
    if out["assessment_events_before_push"]:
        bad.append("assessment event before the EVALUATION_LOCK push")
    late = []
    for dd in UNITS.iterdir():
        cj = dd / "COMPLETE.json"
        if cj.exists() and not dd.name.startswith("outer__") and push and utc(cj.stat().st_mtime) > push:
            late.append(dd.name)
    out["non_outer_units_completed_after_push"] = sorted(late)
    if late:
        bad.append("units fitted after the assessment opened")
    out["evaluation_lock_versions"] = gate.get("commits")
    am = jload(RES / "AMENDMENT_A1.json") if (RES / "AMENDMENT_A1.json").exists() else None
    if am:
        lf = lock_files()
        c0 = lf["AMENDMENT_A1"]["commits"][-1][0] if lf.get("AMENDMENT_A1", {}).get("commits") else None
        ap = first_remote(c0, remote_reflog()) if c0 else None
        sel_start = min((parse_iso(e["at"]) for e in ev if e.get("event") == "start select"), default=None)
        out["amendment_A1"] = {"declared": am.get("changes_previously_locked"), "files": sorted(am.get("code_files", {})),
                               "first_push": ap, "select_start": sel_start,
                               "pushed_before_select": bool(ap and sel_start and ap < sel_start),
                               "files_equal_worktree": all(sha_file(WT / f_) == h for f_, h in am.get("code_files", {}).items()),
                               "declared_equals_files": sorted(am.get("changes_previously_locked") or []) ==
                               sorted(am.get("code_files", {}))}
        if not (out["amendment_A1"]["pushed_before_select"] and out["amendment_A1"]["files_equal_worktree"] and
                out["amendment_A1"]["declared_equals_files"]):
            bad.append("AMENDMENT_A1")
    refusals = [{"label": e["label"], "at": e["at"], "rc": e.get("rc"), "wall_s": e.get("wall_s")} for e in sema
                if e.get("event") == "release" and str(e.get("label", "")).startswith("A:") and e.get("rc") not in (0, None)]
    out["lead_stage_holds_with_nonzero_exit"] = refusals
    post = [e for e in ev if str(e.get("event", "")).startswith("start ") and push and parse_iso(e["at"]) > push and
            e["event"].split(" ", 1)[1] not in ("assess", "outer", "infer")]
    out["post_assessment_stage_starts"] = [{"stage": e["event"], "at": e["at"], "shard": e.get("shard")} for e in post]
    if post and not late:
        warn.append("a non-assessment stage was started after the assessment opened (no unit was created or changed: "
                    "resumable no-op)")
    lc = set(EL.get("locked_code_files", {}))
    unlocked = sorted(str(q.relative_to(WT)) for q in (WT / "qpc").glob("*.py") if str(q.relative_to(WT)) not in lc)
    out["qpc_modules_not_in_evaluation_lock"] = unlocked
    if unlocked:
        warn.append(f"unlocked qpc modules present after the lock: {unlocked} (presentation only; every public number "
                    f"they produce is checked by check_tables)")
    st = "FAIL" if bad else ("WARN" if warn else "PASS")
    return res(st, failures=bad, warnings=warn, reason=warn[0] if warn and not bad else None, **out)


# ------------------------------------------------------------------------------------------------ own attacker refits
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402


class MyCanon:
    """Defence-aware canonicalisation (pinned rule): centre, SVD on the fitting rows, keep s_j > max(n, d) eps scale
    with scale = max(s_1, ||X||_2), whiten the kept directions to unit variance."""

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        self.mu = X.mean(0)
        _, sv, Vt = np.linalg.svd(X - self.mu, full_matrices=False)
        scale = max(float(sv[0]) if sv.size else 0.0, float(np.linalg.norm(X, 2)))
        keep = sv > 1.0 * max(n, d) * float(np.finfo(np.float64).eps) * scale
        self.rank = int(keep.sum())
        self.W = (Vt[keep].T / sv[keep]) * math.sqrt(max(n - 1, 1))
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=np.float64) - self.mu) @ self.W


class MyDA:
    def __init__(self, kind, seed):
        self.kind, self.seed = kind, seed

    def fit(self, X, y):
        self.c = MyCanon().fit(X)
        Z = self.c.transform(X)
        if self.c.rank == 0:
            self.m = DummyClassifier(strategy="prior")
            Z = np.zeros((len(Z), 1))
        elif self.kind == "lr":
            self.m = LogisticRegression(C=1.0, max_iter=3000)
        else:
            self.m = MLPClassifier(hidden_layer_sizes=(128, 128), alpha=1e-4, max_iter=300, early_stopping=True,
                                   validation_fraction=0.1, n_iter_no_change=15, random_state=self.seed)
        self.m.fit(Z, y)
        self.classes_ = self.m.classes_
        return self

    def predict_proba(self, X):
        Z = self.c.transform(X)
        return self.m.predict_proba(Z if self.c.rank else np.zeros((len(Z), 1)))


def my_attacker(name, seed):
    if name.startswith("LR_C"):
        return make_pipeline(StandardScaler(), LogisticRegression(C=float(name[4:]), max_iter=3000))
    if name.startswith("MLP_"):
        h = tuple(int(x) for x in name[4:].split("x"))
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=h, alpha=1e-4, max_iter=300,
                                                             early_stopping=True, validation_fraction=0.1,
                                                             n_iter_no_change=15, random_state=seed))
    if name.startswith("HGB_"):
        lr, lv = name[4:].split("_")
        return HistGradientBoostingClassifier(learning_rate=float(lr), max_leaf_nodes=int(lv), max_iter=200,
                                              early_stopping=False, random_state=seed)
    if name in ("DA_LR", "DA_MLP"):
        return MyDA("lr" if name == "DA_LR" else "mlp", seed)
    raise ValueError(name)


def my_p1(m, X):
    P = m.predict_proba(X)
    cl = [int(c) for c in m.classes_]
    return np.asarray(P[:, cl.index(1)], dtype=np.float64) if 1 in cl else np.zeros(len(X))


def my_columns(tok, D: Data, alphabet):
    """Occurrence-ordered one-hot columns: AUDIT_FIT, then INNER_SELECTION, then all rows; unseen IDs last."""
    tok = np.asarray(tok, dtype=np.int64)
    order, seen = [], set()
    for ix in (D.idx["AUDIT_FIT"], D.idx[INNER], np.arange(len(tok))):
        for v in tok[ix].tolist():
            if v not in seen:
                seen.add(v)
                order.append(v)
    order += [v for v in range(int(alphabet)) if v not in seen]
    col = np.empty(int(alphabet), dtype=np.int64)
    col[np.asarray(order, dtype=np.int64)] = np.arange(int(alphabet))
    return col


def my_onehot(idx, K):
    out = np.zeros((len(idx), int(K)))
    out[np.arange(len(idx)), np.asarray(idx, dtype=np.int64)] = 1.0
    return out


def my_views(rel, D: Data, family="code"):
    X, toks = {}, None
    for i, K in ((1, 2), (2, 6)):
        if family == "code":
            col = my_columns(rel[f"tok{i}"], D, rel[f"alpha{i}"])
            X[f"v{i}"] = np.hstack([my_onehot(col[rel[f"tok{i}"]], rel[f"alpha{i}"]),
                                    np.asarray(rel[f"p{i}"], dtype=np.float64), my_onehot(rel[f"hard{i}"], K)])
            toks = {"v1": rel["tok1"], "v2": rel["tok2"]}
        elif family == "interface":
            X[f"v{i}"] = np.hstack([np.asarray(rel[f"c{i}"], np.float64), np.asarray(rel[f"p{i}"], np.float64)])
    X["pair"] = np.hstack([X["v1"], X["v2"]])
    return X, toks


def my_cc(tok, y, fit_idx, pred_idx, a):
    prior1 = float(np.mean(y[fit_idx] == 1))
    u, inv = np.unique(np.asarray(tok)[fit_idx], return_inverse=True)
    n = np.bincount(inv.ravel(), minlength=len(u)).astype(np.float64)
    n1 = np.bincount(inv.ravel(), weights=(y[fit_idx] == 1).astype(np.float64), minlength=len(u))
    lut = {int(k_): (n1[j] + a * prior1) / (n[j] + a) for j, k_ in enumerate(u.tolist())}
    kk = np.asarray(tok)[pred_idx].tolist()
    seen = np.array([k_ in lut for k_ in kk])
    return np.array([lut.get(k_, prior1) for k_ in kk], dtype=np.float64), seen, prior1


def my_cc_pair(t1, t2, y, fit_idx, pred_idx, sel_pos, a):
    base = int(max(np.asarray(t2).max(), 0)) + 1
    keys = np.asarray(t1, np.int64) * base + np.asarray(t2, np.int64)
    p, seen, prior1 = my_cc(keys, y, fit_idx, pred_idx, a)
    l1 = my_cc(t1, y, fit_idx, pred_idx, a)[0]
    l2 = my_cc(t2, y, fit_idx, pred_idx, a)[0]
    cand = {"local_1": l1, "local_2": l2, "prior": np.full(len(pred_idx), prior1)}
    ys = y[pred_idx][sel_pos]
    un = ~seen[sel_pos]
    if un.sum() == 0:
        rule = "prior"
    else:
        ces = {r_: my_ce(ys[un], cand[r_][sel_pos][un]) for r_ in ("local_1", "local_2", "prior")}
        rule = min(("local_1", "local_2", "prior"), key=lambda r_: (ces[r_], ("local_1", "local_2", "prior").index(r_)))
    p = p.copy()
    p[~seen] = cand[rule][~seen]
    return p, rule


def check_refits(D: Data, L, LU, T, own, preds, EL):
    """Own refits of the scored nominee / comparator attackers (AUC-selected, attacker seeds 0-2, fitted on AUDIT_FIT
    with own design matrices from own releases) vs the stored INNER_SELECTION predictions and the stored assessment
    predictions; SRC|U includes its composed winners (the code reader applied to the code of U)."""
    fa, sel, a_idx = D.idx["AUDIT_FIT"], D.idx[INNER], D.idx[ASSESS]
    y = LU["sex"].astype(np.int64)
    assert (y[fa] >= 0).all() and (y[sel] >= 0).all() and (y[a_idx] >= 0).all()
    pred_idx = np.concatenate([sel, a_idx])
    sel_pos = np.arange(len(sel))
    targets = []
    for role in ("P*", "C_rate", "C_global", "T*", "J*"):
        c_ = EL["resolved"].get(role)
        if c_ and c_ not in targets:
            targets.append(c_)
    targets.append("SRC|U")
    out, mx_in, mx_out, n_fit, bitw, bad = {}, 0.0, 0.0, 0, 0, []
    t0 = time.process_time()
    for k in SEEDS:
        for cid in targets:
            o = own[(k, cid)]
            fam = o["primary"]
            for w in VIEWS:
                if cid == "SRC|U":
                    win = o["composed"][fam]["winner"][w]
                    src_cid = cid if win == "source" else win
                    label = o["fams"][fam]["sel"][w] if win == "source" else own[(k, win)]["fams"]["code"]["sel"][w]
                else:
                    src_cid, label = cid, o["fams"][fam]["sel"][w]
                cand, view, att = label.split(":")
                rel = own_release(k, src_cid, T, {})  if not src_cid.startswith("SRC|") else None
                if src_cid.startswith("SRC|"):
                    t_ = T[(k, "U")]
                    X, toks = my_views({"c1": t_["c1"], "c2": t_["c2"], "p1": t_["p1"], "p2": t_["p2"]}, D, "interface")
                else:
                    X, toks = my_views(rel, D, "code")
                # inner reference: the selected attacker's stored INNER predictions -- for a composed winner, the
                # winning code's own inner unit (the composed reader IS that code reader on the same rows)
                if src_cid == cid:
                    inner_store = np.load(UNITS / f"inner__{release_unit(k, cid)}" / "inner_preds.npz", allow_pickle=False)
                    SA = inner_store[f"SEL_{fam}_auc_{w}"]
                else:
                    inner_store = np.load(UNITS / f"inner__{release_unit(k, src_cid)}" / "inner_preds.npz", allow_pickle=False)
                    SA = inner_store[f"SEL_code_auc_{w}"]
                PO = preds[(k, cid)][f"P_auc_{w}"]
                rec = {"label": label, "source": src_cid, "seeds": []}
                for s_ in range(3):
                    if att.startswith("CCpair"):
                        pr, _ = my_cc_pair(toks["v1"], toks["v2"], y, fa, pred_idx, sel_pos, float(att.split("alpha")[1]))
                    elif att.startswith("CC_"):
                        pr = my_cc(toks[view], y, fa, pred_idx, float(att.split("alpha")[1]))[0]
                    else:
                        m = my_attacker(att, s_).fit(X[view][fa], y[fa])
                        pr = my_p1(m, X[view][pred_idx])
                        n_fit += 1
                    pin, pout = pr[:len(sel)], pr[len(sel):]
                    d_in = maxdiff(pin, SA[s_])
                    d_out = maxdiff(pout, PO[s_][:, 1])
                    b_ = bitwise(pin, SA[s_]) and bitwise(pout, PO[s_][:, 1])
                    bitw += int(b_)
                    mx_in, mx_out = max(mx_in, d_in), max(mx_out, d_out)
                    rec["seeds"].append({"inner_max_abs_diff": d_in, "assessment_max_abs_diff": d_out, "bitwise": b_,
                                         "own_assessment_auc": my_auc(y[a_idx], pout)})
                    if d_in > 1e-9 or d_out > 1e-9:
                        bad.append(f"{cid}#s{k}/{w} seed {s_}: {label} inner {d_in:.2e} assessment {d_out:.2e}")
                out[f"{cid}#s{k}/{w}"] = rec
    n_cmp = sum(len(v["seeds"]) for v in out.values())
    return res("FAIL" if bad else ("PASS" if bitw == n_cmp else "WARN"), comparisons=n_cmp, bitwise_equal=bitw,
               max_abs_diff_inner=mx_in, max_abs_diff_assessment=mx_out, model_fits=n_fit,
               cpu_s=round(time.process_time() - t0, 1), failures=bad[:20], per_view=out,
               reason=None if bitw == n_cmp else "some refits agree within 1e-9 but not bitwise",
               rule="own design matrices (occurrence-ordered one-hot token columns over the full alphabet, decoded q, "
                    "one-hot decision; pair = [v1, v2]; U interface = [c_i, p_i]); own sklearn members from the pinned "
                    "hyperparameters, own defence-aware canonicalisation, own cell readers and pair fallback; fitted on "
                    "AUDIT_FIT SEX; compared on INNER_SELECTION and OSF_DEVELOPMENT_ASSESSMENT rows")


def check_units_complete(prefixes=("tea__", "ref__")):
    bad, n_ok, kinds = [], 0, {}
    for d in sorted(q for q in UNITS.iterdir() if q.is_dir()):
        kinds[d.name.split("__")[0]] = kinds.get(d.name.split("__")[0], 0) + 1
        if not d.name.startswith(prefixes):
            continue
        if not (d / "COMPLETE.json").exists():
            bad.append(f"{d.name}: no COMPLETE.json")
            continue
        c, _ = complete_ok(d)
        if not (c["id_ok"] and c["rehash_ok"]) or c["unlisted"]:
            bad.append(f"{d.name}: {c}")
        else:
            n_ok += 1
    return res("FAIL" if bad else "PASS", units_by_kind=kinds, checked_prefixes=list(prefixes), valid=n_ok, failures=bad)


def check_chronology():
    """Locks committed and pushed before their governing stage starts (ACTIVITY_LOG 'start <stage>' events), no
    unpushed lock version at a stage start, units completed after their governing lock push, PREDICTIONS.json pushed
    once before Stage A, and every lead stage start inside a semaphore hold (SEMA_LOG)."""
    entries = remote_reflog()
    ev = jsonl(RUN / "ACTIVITY_LOG.jsonl")
    sema = jsonl(RUN / "SEMA_LOG.jsonl")
    lf = lock_files()
    info, fails = {}, []
    for name, d in lf.items():
        if not d["exists"]:
            info[name] = {"exists": False}
            continue
        if not d["commits"]:
            info[name] = {"exists": True, "committed": False}
            fails.append(f"{name} not committed")
            continue
        first_c, first_t = d["commits"][-1]
        last_c, _ = d["commits"][0]
        fp = first_remote(first_c, entries)
        blob_ok = git("rev-parse", f"{last_c}:{REL_RES}/{name}.json") == git("hash-object", str(RES / f"{name}.json"))
        info[name] = {"first_commit": first_c, "first_commit_time": parse_iso(first_t), "first_push_time": fp,
                      "versions": len(d["commits"]), "worktree_equals_latest_commit": blob_ok}
        if not blob_ok:
            fails.append(f"{name}: worktree differs from its latest commit")
        if fp is None:
            fails.append(f"{name}: never pushed")
    starts = [e for e in ev if str(e.get("event", "")).startswith("start ")]
    holds, unreleased = [], []
    open_ = {}
    for e in sema:
        key = e.get("wrapper_pid")
        if e.get("event") == "acquire":
            # flock exclusivity: an earlier unreleased holder of this slot must have ended before this acquire
            for k2, (t0_, lab_, sl_) in list(open_.items()):
                if sl_ == e.get("slot") and k2 != key:
                    unreleased.append({"label": lab_, "slot": sl_, "acquired": t0_, "slot_reacquired_at": parse_iso(e["at"]),
                                       "by": e.get("label")})
                    holds.append((t0_, parse_iso(e["at"]), lab_))
                    open_.pop(k2)
            open_[key] = (parse_iso(e["at"]), e.get("label"), e.get("slot"))
        elif e.get("event") == "release" and key in open_:
            t0_, lab_, _ = open_.pop(key)
            holds.append((t0_, parse_iso(e["at"]), lab_))
    holds += [(t0_, None, lab_) for t0_, lab_, _ in open_.values()]
    st_out = []
    for e in starts:
        stage = e["event"].split(" ", 1)[1]
        t = parse_iso(e["at"])
        gov = STAGE_LOCK.get(stage)
        g_ = info.get(gov) or {}
        ok = gov is not None and bool(g_.get("first_push_time")) and g_["first_push_time"] <= t
        unpushed = []
        for name, d in lf.items():
            for c, ct in d["commits"]:
                if parse_iso(ct) <= t:
                    pt = first_remote(c, entries)
                    if pt is None or pt > t:
                        unpushed.append(name)
        in_sema = any(h0 <= t and (h1 is None or t <= h1) and str(lab).startswith("A:") for h0, h1, lab in holds)
        rec = {"stage": stage, "at": t, "lock_named": e.get("lock"), "governing_lock": gov, "ok": ok and not unpushed,
               "governing_lock_push_lead_s": (t - g_["first_push_time"]).total_seconds() if g_.get("first_push_time") else None,
               "unpushed_lock_versions": sorted(set(unpushed)), "inside_lead_semaphore_hold": in_sema}
        if not rec["ok"]:
            fails.append(f"start {stage} at {iso(t)}: governing lock not pushed before it")
        if not in_sema:
            fails.append(f"start {stage} at {iso(t)}: no lead semaphore hold")
        st_out.append(rec)
    # every completed unit after its governing lock's first push (activity event and COMPLETE.json mtime)
    def unit_lock(nm):
        if nm.startswith(("tea__", "ref__")):
            return "SOURCE_ADMISSION_LOCK"
        if nm.startswith(("a1__", "dir__")) or "_DIRECT-TASK_" in nm:
            return "STAGE_A_LOCK"
        if nm.startswith(("fine__", "pol__")):
            return "STAGE_B_LOCK"
        if nm.startswith(("inner__", "ctl__", "src__")):
            return "AUDIT_AND_SELECTION_LOCK"
        if nm.startswith("outer__"):
            return "EVALUATION_LOCK"
        return None
    uev = {e.get("unit"): parse_iso(e["at"]) for e in ev if e.get("event") == "unit complete"}
    early, by_lock, no_event = [], {}, []
    for dd in sorted(q for q in UNITS.iterdir() if q.is_dir() and (q / "COMPLETE.json").exists()):
        gov = unit_lock(dd.name)
        if gov is None:
            continue
        by_lock[gov] = by_lock.get(gov, 0) + 1
        fpush = (info.get(gov) or {}).get("first_push_time")
        tm = utc((dd / "COMPLETE.json").stat().st_mtime)
        te = uev.get(dd.name)
        if te is None:
            no_event.append(dd.name)
        if fpush is None or tm < fpush or (te is not None and te < fpush.replace(microsecond=0)):
            early.append({"unit": dd.name, "complete_mtime": tm, "event_at": te, "governing_lock": gov})
    if early:
        fails.append(f"{len(early)} units completed before their governing lock was pushed")
    # predictions
    prel = f"{REL_RES}/PREDICTIONS.json"
    plog = [l_.split("|") for l_ in (git("log", "--format=%H|%cI", "--", prel) or "").splitlines() if l_]
    pfirst = plog[-1] if plog else None
    p_push = first_remote(pfirst[0], entries) if pfirst else None
    stagea_starts = [parse_iso(e["at"]) for e in starts if e["event"] in ("start stagea", "start gate")]
    pred = {"versions": len(plog), "first_commit": pfirst[0] if pfirst else None, "first_push_time": p_push,
            "worktree_equals_first_commit": bool(pfirst) and git("rev-parse", f"{pfirst[0]}:{prel}") ==
            git("hash-object", str(RES / "PREDICTIONS.json")),
            "first_stagea_start": min(stagea_starts) if stagea_starts else None}
    pred["pushed_before_stage_a"] = bool(p_push) and (not stagea_starts or p_push < min(stagea_starts))
    if not (pred["versions"] == 1 and pred["worktree_equals_first_commit"] and pred["pushed_before_stage_a"]):
        fails.append("PREDICTIONS.json not pushed once before Stage A (or revised)")
    # semaphore concurrency from the log (at most two holds at any instant)
    pts = sorted([(h0, 1) for h0, _, _ in holds] + [(h1, -1) for _, h1, _ in holds if h1 is not None])
    cur, peak = 0, 0
    for _, dlt in pts:
        cur += dlt
        peak = max(peak, cur)
    if peak > 2:
        fails.append(f"semaphore log shows {peak} concurrent holds")
    warns = []
    if unreleased:
        warns.append(f"{len(unreleased)} semaphore acquisition(s) without a release record (wrapper ended unlogged; "
                     f"its slot was re-acquired later, so the wrapper had exited; whether its CHILD outlived it cannot be "
                     f"read from the log)")
    ls = git("ls-remote", "origin", f"refs/heads/{BRANCH}", timeout=60)
    remote = {"ls_remote_head": ls.split()[0] if ls else None, "local_head": git("rev-parse", "HEAD")}
    return res("FAIL" if fails else ("WARN" if warns else "PASS"), locks=info, stage_starts=st_out, predictions=pred,
               warnings=warns, unreleased_semaphore_holds=unreleased, reason=(warns[0] if warns and not fails else None),
               units_by_governing_lock=by_lock, units_completed_before_governing_lock=early[:20],
               units_without_activity_event=no_event[:20],
               semaphore={"holds": len(holds), "peak_concurrent_holds": peak,
                          "verifier_holds": sum(1 for _, _, lab in holds if str(lab).startswith("F:"))},
               remote=remote, failures=fails, absent_later_locks=[n for n in LOCK_ORDER if not lf[n]["exists"]])


PHASE1 = ("roles", "teachers", "a1_historical", "a1_converged", "a2_restarts", "a2_assignments", "class_preservation",
          "inner_utility", "capacity_gate", "chronology_stage_a")
PHASE2 = ("stage_b_fine_partitions", "stage_b_search", "stage_b_class_preservation", "inner_audits",
          "source_composition", "selection", "controls")
PHASE3 = ("evaluation_lock", "outer_units", "endpoints", "published_tables", "deployment_restore",
          "decision_documents", "chronology_assessment")


# ------------------------------------------------------------------------------------------------ selftests
def adversarial_rows(K):
    rows = []
    for i in range(K):
        for j in range(i + 1, K):
            p = np.zeros(K)
            p[i] = p[j] = 0.5
            rows.append(p)
    eye = list(np.eye(K))
    under = []
    for i in range(K):
        p = np.full(K, 1e-300)
        p[i] = 1.0 - (K - 1) * 1e-300
        under.append(p)
    tiny = []
    for i in range(K):
        p = np.full(K, 1e-17)
        p[i] = 1.0 - (K - 1) * 1e-17
        tiny.append(p)
    return np.array(rows + eye + under + tiny + [np.full(K, 1.0 / K)])


def synth_probs(rng, n, K, conc=0.6, minor=None):
    P = rng.dirichlet(np.full(K, conc), n)
    if minor is not None:
        P[:, minor] *= 1e-3
        P /= P.sum(1, keepdims=True)
    return P


def selftest_stage_b(rng):
    n = 900
    P1 = synth_probs(rng, n, 2, conc=0.8)
    P2 = synth_probs(rng, n, 6, conc=0.8, minor=5)
    d1, d2 = P1.argmax(1), P2.argmax(1)
    s = ((P1[:, 1] > np.median(P1[:, 1])) ^ (rng.random(n) < 0.3)).astype(np.int64)
    f1, f2 = fit_partition(P1, d1, 2, 6), fit_partition(P2, d2, 6, 5)
    a1, a2 = assign(P1, d1, f1), assign(P2, d2, f2)
    Tf = np.zeros((2, f1.F, f2.F), dtype=np.int64)
    np.add.at(Tf, (s, a1, a2), 1)
    W = wt_joint(0.5)
    out, errs = {}, []

    def fresh_value(lab1, lab2, W_):
        return MyState(f1, f2, Tf, tokens_and_labels(lab1), tokens_and_labels(lab2)).value(W_)

    # random coarse state: merge a few same-class pairs on both recipients
    st = MyState(f1, f2, Tf, np.arange(f1.F), np.arange(f2.F))
    for r in (1, 2):
        for c in range(st.fine[r].K):
            L = st.labels_of_class(r, c)
            if L.size >= 3:
                st.merge(r, int(L[0]), int(L[1]))
    base = st.value(W)
    for r in (1, 2):
        for c in range(st.fine[r].K):
            L, ia, ib, delta, dD, dI, dJ = st.merge_cands(r, c, W)
            for j in range(min(len(ia), 6)):
                lab = {1: st.lab[1].copy(), 2: st.lab[2].copy()}
                a_, b_ = int(L[ia[j]]), int(L[ib[j]])
                lab[r][lab[r] == b_] = a_
                errs.append(abs((fresh_value(lab[1], lab[2], W) - base) - float(delta[j])))
        G = st.G(r)
        for f in range(0, st.fine[r].F, 3):
            res_ = st.move_cands(r, f, W, G)
            if res_ is None:
                continue
            B, delta, *_ = res_
            for j in range(len(B)):
                lab = {1: st.lab[1].copy(), 2: st.lab[2].copy()}
                lab[r][f] = int(B[j])
                errs.append(abs((fresh_value(lab[1], lab[2], W) - base) - float(delta[j])))
    out["candidates_checked"] = len(errs)
    out["max_abs_increment_error"] = float(max(errs)) if errs else None
    # families: telescoping logs, row-level terms, frozen first map, witness dominance
    fines = {1: f1, 2: f2}
    labs, tele, rowd = {}, [], []
    for fam, lam in (("FINE-TASK", None), ("LOCAL", 0.5), ("SEQ-12", 0.5), ("SEQ-21", 0.5), ("JOINT", 0.5),
                     ("CLASS", None)):
        wl = {wf: labs[wf] for wf in WITNESS_FAMS} if fam == "JOINT" else None
        lb, log, stf = my_family(fam, fines, Tf, 2, 3, lam, wl)
        labs[fam] = (lb[1], lb[2])
        po1, po2 = Pol(f1, tokens_from_lab(lb[1])), Pol(f2, tokens_from_lab(lb[2]))
        t1, q1, h1, _ = po1.release(P1, d1)
        t2, q2, h2, _ = po2.release(P2, d2)
        o = objectives(P1, t1, q1, P2, t2, q2, s, lam if lam is not None else 0.0)
        rowd.append(terms_diff(o, stf.terms()))
        if not (np.array_equal(h1, d1) and np.array_equal(h2, d2)):
            errs.append(float("inf"))
        if fam.startswith("SEQ"):
            out[f"{fam}_first_map_never_revised"] = log["baseline_correction"]["first_map_never_revised"]
            for stg in log["stages"]:
                inc = sum(x["increment"] for x in stg["merges"])
                mv = sum(x.get("delta", x.get("increment", 0.0)) for x in stg["moves"])
                tele.append(abs(stg["stage_objective_after_refine"] - stg["stage_objective_after_greedy"] - mv))
                del inc
        if fam == "JOINT":
            fj = F3(stf.terms(), lam)["F_joint"]
            out["joint_dominates_witnesses"] = all(fj <= v + 1e-12 for v in log["witness_F_joint"].values())
            g_ = log["starts"]["JOINT-GREEDY"]
            st0 = MyState(f1, f2, Tf, np.arange(f1.F), np.arange(f2.F)).value(W)
            tele.append(abs(g_["stage_objective_after_greedy"] - st0 - sum(x["increment"] for x in g_["merges"])))
            tele.append(abs(g_["stage_objective_after_refine"] - g_["stage_objective_after_greedy"] -
                            sum(x.get("delta", x.get("increment", 0.0)) for x in g_["moves"])))
            out["joint_I12_below_local"] = F3(stf.terms(), lam)["F_joint"] <= F3(
                MyState(f1, f2, Tf, *labs["LOCAL"]).terms(), lam)["F_joint"] + 1e-12
    out["telescoping_max_abs_error"] = float(max(tele))
    out["engine_vs_row_level_max_abs_diff"] = float(max(rowd))
    # XOR coalition fixture: S = u XOR v with within-class clues u (income) and v (occupation)
    m = 1600
    u, v = rng.integers(0, 2, m), rng.integers(0, 2, m)
    sx = u ^ v
    Px1 = np.stack([0.8 + 0.1 * u + 0.01 * rng.random(m), np.zeros(m)], 1)
    Px1[:, 1] = 1 - Px1[:, 0]
    Px2 = np.full((m, 6), 0.02)
    Px2[:, 0] = 0.8 + 0.05 * v + 0.01 * rng.random(m)
    Px2 /= Px2.sum(1, keepdims=True)
    g1 = fit_partition(Px1, Px1.argmax(1), 2, 4)
    g2 = fit_partition(Px2, Px2.argmax(1), 6, 4)
    Tx = np.zeros((2, g1.F, g2.F), dtype=np.int64)
    np.add.at(Tx, (sx, assign(Px1, Px1.argmax(1), g1), assign(Px2, Px2.argmax(1), g2)), 1)
    fx = {1: g1, 2: g2}
    lab_l, _, st_l = my_family("LOCAL", fx, Tx, 2, 2, 2.0)
    wl = {"FINE-TASK": my_family("FINE-TASK", fx, Tx, 2, 2, None)[0], "LOCAL": lab_l,
          "SEQ-12": my_family("SEQ-12", fx, Tx, 2, 2, 2.0)[0], "SEQ-21": my_family("SEQ-21", fx, Tx, 2, 2, 2.0)[0]}
    wl = {k_: (v_[1], v_[2]) for k_, v_ in wl.items()}
    _, _, st_j = my_family("JOINT", fx, Tx, 2, 2, 2.0, wl)
    tl, tj = st_l.terms(), st_j.terms()
    out["xor_fixture"] = {"local_I1": tl["I1"], "local_I2": tl["I2"], "local_I12": tl["I12"], "joint_I12": tj["I12"],
                          "joint_F_below_local": F3(tj, 2.0)["F_joint"] < F3(tl, 2.0)["F_joint"]}
    ok = (out["max_abs_increment_error"] is not None and out["max_abs_increment_error"] < 1e-12 and
          out["telescoping_max_abs_error"] < 1e-10 and out["engine_vs_row_level_max_abs_diff"] < 1e-12 and
          out["SEQ-12_first_map_never_revised"] and out["SEQ-21_first_map_never_revised"] and
          out["joint_dominates_witnesses"] and np.isfinite(max(errs)) and tl["I12"] > 0.1 and tj["I12"] < tl["I12"] and
          out["xor_fixture"]["joint_F_below_local"])
    return res("PASS" if ok else "FAIL", **out)


def tokens_and_labels(lab):
    """Any per-cell grouping -> valid persistent labels (lowest member index)."""
    return lab_from_tokens(np.asarray(lab))


def selftests():
    out = {}
    rng = np.random.default_rng(20261006)
    # ---- import guard (also for unpickling a project class reference)
    _IN_SELFTEST[0] = True
    tried = []
    for m in ("qpc", "qpc.kmeans", "qpc.stagea", "qpc.utility", "qpc.sema", "dpc", "dpc.partition", "osf.data",
              "smf.audit", "rgj.finalize", "jcv.train", "pnx", "oar.study", "stored_model_eval.pilot_infer", "pcrl"):
        try:
            importlib.import_module(m)
            tried.append((m, False))
        except ImportError:
            tried.append((m, True))
    unp = {}
    for blob, nm in ((b"cqpc.kmeans\nfit\n.", "qpc.kmeans"), (b"cdpc.partition\nFinePartition\n.", "dpc.partition")):
        try:
            pickle.loads(blob)
            unp[nm] = False
        except ImportError:
            unp[nm] = True
        except Exception:  # noqa: BLE001
            unp[nm] = False
    _IN_SELFTEST[0] = False
    loaded = sorted(m for m in sys.modules if m.split(".")[0] in _FORBIDDEN_TOP)
    out["import_guard"] = res("PASS" if all(b for _, b in tried) and all(unp.values()) and not loaded else "FAIL",
                              refused=sum(b for _, b in tried), attempted=len(tried), unpickle_refused=unp,
                              forbidden_loaded=loaded)
    # ---- forward pass equals torch.nn.Sequential; head outputs (binary centring, first-index argmax)
    seq = torch.nn.Sequential(torch.nn.Linear(83, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(),
                              torch.nn.Linear(64, 16))
    sd = {f"enc.1.{k}": v for k, v in seq.state_dict().items()}
    Xs = rng.normal(size=(50, 83)).astype(np.float32)
    with torch.no_grad():
        ref_ = seq(torch.from_numpy(Xs)).double().numpy()
    R = encode(sd, 1, Xs)
    yb = (R[:, 0] > np.median(R[:, 0])).astype(int)
    yk = np.digitize(R[:, 1], np.quantile(R[:, 1], [0.2, 0.4, 0.6, 0.8]))
    hb = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(R, yb)
    hk = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(R, yk)
    cb, pb, db = head_outputs(hb, R)
    ck, pk, dk = head_outputs(hk, R)
    ho = (np.allclose(cb[:, 1] - cb[:, 0], hb.decision_function(R)) and np.array_equal(db, hb.predict(R))
          and np.allclose(ck.sum(1), 0) and np.array_equal(dk, pk.argmax(1)) and np.allclose(pk.sum(1), 1))
    out["forward_pass"] = res("PASS" if np.array_equal(R, ref_) and ho else "FAIL", bitwise_vs_sequential=bool(
        np.array_equal(R, ref_)), head_outputs_ok=bool(ho))
    # ---- smoothing: normalisation, strict argmax for ties / uniform / one-hot / underflow
    ok_s = True
    for K in KS:
        A_ = adversarial_rows(K)
        for row in A_:
            for c in np.flatnonzero(row == row.max()):
                q = smooth(row, c)
                ok_s &= strict_argmax_ok(q, c) and abs(float(q.sum()) - 1.0) <= SUM_TOL and bool((q > 0).all())
    out["smoothing"] = res("PASS" if ok_s else "FAIL", rule="(mean + eps 1 + eps e_c) / (1 + (K+1) eps), eps 1e-12")
    # ---- masked KL equals the direct formula; finite with zeros; roundoff guard
    Pz = np.array([[1.0, 0.0, 0.0], [0.2, 0.3, 0.5], [0.0, 0.5, 0.5]])
    Qz = smooth(np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]]), [0, 1, 2])
    kd = np.array([[sum(p * math.log(p / q) for p, q in zip(pr, qr) if p > 0) for qr in Qz] for pr in Pz])
    km = kl_to_many(Pz, Qz)
    kp = kl_paired(Pz, Qz)
    g0, n0 = guard_nonneg(np.array([0.1, -1e-17, 0.0]))
    try:
        guard_nonneg(np.array([0.1, -1e-6]))
        g_err = False
    except ValueError:
        g_err = True
    out["kl_rules"] = res("PASS" if np.isfinite(km).all() and np.abs(km - kd).max() < 1e-15 and
                          np.abs(kp - np.diag(kd)).max() < 1e-15 and n0 == 1 and g0[1] == 0.0 and g_err else "FAIL",
                          max_abs_diff_vs_direct=float(np.abs(km - kd).max()), roundoff_zeroed=n0,
                          large_negative_refused=g_err)
    # ---- sequential sums: np.add.at equals a Python loop and np.bincount (row order)
    cell = rng.integers(0, 7, 3000)
    W = rng.random(3000)
    loop = np.zeros(7)
    for c_, w_ in zip(cell.tolist(), W.tolist()):
        loop[c_] += w_
    out["sequential_sums"] = res("PASS" if np.array_equal(group_sums(cell, W, 7), loop) and
                                 np.array_equal(np.bincount(cell, weights=W, minlength=7), loop) else "FAIL")
    # ---- source initialisation positions on a hand fixture
    Ph = np.array([[0.9, 0.1], [0.8, 0.2], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.55, 0.45]])
    C0, ir = init_source(Ph, 0, 2)
    want = np.array([[0.8, 0.2], [0.6, 0.4]])            # distinct 5 rows sorted p0 desc; positions 1 and 3
    out["source_init"] = res("PASS" if ir["positions"] == [1, 3] and np.array_equal(C0, want) and
                             ir["distinct_vectors"] == 5 else "FAIL", positions=ir["positions"])
    # ---- historical 20-round rule: coherent statistics, pruning, fallback for an absent class
    P6 = synth_probs(rng, 900, 6, minor=5)
    d6 = P6.argmax(1)
    f20 = fit_partition(P6, d6, 6, 8, starts=("source",), rule="dpc")
    a20 = assign(P6, d6, f20)
    n_, S_, A_ = stats_of(P6, a20, f20.F)
    fb_ok = bool(f20.fb.any()) == bool((np.bincount(d6, minlength=6) == 0).any())
    out["historical_rule_coherent"] = res("PASS" if np.array_equal(n_, f20.n) and np.array_equal(S_, f20.S) and
                                          np.array_equal(A_, f20.A) and fb_ok and bool((f20.n[~f20.fb] > 0).all())
                                          else "FAIL", fallback_classes=[int(x) for x in f20.cls[f20.fb]],
                                          cells=int(f20.F))
    # ---- 200-round rule: coherent best iterate; never worse than the 20-round result from the same start
    f200 = fit_partition(P6, d6, 6, 8, starts=("source",), rule="qpc")
    a200 = assign(P6, d6, f200)
    n2, S2, A2 = stats_of(P6, a200, f200.F)
    pol20, pol200 = direct_policy(f20), direct_policy(f200)
    D20 = float(np.mean(kl_paired(P6, pol20.proto[pol20.cell_token[a20]])))
    D200 = float(np.mean(kl_paired(P6, pol200.proto[pol200.cell_token[a200]])))
    coh = np.array_equal(n2, f200.n) and np.array_equal(S2, f200.S) and np.array_equal(A2, f200.A)
    stops = [x["starts"][0]["stop"] for x in f200.receipt["per_class"] if not x.get("fallback")]
    out["capped_rule_coherent"] = res("PASS" if coh and D200 <= D20 + 1e-12 else "FAIL", D_20=D20, D_200=D200,
                                      stops=stops)
    # ---- stop rules: cap with the final pass, relative tolerance, best iterate (later ties), kpp details
    Pst = synth_probs(rng, 1500, 2, conc=0.4)
    Pst = Pst[Pst.argmax(1) == 0]
    C0s, _ = init_source(Pst, 0, 12)
    Qc, ac, rc = lloyd_qpc(Pst, 0, C0s, rounds=2)
    Qf, af, rf = lloyd_qpc(Pst, 0, C0s, rounds=200)
    Qt, at, rt = lloyd_qpc(Pst, 0, C0s, rounds=200, rtol=1.0)          # every pass "small" -> tolerance after 4 passes
    best_ok = all(rr["J_best"] == min(rr["J_trace"]) and rr["J_trace"][rr["best_pass"] - 1] == rr["J_best"] and
                  all(J_ > rr["J_best"] for J_ in rr["J_trace"][rr["best_pass"]:]) for rr in (rc, rf, rt))
    cap_ok = rc["assignment_passes"] == 3 and rc["stop"] in ("cap", "assignment_fixed_point") and \
        rc["converged"] == (rc["stop"] == "assignment_fixed_point")
    tol_ok = rt["stop"] == "relative_tolerance" and rt["assignment_passes"] == 5 and rt["converged"]
    fix_ok = rf["stop"] in ("assignment_fixed_point", "relative_tolerance") and rf["assignment_passes"] < 200
    coh_ok = np.array_equal(kl_to_many(Pst, Qf).argmin(1), af)
    U_, w_ = distinct_counts(np.repeat(Pst[:40], 3, axis=0))
    Un, wn = np.unique(np.repeat(Pst[:40], 3, axis=0), axis=0, return_counts=True)
    uniq_ok = np.array_equal(U_, Un) and np.array_equal(w_, wn)
    Pdeg = np.vstack([np.repeat([[0.9, 0.1]], 50, axis=0), [[0.8, 0.2]], [[0.7, 0.3]]])
    Cd, idg = init_kmeanspp(Pdeg, 0, 3, KPP_SEEDS[0])
    deg_ok = len(idg["chosen_distinct_index"]) == 3 == len(set(idg["chosen_distinct_index"]))
    tie_ok = True
    J0 = 10.0
    for Jn, expect in ((J0 - 1e-12 * J0 * 0.5, 0), (J0 - 1e-12 * J0 * 2, 1)):
        objs = [J0, Jn]
        w_ = 0 if not objs[1] < objs[0] - 1e-12 * max(abs(objs[0]), 1.0) else 1
        tie_ok &= w_ == expect
    out["stop_rules"] = res("PASS" if best_ok and cap_ok and tol_ok and fix_ok and coh_ok and uniq_ok and deg_ok and tie_ok
                            else "FAIL", best_iterate=bool(best_ok), cap_with_final_pass=bool(cap_ok),
                            relative_tolerance=bool(tol_ok), fixed_point=bool(fix_ok), coherent=bool(coh_ok),
                            distinct_counts_equal_np_unique=bool(uniq_ok), kpp_distinct_centres=bool(deg_ok),
                            start_tie_tolerance=bool(tie_ok), passes={"cap2": rc["assignment_passes"],
                                                                      "full": rf["assignment_passes"],
                                                                      "tol": rt["assignment_passes"]})
    # ---- restarts: determinism and lowest-distortion selection with first-start ties
    fA = fit_partition(P6, d6, 6, 8, starts=START_ORDER, rule="qpc")
    fB = fit_partition(P6, d6, 6, 8, starts=START_ORDER, rule="qpc")
    det = all(fA.equal(fB).values())
    sel_ok = True
    for pc in fA.receipt["per_class"]:
        if pc.get("fallback"):
            continue
        objs = [s_["J"] for s_ in pc["starts"]]
        w_ = 0
        for j_ in range(1, len(objs)):                   # registered: later start wins only beyond 1e-12 max(|J|, 1)
            if objs[j_] < objs[w_] - 1e-12 * max(abs(objs[w_]), 1.0):
                w_ = j_
        sel_ok &= pc["winner"] == START_ORDER[w_]
    _, cand_a, _ = fit_class(P6[d6 == 0], 0, 8, START_ORDER, "qpc")
    _, cand_b, _ = fit_class(P6[d6 == 0], 0, 8, (START_ORDER[1], START_ORDER[1]), "qpc")
    seeds_differ = not np.array_equal(cand_a[1]["a"], cand_a[2]["a"]) or \
        cand_a[1]["init"]["chosen_distinct_index"] != cand_a[2]["init"]["chosen_distinct_index"]
    same_seed_same = np.array_equal(cand_b[0]["a"], cand_b[1]["a"])
    winners = {}
    for pc in fA.receipt["per_class"]:
        if not pc.get("fallback"):
            winners[pc["winner"]] = winners.get(pc["winner"], 0) + 1
    out["restarts"] = res("PASS" if det and sel_ok and seeds_differ and same_seed_same else "FAIL",
                          deterministic=bool(det), winner_is_first_minimum=bool(sel_ok), seeds_differ=bool(seeds_differ),
                          same_seed_identical=bool(same_seed_same), winners=winners)
    # ---- larger capacities create new cells; caps limited by distinct vectors; asymmetric caps
    P2c = synth_probs(rng, 4000, 2, conc=1.0)
    d2c = P2c.argmax(1)
    caps = {m: fit_partition(P2c, d2c, 2, m, rule="qpc").F for m in (4, 8, 32, 64)}
    Pdup = np.repeat(synth_probs(rng, 10, 6), 30, axis=0)
    ddup = Pdup.argmax(1)
    fdup = fit_partition(Pdup, ddup, 6, 64, rule="qpc")
    distinct_cap = all(int(np.sum(fdup.cls == c)) <= len(distinct_lex(Pdup[ddup == c])) for c in range(6)
                       if (ddup == c).any())
    asym = (fit_partition(P2c, d2c, 2, 4, rule="qpc").F <= 8 and fit_partition(P6, d6, 6, 32, rule="qpc").F > 6 * 8 // 2)
    out["capacity"] = res("PASS" if caps[4] < caps[8] < caps[32] < caps[64] and caps[64] > 32 and distinct_cap and asym
                          else "FAIL", income_cells_by_cap=caps, distinct_vector_cap_respected=bool(distinct_cap),
                          asymmetric_caps=bool(asym))
    # ---- class preservation: adversarial rows through every fitted partition (ties, one-hot, underflow, unseen class)
    cp_ok = True
    for fine in (f20, f200, fA):
        pol = direct_policy(fine)
        Pa = adversarial_rows(6)
        da = Pa.argmax(1)
        tok, q, hard, _ = pol.release(Pa, da)
        cp_ok &= bool(np.array_equal(hard, da) and np.array_equal(q.argmax(1), da) and (tok >= 0).all())
        cp_ok &= all(pol.structure()[k_] for k_ in ("no_class_mixing", "all_ids_used", "canonical_ids",
                                                    "every_class_has_a_token", "prototype_strict_argmax_is_class",
                                                    "prototype_sums_within_tol", "fallback_tokens_have_no_fitting_rows"))
    co = class_only_policy(6, P6, d6)
    t_, q_, h_, _ = co.release(P6, d6)
    cp_ok &= bool(np.array_equal(h_, d6) and np.array_equal(q_.argmax(1), d6))
    out["class_preservation"] = res("PASS" if cp_ok else "FAIL",
                                    cases="two-way ties, one-hot, 1e-300 underflow, 1e-17 tiny, uniform, absent class")
    # ---- MI: entropy form, relabelling invariance, token collision, XOR coalition, null
    s = rng.integers(0, 2, 5000)
    c = (rng.integers(0, 5, 5000) + s) % 7
    e1 = abs(mi_of(s, c) - entropy_mi(s, c))
    e2 = abs(mi_of(s, c) - mi_of(s, rng.permutation(7)[c]))
    a1 = rng.integers(0, 2, 5000)
    xor = mi_of(s, a1) < 0.005 and mi_of(s, a1 ^ s) < 0.005 and mi_of(s, a1, a1 ^ s) > 0.6
    coll = mi_of(s, 2 * a1 + s) > mi_of(s, a1) + 0.5       # two IDs per decoded vector still disclose s
    null = mi_of(rng.integers(0, 2, 5000), rng.integers(0, 4, 5000)) < 0.005
    out["plug_in_mi"] = res("PASS" if e1 < 1e-12 and e2 < 1e-15 and xor and coll and null else "FAIL",
                            vs_entropy_form=e1, relabelling=e2, coalition_xor=bool(xor), token_collision=bool(coll),
                            null_small=bool(null))
    # ---- engine: objectives from labels equal row-level objectives; merge delta vs brute force; stage-one rule
    sx = rng.integers(0, 2, 900)
    P1e = synth_probs(rng, 900, 2)
    d1e = P1e.argmax(1)
    g1 = fit_partition(P1e, d1e, 2, 6)
    g2 = fit_partition(P6, d6, 6, 4)
    a1e, a2e = assign(P1e, d1e, g1), assign(P6, d6, g2)
    eng = Engine(g1, g2, a1e, a2e, sx)
    lab = {1: np.arange(g1.F), 2: np.arange(g2.F)}
    t_eng = eng.terms(lab)
    q1r = smooth(g1.S[a1e] / g1.n[a1e, None], d1e)
    q2r = smooth(g2.S[a2e] / g2.n[a2e, None], d6)
    o_row = objectives(P1e, a1e, q1r, P6, a2e, q2r, sx, 0.1)
    e_eng = max(abs(t_eng[k_] - o_row[k_]) for k_ in ("D1", "D2", "I1", "I2", "I12"))
    # merge two same-class cells of recipient 2 and compare the engine delta with brute-force row recomputation
    c2 = [int(x) for x in np.flatnonzero(g2.cls == int(np.bincount(g2.cls).argmax()))[:2]]
    lab_m = {1: lab[1].copy(), 2: lab[2].copy()}
    lab_m[2][c2[1]] = c2[0]
    lab_m[2] = np.unique(lab_m[2], return_inverse=True)[1]
    t_m = eng.terms(lab_m)
    tok2 = lab_m[2][a2e]
    G2 = int(lab_m[2].max()) + 1
    n_m, S_m, _ = stats_of(P6, tok2, G2)
    cls_m = np.zeros(G2, dtype=np.int64)
    cls_m[lab_m[2]] = g2.cls
    q_m = smooth(S_m[tok2] / n_m[tok2, None], cls_m[tok2])
    o_m = objectives(P1e, a1e, q1r, P6, tok2, q_m, sx, 0.1)
    e_delta = abs((F_of(t_m, 0.1, "joint") - F_of(t_eng, 0.1, "joint")) - (o_m["F_joint"] - o_row["F_joint"]))
    st1 = stage1_objective(eng, 1, lab[1], 0.1)
    co2 = class_only_policy(6, P6, d6)
    t2co, q2co, _, _ = co2.release(P6, d6)
    o_st = objectives(P1e, a1e, q1r, P6, t2co, q2co, sx, 0.1)
    e_st = abs(st1 - o_st["F_joint"])
    out["objective_engine"] = res("PASS" if e_eng < 1e-12 and e_delta < 1e-12 and e_st < 1e-12 else "FAIL",
                                  engine_vs_rows=e_eng, merge_delta_vs_brute_force=e_delta,
                                  stage_one_class_only_counterpart_vs_rows=e_st)
    # ---- own Stage B engine: candidate increments vs brute force, telescoping logs, row-level terms, families
    out["stage_b_engine"] = selftest_stage_b(rng)
    # ---- utility / gate / rate selection on synthetic numbers
    y = rng.integers(0, 6, 400)
    Pu = synth_probs(rng, 400, 6)
    u = my_utility(Pu, Pu.argmax(1), y, 6, 0)
    Y = np.eye(6)[y]
    sk_ok = abs(u["logloss"] - sk_log_loss(y, Pu, labels=list(range(6)))) < 1e-12 and \
        abs(u["brier"] - float(np.mean(np.sum((Pu - Y) ** 2, 1)))) < 1e-15
    uU = {"acc": 0.80, "logloss": 0.40, "brier": 0.25, "const_acc": 0.30}
    edge = gate_task({"acc": 0.80, "logloss": 0.41, "brier": 0.255, "const_acc": 0.30}, uU)
    over = gate_task({"acc": 0.80, "logloss": 0.4101, "brier": 0.25, "const_acc": 0.30}, uU)
    weak = gate_task({"acc": 0.79, "logloss": 0.40, "brier": 0.25, "const_acc": 0.77}, uU)
    gate_ok = edge["eligible"] and not over["eligible"] and not weak["eligible"] and not weak["checks"]["gain"]

    def row(eligible, head, states, occ, inc):
        return {"eligible": eligible, "headroom_all_seeds": head, "mean_states": states, "worst_occ_ll": occ,
                "worst_income_ll": inc, "mean_income_ll": inc, "worst_normalized_excess": occ}
    tab = {"U|DIRECT-TASK|i4o16": row(True, False, 30, 0.9, 0.3), "U|DIRECT-TASK|i4o32": row(True, True, 50, 0.8, 0.3),
           "U|DIRECT-TASK|i8o16": row(True, True, 40, 0.85, 0.3), "U|DIRECT-TASK|i4o8": row(False, False, 20, 1, 0.3)}
    sel = select_rates(tab)
    one = select_rates({k_: v for k_, v in tab.items() if k_ != "U|DIRECT-TASK|i4o32"})
    none = select_rates({"U|DIRECT-TASK|i4o8": tab["U|DIRECT-TASK|i4o8"]})
    rs_ok = sel["selected"] == ["U|DIRECT-TASK|i8o16", "U|DIRECT-TASK|i4o32"] and \
        one["selected"] == ["U|DIRECT-TASK|i8o16", "U|DIRECT-TASK|i4o16"] and \
        none["decision"] == "CAPACITY_GATE_NOT_MET" and q_star(tab) == "U|DIRECT-TASK|i4o32"
    out["utility_gate_selection"] = res("PASS" if sk_ok and gate_ok and rs_ok else "FAIL", sklearn_parity=bool(sk_ok),
                                        gate_truth=bool(gate_ok), rate_selection=bool(rs_ok),
                                        note="rate-selection income tie-break statistic pending the STAGE_A_LOCK text")
    # ---- inference constants, AUC, group bootstrap
    zc = z_primary()
    ys = rng.integers(0, 2, 800)
    sc = rng.normal(size=800) + 0.5 * ys
    sc[:100] = np.round(sc[:100], 1)                    # ties
    auc_ok = abs(my_auc(ys, sc) - roc_auc_score(ys, sc)) < 1e-12 and \
        abs(weighted_auc(sc, ys == 1, np.ones(800)) - my_auc(ys, sc)) < 1e-12
    grp = rng.integers(0, 600, 800)
    b1, b2 = GroupBoot(grp, B=50, seed=BOOT_SEED), GroupBoot(grp, B=50, seed=BOOT_SEED)
    boot_ok = np.array_equal(b1.counts, b2.counts) and bool((b1.counts.sum(1) == b1.G).all())
    w0 = b1.row_weights(0)
    rep = np.repeat(np.arange(800), w0.astype(int))
    wa_ok = abs(weighted_auc(sc, ys == 1, w0) - my_auc(ys[rep], sc[rep])) < 1e-12
    out["inference_primitives"] = res("PASS" if zc == Z_PRIMARY and auc_ok and boot_ok and wa_ok else "FAIL",
                                      z=zc, z_equals_registered=zc == Z_PRIMARY, slots=N_ENDPOINTS,
                                      auc_vs_sklearn=bool(auc_ok), bootstrap_deterministic=bool(boot_ok),
                                      weighted_auc_equals_expanded=bool(wa_ok))
    return out


# ------------------------------------------------------------------------------------------------ main
def scrub_check(text: str):
    home = str(Path.home())
    probes = [home, "/Users/", "/Volumes/", home.split("/")[-1], "BackgroundSync"]
    low = text.lower()
    bad = [s_ for s_ in probes if s_ and s_.lower() in low]
    if bad:
        raise RuntimeError(f"refusing to write identifying strings into the public JSON: {len(bad)} hits")


def walk_flags(checks):
    counts_all, flagged = {}, []

    def walk(node, path):
        if isinstance(node, dict):
            st_ = node.get("status")
            if isinstance(st_, str) and st_ in STATUS_RANK:
                counts_all[st_] = counts_all.get(st_, 0) + 1
                if st_ in ("FAIL", "WARN"):
                    leaf = not any(isinstance(v, dict) and v.get("status") in ("FAIL", "WARN") for v in node.values())
                    if leaf:
                        flagged.append({"path": path, "status": st_, "cause": node.get("failures") or node.get("failing")
                                        or node.get("reason") or node.get("note")})
            for k, v in node.items():
                walk(v, f"{path}.{k}" if path else k)

    for k, v in checks.items():
        walk(v, k)
    return counts_all, flagged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, default=0, choices=(0, 1, 2, 3))
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-refit", action="store_true", help="skip the descriptive own head refits")
    ap.add_argument("--label", default=None, help="phase label written to the report (e.g. PHASE_2A)")
    args = ap.parse_args()
    t0, c0 = time.time(), time.process_time()
    others_start = heavy_processes()
    report = {"schema": "qpc-independent-verification-v1", "phase": args.label or f"PHASE_{args.phase}",
              "generated_at": iso(datetime.now(timezone.utc)),
              "verifier": f"{REL_RES}/verification/replay_qpc.py", "verifier_sha256": sha_file(Path(__file__)),
              "worktree_head": git("rev-parse", "HEAD"), "branch": BRANCH, "source_evidence_sha": SOURCE_SHA,
              "inputs": {"source_npz": "<PRIVATE_CACHE>/jcv_v1/inputs/adult_jcv.npz",
                         "qpc_units": "<PRIVATE_CACHE>/qpc_v1/run/units", "dpc_units": "<PRIVATE_CACHE>/dpc_v1/run/units",
                         "dpc_admitted": "<PRIVATE_CACHE>/dpc_v1/admitted"},
              "libraries": {"numpy": np.__version__, "torch": torch.__version__, "joblib": joblib.__version__,
                            "sklearn": __import__("sklearn").__version__, "scipy": __import__("scipy").__version__}}
    checks = {}
    st = selftests()
    checks["selftests"] = {"status": worst(*[v["status"] for v in st.values()]), **st}
    for k in PHASE1:
        checks[k] = phase1_pending(k)
    for k in PHASE2:
        checks[k] = res("PENDING", reason="PHASE 2 (after Stage B fits and inner selection)")
    for k in PHASE3:
        checks[k] = res("PENDING", reason="PHASE 3 (after the assessment and inference)")
    report["real_data_read"] = args.phase >= 1
    if args.phase >= 1:
        D = Data()
        L = D.labels()
        qman = RES / "ROLE_MANIFEST.json"
        dman = git_show_bytes(SOURCE_SHA, f"{DPC_REL}/ROLE_MANIFEST.json")
        manifests = {"qpc_ROLE_MANIFEST": jload(qman) if qman.exists() else None,
                     "dpc_ROLE_MANIFEST_at_source_sha": json.loads(dman) if dman else None}
        checks["roles"] = check_roles(D, manifests)
        checks["teachers"], T = check_teachers(D, L, refit=not args.no_refit)
        refs = {}
        checks["references"] = check_references(D, T, keep=refs)
        checks["a1_source_replay"], a1_keep = check_a1_source_replay(D, T)
        checks["unit_custody"] = check_units_complete()
        checks["chronology_stage_a"] = check_chronology()
        checks["code_hashes"] = check_code_hashes()
        a1_src = {(k, r): f for (k, r), f in a1_keep.items()}
        if not any(UNITS.glob("a1__*")):
            for k in ("a1_historical", "a1_converged", "a2_restarts", "a2_assignments", "class_preservation",
                      "inner_utility", "capacity_gate"):
                checks[k] = res("PENDING", reason="Stage A units not yet present")
        else:
            sa, util, sa_cpu = check_stage_a(D, L, T, a1_src)
            checks.update(sa)
            checks["a1_converged"] = summarize_a1(sa["a1_historical"], sa["a2_restarts"])
            checks["inner_utility"] = check_inner_utility(util)
            checks["capacity_gate"] = check_gate(util, D)
            checks["capacity_curve"] = check_capacity_curve(util, checks["capacity_gate"])
            checks["mutation_power"] = mutation_power_stage_a(D, T, L)
            checks["unit_custody"] = check_units_complete(("tea__", "ref__", "a1__", "dir__", "pol__"))
            report["own_stage_a_fit_cpu_s"] = sa_cpu
        if args.phase >= 2:
            if any(UNITS.glob("fine__*")):
                sb, own_b_maps, own_b_terms = check_stage_b(D, L, T)
                checks.update(sb)
                checks["unit_custody"] = check_units_complete(("tea__", "ref__", "a1__", "dir__", "pol__", "fine__"))
            else:
                checks["stage_b_fine_partitions"] = res("PENDING", reason="Stage B units not yet present")
        if args.phase >= 3:
            refs_ok = refs
            checks["inner_audits"], checks["source_composition"], own_in = check_inner(D, L, T, refs_ok)
            checks["selection"], sel_mine = check_selection(own_in)
            gate = evaluation_lock_gate(fetch=True)
            report["assessment_label_gate"] = {k_: gate.get(k_) for k_ in ("checked_at", "git_fetch_ok", "exists",
                                                                         "commit", "commit_time", "first_push_time",
                                                                         "origin_show_equals_worktree", "commit_on_origin",
                                                                         "commits", "sha256", "ok")}
            checks["evaluation_lock"], EL = check_eval_lock(D, L, gate, sel_mine)
            LU = unsealed_labels(D, gate)
            checks["controls"] = check_controls(D, L, EL)
            checks["outer_units"], preds = check_outer(D, LU, T, refs_ok, own_in, EL, gate)
            checks["endpoints"], eps, lev = check_endpoints(preds, EL)
            checks["published_tables"] = check_tables(own_in, eps, lev, sel_mine, preds, EL)
            checks["chronology_assessment"] = check_late_chronology(gate, EL)
            if not args.no_refit:
                checks["attacker_refits"] = check_refits(D, L, LU, T, own_in, preds, EL)
            checks["unit_custody"] = check_units_complete(("tea__", "ref__", "a1__", "dir__", "pol__", "fine__", "inner__",
                                                           "outer__"))
    loaded = sorted(m for m in sys.modules if m.split(".")[0] in _FORBIDDEN_TOP)
    report["independence"] = res("PASS" if not loaded and not _BLOCKED and not _PRELOADED else "FAIL",
                                 guard="sys.meta_path finder refusing " + ", ".join(_FORBIDDEN_TOP) +
                                       " (also during joblib unpickling); torch.load(weights_only=True); worktree root "
                                       "removed from sys.path; semaphore invoked by path as the parent process",
                                 loaded_forbidden_modules=loaded, blocked_attempts_during_run=_BLOCKED,
                                 preloaded_before_guard=_PRELOADED,
                                 libraries_used=["numpy", "scipy", "sklearn", "torch", "joblib", "stdlib"],
                                 adapted_from_predecessor_verifier="import guard, role reconstruction, forward "
                                                                   "application, lock-chronology helpers")
    report["checks"] = checks
    report["verifier_corrections"] = VERIFIER_CORRECTIONS
    status = {k: (v.get("status") if isinstance(v, dict) else None) for k, v in checks.items()}
    counts_all, flagged = walk_flags(checks)
    top = {}
    for s_ in status.values():
        top[s_] = top.get(s_, 0) + 1
    report["summary"] = {"status_by_check": status, "top_level_counts": top, "status_counts_all_nodes": counts_all,
                         "flagged_fail_warn": flagged, "independence": report["independence"]["status"],
                         "overall": worst(*[s_ for s_ in status.values() if s_ not in ("PENDING", "INFO")],
                                          report["independence"]["status"])}
    report["compute"] = {"wall_s": round(time.time() - t0, 2), "cpu_s_process": round(time.process_time() - c0, 2),
                         "threads": {"OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "torch": torch.get_num_threads()},
                         "heavy": args.phase != 0, "other_qpc_processes_at_start": len(others_start),
                         "other_qpc_processes_at_end": len(heavy_processes()),
                         "semaphore_slots_at_end": sema_status()}
    report["pending"] = sorted(k for k, v in status.items() if v == "PENDING")
    sema_runs = [{"label": e.get("label"), "released_at": e.get("at"), "wall_s": e.get("wall_s"), "cpu_s": e.get("cpu_s"),
                  "rc": e.get("rc")} for e in jsonl(RUN / "SEMA_LOG.jsonl")
                 if e.get("event") == "release" and str(e.get("label", "")).startswith("F:")]
    light = VERIFIER_RUNS + [{"run": f"this run (phase {args.phase})", "start": report["generated_at"],
                              "wall_s": report["compute"]["wall_s"], "cpu_s": report["compute"]["cpu_s_process"],
                              "heavy": args.phase != 0, "note": "inside the semaphore hold" if args.phase else "light"}]
    report["verifier_compute_ledger"] = {
        "semaphore_holds_completed": sema_runs,
        "semaphore_cpu_s_total": round(sum(float(r["cpu_s"] or 0) for r in sema_runs), 1),
        "semaphore_wall_s_total": round(sum(float(r["wall_s"] or 0) for r in sema_runs), 1),
        "this_and_light_runs": light,
        "note": "every real-data verifier process ran inside a semaphore hold labelled F:*; self-tests (synthetic, "
                "< 1 CPU-s) ran outside; the current run's hold is released after this file is written"}
    text = json.dumps(jsonable(report), indent=1)
    scrub_check(text)
    if not args.no_write:
        dest = Path(args.out) if args.out else OUT
        tmp = dest.with_suffix(".json.tmp")
        tmp.write_text(text + "\n")
        tmp.replace(dest)
    print(json.dumps(jsonable({"summary": report["summary"], "compute": report["compute"]}), indent=1))


if __name__ == "__main__":
    main()
