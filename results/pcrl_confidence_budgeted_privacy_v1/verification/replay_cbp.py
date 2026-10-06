#!/usr/bin/env python3
"""Independent verifier for the confidence-budgeted privacy study (cbp).

Owner: role F (independent verifier and custody owner; prompt section 14). Exclusive files: this script and
results/pcrl_confidence_budgeted_privacy_v1/INDEPENDENT_VERIFICATION.json.

Provenance of the code. ADAPTED (not rewritten) from the predecessor's independent verifier
results/pcrl_confidence_capacity_v1/verification/replay_qpc.py at the source tip d0c8a45 (itself independent code,
written by an earlier verifier from the registered definitions): the import guard, the role reconstruction, the own
forward pass, smoothing / KL / KL k-means with the registered starts and stop rules, the own coarse-state search engine
(merges, exchange moves, sequential class-only first stage, JOINT starts and witnesses), plug-in MI, utility, AUC,
the exact-record-group bootstrap, the inner-audit and composition replay, the attacker refits and the lock-chronology
helpers. NEW here, written from the cbp prompt (sections 6-15) and to be reconciled against the cbp locks as they are
written: the six-lambda bank (0.01, 0.025, 0.04, 0.06, 0.08, 0.1) with reuse / new-fit accounting, the 11 composition-
only public maps, ordinary AND headroom eligibility per seed and task, the three registered shortfalls and the fixed
fallback order, the cbp roles T* (closed privacy-untrained list, RAW-J excluded), C_rate, C_global, P*, J*, Q and the
prespecified diagnostics, the cbp label truth table, the bootstrap seed 20261008, the SEMA_LOG budget audit, the
restore replay from the cbp copy and the deliberate-defect tests of prompt section 14.

Independence:
  * a sys.meta_path guard refuses every import under cbp, qpc, dpc, osf, smf, rgj, jcv, pnx, oar, stored_model_eval
    and pcrl, including imports triggered while unpickling joblib heads; torch.load always uses weights_only=True; the
    run asserts at the end that none of these packages was loaded;
  * the worktree root is removed from sys.path; the shared semaphore is invoked BY PATH as the parent process with -P
    (OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ~/PCRL/.venv/bin/python -P <WORKTREE>/cbp/sema.py
    --label F:<what> -- <this command>), never imported;
  * libraries: numpy, scipy, scikit-learn, torch, joblib and the standard library.

Phases (status PASS / FAIL / WARN / INFO / PENDING):
  PHASE_0  scaffold + synthetic self-tests (incl. the deliberate-defect tests); no real data is read
  PHASE_1  pins, roles, teachers (own forward pass), references, custody of the 84 admitted units (bytes vs the qpc
           store, the qpc copy v2 SHA256SUMS and the qpc EVALUATION_LOCK), own replay of the fine partitions and of
           every admitted code (DIRECT-TASK at all 8 rates, FINE-TASK, CLASS-ONLY, LOCAL / SEQ-12 / SEQ-21 / JOINT at
           lambda 0.01, 0.1 and 1): partitions, search logs, tokens, decoded vectors, decisions, final objective terms,
           class preservation; chronology
  PHASE_2  the 48 new intermediate-lambda fits (same replay), inner audits (every stored candidate, selected attackers),
           U composition closure over all 38 public maps, controls, ordinary and headroom eligibility for every seed and
           configuration, roles, nominees, fallbacks, diagnostics -- finished BEFORE the assessment opens
  PHASE_3  EVALUATION_LOCK (fetch + show before any assessment label), outer units, the 37 endpoints and labels,
           tables / figures, attacker refits, deployment, restore from the copy, budget and process counts

Usage (from the worktree root; every run, including PHASE_0, under the semaphore):
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ~/PCRL/.venv/bin/python -P <WORKTREE>/cbp/sema.py \\
        --label F:phase0 -- ~/PCRL/.venv/bin/python \\
        <WORKTREE>/results/pcrl_confidence_budgeted_privacy_v1/verification/replay_cbp.py --phase 0
Writes results/pcrl_confidence_budgeted_privacy_v1/INDEPENDENT_VERIFICATION.json (aggregates, hashes and placeholders
only; finite JSON) unless --no-write.
"""


from __future__ import annotations

import importlib
import importlib.abc
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

# ------------------------------------------------------------------------------------------------ import guard
_FORBIDDEN_TOP = ("cbp", "qpc", "dpc", "osf", "smf", "rgj", "jcv", "pnx", "oar", "stored_model_eval", "pcrl")
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
REL_RES = "results/pcrl_confidence_budgeted_privacy_v1"
OUT = RES / "INDEPENDENT_VERIFICATION.json"
CACHE = Path.home() / "PCRL_eval_cache_private"
SRC = CACHE / "jcv_v1" / "inputs" / "adult_jcv.npz"
SRC_SHA = "e0d9e54af780f30788ee29cfe6795ec82cbdcadc127b1978c69a3891485d2f12"
PRIV = CACHE / "cbp_v1"
RUN = PRIV / "run"
UNITS = RUN / "units"
ADM = PRIV / "admitted"
QPC_PRIV = CACHE / "qpc_v1"
QPC_UNITS = QPC_PRIV / "run" / "units"
QPC_ADM = QPC_PRIV / "admitted"
QPC_COPY = CACHE / "qpc_v1_local_copy_20261006_v2"
DPC_PRIV = CACHE / "dpc_v1"
DPC_UNITS = DPC_PRIV / "run" / "units"
DPC_ADMITTED = DPC_PRIV / "admitted"
ADMISSION_NORM = WT / "results" / "pcrl_joint_complete_view_method_v1" / "DATA_ADMISSION.json"
BRANCH = "research/pcrl-confidence-budgeted-privacy-v1"
QPC_EVIDENCE = "9dd06da6b64e558e1c079f76e43982b60b327e63"      # qpc evidence commit (source of every admitted unit)
QPC_TIP = "d0c8a45c879d01fb8b736ccc091ec3e2c3e9b351"           # qpc handoff tip = this branch's base
QPC_REL = "results/pcrl_confidence_capacity_v1"
SOURCE_SHA = "0a7b05a52746544213742f50efd0a48167efffb1"        # dpc evidence (roles, teacher units, references)
TEACHER_PIN = "925e0fddfcb666116c6179575339728a324ed78e"       # osf teacher provenance
DPC_REL = "results/pcrl_decision_preserving_compression_v1"
OSF_REL = "results/pcrl_online_strength_frontier_v1"
STUDY_START = "2026-10-06T16:44:45Z"

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
N_COLUMNS = 83

# fixed release / method rules (prompt sections 7-8; qpc METHOD_CARD, unchanged)
EPS = 1e-12                  # smoothing
LL_CLIP = 1e-12              # log-loss clip
SUM_TOL = 1e-12              # prototype normalisation
KL_ROUNDOFF = 1e-12          # |negative KL| below this is roundoff (k-means++ weights); larger negatives are errors
SPARSE_N = 5
A1_ROUNDS = 20               # historical dpc rule (kept for the self-tests of the k-means engine only)
A2_ROUNDS = 200              # registered qpc cap
PATIENCE = 3
RTOL = 1e-9
KPP_SEEDS = (20261006, 20261007)
START_ORDER = ("source", f"kpp:{KPP_SEEDS[0]}", f"kpp:{KPP_SEEDS[1]}")
RATE = (8, 64)               # the ONE rate (prompt section 7)
FINE_CAPS = (32, 128)        # admitted fine partitions (income, occupation) per predicted class
LAMS = (0.01, 0.025, 0.04, 0.06, 0.08, 0.1)
REUSED_LAMS = (0.01, 0.1)
NEW_LAMS = (0.025, 0.04, 0.06, 0.08)
PRIV_FAMS = ("LOCAL", "SEQ-12", "SEQ-21", "JOINT")
NONJOINT = ("DIRECT-TASK", "FINE-TASK", "LOCAL", "SEQ-12", "SEQ-21")
SWEEPS = 5
RATES_INCOME = (4, 8)        # composition-only DIRECT-TASK rates (admitted qpc Stage A codes)
RATES_OCC = (8, 16, 32, 64)
COMPOSED_EXTRA_LAM = 1.0     # composition-only qpc privacy maps at lambda 1

# eligibility contract (prompt section 9). Ordinary = the unchanged source rule; HEADROOM = the new tighter nominee rule
GATE = {"acc_drop": 0.01, "ll_excess": 0.01, "brier_excess": 0.005, "retain": 0.8, "gain": 0.03}
HEADROOM = {"ll_excess": 0.006, "brier_excess": 0.0035}
BUFFER = 0.005               # local guard: individual inner AUC <= comparator + 0.005, per recipient and seed
T_STAR_LIST = ("U|DIRECT-TASK|i8o64", "U|FINE-TASK|i8o64", "SRC|U", "U|CLASS|i1o1", "REF|F0")   # RAW-J excluded
ORDER_DECIMALS = 12          # provisional floating tie rule of the ordering (qpc convention; reconciled with the lock)

# inference (prompt section 11)
N_ENDPOINTS = 37
Z_PRIMARY = 3.2048452050105634
B_BOOT, BOOT_SEED = 1999, 20261008
MARGIN_PAIR, MARGIN_LOCAL = 0.02, 0.01

STATUS_RANK = {"FAIL": 4, "PENDING": 3, "WARN": 2, "PASS": 1, "INFO": 0, "NOT_APPLICABLE": 0}

# own full-row releases of every replayed code: OWN_REL[(seed, cid)] = {tok1, q1, hard1, tok2, q2, hard2, alpha1, alpha2}
OWN_REL: dict = {}
# every correction made to this verifier's own code (kept; disclosed in the report)
VERIFIER_CORRECTIONS: list = []
# the verifier's own light runs outside the semaphore (none: every run, including the self-tests, holds a slot)
VERIFIER_RUNS: list = []


def g(x) -> str:
    return f"{x:g}"


def direct_id(m1, m2):
    return f"U|DIRECT-TASK|i{m1}o{m2}"


def b_config(fam, lam=None):
    if fam == "CLASS":
        return "U|CLASS|i1o1"
    if fam == "DIRECT-TASK":
        return direct_id(*RATE)
    return f"U|{fam}|i{RATE[0]}o{RATE[1]}" + (f"|l{lam:g}" if fam in PRIV_FAMS else "")


def privacy_cids(lams=LAMS):
    return [b_config(f, lam) for lam in lams for f in PRIV_FAMS]


def task_cids():
    return [direct_id(*RATE), b_config("FINE-TASK"), b_config("CLASS")]


def code_cids():
    """The 27 registered codes per seed (prompt section 7): 3 task-only references + 6 lambdas x 4 families."""
    return task_cids() + privacy_cids()


def extra_cids():
    """The 11 composition-only qpc public maps per seed (never candidates; U composes with them)."""
    return [direct_id(a, b) for a in RATES_INCOME for b in RATES_OCC if (a, b) != RATE] + \
        [b_config(f, COMPOSED_EXTRA_LAM) for f in PRIV_FAMS]


def composition_cids():
    return code_cids() + extra_cids()


def all_cids():
    """Every inner-selection candidate (32 per seed): 27 codes, 2 continuous sources, 3 references."""
    return code_cids() + ["SRC|U", "SRC|RAW-J_b0.3", "REF|E", "REF|F", "REF|F0"]


def reused_cids():
    return task_cids() + privacy_cids(REUSED_LAMS)


def new_cids():
    return privacy_cids(NEW_LAMS)


def cid_family(cid):
    if cid.startswith("SRC|"):
        return "SRC"
    if cid.startswith("REF|"):
        return "REF"
    return cid.split("|")[1]


def cid_lam(cid):
    return float(cid.split("|l")[1]) if "|l" in cid else None


def release_unit(k, cid):
    if cid.startswith("SRC|"):
        return f"tea__s{k}__{cid.split('|')[1]}"
    if cid.startswith("REF|"):
        return f"ref__s{k}__{cid.split('|')[1]}"
    return f"pol__s{k}__{cid.replace('|', '_')}"


def admitted_units(k):
    """The 28 units per seed admitted by verified copy from the qpc store (SOURCE_ADMISSION.json)."""
    u = [f"tea__s{k}__{t}" for t in TEACHERS] + [f"ref__s{k}__{r}" for r in REFS] + [f"fine__s{k}"]
    u += [release_unit(k, c) for c in reused_cids() + extra_cids()]
    return u


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
        return x if math.isfinite(x) else None          # new JSON is finite-or-null (prompt section 6)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, Path):
        return scrub_path(o)
    if isinstance(o, datetime):
        return iso(o)
    return o


def scrub_path(p) -> str:
    s = str(p)
    for root, ph in ((str(PRIV), "<PRIVATE_CACHE>/cbp_v1"), (str(QPC_COPY), "<PRIVATE_CACHE>/" + QPC_COPY.name),
                     (str(QPC_PRIV), "<PRIVATE_CACHE>/qpc_v1"), (str(DPC_PRIV), "<PRIVATE_CACHE>/dpc_v1"),
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


def schema_sha(names) -> str:
    """Registered schema binding convention: sha256 of the newline-joined pinned column names."""
    return hashlib.sha256("\n".join(str(x) for x in names).encode()).hexdigest()


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
        if "python" in cmd and ("-m cbp" in cmd or "cbp/sema.py" in cmd or "replay_cbp" in cmd):
            out.append({"pid": int(pid), "kind": "sema" if "sema" in cmd else ("verifier" if "replay_cbp" in cmd
                                                                               else "cbp")})
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

    def __init__(self, src=None):
        self.src = Path(src or SRC)
        self.src_sha = sha_file(self.src)
        if self.src_sha != SRC_SHA:
            raise RuntimeError("REFUSED: the input file fails its pinned hash")
        z = np.load(self.src, allow_pickle=False)
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
            z = np.load(self.src, allow_pickle=False)
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
        rm = {r: (r in roles and all(k in roles[r] for k in ("rows", "row_id_sha256")) and
                  all(mine[r][k] == roles[r][k] for k in keys4 if k in roles[r])) for r in ROLES}
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
    if ok and manifests.get("cbp_ROLE_MANIFEST") is None:
        st = "WARN"
        out["note"] = "cbp ROLE_MANIFEST.json not yet present; checked against the pinned qpc / dpc manifests"
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


# ------------------------------------------------------------------------------------------------ own utility
def fitting_constant(y_fit, K):
    """Fitting-prior constant: majority class of the OSF_DEFENSE_FIT labels (first index on ties)."""
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


# ------------------------------------------------------------------------------------------------ own cbp eligibility
# Implemented from HEADROOM_SELECTION_RULES.json, LABEL_TRUTH_TABLE.json and PROTOCOL.md sections 8-10 (FIT_LOCK
# bc0022f / AUDIT_AND_SELECTION_LOCK 25d20f2); no runner code is imported.
FAMILY_SIMPLICITY = {"LOCAL": 0, "SEQ-12": 1, "SEQ-21": 1, "JOINT": 2}


def seed_mean(xs):
    """Registered seed mean: (x_seed0 + x_seed1 + x_seed2) / 3, summed in seed order."""
    return (float(xs[0]) + float(xs[1]) + float(xs[2])) / 3


def task_check(u, uU, headroom=None):
    """One task on one seed. Ordinary checks in the registered margin form (U value + allowance) - value >= 0
    (inclusive); headroom from the EXCESSES: (value - U) <= 0.006 / 0.0035 (inclusive). `headroom` overrides the
    headroom limits (defect tests only). Verdicts that would differ under the other algebraic form are listed as
    borderline (diagnostic). Returns the checks, the excesses and the registered shortfall components."""
    hr = HEADROOM if headroom is None else headroom
    a, aU, c = float(u["acc"]), float(uU["acc"]), float(u["const_acc"])
    if abs(c - float(uU["const_acc"])) > 1e-15:
        raise ValueError("candidate and U evaluated on different rows (constant accuracies differ)")
    gN, gU = a - c, aU - c
    dLL, dB = float(u["logloss"]) - float(uU["logloss"]), float(u["brier"]) - float(uU["brier"])
    margins = {"acc": a - (aU - GATE["acc_drop"]), "logloss": (float(uU["logloss"]) + GATE["ll_excess"]) - float(u["logloss"]),
               "brier": (float(uU["brier"]) + GATE["brier_excess"]) - float(u["brier"]),
               "retain": gN - GATE["retain"] * gU, "gain": gN - GATE["gain"]}
    ordinary = {k_: v >= 0 for k_, v in margins.items()}
    head = {"logloss": dLL <= hr["ll_excess"], "brier": dB <= hr["brier_excess"]}
    alt = {"logloss": dLL <= GATE["ll_excess"], "brier": dB <= GATE["brier_excess"],
           "h_logloss": (float(uU["logloss"]) + hr["ll_excess"]) - float(u["logloss"]) >= 0,
           "h_brier": (float(uU["brier"]) + hr["brier_excess"]) - float(u["brier"]) >= 0}
    borderline = [k_ for k_, ok in (("logloss", ordinary["logloss"]), ("brier", ordinary["brier"]),
                                    ("h_logloss", head["logloss"]), ("h_brier", head["brier"])) if alt[k_] != ok]
    sf_ord = max((aU - GATE["acc_drop"] - a) / GATE["acc_drop"], (dLL - GATE["ll_excess"]) / GATE["ll_excess"],
                 (dB - GATE["brier_excess"]) / GATE["brier_excess"],
                 (GATE["retain"] * gU - gN) / max(GATE["retain"] * gU, GATE["gain"]), (GATE["gain"] - gN) / GATE["gain"])
    sf_head = max((dLL - hr["ll_excess"]) / hr["ll_excess"], (dB - hr["brier_excess"]) / hr["brier_excess"])
    return {"ordinary": ordinary, "headroom": head, "ordinary_ok": all(ordinary.values()), "headroom_ok": all(head.values()),
            "margins": margins, "excess": {"logloss": dLL, "brier": dB},
            "gain": gN, "u_gain": gU, "shortfall_ordinary_raw": sf_ord, "shortfall_headroom_raw": sf_head,
            "borderline_forms": borderline}


def _finite_num(v):
    return isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool) and math.isfinite(float(v))


def config_row(cid, per_seed, headroom=None, seed_average=False):
    """per_seed[k] = {"auc": {v1, v2, pair}, "util": {1: u, 2: u}, "U": {1: u, 2: u}, "preserved": {1: b, 2: b},
    "states": float (inf for continuous), "emitted": float or None, "pair_fp": str or None}. Eligibility per task AND per
    seed (never averaged; `seed_average=True` is the deliberate defect). Technical failures carry the registered reason
    codes FIT_OR_ADMISSION_FAILURE / DECISION_PRESERVATION_FAILURE / NON_ESTIMABLE_INNER_METRIC (never a shortfall)."""
    seeds, reasons, detail = {}, [], []
    for k in SEEDS:
        s = per_seed.get(k)
        if s is None:
            reasons.append("FIT_OR_ADMISSION_FAILURE")
            detail.append(f"s{k}: unit or inner audit missing")
            continue
        try:
            vals = [s["auc"][w] for w in ("v1", "v2", "pair")] + [s["util"][i][q] for i in (1, 2)
                                                                  for q in ("acc", "logloss", "brier", "const_acc")]
        except (KeyError, TypeError):
            vals = [None]
        if not all(_finite_num(v) for v in vals):
            reasons.append("NON_ESTIMABLE_INNER_METRIC")
            detail.append(f"s{k}: missing or nonfinite inner field")
            continue
        if not all(bool(s["preserved"][i]) for i in (1, 2)):
            reasons.append("DECISION_PRESERVATION_FAILURE")
            detail.append(f"s{k}: exact decision preservation failed")
        tc = {i: task_check(s["util"][i], s["U"][i], headroom) for i in (1, 2)}
        seeds[k] = {"auc": {w: float(s["auc"][w]) for w in ("v1", "v2", "pair")}, "tasks": tc,
                    "ordinary_ok": all(tc[i]["ordinary_ok"] for i in (1, 2)),
                    "headroom_ok": all(tc[i]["headroom_ok"] for i in (1, 2)),
                    "sum_ll": float(s["util"][1]["logloss"]) + float(s["util"][2]["logloss"]),
                    "states": float(s["states"]), "emitted": s.get("emitted"), "pair_fp": s.get("pair_fp"),
                    "preserved": {i: bool(s["preserved"][i]) for i in (1, 2)}}
    reason = "+".join(dict.fromkeys(reasons)) or None
    row = {"config": cid, "family": cid_family(cid), "lam": cid_lam(cid), "seeds": seeds, "invalid": detail,
           "invalid_reason": reason, "valid": reason is None and len(seeds) == len(SEEDS)}
    if not row["valid"]:
        row.update({"ordinary_eligible": False, "headroom_eligible": False})
        return row
    sm = lambda f: seed_mean([f(seeds[k]) for k in SEEDS])  # noqa: E731
    row.update({"mean_pair": sm(lambda s: s["auc"]["pair"]), "mean_v1": sm(lambda s: s["auc"]["v1"]),
                "mean_v2": sm(lambda s: s["auc"]["v2"]), "mean_sum_logloss": sm(lambda s: s["sum_ll"]),
                "mean_states": sm(lambda s: s["states"])})
    if seed_average:                    # DEFECT: seed-averaged eligibility (a failing seed averaged away)
        avg = {i: {q: seed_mean([per_seed[k]["util"][i][q] for k in SEEDS]) for q in ("acc", "logloss", "brier")}
               for i in (1, 2)}
        avgU = {i: {q: seed_mean([per_seed[k]["U"][i][q] for k in SEEDS]) for q in ("acc", "logloss", "brier",
                                                                                    "const_acc")} for i in (1, 2)}
        for i in (1, 2):
            avg[i]["const_acc"] = avgU[i]["const_acc"]
        tc = {i: task_check(avg[i], avgU[i], headroom) for i in (1, 2)}
        row["ordinary_eligible"] = all(tc[i]["ordinary_ok"] for i in (1, 2))
        row["headroom_eligible"] = row["ordinary_eligible"] and all(tc[i]["headroom_ok"] for i in (1, 2))
    else:
        row["ordinary_eligible"] = all(seeds[k]["ordinary_ok"] for k in SEEDS)
        row["headroom_eligible"] = row["ordinary_eligible"] and all(seeds[k]["headroom_ok"] for k in SEEDS)
    row["shortfall_ordinary"] = max(0.0, max(seeds[k]["tasks"][i]["shortfall_ordinary_raw"] for k in SEEDS for i in (1, 2)))
    row["shortfall_headroom"] = max(0.0, max(seeds[k]["tasks"][i]["shortfall_headroom_raw"] for k in SEEDS for i in (1, 2)))
    row["borderline_forms"] = sorted({f"s{k}/t{i}/{b}" for k in SEEDS for i in (1, 2)
                                      for b in seeds[k]["tasks"][i]["borderline_forms"]})
    row["failing"] = {f"s{k}": {TASKS[i - 1]: [g_ for g_, v in seeds[k]["tasks"][i]["ordinary"].items() if not v] +
                                [f"headroom_{g_}" for g_, v in seeds[k]["tasks"][i]["headroom"].items() if not v]
                                for i in (1, 2)} for k in SEEDS}
    return row


def okey(r, decimals=ORDER_DECIMALS):
    """Ordering for every control and nominee role: round(seed mean pair AUC, 12), round(seed mean summed true-label log
    loss, 12), mean actual total token states (alpha1 + alpha2; continuous = +inf; exact), configuration ID."""
    return (round(r["mean_pair"], decimals), round(r["mean_sum_logloss"], decimals), r["mean_states"], r["config"])


def guard_check(r, guards):
    """Registered inclusive guard a <= g + 0.005 for each guard, seed and recipient; the normalised shortfall
    max(0, (a - g - 0.005) / 0.005) is exactly 0.0 when the guard holds and is used for ranking only."""
    ok, xs = True, []
    for g_ in guards:
        for k in SEEDS:
            for w in ("v1", "v2"):
                a, gv = r["seeds"][k]["auc"][w], g_["seeds"][k]["auc"][w]
                ok &= a <= gv + BUFFER
                xs.append((a - gv - BUFFER) / BUFFER)
    return ok, (0.0 if ok else max(0.0, max(xs)))


def pick_role(rows, cands, nominee, need_headroom, guard_names=(), statuses=None, reverse=False):
    """One role (HEADROOM_SELECTION_RULES.json roles / statuses / fallback_ordering / invalid_candidates /
    missing_guard). `reverse=True` is the deliberate defect 'reversed best / worst'."""
    none = "NO_ELIGIBLE_NOMINEE" if nominee else "NO_ELIGIBLE_COMPARATOR"
    bad = "INVALID_NOMINEE" if nominee else "INVALID_COMPARATOR"
    statuses = statuses or {}
    if not cands:
        return {"status": bad, "config": None, "reason": "FIT_OR_ADMISSION_FAILURE", "detail": "empty candidate set"}
    inval = {c_: (rows.get(c_) or {}).get("invalid_reason") or "FIT_OR_ADMISSION_FAILURE" for c_ in cands
             if not (rows.get(c_) or {}).get("valid")}
    if inval:
        return {"status": bad, "config": None, "reason": "+".join(dict.fromkeys(
            x for v in inval.values() for x in v.split("+"))), "invalid_candidates": inval}
    missing_guard = [g_ for g_ in guard_names if (statuses.get(g_) or {}).get("status") != "NOMINEE"]
    guards = [rows[statuses[g_]["config"]] for g_ in guard_names if g_ not in missing_guard]
    ev = []
    for c_ in cands:
        r = rows[c_]
        el = r["headroom_eligible"] if need_headroom else r["ordinary_eligible"]
        gok, gs = guard_check(r, guards)
        e = {"config": c_, "ordinary_eligible": r["ordinary_eligible"], "headroom_eligible": r["headroom_eligible"],
             "eligible_for_role": el, "shortfall_ordinary": r["shortfall_ordinary"],
             "shortfall_headroom": r["shortfall_headroom"] if need_headroom else 0.0}
        if missing_guard:
            e.update({"guard_ok": None, "shortfall_guard": None, "nominable": False,
                      "fallback_key": (round(r["shortfall_ordinary"], ORDER_DECIMALS),
                                       round(e["shortfall_headroom"], ORDER_DECIMALS)) + okey(r)})
        else:
            e.update({"guard_ok": gok, "shortfall_guard": gs, "nominable": el and gok,
                      "fallback_key": (round(r["shortfall_ordinary"], ORDER_DECIMALS),
                                       round(e["shortfall_headroom"], ORDER_DECIMALS), round(gs, ORDER_DECIMALS)) + okey(r)})
        ev.append(e)
    if missing_guard and any(e["eligible_for_role"] for e in ev):
        return {"status": bad, "config": None, "reason": "MISSING_GUARD_COMPARATOR", "missing_guards": missing_guard,
                "blocked_eligible": [e["config"] for e in ev if e["eligible_for_role"]], "evaluated": ev}
    nom = [e for e in ev if e["nominable"]]
    if nom:
        best = (max if reverse else min)(nom, key=lambda e: okey(rows[e["config"]]))
        return {"status": "NOMINEE", "config": best["config"], "evaluated": ev}
    if not any(e["ordinary_eligible"] for e in ev):
        why = "ORDINARY_UTILITY_FAILURE"
    elif need_headroom and not any(e["headroom_eligible"] for e in ev):
        why = "HEADROOM_SELECTION_FAILURE"
    else:
        why = "LOCAL_GUARD_FAILURE"
    fb = min(ev, key=lambda e: e["fallback_key"])
    out = {"status": none, "config": None, "descriptive_config": fb["config"], "reason": why, "evaluated": ev,
           "fallback_shortfalls": {x: fb[x] for x in ("shortfall_ordinary", "shortfall_headroom", "shortfall_guard")}}
    if missing_guard:
        out.update({"missing_guards": missing_guard, "fallback_rank_status": "INVALID_MISSING_GUARD_COMPARATOR"})
    return out


def simplest_family(fams):
    """LOCAL < SEQ-12 = SEQ-21 < JOINT; equally simple families joined with '=' (LABEL_TRUTH_TABLE winning_family_naming)."""
    fs = sorted(set(fams), key=lambda f: (FAMILY_SIMPLICITY[f], f))
    lo = FAMILY_SIMPLICITY[fs[0]]
    return "=".join(f for f in fs if FAMILY_SIMPLICITY[f] == lo)


def alias_record(rows, cid):
    """Configurations whose deployed pair maps (own pair fingerprint) equal `cid`'s on ALL seeds, partial (per-seed)
    aliases, the simplest family of the full alias set and identity with the privacy-untrained DIRECT / FINE maps."""
    if not cid or cid not in rows or not rows[cid].get("valid"):
        return None
    fp = {k: rows[cid]["seeds"][k].get("pair_fp") for k in SEEDS}
    if any(v is None for v in fp.values()):
        return {"available": False}
    full, partial = [], {}
    for c_, r in rows.items():
        if c_ == cid or not r.get("valid") or cid_family(c_) in ("SRC", "REF"):
            continue
        same = [k for k in SEEDS if r["seeds"][k].get("pair_fp") == fp[k]]
        if len(same) == len(SEEDS):
            full.append(c_)
        elif same:
            partial[c_] = same
    priv = [c_ for c_ in [cid] + full if cid_family(c_) in PRIV_FAMS]
    return {"available": True, "alias_set": sorted(full), "partial_aliases": partial,
            "simplest_family": simplest_family([cid_family(c_) for c_ in priv]) if priv else None,
            "identical_to_untrained": sorted(c_ for c_ in full if cid_family(c_) in ("DIRECT-TASK", "FINE-TASK"))}


def my_selection_cbp(rows, reverse=False, headroom_as_ordinary=False):
    """Own implementation of HEADROOM_SELECTION_RULES.json on own inner rows: statuses, resolved IDs, aliases,
    diagnostics. headroom_as_ordinary=True is the deliberate defect 'changing headroom to the ordinary limit'."""
    fam = {c_: cid_family(c_) for c_ in all_cids()}
    priv = privacy_cids()                                                   # 24: LOCAL, SEQ-12, SEQ-21, JOINT x 6
    joint = [c_ for c_ in priv if fam[c_] == "JOINT"]
    rate_ctl = [direct_id(*RATE), b_config("FINE-TASK")] + [c_ for c_ in priv if fam[c_] != "JOINT"]     # 20
    glob_ctl = rate_ctl + [b_config("CLASS"), "SRC|U", "SRC|RAW-J_b0.3", "REF|E", "REF|F", "REF|F0"]   # 26
    need_h = not headroom_as_ordinary
    st = {}
    st["T*"] = pick_role(rows, list(T_STAR_LIST), False, False, reverse=reverse)
    st["C_rate"] = pick_role(rows, rate_ctl, False, False, reverse=reverse)
    st["C_global"] = pick_role(rows, glob_ctl, False, False, reverse=reverse)
    st["P*"] = pick_role(rows, priv, True, need_h, ("T*",), st, reverse=reverse)
    st["J*"] = pick_role(rows, joint, True, need_h, ("C_rate", "C_global"), st, reverse=reverse)
    q = direct_id(*RATE)
    qr = rows.get(q)
    if qr is None or not qr.get("valid"):
        st["Q"] = {"status": "INVALID_NOMINEE", "config": None, "reason": (qr or {}).get("invalid_reason") or
                   "FIT_OR_ADMISSION_FAILURE"}
    elif qr["ordinary_eligible"]:
        st["Q"] = {"status": "NOMINEE", "config": q}
    else:
        st["Q"] = {"status": "NO_ELIGIBLE_NOMINEE", "config": None, "descriptive_config": q,
                   "reason": "ORDINARY_UTILITY_FAILURE"}
    for x in ("P*", "J*"):
        c_ = st[x].get("config") or st[x].get("descriptive_config")
        st[x]["aliases"] = alias_record(rows, c_)
        if st[x]["status"] == "NOMINEE":
            al = st[x]["aliases"] or {}
            st[x]["family_of_config"] = fam[st[x]["config"]]
            st[x]["winning_family"] = al.get("simplest_family") or fam[st[x]["config"]]
    resolved = {x: (v.get("config") or v.get("descriptive_config")) for x, v in st.items()}
    nh = pick_role(rows, priv, True, False, ("T*",), st)
    diag = {"ordinary_privacy_winner_no_headroom": nh,
            "strongest_ordinary_privacy_unguarded": pick_role(rows, priv, True, False),
            "family_headroom_winners": {f_: pick_role(rows, [c_ for c_ in priv if fam[c_] == f_], True, need_h, ("T*",), st)
                                        for f_ in PRIV_FAMS},
            "source_lambda_0.1_controls": {c_: {x: rows[c_].get(x) for x in ("ordinary_eligible", "headroom_eligible",
                                                                             "mean_pair", "mean_v1", "mean_v2",
                                                                             "shortfall_ordinary", "shortfall_headroom")}
                                           for c_ in (b_config("JOINT", 0.1), b_config("SEQ-12", 0.1),
                                                      b_config("SEQ-21", 0.1)) if c_ in rows}}
    for f_, v in diag["family_headroom_winners"].items():
        v["aliases"] = alias_record(rows, v.get("config") or v.get("descriptive_config"))
    if st["P*"]["status"] == "NOMINEE" and nh["status"] == "NOMINEE":
        a_, b_ = rows[st["P*"]["config"]], rows[nh["config"]]
        diag["headroom_changes_winner"] = {"changed": st["P*"]["config"] != nh["config"],
                                           "give_up_mean_pair_auc": a_["mean_pair"] - b_["mean_pair"],
                                           "give_up_per_seed": {f"s{k}": a_["seeds"][k]["auc"]["pair"] -
                                                                b_["seeds"][k]["auc"]["pair"] for k in SEEDS}}
    else:
        diag["headroom_changes_winner"] = {"status": f"P* {st['P*']['status']}; ordinary winner {nh['status']}"}
    for x in ("T*", "C_rate", "C_global", "P*", "J*", "Q"):
        c_ = resolved[x]
        st[x]["role_row"] = None if c_ is None or not rows.get(c_, {}).get("valid") else \
            {k_: rows[c_].get(k_) for k_ in ("mean_pair", "mean_v1", "mean_v2", "mean_sum_logloss", "mean_states")}
    role_alias = sorted({(a_, b_) for a_ in resolved for b_ in resolved if a_ < b_ and resolved[a_] and
                         resolved[a_] == resolved[b_]})
    return {"statuses": st, "resolved": resolved, "diagnostics": diag, "aliases": role_alias}


# ------------------------------------------------------------------------------------------------ own cbp labels
CLAIMS = {"A": ("J*", "C_rate"), "B": ("J*", "C_global"), "C": ("P*", "T*")}
CLAIM_SLOTS = {"A": [f"P{i:02d}" for i in range(1, 12)], "B": [f"P{i:02d}" for i in range(12, 23)],
               "C": [f"P{i:02d}" for i in range(23, 34)], "Q": [f"P{i:02d}" for i in range(34, 38)]}
TASKS = ("income", "occupation")


def role_state(st):
    s_ = (st or {}).get("status")
    if s_ == "NOMINEE" and (st or {}).get("config"):
        return "ELIGIBLE"
    if s_ in ("NO_ELIGIBLE_NOMINEE", "NO_ELIGIBLE_COMPARATOR"):
        return "NO_ELIGIBLE"
    return "INVALID"


def clause_outcome(e):
    """LABEL_TRUTH_TABLE clause outcomes: PASS / NOT_ESTABLISHED_PRECISION / NOT_ESTABLISHED_POINT / MEASURED_VIOLATION
    / INVALID (any nonfinite point or bound). Returns (outcome, point side)."""
    pt, lo, hi, tgt = e.get("point"), e.get("lower"), e.get("upper"), e["target"]
    if not all(_finite_num(x) for x in (pt, lo, hi)) or e.get("nonfinite"):
        return "INVALID", None
    if e["side"] == "lower>":
        if lo > tgt:
            return "PASS", "pass"
        if hi < tgt:
            return "MEASURED_VIOLATION", "fail"
        return ("NOT_ESTABLISHED_PRECISION", "pass") if pt > tgt else ("NOT_ESTABLISHED_POINT", "fail")
    if hi < tgt:
        return "PASS", "pass"
    if lo > tgt:
        return "MEASURED_VIOLATION", "fail"
    return ("NOT_ESTABLISHED_PRECISION", "pass") if pt < tgt else ("NOT_ESTABLISHED_POINT", "fail")


def _fail_root(outcomes):
    failing = [o for o in outcomes.values() if o != "PASS"]
    if any(o == "MEASURED_VIOLATION" for o in failing):
        return "MEASURED_VIOLATION_SUPPORTED_BY_BOUND"
    if failing and all(o == "NOT_ESTABLISHED_PRECISION" for o in failing):
        return "ASSESSMENT_PRECISION_FAILURE"
    return "CLAUSE_NOT_ESTABLISHED"


def claim_status_cbp(nom_st, cmp_st, outcomes, claim, control_ok=True):
    """(status, root cause) in the registered precedence. nom_st / cmp_st: role status records; outcomes:
    {slot id: clause outcome} for the claim's slots."""
    nom, cmp_ = role_state(nom_st), role_state(cmp_st)
    if not control_ok:
        return "INCOMPLETE_OR_INVALID", "FAILED_REQUIRED_CONTROL"
    if cmp_ == "INVALID":
        return "INCOMPLETE_OR_INVALID", "INVALID_OR_MISSING_COMPARATOR"
    if nom == "INVALID":
        return "INCOMPLETE_OR_INVALID", (nom_st or {}).get("reason") or "NOMINEE_TECHNICAL_FAILURE"
    if nom == "NO_ELIGIBLE":
        return "NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE", (nom_st or {}).get("reason")
    if cmp_ == "NO_ELIGIBLE":
        return "INCOMPLETE_OR_INVALID", "NO_ELIGIBLE_COMPARATOR"
    if not outcomes:
        return "INCOMPLETE_OR_INVALID", "NO_CLAUSES"
    if sorted(outcomes) != CLAIM_SLOTS[claim]:
        return "INCOMPLETE_OR_INVALID", "MISSING_SLOTS"
    if any(o == "INVALID" for o in outcomes.values()):
        return "INCOMPLETE_OR_INVALID", "NONFINITE_PRIMARY_QUANTITY"
    if all(o == "PASS" for o in outcomes.values()):
        return "PASS", "COMPLETE_PASSING_CONJUNCTION"
    return "NOT_ESTABLISHED", _fail_root(outcomes)


def q_status_cbp(q_st, outcomes):
    if role_state(q_st) == "INVALID":
        return "INCOMPLETE_OR_INVALID", "Q_NOT_RESOLVED"
    if role_state(q_st) == "NO_ELIGIBLE":
        return "NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE", "ORDINARY_UTILITY_FAILURE"
    if not outcomes:
        return "INCOMPLETE_OR_INVALID", "NO_CLAUSES"
    if sorted(outcomes) != CLAIM_SLOTS["Q"]:
        return "INCOMPLETE_OR_INVALID", "MISSING_SLOTS"
    if any(o == "INVALID" for o in outcomes.values()):
        return "INCOMPLETE_OR_INVALID", "NONFINITE_PRIMARY_QUANTITY"
    if all(o == "PASS" for o in outcomes.values()):
        return "PASS", "COMPLETE_PASSING_CONJUNCTION"
    return "NOT_ESTABLISHED", _fail_root(outcomes)


def labels_cbp(statuses, eps, technical_valid=True, control_ok=None):
    """Own cbp label truth table. eps: endpoint dicts (id, claim, point / lower / upper / target / side);
    control_ok: {claim: bool} for per-claim required-control failures (failure_scope)."""
    control_ok = control_ok or {}
    oc = {e["id"]: clause_outcome(e)[0] for e in eps}
    claims, roots = {}, {}
    for c_, (n_, m_) in CLAIMS.items():
        cl = {e["id"]: oc[e["id"]] for e in eps if e["claim"] == c_}
        claims[c_], roots[c_] = claim_status_cbp(statuses.get(n_), statuses.get(m_), cl, c_, control_ok.get(c_, True))
    q, q_root = q_status_cbp(statuses.get("Q"), {e["id"]: oc[e["id"]] for e in eps if e["claim"] == "Q"})
    if not control_ok.get("Q", True):
        q, q_root = "INCOMPLETE_OR_INVALID", "FAILED_REQUIRED_CONTROL"
    p_st = statuses.get("P*") or {}
    fam = p_st.get("winning_family") or (cid_family(p_st["config"]) if p_st.get("config") else None)
    shown = {**claims, "Q": q}
    if not technical_valid:
        label = "INCOMPLETE_OR_INVALID"
    else:
        labels = []
        if claims["C"] == "PASS":
            labels.append("PRIVACY_COMPRESSION_DEVELOPMENT_CRITERION_MET" + (f" ({fam})" if fam else ""))
        if claims["A"] == "PASS" and claims["B"] == "PASS":
            labels.append("JOINT_DEVELOPMENT_CRITERION_MET")
        if labels:
            label = " + ".join(labels)
        elif any(v == "INCOMPLETE_OR_INVALID" for v in shown.values()):
            label = "INCOMPLETE_OR_INVALID"
        elif q == "PASS":
            label = "CONFIDENCE_FEASIBILITY_ESTABLISHED_NO_METHOD_CRITERION"
        else:
            label = "EXPERIMENTAL_NO_ADVANTAGE"
    return {"label": label, "claims": claims, "root_causes": roots, "q": q, "q_root_cause": q_root,
            "incomplete_displayed": sorted(k_ for k_, v in shown.items() if v == "INCOMPLETE_OR_INVALID"),
            "clause_outcomes": oc}


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
    draw sequence serves every arm and seed (the convention reconciled with the qpc infer rule by the predecessor verifier;
    re-checked against the cbp infer rule text in PHASE 3)."""

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


class OwnBoot:
    """Paired exact-record-group bootstrap (registered convention): groups = sorted unique assessment group ids;
    rng = default_rng(BOOT_SEED = 20261008); replicate b = rng.multinomial(G, uniform), drawn sequentially; row weight = count of
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


LOCK_ORDER = ["SOURCE_ADMISSION_LOCK", "FIT_LOCK", "AUDIT_AND_SELECTION_LOCK", "EVALUATION_LOCK"]
# governing locks of every lead stage (a tuple: ALL must be pushed before the stage starts; prompt section 12 freezes
# every scientific definition, including the audit and selection lock, before ANY new fit)
STAGE_LOCK = {"admit": ("SOURCE_ADMISSION_LOCK",), "fit": ("FIT_LOCK", "AUDIT_AND_SELECTION_LOCK"),
              "inner": ("AUDIT_AND_SELECTION_LOCK",), "inner_src": ("AUDIT_AND_SELECTION_LOCK",),
              "controls": ("AUDIT_AND_SELECTION_LOCK",), "select": ("AUDIT_AND_SELECTION_LOCK",),
              "assess": ("EVALUATION_LOCK",), "outer": ("EVALUATION_LOCK",), "infer": ("EVALUATION_LOCK",)}


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
STATEMENT = ("The design is motivated by opened Adult development results, including the completed qpc confidence-capacity "
             "study. The fitting, inner-selection and assessment data have all been used historically. This is an "
             "exploratory, locked development comparison. Its nominal intervals condition on fitted artifacts and do not "
             "account for the adaptive research history. It is not fresh confirmation or a population privacy guarantee.")


def pending(name, why):
    return res("PENDING", reason=why, check=name)


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


def dir_hashes(d: Path):
    return {str(q.relative_to(d)): sha_file(q) for q in sorted(d.rglob("*")) if q.is_file()}


def copy_sums(folder: Path):
    out = {}
    p = folder / "SHA256SUMS"
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                h, rel = line.split("  ", 1)
                out[rel.strip()] = h
    return out


def check_pins():
    """Source pins: input file, branch base, SOURCE_INDEX (every pinned module equal to the qpc tip blob and to the
    worktree), source results hashes, SOURCE_ADMISSION_LOCK (committed, byte-identical on origin, code hashes at its
    commit, exposure statement, budget, inputs) and SOURCE_ADMISSION.json (pins the same hashes)."""
    out, fails = {}, []
    out["input_sha256_ok"] = sha_file(SRC) == SRC_SHA
    base = git("merge-base", "HEAD", QPC_TIP)
    out["branch_base_is_qpc_tip"] = base == QPC_TIP
    out["qpc_tip_contains_evidence"] = git_ok("merge-base", "--is-ancestor", QPC_EVIDENCE, QPC_TIP)
    si = jload(RES / "SOURCE_INDEX.json") if (RES / "SOURCE_INDEX.json").exists() else None
    if si is None:
        fails.append("SOURCE_INDEX.json absent")
    else:
        bad = []
        for f_, v in si.get("files", {}).items():
            blob = git_show_bytes(QPC_TIP, f_)
            own = sha_file(WT / f_) if (WT / f_).exists() else None
            if blob is None or hashlib.sha256(blob).hexdigest() != v.get("sha256") or own != v.get("sha256"):
                bad.append(f_)
        out["source_index_files"] = len(si.get("files", {}))
        out["source_index_mismatches"] = bad
        srh = si.get("source_results_hashes") or {}
        rbad = []
        for f_, h in (srh.items() if isinstance(srh, dict) else []):
            hh = h.get("sha256") if isinstance(h, dict) else h
            rel = f_ if f_.startswith("results/") else f"{QPC_REL}/{f_}"
            blob = git_show_bytes(QPC_EVIDENCE, rel)
            if blob is None or hashlib.sha256(blob).hexdigest() != hh:
                rbad.append(f_)
        out["source_results_hashes"] = len(srh) if isinstance(srh, dict) else srh
        out["source_results_mismatches_at_evidence"] = rbad
        if bad or rbad or not si.get("all_equal_to_source_tip"):
            fails.append("SOURCE_INDEX")
        # every qpc/dpc/osf/smf module in the tree is still the source-tip blob (never edited in this branch)
        edited = []
        for top in ("qpc", "dpc", "osf", "smf", "rgj", "jcv", "stored_model_eval", "oar"):
            for p in sorted((WT / top).glob("*.py")) if (WT / top).exists() else []:
                rel = str(p.relative_to(WT))
                blob = git_show_bytes(QPC_TIP, rel)
                if blob is None or blob != p.read_bytes():
                    edited.append(rel)
        out["pinned_packages_edited_in_this_branch"] = edited
        if edited:
            fails.append("pinned package edited")
    sl = RES / "SOURCE_ADMISSION_LOCK.json"
    if not sl.exists():
        fails.append("SOURCE_ADMISSION_LOCK absent")
    else:
        L_ = jload(sl)
        rel = f"{REL_RES}/SOURCE_ADMISSION_LOCK.json"
        shown = git_show_bytes(f"origin/{BRANCH}", rel)
        commits = [l_.split("|") for l_ in (git("log", "--format=%H|%cI", "--", rel) or "").splitlines() if l_]
        c0 = commits[-1][0] if commits else None
        mism = [f_ for f_, h in (L_.get("code_files") or {}).items()
                if (lambda b: b is None or hashlib.sha256(b).hexdigest() != h)(git_show_bytes(c0, f_) if c0 else None)]
        out["source_admission_lock"] = {
            "committed": bool(c0), "versions": len(commits), "on_origin_byte_identical": shown is not None and
            shown == sl.read_bytes(), "first_commit": c0, "first_push_time": first_remote(c0, remote_reflog()) if c0 else None,
            "code_files": len(L_.get("code_files") or {}), "code_files_differing_at_lock_commit": mism,
            "statement_equals_prompt": L_.get("statement") == STATEMENT,
            "input_pin_ok": (L_.get("inputs") or {}).get("source_npz_sha256") == SRC_SHA,
            "evidence_pin_ok": (L_.get("inputs") or {}).get("source_evidence_commit") == QPC_EVIDENCE,
            "budget": L_.get("budget")}
        b = L_.get("budget") or {}
        out["source_admission_lock"]["budget_ok"] = (b.get("elapsed_h") == 10 and b.get("cpu_h") == 20 and
                                                     b.get("heavy_processes_total") == 2 and b.get("memory_gib") == 8 and
                                                     b.get("reserve_final_h") == 2 and b.get("reserve_final_cpu_h") == 4)
        sa = out["source_admission_lock"]
        if not (sa["committed"] and sa["on_origin_byte_identical"] and not mism and sa["statement_equals_prompt"] and
                sa["input_pin_ok"] and sa["evidence_pin_ok"] and sa["budget_ok"] and sa["first_push_time"]):
            fails.append("SOURCE_ADMISSION_LOCK")
    sa_doc = RES / "SOURCE_ADMISSION.json"
    if sa_doc.exists():
        A = jload(sa_doc)
        out["source_admission_doc"] = {"input_ok": (A.get("input") or {}).get("sha256") == SRC_SHA,
                                       "evidence_ok": (A.get("source") or {}).get("evidence_commit") == QPC_EVIDENCE,
                                       "units": len(A.get("units") or {}),
                                       "composition_only_ids_equal_own": sorted((A.get("composition_only_public_maps") or
                                                                                 {}).get("ids", [])) == sorted(extra_cids())}
        if not all(v for k_, v in out["source_admission_doc"].items() if isinstance(v, bool)) or \
                out["source_admission_doc"]["units"] != 84:
            fails.append("SOURCE_ADMISSION.json")
    else:
        fails.append("SOURCE_ADMISSION.json absent")
    ok = not fails and out["input_sha256_ok"] and out["branch_base_is_qpc_tip"] and out["qpc_tip_contains_evidence"]
    return res("PASS" if ok else "FAIL", failures=fails, **out)


def check_reuse_custody():
    """Every admitted unit (84), teacher artifact directory (6) and deployment input file (2): the cbp bytes equal the
    live qpc store, the qpc same-device copy v2 SHA256SUMS and (scored qpc units) the qpc EVALUATION_LOCK pin at the
    evidence commit; each unit's COMPLETE.json is internally consistent; the public SOURCE_ADMISSION.json lists the same
    hashes; the private admission receipt says ADMITTED. Counts reuse separately from composition-only maps."""
    sums = copy_sums(QPC_COPY)
    el = json.loads(git_show_bytes(QPC_EVIDENCE, f"{QPC_REL}/EVALUATION_LOCK.json") or b"{}")
    lockf = {}
    for s_ in (el.get("seeds") or {}).values():
        lockf.update(s_.get("unit_file_sha256") or {})
    pub = jload(RES / "SOURCE_ADMISSION.json") if (RES / "SOURCE_ADMISSION.json").exists() else {}
    pub_units = pub.get("units") or {}
    out, bad = {"units": 0, "files": 0, "kinds": {}}, []
    expected = [u for k in SEEDS for u in admitted_units(k)]
    for u in expected:
        cd, qd = UNITS / u, QPC_UNITS / u
        if not cd.exists():
            bad.append(f"{u}: absent from the cbp store")
            continue
        ch, qh = dir_hashes(cd), dir_hashes(qd) if qd.exists() else {}
        out["units"] += 1
        out["files"] += len(ch)
        kind = u.split("__")[0]
        out["kinds"][kind] = out["kinds"].get(kind, 0) + 1
        if ch != qh:
            bad.append(f"{u}: bytes differ from the live qpc store")
        for f_, h in ch.items():
            if sums.get(f"qpc_v1/run/units/{u}/{f_}") != h:
                bad.append(f"{u}/{f_}: differs from / absent in the qpc copy v2 SHA256SUMS")
        if u in lockf and {f_: h for f_, h in lockf[u].items()} != {f_: ch.get(f_) for f_ in lockf[u]}:
            bad.append(f"{u}: differs from the qpc EVALUATION_LOCK pin")
        cu, _ = complete_ok(cd)
        if not (cu["id_ok"] and cu["rehash_ok"] and not cu["unlisted"]):
            bad.append(f"{u}: COMPLETE.json inconsistent")
        pf = (pub_units.get(u) or {}).get("files_sha256")
        if pf is not None and pf != ch:
            bad.append(f"{u}: differs from SOURCE_ADMISSION.json")
    out["scored_qpc_units_checked_against_qpc_evaluation_lock"] = sum(1 for u in expected if u in lockf)
    adm = {}
    for k in SEEDS:
        for t in TEACHERS:
            a_ = f"rel__s{k}__{t}"
            ch, qh = dir_hashes(ADM / a_) if (ADM / a_).exists() else {}, dir_hashes(QPC_ADM / a_)
            adm[a_] = ch == qh and bool(ch) and all(sums.get(f"qpc_v1/admitted/{a_}/{f_}") == h for f_, h in ch.items())
            if not adm[a_]:
                bad.append(f"admitted {a_}: differs from the qpc admitted copy / SHA256SUMS")
    out["admitted_teacher_dirs_equal"] = adm
    inp = {}
    for f_ in ("deploy_input.npz", "schema.json"):
        p, q = PRIV / "inputs" / f_, QPC_PRIV / "inputs" / f_
        inp[f_] = p.exists() and q.exists() and sha_file(p) == sha_file(q) == sums.get(f"qpc_v1/inputs/{f_}")
        if not inp[f_]:
            bad.append(f"inputs/{f_}")
    out["inputs_equal"] = inp
    rc = jload(ADM / "ADMISSION_RECEIPT.json") if (ADM / "ADMISSION_RECEIPT.json").exists() else {}
    out["receipt_verdict"] = rc.get("verdict")
    out["receipt_teacher_parity_all_ok"] = bool(rc.get("teachers")) and all(v.get("ok") for v in rc["teachers"].values())
    out["receipt_code_parity_all_ok"] = len(rc.get("codes") or {}) == 66 and all(v.get("ok") for v in rc["codes"].values())
    if rc.get("verdict") != "ADMITTED" or not out["receipt_teacher_parity_all_ok"] or not out["receipt_code_parity_all_ok"]:
        bad.append("admission receipt")
    present = sorted(q.name for q in UNITS.iterdir() if q.is_dir()) if UNITS.exists() else []
    out["units_on_disk_by_kind"] = {}
    for n_ in present:
        kk = n_.split("__")[0]
        out["units_on_disk_by_kind"][kk] = out["units_on_disk_by_kind"].get(kk, 0) + 1
    out["pol_units_admitted_reused"] = sum(1 for k in SEEDS for c_ in reused_cids() if (UNITS / release_unit(k, c_)).exists())
    out["pol_units_admitted_composition_only"] = sum(1 for k in SEEDS for c_ in extra_cids()
                                                     if (UNITS / release_unit(k, c_)).exists())
    out["pol_units_new_present"] = sum(1 for k in SEEDS for c_ in new_cids() if unit_done(release_unit(k, c_)))
    unexpected = [n_ for n_ in present if n_.startswith(("tea__", "ref__", "fine__", "pol__")) and n_ not in expected and
                  n_ not in {release_unit(k, c_) for k in SEEDS for c_ in new_cids()}]
    out["unexpected_release_units"] = unexpected
    if unexpected:
        bad.append(f"unexpected units {unexpected[:5]}")
    ok = not bad and out["units"] == 84 and out["pol_units_admitted_reused"] == 33 and \
        out["pol_units_admitted_composition_only"] == 33
    return res("PASS" if ok else "FAIL", failures=bad[:30], n_failures=len(bad), **out,
               rule="own SHA-256 of every file in the cbp store vs the live qpc store, the qpc copy v2 SHA256SUMS and "
                    "(scored units) the qpc EVALUATION_LOCK at the evidence commit")


def pinned_osf_manifest():
    return json.loads(git_show_bytes(TEACHER_PIN, f"{OSF_REL}/MODEL_MANIFEST.json") or b"{}")


def check_teachers(D: Data, L, refit=True):
    """Own forward application of every admitted teacher (cbp admitted copy) on own X vs the cbp, qpc and dpc tea__
    units and the admitted release; custody of the cbp copy vs the qpc copy and the pinned osf MODEL_MANIFEST;
    deployed-head structure and scaler moments; optional own head refit (DEFENSE_FIT / HEAD_VALIDATION task labels)."""
    man = pinned_osf_manifest()
    entries = {(u["label"], u["seed"], u["unit"]): u for u in man.get("units", [])}
    out, T = {}, {}
    tr, va = D.idx[FIT], D.idx["HEAD_VALIDATION"]
    for k in SEEDS:
        for t in TEACHERS:
            name = f"rel__s{k}__{t}"
            cd_, qd = ADM / name, QPC_ADM / name
            info, fails = {"unit": name}, []
            entry = entries.get((TEACHER_LABEL[t], k, name))
            cc, fc = complete_ok(cd_)
            cq, fq = complete_ok(qd)
            info["custody"] = {"cbp_copy_complete": cc, "qpc_copy_complete": cq, "cbp_files_equal_qpc": fc == fq,
                               "cbp_files_equal_pinned_osf_manifest": bool(entry) and
                               fc == entry.get("complete_files_sha256")}
            if not (cc["id_ok"] and cc["rehash_ok"] and not cc["unlisted"] and cq["rehash_ok"] and fc == fq and
                    info["custody"]["cbp_files_equal_pinned_osf_manifest"]):
                fails.append("custody")
            mine = own_teacher(cd_ / "model.pt", [cd_ / "head_0.joblib", cd_ / "head_1.joblib"], D.X)
            sd = tload(cd_ / "model.pt")
            exp = {f"enc.{i}.{j}.{w}" for i in (0, 1) for j in (0, 2, 4) for w in ("weight", "bias")} | \
                  {f"head.{i}.{w}" for i in (0, 1) for w in ("weight", "bias")}
            info["state_keys_ok"] = set(sd) == exp and tuple(sd["enc.0.0.weight"].shape) == (64, N_COLUMNS) and \
                all(v.dtype == torch.float32 for v in sd.values())
            if not info["state_keys_ok"]:
                fails.append("state dict")
            cu, _ = complete_ok(UNITS / f"tea__s{k}__{t}")
            info["cbp_unit_complete"] = cu
            if not (cu["id_ok"] and cu["rehash_ok"] and not cu["unlisted"]):
                fails.append("cbp teacher unit custody")
            tc = np.load(UNITS / f"tea__s{k}__{t}" / "teacher.npz", allow_pickle=False)
            tq = np.load(QPC_UNITS / f"tea__s{k}__{t}" / "teacher.npz", allow_pickle=False)
            td = np.load(DPC_UNITS / f"tea__s{k}__{t}" / "teacher.npz", allow_pickle=False)
            rel = np.load(cd_ / "release.npz", allow_pickle=False)
            trec = jload(UNITS / f"tea__s{k}__{t}" / "record.json")
            info["teacher_unit_keys"] = sorted(tc.files)
            info["row_order_equals_own"] = all(bool(np.array_equal(z["row_id"], D.row_id)) for z in (tc, tq, td, rel))
            if not info["row_order_equals_own"]:
                fails.append("row order")
            par = {}
            for key in ("r1", "r2", "c1", "c2", "p1", "p2", "d1", "d2"):
                a = mine[key]
                rk = key if key[0] != "d" else f"hard{key[1]}"
                par[key] = {"cbp_unit": bitwise(a.astype(tc[key].dtype), tc[key]) and a.dtype == tc[key].dtype,
                            "qpc_unit": bitwise(a.astype(tq[key].dtype), tq[key]),
                            "dpc_unit": bitwise(a.astype(td[key].dtype), td[key]),
                            "admitted_release": bitwise(a.astype(rel[rk].dtype), rel[rk]),
                            "max_abs_diff_cbp_unit": maxdiff(a, tc[key])}
            info["parity"] = par
            if not all(v["cbp_unit"] and v["qpc_unit"] and v["dpc_unit"] and v["admitted_release"] for v in par.values()):
                fails.append("forward parity")
            info["record_model_sha_equals_own"] = trec.get("model_sha256") == sha_file(cd_ / "model.pt") == fc.get("model.pt")
            if not info["record_model_sha_equals_own"]:
                fails.append("teacher binding hash")
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
                    hi.update({"refit_C": C, "saved_head_C": float(head[-1].C), "refit_p_max_abs_diff": maxdiff(mp, P),
                               "refit_decision_mismatches": int((md != mine[f"d{j}"]).sum())})
                    if C != float(head[-1].C) or hi["refit_p_max_abs_diff"] > 1e-8 or hi["refit_decision_mismatches"]:
                        fails.append(f"recipient {j} head refit")
                if not all(v for v in hi.values() if isinstance(v, bool)):
                    fails.append(f"recipient {j} head")
                heads[f"recipient_{j}"] = hi
            info["heads"] = heads
            info["model_pt_sha256"] = fc.get("model.pt")
            info["failures"] = fails
            T[(k, t)] = {kk: mine[kk] for kk in ("p1", "p2", "d1", "d2", "r1", "r2", "c1", "c2")}
            T[(k, t)]["model_sha"] = fc.get("model.pt")
            out[f"{t}|s{k}"] = res("FAIL" if fails else "PASS", **info)
    st = worst(*[v["status"] for v in out.values()]) if len(out) == 6 else "FAIL"
    return res(st, units=out, note="own forward pass on own X (83 permitted columns) from the cbp admitted copies; "
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
    admitted artifacts (read-only) and the deployed heads vs the cbp ref__ units (= qpc = dpc ref__ units)."""
    out = {}
    keep = {} if keep is None else keep
    for k in SEEDS:
        for lab in REFS:
            info, fails = {}, []
            cu_d = UNITS / f"ref__s{k}__{lab}"
            cu, fu = complete_ok(cu_d)
            info["custody"] = {"cbp_unit": cu, "cbp_unit_equals_qpc_unit": dir_hashes(cu_d) ==
                               dir_hashes(QPC_UNITS / f"ref__s{k}__{lab}")}
            if not (cu["rehash_ok"] and not cu["unlisted"] and info["custody"]["cbp_unit_equals_qpc_unit"]):
                fails.append("custody")
            ref = np.load(cu_d / "reference.npz", allow_pickle=False)
            dref = np.load(DPC_UNITS / f"ref__s{k}__{lab}" / "reference.npz", allow_pickle=False)
            info["cbp_unit_equals_dpc_unit"] = sorted(ref.files) == sorted(dref.files) and \
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
                    R = np.eye(int(rec["n_cells"]))[cells]
                    c, p, d = head_outputs(joblib.load(ud / "head.joblib"), R)
                    own.update({f"r{j}": R, f"c{j}": c, f"p{j}": p, f"d{j}": d.astype(np.int64), f"cells{j}": cells})
                    info[f"recipient_{j}_purpose_ok"] = rec["purpose"] == i and bool((cells >= 0).all())
                    if not info[f"recipient_{j}_purpose_ok"]:
                        fails.append(f"FARE recipient {j}")
            par = {a: bitwise(own[a].astype(ref[a].dtype), ref[a]) for a in own}
            info["own_application_bitwise_vs_cbp_unit"] = par
            info["max_abs_diff_p"] = max(maxdiff(own["p1"], ref["p1"]), maxdiff(own["p2"], ref["p2"]))
            info["row_order_ok"] = bool(np.array_equal(ref["row_id"], D.row_id))
            if not (all(par.values()) and info["cbp_unit_equals_dpc_unit"] and info["row_order_ok"]):
                fails.append("application parity")
            keep[(lab, k)] = own
            info["failures"] = fails
            out[f"{lab}|s{k}"] = res("FAIL" if fails else "PASS", **info)
    return res(worst(*[v["status"] for v in out.values()]) if len(out) == 9 else "FAIL", units=out,
               note="label-free: own LEACE formula on own U features; own traversal of the official FARE trees; "
                    "deployed heads; cbp ref__ units must equal the qpc and dpc ref__ units")


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


def own_fit_distortion(P_fit, q_fit):
    return float(np.mean(kl_paired(P_fit, q_fit)))


def token_entropy(tok):
    """Empirical entropy (nats) of the emitted token distribution."""
    cnt = np.bincount(np.asarray(tok, dtype=np.int64))
    pr = cnt[cnt > 0] / cnt.sum()
    return float(-np.sum(pr * np.log(pr)))


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


def b_bank(lams):
    out = [("CLASS", None), ("FINE-TASK", None)]
    for lam in lams:
        out += [(f, lam) for f in ("LOCAL", "SEQ-12", "SEQ-21")]
    out += [("JOINT", lam) for lam in lams]          # witnesses first
    return out


def code_origin(fam, lam):
    if fam in ("CLASS", "FINE-TASK", "DIRECT-TASK") or lam in REUSED_LAMS:
        return "REUSED"
    if lam == COMPOSED_EXTRA_LAM:
        return "COMPOSITION_ONLY"
    return "NEW"


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


def check_codes(D: Data, L, T, lams, allow_missing=()):
    """Code replay at the ONE rate (8, 64) for the lambdas `lams` (prompt sections 7-8; qpc rules unchanged): own fine partitions (three starts, <= 200 rounds),
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
        for fam, lam in b_bank(lams):
            cid = b_config(fam, lam)
            uname = f"pol__s{k}__{cid.replace('|', '_')}"
            origin = code_origin(fam, lam)
            info, fails = {"config": cid, "seed": k, "origin": origin}, []
            if not unit_done(uname):
                if lam in allow_missing:
                    unit_out[f"{cid}#s{k}"] = res("PENDING", reason="not yet fitted", origin=origin)
                else:
                    unit_out[f"{cid}#s{k}"] = res("FAIL", failures=["unit missing"], origin=origin)
                continue
            if fam == "JOINT" and any(b_config(wf, None if wf == "FINE-TASK" else lam) not in mine_lab
                                      for wf in WITNESS_FAMS):
                unit_out[f"{cid}#s{k}"] = res("FAIL", failures=["a witness map was not replayed"], origin=origin)
                continue
            cu, _ = complete_ok(UNITS / uname)
            if not (cu["id_ok"] and cu["rehash_ok"] and not cu["unlisted"]):
                fails.append("custody")
            rec = jload(UNITS / uname / "record.json")
            pj = jload(UNITS / uname / "policy.json")
            relz = np.load(UNITS / uname / "release.npz", allow_pickle=False)
            info["release_keys_exact"] = sorted(relz.files) == sorted(RELEASE_KEYS)
            if not info["release_keys_exact"]:
                fails.append("release keys (no continuous scores, fine IDs or debug arrays may be exported)")
            if origin == "NEW":
                info["record_origin_new_fit"] = rec.get("origin") == "NEW_FIT"
                info["absent_from_qpc_store"] = not (QPC_UNITS / uname).exists()
                if not (info["record_origin_new_fit"] and info["absent_from_qpc_store"]):
                    fails.append("new-fit provenance")
            else:
                info["bytes_equal_qpc_unit"] = dir_hashes(UNITS / uname) == dir_hashes(QPC_UNITS / uname)
                if not info["bytes_equal_qpc_unit"]:
                    fails.append("reused unit differs from its qpc source")
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
            own_pair_fp = hashlib.sha256((Pol(fines[1], tokens_from_lab(labs[1])).fingerprint_dpc() +
                                          Pol(fines[2], tokens_from_lab(labs[2])).fingerprint_dpc()).encode()).hexdigest()
            info["pair_fingerprint_equal"] = own_pair_fp == pj.get("fingerprint")
            info["binding_ok"] = (pj.get("config") or {}).get("config") == cid and \
                (pj.get("config") or {}).get("teacher_model_sha256") == tt["model_sha"] and \
                (pj.get("config") or {}).get("feature_names_sha256") == schema_sha(D.feature_names)
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
                                  "hard2": hards[2], "alpha1": int(T1), "alpha2": int(T2),
                                  "emitted": int(info["recipient_1"]["effective_states"] +
                                                 info["recipient_2"]["effective_states"]), "pair_fp": own_pair_fp}
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
    n_exp = 3 * len(b_bank(lams))
    pend = sorted(k_ for k_, v in unit_out.items() if v["status"] == "PENDING")
    by_origin = {}
    for v in unit_out.values():
        o_ = v.get("origin")
        by_origin.setdefault(o_, {"replayed": 0, "pending": 0, "failing": 0})
        by_origin[o_]["pending" if v["status"] == "PENDING" else "replayed"] += 1
        by_origin[o_]["failing"] += int(v["status"] == "FAIL")
    summary["by_origin"] = by_origin
    summary["pending_units"] = pend
    summary["lambdas"] = list(lams)
    st_s = "FAIL" if fails_u or n_units != n_exp else ("WARN" if warns_u else ("PENDING" if pend and n_units == len(pend)
                                                                               else "PASS"))
    return {"fine_partitions": res(worst(*[v["status"] for v in fine_out.values()]) if len(fine_out) == 3
                                   else "FAIL", units=fine_out),
            "code_search": res(st_s, **summary, per_unit=unit_out),
            "code_class_preservation": res("PASS" if not cp_fail and not adv_fail and n_units == n_exp else "FAIL",
                                           row_checks_passed_by_role=cp_rows, failures=cp_fail,
                                           adversarial_failures=adv_fail, units_checked=n_units - len(pend))}, \
        own_maps, own_terms


RELEASE_KEYS = frozenset({"row_id", "tok1", "q1", "hard1", "alpha1", "tok2", "q2", "hard2", "alpha2"})


def section13_own(rel, fit):
    """Own alphabet / support receipts of one code on OSF_DEFENSE_FIT rows (prompt section 13)."""
    t1, t2 = np.asarray(rel["tok1"])[fit], np.asarray(rel["tok2"])[fit]
    a1, a2 = int(rel["alpha1"]), int(rel["alpha2"])

    def one(codes, alpha):
        _, cnt = np.unique(codes, return_counts=True)
        p_ = cnt / cnt.sum()
        return {"full": int(alpha), "occupied": int(len(cnt)), "entropy": float(-np.sum(p_ * np.log(p_))),
                "singletons": int(np.sum(cnt == 1)), "lt5_occupied": int(np.sum(cnt < 5)),
                "lt5_with_unseen": int(np.sum(cnt < 5) + alpha - len(cnt)), "unseen_fraction": (alpha - len(cnt)) / alpha}
    return {"r1": one(t1, a1), "r2": one(t2, a2), "pair": one(t1.astype(np.int64) * a2 + t2, a1 * a2)}


def check_endpoint_parity(D: Data):
    """The lead's endpoint-parity record of every reused code (run/endpoint_parity.json, written before any new fit)
    vs the verifier's own replay: pair fingerprint, release bitwise flags, class preservation counts and the
    section-13 alphabets, occupancy, entropy, singletons, <5 cells and unseen fractions."""
    p = RUN / "endpoint_parity.json"
    if not p.exists():
        return res("PENDING", reason="endpoint_parity.json not yet written")
    E = jload(p)
    fit = D.fit_idx
    bad, n, conv = [], 0, set()
    for k in SEEDS:
        for cid in reused_cids():
            u = release_unit(k, cid)
            r_ = E.get(u)
            o = OWN_REL.get((k, cid))
            if r_ is None or o is None:
                bad.append(f"{u}: {'no parity record' if r_ is None else 'no own replay'}")
                continue
            n += 1
            if r_.get("pair_fingerprint") != o.get("pair_fp"):
                bad.append(f"{u}: pair fingerprint")
            if not all(v for v in (r_.get("release_bitwise_detail") or {}).values()):
                bad.append(f"{u}: lead release flags")
            cp = r_.get("class_preservation") or {}
            if any((cp.get(x) or {}).get(y) for x in ("r1", "r2") for y in ("stored_failures", "stored_argmax_failures",
                                                                             "reencoded_failures")):
                bad.append(f"{u}: class preservation")
            s13 = r_.get("section13") or {}
            own = section13_own(o, fit)
            al, oc = s13.get("alphabets", {}), s13.get("alphabets", {}).get("occupied_fit", {})
            for v in ("r1", "r2", "pair"):
                if (al.get("full") or {}).get(v) != own[v]["full"] or oc.get(v) != own[v]["occupied"]:
                    bad.append(f"{u}/{v}: alphabet / occupancy")
                if abs(float((s13.get("entropy_nats_fit") or {}).get(v, np.nan)) - own[v]["entropy"]) > 1e-12:
                    bad.append(f"{u}/{v}: entropy")
                if (s13.get("singletons_fit") or {}).get(v) != own[v]["singletons"]:
                    bad.append(f"{u}/{v}: singletons")
                lt = (s13.get("lt5_fit") or {}).get(v)
                if lt == own[v]["lt5_occupied"]:
                    conv.add("occupied")
                elif lt == own[v]["lt5_with_unseen"]:
                    conv.add("with_unseen")
                else:
                    bad.append(f"{u}/{v}: <5 cells")
                if abs(float((s13.get("unseen_fraction_fit") or {}).get(v, np.nan)) - own[v]["unseen_fraction"]) > 1e-15:
                    bad.append(f"{u}/{v}: unseen fraction")
    return res("FAIL" if bad or n != 3 * len(reused_cids()) else "PASS", units=n, failures=bad[:20],
               lt5_convention=sorted(conv), rule="own replay vs the lead's endpoint-parity record (exact counts; "
                                                 "entropy 1e-12; unseen fraction 1e-15)")


def check_direct(D: Data, L, T):
    """DIRECT-TASK replay at all 8 admitted rates (i8o64 is a registered code: Q and a T* / C_rate candidate; the other
    7 are composition-only public maps): own per-recipient three-start KL k-means (qpc rule: <= 200 passes, best
    coherent iterate, later start wins only beyond 1e-12 max(|J|, 1)) on OSF_DEFENSE_FIT teacher probabilities;
    receipts vs the qpc dir__ units (read-only); every pair unit: partition bitwise, identity token map, fingerprints,
    binding, tokens / decoded vectors / decisions bitwise on all rows, class preservation incl. adversarial rows."""
    fit = D.fit_idx
    fits, pair_out, rec_out, cp_fail, adv_fail = {}, {}, {}, [], []
    cp_rows = {r_: 0 for r_ in ROLES}
    advP = {K: adversarial_rows(K) for K in KS}
    t0 = time.process_time()
    for k in SEEDS:
        tt = T[(k, "U")]
        P, d = {1: tt["p1"], 2: tt["p2"]}, {1: tt["d1"], 2: tt["d2"]}
        for r, rates in ((1, RATES_INCOME), (2, RATES_OCC)):
            for m in rates:
                fine = fit_partition(P[r][fit], d[r][fit], KS[r - 1], m, starts=START_ORDER, rule="qpc")
                fits[(k, r, m)] = fine
                qn = QPC_UNITS / f"dir__s{k}__r{r}__m{m}"
                info = {}
                if (qn / "policy.json").exists():
                    stored = Pol.from_json(jload(qn / "policy.json"))
                    rr = jload(qn / "record.json").get("receipts", {})
                    rc = compare_start_receipts(fine.receipt["per_class"], rr.get("per_class", []))
                    fps = rr.get("fingerprints", {})
                    own_fps = {"winner": fine.fingerprint_dpc(), **{st_: fine.receipt["by_start"][st_].fingerprint_dpc()
                                                                    for st_ in START_ORDER}}
                    info = {"partition_bitwise_vs_qpc_dir_unit": all(fine.equal(stored.fine).values()),
                            "receipts_ok": receipts_ok(rc), "start_fingerprints_equal": all(fps.get(x) == own_fps[x]
                                                                                           for x in own_fps),
                            "fit_distortion_rel_diff": rel_diff(own_fit_distortion(P[r][fit],
                                                                                   direct_policy(fine).release(P[r][fit], d[r][fit])[1]),
                                                                rr.get("fit_distortion", np.nan))}
                    info["ok"] = info["partition_bitwise_vs_qpc_dir_unit"] and info["receipts_ok"] and \
                        info["start_fingerprints_equal"] and info["fit_distortion_rel_diff"] <= 1e-12
                else:
                    info = {"ok": None, "note": "qpc dir__ unit not present (receipts not compared)"}
                rec_out[f"s{k}/r{r}/m{m}"] = info
        for m1 in RATES_INCOME:
            for m2 in RATES_OCC:
                cid = direct_id(m1, m2)
                name = release_unit(k, cid)
                info, fails = {"origin": "REUSED" if (m1, m2) == RATE else "COMPOSITION_ONLY"}, []
                if not unit_done(name):
                    pair_out[f"{cid}#s{k}"] = res("FAIL", failures=["unit missing"])
                    continue
                cu, _ = complete_ok(UNITS / name)
                if not (cu["id_ok"] and cu["rehash_ok"] and not cu["unlisted"]):
                    fails.append("custody")
                pj = jload(UNITS / name / "policy.json")
                rec = jload(UNITS / name / "record.json")
                relz = np.load(UNITS / name / "release.npz", allow_pickle=False)
                info["release_keys_exact"] = sorted(relz.files) == sorted(RELEASE_KEYS)
                info["row_order_ok"] = bool(np.array_equal(relz["row_id"], D.row_id))
                cfg = pj.get("config") or {}
                info["binding"] = {"config": cfg.get("config") == cid, "teacher_model_sha256":
                                   cfg.get("teacher_model_sha256") == tt["model_sha"],
                                   "feature_names_sha256": cfg.get("feature_names_sha256") == schema_sha(D.feature_names)}
                pols = {}
                for r, m in ((1, m1), (2, m2)):
                    stored = Pol.from_json(pj[f"p{r}"])
                    cmp_, pol, tok, q, hard = compare_policy_to_own(stored, fits[(k, r, m)], P[r], d[r], relz, r)
                    pols[r] = (pol, tok, q, hard)
                    ok_cp = (hard == d[r]) & (q.argmax(1) == d[r]) & (relz[f"hard{r}"] == d[r])
                    for ro in ROLES:
                        cp_rows[ro] += int((ok_cp & D.mask[ro]).sum())
                    if not ok_cp.all():
                        cp_fail.append(f"{cid}#s{k} r{r}")
                    Pa = advP[KS[r - 1]]
                    _, qa, ha, _ = pol.release(Pa, Pa.argmax(1))
                    if not (np.array_equal(ha, Pa.argmax(1)) and np.array_equal(qa.argmax(1), Pa.argmax(1))):
                        adv_fail.append(f"{cid}#s{k} r{r}")
                    cmp_["fit_distortion_rel_diff"] = rel_diff(own_fit_distortion(P[r][fit], q[fit]),
                                                               (rec.get("fit_distortion") or {}).get(f"D{r}", np.nan))
                    cmp_["support"] = {"tokens_with_n_lt_5": int(np.sum((pol.token_n < SPARSE_N) & ~pol.token_fb)),
                                       "singleton_tokens": int(np.sum(pol.token_n == 1)),
                                       "fallback_tokens": int(pol.token_fb.sum()), "alphabet": pol.T,
                                       "emitted_fit": int(sum(pol.effective_per_class()))}
                    cmp_.pop("partition_fields", None)
                    info[f"recipient_{r}"] = cmp_
                    if not (cmp_["partition_bitwise"] and cmp_["identity_map"] and cmp_["policy_fingerprint_equal"] and
                            cmp_["tokens_bitwise"] and cmp_["decoded_bitwise"] and cmp_["decisions_bitwise"] and
                            cmp_["alpha_equal"] and cmp_["decisions_equal_teacher_all_rows"] and
                            cmp_["own_structure_ok"] and cmp_["stored_structure_ok"] and
                            cmp_["fit_distortion_rel_diff"] <= 1e-12):
                        fails.append(f"recipient {r}")
                p1, p2 = pols[1][0], pols[2][0]
                own_pair_fp = hashlib.sha256((p1.fingerprint_dpc() + p2.fingerprint_dpc()).encode()).hexdigest()
                info["pair_fingerprint_equal"] = own_pair_fp == pj.get("fingerprint")
                info["record_states_equal_own_emitted"] = rec.get("total_states") == int(sum(p1.effective_per_class()) +
                                                                                         sum(p2.effective_per_class()))
                info["alphabet"] = [p1.T, p2.T]
                if not (info["release_keys_exact"] and info["row_order_ok"] and all(info["binding"].values()) and
                        info["pair_fingerprint_equal"] and info["record_states_equal_own_emitted"]):
                    fails.append("binding / fingerprint / states / keys")
                OWN_REL[(k, cid)] = {"tok1": pols[1][1], "q1": pols[1][2], "hard1": pols[1][3], "tok2": pols[2][1],
                                     "q2": pols[2][2], "hard2": pols[2][3], "alpha1": p1.T, "alpha2": p2.T,
                                     "emitted": int(sum(p1.effective_per_class()) + sum(p2.effective_per_class())),
                                     "pair_fp": own_pair_fp}
                info["failures"] = fails
                pair_out[f"{cid}#s{k}"] = res("FAIL" if fails else "PASS", **info)
    rbad = [k_ for k_, v in rec_out.items() if v.get("ok") is False]
    st = worst(*[v["status"] for v in pair_out.values()]) if len(pair_out) == 24 else "FAIL"
    if rbad:
        st = "FAIL"
    return {"direct_task_codes": res(st, units=pair_out, recipient_fits_vs_qpc_dir_units=rec_out,
                                     receipt_failures=rbad, own_fit_cpu_s=round(time.process_time() - t0, 1)),
            "direct_task_class_preservation": res("PASS" if not cp_fail and not adv_fail and len(pair_out) == 24 else
                                                  "FAIL", row_checks_passed_by_role=cp_rows, failures=cp_fail,
                                                  adversarial_failures=adv_fail)}


STAGE_MODULES = {"admit": ["cbp/data.py", "cbp/admit.py", "cbp/run.py"],
                 "fit": ["cbp/fit.py", "cbp/run.py", "qpc/compress.py", "qpc/release.py"],
                 "inner": ["cbp/audit.py", "cbp/run.py"], "inner_src": ["cbp/audit.py", "cbp/run.py"],
                 "controls": ["cbp/audit.py", "cbp/run.py"], "select": ["cbp/select.py", "cbp/family.py", "cbp/run.py"]}


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
        gov = STAGE_LOCK[stage][0]
        L_ = jload(RES / f"{gov}.json") if (RES / f"{gov}.json").exists() else {}
        cf = L_.get("code_files") or {}
        stage_mod[stage] = {m: (m in cf) for m in mods}
        for m in mods:
            if m in current and (WT / m).exists() and sha_file(WT / m) != current[m][0]:
                wt_mism.append(f"{m} (governed by {current[m][1]})")
    missing = [f"{st}:{m}" for st, v in stage_mod.items() for m, ok in v.items() if not ok and
               (RES / f"{STAGE_LOCK[st][0]}.json").exists()]
    bad = any(v["files_differing_from_the_lock_commit"] for v in at_commit.values()) or relocks or missing
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
    st = "FAIL" if bad else ("WARN" if doc_changed or wt_mism else "PASS")
    return res(st, locked_documents_changed_since_their_latest_lock=doc_changed, post_lock_document_changes=post_lock,
               reason=("locked files or documents modified after their latest governing lock (allowed before the next "
                       "named lock only if that lock declares each change; a FAIL once a stage runs on them)")
               if doc_changed or wt_mism else None,
               locks_in_order=[n for _, n in order], files_governed=len(current),
               lock_vs_its_commit=at_commit, undeclared_relocks=relocks, stage_modules_locked=stage_mod,
               stage_modules_missing_from_lock=missing, stage_modules_changed_in_worktree=wt_mism)


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
    once before any new fit, and every lead stage start inside a semaphore hold (SEMA_LOG)."""
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
        govs = STAGE_LOCK.get(stage) or ()
        pushes = [(info.get(g0) or {}).get("first_push_time") for g0 in govs]
        gov = "+".join(govs) if govs else None
        g_ = {"first_push_time": max(pushes) if pushes and all(pushes) else None}
        ok = bool(govs) and bool(g_.get("first_push_time")) and g_["first_push_time"] <= t
        unpushed = []
        for name, d in lf.items():
            for c, ct in d["commits"]:
                if parse_iso(ct) <= t:
                    pt = first_remote(c, entries)
                    if pt is None or pt > t:
                        unpushed.append(name)
        in_sema = any(h0 <= t and (h1 is None or t <= h1) and str(lab).startswith("A:") for h0, h1, lab in holds)
        if stage == "fit":
            gs = [(info.get(g0) or {}).get("first_push_time") for g0 in STAGE_LOCK["fit"]]
            if not all(gs) or max(gs) > t:
                fails.append(f"start fit at {iso(t)}: FIT_LOCK and AUDIT_AND_SELECTION_LOCK not both pushed before it")
        rec = {"stage": stage, "at": t, "lock_named": e.get("lock"), "governing_lock": gov, "ok": ok and not unpushed,
               "governing_lock_push_lead_s": (t - g_["first_push_time"]).total_seconds() if g_.get("first_push_time") else None,
               "unpushed_lock_versions": sorted(set(unpushed)), "inside_lead_semaphore_hold": in_sema}
        if not rec["ok"]:
            fails.append(f"start {stage} at {iso(t)}: governing lock not pushed before it")
        if not in_sema:
            fails.append(f"start {stage} at {iso(t)}: no lead semaphore hold")
        st_out.append(rec)
    # every completed unit after its governing lock's first push (activity event and COMPLETE.json mtime)
    admitted = {u_ for k in SEEDS for u_ in admitted_units(k)}
    adm_starts = [parse_iso(e["at"]) for e in starts if e["event"] == "start admit"]
    adm_push = (info.get("SOURCE_ADMISSION_LOCK") or {}).get("first_push_time")
    admission = {"units_admitted_by_verified_copy": sum(1 for q in UNITS.iterdir() if q.name in admitted)
                 if UNITS.exists() else 0, "admit_starts": [iso(t_) for t_ in adm_starts],
                 "admit_started_after_lock_push": bool(adm_starts) and bool(adm_push) and adm_push <= min(adm_starts),
                 "rule": "copied units keep their qpc file times; their custody is the byte equality checked in "
                         "admitted_custody, and the admit stage must start after the SOURCE_ADMISSION_LOCK push"}
    if adm_starts and not admission["admit_started_after_lock_push"]:
        fails.append("admit stage started before the SOURCE_ADMISSION_LOCK push")

    def unit_lock(nm):
        if nm in admitted:
            return None
        if nm.startswith(("tea__", "ref__", "fine__")):
            return "SOURCE_ADMISSION_LOCK"
        if nm.startswith("pol__"):
            return "FIT_LOCK"
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
    fit_starts = [parse_iso(e["at"]) for e in starts if e["event"] == "start fit"]
    pred = {"versions": len(plog), "first_commit": pfirst[0] if pfirst else None, "first_push_time": p_push,
            "worktree_equals_first_commit": bool(pfirst) and git("rev-parse", f"{pfirst[0]}:{prel}") ==
            git("hash-object", str(RES / "PREDICTIONS.json")),
            "first_fit_start": min(fit_starts) if fit_starts else None}
    pred["pushed_before_any_new_fit"] = bool(p_push) and (not fit_starts or p_push < min(fit_starts))
    if not (pred["versions"] == 1 and pred["worktree_equals_first_commit"] and pred["pushed_before_any_new_fit"]):
        fails.append("PREDICTIONS.json not pushed once before any new fit (or revised)")
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
               remote=remote, failures=fails, absent_later_locks=[n for n in LOCK_ORDER if not lf[n]["exists"]],
               admission=admission)


VIEWS = ("v1", "v2", "pair")
SLATE_15 = [f"LR_C{C}" for C in (0.01, 0.1, 1.0, 10.0, 100.0)] + ["MLP_64", "MLP_128", "MLP_64x64", "MLP_128x128"] + \
    [f"HGB_{lr}_{lv}" for lr in (0.05, 0.1) for lv in (15, 31)] + ["DA_LR", "DA_MLP"]
CC_LOCAL = [f"CC_alpha{a}" for a in (0.5, 1.0, 5.0)]
CC_PAIR = [f"CCpair_alpha{a}" for a in (0.5, 1.0, 5.0)]


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
            "alpha1": o["alpha1"], "alpha2": o["alpha2"], "emitted": o.get("emitted"), "finite": True}


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
        for cid in all_cids() + extra_cids():
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
                             "states": states, "emitted": rel.get("emitted"), "rec": rec,
                             "composition_only": cid in extra_cids()}
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
            cc_ = u_ if u_ in composition_cids() else next((c_ for c_ in composition_cids()
                                                            if release_unit(k, c_) == u_), None)
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
        rf = sorted({(x if x in composition_cids() else next((c_ for c_ in composition_cids()
                                                              if release_unit(k, c_) == x), x)) for x in rf})
        comp_out[f"s{k}"]["freeze_own"] = own_freeze
        if rf != own_freeze:
            comp_bad.append(f"SRC|U#s{k}: freeze {rf} vs own {own_freeze}")
        expected = len(composition_cids())                     # 27 registered codes + 11 composition-only maps
        if n_pol != expected or sorted(c_ for c_ in pol_cids if c_) != sorted(composition_cids()):
            comp_bad.append(f"SRC|U#s{k}: composed over {n_pol} codes, expected the closure of {expected}")
    tol_ok = mx["selected_value"] <= 1e-12 and mx["table"] <= 1e-12 and mx["recovery"] <= 1e-12 and mx["utility"] <= 1e-12
    bad = fails or slate_bad or sel0_bad or cov_bad or not tol_ok
    inner = res("FAIL" if bad else "PASS", units=len(own), candidates_recomputed=n_cand,
                max_abs_diff=mx, slate_incomplete=slate_bad[:20], seed0_selected_not_bank_member=sel0_bad[:20],
                coverage_mismatches=cov_bad[:20], failures=fails[:30],
                rule="own Mann-Whitney AUC and 1e-12-clipped CE of every stored candidate; v_i bank = own slate (+ CC on "
                     "finite views); pair bank = coalition (+ CCpair) + ignore-recipient-2 (v1) + ignore-recipient-1 (v2); "
                     "first maximum / minimum beyond 1e-12; recovery = mean over attacker seeds 0-2 of the selected "
                     "attacker; utility from own releases")
    comp = res("FAIL" if comp_bad or comp_mx > 1e-12 or not comp_out else "PASS", per_seed=comp_out, max_abs_diff=comp_mx,
               failures=comp_bad[:20], rule="per family and view the winner is the first bank (own, then every fitted "
               "code of the seed in lock order) whose seed-0 selected value beats the running best by > 1e-12; "
               "decisions compose only with the class-only code; recovery = the winner's seed 0-2 mean")
    return inner, comp, own


# ------------------------------------------------------------------------------------------------ PHASE 2 selection
def own_rows_from_inner(own, states_key="alphabet"):
    """config_row inputs from the own inner replay: per seed the AUC-selected recovery of the primary family (for
    SRC|U already the composed recovery), own INNER utility, U's utility of the same seed, decision preservation and
    the token states (alphabet alpha1 + alpha2, or the emitted fitting-row states; continuous = +inf)."""
    rows = {}
    for cid in all_cids():
        per = {}
        for k in SEEDS:
            o = own.get((k, cid))
            if o is None:
                continue
            a = o["fams"][o["primary"]]["auc"]
            stv = o["states"] if states_key == "alphabet" else o.get("emitted")
            per[k] = {"auc": {w: float(a[w]) for w in VIEWS}, "util": o["utility"], "U": o["U"],
                      "preserved": o["preserved"], "states": math.inf if stv is None else float(stv),
                      "emitted": o.get("emitted"), "pair_fp": (OWN_REL.get((k, cid)) or {}).get("pair_fp")}
        rows[cid] = config_row(cid, per)
    return rows


def eligibility_table(rows):
    """Ordinary AND headroom eligibility for every seed and configuration (aggregates only)."""
    out = {}
    for cid, r in rows.items():
        if not r.get("valid"):
            out[cid] = {"valid": False, "invalid": r.get("invalid")}
            continue
        out[cid] = {"valid": True, "ordinary_eligible": r["ordinary_eligible"], "headroom_eligible": r["headroom_eligible"],
                    "shortfall_ordinary": r["shortfall_ordinary"], "shortfall_headroom": r["shortfall_headroom"],
                    "mean_pair": r["mean_pair"], "mean_v1": r["mean_v1"], "mean_v2": r["mean_v2"],
                    "mean_sum_logloss": r["mean_sum_logloss"], "mean_states": r["mean_states"]
                    if math.isfinite(r["mean_states"]) else None,
                    "per_seed": {f"s{k}": {"ordinary": r["seeds"][k]["ordinary_ok"], "headroom": r["seeds"][k]["headroom_ok"],
                                           "excess": {TASKS[i - 1]: r["seeds"][k]["tasks"][i]["excess"] for i in (1, 2)},
                                           "failing": r["failing"][f"s{k}"]} for k in SEEDS},
                    "borderline_forms": r["borderline_forms"]}
    return out


def brief_role(v):
    return {kk: v.get(kk) for kk in ("status", "config", "descriptive_config", "reason", "aliases",
                                     "fallback_shortfalls", "fallback_rank_status") if kk in v}


def brief_diagnostics(d):
    return {"ordinary_privacy_winner_no_headroom": brief_role(d["ordinary_privacy_winner_no_headroom"]),
            "strongest_ordinary_privacy_unguarded": brief_role(d["strongest_ordinary_privacy_unguarded"]),
            "family_headroom_winners": {f_: brief_role(v) for f_, v in d["family_headroom_winners"].items()},
            "source_lambda_0.1_controls": d["source_lambda_0.1_controls"],
            "headroom_changes_winner": d["headroom_changes_winner"]}


def _alias_equal(lead_al, own_al, cid):
    if not lead_al and not own_al:
        return True
    if not lead_al or not own_al or not own_al.get("available"):
        return False
    return (sorted(lead_al.get("full") or []) == sorted([cid] + own_al["alias_set"]) and
            {k_: sorted(v) for k_, v in (lead_al.get("partial") or {}).items()} ==
            {k_: sorted(v) for k_, v in own_al["partial_aliases"].items()} and
            lead_al.get("simplest_family") == own_al["simplest_family"] and
            sorted(lead_al.get("identical_to_untrained") or []) == own_al["identical_to_untrained"])


def compare_selection_detail(S, mine, rows):
    """Field-by-field comparison of the lead's selection record with the own selection: every role's evaluated
    candidates (eligibility, guard, nominability, the three shortfalls), aliases, fallbacks and every diagnostic."""
    diffs, n = [], 0
    tol = 1e-9

    def num(a, b, what):
        nonlocal n
        n += 1
        if (a is None) != (b is None) or (a is not None and abs(float(a) - float(b)) > tol):
            diffs.append(f"{what}: {a} vs own {b}")

    def eq(a, b, what):
        nonlocal n
        n += 1
        if a != b:
            diffs.append(f"{what}: {a} vs own {b}")
    st_l, st_m = S.get("statuses") or {}, mine["statuses"]
    for x in ("T*", "C_rate", "C_global", "P*", "J*"):
        L_, M_ = st_l.get(x) or {}, st_m[x]
        eq(L_.get("reason"), M_.get("reason"), f"{x}.reason")
        eq(L_.get("fallback_rank_status") if L_.get("fallback_rank_status") != "VALID" else None,
           M_.get("fallback_rank_status"), f"{x}.fallback_rank_status")
        own_ev = {e["config"]: e for e in M_.get("evaluated") or []}
        lead_ev = {e["config"]: e for e in L_.get("evaluated") or []}
        eq(sorted(lead_ev), sorted(own_ev), f"{x}.evaluated candidates")
        nominee_role = x in ("P*", "J*")
        for c_, le in lead_ev.items():
            me = own_ev.get(c_)
            if me is None:
                continue
            eq(bool(le.get("ordinary")), me["ordinary_eligible"], f"{x}/{c_}.ordinary")
            eq(bool(le.get("headroom")), me["headroom_eligible"], f"{x}/{c_}.headroom")
            eq(bool(le.get("eligible")), me["eligible_for_role"], f"{x}/{c_}.eligible")
            eq(le.get("guard_ok"), me["guard_ok"], f"{x}/{c_}.guard_ok")
            eq(bool(le.get("nominable")), me["nominable"], f"{x}/{c_}.nominable")
            num(le.get("ordinary_shortfall"), me["shortfall_ordinary"], f"{x}/{c_}.ordinary_shortfall")
            if nominee_role:
                num(le.get("headroom_shortfall"), me["shortfall_headroom"], f"{x}/{c_}.headroom_shortfall")
            num(le.get("guard_shortfall"), me["shortfall_guard"], f"{x}/{c_}.guard_shortfall")
        if x in ("P*", "J*"):
            c_ = M_.get("config") or M_.get("descriptive_config")
            n += 1
            if not _alias_equal(L_.get("aliases"), M_.get("aliases"), c_):
                diffs.append(f"{x}.aliases: {L_.get('aliases')} vs own {M_.get('aliases')}")
    eq((st_l.get("Q") or {}).get("status"), st_m["Q"]["status"], "Q.status")
    eq((st_l.get("Q") or {}).get("config"), st_m["Q"].get("config"), "Q.config")
    DL, DM = S.get("diagnostics") or {}, mine["diagnostics"]
    for k_ in ("ordinary_privacy_winner_no_headroom", "strongest_ordinary_privacy_unguarded"):
        for f_ in ("status", "config", "descriptive_config", "reason"):
            eq((DL.get(k_) or {}).get(f_), DM[k_].get(f_), f"{k_}.{f_}")
    for f_, v in DM["family_headroom_winners"].items():
        lv = (DL.get("family_headroom_winners") or {}).get(f_) or {}
        for g_ in ("status", "config", "descriptive_config", "reason"):
            eq(lv.get(g_), v.get(g_), f"family_headroom_winners.{f_}.{g_}")
        n += 1
        if not _alias_equal(lv.get("aliases"), v.get("aliases"), v.get("config") or v.get("descriptive_config")):
            diffs.append(f"family_headroom_winners.{f_}.aliases")
    jn = DL.get("joint_family_winner_is_not_J*") or {}
    jf = DM["family_headroom_winners"]["JOINT"]
    eq(jn.get("joint_family_T*_guarded"), jf.get("config") or jf.get("descriptive_config"), "joint family winner")
    eq(jn.get("J*"), mine["resolved"]["J*"], "J* (descriptive) beside the JOINT family winner")
    for c_, v in DM["source_lambda_0.1_controls"].items():
        lv = (DL.get("source_lambda_0.1_controls") or {}).get(c_) or {}
        eq(bool(lv.get("ordinary")), v["ordinary_eligible"], f"source {c_}.ordinary")
        eq(bool(lv.get("headroom")), v["headroom_eligible"], f"source {c_}.headroom")
        for a_, b_ in (("mean_pair", "mean_pair"), ("mean_v1", "mean_v1"), ("mean_v2", "mean_v2"),
                       ("ordinary_shortfall", "shortfall_ordinary"), ("headroom_shortfall", "shortfall_headroom")):
            num(lv.get(a_), v[b_], f"source {c_}.{a_}")
    hl, hm = DL.get("headroom_changes_winner") or {}, DM["headroom_changes_winner"]
    eq(hl.get("changed"), hm.get("changed"), "headroom_changes_winner.changed")
    gl = hl.get("pair_auc_given_up_by_headroom") or {}
    if "give_up_mean_pair_auc" in hm:
        num(gl.get("mean"), hm["give_up_mean_pair_auc"], "give-up mean")
        for k in SEEDS:
            num((gl.get("per_seed") or {}).get(str(k)), hm["give_up_per_seed"][f"s{k}"], f"give-up s{k}")
    return {"fields_compared": n, "differences": diffs[:30], "n_differences": len(diffs)}


def mutation_power_selection(rows, own, S):
    """Real-data deliberate defects (prompt section 14) applied to the own inner rows: each mutated selection must
    differ from the lead's recorded resolution (so the comparison above would have caught it). A defect that changes
    nothing on these data is reported as not exercised, not as detected."""
    rec = S.get("resolved") or {}
    out = {}
    m = my_selection_cbp(rows, reverse=True)["resolved"]
    out["reversed_best_worst_order"] = m != rec
    m = my_selection_cbp(rows, headroom_as_ordinary=True)["resolved"]
    out["headroom_changed_to_ordinary_limit"] = m != rec
    per = {}
    for cid in all_cids():
        per[cid] = {k: {"auc": own[(k, cid)]["fams"][own[(k, cid)]["primary"]]["auc"], "util": own[(k, cid)]["utility"],
                        "U": own[(k, cid)]["U"], "preserved": own[(k, cid)]["preserved"],
                        "states": math.inf if own[(k, cid)]["states"] is None else float(own[(k, cid)]["states"])}
                    for k in SEEDS if (k, cid) in own}
    avg_rows = {c_: config_row(c_, v, seed_average=True) for c_, v in per.items()}
    changed = sorted(c_ for c_ in rows if rows[c_].get("valid") and (avg_rows[c_]["ordinary_eligible"],
                                                                      avg_rows[c_]["headroom_eligible"]) !=
                     (rows[c_]["ordinary_eligible"], rows[c_]["headroom_eligible"]))
    m = my_selection_cbp(avg_rows)["resolved"]
    out["seed_averaged_eligibility"] = {"configs_whose_eligibility_changes": changed, "resolution_changes": m != rec,
                                        "exercised": bool(changed)}
    wrong_auc = {}
    for c_, r in rows.items():
        r = dict(r)
        if r.get("valid"):
            r["seeds"] = {k: {**r["seeds"][k], "auc": {w: 1.0 - v for w, v in r["seeds"][k]["auc"].items()}}
                          for k in SEEDS}
            for w, f_ in (("pair", "mean_pair"), ("v1", "mean_v1"), ("v2", "mean_v2")):
                r[f_] = seed_mean([r["seeds"][k]["auc"][w] for k in SEEDS])
        wrong_auc[c_] = r
    out["wrong_auc_orientation"] = my_selection_cbp(wrong_auc)["resolved"] != rec
    out["all_detected"] = all(v for k_, v in out.items() if isinstance(v, bool))
    return out


def _lead_selection():
    for p in (RUN / "selection.json", RES / "SELECTION.json"):
        if p.exists():
            return jload(p), p.name
    return None, None


def check_selection(own):
    """Own selection (prompt section 9) on own inner rows vs the lead's selection record and public tables."""
    rows = own_rows_from_inner(own)
    rows_emit = own_rows_from_inner(own, "emitted")
    mine = my_selection_cbp(rows)
    mine_emit = my_selection_cbp(rows_emit)
    st_m = {x: {kk: v.get(kk) for kk in ("status", "config", "descriptive_config", "reason", "winning_family",
                                        "family_of_config", "missing_guards", "fallback_shortfalls",
                                        "fallback_rank_status", "aliases", "invalid_candidates")}
            for x, v in mine["statuses"].items()}
    out = {"own_statuses": st_m, "own_resolved": mine["resolved"], "own_aliases": mine["aliases"],
           "own_diagnostics": brief_diagnostics(mine["diagnostics"]),
           "state_count_convention_changes_resolution": mine_emit["resolved"] != mine["resolved"],
           "eligibility": eligibility_table(rows),
           "counts": {"configs": len(rows), "valid": sum(1 for r in rows.values() if r.get("valid")),
                      "ordinary_eligible": sorted(c_ for c_, r in rows.items() if r.get("ordinary_eligible")),
                      "headroom_eligible": sorted(c_ for c_, r in rows.items() if r.get("headroom_eligible")),
                      "borderline_form_cases": sorted(c_ for c_, r in rows.items() if r.get("borderline_forms"))}}
    S, src = _lead_selection()
    if S is None:
        return res("PENDING", reason="the lead's selection record is not yet written; own selection computed", **out), \
            mine, rows
    diffs, elig = [], []
    lst = S.get("statuses") or {}
    keymap = {"Q": ("Q", "Q*")}
    for x in mine["statuses"]:
        v = next((lst[a_] for a_ in keymap.get(x, (x,)) if a_ in lst), None)
        if v is None:
            diffs.append(f"{x}: absent from {src}")
            continue
        m = mine["statuses"][x]
        for f_ in ("config", "descriptive_config", "winning_family"):
            if (v.get(f_) or None) != (m.get(f_) or None):
                diffs.append(f"{x}.{f_}: {v.get(f_)} vs own {m.get(f_)}")
        if (v.get("status") or "") != (m.get("status") or ""):
            diffs.append(f"{x}.status: {v.get('status')} vs own {m.get('status')} (vocabulary to reconcile)")
    lrows = S.get("rows") or {}
    mx, sfx, cmp_n = 0.0, 0.0, 0
    for cid in all_cids():
        r_, m = lrows.get(cid), rows.get(cid)
        if r_ is None:
            elig.append(f"{cid}: absent from the lead's rows")
            continue
        if bool(r_.get("ok", True)) != bool(m.get("valid")):
            elig.append(f"{cid}: validity {r_.get('ok')} vs own {m.get('valid')}")
            continue
        if not m.get("valid"):
            continue
        cmp_n += 1
        for f_ in ("mean_pair", "mean_v1", "mean_v2", "mean_sum_logloss"):
            if r_.get(f_) is not None:
                mx = max(mx, abs(float(r_[f_]) - m[f_]))
        ms = m["mean_states"] if math.isfinite(m["mean_states"]) else None
        if (r_.get("mean_states") is None) != (ms is None) or (ms is not None and float(r_["mean_states"]) != ms):
            elig.append(f"{cid}: mean_states {r_.get('mean_states')} vs own {ms}")
        for f_, own_v in (("ordinary", m["ordinary_eligible"]), ("headroom", m["headroom_eligible"])):
            if f_ in r_ and bool(r_[f_]) != own_v:
                elig.append(f"{cid}.{f_}: {r_[f_]} vs own {own_v}")
        for f_, own_v in (("ordinary_shortfall", m["shortfall_ordinary"]), ("headroom_shortfall", m["shortfall_headroom"])):
            if r_.get(f_) is not None:
                sfx = max(sfx, abs(float(r_[f_]) - own_v))
        for k in SEEDS:
            ls = (r_.get("seeds") or {}).get(str(k)) or (r_.get("seeds") or {}).get(k) or {}
            ms_ = m["seeds"][k]
            for f_, own_v in (("ordinary", ms_["ordinary_ok"]), ("headroom", ms_["ordinary_ok"] and ms_["headroom_ok"])):
                if f_ in ls and bool(ls[f_]) != bool(own_v):
                    elig.append(f"{cid}#s{k}.{f_}: {ls[f_]} vs own {own_v}")
            for w in VIEWS:
                if (ls.get("auc") or {}).get(w) is not None:
                    mx = max(mx, abs(float(ls["auc"][w]) - ms_["auc"][w]))
            for i, t_ in ((1, "income"), (2, "occupation")):
                for f_, kk in (("ll_excess", "logloss"), ("brier_excess", "brier")):
                    if (ls.get(f_) or {}).get(t_) is not None:
                        mx = max(mx, abs(float(ls[f_][t_]) - ms_["tasks"][i]["excess"][kk]))
    det = compare_selection_detail(S, mine, rows)
    pub = jload(RES / "SELECTION.json") if (RES / "SELECTION.json").exists() else {}
    pub_ok = all(((pub.get("statuses") or {}).get(x) or {}).get("status") == (S.get("statuses") or {}).get(x, {}).get(
        "status") and ((pub.get("statuses") or {}).get(x) or {}).get("config") == (S.get("statuses") or {}).get(x, {}).get(
        "config") for x in (S.get("statuses") or {})) if pub else None
    out.update({"lead_record": src, "differences": diffs, "eligibility_differences": elig[:30],
                "n_eligibility_differences": len(elig), "rows_compared": cmp_n, "row_max_abs_diff": mx,
                "shortfall_max_abs_diff": sfx, "detail": det, "public_SELECTION_json_consistent": pub_ok,
                "resolved_equal": (S.get("resolved") or {}) == mine["resolved"],
                "mutation_power_on_real_rows": mutation_power_selection(rows, own, S)})
    ok = not diffs and not elig and mx <= 1e-12 and sfx <= 1e-9 and not det["n_differences"] and pub_ok is not False \
        and out["resolved_equal"] and out["mutation_power_on_real_rows"]["all_detected"]
    return res("PASS" if ok else "FAIL", **out), mine, rows


def unsealed_labels(D: Data, gate):
    """Assessment labels, read ONLY after evaluation_lock_gate() verified the committed lock on origin."""
    if not gate.get("ok"):
        raise RuntimeError("REFUSED: EVALUATION_LOCK not verified on origin; assessment labels stay sealed")
    z = np.load(SRC, allow_pickle=False)
    return {k: z[v].astype(np.int64)[D.keep].copy() for k, v in LABEL_KEYS.items()}


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


def own_endpoints(preds, EL, levels=True, boot=None, z=Z_PRIMARY):
    """Own 37 endpoints from the stored assessment predictions and the lock (prompt section 11; the slots are re-derived
    from the prompt definitions and only compared with the published PRIMARY_FAMILY text). `boot` / `z` are injectable
    for the deliberate-defect tests (wrong bootstrap group, wrong family critical value)."""
    labels = EL["scored_labels"]
    p0 = preds[(0, labels[0])]
    sex, units = p0["sex"], p0["assess_unit"]
    y = {0: p0["y_income"], 1: p0["y_occ"]}
    const = {j: int(p0["const_class"][j]) for j in (0, 1)}
    boot = OwnBoot(units) if boot is None else boot
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

    res_ = dict(EL["resolved"])
    if "Q" not in res_ and "Q*" in res_:                 # lock vocabulary: the fixed DIRECT-TASK i8o64 confidence code
        res_["Q"] = res_["Q*"]
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
            slots.append({"id": f"P{i_:02d}", "claim": "Q", "kind": kind, "task": j, "nominee": "Q", "target": tgt,
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
        lo, hi = pt - z * se, pt + z * se
        dec = "INVALID" if nonfin or not np.isfinite(pt) else \
            (("PASS" if lo > e["target"] else "NOT_ESTABLISHED") if e["side"] == "lower>" else
             ("PASS" if hi < e["target"] else "NOT_ESTABLISHED"))
        rl = [e["nominee"]] + ([e["ref"]] if e.get("ref") else [])
        descriptive = any((EL["statuses"].get(x) or EL["statuses"].get(x + "*") or {}).get("status") != "NOMINEE"
                          for x in rl)
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


CONTROL_SEED, SPLIT_SEED, NULL_Z, PLANT_MIN = 20261021, 20261022, 3.5, 0.75      # source limits (to reconcile with cbp)
CONTROL_FILES = ("AUDIT_PRELOCK_CHECKS.json", "CONTROLS.json")


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
    cf = next((RES / n_ for n_ in CONTROL_FILES if (RES / n_).exists()), None)
    if cf is None:
        return res("PENDING", reason="the controls verdict file is not yet written")
    a = jload(cf)
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
           "verdict_file_sha256_equals_lock_pin": (sha_file(cf) == (EL.get("technical_validity") or {}).get(
               "controls_verdict_sha256")) if EL else None,
           "verdict_file_sha256": sha_file(cf),
           "decoded_probability_only_audits_empty": vac, "failures": bad[:20],
           "not_replayed": "the held-out AUCs come from attacker fits on planted / permuted labels that this check does "
                           "not refit; the split, permutation, thresholds and every pass rule are own"}
    st = "FAIL" if bad or nc_bad or not out["verdict_equals_own"] or out["verdict_file_sha256_equals_lock_pin"] is False else \
        ("WARN" if vac else "PASS")
    if vac:
        out["reason"] = (f"{len(vac)} COLL checks record decoded_probability_only_misses_it = true with an EMPTY decoded-"
                         f"probability-only audit: the stated complement is not evidenced (pass rules unaffected)")
    return res(st, **out)


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


def pred_in_table(text, p_):
    line = next((l_ for l_ in text.splitlines() if l_.startswith(f"| {p_['id']} |")), "")
    pr = p_.get("probability", p_.get("probabilities"))
    if isinstance(pr, (int, float)):
        return f"| {pr} |" in line or f"| {pr:.2f} |" in line
    if isinstance(pr, dict):
        return all((f"{k_} {v:.2f}" in line) for k_, v in pr.items() if isinstance(v, (int, float)) and k_ in line)
    return False


def _dec(sh):
    t = sh.replace("−", "-").replace("+", "").replace(",", "")
    return t, (len(t.split(".")[1]) if "." in t else 0)


def frag_check(text, fragment, pairs):
    """fragment must occur verbatim; every (shown, own) pair must agree at the printed precision."""
    if fragment not in text:
        return {"fragment": fragment[:120], "present": False, "ok": False}
    bad = []
    for shown, own in pairs:
        if isinstance(own, (bool, np.bool_)):
            if not bool(own):
                bad.append(f"{shown}: own condition false")
            continue
        if shown not in fragment:
            bad.append(f"{shown}: not in fragment")
            continue
        t, dd = _dec(shown)
        if abs(float(t) - float(own)) > 0.5 * 10 ** (-dd) + 1e-12:
            bad.append(f"{shown}: own {own:.{dd + 2}f}")
    return {"fragment": fragment[:120], "present": True, "ok": not bad, "slips": bad}


# ------------------------------------------------------------------------------------------------ PHASE 3 (cbp)
def scored_list_rule(sel_mine):
    """PROTOCOL section 12 scored list from the own selection: P*, J*, T*, C_rate, C_global, Q (or their registered
    fallbacks); U continuous and CLASS-ONLY; source JOINT / SEQ-12 / SEQ-21 at lambda 0.1; the T*-guarded ordinary
    privacy winner without headroom; each family's headroom winner or fixed fallback; RAW-J, FARE, F0 and LEACE.
    Exact aliases are scored once (role mappings are kept by the lock)."""
    r = sel_mine["resolved"]
    d = sel_mine["diagnostics"]
    want = {v for v in r.values() if v}
    want |= {"SRC|U", b_config("CLASS"), b_config("JOINT", 0.1), b_config("SEQ-12", 0.1), b_config("SEQ-21", 0.1),
             "SRC|RAW-J_b0.3", "REF|F", "REF|F0", "REF|E"}
    nh = d["ordinary_privacy_winner_no_headroom"]
    if nh.get("config") or nh.get("descriptive_config"):
        want.add(nh.get("config") or nh.get("descriptive_config"))
    for v in d["family_headroom_winners"].values():
        c_ = v.get("config") or v.get("descriptive_config")
        if c_:
            want.add(c_)
    return sorted(want)


def composed_winners_own(own_in):
    out = {}
    for k in SEEDS:
        o = own_in.get((k, "SRC|U")) or {}
        fr = set()
        for f_ in (o.get("composed") or {}).values():
            for crit in ("winner", "ce_winner"):
                fr |= {v for v in f_[crit].values() if v != "source"}
        out[k] = sorted(fr)
    return out


def check_eval_lock(D: Data, L, gate, sel_mine, own_in):
    """EVALUATION_LOCK content vs the own selection and scored-list rule, the composed-source winners (every composed
    reader selected as a U source winner retained), unit file hashes, assessment groups, inference settings, the lock /
    selection / controls / endpoint-parity hashes and the fitting SEX prior."""
    EL = jload(RES / "EVALUATION_LOCK.json")
    out, bad = {"gate": {k_: gate.get(k_) for k_ in ("ok", "commit", "first_push_time", "commits", "sha256")}}, []
    out["resolved_equal_own"] = (EL.get("resolved") or {}) == sel_mine["resolved"]
    st_l = EL.get("statuses") or {}
    out["statuses_equal_own"] = all((st_l.get(x) or {}).get(f_) == v.get(f_) for x, v in sel_mine["statuses"].items()
                                    for f_ in ("status", "config", "descriptive_config", "reason"))
    sl = EL.get("scored_labels") or []
    out["scored_labels"] = len(sl)
    out["scored_labels_equal_own_rule"] = sorted(sl) == scored_list_rule(sel_mine) and len(set(sl)) == len(sl)
    cw = composed_winners_own(own_in)
    miss = {}
    for k in SEEDS:
        se = (EL.get("seeds") or {}).get(str(k)) or {}
        locked = set(se.get("composed_policies") or [])
        want_units = {release_unit(k, c_) for c_ in cw[k]}
        if want_units - locked:
            miss[f"s{k}"] = sorted(want_units - locked)
        out.setdefault("composed_policies_closure_38", {})[f"s{k}"] = sorted(locked) == sorted(
            release_unit(k, c_) for c_ in composition_cids())
        out.setdefault("score_units_equal_scored_labels", {})[f"s{k}"] = sorted((se.get("score") or {})) == sorted(sl) and \
            all(v.get("unit") == release_unit(k, lab) for lab, v in (se.get("score") or {}).items())
    out["composed_source_winners_own"] = {f"s{k}": v for k, v in cw.items()}
    out["composed_source_winners_missing_from_lock"] = miss
    am = D.mask[ASSESS]
    ar = EL.get("assessment_role") or {}
    out["assessment_groups_ok"] = ar.get("rows") == int(am.sum()) and ar.get("groups") == len(np.unique(D.unit[am])) and \
        ar.get("row_id_sha256") == rowid_hash(D.row_id[am])
    ep = EL.get("endpoints") or {}
    out["inference_settings_ok"] = ep.get("z") == Z_PRIMARY and ep.get("B") == B_BOOT and ep.get("boot_seed") == BOOT_SEED \
        and ep.get("size") == N_ENDPOINTS and ep.get("primary") == [f"P{i:02d}" for i in range(1, 38)]
    out["locks_sha256_ok"] = all((RES / n_).exists() and sha_file(RES / n_) == h for n_, h in (EL.get("locks_sha256") or
                                                                                            {}).items()) and \
        sorted(EL.get("locks_sha256") or {}) == ["AUDIT_AND_SELECTION_LOCK.json", "FIT_LOCK.json", "SOURCE_ADMISSION_LOCK.json"]
    out["amendments_sha256_ok"] = sorted((EL.get("amendments_sha256") or {})) == sorted(q.name for q in RES.glob("AMENDMENT*.json"))
    out["selection_sha256_ok"] = EL.get("selection_sha256") == sha_file(RUN / "selection.json") and \
        EL.get("selection_public_sha256") == sha_file(RES / "SELECTION.json")
    tv = EL.get("technical_validity") or {}
    out["technical_validity_ok"] = tv.get("ok") is True and tv.get("controls_verdict_sha256") == \
        sha_file(RES / "AUDIT_PRELOCK_CHECKS.json") and (tv.get("endpoint_parity") or {}).get("sha256") == \
        sha_file(RUN / "endpoint_parity.json")
    fit = D.fit_idx
    out["sex_prior_hash_ok"] = hashlib.sha256(np.bincount(L["sex"][fit], minlength=2).astype(np.int64).tobytes()
                                              ).hexdigest() == EL.get("sex_prior_defense_fit_sha256")
    out["alias_of_by_role_ok"] = (EL.get("alias_of_by_role") or {}) == ({f"P{i:02d}": f"P{i - 11:02d}" for i in range(12, 23)}
                                                                      if sel_mine["resolved"]["C_rate"] ==
                                                                      sel_mine["resolved"]["C_global"] else {})
    lc = EL.get("locked_code_files") or {}
    out["locked_code_changed_in_worktree"] = [f_ for f_, h in lc.items() if not ((WT / f_).exists() and sha_file(WT / f_) == h)]
    files_ok, n_units = True, 0
    for se in (EL.get("seeds") or {}).values():
        for u_, h in (se.get("unit_file_sha256") or {}).items():
            n_units += 1
            cpl = UNITS / u_ / "COMPLETE.json"
            if not cpl.exists() or h != jload(cpl)["files"] or not all(sha_file(UNITS / u_ / f_) == hh for f_, hh in h.items()):
                files_ok = False
    out.update({"unit_file_hashes_ok": files_ok, "units_hashed": n_units})
    for kk, v in out.items():
        if isinstance(v, bool) and not v:
            bad.append(kk)
        if isinstance(v, dict) and kk in ("composed_policies_closure_38", "score_units_equal_scored_labels") and \
                not all(v.values()):
            bad.append(kk)
    if miss:
        bad.append("composed-source winner missing from the lock")
    if out["locked_code_changed_in_worktree"]:
        bad.append("locked code changed in worktree")
    if not gate.get("ok"):
        bad.append("gate")
    return res("FAIL" if bad else "PASS", failures=bad, **out), EL


def check_endpoints_cbp(preds, EL, technical_valid=True):
    """Own 37 endpoints (points, SEs, bounds), own clause outcomes and decisions, own claim / Q statuses and the own label
    (LABEL_TRUTH_TABLE.json) vs the lead's inference record, PRIMARY_ENDPOINTS.csv and LABEL_RESULT.json, plus every
    reported level (points and SEs)."""
    eps, lev, boot = own_endpoints(preds, EL)
    st = dict(EL.get("statuses") or {})
    for e in eps:
        e["outcome"], e["point_side"] = clause_outcome(e)
        e["decision_own"] = "DESCRIPTIVE_ONLY" if e["decision"] == "DESCRIPTIVE_ONLY" else e["outcome"]
    lab = labels_cbp(st, eps, technical_valid)
    out = {"own_label": {k_: lab[k_] for k_ in ("label", "claims", "root_causes", "q", "q_root_cause",
                                                 "incomplete_displayed")},
           "own_failing_clauses": {c_: sorted(e["id"] for e in eps if e["claim"] == c_ and e["outcome"] != "PASS")
                                   for c_ in ("A", "B", "C", "Q")},
           "endpoints": [{k_: e.get(k_) for k_ in ("id", "claim", "kind", "nominee_config", "ref_config", "point", "se",
                                                    "lower", "upper", "target", "side", "outcome", "decision_own")}
                         for e in eps],
           "bootstrap": {"B": boot.B, "groups": boot.G, "rows": boot.n, "seed": BOOT_SEED, "z": Z_PRIMARY,
                         "counts_sha256": hashlib.sha256(boot.counts.tobytes()).hexdigest()}}
    inf_p = RUN / "inference.json"
    if not inf_p.exists():
        return res("PENDING", reason="inference.json not yet written; own endpoints computed", **out), eps, lev
    inf = jload(inf_p)
    rec = {e["id"]: e for e in inf.get("primary", [])}
    mx = {"point": 0.0, "se": 0.0, "bounds": 0.0}
    dec_bad = []
    for e in eps:
        r_ = rec.get(e["id"], {})
        if r_.get("point") is None:
            dec_bad.append(f"{e['id']}: no recorded point")
            continue
        mx["point"] = max(mx["point"], abs(e["point"] - r_["point"]))
        mx["se"] = max(mx["se"], abs(e["se"] - r_["se"]))
        mx["bounds"] = max(mx["bounds"], abs(e["lower"] - r_["lower"]), abs(e["upper"] - r_["upper"]))
        if e["outcome"] != r_.get("outcome") or e["decision_own"] != r_.get("decision"):
            dec_bad.append(f"{e['id']}: {e['outcome']}/{e['decision_own']} vs {r_.get('outcome')}/{r_.get('decision')}")
        if e["target"] != r_.get("target") or e["side"] != r_.get("side"):
            dec_bad.append(f"{e['id']}: slot definition")
    cs = inf.get("claim_status") or {}
    claims_eq = all((cs.get(c_) or {}).get("status") == lab["claims"][c_] and
                    (cs.get(c_) or {}).get("root_cause") == lab["root_causes"][c_] and
                    sorted((cs.get(c_) or {}).get("failing") or []) == (out["own_failing_clauses"][c_]
                                                                        if lab["claims"][c_] == "NOT_ESTABLISHED" else [])
                    for c_ in ("A", "B", "C"))
    q_eq = (inf.get("q_status") or {}).get("status") == lab["q"] and (inf.get("q_status") or {}).get("root_cause") == \
        lab["q_root_cause"]
    lr = jload(RES / "LABEL_RESULT.json") if (RES / "LABEL_RESULT.json").exists() else {}
    lv_mx, lv_se_mx, lv_missing = 0.0, 0.0, []
    rl = inf.get("levels", {})
    for nm, (pt, rp) in lev.items():
        r_ = rl.get(nm.replace("#None#", "#primary#")) or rl.get(nm)
        if r_ is None:
            lv_missing.append(nm)
            continue
        lv_mx = max(lv_mx, abs(pt - r_["point"]))
        lv_se_mx = max(lv_se_mx, abs(float(np.std(rp[np.isfinite(rp)], ddof=1)) - r_["se"]))
    extra = sorted(set(rl) - {n_.replace("#None#", "#primary#") for n_ in lev})
    csv_bad = []
    pe = RES / "PRIMARY_ENDPOINTS.csv"
    if pe.exists():
        import csv
        pub = {r_["id"]: r_ for r_ in csv.DictReader(pe.open())}
        for e in eps:
            c_ = pub.get(e["id"])
            if c_ is None or c_["outcome"] != e["outcome"] or c_["decision"] != e["decision_own"] or \
                    any(abs(float(c_[f_]) - e[f_]) > 5e-7 + 1e-12 for f_ in ("point", "se", "lower", "upper")):
                csv_bad.append(e["id"])
    out.update({"max_abs_diff": mx, "decision_mismatches": dec_bad, "recorded_label": inf.get("label"),
                "label_equal": inf.get("label") == lab["label"], "claim_statuses_equal": claims_eq,
                "q_status_equal": q_eq, "technical_valid_recorded": inf.get("technical_valid"),
                "winning_family_recorded": inf.get("winning_family"),
                "label_result_json_consistent": lr.get("label") == lab["label"] and
                (lr.get("displayed_statuses") or {}) == {**lab["claims"], "Q": lab["q"]},
                "levels_checked": len(lev), "levels_point_max_abs_diff": lv_mx, "levels_se_max_abs_diff": lv_se_mx,
                "levels_missing_in_record": lv_missing[:10], "levels_only_in_record": extra[:10],
                "primary_endpoints_csv_mismatches": csv_bad,
                "inference_settings_recorded_ok": inf.get("B") == B_BOOT and inf.get("seed") == BOOT_SEED and
                inf.get("z") == Z_PRIMARY and inf.get("n_groups") == boot.G and inf.get("n_assessment") == boot.n})
    ok = (not dec_bad and out["label_equal"] and claims_eq and q_eq and out["label_result_json_consistent"] and
          mx["point"] <= 1e-12 and mx["se"] <= 1e-12 and mx["bounds"] <= 1e-11 and lv_mx <= 1e-12 and lv_se_mx <= 1e-12
          and not lv_missing and not extra and not csv_bad and out["inference_settings_recorded_ok"] and
          inf.get("technical_valid") is True)
    return res("PASS" if ok else "FAIL", **out), eps, lev


def check_tables_cbp(own_in, sel_mine, sel_rows, eps, lev, EL):
    """Public tables vs own numbers (6-decimal prints: |diff| <= 5e-7; flags and IDs exact): ALL_LEVELS.csv,
    ASSESSMENT_COMPARISON.csv, INNER_SELECTION_TABLE.csv, HEADROOM_VS_STANDARD_SELECTION.csv,
    INNER_STATES_VS_RECOVERY.csv. These hold the values plotted in figures 1-4."""
    import csv
    out, bad = {}, []
    tol = 5e-7 + 1e-12

    def cmp(tag, shown, own_v):
        if shown in ("", None):
            if own_v is None:
                return
            bad.append(f"{tag}: blank vs own {own_v}")
            return
        out.setdefault(tag, [0, 0.0])
        out[tag][0] += 1
        d_ = abs(float(shown) - float(own_v))
        out[tag][1] = max(out[tag][1], d_)
        if d_ > tol:
            bad.append(f"{tag}: {shown} vs own {own_v:.7f}")

    def flag(tag, shown, own_v):
        out.setdefault(tag + "_flags", [0, 0])
        out[tag + "_flags"][0] += 1
        if str(shown) != str(own_v):
            out[tag + "_flags"][1] += 1
            bad.append(f"{tag}: {shown} vs own {own_v}")

    def se(nm):
        rp = lev[nm][1]
        return float(np.std(rp[np.isfinite(rp)], ddof=1))
    f_ = RES / "ALL_LEVELS.csv"
    n_all = 0
    if f_.exists():
        for r_ in csv.DictReader(f_.open()):
            q_, k_, lab_, det = r_["quantity"], r_["seed"], r_["label"], r_["detail"]
            if q_ in ("R", "Rmean"):
                fam, w = det.split("|")
                nm = f"R#{k_}#{lab_}#{fam}#{w}" if q_ == "R" else f"Rmean#{lab_}#{fam}#{w}"
                nm2 = nm.replace("#primary#", "#None#")
                key = nm if nm in lev else nm2
            elif q_ == "const":
                key = f"const#{det}"
            elif q_.endswith("mean"):
                key = f"{q_}#{lab_}#{det}"
            else:
                key = f"{q_}#{k_}#{lab_}#{det}"
            if key not in lev:
                bad.append(f"ALL_LEVELS row without own level: {q_},{k_},{lab_},{det}")
                continue
            n_all += 1
            cmp("all_levels_point", r_["point"], lev[key][0])
            cmp("all_levels_se", r_["se"], se(key))
    out["all_levels_rows"] = n_all
    f_ = RES / "ASSESSMENT_COMPARISON.csv"
    n_ac = 0
    if f_.exists():
        for r_ in csv.DictReader(f_.open()):
            lab_ = r_["release"]
            n_ac += 1
            for w in VIEWS:
                nm = f"Rmean#{lab_}#None#{w}" if f"Rmean#{lab_}#None#{w}" in lev else f"Rmean#{lab_}#primary#{w}"
                cmp("comparison_auc", r_[f"auc_{w}"], lev[nm][0])
                cmp("comparison_auc_se", r_[f"auc_{w}_se"], se(nm))
            for j, t_ in ((0, "income"), (1, "occupation")):
                cmp("comparison_acc", r_[f"acc_{t_}"], lev[f"accmean#{lab_}#{j}"][0])
                if lab_ != "SRC|U":
                    for kk, col in (("llx", "ll_excess"), ("brx", "brier_excess")):
                        cmp("comparison_excess", r_[f"{col}_{t_}"], lev[f"{kk}mean#{lab_}#{j}"][0])
                        cmp("comparison_excess_se", r_[f"{col}_{t_}_se"], se(f"{kk}mean#{lab_}#{j}"))
            if lab_ in sel_rows and sel_rows[lab_].get("valid"):
                flag("comparison_inner_flags", r_["inner_ordinary"], sel_rows[lab_]["ordinary_eligible"])
                flag("comparison_inner_flags", r_["inner_headroom"], sel_rows[lab_]["headroom_eligible"])
    out["assessment_comparison_rows"] = n_ac
    if n_ac != len(EL.get("scored_labels") or []):
        bad.append(f"ASSESSMENT_COMPARISON rows {n_ac} vs scored labels {len(EL.get('scored_labels') or [])}")
    f_ = RES / "INNER_SELECTION_TABLE.csv"
    n_is = 0
    cw = {k: (own_in.get((k, "SRC|U")) or {}).get("composed", {}) for k in SEEDS}
    if f_.exists():
        for r_ in csv.DictReader(f_.open()):
            cid, k = r_["config"], int(r_["seed"])
            m = sel_rows.get(cid)
            if m is None or not m.get("valid"):
                bad.append(f"inner table {cid}#s{k}: no valid own row")
                continue
            n_is += 1
            s_ = m["seeds"][k]
            o = own_in[(k, cid)]
            for w in VIEWS:
                cmp("inner_table_auc", r_[f"auc_{w}"], s_["auc"][w])
            for i, t_ in ((1, "income"), (2, "occupation")):
                cmp("inner_table_utility", r_[f"ll_{t_}"], o["utility"][i]["logloss"])
                cmp("inner_table_utility", r_[f"brier_{t_}"], o["utility"][i]["brier"])
                cmp("inner_table_excess", r_[f"ll_excess_{t_}"], s_["tasks"][i]["excess"]["logloss"])
                cmp("inner_table_excess", r_[f"brier_excess_{t_}"], s_["tasks"][i]["excess"]["brier"])
            flag("inner_table", r_["ordinary_seed"], s_["ordinary_ok"])
            flag("inner_table", r_["headroom_seed"], s_["ordinary_ok"] and s_["headroom_ok"])
            cmp("inner_table_shortfall", r_["ordinary_shortfall_seed"],
                max(0.0, max(s_["tasks"][i]["shortfall_ordinary_raw"] for i in (1, 2))))
            cmp("inner_table_shortfall", r_["headroom_shortfall_seed"],
                max(0.0, max(s_["tasks"][i]["shortfall_headroom_raw"] for i in (1, 2))))
            ts = None if o["states"] is None else int(o["states"])
            flag("inner_table_states", r_["token_states"] or None, ts)
            if cid == "SRC|U":
                win = ((cw[k].get(o["primary"]) or {}).get("winner") or {}).get("pair")
                flag("inner_table_composed_winner", r_["composed_winner_pair"] or None, win)
    out["inner_selection_rows"] = n_is
    if n_is != 3 * len(all_cids()):
        bad.append(f"INNER_SELECTION_TABLE rows {n_is}")
    f_ = RES / "HEADROOM_VS_STANDARD_SELECTION.csv"
    d = sel_mine["diagnostics"]
    stt = sel_mine["statuses"]
    roles = {"P* (headroom)": stt["P*"], "ordinary privacy winner": d["ordinary_privacy_winner_no_headroom"],
             "strongest ordinary privacy (unguarded)": d["strongest_ordinary_privacy_unguarded"],
             "J* (headroom)": stt["J*"], **{f"{f_} headroom winner": v for f_, v in d["family_headroom_winners"].items()}}
    n_h = 0
    if f_.exists():
        for r_ in csv.DictReader(f_.open()):
            sel_, cfg = r_["selection"], r_["config"]
            n_h += 1
            if sel_ in roles:
                v = roles[sel_]
                flag("headroom_table_role", r_["status"], v["status"])
                flag("headroom_table_role", cfg, v.get("config") or v.get("descriptive_config"))
                flag("headroom_table_role", r_["reason"] or None, v.get("reason"))
            if cfg in sel_rows and sel_rows[cfg].get("valid"):
                m = sel_rows[cfg]
                for col in ("mean_pair", "mean_v1", "mean_v2"):
                    cmp("headroom_table_means", r_[col], m[col])
                cmp("headroom_table_worst", r_["worst_ll_excess"], max(m["seeds"][k]["tasks"][i]["excess"]["logloss"]
                                                                       for k in SEEDS for i in (1, 2)))
                cmp("headroom_table_worst", r_["worst_brier_excess"], max(m["seeds"][k]["tasks"][i]["excess"]["brier"]
                                                                          for k in SEEDS for i in (1, 2)))
                flag("headroom_table_flags", r_["ordinary"], m["ordinary_eligible"])
                flag("headroom_table_flags", r_["headroom"], m["headroom_eligible"])
            if sel_ == "headroom changes winner":
                hc = d["headroom_changes_winner"]
                flag("headroom_table_change", r_["changed"], hc.get("changed"))
                cmp("headroom_table_give_up", r_["give_up_mean"], hc.get("give_up_mean_pair_auc"))
                for k in SEEDS:
                    cmp("headroom_table_give_up", r_[f"give_up_s{k}"], hc["give_up_per_seed"][f"s{k}"])
    out["headroom_table_rows"] = n_h
    f_ = RES / "INNER_STATES_VS_RECOVERY.csv"
    n_st = 0
    if f_.exists():
        for r_ in csv.DictReader(f_.open()):
            cid = r_["config"]
            if cid not in sel_rows:
                bad.append(f"states table {cid}: unknown")
                continue
            n_st += 1
            em = [own_in[(k, cid)].get("emitted") for k in SEEDS]
            if all(x is not None for x in em):
                cmp("states_table_occupied", r_["occupied_states_fit_mean"], seed_mean(em))
            cmp("states_table_pair_auc", r_["inner_pair_auc"], sel_rows[cid]["mean_pair"])
    out["states_table_rows"] = n_st
    fig = sorted(q.name for q in (RES / "figures").glob("*")) if (RES / "figures").exists() else []
    out["figures_present"] = fig
    return res("FAIL" if bad else "PASS", cells_checked_and_max_abs_diff=out, mismatches=bad[:30],
               n_mismatches=len(bad), note="figures 1-4 are rendered from these tables (their source values are the "
                                           "checked cells); the images themselves are not parsed")


def check_deployment(D: Data, T):
    """Own deployment of the packaged codes (Q, P*, the J* fallback; seed 1) on the admitted 83-column deployment input:
    the input equals the own X; own forward pass of the admitted U teacher; own policy application; bitwise vs the
    stored releases and vs the lead's cbp.deploy outputs (exactly six arrays). Then the cbp.deploy CLI is exercised as a
    BLACK BOX in a child process (never imported): one fresh deployment compared bitwise with the own application, and
    every registered refusal (exit code 2, no output written)."""
    tp = RUN / "closeout_targets.json"
    tg = jload(tp) if tp.exists() else {}
    units = [u_ for u_ in (tg.get("policies") or {}).values() if u_.startswith("pol__")]
    dt = PRIV / "deploy_test"
    if not units:
        return res("PENDING", reason="packaged codes not yet named")
    zi = np.load(PRIV / "inputs" / "deploy_input.npz", allow_pickle=False)
    names = [str(x) for x in zi["feature_names"]]
    out = {"input_X_bitwise_equals_own_X": bitwise(np.asarray(zi["X"]), D.X), "input_names_equal_pinned": names ==
           D.feature_names, "input_arrays": sorted(zi.files),
           "schema_json_equals_pinned": jload(PRIV / "inputs" / "schema.json") == D.feature_names}
    bad, per = [], {}
    k = int(tg.get("seed", 1))
    ud = ADM / f"rel__s{k}__U"
    mine = own_teacher(ud / "model.pt", [ud / "head_0.joblib", ud / "head_1.joblib"], np.asarray(zi["X"]))
    P, d = {1: mine["p1"], 2: mine["p2"]}, {1: mine["d1"], 2: mine["d2"]}
    out["teacher_forward_bitwise"] = all(bitwise(mine[x].astype(T[(k, "U")][x].dtype), T[(k, "U")][x])
                                         for x in ("p1", "p2", "d1", "d2"))
    own_rel = {}
    for un in units:
        rel = policy_pair_release(UNITS / un / "policy.json", P, d)
        own_rel[un] = rel
        st_ = np.load(UNITS / un / "release.npz", allow_pickle=False)
        r_ = {"release_bitwise_vs_stored": all(bitwise(rel[f"{a}{r}"], st_[f"{a}{r}"]) for a in ("tok", "q", "hard")
                                               for r in (1, 2)),
              "decisions_equal_teacher": all(bool(np.array_equal(rel[f"hard{r}"], d[r])) for r in (1, 2))}
        dp = dt / f"{un.split('__', 2)[2]}.npz"
        if dp.exists():
            dz = np.load(dp, allow_pickle=False)
            r_["lead_deploy_output_exactly_six_arrays"] = sorted(dz.files) == sorted(
                [f"{a}_{r}" for a in ("tokens", "probs", "decision") for r in (1, 2)])
            r_["lead_deploy_output_bitwise_vs_own"] = all(
                bitwise(dz[f"tokens_{r}"], rel[f"tok{r}"]) and bitwise(dz[f"probs_{r}"], rel[f"q{r}"]) and
                bitwise(dz[f"decision_{r}"], rel[f"hard{r}"]) for r in (1, 2))
        per[un] = r_
        if not all(v for v in r_.values()):
            bad.append(un)
    out["per_unit"] = per
    out["lead_deploy_outputs_present"] = sorted(q.name for q in dt.glob("*.npz")) if dt.exists() else []
    out["black_box_cli"] = deploy_cli_black_box(D, tg, own_rel, k)
    if out["black_box_cli"]["status"] != "PASS":
        bad.append("deploy CLI black-box checks")
    ok = not bad and all(v for v in out.values() if isinstance(v, bool))
    return res("PASS" if ok else "FAIL", **out, failures=bad)


def deploy_cli_black_box(D: Data, tg, own_rel, k):
    """Run `python -m cbp.deploy` as an external process (the verifier never imports cbp). Inputs are written to a
    private scratch folder and removed afterwards; only aggregate verdicts are reported."""
    import shutil as _sh
    import tempfile
    py = sys.executable
    env = {**os.environ, "PYTHONPATH": str(WT), "OMP_NUM_THREADS": "1"}
    pol = [u_ for u_ in (tg.get("policies") or {}).values() if "JOINT" in u_ and "l0.01" in u_] or \
        [u_ for u_ in (tg.get("policies") or {}).values() if u_.startswith("pol__")]
    un = pol[0]
    zi = np.load(PRIV / "inputs" / "deploy_input.npz", allow_pickle=False)
    X, names = np.asarray(zi["X"]), [str(x) for x in zi["feature_names"]]
    tmp = Path(tempfile.mkdtemp(prefix="cbp_F_deploy_"))
    res_ = {}
    try:
        def write_x(nm, Xv, nv, extra=None):
            p = tmp / nm
            arrs = {"X": Xv, "feature_names": np.asarray(nv)}
            if extra:
                arrs.update(extra)
            np.savez(p, **arrs)
            return p
        good = write_x("x.npz", X, names)
        perm = list(range(len(names)))
        perm[0], perm[1] = perm[1], perm[0]
        cases = {"reordered_columns": write_x("xr.npz", X[:, perm], [names[i] for i in perm]),
                 "84_columns": write_x("x84.npz", np.hstack([X, X[:, :1]]), names + ["extra_col"]),
                 "82_columns": write_x("x82.npz", X[:, :-1], names[:-1]),
                 "extra_input_array": write_x("xe.npz", X, names, {"sex": np.zeros(len(X), dtype=np.int64)})}

        def run(args, out_name):
            outp = tmp / out_name
            cmd = [py, "-m", "cbp.deploy", "--unit", str(ADM / f"rel__s{k}__U"), "--policy",
                   str(UNITS / un / "policy.json"), "--X", str(good), "--schema", str(PRIV / "inputs" / "schema.json"),
                   "--out", str(outp), "--seed", str(k)] + args
            r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(tmp), timeout=600)
            return r.returncode, outp

        rc, outp = run([], "ok.npz")
        okz = np.load(outp, allow_pickle=False) if outp.exists() else None
        rel = own_rel[un]
        res_["fresh_deploy"] = {"rc": rc, "exactly_six_arrays": okz is not None and sorted(okz.files) == sorted(
            [f"{a}_{r}" for a in ("tokens", "probs", "decision") for r in (1, 2)]),
            "bitwise_vs_own": okz is not None and all(bitwise(okz[f"tokens_{r}"], rel[f"tok{r}"]) and
                                                      bitwise(okz[f"probs_{r}"], rel[f"q{r}"]) and
                                                      bitwise(okz[f"decision_{r}"], rel[f"hard{r}"]) for r in (1, 2))}
        refusals = {}
        for fl in ("--include-sex", "--export-fine-ids", "--raw-scores", "--debug", "--foo"):
            rc, outp = run([fl], f"r{fl.strip('-')}.npz")
            refusals[f"flag {fl}"] = {"rc": rc, "no_output": not outp.exists()}
        for nm, p in cases.items():
            cmd_rc, outp = None, tmp / f"{nm}.npz"
            r = subprocess.run([py, "-m", "cbp.deploy", "--unit", str(ADM / f"rel__s{k}__U"), "--policy",
                                str(UNITS / un / "policy.json"), "--X", str(p), "--schema",
                                str(PRIV / "inputs" / "schema.json"), "--out", str(outp), "--seed", str(k)],
                               capture_output=True, text=True, env=env, cwd=str(tmp), timeout=600)
            refusals[nm] = {"rc": r.returncode, "no_output": not outp.exists()}
        other = 0 if k != 0 else 1
        outp = tmp / "mismatch.npz"
        r = subprocess.run([py, "-m", "cbp.deploy", "--unit", str(ADM / f"rel__s{other}__U"), "--policy",
                            str(UNITS / un / "policy.json"), "--X", str(good), "--schema",
                            str(PRIV / "inputs" / "schema.json"), "--out", str(outp), "--seed", str(other)],
                           capture_output=True, text=True, env=env, cwd=str(tmp), timeout=600)
        refusals["mismatched_teacher"] = {"rc": r.returncode, "no_output": not outp.exists()}
        outp = tmp / "unreg.npz"
        r = subprocess.run([py, "-m", "cbp.deploy", "--unit", str(ADM / f"rel__s{k}__U"), "--policy",
                            str(UNITS / release_unit(k, b_config("JOINT", COMPOSED_EXTRA_LAM)) / "policy.json"), "--X",
                            str(good), "--schema", str(PRIV / "inputs" / "schema.json"), "--out", str(outp), "--seed",
                            str(k)], capture_output=True, text=True, env=env, cwd=str(tmp), timeout=600)
        refusals["unregistered_lambda_1_policy"] = {"rc": r.returncode, "no_output": not outp.exists()}
        res_["refusals"] = refusals
    finally:
        _sh.rmtree(tmp, ignore_errors=True)
    ok = res_["fresh_deploy"]["rc"] == 0 and res_["fresh_deploy"]["exactly_six_arrays"] and \
        res_["fresh_deploy"]["bitwise_vs_own"] and all(v["rc"] == 2 and v["no_output"] for v in res_["refusals"].values())
    return {"status": "PASS" if ok else "FAIL", "unit": un, **res_,
            "note": "the CLI is executed as an external process; the verifier imports no cbp module"}


def check_budget():
    """Budget and process counts from SEMA_LOG (the single resource ledger): holds and CPU per role, peak concurrent
    holds (<= 2), nonzero exits, signals, unreleased holds, total measured CPU vs the 20 CPU-h ceiling and the 4 CPU-h
    reserve, elapsed time vs the 10 h ceiling."""
    sema = jsonl(RUN / "SEMA_LOG.jsonl")
    holds, open_, unreleased = [], {}, []
    for e in sema:
        key = (e.get("wrapper_pid"), e.get("slot"))
        if e.get("event") == "acquire":
            for k2 in [k2 for k2 in open_ if k2[1] == e.get("slot") and k2 != key]:
                t0_, lab_ = open_.pop(k2)
                unreleased.append({"label": lab_, "slot": k2[1], "acquired": iso(t0_), "slot_reacquired_at": e["at"]})
                holds.append((t0_, parse_iso(e["at"]), lab_, None, None))
            open_[key] = (parse_iso(e["at"]), e.get("label"))
        elif e.get("event") == "release" and key in open_:
            t0_, lab_ = open_.pop(key)
            holds.append((t0_, parse_iso(e["at"]), lab_, e.get("cpu_s"), e.get("rc")))
    still_open = [{"label": lab_, "slot": k_[1], "acquired": iso(t0_)} for k_, (t0_, lab_) in open_.items()]
    holds += [(t0_, None, lab_, None, None) for (t0_, lab_) in open_.values()]
    pts = sorted([(h[0], 1) for h in holds] + [(h[1], -1) for h in holds if h[1] is not None], key=lambda x: (x[0], x[1]))
    cur, peak = 0, 0
    for _, dlt in pts:
        cur += dlt
        peak = max(peak, cur)
    by_role = {}
    for h in holds:
        r_ = str(h[2]).split(":")[0]
        b = by_role.setdefault(r_, {"holds": 0, "cpu_s": 0.0, "wall_s": 0.0, "nonzero_rc": 0})
        b["holds"] += 1
        b["cpu_s"] += float(h[3] or 0.0)
        if h[1] is not None:
            b["wall_s"] += (h[1] - h[0]).total_seconds()
        b["nonzero_rc"] += int(h[4] not in (0, None))
    tot = sum(b["cpu_s"] for b in by_role.values())
    signals = [e for e in sema if e.get("event") == "signal"]
    start = parse_iso(STUDY_START)
    now_ = datetime.now(timezone.utc)
    out = {"holds": len(holds), "peak_concurrent_holds": peak, "by_role": by_role, "measured_cpu_h": tot / 3600,
           "cpu_ceiling_h": 20, "reserve_cpu_h": 4, "elapsed_h_at_check": (now_ - start).total_seconds() / 3600,
           "unreleased_holds_closed_by_reacquire": unreleased, "open_holds_now": still_open, "signals": len(signals),
           "verifier_holds": by_role.get("F", {}).get("holds", 0), "verifier_cpu_s": by_role.get("F", {}).get("cpu_s", 0.0)}
    fails = []
    if peak > 2:
        fails.append(f"{peak} concurrent semaphore holds")
    if tot / 3600 > 20:
        fails.append("measured CPU above 20 CPU-h")
    st = "FAIL" if fails else ("WARN" if unreleased else "PASS")
    return res(st, failures=fails, **out, note="measured CPU = child CPU logged by the semaphore wrapper; unlogged work "
                                              "(e.g. agent reasoning) is outside this ledger and reported separately")


def check_late_chronology(gate, EL):
    """Assessment chronology: every assess / outer / infer event after the first EVALUATION_LOCK push; the lock has one
    version; no non-outer unit completed after the push; the duplicate outer units of the load-balance overlap are
    bitwise identical to the kept units and were moved (not deleted); post-lock stage starts listed."""
    ev = jsonl(RUN / "ACTIVITY_LOG.jsonl")
    push = gate.get("first_push_time")
    out, bad, warn = {}, [], []
    a_ev = [(e.get("event"), parse_iso(e["at"])) for e in ev if re.search(r"assess|unseal|outer|infer", str(e.get("event", "")))]
    out["assessment_events"] = len(a_ev)
    out["assessment_events_before_push"] = [f"{n_} at {iso(t)}" for n_, t in a_ev if push is None or t < push]
    if out["assessment_events_before_push"]:
        bad.append("assessment event before the EVALUATION_LOCK push")
    out["first_assessment_event"] = iso(min((t for _, t in a_ev), default=None)) if a_ev else None
    sema = jsonl(RUN / "SEMA_LOG.jsonl")
    holds = [(e.get("label"), parse_iso(e["at"])) for e in sema if e.get("event") == "acquire" and
             re.match(r"A:(assess|infer|outer)", str(e.get("label", "")))]
    out["assessment_semaphore_holds"] = [{"label": lab_, "acquired": iso(t)} for lab_, t in holds]
    out["assessment_holds_before_push"] = [lab_ for lab_, t in holds if push is None or t < push]
    out["first_assessment_hold_after_push_s"] = (min(t for _, t in holds) - push).total_seconds() if holds and push else None
    if out["assessment_holds_before_push"] or not holds:
        bad.append("assessment semaphore hold before the EVALUATION_LOCK push (or none recorded)")
    oc = [parse_iso(e["at"]) for e in ev if e.get("event") == "unit complete" and str(e.get("unit", "")).startswith("outer__")]
    out["outer_unit_complete_events"] = len(oc)
    if any(push is None or t < push for t in oc):
        bad.append("outer unit completion event before the push")
    late = sorted(dd.name for dd in UNITS.iterdir() if (dd / "COMPLETE.json").exists() and
                  not dd.name.startswith("outer__") and push and utc((dd / "COMPLETE.json").stat().st_mtime) > push)
    out["non_outer_units_completed_after_push"] = late
    if late:
        bad.append("units fitted after the assessment opened")
    outer = sorted(dd.name for dd in UNITS.iterdir() if dd.name.startswith("outer__"))
    out["outer_units"] = len(outer)
    early = [n_ for n_ in outer if push is None or utc((UNITS / n_ / "COMPLETE.json").stat().st_mtime) < push]
    out["outer_units_completed_before_push"] = early
    if early:
        bad.append("outer unit completed before the lock push")
    out["evaluation_lock_versions"] = gate.get("commits")
    qd = RUN / "quarantine_assess_duplicates"
    dup = []
    if qd.exists():
        for q in sorted(qd.iterdir()):
            nm = q.name.replace(".quarantined", "")
            live = UNITS / nm
            same = (q / "preds.npz").exists() and (live / "preds.npz").exists() and \
                sha_file(q / "preds.npz") == sha_file(live / "preds.npz")
            dup.append({"unit": nm, "preds_bitwise_equal_kept": same})
            if not same:
                bad.append(f"duplicate {nm} differs from the kept unit")
    out["duplicate_outer_units"] = dup
    att = RES / "ASSESSMENT_ATTEMPTS.json"
    if att.exists():
        A = jload(att)
        out["attempts_record_duplicates"] = len(A.get("duplicates") or [])
        out["attempts_record_equals_own"] = sorted(x["unit"] for x in A.get("duplicates") or []) == sorted(
            x["unit"] for x in dup) and all(x.get("bitwise_equal") for x in A.get("duplicates") or [])
        if not out["attempts_record_equals_own"]:
            bad.append("ASSESSMENT_ATTEMPTS.json disagrees with the quarantine folder")
    if dup:
        warn.append(f"{len(dup)} outer units were computed twice by overlapping load-balance workers (bitwise identical; "
                    f"first copies quarantined, not deleted)")
    post = [e for e in ev if str(e.get("event", "")).startswith("start ") and push and parse_iso(e["at"]) > push and
            e["event"].split(" ", 1)[1] not in ("assess", "outer", "infer")]
    out["post_assessment_non_assessment_stage_starts"] = [{"stage": e["event"], "at": e["at"]} for e in post]
    if post and not late:
        warn.append("a non-assessment stage was started after the assessment opened (no unit created or changed)")
    st = "FAIL" if bad else ("WARN" if warn else "PASS")
    return res(st, failures=bad, warnings=warn, reason=warn[0] if warn and not bad else None, **out)


def policy_pair_release(pj_path, P, d):
    pj = jload(pj_path)
    out = {}
    for r in (1, 2):
        pol = Pol.from_json(pj[f"p{r}"])
        tok, q, hard, _ = pol.release(P[r], d[r])
        out.update({f"tok{r}": tok, f"q{r}": q, f"hard{r}": hard, f"alpha{r}": pol.T})
    return out


def restore_from_copy(root: Path, targets: dict, Dlive: "Data"):
    """Independent restore from one copy root (folder holding SHA256SUMS, cbp_v1/ and the bundled pinned input):
    every SHA256SUMS entry re-hashed (own); the copy's input loaded by this verifier's own role reconstruction (roles
    and X equal the live ones); own forward pass of the U teacher from the copied model.pt + heads; own application of
    every copied target policy.json, bitwise vs the copied and live releases with decisions preserved; the selected
    attacker of the targets refit from the COPY (own design matrix, AUDIT_FIT SEX of the copy's input) vs the copied
    and live saved assessment predictions."""
    rec, bad = {}, []
    sums = root / "SHA256SUMS"
    lines = [l_.split("  ", 1) for l_ in sums.read_text().splitlines() if l_.strip()] if sums.exists() else []
    mism = [rel.strip() for h, rel in lines if not (root / rel.strip()).is_file() or sha_file(root / rel.strip()) != h]
    rec.update({"entries": len(lines), "mismatches": mism[:10], "n_mismatches": len(mism)})
    if mism or not lines:
        bad.append("SHA256SUMS")
    cp = root / "cbp_v1"
    dep = next((q for q in (root / "dependencies").rglob("adult_jcv.npz")), None) if (root / "dependencies").exists() else None
    rec["copied_input_present_and_pinned"] = bool(dep) and sha_file(dep) == SRC_SHA
    if not rec["copied_input_present_and_pinned"]:
        bad.append("pinned input not in the copy")
        return {**rec, "status": "FAIL", "failures": bad}
    k = int(targets.get("seed", 1))
    Dc = Data(src=dep)
    rec["copy_input_roles_and_X_equal_live"] = bitwise(Dc.X, Dlive.X) and all(bitwise(Dc.idx[r_], Dlive.idx[r_])
                                                                              for r_ in ROLES)
    if not rec["copy_input_roles_and_X_equal_live"]:
        bad.append("copy input roles / X")
    ud = cp / "admitted" / f"rel__s{k}__U"
    mine = own_teacher(ud / "model.pt", [ud / "head_0.joblib", ud / "head_1.joblib"], Dc.X)
    tz = np.load(cp / "run" / "units" / f"tea__s{k}__U" / "teacher.npz", allow_pickle=False)
    rec["teacher_U_forward_bitwise_vs_copied_unit"] = all(bitwise(mine[x].astype(tz[x].dtype), tz[x])
                                                         for x in ("p1", "p2", "d1", "d2", "c1", "c2", "r1", "r2"))
    if not rec["teacher_U_forward_bitwise_vs_copied_unit"]:
        bad.append("teacher U")
    P, d = {1: mine["p1"], 2: mine["p2"]}, {1: mine["d1"], 2: mine["d2"]}
    rels = {}
    for lab, un in (targets.get("policies") or {}).items():
        if not un.startswith("pol__"):
            continue
        rel = policy_pair_release(cp / "run" / "units" / un / "policy.json", P, d)
        rels[un] = rel
        cz = np.load(cp / "run" / "units" / un / "release.npz", allow_pickle=False)
        lz = np.load(UNITS / un / "release.npz", allow_pickle=False)
        ok = all(bitwise(rel[f"{a}{r}"], cz[f"{a}{r}"]) and bitwise(rel[f"{a}{r}"], lz[f"{a}{r}"])
                 for a in ("tok", "q", "hard") for r in (1, 2)) and all(np.array_equal(rel[f"hard{r}"], d[r]) for r in (1, 2))
        rec[f"policy: {lab}"] = {"unit": un, "own_application_bitwise_vs_copy_and_live_decisions_preserved": ok}
        if not ok:
            bad.append(f"policy {lab}")
    at = targets.get("attacker") or {}
    kw = at.get("kwargs") or {}
    if kw.get("release_unit") in rels and kw.get("outer_unit"):
        orec = jload(cp / "run" / "units" / kw["outer_unit"] / "record.json")
        view = kw.get("view", "pair")
        sc = orec["families"][orec["primary_family"]]["scored"][view]["auc"]
        cand, sview, att = sc["label"].split(":")
        r_ = rels[kw["release_unit"]]
        X, toks = my_views({"tok1": r_["tok1"], "tok2": r_["tok2"], "p1": r_["q1"], "p2": r_["q2"], "hard1": r_["hard1"],
                            "hard2": r_["hard2"], "alpha1": r_["alpha1"], "alpha2": r_["alpha2"]}, Dc, "code")
        Lc = Dc.labels()
        y = Lc["sex"].astype(np.int64)
        fa, sel, a_idx = Dc.idx["AUDIT_FIT"], Dc.idx[INNER], Dc.idx[ASSESS]
        assert (y[fa] >= 0).all() and (y[sel] >= 0).all()
        pred_idx = np.concatenate([sel, a_idx])
        key = (at.get("saved") or {}).get("key", f"P_auc_{view}")
        oc = np.load(cp / "run" / "units" / kw["outer_unit"] / "preds.npz", allow_pickle=False)[key]
        ol = np.load(UNITS / kw["outer_unit"] / "preds.npz", allow_pickle=False)[key]
        diffs = []
        for s_ in range(3):
            if att.startswith("CCpair"):
                pr, _ = my_cc_pair(toks["v1"], toks["v2"], y, fa, pred_idx, np.arange(len(sel)), float(att.split("alpha")[1]))
            elif att.startswith("CC_"):
                pr = my_cc(toks[sview], y, fa, pred_idx, float(att.split("alpha")[1]))[0]
            else:
                pr = my_p1(my_attacker(att, s_).fit(X[sview][fa], y[fa]), X[sview][pred_idx])
            pa = pr[len(sel):]
            diffs.append(max(maxdiff(pa, oc[s_][:, 1]), maxdiff(pa, ol[s_][:, 1])))
        rec["selected_attacker_refit_from_copy"] = {"label": sc["label"], "max_abs_diff_per_seed_vs_copy_and_live": diffs,
                                                    "bitwise": all(x == 0.0 for x in diffs),
                                                    "labels_read": "AUDIT_FIT / INNER_SELECTION SEX only"}
        if not all(x <= 1e-12 for x in diffs):
            bad.append("attacker refit")
    else:
        rec["selected_attacker_refit_from_copy"] = {"status": "PENDING", "reason": "no attacker target"}
    rec["status"] = "FAIL" if bad else "PASS"
    rec["failures"] = bad
    return rec


def check_restore(Dlive: "Data"):
    """Every cbp same-device copy restored independently (the closeout's own receipts are compared, never trusted)."""
    copies = sorted(q for q in CACHE.glob("cbp_v1_local_copy_*") if q.is_dir())
    tp = RUN / "closeout_targets.json"
    targets = jload(tp) if tp.exists() else {"seed": 1, "policies": {}}
    out = {"copies_present": [q.name for q in copies], "targets_present": tp.exists()}
    if not copies:
        return res("PENDING", reason="no cbp copy yet (closeout)", **out)
    bad = []
    for C in copies:
        r_ = restore_from_copy(C, targets, Dlive)
        out[C.name] = r_
        if r_["status"] != "PASS":
            bad.append(C.name)
        br = C / "BACKUP_RECORD.json"
        if br.exists():
            b = jload(br)
            r_["backup_record_sums_sha256_equal_own"] = b.get("SHA256SUMS_sha256") == sha_file(C / "SHA256SUMS")
            r_["backup_record_files_equal_own_entries"] = b.get("files") == r_["entries"]
            if not (r_["backup_record_sums_sha256_equal_own"] and r_["backup_record_files_equal_own_entries"]):
                bad.append(f"{C.name}: backup record")
    bv = RES / "BACKUP_VERIFICATION.json"
    if bv.exists():
        B = jload(bv)
        out["backup_verification"] = {k_: B.get(k_) for k_ in ("status", "off_device_backup", "destination", "files",
                                                               "uncached_readback_match", "required_restores",
                                                               "restore_all_pass")}
        newest = copies[-1].name
        out["backup_verification_points_to_newest_copy"] = newest in str(B.get("destination"))
        out["backup_verification_states_off_device_pending"] = B.get("status") == \
            "LOCAL_SAME_DEVICE_COPY_VERIFIED_OFF_DEVICE_BACKUP_PENDING" and str(B.get("off_device_backup", "")).startswith(
                "PENDING")
        if not (out["backup_verification_points_to_newest_copy"] and B.get("restore_all_pass") is True):
            bad.append("BACKUP_VERIFICATION.json")
    else:
        bad.append("BACKUP_VERIFICATION.json absent")
    return res("FAIL" if bad else "PASS", failures=bad, **out,
               note="a same-device copy proves restorability, not off-device custody (drive absent: PENDING)")


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


# ------------------------------------------------------------------------------------------------ cbp self-tests
def composition_closure_ok(pol_cids):
    """The U source bank must compose with EVERY public map of its seed: the 27 registered codes + 11 extras."""
    got = sorted(c_ for c_ in pol_cids if c_)
    return len(got) == len(composition_cids()) and got == sorted(composition_cids())


def compose_first_better(src_val, cand, maximize=True):
    """Composed winner rule: per view the first bank (own, then every public map in lock order) whose seed-0
    selected value beats the running best by more than 1e-12. cand: [(cid, value)]."""
    best, win = src_val, "source"
    for c_, v in cand:
        if (v > best + 1e-12) if maximize else (v < best - 1e-12):
            best, win = v, c_
    return win, best


def _u(acc, ll, br, c):
    return {"acc": acc, "logloss": ll, "brier": br, "const_acc": c}


UU = {1: _u(0.844, 0.330, 0.215, 0.755), 2: _u(0.475, 1.300, 0.660, 0.300)}


def _seed(dll=(0.0, 0.0), dbr=(0.0, 0.0), auc=(0.75, 0.75, 0.85), states=328.0, pres=True, dacc=(0.0, 0.0)):
    util = {i: _u(UU[i]["acc"] + dacc[i - 1], UU[i]["logloss"] + dll[i - 1], UU[i]["brier"] + dbr[i - 1],
                  UU[i]["const_acc"]) for i in (1, 2)}
    return {"auc": {"v1": auc[0], "v2": auc[1], "pair": auc[2]}, "util": util, "U": UU, "preserved": {1: pres, 2: pres},
            "states": states, "emitted": states}


def synthetic_bank(overrides=None):
    """All 32 candidates on 3 seeds with known answers. Privacy codes: pair AUC falls with lambda; occupation log-loss
    excess rises with lambda (headroom 0.006 crossed between 0.04 and 0.06, ordinary 0.01 never crossed)."""
    rows_in = {}
    lam_pair = {0.01: 0.840, 0.025: 0.835, 0.04: 0.830, 0.06: 0.824, 0.08: 0.820, 0.1: 0.815}
    lam_ll = {0.01: 0.0035, 0.025: 0.0045, 0.04: 0.0055, 0.06: 0.0068, 0.08: 0.0076, 0.1: 0.0085}
    fam_shift = {"LOCAL": 0.010, "SEQ-12": 0.002, "SEQ-21": 0.003, "JOINT": 0.0}
    for f_ in PRIV_FAMS:
        for lam in LAMS:
            p_ = lam_pair[lam] + fam_shift[f_]
            rows_in[b_config(f_, lam)] = {k: _seed((0.0005, lam_ll[lam]), (0.0003, 0.002), (0.760, 0.770, p_), 300.0)
                                          for k in SEEDS}
    rows_in[direct_id(*RATE)] = {k: _seed((0.0008, 0.0030), (0.0005, 0.0017), (0.762, 0.790, 0.849), 328.0) for k in SEEDS}
    rows_in[b_config("FINE-TASK")] = {k: _seed((0.0008, 0.0031), (0.0005, 0.0017), (0.761, 0.788, 0.846), 329.0)
                                      for k in SEEDS}
    rows_in[b_config("CLASS")] = {k: _seed((0.03, 0.20), (0.02, 0.10), (0.70, 0.70, 0.739), 8.0) for k in SEEDS}
    rows_in["SRC|U"] = {k: _seed(auc=(0.77, 0.80, 0.858), states=math.inf) for k in SEEDS}
    rows_in["SRC|RAW-J_b0.3"] = {k: _seed((0.001, 0.02), (0.001, 0.01), (0.70, 0.71, 0.810), math.inf, dacc=(0, -0.009))
                                 for k in SEEDS}
    rows_in["REF|E"] = {k: _seed((0.08, 0.05), (0.03, 0.02), (0.60, 0.70, 0.75), math.inf, dacc=(-0.05, 0)) for k in SEEDS}
    rows_in["REF|F"] = {k: _seed((0.03, 0.046), (0.01, 0.02), (0.65, 0.66, 0.704), math.inf, dacc=(0, -0.024))
                        for k in SEEDS}
    rows_in["REF|F0"] = {k: _seed((0.02, 0.03), (0.01, 0.015), (0.68, 0.70, 0.76), math.inf, dacc=(0, -0.015))
                         for k in SEEDS}
    for c_, f_ in (overrides or {}).items():
        rows_in[c_] = f_(rows_in[c_])
    return {c_: config_row(c_, v) for c_, v in rows_in.items()}, rows_in


def selftest_selection():
    out = {}
    rows, rows_in = synthetic_bank()
    sel = my_selection_cbp(rows)
    st = sel["statuses"]
    # expected: T* = FINE-TASK (0.846 < DIRECT 0.849 < U 0.858; CLASS / F0 ineligible); C_rate = SEQ-12 l0.1 (0.817);
    # C_global = C_rate (RAW-J, E, F, F0 ineligible); P* = JOINT l0.04 (lowest pair among headroom-eligible: 0.830);
    # J* = JOINT l0.04 (its locals 0.760 / 0.770 <= C_rate 0.760 / 0.770 + 0.005)
    exp = {"T*": "U|FINE-TASK|i8o64", "C_rate": "U|SEQ-12|i8o64|l0.1", "C_global": "U|SEQ-12|i8o64|l0.1",
           "P*": "U|JOINT|i8o64|l0.04", "J*": "U|JOINT|i8o64|l0.04", "Q": "U|DIRECT-TASK|i8o64"}
    got = {x: st[x].get("config") for x in exp}
    out["roles_expected"] = got == exp
    out["roles"] = got
    out["winning_family"] = st["P*"].get("winning_family") == "JOINT"
    d = sel["diagnostics"]
    out["no_headroom_winner_is_JOINT_l0.1"] = d["ordinary_privacy_winner_no_headroom"]["config"] == "U|JOINT|i8o64|l0.1"
    hc = d["headroom_changes_winner"]
    out["headroom_changes_winner"] = hc.get("changed") is True and abs(hc["give_up_mean_pair_auc"] - (0.830 - 0.815)) < 1e-12 \
        and len(hc["give_up_per_seed"]) == 3
    out["family_winners"] = {f_: v.get("config") for f_, v in d["family_headroom_winners"].items()} == \
        {f_: b_config(f_, 0.04) for f_ in PRIV_FAMS}
    out["role_aliases_recorded"] = ("C_global", "C_rate") in sel["aliases"] and ("J*", "P*") in sel["aliases"]

    # guard (inclusive a <= g + 0.005): a JOINT local AUC above C_rate + 0.005 on ONE seed blocks J*, not P*
    def bump(v):
        v = {k: dict(s) for k, s in v.items()}
        v[2] = dict(v[2])
        v[2]["auc"] = {"v1": 0.760, "v2": 0.7760, "pair": v[2]["auc"]["pair"]}
        return v
    sg = my_selection_cbp(synthetic_bank({b_config("JOINT", 0.04): bump})[0])["statuses"]
    out["one_seed_guard_breach_blocks"] = sg["J*"]["config"] == "U|JOINT|i8o64|l0.025" and \
        sg["P*"]["config"] == "U|JOINT|i8o64|l0.04"
    r_eq = config_row(b_config("LOCAL", 0.01), {k: _seed(auc=(0.765, 0.770, 0.8)) for k in SEEDS})
    r_g = config_row(b_config("FINE-TASK"), {k: _seed(auc=(0.760, 0.770, 0.8)) for k in SEEDS})
    out["guard_inclusive_and_zero_shortfall"] = guard_check(r_eq, [r_g])[0] == (0.765 <= 0.760 + 0.005) and \
        guard_check(r_g, [r_g]) == (True, 0.0) and guard_check(r_eq, [r_g])[1] == (0.0 if guard_check(r_eq, [r_g])[0]
                                                                                 else guard_check(r_eq, [r_g])[1])

    # no headroom-eligible privacy code -> NO_ELIGIBLE_NOMINEE (HEADROOM_SELECTION_FAILURE), fixed fallback, no relaxing
    def worse(v):
        return {k: _seed((0.0005, 0.0068), (0.0003, 0.002), (0.760, 0.770, v[k]["auc"]["pair"]), 300.0) for k in SEEDS}
    sn = my_selection_cbp(synthetic_bank({c_: worse for c_ in privacy_cids((0.01, 0.025, 0.04))})[0])["statuses"]
    fb = sn["P*"]
    out["no_eligible_fallback"] = fb["status"] == "NO_ELIGIBLE_NOMINEE" and fb["reason"] == "HEADROOM_SELECTION_FAILURE" \
        and fb["descriptive_config"] == "U|JOINT|i8o64|l0.06" and \
        abs(fb["fallback_shortfalls"]["shortfall_headroom"] - (0.0068 - 0.006) / 0.006) < 1e-9
    # missing guard comparator with an eligible candidate -> INVALID_NOMINEE (MISSING_GUARD_COMPARATOR)
    rows_m = dict(rows)
    rows_m["SRC|U"] = config_row("SRC|U", {k: rows_in["SRC|U"][k] for k in (0, 1)})
    sm = my_selection_cbp(rows_m)["statuses"]
    out["missing_comparator_invalidates"] = sm["T*"]["status"] == "INVALID_COMPARATOR" and \
        sm["T*"]["reason"] == "FIT_OR_ADMISSION_FAILURE" and sm["P*"]["status"] == "INVALID_NOMINEE" and \
        sm["P*"]["reason"] == "MISSING_GUARD_COMPARATOR" and sm["P*"]["missing_guards"] == ["T*"]
    # missing guard and nothing eligible -> NO_ELIGIBLE with its own reason; fallback flagged, no zero guard field
    rows_mn = synthetic_bank({c_: worse for c_ in privacy_cids()})[0]
    rows_mn["SRC|U"] = rows_m["SRC|U"]
    smn = my_selection_cbp(rows_mn)["statuses"]["P*"]
    out["missing_guard_without_eligible"] = smn["status"] == "NO_ELIGIBLE_NOMINEE" and \
        smn["reason"] == "HEADROOM_SELECTION_FAILURE" and \
        smn["fallback_rank_status"] == "INVALID_MISSING_GUARD_COMPARATOR" and smn["fallback_shortfalls"]["shortfall_guard"] is None
    # decision failure is a technical failure of the role, never a shortfall
    rows_d = synthetic_bank({b_config("LOCAL", 0.01): lambda v: {k: {**s, "preserved": {1: True, 2: k != 1}}
                                                                  for k, s in v.items()}})[0]
    pd_ = my_selection_cbp(rows_d)["statuses"]["P*"]
    out["decision_failure_invalid"] = rows_d[b_config("LOCAL", 0.01)]["valid"] is False and \
        pd_["status"] == "INVALID_NOMINEE" and pd_["reason"] == "DECISION_PRESERVATION_FAILURE"
    rows_nf = synthetic_bank({"REF|E": lambda v: {k: {**s, "auc": {**s["auc"], "v1": float("nan")}} for k, s in v.items()}})[0]
    cg = my_selection_cbp(rows_nf)["statuses"]["C_global"]
    out["non_estimable_invalid"] = cg["status"] == "INVALID_COMPARATOR" and cg["reason"] == "NON_ESTIMABLE_INNER_METRIC"
    # ordinary-utility failure of every candidate -> ORDINARY_UTILITY_FAILURE
    bad_all = {c_: (lambda v: {k: _seed((0.0005, 0.02), (0.0003, 0.002), (0.76, 0.77, 0.8), 300.0) for k in SEEDS})
               for c_ in privacy_cids()}
    out["ordinary_failure_reason"] = my_selection_cbp(synthetic_bank(bad_all)[0])["statuses"]["P*"]["reason"] == \
        "ORDINARY_UTILITY_FAILURE"
    # local-guard failure: headroom-eligible codes exist but every one breaches T* + 0.005
    hi = {c_: (lambda v: {k: {**s, "auc": {"v1": 0.80, "v2": 0.80, "pair": s["auc"]["pair"]}} for k, s in v.items()})
          for c_ in privacy_cids()}
    out["guard_failure_reason"] = my_selection_cbp(synthetic_bank(hi)[0])["statuses"]["P*"]["reason"] == "LOCAL_GUARD_FAILURE"

    # aliases: identical deployed maps on all seeds (own pair fingerprints) -> simplest family named
    def fp(tag):
        return lambda v: {k: {**s, "pair_fp": f"{tag}{k}"} for k, s in v.items()}
    ov = {c_: fp(c_) for c_ in all_cids()}
    ov[b_config("JOINT", 0.04)] = fp("same")
    ov[b_config("SEQ-21", 0.04)] = fp("same")
    ov[b_config("LOCAL", 0.025)] = lambda v: {k: {**s, "pair_fp": "same0" if k == 0 else f"loc{k}"} for k, s in v.items()}
    sa = my_selection_cbp(synthetic_bank(ov)[0])["statuses"]["P*"]
    out["alias_simplest_family"] = sa["config"] == "U|JOINT|i8o64|l0.04" and sa["winning_family"] == "SEQ-21" and \
        sa["family_of_config"] == "JOINT" and sa["aliases"]["alias_set"] == ["U|SEQ-21|i8o64|l0.04"] and \
        sa["aliases"]["partial_aliases"] == {"U|LOCAL|i8o64|l0.025": [0]}
    out["simplest_family_order"] = simplest_family(["JOINT", "SEQ-21", "SEQ-12"]) == "SEQ-12=SEQ-21" and \
        simplest_family(["JOINT", "LOCAL"]) == "LOCAL"
    # shortfall formulas and the registered seed mean
    tc = task_check(_u(0.80, 0.42, 0.26, 0.755), _u(0.844, 0.40, 0.25, 0.755))
    gU, gN = 0.844 - 0.755, 0.80 - 0.755
    want = max((0.844 - 0.01 - 0.80) / 0.01, (0.02 - 0.01) / 0.01, (0.01 - 0.005) / 0.005,
               (0.8 * gU - gN) / max(0.8 * gU, 0.03), (0.03 - gN) / 0.03)
    out["shortfall_formula"] = abs(tc["shortfall_ordinary_raw"] - want) < 1e-12 and \
        abs(tc["shortfall_headroom_raw"] - max((0.02 - 0.006) / 0.006, (0.01 - 0.0035) / 0.0035)) < 1e-12
    inside = task_check(_u(0.844, 0.25 + 0.0059, 0.125 + 0.0034, 0.755), _u(0.844, 0.25, 0.125, 0.755))
    outside = task_check(_u(0.844, 0.25 + 0.0061, 0.125, 0.755), _u(0.844, 0.25, 0.125, 0.755))
    out["headroom_from_excess"] = inside["headroom_ok"] and inside["ordinary_ok"] and not outside["headroom_ok"] and \
        outside["ordinary_ok"]
    out["seed_mean_in_seed_order"] = seed_mean([0.1, 0.2, 0.3]) == (0.1 + 0.2 + 0.3) / 3
    ok = all(v for v in out.values() if isinstance(v, bool))
    return res("PASS" if ok else "FAIL", **out)


def _ep(i, claim, kind, point, lo, hi, tgt, side):
    return {"id": f"P{i:02d}", "claim": claim, "kind": kind, "point": point, "lower": lo, "upper": hi, "target": tgt,
            "side": side, "se": 0.001, "nonfinite": 0}


def synthetic_endpoints(fail_id=None, how="precision", nonfinite_id=None):
    eps, i = [], 1
    for claim in ("A", "B", "C"):
        eps.append(_ep(i, claim, "coalition", 0.03, 0.025, 0.035, MARGIN_PAIR, "lower>"))
        i += 1
        for _ in ("v1", "v2"):
            eps.append(_ep(i, claim, "local", -0.01, -0.012, -0.008, MARGIN_LOCAL, "upper<"))
            i += 1
        for kind, tgt, side, pt in (("acc", -0.01, "lower>", 0.0), ("logloss", 0.01, "upper<", 0.003),
                                    ("brier", 0.005, "upper<", 0.001), ("retain", 0.0, "lower>", 0.05)):
            for _ in (0, 1):
                lo, hi = (pt, pt) if kind in ("acc", "retain") else (pt - 0.002, pt + 0.002)    # zero-variance identities
                eps.append(_ep(i, claim, kind, pt, lo, hi, tgt, side))
                i += 1
    for kind, tgt in (("logloss", 0.01), ("brier", 0.005)):
        for _ in (0, 1):
            eps.append(_ep(i, "Q", kind, 0.003, 0.001, 0.005 if kind == "logloss" else 0.004, tgt, "upper<"))
            i += 1
    for e in eps:
        if e["id"] == fail_id:
            t = e["target"]
            if e["side"] == "lower>":
                e.update({"precision": {"lower": t - 0.001, "point": t + 0.001, "upper": t + 0.003},
                          "violation": {"lower": t - 0.01, "upper": t - 0.001, "point": t - 0.005},
                          "point": {"lower": t - 0.002, "point": t - 0.001, "upper": t + 0.001}}[how])
            else:
                e.update({"precision": {"upper": t + 0.001, "point": t - 0.001, "lower": t - 0.003},
                          "violation": {"lower": t + 0.001, "upper": t + 0.01, "point": t + 0.005},
                          "point": {"lower": t - 0.001, "point": t + 0.001, "upper": t + 0.002}}[how])
        if e["id"] == nonfinite_id:
            e.update({"upper": float("nan")})
    return eps


def selftest_labels():
    """Truth-table tests against LABEL_TRUTH_TABLE.json: absent nominee, absent comparator, aliases, every single failed
    clause (precision / point / violation), nonfinite values, zero-variance identities, missing slots, mixed valid /
    incomplete claims, global and per-claim control failures, precedence."""
    good = {"T*": {"status": "NOMINEE", "config": "U|FINE-TASK|i8o64"}, "C_rate": {"status": "NOMINEE", "config": "S"},
            "C_global": {"status": "NOMINEE", "config": "S"},
            "P*": {"status": "NOMINEE", "config": "U|SEQ-21|i8o64|l0.04", "winning_family": "SEQ-21"},
            "J*": {"status": "NOMINEE", "config": "J"}, "Q": {"status": "NOMINEE", "config": "U|DIRECT-TASK|i8o64"}}
    out = {}
    eps = synthetic_endpoints()
    L0 = labels_cbp(good, eps)
    out["all_pass"] = L0["label"] == "PRIVACY_COMPRESSION_DEVELOPMENT_CRITERION_MET (SEQ-21) + " \
                                     "JOINT_DEVELOPMENT_CRITERION_MET" and L0["q"] == "PASS"
    single, roots_want = {}, {"precision": "ASSESSMENT_PRECISION_FAILURE", "violation": "MEASURED_VIOLATION_SUPPORTED_BY_BOUND",
                              "point": "CLAUSE_NOT_ESTABLISHED"}
    for e in eps:
        for how in ("precision", "violation", "point"):
            Lx = labels_cbp(good, synthetic_endpoints(e["id"], how))
            c_ = e["claim"]
            got = Lx["q"] if c_ == "Q" else Lx["claims"][c_]
            root = Lx["q_root_cause"] if c_ == "Q" else Lx["root_causes"][c_]
            others = [o for o in ("A", "B", "C") if o != c_]
            single[f"{e['id']}/{how}"] = got == "NOT_ESTABLISHED" and root == roots_want[how] and \
                all(Lx["claims"][o] == "PASS" for o in others)
    out["every_single_failed_clause"] = all(single.values())
    out["single_clause_cases"] = len(single)
    out["c_fails_joint_passes"] = labels_cbp(good, synthetic_endpoints("P23", "precision"))["label"] == \
        "JOINT_DEVELOPMENT_CRITERION_MET"
    out["a_fails_c_passes"] = labels_cbp(good, synthetic_endpoints("P01", "violation"))["label"] == \
        "PRIVACY_COMPRESSION_DEVELOPMENT_CRITERION_MET (SEQ-21)"
    bad_all = synthetic_endpoints("P01", "violation")
    for e in bad_all:
        if e["id"] == "P23":
            e.update({"lower": 0.01, "point": 0.015})
    out["q_only"] = labels_cbp(good, bad_all)["label"] == "CONFIDENCE_FEASIBILITY_ESTABLISHED_NO_METHOD_CRITERION"
    for e in bad_all:
        if e["id"] == "P34":
            e.update({"upper": 0.012})
    out["no_advantage"] = labels_cbp(good, bad_all)["label"] == "EXPERIMENTAL_NO_ADVANTAGE"
    # an INCOMPLETE claim outranks Q PASS when no favourable label exists
    bad_q = synthetic_endpoints("P01", "violation")
    for e in bad_q:
        if e["id"] == "P23":
            e.update({"lower": 0.01, "point": 0.015})
    st_inv = dict(good, **{"C_global": {"status": "INVALID_COMPARATOR", "config": None}})
    Li = labels_cbp(st_inv, bad_q)
    out["incomplete_outranks_q_pass"] = Li["label"] == "INCOMPLETE_OR_INVALID" and Li["q"] == "PASS" and \
        Li["incomplete_displayed"] == ["B"]
    st_noP = dict(good, **{"P*": {"status": "NO_ELIGIBLE_NOMINEE", "config": None, "descriptive_config": "X",
                                  "reason": "HEADROOM_SELECTION_FAILURE"}})
    Ln = labels_cbp(st_noP, synthetic_endpoints())
    out["absent_nominee"] = Ln["claims"]["C"] == "NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE" and \
        Ln["root_causes"]["C"] == "HEADROOM_SELECTION_FAILURE" and Ln["label"] == "JOINT_DEVELOPMENT_CRITERION_MET"
    st_noT = dict(good, **{"T*": {"status": "INVALID_COMPARATOR", "config": None}})
    Lt = labels_cbp(st_noT, synthetic_endpoints())
    out["absent_comparator_never_passes"] = Lt["claims"]["C"] == "INCOMPLETE_OR_INVALID" and \
        Lt["root_causes"]["C"] == "INVALID_OR_MISSING_COMPARATOR" and "C" in Lt["incomplete_displayed"] and \
        "PRIVACY_COMPRESSION" not in Lt["label"]
    st_noC = dict(good, **{"C_global": {"status": "NO_ELIGIBLE_COMPARATOR", "config": None}})
    Lc = labels_cbp(st_noC, synthetic_endpoints())
    out["no_eligible_comparator_is_incomplete"] = Lc["claims"]["B"] == "INCOMPLETE_OR_INVALID" and \
        Lc["root_causes"]["B"] == "NO_ELIGIBLE_COMPARATOR"
    st_both = dict(good, **{"P*": st_noP["P*"], "T*": {"status": "NO_ELIGIBLE_COMPARATOR", "config": None}})
    out["no_eligible_nominee_precedes_no_eligible_comparator"] = labels_cbp(st_both, synthetic_endpoints())["claims"]["C"] \
        == "NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE"
    Lnf = labels_cbp(good, synthetic_endpoints(nonfinite_id="P29"))
    out["nonfinite_invalid"] = Lnf["claims"]["C"] == "INCOMPLETE_OR_INVALID" and Lnf["clause_outcomes"]["P29"] == "INVALID" \
        and Lnf["root_causes"]["C"] == "NONFINITE_PRIMARY_QUANTITY"
    out["missing_slot_incomplete"] = labels_cbp(good, [e for e in synthetic_endpoints() if e["id"] != "P30"])["root_causes"][
        "C"] == "MISSING_SLOTS"
    out["zero_variance_identity_passes"] = clause_outcome(_ep(4, "A", "acc", 0.0, 0.0, 0.0, -0.01, "lower>"))[0] == "PASS"
    out["aliases_keep_slots"] = len(synthetic_endpoints()) == N_ENDPOINTS and \
        labels_cbp(dict(good, **{"J*": good["P*"]}), synthetic_endpoints())["claims"]["A"] == "PASS"
    out["global_technical_failure"] = labels_cbp(good, synthetic_endpoints(), technical_valid=False)["label"] == \
        "INCOMPLETE_OR_INVALID"
    Lpc = labels_cbp(good, synthetic_endpoints(), control_ok={"C": False})
    out["per_claim_control_failure"] = Lpc["claims"]["C"] == "INCOMPLETE_OR_INVALID" and \
        Lpc["root_causes"]["C"] == "FAILED_REQUIRED_CONTROL" and Lpc["label"] == "JOINT_DEVELOPMENT_CRITERION_MET"
    st_mix = dict(good, **{"C_rate": {"status": "INVALID_COMPARATOR", "config": None}})
    Lm = labels_cbp(st_mix, synthetic_endpoints())
    out["mixed_valid_and_incomplete_displayed"] = Lm["label"] == "PRIVACY_COMPRESSION_DEVELOPMENT_CRITERION_MET " \
                                                                 "(SEQ-21)" and Lm["incomplete_displayed"] == ["A"]
    out["q_no_eligible"] = labels_cbp(dict(good, Q={"status": "NO_ELIGIBLE_NOMINEE", "config": None}),
                                      synthetic_endpoints())["q"] == "NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE"
    ok = all(v for v in out.values() if isinstance(v, bool))
    return res("PASS" if ok else "FAIL", **out,
               note="implemented from LABEL_TRUTH_TABLE.json (FIT_LOCK); the clause-outcome boundaries are strict as "
                    "registered")


def selftest_defects(rng):
    """Deliberate defects of prompt section 14: each must be DETECTED by the verifier's own comparators."""
    out = {}
    # 1. reversed best / worst reader (bank level and role level)
    vals = [0.61, 0.74, 0.69, 0.74 + 5e-13, 0.52]
    out["reversed_reader_detected"] = pick_first(vals, True) != int(np.argmin(vals)) and pick_first(vals, True) == 1
    rows, _ = synthetic_bank()
    out["reversed_role_order_detected"] = my_selection_cbp(rows, reverse=True)["resolved"]["P*"] != \
        my_selection_cbp(rows)["resolved"]["P*"]
    # 2. wrong AUC orientation (and the forbidden max(AUC, 1 - AUC))
    ys = rng.integers(0, 2, 600)
    sc = rng.normal(size=600) - 0.6 * ys
    a = my_auc(ys, sc)
    out["wrong_orientation_detected"] = a < 0.5 and abs((1 - a) - a) > 1e-3 and abs(my_auc(ys, -sc) - (1 - a)) < 1e-12
    # 3. probabilities audited while token identities are omitted: two IDs share one decoded vector
    n = 3000
    s = rng.integers(0, 2, n)
    base = rng.integers(0, 4, n)
    tok = 2 * base + s                                     # the sensitive bit lives ONLY in the token identity
    q = np.stack([0.6 + 0.05 * base, 0.4 - 0.05 * base], 1)
    hard = np.zeros(n, dtype=np.int64)
    fa_, sl_ = np.arange(0, 2000), np.arange(2000, n)
    full = np.hstack([np.eye(8)[tok], q, np.eye(2)[hard]])
    noid = np.hstack([q, np.eye(2)[hard]])
    auc_full = my_auc(s[sl_], my_p1(my_attacker("LR_C1.0", 0).fit(full[fa_], s[fa_]), full[sl_]))
    auc_noid = my_auc(s[sl_], my_p1(my_attacker("LR_C1.0", 0).fit(noid[fa_], s[fa_]), noid[sl_]))
    out["token_identity_omission_detected"] = auc_full > 0.95 and abs(auc_noid - 0.5) < 0.06
    # 4. pair alignment shuffled: S = u XOR v readable only from the aligned pair
    u, v = rng.integers(0, 2, n), rng.integers(0, 2, n)
    sx = u ^ v
    t1, t2 = u, v
    al = my_cc_pair(t1, t2, sx, fa_, sl_, np.arange(len(sl_)), 1.0)[0]
    t2s = t2.copy()
    t2s[sl_] = rng.permutation(t2[sl_])
    sh = my_cc_pair(t1, t2s, sx, fa_, sl_, np.arange(len(sl_)), 1.0)[0]
    out["pair_shuffle_detected"] = my_auc(sx[sl_], al) > 0.99 and abs(my_auc(sx[sl_], sh) - 0.5) < 0.06
    # 5. clean probabilities accidentally appended: the release key contract refuses it
    keys_ok = sorted(RELEASE_KEYS)
    out["appended_clean_probabilities_detected"] = sorted(list(RELEASE_KEYS) + ["p1"]) != keys_ok and \
        sorted(list(RELEASE_KEYS) + ["fine1"]) != keys_ok
    # 6. missing composed-source winner: closure refuses an incomplete bank; the winner changes when it is dropped
    pol = composition_cids()
    cand = [(c_, 0.80 + (0.07 if c_ == "U|DIRECT-TASK|i8o64" else 0.0)) for c_ in pol]
    w_full = compose_first_better(0.85, cand)[0]
    w_miss = compose_first_better(0.85, [x for x in cand if x[0] != w_full])[0]
    out["missing_composed_winner_detected"] = composition_closure_ok(pol) and not composition_closure_ok(pol[:-1]) and \
        not composition_closure_ok([c_ for c_ in pol if c_ != w_full]) and w_full != w_miss
    # 7. seed-averaged eligibility instead of every seed
    def one_bad_seed(vv):
        vv = {k: dict(x) for k, x in vv.items()}
        vv[1] = _seed((0.0005, 0.0125), (0.0003, 0.002), (0.760, 0.770, 0.800), 300.0)
        vv[0] = _seed((0.0005, 0.0020), (0.0003, 0.002), (0.760, 0.770, 0.800), 300.0)
        vv[2] = _seed((0.0005, 0.0020), (0.0003, 0.002), (0.760, 0.770, 0.800), 300.0)
        return vv
    _, rin = synthetic_bank({b_config("JOINT", 0.08): one_bad_seed})
    right = config_row(b_config("JOINT", 0.08), rin[b_config("JOINT", 0.08)])
    wrong = config_row(b_config("JOINT", 0.08), rin[b_config("JOINT", 0.08)], seed_average=True)
    out["seed_averaged_eligibility_detected"] = (not right["ordinary_eligible"]) and wrong["ordinary_eligible"]
    # 8. headroom changed to the ordinary limit
    out["headroom_relaxed_detected"] = my_selection_cbp(rows, headroom_as_ordinary=True)["resolved"]["P*"] == \
        "U|JOINT|i8o64|l0.1" != my_selection_cbp(rows)["resolved"]["P*"]
    # 9. KL instead of true-label loss
    m = 5000
    Pt = rng.dirichlet(np.ones(6) * 0.7, m)
    Ptrue = Pt ** 3 / np.sum(Pt ** 3, 1, keepdims=True)    # labels follow a sharper law than the teacher (real data)
    yv = np.array([rng.choice(6, p=p_) for p_ in Ptrue])
    Qc = 0.85 * Pt + 0.15 * Pt.mean(0, keepdims=True)
    kl = float(np.mean(kl_paired(Pt, Qc)))
    llx = float(np.mean(-np.log(Qc[np.arange(m), yv]) + np.log(Pt[np.arange(m), yv])))
    out["kl_instead_of_true_label_loss_detected"] = abs(kl - llx) > 1e-3
    out["kl_vs_true_label_excess"] = {"kl": kl, "true_label_ll_excess": llx}
    # 10. wrong bootstrap group (rows instead of exact-record groups) or wrong family critical value
    G_ = 400
    gid = np.repeat(np.arange(G_), 3)
    xg = np.repeat(rng.normal(size=G_), 3) + 0.01 * rng.normal(size=3 * G_)
    bg = OwnBoot(gid, B=300, seed=BOOT_SEED, chunk=100)
    br = OwnBoot(np.arange(3 * G_), B=300, seed=BOOT_SEED, chunk=100)
    se_g = float(np.std(np.concatenate([WMean(xg)(W) for W in bg.chunks()]), ddof=1))
    se_r = float(np.std(np.concatenate([WMean(xg)(W) for W in br.chunks()]), ddof=1))
    z_wrong = NormalDist().inv_cdf(1 - 0.05 / 37)
    out["wrong_bootstrap_group_detected"] = se_g > 1.4 * se_r
    out["wrong_critical_value_detected"] = abs(z_wrong - Z_PRIMARY) > 0.1 and z_primary() == Z_PRIMARY
    # 11. absent comparator allowed to pass
    all_pass = {sid: "PASS" for sid in CLAIM_SLOTS["C"]}
    nomst = {"status": "NOMINEE", "config": "U|SEQ-21|i8o64|l0.04"}
    out["absent_comparator_pass_detected"] = \
        claim_status_cbp(nomst, {"status": "INVALID_COMPARATOR", "config": None}, all_pass, "C")[0] == "INCOMPLETE_OR_INVALID" \
        and claim_status_cbp(nomst, {"status": "NO_ELIGIBLE_COMPARATOR", "config": None}, all_pass, "C")[0] == \
        "INCOMPLETE_OR_INVALID" and claim_status_cbp(nomst, None, all_pass, "C")[0] == "INCOMPLETE_OR_INVALID"
    # 12. an assessment-rescored weight selected as nominee: the lock must name the INNER selection
    inner_p = my_selection_cbp(rows)["resolved"]["P*"]
    assess_best = "U|JOINT|i8o64|l0.1"                    # e.g. the lowest assessment pair AUC among privacy codes
    out["assessment_rescored_nominee_detected"] = nominee_matches_inner({"P*": assess_best}, {"P*": inner_p}) is False \
        and nominee_matches_inner({"P*": inner_p}, {"P*": inner_p}) is True
    ok = all(v for v in out.values() if isinstance(v, bool))
    return res("PASS" if ok else "FAIL", **out, defects_tested=12)


def nominee_matches_inner(lock_resolved, own_resolved):
    """The evaluation lock must carry exactly the inner-selected role configurations (no assessment re-scoring)."""
    return all(lock_resolved.get(x) == own_resolved.get(x) for x in own_resolved)


def selftests():
    out = {}
    rng = np.random.default_rng(20261006)
    # ---- import guard (also for unpickling a project class reference)
    _IN_SELFTEST[0] = True
    tried = []
    for m in ("cbp", "cbp.run", "cbp.select", "cbp.sema", "cbp.lock", "qpc", "qpc.kmeans", "qpc.compress", "qpc.utility",
              "qpc.sema", "dpc", "dpc.partition", "osf.data", "smf.audit", "rgj.finalize", "jcv.train", "pnx",
              "oar.study", "stored_model_eval.pilot_infer", "pcrl"):
        try:
            importlib.import_module(m)
            tried.append((m, False))
        except ImportError:
            tried.append((m, True))
    unp = {}
    for blob, nm in ((b"cqpc.kmeans\nfit\n.", "qpc.kmeans"), (b"cdpc.partition\nFinePartition\n.", "dpc.partition"),
                     (b"ccbp.select\nselect_all\n.", "cbp.select")):
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
    # ---- utility (sklearn parity), cbp eligibility / headroom / shortfalls / roles, labels, deliberate defects
    y = rng.integers(0, 6, 400)
    Pu = synth_probs(rng, 400, 6)
    u = my_utility(Pu, Pu.argmax(1), y, 6, 0)
    Y = np.eye(6)[y]
    sk_ok = abs(u["logloss"] - sk_log_loss(y, Pu, labels=list(range(6)))) < 1e-12 and \
        abs(u["brier"] - float(np.mean(np.sum((Pu - Y) ** 2, 1)))) < 1e-15
    out["utility_sklearn_parity"] = res("PASS" if sk_ok else "FAIL", sklearn_parity=bool(sk_ok))
    out["selection_rules"] = selftest_selection()
    out["label_truth_table"] = selftest_labels()
    out["deliberate_defects"] = selftest_defects(rng)
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


def scrub_check(text: str):
    home = str(Path.home())
    probes = [home, "/Users/", "/Volumes/", "/private/", "/tmp/", home.split("/")[-1], "BackgroundSync"]
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


PHASE1 = ("pins", "roles", "admitted_custody", "teachers", "references", "direct_task_codes",
          "direct_task_class_preservation", "fine_partitions", "code_search", "code_class_preservation", "unit_custody",
          "chronology", "code_hashes")
PHASE2 = ("endpoint_parity", "inner_audits", "source_composition", "selection", "controls")
PHASE3 = ("evaluation_lock", "outer_units", "endpoints", "published_tables", "chronology_assessment", "attacker_refits",
          "deployment_parity", "restore_parity", "budget")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, default=0, choices=(0, 1, 2, 3))
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-refit", action="store_true", help="skip the descriptive own head refits")
    ap.add_argument("--label", default=None, help="phase label written to the report (e.g. PHASE_1)")
    ap.add_argument("--refresh-restore", action="store_true",
                    help="re-run ONLY the independent restore check on the (refreshed) copy and update that node of the "
                         "existing report (no other check is recomputed)")
    args = ap.parse_args()
    t0, c0 = time.time(), time.process_time()
    if args.refresh_restore:
        return refresh_restore_only(args, t0, c0)
    others_start = heavy_processes()
    report = {"schema": "cbp-independent-verification-v1", "phase": args.label or f"PHASE_{args.phase}",
              "generated_at": iso(datetime.now(timezone.utc)),
              "verifier": f"{REL_RES}/verification/replay_cbp.py", "verifier_sha256": sha_file(Path(__file__)),
              "adapted_from": f"{QPC_REL}/verification/replay_qpc.py at {QPC_TIP[:7]} (sha256 " +
                              (sha_file(WT / QPC_REL / "verification" / "replay_qpc.py")
                               if (WT / QPC_REL / "verification" / "replay_qpc.py").exists() else "absent") + ")",
              "worktree_head": git("rev-parse", "HEAD"), "branch": BRANCH, "source_evidence_sha": QPC_EVIDENCE,
              "inputs": {"source_npz": "<PRIVATE_CACHE>/jcv_v1/inputs/adult_jcv.npz",
                         "cbp_units": "<PRIVATE_CACHE>/cbp_v1/run/units", "cbp_admitted": "<PRIVATE_CACHE>/cbp_v1/admitted",
                         "qpc_units_read_only": "<PRIVATE_CACHE>/qpc_v1/run/units",
                         "dpc_units_and_admitted_read_only": "<PRIVATE_CACHE>/dpc_v1"},
              "tolerances": {"tokens_decisions_partitions_bindings_hashes": "exact (bitwise / byte-identical)",
                             "decoded_probabilities_teacher_outputs": "exact (bitwise)",
                             "search_increments": LOG_TOL_B, "objective_terms": OBJ_TOL_B,
                             "k-means_receipt_objectives_relative": RECEIPT_REL_TOL, "fit_distortion_relative": 1e-12,
                             "inner_auc_ce_utility": 1e-12, "endpoint_point_se": 1e-12, "endpoint_bounds": 1e-11,
                             "published_6_decimal_prints": 5e-7, "verdicts_labels_roles": "exact agreement",
                             "selection_ordering_ties": f"{ORDER_DECIMALS}-decimal rounding (provisional; reconciled "
                                                        "with HEADROOM_SELECTION_RULES.json)"},
              "libraries": {"numpy": np.__version__, "torch": torch.__version__, "joblib": joblib.__version__,
                            "sklearn": __import__("sklearn").__version__, "scipy": __import__("scipy").__version__}}
    checks = {}
    st = selftests()
    checks["selftests"] = {"status": worst(*[v["status"] for v in st.values()]), **st}
    for k in PHASE1:
        checks[k] = pending(k, "PHASE 1 (after source admission)")
    for k in PHASE2:
        checks[k] = pending(k, "PHASE 2 (after fits, inner audits and select; before the assessment)")
    for k in PHASE3:
        checks[k] = pending(k, "PHASE 3 (after the assessment and inference)")
    report["real_data_read"] = args.phase >= 1
    report["assessment_labels_read"] = False
    if args.phase >= 1:
        D = Data()
        L = D.labels()
        assert all((L[k_][D.mask[ASSESS]] == -1).all() for k_ in LABEL_KEYS), "assessment labels must stay sealed"
        man = {"cbp_ROLE_MANIFEST": jload(RES / "ROLE_MANIFEST.json") if (RES / "ROLE_MANIFEST.json").exists() else None,
               "qpc_ROLE_MANIFEST_at_evidence": json.loads(git_show_bytes(QPC_EVIDENCE, f"{QPC_REL}/ROLE_MANIFEST.json")
                                                           or b"null"),
               "dpc_ROLE_MANIFEST_at_source_sha": json.loads(git_show_bytes(SOURCE_SHA, f"{DPC_REL}/ROLE_MANIFEST.json")
                                                             or b"null")}
        checks["roles"] = check_roles(D, man)
        checks["pins"] = check_pins()
        checks["admitted_custody"] = check_reuse_custody()
        checks["teachers"], T = check_teachers(D, L, refit=not args.no_refit)
        refs = {}
        checks["references"] = check_references(D, T, keep=refs)
        checks.update(check_direct(D, L, T))
        lams = (REUSED_LAMS + (COMPOSED_EXTRA_LAM,)) if args.phase == 1 else (LAMS + (COMPOSED_EXTRA_LAM,))
        cc, own_maps, own_terms = check_codes(D, L, T, lams)
        checks.update(cc)
        checks["unit_custody"] = check_units_complete(("tea__", "ref__", "fine__", "pol__"))
        checks["chronology"] = check_chronology()
        checks["code_hashes"] = check_code_hashes()
        if args.phase >= 2:
            checks["endpoint_parity"] = check_endpoint_parity(D)
            if any(UNITS.glob("inner__*")):
                checks["inner_audits"], checks["source_composition"], own_in = check_inner(D, L, T, refs)
                checks["selection"], sel_mine, sel_rows = check_selection(own_in)
            checks["controls"] = check_controls(D, L, {})
            checks["unit_custody"] = check_units_complete(("tea__", "ref__", "fine__", "pol__", "inner__"))
        if args.phase >= 3:
            gate = evaluation_lock_gate(fetch=True)
            report["assessment_label_gate"] = {k_: gate.get(k_) for k_ in ("checked_at", "git_fetch_ok", "exists",
                                                                         "commit", "commit_time", "first_push_time",
                                                                         "origin_show_equals_worktree", "commit_on_origin",
                                                                         "commits", "sha256", "ok")}
            checks["evaluation_lock"], EL = check_eval_lock(D, L, gate, sel_mine, own_in)
            LU = unsealed_labels(D, gate)
            report["assessment_labels_read"] = True
            checks["outer_units"], preds = check_outer(D, LU, T, refs, own_in, EL, gate)
            checks["endpoints"], eps, lev = check_endpoints_cbp(preds, EL)
            checks["published_tables"] = check_tables_cbp(own_in, sel_mine, sel_rows, eps, lev, EL)
            checks["chronology_assessment"] = check_late_chronology(gate, EL)
            if not args.no_refit:
                checks["attacker_refits"] = check_refits(D, L, LU, T, own_in, preds, EL)
            checks["deployment_parity"] = check_deployment(D, T)
            checks["restore_parity"] = check_restore(D)
            checks["budget"] = check_budget()
            checks["unit_custody"] = check_units_complete(("tea__", "ref__", "fine__", "pol__", "inner__", "outer__"))
    loaded = sorted(m for m in sys.modules if m.split(".")[0] in _FORBIDDEN_TOP)
    report["independence"] = res("PASS" if not loaded and not _BLOCKED and not _PRELOADED else "FAIL",
                                 guard="sys.meta_path finder refusing " + ", ".join(_FORBIDDEN_TOP) +
                                       " (also during joblib unpickling); torch.load(weights_only=True); worktree root "
                                       "removed from sys.path; semaphore invoked by path (-P) as the parent process",
                                 loaded_forbidden_modules=loaded, blocked_attempts_during_run=_BLOCKED,
                                 preloaded_before_guard=_PRELOADED,
                                 libraries_used=["numpy", "scipy", "sklearn", "torch", "joblib", "stdlib"],
                                 adapted_from_predecessor_verifier="import guard, role reconstruction, forward pass, "
                                                                   "KL k-means, search replay, attacker refits, "
                                                                   "bootstrap, inner / composition replay, chronology")
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
                         "threads": {"OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                                     "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                                     "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"), "torch": torch.get_num_threads()},
                         "other_cbp_processes_at_start": len(others_start),
                         "other_cbp_processes_at_end": len(heavy_processes()), "semaphore_slots_at_end": sema_status()}
    report["pending"] = sorted(k for k, v in status.items() if v == "PENDING")
    sema_runs = [{"label": e.get("label"), "released_at": e.get("at"), "wall_s": e.get("wall_s"), "cpu_s": e.get("cpu_s"),
                  "rc": e.get("rc")} for e in jsonl(RUN / "SEMA_LOG.jsonl")
                 if e.get("event") == "release" and str(e.get("label", "")).startswith("F:")]
    report["verifier_compute_ledger"] = {
        "semaphore_holds_completed": sema_runs,
        "semaphore_cpu_s_total": round(sum(float(r["cpu_s"] or 0) for r in sema_runs), 1),
        "semaphore_wall_s_total": round(sum(float(r["wall_s"] or 0) for r in sema_runs), 1),
        "this_run": {"phase": args.phase, "start": report["generated_at"], "wall_s": report["compute"]["wall_s"],
                     "cpu_s": report["compute"]["cpu_s_process"]},
        "light_runs_outside_the_semaphore": VERIFIER_RUNS,
        "note": "every verifier process, including the synthetic self-tests, runs inside a semaphore hold labelled F:*; "
                "the current run's hold is released after this file is written"}
    text = json.dumps(jsonable(report), indent=1, allow_nan=False)
    scrub_check(text)
    if not args.no_write:
        dest = Path(args.out) if args.out else OUT
        tmp = dest.with_suffix(".json.tmp")
        tmp.write_text(text + "\n")
        tmp.replace(dest)
    print(json.dumps(jsonable({"summary": {k_: v for k_, v in report["summary"].items() if k_ != "flagged_fail_warn"},
                               "flagged": report["summary"]["flagged_fail_warn"][:15], "compute": report["compute"]}),
                     indent=1, allow_nan=False))


def refresh_restore_only(args, t0, c0):
    """Update the restore_parity node of the existing report after an incremental refresh of the same-device copy."""
    dest = Path(args.out) if args.out else OUT
    report = jload(dest)
    D = Data()
    rp = check_restore(D)
    report["checks"]["restore_parity"] = jsonable(rp)
    status = {k: (v.get("status") if isinstance(v, dict) else None) for k, v in report["checks"].items()}
    counts_all, flagged = walk_flags(report["checks"])
    top = {}
    for s_ in status.values():
        top[s_] = top.get(s_, 0) + 1
    report["summary"].update({"status_by_check": status, "top_level_counts": top, "status_counts_all_nodes": counts_all,
                              "flagged_fail_warn": flagged,
                              "overall": worst(*[s_ for s_ in status.values() if s_ not in ("PENDING", "INFO")],
                                               report["independence"]["status"])})
    loaded = sorted(m for m in sys.modules if m.split(".")[0] in _FORBIDDEN_TOP)
    report["restore_refresh"] = {"at": iso(datetime.now(timezone.utc)), "verifier_sha256": sha_file(Path(__file__)),
                                 "status": rp["status"], "wall_s": round(time.time() - t0, 2),
                                 "cpu_s": round(time.process_time() - c0, 2), "forbidden_modules_loaded": loaded,
                                 "note": "only restore_parity was recomputed (after the incremental copy refresh)"}
    text = json.dumps(jsonable(report), indent=1, allow_nan=False)
    scrub_check(text)
    if loaded:
        raise RuntimeError("independence violated")
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(text + "\n")
    tmp.replace(dest)
    print(json.dumps({"restore_parity": rp["status"], "overall": report["summary"]["overall"]}))


if __name__ == "__main__":
    main()
