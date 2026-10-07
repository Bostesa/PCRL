#!/usr/bin/env python3
"""Independent verifier for the learned-decoder constrained-release study (lcr).

Owner: role E (independent verifier; prompt section 14). Exclusive files: this script,
results/pcrl_learned_decoder_constrained_release_v1/INDEPENDENT_VERIFICATION.json and
results/pcrl_learned_decoder_constrained_release_v1/FIXTURE_ORACLE_REPORT.md.

Provenance of the code. ADAPTED (not rewritten) from the cbp independent verifier
results/pcrl_confidence_budgeted_privacy_v1/verification/replay_cbp.py at the cbp final tip 7f3ec67 (sha256
bfac817d4db32da0cba7e84c91d7ef7368e2cc95979213af0a12a5bd66b171a2; itself adapted from replay_qpc.py): the import guard,
role reconstruction, own forward pass, smoothing / KL / KL k-means, the own coarse-state search engine, plug-in MI,
utility, AUC, the exact-record-group bootstrap, inner-audit / composition replay, attacker refits, lock chronology and
the eligibility / role / label machinery are kept and re-targeted. NEW here, written from the lcr prompt (sections
6-14) and to be reconciled against the lcr registrations (PROTOCOL / SEARCH_RULES / FIXTURE_LAWS / FIXTURE_GATE_RULE /
SELECTION_RULES / LABEL_TRUTH_TABLE / PRIMARY_FAMILY) as the lead writes them:
  * an INDEPENDENT D1 decoder (section 6): dual water-filling in the simplex multiplier nu with star-order pooling of
    the class-dominance constraints (closed-form quadratic roots for every free coordinate, Brent roots for nu and for
    each pooled block), a deterministic monotone renormalisation, and a solver-free certificate on the ACTUAL released
    q: primal residuals, exact class dominance, strict float64 argmax of q, the Frank-Wolfe duality gap over the
    vertices of the class-dominant simplex (a proof of epsilon-optimality that does not rely on any solver status),
    KKT stationarity with recovered multipliers, and the affine smoothing identity; cross-checked in the self-tests
    against scipy SLSQP and against exhaustive enumeration of all 3^(K-1) KKT active patterns;
  * an exact-law fixture oracle (section 9): canonical same-class partitions (restricted-growth strings under the
    per-class state caps), integer-count laws at N = 4096, exact plug-in MI from integer tables, D0 / D1 decoders from
    exact expected counts, exact expected log loss / Brier / accuracy, exhaustive feasible references and the gate;
  * the lcr fitting budgets (L_i <= L_i(U) + 0.005, B_i <= B_i(U) + 0.003, local caps I_i <= I_i(C-TASK)) and the
    sequential / paired-move feasibility rules (section 7-8);
  * the lcr roles T*, P*, C*, N*, C_pair*, J*, Q with the ORIGINAL every-seed eligibility (no cbp headroom buffer),
    guards, ordering, fallbacks (section 10) and the 37-slot claims A / B / C / Q with z = Phi^-1(1 - 0.05/74),
    bootstrap seed 20261009 (section 12);
  * the deliberate-defect tests of section 14, each shown to be CAUGHT on synthetic data.

Independence:
  * a sys.meta_path guard refuses every import under lcr, cbp, qpc, dpc, osf, smf, rgj, jcv, pnx, oar,
    stored_model_eval and pcrl, including imports triggered while unpickling joblib heads; torch.load always uses
    weights_only=True; the run asserts at the end that none of these packages was loaded;
  * the worktree root is removed from sys.path; the shared semaphore is invoked BY PATH as the parent process with -P
    (OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python -P <WORKTREE>/lcr/sema.py --label E:<what> -- env OMP_NUM_THREADS=1
    OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ~/PCRL/.venv/bin/python <this script> ...), never imported;
  * libraries: numpy, scipy, scikit-learn, torch, joblib and the standard library;
  * study code is READ only to learn file formats and registered rules; nothing of it is executed in-process.

Phases (status PASS / FAIL / WARN / INFO / PENDING):
  PHASE_0  scaffold + synthetic self-tests (own D1 decoder + certificate, fixture oracle, selection / labels, the
           section-14 deliberate defects); no real data is read
  PHASE_1  fixture gate replay (FIXTURE_GATE.json: exhaustive references, arm values, gate verdict, route) and source
           admission (hashes vs the cbp store and the cbp copy SHA256SUMS, teacher forward parity, roles and groups,
           D0 release re-encode)
  PHASE_2  D1 decoders of the fixed maps and of the new units re-solved and certified, deployed fitting budgets and
           local caps, incremental loss / MI parity, decision preservation, inner units re-scored, composition coverage,
           eligibility / guards / aliases / fallbacks / selection -- before the EVALUATION_LOCK
  PHASE_3  EVALUATION_LOCK (on origin, checked before any assessment label), outer units, the 37 endpoints and labels,
           tables / figures, attacker refits, deployment bindings / refusals (external process), restore from the
           backup copy, SEMA_LOG budget / process audit

Usage (from the worktree root; every run, including PHASE_0, under the semaphore):
    OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python -P <WORKTREE>/lcr/sema.py --label E:phase0 -- \\
        env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ~/PCRL/.venv/bin/python \\
        results/pcrl_learned_decoder_constrained_release_v1/verification/replay_lcr.py --phase 0
Writes results/pcrl_learned_decoder_constrained_release_v1/INDEPENDENT_VERIFICATION.json (aggregates, hashes and
placeholders only; finite JSON) unless --no-write.
"""


from __future__ import annotations

import importlib
import importlib.abc
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

# ------------------------------------------------------------------------------------------------ import guard
_FORBIDDEN_TOP = ("lcr", "cbp", "qpc", "dpc", "osf", "smf", "rgj", "jcv", "pnx", "oar", "stored_model_eval", "pcrl")
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
REL_RES = "results/pcrl_learned_decoder_constrained_release_v1"
OUT = RES / "INDEPENDENT_VERIFICATION.json"
ORACLE_REPORT = RES / "FIXTURE_ORACLE_REPORT.md"
CACHE = Path.home() / "PCRL_eval_cache_private"
SRC = CACHE / "jcv_v1" / "inputs" / "adult_jcv.npz"
SRC_SHA = "e0d9e54af780f30788ee29cfe6795ec82cbdcadc127b1978c69a3891485d2f12"
PRIV = CACHE / "lcr_v1"
RUN = PRIV / "run"
UNITS = RUN / "units"
ADM = PRIV / "admitted"
CBP_PRIV = CACHE / "cbp_v1"                                     # source store (read only)
CBP_RUN = CBP_PRIV / "run"
CBP_UNITS = CBP_RUN / "units"
CBP_ADM = CBP_PRIV / "admitted"
CBP_COPY = CACHE / "cbp_v1_local_copy_20261006"                # cbp verified same-device copy (SHA256SUMS)
CBP_REL = "results/pcrl_confidence_budgeted_privacy_v1"
CBP_TIP = "7f3ec67b2ecd86d474e2ff27167091af9923f572"           # cbp pinned final commit = this branch's base
QPC_PRIV = CACHE / "qpc_v1"
QPC_UNITS = QPC_PRIV / "run" / "units"
QPC_ADM = QPC_PRIV / "admitted"
QPC_COPY = CACHE / "qpc_v1_local_copy_20261006_v2"
DPC_PRIV = CACHE / "dpc_v1"
DPC_UNITS = DPC_PRIV / "run" / "units"
DPC_ADMITTED = DPC_PRIV / "admitted"
ADMISSION_NORM = WT / "results" / "pcrl_joint_complete_view_method_v1" / "DATA_ADMISSION.json"
BRANCH = "research/pcrl-learned-decoder-constrained-release-v1"
QPC_EVIDENCE = "9dd06da6b64e558e1c079f76e43982b60b327e63"      # qpc evidence commit (source of every admitted unit)
QPC_TIP = "d0c8a45c879d01fb8b736ccc091ec3e2c3e9b351"           # qpc handoff tip = this branch's base
QPC_REL = "results/pcrl_confidence_capacity_v1"
SOURCE_SHA = "0a7b05a52746544213742f50efd0a48167efffb1"        # dpc evidence (roles, teacher units, references)
TEACHER_PIN = "925e0fddfcb666116c6179575339728a324ed78e"       # osf teacher provenance
DPC_REL = "results/pcrl_decision_preserving_compression_v1"
OSF_REL = "results/pcrl_online_strength_frontier_v1"
STUDY_START = "2026-10-06T23:42:46Z"                  # lcr session start (TEAM_PLAN)
CBP_STUDY_START = "2026-10-06T16:44:45Z"

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
B_BOOT, BOOT_SEED = 1999, 20261009                 # lcr: new fixed bootstrap seed (section 12)
MARGIN_PAIR, MARGIN_LOCAL = 0.02, 0.01

STATUS_RANK = {"FAIL": 4, "PENDING": 3, "WARN": 2, "PASS": 1, "INFO": 0, "NOT_APPLICABLE": 0}

# ---------------------------------------------------------------- lcr registered constants (prompt sections 6-12)
KAPPA = 32.0                 # teacher pseudo-observations per occupied token (fixed; not tuned)
BRIER_COEF = 0.5             # coefficient of the summed Brier term in the per-token decoder objective (fixed)
FIT_BUDGET = {"ll": 0.005, "brier": 0.003}          # fitting budgets: L_i <= L_i(U) + 0.005, B_i <= B_i(U) + 0.003
FIXTURE_N = 4096             # fixture effective fitting size; atom probabilities are integer multiples of 1/4096
FIXTURE_MAX_PAIRS = 100_000  # canonical mapping pairs enumerated per fixture (cap)
GATE_PAIR_MI = 0.01          # trigger: pair MI reduction (nats) beyond the strongest feasible task-only D1 compression
GATE_ACC_GAIN = 0.03         # trigger: each task's accuracy gain over its constant
LCR_SWEEPS = 5
LCR_TARGETS = (4, 4)         # four nearest decoded-probability targets + four best objective targets per fine cell
PAIR_PROPOSALS = 8           # retained proposals per recipient for the paired step (4 by one-sided Phi, 4 by task loss)
Z_LCR = 3.2048452050105634   # NormalDist().inv_cdf(1 - 0.05/74)
# registered verifier tolerances for the own D1 solve and certificate (frozen in PHASE_0, before any study output):
#   q agreement (stored vs own)          max |dq| <= 1e-9 (absolute)
#   smoothing identity of a stored q     max |q - (u + eps 1 + eps e_d)/(1 + (K+1) eps)| <= 4e-16
#   simplex sum of u                     |sum u - 1| <= 1e-12
#   class dominance / strict argmax      EXACT in float64 (u_d >= u_k; q_d > q_k for every k != d)
#   Frank-Wolfe duality gap              gap / max(1, n_t + kappa) <= 1e-10
#   KKT stationarity (recovered nu)      max residual / max(1, n_t + kappa) <= 1e-8
#   objective agreement                  |f(stored) - f(own)| / max(1, n_t + kappa) <= 1e-10
#   sufficient statistics                n_t, y_t exact (integers); teacher sums |dS| <= 1e-9 * max(1, n_t)
#   fitting losses / MI (summation order) 1e-12 absolute on means; budgets exact on the stored margin sign with a
#                                        1e-12 borderline band reported (never silently flipped)
D1_TOL = {"q": 1e-9, "smoothing_identity": 4e-16, "sum": 1e-12, "fw_gap_rel": 1e-10, "stationarity_rel": 1e-8,
          "objective_rel": 1e-10, "teacher_sum_rel": 1e-9, "loss_mean": 1e-12, "budget_band": 1e-12, "mi": 1e-12}

# own full-row releases of every replayed code: OWN_REL[(seed, cid)] = {tok1, q1, hard1, tok2, q2, hard2, alpha1, alpha2}
OWN_REL: dict = {}
# every correction made to this verifier's own code (kept; disclosed in the report)
VERIFIER_CORRECTIONS: list = [
    {"phase": "PHASE_0", "item": "cid_lam (adapted cbp helper)", "change": "parse the 'l<value>' segment after the rate; "
     "the cbp split('|l') form raised on lcr IDs with a trailing '|D1'"},
    {"phase": "PHASE_0", "item": "SLSQP cross-check", "change": "compare the FEASIBLE projection of the SLSQP point: SLSQP "
     "ends infeasible by ~1e-7 on tied dominance constraints, which made its raw objective look lower than the own "
     "certified optimum"},
    {"phase": "PHASE_0", "item": "gradient self-test", "change": "Richardson-extrapolated central differences at an "
     "interior point (plain h = 1e-6 differences were truncation-limited near small q)"},
    {"phase": "PHASE_0", "item": "projection self-test", "change": "the planted 1e-10 dominance residual now preserves the "
     "simplex sum (the first draft perturbed the sum, inflating the reported projection magnitude)"},
    {"phase": "PHASE_0", "item": "roles / labels", "change": "reconciled with SELECTION_RULES.json and "
     "LABEL_TRUTH_TABLE.json at a94e944: C_pair* = every scored candidate except JOINT-PAIR (87); Q == PASS is checked "
     "before the INCOMPLETE rule; P* named by the simplest family / construction over exact (tokens AND q) aliases"},
]
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
    """lambda segment 'l<value>' after the rate (lcr IDs may carry a trailing '|D1'; corrected for lcr)."""
    for part in cid.split("|")[3:]:
        if part.startswith("l"):
            return float(part[1:])
    return None


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
    """Other heavy study processes visible now (python processes whose command line names lcr / the semaphore), for the
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
        if "python" in cmd and ("-m lcr" in cmd or "lcr/sema.py" in cmd or "replay_lcr" in cmd):
            out.append({"pid": int(pid), "kind": "sema" if "sema" in cmd else ("verifier" if "replay_lcr" in cmd
                                                                               else "lcr")})
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
    if ok and manifests.get("lcr_ROLE_MANIFEST") is None:
        st = "WARN"
        out["note"] = "lcr ROLE_MANIFEST.json not present; checked against the pinned cbp manifest only"
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


# ================================================================================================ lcr: own D1 decoder
# Written from prompt section 6 only (no study code). For a token with fitting count n, true-label counts y (length K),
# teacher-probability sum S (pbar = S / n) and teacher-predicted class d, minimise over the class-dominant simplex
#     U_d = {u : u >= 0, sum u = 1, u_d >= u_k for every k}
# the registered objective at the ACTUAL released vector q = (u + eps 1 + eps e_d) / Z, Z = 1 + (K+1) eps:
#     F(q) = sum_k y_k (-log q_k) + 0.5 * sum_rows ||q - e_Y||^2 + kappa * KL(pbar || q).
# Varying part: f(q) = -sum_k a_k log q_k + (n/2) ||q||^2 - y.q with a = y + kappa * pbar; grad_u f = (-a/q + n q - y)/Z.
# f is strictly convex in q (Hessian diag(a / q^2) + n I, n > 0) and q is an injective affine image of u, so the token
# problem has a unique minimiser. Own algorithm (dual water-filling): for a simplex multiplier nu, every free coordinate
# solves g_k = nu in closed form (positive root of n q^2 - (y_k + Z nu) q - a_k = 0); the star-order dominance
# constraints are pooled (children of d with the largest unconstrained value are tied to d while they exceed the pooled
# value, which solves sum_{j in B} g_j(v) = |B| nu); the mass M(nu) = sum u(nu) is continuous and nondecreasing and
# M(nu) = 1 is solved by Brent's method. The result is renormalised by its float sum (monotone rounding keeps u >= 0 and
# u_d >= u_k exactly), and certified on the RELEASED q without reference to any solver status.
from scipy.optimize import brentq as _brentq  # noqa: E402

_MACH = float(np.finfo(np.float64).eps)


def d1_Z(K, eps=EPS):
    return 1.0 + (K + 1) * eps


def d1_q_of_u(u, d, eps=EPS):
    """The registered affine smoothing (u + eps 1 + eps e_d) / (1 + (K+1) eps) (float64)."""
    q = np.asarray(u, dtype=np.float64) + eps
    q[d] += eps
    return q / d1_Z(q.shape[-1], eps)


def d1_parts(n, y, S, kappa=KAPPA):
    n = float(n)
    y = np.asarray(y, dtype=np.float64)
    pbar = np.asarray(S, dtype=np.float64) / n
    return n, y, pbar, y + kappa * pbar


def d1_objective(q, n, y, S, kappa=KAPPA):
    """Registered per-token objective (full form, constants included) at a released q."""
    n, y, pbar, _ = d1_parts(n, y, S, kappa)
    q = np.asarray(q, dtype=np.float64)
    ll = float(np.sum(y * -np.log(q)))
    br = n * float(q @ q) - 2.0 * float(y @ q) + n
    m = pbar > 0
    kl = float(np.sum(pbar[m] * (np.log(pbar[m]) - np.log(q[m]))))
    return ll + BRIER_COEF * br + kappa * kl


def d1_f(q, n, y, S, kappa=KAPPA):
    """Varying part f(q) = -sum a log q + (n/2)||q||^2 - y.q."""
    n, y, _, a = d1_parts(n, y, S, kappa)
    q = np.asarray(q, dtype=np.float64)
    return float(-(a @ np.log(q)) + 0.5 * n * (q @ q) - y @ q)


def d1_grad_u(q, n, y, S, kappa=KAPPA):
    n, y, _, a = d1_parts(n, y, S, kappa)
    q = np.asarray(q, dtype=np.float64)
    return (-a / q + n * q - y) / d1_Z(len(q))


def _qroot(b, a, n):
    """Positive root of n q^2 - b q - a = 0 (a >= 0, n > 0), scalar, cancellation-free."""
    disc = math.sqrt(b * b + 4.0 * n * a)
    if b >= 0:
        return (b + disc) / (2.0 * n)
    return 2.0 * a / (disc - b) if a > 0 else 0.0


def own_d1_solve(n, y, S, d, kappa=KAPPA, eps=EPS, drop_dominance=False, override_a=None):
    """Own D1 solve. Returns (u, q, info). drop_dominance / override_a exist ONLY for the deliberate-defect tests."""
    K = len(y)
    if not float(n) > 0:
        raise ValueError("an empty token has no supervised statistics (pinned fallback applies)")
    n_, y_, pbar, a = d1_parts(n, y, S, kappa)
    if override_a is not None:
        a = np.asarray(override_a, dtype=np.float64)
    Z = d1_Z(K, eps)
    off = [eps] * K
    off[d] = 2.0 * eps
    al, yl = [float(x) for x in a], [float(x) for x in y_]
    others = [k for k in range(K) if k != d]
    cnt = {"M": 0, "inner": 0}

    def g_at(j, v):
        q = (v + off[j]) / Z
        if q <= 0:
            return -math.inf if al[j] > 0 else (-yl[j]) / Z
        return (-al[j] / q + n_ * q - yl[j]) / Z

    def pooled(nu):
        uf = [Z * _qroot(yl[j] + Z * nu, al[j], n_) - off[j] for j in range(K)]
        block, v = [d], uf[d]
        if not drop_dominance:
            for k in sorted(others, key=lambda k_: (-uf[k_], k_)):
                if not uf[k] > v:
                    break
                block.append(k)
                B, m = tuple(block), len(block)

                def H(vv, B=B, m=m):
                    cnt["inner"] += 1
                    return math.fsum(g_at(j, vv) for j in B) - m * nu
                lo, hi = v, uf[k]
                hl, hh = H(lo), H(hi)
                if hl >= 0:
                    v = lo
                elif hh <= 0:
                    v = hi
                else:
                    v = _brentq(H, lo, hi, xtol=1e-18, rtol=4 * _MACH, maxiter=500)
        u = [max(x, 0.0) for x in uf]
        vv = max(v, 0.0)
        for j in block:
            u[j] = vv
        return u, block

    def M(nu):
        cnt["M"] += 1
        return math.fsum(pooled(nu)[0]) - 1.0

    q0 = 1.0 / (2 * K)
    nu_lo = min(g_at(j, Z * q0 - off[j]) for j in range(K)) - 1.0
    nu_hi = max(g_at(j, Z * 1.0 - off[j]) for j in range(K)) + 1.0
    flo, fhi = M(nu_lo), M(nu_hi)
    it = 0
    while flo > 0 and it < 60:
        nu_lo -= max(1.0, abs(nu_lo))
        flo, it = M(nu_lo), it + 1
    while fhi < 0 and it < 120:
        nu_hi += max(1.0, abs(nu_hi))
        fhi, it = M(nu_hi), it + 1
    if not (flo <= 0 <= fhi):
        raise RuntimeError("own D1 bracket failed")
    nu = _brentq(M, nu_lo, nu_hi, xtol=1e-15 * max(1.0, n_ + kappa), rtol=4 * _MACH, maxiter=1000)
    u, block = pooled(nu)
    u = np.asarray(u, dtype=np.float64)
    s_ = float(np.sum(u))
    u_n = u / s_
    q = d1_q_of_u(u_n, d, eps)
    info = {"nu": float(nu), "tied": sorted(int(k) for k in block if k != d),
            "zero": sorted(int(k) for k in others if u_n[k] == 0.0), "renorm_abs": abs(s_ - 1.0),
            "renorm_max_du": float(np.max(np.abs(u_n - u))), "mass_evals": cnt["M"], "inner_evals": cnt["inner"]}
    return u_n, q, info


def d1_certificate(u, q, n, y, S, d, kappa=KAPPA, eps=EPS, tol=None):
    """Solver-free certificate of a released vector: affine identity, primal feasibility (sum, nonnegativity, EXACT
    dominance), strict float64 argmax of the RELEASED q, Frank-Wolfe duality gap over the vertices 1_B/|B| (B contains d)
    of the class-dominant simplex (f(u) - f* <= gap for convex f), KKT stationarity with recovered multipliers."""
    tol = tol or D1_TOL
    u = np.asarray(u, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    K = len(u)
    n_, _, _, _ = d1_parts(n, y, S, kappa)
    scale = max(1.0, n_ + kappa)
    others = [k for k in range(K) if k != d]
    ident = float(np.max(np.abs(q - d1_q_of_u(u, d, eps))))
    nonneg = bool((u >= 0).all())
    ssum = abs(math.fsum(u.tolist()) - 1.0)
    dom = all(u[d] >= u[k] for k in others)
    strict = bool(all(q[d] > q[k] for k in others) and int(np.argmax(q)) == d and (q > 0).all())
    if not (q > 0).all():
        return {"ok": False, "reason": "nonpositive released probability", "strict_argmax": False}
    g = d1_grad_u(q, n, y, S, kappa)
    gs = sorted(float(g[k]) for k in others)
    best, acc = float(g[d]), float(g[d])
    for m, gk in enumerate(gs, 1):
        acc += gk
        best = min(best, acc / (m + 1))
    gap = float(math.fsum((g * u).tolist())) - best
    thr = 1e-9
    zero = [k for k in others if u[k] <= thr]
    tied = [k for k in others if k not in zero and u[d] - u[k] <= thr]
    free = [k for k in others if k not in zero and k not in tied]
    blk = [d] + tied
    nu = (math.fsum(float(g[k]) for k in free) + math.fsum(float(g[j]) for j in blk)) / (len(free) + len(blk))
    r = [abs(float(g[k]) - nu) for k in free]
    r.append(abs(math.fsum(float(g[j]) for j in blk) - len(blk) * nu))
    r += [max(0.0, nu - float(g[k])) for k in zero]
    r += [max(0.0, float(g[k]) - nu) for k in tied]
    stat = max(r) / scale
    ok = (ident <= tol["smoothing_identity"] and nonneg and ssum <= tol["sum"] and dom and strict
          and gap / scale <= tol["fw_gap_rel"] and stat <= tol["stationarity_rel"])
    return {"ok": bool(ok), "identity_max_abs": ident, "nonnegative": nonneg, "sum_residual": ssum, "dominance_exact": dom,
            "strict_argmax": strict, "fw_gap": gap, "fw_gap_rel": gap / scale, "stationarity_rel": stat,
            "active": {"tied": tied, "zero": zero, "free": free}, "nu": nu,
            "min_margin_q": float(min(q[d] - q[k] for k in others)) if others else None,
            "objective": d1_objective(q, n, y, S, kappa)}


def slsqp_d1(n, y, S, d, kappa=KAPPA):
    """Independent trusted cross-check (self-tests and samples): scipy SLSQP in u-space with bounds, the simplex equality
    and the dominance inequalities; analytic gradient."""
    from scipy.optimize import minimize
    K = len(y)

    def f(u):
        return d1_f(d1_q_of_u(np.clip(u, 0.0, 1.0), d), n, y, S, kappa)

    def jac(u):
        return d1_grad_u(d1_q_of_u(np.clip(u, 0.0, 1.0), d), n, y, S, kappa)
    cons = [{"type": "eq", "fun": lambda u: np.sum(u) - 1.0, "jac": lambda u: np.ones(K)}]
    for k in range(K):
        if k != d:
            e = np.zeros(K)
            e[d], e[k] = 1.0, -1.0
            cons.append({"type": "ineq", "fun": (lambda u, e=e: float(e @ u)), "jac": (lambda u, e=e: e)})
    r = minimize(f, np.full(K, 1.0 / K), jac=jac, bounds=[(0.0, 1.0)] * K, constraints=cons, method="SLSQP",
                 options={"ftol": 1e-15, "maxiter": 3000})
    infeas = float(max([0.0] + [r.x[k] - r.x[d] for k in range(K) if k != d] + [-float(r.x.min()), abs(r.x.sum() - 1)]))
    u = proj_class_simplex(r.x, d)              # SLSQP may end slightly infeasible: compare its FEASIBLE projection
    return u, d1_q_of_u(u, d), infeas


def kkt_enumerate_d1(n, y, S, d, kappa=KAPPA, eps=EPS, tol=1e-9):
    """Exhaustive KKT active-pattern enumeration (self-tests): every non-d class is ZERO, TIED to d or FREE (3^(K-1)
    patterns); each pattern is solved as an equality-constrained problem (nested 1-D roots) and kept iff it satisfies
    primal and dual feasibility. Returns the list of KKT points (q vectors)."""
    import itertools
    K = len(y)
    n_, y_, _, a = d1_parts(n, y, S, kappa)
    Z = d1_Z(K, eps)
    off = np.full(K, eps)
    off[d] = 2 * eps
    others = [k for k in range(K) if k != d]

    def g_at(j, v):
        q = (v + off[j]) / Z
        if q <= 0:
            return -math.inf
        return (-a[j] / q + n_ * q - y_[j]) / Z
    pts = []
    for pat in itertools.product("ZTF", repeat=K - 1):
        T = [k for k, p in zip(others, pat) if p == "T"]
        F = [k for k, p in zip(others, pat) if p == "F"]
        Zs = [k for k, p in zip(others, pat) if p == "Z"]
        B = [d] + T

        def vblock(nu):
            H = lambda v: sum(g_at(j, v) for j in B) - len(B) * nu  # noqa: E731
            if H(1e-300) >= 0:
                return 0.0
            if H(1.0) <= 0:
                return 1.0
            return _brentq(H, 1e-300, 1.0, xtol=1e-18, rtol=4 * _MACH, maxiter=500)

        def mass(nu):
            v = vblock(nu)
            uf = [Z * _qroot(y_[k] + Z * nu, a[k], n_) - off[k] for k in F]
            return len(B) * v + sum(uf) - 1.0
        lo, hi = -1e3 * (n_ + kappa) - 10, 1e3 * (n_ + kappa) + 10
        try:
            nu = _brentq(mass, lo, hi, xtol=1e-15 * (n_ + kappa), rtol=4 * _MACH, maxiter=1000)
        except ValueError:
            continue
        v = vblock(nu)
        u = np.zeros(K)
        u[B] = v
        for k in F:
            u[k] = Z * _qroot(y_[k] + Z * nu, a[k], n_) - off[k]
        if v <= 0 or any(u[k] < -tol or u[k] > v + tol for k in F):
            continue
        if any(g_at(k, 0.0) < nu - tol * (n_ + kappa) for k in Zs):
            continue
        if any(g_at(k, v) > nu + tol * (n_ + kappa) for k in T):
            continue
        u = np.clip(u, 0, None)
        pts.append({"pattern": "".join(pat), "q": d1_q_of_u(u / u.sum(), d)})
    return pts


def proj_class_simplex(w, d):
    """Exact Euclidean projection onto the class-dominant simplex {u >= 0, sum u = 1, u_d >= u_k} (own derivation: the
    pooled block is nu-independent for a quadratic; the remaining coordinates follow the sorted simplex rule)."""
    w = np.asarray(w, dtype=np.float64)
    K = len(w)
    kids = sorted((k for k in range(K) if k != d), key=lambda k: (-w[k], k))
    B, sB = [d], float(w[d])
    for k in kids:
        if w[k] > sB / len(B):
            B.append(k)
            sB += float(w[k])
        else:
            break
    rest = [k for k in kids if k not in B]
    act = []
    nu = (1.0 - sB) / len(B)
    for k in rest:
        if w[k] + nu > 0:
            act.append(k)
            nu = (1.0 - sB - math.fsum(float(w[j]) for j in act)) / (len(B) + len(act))
        else:
            break
    u = np.zeros(K)
    u[B] = sB / len(B) + nu
    for k in act:
        u[k] = w[k] + nu
    return u


def proj_registered(u, d):
    """Own implementation of the REGISTERED frozen repair (PROTOCOL section 6 / METHOD_CARD): clip at 0, cap every
    k != d at u_d, renormalise by the k-ordered float sum. Returns (u_out, magnitude max |u_out - u_in|)."""
    u_in = np.asarray(u, dtype=np.float64)
    w = np.maximum(u_in, 0.0)
    for k in range(len(w)):
        if k != d and w[k] > w[d]:
            w[k] = w[d]
    tot = 0.0
    for k in range(len(w)):
        tot += float(w[k])
    out = w / tot
    return out, float(np.max(np.abs(out - u_in)))


def suff_hash(K, d, n, y, S, kappa=KAPPA, eps=EPS):
    """Own sufficient-statistic key (every quantity the solve depends on; exact bytes)."""
    h = hashlib.sha256()
    h.update(json.dumps({"K": int(K), "d": int(d), "n": float(n), "kappa": float(kappa), "eps": float(eps)},
                        sort_keys=True).encode())
    h.update(np.ascontiguousarray(np.asarray(y, dtype=np.float64)).tobytes())
    h.update(np.ascontiguousarray(np.asarray(S, dtype=np.float64)).tobytes())
    return h.hexdigest()


class D1Cache:
    """Own solve cache keyed by the COMPLETE sufficient statistic (never by counts alone)."""

    def __init__(self, key_fn=None):
        self.key_fn = key_fn or (lambda K, d, n, y, S: suff_hash(K, d, n, y, S))
        self.c, self.hits, self.miss = {}, 0, 0

    def solve(self, n, y, S, d):
        k = self.key_fn(len(y), d, n, y, S)
        if k in self.c:
            self.hits += 1
            return self.c[k]
        self.miss += 1
        u, q, info = own_d1_solve(n, y, S, d)
        self.c[k] = (u, q, info)
        return self.c[k]


# ------------------------------------------------------------------------------------------------ lcr: fitting rows
def token_stats(tok, y, P, K, ntok=None):
    """n_t, y_t (true-label counts) and S_t (teacher-probability sums, sequential row order) of each token."""
    tok = np.asarray(tok, dtype=np.int64)
    G = int(ntok if ntok is not None else tok.max() + 1)
    n = np.bincount(tok, minlength=G).astype(np.float64)
    Y = np.zeros((G, K))
    np.add.at(Y, (tok, np.asarray(y, dtype=np.int64)), 1.0)
    S = np.zeros((G, K))
    np.add.at(S, tok, np.asarray(P, dtype=np.float64))
    return n, Y, S


def token_class(tok, dec, G):
    """Teacher-predicted class common to each token (refuses class mixing); -1 for an empty token."""
    tok, dec = np.asarray(tok, np.int64), np.asarray(dec, np.int64)
    cls = np.full(G, -1, dtype=np.int64)
    for t_, c_ in zip(tok.tolist(), dec.tolist()):
        if cls[t_] == -1:
            cls[t_] = c_
        elif cls[t_] != c_:
            raise ValueError(f"token {t_} mixes predicted classes")
    return cls


def own_d1_release(tok, y, P, dec, K, cache=None, fallback=None):
    """Own D1 decoded table for one recipient on the fitting rows: per-token (u, q) from the own solve; empty tokens take
    the pinned fallback (`fallback[t]`, required)."""
    tok = np.asarray(tok, dtype=np.int64)
    G = int(tok.max() + 1) if fallback is None else max(int(tok.max() + 1), len(fallback))
    n, Y, S = token_stats(tok, y, P, K, G)
    cls = token_class(tok, dec, G)
    cache = cache or D1Cache()
    Qt = np.zeros((G, K))
    Ut = np.zeros((G, K))
    for t_ in range(G):
        if n[t_] == 0:
            if fallback is None:
                raise ValueError("empty token without a pinned fallback")
            Qt[t_] = fallback[t_]
            Ut[t_] = np.nan
            continue
        Ut[t_], Qt[t_], _ = cache.solve(n[t_], Y[t_], S[t_], int(cls[t_]))
    return {"n": n, "y": Y, "S": S, "cls": cls, "u": Ut, "q": Qt}


def fit_losses(Q_rows, y):
    """Source-convention true-label log loss (clip 1e-12) and multiclass Brier, means over rows (float64)."""
    u = my_utility(Q_rows, np.argmax(Q_rows, 1), y, Q_rows.shape[1], 0)
    return u["logloss"], u["brier"]


def budget_check(L, B, LU, BU, budget=None):
    """Fitting budgets in the registered margin form (U value + allowance) - value >= 0 (inclusive); the borderline band
    lists verdicts within 1e-12 of the boundary (reported, never flipped)."""
    b = budget or FIT_BUDGET
    mL, mB = (LU + b["ll"]) - L, (BU + b["brier"]) - B
    return {"ok": bool(mL >= 0 and mB >= 0), "ll_ok": bool(mL >= 0), "brier_ok": bool(mB >= 0), "ll_margin": mL,
            "brier_margin": mB, "borderline": [x for x, m_ in (("ll", mL), ("brier", mB)) if abs(m_) <= D1_TOL["budget_band"]]}


def phi_of(I1, I2, I12):
    return I12 + 0.5 * (I1 + I2)


class LcrEngine:
    """From-scratch evaluation of a pair of coarse maps (fine cell -> canonical token) on fitting rows (own arithmetic):
    D1 decoders, true-label losses, plug-in MI of SEX with full token identities, Phi, T and the feasibility rules."""

    def __init__(self, f, y, P, dec, s, K, cache=None):
        self.f = {i: np.asarray(f[i], np.int64) for i in (1, 2)}
        self.y = {i: np.asarray(y[i], np.int64) for i in (1, 2)}
        self.P = {i: np.asarray(P[i], np.float64) for i in (1, 2)}
        self.dec = {i: np.asarray(dec[i], np.int64) for i in (1, 2)}
        self.s = np.asarray(s, np.int64)
        self.K = K
        self.cache = cache or D1Cache()
        self.U = {i: fit_losses(self.P[i], self.y[i]) for i in (1, 2)}

    def recipient(self, i, cmap, decoder="D1"):
        tok = np.asarray(cmap, np.int64)[self.f[i]]
        _, tok = np.unique(tok, return_inverse=True)
        r = own_d1_release(tok, self.y[i], self.P[i], self.dec[i], self.K[i], self.cache)
        if decoder == "D0":
            r["q"] = smooth(r["S"] / r["n"][:, None], r["cls"])
        Qr = r["q"][tok]
        L, B = fit_losses(Qr, self.y[i])
        return {"tok": tok, "L": L, "B": B, "I": mi_of(self.s, tok), "alpha": int(tok.max() + 1), "dec": r}

    def pair(self, m1, m2, decoder="D1"):
        r1, r2 = self.recipient(1, m1, decoder), self.recipient(2, m2, decoder)
        I12 = mi_of(self.s, r1["tok"], r2["tok"])
        return {"r1": r1, "r2": r2, "L1": r1["L"], "B1": r1["B"], "I1": r1["I"], "L2": r2["L"], "B2": r2["B"],
                "I2": r2["I"], "I12": I12, "Phi": phi_of(r1["I"], r2["I"], I12),
                "T": r1["L"] + r2["L"] + 0.5 * (r1["B"] + r2["B"])}

    def feasible(self, i, r, cap=None):
        bc = budget_check(r["L"], r["B"], *self.U[i])
        return bool(bc["ok"] and (cap is None or r["I"] <= cap)), bc


def seq_stage1_feasible(eng: "LcrEngine", first, r_first, r_partner, cap_first=None, partner_required=False):
    """Registered sequential temporary stage: ONLY the first recipient's utility (and local) constraints apply; the
    CLASS-ONLY partner is an intermediate counterpart and is NOT required to pass its own budget.
    partner_required=True is the deliberate defect."""
    ok, _ = eng.feasible(first, r_first, cap_first)
    if partner_required:
        ok_p, _ = eng.feasible(3 - first, r_partner)
        ok = ok and ok_p
    return ok


def release_is_token_function(tok, q):
    """A released probability vector must be a deterministic function of the token identity: every row of a token
    carries a bitwise-identical q (a clean-output bypass breaks this)."""
    tok = np.asarray(tok, np.int64)
    q = np.asarray(q, np.float64)
    first = {}
    for i, t_ in enumerate(tok.tolist()):
        j = first.setdefault(t_, i)
        if j != i and not np.array_equal(q[i], q[j]):
            return False
    return True


# ------------------------------------------------------------------------------------------------ lcr: fixture oracle
# Exact-law oracle (prompt section 9). A law is a list of integer-count atoms summing to N = 4096: each atom carries the
# fine cell of each recipient, the teacher probability vector of each recipient (rationals or floats), the true task
# label of each recipient and the sensitive label. Every fine cell lies inside one teacher-predicted class. A recipient
# map is a canonical same-class partition of its fine cells (restricted-growth strings per class, at most cap blocks);
# tokens are ordered by (class, block). Every quantity is computed from the exact integer tables of the law.
def rgs_partitions(m, cap):
    """All restricted-growth strings of length m with at most `cap` blocks (canonical set partitions)."""
    out = []

    def rec(prefix, mx):
        if len(prefix) == m:
            out.append(tuple(prefix))
            return
        for b in range(min(mx + 2, cap)):
            rec(prefix + [b], max(mx, b))
    if m == 0:
        return [()]
    rec([0], 0)
    return out


def stirling2_capped(m, cap):
    """Number of set partitions of m items into at most cap blocks (independent count: Stirling numbers, 2nd kind)."""
    S = [[0] * (m + 1) for _ in range(m + 1)]
    S[0][0] = 1
    for i in range(1, m + 1):
        for j in range(1, i + 1):
            S[i][j] = j * S[i - 1][j] + S[i - 1][j - 1]
    return sum(S[m][j] for j in range(0, min(cap, m) + 1))


def _frac(x):
    return Fraction(x) if isinstance(x, (str, int, Fraction)) else Fraction(float(x)).limit_denominator(1 << 40)


class Law:
    """atoms: list of dicts {count, f1, f2, p1, p2, y1, y2, s}; K = {1: K1, 2: K2}; caps = {1: m1, 2: m2}."""

    def __init__(self, atoms, K, caps, N=FIXTURE_N, name="law"):
        self.name, self.N, self.K, self.caps = name, int(N), dict(K), dict(caps)
        self.atoms = atoms
        cnt = [int(a["count"]) for a in atoms]
        if sum(cnt) != self.N or min(cnt) < 0:
            raise ValueError(f"{name}: atom counts must be nonnegative integers summing to N")
        self.c = np.asarray(cnt, dtype=np.int64)
        self.s = np.asarray([int(a["s"]) for a in atoms], np.int64)
        self.f = {i: np.asarray([int(a[f"f{i}"]) for a in atoms], np.int64) for i in (1, 2)}
        self.y = {i: np.asarray([int(a[f"y{i}"]) for a in atoms], np.int64) for i in (1, 2)}
        self.p = {i: np.asarray([[float(_frac(v)) for v in a[f"p{i}"]] for a in atoms], np.float64) for i in (1, 2)}
        self.d = {i: self.p[i].argmax(1) for i in (1, 2)}
        self.cells = {}
        for i in (1, 2):
            cc = {}
            for f_, d_ in zip(self.f[i].tolist(), self.d[i].tolist()):
                if cc.setdefault(f_, d_) != d_:
                    raise ValueError(f"{name}: fine cell {f_} of recipient {i} mixes teacher-predicted classes")
            self.cells[i] = cc

    def maps(self, i):
        """Every canonical same-class map of recipient i as a dict fine cell -> canonical token id."""
        import itertools
        by_c = {}
        for f_, c_ in sorted(self.cells[i].items()):
            by_c.setdefault(c_, []).append(f_)
        cls = sorted(by_c)
        per = [rgs_partitions(len(by_c[c_]), self.caps[i]) for c_ in cls]
        out = []
        for combo in itertools.product(*per):
            mp, t0 = {}, 0
            for c_, rgs in zip(cls, combo):
                for f_, b in zip(by_c[c_], rgs):
                    mp[f_] = t0 + b
                t0 += max(rgs) + 1
            out.append(mp)
        return out

    def tokens(self, i, mp):
        return np.asarray([mp[f_] for f_ in self.f[i].tolist()], np.int64)

    def stats(self, i, tok):
        G = int(tok.max() + 1)
        n = np.bincount(tok, weights=self.c, minlength=G)
        Y = np.zeros((G, self.K[i]))
        np.add.at(Y, (tok, self.y[i]), self.c.astype(np.float64))
        S = np.zeros((G, self.K[i]))
        np.add.at(S, tok, self.p[i] * self.c[:, None])
        cls = np.zeros(G, np.int64)
        cls[tok] = self.d[i]
        return n, Y, S, cls

    def mi(self, *toks):
        """Exact plug-in MI of S with the token tuple from the integer table of the law."""
        keys = {}
        tab = {}
        for a_, s_, cnt in zip(zip(*[t_.tolist() for t_ in toks]), self.s.tolist(), self.c.tolist()):
            j = keys.setdefault(a_, len(keys))
            tab[(s_, j)] = tab.get((s_, j), 0) + cnt
        T = np.zeros((2, len(keys)))
        for (s_, j), v in tab.items():
            T[s_, j] = v
        return mi_counts(T)

    def u_losses(self, i):
        """Continuous teacher U: exact expected clipped log loss, Brier and accuracy; fitting-majority constant."""
        w = self.c / self.N
        py = self.p[i][np.arange(len(self.c)), self.y[i]]
        ll = float(np.sum(w * -np.log(np.clip(py, LL_CLIP, 1.0))))
        Yo = np.eye(self.K[i])[self.y[i]]
        br = float(np.sum(w * np.sum((self.p[i] - Yo) ** 2, 1)))
        acc = float(np.sum(w * (self.d[i] == self.y[i])))
        prior = np.bincount(self.y[i], weights=self.c, minlength=self.K[i])
        const = int(np.argmax(prior))
        return {"L": ll, "B": br, "acc": acc, "const_acc": float(prior[const] / self.N), "const": const}

    def code_losses(self, i, tok, decoder, cache):
        """Exact expected log loss / Brier of a released code (decoder D0 = smoothed token mean teacher; D1 = own solve
        from the exact expected counts n_t, y_t, S_t at N = 4096)."""
        n, Y, S, cls = self.stats(i, tok)
        if decoder == "D0":
            Qt = smooth(S / n[:, None], cls)
        else:
            Qt = np.stack([cache.solve(n[t_], Y[t_], S[t_], int(cls[t_]))[1] for t_ in range(len(n))])
        L = float(np.sum(Y * -np.log(np.clip(Qt, LL_CLIP, 1.0))) / self.N)
        B = float(np.sum([Y[t_, y_] * float(np.sum((Qt[t_] - np.eye(self.K[i])[y_]) ** 2))
                          for t_ in range(len(n)) for y_ in range(self.K[i]) if Y[t_, y_] > 0]) / self.N)
        return L, B, Qt


def oracle_table(law: Law, decoders=("D0", "D1"), cache=None, max_pairs=FIXTURE_MAX_PAIRS):
    """Exhaustive table: every canonical map of each recipient (exact losses under D0 and D1, I_i, alphabet) and every
    mapping pair (exact I12, Phi). Refuses more than max_pairs pairs."""
    cache = cache or D1Cache()
    maps = {i: law.maps(i) for i in (1, 2)}
    npairs = len(maps[1]) * len(maps[2])
    if npairs > max_pairs:
        raise ValueError(f"{law.name}: {npairs} mapping pairs exceed the registered cap {max_pairs}")
    per = {}
    for i in (1, 2):
        rows = []
        for mp in maps[i]:
            tok = law.tokens(i, mp)
            r = {"map": mp, "tok": tok, "I": law.mi(tok), "alpha": int(tok.max() + 1)}
            for dcd in decoders:
                r[f"L_{dcd}"], r[f"B_{dcd}"], _ = law.code_losses(i, tok, dcd, cache)
            rows.append(r)
        per[i] = rows
    I12 = np.zeros((len(maps[1]), len(maps[2])))
    for a_, r1 in enumerate(per[1]):
        for b_, r2 in enumerate(per[2]):
            I12[a_, b_] = law.mi(r1["tok"], r2["tok"])
    return {"maps": maps, "per": per, "I12": I12, "pairs": npairs, "U": {i: law.u_losses(i) for i in (1, 2)},
            "acc": {i: float(np.sum(law.c * (law.d[i] == law.y[i])) / law.N) for i in (1, 2)}}


def oracle_references(tab, decoder="D1", budget=None):
    """Exhaustive references under the fitting budgets: the task-only optimum (min L_i + 0.5 B_i per recipient among
    feasible maps; first canonical map on ties) and the constrained privacy optimum (min Phi over feasible pairs with
    I_i <= I_i(task-only)); canonical first-index tie rule."""
    b = budget or FIT_BUDGET
    feas = {}
    for i in (1, 2):
        U_ = tab["U"][i]
        feas[i] = [budget_check(r[f"L_{decoder}"], r[f"B_{decoder}"], U_["L"], U_["B"], b)["ok"] for r in tab["per"][i]]
    task = {}
    for i in (1, 2):
        cands = [(r[f"L_{decoder}"] + 0.5 * r[f"B_{decoder}"], j) for j, r in enumerate(tab["per"][i]) if feas[i][j]]
        task[i] = min(cands)[1] if cands else None
    out = {"feasible_maps": {i: int(sum(feas[i])) for i in (1, 2)}, "task_only": task}
    if task[1] is None or task[2] is None:
        out["private"] = None
        return out
    cap = {i: tab["per"][i][task[i]]["I"] for i in (1, 2)}
    best = None
    for a_, r1 in enumerate(tab["per"][1]):
        if not feas[1][a_] or r1["I"] > cap[1]:
            continue
        for b_, r2 in enumerate(tab["per"][2]):
            if not feas[2][b_] or r2["I"] > cap[2]:
                continue
            ph = phi_of(r1["I"], r2["I"], tab["I12"][a_, b_])
            if best is None or ph < best[0]:
                best = (ph, a_, b_)
    out["caps"] = cap
    out["task_only_pair_I12"] = float(tab["I12"][task[1], task[2]])
    out["task_only_Phi"] = phi_of(cap[1], cap[2], tab["I12"][task[1], task[2]])
    out["private"] = None if best is None else {"Phi": best[0], "i1": best[1], "i2": best[2],
                                                 "I12": float(tab["I12"][best[1], best[2]])}
    return out


# ------------------------------------------------------------------------------------------------ lcr: own config ids
# The shared ID strings are a registered contract (TEAM_PLAN.md); the helpers below are own code.
LCR_PRIV = ("LOCAL", "SEQ-12", "SEQ-21", "JOINT")
LCR_K = ("LOCAL", "SEQ-12", "SEQ-21", "JOINT-SINGLE", "JOINT-PAIR")
L_CTASK = "U|C-TASK|i8o64|D1"
L_Q = "U|DIRECT-TASK|i8o64"


def L_d0(fam, lam=None):
    if fam == "CLASS":
        return "U|CLASS|i1o1"
    if fam in ("DIRECT-TASK", "FINE-TASK"):
        return f"U|{fam}|i8o64"
    return f"U|{fam}|i8o64|l{g(lam)}"


def L_d1(fam, lam=None):
    return L_d0(fam, lam) + "|D1"


def L_w(fam, lam):
    return f"U|W-{fam}|i8o64|l{g(lam)}|D1"


def L_k(arm):
    return f"U|K-{arm}|i8o64|D1"


def lcr_code_ids():
    d0 = [L_d0("DIRECT-TASK"), L_d0("FINE-TASK"), L_d0("CLASS")] + [L_d0(f, l_) for l_ in LAMS for f in LCR_PRIV]
    d1 = [L_d1("DIRECT-TASK"), L_d1("FINE-TASK")] + [L_d1(f, l_) for l_ in LAMS for f in LCR_PRIV]
    new = [L_CTASK] + [L_w(f, l_) for l_ in LAMS for f in LCR_PRIV] + [L_k(a) for a in LCR_K]
    return d0 + d1 + new


def lcr_scored_ids():
    return lcr_code_ids() + ["SRC|U", "SRC|RAW-J_b0.3", "REF|E", "REF|F", "REF|F0"]


def lcr_arm(cid):
    if cid.startswith("SRC|"):
        return "source"
    if cid.startswith("REF|"):
        return "reference"
    parts = cid.split("|")
    fam, d1 = parts[1], parts[-1] == "D1"
    if fam.startswith("W-"):
        return "weighted"
    if fam.startswith("K-"):
        return "constrained"
    if fam == "C-TASK":
        return "ctask"
    return "d1_fixed" if d1 else "d0"


def lcr_family(cid):
    if cid.startswith(("SRC|", "REF|")):
        return cid
    fam = cid.split("|")[1]
    return fam[2:] if fam.startswith(("W-", "K-")) else fam


CONSTRUCTION = {"d0": "existing", "d1_fixed": "calibrated", "weighted": "weighted", "constrained": "constrained",
                "ctask": "task-only", "source": "source", "reference": "reference"}


def lcr_role_lists(private_extra=()):
    """Candidate lists (prompt section 10, reconciled with SELECTION_RULES.json at a94e944): T* privacy-untrained
    (8; RAW-J excluded); P* the 77 privacy-trained code arms (RAW-J / FARE / LEACE are not P* candidates); C* the 72
    D0 / D1 fixed-map privacy maps and weighted controls (no constrained arm); N* the five constrained arms; C_pair*
    EVERY scored candidate except JOINT-PAIR (87: incl. the other constrained arms, every incumbent control, sources and
    references); J* JOINT-PAIR."""
    d0p = [L_d0(f, l_) for l_ in LAMS for f in LCR_PRIV]
    d1p = [L_d1(f, l_) for l_ in LAMS for f in LCR_PRIV]
    w = [L_w(f, l_) for l_ in LAMS for f in LCR_PRIV]
    k = [L_k(a) for a in LCR_K]
    T = [L_CTASK, L_d0("DIRECT-TASK"), L_d0("FINE-TASK"), L_d1("DIRECT-TASK"), L_d1("FINE-TASK"), "SRC|U",
         L_d0("CLASS"), "REF|F0"]
    C = d0p + d1p + w + list(private_extra)
    return {"T*": T, "P*": d0p + d1p + w + k + list(private_extra), "C*": C, "N*": k,
            "C_pair*": [x for x in lcr_scored_ids() if x != L_k("JOINT-PAIR")], "J*": [L_k("JOINT-PAIR")]}


FAMILY_RANK_LCR = {"LOCAL": 0, "SEQ-12": 1, "SEQ-21": 1, "JOINT": 2, "JOINT-SINGLE": 2, "JOINT-PAIR": 3}
CONSTRUCTION_RANK_LCR = {"d0": 0, "d1_fixed": 1, "weighted": 2, "constrained": 3}


def lcr_alias_record(rows, cid):
    """Exact release aliases (LABEL_TRUTH_TABLE winning_family_naming): configurations whose deployed release (tokens
    AND released probabilities; own fingerprint) equals cid's on all three seeds; the simplest family (LOCAL < SEQ-12 =
    SEQ-21 < JOINT(-SINGLE) < JOINT-PAIR, equal ranks joined with '=') and the simplest construction (existing <
    calibrated < weighted < constrained) over the full set; partial aliases disclosed."""
    if not cid or cid not in rows or not rows[cid].get("valid"):
        return None
    fp = {k: rows[cid]["seeds"][k].get("pair_fp") for k in SEEDS}
    if any(v is None for v in fp.values()):
        return {"available": False}
    full, partial = [cid], {}
    for c_, r in rows.items():
        if c_ == cid or not r.get("valid") or c_.startswith(("SRC|", "REF|")):
            continue
        same = [k for k in SEEDS if r["seeds"][k].get("pair_fp") == fp[k]]
        if len(same) == len(SEEDS):
            full.append(c_)
        elif same:
            partial[c_] = same
    fams = sorted({lcr_family(c_) for c_ in full}, key=lambda f: (FAMILY_RANK_LCR.get(f, 9), f))
    lo = FAMILY_RANK_LCR.get(fams[0], 9)
    cons = sorted({lcr_arm(c_) for c_ in full}, key=lambda a: CONSTRUCTION_RANK_LCR.get(a, 9))
    return {"available": True, "full": sorted(full), "partial": partial,
            "simplest_family": "=".join(f for f in fams if FAMILY_RANK_LCR.get(f, 9) == lo),
            "simplest_construction": CONSTRUCTION.get(cons[0]), "decided_by_config_id_tiebreak": len(full) > 1}


# ------------------------------------------------------------------------------------------------ lcr: own selection
def lcr_task_check(u, uU):
    """ORIGINAL inner eligibility for one task on one seed (no cbp headroom buffer): acc >= U - 0.01, LL <= U + 0.01,
    Brier <= U + 0.005, gain over the fitting-majority constant >= 0.8 x U's gain, gain >= 0.03 (margin form,
    inclusive)."""
    return task_check(u, uU, headroom={"ll_excess": GATE["ll_excess"], "brier_excess": GATE["brier_excess"]})


def lcr_config_row(cid, per_seed, seed_average=False):
    """Every-seed, every-task ordinary eligibility; seed_average=True is the deliberate defect. A constrained (K-) arm
    is eligible only if its fit is FEASIBLE (status FEASIBLE and deployed feasible) on EVERY seed (SELECTION_RULES
    constrained_fit_feasibility, review R-4): per_seed[k]["fit_feasible"]; other arms carry no fitting requirement."""
    r = config_row(cid, per_seed, headroom={"ll_excess": GATE["ll_excess"], "brier_excess": GATE["brier_excess"]},
                   seed_average=seed_average)
    r["family"], r["lam"], r["arm"] = lcr_family(cid), cid_lam(cid), lcr_arm(cid)
    r.pop("headroom_eligible", None)
    r.pop("shortfall_headroom", None)
    if lcr_arm(cid) == "constrained":
        r["fit_feasible"] = bool(r.get("valid")) and all(bool((per_seed.get(k) or {}).get("fit_feasible")) for k in SEEDS)
    else:
        r["fit_feasible"] = True
    r["ordinary_inner"] = r.get("ordinary_eligible", False)
    r["ordinary_eligible"] = bool(r["ordinary_inner"] and r["fit_feasible"])
    return r


def lcr_pick_role(rows, cands, nominee, guard_names=(), statuses=None, reverse=False):
    """One role (SELECTION_RULES.json roles / statuses / fallback_ordering / constrained_fit_feasibility; own code).
    reverse=True is the deliberate defect 'reversed best / worst'."""
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
        el = r["ordinary_eligible"]
        gok, gs = guard_check(r, guards)
        e = {"config": c_, "ordinary_eligible": el, "ordinary_inner": r["ordinary_inner"], "fit_feasible": r["fit_feasible"],
             "shortfall_ordinary": r["shortfall_ordinary"]}
        if missing_guard:
            e.update({"guard_ok": None, "shortfall_guard": None, "nominable": False,
                      "fallback_key": (not r["fit_feasible"], round(r["shortfall_ordinary"], ORDER_DECIMALS)) + okey(r)})
        else:
            e.update({"guard_ok": gok, "shortfall_guard": gs, "nominable": el and gok,
                      "fallback_key": (not r["fit_feasible"], round(r["shortfall_ordinary"], ORDER_DECIMALS),
                                       round(gs, ORDER_DECIMALS)) + okey(r)})
        ev.append(e)
    if missing_guard and any(e["ordinary_eligible"] for e in ev):
        return {"status": bad, "config": None, "reason": "MISSING_GUARD_COMPARATOR", "missing_guards": missing_guard,
                "blocked_eligible": [e["config"] for e in ev if e["ordinary_eligible"]], "evaluated": ev}
    nom = [e for e in ev if e["nominable"]]
    if nom:
        best = (max if reverse else min)(nom, key=lambda e: okey(rows[e["config"]]))
        return {"status": "NOMINEE", "config": best["config"], "evaluated": ev}
    if not any(e["fit_feasible"] for e in ev):
        why = "CONSTRAINED_FIT_INFEASIBLE"
    elif not any(e["ordinary_eligible"] for e in ev):
        why = "ORDINARY_UTILITY_FAILURE"
    else:
        why = "LOCAL_GUARD_FAILURE"
    fb = min(ev, key=lambda e: e["fallback_key"])
    out = {"status": none, "config": None, "descriptive_config": fb["config"], "reason": why, "evaluated": ev,
           "fallback_fit_feasible": fb["fit_feasible"],
           "fallback_shortfalls": {x: fb[x] for x in ("shortfall_ordinary", "shortfall_guard")}}
    if missing_guard:
        out.update({"missing_guards": missing_guard, "fallback_rank_status": "INVALID_MISSING_GUARD_COMPARATOR"})
    else:
        out["fallback_rank_status"] = "VALID"
    return out


def my_selection_lcr(rows, reverse=False, private_extra=(), guard_overrides=None):
    """Own implementation of the lcr roles (prompt section 10) on own inner rows."""
    L = lcr_role_lists(private_extra)
    G = {"T*": (), "C*": (), "P*": ("T*",), "N*": ("T*", "C*"), "C_pair*": (), "J*": ("T*", "C_pair*")}
    if guard_overrides:
        G.update(guard_overrides)
    L["C_pair*"] = [c_ for c_ in L["C_pair*"] if (rows.get(c_) or {}).get("fit_feasible") is not False]   # R-4
    st = {}
    for role in ("T*", "C*", "C_pair*", "P*", "N*", "J*"):
        st[role] = lcr_pick_role(rows, L[role], role in ("P*", "N*", "J*"), G[role], st, reverse=reverse)
    qr = rows.get(L_Q)
    if qr is None or not qr.get("valid"):
        st["Q"] = {"status": "INVALID_NOMINEE", "config": None, "reason": (qr or {}).get("invalid_reason") or
                   "FIT_OR_ADMISSION_FAILURE"}
    elif qr["ordinary_eligible"]:
        st["Q"] = {"status": "NOMINEE", "config": L_Q}
    else:
        st["Q"] = {"status": "NO_ELIGIBLE_NOMINEE", "config": None, "descriptive_config": L_Q,
                   "reason": "ORDINARY_UTILITY_FAILURE"}
    for x in ("P*", "N*", "J*", "T*", "C*", "C_pair*"):
        c_ = st[x].get("config") or st[x].get("descriptive_config")
        st[x]["aliases"] = lcr_alias_record(rows, c_)
        if st[x]["status"] == "NOMINEE":
            st[x]["family_of_config"] = lcr_family(c_)
            st[x]["construction"] = CONSTRUCTION[lcr_arm(c_)]
    resolved = {x: (v.get("config") or v.get("descriptive_config")) for x, v in st.items()}
    return {"statuses": st, "resolved": resolved,
            "aliases": sorted({(a_, b_) for a_ in resolved for b_ in resolved if a_ < b_ and resolved[a_] and
                               resolved[a_] == resolved[b_]})}


# ------------------------------------------------------------------------------------------------ lcr: own labels
LCR_CLAIMS = {"A": ("P*", "T*"), "B": ("N*", "C*"), "C": ("J*", "C_pair*")}
LCR_SLOTS = {"A": [f"P{i:02d}" for i in range(1, 12)], "B": [f"P{i:02d}" for i in range(12, 23)],
             "C": [f"P{i:02d}" for i in range(23, 34)], "Q": [f"P{i:02d}" for i in range(34, 38)]}
LCR_METHOD_LABEL = {"A": "PRIVACY_RELEASE_DEVELOPMENT_CRITERION_MET", "B": "CONSTRAINED_SEARCH_INCREMENT_ESTABLISHED",
                    "C": "PAIRED_JOINT_INCREMENT_ESTABLISHED"}


def lcr_clause_specs(claim):
    """The 11 clauses of a method claim (prompt section 12) as (kind, task, target, side) in slot order."""
    out = [("pair_auc_gap", None, MARGIN_PAIR, "lower>"), ("local_auc_excess", "income", MARGIN_LOCAL, "upper<"),
           ("local_auc_excess", "occupation", MARGIN_LOCAL, "upper<")]
    out += [("acc_vs_U", t_, -0.01, "lower>") for t_ in TASKS]
    out += [("ll_vs_U", t_, 0.01, "upper<") for t_ in TASKS]
    out += [("brier_vs_U", t_, 0.005, "upper<") for t_ in TASKS]
    out += [("retain_vs_U", t_, 0.0, "lower>") for t_ in TASKS]
    return out


def claim_status_lcr(nom_st, cmp_st, outcomes, claim, control_ok=True):
    """(status, root cause) of a claim in the registered precedence (cbp precedence carried to the lcr slots)."""
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
    if sorted(outcomes) != LCR_SLOTS[claim]:
        return "INCOMPLETE_OR_INVALID", "MISSING_SLOTS"
    if any(o == "INVALID" for o in outcomes.values()):
        return "INCOMPLETE_OR_INVALID", "NONFINITE_PRIMARY_QUANTITY"
    if all(o == "PASS" for o in outcomes.values()):
        return "PASS", "COMPLETE_PASSING_CONJUNCTION"
    return "NOT_ESTABLISHED", _fail_root(outcomes)


def labels_lcr(statuses, eps, technical_valid=True, control_ok=None, gate_passed=True):
    """Own lcr label truth table (provisional until reconciled with LABEL_TRUTH_TABLE.json)."""
    if not gate_passed:
        return {"label": "MECHANISM_GATE_NOT_MET", "claims": {}, "q": None}
    control_ok = control_ok or {}
    oc = {e["id"]: clause_outcome(e)[0] for e in eps}
    claims, roots = {}, {}
    for c_, (n_, m_) in LCR_CLAIMS.items():
        cl = {e["id"]: oc[e["id"]] for e in eps if e["claim"] == c_}
        claims[c_], roots[c_] = claim_status_lcr(statuses.get(n_), statuses.get(m_), cl, c_, control_ok.get(c_, True))
    qo = {e["id"]: oc[e["id"]] for e in eps if e["claim"] == "Q"}
    q_st = statuses.get("Q")
    if role_state(q_st) == "INVALID":
        q, q_root = "INCOMPLETE_OR_INVALID", "Q_NOT_RESOLVED"
    elif role_state(q_st) == "NO_ELIGIBLE":
        q, q_root = "NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE", "ORDINARY_UTILITY_FAILURE"
    elif sorted(qo) != LCR_SLOTS["Q"]:
        q, q_root = "INCOMPLETE_OR_INVALID", "MISSING_SLOTS"
    elif any(o == "INVALID" for o in qo.values()):
        q, q_root = "INCOMPLETE_OR_INVALID", "NONFINITE_PRIMARY_QUANTITY"
    elif all(o == "PASS" for o in qo.values()):
        q, q_root = "PASS", "COMPLETE_PASSING_CONJUNCTION"
    else:
        q, q_root = "NOT_ESTABLISHED", _fail_root(qo)
    if not control_ok.get("Q", True):
        q, q_root = "INCOMPLETE_OR_INVALID", "FAILED_REQUIRED_CONTROL"
    shown = {**claims, "Q": q}
    p_st = statuses.get("P*") or {}
    fam = None
    if p_st.get("config"):
        al = p_st.get("aliases") or {}
        fam = (f"{al['simplest_family']}; {al['simplest_construction']}" if al.get("available") else
               f"{lcr_family(p_st['config'])}; {CONSTRUCTION[lcr_arm(p_st['config'])]}")
    if not technical_valid:
        label = "INCOMPLETE_OR_INVALID"
    else:
        parts = [LCR_METHOD_LABEL[c_] + (f" ({fam})" if c_ == "A" and fam else "") for c_ in ("A", "B", "C")
                 if claims[c_] == "PASS"]
        if parts:
            label = " + ".join(parts)
        elif q == "PASS":                       # LABEL_TRUTH_TABLE precedence: before the INCOMPLETE rule
            label = "CONFIDENCE_FEASIBILITY_ESTABLISHED_NO_METHOD_CRITERION"
        elif any(v == "INCOMPLETE_OR_INVALID" for v in shown.values()):
            label = "INCOMPLETE_OR_INVALID"
        else:
            label = "EXPERIMENTAL_NO_ADVANTAGE"
    return {"label": label, "claims": claims, "root_causes": roots, "q": q, "q_root_cause": q_root,
            "incomplete_displayed": sorted(k_ for k_, v in shown.items() if v == "INCOMPLETE_OR_INVALID"),
            "clause_outcomes": oc}


def lcr_composition_closure_ok(ids, per_seed_bank=None):
    """Composition through the COMPLETE registered map / decoder bank: every code release of the bank must be present."""
    bank = set(per_seed_bank if per_seed_bank is not None else lcr_code_ids())
    return set(ids) == bank


RELEASE_KEYS_LCR = frozenset({"row_id", "tok1", "q1", "hard1", "alpha1", "tok2", "q2", "hard2", "alpha2"})


def release_keys_ok(keys):
    return frozenset(keys) == RELEASE_KEYS_LCR


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


# ================================================================================================ lcr self-tests
def _tok_rows(rng, n, K, d, conc=0.6, lab=None):
    P = rng.dirichlet(np.ones(K) * conc, n)
    P[:, d] += P.max(1) + 1e-3
    P /= P.sum(1, keepdims=True)
    lab = rng.dirichlet(np.ones(K) * 0.7) if lab is None else np.asarray(lab, float)
    return P, rng.choice(K, n, p=lab)


def check_decoder_entry(entry, y_rows, P_rows, d, tol=None):
    """The P2 check of ONE stored decoder entry against the fitting rows of its token: exact recount of n_t / y_t,
    teacher sums within tolerance, own sufficient-statistic key, own re-solve, certificate of the STORED released q."""
    tol = tol or D1_TOL
    K = P_rows.shape[1]
    n = float(len(y_rows))
    Y = np.bincount(np.asarray(y_rows, np.int64), minlength=K).astype(np.float64)
    S = np.zeros(K)
    for row in np.asarray(P_rows, np.float64):
        S += row
    counts_ok = float(entry["n"]) == n and np.array_equal(np.asarray(entry["y"], np.float64), Y)
    s_ok = float(np.max(np.abs(np.asarray(entry["S"], np.float64) - S))) <= tol["teacher_sum_rel"] * max(1.0, n)
    key_ok = entry.get("hash") is None or entry["hash"] == suff_hash(K, d, n, Y, S)
    _, q_own, _ = own_d1_solve(n, Y, S, d)
    dq = float(np.max(np.abs(np.asarray(entry["q"], np.float64) - q_own)))
    cert = d1_certificate(entry["u"], entry["q"], n, Y, S, d)
    ok = counts_ok and s_ok and key_ok and dq <= tol["q"] and cert["ok"]
    return {"ok": bool(ok), "counts_exact": bool(counts_ok), "teacher_sums_ok": bool(s_ok), "key_ok": bool(key_ok),
            "max_dq_vs_own": dq, "certificate_ok": cert["ok"], "strict_argmax": cert["strict_argmax"],
            "fw_gap_rel": cert.get("fw_gap_rel"), "dominance_exact": cert.get("dominance_exact")}


def selftest_decoder(rng):
    out = {}
    worst = {"fw_gap_rel": 0.0, "stationarity_rel": 0.0, "identity_max_abs": 0.0, "sum_residual": 0.0}
    nfail, ntied, nzero, t0 = 0, 0, 0, time.time()
    for trial in range(400):
        K = (2, 6)[trial % 2]
        n = int(rng.integers(1, 4000))
        d = int(rng.integers(0, K))
        P, y = _tok_rows(rng, n, K, d)
        Y = np.bincount(y, minlength=K).astype(float)
        u, q, info = own_d1_solve(n, Y, P.sum(0), d)
        c = d1_certificate(u, q, n, Y, P.sum(0), d)
        nfail += not c["ok"]
        ntied += bool(info["tied"])
        nzero += bool(info["zero"])
        for k_ in worst:
            worst[k_] = max(worst[k_], float(c[k_]))
    out["random_tokens"] = {"tokens": 400, "certificate_failures": nfail, "with_tied_classes": ntied,
                            "with_zero_bounds": nzero, "worst": worst, "ms_per_solve": 1000 * (time.time() - t0) / 400}
    # edge cases: n = 1; every label on another class (forces a tie); zero teacher mass and no labels (zero bound);
    # very large n; labels only on d; exact teacher tie between d and another class
    edge = []
    cases = [(1, [0, 1], [0.6, 0.4], 0), (50, [0, 50, 0], [0.5, 0.3, 0.2], 0), (40, [40, 0, 0, 0], [30, 10, 0, 0], 0),
             (100000, [20000, 50000, 30000], [40000, 35000, 25000], 0), (30, [30, 0], [20, 10], 0),
             (10, [3, 7], [5, 5], 0), (7, [0, 0, 0, 0, 0, 7], [2, 1, 1, 1, 1, 1], 0)]
    for n, y, S, d in cases:
        y, S = np.asarray(y, float), np.asarray(S, float)
        S = S if S.sum() == n else S / S.sum() * n
        u, q, info = own_d1_solve(n, y, S, d)
        c = d1_certificate(u, q, n, y, S, d)
        pts = kkt_enumerate_d1(n, y, S, d)
        edge.append({"n": n, "K": len(y), "ok": c["ok"], "tied": info["tied"], "zero": info["zero"],
                     "kkt_points": len(pts), "kkt_dq": max([float(np.max(np.abs(p_["q"] - q))) for p_ in pts] or [1.0])})
    out["edge_cases"] = edge
    edge_ok = all(e["ok"] and e["kkt_points"] >= 1 and e["kkt_dq"] <= 1e-9 for e in edge)
    # independent trusted solves: SLSQP (feasible projection) and exhaustive KKT active-pattern enumeration
    sl, kk = [], []
    for trial in range(24):
        K = (2, 3, 4, 6)[trial % 4]
        n = int(rng.integers(3, 800))
        d = int(rng.integers(0, K))
        P, y = _tok_rows(rng, n, K, d, lab=rng.dirichlet(np.ones(K) * 0.4))
        Y, S = np.bincount(y, minlength=K).astype(float), P.sum(0)
        u, q, info = own_d1_solve(n, Y, S, d)
        us, qs, infeas = slsqp_d1(n, Y, S, d)
        sl.append({"K": K, "dq": float(np.max(np.abs(qs - q))), "f_slsqp_minus_own": d1_f(qs, n, Y, S) - d1_f(q, n, Y, S),
                   "slsqp_raw_infeasibility": infeas, "scale": n + KAPPA})
        if trial < 12:
            pts = kkt_enumerate_d1(n, Y, S, d)
            kk.append({"K": K, "points": len(pts), "patterns": sorted({p_["pattern"] for p_ in pts}),
                       "dq": max([float(np.max(np.abs(p_["q"] - q))) for p_ in pts] or [1.0]),
                       "spread": max([float(np.max(np.abs(p_["q"] - pts[0]["q"]))) for p_ in pts] or [0.0])})
    sl_ok = all(r["f_slsqp_minus_own"] >= -1e-10 * r["scale"] and r["dq"] <= 1e-4 for r in sl)
    kk_ok = all(r["points"] >= 1 and r["dq"] <= 1e-9 and r["spread"] <= 1e-9 for r in kk)
    out["slsqp_cross_check"] = {"tokens": len(sl), "own_never_worse": sl_ok,
                                "max_dq": max(r["dq"] for r in sl),
                                "min_f_slsqp_minus_own_rel": min(r["f_slsqp_minus_own"] / r["scale"] for r in sl)}
    out["kkt_enumeration"] = {"tokens": len(kk), "unique_kkt_point_equals_own": kk_ok, "max_dq": max(r["dq"] for r in kk),
                              "points_per_token": [r["points"] for r in kk]}
    # convexity (midpoint inequality on random feasible pairs) and the analytic gradient (central differences)
    cv_ok, gd = True, 0.0
    for trial in range(60):
        K = 6
        n, d = int(rng.integers(5, 300)), int(rng.integers(0, K))
        P, y = _tok_rows(rng, n, K, d)
        Y, S = np.bincount(y, minlength=K).astype(float), P.sum(0)
        u1, u2 = proj_class_simplex(rng.dirichlet(np.ones(K)), d), proj_class_simplex(rng.dirichlet(np.ones(K)), d)
        f1, f2 = d1_objective(d1_q_of_u(u1, d), n, Y, S), d1_objective(d1_q_of_u(u2, d), n, Y, S)
        fm = d1_objective(d1_q_of_u(0.5 * (u1 + u2), d), n, Y, S)
        cv_ok &= fm <= 0.5 * (f1 + f2) + 1e-9 * (n + KAPPA)
        u1 = 0.9 * u1 + 0.1 / K                    # interior point (finite differences stay inside q > 0)
        q1 = d1_q_of_u(u1, d)
        g_ = d1_grad_u(q1, n, Y, S)

        def cd(k, h):
            e = np.zeros(K)
            e[k] = h
            return (d1_objective(d1_q_of_u(u1 + e, d), n, Y, S) - d1_objective(d1_q_of_u(u1 - e, d), n, Y, S)) / (2 * h)
        for k in range(K):
            num = (4.0 * cd(k, 5e-6) - cd(k, 1e-5)) / 3.0      # Richardson extrapolation (O(h^4) truncation)
            gd = max(gd, abs(num - g_[k]) / (abs(g_[k]) + n + KAPPA))
    out["convexity_and_gradient"] = {"midpoint_convex": bool(cv_ok), "max_rel_gradient_error": gd}
    # projection: exact projection equals a brute-force minimiser; tiny dominance residual repaired, released q strict
    pr_ok, mag = True, 0.0
    from scipy.optimize import minimize
    for trial in range(40):
        K = (2, 6)[trial % 2]
        d = int(rng.integers(0, K))
        w = rng.normal(size=K) * 0.5 + 1.0 / K
        u = proj_class_simplex(w, d)
        cons = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}] + [
            {"type": "ineq", "fun": (lambda x, k=k: x[d] - x[k])} for k in range(K) if k != d]
        r = minimize(lambda x: 0.5 * float((x - w) @ (x - w)), np.full(K, 1.0 / K), jac=lambda x: x - w,
                     bounds=[(0, 1)] * K, constraints=cons, method="SLSQP", options={"ftol": 1e-16, "maxiter": 500})
        pr_ok &= bool((u >= 0).all() and abs(u.sum() - 1) <= 1e-12 and all(u[d] >= u[k] for k in range(K)) and
                      0.5 * float((u - w) @ (u - w)) <= 0.5 * float((r.x - w) @ (r.x - w)) + 1e-10)
    rep_ok, mag_reg = True, 0.0
    for trial in range(200):
        K, d = 6, int(rng.integers(0, 6))
        u = rng.dirichlet(np.ones(K))
        u[d] = u.max()
        u /= u.sum()
        k, j = (d + 1) % K, (d + 2) % K
        u[k] = u[d]
        u[j] = 1.0 - (u.sum() - u[j])             # keep the simplex sum
        if u[j] < 2e-10:
            continue
        u[k] += 1e-10                             # a solver residual that would beat the 1e-12 smoothing advantage
        u[j] -= 1e-10
        qbad = d1_q_of_u(u, d)
        up = proj_class_simplex(u, d)
        qp = d1_q_of_u(up, d)
        mag = max(mag, float(np.max(np.abs(up - u))))
        rep_ok &= (int(np.argmax(qbad)) != d) and strict_argmax_ok(qp, d) and all(up[d] >= up[j] for j in range(K))
        ur, mr = proj_registered(u, d)
        mag_reg = max(mag_reg, mr)
        rep_ok &= strict_argmax_ok(d1_q_of_u(ur, d), d) and all(ur[d] >= ur[j] for j in range(K)) and \
            bool((ur >= 0).all()) and abs(ur.sum() - 1.0) <= 1e-12
    out["projection"] = {"exact_vs_bruteforce": bool(pr_ok), "tiny_residual_repaired_and_strict": bool(rep_ok),
                         "unrepaired_residual_breaks_argmax": True, "max_projection_magnitude_euclidean": mag,
                         "max_projection_magnitude_registered_rule": mag_reg}
    # calibrated-teacher null at a fixed token: labels equal the teacher mean exactly -> D1 = pbar (= D0 up to eps)
    cal = 0.0
    for trial in range(50):
        K, n = 6, float(rng.integers(10, 2000))
        d = int(rng.integers(0, K))
        pb = rng.dirichlet(np.ones(K))
        pb[d] = pb.max() + 0.05
        pb /= pb.sum()
        _, q, _ = own_d1_solve(n, n * pb, n * pb, d)
        cal = max(cal, float(np.max(np.abs(q - smooth(pb, d)[0]))))
    out["calibrated_null_token"] = {"max_abs_D1_minus_D0": cal, "ok": cal <= 1e-10}
    ok = (nfail == 0 and edge_ok and sl_ok and kk_ok and cv_ok and gd <= 1e-6 and pr_ok and rep_ok and cal <= 1e-10)
    return res("PASS" if ok else "FAIL", **out)


# ---- fixture-oracle self-tests on own synthetic laws (NOT the registered fixtures)
def _law_from(rng, K, cells, teach, label_law, s_bias, name, xor=False):
    """cells[i]: number of fine cells of recipient i per class; teach[i][f]: teacher vector (eighths) of fine cell f;
    label_law[i][f]: true label law (eighths) of fine cell f; atoms (f1, f2, s, y1, y2) with integer counts summing to
    4096: base masses are multiples of 64 so the label splits in eighths are exact."""
    F = {i: sum(cells[i]) for i in (1, 2)}
    combos = [(f1, f2, s_) for f1 in range(F[1]) for f2 in range(F[2]) for s_ in (0, 1)]
    if xor:
        combos = [c_ for c_ in combos if (c_[0] % 2) ^ (c_[1] % 2) == c_[2]]
    units = FIXTURE_N // 64
    w = np.array([1.0 + (s_bias(c_) if not xor else 0.0) for c_ in combos])
    base = np.ones(len(combos), dtype=np.int64)
    rest = units - len(combos)
    if rest < 0:
        raise ValueError("too many combos")
    add = np.floor(rest * w / w.sum()).astype(np.int64)
    base += add
    for j in np.argsort(-w, kind="stable")[: units - int(base.sum())]:
        base[j] += 1
    atoms = []
    for (f1, f2, s_), b in zip(combos, base.tolist()):
        for y1 in range(K[1]):
            for y2 in range(K[2]):
                c = 64 * b * label_law[1][f1][y1] * label_law[2][f2][y2] // 64
                if c:
                    atoms.append({"count": int(c), "f1": f1, "f2": f2, "s": s_, "y1": y1, "y2": y2,
                                  "p1": [f"{v}/8" for v in teach[1][f1]], "p2": [f"{v}/8" for v in teach[2][f2]]})
    tot = sum(a["count"] for a in atoms)
    if tot != FIXTURE_N:
        raise ValueError(f"{name}: counts sum to {tot}")
    return Law(atoms, K, {1: 2, 2: 2}, name=name)


def own_laws(rng):
    K = {1: 2, 2: 3}
    cells = {1: [2, 2], 2: [2, 2, 2]}
    t1 = [[5, 3], [7, 1], [3, 5], [2, 6]]
    t2 = [[4, 3, 1], [6, 1, 1], [2, 5, 1], [1, 4, 3], [1, 2, 5], [2, 1, 5]]
    mis1 = [[8, 0], [6, 2], [1, 7], [4, 4]]
    mis2 = [[7, 1, 0], [3, 3, 2], [0, 8, 0], [2, 3, 3], [0, 1, 7], [3, 1, 4]]
    bias = (lambda c_: 1.5 * ((c_[0] + c_[1] + c_[2]) % 2))
    return {"own_calibrated": _law_from(rng, K, cells, {1: t1, 2: t2}, {1: t1, 2: t2}, bias, "own_calibrated"),
            "own_miscalibrated": _law_from(rng, K, cells, {1: t1, 2: t2}, {1: mis1, 2: mis2}, bias, "own_miscalibrated"),
            "own_xor": _law_from(rng, {1: 2, 2: 2}, {1: [2, 2], 2: [2, 2]}, {1: t1, 2: t1}, {1: mis1, 2: mis1}, bias,
                                 "own_xor", xor=True)}


def expand_law(law: Law, i):
    rep = np.repeat(np.arange(len(law.c)), law.c)
    return rep, law.f[i][rep], law.y[i][rep], law.p[i][rep], law.s[rep]


def selftest_oracle(rng):
    out = {}
    cnt_ok = all(len(rgs_partitions(m, c)) == stirling2_capped(m, c) for m in range(0, 8) for c in range(1, 6))
    out["canonical_partition_counts_equal_stirling"] = bool(cnt_ok)
    laws = own_laws(rng)
    cache = D1Cache()
    tabs = {k: oracle_table(L_, cache=cache) for k, L_ in laws.items()}
    # exact-law quantities equal row-level recomputation on the expanded law (N = 4096 rows)
    par = 0.0
    for nm, L_ in laws.items():
        for i in (1, 2):
            rep, f_, y_, P_, s_ = expand_law(L_, i)
            for r in tabs[nm]["per"][i][:6]:
                tok = r["tok"][rep]
                par = max(par, abs(mi_of(s_, tok) - r["I"]))
                rel = own_d1_release(tok, y_, P_, P_.argmax(1), L_.K[i])
                Lr, Br = fit_losses(rel["q"][tok], y_)
                par = max(par, abs(Lr - r["L_D1"]), abs(Br - r["B_D1"]))
                L0, B0 = fit_losses(smooth(rel["S"] / rel["n"][:, None], rel["cls"])[tok], y_)
                par = max(par, abs(L0 - r["L_D0"]), abs(B0 - r["B_D0"]))
    out["exact_law_vs_expanded_rows_max_abs"] = par
    # calibrated null: no population decoder improvement on any fixed map
    cal = tabs["own_calibrated"]
    gain = max(r["L_D0"] - r["L_D1"] for i in (1, 2) for r in cal["per"][i])
    bgain = max(r["B_D0"] - r["B_D1"] for i in (1, 2) for r in cal["per"][i])
    out["calibrated_null_max_population_gain"] = {"logloss": gain, "brier": bgain}
    # miscalibrated: D1 improves confidence for the SAME map, full-token MI identical (tokens unchanged)
    mis = laws["own_miscalibrated"]
    imp = max(r["L_D0"] - r["L_D1"] for r in tabs["own_miscalibrated"]["per"][2])
    same = True
    for i in (1, 2):
        for r in tabs["own_miscalibrated"]["per"][i]:
            n_, Y_, S_, cls_ = mis.stats(i, r["tok"])
            q0 = smooth(S_ / n_[:, None], cls_)
            q1 = np.stack([cache.solve(n_[t_], Y_[t_], S_[t_], int(cls_[t_]))[1] for t_ in range(len(n_))])
            same &= mis.mi(np.unique(q0[r["tok"]], axis=0, return_inverse=True)[1]) <= r["I"] + 1e-15
            same &= mis.mi(r["tok"]) == r["I"]
            same &= bool(all(strict_argmax_ok(q1[t_], cls_[t_]) for t_ in range(len(n_))))
    out["miscalibrated"] = {"max_logloss_gain_same_map": imp, "full_token_mi_unchanged_and_decisions_kept": bool(same)}
    # complementary clues: individual MI 0, pair MI log 2 (finest maps)
    xo = tabs["own_xor"]
    fi = {i: max(range(len(xo["per"][i])), key=lambda j: xo["per"][i][j]["alpha"]) for i in (1, 2)}
    out["xor"] = {"I1": xo["per"][1][fi[1]]["I"], "I2": xo["per"][2][fi[2]]["I"], "I12": float(xo["I12"][fi[1], fi[2]])}
    xor_ok = out["xor"]["I1"] < 1e-12 and out["xor"]["I2"] < 1e-12 and abs(out["xor"]["I12"] - math.log(2)) < 1e-12
    # references respect the budgets and caps
    ref_ok = True
    for nm, tb in tabs.items():
        rf = oracle_references(tb)
        if rf.get("private"):
            for i, j in ((1, rf["private"]["i1"]), (2, rf["private"]["i2"])):
                r = tb["per"][i][j]
                ref_ok &= budget_check(r["L_D1"], r["B_D1"], tb["U"][i]["L"], tb["U"][i]["B"])["ok"]
                ref_ok &= r["I"] <= rf["caps"][i]
            ref_ok &= rf["private"]["Phi"] <= rf["task_only_Phi"] + 1e-15
        out[f"refs_{nm}"] = {"pairs": tb["pairs"], "feasible_maps": rf["feasible_maps"],
                             "task_only_I12": rf.get("task_only_pair_I12"),
                             "private_I12": (rf.get("private") or {}).get("I12")}
    ok = (cnt_ok and par <= 1e-12 and gain <= 1e-9 and bgain <= 1e-9 and imp > 1e-3 and same and xor_ok and ref_ok)
    out["solves_cached"] = {"hits": cache.hits, "misses": cache.miss}
    return res("PASS" if ok else "FAIL", **out)


# ---- own engine + incremental parity
def synth_fitting(rng, n=3000, K=(2, 6), cells=(6, 5), noise=(0.05, 1.0)):
    """Fitting rows with fine cells inside teacher-predicted classes (teacher = cell prototype + row noise), a
    miscalibrated (sharper) label law and a SEX signal."""
    K1, K2 = K
    out = {"K": {1: K1, 2: K2}}
    s = rng.integers(0, 2, n)
    for i, Ki, ci, w in ((1, K1, cells[0], noise[0]), (2, K2, cells[1], noise[1])):
        cls = rng.integers(0, Ki, n)
        f = cls * ci + np.minimum(ci - 1, (rng.integers(0, ci, n) + s * rng.integers(0, 2, n)) % ci)
        proto = rng.dirichlet(np.ones(Ki) * 0.8, Ki * ci)
        P = proto[f] + w * rng.dirichlet(np.ones(Ki) * 0.8, n)
        P[np.arange(n), cls] += P.max(1) + 0.02 * (f % ci)
        P /= P.sum(1, keepdims=True)
        Ptrue = P ** 2.2
        Ptrue /= Ptrue.sum(1, keepdims=True)
        y = np.array([rng.choice(Ki, p=p_) for p_ in Ptrue])
        out[i] = {"f": f, "P": P, "y": y, "dec": cls, "ncells": Ki * ci, "cell_class": np.arange(Ki * ci) // ci}
    out["s"] = s
    return out


def engine_of(sf, cache=None):
    return LcrEngine({i: sf[i]["f"] for i in (1, 2)}, {i: sf[i]["y"] for i in (1, 2)}, {i: sf[i]["P"] for i in (1, 2)},
                     {i: sf[i]["dec"] for i in (1, 2)}, sf["s"], sf["K"], cache)


def incremental_move(eng, i, cmap, cell, target, base=None):
    """Own incremental update after moving fine cell `cell` to token `target` (same class): subtract / add the cell's
    sufficient statistics and SEX table, re-solve only the two affected tokens, recompute the losses and MI from the
    token-level tables (a different summation order from the from-scratch row-level path)."""
    tok = np.asarray(cmap, np.int64)[eng.f[i]]
    G = int(max(tok.max(), target) + 1)
    K = eng.K[i]
    n, Y, S = token_stats(tok, eng.y[i], eng.P[i], K, G)
    st = np.zeros((2, G))
    np.add.at(st, (eng.s, tok), 1.0)
    rows = eng.f[i] == cell
    src = int(cmap[cell])
    cn, cY, cS = token_stats(np.zeros(int(rows.sum()), np.int64), eng.y[i][rows], eng.P[i][rows], K, 1)
    cst = np.bincount(eng.s[rows], minlength=2).astype(np.float64)
    n[src] -= cn[0]
    Y[src] -= cY[0]
    S[src] -= cS[0]
    st[:, src] -= cst
    n[target] += cn[0]
    Y[target] += cY[0]
    S[target] += cS[0]
    st[:, target] += cst
    cls = token_class(tok, eng.dec[i], G)
    Q = np.zeros((G, K))
    for t_ in range(G):
        if n[t_] > 0:
            Q[t_] = eng.cache.solve(n[t_], Y[t_], S[t_], int(cls[t_]))[1]
    N = float(len(tok))
    occ = n > 0
    L = float(np.sum(Y[occ] * -np.log(np.clip(Q[occ], LL_CLIP, 1.0)))) / N
    B = float(np.sum(n[occ] * np.sum(Q[occ] ** 2, 1) - 2 * np.sum(Y[occ] * Q[occ], 1) + n[occ])) / N
    return {"L": L, "B": B, "I": mi_counts(st[:, occ]), "src_empty": bool(n[src] == 0)}


def selftest_engine(rng):
    sf = synth_fitting(rng)
    eng = engine_of(sf)
    ident = {i: np.arange(sf[i]["ncells"]) for i in (1, 2)}
    base = eng.pair(ident[1], ident[2])
    md = 0.0
    nmv = 0
    for i in (1, 2):
        cc = sf[i]["cell_class"]
        for cell in range(0, sf[i]["ncells"], 3):
            same = [c_ for c_ in range(sf[i]["ncells"]) if cc[c_] == cc[cell] and c_ != cell]
            tgt = same[int(rng.integers(0, len(same)))]
            inc = incremental_move(eng, i, ident[i], cell, int(ident[i][tgt]))
            m2 = ident[i].copy()
            m2[cell] = ident[i][tgt]
            r = eng.recipient(i, m2)
            md = max(md, abs(inc["L"] - r["L"]), abs(inc["B"] - r["B"]), abs(inc["I"] - r["I"]))
            nmv += 1
    # decoder-only change: identical tokens -> identical MI (exact) and identical decisions
    d0 = eng.pair(ident[1], ident[2], decoder="D0")
    same_mi = d0["I1"] == base["I1"] and d0["I2"] == base["I2"] and d0["I12"] == base["I12"]
    dec_ok = all(strict_argmax_ok(base[f"r{i}"]["dec"]["q"][t_], base[f"r{i}"]["dec"]["cls"][t_])
                 for i in (1, 2) for t_ in range(base[f"r{i}"]["alpha"]))
    tf_ok = release_is_token_function(base["r2"]["tok"], base["r2"]["dec"]["q"][base["r2"]["tok"]])
    ok = md <= D1_TOL["loss_mean"] and same_mi and dec_ok and tf_ok
    return res("PASS" if ok else "FAIL", moves=nmv, max_abs_incremental_vs_scratch=md,
               decoder_change_keeps_full_token_mi=bool(same_mi), d1_decisions_preserved=bool(dec_ok),
               release_is_token_function=bool(tf_ok),
               d0_vs_d1_fit_logloss={"D0": [d0["L1"], d0["L2"]], "D1": [base["L1"], base["L2"]]},
               solves={"hits": eng.cache.hits, "misses": eng.cache.miss})


# ---- lcr selection self-tests
def synthetic_bank_lcr(overrides=None):
    """Every lcr scored candidate on 3 seeds with known answers (see selftest_selection_lcr)."""
    rows_in = {}
    lam_pair = {0.01: 0.840, 0.025: 0.835, 0.04: 0.830, 0.06: 0.824, 0.08: 0.820, 0.1: 0.815}
    shift = {"LOCAL": 0.010, "SEQ-12": 0.002, "SEQ-21": 0.003, "JOINT": 0.0}
    ll0 = {0.01: 0.0035, 0.025: 0.0055, 0.04: 0.0075, 0.06: 0.0102, 0.08: 0.0115, 0.1: 0.013}
    ll1 = {0.01: 0.0010, 0.025: 0.0025, 0.04: 0.0045, 0.06: 0.0072, 0.08: 0.0088, 0.1: 0.0105}
    for f_ in LCR_PRIV:
        for lam in LAMS:
            p_ = lam_pair[lam] + shift[f_]
            rows_in[L_d0(f_, lam)] = {k: _seed((0.0005, ll0[lam]), (0.0003, 0.002), (0.760, 0.770, p_), 300.0) for k in SEEDS}
            rows_in[L_d1(f_, lam)] = {k: _seed((0.0005, ll1[lam]), (0.0003, 0.002), (0.760, 0.770, p_), 300.0) for k in SEEDS}
            rows_in[L_w(f_, lam)] = {k: _seed((0.0005, ll1[lam] + 0.0005), (0.0003, 0.002), (0.760, 0.770, p_ + 0.001),
                                              290.0) for k in SEEDS}
    for a_, p_ in zip(LCR_K, (0.834, 0.826, 0.827, 0.823, 0.817)):
        rows_in[L_k(a_)] = {k: {**_seed((0.0005, 0.006), (0.0003, 0.002), (0.760, 0.770, p_), 280.0), "fit_feasible": True}
                            for k in SEEDS}
    rows_in[L_CTASK] = {k: _seed((0.0004, 0.0020), (0.0003, 0.0012), (0.762, 0.790, 0.845), 320.0) for k in SEEDS}
    rows_in[L_d0("DIRECT-TASK")] = {k: _seed((0.0008, 0.0030), (0.0005, 0.0017), (0.762, 0.790, 0.849), 328.0) for k in SEEDS}
    rows_in[L_d0("FINE-TASK")] = {k: _seed((0.0008, 0.0031), (0.0005, 0.0017), (0.761, 0.788, 0.846), 329.0) for k in SEEDS}
    rows_in[L_d1("DIRECT-TASK")] = {k: _seed((0.0003, 0.0010), (0.0002, 0.0010), (0.762, 0.790, 0.848), 328.0) for k in SEEDS}
    rows_in[L_d1("FINE-TASK")] = {k: _seed((0.0003, 0.0011), (0.0002, 0.0010), (0.761, 0.788, 0.8465), 329.0) for k in SEEDS}
    rows_in[L_d0("CLASS")] = {k: _seed((0.03, 0.20), (0.02, 0.10), (0.70, 0.70, 0.739), 8.0) for k in SEEDS}
    rows_in["SRC|U"] = {k: _seed(auc=(0.77, 0.80, 0.858), states=math.inf) for k in SEEDS}
    rows_in["SRC|RAW-J_b0.3"] = {k: _seed((0.001, 0.02), (0.001, 0.01), (0.70, 0.71, 0.800), math.inf) for k in SEEDS}
    rows_in["REF|E"] = {k: _seed((0.08, 0.05), (0.03, 0.02), (0.60, 0.70, 0.75), math.inf, dacc=(-0.05, 0)) for k in SEEDS}
    rows_in["REF|F"] = {k: _seed((0.03, 0.046), (0.01, 0.02), (0.65, 0.66, 0.704), math.inf, dacc=(0, -0.024)) for k in SEEDS}
    rows_in["REF|F0"] = {k: _seed((0.02, 0.03), (0.01, 0.015), (0.68, 0.70, 0.76), math.inf, dacc=(0, -0.015)) for k in SEEDS}
    for c_, f_ in (overrides or {}).items():
        rows_in[c_] = f_(rows_in[c_])
    return {c_: lcr_config_row(c_, v) for c_, v in rows_in.items()}, rows_in


LCR_EXPECTED = {"T*": L_CTASK, "C*": L_d1("JOINT", 0.08), "C_pair*": L_d1("JOINT", 0.08), "P*": L_k("JOINT-PAIR"),
                "N*": L_k("JOINT-PAIR"), "J*": L_k("JOINT-PAIR"), "Q": L_Q}


def selftest_selection_lcr():
    out = {}
    rows, rin = synthetic_bank_lcr()
    S = my_selection_lcr(rows)
    out["resolved"] = S["resolved"]
    out["expected_roles"] = S["resolved"] == LCR_EXPECTED
    ok_rj = (lambda v: {k: _seed((0.001, 0.005), (0.001, 0.003), (0.770, 0.800, 0.800), math.inf) for k in SEEDS})
    rows_rj, _ = synthetic_bank_lcr({"SRC|RAW-J_b0.3": ok_rj})
    S_rj = my_selection_lcr(rows_rj)
    out["raw_j_excluded_from_T"] = "SRC|RAW-J_b0.3" not in lcr_role_lists()["T*"] and S_rj["resolved"]["T*"] == \
        L_CTASK and rows_rj["SRC|RAW-J_b0.3"]["ordinary_eligible"] and S_rj["resolved"]["C_pair*"] == "SRC|RAW-J_b0.3"
    out["c_pair_includes_sources_and_references"] = S_rj["resolved"]["C_pair*"] == "SRC|RAW-J_b0.3" and \
        len(lcr_role_lists()["C_pair*"]) == 87 and len(lcr_role_lists()["P*"]) == 77 and len(lcr_role_lists()["C*"]) == 72
    out["d1_privacy_not_task_only"] = all(c_ not in lcr_role_lists()["T*"] for c_ in [L_d1(f, l_) for f in LCR_PRIV
                                                                                     for l_ in LAMS])
    out["d0_large_lambda_ineligible_d1_eligible"] = (not rows[L_d0("JOINT", 0.08)]["ordinary_eligible"]) and \
        rows[L_d1("JOINT", 0.08)]["ordinary_eligible"]
    out["no_headroom_buffer"] = rows[L_d1("JOINT", 0.08)]["ordinary_eligible"]       # LL excess 0.0088 > cbp 0.006
    # local guard failure of J* (descriptive fallback = JOINT-PAIR, reason LOCAL_GUARD_FAILURE)

    def guard_bad(v):
        return {k: {**x, "auc": {**x["auc"], "v1": 0.80}} for k, x in v.items()}
    rows_g, _ = synthetic_bank_lcr({L_k("JOINT-PAIR"): guard_bad})
    Sg = my_selection_lcr(rows_g)
    out["guard_failure"] = {"J*": Sg["statuses"]["J*"]["status"], "reason": Sg["statuses"]["J*"].get("reason"),
                            "descriptive": Sg["statuses"]["J*"].get("descriptive_config")}
    g_ok = Sg["statuses"]["J*"]["status"] == "NO_ELIGIBLE_NOMINEE" and Sg["statuses"]["J*"]["reason"] == \
        "LOCAL_GUARD_FAILURE" and Sg["statuses"]["J*"]["descriptive_config"] == L_k("JOINT-PAIR")
    # missing guard comparator: every T* candidate ineligible -> P*, N*, J* refuse with MISSING_GUARD_COMPARATOR
    bad_util = (lambda v: {k: {**x, "util": {1: x["util"][1], 2: {**x["util"][2], "logloss": x["util"][2]["logloss"] + 0.05}}}
                           for k, x in v.items()})
    rows_m, _ = synthetic_bank_lcr({c_: bad_util for c_ in lcr_role_lists()["T*"]})
    Sm = my_selection_lcr(rows_m)
    m_ok = Sm["statuses"]["T*"]["status"] == "NO_ELIGIBLE_COMPARATOR" and all(
        Sm["statuses"][x]["status"] == "INVALID_NOMINEE" and Sm["statuses"][x]["reason"] == "MISSING_GUARD_COMPARATOR"
        for x in ("P*", "N*", "J*"))
    out["missing_guard_comparator"] = {x: (Sm["statuses"][x]["status"], Sm["statuses"][x].get("reason"))
                                       for x in ("T*", "P*", "N*", "J*")}
    # technical failure: a seed missing for a candidate -> the role is INVALID (FIT_OR_ADMISSION_FAILURE)
    rows_t, _ = synthetic_bank_lcr({L_k("JOINT-SINGLE"): lambda v: {k: x for k, x in v.items() if k != 2}})
    St = my_selection_lcr(rows_t)
    t_ok = St["statuses"]["N*"]["status"] == "INVALID_NOMINEE" and "FIT_OR_ADMISSION_FAILURE" in \
        St["statuses"]["N*"]["reason"]
    out["technical_failure"] = (St["statuses"]["N*"]["status"], St["statuses"]["N*"].get("reason"))
    # utility-only failure fallback: all constrained arms fail occupation LL on seed 1 -> N* minimum-shortfall fallback
    one_bad = (lambda v: {k: ({**x, "util": {1: x["util"][1], 2: {**x["util"][2], "logloss": x["util"][2]["logloss"] +
                                                                  (0.006 if k == 1 else 0.0)}}}) for k, x in v.items()})
    rows_u, _ = synthetic_bank_lcr({L_k(a_): one_bad for a_ in LCR_K})
    Su = my_selection_lcr(rows_u)
    u_ok = Su["statuses"]["N*"]["status"] == "NO_ELIGIBLE_NOMINEE" and Su["statuses"]["N*"]["reason"] == \
        "ORDINARY_UTILITY_FAILURE" and Su["statuses"]["N*"]["descriptive_config"] in [L_k(a_) for a_ in LCR_K]
    out["utility_fallback"] = (Su["statuses"]["N*"]["status"], Su["statuses"]["N*"].get("reason"),
                               Su["statuses"]["N*"].get("descriptive_config"))
    # ordering ties: equal pair AUC -> lower summed LL -> fewer states -> config ID
    a_, b_ = L_k("SEQ-12"), L_k("SEQ-21")
    ra, rb = dict(rows[a_]), dict(rows[b_])
    ra["mean_pair"] = rb["mean_pair"] = 0.8
    ra["mean_sum_logloss"] = rb["mean_sum_logloss"] = 1.0
    ra["mean_states"] = rb["mean_states"] = 280.0
    tie_ok = min([ra, rb], key=okey)["config"] == min(a_, b_)
    # R-4: an infeasible constrained fit (best pair AUC) is never a nominee, leaves C_pair*, and gives
    # CONSTRAINED_FIT_INFEASIBLE when no candidate of the role is fit-feasible; R-5: fit-feasible fallbacks first
    infeas = (lambda v: {k: {**x, "fit_feasible": k != 1} for k, x in v.items()})
    rows_f, _ = synthetic_bank_lcr({L_k("JOINT-PAIR"): infeas, L_k("JOINT-SINGLE"): lambda v: {
        k: {**x, "auc": {**x["auc"], "pair": 0.810}} for k, x in v.items()}})
    Sf = my_selection_lcr(rows_f)
    r4_ok = (Sf["statuses"]["J*"]["status"] == "NO_ELIGIBLE_NOMINEE" and
             Sf["statuses"]["J*"]["reason"] == "CONSTRAINED_FIT_INFEASIBLE" and
             Sf["statuses"]["J*"]["descriptive_config"] == L_k("JOINT-PAIR") and
             Sf["resolved"]["N*"] == L_k("JOINT-SINGLE") and Sf["resolved"]["P*"] == L_k("JOINT-SINGLE") and
             Sf["resolved"]["C_pair*"] == L_k("JOINT-SINGLE") and rows_f[L_k("JOINT-PAIR")]["ordinary_inner"] and
             not rows_f[L_k("JOINT-PAIR")]["ordinary_eligible"])
    out["r4_infeasible_constrained_fit"] = {x: Sf["resolved"][x] for x in ("P*", "N*", "C_pair*", "J*")} | {
        "J*_reason": Sf["statuses"]["J*"].get("reason")}
    bad_util_s1 = (lambda v: {k: {**x, "util": {1: x["util"][1], 2: {**x["util"][2], "logloss": x["util"][2]["logloss"] +
                                                                    (0.006 if k == 1 else 0.0)}}} for k, x in v.items()})
    ov = {L_k(a_): bad_util_s1 for a_ in LCR_K if a_ != "LOCAL"}
    ov[L_k("LOCAL")] = lambda v: {k: {**x, "fit_feasible": False} for k, x in v.items()}
    rows_r5, _ = synthetic_bank_lcr(ov)
    S5 = my_selection_lcr(rows_r5)
    r5_ok = (S5["statuses"]["N*"]["status"] == "NO_ELIGIBLE_NOMINEE" and S5["statuses"]["N*"]["reason"] ==
             "ORDINARY_UTILITY_FAILURE" and S5["statuses"]["N*"]["descriptive_config"] != L_k("LOCAL") and
             S5["statuses"]["N*"]["fallback_fit_feasible"] is True)
    out["r5_fit_feasible_fallback_first"] = (S5["statuses"]["N*"].get("reason"), S5["statuses"]["N*"].get(
        "descriptive_config"))
    ok = (r4_ok and r5_ok and out["expected_roles"] and out["raw_j_excluded_from_T"] and
          out["c_pair_includes_sources_and_references"] and
          out["d1_privacy_not_task_only"] and
          out["d0_large_lambda_ineligible_d1_eligible"] and out["no_headroom_buffer"] and g_ok and m_ok and t_ok and
          u_ok and tie_ok)
    return res("PASS" if ok else "FAIL", **out, tie_rule=bool(tie_ok))


def synthetic_endpoints_lcr(fail_id=None, how="precision", nonfinite_id=None, pass_claims=("A", "B", "C", "Q")):
    eps = []
    for claim in ("A", "B", "C"):
        for sid, (kind, task, tgt, side) in zip(LCR_SLOTS[claim], lcr_clause_specs(claim)):
            good = claim in pass_claims
            if side == "lower>":
                pt = tgt + 0.01 if good else tgt - 0.01
                lo, hi = pt - 0.005, pt + 0.005
            else:
                pt = tgt - 0.004 if good else tgt + 0.004
                lo, hi = pt - 0.002, pt + 0.002
            eps.append({"id": sid, "claim": claim, "point": pt, "lower": lo, "upper": hi, "target": tgt, "side": side})
    for sid, tgt in zip(LCR_SLOTS["Q"], (0.01, 0.01, 0.005, 0.005)):
        good = "Q" in pass_claims
        pt = tgt - 0.004 if good else tgt + 0.004
        eps.append({"id": sid, "claim": "Q", "point": pt, "lower": pt - 0.002, "upper": pt + 0.002, "target": tgt,
                    "side": "upper<"})
    for e in eps:
        if e["id"] == fail_id:
            if how == "precision":                       # point passes, bound does not
                if e["side"] == "lower>":
                    e["point"], e["lower"], e["upper"] = e["target"] + 0.001, e["target"] - 0.002, e["target"] + 0.004
                else:
                    e["point"], e["lower"], e["upper"] = e["target"] - 0.001, e["target"] - 0.004, e["target"] + 0.002
            else:                                       # measured violation
                if e["side"] == "lower>":
                    e["point"], e["lower"], e["upper"] = e["target"] - 0.01, e["target"] - 0.015, e["target"] - 0.005
                else:
                    e["point"], e["lower"], e["upper"] = e["target"] + 0.01, e["target"] + 0.005, e["target"] + 0.015
        if e["id"] == nonfinite_id:
            e["upper"] = None
    return eps


def selftest_labels_lcr():
    rows, _ = synthetic_bank_lcr()
    st = my_selection_lcr(rows)["statuses"]
    out = {}
    z = NormalDist().inv_cdf(1 - 0.05 / 74)
    out["z"] = {"value": z, "equals_registered": z == Z_LCR, "slots": sum(len(v) for v in LCR_SLOTS.values())}
    lab = labels_lcr(st, synthetic_endpoints_lcr())
    out["all_pass"] = lab["label"]
    a_only = labels_lcr(st, synthetic_endpoints_lcr(pass_claims=("A",)))["label"]
    out["A_only"] = a_only
    one = labels_lcr(st, synthetic_endpoints_lcr(fail_id="P05", pass_claims=("A", "Q")))
    out["A_one_precision_failure"] = (one["claims"]["A"], one["root_causes"]["A"], one["label"])
    viol = labels_lcr(st, synthetic_endpoints_lcr(fail_id="P07", how="violation", pass_claims=("A",)))
    out["A_measured_violation"] = (viol["claims"]["A"], viol["root_causes"]["A"])
    nf = labels_lcr(st, synthetic_endpoints_lcr(nonfinite_id="P13"))
    out["B_nonfinite"] = (nf["claims"]["B"], nf["root_causes"]["B"], nf["label"])
    qonly = labels_lcr(st, synthetic_endpoints_lcr(pass_claims=("Q",)))["label"]
    none_ = labels_lcr(st, synthetic_endpoints_lcr(pass_claims=()))["label"]
    gate = labels_lcr(st, synthetic_endpoints_lcr(), gate_passed=False)["label"]
    qinc = labels_lcr(st, synthetic_endpoints_lcr(nonfinite_id="P13", pass_claims=("Q",)))
    out["Q_pass_with_incomplete_B"] = (qinc["label"], qinc["incomplete_displayed"])
    nom_none = {**st, "N*": {"status": "NO_ELIGIBLE_NOMINEE", "config": None, "reason": "ORDINARY_UTILITY_FAILURE"}}
    nn = labels_lcr(nom_none, synthetic_endpoints_lcr(pass_claims=("A",)))
    out.update({"Q_only": qonly, "none": none_, "gate_failed": gate, "B_no_nominee": (nn["claims"]["B"], nn["label"])})
    ok = (out["z"]["equals_registered"] and out["z"]["slots"] == 37 and
          lab["label"].startswith("PRIVACY_RELEASE_DEVELOPMENT_CRITERION_MET (JOINT-PAIR; constrained)") and
          "CONSTRAINED_SEARCH_INCREMENT_ESTABLISHED" in lab["label"] and "PAIRED_JOINT_INCREMENT_ESTABLISHED" in lab["label"]
          and a_only == "PRIVACY_RELEASE_DEVELOPMENT_CRITERION_MET (JOINT-PAIR; constrained)"
          and one["claims"]["A"] == "NOT_ESTABLISHED" and one["root_causes"]["A"] == "ASSESSMENT_PRECISION_FAILURE"
          and one["label"] == "CONFIDENCE_FEASIBILITY_ESTABLISHED_NO_METHOD_CRITERION"
          and viol["root_causes"]["A"] == "MEASURED_VIOLATION_SUPPORTED_BY_BOUND"
          and nf["claims"]["B"] == "INCOMPLETE_OR_INVALID" and "INCOMPLETE" not in nf["label"].split(" + ")[0]
          and "B" in nf["incomplete_displayed"]
          and qonly == "CONFIDENCE_FEASIBILITY_ESTABLISHED_NO_METHOD_CRITERION" and none_ == "EXPERIMENTAL_NO_ADVANTAGE"
          and gate == "MECHANISM_GATE_NOT_MET" and nn["claims"]["B"] == "NOT_ESTABLISHED_NO_ELIGIBLE_NOMINEE"
          and qinc["label"] == "CONFIDENCE_FEASIBILITY_ESTABLISHED_NO_METHOD_CRITERION" and qinc["incomplete_displayed"] == ["B"])
    return res("PASS" if ok else "FAIL", **out)


# ---- the section-14 deliberate defects (each must be CAUGHT by the verifier's own check)
def selftest_defects_lcr(rng):
    out = {}
    K, d, n = 6, 2, 600
    P, y = _tok_rows(rng, n, K, d, lab=[0.10, 0.10, 0.30, 0.35, 0.10, 0.05])
    Y, S = np.bincount(y, minlength=K).astype(float), P.sum(0)
    u, q, _ = own_d1_solve(n, Y, S, d)
    good = {"n": n, "y": Y, "S": S, "u": u, "q": q, "hash": suff_hash(K, d, n, Y, S)}
    base_ok = check_decoder_entry(good, y, P, d)["ok"]

    def caught(entry):
        c = check_decoder_entry(entry, y, P, d)
        return base_ok and not c["ok"], c
    # 1. wrong label counts in the decoder (one label moved between classes; the other recipient's labels)
    Yw = Y.copy()
    Yw[3] -= 5
    Yw[0] += 5
    uw, qw, _ = own_d1_solve(n, Yw, S, d)
    c1, r1 = caught({"n": n, "y": Yw, "S": S, "u": uw, "q": qw, "hash": suff_hash(K, d, n, Yw, S)})
    out["wrong_label_counts"] = {"caught": c1, "counts_exact": r1["counts_exact"], "max_dq": r1["max_dq_vs_own"]}
    # 2. stale cache key: a cache keyed by (K, d, n, y) only reuses a solve with different teacher sums
    stale = D1Cache(key_fn=lambda K_, d_, n_, y_, S_: hashlib.sha256(np.asarray(y_, float).tobytes() +
                                                                     f"{K_}|{d_}|{n_}".encode()).hexdigest())
    P2_ = rng.dirichlet(np.ones(K) * 3.0, n)
    P2_[:, d] += 0.8
    P2_ /= P2_.sum(1, keepdims=True)
    S2 = P2_.sum(0)
    stale.solve(n, Y, S2, d)                          # first token: same labels, other teacher sums
    us, qs, _ = stale.solve(n, Y, S, d)               # second token gets the stale solve
    c2, r2 = caught({"n": n, "y": Y, "S": S, "u": us, "q": qs, "hash": suff_hash(K, d, n, Y, S)})
    own_keys_differ = suff_hash(K, d, n, Y, S) != suff_hash(K, d, n, Y, S2)
    out["stale_cache_key"] = {"caught": c2 and own_keys_differ and stale.hits == 1, "max_dq": r2["max_dq_vs_own"],
                              "own_full_key_distinguishes": own_keys_differ}
    # 3. omitted prior (kappa = 0)
    u3, q3, _ = own_d1_solve(n, Y, S, d, kappa=0.0)
    c3, r3 = caught({"n": n, "y": Y, "S": S, "u": u3, "q": q3, "hash": good["hash"]})
    out["omitted_prior"] = {"caught": c3, "max_dq": r3["max_dq_vs_own"], "fw_gap_rel": r3["fw_gap_rel"]}
    # 4. class constraint omitted (labels favour class 3 over the predicted class 2)
    u4, q4, _ = own_d1_solve(n, Y, S, d, drop_dominance=True)
    c4, r4 = caught({"n": n, "y": Y, "S": S, "u": u4, "q": q4, "hash": good["hash"]})
    out["class_constraint_omitted"] = {"caught": c4 and int(np.argmax(q4)) != d, "argmax_of_defective_q": int(np.argmax(q4)),
                                       "strict_argmax": r4["strict_argmax"]}
    # fitting-row engine for the budget / search defects
    sf = synth_fitting(rng)
    eng = engine_of(sf)
    ident = {i: np.arange(sf[i]["ncells"]) for i in (1, 2)}
    cls_only = {i: sf[i]["cell_class"].copy() for i in (1, 2)}
    st = eng.pair(ident[1], ident[2])
    # 5. proxy KL substituted for the true-label log loss in the budget term
    Pi, qi, yi = eng.P[2], st["r2"]["dec"]["q"][st["r2"]["tok"]], eng.y[2]
    proxy = eng.U[2][0] + float(np.mean(kl_paired(Pi, qi)))
    true_ = fit_losses(qi, yi)[0]
    rec5 = {"L": proxy}
    c5 = abs(rec5["L"] - true_) > D1_TOL["loss_mean"]
    out["proxy_kl_for_true_logloss"] = {"caught": bool(c5), "recorded_proxy": proxy, "own_true_label": true_,
                                        "abs_diff": abs(proxy - true_)}
    # 6. missing Brier constraint: a state with LL inside its budget but Brier outside is accepted by an LL-only rule
    LU, BU = eng.U[2]
    bc_full = budget_check(LU + 0.004, BU + 0.0035, LU, BU)
    ll_only = (LU + FIT_BUDGET["ll"]) - (LU + 0.004) >= 0
    out["missing_brier_constraint"] = {"caught": bool(ll_only and not bc_full["ok"] and not bc_full["brier_ok"]),
                                       "own_verdict": bc_full["ok"], "ll_only_verdict": bool(ll_only)}
    # 7. temporary sequential partner wrongly required feasible (SEQ-12 stage 1 with recipient 2 CLASS-ONLY)
    r2c = eng.recipient(2, cls_only[2])
    partner_feasible, _ = eng.feasible(2, r2c)
    acc_right, acc_wrong, tried = 0, 0, 0
    cc = sf[1]["cell_class"]
    for cell in range(sf[1]["ncells"]):
        tgt = [c_ for c_ in range(sf[1]["ncells"]) if cc[c_] == cc[cell] and c_ != cell][0]
        m1 = ident[1].copy()
        m1[cell] = tgt
        r1_ = eng.recipient(1, m1)
        tried += 1
        acc_right += seq_stage1_feasible(eng, 1, r1_, r2c)
        acc_wrong += seq_stage1_feasible(eng, 1, r1_, r2c, partner_required=True)
    out["temporary_partner_required_feasible"] = {"caught": bool((not partner_feasible) and acc_right > 0 and acc_wrong == 0),
                                                   "class_only_partner_feasible": bool(partner_feasible),
                                                   "stage1_moves": tried, "feasible_registered_rule": acc_right,
                                                   "feasible_defective_rule": acc_wrong}
    # 8. infeasible paired update: an accepted atomic pair whose post-state violates recipient 2's budget
    post = eng.pair(ident[1], cls_only[2])
    rec8 = {"accepted": True, "Phi_pre": st["Phi"], "Phi_post_recorded": st["Phi"] - 0.01}
    f1_, _ = eng.feasible(1, post["r1"])
    f2_, b2_ = eng.feasible(2, post["r2"])
    accepted_ok = f1_ and f2_ and post["Phi"] < rec8["Phi_pre"]
    out["infeasible_paired_update"] = {"caught": bool(rec8["accepted"] and not accepted_ok and not f2_),
                                       "post_recipient2_feasible": bool(f2_), "post_ll_margin": b2_["ll_margin"],
                                       "post_brier_margin": b2_["brier_margin"]}
    # 9. mean-seed eligibility

    def one_bad_seed(v):
        v = {k: dict(x) for k, x in v.items()}
        v[1] = {**_seed((0.0005, 0.0125), (0.0003, 0.002), (0.760, 0.770, 0.800), 300.0), "fit_feasible": True}
        v[0] = {**_seed((0.0005, 0.0020), (0.0003, 0.002), (0.760, 0.770, 0.800), 300.0), "fit_feasible": True}
        v[2] = {**_seed((0.0005, 0.0020), (0.0003, 0.002), (0.760, 0.770, 0.800), 300.0), "fit_feasible": True}
        return v
    _, rin = synthetic_bank_lcr({L_k("JOINT-PAIR"): one_bad_seed})
    right = lcr_config_row(L_k("JOINT-PAIR"), rin[L_k("JOINT-PAIR")])
    wrong = lcr_config_row(L_k("JOINT-PAIR"), rin[L_k("JOINT-PAIR")], seed_average=True)
    out["mean_seed_eligibility"] = {"caught": (not right["ordinary_eligible"]) and wrong["ordinary_eligible"]}
    # 10. missing comparator: a claim with every clause passing but no valid comparator is not PASS
    rows, _ = synthetic_bank_lcr()
    stt = my_selection_lcr(rows)["statuses"]
    passing = {sid: "PASS" for sid in LCR_SLOTS["B"]}
    mc = [claim_status_lcr(stt["N*"], c_, passing, "B")[0] for c_ in
          ({"status": "INVALID_COMPARATOR", "config": None}, {"status": "NO_ELIGIBLE_COMPARATOR", "config": None}, None)]
    out["missing_comparator"] = {"caught": all(x == "INCOMPLETE_OR_INVALID" for x in mc) and
                                 claim_status_lcr(stt["N*"], stt["C*"], passing, "B")[0] == "PASS", "statuses": mc}
    # 11. reversed AUC (orientation flipped or max(AUC, 1 - AUC)) and reversed best / worst ordering
    ys = rng.integers(0, 2, 600)
    sc = rng.normal(size=600) - 0.6 * ys
    a_ = my_auc(ys, sc)
    rev_role = my_selection_lcr(rows, reverse=True)["resolved"]["P*"] != my_selection_lcr(rows)["resolved"]["P*"]
    out["reversed_auc"] = {"caught": bool(a_ < 0.5 and abs(my_auc(ys, -sc) - (1 - a_)) < 1e-12 and max(a_, 1 - a_) != a_
                                          and rev_role), "auc": a_}
    # 12. pair misalignment: S = u XOR v is readable only from the aligned pair
    m = 3000
    fa_, sl_ = np.arange(0, 2000), np.arange(2000, m)
    uu, vv = rng.integers(0, 2, m), rng.integers(0, 2, m)
    sx = uu ^ vv
    al = my_cc_pair(uu, vv, sx, fa_, sl_, np.arange(len(sl_)), 1.0)[0]
    v2 = vv.copy()
    v2[sl_] = rng.permutation(vv[sl_])
    sh = my_cc_pair(uu, v2, sx, fa_, sl_, np.arange(len(sl_)), 1.0)[0]
    out["pair_misalignment"] = {"caught": bool(my_auc(sx[sl_], al) > 0.99 and abs(my_auc(sx[sl_], sh) - 0.5) < 0.06),
                                "aligned_auc": my_auc(sx[sl_], al), "misaligned_auc": my_auc(sx[sl_], sh)}
    # 13. omitted token identities: two IDs share one decoded vector; the bit lives only in the identity
    s3 = rng.integers(0, 2, m)
    b3 = rng.integers(0, 4, m)
    tok = 2 * b3 + s3
    q3_ = np.stack([0.6 + 0.05 * b3, 0.4 - 0.05 * b3], 1)
    full = np.hstack([np.eye(8)[tok], q3_])
    noid = q3_
    af = my_auc(s3[sl_], my_p1(my_attacker("LR_C1.0", 0).fit(full[fa_], s3[fa_]), full[sl_]))
    an = my_auc(s3[sl_], my_p1(my_attacker("LR_C1.0", 0).fit(noid[fa_], s3[fa_]), noid[sl_]))
    out["omitted_token_identities"] = {"caught": bool(af > 0.95 and abs(an - 0.5) < 0.06), "full_interface_auc": af,
                                       "probability_only_auc": an}
    # 14. clean-output bypass: clean teacher probabilities appended or substituted on some rows
    tokb = st["r2"]["tok"]
    qrows = st["r2"]["dec"]["q"][tokb].copy()
    byp = qrows.copy()
    byp[::7] = eng.P[2][::7]
    out["clean_output_bypass"] = {"caught": bool(release_keys_ok(RELEASE_KEYS_LCR) and
                                                 not release_keys_ok(set(RELEASE_KEYS_LCR) | {"p1"}) and
                                                 not release_keys_ok(set(RELEASE_KEYS_LCR) | {"fine2"}) and
                                                 release_is_token_function(tokb, qrows) and
                                                 not release_is_token_function(tokb, byp))}
    # 15. missing source composition winner: the closure refuses an incomplete bank; the winner changes when dropped
    bank = lcr_code_ids()
    cand = [(c_, 0.80 + (0.07 if c_ == L_k("JOINT-PAIR") else 0.0)) for c_ in bank]
    w_full = compose_first_better(0.85, cand)[0]
    w_miss = compose_first_better(0.85, [x for x in cand if x[0] != w_full])[0]
    out["missing_source_composition_winner"] = {
        "caught": bool(lcr_composition_closure_ok(bank) and not lcr_composition_closure_ok([c_ for c_ in bank
                                                                                            if c_ != w_full])
                       and w_full != w_miss), "bank_size": len(bank), "winner": w_full}
    ok = all(v["caught"] for v in out.values())
    return res("PASS" if ok else "FAIL", defects_tested=len(out), caught=sum(bool(v["caught"]) for v in out.values()),
               baseline_entry_passes=bool(base_ok), **out)


# ================================================================================================ lcr PHASE 1: source admission
STATEMENT_LCR = ("This study is motivated by opened Adult development results from qpc and cbp and the earlier PCRL lineage. "
                 "All real-data roles have been used historically. This is exploratory development evidence. Nominal "
                 "intervals condition on fitted artifacts and do not correct for the adaptive research history. A masked "
                 "assessment prevents additional selection leakage but does not make the rows fresh. No confirmation "
                 "population is opened.")
LCR_ADMITTED_COUNTS = {"tea": 6, "ref": 9, "fine": 3, "pol": 81, "inner": 90}


def lcr_d0_codes():
    return [L_d0("DIRECT-TASK"), L_d0("FINE-TASK"), L_d0("CLASS")] + [L_d0(f, l_) for l_ in LAMS for f in LCR_PRIV]


def lcr_pol_unit(k, cid):
    return f"pol__s{k}__{cid.replace('|', '_')}"


def lcr_admitted_units(k):
    base = [f"tea__s{k}__{t}" for t in TEACHERS] + [f"ref__s{k}__{r}" for r in REFS] + [f"fine__s{k}"]
    pols = [lcr_pol_unit(k, c_) for c_ in lcr_d0_codes()]
    return base + pols + [f"inner__{x}" for x in pols + [f"ref__s{k}__{r}" for r in REFS]]


def check_pins_lcr():
    """Input pin, branch base = cbp final tip, SOURCE_INDEX (every pinned module equal to the cbp tip blob and to the
    worktree; source results hashes at the tip; no pinned package edited), SOURCE_ADMISSION_LOCK (committed, on origin
    byte-identical, code hashes at its first commit, verbatim exposure statement, input pins, budget) and the public
    SOURCE_ADMISSION.json counts."""
    out, fails = {}, []
    out["input_sha256_ok"] = sha_file(SRC) == SRC_SHA
    out["branch_base_is_cbp_tip"] = git("merge-base", "HEAD", CBP_TIP) == CBP_TIP
    si = jload(RES / "SOURCE_INDEX.json") if (RES / "SOURCE_INDEX.json").exists() else None
    if si is None:
        fails.append("SOURCE_INDEX.json absent")
    else:
        bad = []
        for f_, v in si.get("files", {}).items():
            blob = git_show_bytes(CBP_TIP, f_)
            own = sha_file(WT / f_) if (WT / f_).exists() else None
            if blob is None or hashlib.sha256(blob).hexdigest() != v.get("sha256") or own != v.get("sha256"):
                bad.append(f_)
        out["source_index_files"] = len(si.get("files", {}))
        out["source_index_mismatches"] = bad
        rbad = []
        for f_, h in (si.get("source_results_hashes") or {}).items():
            blob = git_show_bytes(CBP_TIP, f"results/{f_}")
            if blob is None or hashlib.sha256(blob).hexdigest() != (h.get("sha256") if isinstance(h, dict) else h):
                rbad.append(f_)
        out["source_results_hashes"] = len(si.get("source_results_hashes") or {})
        out["source_results_mismatches_at_cbp_tip"] = rbad
        out["source_tip_field_ok"] = si.get("source_tip") == CBP_TIP
        if bad or rbad or not si.get("all_equal_to_source_tip") or not out["source_tip_field_ok"]:
            fails.append("SOURCE_INDEX")
        edited = []
        for top in ("cbp", "qpc", "dpc", "osf", "smf", "rgj", "jcv", "stored_model_eval", "oar"):
            for p in sorted((WT / top).glob("*.py")) if (WT / top).exists() else []:
                rel = str(p.relative_to(WT))
                blob = git_show_bytes(CBP_TIP, rel)
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
        b = L_.get("budget") or {}
        inp = L_.get("inputs") or {}
        sa = {"committed": bool(c0), "versions": len(commits),
              "on_origin_byte_identical": shown is not None and shown == sl.read_bytes(), "first_commit": c0,
              "first_push_time": first_remote(c0, remote_reflog()) if c0 else None,
              "code_files": len(L_.get("code_files") or {}), "code_files_differing_at_lock_commit": mism,
              "statement_equals_prompt": L_.get("statement") == STATEMENT_LCR,
              "input_pin_ok": inp.get("source_npz_sha256") == SRC_SHA,
              "evidence_pin_ok": inp.get("source_evidence_commit") == CBP_TIP, "parent_is_cbp_tip": L_.get(
                  "parent_commit") == CBP_TIP,
              "budget_ok": (b.get("elapsed_h") == 10 and b.get("cpu_h") == 20 and b.get("heavy_processes_total") == 2 and
                            b.get("memory_gib") == 8 and b.get("reserve_final_h") == 2 and
                            b.get("reserve_final_cpu_h") == 4 and b.get("start") == STUDY_START)}
        out["source_admission_lock"] = sa
        docs = L_.get("documents_sha256") or {}
        sa["documents_at_lock_commit"] = {d_: (lambda bb: bb is not None and hashlib.sha256(bb).hexdigest() == h)(
            git_show_bytes(c0, f"{REL_RES}/{d_}") if c0 else None) for d_, h in docs.items()}
        if not (sa["committed"] and sa["on_origin_byte_identical"] and not mism and sa["statement_equals_prompt"] and
                sa["input_pin_ok"] and sa["evidence_pin_ok"] and sa["budget_ok"] and sa["first_push_time"] and
                all(sa["documents_at_lock_commit"].values())):
            fails.append("SOURCE_ADMISSION_LOCK")
    A = jload(RES / "SOURCE_ADMISSION.json") if (RES / "SOURCE_ADMISSION.json").exists() else None
    if A is None:
        fails.append("SOURCE_ADMISSION.json absent")
    else:
        own_units = {u for k in SEEDS for u in lcr_admitted_units(k)}
        out["source_admission_doc"] = {"units": len(A.get("units") or {}), "units_equal_own_list": set(A.get(
            "units") or {}) == own_units, "counts": A.get("counts"), "verdict_field_free": True}
        if not out["source_admission_doc"]["units_equal_own_list"] or out["source_admission_doc"]["units"] != 189:
            fails.append("SOURCE_ADMISSION.json units")
    ok = not fails and out["input_sha256_ok"] and out["branch_base_is_cbp_tip"]
    return res("PASS" if ok else "FAIL", failures=fails, **out)


def check_custody_lcr():
    """Every admitted unit (189), teacher artifact directory (6) and deployment input (2): own SHA-256 of the lcr bytes
    equal the live cbp store, the cbp same-device copy SHA256SUMS and (scored units) the cbp EVALUATION_LOCK pin at the
    cbp tip; each unit's COMPLETE.json is internally consistent; SOURCE_ADMISSION.json lists the same hashes; the private
    receipt says ADMITTED; no unexpected unit."""
    sums = copy_sums(CBP_COPY)
    el = json.loads(git_show_bytes(CBP_TIP, f"{CBP_REL}/EVALUATION_LOCK.json") or b"{}")
    lockf = {}
    for s_ in (el.get("seeds") or {}).values():
        lockf.update(s_.get("unit_file_sha256") or {})
    pub = (jload(RES / "SOURCE_ADMISSION.json") if (RES / "SOURCE_ADMISSION.json").exists() else {}).get("units") or {}
    out, bad = {"units": 0, "files": 0, "kinds": {}}, []
    expected = [u for k in SEEDS for u in lcr_admitted_units(k)]
    for u in expected:
        ld, cd = UNITS / u, CBP_UNITS / u
        if not ld.exists():
            bad.append(f"{u}: absent from the lcr store")
            continue
        lh, ch = dir_hashes(ld), (dir_hashes(cd) if cd.exists() else {})
        out["units"] += 1
        out["files"] += len(lh)
        kd = u.split("__")[0]
        out["kinds"][kd] = out["kinds"].get(kd, 0) + 1
        if lh != ch:
            bad.append(f"{u}: bytes differ from the live cbp store")
        for f_, h in lh.items():
            if sums.get(f"cbp_v1/run/units/{u}/{f_}") != h:
                bad.append(f"{u}/{f_}: differs from / absent in the cbp copy SHA256SUMS")
        if u in lockf and lockf[u] != {f_: lh.get(f_) for f_ in lockf[u]}:
            bad.append(f"{u}: differs from the cbp EVALUATION_LOCK pin")
        cu, _ = complete_ok(ld)
        if not (cu["id_ok"] and cu["rehash_ok"] and not cu["unlisted"]):
            bad.append(f"{u}: COMPLETE.json inconsistent")
        pf = (pub.get(u) or {}).get("files_sha256")
        if pf != lh:
            bad.append(f"{u}: differs from / absent in SOURCE_ADMISSION.json")
    out["scored_units_checked_against_cbp_evaluation_lock"] = sum(1 for u in expected if u in lockf)
    adm = {}
    for k in SEEDS:
        for t in TEACHERS:
            a_ = f"rel__s{k}__{t}"
            lh = dir_hashes(ADM / a_) if (ADM / a_).exists() else {}
            ch = dir_hashes(CBP_ADM / a_) if (CBP_ADM / a_).exists() else {}
            adm[a_] = lh == ch and bool(lh) and all(sums.get(f"cbp_v1/admitted/{a_}/{f_}") == h for f_, h in lh.items())
            if not adm[a_]:
                bad.append(f"admitted {a_}: differs from the cbp admitted copy / SHA256SUMS")
    out["admitted_teacher_dirs_equal"] = adm
    inp = {}
    for f_ in ("deploy_input.npz", "schema.json"):
        p, q = PRIV / "inputs" / f_, CBP_PRIV / "inputs" / f_
        inp[f_] = p.exists() and q.exists() and sha_file(p) == sha_file(q) == sums.get(f"cbp_v1/inputs/{f_}")
        if not inp[f_]:
            bad.append(f"inputs/{f_}")
    out["inputs_equal"] = inp
    rc = jload(ADM / "ADMISSION_RECEIPT.json") if (ADM / "ADMISSION_RECEIPT.json").exists() else {}
    out["receipt_verdict"] = rc.get("verdict")
    out["receipt_teacher_parity_all_ok"] = len(rc.get("teachers") or {}) == 6 and all(
        v.get("ok") for v in rc["teachers"].values())
    out["receipt_code_parity_all_ok"] = len(rc.get("codes") or {}) == 81 and all(v.get("ok") for v in rc["codes"].values())
    if rc.get("verdict") != "ADMITTED" or not out["receipt_teacher_parity_all_ok"] or not out["receipt_code_parity_all_ok"]:
        bad.append("admission receipt")
    present = sorted(q.name for q in UNITS.iterdir() if q.is_dir()) if UNITS.exists() else []
    unexpected = [n_ for n_ in present if n_.startswith(("tea__", "ref__", "fine__", "pol__", "inner__")) and
                  n_ not in set(expected)]
    out["unexpected_admitted_namespace_units"] = unexpected
    if unexpected:
        bad.append(f"unexpected units {unexpected[:5]}")
    ok = not bad and out["units"] == 189 and out["kinds"] == LCR_ADMITTED_COUNTS
    return res("PASS" if ok else "FAIL", failures=bad[:30], n_failures=len(bad), **out,
               rule="own SHA-256 of every file of the lcr admitted units vs the live cbp store, the cbp copy SHA256SUMS "
                    "and (scored units) the cbp EVALUATION_LOCK at 7f3ec67")


def check_teachers_lcr(D: Data, L, refit=False):
    """Own forward application of every admitted teacher (lcr admitted copy) on own X vs the lcr and cbp tea__ units and
    the admitted release; custody vs the cbp admitted copy and the pinned osf MODEL_MANIFEST; head structure and scaler
    moments; optional own head refit."""
    man = pinned_osf_manifest()
    entries = {(u["label"], u["seed"], u["unit"]): u for u in man.get("units", [])}
    out, T = {}, {}
    tr = D.idx[FIT]
    for k in SEEDS:
        for t in TEACHERS:
            name = f"rel__s{k}__{t}"
            ld, cd_ = ADM / name, CBP_ADM / name
            info, fails = {"unit": name}, []
            entry = entries.get((TEACHER_LABEL[t], k, name))
            cl, fl = complete_ok(ld)
            cc, fc = complete_ok(cd_)
            info["custody"] = {"lcr_copy_complete": cl, "cbp_copy_complete": cc, "lcr_files_equal_cbp": fl == fc,
                               "files_equal_pinned_osf_manifest": bool(entry) and fl == entry.get("complete_files_sha256")}
            if not (cl["id_ok"] and cl["rehash_ok"] and not cl["unlisted"] and fl == fc and
                    info["custody"]["files_equal_pinned_osf_manifest"]):
                fails.append("custody")
            mine = own_teacher(ld / "model.pt", [ld / "head_0.joblib", ld / "head_1.joblib"], D.X)
            tl = np.load(UNITS / f"tea__s{k}__{t}" / "teacher.npz", allow_pickle=False)
            tc = np.load(CBP_UNITS / f"tea__s{k}__{t}" / "teacher.npz", allow_pickle=False)
            rel = np.load(ld / "release.npz", allow_pickle=False)
            trec = jload(UNITS / f"tea__s{k}__{t}" / "record.json")
            info["row_order_equals_own"] = all(bool(np.array_equal(z["row_id"], D.row_id)) for z in (tl, tc, rel))
            if not info["row_order_equals_own"]:
                fails.append("row order")
            par = {}
            for key in ("r1", "r2", "c1", "c2", "p1", "p2", "d1", "d2"):
                a = mine[key]
                rk = key if key[0] != "d" else f"hard{key[1]}"
                par[key] = {"lcr_unit": bitwise(a.astype(tl[key].dtype), tl[key]) and a.dtype == tl[key].dtype,
                            "cbp_unit": bitwise(a.astype(tc[key].dtype), tc[key]),
                            "admitted_release": (rk not in rel.files) or bitwise(a.astype(rel[rk].dtype), rel[rk]),
                            "max_abs_diff_lcr_unit": maxdiff(a, tl[key])}
            info["parity"] = par
            if not all(v["lcr_unit"] and v["cbp_unit"] and v["admitted_release"] for v in par.values()):
                fails.append("forward parity")
            info["record_model_sha_equals_own"] = trec.get("model_sha256") == sha_file(ld / "model.pt") == fl.get("model.pt")
            if not info["record_model_sha_equals_own"]:
                fails.append("teacher binding hash")
            heads = {}
            for i in (0, 1):
                j = i + 1
                head = mine[f"head{j}"]
                ref = StandardScaler().fit(mine[f"r{j}"][tr])
                P = mine[f"p{j}"]
                hi = {"pipeline_scaler_logreg": isinstance(head, Pipeline) and len(head.steps) == 2,
                      "scaler_moments_bitwise_OSF_DEFENSE_FIT": bool(np.array_equal(head[0].mean_, ref.mean_) and
                                                                     np.array_equal(head[0].var_, ref.var_)),
                      "finite": bool(np.isfinite(P).all()), "min_probability": float(P.min()),
                      "rows_with_exact_max_ties": int(np.sum((P == P.max(1, keepdims=True)).sum(1) > 1)),
                      "predicted_class_counts_fit": np.bincount(mine[f"d{j}"][D.mask[FIT]], minlength=KS[i]).tolist()}
                if not all(v for v in hi.values() if isinstance(v, bool)):
                    fails.append(f"recipient {j} head")
                heads[f"recipient_{j}"] = hi
            info["heads"] = heads
            info["failures"] = fails
            T[(k, t)] = {kk: mine[kk] for kk in ("p1", "p2", "d1", "d2", "r1", "r2", "c1", "c2")}
            T[(k, t)]["model_sha"] = fl.get("model.pt")
            out[f"{t}|s{k}"] = res("FAIL" if fails else "PASS", **info)
    st = worst(*[v["status"] for v in out.values()]) if len(out) == 6 else "FAIL"
    return res(st, units=out, note="own forward pass on own X (83 permitted columns) from the lcr admitted copies"), T


def check_d0_reencode(D: Data, T):
    """Every admitted D0 code (27 per seed): own re-encode from its policy.json and the own U teacher on all rows equals
    the stored release.npz bitwise (keys, row ids, tokens, decoded vectors, decisions, alphabets); decisions equal the
    teacher's; class preservation (strict argmax of every released vector)."""
    out, bad = {}, []
    for k in SEEDS:
        P = {1: T[(k, "U")]["p1"], 2: T[(k, "U")]["p2"]}
        d = {1: T[(k, "U")]["d1"], 2: T[(k, "U")]["d2"]}
        for cid in lcr_d0_codes():
            un = lcr_pol_unit(k, cid)
            z = np.load(UNITS / un / "release.npz", allow_pickle=False)
            mine = policy_pair_release(UNITS / un / "policy.json", P, d)
            r = {"keys_equal": frozenset(z.files) == RELEASE_KEYS_LCR,
                 "row_id": bool(np.array_equal(z["row_id"], D.row_id))}
            for x in ("tok1", "q1", "hard1", "tok2", "q2", "hard2"):
                r[x] = bitwise(np.asarray(mine[x]).astype(z[x].dtype), z[x])
            r["alpha"] = int(z["alpha1"]) == int(mine["alpha1"]) and int(z["alpha2"]) == int(mine["alpha2"])
            r["decisions_equal_teacher"] = bool(np.array_equal(z["hard1"], d[1]) and np.array_equal(z["hard2"], d[2]))
            r["strict_argmax"] = strict_argmax_ok(z["q1"], z["hard1"]) and strict_argmax_ok(z["q2"], z["hard2"])
            r["token_function"] = release_is_token_function(z["tok1"], z["q1"]) and release_is_token_function(z["tok2"],
                                                                                                             z["q2"])
            if not all(r.values()):
                bad.append(un)
            out[un] = all(r.values())
            OWN_REL[(k, cid)] = {x: np.asarray(z[x]) for x in z.files}
    return res("PASS" if not bad and len(out) == 81 else "FAIL", units=len(out), failures=bad,
               rule="own qpc-convention re-encode (own assignment of the own teacher to the fine partition, the stored "
                    "cell->token map, prototypes recomputed from the cell sums) bitwise vs release.npz on every row")


# ================================================================================================ lcr PHASE 1: fixture replay
# Written from FIXTURE_LAWS.json / FIXTURE_GATE_RULE.json (registered at dd1cf23); lcr.fixtures is never imported. The
# registered format conventions used here (laws hash = sha256 of the sorted-key JSON body without its own hash field;
# atoms [f1, f2, s, y1, y2, count]; north-west-corner joint labels, classes ascending) are read from the files' own text.
GATE_TOL = {"TOL_BUDGET": 1e-12, "TOL_MI": 1e-12, "TOL_TERMS": 1e-10, "TOL_NULL_Q": 1e-9, "TOL_NULL_LOSS": 1e-12,
            "TOL_OPT": 1e-10, "TRIGGER_MI": 0.01, "MIN_GAIN": 0.03}
FIX_TASK_ONLY = ("U|C-TASK|i8o64|D1", "U|FINE-TASK|i8o64|D1", "U|DIRECT-TASK|i8o64|D1", "U|CLASS|i1o1|D1")


def registered_laws_hash(body):
    z = {k: v for k, v in body.items() if k != "laws_sha256"}
    return hashlib.sha256(json.dumps(z, sort_keys=True, allow_nan=False).encode()).hexdigest()


def nw_corner(m1, m2):
    """North-west-corner joint table of two integer margins (classes ascending)."""
    a, b = [int(x) for x in m1], [int(x) for x in m2]
    if sum(a) != sum(b):
        raise ValueError("margins differ")
    out, i, j = {}, 0, 0
    while i < len(a) and j < len(b):
        x = min(a[i], b[j])
        if x > 0:
            out[(i, j)] = out.get((i, j), 0) + x
        a[i] -= x
        b[j] -= x
        if a[i] == 0:
            i += 1
        else:
            j += 1
    return out


def rebuild_atoms(fam):
    """Own rebuild of the atoms from the explicit tables: (f1, f2) groups of pair_counts rows, SEX split by
    sex_num / sex_den, per-recipient label margins g * labels / label_den (exact integers), NW-corner joint."""
    R = {i: fam["recipients"][str(i)] for i in (1, 2)}
    pc, sn, sd = fam["pair_counts"], fam["sex_num"], int(fam["sex_den"])
    atoms = []
    for f1, row in enumerate(pc):
        for f2, g in enumerate(row):
            g = int(g)
            if g == 0:
                continue
            g1 = Fraction(g * int(sn[f1][f2]), sd)
            if g1.denominator != 1:
                raise ValueError("non-integer SEX split")
            for s_, gs in ((0, g - int(g1)), (1, int(g1))):
                if gs == 0:
                    continue
                marg = {}
                for i, f_ in ((1, f1), (2, f2)):
                    lab, den = R[i]["cells"][f_]["labels"], int(R[i]["label_den"])
                    m_ = [Fraction(gs * int(v), den) for v in lab]
                    if any(x.denominator != 1 for x in m_):
                        raise ValueError("non-integer label split")
                    marg[i] = [int(x) for x in m_]
                for (y1, y2), c_ in sorted(nw_corner(marg[1], marg[2]).items()):
                    atoms.append([f1, f2, s_, y1, y2, int(c_)])
    return atoms


def law_static_properties(fam, atoms):
    """Own static properties (the C7 comparison): rows, per recipient cell counts, label counts, constant class
    (fitting majority, ties -> lower class) and accuracy, teacher-decision accuracy, gain, cells per class, canonical
    partitions under the cap; mapping pairs; SEX counts."""
    A = np.asarray(atoms, dtype=np.int64)
    out = {"rows": int(A[:, 5].sum())}
    npart = {}
    for i in (1, 2):
        Rr = fam["recipients"][str(i)]
        K = int(Rr["K"])
        cells = Rr["cells"]
        f_ = A[:, i - 1]
        y_ = A[:, 2 + i]
        cc = np.bincount(f_, weights=A[:, 5], minlength=len(cells)).astype(np.int64)
        lc = np.bincount(y_, weights=A[:, 5], minlength=K).astype(np.int64)
        const = int(np.argmax(lc))
        cls = np.asarray([c_["class"] for c_ in cells])
        dec = cls[f_]
        acc = Fraction(int(A[dec == y_, 5].sum()), out["rows"])
        cacc = Fraction(int(lc[const]), out["rows"])
        per_class = [int(np.sum(cls == c_)) for c_ in range(K)]
        parts = 1
        for m_ in per_class:
            parts *= stirling2_capped(m_, int(fam["caps"][i - 1]))
        npart[i] = parts
        out[f"r{i}"] = {"cell_counts": cc.tolist(), "cell_counts_match_declared": cc.tolist() == [int(c_["count"]) for c_ in
                                                                                            cells],
                        "label_counts": lc.tolist(), "constant_class": const, "constant_acc": float(cacc),
                        "teacher_decision_acc": float(acc), "accuracy_gain": float(acc - cacc),
                        "cells_per_class": per_class, "canonical_partitions": parts,
                        "teacher_strict_argmax_is_declared_class": all(
                            int(np.argmax(c_["teacher"])) == int(c_["class"]) and sorted(c_["teacher"])[-1] >
                            sorted(c_["teacher"])[-2] for c_ in cells)}
    out["mapping_pairs"] = npart[1] * npart[2]
    out["sex_counts"] = np.bincount(A[:, 2], weights=A[:, 5], minlength=2).astype(np.int64).tolist()
    return out


def law_of_family(fam):
    atoms = []
    R = {i: fam["recipients"][str(i)] for i in (1, 2)}
    for f1, f2, s_, y1, y2, c_ in fam["atoms"]:
        atoms.append({"count": c_, "f1": f1, "f2": f2, "s": s_, "y1": y1, "y2": y2,
                      "p1": [f"{v}/{R[1]['teacher_den']}" for v in R[1]["cells"][f1]["teacher"]],
                      "p2": [f"{v}/{R[2]['teacher_den']}" for v in R[2]["cells"][f2]["teacher"]]})
    return Law(atoms, {1: int(R[1]["K"]), 2: int(R[2]["K"])}, {1: int(fam["caps"][0]), 2: int(fam["caps"][1])},
               N=int(fam["N"]), name=fam["id"])


def part_key(cell_labels):
    """Canonical partition key from per-cell labels (blocks = cells sharing a label)."""
    blocks = {}
    for f_, l_ in enumerate(cell_labels):
        blocks.setdefault(l_, []).append(f_)
    return tuple(sorted(tuple(b) for b in blocks.values()))


def own_fixture_tables(law: Law, cache=None):
    """Own exhaustive tables of one registered law: per canonical map of each recipient the exact L / B under D0 and
    D1, the teacher KL distortion D of the D0 decoder (per N), I_i, tokens per class, D1 class preservation and the
    calibrated-null quantities per token; the pair I12 matrix (vectorised exact counts)."""
    cache = cache or D1Cache()
    per, toks, keys = {}, {}, {}
    null = {"max_q_diff": 0.0, "min_law_loss_gap": 0.0, "tokens": 0}
    for i in (1, 2):
        rows, tk = [], []
        Ki = law.K[i]
        for mp in law.maps(i):
            tok = law.tokens(i, mp)
            n, Y, S, cls = law.stats(i, tok)
            Q0 = smooth(S / n[:, None], cls)
            Q1 = np.stack([cache.solve(n[t_], Y[t_], S[t_], int(cls[t_]))[1] for t_ in range(len(n))])
            E = np.eye(Ki)
            L0 = float(np.sum(Y * -np.log(np.clip(Q0, LL_CLIP, 1.0)))) / law.N
            L1 = float(np.sum(Y * -np.log(np.clip(Q1, LL_CLIP, 1.0)))) / law.N
            B0 = float(sum(Y[t_, y_] * float(np.sum((Q0[t_] - E[y_]) ** 2)) for t_ in range(len(n)) for y_ in range(Ki)
                           if Y[t_, y_] > 0)) / law.N
            B1 = float(sum(Y[t_, y_] * float(np.sum((Q1[t_] - E[y_]) ** 2)) for t_ in range(len(n)) for y_ in range(Ki)
                           if Y[t_, y_] > 0)) / law.N
            p = law.p[i]
            q0r = Q0[tok]
            m = p > 0
            Dkl = float(np.sum(law.c[:, None] * np.where(m, p * (np.log(np.where(m, p, 1.0)) - np.log(q0r)), 0.0)))
            for t_ in range(len(n)):
                null["tokens"] += 1
                null["max_q_diff"] = max(null["max_q_diff"], float(np.max(np.abs(Q1[t_] - Q0[t_]))))
                l0 = float(np.sum(Y[t_] * -np.log(Q0[t_])))
                l1 = float(np.sum(Y[t_] * -np.log(Q1[t_])))
                b0 = float(n[t_] * (Q0[t_] @ Q0[t_]) - 2 * (Y[t_] @ Q0[t_]) + n[t_])
                b1 = float(n[t_] * (Q1[t_] @ Q1[t_]) - 2 * (Y[t_] @ Q1[t_]) + n[t_])
                null["min_law_loss_gap"] = min(null["min_law_loss_gap"], (l1 - l0) / n[t_], (b1 - b0) / n[t_])
            cell_lab = [None] * len(law.cells[i])
            for f_, t_ in mp.items():
                cell_lab[f_] = t_
            k_ = part_key(cell_lab)
            rows.append({"key": k_, "L_D1": L1, "B_D1": B1, "L_D0": L0, "B_D0": B0, "D": Dkl / law.N, "I": law.mi(tok),
                         "alpha": int(len(n)), "tokens_per_class": [int(np.sum(cls == c_)) for c_ in range(Ki)],
                         "d1_strict": bool(all(strict_argmax_ok(Q1[t_], cls[t_]) for t_ in range(len(n))))})
            tk.append(tok)
        per[i], toks[i] = rows, np.stack(tk)
        keys[i] = {r["key"]: j for j, r in enumerate(rows)}
    G2 = int(toks[2].max()) + 1
    I12 = np.zeros((len(per[1]), len(per[2])))
    w = law.c.astype(np.float64)
    for a_ in range(len(per[1])):
        base = toks[1][a_] * G2
        for b_ in range(len(per[2])):
            key = (base + toks[2][b_]) * 2 + law.s
            cnt = np.bincount(key, weights=w)
            tab = cnt.reshape(-1, 2).T if cnt.size % 2 == 0 else np.r_[cnt, 0.0].reshape(-1, 2).T
            tab = tab[:, tab.sum(0) > 0]
            I12[a_, b_] = mi_counts(tab)
    U = law.u_losses(1), law.u_losses(2)
    return {"per": per, "I12": I12, "keys": keys, "U": {1: U[0], 2: U[1]}, "null": null,
            "acc": {i: float(np.sum(law.c * (law.d[i] == law.y[i])) / law.N) for i in (1, 2)},
            "pairs": len(per[1]) * len(per[2])}


def fix_feasible(tab, i, j):
    r, U_ = tab["per"][i][j], tab["U"][i]
    return (r["L_D1"] <= U_["L"] + FIT_BUDGET["ll"] + GATE_TOL["TOL_BUDGET"] and
            r["B_D1"] <= U_["B"] + FIT_BUDGET["brier"] + GATE_TOL["TOL_BUDGET"] and r["d1_strict"])


def own_fixture_references(tab, caps_local=None):
    """Exhaustive optimum of every registered own problem (FIXTURE_GATE_RULE own_problem_objectives), first canonical
    pair on ties; K- references use the local caps I_i <= caps_local[i] + TOL_MI (the C-TASK release's I_i)."""
    p1, p2, I12 = tab["per"][1], tab["per"][2], tab["I12"]
    n1, n2 = len(p1), len(p2)
    A = lambda k_, i: np.asarray([r[k_] for r in tab["per"][i]])  # noqa: E731
    D = A("D", 1)[:, None] + A("D", 2)[None, :]
    T = (A("L_D1", 1) + 0.5 * A("B_D1", 1))[:, None] + (A("L_D1", 2) + 0.5 * A("B_D1", 2))[None, :]
    Is = 0.5 * (A("I", 1)[:, None] + A("I", 2)[None, :])
    Phi = I12 + Is
    out = {}

    def best(V, mask=None):
        V = np.where(mask, V, np.inf) if mask is not None else V
        if not np.isfinite(V).any():
            return {"value": None, "index": None}
        j = int(np.argmin(V))
        return {"value": float(V.flat[j]), "index": [j // n2, j % n2]}
    out["D0 FINE-TASK"] = best(D)
    out["C-TASK"] = best(T)
    for lam in LAMS:
        out[f"D0 LOCAL|l{lam:g}"] = best(D + lam * Is)
        out[f"D0 SEQ/JOINT|l{lam:g}"] = best(D + lam * (Is + I12))
        out[f"W-LOCAL|l{lam:g}"] = best(T + lam * Is)
        out[f"W-SEQ/JOINT|l{lam:g}"] = best(T + lam * Phi)
    F1 = np.asarray([fix_feasible(tab, 1, j) for j in range(n1)])
    F2 = np.asarray([fix_feasible(tab, 2, j) for j in range(n2)])
    out["feasible_maps"] = {"1": int(F1.sum()), "2": int(F2.sum())}
    if caps_local is not None:
        L1 = F1 & (A("I", 1) <= caps_local[1] + GATE_TOL["TOL_MI"])
        L2 = F2 & (A("I", 2) <= caps_local[2] + GATE_TOL["TOL_MI"])
        out["K-LOCAL"] = {str(i): ({"value": float(np.min(np.where(Lm, A("I", i), np.inf))),
                                    "index": int(np.argmin(np.where(Lm, A("I", i), np.inf)))} if Lm.any() else
                                   {"value": None, "index": None}) for i, Lm in ((1, L1), (2, L2))}
        out["K-SEQ/JOINT"] = best(Phi, L1[:, None] & L2[None, :])
        out["most_private_feasible_local_pair_I12"] = best(I12, L1[:, None] & L2[None, :])
    return out


def class_map_key(law: Law, i):
    """CLASS map: every fine cell of a predicted class in one token."""
    by = {}
    for f_, c_ in law.cells[i].items():
        by.setdefault(c_, []).append(f_)
    return tuple(sorted(tuple(sorted(v)) for v in by.values()))


def own_direct_key(law: Law, i):
    """Own DIRECT-TASK map on the fixture rows: own per-class KL k-means at the cap (registered starts, qpc rule) on
    the rows' teacher vectors; each fine cell has one vector, so the clusters are a partition of the fine cells."""
    rep = np.repeat(np.arange(len(law.c)), law.c)
    P, d = law.p[i][rep], law.d[i][rep]
    fp = fit_partition(P, d, law.K[i], law.caps[i], starts=START_ORDER, rule="qpc")
    cl = assign(P, d, fp)
    f_rows = law.f[i][rep]
    lab = {}
    for f_, c_ in zip(f_rows.tolist(), cl.tolist()):
        if lab.setdefault(f_, c_) != c_:
            raise ValueError("a fine cell split across k-means cells")
    return part_key([lab[f_] for f_ in range(len(law.cells[i]))])


def check_fixture_laws():
    """C7 law integrity and own exhaustive tables / references for the four registered laws (before FIXTURE_GATE)."""
    p = RES / "FIXTURE_LAWS.json"
    body = jload(p)
    out, fails, tabs = {"laws_sha256": body.get("laws_sha256")}, [], {}
    out["hash_ok"] = registered_laws_hash(body) == body.get("laws_sha256")
    rule = jload(RES / "FIXTURE_GATE_RULE.json")
    out["gate_rule_binds_same_laws"] = rule.get("laws_sha256") == body.get("laws_sha256")
    out["tolerances_equal_registered"] = {k: rule["tolerances"].get(k) == v for k, v in GATE_TOL.items()}
    out["task_only_candidates_equal"] = tuple(rule.get("task_only_D1_candidates") or ()) == FIX_TASK_ONLY
    fams = {}
    cache = D1Cache()
    for fam in body["families"]:
        fid = fam["id"]
        own_atoms = rebuild_atoms(fam)
        st = law_static_properties(fam, own_atoms)
        props = fam["properties"]
        cmp_keys = ("cell_counts", "cell_counts_match_declared", "label_counts", "constant_class", "constant_acc",
                    "teacher_decision_acc", "accuracy_gain", "cells_per_class", "canonical_partitions")
        prop_ok = (st["rows"] == props["rows"] == FIXTURE_N and st["mapping_pairs"] == props["mapping_pairs"] and
                   st["sex_counts"] == props["sex_counts"] and
                   all(st[f"r{i}"][k_] == props[f"r{i}"][k_] for i in (1, 2) for k_ in cmp_keys))
        law = law_of_family(fam)
        t0 = time.time()
        tab = own_fixture_tables(law, cache)
        ref = own_fixture_references(tab)
        dk = {i: own_direct_key(law, i) for i in (1, 2)}
        ck = {i: class_map_key(law, i) for i in (1, 2)}
        f = {"atoms_rebuilt_exactly": own_atoms == fam["atoms"],
             "atoms_rebuilt_as_set": sorted(map(tuple, own_atoms)) == sorted(map(tuple, fam["atoms"])),
             "static_properties_equal": prop_ok, "rows": st["rows"], "mapping_pairs": st["mapping_pairs"],
             "pairs_within_cap": st["mapping_pairs"] <= FIXTURE_MAX_PAIRS,
             "teacher_argmax_ok": all(st[f"r{i}"]["teacher_strict_argmax_is_declared_class"] for i in (1, 2)),
             "enumerated_pairs": tab["pairs"], "enumeration_equals_static_count": tab["pairs"] == st["mapping_pairs"],
             "d1_class_preserved_every_partition": all(r["d1_strict"] for i in (1, 2) for r in tab["per"][i]),
             "U": {str(i): tab["U"][i] for i in (1, 2)}, "teacher_acc": tab["acc"],
             "feasible_maps": ref["feasible_maps"],
             "exhaustive_references_without_local_caps": {k_: v for k_, v in ref.items() if k_ != "feasible_maps"},
             "own_class_map_I": {str(i): tab["per"][i][tab["keys"][i][ck[i]]]["I"] for i in (1, 2)},
             "own_direct_task_map_in_family": {str(i): dk[i] in tab["keys"][i] for i in (1, 2)},
             "wall_s": round(time.time() - t0, 2)}
        if fid == "F1_CALIBRATED_NULL":
            f["calibrated_null"] = {**tab["null"], "ok": tab["null"]["max_q_diff"] <= GATE_TOL["TOL_NULL_Q"] and
                                    tab["null"]["min_law_loss_gap"] >= -GATE_TOL["TOL_NULL_LOSS"]}
        ok = (f["atoms_rebuilt_as_set"] and prop_ok and f["pairs_within_cap"] and f["teacher_argmax_ok"] and
              f["enumeration_equals_static_count"] and f["d1_class_preserved_every_partition"] and
              f.get("calibrated_null", {"ok": True})["ok"])
        if not ok:
            fails.append(fid)
        fams[fid] = res("PASS" if ok else "FAIL", **f)
        tabs[fid] = {"law": law, "tab": tab, "direct_key": dk, "class_key": ck}
    out["families"] = fams
    ok = (out["hash_ok"] and out["gate_rule_binds_same_laws"] and all(out["tolerances_equal_registered"].values()) and
          out["task_only_candidates_equal"] and not fails and len(fams) == 4)
    out["solves"] = {"hits": cache.hits, "misses": cache.miss}
    return res("PASS" if ok else "FAIL", failures=fails, **out), tabs


def _read_csv(p):
    import csv
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def check_fixture_gate_replay(tabs):
    """After the fixture stage: B's oracle tables row by row vs the own tables (partitions matched by key), every arm's
    terms from the own tables via its partition, feasibility / local budget / T_star / qualifying / trigger / route /
    verdict recomputed from the registered rule, C1-C7 replayed where the own data allows."""
    gp = RES / "FIXTURE_GATE.json"
    if not gp.exists():
        return pending("fixture_gate", "awaiting FIXTURE_GATE.json")
    G = jload(gp)
    od = RES / "fixture_oracle"
    out, fails = {"lead_verdict": G.get("verdict"), "lead_route": G.get("route"), "lead_reasons": G.get("reasons")}, []
    trig_any, helps_any, per_fix = [], [], {}
    for fx in G.get("fixtures", []):
        fid = fx["fixture"]
        if fid not in tabs:
            fails.append(f"{fid}: unknown fixture")
            continue
        law, tab = tabs[fid]["law"], tabs[fid]["tab"]
        fo, ff = {}, []
        # oracle tables
        bparts = {}
        worst = 0.0
        for i in (1, 2):
            rows = _read_csv(od / f"{fid}_partitions_r{i}.csv")
            bparts[i] = [part_key([int(x) for x in r["labels"].split()]) for r in rows]
            if sorted(bparts[i]) != sorted(tab["keys"][i]):
                ff.append(f"r{i}: partition sets differ")
                continue
            for r, k_ in zip(rows, bparts[i]):
                mine = tab["per"][i][tab["keys"][i][k_]]
                for col, mk in (("L_D1", "L_D1"), ("B_D1", "B_D1"), ("L_D0", "L_D0"), ("B_D0", "B_D0"),
                                ("D_teacher_kl", "D"), ("I", "I")):
                    worst = max(worst, abs(float(r[col]) - mine[mk]))
        pi = _read_csv(od / f"{fid}_pair_I12.csv")
        for r in pi:
            a_, b_ = bparts[1][int(r["index1"])], bparts[2][int(r["index2"])]
            worst = max(worst, abs(float(r["I12"]) - float(tab["I12"][tab["keys"][1][a_], tab["keys"][2][b_]])))
        fo["oracle_tables_max_abs_diff"] = worst
        if worst > GATE_TOL["TOL_TERMS"]:
            ff.append("oracle table values")
        # arms from the own tables
        arms = fx.get("arms") or {}
        ct = arms.get(L_CTASK)
        own = {}
        for cid, a in arms.items():
            k1, k2 = (a.get("oracle_index") or [None, None])
            if k1 is None:
                ff.append(f"{cid}: no oracle index")
                continue
            j1, j2 = tab["keys"][1][bparts[1][k1]], tab["keys"][2][bparts[2][k2]]
            dec = "D1" if cid.endswith("|D1") else "D0"
            r1, r2 = tab["per"][1][j1], tab["per"][2][j2]
            m = {"L1": r1[f"L_{dec}"], "B1": r1[f"B_{dec}"], "L2": r2[f"L_{dec}"], "B2": r2[f"B_{dec}"], "I1": r1["I"],
                 "I2": r2["I"], "I12": float(tab["I12"][j1, j2])}
            m["T"] = m["L1"] + m["L2"] + 0.5 * (m["B1"] + m["B2"])
            m["Phi"] = phi_of(m["I1"], m["I2"], m["I12"])
            caps_ok = all(max(r_["tokens_per_class"]) <= law.caps[i] for i, r_ in ((1, r1), (2, r2)))
            feas = (all(m[f"L{i}"] <= tab["U"][i]["L"] + FIT_BUDGET["ll"] + GATE_TOL["TOL_BUDGET"] and
                        m[f"B{i}"] <= tab["U"][i]["B"] + FIT_BUDGET["brier"] + GATE_TOL["TOL_BUDGET"] for i in (1, 2))
                    and caps_ok and (dec == "D0" or (r1["d1_strict"] and r2["d1_strict"])))
            own[cid] = {"m": m, "feasible": bool(feas), "j": (j1, j2)}
            dmax = max(abs(m[k_] - float(a[k_])) for k_ in ("L1", "L2", "B1", "B2", "I1", "I2", "I12", "T", "Phi"))
            if dmax > GATE_TOL["TOL_TERMS"]:
                ff.append(f"{cid}: terms differ by {dmax:.3g}")
            if bool(a["feasible"]) != own[cid]["feasible"]:
                ff.append(f"{cid}: feasibility differs (lead {a['feasible']}, own {own[cid]['feasible']})")
        fo["arms_checked"] = len(own)
        ident = {}
        for c_, keyf in (("U|CLASS|i1o1", "class_key"), ("U|DIRECT-TASK|i8o64", "direct_key")):
            a = arms.get(c_)
            if a and a.get("oracle_index"):
                ident[c_] = all(bparts[i][a["oracle_index"][i - 1]] == tabs[fid][keyf][i] for i in (1, 2))
                if not ident[c_]:
                    ff.append(f"{c_}: partition differs from the own construction")
        fo["own_partition_identity"] = ident
        if ct is None or L_CTASK not in own:
            ff.append("C-TASK arm absent")
            per_fix[fid] = res("FAIL", failures=ff, **fo)
            fails.append(fid)
            continue
        cap = {1: own[L_CTASK]["m"]["I1"], 2: own[L_CTASK]["m"]["I2"]}
        lok = {cid: all(o["m"][f"I{i}"] <= cap[i] + GATE_TOL["TOL_MI"] for i in (1, 2)) for cid, o in own.items()}
        for cid, a in arms.items():
            if cid in lok and a.get("local_ok") is not None and bool(a["local_ok"]) != lok[cid]:
                ff.append(f"{cid}: local_ok differs")
        ref = own_fixture_references(tab, cap)
        fo["own_references_with_local_caps"] = {k_: ref[k_] for k_ in ("K-LOCAL", "K-SEQ/JOINT",
                                                                       "most_private_feasible_local_pair_I12")}
        # trigger (registered rule)
        const = {i: tab["U"][i]["const_acc"] for i in (1, 2)}
        gains = [tab["acc"][i] - const[i] for i in (1, 2)]
        task = sorted((own[c_]["m"]["I12"], own[c_]["m"]["T"], c_) for c_ in FIX_TASK_ONLY
                      if c_ in own and own[c_]["feasible"])
        tr = {"accuracy_gain": gains}
        if not task:
            tr.update({"T_star": None, "triggered": False, "reason": "NO_FEASIBLE_TASK_ONLY", "nontrivial": False})
            lt = fx.get("trigger") or {}
            for k_ in ("T_star", "triggered", "reason", "nontrivial"):
                if lt.get(k_) != tr.get(k_):
                    ff.append(f"trigger.{k_} differs (lead {lt.get(k_)}, own {tr.get(k_)})")
            if lt.get("qualifying"):
                ff.append("qualifying set nonempty without a task-only reference")
        else:
            i12t, _, tstar = task[0]
            qual = []
            for c_, o in sorted(own.items()):
                arm = lcr_arm(c_)
                if arm not in ("d1_fixed", "weighted", "constrained") or not c_.endswith("|D1"):
                    continue
                if lcr_family(c_) in ("DIRECT-TASK", "FINE-TASK", "CLASS"):
                    continue
                if o["feasible"] and lok[c_] and i12t - o["m"]["I12"] >= GATE_TOL["TRIGGER_MI"] and \
                        min(gains) >= GATE_TOL["MIN_GAIN"]:
                    qual.append({"cid": c_, "arm": arm, "I12": o["m"]["I12"], "reduction": i12t - o["m"]["I12"]})
            nontriv = min(gains) >= GATE_TOL["MIN_GAIN"] and i12t >= GATE_TOL["TRIGGER_MI"]
            fm = [q for q in qual if q["arm"] == "d1_fixed"]
            new = [q for q in qual if q["arm"] in ("weighted", "constrained")]
            bfm = min((q["I12"] for q in fm), default=None)
            bnew = min((q["I12"] for q in new), default=None)
            helps = bool(new) and (not fm or bnew <= bfm - GATE_TOL["TRIGGER_MI"])
            d0_ok = [q["cid"][:-3] for q in fm if q["cid"][:-3] in own and own[q["cid"][:-3]]["feasible"] and
                     lok[q["cid"][:-3]] and i12t - own[q["cid"][:-3]]["m"]["I12"] >= GATE_TOL["TRIGGER_MI"]]
            tr.update({"T_star": tstar, "T_star_I12": i12t, "nontrivial": bool(nontriv),
                       "triggered": bool(nontriv and qual), "qualifying": [q["cid"] for q in qual],
                       "decoder_enabled": bool(fm), "assignment_search_helps": bool(helps), "best_fm_I12": bfm,
                       "best_new_I12": bnew, "d0_version_already_qualifies": d0_ok,
                       "exhaustive_most_private_feasible_local_I12": ref["most_private_feasible_local_pair_I12"]["value"]})
            lt = fx.get("trigger") or {}
            for k_ in ("T_star", "triggered", "nontrivial"):
                if lt.get(k_) != tr.get(k_):
                    ff.append(f"trigger.{k_} differs (lead {lt.get(k_)}, own {tr.get(k_)})")
            if sorted(q["cid"] for q in lt.get("qualifying") or []) != sorted(tr["qualifying"]):
                ff.append("qualifying set differs")
            lr = lt.get("route") or {}
            for k_ in ("decoder_enabled", "assignment_search_helps"):
                if lr.get(k_) is not None and bool(lr.get(k_)) != bool(tr[k_]):
                    ff.append(f"route.{k_} differs")
        fo["own_trigger"] = tr
        if tr.get("triggered"):
            trig_any.append(fid)
            helps_any.append(bool(tr.get("assignment_search_helps")))
        # C2 own (F1 only) and C1 (decoder change keeps every token: identical partition => identical MI by construction)
        if fid == "F1_CALIBRATED_NULL":
            fo["own_C2"] = tab["null"]
        lead_checks = {k_: bool(v.get("pass")) for k_, v in (fx.get("checks") or {}).items()}
        fo["lead_checks"] = lead_checks
        # C6 own labels: each searched arm's own-problem objective vs the own exhaustive optimum
        c6 = {}
        for c_, o in own.items():
            key, val = None, None
            arm, lam = lcr_arm(c_), cid_lam(c_)
            m = o["m"]
            Is = 0.5 * (m["I1"] + m["I2"])
            Dv = tab["per"][1][o["j"][0]]["D"] + tab["per"][2][o["j"][1]]["D"]
            fam_ = lcr_family(c_)
            if arm == "d0" and fam_ == "FINE-TASK":
                key, val = "D0 FINE-TASK", Dv
            elif arm == "d0" and fam_ == "LOCAL":
                key, val = f"D0 LOCAL|l{lam:g}", Dv + lam * Is
            elif arm == "d0" and fam_ in ("SEQ-12", "SEQ-21", "JOINT"):
                key, val = f"D0 SEQ/JOINT|l{lam:g}", Dv + lam * (Is + m["I12"])
            elif arm == "ctask":
                key, val = "C-TASK", m["T"]
            elif arm == "weighted":
                key = f"W-LOCAL|l{lam:g}" if fam_ == "LOCAL" else f"W-SEQ/JOINT|l{lam:g}"
                val = m["T"] + lam * (Is if fam_ == "LOCAL" else m["Phi"])
            elif arm == "constrained" and fam_ != "LOCAL":
                key, val = "K-SEQ/JOINT", m["Phi"]
            elif arm == "constrained" and fam_ == "LOCAL":
                gaps = {str(i): (None if ref["K-LOCAL"][str(i)]["value"] is None else
                                 m[f"I{i}"] - ref["K-LOCAL"][str(i)]["value"]) for i in (1, 2)}
                feas = o["feasible"] and lok[c_]
                lab = "EXHAUSTIVE_OPTIMAL" if feas and all(v is not None and v <= GATE_TOL["TOL_OPT"] for v in
                                                           gaps.values()) else "HEURISTIC"
                c6[c_] = {"label": lab, "gap": max([v for v in gaps.values() if v is not None] or [0.0])}
                ll = (((fx.get("checks") or {}).get("C6_HEURISTIC_LABELLING") or {}).get("labels") or {}).get(c_, {})
                if ll.get("label") and ll["label"] != lab:
                    ff.append(f"{c_}: C6 label differs (lead {ll['label']}, own {lab})")
                continue
            if key is None:
                continue
            rv = ref[key]["value"]
            feas = (o["feasible"] and lok[c_]) if arm == "constrained" else True
            lab = "EXHAUSTIVE_OPTIMAL" if (feas and rv is not None and val - rv <= GATE_TOL["TOL_OPT"]) else "HEURISTIC"
            c6[c_] = {"label": lab, "gap": None if rv is None else val - rv}
            ll = (((fx.get("checks") or {}).get("C6_HEURISTIC_LABELLING") or {}).get("labels") or {}).get(c_, {})
            if ll.get("label") and ll["label"] != lab:
                ff.append(f"{c_}: C6 label differs (lead {ll['label']}, own {lab})")
        fo["own_C6"] = {"labels": {l_: sum(1 for v in c6.values() if v["label"] == l_) for l_ in
                                   ("EXHAUSTIVE_OPTIMAL", "HEURISTIC")}, "max_heuristic_gap": max(
            [v["gap"] for v in c6.values() if v["gap"] is not None] or [0.0])}
        per_fix[fid] = res("FAIL" if ff else "PASS", failures=ff[:30], n_failures=len(ff), **fo)
        if ff:
            fails.append(fid)
    lead_bad = sorted({f"{fx['fixture']}:{k_}" for fx in G.get("fixtures", []) for k_, v in (fx.get("checks") or {}).items()
                       if not v.get("pass")})
    met = not lead_bad and bool(trig_any) and len(per_fix) == 4 and not fails
    own_verdict = "GATE_MET" if (not lead_bad and trig_any and len(per_fix) == 4) else "GATE_NOT_MET"
    own_route = ("ASSIGNMENT_SEARCH_ROUTE" if any(helps_any) else "DECODER_ENABLED_ROUTE") if own_verdict == "GATE_MET" \
        else None
    out.update({"own_verdict": own_verdict, "own_route": own_route, "own_triggered": trig_any,
                "lead_correctness_failures": lead_bad, "fixtures": per_fix,
                "verdict_agrees": own_verdict == G.get("verdict"), "route_agrees": own_route == G.get("route")})
    ok = not fails and out["verdict_agrees"] and out["route_agrees"] and len(per_fix) == 4
    _ = met
    return res("PASS" if ok else "FAIL", failures=fails, **out)


# ================================================================================================ lcr PHASE 1B: fixture gate
FIXTURE_RULE_SHA = "6f3bb44a3f4fd9db6aa986e697929689e1e49c43d9e5f88ae2136a482b0ed844"   # re-bound (FIXTURE_LOCK 9ac4cb7)
FIXTURE_LAWS_SHA_FILE = "24f7074519dc9d24afb75a14e3601be61ee0e17929fe150a2fb8ca0db5ce9d9c"
FIX_ATT1 = RUN / "attempts" / "fixture_attempt1"
MAPPER_BUDGET_MARGIN = 1e-10        # registered search feasibility margin (SEARCH_RULES / mapper text), L and B


def _lock_doc_ok(lock_name, rel_doc):
    p = RES / f"{lock_name}.json"
    if not p.exists():
        return None
    L_ = jload(p)
    h = (L_.get("documents_sha256") or {}).get(rel_doc)
    return h is not None and h == sha_file(RES / rel_doc)


def _push_time(commit):
    t = first_remote(commit, remote_reflog())
    return iso(t) if isinstance(t, datetime) else t


def check_fixture_binding():
    """C7 re-binding (rule text 6f3bb44a): laws and rule equal FIXTURE_LOCK documents_sha256; the lock and A1 are on
    origin before the attempts ran; FIXTURE_GATE.json binds the same rule / laws; oracle table hashes re-hash; every
    fix__ unit (attempts 1 and 2) carries the laws hash and a consistent COMPLETE.json."""
    out, f = {}, []
    out["rule_sha256"] = sha_file(RES / "FIXTURE_GATE_RULE.json")
    out["rule_equals_rebound"] = out["rule_sha256"] == FIXTURE_RULE_SHA
    out["laws_file_sha256"] = sha_file(RES / "FIXTURE_LAWS.json")
    out["laws_equal_lock_documents"] = _lock_doc_ok("FIXTURE_LOCK", "FIXTURE_LAWS.json")
    out["rule_equals_lock_documents"] = _lock_doc_ok("FIXTURE_LOCK", "FIXTURE_GATE_RULE.json")
    laws = jload(RES / "FIXTURE_LAWS.json")
    rule = jload(RES / "FIXTURE_GATE_RULE.json")
    out["laws_hash_rule_ok"] = registered_laws_hash(laws) == laws["laws_sha256"] == rule.get("laws_sha256")
    out["laws_unchanged_since_dd1cf23"] = git_show_bytes("dd1cf2373", f"{REL_RES}/FIXTURE_LAWS.json") == \
        (RES / "FIXTURE_LAWS.json").read_bytes()
    lock_c = git("log", "--format=%H", "-1", "--", f"{REL_RES}/FIXTURE_LOCK.json")
    am_c = git("log", "--format=%H", "-1", "--", f"{REL_RES}/AMENDMENT_A1_FIXTURE_C5_SCOPE.json")
    out["lock_commit"], out["a1_commit"] = lock_c, am_c
    out["lock_on_origin_byte_identical"] = git_show_bytes(f"origin/{BRANCH}", f"{REL_RES}/FIXTURE_LOCK.json") == \
        (RES / "FIXTURE_LOCK.json").read_bytes()
    out["a1_on_origin_byte_identical"] = git_show_bytes(f"origin/{BRANCH}",
                                                        f"{REL_RES}/AMENDMENT_A1_FIXTURE_C5_SCOPE.json") == \
        (RES / "AMENDMENT_A1_FIXTURE_C5_SCOPE.json").read_bytes()
    out["lock_first_push"] = _push_time(lock_c) if lock_c else None
    out["a1_first_push"] = _push_time(am_c) if am_c else None
    att = jload(RES / "FIXTURE_ATTEMPTS.json")
    runs = {a["attempt"]: a["ran"].split("/") for a in att["attempts"]}
    out["attempt_runs"] = runs
    out["lock_pushed_before_attempt1"] = bool(out["lock_first_push"]) and parse_iso(out["lock_first_push"]) <= \
        parse_iso(runs[1][0])
    out["a1_pushed_before_attempt2"] = bool(out["a1_first_push"]) and parse_iso(out["a1_first_push"]) <= \
        parse_iso(runs[2][0])
    G = jload(RES / "FIXTURE_GATE.json")
    out["gate_binds_rule_and_laws"] = G.get("gate_rule_sha256") == out["rule_sha256"] and G.get(
        "laws_sha256") == laws["laws_sha256"]
    out["gate_file_sha_equals_ledger"] = sha_file(RES / "FIXTURE_GATE.json") == att["attempts"][1]["file_sha256"]
    out["attempt1_file_sha_equals_ledger"] = sha_file(RES / "fixture_attempts" / "attempt1" / "FIXTURE_GATE.json") == \
        att["attempts"][0]["file_sha256"]
    oh = G.get("oracle_table_sha256") or {}
    out["oracle_tables_rehash"] = bool(oh) and all(sha_file(RES / "fixture_oracle" / n_) == h for n_, h in oh.items())
    units = {}
    for base, tag in ((UNITS, "attempt2"), (FIX_ATT1, "attempt1")):
        for fam in laws["families"]:
            d = base / f"fix__{fam['id']}"
            cu, _ = complete_ok(d) if d.exists() else ({"id_ok": False, "rehash_ok": False, "unlisted": []}, {})
            rec = jload(d / "record.json") if d.exists() else {}
            units[f"{tag}:{fam['id']}"] = bool(cu["id_ok"] and cu["rehash_ok"] and not cu["unlisted"] and
                                               rec.get("laws_sha256") == laws["laws_sha256"])
    out["fix_units_complete_and_bound"] = units
    for k_, v in out.items():
        if isinstance(v, bool) and not v:
            f.append(k_)
    if not all(units.values()):
        f.append("fix units")
    return res("PASS" if not f else "FAIL", failures=f, **out)


def _arms_of(fx):
    return fx.get("arms") or {}


def _diff_paths(a, b, path="", skip=("wall_s", "cpu_s"), out=None, limit=200):
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in skip:
                continue
            if k not in a or k not in b:
                out.append(f"{path}.{k}:missing")
                continue
            _diff_paths(a[k], b[k], f"{path}.{k}", skip, out, limit)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}:len")
        for i_, (x, y) in enumerate(zip(a, b)):
            _diff_paths(x, y, f"{path}[{i_}]", skip, out, limit)
    elif a != b:
        out.append(path)
    return out


def check_fixture_extras(tabs):
    """C1-C4 / descriptive flags / structural bound / A1 cause + correction / attempt identity, by own code from the
    re-bound rule (6f3bb44a) on the registered result (attempt 2)."""
    G = jload(RES / "FIXTURE_GATE.json")
    G1 = jload(RES / "fixture_attempts" / "attempt1" / "FIXTURE_GATE.json")
    node = {}
    c1f, c2f, c3f, c4f, desc_f, struct_f = [], [], [], [], [], []
    desc_own, struct = {}, {}
    for fx in G["fixtures"]:
        fid = fx["fixture"]
        law, tab = tabs[fid]["law"], tabs[fid]["tab"]
        arms = _arms_of(fx)
        od = RES / "fixture_oracle"
        bp = {}
        for i in (1, 2):
            rows = _read_csv(od / f"{fid}_partitions_r{i}.csv")
            if [int(r["index"]) for r in rows] != list(range(len(rows))):
                c1f.append(f"{fid}: r{i} partition index column is not 0..n-1")
            bp[i] = [part_key([int(x) for x in r["labels"].split()]) for r in rows]

        def own_j(a):
            k1, k2 = a["oracle_index"]
            return tab["keys"][1][bp[1][k1]], tab["keys"][2][bp[2][k2]]
        # C1: every D1 version of a D0 map has the same partition (tokens), bitwise-identical I1, I2, I12 and fingerprint
        for cid, a in arms.items():
            if cid.endswith("|D1") and lcr_arm(cid) == "d1_fixed":
                b = arms.get(cid[:-3])
                if b is None:
                    c1f.append(f"{fid}:{cid}: D0 map absent")
                    continue
                if a["oracle_index"] != b["oracle_index"] or any(a[k_] != b[k_] for k_ in ("I1", "I2", "I12")) or \
                        a["pair_fingerprint"] != b["pair_fingerprint"]:
                    c1f.append(f"{fid}:{cid}")
        # C2 (F1): every D1 arm vs the D0 decoding of the same partition (own tables), population loss per row
        if fid == "F1_CALIBRATED_NULL":
            gmin = 0.0
            for cid, a in arms.items():
                if not cid.endswith("|D1"):
                    continue
                j1, j2 = own_j(a)
                for i, j in ((1, j1), (2, j2)):
                    r = tab["per"][i][j]
                    g_ = min(r["L_D1"] - r["L_D0"], r["B_D1"] - r["B_D0"])
                    gmin = min(gmin, g_)
                    if g_ < -GATE_TOL["TOL_NULL_LOSS"]:
                        c2f.append(f"{cid}:r{i}")
            node["C2_per_arm_min_D1_minus_D0"] = gmin
        # C3: mapper statuses (only K- may be INFEASIBLE; K- FEASIBLE must be own-feasible and local-ok); U values
        ms = fx.get("mapper_status") or {}
        for i in (1, 2):
            if abs(fx["U"][str(i)]["L"] - tab["U"][i]["L"]) > 1e-12 or abs(fx["U"][str(i)]["B"] - tab["U"][i]["B"]) > 1e-12:
                c3f.append(f"{fid}: U{i} differs from own")
        ct = arms.get(L_CTASK)
        capI = None
        if ct:
            j1, j2 = own_j(ct)
            capI = {1: tab["per"][1][j1]["I"], 2: tab["per"][2][j2]["I"]}
        for cid, st in ms.items():
            arm = lcr_arm(cid)
            if st["status"] == "INFEASIBLE" and arm != "constrained":
                c3f.append(f"{fid}:{cid}: INFEASIBLE on an unconstrained arm")
            if st["status"] == "FEASIBLE" and arm == "constrained":
                j1, j2 = own_j(arms[cid])
                ok_ = all(fix_feasible(tab, i, j) and tab["per"][i][j]["I"] <= capI[i] + GATE_TOL["TOL_MI"]
                          for i, j in ((1, j1), (2, j2)))
                if not ok_:
                    c3f.append(f"{fid}:{cid}: reported FEASIBLE, own infeasible")
        # C4: decisions / caps parts of every arm (B's) and own D1 strict argmax on every partition
        for cid, a in arms.items():
            fp_ = a.get("feasible_parts") or {}
            if not all(fp_.get(f"{x}{i}") for x in ("dec", "cap") for i in (1, 2)):
                c4f.append(f"{fid}:{cid}")
        if not all(r["d1_strict"] for i in (1, 2) for r in tab["per"][i]):
            c4f.append(f"{fid}: own D1 strict argmax")
        # descriptive flags (own recomputation of the registered definition)
        groups = {"constrained": lambda c_: lcr_arm(c_) == "constrained",
                  "weighted": lambda c_: lcr_arm(c_) == "weighted",
                  "d1_fixed_joint": lambda c_: lcr_arm(c_) == "d1_fixed" and lcr_family(c_) == "JOINT",
                  "d1_fixed_seq": lambda c_: lcr_arm(c_) == "d1_fixed" and lcr_family(c_) in ("SEQ-12", "SEQ-21"),
                  "weighted_joint": lambda c_: lcr_arm(c_) == "weighted" and lcr_family(c_) == "JOINT",
                  "weighted_seq": lambda c_: lcr_arm(c_) == "weighted" and lcr_family(c_) in ("SEQ-12", "SEQ-21"),
                  "constrained_joint": lambda c_: lcr_arm(c_) == "constrained" and lcr_family(c_) in ("JOINT-SINGLE",
                                                                                                    "JOINT-PAIR"),
                  "constrained_seq": lambda c_: lcr_arm(c_) == "constrained" and lcr_family(c_) in ("SEQ-12", "SEQ-21")}
        bi = {}
        for gname, fn in groups.items():
            mem = [c_ for c_ in arms if c_.endswith("|D1") and fn(c_)]
            vals, el = [], []
            for c_ in mem:
                j1, j2 = own_j(arms[c_])
                v = float(tab["I12"][j1, j2])
                vals.append(v)
                feas = fix_feasible(tab, 1, j1) and fix_feasible(tab, 2, j2)
                loc = capI is None or all(tab["per"][i][j]["I"] <= capI[i] + GATE_TOL["TOL_MI"] for i, j in
                                          ((1, j1), (2, j2)))
                if feas and loc:
                    el.append(v)
            bi[gname] = {"all": min(vals) if vals else None, "eligible": min(el) if el else None, "n": len(vals),
                         "n_eligible": len(el)}

        def dif(a_, b_):
            return {x: (None if bi[a_][x] is None or bi[b_][x] is None else bi[a_][x] - bi[b_][x])
                    for x in ("all", "eligible")}
        own_d = {"best_I12": bi, "constrained_minus_weighted": dif("constrained", "weighted"),
                 "joint_minus_sequential": {"d1_fixed": dif("d1_fixed_joint", "d1_fixed_seq"),
                                            "weighted": dif("weighted_joint", "weighted_seq"),
                                            "constrained": dif("constrained_joint", "constrained_seq")}}
        desc_own[fid] = own_d
        ld = (fx.get("trigger") or {}).get("descriptive") or {}

        def close(x, y):
            return (x is None and y is None) or (x is not None and y is not None and abs(x - y) <= GATE_TOL["TOL_TERMS"])
        for gname, v in bi.items():
            lv = (ld.get("best_I12") or {}).get(gname) or {}
            if not (close(v["all"], lv.get("all")) and close(v["eligible"], lv.get("eligible")) and v["n"] == lv.get("n")
                    and v["n_eligible"] == lv.get("n_eligible")):
                desc_f.append(f"{fid}:{gname}")
        # structural bound: I(S; d1, d2) exactly 0 (integer table), CLASS pair I12 == 0, every pair I12 >= it
        cnt = {}
        for d1_, d2_, s_, c_ in zip(law.d[1].tolist(), law.d[2].tolist(), law.s.tolist(), law.c.tolist()):
            cnt[(d1_, d2_, s_)] = cnt.get((d1_, d2_, s_), 0) + c_
        ns = {s_: sum(v for (a_, b_, t_), v in cnt.items() if t_ == s_) for s_ in (0, 1)}
        cells = {(a_, b_) for (a_, b_, _) in cnt}
        exact_zero = all(Fraction(cnt.get((a_, b_, s_), 0) * law.N) == Fraction(ns[s_]) *
                         (cnt.get((a_, b_, 0), 0) + cnt.get((a_, b_, 1), 0)) for (a_, b_) in cells for s_ in (0, 1))
        jc1, jc2 = tab["keys"][1][tabs[fid]["class_key"][1]], tab["keys"][2][tabs[fid]["class_key"][2]]
        i12_class = float(tab["I12"][jc1, jc2])
        min_all = float(tab["I12"].min())
        arm_min = min(float(a["I12"]) for a in arms.values())
        struct[fid] = {"I_S_given_decision_pair_exactly_zero": bool(exact_zero), "I12_CLASS": i12_class,
                       "min_I12_over_all_enumerated_pairs": min_all, "min_reported_arm_I12": arm_min,
                       "pairs_with_I12_below_CLASS": int(np.sum(tab["I12"] < i12_class - 1e-15)),
                       "CLASS_D1_budget_feasible": bool(fix_feasible(tab, 1, jc1) and fix_feasible(tab, 2, jc2))}
        if not (exact_zero and abs(i12_class) <= 1e-15 and min_all >= i12_class - 1e-15 and arm_min >= -1e-15):
            struct_f.append(fid)
    node["C1_d1_vs_d0_identity"] = {"failures": c1f}
    node["C2_per_arm"] = {"failures": c2f}
    node["C3_mapper_status_and_U"] = {"failures": c3f}
    node["C4_decisions_caps"] = {"failures": c4f}
    node["descriptive_flags"] = {"failures": desc_f, "own": desc_own}
    node["structural_bound"] = {"failures": struct_f, "per_fixture": struct,
                                "argument": "every class-preserving token refines the teacher-predicted class, so the "
                                            "token tuple determines (d1, d2) and I(S; t1, t2) >= I(S; d1, d2) = "
                                            "I12(CLASS) by data processing; checked exhaustively on every enumerated "
                                            "pair and exactly (integer table) for I(S; d1, d2)"}
    # A1: cause and correction
    a1 = {}
    fx1 = {f_["fixture"]: f_ for f_ in G["fixtures"]}
    fx0 = {f_["fixture"]: f_ for f_ in G1["fixtures"]}
    nsl = {fid: (f_["checks"]["C5_TERM_RECONSTRUCTION"].get("incremental_not_applicable_no_search_state") or [])
           for fid, f_ in fx1.items()}
    a1["no_search_state_lists_attempt2"] = nsl
    a1["attempt1_C5_failures"] = {fid: f_["checks"]["C5_TERM_RECONSTRUCTION"].get("failures") for fid, f_ in fx0.items()}
    ms1 = fx1["F1_CALIBRATED_NULL"]["mapper_status"]
    a1["F1_constrained_mapper_status"] = {c_: {"status": v["status"], "winner_kind": v["winner"]["kind"],
                                               "winner_start": v["winner"]["start"]} for c_, v in ms1.items()
                                          if lcr_arm(c_) == "constrained"}
    # own derivation of the cause from the registered sequential driver and the own oracle
    tab, law = tabs["F1_CALIBRATED_NULL"]["tab"], tabs["F1_CALIBRATED_NULL"]["law"]
    arms1 = _arms_of(fx1["F1_CALIBRATED_NULL"])
    bp1 = {i: [part_key([int(x) for x in r["labels"].split()]) for r in
               _read_csv(RES / "fixture_oracle" / f"F1_CALIBRATED_NULL_partitions_r{i}.csv")] for i in (1, 2)}

    def own_j1(a):
        return tab["keys"][1][bp1[1][a["oracle_index"][0]]], tab["keys"][2][bp1[2][a["oracle_index"][1]]]
    ctj = own_j1(arms1[L_CTASK])
    cap1 = {1: tab["per"][1][ctj[0]]["I"], 2: tab["per"][2][ctj[1]]["I"]}

    def search_feasible(i, j):
        r, U_ = tab["per"][i][j], tab["U"][i]
        return (r["L_D1"] <= U_["L"] + FIT_BUDGET["ll"] - MAPPER_BUDGET_MARGIN and
                r["B_D1"] <= U_["B"] + FIT_BUDGET["brier"] - MAPPER_BUDGET_MARGIN and r["I"] <= cap1[i])
    a1["F1_recipient2_maps_search_feasible"] = int(sum(search_feasible(2, j) for j in range(len(tab["per"][2]))))
    a1["F1_recipient1_maps_search_feasible"] = int(sum(search_feasible(1, j) for j in range(len(tab["per"][1]))))
    starts = {}
    for arm in ("SEQ-12", "SEQ-21"):
        first = 1 if arm == "SEQ-12" else 2
        rows = []
        for sc in [L_CTASK] + [L_d0(arm, l_) for l_ in LAMS]:
            a = arms1.get(sc)
            if a is None:
                rows.append({"start": sc, "present": False})
                continue
            j1, j2 = own_j1(a)
            jf = j1 if first == 1 else j2
            st1 = search_feasible(first, jf)
            rows.append({"start": sc, "present": True, "stage1_recipient": first, "stage1_feasible": bool(st1),
                         "stage2_recipient": 3 - first,
                         "stage2_can_be_feasible": bool(a1["F1_recipient2_maps_search_feasible"] > 0
                                                        if first == 1 else st1),
                         "reaches_final": False})
        starts[f"K-{arm}"] = rows
    a1["own_stage_analysis"] = starts
    a1["no_start_can_reach_a_refined_final"] = all(
        not r.get("stage1_feasible") or not r.get("stage2_can_be_feasible") for v in starts.values() for r in v)
    a1["mismatch_with_amendment_wording"] = [
        f"{arm}: start {r['start']} is stage-1 feasible (recipient 1) and fails at stage 2 (recipient 2), not at stage 1"
        for arm, v in starts.items() for r in v if r.get("stage1_feasible")]
    a1["applies_exactly_to_K_SEQ"] = nsl == {"F1_CALIBRATED_NULL": [L_k("SEQ-12"), L_k("SEQ-21")],
                                             "F2_MISCALIBRATED": [], "F3_COMPLEMENTARY_XOR": [], "F4_REDUNDANT": []}
    a1["other_F1_constrained_have_state"] = {
        L_k("LOCAL"): ms1[L_k("LOCAL")]["winner"]["kind"] == "refined_descriptive",
        L_k("JOINT-SINGLE"): "joint driver records an 'unchanged' block per witness (terms exist)",
        L_k("JOINT-PAIR"): "joint driver records an 'unchanged' block per witness (terms exist)"}
    # attempt identity outside C5 and timings
    d_gate = _diff_paths(G1, G, skip=("wall_s", "cpu_s"))
    d_gate_nc5 = [p_ for p_ in d_gate if "C5_TERM_RECONSTRUCTION" not in p_ and p_ not in (".verdict", ".reasons",
                                                                                           ".reasons[0]", ".reasons:len")]
    a1["gate_json_differing_paths"] = d_gate
    a1["gate_json_differences_outside_C5_and_verdict_reasons"] = d_gate_nc5
    csv_same = {}
    for p_ in sorted((RES / "fixture_oracle").glob("*.csv")):
        q_ = RES / "fixture_attempts" / "attempt1" / "fixture_oracle" / p_.name
        csv_same[p_.name] = q_.exists() and q_.read_bytes() == p_.read_bytes()
    a1["oracle_csv_byte_identical"] = all(csv_same.values()) and len(csv_same) == 16
    ud = {}
    for fid in fx1:
        r2 = jload(UNITS / f"fix__{fid}" / "result.json")
        r1 = jload(FIX_ATT1 / f"fix__{fid}" / "result.json")
        ud[fid] = [p_ for p_ in _diff_paths(r1, r2) if "C5_TERM_RECONSTRUCTION" not in p_]
    a1["fix_unit_results_differences_outside_C5_and_timing"] = ud
    a1["attempt1_reasons"] = G1.get("reasons")
    a1["attempt2_reasons"] = G.get("reasons")
    a1["code_change_files"] = sorted((git("diff", "--name-only", "cc0083aaf", "c9a7150c1") or "").split())
    a1["rule_and_laws_unchanged_by_A1"] = all(
        git_show_bytes("cc0083aaf", f"{REL_RES}/{d_}") == git_show_bytes("c9a7150c1", f"{REL_RES}/{d_}")
        for d_ in ("FIXTURE_GATE_RULE.json", "FIXTURE_LAWS.json"))
    a1_ok = (a1["applies_exactly_to_K_SEQ"] and a1["no_start_can_reach_a_refined_final"] and
             a1["F1_recipient2_maps_search_feasible"] == 0 and not d_gate_nc5 and a1["oracle_csv_byte_identical"] and
             not any(ud.values()) and a1["rule_and_laws_unchanged_by_A1"] and
             all(v["status"] == "INFEASIBLE" for v in a1["F1_constrained_mapper_status"].values()) and
             all(a1["F1_constrained_mapper_status"][L_k(x)]["winner_kind"] == "unchanged_descriptive"
                 for x in ("SEQ-12", "SEQ-21")) and
             a1["attempt1_C5_failures"]["F1_CALIBRATED_NULL"] == [f"{L_k('SEQ-12')}:incremental_terms_missing",
                                                                  f"{L_k('SEQ-21')}:incremental_terms_missing"])
    node["amendment_A1"] = res("PASS" if a1_ok else "FAIL", **a1)
    fails = c1f + c2f + c3f + c4f + desc_f + struct_f
    st = "PASS" if not fails and a1_ok else "FAIL"
    if a1_ok and a1["mismatch_with_amendment_wording"] and not fails:
        st = "PASS"
    return res(st, failures=fails[:30], **node)


def check_code_hashes_lcr():
    """Locked code: every FIXTURE_LOCK / A1 code file equals its blob at the lock (amendment) commit; the code at the
    attempt-1 / attempt-2 evidence commits equals the lock (attempt 1) and lock + A1 (attempt 2) for the files the
    fixture stage loads; the worktree still equals the latest locked value (else WARN: changed after the stage)."""
    out, f, w = {}, [], []
    L_ = jload(RES / "FIXTURE_LOCK.json")
    A_ = jload(RES / "AMENDMENT_A1_FIXTURE_C5_SCOPE.json")
    lc = git("log", "--format=%H", "-1", "--", f"{REL_RES}/FIXTURE_LOCK.json")
    ac = git("log", "--format=%H", "-1", "--", f"{REL_RES}/AMENDMENT_A1_FIXTURE_C5_SCOPE.json")
    sha_at = (lambda c, f_: (lambda b: None if b is None else hashlib.sha256(b).hexdigest())(git_show_bytes(c, f_)))
    bad_lock = [f_ for f_, h in L_["code_files"].items() if sha_at(lc, f_) != h]
    bad_am = [f_ for f_, h in A_["code_files"].items() if sha_at(ac, f_) != h]
    latest = {**L_["code_files"], **A_["code_files"]}
    stage = [f_ for f_ in latest if f_.startswith(("lcr/fixtures.py", "lcr/decoder.py", "lcr/mapper.py", "lcr/run.py",
                                                    "lcr/lock.py", "lcr/sema.py")) or not f_.startswith("lcr/")]
    att1 = [f_ for f_ in stage if sha_at("cc0083aaf", f_) != L_["code_files"][f_]]
    att2 = [f_ for f_ in stage if sha_at("18e41ab13", f_) != latest[f_]]
    wt = [f_ for f_, h in latest.items() if not (WT / f_).exists() or sha_file(WT / f_) != h]
    out.update({"fixture_lock_code_files": len(L_["code_files"]), "fixture_lock_mismatch_at_commit": bad_lock,
                "a1_code_files": sorted(A_["code_files"]), "a1_mismatch_at_commit": bad_am,
                "stage_files_checked": len(stage), "attempt1_evidence_code_differs_from_lock": att1,
                "attempt2_evidence_code_differs_from_lock_plus_a1": att2,
                "worktree_differs_from_latest_locked": wt,
                "a1_changes_only": sorted(A_.get("changes_previously_locked") or []),
                "unlocked_present_at_fixture_lock": sorted((L_.get("unlocked_present") or {}).keys())})
    if bad_lock or bad_am or att1 or att2:
        f.append("locked code mismatch")
    if wt:
        w.append("locked files changed in the worktree after the fixture stage")
    if sorted(A_.get("changes_previously_locked") or []) != ["lcr/fixtures.py", "lcr/tests/test_fixtures.py"]:
        f.append("A1 scope")
    return res("FAIL" if f else ("WARN" if w else "PASS"), failures=f, warnings=w, **out)


def check_chronology_lcr():
    """Stage chronology from the private ACTIVITY_LOG vs first push of each governing lock; no stage other than admit /
    fixture has run; no SCIENCE_LOCK / EVALUATION_LOCK (Adult not launched); assessment labels never unsealed."""
    ev = jsonl(RUN / "ACTIVITY_LOG.jsonl")
    starts = [(e["at"], e["event"].split(" ", 1)[1], e.get("lock")) for e in ev if str(e.get("event", "")).startswith("start ")]
    push = {}
    for nm in ("SOURCE_ADMISSION_LOCK", "FIXTURE_LOCK", "AMENDMENT_A1_FIXTURE_C5_SCOPE"):
        c = git("log", "--format=%H", "--reverse", "--", f"{REL_RES}/{nm}.json")
        c = (c or "").split()[0] if c else None
        push[nm] = _push_time(c) if c else None
    rows, f = [], []
    for at, stg, lk in starts:
        need = {"admit": ["SOURCE_ADMISSION_LOCK"], "fixture": ["FIXTURE_LOCK"]}.get(stg)
        if need is None:
            f.append(f"unexpected stage {stg} at {at}")
            continue
        if stg == "fixture" and parse_iso(at) > parse_iso(push["AMENDMENT_A1_FIXTURE_C5_SCOPE"] or "2100-01-01T00:00:00Z"):
            need = need + ["AMENDMENT_A1_FIXTURE_C5_SCOPE"]
        ok = all(push.get(n_) and parse_iso(push[n_]) <= parse_iso(at) for n_ in need)
        rows.append({"stage": stg, "started": at, "governing": need, "governing_first_push": [push.get(n_) for n_ in need],
                     "after_push": bool(ok)})
        if not ok:
            f.append(f"{stg} at {at} before its lock reached origin")
    later = [p_.name for p_ in RES.glob("*.json") if p_.name in ("SCIENCE_LOCK.json", "EVALUATION_LOCK.json")]
    unsealed = [e for e in ev if "unseal" in str(e.get("event", "")).lower() or "assess" in str(e.get("event", "")).lower()]
    if later:
        f.append(f"later locks present: {later}")
    if unsealed:
        f.append("an assessment / unseal event exists")
    return res("PASS" if not f else "FAIL", failures=f, stages=rows, lock_first_push=push, later_locks_present=later,
               assessment_events=len(unsealed))


FIXTURE_TABS: dict = {}


def _fmt(x, nd=6):
    return "n/a" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def write_oracle_report(orc, gate, path=ORACLE_REPORT):
    """FIXTURE_ORACLE_REPORT.md from the own oracle (fixture_oracle) and the replay (fixture_gate); aggregates only."""
    L = ["# Fixture oracle report (role E, independent verifier)", "",
         f"Verifier: `{REL_RES}/verification/replay_lcr.py` (sha256 `{sha_file(Path(__file__))}`); generated "
         f"{iso(datetime.now(timezone.utc))}. Own code only: no lcr / cbp / qpc module was imported.", "",
         f"Laws: FIXTURE_LAWS.json, laws_sha256 `{orc.get('laws_sha256')}`. Hash recomputed by the registered rule: "
         f"{'verified' if orc.get('hash_ok') else 'MISMATCH'}. Gate rule binds the same laws: "
         f"{orc.get('gate_rule_binds_same_laws')}. Registered tolerances equal: "
         f"{all((orc.get('tolerances_equal_registered') or {}).values())}.", "",
         "## Method", "",
         "- Atoms rebuilt from the explicit tables (pair counts, SEX numerators, per-cell label laws, north-west-corner "
         "joint labels), compared with the stored atoms; static properties recomputed from own atoms.",
         "- Every canonical same-class partition of each recipient's fine cells under the per-class cap enumerated as "
         "restricted-growth strings (counts checked against Stirling sums); all mapping pairs evaluated.",
         "- Exact law quantities from integer tables at N = 4096: plug-in MI of SEX with full token identities; D0 = "
         "smoothed token-mean teacher; D1 = own dual water-filling solve (kappa 32, eps 1e-12, class-dominant simplex) "
         "from exact expected counts, certified by the own Frank-Wolfe gap / KKT certificate; log loss (clip 1e-12), "
         "multiclass Brier, teacher KL of the D0 decoder.",
         "- Exhaustive optimum of every registered own-problem objective; arms' values recomputed from their "
         "partitions; feasibility, local budget, T_star, qualifying set, trigger, route and verdict recomputed from "
         "FIXTURE_GATE_RULE.json by own code.",
         "- Continuous optimisation is certified numerically (registered tolerances); exhaustive enumeration makes the "
         "DISCRETE references exact, not the convex solves.", "", "## Per fixture", ""]
    fams = orc.get("families") or {}
    gf = (gate or {}).get("fixtures") or {}
    for fid, f in fams.items():
        L.append(f"### {fid}")
        L.append("")
        L.append(f"- Law integrity ({f.get('status')}): atoms rebuilt exactly {f.get('atoms_rebuilt_exactly')} "
                 f"(as a set {f.get('atoms_rebuilt_as_set')}); static properties equal {f.get('static_properties_equal')}; "
                 f"rows {f.get('rows')}; mapping pairs {f.get('mapping_pairs')} enumerated {f.get('enumerated_pairs')}.")
        L.append(f"- D1 class preservation on every partition: {f.get('d1_class_preserved_every_partition')}; "
                 f"feasible maps under the fitting budgets (D1): {f.get('feasible_maps')}.")
        if f.get("calibrated_null"):
            cn = f["calibrated_null"]
            L.append(f"- Calibrated null: max |q_D1 - q_D0| = {cn['max_q_diff']:.3g} over {cn['tokens']} token "
                     f"evaluations; min (D1 - D0) law loss per row = {cn['min_law_loss_gap']:.3g} (ok {cn['ok']}).")
        ref = f.get("exhaustive_references_without_local_caps") or {}
        for k_ in ("C-TASK", "D0 FINE-TASK"):
            if k_ in ref:
                L.append(f"- Exhaustive {k_}: value {_fmt(ref[k_]['value'])} at own-enumeration partition indices {ref[k_]['index']}.")
        g = gf.get(fid)
        if g:
            L.append(f"- Replay vs FIXTURE_GATE.json ({g.get('status')}): oracle tables max |diff| "
                     f"{_fmt(g.get('oracle_tables_max_abs_diff'), 3)}; arms checked {g.get('arms_checked')}; "
                     f"partition identity (own CLASS / DIRECT-TASK) {g.get('own_partition_identity')}.")
            tr = g.get("own_trigger") or {}
            L.append(f"- Own trigger: T_star {tr.get('T_star')} (I12 {_fmt(tr.get('T_star_I12'))}); nontrivial "
                     f"{tr.get('nontrivial')}; triggered {tr.get('triggered')}; qualifying {len(tr.get('qualifying') or [])}; "
                     f"decoder_enabled {tr.get('decoder_enabled')}; assignment_search_helps "
                     f"{tr.get('assignment_search_helps')}; best fixed-map I12 {_fmt(tr.get('best_fm_I12'))}; best new I12 "
                     f"{_fmt(tr.get('best_new_I12'))}; exhaustive most-private feasible local pair I12 "
                     f"{_fmt(tr.get('exhaustive_most_private_feasible_local_I12'))}.")
            c6 = g.get("own_C6") or {}
            L.append(f"- Own heuristic labelling: {c6.get('labels')}; max heuristic gap {_fmt(c6.get('max_heuristic_gap'), 3)}.")
            if g.get("failures"):
                L.append(f"- Disagreements: {g['failures']}")
        L.append("")
    ex = (gate or {}).get("checks_c1_c4_descriptive_structural_a1") or {}
    bd = (gate or {}).get("binding") or {}
    if gate and gate.get("status") != "PENDING":
        L += ["## Binding and chronology", "",
              f"- Rule FIXTURE_GATE_RULE.json sha256 `{bd.get('rule_sha256')}` (re-bound; equals the FIXTURE_LOCK "
              f"documents_sha256: {bd.get('rule_equals_lock_documents')}); FIXTURE_LAWS.json equals the lock documents: "
              f"{bd.get('laws_equal_lock_documents')}; laws unchanged since dd1cf23: {bd.get('laws_unchanged_since_dd1cf23')}.",
              f"- FIXTURE_LOCK {str(bd.get('lock_commit'))[:7]} first on origin {bd.get('lock_first_push')}; attempt 1 ran "
              f"{(bd.get('attempt_runs') or {}).get(1)}; AMENDMENT_A1 {str(bd.get('a1_commit'))[:7]} first on origin "
              f"{bd.get('a1_first_push')}; attempt 2 ran {(bd.get('attempt_runs') or {}).get(2)}. Lock before attempt 1: "
              f"{bd.get('lock_pushed_before_attempt1')}; amendment before attempt 2: {bd.get('a1_pushed_before_attempt2')}.",
              f"- FIXTURE_GATE.json binds the rule and laws: {bd.get('gate_binds_rule_and_laws')}; its oracle-table hashes "
              f"re-hash: {bd.get('oracle_tables_rehash')}; fix__ units of both attempts complete and bound to the laws hash: "
              f"{all((bd.get('fix_units_complete_and_bound') or {}).values())}.", "",
              "## Registered expectation vs result", "",
              "| Fixture | Registered expectation (PROTOCOL sec. 9, before the stage) | Lead result | Own replay |",
              "|---|---|---|---|"]
        exp = {"F1_CALIBRATED_NULL": "NO_FEASIBLE_TASK_ONLY (no budget-feasible task-only map)"}
        for fid, g in gf.items():
            tr = g.get("own_trigger") or {}
            own_txt = (tr.get("reason") or f"T* = {tr.get('T_star')}, I12(T*) = {_fmt(tr.get('T_star_I12'))}, "
                                              f"nontrivial {tr.get('nontrivial')}") + f"; triggered {tr.get('triggered')}"
            L.append(f"| {fid} | {exp.get(fid, 'T* = CLASS|D1 with I12 = 0: not nontrivial')} | as own (replay "
                     f"{g.get('status')}) | {own_txt} |")
        L.append("")
        sb = (ex.get("structural_bound") or {}).get("per_fixture") or {}
        if sb:
            L += ["## Structural bound I12(R) >= I12(CLASS|D1)", "",
                  "Every class-preserving token refines the teacher-predicted class, so the token tuple determines (d1, d2) "
                  "and I(S; t1, t2) >= I(S; d1, d2) = I12(CLASS) (data processing). Checked exactly and exhaustively:", "",
                  "| Fixture | I(S; d1, d2) = 0 exactly (integer table) | I12(CLASS) | min I12 over every enumerated pair | "
                  "pairs below CLASS | min reported arm I12 | CLASS|D1 budget-feasible |", "|---|---|---|---|---|---|---|"]
            for fid, v in sb.items():
                L.append(f"| {fid} | {v['I_S_given_decision_pair_exactly_zero']} | {_fmt(v['I12_CLASS'])} | "
                         f"{_fmt(v['min_I12_over_all_enumerated_pairs'])} | {v['pairs_with_I12_below_CLASS']} | "
                         f"{_fmt(v['min_reported_arm_I12'])} | {v['CLASS_D1_budget_feasible']} |")
            L += ["", "Consequence: on F2-F4 the feasible task-only reference already has I12 = 0, so no release (not even "
                      "the exhaustive most-private feasible local pair, I12 = 0) can reduce pair MI by 0.01; on F1 no "
                      "task-only candidate is feasible. The registered fixtures could not trigger; this is not evidence "
                      "about the mechanism on Adult.", ""]
        dfl = (ex.get("descriptive_flags") or {}).get("own") or {}
        if dfl:
            L += ["## Descriptive flags (own recomputation; never used by the verdict)", "",
                  "| Fixture | best I12 constrained (all / eligible) | best I12 weighted (all / eligible) | constrained - "
                  "weighted (eligible) | joint - sequential (d1 / weighted / constrained, eligible) |", "|---|---|---|---|---|"]
            for fid, v in dfl.items():
                bi = v["best_I12"]
                jm = v["joint_minus_sequential"]
                L.append(f"| {fid} | {_fmt(bi['constrained']['all'])} / {_fmt(bi['constrained']['eligible'])} | "
                         f"{_fmt(bi['weighted']['all'])} / {_fmt(bi['weighted']['eligible'])} | "
                         f"{_fmt(v['constrained_minus_weighted']['eligible'])} | {_fmt(jm['d1_fixed']['eligible'])} / "
                         f"{_fmt(jm['weighted']['eligible'])} / {_fmt(jm['constrained']['eligible'])} |")
            L.append("")
        a1 = ex.get("amendment_A1") or {}
        if a1:
            L += ["## Amendment A1 (C5 scope): cause and correction", "",
                  f"- Status {a1.get('status')}. Attempt 1 C5 failures on F1: {a1.get('attempt1_C5_failures', {}).get('F1_CALIBRATED_NULL')}; "
                  f"attempt 2 NO_SEARCH_STATE lists: {a1.get('no_search_state_lists_attempt2')} (exactly K-SEQ-12 and "
                  f"K-SEQ-21 on F1: {a1.get('applies_exactly_to_K_SEQ')}).",
                  f"- F1 constrained mapper statuses (persisted fix__ unit): {a1.get('F1_constrained_mapper_status')}.",
                  f"- Own cause derivation (own oracle + the registered sequential driver): recipient-2 maps that pass "
                  f"the search feasibility (budgets - 1e-10, local cap) on F1: {a1.get('F1_recipient2_maps_search_feasible')}; "
                  f"recipient-1 maps: {a1.get('F1_recipient1_maps_search_feasible')}. K-SEQ-21 (recipient 2 first) "
                  f"is infeasible at stage 1 from every start; K-SEQ-12 is feasible at stage 1 from every start and "
                  f"infeasible at stage 2. No start can reach a refined final: {a1.get('no_start_can_reach_a_refined_final')}. "
                  "K-LOCAL refines recipient 1 (refined_descriptive) and the joint drivers record an 'unchanged' witness "
                  "block, so a state exists for them; NO_SEARCH_STATE can apply only to the sequential arms.",
                  f"- Wording disagreement (does not affect the correction): the amendment says both arms hit "
                  f"INFEASIBLE_START at stage 1; for K-SEQ-12 the failure is at stage 2 ({len(a1.get('mismatch_with_amendment_wording') or [])} starts).",
                  f"- Identity: FIXTURE_GATE.json attempt 1 vs 2 differ only at {a1.get('gate_json_differing_paths')} "
                  f"(outside C5 / reasons: {a1.get('gate_json_differences_outside_C5_and_verdict_reasons')}); oracle CSVs "
                  f"byte-identical: {a1.get('oracle_csv_byte_identical')}; fix__ results identical outside C5 and timing: "
                  f"{not any((a1.get('fix_unit_results_differences_outside_C5_and_timing') or {}).values())}; rule and laws "
                  f"unchanged by A1: {a1.get('rule_and_laws_unchanged_by_A1')}; code files changed: "
                  f"{[f_ for f_ in a1.get('code_change_files') or [] if f_.startswith('lcr/')]}.", ""]
        L += ["## Gate", "",
              f"- Lead verdict {gate.get('lead_verdict')}, route {gate.get('lead_route')}, reasons {gate.get('lead_reasons')}.",
              f"- Own verdict {gate.get('own_verdict')}, route {gate.get('own_route')}, triggered {gate.get('own_triggered')}.",
              f"- Verdict agrees: {gate.get('verdict_agrees')}; route agrees: {gate.get('route_agrees')}; replay status "
              f"{gate.get('status')}.", ""]
    else:
        L += ["## Gate", "", "- PENDING: FIXTURE_GATE.json not yet written (own references precomputed).", ""]
    L += ["## Scope", "",
          "Fixture MI is the exact law MI of these finite laws, not a population guarantee for Adult. Optimisers that "
          "differ from the exhaustive references are labelled HEURISTIC with their gap; that is a report, not a failure.",
          ""]
    text = "\n".join(L)
    scrub_check(text)
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def check_fixture_gate():
    """PHASE 1 fixture half: own law rebuild / hash / static properties / exhaustive oracle (fixture_oracle) and, once
    the fixture stage has written FIXTURE_GATE.json, the replay of B's tables, arms, trigger, verdict and route
    (fixture_gate)."""
    if not (RES / "FIXTURE_LAWS.json").exists() or not (RES / "FIXTURE_GATE_RULE.json").exists():
        return pending("fixture_gate", "awaiting the laws"), pending("fixture_oracle", "awaiting the laws")
    orc, tabs = check_fixture_laws()
    FIXTURE_TABS.update(tabs)
    gate = check_fixture_gate_replay(tabs)
    if gate.get("status") == "PENDING":
        return gate, orc
    gate["binding"] = check_fixture_binding()
    gate["checks_c1_c4_descriptive_structural_a1"] = check_fixture_extras(tabs)
    gate["status"] = worst(gate["status"], gate["binding"]["status"], gate["checks_c1_c4_descriptive_structural_a1"]["status"])
    return gate, orc


# ================================================================================================ lcr PHASE 2: D1 units
def lcr_unit(k, cid):
    if cid.startswith("SRC|"):
        return f"tea__s{k}__{cid.split('|')[1]}"
    if cid.startswith("REF|"):
        return f"ref__s{k}__{cid.split('|')[1]}"
    pre = {"d0": "pol", "d1_fixed": "dec"}.get(lcr_arm(cid), "new")
    return f"{pre}__s{k}__{cid.replace('|', '_')}"


def lcr_d1_ids():
    fixed = [L_d1("DIRECT-TASK"), L_d1("FINE-TASK")] + [L_d1(f, l_) for l_ in LAMS for f in LCR_PRIV]
    new = [L_CTASK] + [L_w(f, l_) for l_ in LAMS for f in LCR_PRIV] + [L_k(a) for a in LCR_K]
    return fixed, new


def registered_decoder_hash(body):
    z = {k: v for k, v in body.items() if k != "decoder_sha256"}
    return hashlib.sha256(json.dumps(z, sort_keys=True, allow_nan=False).encode()).hexdigest()


def fit_labels(D: Data, L):
    tr = D.idx[FIT]
    y = {1: L["y_income"][tr], 2: L["y_occ"][tr]}
    assert all((y[i] >= 0).all() for i in (1, 2)) and (L["sex"][tr] >= 0).all()
    return tr, y, L["sex"][tr]


def check_one_d1_unit(un, k, cid, D, L, T, cache, tr, yfit, sfit, d0_rel=None):
    """One D1 unit: binding and hash of decoder.json; per recipient: exact recount of n_t / y_t from the fitting rows,
    teacher sums, own re-solve (q within 1e-9), own certificate of the STORED released q, fallback tokens = the pinned
    D0 vector, release rows = table[tok] bitwise, decisions = teacher, token function; tokens equal the D0 release
    (fixed maps) or the own re-encode of policy.json (new units); fitting L_i, B_i, I_i, I12, T, Phi from the release."""
    d = UNITS / un
    body = jload(d / "decoder.json")
    rec = jload(d / "record.json")
    z = np.load(d / "release.npz", allow_pickle=False)
    P = {1: T[(k, "U")]["p1"], 2: T[(k, "U")]["p2"]}
    dd = {1: T[(k, "U")]["d1"], 2: T[(k, "U")]["d2"]}
    own = policy_pair_release(d / "policy.json", P, dd)
    f, info = [], {}
    if body.get("decoder_sha256") != registered_decoder_hash(body):
        f.append("decoder_sha256")
    if body.get("config") != cid:
        f.append("decoder config binding")
    if frozenset(z.files) != RELEASE_KEYS_LCR or not np.array_equal(z["row_id"], D.row_id):
        f.append("release keys / rows")
    worst = {"dq": 0.0, "fw_gap_rel": 0.0, "stat_rel": 0.0, "S": 0.0, "proj_reported": 0.0}
    terms = {}
    ntok = nfb = 0
    for i in (1, 2):
        tab = body[f"r{i}"]
        K = int(tab["K"])
        tc = np.asarray(tab["token_class"], np.int64)
        n_ = np.asarray(tab["n"], np.float64)
        Y_ = np.asarray(tab["y"], np.float64).reshape(-1, K)
        S_ = np.asarray(tab["s"], np.float64).reshape(-1, K)
        U_ = np.asarray(tab["u"], np.float64).reshape(-1, K)
        Q_ = np.asarray(tab["q"], np.float64).reshape(-1, K)
        fb = np.asarray(tab["fallback"], bool)
        Tn = len(tc)
        tok = np.asarray(z[f"tok{i}"], np.int64)
        if d0_rel is not None and not np.array_equal(tok, d0_rel[f"tok{i}"]):
            f.append(f"r{i}: tokens differ from the D0 release")
        if not np.array_equal(tok, own[f"tok{i}"]):
            f.append(f"r{i}: tokens differ from the own re-encode of policy.json")
        if not np.array_equal(z[f"hard{i}"], dd[i]):
            f.append(f"r{i}: decisions differ from the teacher")
        if int(z[f"alpha{i}"]) != Tn:
            f.append(f"r{i}: alphabet")
        if not np.array_equal(z[f"q{i}"], Q_[tok]):
            f.append(f"r{i}: release rows differ from the decoder table")
        nt, Yt, St = token_stats(tok[tr], yfit[i], P[i][tr], K, Tn)
        if not (np.array_equal(nt, n_) and np.array_equal(Yt, Y_)):
            f.append(f"r{i}: n_t / y_t differ from the own recount on OSF_DEFENSE_FIT")
        worst["S"] = max(worst["S"], float(np.max(np.abs(St - S_) / np.maximum(1.0, n_)[:, None])))
        if not np.array_equal(fb, n_ == 0):
            f.append(f"r{i}: fallback flags")
        own_pol = Pol.from_json(jload(d / "policy.json")[f"p{i}"])     # own D0 prototypes of the same map
        if not np.array_equal(own_pol.token_class, tc):
            f.append(f"r{i}: token classes differ from the own policy")
        for t_ in range(Tn):
            if fb[t_]:
                nfb += 1
                if not np.array_equal(Q_[t_], own_pol.proto[t_]):
                    f.append(f"r{i} t{t_}: fallback is not the pinned D0 vector")
                continue
            ntok += 1
            _, q_own, _ = cache.solve(n_[t_], Y_[t_], S_[t_], int(tc[t_]))
            dq = float(np.max(np.abs(Q_[t_] - q_own)))
            c = d1_certificate(U_[t_], Q_[t_], n_[t_], Y_[t_], S_[t_], int(tc[t_]))
            worst["dq"] = max(worst["dq"], dq)
            worst["fw_gap_rel"] = max(worst["fw_gap_rel"], float(c.get("fw_gap_rel") or 0.0))
            worst["stat_rel"] = max(worst["stat_rel"], float(c.get("stationarity_rel") or 0.0))
            if dq > D1_TOL["q"] or not c["ok"]:
                f.append(f"r{i} t{t_}: {'q' if dq > D1_TOL['q'] else 'certificate'} (dq {dq:.2e})")
        for ce in tab.get("certs") or []:
            if isinstance(ce, dict) and ce.get("projection_magnitude") is not None:
                worst["proj_reported"] = max(worst["proj_reported"], float(ce["projection_magnitude"]))
        Lr, Br = fit_losses(np.asarray(z[f"q{i}"])[tr], yfit[i])
        LU, BU = fit_losses(P[i][tr], yfit[i])
        terms[i] = {"L": Lr, "B": Br, "L_U": LU, "B_U": BU, "I": mi_of(sfit, tok[tr]),
                    "budget": budget_check(Lr, Br, LU, BU), "tok": tok[tr]}
        if not (strict_argmax_ok(z[f"q{i}"], z[f"hard{i}"]) and release_is_token_function(tok, z[f"q{i}"])):
            f.append(f"r{i}: class preservation / token function")
    if worst["S"] > D1_TOL["teacher_sum_rel"]:
        f.append(f"teacher sums differ by {worst['S']:.2e} relative")
    I12 = mi_of(sfit, terms[1]["tok"], terms[2]["tok"])
    own_terms = {"L1": terms[1]["L"], "L2": terms[2]["L"], "B1": terms[1]["B"], "B2": terms[2]["B"],
                 "I1": terms[1]["I"], "I2": terms[2]["I"], "I12": I12}
    own_terms["T"] = own_terms["L1"] + own_terms["L2"] + 0.5 * (own_terms["B1"] + own_terms["B2"])
    own_terms["Phi"] = phi_of(own_terms["I1"], own_terms["I2"], I12)
    info.update({"tokens_solved": ntok, "fallback_tokens": nfb, "worst": worst, "own_terms": own_terms,
                 "budgets": {str(i): {x: terms[i][x] for x in ("L", "B", "L_U", "B_U")} | {
                     "ok": terms[i]["budget"]["ok"], "ll_margin": terms[i]["budget"]["ll_margin"],
                     "brier_margin": terms[i]["budget"]["brier_margin"],
                     "borderline": terms[i]["budget"]["borderline"]} for i in (1, 2)},
                 "pair_fp": hashlib.sha256(b"".join(np.ascontiguousarray(np.asarray(z[x])).tobytes() for x in
                                                    ("tok1", "q1", "tok2", "q2"))).hexdigest()})
    return f, info, rec


def check_d1_units(D: Data, L, T, which="fixed"):
    """PHASE 2: every D1 unit of the registered bank (26 fixed-map controls or 30 new fits per seed)."""
    tr, yfit, sfit = fit_labels(D, L)
    fixed, new = lcr_d1_ids()
    cids = fixed if which == "fixed" else new
    cache = D1Cache()
    out, bad, info_all = {}, [], {}
    for k in SEEDS:
        for cid in cids:
            un = lcr_unit(k, cid)
            if not (UNITS / un / "COMPLETE.json").exists():
                bad.append(f"{un}: absent")
                continue
            cu, _ = complete_ok(UNITS / un)
            d0_rel = None
            if which == "fixed":
                z0 = np.load(UNITS / lcr_pol_unit(k, cid[:-3]) / "release.npz", allow_pickle=False)
                d0_rel = {x: z0[x] for x in z0.files}
            f, info, rec = check_one_d1_unit(un, k, cid, D, L, T, cache, tr, yfit, sfit, d0_rel)
            if not (cu["id_ok"] and cu["rehash_ok"] and not cu["unlisted"]):
                f.append("COMPLETE.json")
            info_all[(k, cid)] = info
            out[un] = {"ok": not f, "failures": f[:8]}
            if f:
                bad.append(un)
    worst = {x: max([v["worst"][x] for v in info_all.values()] or [0.0]) for x in
             ("dq", "fw_gap_rel", "stat_rel", "S", "proj_reported")}
    ntok = sum(v["tokens_solved"] for v in info_all.values())
    exp = len(cids) * len(SEEDS)
    return res("PASS" if not bad and len(out) == exp else "FAIL", units=len(out), expected=exp, failures=bad[:30],
               tokens_certified=ntok, worst=worst, solves={"hits": cache.hits, "misses": cache.miss},
               unit_status={u: v for u, v in out.items() if not v["ok"]}), info_all


def nominee_matches_inner(lock_resolved, own_resolved):
    """The evaluation lock must carry exactly the inner-selected role configurations (no assessment re-scoring)."""
    return all(lock_resolved.get(x) == own_resolved.get(x) for x in own_resolved)


def selftests():
    out = {}
    rng = np.random.default_rng(20261006)
    # ---- import guard (also for unpickling a project class reference)
    _IN_SELFTEST[0] = True
    tried = []
    for m in ("lcr", "lcr.run", "lcr.decoder", "lcr.mapper", "lcr.fixtures", "lcr.select", "lcr.audit", "lcr.sema",
              "lcr.lock", "lcr.deploy", "cbp", "cbp.run", "cbp.select", "cbp.sema", "cbp.lock", "qpc", "qpc.kmeans",
              "qpc.compress", "qpc.utility", "qpc.sema", "dpc", "dpc.partition", "osf.data", "smf.audit", "rgj.finalize",
              "jcv.train", "pnx", "oar.study", "stored_model_eval.pilot_infer", "pcrl"):
        try:
            importlib.import_module(m)
            tried.append((m, False))
        except ImportError:
            tried.append((m, True))
    unp = {}
    for blob, nm in ((b"cqpc.kmeans\nfit\n.", "qpc.kmeans"), (b"cdpc.partition\nFinePartition\n.", "dpc.partition"),
                     (b"ccbp.select\nselect_all\n.", "cbp.select"), (b"clcr.decoder\nsolve\n.", "lcr.decoder"),
                     (b"clcr.mapper\nPolicyPair\n.", "lcr.mapper")):
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
    out["cbp_selection_rules_retained"] = selftest_selection()      # the adapted cbp machinery still behaves
    out["cbp_label_truth_table_retained"] = selftest_labels()
    out["cbp_deliberate_defects_retained"] = selftest_defects(rng)
    # ---- lcr (prompt sections 6-14)
    rl = np.random.default_rng(20261007)
    t_ = time.time()
    out["lcr_d1_decoder"] = selftest_decoder(rl)
    out["lcr_fixture_oracle"] = selftest_oracle(rl)
    out["lcr_engine_incremental"] = selftest_engine(rl)
    out["lcr_selection_rules"] = selftest_selection_lcr()
    out["lcr_label_truth_table"] = selftest_labels_lcr()
    out["lcr_deliberate_defects"] = selftest_defects_lcr(rl)
    out["lcr_selftest_wall_s"] = res("INFO", wall_s=round(time.time() - t_, 2))
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


PHASE1 = ("fixture_gate", "fixture_oracle", "pins", "roles", "admitted_custody", "teachers", "references",
          "d0_release_reencode", "code_hashes", "chronology")
PHASE2 = ("d1_fixed_maps", "d1_new_units", "fitting_budgets_and_caps", "incremental_parity", "decision_preservation",
          "inner_audits", "source_composition", "selection", "controls")
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
    report = {"schema": "lcr-independent-verification-v1", "phase": args.label or f"PHASE_{args.phase}",
              "generated_at": iso(datetime.now(timezone.utc)),
              "verifier": f"{REL_RES}/verification/replay_lcr.py", "verifier_sha256": sha_file(Path(__file__)),
              "adapted_from": f"{CBP_REL}/verification/replay_cbp.py at {CBP_TIP[:7]} (sha256 " +
                              (sha_file(WT / CBP_REL / "verification" / "replay_cbp.py")
                               if (WT / CBP_REL / "verification" / "replay_cbp.py").exists() else "absent") + ")",
              "worktree_head": git("rev-parse", "HEAD"), "branch": BRANCH, "source_tip": CBP_TIP,
              "inputs": {"source_npz": "<PRIVATE_CACHE>/jcv_v1/inputs/adult_jcv.npz",
                         "lcr_units": "<PRIVATE_CACHE>/lcr_v1/run/units", "lcr_admitted": "<PRIVATE_CACHE>/lcr_v1/admitted",
                         "cbp_units_read_only": "<PRIVATE_CACHE>/cbp_v1/run/units",
                         "cbp_copy_read_only": "<PRIVATE_CACHE>/cbp_v1_local_copy_20261006"},
              "tolerances": {"lcr_d1_decoder_and_certificate": D1_TOL,
                             "lcr_d1_note": "q agreement absolute; gap / stationarity / objective relative to "
                                            "max(1, n_t + kappa); dominance and strict argmax of the released q exact",
                             "tokens_decisions_partitions_bindings_hashes": "exact (bitwise / byte-identical)",
                             "decoded_probabilities_teacher_outputs": "exact (bitwise)",
                             "search_increments": LOG_TOL_B, "objective_terms": OBJ_TOL_B,
                             "k-means_receipt_objectives_relative": RECEIPT_REL_TOL, "fit_distortion_relative": 1e-12,
                             "inner_auc_ce_utility": 1e-12, "endpoint_point_se": 1e-12, "endpoint_bounds": 1e-11,
                             "published_6_decimal_prints": 5e-7, "verdicts_labels_roles": "exact agreement",
                             "selection_ordering_ties": f"{ORDER_DECIMALS}-decimal rounding (provisional; reconciled "
                                                        "with SELECTION_RULES.json)"},
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
        man = {"lcr_ROLE_MANIFEST": jload(RES / "ROLE_MANIFEST.json") if (RES / "ROLE_MANIFEST.json").exists() else None,
               "cbp_ROLE_MANIFEST_at_tip": json.loads(git_show_bytes(CBP_TIP, f"{CBP_REL}/ROLE_MANIFEST.json") or b"null")}
        checks["roles"] = check_roles(D, man)
        checks["pins"] = check_pins_lcr()
        checks["admitted_custody"] = check_custody_lcr()
        checks["teachers"], T = check_teachers_lcr(D, L)
        checks["d0_release_reencode"] = check_d0_reencode(D, T)
        checks["fixture_gate"], checks["fixture_oracle"] = check_fixture_gate()
        checks["code_hashes"] = check_code_hashes_lcr()
        checks["chronology"] = check_chronology_lcr()
        checks["references"] = res("INFO", note="admitted REF|E / REF|F / REF|F0 units are custody-verified by hash in "
                                                "admitted_custody; their numeric recheck belongs to the Adult inner / "
                                                "assessment replay")
        if checks["fixture_gate"].get("own_verdict") == "GATE_NOT_MET" and \
                checks["fixture_gate"].get("lead_verdict") == "GATE_NOT_MET":
            why = ("the registered fixture gate is not met (own replay agrees: GATE_NOT_MET); no Adult fit is launched "
                   "(prompt section 9), so this Adult replay has no object")
            for k_ in PHASE2 + ("evaluation_lock", "outer_units", "endpoints", "chronology_assessment", "attacker_refits"):
                checks[k_] = res("NOT_APPLICABLE", reason=why)
        if checks["fixture_oracle"].get("status") != "PENDING" and not args.no_write and args.out is None:
            report["fixture_oracle_report"] = {"path": f"{REL_RES}/FIXTURE_ORACLE_REPORT.md",
                                               "sha256": write_oracle_report(checks["fixture_oracle"],
                                                                             checks["fixture_gate"])}
        if args.phase >= 2:
            raise SystemExit("PHASE_2+ of the lcr verifier is adapted after the fits (not yet)")
    if False:
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
                         "other_lcr_processes_at_start": len(others_start),
                         "other_lcr_processes_at_end": len(heavy_processes()), "semaphore_slots_at_end": sema_status()}
    report["pending"] = sorted(k for k, v in status.items() if v == "PENDING")
    sema_runs = [{"label": e.get("label"), "released_at": e.get("at"), "wall_s": e.get("wall_s"), "cpu_s": e.get("cpu_s"),
                  "rc": e.get("rc")} for e in jsonl(RUN / "SEMA_LOG.jsonl")
                 if e.get("event") == "release" and str(e.get("label", "")).startswith("E:")]
    report["verifier_compute_ledger"] = {
        "semaphore_holds_completed": sema_runs,
        "semaphore_cpu_s_total": round(sum(float(r["cpu_s"] or 0) for r in sema_runs), 1),
        "semaphore_wall_s_total": round(sum(float(r["wall_s"] or 0) for r in sema_runs), 1),
        "this_run": {"phase": args.phase, "start": report["generated_at"], "wall_s": report["compute"]["wall_s"],
                     "cpu_s": report["compute"]["cpu_s_process"]},
        "light_runs_outside_the_semaphore": VERIFIER_RUNS,
        "note": "every verifier process, including the synthetic self-tests, runs inside a semaphore hold labelled E:*; "
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
