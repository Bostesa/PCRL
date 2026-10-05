#!/usr/bin/env python3
"""Independent verifier for the online-strength frontier study (osf).

Owner: the independent verifier (prompt section 4 role 5; section 17). Adapted from the predecessor's independent
verifier (results/pcrl_strength_matched_feedback_v1/verification/replay_smf.py, itself independent code). Works from the
registered definitions (PROTOCOL.md, METHOD_CARD.md, ROLE_MANIFEST.json rule text, the named locks and dated amendments,
PRIMARY_FAMILY.json / SECONDARY_FAMILY.json) and the saved private unit artifacts. The lead's source was READ to learn
file formats and registered rules; nothing of it is executed here:

  * a sys.meta_path guard refuses every import under osf, smf, rgj, jcv, pnx, oar and stored_model_eval (this includes
    imports triggered while unpickling joblib heads); torch.load is always called with weights_only=True; the run asserts
    at the end that none of these packages is loaded;
  * scientific libraries: numpy, scipy, scikit-learn, torch, joblib (+ json, hashlib and the standard library).

PHASE 1 checks (status PASS / FAIL / WARN / PENDING / INFO):
  roles         exact-rational group-hash recomputation of the old/rgj/smf/osf roles, subroles and the four
                assessment pools (minus groups touching fitting roles or exclusions); counts / row-id / group-set hashes /
                partition fingerprint / numeric refit vs ROLE_MANIFEST.json; assessment labels unused (no unseal, no
                outer units, releases carry no label arrays, heads fitted on OSF_DEFENSE_FIT rows only)
  releases      own forward pass (Linear-ReLU-Linear-ReLU-Linear 83-64-64-16 per encoder) from model.pt on the
                verifier's own X; own deployed-head refit (StandardScaler + LogisticRegression, C grid, HEAD_VALIDATION
                log loss, strict improvement > 1e-12, ties -> smaller C) reproducing the released centred logits,
                probabilities and hard decisions; admitted units equal the smf releases on every smf row
  receipts      per-step identities from every run__*/steps.npz (RAW q/t = ratio, scale = beta, p = q/beta; NORM
                ratio = rho s_i on non-zero non-capped steps, cap/zero semantics, RMS identity, combined ratio, clip
                factor min(1, 5/pre), post = min(pre, 5), update-norm algebra |t+q|^2 = t^2+q^2+2cos t q, zero codes vs
                R, no pair encoder gradient in local arms, critic updates 5 x 6 per step, epoch/diag/summary sums)
  common_rng    minibatch fingerprints identical across every configuration of a seed AND equal to the verifier's own
                permutation fingerprints; step-1 task norms / transforms / R identical across configurations
  step_replay   independent recomputation (own functional forward, transforms, critics, surrogate, RAW/NORM update and
                clipped SGD step) of the final update theta_{T-1} -> theta_T of every run and of every captured step
                (fractions 0, .25, .5, .75, 1) against the saved parameters and the receipts
  train_replay  full independent 40-epoch re-training from the admitted warm state (own SGD/critic loop) for U on every
                seed and a declared set of protected runs; final parameters and every per-step receipt compared
  replays       the 15 admitted replays: ck20 / final theta_T / final critics equal the admitted smf checkpoints (own
                tensor comparison); 40-epoch receipts reproduce the admitted rgj logged norms (pre-clip total exactly)
  engineering   parity receipts; fidelity receipts incl. algebraic re-derivation of the frozen-minibatch equivalence
                numbers and of the registered expected failures; equivalence t_norm equals the replay receipt
  integrity     COMPLETE.json hashes; lock / amendment push before every governed stage start and unit; locked code
                hashes vs the working tree; quarantined pre-lock receipts inventoried (kept, not evidence)
  controls      cheap sanity controls under the permitted attacker interface (released views only): shuffled-SEX null on
                a separate split and a planted rotated clue inside a released view
PHASE 2 (selection, assessment-opening order, endpoints, conjunctions, attacker restore) are PENDING until the
SELECTION_AND_AUDIT_LOCK / EVALUATION_LOCK / inference artifacts exist.

Usage (from the worktree root):
    OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python results/pcrl_online_strength_frontier_v1/verification/replay_osf.py
        [--selftest-only] [--seeds 0,1,2] [--releases all|seed0|none] [--train-replay default|none|<run,run>]
        [--no-write]
Writes results/pcrl_online_strength_frontier_v1/INDEPENDENT_VERIFICATION.json (aggregates and placeholders only).
"""
from __future__ import annotations

import importlib
import importlib.abc
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

# ------------------------------------------------------------------------------------------------ import guard
_FORBIDDEN_TOP = ("osf", "smf", "rgj", "jcv", "pnx", "oar", "stored_model_eval")
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
# the worktree root must not shadow the guard through a cached path entry: drop it from sys.path as well
_WT_GUESS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _WT_GUESS]

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from fractions import Fraction  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.stats import norm  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import log_loss, roc_auc_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

torch.set_num_threads(1)

# ------------------------------------------------------------------------------------------------ constants
HERE = Path(__file__).resolve().parent
RES = HERE.parent
WT = RES.parents[1]
REL_RES = "results/pcrl_online_strength_frontier_v1"
OUT = RES / "INDEPENDENT_VERIFICATION.json"
CACHE = Path.home() / "PCRL_eval_cache_private"
SRC = CACHE / "jcv_v1" / "inputs" / "adult_jcv.npz"
SRC_SHA = "e0d9e54af780f30788ee29cfe6795ec82cbdcadc127b1978c69a3891485d2f12"
PRIV = CACHE / "osf_v1"
RUN = PRIV / "run"
UNITS = RUN / "units"
ADMITTED = PRIV / "admitted"
ADMISSION_NORM = WT / "results" / "pcrl_joint_complete_view_method_v1" / "DATA_ADMISSION.json"
BRANCH = "research/pcrl-online-strength-frontier-v1"

SEEDS = (0, 1, 2)
ROLES = ("OSF_DEFENSE_FIT", "OSF_DEVELOPMENT_ASSESSMENT", "HEAD_VALIDATION", "AUDIT_FIT", "INNER_SELECTION")
SUBROLES = ("CRITIC_FIT", "CRITIC_VAL", "DIAGNOSTIC_CALIB")
FIT_ROLES = ("OSF_DEFENSE_FIT", "HEAD_VALIDATION", "AUDIT_FIT", "INNER_SELECTION")
POOLS = ("ORIG_ASSESSMENT", "RGJ_DEV", "SMF_DEV", "CERT")
EXCLUSIONS = ("excluded_exposure", "excluded_dup")
FIT, ASSESS = "OSF_DEFENSE_FIT", "OSF_DEVELOPMENT_ASSESSMENT"
NUMERIC = ("age", "education-num", "capital-gain", "capital-loss", "hours-per-week")
LABEL_KEYS = {"sex": "sex", "race": "race", "y_income": "y_income", "y_occ": "y_occupation_group"}
RELEASE_KEYS = {"row_id", "r1", "c1", "p1", "hard1", "r2", "c2", "p2", "hard2"}
KS = (2, 6)

# registered training numbers (PROTOCOL section 4; DATA_AND_ENGINEERING_LOCK HP)
BATCH, LR, CLIP, EPOCHS = 256, 0.05, 5.0, 40
A_MAX, ZERO_TOL = 100.0, 1e-12
CRITIC_LR, CRITIC_STEPS, WHITEN_FLOOR = 3e-3, 5, 1e-8
REF_SIZE, REF_SEED = 4096, 20261004
VIEWS, KINDS = ("v1", "v2", "pair"), ("A", "B")
DV = {"v1": 16 + KS[0], "v2": 16 + KS[1], "pair": 32 + KS[0] + KS[1]}
HEAD_C = (0.01, 0.1, 1.0, 10.0, 100.0)
PROGRESS = (0.0, 0.25, 0.5, 0.75, 1.0)
CAPTURE_CONFIGS = ("RAW-J|b0.3", "RAW-L|b0.3", "NORM-J|r3|a1", "NORM-L|r3|a1")
ADMITTED_IDS = ("U", "RAW-J|b0.1", "RAW-J|b0.3", "RAW-L|b0.1", "RAW-L|b0.3")
B_BOOT, BOOT_SEED, CHUNK = 1999, 20261006, 250

STATUS_RANK = {"FAIL": 4, "PENDING": 3, "WARN": 2, "PASS": 1, "INFO": 0, "NOT_APPLICABLE": 0}

# every correction made to this verifier's own code (kept; prompt section 17 "preserve every correction")
VERIFIER_CORRECTIONS = [
    {"at": "2026-10-05T04:12Z", "where": "selftests.norm_ratio_and_local_isolation",
     "what": "the toy NORM fixture compared ||q||/||t|| with rho*s_i also on encoders whose scalar hit the cap 100 "
             "(tiny proxy gradient of random critics); the own receipt now records the cap and capped encoders are "
             "checked for ratio < rho*s_i instead", "effect": "self-test only; no study number involved"},
    {"at": "2026-10-05T04:20Z", "where": "check_label_custody",
     "what": "with no release replayed the head-fit-row and label-array conditions were vacuously true; they are now "
             "None and the check is PENDING until releases are replayed", "effect": "status only"},
    {"at": "2026-10-05T04:20Z", "where": "check_steps (local arms)",
     "what": "the receipts-only 'pair signal but no own-view signal' test is vacuous because local arms never evaluate "
             "R_pair (NaN by design); kept as a descriptive count, and a power check was added to step_replay: the "
             "final update recomputed WITH the trained shadow pair critic must not reproduce theta_T",
     "effect": "adds a discriminating test; no study number involved"},
    {"at": "2026-10-05T04:35Z", "where": "replay_selection (F, F0 rows)",
     "what": "the binding reference status (F NO_FEASIBLE_NOMINEE, F0 INFEASIBLE_CONTROL) was first applied to the row "
             "only; it is now applied per seed as well, as the registered rule makes the reference path's status binding",
     "effect": "corrected before the first comparison with selection.json; statuses unaffected"},
    {"at": "2026-10-05T04:50Z", "where": "check_label_custody / check_lock_order / check_assessment_order",
     "what": "the assessment start is logged as the ACTIVITY_LOG event 'assessment opened' (not 'start assess'); the "
             "event is now recognised and every assessment event is required to follow the EVALUATION_LOCK push",
     "effect": "the ordering test became non-vacuous; no study number involved"},
    {"at": "2026-10-05T05:10Z", "where": "check_endpoints",
     "what": "assessment predictions are read on demand (LazyNpz) instead of decompressing every array of 72 files",
     "effect": "memory only"},
]


def g(x) -> str:
    return f"{x:g}"


def bank_ids(kind="full"):
    """The registered bank (PROTOCOL section 5): U; RAW beta; NORM symmetric rho (a=1); NORM allocations a 0.5/2."""
    betas = (0.1, 0.3, 0.6) if kind == "full" else (0.1, 0.3)
    sym = (1.5, 3.0, 5.0) if kind == "full" else (1.5, 3.0)
    arho = (3.0, 5.0) if kind == "full" else (3.0,)
    out = ["U"]
    for t in ("J", "L"):
        out += [f"RAW-{t}|b{g(b)}" for b in betas]
        out += [f"NORM-{t}|r{g(r)}|a1" for r in sym]
        out += [f"NORM-{t}|r{g(r)}|a{g(a)}" for r in arho for a in (0.5, 2.0)]
    return out


def parse_cid(cid):
    if cid == "U":
        return {"mode": "TASK", "treat": None}
    fam, *rest = cid.split("|")
    mode, treat = fam.split("-")
    kv = {r[0]: float(r[1:]) for r in rest}
    if mode == "RAW":
        return {"mode": "RAW", "treat": treat, "beta": kv["b"]}
    return {"mode": "NORM", "treat": treat, "rho": kv["r"], "a": kv["a"]}


def alloc(a):
    r = math.sqrt((a * a + 1.0) / 2.0)
    return a / r, 1.0 / r


def safe(cid: str) -> str:
    return cid.replace("|", "_")


def rel_name(k, cid):
    return f"rel__s{k}__{safe(cid)}"


def run_name(k, cid):
    return f"run__s{k}__{safe(cid)}"


def smf_name(k, cid):
    if cid == "U":
        return f"tl__s{k}__e40"
    c = parse_cid(cid)
    return f"raw__s{k}__RAW-{c['treat']}__b{g(c['beta'])}__e40"


# ------------------------------------------------------------------------------------------------ helpers
def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
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
    for root, ph in ((str(PRIV), "<PRIVATE_CACHE>/osf_v1"), (str(CACHE), "<PRIVATE_CACHE>"), (str(WT), "<WORKTREE>")):
        s = s.replace(root, ph)
    return s


def git(*args):
    r = subprocess.run(["git", "-C", str(WT), *args], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def git_ok(*args) -> bool:
    return subprocess.run(["git", "-C", str(WT), *args], capture_output=True, text=True).returncode == 0


def unit_done(name) -> bool:
    return (UNITS / name / "COMPLETE.json").exists()


def unit_rec(name):
    p = UNITS / name / "record.json"
    return jload(p) if p.exists() and unit_done(name) else None


def tload(p):
    """torch.load restricted to tensors / containers / primitives (no code objects)."""
    return torch.load(p, weights_only=True, map_location="cpu")


def maxdiff(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != b.shape:
        return float("inf")
    if a.size == 0:
        return 0.0
    both_nan = np.isnan(a) & np.isnan(b)
    d = np.abs(np.where(both_nan, 0.0, a - b))
    return float(np.nanmax(np.where(np.isnan(d), np.inf, d)))


def tensor_dict_equal(a: dict, b: dict):
    """(bitwise equal, max abs diff, key sets equal) for two state dicts of tensors."""
    if set(a) != set(b):
        return False, float("inf"), False
    eq, md = True, 0.0
    for k in a:
        x, y = a[k], b[k]
        if x.shape != y.shape or x.dtype != y.dtype:
            return False, float("inf"), True
        if not torch.equal(x, y):
            eq = False
            md = max(md, float((x.double() - y.double()).abs().max()))
    return eq, md, True


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
    """Independent reconstruction of the osf roles, subroles, pools and the 83-column inputs (no osf/smf/jcv loader)."""

    def __init__(self):
        self.src_sha = sha_file(SRC)
        z = np.load(SRC, allow_pickle=False)
        old = z["role"].astype(str)
        unit = z["unit"].astype(np.int64)
        rid = z["row_id"].astype(np.int64)
        X = np.asarray(z["X"])
        self.feature_names = [str(s) for s in z["feature_names"]]
        n = len(old)
        # rgj roles (seed 20261004): defense_val split 'dev' < 0.3 -> HEAD_VALIDATION; DEFENSE_FIT subroles 'critic'
        rgj = np.array([""] * n, dtype=object)
        for o, r in (("defense_train", "DEFENSE_FIT"), ("attacker_fit", "AUDIT_FIT"), ("attacker_val", "INNER_SELECTION")):
            rgj[old == o] = r
        dv = old == "defense_val"
        lab, d1 = _split(unit[dv], 20261004, "dev", [Fraction(3, 10)], ["HEAD_VALIDATION", "DEVELOPMENT_ASSESSMENT"])
        rgj[dv] = lab
        df = rgj == "DEFENSE_FIT"
        rsub = np.array([""] * n, dtype=object)
        lab, d2 = _split(unit[df], 20261004, "critic", [Fraction(7, 10), Fraction(17, 20)], ["CRITIC_FIT", "CRITIC_VAL", "CALIB"])
        rsub[df] = lab
        # smf roles (seed 20261005): rgj DEFENSE_FIT split 'assess' < 0.2 -> NEW_DEVELOPMENT_ASSESSMENT
        smf = np.array([""] * n, dtype=object)
        lab, d3 = _split(unit[df], 20261005, "assess", [Fraction(1, 5)], ["NEW_DEVELOPMENT_ASSESSMENT", "NEW_DEFENSE_FIT"])
        smf[df] = lab
        for r in ("HEAD_VALIDATION", "AUDIT_FIT", "INNER_SELECTION"):
            smf[rgj == r] = r
        ssub = np.array([""] * n, dtype=object)
        nf = smf == "NEW_DEFENSE_FIT"
        lab, d4 = _split(unit[nf], 20261005, "critic", [Fraction(7, 10), Fraction(17, 20)], ["CRITIC_FIT", "CRITIC_VAL", "CONTROLLER_CALIB"])
        ssub[nf] = lab
        self.disagree = {"rgj_dev_split": d1, "rgj_critic_split": d2, "smf_assess_split": d3, "smf_critic_split": d4}
        # osf roles: fitting roles = smf's; assessment = union of four pools minus groups touching fitting roles/exclusions
        role = np.array([""] * n, dtype=object)
        role[smf == "NEW_DEFENSE_FIT"] = FIT
        for r in ("HEAD_VALIDATION", "AUDIT_FIT", "INNER_SELECTION"):
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
        self.full = {"old": old, "rgj": rgj.astype(str), "rsub": rsub.astype(str), "smf": smf.astype(str),
                     "ssub": ssub.astype(str), "role": role.astype(str), "sub": sub.astype(str),
                     "pool": pool.astype(str), "nominal_pool": nominal.astype(str), "unit": unit, "row_id": rid}
        keep = np.flatnonzero(role != "")
        self.keep = keep
        self.row_id = rid[keep]
        self.unit = unit[keep]
        self.role = role[keep].astype(str)
        self.sub = sub[keep].astype(str)
        self.pool = pool[keep].astype(str)
        self.mask = {r: self.role == r for r in ROLES}
        self.mask.update({r: self.sub == r for r in SUBROLES})
        # numeric refit: exact integer inversion of the admitted normalisation, re-standardised on OSF_DEFENSE_FIT
        # (population sd), applied to every kept row; one-hot columns unchanged
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
            self.numeric[c] = {"inversion_max_abs_err_all_rows": float(np.abs(inv - r).max()),
                               "inversion_max_abs_err_kept_rows": float(np.abs(inv - r)[keep].max()),
                               "mean_osf_fit": m, "sd_osf_fit": s}
        self.X = np.ascontiguousarray(Xk.astype(np.float32))
        self._sealed = None
        self._unsealed = None
        self.unseal_log = []
        # fitting-row tensors for training recomputation (OSF_DEFENSE_FIT in kept order)
        self.fit_idx = np.flatnonzero(self.mask[FIT])
        pos = {int(r): j for j, r in enumerate(self.fit_idx)}
        self.cf = np.array([pos[int(i)] for i in np.flatnonzero(self.mask["CRITIC_FIT"])])
        self.cal = np.array([pos[int(i)] for i in np.flatnonzero(self.mask["DIAGNOSTIC_CALIB"])])
        self.ref = np.sort(np.random.default_rng(REF_SEED).choice(self.cf, min(REF_SIZE, len(self.cf)), replace=False))

    def labels(self, unseal=False, why=""):
        """Labels on kept rows. OSF_DEVELOPMENT_ASSESSMENT labels are masked to -1 unless unseal=True, which callers
        pass only after verifying the pushed EVALUATION_LOCK (logged)."""
        if unseal:
            if self._unsealed is None:
                self.unseal_log.append(why)
                z = np.load(SRC, allow_pickle=False)
                self._unsealed = {k: z[v].astype(np.int64)[self.keep].copy() for k, v in LABEL_KEYS.items()}
            return self._unsealed
        if self._sealed is None:
            z = np.load(SRC, allow_pickle=False)
            out = {}
            for k, v in LABEL_KEYS.items():
                a = z[v].astype(np.int64)[self.keep].copy()
                a[self.mask[ASSESS]] = -1
                out[k] = a
            self._sealed = out
        return self._sealed

    def train_tensors(self):
        """OSF_DEFENSE_FIT tensors (labels allowed for training); never touches assessment labels."""
        L = self.labels()
        fi = self.fit_idx
        S = L["sex"][fi]
        assert (S >= 0).all() and (L["y_income"][fi] >= 0).all() and (L["y_occ"][fi] >= 0).all()
        prior = np.bincount(S, minlength=2) / len(fi)
        return {"X": torch.from_numpy(self.X[fi]), "Y": {0: torch.from_numpy(L["y_income"][fi]),
                                                          1: torch.from_numpy(L["y_occ"][fi])},
                "S": torch.from_numpy(S), "n": len(fi), "prior": prior,
                "H": float(-(prior * np.log(prior)).sum()),
                "logprior": torch.tensor(np.log(prior), dtype=torch.float32), "cf": self.cf, "ref": self.ref}


def check_roles(D: Data):
    man = jload(RES / "ROLE_MANIFEST.json")
    f = D.full
    out = {"source_sha256_ok": D.src_sha == SRC_SHA, "float_vs_exact_disagreements": D.disagree,
           "pool_rows_with_a_fitting_role_before_exclusion": D.pool_overlap_with_role}

    def rec(ix):
        return {"rows": int(len(ix)), "groups": int(len(np.unique(f["unit"][ix]))),
                "row_id_sha256": rowid_hash(f["row_id"][ix]), "group_id_set_sha256": unit_set_hash(f["unit"][ix])}

    keys4 = ("rows", "groups", "row_id_sha256", "group_id_set_sha256")
    mine = {r: rec(np.flatnonzero(f["role"] == r)) for r in ROLES}
    mine.update({r: rec(np.flatnonzero(f["sub"] == r)) for r in SUBROLES})
    pub = dict(man["roles"])
    pub.update(man["defense_fit_subroles"])
    out["roles_match_manifest"] = {r: all(mine[r][k] == pub[r][k] for k in keys4) for r in mine}
    out["counts"] = {r: {"rows": mine[r]["rows"], "groups": mine[r]["groups"]} for r in mine}
    # pools
    am = f["role"] == ASSESS
    bp = man["roles"][ASSESS]["by_pool"]
    out["pools_match_manifest"] = {p: (int((am & (f["pool"] == p)).sum()) == bp[p]["rows"] and
                                       len(np.unique(f["unit"][am & (f["pool"] == p)])) == bp[p]["groups"] and
                                       rowid_hash(f["row_id"][am & (f["pool"] == p)]) == bp[p]["row_id_sha256"])
                                   for p in POOLS}
    ap = man["assessment_pools"]
    out["nominal_pools_match_manifest"] = {p: (int((f["nominal_pool"] == p).sum()) == ap[p]["nominal_rows"] and
                                               rowid_hash(f["row_id"][f["nominal_pool"] == p]) == ap[p]["nominal_row_id_sha256"]
                                               and D.pool_excluded_rows[p] == ap[p]["excluded_rows_group_overlap"])
                                           for p in POOLS}
    out["pool_counts"] = {p: {"nominal": int((f["nominal_pool"] == p).sum()), "kept": int((am & (f["pool"] == p)).sum()),
                              "excluded_group_overlap": D.pool_excluded_rows[p]} for p in POOLS}
    gs = {p: set(np.unique(f["unit"][f["pool"] == p]).tolist()) for p in POOLS}
    out["groups_shared_between_pools"] = {f"{a}&{b}": len(gs[a] & gs[b]) for i, a in enumerate(POOLS) for b in POOLS[i + 1:]}
    out["groups_shared_match_manifest"] = out["groups_shared_between_pools"] == man["groups_shared_between_pools"]
    # inherited roles (rgj and smf) and old roles
    inh = man["inherited_roles_recomputed"]
    rg = {r: rec(np.flatnonzero(f["rgj"] == r)) for r in ("DEFENSE_FIT", "HEAD_VALIDATION", "DEVELOPMENT_ASSESSMENT",
                                                           "AUDIT_FIT", "INNER_SELECTION")}
    rg.update({r: rec(np.flatnonzero(f["rsub"] == r)) for r in ("CRITIC_FIT", "CRITIC_VAL", "CALIB")})
    out["rgj_roles_match_manifest"] = {r: all(rg[r][k] == inh["rgj"]["per_role"][r][k] for k in keys4) for r in rg}
    sm = {r: rec(np.flatnonzero(f["smf"] == r)) for r in ("NEW_DEFENSE_FIT", "NEW_DEVELOPMENT_ASSESSMENT",
                                                          "HEAD_VALIDATION", "AUDIT_FIT", "INNER_SELECTION")}
    sm.update({r: rec(np.flatnonzero(f["ssub"] == r)) for r in ("CRITIC_FIT", "CRITIC_VAL", "CONTROLLER_CALIB")})
    smp = inh["smf"]["per_role"]
    out["smf_roles_match_manifest"] = {r: all(sm[r][k] == smp[r][k] for k in keys4) for r in sm if r in smp}
    old = man["old_roles"]["per_role"]
    out["old_roles_match_manifest"] = {r: all(rec(np.flatnonzero(f["old"] == r))[k] == old[r][k] for k in keys4)
                                       for r in old}
    fp = hashlib.sha256("|".join(f"{r}:{mine[r]['row_id_sha256']}" for r in ROLES + SUBROLES).encode()).hexdigest()
    out["fingerprint_match"] = fp == man.get("role_partition_fingerprint_sha256")
    # disjointness and partitions
    groups = {r: set(np.unique(f["unit"][f["role"] == r]).tolist()) for r in ROLES}
    out["roles_group_disjoint"] = all(not (groups[a] & groups[b]) for i, a in enumerate(ROLES) for b in ROLES[i + 1:])
    out["subroles_partition_fit"] = bool(np.array_equal(np.sort(np.flatnonzero(np.isin(f["sub"], SUBROLES))),
                                                        np.flatnonzero(f["role"] == FIT)))
    excl = np.isin(f["old"], EXCLUSIONS)
    out["no_exclusion_row_kept"] = bool((f["role"][excl] == "").all())
    kept_groups = set(np.unique(f["unit"][f["role"] != ""]).tolist())
    dropped_groups = set(np.unique(f["unit"][f["role"] == ""]).tolist())
    out["no_kept_group_in_a_dropped_row"] = not (kept_groups & dropped_groups)
    out["dropped_rows"] = int((f["role"] == "").sum())
    out["dropped_rows_by_old_role"] = {r: int(((f["role"] == "") & (f["old"] == r)).sum())
                                       for r in np.unique(f["old"][f["role"] == ""]).tolist()}
    out["assessment_from_named_pools_only"] = bool(np.isin(f["old"][am], ("assessment", "cert", "defense_val",
                                                                          "defense_train")).all())
    # numeric refit
    pc = man["numeric_refit"]["per_column"]
    out["numeric_refit_match"] = {c: bool(D.numeric[c]["mean_osf_fit"] == pc[c]["mean_osf_fit"]
                                          and D.numeric[c]["sd_osf_fit"] == pc[c]["sd_osf_fit"]
                                          and D.numeric[c]["inversion_max_abs_err_all_rows"] < 0.05
                                          and D.numeric[c]["inversion_max_abs_err_all_rows"]
                                          == pc[c]["inversion_max_abs_err_all_rows"]) for c in NUMERIC}
    out["feature_names_sha256_match"] = hashlib.sha256("\n".join(D.feature_names).encode()).hexdigest() == \
        man["permitted_columns"]["feature_names_sha256"]
    low = [s.lower().split("=")[0] for s in D.feature_names]
    out["n_columns"] = len(D.feature_names)
    out["forbidden_columns_present"] = sorted({s for s in low if s in ("sex", "race", "income", "occupation", "fnlwgt")})
    out["husband_wife_proxies_present"] = any("husband" in s.lower() for s in D.feature_names) and \
        any("wife" in s.lower() for s in D.feature_names)
    ok = (out["source_sha256_ok"] and all(out["roles_match_manifest"].values()) and all(out["pools_match_manifest"].values())
          and all(out["nominal_pools_match_manifest"].values()) and out["groups_shared_match_manifest"]
          and all(out["rgj_roles_match_manifest"].values()) and all(out["smf_roles_match_manifest"].values())
          and all(out["old_roles_match_manifest"].values()) and out["fingerprint_match"] and out["roles_group_disjoint"]
          and out["subroles_partition_fit"] and out["no_exclusion_row_kept"] and out["no_kept_group_in_a_dropped_row"]
          and out["assessment_from_named_pools_only"] and all(out["numeric_refit_match"].values())
          and out["feature_names_sha256_match"] and out["n_columns"] == 83 and not out["forbidden_columns_present"]
          and sum(D.disagree.values()) == 0 and D.pool_overlap_with_role == 0)
    return res("PASS" if ok else "FAIL", **out)


# ------------------------------------------------------------------------------------------------ own forward pass
ENC_KEYS = [f"{j}.{w}" for j in (0, 2, 4) for w in ("weight", "bias")]


def encode(sd, i, X):
    """Recipient i encoder: Linear(83,64)-ReLU-Linear(64,64)-ReLU-Linear(64,16), float32 as stored -> float64."""
    h = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    with torch.no_grad():
        for j, act in ((0, True), (2, True), (4, False)):
            h = F.linear(h, sd[f"enc.{i}.{j}.weight"], sd[f"enc.{i}.{j}.bias"])
            if act:
                h = torch.relu(h)
    return h.double().numpy()


def centred_logits(head, r):
    d = np.asarray(head.decision_function(r), dtype=np.float64)
    if d.ndim == 1:
        d = np.stack([np.zeros_like(d), d], 1)
    return d - d.mean(1, keepdims=True)


def outputs_of(head, r):
    P = head.predict_proba(r)
    return centred_logits(head, r), P, P.argmax(1)


def my_fit_head(R, y, tr, va, K):
    """Own deployed-head procedure: StandardScaler + LogisticRegression(C, max_iter 3000) on OSF_DEFENSE_FIT, C by
    HEAD_VALIDATION log loss; a later (larger) C replaces the incumbent only on strict improvement > 1e-12."""
    best, table = None, []
    for C in HEAD_C:
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
        m.fit(R[tr], y[tr])
        ll = float(log_loss(y[va], m.predict_proba(R[va]), labels=list(range(K))))
        table.append({"C": C, "log_loss": ll})
        if best is None or ll < best[0] - 1e-12:
            best = (ll, C, m)
    return best[2], best[1], table


def load_state(ud: Path):
    p = ud / "model.pt"
    if not p.exists():
        return None
    sd = tload(p)
    return sd if isinstance(sd, dict) and "enc.0.0.weight" in sd else None


def replay_release(D: Data, name: str, L, refit=True):
    ud = UNITS / name
    rec = unit_rec(name) or {}
    out = {"unit": name}
    rel = np.load(ud / "release.npz", allow_pickle=False)
    keys = set(rel.files)
    out["keys_ok"] = keys == RELEASE_KEYS
    out["label_like_keys"] = sorted(k for k in keys if k in ("sex", "race", "y_income", "y_occ", "y", "labels"))
    out["row_id_equals_kept_rows"] = bool(np.array_equal(rel["row_id"], D.row_id))
    fails = []
    if not out["keys_ok"] or out["label_like_keys"] or not out["row_id_equals_kept_rows"]:
        fails.append("rows/keys")
    sd = load_state(ud)
    if sd is None:
        fails.append("model.pt not a state dict")
        return res("FAIL", failures=fails, **out), rel
    out["state_keys"] = sorted(sd)
    exp_keys = {f"enc.{i}.{k}" for i in (0, 1) for k in ENC_KEYS} | {f"head.{i}.{w}" for i in (0, 1) for w in ("weight", "bias")}
    out["state_keys_ok"] = set(sd) == exp_keys and tuple(sd["enc.0.0.weight"].shape) == (64, 83) and \
        tuple(sd["enc.0.4.weight"].shape) == (16, 64)
    if not out["state_keys_ok"]:
        fails.append("state dict keys/shapes")
    tr, va = np.flatnonzero(D.mask[FIT]), np.flatnonzero(D.mask["HEAD_VALIDATION"])
    for i, (rk, ck, pk, hk) in enumerate((("r1", "c1", "p1", "hard1"), ("r2", "c2", "p2", "hard2"))):
        info = {}
        r = encode(sd, i, D.X)
        info["features_bit_exact"] = bool(np.array_equal(r, rel[rk]))
        info["features_max_abs_diff"] = maxdiff(r, rel[rk])
        if not info["features_bit_exact"]:
            fails.append(f"{rk} features")
        h = joblib.load(ud / f"head_{i}.joblib")
        c, p, hard = outputs_of(h, r)
        info["saved_head_outputs_bit_exact"] = bool(np.array_equal(c, rel[ck]) and np.array_equal(p, rel[pk]) and
                                                    np.array_equal(hard, rel[hk]))
        info["saved_head_c_max_abs_diff"] = maxdiff(c, rel[ck])
        sc = h[0]
        info["saved_scaler_n_seen"] = int(np.asarray(sc.n_samples_seen_).max())
        info["saved_scaler_fit_rows_are_OSF_DEFENSE_FIT"] = bool(info["saved_scaler_n_seen"] == len(tr) and
                                                                 np.array_equal(sc.mean_, StandardScaler().fit(rel[rk][tr]).mean_))
        if not info["saved_head_outputs_bit_exact"]:
            fails.append(f"saved head {i} outputs")
        if not info["saved_scaler_fit_rows_are_OSF_DEFENSE_FIT"]:
            fails.append(f"head {i} fit rows")
        if refit:
            y = L["y_income"] if i == 0 else L["y_occ"]
            assert (y[tr] >= 0).all() and (y[va] >= 0).all()
            mh, C, table = my_fit_head(r, y, tr, va, KS[i])
            mc, mp, mhard = outputs_of(mh, r)
            info["refit_C"] = C
            info["recorded_C"] = (rec.get("heads") or {}).get(str(i), {}).get("selected_C")
            info["saved_head_C"] = float(h[-1].C)
            rt = {float(t["C"]): t["defense_val_log_loss"] for t in (rec.get("heads") or {}).get(str(i), {}).get("table", [])}
            info["refit_table_max_abs_diff_vs_record"] = max((abs(t["log_loss"] - rt.get(t["C"], np.inf)) for t in table),
                                                              default=None)
            info["refit_coef_bit_exact"] = bool(np.array_equal(mh[-1].coef_, h[-1].coef_) and
                                                np.array_equal(mh[-1].intercept_, h[-1].intercept_))
            info["refit_c_max_abs_diff"] = maxdiff(mc, rel[ck])
            info["refit_p_max_abs_diff"] = maxdiff(mp, rel[pk])
            info["refit_hard_mismatches"] = int((mhard != rel[hk]).sum())
            info["refit_outputs_bit_exact"] = bool(np.array_equal(mc, rel[ck]) and np.array_equal(mp, rel[pk]) and
                                                   np.array_equal(mhard, rel[hk]))
            # declared tolerance when not bit-exact: centred logits 1e-6, probabilities 1e-8, no decision flip
            tol_ok = (info["refit_c_max_abs_diff"] <= 1e-6 and info["refit_p_max_abs_diff"] <= 1e-8
                      and info["refit_hard_mismatches"] == 0)
            if not (C == info["recorded_C"] == info["saved_head_C"]) or not tol_ok or \
                    (info["refit_table_max_abs_diff_vs_record"] or 0) > 1e-9:
                fails.append(f"head {i} refit")
        out[f"recipient_{i + 1}"] = info
    out["failures"] = fails
    out["status"] = "FAIL" if fails else "PASS"
    return out, rel


def admitted_release_equal(D: Data, name: str, rel, cid, k):
    """Admitted rel unit: model.pt tensors equal the admitted smf e40 model; release equals the smf release on every
    smf row (own row-id alignment)."""
    src = ADMITTED / smf_name(k, cid)
    out = {"admitted_from": smf_name(k, cid)}
    a = tload(src / "model.pt")
    b = tload(UNITS / name / "model.pt")
    eq, md, _ = tensor_dict_equal(a, b)
    out["model_equals_admitted"] = eq
    z = np.load(src / "release.npz", allow_pickle=False)
    pos = {int(r): j for j, r in enumerate(D.row_id)}
    ix = np.array([pos[int(r)] for r in z["row_id"]])
    out["smf_rows"] = int(len(ix))
    out["release_equal_on_smf_rows"] = {kk: bool(np.array_equal(rel[kk][ix], z[kk])) for kk in z.files if kk != "row_id"}
    out["status"] = "PASS" if eq and all(out["release_equal_on_smf_rows"].values()) else "FAIL"
    return out


# ------------------------------------------------------------------------------------------------ receipts
def fp64(arr) -> int:
    """Own minibatch fingerprint: sha256 of the int64 index bytes, first 8 bytes little-endian, >> 1."""
    return int.from_bytes(hashlib.sha256(np.ascontiguousarray(arr).tobytes()).digest()[:8], "little") >> 1


def my_batches(seed, n, salt=0):
    out = []
    for ep in range(EPOCHS):
        perm = np.random.default_rng([seed, salt, ep]).permutation(n)
        for s in range(0, n, BATCH):
            out.append((ep, perm[s:s + BATCH]))
    return out


def stats_of(x):
    x = np.asarray(x, dtype=np.float64)
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "mean": float(x.mean()), "rms": float(np.sqrt(np.mean(x * x))),
            "median": float(np.percentile(x, 50)), "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90))}


def _close_rel(a, b, rtol, atol=0.0):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return np.abs(a - b) <= atol + rtol * np.maximum(np.abs(a), np.abs(b))


def check_steps(name, rec, z, n_steps_expected):
    """Per-step identities of one run's receipts (no RNG, no labels)."""
    d = rec["diag"]
    c = parse_cid(rec["config"])
    mode = c["mode"]
    fails, info = [], {"config": rec["config"], "mode": mode}
    S = len(z["step"])
    per_ep = n_steps_expected // EPOCHS
    if S != n_steps_expected:
        fails.append(f"steps {S} != {n_steps_expected}")
    if not np.array_equal(z["step"], np.arange(1, S + 1)) or not np.array_equal(z["epoch"], np.arange(S) // per_ep):
        fails.append("step/epoch indexing")
    zr, cap = z["zero"].astype(int), z["cap"].astype(bool)
    tn, qn, pn, ratio, real = z["t_norm"], z["q_norm"], z["p_norm"], z["ratio"], z["realized_ratio"]
    R, applied = z["R"], z["applied"].astype(bool)
    info["applied"] = int(applied.sum())
    if not applied.all():
        fails.append(f"{int((~applied).sum())} steps not applied")
    nonzero = zr == 0
    # --- mode-specific identities
    if mode == "TASK":
        ok = (zr == 5).all() and (qn == 0).all() and np.isnan(R).all() and (z["critic_updates"] == 0).all() \
            and np.isnan(pn).all() and np.isnan(ratio).all() and (real == 0).all()
        if not ok:
            fails.append("TASK receipts: zero code 5, q = 0, no R, no critic updates")
        info["zero_codes"] = {str(v): int((zr == v).sum()) for v in np.unique(zr)}
    else:
        if (zr == 5).any():
            fails.append("protected arm has 'off' zero code")
        exp_cu = (CRITIC_STEPS * 6) * np.arange(1, S + 1)
        info["critic_updates_5x6_per_step"] = bool(np.array_equal(z["critic_updates"], exp_cu))
        if not info["critic_updates_5x6_per_step"]:
            fails.append("critic update counts")
        local = c["treat"] == "L"
        act = [0, 1] if local else [0, 1, 2]
        info["R_pair_all_nan"] = bool(np.isnan(R[:, 2]).all())
        if local and not info["R_pair_all_nan"]:
            fails.append("local arm has a pair R value")
        if not local and np.isnan(R[:, 2]).any():
            fails.append("joint arm missing pair R")
        if np.isnan(R[:, act]).any() or (R[:, act] < 0).any():
            fails.append("R missing or negative on an active view")
        allzero = (R[:, act] == 0).all(1)
        info["no_gradient_path_steps"] = int(allzero.sum())
        if not np.array_equal(allzero, (zr == 1).all(1)) or ((zr == 1).any(1) != (zr == 1).all(1)).any():
            fails.append("zero code 1 != constant wins every active view")
        # p_i = 0 rule from the views that reach encoder i: v_i (+ pair in joint arms). Local arms: pair never counts.
        for i in (0, 1):
            reach = (R[:, i] == 0) & ((R[:, 2] == 0) if not local else True)
            expect2 = reach & ~allzero
            got2 = zr[:, i] == 2
            if mode == "RAW" and not np.array_equal(expect2, got2):
                fails.append(f"enc{i + 1}: zero code 2 != (no reaching view gradient) [{int(expect2.sum())} vs {int(got2.sum())}]")
            if mode == "NORM" and not np.array_equal(expect2 & (tn[:, i] > 0), got2):
                fails.append(f"enc{i + 1}: zero code 2 (NORM) != (no reaching view gradient)")
        if local:
            # direct test of 'no pair encoder gradient': steps where v_i lost to the constant but the pair critic did not
            pair_only = [int(((R[:, i] == 0) & (R[:, 2] > 0) & ~allzero).sum()) for i in (0, 1)]
            info["local_steps_with_pair_signal_but_no_own_view"] = pair_only
            info["local_those_steps_have_p_i_zero"] = [bool((zr[(R[:, i] == 0) & (R[:, 2] > 0) & ~allzero, i] == 2).all())
                                                       for i in (0, 1)]
            if not all(info["local_those_steps_have_p_i_zero"]):
                fails.append("local arm: pair signal reached an encoder")
        if (zr == 4).any():
            fails.append("nonfinite steps present")
        if mode == "RAW":
            b = c["beta"]
            m = nonzero
            ok = (np.array_equal(ratio[m], qn[m] / tn[m]) and np.array_equal(real[m], ratio[m]) and
                  (z["scale"][m] == b).all() and np.array_equal(pn[m], qn[m] / b) and not cap.any())
            if not ok:
                fails.append("RAW: ratio = q/t, realized = ratio, scale = beta, p = q/beta, no cap")
            zm = ~nonzero
            if not ((qn[zm] == 0).all() and np.isnan(ratio[zm]).all() and (real[zm] == 0).all()):
                fails.append("RAW zero steps: q = 0, ratio NaN, realized 0")
        else:
            s = alloc(c["a"])
            for i in (0, 1):
                tgt = c["rho"] * s[i]
                m = nonzero[:, i] & ~cap[:, i]
                mc = nonzero[:, i] & cap[:, i]
                r_ok = _close_rel(ratio[m, i], tgt, 1e-6).all()
                a_ok = _close_rel(z["scale"][m, i] * pn[m, i] / tn[m, i], tgt, 1e-12).all()
                q_ok = _close_rel(qn[m, i], z["scale"][m, i] * pn[m, i], 1e-6).all()
                cap_ok = ((z["scale"][mc, i] == A_MAX).all() and (tgt * tn[mc, i] / pn[mc, i] > A_MAX).all()
                          and (ratio[mc, i] < tgt).all())
                uncap_ok = (tgt * tn[m, i] / pn[m, i] <= A_MAX * (1 + 1e-12)).all()
                info[f"enc{i + 1}_target_rho_s"] = tgt
                info[f"enc{i + 1}_max_rel_dev_ratio_vs_rho_s_uncapped"] = float(np.max(np.abs(ratio[m, i] / tgt - 1))) if m.any() else None
                info[f"enc{i + 1}_cap_steps"] = int(mc.sum())
                if not (r_ok and a_ok and q_ok and cap_ok and uncap_ok):
                    fails.append(f"NORM enc{i + 1}: ratio = rho s_i (uncapped), scale p/t = rho s_i, cap semantics")
                zm = ~nonzero[:, i]
                if not ((qn[zm, i] == 0).all() and (real[zm, i] == 0).all()):
                    fails.append(f"NORM enc{i + 1}: zero steps carry protection")
                z2 = zr[:, i] == 2
                if not (pn[z2, i] <= ZERO_TOL).all():
                    fails.append(f"NORM enc{i + 1}: zero code 2 with ||p|| > tol")
                z3 = zr[:, i] == 3
                if not (tn[z3, i] == 0).all():
                    fails.append(f"NORM enc{i + 1}: zero code 3 with ||t|| > 0")
            both = nonzero.all(1) & ~cap.any(1)
            rms = np.sqrt((ratio[both, 0] ** 2 + ratio[both, 1] ** 2) / 2)
            info["rms_identity_steps"] = int(both.sum())
            info["rms_identity_max_rel_dev"] = float(np.max(np.abs(rms / c["rho"] - 1))) if both.any() else None
            if both.any() and info["rms_identity_max_rel_dev"] > 1e-6:
                fails.append("RMS identity")
        info["zero_codes"] = {f"enc{i + 1}": {str(v): int((zr[:, i] == v).sum()) for v in np.unique(zr[:, i])} for i in (0, 1)}
        info["cap_steps"] = [int(cap[:, i].sum()) for i in (0, 1)]
    # --- common identities
    tt2 = tn[:, 0] * tn[:, 0] + tn[:, 1] * tn[:, 1]
    tq2 = qn[:, 0] * qn[:, 0] + qn[:, 1] * qn[:, 1]
    comb = np.where(tt2 > 0, np.sqrt(np.where(tt2 > 0, tq2 / np.where(tt2 > 0, tt2, 1), np.nan)), np.nan)
    info["combined_ratio_identity_max_abs"] = maxdiff(comb, z["comb_ratio"])
    if info["combined_ratio_identity_max_abs"] > 1e-12:
        fails.append("combined ratio != sqrt(sum q^2 / sum t^2)")
    rmsr = np.sqrt(np.mean(np.square(real), axis=1))
    info["rms_ratio_identity_max_abs"] = maxdiff(rmsr, z["rms_ratio"])
    if info["rms_ratio_identity_max_abs"] > 1e-12:
        fails.append("rms_ratio != sqrt(mean realized^2)")
    pre, post, kap = z["pre_total"], z["post_total"], z["kappa"]
    kexp = np.where(pre > CLIP, CLIP / pre, 1.0)
    info["clip_factor_identity"] = bool(np.array_equal(kap, kexp))
    info["post_total_max_rel_dev_vs_min_pre_5"] = float(np.max(np.abs(post / np.minimum(pre, CLIP) - 1)))
    info["post_enc_max_rel_dev_vs_kappa_pre_enc"] = float(np.max(np.abs(z["post_enc"] - kap[:, None] * z["pre_enc"])
                                                                / np.maximum(z["pre_enc"], 1e-30)))
    tot2 = z["pre_enc"][:, 0] ** 2 + z["pre_enc"][:, 1] ** 2 + z["pre_head"] ** 2
    info["pre_total_sq_max_rel_dev_vs_blocks"] = float(np.max(np.abs(pre ** 2 - tot2) / tot2))
    # |t_i + q_i|^2 = t^2 + q^2 + 2 cos t q (sign of the protection direction; q added to the task gradient)
    cosv = np.where(np.isnan(z["cos"]), 0.0, z["cos"])
    alg = tn ** 2 + qn ** 2 + 2 * cosv * tn * qn
    info["update_norm_algebra_max_rel_dev"] = float(np.max(np.abs(z["pre_enc"] ** 2 - alg) / (tn ** 2 + qn ** 2)))
    if not info["clip_factor_identity"] or info["post_total_max_rel_dev_vs_min_pre_5"] > 1e-5 or \
            info["post_enc_max_rel_dev_vs_kappa_pre_enc"] > 1e-5 or info["pre_total_sq_max_rel_dev_vs_blocks"] > 1e-5 or \
            info["update_norm_algebra_max_rel_dev"] > 1e-4:
        fails.append("clip / update-norm identities")
    if mode != "TASK" and np.isnan(z["cos"][nonzero]).any():
        fails.append("cos missing on a nonzero step")
    info["clip_steps"] = int((kap < 1).sum())
    # --- diag totals and epoch summaries
    zev = [int(((zr[:, i] != 0) & (zr[:, i] != 5)).sum()) for i in (0, 1)]
    tot_ok = (d["encoder_updates"] == int(applied.sum()) and d["clip_hits"] == info["clip_steps"]
              and list(d["zero_events"]) == zev and list(d["cap_hits"]) == [int(cap[:, i].sum()) for i in (0, 1)]
              and d["critic_online_updates"] == (int(z["critic_updates"][-1]) if mode != "TASK" else 0)
              and d["nonfinite"] == 0)
    if not tot_ok:
        fails.append("diag totals != receipt sums")
    ep_bad = 0
    for E in d["epochs"]:
        e = E["epoch"]
        sl = slice(e * per_ep, (e + 1) * per_ep)
        want = {"clip": int((kap[sl] < 1).sum()), "rms_ratio_mean": float(z["rms_ratio"][sl].mean())}
        for i in (0, 1):
            zz = zr[sl, i]
            want[f"realized_ratio_mean_{i + 1}"] = float(real[sl, i].mean())
            want[f"zero_{i + 1}"] = int(((zz != 0) & (zz != 5)).sum())
            want[f"cap_{i + 1}"] = int(cap[sl, i].sum())
            want[f"t_norm_mean_{i + 1}"] = float(tn[sl, i].mean())
            want[f"q_norm_mean_{i + 1}"] = float(qn[sl, i].mean())
        for kk, v in want.items():
            if E.get(kk) is None or abs(E[kk] - v) > 1e-12 * max(1.0, abs(v)):
                ep_bad += 1
    if ep_bad or len(d["epochs"]) != EPOCHS:
        fails.append(f"epoch summaries ({ep_bad} mismatching values)")
    sm = d["summary"]
    smb = 0
    for i in (0, 1):
        mine = stats_of(real[:, i])
        for kk in ("n", "mean", "rms", "median", "p10", "p90"):
            if abs(sm[f"enc{i + 1}"]["realized_ratio"][kk] - mine[kk]) > 1e-12 * max(1.0, abs(mine[kk])):
                smb += 1
        zf = float(((zr[:, i] != 0) & (zr[:, i] != 5)).mean())
        if abs(sm[f"enc{i + 1}"]["zero_fraction"] - zf) > 1e-15:
            smb += 1
    cr = z["comb_ratio"][np.isfinite(z["comb_ratio"])]
    mine = stats_of(cr)
    for kk in ("n", "mean", "rms", "median", "p10", "p90"):
        if kk in sm["combined_ratio"] and abs(sm["combined_ratio"][kk] - mine[kk]) > 1e-12 * max(1.0, abs(mine[kk])):
            smb += 1
    if smb or sm["enc1"]["realized_ratio"]["n"] != S:
        fails.append(f"run summary ({smb} mismatching values; realized statistics must count zero steps)")
    info["realized_ratio_mean"] = [float(real[:, i].mean()) for i in (0, 1)]
    info["realized_ratio_rms"] = [float(np.sqrt(np.mean(real[:, i] ** 2))) for i in (0, 1)]
    info["combined_ratio_mean"] = float(np.nanmean(z["comb_ratio"])) if np.isfinite(z["comb_ratio"]).any() else None
    info["zero_fraction"] = [zev[i] / S for i in (0, 1)]
    info["clip_fraction"] = info["clip_steps"] / S
    info["failures"] = fails[:20]
    info["status"] = "FAIL" if fails else "PASS"
    return info


def strength_reports_check(seeds):
    """Recompute the lead's STRENGTH_PROFILES.csv (run and epoch scopes) and STRENGTH_COMPARISON.json (pooled over
    seeds; early = epochs 0-4, late = epochs 30-39) from the raw receipts (CSV values carry 6 significant digits)."""
    p = RES / "STRENGTH_PROFILES.csv"
    if not p.exists():
        return res("PENDING", reason="STRENGTH_PROFILES.csv not written")
    cache = {}

    def arrays(cid, k):
        if (cid, k) not in cache:
            rn = run_name(k, cid)
            if not unit_done(rn):
                cache[(cid, k)] = None
            else:
                z = np.load(UNITS / rn / "steps.npz", allow_pickle=False)
                cache[(cid, k)] = {f: z[f] for f in z.files}
        return cache[(cid, k)]

    def run_quantity(z, q, stat):
        zr = z["zero"]
        m = re.match(r"(.*)_enc(\d)(.*)", q)
        if q == "rms_ratio_realized":
            x = z["rms_ratio"]
        elif q == "combined_ratio":
            x = z["comb_ratio"][np.isfinite(z["comb_ratio"])]
        elif q == "clip_fraction":
            return float((z["kappa"] < 1).mean())
        elif m:
            base, i = m.group(1), int(m.group(2)) - 1
            if base == "zero_fraction":
                return float(((zr[:, i] != 0) & (zr[:, i] != 5)).mean())
            if base == "cap_fraction":
                return float(z["cap"][:, i].mean())
            x = {"realized_ratio": z["realized_ratio"][:, i], "t_norm": z["t_norm"][:, i], "q_norm": z["q_norm"][:, i],
                 "post_clip_update_norm": z["post_enc"][:, i], "cos_t_p": z["cos"][:, i][np.isfinite(z["cos"][:, i])],
                 "conditional_ratio": z["ratio"][:, i][np.isfinite(z["ratio"][:, i])]}.get(base)
            if x is None:
                return None
        else:
            return None
        st = stats_of(x)
        return st.get(stat) if stat != "value" else None

    def epoch_quantity(z, e, q):
        sl = z["epoch"] == e
        if q == "clip_fraction":
            return float((z["kappa"][sl] < 1).mean())
        if q == "combined_ratio":
            c = z["comb_ratio"][sl]
            return float(np.nanmean(c)) if np.isfinite(c).any() else None
        m = re.match(r"realized_ratio_enc(\d)", q)
        return float(z["realized_ratio"][sl, int(m.group(1)) - 1].mean()) if m else None

    bad, n, skipped = [], 0, 0
    for r in csv.DictReader(open(p)):
        k = int(r["seed"])
        if k not in seeds:
            continue
        z = arrays(r["config"], k)
        if z is None:
            skipped += 1
            continue
        if r["scope"] == "run":
            v = run_quantity(z, r["quantity"], r["stat"])
        else:
            v = epoch_quantity(z, int(r["scope"].split("_")[1]), r["quantity"])
        if v is None:
            skipped += 1
            continue
        n += 1
        x = float(r["value"])
        if abs(x - v) > 1e-5 * max(abs(x), abs(v)) + 1e-12:
            bad.append(f"{r['config']} s{k} {r['scope']} {r['quantity']} {r['stat']}: {x} vs {v:.8g}")
    out = {"STRENGTH_PROFILES.csv": res("FAIL" if bad else "PASS", rows_compared=n, rows_not_recomputed=skipped,
                                        differences=bad[:20], n_differences=len(bad))}
    q = RES / "STRENGTH_COMPARISON.json"
    if q.exists() and tuple(seeds) == SEEDS:
        SC = jload(q)
        bad2, n2 = [], 0
        for cid, t in SC["configs"].items():
            zs = [arrays(cid, k) for k in SEEDS]
            if any(z is None for z in zs):
                continue
            cat = lambda f: np.concatenate([z[f] for z in zs])      # noqa: E731
            ep = cat("epoch")
            mine = {}
            for i in (0, 1):
                x = np.concatenate([z["realized_ratio"][:, i] for z in zs])
                mine[f"enc{i + 1}"] = {**stats_of(x), "early_mean": float(x[ep <= 4].mean()),
                                       "late_mean": float(x[ep >= 30].mean()),
                                       "cv": float(x.std() / x.mean()) if x.mean() > 0 else None}
            c, rr = cat("comb_ratio"), cat("rms_ratio")
            mine["combined"] = {**stats_of(c[np.isfinite(c)]), "early_mean": float(np.nanmean(c[ep <= 4])),
                                "late_mean": float(np.nanmean(c[ep >= 30]))}
            mine["rms"] = {**stats_of(rr), "early_mean": float(rr[ep <= 4].mean()), "late_mean": float(rr[ep >= 30].mean())}
            mine["clip_fraction"] = float((cat("kappa") < 1).mean())
            for blk, d in mine.items():
                if isinstance(d, dict):
                    for kk, v in d.items():
                        tv = (t.get(blk) or {}).get(kk)
                        if tv is None or v is None:
                            continue
                        n2 += 1
                        if abs(tv - v) > 1e-9 * max(1.0, abs(v)):
                            bad2.append(f"{cid} {blk}.{kk}: {tv} vs {v}")
                else:
                    n2 += 1
                    if abs(t.get(blk, np.nan) - d) > 1e-12:
                        bad2.append(f"{cid} {blk}")
        out["STRENGTH_COMPARISON.json"] = res("FAIL" if bad2 else "PASS", values_compared=n2, differences=bad2[:20])
    return res(worst(*[v["status"] for v in out.values()]), **out)


def receipt_power(n_fit):
    """Mutation check on real receipts: each injected inconsistency must make check_steps FAIL."""
    out = {}
    cases = [("run__s0__NORM-J_r3_a1", "ratio_off_1e-4", lambda z: z["ratio"].__setitem__((100, 0), z["ratio"][100, 0] * (1 + 1e-4))),
             ("run__s0__NORM-J_r3_a1", "q_norm_off_1e-4", lambda z: z["q_norm"].__setitem__((50, 1), z["q_norm"][50, 1] * (1 + 1e-4))),
             ("run__s0__NORM-L_r3_a1", "pair_R_in_local_arm", lambda z: z["R"].__setitem__((7, 2), 0.1)),
             ("run__s0__RAW-J_b0.3", "scale_not_beta", lambda z: z["scale"].__setitem__((9, 0), 0.31)),
             ("run__s0__RAW-J_b0.3", "zero_code_flip", lambda z: z["zero"].__setitem__((11, 1), 2)),
             ("run__s0__RAW-J_b0.3", "kappa_off", lambda z: z["kappa"].__setitem__(5, 0.99)),
             ("run__s0__RAW-L_b0.3", "critic_updates_off", lambda z: z["critic_updates"].__setitem__(30, z["critic_updates"][30] - 1)),
             ("run__s0__NORM-J_r5_a2", "post_total_unclipped", lambda z: z["post_total"].__setitem__(
                 int(np.argmax(z["pre_total"])), z["pre_total"][int(np.argmax(z["pre_total"]))]))]
    S = EPOCHS * math.ceil(n_fit / BATCH)
    for rn, nm, mut in cases:
        if not unit_done(rn):
            out[nm] = "PENDING"
            continue
        zz = np.load(UNITS / rn / "steps.npz", allow_pickle=False)
        z = {k: zz[k].copy() for k in zz.files}
        base = check_steps(rn, unit_rec(rn), z, S)["status"]
        mut(z)
        out[nm] = {"clean": base, "mutated": check_steps(rn, unit_rec(rn), z, S)["status"]}
    caught = [k for k, v in out.items() if isinstance(v, dict) and v["clean"] == "PASS" and v["mutated"] == "FAIL"]
    return res("PASS" if len(caught) == len(cases) else ("PENDING" if any(v == "PENDING" for v in out.values()) else "FAIL"),
               mutations=len(cases), caught=len(caught), per_mutation=out)


def check_receipts_and_rng(inv, seeds, n_fit):
    runs = sorted(n for n in inv if n.startswith("run__") and unit_done(n) and "__nonfinite" not in n
                  and int(n.split("__")[1][1:]) in seeds)
    if not runs:
        return res("PENDING", reason="no run receipts"), res("PENDING")
    S = EPOCHS * math.ceil(n_fit / BATCH)
    per, cache = {}, {}
    for rn in runs:
        rec = unit_rec(rn)
        z = np.load(UNITS / rn / "steps.npz", allow_pickle=False)
        cache[rn] = {k: z[k] for k in ("mb_fp", "tf_fp", "t_norm", "p_norm", "R", "zero")}
        cache[rn]["config"], cache[rn]["seed"] = rec["config"], rec["seed"]
        per[rn] = check_steps(rn, rec, z, S)
    # common RNG and step-1 identity across configurations of a seed
    rng = {}
    for k in seeds:
        rs = [rn for rn in runs if cache[rn]["seed"] == k]
        if not rs:
            continue
        mine = np.array([fp64(b.astype(np.int64)) for _, b in my_batches(k, n_fit)], dtype=np.int64)
        same_fp = all(np.array_equal(cache[rn]["mb_fp"], mine) for rn in rs)
        t1 = {rn: cache[rn]["t_norm"][0].tolist() for rn in rs}
        prot = [rn for rn in rs if cache[rn]["config"] != "U"]
        tf1 = {cache[rn]["tf_fp"][0] for rn in prot}
        joint = [rn for rn in prot if "-J" in cache[rn]["config"]]
        local = [rn for rn in prot if "-L" in cache[rn]["config"]]
        R1j = {tuple(cache[rn]["R"][0].tolist()) for rn in joint}
        R1l = {tuple(cache[rn]["R"][0, :2].tolist()) for rn in local}
        R1_shadow = {tuple(cache[rn]["R"][0, :2].tolist()) for rn in prot}
        p1j = np.array([cache[rn]["p_norm"][0] for rn in joint if (cache[rn]["zero"][0] == 0).all()])
        p1l = np.array([cache[rn]["p_norm"][0] for rn in local if (cache[rn]["zero"][0] == 0).all()])
        ent = {"runs": len(rs), "minibatch_fingerprints_equal_across_configs_and_to_own_permutation": same_fp,
               "step1_task_norm_identical_all_configs": len({tuple(v) for v in t1.values()}) == 1,
               "step1_transform_identical_protected_configs": len(tf1) <= 1,
               "step1_R_identical_joint_configs": len(R1j) <= 1, "step1_R_v1v2_identical_local_configs": len(R1l) <= 1,
               "step1_R_v1v2_identical_joint_vs_local": len(R1_shadow) <= 1,
               "step1_p_norm_rel_spread_joint": float(np.max(np.abs(p1j / p1j[0] - 1))) if len(p1j) else None,
               "step1_p_norm_rel_spread_local": float(np.max(np.abs(p1l / p1l[0] - 1))) if len(p1l) else None,
               "steps": int(len(mine))}
        ok = (same_fp and ent["step1_task_norm_identical_all_configs"] and ent["step1_transform_identical_protected_configs"]
              and ent["step1_R_identical_joint_configs"] and ent["step1_R_v1v2_identical_local_configs"]
              and ent["step1_R_v1v2_identical_joint_vs_local"]
              and (ent["step1_p_norm_rel_spread_joint"] or 0) < 1e-5 and (ent["step1_p_norm_rel_spread_local"] or 0) < 1e-5)
        ent["status"] = "PASS" if ok else "FAIL"
        rng[k] = ent
    fail_runs = sorted(n for n, v in per.items() if v["status"] != "PASS")
    summary = {n: {kk: v.get(kk) for kk in ("config", "realized_ratio_mean", "realized_ratio_rms", "combined_ratio_mean",
                                            "zero_fraction", "clip_fraction", "cap_steps", "rms_identity_max_rel_dev")}
               for n, v in per.items()}
    rec_out = res(worst(*[v["status"] for v in per.values()]), runs_checked=len(per), failing_runs=fail_runs,
                  failures={n: per[n]["failures"] for n in fail_runs}, strength_summary=summary, per_run=per)
    rng_out = res(worst(*[v["status"] for v in rng.values()]) if rng else "PENDING", per_seed=rng,
                  note="fingerprint = sha256(int64 minibatch indices)[:8] little-endian >> 1 of the verifier's own "
                       "default_rng([seed, 0, epoch]).permutation(15434)")
    return rec_out, rng_out


# ------------------------------------------------------------------------------------------------ own training engine
class Net:
    """Functional model held as leaf tensors in the runner's parameter order: enc0 (6), enc1 (6), head0 (2), head1 (2)."""
    ORDER = [f"enc.{i}.{k}" for i in (0, 1) for k in ENC_KEYS] + [f"head.{i}.{w}" for i in (0, 1) for w in ("weight", "bias")]

    def __init__(self, sd):
        self.p = {k: sd[k].detach().clone().float() for k in self.ORDER}
        self.n_enc0 = sum(self.p[f"enc.0.{k}"].numel() for k in ENC_KEYS)
        self.n_enc = 2 * self.n_enc0

    def leaves(self):
        for v in self.p.values():
            v.requires_grad_(True)
        return [self.p[k] for k in self.ORDER]

    def freeze(self):
        for v in self.p.values():
            v.requires_grad_(False)

    def enc(self, i, X):
        h = F.linear(X, self.p[f"enc.{i}.0.weight"], self.p[f"enc.{i}.0.bias"])
        h = torch.relu(h)
        h = F.linear(h, self.p[f"enc.{i}.2.weight"], self.p[f"enc.{i}.2.bias"])
        h = torch.relu(h)
        return F.linear(h, self.p[f"enc.{i}.4.weight"], self.p[f"enc.{i}.4.bias"])

    def head(self, i, r):
        return F.linear(r, self.p[f"head.{i}.weight"], self.p[f"head.{i}.bias"])

    def state(self):
        return {k: v.detach().clone() for k, v in self.p.items()}


def views(net, X, which, head, grad=False):
    need = set()
    for w in which:
        need |= {0, 1} if w == "pair" else {int(w[1]) - 1}
    v = {}
    with (torch.enable_grad() if grad else torch.no_grad()):
        for i in sorted(need):
            r = net.enc(i, X)
            lg = F.linear(r, head[i][0], head[i][1])
            v[i] = torch.cat([r, lg - lg.mean(1, keepdim=True)], 1)
        return {w: (v[0] if w == "v1" else v[1] if w == "v2" else torch.cat([v[0], v[1]], 1)) for w in which}


class Tf:
    """Floored ZCA transform (V - mu) @ W, statistics in float64, applied in float32."""

    def __init__(self, V=None, state=None):
        if state is not None:
            self.mu64, self.W64 = state["mu"], state["W"]
        else:
            V = V.double()
            mu = V.mean(0)
            C = torch.cov((V - mu).T)
            ev, U = torch.linalg.eigh(C)
            W = (U / torch.sqrt(torch.clamp(ev, min=float(ev.max()) * WHITEN_FLOOR))) @ U.T
            self.mu64, self.W64 = mu, W
        self.mu, self.W = self.mu64.float(), self.W64.float()

    def __call__(self, V):
        return (V - self.mu) @ self.W


def critic_fwd(sd, kind, Z):
    h = torch.relu(F.linear(Z, sd["0.weight"], sd["0.bias"]))
    if kind == "A":
        return F.linear(h, sd["2.weight"], sd["2.bias"])
    h = torch.relu(F.linear(h, sd["2.weight"], sd["2.bias"]))
    return F.linear(h, sd["4.weight"], sd["4.bias"])


def surrogate(logits, S, logprior, H):
    """R = (CE_const - min(CE_const, CE_A, CE_B)) / H (exact zero, no gradient path, if the constant wins)."""
    ces = [F.cross_entropy(l, S) for l in logits]
    cc = F.nll_loss(logprior.expand(len(S), 2), S)
    allc = ces + [cc]
    vals = [float(x.detach()) for x in allc]
    j = int(np.argmin(vals))
    return (cc - allc[j]) / H, j


def coef_of(c):
    if c["mode"] == "RAW":
        b = c["beta"]
        return {"v1": b / 3, "v2": b / 3, "pair": b / 3} if c["treat"] == "J" else {"v1": b / 2, "v2": b / 2, "pair": 0.0}
    if c["mode"] == "NORM":
        return {"v1": 1.0 / 3, "v2": 1.0 / 3, "pair": 1.0 / 3} if c["treat"] == "J" else {"v1": 0.5, "v2": 0.5, "pair": 0.0}
    return {"v1": 0.0, "v2": 0.0, "pair": 0.0}


def fgrad(loss, params):
    gs = torch.autograd.grad(loss, params, allow_unused=True)
    return torch.cat([(gg if gg is not None else torch.zeros_like(p)).reshape(-1) for gg, p in zip(gs, params)])


def one_update(net, c, T, critics, head, TD, bi, lr=LR):
    """One registered encoder/head update on minibatch bi from the current parameters, critics and transforms (the
    critics are given as state dicts aligned with this step). Returns (receipt dict, update vector u)."""
    mode = c["mode"]
    params = net.leaves()
    ne = net.n_enc
    blocks = ((0, net.n_enc0), (net.n_enc0, ne))
    b = torch.from_numpy(bi)
    Xb = TD["X"][b]
    Lt = {i: F.cross_entropy(net.head(i, net.enc(i, Xb)), TD["Y"][i][b]) for i in (0, 1)}
    gt = fgrad(sum(Lt.values()), params)
    t = -gt
    coef = coef_of(c)
    active = [v for v in VIEWS if coef[v] > 0]
    st = c.get("beta", c.get("rho", 0.0)) if mode != "TASK" else 0.0
    protect = mode != "TASK" and bool(active) and st > 0
    d_enc = torch.zeros(ne)
    zero = [5, 5] if not protect else [0, 0]
    Rrec = [float("nan")] * 3
    infos = [None, None]
    if protect:
        V = views(net, Xb, active, head, grad=True)
        P = 0.0
        for vi, v in enumerate(VIEWS):
            if v in active:
                Rv, _ = surrogate([critic_fwd(critics[v][k], k, T[v](V[v])) for k in KINDS], TD["S"][b],
                                  TD["logprior"], TD["H"])
                Rrec[vi] = float(Rv.detach())
                P = P + coef[v] * Rv
        if not (isinstance(P, torch.Tensor) and P.requires_grad):
            zero = [1, 1]
        elif mode == "RAW":
            d_enc = -fgrad(P, params[:12])
        else:
            pvec = fgrad(P, params[:12])
            s = alloc(c["a"])
            parts = []
            with torch.no_grad():
                for i, (lo, hi) in enumerate(blocks):
                    nt, npn = float(gt[lo:hi].double().norm()), float(pvec[lo:hi].double().norm())
                    info = {"p_norm": npn, "cap": False, "a": 0.0}
                    if npn <= ZERO_TOL or nt == 0.0:
                        parts.append(torch.zeros_like(pvec[lo:hi]))
                        zero[i] = 2 if npn <= ZERO_TOL else 3
                    else:
                        a = st * s[i] * nt / npn
                        if a > A_MAX:
                            a, info["cap"] = A_MAX, True
                        info["a"] = a
                        info["cos"] = float(torch.dot(gt[lo:hi].double(), pvec[lo:hi].double()) / (nt * npn))
                        parts.append((pvec[lo:hi].double() * a).to(pvec.dtype))
                    infos[i] = info
            d_enc = -torch.cat(parts)
    with torch.no_grad():
        q = -d_enc
        rcpt = {"t_norm": [], "q_norm": [], "cos": [], "R": Rrec, "zero": zero,
                "cap": [bool(infos[i]["cap"]) if infos[i] is not None else False for i in (0, 1)]}
        for i, (lo, hi) in enumerate(blocks):
            gi, qi = gt[lo:hi].double(), q[lo:hi].double()
            tn_, qn_ = float(gi.norm()), float(qi.norm())
            rcpt["t_norm"].append(tn_)
            rcpt["q_norm"].append(qn_)
            if mode == "RAW" and zero[i] == 0 and qn_ == 0.0:
                zero[i] = 2
            if mode == "RAW" and zero[i] == 0:
                rcpt["cos"].append(float(torch.dot(gi, qi)) / (tn_ * qn_))
            elif mode == "NORM" and infos[i] is not None and "cos" in infos[i]:
                rcpt["cos"].append(infos[i]["cos"])
            else:
                rcpt["cos"].append(float("nan"))
        u = torch.cat([t[:ne] + d_enc, t[ne:]])
        nrm = float(u.norm())
        rcpt["pre_total"] = nrm
        kappa = 1.0
        if nrm > CLIP:
            kappa = CLIP / nrm
            u = u * kappa
        rcpt["kappa"] = kappa
        rcpt["post_total"] = float(u.double().norm())
        o = 0
        for p in params:
            k_ = p.numel()
            p.add_(u[o:o + k_].view_as(p), alpha=lr)
            o += k_
    net.freeze()
    return rcpt, u


def compare_receipt(mine, z, j):
    """Max relative deviation between an own recomputed receipt and the saved receipt row j."""
    dev = {}
    for kk in ("t_norm", "q_norm"):
        a, b = np.asarray(mine[kk]), z[kk][j]
        dev[kk] = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-30))) if (np.abs(b) > 0).any() or (np.abs(a) > 0).any() else 0.0
    a, b = np.asarray(mine["R"]), z["R"][j]
    dev["R"] = maxdiff(a, b)
    a, b = np.asarray(mine["cos"]), z["cos"][j]
    dev["cos"] = maxdiff(a, b)
    dev["pre_total"] = abs(mine["pre_total"] - z["pre_total"][j]) / max(z["pre_total"][j], 1e-30)
    dev["kappa"] = abs(mine["kappa"] - z["kappa"][j])
    dev["zero_equal"] = bool(list(mine["zero"]) == [int(x) for x in z["zero"][j]])
    return dev


def check_step_replay(D: Data, inv, seeds, captures=True):
    """Independent re-derivation of the last update of every run (and of every captured step) from saved snapshots."""
    runs = sorted(n for n in inv if n.startswith("run__") and unit_done(n) and "__nonfinite" not in n
                  and int(n.split("__")[1][1:]) in seeds and (UNITS / n / "final.pt").exists())
    if not runs:
        return res("PENDING", reason="no runs")
    TD = D.train_tensors()
    per = {}
    batches = {k: my_batches(k, TD["n"]) for k in seeds}
    heads = {}
    for rn in runs:
        rec = unit_rec(rn)
        k, c = rec["seed"], parse_cid(rec["config"])
        if k not in heads:
            w = tload(UNITS / f"warm__s{k}" / "warm.pt")
            heads[k] = {i: (w[f"head.{i}.weight"].detach().clone().float(), w[f"head.{i}.bias"].detach().clone().float())
                        for i in (0, 1)}
        z = np.load(UNITS / rn / "steps.npz", allow_pickle=False)
        fin = tload(UNITS / rn / "final.pt")
        snap = fin["theta_T_minus_1"]
        S = len(z["step"])
        ent = {"config": rec["config"]}
        fails = []
        if c["mode"] != "TASK":
            ch = snap["critic_head"]
            ent["snapshot_critic_head_equals_warm_head"] = all(torch.equal(ch[i][0], heads[k][i][0]) and
                                                               torch.equal(ch[i][1], heads[k][i][1]) for i in (0, 1))
            if not ent["snapshot_critic_head_equals_warm_head"]:
                fails.append("critic head != warm head")
        todo = [("final", S, snap, fin["theta_T"])]
        if captures and (UNITS / rn / "captures.pt").exists():
            cap = tload(UNITS / rn / "captures.pt")
            exp_steps = sorted({max(1, int(round(f * S))) for f in PROGRESS})
            ent["capture_steps"] = sorted(cap)
            if sorted(cap) != exp_steps:
                fails.append(f"capture steps {sorted(cap)} != {exp_steps}")
            todo += [(f"capture_{s}", s, cap[s], None) for s in sorted(cap)]
        for lab, s, sn, theta_next in todo:
            if sn.get("step") != s:
                fails.append(f"{lab}: snapshot step {sn.get('step')}")
            net = Net(sn["model"])
            T = {v: Tf(state=sn["transforms"][v]) for v in VIEWS} if c["mode"] != "TASK" else None
            crit = sn.get("critics")
            mine, u = one_update(net, c, T, crit, heads[k], TD, batches[k][s - 1][1])
            dev = compare_receipt(mine, z, s - 1)
            e = {"receipt_dev": dev}
            if theta_next is not None:
                eq, md, _ = tensor_dict_equal(net.state(), {kk: theta_next[kk] for kk in Net.ORDER})
                e["theta_T_bit_exact"], e["theta_T_max_abs_diff"] = eq, md
                if md > 1e-6:
                    fails.append(f"{lab}: recomputed theta_T differs by {md:.3g}")
                if c["mode"] != "TASK" and c["treat"] == "L":
                    # power check for 'no pair encoder gradient': the same step WITH the (trained shadow) pair critic
                    # in the proxy must NOT reproduce theta_T
                    cf_net = Net(sn["model"])
                    one_update(cf_net, dict(c, treat="J"), T, crit, heads[k], TD, batches[k][s - 1][1])
                    eqc, mdc, _ = tensor_dict_equal(cf_net.state(), {kk: theta_next[kk] for kk in Net.ORDER})
                    e["pair_counterfactual_reproduces_theta_T"] = eqc
                    e["pair_counterfactual_theta_T_max_abs_diff"] = mdc
                    if eqc:
                        fails.append(f"{lab}: local theta_T also reproduced with the pair critic (test has no power)")
            if dev["t_norm"] > 1e-6 or dev["q_norm"] > 1e-5 or dev["R"] > 1e-6 or dev["pre_total"] > 1e-6 \
                    or dev["kappa"] > 1e-9 or not dev["zero_equal"] or (dev["cos"] > 1e-5 and math.isfinite(dev["cos"])):
                fails.append(f"{lab}: receipt mismatch {json.dumps(jsonable(dev))}")
            ent[lab] = e
        ent["failures"] = fails
        ent["status"] = "FAIL" if fails else "PASS"
        per[rn] = ent
    fin_exact = sum(1 for v in per.values() if v.get("final", {}).get("theta_T_bit_exact"))
    max_theta = max((v.get("final", {}).get("theta_T_max_abs_diff", 0) for v in per.values()), default=None)
    ncap = sum(1 for v in per.values() for kk in v if kk.startswith("capture_"))
    return res(worst(*[v["status"] for v in per.values()]), runs=len(per), final_updates_bit_exact=fin_exact,
               max_theta_T_abs_diff=max_theta, captured_steps_recomputed=ncap,
               failing=sorted(n for n, v in per.items() if v["status"] != "PASS"), per_run=per,
               method="own functional forward/backward (encoders, fixed warm-head views, saved floored-ZCA transforms, "
                      "saved critics, surrogate with the constant), RAW beta*p / NORM rho*s_i*|t|/|p|*p with cap 100 and "
                      "zero tolerance 1e-12, global clip 5, SGD 0.05, on the verifier's own salt-0 minibatch")


def new_critic_module(seed, view, kind):
    torch.manual_seed(int(hashlib.sha256("|".join(map(str, ("rgj-critic", seed, view, kind, "init"))).encode()).hexdigest()[:8], 16))
    dv = DV[view]
    if kind == "A":
        return torch.nn.Sequential(torch.nn.Linear(dv, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))
    return torch.nn.Sequential(torch.nn.Linear(dv, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(),
                               torch.nn.Linear(64, 2))


def train_replay(D: Data, TD, k, cid, n_epochs=EPOCHS):
    """Full independent re-training of one configuration from the admitted warm state (own loop). Returns final state,
    per-step receipts and the epoch-20 state."""
    c = parse_cid(cid)
    w = tload(UNITS / f"warm__s{k}" / "warm.pt")
    head = {i: (w[f"head.{i}.weight"].detach().clone().float(), w[f"head.{i}.bias"].detach().clone().float()) for i in (0, 1)}
    net = Net(w)
    has = c["mode"] != "TASK"
    banks, opts = {}, {}
    if has:
        for v in VIEWS:
            banks[v] = {kk: new_critic_module(k, v, kk) for kk in KINDS}
            opts[v] = {kk: torch.optim.Adam(banks[v][kk].parameters(), lr=CRITIC_LR) for kk in KINDS}
    n = TD["n"]
    rec = []
    ck20 = None
    step = 0
    for ep in range(n_epochs):
        if ep == 20:
            ck20 = net.state()
        perm = np.random.default_rng([k, 0, ep]).permutation(n)
        crng = np.random.default_rng([k, 0, ep, 7])
        for s0 in range(0, n, BATCH):
            bi = perm[s0:s0 + BATCH]
            step += 1
            T = None
            if has:
                net.freeze()
                Vref = views(net, TD["X"][torch.from_numpy(TD["ref"])], list(VIEWS), head)
                T = {v: Tf(Vref[v]) for v in VIEWS}
                for _ in range(CRITIC_STEPS):
                    cb = torch.from_numpy(crng.choice(TD["cf"], BATCH, replace=False))
                    V = views(net, TD["X"][cb], list(VIEWS), head)
                    for v in VIEWS:
                        for kk in KINDS:
                            loss = F.cross_entropy(banks[v][kk](T[v](V[v])), TD["S"][cb])
                            opts[v][kk].zero_grad()
                            loss.backward()
                            opts[v][kk].step()
            crit = {v: {kk: banks[v][kk].state_dict() for kk in KINDS} for v in banks} if has else None
            r, _ = one_update(net, c, T, crit, head, TD, bi)
            rec.append(r)
    if n_epochs == 20:
        ck20 = net.state()
    return net.state(), rec, ck20, ({v: {kk: {a: t.detach().clone() for a, t in banks[v][kk].state_dict().items()}
                                         for kk in KINDS} for v in banks} if has else None)


def check_train_replay(D: Data, which):
    if not which:
        return res("NOT_APPLICABLE", reason="no training replay requested")
    TD = D.train_tensors()
    per = {}
    for k, cid in which:
        rn, ln = run_name(k, cid), rel_name(k, cid)
        if not (unit_done(rn) and unit_done(ln)):
            per[f"{k}|{cid}"] = res("PENDING", reason="unit not complete")
            continue
        t0 = time.time()
        st, recs, ck20, crit = train_replay(D, TD, k, cid)
        wall = time.time() - t0
        model = tload(UNITS / ln / "model.pt")
        eq, md, _ = tensor_dict_equal(st, {kk: model[kk] for kk in Net.ORDER})
        ent = {"run": rn, "wall_s": wall, "final_model_bit_exact": eq, "final_model_max_abs_diff": md}
        ckp = UNITS / rn / "ck20.pt"
        if ckp.exists():
            c20 = tload(ckp)["model"]
            e20, m20, _ = tensor_dict_equal(ck20, {kk: c20[kk] for kk in Net.ORDER})
            ent["epoch20_bit_exact"], ent["epoch20_max_abs_diff"] = e20, m20
        if crit is not None and (UNITS / ln / "critics.pt").exists():
            cs = tload(UNITS / ln / "critics.pt")["critics"]
            ent["final_critics_bit_exact"] = all(tensor_dict_equal(crit[v][kk], cs[v][kk])[0] for v in VIEWS for kk in KINDS)
        z = np.load(UNITS / rn / "steps.npz", allow_pickle=False)
        dev = {"t_norm": 0.0, "q_norm": 0.0, "R": 0.0, "pre_total": 0.0, "kappa": 0.0, "zero_mismatch_steps": 0}
        exact = 0
        for j, r in enumerate(recs):
            d = compare_receipt(r, z, j)
            for kk in ("t_norm", "q_norm", "R", "pre_total", "kappa"):
                if math.isfinite(d[kk]):
                    dev[kk] = max(dev[kk], d[kk])
            dev["zero_mismatch_steps"] += int(not d["zero_equal"])
            exact += int(r["pre_total"] == z["pre_total"][j] and r["t_norm"] == z["t_norm"][j].tolist())
        ent["receipt_max_rel_dev"] = dev
        ent["steps_with_bit_exact_pre_total_and_task_norms"] = exact
        ent["steps"] = len(recs)
        ok = eq or md <= 1e-5
        ent["status"] = "PASS" if (eq and ent.get("epoch20_bit_exact", True) and ent.get("final_critics_bit_exact", True)
                                   and dev["zero_mismatch_steps"] == 0) else ("WARN" if ok else "FAIL")
        per[rn] = ent
    return res(worst(*[v["status"] for v in per.values()]), per_run=per,
               method="own 40-epoch loop from the admitted warm state: salt-0 task order, critics initialised by the "
                      "registered seed rule and trained 5 Adam(3e-3) steps per encoder step on the own CRITIC_FIT draw "
                      "with the floored ZCA transform recomputed on the own 4096-row reference subset; OSF_DEFENSE_FIT "
                      "labels only")


# ------------------------------------------------------------------------------------------------ admitted replays
def check_replays(inv, seeds):
    per = {}
    for k in seeds:
        for cid in ADMITTED_IDS:
            rn = run_name(k, cid)
            if not unit_done(rn):
                per[rn] = res("PENDING", reason="replay unit missing")
                continue
            rec = unit_rec(rn)
            ent = {"record_bitwise": rec.get("bitwise")}
            fails = []
            fin = tload(UNITS / rn / "final.pt")
            a40 = tload(ADMITTED / smf_name(k, cid) / "model.pt")
            eq, md, _ = tensor_dict_equal(fin["theta_T"], a40)
            ent["final_theta_T_equals_admitted_e40"] = eq
            rel = tload(UNITS / rel_name(k, cid) / "model.pt")
            ent["released_model_equals_admitted_e40"] = tensor_dict_equal(rel, a40)[0]
            if not (eq and ent["released_model_equals_admitted_e40"]):
                fails.append("e40")
            if cid != "U":
                c20 = tload(UNITS / rn / "ck20.pt")
                a20 = tload(ADMITTED / smf_name(k, cid).replace("__e40", "__e20") / "model.pt")
                ent["ck20_equals_admitted_e20"] = tensor_dict_equal(c20["model"], a20)[0]
                ent["ck20_step"] = c20.get("step")
                acr = tload(ADMITTED / smf_name(k, cid) / "critics.pt")["critics"]
                fc = fin["theta_T_minus_1"]["critics"]
                ent["final_critics_equal_admitted_e40_critics"] = all(tensor_dict_equal(fc[v][kk], acr[v][kk])[0]
                                                                      for v in VIEWS for kk in KINDS)
                rc = tload(UNITS / rel_name(k, cid) / "critics.pt")["critics"]
                ent["released_critics_equal_admitted"] = all(tensor_dict_equal(rc[v][kk], acr[v][kk])[0]
                                                             for v in VIEWS for kk in KINDS)
                if not (ent["ck20_equals_admitted_e20"] and ent["ck20_step"] == 20 * 61 and
                        ent["final_critics_equal_admitted_e40_critics"] and ent["released_critics_equal_admitted"]):
                    fails.append("e20/critics")
                # the admitted rgj run receipts: logged norms every 20 steps over all 40 epochs
                c = parse_cid(cid)
                arn = ADMITTED / f"run__raw__s{k}__RAW-{c['treat']}__b{g(c['beta'])}"
                adg = jload(arn / "record.json")["diag"]
                z = np.load(UNITS / rn / "steps.npz", allow_pickle=False)
                tot_exact, dev_t, dev_p = 0, 0.0, 0.0
                for e in adg["norms"]:
                    j = e["step"] - 1
                    tot_exact += int(z["pre_total"][j] == e["total"])
                    dev_t = max(dev_t, abs(math.sqrt(float((z["t_norm"][j] ** 2).sum())) - e["task"]) / e["task"])
                    dev_p = max(dev_p, abs(math.sqrt(float((z["q_norm"][j] ** 2).sum())) - e["penalty"]) / max(e["penalty"], 1e-30))
                ent["rgj_logged_entries"] = len(adg["norms"])
                ent["pre_clip_total_bit_exact_entries"] = tot_exact
                ent["task_norm_max_rel_dev"] = dev_t
                ent["penalty_norm_max_rel_dev"] = dev_p
                ent["clip_hits_equal_rgj"] = adg["clip_hits"] == rec["diag"]["clip_hits"]
                ent["critic_updates_equal_rgj"] = adg["critic_online_updates"] == rec["diag"]["critic_online_updates"]
                try:
                    af = tload(arn / "final.pt")
                    ent["final_theta_T_equals_admitted_run_receipt"] = tensor_dict_equal(af["theta_T"], fin["theta_T"])[0]
                except Exception as ex:  # noqa: BLE001
                    ent["final_theta_T_equals_admitted_run_receipt"] = f"not loadable with weights_only: {type(ex).__name__}"
                if not (tot_exact == len(adg["norms"]) == 122 and dev_t < 1e-5 and dev_p < 1e-5 and
                        ent["clip_hits_equal_rgj"] and ent["critic_updates_equal_rgj"]):
                    fails.append("rgj logged norms")
            ent["failures"] = fails
            ent["status"] = "FAIL" if fails else "PASS"
            per[rn] = ent
    return res(worst(*[v["status"] for v in per.values()]), replays=len(per),
               bitwise=sum(1 for v in per.values() if v.get("status") == "PASS"), per_run=per)


def check_engineering(inv, seeds):
    out = {}
    par = {}
    for k in seeds:
        for cid in ("RAW-J|b0", "RAW-L|b0", "NORM-J|r0|a1", "NORM-L|r0|a2"):
            n = f"parity__s{k}__{safe(cid)}"
            r = unit_rec(n)
            par[n] = res("PENDING") if r is None else res(
                "PASS" if r.get("pass") is True and r.get("critic_online_updates") == 4 * 61 * 30 and r.get("config") == cid
                else "FAIL", passed=r.get("pass"), critic_updates=r.get("critic_online_updates"))
    out["parity"] = res(worst(*[v["status"] for v in par.values()]), units=par)
    fid = {}
    for k in seeds:
        for t in ("J", "L"):
            n = f"fid__s{k}__RAW-{t}_b0.3"
            r = unit_rec(n)
            if r is None:
                fid[n] = res("PENDING")
                continue
            e, ec, ecap = r["equivalence"], r["expected_failure_common_rho"], r["expected_failure_cap"]
            f = []
            if not (r["bitwise_model"] and r["bitwise_critics"] and r["logged_norm_max_rel_dev"] < 1e-5
                    and r["logged_norm_entries"] == 6 and r["pass"] is True):
                f.append("fidelity receipt")
            rr = []
            for x in e["encoders"]:
                rr.append(x["r_i"])
                if not (abs(x["r_i"] - x["q_raw_norm"] / x["t_norm"]) <= 1e-15 * x["r_i"] and x["rho_used"] == x["r_i"]
                        and abs(x["rel_err"] - x["l2_err"] / x["q_raw_norm"]) <= 1e-15 and x["rel_err"] <= 1e-4
                        and x["equivalent"] is True and x["cap"] is False):
                    f.append("equivalence algebra")
            rc = math.sqrt((rr[0] ** 2 + rr[1] ** 2) / 2)
            dev_c = []
            for x, ri in zip(ec["encoders"], rr):
                dev_c.append(abs(x["rel_err"] - abs(rc / ri - 1)))
                if abs(x["rho_used"] - rc) > 1e-15 * rc:
                    f.append("common rho != RMS of r_i")
            close = abs(rr[0] - rr[1]) <= 10 * 1e-4 * max(rr)
            if (ec["equivalent"] and not close) or max(dev_c) > 2e-6:
                f.append("expected failure (common rho)")
            dev_cap = [abs(x["rel_err"] - abs(1e-6 / 0.3 - 1)) for x in ecap["encoders"]]
            if ecap["equivalent"] or not all(x["cap"] for x in ecap["encoders"]) or max(dev_cap) > 1e-4:
                f.append("expected failure (cap)")
            ent = {"bitwise_model_and_critics_vs_rgj": bool(r["bitwise_model"] and r["bitwise_critics"]),
                   "logged_norm_max_rel_dev": r["logged_norm_max_rel_dev"], "equivalence_rel_err": [x["rel_err"] for x in e["encoders"]],
                   "r_i": rr, "common_rho_rel_err_rederived_max_abs_dev": max(dev_c),
                   "cap_rel_err_rederived_max_abs_dev": max(dev_cap), "batch_index": e.get("batch_index"),
                   "online_critic_applicability": e.get("online_critic_applicability")}
            # the replay of the same trajectory: step 123 (= epoch 2, batch 0) receipt of run__s{k}__RAW-t_b0.3
            rn = run_name(k, f"RAW-{t}|b0.3")
            if e.get("batch_index") == 0 and unit_done(rn):
                z = np.load(UNITS / rn / "steps.npz", allow_pickle=False)
                ent["equivalence_t_norm_equals_replay_receipt_step_123"] = [x["t_norm"] for x in e["encoders"]] == z["t_norm"][122].tolist()
                if not ent["equivalence_t_norm_equals_replay_receipt_step_123"]:
                    f.append("equivalence t_norm vs replay receipt")
            ent["failures"] = f
            ent["status"] = "FAIL" if f else "PASS"
            fid[n] = ent
    out["fidelity"] = res(worst(*[v["status"] for v in fid.values()]), units=fid)
    tm = unit_rec("timing__s0__NORM-J_r3_a1")
    out["timing"] = res("INFO" if tm else "PENDING", **({"wall_s": tm.get("wall_s"), "epochs": tm.get("epochs"),
                                                         "role": tm.get("role")} if tm else {}))
    out["status"] = worst(out["parity"]["status"], out["fidelity"]["status"])
    return out


# ------------------------------------------------------------------------------------------------ integrity and timing
def check_complete(inv):
    bad, incomplete, extra, n_ok = [], [], [], 0
    for name, d in sorted(inv.items()):
        cp = d / "COMPLETE.json"
        if not cp.exists():
            incomplete.append(name)
            continue
        c = jload(cp)
        files = c.get("files", {})
        good = True
        for f_, h in files.items():
            p = d / f_
            if not p.exists() or sha_file(p) != h:
                bad.append(f"{name}/{f_}")
                good = False
        present = {str(p.relative_to(d)) for p in d.rglob("*") if p.is_file()} - {"COMPLETE.json"}
        ex = sorted(present - set(files))
        if ex:
            extra.append({name: ex})
        if c.get("id") != name:
            bad.append(f"{name}: COMPLETE id {c.get('id')}")
            good = False
        n_ok += int(good)
    st = "FAIL" if bad else ("WARN" if extra or incomplete else "PASS")
    return res(st, units_with_complete=n_ok, hash_failures=bad, files_not_listed=extra,
               units_without_complete_json=incomplete)


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
            return t, gs
    return None, None


def activity():
    p = RUN / "ACTIVITY_LOG.jsonl"
    ev = []
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            ev.append(e)
    return ev


STAGE_LOCK = {"admit": "DATA_AND_ENGINEERING_LOCK", "parity": "DATA_AND_ENGINEERING_LOCK",
              "fidelity": "DATA_AND_ENGINEERING_LOCK", "replay": "DATA_AND_ENGINEERING_LOCK",
              "timing": "DATA_AND_ENGINEERING_LOCK", "bank": "TRAINING_PROTOCOL_LOCK", "references": "TRAINING_PROTOCOL_LOCK",
              "inner": "SELECTION_AND_AUDIT_LOCK", "select": "SELECTION_AND_AUDIT_LOCK", "tracking": "SELECTION_AND_AUDIT_LOCK",
              "assess": "EVALUATION_LOCK", "outer": "EVALUATION_LOCK", "infer": "EVALUATION_LOCK"}
LOCK_ORDER = ["DATA_AND_ENGINEERING_LOCK", "TRAINING_PROTOCOL_LOCK", "SELECTION_AND_AUDIT_LOCK", "EVALUATION_LOCK"]


def lock_files():
    out = {}
    for name in LOCK_ORDER + sorted(p.stem for p in RES.glob("AMENDMENT_A*.json")):
        p = RES / f"{name}.json"
        rel = f"{REL_RES}/{name}.json"
        log = git("log", "--format=%H|%cI", "--", rel) or ""
        commits = [l.split("|") for l in log.splitlines() if l]
        out[name] = {"exists": p.exists(), "commits": commits}
    return out


def check_lock_order(inv):
    """Every stage start / governed unit after the push of its governing lock and of every lock/amendment committed
    before it (remote-tracking reflog), and the worktree copy of each lock equals its latest commit."""
    entries = remote_reflog()
    ev = activity()
    lf = lock_files()
    info, fails, warns = {}, [], []
    for name, d in lf.items():
        if not d["exists"]:
            info[name] = {"exists": False}
            continue
        if not d["commits"]:
            info[name] = {"exists": True, "committed": False}
            if name in LOCK_ORDER[:2] or name.startswith("AMENDMENT"):
                fails.append(f"{name} not committed")
            continue
        first_c, first_t = d["commits"][-1]
        last_c, last_t = d["commits"][0]
        fp, _ = first_remote(first_c, entries)
        lp, _ = first_remote(last_c, entries)
        rel = f"{REL_RES}/{name}.json"
        blob_ok = git("rev-parse", f"{last_c}:{rel}") == git("hash-object", str(RES / f"{name}.json"))
        info[name] = {"first_commit": first_c, "first_commit_time": parse_iso(first_t), "first_push_time": fp,
                      "latest_commit": last_c, "latest_push_time": lp, "versions": len(d["commits"]),
                      "worktree_equals_latest_commit": blob_ok}
        if not blob_ok:
            fails.append(f"{name}: worktree differs from its latest commit")
        if fp is None:
            fails.append(f"{name}: never pushed")
    starts = [e for e in ev if str(e.get("event", "")).startswith("start ") or e.get("event") == "assessment opened"]
    st_out = []
    for e in starts:
        stage = "assess" if e["event"] == "assessment opened" else e["event"].split(" ", 1)[1]
        t = parse_iso(e["at"])
        gov = STAGE_LOCK.get(stage)
        rec = {"stage": stage, "at": t, "lock_named": e.get("lock"), "governing_lock": gov}
        ok = True
        named = e.get("lock")
        if gov in LOCK_ORDER and named in LOCK_ORDER and LOCK_ORDER.index(named) < LOCK_ORDER.index(gov):
            ok = False                       # stage run under a lock older than its governing lock
        g_ = info.get(gov) or {}
        if gov and not (g_.get("first_push_time") and g_["first_push_time"] <= t):
            ok = False
        # no lock/amendment version committed before t may be unpushed at t
        unpushed = []
        for name, d in lf.items():
            for c, ct in d["commits"]:
                if parse_iso(ct) <= t:
                    pt, _ = first_remote(c, entries)
                    if pt is None or pt > t:
                        unpushed.append(name)
        rec["locks_committed_but_unpushed_at_start"] = sorted(set(unpushed))
        if unpushed:
            ok = False
        rec["ok"] = ok
        if not ok:
            fails.append(f"start {stage} at {iso(t)}: governing lock not pushed before (or unpushed lock versions)")
        st_out.append(rec)
    # governed units: completion after the governing lock's first push
    unit_gov = {"warm__": "DATA_AND_ENGINEERING_LOCK", "parity__": "DATA_AND_ENGINEERING_LOCK",
                "fid__": "DATA_AND_ENGINEERING_LOCK", "timing__": "DATA_AND_ENGINEERING_LOCK",
                "inner__": "SELECTION_AND_AUDIT_LOCK", "track__": "SELECTION_AND_AUDIT_LOCK", "outer__": "EVALUATION_LOCK",
                "lc__": "TRAINING_PROTOCOL_LOCK", "fare__": "TRAINING_PROTOCOL_LOCK"}
    early = []
    for name in inv:
        if not unit_done(name):
            continue
        t = utc((UNITS / name / "COMPLETE.json").stat().st_mtime)
        gov = next((v for p, v in unit_gov.items() if name.startswith(p)), None)
        if name.startswith(("rel__", "run__")):
            admitted = any(name in (rel_name(k, a), run_name(k, a)) for k in SEEDS for a in ADMITTED_IDS)
            gov = "DATA_AND_ENGINEERING_LOCK" if admitted else "TRAINING_PROTOCOL_LOCK"
        if gov is None:
            continue
        fpush = (info.get(gov) or {}).get("first_push_time")
        if fpush is None or t < fpush:
            early.append({"unit": name, "complete_at": t, "governing_lock": gov})
    if early:
        fails.append(f"{len(early)} units completed before their governing lock was pushed")
    outer = sorted(n for n in inv if n.startswith("outer__"))
    return res("FAIL" if fails else ("WARN" if warns else "PASS"), locks=info, stage_starts=st_out,
               units_completed_before_governing_lock=early[:30], outer_units=len(outer), failures=fails)


def check_code_hashes():
    order = []
    for name in LOCK_ORDER + sorted(p.stem for p in RES.glob("AMENDMENT_A*.json")):
        p = RES / f"{name}.json"
        if p.exists():
            d = jload(p)
            order.append((d.get("written_at", ""), name, d))
    order.sort()
    current, violations, relocks = {}, [], []
    for _, name, d in order:
        cf = d.get("code_files") or {}
        declared = set(d.get("changes_previously_locked") or [])
        for f_, h in cf.items():
            if f_ in current and current[f_][0] != h:
                if name.startswith("AMENDMENT") and f_ not in declared:
                    violations.append(f"{name}: {f_} changed without being declared")
                if not name.startswith("AMENDMENT"):
                    prev = current[f_][1]
                    if not (prev.startswith("AMENDMENT")):
                        relocks.append(f"{name} re-hashes {f_} (previously {prev}) without a declared change")
            current[f_] = (h, name)
    mism = []
    for f_, (h, src) in sorted(current.items()):
        p = WT / f_
        wh = sha_file(p) if p.exists() else None
        if wh != h:
            mism.append({"file": f_, "governing_lock": src, "worktree_differs": True})
    docs = {}
    latest = next((nm for _, nm, _ in reversed(order) if not nm.startswith("AMENDMENT")), None)
    if latest:
        for doc, h in (jload(RES / f"{latest}.json").get("documents_sha256") or {}).items():
            p = RES / doc
            docs[doc] = (sha_file(p) == h) if p.exists() else None
    st = "FAIL" if (mism or violations or relocks) else "PASS"
    return res(st, locks_in_order=[nm for _, nm, _ in order], files_governed=len(current), worktree_mismatches=mism,
               undeclared_amendment_changes=violations, undeclared_relocks=relocks, latest_named_lock=latest,
               latest_named_lock_documents_match=docs)


def check_quarantine(inv):
    """Inventory of the quarantined early receipts: kept (not deleted), not present among the evidence units."""
    out, fails = {}, []
    for q in sorted(PRIV.glob("quarantine_*")):
        names = sorted(p.name for p in (q / "units" if (q / "units").exists() else q).iterdir() if p.is_dir())
        files = sorted(p.name for p in q.iterdir() if p.is_file())
        kinds = {}
        for n in names:
            kk = n.split("__")[0] + ("__" + n.split("__")[1] if n.startswith("inner__") else "")
            kinds[kk] = kinds.get(kk, 0) + 1
        same = []
        for n in names:
            src = (q / "units" / n) if (q / "units").exists() else (q / n)
            if n in inv and (src / "COMPLETE.json").exists() and (UNITS / n / "COMPLETE.json").exists() \
                    and sha_file(src / "COMPLETE.json") == sha_file(UNITS / n / "COMPLETE.json"):
                same.append(n)
        detail = {}
        if q.name.startswith("quarantine_amendment"):
            for n in names:
                r = jload(q / n / "record.json")
                e = r.get("equivalence") or {}
                detail[n] = {"pass": r.get("pass"), "bitwise_model": r.get("bitwise_model"),
                             "bitwise_critics": r.get("bitwise_critics"), "equivalence_applicable": e.get("applicable"),
                             "equivalence_equivalent": e.get("equivalent")}
        out[q.name] = {"units": len(names), "unit_kinds": kinds, "files": files,
                       "identical_receipt_reused_as_evidence": same, "fidelity_receipts": detail or None}
        if same:
            fails.append(f"{q.name}: {len(same)} quarantined receipts identical to evidence units")
    ql = RUN / "QUARANTINE.log"
    return res("FAIL" if fails else "INFO", quarantine=out, quarantine_log_lines=len(ql.read_text().splitlines()) if ql.exists() else 0,
               note="quarantined receipts are retained and are not evidence; the pre-lock reference/inner receipts read "
                    "only fitting/selection roles (no assessment label)", failures=fails)


def check_label_custody(D, inv, rel_out):
    ev = activity()
    ls = lock_commit_state()
    fp = ls.get("first_push_time")
    a_ev = [(e.get("event"), parse_iso(e["at"])) for e in ev if re.search(r"assess|unseal|outer|infer", str(e.get("event", "")))]
    bad_ev = [f"{n} at {iso(t)}" for n, t in a_ev if fp is None or t < fp]
    outer = sorted(n for n in inv if n.startswith("outer__"))
    heads_ok = all(r.get(f"recipient_{i}", {}).get("saved_scaler_fit_rows_are_OSF_DEFENSE_FIT", False)
                   for r in rel_out.values() for i in (1, 2)) if rel_out else None
    no_labels = all(not r.get("label_like_keys") and r.get("keys_ok") for r in rel_out.values()) if rel_out else None
    man = jload(RES / "ROLE_MANIFEST.json")
    lm = man["loader_comparison"]["assessment_labels_masked_to_minus_one"]
    el = (RES / "EVALUATION_LOCK.json").exists()
    out = {"verifier_assessment_labels_sealed_during_phase_1_checks": not D.unseal_log,
           "assessment_events_in_activity_log": len(a_ev), "assessment_events_before_lock_push": bad_ev,
           "evaluation_lock_first_push": fp,
           "outer_units": len(outer), "evaluation_lock_present": el,
           "heads_fitted_on_OSF_DEFENSE_FIT_rows_only": heads_ok, "release_files_without_label_arrays": no_labels,
           "manifest_loader_masking_echo": all(lm.values()),
           "note": "training replays (own loop, OSF_DEFENSE_FIT labels only) reproducing the released models show that no "
                   "other label influenced those fits"}
    out["releases_inspected"] = len(rel_out)
    ok = (not D.unseal_log and not bad_ev and not (outer and not el) and heads_ok is not False
          and no_labels is not False and all(lm.values()))
    return res(("PASS" if rel_out else "PENDING") if ok else "FAIL", **out)


# ------------------------------------------------------------------------------------------------ sanity controls
def check_controls(D: Data, L, unit="rel__s0__RAW-J_b0.3"):
    """Cheap controls under the permitted attacker interface (released view [r_i, c_i] only; no raw inputs):
      null:    SEX shuffled within each role; attacker (StandardScaler+LR) fit on AUDIT_FIT, scored on a separate
               diagnostic split (INNER_SELECTION) -> AUC ~ 0.5 (|AUC - 0.5| < 4.5 SE of the null);
      planted: a rotated 1e-6 * (2 SEX - 1) clue placed in the released view's own affine-logit null direction (a
               direction with zero variance in the clean view), serialised as float64 like the release; a reader
               that whitens the view on AUDIT_FIT (variance floor 1e-30) and fits LR recovers it (AUC > 0.99) while
               the same reader on the clean view stays at the ordinary recovery level (no hallucinated null signal)."""
    if not unit_done(unit):
        return res("PENDING", reason=f"{unit} missing")
    rel = np.load(UNITS / unit / "release.npz", allow_pickle=False)
    s = L["sex"]
    fa, se = np.flatnonzero(D.mask["AUDIT_FIT"]), np.flatnonzero(D.mask["INNER_SELECTION"])
    assert (s[fa] >= 0).all() and (s[se] >= 0).all()
    V = np.hstack([rel["r1"], rel["c1"]])
    out = {"unit": unit, "view": "v1 = [r1, c1] (18 columns)"}
    rng = np.random.default_rng(20261099)
    sf, ss = rng.permutation(s[fa]), rng.permutation(s[se])
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(V[fa], sf)
    a0 = roc_auc_score(ss, m.predict_proba(V[se])[:, 1])
    n1, n0 = int(ss.sum()), int(len(ss) - ss.sum())
    se_null = math.sqrt((n1 + n0 + 1) / (12 * n1 * n0))
    out["null_shuffled_sex_auc"] = a0
    out["null_se"] = se_null
    out["null_ok"] = abs(a0 - 0.5) < 4.5 * se_null
    # null direction of the clean view on AUDIT_FIT (smallest-variance eigenvector)
    Vc = V[fa] - V[fa].mean(0)
    ev, U = np.linalg.eigh(Vc.T @ Vc / len(fa))
    u0 = U[:, 0]
    out["clean_view_min_eigenvalue"] = float(ev[0])
    out["clean_view_rank_deficient"] = bool(ev[0] < 1e-12 * ev[-1])

    def whitened_reader(Vx):
        mu = Vx[fa].mean(0)
        C = (Vx[fa] - mu).T @ (Vx[fa] - mu) / len(fa)
        e, W = np.linalg.eigh(C)
        keep = e > 1e-30
        Z = (Vx - mu) @ W[:, keep] / np.sqrt(e[keep])
        q = LogisticRegression(C=1.0, max_iter=5000).fit(Z[fa], s[fa])
        return roc_auc_score(s[se], q.predict_proba(Z[se])[:, 1])

    a_clean = whitened_reader(V)
    Vp = V + 1e-6 * (2 * s - 1)[:, None] * u0[None, :]
    Vp = Vp.astype(np.float64)
    a_plant = whitened_reader(Vp)
    plain = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(V[fa], s[fa])
    a_plain = roc_auc_score(s[se], plain.predict_proba(V[se])[:, 1])
    out.update(clean_view_whitened_reader_auc=a_clean, planted_view_whitened_reader_auc=a_plant,
               clean_view_standard_lr_auc=a_plain)
    out["planted_detected"] = a_plant > 0.99
    out["clean_not_inflated"] = a_clean < 0.99 and abs(a_clean - a_plain) < 0.05
    out["note"] = ("labels used: AUDIT_FIT / INNER_SELECTION SEX only (permitted for attacker fitting/selection); "
                   "no assessment row, no raw input column")
    ok = out["null_ok"] and out["planted_detected"] and out["clean_not_inflated"]
    return res("PASS" if ok else "WARN", **out)


# ------------------------------------------------------------------------------------------------ critic tracking
def check_tracking(D: Data, L):
    """track__ records (diagnostic only): (1) gap arithmetic (registered = mean over kinds of online - fresh CE;
    best-of-bank; best online / fresh) re-derived from the record's CE values and compared with CRITIC_TRACKING.csv;
    (2) the ONLINE critic CE and constant-prior CE of every snapshot recomputed independently (own frozen views,
    saved / own recomputed floored-ZCA transforms, saved critics) on DIAGNOSTIC_CALIB and INNER_SELECTION rows. The
    fresh bounded refits are not re-run."""
    names = sorted(p.name for p in UNITS.glob("track__*") if unit_done(p.name))
    if not names:
        return res("PENDING", reason="no track__ records")
    TD = D.train_tensors()
    sel = np.flatnonzero(D.mask["INNER_SELECTION"])
    Xin = torch.from_numpy(D.X[sel])
    Sin = torch.from_numpy(L["sex"][sel])
    assert (Sin >= 0).all()
    cal = torch.from_numpy(D.cal)
    bad, ce_dev, n_ce, table = [], 0.0, 0, {}
    for n in names:
        rec = unit_rec(n)
        k, cid = rec["seed"], rec["config"]
        rn = rec["run"]
        cap = tload(UNITS / rn / "captures.pt")
        fin = tload(UNITS / rn / "final.pt")
        last = fin["theta_T_minus_1"]
        snaps = {f"f{f:g}": cap[s] for f, s in ((f, max(1, int(round(f * 2440)))) for f in PROGRESS) if f < 1.0}
        snaps["final_aligned_theta_T_minus_1"] = last
        snaps["theta_T_stale_transform_artifact"] = {"model": fin["theta_T"], "critics": last["critics"],
                                                     "transforms": last["transforms"], "critic_head": last["critic_head"]}
        net_T = Net(fin["theta_T"])
        Vref = views(net_T, TD["X"][torch.from_numpy(TD["ref"])], list(VIEWS), last["critic_head"])
        snaps["theta_T_recomputed_transform"] = {"model": fin["theta_T"], "critics": last["critics"],
                                                 "transforms": {v: {"mu": Tf(Vref[v]).mu64, "W": Tf(Vref[v]).W64} for v in VIEWS},
                                                 "critic_head": last["critic_head"]}
        if sorted(snaps) != sorted(rec["snapshots"]):
            bad.append(f"{n}: snapshot names {sorted(rec['snapshots'])}")
            continue
        for sn, snap in snaps.items():
            net = Net(snap["model"])
            Vall = views(net, TD["X"], list(VIEWS), snap["critic_head"])
            Vin = views(net, Xin, list(VIEWS), snap["critic_head"])
            for v in VIEWS:
                rv = rec["snapshots"][sn][v]
                Tv = Tf(state=snap["transforms"][v])
                rows = {"calib": (Tv(Vall[v][cal]), TD["S"][cal]), "inner": (Tv(Vin[v]), Sin)}
                with torch.no_grad():
                    for r, (zz, ss) in rows.items():
                        cc = float(F.nll_loss(TD["logprior"].expand(len(ss), 2), ss))
                        d0 = abs(cc - rv["const_ce"][r])
                        ce_dev, n_ce = max(ce_dev, d0), n_ce + 1
                        for kk in KINDS:
                            ce = float(F.cross_entropy(critic_fwd(snap["critics"][v][kk], kk, zz), ss))
                            d = abs(ce - rv["kinds"][kk]["online"][r])
                            ce_dev, n_ce = max(ce_dev, d), n_ce + 1
                for r in ("calib", "inner"):
                    on = [rv["kinds"][kk]["online"][r] for kk in KINDS]
                    fr = [rv["kinds"][kk]["fresh"][r] for kk in KINDS]
                    reg = float(np.mean([a - b for a, b in zip(on, fr)]))
                    bob = min(on) - min(fr)
                    if abs(reg - rv[f"gap_registered_{r}"]) > 1e-12 or abs(bob - rv[f"gap_best_of_bank_{r}"]) > 1e-12:
                        bad.append(f"{n}/{sn}/{v}/{r} gap arithmetic")
                    table[(cid, k, sn, v, r)] = {"gap_registered": reg, "gap_best_of_bank": bob, "online_best": min(on),
                                                 "fresh_best": min(fr), "const_ce": rv["const_ce"][r]}
    if ce_dev > 1e-5:
        bad.append(f"online/const CE recomputation max abs diff {ce_dev:.3g}")
    csv_cmp = res("PENDING")
    p = RES / "CRITIC_TRACKING.csv"
    if p.exists():
        cd, nn = [], 0
        for r in csv.DictReader(open(p)):
            t = table.get((r["config"], int(r["seed"]), r["snapshot"], r["view"], r["rows"]))
            if t is None:
                cd.append(f"row not in records {r['config']}/{r['seed']}/{r['snapshot']}")
                continue
            nn += 1
            for col, key in (("gap_registered", "gap_registered"), ("gap_best_of_bank", "gap_best_of_bank"),
                             ("online_best_ce", "online_best"), ("fresh_best_ce", "fresh_best"), ("const_ce", "const_ce")):
                if abs(float(r[col]) - t[key]) > 5.1e-7:
                    cd.append(f"{r['config']}/{r['seed']}/{r['snapshot']}/{r['view']}/{r['rows']} {col}")
        csv_cmp = res("FAIL" if cd else ("PASS" if nn == len(table) else "WARN"), rows_compared=nn,
                      record_rows=len(table), differences=cd[:20])
    summ = {}
    for (cid, k, sn, v, r), t in table.items():
        summ.setdefault(f"{cid}|{sn}|{r}", []).append(t["gap_registered"])
    summ = {kk: {"mean_registered_gap": float(np.mean(vv)), "n": len(vv)} for kk, vv in sorted(summ.items())}
    return res(worst("FAIL" if bad else "PASS", csv_cmp["status"] if csv_cmp["status"] != "PENDING" else None),
               records=len(names), online_ce_values_recomputed=n_ce, online_ce_max_abs_diff=ce_dev,
               failures=bad[:20], csv_comparison=csv_cmp, mean_registered_gap_by_config_snapshot_rows=summ,
               note="diagnostic only; the stale-transform theta_T variant measures a whitening artifact (review A8)")


# ------------------------------------------------------------------------------------------------ reference releases
def leace_apply(ud: Path, i: int, r_u):
    """Official LEACE eraser applied with its saved map: x - ((x - mean_x) @ proj_right^T) @ proj_left^T."""
    z = np.load(ud / f"leace_{i}" / "leace_map.npz", allow_pickle=False)
    meta = jload(ud / f"leace_{i}" / "leace_map.json")
    return r_u - ((r_u - z["mean_x"]) @ z["proj_right"].T) @ z["proj_left"].T, z, meta


def check_reference_releases(D: Data, L, seeds):
    """LEACE E: own U forward + saved official map; FARE purpose units: one-hot cells + own head refit; FARE pair units
    (F, F0) = the purpose units' arrays."""
    per, fails = {}, []
    tr, va = np.flatnonzero(D.mask[FIT]), np.flatnonzero(D.mask["HEAD_VALIDATION"])
    fit_rid_hash = hashlib.sha256(np.ascontiguousarray(D.row_id[tr].astype("<i8")).tobytes()).hexdigest()
    for k in seeds:
        n = f"lc__s{k}__E"
        if unit_done(n):
            ud = UNITS / n
            rel = np.load(ud / "release.npz", allow_pickle=False)
            sd = tload(ud / "model.pt")
            u_sd = tload(UNITS / rel_name(k, "U") / "model.pt")
            ent = {"model_equals_rel_U": tensor_dict_equal(sd, u_sd)[0], "row_id_ok": bool(np.array_equal(rel["row_id"], D.row_id))}
            f = []
            for i in (0, 1):
                ru = encode(sd, i, D.X)
                r, z, meta = leace_apply(ud, i, ru)
                h = joblib.load(ud / f"head_{i}.joblib")
                c, p, hard = outputs_of(h, rel[f"r{i + 1}"])
                y = L["y_income"] if i == 0 else L["y_occ"]
                mh, C, _ = my_fit_head(rel[f"r{i + 1}"], y, tr, va, KS[i])
                mc, mp, mhard = outputs_of(mh, rel[f"r{i + 1}"])
                e = {"features_max_abs_diff": maxdiff(r, rel[f"r{i + 1}"]),
                     "map_mean_x_equals_U_fit_row_mean": bool(np.allclose(z["mean_x"], ru[tr].mean(0), rtol=1e-12, atol=1e-14)),
                     "map_fit_rows_are_OSF_DEFENSE_FIT": meta.get("fit_row_ids_sha256") == fit_rid_hash and meta.get("n_fit") == len(tr),
                     "saved_head_outputs_bit_exact": bool(np.array_equal(c, rel[f"c{i + 1}"]) and np.array_equal(p, rel[f"p{i + 1}"])
                                                          and np.array_equal(hard, rel[f"hard{i + 1}"])),
                     "refit_C": C, "saved_C": float(h[-1].C), "refit_c_max_abs_diff": maxdiff(mc, rel[f"c{i + 1}"]),
                     "refit_hard_mismatches": int((mhard != rel[f"hard{i + 1}"]).sum())}
                ent[f"recipient_{i + 1}"] = e
                if not (e["features_max_abs_diff"] <= 1e-12 and e["map_mean_x_equals_U_fit_row_mean"] and
                        e["map_fit_rows_are_OSF_DEFENSE_FIT"] and e["saved_head_outputs_bit_exact"] and C == e["saved_C"]
                        and e["refit_c_max_abs_diff"] <= 1e-6 and e["refit_hard_mismatches"] == 0):
                    f.append(f"recipient {i + 1}")
            if not (ent["model_equals_rel_U"] and ent["row_id_ok"]):
                f.append("model/rows")
            ent["failures"], ent["status"] = f, ("FAIL" if f else "PASS")
            per[n] = ent
        for nm in sorted(p.name for p in UNITS.glob(f"fare__s{k}__p*") if unit_done(p.name)):
            ud = UNITS / nm
            rec = unit_rec(nm)
            i = int(rec["purpose"])
            rel = np.load(ud / "release.npz", allow_pickle=False)
            cells, r = rel["cells"], rel["r"]
            h = joblib.load(ud / "head.joblib")
            c, p, hard = outputs_of(h, r)
            y = L["y_income"] if i == 0 else L["y_occ"]
            mh, C, _ = my_fit_head(r, y, tr, va, KS[i])
            mc, mp, mhard = outputs_of(mh, r)
            ent = {"purpose": i, "one_hot_cells": bool(np.array_equal(r, np.eye(r.shape[1])[cells])),
                   "row_id_ok": bool(np.array_equal(rel["row_id"], D.row_id)),
                   "saved_head_outputs_bit_exact": bool(np.array_equal(c, rel["c"]) and np.array_equal(p, rel["p"]) and
                                                        np.array_equal(hard, rel["hard"])),
                   "head_scaler_n_seen": int(np.asarray(h[0].n_samples_seen_).max()), "refit_C": C,
                   "saved_C": float(h[-1].C), "refit_c_max_abs_diff": maxdiff(mc, rel["c"]),
                   "refit_hard_mismatches": int((mhard != rel["hard"]).sum())}
            ok = (ent["one_hot_cells"] and ent["row_id_ok"] and ent["saved_head_outputs_bit_exact"] and
                  ent["head_scaler_n_seen"] == len(tr) and C == ent["saved_C"] and ent["refit_c_max_abs_diff"] <= 1e-6
                  and ent["refit_hard_mismatches"] == 0)
            ent["status"] = "PASS" if ok else "FAIL"
            per[nm] = ent
        for nm in sorted(p.name for p in UNITS.glob(f"fare__s{k}__F*") if unit_done(p.name)):
            rec = unit_rec(nm)
            rel = np.load(UNITS / nm / "release.npz", allow_pickle=False)
            a, b = (np.load(UNITS / u / "release.npz", allow_pickle=False) for u in rec["purpose_units"])
            same = all(np.array_equal(rel[f"{x}1"], a[x]) and np.array_equal(rel[f"{x}2"], b[x]) for x in ("r", "c", "p", "hard"))
            per[nm] = {"purpose_units": rec["purpose_units"], "arrays_equal_purpose_units": bool(same),
                       "status": "PASS" if same else "FAIL"}
    return res(worst(*[v["status"] for v in per.values()]) if per else "PENDING", units=len(per),
               failing=sorted(n for n, v in per.items() if v["status"] != "PASS"), per_unit=per,
               note="FARE trees are not re-run (official implementation lives in a module this verifier may not import); "
                    "cells, one-hot features and heads are checked")


# ------------------------------------------------------------------------------------------------ PHASE 2: selection
def counts_of(D, L, hard, task):
    sel, fit = D.mask["INNER_SELECTION"], D.mask[FIT]
    y = L["y_income"] if task == 0 else L["y_occ"]
    assert (y[sel] >= 0).all() and (y[fit] >= 0).all()
    maj = int(np.argmax(np.bincount(y[fit], minlength=KS[task])))
    return {"k": int((hard[sel] == y[sel]).sum()), "n": int(sel.sum()), "k_const": int((y[sel] == maj).sum()),
            "const_class": maj}


def gates_of(c, cu):
    n = c["n"]
    a, au, cc = Fraction(c["k"], n), Fraction(cu["k"], cu["n"]), Fraction(c["k_const"], n)
    ex = {"G1": a - au + Fraction(1, 100), "G2": (a - cc) - Fraction(4, 5) * (au - cc), "G3": (a - cc) - Fraction(3, 100)}
    af, auf, cf = c["k"] / n, cu["k"] / cu["n"], c["k_const"] / n
    fl = {"G1": af - (auf - 0.01), "G2": (af - cf) - 0.8 * (auf - cf), "G3": (af - cf) - 0.03}
    return ex, fl


def inner_rec_check(name, D=None, V=None, L=None, refit=()):
    """Inner record consistency (auc = selected bank entry = first bank maximum within 1e-12; coalition bank = pair
    table + ignore-recipient tables) and optional own refits of named slate members (inner AUC on INNER_SELECTION)."""
    rec = unit_rec(name)
    if rec is None:
        return None
    R = rec["recovery"]
    out, fails = {}, []
    views = list(R["auc"])
    for w in views:
        if w == "pair" and "v1" in R["tables"] and "v2" in R["tables"]:
            bank = R["tables"]["pair"] + R["tables"]["v1"] + R["tables"]["v2"]
        else:
            bank = R["tables"][w]
        best = None
        for j, b in enumerate(bank):
            if best is None or b["inner_auc"] > bank[best]["inner_auc"] + 1e-12:
                best = j
        if R["auc"][w] != bank[best]["inner_auc"]:
            fails.append(f"auc[{w}] != first bank maximum")
    if R.get("n_fit") is not None and D is not None and (R["n_fit"] != int(D.mask["AUDIT_FIT"].sum()) or
                                                         R["n_select"] != int(D.mask["INNER_SELECTION"].sum())):
        fails.append("attacker roles")
    if refit and V is not None:
        s = L["sex"]
        fa, se = np.flatnonzero(D.mask["AUDIT_FIT"]), np.flatnonzero(D.mask["INNER_SELECTION"])
        spot = {}
        for w in views:
            for att in refit:
                tab = [t for t in R["tables"][w] if t["attacker"] == att]
                if not tab:
                    continue
                m = attacker_factory(att, 0).fit(V[w][fa], s[fa])
                a = roc_auc_score(s[se] == 1, proba_of(m, V[w][se], 2)[:, 1])
                spot[f"{w}|{att}"] = abs(a - tab[0]["inner_auc"])
        out["own_refit_abs_diff"] = spot
        out["own_refit_max_abs_diff"] = max(spot.values(), default=None)
        if spot and max(spot.values()) > 1e-9:
            fails.append("own attacker refit differs from the record")
    out["auc"] = R["auc"]
    out["failures"], out["status"] = fails, ("FAIL" if fails else "PASS")
    return out


class Canon:
    """Own implementation of the registered defense-aware reader's canonicalisation: float64 centre, SVD on the fitting
    rows, drop s_j <= max(n, d) eps64 max(s_1, ||X||_2), whiten the kept directions (x sqrt(n - 1))."""

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        self.mu_ = X.mean(0)
        _, s, Vt = np.linalg.svd(X - self.mu_, full_matrices=False)
        scale = max(float(s[0]) if s.size else 0.0, float(np.linalg.norm(X, 2)))
        keep = s > max(n, d) * float(np.finfo(np.float64).eps) * scale
        self.rank_ = int(keep.sum())
        self.W_ = (Vt[keep].T / s[keep]) * math.sqrt(max(n - 1, 1))
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=np.float64) - self.mu_) @ self.W_


class DAReader:
    def __init__(self, kind, seed):
        self.kind, self.seed = kind, seed

    def fit(self, X, y):
        from sklearn.dummy import DummyClassifier
        from sklearn.neural_network import MLPClassifier
        self.c_ = Canon().fit(X)
        Z = self.c_.transform(X)
        if self.c_.rank_ == 0:
            self.m_, Z = DummyClassifier(strategy="prior"), np.zeros((len(Z), 1))
        elif self.kind == "lr":
            self.m_ = LogisticRegression(C=1.0, max_iter=3000)
        else:
            self.m_ = MLPClassifier(hidden_layer_sizes=(128, 128), alpha=1e-4, max_iter=300, early_stopping=True,
                                    validation_fraction=0.1, n_iter_no_change=15, random_state=self.seed)
        self.m_.fit(Z, y)
        self.classes_ = self.m_.classes_
        return self

    def predict_proba(self, X):
        Z = self.c_.transform(X)
        return self.m_.predict_proba(Z if self.c_.rank_ else np.zeros((len(Z), 1)))


def attacker_factory(name, seed):
    """Own re-implementation of the registered slate members by name."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
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
    if name == "DA_LR":
        return DAReader("lr", seed)
    if name == "DA_MLP":
        return DAReader("mlp", seed)
    raise KeyError(name)


def proba_of(m, X, K):
    P = m.predict_proba(X)
    out = np.zeros((len(X), K))
    for j, c in enumerate(getattr(m, "classes_", range(P.shape[1]))):
        out[:, int(c)] = P[:, j]
    return out


def release_views(rel):
    v1 = np.hstack([rel["r1"], rel["c1"]])
    v2 = np.hstack([rel["r2"], rel["c2"]])
    return {"v1": v1, "v2": v2, "pair": np.hstack([v1, v2])}


def fare_selection(D, L, ucnt, seeds):
    """Per purpose: configurations passing that purpose's gates vs U on INNER_SELECTION on every seed; lowest seed-mean
    inner local AUC (ties lower id); none -> NO_FEASIBLE_CONFIGURATION and the closest configuration (largest
    min-over-seeds worst-gate margin, ties lower id) kept descriptively."""
    out = {}
    for i in (0, 1):
        by = {}
        for cid in range(1, 7):
            rows = []
            for k in seeds:
                u = f"fare__s{k}__p{i}__c{cid}"
                ir = unit_rec(f"inner__{u}")
                if ir is None or not unit_done(u):
                    rows = None
                    break
                rel = np.load(UNITS / u / "release.npz", allow_pickle=False)
                c = counts_of(D, L, rel["hard"], i)
                ex, fl = gates_of(c, ucnt[k][i])
                rows.append({"seed": k, "R_local": ir["recovery"]["auc"]["v"], "gate_ok": all(v >= 0 for v in ex.values()),
                             "gate_ok_float": all(v >= 0 for v in fl.values()), "worst_margin": min(fl.values()),
                             "util_match": abs(ir["utility"]["acc"] - c["k"] / c["n"]) == 0})
            if rows is None:
                by = None
                break
            by[cid] = {"gate_ok_all": all(r["gate_ok"] for r in rows), "gate_ok_all_float": all(r["gate_ok_float"] for r in rows),
                       "mean_R_local": sum(r["R_local"] for r in rows) / len(rows),
                       "min_margin": min(r["worst_margin"] for r in rows), "utility_records_match": all(r["util_match"] for r in rows)}
        if by is None:
            out[i] = {"status": "MISSING"}
            continue
        ok = [c for c, v in by.items() if v["gate_ok_all"]]
        if ok:
            sel = min(ok, key=lambda c: (by[c]["mean_R_local"], c))
            out[i] = {"status": "SELECTED", "config": sel, "by_config": by}
        else:
            sel = max(by, key=lambda c: (by[c]["min_margin"], -c))
            out[i] = {"status": "NO_FEASIBLE_CONFIGURATION", "config": sel, "by_config": by}
    return out


def replay_selection(D: Data, L, seeds=SEEDS, spot_refit=True):
    ids = bank_ids(jload(RES / "TRAINING_PROTOCOL_LOCK.json")["protocol"]["bank"])
    rel_cache, inner_checks = {}, {}

    def rel_of(u):
        if u not in rel_cache:
            rel_cache[u] = np.load(UNITS / u / "release.npz", allow_pickle=False)
        return rel_cache[u]

    ucnt = {}
    for k in seeds:
        z = rel_of(rel_name(k, "U"))
        ucnt[k] = {i: counts_of(D, L, z[f"hard{i + 1}"], i) for i in (0, 1)}
    u_valid_exact = all(gates_of(ucnt[k][i], ucnt[k][i])[0]["G3"] >= 0 for k in seeds for i in (0, 1))
    u_valid_float = all(ucnt[k][i]["k"] / ucnt[k][i]["n"] - ucnt[k][i]["k_const"] / ucnt[k][i]["n"] >= 0.03
                        for k in seeds for i in (0, 1))
    fsel = fare_selection(D, L, ucnt, seeds)
    rows, util_mismatch, missing = {}, [], []

    def seed_point(k, cid):
        if cid in ("E", "F", "F0"):
            if cid == "E":
                unit, inner_name = f"lc__s{k}__E", f"inner__lc__s{k}__E"
                z = rel_of(unit)
                hards = (z["hard1"], z["hard2"])
            else:
                if fsel[0].get("config") is None or fsel[1].get("config") is None:
                    return None
                a, b = fsel[0]["config"], fsel[1]["config"]
                tag = f"F__c{a}_c{b}" if cid == "F" else f"F0__Z{a}_Z{b}"
                unit, inner_name = f"fare__s{k}__{tag}", f"inner__fare__s{k}__{tag}"
                pu = [f"fare__s{k}__p0__{'c' if cid == 'F' else 'Z'}{a}", f"fare__s{k}__p1__{'c' if cid == 'F' else 'Z'}{b}"]
                if not all(unit_done(x) for x in pu):
                    return None
                hards = (rel_of(pu[0])["hard"], rel_of(pu[1])["hard"])
        else:
            unit, inner_name = rel_name(k, cid), f"inner__{rel_name(k, cid)}"
            if not unit_done(unit):
                return None
            z = rel_of(unit)
            hards = (z["hard1"], z["hard2"])
        ir = unit_rec(inner_name)
        if ir is None:
            return None
        if inner_name not in inner_checks:
            V = release_views(rel_of(unit)) if spot_refit and unit_done(unit) and "r1" in rel_of(unit).files else None
            inner_checks[inner_name] = inner_rec_check(inner_name, D, V, L, refit=("LR_C1", "DA_LR") if V is not None else ())
        cnt = {i: counts_of(D, L, hards[i], i) for i in (0, 1)}
        for i in (0, 1):
            u = ir["utility"][str(i)]
            if u["acc"] != cnt[i]["k"] / cnt[i]["n"] or u["const_acc"] != cnt[i]["k_const"] / cnt[i]["n"]:
                util_mismatch.append(f"{inner_name} task {i}")
        gm = {i: gates_of(cnt[i], ucnt[k][i]) for i in (0, 1)}
        ex = {i: gm[i][0] for i in (0, 1)}
        fl = {i: gm[i][1] for i in (0, 1)}
        short = sum(max(0.0, -x) for i in fl for x in fl[i].values())
        return {"seed": k, "unit": unit, "auc": {w: float(ir["recovery"]["auc"][w]) for w in ("v1", "v2", "pair")},
                "counts": cnt, "gates_float": fl, "gate_shortfall": short, "task_feasible": short == 0.0,
                "task_feasible_exact": all(v >= 0 for i in ex for v in ex[i].values()),
                "acc": {i: cnt[i]["k"] / cnt[i]["n"] for i in (0, 1)}}

    for cid in ids + ["E", "F", "F0"]:
        sp = {k: seed_point(k, cid) for k in seeds}
        if any(v is None for v in sp.values()):
            missing.append(cid)
            continue
        rows[cid] = {"config": cid, "seeds": sp, "mean_pair": sum(s["auc"]["pair"] for s in sp.values()) / len(seeds),
                     "task_feasible": all(s["task_feasible"] for s in sp.values()),
                     "task_feasible_exact": all(s["task_feasible_exact"] for s in sp.values()),
                     "gate_shortfall": sum(s["gate_shortfall"] for s in sp.values())}
    if missing:
        return {"missing": missing}
    # reference statuses (recomputed): E/F0 NOMINEE iff task-feasible on every seed; F NOMINEE iff both purposes have a
    # feasible configuration and F is task-feasible on every seed
    ref_status = {"E": "NOMINEE" if rows["E"]["task_feasible"] else "INFEASIBLE_CONTROL",
                  "F0": "NOMINEE" if rows["F0"]["task_feasible"] else "INFEASIBLE_CONTROL",
                  "F": "NOMINEE" if (fsel[0]["status"] == fsel[1]["status"] == "SELECTED" and rows["F"]["task_feasible"])
                  else "NO_FEASIBLE_NOMINEE"}
    for c in ("F", "F0"):                 # the reference path's own feasibility status is binding (per seed and row)
        if ref_status[c] != "NOMINEE":
            rows[c]["task_feasible"] = False
            rows[c]["reference_status"] = ref_status[c]
            for sp in rows[c]["seeds"].values():
                sp["task_feasible"] = False
    fam = lambda c: c if c in ("U", "E", "F", "F0") else c.split("|")[0]      # noqa: E731

    def excess(r, guards):
        return {g: {k: {w: r["seeds"][k]["auc"][w] - (gr["seeds"][k]["auc"][w] + 0.005) for w in ("v1", "v2")}
                    for k in seeds} for g, gr in guards.items()}

    def pick(cands, guards, tie):
        ev = []
        for r in cands:
            ge = excess(r, guards)
            gok = all(x <= 0 for g in ge.values() for kk in g.values() for x in kk.values())
            ns = r["gate_shortfall"] + sum(max(0.0, x) for g in ge.values() for kk in g.values() for x in kk.values())
            ev.append({**r, "guard_excess": ge, "guard_ok": gok, "eligible": r["task_feasible"] and gok,
                       "nomination_shortfall": ns})
        el = [r for r in ev if r["eligible"]]
        if el:
            b = min(el, key=lambda r: (r["mean_pair"],) + tie(r))
            near = [r["config"] for r in el if r is not b and abs(r["mean_pair"] - b["mean_pair"]) <= 1e-12]
            return {"status": "NOMINEE", "config": b["config"], "row": b, "evaluated": ev, "near_ties": near}
        b = min(ev, key=lambda r: (r["nomination_shortfall"], r["mean_pair"]) + tie(r))
        return {"status": "NO_FEASIBLE_NOMINEE", "config": None, "descriptive_config": b["config"], "row": b,
                "evaluated": ev}

    st = {}
    if not u_valid_exact:
        st = {x: {"status": "INVALID", "config": None} for x in ("L*", "C*", "N*", "R*")}
    else:
        loc = [rows[c] for c in ids if fam(c) in ("RAW-L", "NORM-L")]
        st["L*"] = pick(loc, {}, lambda r: (r["config"],))   # measured compute is identical across protected arms
        ctl = [rows[c] for c in ids if fam(c) in ("RAW-J", "RAW-L", "NORM-L", "U")] + [rows[c] for c in ("E", "F", "F0")]
        st["C*"] = pick(ctl, {}, lambda r: (r["config"],))
        gL = {"L*": rows[st["L*"]["config"]]} if st["L*"]["status"] == "NOMINEE" else {}
        gN = dict(gL)
        if st["C*"]["status"] == "NOMINEE":
            gN["C*"] = rows[st["C*"]["config"]]
        st["N*"] = pick([rows[c] for c in ids if fam(c) == "NORM-J"], gN,
                        lambda r: (parse_cid(r["config"])["rho"], abs(parse_cid(r["config"])["a"] - 1), r["config"]))
        st["R*"] = pick([rows[c] for c in ids if fam(c) == "RAW-J"], gL,
                        lambda r: (parse_cid(r["config"])["beta"], r["config"]))
        for x, need in (("N*", ("L*", "C*")), ("R*", ("L*",))):
            miss = [g for g in need if (g == "L*" and not gL) or (g == "C*" and "C*" not in gN)]
            st[x]["missing_guards"] = miss
            if miss and st[x]["status"] == "NOMINEE":
                st[x] = {**st[x], "status": "INVALID_MISSING_COMPARATOR", "descriptive_config": st[x]["config"], "config": None}
    valid = [(rows[s["config"]]["mean_pair"], s["config"], x) for x, s in st.items()
             if x in ("N*", "R*", "C*") and s["status"] == "NOMINEE"]
    best = min(valid) if valid else None
    deploy = {"config": best[1], "as": best[2]} if best else {"config": "U", "as": "truthful baseline"}
    disagree = sorted({c for c, r in rows.items() if r["task_feasible"] != r["task_feasible_exact"]
                       and "reference_status" not in r})
    return {"ids": ids, "rows": rows, "statuses": st, "deployable_best": deploy, "u_valid": u_valid_exact,
            "u_valid_float": u_valid_float, "fare_selection": fsel, "reference_status": ref_status,
            "utility_record_mismatches": util_mismatch, "inner_checks": inner_checks,
            "float_vs_exact_feasibility_disagreements": disagree}


def check_selection(D: Data, L):
    if not (RUN / "selection.json").exists():
        return res("PENDING", reason="run/selection.json not written"), None
    mine = replay_selection(D, L)
    if "missing" in mine:
        return res("PENDING", reason=f"inputs missing for {mine['missing']}"), None
    diffs = []
    S = jload(RUN / "selection.json")
    for x, s in mine["statuses"].items():
        t = S["statuses"].get(x, {})
        mc = s.get("config") or s.get("descriptive_config")
        tc = t.get("config") or t.get("descriptive_config")
        if t.get("status") != s["status"] or mc != tc:
            diffs.append(f"selection.json {x}: {t.get('status')}/{tc} vs {s['status']}/{mc}")
    if S.get("U_valid") != mine["u_valid"]:
        diffs.append("U_valid")
    if (S.get("deployable_best") or {}).get("config") != mine["deployable_best"]["config"]:
        diffs.append(f"deployable best {S.get('deployable_best')} vs {mine['deployable_best']}")
    nrow = 0
    for cid, r in mine["rows"].items():
        t = S["rows"].get(cid)
        if t is None:
            diffs.append(f"row {cid} missing")
            continue
        nrow += 1
        if abs(t["mean_pair"] - r["mean_pair"]) > 1e-15 or t["task_feasible"] != r["task_feasible"] \
                or abs(t["gate_shortfall"] - r["gate_shortfall"]) > 1e-12:
            diffs.append(f"row {cid}: mean_pair/feasible/shortfall")
        for k, sp in r["seeds"].items():
            ts = t["seeds"].get(str(k)) or t["seeds"].get(k)
            if any(ts["auc"][w] != sp["auc"][w] for w in ("v1", "v2", "pair")):
                diffs.append(f"row {cid} seed {k}: auc")
            if ts["task_feasible"] != sp["task_feasible"]:
                diffs.append(f"row {cid} seed {k}: task_feasible {ts['task_feasible']} vs {sp['task_feasible']}")
            for i in (0, 1):
                tg = ts["gate_margins"].get(str(i)) or ts["gate_margins"].get(i)
                if any(abs(tg[gk] - sp["gates_float"][i][gk]) > 1e-12 for gk in ("G1", "G2", "G3")):
                    diffs.append(f"row {cid} seed {k} task {i}: gate margins")
    for x in ("N*", "R*"):
        for r in mine["statuses"][x].get("evaluated", []):
            t = (S.get("nomination") or {}).get(r["config"])
            if t is None:
                continue
            if t["guard_ok"] != r["guard_ok"] or t["eligible"] != r["eligible"] or \
                    abs(t["nomination_shortfall"] - r["nomination_shortfall"]) > 1e-12:
                diffs.append(f"nomination {r['config']}")
    # reference selection file of the audit owner
    rs = RUN / "references_selection.json"
    ref_cmp = {}
    if rs.exists():
        RS = jload(rs)
        for i in (0, 1):
            t, m = RS["per_purpose"][str(i)], mine["fare_selection"][i]
            ref_cmp[f"purpose_{i}"] = {"status": [t["status"], m["status"]], "config": [t["config"], m.get("config")]}
            if t["status"] != m["status"] or t["config"] != m.get("config"):
                diffs.append(f"FARE purpose {i}: {t['status']}/{t['config']} vs {m['status']}/{m.get('config')}")
            for cid, v in m.get("by_config", {}).items():
                tv = t["by_config"].get(str(cid))
                if tv and (tv["gate_ok_all"] != v["gate_ok_all"] or abs(tv["mean_R_local"] - v["mean_R_local"]) > 1e-15
                           or abs(tv["min_margin"] - v["min_margin"]) > 1e-12):
                    diffs.append(f"FARE purpose {i} config {cid}")
        for arm in ("E", "F", "F0"):
            ref_cmp[arm] = [RS["arms"][arm]["status"], mine["reference_status"][arm]]
            if RS["arms"][arm]["status"] != mine["reference_status"][arm]:
                diffs.append(f"reference status {arm}: {RS['arms'][arm]['status']} vs {mine['reference_status'][arm]}")
    # public files
    pub = {}
    if (RES / "SELECTION.json").exists():
        P = jload(RES / "SELECTION.json")
        bad = [x for x, s in mine["statuses"].items() if P["statuses"][x]["status"] != s["status"] or
               (P["statuses"][x].get("config") or P["statuses"][x].get("descriptive_config")) != (s.get("config") or s.get("descriptive_config"))]
        if P["deployable_best"]["config"] != mine["deployable_best"]["config"]:
            bad.append("deployable_best")
        pub["SELECTION.json"] = res("FAIL" if bad else "PASS", differences=bad)
        diffs += [f"SELECTION.json {b}" for b in bad]
    if (RES / "SELECTION_TABLE.csv").exists():
        bad = []
        for r in csv.DictReader(open(RES / "SELECTION_TABLE.csv")):
            m = mine["rows"].get(r["config"])
            if m is None:
                bad.append(f"{r['config']} unknown")
                continue
            if abs(float(r["mean_auc_pair"]) - m["mean_pair"]) > 5.1e-7 or (r["task_feasible_all_seeds"] == "True") != m["task_feasible"]:
                bad.append(r["config"])
        pub["SELECTION_TABLE.csv"] = res("FAIL" if bad else "PASS", differences=bad)
        diffs += [f"SELECTION_TABLE {b}" for b in bad]
    if (RES / "INNER_FRONTIERS.csv").exists():
        bad, n = [], 0
        for r in csv.DictReader(open(RES / "INNER_FRONTIERS.csv")):
            m = mine["rows"].get(r["config"])
            if m is None:
                continue
            sp = m["seeds"][int(r["seed"])]
            n += 1
            vals = {"auc_pair": sp["auc"]["pair"], "auc_v1": sp["auc"]["v1"], "auc_v2": sp["auc"]["v2"],
                    "acc_income": sp["acc"][0], "acc_occ": sp["acc"][1]}
            if any(abs(float(r[c]) - v) > 5.1e-7 for c, v in vals.items()) or (r["seed_task_feasible"] == "True") != sp["task_feasible"]:
                bad.append(f"{r['config']} s{r['seed']}")
        pub["INNER_FRONTIERS.csv"] = res("FAIL" if bad else "PASS", rows=n, differences=bad[:20])
        diffs += [f"INNER_FRONTIERS {b}" for b in bad[:20]]
    el = {}
    if (RES / "EVALUATION_LOCK.json").exists():
        EL = jload(RES / "EVALUATION_LOCK.json")
        bad = []
        for x, s in mine["statuses"].items():
            t = EL["statuses"][x]
            if t["status"] != s["status"] or (t.get("config") or t.get("descriptive_config")) != (s.get("config") or s.get("descriptive_config")):
                bad.append(f"status {x}")
            if EL["resolved"][x] != (s.get("config") or s.get("descriptive_config")):
                bad.append(f"resolved {x}")
        if EL["deployable_best"]["config"] != mine["deployable_best"]["config"] or EL["U_valid"] != mine["u_valid"]:
            bad.append("deployable/U_valid")
        for k in SEEDS:
            sc = EL["seeds"][str(k)]["score"]
            want = {cid: rel_name(k, cid) for cid in mine["ids"]}
            for cid, u in want.items():
                if (sc.get(cid) or {}).get("unit") != u:
                    bad.append(f"seed {k} {cid} unit")
            for lab in ("E", "F", "F0"):
                spec = sc.get(lab) or {}
                m = mine["rows"][lab]["seeds"][k]["unit"]
                if lab == "E" and spec.get("unit") != m:
                    bad.append(f"seed {k} E unit")
                if lab in ("F", "F0"):
                    a, b = mine["fare_selection"][0]["config"], mine["fare_selection"][1]["config"]
                    t = "c" if lab == "F" else "Z"
                    if list(spec.get("units") or []) != [f"fare__s{k}__p0__{t}{a}", f"fare__s{k}__p1__{t}{b}"]:
                        bad.append(f"seed {k} {lab} units")
            rsl = EL["seeds"][str(k)].get("reference_status") or {}
            if any(rsl.get(a) != mine["reference_status"][a] for a in ("E", "F", "F0")):
                bad.append(f"seed {k} reference statuses {rsl}")
        el = res("FAIL" if bad else "PASS", differences=bad)
        diffs += [f"EVALUATION_LOCK {b}" for b in bad]
    inner_fail = sorted(n for n, v in mine["inner_checks"].items() if v and v["status"] != "PASS")
    spot = [v["own_refit_max_abs_diff"] for v in mine["inner_checks"].values() if v and v.get("own_refit_max_abs_diff") is not None]
    if mine["utility_record_mismatches"]:
        diffs.append(f"inner utility records differ from own counts: {mine['utility_record_mismatches'][:5]}")
    stx = mine["statuses"]
    summary = {x: {"status": s["status"], "config": s.get("config"), "descriptive_config": s.get("descriptive_config"),
                   "mean_inner_pair_auc": s["row"]["mean_pair"] if s.get("row") else None,
                   "nomination_shortfall": s["row"].get("nomination_shortfall") if s.get("row") else None,
                   "near_ties": s.get("near_ties"), "missing_guards": s.get("missing_guards")} for x, s in stx.items()}
    elig = {x: {r["config"]: {"task_feasible": r["task_feasible"], "guard_ok": r["guard_ok"],
                              "nomination_shortfall": r["nomination_shortfall"], "mean_pair": r["mean_pair"]}
                for r in stx[x].get("evaluated", [])} for x in ("N*", "R*") if x in stx}
    table = {c: {"mean_pair": r["mean_pair"], "task_feasible": r["task_feasible"], "task_feasible_exact": r["task_feasible_exact"],
                 "gate_shortfall": r["gate_shortfall"], "reference_status": r.get("reference_status")}
             for c, r in mine["rows"].items()}
    st = "FAIL" if (diffs or inner_fail) else "PASS"
    return res(st, statuses=summary, deployable_best=mine["deployable_best"], U_valid=mine["u_valid"],
               reference_status=mine["reference_status"],
               fare_purpose_selection={i: {"status": v["status"], "config": v.get("config")} for i, v in mine["fare_selection"].items()},
               nomination_detail=elig, config_table=table,
               float_vs_exact_feasibility_disagreements=mine["float_vs_exact_feasibility_disagreements"],
               inner_records_checked=len(mine["inner_checks"]), inner_record_failures=inner_fail,
               own_LR_C1_DA_LR_refits=len(spot), own_refit_max_abs_diff=max(spot, default=None),
               reference_selection_comparison=ref_cmp, public_files=pub, evaluation_lock=el, differences=diffs[:40],
               rows_compared=nrow), mine


# ------------------------------------------------------------------------------------------------ PHASE 2: lock order
def lock_commit_state():
    rel = f"{REL_RES}/EVALUATION_LOCK.json"
    p = RES / "EVALUATION_LOCK.json"
    if not p.exists():
        return {"exists": False, "verified": False}
    c = git("log", "-1", "--format=%H", "--", rel)
    blob_ok = bool(c) and git("rev-parse", f"{c}:{rel}") == git("hash-object", str(p))
    pushed = bool(c) and f"origin/{BRANCH}" in (git("branch", "-r", "--contains", c) or "").split()
    fp, _ = first_remote(c, remote_reflog()) if c else (None, None)
    return {"exists": True, "commit": c, "worktree_equals_commit": blob_ok, "pushed": pushed, "first_push_time": fp,
            "sha256": sha_file(p), "verified": bool(c and blob_ok and pushed)}


def check_assessment_order(D: Data, inv):
    ls = lock_commit_state()
    if not ls["exists"]:
        return res("PENDING", reason="EVALUATION_LOCK.json not written", lock=ls)
    EL = jload(RES / "EVALUATION_LOCK.json")
    fails, info = [], {"lock": ls}
    ev = activity()
    starts = [parse_iso(e["at"]) for e in ev if e.get("event") in ("start assess", "start outer", "start infer",
                                                                  "assessment opened")]
    info["first_assess_start"] = min(starts) if starts else None
    outer = sorted(n for n in inv if n.startswith("outer__") and unit_done(n))
    ustart, recs_bad = [], []
    for n in outer:
        r = unit_rec(n)
        t = utc((UNITS / n / "COMPLETE.json").stat().st_mtime)
        ustart.append(datetime.fromtimestamp(t.timestamp() - float(r.get("wall_s") or 0), timezone.utc))
        el = r.get("evaluation_lock") or {}
        if el.get("commit") != ls["commit"] or el.get("sha256") != ls["sha256"]:
            recs_bad.append(n)
    logs = [utc(getattr(p.stat(), "st_birthtime", p.stat().st_mtime)) for p in RUN.glob("work_assess*.log")]
    info["first_outer_unit_est_start"] = min(ustart) if ustart else None
    info["assess_logs_created"] = min(logs) if logs else None
    info["outer_units_complete"] = len(outer)
    info["outer_records_with_other_lock"] = recs_bad
    first = min([t for t in [info["first_assess_start"], info["first_outer_unit_est_start"], info["assess_logs_created"]] if t],
                default=None)
    info["first_assessment_activity"] = first
    if not ls["verified"]:
        fails.append("EVALUATION_LOCK not committed/unchanged/pushed")
    if first is not None and (ls["first_push_time"] is None or ls["first_push_time"] > first):
        fails.append("assessment activity before the lock push")
    if recs_bad:
        fails.append(f"{len(recs_bad)} outer records cite another lock commit/hash")
    # pins
    bad = []
    n_units = 0
    for k, sk in EL["seeds"].items():
        for u, files in sk["unit_file_sha256"].items():
            n_units += 1
            cp = UNITS / u / "COMPLETE.json"
            if not cp.exists() or jload(cp).get("files") != files:
                bad.append(u)
        scored = set()
        for spec in sk["score"].values():
            scored |= set([spec["unit"]] if spec.get("unit") else spec.get("units") or [])
        if scored - set(sk["unit_file_sha256"]):
            bad.append(f"seed {k}: unpinned scored units")
    for grp in ("locks_sha256", "amendments_sha256"):
        for f_, h in (EL.get(grp) or {}).items():
            if not (RES / f_).exists() or sha_file(RES / f_) != h:
                bad.append(f"{grp}:{f_}")
    if sha_file(RUN / "selection.json") != EL["selection_sha256"] or sha_file(RES / "SELECTION.json") != EL["selection_public_sha256"]:
        bad.append("selection hashes")
    am = D.mask[ASSESS]
    ar = EL["assessment_role"]
    if ar["rows"] != int(am.sum()) or ar["groups"] != len(np.unique(D.unit[am])) or ar["row_id_sha256"] != rowid_hash(D.row_id[am]):
        bad.append("assessment_role")
    Ls = D.labels()
    cnt = np.bincount(Ls["sex"][D.mask[FIT]], minlength=2).astype(np.int64)
    info["sex_prior_hash_matches_lock"] = hashlib.sha256(cnt.tobytes()).hexdigest() == EL["sex_prior_defense_fit_sha256"]
    if not info["sex_prior_hash_matches_lock"]:
        bad.append("sex prior hash")
    changed = [f_ for f_, h in EL["locked_code_files"].items() if not (WT / f_).exists() or sha_file(WT / f_) != h]
    info["locked_code_files_changed_since_lock"] = changed
    info["amendments_after_lock"] = sorted(p.name for p in RES.glob("AMENDMENT_A*.json") if p.name not in EL["amendments_sha256"])
    info["units_pinned"] = n_units
    info["pin_failures"] = bad
    if bad:
        fails.append("lock pins")
    chain = [f_ for f_ in ("osf/assess.py", "osf/audit.py", "osf/baselines.py", "osf/data.py", "smf/audit.py", "smf/data.py",
                           "jcv/audit.py", "jcv/finalize.py", "rgj/finalize.py", "rgj/data.py") if f_ in changed]
    if chain:
        fails.append(f"scoring-chain code changed after the lock: {chain}")
    st = "FAIL" if fails else ("WARN" if changed or info["amendments_after_lock"] else "PASS")
    return res(st, failures=fails, **info)


# ------------------------------------------------------------------------------------------------ PHASE 2: endpoints
class LazyNpz:
    """Read-on-demand view of a private preds.npz (only the arrays an endpoint needs are decompressed and kept)."""

    def __init__(self, path):
        self.z = np.load(path, allow_pickle=False)
        self.files = list(self.z.files)
        self.c = {}

    def __getitem__(self, k):
        if k not in self.c:
            self.c[k] = self.z[k]
        return self.c[k]

    def __contains__(self, k):
        return k in self.files


class Boot:
    """Exact-record-group multinomial bootstrap: groups = sorted unique assess_unit; per replicate
    rng.multinomial(n_groups, uniform) drawn sequentially from default_rng(seed), produced in chunks."""

    def __init__(self, assess_unit, B=B_BOOT, seed=BOOT_SEED, chunk=CHUNK):
        self.groups, self.ginv = np.unique(np.asarray(assess_unit), return_inverse=True)
        G = len(self.groups)
        rng = np.random.default_rng(seed)
        p = np.full(G, 1.0 / G)
        self.counts = np.stack([rng.multinomial(G, p).astype(np.int32) for _ in range(B)])
        self.B, self.G, self.chunk = B, G, chunk
        self.draws_sha256 = hashlib.sha256(self.counts.tobytes()).hexdigest()

    def chunks(self):
        for b0 in range(0, self.B, self.chunk):
            yield self.counts[b0:b0 + self.chunk][:, self.ginv].astype(np.float64)


class AUCPrep:
    def __init__(self, score, y):
        order = np.argsort(score, kind="mergesort")
        s = np.asarray(score)[order]
        self.order = order
        self.ypos = np.asarray(y)[order].astype(bool)
        self.starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])

    def __call__(self, W):
        Wo = W[:, self.order]
        Wp = np.where(self.ypos[None, :], Wo, 0.0)
        Wn = Wo - Wp
        Gp = np.add.reduceat(Wp, self.starts, axis=1)
        Gn = np.add.reduceat(Wn, self.starts, axis=1)
        below = np.cumsum(Gn, axis=1) - Gn
        num = (Gp * (below + 0.5 * Gn)).sum(1)
        den = Gp.sum(1) * Gn.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, np.nan)


def endpoint_engine(preds, labels_by_role, statuses, prim, sec, boot, ll_const, const_correct, levels_labels=()):
    """Own endpoint computation: per-seed levels (AUC averaged over the 3 attacker seeds, accuracy, constant accuracy,
    LLR), per-seed paired statistics, mean over seeds, bootstrap replicates with the common group draws."""
    n = len(next(iter(preds.values()))["sex"])
    jobs = {}

    def lab(x):
        return labels_by_role.get(x, x)

    def R(k, la, view, fmt="P_auc"):
        key = ("R", k, la, view)
        if key not in jobs:
            p = preds[(k, la)]
            P = p[f"{fmt}_{view}"]
            jobs[key] = ("auc", [AUCPrep(P[a][:, 1], p["sex"] == 1) for a in range(P.shape[0])])
        return key

    def ACC(k, la, t):
        key = ("acc", k, la, t)
        if key not in jobs:
            p = preds[(k, la)]
            y = p["y_income"] if t == 0 else p["y_occ"]
            jobs[key] = ("mean", (p[f"hard{t + 1}"] == y).astype(np.float64))
        return key

    def CONST(t):
        key = ("const", t)
        if key not in jobs:
            jobs[key] = ("mean", const_correct[t].astype(np.float64))
        return key

    def LLR(k, la, view):
        key = ("LLR", k, la, view)
        if key not in jobs:
            p = preds[(k, la)]
            P = p[f"P_ce_{view}"]
            s = p["sex"]
            jobs[key] = ("llr", [-np.log(np.clip(P[a][np.arange(n), s], 1e-12, 1.0)) for a in range(P.shape[0])])
        return key

    def stat(e):
        kind = e["kind"]
        if kind == "coalition":
            return lambda g, k: g(R(k, lab(e["ref"]), "pair")) - g(R(k, lab(e["nominee"]), "pair"))
        if kind == "local":
            return lambda g, k: g(R(k, lab(e["nominee"]), e["view"])) - g(R(k, lab(e["ref"]), e["view"]))
        if kind == "acc":
            return lambda g, k: g(ACC(k, lab(e["nominee"]), e["task"])) - g(ACC(k, "U", e["task"]))
        if kind == "retain":
            return lambda g, k: g(ACC(k, lab(e["nominee"]), e["task"])) - 0.8 * g(ACC(k, "U", e["task"])) - 0.2 * g(CONST(e["task"]))
        if kind == "useful":
            return lambda g, k: g(ACC(k, lab(e["nominee"]), e["task"])) - g(CONST(e["task"]))
        if kind == "rec":
            return lambda g, k: g(R(k, lab(e["a"]), e["view"])) - g(R(k, lab(e["b"]), e["view"]))
        if kind == "accdiff":
            return lambda g, k: g(ACC(k, lab(e["a"]), e["task"])) - g(ACC(k, lab(e["b"]), e["task"]))
        if kind == "synergy":
            return lambda g, k: g(R(k, lab(e["arm"]), "pair")) - np.maximum(g(R(k, lab(e["arm"]), "v1")), g(R(k, lab(e["arm"]), "v2")))
        if kind == "logloss":
            return lambda g, k: g(LLR(k, lab(e["a"]), e["view"])) - g(LLR(k, lab(e["b"]), e["view"]))
        raise ValueError(kind)

    entries = [("primary", e) for e in prim] + [("secondary", e) for e in sec]
    fns = {e["id"]: stat(e) for _, e in entries}
    avail = {}
    for _, e in entries:
        need = [lab(e[x]) for x in ("nominee", "ref", "a", "b", "arm") if x in e]
        avail[e["id"]] = all(x is not None and all((k, x) in preds for k in SEEDS) for x in need)
        if avail[e["id"]]:
            for k in SEEDS:
                fns[e["id"]](lambda key: 0.0, k)
    for la in levels_labels:
        for k in SEEDS:
            for v in ("v1", "v2", "pair"):
                R(k, la, v)
                LLR(k, la, v)
            for t in (0, 1):
                ACC(k, la, t)
    for t in (0, 1):
        CONST(t)
    ones = np.ones((1, n))
    point, reps = {}, {k: [] for k in jobs}
    ll_c = ll_const

    def evalj(j, W):
        if j[0] == "auc":
            return np.mean([p(W) for p in j[1]], axis=0)
        if j[0] == "mean":
            return (W @ j[1]) / W.sum(1)
        d = W @ ll_c
        return np.mean([np.where(d > 0, 1.0 - (W @ v) / d, np.nan) for v in j[1]], axis=0)

    for key, j in jobs.items():
        point[key] = float(evalj(j, ones)[0])
    for W in boot.chunks():
        for key, j in jobs.items():
            reps[key].append(evalj(j, W))
    vals = {k: (point[k], np.concatenate(v)) for k, v in reps.items()}
    out = {}
    zp = float(norm.ppf(1 - 0.05 / (2 * len(prim))))
    zs = float(norm.ppf(1 - 0.05 / (2 * len(sec))))
    for fam_name, e in entries:
        if not avail[e["id"]]:
            out[e["id"]] = {"family": fam_name, "point": None,
                            "decision": "NOT_ESTABLISHED" if fam_name == "primary" else "NOT_ESTIMABLE"}
            continue
        pt = float(np.mean([fns[e["id"]](lambda key: vals[key][0], k) for k in SEEDS]))
        rp = np.mean([fns[e["id"]](lambda key: vals[key][1], k) for k in SEEDS], axis=0)
        fin = rp[np.isfinite(rp)]
        se = float(np.std(fin, ddof=1))
        z = zp if fam_name == "primary" else zs
        lo, hi = pt - z * se, pt + z * se
        if e["side"] == "lower>":
            d = "PASS" if lo > e["target"] else "NOT_ESTABLISHED"
        elif e["side"] == "upper<":
            d = "PASS" if hi < e["target"] else "NOT_ESTABLISHED"
        else:
            d = "ABOVE" if lo > e["target"] else ("BELOW" if hi < e["target"] else "NOT_RESOLVED")
        r = {"family": fam_name, "point": pt, "se": se, "lower": lo, "upper": hi, "z": z, "decision": d,
             "n_finite_replicates": int(len(fin)), "alias_of": e.get("alias_of")}
        if fam_name == "primary":
            roles = [e.get("nominee")] + ([e["ref"]] if "ref" in e else [])
            if any(statuses.get(x, {}).get("status") != "NOMINEE" for x in roles):
                r["decision_numeric"], r["decision"] = d, "DESCRIPTIVE_ONLY"
        out[e["id"]] = r
    levels = {}
    for key, (p, rp) in vals.items():
        fr = rp[np.isfinite(rp)]
        levels[key] = {"point": p, "se": float(np.std(fr, ddof=1))}
    return out, levels, {"z_primary": zp, "z_secondary": zs}


def runner_level_key(key):
    if key[0] == "R":
        return f"R#{key[1]}#{key[2]}#prim#{key[3]}"
    if key[0] == "acc":
        return f"acc#{key[1]}#{key[2]}#{key[3]}"
    if key[0] == "LLR":
        return f"LLR#{key[1]}#{key[2]}#{key[3]}"
    if key[0] == "const":
        return f"const#{key[1]}"
    return None


def check_endpoints(D: Data, all_levels=False):
    ls = lock_commit_state()
    if not ls["exists"] or not ls["verified"]:
        return res("PENDING", reason="EVALUATION_LOCK not committed+pushed+unchanged; assessment labels stay sealed", lock=ls), None
    EL = jload(RES / "EVALUATION_LOCK.json")
    labels = list(EL["seeds"]["0"]["score"])
    preds, missing = {}, []
    for k in SEEDS:
        for la in labels:
            u = f"outer__s{k}__{la.replace('*', 'star').replace('/', '_').replace('|', '_').replace(' ', '_')}"
            if unit_done(u):
                preds[(k, la)] = LazyNpz(UNITS / u / "preds.npz")
            else:
                missing.append(u)
    if missing:
        return res("PENDING", reason=f"{len(missing)} outer units missing", missing=missing[:10]), None
    LU = D.labels(unseal=True, why="EVALUATION_LOCK verified by the verifier (committed, unchanged, on origin)")
    am = D.mask[ASSESS]
    fit = D.mask[FIT]
    align = []
    for (k, la), p in preds.items():
        if not np.array_equal(p["assess_row_id"], D.row_id[am]) or not np.array_equal(p["assess_unit"], D.unit[am]):
            align.append(f"{k}/{la}: rows/groups")
        for f_, lk in (("sex", "sex"), ("race", "race"), ("y_income", "y_income"), ("y_occ", "y_occ")):
            if not np.array_equal(p[f_], LU[lk][am]):
                align.append(f"{k}/{la}: {f_}")
        spec = EL["seeds"][str(k)]["score"][la]
        if spec.get("unit"):
            rel = np.load(UNITS / spec["unit"] / "release.npz", allow_pickle=False)
            for hk in ("hard1", "hard2", "p1", "p2"):
                if not np.array_equal(p[hk], rel[hk][am]):
                    align.append(f"{k}/{la}: {hk} != frozen release")
        else:
            for i, uu in enumerate(spec["units"]):
                rel = np.load(UNITS / uu / "release.npz", allow_pickle=False)
                if not (np.array_equal(p[f"hard{i + 1}"], rel["hard"][am]) and np.array_equal(p[f"p{i + 1}"], rel["p"][am])):
                    align.append(f"{k}/{la}: purpose {i} != frozen FARE release")
        relU = np.load(UNITS / EL["seeds"][str(k)]["score"]["U"]["unit"] / "release.npz", allow_pickle=False)
        if not (np.array_equal(p["u_hard1"], relU["hard1"][am]) and np.array_equal(p["u_hard2"], relU["hard2"][am])):
            align.append(f"{k}/{la}: u_hard != U release")
    cst = {t: int(np.argmax(np.bincount((LU["y_income"] if t == 0 else LU["y_occ"])[fit], minlength=KS[t]))) for t in (0, 1)}
    p0 = next(iter(preds.values()))
    if list(p0["const_class"]) != [cst[0], cst[1]]:
        align.append("const_class")
    prior = np.bincount(LU["sex"][fit], minlength=2) / int(fit.sum())
    ll_const = -np.log(prior[LU["sex"][am]])
    const_correct = {0: LU["y_income"][am] == cst[0], 1: LU["y_occ"][am] == cst[1]}
    prim = jload(RES / "PRIMARY_FAMILY.json")["slots"]
    sec = jload(RES / "SECONDARY_FAMILY.json")["slots"]
    resolved = EL["resolved"]
    statuses = EL["statuses"]
    t0 = time.time()
    boot = Boot(D.unit[am])
    mine, levels, zz = endpoint_engine(preds, resolved, statuses, prim, sec, boot, ll_const, const_correct,
                                       levels_labels=labels if all_levels else ())
    wall = time.time() - t0
    zok = abs(zz["z_primary"] - 3.113017) < 5e-7 and len(prim) == 27 and len(sec) == 61
    comp = {}
    inf = RUN / "inference.json"
    if inf.exists():
        I = jload(inf)
        diffs, n = [], 0
        for fam_name in ("primary", "secondary"):
            for r in I[fam_name]:
                m = mine.get(r["id"])
                if m is None:
                    diffs.append(f"{r['id']} unknown")
                    continue
                if r.get("point") is None or m.get("point") is None:
                    if (r.get("point") is None) != (m.get("point") is None) or r["decision"] != m["decision"]:
                        diffs.append(f"{r['id']} estimability")
                    continue
                n += 1
                for f_ in ("point", "se", "lower", "upper"):
                    if abs(r[f_] - m[f_]) > 1e-10:
                        diffs.append(f"{r['id']} {f_} {r[f_]} vs {m[f_]}")
                if r["decision"] != m["decision"] or r.get("decision_numeric") != m.get("decision_numeric"):
                    diffs.append(f"{r['id']} decision {r['decision']}/{r.get('decision_numeric')} vs {m['decision']}/{m.get('decision_numeric')}")
                if abs(r["z"] - m["z"]) > 1e-12:
                    diffs.append(f"{r['id']} z")
        comp["inference.json"] = res("FAIL" if diffs else "PASS", slots_compared=n, differences=diffs[:40], n_differences=len(diffs))
        lv = I.get("levels") or {}
        lb, nl = [], 0
        for key, v in levels.items():
            rk = runner_level_key(key)
            if rk is None or rk not in lv:
                continue
            nl += 1
            if abs(lv[rk]["point"] - v["point"]) > 1e-10 or abs(lv[rk]["se"] - v["se"]) > 1e-10:
                lb.append(f"{rk}: {lv[rk]['point']}/{lv[rk]['se']} vs {v['point']}/{v['se']}")
        comp["inference_levels"] = res("FAIL" if lb else ("PASS" if nl else "INFO"), levels_compared=nl,
                                       differences=lb[:20], n_differences=len(lb))
        for c in "ABC":
            comp[f"claim{c}_runner"] = I.get(f"claim{c}")
        comp["label_runner"] = I.get("label")
        comp["B_seed_runner"] = [I.get("B"), I.get("seed"), I.get("n_assessment"), I.get("n_groups")]
    else:
        comp["inference.json"] = res("PENDING")
    for name, fam_name in (("PRIMARY_ENDPOINTS.csv", "primary"), ("SECONDARY_ENDPOINTS.csv", "secondary")):
        p = RES / name
        if not p.exists():
            comp[name] = res("PENDING")
            continue
        diffs = []
        for r in csv.DictReader(open(p)):
            m = mine.get(r["id"])
            if m is None or m.get("point") is None:
                continue
            for f_ in ("point", "se", "lower", "upper"):
                if abs(float(r[f_]) - m[f_]) > 5.1e-7:
                    diffs.append(f"{r['id']} {f_}")
            if r["decision"] != m["decision"]:
                diffs.append(f"{r['id']} decision {r['decision']} vs {m['decision']}")
        comp[name] = res("FAIL" if diffs else "PASS", differences=diffs[:30])
    aliases = {e["id"]: e.get("alias_of") for e in prim if e.get("alias_of")}
    alias_ok = all(mine[a]["point"] == mine[b]["point"] and mine[a]["se"] == mine[b]["se"]
                   for a, b in aliases.items() if mine[a].get("point") is not None and mine[b].get("point") is not None)
    st = worst("FAIL" if align else "PASS", "PASS" if zok else "FAIL", "PASS" if alias_ok else "FAIL",
               *[v["status"] for v in comp.values() if isinstance(v, dict) and "status" in v])
    brief = {i: {kk: v.get(kk) for kk in ("point", "se", "lower", "upper", "decision", "decision_numeric")} for i, v in mine.items()}
    nonnom = [x for x, v in statuses.items() if v.get("status") != "NOMINEE"]
    sec_desc = {e["id"]: {"roles": [e[x] for x in ("a", "b", "arm") if x in e and e[x] in nonnom],
                          "resolved_to": [resolved[e[x]] for x in ("a", "b", "arm") if x in e and e[x] in nonnom],
                          "decision": mine[e["id"]]["decision"]}
                for e in sec if any(e.get(x) in nonnom for x in ("a", "b", "arm"))}
    return res(st, lock=ls, n_assessment=int(am.sum()), n_groups=boot.G, B=boot.B, boot_seed=BOOT_SEED,
               draws_sha256=boot.draws_sha256, z=zz, z_ok=zok, alignment_failures=align[:20], alias_slots_identical=alias_ok,
               comparisons=comp, endpoints=brief, wall_s=wall, levels_computed=len(levels),
               secondary_slots_scoring_a_non_nominee_role=res(
                   "INFO", slots=sec_desc,
                   note="these secondary contrasts score the DESCRIPTIVE fallback configuration of a NO_FEASIBLE_NOMINEE "
                        "role (EVALUATION_LOCK 'resolved'); the runner and this verifier agree on the numbers, but their "
                        "ABOVE/BELOW labels describe the fallback configuration, not a nominee")), mine


def check_conjunctions(mine_end):
    if mine_end is None:
        return res("PENDING", reason="endpoints pending")
    EL = jload(RES / "EVALUATION_LOCK.json")
    prim = jload(RES / "PRIMARY_FAMILY.json")["slots"]
    st = EL["statuses"]
    dec = {}
    for claim, (nom, ref) in {"A": ("N*", "L*"), "B": ("N*", "C*"), "C": ("R*", "L*")}.items():
        ids = [e["id"] for e in prim if e["claim"] == claim]
        all9 = all(mine_end[i]["decision"] == "PASS" for i in ids)
        req = all(st.get(x, {}).get("status") == "NOMINEE" and st.get(x, {}).get("config") for x in (nom, ref))
        dec[claim] = {"all_nine_pass": all9, "status_requirements_met": bool(req),
                      "clauses_passing": sum(mine_end[i]["decision"] == "PASS" for i in ids),
                      "clauses_numeric_pass": sum(mine_end[i].get("decision_numeric", mine_end[i]["decision"]) == "PASS" for i in ids),
                      "decision": "PASS" if (all9 and req) else "NOT_ESTABLISHED"}
    complete = bool(EL.get("U_valid")) and not any(str(s.get("status", "")).startswith("INVALID") for s in st.values())
    labels = []
    if dec["A"]["decision"] == "PASS" and dec["B"]["decision"] == "PASS":
        labels.append("NORM_DEVELOPMENT_CRITERION_MET")
    if dec["C"]["decision"] == "PASS":
        labels.append("RAW_JOINT_DEVELOPMENT_CRITERION_MET")
    overall = "INCOMPLETE_OR_INVALID" if not complete else (" + ".join(labels) if labels else "EXPERIMENTAL_NO_ADVANTAGE")
    diffs = []
    inf = RUN / "inference.json"
    if inf.exists():
        I = jload(inf)
        for c in "ABC":
            t = I.get(f"claim{c}") or {}
            for kk in ("decision", "all_nine_pass", "status_requirements_met", "clauses_passing"):
                if t.get(kk) != dec[c][kk]:
                    diffs.append(f"claim{c}.{kk}: {t.get(kk)} vs {dec[c][kk]}")
        if I.get("label") != overall:
            diffs.append(f"label {I.get('label')} vs {overall}")
    return res("FAIL" if diffs else ("PASS" if inf.exists() else "PENDING"), claims=dec, complete=complete,
               overall=overall, differences=diffs)


def check_attacker_restore(D: Data, label="RAW-J|b0.3", seed=0, views=("pair", "v1")):
    """Refit the AUC-selected final attacker(s) of one outer unit from its record and the frozen release (own slate
    implementation; AUDIT_FIT rows, SEX) at attacker seeds 0, 1, 2 and reproduce the saved assessment probabilities."""
    ls = lock_commit_state()
    if not ls.get("verified"):
        return res("PENDING", reason="EVALUATION_LOCK not verified")
    name = f"outer__s{seed}__{safe(label)}"
    if not unit_done(name):
        return res("PENDING", reason=f"{name} missing")
    rec = unit_rec(name)
    preds = np.load(UNITS / name / "preds.npz", allow_pickle=False)
    rel = np.load(UNITS / rel_name(seed, label) / "release.npz", allow_pickle=False)
    V = release_views(rel)
    LU = D.labels(unseal=True, why="EVALUATION_LOCK verified by the verifier (committed, unchanged, on origin)")
    s = LU["sex"]
    fa, se, am = (np.flatnonzero(D.mask[r]) for r in ("AUDIT_FIT", "INNER_SELECTION", ASSESS))
    out, fails = {"unit": name}, []
    for w in views:
        for crit in ("auc", "ce"):
            sc = rec["primary"]["scored"][w][crit]
            src, att = sc["source_view"], sc["attacker"]
            sel = rec["primary"]["selection"][w][crit]
            bank = (rec["primary"]["tables"]["pair"] + rec["primary"]["tables"]["v1"] + rec["primary"]["tables"]["v2"]
                    if w == "pair" else rec["primary"]["tables"][w])
            key = "inner_auc" if crit == "auc" else "inner_ce"
            best = None
            for j, b in enumerate(bank):
                if best is None or ((b[key] > bank[best][key] + 1e-12) if crit == "auc" else (b[key] < bank[best][key] - 1e-12)):
                    best = j
            e = {"source_view": src, "attacker": att, "bank_index_record": sel.get("bank_index"), "bank_index_own": best,
                 "selection_is_bank_optimum": best == sel.get("bank_index") and bank[best]["attacker"] == att}
            devs, inner_dev = [], None
            for a_seed in (0, 1, 2):
                m = attacker_factory(att, a_seed).fit(V[src][fa], s[fa])
                if a_seed == 0:
                    Pi = proba_of(m, V[src][se], 2)
                    inner_dev = abs(roc_auc_score(s[se] == 1, Pi[:, 1]) - sel["inner_auc"])
                P = proba_of(m, V[src][am], 2)
                devs.append(maxdiff(P, preds[f"P_{crit}_{w}"][a_seed]))
            e["saved_assessment_proba_max_abs_diff_per_attacker_seed"] = devs
            e["inner_auc_abs_diff"] = inner_dev
            e["bit_exact"] = all(d == 0.0 for d in devs)
            if max(devs) > 1e-9 or not e["selection_is_bank_optimum"] or inner_dev > 1e-9:
                fails.append(f"{w}/{crit}")
            out[f"{w}|{crit}"] = e
    out["failures"] = fails
    out["status"] = "FAIL" if fails else "PASS"
    return out


# ------------------------------------------------------------------------------------------------ self tests
def selftests():
    out = {}
    _IN_SELFTEST[0] = True
    tried = []
    for m in ("osf.train", "osf.select", "smf.train", "rgj.train", "jcv.infer", "oar.study", "pnx", "stored_model_eval.pilot_infer"):
        try:
            importlib.import_module(m)
            tried.append((m, False))
        except ImportError:
            tried.append((m, True))
    _IN_SELFTEST[0] = False
    out["import_guard_blocks"] = res("PASS" if all(b for _, b in tried) else "FAIL", tried=tried)
    rng = np.random.default_rng(7)
    # forward pass reproduces torch.nn.Sequential bit-exactly
    seq = torch.nn.Sequential(torch.nn.Linear(83, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(),
                              torch.nn.Linear(64, 16))
    sd = {f"enc.0.{k}": v for k, v in seq.state_dict().items()}
    Xs = rng.normal(size=(50, 83)).astype(np.float32)
    with torch.no_grad():
        ref_ = seq(torch.from_numpy(Xs)).double().numpy()
    out["forward_pass"] = res("PASS" if np.array_equal(encode(sd, 0, Xs), ref_) else "FAIL")
    # allocation RMS identity
    errs = [abs(math.sqrt((alloc(a)[0] ** 2 + alloc(a)[1] ** 2) / 2) - 1) for a in (0.5, 1.0, 2.0)]
    out["allocation_rms"] = res("PASS" if max(errs) < 1e-15 and alloc(1.0) == (1.0, 1.0) and
                                abs(alloc(2.0)[0] / alloc(2.0)[1] - 2) < 1e-15 else "FAIL", max_err=max(errs))
    # bank enumeration
    full = bank_ids("full")
    out["bank"] = res("PASS" if len(full) == 21 and len(bank_ids("reduced")) == 13 else "FAIL", n_full=len(full))
    # one functional update (TASK and NORM) equals a hand computation on a toy net
    torch.manual_seed(0)
    sdm = {}
    for i, K in enumerate(KS):
        enc = torch.nn.Sequential(torch.nn.Linear(83, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(),
                                  torch.nn.Linear(64, 16))
        hd = torch.nn.Linear(16, K)
        sdm.update({f"enc.{i}.{k}": v for k, v in enc.state_dict().items()})
        sdm.update({f"head.{i}.{k}": v for k, v in hd.state_dict().items()})
    n = 300
    TD = {"X": torch.from_numpy(rng.normal(size=(n, 83)).astype(np.float32)),
          "Y": {0: torch.from_numpy(rng.integers(0, 2, n)), 1: torch.from_numpy(rng.integers(0, 6, n))},
          "S": torch.from_numpy(rng.integers(0, 2, n)), "n": n}
    prior = np.bincount(TD["S"].numpy(), minlength=2) / n
    TD.update(prior=prior, H=float(-(prior * np.log(prior)).sum()), logprior=torch.tensor(np.log(prior), dtype=torch.float32))
    bi = np.arange(0, 256)
    net = Net(sdm)
    r, u = one_update(net, {"mode": "TASK", "treat": None}, None, None, None, TD, bi)
    # hand: sum of CE losses -> grads -> clip -> step
    m2 = {k: v.clone().requires_grad_(True) for k, v in sdm.items()}
    loss = 0
    for i in (0, 1):
        h = torch.relu(F.linear(TD["X"][bi], m2[f"enc.{i}.0.weight"], m2[f"enc.{i}.0.bias"]))
        h = torch.relu(F.linear(h, m2[f"enc.{i}.2.weight"], m2[f"enc.{i}.2.bias"]))
        h = F.linear(h, m2[f"enc.{i}.4.weight"], m2[f"enc.{i}.4.bias"])
        loss = loss + F.cross_entropy(F.linear(h, m2[f"head.{i}.weight"], m2[f"head.{i}.bias"]), TD["Y"][i][torch.from_numpy(bi)])
    gs = torch.autograd.grad(loss, [m2[k] for k in Net.ORDER])
    gv = torch.cat([x.reshape(-1) for x in gs])
    kap = min(1.0, CLIP / float(gv.norm()))
    dmax = max(float((net.p[k] - (sdm[k] - LR * kap * gg)).abs().max()) for k, gg in zip(Net.ORDER, gs))
    out["task_update"] = res("PASS" if dmax < 1e-6 and r["zero"] == [5, 5] else "FAIL", max_abs_diff=dmax)
    # NORM update: ratio == rho s_i, direction of p preserved, local pair contributes nothing
    head = {i: (sdm[f"head.{i}.weight"], sdm[f"head.{i}.bias"]) for i in (0, 1)}
    T = {v: Tf(torch.from_numpy(rng.normal(size=(500, DV[v])).astype(np.float32))) for v in VIEWS}
    crit = {v: {kk: new_critic_module(3, v, kk).state_dict() for kk in KINDS} for v in VIEWS}
    okn, okl, n_checked = True, True, 0
    for cfg in ({"mode": "NORM", "treat": "J", "rho": 3.0, "a": 2.0}, {"mode": "NORM", "treat": "L", "rho": 3.0, "a": 0.5}):
        net = Net(sdm)
        r, u = one_update(net, cfg, T, crit, head, TD, bi)
        s_ = alloc(cfg["a"])
        for i in (0, 1):
            if r["zero"][i] == 0 and not r["cap"][i]:
                n_checked += 1
                okn &= abs(r["q_norm"][i] / r["t_norm"][i] - 3.0 * s_[i]) < 1e-5 * 3.0 * s_[i]
            elif r["cap"][i]:
                okn &= r["q_norm"][i] / r["t_norm"][i] < 3.0 * s_[i]
    # local: pair critic changed -> identical update
    crit2 = {v: dict(crit[v]) for v in VIEWS}
    crit2["pair"] = {kk: {a: t * 3.0 for a, t in crit["pair"][kk].items()} for kk in KINDS}
    cfgL = {"mode": "RAW", "treat": "L", "beta": 0.3}
    n1, n2 = Net(sdm), Net(sdm)
    _, u1 = one_update(n1, cfgL, T, crit, head, TD, bi)
    _, u2 = one_update(n2, cfgL, T, crit2, head, TD, bi)
    okl = bool(torch.equal(u1, u2))
    cfgJ = {"mode": "RAW", "treat": "J", "beta": 0.3}
    n3, n4 = Net(sdm), Net(sdm)
    _, u3 = one_update(n3, cfgJ, T, crit, head, TD, bi)
    _, u4 = one_update(n4, cfgJ, T, crit2, head, TD, bi)
    out["norm_ratio_and_local_isolation"] = res("PASS" if okn and n_checked >= 1 and okl and not torch.equal(u3, u4) else "FAIL",
                                                norm_ratio_ok=okn, norm_uncapped_encoders_checked=n_checked, local_update_invariant_to_pair_critic=okl,
                                                joint_update_depends_on_pair_critic=not torch.equal(u3, u4))
    # receipt checker catches a corrupted ratio
    S = EPOCHS * 61
    base = {"epoch": np.arange(S) // 61, "step": np.arange(1, S + 1)}
    out["fingerprint"] = res("PASS" if fp64(np.arange(5, dtype=np.int64)) == fp64(np.arange(5, dtype=np.int64)) and
                             fp64(np.arange(5, dtype=np.int64)) != fp64(np.arange(1, 6, dtype=np.int64)) else "FAIL")
    del base
    # weighted AUC == replicated-rows AUC (ties included); bootstrap draws; Canon drops an exact null direction
    sc = np.round(rng.normal(size=300), 1)
    yy = (rng.random(300) < 0.4).astype(int)
    errs = []
    for _ in range(5):
        w = rng.integers(0, 4, 300)
        errs.append(abs(AUCPrep(sc, yy)(w[None, :].astype(float))[0] - roc_auc_score(np.repeat(yy, w), np.repeat(sc, w))))
    errs.append(abs(AUCPrep(sc, yy)(np.ones((1, 300)))[0] - roc_auc_score(yy, sc)))
    bt = Boot(rng.integers(0, 120, 400), B=60, seed=11, chunk=25)
    rr = np.random.default_rng(11)
    ref_counts = np.stack([rr.multinomial(bt.G, np.full(bt.G, 1.0 / bt.G)) for _ in range(60)])
    Xc = rng.normal(size=(200, 3))
    Xc = np.hstack([Xc, Xc[:, :1] + Xc[:, 1:2]])
    cn = Canon().fit(Xc)
    out["endpoint_primitives"] = res("PASS" if max(errs) < 1e-12 and np.array_equal(bt.counts, ref_counts)
                                     and (bt.counts.sum(1) == bt.G).all() and cn.rank_ == 3 else "FAIL",
                                     weighted_auc_max_err=max(errs), canon_rank=cn.rank_)
    # z values
    z1, z2 = float(norm.ppf(1 - 0.05 / 54)), float(norm.ppf(1 - 0.05 / 122))
    out["z_values"] = res("PASS" if abs(z1 - 3.113017) < 5e-7 else "FAIL", z_primary_27=z1, z_secondary_61=z2)
    return out


# ------------------------------------------------------------------------------------------------ main
def scrub_check(text: str):
    home = str(Path.home())
    probes = [home, "/Users/", "/Volumes/", home.split("/")[-1]]
    low = text.lower()
    bad = [s for s in probes if s and s.lower() in low]
    if bad:
        raise RuntimeError(f"refusing to write identifying strings into the public JSON: {len(bad)} hits")


def walk_flags(checks):
    counts_all, flagged = {}, []

    def walk(node, path):
        if isinstance(node, dict):
            st_ = node.get("status")
            if isinstance(st_, str) and st_ in STATUS_RANK:
                counts_all[st_] = counts_all.get(st_, 0) + 1
                if st_ in ("FAIL", "WARN") and not path.startswith("selftests"):
                    leaf = not any(isinstance(v, dict) and v.get("status") in ("FAIL", "WARN") for v in node.values())
                    if leaf:
                        flagged.append({"path": path, "status": st_, "cause": node.get("reason") or node.get("failures")
                                        or node.get("differences") or node.get("notes")})
            for k, v in node.items():
                if k in ("levels",):
                    continue
                walk(v, f"{path}.{k}" if path else k)

    for k, v in checks.items():
        walk(v, k)
    return counts_all, flagged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest-only", action="store_true")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--releases", default="all", help="all | seed0 | none")
    ap.add_argument("--train-replay", default="default", help="default | none | k:cid;k:cid")
    ap.add_argument("--no-captures", action="store_true")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--phase", type=int, default=1, choices=(1, 2))
    ap.add_argument("--selection", action="store_true", help="replay the inner-only selection (no assessment labels)")
    ap.add_argument("--all-levels", action="store_true", help="phase 2: also every primary-view level of every label")
    args = ap.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(","))
    t0, c0 = time.time(), time.process_time()
    report = {"schema": "osf-independent-verification-v1", "phase": "PHASE_1",
              "generated_at": iso(datetime.now(timezone.utc)),
              "verifier": f"{REL_RES}/verification/replay_osf.py", "verifier_sha256": sha_file(Path(__file__)),
              "worktree_head": git("rev-parse", "HEAD"),
              "inputs": {"source_npz": "<PRIVATE_CACHE>/jcv_v1/inputs/adult_jcv.npz", "units": "<PRIVATE_CACHE>/osf_v1/run/units",
                         "admitted": "<PRIVATE_CACHE>/osf_v1/admitted", "activity_log": "<PRIVATE_CACHE>/osf_v1/run/ACTIVITY_LOG.jsonl"},
              "libraries": {"numpy": np.__version__, "torch": torch.__version__, "joblib": joblib.__version__,
                            "sklearn": __import__("sklearn").__version__, "scipy": __import__("scipy").__version__}}
    checks = {}
    st = selftests()
    checks["selftests"] = {"status": worst(*[v["status"] for v in st.values()]), **st}
    if args.selftest_only:
        print(json.dumps(jsonable(checks["selftests"]), indent=1))
        return
    D = Data()
    L = D.labels()
    inv = {p.name: p for p in sorted(UNITS.iterdir()) if p.is_dir() and not p.name.endswith(".tmp")}
    report["in_progress_unit_dirs_skipped"] = sorted(p.name for p in UNITS.iterdir() if p.name.endswith(".tmp"))
    checks["roles"] = check_roles(D)
    # releases
    rel_units = sorted(n for n in inv if n.startswith("rel__") and unit_done(n) and int(n.split("__")[1][1:]) in seeds)
    if args.releases == "seed0":
        rel_units = [n for n in rel_units if n.startswith("rel__s0__")]
    elif args.releases == "none":
        rel_units = []
    rel_out, adm_out = {}, {}
    for n in rel_units:
        try:
            r, rel = replay_release(D, n, L, refit=True)
        except ImportError as e:
            rel_out[n] = res("FAIL", reason=f"unpickling needed a forbidden module: {e}")
            continue
        rel_out[n] = r
        k = int(n.split("__")[1][1:])
        cid = (unit_rec(n) or {}).get("config")
        if cid in ADMITTED_IDS:
            adm_out[n] = admitted_release_equal(D, n, rel, cid, k)
    expected = [rel_name(k, cid) for k in seeds for cid in bank_ids("full")]
    missing = sorted(set(expected) - set(inv))
    checks["releases"] = res(worst(*[r["status"] for r in rel_out.values()]) if rel_out else "PENDING",
                             units_replayed=len(rel_out), expected_bank_units=len(expected), bank_units_missing=missing,
                             features_bit_exact=sum(1 for r in rel_out.values() if r.get("recipient_1", {}).get("features_bit_exact")
                                                    and r.get("recipient_2", {}).get("features_bit_exact")),
                             refit_heads_bit_exact=sum(1 for r in rel_out.values() for i in (1, 2)
                                                       if r.get(f"recipient_{i}", {}).get("refit_outputs_bit_exact")),
                             refit_heads=sum(1 for r in rel_out.values() for i in (1, 2) if "refit_C" in r.get(f"recipient_{i}", {})),
                             max_refit_c_abs_diff=max((r.get(f"recipient_{i}", {}).get("refit_c_max_abs_diff", 0)
                                                       for r in rel_out.values() for i in (1, 2)), default=None),
                             hard_decision_mismatches=sum(r.get(f"recipient_{i}", {}).get("refit_hard_mismatches", 0)
                                                          for r in rel_out.values() for i in (1, 2)),
                             failing_units=sorted(n for n, r in rel_out.items() if r["status"] != "PASS"),
                             admitted_equal_smf=res(worst(*[v["status"] for v in adm_out.values()]) if adm_out else "PENDING",
                                                    units=adm_out),
                             note="deployment replay uses only X (83 permitted columns) and saved parameters; head refits "
                                  "read OSF_DEFENSE_FIT / HEAD_VALIDATION task labels only",
                             per_unit=rel_out)
    if missing and checks["releases"]["status"] == "PASS":
        checks["releases"]["status"] = "PENDING"
    checks["releases"]["status"] = worst(checks["releases"]["status"], checks["releases"]["admitted_equal_smf"]["status"])
    checks["label_custody"] = check_label_custody(D, inv, rel_out)
    # receipts / RNG
    rc, rg = check_receipts_and_rng(inv, seeds, int(D.mask[FIT].sum()))
    rc["mutation_power"] = receipt_power(int(D.mask[FIT].sum()))
    rc["lead_strength_reports"] = strength_reports_check(seeds)
    rc["status"] = worst(rc["status"], rc["mutation_power"]["status"], rc["lead_strength_reports"]["status"])
    checks["receipts"], checks["common_rng"] = rc, rg
    checks["step_replay"] = check_step_replay(D, inv, seeds, captures=not args.no_captures)
    if args.train_replay == "none":
        which = []
    elif args.train_replay == "default":
        which = [(k, "U") for k in seeds] + [(0, c) for c in ("RAW-J|b0.3", "RAW-J|b0.6", "NORM-J|r3|a1", "NORM-L|r3|a1", "NORM-J|r3|a0.5")
                                            if 0 in seeds]
    else:
        which = [(int(x.split(":")[0]), x.split(":", 1)[1]) for x in args.train_replay.split(";") if x]
    checks["train_replay"] = check_train_replay(D, which)
    checks["replays"] = check_replays(inv, seeds)
    checks["engineering"] = check_engineering(inv, seeds)
    checks["integrity"] = {"complete_json": check_complete(inv), "lock_order": check_lock_order(inv),
                           "code_hashes": check_code_hashes(), "quarantine": check_quarantine(inv)}
    checks["integrity"]["status"] = worst(*[v["status"] for v in checks["integrity"].values()])
    checks["controls"] = check_controls(D, L)
    checks["reference_releases"] = check_reference_releases(D, L, seeds)
    checks["critic_tracking"] = check_tracking(D, L) if tuple(seeds) == SEEDS else res("PENDING", reason="subset of seeds")
    if args.selection:
        checks["selection"], _ = check_selection(D, L)
    else:
        checks["selection"] = res("PENDING", reason="not run in this invocation")
    if args.phase == 2:
        checks["assessment_opening_order"] = check_assessment_order(D, inv)
        ep, mine_end = check_endpoints(D, all_levels=args.all_levels)
        checks["endpoints"] = ep
        checks["conjunctions"] = check_conjunctions(mine_end)
        checks["final_attacker_restore"] = check_attacker_restore(D) if mine_end is not None else res("PENDING")
        report["phase"] = "PHASE_2"
    else:
        for k in ("assessment_opening_order", "endpoints", "conjunctions", "final_attacker_restore"):
            checks[k] = res("PENDING", reason="PHASE 2 (after the lead reports that inference has run)")
    loaded = sorted(m for m in sys.modules if m.split(".")[0] in _FORBIDDEN_TOP)
    report["independence"] = res("PASS" if not loaded and not _BLOCKED and not _PRELOADED else "FAIL",
                                 guard="sys.meta_path finder refusing osf, smf, rgj, jcv, pnx, oar, stored_model_eval "
                                       "(also during joblib unpickling); torch.load(weights_only=True)",
                                 loaded_forbidden_modules=loaded, blocked_attempts_during_run=_BLOCKED,
                                 preloaded_before_guard=_PRELOADED,
                                 libraries_used=["numpy", "scipy", "sklearn", "torch", "joblib", "json", "hashlib", "stdlib"])
    report["checks"] = checks
    report["verifier_corrections"] = VERIFIER_CORRECTIONS
    status = {k: (v.get("status") if isinstance(v, dict) else None) for k, v in checks.items()}
    counts_all, flagged = walk_flags(checks)
    report["summary"] = {"status_by_check": status, "independence": report["independence"]["status"],
                         "overall_phase_1": worst(*[s for k, s in status.items() if k not in
                                                    ("selection", "assessment_opening_order", "endpoints", "conjunctions",
                                                     "final_attacker_restore")], report["independence"]["status"]),
                         "overall": worst(*status.values(), report["independence"]["status"]),
                         "status_counts_all_nodes": counts_all, "flagged_fail_warn": flagged,
                         "wall_s": time.time() - t0, "cpu_s": time.process_time() - c0}
    report["pending"] = sorted(k for k, v in status.items() if v == "PENDING")
    text = json.dumps(jsonable(report), indent=1)
    scrub_check(text)
    if not args.no_write:
        dest = Path(args.out) if args.out else OUT
        tmp = dest.with_suffix(".json.tmp")
        tmp.write_text(text + "\n")
        tmp.replace(dest)
    print(json.dumps(jsonable(report["summary"]), indent=1))


if __name__ == "__main__":
    main()
