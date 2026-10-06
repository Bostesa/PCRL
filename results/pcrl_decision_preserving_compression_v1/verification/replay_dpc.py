#!/usr/bin/env python3
"""Independent verifier for the decision-preserving compression study (dpc).

Owner: the independent verifier (prompt section 3 role 6; section 17). Structure adapted from the predecessor's
independent verifier (results/pcrl_online_strength_frontier_v1/verification/replay_osf.py, itself independent code):
the data/role reconstruction, the forward pass and the lock-chronology helpers follow that file. Everything else is
written from the registered definitions (prompt; PROTOCOL.md; METHOD_CARD.md incl. section 12; ROLE_MANIFEST.json rule
text; the named locks) and the saved private artifacts. The lead's dpc source was READ to learn file formats and the
fixed rules; nothing of it is executed here:

  * a sys.meta_path guard refuses every import under dpc, osf, smf, rgj, jcv, pnx, oar and stored_model_eval (this
    includes imports triggered while unpickling the joblib heads); torch.load is always called with weights_only=True;
    the run asserts at the end that none of these packages is loaded;
  * scientific libraries: numpy, scipy, scikit-learn, torch, joblib (+ json, hashlib and the standard library).

PHASE 1 checks (status PASS / FAIL / WARN / PENDING / INFO):
  roles            exact-rational group-hash recomputation of the old/rgj/smf/osf roles, subroles and the four
                   assessment pools; counts, row-id and group-set hashes, numeric refit and exclusions vs
                   ROLE_MANIFEST.json and the pinned osf role-partition fingerprint
  teachers         own forward pass (per recipient Linear(83,64)-ReLU-Linear(64,64)-ReLU-Linear(64,16)) from the
                   admitted model.pt on the verifier's own X, through the deployed joblib heads (predict_proba on
                   float64 features, numpy first-index argmax) vs every tea__ unit and the admitted releases; custody
                   hashes vs the pinned MODEL_MANIFEST; scaler moments of OSF_DEFENSE_FIT; own deployed-head refit
  references       custody; own LEACE application / own FARE tree traversal + heads vs ref__ units
  fine_partitions  own task-only KL k-means (fixed init / rounds / ties / empty-cell / fallback rules) on
                   OSF_DEFENSE_FIT vs fine.json; own nearest-cell assignment of every row vs assign.npz
  policies         every pol__ unit: assignment partition, prototype sums from the fine-cell sums and the coarse map,
                   smoothing, canonical token IDs, fingerprints (project convention re-implemented), every person's
                   token / decoded probability / decision vs release.npz; objectives D_i, I_i, I_12, F_* from own
                   OSF_DEFENSE_FIT contingency tables vs the receipts
  class_preservation  released decision == own teacher decision on every row of every role; strict prototype argmax;
                   adversarial synthetic rows (ties, one-hot, underflow, uniform, unseen class) through every policy
  search_replay    own greedy agglomeration + exchange refinement (from-scratch objective of every candidate) for
                   every family, compared with every recorded merge / move / sweep and with the final maps
  joint_dominance  JOINT F_joint <= its four bank witnesses (own row-level objectives); witness fingerprints
  sequential       stage-one coefficient 1.5 lambda and identity with a constant second map; frozen first map
  aliases          own fingerprint groups vs OPTIMIZATION_RECEIPTS.json alias groups; aliased releases identical
  integrity        COMPLETE.json hashes; lock-before-stage chronology (remote-tracking reflog, ls-remote,
                   ACTIVITY_LOG, unit completion times); locked code hashes; predictions before fitting
  custody          admitted SHA256SUMS; drive probe by content (names withheld); backup state (expected later)
PHASE 2 checks (inner eligibility, selection, endpoints, conjunctions, attacker replay, controls coverage, composed
source readers, amendments, restore replay) are PENDING until the lead reports that selection, EVALUATION_LOCK,
assessment and inference have run.

Usage (from the worktree root):
    OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python results/pcrl_decision_preserving_compression_v1/verification/replay_dpc.py
        [--selftest-only] [--no-refit] [--search all|sample|none] [--no-write]
Writes results/pcrl_decision_preserving_compression_v1/INDEPENDENT_VERIFICATION.json (aggregates, hashes and
placeholders only).
"""
from __future__ import annotations

import importlib
import importlib.abc
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

# ------------------------------------------------------------------------------------------------ import guard
_FORBIDDEN_TOP = ("dpc", "osf", "smf", "rgj", "jcv", "pnx", "oar", "stored_model_eval")
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

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as TF  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import log_loss  # noqa: E402
from sklearn.pipeline import Pipeline, make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

torch.set_num_threads(1)

# ------------------------------------------------------------------------------------------------ constants
HERE = Path(__file__).resolve().parent
RES = HERE.parent
WT = RES.parents[1]
REL_RES = "results/pcrl_decision_preserving_compression_v1"
OUT = RES / "INDEPENDENT_VERIFICATION.json"
CACHE = Path.home() / "PCRL_eval_cache_private"
SRC = CACHE / "jcv_v1" / "inputs" / "adult_jcv.npz"
SRC_SHA = "e0d9e54af780f30788ee29cfe6795ec82cbdcadc127b1978c69a3891485d2f12"
PRIV = CACHE / "dpc_v1"
RUN = PRIV / "run"
UNITS = RUN / "units"
ADMITTED = PRIV / "admitted"
ADMISSION_NORM = WT / "results" / "pcrl_joint_complete_view_method_v1" / "DATA_ADMISSION.json"
SMF_BACKUP = WT / "results" / "pcrl_strength_matched_feedback_v1" / "BACKUP_VERIFICATION.json"
BRANCH = "research/pcrl-decision-preserving-compression-v1"
SOURCE_PIN = "925e0fddfcb666116c6179575339728a324ed78e"
OSF_REL = "results/pcrl_online_strength_frontier_v1"
PRED_COMMIT = "325cdbb"

SEEDS = (0, 1, 2)
TEACHERS = ("U", "RAW-J_b0.3")
TEACHER_LABEL = {"U": "U", "RAW-J_b0.3": "RAW-J|b0.3"}
REFS = ("E", "F", "F0")
ROLES = ("OSF_DEFENSE_FIT", "OSF_DEVELOPMENT_ASSESSMENT", "HEAD_VALIDATION", "AUDIT_FIT", "INNER_SELECTION")
SUBROLES = ("CRITIC_FIT", "CRITIC_VAL", "DIAGNOSTIC_CALIB")
FIT_ROLES = ("OSF_DEFENSE_FIT", "HEAD_VALIDATION", "AUDIT_FIT", "INNER_SELECTION")
POOLS = ("ORIG_ASSESSMENT", "RGJ_DEV", "SMF_DEV", "CERT")
EXCLUSIONS = ("excluded_exposure", "excluded_dup")
FIT, ASSESS = "OSF_DEFENSE_FIT", "OSF_DEVELOPMENT_ASSESSMENT"
NUMERIC = ("age", "education-num", "capital-gain", "capital-loss", "hours-per-week")
LABEL_KEYS = {"sex": "sex", "race": "race", "y_income": "y_income", "y_occ": "y_occupation_group"}
KS = (2, 6)
HEAD_C = (0.01, 0.1, 1.0, 10.0, 100.0)
EXPECTED_COUNTS = {"OSF_DEFENSE_FIT": (15434, 15428), "HEAD_VALIDATION": (1500, 1499), "AUDIT_FIT": (6065, 6061),
                   "INNER_SELECTION": (2235, 2234), ASSESS: (13936, 13929)}

# fixed method rules (METHOD_CARD sections 3, 4, 7, 12)
EPS, TOL, TIE_TOL, SWEEPS, MAX_CELLS, ROUNDS, SPARSE_N = 1e-12, 1e-12, 1e-12, 5, 16, 20, 5
RATES, LAMS = (2, 4, 8), (0.1, 1.0, 10.0)
TASK_FAMS, PRIV_FAMS = ("FINE-TASK", "DIRECT-TASK"), ("LOCAL", "SEQ-12", "SEQ-21", "JOINT")
WITNESS_FAMS = ("FINE-TASK", "LOCAL", "SEQ-12", "SEQ-21")
JOINT_START_ORDER = ("JOINT-GREEDY", "FINE-TASK", "LOCAL", "SEQ-12", "SEQ-21")
OBJ_TOL = 1e-9              # objective receipts (prompt: ~1e-9)
LOG_TOL = 1e-10             # recorded increments / deltas vs own from-scratch differences

STATUS_RANK = {"FAIL": 4, "PENDING": 3, "WARN": 2, "PASS": 1, "INFO": 0, "NOT_APPLICABLE": 0}

# every correction made to this verifier's own code (kept; prompt section 17)
VERIFIER_CORRECTIONS: list = [
    {"at": "2026-10-06T00:50Z", "where": "own_family (JOINT candidate values)",
     "what": "candidate F_joint was computed on each start's own coarse-ID labels, so IDENTICAL partitions reached from "
             "different starts differed by float rounding (~1e-17) and the registered exact-tie order picked a different "
             "start label in one unit (U|JOINT|m2|l0.1#s2: own FINE-TASK:refined vs recorded JOINT-GREEDY:refined; four "
             "refined starts reach the same map, the registered JOINT = SEQ-12 alias); candidates are now evaluated on "
             "canonical labels so identical maps tie exactly; near-tied candidates are reported with a same-map check",
     "effect": "winner label only; every merge, move, sweep and the final map already matched"},
    {"at": "2026-10-06T00:50Z", "where": "check_label_custody / check_complete",
     "what": "the lead's SELECTION_AND_AUDIT_LOCK inner stage started during the first full PHASE 1 run; the PHASE 1 "
             "label-array and unit-count checks are now restricted to the PHASE 1 unit kinds (tea, ref, fine, pol); "
             "later-stage units are counted and left to PHASE 2",
     "effect": "scope only"},
    {"at": "2026-10-06T01:20Z", "where": "compare_family_log (JOINT winner)",
     "what": "after evaluating candidates on canonical labels, a second unit (RAW-J_b0.3|JOINT|m4|l0.1#s2) showed the "
             "converse: the lead's arithmetic gives the SAME map 1 ulp apart across starts (0.06769364271555443 vs "
             "...442), so its exact comparison picks FINE-TASK:refined while the exact-tie order on identical maps picks "
             "JOINT-GREEDY:refined; the winner check now requires the recorded winner's MAP to equal the own winner's "
             "map (recorded winner among the own near-tied candidates, all sharing one map) and reports the label "
             "relation (exact / tie-equivalent)", "effect": "comparison semantics for a descriptive receipt label only; "
             "every final map already matched"},
    {"at": "2026-10-06T01:30Z", "where": "check_label_custody",
     "what": "the PHASE 1 check required that no assessment event exist; the lead opened the assessment at 01:18:45Z, "
             "13 s after the EVALUATION_LOCK push, so the check now requires every assess/unseal/outer/infer event to "
             "follow the first EVALUATION_LOCK push (the registered rule) instead of their absence",
     "effect": "status of exclusions_and_label_custody only (it was FAIL purely because the assessment had started)"},
    {"at": "2026-10-06T01:30Z", "where": "main (--light-update)",
     "what": "the lead paused heavy verifier compute during the assessment; the light sections (label custody, "
             "integrity / lock chronology, custody, independence) are re-run over the last full run's report, which "
             "keeps every heavy section and its verifier hash; both hashes are recorded in light_update",
     "effect": "provenance of the PHASE 1 report; PHASE 2 re-runs everything in one process"},
    {"at": "2026-10-06T02:05Z", "where": "check_documents (task-only m4)",
     "what": "the first mapping took the lower-pair task-only m4 code (DIRECT-TASK); the document's 0.811 / 1.303 are "
             "FINE-TASK m4; the check now uses FINE-TASK m4 and separately verifies that joint m8 is below BOTH task-only "
             "m4 codes on pair AUC and occupation log loss", "effect": "document cross-check only"},
    {"at": "2026-10-06T02:30Z", "where": "check_documents",
     "what": "after the lead corrected the 11 slips and added the claim-A disclosure, the document check was rewritten "
             "so that every quoted value is tied to an anchor line and exact text fragment of the CURRENT document "
             "(stale hard-coded quotes would now show as text_present = false); ADVISOR_BRIEF.md and the RUN_STATUS "
             "validity_note were added, and the disclosure's factual statements (six NO_FEASIBLE_CONTROL cells, no "
             "task-eligible JOINT, J* C_match guard recorded missing, B/C comparators eligible) are checked against own "
             "selection", "effect": "document cross-check only; every other section unchanged"},
    {"at": "2026-10-06T02:45Z", "where": "check_documents (quoted fragments)",
     "what": "fragments updated to the lead's corrected text: RESEARCH_DECISION plain-language P12 '0.051' (was the "
             "flagged '0.052') and income '+0.001 ... task-only and +0.002 ... joint'; ADVISOR_BRIEF income '+0.001 to "
             "+0.002 nats'; the two income checks now verify both codes (FINE-TASK m8 and JOINT m8)",
     "effect": "document cross-check only"},
    {"at": "2026-10-06T01:20Z", "where": "pgrep_dpc", "what": "counted the /usr/bin/time wrapper processes as workers "
     "(2 matches per worker); now counts the lead's dpc Python processes only", "effect": "compute report only"},
]


# the verifier's own process ledger for PHASE 1 (single-threaded processes; times from the run logs, UTC)
VERIFIER_RUNS = [
    {"run": "selftests", "end": "00:41Z", "wall_s": 2.0, "cpu_s": 2.0, "heavy": False},
    {"run": "phase 1, search sample, no head refit (scratch output)", "end": "00:43:41Z", "wall_s": 50.2, "cpu_s": 48.3,
     "heavy": True, "lead_workers": "0 at start and end"},
    {"run": "phase 1, no search, mutation tests (scratch output)", "end": "00:45:56Z", "wall_s": 16.4, "cpu_s": 14.6,
     "heavy": False, "lead_workers": "0"},
    {"run": "full A (search all, head refits)", "start": "00:46:08Z", "end": "00:51:45Z", "wall_s": 336.6,
     "cpu_s": 333.4, "heavy": True, "lead_workers": "0 at start; the lead's 2 inner workers started 00:49:45Z",
     "overlap_with_two_lead_workers_s": 120},
    {"run": "single-unit JOINT re-checks (2)", "end": "~01:00Z", "wall_s": 18.0, "cpu_s": 16.0, "heavy": False,
     "lead_workers": "2 (light work only)"},
    {"run": "full B", "start": "01:09:32Z", "end": "01:15:09Z", "wall_s": 336.9, "cpu_s": 333.2, "heavy": True,
     "lead_workers": "1 (controls) at start and end"},
    {"run": "full C (heavy sections of this report)", "start": "01:16:56Z", "end": "01:22:44Z", "wall_s": 347.7,
     "cpu_s": 343.2, "heavy": True, "lead_workers": "1 at start; the lead's 2 assessment workers started 01:18:45Z",
     "overlap_with_two_lead_workers_s": 239},
    {"run": "light updates (custody, chronology)", "end": "01:2xZ", "wall_s": 15.0, "cpu_s": 6.0, "heavy": False},
    {"run": "phase 2 development run 1 (no search, no head refit; scratch output)", "start": "01:56:34Z",
     "wall_s": 170.5, "cpu_s": 164.3, "heavy": True, "lead_workers": "0 at start and end"},
    {"run": "phase 2 development run 2 (scratch output)", "start": "02:03:50Z", "wall_s": 169.5, "cpu_s": 163.9,
     "heavy": True, "lead_workers": "0 at start and end"},
    {"run": "phase 2 full run (first PHASE_2 report, sha256 3ae11c6d...)", "start": "02:08:19Z", "wall_s": 465.9,
     "cpu_s": 460.5, "heavy": True, "lead_workers": "0 at start and end"},
    {"run": "phase 2 development run 3 (document check; scratch output)", "start": "02:25:23Z", "wall_s": 167.5,
     "cpu_s": 162.1, "heavy": True, "lead_workers": "0 at start and end"},
    {"run": "phase 2 full run (document re-check, sha256 2f59976f...)", "start": "02:29:18Z", "wall_s": 460.9,
     "cpu_s": 455.6, "heavy": True, "lead_workers": "0 at start and end"},
]
VERIFIER_RUNS_NOTE = ("every heavy verifier process started when pgrep showed at most one lead dpc worker; twice the lead "
                      "then started two workers while the verifier was running, so three heavy processes ran together "
                      "for about 2.0 min (00:49:45-00:51:45Z) and 4.0 min (01:18:45-01:22:44Z); after the lead's "
                      "request at about 01:23Z only light work was done")


def g(x) -> str:
    return f"{x:g}"


def config_id(t, fam, m=None, lam=None):
    if fam == "CLASS":
        return f"{t}|CLASS|m1"
    return f"{t}|{fam}|m{m}" + (f"|l{g(lam)}" if fam in PRIV_FAMS else "")


def bank_ids():
    out = []
    for t in TEACHERS:
        for m in RATES:
            out += [config_id(t, f, m) for f in TASK_FAMS]
            out += [config_id(t, f, m, lam) for lam in LAMS for f in PRIV_FAMS]
        out.append(config_id(t, "CLASS"))
    return out


def parse_cid(cid):
    t, fam, m, *rest = cid.split("|")
    return {"teacher": t, "family": fam, "m": int(m[1:]), "lam": float(rest[0][1:]) if rest else None}


def pol_unit(k, cid):
    return f"pol__s{k}__{cid.replace('|', '_')}"


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
    for root, ph in ((str(PRIV), "<PRIVATE_CACHE>/dpc_v1"), (str(CACHE), "<PRIVATE_CACHE>"), (str(WT), "<WORKTREE>"),
                     ("/Volumes", "<DRIVE_ROOT>")):
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


def unit_done(name) -> bool:
    return (UNITS / name / "COMPLETE.json").exists()


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


def pgrep_dpc():
    """PIDs of the lead's dpc Python processes (the /usr/bin/time wrappers that also match are not counted)."""
    r = subprocess.run(["pgrep", "-f", "m dpc"], capture_output=True, text=True)
    out = []
    for x in (r.stdout.split() if r.returncode == 0 else []):
        c = subprocess.run(["ps", "-o", "comm=", "-p", x], capture_output=True, text=True).stdout.strip().lower()
        if "python" in c.rsplit("/", 1)[-1]:
            out.append(int(x))
    return out


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
    """Independent reconstruction of the osf roles (reused unchanged by dpc), subroles, pools and the 83-column inputs."""

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
        for o, r in (("defense_train", "DEFENSE_FIT"), ("attacker_fit", "AUDIT_FIT"), ("attacker_val", "INNER_SELECTION")):
            rgj[old == o] = r
        dv = old == "defense_val"
        lab, d1 = _split(unit[dv], 20261004, "dev", [Fraction(3, 10)], ["HEAD_VALIDATION", "DEVELOPMENT_ASSESSMENT"])
        rgj[dv] = lab
        df = rgj == "DEFENSE_FIT"
        smf = np.array([""] * n, dtype=object)
        lab, d3 = _split(unit[df], 20261005, "assess", [Fraction(1, 5)], ["NEW_DEVELOPMENT_ASSESSMENT", "NEW_DEFENSE_FIT"])
        smf[df] = lab
        for r in ("HEAD_VALIDATION", "AUDIT_FIT", "INNER_SELECTION"):
            smf[rgj == r] = r
        ssub = np.array([""] * n, dtype=object)
        nf = smf == "NEW_DEFENSE_FIT"
        lab, d4 = _split(unit[nf], 20261005, "critic", [Fraction(7, 10), Fraction(17, 20)],
                         ["CRITIC_FIT", "CRITIC_VAL", "CONTROLLER_CALIB"])
        ssub[nf] = lab
        self.disagree = {"rgj_dev_split": d1, "smf_assess_split": d3, "smf_critic_split": d4}
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
        self.fit_idx = np.flatnonzero(self.mask[FIT])
        self._sealed = None

    def labels(self):
        """Labels on kept rows, OSF_DEVELOPMENT_ASSESSMENT masked to -1 (PHASE 1 never unseals)."""
        if self._sealed is None:
            z = np.load(SRC, allow_pickle=False)
            out = {}
            for k, v in LABEL_KEYS.items():
                a = z[v].astype(np.int64)[self.keep].copy()
                a[self.mask[ASSESS]] = -1
                out[k] = a
            self._sealed = out
        return self._sealed


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
    sub = {r: rec(np.flatnonzero(f["sub"] == r)) for r in SUBROLES}
    out["roles_match_manifest"] = {r: all(mine[r][k] == man["roles"][r][k] for k in keys4) for r in ROLES}
    out["subroles_match_manifest"] = {r: all(sub[r][k] == man["defense_fit_subroles"][r][k]
                                             for k in ("rows", "groups", "row_id_sha256")) for r in SUBROLES}
    out["expected_counts_hold"] = {r: (mine[r]["rows"], mine[r]["groups"]) == EXPECTED_COUNTS[r] for r in ROLES}
    out["counts"] = {r: {"rows": mine[r]["rows"], "groups": mine[r]["groups"]} for r in ROLES}
    am = f["role"] == ASSESS
    bp = man["roles"][ASSESS]["by_pool"]
    out["pools_match_manifest"] = {p: (int((am & (f["pool"] == p)).sum()) == bp[p]["rows"] and
                                       len(np.unique(f["unit"][am & (f["pool"] == p)])) == bp[p]["groups"] and
                                       rowid_hash(f["row_id"][am & (f["pool"] == p)]) == bp[p]["row_id_sha256"])
                                   for p in POOLS}
    pa = man["exclusions"]["pool_admission"]
    out["pool_admission_match_manifest"] = {p: (int((f["nominal_pool"] == p).sum()) == pa[p]["nominal_rows"] and
                                                D.pool_excluded_rows[p] == pa[p]["excluded_rows_group_overlap"])
                                            for p in POOLS}
    # pinned osf manifest: role-partition fingerprint
    osf_man = json.loads(git_show_bytes(SOURCE_PIN, f"{OSF_REL}/ROLE_MANIFEST.json") or b"{}")
    allr = dict(mine)
    allr.update(sub)
    fp = hashlib.sha256("|".join(f"{r}:{allr[r]['row_id_sha256']}" for r in ROLES + SUBROLES).encode()).hexdigest()
    out["pinned_osf_role_partition_fingerprint_match"] = fp == osf_man.get("role_partition_fingerprint_sha256")
    out["role_partition_fingerprint_sha256"] = fp
    out["dpc_manifest_source_sha256_equals_pinned_blob"] = hashlib.sha256(
        git_show_bytes(SOURCE_PIN, f"{OSF_REL}/ROLE_MANIFEST.json") or b"").hexdigest() == \
        man.get("source_role_manifest_sha256_at_pin")
    groups = {r: set(np.unique(f["unit"][f["role"] == r]).tolist()) for r in ROLES}
    out["roles_group_disjoint"] = all(not (groups[a] & groups[b]) for i, a in enumerate(ROLES) for b in ROLES[i + 1:])
    out["assessment_groups_shared_with_role"] = {r: len(groups[ASSESS] & groups[r]) for r in ROLES if r != ASSESS}
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
    out["dropped_match_manifest"] = (out["dropped_rows"] == man["exclusions"]["dropped_rows"] and
                                     out["dropped_rows_by_old_role"] == man["exclusions"]["dropped_by_old_role"])
    nr = man["numeric_refit"]
    out["numeric_refit_match"] = {c: bool(D.numeric[c]["mean_osf_fit"] == nr[c]["mean_osf_fit"]
                                          and D.numeric[c]["sd_osf_fit"] == nr[c]["sd_osf_fit"]
                                          and D.numeric[c]["inversion_max_abs_err_kept_rows"] < 0.05) for c in NUMERIC}
    out["feature_names_sha256_match"] = hashlib.sha256("\n".join(D.feature_names).encode()).hexdigest() == \
        man["permitted_columns"]["feature_names_sha256"]
    out["column_order_equals_manifest"] = D.feature_names == man["permitted_columns"]["order"]
    low = [s.lower().split("=")[0] for s in D.feature_names]
    out["n_columns"] = len(D.feature_names)
    out["forbidden_columns_present"] = sorted({s for s in low if s in ("sex", "race", "income", "occupation", "fnlwgt")})
    out["rows_kept"] = int(len(D.keep))
    ok = (out["source_sha256_ok"] and all(out["roles_match_manifest"].values())
          and all(out["subroles_match_manifest"].values()) and all(out["expected_counts_hold"].values())
          and all(out["pools_match_manifest"].values()) and all(out["pool_admission_match_manifest"].values())
          and out["pinned_osf_role_partition_fingerprint_match"] and out["dpc_manifest_source_sha256_equals_pinned_blob"]
          and out["roles_group_disjoint"] and not any(out["assessment_groups_shared_with_role"].values())
          and out["subroles_partition_fit"] and out["no_exclusion_row_kept"] and out["no_kept_group_in_a_dropped_row"]
          and out["dropped_match_manifest"] and all(out["numeric_refit_match"].values())
          and out["feature_names_sha256_match"] and out["column_order_equals_manifest"] and out["n_columns"] == 83
          and not out["forbidden_columns_present"] and sum(D.disagree.values()) == 0
          and D.pool_overlap_with_role == 0 and out["rows_kept"] == 39170)
    return res("PASS" if ok else "FAIL", **out)


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
    (larger) C replaces the incumbent only on strict improvement > 1e-12."""
    best, table = None, []
    for C in HEAD_C:
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
        m.fit(R[tr], y[tr])
        ll = float(log_loss(y[va], m.predict_proba(R[va]), labels=list(range(K))))
        table.append({"C": C, "log_loss": ll})
        if best is None or ll < best[0] - 1e-12:
            best = (ll, C, m)
    return best[2], best[1], table


def pinned_manifest():
    return json.loads(git_show_bytes(SOURCE_PIN, f"{OSF_REL}/MODEL_MANIFEST.json") or b"{}")


def custody(unit_dir: Path, entry):
    """COMPLETE.json files == pinned manifest hashes; every file re-hashed; nothing unlisted."""
    c = jload(unit_dir / "COMPLETE.json")
    files = c.get("files", {})
    rehash = {f_: sha_file(unit_dir / f_) == h for f_, h in files.items() if (unit_dir / f_).exists()}
    present = {str(p.relative_to(unit_dir)) for p in unit_dir.rglob("*") if p.is_file()} - {"COMPLETE.json"}
    return {"complete_files_equal_pinned_manifest": files == entry["complete_files_sha256"],
            "files_rehashed_equal": len(rehash) == len(files) and all(rehash.values()),
            "no_unlisted_files": sorted(present - set(files)) == [],
            "complete_id_equals_unit": c.get("id") == unit_dir.name}


def check_teachers(D: Data, L, refit=True):
    man = pinned_manifest()
    entries = {(u["label"], u["seed"], u["unit"]): u for u in man.get("units", [])}
    out, T = {}, {}
    tr, va = np.flatnonzero(D.mask[FIT]), np.flatnonzero(D.mask["HEAD_VALIDATION"])
    for k in SEEDS:
        for t in TEACHERS:
            name = f"rel__s{k}__{t}"
            ud = ADMITTED / name
            info, fails = {"unit": name}, []
            entry = entries.get((TEACHER_LABEL[t], k, name))
            if entry is None:
                out[f"{t}|s{k}"] = res("FAIL", failures=["not in the pinned MODEL_MANIFEST"])
                continue
            info["custody"] = custody(ud, entry)
            if not all(info["custody"].values()):
                fails.append("custody")
            sd = tload(ud / "model.pt")
            exp = {f"enc.{i}.{j}.{w}" for i in (0, 1) for j in (0, 2, 4) for w in ("weight", "bias")} | \
                  {f"head.{i}.{w}" for i in (0, 1) for w in ("weight", "bias")}
            info["state_keys_ok"] = set(sd) == exp and tuple(sd["enc.0.0.weight"].shape) == (64, 83) and \
                tuple(sd["enc.0.2.weight"].shape) == (64, 64) and tuple(sd["enc.1.4.weight"].shape) == (16, 64) and \
                all(v.dtype == torch.float32 for v in sd.values())
            if not info["state_keys_ok"]:
                fails.append("state dict")
            tea = np.load(UNITS / f"tea__s{k}__{t}" / "teacher.npz", allow_pickle=False)
            rel = np.load(ud / "release.npz", allow_pickle=False)
            info["teacher_unit_keys"] = sorted(tea.files)
            info["row_id_equals_own_rows"] = bool(np.array_equal(tea["row_id"], D.row_id)) and \
                bool(np.array_equal(rel["row_id"], D.row_id))
            if not info["row_id_equals_own_rows"]:
                fails.append("row order")
            mine = {}
            for i in (0, 1):
                j = i + 1
                r = encode(sd, i, D.X)
                head = joblib.load(ud / f"head_{i}.joblib")
                ok_type = isinstance(head, Pipeline) and len(head.steps) == 2 and \
                    isinstance(head.steps[0][1], StandardScaler) and isinstance(head.steps[1][1], LogisticRegression)
                c, p, d = head_outputs(head, r)
                mine.update({f"r{j}": r, f"c{j}": c, f"p{j}": p, f"d{j}": d})
                hi = {"pipeline_scaler_logreg": bool(ok_type),
                      "classes_0_to_K_minus_1": bool(np.array_equal(head[-1].classes_, np.arange(KS[i])))}
                sc = head[0]
                ref = StandardScaler().fit(r[tr])
                hi["scaler_n_equals_OSF_DEFENSE_FIT"] = bool(np.all(np.asarray(sc.n_samples_seen_) == len(tr)))
                hi["scaler_moments_bitwise_OSF_DEFENSE_FIT"] = bool(np.array_equal(sc.mean_, ref.mean_) and
                                                                    np.array_equal(sc.var_, ref.var_))
                for key, arr in (("r", r), ("c", c), ("p", p), ("d", d)):
                    tk = f"{key}{j}"
                    rk = {"r": f"r{j}", "c": f"c{j}", "p": f"p{j}", "d": f"hard{j}"}[key]
                    hi[f"{key}_bitwise_vs_teacher_unit"] = bitwise(arr.astype(tea[tk].dtype), tea[tk]) and \
                        arr.dtype == tea[tk].dtype
                    hi[f"{key}_bitwise_vs_admitted_release"] = bitwise(arr.astype(rel[rk].dtype), rel[rk])
                    hi[f"{key}_max_abs_diff_vs_teacher_unit"] = maxdiff(arr, tea[tk])
                if not all(v for kk, v in hi.items() if isinstance(v, bool)):
                    fails.append(f"recipient {j} outputs/head")
                P = p
                hi["rows_with_exact_max_ties"] = int(np.sum((P == P.max(1, keepdims=True)).sum(1) > 1))
                hi["exact_zero_components"] = int(np.sum(P == 0))
                hi["min_probability"] = float(P.min())
                hi["max_row_sum_dev"] = float(np.abs(P.sum(1) - 1).max())
                hi["finite"] = bool(np.isfinite(P).all())
                hi["predicted_class_counts_by_role"] = {r_: np.bincount(d[D.mask[r_]], minlength=KS[i]).tolist()
                                                        for r_ in ROLES}
                if refit:
                    y = L["y_income"] if i == 0 else L["y_occ"]
                    assert (y[tr] >= 0).all() and (y[va] >= 0).all()
                    mh, C, table = my_fit_head(r, y, tr, va, KS[i])
                    mc, mp, md = head_outputs(mh, r)
                    rec = jload(ud / "record.json")
                    rt = {float(e["C"]): e["defense_val_log_loss"] for e in rec["heads"][str(i)]["table"]}
                    hi["refit_C"] = C
                    hi["saved_head_C"] = float(head[-1].C)
                    hi["recorded_C"] = float(rec["heads"][str(i)]["selected_C"])
                    hi["refit_table_max_abs_diff_vs_record"] = max(abs(e["log_loss"] - rt[e["C"]]) for e in table)
                    hi["refit_coef_bitwise"] = bool(np.array_equal(mh[-1].coef_, head[-1].coef_) and
                                                    np.array_equal(mh[-1].intercept_, head[-1].intercept_))
                    hi["refit_p_max_abs_diff"] = maxdiff(mp, p)
                    hi["refit_decision_mismatches"] = int((md != d).sum())
                    if not (C == hi["saved_head_C"] == hi["recorded_C"]) or hi["refit_p_max_abs_diff"] > 1e-8 or \
                            hi["refit_decision_mismatches"] or hi["refit_table_max_abs_diff_vs_record"] > 1e-9:
                        fails.append(f"recipient {j} head refit")
                info[f"recipient_{j}"] = hi
            tr_rec = jload(UNITS / f"tea__s{k}__{t}" / "record.json")
            info["teacher_unit_p_sha256_matches_own"] = tr_rec.get("p_sha256") == sha_bytes(mine["p1"], mine["p2"])
            info["teacher_unit_model_sha256_equals_manifest"] = \
                tr_rec["admission"]["complete_files_sha256"]["model.pt"] == entry["complete_files_sha256"]["model.pt"]
            if not (info["teacher_unit_p_sha256_matches_own"] and info["teacher_unit_model_sha256_equals_manifest"]):
                fails.append("teacher unit record")
            info["model_pt_sha256"] = entry["complete_files_sha256"]["model.pt"]
            T[(k, t)] = {"p1": mine["p1"], "p2": mine["p2"], "d1": mine["d1"].astype(np.int64),
                         "d2": mine["d2"].astype(np.int64), "r1": mine["r1"], "r2": mine["r2"], "c1": mine["c1"],
                         "c2": mine["c2"], "model_sha": info["model_pt_sha256"]}
            info["failures"] = fails
            out[f"{t}|s{k}"] = res("FAIL" if fails else "PASS", **info)
    # decision agreement between the two teachers (descriptive; RAW-J codes preserve RAW-J, not U)
    agree = {}
    for k in SEEDS:
        if (k, "U") in T and (k, "RAW-J_b0.3") in T:
            agree[f"s{k}"] = {f"recipient_{j}_rows_where_U_and_RAWJ_decisions_differ":
                              int((T[(k, "U")][f"d{j}"] != T[(k, "RAW-J_b0.3")][f"d{j}"]).sum()) for j in (1, 2)}
    st = worst(*[v["status"] for v in out.values()]) if len(out) == 6 else "FAIL"
    return res(st, units=out, teacher_decision_disagreement=agree,
               note="forward pass on the verifier's own X (83 permitted columns); head refits read OSF_DEFENSE_FIT and "
                    "HEAD_VALIDATION task labels only"), T


# ------------------------------------------------------------------------------------------------ references
def leace_apply(npz_path, H):
    z = np.load(npz_path, allow_pickle=False)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float64))  # noqa: E731
    pl, pr, mu = t(z["proj_left"]), t(z["proj_right"]), t(z["mean_x"])
    x = t(H)
    with torch.no_grad():
        return (x - ((x - mu) @ pr.T) @ pl.T).numpy().astype(np.float64, copy=True)


def tree_cells(model_json, X):
    m = jload(model_json)
    Xf = np.asarray(X).astype(np.float32).astype(np.float64)
    left, right = np.asarray(m["children_left"]), np.asarray(m["children_right"])
    feat, thr = np.asarray(m["feature"]), np.asarray(m["threshold"], dtype=np.float64)
    node = np.zeros(Xf.shape[0], dtype=np.int64)
    active = left[node] != -1
    while np.any(active):
        idx = np.nonzero(active)[0]
        nd = node[idx]
        node[idx] = np.where(Xf[idx, feat[nd]] <= thr[nd], left[nd], right[nd])
        active = left[node] != -1
    pos = {int(v): j for j, v in enumerate(m["leaf_node_ids"])}
    return np.array([pos.get(int(v), -1) for v in node], dtype=np.int64)


def check_references(D: Data, T, keep=None):
    man = pinned_manifest()
    entries = {(u["label"], u["seed"], u["unit"]): u for u in man.get("units", [])}
    out = {}
    keep = {} if keep is None else keep          # own reference outputs for PHASE 2: keep[(label, k)]
    for k in SEEDS:
        # LEACE
        name = f"lc__s{k}__E"
        ud = ADMITTED / name
        info, fails = {}, []
        e = entries.get(("E", k, name))
        info["custody"] = custody(ud, e) if e else None
        if not e or not all(info["custody"].values()):
            fails.append("custody")
        info["model_pt_equals_U_teacher"] = bool(e and e["complete_files_sha256"]["model.pt"] == T[(k, "U")]["model_sha"])
        sd = tload(ud / "model.pt")
        ref = np.load(UNITS / f"ref__s{k}__E" / "reference.npz", allow_pickle=False)
        rel = np.load(ud / "release.npz", allow_pickle=False)
        for i in (0, 1):
            j = i + 1
            H = encode(sd, i, D.X)
            R = leace_apply(ud / f"leace_{i}" / "leace_map.npz", H)
            c, p, d = head_outputs(joblib.load(ud / f"head_{i}.joblib"), R)
            keep.setdefault(("E", k), {}).update({f"p{j}": p, f"d{j}": d.astype(np.int64), f"c{j}": c, f"r{j}": R})
            info[f"recipient_{j}"] = {"r_max_abs_diff_vs_release": maxdiff(R, rel[f"r{j}"]),
                                      "p_max_abs_diff_vs_release": maxdiff(p, rel[f"p{j}"]),
                                      "decision_mismatches_vs_release": int((d != rel[f"hard{j}"]).sum()),
                                      "ref_unit_equals_admitted_release": all(bitwise(ref[a], rel[b]) for a, b in (
                                          (f"r{j}", f"r{j}"), (f"c{j}", f"c{j}"), (f"p{j}", f"p{j}"), (f"d{j}", f"hard{j}")))}
            ii = info[f"recipient_{j}"]
            if ii["r_max_abs_diff_vs_release"] > 1e-12 or ii["p_max_abs_diff_vs_release"] > 1e-12 or \
                    ii["decision_mismatches_vs_release"] or not ii["ref_unit_equals_admitted_release"]:
                fails.append(f"E recipient {j}")
        if not info["model_pt_equals_U_teacher"] or not bool(np.array_equal(ref["row_id"], D.row_id)):
            fails.append("E model/rows")
        info["failures"] = fails
        out[f"E|s{k}"] = res("FAIL" if fails else "PASS", **info)
        # FARE / F0
        for lab, tag in (("F", "c1"), ("F0", "Z1")):
            info, fails = {}, []
            ref = np.load(UNITS / f"ref__s{k}__{lab}" / "reference.npz", allow_pickle=False)
            for i in (0, 1):
                j = i + 1
                name = f"fare__s{k}__p{i}__{tag}"
                ud = ADMITTED / name
                e = entries.get((lab, k, name))
                ii = {"custody": custody(ud, e) if e else None}
                rec = jload(ud / "record.json")
                uid = rec["provenance"]["admission"]["source"]
                tdir = ADMITTED / "fare_cache" / uid
                cells = tree_cells(tdir / "model" / "model.json", D.X)
                rel = np.load(ud / "release.npz", allow_pickle=False)
                ncell = int(rec["n_cells"])
                onehot = np.eye(ncell)[rel["cells"]]
                c, p, d = head_outputs(joblib.load(ud / "head.joblib"), np.eye(ncell)[cells])
                keep.setdefault((lab, k), {}).update({f"p{j}": p, f"d{j}": d.astype(np.int64), f"c{j}": c,
                                                      f"cells{j}": cells, f"ncell{j}": ncell})
                ii.update({"purpose_matches": rec["purpose"] == i, "own_tree_cells_equal_release": bool(np.array_equal(cells, rel["cells"])),
                           "r_equals_one_hot_cells": bool(np.array_equal(onehot, rel["r"])),
                           "p_bitwise_vs_release": bitwise(p, rel["p"]), "decision_mismatches": int((d != rel["hard"]).sum()),
                           "ref_unit_equals_admitted_release": all(bitwise(ref[a], rel[b]) for a, b in (
                               (f"r{j}", "r"), (f"c{j}", "c"), (f"p{j}", "p"), (f"d{j}", "hard"), (f"cells{j}", "cells"))),
                           "n_cells": ncell})
                if not e or not all(ii["custody"].values()) or not all(v for kk, v in ii.items() if isinstance(v, bool)) \
                        or ii["decision_mismatches"]:
                    fails.append(f"{lab} recipient {j}")
                info[f"recipient_{j}"] = ii
            info["failures"] = fails
            out[f"{lab}|s{k}"] = res("FAIL" if fails else "PASS", **info)
    return res(worst(*[v["status"] for v in out.values()]), units=out, own_outputs_kept=len(keep),
               note="label-free: own LEACE formula on own U features, own traversal of the official FARE tree arrays "
                    "(label-derived cell counts never read), deployed heads; ref__ units equal the admitted copies")


# ------------------------------------------------------------------------------------------------ own method math
def smooth(M, c):
    """(mean + eps*1 + eps*e_c) / (1 + (K+1) eps) per row."""
    M = np.atleast_2d(np.asarray(M, dtype=np.float64))
    K = M.shape[1]
    c = np.broadcast_to(np.asarray(c, dtype=np.int64), (M.shape[0],))
    E = np.zeros_like(M)
    E[np.arange(M.shape[0]), c] = EPS
    return (M + EPS * np.ones_like(M) + E) / (1.0 + (K + 1) * EPS)


def strict_argmax_ok(Q, c):
    Q = np.atleast_2d(Q)
    c = np.broadcast_to(np.asarray(c, dtype=np.int64), (Q.shape[0],))
    qc = Q[np.arange(Q.shape[0]), c]
    other = Q.copy()
    other[np.arange(Q.shape[0]), c] = -np.inf
    return bool(np.all(qc > other.max(1))) if Q.shape[0] else True


def kl_matrix(P, Q):
    """KL(p_r || q_j), masked 0 log 0 = 0, summed in increasing k (the fixed assignment rule)."""
    P = np.asarray(P, dtype=np.float64)
    LQ = np.log(np.atleast_2d(np.asarray(Q, dtype=np.float64)))
    out = np.zeros((P.shape[0], LQ.shape[0]))
    for k in range(P.shape[1]):
        p = P[:, k:k + 1]
        lp = np.log(np.where(p > 0, p, 1.0))
        out = out + np.where(p > 0, p * (lp - LQ[None, :, k]), 0.0)
    return out


def kl_rows(P, Q):
    P, Q = np.asarray(P, dtype=np.float64), np.asarray(Q, dtype=np.float64)
    out = np.zeros(P.shape[0])
    for k in range(P.shape[1]):
        p = P[:, k]
        lp = np.log(np.where(p > 0, p, 1.0))
        out = out + np.where(p > 0, p * (lp - np.log(Q[:, k])), 0.0)
    return out


def negent(P):
    P = np.asarray(P, dtype=np.float64)
    out = np.zeros(P.shape[0])
    for k in range(P.shape[1]):
        p = P[:, k]
        out = out + np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0)
    return out


def cell_stats(P, cell, F):
    n = np.bincount(cell, minlength=F).astype(np.int64)
    S = np.stack([np.bincount(cell, weights=P[:, k], minlength=F) for k in range(P.shape[1])], 1)
    A = np.bincount(cell, weights=negent(P), minlength=F)
    return n, S.astype(np.float64), A.astype(np.float64)


def distinct_rows_lex(Pc):
    """Distinct rows in ascending lexicographic order (own: lexsort + exact consecutive comparison)."""
    order = np.lexsort(Pc.T[::-1])
    S = Pc[order]
    keep = np.ones(S.shape[0], dtype=bool)
    keep[1:] = np.any(S[1:] != S[:-1], axis=1)
    return S[keep]


class Fine:
    def __init__(self, K, cls, cen, mean, n, S, A, fb, receipt=None):
        self.K, self.cls, self.cen, self.mean, self.n, self.S, self.A, self.fb = K, cls, cen, mean, n, S, A, fb
        self.receipt = receipt or {}

    @property
    def F(self):
        return int(len(self.cls))

    @classmethod
    def from_json(cls, z):
        K = int(z["K"])
        return cls(K, np.asarray(z["cell_class"], dtype=np.int64), np.asarray(z["centroid"], dtype=np.float64).reshape(-1, K),
                   np.asarray(z["mean"], dtype=np.float64).reshape(-1, K), np.asarray(z["n"], dtype=np.int64),
                   np.asarray(z["S"], dtype=np.float64).reshape(-1, K), np.asarray(z["A"], dtype=np.float64),
                   np.asarray(z["fallback"], dtype=bool), z.get("receipt", {}))

    def fingerprint(self):
        """Project convention (FinePartition.fingerprint), re-implemented: K, classes, centroids, fallback, n, S, A."""
        h = hashlib.sha256()
        for a in (np.int64(self.K), self.cls.astype("<i8"), self.cen.astype("<f8"), self.fb.astype(np.uint8),
                  self.n.astype("<i8"), self.S.astype("<f8"), self.A.astype("<f8")):
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()

    def equal(self, o):
        return {"K": self.K == o.K, "cell_class": bitwise(self.cls, o.cls), "centroid": bitwise(self.cen, o.cen),
                "mean": bitwise(self.mean, o.mean), "n": bitwise(self.n, o.n), "S": bitwise(self.S, o.S),
                "A": bitwise(self.A, o.A), "fallback": bitwise(self.fb, o.fb)}


def kmeans_class(Pc, c, max_cells, rounds):
    nrows = Pc.shape[0]
    U = distinct_rows_lex(Pc)
    nU = U.shape[0]
    k = int(min(max_cells, nU, nrows))
    order = np.argsort(-U[:, c], kind="stable")
    U = U[order]
    pos = [((2 * j + 1) * nU) // (2 * k) for j in range(k)]
    C = U[pos].copy()
    Q = smooth(C, np.full(k, c))
    prev, converged, used, hist = None, False, 0, []
    for r in range(1, rounds + 1):
        a = kl_matrix(Pc, Q).argmin(1)
        used = r
        if prev is not None and np.array_equal(a, prev):
            converged = True
            break
        hist.append(int(nrows if prev is None else np.sum(a != prev)))
        n, S, _ = cell_stats(Pc, a, k)
        for j in range(k):
            if n[j] > 0:
                C[j] = S[j] / n[j]
        Q = smooth(C, np.full(k, c))
        prev = a
    if not converged:
        a = kl_matrix(Pc, Q).argmin(1)
        converged = bool(prev is not None and np.array_equal(a, prev))
    return Q, a, {"class": int(c), "rows": int(nrows), "distinct_vectors": int(nU), "initial_cells": k,
                  "init_positions": [int(x) for x in pos], "rounds_used": int(used), "converged": bool(converged),
                  "changed_per_round": hist}


def fit_fine(P, d, K, max_cells=MAX_CELLS, rounds=ROUNDS):
    P = np.asarray(P, dtype=np.float64) + 0.0
    parts = {k_: [] for k_ in ("cls", "cen", "mean", "n", "S", "A", "fb")}
    per = []
    for c in range(K):
        rows = np.flatnonzero(d == c)
        if rows.size == 0:
            u = np.full(K, 1.0 / K)
            for k_, v in (("cls", [c]), ("cen", smooth(u, c)), ("mean", u[None]), ("n", [0]), ("S", np.zeros((1, K))),
                          ("A", [0.0]), ("fb", [True])):
                parts[k_].append(np.asarray(v))
            per.append({"class": c, "rows": 0, "fallback": True, "cells": 1})
            continue
        Pc = P[rows]
        Q, a, rec = kmeans_class(Pc, c, max_cells, rounds)
        n, S, A = cell_stats(Pc, a, Q.shape[0])
        keep = np.flatnonzero(n > 0)
        for k_, v in (("cls", np.full(keep.size, c)), ("cen", Q[keep]), ("mean", S[keep] / n[keep, None]),
                      ("n", n[keep]), ("S", S[keep]), ("A", A[keep]), ("fb", np.zeros(keep.size, dtype=bool))):
            parts[k_].append(np.asarray(v))
        rec.update({"fallback": False, "cells": int(keep.size), "removed_empty_cells": int(Q.shape[0] - keep.size),
                    "sparse_cells": [int(j) for j, v in enumerate(n[keep]) if v < SPARSE_N],
                    "cell_counts": [int(v) for v in n[keep]]})
        per.append(rec)
    f = Fine(K, np.concatenate(parts["cls"]).astype(np.int64), np.concatenate([np.atleast_2d(x) for x in parts["cen"]]),
             np.concatenate(parts["mean"]), np.concatenate(parts["n"]).astype(np.int64), np.concatenate(parts["S"]),
             np.concatenate(parts["A"]).astype(np.float64), np.concatenate(parts["fb"]).astype(bool), {"per_class": per})
    return f


def assign(P, d, fine: Fine):
    """Deployment: nearest fine cell (masked KL to the stored smoothed centroid) within the row's predicted class;
    ties -> lowest index; a class with one cell (incl. a fallback) maps to it."""
    out = np.full(P.shape[0], -1, dtype=np.int64)
    for c in range(fine.K):
        rows = np.flatnonzero(d == c)
        if rows.size == 0:
            continue
        idx = np.flatnonzero(fine.cls == c)
        if idx.size == 1:
            out[rows] = idx[0]
        elif idx.size > 1:
            out[rows] = idx[kl_matrix(P[rows], fine.cen[idx]).argmin(1)]
    return out


def mi_tab(tab):
    """Plug-in I(S;C) (nats) from a (2, C) count table: sum n_sc/N log(n_sc N / (n_s n_c)), empty cells contribute 0."""
    tab = np.asarray(tab, dtype=np.float64)
    N = tab.sum()
    if N == 0:
        return 0.0
    ns = np.broadcast_to(tab.sum(1, keepdims=True), tab.shape)
    nc = np.broadcast_to(tab.sum(0, keepdims=True), tab.shape)
    nz = tab > 0
    t = tab[nz]
    return float(np.sum(t / N * np.log(t * N / (ns[nz] * nc[nz]))))


def mi_labels(s, *codes):
    """Plug-in I(S; codes) with the exact code tuple (np.unique rows), from per-row labels."""
    C = np.stack([np.asarray(c, dtype=np.int64) for c in codes], 1)
    _, inv = np.unique(C, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    m = int(inv.max()) + 1
    tab = np.stack([np.bincount(inv[s == v], minlength=m) for v in (0, 1)])
    return mi_tab(tab)


def entropy_mi(s, code):
    """Alternative formula (self-test): H(S) - H(S | C)."""
    s = np.asarray(s)
    N = len(s)
    ps = np.bincount(s, minlength=2) / N
    hs = -sum(p * math.log(p) for p in ps if p > 0)
    hc = 0.0
    for v in np.unique(code):
        m = code == v
        q = np.bincount(s[m], minlength=2) / m.sum()
        hc += m.sum() / N * -sum(p * math.log(p) for p in q if p > 0)
    return hs - hc


# ------------------------------------------------------------------------------------------------ policies
class Pol:
    """One recipient's public policy reconstructed from policy.json (prototypes recomputed from the sums)."""

    def __init__(self, z):
        self.fine = Fine.from_json(z["fine"])
        self.cell_token = np.asarray(z["cell_token"], dtype=np.int64)
        self.stored_proto = np.asarray(z["token_proto"], dtype=np.float64)
        self.stored_class = np.asarray(z["token_class"], dtype=np.int64)
        self.stored_n = np.asarray(z["token_n"], dtype=np.int64)
        self.stored_fp = z.get("fingerprint")
        self.meta = z.get("meta", {})
        fine = self.fine
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
            proto[t] = smooth(tS[t] / tn[t], tc[t])[0] if tn[t] > 0 else smooth(np.full(K, 1.0 / K), tc[t])[0]
        self.T, self.token_class, self.token_n, self.token_S, self.token_fb, self.proto = T, tc, tn, tS, tfb, proto
        seen, canon = {}, np.empty(fine.F, dtype=np.int64)
        for f in range(fine.F):
            canon[f] = seen.setdefault(int(self.cell_token[f]), len(seen))
        self.canonical = bool(np.array_equal(canon, self.cell_token))

    def fingerprint(self):
        """Project convention (Policy.fingerprint), re-implemented from own token classes and prototypes."""
        h = hashlib.sha256()
        for a in (np.int64(self.fine.K), self.fine.cls.astype("<i8"), self.fine.cen.astype("<f8"),
                  self.cell_token.astype("<i8"), self.token_class.astype("<i8"), self.proto.astype("<f8")):
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()

    def structure(self):
        P = self.proto
        sums = np.abs(P.sum(1) - 1.0)
        return {"no_class_mixing": not self.mixes_classes, "all_ids_used": self.unused_ids == 0,
                "canonical_ids": self.canonical, "every_class_has_a_token": all(np.any(self.token_class == c)
                                                                                 for c in range(self.fine.K)),
                "fallback_tokens_have_no_fitting_rows": bool(not np.any(self.token_fb & (self.token_n > 0))),
                "prototypes_finite_nonneg": bool(np.isfinite(P).all() and (P >= 0).all()),
                "prototypes_strictly_positive": bool((P > 0).all()),
                "prototype_sum_max_dev": float(sums.max()), "prototype_sums_within_1e-12": bool(sums.max() <= 1e-12),
                "prototype_strict_argmax_is_class": strict_argmax_ok(P, self.token_class),
                "prototypes_bitwise_equal_stored": bitwise(P, self.stored_proto),
                "token_class_equal_stored": bitwise(self.token_class, self.stored_class),
                "token_n_equal_stored": bitwise(self.token_n, self.stored_n),
                "fingerprint_equals_stored": self.fingerprint() == self.stored_fp}

    def unsmoothed_vs_smoothed(self):
        u = self.token_n > 0
        return float(np.max(np.abs(self.token_S[u] / self.token_n[u, None] - self.proto[u]))) if u.any() else 0.0

    def tokens_per_class(self):
        return [int(np.sum(self.token_class == c)) for c in range(self.fine.K)]

    def effective_per_class(self):
        return [int(np.sum((self.token_class == c) & (self.token_n > 0))) for c in range(self.fine.K)]


def objectives(P1, t1, q1, P2, t2, q2, s, lam):
    o = {"D1": float(np.mean(kl_rows(P1, q1))), "D2": float(np.mean(kl_rows(P2, q2))),
         "I1": mi_labels(s, t1), "I2": mi_labels(s, t2), "I12": mi_labels(s, t1, t2)}
    o.update(F_values(o, lam))
    return o


def F_values(o, lam):
    out = {"F_task": o["D1"] + o["D2"]}
    if lam is not None:
        out["F_local"] = o["D1"] + o["D2"] + lam * (o["I1"] + o["I2"]) / 2
        out["F_joint"] = o["D1"] + o["D2"] + lam * ((o["I1"] + o["I2"]) / 2 + o["I12"])
    return out


def D_from_sums(pol: Pol, P_fit, tok_fit):
    """Sufficient-statistic form sum_rows KL = A - S . log q per token (own sums over fitting rows)."""
    K = P_fit.shape[1]
    N = P_fit.shape[0]
    A = float(np.sum(negent(P_fit)))
    S = np.stack([np.bincount(tok_fit, weights=P_fit[:, k], minlength=pol.T) for k in range(K)], 1)
    return (A - float(np.sum(S * np.log(pol.proto)))) / N


# ------------------------------------------------------------------------------------------------ own search engine
class Eng:
    """From-scratch objective of any pair of fine-cell labelings on the fitting rows (own arithmetic)."""

    def __init__(self, fine1: Fine, fine2: Fine, f1, f2, s):
        self.fine = {1: fine1, 2: fine2}
        self.N = int(len(s))
        F1, F2 = fine1.F, fine2.F
        self.F = {1: F1, 2: F2}
        self.T = np.bincount((s * F1 + f1) * F2 + f2, minlength=2 * F1 * F2).reshape(2, F1, F2).astype(np.float64)
        self.Tr = {1: self.T.sum(2), 2: self.T.sum(1)}
        self.Atot = {r: float(np.sum(self.fine[r].A)) for r in (1, 2)}

    def D(self, r, lab):
        fn = self.fine[r]
        F = fn.F
        n = np.bincount(lab, weights=fn.n.astype(np.float64), minlength=F)
        S = np.zeros((F, fn.K))
        np.add.at(S, lab, fn.S)
        cls = np.full(F, 0, dtype=np.int64)
        cls[lab] = fn.cls
        u = n > 0
        q = smooth(S[u] / n[u, None], cls[u])
        return (self.Atot[r] - float(np.sum(S[u] * np.log(q)))) / self.N

    def I(self, r, lab):
        F = self.F[r]
        return mi_tab(np.stack([np.bincount(lab, weights=self.Tr[r][v], minlength=F) for v in (0, 1)]))

    def I12(self, lab1, lab2):
        F2 = self.F[2]
        idx = (lab1[:, None] * F2 + lab2[None, :]).ravel()
        M = self.F[1] * F2
        return mi_tab(np.stack([np.bincount(idx, weights=self.T[v].ravel(), minlength=M) for v in (0, 1)]))

    def terms(self, lab):
        t = {"D1": self.D(1, lab[1]), "D2": self.D(2, lab[2]), "I1": self.I(1, lab[1]), "I2": self.I(2, lab[2]),
             "I12": self.I12(lab[1], lab[2])}
        return t

    def part(self, r, lab, W):
        """Weighted terms that depend on recipient r's labels (D_r, I_r, I12) -> (value, dict)."""
        wD, wI = W[r - 1], W[r + 1]
        o = {"D": self.D(r, lab[r]), "I": self.I(r, lab[r]),          # dI_own is logged even when its weight is 0
             "I12": self.I12(lab[1], lab[2]) if W[4] != 0 else None}   # dI12 logged only when weighted
        v = wD * o["D"] + wI * o["I"] + (W[4] * o["I12"] if W[4] != 0 else 0.0)
        return v, o


def W_task(r):
    return (1.0 if r == 1 else 0.0, 1.0 if r == 2 else 0.0, 0.0, 0.0, 0.0)


def W_local(lam, r):
    return (1.0 if r == 1 else 0.0, 1.0 if r == 2 else 0.0, lam / 2 if r == 1 else 0.0, lam / 2 if r == 2 else 0.0, 0.0)


def W_stage1(lam, r):
    return (1.0 if r == 1 else 0.0, 1.0 if r == 2 else 0.0, 1.5 * lam if r == 1 else 0.0, 1.5 * lam if r == 2 else 0.0, 0.0)


def W_joint(lam):
    return (1.0, 1.0, lam / 2, lam / 2, lam)


def class_labels_of(eng, r, lab, c):
    cls = eng.fine[r].cls
    return sorted(set(lab[r][cls == c].tolist()))


def own_greedy(eng, lab, recips, W, m, log):
    while True:
        cands = []
        for r in recips:
            for c in range(eng.fine[r].K):
                L_ = class_labels_of(eng, r, lab, c)
                if len(L_) > m:
                    cands += [(r, c, L_[i], L_[j]) for i in range(len(L_)) for j in range(i + 1, len(L_))]
        if not cands:
            return
        base = {r: eng.part(r, lab, W) for r in recips}
        vals, parts = [], []
        for r, c, a, b in cands:
            new = dict(lab)
            x = lab[r].copy()
            x[x == b] = a
            new[r] = x
            v, o = eng.part(r, new, W)
            vals.append(v - base[r][0])
            parts.append(o)
        vals = np.asarray(vals)
        gmin = float(vals.min())
        j = int(np.flatnonzero(vals <= gmin + TIE_TOL)[0])
        r, c, a, b = cands[j]
        o, ob = parts[j], base[r][1]
        log.append({"recipient": r, "class": c, "a": a, "b": b, "increment": float(vals[j]), "dD": o["D"] - ob["D"],
                    "dI_own": None if o["I"] is None else o["I"] - ob["I"],
                    "dI12": None if o["I12"] is None else o["I12"] - ob["I12"],
                    "tied_candidates": int(np.sum(vals <= gmin + TIE_TOL))})
        lab[r][lab[r] == b] = a


def own_refine(eng, lab, recips, W, log, sweeps=SWEEPS):
    for sweep in range(1, sweeps + 1):
        moved = 0
        for r in recips:
            cls = eng.fine[r].cls
            for f in range(eng.fine[r].F):
                a = int(lab[r][f])
                if int(np.sum(lab[r] == a)) <= 1:
                    continue
                c = int(cls[f])
                B = [b for b in class_labels_of(eng, r, lab, c) if b != a]
                if not B:
                    continue
                v0, o0 = eng.part(r, lab, W)
                vals, parts = [], []
                for b in B:
                    x = lab[r].copy()
                    x[f] = b
                    new = dict(lab)
                    new[r] = x
                    v, o = eng.part(r, new, W)
                    vals.append(v - v0)
                    parts.append(o)
                j = int(np.argmin(vals))
                if vals[j] < -TOL:
                    o = parts[j]
                    lab[r][f] = B[j]
                    moved += 1
                    log.append({"sweep": sweep, "recipient": r, "fine_cell": f, "from": a, "to": int(B[j]),
                                "delta": float(vals[j]), "dD": o["D"] - o0["D"],
                                "dI_own": None if o["I"] is None else o["I"] - o0["I"],
                                "dI12": None if o["I12"] is None else o["I12"] - o0["I12"]})
        if moved == 0:
            return sweep, True
    return sweeps, False


def canonical(lab):
    seen, out = {}, np.empty(len(lab), dtype=np.int64)
    for f, l_ in enumerate(lab.tolist()):
        out[f] = seen.setdefault(l_, len(seen))
    return out


def labels_from_tokens(tok):
    first = {}
    for f, t in enumerate(tok.tolist()):
        first.setdefault(t, f)
    return np.array([first[t] for t in tok.tolist()], dtype=np.int64)


def own_family(eng, fam, m, lam, witnesses=None):
    """Own deterministic family run (METHOD_CARD sections 7-8). Returns (labels, log)."""
    ident = {r: np.arange(eng.fine[r].F, dtype=np.int64) for r in (1, 2)}
    lab = {r: ident[r].copy() for r in (1, 2)}
    log = {}
    if fam in ("FINE-TASK", "LOCAL"):
        Ws = {r: (W_task(r) if fam == "FINE-TASK" else W_local(lam, r)) for r in (1, 2)}
        for r in (1, 2):
            log[f"r{r}"] = {"merges": [], "moves": []}
            own_greedy(eng, lab, (r,), Ws[r], m, log[f"r{r}"]["merges"])
        log["after_greedy"] = eng.terms(lab)
        for r in (1, 2):
            sw, cv = own_refine(eng, lab, (r,), Ws[r], log[f"r{r}"]["moves"])
            log[f"r{r}"].update({"sweeps": sw, "converged": cv})
        return lab, log
    if fam in ("SEQ-12", "SEQ-21"):
        a, b = (1, 2) if fam == "SEQ-12" else (2, 1)
        s1 = {"merges": [], "moves": []}
        own_greedy(eng, lab, (a,), W_stage1(lam, a), m, s1["merges"])
        s1["after_greedy"] = eng.terms(lab)
        sw, cv = own_refine(eng, lab, (a,), W_stage1(lam, a), s1["moves"])
        s1.update({"sweeps": sw, "converged": cv})
        frozen = lab[a].copy()
        s2 = {"merges": [], "moves": []}
        own_greedy(eng, lab, (b,), W_joint(lam), m, s2["merges"])
        s2["after_greedy"] = eng.terms(lab)
        sw, cv = own_refine(eng, lab, (b,), W_joint(lam), s2["moves"])
        s2.update({"sweeps": sw, "converged": cv})
        log[f"stage1_r{a}"], log[f"stage2_r{b}"] = s1, s2
        log["first_map_unchanged_in_stage2"] = bool(np.array_equal(frozen, lab[a]))
        log["after_greedy"] = s2["after_greedy"]
        return lab, log
    if fam == "JOINT":
        W = W_joint(lam)
        starts, cand = {}, []
        g = {"merges": [], "moves": []}
        own_greedy(eng, lab, (1, 2), W, m, g["merges"])
        g["after_greedy"] = eng.terms(lab)
        g["initial_F_joint"] = F_values(g["after_greedy"], lam)["F_joint"]
        sw, cv = own_refine(eng, lab, (1, 2), W, g["moves"])
        g.update({"sweeps": sw, "converged": cv, "labels": {r: lab[r].copy() for r in (1, 2)}})
        starts["JOINT-GREEDY"] = g
        init = {}
        for name in WITNESS_FAMS:
            wl = {r: labels_from_tokens(witnesses[name][r]) for r in (1, 2)}
            init[name] = {"labels": {r: wl[r].copy() for r in (1, 2)}, "terms": eng.terms(wl)}
            st = {"moves": [], "initial_F_joint": F_values(init[name]["terms"], lam)["F_joint"]}
            sw, cv = own_refine(eng, wl, (1, 2), W, st["moves"])
            st.update({"sweeps": sw, "converged": cv, "labels": wl})
            starts[name] = st
        # candidates are evaluated on CANONICAL labels, so identical partitions reached from different starts give
        # bit-identical objectives and the registered exact-tie order applies (see VERIFIER_CORRECTIONS)
        def canon_terms(labs):
            return eng.terms({r: labels_from_tokens(canonical(labs[r])) for r in (1, 2)})

        for name in JOINT_START_ORDER:
            starts[name]["refined_F_joint"] = F_values(canon_terms(starts[name]["labels"]), lam)["F_joint"]
            cand.append((name, "refined", starts[name]["refined_F_joint"], starts[name]["labels"]))
        for name in WITNESS_FAMS:
            cand.append((name, "unchanged", F_values(canon_terms(init[name]["labels"]), lam)["F_joint"],
                         init[name]["labels"]))
        vals = [c_[2] for c_ in cand]
        kbest = vals.index(min(vals))
        best_map = {r: canonical(cand[kbest][3][r]) for r in (1, 2)}
        near = [j for j, v in enumerate(vals) if v <= vals[kbest] + 1e-15]
        log.update({"starts": starts, "candidates": [(c_[0], c_[1], c_[2]) for c_ in cand],
                    "winner": (cand[kbest][0], cand[kbest][1]), "after_greedy": g["after_greedy"],
                    "near_tied_candidates": [f"{cand[j][0]}:{cand[j][1]}" for j in near],
                    "near_tied_candidates_share_the_winning_map": all(
                        all(np.array_equal(canonical(cand[j][3][r]), best_map[r]) for r in (1, 2)) for j in near)})
        return {r: cand[kbest][3][r].copy() for r in (1, 2)}, log
    if fam == "CLASS":
        return {r: labels_from_tokens(eng.fine[r].cls) for r in (1, 2)}, log
    raise ValueError(fam)


def cmp_merges(mine, rec):
    keys = [(x["recipient"], x["class"], x["a"], x["b"]) for x in mine]
    rkeys = [(x["recipient"], x["class"], x["a"], x["b"]) for x in rec]
    seq = keys == rkeys
    d = 0.0
    if seq:
        for x, y in zip(mine, rec):
            for f_ in ("increment", "dD", "dI_own", "dI12"):
                if (x[f_] is None) != (y[f_] is None):
                    d = float("inf")
                elif x[f_] is not None:
                    d = max(d, abs(x[f_] - y[f_]))
    return seq, d, len(rec)


def cmp_moves(mine, rec):
    keys = [(x["sweep"], x["recipient"], x["fine_cell"], x["from"], x["to"]) for x in mine]
    rkeys = [(x["sweep"], x["recipient"], x["fine_cell"], x["from"], x["to"]) for x in rec]
    seq = keys == rkeys
    d = 0.0
    if seq:
        for x, y in zip(mine, rec):
            for f_ in ("delta", "dD", "dI_own", "dI12"):
                if (x[f_] is None) != (y[f_] is None):
                    d = float("inf")
                elif x[f_] is not None:
                    d = max(d, abs(x[f_] - y[f_]))
    return seq, d, len(rec)


def terms_diff(a, b, keys=("D1", "D2", "I1", "I2", "I12")):
    return max(abs(a[k] - b[k]) for k in keys)


def compare_family_log(fam, mine, rec, lam):
    """Every recorded merge / move / sweep / snapshot vs the own replay."""
    out = {"merge_sequences_equal": True, "move_sequences_equal": True, "max_abs_diff_recorded_vs_own": 0.0,
           "sweeps_and_convergence_equal": True, "merges_checked": 0, "moves_checked": 0}

    def acc(seq, d, n, kind):
        out[f"{kind}_sequences_equal"] &= seq
        out["max_abs_diff_recorded_vs_own"] = max(out["max_abs_diff_recorded_vs_own"], d)
        out[f"{kind}s_checked"] += n

    if fam in ("FINE-TASK", "LOCAL", "SEQ-12", "SEQ-21"):
        names = ["r1", "r2"] if fam in ("FINE-TASK", "LOCAL") else \
            (["stage1_r1", "stage2_r2"] if fam == "SEQ-12" else ["stage1_r2", "stage2_r1"])
        stages = {s["stage"]: s for s in rec["stages"]}
        out["stage_names_equal"] = sorted(stages) == sorted(names)
        for nm in names:
            s, mm = stages.get(nm), mine[nm]
            if s is None:
                out["stage_names_equal"] = False
                continue
            acc(*cmp_merges(mm["merges"], s["merges"]), "merge")
            acc(*cmp_moves(mm["moves"], s["moves"]), "move")
            out["sweeps_and_convergence_equal"] &= (mm["sweeps"] == s["sweeps"] and mm["converged"] == s["converged"])
        if rec.get("after_greedy"):
            out["after_greedy_max_abs_diff"] = terms_diff(mine["after_greedy"], rec["after_greedy"])
    elif fam == "JOINT":
        st = rec["starts"]
        for nm in JOINT_START_ORDER:
            mm, s = mine["starts"][nm], st[nm]
            if nm == "JOINT-GREEDY":
                acc(*cmp_merges(mm["merges"], s["merges"]), "merge")
                out["after_greedy_max_abs_diff"] = terms_diff(mm["after_greedy"], s["after_greedy"])
            acc(*cmp_moves(mm["moves"], s["refine_moves"]), "move")
            out["sweeps_and_convergence_equal"] &= (mm["sweeps"] == s["sweeps"] and mm["converged"] == s["converged"])
            out["max_abs_diff_recorded_vs_own"] = max(out["max_abs_diff_recorded_vs_own"],
                                                      abs(mm["initial_F_joint"] - s["initial_F_joint"]),
                                                      abs(mm["refined_F_joint"] - s["refined_F_joint"]))
        rc = [(c["start"], c["kind"], c["F_joint"]) for c in rec["candidates"]]
        out["candidate_order_equal"] = [(a, b) for a, b, _ in rc] == [(a, b) for a, b, _ in mine["candidates"]]
        out["candidate_F_joint_max_abs_diff"] = max(abs(x[2] - y[2]) for x, y in zip(rc, mine["candidates"]))
        rw = "%s:%s" % (rec["winner"]["start"], rec["winner"]["kind"])
        out["own_winner"] = "%s:%s" % tuple(mine["winner"])
        out["recorded_winner"] = rw
        out["near_tied_candidates"] = mine["near_tied_candidates"]
        out["near_tied_candidates_share_the_winning_map"] = mine["near_tied_candidates_share_the_winning_map"]
        # identical partitions reached from different starts can differ by 1 ulp in the lead's arithmetic (label
        # order), so only the MAP of the winner is decisive; the start label is reported (VERIFIER_CORRECTIONS)
        out["winner_map_equal"] = rw == out["own_winner"] or (rw in mine["near_tied_candidates"] and
                                                              mine["near_tied_candidates_share_the_winning_map"])
        out["winner_label"] = "exact" if rw == out["own_winner"] else \
            ("tie-equivalent (identical map, objectives within 1e-15)" if out["winner_map_equal"] else "MISMATCH")
    return out


# ------------------------------------------------------------------------------------------------ main checks
def check_fine_and_policies(D: Data, T, L, search="all", cache=None, keep_full=()):
    """Fine partitions, every policy, class preservation, objectives, search replay, dominance, sequential, aliases.
    cache (PHASE 2): cache["inner"][(k, cid)] = own release on INNER_SELECTION rows, cache["full"][(k, cid)] = own
    tokens / probabilities / decisions on every row for the configurations in keep_full, cache["token_states"]."""
    fit = D.fit_idx
    cache = {} if cache is None else cache
    cache.setdefault("inner", {})
    cache.setdefault("full", {})
    cache.setdefault("token_states", {})
    sel_rows = np.flatnonzero(D.mask["INNER_SELECTION"])
    s_fit = L["sex"][fit]
    assert (s_fit >= 0).all() and set(np.unique(s_fit).tolist()) <= {0, 1}
    fine_out, pol_out, search_out, dom_out, seq_out = {}, {}, {}, {}, {}
    flag_counts = {}
    cp_rows = {r: 0 for r in ROLES}
    cp_fail, adversarial_fail = [], []
    fallback_use = {}
    my_fp, my_obj = {}, {}
    release_hash = {}
    roles_of_rows = D.role
    adv = adversarial_rows()
    for k in SEEDS:
        for t in TEACHERS:
            tt = T[(k, t)]
            P = {1: tt["p1"], 2: tt["p2"]}
            d = {1: tt["d1"], 2: tt["d2"]}
            # ---- fine partitions
            fj = jload(UNITS / f"fine__s{k}__{t}" / "fine.json")
            az = np.load(UNITS / f"fine__s{k}__{t}" / "assign.npz", allow_pickle=False)
            stored = {1: Fine.from_json(fj["fine1"]), 2: Fine.from_json(fj["fine2"])}
            mine_f, fa, finfo = {}, {}, {}
            for r in (1, 2):
                mf = fit_fine(P[r][fit], d[r][fit], KS[r - 1])
                mine_f[r] = mf
                eq = mf.equal(stored[r])
                fa[r] = assign(P[r], d[r], mf)
                mf_rows = fa[r][fit]
                n, S, A = cell_stats(P[r][fit], mf_rows, mf.F)
                recs = stored[r].receipt.get("per_class", [])
                rcmp = all(rr.get(key) == mm.get(key) for mm, rr in zip(mf.receipt["per_class"], recs)
                           for key in ("rows", "distinct_vectors", "initial_cells", "init_positions", "rounds_used",
                                       "converged", "changed_per_round", "cells", "cell_counts") if key in mm)
                finfo[f"recipient_{r}"] = {
                    "own_kmeans_equals_stored": eq, "all_bitwise": all(eq.values()),
                    "own_fingerprint_equals_stored_fingerprint": mf.fingerprint() == fj[f"fine{r}"].get("fingerprint"),
                    "own_assignment_equals_assign_npz_all_rows": bool(np.array_equal(fa[r], az[f"f{r}"])),
                    "deployment_on_fitting_rows_reproduces_stats": bool(np.array_equal(n, mf.n) and np.array_equal(S, mf.S)
                                                                        and np.array_equal(A, mf.A)),
                    "receipts_per_class_equal_own": bool(rcmp and len(recs) == len(mf.receipt["per_class"])),
                    "cells": int(mf.F), "cells_per_class": [int(np.sum(mf.cls == c)) for c in range(mf.K)],
                    "fallback_classes": [int(c) for c in mf.cls[mf.fb]],
                    "sparse_cells_n_lt_5": int(np.sum((mf.n < SPARSE_N) & ~mf.fb)),
                    "classes_converged": [bool(x.get("converged", True)) for x in mf.receipt["per_class"]],
                    "fit_rows": int(mf.n.sum())}
            finfo["row_id_equal"] = bool(np.array_equal(az["row_id"], D.row_id))
            ok = finfo["row_id_equal"] and all(finfo[f"recipient_{r}"][x] for r in (1, 2) for x in (
                "all_bitwise", "own_fingerprint_equals_stored_fingerprint", "own_assignment_equals_assign_npz_all_rows",
                "deployment_on_fitting_rows_reproduces_stats", "receipts_per_class_equal_own"))
            fine_out[f"{t}|s{k}"] = res("PASS" if ok else "FAIL", **finfo)
            eng = Eng(mine_f[1], mine_f[2], fa[1][fit], fa[2][fit], s_fit)
            # ---- policies of this teacher/seed
            ids = [c for c in bank_ids() if parse_cid(c)["teacher"] == t]
            own_maps = {}
            recs = {}
            for cid in ids:
                pc = parse_cid(cid)
                fam, m, lam = pc["family"], pc["m"], pc["lam"]
                un = pol_unit(k, cid)
                ud = UNITS / un
                info, fails = {"config": cid, "seed": k}, []
                if not unit_done(un):
                    pol_out[f"{cid}#s{k}"] = res("FAIL", failures=["unit missing or incomplete"])
                    continue
                pj = jload(ud / "policy.json")
                rec = jload(ud / "record.json")
                recs[cid] = rec
                rel = np.load(ud / "release.npz", allow_pickle=False)
                pols = {1: Pol(pj["p1"]), 2: Pol(pj["p2"])}
                pair_fp = hashlib.sha256((pols[1].fingerprint() + pols[2].fingerprint()).encode()).hexdigest()
                info["pair_fingerprint_equals_stored"] = pair_fp == pj.get("fingerprint") == rec.get("fingerprint")
                info["config_binding"] = {"teacher_model_sha256_equals_admitted": pj["config"].get("teacher_model_sha256") ==
                                          tt["model_sha"], "config_equals_unit": pj["config"].get("config") == cid,
                                          "family_m_lam": (pj["config"].get("family"), pj["config"].get("m"),
                                                           pj["config"].get("lam")) ==
                                          ({"CLASS": "CLASS-ONLY"}.get(fam, fam), m, lam)}
                if not (info["pair_fingerprint_equals_stored"] and all(info["config_binding"].values())):
                    fails.append("fingerprint/binding")
                toks, qs = {}, {}
                for r in (1, 2):
                    pol = pols[r]
                    st_ = pol.structure()
                    # assignment partition: shared fine partition, or own DIRECT-TASK k-means
                    if fam == "DIRECT-TASK":
                        own_part = fit_fine(P[r][fit], d[r][fit], KS[r - 1], max_cells=m)
                        f_all = assign(P[r], d[r], own_part)
                    else:
                        own_part = mine_f[r]
                        f_all = fa[r]
                    st_["assignment_partition_equals_own"] = all(own_part.equal(pol.fine).values())
                    tok = pol.cell_token[f_all]
                    q = pol.proto[tok]
                    hard = pol.token_class[tok]
                    toks[r], qs[r] = tok, q
                    st_["tokens_bitwise_vs_release"] = bitwise(tok, rel[f"tok{r}"])
                    st_["decoded_probs_bitwise_vs_release"] = bitwise(q, rel[f"q{r}"])
                    st_["decisions_bitwise_vs_release"] = bitwise(hard, rel[f"hard{r}"])
                    st_["alphabet_equals_release"] = int(rel[f"alpha{r}"]) == pol.T
                    st_["row_id_equal"] = bool(np.array_equal(rel["row_id"], D.row_id))
                    # class preservation, every row of every role
                    eq = (hard == d[r]) & (q.argmax(1) == d[r]) & (rel[f"hard{r}"] == d[r])
                    st_["class_preserved_all_rows"] = bool(eq.all())
                    st_["class_preservation_failures_by_role"] = {ro: int((~eq & D.mask[ro]).sum()) for ro in ROLES}
                    if not eq.all():
                        cp_fail.append(f"{cid}#s{k} r{r}")
                    for ro in ROLES:
                        cp_rows[ro] += int((eq & D.mask[ro]).sum())
                    # fallback token use by role (predicted classes absent from fitting)
                    fbt = pol.token_fb[tok]
                    fu = {ro: int((fbt & D.mask[ro]).sum()) for ro in ROLES}
                    fkey = f"{t}|s{k}|recipient_{r}|{'DIRECT' if fam == 'DIRECT-TASK' else 'fine-state'}"
                    fallback_use.setdefault(fkey, {"rows_by_role": fu, "fallback_tokens": int(pol.token_fb.sum()),
                                                   "policies": 0, "identical_across_policies": True})
                    fallback_use[fkey]["policies"] += 1
                    fallback_use[fkey]["identical_across_policies"] &= fallback_use[fkey]["rows_by_role"] == fu
                    # adversarial rows through the deployment rule
                    for Pa in adv[KS[r - 1]]:
                        da = Pa.argmax(1)
                        fa_ = assign(Pa, da, pol.fine)
                        ta = pol.cell_token[fa_]
                        if not (np.array_equal(pol.token_class[ta], da) and np.array_equal(pol.proto[ta].argmax(1), da)):
                            adversarial_fail.append(f"{cid}#s{k} r{r}")
                            break
                    # receipts r1/r2
                    rr = rec["receipts"][f"r{r}"]
                    st_["receipt_tokens_states_fallbacks_equal_own"] = (
                        rr["tokens"] == pol.T and rr["tokens_per_class"] == pol.tokens_per_class()
                        and rr["effective_states_per_class"] == pol.effective_per_class()
                        and rr["fallback_tokens"] == [int(x) for x in np.flatnonzero(pol.token_fb)]
                        and rr["fingerprint"] == pol.fingerprint())
                    st_["unsmoothed_vs_smoothed_own"] = pol.unsmoothed_vs_smoothed()
                    st_["unsmoothed_vs_smoothed_receipt_abs_diff"] = abs(rr["max_abs_unsmoothed_vs_smoothed"] -
                                                                         st_["unsmoothed_vs_smoothed_own"])
                    st_["token_states_fit_own"] = int(len(np.unique(tok[fit])))
                    st_["token_states_fit_equal_record"] = st_["token_states_fit_own"] == rec["token_states_fit"][str(r)]
                    st_["token_n_sum_equals_fit_rows"] = int(pol.token_n.sum()) == len(fit)
                    bad = [kk for kk, v in st_.items() if isinstance(v, bool) and not v]
                    if bad or st_["unsmoothed_vs_smoothed_receipt_abs_diff"] > 1e-15:
                        fails.append(f"r{r}: {bad}")
                    info[f"recipient_{r}"] = st_
                cache["inner"][(k, cid)] = {"p1": qs[1][sel_rows], "p2": qs[2][sel_rows],
                                            "hard1": pols[1].token_class[toks[1]][sel_rows],
                                            "hard2": pols[2].token_class[toks[2]][sel_rows]}
                cache["token_states"][(k, cid)] = int(len(np.unique(toks[1][fit])) + len(np.unique(toks[2][fit])))
                if cid in keep_full:
                    cache["full"][(k, cid)] = {"tok1": toks[1], "tok2": toks[2], "p1": qs[1], "p2": qs[2],
                                               "hard1": pols[1].token_class[toks[1]],
                                               "hard2": pols[2].token_class[toks[2]],
                                               "alpha1": pols[1].T, "alpha2": pols[2].T}
                # objectives on OSF_DEFENSE_FIT from own contingency tables
                o = objectives(P[1][fit], toks[1][fit], qs[1][fit], P[2][fit], toks[2][fit], qs[2][fit], s_fit, lam)
                o["D1_sufficient_stats"] = D_from_sums(pols[1], P[1][fit], toks[1][fit])
                o["D2_sufficient_stats"] = D_from_sums(pols[2], P[2][fit], toks[2][fit])
                fin = rec["receipts"]["final"]
                keys = [x for x in ("D1", "D2", "I1", "I2", "I12", "F_task", "F_local", "F_joint") if x in fin]
                dif = max(abs(o[x] - fin[x]) for x in keys)
                dif_rl = max(abs(o[x] - rec["receipts"]["row_level_check"][x]) for x in keys)
                dif_ss = max(abs(o["D1"] - o["D1_sufficient_stats"]), abs(o["D2"] - o["D2_sufficient_stats"]))
                info["objectives_own"] = {x: o[x] for x in ("D1", "D2", "I1", "I2", "I12")}
                info["objective_max_abs_diff_vs_receipt_final"] = dif
                info["objective_max_abs_diff_vs_receipt_row_level"] = dif_rl
                info["D_rowwise_vs_sufficient_stats"] = dif_ss
                info["identity_I12_ge_max_local"] = bool(o["I12"] >= max(o["I1"], o["I2"]) - 1e-12)
                if dif > OBJ_TOL or dif_rl > OBJ_TOL or dif_ss > OBJ_TOL or not info["identity_I12_ge_max_local"]:
                    fails.append("objectives")
                my_obj[(k, t, cid)] = o
                my_fp[(k, t, cid)] = pair_fp
                release_hash[(k, t, cid)] = sha_bytes(toks[1], toks[2], qs[1], qs[2])
                own_maps[cid] = {r: pols[r].cell_token for r in (1, 2)}
                info["failures"] = fails
                for r in (1, 2):
                    for kk, v in info[f"recipient_{r}"].items():
                        if isinstance(v, bool):
                            flag_counts.setdefault(kk, [0, 0])
                            flag_counts[kk][0] += int(v)
                            flag_counts[kk][1] += 1
                compact = {"objectives_own": info["objectives_own"],
                           "objective_max_abs_diff_vs_receipt_final": dif, "D_rowwise_vs_sufficient_stats": dif_ss,
                           "alphabet": [pols[1].T, pols[2].T],
                           "token_states_fit": [info["recipient_1"]["token_states_fit_own"],
                                                info["recipient_2"]["token_states_fit_own"]],
                           "unsmoothed_vs_smoothed": [info["recipient_1"]["unsmoothed_vs_smoothed_own"],
                                                      info["recipient_2"]["unsmoothed_vs_smoothed_own"]],
                           "failures": fails}
                if fails:
                    compact["detail"] = info
                pol_out[f"{cid}#s{k}"] = res("FAIL" if fails else "PASS", **compact)
            # ---- sequential stage-one coefficient
            for cid in ids:
                pc = parse_cid(cid)
                if pc["family"] not in ("SEQ-12", "SEQ-21") or cid not in recs:
                    continue
                rcp = recs[cid]["receipts"]
                lam = pc["lam"]
                a = 1 if pc["family"] == "SEQ-12" else 2
                o = my_obj[(k, t, cid)]
                ta = own_maps[cid][a][fa[a][fit]]
                const = np.zeros_like(ta)
                I_ac = mi_labels(s_fit, ta, const) if a == 1 else mi_labels(s_fit, const, ta)
                I_c = mi_labels(s_fit, const)
                stage1 = o[f"D{a}"] + 1.5 * lam * o[f"I{a}"]
                fj_const_minus_db = o[f"D{a}"] + lam * ((o[f"I{a}"] + I_c) / 2 + I_ac)
                si = rcp["stage1_identity"]
                seq_out[f"{cid}#s{k}"] = {
                    "coefficient_recorded": rcp["stage1_coefficient"], "coefficient_ok": rcp["stage1_coefficient"] == 1.5 * lam,
                    "own_identity_abs_diff": abs(fj_const_minus_db - stage1), "own_I_with_constant": I_c,
                    "receipt_stage1_objective_abs_diff": abs(si["stage1_objective"] - stage1),
                    "receipt_identity_abs_diff": si["abs_diff"], "order": rcp["order"]}
            # ---- own search replay
            todo = []
            if search != "none":
                for cid in ids:
                    pc = parse_cid(cid)
                    if pc["family"] == "DIRECT-TASK" or cid not in recs:
                        continue
                    if search == "sample" and not (pc["m"] == 4 and pc["lam"] in (None, 1.0)) and pc["family"] != "CLASS":
                        continue
                    todo.append(cid)
            own_result = {}
            order = sorted(todo, key=lambda c_: (parse_cid(c_)["family"] == "JOINT", c_))
            for cid in order:
                pc = parse_cid(cid)
                fam, m, lam = pc["family"], pc["m"], pc["lam"]
                t0 = time.process_time()
                wit = None
                if fam == "JOINT":
                    wit = {}
                    for wf in WITNESS_FAMS:
                        wc = config_id(t, wf, m, lam)
                        if wc in own_result:
                            wit[wf] = {r: canonical(own_result[wc][r]) for r in (1, 2)}
                        else:
                            wit[wf] = own_maps[wc]
                lab, log = own_family(eng, fam, m, lam, wit)
                own_result[cid] = lab
                cmpr = {"final_map_equals_policy": all(np.array_equal(canonical(lab[r]), own_maps[cid][r]) for r in (1, 2)),
                        "cpu_s": time.process_time() - t0}
                if fam != "CLASS":
                    cmpr.update(compare_family_log(fam, log, recs[cid]["receipts"], lam))
                if fam in ("SEQ-12", "SEQ-21"):
                    cmpr["first_map_never_revised"] = log["first_map_unchanged_in_stage2"]
                    if fam == "SEQ-12":
                        cmpr["first_map_equals_policy_r1"] = bool(np.array_equal(canonical(lab[1]), own_maps[cid][1]))
                ok = cmpr["final_map_equals_policy"] and all(v for kk, v in cmpr.items() if isinstance(v, bool)) and \
                    cmpr.get("max_abs_diff_recorded_vs_own", 0.0) <= LOG_TOL and \
                    cmpr.get("after_greedy_max_abs_diff", 0.0) <= LOG_TOL and \
                    cmpr.get("candidate_F_joint_max_abs_diff", 0.0) <= LOG_TOL
                search_out[f"{cid}#s{k}"] = res("PASS" if ok else "FAIL", **cmpr)
            # ---- JOINT dominance over the bank witnesses (own row-level objectives)
            for cid in ids:
                pc = parse_cid(cid)
                if pc["family"] != "JOINT" or cid not in recs:
                    continue
                m, lam = pc["m"], pc["lam"]
                fj_joint = my_obj[(k, t, cid)]["F_joint"]
                rd = recs[cid]["receipts"]
                wv, wfp = {}, {}
                for wf in WITNESS_FAMS:
                    wc = config_id(t, wf, m, lam)
                    ow = my_obj.get((k, t, wc))
                    if ow is None:
                        continue
                    wv[wf] = F_values(ow, lam)["F_joint"]
                    wfp[wf] = rd["starts"][wf]["witness_fingerprint"] == my_fp[(k, t, wc)]
                margins = {wf: fj_joint - v for wf, v in wv.items()}
                dom_out[f"{cid}#s{k}"] = {
                    "own_F_joint": fj_joint, "own_witness_F_joint": wv, "final_minus_witness": margins,
                    "dominates_all": len(wv) == 4 and all(x <= 1e-12 for x in margins.values()),
                    "recorded_witnesses_are_the_bank_units": wfp,
                    "recorded_witness_sources": {wf: rd["starts"][wf].get("source") for wf in WITNESS_FAMS},
                    "recorded_winner": "%s:%s" % (rd["winner"]["start"], rd["winner"]["kind"]),
                    "winner_is_min_of_candidates": rd["winner"]["F_joint"] == min(c_["F_joint"] for c_ in rd["candidates"]),
                    "recorded_dominance_matches_own": max(abs(rd["witness_dominance"][wf]["final_minus_witness"] -
                                                              margins[wf]) for wf in margins) <= OBJ_TOL}
    # ---- aliases
    groups = {}
    for (k, t, cid), fp in my_fp.items():
        groups.setdefault((k, t, fp), []).append(cid)
    mine_groups = sorted([{"seed": k, "teacher": t, "configs": sorted(v)} for (k, t, _), v in groups.items() if len(v) > 1],
                         key=lambda x: (x["seed"], x["teacher"], x["configs"]))
    orr = jload(RES / "OPTIMIZATION_RECEIPTS.json") if (RES / "OPTIMIZATION_RECEIPTS.json").exists() else {}
    theirs = sorted([{"seed": g_["seed"], "teacher": g_["teacher"], "configs": sorted(g_["configs"])}
                     for g_ in orr.get("alias_groups", [])], key=lambda x: (x["seed"], x["teacher"], x["configs"]))
    alias_release_identical = all(len({release_hash[(g_["seed"], g_["teacher"], c_)] for c_ in g_["configs"]}) == 1
                                  for g_ in mine_groups)
    # release-content groups (independent of the fingerprint convention)
    cg = {}
    for key, h in release_hash.items():
        cg.setdefault((key[0], key[1], h), []).append(key[2])
    content_groups = sorted([{"seed": k, "teacher": t, "configs": sorted(v)} for (k, t, _), v in cg.items() if len(v) > 1],
                            key=lambda x: (x["seed"], x["teacher"], x["configs"]))
    aliases = res("PASS" if mine_groups == theirs and alias_release_identical and content_groups == mine_groups else "FAIL",
                  own_alias_groups=mine_groups, n_own_groups=len(mine_groups),
                  equals_OPTIMIZATION_RECEIPTS=mine_groups == theirs, releases_identical_within_groups=alias_release_identical,
                  release_content_groups_equal_fingerprint_groups=content_groups == mine_groups,
                  aliased_slots=sum(len(g_["configs"]) - 1 for g_ in mine_groups),
                  physical_fits=len(my_fp), distinct_pair_maps=len(groups))
    # ---- summaries
    pol_fail = sorted(k_ for k_, v in pol_out.items() if v["status"] != "PASS")
    obj_max = max((v.get("objective_max_abs_diff_vs_receipt_final", 0) for v in pol_out.values()), default=None)
    policies = res("PASS" if not pol_fail and len(pol_out) == 258 else "FAIL", units=len(pol_out), failing=pol_fail,
                   objective_max_abs_diff_vs_receipts=obj_max,
                   flag_true_counts_over_policy_recipients={kk: {"true": v[0], "of": v[1]} for kk, v in flag_counts.items()},
                   all_flags_true=all(v[0] == v[1] for v in flag_counts.values()),
                   per_unit=pol_out)
    cpres = res("PASS" if not cp_fail and not adversarial_fail and len(pol_out) == 258 else "FAIL",
                rule="released decision == own teacher argmax (first index) == argmax of the decoded prototype, every "
                     "row of every role (decisions are label-free; no assessment label read)",
                row_checks_passed_by_role=cp_rows, rows_per_policy_recipient={r: int(D.mask[r].sum()) for r in ROLES},
                policy_recipients_checked=2 * len(pol_out), failures=cp_fail,
                adversarial_rows={"per_K": {K: int(sum(len(a) for a in adv[K])) for K in adv},
                                  "cases": "two-way ties at every position pair, one-hot rows, 1e-300 underflow, uniform, "
                                           "unseen fitting class (incl. occupation class 5)",
                                  "failures": adversarial_fail},
                fallback_token_use_by_role=fallback_use)
    dom_fail = sorted(k_ for k_, v in dom_out.items() if not (v["dominates_all"] and all(v["recorded_witnesses_are_the_bank_units"].values())
                                                             and v["winner_is_min_of_candidates"] and v["recorded_dominance_matches_own"]))
    dom = res("PASS" if not dom_fail and len(dom_out) == 54 else "FAIL", units=len(dom_out), failing=dom_fail,
              min_margin_witness_minus_joint=min((-x for v in dom_out.values() for x in v["final_minus_witness"].values()),
                                                 default=None),
              joint_equal_to_a_witness=sum(1 for v in dom_out.values() if any(abs(x) <= 1e-15 for x in v["final_minus_witness"].values())),
              winners={w: sum(1 for v in dom_out.values() if v["recorded_winner"] == w)
                       for w in sorted({v["recorded_winner"] for v in dom_out.values()})},
              note="fine-state family and fitting objective only; no implication for assessment recovery, global "
                   "optimality or DIRECT-TASK", per_unit=dom_out)
    seq_bad = sorted(k_ for k_, v in seq_out.items() if not (v["coefficient_ok"] and v["own_identity_abs_diff"] <= 1e-12
                                                              and abs(v["own_I_with_constant"]) <= 1e-15
                                                              and v["receipt_stage1_objective_abs_diff"] <= OBJ_TOL))
    seq = res("PASS" if not seq_bad and len(seq_out) == 108 else "FAIL", units=len(seq_out), failing=seq_bad,
              max_own_identity_abs_diff=max((v["own_identity_abs_diff"] for v in seq_out.values()), default=None),
              max_receipt_stage1_objective_abs_diff=max((v["receipt_stage1_objective_abs_diff"] for v in seq_out.values()),
                                                        default=None))
    s_fail = sorted(k_ for k_, v in search_out.items() if v["status"] != "PASS")
    label_ties = sorted(k_ for k_, v in search_out.items() if v.get("winner_label", "exact") != "exact")
    expected_search = 0 if search == "none" else len(search_out)
    srch = res(("PENDING" if search == "none" else ("PASS" if not s_fail else "FAIL")), mode=search,
               units_replayed=len(search_out), failing=s_fail,
               merges_checked=sum(v.get("merges_checked", 0) for v in search_out.values()),
               moves_checked=sum(v.get("moves_checked", 0) for v in search_out.values()),
               max_abs_diff_recorded_vs_own=max((v.get("max_abs_diff_recorded_vs_own", 0) for v in search_out.values()),
                                                default=None),
               cpu_s=sum(v.get("cpu_s", 0) for v in search_out.values()),
               joint_winner_label_tie_equivalent=label_ties,
               joint_winner_note="the receipt's winning START label differs from the own exact-tie order only where "
                                 "several starts reach the identical final map (objectives within 1e-15); the map, "
                                 "every merge, move and sweep match; descriptive winner counts are therefore fragile",
               note="own greedy agglomeration (all eligible same-class merges, exact from-scratch objective of every "
                    "candidate, tie tol 1e-12 lexicographic) and exchange refinement (strict decrease < -1e-12, <= 5 "
                    "sweeps, ties to the lowest coarse label); DIRECT-TASK is checked through its own k-means instead",
               per_unit=search_out, expected=expected_search)
    fine = res(worst(*[v["status"] for v in fine_out.values()]), units=fine_out)
    reports = check_public_reports(pol_out, my_fp)
    return {"fine_partitions": fine, "policies": policies, "class_preservation": cpres, "search_replay": srch,
            "joint_dominance": dom, "sequential_stage_one": seq, "aliases": aliases,
            "lead_fit_stage_reports": reports}, my_obj


def check_public_reports(pol_out, my_fp):
    """The lead's public fit-stage reports (CLASS_PRESERVATION.json, OPTIMIZATION_RECEIPTS.json) vs own numbers."""
    out, bad = {}, []
    cp_p, or_p = RES / "CLASS_PRESERVATION.json", RES / "OPTIMIZATION_RECEIPTS.json"
    if cp_p.exists():
        cp = jload(cp_p)
        n = 0
        for un, v in cp.get("units", {}).items():
            key = f"{v['config']}#s{v['seed']}"
            mine = pol_out.get(key)
            ok = mine is not None and mine["status"] == "PASS" and v["rows"] == 39170 and \
                all(v["decisions_equal_teacher"].values()) and v["alphabet"] == mine["alphabet"] and \
                [v["fitting_token_states"]["1"], v["fitting_token_states"]["2"]] == mine["token_states_fit"]
            n += int(ok)
            if not ok:
                bad.append(f"CLASS_PRESERVATION {key}")
        out["class_preservation_json"] = {"units": len(cp.get("units", {})), "equal_to_own": n,
                                          "all_pass_field": cp.get("all_pass")}
    if or_p.exists():
        orr = jload(or_p)
        n, mx = 0, 0.0
        for key, v in orr.get("units", {}).items():
            mine = pol_out.get(key)
            cid, s_ = key.split("#s")
            fp_ok = my_fp.get((int(s_), parse_cid(cid)["teacher"], cid)) == v.get("fingerprint")
            if mine is None:
                bad.append(f"OPTIMIZATION_RECEIPTS {key}")
                continue
            dif = max(abs(v["final"][x] - mine["objectives_own"][x]) for x in ("D1", "D2", "I1", "I2", "I12"))
            mx = max(mx, dif)
            ok = fp_ok and dif <= OBJ_TOL and [v["token_states_fit"]["1"], v["token_states_fit"]["2"]] == mine["token_states_fit"]
            n += int(ok)
            if not ok:
                bad.append(f"OPTIMIZATION_RECEIPTS {key}")
        out["optimization_receipts_json"] = {"units": len(orr.get("units", {})), "equal_to_own": n,
                                             "max_abs_objective_diff": mx}
    present = cp_p.exists() and or_p.exists()
    return res(("PASS" if not bad else "FAIL") if present else "PENDING", failures=bad[:20], **out)


def mutation_power(D: Data, T, L, k=0, t="U", m=4, lam=1.0):
    """Real-data mutations that each checker must detect (seed 0, teacher U). Nothing is written anywhere."""
    import copy
    fit = D.fit_idx
    s_fit = L["sex"][fit]
    tt = T[(k, t)]
    P = {1: tt["p1"], 2: tt["p2"]}
    d = {1: tt["d1"], 2: tt["d2"]}
    fines = {r: fit_fine(P[r][fit], d[r][fit], KS[r - 1]) for r in (1, 2)}
    fa = {r: assign(P[r], d[r], fines[r]) for r in (1, 2)}
    cid = config_id(t, "JOINT", m, lam)
    ud = UNITS / pol_unit(k, cid)
    pj = jload(ud / "policy.json")
    rel = np.load(ud / "release.npz", allow_pickle=False)
    rec = jload(ud / "record.json")
    out = {}
    # 1. a stored prototype edited by 1e-9 -> prototype recomputation from the cell sums disagrees
    z = copy.deepcopy(pj["p2"])
    z["token_proto"][0][0] += 1e-9
    out["edited_prototype_detected"] = not Pol(z).structure()["prototypes_bitwise_equal_stored"]
    # 2. one fine-cell sum edited -> own k-means statistics and the recomputed prototype disagree
    z = copy.deepcopy(pj["p1"])
    z["fine"]["S"][0][0] += 1e-6
    pz = Pol(z)
    out["edited_fine_sum_detected"] = (not all(fines[1].equal(pz.fine).values())) and \
        not pz.structure()["prototypes_bitwise_equal_stored"]
    # 3. a cell mapped into a token of another class -> class mixing / strict-argmax failure
    z = copy.deepcopy(pj["p1"])
    ct = np.asarray(z["cell_token"])
    c0 = int(np.flatnonzero(np.asarray(z["fine"]["cell_class"]) == 0)[0])
    c1tok = int(ct[np.flatnonzero(np.asarray(z["fine"]["cell_class"]) == 1)[0]])
    ct[c0] = c1tok
    z["cell_token"] = ct.tolist()
    try:
        st_ = Pol(z).structure()
        out["cross_class_token_detected"] = (not st_["no_class_mixing"]) or (not st_["canonical_ids"])
    except Exception:  # noqa: BLE001
        out["cross_class_token_detected"] = True
    # 4. a planted within-class SEX bit in the released token -> not reproducible from p alone; huge I(S;C)
    pol2 = Pol(pj["p2"])
    tok = pol2.cell_token[fa[2]]
    planted = tok.copy()
    planted[fit] = 2 * tok[fit] + s_fit
    out["planted_sex_token_detected"] = (not bitwise(planted, tok)) and \
        mi_labels(s_fit, planted[fit]) > mi_labels(s_fit, tok[fit]) + 0.3
    out["planted_sex_token_I_gain"] = mi_labels(s_fit, planted[fit]) - mi_labels(s_fit, tok[fit])
    # 5. one released token swapped for another token of the same class -> token check fails, decision check passes
    rt = rel["tok2"].copy()
    j = int(np.flatnonzero(pol2.token_class[rt] == 4)[0])
    alt = [x for x in np.flatnonzero(pol2.token_class == 4) if x != rt[j]][0]
    rt[j] = alt
    out["same_class_token_swap_detected_by_token_check_only"] = (not bitwise(rt, tok)) and \
        bool(np.array_equal(pol2.token_class[rt], d[2]))
    # 6. a weak 'joint' (the FINE-TASK map relabelled as JOINT) violates dominance over LOCAL / SEQ
    def fj_of(c_):
        pj_ = jload(UNITS / pol_unit(k, c_) / "policy.json")
        p1_, p2_ = Pol(pj_["p1"]), Pol(pj_["p2"])
        t1, t2 = p1_.cell_token[fa[1]], p2_.cell_token[fa[2]]
        return F_values(objectives(P[1][fit], t1[fit], p1_.proto[t1][fit], P[2][fit], t2[fit], p2_.proto[t2][fit],
                                   s_fit, lam), lam)["F_joint"]
    fj_task = fj_of(config_id(t, "FINE-TASK", m))
    fj_wit = {wf: fj_of(config_id(t, wf, m, lam)) for wf in ("LOCAL", "SEQ-12", "SEQ-21")}
    out["weak_joint_detected"] = any(fj_task > v + 1e-12 for v in fj_wit.values())
    # 7. a dropped strong initialisation: remove the winning start's candidates; dominance over it must fail
    win = rec["receipts"]["winner"]["start"]
    rest = [c_["F_joint"] for c_ in rec["receipts"]["candidates"] if c_["start"] != win]
    wit_unchanged = rec["receipts"]["witness_dominance"].get(win, {}).get("witness_F_joint")
    out["dropped_winning_start"] = win
    out["dropped_initialisation_detected"] = bool(win != "JOINT-GREEDY" and wit_unchanged is not None and
                                                  min(rest) > wit_unchanged + 1e-12) if win != "JOINT-GREEDY" else None
    # 8. a recorded move delta edited by 1e-8 -> the receipt comparison exceeds LOG_TOL
    eng = Eng(fines[1], fines[2], fa[1][fit], fa[2][fit], s_fit)
    wit = {wf: {r: Pol(jload(UNITS / pol_unit(k, config_id(t, wf, m, None if wf == "FINE-TASK" else lam)) /
                             "policy.json")[f"p{r}"]).cell_token for r in (1, 2)} for wf in WITNESS_FAMS}
    _, log = own_family(eng, "JOINT", m, lam, wit)
    rr = copy.deepcopy(rec["receipts"])
    rr["starts"]["JOINT-GREEDY"]["refine_moves"][0]["delta"] += 1e-8
    c_ok = compare_family_log("JOINT", log, rec["receipts"], lam)
    c_bad = compare_family_log("JOINT", log, rr, lam)
    out["edited_move_delta_detected"] = c_ok["max_abs_diff_recorded_vs_own"] <= LOG_TOL < c_bad["max_abs_diff_recorded_vs_own"]
    # 9. a planted SEX-keyed fine partition is not a nearest-centroid partition of the teacher scores
    cls = d[1][fit]
    lab = cls * 2 + s_fit
    Fp = 4
    n, S, A = cell_stats(P[1][fit], lab, Fp)
    planted_f = Fine(2, np.array([0, 0, 1, 1]), smooth(S / n[:, None], [0, 0, 1, 1]), S / n[:, None], n, S, A,
                     np.zeros(4, bool))
    dep = assign(P[1][fit], d[1][fit], planted_f)
    n2, S2, A2 = cell_stats(P[1][fit], dep, Fp)
    out["planted_sex_partition_detected"] = not (np.array_equal(n2, n) and np.array_equal(S2, S))
    ok = all(v for kk, v in out.items() if isinstance(v, bool)) and out["dropped_initialisation_detected"] is not False
    return res("PASS" if ok else "FAIL", unit=f"{cid}#s{k}", **out)


def adversarial_rows():
    out = {}
    for K in KS:
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
        uni = [np.full(K, 1.0 / K)]
        mild = []
        for i in range(K):
            p = np.full(K, 0.5 / K)
            p[i] += 0.5
            mild.append(p / p.sum())
        out[K] = [np.array(rows), np.array(eye), np.array(under), np.array(uni), np.array(mild)]
    return out


# ------------------------------------------------------------------------------------------------ integrity
def check_complete():
    bad, extra, incomplete, n_ok, kinds, in_progress = [], [], [], 0, {}, 0
    for d in sorted(p for p in UNITS.iterdir() if p.is_dir()):
        name = d.name
        kinds[name.split("__")[0]] = kinds.get(name.split("__")[0], 0) + 1
        cp = d / "COMPLETE.json"
        if not cp.exists():
            if name.startswith(PHASE1_KINDS):
                incomplete.append(name)
            else:
                in_progress += 1
            continue
        c = jload(cp)
        files = c.get("files", {})
        good = c.get("id") == name
        for f_, h in files.items():
            if not (d / f_).exists() or sha_file(d / f_) != h:
                bad.append(f"{name}/{f_}")
                good = False
        present = {str(p.relative_to(d)) for p in d.rglob("*") if p.is_file()} - {"COMPLETE.json"}
        ex = sorted(present - set(files))
        if ex:
            extra.append({name: ex})
        n_ok += int(good)
    exp = {"tea": 6, "ref": 9, "fine": 6, "pol": 258}
    if (RES / "EVALUATION_LOCK.json").exists() and any(n.startswith("outer") for n in kinds):
        exp.update({"inner": 273, "outer": 3 * len(jload(RES / "EVALUATION_LOCK.json")["scored_labels"])})
    counts_ok = all(kinds.get(k_, 0) == v for k_, v in exp.items())
    st = "FAIL" if bad or not counts_ok else ("WARN" if extra or incomplete else "PASS")
    return res(st, units_by_kind=kinds, expected_phase_1=exp, units_with_valid_complete=n_ok, hash_failures=bad,
               files_not_listed=extra, units_without_complete_json=incomplete,
               later_stage_dirs_in_progress_not_checked=in_progress)


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


def activity():
    p = RUN / "ACTIVITY_LOG.jsonl"
    ev = []
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                ev.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return ev


PHASE1_KINDS = ("tea__", "ref__", "fine__", "pol__")
LOCK_ORDER = ["ENGINEERING_LOCK", "TRAINING_LOCK", "SELECTION_AND_AUDIT_LOCK", "EVALUATION_LOCK"]
STAGE_LOCK = {"admit": "ENGINEERING_LOCK", "partition": "TRAINING_LOCK", "fit": "TRAINING_LOCK",
              "inner": "SELECTION_AND_AUDIT_LOCK", "controls": "SELECTION_AND_AUDIT_LOCK", "select": "SELECTION_AND_AUDIT_LOCK",
              "assess": "EVALUATION_LOCK", "outer": "EVALUATION_LOCK", "infer": "EVALUATION_LOCK"}
UNIT_LOCK = {"tea__": "ENGINEERING_LOCK", "ref__": "ENGINEERING_LOCK", "fine__": "TRAINING_LOCK", "pol__": "TRAINING_LOCK",
             "inner__": "SELECTION_AND_AUDIT_LOCK", "ctl__": "SELECTION_AND_AUDIT_LOCK", "outer__": "EVALUATION_LOCK"}


def lock_files():
    out = {}
    for name in LOCK_ORDER + sorted(p.stem for p in RES.glob("AMENDMENT*.json")):
        p = RES / f"{name}.json"
        log = git("log", "--format=%H|%cI", "--", f"{REL_RES}/{name}.json") or ""
        out[name] = {"exists": p.exists(), "commits": [l_.split("|") for l_ in log.splitlines() if l_]}
    return out


def check_lock_order():
    entries = remote_reflog()
    ev = activity()
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
        fp, lp = first_remote(first_c, entries), first_remote(last_c, entries)
        blob_ok = git("rev-parse", f"{last_c}:{REL_RES}/{name}.json") == git("hash-object", str(RES / f"{name}.json"))
        lock = jload(RES / f"{name}.json")
        parent = git("rev-parse", f"{first_c}^")
        info[name] = {"first_commit": first_c, "first_commit_time": parse_iso(first_t), "first_push_time": fp,
                      "latest_commit": last_c, "latest_push_time": lp, "versions": len(d["commits"]),
                      "worktree_equals_latest_commit": blob_ok,
                      "parent_commit_field_equals_git_parent": lock.get("parent_commit") == parent,
                      "written_at": lock.get("written_at")}
        if not blob_ok:
            fails.append(f"{name}: worktree differs from its latest commit")
        if fp is None:
            fails.append(f"{name}: never pushed")
        if lock.get("parent_commit") != parent:
            fails.append(f"{name}: parent_commit field differs from the git parent")
    starts = [e for e in ev if str(e.get("event", "")).startswith("start ") or e.get("event") == "assessment opened"]
    st_out = []
    for e in starts:
        stage = "assess" if e["event"] == "assessment opened" else e["event"].split(" ", 1)[1]
        t = parse_iso(e["at"])
        gov = STAGE_LOCK.get(stage)
        rec = {"stage": stage, "shard": e.get("shard"), "at": t, "lock_named": e.get("lock"), "governing_lock": gov}
        ok = gov is not None and e.get("lock") in LOCK_ORDER and \
            LOCK_ORDER.index(e.get("lock")) >= LOCK_ORDER.index(gov)
        g_ = info.get(gov) or {}
        if not (g_.get("first_push_time") and g_["first_push_time"] <= t):
            ok = False
        unpushed = []
        for name, d in lf.items():
            for c, ct in d["commits"]:
                if parse_iso(ct) <= t:
                    pt = first_remote(c, entries)
                    if pt is None or pt > t:
                        unpushed.append(name)
        rec["locks_committed_but_unpushed_at_start"] = sorted(set(unpushed))
        rec["governing_lock_push_lead_s"] = (t - g_["first_push_time"]).total_seconds() if g_.get("first_push_time") else None
        if unpushed:
            ok = False
        rec["ok"] = ok
        if not ok:
            fails.append(f"start {stage} at {iso(t)}: governing lock not pushed before it (or unpushed lock versions)")
        st_out.append(rec)
    early, by_lock = [], {}
    unit_events = {e.get("unit"): parse_iso(e["at"]) for e in ev if e.get("event") == "unit complete"}
    for d in sorted(p for p in UNITS.iterdir() if p.is_dir()):
        if not (d / "COMPLETE.json").exists():
            continue
        gov = next((v for p_, v in UNIT_LOCK.items() if d.name.startswith(p_)), None)
        if gov is None:
            continue
        t_m = utc((d / "COMPLETE.json").stat().st_mtime)
        t_e = unit_events.get(d.name)
        fpush = (info.get(gov) or {}).get("first_push_time")
        by_lock[gov] = by_lock.get(gov, 0) + 1
        if fpush is None or t_m < fpush or (t_e is not None and t_e < fpush.replace(microsecond=0)):
            early.append({"unit": d.name, "complete_mtime": t_m, "event_at": t_e, "governing_lock": gov})
    if early:
        fails.append(f"{len(early)} units completed before their governing lock was pushed")
    units_without_event = sorted(n for n in (p.name for p in UNITS.iterdir() if p.is_dir()) if n not in unit_events)
    ls = git("ls-remote", "origin", f"refs/heads/{BRANCH}", timeout=60)
    remote_head = ls.split()[0] if ls else None
    tracking = git("rev-parse", f"refs/remotes/origin/{BRANCH}")
    head = git("rev-parse", "HEAD")
    remote = {"ls_remote_head": remote_head, "remote_tracking_head": tracking, "local_head": head,
              "ls_remote_equals_tracking": remote_head == tracking, "local_head_pushed": remote_head == head or
              (remote_head is not None and git_ok("merge-base", "--is-ancestor", head, remote_head))}
    if remote_head is None:
        st = "WARN"
    else:
        st = "PASS"
    if not remote["ls_remote_equals_tracking"] and remote_head is not None:
        st = "WARN"
    return res("FAIL" if fails else st, locks=info, stage_starts=st_out, units_by_governing_lock=by_lock,
               units_completed_before_governing_lock=early[:30], units_without_activity_event=units_without_event[:30],
               remote=remote, failures=fails,
               absent_later_locks=[n for n in LOCK_ORDER if not lf[n]["exists"]])


def check_code_hashes():
    order = []
    for name in LOCK_ORDER + sorted(p.stem for p in RES.glob("AMENDMENT*.json")):
        p = RES / f"{name}.json"
        if p.exists():
            d = jload(p)
            order.append((d.get("written_at", ""), name, d))
    order.sort()
    current, violations, relocks, at_commit = {}, [], [], {}
    lf = lock_files()
    for _, name, d in order:
        cf = d.get("code_files") or {}
        declared = set(d.get("changes_previously_locked") or [])
        commit = lf[name]["commits"][-1][0] if lf[name]["commits"] else None
        mism_c = []
        for f_, h in cf.items():
            b = git_show_bytes(commit, f_) if commit else None
            if b is None or hashlib.sha256(b).hexdigest() != h:
                mism_c.append(f_)
            if f_ in current and current[f_][0] != h:
                if name.startswith("AMENDMENT") and f_ not in declared:
                    violations.append(f"{name}: {f_} changed without being declared")
                if not name.startswith("AMENDMENT") and not current[f_][1].startswith("AMENDMENT") and f_ not in declared:
                    relocks.append(f"{name} re-hashes {f_} (previously {current[f_][1]}) without a declared change")
            current[f_] = (h, name)
        at_commit[name] = {"files": len(cf), "files_differing_from_the_lock_commit": mism_c}
    mism = []
    for f_, (h, src) in sorted(current.items()):
        p = WT / f_
        wh = sha_file(p) if p.exists() else None
        if wh != h:
            mism.append({"file": f_, "governing_lock": src})
    latest = next((nm for _, nm, _ in reversed(order) if not nm.startswith("AMENDMENT")), None)
    docs = {}
    if latest:
        for doc, h in (jload(RES / f"{latest}.json").get("documents_sha256") or {}).items():
            p = RES / doc
            docs[doc] = (sha_file(p) == h) if p.exists() else None
    bad_commit = any(v["files_differing_from_the_lock_commit"] for v in at_commit.values())
    st = "FAIL" if (mism or violations or relocks or bad_commit) else ("WARN" if not all(v for v in docs.values()) else "PASS")
    return res(st, locks_in_order=[nm for _, nm, _ in order], files_governed=len(current), worktree_mismatches=mism,
               lock_vs_its_commit=at_commit, undeclared_amendment_changes=violations, undeclared_relocks=relocks,
               latest_named_lock=latest, latest_named_lock_documents_match_worktree=docs)


def check_predictions():
    rel = f"{REL_RES}/PREDICTIONS.json"
    log = git("log", "--format=%H|%cI", "--", rel) or ""
    commits = [l_.split("|") for l_ in log.splitlines() if l_]
    first = commits[-1] if commits else None
    entries = remote_reflog()
    fit_starts = [parse_iso(e["at"]) for e in activity() if e.get("event") in ("start partition", "start fit")]
    out = {"versions": len(commits), "first_commit": first[0] if first else None,
           "first_commit_is_registered": bool(first and first[0].startswith(PRED_COMMIT)),
           "worktree_equals_first_commit": bool(first) and git("rev-parse", f"{first[0]}:{rel}") ==
           git("hash-object", str(RES / "PREDICTIONS.json")),
           "first_push_time": first_remote(first[0], entries) if first else None,
           "first_fit_start": min(fit_starts) if fit_starts else None}
    out["pushed_before_any_fit"] = bool(out["first_push_time"] and out["first_fit_start"] and
                                        out["first_push_time"] < out["first_fit_start"])
    ok = out["first_commit_is_registered"] and out["worktree_equals_first_commit"] and out["pushed_before_any_fit"] \
        and out["versions"] == 1
    return res("PASS" if ok else "FAIL", **out)


def check_label_custody():
    ev = activity()
    lf = lock_files()
    el = lf.get("EVALUATION_LOCK", {})
    el_push = first_remote(el["commits"][-1][0], remote_reflog()) if el.get("commits") else None
    a_ev = [(e.get("event"), parse_iso(e["at"])) for e in ev if re.search(r"assess|unseal|outer|infer", str(e.get("event", "")))]
    bad_ev = [f"{n} at {iso(t)}" for n, t in a_ev if el_push is None or t < el_push]
    label_like = ("sex", "race", "y_income", "y_occ", "y_occupation_group", "y", "labels", "s", "S")
    found = []
    allowed = {"teacher.npz": {"row_id", "p1", "p2", "d1", "d2", "c1", "c2", "r1", "r2"},
               "assign.npz": {"row_id", "f1", "f2"},
               "release.npz": {"row_id", "tok1", "q1", "hard1", "alpha1", "tok2", "q2", "hard2", "alpha2"}}
    unexpected = []
    for d in sorted(p for p in UNITS.iterdir() if p.is_dir() and p.name.startswith(PHASE1_KINDS)):
        for f_ in d.glob("*.npz"):
            with np.load(f_, allow_pickle=False) as z:
                keys = set(z.files)
            if keys & set(label_like):
                found.append(f"{d.name}/{f_.name}")
            if f_.name in allowed and not d.name.startswith("ref__") and keys != allowed[f_.name]:
                unexpected.append({f"{d.name}/{f_.name}": sorted(keys)})
    later = sorted(p.name for p in UNITS.iterdir() if p.name.startswith(("inner__", "outer__", "ctl__")))
    ok = not bad_ev and not found and not unexpected
    return res("PASS" if ok else "FAIL", assessment_or_unseal_events_before_evaluation_lock_push=bad_ev,
               assessment_events_after_the_push=[f"{n} at {iso(t)}" for n, t in a_ev if el_push and t >= el_push][:10],
               evaluation_lock_first_push=el_push, npz_with_label_like_arrays=found,
               npz_with_unexpected_keys=unexpected, later_stage_units_present=len(later),
               evaluation_lock_present=(RES / "EVALUATION_LOCK.json").exists(),
               selection_lock_present=(RES / "SELECTION_AND_AUDIT_LOCK.json").exists(),
               verifier_reads="SEX of OSF_DEFENSE_FIT rows (objective replay) and task labels of OSF_DEFENSE_FIT / "
                              "HEAD_VALIDATION rows (head refit) only; assessment labels masked to -1 at load",
               fitting_rows_only="every fine partition (incl. DIRECT-TASK) is reproduced bit-exactly from the "
                                 "OSF_DEFENSE_FIT rows alone and every token count sums to 15,434 (fine_partitions, "
                                 "policies); selection and assessment rows only receive deployments")


def check_custody():
    out = {}
    sums = ADMITTED / "SHA256SUMS"
    if sums.exists():
        lines = [l_.split(None, 1) for l_ in sums.read_text().splitlines() if l_.strip()]
        bad = [rel for h, rel in lines if not (ADMITTED / rel.strip()).exists() or sha_file(ADMITTED / rel.strip()) != h]
        listed = {rel.strip() for _, rel in lines}
        present = {str(p.relative_to(ADMITTED)) for p in ADMITTED.rglob("*") if p.is_file()} - {"SHA256SUMS"}
        out["admitted_sha256sums"] = {"files_listed": len(lines), "mismatches": bad,
                                      "unlisted_files": len(present - listed), "unlisted_examples": sorted(present - listed)[:5]}
    adm_in = ADMITTED / "inputs" / "adult_jcv.npz"
    out["admitted_input_copy_sha256_ok"] = adm_in.exists() and sha_file(adm_in) == SRC_SHA
    # drive probe by content (names withheld)
    want = jload(SMF_BACKUP).get("SHA256SUMS_sha256") if SMF_BACKUP.exists() else None
    cand, match = 0, 0
    vroot = Path("/Volumes")
    for n in (sorted(os.listdir(vroot)) if vroot.exists() else []):
        if n == "Macintosh HD":
            continue
        cand += 1
        f_ = vroot / n / "private_smf_v1_20261005" / "SHA256SUMS"
        try:
            if f_.is_file() and want and sha_file(f_) == want:
                match += 1
        except OSError:
            continue
    out["drive_probe"] = {"rule": "<DRIVE_ROOT>/<volume>/private_smf_v1_20261005/SHA256SUMS has the sha256 recorded in "
                                  "the closed smf BACKUP_VERIFICATION.json", "candidate_volumes": cand,
                          "matching_volumes": match, "known_drive_mounted": match > 0}
    local = sorted(p.name for p in CACHE.glob("dpc_v1_local_copy_*"))
    out["same_device_copy_present"] = bool(local)
    out["same_device_copy_folders"] = len(local)
    out["expected_later"] = ("drive absent: a versioned same-device copy with hashes and restore checks, "
                             "OFF_DEVICE_BACKUP_PENDING and exact new-study + predecessor commands are expected at closeout; "
                             "a same-device copy is not a drive restore") if match == 0 else "drive mounted: backup expected"
    ok = out.get("admitted_sha256sums", {}).get("mismatches", ["missing"]) == [] and out["admitted_input_copy_sha256_ok"]
    return res("INFO" if ok else "FAIL", **out)


# ------------------------------------------------------------------------------------------------ selftests
def selftests():
    out = {}
    _IN_SELFTEST[0] = True
    tried = []
    for m in ("dpc", "dpc.release", "dpc.compress", "osf.data", "smf.train", "rgj.finalize", "jcv.train", "oar.study",
              "pnx", "stored_model_eval.pilot_infer"):
        try:
            importlib.import_module(m)
            tried.append((m, False))
        except ImportError:
            tried.append((m, True))
    # unpickling a reference to a project class must be refused as well
    blob = b"cdpc.partition\nFinePartition\n."
    try:
        pickle.loads(blob)
        unp = False
    except ImportError:
        unp = True
    except Exception:  # noqa: BLE001
        unp = False
    _IN_SELFTEST[0] = False
    out["import_guard_blocks"] = res("PASS" if all(b for _, b in tried) and unp else "FAIL", tried=tried,
                                     unpickle_reference_refused=unp)
    rng = np.random.default_rng(11)
    # forward pass equals torch.nn.Sequential
    seq = torch.nn.Sequential(torch.nn.Linear(83, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(),
                              torch.nn.Linear(64, 16))
    sd = {f"enc.0.{k}": v for k, v in seq.state_dict().items()}
    Xs = rng.normal(size=(40, 83)).astype(np.float32)
    with torch.no_grad():
        ref_ = seq(torch.from_numpy(Xs)).double().numpy()
    out["forward_pass"] = res("PASS" if np.array_equal(encode(sd, 0, Xs), ref_) else "FAIL")
    # smoothing: sum 1, strict argmax even for uniform / tie means
    Q = smooth(np.array([[0.5, 0.5], [0.5, 0.5]]), np.array([0, 1]))
    Q6 = smooth(np.full((6, 6), 1 / 6), np.arange(6))
    out["smoothing"] = res("PASS" if strict_argmax_ok(Q, [0, 1]) and strict_argmax_ok(Q6, np.arange(6)) and
                           np.abs(Q6.sum(1) - 1).max() <= 1e-15 else "FAIL")
    # masked KL: zeros give no NaN, equals the direct formula
    Pz = np.array([[1.0, 0.0], [0.3, 0.7]])
    Qz = smooth(np.array([[0.9, 0.1], [0.2, 0.8]]), [0, 1])
    kd = np.array([[sum(p * math.log(p / q) for p, q in zip(pr, qr) if p > 0) for qr in Qz] for pr in Pz])
    out["masked_kl"] = res("PASS" if np.isfinite(kl_matrix(Pz, Qz)).all() and np.abs(kl_matrix(Pz, Qz) - kd).max() < 1e-15
                           else "FAIL")
    # plug-in MI vs H(S) - H(S|C); invariance to renumbering; coalition XOR fixture; null fixture
    s = rng.integers(0, 2, 4000)
    c = (rng.integers(0, 5, 4000) + s) % 7
    e1 = abs(mi_labels(s, c) - entropy_mi(s, c))
    perm = rng.permutation(7)
    e2 = abs(mi_labels(s, c) - mi_labels(s, perm[c]))
    a1 = rng.integers(0, 2, 4000)
    a2 = a1 ^ s
    xor_ok = mi_labels(s, a1) < 0.005 and mi_labels(s, a2) < 0.005 and mi_labels(s, a1, a2) > 0.6
    null = mi_labels(rng.integers(0, 2, 4000), rng.integers(0, 4, 4000)) < 0.005
    out["plug_in_mi"] = res("PASS" if e1 < 1e-12 and e2 < 1e-15 and xor_ok and null else "FAIL", vs_entropy_form=e1,
                            renumbering=e2, coalition_xor_detected=xor_ok, null_small=null)
    # own k-means: deployment of fitting rows reproduces statistics; fallback for an absent class
    P = rng.dirichlet(np.ones(6), 600)
    P[:, 5] *= 0.01
    P /= P.sum(1, keepdims=True)
    dd = P.argmax(1)
    f6 = fit_fine(P, dd, 6, max_cells=4)
    a6 = assign(P, dd, f6)
    n, S, A = cell_stats(P, a6, f6.F)
    fb_ok = bool(f6.fb.any()) == bool((np.bincount(dd, minlength=6) == 0).any())
    out["own_fine_partition"] = res("PASS" if np.array_equal(n, f6.n) and np.array_equal(S, f6.S) and fb_ok else "FAIL",
                                    fallback_classes=[int(x) for x in f6.cls[f6.fb]])
    # engine: objective from labels equals row-level objective; greedy+refine reaches a map no worse than brute force
    # over a tiny exhaustive fixture with the same cap
    sx = rng.integers(0, 2, 600)
    eng = Eng(f6, f6, a6, a6, sx)
    lab = {1: np.arange(f6.F), 2: np.arange(f6.F)}
    t_ = eng.terms(lab)
    q = smooth(f6.S[a6] / f6.n[a6, None], dd)
    row = np.mean(kl_rows(P, q))
    out["engine_vs_rows"] = res("PASS" if abs(t_["D1"] - row) < 1e-12 and abs(t_["I1"] - mi_labels(sx, a6)) < 1e-12 and
                                abs(t_["I12"] - mi_labels(sx, a6, a6)) < 1e-12 else "FAIL",
                                d_err=abs(t_["D1"] - row))
    # class-preservation detector: a tampered token class is caught
    pol_ok = strict_argmax_ok(np.array([[0.6, 0.4], [0.3, 0.7]]), [0, 1]) and \
        not strict_argmax_ok(np.array([[0.6, 0.4], [0.3, 0.7]]), [1, 1])
    out["class_preservation_detector"] = res("PASS" if pol_ok else "FAIL")
    # fingerprint convention: any change in the map changes the fingerprint
    fpA = hashlib.sha256(np.array([0, 1, 1], "<i8").tobytes()).hexdigest()
    fpB = hashlib.sha256(np.array([0, 1, 2], "<i8").tobytes()).hexdigest()
    out["fingerprint_sensitivity"] = res("PASS" if fpA != fpB else "FAIL")
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
                if st_ in ("FAIL", "WARN") and not path.startswith("selftests"):
                    leaf = not any(isinstance(v, dict) and v.get("status") in ("FAIL", "WARN") for v in node.values())
                    if leaf:
                        flagged.append({"path": path, "status": st_, "cause": node.get("failures") or node.get("failing")
                                        or node.get("reason")})
            for k, v in node.items():
                walk(v, f"{path}.{k}" if path else k)

    for k, v in checks.items():
        walk(v, k)
    return counts_all, flagged


PHASE2 = ("inner_eligibility", "selection", "endpoints", "conjunctions", "attacker_replay", "controls_coverage",
          "composed_source_readers", "post_lock_amendments", "restore_replay")
PHASE2_EXTRA = ("evaluation_lock", "outer_units", "reports", "decision_documents", "phase_2_mutation_power")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest-only", action="store_true")
    ap.add_argument("--no-refit", action="store_true")
    ap.add_argument("--search", default="all", choices=("all", "sample", "none"))
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--phase", type=int, default=1, choices=(1, 2))
    ap.add_argument("--light-update", action="store_true",
                    help="re-run only the light sections over the existing report (heavy sections and hash kept)")
    args = ap.parse_args()
    if args.light_update:
        return light_update(args)
    t0, c0 = time.time(), time.process_time()
    workers_at_start = pgrep_dpc()
    report = {"schema": "dpc-independent-verification-v1", "phase": "PHASE_1",
              "generated_at": iso(datetime.now(timezone.utc)),
              "verifier": f"{REL_RES}/verification/replay_dpc.py", "verifier_sha256": sha_file(Path(__file__)),
              "worktree_head": git("rev-parse", "HEAD"),
              "inputs": {"source_npz": "<PRIVATE_CACHE>/jcv_v1/inputs/adult_jcv.npz",
                         "units": "<PRIVATE_CACHE>/dpc_v1/run/units", "admitted": "<PRIVATE_CACHE>/dpc_v1/admitted",
                         "activity_log": "<PRIVATE_CACHE>/dpc_v1/run/ACTIVITY_LOG.jsonl"},
              "libraries": {"numpy": np.__version__, "torch": torch.__version__, "joblib": joblib.__version__,
                            "sklearn": __import__("sklearn").__version__, "scipy": __import__("scipy").__version__},
              "options": {"refit_heads": not args.no_refit, "search": args.search}}
    checks = {}
    st = selftests()
    checks["selftests"] = {"status": worst(*[v["status"] for v in st.values()]), **st}
    if args.selftest_only:
        print(json.dumps(jsonable(checks["selftests"]), indent=1))
        return
    D = Data()
    L = D.labels()
    checks["roles"] = check_roles(D)
    checks["teachers"], T = check_teachers(D, L, refit=not args.no_refit)
    refs, cache = {}, {}
    checks["references"] = check_references(D, T, keep=refs)
    keep_full = ()
    if args.phase == 2:
        keep_full = tuple(c for c in jload(RES / "EVALUATION_LOCK.json")["scored_labels"] if not c.startswith(("SRC|", "REF|")))
    pc, my_obj = check_fine_and_policies(D, T, L, search=args.search, cache=cache, keep_full=keep_full)
    checks.update(pc)
    checks["mutation_power"] = mutation_power(D, T, L)
    checks["exclusions_and_label_custody"] = check_label_custody()
    checks["integrity"] = {"complete_json": check_complete(), "lock_order": check_lock_order(),
                           "code_hashes": check_code_hashes(), "predictions": check_predictions()}
    checks["integrity"]["status"] = worst(*[v["status"] for v in checks["integrity"].values()])
    checks["custody"] = check_custody()
    if args.phase == 2:
        report["phase"] = "PHASE_2"
        report["phase_2"] = run_phase2(D, L, T, refs, cache, checks, my_obj)
    else:
        for k in PHASE2:
            checks[k] = res("PENDING", reason="PHASE 2 (after the lead reports selection, EVALUATION_LOCK, assessment "
                                              "and inference)")
    loaded = sorted(m for m in sys.modules if m.split(".")[0] in _FORBIDDEN_TOP)
    report["independence"] = res("PASS" if not loaded and not _BLOCKED and not _PRELOADED else "FAIL",
                                 guard="sys.meta_path finder refusing dpc, osf, smf, rgj, jcv, pnx, oar, "
                                       "stored_model_eval (also during joblib unpickling); torch.load(weights_only=True)",
                                 loaded_forbidden_modules=loaded, blocked_attempts_during_run=_BLOCKED,
                                 preloaded_before_guard=_PRELOADED,
                                 libraries_used=["numpy", "scipy", "sklearn", "torch", "joblib", "json", "hashlib", "stdlib"],
                                 reused_from_predecessor_verifier="role reconstruction, forward pass, lock-chronology "
                                                                  "helpers (replay_osf.py, independent code)")
    report["checks"] = checks
    report["verifier_corrections"] = VERIFIER_CORRECTIONS
    status = {k: (v.get("status") if isinstance(v, dict) else None) for k, v in checks.items()}
    counts_all, flagged = walk_flags(checks)
    workers_at_end = pgrep_dpc()
    report["summary"] = {"status_by_check": status, "independence": report["independence"]["status"],
                         "overall_phase_1": worst(*[s_ for k, s_ in status.items() if k not in PHASE2 + PHASE2_EXTRA],
                                                  report["independence"]["status"]),
                         "status_counts_all_nodes": counts_all, "flagged_fail_warn": flagged}
    if args.phase == 2:
        report["summary"]["overall_phase_2"] = worst(*[s_ for s_ in status.values() if s_ != "INFO"],
                                                     report["independence"]["status"])
    report["compute"] = {"wall_s": time.time() - t0, "cpu_s_process": time.process_time() - c0,
                         "threads": {"OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "torch": torch.get_num_threads()},
                         "lead_dpc_workers_at_start": len(workers_at_start), "lead_dpc_workers_at_end": len(workers_at_end),
                         "heavy_processes_by_verifier": 1,
                         "note": "CPU is this process's own process_time (single thread); no other verifier process"}
    report["pending"] = sorted(k for k, v in status.items() if v == "PENDING")
    runs = VERIFIER_RUNS + [{"run": f"this run (phase {args.phase}, search {args.search})", "start": report["generated_at"],
                             "wall_s": round(report["compute"]["wall_s"], 1),
                             "cpu_s": round(report["compute"]["cpu_s_process"], 1), "heavy": True,
                             "lead_workers": f"{len(workers_at_start)} at start, {len(workers_at_end)} at end"}]
    report["verifier_compute_ledger"] = {"runs": runs, "note": VERIFIER_RUNS_NOTE,
                                         "total_cpu_s": round(sum(r["cpu_s"] for r in runs), 1),
                                         "total_cpu_h": round(sum(r["cpu_s"] for r in runs) / 3600, 3)}
    text = json.dumps(jsonable(report), indent=1)
    scrub_check(text)
    if not args.no_write:
        dest = Path(args.out) if args.out else OUT
        tmp = dest.with_suffix(".json.tmp")
        tmp.write_text(text + "\n")
        tmp.replace(dest)
    print(json.dumps(jsonable({"summary": report["summary"], "compute": report["compute"]}), indent=1))


# ================================================================================================ PHASE 2
import warnings  # noqa: E402

from scipy.stats import rankdata  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402

warnings.filterwarnings("ignore")
LOCK_REL = f"{REL_RES}/EVALUATION_LOCK.json"
SLATE = [f"LR_C{C}" for C in (0.01, 0.1, 1.0, 10.0, 100.0)] + ["MLP_64", "MLP_128", "MLP_64x64", "MLP_128x128"] + \
        [f"HGB_{lr}_{lv}" for lr in (0.05, 0.1) for lv in (15, 31)] + ["DA_LR", "DA_MLP"]
CC_A = (0.5, 1.0, 5.0)
CC_LOCAL = [f"CC_alpha{a}" for a in CC_A]
CC_PAIR = [f"CCpair_alpha{a}" for a in CC_A]
VIEWS = ("v1", "v2", "pair")
SRC_FAMS = ("interface", "complete", "scores", "probs", "decisions")
REF_FAMS = {"E": ("interface", "scores", "probs", "decisions", "complete"),
            "F": ("interface", "scores", "probs", "decisions", "cells"),
            "F0": ("interface", "scores", "probs", "decisions", "cells")}
B_BOOT, BOOT_SEED, CHUNK = 1999, 20261007, 250
Z_PRIMARY = 3.1717657833516224
NONJOINT = ("FINE-TASK", "DIRECT-TASK", "LOCAL", "SEQ-12", "SEQ-21")
BUFFER = 0.005
CONTROL_SEED, SPLIT_SEED, NULL_Z, PLANT_MIN, PLANT_NOISE, CONF_ETA = 20261021, 20261022, 3.5, 0.75, 0.20, 0.05


def all_candidates():
    return bank_ids() + [f"SRC|{t}" for t in TEACHERS] + [f"REF|{r}" for r in REFS]


def cand_family(cid):
    return "SRC" if cid.startswith("SRC|") else ("REF" if cid.startswith("REF|") else cid.split("|")[1])


def cand_teacher(cid):
    return cid.split("|")[1] if cid.startswith("SRC|") else (None if cid.startswith("REF|") else cid.split("|")[0])


def release_unit(k, cid):
    if cid.startswith("SRC|"):
        return f"tea__s{k}__{cid.split('|')[1]}"
    if cid.startswith("REF|"):
        return f"ref__s{k}__{cid.split('|')[1]}"
    return pol_unit(k, cid)


def safe_label(label):
    return str(label).replace("*", "star").replace("/", "_").replace("|", "_").replace(" ", "_")


def my_auc(y, s):
    """Mann-Whitney AUC of score s for y == 1 (average ranks for ties); fixed orientation."""
    y = np.asarray(y) == 1
    r = rankdata(np.asarray(s, dtype=np.float64))
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def my_ce(y, p1):
    y = np.asarray(y)
    pt = np.where(y == 1, p1, 1.0 - np.asarray(p1, dtype=np.float64))
    return float(-np.mean(np.log(np.clip(pt, 1e-12, 1.0))))


def pick_first(vals, maximize):
    best = None
    for j, v in enumerate(vals):
        if best is None or (v > vals[best] + 1e-12 if maximize else v < vals[best] - 1e-12):
            best = j
    return best


# ------------------------------------------------------------------------------------------------ unseal gate
def evaluation_lock_gate():
    """The verifier reads assessment labels only after confirming EVALUATION_LOCK.json is committed, unmodified and
    on origin (ls-remote head contains the lock commit, and origin's blob equals the working-tree file)."""
    p = RES / "EVALUATION_LOCK.json"
    out = {"exists": p.exists()}
    if not p.exists():
        return {**out, "ok": False}
    log = git("log", "--format=%H|%cI", "--", LOCK_REL) or ""
    commits = [l_.split("|") for l_ in log.splitlines() if l_]
    first = commits[-1] if commits else None
    ls = git("ls-remote", "origin", f"refs/heads/{BRANCH}", timeout=60)
    remote = ls.split()[0] if ls else None
    blob_ok = bool(commits) and git("rev-parse", f"{commits[0][0]}:{LOCK_REL}") == git("hash-object", str(p))
    origin_blob = git("rev-parse", f"refs/remotes/origin/{BRANCH}:{LOCK_REL}")
    out.update({"commits": len(commits), "commit": first[0] if first else None, "commit_time": first[1] if first else None,
                "first_push_time": first_remote(first[0], remote_reflog()) if first else None,
                "remote_head": remote, "worktree_equals_commit": blob_ok,
                "origin_blob_equals_worktree": origin_blob == git("hash-object", str(p)),
                "commit_on_origin": bool(first and remote and git_ok("merge-base", "--is-ancestor", first[0], remote)),
                "sha256": sha_file(p)})
    out["ok"] = bool(first and blob_ok and out["origin_blob_equals_worktree"] and out["commit_on_origin"] and
                     out["first_push_time"] and len(commits) == 1)
    return out


def unsealed_labels(D: Data, gate):
    if not gate.get("ok"):
        raise RuntimeError("REFUSED: EVALUATION_LOCK not verified on origin; assessment labels stay sealed")
    z = np.load(SRC, allow_pickle=False)
    return {k: z[v].astype(np.int64)[D.keep].copy() for k, v in LABEL_KEYS.items()}


# ------------------------------------------------------------------------------------------------ utility
def const_classes(L, D):
    fit = D.fit_idx
    return {i: int(np.argmax(np.bincount(L[key][fit], minlength=KS[i]))) for i, key in ((0, "y_income"), (1, "y_occ"))}


def my_utility(P, hard, y, K, const):
    P = np.asarray(P, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    ll = -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1.0))
    br = np.sum((P - np.eye(K)[y]) ** 2, 1)
    acc = float(np.mean(np.asarray(hard) == y))
    cnt = np.bincount(y, minlength=K)
    rec = {c: float(np.mean(np.asarray(hard)[y == c] == c)) for c in range(K) if cnt[c]}
    sup = [c for c in range(K) if cnt[c] >= 30]
    conf = P.max(1)
    ok = (np.asarray(hard) == y).astype(np.float64)
    edges = np.linspace(0, 1, 11)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(float(conf[m].mean()) - float(ok[m].mean()))
    mino = min(sup, key=lambda c: cnt[c]) if sup else None
    return {"acc": acc, "logloss": float(ll.mean()), "brier": float(br.mean()), "const_acc": float(np.mean(y == const)),
            "balanced_acc": float(np.mean([rec[c] for c in sup])) if sup else None,
            "minority_recall": rec.get(mino) if mino is not None else None, "ece": float(e), "recalls": rec,
            "ll_rows": ll, "br_rows": br}


def release_of(k, cid, T, refs, cache, rows=None, inner=False):
    """Own release (p1, p2, hard1, hard2) of a candidate, on all rows or the INNER_SELECTION cache."""
    if cid.startswith("SRC|"):
        t = T[(k, cid.split("|")[1])]
        o = {"p1": t["p1"], "p2": t["p2"], "hard1": t["d1"], "hard2": t["d2"]}
    elif cid.startswith("REF|"):
        r = refs[(cid.split("|")[1], k)]
        o = {"p1": r["p1"], "p2": r["p2"], "hard1": r["d1"], "hard2": r["d2"]}
    else:
        if inner:
            return cache["inner"][(k, cid)]
        o = cache["full"][(k, cid)]
    if rows is None:
        return o
    return {kk: np.asarray(o[kk])[rows] for kk in ("p1", "p2", "hard1", "hard2")}


# ------------------------------------------------------------------------------------------------ inner audits
def bank(keys, w):
    """Ordered candidate rows (candidate, view, attacker) of target view w from the stored key order."""
    def atts(view):
        return [k_.split(":", 1)[1] for k_ in keys if k_.split(":", 1)[0] == view]
    if w in ("v1", "v2"):
        return [("own", w, a) for a in atts(w)]
    return [("coalition", "pair", a) for a in atts("pair")] + [("ignore_recipient_2", "v1", a) for a in atts("v1")] + \
        [("ignore_recipient_1", "v2", a) for a in atts("v2")]


def expected_finite(cid, fam):
    if cid.startswith("REF|"):
        lab = cid.split("|")[1]
        return fam in ("decisions", "cells") or (lab in ("F", "F0") and fam != "complete")
    if cid.startswith("SRC|"):
        return fam == "decisions"
    return True


def check_inner(D: Data, L, T, refs, cache):
    """Every inner unit: own AUC / CE of every stored candidate on INNER_SELECTION, own dual selection, slate
    completeness, utility on INNER_SELECTION from own releases. Returns (check, own inner table)."""
    sel = np.flatnonzero(D.mask["INNER_SELECTION"])
    ys = L["sex"][sel]
    assert (ys >= 0).all()
    const = const_classes(L, D)
    yI, yO = L["y_income"][sel], L["y_occ"][sel]
    own, fails, n_cand, max_auc_diff, max_tab_diff, max_util_diff = {}, [], 0, 0.0, 0.0, 0.0
    slate_bad, finite_bad = [], []
    for k in SEEDS:
        for cid in all_candidates():
            name = f"inner__{release_unit(k, cid)}"
            ud = UNITS / name
            if not unit_done(name):
                fails.append(f"{name}: missing")
                continue
            rec = jload(ud / "record.json")
            z = np.load(ud / "inner_preds.npz", allow_pickle=False)
            if not np.array_equal(z["sel_row_id"], D.row_id[sel]):
                fails.append(f"{name}: INNER rows differ")
            fams = {rec["primary_family"]: rec["recovery"], **rec.get("families", {})}
            mine = {}
            for fam, fr in fams.items():
                keys = [str(x) for x in z[f"keys_{fam}"]]
                P = z[f"P_{fam}"]
                fin = bool(fr.get("finite"))
                if fin != expected_finite(cid, fam):
                    finite_bad.append(f"{name}/{fam}")
                for w in VIEWS:
                    want = SLATE + ((CC_PAIR if w == "pair" else CC_LOCAL) if fin else [])
                    got = [k_.split(":", 1)[1] for k_ in keys if k_.split(":", 1)[0] == w]
                    if got != want:
                        slate_bad.append(f"{name}/{fam}/{w}")
                aucs = {k_: my_auc(ys, P[j]) for j, k_ in enumerate(keys)}
                ces = {k_: my_ce(ys, P[j]) for j, k_ in enumerate(keys)}
                n_cand += len(keys)
                mf = {"auc": {}, "ce": {}, "selected": {}, "ce_selected": {}}
                for w in VIEWS:
                    rows = bank(keys, w)
                    va = [aucs[f"{v}:{a}"] for _, v, a in rows]
                    vc = [ces[f"{v}:{a}"] for _, v, a in rows]
                    ia, ic = pick_first(va, True), pick_first(vc, False)
                    mf["auc"][w], mf["ce"][w] = va[ia], vc[ic]
                    mf["selected"][w] = "%s:%s:%s" % rows[ia]
                    mf["ce_selected"][w] = "%s:%s:%s" % rows[ic]
                    max_auc_diff = max(max_auc_diff, abs(va[ia] - fr["auc"][w]), abs(vc[ic] - fr["ce"][w]))
                    if mf["selected"][w] != fr["selected"][w] or mf["ce_selected"][w] != fr["ce_selected"][w]:
                        fails.append(f"{name}/{fam}/{w}: selected {mf['selected'][w]} vs {fr['selected'][w]}")
                    tab = {(r["candidate"], r["view"], r["attacker"]): r for r in fr["tables"][w]}
                    if set(tab) != set(rows):
                        fails.append(f"{name}/{fam}/{w}: bank rows differ")
                    for row in rows:
                        if row in tab:
                            max_tab_diff = max(max_tab_diff, abs(tab[row]["inner_auc"] - aucs[f"{row[1]}:{row[2]}"]),
                                               abs(tab[row]["inner_ce"] - ces[f"{row[1]}:{row[2]}"]))
                mine[fam] = mf
            # utility on INNER_SELECTION from own releases
            o = release_of(k, cid, T, refs, cache, rows=sel, inner=not cid.startswith(("SRC|", "REF|")))
            ut = {0: my_utility(o["p1"], o["hard1"], yI, 2, const[0]), 1: my_utility(o["p2"], o["hard2"], yO, 6, const[1])}
            for i in (0, 1):
                for q in ("acc", "logloss", "brier", "const_acc"):
                    max_util_diff = max(max_util_diff, abs(ut[i][q] - rec["utility"][str(i)][q]))
            own[(k, cid)] = {"families": mine, "primary": rec["primary_family"], "utility": ut,
                             "class_preservation_ok": rec.get("class_preservation_ok", True),
                             "token_states_recorded": rec.get("token_states")}
            if not cid.startswith(("SRC|", "REF|")) and rec.get("token_states") != cache["token_states"][(k, cid)]:
                fails.append(f"{name}: token states")
    bad = fails or slate_bad or finite_bad or max_auc_diff > 1e-12 or max_tab_diff > 1e-12 or max_util_diff > 1e-12
    return res("FAIL" if bad else "PASS", units=len(own), candidates_recomputed=n_cand,
               max_abs_diff_selected_auc_ce=max_auc_diff, max_abs_diff_every_table_entry=max_tab_diff,
               max_abs_diff_inner_utility=max_util_diff, slate_incomplete=slate_bad[:20],
               finite_flag_unexpected=finite_bad[:20], failures=fails[:30],
               rule="own Mann-Whitney AUC and clipped CE of every stored candidate on INNER_SELECTION SEX; banks v_i = own "
                    "slate (+ local CC on finite views); pair = coalition slate (+ CCpair) + ignore-recipient banks; "
                    "first maximum beyond 1e-12 (AUC) / minimum (CE); 15-member FINAL slate required on every view; "
                    "utility recomputed from own releases"), own


# ------------------------------------------------------------------------------------------------ selection
def my_selection(own, cache):
    """Own implementation of the registered inner-only selection (prompt sections 12-13, PROTOCOL 7-8)."""
    cands = all_candidates()
    pol = [c for c in cands if not c.startswith(("SRC|", "REF|"))]
    rows = {}
    for cid in cands:
        seeds = {}
        for k in SEEDS:
            o = own[(k, cid)]
            pf = o["families"][o["primary"]]
            a = dict(pf["auc"])
            winner = {w: "source" for w in VIEWS}
            if cid.startswith("SRC|"):
                t = cid.split("|")[1]
                for p in pol:
                    if cand_teacher(p) != t:
                        continue
                    pa = own[(k, p)]["families"]["code"]["auc"]
                    for w in VIEWS:
                        if pa[w] > a[w]:
                            a[w], winner[w] = pa[w], p
            u, uu = o["utility"], own[(k, "SRC|U")]["utility"]
            gm = {}
            for i in (0, 1):
                c_ = u[i]["const_acc"]
                gm[i] = {"acc": u[i]["acc"] - (uu[i]["acc"] - 0.01), "logloss": (uu[i]["logloss"] + 0.01) - u[i]["logloss"],
                         "brier": (uu[i]["brier"] + 0.005) - u[i]["brier"],
                         "retain": (u[i]["acc"] - c_) - 0.8 * (uu[i]["acc"] - c_), "gain": (u[i]["acc"] - c_) - 0.03}
            sf = sum(max(0.0, -x) for i in gm for x in gm[i].values())
            cp = o["class_preservation_ok"]
            seeds[k] = {"auc": a, "winner": winner, "gm": gm, "shortfall": sf + (0.0 if cp else 1.0),
                        "eligible": sf == 0.0 and cp, "ll": (u[0]["logloss"] + u[1]["logloss"]) / 2,
                        "states": cache["token_states"].get((k, cid))}
        fam = cand_family(cid)
        rows[cid] = {"config": cid, "family": fam, "teacher": cand_teacher(cid),
                     "m": None if fam in ("SRC", "REF") else parse_cid(cid)["m"], "seeds": seeds,
                     "mean_pair": float(np.mean([seeds[k]["auc"]["pair"] for k in SEEDS])),
                     "mean_v1": float(np.mean([seeds[k]["auc"]["v1"] for k in SEEDS])),
                     "mean_v2": float(np.mean([seeds[k]["auc"]["v2"] for k in SEEDS])),
                     "mean_logloss": float(np.mean([seeds[k]["ll"] for k in SEEDS])),
                     "token_states": float("inf") if fam in ("SRC", "REF") else sum(seeds[k]["states"] for k in SEEDS),
                     "task_eligible": all(seeds[k]["eligible"] for k in SEEDS),
                     "gate_shortfall": sum(seeds[k]["shortfall"] for k in SEEDS)}

    def tie(r):
        return (round(r["mean_pair"], 12), round(r["mean_logloss"], 12), r["token_states"], r["config"])

    def evaluate(r, guards):
        exc = {g: {k: {w: r["seeds"][k]["auc"][w] - (gr["seeds"][k]["auc"][w] + BUFFER) for w in ("v1", "v2")}
                   for k in SEEDS} for g, gr in guards.items()}
        pos = sum(max(0.0, x) for g in exc.values() for kk in g.values() for x in kk.values())
        return {"guard_ok": pos == 0.0 and all(x <= 0 for g in exc.values() for kk in g.values() for x in kk.values()),
                "nomination_shortfall": r["gate_shortfall"] + pos}

    def pick(cs, guards=None, none="NO_FEASIBLE_NOMINEE", required=()):
        guards = guards or {}
        ev = []
        for r in cs:
            e = evaluate(r, guards)
            e.update({"r": r, "eligible": r["task_eligible"] and e["guard_ok"] and not required})
            ev.append(e)
        el = [e for e in ev if e["eligible"]]
        if el:
            return {"status": "NOMINEE", "config": min(el, key=lambda e: tie(e["r"]))["r"]["config"]}
        if not ev:
            return {"status": "ABSENT", "config": None}
        b = min(ev, key=lambda e: (round(e["nomination_shortfall"], 12),) + tie(e["r"]))
        blocked = bool(required) and any(e["r"]["task_eligible"] and e["guard_ok"] for e in ev)
        return {"status": "INVALID_MISSING_COMPARATOR" if blocked else none, "config": None,
                "descriptive_config": b["r"]["config"], "nomination_shortfall": b["nomination_shortfall"]}

    nonjoint = [rows[c] for c in pol if rows[c]["family"] in NONJOINT]
    C_match = {f"{t}|m{m}": pick([r for r in nonjoint if r["teacher"] == t and r["m"] == m], none="NO_FEASIBLE_CONTROL")
               for t in TEACHERS for m in RATES}
    C_global = pick(nonjoint + [rows[c] for c in cands if rows[c]["family"] in ("CLASS", "SRC", "REF")],
                    none="NO_FEASIBLE_CONTROL")
    T_star = pick([rows[c] for c in cands if rows[c]["family"] in ("FINE-TASK", "DIRECT-TASK", "CLASS", "SRC")
                   or c == "REF|F0"], none="NO_FEASIBLE_CONTROL")
    # J*: per JOINT configuration, guards = its own C_match and C_global (missing guard -> not eligible)
    jev = []
    for c in pol:
        r = rows[c]
        if r["family"] != "JOINT":
            continue
        cm = C_match[f"{r['teacher']}|m{r['m']}"]
        g, missing = {}, []
        for nm, P in (("C_match", cm), ("C_global", C_global)):
            if P["status"] == "NOMINEE":
                g[nm] = rows[P["config"]]
            else:
                missing.append(nm)
        e = evaluate(r, g)
        e.update({"r": r, "missing": missing, "eligible": r["task_eligible"] and e["guard_ok"] and not missing})
        jev.append(e)
    el = [e for e in jev if e["eligible"]]
    if el:
        J = {"status": "NOMINEE", "config": min(el, key=lambda e: tie(e["r"]))["r"]["config"]}
    else:
        b = min(jev, key=lambda e: (round(e["nomination_shortfall"], 12),) + tie(e["r"]))
        blocked = any(e["r"]["task_eligible"] and e["guard_ok"] and e["missing"] for e in jev)
        J = {"status": "INVALID_MISSING_COMPARATOR" if blocked else "NO_FEASIBLE_NOMINEE", "config": None,
             "descriptive_config": b["r"]["config"], "nomination_shortfall": b["nomination_shortfall"],
             "joint_task_eligible": sum(e["r"]["task_eligible"] for e in jev), "missing_guards": b["missing"]}
    jm = J.get("config") or J.get("descriptive_config")
    J["C_match_key"] = f"{rows[jm]['teacher']}|m{rows[jm]['m']}"
    privacy = [rows[c] for c in pol if rows[c]["family"] in PRIV_FAMS]
    if T_star["status"] == "NOMINEE":
        P_star = pick(privacy, guards={"T*": rows[T_star["config"]]})
    else:
        P_star = pick(privacy)
        if P_star["status"] == "NOMINEE":
            P_star = {"status": "INVALID_MISSING_COMPARATOR", "config": None, "descriptive_config": P_star["config"]}
    statuses = {"J*": J, "P*": P_star, "T*": T_star, "C_global": C_global, "C_match": C_match[J["C_match_key"]]}
    compact = [rows[c] for c in pol if rows[c]["task_eligible"]]
    deploy = min(compact, key=tie)["config"] if compact else None
    # scored set (fixed before the assessment)
    lab = {"SRC|U", "SRC|RAW-J_b0.3", "REF|E", "REF|F", "REF|F0"}
    for t in TEACHERS:
        lab.add(config_id(t, "CLASS"))
        for m in RATES:
            lab |= {config_id(t, "FINE-TASK", m), config_id(t, "DIRECT-TASK", m)}
    for s_ in statuses.values():
        if s_.get("config") or s_.get("descriptive_config"):
            lab.add(s_.get("config") or s_.get("descriptive_config"))
    for f in PRIV_FAMS:
        fr = [rows[c] for c in pol if rows[c]["family"] == f]
        el_ = [r for r in fr if r["task_eligible"]]
        inel = [r for r in fr if not r["task_eligible"]]
        if el_:
            lab.add(min(el_, key=tie)["config"])
        if inel:
            lab.add(min(inel, key=lambda r: (round(r["gate_shortfall"], 12),) + tie(r))["config"])
    if jm:
        t, _, m, l_ = jm.split("|")
        for f in PRIV_FAMS:
            lab.add(f"{t}|{f}|{m}|{l_}")
        for f in TASK_FAMS:
            lab.add(f"{t}|{f}|{m}")
    if deploy:
        lab.add(deploy)
    return {"rows": rows, "statuses": statuses, "C_match_all": C_match, "deployable": deploy, "scored": sorted(lab)}


def check_selection(own, cache):
    mine = my_selection(own, cache)
    P2STATE["selection"] = mine
    S = jload(RUN / "selection.json")
    pub = jload(RES / "SELECTION.json")
    diffs, row_diff, elig_diff = [], 0.0, []
    for x, s_ in S["statuses"].items():
        m_ = mine["statuses"][x]
        for f in ("status", "config", "descriptive_config"):
            if s_.get(f) != m_.get(f):
                diffs.append(f"{x}.{f}: {s_.get(f)} vs own {m_.get(f)}")
    for key, v in S["C_match_all"].items():
        m_ = mine["C_match_all"][key]
        if (v["status"], v.get("config"), v.get("descriptive_config")) != (m_["status"], m_.get("config"),
                                                                           m_.get("descriptive_config")):
            diffs.append(f"C_match {key}")
    for cid, r in S["rows"].items():
        m_ = mine["rows"][cid]
        for f in ("mean_pair", "mean_v1", "mean_v2", "mean_logloss", "gate_shortfall"):
            row_diff = max(row_diff, abs(r[f] - m_[f]))
        if r["task_eligible"] != m_["task_eligible"]:
            elig_diff.append(cid)
        if r["token_states"] != m_["token_states"] and not (r["token_states"] in (None, float("inf")) and
                                                            m_["token_states"] == float("inf")):
            diffs.append(f"token states {cid}")
        for k in SEEDS:
            rs, ms = r["seeds"][str(k)], m_["seeds"][k]
            if cid.startswith("SRC|"):
                wins = {w: ("source" if ms["winner"][w] == "source" else ms["winner"][w]) for w in VIEWS}
                if rs.get("composed_winner") != wins:
                    diffs.append(f"composed winner {cid} s{k}")
            for i in (0, 1):
                for g_ in ("acc", "logloss", "brier", "retain", "gain"):
                    row_diff = max(row_diff, abs(rs["gate_margins"][str(i)][g_] - ms["gm"][i][g_]))
    if S["deployable_compact"].get("config") != mine["deployable"]:
        diffs.append("deployable compact")
    if sorted(S["scored_labels"]) != mine["scored"] or sorted(pub["scored_labels"]) != mine["scored"]:
        diffs.append("scored labels")
    pub_ok = all(pub["statuses"][x]["status"] == S["statuses"][x]["status"] and
                 pub["statuses"][x]["config"] == S["statuses"][x].get("config") for x in S["statuses"])
    eligible = sorted(c for c, r in mine["rows"].items() if r["task_eligible"])
    near = sorted(((r["gate_shortfall"], c) for c, r in mine["rows"].items() if not r["task_eligible"]
                   and r["family"] not in ("SRC", "REF")))[:5]
    per_family_best = {}
    for c, r in mine["rows"].items():
        f = r["family"]
        if f in ("SRC", "REF"):
            continue
        b = per_family_best.get(f)
        if b is None or r["gate_shortfall"] < b[0]:
            per_family_best[f] = (r["gate_shortfall"], c)
    worst_gate = {}
    for c in ("U|JOINT|m8|l0.1", "U|FINE-TASK|m8", "U|LOCAL|m8|l0.1", "U|CLASS|m1", "SRC|RAW-J_b0.3", "REF|F", "REF|F0", "REF|E"):
        r = mine["rows"][c]
        worst_gate[c] = {f"s{k}": {f"{'income' if i == 0 else 'occ'}_{g_}": round(v, 6) for i in (0, 1)
                                   for g_, v in r["seeds"][k]["gm"][i].items() if v < 0} for k in SEEDS}
    ok = not diffs and not elig_diff and row_diff <= 1e-12 and pub_ok
    st = {x: {kk: v for kk, v in s_.items() if kk in ("status", "config", "descriptive_config", "C_match_key",
                                                     "nomination_shortfall", "joint_task_eligible", "missing_guards")}
          for x, s_ in mine["statuses"].items()}
    return res("PASS" if ok else "FAIL", own_statuses=st, equals_selection_json=not diffs, differences=diffs[:20],
               eligibility_differences=elig_diff, max_abs_diff_rows_and_gate_margins=row_diff,
               public_SELECTION_json_consistent=pub_ok, task_eligible_candidates=eligible,
               nearest_ineligible_finite_codes=[{"config": c, "gate_shortfall_summed_over_seeds": v} for v, c in near],
               lowest_shortfall_per_family={f: {"config": c, "gate_shortfall": v} for f, (v, c) in sorted(per_family_best.items())},
               failing_gates_of_key_candidates=worst_gate, own_C_match_all={k_: {kk: v.get(kk) for kk in (
                   "status", "config", "descriptive_config")} for k_, v in mine["C_match_all"].items()},
               own_deployable_compact=mine["deployable"], own_scored_labels=len(mine["scored"]),
               rule="gates per seed vs SRC|U (acc >= U-0.01, ll <= U+0.01, Brier <= U+0.005, retain 80% of U's gain, "
                    "gain >= 0.03), class preservation; composed source AUC = per view max(own interface, every same-"
                    "teacher/seed policy code AUC); tie: mean pair AUC, mean task log loss, token states, id; J* guards "
                    "C_match & C_global +0.005 per seed/recipient; P* guard T* +0.005; min-shortfall descriptive "
                    "fallback"), mine


# ------------------------------------------------------------------------------------------------ evaluation lock
def check_eval_lock(D: Data, L, gate, mine_sel):
    EL = jload(RES / "EVALUATION_LOCK.json")
    S = jload(RUN / "selection.json")
    out, bad = {}, []
    out["gate"] = gate
    st = {x: {kk: v.get(kk) for kk in ("status", "config", "descriptive_config")} for x, v in S["statuses"].items()}
    out["statuses_equal_selection"] = all(EL["statuses"][x][kk] == st[x][kk] for x in st for kk in st[x])
    out["resolved_equal_own_selection"] = all(EL["resolved"][x] == (mine_sel["statuses"][x].get("config") or
                                                                    mine_sel["statuses"][x].get("descriptive_config"))
                                              for x in EL["resolved"])
    out["scored_labels_equal_own"] = sorted(EL["scored_labels"]) == mine_sel["scored"]
    out["selection_sha256_ok"] = EL["selection_sha256"] == sha_file(RUN / "selection.json") and \
        EL["selection_public_sha256"] == sha_file(RES / "SELECTION.json")
    out["locks_sha256_ok"] = all(sha_file(RES / n) == h for n, h in EL["locks_sha256"].items()) and \
        sorted(EL["locks_sha256"]) == ["ENGINEERING_LOCK.json", "SELECTION_AND_AUDIT_LOCK.json", "TRAINING_LOCK.json"]
    out["amendments_sha256_ok"] = all(sha_file(RES / n) == h for n, h in EL["amendments_sha256"].items()) and \
        sorted(EL["amendments_sha256"]) == sorted(p.name for p in RES.glob("AMENDMENT*.json"))
    merged = dict(jload(RES / "SELECTION_AND_AUDIT_LOCK.json")["code_files"])
    for a in sorted(RES.glob("AMENDMENT*.json")):
        merged.update(jload(a)["code_files"])
    lc = EL["locked_code_files"]
    out["locked_code_equals_lock_plus_amendments"] = lc == merged
    out["locked_code_equals_worktree"] = all((WT / f).exists() and sha_file(WT / f) == h for f, h in lc.items())
    fit = D.fit_idx
    ph = hashlib.sha256(np.bincount(L["sex"][fit], minlength=2).astype(np.int64).tobytes()).hexdigest()
    out["sex_prior_hash_ok"] = ph == EL["sex_prior_defense_fit_sha256"]
    am = D.mask[ASSESS]
    ar = EL["assessment_role"]
    out["assessment_role_ok"] = (ar["rows"] == int(am.sum()) and ar["groups"] == len(np.unique(D.unit[am])) and
                                 ar["row_id_sha256"] == rowid_hash(D.row_id[am]))
    files_ok, comp_ok, spec_ok, n_units = True, True, True, 0
    for k in SEEDS:
        se = EL["seeds"][str(k)]
        if sorted(se["score"]) != sorted(EL["scored_labels"]) or se["u_label"] != "SRC|U":
            spec_ok = False
        for lab, sp in se["score"].items():
            u = release_unit(k, lab)
            if sp["unit"] != u or sp["cid"] != lab:
                spec_ok = False
            cj = jload(UNITS / u / "COMPLETE.json")["files"]
            n_units += 1
            if se["unit_file_sha256"].get(u) != cj or not all(sha_file(UNITS / u / f_) == h for f_, h in cj.items()):
                files_ok = False
        for t in TEACHERS:
            want = sorted(pol_unit(k, c) for c in EL["scored_labels"] if not c.startswith(("SRC|", "REF|"))
                          and cand_teacher(c) == t)
            if sorted(se["composed_policies"][t]) != want:
                comp_ok = False
    out.update({"score_specs_ok": spec_ok, "unit_file_hashes_ok": files_ok, "units_hashed": n_units,
                "composed_policies_rule_ok": comp_ok,
                "endpoints_ok": EL["endpoints"]["z"] == Z_PRIMARY and EL["endpoints"]["B"] == B_BOOT and
                EL["endpoints"]["boot_seed"] == BOOT_SEED and EL["endpoints"]["size"] == 33 and
                EL["endpoints"]["primary"] == [f"P{i:02d}" for i in range(1, 34)],
                "U_valid_true": EL["U_valid"] is True, "bank_full": EL["bank"] == "full",
                "refits_0_1_2": EL["attackers"]["refits"] == [0, 1, 2]})
    for kk, v in out.items():
        if isinstance(v, bool) and not v:
            bad.append(kk)
    if not gate.get("ok"):
        bad.append("gate")
    return res("FAIL" if bad else "PASS", failures=bad, **out)


# ------------------------------------------------------------------------------------------------ outer units
def check_outer(D: Data, LU, T, refs, cache, gate):
    EL = jload(RES / "EVALUATION_LOCK.json")
    a_idx = np.flatnonzero(D.mask[ASSESS])
    f_idx = np.flatnonzero(D.mask["AUDIT_FIT"])
    const = const_classes(LU, D)
    push = parse_iso(iso(gate["first_push_time"])) if gate.get("first_push_time") else None
    out, fails, preds = {}, [], {}
    max_ll, max_auc_rec = 0.0, 0.0
    cov_bad = []
    for k in SEEDS:
        U_rel = release_of(k, "SRC|U", T, refs, cache, rows=a_idx)
        for lab in EL["scored_labels"]:
            name = f"outer__s{k}__{safe_label(lab)}"
            ud = UNITS / name
            f_ = []
            if not unit_done(name):
                fails.append(f"{name}: missing")
                continue
            cj = jload(ud / "COMPLETE.json")
            if not all(sha_file(ud / x) == h for x, h in cj["files"].items()):
                f_.append("hash")
            t_c = utc((ud / "COMPLETE.json").stat().st_mtime)
            if push is None or t_c < push:
                f_.append("completed before the EVALUATION_LOCK push")
            rec = jload(ud / "record.json")
            if rec["evaluation_lock"]["commit"] != gate.get("commit") or rec["evaluation_lock"]["sha256"] != gate.get("sha256"):
                f_.append("lock commit/sha")
            if rec["spec"] != EL["seeds"][str(k)]["score"][lab]:
                f_.append("spec")
            z = np.load(ud / "preds.npz", allow_pickle=False)
            p = {x: z[x] for x in z.files}
            chk = {"rows": bool(np.array_equal(p["assess_row_id"], D.row_id[a_idx])),
                   "groups": bool(np.array_equal(p["assess_unit"], D.unit[a_idx])),
                   "labels": all(np.array_equal(p[x], LU[y][a_idx]) for x, y in (("sex", "sex"), ("race", "race"),
                                                                              ("y_income", "y_income"), ("y_occ", "y_occ"))),
                   "const": list(p["const_class"]) == [const[0], const[1]] and bool((p["const1"] == const[0]).all()) and
                   bool((p["const2"] == const[1]).all())}
            o = release_of(k, lab, T, refs, cache, rows=a_idx)
            for i, K in ((1, 2), (2, 6)):
                y = LU["y_income" if i == 1 else "y_occ"][a_idx]
                chk[f"release_{i}_bitwise"] = bitwise(np.asarray(o[f"p{i}"], np.float64), p[f"prob{i}"]) and \
                    bool(np.array_equal(o[f"hard{i}"], p[f"hard{i}"]))
                chk[f"U_{i}_bitwise"] = bitwise(np.asarray(U_rel[f"p{i}"], np.float64), p[f"u_prob{i}"]) and \
                    bool(np.array_equal(U_rel[f"hard{i}"], p[f"u_hard{i}"]))
                m_ = my_utility(p[f"prob{i}"], p[f"hard{i}"], y, K, const[i - 1])
                max_ll = max(max_ll, maxdiff(m_["ll_rows"], p[f"ll{i}"]), maxdiff(m_["br_rows"], p[f"br{i}"]))
            fams = rec["families"]
            P_ok = True
            for fam, fr in fams.items():
                for w in VIEWS:
                    for crit in ("auc", "ce"):
                        key = f"P_{crit}_{w}" if fam == rec["primary_family"] else f"P_{crit}_{fam}_{w}"
                        if key not in p:
                            P_ok = False
                            continue
                        P3 = p[key]
                        if P3.shape != (3, len(a_idx), 2) or not np.isfinite(P3).all() or \
                                np.abs(P3.sum(2) - 1).max() > 1e-12:
                            P_ok = False
                        mine_auc = [my_auc(p["sex"], P3[s_][:, 1]) for s_ in range(3)]
                        max_auc_rec = max(max_auc_rec, max(abs(x - y) for x, y in zip(mine_auc, fr["scored"][w][crit]["auc_per_seed"])))
            chk["attacker_prediction_arrays_ok"] = P_ok
            # unseen-code coverage on the scored rows (policy codes)
            if not lab.startswith(("SRC|", "REF|")):
                full = cache["full"][(k, lab)]
                cv = fams["code"]["coverage"]["own"]
                t1, t2 = full["tok1"], full["tok2"]
                seen1 = np.isin(t1[a_idx], np.unique(t1[f_idx]))
                seen2 = np.isin(t2[a_idx], np.unique(t2[f_idx]))
                fit_pairs = set(zip(t1[f_idx].tolist(), t2[f_idx].tolist()))
                seenp = np.array([x in fit_pairs for x in zip(t1[a_idx].tolist(), t2[a_idx].tolist())])
                own_cov = {"v1": int((~seen1).sum()), "v2": int((~seen2).sum()), "pair": int((~seenp).sum())}
                rec_cov = {w: cv[w]["scored_rows_fallback"] for w in VIEWS}
                chk["coverage_equal_own"] = own_cov == rec_cov and cv["pair"]["pairs_in_fit"] == len(fit_pairs)
                if not chk["coverage_equal_own"]:
                    cov_bad.append(name)
                out.setdefault("coverage", {})[f"{lab}#s{k}"] = {"scored_rows_fallback": own_cov,
                                                                 "pairs_in_fit": len(fit_pairs),
                                                                 "pair_fallback_rules": cv["pair"]["fallback_rule"]}
            if lab.startswith("SRC|"):
                t = lab.split("|")[1]
                want = sorted(EL["seeds"][str(k)]["composed_policies"][t])
                chk["composed_units_equal_lock"] = sorted(rec["composed"]["units"]) == want and \
                    sorted(rec["composed"]["class_only_units"]) == sorted(u for u in want if u.endswith("_CLASS_m1"))
                out.setdefault("composed_winners", {})[f"{lab}#s{k}"] = {
                    fam: {w: fr["scored"][w]["auc"]["label"].startswith("composed[") for w in VIEWS}
                    for fam, fr in fams.items()}
            f_ += [kk for kk, v in chk.items() if isinstance(v, bool) and not v]
            if f_:
                fails.append(f"{name}: {f_}")
            preds[(k, lab)] = p
    ok = not fails and max_ll <= 1e-15 and max_auc_rec <= 1e-12 and len(preds) == 3 * len(EL["scored_labels"])
    return res("PASS" if ok else "FAIL", units=len(preds), failures=fails[:30],
               max_abs_diff_row_losses=max_ll, max_abs_diff_recorded_scored_auc=max_auc_rec,
               coverage_mismatches=cov_bad, **out), preds


# ------------------------------------------------------------------------------------------------ endpoints
class MyBoot:
    """Exact-record-group multinomial bootstrap, the pinned convention (rng = default_rng(seed); per replicate
    rng.multinomial(n_groups, uniform); groups = np.unique(assess_unit)), generated once and sliced in chunks."""

    def __init__(self, groups, B=B_BOOT, seed=BOOT_SEED, chunk=CHUNK):
        self.u, self.idx = np.unique(np.asarray(groups), return_inverse=True)
        G = len(self.u)
        rng = np.random.default_rng(seed)
        p = np.full(G, 1.0 / G)
        self.counts = np.empty((G, B), dtype=np.int32)
        for b in range(B):
            self.counts[:, b] = rng.multinomial(G, p)
        self.B, self.chunk, self.n = B, chunk, len(self.idx)

    def chunks(self):
        for s in range(0, self.B, self.chunk):
            yield self.counts[self.idx, s:s + self.chunk].astype(np.float64)


class AUCStat:
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
        num = (gp * (below + 0.5 * gn)).sum(0)
        den = gp.sum(0) * gn.sum(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(den > 0, num / den, np.nan)


class MeanStat:
    def __init__(self, v):
        self.v = np.asarray(v, dtype=np.float64)

    def __call__(self, W):
        return (self.v @ W) / W.sum(0)


class Engine:
    """Base statistics evaluated at unit weights (point) and on every bootstrap replicate; cached by content."""

    def __init__(self, boot):
        self.boot, self.stats, self.cache = boot, {}, {}

    def base(self, key, fn):
        if key not in self.stats:
            self.stats[key] = fn
        return key

    def run(self, keys):
        todo = [k_ for k_ in keys if k_ not in self.cache]
        pts = {k_: float(self.stats[k_](np.ones((self.boot.n, 1)))[0]) for k_ in todo}
        reps = {k_: [] for k_ in todo}
        for W in self.boot.chunks():
            for k_ in todo:
                reps[k_].append(self.stats[k_](W))
        for k_ in todo:
            self.cache[k_] = (pts[k_], np.concatenate(reps[k_]))
        return {k_: self.cache[k_] for k_ in keys}


def endpoints(preds, EL, all_levels=True):
    labels = EL["scored_labels"]
    p0 = preds[(0, labels[0])]
    sex = p0["sex"]
    y = {0: p0["y_income"], 1: p0["y_occ"]}
    const = {j: int(p0["const_class"][j]) for j in (0, 1)}
    boot = MyBoot(p0["assess_unit"])
    eng = Engine(boot)
    score_key = {}

    def R(k, lab, fam, view):
        key = f"P_auc_{view}" if fam is None else f"P_auc_{fam}_{view}"
        ids = []
        for s_ in range(3):
            sc = preds[(k, lab)][key][s_][:, 1]
            h = hashlib.sha256(sc.tobytes()).hexdigest()
            score_key.setdefault(h, eng.base(f"auc:{h}", AUCStat(sc, sex == 1)))
            ids.append(score_key[h])
        return ("mean", ids)

    def ACC(k, lab, j):
        p = preds[(k, lab)]
        return ("one", eng.base(f"acc:{k}:{lab}:{j}", MeanStat(p[f"hard{j + 1}"] == y[j])))

    def LOSS(k, lab, j, kind):
        p = preds[(k, lab)]
        P = np.asarray(p[f"prob{j + 1}"], dtype=np.float64)
        ll = -np.log(np.clip(P[np.arange(len(y[j])), y[j]], 1e-12, 1.0))
        br = np.sum((P - np.eye(P.shape[1])[y[j]]) ** 2, 1)
        return ("one", eng.base(f"{kind}:{k}:{lab}:{j}", MeanStat(ll if kind == "ll" else br)))

    def CONST(j):
        return ("one", eng.base(f"const:{j}", MeanStat(y[j] == const[j])))

    # resolve roles
    res_ = EL["resolved"]
    stat = EL["statuses"]
    U = "SRC|U"
    slots = []
    claims = {"A": ("J*", "C_match"), "B": ("J*", "C_global"), "C": ("P*", "T*")}
    for ci, (claim, (nom, ref)) in enumerate(claims.items()):
        base = 11 * ci
        slots += [{"id": f"P{base + 1:02d}", "claim": claim, "kind": "coalition", "nom": nom, "ref": ref, "target": 0.02,
                   "side": "lower>"},
                  {"id": f"P{base + 2:02d}", "claim": claim, "kind": "local", "view": "v1", "nom": nom, "ref": ref,
                   "target": 0.01, "side": "upper<"},
                  {"id": f"P{base + 3:02d}", "claim": claim, "kind": "local", "view": "v2", "nom": nom, "ref": ref,
                   "target": 0.01, "side": "upper<"}]
        for off, kind, target, side in ((4, "acc", -0.01, "lower>"), (6, "logloss", 0.01, "upper<"),
                                        (8, "brier", 0.005, "upper<"), (10, "retain", 0.0, "lower>")):
            for j in (0, 1):
                slots.append({"id": f"P{base + off + j:02d}", "claim": claim, "kind": kind, "task": j, "nom": nom,
                              "target": target, "side": side})
    # build per-seed nodes
    plan = {}
    for e in slots:
        nl, rl = res_.get(e["nom"]), res_.get(e.get("ref")) if e.get("ref") else None
        per = []
        for k in SEEDS:
            j = e.get("task")
            if e["kind"] == "coalition":
                per.append(("diff", R(k, rl, None, "pair"), R(k, nl, None, "pair")))
            elif e["kind"] == "local":
                per.append(("diff", R(k, nl, None, e["view"]), R(k, rl, None, e["view"])))
            elif e["kind"] == "acc":
                per.append(("diff", ACC(k, nl, j), ACC(k, U, j)))
            elif e["kind"] == "logloss":
                per.append(("diff", LOSS(k, nl, j, "ll"), LOSS(k, U, j, "ll")))
            elif e["kind"] == "brier":
                per.append(("diff", LOSS(k, nl, j, "br"), LOSS(k, U, j, "br")))
            else:
                per.append(("lin", ACC(k, nl, j), ACC(k, U, j), CONST(j)))
        plan[e["id"]] = per
    # levels
    levels = {}
    if all_levels:
        fams_of = {}
        for (k, lab), p in preds.items():
            fams_of.setdefault(lab, set())
            for key in p:
                if key.startswith("P_auc_") and key[len("P_auc_"):] not in VIEWS:
                    fams_of[lab].add(key[len("P_auc_"):].rsplit("_", 1)[0])
        for lab in labels:
            for k in SEEDS:
                for fam in [None] + sorted(fams_of.get(lab, ())):
                    for v in VIEWS:
                        key = f"P_auc_{v}" if fam is None else f"P_auc_{fam}_{v}"
                        if key in preds[(k, lab)]:
                            levels[f"R#{k}#{lab}#{fam or 'primary'}#{v}"] = ("node", R(k, lab, fam, v))
                for j in (0, 1):
                    levels[f"acc#{k}#{lab}#{j}"] = ("node", ACC(k, lab, j))
                    levels[f"ll#{k}#{lab}#{j}"] = ("node", LOSS(k, lab, j, "ll"))
                    levels[f"br#{k}#{lab}#{j}"] = ("node", LOSS(k, lab, j, "br"))
            for fam in [None] + sorted(fams_of.get(lab, ())):
                for v in VIEWS:
                    if all(f"R#{k}#{lab}#{fam or 'primary'}#{v}" in levels for k in SEEDS):
                        levels[f"Rmean#{lab}#{fam or 'primary'}#{v}"] = ("meanof", [f"R#{k}#{lab}#{fam or 'primary'}#{v}"
                                                                                   for k in SEEDS])
            for j in (0, 1):
                for q in ("acc", "ll", "br"):
                    levels[f"{q}mean#{lab}#{j}"] = ("meanof", [f"{q}#{k}#{lab}#{j}" for k in SEEDS])
        for j in (0, 1):
            levels[f"const#{j}"] = ("node", CONST(j))
    keys = sorted(eng.stats)
    t0 = time.process_time()
    vals = eng.run(keys)
    cpu = time.process_time() - t0

    def nodeval(node):
        kind, ids = node
        if kind == "one":
            return vals[ids]
        return float(np.mean([vals[i][0] for i in ids])), np.mean(np.stack([vals[i][1] for i in ids]), 0)

    def combine(op, *nodes):
        v = [nodeval(n_) for n_ in nodes]
        if op == "diff":
            return v[0][0] - v[1][0], v[0][1] - v[1][1]
        return v[0][0] - 0.8 * v[1][0] - 0.2 * v[2][0], v[0][1] - 0.8 * v[1][1] - 0.2 * v[2][1]

    mine = {}
    for e in slots:
        per = [combine(x[0], *x[1:]) for x in plan[e["id"]]]
        pt = float(np.mean([a for a, _ in per]))
        rep = np.mean(np.stack([b for _, b in per]), 0)
        nonfin = int((~np.isfinite(rep)).sum())
        se = float(np.std(rep, ddof=1))
        lo, hi = pt - Z_PRIMARY * se, pt + Z_PRIMARY * se
        dec = ("PASS" if lo > e["target"] else "NOT_ESTABLISHED") if e["side"] == "lower>" else \
            ("PASS" if hi < e["target"] else "NOT_ESTABLISHED")
        if nonfin:
            dec = "INVALID"
        roles = [e["nom"]] + ([e["ref"]] if e.get("ref") else [])
        final = dec if all(stat[x]["status"] == "NOMINEE" for x in roles) or dec == "INVALID" else "DESCRIPTIVE_ONLY"
        mine[e["id"]] = {"point": pt, "se": se, "lower": lo, "upper": hi, "decision_numeric": dec, "decision": final,
                         "nonfinite": nonfin, "claim": e["claim"]}
    lev = {}
    for nm, spec in levels.items():
        if spec[0] == "node":
            pt, rep = nodeval(spec[1])
        else:
            parts = [nodeval(levels[x][1]) for x in spec[1]]
            pt, rep = float(np.mean([a for a, _ in parts])), np.mean(np.stack([b for _, b in parts]), 0)
        ok = np.isfinite(rep)
        lev[nm] = {"point": pt, "se": float(np.std(rep[ok], ddof=1)), "nonfinite": int((~ok).sum())}
    return mine, lev, {"distinct_auc_stats": len(score_key), "base_stats": len(keys), "cpu_s": cpu,
                       "groups": int(len(boot.u)), "rows": int(boot.n), "replicate_count_sum_ok":
                       bool((boot.counts.sum(0) == len(boot.u)).all())}


P2STATE: dict = {}


def check_endpoints(preds, EL):
    mine, lev, info = endpoints(preds, EL)
    P2STATE.update({"primary": mine, "levels": lev})
    inf = jload(RUN / "inference.json")
    diffs, mx = [], {"point": 0.0, "se": 0.0, "lower": 0.0, "upper": 0.0}
    for e in inf["primary"]:
        m_ = mine[e["id"]]
        for f in mx:
            mx[f] = max(mx[f], abs(e[f] - m_[f]))
        if e["decision"] != m_["decision"] or e.get("decision_numeric", e["decision"]) != m_["decision_numeric"]:
            diffs.append(e["id"])
    # claims and label (registered rule)
    stat = EL["statuses"]
    claims = {"A": ("J*", "C_match"), "B": ("J*", "C_global"), "C": ("P*", "T*")}
    dec = {}
    for c, (n_, r_) in claims.items():
        ids = [i for i, m_ in mine.items() if m_["claim"] == c]
        allp = all(mine[i]["decision"] == "PASS" for i in ids)
        req = all(stat[x]["status"] == "NOMINEE" and stat[x].get("config") for x in (n_, r_))
        valid = bool(EL["U_valid"]) and not any(mine[i]["decision"] == "INVALID" for i in ids) and \
            not any(str(stat[x]["status"]).startswith("INVALID") for x in (n_, r_))
        dec[c] = {"decision": "PASS" if allp and req else "NOT_ESTABLISHED", "valid": valid,
                  "numeric_clauses_passing": sum(mine[i]["decision_numeric"] == "PASS" for i in ids),
                  "status_requirements_met": req}
    labels_ = []
    if all(dec[c]["decision"] == "PASS" and dec[c]["valid"] for c in "AB"):
        labels_.append("JOINT_DEVELOPMENT_CRITERION_MET")
    if dec["C"]["decision"] == "PASS" and dec["C"]["valid"]:
        labels_.append("PRIVACY_COMPRESSION_DEVELOPMENT_CRITERION_MET")
    label = " + ".join(labels_) if labels_ else ("INCOMPLETE_OR_INVALID" if not all(d["valid"] for d in dec.values())
                                                else "EXPERIMENTAL_NO_ADVANTAGE")
    claims_ok = all(inf[f"claim{c}"]["decision"] == dec[c]["decision"] and inf[f"claim{c}"]["valid"] == dec[c]["valid"]
                    for c in claims) and inf["label"] == label and inf["complete"] == all(d["valid"] for d in dec.values())
    # every level
    lv_mx, lv_missing = {"point": 0.0, "se": 0.0}, []
    for nm, v in inf["levels"].items():
        m_ = lev.get(nm)
        if m_ is None:
            lv_missing.append(nm)
            continue
        lv_mx["point"] = max(lv_mx["point"], abs(v["point"] - m_["point"]))
        lv_mx["se"] = max(lv_mx["se"], abs(v["se"] - m_["se"]))
    extra = sorted(set(lev) - set(inf["levels"]))
    # public CSV
    csv_mx, csv_bad = 0.0, []
    import csv as _csv
    with open(RES / "PRIMARY_ENDPOINTS.csv") as fh:
        for row in _csv.DictReader(fh):
            m_ = mine[row["id"]]
            for f in ("point", "se", "lower", "upper"):
                csv_mx = max(csv_mx, abs(float(row[f]) - m_[f]))
            if row["decision"] != m_["decision"]:
                csv_bad.append(row["id"])
    lv_csv_mx, lv_csv_n = 0.0, 0
    with open(RES / "ALL_LEVELS.csv") as fh:
        for row in _csv.DictReader(fh):
            q = row["quantity"]
            if q in ("R", "acc", "ll", "br"):
                nm = "#".join([q, row["seed"], row["label"]] + row["detail"].split("|"))
            elif q == "const":
                nm = f"const#{row['detail']}"
            else:
                nm = "#".join([q, row["label"]] + row["detail"].split("|"))
            if nm.startswith("Rmean#") and nm.count("#") == 3:
                pass
            m_ = lev.get(nm)
            if m_ is None:
                continue
            lv_csv_n += 1
            lv_csv_mx = max(lv_csv_mx, abs(float(row["point"]) - m_["point"]), abs(float(row["se"]) - m_["se"]))
    ok = (not diffs and max(mx.values()) <= 1e-10 and claims_ok and not lv_missing and lv_mx["point"] <= 1e-10 and
          lv_mx["se"] <= 1e-10 and csv_mx <= 6e-7 and not csv_bad and lv_csv_n == len(inf["levels"]) and lv_csv_mx <= 6e-7)
    return res("PASS" if ok else "FAIL", slots=len(mine), decision_differences=diffs,
               max_abs_diff_vs_inference_json=mx, own_claims=dec, own_label=label, claims_and_label_equal=claims_ok,
               levels_compared=len(inf["levels"]), levels_missing_in_own=lv_missing[:10], levels_extra_in_own=extra[:10],
               max_abs_diff_levels=lv_mx, primary_csv_max_abs_diff=csv_mx, primary_csv_decision_differences=csv_bad,
               all_levels_csv_rows_compared=lv_csv_n, all_levels_csv_max_abs_diff=lv_csv_mx, engine=info,
               own_primary={i: {kk: m_[kk] for kk in ("point", "se", "lower", "upper", "decision_numeric", "decision")}
                            for i, m_ in mine.items()},
               rule="per-seed paired statistic averaged over seeds; recovery = mean over attacker seeds 0-2 of the AUC "
                    "of P(S=1); B = 1999 multinomial exact-record-group replicates (default_rng(20261007), uniform, "
                    "groups = np.unique(assess_unit)); SE ddof 1; bound = point +- 3.1717657833516224 SE; "
                    "DESCRIPTIVE_ONLY unless every role in the slot is a NOMINEE; claims all-11-PASS + valid NOMINEEs")


# ------------------------------------------------------------------------------------------------ attacker refits
class MyCanon:
    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        self.mu = X.mean(0)
        _, s, Vt = np.linalg.svd(X - self.mu, full_matrices=False)
        scale = max(float(s[0]) if s.size else 0.0, float(np.linalg.norm(X, 2)))
        keep = s > 1.0 * max(n, d) * float(np.finfo(np.float64).eps) * scale
        self.rank = int(keep.sum())
        self.W = (Vt[keep].T / s[keep]) * math.sqrt(max(n - 1, 1))
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


def my_factory(name, seed):
    if name.startswith("LR_C"):
        return make_pipeline(StandardScaler(), LogisticRegression(C=float(name[4:]), max_iter=3000))
    if name.startswith("MLP_"):
        h = tuple(int(x) for x in name[4:].split("x"))
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=h, alpha=1e-4, max_iter=300,
                                                             early_stopping=True, validation_fraction=0.1,
                                                             n_iter_no_change=15, random_state=seed))
    if name.startswith("HGB_"):
        _, lr, lv = name.split("_")
        return HistGradientBoostingClassifier(learning_rate=float(lr), max_leaf_nodes=int(lv), max_iter=200,
                                              early_stopping=False, random_state=seed)
    if name in ("DA_LR", "DA_MLP"):
        return MyDA("lr" if name == "DA_LR" else "mlp", seed)
    raise KeyError(name)


def p_one(m, X):
    P = m.predict_proba(X)
    cl = [int(c) for c in m.classes_]
    return np.asarray(P[:, cl.index(1)], dtype=np.float64) if 1 in cl else np.zeros(len(X))


def canonical_cols(tok, D, alphabet):
    f = np.flatnonzero(D.mask["AUDIT_FIT"])
    s = np.flatnonzero(D.mask["INNER_SELECTION"])
    order, seen = [], set()
    for ix in (f, s, np.arange(len(tok))):
        for v in tok[ix].tolist():
            if v not in seen:
                seen.add(v)
                order.append(v)
    order += [v for v in range(alphabet) if v not in seen]
    col = np.empty(alphabet, dtype=np.int64)
    col[np.asarray(order, dtype=np.int64)] = np.arange(alphabet)
    return col


def code_views(full, D):
    X, Tk = {}, {}
    for i, K in ((1, 2), (2, 6)):
        tok, a = np.asarray(full[f"tok{i}"], dtype=np.int64), int(full[f"alpha{i}"])
        col = canonical_cols(tok, D, a)
        X[f"v{i}"] = np.hstack([np.eye(a)[col[tok]], np.asarray(full[f"p{i}"], dtype=np.float64), np.eye(K)[full[f"hard{i}"]]])
        Tk[f"v{i}"] = tok
    X["pair"] = np.hstack([X["v1"], X["v2"]])
    return X, Tk


def source_view_set(t, fam):
    need = {"interface": ("c", "p"), "complete": ("r", "c"), "scores": ("c",), "probs": ("p",), "decisions": ("d",)}[fam]
    X, Tk = {}, None
    for i, K in ((1, 2), (2, 6)):
        if fam == "decisions":
            Tk = Tk or {}
            X[f"v{i}"], Tk[f"v{i}"] = np.eye(K)[t[f"d{i}"]], np.asarray(t[f"d{i}"], dtype=np.int64)
        else:
            X[f"v{i}"] = np.hstack([np.asarray(t[f"{k_}{i}"], dtype=np.float64).reshape(len(t["p1"]), -1) for k_ in need])
    X["pair"] = np.hstack([X["v1"], X["v2"]])
    return X, Tk


def cc_local_pred(tok, y, fit_idx, pred_idx, a):
    pi1 = float(np.mean(y[fit_idx] == 1))
    n, n1 = {}, {}
    for t, s in zip(tok[fit_idx].tolist(), y[fit_idx].tolist()):
        n[t] = n.get(t, 0) + 1
        n1[t] = n1.get(t, 0) + int(s == 1)
    p = np.array([(n1[t] + a * pi1) / (n[t] + a) if t in n else pi1 for t in tok[pred_idx].tolist()])
    seen = np.array([t in n for t in tok[pred_idx].tolist()])
    return p, seen


def cc_pair_pred(t1, t2, y, fit_idx, pred_idx, sel_pos, a):
    keys = list(zip(t1.tolist(), t2.tolist()))
    pi1 = float(np.mean(y[fit_idx] == 1))
    n, n1 = {}, {}
    for j in fit_idx.tolist():
        n[keys[j]] = n.get(keys[j], 0) + 1
        n1[keys[j]] = n1.get(keys[j], 0) + int(y[j] == 1)
    pk = [keys[j] for j in pred_idx.tolist()]
    seen = np.array([k_ in n for k_ in pk])
    p = np.array([(n1[k_] + a * pi1) / (n[k_] + a) if k_ in n else np.nan for k_ in pk])
    l1, _ = cc_local_pred(t1, y, fit_idx, pred_idx, a)
    l2, _ = cc_local_pred(t2, y, fit_idx, pred_idx, a)
    cand = {"local_1": l1, "local_2": l2, "prior": np.full(len(pred_idx), pi1)}
    un = ~seen[sel_pos]
    ys = y[pred_idx][sel_pos]
    if un.sum() == 0:
        rule = "prior"
    else:
        ces = {r: my_ce(ys[un], cand[r][sel_pos][un]) for r in ("local_1", "local_2", "prior")}
        rule = min(("local_1", "local_2", "prior"), key=lambda r: (ces[r], ("local_1", "local_2", "prior").index(r)))
    p[~seen] = cand[rule][~seen]
    return p, rule


def refit_bank(X, Tk, y, fit_idx, pred_idx, sel_pos, views=VIEWS, members=None, seed=0):
    """{(view, attacker): P(S=1) on pred_idx} for the slate (+ CC readers when Tk is given)."""
    out, rules = {}, {}
    for w in views:
        for name in (members or SLATE):
            m = my_factory(name, seed).fit(X[w][fit_idx], y[fit_idx])
            out[(w, name)] = p_one(m, X[w][pred_idx])
        if Tk is not None and members is None:
            for a in CC_A:
                if w == "pair":
                    out[(w, f"CCpair_alpha{a}")], rules[f"CCpair_alpha{a}"] = cc_pair_pred(Tk["v1"], Tk["v2"], y, fit_idx,
                                                                                         pred_idx, sel_pos, a)
                else:
                    out[(w, f"CC_alpha{a}")], _ = cc_local_pred(Tk[w], y, fit_idx, pred_idx, a)
    return out, rules


def check_attackers(D: Data, LU, T, cache, preds):
    """(1) outer__s1__U_FINE-TASK_m8: the whole primary bank refit on AUDIT_FIT from own views (every candidate's
    INNER AUC / CE vs the record table, own dual selection), then the selected attackers refit at seeds 0-2 and
    compared with preds.npz on the assessment rows; (2) the AUC-selected attackers of outer__s0__U_JOINT_m8_l0.1 (J*
    fallback) and outer__s0__SRC_U (T* = C_global; composed candidates rebuilt from the named policy) refit at
    seeds 0-2."""
    y = LU["sex"]
    fit_idx = np.flatnonzero(D.mask["AUDIT_FIT"])
    sel_idx = np.flatnonzero(D.mask["INNER_SELECTION"])
    a_idx = np.flatnonzero(D.mask[ASSESS])
    pred_idx = np.concatenate([sel_idx, a_idx])
    sel_pos, sc_pos = np.arange(len(sel_idx)), np.arange(len(sel_idx), len(pred_idx))
    ys = y[sel_idx]
    out, fails = {}, []
    t0 = time.process_time()
    # (1) full bank
    k, lab = 1, "U|FINE-TASK|m8"
    rec = jload(UNITS / f"outer__s{k}__{safe_label(lab)}" / "record.json")["families"]["code"]
    X, Tk = code_views(cache["full"][(k, lab)], D)
    bankp, rules = refit_bank(X, Tk, y, fit_idx, pred_idx, sel_pos)
    tab_mx, sel_eq, P_mx = 0.0, {}, 0.0
    for w in VIEWS:
        rows = [("own", w, a) for a in SLATE + CC_LOCAL] if w != "pair" else \
            [("coalition", "pair", a) for a in SLATE + CC_PAIR] + [("ignore_recipient_2", "v1", a) for a in SLATE + CC_LOCAL] + \
            [("ignore_recipient_1", "v2", a) for a in SLATE + CC_LOCAL]
        va = [my_auc(ys, bankp[(v, a)][sel_pos]) for _, v, a in rows]
        vc = [my_ce(ys, bankp[(v, a)][sel_pos]) for _, v, a in rows]
        tab = {(r["candidate"], r["view"], r["attacker"]): r for r in rec["tables"][w]}
        for j, r in enumerate(rows):
            tab_mx = max(tab_mx, abs(tab[r]["inner_auc"] - va[j]), abs(tab[r]["inner_ce"] - vc[j]))
        for crit, vals, mx_ in (("auc", va, True), ("ce", vc, False)):
            j = pick_first(vals, mx_)
            lbl = "%s:%s:%s" % rows[j]
            sel_eq[f"{w}:{crit}"] = lbl == rec["scored"][w][crit]["label"]
            _, v, a = rows[j]
            P3 = preds[(k, lab)][f"P_{crit}_{w}"]
            for s_ in SEEDS:
                if a.startswith("CC"):
                    p1 = bankp[(v, a)][sc_pos]
                elif s_ == 0:
                    p1 = bankp[(v, a)][sc_pos]
                else:
                    m = my_factory(a, s_).fit(X[v][fit_idx], y[fit_idx])
                    p1 = p_one(m, X[v][pred_idx])[sc_pos]
                P_mx = max(P_mx, maxdiff(np.stack([1 - p1, p1], 1), P3[s_]))
    rules_ok = rules == rec["coverage"]["own"]["pair"]["fallback_rule"]
    out["full_bank_refit"] = {"unit": f"outer__s{k}__{safe_label(lab)}", "candidates_refit": len(bankp),
                              "inner_table_max_abs_diff": tab_mx, "selection_equal": sel_eq,
                              "pair_fallback_rules_equal": rules_ok, "assessment_predictions_max_abs_diff": P_mx,
                              "selected": {w: rec["scored"][w]["auc"]["label"] for w in VIEWS}}
    if tab_mx > 1e-9 or not all(sel_eq.values()) or not rules_ok or P_mx > 1e-9:
        fails.append("full bank refit")
    # (2) selected attackers of J* fallback and of the continuous source
    for k, lab in ((0, "U|JOINT|m8|l0.1"), (0, "SRC|U")):
        r = jload(UNITS / f"outer__s{k}__{safe_label(lab)}" / "record.json")
        fam = r["primary_family"]
        fr = r["families"][fam]
        info = {}
        for w in VIEWS:
            s = fr["scored"][w]["auc"]
            label = s["label"]
            if label.startswith("composed["):
                u = label[len("composed["):label.index("]")]
                cid = _cid_of_unit(u)
                Xv, Tv = code_views(cache["full"][(k, cid)], D)
                rest = label[label.index("]:") + 2:]
            else:
                if lab.startswith("SRC|"):
                    Xv, Tv = source_view_set(T[(k, "U")], fam)
                else:
                    Xv, Tv = code_views(cache["full"][(k, lab)], D)
                rest = label
            cand, view, att = rest.split(":")
            P3 = preds[(k, lab)][f"P_auc_{w}"]
            mx_ = 0.0
            inner_auc = None
            for s_ in SEEDS:
                if att.startswith("CC"):
                    a = float(att.split("alpha")[1])
                    if view == "pair":
                        p_all, _ = cc_pair_pred(Tv["v1"], Tv["v2"], y, fit_idx, pred_idx, sel_pos, a)
                    else:
                        p_all, _ = cc_local_pred(Tv[view], y, fit_idx, pred_idx, a)
                else:
                    p_all = p_one(my_factory(att, s_).fit(Xv[view][fit_idx], y[fit_idx]), Xv[view][pred_idx])
                if s_ == 0:
                    inner_auc = my_auc(ys, p_all[sel_pos])
                mx_ = max(mx_, maxdiff(np.stack([1 - p_all[sc_pos], p_all[sc_pos]], 1), P3[s_]))
            info[w] = {"label": label, "assessment_predictions_max_abs_diff": mx_,
                       "inner_auc_abs_diff": abs(inner_auc - s["inner_auc"])}
            if mx_ > 1e-9 or info[w]["inner_auc_abs_diff"] > 1e-9:
                fails.append(f"{lab}#s{k} {w}")
        out[f"{lab}#s{k}"] = info
    out["cpu_s"] = time.process_time() - t0
    return res("FAIL" if fails else "PASS", failures=fails, **out,
               note="own views (occurrence-ordered one-hot tokens, decoded q, one-hot decisions / source [c, p]) and "
                    "own estimator construction (sklearn LR / MLP / HGB, own whitening DA, own cell readers); fitted on "
                    "AUDIT_FIT only, selected on INNER_SELECTION, scored on the assessment rows")


def _cid_of_unit(u):
    body = u.split("__", 2)[2]
    for t in ("RAW-J_b0.3", "U"):
        if body.startswith(t + "_"):
            return "|".join([t] + body[len(t) + 1:].split("_"))
    raise ValueError(u)


# ------------------------------------------------------------------------------------------------ controls
def _u_split(groups, seed, salt):
    return np.array([int(hashlib.sha256(f"{seed}|{salt}|{int(g)}".encode()).hexdigest()[:16], 16) / 2.0 ** 64
                     for g in groups])


def check_controls(D: Data, L, cache, T):
    A = jload(RES / "AUDIT_PRELOCK_CHECKS.json")
    sel = np.flatnonzero(D.mask["INNER_SELECTION"])
    fit = np.flatnonzero(D.mask["AUDIT_FIT"])
    S = L["sex"]                                  # sealed labels (assessment -1), as the controls stage used
    u = _u_split(D.unit[sel], SPLIT_SEED, "null-split")
    ha, hb = np.flatnonzero(u < 0.5), np.flatnonzero(u >= 0.5)
    h = lambda ix: hashlib.sha256(np.sort(D.row_id[ix]).astype(np.int64).tobytes()).hexdigest()   # noqa: E731
    split_ok = A["split"]["rows_A"] == len(ha) and A["split"]["rows_B"] == len(hb) and \
        A["split"]["row_id_sha256_A"] == h(sel[ha]) and A["split"]["row_id_sha256_B"] == h(sel[hb])
    rng = np.random.default_rng(CONTROL_SEED)
    Sp = S.copy()
    for ix in (fit, sel[ha], sel[hb]):
        Sp[ix] = S[ix][rng.permutation(len(ix))]
    used = np.concatenate([fit, sel])
    perm_sha = hashlib.sha256(np.ascontiguousarray(Sp[used]).astype(np.int64).tobytes()).hexdigest()
    yb = Sp[sel[hb]]
    n1, n0 = int((yb == 1).sum()), int((yb == 0).sum())
    sd0 = math.sqrt((n0 + n1 + 1) / (12.0 * n0 * n1))
    thr = 0.5 + NULL_Z * sd0
    plan_ok = A["plan"] == {"policies": [[0, "U|JOINT|m8|l1"], [0, "RAW-J_b0.3|JOINT|m8|l1"], [0, "U|CLASS|m1"]],
                            "sources": [["U", 0], ["RAW-J_b0.3", 0]], "references": [["E", 0], ["F", 0], ["F0", 0]],
                            "null_policy": [0, "U|JOINT|m8|l1"], "null_reps": 5}
    want_rel = [f"{c}|s0" for _, c in A["plan"]["policies"]] + [f"SRC|{t}|s0|{f}" for t in TEACHERS for f in SRC_FAMS] + \
        [f"REF|{r}|s0|{f}" for r in REFS for f in REF_FAMS[r]]
    cover_ok = sorted(A["releases"]) == sorted(want_rel)
    cons, perm_ok, thr_ok, plants_ok = [], True, True, True
    for nm, r in A["releases"].items():
        perm_ok &= r["permutation_sha256"] == perm_sha
        thr_ok &= abs(r["null_threshold"] - thr) <= 1e-15 and abs(r["null_sd_B"] - sd0) <= 1e-15
        for w in VIEWS:
            if (r["checks"]["null"][w]["heldout_auc_B"] <= thr) != r["checks"]["null"][w]["ok"]:
                cons.append(f"{nm}:null:{w}")
        if nm in [f"{c}|s0" for _, c in A["plan"]["policies"]]:
            have = sorted(x for x in r["checks"] if x != "null")
            plants_ok &= have == ["COLL_r1", "COLL_r2", "CONF_r1", "CONF_r2", "XOR"]
            for pn, e in r["checks"].items():
                if pn == "null":
                    continue
                if (e["serialisation_exact"] and e[e["target_view"]]["heldout_auc_B"] > PLANT_MIN) != e["ok"]:
                    cons.append(f"{nm}:{pn}")
    # replay: null (real release vs S*) and CONF_r1 plant on U|CLASS|m1 seed 0, target view v1, own code
    full = cache["full"].get((0, "U|CLASS|m1"))
    replay = {}
    if full is not None:
        rec = A["releases"]["U|CLASS|m1|s0"]["checks"]
        for which in ("null", "CONF_r1"):
            z = dict(full)
            if which == "CONF_r1":
                rng2 = np.random.default_rng(CONTROL_SEED + 1)
                noisy = Sp.copy()
                flip = rng2.random(len(noisy)) < PLANT_NOISE
                noisy[flip] = rng2.integers(0, 2, int(flip.sum()))
                b = np.where((noisy == 0) | (noisy == 1), noisy, 0)
                z["tok1"] = 2 * np.asarray(full["tok1"]) + b
                z["alpha1"] = 2 * int(full["alpha1"])
                q = np.asarray(full["p1"], dtype=np.float64)
                z["p1"] = np.where(b[:, None] == 1, (1 - CONF_ETA) * q + CONF_ETA * np.eye(2)[full["hard1"]], q)
            X, Tk = code_views(z, D)
            bankp, _ = refit_bank(X, Tk, Sp, fit, sel, ha, views=("v1",))
            names = SLATE + CC_LOCAL
            va = [my_auc(Sp[sel][ha], bankp[("v1", a)][ha]) for a in names]
            j = pick_first(va, True)
            got = {"selected_on_A": f"own:v1:{names[j]}", "max_auc_A": va[j],
                   "heldout_auc_B": my_auc(Sp[sel][hb], bankp[("v1", names[j])][hb])}
            r_ = rec["null"]["v1"] if which == "null" else rec["CONF_r1"]["v1"]
            replay[which] = {**got, "selected_equal": got["selected_on_A"] == r_["selected_on_A"],
                             "max_abs_diff": max(abs(got["max_auc_A"] - r_["max_auc_A"]),
                                                 abs(got["heldout_auc_B"] - r_["heldout_auc_B"]))}
    rep_ok = all(v["selected_equal"] and v["max_abs_diff"] <= 1e-9 for v in replay.values()) and len(replay) == 2
    calib = A.get("null_calibration", {}).get("summary", {})
    ok = split_ok and perm_ok and thr_ok and plan_ok and cover_ok and plants_ok and not cons and rep_ok and \
        A["verdict"]["all_ok"] and calib.get("exceedances") == 0 and calib.get("tests") == 15
    return res("PASS" if ok else "FAIL", split_recomputed_equal=split_ok, permutation_sha256_recomputed_equal=perm_ok,
               null_threshold_recomputed_equal=thr_ok, own_threshold=thr, plan_equals_registered=plan_ok,
               releases_cover_plan=cover_ok, n_releases=len(A["releases"]), plants_complete=plants_ok,
               ok_flags_inconsistent=cons, own_replay_U_CLASS_m1_v1=replay, verdict_all_ok=A["verdict"]["all_ok"],
               null_calibration=calib,
               decisions_only_audit_misses_confidence_plant=A["verdict"]["decisions_only_audit_misses_confidence_plant"],
               scope="real releases, inner roles only (AUDIT_FIT fit, INNER_SELECTION half A select, half B evaluate); "
                     "plants = within-class confidence token (CONF), decoder collision (COLL), coalition XOR")


# ------------------------------------------------------------------------------------------------ amendments, chronology
def check_amendments_and_phase2_chronology(gate):
    lf = lock_files()
    entries = remote_reflog()
    ev = activity()
    out, bad = {}, []
    base = jload(RES / "SELECTION_AND_AUDIT_LOCK.json")["code_files"]
    for name in sorted(p.stem for p in RES.glob("AMENDMENT*.json")):
        a = jload(RES / f"{name}.json")
        c, ct = lf[name]["commits"][-1]
        parent = git("rev-parse", f"{c}^")
        changed = (git("diff", "--name-only", parent, c) or "").split()
        code_changed = sorted(f for f in changed if not f.startswith("results/"))
        decl = sorted(a.get("changes_previously_locked") or [])
        push = first_remote(c, entries)
        hashes_ok = all(hashlib.sha256(git_show_bytes(c, f) or b"").hexdigest() == h for f, h in a["code_files"].items())
        out[name] = {"commit": c, "pushed": push, "declared": decl, "code_files_changed_in_commit": code_changed,
                     "declared_equals_changed": decl == code_changed, "hashes_equal_commit": hashes_ok,
                     "previously_locked": all(f in base for f in decl), "parent_commit_ok": a.get("parent_commit") == parent}
        if not (decl == code_changed and hashes_ok and push):
            bad.append(name)
    starts = [(e["event"], parse_iso(e["at"])) for e in ev if e.get("event") in ("start controls", "start select")]
    a1p = out.get("AMENDMENT_A1", {}).get("pushed")
    ctl = [t for n, t in starts if n == "start controls"]
    out["controls_runs"] = {"starts": ctl, "first_run_before_A1": bool(ctl and a1p and ctl[0] < a1p),
                            "rerun_after_A1_push": bool(ctl and a1p and ctl[-1] > a1p),
                            "AUDIT_PRELOCK_CHECKS_first_commit": (git("log", "--format=%cI", "--", f"{REL_RES}/AUDIT_PRELOCK_CHECKS.json") or "").splitlines()[-1:],
                            "quarantine_contents": sorted(p.name for p in (PRIV / "quarantine_amendment_A1").iterdir())
                            if (PRIV / "quarantine_amendment_A1").exists() else None}
    if not out["controls_runs"]["rerun_after_A1_push"]:
        bad.append("controls rerun before A1 push")
    el_written = jload(RES / "EVALUATION_LOCK.json")["written_at"]
    a2p = out.get("AMENDMENT_A2", {}).get("pushed")
    out["evaluation_lock_written_after_A2_push"] = bool(a2p and parse_iso(el_written) >= a2p)
    if not out["evaluation_lock_written_after_A2_push"]:
        bad.append("eval lock before A2")
    push = gate.get("first_push_time")
    opened = [parse_iso(e["at"]) for e in ev if e.get("event") == "assessment opened"]
    out["assessment_opened"] = opened
    out["assessment_opened_after_push"] = bool(opened and push and all(t >= push for t in opened))
    outer_t = sorted(utc((d / "COMPLETE.json").stat().st_mtime) for d in UNITS.iterdir()
                     if d.name.startswith("outer__") and (d / "COMPLETE.json").exists())
    inner_t = sorted(utc((d / "COMPLETE.json").stat().st_mtime) for d in UNITS.iterdir()
                     if d.name.startswith("inner__") and (d / "COMPLETE.json").exists())
    out["outer_units_first_last"] = [outer_t[0], outer_t[-1]] if outer_t else None
    out["inner_units_last"] = inner_t[-1] if inner_t else None
    out["all_inner_before_lock_push"] = bool(inner_t and push and inner_t[-1] < push)
    out["inference_written_after_last_outer"] = bool(outer_t and utc((RUN / "inference.json").stat().st_mtime) > outer_t[-1])
    after = [n for n in sorted(p.stem for p in RES.glob("AMENDMENT*.json"))
             if push and parse_iso(jload(RES / f"{n}.json")["written_at"]) > push]
    out["amendments_after_evaluation_lock"] = after
    later = git("log", "--format=%H", f"{gate.get('commit')}..HEAD") or ""
    lc = jload(RES / "EVALUATION_LOCK.json")["locked_code_files"]
    changed_after = sorted(set((git("diff", "--name-only", gate.get("commit"), "HEAD") or "").split()) & set(lc)) \
        if gate.get("commit") else []
    out["commits_after_evaluation_lock"] = len([x for x in later.splitlines() if x])
    out["locked_code_changed_after_evaluation_lock"] = changed_after
    for kk in ("assessment_opened_after_push", "all_inner_before_lock_push", "inference_written_after_last_outer"):
        if not out[kk]:
            bad.append(kk)
    if after or changed_after:
        bad.append("post-evaluation-lock changes")
    return res("FAIL" if bad else "PASS", failures=bad, **out)


# ------------------------------------------------------------------------------------------------ restore replay
def check_restore():
    roots = sorted(CACHE.glob("dpc_v1_local_copy_*"))
    if not roots:
        return res("PENDING", reason="no same-device copy present")
    root = roots[-1]
    t0 = time.process_time()
    sums = root / "SHA256SUMS"
    lines = [l_.split(None, 1) for l_ in sums.read_text().splitlines() if l_.strip()]
    bad = [rel.strip() for hsh, rel in lines if not (root / rel.strip()).exists() or sha_file(root / rel.strip()) != hsh]
    copy = root / "dpc_v1"
    cu, ca = copy / "run" / "units", copy / "admitted"
    out = {"copy": "<PRIVATE_CACHE>/" + root.name + "/dpc_v1", "kind": "SAME-DEVICE copy (not a drive restore)",
           "sha256sums_files": len(lines), "sha256sums_mismatches": bad[:10],
           "sha256sums_file_sha256_equals_BACKUP_VERIFICATION": sha_file(sums) ==
           jload(RES / "BACKUP_VERIFICATION.json").get("SHA256SUMS_sha256")}
    # rebuild everything from the copy only: input, teachers, policies, an attacker
    global SRC
    live_src = SRC
    SRC = ca / "inputs" / "adult_jcv.npz"
    try:
        Dc = Data()
    finally:
        SRC = live_src
    out["input_sha256_ok"] = Dc.src_sha == SRC_SHA
    k = 1
    Tc, fails = {}, []
    for t in TEACHERS:
        ud = ca / f"rel__s{k}__{t}"
        sd = tload(ud / "model.pt")
        o = {}
        for i in (0, 1):
            r = encode(sd, i, Dc.X)
            c, p, d = head_outputs(joblib.load(ud / f"head_{i}.joblib"), r)
            o.update({f"p{i + 1}": p, f"d{i + 1}": d.astype(np.int64), f"c{i + 1}": c, f"r{i + 1}": r})
        tz = np.load(cu / f"tea__s{k}__{t}" / "teacher.npz", allow_pickle=False)
        lz = np.load(UNITS / f"tea__s{k}__{t}" / "teacher.npz", allow_pickle=False)
        eq = all(bitwise(o[x], tz[x]) and bitwise(o[x], lz[x]) for x in ("p1", "p2", "d1", "d2", "c1", "c2", "r1", "r2"))
        out[f"teacher_{t}_s{k}_outputs_bitwise_copy_and_live"] = eq
        if not eq:
            fails.append(t)
        Tc[t] = o
    fit = Dc.fit_idx
    for cid in ("U|JOINT|m8|l0.1", "U|FINE-TASK|m8"):
        pj = jload(cu / pol_unit(k, cid) / "policy.json")
        rz = np.load(cu / pol_unit(k, cid) / "release.npz", allow_pickle=False)
        lz = np.load(UNITS / pol_unit(k, cid) / "release.npz", allow_pickle=False)
        ok = True
        full = {}
        for i in (1, 2):
            pol = Pol(pj[f"p{i}"])
            P, d = Tc["U"][f"p{i}"], Tc["U"][f"d{i}"]
            fine = fit_fine(P[fit], d[fit], KS[i - 1])
            ok &= all(fine.equal(pol.fine).values())
            tok = pol.cell_token[assign(P, d, fine)]
            q, hard = pol.proto[tok], pol.token_class[tok]
            ok &= bitwise(tok, rz[f"tok{i}"]) and bitwise(q, rz[f"q{i}"]) and bitwise(hard, rz[f"hard{i}"])
            ok &= bitwise(tok, lz[f"tok{i}"]) and bitwise(q, lz[f"q{i}"]) and bool(np.array_equal(hard, d))
            full.update({f"tok{i}": tok, f"p{i}": q, f"hard{i}": hard, f"alpha{i}": pol.T})
        out[f"policy_{cid}_s{k}_rebuilt_from_copy_bitwise"] = bool(ok)
        if not ok:
            fails.append(cid)
        if cid == "U|FINE-TASK|m8":
            # attacker: the AUC-selected pair attacker of outer__s1__U_FINE-TASK_m8 refit from the copy only
            src_c = ca / "inputs" / "adult_jcv.npz"
            Sall = np.load(src_c, allow_pickle=False)["sex"].astype(np.int64)[Dc.keep]
            fit_idx = np.flatnonzero(Dc.mask["AUDIT_FIT"])
            sel_idx = np.flatnonzero(Dc.mask["INNER_SELECTION"])
            a_idx = np.flatnonzero(Dc.mask[ASSESS])
            pred_idx = np.concatenate([sel_idx, a_idx])
            rec = jload(cu / f"outer__s{k}__U_FINE-TASK_m8" / "record.json")["families"]["code"]["scored"]["pair"]["auc"]
            cand, view, att = rec["label"].split(":")
            X, _ = code_views(full, Dc)
            pz = np.load(cu / f"outer__s{k}__U_FINE-TASK_m8" / "preds.npz", allow_pickle=False)["P_auc_pair"]
            mx_ = 0.0
            for s_ in SEEDS:
                p1 = p_one(my_factory(att, s_).fit(X[view][fit_idx], Sall[fit_idx]), X[view][pred_idx])[len(sel_idx):]
                mx_ = max(mx_, maxdiff(np.stack([1 - p1, p1], 1), pz[s_]))
            out["attacker_refit_from_copy"] = {"unit": f"outer__s{k}__U_FINE-TASK_m8", "attacker": rec["label"],
                                               "assessment_predictions_max_abs_diff": mx_}
            if mx_ > 1e-9:
                fails.append("attacker")
    out["cpu_s"] = time.process_time() - t0
    out["status_note"] = ("SAME_DEVICE_RESTORE_VERIFIED; OFF_DEVICE_BACKUP_PENDING (a same-device copy is not a drive "
                          "restore)")
    st = "FAIL" if (bad or fails or not out["input_sha256_ok"]) else "INFO"
    return res(st, failures=fails, **out)


# ------------------------------------------------------------------------------------------------ reports
def check_reports(D: Data, LU, preds, EL):
    import csv as _csv
    a_idx = np.flatnonzero(D.mask[ASSESS])
    const = const_classes(LU, D)
    mx, n, miss = 0.0, 0, []
    with open(RES / "ACTUAL_TASK_UTILITY.csv") as fh:
        for row in _csv.DictReader(fh):
            if row["seed"] in ("mean", ""):
                continue
            k, lab = int(row["seed"]), row["label"]
            if (k, lab) not in preds:
                miss.append(f"{lab}#s{k}")
                continue
            i = 0 if row["task"] == "income" else 1
            p = preds[(k, lab)]
            y = LU["y_income" if i == 0 else "y_occ"][a_idx]
            m_ = my_utility(p[f"prob{i + 1}"], p[f"hard{i + 1}"], y, KS[i], const[i])
            m_["gain"] = m_["acc"] - m_["const_acc"]
            P2STATE.setdefault("assess_utility", {})[(k, lab, i)] = {kk: v for kk, v in m_.items()
                                                                     if kk not in ("ll_rows", "br_rows")}
            for f in ("acc", "logloss", "brier", "const_acc", "gain", "balanced_acc", "minority_recall", "ece"):
                if row.get(f) not in (None, ""):
                    mx = max(mx, abs(float(row[f]) - m_[f]))
            rr = json.loads(row["recalls"])
            mx = max(mx, max(abs(float(v) - m_["recalls"][int(c)]) for c, v in rr.items()))
            n += 1
    ok = n == 3 * len(EL["scored_labels"]) * 2 and mx <= 6e-7 and not miss
    return res("PASS" if ok else "FAIL", actual_task_utility_rows_compared=n, max_abs_diff_6dp=mx, missing=miss[:10],
               note="per seed / label / task on the assessment rows: accuracy, log loss, Brier, constant accuracy, gain, "
                    "balanced accuracy (classes >= 30 rows), minority recall, ECE (10 bins), every class recall")


# ------------------------------------------------------------------------------------------------ decision documents
def check_documents(my_obj, checks):
    """Numbers and statements in RESEARCH_DECISION.md, PAPER_ADDENDUM.md, ADVISOR_BRIEF.md and RUN_STATUS.json
    (validity_note) vs own recomputation. Every check names an anchor and the exact text fragment that must appear on
    one line of the CURRENT document (so the check follows the document, not a stale copy), and the own value(s) the
    fragment's number(s) must equal at the quoted rounding."""
    lev, prim, sel, ut = P2STATE["levels"], P2STATE["primary"], P2STATE["selection"], P2STATE["assess_utility"]
    rowsS = sel["rows"]
    docs = {n: (RES / n).read_text().splitlines() for n in ("RESEARCH_DECISION.md", "PAPER_ADDENDUM.md", "ADVISOR_BRIEF.md")}
    rs = jload(RES / "RUN_STATUS.json") if (RES / "RUN_STATUS.json").exists() else {}
    docs["RUN_STATUS.json"] = [str(rs.get("validity_note", ""))]
    rows = []

    def present(doc, anchor, text):
        return any(anchor in ln and text in ln for ln in docs[doc])

    def chk(doc, anchor, text, what, quoted, own, dec, kind="num"):
        tol = 0.5 * 10 ** (-dec) + 1e-12
        if kind == "bool":
            ok_v = bool(own) is bool(quoted)
        elif kind == "int":
            ok_v = quoted == own
        elif kind == "le":
            ok_v = own <= quoted + tol
        else:
            q = quoted if isinstance(quoted, (tuple, list)) else (quoted,)
            o = own if isinstance(own, (tuple, list)) else (own,)
            ok_v = len(q) == len(o) and all(abs(x - y) <= tol for x, y in zip(q, o))
        pr = present(doc, anchor, text)
        rows.append({"doc": doc, "anchor": anchor, "text": text, "what": what, "quoted": quoted, "own": own,
                     "decimals": dec, "text_present": pr, "value_ok": bool(ok_v), "ok": bool(pr and ok_v)})

    def R(lab, fam, v):
        return lev[f"Rmean#{lab}#{fam}#{v}"]["point"]

    def m(q, lab, j):
        return lev[f"{q}mean#{lab}#{j}"]["point"]

    RD, PA, AB, RS = "RESEARCH_DECISION.md", "PAPER_ADDENDUM.md", "ADVISOR_BRIEF.md", "RUN_STATUS.json"
    U, J, FT8, LO = "SRC|U", "U|JOINT|m8|l0.1", "U|FINE-TASK|m8", "U|LOCAL|m8|l0.1"
    S12, S21, CL, RJ = "U|SEQ-12|m8|l0.1", "U|SEQ-21|m8|l0.1", "U|CLASS|m1", "SRC|RAW-J_b0.3"
    P = "primary"
    # ---- RESEARCH_DECISION plain-language answer
    a_ = "Even the finest code (8 states per class, U teacher)"
    chk(RD, a_, "about +0.020 nats of occupation log loss and +0.007 Brier", "FINE-TASK m8 occupation cost vs U",
        (0.020, 0.007), (m("ll", FT8, 1) - m("ll", U, 1), m("br", FT8, 1) - m("br", U, 1)), 3)
    chk(RD, a_, "Income confidence is nearly free: +0.001 nats for that task-only code and +0.002 for the joint code",
        "FINE-TASK m8 and JOINT m8 income log loss vs U", (0.001, 0.002),
        (m("ll", FT8, 0) - m("ll", U, 0), m("ll", J, 0) - m("ll", U, 0)), 3)
    chk(RD, "class-only (decision-only) code fails it", "by about 0.1 nats", "class-only occupation cost", 0.1,
        m("ll", CL, 1) - m("ll", U, 1), 1)
    a_ = "Its descriptive fallback (U, JOINT, m 8"
    chk(RD, a_, "leaked 0.018 less on the pair", "P01 point", 0.018, prim["P01"]["point"], 3)
    chk(RD, a_, "(pair 0.807 vs 0.807 and 0.809)", "joint vs SEQ-21 and SEQ-12 pair", (0.807, 0.807, 0.809),
        (R(J, P, "pair"), R(S21, P, "pair"), R(S12, P, "pair")), 3)
    chk(RD, "less than the continuous scores", "It leaked 0.051 less than the continuous scores", "P12 point", 0.051,
        prim["P12"]["point"], 3)
    chk(RD, "This is an exploratory, locked benchmark", "13,936 previously used Adult rows", "assessment rows", 13936,
        int(checks["roles"]["counts"][ASSESS]["rows"]), 0, "int")
    # ---- RESEARCH_DECISION results table
    tab = [("| U: features + scores", U, "complete", ("0.883", "0.859", "0.878"), None, None, "0.844 / 0.475"),
           ("| U: continuous scores", U, P, ("0.858", "0.697", "0.856"), "0.338 / 1.269", "0.214 / 0.652", "0.844 / 0.475"),
           ("| U FINE-TASK m8", FT8, P, ("0.825", "0.697", "0.786"), "0.339 / 1.289", "0.215 / 0.660", None),
           ("| U LOCAL m8", LO, P, ("0.820", "0.695", "0.770"), "0.339 / 1.289", "0.215 / 0.660", None),
           ("**U JOINT m8", J, P, ("**0.807**", "0.696", "0.764"), "0.340 / 1.290", "0.215 / 0.660", None),
           ("| U class-only (decisions alone)", CL, P, ("0.739", "0.587", "0.687"), "0.423 / 1.379", "0.257 / 0.693", None),
           ("| RAW-J β 0.3 continuous", RJ, P, ("0.788", "0.685", "0.774"), "0.335 / 1.282", "0.212 / 0.659", "0.845 / 0.466"),
           ("| RAW-J FINE-TASK m8", "RAW-J_b0.3|FINE-TASK|m8", P, ("0.752", "0.684", "0.704"), "0.336 / 1.300",
            "0.214 / 0.667", None),
           ("| FARE (official)", "REF|F", P, ("0.704", "0.685", "0.636"), "0.352 / 1.315", "0.224 / 0.676", "0.844 / 0.451"),
           ("| F0 (no-fairness", "REF|F0", P, ("0.865", "0.803", "0.851"), "0.327 / 1.298", "0.209 / 0.665", "0.849 / 0.460"),
           ("| LEACE (official)", "REF|E", P, ("0.808", "0.554", "0.801"), "0.454 / 1.320", "0.291 / 0.680", "0.794 / 0.436")]

    def nums(t):
        return tuple(float(x.strip("*+")) for x in t.replace(" / ", " ").replace("–", " ").split())

    for anchor, lab, fam, aucs, ll, br, acc in tab:
        for v, t in zip(("pair", "v1", "v2"), aucs):
            chk(RD, anchor, t if t.startswith("**") else f"| {t} |", f"{lab} {fam} {v} AUC", nums(t)[0], R(lab, fam, v), 3)
        for q, t in (("ll", ll), ("br", br), ("acc", acc)):
            if t:
                chk(RD, anchor, t, f"{lab} {q} income / occupation", nums(t), (m(q, lab, 0), m(q, lab, 1)), 3)
    A_ = "| U SEQ-12 / SEQ-21"
    chk(RD, A_, "0.809 / 0.807", "SEQ-12 / SEQ-21 pair", (0.809, 0.807), (R(S12, P, "pair"), R(S21, P, "pair")), 3)
    chk(RD, A_, "0.695 / 0.695", "SEQ-12 / SEQ-21 income AUC", (0.695, 0.695), (R(S12, P, "v1"), R(S21, P, "v1")), 3)
    chk(RD, A_, "0.763 / 0.752", "SEQ-12 / SEQ-21 occupation AUC", (0.763, 0.752), (R(S12, P, "v2"), R(S21, P, "v2")), 3)
    chk(RD, A_, "0.339 / 1.291 and 0.340 / 1.292", "SEQ-12 and SEQ-21 log loss", (0.339, 1.291, 0.340, 1.292),
        (m("ll", S12, 0), m("ll", S12, 1), m("ll", S21, 0), m("ll", S21, 1)), 3)
    chk(RD, A_, "0.215 / 0.660 and 0.215 / 0.661", "SEQ-12 and SEQ-21 Brier", (0.215, 0.660, 0.215, 0.661),
        (m("br", S12, 0), m("br", S12, 1), m("br", S21, 0), m("br", S21, 1)), 3)
    # ---- primary claims table
    for anchor, sid, t_pt, pt, t_iv, iv, dpt, div in (
            ("| A: J", "P01", "0.0176 [0.014, 0.021]", 0.0176, None, (0.014, 0.021), 4, 3),
            ("| A: J", "P07", "+0.0214 [0.015, 0.028]", 0.0214, None, (0.015, 0.028), 4, 3),
            ("| A: J", "P09", "+0.0079 [0.0055, 0.0103]", 0.0079, None, (0.0055, 0.0103), 4, 4),
            ("| B: J", "P12", "0.0515 [0.045, 0.058]", 0.0515, None, (0.045, 0.058), 4, 3),
            ("| C: P", "P23", "0.0389 [0.033, 0.045]", 0.0389, None, (0.033, 0.045), 4, 3),
            ("| C: P", "P29", "+0.0209 [0.014, 0.027]", 0.0209, None, (0.014, 0.027), 4, 3),
            ("| C: P", "P31", "+0.0075 [0.005, 0.010]", 0.0075, None, (0.005, 0.010), 4, 3)):
        chk(RD, anchor, t_pt, f"{sid} point", pt, prim[sid]["point"], dpt)
        chk(RD, anchor, t_pt, f"{sid} interval", iv, (prim[sid]["lower"], prim[sid]["upper"]), div)
    chk(RD, "| B: J", "+0.0214", "P18 (B occupation log loss, alias of P07)", 0.0214, prim["P18"]["point"], 4)
    chk(RD, "| B: J", "+0.0079", "P20 (B occupation Brier, alias of P09)", 0.0079, prim["P20"]["point"], 4)
    for anchor, c, n_ in (("| A: J", "A", 8), ("| B: J", "B", 9), ("| C: P", "C", 9)):
        chk(RD, anchor, f"{n_}/11", f"claim {c} numeric clauses passing", n_,
            checks["endpoints"]["own_claims"][c]["numeric_clauses_passing"], 0, "int")
    # ---- views, explanatory findings
    chk(RD, "| Features + scores |", "0.883", "features + scores pair", 0.883, R(U, "complete", "pair"), 3)
    lo_hi = (min(R(U, P, "pair"), R(U, "scores", "pair")), max(R(U, P, "pair"), R(U, "scores", "pair")))
    chk(RD, "| Scores alone |", "0.858–0.864", "scores alone (interface, scores)", (0.858, 0.864), lo_hi, 3)
    tc = [f"U|{f}|m{mm}" for f in TASK_FAMS for mm in RATES]
    chk(RD, "| Compact token", "0.789–0.825", "task-only m2-m8 range", (0.789, 0.825),
        (min(R(x, P, "pair") for x in tc), max(R(x, P, "pair") for x in tc)), 3)
    chk(RD, "| Compact token", "0.807 (joint m8)", "joint m8 pair", 0.807, R(J, P, "pair"), 3)
    chk(RD, "| Decisions alone |", "0.739", "decisions alone (U class-only)", 0.739, R(CL, P, "pair"), 3)
    chk(RD, "decision vector alone", "at 0.74", "decision vector pair", 0.74, R(CL, P, "pair"), 2)
    a_ = "privacy training lowers pair AUC"
    chk(RD, a_, "by 0.018 (joint)", "task-only m8 - joint m8 pair", 0.018, R(FT8, P, "pair") - R(J, P, "pair"), 3)
    chk(RD, a_, "or 0.005 (local)", "task-only m8 - local m8 pair", 0.005, R(FT8, P, "pair") - R(LO, P, "pair"), 3)
    chk(RD, a_, "about 0.001 nats", "joint - task-only occupation log loss", 0.001, m("ll", J, 1) - m("ll", FT8, 1), 3)
    a_ = "Joint m8 dominates task-only m4"
    chk(RD, a_, "(0.807 vs 0.811)", "joint m8 vs FINE-TASK m4 pair", (0.807, 0.811),
        (R(J, P, "pair"), R("U|FINE-TASK|m4", P, "pair")), 3)
    chk(RD, a_, "(1.290 vs 1.303)", "joint m8 vs FINE-TASK m4 occupation log loss", (1.290, 1.303),
        (m("ll", J, 1), m("ll", "U|FINE-TASK|m4", 1)), 3)
    dom = all(R(J, P, "pair") < R(f"U|{f}|m4", P, "pair") and m("ll", J, 1) < m("ll", f"U|{f}|m4", 1) for f in TASK_FAMS)
    chk(RD, a_, "dominates", "joint m8 below both task-only m4 codes (pair and occupation log loss)", True, dom, 0, "bool")
    for fam_, lab in (("JOINT", J), ("SEQ-21", S21), ("SEQ-12", S12), ("LOCAL", LO)):
        pr_, ll_ = round(R(lab, P, "pair"), 3), round(m("ll", lab, 1), 3)
        chk(RD, f"| {fam_} |", f"| {pr_:.3f} | ≈ {ll_:.3f} |", f"{fam_} family table", (pr_, ll_),
            (R(lab, P, "pair"), m("ll", lab, 1)), 3)
    a_ = "Joint fitting is marginally below both sequential orders"
    chk(RD, a_, "by 0.0004 and 0.0016", "SEQ-21 - joint, SEQ-12 - joint (assessment pair)", (0.0004, 0.0016),
        (R(S21, P, "pair") - R(J, P, "pair"), R(S12, P, "pair") - R(J, P, "pair")), 4)
    cells = [(t, mm, l_) for t in TEACHERS for mm in RATES for l_ in LAMS]
    lowest, below_local, margins = 0, 0, []
    for t, mm, l_ in cells:
        jp = rowsS[config_id(t, "JOINT", mm, l_)]["mean_pair"]
        sq = min(rowsS[config_id(t, "SEQ-12", mm, l_)]["mean_pair"], rowsS[config_id(t, "SEQ-21", mm, l_)]["mean_pair"])
        if jp < sq:
            lowest += 1
            margins.append(sq - jp)
        below_local += int(jp < rowsS[config_id(t, "LOCAL", mm, l_)]["mean_pair"])
    chk(RD, a_, "lowest in 11 of 18", "inner cells JOINT below both sequential orders", 11, lowest, 0, "int")
    chk(RD, a_, "less than 0.007", "largest such inner margin", 0.007, max(margins), 3, "le")
    chk(RD, a_, "below local in 14 of 18", "inner cells JOINT below LOCAL", 14, below_local, 0, "int")
    a_ = "lowers fitted I(S; C1, C2)"
    chk(RD, a_, "falls 0.825 → 0.805", "inner pair task-only m8 -> joint m8", (0.825, 0.805),
        (rowsS[FT8]["mean_pair"], rowsS[J]["mean_pair"]), 3)
    if my_obj:
        def obj(cid, q):
            return float(np.mean([my_obj[(k, "U", cid)][q] for k in SEEDS]))
        chk(RD, a_, "from 0.201 to 0.178 nats", "fitted I12 task-only m8 -> joint m8", (0.201, 0.178),
            (obj(FT8, "I12"), obj(J, "I12")), 3)
        a2 = "λ 10 lowers I12"
        chk(RD, a2, "to 0.112", "fitted I12 joint m8 l10", 0.112, obj("U|JOINT|m8|l10", "I12"), 3)
        chk(RD, a2, "D1 0.002 → 0.068", "fitted D1 task-only m8 -> joint l10", (0.002, 0.068),
            (obj(FT8, "D1"), obj("U|JOINT|m8|l10", "D1")), 3)
        chk(RD, a2, "D2 0.034 → 0.074", "fitted D2 task-only m8 -> joint l10", (0.034, 0.074),
            (obj(FT8, "D2"), obj("U|JOINT|m8|l10", "D2")), 3)
    chk(RD, "λ 10 lowers I12", "inner pair AUC to 0.757", "inner pair joint m8 l10", 0.757,
        rowsS["U|JOINT|m8|l10"]["mean_pair"], 3)
    chk(RD, "Confidence cost of m = 1", "+0.085 nats (income) and +0.110 nats (occupation)", "class-only cost",
        (0.085, 0.110), (m("ll", CL, 0) - m("ll", U, 0), m("ll", CL, 1) - m("ll", U, 1)), 3)
    a_ = "U's occupation recalls"
    for c, t in enumerate(("0.495–0.499", "0.466–0.483", "0.390–0.400", "0.020–0.024", "0.703–0.709")):
        vals = [ut[(k, U, 1)]["recalls"][c] for k in SEEDS]
        chk(RD, a_, t, f"U occupation recall class {c} range over seeds", nums(t), (min(vals), max(vals)), 3)
    chk(RD, a_, "**0.000**", "U occupation recall class 5", 0.0, max(ut[(k, U, 1)]["recalls"].get(5, 0.0) for k in SEEDS), 3)
    ba = [ut[(k, U, 1)]["balanced_acc"] for k in SEEDS]
    chk(RD, a_, "balanced accuracy 0.417–0.421", "U balanced accuracy range", (0.417, 0.421), (min(ba), max(ba)), 3)
    never5 = all(v["recipient_2"]["predicted_class_counts_by_role"][r_][5] == 0 for v in checks["teachers"]["units"].values()
                 for r_ in ROLES)
    chk(RD, a_, "Class 5 is never predicted by any teacher", "class 5 never predicted (all teachers, roles)", True, never5,
        0, "bool")
    r3 = [ut[(k, RJ, 1)]["recalls"][3] for k in SEEDS]
    chk(RD, "RAW-J is similar", "class 3 recall 0.015–0.028", "RAW-J class 3 recall range", (0.015, 0.028), (min(r3), max(r3)), 3)
    chk(RD, "RAW-J is similar", "0.9 occupation-accuracy points", "RAW-J occupation accuracy below U (points)", 0.9,
        100 * (m("acc", U, 1) - m("acc", RJ, 1)), 1)
    rc = [f"RAW-J_b0.3|{f}|m{mm}" for f in TASK_FAMS for mm in RATES]
    a_ = "**RAW-J teacher codes.**"
    chk(RD, a_, "pair 0.737–0.755", "RAW-J task-only codes pair range", (0.737, 0.755),
        (min(R(x, P, "pair") for x in rc), max(R(x, P, "pair") for x in rc)), 3)
    chk(RD, a_, "0.693 class-only", "RAW-J class-only pair", 0.693, R("RAW-J_b0.3|CLASS|m1", P, "pair"), 3)
    chk(RD, a_, "+0.013-nat", "RAW-J occupation log-loss gap vs U", 0.013, m("ll", RJ, 1) - m("ll", U, 1), 3)
    chk(RD, "FARE leaks least", "(pair 0.704)", "FARE pair", 0.704, R("REF|F", P, "pair"), 3)
    chk(RD, "FARE leaks least", "loses 2.4 occupation points and 0.046 nats", "FARE occupation cost", (2.4, 0.046),
        (round(100 * (m("acc", U, 1) - m("acc", "REF|F", 1)), 1), m("ll", "REF|F", 1) - m("ll", U, 1)), 3)
    chk(RD, "All three are task-ineligible", "task-ineligible", "E, F, F0 all task-ineligible", True,
        not any(rowsS[f"REF|{r_}"]["task_eligible"] for r_ in REFS), 0, "bool")
    chk(RD, "LEACE loses", "5 income points", "LEACE income accuracy loss (points)", 5,
        100 * (m("acc", U, 0) - m("acc", "REF|E", 0)), 0)
    pol = [c for c in rowsS if not c.startswith(("SRC|", "REF|"))]
    pr1 = all(any(rowsS[c]["seeds"][k]["gm"][1]["logloss"] < 0 or rowsS[c]["seeds"][k]["gm"][1]["brier"] < 0
                  for k in SEEDS) for c in pol)
    chk(RD, "| PR1 |", "All fail occupation log loss and/or Brier on some seed", "PR1 statement", True, pr1, 0, "bool")
    chk(RD, "| PR3 |", "0.805 vs 0.812 / 0.811", "PR3 inner pair JOINT / SEQ-12 / SEQ-21 at the fallback cell",
        (0.805, 0.812, 0.811), (rowsS[J]["mean_pair"], rowsS[S12]["mean_pair"], rowsS[S21]["mean_pair"]), 3)
    chk(RD, "| PR3 |", "in 11 of 18 cells overall", "PR3 cell count", 11, lowest, 0, "int")
    chk(RD, "| PR4 |", "T\\* = C_global = U continuous", "PR4 statement", True,
        sel["statuses"]["T*"].get("config") == U and sel["statuses"]["C_global"].get("config") == U, 0, "bool")
    cc = checks["controls_coverage"]["null_calibration"]
    chk(RD, "**Controls.**", "0 of 15 exceedances (held-out mean AUC 0.500)", "null calibration", (0, 15, 0.500),
        (cc["exceedances"], cc["tests"], cc["heldout_mean"]), 3)
    n_rows = int(sum(checks["class_preservation"]["row_checks_passed_by_role"].values()))
    chk(RD, "That is", "20.2 million row checks", "recipient-row class-preservation checks (millions)", 20.2, n_rows / 1e6, 1)
    # ---- the claim-A disclosure
    cm_all = sel["C_match_all"]
    n_nf = sum(v["status"] == "NO_FEASIBLE_CONTROL" for v in cm_all.values())
    a_ = "**Matched-control coverage failure (claim A).**"
    chk(RD, a_, "NO_FEASIBLE_CONTROL at all six teacher/rate cells", "C_match NO_FEASIBLE_CONTROL cells", 6, n_nf, 0, "int")
    chk(RD, a_, "INCOMPLETE_OR_INVALID", "strict-reading label stated", True, True, 0, "bool")
    chk(RD, a_, "No JOINT policy is task-eligible", "no JOINT policy task-eligible", True,
        sel["statuses"]["J*"].get("joint_task_eligible") == 0, 0, "bool")
    chk(RD, a_, "`missing_guards`", "J* C_match guard recorded missing (SELECTION.json and own)", True,
        jload(RES / "SELECTION.json")["statuses"]["J*"].get("missing_guards") == ["C_match"] and
        sel["statuses"]["J*"].get("missing_guards") == ["C_match"], 0, "bool")
    chk(RD, a_, "Claims B and C have a valid, eligible comparator (U continuous)", "B and C comparators NOMINEE U", True,
        all(sel["statuses"][x]["status"] == "NOMINEE" and sel["statuses"][x].get("config") == U for x in ("C_global", "T*")),
        0, "bool")
    # ---- PAPER_ADDENDUM
    pa = [("| U continuous (features + scores view)", U, "complete", "0.883", "0.878", None, None),
          ("| U continuous (released interface)", U, P, "0.858", "0.856", "1.269", "0.652"),
          ("| U task-only code, m 8", FT8, P, "0.825", "0.786", "1.289", "0.660"),
          ("| U joint code, m 8", J, P, "0.807", "0.764", "1.290", "0.660"),
          ("| U class-only (decisions)", CL, P, "0.739", "0.687", "1.379", "0.693"),
          ("| FARE (official)", "REF|F", P, "0.704", "0.636", "1.315", "0.676")]
    for anchor, lab, fam, pr_, v2, ll, br in pa:
        chk(PA, anchor, f"| {pr_} |", f"{lab} pair", float(pr_), R(lab, fam, "pair"), 3)
        chk(PA, anchor, f"| {v2} |", f"{lab} occupation AUC", float(v2), R(lab, fam, "v2"), 3)
        if ll:
            chk(PA, anchor, f"| {ll} |", f"{lab} occupation log loss", float(ll), m("ll", lab, 1), 3)
            chk(PA, anchor, f"| {br} |", f"{lab} occupation Brier", float(br), m("br", lab, 1), 3)
    a_ = "| U sequential codes"
    for t, f_ in (("0.807–0.809", lambda x: R(x, P, "pair")), ("0.752–0.763", lambda x: R(x, P, "v2")),
                  ("1.291–1.292", lambda x: m("ll", x, 1)), ("0.660–0.661", lambda x: m("br", x, 1))):
        chk(PA, a_, t, f"sequential range {t}", nums(t), (min(f_(S12), f_(S21)), max(f_(S12), f_(S21))), 3)
    a_ = "**Eligibility.**"
    chk(PA, a_, "+0.021 nats [0.015, 0.028]", "P07 point and interval", (0.021, 0.015, 0.028),
        (prim["P07"]["point"], prim["P07"]["lower"], prim["P07"]["upper"]), 3)
    chk(PA, a_, "+0.008 Brier [0.006, 0.010]", "P09 point and interval", (0.008, 0.006, 0.010),
        (prim["P09"]["point"], prim["P09"]["lower"], prim["P09"]["upper"]), 3)
    chk(PA, a_, "within +0.002 nats", "P06 income log loss vs U", 0.002, prim["P06"]["point"], 3, "le")
    chk(PA, "Joint versus matched task-only coding", "0.018 [0.014, 0.021]", "P01", (0.018, 0.014, 0.021),
        (prim["P01"]["point"], prim["P01"]["lower"], prim["P01"]["upper"]), 3)
    chk(PA, "Joint versus sequential", "within 0.002", "max(SEQ - joint) pair", 0.002,
        max(abs(R(S12, P, "pair") - R(J, P, "pair")), abs(R(S21, P, "pair") - R(J, P, "pair"))), 3, "le")
    chk(PA, "Joint versus the continuous scores", "0.051 [0.045, 0.058]", "P12", (0.051, 0.045, 0.058),
        (prim["P12"]["point"], prim["P12"]["lower"], prim["P12"]["upper"]), 3)
    chk(PA, "joint decision vector alone", "about 0.74", "decision vector pair", 0.74, R(CL, P, "pair"), 2)
    chk(PA, "confidence-borne leakage", "about 0.05 pair AUC", "U continuous - joint pair", 0.05,
        R(U, P, "pair") - R(J, P, "pair"), 2)
    chk(PA, "Coordinated fitting", "about 0.018 pair AUC", "P01 point", 0.018, prim["P01"]["point"], 3)
    chk(PA, "every row of every role", "20.2 million recipient-row checks", "row checks (millions)", 20.2, n_rows / 1e6, 1)
    chk(PA, "**Matched-control coverage.**", "INCOMPLETE_OR_INVALID", "strict-reading label stated", True, True, 0, "bool")
    chk(PA, "**Matched-control coverage.**", "(NO_FEASIBLE_CONTROL)", "every matched control NO_FEASIBLE_CONTROL", 6, n_nf, 0,
        "int")
    # ---- ADVISOR_BRIEF
    chk(AB, "income is nearly free", "(+0.001 to +0.002 nats)", "finest codes (task-only, joint m8) income log loss vs U",
        (0.001, 0.002), (m("ll", FT8, 0) - m("ll", U, 0), m("ll", J, 0) - m("ll", U, 0)), 3)
    occ_ll = [m("ll", x, 1) - m("ll", U, 1) for x in (FT8, J)]
    occ_br = [m("br", x, 1) - m("br", U, 1) for x in (FT8, J)]
    chk(AB, "income is nearly free", "+0.020 to +0.021 nats", "finest codes (task-only, joint m8) occupation log loss",
        (0.020, 0.021), (min(occ_ll), max(occ_ll)), 3)
    chk(AB, "income is nearly free", "+0.007 to +0.008 Brier", "finest codes occupation Brier", (0.007, 0.008),
        (min(occ_br), max(occ_br)), 3)
    chk(AB, "The class-only code costs", "+0.11 nats", "class-only occupation cost", 0.11, m("ll", CL, 1) - m("ll", U, 1), 2)
    for anchor, val in (("| Continuous U scores |", R(U, P, "pair")), ("| Task-only code (m 8) |", R(FT8, P, "pair")),
                        ("| Joint code (m 8, λ 0.1) |", R(J, P, "pair")), ("| Decisions alone |", R(CL, P, "pair")),
                        ("| FARE |", R("REF|F", P, "pair"))):
        chk(AB, anchor, f"{round(val, 3):.3f}", f"advisor table {anchor}", round(val, 3), val, 3)
    chk(AB, "| Sequential codes |", "0.807–0.809", "advisor sequential range", (0.807, 0.809),
        (min(R(S12, P, "pair"), R(S21, P, "pair")), max(R(S12, P, "pair"), R(S21, P, "pair"))), 3)
    chk(AB, "| FARE |", "loses 2.4 occupation-accuracy points", "FARE occupation accuracy loss", 2.4,
        100 * (m("acc", U, 1) - m("acc", "REF|F", 1)), 1)
    chk(AB, "**Joint versus matched task-only:**", "0.018 [0.014, 0.021]", "P01", (0.018, 0.014, 0.021),
        (prim["P01"]["point"], prim["P01"]["lower"], prim["P01"]["upper"]), 3)
    chk(AB, "The decision vector alone", "about 0.74", "decision vector pair", 0.74, R(CL, P, "pair"), 2)
    chk(AB, "The decision vector alone", "the 0.86 that the scores carry", "U continuous pair", 0.86, R(U, P, "pair"), 2)
    chk(AB, "Joint fitting and privacy training help a little", "about 0.018 pair AUC", "P01 point", 0.018,
        prim["P01"]["point"], 3)
    chk(AB, "**Decisions:**", "all 258 fitted codes", "policies verified", 258, checks["policies"]["units"], 0, "int")
    chk(AB, "**Validity caveat:**", "INCOMPLETE_OR_INVALID", "strict-reading label stated", True, True, 0, "bool")
    # ---- RUN_STATUS validity note
    chk(RS, "", "NO_FEASIBLE_CONTROL at all 6 teacher/rate cells", "validity_note C_match cells", 6, n_nf, 0, "int")
    chk(RS, "", "INCOMPLETE_OR_INVALID", "validity_note strict reading", True, True, 0, "bool")
    chk(RS, "", "No claim passes under either reading", "no claim passes", True,
        all(v["decision"] != "PASS" for v in checks["endpoints"]["own_claims"].values()), 0, "bool")
    bad = [r for r in rows if not r["ok"]]
    return res("WARN" if bad else "PASS", numbers_and_statements_checked=len(rows), mismatches=bad,
               documents_sha256={n: sha_file(RES / n) for n in ("RESEARCH_DECISION.md", "PAPER_ADDENDUM.md",
                                                                  "ADVISOR_BRIEF.md", "RUN_STATUS.json")},
               all_rows=rows,
               note="each row: the exact text fragment must appear on a line containing the anchor in the current "
                    "document (text_present), and the fragment's value(s) must equal the own recomputation at the quoted "
                    "rounding (value_ok)")


def phase2_mutations(preds, EL, own):
    """The PHASE 2 recomputations are sensitive: a changed bootstrap seed or row-level (ungrouped) resampling moves
    the P01 SE; a swapped attacker column changes an inner selection; a flipped status changes a decision."""
    out = {}
    p0 = preds[(0, EL["scored_labels"][0])]
    sex = p0["sex"]
    J, C = EL["resolved"]["J*"], EL["resolved"]["C_match"]
    stats = []
    for k in SEEDS:
        for lab in (C, J):
            stats.append([AUCStat(preds[(k, lab)]["P_auc_pair"][s_][:, 1], sex == 1) for s_ in range(3)])

    def p01_se(groups, seed):
        b = MyBoot(groups, seed=seed)
        reps = []
        for W in b.chunks():
            per = []
            for k in range(3):
                c_ = np.mean([f(W) for f in stats[2 * k]], 0)
                j_ = np.mean([f(W) for f in stats[2 * k + 1]], 0)
                per.append(c_ - j_)
            reps.append(np.mean(per, 0))
        return float(np.std(np.concatenate(reps), ddof=1))

    rec = P2STATE["primary"]["P01"]["se"]
    se_seed = p01_se(p0["assess_unit"], BOOT_SEED + 1)
    se_rows = p01_se(np.arange(len(sex)), BOOT_SEED)
    out["P01_se_recorded"] = rec
    out["P01_se_other_seed"] = se_seed
    out["P01_se_row_level"] = se_rows
    out["seed_change_detected"] = abs(se_seed - rec) > 1e-10
    out["grouping_change_detected"] = abs(se_rows - rec) > 1e-12
    k0 = (0, "U|JOINT|m8|l0.1")
    sel_label = own[k0]["families"]["code"]["selected"]["pair"]
    out["inner_selected_pair_label"] = sel_label
    ok = out["seed_change_detected"] and out["grouping_change_detected"]
    return res("PASS" if ok else "FAIL", **out,
               note="the 7 duplicate-row groups make row-level and group-level SEs differ slightly; a different seed "
                    "gives different draws")


def run_phase2(D: Data, L, T, refs, cache, checks, my_obj=None):
    t0 = time.time()
    gate = evaluation_lock_gate()
    checks["inner_eligibility"], own = check_inner(D, L, T, refs, cache)
    checks["selection"], mine_sel = check_selection(own, cache)
    checks["evaluation_lock"] = check_eval_lock(D, L, gate, mine_sel)
    LU = unsealed_labels(D, gate)
    unseal = {"at": iso(datetime.now(timezone.utc)), "after_lock_push": gate.get("first_push_time")}
    EL = jload(RES / "EVALUATION_LOCK.json")
    checks["outer_units"], preds = check_outer(D, LU, T, refs, cache, gate)
    checks["endpoints"] = check_endpoints(preds, EL)
    checks["conjunctions"] = res("PASS" if checks["endpoints"]["claims_and_label_equal"] else "FAIL",
                                 own_claims=checks["endpoints"]["own_claims"], own_label=checks["endpoints"]["own_label"],
                                 recorded_label=jload(RUN / "inference.json")["label"])
    checks["attacker_replay"] = check_attackers(D, LU, T, cache, preds)
    checks["controls_coverage"] = check_controls(D, L, cache, T)
    checks["composed_source_readers"] = res(
        "PASS" if checks["selection"]["status"] == "PASS" and checks["outer_units"]["status"] == "PASS" and
        checks["evaluation_lock"]["composed_policies_rule_ok"] else "FAIL",
        inner="composed source AUC recomputed per view as max(own interface, every same-teacher/seed policy code AUC) "
              "from own AUCs of the stored inner predictions; composed winners equal selection.json (selection)",
        final="outer source units carry exactly the locked composed policies (scored policies of the same teacher; "
              "class-only only in the decisions family) (outer_units.composed_units_equal_lock)",
        winners_on_scored_sources=checks["outer_units"].get("composed_winners"))
    checks["post_lock_amendments"] = check_amendments_and_phase2_chronology(gate)
    checks["restore_replay"] = check_restore()
    checks["reports"] = check_reports(D, LU, preds, EL)
    checks["decision_documents"] = check_documents(my_obj or {}, checks)
    checks["phase_2_mutation_power"] = phase2_mutations(preds, EL, own)
    return {"unseal": unseal, "wall_s": time.time() - t0}


LIGHT = ("exclusions_and_label_custody", "integrity", "custody")


def light_update(args):
    t0, c0 = time.time(), time.process_time()
    dest = Path(args.out) if args.out else OUT
    report = jload(dest)
    checks = report["checks"]
    prev = {"verifier_sha256": report["verifier_sha256"], "generated_at": report["generated_at"],
            "worktree_head": report["worktree_head"], "compute": report.get("compute"),
            "status_by_check": {k: checks[k].get("status") for k in LIGHT}}
    checks["exclusions_and_label_custody"] = check_label_custody()
    checks["integrity"] = {"complete_json": check_complete(), "lock_order": check_lock_order(),
                           "code_hashes": check_code_hashes(), "predictions": check_predictions()}
    checks["integrity"]["status"] = worst(*[v["status"] for v in checks["integrity"].values()])
    checks["custody"] = check_custody()
    loaded = sorted(m for m in sys.modules if m.split(".")[0] in _FORBIDDEN_TOP)
    report["independence"]["light_update_run"] = {"loaded_forbidden_modules": loaded, "blocked_attempts": _BLOCKED,
                                                  "preloaded_before_guard": _PRELOADED}
    if loaded or _BLOCKED or _PRELOADED:
        report["independence"]["status"] = "FAIL"
    report["light_update"] = {"at": iso(datetime.now(timezone.utc)), "sections_rerun": list(LIGHT),
                              "verifier_sha256": sha_file(Path(__file__)), "worktree_head": git("rev-parse", "HEAD"),
                              "heavy_sections_from": prev, "wall_s": time.time() - t0,
                              "cpu_s_process": time.process_time() - c0, "lead_dpc_workers": len(pgrep_dpc())}
    report["verifier_corrections"] = VERIFIER_CORRECTIONS
    report["verifier_compute_ledger"] = {"runs": VERIFIER_RUNS, "note": VERIFIER_RUNS_NOTE,
                                         "total_cpu_s": round(sum(r["cpu_s"] for r in VERIFIER_RUNS), 1),
                                         "total_cpu_h": round(sum(r["cpu_s"] for r in VERIFIER_RUNS) / 3600, 3)}
    status = {k: (v.get("status") if isinstance(v, dict) else None) for k, v in checks.items()}
    counts_all, flagged = walk_flags(checks)
    report["summary"] = {"status_by_check": status, "independence": report["independence"]["status"],
                         "overall_phase_1": worst(*[s_ for k, s_ in status.items() if k not in PHASE2],
                                                  report["independence"]["status"]),
                         "status_counts_all_nodes": counts_all, "flagged_fail_warn": flagged}
    report["pending"] = sorted(k for k, v in status.items() if v == "PENDING")
    text = json.dumps(jsonable(report), indent=1)
    scrub_check(text)
    if not args.no_write:
        tmp = dest.with_suffix(".json.tmp")
        tmp.write_text(text + "\n")
        tmp.replace(dest)
    print(json.dumps(jsonable({"summary": report["summary"], "light_update": report["light_update"]}), indent=1))


if __name__ == "__main__":
    main()
