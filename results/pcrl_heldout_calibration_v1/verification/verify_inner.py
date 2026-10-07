"""Role E independent INNER replay of the held-out calibration study (hcal; prompt section 13; owner: role E).

Reads only serialized artifacts and the sealed D, and re-derives every inner-stage number with its OWN code:

  (1) ROLE_SPLIT        CALIBRATION_HELDOUT / ATTACK_FIT_NEW / CALIBRATION_TRAIN_MATCHED from D["unit"], D["row_id"] and
                        the registered salts (sha256("hcal-v1|heldout|20261011|" + str(int(u))), ties by integer id;
                        representative = smallest row_id); counts, purity, disjointness; ROLE_MANIFEST.json receipts.
  (0) FROZEN_BANK       the admitted bank npz files: sha256 vs the admission receipt; internal consistency (classes,
                        decisions, teacher decisions, counts / sums on OSF_DEFENSE_FIT, mu = S / n, q0 = smooth(mu)).
  (2) CALIBRATION       H-GLOBAL-TEMP / H-CLASS-TEMP (codes and continuous U) by an own safeguarded-Newton solve of the
                        UNCLIPPED calibration NLL on [0.25, 4]; stored alphas certified with an own derivative; tables
                        recomputed; H-TOKEN32 / T-TOKEN32 by an own KKT (Frank-Wolfe vertex gap) certificate of
                        sum_i [-log q(Y_i) + 0.5 ||q - e_Yi||^2] + 32 KL(mu_t || q) on the class-dominant simplex for every
                        fitted token; q0 exactly for zero-count / reserved tokens; own label counts.
  (3) UTILITY           inner acc / log loss (clip 1e-12) / multiclass Brier / constant-class gain of every release and
                        seed (+ calibration and train-matched rows); decision preservation; U0 gate; Ucal*; Ucal gate;
                        utility table; the audit plan.
  (4) COMMON_RECORDS    every within-bank AUC / CE selection (fixed orientation P(SEX=1), first-in-bank-order ties 1e-12)
                        from the stored INNER predictions of every admitted legacy bank, every fresh bank and U's own
                        interface bank; attacker-seed means; the union rule (common records, the lra SRC|U composition
                        and hcal's U composition).
  (5) SELECTION         T* and P* with every rejection reason (guard <= T* + 0.005 inclusive per recipient per seed;
                        mean pair benefit >= 0.02; ordering keys rounded to 12 decimals; registered ID order).

INDEPENDENCE (prompt section 13). No hcal module is imported (checked at exit: sys.modules). Imports: json, hashlib,
math, os, subprocess, sys, time, pathlib, numpy; lra.data (pinned; ONLY lra.data.load() for the sealed D, which pulls
in the pinned qpc.data / dpc.data / osf.data loaders). Formulas, grouping, metrics, selection and union rules are
reconstructed here from PROTOCOL.md / CALIBRATION_RULES.json / SELECTION_RULES.json.

LABELS. No task or SEX label is read unless results/pcrl_heldout_calibration_v1/SCIENCE_LOCK.json exists and is
byte-identical on origin/research/pcrl-heldout-calibration-v1 (protocol section 10: no Adult label before the pushed
SCIENCE_LOCK). Then, recorded in "label_reads": task labels of the CALIBRATION_HELDOUT and CALIBRATION_TRAIN_MATCHED
representatives and of INNER_SELECTION; task labels of OSF_DEFENSE_FIT (the constant class only); SEX of
INNER_SELECTION (AUC / CE of stored predictions). HEAD_VALIDATION and assessment labels are never read; no outer__,
oprob__ or oatt__ unit is ever opened.

Status per check: PASS / FAIL / PENDING (inputs missing or labels not yet allowed). verdict = PASS only if every check
is PASS; FAIL if any check fails; PENDING otherwise. Tolerances: 1e-12 absolute for utility / AUC / CE / tables;
exact for selections, winners, statuses and plans; calibrator parameters by an own KKT certificate (1e-9 relative) and
|alpha_stored - alpha_own| <= 1e-9.

    ~/PCRL/.venv/bin/python -P <WT>/hcal/sema.py --label E:verify_inner -- env OMP_NUM_THREADS=1 \\
        PCRL_HCAL_PRIVATE_CACHE=$HOME/PCRL_eval_cache_private/hcal_v1 ~/PCRL/.venv/bin/python \\
        <WT>/results/pcrl_heldout_calibration_v1/verification/verify_inner.py
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
WT = HERE.parents[3]
REL = "results/pcrl_heldout_calibration_v1"
PKG = WT / REL
OUT = HERE.parent / "E_INNER_REPLAY.json"
BRANCH = "research/pcrl-heldout-calibration-v1"
PRIV = Path(os.environ.get("PCRL_HCAL_PRIVATE_CACHE") or (Path.home() / "PCRL_eval_cache_private" / "hcal_v1"))
ADM = PRIV / "admitted"
ADM_UNITS = ADM / "units"
RUN = PRIV / "run"
UNITS = RUN / "units"

TOL = 1e-12
TIE = 1e-12
CLIP = 1e-12
EPS = 1e-12
KAPPA = 32.0
TEMP_LO, TEMP_HI = 0.25, 4.0
CLASS_MIN = 50
KKT_REL_TOL = 1e-9
ALPHA_TOL = 1e-9
SEEDS = (0, 1, 2)
TASKS = ("income", "occupation")
LABEL_KEY = {"income": "y_income", "occupation": "y_occupation_group"}
KS = {1: 2, 2: 6}
VIEWS = ("v1", "v2", "pair")
CRITS = ("auc", "ce")
SALT_H = "hcal-v1|heldout|20261011|"
SALT_T = "hcal-v1|train-matched|20261011|"
N_CAL = 2000
GUARD, BENEFIT = 0.005, 0.02
ALLOW_ACC, ALLOW_LL, ALLOW_BR, RETAIN, MIN_GAIN = 0.01, 0.01, 0.005, 0.8, 0.03
FORBIDDEN_UNIT_PREFIXES = ("outer__", "oprob__", "oatt__")

# ------------------------------------------------------------------ registry (reconstructed from PROTOCOL.md section 3/4)
LAMS = (0.01, 0.025, 0.04, 0.06, 0.08, 0.1)
PFAMS = ("LOCAL", "SEQ-12", "SEQ-21", "JOINT")
KFAMS = ("LOCAL", "SEQ-12", "SEQ-21", "JOINT-SINGLE", "JOINT-PAIR")
LEGACY = (["U|DIRECT-TASK|i8o64", "U|FINE-TASK|i8o64", "U|CLASS|i1o1"]
          + [f"U|{f}|i8o64|l{lam:g}" for lam in LAMS for f in PFAMS])
LRA = (["U|C-TASK|i8o64"] + [f"U|W-{f}|i8o64|l{lam:g}" for lam in LAMS for f in PFAMS]
       + [f"U|K-{a}|i8o64" for a in KFAMS])
PARTS = LEGACY + LRA
DIAG = ("U|DIRECT-TASK|i8o64", "U|FINE-TASK|i8o64", "U|CLASS|i1o1", "U|JOINT|i8o64|l0.1")
TASK_ONLY = ("U|DIRECT-TASK|i8o64", "U|FINE-TASK|i8o64", "U|CLASS|i1o1", "U|C-TASK|i8o64")
NEW_DECS = ("H-TOKEN32", "H-GLOBAL-TEMP", "H-CLASS-TEMP")
U_ID = "SRC|U"
U_RIDS = [U_ID, f"{U_ID}|H-GLOBAL-TEMP", f"{U_ID}|H-CLASS-TEMP"]
U_FAM = {U_ID: "identity", f"{U_ID}|H-GLOBAL-TEMP": "H-GLOBAL-TEMP", f"{U_ID}|H-CLASS-TEMP": "H-CLASS-TEMP"}


def decoders(p):
    out = (["D0", "D1"] if p in LEGACY else ["D1", "MEAN"]) + list(NEW_DECS)
    if p in DIAG:
        out.append("T-TOKEN32")
    return out


def rid_of(p, d):
    return p if d == "D0" else f"{p}|{d}"


CODE_RIDS = [rid_of(p, d) for p in PARTS for d in decoders(p)]
ALL_RIDS = CODE_RIDS + U_RIDS
RID_INFO = {rid_of(p, d): (p, d) for p in PARTS for d in decoders(p)}
ORIGINAL_84 = LEGACY + [f"{p}|D1" for p in LEGACY] + [f"{p}|D1" for p in LRA]


def privacy_trained(p):
    f = p.split("|")[1]
    return f in PFAMS or f.startswith(("W-", "K-"))


def safe(x):
    return str(x).replace("|", "_").replace("*", "star").replace("/", "_").replace(" ", "_")


def lra_release_unit(k, rid):
    p, d = RID_INFO[rid]
    if d == "D0":
        return f"pol__s{k}__{safe(rid)}"
    return f"dec__s{k}__{safe(rid)}" if p in LEGACY else f"new__s{k}__{safe(rid)}"


def orig_rids_of(p):
    return [p, f"{p}|D1"] if p in LEGACY else [f"{p}|D1"]


# ------------------------------------------------------------------ bookkeeping
CHECKS = {}
LABEL_READS = []
INPUT_HASHES = {}


class Missing(Exception):
    pass


class Corrupt(Exception):
    pass


def pub(path):
    s = str(path)
    s = s.replace(str(PRIV), "<PRIVATE_CACHE>/hcal_v1").replace(str(WT), "<WT>").replace(str(Path.home()), "~")
    return s


def jsafe(o):
    if isinstance(o, dict):
        return {str(k): jsafe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsafe(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsafe(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if math.isfinite(v) else repr(v)
    if isinstance(o, Path):
        return pub(o)
    return o


class Check:
    """Accumulates one check: n compared items, max abs differences per quantity, failures (first 40 kept)."""

    def __init__(self, name, what):
        self.name, self.what = name, what
        self.n, self.maxdiff, self.fail, self.nfail, self.pending, self.notes = 0, {}, [], 0, [], []
        self.info = {}

    def diff(self, key, a, b, tol=TOL, where=""):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        self.n += 1
        if a.shape != b.shape:
            self.bad(f"{where} {key}: shape {a.shape} vs {b.shape}")
            return False
        if a.size == 0:
            return True
        fin = np.isfinite(a) & np.isfinite(b)
        if not np.array_equal(np.isfinite(a), np.isfinite(b)) or not np.array_equal(a[~fin], b[~fin]):
            self.bad(f"{where} {key}: non-finite pattern differs")
            return False
        d = float(np.max(np.abs(a[fin] - b[fin]))) if fin.any() else 0.0
        self.maxdiff[key] = max(self.maxdiff.get(key, 0.0), d)
        if d > tol:
            self.bad(f"{where} {key}: |diff| {d:.3e} > {tol:g}")
            return False
        return True

    def same(self, key, a, b, where=""):
        self.n += 1
        if a != b:
            self.bad(f"{where} {key}: {str(a)[:160]!s} != {str(b)[:160]!s}")
            return False
        return True

    def ok(self, cond, msg):
        self.n += 1
        if not cond:
            self.bad(msg)
        return bool(cond)

    def bad(self, msg):
        self.nfail += 1
        if len(self.fail) < 40:
            self.fail.append(msg)

    def pend(self, msg):
        if len(self.pending) < 40:
            self.pending.append(msg)

    def status(self):
        if self.nfail:
            return "FAIL"
        if self.pending:
            return "PENDING"
        if self.n == 0:
            return "PENDING"
        return "PASS"

    def done(self):
        CHECKS[self.name] = {"status": self.status(), "what": self.what, "compared": self.n,
                             "max_abs_diff": self.maxdiff, "failures": self.nfail, "first_failures": self.fail,
                             "pending": self.pending, "notes": self.notes, **self.info}
        return CHECKS[self.name]


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha_arr(a):
    """sha256(str(dtype) + str(shape) + bytes): the receipt convention of the role manifest and dpc.audit.sha_arrays."""
    a = np.ascontiguousarray(np.asarray(a))
    h = hashlib.sha256()
    h.update(str(a.dtype).encode() + str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def sha_arrays(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def unit_dir(name, root):
    if name.startswith(FORBIDDEN_UNIT_PREFIXES):
        raise PermissionError(f"REFUSED: role E never opens {name} before the EVALUATION_LOCK")
    return Path(root) / name


def load_unit(name, root=UNITS):
    """{file: path} of a hash-complete unit (every file listed in COMPLETE.json re-hashed here)."""
    d = unit_dir(name, root)
    c = d / "COMPLETE.json"
    if not c.exists():
        raise Missing(f"{name}: no COMPLETE.json")
    files = json.loads(c.read_text())["files"]
    out = {}
    for f, h in files.items():
        p = d / f
        if not p.exists():
            raise Corrupt(f"{name}/{f}: missing")
        got = sha_file(p)
        if got != h:
            raise Corrupt(f"{name}/{f}: sha256 differs from COMPLETE.json")
        out[f] = p
        INPUT_HASHES[pub(p)] = got
    return out


_REC_CACHE = {}


def unit_record(name, root=UNITS):
    key = (str(root), name)
    if key not in _REC_CACHE:
        f = load_unit(name, root)
        _REC_CACHE[key] = (json.loads(f["record.json"].read_text()), f)
    return _REC_CACHE[key]


def npz(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def read_json(p):
    if not p.exists():
        raise Missing(f"{pub(p)} missing")
    INPUT_HASHES[pub(p)] = sha_file(p)
    return json.loads(p.read_text())


# ------------------------------------------------------------------ locks / label gate
def git(*a, text=True):
    return subprocess.run(["git", "-C", str(WT), *a], capture_output=True, text=text)


def on_origin(rel):
    p = WT / rel
    if not p.exists():
        return False, f"{rel} does not exist"
    git("fetch", "-q", "origin", BRANCH)
    r = git("show", f"origin/{BRANCH}:{rel}", text=False)
    if r.returncode != 0:
        return False, f"{rel} is not on origin/{BRANCH}"
    if r.stdout != p.read_bytes():
        return False, f"local {rel} differs from origin"
    return True, f"{rel} byte-identical on origin"


# ------------------------------------------------------------------ own numerics
def avg_ranks(x):
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    b = np.flatnonzero(np.diff(xs) != 0) + 1
    starts = np.r_[0, b]
    ends = np.r_[b, n]
    r = np.empty(n)
    r[order] = np.repeat((starts + ends + 1) / 2.0, ends - starts)
    return r


def auc_fixed(y, s):
    """Mann-Whitney AUC of the score s for the positive class y == 1 (never flipped, never clipped)."""
    pos = np.asarray(y) == 1
    s = np.asarray(s, dtype=np.float64)
    if not np.all(np.isfinite(s)):
        return float("nan")
    n1 = int(pos.sum())
    n0 = pos.size - n1
    r = avg_ranks(s)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def ce_fixed(y, p):
    p = np.asarray(p, dtype=np.float64)
    pt = np.where(np.asarray(y) == 1, p, 1.0 - p)
    return float(-np.mean(np.log(np.clip(pt, CLIP, 1.0))))


def mean3(v):
    return (float(v[0]) + float(v[1]) + float(v[2])) / 3.0


def r12(x):
    return round(float(x), 12)


def smooth(u, d, K):
    e = np.zeros(K)
    e[d] = EPS
    return (u + EPS + e) / (1.0 + (K + 1) * EPS)


def softmax_rows(Z):
    E = np.exp(Z - Z.max(1, keepdims=True))
    return E / E.sum(1, keepdims=True)


def task_metrics(P, hard, y, K, const):
    P = np.asarray(P, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    n = y.size
    ll = -np.log(np.clip(P[np.arange(n), y], CLIP, 1.0))
    oh = np.zeros((n, K))
    oh[np.arange(n), y] = 1.0
    br = ((P - oh) ** 2).sum(1)
    acc = float(np.mean(np.asarray(hard) == y))
    cacc = float(np.mean(y == const))
    return {"acc": acc, "logloss": float(ll.mean()), "brier": float(br.mean()), "const_acc": cacc,
            "gain": acc - cacc, "n": int(n)}


# ---- temperature (own convex 1-D solve of the UNCLIPPED mean NLL in alpha)
def temp_terms(L, y, a):
    Z = a * L
    m = Z.max(1)
    E = np.exp(Z - m[:, None])
    s = E.sum(1)
    P = E / s[:, None]
    rows = np.arange(L.shape[0])
    EL = (P * L).sum(1)
    nll = float(np.mean(m + np.log(s) - Z[rows, y]))
    g = float(np.mean(EL - L[rows, y]))
    c = float(np.mean((P * (L - EL[:, None]) ** 2).sum(1)))
    return nll, g, c


def solve_temp(L, y):
    """Safeguarded Newton on the derivative of the convex NLL over [0.25, 4]; boundary by KKT sign."""
    g_lo = temp_terms(L, y, TEMP_LO)[1]
    g_hi = temp_terms(L, y, TEMP_HI)[1]
    if g_lo >= 0:
        return TEMP_LO, "BOUNDARY_LOW", g_lo, g_hi
    if g_hi <= 0:
        return TEMP_HI, "BOUNDARY_HIGH", g_lo, g_hi
    lo, hi, a = TEMP_LO, TEMP_HI, 1.0
    for _ in range(300):
        _, g, c = temp_terms(L, y, a)
        if g > 0:
            hi = a
        else:
            lo = a
        nxt = a - g / c if c > 0 else 0.5 * (lo + hi)
        if not (lo < nxt < hi):
            nxt = 0.5 * (lo + hi)
        if abs(nxt - a) <= 4e-16 * max(1.0, abs(a)) or hi - lo <= 4e-16 * hi:
            a = nxt
            break
        a = nxt
    return a, "INTERIOR", g_lo, g_hi


def certify_alpha(L, y, alpha, status, chk, where):
    """Stored alpha: own KKT certificate + agreement with the own solve. Returns the own alpha."""
    scale = float(np.mean(np.abs(L).max(1)))
    a_own, st_own, g_lo, g_hi = solve_temp(L, y)
    _, g, c = temp_terms(L, y, alpha)
    chk.n += 1
    if status == "INTERIOR":
        ok = TEMP_LO < alpha < TEMP_HI and abs(g) <= KKT_REL_TOL * scale
    elif status == "BOUNDARY_LOW":
        ok = alpha == TEMP_LO and g >= -KKT_REL_TOL * scale
    elif status == "BOUNDARY_HIGH":
        ok = alpha == TEMP_HI and g <= KKT_REL_TOL * scale
    else:
        ok = False
    if not ok:
        chk.bad(f"{where}: stored alpha {alpha!r} ({status}) fails the own KKT check (g = {g:.3e}, scale {scale:.3e})")
    chk.info.setdefault("max_rel_grad_interior", 0.0)
    if status == "INTERIOR":
        chk.info["max_rel_grad_interior"] = max(chk.info["max_rel_grad_interior"], abs(g) / scale)
    chk.ok(st_own == status or (min(abs(g_lo), abs(g_hi)) <= KKT_REL_TOL * scale),
           f"{where}: own status {st_own} != stored {status}")
    chk.diff("alpha", a_own, alpha, ALPHA_TOL, where)
    chk.ok(c >= 0, f"{where}: negative curvature")
    return a_own


def temp_table(q0, alpha):
    if alpha == 1.0:
        return q0.copy()
    return softmax_rows(alpha * np.log(q0))


# ---- token32 KKT (Frank-Wolfe vertex gap on the class-dominant simplex)
def token32_kkt(q, y, mu, d, K):
    """(gap / (n + 32), feasibility violation, strict margin) for one fitted token. q is the released smoothed
    vector, u = q Z - eps - eps e_d its simplex point. Vertices of {u >= 0, sum u = 1, u_d >= u_k} are the uniform
    vectors on sets S containing d, so min_v G.v = min_m mean(G_d, m smallest other G_k)."""
    Z = 1.0 + (K + 1) * EPS
    e = np.zeros(K)
    e[d] = 1.0
    u = q * Z - EPS - EPS * e
    n = float(y.sum())
    A = y + KAPPA * mu
    G = (-A / q + n * q - y) / Z
    others = np.sort(np.delete(G, d))
    best, cs = G[d], G[d]
    for m, gk in enumerate(others, start=1):
        cs += gk
        best = min(best, cs / (m + 1))
    gap = float(G @ u - best)
    feas = max(abs(float(u.sum()) - 1.0), max(0.0, -float(u.min())),
               max(0.0, float(np.max(np.delete(u, d)) - u[d])))
    margin = float(q[d] - np.max(np.delete(q, d)))
    return gap / (n + KAPPA), feas, margin


def token32_objective(q, y, mu):
    n = float(y.sum())
    ll = -float(y @ np.log(q))
    br = 0.5 * float(n * (q @ q) - 2.0 * (y @ q) + n)
    m = mu > 0
    kl = float(np.sum(mu[m] * (np.log(mu[m]) - np.log(q[m]))))
    return ll + br + KAPPA * kl


# ------------------------------------------------------------------ (1) role split
def split(D, parent, salt):
    rid = np.asarray(D["row_id"], dtype=np.int64)
    grp = np.asarray(D["unit"], dtype=np.int64)
    pos = np.asarray(D["idx"][parent], dtype=np.int64)
    uniq = np.unique(grp[pos])
    keyed = sorted((hashlib.sha256((salt + str(int(u))).encode("utf-8")).hexdigest(), int(u)) for u in uniq)
    order = np.asarray([u for _, u in keyed], dtype=np.int64)
    chosen, rest = order[:N_CAL], order[N_CAL:]
    in_ch = np.isin(grp, chosen)
    members = np.flatnonzero(in_ch)
    reps = []
    for u in chosen:
        m = members[grp[members] == u]
        reps.append(int(m[np.argmin(rid[m])]))
    rest_rows = np.flatnonzero(np.isin(grp, rest))
    return {"chosen": chosen, "reps_rank_order": np.asarray(reps, dtype=np.int64),
            "reps": np.sort(np.asarray(reps, dtype=np.int64)), "members": members, "rest_rows": rest_rows,
            "parent_rows": pos, "parent_groups": int(uniq.size)}


def check_roles(D):
    chk = Check("1_ROLE_SPLIT", "hash-ranked calibration split from D['unit'] / D['row_id'] (no labels)")
    rid = np.asarray(D["row_id"], dtype=np.int64)
    grp = np.asarray(D["unit"], dtype=np.int64)
    idx = {r: np.asarray(v, dtype=np.int64) for r, v in D["idx"].items()}
    h = split(D, "AUDIT_FIT", SALT_H)
    t = split(D, "OSF_DEFENSE_FIT", SALT_T)
    roles = {"CALIBRATION_HELDOUT": h["reps"], "CALIBRATION_HELDOUT_GROUP_ROWS": h["members"],
             "ATTACK_FIT_NEW": h["rest_rows"], "CALIBRATION_TRAIN_MATCHED": t["reps"],
             "CALIBRATION_TRAIN_MATCHED_GROUP_ROWS": t["members"]}
    ng = lambda pos: int(np.unique(grp[pos]).size)  # noqa: E731
    want = {"CALIBRATION_HELDOUT": (2000, 2000), "CALIBRATION_HELDOUT_GROUP_ROWS": (None, 2000),
            "ATTACK_FIT_NEW": (None, 4061), "CALIBRATION_TRAIN_MATCHED": (2000, 2000),
            "CALIBRATION_TRAIN_MATCHED_GROUP_ROWS": (None, 2000)}
    counts = {}
    for r, pos in roles.items():
        counts[r] = {"rows": int(pos.size), "groups": ng(pos)}
        wr, wg = want[r]
        if wr is not None:
            chk.same(f"{r}.rows", int(pos.size), wr)
        chk.same(f"{r}.groups", ng(pos), wg)
    chk.ok(np.all(np.isin(h["members"], idx["AUDIT_FIT"])), "a selected AUDIT_FIT group has rows outside AUDIT_FIT")
    chk.ok(np.all(np.isin(t["members"], idx["OSF_DEFENSE_FIT"])), "a selected DEFENSE group has rows outside it")
    chk.ok(np.all(np.isin(h["rest_rows"], idx["AUDIT_FIT"])), "ATTACK_FIT_NEW rows outside AUDIT_FIT")
    chk.ok(np.array_equal(np.sort(np.r_[h["members"], h["rest_rows"]]), np.sort(idx["AUDIT_FIT"])),
           "CALIBRATION_HELDOUT group rows + ATTACK_FIT_NEW != AUDIT_FIT")
    for nm, s in (("H", h), ("T", t)):
        for u, r_ in zip(s["chosen"], s["reps_rank_order"]):
            pass
        # representative = smallest row id of its group (vectorised re-check)
        g_of = grp[s["reps"]]
        mins = {}
        for p_ in s["members"]:
            u = int(grp[p_])
            mins[u] = min(mins.get(u, (np.inf, -1)), (rid[p_], p_))
        chk.ok(all(mins[int(u)][1] == int(r_) for u, r_ in zip(g_of, s["reps"])), f"{nm}: a representative is not "
               "its group's smallest row id")
    cal, att, tm = h["members"], h["rest_rows"], t["members"]
    pairs = {"CALIBRATION_HELDOUT vs OSF_DEFENSE_FIT": (cal, idx["OSF_DEFENSE_FIT"]),
             "CALIBRATION_HELDOUT vs HEAD_VALIDATION": (cal, idx["HEAD_VALIDATION"]),
             "CALIBRATION_HELDOUT vs ATTACK_FIT_NEW": (cal, att),
             "CALIBRATION_HELDOUT vs INNER_SELECTION": (cal, idx["INNER_SELECTION"]),
             "CALIBRATION_HELDOUT vs OSF_DEVELOPMENT_ASSESSMENT": (cal, idx["OSF_DEVELOPMENT_ASSESSMENT"]),
             "ATTACK_FIT_NEW vs INNER_SELECTION": (att, idx["INNER_SELECTION"]),
             "ATTACK_FIT_NEW vs OSF_DEVELOPMENT_ASSESSMENT": (att, idx["OSF_DEVELOPMENT_ASSESSMENT"]),
             "ATTACK_FIT_NEW vs OSF_DEFENSE_FIT": (att, idx["OSF_DEFENSE_FIT"]),
             "ATTACK_FIT_NEW vs HEAD_VALIDATION": (att, idx["HEAD_VALIDATION"]),
             "CALIBRATION_TRAIN_MATCHED vs INNER_SELECTION": (tm, idx["INNER_SELECTION"]),
             "CALIBRATION_TRAIN_MATCHED vs OSF_DEVELOPMENT_ASSESSMENT": (tm, idx["OSF_DEVELOPMENT_ASSESSMENT"]),
             "CALIBRATION_TRAIN_MATCHED vs AUDIT_FIT": (tm, idx["AUDIT_FIT"]),
             "CALIBRATION_TRAIN_MATCHED vs HEAD_VALIDATION": (tm, idx["HEAD_VALIDATION"])}
    dj = {}
    for nm, (a, b) in pairs.items():
        sr = int(np.intersect1d(a, b).size)
        sg = int(np.intersect1d(np.unique(grp[a]), np.unique(grp[b])).size)
        dj[nm] = {"shared_rows": sr, "shared_groups": sg}
        chk.ok(sr == 0 and sg == 0, f"{nm}: {sr} shared rows, {sg} shared groups")
    # ROLE_MANIFEST.json receipts (committed by role A)
    try:
        man = read_json(PKG / "ROLE_MANIFEST.json")
        for r, pos in roles.items():
            m = man["roles"][r]
            chk.same(f"manifest {r}.rows", m["rows"], int(pos.size))
            chk.same(f"manifest {r}.groups", m["groups"], ng(pos))
            chk.same(f"manifest {r}.row_id_sha256", m["row_id_sha256"], sha_arr(np.sort(rid[pos])))
            chk.same(f"manifest {r}.group_id_sha256", m["group_id_sha256"], sha_arr(np.unique(grp[pos])))
        for r, s in (("CALIBRATION_HELDOUT", h), ("CALIBRATION_TRAIN_MATCHED", t)):
            chk.same(f"manifest {r} rank order", man["roles"][r]["selected_groups_in_rank_order_sha256"],
                     sha_arr(s["chosen"]))
            chk.same(f"manifest {r} reps rank order", man["roles"][r]["representatives_in_rank_order_sha256"],
                     sha_arr(rid[s["reps_rank_order"]]))
        chk.same("manifest parent groups", man["parent_groups"],
                 {"AUDIT_FIT": h["parent_groups"], "OSF_DEFENSE_FIT": t["parent_groups"]})
    except Missing as e:
        chk.pend(str(e))
    chk.info.update({"counts": counts, "disjointness": dj,
                     "parent_groups": {"AUDIT_FIT": h["parent_groups"], "OSF_DEFENSE_FIT": t["parent_groups"]}})
    chk.done()
    return roles


# ------------------------------------------------------------------ (0) frozen bank
def bank_hashes():
    r = read_json(ADM / "ADMISSION_RECEIPT.json")
    if r.get("verdict") != "ADMITTED":
        raise Corrupt("admission receipt is not ADMITTED")
    return {key: v["bank_sha256"] for key, v in r["bank"].items()}, list(r["bank"])


_BANK = {}
_TEACH = {}


def bank(k, p):
    if (k, p) not in _BANK:
        path = ADM / "bank" / f"bank__s{k}__{safe(p)}.npz"
        if not path.exists():
            raise Missing(f"{pub(path)} missing")
        _BANK[(k, p)] = (npz(path), sha_file(path))
        INPUT_HASHES[pub(path)] = _BANK[(k, p)][1]
    return _BANK[(k, p)][0]


def teacher(k):
    if k not in _TEACH:
        f = load_unit(f"tea__s{k}__U", ADM_UNITS)
        _TEACH[k] = npz(f["teacher.npz"])
    return _TEACH[k]


def check_bank(D):
    chk = Check("0_FROZEN_BANK", "admitted frozen-bank npz files: hashes and internal consistency (no labels)")
    try:
        want, order = bank_hashes()
    except (Missing, Corrupt) as e:
        chk.pend(str(e))
        chk.done()
        return
    chk.same("receipt partition order", [x.split("|", 1)[1] for x in order[:len(PARTS)]], PARTS)
    fit = np.asarray(D["idx"]["OSF_DEFENSE_FIT"], dtype=np.int64)
    for k in SEEDS:
        T = teacher(k)
        chk.ok(np.array_equal(T["row_id"], D["row_id"]), f"s{k}: teacher rows not in D order")
        for p in PARTS:
            b = bank(k, p)
            where = f"s{k} {p}"
            chk.same("bank sha256", _BANK[(k, p)][1], want.get(f"{k}|{p}"), where)
            chk.ok(np.array_equal(b["row_id"], D["row_id"]), f"{where}: rows not in D order")
            for i in (1, 2):
                K = KS[i]
                tok, hard, cls = b[f"tok{i}"], b[f"hard{i}"], b[f"class{i}"]
                n, S, mu, q0 = b[f"n_fit{i}"], b[f"S{i}"], b[f"mu{i}"], b[f"q0{i}"]
                Tn = int(b[f"alpha{i}"])
                chk.ok(cls.size == Tn and tok.min() >= 0 and tok.max() < Tn, f"{where} r{i}: alphabet")
                chk.ok(np.array_equal(cls[tok], hard), f"{where} r{i}: decision != token class")
                chk.ok(np.array_equal(hard, T[f"d{i}"]), f"{where} r{i}: decision != teacher decision")
                chk.ok(np.array_equal(T[f"d{i}"], np.argmax(T[f"p{i}"], 1)), f"{where} r{i}: teacher d != argmax p")
                chk.ok(np.array_equal(np.bincount(tok[fit], minlength=Tn), n), f"{where} r{i}: token_n")
                Sr = np.stack([np.bincount(tok[fit], weights=T[f"p{i}"][fit, c], minlength=Tn) for c in range(K)], 1)
                chk.ok(bool(np.all(np.abs(Sr - S) <= 1e-9 * np.maximum(n, 1)[:, None])), f"{where} r{i}: token_S")
                res = n == 0
                chk.ok(bool(np.all(np.isnan(mu[res]))) and bool(np.all(np.isfinite(mu[~res]))), f"{where} r{i}: mu NaN")
                with np.errstate(invalid="ignore", divide="ignore"):
                    chk.diff("mu=S/n", mu[~res], S[~res] / n[~res][:, None], 1e-15, where)
                sm = np.stack([smooth(mu[j] if not res[j] else np.full(K, 1.0 / K), cls[j], K) for j in range(Tn)])
                chk.diff("q0=smooth(mu)", q0, sm, 1e-15, where)
                oth = q0.copy()
                oth[np.arange(Tn), cls] = -np.inf
                chk.ok(bool(np.all(q0[np.arange(Tn), cls] > oth.max(1))), f"{where} r{i}: q0 not strictly class-led")
                oth = b[f"qD1{i}"].copy()
                oth[np.arange(Tn), cls] = -np.inf
                chk.ok(bool(np.all(b[f"qD1{i}"][np.arange(Tn), cls] > oth.max(1))), f"{where} r{i}: D1 not class-led")
    chk.done()


# ------------------------------------------------------------------ labels (after the pushed SCIENCE_LOCK only)
class Labels:
    def __init__(self, D, roles):
        self.D, self.roles = D, roles
        self.assess = np.asarray(D["idx"]["OSF_DEVELOPMENT_ASSESSMENT"], dtype=np.int64)
        self.head = np.asarray(D["idx"]["HEAD_VALIDATION"], dtype=np.int64)
        self._cache = {}

    def _rows(self, role):
        if role in self.roles:
            return np.asarray(self.roles[role], dtype=np.int64)
        return np.asarray(self.D["idx"][role], dtype=np.int64)

    def get(self, key, role, purpose):
        if (key, role) in self._cache:
            return self._cache[(key, role)]
        allowed = {("task", "CALIBRATION_HELDOUT"), ("task", "CALIBRATION_TRAIN_MATCHED"), ("task", "INNER_SELECTION"),
                   ("task", "OSF_DEFENSE_FIT"), ("sex", "INNER_SELECTION")}
        kind = "sex" if key == "sex" else "task"
        if (kind, role) not in allowed:
            raise PermissionError(f"REFUSED: role E does not read {key} of {role}")
        rows = self._rows(role)
        if np.intersect1d(rows, self.assess).size or np.intersect1d(rows, self.head).size:
            raise PermissionError(f"REFUSED: {role} rows touch the assessment or HEAD_VALIDATION")
        y = np.asarray(self.D[key], dtype=np.int64)[rows]
        if np.any(y < 0):
            raise PermissionError(f"REFUSED: sealed labels on {role}")
        LABEL_READS.append({"label": key, "role": role, "rows": int(rows.size), "purpose": purpose,
                            "row_id_sha256": sha_arr(np.sort(np.asarray(self.D["row_id"])[rows]))})
        self._cache[(key, role)] = (rows, y)
        return rows, y

    def task(self, task, role, purpose):
        return self.get(LABEL_KEY[task], role, purpose)


# ------------------------------------------------------------------ (2) calibration
def cal_tables(k, p):
    rec, f = unit_record(f"cal__s{k}__{safe(p)}")
    return rec, npz(f["tables.npz"]), json.loads(f["tables.json"].read_text())


def calu(k):
    rec, f = unit_record(f"calU__s{k}")
    return rec, npz(f["u.npz"])


TABLES = {}       # (k, p) -> {decoder: (T1, T2)} verified (own temperature tables; stored token32 tables)
STORED_TABLES = {}  # (k, p) -> {decoder: (T1, T2)} exactly as stored (bank D0 / D1 / MEAN; cal__ tables.npz)
UQ = {}           # (k, fam) -> {1: q rows, 2: q rows}


def check_calibration(D, roles, LB):
    chk = Check("2_CALIBRATION", "H-TOKEN32 / T-TOKEN32 KKT, H-GLOBAL-TEMP / H-CLASS-TEMP own solves, U temperatures")
    h_rows, t_rows = roles["CALIBRATION_HELDOUT"], roles["CALIBRATION_TRAIN_MATCHED"]
    chk.info.update({"token32_fitted": 0, "token32_fallback": 0, "max_token32_gap_rel": 0.0,
                     "max_token32_feasibility": 0.0, "min_token32_margin": None, "temperature_solves": 0,
                     "temperature_boundaries": 0, "class_fallbacks": 0})
    yH = yT = None
    if LB is not None:
        yH = {t: LB.task(t, "CALIBRATION_HELDOUT", "calibration replay")[1] for t in TASKS}
        yT = {t: LB.task(t, "CALIBRATION_TRAIN_MATCHED", "T-TOKEN32 replay")[1] for t in TASKS}
        chk.ok(np.array_equal(LB.task("income", "CALIBRATION_HELDOUT", "x")[0], h_rows), "label rows != own reps")
    missing = 0
    for k in SEEDS:
        for p in PARTS:
            b = bank(k, p)
            where = f"s{k} {p}"
            try:
                rec, tz, tj = cal_tables(k, p)
            except Missing:
                missing += 1
                continue
            except Corrupt as e:
                chk.bad(str(e))
                continue
            fams = [d for d in decoders(p) if d in NEW_DECS or d == "T-TOKEN32"]
            chk.same("families", list(rec.get("families", [])), fams, where)
            tabs = {}
            for d in decoders(p):
                if d in ("D0", "MEAN"):
                    tabs[d] = (b["q01"], b["q02"])
                elif d == "D1":
                    tabs[d] = (b["qD11"], b["qD12"])
            for fam in fams:
                rows = t_rows if fam == "T-TOKEN32" else h_rows
                ys = (yT if fam == "T-TOKEN32" else yH)
                pair = []
                for i in (1, 2):
                    K = KS[i]
                    w = f"{where} {fam} r{i}"
                    Qs = np.asarray(tz[f"{fam}|q{i}"], dtype=np.float64)
                    rj = tj[f"{fam}|r{i}"]
                    q0, cls, n_fit, mu = b[f"q0{i}"], b[f"class{i}"], b[f"n_fit{i}"], b[f"mu{i}"]
                    Tn = q0.shape[0]
                    chk.ok(Qs.shape == q0.shape, f"{w}: table shape")
                    chk.diff("npz==json q", Qs, np.asarray(rj["q"], dtype=np.float64), 0.0, w)
                    tok = b[f"tok{i}"][rows]
                    n_cal = np.bincount(tok, minlength=Tn)
                    chk.same("n_cal", list(map(int, rj["n_cal"])), n_cal.tolist(), w)
                    chk.diff("row sums", Qs.sum(1), np.ones(Tn), 1e-12, w)
                    oth = Qs.copy()
                    oth[np.arange(Tn), cls] = -np.inf
                    chk.ok(bool(np.all(Qs[np.arange(Tn), cls] > oth.max(1))), f"{w}: strict argmax != token class")
                    if fam in ("H-TOKEN32", "T-TOKEN32"):
                        st = ["RESERVED_EMPTY_ORIGINAL_FALLBACK" if n_fit[t] == 0 else
                              ("NO_CALIBRATION_OBSERVATIONS" if n_cal[t] == 0 else "FITTED") for t in range(Tn)]
                        chk.same("token statuses", list(rj["status"]), st, w)
                        fb = np.asarray([s != "FITTED" for s in st])
                        chk.ok(np.array_equal(Qs[fb], q0[fb]), f"{w}: a fallback token is not q0 exactly")
                        chk.info["token32_fallback"] += int(fb.sum())
                        chk.same("parameter_count", rj.get("parameter_count"), int((~fb).sum()) * (K - 1), w)
                        if ys is None:
                            chk.pend("token32 KKT needs calibration labels (SCIENCE_LOCK not pushed)")
                            pair.append(Qs)
                            continue
                        y = ys[TASKS[i - 1]]
                        ycnt = np.bincount(tok * K + y, minlength=Tn * K).reshape(Tn, K)
                        chk.same("y_cal", np.asarray(rj["y_cal"]).astype(int).tolist(), ycnt.tolist(), w)
                        for t in np.flatnonzero(~fb):
                            gap, feas, marg = token32_kkt(Qs[t], ycnt[t].astype(float), mu[t], int(cls[t]), K)
                            chk.info["token32_fitted"] += 1
                            chk.info["max_token32_gap_rel"] = max(chk.info["max_token32_gap_rel"], gap)
                            chk.info["max_token32_feasibility"] = max(chk.info["max_token32_feasibility"], feas)
                            mm = chk.info["min_token32_margin"]
                            chk.info["min_token32_margin"] = marg if mm is None else min(mm, marg)
                            chk.ok(gap <= KKT_REL_TOL and feas <= 1e-12 and marg > 0,
                                   f"{w} token {t}: KKT gap_rel {gap:.3e}, feasibility {feas:.3e}, margin {marg:.3e}")
                            # objective not worse than q0 (the frozen original vector is feasible)
                            chk.ok(token32_objective(Qs[t], ycnt[t].astype(float), mu[t])
                                   <= token32_objective(q0[t], ycnt[t].astype(float), mu[t]) + 1e-9 * (n_cal[t] + KAPPA),
                                   f"{w} token {t}: objective above the q0 objective")
                        pair.append(Qs)
                    else:
                        if ys is None:
                            chk.pend("temperature replay needs calibration labels (SCIENCE_LOCK not pushed)")
                            pair.append(Qs)
                            continue
                        y = ys[TASKS[i - 1]]
                        L = np.log(q0[tok])
                        if fam == "H-GLOBAL-TEMP":
                            alpha = float(rj["alpha"])
                            stt = rj["certificate"]["status"]
                            chk.info["temperature_solves"] += 1
                            chk.info["temperature_boundaries"] += int(stt != "INTERIOR")
                            certify_alpha(L, y, alpha, stt, chk, w)
                            Qm = temp_table(q0, alpha)
                            if alpha == 1.0:
                                chk.ok(np.array_equal(Qs, q0), f"{w}: alpha = 1 but table != q0 bitwise")
                        else:
                            alphas = [float(a) for a in rj["alphas"]]
                            recs = rj["certificate"]["classes"]
                            Qm = q0.copy()
                            ctok = cls[tok]
                            for c in range(K):
                                m = ctok == c
                                nc = int(m.sum())
                                if nc < CLASS_MIN:
                                    chk.info["class_fallbacks"] += 1
                                    chk.same("class fallback alpha", alphas[c], 1.0, f"{w} class {c} (n={nc})")
                                    chk.same("class fallback status", recs[c].get("status"), "CLASS_FALLBACK_LT50",
                                             f"{w} class {c}")
                                    continue
                                stt = recs[c].get("solve_status")
                                chk.info["temperature_solves"] += 1
                                chk.info["temperature_boundaries"] += int(stt != "INTERIOR")
                                certify_alpha(L[m], y[m], alphas[c], stt, chk, f"{w} class {c}")
                                if alphas[c] != 1.0 and (cls == c).any():
                                    Qm[cls == c] = softmax_rows(alphas[c] * np.log(q0[cls == c]))
                            chk.same("parameter_count", rj.get("parameter_count"),
                                     sum(1 for c in range(K) if int((ctok == c).sum()) >= CLASS_MIN), w)
                        chk.diff("temperature table", Qs, Qm, TOL, w)
                        pair.append(Qm)
                tabs[fam] = tuple(pair)
            TABLES[(k, p)] = {d: tabs[d] for d in decoders(p) if d in tabs}
            STORED_TABLES[(k, p)] = {d: (tabs[d] if d in ("D0", "D1", "MEAN") else
                                         (np.asarray(tz[f"{d}|q1"], dtype=np.float64),
                                          np.asarray(tz[f"{d}|q2"], dtype=np.float64))) for d in decoders(p)}
    if missing:
        chk.pend(f"{missing} of {len(PARTS) * 3} cal__ units missing")
    # continuous U
    for k in SEEDS:
        T = teacher(k)
        try:
            rec, uz = calu(k)
        except Missing:
            chk.pend(f"calU__s{k} missing")
            continue
        except Corrupt as e:
            chk.bad(str(e))
            continue
        for fam in ("H-GLOBAL-TEMP", "H-CLASS-TEMP"):
            UQ[(k, fam)] = {}
            for i in (1, 2):
                K = KS[i]
                w = f"s{k} U {fam} r{i}"
                P = np.asarray(T[f"p{i}"], dtype=np.float64)
                d = np.asarray(T[f"d{i}"], dtype=np.int64)
                Qs = np.asarray(uz[f"{fam}|p{i}"], dtype=np.float64)
                rj = rec["tables"][f"{fam}|r{i}"]
                chk.ok(np.array_equal(Qs.argmax(1), d), f"{w}: argmax != teacher decision")
                Pp = np.maximum(P, 1e-12)
                Pp = Pp / Pp.sum(1, keepdims=True)
                LP = np.log(Pp)
                if LB is None:
                    chk.pend("U temperature replay needs calibration labels (SCIENCE_LOCK not pushed)")
                    UQ[(k, fam)][i] = Qs
                    continue
                y = yH[TASKS[i - 1]]
                Lc = LP[h_rows]
                if fam == "H-GLOBAL-TEMP":
                    alpha = float(rj["alpha"])
                    certify_alpha(Lc, y, alpha, rj["certificate"]["status"], chk, w)
                    chk.info["temperature_solves"] += 1
                    Qm = P.copy() if alpha == 1.0 else softmax_rows(alpha * LP)
                else:
                    alphas = [float(a) for a in rj["alphas"]]
                    recs = rj["certificate"]["classes"]
                    Qm = P.copy()
                    dc = d[h_rows]
                    for c in range(K):
                        m = dc == c
                        if int(m.sum()) < CLASS_MIN:
                            chk.info["class_fallbacks"] += 1
                            chk.same("U class fallback alpha", alphas[c], 1.0, f"{w} class {c}")
                            continue
                        chk.info["temperature_solves"] += 1
                        certify_alpha(Lc[m], y[m], alphas[c], recs[c].get("solve_status"), chk, f"{w} class {c}")
                        if alphas[c] != 1.0:
                            Qm[d == c] = softmax_rows(alphas[c] * LP[d == c])
                chk.diff("U temperature rows", Qs, Qm, TOL, w)
                UQ[(k, fam)][i] = Qm
    chk.done()


# ------------------------------------------------------------------ (3) utility
def release_rows(k, rid):
    """(q {1,2} over all rows, hard {1,2}) of a release from the verified tables (code) or U arrays."""
    T = teacher(k)
    if rid in U_RIDS:
        fam = U_FAM[rid]
        hard = {i: np.asarray(T[f"d{i}"], dtype=np.int64) for i in (1, 2)}
        if fam == "identity":
            return {i: np.asarray(T[f"p{i}"], dtype=np.float64) for i in (1, 2)}, hard
        if (k, fam) not in UQ or len(UQ[(k, fam)]) < 2:
            raise Missing(f"U {fam} s{k} not available")
        return UQ[(k, fam)], hard
    p, d = RID_INFO[rid]
    if (k, p) not in TABLES or d not in TABLES[(k, p)]:
        raise Missing(f"tables of {p} s{k} not available")
    b = bank(k, p)
    q = {i: np.asarray(TABLES[(k, p)][d][i - 1], dtype=np.float64)[b[f"tok{i}"]] for i in (1, 2)}
    return q, {i: np.asarray(b[f"hard{i}"], dtype=np.int64) for i in (1, 2)}


def gate_u0(m, u):
    out = {}
    for t in TASKS:
        c, uu = m[t], u[t]
        ug = uu["acc"] - uu["const_acc"]
        g = {"acc": c["acc"] - (uu["acc"] - ALLOW_ACC), "ll": (uu["logloss"] + ALLOW_LL) - c["logloss"],
             "brier": (uu["brier"] + ALLOW_BR) - c["brier"], "ret": c["gain"] - RETAIN * ug,
             "gain": c["gain"] - MIN_GAIN}
        out[t] = g
    return out


def gate_ucal(m, uc):
    return {t: {"ll": (uc[t]["logloss"] + ALLOW_LL) - m[t]["logloss"],
                "brier": (uc[t]["brier"] + ALLOW_BR) - m[t]["brier"]} for t in TASKS}


UTIL = {}       # (k, rid) -> {"inner": {task: m}, "preserved": {1, 2}, "finite": bool}
TABLE = {}      # rid -> own utility table row
UCAL = {}
PLAN = {}


def check_utility(D, roles, LB):
    chk = Check("3_UTILITY", "inner utility of every release and seed, gates, Ucal*, utility table, audit plan")
    if LB is None:
        chk.pend("utility replay needs INNER_SELECTION task labels (SCIENCE_LOCK not pushed)")
        chk.done()
        return
    inner = np.asarray(D["idx"]["INNER_SELECTION"], dtype=np.int64)
    yI = {t: LB.task(t, "INNER_SELECTION", "inner utility replay")[1] for t in TASKS}
    yC = {t: LB.task(t, "CALIBRATION_HELDOUT", "calibration-row utility replay")[1] for t in TASKS}
    yM = {t: LB.task(t, "CALIBRATION_TRAIN_MATCHED", "train-matched-row utility replay")[1] for t in TASKS}
    const = {}
    for j, t in enumerate(TASKS):
        _, yf = LB.task(t, "OSF_DEFENSE_FIT", "constant class only")
        const[t] = int(np.argmax(np.bincount(yf, minlength=KS[j + 1])))
    chk.info["constant_class"] = const
    stored = {}
    for k in SEEDS:
        try:
            rec, _ = unit_record(f"util__s{k}")
            stored[k] = rec["releases"]
        except Missing as e:
            chk.pend(str(e))
        except Corrupt as e:
            chk.bad(str(e))
    for k in SEEDS:
        T = teacher(k)
        for rid in ALL_RIDS:
            try:
                q, hard = release_rows(k, rid)
            except Missing as e:
                chk.pend(str(e))
                continue
            fin = all(bool(np.all(np.isfinite(q[i]))) for i in (1, 2))
            pres = {i: bool(np.array_equal(hard[i], T[f"d{i}"]) and np.array_equal(q[i].argmax(1), hard[i]))
                    for i in (1, 2)}
            m = {t: task_metrics(q[i][inner], hard[i][inner], yI[t], KS[i], const[t]) for i, t in ((1, "income"),
                                                                                                    (2, "occupation"))}
            mc = {t: task_metrics(q[i][roles["CALIBRATION_HELDOUT"]], hard[i][roles["CALIBRATION_HELDOUT"]], yC[t],
                                  KS[i], const[t]) for i, t in ((1, "income"), (2, "occupation"))}
            mm = {t: task_metrics(q[i][roles["CALIBRATION_TRAIN_MATCHED"]], hard[i][roles["CALIBRATION_TRAIN_MATCHED"]],
                                  yM[t], KS[i], const[t]) for i, t in ((1, "income"), (2, "occupation"))}
            UTIL[(k, rid)] = {"inner": m, "preserved": pres, "finite": fin}
            s = (stored.get(k) or {}).get(rid)
            if s is None:
                if k in stored:
                    chk.bad(f"s{k} {rid}: missing from util__s{k}")
                continue
            w = f"s{k} {rid}"
            chk.same("preserved", {int(a): bool(b) for a, b in s["preserved"].items()}, pres, w)
            chk.same("finite", bool(s["finite"]), fin, w)
            for tag, mine in (("inner", m), ("cal", mc), ("tm", mm)):
                for t in TASKS:
                    for key in ("acc", "logloss", "brier", "const_acc", "gain"):
                        chk.diff(f"{tag}.{key}", s[tag][t][key], mine[t][key], TOL, f"{w} {t}")
                    chk.same(f"{tag}.n", s[tag][t]["n"], mine[t]["n"], f"{w} {t}")
                    chk.same(f"{tag}.const_class", s[tag][t].get("const_class", const[t]), const[t], f"{w} {t}")
        if k in stored:
            extra = sorted(set(stored[k]) - set(ALL_RIDS))
            chk.ok(not extra, f"util__s{k} holds unregistered releases {extra[:5]}")
    if any((k, U_ID) not in UTIL for k in SEEDS):
        chk.pend("utility of U incomplete: gates not replayed")
        chk.done()
        return
    # Ucal*
    rows = []
    for rid in U_RIDS:
        per, ok = [], True
        for k in SEEDS:
            u = UTIL.get((k, rid))
            if u is None or not u["finite"] or not all(u["preserved"].values()):
                ok = False
                per.append(None)
                continue
            per.append(u["inner"]["income"]["logloss"] + u["inner"]["occupation"]["logloss"])
        rows.append((rid, mean3(per) if ok else None))
    valid = [(r, v) for r, v in rows if v is not None and math.isfinite(v)]
    if not valid:
        chk.bad("no valid Ucal* candidate")
        chk.done()
        return
    best = min(v for _, v in valid)
    tied = [r for r, v in valid if v <= best + 1e-12]
    ucal = tied[0]
    UCAL.update({"release": ucal, "family": U_FAM[ucal], "means": dict(rows), "tied": tied})
    # gates and table
    for rid in ALL_RIDS:
        if any((k, rid) not in UTIL for k in SEEDS):
            continue
        seeds = {}
        for k in SEEDS:
            u = UTIL[(k, rid)]
            g0 = gate_u0(u["inner"], UTIL[(k, U_ID)]["inner"])
            gc = gate_ucal(u["inner"], UTIL[(k, ucal)]["inner"])
            e0 = all(v >= 0 for t in TASKS for v in g0[t].values()) and all(u["preserved"].values())
            ec = all(v >= 0 for t in TASKS for v in gc[t].values())
            minm = min([v for t in TASKS for v in g0[t].values()] + [v for t in TASKS for v in gc[t].values()])
            seeds[k] = {"u0": e0, "ucal": ec, "eligible": e0 and ec and u["finite"], "min_margin": minm}
        ne = max(max((UTIL[(k, rid)]["inner"][t]["logloss"] - UTIL[(k, ref)]["inner"][t]["logloss"]) / ALLOW_LL,
                      (UTIL[(k, rid)]["inner"][t]["brier"] - UTIL[(k, ref)]["inner"][t]["brier"]) / ALLOW_BR)
                 for k in SEEDS for t in TASKS for ref in (U_ID, ucal))
        snll = mean3([UTIL[(k, rid)]["inner"]["income"]["logloss"] + UTIL[(k, rid)]["inner"]["occupation"]["logloss"]
                      for k in SEEDS])
        TABLE[rid] = {"eligible_all_seeds": all(seeds[k]["eligible"] for k in SEEDS),
                      "eligible_u0_all_seeds": all(seeds[k]["u0"] for k in SEEDS),
                      "eligible_ucal_all_seeds": all(seeds[k]["ucal"] for k in SEEDS),
                      "worst_norm_excess": ne, "mean_summed_nll": snll, "seeds": seeds,
                      "min_abs_margin": min(abs(seeds[k]["min_margin"]) for k in SEEDS)}
    near = sorted((v["min_abs_margin"], r) for r, v in TABLE.items() if v["min_abs_margin"] <= 1e-9)
    chk.info["gate_margins_within_1e-9_of_zero"] = [r for _, r in near][:20]
    # audit plan
    for p in PARTS:
        elig = [rid_of(p, d) for d in decoders(p) if d != "T-TOKEN32" and TABLE.get(rid_of(p, d), {}).get(
            "eligible_all_seeds")]
        if elig:
            PLAN[p] = ("UTILITY_ELIGIBLE_VARIANT", elig)
        elif p in TASK_ONLY or p in DIAG:
            PLAN[p] = ("ALWAYS_AUDITED_TASK_ONLY_OR_DIAGNOSTIC", [])
        else:
            PLAN[p] = ("PREDECLARED_UTILITY_INELIGIBLE", [])
    chk.info["ucal_star"] = {"release": ucal, "means": dict(rows), "tied_within_1e-12": tied}
    chk.info["audited"] = [p for p in PARTS if PLAN[p][0] != "PREDECLARED_UTILITY_INELIGIBLE"]
    chk.info["eligible_releases"] = [r for r in ALL_RIDS if TABLE.get(r, {}).get("eligible_all_seeds")]
    # compare with the stored audit plan / utility table
    try:
        plan = read_json(RUN / "audit_plan.json")
        chk.same("Ucal* family", plan["ucal_star"]["family"], U_FAM[ucal])
        chk.same("audited", plan["audited"], chk.info["audited"])
        for p in PARTS:
            sp = plan["partitions"][p]
            chk.same("plan reason", sp["reason"], PLAN[p][0], p)
            chk.same("plan eligible variants", sp.get("eligible_variants", []), PLAN[p][1], p)
    except Missing as e:
        chk.pend(str(e))
    try:
        st = read_json(RUN / "utility_table.json")
        for rid, v in TABLE.items():
            s = st.get(rid)
            if s is None:
                chk.bad(f"{rid}: missing from utility_table.json")
                continue
            for key in ("eligible_all_seeds", "eligible_u0_all_seeds", "eligible_ucal_all_seeds"):
                chk.same(key, bool(s[key]), v[key], rid)
            chk.diff("worst_norm_excess", s["worst_norm_excess"], v["worst_norm_excess"], 1e-9, rid)
            chk.diff("mean_summed_nll", s["mean_summed_nll"], v["mean_summed_nll"], TOL, rid)
            for k in SEEDS:
                ss = s["seeds"][str(k)]
                chk.same("seed eligible_u0", bool(ss["eligible_u0"]), v["seeds"][k]["u0"], f"{rid} s{k}")
                chk.same("seed eligible_ucal", bool(ss["eligible_ucal"]), v["seeds"][k]["ucal"], f"{rid} s{k}")
    except Missing as e:
        chk.pend(str(e))
    chk.done()


# ------------------------------------------------------------------ (4) attack banks
def bank_lists(keys, prefix=""):
    lists = {"v1": [("own", "v1")], "v2": [("own", "v2")],
             "pair": [("coalition", "pair"), ("ignore_recipient_2", "v1"), ("ignore_recipient_1", "v2")]}
    out = {}
    for w in VIEWS:
        rows = []
        for cand, view in lists[w]:
            pre = f"{prefix}{view}:"
            rows += [(cand, view, k_[len(pre):], k_) for k_ in keys if k_.startswith(pre) and ":" not in k_[len(pre):]]
        out[w] = rows
    return out


_ENTRY = {}


def replay_family(name, keys, P, SEL, ys, chk, stored=None, where=""):
    """Own within-bank selection (fixed orientation, first-in-bank-order ties) and attacker-seed means of one family:
    keys (n_keys,), P (n_keys, n_INNER) seed-0 predictions, SEL {(crit, w): (seeds, n_INNER)}. stored: the record's
    family dict to compare (selection pred keys exact; values <= 1e-12)."""
    if name in _ENTRY:
        return _ENTRY[name]
    keys = [str(x) for x in keys]
    Pk = dict(zip(keys, P))
    cache = {}

    def stats(key):
        if key not in cache:
            cache[key] = (auc_fixed(ys, Pk[key]), ce_fixed(ys, Pk[key]))
        return cache[key]
    e = {"auc": {}, "ce": {}, "auc_seed0": {}, "ce_seed0": {}, "auc_per_seed": {}, "ce_per_seed": {},
         "pred_key": {}, "rows": {}}
    lists = bank_lists(keys)
    for w in VIEWS:
        rows = lists[w]
        ia = ic = None
        for j, (_, _, _, key) in enumerate(rows):
            a, c = stats(key)
            if ia is None or a > stats(rows[ia][3])[0] + TIE:
                ia = j
            if ic is None or c < stats(rows[ic][3])[1] - TIE:
                ic = j
        e["rows"][w] = [(r[0], r[1], r[2]) for r in rows]
        for crit, j in (("auc", ia), ("ce", ic)):
            key = rows[j][3]
            A = np.asarray(SEL[(crit, w)], dtype=np.float64)
            chk.ok(np.array_equal(A[0], Pk[key]), f"{where}{name} {crit}/{w}: stored seed-0 refit != bank prediction")
            per = [auc_fixed(ys, a) if crit == "auc" else ce_fixed(ys, a) for a in A]
            e["pred_key"][(crit, w)] = key
            e[f"{crit}_seed0"][w] = stats(key)[0 if crit == "auc" else 1]
            e[f"{crit}_per_seed"][w] = per
            e[crit][w] = float(np.mean(per))
            chk.diff("seed0 value == per-seed[0]", per[0], e[f"{crit}_seed0"][w], TOL, f"{where}{name} {crit}/{w}")
    if stored is not None:
        for w in VIEWS:
            for crit in CRITS:
                s = stored["selection"][w][crit]
                chk.same("selected pred_key", s["pred_key"], e["pred_key"][(crit, w)], f"{where}{name} {crit}/{w}")
                chk.diff("selection value", s["inner_auc" if crit == "auc" else "inner_ce"], e[f"{crit}_seed0"][w],
                         TOL, f"{where}{name} {crit}/{w}")
                for key in (crit, f"{crit}_seed0"):
                    chk.diff(key, stored[key][w], e[key][w], TOL, f"{where}{name} {w}")
                chk.diff(f"{crit}_per_seed", stored[f"{crit}_per_seed"][w], e[f"{crit}_per_seed"][w], TOL,
                         f"{where}{name} {w}")
            tb = (stored.get("tables") or {}).get(w)
            if tb:
                chk.same("bank row order", [(r["candidate"], r["view"], r["attacker"]) for r in tb], e["rows"][w],
                         f"{where}{name} {w}")
                d = max(abs(r["inner_auc"] - stats(r["pred_key"])[0]) for r in tb)
                chk.diff("bank row AUC", d, 0.0, TOL, f"{where}{name} {w}")
    _ENTRY[name] = e
    return e


def legacy_family(k, rid, ys, chk, sel_rid):
    name = f"aud__{lra_release_unit(k, rid)}"
    rec, f = unit_record(name, ADM_UNITS)
    chk.same("cid", rec.get("cid"), rid, name)
    chk.same("primary family", (rec.get("primary_family"), rec["recovery"].get("family")), ("code", "code"), name)
    r = rec["recovery"]
    chk.same("attacker seeds", list(r["attacker_seeds"]), [0, 1, 2], name)
    z = npz(f["inner_preds.npz"])
    chk.ok(np.array_equal(z["sel_row_id"], sel_rid), f"{name}: INNER rows differ from D's INNER_SELECTION")
    SEL = {(c, w): z[f"SEL_code_{c}_{w}"] for c in CRITS for w in VIEWS}
    e = replay_family(name, z["keys_code"], z["P_code"], SEL, ys, chk, stored=r)
    return name, e


def union(entries):
    out = {"winner": {}, "ce_winner": {}}
    for crit in CRITS:
        for key in (crit, f"{crit}_seed0", f"{crit}_per_seed"):
            out[key] = {}
    out["bank_seed0"] = {c: {} for c in CRITS}
    for w in VIEWS:
        ia = ic = 0
        for j, (_, e) in enumerate(entries):
            if e["auc_seed0"][w] > entries[ia][1]["auc_seed0"][w] + TIE:
                ia = j
            if e["ce_seed0"][w] < entries[ic][1]["ce_seed0"][w] - TIE:
                ic = j
        for crit, j in (("auc", ia), ("ce", ic)):
            nm, e = entries[j]
            out["winner" if crit == "auc" else "ce_winner"][w] = nm
            out[crit][w] = e[crit][w]
            out[f"{crit}_seed0"][w] = e[f"{crit}_seed0"][w]
            out[f"{crit}_per_seed"][w] = list(e[f"{crit}_per_seed"][w])
            out["bank_seed0"][crit][w] = {n_: x[f"{crit}_seed0"][w] for n_, x in entries}
        # near-tie receipt (a float difference could flip a winner only inside this band)
        s0 = sorted(x["auc_seed0"][w] for _, x in entries)
        out.setdefault("near_ties", {})[w] = bool(len(s0) > 1 and any(0 < abs(a - b) <= 1e-12 + 1e-15
                                                                       for a, b in zip(s0, s0[1:])))
    return out


def compare_union(chk, mine, stored, where):
    for w in VIEWS:
        chk.same("auc winner", stored["winner"][w], mine["winner"][w], f"{where} {w}")
        chk.same("ce winner", stored["ce_winner"][w], mine["ce_winner"][w], f"{where} {w}")
        for crit in CRITS:
            chk.diff(crit, stored[crit][w], mine[crit][w], TOL, f"{where} {w}")
            chk.diff(f"{crit}_seed0", stored[f"{crit}_seed0"][w], mine[f"{crit}_seed0"][w], TOL, f"{where} {w}")
            chk.diff(f"{crit}_per_seed", stored[f"{crit}_per_seed"][w], mine[f"{crit}_per_seed"][w], TOL,
                     f"{where} {w}")
            sb = stored.get("bank_seed0", {}).get(crit, {}).get(w)
            if sb is not None:
                chk.same("bank names", list(sb), list(mine["bank_seed0"][crit][w]), f"{where} {w}")
                chk.diff("bank seed0", [sb[n] for n in sb], [mine["bank_seed0"][crit][w][n] for n in sb], TOL,
                         f"{where} {w}")
    if any(mine.get("near_ties", {}).values()):
        chk.notes.append(f"{where}: seed-0 values within 1e-12 of each other (tie band)")


def canonical_cols(tok, D, alphabet):
    """Registered one-hot column order of the fresh token block (re-derived): first occurrence of each token value in
    AUDIT_FIT (D order), then INNER_SELECTION, then all rows; never-occurring values last."""
    order, seen = [], set()
    for ix in (np.sort(np.asarray(D["idx"]["AUDIT_FIT"])), np.sort(np.asarray(D["idx"]["INNER_SELECTION"])),
               np.arange(tok.size)):
        t = tok[ix]
        _, first = np.unique(t, return_index=True)
        for v in t[np.sort(first)]:
            if int(v) not in seen:
                seen.add(int(v))
                order.append(int(v))
    order += [v for v in range(int(alphabet)) if v not in seen]
    col = np.empty(int(alphabet), dtype=np.int64)
    col[np.asarray(order, dtype=np.int64)] = np.arange(int(alphabet))
    return col


def onehot(idx, K):
    out = np.zeros((idx.size, int(K)))
    out[np.arange(idx.size), idx] = 1.0
    return out


def hygiene(X, F):
    """Registered FRESH VIEW HYGIENE (PROTOCOL section 5), re-derived: drop columns constant on the ATTACK_FIT_NEW rows
    F, then columns bitwise identical on F to an earlier kept column."""
    Xf = X[F]
    const = np.all(Xf == Xf[:1], axis=0)
    kept, seen, ndup = [], set(), 0
    cols = np.ascontiguousarray(Xf.T)
    for j in range(X.shape[1]):
        if const[j]:
            continue
        key = cols[j].tobytes()
        if key in seen:
            ndup += 1
            continue
        seen.add(key)
        kept.append(j)
    kept = np.asarray(kept, dtype=np.int64)
    return X[:, kept], {"n_in": int(X.shape[1]), "n_kept": int(kept.size), "dropped_constant": int(const.sum()),
                        "dropped_duplicate": int(ndup), "kept_sha256": sha_arrays(kept)}


def fresh_view_receipt(chk, D, roles, b, tables, p, frec, where):
    """Rebuild the complete fresh interface of partition p from the bank and the stored registered tables
    ([one-hot token, every table in registered order, one-hot decision], then hygiene on ATTACK_FIT_NEW; pair =
    hygiene([v1', v2'])) and compare the hygiene receipt and the view fingerprint with the fresh record."""
    F = np.sort(np.asarray(roles["ATTACK_FIT_NEW"], dtype=np.int64))
    X, rec = {}, {}
    for i in (1, 2):
        tok = np.asarray(b[f"tok{i}"], dtype=np.int64)
        a = int(b[f"alpha{i}"])
        blocks = [onehot(canonical_cols(tok, D, a)[tok], a)]
        blocks += [np.asarray(tables[d][i - 1], dtype=np.float64)[tok] for d in decoders(p)]
        blocks.append(onehot(np.asarray(b[f"hard{i}"], dtype=np.int64), KS[i]))
        X[f"v{i}"], rec[f"v{i}"] = hygiene(np.hstack(blocks), F)
    X["pair"], rec["pair"] = hygiene(np.hstack([X["v1"], X["v2"]]), F)
    hyg = (frec.get("meta") or {}).get("hygiene") or {}
    for w in VIEWS:
        s = hyg.get(w) or {}
        for key in ("n_in", "n_kept", "dropped_constant", "dropped_duplicate", "kept_sha256"):
            chk.same(f"hygiene {key}", s.get(key), rec[w][key], f"{where} {w}")
    chk.same("hygiene fit rows", (hyg.get("fit_rows") or {}).get("rows"), int(F.size), where)
    chk.same("fresh view fingerprint", frec.get("view_fingerprint"), sha_arrays(X["v1"], X["v2"], X["pair"]), where)
    chk.info.setdefault("hygiene_dropped", {"constant": 0, "duplicate": 0})
    for w in VIEWS:
        chk.info["hygiene_dropped"]["constant"] += rec[w]["dropped_constant"]
        chk.info["hygiene_dropped"]["duplicate"] += rec[w]["dropped_duplicate"]


REC = {}        # (k, key) -> own recovery {"auc": {w: mean}, "auc_per_seed": ...} (key = partition or SRC|U)


def check_attacks(D, roles, LB, audited):
    chk = Check("4_COMMON_RECORDS", "within-bank selections, seed means and union records (codes and U)")
    if LB is None:
        chk.pend("attack replay needs INNER_SELECTION SEX (SCIENCE_LOCK not pushed)")
        chk.done()
        return
    inner = np.asarray(D["idx"]["INNER_SELECTION"], dtype=np.int64)
    sel_rid = np.asarray(D["row_id"])[inner]
    _, ys = LB.get("sex", "INNER_SELECTION", "AUC / CE of stored INNER predictions")
    att = roles["ATTACK_FIT_NEW"]
    fit_sha_new = sha_arrays(np.asarray(D["row_id"])[att])
    fit_sha_old = sha_arrays(np.asarray(D["row_id"])[np.asarray(D["idx"]["AUDIT_FIT"])])
    sel_sha = sha_arrays(sel_rid)
    chk.info["audited_replayed"] = list(audited)
    for k in SEEDS:
        fresh_entries = []
        for p in audited:
            where = f"s{k} {p}"
            try:
                frec, ff = unit_record(f"fam__s{k}__{safe(p)}")
                crec, _ = unit_record(f"com__s{k}__{safe(p)}")
            except Missing as e:
                chk.pend(str(e))
                continue
            except Corrupt as e:
                chk.bad(str(e))
                continue
            # fresh bank: roles, rows, registered table set
            chk.same("fresh fit role", (frec.get("roles") or {}).get("fit"), "ATTACK_FIT_NEW", where)
            chk.same("fresh fit rows", frec.get("fit_row_id_sha256"), fit_sha_new, where)
            chk.same("fresh INNER rows", frec.get("sel_row_id_sha256"), sel_sha, where)
            chk.same("fresh n_fit", frec.get("n_fit"), int(att.size), where)
            chk.same("fresh attacker seeds", list(frec.get("attacker_seeds", [])), [0, 1, 2], where)
            meta = frec.get("meta") or {}
            chk.same("fresh view tables (registered order)", list(meta.get("tables", [])), decoders(p), where)
            if (k, p) in STORED_TABLES:
                st_t = STORED_TABLES[(k, p)]
                for d in decoders(p):
                    chk.same("fresh view table sha256", (meta.get("table_sha256") or {}).get(d),
                             sha_arrays(np.asarray(st_t[d][0], dtype=np.float64),
                                        np.asarray(st_t[d][1], dtype=np.float64)), f"{where} {d}")
                fresh_view_receipt(chk, D, roles, bank(k, p), st_t, p, frec, where)
            else:
                chk.pend(f"{where}: fresh view not reconstructed (cal__ tables unavailable)")
            z = npz(ff["inner_preds.npz"])
            chk.ok(np.array_equal(z["sel_row_id"], sel_rid), f"{where}: fresh INNER rows differ")
            SEL = {(c, w): z[f"SEL_fresh_{c}_{w}"] for c in CRITS for w in VIEWS}
            fname = f"fam__s{k}__{safe(p)}"
            fe = replay_family(fname, z["keys_fresh"], z["P_fresh"], SEL, ys, chk, stored=frec, where="")
            fresh_entries.append((f"fresh:{p}", fe))
            # legacy banks (admitted lra inner audits of the original releases), then the fresh bank
            ents = []
            try:
                for rid in orig_rids_of(p):
                    nm, le = legacy_family(k, rid, ys, chk, sel_rid)
                    lrec = unit_record(nm, ADM_UNITS)[0]["recovery"]
                    chk.same("legacy fit rows", lrec["fit_row_id_sha256"], fit_sha_old, nm)
                    chk.same("legacy INNER rows", lrec["sel_row_id_sha256"], sel_sha, nm)
                    ents.append((nm, le))
            except (Missing, Corrupt) as e:
                chk.bad(f"{where}: admitted legacy bank: {e}")
                continue
            ents.append(("fresh", fe))
            mine = union(ents)
            com = crec["recovery"]
            chk.same("common partition / seed", (com.get("partition"), com.get("seed")), (p, k), where)
            chk.same("bank order", [b["name"] for b in com["banks"]], [n for n, _ in ents], where)
            compare_union(chk, mine, com, where)
            for w in VIEWS:
                for crit in CRITS:
                    det = com["winner_detail"][w][crit]
                    wn = mine["winner" if crit == "auc" else "ce_winner"][w]
                    e = dict(ents)[wn]
                    chk.same("winner pred_key", det.get("pred_key"), e["pred_key"][(crit, w)], f"{where} {w}/{crit}")
                    pref = "SEL_fresh" if wn == "fresh" else "SEL_code"
                    chk.same("winner stored_key", det.get("stored_key"), f"{pref}_{crit}_{w}", f"{where} {w}/{crit}")
                    chk.same("winner fit role", det.get("fit_role"), "ATTACK_FIT_NEW" if wn == "fresh" else "AUDIT_FIT",
                             f"{where} {w}/{crit}")
            REC[(k, p)] = mine
        # continuous U: the admitted lra SRC|U composed bank (re-derived here), then the fresh banks in order
        try:
            urec, uf = unit_record(f"aud__tea__s{k}__U", ADM_UNITS)
        except (Missing, Corrupt) as e:
            chk.bad(f"s{k}: admitted SRC|U record: {e}")
            continue
        r = urec["recovery"]
        uz = npz(uf["inner_preds.npz"])
        chk.ok(np.array_equal(uz["sel_row_id"], sel_rid), f"s{k} SRC|U: INNER rows differ")
        own = replay_family(f"aud__tea__s{k}__U", uz["keys_interface"], uz["P_interface"],
                            {(c, w): uz[f"SEL_interface_{c}_{w}"] for c in CRITS for w in VIEWS}, ys, chk)
        for w in VIEWS:
            for crit in CRITS:
                chk.diff(f"own {crit}", r["own"][crit][w], own[crit][w], TOL, f"s{k} SRC|U {w}")
                chk.diff(f"own {crit}_seed0", r["own"][f"{crit}_seed0"][w], own[f"{crit}_seed0"][w], TOL,
                         f"s{k} SRC|U {w}")
                chk.diff(f"own {crit}_per_seed", r[f"{crit}_per_seed"][w], own[f"{crit}_per_seed"][w], TOL,
                         f"s{k} SRC|U {w}")
        comp = r["composed"]
        chk.same("lra composition order (84 codes)", list(comp["policies"]), ORIGINAL_84, f"s{k} SRC|U")
        lra_ents = [("source", own)]
        try:
            for rid in ORIGINAL_84:
                _, le = legacy_family(k, rid, ys, chk, sel_rid)
                lra_ents.append((rid, le))
        except (Missing, Corrupt) as e:
            chk.bad(f"s{k}: admitted code bank: {e}")
            continue
        lra_u = union(lra_ents)
        for w in VIEWS:
            chk.same("lra composed auc winner", comp["winner"][w], lra_u["winner"][w], f"s{k} SRC|U {w}")
            chk.same("lra composed ce winner", comp["ce_winner"][w], lra_u["ce_winner"][w], f"s{k} SRC|U {w}")
            for crit in CRITS:
                chk.diff(f"lra composed {crit}", comp[crit][w], lra_u[crit][w], TOL, f"s{k} SRC|U {w}")
                chk.diff(f"lra composed {crit}_seed0", comp[f"{crit}_seed0"][w], lra_u[f"{crit}_seed0"][w], TOL,
                         f"s{k} SRC|U {w}")
        lra_entry = {key: lra_u[key] for key in ("auc", "ce", "auc_seed0", "ce_seed0", "auc_per_seed", "ce_per_seed")}
        uname = f"aud__tea__s{k}__U"
        if len(fresh_entries) != len(audited):
            chk.pend(f"s{k} SRC|U: {len(audited) - len(fresh_entries)} audited fresh bank(s) missing; U not composed")
            continue
        ents = [(uname, lra_entry)] + fresh_entries
        mine = union(ents)
        REC[(k, U_ID)] = mine
        try:
            crec, _ = unit_record(f"com__s{k}__SRC_U")
        except Missing as e:
            chk.pend(str(e))
            continue
        com = crec["recovery"]
        chk.same("U bank order", [b["name"] for b in com["banks"]], [n for n, _ in ents], f"s{k} SRC|U")
        compare_union(chk, mine, com, f"s{k} SRC|U")
    # every decoder variant shares its partition's record: the selection table must agree
    chk.done()


# ------------------------------------------------------------------ (5) T* / P*
def recovery(k, rid):
    key = U_ID if rid in U_RIDS else RID_INFO[rid][0]
    return REC.get((k, key))


FORM_DISAGREE = []


def rank(pool, ref=None, audited=()):
    order = {r: n for n, r in enumerate(ALL_RIDS)}
    rows, rej = [], []
    for rid in pool:
        t = TABLE.get(rid)
        if t is None:
            rej.append((rid, "NO_UTILITY"))
            continue
        if not t["eligible_all_seeds"]:
            rej.append((rid, "UTILITY_INELIGIBLE"))
            continue
        p = None if rid in U_RIDS else RID_INFO[rid][0]
        if p is not None and p not in audited:
            rej.append((rid, "NOT_AUDITED_TECHNICAL"))
            continue
        rec = {w: [recovery(k, rid)["auc"][w] if recovery(k, rid) else float("nan") for k in SEEDS] for w in VIEWS}
        row = {"release": rid, "mean_pair_auc": mean3(rec["pair"]), "worst_norm_excess": t["worst_norm_excess"],
               "mean_summed_nll": t["mean_summed_nll"], "order": order[rid], "auc_per_seed": rec}
        if ref is not None:
            # PROTOCOL section 6 literally: AUC_i(P*) <= AUC_i(T*) + 0.005 (each recipient, each seed); the
            # difference form AUC_i(P*) - AUC_i(T*) <= 0.005 (SELECTION_RULES.json wording) can differ by one ulp
            g_sum = all(rec[w][k] <= ref[w][k] + GUARD for w in ("v1", "v2") for k in range(3))
            g_diff = all(rec[w][k] - ref[w][k] <= GUARD for w in ("v1", "v2") for k in range(3))
            ben = mean3([ref["pair"][k] - rec["pair"][k] for k in range(3)])
            ben2 = mean3(ref["pair"]) - mean3(rec["pair"])
            row.update({"guard_ok": g_sum, "guard_forms_agree": g_sum == g_diff, "pair_benefit": ben,
                        "benefit_forms_agree": (ben >= BENEFIT) == (ben2 >= BENEFIT)})
            if not (row["guard_forms_agree"] and row["benefit_forms_agree"]):
                FORM_DISAGREE.append(rid)
            if not g_sum:
                rej.append((rid, "LOCAL_GUARD_FAILURE"))
                continue
            if ben < BENEFIT:
                rej.append((rid, "PAIR_BENEFIT_BELOW_0.02"))
                continue
        rows.append(row)
    rows.sort(key=lambda r: (r12(r["mean_pair_auc"]), r12(r["worst_norm_excess"]), r12(r["mean_summed_nll"]),
                             r["order"]))
    return rows, rej


def check_selection(audited):
    chk = Check("5_SELECTION", "T* and P* with every rejection reason (protocol section 6)")
    if not TABLE or any((k, U_ID) not in REC for k in SEEDS):
        chk.pend("selection replay needs the utility table and every common record")
        chk.done()
        return
    t_pool = [r for r in ALL_RIDS if r in U_RIDS or (RID_INFO[r][0] in TASK_ONLY and RID_INFO[r][1] != "T-TOKEN32")]
    p_pool = [r for r in CODE_RIDS if privacy_trained(RID_INFO[r][0])
              and RID_INFO[r][1] in ("D0", "D1") + NEW_DECS]
    chk.info["pool_sizes"] = {"T*": len(t_pool), "P*": len(p_pool)}
    trows, trej = rank(t_pool, audited=audited)
    tstar = trows[0]["release"] if trows else None
    prows, prej = [], []
    if tstar is not None:
        prows, prej = rank(p_pool, ref=trows[0]["auc_per_seed"], audited=audited)
    pstar = prows[0]["release"] if prows else None
    chk.info.update({"T*": tstar, "P*": pstar, "T*_ranking": [r["release"] for r in trows][:10],
                     "P*_ranking": [r["release"] for r in prows][:10],
                     "P*_rejections": {reason: sum(1 for _, x in prej if x == reason) for reason in
                                       sorted({x for _, x in prej})},
                     "guard_or_benefit_form_disagreements": sorted(set(FORM_DISAGREE))})
    try:
        sel = read_json(RUN / "selection.json")
    except Missing as e:
        chk.pend(str(e))
        chk.done()
        return
    chk.same("Ucal*", sel["ucal_star"]["release"], UCAL.get("release"))
    # select_all overwrites T* / P* statuses with TECHNICAL_FAILURE when the controls are not all_ok; the replay
    # compares the pre-control status (status_if_controls_had_passed) and reports the control gate separately
    st_of = lambda role: sel[role].get("status_if_controls_had_passed", sel[role]["status"])  # noqa: E731
    chk.info["controls_all_ok_in_selection"] = sel.get("controls_all_ok")
    chk.info["recorded_statuses"] = {r: sel[r]["status"] for r in ("T*", "P*")}
    chk.same("T* release", sel["T*"]["release"], tstar)
    chk.same("T* status", st_of("T*"), "NOMINEE" if tstar else "NO_ELIGIBLE")
    chk.same("T* ranking", [r["release"] for r in sel["T*"]["ranking"]], [r["release"] for r in trows])
    chk.same("T* rejections", sorted((r["release"], r["reason"]) for r in sel["T*"]["rejected"]), sorted(trej))
    for s, m in zip(sel["T*"]["ranking"], trows):
        chk.diff("T* mean pair AUC", s["mean_pair_auc"], m["mean_pair_auc"], TOL, s["release"])
    if tstar is None:
        chk.same("P* status", st_of("P*"), "NOT_SELECTED_NO_COMPARATOR")
    else:
        chk.same("P* release", sel["P*"]["release"], pstar)
        chk.same("P* status", st_of("P*"), "NOMINEE" if pstar else "NO_ELIGIBLE_COMPETITIVE_NOMINEE")
        chk.same("P* ranking", [r["release"] for r in sel["P*"]["ranking"]], [r["release"] for r in prows])
        chk.same("P* rejections", sorted((r["release"], r["reason"]) for r in sel["P*"]["rejected"]), sorted(prej))
        for s, m in zip(sel["P*"]["ranking"], prows):
            chk.diff("P* mean pair AUC", s["mean_pair_auc"], m["mean_pair_auc"], TOL, s["release"])
            chk.diff("P* pair benefit", s["pair_benefit"], m["pair_benefit"], TOL, s["release"])
        if pstar is not None:
            p, d = RID_INFO[pstar]
            disc = sel["P*"].get("disclosures") or []
            chk.same("incumbent disclosure", any("incumbent" in x for x in disc), d in ("D0", "D1"))
            fam = p.split("|")[1]
            loc = fam in ("LOCAL", "SEQ-12", "SEQ-21") or fam in ("W-LOCAL", "W-SEQ-12", "W-SEQ-21") or \
                fam in ("K-LOCAL", "K-SEQ-12", "K-SEQ-21")
            chk.same("joint-credit disclosure", any("joint-design" in x for x in disc), loc)
    # every decoder variant of a partition shares one recovery in the stored selection rows
    for part in (sel.get("T*", {}).get("ranking", []) + sel.get("P*", {}).get("ranking", [])):
        rid = part["release"]
        mine = {w: [recovery(k, rid)["auc"][w] for k in SEEDS] for w in VIEWS}
        chk.diff("selection row recovery", [part["auc_per_seed"][w] for w in VIEWS], [mine[w] for w in VIEWS], TOL,
                 rid)
    chk.done()


# ------------------------------------------------------------------ informational inputs
def input_status():
    out = {}
    for nm in ("controls.json", "replay.json"):
        p = RUN / nm
        if not p.exists():
            out[nm] = "MISSING"
            continue
        j = read_json(p)
        if nm == "controls.json":
            out[nm] = {"top_level_all_ok": j.get("all_ok"), "verdict_all_ok": (j.get("verdict") or {}).get("all_ok"),
                       "failures": (j.get("verdict") or {}).get("failures")}
        else:
            out[nm] = {"all_ok": j.get("all_ok"), "rows": len(j.get("rows", []))}
    return out


# ------------------------------------------------------------------ main
def main():
    t0, c0 = time.time(), time.process_time()
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("REFUSED: OMP_NUM_THREADS must be 1")
    sys.path.insert(0, str(WT))
    from lra import data as LD                       # pinned loader only (sealed D)
    D = LD.load(verify=True, unseal=False)
    a = np.asarray(D["idx"]["OSF_DEVELOPMENT_ASSESSMENT"])
    assert D["sealed"] and all(np.all(np.asarray(D[x])[a] == -1) for x in ("sex", "y_income", "y_occupation_group"))
    roles = check_roles(D)
    check_bank(D)
    lock_ok, lock_why = on_origin(f"{REL}/SCIENCE_LOCK.json")
    LB = Labels(D, roles) if lock_ok else None
    check_calibration(D, roles, LB)
    check_utility(D, roles, LB)
    audited = [p for p in PARTS if PLAN.get(p, ("PREDECLARED_UTILITY_INELIGIBLE",))[0] !=
               "PREDECLARED_UTILITY_INELIGIBLE"] if PLAN else []
    if LB is not None and not PLAN:
        c = Check("4_COMMON_RECORDS", "within-bank selections, seed means and union records (codes and U)")
        c.pend("audit plan not replayed (utility incomplete)")
        c.done()
    else:
        check_attacks(D, roles, LB, audited)
    check_selection(audited)
    loaded_hcal = sorted(m for m in sys.modules if m == "hcal" or m.startswith("hcal."))
    worktree_modules = sorted({m.split(".")[0] for m, mod in list(sys.modules.items())
                               if str(getattr(mod, "__file__", "") or "").startswith(str(WT))})
    statuses = {k: v["status"] for k, v in CHECKS.items()}
    if loaded_hcal:
        verdict = "FAIL"
    elif any(s == "FAIL" for s in statuses.values()):
        verdict = "FAIL"
    elif all(s == "PASS" for s in statuses.values()):
        verdict = "PASS"
    else:
        verdict = "PENDING"
    out = {"schema": "hcal-E-inner-replay-v1", "role": "E (independent verifier)",
           "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "script": f"{REL}/verification/verify_inner.py", "script_sha256": sha_file(HERE),
           "head": git("rev-parse", "HEAD").stdout.strip(),
           "science_lock": {"pushed": lock_ok, "reason": lock_why},
           "labels_policy": "no label read unless SCIENCE_LOCK.json is byte-identical on origin; then only the reads "
                            "listed in label_reads (HEAD_VALIDATION and assessment labels never read)",
           "label_reads": LABEL_READS,
           "independence": {"hcal_modules_loaded": loaded_hcal, "worktree_top_level_modules_loaded": worktree_modules,
                            "imports": ["json", "hashlib", "math", "os", "subprocess", "sys", "time", "pathlib",
                                        "numpy", "lra.data (pinned loader; load() only)"]},
           "tolerances": {"utility_auc_ce_tables": TOL, "selections": "exact", "kkt_rel": KKT_REL_TOL,
                          "alpha": ALPHA_TOL, "tie": TIE},
           "statuses": statuses, "checks": CHECKS, "inputs_status": input_status(),
           "inputs_sha256": dict(sorted(INPUT_HASHES.items())),
           "wall_s": round(time.time() - t0, 1), "cpu_s": round(time.process_time() - c0, 1), "verdict": verdict}
    txt = json.dumps(jsafe(out), indent=1, allow_nan=False) + "\n"
    for bad in (str(Path.home()) + "/", "/Users/", "/private/"):
        if bad in txt:
            raise SystemExit("REFUSED: a private path would enter E_INNER_REPLAY.json")
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(txt)
    tmp.rename(OUT)
    print(json.dumps({"verdict": verdict, "statuses": statuses, "science_lock": lock_why,
                      "wall_s": out["wall_s"]}, indent=1))


if __name__ == "__main__":
    main()
