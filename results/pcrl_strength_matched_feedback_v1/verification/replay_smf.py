#!/usr/bin/env python3
"""Independent verifier for the strength-matched feedback study (smf).

Owner: the independent verifier (prompt section 4, role 5). Works from the registered definitions (PROTOCOL.md incl.
section 11, METHOD_CARD.md, METHOD_DELTA.md, PRIMARY_FAMILY.json, ROLE_MANIFEST.json, the named locks and dated
amendments) and from the saved private unit artifacts. It does not use the runner's scientific implementation:

  * a sys.meta_path guard refuses every import under smf, rgj, jcv, pnx, oar and stored_model_eval (stricter than the
    required list smf.select/infer/family/eval_lock/report/audit/assess/preflight/track/train, rgj.*, jcv.infer,
    jcv.select, pnx.*, oar.*, stored_model_eval.bench_infer/pilot_infer); the run asserts none of them is loaded at
    the end, including through unpickling (torch.load(weights_only=True); heads are plain sklearn pipelines);
  * scientific libraries: numpy, scipy, sklearn, torch, joblib (+ json, hashlib and the standard library).

Checks (each returns PASS / FAIL / WARN / PENDING when its inputs do not exist yet / INFO):
  1 roles        recompute the role partition from PROTOCOL section 1 and the source npz; compare with ROLE_MANIFEST;
                 assessment labels unused before EVALUATION_LOCK; closed pools absent from every release row_id
  2 releases     re-implemented forward pass (83-64-64-16 ReLU per recipient) from model.pt + sklearn heads:
                 features, centred logits, probabilities, hard decisions vs release.npz (no labels needed); heads
                 fitted on NEW_DEFENSE_FIT only (scaler moments; exact refit on a sample; HEAD_VALIDATION log loss)
  3 gradients    realized vs conditional ratios vs declared rho (Phase A s=1; Phase B RMS identity with the logged
                 allocation), zero / cap / clip counts, ONLINE_MATCHED extra updates = REFRESHED refit receipt
  4 controller   replay of w <- clip(w + clip((AUC - b)/0.01, -1, 1), 0.25, 8) from diag.controller and calib__s{k};
                 applied (feedback) and hypothetical (twin) weights, allocation, activation status, alias check
  5 selection    Phase A schedule rule / fallback / local reference; Phase B controls, C*, J-F nomination (+0.005
                 buffers), descriptive fallback; compared with selection files and EVALUATION_LOCK
  6 endpoints    own paired multinomial group bootstrap (B=1999, default_rng(20261005), chunks of 250) of all 18
                 primary and 34 secondary slots; compared with PRIMARY/SECONDARY_ENDPOINTS.csv and inference.json
  7 conjunctions claim A / claim B
  8 integrity    COMPLETE.json hashes, lock code hashes vs working tree, lock push before first governed unit
  9 tracking     registered critic gap (mean over kinds of paired CE online - fresh) and best-of-bank separately

  10 drive        restore seed-1 U, J-F and L-F from the backup copy alone (uncached reads) and compare with the local
                 releases; PENDING when the volume is not mounted

Usage (from the worktree root):
    OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python results/pcrl_strength_matched_feedback_v1/verification/replay_smf.py
        [--selftest-only] [--full] [--no-write] [--drive-root <DRIVE_ROOT>]
Writes results/pcrl_strength_matched_feedback_v1/INDEPENDENT_VERIFICATION.json (placeholders only).
"""
from __future__ import annotations

import importlib
import importlib.abc
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

# ------------------------------------------------------------------------------------------------ import guard
_FORBIDDEN_TOP = ("smf", "rgj", "jcv", "pnx", "oar", "stored_model_eval")
_REQUIRED_ABSENT = ("smf.select", "smf.infer", "smf.family", "smf.eval_lock", "smf.report", "smf.audit", "smf.assess",
                    "smf.preflight", "smf.track", "smf.train", "rgj", "jcv.infer", "jcv.select", "pnx", "oar",
                    "stored_model_eval.bench_infer", "stored_model_eval.pilot_infer")
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
OUT = RES / "INDEPENDENT_VERIFICATION.json"
CACHE = Path.home() / "PCRL_eval_cache_private"
SRC = CACHE / "jcv_v1" / "inputs" / "adult_jcv.npz"
SRC_SHA = "e0d9e54af780f30788ee29cfe6795ec82cbdcadc127b1978c69a3891485d2f12"
RUN = CACHE / "smf_v1" / "run"
UNITS = RUN / "units"
ADMISSION = WT / "results" / "pcrl_joint_complete_view_method_v1" / "DATA_ADMISSION.json"
BRANCH = "research/pcrl-strength-matched-feedback-v1"

SEEDS = (0, 1, 2)
RHOS = (0.25, 0.75, 1.5)
BETAS = (0.1, 0.3)
SCHEDS = ("ONLINE", "REFRESHED", "ONLINE_MATCHED")
SCHED_TIE_ORDER = ("ONLINE", "ONLINE_MATCHED", "REFRESHED")
B_ARMS = ("J-F", "L-F", "J-N", "L-N")
CONTROL_ORDER = ("L-F", "J-N", "L-N", "RAW-J", "RAW-L", "E", "F", "F0", "U")
NEW_ROLES = ("NEW_DEFENSE_FIT", "NEW_DEVELOPMENT_ASSESSMENT", "HEAD_VALIDATION", "AUDIT_FIT", "INNER_SELECTION")
SUBROLES = ("CRITIC_FIT", "CRITIC_VAL", "CONTROLLER_CALIB")
RGJ_SUBROLES = ("CRITIC_FIT", "CRITIC_VAL", "CALIB")
ASSESS = "NEW_DEVELOPMENT_ASSESSMENT"
FIT = "NEW_DEFENSE_FIT"
NUMERIC = ("age", "education-num", "capital-gain", "capital-loss", "hours-per-week")
LABEL_KEYS = {"sex": "sex", "race": "race", "y_income": "y_income", "y_occ": "y_occupation_group"}
RELEASE_KEYS = {"row_id", "r1", "c1", "p1", "hard1", "r2", "c2", "p2", "hard2"}

# registered numbers (PROTOCOL section 2/5/7, PRIMARY_FAMILY.json)
A_MAX, ZERO_TOL = 100.0, 1e-12
CTRL_DIV, W_MIN, W_MAX, TARGET_OFFSET, TARGET_FLOOR = 0.01, 0.25, 8.0, 0.01, 0.5
ASYM_THR = 0.05
BUFFER = 0.005
B_BOOT, BOOT_SEED, CHUNK = 1999, 20261005, 250
REFIT_EPOCHS_MATCHED = (0, 4, 8, 12, 16)

STATUS_RANK = {"FAIL": 4, "PENDING": 3, "WARN": 2, "PASS": 1, "INFO": 0, "NOT_APPLICABLE": 0}


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


def fr(x) -> str:
    return f"{x:g}"


def a_unit(k, base, sched, rho):
    return f"A__s{k}__{base}__{sched}__r{fr(rho)}__e20"


def a_run(k, base, sched, rho):
    return f"run__A__s{k}__{base}__{sched}__r{fr(rho)}"


def b_unit(k, arm, rho):
    return f"B__s{k}__{arm}__r{fr(rho)}__e20"


def b_run(k, arm, rho):
    return f"run__B__s{k}__{arm}__r{fr(rho)}"


def raw_unit(k, arm, beta, ep):
    return f"raw__s{k}__{arm}__b{fr(beta)}__e{ep}"


def tl_unit(k, ep):
    return f"tl__s{k}__e{ep}"


def safe(label: str) -> str:
    return label.replace("*", "star").replace("/", "_").replace(" ", "_")


def res(status, **kw):
    return {"status": status, **kw}


def worst(*statuses):
    s = [x for x in statuses if x]
    return max(s, key=lambda x: STATUS_RANK.get(x, 0)) if s else "PENDING"


def rollup(d: dict) -> str:
    return worst(*[v.get("status") for v in d.values() if isinstance(v, dict) and "status" in v])


def utc(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, timezone.utc)


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ") if dt else None


def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
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
    roots = [(str(UNITS.parent.parent), "<PRIVATE_CACHE>/smf_v1"), (str(CACHE), "<PRIVATE_CACHE>"), (str(WT), "<WORKTREE>")]
    if DRIVE_ROOT[0]:
        roots.insert(0, (str(DRIVE_ROOT[0]), "<DRIVE_ROOT>"))
    for root, ph in roots:
        s = s.replace(root, ph)
    return s


def git(*args):
    r = subprocess.run(["git", "-C", str(WT), *args], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def git_ok(*args) -> bool:
    return subprocess.run(["git", "-C", str(WT), *args], capture_output=True, text=True).returncode == 0


def unit_dir(name) -> Path:
    return UNITS / name


def unit_done(name) -> bool:
    return (UNITS / name / "COMPLETE.json").exists()


def unit_rec(name):
    p = UNITS / name / "record.json"
    return jload(p) if p.exists() and unit_done(name) else None


# ------------------------------------------------------------------------------------------------ data and roles
def _u_int(seed, salt, g) -> int:
    return int(hashlib.sha256(f"{seed}|{salt}|{int(g)}".encode()).hexdigest()[:16], 16)


def _split(units, seed, salt, cuts, names):
    """Group-hash split: exact rational rule (authoritative) and float rule, with disagreement count."""
    m, dis = {}, 0
    for g in np.unique(units).tolist():
        n = _u_int(seed, salt, g)
        x, f = Fraction(n, 2 ** 64), n / 2.0 ** 64
        kx = next((names[i] for i, c in enumerate(cuts) if x < c), names[-1])
        kf = next((names[i] for i, c in enumerate(cuts) if f < float(c)), names[-1])
        dis += int(kx != kf)
        m[g] = kx
    return np.array([m[g] for g in units.tolist()], dtype=object), dis


class Data:
    """Independent reconstruction of roles, subroles and the 83-column inputs (no smf/jcv loader)."""

    def __init__(self):
        self.src_sha = sha_file(SRC)
        z = np.load(SRC, allow_pickle=False)
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
        rsub = np.array([""] * n, dtype=object)
        lab, d2 = _split(unit[df], 20261004, "critic", [Fraction(7, 10), Fraction(17, 20)], list(RGJ_SUBROLES))
        rsub[df] = lab
        new = np.array([""] * n, dtype=object)
        lab, d3 = _split(unit[df], 20261005, "assess", [Fraction(1, 5)], [ASSESS, FIT])
        new[df] = lab
        for r in ("HEAD_VALIDATION", "AUDIT_FIT", "INNER_SELECTION"):
            new[rgj == r] = r
        sub = np.array([""] * n, dtype=object)
        nf = new == FIT
        lab, d4 = _split(unit[nf], 20261005, "critic", [Fraction(7, 10), Fraction(17, 20)], list(SUBROLES))
        sub[nf] = lab
        self.disagree = {"rgj_dev_split": d1, "rgj_critic_split": d2, "new_assess_split": d3, "new_critic_split": d4}
        self.full = {"old": old, "rgj": rgj.astype(str), "rsub": rsub.astype(str), "new": new.astype(str),
                     "sub": sub.astype(str), "unit": unit, "row_id": rid}
        keep = np.flatnonzero(new != "")
        self.keep = keep
        self.row_id = rid[keep]
        self.unit = unit[keep]
        self.role = new[keep].astype(str)
        self.sub = sub[keep].astype(str)
        self.mask = {r: self.role == r for r in NEW_ROLES}
        self.mask.update({r: self.sub == r for r in SUBROLES})
        # numeric refit: invert the admitted normalisation, round, re-standardise on NEW_DEFENSE_FIT (population sd)
        adm = jload(ADMISSION)
        Xk = X[keep].copy()
        self.numeric = {}
        nfk = self.mask[FIT]
        for c in NUMERIC:
            j = self.feature_names.index(c)
            mu, sd = adm["numeric_norm"][c]
            inv = X[:, j].astype(np.float64) * sd + mu
            r = np.round(inv)
            rk = r[keep]
            m, s = float(rk[nfk].mean()), float(rk[nfk].std())
            Xk[:, j] = ((rk - m) / s).astype(np.float32)
            self.numeric[c] = {"inversion_max_abs_err": float(np.abs(inv - r).max()), "mean_new_fit": m, "sd_new_fit": s}
        self.X = np.ascontiguousarray(Xk, dtype=np.float32)
        self._sealed = None
        self._unsealed = None
        self.unseal_log = []

    def labels(self, unseal=False, why=""):
        """Labels on kept rows. NEW_DEVELOPMENT_ASSESSMENT labels are masked to -1 unless unseal=True, which callers
        only pass after the EVALUATION_LOCK verification (logged)."""
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


def check_roles(D: Data):
    man = jload(RES / "ROLE_MANIFEST.json")
    out = {"source_sha256_ok": D.src_sha == SRC_SHA, "float_vs_exact_disagreements": D.disagree}
    f = D.full
    rec = lambda ix: {"rows": int(len(ix)), "groups": int(len(np.unique(f["unit"][ix]))),
                      "row_id_sha256": rowid_hash(f["row_id"][ix]), "group_id_set_sha256": unit_set_hash(f["unit"][ix])}
    mine = {r: rec(np.flatnonzero(f["new"] == r)) for r in NEW_ROLES}
    mine.update({r: rec(np.flatnonzero(f["sub"] == r)) for r in SUBROLES})
    pub = dict(man["roles"])
    pub.update(man["new_defense_fit_subroles"])
    keys = ("rows", "groups", "row_id_sha256", "group_id_set_sha256")
    cmp = {r: all(mine[r][k] == pub[r][k] for k in keys) for r in mine}
    out["roles_match_manifest"] = cmp
    out["counts"] = {r: mine[r]["rows"] for r in mine}
    rgj_pub = man["refreshed_roles_recomputed"]["per_role"]
    rgj_mine = {r: rec(np.flatnonzero(f["rgj"] == r)) for r in ("DEFENSE_FIT", "HEAD_VALIDATION",
                                                               "DEVELOPMENT_ASSESSMENT", "AUDIT_FIT", "INNER_SELECTION")}
    rgj_mine.update({r: rec(np.flatnonzero(f["rsub"] == r)) for r in RGJ_SUBROLES})
    out["refreshed_roles_match_manifest"] = {r: all(rgj_mine[r][k] == rgj_pub[r][k] for k in keys) for r in rgj_mine}
    fp = hashlib.sha256("|".join(f"{r}:{mine[r]['row_id_sha256']}" for r in NEW_ROLES + SUBROLES).encode()).hexdigest()
    out["fingerprint_match"] = fp == man.get("role_partition_fingerprint_sha256")
    # disjointness and partitions
    groups = {r: set(np.unique(f["unit"][f["new"] == r]).tolist()) for r in NEW_ROLES}
    out["roles_group_disjoint"] = all(not (groups[a] & groups[b]) for i, a in enumerate(NEW_ROLES) for b in NEW_ROLES[i + 1:])
    out["subroles_partition_fit"] = bool(np.array_equal(np.sort(np.flatnonzero(np.isin(f["sub"], SUBROLES))),
                                                        np.flatnonzero(f["new"] == FIT)))
    dropped = {"refreshed DEVELOPMENT_ASSESSMENT": f["rgj"] == "DEVELOPMENT_ASSESSMENT"}
    for p in ("assessment", "cert", "excluded_exposure", "excluded_dup"):
        dropped[f"old {p}"] = f["old"] == p
    pub_drop = man["dropped_pools"]["row_id_sha256"]
    out["dropped_pools_match_manifest"] = {p: rowid_hash(f["row_id"][m]) == pub_drop.get(p) for p, m in dropped.items()}
    any_drop = np.zeros(len(f["new"]), bool)
    for m in dropped.values():
        any_drop |= m
    out["dropped_rows_receive_no_new_role"] = bool((f["new"][any_drop] == "").all())
    kept_groups = set(np.unique(f["unit"][f["new"] != ""]).tolist())
    out["no_kept_group_in_dropped_pool"] = not (kept_groups & set(np.unique(f["unit"][any_drop]).tolist()))
    D.dropped_row_ids = set(f["row_id"][any_drop].tolist())
    # numeric refit vs manifest
    pc = man["numeric_refit"]["per_column"]
    out["numeric_refit_match"] = {c: bool(abs(D.numeric[c]["mean_new_fit"] - pc[c]["mean_new_fit"]) == 0
                                          and abs(D.numeric[c]["sd_new_fit"] - pc[c]["sd_new_fit"]) == 0
                                          and D.numeric[c]["inversion_max_abs_err"] < 0.05) for c in NUMERIC}
    out["feature_names_sha256_match"] = hashlib.sha256("\n".join(D.feature_names).encode()).hexdigest() == \
        man["permitted_columns"]["feature_names_sha256"]
    ok = (out["source_sha256_ok"] and all(cmp.values()) and all(out["refreshed_roles_match_manifest"].values())
          and out["fingerprint_match"] and out["roles_group_disjoint"] and out["subroles_partition_fit"]
          and all(out["dropped_pools_match_manifest"].values()) and out["dropped_rows_receive_no_new_role"]
          and out["no_kept_group_in_dropped_pool"] and all(out["numeric_refit_match"].values())
          and out["feature_names_sha256_match"] and sum(D.disagree.values()) == 0)
    return res("PASS" if ok else "FAIL", **out)


# ------------------------------------------------------------------------------------------------ release replay
def load_state(ud: Path):
    p = ud / "model.pt"
    if not p.exists():
        return None
    sd = torch.load(p, weights_only=True, map_location="cpu")
    if isinstance(sd, dict) and "enc.0.0.weight" in sd:
        return sd
    if isinstance(sd, dict):
        for k in ("model", "state_dict", "theta"):
            if isinstance(sd.get(k), dict) and "enc.0.0.weight" in sd[k]:
                return sd[k]
    return None


def encode(sd, i, X):
    """Recipient i encoder: Linear(83,64)-ReLU-Linear(64,64)-ReLU-Linear(64,16), float32 as stored."""
    h = torch.from_numpy(X)
    with torch.no_grad():
        for j, act in ((0, True), (2, True), (4, False)):
            h = torch.nn.functional.linear(h, sd[f"enc.{i}.{j}.weight"], sd[f"enc.{i}.{j}.bias"])
            if act:
                h = torch.relu(h)
    return h.numpy().astype(np.float64)


def centred_logits(head, r):
    d = head.decision_function(r)
    if d.ndim == 1:
        d = np.stack([np.zeros_like(d), d], 1)
    return d - d.mean(1, keepdims=True)


def maxdiff(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max()) if a.shape == b.shape else float("inf")


def sha_arrays(*arrays) -> str:
    """dtype + shape + bytes fingerprint (the convention the FARE provenance records publish)."""
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def leace_apply(ud: Path, i: int, r_u):
    """Official LEACE eraser applied with its saved map: x - ((x - mean_x) @ proj_right^T) @ proj_left^T."""
    z = np.load(ud / f"leace_{i}" / "leace_map.npz", allow_pickle=False)
    meta = jload(ud / f"leace_{i}" / "leace_map.json")
    delta = r_u - z["mean_x"]
    return r_u - (delta @ z["proj_right"].T) @ z["proj_left"].T, z, meta


def head_checks(D, name, i, h, r_rel, y, info, fails, refit, head_rec):
    """Scaler moments == NEW_DEFENSE_FIT rows; optional exact refit; HEAD_VALIDATION log-loss table and C argmin."""
    sc = h[0] if hasattr(h, "__getitem__") else None
    if isinstance(sc, StandardScaler):
        nseen = int(np.asarray(sc.n_samples_seen_).max())
        info["head_scaler_n_seen"] = nseen
        info["head_scaler_mean_matches_fit_rows"] = bool(
            nseen == int(D.mask[FIT].sum()) and np.allclose(sc.mean_, r_rel[D.mask[FIT]].mean(0), rtol=1e-9, atol=1e-12))
        if not info["head_scaler_mean_matches_fit_rows"]:
            fails.append(f"head_{i} fit rows")
    if refit and isinstance(sc, StandardScaler):
        lr = h[-1]
        q = make_pipeline(StandardScaler(), LogisticRegression(C=lr.C, max_iter=lr.max_iter))
        q.fit(r_rel[D.mask[FIT]], y[D.mask[FIT]])
        info["head_refit_coef_max_abs_diff"] = maxdiff(q[-1].coef_, lr.coef_)
        info["head_refit_intercept_max_abs_diff"] = maxdiff(q[-1].intercept_, lr.intercept_)
        if info["head_refit_coef_max_abs_diff"] > 1e-6:
            fails.append(f"head_{i} refit")
        hv = D.mask["HEAD_VALIDATION"]
        if head_rec:
            ll = log_loss(y[hv], h.predict_proba(r_rel[hv]), labels=lr.classes_)
            tab = {float(t["C"]): t["defense_val_log_loss"] for t in head_rec["table"]}
            info["head_validation_ll_matches_table"] = bool(abs(ll - tab.get(float(lr.C), np.inf)) < 1e-9)
            best = min(tab.items(), key=lambda kv: (kv[1], kv[0]))[0]
            info["selected_C_is_table_argmin"] = bool(best == float(head_rec["selected_C"]) == float(lr.C))
            if not (info["head_validation_ll_matches_table"] and info["selected_C_is_table_argmin"]):
                fails.append(f"head_{i} C selection")


def replay_fare(D, name, rel, refit, L):
    ud = unit_dir(name)
    rec = unit_rec(name) or {}
    i = int(rec.get("purpose", 0))
    out = {"unit": name, "kind": "fare", "purpose": i,
           "label_like_keys": sorted(k for k in rel.files if k in ("sex", "race", "y_income", "y_occ", "y", "labels"))}
    fails = []
    out["row_id_equals_kept_rows"] = bool(np.array_equal(rel["row_id"], D.row_id))
    out["dropped_row_ids_present"] = int(len(set(rel["row_id"].tolist()) & D.dropped_row_ids))
    cells, r = rel["cells"], rel["r"]
    onehot = r.shape[1] == int(rec.get("n_cells", r.shape[1])) and np.array_equal(r, np.eye(r.shape[1])[cells])
    out["features_are_one_hot_cells"] = bool(onehot)
    prov = (rec.get("provenance") or {}).get("fare_rec") or {}
    fit = D.mask[FIT]
    out["fare_fit_rows_fingerprint_matches_NEW_DEFENSE_FIT"] = (
        prov.get("fit_rows_sha256") == sha_arrays(np.ascontiguousarray(D.X[fit], dtype=np.float64))) if prov else None
    out["fare_fit_cell_counts_match"] = (list(np.bincount(cells[fit], minlength=r.shape[1])) == prov.get("n_fit_per_cell")) if prov else None
    if not onehot or out["fare_fit_rows_fingerprint_matches_NEW_DEFENSE_FIT"] is False or out["fare_fit_cell_counts_match"] is False:
        fails.append("fare features/fit rows")
    if not out["row_id_equals_kept_rows"] or out["dropped_row_ids_present"] or out["label_like_keys"]:
        fails.append("rows/keys")
    h = joblib.load(ud / "head.joblib")
    info = {"features": "FARE tree not replayed (official implementation not importable here); one-hot cells checked"}
    c, p, hard = centred_logits(h, r), h.predict_proba(r), h.predict(r)
    info["centred_logits_max_abs_diff"] = maxdiff(c, rel["c"])
    info["prob_max_abs_diff"] = maxdiff(p, rel["p"])
    info["hard_mismatches"] = int((hard != rel["hard"]).sum())
    if info["centred_logits_max_abs_diff"] > 1e-4 or info["prob_max_abs_diff"] > 1e-6 or info["hard_mismatches"]:
        fails.append("c/p/hard")
    y = L["y_income"] if i == 0 else L["y_occ"]
    head_checks(D, name, 0, h, r, y, info, fails, refit, rec.get("head"))
    out[f"recipient_{i + 1}"] = info
    out["status"] = "FAIL" if fails else "PASS"
    out["failures"] = fails
    return out, rel


def replay_release(D: Data, name: str, refit: bool, L):
    ud = unit_dir(name)
    out = {"unit": name}
    rel = np.load(ud / "release.npz", allow_pickle=False)
    if "r" in rel.files and "cells" in rel.files:
        return replay_fare(D, name, rel, refit, L)
    keys = set(rel.files)
    out["extra_keys"] = sorted(keys - RELEASE_KEYS)
    out["label_like_keys"] = sorted(k for k in keys if k in ("sex", "race", "y_income", "y_occ", "y", "labels"))
    rid = rel["row_id"]
    out["row_id_equals_kept_rows"] = bool(np.array_equal(rid, D.row_id))
    out["dropped_row_ids_present"] = int(len(set(rid.tolist()) & D.dropped_row_ids))
    heads = []
    for i in (0, 1):
        hp = ud / f"head_{i}.joblib"
        heads.append(joblib.load(hp) if hp.exists() else None)
    sd = load_state(ud)
    fails = []
    if not out["row_id_equals_kept_rows"] or out["dropped_row_ids_present"] or out["label_like_keys"]:
        fails.append("rows/keys")
    rec = unit_rec(name) or {}
    is_leace = (ud / "leace_0").exists()
    if is_leace:
        out["kind"] = "leace"
        src = rec.get("source_unit")
        out["leace_source_unit"] = src
        if src and unit_done(src):
            out["leace_model_equals_source_model"] = sha_file(ud / "model.pt") == sha_file(unit_dir(src) / "model.pt")
            if not out["leace_model_equals_source_model"]:
                fails.append("LEACE source model")
    for i, (rk, ck, pk, hk) in enumerate((("r1", "c1", "p1", "hard1"), ("r2", "c2", "p2", "hard2"))):
        r_rel = rel[rk]
        info = {}
        if sd is not None:
            r = encode(sd, i, D.X)
            if is_leace:
                r_u = r
                r, z, meta = leace_apply(ud, i, r_u)
                info["leace_mean_x_equals_fit_row_mean"] = bool(np.allclose(z["mean_x"], r_u[D.mask[FIT]].mean(0), rtol=1e-9, atol=1e-12))
                info["leace_fit_rows_are_NEW_DEFENSE_FIT"] = meta.get("fit_row_ids_sha256") == rowid_hash(D.row_id[D.mask[FIT]])
                info["leace_inference_uses_labels"] = meta.get("inference_uses_labels")
                if not (info["leace_mean_x_equals_fit_row_mean"] and info["leace_fit_rows_are_NEW_DEFENSE_FIT"]):
                    fails.append(f"LEACE map {i}")
            info["features_max_abs_diff"] = maxdiff(r, r_rel)
            info["features_bit_exact"] = bool(np.array_equal(r, r_rel))
            if info["features_max_abs_diff"] > 1e-5:
                fails.append(f"{rk} features")
        else:
            r = r_rel
            info["features"] = "NOT_REPLAYED (no MLP state in model.pt)"
        h = heads[i]
        if h is not None:
            c = centred_logits(h, r)
            p = h.predict_proba(r)
            hard = h.predict(r)
            info["centred_logits_max_abs_diff"] = maxdiff(c, rel[ck])
            info["prob_max_abs_diff"] = maxdiff(p, rel[pk])
            info["hard_mismatches"] = int((hard != rel[hk]).sum())
            if info["centred_logits_max_abs_diff"] > 1e-4 or info["prob_max_abs_diff"] > 1e-6 or info["hard_mismatches"]:
                fails.append(f"{ck}/{pk}/{hk}")
            y = L["y_income"] if i == 0 else L["y_occ"]
            head_checks(D, name, i, h, r_rel, y, info, fails, refit, (rec.get("heads") or {}).get(str(i)))
        out[f"recipient_{i + 1}"] = info
    out["status"] = "FAIL" if fails else "PASS"
    out["failures"] = fails
    return out, rel


# ------------------------------------------------------------------------------------------------ utility / gates
def task_counts(D, L, rel, role="INNER_SELECTION", purpose=None):
    """Correct-decision and constant-predictor counts on `role` rows. Neural releases carry both tasks (hard1, hard2);
    a FARE purpose release carries only its own task (hard)."""
    m = D.mask[role]
    out = {}
    tasks = ((0, "hard1", "y_income"), (1, "hard2", "y_occ"))
    if "hard1" not in rel.files:
        tasks = ((purpose, "hard", "y_income" if purpose == 0 else "y_occ"),)
    for t, hk, yk in tasks:
        y = L[yk]
        maj = int(np.bincount(y[D.mask[FIT]]).argmax())
        out[t] = {"k": int((rel[hk][m] == y[m]).sum()), "n": int(m.sum()), "k_const": int((y[m] == maj).sum())}
    return out


def unit_counts(D, L, name, cache):
    """Cached task counts of a unit's release, or of a FARE pair (list of two purpose units)."""
    key = tuple(name) if isinstance(name, (list, tuple)) else name
    if key in cache:
        return cache[key]
    if isinstance(name, (list, tuple)):
        parts = [unit_counts(D, L, u, cache) for u in name]
        if any(p is None for p in parts):
            return None
        out = {0: parts[0][0], 1: parts[1][1]}
    else:
        p = unit_dir(name) / "release.npz"
        if not (unit_done(name) and p.exists()):
            return None
        rec = unit_rec(name) or {}
        out = task_counts(D, L, np.load(p, allow_pickle=False), purpose=rec.get("purpose"))
    cache[key] = out
    return out


def gate_margins(cand, ref):
    """Exact (Fraction) and float margins of G1-G3 for both tasks. cand/ref = task_counts dicts."""
    mg, mf = {}, {}
    for t in (0, 1):
        n = cand[t]["n"]
        a = Fraction(cand[t]["k"], n)
        au = Fraction(ref[t]["k"], ref[t]["n"])
        c = Fraction(cand[t]["k_const"], n)
        mg[t] = {"G1": a - au + Fraction(1, 100), "G2": a - Fraction(4, 5) * au - Fraction(1, 5) * c,
                 "G3": a - c - Fraction(3, 100)}
        af, auf, cf = cand[t]["k"] / n, ref[t]["k"] / ref[t]["n"], cand[t]["k_const"] / n
        mf[t] = {"G1": af - (auf - 0.01), "G2": af - 0.8 * auf - 0.2 * cf, "G3": af - cf - 0.03}
    return mg, mf


def margins_from_utility(u, uref):
    """Float margins from record utility {t: {acc, const_acc}} (used when no release is available)."""
    out = {}
    for t in (0, 1):
        a, au, c = u[str(t)]["acc"], uref[str(t)]["acc"], u[str(t)]["const_acc"]
        out[t] = {"G1": a - (au - 0.01), "G2": a - 0.8 * au - 0.2 * c, "G3": a - c - 0.03}
    return out


def shortfall(m):
    return sum(max(0, -v) for t in m for v in m[t].values())


def feasible(m):
    return all(v >= 0 for t in m for v in m[t].values())


# ------------------------------------------------------------------------------------------------ inner records
def inner_consistency(D, name, rec, rel, L, lr_spot: bool, counts=None):
    R = rec["recovery"]
    out = {"of": rec.get("of")}
    fails = []
    for w in R["auc"]:
        bank = R["selection"][w].get("bank") or R["tables"][w]
        if max(b["inner_auc"] for b in bank) != R["auc"][w]:
            fails.append(f"auc.{w} != bank max")
        if w == "pair":
            out["coalition_bank_size"] = len(bank)
    if R.get("worse_local") is not None and R["worse_local"] != max(R["auc"]["v1"], R["auc"]["v2"]):
        fails.append("worse_local")
    if R.get("n_fit") is not None:
        out["n_fit_is_AUDIT_FIT"] = R["n_fit"] == int(D.mask["AUDIT_FIT"].sum())
        out["n_select_is_INNER"] = R["n_select"] == int(D.mask["INNER_SELECTION"].sum())
        if not (out["n_fit_is_AUDIT_FIT"] and out["n_select_is_INNER"]):
            fails.append("attacker roles")
    if counts is not None:
        U_ = rec.get("utility", {})
        out["utility_recomputed"] = {t: {"acc": c["k"] / c["n"], "const": c["k_const"] / c["n"]} for t, c in counts.items()}
        for t, c in counts.items():
            u = U_.get(str(t)) if str(t) in U_ else (U_ if "acc" in U_ else None)
            if u is None or abs(u["acc"] - c["k"] / c["n"]) > 1e-12 or abs(u["const_acc"] - c["k_const"] / c["n"]) > 1e-12:
                fails.append(f"utility task {t}")
    if rel is not None and "r1" in rel.files:
        if lr_spot:
            s = L["sex"]
            fa, se = D.mask["AUDIT_FIT"], D.mask["INNER_SELECTION"]
            V = {"v1": np.hstack([rel["r1"], rel["c1"]]), "v2": np.hstack([rel["r2"], rel["c2"]])}
            V["pair"] = np.hstack([V["v1"], V["v2"]])
            spot = {}
            for w in ("v1", "v2", "pair"):
                tab = [t for t in R["tables"][w] if t["attacker"] == "LR_C1"]
                if not tab:
                    continue
                q = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(V[w][fa], s[fa])
                auc = roc_auc_score(s[se], q.predict_proba(V[w][se])[:, 1])
                spot[w] = {"mine": auc, "record": tab[0]["inner_auc"], "abs_diff": abs(auc - tab[0]["inner_auc"])}
            out["LR_C1_spot_check"] = spot
            out["LR_C1_spot_max_abs_diff"] = max((v["abs_diff"] for v in spot.values()), default=None)
    out["failures"] = fails
    out["status"] = "FAIL" if fails else "PASS"
    return out


# ------------------------------------------------------------------------------------------------ gradients
def controller_replay(ctrl, b, ref_auc, feedback: bool):
    """Replay the registered additive rule from logged measurements. Returns (expected applied w_after per epoch,
    expected hypothetical w per epoch, failures)."""
    fails = []
    ms = sorted(ctrl, key=lambda c: c["epoch"])
    epochs = [c["epoch"] for c in ms]
    if epochs != list(range(21)):
        fails.append(f"measurement epochs {epochs}")
    w_app, w_hyp = np.array([1.0, 1.0]), np.array([1.0, 1.0])
    exp_app, exp_hyp, alloc = {}, {}, {}
    for c in ms:
        e = c["epoch"]
        auc = np.asarray(c["auc"], np.float64)
        if np.any(np.abs(np.asarray(c["b"]) - b) > 0):
            fails.append(f"e{e}: b differs from calib")
        v = auc - b
        if np.any(np.abs(np.asarray(c["v"]) - v) > 1e-12):
            fails.append(f"e{e}: v != auc - b")
        diag = bool(c.get("diagnostic_only"))
        if diag != (e == 20):
            fails.append(f"e{e}: diagnostic_only={diag}")
        if e == 0:
            if not c.get("reused_reference") or np.any(auc != ref_auc):
                fails.append("e0: reference receipt not reused")
        step = np.clip(v / CTRL_DIV, -1.0, 1.0)
        if not diag:
            w_hyp = np.clip(w_hyp + step, W_MIN, W_MAX)
            if feedback:
                if np.any(np.abs(np.asarray(c["w_before"]) - w_app) > 1e-12):
                    fails.append(f"e{e}: w_before")
                w_app = np.clip(w_app + step, W_MIN, W_MAX)
        exp_app[e], exp_hyp[e] = w_app.copy(), w_hyp.copy()
        rms = math.sqrt((w_app[0] ** 2 + w_app[1] ** 2) / 2)
        alloc[e] = w_app / rms
        if np.any(np.abs(np.asarray(c["w_after"]) - w_app) > 1e-12):
            fails.append(f"e{e}: w_after")
        if np.any(np.abs(np.asarray(c["w_hypothetical"]) - w_hyp) > 1e-12):
            fails.append(f"e{e}: w_hypothetical")
        if np.any(np.abs(np.asarray(c["allocation"]) - alloc[e]) > 1e-12):
            fails.append(f"e{e}: allocation")
    return exp_app, exp_hyp, alloc, fails


def activation_status(base, exp_w, alloc):
    applied = [e for e in range(20) if e in exp_w]
    asym = any(abs(alloc[e][0] - alloc[e][1]) > ASYM_THR for e in applied)
    moved = any(np.any(exp_w[e] != 1.0) for e in applied)
    first_asym = next((e for e in applied if abs(alloc[e][0] - alloc[e][1]) > ASYM_THR), None)
    if base == "local":
        st = "ACTIVE" if asym else "INACTIVE_OR_ALIAS"
    else:
        st = "ACTIVE_ASYMMETRIC" if asym else ("ACTIVE_COMMON_ONLY" if moved else "INACTIVE")
    return st, first_asym, moved


def gradient_check(rec, s_by_epoch=None, tol=1e-6):
    """Per-epoch ratio identities for one normalized run. s_by_epoch: {e: (s1, s2)} (Phase B) or None (s = 1)."""
    d = rec["diag"]
    rho = float(rec["rho"])
    fails, flags = [], []
    tot = {"n": 0, "zero": [0, 0], "cap": [0, 0], "clip": 0}
    exp_real = [0.0, 0.0]
    real = [0.0, 0.0]
    cond_sum = [0.0, 0.0]
    cap_deficit = [0.0, 0.0]
    rms_dev = 0.0
    for E in d["epochs"]:
        e, n = E["epoch"], E["n_steps"]
        s = (1.0, 1.0) if s_by_epoch is None else s_by_epoch[e]
        if s_by_epoch is None and (E.get("w") not in (None, [1.0, 1.0]) or E.get("alloc") not in (None, [1.0, 1.0])):
            fails.append(f"e{e}: Phase A weights/alloc not 1")
        if s_by_epoch is not None and np.any(np.abs(np.asarray(E["alloc"]) - np.asarray(s)) > 1e-12):
            fails.append(f"e{e}: logged alloc != replayed allocation")
        tot["n"] += n
        tot["clip"] += E["clip"]
        if E["clip"] == 0 and E.get("kappa_min", 1.0) != 1.0:
            fails.append(f"e{e}: kappa<1 without clip")
        rm = []
        for i in (1, 2):
            z, c = E[f"zero_{i}"], E[f"cap_{i}"]
            tot["zero"][i - 1] += z
            tot["cap"][i - 1] += c
            target = rho * s[i - 1]
            rmean, rreal = E[f"ratio_mean_{i}"], E[f"realized_ratio_mean_{i}"]
            nz = n - z
            if nz > 0:
                if rmean is None or not math.isfinite(rmean):
                    fails.append(f"e{e} r{i}: ratio undefined with nonzero steps")
                    continue
                if c == 0 and abs(rmean - target) > tol * max(1.0, target):
                    fails.append(f"e{e} r{i}: conditional ratio {rmean:.8f} != rho*s {target:.8f}")
                if c > 0:
                    if rmean > target * (1 + tol):
                        fails.append(f"e{e} r{i}: capped ratio above target")
                    cap_deficit[i - 1] += (target - rmean) * nz
                if abs(rreal - rmean * nz / n) > 1e-9 * max(1.0, target):
                    fails.append(f"e{e} r{i}: realized != conditional*(n-zero)/n")
                cond_sum[i - 1] += rmean * nz
                rm.append(rmean)
            else:
                if rreal not in (0, 0.0) and rreal is not None:
                    fails.append(f"e{e} r{i}: realized nonzero with all-zero steps")
            exp_real[i - 1] += target * nz
            real[i - 1] += (rreal or 0.0) * n
        if s_by_epoch is not None and len(rm) == 2 and E["cap_1"] == 0 and E["cap_2"] == 0:
            rms_dev = max(rms_dev, abs(math.sqrt((rm[0] ** 2 + rm[1] ** 2) / 2) - rho))
    if tot["n"] != d["encoder_updates"]:
        fails.append("n_steps sum != encoder_updates")
    if list(tot["zero"]) != list(d["zero_events"]) or list(tot["cap"]) != list(d["cap_hits"]) or tot["clip"] != d["clip_hits"]:
        fails.append("unit totals (zero/cap/clip) != epoch sums")
    N = max(tot["n"], 1)
    unit = {}
    for i in (0, 1):
        r_unit, e_unit = real[i] / N, exp_real[i] / N
        unit[f"realized_{i + 1}"] = r_unit
        unit[f"expected_realized_from_zero_events_{i + 1}"] = e_unit
        unit[f"cap_deficit_{i + 1}"] = cap_deficit[i] / N
        unit[f"zero_fraction_{i + 1}"] = tot["zero"][i] / N
        unexplained = r_unit - (e_unit - cap_deficit[i] / N)
        unit[f"unexplained_{i + 1}"] = unexplained
        if abs(unexplained) > 1e-6 * max(1.0, rho):
            flags.append(f"recipient {i + 1}: realized ratio departs from rho beyond zero/cap events")
        if tot["zero"][i] / N > 0.10:
            flags.append(f"recipient {i + 1}: zero-direction steps {tot['zero'][i]}/{N}")
        if tot["cap"][i]:
            flags.append(f"recipient {i + 1}: cap active {tot['cap'][i]} steps (strength not exactly matched)")
    unit["realized_rms"] = math.sqrt((unit["realized_1"] ** 2 + unit["realized_2"] ** 2) / 2)
    unit["max_epoch_rms_deviation_no_cap"] = rms_dev if s_by_epoch is not None else None
    if s_by_epoch is not None and rms_dev > tol * max(1.0, rho):
        fails.append(f"RMS identity off by {rms_dev}")
    unit.update(rho=rho, zero_events=tot["zero"], cap_hits=tot["cap"], clip_hits=tot["clip"], steps=tot["n"])
    st = "FAIL" if fails else ("WARN" if any("departs" in f for f in flags) else "PASS")
    return {"status": st, "failures": fails[:20], "n_failures": len(fails), "flags": flags, **unit}


def refit_receipt_counts(rec):
    """Per-bank continued + restart updates of refits at epochs 0-16 (non-diagnostic)."""
    per, tot, diag_tot, epochs = {}, 0, 0, []
    for rf in rec["diag"]["refits"]:
        epochs.append((rf["epoch"], bool(rf["diagnostic_only"])))
        for bank, rc in rf["receipts"].items():
            for kind, kr in rc["kinds"].items():
                u = int(kr["continued"]["updates"]) + int(kr["restart"]["updates"])
                if rf["diagnostic_only"]:
                    diag_tot += u
                elif rf["epoch"] in REFIT_EPOCHS_MATCHED:
                    per[f"{bank}|{kind}"] = per.get(f"{bank}|{kind}", 0) + u
                    tot += u
    return per, tot, diag_tot, epochs


def unit_times(name):
    """(complete time, estimated start = complete - wall_s) in UTC."""
    p = UNITS / name / "COMPLETE.json"
    if not p.exists():
        return None, None
    t = utc(p.stat().st_mtime)
    rec = unit_rec(name) or {}
    w = rec.get("wall_s") or (rec.get("diag") or {}).get("wall_s")
    return t, (datetime.fromtimestamp(t.timestamp() - float(w), timezone.utc) if w else None)


def check_gradients(inv):
    runs = sorted(n for n in inv if (n.startswith("run__A__") or n.startswith("run__B__")) and unit_done(n))
    if not runs:
        return res("PENDING", reason="no normalized run receipts"), {}
    per, matched, ctrl_info = {}, {}, {}
    calib = {k: unit_rec(f"calib__s{k}") for k in SEEDS}
    for rn in runs:
        rec = unit_rec(rn)
        s_by = None
        if rec["stage"] == "B":
            k = rec["seed"]
            cal = calib.get(k)
            if cal is None:
                per[rn] = res("PENDING", reason="calib receipt missing")
                continue
            b = np.asarray(cal["b"], np.float64)
            ref_auc = np.array([cal["receipt"]["v1"]["calib_auc"], cal["receipt"]["v2"]["calib_auc"]])
            fb = bool(rec["spec"].get("feedback"))
            w_app, w_hyp, alloc, cf = controller_replay(rec["diag"]["controller"], b, ref_auc, fb)
            s_by = {e: tuple(alloc[e]) for e in range(20)} if fb else {e: (1.0, 1.0) for e in range(20)}
            for E in rec["diag"]["epochs"]:
                e = E["epoch"]
                if np.any(np.abs(np.asarray(E["w"]) - (w_app[e] if fb else np.ones(2))) > 1e-12):
                    cf.append(f"epoch {e}: training weights != w_after of measurement {e}")
            ctrl_info[rn] = {"feedback": fb, "w_app": w_app, "w_hyp": w_hyp, "alloc": alloc, "failures": cf,
                             "base": rec["spec"]["base"], "seed": k, "rho": rec["rho"], "arm": rec["arm"],
                             "auc_trace": [c["auc"] for c in sorted(rec["diag"]["controller"], key=lambda c: c["epoch"])]}
        g = gradient_check(rec, s_by)
        g["compute_critic_updates"] = (rec["diag"]["critic_online_updates"] + rec["diag"]["critic_matched_extra_updates"]
                                       + rec["diag"]["critic_refit_updates"])
        if rec["schedule"] == "REFRESHED":
            pb, tot, dtot, eps = refit_receipt_counts(rec)
            g["refit_epochs"] = eps
            g["refit_receipt_total_e0_16"] = tot
            ok = [e for e, dg in eps if not dg] == list(REFIT_EPOCHS_MATCHED) and [e for e, dg in eps if dg] == [20]
            if not ok or tot != rec["diag"]["critic_refit_updates"]:
                g["status"] = "FAIL"
                g["failures"].append("refit schedule or critic_refit_updates != receipt sum")
        if rec["schedule"] == "ONLINE_MATCHED":
            tn = rec.get("matched_template")
            trec = unit_rec(tn) if tn else None
            m = {"template": tn}
            if trec is None:
                m["status"] = "FAIL"
                m["reason"] = "template receipt missing"
            else:
                pb, tot, _, _ = refit_receipt_counts(trec)
                same = (trec["seed"] == rec["seed"] and trec["rho"] == rec["rho"] and
                        trec["spec"]["base"] == rec["spec"]["base"] and trec["schedule"] == "REFRESHED" and
                        trec.get("stage") == rec.get("stage"))
                m["template_pairing_ok"] = same
                m["extra_updates"] = rec["diag"]["critic_matched_extra_updates"]
                m["template_receipt_sum"] = tot
                m["per_bank_equal"] = (rec["diag"].get("matched_counts") == pb)
                t_done, _ = unit_times(tn)
                _, start = unit_times(rn)
                m["template_completed_before_matched_start"] = bool(t_done and start and t_done.timestamp() <= start.timestamp() + 2)
                m["status"] = "PASS" if (same and m["extra_updates"] == tot and m["per_bank_equal"]
                                         and m["template_completed_before_matched_start"]) else "FAIL"
            matched[rn] = m
        per[rn] = g
    cap_units = sorted(n for n, g in per.items() if any(g.get("cap_hits", [0, 0])))
    zero_heavy = sorted(n for n, g in per.items() if max(g.get("zero_fraction_1", 0), g.get("zero_fraction_2", 0)) > 0.10)
    clip_units = sorted(n for n, g in per.items() if g.get("clip_hits"))
    unexplained = sorted(n for n, g in per.items() if g.get("status") == "WARN")
    st = worst(*[g["status"] for g in per.values()], *[m["status"] for m in matched.values()])
    short = {n: {"rho": g["rho"], "realized_rms": round(g["realized_rms"], 4), "zero_events": g["zero_events"],
                 "cap_hits": g["cap_hits"]} for n, g in per.items() if "realized_rms" in g and g["realized_rms"] < 0.9 * g["rho"]}
    summary = {"runs_checked": len(per), "failing_runs": sorted(n for n, g in per.items() if g["status"] == "FAIL"),
               "runs_with_realized_rms_below_90pct_of_rho_explained_by_zero_or_cap": short,
               "runs_with_unexplained_ratio_departure": unexplained, "runs_with_cap_events": cap_units,
               "runs_with_clip_events": clip_units, "runs_with_zero_direction_fraction_over_10pct": zero_heavy,
               "online_matched": matched}
    return res(st, **summary, per_run=per), ctrl_info


def compare_gradient_csv(per_run):
    p = RES / "GRADIENT_MATCHING.csv"
    if not p.exists():
        return res("PENDING", reason="GRADIENT_MATCHING.csv not written")
    rows = {r["unit"]: r for r in csv.DictReader(open(p))}
    bad, n = [], 0
    for name, g in per_run.items():
        r = rows.get(name)
        if r is None or "realized_1" not in g:
            continue
        n += 1
        rec = unit_rec(name)
        E = rec["diag"]["epochs"]
        for i in (1, 2):
            m_cond = float(np.mean([e[f"ratio_mean_{i}"] for e in E]))
            m_real = float(np.mean([e[f"realized_ratio_mean_{i}"] for e in E]))
            for col, v in ((f"ratio_pre_clip_mean_{i}", m_cond), (f"realized_ratio_mean_{i}", m_real)):
                if r.get(col) not in (None, "") and abs(float(r[col]) - v) > 1e-5 * max(1, abs(v)):
                    bad.append((name, col, r[col], v))
            if int(float(r[f"zero_events_{i}"])) != g["zero_events"][i - 1] or int(float(r[f"cap_hits_{i}"])) != g["cap_hits"][i - 1]:
                bad.append((name, f"zero/cap_{i}"))
        if int(float(r["clip_hits"])) != g["clip_hits"]:
            bad.append((name, "clip_hits"))
    missing = sorted(set(per_run) - set(rows))
    return res("FAIL" if bad else ("WARN" if missing else "PASS"), rows_compared=n, mismatches=bad[:30],
               runs_missing_from_csv=missing)


# ------------------------------------------------------------------------------------------------ controller
def check_controller(ctrl_info):
    if not ctrl_info:
        return res("PENDING", reason="no Phase B receipts")
    out, arms = {}, {}
    for rn, ci in sorted(ctrl_info.items()):
        fb, base = ci["feedback"], ci["base"]
        w = ci["w_app"] if fb else ci["w_hyp"]
        alloc_h = {e: w[e] / math.sqrt((w[e][0] ** 2 + w[e][1] ** 2) / 2) for e in w}
        st, first, moved = activation_status(base, w, alloc_h if not fb else ci["alloc"])
        arms[rn] = {"arm": ci["arm"], "seed": ci["seed"], "rho": ci["rho"], "feedback": fb,
                    ("activation" if fb else "hypothetical_activation"): st, "first_asymmetric_update_epoch": first,
                    "final_weights": (w[20] if 20 in w else w[max(w)]).tolist(),
                    "max_abs_s1_minus_s2": max(abs(a[0] - a[1]) for e, a in (ci["alloc"] if fb else alloc_h).items() if e < 20),
                    "replay_failures": ci["failures"][:10], "status": "FAIL" if ci["failures"] else "PASS"}
    # calib receipts and targets
    calib = {}
    for k in SEEDS:
        c = unit_rec(f"calib__s{k}")
        if c is None:
            calib[k] = res("PENDING")
            continue
        ra = [c["receipt"]["v1"]["calib_auc"], c["receipt"]["v2"]["calib_auc"]]
        bb = [max(TARGET_FLOOR, a - TARGET_OFFSET) for a in ra]
        sel_ok = all(c["receipt"][v]["selected"] in c["receipt"][v]["table"] and
                     c["receipt"][v]["calib_auc"] == c["receipt"][v]["table"][c["receipt"][v]["selected"]]["calib_auc"]
                     for v in ("v1", "v2"))
        val_ok = all(c["receipt"][v]["table"][c["receipt"][v]["selected"]]["val_auc_oriented"] ==
                     max(t["val_auc_oriented"] for t in c["receipt"][v]["table"].values()) for v in ("v1", "v2"))
        ok = bb == list(c["b"]) and [a < TARGET_FLOOR + TARGET_OFFSET for a in ra] == list(c["floor_active"])
        calib[k] = res("PASS" if ok and sel_ok and val_ok else "FAIL", b=c["b"], reference=c["reference"]["unit"],
                       probe_selected_on_val=val_ok, floor_active=c["floor_active"])
    # alias check: local INACTIVE_OR_ALIAS feedback arms vs twins; twin/feedback trace agreement before divergence
    alias = {}
    for rn, a in arms.items():
        if not a["feedback"]:
            continue
        twin_arm = a["arm"].replace("-F", "-N")
        tw = b_run(a["seed"], twin_arm, a["rho"])
        if tw not in ctrl_info:
            continue
        fu, tu = b_unit(a["seed"], a["arm"], a["rho"]), b_unit(a["seed"], twin_arm, a["rho"])
        entry = {"twin": tw}
        if unit_done(fu) and unit_done(tu):
            sa, sb = load_state(unit_dir(fu)), load_state(unit_dir(tu))
            if sa is not None and sb is not None:
                entry["max_abs_param_diff_vs_twin"] = max(float((sa[k] - sb[k]).abs().max()) for k in sa)
        ta, tb = ctrl_info[rn]["auc_trace"], ctrl_info[tw]["auc_trace"]
        last_same = a["first_asymmetric_update_epoch"] if ctrl_info[rn]["base"] == "local" else 0
        if last_same is None:
            last_same = 20
        entry["probe_traces_equal_through_epoch"] = last_same
        entry["probe_traces_equal_before_divergence"] = all(ta[e] == tb[e] for e in range(last_same + 1))
        if a["activation"] == "INACTIVE_OR_ALIAS":
            d = entry.get("max_abs_param_diff_vs_twin")
            entry["status"] = "PASS" if d == 0.0 else ("PENDING" if d is None else "FAIL")
        else:
            entry["status"] = "PASS" if entry["probe_traces_equal_before_divergence"] else "WARN"
        alias[rn] = entry
    fb_arms = [a for a in arms.values() if a["feedback"]]
    any_active = any(a["activation"] in ("ACTIVE", "ACTIVE_ASYMMETRIC") for a in fb_arms)
    counts = {}
    for a in fb_arms:
        counts.setdefault(a["arm"], {}).setdefault(a["activation"], 0)
        counts[a["arm"]][a["activation"]] += 1
    comp = compare_activation_files(arms, ctrl_info)
    st = worst(*[a["status"] for a in arms.values()], *[c["status"] for c in calib.values()],
               *[e["status"] for e in alias.values()], *[c["status"] for c in comp.values() if c["status"] != "PENDING"])
    return res(st, component_status="ACTIVE_SOMEWHERE" if any_active else ("FEEDBACK_NOT_TESTABLE" if fb_arms else "PENDING"),
               activation_counts=counts, calib=calib, per_run=arms, alias_and_divergence=alias, csv_comparison=comp,
               complete_feedback_runs=len(fb_arms), expected_feedback_runs=18)


def compare_activation_files(arms, ctrl_info):
    """CONTROLLER_SUMMARY.json (status, asymmetric/common-mode counts, max |w - 1|) and the per-measurement
    CONTROLLER_ACTIVATION.csv (auc, b, v, applied or hypothetical w and s) against the replay."""
    out = {}
    mine = {}
    for rn, ci in ctrl_info.items():
        fb = ci["feedback"]
        w = ci["w_app"] if fb else ci["w_hyp"]
        al = {e: w[e] / math.sqrt((w[e][0] ** 2 + w[e][1] ** 2) / 2) for e in w}
        n_asym = sum(abs(al[e][0] - al[e][1]) > ASYM_THR for e in range(20))
        mine[rn] = {"w": w, "s": al, "n_asym": n_asym, "dev": max(float(np.abs(w[e] - 1).max()) for e in range(21)),
                    "status": arms[rn].get("activation") if fb else "TWIN (not applied)"}
    p = RES / "CONTROLLER_SUMMARY.json"
    if p.exists():
        S, bad = jload(p), []
        for rn, m in mine.items():
            t = S.get(rn)
            if t is None:
                bad.append(f"{rn}: missing")
                continue
            if t.get("status") != m["status"]:
                bad.append(f"{rn}: status {t.get('status')} vs {m['status']}")
            if t.get("asymmetric_updates") != m["n_asym"] or t.get("common_mode_updates") != 20 - m["n_asym"]:
                bad.append(f"{rn}: asym/common {t.get('asymmetric_updates')}/{t.get('common_mode_updates')} vs {m['n_asym']}")
            if not _close(t.get("max_abs_weight_deviation"), m["dev"], 1e-9):
                bad.append(f"{rn}: max |w-1|")
        out["CONTROLLER_SUMMARY.json"] = res("FAIL" if bad else "PASS", runs=len(mine), differences=bad[:30])
    else:
        out["CONTROLLER_SUMMARY.json"] = res("PENDING")
    p = RES / "CONTROLLER_ACTIVATION.csv"
    if p.exists():
        bad, n = [], 0
        for r in csv.DictReader(open(p)):
            rn, e = r["unit"], int(r["epoch"])
            m = mine.get(rn)
            if m is None:
                continue
            n += 1
            ci = ctrl_info[rn]
            vals = {"w_1": m["w"][e][0], "w_2": m["w"][e][1], "s_1": m["s"][e][0], "s_2": m["s"][e][1],
                    "auc_1": ci["auc_trace"][e][0], "auc_2": ci["auc_trace"][e][1]}
            for col, v in vals.items():
                if abs(float(r[col]) - v) > 5e-6 * max(1.0, abs(v)):
                    bad.append(f"{rn} e{e} {col} {r[col]} vs {v:.6g}")
            want = "applied" if ci["feedback"] else "hypothetical (twin)"
            if r.get("w_kind") != want:
                bad.append(f"{rn} e{e} w_kind")
        out["CONTROLLER_ACTIVATION.csv"] = res("FAIL" if bad else "PASS", rows_compared=n, differences=bad[:30])
    else:
        out["CONTROLLER_ACTIVATION.csv"] = res("PENDING")
    return out


# ------------------------------------------------------------------------------------------------ selection A
def point(D, L, unit, ref_counts, releases_cache, inner_name=None, of=None):
    rec = unit_rec(inner_name or f"inner__{unit}")
    if rec is None:
        return None
    rel_counts = unit_counts(D, L, of if of is not None else unit, releases_cache)
    auc = rec["recovery"]["auc"]
    p = {"unit": unit, "auc": {k: auc[k] for k in ("v1", "v2", "pair")}, "counts": rel_counts}
    if rel_counts is not None and ref_counts is not None:
        mg, mf = gate_margins(rel_counts, ref_counts)
        p["margins_exact"], p["margins_float"] = mg, mf
        p["feasible"] = feasible(mg)
        p["feasible_float"] = feasible(mf)
        p["shortfall"] = float(shortfall(mg))
        p["shortfall_float"] = shortfall(mf)
    p["worse_local"] = max(auc["v1"], auc["v2"])
    p["mean_local"] = (auc["v1"] + auc["v2"]) / 2
    return p


def replay_selection_A(D, L, cache):
    U20 = {}
    for k in SEEDS:
        u = tl_unit(k, 20)
        if not unit_done(u):
            return None
        U20[k] = task_counts(D, L, np.load(unit_dir(u) / "release.npz", allow_pickle=False))
        cache[u] = U20[k]
    table = {}
    for s in SCHEDS:
        for k in SEEDS:
            for base in ("NJ", "NL"):
                pts = []
                for rho in RHOS:
                    p = point(D, L, a_unit(k, base, s, rho), U20[k], cache)
                    if p is None:
                        return None
                    rr = unit_rec(a_run(k, base, s, rho))
                    p["rho"] = rho
                    p["compute"] = (rr["diag"]["critic_online_updates"] + rr["diag"]["critic_matched_extra_updates"]
                                    + rr["diag"]["critic_refit_updates"]) if rr else None
                    pts.append(p)
                table[(s, k, base)] = pts
    per_sched, complete = {}, []
    for s in SCHEDS:
        per_sched[s] = {}
        for k in SEEDS:
            fe = [p for p in table[(s, k, "NJ")] if p["feasible"]]
            per_sched[s][k] = min(fe, key=lambda p: (p["auc"]["pair"], p["rho"])) if fe else None
        if all(per_sched[s][k] is not None for k in SEEDS):
            complete.append(s)
    mean_pair = {s: float(np.mean([per_sched[s][k]["auc"]["pair"] for k in SEEDS])) for s in complete}
    if complete:
        sel = min(complete, key=lambda s: (mean_pair[s], sum(per_sched[s][k]["compute"] for k in SEEDS),
                                           SCHED_TIE_ORDER.index(s)))
        status = "SELECTED"
        fb = None
    else:
        fb = {}
        for s in SCHEDS:
            best = [min(table[(s, k, "NJ")], key=lambda p: (p["shortfall"], p["rho"])) for k in SEEDS]
            fb[s] = (sum(p["shortfall"] for p in best), sum(p["compute"] for p in best), SCHED_TIE_ORDER.index(s))
        sel = min(SCHEDS, key=lambda s: fb[s])
        status = "FALLBACK"
    refs = {}
    for k in SEEDS:
        fe = [p for p in table[(sel, k, "NL")] if p["feasible"]]
        if fe:
            p = min(fe, key=lambda p: (p["worse_local"], p["mean_local"], p["rho"]))
            refs[k] = {"status": "NOMINEE", "unit": p["unit"], "rho": p["rho"], "auc": p["auc"]}
        else:
            mg, _ = gate_margins(U20[k], U20[k])
            refs[k] = ({"status": "TASK_ONLY_ALIAS", "unit": tl_unit(k, 20)} if feasible(mg)
                       else {"status": "NO_VALID_REFERENCE", "unit": None})
    boundary = sorted({p["unit"] for pts in table.values() for p in pts if p["feasible"] != p["feasible_float"]})
    return {"schedule": sel, "status": status, "complete": complete, "mean_pair": mean_pair, "fallback_keys": fb,
            "per_sched": {s: {k: (per_sched[s][k]["unit"] if per_sched[s][k] else None) for k in SEEDS} for s in SCHEDS},
            "references": refs, "table": table, "float_vs_exact_gate_disagreements": boundary}


def check_selection_A(D, L, cache):
    mine = replay_selection_A(D, L, cache)
    if mine is None:
        return res("PENDING", reason="Phase A units or inner records missing"), None
    out = {"schedule": mine["schedule"], "schedule_status": mine["status"], "complete_schedules": mine["complete"],
           "mean_selected_pair_auc": mine["mean_pair"], "per_schedule_selected": mine["per_sched"],
           "references": mine["references"], "float_vs_exact_gate_disagreements": mine["float_vs_exact_gate_disagreements"]}
    diffs = []
    for name, path in (("selection_A.json", RUN / "selection_A.json"), ("PHASE_A_SELECTION.json", RES / "PHASE_A_SELECTION.json")):
        if not path.exists():
            diffs.append(f"{name} missing")
            continue
        S = jload(path)
        if S["schedule"]["selected"] != mine["schedule"] or S["schedule"]["status"] != mine["status"]:
            diffs.append(f"{name}: schedule {S['schedule']['selected']}/{S['schedule']['status']}")
        if sorted(S["schedule"].get("complete_schedules", [])) != sorted(mine["complete"]):
            diffs.append(f"{name}: complete schedules")
        for s, v in S["schedule"].get("mean_selected_pair_auc", {}).items():
            if abs(v - mine["mean_pair"].get(s, np.nan)) > 1e-12:
                diffs.append(f"{name}: mean pair {s}")
        pss = S.get("per_schedule_selected") or {s: {k: (v.get("selected") or {}).get("unit") for k, v in S["per_schedule"][s].items()}
                                                 for s in S.get("per_schedule", {})}
        for s in SCHEDS:
            for k in SEEDS:
                if pss.get(s, {}).get(str(k)) != mine["per_sched"][s][k]:
                    diffs.append(f"{name}: {s} seed {k} selected {pss.get(s, {}).get(str(k))} vs {mine['per_sched'][s][k]}")
        for k in SEEDS:
            r = S["seeds"][str(k)]["reference"]
            m = mine["references"][k]
            if r.get("status") != m["status"] or r.get("unit") != m["unit"]:
                diffs.append(f"{name}: seed {k} reference {r.get('unit')}/{r.get('status')} vs {m['unit']}/{m['status']}")
        if "per_schedule" in S:
            for s in SCHEDS:
                for k in SEEDS:
                    rows = {t["unit"]: t for t in S["per_schedule"][s][str(k)]["table"]}
                    for p in mine["table"][(s, k, "NJ")]:
                        t = rows.get(p["unit"])
                        if t is None or t["task_feasible"] != p["feasible"] or abs(t["gate_shortfall"] - p["shortfall_float"]) > 1e-12 \
                                or t.get("compute") != p["compute"]:
                            diffs.append(f"{name}: point {p['unit']}")
    # calib references and Phase B run references
    for k in SEEDS:
        c = unit_rec(f"calib__s{k}")
        if c and c["reference"]["unit"] != mine["references"][k]["unit"]:
            diffs.append(f"calib__s{k} reference")
    for n in os.listdir(UNITS):
        if n.startswith("run__B__") and unit_done(n):
            r = unit_rec(n)
            if r.get("reference") != mine["references"][r["seed"]]["unit"]:
                diffs.append(f"{n} reference {r.get('reference')}")
    fr_ = RES / "PHASE_A_FRONTIERS.csv"
    if fr_.exists():
        for r in csv.DictReader(open(fr_)):
            local = r["schedule"].endswith(" (local)")
            s, k, unit = r["schedule"].replace(" (local)", ""), int(r["seed"]), r["unit"]
            p = next((p for p in mine["table"].get((s, k, "NL" if local else "NJ"), []) if p["unit"] == unit), None)
            if p is None:
                diffs.append(f"frontier row {unit} unknown")
                continue
            chosen = (mine["references"][k]["unit"] == unit) if local else (mine["per_sched"][s][k] == unit)
            if (r["task_feasible"] == "True") != p["feasible"] or (r["selected_for_schedule"] == "True") != chosen \
                    or abs(float(r["auc_pair"]) - p["auc"]["pair"]) > 1e-6 or \
                    abs(float(r["acc_income"]) - p["counts"][0]["k"] / p["counts"][0]["n"]) > 1e-6 or \
                    abs(float(r["acc_occ"]) - p["counts"][1]["k"] / p["counts"][1]["n"]) > 1e-6:
                diffs.append(f"frontier row {unit}")
    out["differences"] = diffs
    return res("FAIL" if diffs else "PASS", **out), mine


# ------------------------------------------------------------------------------------------------ selection B
def single_task_margins(c, ref_t):
    n = c["n"]
    a, au, cc = Fraction(c["k"], n), Fraction(ref_t["k"], ref_t["n"]), Fraction(c["k_const"], n)
    return {"G1": a - au + Fraction(1, 100), "G2": a - Fraction(4, 5) * au - Fraction(1, 5) * cc,
            "G3": a - cc - Fraction(3, 100)}


def fare_selection(D, L, k, ref, cache):
    """Per-purpose FARE rule (inherited, reselected on inner roles): among configurations whose own task passes
    G1-G3 against U e40, the lowest local inner AUC (ties lower configuration id); otherwise NO_FEASIBLE_NOMINEE and
    the closest configuration (largest worst gate margin, ties lower id) is kept descriptively."""
    per = {}
    for i in (0, 1):
        rows = []
        for cid in range(1, 100):
            u = f"fare__s{k}__p{i}__c{cid}"
            if not (UNITS / u).exists():
                break
            ir = unit_rec(f"inner__{u}")
            cnt = unit_counts(D, L, u, cache)
            if ir is None or cnt is None:
                rows.append({"config": cid, "unit": u, "missing": True})
                continue
            mg = single_task_margins(cnt[i], ref[i])
            rows.append({"config": cid, "unit": u, "gate_ok": all(v >= 0 for v in mg.values()),
                         "margin": float(min(mg.values())), "gates": {g: float(v) for g, v in mg.items()},
                         "R_local": ir["recovery"]["auc"]["v"]})
        good = [r for r in rows if not r.get("missing")]
        ok = [r for r in good if r["gate_ok"]]
        if any(r.get("missing") for r in rows) or not rows:
            per[i] = {"status": "MISSING", "table": rows}
            continue
        if ok:
            s, st = min(ok, key=lambda r: (r["R_local"], r["config"])), "NOMINEE"
        else:
            s, st = max(good, key=lambda r: (r["margin"], -r["config"])), "NO_FEASIBLE_NOMINEE"
        per[i] = {"status": st, "config": s["config"], "unit": s["unit"], "table": rows}
    return per


def replay_selection_B(D, L, inv, cache, selA):
    if selA is None:
        return None
    U40 = {}
    for k in SEEDS:
        U40[k] = unit_counts(D, L, tl_unit(k, 40), cache)
        if U40[k] is None:
            return None
    out = {}
    for k in SEEDS:
        ref = U40[k]
        arms, missing = {}, []

        def grid(pts, tie):
            pts = [p for p in pts if p is not None]
            fe = [p for p in pts if p.get("feasible")]
            if not pts:
                return {"status": "MISSING"}
            if not fe:
                return {"status": "NO_FEASIBLE_NOMINEE", "points": [p["unit"] for p in pts]}
            p = min(fe, key=lambda p: (p["auc"]["pair"], p[tie]))
            return {"status": "NOMINEE", "unit": p["unit"], "rho": p[tie], "auc": p["auc"]}

        def single(p, units=None):
            if p is None:
                return {"status": "MISSING"}
            d = {"status": "NOMINEE" if p.get("feasible") else "INFEASIBLE_CONTROL", "unit": p["unit"], "auc": p["auc"],
                 "shortfall": p.get("shortfall_float")}
            if units:
                d["units"] = units
            return d

        for arm in B_ARMS:
            pts = []
            for rho in RHOS:
                p = point(D, L, b_unit(k, arm, rho), ref, cache)
                if p is None:
                    missing.append(b_unit(k, arm, rho))
                else:
                    p["rho"] = rho
                pts.append(p)
            arms[arm] = pts
        for arm in ("RAW-J", "RAW-L"):
            pts = []
            for beta in BETAS:
                p = point(D, L, raw_unit(k, arm, beta, 40), ref, cache)
                if p is None:
                    missing.append(raw_unit(k, arm, beta, 40))
                else:
                    p["beta"] = beta
                pts.append(p)
            arms[arm] = pts
        sel = {arm: grid(arms[arm], "rho") for arm in ("L-F", "J-N", "L-N")}
        sel.update({arm: grid(arms[arm], "beta") for arm in ("RAW-J", "RAW-L")})
        pu = point(D, L, tl_unit(k, 40), ref, cache)
        arms["U"] = [pu]
        sel["U"] = single(pu)
        pe = point(D, L, f"lc__s{k}__E", ref, cache)
        if pe is None:
            missing.append(f"lc__s{k}__E")
        arms["E"] = [pe]
        sel["E"] = single(pe)
        fs = fare_selection(D, L, k, ref, cache)
        if all(fs[i]["status"] != "MISSING" for i in (0, 1)):
            fu = [fs[0]["unit"], fs[1]["unit"]]
            ir = unit_rec(f"inner__pair__s{k}__F")
            if ir is None or list(ir.get("of", [])) != fu:
                missing.append(f"inner__pair__s{k}__F for {fu}")
                sel["F"] = {"status": "MISSING", "units": fu}
            else:
                p = point(D, L, None, ref, cache, inner_name=f"inner__pair__s{k}__F", of=fu)
                p["unit"] = f"inner__pair__s{k}__F"
                ok = fs[0]["status"] == fs[1]["status"] == "NOMINEE" and p["feasible"]
                sel["F"] = {"status": "NOMINEE" if ok else "NO_FEASIBLE_NOMINEE", "units": fu, "auc": p["auc"],
                            "configs": [fs[0]["config"], fs[1]["config"]],
                            "per_purpose_status": {i: fs[i]["status"] for i in (0, 1)}}
            zu = [f"fare__s{k}__p{i}__Z{fs[i]['config']}" for i in (0, 1)]
            ir0 = unit_rec(f"inner__pair__s{k}__F0")
            if ir0 is None or list(ir0.get("of", [])) != zu:
                missing.append(f"inner__pair__s{k}__F0 for {zu}")
                sel["F0"] = {"status": "MISSING", "units": zu}
            else:
                p0 = point(D, L, None, ref, cache, inner_name=f"inner__pair__s{k}__F0", of=zu)
                p0["unit"] = f"inner__pair__s{k}__F0"
                sel["F0"] = single(p0, units=zu)
        else:
            missing.append(f"FARE seed {k} per-purpose inputs")
            sel["F"], sel["F0"] = {"status": "MISSING"}, {"status": "MISSING"}
        cands = sorted((sel[a]["auc"]["pair"], CONTROL_ORDER.index(a), a) for a in CONTROL_ORDER
                       if sel[a].get("status") == "NOMINEE")
        cstar = cands[0][2] if cands else None
        ties = [c[2] for c in cands if cands and c[0] == cands[0][0]][1:]
        valid_ref = selA["references"][k]["status"] in ("NOMINEE", "TASK_ONLY_ALIAS")
        comps = {}
        if sel["L-F"]["status"] == "NOMINEE":
            comps["L-F"] = sel["L-F"]["auc"]
        if cstar:
            comps["C*"] = sel[cstar]["auc"]
        jf = []
        for p in arms["J-F"]:
            if p is None:
                continue
            ex = {c: {v: p["auc"][v] - (a[v] + BUFFER) for v in ("v1", "v2")} for c, a in comps.items()}
            p["guard_excess"] = ex
            p["guard_ok"] = all(x <= 0 for d in ex.values() for x in d.values())
            p["eligible"] = bool(p.get("feasible")) and p["guard_ok"]
            p["nomination_shortfall"] = p.get("shortfall_float", np.inf) + sum(max(0.0, x) for d in ex.values() for x in d.values())
            jf.append(p)
        el = [p for p in jf if p["eligible"]]
        if el:
            p = min(el, key=lambda p: (p["auc"]["pair"], p["rho"]))
            sel["J-F"] = {"status": "NOMINEE", "unit": p["unit"], "rho": p["rho"], "auc": p["auc"],
                          "guard_excess": p["guard_excess"], "nomination_shortfall": p["nomination_shortfall"]}
        elif jf and len(jf) == len(RHOS):
            p = min(jf, key=lambda p: (p["nomination_shortfall"], p["auc"]["pair"], p["rho"]))
            sel["J-F"] = {"status": "NO_FEASIBLE_NOMINEE", "descriptive_fallback": p["unit"], "rho": p["rho"],
                          "auc": p["auc"], "guard_excess": p["guard_excess"], "nomination_shortfall": p["nomination_shortfall"]}
        else:
            sel["J-F"] = {"status": "MISSING"}
        points = {arm: [p for p in arms[arm] if p is not None] for arm in arms}
        out[k] = {"arms": sel, "C*": ({"arm": cstar, "unit": sel[cstar].get("unit"), "units": sel[cstar].get("units"),
                                       "candidates": [c[2] for c in cands], "tied_with": ties} if cstar else None),
                  "valid_reference": valid_ref, "reference": selA["references"][k], "missing_inputs": missing,
                  "fare_selection": fs, "points": points, "nomination_guards": comps}
    return out


def _dig(d, *paths):
    for path in paths:
        x = d
        ok = True
        for k in path:
            if isinstance(x, dict) and k in x:
                x = x[k]
            else:
                ok = False
                break
        if ok:
            return x
    return None


def _close(a, b, tol=1e-12):
    return a is not None and b is not None and abs(float(a) - float(b)) <= tol


FIXED_RHO = {"A-ONLINE@r0.75": lambda k: a_unit(k, "NJ", "ONLINE", 0.75),
             "A-REFRESHED@r0.75": lambda k: a_unit(k, "NJ", "REFRESHED", 0.75),
             "A-MATCHED@r0.75": lambda k: a_unit(k, "NJ", "ONLINE_MATCHED", 0.75),
             "J-F@r0.75": lambda k: b_unit(k, "J-F", 0.75), "L-F@r0.75": lambda k: b_unit(k, "L-F", 0.75),
             "J-N@r0.75": lambda k: b_unit(k, "J-N", 0.75), "L-N@r0.75": lambda k: b_unit(k, "L-N", 0.75)}


def compare_selection_B_files(mine, mineA):
    diffs, compared = [], []
    sb = RUN / "selection_B.json"
    if sb.exists():
        S = jload(sb)
        compared.append("selection_B.json")
        for k in SEEDS:
            sk, mk = S.get(str(k), {}), mine[k]
            if bool(sk.get("valid_reference")) != mk["valid_reference"] or (sk.get("reference") or {}).get("unit") != mk["reference"]["unit"]:
                diffs.append(f"selB s{k}: reference")
            for arm, m in mk["arms"].items():
                t = (sk.get("arms") or {}).get(arm)
                if t is None:
                    diffs.append(f"selB s{k} {arm}: missing")
                    continue
                if t.get("status") != m["status"]:
                    diffs.append(f"selB s{k} {arm}: status {t.get('status')} vs {m['status']}")
                mu = m.get("unit") or m.get("descriptive_fallback")
                if arm in ("F", "F0"):
                    if list(t.get("units") or []) != list(m.get("units") or []):
                        diffs.append(f"selB s{k} {arm}: units {t.get('units')} vs {m.get('units')}")
                elif t.get("unit") != mu:
                    diffs.append(f"selB s{k} {arm}: unit {t.get('unit')} vs {mu}")
                if "auc" in m and t.get("auc") and any(t["auc"][v] != m["auc"][v] for v in ("v1", "v2", "pair")):
                    diffs.append(f"selB s{k} {arm}: auc")
                if arm == "J-F" and "guard_excess" in m:
                    for c, d in m["guard_excess"].items():
                        for v, x in d.items():
                            if not _close(_dig(t, ("guard_excess", c, v)), x):
                                diffs.append(f"selB s{k} J-F guard_excess {c}/{v}")
                    if not _close(t.get("nomination_shortfall"), m["nomination_shortfall"]):
                        diffs.append(f"selB s{k} J-F nomination_shortfall {t.get('nomination_shortfall')} vs {m['nomination_shortfall']}")
                    for c, a in mk["nomination_guards"].items():
                        if _dig(t, ("nomination_guards", c)) != a:
                            diffs.append(f"selB s{k} J-F nomination guard {c}")
                # per-point feasibility / shortfall
                pts = {p["unit"]: p for p in mk["points"].get(arm, [])}
                for row in t.get("table") or []:
                    p = pts.get(row.get("unit"))
                    if p is None:
                        continue
                    if row.get("task_feasible") != p["feasible"] or not _close(row.get("gate_shortfall"), p["shortfall_float"]):
                        diffs.append(f"selB s{k} {arm}: point {row.get('unit')}")
            fsel = sk.get("fare_selection") or {}
            for i in (0, 1):
                t, m = fsel.get(str(i)) or {}, mk["fare_selection"][i]
                if t.get("status") != m["status"] or t.get("config") != m.get("config"):
                    diffs.append(f"selB s{k} FARE purpose {i}: {t.get('status')}/{t.get('config')} vs {m['status']}/{m.get('config')}")
                rows = {r["config"]: r for r in t.get("table") or []}
                for r in m["table"]:
                    tr = rows.get(r["config"])
                    if tr is None or r.get("missing"):
                        continue
                    if tr.get("gate_ok") != r["gate_ok"] or not _close(tr.get("margin"), r["margin"], 1e-12) or tr.get("R_local") != r["R_local"]:
                        diffs.append(f"selB s{k} FARE p{i} c{r['config']}")
            c = sk.get("comparator") or {}
            mc = mk["C*"] or {}
            if c.get("arm") != mc.get("arm") or (c.get("candidates") is not None and list(c["candidates"]) != mc.get("candidates")):
                diffs.append(f"selB s{k}: C* {c.get('arm')} {c.get('candidates')} vs {mc.get('arm')} {mc.get('candidates')}")
    el = RES / "EVALUATION_LOCK.json"
    if el.exists():
        EL = jload(el)
        compared.append("EVALUATION_LOCK.json")
        ps = EL.get("phase_A_schedule") or {}
        if ps.get("selected") != mineA["schedule"] or ps.get("status") != mineA["status"]:
            diffs.append("lock phase_A_schedule")
        for k in SEEDS:
            sk, mk = EL.get("seeds", {}).get(str(k), {}), mine[k]
            score, status = sk.get("score", {}), sk.get("status", {})
            if bool(sk.get("valid_reference")) != mk["valid_reference"]:
                diffs.append(f"lock s{k}: valid_reference")
            for arm, m in mk["arms"].items():
                if status.get(arm) != m["status"]:
                    diffs.append(f"lock s{k} {arm}: status {status.get(arm)} vs {m['status']}")
                spec = score.get(arm) or {}
                if arm in ("F", "F0"):
                    if list(spec.get("units") or []) != list(m.get("units") or []):
                        diffs.append(f"lock s{k} {arm}: units")
                elif spec.get("unit") != (m.get("unit") or m.get("descriptive_fallback")):
                    diffs.append(f"lock s{k} {arm}: unit {spec.get('unit')}")
            for lab, f in FIXED_RHO.items():
                if (score.get(lab) or {}).get("unit") != f(k) or status.get(lab) != "FIXED_RHO_COMPONENT":
                    diffs.append(f"lock s{k} {lab}: fixed-rho component")
            extra = sorted(set(score) - set(mk["arms"]) - set(FIXED_RHO))
            if extra:
                diffs.append(f"lock s{k}: unexpected scored labels {extra}")
            c, mc = sk.get("comparator") or {}, mk["C*"] or {}
            if c.get("arm") != mc.get("arm") or c.get("unit") != mc.get("unit"):
                diffs.append(f"lock s{k}: comparator {c.get('arm')} vs {mc.get('arm')}")
            g = sk.get("J-F_nomination_guards")
            if g is not None and g != mk["nomination_guards"]:
                diffs.append(f"lock s{k}: J-F nomination guards")
            cal = unit_rec(f"calib__s{k}")
            if cal and sk.get("controller_targets") is not None and list(sk["controller_targets"]) != list(cal["b"]):
                diffs.append(f"lock s{k}: controller targets")
    ss = RES / "SEED_STATUS.json"
    if ss.exists():
        S = jload(ss)
        compared.append("SEED_STATUS.json")
        for k in SEEDS:
            sk, mk = S.get(str(k), {}), mine[k]
            if (sk.get("comparator") or {}).get("arm") != (mk["C*"] or {}).get("arm"):
                diffs.append(f"SEED_STATUS s{k}: comparator")
            for arm, m in mk["arms"].items():
                t = (sk.get("arms") or {}).get(arm) or {}
                if t.get("status") != m["status"]:
                    diffs.append(f"SEED_STATUS s{k} {arm}: status {t.get('status')} vs {m['status']}")
    st = RES / "SELECTION_TABLE.csv"
    if st.exists():
        compared.append("SELECTION_TABLE.csv")
        for r in csv.DictReader(open(st)):
            k, arm, unit = int(r["seed"]), r["arm"], r["unit"]
            mk = mine[k]
            m = mk["arms"].get(arm)
            if m is None:
                continue
            if r["arm_status"] != m["status"]:
                diffs.append(f"SELECTION_TABLE s{k} {arm} {unit}: arm_status")
            if arm in ("F", "F0"):
                continue
            p = next((p for p in mk["points"].get(arm, []) if p["unit"] == unit), None)
            if p is None:
                continue
            if (r["task_feasible"] == "True") != p["feasible"] or abs(float(r["auc_pair"]) - p["auc"]["pair"]) > 1e-6:
                diffs.append(f"SELECTION_TABLE s{k} {arm} {unit}")
            chosen = (m.get("unit") or m.get("descriptive_fallback")) == unit
            if (r["outcome"] == "selected") and not (chosen and m["status"] == "NOMINEE"):
                diffs.append(f"SELECTION_TABLE s{k} {arm} {unit}: outcome")
    return diffs, compared


def check_selection_B(D, L, inv, cache, selA):
    mine = replay_selection_B(D, L, inv, cache, selA)
    if mine is None:
        return res("PENDING", reason="Phase A replay or U e40 missing"), None
    miss = sorted({m for k in SEEDS for m in mine[k]["missing_inputs"]})
    out = {"replay": {k: {"arms": {a: {kk: vv for kk, vv in v.items() if kk != "points"} for a, v in mine[k]["arms"].items()},
                          "C*": mine[k]["C*"], "valid_reference": mine[k]["valid_reference"],
                          "fare_per_purpose": {i: {"status": f["status"], "config": f.get("config")}
                                               for i, f in mine[k]["fare_selection"].items()},
                          "jf_points": [{"unit": p["unit"], "feasible": p.get("feasible"), "guard_ok": p.get("guard_ok"),
                                         "eligible": p.get("eligible"), "nomination_shortfall": p.get("nomination_shortfall")}
                                        for p in mine[k]["points"]["J-F"]]} for k in SEEDS},
           "missing_inputs": miss}
    exact_vs_float = sorted({p["unit"] for k in SEEDS for pts in mine[k]["points"].values() for p in pts
                             if p.get("feasible") != p.get("feasible_float")})
    out["float_vs_exact_gate_disagreements"] = exact_vs_float
    diffs, compared = compare_selection_B_files(mine, selA)
    st = "FAIL" if diffs else ("PENDING" if miss or not compared else "PASS")
    return res(st, compared_with=compared, differences=diffs, **out), mine


# ------------------------------------------------------------------------------------------------ bootstrap and endpoints
class Boot:
    def __init__(self, assess_unit, B=B_BOOT, seed=BOOT_SEED, chunk=CHUNK):
        self.groups, self.ginv = np.unique(np.asarray(assess_unit), return_inverse=True)
        G = len(self.groups)
        rng = np.random.default_rng(seed)
        parts, done = [], 0
        while done < B:
            m = min(chunk, B - done)
            parts.append(rng.multinomial(G, np.full(G, 1.0 / G), size=m))
            done += m
        self.counts = np.vstack(parts).astype(np.int64)
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
            return num / den


def wauc(score, y, W):
    return AUCPrep(score, y)(np.atleast_2d(W))


class Levels:
    """Point values and bootstrap replicates of per-seed levels (AUC, accuracy, race AUC, LLR, constants)."""

    def __init__(self, boot: Boot, n):
        self.boot = boot
        self.n = n
        self.jobs = {}
        self.vals = {}

    def add_auc(self, key, comps, rows=None):
        """comps: list of (score, y) whose AUCs are averaged (attacker seeds and/or classes)."""
        self.jobs[key] = ("auc", comps, rows)

    def add_mean(self, key, vec):
        self.jobs[key] = ("mean", vec, None)

    def add_llr(self, key, num, den):
        """1 - weighted mean(num) / weighted mean(den) (proper-loss recovery relative to the prior)."""
        self.jobs[key] = ("llr", num, den)

    def run(self):
        preps = {k: [AUCPrep(s, y) for s, y in j[1]] for k, j in self.jobs.items() if j[0] == "auc"}
        reps = {k: [] for k in self.jobs}
        ones = np.ones((1, self.n))
        point = {}
        for k, j in self.jobs.items():
            if j[0] == "auc":
                Wr = ones if j[2] is None else ones[:, j[2]]
                point[k] = float(np.mean([p(Wr)[0] for p in preps[k]]))
            elif j[0] == "mean":
                point[k] = float(j[1].mean())
            else:
                point[k] = float(1.0 - j[1].mean() / j[2].mean())
        for W in self.boot.chunks():
            for k, j in self.jobs.items():
                if j[0] == "auc":
                    Wr = W if j[2] is None else W[:, j[2]]
                    reps[k].append(np.mean([p(Wr) for p in preps[k]], axis=0))
                elif j[0] == "mean":
                    reps[k].append(W @ j[1] / W.sum(1))
                else:
                    reps[k].append(1.0 - (W @ j[1]) / (W @ j[2]))
        for k in self.jobs:
            self.vals[k] = (point[k], np.concatenate(reps[k]))
        return self.vals


def decide(side, target, lower, upper):
    if side == "lower>":
        return "PASS" if lower > target else "NOT_ESTABLISHED"
    if side == "upper<":
        return "PASS" if upper < target else "NOT_ESTABLISHED"
    return "ABOVE" if lower > target else ("BELOW" if upper < target else "NOT_RESOLVED")


FMT_VIEW = {("prim", "v1"): "v1", ("prim", "v2"): "v2", ("prim", "pair"): "pair", ("prob", "v1"): "p1",
            ("prob", "v2"): "p2", ("prob", "pair"): "ppair", ("hard", "v1"): "h1", ("hard", "v2"): "h2",
            ("hard", "pair"): "hpair"}


def compute_endpoints(fam, preds, cstar, boot, maj, prior, seeds=SEEDS, all_levels=False):
    """fam: PRIMARY_FAMILY dict; preds[(k, label)] -> dict of arrays; cstar[k] -> label used for C*.
    maj: {0: maj_income, 1: maj_occ} from NEW_DEFENSE_FIT; prior: SEX prior (2,) from NEW_DEFENSE_FIT."""
    any_p = next(iter(preds.values()))
    n = len(any_p["sex"])
    LV = Levels(boot, n)

    def lab_of(k, lab):
        return cstar.get(k) if lab == "C*" else lab

    def R(k, lab, view, fmt="prim"):
        lab = lab_of(k, lab)
        key = ("R", k, lab, fmt, view)
        if key not in LV.jobs and (k, lab) in preds:
            P = preds[(k, lab)][f"P_auc_{FMT_VIEW[(fmt, view)]}"]
            s = preds[(k, lab)]["sex"]
            LV.add_auc(key, [(P[a, :, 1], s) for a in range(P.shape[0])])
        return key

    def ACC(k, lab, t):
        lab = lab_of(k, lab)
        key = ("acc", k, lab, t)
        if key not in LV.jobs and (k, lab) in preds:
            p = preds[(k, lab)]
            y = p["y_income"] if t == 0 else p["y_occ"]
            LV.add_mean(key, (p["hard1" if t == 0 else "hard2"] == y).astype(np.float64))
        return key

    def CONST(t):
        key = ("const", t)
        if key not in LV.jobs:
            y = any_p["y_income"] if t == 0 else any_p["y_occ"]
            LV.add_mean(key, (y == maj[t]).astype(np.float64))
        return key

    def RRACE(k, lab, view):
        lab = lab_of(k, lab)
        key = ("Rrace", k, lab, view)
        if key not in LV.jobs and (k, lab) in preds:
            p = preds[(k, lab)]
            P = p.get(f"Prace_auc_{view}")
            if P is not None:
                pos, yr = p["race_pos"], p["race_y"]
                K = P.shape[2]
                LV.add_auc(key, [(P[a, :, c], (yr == c).astype(int)) for a in range(P.shape[0]) for c in range(K)], rows=pos)
        return key

    def LLR(k, lab, view):
        lab = lab_of(k, lab)
        key = ("LLR", k, lab, view)
        if key not in LV.jobs and (k, lab) in preds:
            p = preds[(k, lab)]
            P = p[f"P_ce_{view}"]
            s = p["sex"]
            with np.errstate(divide="ignore"):
                ce = np.mean([-np.log(P[a, np.arange(n), s]) for a in range(P.shape[0])], axis=0)
            LV.add_llr(key, ce, -np.log(prior[s]))
        return key

    def stat_fn(e):
        """Returns per-seed function of a level getter g(key) -> value."""
        kind = e["kind"]
        if kind == "coalition":
            return lambda g, k: g(R(k, e["ref"], "pair")) - g(R(k, "J-F", "pair"))
        if kind == "local":
            return lambda g, k: g(R(k, "J-F", e["view"])) - g(R(k, e["ref"], e["view"]))
        if kind == "acc":
            return lambda g, k: g(ACC(k, "J-F", e["task"])) - g(ACC(k, "U", e["task"]))
        if kind == "retain":
            return lambda g, k: g(ACC(k, "J-F", e["task"])) - 0.8 * g(ACC(k, "U", e["task"])) - 0.2 * g(CONST(e["task"]))
        if kind == "useful":
            return lambda g, k: g(ACC(k, "J-F", e["task"])) - g(CONST(e["task"]))
        if kind == "rec":
            return lambda g, k: g(R(k, e["a"], e["view"])) - g(R(k, e["b"], e["view"]))
        if kind == "accdiff":
            return lambda g, k: g(ACC(k, e["a"], e["task"])) - g(ACC(k, e["b"], e["task"]))
        if kind == "synergy":
            return lambda g, k: g(R(k, e["arm"], "pair")) - np.maximum(g(R(k, e["arm"], "v1")), g(R(k, e["arm"], "v2")))
        if kind == "race":
            return lambda g, k: g(RRACE(k, e["a"], e["view"])) - g(RRACE(k, e["b"], e["view"]))
        if kind == "out":
            return lambda g, k: g(R(k, e["a"], e["view"], e["fmt"])) - g(R(k, e["b"], e["view"], e["fmt"]))
        if kind == "logloss":
            return lambda g, k: g(LLR(k, e["a"], e["view"])) - g(LLR(k, e["b"], e["view"]))
        raise ValueError(kind)

    entries = [("primary", e) for e in fam["primary"]] + [("secondary", e) for e in fam["secondary"]]
    fns = {e["id"]: stat_fn(e) for _, e in entries}
    for _, e in entries:                      # registration pass: collects the level jobs that have predictions
        for k in seeds:
            fns[e["id"]](lambda key: 0.0, k)
    if all_levels:                            # every per-seed level of every scored label (RAW_LEVELS / inference levels)
        for (k, lab) in sorted(preds, key=str):
            for fmt in ("prim", "prob", "hard"):
                for view in ("v1", "v2", "pair"):
                    R(k, lab, view, fmt)
            for t in (0, 1):
                ACC(k, lab, t)
                CONST(t)
            for view in ("v1", "v2", "pair"):
                RRACE(k, lab, view)
                LLR(k, lab, view)
    vals = LV.run()
    if all_levels:                            # seed means (Rmean on primary views, accmean)
        labs = sorted({lab for (_, lab) in preds})
        for lab in labs:
            for view in ("v1", "v2", "pair"):
                ks = [("R", k, lab, "prim", view) for k in seeds]
                if all(x in vals for x in ks):
                    vals[("Rmean", lab, view)] = (float(np.mean([vals[x][0] for x in ks])),
                                                  np.mean([vals[x][1] for x in ks], axis=0))
            for t in (0, 1):
                ks = [("acc", k, lab, t) for k in seeds]
                if all(x in vals for x in ks):
                    vals[("accmean", lab, t)] = (float(np.mean([vals[x][0] for x in ks])),
                                                 np.mean([vals[x][1] for x in ks], axis=0))

    def probe(key):
        if key not in vals:
            raise KeyError(key)
        return 0.0

    results = {}
    for fam_name, e in entries:
        k_ok = []
        for k in seeds:
            try:
                fns[e["id"]](probe, k)
                k_ok.append(k)
            except KeyError:
                pass
        if not k_ok:
            results[e["id"]] = {"family": fam_name, "status": "NOT_ESTIMABLE", "seeds": []}
            continue
        pt = float(np.mean([fns[e["id"]](lambda key: vals[key][0], k) for k in k_ok]))
        rp = np.mean([fns[e["id"]](lambda key: vals[key][1], k) for k in k_ok], axis=0)
        fin = rp[np.isfinite(rp)]
        se = float(np.std(fin, ddof=1)) if len(fin) > 1 else float("nan")
        z = fam["z_primary"] if fam_name == "primary" else fam["z_secondary"]
        lo, hi = pt - z * se, pt + z * se
        results[e["id"]] = {"family": fam_name, "stat": e["stat"], "target": e["target"], "side": e["side"],
                            "point": pt, "se": se, "lower": lo, "upper": hi, "z": z,
                            "decision": decide(e["side"], e["target"], lo, hi), "seeds": k_ok,
                            "n_finite_replicates": int(len(fin)), "alias_of": e.get("alias_of")}
    levels = {"|".join(map(str, k)): {"point": v[0], "se": float(np.std(v[1][np.isfinite(v[1])], ddof=1))}
              for k, v in vals.items()}
    return results, levels


def lock_state():
    """EVALUATION_LOCK committed, unchanged in the working tree, and contained in the remote-tracking branch."""
    p = RES / "EVALUATION_LOCK.json"
    rel = "results/pcrl_strength_matched_feedback_v1/EVALUATION_LOCK.json"
    if not p.exists():
        return {"exists": False, "verified": False}
    c = git("log", "-1", "--format=%H", "--", rel)
    blob_ok = bool(c) and git("rev-parse", f"{c}:{rel}") == git("hash-object", str(p))
    pushed = bool(c) and bool(git("branch", "-r", "--contains", c))
    return {"exists": True, "commit": c, "worktree_equals_commit": blob_ok, "pushed": pushed,
            "verified": bool(c and blob_ok and pushed)}


def load_preds(EL):
    preds, cstar, units_used, missing = {}, {}, {}, []
    for k in SEEDS:
        sk = EL.get("seeds", {}).get(str(k), {})
        for label in sk.get("score", {}):
            u = UNITS / f"outer__s{k}__{safe(label)}" / "preds.npz"
            if u.exists() and unit_done(f"outer__s{k}__{safe(label)}"):
                z = np.load(u, allow_pickle=False)
                preds[(k, label)] = {f: z[f] for f in z.files}
                units_used[(k, label)] = f"outer__s{k}__{safe(label)}"
            else:
                missing.append(f"outer__s{k}__{safe(label)}")
        comp = sk.get("comparator") or {}
        if (k, "C*") in preds:
            cstar[k] = "C*"
        elif comp.get("arm") and (k, comp["arm"]) in preds:
            cstar[k] = comp["arm"]
    return preds, cstar, units_used, missing


def check_preds_alignment(D, preds, LU, mineB, EL):
    am = D.mask[ASSESS]
    rid, unit = D.row_id[am], D.unit[am]
    bad = []
    for (k, lab), p in preds.items():
        if not np.array_equal(p["assess_row_id"], rid) or not np.array_equal(p["assess_unit"], unit):
            bad.append(f"{k}/{lab}: rows/groups")
        for f, lk in (("sex", "sex"), ("race", "race"), ("y_income", "y_income"), ("y_occ", "y_occ")):
            if f in p and not np.array_equal(p[f], LU[lk][am]):
                bad.append(f"{k}/{lab}: {f}")
        spec = EL["seeds"][str(k)]["score"].get(lab) or {}
        u = spec.get("unit")
        if u and (unit_dir(u) / "release.npz").exists():
            rel = np.load(unit_dir(u) / "release.npz", allow_pickle=False)
            for hk in ("hard1", "hard2"):
                if not np.array_equal(p[hk], rel[hk][am]):
                    bad.append(f"{k}/{lab}: {hk} != frozen release")
            for pk in ("p1", "p2"):
                if pk in p and maxdiff(p[pk], rel[pk][am]) > 1e-12:
                    bad.append(f"{k}/{lab}: {pk} != frozen release")
        elif spec.get("units"):
            for i, uu in enumerate(spec["units"]):
                rel = np.load(unit_dir(uu) / "release.npz", allow_pickle=False)
                if not np.array_equal(p[f"hard{i + 1}"], rel["hard"][am]):
                    bad.append(f"{k}/{lab}: hard{i + 1} != frozen FARE release")
                if f"p{i + 1}" in p and maxdiff(p[f"p{i + 1}"], rel["p"][am]) > 1e-12:
                    bad.append(f"{k}/{lab}: p{i + 1} != frozen FARE release")
    return bad


def compare_endpoints(mine, levels=None):
    out = {}
    for name, fam in (("PRIMARY_ENDPOINTS.csv", "primary"), ("SECONDARY_ENDPOINTS.csv", "secondary")):
        p = RES / name
        if not p.exists():
            out[name] = res("PENDING")
            continue
        diffs = []
        rows = {r["id"]: r for r in csv.DictReader(open(p))}
        for i, m in mine.items():
            if m["family"] != fam:
                continue
            r = rows.get(i)
            if r is None:
                diffs.append(f"{i} missing")
                continue
            if m.get("status") == "NOT_ESTIMABLE":
                continue
            for f in ("point", "se", "lower", "upper"):
                if abs(float(r[f]) - m[f]) > 2e-6:
                    diffs.append(f"{i} {f} {r[f]} vs {m[f]:.6f}")
            if r["decision"] != m["decision"]:
                diffs.append(f"{i} decision {r['decision']} vs {m['decision']}")
        out[name] = res("FAIL" if diffs else "PASS", differences=diffs, rows=len(rows))
    inf = RUN / "inference.json"
    if inf.exists():
        I = jload(inf)
        diffs = []
        for fam in ("primary", "secondary"):
            for r in I.get(fam, []):
                m = mine.get(r["id"])
                if m is None or m.get("status") == "NOT_ESTIMABLE":
                    continue
                for f in ("point", "se", "lower", "upper"):
                    if abs(r[f] - m[f]) > 1e-9 * max(1.0, abs(m[f])):
                        diffs.append(f"{r['id']} {f} {r[f]} vs {m[f]}")
                if r["decision"] != m["decision"]:
                    diffs.append(f"{r['id']} decision")
        out["inference.json"] = res("FAIL" if diffs else "PASS", differences=diffs[:40], n_differences=len(diffs))
        lv = I.get("levels") or {}
        lb, n = [], 0
        for key, m in (levels or {}).items():
            r = lv.get(key)
            if r is None:
                continue
            n += 1
            if abs(r["point"] - m["point"]) > 1e-10 or abs(r["se"] - m["se"]) > 1e-10:
                lb.append(f"{key}: point {r['point']} vs {m['point']}, se {r['se']} vs {m['se']}")
        out["inference_levels"] = res("FAIL" if lb else "PASS", levels_compared=n, levels_only_in_runner=len(set(lv) - set(levels or {})),
                                      levels_only_in_verifier=len(set(levels or {}) - set(lv)), differences=lb[:30],
                                      n_differences=len(lb))
    else:
        out["inference.json"] = res("PENDING")
    rl = RES / "RAW_LEVELS.csv"
    if rl.exists():
        lb, n, unmatched = [], 0, 0
        for r in csv.DictReader(open(rl)):
            q, sd, lab, det = r["quantity"], r["seed"], r["label"], r["detail"]
            key = {"R": f"R|{sd}|{lab}|{det}", "acc": f"acc|{sd}|{lab}|{det}", "Rrace": f"Rrace|{sd}|{lab}|{det}",
                   "LLR": f"LLR|{sd}|{lab}|{det}", "Rmean": f"Rmean|{lab}|{det}", "accmean": f"accmean|{lab}|{det}",
                   "const": f"const|{lab}"}.get(q)
            m = (levels or {}).get(key)
            if m is None:
                unmatched += 1
                continue
            n += 1
            if abs(float(r["point"]) - m["point"]) > 6e-7 or abs(float(r["se"]) - m["se"]) > 6e-7:
                lb.append(f"{key}: {r['point']}/{r['se']} vs {m['point']:.6f}/{m['se']:.6f}")
        out["RAW_LEVELS.csv"] = res("FAIL" if lb else ("WARN" if unmatched else "PASS"), rows_compared=n,
                                    rows_without_verifier_level=unmatched, differences=lb[:30])
    else:
        out["RAW_LEVELS.csv"] = res("PENDING")
    return out


def check_endpoints(D, mineA, mineB):
    ls = lock_state()
    if not ls["exists"]:
        return res("PENDING", reason="EVALUATION_LOCK.json not written", lock=ls), None
    if not ls["verified"]:
        return res("PENDING", reason="EVALUATION_LOCK not committed+pushed+unchanged; assessment labels stay sealed",
                   lock=ls), None
    EL = jload(RES / "EVALUATION_LOCK.json")
    fam = jload(RES / "PRIMARY_FAMILY.json")
    preds, cstar, used, missing = load_preds(EL)
    if not preds:
        return res("PENDING", reason="no outer preds.npz yet", lock=ls), None
    LU = D.labels(unseal=True, why="EVALUATION_LOCK verified (committed, unchanged, pushed)")
    fitm = D.mask[FIT]
    sex_fit = LU["sex"][fitm]
    cnt = np.bincount(sex_fit).astype(np.int64)
    prior = cnt / cnt.sum()
    prior_hash = hashlib.sha256(cnt.tobytes()).hexdigest()
    lock_prior = [v for k, v in EL.items() if "prior" in k.lower()]
    lock_prior += [v for v in (EL.get("priors") or {}).values()] if isinstance(EL.get("priors"), dict) else []
    hashes = [v for v in lock_prior if isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v)]
    hashes += [vv for v in lock_prior if isinstance(v, dict) for vv in v.values() if isinstance(vv, str) and re.fullmatch(r"[0-9a-f]{64}", vv)]
    maj = {0: int(np.bincount(LU["y_income"][fitm]).argmax()), 1: int(np.bincount(LU["y_occ"][fitm]).argmax())}
    align = check_preds_alignment(D, preds, LU, mineB, EL)
    boot = Boot(next(iter(preds.values()))["assess_unit"])
    z1 = float(norm.ppf(1 - 0.05 / (2 * fam["primary_size"])))
    z2 = float(norm.ppf(1 - 0.05 / (2 * fam["secondary_size"])))
    zok = abs(z1 - fam["z_primary"]) < 1e-9 and abs(z2 - fam["z_secondary"]) < 1e-9 and abs(z1 - 2.991316) < 5e-7
    t0 = time.time()
    mine, levels = compute_endpoints(fam, preds, cstar, boot, maj, prior, all_levels=True)
    comp = compare_endpoints(mine, levels)
    aliases = all(mine[a]["point"] == mine[b]["point"] and mine[a]["se"] == mine[b]["se"]
                  for a, b in fam["aliases"].items() if "point" in mine.get(a, {}) and "point" in mine.get(b, {}))
    st = worst("FAIL" if align else "PASS", "PASS" if zok else "FAIL", "PASS" if aliases else "FAIL",
               "PASS" if (not hashes or prior_hash in hashes) else "FAIL", *[c["status"] for c in comp.values()],
               "PENDING" if missing else "PASS")
    return res(st, lock=ls, n_assessment=int(len(boot.ginv)), n_groups=boot.G, draws_sha256=boot.draws_sha256,
               z_primary=z1, z_secondary=z2, z_ok=zok, sex_prior_counts_sha256=prior_hash,
               lock_prior_hashes_found=hashes, prior_hash_matches=(prior_hash in hashes) if hashes else None,
               alias_slots_identical=aliases, alignment_failures=align, missing_outer_units=missing,
               cstar_label_per_seed=cstar, endpoints=mine, comparisons=comp, levels=levels,
               wall_s=time.time() - t0), mine


# ------------------------------------------------------------------------------------------------ conjunctions
def check_conjunctions(mine_end, mineA, mineB, end_status="PASS"):
    if mine_end is None or mineB is None or mineA is None:
        return res("PENDING", reason="endpoints or selection replay pending")
    st = {}
    for k in SEEDS:
        a = mineB[k]["arms"]
        st[k] = {"valid_reference": mineB[k]["valid_reference"], "J-F": a["J-F"]["status"], "L-F": a["L-F"]["status"],
                 "C*": mineB[k]["C*"]["arm"] if mineB[k]["C*"] else None,
                 "J-F_nontrivial": (a["J-F"].get("unit") or a["J-F"].get("descriptive_fallback") or "").startswith("B__"),
                 "L-F_nontrivial": (a["L-F"].get("unit") or "").startswith("B__")}
    passA = all(mine_end[f"P0{i}"]["decision"] == "PASS" for i in range(1, 10) if f"P0{i}" in mine_end)
    passB = all(mine_end[f"P{i}"]["decision"] == "PASS" for i in range(10, 19) if f"P{i}" in mine_end)
    reqA = all(st[k]["valid_reference"] and st[k]["J-F"] == "NOMINEE" and st[k]["L-F"] == "NOMINEE"
               and st[k]["J-F_nontrivial"] and st[k]["L-F_nontrivial"] for k in SEEDS)
    reqB = all(st[k]["J-F"] == "NOMINEE" and st[k]["C*"] is not None for k in SEEDS)
    reqB_family = reqB and all(st[k]["valid_reference"] for k in SEEDS)
    A = "ESTABLISHED" if (passA and reqA) else "NOT_ESTABLISHED"
    B = "ESTABLISHED" if (passB and reqB) else "NOT_ESTABLISHED"
    B_fam = "ESTABLISHED" if (passB and reqB_family) else "NOT_ESTABLISHED"
    out = {"claimA": {"all_nine_pass": passA, "status_requirements_met": reqA, "decision": A,
                      "clauses_passing": sum(mine_end[f"P0{i}"]["decision"] == "PASS" for i in range(1, 10))},
           "claimB": {"all_nine_pass": passB, "status_requirements_met": reqB, "decision": B,
                      "decision_with_family_valid_reference_rule": B_fam,
                      "clauses_passing": sum(mine_end[f"P{i}"]["decision"] == "PASS" for i in range(10, 19))},
           "seed_status": st}
    diffs = []
    inf = RUN / "inference.json"
    if inf.exists():
        I = jload(inf)
        for c in ("claimA", "claimB"):
            if c in I and I[c].get("decision") != out[c]["decision"]:
                diffs.append(f"{c}: {I[c].get('decision')} vs {out[c]['decision']}")
    ss = RES / "SEED_STATUS.json"
    if ss.exists():
        S = jload(ss)
        for k in SEEDS:
            sk = S.get(str(k), {})
            for arm in ("J-F", "L-F"):
                t = _dig(sk, ("arms", arm, "status"), (arm, "status"))
                if t is not None and t != st[k][arm]:
                    diffs.append(f"SEED_STATUS seed {k} {arm} {t} vs {st[k][arm]}")
            c = _dig(sk, ("comparator", "arm"))
            if c is not None and c != st[k]["C*"]:
                diffs.append(f"SEED_STATUS seed {k} C* {c} vs {st[k]['C*']}")
    out["differences"] = diffs
    overall = "DEVELOPMENT_ADVANTAGE_ESTABLISHED" if (A == "ESTABLISHED" and B == "ESTABLISHED") else "NOT_ESTABLISHED"
    out["status_requirements_alone_decide"] = (not reqA) and (not reqB)
    st = "FAIL" if diffs else ("PASS" if end_status in ("PASS", "WARN") or out["status_requirements_alone_decide"] else "PENDING")
    return res(st, overall=overall, endpoint_check_status=end_status, **out)


# ------------------------------------------------------------------------------------------------ integrity and timing
def check_complete(inv):
    bad, incomplete, extra, quarantined, n_ok = [], [], [], [], 0
    for name, d in sorted(inv.items()):
        cp = d / "COMPLETE.json"
        q = ".quarantined" in name
        if q:
            quarantined.append(name)
        if not cp.exists():
            incomplete.append(name)
            continue
        c = jload(cp)
        files = c.get("files", {})
        good = True
        for f, h in files.items():
            p = d / f
            if not p.exists() or sha_file(p) != h:
                bad.append(f"{name}/{f}")
                good = False
        present = {str(p.relative_to(d)) for p in d.rglob("*") if p.is_file()} - {"COMPLETE.json"}
        ex = sorted(present - set(files))
        if ex:
            extra.append({name: ex})
        if not q and c.get("id") != name:
            bad.append(f"{name}: COMPLETE id {c.get('id')}")
            good = False
        n_ok += int(good)
    st = "FAIL" if bad else ("WARN" if extra or incomplete else "PASS")
    return res(st, units_with_complete=n_ok, hash_failures=bad, files_not_listed=extra,
               units_without_complete_json=incomplete, quarantined_receipts=quarantined)


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


def activity_starts():
    p = RUN / "ACTIVITY_LOG.jsonl"
    ev = {}
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(e.get("event", "")).startswith("start "):
                t = parse_iso(e["at"])
                ev.setdefault(e["event"], []).append(t)
    for lg in RUN.glob("w*_assess.log"):           # the assessment workers' logs (created when the workers start)
        st = lg.stat()
        ev.setdefault("assess log created", []).append(utc(getattr(st, "st_birthtime", st.st_mtime)))
    return {k: min(v) for k, v in ev.items()}


GOVERNANCE = [
    ("DATA_AND_ENGINEERING_LOCK.json", ["start warm", "start parity", "start taskline", "start raw"],
     ("warm__", "parity__", "tl__", "raw__", "run__raw__")),
    ("PHASE_A_PROTOCOL_LOCK.json", ["start phaseA"], ("A__", "run__A__")),
    ("AMENDMENT_A1.json", ["start inner", "start controls"], ("inner__", "controls__")),
    ("PHASE_A_SELECTION.json", ["start calibrate", "start preflight", "start phaseB"], ("calib__", "preflight__", "B__", "run__B__")),
    ("AMENDMENT_A2.json", ["start calibrate", "start preflight"], ("calib__", "preflight__")),
    ("AMENDMENT_A3.json", ["start phaseB"], ("B__", "run__B__")),
    ("PHASE_B_PROTOCOL_LOCK.json", ["start phaseB"], ("B__", "run__B__", "lc__", "fare__")),
    ("EVALUATION_LOCK.json", ["start assess", "start outer", "start infer", "assess log created"], ("outer__",)),
    ("AMENDMENT_A4.json", ["start assess", "assess log created"], ("outer__",)),
]


def check_locks(inv):
    entries = remote_reflog()
    starts = activity_starts()
    # earliest estimated start (COMPLETE.json mtime - recorded wall_s) per governed unit prefix
    prefixes = {g for _, _, gs in GOVERNANCE for g in gs}
    ustart = {}
    for name in inv:
        if ".quarantined" in name or not unit_done(name):
            continue
        done, st = unit_times(name)
        t = st or done
        for pre in prefixes:
            if name.startswith(pre):
                ustart[pre] = min(ustart.get(pre, t), t)
    timing, all_amend = {}, sorted(p.name for p in RES.glob("AMENDMENT_A*.json"))
    gov = list(GOVERNANCE)
    for a in all_amend:
        if a not in [g[0] for g in gov]:
            gov.append((a, [], ()))
    for fname, evs, prefixes in gov:
        rel = f"results/pcrl_strength_matched_feedback_v1/{fname}"
        p = RES / fname
        if not p.exists():
            timing[fname] = res("PENDING", reason="not written")
            continue
        log = git("log", "--format=%H|%cI", "--", rel) or ""
        commits = [l.split("|") for l in log.splitlines() if l]
        if not commits:
            timing[fname] = res("FAIL", reason="not committed")
            continue
        first_c, first_t = commits[-1]
        last_c, last_t = commits[0]
        fp, _ = first_remote(first_c, entries)
        lp, _ = first_remote(last_c, entries)
        blob_ok = git("rev-parse", f"{last_c}:{rel}") == git("hash-object", str(p))
        gov_ev = {e: starts[e] for e in evs if e in starts}
        gov_u = {pre: ustart[pre] for pre in prefixes if pre in ustart}
        first_gov = min(list(gov_ev.values()) + list(gov_u.values()), default=None)
        rec = {"first_commit": first_c, "first_commit_time": parse_iso(first_t), "first_push_time": fp,
               "latest_commit": last_c, "latest_push_time": lp, "worktree_equals_latest_commit": blob_ok,
               "governed_stage_starts": gov_ev, "governed_unit_earliest_start": gov_u, "first_governed_start": first_gov}
        if first_gov is None:
            stt = "PENDING" if fp else "WARN"
        else:
            ok_first = fp is not None and fp < first_gov
            ok_last = lp is not None and lp < first_gov
            rec["pushed_before_first_governed_start"] = ok_first
            rec["latest_version_pushed_before_first_governed_start"] = ok_last
            stt = "PASS" if ok_first and ok_last and blob_ok else ("WARN" if ok_first and blob_ok else "FAIL")
        timing[fname] = res(stt, **rec)
    # amendment A3 / preflight receipts: report receipts produced under amended code before the amendment was pushed
    notes = []
    a3 = timing.get("AMENDMENT_A3.json", {})
    if a3.get("first_push_time"):
        pf = [(n, unit_times(n)[0]) for n in inv if re.fullmatch(r"preflight__s\d", n) and unit_done(n)]
        early = [n for n, t in pf if t and t < a3["first_push_time"]]
        if early:
            notes.append(f"{len(early)} final preflight receipts completed before AMENDMENT_A3 was pushed "
                         f"({iso(a3['first_push_time'])}); A3 file written_at {jload(RES / 'AMENDMENT_A3.json').get('written_at')}; "
                         "Phase B started after the push")
    # EVALUATION_LOCK before first outer
    ev = timing.get("EVALUATION_LOCK.json", {})
    outer = sorted(n for n in inv if n.startswith("outer__"))
    if outer and ev.get("status") == "PENDING" and ev.get("reason") == "not written":
        timing["EVALUATION_LOCK.json"] = res("FAIL", reason="outer units exist without EVALUATION_LOCK", outer_units=outer)
    return timing, notes


def check_lock_pins(D):
    """EVALUATION_LOCK pins: scored-unit file hashes, lock/amendment file hashes, selection hashes, assessment rows."""
    p = RES / "EVALUATION_LOCK.json"
    if not p.exists():
        return res("PENDING", reason="EVALUATION_LOCK not written")
    EL = jload(p)
    bad = []
    n_units = 0
    for k, sk in EL.get("seeds", {}).items():
        for u, files in (sk.get("unit_file_sha256") or {}).items():
            n_units += 1
            cp = UNITS / u / "COMPLETE.json"
            if not cp.exists():
                bad.append(f"{u}: no COMPLETE.json")
                continue
            if jload(cp).get("files") != files:
                bad.append(f"{u}: COMPLETE hashes differ from lock pin")
        scored = set()
        for spec in (sk.get("score") or {}).values():
            scored |= set([spec["unit"]] if spec.get("unit") else spec.get("units") or [])
        unpinned = sorted(scored - set(sk.get("unit_file_sha256") or {}))
        if unpinned:
            bad.append(f"seed {k}: scored units without pins {unpinned}")
    for grp in ("locks_sha256", "amendments_sha256"):
        for f, h in (EL.get(grp) or {}).items():
            q = RES / f
            if not q.exists() or sha_file(q) != h:
                bad.append(f"{grp}: {f} changed since EVALUATION_LOCK")
    present_amend = sorted(q.name for q in RES.glob("AMENDMENT_A*.json"))
    later = [a for a in present_amend if a not in (EL.get("amendments_sha256") or {})]
    for name, f in (("selection_A_sha256", RUN / "selection_A.json"), ("selection_B_sha256", RUN / "selection_B.json")):
        if EL.get(name) and (not f.exists() or sha_file(f) != EL[name]):
            bad.append(f"{name} mismatch")
    ar = EL.get("assessment_role") or {}
    am = D.mask[ASSESS]
    if ar and (ar.get("rows") != int(am.sum()) or ar.get("groups") != int(len(np.unique(D.unit[am])))
               or ar.get("row_id_sha256") != rowid_hash(D.row_id[am])):
        bad.append("assessment_role rows/hash")
    return res("FAIL" if bad else "PASS", units_pinned=n_units, differences=bad,
               amendments_after_evaluation_lock=later)


def check_code_hashes():
    order = []
    for fname in ["DATA_AND_ENGINEERING_LOCK.json", "PHASE_A_PROTOCOL_LOCK.json"] + \
            sorted(p.name for p in RES.glob("AMENDMENT_A*.json")) + ["PHASE_B_PROTOCOL_LOCK.json", "EVALUATION_LOCK.json"]:
        p = RES / fname
        if p.exists():
            d = jload(p)
            order.append((d.get("written_at", ""), fname, d))
    order.sort()
    files, current, history, violations = {}, {}, {}, []
    for _, fname, d in order:
        cf = d.get("code_files") or d.get("locked_code_files") or {}
        changed_ok = set(d.get("changes_previously_locked", []))
        prev_excluded = set()
        for f, h in cf.items():
            if f in current and current[f][0] != h and f not in changed_ok and fname.startswith("AMENDMENT"):
                violations.append(f"{fname}: {f} changed without being declared")
            if f in current and current[f][0] != h and not fname.startswith("AMENDMENT"):
                history.setdefault(f, []).append(f"{fname} re-locks a changed hash (previous {current[f][1]})")
            current[f] = (h, fname)
    mism = []
    for f, (h, src) in sorted(current.items()):
        p = WT / f
        wh = sha_file(p) if p.exists() else None
        if wh != h:
            mism.append({"file": f, "governing_lock": src, "locked": h, "worktree": wh})
    latest_named = next((fn for _, fn, _ in reversed(order) if not fn.startswith("AMENDMENT")), None)
    docs = {}
    if latest_named:
        d = jload(RES / latest_named)
        for doc, h in (d.get("documents_sha256") or {}).items():
            p = RES / doc
            docs[doc] = (sha_file(p) == h) if p.exists() else None
    context = {}
    if history:
        ent = remote_reflog()
        for f in history:
            log = git("log", "--format=%H", "--", "results/pcrl_strength_matched_feedback_v1/PHASE_A_PROTOCOL_LOCK.json") or ""
            c = log.splitlines()[-1] if log else None
            pt, _ = first_remote(c, ent) if c else (None, None)
            before = {}
            for pre in ("warm__", "tl__", "raw__", "run__raw__", "parity__"):
                ts = [unit_times(n)[0] for n in os.listdir(UNITS) if n.startswith(pre) and ".quarantined" not in n and unit_done(n)]
                if ts and pt:
                    before[pre] = {"units": len(ts), "completed_before_relock_push": sum(t < pt for t in ts)}
            context[f] = {"relock_push_time": pt, "units_by_prefix": before,
                          "note": "the changed code paths are the controller probe (r-block whitened LR, rank tol 1e-9), "
                                  "zero-direction handling, per-epoch logging and snapshots (review R1/R2/R4); the "
                                  "task-only update is unchanged apart from logging; parity receipts were re-run after the "
                                  "relock; no dated amendment records the change"}
    st = "FAIL" if (mism or violations) else ("WARN" if history else "PASS")
    return res(st, relock_context=context, locks_in_order=[fn for _, fn, _ in order], files_governed=len(current), worktree_mismatches=mism,
               undeclared_changes=violations, relock_notes=history, latest_named_lock=latest_named,
               latest_named_lock_documents_match=docs)


# ------------------------------------------------------------------------------------------------ drive restore
def sha_uncached(p):
    """sha256 with F_NOCACHE on the descriptor where supported (an uncached read is not necessarily a cold read)."""
    import fcntl
    fd = os.open(p, os.O_RDONLY)
    nocache = False
    try:
        try:
            fcntl.fcntl(fd, getattr(fcntl, "F_NOCACHE", 48), 1)
            nocache = True
        except OSError:
            pass
        h = hashlib.sha256()
        while True:
            b = os.read(fd, 1 << 20)
            if not b:
                break
            h.update(b)
    finally:
        os.close(fd)
    return h.hexdigest(), nocache


def check_drive_restore(D, drive_root, seed=1, labels=("U", "J-F", "L-F")):
    if not drive_root:
        return res("PENDING", reason="no --drive-root given")
    root = Path(drive_root)
    if not root.exists():
        return res("PENDING", drive_root="<DRIVE_ROOT>",
                   reason="drive root not present when the verifier ran (external volume not mounted); restore test not executed")
    udir = next((c for c in (root / "run" / "units", root / "units") if c.exists()), None)
    if udir is None:
        return res("FAIL", reason="no units directory under <DRIVE_ROOT>")
    EL = jload(RES / "EVALUATION_LOCK.json")
    sk = EL["seeds"][str(seed)]
    per, fails = {}, []
    for lab in labels:
        u = sk["score"][lab]["unit"]
        dd = udir / u
        info = {"label": lab, "unit": u}
        if not (dd / "COMPLETE.json").exists():
            info["status"] = "FAIL"
            info["reason"] = "unit missing on drive"
            per[lab] = info
            fails.append(lab)
            continue
        dc = jload(dd / "COMPLETE.json")
        info["drive_complete_equals_local"] = dc == jload(UNITS / u / "COMPLETE.json")
        info["drive_complete_equals_lock_pin"] = dc.get("files") == (sk.get("unit_file_sha256") or {}).get(u)
        hs = {f: sha_uncached(dd / f) for f in dc["files"]}
        info["drive_file_hashes_ok"] = all(h == dc["files"][f] for f, (h, _) in hs.items())
        info["uncached_reads"] = all(nc for _, nc in hs.values())
        sd = load_state(dd)
        heads = [joblib.load(dd / f"head_{i}.joblib") for i in (0, 1)]
        loc = np.load(UNITS / u / "release.npz", allow_pickle=False)
        diffs = {}
        for i, (rk, ck, pk, hk) in enumerate((("r1", "c1", "p1", "hard1"), ("r2", "c2", "p2", "hard2"))):
            r = encode(sd, i, D.X)
            diffs[f"recipient_{i + 1}"] = {"features_max_abs_diff": maxdiff(r, loc[rk]),
                                           "centred_logits_max_abs_diff": maxdiff(centred_logits(heads[i], r), loc[ck]),
                                           "prob_max_abs_diff": maxdiff(heads[i].predict_proba(r), loc[pk]),
                                           "hard_mismatches": int((heads[i].predict(r) != loc[hk]).sum())}
        info["replay_from_drive_vs_local_release"] = diffs
        ok = (info["drive_complete_equals_local"] and info["drive_complete_equals_lock_pin"] and info["drive_file_hashes_ok"]
              and all(v["features_max_abs_diff"] == 0 and v["centred_logits_max_abs_diff"] <= 1e-12 and
                      v["prob_max_abs_diff"] <= 1e-12 and v["hard_mismatches"] == 0 for v in diffs.values()))
        info["status"] = "PASS" if ok else "FAIL"
        if not ok:
            fails.append(lab)
        per[lab] = info
    return res("FAIL" if fails else "PASS", drive_root="<DRIVE_ROOT>", seed=seed, restored=per,
               note="inputs X rebuilt from the source npz by this verifier; parameters and heads read from the drive copy only")


def check_audit_controls():
    p = RES / "AUDIT_CONTROLS.json"
    if not p.exists():
        return res("PENDING", reason="AUDIT_CONTROLS.json not written")
    A = jload(p)
    per = {}
    for lab, v in A.items():
        if not isinstance(v, dict):
            continue
        rp = v.get("rotated_plant") or {}
        per[lab] = {"all_ok": v.get("all_ok"), "n_failures": len(v.get("failures") or []),
                    "rotated_plant": {k: x for k, x in rp.items() if isinstance(x, (bool, int, float, str))} if isinstance(rp, dict) else rp}
    ok = bool(A.get("all_ok")) and all(x["all_ok"] for x in per.values())
    return res("INFO" if ok else "WARN", all_ok=A.get("all_ok"), per_label=per,
               note="audit owner's attacker controls echoed, not independently recomputed by this verifier")


# ------------------------------------------------------------------------------------------------ critic tracking
def check_tracking(inv):
    names = sorted(n for n in inv if n.startswith("track__") and unit_done(n))
    if not names:
        return res("PENDING", reason="no track__ records yet")
    bad, table = [], []
    for n in names:
        rec = unit_rec(n)
        snaps = dict(rec.get("snapshots", {}))
        if isinstance(rec.get("final_refit_at_theta_T"), dict):
            snaps["final_refit_at_theta_T"] = rec["final_refit_at_theta_T"]
        for sn, sv in snaps.items():
            for view, vr in sv.items():
                if not isinstance(vr, dict) or "kinds" not in vr:
                    continue
                kinds = vr["kinds"]
                for rows in ("calib", "inner"):
                    on = [kinds[k]["online"][rows] for k in sorted(kinds)]
                    fr_ = [kinds[k]["fresh"][rows] for k in sorted(kinds)]
                    reg = float(np.mean([a - b for a, b in zip(on, fr_)]))
                    bob = min(on) - min(fr_)
                    r_pub, b_pub = vr.get(f"gap_registered_{rows}"), vr.get(f"gap_best_of_bank_{rows}")
                    if r_pub is not None and abs(reg - r_pub) > 1e-9:
                        bad.append(f"{n}/{sn}/{view}/{rows} registered {r_pub} vs {reg}")
                    if b_pub is not None and abs(bob - b_pub) > 1e-9:
                        bad.append(f"{n}/{sn}/{view}/{rows} best-of-bank {b_pub} vs {bob}")
                    table.append({"unit": n, "run": rec.get("run"), "label": rec.get("label"), "seed": rec.get("seed"),
                                  "snapshot": sn, "view": view, "rows": rows, "gap_registered": reg,
                                  "gap_best_of_bank": bob, "n_kinds": len(kinds)})
    comp, cdiff, ncmp = "PENDING", [], 0
    p = RES / "CRITIC_TRACKING.csv"
    if p.exists():
        idx = {(t["run"], t["snapshot"], t["view"], t["rows"]): t for t in table}
        rows_ = list(csv.DictReader(open(p)))
        for r in rows_:
            t = idx.get((r["run"], r["snapshot"], r["view"], r["rows"]))
            if t is None:
                cdiff.append(f"row not in records: {r['run']}/{r['snapshot']}/{r['view']}/{r['rows']}")
                continue
            ncmp += 1
            for col, v in (("gap_registered_mean_paired", t["gap_registered"]), ("gap_best_of_bank", t["gap_best_of_bank"])):
                if abs(float(r[col]) - v) > 5e-6 * max(1.0, abs(v)):
                    cdiff.append(f"{r['run']}/{r['snapshot']}/{r['view']}/{r['rows']} {col}")
        missing = len(table) - ncmp
        comp = "FAIL" if cdiff else ("WARN" if missing else "PASS")
    summary = {}
    for t in table:
        key = f"{t['label']}|{t['snapshot']}|{t['rows']}"
        summary.setdefault(key, []).append(t["gap_registered"])
    summary = {k: {"mean_registered_gap": float(np.mean(v)), "n": len(v)} for k, v in sorted(summary.items())}
    return res(worst("FAIL" if bad else "PASS", comp if comp != "PENDING" else None), records=len(names),
               recompute_failures=bad[:30], csv_comparison=comp, csv_rows_compared=ncmp, csv_differences=cdiff[:20],
               mean_registered_gap_by_label_snapshot_rows=summary)


# ------------------------------------------------------------------------------------------------ self tests
def selftests():
    out = {}
    # import guard
    _IN_SELFTEST[0] = True
    blocked = []
    for m in ("smf.select", "rgj.train", "oar.study", "stored_model_eval.bench_infer"):
        try:
            importlib.import_module(m)
            blocked.append((m, False))
        except ImportError:
            blocked.append((m, True))
    _IN_SELFTEST[0] = False
    out["import_guard_blocks"] = res("PASS" if all(b for _, b in blocked) else "FAIL", tried=blocked)
    rng = np.random.default_rng(7)
    # weighted AUC == replicated-rows AUC (with ties); full sample == sklearn
    n = 300
    sc = np.round(rng.normal(size=n), 1)
    y = (rng.random(n) < 0.4).astype(int)
    errs = []
    for _ in range(5):
        w = rng.integers(0, 4, n)
        mine = wauc(sc, y, w[None, :].astype(float))[0]
        ref = roc_auc_score(np.repeat(y, w), np.repeat(sc, w))
        errs.append(abs(mine - ref))
    errs.append(abs(wauc(sc, y, np.ones((1, n)))[0] - roc_auc_score(y, sc)))
    out["weighted_auc"] = res("PASS" if max(errs) < 1e-12 else "FAIL", max_abs_err=max(errs))
    # bootstrap draws: chunked sequence, row sums
    units = rng.integers(0, 120, 400)
    b1 = Boot(units, B=600, seed=11, chunk=250)
    b2 = Boot(units, B=600, seed=11, chunk=600)
    out["bootstrap_draws"] = res("PASS" if (b1.counts.sum(1) == b1.G).all() else "FAIL",
                                 chunked_equals_single_call=bool(np.array_equal(b1.counts, b2.counts)))
    # controller: binding fixture, asymmetric, selectivity, RMS identity
    b = np.array([0.70, 0.70])
    ref = np.array([0.71, 0.71])
    ctrl = []
    wa, wh = np.ones(2), np.ones(2)
    for e in range(21):
        auc = ref if e == 0 else np.array([0.72, 0.69])
        v = auc - b
        st = np.clip(v / 0.01, -1, 1)
        wb = wa.copy()
        if e < 20:
            wa = np.clip(wa + st, 0.25, 8)
            wh = np.clip(wh + st, 0.25, 8)
        ctrl.append({"epoch": e, "auc": auc.tolist(), "b": b.tolist(), "v": v.tolist(), "w_before": wb.tolist(),
                     "w_after": wa.tolist(), "w_hypothetical": wh.tolist(), "diagnostic_only": e == 20,
                     "reused_reference": e == 0, "allocation": (wa / math.sqrt((wa ** 2).mean())).tolist()})
    ea, eh, al, f = controller_replay(ctrl, b, ref, True)
    st_l, first, _ = activation_status("local", ea, al)
    rms_ok = all(abs(math.sqrt((al[e] ** 2).mean()) - 1) < 1e-12 for e in al)
    sel_ok = ea[1][1] < ea[0][1] and ea[1][0] > ea[0][0]
    bad = [dict(c) for c in ctrl]
    bad[3] = dict(bad[3], w_after=[9.0, 9.0])
    _, _, _, f2 = controller_replay(bad, b, ref, True)
    out["controller_replay"] = res("PASS" if (not f and f2 and st_l == "ACTIVE" and first == 1 and rms_ok and sel_ok
                                              and ea[0].tolist() == [2.0, 2.0] and ea[3].tolist() == [5.0, 0.25]) else "FAIL",
                                   activation=st_l, first_asymmetric=first, corrupted_trace_detected=bool(f2))
    common = {e: np.array([2.0, 2.0]) for e in range(20)}
    calloc = {e: np.ones(2) for e in range(20)}
    out["activation_common_mode"] = res("PASS" if activation_status("local", common, calloc)[0] == "INACTIVE_OR_ALIAS" and
                                        activation_status("joint", common, calloc)[0] == "ACTIVE_COMMON_ONLY" and
                                        activation_status("joint", {e: np.ones(2) for e in range(20)}, calloc)[0] == "INACTIVE"
                                        else "FAIL")
    # gradient identities
    def mk(rho, s, z, cap=0):
        n = 61
        r = rho * s if cap == 0 else rho * s * 0.9
        return {"epoch": 0, "n_steps": n, "clip": 0, "kappa_min": 1.0, "w": [1.0, 1.0], "alloc": [1.0, 1.0],
                **{f"zero_{i}": z for i in (1, 2)}, **{f"cap_{i}": cap for i in (1, 2)},
                **{f"ratio_mean_{i}": r for i in (1, 2)}, **{f"realized_ratio_mean_{i}": r * (n - z) / n for i in (1, 2)}}
    good = {"rho": 0.75, "diag": {"epochs": [mk(0.75, 1, 3)], "encoder_updates": 61, "zero_events": [3, 3],
                                  "cap_hits": [0, 0], "clip_hits": 0}}
    badr = {"rho": 0.75, "diag": {"epochs": [dict(mk(0.75, 1, 3), realized_ratio_mean_1=0.5)], "encoder_updates": 61,
                                  "zero_events": [3, 3], "cap_hits": [0, 0], "clip_hits": 0}}
    capr = {"rho": 0.75, "diag": {"epochs": [mk(0.75, 1, 0, cap=2)], "encoder_updates": 61, "zero_events": [0, 0],
                                  "cap_hits": [2, 2], "clip_hits": 0}}
    g1, g2, g3 = gradient_check(good), gradient_check(badr), gradient_check(capr)
    out["gradient_identities"] = res("PASS" if g1["status"] == "PASS" and g2["status"] == "FAIL" and g3["status"] == "PASS"
                                     and any("cap active" in x for x in g3["flags"]) else "FAIL")
    # selection: ties to lower rho, exact gates
    pts = [{"unit": "a", "rho": 0.75, "auc": {"pair": 0.8}, "feasible": True},
           {"unit": "b", "rho": 0.25, "auc": {"pair": 0.8}, "feasible": True},
           {"unit": "c", "rho": 1.5, "auc": {"pair": 0.7}, "feasible": False}]
    fe = [p for p in pts if p["feasible"]]
    pick = min(fe, key=lambda p: (p["auc"]["pair"], p["rho"]))["unit"]
    cand = {0: {"k": 849, "n": 1000, "k_const": 760}, 1: {"k": 490, "n": 1000, "k_const": 283}}
    refc = {0: {"k": 859, "n": 1000, "k_const": 760}, 1: {"k": 490, "n": 1000, "k_const": 283}}
    mg, mf = gate_margins(cand, refc)
    out["selection_rules"] = res("PASS" if pick == "b" and mg[0]["G1"] == 0 and feasible(mg) else "FAIL",
                                 boundary_G1_exact=str(mg[0]["G1"]), boundary_G1_float=mf[0]["G1"])
    # endpoint plumbing on synthetic predictions
    fam = jload(RES / "PRIMARY_FAMILY.json")
    n = 240
    units = np.arange(n) // 2
    sex = (rng.random(n) < 0.33).astype(int)
    yi = (rng.random(n) < 0.25).astype(int)
    yo = rng.integers(0, 6, n)
    race = rng.integers(0, 3, n)

    def synth(shift):
        P = {}
        for w in ("v1", "v2", "pair", "p1", "p2", "ppair", "h1", "h2", "hpair"):
            s = 1 / (1 + np.exp(-(shift * (2 * sex - 1) + rng.normal(size=(3, n)))))
            P[f"P_auc_{w}"] = np.stack([1 - s, s], 2)
            P[f"P_ce_{w}"] = P[f"P_auc_{w}"]
        for w in ("v1", "v2", "pair"):
            q = rng.random((3, n, 3))
            P[f"Prace_auc_{w}"] = q / q.sum(2, keepdims=True)
        return {**P, "sex": sex, "race": race, "y_income": yi, "y_occ": yo, "assess_unit": units,
                "hard1": np.where(rng.random(n) < 0.85, yi, 1 - yi), "hard2": np.where(rng.random(n) < 0.5, yo, 0),
                "race_pos": np.arange(n), "race_y": race}
    base = synth(1.0)
    preds = {}
    labels = ["J-F", "L-F", "J-N", "L-N", "RAW-J", "RAW-L", "U", "E", "F", "F0", "A-ONLINE@r0.75", "A-REFRESHED@r0.75",
              "A-MATCHED@r0.75", "J-F@r0.75", "L-F@r0.75", "J-N@r0.75", "L-N@r0.75"]
    for k in SEEDS:
        for lab in labels:
            preds[(k, lab)] = base if lab in ("J-F", "L-F", "J-F@r0.75", "L-F@r0.75") else synth(0.5)
    bt = Boot(units, B=200, seed=3)
    m, _ = compute_endpoints(fam, preds, {k: "RAW-J" for k in SEEDS}, bt, {0: 0, 1: 0}, np.array([0.67, 0.33]))
    direct = np.mean([np.mean([roc_auc_score(sex, preds[(k, "RAW-J")]["P_auc_pair"][a, :, 1]) for a in range(3)])
                      for k in SEEDS]) - np.mean([roc_auc_score(sex, base["P_auc_pair"][a, :, 1]) for a in range(3)])
    ok = (abs(m["P01"]["point"]) == 0 and m["P01"]["se"] == 0 and abs(m["P10"]["point"] - direct) < 1e-12
          and m["P13"]["point"] == m["P04"]["point"] and len(m) == 18 + 34
          and all(r.get("n_finite_replicates", 0) == 200 for r in m.values()))
    out["endpoint_pipeline"] = res("PASS" if ok else "FAIL", n_slots=len(m), identical_arms_point=m["P01"]["point"],
                                   identical_arms_se=m["P01"]["se"])
    # LLR definition: 1 - mean(-log P_ce[s]) / mean(-log prior[s]); prior predictor -> 0, sharper predictor -> > 0
    pri = np.array([0.67, 0.33])
    sx = (rng.random(400) < 0.33).astype(int)
    lv = Levels(Boot(np.arange(400), B=20, seed=1), 400)
    lv.add_llr("prior", -np.log(pri[sx]), -np.log(pri[sx]))
    sharp = np.where(sx == 1, 0.9, 0.1)
    psh = np.stack([1 - sharp, sharp], 1)[np.arange(400), sx]
    lv.add_llr("sharp", -np.log(psh), -np.log(pri[sx]))
    vv = lv.run()
    direct = 1 - log_loss(sx, np.stack([1 - sharp, sharp], 1)) / log_loss(sx, np.tile(pri, (400, 1)))
    out["llr_definition"] = res("PASS" if abs(vv["prior"][0]) < 1e-12 and abs(vv["sharp"][0] - direct) < 1e-12
                                and np.allclose(vv["prior"][1], 0) else "FAIL", prior_llr=vv["prior"][0],
                                sharp_llr=vv["sharp"][0], sklearn_reference=direct)
    z1 = float(norm.ppf(1 - 0.05 / 36))
    z2 = float(norm.ppf(1 - 0.05 / 68))
    out["z_values"] = res("PASS" if abs(z1 - 2.991316) < 5e-7 and abs(z1 - fam["z_primary"]) < 1e-12
                          and abs(z2 - fam["z_secondary"]) < 1e-12 and fam["primary_size"] == 18
                          and fam["secondary_size"] == len(fam["secondary"]) == 34 else "FAIL", z_primary=z1, z_secondary=z2)
    # forward pass reproduces torch.nn.Sequential bit-exactly
    seq = torch.nn.Sequential(torch.nn.Linear(83, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(),
                              torch.nn.Linear(64, 16))
    sd = {f"enc.0.{k}": v for k, v in seq.state_dict().items()}
    Xs = rng.normal(size=(50, 83)).astype(np.float32)
    with torch.no_grad():
        ref_ = seq(torch.from_numpy(Xs)).numpy().astype(np.float64)
    out["forward_pass"] = res("PASS" if np.array_equal(encode(sd, 0, Xs), ref_) else "FAIL")
    return out


# ------------------------------------------------------------------------------------------------ main
DRIVE_ROOT = [None]


def scrub_check(text: str):
    home = str(Path.home())
    probes = [home, "/Users/", "/Volumes/", home.split("/")[-1]]
    if DRIVE_ROOT[0]:
        probes.append(str(DRIVE_ROOT[0]))
        probes += [c for c in Path(DRIVE_ROOT[0]).parts if len(c) > 3 and c not in ("/", "Volumes", "smf_v1")]
    low = text.lower()
    bad = [s for s in probes if s and s.lower() in low]
    if bad:
        raise RuntimeError(f"refusing to write identifying strings into the public JSON: {len(bad)} hits")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest-only", action="store_true")
    ap.add_argument("--full", action="store_true", help="head refits and LR spot checks on every unit")
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--out", default=None, help="alternative output path (testing)")
    ap.add_argument("--drive-root", default=None, help="backup copy root (published as <DRIVE_ROOT>)")
    args = ap.parse_args()
    DRIVE_ROOT[0] = args.drive_root
    t0 = time.time()
    report = {"schema": "smf-independent-verification-v1", "phase": "PHASE_2" if lock_state().get("verified") else "PHASE_1",
              "generated_at": iso(datetime.now(timezone.utc)),
              "verifier": "results/pcrl_strength_matched_feedback_v1/verification/replay_smf.py",
              "verifier_sha256": sha_file(Path(__file__)),
              "worktree_head": git("rev-parse", "HEAD"),
              "inputs": {"source_npz": "<PRIVATE_CACHE>/jcv_v1/inputs/adult_jcv.npz",
                         "units": "<PRIVATE_CACHE>/smf_v1/run/units", "selections": "<PRIVATE_CACHE>/smf_v1/run"},
              "libraries": {"numpy": np.__version__, "torch": torch.__version__, "joblib": joblib.__version__,
                            "sklearn": __import__("sklearn").__version__, "scipy": __import__("scipy").__version__}}
    checks = {}
    checks["selftests"] = st = selftests()
    st_status = rollup(st)
    if args.selftest_only:
        report["checks"] = {"selftests": {"status": st_status, **st}}
        print(json.dumps(jsonable(report["checks"]), indent=1))
        return
    D = Data()
    L = D.labels()
    inv = {p.name: p for p in sorted(UNITS.iterdir()) if p.is_dir()}
    checks["roles"] = check_roles(D)
    # releases (+ inner consistency on the same loaded release)
    rel_units = sorted(n for n in inv if ".quarantined" not in n and unit_done(n) and (inv[n] / "release.npz").exists())
    refs = set()
    for f in (RES / "PHASE_A_SELECTION.json",):
        if f.exists():
            S = jload(f)
            refs |= {S["seeds"][str(k)]["reference"].get("unit") for k in SEEDS}
            refs |= {u for s in S.get("per_schedule_selected", {}).values() for u in s.values()}
    sample = set(refs) | {n for n in rel_units if n.startswith(("tl__", "B__", "lc__", "fare__"))}
    sample |= set(rel_units[::6])
    rel_out, inner_out, cache = {}, {}, {}
    for n in rel_units:
        try:
            r, rel = replay_release(D, n, refit=args.full or n in sample, L=L)
        except ImportError as e:
            rel_out[n] = res("FAIL", reason=f"unpickling needed a forbidden module: {e}")
            continue
        rel_out[n] = r
        cache[n] = task_counts(D, L, rel, purpose=(unit_rec(n) or {}).get("purpose"))
        ir = unit_rec(f"inner__{n}")
        if ir is not None:
            inner_out[n] = inner_consistency(D, f"inner__{n}", ir, rel, L, lr_spot=args.full or n in sample,
                                             counts=cache[n])
    for n in sorted(inv):
        if n.startswith("inner__pair__") and unit_done(n):
            ir = unit_rec(n)
            inner_out[n] = inner_consistency(D, n, ir, None, L, False, counts=unit_counts(D, L, ir["of"], cache))
    lr_diffs = [v["LR_C1_spot_max_abs_diff"] for v in inner_out.values() if v.get("LR_C1_spot_max_abs_diff") is not None]
    heads_fit_ok = all(r.get(f"recipient_{i}", {}).get("head_scaler_mean_matches_fit_rows", True)
                       for r in rel_out.values() for i in (1, 2))
    outer_units = sorted(n for n in inv if n.startswith("outer__"))
    ls = lock_state()
    labels_unused = {
        "verifier_assessment_labels_sealed": not D.unseal_log,
        "outer_units_before_verified_lock": [] if ls.get("verified") else outer_units,
        "heads_fitted_on_NEW_DEFENSE_FIT_rows_only": heads_fit_ok,
        "inner_attackers_fit_AUDIT_FIT_select_INNER": all(v.get("n_fit_is_AUDIT_FIT", True) and v.get("n_select_is_INNER", True)
                                                          for v in inner_out.values()),
        "release_files_without_label_arrays": all(not r.get("label_like_keys") for r in rel_out.values()),
        "closed_pools_absent_from_every_release": all(r.get("dropped_row_ids_present", 1) == 0 and r.get("row_id_equals_kept_rows")
                                                      for r in rel_out.values()),
        "manifest_loader_masking": all(jload(RES / "ROLE_MANIFEST.json")["loader_comparison"]["assessment_labels_masked_to_minus_one"].values()),
    }
    lu_ok = (labels_unused["verifier_assessment_labels_sealed"] or ls.get("verified")) and not labels_unused["outer_units_before_verified_lock"] \
        and all(v for k, v in labels_unused.items() if isinstance(v, bool))
    checks["roles"]["assessment_labels_unused_before_lock"] = res("PASS" if lu_ok else "FAIL", **labels_unused)
    checks["roles"]["status"] = worst(checks["roles"]["status"], checks["roles"]["assessment_labels_unused_before_lock"]["status"])
    checks["releases"] = res(worst(*[r["status"] for r in rel_out.values()]) if rel_out else "PENDING",
                             units_replayed=len(rel_out),
                             features_bit_exact=sum(1 for r in rel_out.values() if r.get("recipient_1", {}).get("features_bit_exact")
                                                    and r.get("recipient_2", {}).get("features_bit_exact")),
                             max_feature_abs_diff=max((r.get(f"recipient_{i}", {}).get("features_max_abs_diff", 0) for r in rel_out.values()
                                                       for i in (1, 2)), default=None),
                             max_centred_logit_abs_diff=max((r.get(f"recipient_{i}", {}).get("centred_logits_max_abs_diff", 0)
                                                             for r in rel_out.values() for i in (1, 2)), default=None),
                             max_prob_abs_diff=max((r.get(f"recipient_{i}", {}).get("prob_max_abs_diff", 0) for r in rel_out.values()
                                                    for i in (1, 2)), default=None),
                             hard_decision_mismatches=sum(r.get(f"recipient_{i}", {}).get("hard_mismatches", 0)
                                                          for r in rel_out.values() for i in (1, 2)),
                             head_refits_exact=sum(1 for r in rel_out.values() for i in (1, 2)
                                                   if r.get(f"recipient_{i}", {}).get("head_refit_coef_max_abs_diff") == 0.0),
                             head_refits_run=sum(1 for r in rel_out.values() for i in (1, 2)
                                                 if "head_refit_coef_max_abs_diff" in r.get(f"recipient_{i}", {})),
                             failing_units=sorted(n for n, r in rel_out.items() if r["status"] != "PASS"),
                             note="deployment replay uses only X (83 permitted columns) and the saved parameters; no labels",
                             per_unit=rel_out)
    checks["inner_records"] = res(worst(*[v["status"] for v in inner_out.values()]) if inner_out else "PENDING",
                                  records_checked=len(inner_out),
                                  failing=sorted(n for n, v in inner_out.items() if v["status"] != "PASS"),
                                  LR_C1_spot_checks=len(lr_diffs), LR_C1_spot_max_abs_diff=max(lr_diffs, default=None),
                                  per_unit=inner_out)
    if lr_diffs and max(lr_diffs) > 1e-9 and checks["inner_records"]["status"] == "PASS":
        checks["inner_records"]["status"] = "WARN"
    g, ctrl_info = check_gradients(inv)
    g["csv_comparison"] = compare_gradient_csv(g.get("per_run", {}))
    checks["gradient_matching"] = g
    checks["controller"] = check_controller(ctrl_info)
    sa, mineA = check_selection_A(D, L, cache)
    checks["selection_A"] = sa
    sb, mineB = check_selection_B(D, L, inv, cache, mineA)
    checks["selection_B"] = sb
    ep, mine_end = check_endpoints(D, mineA, mineB)
    checks["endpoints"] = ep
    checks["conjunctions"] = check_conjunctions(mine_end, mineA, mineB, ep["status"])
    timing, notes = check_locks(inv)
    checks["integrity"] = {"complete_json": check_complete(inv), "code_hashes": check_code_hashes(),
                           "evaluation_lock_pins": check_lock_pins(D), "lock_timing": timing,
                           "timing_notes": res("WARN" if notes else "PASS", notes=notes)}
    checks["integrity"]["status"] = worst(checks["integrity"]["complete_json"]["status"],
                                          checks["integrity"]["code_hashes"]["status"],
                                          checks["integrity"]["evaluation_lock_pins"]["status"],
                                          checks["integrity"]["timing_notes"]["status"],
                                          *[v["status"] for v in timing.values()])
    checks["drive_restore"] = check_drive_restore(D, args.drive_root)
    checks["audit_controls_echo"] = check_audit_controls()
    checks["critic_tracking"] = check_tracking(inv)
    checks["selftests"] = {"status": st_status, **st}
    # independence
    loaded = sorted(m for m in sys.modules if m.split(".")[0] in _FORBIDDEN_TOP)
    req_loaded = [m for m in _REQUIRED_ABSENT if any(x == m or x.startswith(m + ".") for x in loaded)]
    report["independence"] = res("PASS" if not loaded and not _BLOCKED and not _PRELOADED else "FAIL",
                                 guard="sys.meta_path finder refusing smf, rgj, jcv, pnx, oar, stored_model_eval",
                                 required_absent=list(_REQUIRED_ABSENT), loaded_forbidden_modules=loaded,
                                 required_absent_violations=req_loaded, blocked_attempts_during_run=_BLOCKED,
                                 preloaded_before_guard=_PRELOADED, torch_load="weights_only=True",
                                 libraries_used=["numpy", "scipy", "sklearn", "torch", "joblib", "json", "hashlib", "stdlib"])
    report["checks"] = checks
    status = {k: (v.get("status") if isinstance(v, dict) else None) for k, v in checks.items()}
    counts_top, counts_all, flagged = {}, {}, []

    def walk(node, path):
        if isinstance(node, dict):
            st_ = node.get("status")
            if isinstance(st_, str) and st_ in STATUS_RANK:
                counts_all[st_] = counts_all.get(st_, 0) + 1
                if st_ in ("FAIL", "WARN", "PENDING") and not path.startswith("selftests"):
                    leaf = not any(isinstance(v, dict) and v.get("status") in ("FAIL", "WARN", "PENDING") for v in node.values())
                    if leaf:
                        flagged.append({"path": path, "status": st_,
                                        "cause": node.get("reason") or node.get("failures") or node.get("differences")
                                        or node.get("notes") or node.get("relock_notes") or node.get("flags")})
            for k, v in node.items():
                if k in ("per_unit", "per_run", "endpoints", "levels"):
                    continue
                walk(v, f"{path}.{k}" if path else k)

    for k, v in checks.items():
        if isinstance(v, dict) and v.get("status") in STATUS_RANK:
            counts_top[v["status"]] = counts_top.get(v["status"], 0) + 1
        walk(v, k)
    report["summary"] = {"status_by_check": status, "independence": report["independence"]["status"],
                         "overall": worst(*status.values(), report["independence"]["status"]),
                         "status_counts_top_level": counts_top, "status_counts_all_nodes": counts_all,
                         "flagged_fail_warn_pending": flagged, "wall_s": time.time() - t0}
    report["implemented"] = [
        "roles: exact-rational group-hash recomputation of refreshed and new roles/subroles, manifest hashes, "
        "fingerprint, disjointness, dropped pools, numeric refit, assessment-label sealing evidence",
        "releases: forward pass from model.pt + sklearn heads vs release.npz (features, centred logits, probabilities, "
        "hard decisions); head fitting rows (scaler moments, all heads), exact head refit and HEAD_VALIDATION C table (sample)",
        "inner records: bank-maximum statistic, worse/mean local, attacker roles, utility recomputed from releases, "
        "LR_C1 attacker refit spot check",
        "gradient matching: per-epoch conditional/realized identities vs rho*s, zero/cap/clip totals, RMS identity, "
        "REFRESHED refit receipts, ONLINE_MATCHED extra updates = template receipt (per bank) and run order, CSV comparison",
        "controller: full replay of applied/hypothetical weights and allocation, calib targets, activation statuses, "
        "alias (parameter equality) and pre-divergence probe agreement with twins",
        "selection A: schedule rule with exact gates, ties, fallback, local reference; compared with selection_A.json, "
        "PHASE_A_SELECTION.json, PHASE_A_FRONTIERS.csv, calib and Phase B references",
        "selection B: controls, C*, J-F nomination with +0.005 buffers and descriptive fallback; compared with "
        "selection_B.json and EVALUATION_LOCK when present",
        "endpoints: own paired multinomial group bootstrap (B=1999, seed 20261005, chunks of 250), weighted AUC/accuracy/"
        "macro race AUC/LLR, all 52 slots, z recomputed, prior hash, alias identity, comparison with CSVs and inference.json",
        "conjunctions A/B (plus the family-file variant of B requiring a valid reference)",
        "integrity: COMPLETE.json hashes, lock code hashes vs worktree, lock push (remote-tracking reflog) before first "
        "governed stage/unit, EVALUATION_LOCK before first outer unit",
        "critic tracking: registered gap and best-of-bank recomputed from track records; CRITIC_TRACKING.csv compared",
        "LEACE releases replayed from the saved official eraser maps (fit rows and means checked); FARE releases checked "
        "as one-hot cells with fit-row fingerprints and cell counts (the FARE tree itself is not re-run: the official "
        "implementation lives in a module this verifier may not import)",
        "all 954 per-seed levels recomputed and compared with inference.json and RAW_LEVELS.csv",
        "EVALUATION_LOCK pins (scored-unit file hashes, lock/amendment hashes, selection hashes, assessment rows)",
        "drive restore of seed-1 U, J-F, L-F from <DRIVE_ROOT> with uncached reads (runs when the volume is mounted)",
    ]
    if mine_end is not None and mineB is not None:
        report["outcome_replicated"] = {
            "phase_A_schedule": mineA["schedule"] if mineA else None,
            "per_seed": {k: {"J-F": mineB[k]["arms"]["J-F"]["status"], "L-F": mineB[k]["arms"]["L-F"]["status"],
                             "C*": (mineB[k]["C*"] or {}).get("unit"), "valid_reference": mineB[k]["valid_reference"]}
                         for k in SEEDS},
            "claimA": checks["conjunctions"].get("claimA"), "claimB": checks["conjunctions"].get("claimB"),
            "overall": checks["conjunctions"].get("overall"),
            "primary_decisions": {i: mine_end[i]["decision"] for i in sorted(mine_end) if i.startswith("P")}}
    report["pending"] = sorted(k for k, v in status.items() if v == "PENDING")
    text = json.dumps(jsonable(report), indent=1, sort_keys=False)
    scrub_check(text)
    if not args.no_write:
        dest = Path(args.out) if args.out else OUT
        tmp = dest.with_suffix(".json.tmp")
        tmp.write_text(text + "\n")
        tmp.replace(dest)
    brief = {k: v for k, v in report["summary"].items()}
    print(json.dumps(jsonable(brief), indent=1))


if __name__ == "__main__":
    main()
