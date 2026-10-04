#!/usr/bin/env python
"""Independent replay of the refreshed guarded joint study (pcrl_refreshed_guarded_joint_v1).

Owner: the independent verifier. This file and INDEPENDENT_VERIFICATION.json are the verifier's only outputs.

The replay works from the registered definitions (PROTOCOL.md sections 1, 2, 4, 6, 7, 8; PRIMARY_FAMILY.json;
ROLE_MANIFEST.json; METHOD_CARD.md) and from the saved private artifacts. It does NOT use the runner's metric, family,
bootstrap, selection, inference, report, audit or assessment code: an import guard on sys.meta_path refuses
rgj.select, rgj.infer, rgj.family, rgj.eval_lock, rgj.report, rgj.critic_track, rgj.whiten_diag, rgj.audit,
rgj.assess, jcv.infer, jcv.select, pnx.*, oar.*, stored_model_eval.bench_infer and stored_model_eval.pilot_infer,
including through unpickling, and the end of the run asserts that none of them was loaded. Everything here is
re-implemented with numpy / scipy / sklearn / torch / joblib.

    OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python \
        results/pcrl_refreshed_guarded_joint_v1/verification/replay_rgj.py [--selftest] [--no-real] [--spot N]

Checks (each PASS / FAIL / WARN / PENDING / SKIP; absent study outputs make a check PENDING, never FAIL):
  1 roles        input hash, role rule from PROTOCOL section 1, row-id / group hashes vs ROLE_MANIFEST.json,
                 disjointness, no dropped row in any saved npz row id, preds rows == DEVELOPMENT_ASSESSMENT.
  2 release      encoder forward pass re-implemented from model.pt (83-64-64-16, ReLU), deployed heads from
                 head_i.joblib, centred decision-function logits, probabilities and hard decisions vs release.npz;
                 deployment receives only the 83 permitted columns.
  3 selection    Stage B (L-R / L-O with the U-B alias), Stage C trained controls, single-configuration controls,
                 C*, J-G nomination with the min-guard; exact-integer utility gates recomputed from release.npz;
                 comparison with selection_B/C.json and EVALUATION_LOCK.json; inner-slate refit spot checks.
  4 endpoints    18 primary + 30 secondary slots from preds.npz; own paired multinomial bootstrap over exact-record
                 groups (B = 1999, default_rng(20261004), sequential multinomial draws); point +- z SE; decisions.
  5 conjunction  Claims A and B.
  6 multipliers  lambda replay from the logged R_calib and the calibrated c; effective weights; checkpoint lambdas.
  7 critic gap   registered (mean over kinds) and best-of-bank gaps from track records; re-evaluation of online
                 critics from captures.pt / final.pt on CALIB rows; epoch-20 R_calib recomputed from refit critics.
  8 integrity    COMPLETE.json hashes; EVALUATION_LOCK unit_file_sha256; CODE_LOCK(+amendments) vs working tree;
                 lock commit / push time vs first outer (or fit) unit.

DEVELOPMENT_ASSESSMENT labels are masked at load until EVALUATION_LOCK.json exists and outer units exist; labels of
the dropped pools (old assessment, cert, excluded) are always masked at load and never used.
"""
from __future__ import annotations

import importlib.abc
import sys

# ---------------------------------------------------------------------------------------------------------------- guard
FORBIDDEN_MODULES = ("rgj.select", "rgj.infer", "rgj.family", "rgj.eval_lock", "rgj.report", "rgj.critic_track",
                     "rgj.whiten_diag", "rgj.audit", "rgj.assess", "jcv.infer", "jcv.select",
                     "stored_model_eval.bench_infer", "stored_model_eval.pilot_infer")
FORBIDDEN_ROOTS = ("pnx", "oar")


def _forbidden(name: str) -> bool:
    if name.split(".")[0] in FORBIDDEN_ROOTS:
        return True
    return any(name == f or name.startswith(f + ".") for f in FORBIDDEN_MODULES)


class _IndependenceGuard(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.refused = []

    def find_spec(self, fullname, path=None, target=None):
        if _forbidden(fullname):
            self.refused.append(fullname)
            raise ImportError(f"independence guard: the verifier may not import {fullname}")
        return None


GUARD = _IndependenceGuard()
sys.meta_path.insert(0, GUARD)
PRELOADED_FORBIDDEN = sorted(m for m in sys.modules if _forbidden(m))

import argparse  # noqa: E402
import copy  # noqa: E402
import csv  # noqa: E402
import datetime as dt  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from scipy.stats import norm  # noqa: E402

torch.set_num_threads(1)

# ------------------------------------------------------------------------------------------------------- constants
HOME = Path.home()
HERE = Path(__file__).resolve()
RES = HERE.parents[1]
WT = HERE.parents[3]
BRANCH = "research/pcrl-refreshed-guarded-joint-v1"
RES_REL = "results/pcrl_refreshed_guarded_joint_v1"
PRIV_RUN = HOME / "PCRL_eval_cache_private" / "rgj_v1" / "run"
SRC = HOME / "PCRL_eval_cache_private" / "jcv_v1" / "inputs" / "adult_jcv.npz"
SRC_SHA = "e0d9e54af780f30788ee29cfe6795ec82cbdcadc127b1978c69a3891485d2f12"
PRED_UNITS = HOME / "PCRL_eval_cache_private" / "jcv_v1" / "run" / "units"

SEEDS = (0, 1, 2)
BETAS = (0.03, 0.1, 0.3)
EPOCHS = (5, 10, 15, 20)
REFIT_EPOCHS = (0, 4, 8, 12, 16, 20)
PROT_EPOCHS = 20
B_BOOT = 1999
BOOT_SEED = 20261004
ROLE_SEED = 20261004
NEW_ROLES = ("DEFENSE_FIT", "HEAD_VALIDATION", "DEVELOPMENT_ASSESSMENT", "AUDIT_FIT", "INNER_SELECTION")
SUBROLES = ("CRITIC_FIT", "CRITIC_VAL", "CALIB")
DROPPED = ("assessment", "cert", "excluded_exposure", "excluded_dup")
OLD_TO_NEW = {"defense_train": "DEFENSE_FIT", "attacker_fit": "AUDIT_FIT", "attacker_val": "INNER_SELECTION"}
FORBIDDEN_FEATURE_TOKENS = ("sex", "race", "income", "occupation", "fnlwgt", "row_id", "record")
TASK_LABEL = {0: "y_income", 1: "y_occupation_group"}
KS = (2, 6)
TRAINED_CONTROLS = ("L-G", "J-R", "J-O")
SINGLE_CONTROLS = ("L-R", "L-O", "U", "E", "F", "F0")
CSTAR_ORDER = ("L-G", "L-R", "L-O", "J-R", "J-O", "U", "E", "F", "F0")
SCORED = ("J-G", "L-G", "J-R", "J-O", "L-R", "L-O", "U", "E", "F", "F0", "J-G@fx", "J-R@fx", "J-O@fx")
FIXED = {"J-G@fx": "J-G", "J-R@fx": "J-R", "J-O@fx": "J-O"}
FIXED_BETA, FIXED_EPOCH = 0.1, 20
Z_PRIMARY = float(norm.ppf(1 - 0.05 / 36))
Z_SECONDARY = float(norm.ppf(1 - 0.05 / 60))
MID_STEP = 10 * 76 + 1
GUARDED = ("J-G", "L-G")
JOINT = ("J-G", "J-R", "J-O")
LOCAL = ("L-G", "L-R", "L-O")


# --------------------------------------------------------------------------------------------------------- helpers
def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def rowid_hash(r) -> str:
    """ROLE_MANIFEST convention: sha256 of the sorted int64 row ids (raw bytes)."""
    return hashlib.sha256(np.ascontiguousarray(np.sort(np.asarray(r)).astype(np.int64)).tobytes()).hexdigest()


def unit_set_hash(u) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.unique(np.asarray(u)).astype(np.int64)).tobytes()).hexdigest()


def tilde(x) -> str:
    return str(x).replace(str(HOME), "~")


def jload(p):
    return json.loads(Path(p).read_text())


def bname(b) -> str:
    return f"b{b:g}"


def safe_label(label: str) -> str:
    return label.replace("*", "star").replace("/", "_").replace(" ", "_")


def ck_name(stage, k, arm, b, e):
    return f"ck__{stage}__s{k}__{arm}__{bname(b)}__e{e}"


def run_name(stage, k, arm, b):
    return f"run__{stage}__s{k}__{arm}__{bname(b)}"


def parse_ck(name):
    m = re.match(r"^ck__([BC])__s(\d+)__(.+?)__b([0-9.eE+-]+)__e(\d+)(.*)$", name)
    if not m:
        return None
    return {"stage": m.group(1), "seed": int(m.group(2)), "arm": m.group(3), "beta": float(m.group(4)),
            "epoch": int(m.group(5)), "tag": m.group(6)}


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if math.isfinite(v) else repr(v)
    if isinstance(o, float):
        return o if math.isfinite(o) else repr(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return tilde(o)
    if isinstance(o, str):
        return tilde(o)
    return o


def torch_load(p):
    try:
        return torch.load(p, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001  (guard stays active while unpickling)
        return torch.load(p, map_location="cpu", weights_only=False)


def flatten(o, path=()):
    """(path tuple, leaf) pairs of a JSON-like object."""
    if isinstance(o, dict):
        for k, v in o.items():
            yield from flatten(v, path + (str(k),))
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            yield from flatten(v, path + (str(i),))
    else:
        yield path, o


def num(x):
    """A float from a number or from a small dict carrying one (auc / value / selected)."""
    if isinstance(x, dict):
        for key in ("auc", "value", "selected_auc", "selected", "point"):
            if key in x and not isinstance(x[key], (dict, list)):
                return float(x[key])
        raise KeyError(f"no numeric field in {list(x)[:6]}")
    return float(x)


class Report:
    def __init__(self):
        self.checks = []

    def add(self, cid, section, status, detail, **data):
        assert status in ("PASS", "FAIL", "WARN", "INFO", "PENDING", "SKIP"), status
        self.checks.append({"id": cid, "section": section, "status": status, "detail": detail, **jsonable(data)})
        return status

    def summary(self):
        out = {}
        for c in self.checks:
            out[c["status"]] = out.get(c["status"], 0) + 1
        return out

    def by_id(self, cid):
        return [c for c in self.checks if c["id"] == cid]


# ------------------------------------------------------------------------------------------------------------ data
class Data:
    """Source inputs with the role partition recomputed from the PROTOCOL section 1 rule (no rgj.data import)."""

    def __init__(self, path, expected_sha=SRC_SHA, dev_open=False):
        self.path = Path(path)
        self.sha = sha_file(self.path)
        self.expected_sha = expected_sha
        if expected_sha is not None and self.sha != expected_sha:
            raise RuntimeError(f"input sha256 mismatch: {self.sha}")
        self.z = np.load(self.path, allow_pickle=False)
        self.files = list(self.z.files)
        self.row_id = self.z["row_id"].astype(np.int64)
        self.unit = self.z["unit"].astype(np.int64)
        self.role = self.z["role"].astype(str)
        self.feature_names = [str(x) for x in self.z["feature_names"]]
        self.n = len(self.row_id)
        self.arrays_read = ["row_id", "unit", "role", "feature_names"]
        self._X = None
        self._lab = {}
        self.dev_open = dev_open
        order = np.argsort(self.row_id, kind="mergesort")
        self._ord, self._sorted = order, self.row_id[order]
        self._assign()

    def _assign(self):
        uniq, inv = np.unique(self.unit, return_inverse=True)
        two64 = 2 ** 64

        def u_int(salt):
            return [int(hashlib.sha256(f"{ROLE_SEED}|{salt}|{int(u)}".encode()).hexdigest()[:16], 16) for u in uniq]

        ud, uc = u_int("dev"), u_int("critic")
        head_exact = np.array([x * 100 < 30 * two64 for x in ud])
        head_float = np.array([x / 2.0 ** 64 < 0.30 for x in ud])
        cf_exact = np.array([x * 100 < 70 * two64 for x in uc])
        cv_exact = np.array([(x * 100 >= 70 * two64) and (x * 100 < 85 * two64) for x in uc])
        cf_float = np.array([x / 2.0 ** 64 < 0.70 for x in uc])
        cv_float = np.array([0.70 <= x / 2.0 ** 64 < 0.85 for x in uc])
        old = self.role
        m = {"DEFENSE_FIT": old == "defense_train", "AUDIT_FIT": old == "attacker_fit",
             "INNER_SELECTION": old == "attacker_val"}
        dv = old == "defense_val"
        m["HEAD_VALIDATION"] = dv & head_exact[inv]
        m["DEVELOPMENT_ASSESSMENT"] = dv & ~head_exact[inv]
        df = m["DEFENSE_FIT"]
        m["CRITIC_FIT"] = df & cf_exact[inv]
        m["CRITIC_VAL"] = df & cv_exact[inv]
        m["CALIB"] = df & ~cf_exact[inv] & ~cv_exact[inv]
        self.masks = m
        self.idx = {r: np.flatnonzero(v) for r, v in m.items()}
        self.dropped_mask = np.isin(old, DROPPED)
        self.kept_mask = np.zeros(self.n, bool)
        for r in NEW_ROLES:
            self.kept_mask |= m[r]
        groups_dv = np.unique(inv[dv])
        groups_df = np.unique(inv[df])
        self.float_exact_disagreements = int((head_exact[groups_dv] != head_float[groups_dv]).sum()
                                             + (cf_exact[groups_df] != cf_float[groups_df]).sum()
                                             + (cv_exact[groups_df] != cv_float[groups_df]).sum())

    def pos_of(self, rids):
        rids = np.asarray(rids, np.int64)
        i = np.searchsorted(self._sorted, rids)
        i = np.minimum(i, self.n - 1)
        if not np.array_equal(self._sorted[i], rids):
            raise KeyError("row ids absent from the source")
        return self._ord[i]

    @property
    def X(self):
        if self._X is None:
            self._X = np.asarray(self.z["X"], dtype=np.float32)
            self.arrays_read.append("X")
        return self._X

    def label(self, name):
        """Label array with the dropped pools always masked (-1) and DEVELOPMENT_ASSESSMENT masked until opened."""
        key = (name, self.dev_open)
        if key not in self._lab:
            a = np.asarray(self.z[name]).astype(np.int64).copy()
            a[self.dropped_mask] = -1
            if not self.dev_open:
                a[self.masks["DEVELOPMENT_ASSESSMENT"]] = -1
            self._lab[key] = a
            if name not in self.arrays_read:
                self.arrays_read.append(name)
        return self._lab[key]

    def majority(self, name):
        y = self.label(name)[self.idx["DEFENSE_FIT"]]
        return int(np.argmax(np.bincount(y)))

    def sex_prior(self):
        s = self.label("sex")[self.idx["DEFENSE_FIT"]]
        return np.bincount(s, minlength=2) / len(s)


class Ctx:
    def __init__(self, units, res, src, role_manifest, wt=WT, priv_run=PRIV_RUN, B=B_BOOT, boot_seed=BOOT_SEED,
                 src_sha=SRC_SHA, synthetic=False, spot=3, sample=8, git=True, label="real"):
        self.units, self.res, self.src, self.role_manifest = Path(units), Path(res), Path(src), Path(role_manifest)
        self.wt, self.priv_run, self.B, self.boot_seed, self.src_sha = Path(wt), Path(priv_run), B, boot_seed, src_sha
        self.synthetic, self.spot, self.sample, self.git, self.label = synthetic, spot, sample, git, label
        self._data = None
        self._rec = {}
        self._release = {}

    # -- inputs
    def data(self):
        if self._data is None:
            self._data = Data(self.src, expected_sha=self.src_sha)
        return self._data

    def udir(self, name):
        return self.units / name

    def complete(self, name):
        return (self.udir(name) / "COMPLETE.json").exists()

    def rec(self, name):
        if name not in self._rec:
            p = self.udir(name) / "record.json"
            self._rec[name] = jload(p) if p.exists() else None
        return self._rec[name]

    def release(self, name):
        if name not in self._release:
            p = self.udir(name) / "release.npz"
            if not p.exists():
                return None
            z = np.load(p, allow_pickle=False)
            self._release[name] = {k: z[k] for k in z.files}
            if len(self._release) > 64:
                self._release.pop(next(iter(self._release)))
        return self._release.get(name)

    def unit_names(self, prefix=""):
        if not self.units.exists():
            return []
        return sorted(d.name for d in self.units.iterdir() if d.is_dir() and d.name.startswith(prefix)
                      and not d.name.endswith(".tmp"))

    def lock(self):
        p = self.res / "EVALUATION_LOCK.json"
        return jload(p) if p.exists() else None

    def sel(self, stage):
        p = self.priv_run / f"selection_{stage}.json"
        return jload(p) if p.exists() else None

    def outer_names(self):
        return [n for n in self.unit_names("outer__") if self.complete(n)]

    def assessment_open(self):
        return self.lock() is not None and len(self.outer_names()) > 0


# ================================================================================================ 1. roles
def check_roles(ctx: Ctx, rep: Report):
    sec = "1-roles"
    D = ctx.data()
    man = jload(ctx.role_manifest)
    rep.add("1.1-input-hash", sec, "PASS" if D.sha == ctx.src_sha == man["source"]["sha256"] else "FAIL",
            "source npz sha256 equals the admitted hash and ROLE_MANIFEST.source.sha256",
            file=tilde(D.path), sha256=D.sha)
    fn = D.feature_names
    bad = [f for f in fn if any(t in f.lower() for t in FORBIDDEN_FEATURE_TOKENS)]
    rep.add("1.2-permitted-columns", sec, "PASS" if (len(fn) == 83 and not bad) else "FAIL",
            "83 permitted input columns, none naming SEX, race, income, occupation, fnlwgt or a record key",
            n_columns=len(fn), forbidden_names=bad)
    rows = {}
    mism = []
    for r in NEW_ROLES + SUBROLES:
        ix = D.idx[r]
        mine = {"rows": int(len(ix)), "groups": int(len(np.unique(D.unit[ix]))), "row_id_sha256": rowid_hash(D.row_id[ix]),
                "group_id_set_sha256": unit_set_hash(D.unit[ix])}
        ref = man["roles"].get(r) if r in NEW_ROLES else man["defense_fit_subroles"].get(r)
        ok = ref is not None and all(ref.get(k) == v for k, v in mine.items())
        rows[r] = {**mine, "matches_manifest": ok}
        if not ok:
            mism.append(r)
    old_m = []
    for o, ref in man["old_roles"]["per_role"].items():
        ix = np.flatnonzero(D.role == o)
        mine = {"rows": int(len(ix)), "groups": int(len(np.unique(D.unit[ix]))), "row_id_sha256": rowid_hash(D.row_id[ix]),
                "group_id_set_sha256": unit_set_hash(D.unit[ix])}
        if not all(ref.get(k) == v for k, v in mine.items()):
            old_m.append(o)
    rep.add("1.3-role-hashes", sec, "PASS" if not mism and not old_m else "FAIL",
            "roles recomputed from the PROTOCOL section 1 rule (exact rational hash cut points); counts, row-id and "
            "group-set hashes equal ROLE_MANIFEST.json for the five roles, the three critic subroles and the old roles",
            roles=rows, mismatched=mism, old_role_mismatches=old_m,
            float_vs_exact_rule_disagreements=D.float_exact_disagreements)
    # disjointness
    lab = np.array(["DROP:" + o if o in DROPPED else "" for o in D.role], dtype=object)
    for r in NEW_ROLES:
        lab[D.masks[r]] = r
    unassigned = int(sum(1 for x in lab if x == ""))
    pairs = np.unique(np.array([f"{u}|{l}" for u, l in zip(D.unit.tolist(), lab.tolist())]))
    per_group = {}
    for p in pairs:
        u, l = p.split("|", 1)
        per_group.setdefault(u, set()).add(l)
    span_new = sum(1 for s in per_group.values() if len(s) > 1 and any(not x.startswith("DROP:") for x in s))
    span_drop_only = sum(1 for s in per_group.values() if len(s) > 1 and all(x.startswith("DROP:") for x in s))
    sub_lab = np.array([""] * D.n, dtype=object)
    for r in SUBROLES:
        sub_lab[D.masks[r]] = r
    df = D.idx["DEFENSE_FIT"]
    sub_groups = {}
    for u, l in zip(D.unit[df].tolist(), sub_lab[df].tolist()):
        sub_groups.setdefault(u, set()).add(l)
    sub_span = sum(1 for s in sub_groups.values() if len(s) > 1)
    part = np.array_equal(np.sort(np.concatenate([D.idx[r] for r in SUBROLES])), df)
    dv_full = np.array_equal(np.sort(np.concatenate([D.idx["HEAD_VALIDATION"], D.idx["DEVELOPMENT_ASSESSMENT"]])),
                             np.flatnonzero(D.role == "defense_val"))
    rows_unique = len(np.unique(D.row_id)) == D.n
    ok = (unassigned == 0 and span_new == 0 and sub_span == 0 and part and dv_full and rows_unique
          and not (D.kept_mask & D.dropped_mask).any()
          and span_drop_only == man.get("groups_spanning_dropped_pools_only", span_drop_only))
    rep.add("1.4-disjointness", sec, "PASS" if ok else "FAIL",
            "row ids unique; no group spans two new roles or a new role and a dropped pool; the critic subroles "
            "partition DEFENSE_FIT without splitting a group; old defense_val fully allocated; dropped rows get no role",
            rows_without_role_or_pool=unassigned, groups_spanning_a_new_role=span_new,
            groups_spanning_dropped_pools_only=span_drop_only, groups_spanning_subroles=sub_span,
            subroles_partition_defense_fit=part, defense_val_fully_allocated=dv_full)
    # saved npz row ids
    kept_ids = np.sort(D.row_id[D.kept_mask])
    kept_in_order = D.row_id[D.kept_mask]
    dropped_ids = D.row_id[D.dropped_mask]
    dev_ids = D.row_id[D.idx["DEVELOPMENT_ASSESSMENT"]]
    n_files, bad_files, order_bad, n_rel, n_pred = 0, [], [], 0, 0
    for name in ctx.unit_names():
        d = ctx.udir(name)
        for p in sorted(d.rglob("*.npz")):
            n_files += 1
            try:
                z = np.load(p, allow_pickle=False)
            except Exception as e:  # noqa: BLE001
                bad_files.append({"file": tilde(p), "error": repr(e)})
                continue
            for key in z.files:
                if "row_id" not in key:
                    continue
                rid = np.asarray(z[key]).astype(np.int64).ravel()
                if np.isin(rid, dropped_ids).any():
                    bad_files.append({"file": tilde(p), "key": key, "dropped_rows": int(np.isin(rid, dropped_ids).sum())})
                if not np.isin(rid, D.row_id).all():
                    bad_files.append({"file": tilde(p), "key": key, "unknown_rows": True})
                if p.name == "release.npz" and key == "row_id":
                    n_rel += 1
                    if not np.array_equal(np.sort(rid), kept_ids):
                        bad_files.append({"file": tilde(p), "key": key, "not_exactly_the_five_roles": True})
                    elif not np.array_equal(rid, kept_in_order):
                        order_bad.append(tilde(p))
                if p.name == "preds.npz" and key == "assess_row_id":
                    n_pred += 1
                    if not np.array_equal(rid, dev_ids):
                        bad_files.append({"file": tilde(p), "key": key, "not_DEVELOPMENT_ASSESSMENT_in_source_order": True})
    if n_files == 0:
        rep.add("1.5-no-dropped-rows-in-releases", sec, "PENDING", "no npz file in the private units yet")
    else:
        rep.add("1.5-no-dropped-rows-in-releases", sec, "PASS" if not bad_files else "FAIL",
                "every row-id array of every saved npz avoids the old assessment / cert / excluded rows; every "
                "release.npz covers exactly the five new roles; every preds.npz covers exactly DEVELOPMENT_ASSESSMENT",
                npz_files_scanned=n_files, release_files=n_rel, preds_files=n_pred, problems=bad_files[:20],
                n_problems=len(bad_files), releases_not_in_source_order=len(order_bad))


# ============================================================================================ 2. release reconstruction
def encode_state(state, i, Xt):
    h = Xt
    for j in (0, 2, 4):
        h = F.linear(h, state[f"enc.{i}.{j}.weight"], state[f"enc.{i}.{j}.bias"])
        if j < 4:
            h = torch.relu(h)
    return h


def leace_map(unit_dir: Path, i):
    p = unit_dir / f"leace_{i}" / "leace_map.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    return {k: z[k] for k in z.files}


def leace_apply(m, H):
    """Official LEACE eraser form (concept-erasure LeaceEraser, re-implemented): x - ((x - mean_x) P_r^T) P_l^T."""
    return H - ((H - m["mean_x"]) @ m["proj_right"].T) @ m["proj_left"].T


def deploy(unit_dir: Path, X_permitted: np.ndarray, erase=False):
    """Deployment from saved artefacts only: model.pt (encoders), optional leace_i/leace_map.npz (LEACE arm E) and
    head_i.joblib (deployed heads). The input is the 83 permitted columns; no label, role or row id enters."""
    assert X_permitted.ndim == 2 and X_permitted.shape[1] == 83
    state = torch_load(unit_dir / "model.pt")
    if isinstance(state, dict) and "model" in state and "enc.0.0.weight" not in state:
        state = state["model"]
    Xt = torch.from_numpy(np.ascontiguousarray(X_permitted, dtype=np.float32))
    out, heads = {}, {}
    with torch.no_grad():
        for i in (0, 1):
            r = encode_state(state, i, Xt).double().numpy()
            if erase:
                r = leace_apply(leace_map(unit_dir, i), r)
            head = joblib.load(unit_dir / f"head_{i}.joblib")
            z = np.asarray(head.decision_function(r), dtype=np.float64)
            if z.ndim == 1:
                z = np.stack([np.zeros_like(z), z], 1)
            cen = z - z.mean(1, keepdims=True)
            P = head.predict_proba(r)
            cls = np.asarray(head.classes_)
            hard = cls[P.argmax(1)] if not np.array_equal(cls, np.arange(len(cls))) else P.argmax(1)
            out.update({f"r{i + 1}": r, f"c{i + 1}": cen, f"p{i + 1}": P, f"hard{i + 1}": hard})
            heads[i] = head
    return out, heads


def head_profile(head):
    steps = [type(s).__name__ for _, s in getattr(head, "steps", [("?", head)])]
    big = 0
    for _, s in getattr(head, "steps", []):
        for v in vars(s).values():
            if isinstance(v, np.ndarray):
                big = max(big, v.size)
    return {"steps": steps, "n_features_in": int(getattr(head, "n_features_in_", -1)),
            "classes": np.asarray(head.classes_).tolist(), "largest_stored_array": int(big)}


def compare_release(ctx: Ctx, name: str, X: np.ndarray, D: Data, erase=False):
    d = ctx.udir(name)
    rel = ctx.release(name)
    pos = D.pos_of(rel["row_id"])
    mine, heads = deploy(d, X[pos], erase=erase)
    res = {"unit": name, "rows": int(len(pos)), "map": "LEACE (stored map)" if erase else "identity"}
    ok = True
    if erase:   # fitting-row linear guardedness of the stored maps (finite-sample LEACE condition)
        df = D.idx["DEFENSE_FIT"]
        where = {int(p): j for j, p in enumerate(pos)}
        jd = np.array([where[int(p)] for p in df])
        Z = np.eye(2)[D.label("sex")[df]]
        for i in (1, 2):
            R = mine[f"r{i}"][jd]
            cc = (R - R.mean(0)).T @ (Z - Z.mean(0)) / len(df)
            res[f"fit_row_crosscov_max_abs_r{i}"] = float(np.abs(cc).max())
            ok &= res[f"fit_row_crosscov_max_abs_r{i}"] < 1e-10
    for i in (1, 2):
        for key, tol in ((f"r{i}", 0.0 if not erase else 1e-12), (f"c{i}", 1e-9), (f"p{i}", 1e-12)):
            diff = float(np.max(np.abs(mine[key] - rel[key]))) if mine[key].shape == rel[key].shape else float("inf")
            res[f"max_abs_{key}"] = diff
            ok &= diff <= tol
        mism = int((np.asarray(mine[f"hard{i}"]) != np.asarray(rel[f"hard{i}"])).sum())
        res[f"hard{i}_mismatches"] = mism
        ok &= mism == 0
    prof = {i: head_profile(h) for i, h in heads.items()}
    res["heads"] = prof
    ok &= all(p["steps"] == ["StandardScaler", "LogisticRegression"] and p["n_features_in"] == 16
              and p["largest_stored_array"] <= 16 * 6 for p in prof.values())
    res["ok"] = bool(ok)
    return res


def lock_specs(lock):
    """{(seed, label): spec} from EVALUATION_LOCK.json (schema per rgj/assess.py docstring)."""
    out = {}
    if not lock:
        return out
    seeds = lock.get("seeds", {})
    for k, s in seeds.items():
        for label, spec in (s.get("score") or {}).items():
            out[(int(k), label)] = spec
    return out


def check_release(ctx: Ctx, rep: Report, lock):
    sec = "2-release"
    D = ctx.data()
    names = []
    for (k, label), spec in sorted(lock_specs(lock).items()):
        if isinstance(spec, dict) and spec.get("kind") == "neural" and spec.get("unit"):
            names.append(spec["unit"])
    scored = sorted(set(names))
    cks = [n for n in ctx.unit_names("ck__") if ctx.complete(n) and (ctx.udir(n) / "release.npz").exists()]
    cks = sorted(cks, key=lambda n: hashlib.sha256(n.encode()).hexdigest())[:ctx.sample]
    tls = [n for n in ctx.unit_names("tl__") if ctx.complete(n)][:1] if ctx.sample else []
    lcs = [n for n in ctx.unit_names("lc__") if ctx.complete(n)][:1] if ctx.sample else []
    todo = list(dict.fromkeys(scored + cks + tls + lcs))
    if not todo:
        rep.add("2.1-release-reconstruction", sec, "PENDING", "no checkpoint / task-line / scored unit exists yet")
        return
    X = D.X
    results, nonneural, fails = [], [], []
    for n in todo:
        d = ctx.udir(n)
        rec = ctx.rec(n) or {}
        identity = n.startswith(("ck__", "tl__", "tc__")) or str(rec.get("map", "")).startswith("identity")
        erase = n.startswith("lc__") and all((d / f"leace_{i}" / "leace_map.npz").exists() for i in (0, 1))
        if not (identity or erase) or not (d / "model.pt").exists() or not (d / "head_0.joblib").exists():
            nonneural.append({"unit": n, "map": rec.get("map"),
                              "files": sorted(p.name for p in d.iterdir()) if d.exists() else "missing"})
            continue
        try:
            r = compare_release(ctx, n, X, D, erase=erase)
        except Exception as e:  # noqa: BLE001
            r = {"unit": n, "ok": False, "error": f"{type(e).__name__}: {e}"}
        results.append(r)
        if not r["ok"]:
            fails.append(n)
    status = "FAIL" if fails else ("PASS" if results else "PENDING")
    rep.add("2.1-release-reconstruction", sec, status,
            "features rebuilt from model.pt with an own forward pass (83-64-64-16, ReLU); centred logits from the "
            "deployed head's decision_function (binary -> (0, d), then centred); probabilities; hard decisions; all "
            "compared with release.npz (r exact, c 1e-9, p 1e-12, hard exact) for every scored neural unit in "
            "EVALUATION_LOCK.json plus a hash-ordered checkpoint sample",
            n_checked=len(results), scored_neural_units=len(scored), failures=fails, units=results,
            not_reconstructed_non_neural=nonneural, lock_present=lock is not None)
    rep.add("2.2-deployment-label-free", sec, "PASS" if results and not fails else ("PENDING" if not results else "FAIL"),
            "the deploy() path receives only the 83 permitted columns (no label, role or row id); the deployed heads are "
            "StandardScaler + LogisticRegression on the 16 released features and store no training-row arrays",
            deploy_inputs=["model.pt", "head_0.joblib", "head_1.joblib", "X (83 permitted columns)"])


# ================================================================================================ 3. selection
def utility_counts(ctx: Ctx, name: str, D: Data):
    """Exact INNER_SELECTION utility from release.npz hard decisions: {t: (correct, const_correct, N)}."""
    rel = ctx.release(name)
    if rel is None:
        return None
    pos = D.pos_of(rel["row_id"])
    where = {int(p): j for j, p in enumerate(pos)}
    isel = D.idx["INNER_SELECTION"]
    j = np.array([where[int(p)] for p in isel])
    out = {}
    for t in (0, 1):
        y = D.label(TASK_LABEL[t])[isel]
        maj = D.majority(TASK_LABEL[t])
        h = np.asarray(rel[f"hard{t + 1}"])[j]
        out[t] = (int((h == y).sum()), int((y == maj).sum()), int(len(isel)))
    return out


def gates(u, ref):
    """PROTOCOL section 4 G1-G3 for both tasks in exact integer arithmetic (accuracies share the same N rows)."""
    res, fl = {}, {}
    for t in (0, 1):
        ca, cc, n = u[t]
        cr, ccr, nr = ref[t]
        assert cc == ccr and n == nr
        res[t] = (100 * (ca - cr) >= -n, 5 * (ca - cc) >= 4 * (cr - cc), 100 * (ca - cc) >= 3 * n)
        a, c, ar = ca / n, cc / n, cr / n
        fl[t] = (a >= ar - 0.01, a - c >= 0.8 * (ar - c), a - c >= 0.03)
    passed = all(all(v) for v in res.values())
    borderline = any(res[t] != fl[t] for t in (0, 1))
    return passed, res, borderline


def g3(u):
    return all(100 * (u[t][0] - u[t][1]) >= 3 * u[t][2] for t in (0, 1))


def inner_aucs(rec):
    r = rec.get("recovery", rec)
    a = r.get("auc", r)
    return {w: num(a[w]) for w in ("v1", "v2", "pair")}


def inner_utility_record(rec):
    u = rec.get("utility") or {}
    out = {}
    for t in (0, 1):
        x = u.get(str(t), u.get(t))
        if x is None:
            return None
        out[t] = (float(x["acc"]), float(x["const_acc"]))
    return out


class Cand:
    def __init__(self, unit, arm, beta, epoch, auc, util, alias=False, label=None):
        self.unit, self.arm, self.beta, self.epoch, self.auc, self.util, self.alias = unit, arm, beta, epoch, auc, util, alias
        self.label = label or arm

    def worse(self):
        return max(self.auc["v1"], self.auc["v2"])

    def mean(self):
        return (self.auc["v1"] + self.auc["v2"]) / 2

    def info(self):
        return {"unit": self.unit, "beta": self.beta, "epoch": self.epoch, "auc": self.auc, "alias": self.alias}


def select_stage_B(cands, ub):
    """PROTOCOL section 4 step 1. cands: protected candidates; ub: the U-B alias candidate (also the gate reference)."""
    considered = []
    if not g3(ub.util):
        return {"status": "NO_VALID_REFERENCE", "unit": None, "considered": [], "reason": "U-B fails G3"}
    feas = []
    for c in cands + [ub]:
        ok, det, border = gates(c.util, ub.util)
        considered.append({**c.info(), "gates_pass": ok, "borderline_gate": border,
                           "reject": None if ok else "utility gates"})
        if ok:
            feas.append(c)
    best = min(feas, key=lambda c: (c.worse(), c.mean(), c.beta, c.epoch))
    return {"status": "TASK_ONLY_ALIAS" if best.alias else "NOMINEE", "unit": best.unit, "beta": best.beta,
            "epoch": best.epoch, "auc": best.auc, "considered": considered}


def guard_ok(c, bound):
    return c.auc["v1"] <= bound[0] and c.auc["v2"] <= bound[1]


def select_trained(cands, uref, lr_auc):
    considered, feas = [], []
    for c in cands:
        ok, _, border = gates(c.util, uref)
        gd = guard_ok(c, lr_auc)
        reject = None if (ok and gd) else ("utility gates" if not ok else "local guard vs L-R")
        considered.append({**c.info(), "gates_pass": ok, "guard_pass": gd, "borderline_gate": border, "reject": reject})
        if ok and gd:
            feas.append(c)
    if not feas:
        return {"status": "NO_FEASIBLE_NOMINEE", "unit": None, "considered": considered}
    b = min(feas, key=lambda c: (c.auc["pair"], c.beta, c.epoch))
    return {"status": "NOMINEE", "unit": b.unit, "beta": b.beta, "epoch": b.epoch, "auc": b.auc, "considered": considered}


def select_stage_C(trained, singles, uref, lr_auc, jg_cands):
    """PROTOCOL section 4 steps 4-6. trained: {arm: [Cand]}; singles: {label: Cand or None}."""
    out = {"controls": {}, "single": {}}
    for arm in TRAINED_CONTROLS:
        out["controls"][arm] = select_trained(trained.get(arm, []), uref, lr_auc)
    feasible = {}
    for arm in TRAINED_CONTROLS:
        s = out["controls"][arm]
        if s["status"] == "NOMINEE":
            feasible[arm] = (s["auc"], s["unit"])
    for lab in SINGLE_CONTROLS:
        c = singles.get(lab)
        if c is None:
            out["single"][lab] = {"present": False, "feasible": False}
            continue
        ok, _, border = gates(c.util, uref)
        gd = guard_ok(c, lr_auc)
        out["single"][lab] = {"present": True, "unit": c.unit, "auc": c.auc, "gates_pass": ok, "guard_pass": gd,
                              "borderline_gate": border, "feasible": bool(ok and gd)}
        if ok and gd:
            feasible[lab] = (c.auc, c.unit)
    cst = [(feasible[lab][0]["pair"], CSTAR_ORDER.index(lab), lab) for lab in CSTAR_ORDER if lab in feasible]
    if cst:
        _, _, lab = min(cst)
        out["C*"] = {"label": lab, "unit": feasible[lab][1], "auc": feasible[lab][0]}
    else:
        out["C*"] = None
    b1, b2 = lr_auc
    lg = out["controls"]["L-G"]
    if lg["status"] == "NOMINEE":
        b1, b2 = min(b1, lg["auc"]["v1"]), min(b2, lg["auc"]["v2"])
    if out["C*"] is not None:
        b1, b2 = min(b1, out["C*"]["auc"]["v1"]), min(b2, out["C*"]["auc"]["v2"])
    considered, feas = [], []
    for c in jg_cands:
        ok, _, border = gates(c.util, uref)
        gd = guard_ok(c, (b1, b2))
        considered.append({**c.info(), "gates_pass": ok, "guard_pass": gd, "borderline_gate": border,
                           "reject": None if (ok and gd) else ("utility gates" if not ok else "min-guard")})
        if ok and gd:
            feas.append(c)
    if feas:
        b = min(feas, key=lambda c: (c.auc["pair"], c.beta, c.epoch))
        out["J-G"] = {"status": "NOMINEE", "unit": b.unit, "beta": b.beta, "epoch": b.epoch, "auc": b.auc,
                      "bound": [b1, b2], "considered": considered}
    else:
        out["J-G"] = {"status": "NO_FEASIBLE_NOMINEE", "unit": None, "bound": [b1, b2], "considered": considered}
    return out


class SelectionInputs:
    """Loads inner records + own utilities for the candidates of one seed; missing pieces are listed."""

    def __init__(self, ctx: Ctx, D: Data):
        self.ctx, self.D = ctx, D
        self.missing = []
        self.util_mismatch = []
        self.cache = {}

    def cand(self, unit, arm, beta, epoch, alias=False, label=None):
        if unit in self.cache:
            c = copy.copy(self.cache[unit])
            c.arm, c.beta, c.epoch, c.alias, c.label = arm, beta, epoch, alias, label or arm
            return c
        rec = self.ctx.rec(f"inner__{unit}") if self.ctx.complete(f"inner__{unit}") else None
        if rec is None or not self.ctx.complete(unit):
            self.missing.append(unit)
            return None
        util = utility_counts(self.ctx, unit, self.D)
        if util is None:
            self.missing.append(unit + " (release.npz)")
            return None
        ru = inner_utility_record(rec)
        if ru is not None:
            for t in (0, 1):
                if abs(ru[t][0] - util[t][0] / util[t][2]) > 1e-12 or abs(ru[t][1] - util[t][1] / util[t][2]) > 1e-12:
                    self.util_mismatch.append({"unit": unit, "task": t, "record": ru[t],
                                               "mine": (util[t][0] / util[t][2], util[t][1] / util[t][2])})
        c = Cand(unit, arm, beta, epoch, inner_aucs(rec), util, alias, label)
        self.cache[unit] = c
        return c


def fare_units(ctx: Ctx, k):
    """Per-purpose FARE configuration units of seed k: {p: {cfg: unit}} (zero-fairness twins under 'Z...')."""
    out = {0: {}, 1: {}}
    for n in ctx.unit_names(f"fare__s{k}__p"):
        m = re.match(rf"^fare__s{k}__p([01])__([A-Za-z]+\d+)$", n)
        if m and ctx.complete(n):
            out[int(m.group(1))][m.group(2)] = n
    return out


def fare_pair_inner(ctx: Ctx, u0, u1):
    """The inner record of a FARE purpose pair (any inner__* record naming both purpose units and carrying a pair AUC)."""
    for n in ctx.unit_names("inner__"):
        if "fare" not in n.lower():
            continue
        rec = ctx.rec(n)
        if rec is None:
            continue
        txt = json.dumps(rec.get("of", rec.get("units", ""))) + n
        if u0 in txt and u1 in txt:
            try:
                inner_aucs(rec)
                return n, rec
            except Exception:  # noqa: BLE001
                continue
    return None, None


def fare_select(ctx: Ctx, k, uref, D):
    """Predecessor's published per-purpose FARE rule (pcrl_joint_complete_view_method_v1/PROTOCOL.md section 5.4),
    reselected on the new roles: per purpose, the admissible configuration (task-p gates vs U) with the lowest inner
    local AUC wins, ties to the lower id; F0 is the zero-fairness twin at F's configuration."""
    units = fare_units(ctx, k)
    if not units[0] or not units[1]:
        return None, "no per-purpose FARE units"
    chosen, table = {}, {}
    for p in (0, 1):
        cands = []
        for cfg, n in sorted(units[p].items()):
            if cfg.startswith("Z"):
                continue
            rec = ctx.rec(f"inner__{n}")
            if rec is None:
                return None, f"inner__{n} missing"
            loc = rec.get("recovery_local")
            if loc is None:
                try:
                    loc = inner_aucs(rec)[f"v{p + 1}"]
                except Exception:  # noqa: BLE001
                    return None, f"inner__{n}: no local AUC field"
            u = rec.get("utility", {})
            u = u.get(str(p), u.get(p, u)) if isinstance(u, dict) else {}
            if "acc" not in u:
                return None, f"inner__{n}: no utility field"
            N = uref[p][2]
            ca, cc = int(round(float(u["acc"]) * N)), int(round(float(u["const_acc"]) * N))
            cr, ccr, _ = uref[p]
            adm = (cc == ccr and 100 * (ca - cr) >= -N and 5 * (ca - cc) >= 4 * (cr - cc) and 100 * (ca - cc) >= 3 * N)
            idn = int(re.sub(r"\D", "", cfg) or 0)
            cands.append((float(loc), idn, cfg, n, adm))
        table[p] = [{"cfg": c[2], "unit": c[3], "local_auc": c[0], "admissible": c[4]} for c in cands]
        adm = [c for c in cands if c[4]]
        chosen[p] = min(adm) if adm else None
    per = {p: (chosen[p][3] if chosen[p] else None) for p in (0, 1)}
    twins = {p: (units[p].get(f"Z{chosen[p][1]}") if chosen[p] else None) for p in (0, 1)}
    if chosen[0] is None or chosen[1] is None:
        return {"table": table, "per_purpose": per, "twins": twins, "F": None, "F0": None}, None
    F_units = [per[0], per[1]]
    z_units = [twins[0], twins[1]]
    return {"table": table, "per_purpose": per, "twins": twins, "F": F_units,
            "F0": z_units if all(z_units) else None}, None


def replay_selection(ctx: Ctx):
    """Own implementation of PROTOCOL section 4 on the inner records. Returns (per-seed result, problems)."""
    D = ctx.data()
    SI = SelectionInputs(ctx, D)
    out = {}
    for k in SEEDS:
        res = {"seed": k}
        out[k] = res
        ub = SI.cand(f"tl__s{k}__e20", "U-B", 0.0, 20, alias=True)
        if ub is None:
            res["stage_B"] = "PENDING"
            continue
        stB = {}
        for arm in ("L-R", "L-O"):
            cands = [SI.cand(ck_name("B", k, arm, b, e), arm, b, e) for b in BETAS for e in EPOCHS]
            if any(c is None for c in cands):
                stB[arm] = {"status": "PENDING"}
                continue
            stB[arm] = select_stage_B(cands, ub)
        res["stage_B"] = stB
        lr = stB["L-R"]
        if lr.get("status") in ("PENDING", None):
            continue
        if lr["status"] == "NO_VALID_REFERENCE":
            res["stage_C"] = "SKIPPED (NO_VALID_REFERENCE)"
            continue
        e_lr = 20 if lr["status"] == "TASK_ONLY_ALIAS" else lr["epoch"]
        res["e_LR"] = e_lr
        u = SI.cand(f"tl__s{k}__e{20 + e_lr}", "U", 0.0, 20 + e_lr)
        if u is None:
            res["stage_C"] = "PENDING"
            continue
        res["U_unit"] = u.unit
        lr_c = SI.cand(lr["unit"], "L-R", lr.get("beta", 0.0), lr.get("epoch", 20), alias=lr["status"] == "TASK_ONLY_ALIAS")
        lr_auc = (lr_c.auc["v1"], lr_c.auc["v2"])
        trained = {}
        pend = False
        for arm in TRAINED_CONTROLS + ("J-G",):
            cs = [SI.cand(ck_name("C", k, arm, b, e), arm, b, e) for b in BETAS for e in EPOCHS]
            if any(c is None for c in cs):
                pend = True
            trained[arm] = cs
        if pend:
            res["stage_C"] = "PENDING"
            continue
        singles = {"U": u, "L-R": lr_c}
        lo = stB["L-O"]
        if lo.get("status") in ("NOMINEE", "TASK_ONLY_ALIAS"):
            singles["L-O"] = SI.cand(lo["unit"], "L-O", lo.get("beta", 0.0), lo.get("epoch", 20))
        singles["E"] = SI.cand(f"lc__s{k}__E", "E", 0.0, 0)
        fs, why = fare_select(ctx, k, u.util, D)
        res["fare"] = fs if fs is not None else {"pending": why}
        for lab in ("F", "F0"):
            singles[lab] = None
            if fs and fs.get(lab):
                n_in, rec = fare_pair_inner(ctx, *fs[lab])
                if rec is not None:
                    ru = inner_utility_record(rec)
                    N = u.util[0][2]
                    util = {t: (int(round(ru[t][0] * N)), int(round(ru[t][1] * N)), N) for t in (0, 1)}
                    singles[lab] = Cand("+".join(fs[lab]), lab, 0.0, 0, inner_aucs(rec), util, label=lab)
                    singles[lab].inner = n_in
        if singles.get("E") is None or (fs is None):
            res["stage_C_note"] = "E or FARE inner records absent; C* computed over the present controls only"
        stC = select_stage_C({a: trained[a] for a in TRAINED_CONTROLS}, singles, u.util, lr_auc, trained["J-G"])
        res["stage_C"] = stC
        res["fixed"] = {f"{a}@fx": ck_name("C", k, a, FIXED_BETA, FIXED_EPOCH) for a in ("J-G", "J-R", "J-O")}
    return out, {"missing_inputs": SI.missing, "utility_record_mismatches": SI.util_mismatch}


def _seed_dict(src, k):
    if not isinstance(src, dict):
        return None
    for key in (str(k), k, f"s{k}"):
        if key in src:
            return src[key]
    if "seeds" in src:
        return _seed_dict(src["seeds"], k)
    return None


def _find_arm(d, arm):
    if not isinstance(d, dict):
        return None
    for path in ((arm,), ("arms", arm), ("controls", arm), ("trained", arm), ("single", arm), ("singles", arm),
                 ("trained_controls", arm), ("single_controls", arm), ("nominees", arm), ("selected", arm)):
        x = d
        for p in path:
            x = x.get(p) if isinstance(x, dict) else None
        if isinstance(x, dict):
            return x
        if isinstance(x, str):
            return {"unit": x}
    return None


def _find_cstar(d):
    if not isinstance(d, dict):
        return None
    for key in ("C*", "Cstar", "C_star", "c_star", "cstar", "comparator"):
        if key in d:
            x = d[key]
            if x is None or isinstance(x, (str, dict)):
                return {"label": x} if isinstance(x, str) else x
    return None


def _lab_of(x):
    if x is None:
        return None
    if isinstance(x, str):
        return x
    for key in ("label", "arm", "C*", "name"):
        if isinstance(x.get(key), str):
            return x[key]
    return None


def check_selection(ctx: Ctx, rep: Report, lock, mysel, problems):
    sec = "3-selection"
    selB, selC = ctx.sel("B"), ctx.sel("C")
    if not any(isinstance(mysel[k].get("stage_B"), dict) for k in SEEDS):
        rep.add("3.1-stage-B-replay", sec, "PENDING", "task line / Stage B inner records not all present yet",
                missing_inputs=problems["missing_inputs"][:30], n_missing=len(problems["missing_inputs"]))
    else:
        rows, bad = [], []
        for k in SEEDS:
            stB = mysel[k].get("stage_B")
            if not isinstance(stB, dict):
                continue
            for arm in ("L-R", "L-O"):
                m = stB.get(arm, {})
                r = _find_arm(_seed_dict(selB, k), arm) if selB else None
                row = {"seed": k, "arm": arm, "mine_status": m.get("status"), "mine_unit": m.get("unit"),
                       "runner_status": (r or {}).get("status"), "runner_unit": (r or {}).get("unit")}
                if r is not None:
                    row["match"] = (row["mine_status"] == row["runner_status"] and row["mine_unit"] == row["runner_unit"])
                    if not row["match"]:
                        bad.append(row)
                rows.append(row)
        st = "PENDING" if selB is None else ("FAIL" if bad else "PASS")
        rep.add("3.1-stage-B-replay", sec, st,
                "L-R and L-O frozen by the PROTOCOL section 4 step 1 rule (gates vs U-B in exact integers; min worse "
                "local AUC, mean local AUC, beta, checkpoint; U-B alias as beta = 0 candidate -> TASK_ONLY_ALIAS; "
                "U-B failing G3 -> NO_VALID_REFERENCE) and compared with selection_B.json",
                rows=rows, mismatches=bad, selection_B_present=selB is not None,
                considered={f"s{k}/{a}": mysel[k]["stage_B"][a].get("considered")
                            for k in SEEDS if isinstance(mysel[k].get("stage_B"), dict) for a in ("L-R", "L-O")})
    rep.add("3.2-inner-utility-records", sec,
            "PENDING" if not any(isinstance(mysel[k].get("stage_B"), dict) for k in SEEDS)
            else ("FAIL" if problems["utility_record_mismatches"] else "PASS"),
            "inner-record utilities (accuracy, DEFENSE_FIT-majority constant on INNER_SELECTION) equal the values "
            "recomputed from release.npz hard decisions and the source labels",
            mismatches=problems["utility_record_mismatches"][:20])
    have_C = [k for k in SEEDS if isinstance(mysel[k].get("stage_C"), dict)]
    if not have_C:
        rep.add("3.3-stage-C-replay", sec, "PENDING", "Stage C inner records / baselines not all present yet",
                seeds_state={k: (mysel[k].get("stage_C") if not isinstance(mysel[k].get("stage_C"), dict) else "done")
                             for k in SEEDS})
    else:
        rows, bad = [], []
        for k in have_C:
            stC = mysel[k]["stage_C"]
            dC = _seed_dict(selC, k) if selC else None
            for arm in TRAINED_CONTROLS + ("J-G",):
                m = stC["controls"][arm] if arm in TRAINED_CONTROLS else stC["J-G"]
                r = _find_arm(dC, arm)
                row = {"seed": k, "arm": arm, "mine_status": m["status"], "mine_unit": m.get("unit"),
                       "runner_status": (r or {}).get("status"), "runner_unit": (r or {}).get("unit")}
                if r is not None:
                    ok = row["mine_status"] == row["runner_status"]
                    if m["status"] == "NOMINEE":
                        ok &= row["mine_unit"] == row["runner_unit"]
                    row["match"] = bool(ok)
                    if not ok:
                        bad.append(row)
                rows.append(row)
            for lab in SINGLE_CONTROLS:
                m = stC["single"].get(lab, {})
                r = _find_arm(dC, lab)
                row = {"seed": k, "arm": lab, "mine_feasible": m.get("feasible"), "mine_unit": m.get("unit"),
                       "runner_feasible": None if r is None else r.get("feasible", r.get("status"))}
                if r is not None and "feasible" in r:
                    row["match"] = bool(r["feasible"]) == bool(m.get("feasible"))
                    if not row["match"]:
                        bad.append(row)
                rows.append(row)
            rc = _find_cstar(dC)
            mc = stC["C*"]
            row = {"seed": k, "arm": "C*", "mine_label": _lab_of(mc), "mine_unit": (mc or {}).get("unit"),
                   "runner_label": _lab_of(rc), "runner_unit": (rc or {}).get("unit") if isinstance(rc, dict) else None}
            if dC is not None:
                row["match"] = row["mine_label"] == row["runner_label"]
                if not row["match"]:
                    bad.append(row)
            rows.append(row)
        st = "PENDING" if selC is None else ("FAIL" if bad else "PASS")
        rep.add("3.3-stage-C-replay", sec, st,
                "trained controls (gates vs U + zero-buffer guard vs L-R; min coalition AUC, beta, checkpoint), "
                "single-configuration controls (U, E, F, F0, frozen L-R / L-O incl. task-only aliases, review R2), "
                "C* (lowest coalition AUC, ties in the order L-G, L-R, L-O, J-R, J-O, U, E, F, F0) and the J-G "
                "nomination (min-guard over L-R, L-G if NOMINEE, C* if present) compared with selection_C.json",
                rows=rows, mismatches=bad, selection_C_present=selC is not None,
                jg_considered={f"s{k}": mysel[k]["stage_C"]["J-G"].get("considered") for k in have_C},
                fare={f"s{k}": mysel[k].get("fare") for k in have_C})
    # lock
    if lock is None:
        rep.add("3.4-evaluation-lock-units", sec, "PENDING", "EVALUATION_LOCK.json not written yet")
    else:
        specs = lock_specs(lock)
        rows, bad = [], []
        for k in SEEDS:
            s = mysel[k]
            stC = s.get("stage_C")
            if not isinstance(stC, dict):
                continue
            exp = {}
            stB = s["stage_B"]
            exp["L-R"] = stB["L-R"].get("unit")
            exp["L-O"] = stB["L-O"].get("unit")
            exp["U"] = s.get("U_unit")
            exp["E"] = f"lc__s{k}__E"
            for arm in TRAINED_CONTROLS:
                exp[arm] = stC["controls"][arm].get("unit")
            exp["J-G"] = stC["J-G"].get("unit")
            exp.update(s["fixed"])
            if s.get("fare") and s["fare"].get("F"):
                exp["F"] = s["fare"]["F"]
                exp["F0"] = s["fare"].get("F0")
            fz = s.get("fare") or {}
            if fz.get("F") is None and fz.get("per_purpose"):
                lf = (specs.get((k, "F")) or {}).get("units") or [None, None]
                l0 = (specs.get((k, "F0")) or {}).get("units") or [None, None]
                for p in (0, 1):
                    mine_p = fz["per_purpose"].get(p, fz["per_purpose"].get(str(p)))
                    if mine_p is not None:
                        row = {"seed": k, "label": f"F purpose {p} (other purpose has no admissible config)",
                               "lock": lf[p], "replay": mine_p}
                        if lf[p] != mine_p:
                            bad.append(row)
                        rows.append(row)
                for p in (0, 1):
                    m = re.match(r"^fare__s\d+__p[01]__c(\d+)$", lf[p] or "")
                    if m:
                        want_z = re.sub(r"__c\d+$", f"__Z{m.group(1)}", lf[p])
                        row = {"seed": k, "label": f"F0 purpose {p} = zero-fairness twin of the locked F config",
                               "lock": l0[p], "replay": want_z}
                        if l0[p] != want_z:
                            bad.append(row)
                        rows.append(row)
            for lab in SCORED:
                spec = specs.get((k, lab))
                got = None if spec is None else (spec.get("unit") or spec.get("units"))
                want = exp.get(lab)
                row = {"seed": k, "label": lab, "lock": got, "replay": want}
                if want is None:
                    row["note"] = "replay has no nominee (descriptive unit, closest rule not replayed)"
                    if got is not None and lab in ("J-G",) + TRAINED_CONTROLS:
                        pc = parse_ck(got) if isinstance(got, str) else None
                        row["descriptive_unit_in_arm_grid"] = bool(pc and pc["seed"] == k and pc["arm"] == lab)
                        if not row["descriptive_unit_in_arm_grid"]:
                            bad.append(row)
                elif got != want:
                    bad.append(row)
                rows.append(row)
            seed_lock = lock.get("seeds", {}).get(str(k), {})
            cmp_ = seed_lock.get("comparator")
            cl = _lab_of(cmp_) if isinstance(cmp_, (dict, str)) else None
            mc = _lab_of(stC["C*"])
            row = {"seed": k, "label": "comparator(C*)", "lock": cl, "replay": mc}
            if cl != mc:
                bad.append(row)
            rows.append(row)
            stat = seed_lock.get("status") or {}
            if isinstance(stat, dict):
                for lab, mine in (("L-R", stB["L-R"]["status"]), ("L-O", stB["L-O"]["status"]),
                                  ("J-G", stC["J-G"]["status"]), *((a, stC["controls"][a]["status"]) for a in TRAINED_CONTROLS)):
                    if lab in stat:
                        g = stat[lab] if isinstance(stat[lab], str) else (stat[lab] or {}).get("status")
                        row = {"seed": k, "label": f"status({lab})", "lock": g, "replay": mine}
                        if g != mine:
                            bad.append(row)
                        rows.append(row)
        rep.add("3.4-evaluation-lock-units", sec, "FAIL" if bad else ("PASS" if rows else "PENDING"),
                "every scored label's frozen unit, the C* comparator and the statuses in EVALUATION_LOCK.json equal "
                "the independent replay", rows=rows, mismatches=bad)


def inner_slate():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return {
        "LR": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)),
        "MLP": lambda: make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(64, 64), solver="adam", alpha=1e-4, early_stopping=True, validation_fraction=0.1,
            max_iter=300, n_iter_no_change=15, random_state=0)),
        "HGB": lambda: HistGradientBoostingClassifier(learning_rate=0.1, max_leaf_nodes=31, max_iter=200,
                                                      early_stopping=False, random_state=0),
        "LR_unscaled(alt)": lambda: LogisticRegression(C=1.0, max_iter=5000),
    }


def spot_inner(ctx: Ctx, unit: str, D: Data):
    """Refit the inner slate on AUDIT_FIT and score INNER_SELECTION (fixed orientation, column 1 = P(SEX = 1))."""
    from sklearn.metrics import roc_auc_score
    rel = ctx.release(unit)
    pos = D.pos_of(rel["row_id"])
    where = {int(p): j for j, p in enumerate(pos)}
    fit = np.array([where[int(p)] for p in D.idx["AUDIT_FIT"]])
    val = np.array([where[int(p)] for p in D.idx["INNER_SELECTION"]])
    sex = D.label("sex")
    sf, sv = sex[D.idx["AUDIT_FIT"]], sex[D.idx["INNER_SELECTION"]]
    v1 = np.hstack([rel["r1"], rel["c1"]])
    v2 = np.hstack([rel["r2"], rel["c2"]])
    views = {"v1": v1, "v2": v2, "pair": np.hstack([v1, v2])}
    table = {}
    for w, V in views.items():
        for name, mk in inner_slate().items():
            m = mk()
            m.fit(V[fit], sf)
            table[(w, name)] = float(roc_auc_score(sv, m.predict_proba(V[val])[:, 1]))
    main = [n for n in inner_slate() if "alt" not in n]
    loc = {w: max(table[(w, n)] for n in main) for w in ("v1", "v2")}
    pair = max([table[("pair", n)] for n in main] + [loc["v1"], loc["v2"]])   # + ignore-recipient candidates
    alt = {w: max(table[(w, n)] for n in inner_slate()) for w in ("v1", "v2")}
    alt["pair"] = max([table[("pair", n)] for n in inner_slate()] + [alt["v1"], alt["v2"]])
    return {"v1": loc["v1"], "v2": loc["v2"], "pair": pair}, {f"{w}/{n}": v for (w, n), v in table.items()}, alt


def check_spot(ctx: Ctx, rep: Report, mysel, lock):
    sec = "3-selection"
    if ctx.spot <= 0:
        rep.add("3.5-inner-spot-check", sec, "SKIP", "disabled (--spot 0)")
        return
    D = ctx.data()
    pick = []
    for k in SEEDS:
        stC = mysel.get(k, {}).get("stage_C")
        if isinstance(stC, dict) and stC["J-G"].get("unit"):
            pick.append(stC["J-G"]["unit"])
        stB = mysel.get(k, {}).get("stage_B")
        if isinstance(stB, dict) and stB["L-R"].get("unit"):
            pick.append(stB["L-R"]["unit"])
    pool = [n[len("inner__"):] for n in ctx.unit_names("inner__ck__") if ctx.complete(n)]
    pool = sorted(pool, key=lambda n: hashlib.sha256(n.encode()).hexdigest())
    pick = list(dict.fromkeys(pick + pool))
    pick = [n for n in pick if ctx.complete(n) and ctx.complete(f"inner__{n}") and ctx.release(n) is not None][:ctx.spot]
    if not pick:
        rep.add("3.5-inner-spot-check", sec, "PENDING", "no inner-audited checkpoint exists yet")
        return
    rows, worst = [], 0.0
    names = {"LR": "LR_C1", "MLP": "MLP_64x64", "HGB": "HGB_0.1_31"}
    for n in pick:
        t0 = time.time()
        mine, table, alt = spot_inner(ctx, n, D)
        full = ctx.rec(f"inner__{n}")
        rec = inner_aucs(full)
        d = {w: abs(mine[w] - rec[w]) for w in mine}
        worst = max(worst, max(d.values()))
        per = {}
        tabs = (full.get("recovery") or {}).get("tables") or {}
        for w in ("v1", "v2", "pair"):
            for mine_name, rec_name in names.items():
                hit = [t.get("inner_auc") for t in tabs.get(w, []) if t.get("attacker") == rec_name]
                if len(hit) == 1:
                    per[f"{w}/{rec_name}"] = abs(table[f"{w}/{mine_name}"] - float(hit[0]))
        if per:
            worst = max(worst, max(per.values()))
        rows.append({"unit": n, "refit": mine, "record": rec, "abs_diff": d, "per_attacker_abs_diff": per,
                     "slate_table": table, "max_incl_unscaled_LR_alt": alt, "wall_s": time.time() - t0})
    st = "PASS" if worst <= 1e-6 else ("WARN" if worst <= 2e-3 else "FAIL")
    rep.add("3.5-inner-spot-check", sec, st,
            "inner slate refitted on AUDIT_FIT (StandardScaler+LR C=1; StandardScaler+MLP(64,64) adam alpha 1e-4, "
            "early stopping 10%, max_iter 300, n_iter_no_change 15, rs 0; HGB lr 0.1, 31 leaves, 200 iters, no early "
            "stopping, rs 0), AUC on INNER_SELECTION, AUC-selected maximum; coalition bank includes the "
            "ignore-recipient candidates; compared with recovery.auc of the inner record",
            units=rows, max_abs_diff=worst)


# ================================================================================================ 4. endpoints
def wauc_matrix(score, pos, W):
    """Weighted ROC AUC (ties count 1/2) of `score` for the indicator `pos` under each row of weights W (R x n)."""
    score = np.asarray(score, dtype=np.float64)
    pos = np.asarray(pos, dtype=bool)
    order = np.argsort(score, kind="mergesort")
    s, p = score[order], pos[order]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    Wo = W[:, order]
    Wp = np.where(p, Wo, 0.0)
    Wn = np.where(p, 0.0, Wo)
    pb = np.add.reduceat(Wp, starts, axis=1)
    nb = np.add.reduceat(Wn, starts, axis=1)
    below = np.cumsum(nb, axis=1) - nb
    num_ = (pb * (below + 0.5 * nb)).sum(1)
    den = pb.sum(1) * nb.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num_ / np.where(den > 0, den, 1.0), np.nan)


def boot_counts(n_groups, B, seed):
    """B multinomial count vectors over groups (equal probabilities), drawn sequentially in chunks of 250."""
    rng = np.random.default_rng(seed)
    p = np.full(n_groups, 1.0 / n_groups)
    out = np.empty((B, n_groups), dtype=np.int64)
    done = 0
    while done < B:
        m = min(250, B - done)
        out[done:done + m] = rng.multinomial(n_groups, p, size=m)
        done += m
    return out


def boot_weights(assess_unit, B, seed):
    groups, inv = np.unique(np.asarray(assess_unit), return_inverse=True)
    C = boot_counts(len(groups), B, seed)
    return np.vstack([np.ones(len(inv)), C[:, inv].astype(np.float64)]), len(groups)


VIEW_KEYS = {("auc", "pair"): "P_auc_pair", ("auc", "v1"): "P_auc_v1", ("auc", "v2"): "P_auc_v2",
             ("prob", "pair"): "P_auc_ppair", ("prob", "v1"): "P_auc_p1", ("prob", "v2"): "P_auc_p2",
             ("hard", "pair"): "P_auc_hpair", ("hard", "v1"): "P_auc_h1", ("hard", "v2"): "P_auc_h2"}


class Endpoints:
    """Per-seed statistics as (B+1)-vectors (index 0 = point estimate on the observed rows)."""

    def __init__(self, preds, W, maj, prior):
        self.preds, self.W, self.maj, self.prior = preds, W, maj, np.asarray(prior, float)
        self.wsum = W.sum(1)
        self.cache = {}
        any_p = next(iter(preds.values()))
        self.sex = np.asarray(any_p["sex"]).astype(np.int64)
        self.y = {0: np.asarray(any_p["y_income"]).astype(np.int64), 1: np.asarray(any_p["y_occ"]).astype(np.int64)}
        self.const = {t: (W @ (self.y[t] == maj[t]).astype(float)) / self.wsum for t in (0, 1)}

    def has(self, k, lab, key=None):
        p = self.preds.get((k, lab))
        return p is not None and (key is None or key in p)

    def R(self, k, lab, fmt, view):
        key = VIEW_KEYS[(fmt, view)]
        ck = ("R", k, lab, key)
        if ck not in self.cache:
            P = np.asarray(self.preds[(k, lab)][key])
            assert P.ndim == 3 and P.shape[2] == 2, (key, P.shape)
            self.cache[ck] = np.mean([wauc_matrix(P[a][:, 1], self.sex == 1, self.W) for a in range(P.shape[0])], 0)
            self.cache[("nseeds", k, lab, key)] = P.shape[0]
        return self.cache[ck]

    def acc(self, k, lab, t):
        ck = ("acc", k, lab, t)
        if ck not in self.cache:
            h = np.asarray(self.preds[(k, lab)][f"hard{t + 1}"]).astype(np.int64)
            self.cache[ck] = (self.W @ (h == self.y[t]).astype(float)) / self.wsum
        return self.cache[ck]

    def llr(self, k, lab, view):
        ck = ("llr", k, lab, view)
        if ck not in self.cache:
            P = np.asarray(self.preds[(k, lab)][f"P_ce_{view}"])
            ce0 = (self.W @ (-np.log(self.prior[self.sex]))) / self.wsum
            ces = []
            for a in range(P.shape[0]):
                pr = P[a][np.arange(len(self.sex)), self.sex]
                ces.append((self.W @ (-np.log(pr))) / self.wsum)
            self.cache[ck] = 1.0 - np.mean(ces, 0) / ce0
        return self.cache[ck]

    def rrace(self, k, lab, view):
        ck = ("race", k, lab, view)
        if ck not in self.cache:
            p = self.preds[(k, lab)]
            P = np.asarray(p[f"Prace_auc_{view}"])
            ry = np.asarray(p["race_y"]).astype(np.int64)
            Wr = self.W[:, np.asarray(p["race_pos"]).astype(np.int64)]
            K = P.shape[2]
            self.cache[ck] = np.mean([np.mean([wauc_matrix(P[a][:, c], ry == c, Wr) for c in range(K)], 0)
                                      for a in range(P.shape[0])], 0)
        return self.cache[ck]


def summarize(arr, z, target, side):
    point = float(arr[0])
    reps = np.asarray(arr[1:], float)
    if not np.isfinite(reps).all() or not math.isfinite(point):
        return {"point": point, "se": float("nan"), "lower": float("nan"), "upper": float("nan"),
                "decision": "NOT_ESTIMABLE"}
    se = float(np.std(reps, ddof=1))
    lo, hi = point - z * se, point + z * se
    if side == "lower>":
        d = "PASS" if lo > target else "FAIL"
    elif side == "upper<":
        d = "PASS" if hi < target else "FAIL"
    else:
        d = "ABOVE" if lo > target else ("BELOW" if hi < target else "NOT_RESOLVED")
    return {"point": point, "se": se, "lower": lo, "upper": hi, "decision": d}


def primary_stat(E: Endpoints, slot, k, comparator):
    jg = "J-G"
    kind = slot["kind"]
    if kind in ("coalition", "local"):
        ref = "L-G" if slot["claim"] == "A" else comparator
        if ref is None or not E.has(k, ref) or not E.has(k, jg):
            return None
        if kind == "coalition":
            return E.R(k, ref, "auc", "pair") - E.R(k, jg, "auc", "pair")
        return E.R(k, jg, "auc", slot["view"]) - E.R(k, ref, "auc", slot["view"])
    if not E.has(k, jg) or not E.has(k, "U"):
        return None
    t = slot["task"]
    if kind == "acc":
        return E.acc(k, jg, t) - E.acc(k, "U", t)
    if kind == "retain":
        return E.acc(k, jg, t) - 0.8 * E.acc(k, "U", t) - 0.2 * E.const[t]
    if kind == "useful":
        return E.acc(k, jg, t) - E.const[t]
    raise ValueError(kind)


def secondary_stat(E: Endpoints, slot, k):
    kind = slot["kind"]
    if kind == "out":
        a, b = slot["a"], slot["b"]
        key = VIEW_KEYS[(slot["fmt"], slot["view"])]
        if not (E.has(k, a, key) and E.has(k, b, key)):
            return None
        return E.R(k, a, slot["fmt"], slot["view"]) - E.R(k, b, slot["fmt"], slot["view"])
    if kind == "race":
        a, b = slot["a"], slot["b"]
        key = f"Prace_auc_{slot['view']}"
        if not (E.has(k, a, key) and E.has(k, b, key)):
            return None
        return E.rrace(k, a, slot["view"]) - E.rrace(k, b, slot["view"])
    if kind == "logloss":
        a, b = slot["a"], slot["b"]
        key = f"P_ce_{slot['view']}"
        if not (E.has(k, a, key) and E.has(k, b, key)):
            return None
        return E.llr(k, a, slot["view"]) - E.llr(k, b, slot["view"])
    if kind in ("fixed", "rec"):
        a, b = slot["a"], slot["b"]
        view = slot.get("view", "pair")
        if not (E.has(k, a) and E.has(k, b)):
            return None
        return E.R(k, a, "auc", view) - E.R(k, b, "auc", view)
    if kind == "synergy":
        a = slot["arm"]
        if not E.has(k, a):
            return None
        return E.R(k, a, "auc", "pair") - np.maximum(E.R(k, a, "auc", "v1"), E.R(k, a, "auc", "v2"))
    raise ValueError(kind)


def family_values(E: Endpoints, fam, comparators):
    prim, sec_ = {}, {}
    for slot in fam["primary"]:
        per = {k: primary_stat(E, slot, k, comparators.get(k)) for k in SEEDS}
        have = [k for k in SEEDS if per[k] is not None]
        if not have:
            prim[slot["id"]] = {"decision": "NOT_ESTIMABLE", "seeds": []}
            continue
        arr = np.mean([per[k] for k in have], 0)
        prim[slot["id"]] = {**summarize(arr, Z_PRIMARY, slot["target"], slot["side"]), "seeds": have,
                            "per_seed_point": {k: float(per[k][0]) for k in have},
                            "target": slot["target"], "side": slot["side"], "alias_of": slot.get("alias_of")}
    for slot in fam["secondary"]:
        per = {k: secondary_stat(E, slot, k) for k in SEEDS}
        have = [k for k in SEEDS if per[k] is not None]
        if not have:
            sec_[slot["id"]] = {"decision": "NOT_ESTIMABLE", "seeds": []}
            continue
        arr = np.mean([per[k] for k in have], 0)
        sec_[slot["id"]] = {**summarize(arr, Z_SECONDARY, slot["target"], slot["side"]), "seeds": have,
                            "per_seed_point": {k: float(per[k][0]) for k in have},
                            "target": slot["target"], "side": slot["side"]}
    return prim, sec_


def norm_decision(x):
    if x is None:
        return None
    s = str(x).strip().upper()
    if s in ("TRUE", "PASS", "PASSED", "1", "YES"):
        return "PASS"
    if s in ("FALSE", "FAIL", "FAILED", "0", "NO"):
        return "FAIL"
    return s


def read_endpoint_csv(p):
    if not Path(p).exists():
        return None
    with open(p, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    cols = {c.lower(): c for c in rows[0]}

    def col(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None
    cid = col("id", "slot", "endpoint", "endpoint_id")
    cp, cs = col("point", "estimate", "mean", "value"), col("se", "boot_se", "sd")
    cl, cu = col("lower", "lo", "ci_lower", "lower_bound"), col("upper", "hi", "ci_upper", "upper_bound")
    cd = col("decision", "result", "pass", "status", "outcome")
    out = {}
    for r in rows:
        if cid is None:
            break

        def f(c):
            try:
                return float(r[c]) if c and r.get(c) not in (None, "") else None
            except ValueError:
                return None
        out[r[cid]] = {"point": f(cp), "se": f(cs), "lower": f(cl), "upper": f(cu),
                       "decision": norm_decision(r.get(cd)) if cd else None}
    return out


def runner_decision(d, side):
    """Runner vocabulary -> replay vocabulary: a one-sided clause that does not pass is written NOT_ESTABLISHED."""
    d = norm_decision(d)
    if side in ("lower>", "upper<") and d == "NOT_ESTABLISHED":
        return "FAIL"
    return d


def read_levels_csv(p):
    """RAW_LEVELS.csv -> {inference-level key: (point, se)}; seed-mean rows use the key without a seed."""
    if not Path(p).exists():
        return None
    out = {}
    for r in csv.DictReader(open(p, newline="")):
        q, seed, lab, det = r["quantity"], r["seed"], r["label"], r["detail"]
        if q in ("Rmean", "accmean"):
            key = f"{q}|{lab}|{det}"
        elif q == "const":
            key = f"const|{lab}"
        elif q == "R":
            key = f"R|{seed}|{lab}|{det}"
        else:
            key = f"{q}|{seed}|{lab}|{det}"
        out[key] = (float(r["point"]), float(r["se"]))
    return out


def compare_family(mine, theirs, tol=1e-9, abs_tol=0.0):
    """tol: relative tolerance (full-precision sources); abs_tol: absolute tolerance (rounded CSV, 6 decimals)."""
    rows, bad = [], []
    for sid, m in mine.items():
        t = (theirs or {}).get(sid)
        row = {"id": sid, "mine": {k: m.get(k) for k in ("point", "se", "lower", "upper", "decision")}, "runner": t}
        if t is None:
            row["match"] = None
        else:
            ok = True
            for key in ("point", "se", "lower", "upper"):
                a, b = m.get(key), t.get(key)
                if b is None or a is None:
                    continue
                if not (abs(a - b) <= max(tol * max(1.0, abs(a)), abs_tol)):
                    ok = False
                    row.setdefault("diffs", {})[key] = a - b
            if t.get("decision") is not None and m.get("decision") != runner_decision(t.get("decision"), m.get("side")):
                ok = False
            row["match"] = ok
            if not ok:
                bad.append(row)
        rows.append(row)
    return rows, bad


def load_preds(ctx: Ctx, lock):
    """{(seed, label): dict of arrays} for every scored label with a complete outer unit."""
    preds = {}
    for k in SEEDS:
        for lab in SCORED:
            n = f"outer__s{k}__{safe_label(lab)}"
            p = ctx.udir(n) / "preds.npz"
            if ctx.complete(n) and p.exists():
                z = np.load(p, allow_pickle=False)
                preds[(k, lab)] = {key: z[key] for key in z.files}
    return preds


def comparators_from(lock, mysel):
    out, src = {}, {}
    for k in SEEDS:
        mine = None
        stC = mysel.get(k, {}).get("stage_C")
        if isinstance(stC, dict) and stC.get("C*"):
            mine = stC["C*"]["label"]
        lk = None
        if lock:
            c = lock.get("seeds", {}).get(str(k), {}).get("comparator")
            lk = _lab_of(c) if isinstance(c, (dict, str)) else None
        out[k] = mine if mine is not None else lk
        src[k] = {"replay": mine, "lock": lk}
    return out, src


def check_endpoints(ctx: Ctx, rep: Report, lock, mysel):
    sec = "4-endpoints"
    fam = jload(ctx.res / "PRIMARY_FAMILY.json")
    zp_ok = abs(fam["z_primary"] - Z_PRIMARY) < 1e-12 and abs(fam["z_secondary"] - Z_SECONDARY) < 1e-12
    ids_ok = ([s["id"] for s in fam["primary"]] == [f"P{i:02d}" for i in range(1, 19)]
              and len(fam["secondary"]) == 30 and fam["B"] == B_BOOT and fam["boot_seed"] == BOOT_SEED)
    al = {s["id"]: s.get("alias_of") for s in fam["primary"] if s.get("alias_of")}
    al_ok = al == {f"P{i}": f"P{i - 9:02d}" for i in range(13, 19)} and all(
        {k: v for k, v in s.items() if k not in ("id", "claim", "alias_of")}
        == {k: v for k, v in fam["primary"][int(s["alias_of"][1:]) - 1].items() if k not in ("id", "claim", "alias_of")}
        for s in fam["primary"] if s.get("alias_of"))
    rep.add("4.0-family-definition", sec, "PASS" if (zp_ok and ids_ok and al_ok) else "FAIL",
            "z values recomputed as Phi^-1(1 - 0.05/36) and Phi^-1(1 - 0.05/60); 18 primary slots P01-P18 with "
            "P13-P18 identical aliases of P04-P09; 30 secondary slots; B = 1999, bootstrap seed 20261004",
            z_primary=Z_PRIMARY, z_secondary=Z_SECONDARY)
    if not ctx.assessment_open():
        rep.add("4.1-primary-endpoints", sec, "PENDING", "no outer assessment unit yet (EVALUATION_LOCK / outer__*)")
        rep.add("4.2-secondary-endpoints", sec, "PENDING", "no outer assessment unit yet")
        rep.add("4.3-outer-inputs", sec, "PENDING", "no outer assessment unit yet")
        return None
    D = ctx.data()
    D.dev_open = True
    preds = load_preds(ctx, lock)
    if not preds:
        rep.add("4.1-primary-endpoints", sec, "PENDING", "outer units present but no preds.npz for a scored label")
        return None
    # input consistency
    dev = D.idx["DEVELOPMENT_ASSESSMENT"]
    base = None
    issues = []
    for (k, lab), p in sorted(preds.items()):
        for key in ("assess_row_id", "assess_unit", "sex", "race", "y_income", "y_occ"):
            if key not in p:
                issues.append(f"s{k}/{lab}: missing {key}")
        if base is None:
            base = p
        for key in ("assess_row_id", "assess_unit", "sex", "y_income", "y_occ"):
            if key in p and not np.array_equal(p[key], base[key]):
                issues.append(f"s{k}/{lab}: {key} differs between outer units")
        for key in VIEW_KEYS.values():
            if key in p:
                P = np.asarray(p[key])
                if P.shape[0] != 3 or P.shape[1] != len(p["sex"]):
                    issues.append(f"s{k}/{lab}: {key} shape {P.shape}")
    src_ok = (np.array_equal(base["assess_row_id"], D.row_id[dev]) and np.array_equal(base["assess_unit"], D.unit[dev])
              and np.array_equal(base["sex"], D.label("sex")[dev]) and np.array_equal(base["race"], D.label("race")[dev])
              and np.array_equal(base["y_income"], D.label("y_income")[dev])
              and np.array_equal(base["y_occ"], D.label("y_occupation_group")[dev]))
    # race support
    race = D.label("race")
    sup = [int(c) for c in np.unique(race[race >= 0])
           if all((race[D.idx[r]] == c).sum() >= 30 for r in ("AUDIT_FIT", "INNER_SELECTION", "DEVELOPMENT_ASSESSMENT"))]
    race_ok = True
    for (k, lab), p in preds.items():
        if "race_pos" in p:
            rp = np.flatnonzero(np.isin(base["race"], sup))
            ok = np.array_equal(p["race_pos"], rp)
            if "race_codes" in p:
                ok &= list(np.asarray(p["race_codes"]).tolist()) == sup
            ok &= np.array_equal(p["race_y"], np.searchsorted(np.array(sup), base["race"][rp]))
            race_ok &= bool(ok)
    # outputs vs frozen releases
    specs = lock_specs(lock)
    out_bad = []
    n_out_checked = 0
    for (k, lab), p in preds.items():
        spec = specs.get((k, lab))
        if not isinstance(spec, dict):
            continue
        if spec.get("kind") == "neural":
            pairs = [(spec["unit"], "hard1", "hard1"), (spec["unit"], "hard2", "hard2"),
                     (spec["unit"], "p1", "p1"), (spec["unit"], "p2", "p2")]
        elif spec.get("kind") == "fare" and len(spec.get("units") or []) == 2:
            u0, u1 = spec["units"]
            pairs = [(u0, "hard", "hard1"), (u0, "p", "p1"), (u1, "hard", "hard2"), (u1, "p", "p2")]
        else:
            continue
        for unit, rkey, pkey in pairs:
            rel = ctx.release(unit)
            if rel is None or rkey not in rel or pkey not in p:
                out_bad.append(f"s{k}/{lab}:{pkey} (release {unit} / key {rkey} absent)")
                continue
            where = {int(r): j for j, r in enumerate(rel["row_id"])}
            j = np.array([where[int(r)] for r in p["assess_row_id"]])
            n_out_checked += 1
            if not np.array_equal(np.asarray(rel[rkey])[j], p[pkey]):
                out_bad.append(f"s{k}/{lab}:{pkey}")
    have = sorted({lab for (_, lab) in preds})
    miss = [f"s{k}/{lab}" for k in SEEDS for lab in SCORED if (k, lab) not in preds]
    rep.add("4.3-outer-inputs", sec, "PASS" if (not issues and src_ok and race_ok and not out_bad) else "FAIL",
            "preds.npz rows / groups / SEX / race / task labels equal the source DEVELOPMENT_ASSESSMENT rows in source "
            "order; identical across units; race subset = supported classes (>= 30 rows in AUDIT_FIT, "
            "INNER_SELECTION, DEVELOPMENT_ASSESSMENT) remapped 0..K-1; deployed decisions and probabilities equal the "
            "frozen release.npz of the locked unit (FARE: purpose-0 release -> hard1/p1, purpose-1 -> hard2/p2)",
            issues=issues[:20], source_labels_match=src_ok, supported_race_codes=sup, race_ok=race_ok,
            output_mismatches=out_bad, output_arrays_checked=n_out_checked, labels_present=have,
            missing_seed_labels=miss)
    W, G = boot_weights(base["assess_unit"], ctx.B, ctx.boot_seed)
    maj = {0: D.majority("y_income"), 1: D.majority("y_occupation_group")}
    prior = D.sex_prior()
    E = Endpoints(preds, W, maj, prior)
    comps, comp_src = comparators_from(lock, mysel)
    t0 = time.time()
    prim, secd = family_values(E, fam, comps)
    wall = time.time() - t0
    theirs_p = read_endpoint_csv(ctx.res / "PRIMARY_ENDPOINTS.csv")
    theirs_s = read_endpoint_csv(ctx.res / "SECONDARY_ENDPOINTS.csv")
    inf_p = ctx.priv_run / "inference.json"
    inf = jload(inf_p) if inf_p.exists() else None
    rp, bp = compare_family(prim, theirs_p, tol=0.0, abs_tol=5.01e-7)
    rs, bs = compare_family(secd, theirs_s, tol=0.0, abs_tol=5.01e-7)
    alias_bad = [s for s in range(13, 19) if theirs_p and f"P{s}" in theirs_p and f"P{s - 9:02d}" in theirs_p
                 and theirs_p[f"P{s}"] != theirs_p[f"P{s - 9:02d}"]]
    inf_rows, inf_bad = [], []
    if inf is not None:
        for section, mine_sec in (("primary", prim), ("secondary", secd)):
            entries = inf.get(section) or []
            theirs = {e["id"]: e for e in entries if isinstance(e, dict) and "id" in e}
            rows_, bad_ = compare_family(mine_sec, theirs, tol=1e-9)
            inf_rows += [r["id"] for r in rows_ if r["match"] is not None]
            inf_bad += bad_
            for e in theirs.values():
                if e.get("n_finite_replicates") not in (None, ctx.B):
                    inf_bad.append({"id": e["id"], "key": "n_finite_replicates", "inference": e["n_finite_replicates"]})
        missing = [sid for sid in list(prim) + list(secd) if sid not in inf_rows]
        if missing:
            inf_bad.append({"slots_missing_from_inference_json": missing})
    lev_rows, lev_bad = [], []
    if inf is not None and isinstance(inf.get("levels"), dict):
        FMT = {"prim": "auc", "prob": "prob", "hard": "hard"}
        for key, val in inf["levels"].items():
            q = key.split("|")
            try:
                if q[0] == "R":
                    arr = E.R(int(q[1]), q[2], FMT[q[3]], q[4])
                elif q[0] == "acc":
                    arr = E.acc(int(q[1]), q[2], int(q[3]))
                elif q[0] == "accmean":
                    arr = np.mean([E.acc(k, q[1], int(q[2])) for k in SEEDS], 0)
                elif q[0] == "const":
                    arr = E.const[int(q[1])]
                elif q[0] == "LLR":
                    arr = E.llr(int(q[1]), q[2], q[3])
                elif q[0] == "Rmean":
                    arr = np.mean([E.R(k, q[1], "auc", q[2]) for k in SEEDS], 0)
                elif q[0] == "Rrace":
                    arr = E.rrace(int(q[1]), q[2], q[3])
                else:
                    lev_bad.append({"level": key, "problem": "unknown quantity"})
                    continue
            except KeyError as e:
                lev_bad.append({"level": key, "problem": f"missing input {e}"})
                continue
            mp, ms = float(arr[0]), float(np.std(arr[1:], ddof=1))
            ok = (abs(mp - float(val["point"])) <= 1e-9 * max(1.0, abs(mp))
                  and abs(ms - float(val["se"])) <= 1e-9 * max(1.0, abs(ms)))
            lev_rows.append(key)
            if not ok:
                lev_bad.append({"level": key, "mine": [mp, ms], "inference": [val["point"], val["se"]]})
    rl = read_levels_csv(ctx.res / "RAW_LEVELS.csv")
    rl_bad = []
    if rl is not None and inf is not None and isinstance(inf.get("levels"), dict):
        for key, (pt, se) in rl.items():
            v = inf["levels"].get(key)
            if v is None or abs(float(v["point"]) - pt) > 5.01e-7 or abs(float(v["se"]) - se) > 5.01e-7:
                rl_bad.append({"level": key, "csv": [pt, se], "inference": v})
        if len(rl) != len(inf["levels"]):
            rl_bad.append({"csv_rows": len(rl), "inference_levels": len(inf["levels"])})
    rep.add("4.4-per-seed-levels", sec,
            "PENDING" if (inf is None or not isinstance(inf.get("levels"), dict))
            else ("FAIL" if (lev_bad or rl_bad or not lev_rows) else "PASS"),
            "every per-seed level in inference.json (R by format/view, accuracy, constant, LLR, supported-race AUC, "
            "seed means) equals the replay's point and bootstrap SE (relative 1e-9); RAW_LEVELS.csv equals those "
            "levels up to 6-decimal rounding", levels_compared=len(lev_rows), mismatches=lev_bad[:30],
            raw_levels_rows=None if rl is None else len(rl), raw_levels_mismatches=rl_bad[:30])
    st_p = "PENDING" if theirs_p is None else ("FAIL" if (bp or alias_bad or inf_bad) else "PASS")
    rep.add("4.1-primary-endpoints", sec, st_p,
            "18 primary slots recomputed from preds.npz: SEX AUC of each attacker-seed probability (column 1) averaged "
            "over 3 attacker seeds; accuracies from deployed hard decisions; constant = DEFENSE_FIT majority on the "
            "assessment rows; per-seed statistics averaged over seeds; own paired multinomial bootstrap over exact-"
            "record groups (B = 1999, seed 20261004); SE = sd(ddof=1); point +- 2.991316 SE; compared with "
            "PRIMARY_ENDPOINTS.csv (and inference.json when present)",
            groups=G, rows=int(W.shape[1]), B=int(W.shape[0] - 1), comparators=comp_src, values=prim,
            csv_rows=rp, csv_mismatches=bp, alias_csv_mismatches=alias_bad, inference_mismatches=inf_bad[:20],
            inference_slots_compared=len(inf_rows), wall_s=wall)
    st_s = "PENDING" if theirs_s is None else ("FAIL" if (bs or inf_bad) else "PASS")
    rep.add("4.2-secondary-endpoints", sec, st_s,
            "30 secondary slots (output formats, supported race macro one-vs-rest AUC, log-loss recovery LLR = 1 - "
            "CE/CE_prior with the DEFENSE_FIT SEX prior, fixed-beta components, J-G vs controls, coalition minus best "
            "local), z = 3.143980, compared with SECONDARY_ENDPOINTS.csv", values=secd, csv_rows=rs,
            csv_mismatches=bs)
    return {"primary": prim, "secondary": secd, "inference": inf}


# ================================================================================================ 5. conjunctions
def claims(prim, mysel):
    out = {}
    for claim, ids in (("A", [f"P{i:02d}" for i in range(1, 10)]), ("B", [f"P{i:02d}" for i in range(10, 19)])):
        reasons = []
        for sid in ids:
            d = prim.get(sid, {}).get("decision")
            if d != "PASS":
                reasons.append(f"{sid} {d}")
        for k in SEEDS:
            s = mysel.get(k, {})
            stB = s.get("stage_B")
            stC = s.get("stage_C")
            lr = stB.get("L-R", {}).get("status") if isinstance(stB, dict) else None
            if lr not in ("NOMINEE", "TASK_ONLY_ALIAS"):
                reasons.append(f"s{k}: L-R {lr or 'absent'} (no valid reference)")
                continue
            if not isinstance(stC, dict):
                reasons.append(f"s{k}: Stage C selection absent")
                continue
            if stC["J-G"]["status"] != "NOMINEE":
                reasons.append(f"s{k}: J-G {stC['J-G']['status']}")
            if claim == "A" and stC["controls"]["L-G"]["status"] != "NOMINEE":
                reasons.append(f"s{k}: L-G {stC['controls']['L-G']['status']}")
            if claim == "B" and stC.get("C*") is None:
                reasons.append(f"s{k}: no C*")
        out[claim] = {"decision": "ESTABLISHED" if not reasons else "NOT_ESTABLISHED", "reasons": reasons}
    return out


def check_claims(ctx: Ctx, rep: Report, endp, mysel):
    sec = "5-conjunction"
    if endp is None:
        rep.add("5.1-claims", sec, "PENDING", "endpoints not available yet")
        return
    mine = claims(endp["primary"], mysel)
    for c, ids in (("A", range(1, 10)), ("B", range(10, 19))):
        mine[c]["clauses_passing"] = sum(endp["primary"].get(f"P{i:02d}", {}).get("decision") == "PASS" for i in ids)
    theirs, bad = {}, []
    inf = endp.get("inference")
    if inf is not None:
        for c in ("A", "B"):
            x = inf.get(f"claim{c}")
            if isinstance(x, dict):
                theirs[c] = {"decision": x.get("decision"), "clauses_passing": x.get("clauses_passing")}
                if x.get("decision") != mine[c]["decision"]:
                    bad.append({"claim": c, "key": "decision", "runner": x.get("decision"), "mine": mine[c]["decision"]})
                if x.get("clauses_passing") is not None and x["clauses_passing"] != mine[c]["clauses_passing"]:
                    bad.append({"claim": c, "key": "clauses_passing", "runner": x["clauses_passing"],
                                "mine": mine[c]["clauses_passing"]})
                for k, ss in (x.get("seed_status") or {}).items():
                    stC = mysel.get(int(k), {}).get("stage_C")
                    stB = mysel.get(int(k), {}).get("stage_B")
                    if not isinstance(stC, dict):
                        continue
                    mine_ss = {"valid_reference": stB["L-R"]["status"] in ("NOMINEE", "TASK_ONLY_ALIAS"),
                               "J-G": stC["J-G"]["status"], "L-G": stC["controls"]["L-G"]["status"],
                               "C*": _lab_of(stC["C*"])}
                    for key, v in mine_ss.items():
                        if key in ss and ss[key] != v:
                            bad.append({"claim": c, "seed": k, "key": key, "runner": ss[key], "mine": v})
    st = "PENDING" if not theirs else ("FAIL" if bad else "PASS")
    rep.add("5.1-claims", sec, st,
            "Claim A = P01-P09 all PASS + valid L-R + J-G and L-G NOMINEE on seeds 0-2; Claim B = P10-P18 all PASS + "
            "valid L-R + J-G NOMINEE + C* present on seeds 0-2 (statuses from the independent selection replay); "
            "decision, clause count and per-seed statuses compared with inference.json",
            mine=mine, runner=theirs, mismatches=bad)


# ================================================================================================ 6. multipliers
def replay_lambda(trace, beta, c, base, n_epochs=PROT_EPOCHS, update_last=False):
    lam = [0.0, 0.0]
    rows, worst = [], 0.0
    for ent in sorted(trace, key=lambda e: e["epoch"]):
        if ent["epoch"] < n_epochs or update_last:
            lam = [float(min(max(lam[i] + beta * (ent["R_calib"][i] - c[i]), 0.0), 3.0 * beta)) for i in (0, 1)]
        got = [float(x) for x in ent["lambda"]]
        d = max(abs(got[i] - lam[i]) for i in (0, 1))
        if base == "joint":
            eff = {"v1": beta / 3 + lam[0], "v2": beta / 3 + lam[1], "pair": beta / 3}
        else:
            eff = {"v1": beta / 2 + lam[0], "v2": beta / 2 + lam[1], "pair": 0.0}
        de = max(abs(float(ent["effective_weights"][v]) - eff[v]) for v in eff)
        worst = max(worst, d, de)
        rows.append({"epoch": ent["epoch"], "lambda_logged": got, "lambda_replay": lam, "eff_logged": ent["effective_weights"],
                     "eff_replay": eff, "c_logged": ent.get("c")})
    return rows, worst, lam


def check_lambda(ctx: Ctx, rep: Report):
    sec = "6-multipliers"
    runs = [n for n in ctx.unit_names("run__C__") if ctx.complete(n)]
    out, bad = [], []
    max_ratio = {}
    for n in runs:
        rec = ctx.rec(n)
        diag = rec.get("diag", {})
        arm, beta, k = rec.get("arm"), float(rec.get("beta")), int(rec.get("seed"))
        tag = n[len(run_name("C", k, arm, beta)):] if n.startswith(run_name("C", k, arm, beta)) else None
        if tag is None:
            out.append({"run": n, "ok": False, "error": "run name does not match its record"})
            bad.append(n)
            continue
        base = "joint" if arm in JOINT else "local"
        cb = diag.get("coefficients_base")
        exp_base = ({"v1": beta / 3, "v2": beta / 3, "pair": beta / 3} if base == "joint"
                    else {"v1": beta / 2, "v2": beta / 2, "pair": 0.0})
        row = {"run": n, "arm": arm, "beta": beta, "seed": k, "tag": tag, "tkind": rec.get("tkind")}
        ok = True
        if cb is not None:
            row["base_coefficients_ok"] = all(abs(float(cb[v]) - exp_base[v]) < 1e-15 for v in exp_base)
            ok &= row["base_coefficients_ok"]
        trace = diag.get("lambda_trace", [])
        if arm in GUARDED:
            cal = ctx.rec(f"calib__s{k}") if ctx.complete(f"calib__s{k}") else None
            if cal is None:
                row["status"] = "PENDING (calib record absent)"
                out.append(row)
                continue
            c = [float(x) for x in cal["c"]]
            row["c_calib"] = c
            cm = cal.get("calib") or {}
            margin = float(cal.get("margin", 0.005))
            if cm:
                row["c_equals_R_plus_margin"] = all(abs(c[i] - (float(cm[f"v{i + 1}"]["R"]) + 0.005)) < 1e-12
                                                    for i in (0, 1)) and margin == 0.005
                ok &= row["c_equals_R_plus_margin"]
            logged_c = diag.get("c") or rec.get("budgets")
            row["c_logged_equals_calib"] = logged_c is not None and [float(x) for x in logged_c] == c
            ok &= row["c_logged_equals_calib"]
            epochs = [e["epoch"] for e in trace]
            row["trace_epochs_ok"] = epochs == list(REFIT_EPOCHS)
            ok &= row["trace_epochs_ok"]
            rows_, worst, _ = replay_lambda(trace, beta, c, base)
            _, worst_alt, _ = replay_lambda(trace, beta, c, base, update_last=True)
            row["max_abs_diff"] = worst
            row["alt_convention_update_at_epoch20_max_abs_diff"] = worst_alt
            ok &= worst <= 1e-12
            # R_calib vs refit receipts
            refits = diag.get("refits", [])
            rmis = 0
            for ent in trace:
                rf = [r for r in refits if r.get("epoch") == ent["epoch"]]
                if rf and "calib" in rf[0]:
                    if any(abs(float(rf[0]["calib"][f"v{i + 1}"]["R"]) - float(ent["R_calib"][i])) > 0 for i in (0, 1)):
                        rmis += 1
            row["R_calib_vs_refit_receipt_mismatches"] = rmis
            ok &= rmis == 0
            lam_after = {}
            for r_ in rows_:
                lam_after[r_["epoch"]] = r_["lambda_replay"]
            ck_bad, n_ck = [], 0
            for e in EPOCHS:
                cn = ck_name("C", k, arm, beta, e) + tag
                cr = ctx.rec(cn) if ctx.complete(cn) else None
                if cr is None or cr.get("lam") is None:
                    continue
                n_ck += 1
                last = max(x for x in REFIT_EPOCHS if x < e)
                if [float(x) for x in cr["lam"]] != [float(x) for x in lam_after.get(last, [None, None])]:
                    ck_bad.append({"ck": cn, "lam": cr["lam"], "expected_after_refit": last, "replay": lam_after.get(last)})
            row["checkpoint_lambda_mismatches"] = ck_bad
            row["checkpoint_lambdas_compared"] = n_ck
            ok &= not ck_bad and n_ck == len(EPOCHS)
            lmax = max((max(r_["lambda_replay"]) for r_ in rows_), default=0.0)
            row["max_lambda_over_base_weight"] = (lmax / (beta / 3)) if beta > 0 else None
            max_ratio[n] = row["max_lambda_over_base_weight"]
            row["trace"] = rows_
        else:
            row["trace_empty"] = len(trace) == 0
            ok &= row["trace_empty"]
            ck_bad = []
            for e in EPOCHS:
                cn = ck_name("C", k, arm, beta, e) + tag
                cr = ctx.rec(cn) if ctx.complete(cn) else None
                if cr is not None and cr.get("lam") is not None and [float(x) for x in cr["lam"]] != [0.0, 0.0]:
                    ck_bad.append(cn)
            row["checkpoint_lambda_nonzero"] = ck_bad
            ok &= not ck_bad
        row["ok"] = bool(ok)
        if not ok:
            bad.append(n)
        out.append(row)
    # engineering receipts (beta = 0 parity units) exercise the same replay
    par = []
    for n in ctx.unit_names("parity__"):
        if not ctx.complete(n) or ".quarantined" in n:
            continue
        rec = ctx.rec(n)
        if rec.get("arm") in GUARDED and rec.get("lambda_trace"):
            rows_, worst, _ = replay_lambda(rec["lambda_trace"], 0.0, [0.0, 0.0],
                                            "joint" if rec["arm"] in JOINT else "local")
            par.append({"unit": n, "max_abs_diff": worst, "epochs": [r["epoch"] for r in rows_]})
    par_ok = all(p["max_abs_diff"] == 0.0 and p["epochs"] == list(REFIT_EPOCHS) for p in par)
    if not runs:
        rep.add("6.1-lambda-replay", sec, "PENDING",
                "no Stage C run record yet; the replay was exercised on the beta = 0 parity receipts",
                parity_receipts=par, parity_receipts_ok=par_ok)
        return
    rep.add("6.1-lambda-replay", sec, "FAIL" if bad else "PASS",
            "lambda_i <- clip(lambda_i + beta (R_i - c_i), 0, 3 beta) replayed from lambda = 0 with the logged "
            "R_calib and c from calib__s{k} (c_i = R_i + 0.005), updated at refits 0..16 (the epoch-20 refit is "
            "diagnosis only); effective weights (beta/3 + lambda_1, beta/3 + lambda_2, beta/3) joint, (beta/2 + "
            "lambda_1, beta/2 + lambda_2, 0) local; base coefficients; checkpoint lambdas = lambda after the last refit "
            "before the checkpoint; unguarded arms carry no multipliers",
            runs=out, failing=bad, max_lambda_over_base_weight=max_ratio, parity_receipts=par,
            parity_receipts_ok=par_ok)


# ================================================================================================ 7. critic gap
def critic_forward(sd, Z):
    idx = sorted({int(k.split(".")[0]) for k in sd})
    h = Z
    for j, i in enumerate(idx):
        h = F.linear(h, sd[f"{i}.weight"], sd[f"{i}.bias"])
        if j < len(idx) - 1:
            h = torch.relu(h)
    return h


def critic_kind_ok(sd, kind, dv):
    shapes = [tuple(sd[k].shape) for k in sorted(sd, key=lambda s: (int(s.split(".")[0]), s)) if k.endswith("weight")]
    if kind == "A":
        return shapes == [(32, dv), (2, 32)]
    return shapes == [(64, dv), (64, 64), (2, 64)]


def snapshot_views(snap, Xt):
    st = snap["model"]
    head = snap["critic_head"]
    v = {}
    for i in (0, 1):
        r = encode_state(st, i, Xt)
        W, b = head[i] if i in head else head[str(i)]
        lg = F.linear(r, W.float(), b.float())
        v[i] = torch.cat([r, lg - lg.mean(1, keepdim=True)], 1)
    return {"v1": v[0], "v2": v[1], "pair": torch.cat([v[0], v[1]], 1)}


def apply_transform(ts, V):
    return (V - ts["mu"].float()) @ ts["W"].float()


def snapshot_ce(snap, Xt, S, sub=None):
    """{view: {kind: CE}} of the critics in a snapshot on rows (Xt, S), with the snapshot's own transform. With `sub`,
    views are computed on all rows of Xt and then restricted to positions `sub` (float32 batch-shape variant)."""
    V = snapshot_views(snap, Xt)
    if sub is not None:
        V = {v: x[sub] for v, x in V.items()}
    out, shape_ok = {}, True
    with torch.no_grad():
        for v, bank in snap["critics"].items():
            ts = snap["transforms"][v]
            Z = apply_transform(ts, V[v])
            out[v] = {}
            for kind, sd in bank.items():
                shape_ok &= critic_kind_ok(sd, kind, Z.shape[1])
                out[v][kind] = float(F.cross_entropy(critic_forward(sd, Z), S))
    return out, shape_ok


def snapshot_R(snap, Xt, S, prior):
    """Recovery surrogate R_v = (CE_const - min(CE_const, CE_A, CE_B)) / H on the rows."""
    ce, _ = snapshot_ce(snap, Xt, S)
    p = np.asarray(prior, float)
    H = float(-(p * np.log(p)).sum())
    ce0 = float(F.nll_loss(torch.log(torch.tensor(p, dtype=torch.float32)).expand(len(S), 2), S))
    return {v: (ce0 - min([ce0] + list(ce[v].values()))) / H for v in ce}, ce0


REFIT_BLOCK = "final_refit_at_theta_T"
SNAP_TOKEN = {"start": "start", "mid": "mid", "final": "final", "refit20": REFIT_BLOCK}


def track_blocks(rec):
    """Aligned snapshot blocks of a track record: snapshots/{start, mid, final} plus the epoch-20 refit block."""
    out = dict(rec.get("snapshots") or {})
    if isinstance(rec.get(REFIT_BLOCK), dict):
        out[REFIT_BLOCK] = rec[REFIT_BLOCK]
    return out


def check_critic_gap(ctx: Ctx, rep: Report, lock):
    sec = "7-critic-gap"
    tracks = [n for n in ctx.unit_names("track__") if ctx.complete(n)]
    if not tracks:
        rep.add("7.1-gap-recompute", sec, "PENDING", "no track__ record yet")
    else:
        rows, bad, n_cmp, unparsed = [], [], 0, []
        for n in tracks:
            rec = ctx.rec(n)
            snaps = track_blocks(rec)
            if not snaps:
                unparsed.append(n)
                continue
            for snap, views in snaps.items():
                if not isinstance(views, dict):
                    continue
                for v, node in views.items():
                    kinds = (node or {}).get("kinds") if isinstance(node, dict) else None
                    if not isinstance(kinds, dict) or not kinds:
                        continue
                    for rows_ in ("calib", "inner"):
                        try:
                            on = {kk: float(kinds[kk]["online"][rows_]) for kk in kinds}
                            fr = {kk: float(kinds[kk]["fresh"][rows_]) for kk in kinds}
                        except (KeyError, TypeError):
                            continue
                        reg = float(np.mean([on[kk] - fr[kk] for kk in sorted(on)]))
                        bob = float(min(on.values()) - min(fr.values()))
                        r = {"track": n, "snapshot": snap, "view": v, "rows": rows_, "kinds": sorted(on),
                             "registered_gap": reg, "best_of_bank_gap": bob}
                        ok = True
                        for key, val in ((f"gap_registered_{rows_}", reg), (f"gap_best_of_bank_{rows_}", bob),
                                         (f"online_best_{rows_}", min(on.values())),
                                         (f"fresh_best_{rows_}", min(fr.values()))):
                            if key in node:
                                n_cmp += 1
                                d = abs(float(node[key]) - val)
                                r[f"diff_{key}"] = d
                                ok &= d <= 1e-7
                        r["ok"] = bool(ok)
                        if not ok:
                            bad.append(r)
                        rows.append(r)
        st = "FAIL" if (bad or unparsed) else ("PASS" if n_cmp else "FAIL")
        rep.add("7.1-gap-recompute", sec, st,
                "registered gap = mean over kinds of CE(online_j) - CE(fresh_j); best-of-bank = min_j CE(online_j) - "
                "min_j CE(fresh_j); recomputed per snapshot / view / row set (CALIB, INNER_SELECTION) from the per-kind "
                "CEs in every track record (snapshots -> view -> kinds -> online/fresh -> rows) and compared with the "
                "record's gap_registered_*, gap_best_of_bank_*, online_best_*, fresh_best_* fields (1e-7; float32 "
                "inputs); a record with no comparable field fails", tracks=len(tracks), comparisons=n_cmp,
                unparsed_tracks=unparsed, rows=rows[:400], mismatches=bad[:30])
    # re-evaluation of online critics from saved snapshots on CALIB rows
    D = ctx.data()
    cal = D.idx["CALIB"]
    X = None
    Xdf = None
    S = torch.from_numpy(D.label("sex")[cal])
    sub_cal = torch.from_numpy(np.searchsorted(D.idx["DEFENSE_FIT"], cal))
    prior = D.sex_prior()
    evals, bad2, refit_rows, bad3 = [], [], [], []
    runs_done = 0
    for n in tracks:
        rec = ctx.rec(n)
        run = rec.get("run") or rec.get("of") or rec.get("source_run")
        if not isinstance(run, str) or not run.startswith("run__"):
            pc = None
            for path, v in flatten(rec):
                if isinstance(v, str) and v.startswith("ck__C__"):
                    pc = parse_ck(v)
                    break
            if pc is None:
                continue
            run = run_name("C", pc["seed"], pc["arm"], pc["beta"]) + pc["tag"]
        rd = ctx.udir(run)
        if not ctx.complete(run):
            continue
        if X is None:
            X = torch.from_numpy(np.ascontiguousarray(D.X[cal], dtype=np.float32))
        snaps = {}
        if (rd / "final.pt").exists():
            fin = torch_load(rd / "final.pt")
            if "theta_T_minus_1" in fin:
                snaps["final"] = fin["theta_T_minus_1"]
            if "refit_critics" in fin:
                snaps["refit20"] = fin["refit_critics"]
        if (rd / "captures.pt").exists():
            cap = torch_load(rd / "captures.pt")
            for key, nm in ((1, "start"), (MID_STEP, "mid")):
                if key in cap:
                    snaps[nm] = cap[key]
                elif str(key) in cap:
                    snaps[nm] = cap[str(key)]
        runs_done += 1
        flat = list(flatten(rec))
        for nm, snap in snaps.items():
            tok = SNAP_TOKEN[nm]
            ce, shape_ok = snapshot_ce(snap, X, S)
            ce_df = None
            for v in ce:
                for kind, val in ce[v].items():
                    hits = [x for p, x in flat if isinstance(x, (int, float)) and tok in p and v in p and kind in p
                            and "online" in p and any("calib" in q.lower() for q in p)]
                    r = {"track": n, "run": run, "snapshot": nm, "view": v, "kind": kind, "mine": val,
                         "record": hits[0] if len(hits) == 1 else hits, "shape_ok": shape_ok, "variant": "calib-batch"}
                    if len(hits) == 1:
                        r["abs_diff"] = abs(hits[0] - val)
                        if r["abs_diff"] > 1e-5:
                            if ce_df is None:
                                if Xdf is None:
                                    Xdf = torch.from_numpy(np.ascontiguousarray(D.X[D.idx["DEFENSE_FIT"]],
                                                                                dtype=np.float32))
                                ce_df, _ = snapshot_ce(snap, Xdf, S, sub=sub_cal)
                            d2 = abs(hits[0] - ce_df[v][kind])
                            if d2 < r["abs_diff"]:
                                r.update({"abs_diff": d2, "mine": ce_df[v][kind], "variant": "defense-fit-subset"})
                        if r["abs_diff"] > 1e-5 or not shape_ok:
                            bad2.append(r)
                    evals.append(r)
        # epoch-20 R_calib from the refit critics (guarded runs log it in lambda_trace)
        rr = ctx.rec(run)
        tr = (rr or {}).get("diag", {}).get("lambda_trace") or []
        if "refit20" in snaps:
            Rm, ce0 = snapshot_R(snaps["refit20"], X, S, prior)
            r20 = [e for e in tr if e["epoch"] == PROT_EPOCHS]
            refits = (rr or {}).get("diag", {}).get("refits") or []
            rc = [r for r in refits if r.get("epoch") == PROT_EPOCHS]
            logged = None
            if r20:
                logged = {"v1": r20[0]["R_calib"][0], "v2": r20[0]["R_calib"][1]}
            elif rc and "calib" in rc[0]:
                logged = {v: rc[0]["calib"][v]["R"] for v in ("v1", "v2", "pair") if v in rc[0]["calib"]}
            if logged:
                d = max(abs(float(logged[v]) - Rm[v]) for v in logged)
                row = {"run": run, "logged": logged, "recomputed": Rm, "max_abs_diff": d}
                refit_rows.append(row)
                if d > 1e-5:
                    bad3.append(row)
    matched = [e for e in evals if "abs_diff" in e]
    if not tracks:
        rep.add("7.2-online-critic-reeval", sec, "PENDING", "no track__ record yet")
    else:
        hard = [e for e in bad2 if e["abs_diff"] > 1e-3 or not e["shape_ok"]]
        st = "FAIL" if hard else ("WARN" if (bad2 or not matched) else "PASS")
        rep.add("7.2-online-critic-reeval", sec, st,
                "online critics re-evaluated from captures.pt (step 1, step 761) and final.pt (theta_{T-1}; and the "
                "epoch-20 refit critics at theta_T for refreshed arms) with an "
                "own forward pass: v_i = [r_i, centred(W_w r_i + b_w)] from the stored critic_head, T(V) = (V - mu) @ W "
                "(float32), critic A = 32-unit, B = 64-64 MLP; CE on CALIB rows (DEFENSE_FIT, critic hash u >= 0.85) "
                "compared with the track record's online CALIB values. PASS <= 1e-5; WARN <= 1e-3 (float32 rounding "
                "amplified ~1e4 along the floored null directions of the view transform); FAIL otherwise",
                runs_evaluated=runs_done, values=evals[:120], matched=len(matched), mismatches=bad2[:30])
        rep.add("7.3-refit-R-calib", sec, "FAIL" if bad3 else ("PASS" if refit_rows else "PENDING"),
                "R_v = (CE_const - min(CE_const, CE_A, CE_B)) / H recomputed on CALIB from the epoch-20 refit critics "
                "(final.pt) and compared with the logged epoch-20 R_calib", rows=refit_rows, mismatches=bad3)


def _close(a, b, rel=6e-6, abs_=1e-12):
    """Equality up to %g (6 significant digit) CSV formatting."""
    return abs(a - b) <= rel * max(abs(a), abs(b)) + abs_


def check_tracking_csv(ctx: Ctx, rep: Report, mysel):
    sec = "7-critic-gap"
    p = ctx.res / "CRITIC_TRACKING.csv"
    if not p.exists():
        rep.add("7.4-critic-tracking-csv", sec, "PENDING", "CRITIC_TRACKING.csv not written yet")
        return
    rows = list(csv.DictReader(open(p, newline="")))
    bad, n_cmp, n_rows = [], 0, 0
    for r in rows:
        tn = f"track__s{r['seed']}__{r['arm']}"
        rec = ctx.rec(tn) if ctx.complete(tn) else None
        if rec is None:
            bad.append({"row": r, "problem": f"{tn} absent"})
            continue
        try:
            node = track_blocks(rec)[r["snapshot"]][r["view"]]
            rows_ = r["rows"]
            kinds = node["kinds"]
            on = {kk: float(kinds[kk]["online"][rows_]) for kk in kinds}
            fr = {kk: float(kinds[kk]["fresh"][rows_]) for kk in kinds}
        except (KeyError, TypeError) as e:
            bad.append({"row": {k: r[k] for k in ("seed", "arm", "snapshot", "view", "rows")}, "problem": repr(e)})
            continue
        n_rows += 1
        mine = {"gap_registered_mean_paired": float(np.mean([on[kk] - fr[kk] for kk in sorted(on)])),
                "gap_best_of_bank": min(on.values()) - min(fr.values()), "online_best_ce": min(on.values()),
                "fresh_best_ce": min(fr.values()), "const_ce": float(node["const_ce"][rows_])}
        for kk in on:
            mine[f"online_ce_{kk}"] = on[kk]
            mine[f"fresh_ce_{kk}"] = fr[kk]
        for key, val in mine.items():
            if r.get(key) not in (None, ""):
                n_cmp += 1
                if not _close(float(r[key]), val):
                    bad.append({"row": {k: r[k] for k in ("seed", "arm", "snapshot", "view", "rows")}, "key": key,
                                "csv": r[key], "recomputed": val})
        arm = r["arm"]
        k = int(r["seed"])
        stC = mysel.get(k, {}).get("stage_C")
        if isinstance(stC, dict) and arm in ("J-G",) + TRAINED_CONTROLS and r.get("selection_status"):
            mine_st = stC["J-G"]["status"] if arm == "J-G" else stC["controls"][arm]["status"]
            n_cmp += 1
            if r["selection_status"] != mine_st:
                bad.append({"row": {"seed": k, "arm": arm}, "key": "selection_status", "csv": r["selection_status"],
                            "replay": mine_st})
    st = "FAIL" if bad else ("PASS" if n_cmp else "FAIL")
    rep.add("7.4-critic-tracking-csv", sec, st,
            "every CRITIC_TRACKING.csv row equals the registered gap, best-of-bank gap, best CEs, constant CE and "
            "per-kind CEs recomputed from the track record (up to %g formatting), and its selection status equals "
            "the independent replay", csv_rows=len(rows), rows_matched_to_records=n_rows, comparisons=n_cmp,
            mismatches=bad[:30])


def check_weight_csv(ctx: Ctx, rep: Report):
    sec = "6-multipliers"
    p = ctx.res / "GRADIENT_AND_WEIGHT_DIAGNOSTICS.csv"
    if not p.exists():
        rep.add("6.2-weight-diagnostics-csv", sec, "PENDING", "GRADIENT_AND_WEIGHT_DIAGNOSTICS.csv not written yet")
        return
    rows = list(csv.DictReader(open(p, newline="")))
    bad, n_cmp, defs = [], 0, {"over_beta/3": 0, "over_arm_base_weight": 0, "neither": 0}
    for r in rows:
        u = r.get("unit", "")
        if not u.startswith("run__") or not ctx.complete(u):
            continue
        rec = ctx.rec(u)
        arm, beta = rec.get("arm"), float(rec.get("beta"))
        base = "joint" if arm in JOINT else "local"
        exp = ({"v1": beta / 3, "v2": beta / 3, "pair": beta / 3} if base == "joint"
               else {"v1": beta / 2, "v2": beta / 2, "pair": 0.0})
        for v in ("v1", "v2", "pair"):
            key = f"base_w_{v}"
            if r.get(key) not in (None, ""):
                n_cmp += 1
                if not _close(float(r[key]), exp[v]):
                    bad.append({"unit": u, "key": key, "csv": r[key], "expected": exp[v]})
        lam = [0.0, 0.0]
        lmax = 0.0
        if arm in GUARDED:
            k = int(rec.get("seed"))
            cal = ctx.rec(f"calib__s{k}") if ctx.complete(f"calib__s{k}") else None
            if cal is None:
                continue
            c = [float(x) for x in cal["c"]]
            for i, key in enumerate(("budget_c1", "budget_c2")):
                if r.get(key) not in (None, ""):
                    n_cmp += 1
                    if not _close(float(r[key]), c[i]):
                        bad.append({"unit": u, "key": key, "csv": r[key], "calib": c[i]})
            rows_, _, lam = replay_lambda(rec.get("diag", {}).get("lambda_trace", []), beta, c, base)
            lmax = max([max(x["lambda_replay"]) for x in rows_] + [0.0])
        for i, key in enumerate(("lambda1_final", "lambda2_final")):
            if r.get(key) not in (None, ""):
                n_cmp += 1
                if not _close(float(r[key]), lam[i]):
                    bad.append({"unit": u, "key": key, "csv": r[key], "replay": lam[i]})
        if r.get("lambda_max_over_base") not in (None, "") and beta > 0:
            got = float(r["lambda_max_over_base"])
            if _close(got, lmax / (beta / 3)):
                defs["over_beta/3"] += 1
            elif _close(got, lmax / (beta / 3 if base == "joint" else beta / 2)):
                defs["over_arm_base_weight"] += 1
            else:
                defs["neither"] += 1
    st = "FAIL" if bad else ("PASS" if n_cmp else "PENDING")
    rep.add("6.2-weight-diagnostics-csv", sec, st,
            "GRADIENT_AND_WEIGHT_DIAGNOSTICS.csv base weights equal (beta/3, beta/3, beta/3) joint and (beta/2, "
            "beta/2, 0) local; budgets equal calib__s{k}.c; final lambdas equal the independent replay (up to %g "
            "formatting); the lambda_max_over_base definition matched is reported (descriptive)",
            comparisons=n_cmp, mismatches=bad[:30], lambda_max_over_base_definition_matches=defs)


def check_stage_b_freeze(ctx: Ctx, rep: Report, mysel):
    sec = "8-integrity"
    p = ctx.res / "STAGE_B_SELECTION.json"
    if not p.exists():
        rep.add("8.7-stage-B-freeze", sec, "PENDING", "STAGE_B_SELECTION.json not present")
        return
    fr = jload(p)
    rows, bad = [], []
    for k in SEEDS:
        stB = mysel.get(k, {}).get("stage_B")
        if not isinstance(stB, dict):
            continue
        d = _seed_dict(fr, k)
        for arm in ("L-R", "L-O"):
            f = _find_arm(d, arm) or {}
            row = {"seed": k, "arm": arm, "freeze_status": f.get("status"), "freeze_unit": f.get("unit"),
                   "replay_status": stB[arm].get("status"), "replay_unit": stB[arm].get("unit")}
            row["match"] = row["freeze_status"] == row["replay_status"] and row["freeze_unit"] == row["replay_unit"]
            if not row["match"]:
                bad.append(row)
            rows.append(row)
    timing = {}
    st_t = "SKIP"
    if ctx.git:
        st_t, timing = lock_before(ctx, f"{RES_REL}/STAGE_B_SELECTION.json", "ck__C__")
    st = "FAIL" if (bad or st_t == "FAIL") else ("PASS" if rows else "PENDING")
    rep.add("8.7-stage-B-freeze", sec, st,
            "the pushed Stage B freeze (STAGE_B_SELECTION.json) equals the independent Stage B replay and was "
            "committed / pushed before the first Stage C checkpoint unit (ck__C__)", rows=rows, mismatches=bad,
            timing_status=st_t, timing=timing)


def check_selection_tables(ctx: Ctx, rep: Report, mysel):
    sec = "3-selection"
    p = ctx.res / "SELECTION_TABLE.csv"
    if not p.exists():
        rep.add("3.6-selection-table", sec, "PENDING", "SELECTION_TABLE.csv not written yet")
        return
    mine = {}
    for k in SEEDS:
        stB, stC = mysel.get(k, {}).get("stage_B"), mysel.get(k, {}).get("stage_C")
        if isinstance(stB, dict):
            for arm in ("L-R", "L-O"):
                for c in stB[arm].get("considered") or []:
                    if not c.get("alias"):
                        mine[("B", k, c["unit"])] = {**c, "arm_status": stB[arm]["status"], "guard_pass": None}
        if isinstance(stC, dict):
            for arm in TRAINED_CONTROLS:
                for c in stC["controls"][arm].get("considered") or []:
                    mine[("C", k, c["unit"])] = {**c, "arm_status": stC["controls"][arm]["status"]}
            for c in stC["J-G"].get("considered") or []:
                mine[("C", k, c["unit"])] = {**c, "arm_status": stC["J-G"]["status"]}
    D = ctx.data()
    rows = list(csv.DictReader(open(p, newline="")))
    bad, n_cmp = [], 0

    def tf(x):
        return None if x in (None, "") else (x.strip().lower() == "true")
    for r in rows:
        key = (r["stage"], int(r["seed"]), r["unit"])
        m = mine.get(key)
        if m is None:
            bad.append({"unit": r["unit"], "problem": "not among the replay's candidates"})
            continue
        util = utility_counts(ctx, r["unit"], D)
        checks = {"gates_ok": (tf(r["gates_ok"]), m["gates_pass"]),
                  "guard_ok": (tf(r["guard_ok"]), m.get("guard_pass")),
                  "feasible": (tf(r["feasible"]), m["gates_pass"] and (m.get("guard_pass") in (None, True))),
                  "arm_status": (r["arm_status"], m["arm_status"])}
        for kk, (a, b) in checks.items():
            if a is None and b is None:
                continue
            n_cmp += 1
            if a != b:
                bad.append({"unit": r["unit"], "key": kk, "table": a, "replay": b})
        for w in ("v1", "v2", "pair"):
            n_cmp += 1
            if abs(float(r[f"auc_{w}"]) - m["auc"][w]) > 5.01e-7:
                bad.append({"unit": r["unit"], "key": f"auc_{w}", "table": r[f"auc_{w}"], "inner_record": m["auc"][w]})
        for t, col in ((0, "acc_income"), (1, "acc_occ")):
            n_cmp += 1
            if abs(float(r[col]) - util[t][0] / util[t][2]) > 5.01e-7:
                bad.append({"unit": r["unit"], "key": col, "table": r[col], "replay": util[t][0] / util[t][2]})
    unmatched = len(mine) - len(rows)
    seed_bad, n_seed = [], 0
    ss_p = ctx.res / "SEED_STATUS.json"
    if ss_p.exists():
        ss = jload(ss_p)
        for k in SEEDS:
            stB, stC = mysel.get(k, {}).get("stage_B"), mysel.get(k, {}).get("stage_C")
            d = _seed_dict(ss, k)
            if not isinstance(stC, dict) or not isinstance(d, dict):
                continue
            exp = {"valid_reference": stB["L-R"]["status"] in ("NOMINEE", "TASK_ONLY_ALIAS"), "U": mysel[k]["U_unit"],
                   "L-R unit": stB["L-R"]["unit"], "L-R status": stB["L-R"]["status"],
                   "comparator arm": _lab_of(stC["C*"]), "comparator unit": (stC["C*"] or {}).get("unit")}
            got = {"valid_reference": d.get("valid_reference"), "U": d.get("U"),
                   "L-R unit": (d.get("L-R_reference") or {}).get("unit"),
                   "L-R status": (d.get("L-R_reference") or {}).get("status"),
                   "comparator arm": (d.get("comparator") or {}).get("arm"),
                   "comparator unit": (d.get("comparator") or {}).get("unit")}
            for arm in TRAINED_CONTROLS + ("J-G",):
                m = stC["J-G"] if arm == "J-G" else stC["controls"][arm]
                g = ((d.get("arms") or {}).get(arm) or {})
                exp[f"{arm} status"], got[f"{arm} status"] = m["status"], g.get("status")
                if m["status"] == "NOMINEE":
                    exp[f"{arm} unit"], got[f"{arm} unit"] = m["unit"], g.get("unit")
            for kk in exp:
                n_seed += 1
                if exp[kk] != got[kk]:
                    seed_bad.append({"seed": k, "key": kk, "seed_status": got[kk], "replay": exp[kk]})
    st = "FAIL" if (bad or seed_bad or unmatched) else ("PASS" if n_cmp else "PENDING")
    rep.add("3.6-selection-table", sec, st,
            "every candidate row of SELECTION_TABLE.csv (gates_ok, guard_ok, feasible, arm status, inner AUCs, "
            "INNER_SELECTION accuracies) equals the independent replay, and SEED_STATUS.json (valid reference, U, "
            "L-R reference, comparator, arm statuses / nominee units) equals the replay",
            table_rows=len(rows), replay_candidates=len(mine), comparisons=n_cmp, mismatches=bad[:30],
            seed_status_comparisons=n_seed, seed_status_mismatches=seed_bad)


def check_restore(ctx: Ctx, rep: Report, lock, drive):
    sec = "9-restore"
    if drive is None:
        rep.add("9.1-restore-from-drive", sec, "SKIP", "no --drive given")
        return
    drive = Path(drive)
    bv = ctx.res / "BACKUP_VERIFICATION.json"
    if not drive.exists():
        rep.add("9.1-restore-from-drive", sec, "PENDING", "drive copy not mounted / absent")
        return
    if not bv.exists():
        rep.add("9.1-restore-from-drive", sec, "PENDING", "BACKUP_VERIFICATION.json not written yet (copy in progress)")
        return
    specs = lock_specs(lock)
    want = [(lab, (specs.get((1, lab)) or {}).get("unit")) for lab in ("U", "J-G", "L-G")]
    droot = drive / "run" / "units" if (drive / "run" / "units").exists() else drive / "units"
    D = ctx.data()
    rows, ok_all = [], True
    for lab, u in want:
        row = {"label": lab, "seed": 1, "unit": u}
        if u is None:
            row["problem"] = "no unit in EVALUATION_LOCK"
            ok_all = False
            rows.append(row)
            continue
        dd = droot / u
        st, badf, extra = verify_unit_dir(dd)
        row["drive_complete_json"] = st
        row["drive_bad_files"] = badf
        lc = ctx.udir(u) / "COMPLETE.json"
        row["complete_json_identical_to_local"] = dd.joinpath("COMPLETE.json").exists() and \
            dd.joinpath("COMPLETE.json").read_bytes() == lc.read_bytes()
        rel = ctx.release(u)
        pos = D.pos_of(rel["row_id"])
        mine, heads = deploy(dd, D.X[pos])
        diffs, ok = {}, st == "OK" and row["complete_json_identical_to_local"]
        for i in (1, 2):
            for key, tol in ((f"r{i}", 0.0), (f"c{i}", 1e-9), (f"p{i}", 1e-12)):
                diffs[f"max_abs_{key}"] = float(np.max(np.abs(mine[key] - rel[key])))
                ok &= diffs[f"max_abs_{key}"] <= tol
            diffs[f"hard{i}_mismatches"] = int((mine[f"hard{i}"] != rel[f"hard{i}"]).sum())
            ok &= diffs[f"hard{i}_mismatches"] == 0
        row.update(diffs)
        row["drive_release_npz_sha_equals_local"] = sha_file(dd / "release.npz") == sha_file(ctx.udir(u) / "release.npz")
        ok &= row["drive_release_npz_sha_equals_local"]
        row["heads"] = {i: head_profile(h) for i, h in heads.items()}
        row["ok"] = bool(ok)
        ok_all &= ok
        rows.append(row)
    rep.add("9.1-restore-from-drive", sec, "PASS" if ok_all else "FAIL",
            "U, J-G and L-G of seed 1 (units from EVALUATION_LOCK score) restored from the drive copy alone: drive "
            "COMPLETE.json re-hashes and equals the local one; releases rebuilt with the own forward pass from the "
            "drive model.pt and head_i.joblib (83 permitted columns only) equal the local release.npz (r exact, c "
            "1e-9, p 1e-12, hard exact)", drive="<drive>", units=rows,
            backup_verification_present=True)


def info_summaries(ctx: Ctx, rep: Report, mysel, endp):
    sel = _sel_summary(mysel)
    rep.add("I.1-selection-outcome", "info", "INFO",
            "independent selection outcome per seed (L-R / L-O freeze, Stage C statuses, C*, J-G bound)", seeds=sel)
    lam = [c for c in rep.checks if c["id"] == "6.1-lambda-replay"]
    if lam and lam[0].get("max_lambda_over_base_weight"):
        r = lam[0]["max_lambda_over_base_weight"]
        main = {k: v for k, v in r.items() if "capped" not in k}
        rep.add("I.2-multiplier-magnitude", "info", "INFO",
                "largest replayed lambda_i relative to beta/3 per guarded run (main grid and capped ablation)",
                max_over_main_runs=max(main.values()) if main else None, runs=r)
    if endp is not None:
        prim = endp["primary"]
        rep.add("I.3-primary-decisions", "info", "INFO", "replayed primary decisions (point, lower, upper)",
                slots={k: {x: v.get(x) for x in ("point", "lower", "upper", "decision")} for k, v in prim.items()})


# ================================================================================================ 8. integrity
def verify_unit_dir(d: Path):
    c = d / "COMPLETE.json"
    if not c.exists():
        return "INCOMPLETE", [], []
    man = jload(c)
    bad, extra = [], []
    files = man.get("files", {})
    for f, h in files.items():
        p = d / f
        if not p.exists() or sha_file(p) != h:
            bad.append(f)
    listed = set(files) | {"COMPLETE.json"}
    for p in d.rglob("*"):
        if p.is_file() and str(p.relative_to(d)) not in listed:
            extra.append(str(p.relative_to(d)))
    return ("OK" if not bad else "BAD"), bad, extra


def git(ctx: Ctx, *args, check=False):
    try:
        r = subprocess.run(["git", "-C", str(ctx.wt), *args], capture_output=True, text=True, timeout=60)
    except Exception as e:  # noqa: BLE001
        return None, str(e)
    if r.returncode != 0:
        return None, r.stderr.strip()
    return r.stdout, None


def lock_push_record(ctx: Ctx, rel_path):
    """Commit (and local remote-tracking reflog) times of the latest commit touching rel_path; tree equality."""
    out, err = git(ctx, "log", "--format=%H|%cI", "--", rel_path)
    if out is None or not out.strip():
        return {"committed": False, "error": err}
    lines = [x.split("|") for x in out.strip().splitlines()]
    latest, ltime = lines[0]
    first, ftime = lines[-1]
    blob, _ = git(ctx, "rev-parse", f"{latest}:{rel_path}")
    wt_blob, _ = git(ctx, "hash-object", str(ctx.wt / rel_path))
    contains, _ = git(ctx, "branch", "-r", "--contains", latest)
    remote = bool(contains and any(x.strip() == f"origin/{BRANCH}" for x in contains.splitlines()))
    push_time = None
    rl, _ = git(ctx, "reflog", "show", "--date=iso-strict", "--format=%H %gd", f"refs/remotes/origin/{BRANCH}")
    if rl:
        cands = []
        for line in rl.strip().splitlines():
            sha, gd = line.split(" ", 1)
            m = re.search(r"@\{(.+)\}", gd)
            if not m:
                continue
            anc, _ = git(ctx, "merge-base", "--is-ancestor", latest, sha)
            if anc is not None:
                cands.append(m.group(1))
        if cands:
            push_time = min(cands, key=lambda s: dt.datetime.fromisoformat(s))
    return {"committed": True, "latest_commit": latest, "latest_commit_time": ltime, "first_commit": first,
            "first_commit_time": ftime, "working_tree_equals_commit": (blob or "").strip() == (wt_blob or "").strip(),
            "in_origin_branch": remote, "first_remote_tracking_time_containing_commit": push_time}


def earliest_unit_time(ctx: Ctx, prefix):
    times = []
    for n in ctx.unit_names(prefix):
        d = ctx.udir(n)
        for p in d.rglob("*"):
            if p.is_file():
                times.append(p.stat().st_mtime)
    log = ctx.priv_run / "ACTIVITY_LOG.jsonl"
    ev = None
    if log.exists():
        for line in log.read_text().splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            u = str(e.get("unit", ""))
            if u.startswith(prefix):
                t = dt.datetime.strptime(e["at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).timestamp()
                ev = t if ev is None else min(ev, t)
    if ev is not None:
        times.append(ev)
    return min(times) if times else None


def iso(ts):
    return None if ts is None else dt.datetime.fromtimestamp(ts, dt.timezone.utc).isoformat()


def lock_before(ctx: Ctx, rel_path, prefix):
    pr = lock_push_record(ctx, rel_path)
    first = earliest_unit_time(ctx, prefix)
    res = {"lock": rel_path, "units": prefix, **pr, "earliest_unit_time": iso(first)}
    if first is None:
        return "PENDING", res
    if not pr.get("committed"):
        return "FAIL", res
    ct = dt.datetime.fromisoformat(pr["latest_commit_time"]).timestamp()
    ft = dt.datetime.fromisoformat(pr["first_commit_time"]).timestamp()
    pt = pr.get("first_remote_tracking_time_containing_commit")
    ptt = dt.datetime.fromisoformat(pt).timestamp() if pt else None
    res["latest_commit_before_units"] = ct < first
    res["first_commit_before_units"] = ft < first
    res["push_observed_before_units"] = None if ptt is None else ptt < first
    ok = res["latest_commit_before_units"] and pr["in_origin_branch"] and pr["working_tree_equals_commit"]
    if ptt is not None:
        ok &= ptt < first
    return ("PASS" if ok else "FAIL"), res


def check_integrity(ctx: Ctx, rep: Report, lock):
    sec = "8-integrity"
    names = ctx.unit_names()
    stats = {"OK": 0, "BAD": 0, "INCOMPLETE": 0}
    bad, extras, quarantined = [], [], []
    for n in names:
        st, b, ex = verify_unit_dir(ctx.udir(n))
        if ".quarantined" in n:
            quarantined.append({"unit": n, "status": st, "bad": b})
            continue
        stats[st] += 1
        if st == "BAD":
            bad.append({"unit": n, "files": b})
        if ex:
            extras.append({"unit": n, "unlisted": ex})
    if not names:
        rep.add("8.1-complete-json", sec, "PENDING", "no private unit yet")
    else:
        rep.add("8.1-complete-json", sec, "FAIL" if bad else "PASS",
                "every unit's COMPLETE.json lists files whose sha256 re-hash matches (retained quarantined receipts "
                "reported separately); unlisted files reported", counts=stats, bad=bad, unlisted=extras,
                quarantined_receipts=quarantined)
    # parity receipts
    par = [ctx.rec(n) for n in names if n.startswith(("parity__", "tc__")) and ".quarantined" not in n and ctx.complete(n)]
    if par:
        failing = [r.get("unit", r.get("arm")) for r in par if r is not None and r.get("pass") is False]
        rep.add("8.2-parity-receipts", sec, "FAIL" if failing else "PASS",
                "registered beta = 0 / environment parity receipts (parity__*, tc__*) all record pass = true",
                n=len(par), failing=failing)
    # evaluation lock unit hashes
    if lock is None:
        rep.add("8.3-evaluation-lock-hashes", sec, "PENDING", "EVALUATION_LOCK.json not written yet")
    else:
        maps = [v for p, v in _find_key(lock, "unit_file_sha256")]
        n_files, badh, missing = 0, [], []
        for m in maps:
            for unit, files in (m or {}).items():
                for f, h in (files or {}).items():
                    n_files += 1
                    p = ctx.udir(unit) / f
                    if not p.exists():
                        missing.append(f"{unit}/{f}")
                    elif sha_file(p) != h:
                        badh.append(f"{unit}/{f}")
        st = "PENDING" if not maps else ("FAIL" if (badh or missing) else "PASS")
        rep.add("8.3-evaluation-lock-hashes", sec, st,
                "every file hash in EVALUATION_LOCK.json unit_file_sha256 equals the file on disk",
                files=n_files, mismatched=badh, missing=missing)
    # code lock + amendments
    locks = []
    cl = ctx.res / "CODE_LOCK.json"
    if cl.exists():
        locks.append(("CODE_LOCK.json", jload(cl)))
    for p in sorted(ctx.res.glob("CODE_LOCK_A*.json"), key=lambda q: int(re.sub(r"\D", "", q.stem) or 0)):
        locks.append((p.name, jload(p)))
    if not locks:
        rep.add("8.4-code-lock", sec, "PENDING", "no CODE_LOCK.json")
    else:
        eff = {}
        for lname, L in locks:
            for key, base in (("code_files", ctx.wt), ("files", ctx.wt), ("documents_sha256", ctx.res)):
                m = L.get(key)
                if isinstance(m, dict):
                    for f, h in m.items():
                        if isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h):
                            eff[(str(base), f)] = (h, lname)
        rows, bad_ = [], []
        for (base, f), (h, lname) in sorted(eff.items()):
            p = Path(base) / f
            got = sha_file(p) if p.exists() else None
            r = {"file": f, "under": "worktree" if base == str(ctx.wt) else RES_REL, "locked_by": lname,
                 "match": got == h}
            if got != h:
                r["tree_sha256"] = got
                r["locked_sha256"] = h
                bad_.append(r)
            rows.append(r)
        later = [lk for _, L in locks for lk in (L.get("later_locked") or {})]
        unlocked_later = [f for f in later if (ctx.wt / f).exists() and (str(ctx.wt), f) not in eff]
        commits = {}
        for lname, _ in locks:
            pr = lock_push_record(ctx, f"{RES_REL}/{lname}") if ctx.git else {"committed": None}
            commits[lname] = pr
        # a locked JSON document edited later: classify additive closeout edits (old keys and values kept except
        # the free-text 'status'; new keys only added) against the version committed with its lock
        hard = []
        for r in bad_:
            r["classification"] = "CHANGED"
            if r["under"] == RES_REL and r["file"].endswith(".json") and ctx.git:
                lc = commits.get(r["locked_by"], {}).get("first_commit")
                old_txt, _ = git(ctx, "show", f"{lc}:{RES_REL}/{r['file']}") if lc else (None, None)
                if old_txt is not None and hashlib.sha256(old_txt.encode()).hexdigest() == r["locked_sha256"]:
                    old_j = json.loads(old_txt)
                    new_j = jload(ctx.res / r["file"])
                    changed = [k for k in old_j if k != "status" and old_j.get(k) != new_j.get(k)]
                    added = [k for k in new_j if k not in old_j]
                    r["locked_version_in_git"] = True
                    r["changed_keys"] = changed + (["status"] if old_j.get("status") != new_j.get("status") else [])
                    r["added_keys"] = added
                    r["old_status"], r["new_status"] = old_j.get("status"), new_j.get("status")
                    if not changed:
                        r["classification"] = "ADDITIVE_EDIT_AFTER_LOCK (locked content preserved; no amendment)"
                        continue
            hard.append(r)
        st = "FAIL" if hard else ("WARN" if bad_ else "PASS")
        rep.add("8.4-code-lock", sec, st,
                "files hashed in CODE_LOCK.json and its dated amendments (latest definition wins) equal the working "
                "tree; later-locked files not yet covered by an amendment are listed. A locked JSON document whose "
                "committed locked version is intact in git and whose later edit only adds keys / changes the free-text "
                "status is a WARN (locked content preserved, but edited without an amendment); any other change FAILs",
                locks=[n for n, _ in locks],
                n_files=len(rows), mismatches=bad_, later_locked_files_without_amendment=unlocked_later,
                lock_commits=commits)
    if not ctx.git:
        rep.add("8.5-lock-before-assessment", sec, "SKIP", "git checks disabled (synthetic run)")
        return
    st, res = lock_before(ctx, f"{RES_REL}/EVALUATION_LOCK.json", "outer__")
    if not (ctx.res / "EVALUATION_LOCK.json").exists() and st != "PENDING":
        st = "FAIL"
    rep.add("8.5-lock-before-assessment", sec, st,
            "EVALUATION_LOCK.json committed (working tree identical), contained in origin/" + BRANCH +
            " and committed / observed on the remote-tracking ref before the first outer__ unit file or activity-log "
            "event", **res)
    st, res = lock_before(ctx, f"{RES_REL}/CODE_LOCK.json", "ck__")
    rep.add("8.6-code-lock-before-fits", sec, st,
            "CODE_LOCK.json committed and pushed before the first nonzero-fit checkpoint unit (ck__)", **res)


def _find_key(o, key, path=()):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == key:
                yield path + (k,), v
            yield from _find_key(v, key, path + (str(k),))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _find_key(v, key, path + (str(i),))


# ================================================================================================ driver
def run_all(ctx: Ctx, rep: Report, drive=None):
    lock = ctx.lock()
    steps = [("1-roles", lambda: check_roles(ctx, rep)),
             ("2-release", lambda: check_release(ctx, rep, lock))]
    mysel = {k: {} for k in SEEDS}
    problems = {"missing_inputs": [], "utility_record_mismatches": []}
    endp = None
    for sec, fn in steps:
        _guarded(rep, sec, fn)

    def sel():
        nonlocal mysel, problems
        mysel, problems = replay_selection(ctx)
        check_selection(ctx, rep, lock, mysel, problems)
    _guarded(rep, "3-selection", sel)
    _guarded(rep, "3-selection", lambda: check_spot(ctx, rep, mysel, lock))
    _guarded(rep, "3-selection", lambda: check_selection_tables(ctx, rep, mysel))

    def ep():
        nonlocal endp
        endp = check_endpoints(ctx, rep, lock, mysel)
    _guarded(rep, "4-endpoints", ep)
    _guarded(rep, "5-conjunction", lambda: check_claims(ctx, rep, endp, mysel))
    _guarded(rep, "6-multipliers", lambda: check_lambda(ctx, rep))
    _guarded(rep, "6-multipliers", lambda: check_weight_csv(ctx, rep))
    _guarded(rep, "7-critic-gap", lambda: check_critic_gap(ctx, rep, lock))
    _guarded(rep, "7-critic-gap", lambda: check_tracking_csv(ctx, rep, mysel))
    _guarded(rep, "8-integrity", lambda: check_integrity(ctx, rep, lock))
    _guarded(rep, "8-integrity", lambda: check_stage_b_freeze(ctx, rep, mysel))
    _guarded(rep, "9-restore", lambda: check_restore(ctx, rep, lock, drive))
    _guarded(rep, "info", lambda: info_summaries(ctx, rep, mysel, endp))
    return {"selection_replay": _sel_summary(mysel), "endpoints": endp is not None}


def _sel_summary(mysel):
    out = {}
    for k, s in mysel.items():
        d = {}
        if isinstance(s.get("stage_B"), dict):
            d["L-R"] = {x: s["stage_B"]["L-R"].get(x) for x in ("status", "unit", "beta", "epoch")}
            d["L-O"] = {x: s["stage_B"]["L-O"].get(x) for x in ("status", "unit", "beta", "epoch")}
        else:
            d["stage_B"] = s.get("stage_B")
        if isinstance(s.get("stage_C"), dict):
            c = s["stage_C"]
            d["U"] = s.get("U_unit")
            d["e_LR"] = s.get("e_LR")
            d["controls"] = {a: {"status": c["controls"][a]["status"], "unit": c["controls"][a].get("unit")}
                             for a in TRAINED_CONTROLS}
            d["single_feasible"] = {lab: c["single"][lab].get("feasible") for lab in SINGLE_CONTROLS}
            d["C*"] = c["C*"] and {"label": c["C*"]["label"], "unit": c["C*"]["unit"]}
            d["J-G"] = {"status": c["J-G"]["status"], "unit": c["J-G"].get("unit"), "bound": c["J-G"]["bound"]}
        else:
            d["stage_C"] = s.get("stage_C")
        out[f"s{k}"] = d
    return out


def _guarded(rep, sec, fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        rep.add(f"{sec}-ERROR", sec, "FAIL", f"{type(e).__name__}: {e}",
                trace=traceback.format_exc().splitlines()[-8:])


# ================================================================================================ self-tests
def _sk_auc(y, s, w):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s, sample_weight=w))


def selftest_primitives(st: Report):
    sec = "selftest"
    rng = np.random.default_rng(7)
    worst = 0.0
    for case in range(25):
        n = int(rng.integers(20, 300))
        y = rng.random(n) < rng.uniform(0.2, 0.8)
        y[0], y[1] = True, False
        s = np.round(rng.normal(size=n) + y * rng.uniform(0, 2), int(rng.integers(0, 3)))   # ties
        W = rng.integers(0, 4, size=(6, n)).astype(float)
        W[:, 0] = 1
        W[:, 1] = 1
        mine = wauc_matrix(s, y, W)
        ref = np.array([_sk_auc(y, s, w) for w in W])
        worst = max(worst, float(np.max(np.abs(mine - ref))))
    st.add("T1-weighted-auc", sec, "PASS" if worst < 1e-12 else "FAIL",
           "weighted AUC (ties 1/2) equals sklearn roc_auc_score(sample_weight) on 25 random cases with ties", max_abs=worst)
    G, B = 517, 600
    p = np.full(G, 1 / G)
    r1 = np.random.default_rng(20261004)
    seq = np.vstack([r1.multinomial(G, p) for _ in range(B)])
    ok = np.array_equal(seq, boot_counts(G, B, 20261004)) and (seq.sum(1) == G).all()
    st.add("T2-bootstrap-draws", sec, "PASS" if ok else "FAIL",
           "chunked (250) multinomial draws equal one-replicate-at-a-time sequential draws; each replicate sums to G")
    # summarize / decisions
    arr = np.r_[0.05, 0.05 + np.r_[-1, 1] * 0.01]
    s1 = summarize(arr, 2.0, 0.02, "lower>")
    s2 = summarize(np.r_[0.0, np.r_[-1, 1] * 0.001], 2.0, 0.01, "upper<")
    s3 = summarize(np.r_[0.0, np.r_[-1, 1] * 0.001], 2.0, 0.0, "two_sided")
    s4 = summarize(np.r_[0.1, np.r_[-1, 1] * 0.001], 2.0, 0.0, "two_sided")
    s5 = summarize(np.r_[-0.1, np.r_[-1, 1] * 0.001], 2.0, 0.0, "two_sided")
    se = float(np.std(np.r_[-1, 1] * 0.01, ddof=1))
    ok = (s1["decision"] == "PASS" and abs(s1["lower"] - (0.05 - 2 * se)) < 1e-15 and s2["decision"] == "PASS"
          and s3["decision"] == "NOT_RESOLVED" and s4["decision"] == "ABOVE" and s5["decision"] == "BELOW")
    st.add("T3-decisions", sec, "PASS" if ok else "FAIL",
           "lower> passes iff lower > target; upper< passes iff upper < target; two_sided gives ABOVE / BELOW / "
           "NOT_RESOLVED; SE uses ddof = 1")
    # selection logic
    u_good = {0: (900, 700, 1000), 1: (500, 300, 1000)}
    u_bad = {0: (700, 700, 1000), 1: (300, 300, 1000)}
    u_g2edge = {0: (860, 700, 1000), 1: (500, 300, 1000)}   # 5*(160) = 800 >= 4*(200) = 800 -> passes G2 exactly
    u_g1fail = {0: (889, 700, 1000), 1: (500, 300, 1000)}   # 100*(-11) = -1100 < -1000 -> fails G1

    def C(u, b, e, v1, v2, pair, util=u_good, alias=False):
        return Cand(u, "x", b, e, {"v1": v1, "v2": v2, "pair": pair}, util, alias)
    ub = C("UB", 0.0, 20, 0.7, 0.7, 0.75, alias=True)
    r = select_stage_B([C("a", 0.1, 10, .6, .58, .7), C("b", 0.3, 5, .6, .59, .7), C("c", 0.1, 15, .6, .58, .7),
                        C("d", 0.03, 20, .5, .5, .5, util=u_bad)], ub)
    t1 = r["status"] == "NOMINEE" and r["unit"] == "a"
    r = select_stage_B([C("a", 0.1, 10, .72, .7, .7), C("b", 0.3, 5, .7, .7, .7)], ub)
    t2 = r["status"] == "TASK_ONLY_ALIAS" and r["unit"] == "UB"         # tie on worse+mean -> beta 0 alias
    r = select_stage_B([C("a", 0.1, 10, .5, .5, .5)], C("UB", 0.0, 20, .7, .7, .7, util=u_bad, alias=True))
    t3 = r["status"] == "NO_VALID_REFERENCE"
    t4 = gates(u_g2edge, u_good)[0] is False and gates(u_g2edge, u_good)[1][0] == (False, True, True)
    t4 &= gates(u_g1fail, u_good)[0] is False
    t4 &= gates({0: (890, 700, 1000), 1: (500, 300, 1000)}, u_good)[0] is True   # 100*(-10) = -1000 >= -1000
    lr = (0.60, 0.58)
    trained = {"L-G": [C("lg1", 0.1, 5, .61, .55, .55), C("lg2", 0.03, 10, .60, .58, .60), C("lg3", 0.3, 20, .59, .57, .60)],
               "J-R": [C("jr1", 0.03, 5, .59, .59, .5)], "J-O": [C("jo1", 0.03, 5, .59, .58, .64)]}
    singles = {"U": C("U", 0, 0, .7, .7, .75), "L-R": C("LR", 0.1, 10, .60, .58, .63), "L-O": None,
               "E": C("E", 0, 0, .55, .55, .61), "F": C("F", 0, 0, .58, .57, .65), "F0": C("F0", 0, 0, .5, .5, .5, util=u_bad)}
    jg = [C("jg1", 0.1, 15, .605, .5, .5), C("jg2", 0.3, 10, .58, .56, .52), C("jg3", 0.03, 5, .58, .56, .57)]
    out = select_stage_C(trained, singles, u_good, lr, jg)
    t5 = (out["controls"]["L-G"]["unit"] == "lg2" and out["controls"]["J-R"]["status"] == "NO_FEASIBLE_NOMINEE"
          and out["controls"]["J-O"]["unit"] == "jo1" and out["C*"]["label"] == "L-G" and out["J-G"]["unit"] == "jg2"
          and not out["single"]["U"]["feasible"] and not out["single"]["F0"]["feasible"] and out["single"]["E"]["feasible"])
    # C* tie order (E before F) and J-G bound from C* when L-G is infeasible
    trained2 = {"L-G": [C("lg", 0.1, 5, .5, .5, .5, util=u_bad)], "J-R": [C("jr", 0.03, 5, .69, .69, .71)],
                "J-O": [C("jo", 0.03, 5, .69, .69, .73)]}
    singles2 = {"U": C("U", 0, 0, .71, .71, .8), "L-R": C("LR", 0, 20, .70, .70, .74), "L-O": C("LO", .3, 20, .65, .66, .70),
                "E": C("E", 0, 0, .65, .66, .68), "F": C("F", 0, 0, .66, .65, .68), "F0": None}
    jg2 = [C("a", 0.03, 5, .66, .60, .55), C("b", 0.03, 10, .64, .65, .62), C("c", 0.1, 5, .64, .65, .62)]
    out2 = select_stage_C(trained2, singles2, u_good, (0.70, 0.70), jg2)
    t6 = (out2["controls"]["L-G"]["status"] == "NO_FEASIBLE_NOMINEE" and out2["C*"]["label"] == "E"
          and out2["J-G"]["unit"] == "b" and out2["J-G"]["bound"] == [0.65, 0.66])
    out3 = select_stage_C(trained2, singles2, u_good, (0.70, 0.70), [C("z", 0.1, 5, .9, .9, .1)])
    t7 = out3["J-G"]["status"] == "NO_FEASIBLE_NOMINEE"
    st.add("T4-selection-logic", sec, "PASS" if all([t1, t2, t3, t4, t5, t6, t7]) else "FAIL",
           "Stage B tie-breaks (worse local, mean, beta, epoch), alias on tie -> TASK_ONLY_ALIAS, U-B failing G3 -> "
           "NO_VALID_REFERENCE; exact-integer gate edges (G2 equality passes, G1 = -0.01 passes); trained-control guard; "
           "C* order tie-break; J-G min-guard via C* when L-G infeasible; NO_FEASIBLE_NOMINEE",
           parts={"stageB_tiebreak": t1, "alias": t2, "no_valid_ref": t3, "gate_edges": t4, "stageC": t5,
                  "cstar_order_and_bound": t6, "no_feasible": t7})
    # lambda replay
    beta, c = 0.1, [0.20, 0.30]
    Rs = [[0.25, 0.28], [0.27, 0.35], [0.10, 0.40], [0.50, 0.29], [0.22, 0.31], [0.9, 0.9]]
    lam, trace = [0.0, 0.0], []
    for e, R in zip(REFIT_EPOCHS, Rs):
        if e < 20:
            lam = [min(max(lam[i] + beta * (R[i] - c[i]), 0), 0.3) for i in (0, 1)]
        trace.append({"epoch": e, "R_calib": R, "c": c, "lambda": list(lam),
                      "effective_weights": {"v1": beta / 3 + lam[0], "v2": beta / 3 + lam[1], "pair": beta / 3}})
    _, w0, _ = replay_lambda(trace, beta, c, "joint")
    bad_t = copy.deepcopy(trace)
    bad_t[3]["lambda"][0] += 1e-6
    _, w1, _ = replay_lambda(bad_t, beta, c, "joint")
    _, w2, _ = replay_lambda(trace, beta, c, "local")
    _, w3, _ = replay_lambda(trace, beta, c, "joint", update_last=True)
    st.add("T5-lambda-replay", sec, "PASS" if (w0 == 0 and w1 > 1e-7 and w2 > 0.01 and w3 > 0) else "FAIL",
           "replay reproduces a rule-generated trace exactly; detects a 1e-6 lambda perturbation, a local/joint weight "
           "mix-up and the update-at-epoch-20 convention", exact=w0, perturbed=w1, wrong_base=w2, alt_convention=w3)
    # critic CE vs an independent numpy computation
    torch.manual_seed(3)
    st_, snap = _syn_snapshot(seed=3, n_ref=300)
    Xn = np.random.default_rng(4).normal(size=(120, 83)).astype(np.float32)
    Sn = np.random.default_rng(5).integers(0, 2, 120)
    mine, ok_shape = snapshot_ce(snap, torch.from_numpy(Xn), torch.from_numpy(Sn))
    ref = _np_snapshot_ce(snap, Xn, Sn)
    d = max(abs(mine[v][k] - ref[v][k]) for v in mine for k in mine[v])
    mod = _mod_snapshot_ce(snap, Xn, Sn)
    d_mod = max(abs(mine[v][k] - mod[v][k]) for v in mine for k in mine[v])
    wrong = _np_snapshot_ce(snap, Xn, Sn, centre=False)
    d_wrong = max(abs(wrong[v][k] - ref[v][k]) for v in ("v1", "v2", "pair") for k in ref[v])
    st.add("T6-critic-ce", sec, "PASS" if (d < 1e-3 and d_mod < 1e-6 and d_wrong > 1e-2 and ok_shape) else "FAIL",
           "critic CE from the torch re-implementation equals an independent float64 numpy computation (within float32 "
           "null-direction amplification, 1e-3) and a torch nn.Module path (1e-6); an uncentred-logit view convention "
           "differs by > 1e-2 (power of the comparison)", max_abs_numpy=d, max_abs_module=d_mod,
           wrong_convention_diff=d_wrong)
    # claims logic
    prim = {f"P{i:02d}": {"decision": "PASS"} for i in range(1, 19)}

    def ms(jg="NOMINEE", lg="NOMINEE", cstar=True, lr="NOMINEE"):
        return {k: {"stage_B": {"L-R": {"status": lr}},
                    "stage_C": {"J-G": {"status": jg}, "controls": {"L-G": {"status": lg}},
                                "C*": {"label": "L-G"} if cstar else None}} for k in SEEDS}
    a = claims(prim, ms())
    b = claims({**prim, "P02": {"decision": "FAIL"}}, ms())
    c_ = claims(prim, ms(lg="NO_FEASIBLE_NOMINEE"))
    d_ = claims(prim, ms(cstar=False))
    e_ = claims(prim, ms(lr="NO_VALID_REFERENCE"))
    f_ = claims(prim, ms(lr="TASK_ONLY_ALIAS"))
    mm = ms()
    mm.pop(2)
    g_ = claims(prim, mm)
    ok = (a["A"]["decision"] == a["B"]["decision"] == "ESTABLISHED" and b["A"]["decision"] == "NOT_ESTABLISHED"
          and b["B"]["decision"] == "ESTABLISHED" and c_["A"]["decision"] == "NOT_ESTABLISHED"
          and c_["B"]["decision"] == "ESTABLISHED" and d_["B"]["decision"] == "NOT_ESTABLISHED"
          and d_["A"]["decision"] == "ESTABLISHED" and e_["A"]["decision"] == e_["B"]["decision"] == "NOT_ESTABLISHED"
          and f_["A"]["decision"] == "ESTABLISHED" and g_["A"]["decision"] == g_["B"]["decision"] == "NOT_ESTABLISHED")
    st.add("T7-conjunction-logic", sec, "PASS" if ok else "FAIL",
           "one failed clause, an infeasible L-G (claim A), a missing C* (claim B), an invalid reference or a missing "
           "seed each make the claim NOT_ESTABLISHED; a task-only L-R reference alone does not")


def _syn_state(seed, zero_last=False):
    g = torch.Generator().manual_seed(seed)
    st = {}
    for i, K in enumerate(KS):
        for j, (o, n_in) in zip((0, 2, 4), ((64, 83), (64, 64), (16, 64))):
            st[f"enc.{i}.{j}.weight"] = torch.randn(o, n_in, generator=g) / math.sqrt(n_in) * 1.5
            st[f"enc.{i}.{j}.bias"] = torch.randn(o, generator=g) * 0.1
        st[f"enc.{i}.0.weight"][:, :9] *= 8.0
        if zero_last:
            st[f"enc.{i}.4.weight"] = torch.zeros(16, 64)
        st[f"head.{i}.weight"] = torch.randn(K, 16, generator=g) * 0.3
        st[f"head.{i}.bias"] = torch.randn(K, generator=g) * 0.1
    return st


def _syn_critic(kind, dv, g):
    if kind == "A":
        dims = [(32, dv), (2, 32)]
    else:
        dims = [(64, dv), (64, 64), (2, 64)]
    sd = {}
    for j, (o, n_in) in enumerate(dims):
        sd[f"{2 * j}.weight"] = torch.randn(o, n_in, generator=g) / math.sqrt(n_in)
        sd[f"{2 * j}.bias"] = torch.randn(o, generator=g) * 0.1
    return sd


def _syn_snapshot(seed, n_ref=300, state=None, Xref=None):
    g = torch.Generator().manual_seed(seed + 1000)
    st = state if state is not None else _syn_state(seed)
    head = {i: (torch.randn(K, 16, generator=g) * 0.4, torch.randn(K, generator=g) * 0.1) for i, K in enumerate(KS)}
    if Xref is None:
        Xref = torch.randn(n_ref, 83, generator=g)
    snap = {"model": st, "critic_head": head, "critics": {}, "transforms": {}, "lam": [0.0, 0.0]}
    V = snapshot_views(snap, Xref)
    for v in ("v1", "v2", "pair"):
        Vd = V[v].double()
        mu = Vd.mean(0)
        C = torch.cov((Vd - mu).T)
        ev, U = torch.linalg.eigh(C)
        Wt = (U / torch.sqrt(torch.clamp(ev, min=float(ev.max()) * 1e-8))) @ U.T
        snap["transforms"][v] = {"kind": "floored", "mu": mu, "W": Wt, "ev": ev}
        snap["critics"][v] = {k: _syn_critic(k, Vd.shape[1], g) for k in ("A", "B")}
    return st, snap


def _mod_snapshot_ce(snap, X, S):
    """Independent torch nn.Module path (float32, the runner-like arithmetic) used by the synthetic builder."""
    import torch.nn as nn
    Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    St = torch.from_numpy(np.asarray(S).astype(np.int64))
    vs = {}
    with torch.no_grad():
        for i in (0, 1):
            enc = nn.Sequential(nn.Linear(83, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 16))
            enc.load_state_dict({k[len(f"enc.{i}."):]: v for k, v in snap["model"].items() if k.startswith(f"enc.{i}.")})
            r = enc(Xt)
            W, b = snap["critic_head"][i]
            hd = nn.Linear(16, W.shape[0])
            hd.load_state_dict({"weight": W.float(), "bias": b.float()})
            lg = hd(r)
            vs[i] = torch.cat([r, lg - lg.mean(1, keepdim=True)], 1)
        V = {"v1": vs[0], "v2": vs[1], "pair": torch.cat([vs[0], vs[1]], 1)}
        out = {}
        for v in V:
            ts = snap["transforms"][v]
            Z = (V[v] - ts["mu"].float()) @ ts["W"].float()
            out[v] = {}
            for kind, sd in snap["critics"][v].items():
                dv = Z.shape[1]
                m = (nn.Sequential(nn.Linear(dv, 32), nn.ReLU(), nn.Linear(32, 2)) if kind == "A" else
                     nn.Sequential(nn.Linear(dv, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2)))
                m.load_state_dict(sd)
                out[v][kind] = float(nn.functional.cross_entropy(m(Z), St))
    return out


def _np_snapshot_ce(snap, X, S, centre=True):
    """Independent numpy path (float64) for the self-test."""
    def relu(a):
        return np.maximum(a, 0)
    st = {k: v.double().numpy() for k, v in snap["model"].items()}
    vs = {}
    for i in (0, 1):
        h = X.astype(np.float64)
        h = relu(h @ st[f"enc.{i}.0.weight"].T + st[f"enc.{i}.0.bias"])
        h = relu(h @ st[f"enc.{i}.2.weight"].T + st[f"enc.{i}.2.bias"])
        r = h @ st[f"enc.{i}.4.weight"].T + st[f"enc.{i}.4.bias"]
        W, b = (t.double().numpy() for t in snap["critic_head"][i])
        lg = r @ W.T + b
        vs[i] = np.hstack([r, lg - lg.mean(1, keepdims=True) if centre else lg])
    V = {"v1": vs[0], "v2": vs[1], "pair": np.hstack([vs[0], vs[1]])}
    out = {}
    for v in V:
        ts = snap["transforms"][v]
        Z = (V[v] - ts["mu"].numpy()) @ ts["W"].numpy()
        out[v] = {}
        for kind, sd in snap["critics"][v].items():
            sdn = {k: t.double().numpy() for k, t in sd.items()}
            idx = sorted({int(k.split(".")[0]) for k in sdn})
            h = Z
            for j, i in enumerate(idx):
                h = h @ sdn[f"{i}.weight"].T + sdn[f"{i}.bias"]
                if j < len(idx) - 1:
                    h = relu(h)
            m = h.max(1, keepdims=True)
            lse = m[:, 0] + np.log(np.exp(h - m).sum(1))
            out[v][kind] = float(np.mean(lse - h[np.arange(len(S)), S]))
    return out


def selftest_predecessor_fixture(st: Report):
    """Forward-pass fixture on a same-architecture predecessor model (jcv nn__s0__U): features, probabilities and hard
    decisions recomputed only for DEFENSE_FIT / AUDIT_FIT / INNER_SELECTION rows (no dropped or assessment row)."""
    sec = "selftest"
    d = PRED_UNITS / "nn__s0__U"
    if not (d / "model.pt").exists() or not SRC.exists():
        st.add("T8-forward-fixture", sec, "SKIP", "predecessor fixture not available")
        return
    D = Data(SRC)
    rows = np.sort(np.concatenate([D.idx["DEFENSE_FIT"], D.idx["AUDIT_FIT"], D.idx["INNER_SELECTION"]]))
    z = np.load(d / "release.npz", allow_pickle=False)
    pos = {int(r): j for j, r in enumerate(z["row_id"])}
    j = np.array([pos[int(r)] for r in D.row_id[rows]])
    mine, _ = deploy(d, D.X[rows])
    res = {}
    ok = True
    for i in (1, 2):
        res[f"max_abs_r{i}"] = float(np.max(np.abs(mine[f"r{i}"] - z[f"r{i}"][j])))
        res[f"max_abs_p{i}"] = float(np.max(np.abs(mine[f"p{i}"] - z[f"p{i}"][j])))
        res[f"max_abs_c{i}"] = float(np.max(np.abs(mine[f"c{i}"] - z[f"c{i}"][j])))
        res[f"hard{i}_mismatch"] = int((mine[f"hard{i}"] != z[f"hard{i}"][j]).sum())
        ok &= res[f"max_abs_r{i}"] == 0.0 and res[f"max_abs_p{i}"] < 1e-12 and res[f"hard{i}_mismatch"] == 0
    st.add("T8-forward-fixture", sec, "PASS" if ok else "FAIL",
           "own forward pass + head outputs reproduce a same-architecture predecessor release (nn__s0__U) on "
           "DEFENSE_FIT / AUDIT_FIT / INNER_SELECTION rows (r exact, p 1e-12, hard exact; c reported)",
           rows=int(len(rows)), **res)


# ------------------------------------------------------------------------------------------- synthetic world
class SynWorld:
    """A small synthetic study tree in the formats documented by the protocol / assess docstring, with designed
    selection outcomes (hand-derived, not computed by the verifier) and brute-force runner-side endpoints."""

    ROLE_N = {"defense_train": 700, "defense_val": 420, "attacker_fit": 360, "attacker_val": 300, "assessment": 60,
              "cert": 30, "excluded_exposure": 3, "excluded_dup": 4}

    def __init__(self, root: Path, B=19):
        self.root = Path(root)
        self.B = B
        self.units = self.root / "private" / "units"
        self.run = self.root / "private"
        self.res = self.root / "results"
        for p in (self.units, self.res):
            p.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(11)

    # -- source
    def make_source(self):
        rng = self.rng
        roles = np.concatenate([[r] * n for r, n in self.ROLE_N.items()])
        rng.shuffle(roles)
        n = len(roles)
        unit = np.arange(n, dtype=np.int64) + 5000
        for r in ("defense_train", "defense_val", "attacker_fit"):
            ix = np.flatnonzero(roles == r)
            for a, b in rng.choice(ix, size=(4, 2), replace=False):
                unit[b] = unit[a]
        ex = np.flatnonzero(roles == "excluded_exposure")
        dup = np.flatnonzero(roles == "excluded_dup")
        for a, b in zip(ex, dup):
            unit[b] = unit[a]
        X = rng.normal(size=(n, 83)).astype(np.float32)
        sex = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(size=n) * 0.8 > 0).astype(np.int64)
        y_inc = (X[:, 2] + rng.normal(size=n) * 0.3 > 0.4).astype(np.int64)
        y_occ = np.argmax(X[:, 3:9] * 1.5 + rng.normal(size=(n, 6)) * 0.3, 1).astype(np.int64)
        race = rng.choice([0, 1, 2, 4], size=n, p=[0.02, 0.5, 0.25, 0.23]).astype(np.int64)
        fn = np.array([f"feat_{j:02d}" for j in range(83)])
        self.src = self.root / "syn_source.npz"
        np.savez(self.src, row_id=np.arange(n, dtype=np.int64), unit=unit, role=roles, X=X, feature_names=fn, sex=sex,
                 race=race, y_income=y_inc, y_occupation_group=y_occ)
        self.src_sha = sha_file(self.src)
        # builder-side role derivation (loop form) + manifest
        two64 = 2 ** 64

        def uf(salt, u):
            return int(hashlib.sha256(f"{ROLE_SEED}|{salt}|{int(u)}".encode()).hexdigest()[:16], 16)
        new = {r: [] for r in NEW_ROLES + SUBROLES}
        for i in range(n):
            r = roles[i]
            if r in OLD_TO_NEW:
                new[OLD_TO_NEW[r]].append(i)
            elif r == "defense_val":
                new["HEAD_VALIDATION" if uf("dev", unit[i]) * 10 < 3 * two64 else "DEVELOPMENT_ASSESSMENT"].append(i)
            if r == "defense_train":
                x = uf("critic", unit[i])
                new["CRITIC_FIT" if x * 100 < 70 * two64 else ("CRITIC_VAL" if x * 100 < 85 * two64 else "CALIB")].append(i)
        self.rows = {r: np.array(v, dtype=np.int64) for r, v in new.items()}
        rid = np.arange(n, dtype=np.int64)

        def rec(ix):
            return {"rows": len(ix), "groups": len(np.unique(unit[ix])), "row_id_sha256": rowid_hash(rid[ix]),
                    "group_id_set_sha256": unit_set_hash(unit[ix])}
        man = {"source": {"sha256": self.src_sha},
               "roles": {r: rec(self.rows[r]) for r in NEW_ROLES},
               "defense_fit_subroles": {r: rec(self.rows[r]) for r in SUBROLES},
               "old_roles": {"per_role": {r: rec(np.flatnonzero(roles == r)) for r in self.ROLE_N}},
               "groups_spanning_dropped_pools_only": len(ex)}
        (self.res / "ROLE_MANIFEST.json").write_text(json.dumps(man))
        self.n, self.X, self.unit_arr = n, X, unit
        self.lab = {"sex": sex, "race": race, "y_income": y_inc, "y_occupation_group": y_occ}
        self.kept = np.sort(np.concatenate([self.rows[r] for r in NEW_ROLES]))
        shutil.copy(RES / "PRIMARY_FAMILY.json", self.res / "PRIMARY_FAMILY.json")

    # -- units
    def save_unit(self, name, files: dict, record: dict):
        d = self.units / name
        d.mkdir(parents=True, exist_ok=True)
        for fname, writer in files.items():
            writer(d / fname)
        (d / "record.json").write_text(json.dumps(record, default=float))
        hashes = {p.name: sha_file(p) for p in sorted(d.iterdir()) if p.is_file() and p.name != "COMPLETE.json"}
        (d / "COMPLETE.json").write_text(json.dumps({"id": name, "files": hashes}))

    def release_from(self, state, leace=False):
        import torch.nn as nn
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = nn.ModuleList([nn.Sequential(nn.Linear(83, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                                                        nn.Linear(64, 16)) for _ in KS])
                self.head = nn.ModuleList([nn.Linear(16, K) for K in KS])
        m = M()
        m.load_state_dict(state)
        Xk = torch.from_numpy(self.X[self.kept])
        out, heads = {"row_id": self.kept.copy()}, {}
        df_pos = np.searchsorted(self.kept, self.rows["DEFENSE_FIT"])
        maps = {}
        with torch.no_grad():
            for i, t in enumerate(("y_income", "y_occupation_group")):
                R = m.enc[i](Xk).double().numpy()
                if leace:   # builder-side LEACE projection (numpy): P = W^-1 U U^T W, U spans W Sigma_xz
                    Hf = R[df_pos]
                    Zf = np.eye(2)[self.lab["sex"][self.kept][df_pos]]
                    mx = Hf.mean(0)
                    Sxx = np.cov(Hf.T)
                    Sxz = (Hf - mx).T @ (Zf - Zf.mean(0)) / (len(Hf) - 1)
                    ev, Ue = np.linalg.eigh(Sxx)
                    Wm = (Ue / np.sqrt(ev)) @ Ue.T
                    Wi = (Ue * np.sqrt(ev)) @ Ue.T
                    u, sv, _ = np.linalg.svd(Wm @ Sxz, full_matrices=False)
                    u = u[:, sv > 1e-10 * sv.max()]
                    maps[i] = {"proj_left": Wi @ u, "proj_right": u.T @ Wm, "mean_x": mx}
                    R = R - ((R - mx) @ maps[i]["proj_right"].T) @ maps[i]["proj_left"].T
                h = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
                h.fit(R[df_pos], self.lab[t][self.kept][df_pos])
                zz = h.decision_function(R)
                if zz.ndim == 1:
                    zz = np.stack([np.zeros_like(zz), zz], 1)
                P = h.predict_proba(R)
                out.update({f"r{i + 1}": R, f"c{i + 1}": zz - zz.mean(1, keepdims=True), f"p{i + 1}": P,
                            f"hard{i + 1}": P.argmax(1)})
                heads[i] = h
        if leace:
            return out, heads, maps
        return out, heads

    def write_release_unit(self, name, kind, record, critics=None):
        rel, heads, state = self.cache_rel[kind]
        files = {"model.pt": lambda p: torch.save(state, p),
                 "release.npz": lambda p: np.savez_compressed(p, **rel),
                 "head_0.joblib": lambda p: joblib.dump(heads[0], p),
                 "head_1.joblib": lambda p: joblib.dump(heads[1], p)}
        if critics is not None:
            files["critics.pt"] = lambda p: torch.save(critics, p)
        self.save_unit(name, files, record)

    def util_floats(self, kind):
        rel = self.cache_rel[kind][0]
        isel = self.rows["INNER_SELECTION"]
        pos = np.searchsorted(self.kept, isel)
        df = self.rows["DEFENSE_FIT"]
        out = {}
        for t, lab in enumerate(("y_income", "y_occupation_group")):
            y = self.lab[lab]
            maj = int(np.argmax(np.bincount(y[df])))
            out[str(t)] = {"acc": float(np.mean(rel[f"hard{t + 1}"][pos] == y[isel])),
                           "const_acc": float(np.mean(y[isel] == maj))}
        return out

    def inner(self, unit, v1, v2, pair, kind):
        self.save_unit(f"inner__{unit}", {}, {"of": unit, "recovery": {"auc": {"v1": v1, "v2": v2, "pair": pair}},
                                               "utility": self.util_floats(kind)})

    def build(self):
        self.make_source()
        self.cache_rel = {}
        for k in SEEDS:
            for kind, zl in (("good", False), ("bad", True)):
                st_ = _syn_state(100 + k, zero_last=zl)
                rel, heads = self.release_from(st_)
                self.cache_rel[(k, kind)] = (rel, heads, st_)
        u = self.util_floats((0, "good"))
        assert u["0"]["acc"] - u["0"]["const_acc"] > 0.05 and u["1"]["acc"] - u["1"]["const_acc"] > 0.05, u
        self.design()
        self.write_units()
        self.write_runner_outputs()
        return self

    # designed inner AUCs (v1, v2, pair) and utility kinds; expected outcomes are hand-derived in self.expect
    def design(self):
        D = {}
        exp = {}
        # ---------------- seed 0
        d = {"tl20": (0.70, 0.70, 0.75)}
        for b in BETAS:
            for e in EPOCHS:
                d[("B", "L-R", b, e)] = ((0.66, 0.66, 0.70), "good")
                d[("B", "L-O", b, e)] = ((0.72, 0.71, 0.75), "good")
                d[("C", "L-G", b, e)] = ((0.59, 0.57, 0.62), "good")
                d[("C", "J-R", b, e)] = ((0.59, 0.59, 0.61), "good")
                d[("C", "J-O", b, e)] = ((0.59, 0.58, 0.64), "good")
                d[("C", "J-G", b, e)] = ((0.58, 0.56, 0.57), "good")
        d[("B", "L-R", 0.1, 10)] = ((0.60, 0.58, 0.63), "good")
        d[("B", "L-R", 0.3, 5)] = ((0.60, 0.59, 0.63), "good")
        d[("B", "L-R", 0.1, 15)] = ((0.60, 0.58, 0.63), "good")
        d[("B", "L-R", 0.03, 20)] = ((0.50, 0.50, 0.50), "bad")
        d[("C", "L-G", 0.1, 5)] = ((0.61, 0.55, 0.55), "good")
        d[("C", "L-G", 0.03, 10)] = ((0.60, 0.58, 0.60), "good")
        d[("C", "L-G", 0.3, 20)] = ((0.59, 0.57, 0.60), "good")
        d[("C", "J-G", 0.1, 15)] = ((0.605, 0.50, 0.50), "good")
        d[("C", "J-G", 0.3, 10)] = ((0.58, 0.56, 0.52), "good")
        d["U"] = (0.70, 0.70, 0.75)
        d["E"] = ((0.55, 0.55, 0.61), "good")
        d["F"] = ((0.58, 0.57, 0.65), "good")
        d["F0"] = ((0.50, 0.50, 0.50), "bad1")
        D[0] = d
        exp[0] = {"L-R": ("NOMINEE", ck_name("B", 0, "L-R", 0.1, 10)), "L-O": ("TASK_ONLY_ALIAS", "tl__s0__e20"),
                  "e_LR": 10, "L-G": ("NOMINEE", ck_name("C", 0, "L-G", 0.03, 10)), "J-R": ("NO_FEASIBLE_NOMINEE", None),
                  "J-O": ("NOMINEE", ck_name("C", 0, "J-O", 0.03, 5)), "C*": "L-G",
                  "J-G": ("NOMINEE", ck_name("C", 0, "J-G", 0.3, 10))}
        # ---------------- seed 1
        d = {"tl20": (0.70, 0.70, 0.74)}
        for b in BETAS:
            for e in EPOCHS:
                d[("B", "L-R", b, e)] = ((0.72, 0.71, 0.75), "good")
                d[("B", "L-O", b, e)] = ((0.72, 0.71, 0.75), "good")
                d[("C", "L-G", b, e)] = ((0.50, 0.50, 0.50), "bad")
                d[("C", "J-R", b, e)] = ((0.69, 0.69, 0.71), "good")
                d[("C", "J-O", b, e)] = ((0.69, 0.69, 0.73), "good")
                d[("C", "J-G", b, e)] = ((0.64, 0.65, 0.62), "good")
        d[("B", "L-O", 0.3, 20)] = ((0.65, 0.66, 0.70), "good")
        d[("C", "J-G", 0.03, 5)] = ((0.67, 0.60, 0.55), "good")   # passes vs L-R, fails the C* bound
        d["U"] = (0.71, 0.71, 0.80)
        d["E"] = ((0.65, 0.66, 0.69), "good")   # C* below does not depend on E's (utility-dependent) feasibility
        d["F"] = ((0.66, 0.65, 0.68), "good")
        d["F0"] = ((0.60, 0.60, 0.69), "good")
        D[1] = d
        exp[1] = {"L-R": ("TASK_ONLY_ALIAS", "tl__s1__e20"), "L-O": ("NOMINEE", ck_name("B", 1, "L-O", 0.3, 20)),
                  "e_LR": 20, "L-G": ("NO_FEASIBLE_NOMINEE", None), "J-R": ("NOMINEE", ck_name("C", 1, "J-R", 0.03, 5)),
                  "J-O": ("NOMINEE", ck_name("C", 1, "J-O", 0.03, 5)), "C*": "F",
                  "J-G": ("NOMINEE", ck_name("C", 1, "J-G", 0.03, 10))}
        # ---------------- seed 2
        d = {"tl20": (0.70, 0.70, 0.75)}
        for b in BETAS:
            for e in EPOCHS:
                d[("B", "L-R", b, e)] = ((0.66, 0.66, 0.70), "good")
                d[("B", "L-O", b, e)] = ((0.67, 0.66, 0.70), "good")
                d[("C", "L-G", b, e)] = ((0.61, 0.62, 0.63), "good")
                d[("C", "J-R", b, e)] = ((0.61, 0.62, 0.64), "good")
                d[("C", "J-O", b, e)] = ((0.63, 0.62, 0.60), "good")
                d[("C", "J-G", b, e)] = ((0.60, 0.61, 0.58), "good")
        d[("B", "L-R", 0.3, 15)] = ((0.62, 0.63, 0.66), "good")
        d[("B", "L-O", 0.1, 20)] = ((0.64, 0.64, 0.69), "good")
        d["U"] = (0.72, 0.72, 0.80)
        d["E"] = ((0.60, 0.61, 0.66), "good")
        d["F"] = ((0.60, 0.60, 0.67), "good")
        d["F0"] = ((0.60, 0.60, 0.69), "good")
        D[2] = d
        exp[2] = {"L-R": ("NOMINEE", ck_name("B", 2, "L-R", 0.3, 15)), "L-O": ("NOMINEE", ck_name("B", 2, "L-O", 0.1, 20)),
                  "e_LR": 15, "L-G": ("NOMINEE", ck_name("C", 2, "L-G", 0.03, 5)),
                  "J-R": ("NOMINEE", ck_name("C", 2, "J-R", 0.03, 5)), "J-O": ("NO_FEASIBLE_NOMINEE", None), "C*": "L-G",
                  "J-G": ("NOMINEE", ck_name("C", 2, "J-G", 0.03, 5))}
        self.D, self.expect = D, exp

    def write_units(self):
        self.fare_sel = {}
        for k in SEEDS:
            d = self.D[k]
            good, bad = (k, "good"), (k, "bad")
            e_lr = self.expect[k]["e_LR"]
            for e in range(5, 41, 5):
                n = f"tl__s{k}__e{e}"
                self.write_release_unit(n, good, {"arm": "TASK", "epoch": e})
                a = d["tl20"] if e == 20 else (d["U"] if e == 20 + e_lr else (0.71, 0.71, 0.76))
                self.inner(n, *a, good)
            for key, val in d.items():
                if not (isinstance(key, tuple) and len(key) == 4):
                    continue
                a, kind = val
                stg, arm, b, e = key
                n = ck_name(stg, k, arm, b, e)
                lam = None
                self.write_release_unit(n, (k, "good" if kind == "good" else "bad"),
                                        {"stage": stg, "arm": arm, "beta": b, "seed": k, "epoch": e, "lam": lam},
                                        critics={"lam": [0.0, 0.0]})
                self.inner(n, *a, (k, "good" if kind == "good" else "bad"))
            # LEACE arm: U's encoder + stored LEACE maps (leace_i/leace_map.npz) + heads refit on erased features
            a, kind = d["E"]
            st_good = self.cache_rel[good][2]
            rel_e, heads_e, maps_e = self.release_from(st_good, leace=True)
            self.cache_rel[(k, "leace")] = (rel_e, heads_e, st_good)
            files = {"model.pt": lambda q: torch.save(st_good, q),
                     "release.npz": lambda q: np.savez_compressed(q, **rel_e),
                     "head_0.joblib": lambda q: joblib.dump(heads_e[0], q),
                     "head_1.joblib": lambda q: joblib.dump(heads_e[1], q)}
            ud = self.units / f"lc__s{k}__E"
            for i in (0, 1):
                (ud / f"leace_{i}").mkdir(parents=True, exist_ok=True)
                np.savez(ud / f"leace_{i}" / "leace_map.npz", **maps_e[i])
            self.save_unit(f"lc__s{k}__E", files, {"arm": "E"})
            lhash = {f"leace_{i}/leace_map.npz": sha_file(ud / f"leace_{i}" / "leace_map.npz") for i in (0, 1)}
            cj = json.loads((ud / "COMPLETE.json").read_text())
            cj["files"].update(lhash)
            (ud / "COMPLETE.json").write_text(json.dumps(cj))
            self.inner(f"lc__s{k}__E", *a, (k, "leace"))
            # FARE per-purpose configs + twins, and pair records
            futil = self.util_floats(good)
            bad_u = {t: {"acc": futil[t]["const_acc"], "const_acc": futil[t]["const_acc"]} for t in ("0", "1")}
            per = {0: {"c1": (0.64, True), "c2": (0.62, True)}, 1: {"c1": (0.60, True), "c2": (0.55, False)}}
            for p in (0, 1):
                for cfg, (loc, adm) in per[p].items():
                    gd = self.cache_rel[good][0]
                    for name in (f"fare__s{k}__p{p}__{cfg}", f"fare__s{k}__p{p}__Z{cfg[1:]}"):
                        rel = {"row_id": self.kept.copy(), "cells": np.zeros(len(self.kept), dtype=np.int64),
                               "r": np.zeros((len(self.kept), 3)), "c": gd[f"c{p + 1}"], "p": gd[f"p{p + 1}"],
                               "hard": gd[f"hard{p + 1}"]}
                        self.save_unit(name, {"release.npz": lambda q, rel=rel: np.savez_compressed(q, **rel)},
                                       {"purpose": p})
                    self.save_unit(f"inner__fare__s{k}__p{p}__{cfg}", {},
                                   {"of": f"fare__s{k}__p{p}__{cfg}", "purpose": p, "recovery_local": loc,
                                    "utility": futil[str(p)] if adm else bad_u[str(p)]})
            F_units = [f"fare__s{k}__p0__c2", f"fare__s{k}__p1__c1"]
            F0_units = [f"fare__s{k}__p0__Z2", f"fare__s{k}__p1__Z1"]
            self.fare_sel[k] = (F_units, F0_units)
            for lab, us in (("F", F_units), ("F0", F0_units)):
                a, kind = d[lab]
                ut = futil if kind == "good" else {"0": futil["0"], "1": bad_u["1"]}
                self.save_unit(f"inner__fare__s{k}__{lab}", {}, {"of": us, "recovery": {"auc": dict(zip(("v1", "v2", "pair"), a))},
                                                                 "utility": ut})
            # calibration + Stage C run records with lambda traces (generated by the rule, loop form)
            c = [0.11 + 0.01 * k, 0.21 + 0.01 * k]
            self.save_unit(f"calib__s{k}", {}, {"seed": k, "c": c, "margin": 0.005,
                                                "calib": {"v1": {"R": c[0] - 0.005}, "v2": {"R": c[1] - 0.005}}})
            for arm in ("J-G", "L-G", "J-R", "J-O"):
                for b in BETAS:
                    self.write_run(k, arm, b, c)
        self.write_tracks()

    def write_run(self, k, arm, b, c):
        rng = np.random.default_rng(k * 100 + int(b * 1000) + len(arm))
        guarded = arm in GUARDED
        base = "joint" if arm in JOINT else "local"
        lam = [0.0, 0.0]
        trace, refits, lam_after = [], [], {}
        for e in REFIT_EPOCHS:
            R = [float(c[0] + rng.normal() * 0.03), float(c[1] + rng.normal() * 0.03)]
            refits.append({"epoch": e, "calib": {"v1": {"R": R[0]}, "v2": {"R": R[1]}, "pair": {"R": 0.3}}})
            if guarded:
                if e < 20:
                    for i in (0, 1):
                        x = lam[i] + b * (R[i] - c[i])
                        lam[i] = 0.0 if x < 0 else (3 * b if x > 3 * b else x)
                w = (b / 3, b / 3, b / 3) if base == "joint" else (b / 2, b / 2, 0.0)
                trace.append({"epoch": e, "R_calib": R, "c": c, "lambda": list(lam),
                              "effective_weights": {"v1": w[0] + lam[0], "v2": w[1] + lam[1], "pair": w[2]}})
            lam_after[e] = list(lam)
        cb = {"v1": b / 3, "v2": b / 3, "pair": b / 3} if base == "joint" else {"v1": b / 2, "v2": b / 2, "pair": 0.0}
        if not hasattr(self, "grad_rows"):
            self.grad_rows = []
        lmax = max([max(e["lambda"]) for e in trace] + [0.0])
        self.grad_rows.append({"unit": run_name("C", k, arm, b), "stage": "C", "seed": k, "arm": arm, "beta": b,
                               "tkind": "floored", "base_w_v1": f"{cb['v1']:g}", "base_w_v2": f"{cb['v2']:g}",
                               "base_w_pair": f"{cb['pair']:g}", "budget_c1": f"{c[0]:g}" if guarded else "",
                               "budget_c2": f"{c[1]:g}" if guarded else "", "lambda1_final": f"{lam[0]:g}",
                               "lambda2_final": f"{lam[1]:g}",
                               "lambda_max_over_base": f"{lmax / (b / 3 if base == 'joint' else b / 2):g}"})
        files = {}
        tracked = (arm, b) in self.tracked_runs(k)
        if tracked:
            st_ = self.cache_rel[(k, "good")][2]
            Xref = torch.from_numpy(self.X[self.rows["CRITIC_FIT"][:200]])
            _, s_final = _syn_snapshot(seed=7 + k, state=st_, Xref=Xref)
            _, s_refit = _syn_snapshot(seed=17 + k, state=st_, Xref=Xref)
            _, s1 = _syn_snapshot(seed=27 + k, state=st_, Xref=Xref)
            _, s2 = _syn_snapshot(seed=37 + k, state=st_, Xref=Xref)
            refreshed = arm != "J-O"
            fin = {"theta_T_minus_1": s_final, "theta_T": st_}
            if refreshed:
                fin["refit_critics"] = s_refit
            files["final.pt"] = lambda p: torch.save(fin, p)
            files["captures.pt"] = lambda p: torch.save({1: s1, MID_STEP: s2}, p)
            self.snaps[(k, arm)] = {"start": s1, "mid": s2, "final": s_final}
            if refreshed:
                self.snaps[(k, arm)]["refit20"] = s_refit
                Rn = self._mod_R(s_refit)   # logged epoch-20 R = R from the refit critics (builder's module path)
                for v in ("v1", "v2", "pair"):
                    refits[-1]["calib"][v]["R"] = Rn[v]
                if guarded:
                    trace[-1]["R_calib"] = [Rn["v1"], Rn["v2"]]
        else:
            files["final.pt"] = lambda p: torch.save({"theta_T": {}}, p)
        self.save_unit(run_name("C", k, arm, b), files,
                       {"stage": "C", "arm": arm, "beta": b, "seed": k, "budgets": c if guarded else None,
                        "diag": {"lambda_trace": trace, "c": c if guarded else None, "coefficients_base": cb,
                                 "refits": refits}})
        for e in EPOCHS:
            n = ck_name("C", k, arm, b, e)
            rec_p = self.units / n / "record.json"
            r = json.loads(rec_p.read_text())
            r["lam"] = lam_after[max(x for x in REFIT_EPOCHS if x < e)] if guarded else [0.0, 0.0]
            rec_p.write_text(json.dumps(r))
            d = self.units / n
            hashes = {p.name: sha_file(p) for p in sorted(d.iterdir()) if p.is_file() and p.name != "COMPLETE.json"}
            (d / "COMPLETE.json").write_text(json.dumps({"id": n, "files": hashes}))

    def tracked_runs(self, k):
        if not hasattr(self, "snaps"):
            self.snaps = {}
        ex = self.expect[k]
        out = []
        for arm in ("J-G", "L-G", "J-R", "J-O"):
            u = ex[arm][1]
            out.append((arm, parse_ck(u)["beta"] if u else FIXED_BETA))
        return out

    def _np_X_S(self, rows):
        return self.X[rows], self.lab["sex"][rows]

    def _mod_R(self, snap):
        X, S = self._np_X_S(self.rows["CALIB"])
        ce = _mod_snapshot_ce(snap, X, S)
        df = self.rows["DEFENSE_FIT"]
        p = np.bincount(self.lab["sex"][df], minlength=2) / len(df)
        H = float(-(p * np.log(p)).sum())
        ce0 = float(np.mean(-np.log(p[S])))
        return {v: (ce0 - min([ce0] + list(ce[v].values()))) / H for v in ce}

    def write_tracks(self):
        Xc, Sc = self._np_X_S(self.rows["CALIB"])
        Xi, Si = self._np_X_S(self.rows["INNER_SELECTION"])
        df = self.rows["DEFENSE_FIT"]
        pr = np.bincount(self.lab["sex"][df], minlength=2) / len(df)
        csv_rows = []
        for (k, arm), snaps in self.snaps.items():
            ex = self.expect[k][arm][1]
            b = parse_ck(ex)["beta"] if ex else FIXED_BETA
            rec = {"seed": k, "arm": arm, "run": run_name("C", k, arm, b), "selection_status": self.expect[k][arm][0],
                   "snapshots": {}}
            for nm, snap in snaps.items():
                cec, cei = _mod_snapshot_ce(snap, Xc, Sc), _mod_snapshot_ce(snap, Xi, Si)
                block = {}
                if nm == "refit20":
                    rec[REFIT_BLOCK] = block
                    nm = REFIT_BLOCK
                else:
                    rec["snapshots"][nm] = block
                for v in cec:
                    kinds = {kk: {"online": {"calib": cec[v][kk], "inner": cei[v][kk]},
                                  "fresh": {"calib": cec[v][kk] - 0.01 * (1 + (kk == "B")),
                                            "inner": cei[v][kk] - 0.02}} for kk in cec[v]}
                    node = {"const_ce": {"calib": float(np.mean(-np.log(pr[Sc]))), "inner": float(np.mean(-np.log(pr[Si])))},
                            "kinds": kinds}
                    for rows_ in ("calib", "inner"):
                        on = {kk: kinds[kk]["online"][rows_] for kk in kinds}
                        fr = {kk: kinds[kk]["fresh"][rows_] for kk in kinds}
                        node[f"gap_registered_{rows_}"] = sum(on[kk] - fr[kk] for kk in kinds) / len(kinds)
                        node[f"gap_best_of_bank_{rows_}"] = min(on.values()) - min(fr.values())
                        node[f"online_best_{rows_}"] = min(on.values())
                        node[f"fresh_best_{rows_}"] = min(fr.values())
                        csv_rows.append({"seed": k, "arm": arm, "selection_status": rec["selection_status"],
                                         "run": rec["run"], "snapshot": nm, "view": v, "rows": rows_,
                                         "gap_registered_mean_paired": f"{node[f'gap_registered_{rows_}']:g}",
                                         "gap_best_of_bank": f"{node[f'gap_best_of_bank_{rows_}']:g}",
                                         "online_best_ce": f"{min(on.values()):g}", "fresh_best_ce": f"{min(fr.values()):g}",
                                         "const_ce": f"{node['const_ce'][rows_]:g}",
                                         "online_ce_A": f"{on['A']:g}", "fresh_ce_A": f"{fr['A']:g}",
                                         "online_ce_B": f"{on['B']:g}", "fresh_ce_B": f"{fr['B']:g}"})
                    block[v] = node
            self.save_unit(f"track__s{k}__{arm}", {}, rec)
        with open(self.res / "CRITIC_TRACKING.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
            wr.writeheader()
            wr.writerows(csv_rows)

    # -- runner-side outputs (hand-written selection, brute-force endpoints)
    def write_runner_outputs(self):
        selB, selC, lock = {}, {}, {"seeds": {}}
        for k in SEEDS:
            ex = self.expect[k]
            selB[str(k)] = {a: {"status": ex[a][0], "unit": ex[a][1]} for a in ("L-R", "L-O")}
            selC[str(k)] = {**{a: {"status": ex[a][0], "unit": ex[a][1]} for a in ("L-G", "J-R", "J-O", "J-G")},
                            "C*": {"label": ex["C*"]}}
            score = {}
            for lab in ("L-R", "L-O", "L-G", "J-R", "J-O", "J-G"):
                u = ex[lab][1] or ck_name("C", k, lab, 0.03, 5)
                score[lab] = {"kind": "neural", "unit": u}
            score["U"] = {"kind": "neural", "unit": f"tl__s{k}__e{20 + ex['e_LR']}"}
            score["E"] = {"kind": "neural", "unit": f"lc__s{k}__E"}
            score["F"] = {"kind": "fare", "units": self.fare_sel[k][0]}
            score["F0"] = {"kind": "fare", "units": self.fare_sel[k][1]}
            for fx, arm in FIXED.items():
                score[fx] = {"kind": "neural", "unit": ck_name("C", k, arm, FIXED_BETA, FIXED_EPOCH)}
            ufs = {}
            for spec in score.values():
                for u in ([spec["unit"]] if "unit" in spec else spec["units"]):
                    ufs[u] = {p.name: sha_file(p) for p in (self.units / u).iterdir()
                              if p.is_file() and p.name not in ("COMPLETE.json", "record.json")}
            lock["seeds"][str(k)] = {"score": score, "status": {a: ex[a][0] for a in ("L-R", "L-O", "L-G", "J-R", "J-O", "J-G")},
                                     "comparator": {"label": ex["C*"], "unit": score[ex["C*"]].get("unit")},
                                     "calibrated_budgets": [0.11 + 0.01 * k, 0.21 + 0.01 * k], "unit_file_sha256": ufs}
        (self.run / "selection_B.json").write_text(json.dumps(selB))
        (self.res / "STAGE_B_SELECTION.json").write_text(json.dumps({"seeds": selB}))
        with open(self.res / "GRADIENT_AND_WEIGHT_DIAGNOSTICS.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(self.grad_rows[0]))
            wr.writeheader()
            wr.writerows(self.grad_rows)
        (self.run / "selection_C.json").write_text(json.dumps(selC))
        (self.res / "EVALUATION_LOCK.json").write_text(json.dumps(lock))
        self.lock = lock
        self.write_outer()

    def write_outer(self):
        rng = np.random.default_rng(99)
        dev = self.rows["DEVELOPMENT_ASSESSMENT"]
        n = len(dev)
        sex = self.lab["sex"][dev]
        race = self.lab["race"][dev]
        sup = [int(c) for c in np.unique(race) if all((self.lab["race"][self.rows[r]] == c).sum() >= 30
                                                      for r in ("AUDIT_FIT", "INNER_SELECTION", "DEVELOPMENT_ASSESSMENT"))]
        rpos = np.flatnonzero(np.isin(race, sup))
        ry = np.array([sup.index(int(c)) for c in race[rpos]])
        strength = {"J-G": (0.3, 0.3, 0.3), "J-G@fx": (0.4, 0.4, 0.5)}
        self.preds = {}
        for k in SEEDS:
            cst = self.expect[k]["C*"]
            for lab in SCORED:
                s = strength.get(lab, (1.0, 1.0, 1.2))
                if lab == cst or lab == "L-G":
                    s = (1.6, 1.6, 2.2)
                spec = self.lock["seeds"][str(k)]["score"][lab]
                p = {"assess_row_id": dev.copy(), "assess_unit": self.unit_arr[dev], "sex": sex, "race": race,
                     "y_income": self.lab["y_income"][dev], "y_occ": self.lab["y_occupation_group"][dev]}
                if spec["kind"] == "neural":
                    rel = np.load(self.units / spec["unit"] / "release.npz")
                    pos = np.searchsorted(rel["row_id"], dev)
                    for key in ("hard1", "hard2", "p1", "p2"):
                        p[key] = rel[key][pos]
                else:
                    gd = self.cache_rel[(k, "good")][0]
                    pos = np.searchsorted(gd["row_id"], dev)
                    for key in ("hard1", "hard2", "p1", "p2"):
                        p[key] = gd[key][pos]
                if lab == "J-G":   # J-G deployed outputs identical to U's (same weights in this tree)
                    pass
                for w, a in zip(("v1", "v2", "pair"), s):
                    for pre in ("P_auc_", "P_ce_"):
                        sig = (2 * sex - 1) * a + rng.normal(size=(3, n))
                        q = 1 / (1 + np.exp(-sig))
                        p[pre + w] = np.stack([1 - q, q], 2)
                for w in ("p1", "p2", "ppair", "h1", "h2", "hpair"):
                    for pre in ("P_auc_", "P_ce_"):
                        sig = (2 * sex - 1) * 0.5 + rng.normal(size=(3, n))
                        q = 1 / (1 + np.exp(-sig))
                        p[pre + w] = np.stack([1 - q, q], 2)
                p["race_pos"], p["race_codes"], p["race_y"] = rpos, np.array(sup), ry
                for w in ("v1", "v2", "pair"):
                    lg = rng.normal(size=(3, len(rpos), len(sup))) + np.eye(len(sup))[ry][None] * 0.8
                    pr = np.exp(lg) / np.exp(lg).sum(2, keepdims=True)
                    p[f"Prace_auc_{w}"] = pr
                    p[f"Prace_ce_{w}"] = pr
                self.preds[(k, lab)] = p
                name = f"outer__s{k}__{safe_label(lab)}"
                self.save_unit(name, {"preds.npz": lambda q, p=p: np.savez_compressed(q, **p)}, {"label": lab, "seed": k})
        self.write_endpoints_bruteforce()

    def write_endpoints_bruteforce(self):
        """Runner-side endpoints computed slot by slot with sklearn roc_auc_score(sample_weight) loops."""
        fam = json.loads((self.res / "PRIMARY_FAMILY.json").read_text())
        dev = self.rows["DEVELOPMENT_ASSESSMENT"]
        units = self.unit_arr[dev]
        groups, inv = np.unique(units, return_inverse=True)
        G = len(groups)
        rng = np.random.default_rng(BOOT_SEED)
        reps = [np.ones(len(dev))] + [rng.multinomial(G, np.ones(G) / G)[inv].astype(float) for _ in range(self.B)]
        df = self.rows["DEFENSE_FIT"]
        maj = {0: int(np.argmax(np.bincount(self.lab["y_income"][df]))),
               1: int(np.argmax(np.bincount(self.lab["y_occupation_group"][df])))}
        prior = np.bincount(self.lab["sex"][df], minlength=2) / len(df)
        sex = self.lab["sex"][dev]
        yy = {0: self.lab["y_income"][dev], 1: self.lab["y_occupation_group"][dev]}

        def Rv(k, lab, key, w):
            P = self.preds[(k, lab)][key]
            return float(np.mean([_sk_auc(sex, P[a][:, 1], w) for a in range(3)]))

        def acc(k, lab, t, w):
            return float(np.sum(w * (self.preds[(k, lab)][f"hard{t + 1}"] == yy[t])) / w.sum())

        def const(t, w):
            return float(np.sum(w * (yy[t] == maj[t])) / w.sum())

        def llr(k, lab, v, w):
            P = self.preds[(k, lab)][f"P_ce_{v}"]
            ce0 = np.sum(w * -np.log(prior[sex])) / w.sum()
            return 1 - np.mean([np.sum(w * -np.log(P[a][np.arange(len(sex)), sex])) / w.sum() for a in range(3)]) / ce0

        def rr(k, lab, v, w):
            p = self.preds[(k, lab)]
            P, ry, wr = p[f"Prace_auc_{v}"], p["race_y"], w[p["race_pos"]]
            return float(np.mean([np.mean([_sk_auc(ry == c, P[a][:, c], wr) for c in range(P.shape[2])]) for a in range(3)]))
        vk = {"pair": "P_auc_pair", "v1": "P_auc_v1", "v2": "P_auc_v2"}

        def prim(slot, k, w):
            jg = "J-G"
            ref = "L-G" if slot["claim"] == "A" else self.expect[k]["C*"]
            if slot["kind"] == "coalition":
                return Rv(k, ref, "P_auc_pair", w) - Rv(k, jg, "P_auc_pair", w)
            if slot["kind"] == "local":
                return Rv(k, jg, vk[slot["view"]], w) - Rv(k, ref, vk[slot["view"]], w)
            t = slot["task"]
            if slot["kind"] == "acc":
                return acc(k, jg, t, w) - acc(k, "U", t, w)
            if slot["kind"] == "retain":
                return acc(k, jg, t, w) - 0.8 * acc(k, "U", t, w) - 0.2 * const(t, w)
            return acc(k, jg, t, w) - const(t, w)

        def secd(slot, k, w):
            kd = slot["kind"]
            if kd == "out":
                key = VIEW_KEYS[(slot["fmt"], slot["view"])]
                return Rv(k, slot["a"], key, w) - Rv(k, slot["b"], key, w)
            if kd == "race":
                return rr(k, slot["a"], slot["view"], w) - rr(k, slot["b"], slot["view"], w)
            if kd == "logloss":
                return llr(k, slot["a"], slot["view"], w) - llr(k, slot["b"], slot["view"], w)
            if kd in ("fixed", "rec"):
                key = vk[slot.get("view", "pair")]
                return Rv(k, slot["a"], key, w) - Rv(k, slot["b"], key, w)
            a = slot["arm"]
            return Rv(k, a, "P_auc_pair", w) - max(Rv(k, a, "P_auc_v1", w), Rv(k, a, "P_auc_v2", w))

        def table(slots, f, z):
            rows = []
            for slot in slots:
                vals = np.array([np.mean([f(slot, k, w) for k in SEEDS]) for w in reps])
                se = float(np.std(vals[1:], ddof=1))
                lo, hi = vals[0] - z * se, vals[0] + z * se
                side, tg = slot["side"], slot["target"]
                if side == "lower>":
                    d = "PASS" if lo > tg else "NOT_ESTABLISHED"
                elif side == "upper<":
                    d = "PASS" if hi < tg else "NOT_ESTABLISHED"
                else:
                    d = "ABOVE" if lo > tg else ("BELOW" if hi < tg else "NOT_RESOLVED")
                rows.append({"id": slot["id"], "point": float(vals[0]), "se": se, "lower": float(lo), "upper": float(hi),
                             "decision": d, "n_finite_replicates": self.B})
            return rows
        P = table(fam["primary"], prim, Z_PRIMARY)
        S_ = table(fam["secondary"], secd, Z_SECONDARY)
        for fname, rows in (("PRIMARY_ENDPOINTS.csv", P), ("SECONDARY_ENDPOINTS.csv", S_)):
            with open(self.res / fname, "w", newline="") as f:
                cols = ["id", "point", "se", "lower", "upper", "decision"]
                wr = csv.DictWriter(f, fieldnames=cols)
                wr.writeheader()
                for r in rows:
                    wr.writerow({c: (f"{r[c]:.6f}" if isinstance(r[c], float) else r[c]) for c in cols})
        self.endpoint_rows = {r["id"]: r for r in P + S_}
        nA = sum(r["decision"] == "PASS" for r in P[:9])
        nB = sum(r["decision"] == "PASS" for r in P[9:])
        clA = "NOT_ESTABLISHED"   # seed 1 has no L-G nominee by design
        clB = "ESTABLISHED" if nB == 9 else "NOT_ESTABLISHED"
        (self.run / "inference.json").write_text(json.dumps({
            "primary": P, "secondary": S_, "B": self.B,
            "claimA": {"decision": clA, "clauses_passing": nA},
            "claimB": {"decision": clB, "clauses_passing": nB}}))
        self.claim_expect = {"A": clA, "B": clB}


def selftest_world(st: Report, scratch: Path, B=19):
    sec = "selftest-world"
    root = Path(tempfile.mkdtemp(prefix="rgj_verify_syn_", dir=scratch))
    try:
        t0 = time.time()
        W = SynWorld(root, B=B).build()
        build_s = time.time() - t0
        ctx = Ctx(W.units, W.res, W.src, W.res / "ROLE_MANIFEST.json", wt=root, priv_run=W.run, B=B,
                  src_sha=W.src_sha, synthetic=True, spot=1, sample=4, git=False, label="synthetic")
        rep = Report()
        run_all(ctx, rep)
        stat = {c["id"]: c["status"] for c in rep.checks}
        expected_pass = ["1.1-input-hash", "1.2-permitted-columns", "1.3-role-hashes", "1.4-disjointness",
                         "1.5-no-dropped-rows-in-releases", "2.1-release-reconstruction", "2.2-deployment-label-free",
                         "3.1-stage-B-replay", "3.2-inner-utility-records", "3.3-stage-C-replay",
                         "3.4-evaluation-lock-units", "4.0-family-definition", "4.1-primary-endpoints",
                         "4.2-secondary-endpoints", "4.3-outer-inputs", "5.1-claims", "6.1-lambda-replay",
                         "7.1-gap-recompute", "7.2-online-critic-reeval", "7.3-refit-R-calib", "8.1-complete-json",
                         "8.3-evaluation-lock-hashes", "6.2-weight-diagnostics-csv", "7.4-critic-tracking-csv",
                         "8.7-stage-B-freeze"]
        not_pass = {k: stat.get(k) for k in expected_pass if stat.get(k) != "PASS"}
        errors = [c for c in rep.checks if c["id"].endswith("-ERROR")]
        claims_c = rep.by_id("5.1-claims")
        mine_claims = claims_c[0].get("mine") if claims_c else None
        exp_ok = mine_claims is not None and all(mine_claims[c]["decision"] == W.claim_expect[c] for c in ("A", "B"))
        st.add("W1-synthetic-end-to-end", sec, "PASS" if (not not_pass and not errors and exp_ok) else "FAIL",
               "every check passes on a synthetic study tree with hand-designed selection outcomes (alias, "
               "NO_FEASIBLE_NOMINEE, FARE per-purpose choice, C* = F, min-guard via C*), LEACE maps, rule-generated lambda traces, numpy-computed "
               "critic CEs and brute-force (sklearn sample_weight loop) endpoints with B = %d; 3.5 (inner refit spot "
               "check) is expected to FAIL here because the synthetic inner AUCs are designed, not fitted" % B,
               not_passing=not_pass, errors=[e["detail"] for e in errors], claims=mine_claims,
               claims_expected=W.claim_expect, spot_status=stat.get("3.5-inner-spot-check"), build_s=build_s,
               statuses=stat)
        # tamper tests: each must be detected
        tests = {}
        # (a) corrupt a release file after COMPLETE
        u = W.lock["seeds"]["0"]["score"]["J-G"]["unit"]
        p = W.units / u / "release.npz"
        orig = p.read_bytes()
        p.write_bytes(orig[:-10] + bytes(10))
        r2 = Report()
        check_integrity(Ctx(W.units, W.res, W.src, W.res / "ROLE_MANIFEST.json", wt=root, priv_run=W.run, B=B,
                            src_sha=W.src_sha, synthetic=True, git=False), r2, W.lock)
        tests["corrupt_file"] = {c["id"]: c["status"] for c in r2.checks}
        p.write_bytes(orig)
        # (b) runner selects a different J-G unit
        sc = json.loads((W.run / "selection_C.json").read_text())
        sc["1"]["J-G"]["unit"] = ck_name("C", 1, "J-G", 0.3, 20)
        (W.run / "selection_C.json").write_text(json.dumps(sc))
        r3 = Report()
        c3 = Ctx(W.units, W.res, W.src, W.res / "ROLE_MANIFEST.json", wt=root, priv_run=W.run, B=B, src_sha=W.src_sha,
                 synthetic=True, git=False)
        ms, pr = replay_selection(c3)
        check_selection(c3, r3, W.lock, ms, pr)
        tests["wrong_selection"] = {c["id"]: c["status"] for c in r3.checks}
        sc["1"]["J-G"]["unit"] = W.expect[1]["J-G"][1]
        (W.run / "selection_C.json").write_text(json.dumps(sc))
        # (c) a perturbed multiplier
        rn = run_name("C", 2, "L-G", 0.3)
        rp = W.units / rn / "record.json"
        rr = json.loads(rp.read_text())
        orig_r = rp.read_text()
        rr["diag"]["lambda_trace"][2]["lambda"][1] += 1e-4
        rp.write_text(json.dumps(rr))
        r4 = Report()
        check_lambda(Ctx(W.units, W.res, W.src, W.res / "ROLE_MANIFEST.json", wt=root, priv_run=W.run, B=B,
                         src_sha=W.src_sha, synthetic=True, git=False), r4)
        tests["perturbed_lambda"] = {c["id"]: c["status"] for c in r4.checks}
        rp.write_text(orig_r)
        # (d) a shifted endpoint in the runner CSV
        cp = W.res / "PRIMARY_ENDPOINTS.csv"
        orig_c = cp.read_text()
        rows = list(csv.DictReader(open(cp)))
        rows[0]["point"] = f"{float(rows[0]['point']) + 2e-6:.6f}"
        with open(cp, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        r5 = Report()
        c5 = Ctx(W.units, W.res, W.src, W.res / "ROLE_MANIFEST.json", wt=root, priv_run=W.run, B=B, src_sha=W.src_sha,
                 synthetic=True, git=False)
        ms5, _ = replay_selection(c5)
        check_endpoints(c5, r5, W.lock, ms5)
        tests["shifted_endpoint"] = {c["id"]: c["status"] for c in r5.checks}
        cp.write_text(orig_c)
        # (e) a dropped row smuggled into a release row id
        drop_id = int(np.flatnonzero(np.isin(np.load(W.src)["role"], DROPPED))[0])
        q = W.units / "tl__s0__e5" / "release.npz"
        z = dict(np.load(q))
        z["row_id"] = z["row_id"].copy()
        z["row_id"][0] = drop_id
        orig_q = q.read_bytes()
        np.savez_compressed(q, **z)
        r6 = Report()
        check_roles(Ctx(W.units, W.res, W.src, W.res / "ROLE_MANIFEST.json", wt=root, priv_run=W.run, B=B,
                        src_sha=W.src_sha, synthetic=True, git=False), r6)
        tests["dropped_row_in_release"] = {c["id"]: c["status"] for c in r6.checks}
        q.write_bytes(orig_q)
        # (f) a release whose stored features were altered (reconstruction must notice)
        q = W.units / W.lock["seeds"]["2"]["score"]["U"]["unit"] / "release.npz"
        z = dict(np.load(q))
        z["r1"] = z["r1"].copy()
        z["r1"][5, 3] += 1e-7
        orig_q = q.read_bytes()
        np.savez_compressed(q, **z)
        r7 = Report()
        check_release(Ctx(W.units, W.res, W.src, W.res / "ROLE_MANIFEST.json", wt=root, priv_run=W.run, B=B,
                          src_sha=W.src_sha, synthetic=True, git=False, sample=0), r7, W.lock)
        tests["altered_release_feature"] = {c["id"]: c["status"] for c in r7.checks}
        q.write_bytes(orig_q)
        # (g) a track record whose stored gap disagrees with its per-kind CEs; (h) a shifted tracking-CSV value
        tp = W.units / "track__s0__J-G" / "record.json"
        orig_t = tp.read_text()
        tr = json.loads(orig_t)
        tr["snapshots"]["final"]["pair"]["gap_registered_calib"] += 1e-4
        tp.write_text(json.dumps(tr))
        c8 = Ctx(W.units, W.res, W.src, W.res / "ROLE_MANIFEST.json", wt=root, priv_run=W.run, B=B, src_sha=W.src_sha,
                 synthetic=True, git=False)
        r8 = Report()
        check_critic_gap(c8, r8, W.lock)
        tests["track_gap_field"] = {c["id"]: c["status"] for c in r8.checks}
        tp.write_text(orig_t)
        cp = W.res / "CRITIC_TRACKING.csv"
        orig_c = cp.read_text()
        rows = list(csv.DictReader(open(cp)))
        rows[3]["gap_best_of_bank"] = f"{float(rows[3]['gap_best_of_bank']) + 1e-3:g}"
        with open(cp, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        c9 = Ctx(W.units, W.res, W.src, W.res / "ROLE_MANIFEST.json", wt=root, priv_run=W.run, B=B, src_sha=W.src_sha,
                 synthetic=True, git=False)
        r9 = Report()
        ms9, _ = replay_selection(c9)
        check_tracking_csv(c9, r9, ms9)
        tests["tracking_csv"] = {c["id"]: c["status"] for c in r9.checks}
        cp.write_text(orig_c)
        detected = {
            "track_gap_field": tests["track_gap_field"].get("7.1-gap-recompute") == "FAIL",
            "tracking_csv": tests["tracking_csv"].get("7.4-critic-tracking-csv") == "FAIL",
            "corrupt_file": tests["corrupt_file"].get("8.1-complete-json") == "FAIL"
            and tests["corrupt_file"].get("8.3-evaluation-lock-hashes") == "FAIL",
            "wrong_selection": tests["wrong_selection"].get("3.3-stage-C-replay") == "FAIL",
            "perturbed_lambda": tests["perturbed_lambda"].get("6.1-lambda-replay") == "FAIL",
            "shifted_endpoint": tests["shifted_endpoint"].get("4.1-primary-endpoints") == "FAIL",
            "dropped_row_in_release": tests["dropped_row_in_release"].get("1.5-no-dropped-rows-in-releases") == "FAIL",
            "altered_release_feature": tests["altered_release_feature"].get("2.1-release-reconstruction") == "FAIL"}
        st.add("W2-tamper-detection", sec, "PASS" if all(detected.values()) else "FAIL",
               "on the synthetic tree, a corrupted unit file, a runner selection differing from the replay, a 1e-4 "
               "multiplier perturbation, a 2e-6 shift of a 6-decimal endpoint, a dropped row in a release row id, a 1e-7 change "
               "of a stored feature, a 1e-4 inconsistent track-record gap and a 1e-3 tracking-CSV shift are each detected", detected=detected, statuses=tests)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def selftest_guard(st: Report):
    """The meta_path guard refuses a forbidden module by import and through unpickling (test refusals are kept out
    of the run's refusal log)."""
    import importlib
    import pickle
    sec = "selftest"
    saved = list(GUARD.refused)
    res = {}
    added = str(WT) not in sys.path
    if added:
        sys.path.insert(0, str(WT))
    try:
        try:
            importlib.import_module("rgj.select")
            res["import_rgj_select"] = "IMPORTED"
        except ImportError as e:
            res["import_rgj_select"] = "refused" if "independence guard" in str(e) else f"other ImportError: {e}"
        try:
            pickle.loads(b"coar.study\nanything\n.")
            res["unpickle_oar_global"] = "LOADED"
        except ImportError as e:
            res["unpickle_oar_global"] = "refused" if "independence guard" in str(e) else f"other ImportError: {e}"
        try:
            pickle.loads(b"cpnx.anything\nX\n.")
            res["unpickle_pnx_global"] = "LOADED"
        except ImportError as e:
            res["unpickle_pnx_global"] = "refused" if "independence guard" in str(e) else f"other ImportError: {e}"
    finally:
        if added:
            sys.path.remove(str(WT))
        for m in [m for m in sys.modules if m == "rgj" or m.startswith("rgj.")]:
            del sys.modules[m]
        test_refusals = GUARD.refused[len(saved):]
        GUARD.refused[:] = saved
    loaded = sorted(m for m in sys.modules if _forbidden(m))
    ok = all(v == "refused" for v in res.values()) and not loaded
    st.add("T10-import-guard", sec, "PASS" if ok else "FAIL",
           "the import guard refuses rgj.select by import and oar.* / pnx.* globals through unpickling; nothing "
           "forbidden is left in sys.modules", results=res, test_refusals=test_refusals)


def selftest_git_timing(st: Report):
    sec = "selftest"
    first = dt.datetime(2026, 10, 4, 12, 0, tzinfo=dt.timezone.utc).timestamp()

    class FakeCtx:
        pass
    ok1 = dt.datetime.fromisoformat("2026-10-04T11:59:00+00:00").timestamp() < first
    ok2 = not (dt.datetime.fromisoformat("2026-10-04T08:01:00-04:00").timestamp() < first)   # 12:01Z
    st.add("T9-time-compare", sec, "PASS" if (ok1 and ok2) else "FAIL",
           "commit / push ISO times with offsets are compared to unit file times in UTC epoch seconds")


# ================================================================================================ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true", help="run the synthetic self-tests")
    ap.add_argument("--no-real", action="store_true", help="skip the checks on the real study artefacts")
    ap.add_argument("--spot", type=int, default=3, help="inner-slate spot-check refits (units)")
    ap.add_argument("--sample", type=int, default=8, help="extra hash-ordered checkpoints to reconstruct")
    ap.add_argument("--syn-boot", type=int, default=19, help="bootstrap replicates in the synthetic world")
    ap.add_argument("--out", default=str(RES / "INDEPENDENT_VERIFICATION.json"))
    ap.add_argument("--scratch", default=os.environ.get("TMPDIR", "/tmp"))
    ap.add_argument("--label", default="", help="free-text run label stored in the output (e.g. PHASE_1, PHASE_2)")
    ap.add_argument("--drive", default=None, help="root of the drive copy of the private run (reported as <drive>)")
    a = ap.parse_args(argv)
    t0 = time.time()
    out = {"schema": "rgj-independent-verification-v1",
           "study": "pcrl_refreshed_guarded_joint_v1",
           "verifier": f"{RES_REL}/verification/replay_rgj.py",
           "verifier_sha256": sha_file(HERE),
           "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "run_label": a.label,
           "command": "OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python " + f"{RES_REL}/verification/replay_rgj.py "
                      + " ".join(("<drive>" if a.drive and x == a.drive else x)
                                 for x in (argv if argv is not None else sys.argv[1:]) if not x.startswith(str(HOME))
                                 and "/tmp" not in x and "scratchpad" not in x),
           "independence": {
               "basis": "PROTOCOL.md sections 1, 2, 4, 6, 7, 8; PRIMARY_FAMILY.json; ROLE_MANIFEST.json; METHOD_CARD.md; "
                        "saved private artefacts; preds.npz key names from the rgj/assess.py docstring only",
               "guard": "sys.meta_path finder refusing " + ", ".join(FORBIDDEN_MODULES) + ", pnx.*, oar.*",
               "preloaded_forbidden_modules": PRELOADED_FORBIDDEN},
           "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                           "scikit-learn": sklearn.__version__, "torch": torch.__version__, "joblib": joblib.__version__,
                           "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "torch_threads": torch.get_num_threads(),
                           "machine": platform.machine()},
           "constants": {"B": B_BOOT, "boot_seed": BOOT_SEED, "z_primary": Z_PRIMARY, "z_secondary": Z_SECONDARY}}
    if a.selftest:
        st = Report()
        for fn in (selftest_guard, selftest_primitives, selftest_predecessor_fixture, selftest_git_timing):
            try:
                fn(st)
            except Exception as e:  # noqa: BLE001
                st.add(f"{fn.__name__}-ERROR", "selftest", "FAIL", f"{type(e).__name__}: {e}",
                       trace=traceback.format_exc().splitlines()[-8:])
        try:
            selftest_world(st, Path(a.scratch), B=a.syn_boot)
        except Exception as e:  # noqa: BLE001
            st.add("selftest_world-ERROR", "selftest-world", "FAIL", f"{type(e).__name__}: {e}",
                   trace=traceback.format_exc().splitlines()[-10:])
        out["selftest"] = {"summary": st.summary(), "checks": st.checks}
    if not a.no_real:
        ctx = Ctx(PRIV_RUN / "units", RES, SRC, RES / "ROLE_MANIFEST.json", spot=a.spot, sample=a.sample)
        rep = Report()
        extra = run_all(ctx, rep, drive=a.drive)
        phase = "PHASE_2_POST_ASSESSMENT" if ctx.assessment_open() else "PHASE_1_PRE_ASSESSMENT"
        D = ctx._data
        out["real"] = {"phase": phase, "summary": rep.summary(), "checks": rep.checks, **jsonable(extra),
                       "source_arrays_read": D.arrays_read if D is not None else [],
                       "dev_labels_opened": bool(D is not None and D.dev_open),
                       "units_present": len(ctx.unit_names())}
    loaded = sorted(m for m in sys.modules if _forbidden(m))
    out["independence"]["refused_import_attempts"] = sorted(set(GUARD.refused))
    out["independence"]["forbidden_modules_loaded_at_end"] = loaded
    out["independence"]["runner_packages_loaded_at_end"] = sorted(
        m for m in sys.modules if m.split(".")[0] in ("rgj", "jcv", "pnx", "oar", "stored_model_eval"))
    out["independence"]["ok"] = not loaded and not PRELOADED_FORBIDDEN
    out["wall_s"] = time.time() - t0
    text = json.dumps(jsonable(out), indent=1)
    if a.drive:
        text = text.replace(str(Path(a.drive)), "<drive>").replace(str(Path(a.drive).resolve()), "<drive>")
    text = text.replace(str(HOME), "~")
    assert "/Volumes/" not in text, "drive path in output"
    assert "/Users/" not in text, "absolute user path in output"
    Path(a.out).write_text(text + "\n")
    assert not loaded, f"forbidden modules loaded: {loaded}"
    summ = {k: out[k]["summary"] for k in ("selftest", "real") if k in out}
    print(json.dumps(summ))
    for part in ("selftest", "real"):
        for c in out.get(part, {}).get("checks", []):
            if c["status"] not in ("PASS",):
                print(f"[{part}] {c['status']:7s} {c['id']}: {c['detail'][:150]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
