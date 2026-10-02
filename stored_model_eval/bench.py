"""Matched removal benchmark runner (evaluation owner). Checkpointed per unit, resumable, lock-verified.

    python -m stored_model_eval bench --plan        [--tier 1|2] [--units a,b]   expected vs runnable unit IDs
    python -m stored_model_eval bench --dry-run     [--tier 1|2] [--units a,b]   admission, roles, support, plan; NO fit
    python -m stored_model_eval bench --execute-scientific-fits --tier 1 [--resume] [--units a,b]
    python -m stored_model_eval bench --sigma-star                              attacker_val only -> SIGMA_STAR.json
    python -m stored_model_eval bench --execute-scientific-fits --tier 2 [--resume]   budget-gated, fixed order
    python -m stored_model_eval bench --shuffled-label-sanity --units a,b       fit/validation-only sanity
    python -m stored_model_eval bench --lock-build | --lock-verify               LOCK.json (bench_lock.py)
    python -m stored_model_eval bench --infer | --report                         bench_infer.py / bench_report.py
    python -m stored_model_eval bench --calibrate                                synthetic timing at real sizes

Private root (never inside git; default ~/PCRL_eval_cache_private/bench_v1):
    inputs/INPUTS_INDEX.json  (role 1)          support/SUPPORT_FROZEN.json (frozen at lock build, before any fit)
    defenses/<map_id>/ + MAP_PINS.json          shared/{outputs_only,U2}/<key>/   (alias-recorded shared work)
    units/<unit_id>/{preds.npz, val_preds.npz, models/, fit_records.json, supported.json, COMPLETE.json}
    infer/SIGMA_STAR.json, infer/BENCH_INFER.json  logs/RUN_LOG.jsonl, logs/BUDGET_LEDGER.json

Phases per unit: release (A identity; B/C official LEACE fitted on defense_fit rows ONLY, cached per (dataset, seed,
purpose, concept); D persistent Gaussian draw) -> native checks -> fit phase (receives attacker_fit and attacker_val
views ONLY; every selection happens here) -> predict phase (assessment rows; nothing is selected).
"""
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .admission import sha256_file
from .bench_effective import (BENCH_BRANCH, BENCH_EFFECTIVE, DATASETS, Tracked, all_registered, bench_effective_hash,
                              consumption_report, parse_unit, tier1_units, tier2_units, tier_of, unit_id,
                              validate_bench_effective)
from .guards import FitAuthorization, require_outside_git
from .metrics import to_jsonable
from .recipes import (GaussianClassLRT, _make, fit_family, fit_g2, fit_label_only, fit_rho1, fixed_ridge_fit,
                      full_proba, historical_native_r2, linear_predict, r2_against_prior)
from .surfaces import build_surface

PKG_ROOT = Path(__file__).resolve().parents[1]
PKG_REL = "results/combined_matched_removal_benchmark_v1"
DEFAULT_PRIVATE = Path.home() / "PCRL_eval_cache_private" / "bench_v1"
ROLE_ORDER = ("defense_fit", "attacker_fit", "attacker_val", "assessment")
SCORED = ("attacker_fit", "attacker_val", "assessment")
ADMITTED_LINEAGE = ("VERIFIED", "ADMITTED", "OK")
SURF_KEY = {"rep": "rep", "outputs": "outputs", "rep+outputs": "repPLUSoutputs"}


class BenchRefused(SystemExit):
    pass


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _defenses():
    """Role 2's official-LEACE wrapper (stored_model_eval/defenses.py). Imported lazily so the runner can be planned
    and dry-run before it lands; tests install a local stub module under the same name."""
    from . import defenses  # noqa: WPS433
    return defenses


def _sha_arr(a) -> str:
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def _write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(to_jsonable(obj), indent=1, default=str))
    tmp.replace(p)


# ==================================================================================================================
# worktree / branch
# ==================================================================================================================


def git_branch(root: Path) -> str:
    r = subprocess.run(["git", "-C", str(root), "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True)
    if r.returncode != 0:
        raise BenchRefused(f"REFUSED: {root} is not a git work tree ({r.stderr.strip()})")
    return r.stdout.strip()


def resolve_worktree(explicit: str | None = None, allow_other_branch_for_tests: bool = False) -> dict:
    root = PKG_ROOT if not explicit else Path(explicit).expanduser().resolve()
    if root != PKG_ROOT:
        raise BenchRefused(f"REFUSED: --worktree {root} is not the tree this package runs from ({PKG_ROOT})")
    branch = git_branch(root)
    ok = branch == BENCH_BRANCH
    if not ok and not allow_other_branch_for_tests:
        raise BenchRefused(f"REFUSED: worktree {root} is on branch {branch!r}, expected {BENCH_BRANCH!r}")
    return {"root": str(root), "branch": branch, "branch_ok": ok, "override": not ok}


# ==================================================================================================================
# inputs index (role 1 contract; see notes/evaluation/INPUTS_CONTRACT_CONSUMED.md)
# ==================================================================================================================


def _positions(target_ids, source_ids, what: str) -> np.ndarray:
    """Positions in source for each target id (explicit ID join). Refuses duplicates and missing ids."""
    t, s = np.asarray(target_ids), np.asarray(source_ids)
    for name, ids in (("target", t), ("source", s)):
        u, c = np.unique(ids, return_counts=True)
        if (c > 1).any():
            raise BenchRefused(f"REFUSED: {what}: {name} ids contain {int((c > 1).sum())} duplicates")
    order = np.argsort(s, kind="stable")
    ss = s[order]
    pos = np.searchsorted(ss, t)
    pos = np.clip(pos, 0, len(ss) - 1)
    bad = ss[pos] != t
    if bad.any():
        raise BenchRefused(f"REFUSED: {what}: {int(bad.sum())} row ids absent from the source (e.g. {t[bad][:3].tolist()})")
    return order[pos]


class Inputs:
    """INPUTS_INDEX.json reader (role-1 schema bench_v1.inputs_index/v1; see notes/evaluation/INPUTS_CONTRACT_CONSUMED.md).

    Per dataset: labels_npz (row_id, split, unit, canon_key, role, sensitive attributes, task_<task>), roles_npz
    (<role>__row_id / <role>__unit per role, cross-checked), rows_npz and features_npz (pinned only), purposes
    (index, rep_key, logits_key, labels_task_key), encoders[k] (checkpoint + sha256, lineage_status, forward_npz +
    sha256). Every file is hash-pinned; arrays are joined by row_id, never by position. Rows whose role is not one of
    the four benchmark roles (excluded_dup, unused_train) are dropped before anything else."""

    FILE_KEYS = (("labels_npz", "labels_sha256"), ("roles_npz", "roles_sha256"), ("rows_npz", "rows_sha256"),
                 ("features_npz", "features_sha256"))

    def __init__(self, index_path: Path):
        self.path = Path(index_path).expanduser()
        if not self.path.exists():
            raise BenchRefused(f"REFUSED: inputs index {self.path} not found (role 1 builds it)")
        self.raw = json.loads(self.path.read_text())
        self.sha256 = sha256_file(self.path)
        self.synthetic = bool(self.raw.get("synthetic", False))
        self.base = self.path.parent
        self._ds, self._fw, self._verified = {}, {}, {}
        if "datasets" not in self.raw:
            raise BenchRefused(f"REFUSED: {self.path} has no 'datasets' block")

    def _p(self, s: str) -> Path:
        p = Path(s).expanduser()
        return p if p.is_absolute() else self.base / p

    def all_files(self) -> dict:
        """{resolved path: expected sha256} for every file the index references (the lock pins all of them)."""
        out = {}
        for d, ds in self.raw["datasets"].items():
            for fk, hk in self.FILE_KEYS:
                if ds.get(fk):
                    out[str(self._p(ds[fk]))] = ds[hk]
            for k, e in (ds.get("encoders") or {}).items():
                for fk, hk in (("checkpoint", "checkpoint_sha256"), ("forward_npz", "forward_sha256")):
                    if e.get(fk):
                        out[str(self._p(e[fk]))] = e[hk]
            if self.synthetic and ds.get("historical_native_json"):
                out[str(self._p(ds["historical_native_json"]))] = ds["historical_native_sha256"]
        return out

    def verify_hashes(self) -> list[str]:
        mm = []
        for p, h in sorted(self.all_files().items()):
            if not Path(p).exists():
                mm.append(f"missing input file {p}")
                continue
            got = self._verified.get(p) or sha256_file(Path(p))
            self._verified[p] = got
            if got != h:
                mm.append(f"input file hash mismatch {p}")
        return mm

    def _checked(self, path_s: str, sha: str) -> Path:
        p = self._p(path_s)
        if not p.exists():
            raise BenchRefused(f"REFUSED: missing input file {p}")
        got = self._verified.get(str(p)) or sha256_file(p)
        self._verified[str(p)] = got
        if got != sha:
            raise BenchRefused(f"REFUSED: sha256 mismatch for {p}: index {sha}, file {got}")
        return p

    def datasets(self) -> list[str]:
        return [d for d in self.raw["datasets"] if d in DATASETS]

    def seed_status(self, d: str, k: int) -> dict:
        if d not in self.raw["datasets"]:
            return {"ok": False, "reason": f"dataset {d} absent from {self.path}"}
        e = (self.raw["datasets"][d].get("encoders") or {}).get(str(k))
        if e is None:
            return {"ok": False, "reason": f"encoder seed {k} not declared for {d} in {self.path}"}
        st = str(e.get("lineage_status", "")).upper()
        if st not in ADMITTED_LINEAGE:
            return {"ok": False, "reason": f"lineage {st or 'MISSING'} ({e.get('lineage_reason') or e.get('lineage_record')})",
                    "checkpoint": e.get("checkpoint")}
        for fk in ("checkpoint", "forward_npz"):
            if not e.get(fk):
                return {"ok": False, "reason": f"{fk} not declared"}
            if not self._p(e[fk]).exists():
                return {"ok": False, "reason": f"{fk} file missing: {self._p(e[fk])}"}
        return {"ok": True, "reason": None, "lineage_status": st}

    def dataset(self, d: str) -> dict:
        if d in self._ds:
            return self._ds[d]
        ds = self.raw["datasets"][d]
        lp = self._checked(ds["labels_npz"], ds["labels_sha256"])
        with np.load(lp, allow_pickle=False) as z:
            L = {k: z[k] for k in z.files}
        role_all = L["role"].astype(str)
        keep = np.isin(role_all, ROLE_ORDER)
        row_id = L["row_id"][keep].astype(np.int64)
        roles = role_all[keep]
        # cross-check the per-role lists in roles_npz (ids and units) against the per-row role array
        rp = self._checked(ds["roles_npz"], ds["roles_sha256"])
        with np.load(rp, allow_pickle=False) as z:
            for r in ROLE_ORDER:
                ids = z[f"{r}__row_id"] if f"{r}__row_id" in z.files else np.zeros(0, np.int64)
                if set(ids.tolist()) != set(row_id[roles == r].tolist()):
                    raise BenchRefused(f"REFUSED: {d}: roles_npz {r} row ids disagree with labels_npz role array")
                if f"{r}__unit" in z.files:
                    pos = _positions(ids, row_id, f"{d} roles_npz {r}")
                    if not np.array_equal(z[f"{r}__unit"], L["unit"][keep][pos]):
                        raise BenchRefused(f"REFUSED: {d}: roles_npz {r} units disagree with labels_npz units")
        sens = {a: L[a][keep].astype(np.int64) for a in DATASETS[d]["attr_dims"] if a in L}
        task = {}
        purp = ds.get("purposes") or {}
        for p, v in DATASETS[d]["purposes"].items():
            key = (purp.get(p) or {}).get("labels_task_key") or f"task_{v['task']}"
            if key in L:
                task[p] = L[key][keep].astype(np.int64)
        split = L["split"][keep].astype(str) if "split" in L else None
        test_mask = split == "test" if split is not None else np.isin(roles, SCORED)
        rk = L["canon_key"] if "canon_key" in L else L["record_key"]
        out = {"row_id": row_id, "roles": roles, "units": L["unit"][keep], "record_keys": rk[keep].astype(str),
               "raw_record_keys": L["record_key"][keep].astype(str) if "record_key" in L else rk[keep].astype(str),
               "sens": sens, "task": task, "test_mask": test_mask,
               "test_rule": "labels split == 'test'" if split is not None else "scored roles",
               "idx": {r: np.flatnonzero(roles == r) for r in ROLE_ORDER},
               "purposes_index": {p: int(v["index"]) for p, v in purp.items() if "index" in v},
               "purpose_keys": {p: {"rep": v.get("rep_key", f"rep_p{v.get('index')}"),
                                    "logits": v.get("logits_key", f"logits_{p}")} for p, v in purp.items()},
               "defense_fit_source": {"PREFERRED": "historical_train", "FALLBACK": "fallback_30pct"}.get(
                   str(ds.get("defense_route", "")).upper(), ds.get("defense_route")),
               "n_dropped_rows": {r: int((role_all == r).sum()) for r in np.unique(role_all[~keep])}}
        self._ds[d] = out
        return out

    def forward(self, d: str, k: int) -> dict:
        key = (d, int(k))
        if key in self._fw:
            return self._fw[key]
        st = self.seed_status(d, k)
        if not st["ok"]:
            raise BenchRefused(f"REFUSED: {d} seed {k} not admitted: {st['reason']}")
        e = self.raw["datasets"][d]["encoders"][str(k)]
        self._fw.clear()   # units run dataset -> seed in order; keep one encoder's arrays in memory
        fp = self._checked(e["forward_npz"], e["forward_sha256"])
        D = self.dataset(d)
        out = {}
        with np.load(fp, allow_pickle=False) as z:
            pos = _positions(D["row_id"], z["row_id"], f"{d} s{k} forward cache")
            if "split" in z.files and not np.array_equal(z["split"][pos].astype(str)[D["test_mask"]],
                                                         np.full(int(D["test_mask"].sum()), "test")):
                raise BenchRefused(f"REFUSED: {d} s{k}: forward-cache split disagrees with labels split")
            for name in z.files:
                if name.startswith(("rep_p", "logits_")):
                    out[name] = np.asarray(z[name], dtype=np.float64)[pos]
        ck = self._checked(e["checkpoint"], e["checkpoint_sha256"])
        out["_checkpoint"] = {"path": str(ck), "sha256": e["checkpoint_sha256"]}
        out["_forward_sha256"] = e["forward_sha256"]
        self._fw[key] = out
        return out

    def historical(self, d: str, synthetic: bool) -> dict:
        """dominant_axis_audit.json: from the pinned git blob (real runs) or the index (synthetic fixtures only)."""
        ds = self.raw["datasets"][d]
        if synthetic and ds.get("historical_native_json"):
            p = self._checked(ds["historical_native_json"], ds["historical_native_sha256"])
            return {"source": str(p), "sha256": sha256_file(p), "data": json.loads(p.read_text())}
        from .bench_effective import HISTORICAL_NATIVE
        b = HISTORICAL_NATIVE["blobs"][d]
        r = subprocess.run(["git", "-C", str(PKG_ROOT), "cat-file", "-p", b["blob"]], capture_output=True)
        if r.returncode != 0:
            return {"source": f"git blob {b['blob']}", "error": r.stderr.decode()[:300], "data": None}
        h = subprocess.run(["git", "hash-object", "--stdin"], input=r.stdout, capture_output=True).stdout.decode().strip()
        if h != b["blob"]:
            return {"source": f"git blob {b['blob']}", "error": f"blob id mismatch {h}", "data": None}
        return {"source": f"{HISTORICAL_NATIVE['ref']}:{b['path']} (blob {b['blob']})", "blob": h,
                "data": json.loads(r.stdout)}


# ==================================================================================================================
# roles and support
# ==================================================================================================================


def _uniform(salt: str, key: str) -> float:
    return int(hashlib.sha256((salt + key).encode()).hexdigest()[:8], 16) / 2 ** 32


def role_checks(D: dict, dataset: str, E) -> dict:
    """Admission checks on roles: disjointness by unit and record key, hash rules, Adult pilot counts. No fit."""
    R = E["roles"]
    roles, units, rk = D["roles"], D["units"], D["record_keys"]
    ignored = list(R["ignored_values"])
    fb = (R["fallback_rule"]["salt"], R["fallback_rule"]["share"])
    inrole = np.isin(roles, ROLE_ORDER)
    errs, info = [], {}
    for what, keys in (("unit", units), ("record_key", rk)):
        seen = {}
        cross = 0
        for k, r in zip(np.asarray(keys)[inrole].tolist(), roles[inrole].tolist()):
            if seen.setdefault(k, r) != r:
                cross += 1
        if cross:
            errs.append(f"{cross} rows share a {what} with another role (role disjointness violated)")
    counts = {r: int((roles == r).sum()) for r in ROLE_ORDER}
    info["role_counts"] = counts
    info["n_units_per_role"] = {r: int(len(np.unique(units[roles == r]))) for r in ROLE_ORDER}
    test = D["test_mask"]
    if dataset == "hmda":
        cuts = R["hmda_rule"]["cut_points"]
        salt = R["hmda_rule"]["salt"]
        sc = np.isin(roles, SCORED) & test
        exp = np.array(["attacker_fit" if u < cuts[0] else ("attacker_val" if u < cuts[1] else "assessment")
                        for u in (_uniform(salt, k) for k in rk[sc])])
        agree = float((exp == roles[sc]).mean()) if sc.any() else None
        info["hmda_rule_agreement"] = agree
        if agree is not None and agree < 1.0:
            errs.append(f"HMDA roles disagree with the registered hash rule on {1 - agree:.4%} of scored test rows")
    src = D.get("defense_fit_source")
    info["defense_fit_source"] = src
    info["ignored_role_values_dropped"] = {"declared": ignored, "dropped": D.get("n_dropped_rows", {})}
    if src == "fallback_30pct":
        cand = np.isin(roles, ("defense_fit", "attacker_fit"))
        exp = np.where([_uniform(fb[0], k) < fb[1] for k in rk[cand]], "defense_fit", "attacker_fit")
        agree = float((exp == roles[cand]).mean()) if cand.any() else None
        info["fallback_rule_agreement"] = agree
        if agree is not None and agree < 1.0:
            errs.append("defense_fit fallback roles disagree with the registered hash rule")
    elif src == "historical_train":
        if (roles[D["idx"]["defense_fit"]] == "defense_fit").any() and test[D["idx"]["defense_fit"]].any():
            errs.append("defense_fit rows overlap the test split although defense_fit_source = historical_train")
        exp = dict(_plain(R["adult_pilot_role_counts"]))
        if dataset == "adult":
            got = {k: counts[k] for k in exp}
            info["adult_pilot_counts_match"] = got == exp
            if got != exp:
                info["adult_pilot_counts_note"] = f"expected pilot counts {exp}, got {got}"
    else:
        errs.append(f"defense_fit_source {src!r} is neither historical_train nor fallback_30pct")
    info["errors"] = errs
    return info


def freeze_support_roles(y, roles, thresholds: dict, K: int, min_classes: int, what: str) -> dict:
    from itertools import combinations
    y = np.asarray(y).astype(np.int64)
    roles = np.asarray(roles).astype(str)
    counts = {r: np.bincount(y[roles == r], minlength=K)[:K].tolist() for r in thresholds}
    classes, reasons = [], {}
    for k in range(K):
        why = [f"{r}<{t} (n={counts[r][k]})" for r, t in thresholds.items() if counts[r][k] < t]
        if why:
            reasons[str(k)] = {"code": "class_below_support", "detail": why,
                               "counts": {r: counts[r][k] for r in thresholds}}
        else:
            classes.append(k)
    pairs = [list(p) for p in combinations(classes, 2)]
    allp = [list(p) for p in combinations(range(K), 2)]
    ne = len(classes) < min_classes
    a = roles == "assessment"
    return {"what": what, "n_classes": K, "thresholds": dict(thresholds), "counts_per_role": counts,
            "supported_classes": classes, "supported_pairs": pairs, "unsupported_classes": reasons,
            "unsupported_pairs": [p for p in allp if p not in pairs], "status": "NE" if ne else "ESTIMABLE",
            "ne_reason": "fewer_than_2_supported_classes" if ne else None,
            "coverage": {"classes": [len(classes), K], "pairs": [len(pairs), len(allp)],
                         "assessment_row_fraction_in_supported_classes":
                             float(np.isin(y[a], classes).mean()) if a.any() else 0.0}}


def cell_support(D: dict, dataset: str, purpose: str, attr: str, E) -> dict:
    """Evaluation support (per role, 100/30/100; identical for every arm of the cell) and the defense_fit concept
    support of the B (target) and C (policy) erasers (every class >= min_defense_fit_concept, else that arm is NE)."""
    S = E["support"]
    thr = {"attacker_fit": S["min_attacker_fit"], "attacker_val": S["min_attacker_val"],
           "assessment": S["min_assessment"]}
    if attr not in D["sens"]:
        raise BenchRefused(f"REFUSED: {dataset}: sensitive attribute {attr} absent from labels")
    if purpose not in D["task"]:
        raise BenchRefused(f"REFUSED: {dataset}: task labels for {purpose} absent from labels")
    K_s = int(E["datasets"][dataset]["attr_dims"][attr])
    K_t = int(E["datasets"][dataset]["purposes"][purpose]["n_task_classes"])
    if D["sens"][attr].max() >= K_s or D["task"][purpose].max() >= K_t or D["sens"][attr].min() < 0:
        raise BenchRefused(f"REFUSED: {dataset}/{purpose}/{attr}: labels outside the frozen class counts")
    floor = int(S["min_defense_fit_concept"])
    df = D["idx"]["defense_fit"]
    concept = {}
    for arm, attrs in (("B", [attr]), ("C", list(E["datasets"][dataset]["purposes"][purpose]["disallowed_attrs"]))):
        counts = {a: np.bincount(D["sens"][a][df], minlength=int(E["datasets"][dataset]["attr_dims"][a])).tolist()
                  for a in attrs}
        short = {a: [c for c, n in enumerate(v) if n < floor] for a, v in counts.items()}
        concept[arm] = {"attributes": attrs, "defense_fit_counts": counts, "floor": floor,
                        "status": "FITTABLE" if not any(short.values()) else "NE",
                        "classes_below_floor": {a: v for a, v in short.items() if v}}
    return {"sensitive": freeze_support_roles(D["sens"][attr], D["roles"], thr, K_s, S["min_supported_classes"],
                                              "sensitive"),
            "task": freeze_support_roles(D["task"][purpose], D["roles"], thr, K_t, S["min_supported_classes"], "task"),
            "concept": concept}


def cells_of(units) -> list[tuple]:
    seen = []
    for u in units:
        i = parse_unit(u)
        c = (i["dataset"], i["purpose"], i["attribute"])
        if c not in seen:
            seen.append(c)
    return seen


def freeze_all_support(inputs: Inputs, eff=None) -> dict:
    E = Tracked(BENCH_EFFECTIVE if eff is None else eff)
    cells = {}
    for d, p, a in cells_of(all_registered()):
        if d not in inputs.datasets():
            cells[f"{d}__{p}__{a}"] = {"status": "MISSING_DATASET"}
            continue
        cells[f"{d}__{p}__{a}"] = cell_support(inputs.dataset(d), d, p, a, E)
    return {"schema": "stored_model_eval.bench_support/v1", "frozen_at": _now(),
            "frozen_before": "any eraser, attacker, probe or inference fit (labels and roles only)",
            "inputs_index_sha256": inputs.sha256, "cells": cells}


# ==================================================================================================================
# releases: official LEACE maps (cached + pinned) and persistent noise
# ==================================================================================================================


def onehot(y, K: int) -> np.ndarray:
    return np.eye(K)[np.asarray(y).astype(int)]


def concept_matrix(D, dataset, attrs, rows, E) -> np.ndarray:
    return np.hstack([onehot(D["sens"][a][rows], int(E["datasets"][dataset]["attr_dims"][a])) for a in attrs])


def map_id(d, k, purpose, spec) -> str:
    return f"{d}__s{k}__{purpose}__{spec}"


class MapStore:
    """defenses/<map_id>/ with MAP_RECORD.json + COMPLETE.json; MAP_PINS.json appended on creation, verified on use."""

    def __init__(self, root: Path):
        self.root = Path(root) / "defenses"
        self.pins_path = self.root / "MAP_PINS.json"

    def pins(self) -> dict:
        return json.loads(self.pins_path.read_text()) if self.pins_path.exists() else {"schema": "bench_map_pins/v1",
                                                                                     "maps": {}}

    def verify(self, mid: str | None = None) -> list[str]:
        pins = self.pins()["maps"]
        mm = []
        for m, rec in pins.items():
            if mid and m != mid:
                continue
            d = self.root / m
            for rel, h in rec["files"].items():
                p = d / rel
                if not p.exists():
                    mm.append(f"map {m}: missing {rel}")
                elif sha256_file(p) != h:
                    mm.append(f"map {m}: hash changed {rel}")
        if mid is None and self.root.exists():
            for d in self.root.iterdir():
                if d.is_dir() and d.name not in pins and not d.name.startswith((".", "_")) and ".partial" not in d.name:
                    mm.append(f"map dir {d.name} is not pinned in MAP_PINS.json")
        return mm

    def get(self, mid: str, fit_fn, ctx_log) -> tuple[object, dict]:
        defs = _defenses()
        d = self.root / mid
        pins = self.pins()["maps"]
        if d.exists():
            if mid not in pins:
                raise BenchRefused(f"REFUSED: map {mid} exists but is not pinned in {self.pins_path}")
            mm = self.verify(mid)
            if mm:
                raise BenchRefused(f"REFUSED: pinned map verification failed: {mm}")
            m = defs.LeaceMap.load(d / "map")
            rec = json.loads((d / "MAP_RECORD.json").read_text())
            rec["reused"] = True
            return m, rec
        if mid in pins:
            raise BenchRefused(f"REFUSED: map {mid} is pinned but its directory is missing")
        tmp = self.root / f"_{mid}.partial"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        m, rec = fit_fn()
        m.save(tmp / "map")
        rec["created_at"] = _now()
        (tmp / "MAP_RECORD.json").write_text(json.dumps(to_jsonable(rec), indent=1, default=str))
        files = {str(p.relative_to(tmp)): sha256_file(p) for p in sorted(tmp.rglob("*")) if p.is_file()}
        tmp.rename(d)
        allp = self.pins()
        allp["maps"][mid] = {"files": files, "pinned_at": _now(), "concept_spec": rec["concept_spec"],
                             "defense_fit_row_ids_sha256": rec["defense_fit_row_ids_sha256"]}
        _write_json(self.pins_path, allp)
        ctx_log(f"[bench] fitted and pinned LEACE map {mid}")
        rec["reused"] = False
        return defs.LeaceMap.load(d / "map"), rec


def independent_native_check(H_fit, Z_fit, T_fit) -> dict:
    """Evaluation-side LEACE condition check on defense_fit (independent of the defense module's own check)."""
    H, Z, T = (np.asarray(a, dtype=np.float64) for a in (H_fit, Z_fit, T_fit))
    Zc = Z - Z.mean(0)
    cov0 = (H - H.mean(0)).T @ Zc / (len(H) - 1)
    cov1 = (T - T.mean(0)).T @ Zc / (len(T) - 1)
    X = np.c_[T, np.ones(len(T))]
    B, *_ = np.linalg.lstsq(X, Z, rcond=None)
    res = Z - X @ B
    ss_tot = float((Zc ** 2).sum())
    r2 = 1.0 - float((res ** 2).sum()) / ss_tot if ss_tot > 0 else None
    return {"cross_cov_max_abs_before": float(np.abs(cov0).max()), "cross_cov_max_abs_after": float(np.abs(cov1).max()),
            "cross_cov_rel": float(np.abs(cov1).max() / max(np.abs(cov0).max(), 1e-300)),
            "fit_row_affine_ols_r2": r2, "n_rows": int(len(H)), "concept_columns": int(Z.shape[1])}


def get_release_map(ctx, D, d, k, purpose, attrs, kind, H_all) -> tuple[object, dict]:
    E = ctx["eff"]
    spec = f"{kind}_{'+'.join(attrs)}"
    mid = map_id(d, k, purpose, spec)
    df = D["idx"]["defense_fit"]
    L = E["defenses"]["leace"]
    if L["fit_role"] != "defense_fit" or L["dtype"] != "float64":
        raise ValueError("LEACE must be fitted in float64 on defense_fit")
    want = dict(L["settings"])

    def fit():
        defs = _defenses()
        H_fit = np.asarray(H_all[df], dtype=np.float64)
        Z_fit, cspec = defs.concat_marginal_onehots({a: D["sens"][a][df] for a in attrs},
                                                    {a: int(E["datasets"][d]["attr_dims"][a]) for a in attrs})
        t0 = time.process_time()
        m = defs.fit_leace(H_fit, Z_fit, dtype=np.float64, fit_row_ids=D["row_id"][df], concept_spec=cspec,
                           auth=ctx["auth"], synthetic=ctx["synthetic"])
        cpu = time.process_time() - t0
        if m.metadata.get("settings_used") != want:
            raise BenchRefused(f"REFUSED: LEACE settings used {m.metadata.get('settings_used')} != protocol {want}")
        own = m.native_check(H_fit, Z_fit, tol_rel=E["native"]["BC"]["descriptive_tol_rel"],
                             tol_r2=E["native"]["BC"]["descriptive_tol_r2"])
        ind = independent_native_check(H_fit, Z_fit, m.transform(H_fit))
        return m, {"map_id": mid, "concept_spec": {"kind": kind, "attributes": list(attrs), "blocks": cspec},
                   "fit_role": "defense_fit", "n_fit_rows": int(len(df)),
                   "defense_fit_row_ids_sha256": _sha_arr(np.sort(D["row_id"][df].astype(np.int64))),
                   "map_metadata": {k_: m.metadata.get(k_) for k_ in ("settings_used", "rank", "tolerances",
                                                                      "fit_row_ids_sha256", "H_fit_sha256",
                                                                      "Z_fit_sha256", "diagnostics", "provenance")},
                   "native_check_defense_module": own, "native_check_independent": ind, "fit_cpu_s": cpu,
                   "official": dict(E["defenses"]["leace"]["official"])}
    return ctx["maps"].get(mid, fit, ctx["log"])


def out_of_support(H_fit, rows: dict) -> dict:
    """Label-free transfer limitation: component of (h - fit mean) outside range(sample cov of defense_fit)."""
    H = np.asarray(H_fit, dtype=np.float64)
    mu = H.mean(0)
    S = np.cov(H, rowvar=False)
    L, V = np.linalg.eigh((S + S.T) / 2)
    keep = L > L[-1] * len(L) * np.finfo(np.float64).eps
    Vn = V[:, ~keep]
    out = {"fit_cov_rank": int(keep.sum()), "dim": int(len(L))}
    for name, X in rows.items():
        C = np.asarray(X, dtype=np.float64) - mu
        tot = np.linalg.norm(C, axis=1)
        oos = np.linalg.norm(C @ Vn, axis=1) if Vn.shape[1] else np.zeros(len(C))
        rel = np.where(tot > 0, oos / np.where(tot > 0, tot, 1), 0.0)
        out[name] = {"n_rows": int(len(C)), "oos_norm_median": float(np.median(oos)), "oos_norm_max": float(oos.max()),
                     "oos_rel_median": float(np.median(rel)), "oos_rel_max": float(rel.max()),
                     "centred_norm_median": float(np.median(tot))}
    return out


def noise_matrix_rows(D) -> np.ndarray:
    sc = np.flatnonzero(np.isin(D["roles"], SCORED))
    return sc[np.argsort(D["row_id"][sc], kind="stable")]


def build_release(ctx, info, D, H) -> tuple[np.ndarray, dict, object]:
    arm, d, k, p, a = info["arm"], info["dataset"], info["seed"], info["purpose"], info["attribute"]
    if arm == "A":
        return H, {"kind": "identity", "release_key": "A"}, None
    if arm in ("B", "C"):
        attrs = [a] if arm == "B" else list(ctx["eff"]["datasets"][d]["purposes"][p]["disallowed_attrs"])
        m, rec = get_release_map(ctx, D, d, k, p, attrs, arm, H)
        R = np.asarray(m.transform(H), dtype=np.float64)
        return R, {"kind": "official_leace", "release_key": "B_" + a if arm == "B" else "C", "map": rec,
                   "map_pin": ctx["maps"].pins()["maps"].get(rec["map_id"])}, m
    N = ctx["eff"]["defenses"]["noise"]
    if N["draw_rows"] != "scored_roles_ascending_row_id":
        raise ValueError("unsupported noise draw rows")
    if info["sigma"] not in list(N["sigmas"]) or info["release_seed"] not in list(N["release_seeds"]):
        raise BenchRefused(f"REFUSED: {info['unit_id']}: sigma/seed outside the frozen grid")
    rows = noise_matrix_rows(D)
    R = np.array(H, dtype=np.float64, copy=True)
    R[rows] = np.asarray(_defenses().noise_release(H[rows], info["sigma"], info["release_seed"]), dtype=np.float64)
    rec = {"kind": "gaussian_noise", "release_key": info["arm_tag"], "sigma_abs": info["sigma"],
           "release_seed": info["release_seed"], "persistent": True, "release_count": "one",
           "draw": "defenses.noise_release over scored-role rows in ascending row_id order",
           "release_scored_rows_sha256": _sha_arr(R[rows]), "n_rows_drawn": int(len(rows))}
    try:
        rec["scale_report"] = to_jsonable(_defenses().scale_report(H[D["idx"]["attacker_fit"]]))
    except Exception as e:  # noqa: BLE001 - descriptive only
        rec["scale_report"] = {"error": repr(e)}
    return R, rec, None


# ==================================================================================================================
# attacker slates, attacker-seed retraining, shared work
# ==================================================================================================================


def _access(tag, attacker, surface, contract: dict, fit_inputs, requires=(), status="EXECUTED", note=None):
    return {"tag": tag, "attacker": attacker, "surface": surface, "fit_inputs": list(fit_inputs),
            "eval_inputs": ["release(assessment row), one persistent draw" if contract.get("noise") != "none"
                            else "release(assessment row), deterministic"],
            "requires": list(requires), "contract": contract, "status": status, "note": note}


def fit_slate(Xf, yf, Xv, yv, K, E, auth, syn, what) -> dict:
    att = E["attackers"]
    clip = att["log_loss_clip"]
    for fam in ("L", "GBT", "MLP"):
        if att[fam]["select_metric"] != "attacker_val_log_loss" or att["selection_role"] != "attacker_val":
            raise ValueError(f"{fam} selection must be attacker_val log-loss")
    res = {fam: fit_family(fam, Xf, yf, Xv, yv, K, att[fam], clip, auth=auth, synthetic=syn, what=what)
           for fam in ("L", "GBT", "MLP")}
    if list(att["NL"]["members"]) != ["GBT", "MLP"] or att["NL"]["select_metric"] != "attacker_val_log_loss":
        raise ValueError("NL members/selection must be GBT, MLP on attacker_val log-loss")
    g, m = res["GBT"], res["MLP"]
    res["NL"] = {"selected_family": "GBT" if g["attacker_val_log_loss"] <= m["attacker_val_log_loss"] else "MLP",
                 "candidates": {"GBT": g["attacker_val_log_loss"], "MLP": m["attacker_val_log_loss"]}}
    return res


def retrain_seeds(family: str, hp: dict, grid_model, Xf, yf, E, auth, syn, what) -> tuple[dict, dict]:
    """Selected recipe refit at attacker seeds {0,1,2}. Seed 0 is the grid fit (random_state 0); L is deterministic
    (lbfgs) and aliased. Returns ({seed: model}, record)."""
    A = E["attackers"]
    seeds = [int(s) for s in A["attacker_seeds"]]
    if A["seed_param"] != "random_state":
        raise ValueError("attacker seeds act through random_state")
    models, rec = {}, {"family": family, "hp": hp, "seeds": {}}
    if family in A["deterministic_alias"]:
        for s in seeds:
            models[s] = grid_model
            rec["seeds"][str(s)] = {"status": "ALIAS_DETERMINISTIC", "alias_of": "as0 (grid fit)"}
        return models, rec
    cfg = A[family]
    if int(cfg["random_state"]) != seeds[0]:
        raise ValueError("grid random_state must equal attacker seed 0")
    import warnings
    for s in seeds:
        if s == seeds[0]:
            models[s] = grid_model
            rec["seeds"][str(s)] = {"status": "GRID_FIT", "random_state": s}
            continue
        auth.check(f"{family} attacker-seed {s} ({what})", syn)
        t0 = time.process_time()
        m = _make(family, hp, cfg, random_state=s)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(Xf, np.asarray(yf).astype(int))
        models[s] = m
        rec["seeds"][str(s)] = {"status": "REFIT", "random_state": s, "cpu_s": time.process_time() - t0}
    return models, rec


def select_plus(cands: dict) -> str:
    """Validation-only choice over the C_rep_plus_clean_out slate (ties -> earlier listed candidate)."""
    best = None
    for name, ll in cands.items():
        if ll is None or not np.isfinite(ll):
            continue
        if best is None or ll < cands[best]:
            best = name
    return best


class SharedStore:
    """shared/<kind>/<key>/: shared work reused by several units, alias-recorded, COMPLETE.json-verified."""

    def __init__(self, root: Path):
        self.root = Path(root) / "shared"

    def dir(self, kind, key) -> Path:
        return self.root / kind / key

    @staticmethod
    def verify(d: Path) -> tuple[bool, list]:
        c = d / "COMPLETE.json"
        if not c.exists():
            return False, ["no COMPLETE.json"]
        rec = json.loads(c.read_text())
        errs = [f"{'missing' if not (d / r).exists() else 'hash changed'} {r}" for r, h in rec["files"].items()
                if not (d / r).exists() or sha256_file(d / r) != h]
        return not errs, errs


def write_complete(d: Path, uid: str) -> dict:
    rec = {"id": uid, "completed_at": _now(),
           "files": {str(p.relative_to(d)): sha256_file(p) for p in sorted(d.rglob("*"))
                     if p.is_file() and p.name != "COMPLETE.json"}}
    (d / "COMPLETE.json").write_text(json.dumps(rec, indent=1))
    return rec


def verify_unit(d: Path) -> tuple[bool, list]:
    ok, errs = SharedStore.verify(d)
    if ok:
        files = json.loads((d / "COMPLETE.json").read_text())["files"]
        for need in ("preds.npz", "fit_records.json", "supported.json"):
            if need not in files:
                errs.append(f"{need} not recorded")
    return not errs, errs


def _save_models(d: Path, models: dict):
    import joblib
    (d / "models").mkdir(parents=True, exist_ok=True)
    seen = {}
    out = {}
    for name, m in models.items():
        if id(m) in seen:
            out[name] = {"alias_of": seen[id(m)]}
            continue
        f = f"models/{name.replace('+', 'PLUS').replace('|', '__')}.joblib"
        joblib.dump(m, d / f)
        seen[id(m)] = f
        out[name] = {"file": f}
    return out


def outputs_only(ctx, info, F, V, A, K_s) -> dict:
    """outputs_only (clean logits only) per (dataset, encoder seed, purpose, attribute): fitted once, aliased into every
    arm. Returns assessment predictions, validation log-losses of the NL / L selections and the seed models."""
    E = ctx["eff"]
    if E["surfaces"]["outputs_share_key"] != "dataset|encoder_seed|purpose|attribute":
        raise ValueError("outputs_only share key changed")
    key = f"{info['dataset']}__s{info['seed']}__{info['purpose']}__{info['attribute']}"
    d = ctx["shared"].dir("outputs_only", key)
    if d.exists():
        ok, errs = SharedStore.verify(d)
        if ok:
            with np.load(d / "preds.npz", allow_pickle=False) as z:
                P = {k: z[k] for k in z.files}
            if not np.array_equal(P["assess_row_id"], A["row_id"]):
                raise BenchRefused(f"REFUSED: shared outputs_only {key}: assessment ids differ")
            rec = json.loads((d / "record.json").read_text())
            return {"key": key, "preds": P, "record": rec, "reused": True,
                    "preds_sha256": sha256_file(d / "preds.npz"), "cpu_s": 0.0}
        shutil.move(str(d), str(d.with_name(d.name + f".partial-{int(time.time())}")))
    t0 = time.process_time()
    kind = E["surfaces"]["outputs_kind"]
    Xf, srec = build_surface("outputs", None, F["outputs"], kind)
    Xv, _ = build_surface("outputs", None, V["outputs"], kind)
    Xa, _ = build_surface("outputs", None, A["outputs"], kind)
    what = f"{key}/outputs_only"
    res = fit_slate(Xf, F["s"], Xv, V["s"], K_s, E, ctx["auth"], ctx["synthetic"], what)
    nlf = res["NL"]["selected_family"]
    seedsN, recN = retrain_seeds(nlf, res[nlf]["selected"], res[nlf]["model"], Xf, F["s"], E, ctx["auth"],
                                 ctx["synthetic"], what)
    seedsL, recL = retrain_seeds("L", res["L"]["selected"], res["L"]["model"], Xf, F["s"], E, ctx["auth"],
                                 ctx["synthetic"], what)
    preds = {"assess_row_id": A["row_id"]}
    for fam in ("L", "GBT", "MLP"):
        preds[f"P__outputs__{fam}"] = full_proba(res[fam]["model"], Xa, K_s)
    for s, m in seedsN.items():
        preds[f"P__outputs__NL__as{s}"] = full_proba(m, Xa, K_s)
    for s, m in seedsL.items():
        preds[f"P__outputs__L__as{s}"] = full_proba(m, Xa, K_s)
    d.mkdir(parents=True)
    np.savez_compressed(d / "preds.npz", **preds)
    models = {f"outputs|{fam}": res[fam]["model"] for fam in ("L", "GBT", "MLP")}
    models.update({f"outputs|NL|as{s}": m for s, m in seedsN.items()})
    models.update({f"outputs|L|as{s}": m for s, m in seedsL.items()})
    mf = _save_models(d, models)
    rec = {"key": key, "surface_record": srec.to_json(), "families": {
        fam: {"selected": res[fam]["selected"], "selection_table": res[fam]["selection_table"],
              "attacker_val_log_loss": res[fam]["attacker_val_log_loss"]} for fam in ("L", "GBT", "MLP")},
        "NL": res["NL"], "seed_retrain": {"NL": recN, "L": recL}, "model_files": mf,
        "n_model_fits": sum(len(res[f]["selection_table"]) for f in ("L", "GBT", "MLP"))
        + sum(1 for r in (recN, recL) for v in r["seeds"].values() if v["status"] == "REFIT"),
        "val_log_loss": {"NL": res[nlf]["attacker_val_log_loss"], "L": res["L"]["attacker_val_log_loss"]},
        "cpu_s": time.process_time() - t0, "written_at": _now()}
    (d / "record.json").write_text(json.dumps(to_jsonable(rec), indent=1, default=str))
    write_complete(d, key)
    return {"key": key, "preds": preds, "record": rec, "reused": False, "preds_sha256": sha256_file(d / "preds.npz"),
            "cpu_s": rec["cpu_s"]}


def u2_probe(ctx, info, release_key, F, V, A, K_t) -> dict:
    """U2 LR probe per (dataset, seed, purpose, release): shared across attributes when the release is identical."""
    E = ctx["eff"]
    u2 = E["utility"]["U2"]
    if u2["fit_role"] != "attacker_fit" or u2["select_role"] != "attacker_val":
        raise ValueError("U2 roles must be attacker_fit / attacker_val")
    key = f"{info['dataset']}__s{info['seed']}__{info['purpose']}__{release_key}"
    d = ctx["shared"].dir("U2", key)
    if d.exists():
        ok, _ = SharedStore.verify(d)
        if ok:
            rec = json.loads((d / "record.json").read_text())
            if rec.get("release_sha256") != _sha_arr(A["rep"]):
                raise BenchRefused(f"REFUSED: shared U2 {key}: release differs from the cached release")
            with np.load(d / "preds.npz", allow_pickle=False) as z:
                P = z["U2_P"]
            return {"key": key, "U2_P": P, "record": rec, "reused": True, "cpu_s": 0.0}
        shutil.move(str(d), str(d.with_name(d.name + f".partial-{int(time.time())}")))
    t0 = time.process_time()
    att = E["attackers"]
    cfg = {"C": u2["C"], "max_iter": u2["max_iter"], "solver": att["L"]["solver"], "scaler": att["L"]["scaler"]}
    r = fit_family("L", F["rep"], F["t"], V["rep"], V["t"], K_t, cfg, att["log_loss_clip"], auth=ctx["auth"],
                   synthetic=ctx["synthetic"], what=f"{key}/U2")
    P = full_proba(r["model"], A["rep"], K_t)
    d.mkdir(parents=True)
    np.savez_compressed(d / "preds.npz", U2_P=P, assess_row_id=A["row_id"])
    mf = _save_models(d, {"U2": r["model"]})
    rec = {"key": key, "selected": r["selected"], "selection_table": r["selection_table"],
           "attacker_val_log_loss": r["attacker_val_log_loss"], "release_sha256": _sha_arr(A["rep"]),
           "model_files": mf, "n_model_fits": len(r["selection_table"]), "cpu_s": time.process_time() - t0,
           "target": "y_task", "written_at": _now()}
    (d / "record.json").write_text(json.dumps(to_jsonable(rec), indent=1, default=str))
    write_complete(d, key)
    return {"key": key, "U2_P": P, "record": rec, "reused": False, "cpu_s": rec["cpu_s"]}


# ==================================================================================================================
# phases
# ==================================================================================================================


def native_phase(ctx, info, D, H, s_all, release_rec, m) -> dict:
    E = ctx["eff"]
    arm = info["arm"]
    if arm == "A":
        n = E["native"]["A"]
        tm = D["test_mask"]
        if n["rows"] != "test_split":
            raise ValueError("A native rows must be the test split")
        ctx["auth"].check("A native in-sample statistic", ctx["synthetic"])
        out = {v: historical_native_r2(H[tm], s_all[tm], n["lambda"], v) for v in n["variants"]}
        cv = n["category_variant"]
        out.update(arm="A", rows="test_split", test_rule=D["test_rule"], n_rows=int(tm.sum()), tau=n["tau"],
                   category="above tau (historical check fails)" if out[cv]["clamped"] > n["tau"]
                   else "historical check passes")
        hist = ctx["inputs"].historical(info["dataset"], ctx["synthetic"])
        out["historical_source"] = hist.get("source")
        row = None
        if hist.get("data"):
            for r in hist["data"].get("per_seed", {}).get(str(info["seed"]), {}).get("rows", []):
                if r.get("purpose") == info["purpose"] and r.get("attribute") == info["attribute"]:
                    row = r
        if row is None:
            out["reproduction"] = {"status": "NO_HISTORICAL_ROW", "error": hist.get("error")}
        else:
            diff = abs(out[cv]["clamped"] - float(row["r2_onehot"]))
            out["reproduction"] = {"historical_r2_onehot": float(row["r2_onehot"]), "abs_diff": diff,
                                   "atol": n["reproduce_atol"],
                                   "status": "REPRODUCED" if diff <= n["reproduce_atol"] else "NOT_REPRODUCED"}
        return out
    if arm in ("B", "C"):
        n = E["native"]["BC"]
        if n["rows"] != "defense_fit" or n["primary"] != "implementation_bound_holds":
            raise ValueError("B/C native check: defense_fit rows, implementation bound primary")
        mrec = release_rec["map"]
        own = mrec["native_check_defense_module"]
        ok = bool(own.get("implementation_bound_holds"))
        df = D["idx"]["defense_fit"]
        return {"arm": arm, "rows": "defense_fit", "map_id": mrec["map_id"],
                "status": "PASS" if ok else "FAIL",
                "category_if_fail": "C1 (native check fails as specified by the implementation)",
                "primary": {"check": "implementation_bound_holds",
                            "whitened_residual_spectral_norm": own.get("whitened_residual_spectral_norm"),
                            "svd_tol": own.get("svd_tol"), "holds": ok},
                "exact_zero_crosscov": {"status": own.get("status"),
                                        "crosscov_max_abs_rel_erased": own.get("crosscov_max_abs_rel_erased"),
                                        "tol_rel": own.get("tol_rel"),
                                        "ols_r2_joint_erased": own.get("ols_r2_joint_erased"),
                                        "n_singular_values_nonzero_truncated":
                                            own.get("n_singular_values_nonzero_truncated"),
                                        "note": "descriptive; OUTSIDE_TOLERANCE_SVD_TOL_TRUNCATION is not C1"},
                "independent": mrec["native_check_independent"],
                "out_of_support": out_of_support(H[df], {"attacker_fit": H[D["idx"]["attacker_fit"]],
                                                         "assessment": H[D["idx"]["assessment"]]}),
                "scope": "fitted erasure condition on defense_fit rows for the stated concept only"}
    return {"arm": "D", "status": E["native"]["D"]["status"], "label": E["native"]["D"]["label"]}


def fit_phase(view: dict, ctx: dict) -> tuple[dict, list, dict, dict]:
    """All attacker / probe fitting and selection. `view` holds ONLY attacker_fit ("fit") and attacker_val ("val")."""
    if set(view) != {"fit", "val"}:
        raise ValueError(f"fit_phase accepts only fit/val views, got {sorted(view)}")
    E, auth, syn = ctx["eff"], ctx["auth"], ctx["synthetic"]
    info, K_s, K_t, contract = ctx["info"], ctx["K_s"], ctx["K_t"], ctx["contract"]
    F, V = view["fit"], view["val"]
    att = E["attackers"]
    lc = E["linear_closed_form"]
    models, recs, closed, valp = {}, [], {}, {}
    a1 = ["release(attacker_fit)", "S(attacker_fit)"]
    estimable = ctx["support"]["status"] != "NE"
    if estimable:
        g1 = lc["G1"]
        auth.check("G1 fixed ridge", syn)
        models["G1"] = fixed_ridge_fit(F["rep"], F["s"], K_s, g1["lambda"], np.dtype(g1["dtype"]).type)
        closed["G1"] = {"fit_role": g1["fit_role"], "score_role": g1["score_role"], "clamp": g1["clamp"],
                        "access": _access("A1", "G1_fixed_ridge", "rep", contract, a1)}
        g2 = fit_g2(F["rep"], F["s"], V["rep"], V["s"], K_s, lc["G2"], auth=auth, synthetic=syn)
        if lc["G2"]["select_on"] != "attacker_val" or lc["G2"]["select_metric"] != "heldout_r2_fitmean":
            raise ValueError("G2 selection must be attacker_val held-out R2")
        models["G2"] = g2["model"]
        closed["G2"] = {"selected": g2["selected"], "selection_table": g2["selection_table"]}
        models["RHO1"] = fit_rho1(F["rep"], F["s"], K_s, lc["rho1_heldout"]["eps_rel"], auth=auth, synthetic=syn)
        closed["RHO1"] = {k: v for k, v in models["RHO1"].items() if k not in ("a", "b")}

        # C_rep slate
        what = f"{info['unit_id']}/rep"
        rep = fit_slate(F["rep"], F["s"], V["rep"], V["s"], K_s, E, auth, syn, what)
        for fam in ("L", "GBT", "MLP"):
            models[("rep", fam)] = rep[fam]["model"]
            recs.append({"surface": "rep", "recipe": fam, "status": "EXECUTED", "selected": rep[fam]["selected"],
                         "selection_table": rep[fam]["selection_table"],
                         "attacker_val_log_loss": rep[fam]["attacker_val_log_loss"], "cpu_s": rep[fam]["cpu_s"],
                         "selection_role": att["selection_role"], "access": _access("A1", fam, "rep", contract, a1)})
        nlf = rep["NL"]["selected_family"]
        sN, rN = retrain_seeds(nlf, rep[nlf]["selected"], rep[nlf]["model"], F["rep"], F["s"], E, auth, syn, what)
        sL, rL = retrain_seeds("L", rep["L"]["selected"], rep["L"]["model"], F["rep"], F["s"], E, auth, syn, what)
        for s_, m_ in sN.items():
            models[("rep", f"NL|as{s_}")] = m_
        for s_, m_ in sL.items():
            models[("rep", f"L|as{s_}")] = m_
        recs.append({"surface": "rep", "recipe": "NL", "status": "EXECUTED", "nl_selection": rep["NL"],
                     "selected": {"family": nlf, **rep[nlf]["selected"]}, "seed_retrain": rN,
                     "access": _access("A1", "NL", "rep", contract, a1,
                                       note=("LEACE arm: ordinary slate on erased features = defense-aware "
                                             "simulation (alias); no stronger attack claimed")
                                       if info["arm"] in ("B", "C") else None)})
        recs.append({"surface": "rep", "recipe": "L_seeds", "status": "ALIAS", "seed_retrain": rL})
        valp["VAL__rep__NL__as0"] = full_proba(sN[0], V["rep"], K_s)
        valp["VAL__rep__L"] = full_proba(rep["L"]["model"], V["rep"], K_s)

        # outputs_only (shared, aliased across arms)
        oo = ctx["outputs_only"]()
        orec = oo["record"]
        recs.append({"surface": "outputs", "recipe": "outputs_only", "status": "REUSED" if oo["reused"] else "EXECUTED",
                     "shared_key": oo["key"], "shared_preds_sha256": oo["preds_sha256"],
                     "NL": orec["NL"], "val_log_loss": orec["val_log_loss"],
                     "access": _access("A1", "outputs_only", "outputs", {"noise": "none"},
                                       ["clean outputs(attacker_fit)", "S(attacker_fit)"],
                                       note="clean logits only; identical across methods; fitted once per "
                                            "(dataset, encoder seed, purpose, attribute)")})

        # C_rep_plus_clean_out slate incl. ignore-rep / ignore-outputs candidates
        kind = E["surfaces"]["outputs_kind"]
        Xf, srec = build_surface("rep+outputs", F["rep"], F["outputs"], kind)
        Xv, _ = build_surface("rep+outputs", V["rep"], V["outputs"], kind)
        whatp = f"{info['unit_id']}/rep+outputs"
        plus = fit_slate(Xf, F["s"], Xv, V["s"], K_s, E, auth, syn, whatp)
        for fam in ("L", "GBT", "MLP"):
            models[("rep+outputs", fam)] = plus[fam]["model"]
            recs.append({"surface": "rep+outputs", "recipe": fam, "status": "EXECUTED",
                         "selected": plus[fam]["selected"], "selection_table": plus[fam]["selection_table"],
                         "attacker_val_log_loss": plus[fam]["attacker_val_log_loss"], "cpu_s": plus[fam]["cpu_s"],
                         "surface_record": srec.to_json(),
                         "access": _access("A1", fam, "rep+outputs", contract, a1 + ["clean outputs(attacker_fit)"])})
        ps = att["plus_slate"]
        if list(ps["NL"]) != ["GBT", "MLP", "ignore_rep", "ignore_out"] or list(ps["L"]) != ["L", "ignore_rep",
                                                                                               "ignore_out"]:
            raise ValueError("plus slate changed")
        if ps["select_metric"] != "attacker_val_log_loss":
            raise ValueError("plus slate selection metric must be attacker_val log-loss")
        cand_nl = {"GBT": plus["GBT"]["attacker_val_log_loss"], "MLP": plus["MLP"]["attacker_val_log_loss"],
                   "ignore_rep": orec["val_log_loss"]["NL"], "ignore_out": rep[nlf]["attacker_val_log_loss"]}
        cand_l = {"L": plus["L"]["attacker_val_log_loss"], "ignore_rep": orec["val_log_loss"]["L"],
                  "ignore_out": rep["L"]["attacker_val_log_loss"]}
        selN, selL = select_plus(cand_nl), select_plus(cand_l)
        plus_seeds = {}
        for slot, sel, cands in (("NL", selN, cand_nl), ("Lslate", selL, cand_l)):
            if sel in ("GBT", "MLP", "L"):
                sm, sr = retrain_seeds(sel, plus[sel]["selected"], plus[sel]["model"], Xf, F["s"], E, auth, syn,
                                       whatp)
                for s_, m_ in sm.items():
                    models[("rep+outputs", f"{slot}|as{s_}")] = m_
                src = "fitted on [rep, clean logits]"
            else:
                sr = {"family": sel, "seeds": {str(s_): {"status": "ALIAS_IGNORE_CANDIDATE",
                                                         "alias_of": (f"outputs_only {'NL' if slot == 'NL' else 'L'} "
                                                                      f"as{s_}") if sel == "ignore_rep" else
                                                         (f"C_rep {'NL' if slot == 'NL' else 'L'} as{s_}")}
                                               for s_ in att["attacker_seeds"]}}
                src = sel
            plus_seeds[slot] = {"selected": sel, "candidates_attacker_val_log_loss": cands, "source": src,
                                "seed_retrain": sr}
        recs.append({"surface": "rep+outputs", "recipe": "NL", "status": "EXECUTED", "plus_selection": plus_seeds,
                     "selected": {"candidate": selN}, "access": _access("A1", "NL", "rep+outputs", contract,
                                                                        a1 + ["clean outputs(attacker_fit)"])})

        if info["kind"] == "noise":
            for name, X_src in (("LRT_A2", F["rep"]), ("LRT_A4", F["clean_rep"])):
                c = att[name]
                if c["applies_to"] != "noise" or c["surface"] != "rep" or c["class_prior"] != "attacker_fit":
                    raise ValueError(f"{name} settings not supported")
                exp = "released_class_covariance" if name == "LRT_A2" else "clean_class_covariance"
                if c["floor_trace"] != exp:
                    raise ValueError(f"{name}.floor_trace must be {exp}")
                m = GaussianClassLRT(contract["sigma"], name[-2:], c["eig_floor_rel"], c["cov_ddof"], K_s)
                m.fit(X_src, F["s"], auth=auth, synthetic=syn)
                models[("rep", name)] = m
                recs.append({"surface": "rep", "recipe": name, "status": "EXECUTED", "lrt": m.describe(),
                             "seed_retrain": {"status": "ALIAS_DETERMINISTIC"},
                             "access": _access(name[-2:], name, "rep", contract,
                                               a1 + (["sigma (public mechanism parameter)"] if name == "LRT_A2" else
                                                     ["CLEAN representations(attacker_fit)", "known Sigma"]),
                                               note="defense-informed, released rows only" if name == "LRT_A2" else
                                               "white-box population stress test; never refutes a guarantee")})
    # references
    lo = E["references"]["LO"]
    if lo["fit_role"] != "attacker_fit" or E["references"]["constant"]["prior_role"] != "attacker_fit":
        raise ValueError("references must be fitted on attacker_fit")
    models["LO"] = fit_label_only(F["s"], F["t"], K_s, K_t, lo["laplace_alpha"], auth=auth, synthetic=syn)
    recs.append({"surface": "label_only", "recipe": "LO", "status": "EXECUTED",
                 "access": {"tag": "reference", "fit_inputs": ["y_task(attacker_fit)", "S(attacker_fit)"],
                            "eval_inputs": ["true y_task(assessment)"]}})
    # U2 (shared per identical release)
    u2 = ctx["u2"]()
    recs.append({"surface": "rep", "recipe": "U2", "status": "REUSED" if u2["reused"] else "EXECUTED",
                 "shared_key": u2["key"], "selected": u2["record"]["selected"],
                 "selection_table": u2["record"]["selection_table"], "target": "y_task",
                 "access": {"tag": "utility_probe", "fit_inputs": ["release(attacker_fit)", "y_task(attacker_fit)"],
                            "contract": contract}})
    ctx["_shared_results"] = {"outputs_only": oo if estimable else None, "U2": u2}
    return models, recs, closed, valp


def predict_phase(models: dict, A: dict, ctx: dict) -> dict:
    """Score assessment rows with already selected models. No selection here."""
    K_s, E = ctx["K_s"], ctx["eff"]
    out = {}
    if "G1" in models:
        out["G1_pred"] = linear_predict(models["G1"], A["rep"]).astype(np.float64)
        out["G1_prior"] = np.asarray(models["G1"]["muY"], dtype=np.float64)
        out["G2_pred"] = linear_predict(models["G2"], A["rep"])
        out["G2_prior"] = np.asarray(models["G2"]["muY"], dtype=np.float64)
        if models["RHO1"]["status"] == "OK":
            out["RHO_u"] = np.asarray(A["rep"], dtype=np.float64) @ models["RHO1"]["b"]
            out["RHO_v"] = models["RHO1"]["a"][np.asarray(A["s"]).astype(int)]
    kind = E["surfaces"]["outputs_kind"]
    for key, m in models.items():
        if not isinstance(key, tuple):
            continue
        surf, rec = key
        if rec.startswith("LRT_"):
            P = m.predict_proba(A["rep"])
        else:
            X, _ = build_surface(surf, A["rep"], A["outputs"], kind)
            P = full_proba(m, X, K_s)
        out[f"P__{SURF_KEY[surf]}__{rec.replace('|', '__')}"] = P
    out["LO_P"] = models["LO"][np.asarray(A["t"]).astype(int)]
    return out


# ==================================================================================================================
# one unit
# ==================================================================================================================


def _view(arr: dict, idx) -> dict:
    return {k: (None if v is None else np.asarray(v)[idx]) for k, v in arr.items()}


def run_unit(uid: str, ctx0: dict) -> dict:
    t_wall, t_cpu = time.perf_counter(), time.process_time()
    effp = ctx0["eff_plain"]
    E = Tracked(effp)
    info = parse_unit(uid)
    log = ctx0["log"]
    if uid not in ctx0["registered"]:
        raise BenchRefused(f"REFUSED: {uid} is not registered")
    units_root = require_outside_git(ctx0["root"] / "units", "bench unit outputs")
    out = units_root / uid
    if out.exists():
        raise BenchRefused(f"REFUSED: {out} exists (use --resume; partial dirs are moved aside, never overwritten)")
    inputs = ctx0["inputs"]
    d, k, p, a = info["dataset"], info["seed"], info["purpose"], info["attribute"]
    _panel_check(E, info)
    pidx = inputs.dataset(d)["purposes_index"]
    if pidx and pidx.get(p) != E["datasets"][d]["purposes"][p]["index"]:
        raise BenchRefused(f"REFUSED: {d}: purpose index map {pidx} disagrees with the frozen table")
    D = inputs.dataset(d)
    rc = ctx0.setdefault("_role_checks", {}).get(d) or ctx0["_role_checks"].setdefault(d, role_checks(D, d, E))
    if rc["errors"]:
        raise BenchRefused(f"REFUSED: {d}: role admission errors {rc['errors']}")
    fw = inputs.forward(d, k)
    pk = D["purpose_keys"].get(p, {})
    rk, lk = pk.get("rep", f"rep_p{info['purpose_index']}"), pk.get("logits", f"logits_{p}")
    if rk not in fw or lk not in fw:
        raise BenchRefused(f"REFUSED: {d} s{k} forward cache lacks {rk} or {lk}")
    H, outputs = fw[rk], fw[lk]
    K_s = int(E["datasets"][d]["attr_dims"][a])
    K_t = int(E["datasets"][d]["purposes"][p]["n_task_classes"])
    if outputs.shape[1] != K_t:
        raise BenchRefused(f"REFUSED: {lk} has {outputs.shape[1]} columns, task {p} has {K_t} classes")
    frozen = ctx0["support_frozen"]["cells"].get(info["cell"])
    if not frozen or "sensitive" not in frozen:
        raise BenchRefused(f"REFUSED: no frozen support for {info['cell']}")
    now_sup = cell_support(D, d, p, a, E)
    for w in ("sensitive", "task"):
        if now_sup[w]["counts_per_role"] != frozen[w]["counts_per_role"] or \
                now_sup[w]["supported_classes"] != frozen[w]["supported_classes"]:
            raise BenchRefused(f"REFUSED: {uid}: recomputed {w} support differs from SUPPORT_FROZEN.json")
    support = frozen["sensitive"]
    s_all = D["sens"][a]
    t_all = D["task"][p]
    idx = D["idx"]
    fi, vi, ei = idx["attacker_fit"], idx["attacker_val"], idx["assessment"]
    if E["roles"]["attacker_fit"] != "attacker_fit" or E["roles"]["assessment"] != "assessment" or \
            E["roles"]["attacker_val"] != "attacker_val" or E["roles"]["defense_fit"] != "defense_fit":
        raise ValueError("role names changed")

    if info["arm"] in ("B", "C") and frozen["concept"][info["arm"]]["status"] != "FITTABLE":
        out.mkdir(parents=True)
        (out / "supported.json").write_text(json.dumps({"unit_id": uid, "frozen_at":
                                                        ctx0["support_frozen"]["frozen_at"], **frozen}, indent=1))
        np.savez_compressed(out / "preds.npz", assess_row_id=D["row_id"][ei].astype(np.int64))
        fr = {"unit_id": uid, "info": info, "status": "NE", "ne_reason": "eraser concept below the defense_fit floor",
              "concept_support": frozen["concept"][info["arm"]], "written_at": _now(),
              "effective_protocol_sha256": bench_effective_hash(effp), "n_model_fits": {"total": 0},
              "effective_keys_consumed_in_unit": sorted(E.log),
              "timing": {"wall_s": time.perf_counter() - t_wall, "cpu_s": time.process_time() - t_cpu}}
        _write_json(out / "fit_records.json", fr)
        write_complete(out, uid)
        log(f"[bench] {uid}: NE (concept classes below the defense_fit floor)")
        return {"unit_id": uid, "status": "NE", "timing": fr["timing"], "n_model_fits": 0}
    # alias C -> B: decided from the fitted maps only (no attacker fit)
    ctx = {"eff": E, "auth": ctx0["auth"], "synthetic": ctx0["synthetic"], "inputs": inputs, "maps": ctx0["maps"],
           "shared": ctx0["shared"], "log": log, "info": info, "K_s": K_s, "K_t": K_t, "support": support}
    if info["arm"] == "C":
        mB, recB = get_release_map(ctx, D, d, k, p, [a], "B", H)
        mC, recC = get_release_map(ctx, D, d, k, p, list(E["datasets"][d]["purposes"][p]["disallowed_attrs"]), "C", H)
        al = _defenses().alias_test(mB, mC, atol=E["defenses"]["leace"]["alias_atol"],
                                    H_probe=np.asarray(H[idx["defense_fit"]], dtype=np.float64))
        if al["alias"]:
            src = unit_id(d, k, p, a, "B")
            ok, errs = verify_unit(units_root / src)
            if not ok:
                raise BenchRefused(f"REFUSED: {uid} aliases {src}, which is not complete ({errs}); run {src} first")
            out.mkdir(parents=True)
            (out / "supported.json").write_text(json.dumps({"unit_id": uid, "frozen_at":
                                                            ctx0["support_frozen"]["frozen_at"], **frozen}, indent=1))
            with np.load(units_root / src / "preds.npz") as z:
                np.savez_compressed(out / "preds.npz", assess_row_id=z["assess_row_id"])
            fr = {"unit_id": uid, "info": info, "alias_of": src, "alias_check": al,
                  "maps": {"B": recB["map_id"], "C": recC["map_id"]}, "written_at": _now(),
                  "effective_protocol_sha256": bench_effective_hash(effp), "n_model_fits": {"total": 0},
                  "effective_keys_consumed_in_unit": sorted(E.log),
                  "timing": {"wall_s": time.perf_counter() - t_wall, "cpu_s": time.process_time() - t_cpu}}
            _write_json(out / "fit_records.json", fr)
            write_complete(out, uid)
            log(f"[bench] {uid}: ALIAS of {src} (max|dP| {al['max_abs_P_diff']:.3g})")
            return {"unit_id": uid, "alias_of": src, "timing": fr["timing"], "n_model_fits": 0}
        alias_rec = al
    else:
        alias_rec = None

    R, release_rec, m = build_release(ctx, info, D, H)
    contract = {"noise": "persistent_token" if info["arm"] == "D" else "none", "sigma": info.get("sigma"),
                "seed": info.get("release_seed"), "release_count": "one", "persistent": True,
                "release": release_rec["kind"]}
    ctx["contract"] = contract
    native = native_phase(ctx, info, D, H, s_all, release_rec, m)
    out.mkdir(parents=True)
    (out / "models").mkdir()
    (out / "supported.json").write_text(json.dumps({"unit_id": uid, "frozen_at": ctx0["support_frozen"]["frozen_at"],
                                                    "frozen_before": "any fit or inference", **frozen}, indent=1))
    base = {"row_id": D["row_id"], "rep": R, "outputs": outputs, "s": s_all, "t": t_all,
            "clean_rep": H if info["arm"] == "D" else None}
    Fv, Vv, Av = _view(base, fi), _view(base, vi), _view(base, ei)
    ctx["outputs_only"] = lambda: outputs_only(ctx, info, {k_: Fv[k_] for k_ in ("outputs", "s")},
                                               {k_: Vv[k_] for k_ in ("outputs", "s")},
                                               {"outputs": Av["outputs"], "row_id": Av["row_id"]}, K_s)
    ctx["u2"] = lambda: u2_probe(ctx, info, release_rec["release_key"], {"rep": Fv["rep"], "t": Fv["t"]},
                                 {"rep": Vv["rep"], "t": Vv["t"]}, {"rep": Av["rep"], "row_id": Av["row_id"]}, K_t)
    models, recs, closed, valp = fit_phase({"fit": Fv, "val": Vv}, ctx)
    preds = predict_phase(models, Av, ctx)
    sh = ctx.pop("_shared_results")
    for nm in ("G1", "G2"):
        if f"{nm}_pred" in preds:
            closed[nm]["r2_in_memory_before_save"] = r2_against_prior(s_all[ei], preds[f"{nm}_pred"],
                                                                      preds[f"{nm}_prior"], K_s)
    oo = sh["outputs_only"]
    if oo is not None:
        for kk, v in oo["preds"].items():
            if kk.startswith("P__outputs__"):
                preds[kk] = v
        # ignore candidates: their assessment predictions are exactly the source models' predictions
        preds["P__repPLUSoutputs__ignore_rep"] = oo["preds"]["P__outputs__NL__as0"]
        preds["P__repPLUSoutputs__ignore_out"] = preds["P__rep__NL__as0"]
        sel = next(r for r in recs if r["surface"] == "rep+outputs" and r["recipe"] == "NL")["plus_selection"]
        for slot, srcfam in (("NL", "NL"), ("Lslate", "L")):
            chosen = sel[slot]["selected"]
            for s_ in E["attackers"]["attacker_seeds"]:
                key = f"P__repPLUSoutputs__{slot}__as{s_}"
                if chosen == "ignore_rep":
                    preds[key] = oo["preds"][f"P__outputs__{srcfam}__as{s_}"]
                elif chosen == "ignore_out":
                    preds[key] = preds[f"P__rep__{srcfam}__as{s_}"]
    preds["U2_P"] = sh["U2"]["U2_P"]
    preds.update(assess_row_id=D["row_id"][ei].astype(np.int64), assess_unit=np.asarray(D["units"])[ei],
                 y_s=s_all[ei], y_task=t_all[ei], OUT_logits=outputs[ei],
                 s_prior_fit=np.bincount(s_all[fi], minlength=K_s)[:K_s] / len(fi),
                 t_prior_fit=np.bincount(t_all[fi], minlength=K_t)[:K_t] / len(fi))
    if info["arm"] in E["linear_closed_form"]["cross_cov_heldout"]["arms"]:
        cc = E["linear_closed_form"]["cross_cov_heldout"]
        if cc["score_role"] != "assessment" or list(cc["concepts"]) != ["target", "policy"]:
            raise ValueError("cross-cov held-out settings changed")
        preds["HX"] = R[ei]
        preds["ZP"] = concept_matrix(D, d, E["datasets"][d]["purposes"][p]["disallowed_attrs"], ei, E)
    # U1: frozen head on the transformed representation (hash-pinned checkpoint, weights_only)
    u1cfg = E["utility"]["U1"]
    if u1cfg["checkpoint_load"] != "weights_only":
        raise ValueError("U1 checkpoint load must be weights_only")
    from .forward import frozen_head_logits
    ck = fw["_checkpoint"]
    head_clean, prov = frozen_head_logits(ck["path"], ck["sha256"], p, H[ei])
    diff = float(np.abs(head_clean.astype(np.float64) - outputs[ei]).max())
    u1 = {"checkpoint": prov, "consistency_head_on_clean_vs_stored_logits_max_abs": diff,
          "consistency_ok": diff <= u1cfg["consistency_atol"], "scope": u1cfg["scope"]}
    if not u1["consistency_ok"]:
        raise BenchRefused(f"REFUSED: {uid}: frozen head on clean rep disagrees with stored logits (max |d| {diff})")
    if info["arm"] == "A":
        preds["U1_logits"] = outputs[ei]
        u1["mode"] = "stored clean logits (identity release)"
    else:
        preds["U1_logits"] = frozen_head_logits(ck["path"], ck["sha256"], p, R[ei])[0].astype(np.float64)
        u1["mode"] = "frozen head on transformed representation"

    np.savez_compressed(out / "preds.npz", **{k_: np.asarray(v) for k_, v in preds.items()})
    np.savez_compressed(out / "val_preds.npz", val_row_id=D["row_id"][vi].astype(np.int64), val_y_s=s_all[vi],
                        **valp)
    mf = _save_models(out, {("__".join(k_) if isinstance(k_, tuple) else k_): v for k_, v in models.items()})
    if release_rec.get("map"):
        mf["release_map"] = {"map_id": release_rec["map"]["map_id"],
                             "path_relative_to_private_root": f"defenses/{release_rec['map']['map_id']}"}
    import sklearn
    fr = {"unit_id": uid, "info": info, "tier": tier_of(uid), "written_at": _now(),
          "inputs": {"index": str(inputs.path), "index_sha256": inputs.sha256,
                     "forward_cache_sha256": fw["_forward_sha256"], "checkpoint": ck,
                     "support_frozen_sha256": ctx0["support_frozen_sha256"]},
          "roles": {r: int(len(idx[r])) for r in ROLE_ORDER}, "K_s": K_s, "K_t": K_t,
          "support_status": support["status"], "release": release_rec, "alias_check": alias_rec,
          "contract": contract, "synthetic": ctx0["synthetic"],
          "effective_protocol_sha256": bench_effective_hash(effp), "effective_protocol": effp,
          "native": native, "closed_form": closed, "recipes": recs,
          "shared": {"outputs_only": None if oo is None else {"key": oo["key"], "reused": oo["reused"],
                                                              "preds_sha256": oo["preds_sha256"]},
                     "U2": {"key": sh["U2"]["key"], "reused": sh["U2"]["reused"]}},
          "U1": u1, "model_files": mf, "saved_keys": sorted(preds), "val_saved_keys": sorted(valp),
          "n_model_fits": _count_fits(recs, closed, oo, sh["U2"]),
          "environment": {"OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                          "OMP_NUM_THREADS_required": E["threads"]["OMP_NUM_THREADS"],
                          "python": platform.python_version(), "numpy": np.__version__,
                          "sklearn": sklearn.__version__, "machine": platform.machine()},
          "effective_keys_consumed_in_unit": sorted(E.log),
          "timing": {"wall_s": time.perf_counter() - t_wall, "cpu_s": time.process_time() - t_cpu}}
    _write_json(out / "fit_records.json", fr)
    comp = write_complete(out, uid)
    log(f"[bench] {uid}: done in {fr['timing']['wall_s']:.1f}s wall / {fr['timing']['cpu_s']:.1f}s cpu, "
        f"{fr['n_model_fits']['total']} fits")
    return {"unit_id": uid, "dir": str(out), "timing": fr["timing"], "n_files": len(comp["files"]),
            "n_model_fits": fr["n_model_fits"]["total"]}


def _plain(m):
    """Read every leaf of a Tracked mapping (so that a full-table comparison counts as consumption)."""
    if hasattr(m, "keys") and not isinstance(m, (str, bytes)):
        return {k: _plain(m[k]) for k in m}
    return m


def _panel_check(E, info):
    if _plain(E["datasets"]) != DATASETS:
        raise BenchRefused("REFUSED: effective dataset / purpose tables differ from the frozen PCRL tables")
    P = E["panel"]
    if info["seed"] not in list(P["encoder_seeds"]):
        raise BenchRefused(f"REFUSED: encoder seed {info['seed']} not in the panel")
    t1 = [(c["dataset"], c["purpose"], c["attribute"]) for c in _plain(P["tier1_cells"])]
    t2 = {d: [tuple(x) for x in v] for d, v in _plain(P["tier2_pairs"]).items()}
    cell = (info["dataset"], info["purpose"], info["attribute"])
    if cell not in t1 and (info["purpose"], info["attribute"]) not in t2.get(info["dataset"], []):
        raise BenchRefused(f"REFUSED: cell {cell} not in the registered panel")
    if info["arm"] not in list(P["tier1_arms"]):
        raise BenchRefused(f"REFUSED: arm {info['arm']} not registered")
    S = E["surfaces"]
    if (S["C_rep"], S["C_rep_plus_clean_out"], S["outputs_only"]) != ("rep", "rep+outputs", "outputs"):
        raise ValueError("surface names changed")
    if E["budget"]["cpu_hours"] <= 0 or E["budget"]["inference_reserve_cpu_s_per_unit"] < 0:
        raise ValueError("budget settings invalid")


def _count_fits(recs, closed, oo, u2) -> dict:
    by = {}
    for r in recs:
        if r.get("status") == "EXECUTED" and r.get("selection_table"):
            by[f"{r['surface']}|{r['recipe']}"] = len(r["selection_table"])
        for sr in ([r.get("seed_retrain")] if isinstance(r.get("seed_retrain"), dict) else []) + \
                [v.get("seed_retrain") for v in (r.get("plus_selection") or {}).values()]:
            if sr and "seeds" in sr:
                n = sum(1 for v in sr["seeds"].values() if v.get("status") == "REFIT")
                if n:
                    by[f"{r['surface']}|{r['recipe']}|seed_refits|{sr.get('family')}"] = n
        if r.get("recipe", "").startswith("LRT_") or r.get("recipe") == "LO":
            by[f"{r['surface']}|{r['recipe']}"] = 1
    for nm in ("G1", "RHO1"):
        if nm in closed:
            by[nm] = 1
    if "G2" in closed:
        by["G2"] = len(closed["G2"]["selection_table"])
    if oo is not None and not oo["reused"]:
        by["shared|outputs_only"] = oo["record"]["n_model_fits"]
    if u2 is not None and not u2["reused"]:
        by["shared|U2"] = u2["record"]["n_model_fits"]
    by.pop("rep|U2", None)
    return {"total": int(sum(by.values())), "by_recipe": by}


# ==================================================================================================================
# plan / dry-run / execute / budget
# ==================================================================================================================


def select(tier: str | None, subset: list | None) -> list[str]:
    if tier in (None, "all"):
        expected = all_registered()
    elif str(tier) == "1":
        expected = tier1_units()
    elif str(tier) == "2":
        t2 = tier2_units()
        expected = t2["E1"] + t2["E2"] + t2["E3"]
    else:
        raise BenchRefused(f"REFUSED: unknown tier {tier!r}")
    if subset:
        reg = set(all_registered())
        bad = [u for u in subset if u not in reg]
        if bad:
            raise BenchRefused(f"REFUSED: --units contains unregistered IDs {bad}")
        outside = [u for u in subset if u not in expected]
        if outside:
            raise BenchRefused(f"REFUSED: --units {outside} are not in tier {tier}")
        expected = [u for u in expected if u in subset]
    return expected


def plan(inputs: Inputs, tier, subset) -> dict:
    exp = select(tier, subset)
    runnable, missing = [], []
    status = {}
    for u in exp:
        i = parse_unit(u)
        key = (i["dataset"], i["seed"])
        if key not in status:
            status[key] = inputs.seed_status(*key)
        if status[key]["ok"]:
            runnable.append(u)
        else:
            missing.append({"unit_id": u, "reason": status[key]["reason"]})
    maps = sorted({map_id(parse_unit(u)["dataset"], parse_unit(u)["seed"], parse_unit(u)["purpose"],
                          ("B_" + parse_unit(u)["attribute"]) if parse_unit(u)["arm"] == "B" else
                          "C_" + "+".join(parse_unit(u)["policy_set"]))
                   for u in runnable if parse_unit(u)["arm"] in ("B", "C")})
    oo = sorted({f"{i['dataset']}__s{i['seed']}__{i['purpose']}__{i['attribute']}" for i in map(parse_unit, runnable)})
    return {"mode": "plan", "tier": tier, "n_expected": len(exp), "expected": exp, "runnable": runnable,
            "n_runnable": len(runnable), "missing": missing,
            "missing_seeds": {f"{d}__s{k}": v for (d, k), v in status.items() if not v["ok"]},
            "shared_work": {"leace_maps": maps, "n_leace_maps": len(maps), "outputs_only": oo,
                            "n_outputs_only": len(oo)},
            "status": "OK" if not missing else "PARTIAL (missing units reported; available units continue)",
            "fits_performed": 0}


def _context(args, root: Path, inputs: Inputs, auth, eff_plain, log) -> dict:
    sf = root / "support" / "SUPPORT_FROZEN.json"
    if not sf.exists():
        raise BenchRefused(f"REFUSED: {sf} missing (frozen at lock build, before any fit)")
    return {"root": root, "inputs": inputs, "auth": auth, "synthetic": inputs.synthetic, "eff_plain": eff_plain,
            "maps": MapStore(root), "shared": SharedStore(root), "log": log, "registered": set(all_registered()),
            "support_frozen": json.loads(sf.read_text()), "support_frozen_sha256": sha256_file(sf)}


def dry_run(inputs: Inputs, root: Path, tier, subset, eff_plain) -> dict:
    E = Tracked(eff_plain)
    pl = plan(inputs, tier, subset)
    mism = inputs.verify_hashes()
    ds_checks = {}
    for d in inputs.datasets():
        D = inputs.dataset(d)
        ds_checks[d] = {**role_checks(D, d, E), "n_rows": int(len(D["row_id"])),
                        "test_rule": D["test_rule"], "n_test_rows": int(D["test_mask"].sum()),
                        "purposes_index": D["purposes_index"],
                        "purpose_index_matches_frozen": all(
                            D["purposes_index"].get(p) == E["datasets"][d]["purposes"][p]["index"]
                            for p in E["datasets"][d]["purposes"]) if D["purposes_index"] else None,
                        "sensitive_present": sorted(D["sens"]), "task_present": sorted(D["task"])}
    sf = root / "support" / "SUPPORT_FROZEN.json"
    frozen = json.loads(sf.read_text()) if sf.exists() else None
    cells = {}
    for d, p, a in cells_of(pl["runnable"]):
        now = cell_support(inputs.dataset(d), d, p, a, E)
        fz = (frozen or {}).get("cells", {}).get(f"{d}__{p}__{a}")
        cells[f"{d}__{p}__{a}"] = {"supported_classes": now["sensitive"]["supported_classes"],
                                   "status": now["sensitive"]["status"],
                                   "unsupported": now["sensitive"]["unsupported_classes"],
                                   "task_supported": now["task"]["supported_classes"],
                                   "matches_frozen": None if fz is None else
                                   (fz["sensitive"]["supported_classes"] == now["sensitive"]["supported_classes"])}
    A = E["attackers"]
    n_slate = len(A["L"]["C"]) + len(A["GBT"]["configs"]) + \
        len(A["MLP"]["hidden_layer_sizes"]) * len(A["MLP"]["alpha"]) * len(A["MLP"]["learning_rate_init"])
    rows = []
    for u in pl["runnable"]:
        i = parse_unit(u)
        rows.append({"unit_id": u, "arm": i["arm_tag"], "contract": "persistent_token" if i["arm"] == "D" else "none",
                     "surfaces": ["C_rep", "C_rep_plus_clean_out", "outputs_only (shared)"],
                     "planned_fits_upper": 2 * n_slate + 2 * 2 + len(E["linear_closed_form"]["G2"]["rho_grid"]) + 3
                     + (2 if i["arm"] == "D" else 0),
                     "release": {"A": "identity", "B": "official LEACE (target one-hot), defense_fit",
                                 "C": "official LEACE (policy one-hots), defense_fit; alias check vs B",
                                 "D": f"gaussian sigma={i.get('sigma')} rs={i.get('release_seed')} persistent"}[i["arm"]]})
    return {"mode": "dry-run", "fits_performed": 0, "plan": {k: pl[k] for k in ("n_expected", "n_runnable",
                                                                                 "missing", "missing_seeds",
                                                                                 "shared_work", "status")},
            "input_hash_mismatches": mism, "datasets": ds_checks, "support": cells,
            "support_frozen_present": frozen is not None, "units": rows,
            "consumption_dry_run": sorted(E.log)}


def _ledger_path(root: Path) -> Path:
    return root / "logs" / "BUDGET_LEDGER.json"


def ledger(root: Path) -> dict:
    p = _ledger_path(root)
    return json.loads(p.read_text()) if p.exists() else {"schema": "bench_budget_ledger/v1", "entries": [],
                                                         "stops": []}


def spent_cpu_s(root: Path) -> dict:
    """Science CPU spent so far: every complete unit's fit_records timing + sigma* + sanity + inference records."""
    out = {"units": 0.0, "other": 0.0, "n_units": 0}
    ud = root / "units"
    if ud.exists():
        for d in ud.iterdir():
            f = d / "fit_records.json"
            if d.is_dir() and f.exists() and (d / "COMPLETE.json").exists():
                out["units"] += float(json.loads(f.read_text())["timing"]["cpu_s"])
                out["n_units"] += 1
    for e in ledger(root).get("entries", []):
        if e.get("kind") in ("sigma_star", "sanity", "infer"):
            out["other"] += float(e.get("cpu_s", 0.0))
    out["total"] = out["units"] + out["other"]
    return out


def projected_unit_cpu(root: Path, uid: str, calib: dict | None) -> float:
    i = parse_unit(uid)
    kind = i["arm"]
    vals = []
    ud = root / "units"
    if ud.exists():
        for d in ud.iterdir():
            f = d / "fit_records.json"
            if d.is_dir() and f.exists():
                try:
                    j = parse_unit(d.name)
                except ValueError:
                    continue
                if j["dataset"] == i["dataset"] and j["arm"] == kind:
                    vals.append(float(json.loads(f.read_text())["timing"]["cpu_s"]))
    if vals:
        return float(np.mean(vals))
    if calib:
        return float(calib.get("per_unit_cpu_s", {}).get(i["dataset"], {}).get(kind, 60.0))
    return 60.0


def tier2_gate(root: Path, inputs: Inputs) -> dict:
    """Technical validity only (never a result's sign or significance)."""
    reasons = []
    for u in tier1_units():
        i = parse_unit(u)
        if not inputs.seed_status(i["dataset"], i["seed"])["ok"]:
            continue
        d = root / "units" / u
        ok, errs = verify_unit(d) if d.exists() else (False, ["missing"])
        if not ok:
            reasons.append(f"{u}: {errs}")
            continue
        fr = json.loads((d / "fit_records.json").read_text())
        if fr.get("alias_of"):
            continue
        nat = fr.get("native", {})
        if i["arm"] == "A" and nat.get("reproduction", {}).get("status") not in ("REPRODUCED",):
            reasons.append(f"{u}: A native reproduction {nat.get('reproduction', {}).get('status')}")
        if i["arm"] in ("B", "C") and nat.get("status") != "PASS":
            reasons.append(f"{u}: {i['arm']} native check {nat.get('status')}")
        if not fr.get("U1", {}).get("consistency_ok"):
            reasons.append(f"{u}: U1 consistency failed")
    if not (root / "infer" / "SIGMA_STAR.json").exists():
        reasons.append("SIGMA_STAR.json missing (run --sigma-star after Tier 1)")
    return {"open": not reasons, "reasons": reasons}


def execute(ctx0: dict, units: list[str], *, resume: bool, tier, calib: dict | None) -> dict:
    root = ctx0["root"]
    log = ctx0["log"]
    (root / "units").mkdir(parents=True, exist_ok=True)
    budget_s = float(ctx0["eff_plain"]["budget"]["cpu_hours"]) * 3600.0
    reserve = float(ctx0["eff_plain"]["budget"]["inference_reserve_cpu_s_per_unit"])
    done, skipped, moved, missing, stopped = [], [], [], [], None
    led = ledger(root)
    for u in units:
        i = parse_unit(u)
        st = ctx0["inputs"].seed_status(i["dataset"], i["seed"])
        if not st["ok"]:
            missing.append({"unit_id": u, "reason": st["reason"]})
            continue
        d = root / "units" / u
        if d.exists():
            ok, errs = verify_unit(d)
            if ok and resume:
                skipped.append(u)
                continue
            if ok:
                raise BenchRefused(f"REFUSED: {u} already complete; pass --resume to skip it")
            if (d / "COMPLETE.json").exists():
                raise BenchRefused(f"REFUSED: {u} COMPLETE.json present but hashes fail ({errs}); investigate")
            if not resume:
                raise BenchRefused(f"REFUSED: partial outputs in {d}; pass --resume to move them aside")
            aside = d.with_name(f"{u}.partial-{_dt.datetime.now(_dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
            shutil.move(str(d), str(aside))
            moved.append(str(aside))
        if str(tier) == "2":
            sp = spent_cpu_s(root)
            proj = projected_unit_cpu(root, u, calib)
            need = sp["total"] + proj + reserve * (sp["n_units"] + 1)
            if need > budget_s:
                stopped = {"at_unit": u, "spent_cpu_s": sp["total"], "projected_unit_cpu_s": proj,
                           "inference_reserve_cpu_s": reserve * (sp["n_units"] + 1), "budget_cpu_s": budget_s,
                           "rule": ctx0["eff_plain"]["budget"]["stop_rule"], "at": _now()}
                led["stops"].append(stopped)
                _write_json(_ledger_path(root), led)
                log(f"[bench] budget stop before {u}: projected {need:.0f}s > {budget_s:.0f}s")
                break
        r = run_unit(u, ctx0)
        done.append(r)
        led["entries"].append({"kind": "unit", "unit_id": u, "tier": tier_of(u), "at": _now(),
                               "cpu_s": r["timing"]["cpu_s"], "wall_s": r["timing"]["wall_s"]})
        _write_json(_ledger_path(root), led)
        with open(root / "logs" / "RUN_LOG.jsonl", "a") as f:
            f.write(json.dumps({"at": _now(), **{k: v for k, v in r.items() if k != "dir"}}) + "\n")
    return {"mode": "execute", "tier": tier, "completed": done, "skipped_complete": skipped,
            "moved_partial": moved, "missing": missing, "budget_stop": stopped,
            "fits_performed": int(sum(r["n_model_fits"] for r in done)),
            "spent_cpu_s": spent_cpu_s(root)}


# ==================================================================================================================
# sigma* (attacker_val only) and the shuffled-label sanity
# ==================================================================================================================


def sigma_star(root: Path, inputs: Inputs, eff_plain=None) -> dict:
    """Smallest sigma whose attacker_val C_rep NL macro AUC (mean over encoder seeds x release seeds, attacker seed 0)
    is <= bar. Reads ONLY val_preds.npz (and the frozen support) of Tier-1 noise units."""
    from .metrics import auc_binary
    t0 = time.process_time()
    E = Tracked(BENCH_EFFECTIVE if eff_plain is None else eff_plain)
    S = E["sigma_star"]
    if S["role"] != "attacker_val" or int(S["attacker_seed"]) != 0:
        raise ValueError("sigma* must use attacker_val, attacker seed 0")
    sf = json.loads((root / "support" / "SUPPORT_FROZEN.json").read_text())
    out = {"schema": "bench_sigma_star/v1", "rule": S["rule"], "bar": S["bar"], "computed_at": _now(),
           "fallback_sigma": S["fallback_sigma"],
           "datasets": {}, "inputs_read": []}
    for d, p, a in [(c["dataset"], c["purpose"], c["attribute"]) for c in E["panel"]["tier1_cells"]]:
        classes = sf["cells"][f"{d}__{p}__{a}"]["sensitive"]["supported_classes"]
        curve, per = {}, {}
        for sig in E["defenses"]["noise"]["sigmas"]:
            vals = []
            for k in E["panel"]["encoder_seeds"]:
                if not inputs.seed_status(d, k)["ok"]:
                    continue
                for rs in E["defenses"]["noise"]["release_seeds"]:
                    u = unit_id(d, k, p, a, f"D_sigma{sig:g}_rs{rs}")
                    ud = root / "units" / u
                    comp = ud / "COMPLETE.json"
                    if not comp.exists():
                        raise BenchRefused(f"REFUSED: sigma*: {u} not complete")
                    h = json.loads(comp.read_text())["files"].get("val_preds.npz")
                    if h is None or sha256_file(ud / "val_preds.npz") != h:
                        raise BenchRefused(f"REFUSED: sigma*: {u} val_preds.npz hash mismatch")
                    out["inputs_read"].append({"unit": u, "file": "val_preds.npz", "sha256": h})
                    with np.load(ud / "val_preds.npz", allow_pickle=False) as z:
                        y, P = z["val_y_s"], z["VAL__rep__NL__as0"]
                    if len(classes) < 2:
                        vals.append(np.nan)
                        continue
                    vals.append(float(np.mean([auc_binary(y == c, P[:, c]) for c in classes])))
                    per[u] = vals[-1]
            curve[f"{sig:g}"] = float(np.mean(vals)) if vals and np.all(np.isfinite(vals)) else None
        chosen, flagged = None, False
        for sig in E["defenses"]["noise"]["sigmas"]:
            v = curve[f"{sig:g}"]
            if v is not None and v <= S["bar"]:
                chosen = float(sig)
                break
        if chosen is None:
            chosen, flagged = float(S["fallback_sigma"]), True
        out["datasets"][d] = {"cell": f"{d}__{p}__{a}", "sigma_star": chosen, "flagged_fallback": flagged,
                              "val_curve_mean": curve, "per_unit_val_auc": per, "supported_classes": classes}
    out["cpu_s"] = time.process_time() - t0
    out["consumed"] = sorted(E.log)
    return out


def shuffled_label_sanity(ctx0: dict, units: list[str]) -> dict:
    """Fit/validation-only shuffled-label sanity through the same release and slate code. Assessment rows never enter."""
    E = Tracked(ctx0["eff_plain"])
    sc = E["sanity"]
    res = {}
    for uid in units:
        t0 = time.process_time()
        info = parse_unit(uid)
        d, k, p, a = info["dataset"], info["seed"], info["purpose"], info["attribute"]
        D = ctx0["inputs"].dataset(d)
        fw = ctx0["inputs"].forward(d, k)
        H = fw[D["purpose_keys"].get(p, {}).get("rep", f"rep_p{info['purpose_index']}")]
        K_s = int(E["datasets"][d]["attr_dims"][a])
        ctx = {"eff": E, "auth": ctx0["auth"], "synthetic": ctx0["synthetic"], "inputs": ctx0["inputs"],
               "maps": ctx0["maps"], "shared": ctx0["shared"], "log": ctx0["log"], "info": info}
        R, rel, _ = build_release(ctx, info, D, H)
        rng = np.random.default_rng(int(sc["shuffle_seed"]))
        fi, vi = D["idx"]["attacker_fit"], D["idx"]["attacker_val"]
        yf, yv = rng.permutation(D["sens"][a][fi]), rng.permutation(D["sens"][a][vi])
        slate = fit_slate(R[fi], yf, R[vi], yv, K_s, E, ctx0["auth"], ctx0["synthetic"], f"{uid}/sanity")
        nlf = slate["NL"]["selected_family"]
        from .metrics import auc_binary
        classes = ctx0["support_frozen"]["cells"][info["cell"]]["sensitive"]["supported_classes"]

        def vauc(m):
            P = full_proba(m, R[vi], K_s)
            return float(np.mean([auc_binary(yv == c, P[:, c]) for c in classes])) if len(classes) >= 2 else None
        r = {"unit": uid, "release": rel.get("kind"), "rows": {"attacker_fit": int(len(fi)), "attacker_val":
                                                                int(len(vi))},
             "assessment_rows_used": 0, "val_auc_shuffled": {"NL": vauc(slate[nlf]["model"]),
                                                             "L": vauc(slate["L"]["model"])},
             "nl_selected": nlf, "note": "selected and scored on the same shuffled validation rows: optimistic by "
                                         "selection; a value near 0.5 is the expected outcome"}
        r["flag"] = (r["val_auc_shuffled"]["NL"] or 0) > sc["flag_above_auc"]
        r["cpu_s"] = time.process_time() - t0
        res[uid] = r
    return {"mode": "shuffled-label-sanity", "units": res, "consumed": sorted(E.log)}


# ==================================================================================================================
# synthetic world (tests and calibration only; never used for scientific outputs)
# ==================================================================================================================


def _synthetic_checkpoint(path: Path, d: int, heads: list, input_dim: int = 12, hidden: int = 16, seed: int = 0):
    """PCRL v2 layout accepted by forward.FrozenPCRLv2; heads exactly linear in the representation."""
    import torch
    g = torch.Generator().manual_seed(seed)
    bb = {}
    dims = [input_dim, hidden, hidden]
    for li, i in enumerate((0, 4)):
        bb[f"network.{i}.weight"] = torch.randn(dims[li + 1], dims[li], generator=g) * 0.1
        bb[f"network.{i}.bias"] = torch.zeros(dims[li + 1])
        bn = i + 1
        bb[f"network.{bn}.weight"] = torch.ones(dims[li + 1])
        bb[f"network.{bn}.bias"] = torch.zeros(dims[li + 1])
        bb[f"network.{bn}.running_mean"] = torch.zeros(dims[li + 1])
        bb[f"network.{bn}.running_var"] = torch.ones(dims[li + 1])
        bb[f"network.{bn}.num_batches_tracked"] = torch.tensor(1)
    bb["repr_proj.weight"] = torch.randn(d, hidden, generator=g) * 0.1
    bb["repr_proj.bias"] = torch.zeros(d)
    ad = {}
    ins, outs = [input_dim, hidden, hidden], [hidden, hidden, d]
    for p in range(len(heads)):
        for l in range(3):
            ad[f"{p}.{l}.A.weight"] = torch.zeros(2, ins[l])
            ad[f"{p}.{l}.B.weight"] = torch.zeros(outs[l], 2)
            ad[f"{p}.{l}.bias"] = torch.zeros(outs[l])
    th, A = {}, {}
    for j, (name, C) in enumerate(heads):
        a = np.zeros((C, d), dtype=np.float32)
        a[:, (1 + j) % d] = np.linspace(-2.0, 2.0, C)
        b = np.zeros(C, dtype=np.float32)
        th[f"{name}.network.0.weight"] = torch.cat([torch.eye(d), -torch.eye(d)])
        th[f"{name}.network.0.bias"] = torch.zeros(2 * d)
        th[f"{name}.network.3.weight"] = torch.as_tensor(np.c_[a, -a])
        th[f"{name}.network.3.bias"] = torch.as_tensor(b)
        A[name] = (a, b)
    ck = {"backbone": bb, "lora_adapters": ad, "task_heads": th,
          "config": {"lora_rank": 2, "lora_alpha": 4.0}, "state": {"epoch": 204}}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, path)
    return sha256_file(path), A


def make_bench_world(root: Path, *, n_test: dict | None = None, n_train: dict | None = None, d: int = 8,
                     seeds=(0,), failed_seeds=(), signal: dict | None = None, role_counts: dict | None = None,
                     ethnicity_from_race: bool = False, absent_assessment_class: tuple | None = None,
                     rep_scale: float = 1.0, seed: int = 0, datasets=("adult", "hmda"), n_excluded: int = 5,
                     small_defense_class: tuple | None = None, race_binary_alias: bool = False) -> dict:
    """Synthetic world written in the role-1 INPUTS_INDEX schema (bench_v1.inputs_index/v1).

    signal[dataset][attr] in {'direct', 'null', 'xor', 'contrast'} acts on the purpose-0 representation. Heads are
    linear, so U1 consistency is exact. Roles: train rows -> defense_fit (preferred route), test rows -> attacker
    roles (HMDA by the registered hash rule, Adult by the pilot hash rule or exact role_counts); n_excluded train rows
    duplicate a test record and are marked excluded_dup (they must be dropped)."""
    root = Path(root)
    inp = root / "inputs"
    inp.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_test = n_test or {"adult": 1500, "hmda": 1500}
    n_train = n_train or {"adult": 800, "hmda": 800}
    signal = signal if signal is not None else {"adult": {"sex": "direct"}, "hmda": {"race": "direct"}}
    index = {"schema": "bench_v1.inputs_index/v1", "synthetic": True, "built_at": _now(),
             "role_values": list(ROLE_ORDER) + ["excluded_dup", "unused_train"], "datasets": {}}
    for ds in datasets:
        nt, nr = n_test[ds], n_train[ds]
        n = nt + nr
        row_id = np.arange(n, dtype=np.int64)
        split = np.array(["test"] * nt + ["train"] * nr)
        attrs = DATASETS[ds]["attr_dims"]
        sens = {}
        for a, K in attrs.items():
            pr = np.r_[0.4, 0.3, np.full(K - 2, 0.3 / (K - 2))] if K > 2 else np.array([0.55, 0.45])
            sens[a] = rng.choice(K, n, p=pr / pr.sum()).astype(np.int64)
        if ds == "hmda" and ethnicity_from_race:
            sens["ethnicity"] = (sens["race"] >= 2).astype(np.int64)
        if ds == "hmda" and race_binary_alias:      # span(onehot(race)) == span(onehot(ethnicity)): B == C exactly
            sens["race"] = (sens["race"] % 2).astype(np.int64)
            sens["ethnicity"] = sens["race"].copy()
        if small_defense_class and small_defense_class[0] == ds:
            _, a_s, c_s = small_defense_class
            m = (split == "train") & (sens[a_s] == c_s)
            sens[a_s][np.flatnonzero(m)[50:]] = 0
        rec = np.array([hashlib.sha256(f"{ds}|{i}".encode()).hexdigest()[:20] for i in range(n)])
        rec[1:nt:97] = rec[0:nt - 1:97][: len(rec[1:nt:97])]               # ~1% exact duplicate records in test
        rec[nt:nt + n_excluded] = rec[2:2 + n_excluded]                    # train rows duplicating a test record
        _, unit = np.unique(rec, return_inverse=True)
        roles = np.empty(n, dtype=object)
        roles[split == "train"] = "defense_fit"
        roles[nt:nt + n_excluded] = "excluded_dup"
        test_idx = np.flatnonzero(split == "test")
        if role_counts and ds in role_counts:
            perm = rng.permutation(np.unique(unit[test_idx]))
            c = role_counts[ds]
            cut1 = int(round(len(perm) * c[0] / sum(c)))
            cut2 = cut1 + int(round(len(perm) * c[1] / sum(c)))
            rmap = {uu: ("attacker_fit" if j < cut1 else ("attacker_val" if j < cut2 else "assessment"))
                    for j, uu in enumerate(perm)}
            for i in test_idx:
                roles[i] = rmap[unit[i]]
        else:
            salt = "bench-roles-hmda-v1|" if ds == "hmda" else "pilot-roles-v1|"
            for i in test_idx:
                u = _uniform(salt, rec[i])
                roles[i] = "attacker_fit" if u < 0.5 else ("attacker_val" if u < 0.65 else "assessment")
        roles = roles.astype(str)
        if absent_assessment_class and absent_assessment_class[0] == ds:
            _, a_abs, c_abs = absent_assessment_class
            sens[a_abs][(roles == "assessment") & (sens[a_abs] == c_abs)] = 0
        purposes = DATASETS[ds]["purposes"]
        heads = sorted(((p, v["n_task_classes"], v["index"]) for p, v in purposes.items()), key=lambda x: x[2])
        encoders, y_task = {}, None
        for k in sorted(set(seeds) | set(failed_seeds)):
            ck = root / "checkpoints" / f"{ds}_s{k}.pt"
            ck_sha, A = _synthetic_checkpoint(ck, d, [(p, C) for p, C, _ in heads], seed=k)
            g = np.random.default_rng(1000 + k + (0 if ds == "adult" else 500))
            reps = {}
            for p, C, pi in heads:
                Hm = g.normal(size=(n, d))
                if pi == 0:
                    for a, kind in (signal.get(ds) or {}).items():
                        y = sens[a]
                        if kind == "direct":
                            Hm[:, 0] += 1.2 * (y - y.mean()) / max(y.std(), 1e-9)
                        elif kind == "xor":
                            sgn = np.where(y % 2 == 1, 1.0, -1.0)
                            h0 = g.choice([-1.0, 1.0], n)
                            Hm[:, 0] = h0 * (np.abs(g.normal(size=n)) + 0.3)
                            Hm[:, 1] = sgn * h0 * (np.abs(g.normal(size=n)) + 0.3)
                        elif kind == "contrast":
                            Hm[:, 0] += 2.0 * ((y == attrs[a] - 1).astype(float) - (y == attrs[a] - 2).astype(float))
                reps[p] = (rep_scale * Hm).astype(np.float32).astype(np.float64)
            logits = {p: (reps[p].astype(np.float32) @ A[p][0].T + A[p][1]).astype(np.float32).astype(np.float64)
                      for p, _, _ in heads}
            if y_task is None:
                y_task = {}
                for p, C, pi in heads:
                    L = logits[p] - logits[p].max(1, keepdims=True)
                    P = np.exp(L)
                    P /= P.sum(1, keepdims=True)
                    y_task[p] = np.minimum((P.cumsum(1) < rng.random(n)[:, None]).sum(1), C - 1).astype(np.int64)
            fwd = inp / f"{ds}_s{k}_forward.npz"
            np.savez(fwd, row_id=row_id, split=split, **{f"rep_p{pi}": reps[p] for p, C, pi in heads},
                     **{f"logits_{p}": logits[p] for p, C, pi in heads})
            encoders[str(k)] = {"checkpoint": str(ck), "checkpoint_sha256": ck_sha,
                                "lineage_status": "ADMITTED" if k not in failed_seeds else "FAILED",
                                "lineage_reason": None if k not in failed_seeds else
                                f"synthetic: lineage check failed for {ck}",
                                "forward_npz": str(fwd), "forward_sha256": sha256_file(fwd)}
        lab = {"row_id": row_id, "split": split, "unit": unit.astype(np.int64), "record_key": rec, "canon_key": rec,
               "role": roles, **sens, **{f"task_{purposes[p]['task']}": y for p, y in y_task.items()}}
        np.savez(inp / f"{ds}_labels.npz", **lab)
        np.savez(inp / f"{ds}_rows.npz", **{k: v for k, v in lab.items() if k != "role"}, test_role=roles)
        np.savez(inp / f"{ds}_features.npz", row_id=row_id, features=rng.normal(size=(n, 12)).astype(np.float32))
        rl = {}
        for r in list(ROLE_ORDER) + ["excluded_dup"]:
            m = roles == r
            rl[f"{r}__row_id"] = row_id[m]
            if r in ROLE_ORDER:
                rl[f"{r}__unit"] = unit[m].astype(np.int64)
        np.savez(inp / f"{ds}_roles.npz", **rl)
        tm = split == "test"
        audit = {"dataset": ds, "seeds": list(seeds), "per_seed": {}}
        for k in seeds:
            with np.load(inp / f"{ds}_s{k}_forward.npz") as z:
                rows = [{"purpose": p, "attribute": a, "r2_onehot": historical_native_r2(
                    z[f"rep_p{v['index']}"][tm], sens[a][tm], 1e-6, "historical_mixed_precision")["clamped"]}
                        for p, v in purposes.items() for a in v["disallowed_attrs"]]
            audit["per_seed"][str(k)] = {"epoch": 204, "rows": rows}
        (inp / f"{ds}_dominant_axis_audit.json").write_text(json.dumps(audit))
        entry = {"defense_route": "PREFERRED",
                 "purposes": {p: {"index": v["index"], "task": v["task"], "task_dim": v["n_task_classes"],
                                  "disallowed_attrs": v["disallowed_attrs"], "rep_key": f"rep_p{v['index']}",
                                  "logits_key": f"logits_{p}", "labels_task_key": f"task_{v['task']}"}
                              for p, v in purposes.items()},
                 "encoders": encoders,
                 "historical_native_json": f"{ds}_dominant_axis_audit.json",
                 "historical_native_sha256": sha256_file(inp / f"{ds}_dominant_axis_audit.json")}
        for name in ("labels", "roles", "rows", "features"):
            entry[f"{name}_npz"] = str(inp / f"{ds}_{name}.npz")
            entry[f"{name}_sha256"] = sha256_file(inp / f"{ds}_{name}.npz")
        index["datasets"][ds] = entry
    (inp / "INPUTS_INDEX.json").write_text(json.dumps(index, indent=1))
    return {"root": root, "index": inp / "INPUTS_INDEX.json"}


# ==================================================================================================================
# CLI
# ==================================================================================================================


def build_parser():
    p = argparse.ArgumentParser(prog="stored_model_eval bench", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group()
    for flag in ("--plan", "--dry-run", "--execute-scientific-fits", "--sigma-star", "--shuffled-label-sanity",
                 "--lock-build", "--lock-verify", "--infer", "--report", "--calibrate", "--freeze-support"):
        g.add_argument(flag, action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--units", default=None, help="comma-separated registered unit IDs")
    p.add_argument("--tier", default=None, choices=["1", "2", "all"])
    p.add_argument("--private-root", default=None, help=f"default {DEFAULT_PRIVATE}")
    p.add_argument("--index", default=None, help="default <private-root>/inputs/INPUTS_INDEX.json")
    p.add_argument("--lock", default=None, help=f"default <worktree>/{PKG_REL}/LOCK.json")
    p.add_argument("--package-dir", default=None, help=f"default <worktree>/{PKG_REL}")
    p.add_argument("--calibration", default=None, help=f"default <worktree>/{PKG_REL}/notes/evaluation/calibration.json")
    p.add_argument("--synthetic", action="store_true", help="inputs are synthetic fixtures (tests/calibration)")
    p.add_argument("--effective-override", default=None,
                   help="(synthetic only) JSON deep-merged over the effective protocol (test grids)")
    p.add_argument("--worktree", default=None)
    p.add_argument("--allow-other-branch-for-tests", action="store_true")
    p.add_argument("--calibration-out", default=None)
    return p


def _deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def effective_for(args) -> dict:
    eff = copy.deepcopy(BENCH_EFFECTIVE)
    if args.effective_override:
        if not args.synthetic:
            raise BenchRefused("REFUSED: --effective-override is only allowed with --synthetic")
        eff = _deep_merge(eff, json.loads(Path(args.effective_override).read_text()))
    return eff


def paths(args) -> dict:
    wt = resolve_worktree(args.worktree, args.allow_other_branch_for_tests)
    root = Path(args.private_root).expanduser() if args.private_root else DEFAULT_PRIVATE
    root = require_outside_git(root, "bench private root")
    pkg = Path(args.package_dir).expanduser() if args.package_dir else Path(wt["root"]) / PKG_REL
    return {"wt": wt, "root": root,
            "index": Path(args.index).expanduser() if args.index else root / "inputs" / "INPUTS_INDEX.json",
            "lock": Path(args.lock).expanduser() if args.lock else Path(wt["root"]) / PKG_REL / "LOCK.json",
            "pkg": pkg,
            "calibration": Path(args.calibration).expanduser() if args.calibration else
            Path(wt["root"]) / PKG_REL / "notes" / "evaluation" / "calibration.json"}


def _log(m):
    print(m, file=sys.stderr, flush=True)


def _auth(args, inputs: Inputs) -> FitAuthorization:
    if inputs.synthetic != bool(args.synthetic):
        raise BenchRefused(f"REFUSED: index synthetic={inputs.synthetic} but --synthetic={bool(args.synthetic)}")
    return FitAuthorization(synthetic=inputs.synthetic, execute_scientific_fits=not inputs.synthetic)


def _require_lock(P, eff, inputs) -> dict:
    from .bench_lock import verify_bench_lock
    if not P["lock"].exists():
        raise BenchRefused(f"REFUSED: lock {P['lock']} not found (bench --lock-build first)")
    v = verify_bench_lock(P["lock"], Path(P["wt"]["root"]), P["index"], P["root"], eff)
    if not v["ok"]:
        raise BenchRefused("REFUSED: lock verification failed:\n  " + "\n  ".join(v["mismatches"]))
    return v


def main(argv=None) -> dict:
    a = build_parser().parse_args(argv)
    P = paths(a)
    eff = effective_for(a)
    errs = validate_bench_effective(eff) if not a.effective_override else []
    if errs:
        raise BenchRefused(f"REFUSED: effective protocol invalid: {errs}")
    subset = [u.strip() for u in a.units.split(",")] if a.units else None
    if a.lock_build or a.lock_verify:
        from .bench_lock import build_bench_lock, verify_bench_lock
        if a.lock_build:
            return build_bench_lock(Path(P["wt"]["root"]), P["index"], P["root"], P["lock"], eff,
                                    effective_out=P["pkg"] / "EFFECTIVE_PROTOCOL.json")
        v = verify_bench_lock(P["lock"], Path(P["wt"]["root"]), P["index"], P["root"], eff)
        if not v["ok"]:
            raise BenchRefused("REFUSED: lock verification failed:\n  " + "\n  ".join(v["mismatches"]))
        return v
    if a.infer:
        from .bench_infer import infer_bench
        t0 = time.process_time()
        res = infer_bench(P["root"], eff=eff, tier=a.tier)
        out = P["root"] / "infer" / "BENCH_INFER.json"
        _write_json(out, res)
        led = ledger(P["root"])
        led["entries"].append({"kind": "infer", "at": _now(), "cpu_s": time.process_time() - t0})
        _write_json(_ledger_path(P["root"]), led)
        return {"mode": "infer", "written": str(out), "n_exploratory": len(res["exploratory"]),
                "n_primary": len(res["primary"]["endpoints"]), "cpu_s": time.process_time() - t0}
    if a.report:
        from .bench_report import report_bench
        return report_bench(P["root"], P["pkg"])
    if a.calibrate:
        from .bench_report import calibrate
        return calibrate(P, eff, out=Path(a.calibration_out) if a.calibration_out else P["calibration"])
    inputs = Inputs(P["index"])
    if a.freeze_support:
        sf = P["root"] / "support" / "SUPPORT_FROZEN.json"
        if sf.exists():
            raise BenchRefused(f"REFUSED: {sf} exists (support is frozen once)")
        _write_json(sf, freeze_all_support(inputs, eff))
        return {"mode": "freeze-support", "written": str(sf)}
    if a.plan:
        return {"worktree": P["wt"], **plan(inputs, a.tier or "1", subset)}
    if a.sigma_star:
        _require_lock(P, eff, inputs)
        res = sigma_star(P["root"], inputs, eff)
        _write_json(P["root"] / "infer" / "SIGMA_STAR.json", res)
        led = ledger(P["root"])
        led["entries"].append({"kind": "sigma_star", "at": _now(), "cpu_s": res["cpu_s"], "consumed": res["consumed"]})
        _write_json(_ledger_path(P["root"]), led)
        return {"mode": "sigma-star", **{d: v["sigma_star"] for d, v in res["datasets"].items()},
                "written": str(P["root"] / "infer" / "SIGMA_STAR.json")}
    if a.shuffled_label_sanity:
        if not subset:
            raise BenchRefused("REFUSED: --shuffled-label-sanity needs --units")
        if os.environ.get("OMP_NUM_THREADS") != "1":
            raise BenchRefused("REFUSED: fitting requires OMP_NUM_THREADS=1")
        _require_lock(P, eff, inputs)
        auth = _auth(a, inputs)
        ctx0 = _context(a, P["root"], inputs, auth, eff, _log)
        res = shuffled_label_sanity(ctx0, select(None, subset))
        _write_json(P["root"] / "sanity" / "SANITY.json", res)
        led = ledger(P["root"])
        led["entries"].append({"kind": "sanity", "at": _now(), "consumed": res["consumed"],
                               "cpu_s": sum(r["cpu_s"] for r in res["units"].values())})
        _write_json(_ledger_path(P["root"]), led)
        return res
    if a.execute_scientific_fits:
        if os.environ.get("OMP_NUM_THREADS") != "1":
            raise BenchRefused("REFUSED: scientific execution requires OMP_NUM_THREADS=1")
        if a.tier not in ("1", "2"):
            raise BenchRefused("REFUSED: --execute-scientific-fits needs --tier 1 or --tier 2")
        v = _require_lock(P, eff, inputs)
        auth = _auth(a, inputs)
        ctx0 = _context(a, P["root"], inputs, auth, eff, _log)
        if a.tier == "2":
            gate = tier2_gate(P["root"], inputs)
            if not gate["open"]:
                raise BenchRefused("REFUSED: Tier-2 trigger not met (technical validity):\n  "
                                   + "\n  ".join(gate["reasons"][:20]))
        calib = json.loads(P["calibration"].read_text()) if P["calibration"].exists() else None
        res = execute(ctx0, select(a.tier, subset), resume=a.resume, tier=a.tier, calib=calib)
        res["lock_check"] = {"ok": v["ok"], "n_code_files": v.get("n_code_files")}
        consumed = set()
        for u in res["completed"]:
            f = P["root"] / "units" / u["unit_id"] / "fit_records.json"
            consumed |= set(json.loads(f.read_text()).get("effective_keys_consumed_in_unit", []))
        res["consumption"] = consumption_report(consumed, eff)
        return res
    # default: dry-run
    out = dry_run(inputs, P["root"], a.tier or "1", subset, eff)
    out["worktree"] = P["wt"]
    if P["lock"].exists():
        from .bench_lock import verify_bench_lock
        out["lock_check"] = verify_bench_lock(P["lock"], Path(P["wt"]["root"]), P["index"], P["root"], eff)
    else:
        out["lock_check"] = {"ok": False, "mismatches": [f"no lock at {P['lock']}"]}
    return out


__all__ = ["main", "run_unit", "fit_phase", "predict_phase", "plan", "dry_run", "execute", "sigma_star",
           "shuffled_label_sanity", "make_bench_world", "Inputs", "MapStore", "SharedStore", "select_plus",
           "freeze_all_support", "tier2_gate", "BenchRefused", "verify_unit"]
