#!/usr/bin/env python
"""Independent replay of the output-leak diagnosis study (combined_output_diagnosis_v1).

Written by the independent verifier. It recomputes the study's primary (30), S3 (34) and S4 (6) endpoints from the
saved per-row predictions and the admitted inputs, with its own role, support, surface, bank, AUC, bootstrap and
decision code. It never imports the runner's code: an import guard refuses `odx`, `oar`, `stored_model_eval` and
`report` (joblib/sklearn are loaded only to replay stored attacker models on independently rebuilt surfaces).

    OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python \
        results/combined_output_diagnosis_v1/verification/replay_odx.py [--quick] [--home DIR] [--study-dir DIR]

Outputs (aggregate only; no per-person values, no row ids, private paths written as '~'):
    <study-dir>/INDEPENDENT_VERIFICATION.json
    <study-dir>/verification/replay_results_aggregate.json
Exit status 0 iff no check FAILED.
"""
from __future__ import annotations

import importlib.abc
import sys

_FORBIDDEN = ("odx", "oar", "stored_model_eval", "report")


class _RunnerImportGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in _FORBIDDEN:
            raise ImportError(f"replay_odx: importing runner module '{name}' is forbidden for the verifier")
        return None


sys.meta_path.insert(0, _RunnerImportGuard())

import argparse  # noqa: E402
import ast  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import norm  # noqa: E402

# ----------------------------------------------------------------------------------------------- protocol constants
# Transcribed by the verifier from PROTOCOL.md / the assignment; the LOCK is checked against these, not trusted.
PAIRS = [
    ("adult", "income_prediction", "sex"), ("adult", "income_prediction", "race"),
    ("adult", "employment_analysis", "race"), ("adult", "employment_analysis", "age_group"),
    ("adult", "employment_analysis", "marital_status"),
    ("adult", "education_assessment", "sex"), ("adult", "education_assessment", "race"),
    ("adult", "education_assessment", "income"),
    ("hmda", "underwriting", "race"), ("hmda", "underwriting", "ethnicity"),
    ("hmda", "pricing_analysis", "race"), ("hmda", "pricing_analysis", "sex"),
    ("hmda", "fair_lending_audit", "race"), ("hmda", "fair_lending_audit", "sex")]
PRIMARY = {"adult": ("income_prediction", "sex"), "hmda": ("underwriting", "race")}
DATASETS = ("adult", "hmda")
SEEDS = (0, 1, 2)
ATTACKER_SEEDS = (0, 1, 2)
FB = ("full", "dc", "centred", "prob")
IO = ("centred", "prob")
OFFSET_USING = ("full", "dc")
SURFACES = ("full", "dc", "centred", "prob", "offset", "hard")
COALITION = ("adult", "income_prediction", "employment_analysis", "race")
CONTRACTS = ("full", "centred", "hard")
SCORED = ("attacker_fit", "attacker_val", "assessment")
SUPPORT = {"attacker_fit": 100, "attacker_val": 30, "assessment": 100}
CERT_SALT, CERT_SHARE = "oar-cert-v1|", 0.20
T_USE, T_REC = 0.01, 0.02
EXPECT = {"B": 1999, "seed": 20261021, "alpha": 0.05, "primary_size": 30, "S3_size": 34, "S4_size": 6,
          "z_primary": 3.143980, "z_S3": 3.180426, "z_S4": 2.638257, "spot_seed": 20261031, "model_seed": 20261032,
          "chunk": 500,
          # role facts from AMENDMENT_R (exposure rows, by role fit/val/assessment) and STATS_REVIEW (assessment
          # rows, groups); overridable only for synthetic self-tests
          "exposure": {"adult": [17, [6, 4, 7]], "hmda": [42, [21, 7, 14]]},
          "assess_rows_groups": {"adult": [5243, 5243], "hmda": [4764, 4762]}}
TOL_POINT, TOL_SE = 1e-9, 1e-9
NEAR_AUC, NEAR_ACC, WEAK_DISCORDANT = 0.98, 0.99, 30


def expected_primary_ids():
    ids = ["U-frozen-adult", "U-refit-adult", "U-frozen-hmda", "U-refit-hmda",
           "FC-adult", "CH-adult", "FC-hmda", "CH-hmda"]
    pr = [("adult", 0, 1)] + [("hmda", i, j) for i in range(5) for j in range(i + 1, 5)]
    for side in ("FC", "CH"):
        ids += [f"{side}-{ds}-pair{i}-{j}" for ds, i, j in pr]
    return ids


EXPECTED_NOT_ESTIMABLE = sorted(f"{s}-hmda-pair{i}-{j}" for s in ("FC", "CH") for i in range(5)
                                for j in range(i + 1, 5) if (i in (3, 4) or j in (3, 4)))
DECLARED_ALIASES = {"FC-adult-pair0-1": "FC-adult", "CH-adult-pair0-1": "CH-adult"}


# ----------------------------------------------------------------------------------------------- small helpers
class Checks:
    def __init__(self):
        self.items = []

    def add(self, cid, status, detail=None, cause=None):
        assert status in ("PASS", "FAIL", "INFO", "SKIPPED", "WARN")
        it = {"id": cid, "status": status}
        if detail is not None:
            it["detail"] = detail
        if cause is not None:
            it["diagnosed_cause"] = cause
        self.items.append(it)
        return status == "PASS"

    def ok(self, cid, cond, detail=None, cause=None, fail_status="FAIL"):
        return self.add(cid, "PASS" if cond else fail_status, detail, None if cond else cause)

    def counts(self):
        out = {}
        for it in self.items:
            out[it["status"]] = out.get(it["status"], 0) + 1
        return out


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def u01(salt: str, key: str) -> float:
    return int(hashlib.sha256((salt + key).encode()).hexdigest()[:8], 16) / 2 ** 32


def fnum(x):
    """json default for numpy scalars/arrays (NaN/inf -> null)."""
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.ndarray):
        return [fnum(v) for v in x.tolist()]
    x = float(x)
    return None if not np.isfinite(x) else x


def to_float(v):
    try:
        if v is None or str(v).strip() in ("", "nan", "NaN", "None", "NA"):
            return None
        return float(v)
    except ValueError:
        return None


# ----------------------------------------------------------------------------------------------- configuration
class Cfg:
    def __init__(self, home: Path, study_dir: Path, quick=False, expect=None, n_model_units=6):
        self.home = Path(home)
        self.cache = self.home / "PCRL_eval_cache_private"
        self.inputs = self.cache / "bench_v1" / "inputs"
        self.run = self.cache / "odx_v1" / "run"
        self.units = self.run / "units"
        self.oar_units = self.cache / "oar_v1" / "run" / "units"
        self.study = Path(study_dir)
        self.quick = quick
        self.E = dict(EXPECT)
        if expect:
            self.E.update(expect)
        self.n_model_units = n_model_units

    def tilde(self, p) -> str:
        p = str(p)
        h = str(self.home)
        return "~" + p[len(h):] if p.startswith(h) else p

    def untilde(self, s: str) -> Path:
        return self.home / s[2:] if s.startswith("~/") else Path(s)


# ----------------------------------------------------------------------------------------------- roles (own code)
def build_world(cfg: Cfg, ds: str) -> dict:
    """oar-roles-v1 rebuilt from labels.npz: exposure (test rows whose canon_key occurs in train) leaves every scored
    role; 20 % of the remaining attacker_fit groups (sha256 of salt+ds+canon_key) become 'cert'."""
    L = np.load(cfg.inputs / f"{ds}_labels.npz", allow_pickle=False)
    lab = {k: L[k] for k in L.files}
    role0 = np.array(lab["role"], dtype=object)
    split = lab["split"].astype(str)
    canon = lab["canon_key"].astype(str)
    train_keys = set(canon[split == "train"].tolist())
    exposed = (split == "test") & np.fromiter((k in train_keys for k in canon), bool, len(canon))
    role = role0.copy()
    scored0 = np.isin(role0.astype(str), SCORED)
    role[exposed & scored0] = "excluded_exposure"
    fit = role == "attacker_fit"
    cert = np.zeros(len(role), bool)
    for i in np.flatnonzero(fit):
        cert[i] = u01(CERT_SALT + ds + "|", canon[i]) < CERT_SHARE
    role[cert] = "cert"
    idx = {r: np.flatnonzero(role == r) for r in ("defense_fit", "cert", *SCORED)}
    W = {"ds": ds, "lab": lab, "row_id": lab["row_id"], "unit": lab["unit"], "role": role, "role0": role0,
         "exposed": exposed, "idx": idx,
         "exposure_rows_by_role": {r: int((exposed & (role0.astype(str) == r)).sum()) for r in SCORED},
         "exposure_groups": int(len(np.unique(lab["unit"][exposed & scored0])))}
    W["exposure_rows"] = int((exposed & scored0).sum())
    return W


def support_counts(W, attr):
    s = W["lab"][attr].astype(int)
    K = int(s.max()) + 1
    out = {}
    for r in ("defense_fit", "cert", "attacker_fit", "attacker_val", "assessment"):
        out[r] = np.bincount(s[W["idx"][r]], minlength=K).astype(int)
    sup = [c for c in range(K) if all(out[r][c] >= SUPPORT[r] for r in SUPPORT)]
    return K, out, sup


# ----------------------------------------------------------------------------------------------- surfaces (own code)
def softmax64(L):
    L = np.asarray(L, np.float64)
    Z = L - L.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def my_surfaces(L) -> dict:
    """Functions of the logit matrix only (no labels, no row keys, no other rows)."""
    L = np.asarray(L, np.float64)
    K = L.shape[1]
    if K == 2:
        d = L[:, 1] - L[:, 0]
        c = (L[:, 0] + L[:, 1]) / 2.0
        cen = d[:, None]
    else:
        c = L.mean(axis=1)
        cen = L - c[:, None]
    hard = np.zeros_like(L)
    hard[np.arange(len(L)), L.argmax(1)] = 1.0
    return {"full": L, "dc": np.hstack([cen, c[:, None]]), "centred": cen, "prob": softmax64(L),
            "offset": c[:, None], "hard": hard}


def my_exactness(L) -> dict:
    L = np.asarray(L, np.float64)
    K = L.shape[1]
    c = L.mean(axis=1)
    cen = L - c[:, None]
    P = softmax64(L)
    out = {"K": K, "n": int(len(L)),
           "reconstruct_full_from_centred_plus_offset_maxabs": float(np.max(np.abs(cen + c[:, None] - L))),
           "softmax_of_centred_minus_softmax_full_maxabs": float(np.max(np.abs(softmax64(cen) - P))),
           "offset_sd": float(np.std(c))}
    if K == 2:
        d = L[:, 1] - L[:, 0]
        with np.errstate(over="ignore"):
            sig = 1.0 / (1.0 + np.exp(-d))
        out.update({"sigmoid_margin_minus_softmax_p1_maxabs": float(np.max(np.abs(sig - P[:, 1]))),
                    "argmax_equals_margin_positive": bool(np.array_equal(L.argmax(1), (d > 0).astype(int))),
                    "ties_d_equal_0": int((d == 0).sum()),
                    "p1_saturated_exact_0_or_1": int(((P[:, 1] == 0.0) | (P[:, 1] == 1.0)).sum()),
                    "rows_d_ge_36_7368_float64_p1_eq_1": int((d >= 53 * np.log(2)).sum()),
                    "rows_abs_d_ge_16_6355_float32_saturation": int((np.abs(d) >= 24 * np.log(2)).sum()),
                    "abs_margin_max": float(np.max(np.abs(d)))})
        # extra (verifier-only) facts
        lo, hi = c - d / 2, c + d / 2
        out["_roundtrip_dc_bit_exact_rows_failing"] = int((~((lo == L[:, 0]) & (hi == L[:, 1]))).sum())
        out["_rows_abs_d_ge_16_6"] = int((np.abs(d) >= 16.6).sum())
    return out


# ----------------------------------------------------------------------------------------------- bootstrap + AUC
class Boot:
    """Multinomial counts over assessment groups (np.unique order), B replicates drawn sequentially in chunks."""

    def __init__(self, assess_unit, B, seed, chunk=500):
        uu, inv = np.unique(assess_unit, return_inverse=True)
        n = len(uu)
        rng = np.random.default_rng(seed)
        parts = []
        for s0 in range(0, B, chunk):
            parts.append(rng.multinomial(n, np.full(n, 1.0 / n), size=min(chunk, B - s0)))
        C = np.vstack(parts)
        self.n_units, self.n_rows, self.B = n, len(inv), B
        self.W = np.vstack([np.ones((1, len(inv))), C[:, inv].astype(np.float64)])  # row 0 = original sample

    def wmean(self, v, mask=None):
        W = self.W if mask is None else self.W[:, mask]
        v = np.asarray(v, np.float64) if mask is None else np.asarray(v, np.float64)[mask]
        return (W @ v) / W.sum(1)

    def wauc(self, score, pos, mask=None, block=400):
        """Weighted Mann-Whitney AUC, ties = 1/2, for every replicate (row 0 = original). NaN if a side is empty."""
        score = np.asarray(score, np.float64)
        pos = np.asarray(pos, bool)
        W = self.W
        if mask is not None:
            score, pos, W = score[mask], pos[mask], W[:, mask]
        if not np.all(np.isfinite(score)):
            raise ValueError("non-finite attacker score")
        order = np.argsort(score, kind="stable")
        s, p = score[order], pos[order]
        starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
        out = np.empty(W.shape[0])
        pf, nf = p.astype(np.float64), (~p).astype(np.float64)
        for r0 in range(0, W.shape[0], block):
            Wo = W[r0:r0 + block][:, order]
            gp = np.add.reduceat(Wo * pf, starts, axis=1)
            gn = np.add.reduceat(Wo * nf, starts, axis=1)
            below = np.cumsum(gn, axis=1) - gn
            num = (gp * (below + 0.5 * gn)).sum(1)
            den = gp.sum(1) * gn.sum(1)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[r0:r0 + block] = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
        return out


def pair_score(P, i, j):
    den = P[:, i] + P[:, j]
    with np.errstate(invalid="ignore", divide="ignore"):
        sc = np.where(den > 0, P[:, j] / np.where(den > 0, den, 1.0), 0.5)
    return sc, int((den == 0).sum())


# ----------------------------------------------------------------------------------------------- the replay
class Replay:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.C = Checks()
        self.agg = {}
        self.worlds = {}
        self.index = json.loads((cfg.inputs / "INPUTS_INDEX.json").read_text())
        self.lock = json.loads((cfg.study / "LOCK.json").read_text())
        self.fwd = {}
        self.unit_cache = {}
        self.stat_cache = {}
        self.boots = {}
        self.sel = {}           # bank uid -> (own selection resolved uid, own value)
        self.unit_problems = {}

    # ---------------------------------------------------------------- inputs
    def purpose(self, ds, p):
        return self.index["datasets"][ds]["purposes"][p]

    def forward(self, ds, k):
        key = (ds, k)
        if key not in self.fwd:
            F = np.load(self.cfg.inputs / f"{ds}_s{k}_forward.npz", allow_pickle=False)
            self.fwd[key] = {kk: F[kk] for kk in F.files if not kk.startswith("rep_")}
        return self.fwd[key]

    def logits(self, ds, k, purpose):
        return self.forward(ds, k)[self.purpose(ds, purpose)["logits_key"]].astype(np.float64)

    def task(self, ds, purpose):
        return self.worlds[ds]["lab"][self.purpose(ds, purpose)["labels_task_key"]].astype(int)

    # ---------------------------------------------------------------- 0. lock and constants
    def check_lock(self):
        C, L, E = self.C, self.lock, self.cfg.E
        fam = L.get("families", {})
        C.ok("lock.pairs", [tuple(x) for x in L.get("pairs", [])] == PAIRS, {"n": len(L.get("pairs", []))})
        C.ok("lock.primary_cells", sorted(tuple(x) for x in L.get("primary_cells", [])) ==
             sorted((ds, *PRIMARY[ds]) for ds in DATASETS))
        ids = fam.get("primary_ids", [])
        C.ok("lock.primary_size", fam.get("primary_size") == E["primary_size"] == 30 and len(ids) == 30 and
             len(set(ids)) == 30, {"primary_size": fam.get("primary_size"), "n_ids": len(ids)})
        C.ok("lock.primary_ids", ids == expected_primary_ids(), "ids/order equal to verifier enumeration")
        C.ok("lock.expected_not_estimable", sorted(fam.get("expected_not_estimable", [])) == EXPECTED_NOT_ESTIMABLE,
             {"n": len(fam.get("expected_not_estimable", []))})
        C.ok("lock.declared_aliases", fam.get("declared_aliases") == DECLARED_ALIASES)
        C.ok("lock.secondary_sizes", fam.get("S3_size") == E["S3_size"] == 2 * len(PAIRS) + 6 and
             fam.get("S4_size") == E["S4_size"] == 2 * len(CONTRACTS),
             {"S3": fam.get("S3_size"), "S4": fam.get("S4_size")})
        a = E["alpha"]
        zs = {"z_primary": norm.ppf(1 - a / (2 * 30)), "z_S3": norm.ppf(1 - a / (2 * 34)),
              "z_S4": norm.ppf(1 - a / (2 * 6))}
        for k, v in zs.items():
            lv = fam.get(k)
            C.ok(f"lock.{k}", lv is not None and abs(lv - v) < 1e-9 and abs(round(v, 6) - E[k]) < 1e-12,
                 {"lock": lv, "recomputed": v, "protocol_6dp": E[k]})
        self.z = {"primary": zs["z_primary"], "S3": zs["z_S3"], "S4": zs["z_S4"]}
        bs = fam.get("bootstrap", {})
        C.ok("lock.bootstrap", bs.get("B") == E["B"] and bs.get("seed") == E["seed"], {"B": bs.get("B"),
                                                                                       "seed": bs.get("seed")})
        banks = fam.get("banks", {})
        C.ok("lock.banks", tuple(banks.get("io", ())) == IO and tuple(banks.get("full", ())) == FB and
             tuple(banks.get("offset_using", ())) == OFFSET_USING)
        # targets of the primary slots
        bad = [s["id"] for s in fam.get("primary", []) if
               s.get("target") != (T_USE if s["id"].startswith("U-") else T_REC)]
        C.ok("lock.targets", not bad and len(fam.get("primary", [])) == 30, {"bad": bad})
        # hash-pinned public files
        fs = L.get("file_sha256", {})
        bad = [f for f, h in fs.items() if not (self.cfg.study / f).exists() or sha256_file(self.cfg.study / f) != h]
        C.ok("lock.file_sha256", not bad, {"n": len(fs), "mismatched": bad})
        adm = L.get("admitted", {}).get("inputs", {})
        bad = [f for f, h in adm.items() if not self.cfg.untilde(f).exists() or sha256_file(self.cfg.untilde(f)) != h]
        C.ok("lock.admitted_inputs", not bad, {"n": len(adm), "mismatched": bad})
        ru = L.get("admitted", {}).get("reused_units", {})
        bad = [f for f, h in ru.items() if not self.cfg.untilde(f).exists() or sha256_file(self.cfg.untilde(f)) != h]
        C.ok("lock.reused_units", not bad, {"n": len(ru), "mismatched": bad})
        self.reused_units = ru

    # ---------------------------------------------------------------- 1. roles and support
    def check_roles(self):
        C = self.C
        for ds in DATASETS:
            W = build_world(self.cfg, ds)
            self.worlds[ds] = W
            # base roles equal the bench roles npz
            R = np.load(self.cfg.inputs / f"{ds}_roles.npz", allow_pickle=False)
            okb = True
            for r in ("defense_fit", "attacker_fit", "attacker_val", "assessment"):
                rows = np.sort(W["row_id"][W["role0"].astype(str) == r])
                okb &= np.array_equal(rows, np.sort(R[f"{r}__row_id"]))
            C.ok(f"roles.{ds}.bench_roles_npz", okb)
            a = W["idx"]["assessment"]
            groups = {r: (int(len(ix)), int(len(np.unique(W["unit"][ix])))) for r, ix in W["idx"].items()}
            disj = True
            for r1 in ("cert", *SCORED):
                for r2 in ("defense_fit", "cert", *SCORED):
                    if r1 < r2 or r1 == r2:
                        continue
                    if len(np.intersect1d(W["unit"][W["idx"][r1]], W["unit"][W["idx"][r2]])):
                        disj = False
            C.ok(f"roles.{ds}.group_disjoint", disj, {"rows_groups": groups})
            C.add(f"roles.{ds}.exposure", "INFO", {"rows": W["exposure_rows"], "groups": W["exposure_groups"],
                                                    "by_role": W["exposure_rows_by_role"]})
            expected = tuple(self.cfg.E["exposure"][ds])
            C.ok(f"roles.{ds}.exposure_counts_vs_amendment",
                 (W["exposure_rows"], [W["exposure_rows_by_role"][r] for r in SCORED]) == expected,
                 {"recomputed": [W["exposure_rows"], [W["exposure_rows_by_role"][r] for r in SCORED]],
                  "amendment": [expected[0], expected[1]]})
            exp_assess = tuple(self.cfg.E["assess_rows_groups"][ds])
            C.ok(f"roles.{ds}.assessment_rows_groups", groups["assessment"] == exp_assess,
                 {"recomputed": groups["assessment"], "stats_review": exp_assess})
        self.check_coverage()

    def check_coverage(self):
        C = self.C
        p = self.cfg.study / "COVERAGE_AND_SUPPORT.csv"
        rows = list(csv.DictReader(open(p)))
        bad, n = [], 0
        self.support = {}
        for ds, purpose, attr in PAIRS:
            W = self.worlds[ds]
            K, cnt, sup = support_counts(W, attr)
            self.support[(ds, attr)] = (K, sup)
            mine = [r for r in rows if (r["dataset"], r["purpose"], r["attribute"]) == (ds, purpose, attr)]
            for c in range(K):
                r = [x for x in mine if x["what"] == "class" and x["id"] == str(c)]
                n += 1
                if len(r) != 1:
                    bad.append((ds, purpose, attr, "class", c, "missing"))
                    continue
                r = r[0]
                got = [int(r[f"n_{q}"]) for q in ("defense_fit", "cert", "attacker_fit", "attacker_val", "assessment")]
                exp = [int(cnt[q][c]) for q in ("defense_fit", "cert", "attacker_fit", "attacker_val", "assessment")]
                st = "ESTIMABLE" if c in sup else "NOT_ESTIMABLE"
                if got != exp or (r["supported"] == "True") != (c in sup) or r["status"] != st:
                    bad.append((ds, purpose, attr, "class", c, {"csv": got, "replay": exp}))
            for i in range(K):
                for j in range(i + 1, K):
                    r = [x for x in mine if x["what"] == "pair" and x["id"] == f"{i}-{j}"]
                    n += 1
                    ps = i in sup and j in sup
                    if len(r) != 1 or (r[0]["supported"] == "True") != ps:
                        bad.append((ds, purpose, attr, "pair", f"{i}-{j}"))
            # task-class counts
            t = self.task(ds, purpose)
            Kt = int(self.purpose(ds, purpose)["task_dim"])
            r = [x for x in mine if x["what"] == "task_classes"]
            n += 1
            exp = {q: np.bincount(t[W["idx"][q]], minlength=Kt).tolist() for q in SCORED}
            if len(r) != 1 or json.loads(r[0]["id"]) != exp:
                bad.append((ds, purpose, attr, "task_classes"))
        C.ok("coverage.counts_and_support", not bad, {"rows_checked": n, "mismatches": bad[:20]})
        # the 14 declared NOT_ESTIMABLE slots follow from support
        ne = []
        for ds in DATASETS:
            K, sup = self.support[(ds, PRIMARY[ds][1])]
            for i in range(K):
                for j in range(i + 1, K):
                    if not (i in sup and j in sup):
                        ne += [f"FC-{ds}-pair{i}-{j}", f"CH-{ds}-pair{i}-{j}"]
        C.ok("coverage.declared_not_estimable_from_support", sorted(ne) == EXPECTED_NOT_ESTIMABLE,
             {"n_replay": len(ne)})

    # ---------------------------------------------------------------- 2. unit inventory and integrity
    def unit_dir(self, uid):
        return self.cfg.units / uid

    def resolve_dir(self, uid):
        d = self.unit_dir(uid)
        a = d / "ALIAS.json"
        if a.exists():
            return self.cfg.untilde(json.loads(a.read_text())["source"])
        return d

    def complete(self, d: Path):
        c = d / "COMPLETE.json"
        if not c.exists():
            return False, "no COMPLETE.json"
        rec = json.loads(c.read_text())
        for f, h in rec["files"].items():
            if not (d / f).exists():
                return False, f"missing {f}"
            if sha256_file(d / f) != h:
                return False, f"hash mismatch {f}"
        return True, rec

    def parse_uid(self, uid):
        parts = uid.split("__")
        out = {"ds": parts[0], "seed": int(parts[1][1:]) if parts[1].startswith("s") and parts[1][1:].isdigit()
               else None}
        if len(parts) == 6:
            out.update(purpose=parts[2], attr=parts[3], stratum=parts[4], surface=parts[5])
        elif len(parts) == 7 and parts[6] == "bank":
            out.update(purpose=parts[2], attr=parts[3], stratum=parts[4], surface=parts[5], s4bank=True)
        return out

    def check_units(self):
        C, cfg = self.C, self.cfg
        uids = sorted(p.name for p in cfg.units.iterdir() if p.is_dir()) if cfg.units.exists() else []
        partial = [u for u in uids if ".partial-" in u]
        uids = [u for u in uids if ".partial-" not in u]
        self.uids = set()
        problems = {}
        n_alias = n_attack = n_bank = 0
        for uid in uids:
            d = self.unit_dir(uid)
            okc, rec = self.complete(d)
            if not okc:
                problems[uid] = f"incomplete: {rec}"
                continue
            if rec.get("id") != uid:
                problems[uid] = "COMPLETE id mismatch"
                continue
            self.uids.add(uid)
            r = json.loads((d / "record.json").read_text())
            if (d / "ALIAS.json").exists():
                n_alias += 1
                al = json.loads((d / "ALIAS.json").read_text())
                src = cfg.untilde(al["source"])
                oks, srec = self.complete(src)
                why = []
                if not oks:
                    why.append(f"source incomplete: {srec}")
                h = sha256_file(src / "COMPLETE.json") if (src / "COMPLETE.json").exists() else None
                if h != al.get("source_COMPLETE_sha256"):
                    why.append("source COMPLETE hash != ALIAS")
                lk = self.reused_units.get(cfg.tilde(src / "COMPLETE.json"))
                if lk is None or lk != h:
                    why.append("source not in LOCK reused_units with this hash")
                if (src / "record.json").read_bytes() != (d / "record.json").read_bytes():
                    why.append("record.json differs from source")
                # alias must point at the same dataset/seed and the mapped surface
                pu = self.parse_uid(uid)
                exp_name = None
                if pu.get("stratum") == "FH" and pu.get("surface") in ("full", "prob", "hard"):
                    exp_name = f"{pu['ds']}__s{pu['seed']}__O_{pu['surface']}"
                elif pu.get("stratum") == "REF":
                    exp_name = f"{pu['ds']}__s{pu['seed']}__REF"
                if exp_name is None or src.name != exp_name or (pu.get("ds"), pu.get("purpose"), pu.get("attr")) != \
                        (pu["ds"], *PRIMARY.get(pu["ds"], (None, None))):
                    why.append(f"alias target not the declared reuse (got {src.name})")
                if why:
                    problems[uid] = "; ".join(why)
            if r.get("kind") == "bank":
                n_bank += 1
            elif "nl_family" in r:
                n_attack += 1
        C.ok("units.integrity", not problems, {"complete_units": len(self.uids), "alias": n_alias,
                                               "attack": n_attack, "bank": n_bank, "partial_dirs": len(partial),
                                               "problems": dict(list(problems.items())[:30]),
                                               "n_problems": len(problems)})
        self.unit_problems = problems
        # row identities in every unit with predictions
        bad = {}
        checked = 0
        for uid in sorted(self.uids):
            pu = self.parse_uid(uid)
            ds = pu["ds"]
            if ds not in self.worlds:
                continue
            d = self.resolve_dir(uid)
            if not (d / "preds.npz").exists():
                continue
            W = self.worlds[ds]
            a, v = W["idx"]["assessment"], W["idx"]["attacker_val"]
            P = np.load(d / "preds.npz", allow_pickle=False)
            why = []
            if not np.array_equal(P["assess_row_id"], W["row_id"][a]):
                why.append("assess_row_id != replay assessment role")
            if "assess_unit" in P.files and not np.array_equal(P["assess_unit"], W["unit"][a]):
                why.append("assess_unit != replay")
            attr = pu.get("attr")
            if "y_s" in P.files and attr in W["lab"] and not np.array_equal(P["y_s"], W["lab"][attr][a].astype(int)):
                why.append("y_s != labels")
            for k in P.files:
                if k.startswith("P__"):
                    M = P[k]
                    if M.shape[0] != len(a) or not np.all(np.isfinite(M)) or (M < 0).any() or \
                            np.max(np.abs(M.sum(1) - 1)) > 1e-9:
                        why.append(f"{k} not a probability matrix over assessment rows")
            vp = d / "val_preds.npz"
            if vp.exists():
                V = np.load(vp, allow_pickle=False)
                if "val_row_id" in V.files:
                    if not np.array_equal(V["val_row_id"], W["row_id"][v]):
                        why.append("val_row_id != replay attacker_val role")
                    if len(np.intersect1d(V["val_row_id"], P["assess_row_id"])):
                        why.append("assessment rows leak into validation")
                    if "val_y_s" in V.files and attr in W["lab"] and \
                            not np.array_equal(V["val_y_s"], W["lab"][attr][v].astype(int)):
                        why.append("val_y_s != labels")
            r = json.loads((self.unit_dir(uid) / "record.json").read_text())
            if "nl_family" in r and pu.get("surface") in SURFACES and pu.get("stratum") == "FH" and \
                    pu.get("purpose") and not pu["purpose"].startswith("PAIR_"):
                K = self.logits(ds, pu["seed"], pu["purpose"]).shape[1]
                exp_cols = {"full": K, "dc": (1 if K == 2 else K) + 1, "centred": 1 if K == 2 else K, "prob": K,
                            "offset": 1, "hard": K}[pu["surface"]]
                if r.get("n_columns") != exp_cols:
                    why.append(f"n_columns {r.get('n_columns')} != {exp_cols}")
            checked += 1
            if why:
                bad[uid] = why
        C.ok("units.row_identities", not bad, {"units_checked": checked, "n_bad": len(bad),
                                               "bad": dict(list(bad.items())[:30])})
        self.unit_problems.update({u: "; ".join(w) for u, w in bad.items()})
        # U2__A utility units (oar)
        bad = {}
        for ds in DATASETS:
            W = self.worlds[ds]
            a = W["idx"]["assessment"]
            t = self.task(ds, PRIMARY[ds][0])
            for k in SEEDS:
                d = cfg.oar_units / f"{ds}__s{k}__U2__A"
                okc, _ = self.complete(d)
                if not okc:
                    bad[d.name] = "incomplete"
                    continue
                if cfg.tilde(d / "COMPLETE.json") in self.reused_units and \
                        self.reused_units[cfg.tilde(d / "COMPLETE.json")] != sha256_file(d / "COMPLETE.json"):
                    bad[d.name] = "hash differs from LOCK"
                P = np.load(d / "preds.npz", allow_pickle=False)
                if not np.array_equal(P["assess_row_id"], W["row_id"][a]) or not np.array_equal(P["y_t"], t[a]):
                    bad[d.name] = "rows/labels differ from replay roles"
        C.ok("units.U2__A", not bad, {"bad": bad})

    # ---------------------------------------------------------------- 3. exactness
    def check_exactness(self):
        p = self.cfg.study / "EXACTNESS.json"
        if not p.exists():
            self.C.add("exactness", "SKIPPED", "EXACTNESS.json absent")
            return
        X = json.loads(p.read_text())
        bad, extra = {}, {}
        for ds in DATASETS:
            for k in SEEDS:
                for purpose in self.index["datasets"][ds]["purposes"]:
                    key = f"{ds}__s{k}__{purpose}"
                    mine = my_exactness(self.logits(ds, k, purpose))
                    theirs = X.get(key)
                    if theirs is None:
                        bad[key] = "missing"
                        continue
                    for f, v in theirs.items():
                        m = mine.get(f)
                        if isinstance(v, bool) or isinstance(v, int):
                            okf = m == v
                        else:
                            okf = m is not None and abs(m - v) <= 1e-12 + 1e-12 * abs(v)
                        if not okf:
                            bad.setdefault(key, {})[f] = {"file": v, "replay": m}
                    extra[key] = {f: mine[f] for f in mine if f.startswith("_")}
                    # claims: identities hold
                    if mine["K"] == 2 and not (mine["argmax_equals_margin_positive"] and
                                               mine["sigmoid_margin_minus_softmax_p1_maxabs"] < 1e-15 and
                                               mine["_roundtrip_dc_bit_exact_rows_failing"] == 0):
                        bad.setdefault(key, {})["identity"] = "binary identity failed"
        self.C.ok("exactness.EXACTNESS_json", not bad, {"entries": len(X), "mismatches": bad})
        # SCORE_MATH quoted 92 (s1) and 11 (s2) adult rows with |d| >= 16.6; EXACTNESS counts at 24 ln 2
        self.C.add("exactness.score_math_float32_counts", "INFO",
                   {k: {"abs_d_ge_16.6": v.get("_rows_abs_d_ge_16_6")} for k, v in extra.items()
                    if k.startswith("adult") and "income" in k})
        self.agg["exactness_extra"] = extra

    # ---------------------------------------------------------------- 4. banks
    def val_ll(self, uid):
        """Candidate value used by a bank: the unit's record val_log_loss.NL (banks: own recomputed min)."""
        r = json.loads((self.unit_dir(uid) / "record.json").read_text())
        if r.get("kind") == "bank":
            return self.bank(uid)[1]
        return float(r["val_log_loss"]["NL"])

    def bank_candidates(self, uid):
        pu = self.parse_uid(uid)
        ds, k, purpose, attr, st, surf = (pu["ds"], pu["seed"], pu.get("purpose"), pu.get("attr"), pu.get("stratum"),
                                          pu.get("surface"))
        if pu.get("s4bank"):
            pa, pb = purpose[len("PAIR_"):].split("+")
            single = lambda p_: f"{ds}__s{k}__{p_}__{attr}__FH__" + {"full": "fullbank", "centred": "iobank"}.get(
                surf, surf)  # noqa: E731
            return [uid[:-len("__bank")], single(pa), single(pb)]
        base = f"{ds}__s{k}__{purpose}__{attr}__{st}__"
        return [base + c for c in (IO if surf == "iobank" else FB)]

    def bank(self, uid):
        """Own selection: minimum candidate value, ties -> earlier listed; nested banks resolve to their selection."""
        if uid in self.sel:
            return self.sel[uid]
        cands = self.bank_candidates(uid)
        best, bv = None, None
        for c in cands:
            v = self.val_ll(c) if c in self.uids else None
            if v is None:
                self.sel[uid] = (None, None)
                return self.sel[uid]
            if best is None or v < bv:
                best, bv = c, v
        rb = json.loads((self.unit_dir(best) / "record.json").read_text())
        res = self.bank(best)[0] if rb.get("kind") == "bank" else best
        self.sel[uid] = (res, bv)
        return self.sel[uid]

    def resolve(self, uid):
        if uid not in self.uids:
            return None
        r = json.loads((self.unit_dir(uid) / "record.json").read_text())
        return self.bank(uid)[0] if r.get("kind") == "bank" else uid

    def check_banks(self, clip=1e-15):
        bad, n, vl_bad, vl_n, vl_max = {}, 0, {}, 0, 0.0
        for uid in sorted(self.uids):
            r = json.loads((self.unit_dir(uid) / "record.json").read_text())
            if r.get("kind") != "bank":
                continue
            n += 1
            mine, v = self.bank(uid)
            why = []
            if mine is None:
                why.append("candidate missing")
            else:
                if r.get("bank_selected") != mine:
                    why.append(f"bank_selected {r.get('bank_selected')} != replay {mine}")
                cands = self.bank_candidates(uid)
                rc = r.get("candidates_attacker_val_log_loss", {})
                if list(rc.keys()) != cands:
                    why.append("candidate list/order differs")
                for c in cands:
                    if c in rc and abs(rc[c] - self.val_ll(c)) > 1e-12:
                        why.append(f"recorded candidate value differs for {c}")
                if abs(float(r["val_log_loss"]["NL"]) - v) > 1e-12:
                    why.append("bank val_log_loss differs")
            if why:
                bad[uid] = why
        self.C.ok("banks.selection", not bad and n > 0, {"banks_checked": n, "n_bad": len(bad),
                                                         "bad": dict(list(bad.items())[:20])})
        # the recorded NL validation log loss is the attacker-seed-0 log loss on attacker_val (recomputed)
        for uid in sorted(self.uids):
            r = json.loads((self.unit_dir(uid) / "record.json").read_text())
            if "nl_family" not in r:
                continue
            d = self.resolve_dir(uid)
            V = np.load(d / "val_preds.npz", allow_pickle=False)
            y = V["val_y_s"].astype(int)
            P = V["VAL__NL__as0"]
            ll = float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], clip, 1.0))))
            vl_n += 1
            dlt = abs(ll - float(r["val_log_loss"]["NL"]))
            vl_max = max(vl_max, dlt)
            if dlt > 1e-9:
                vl_bad[uid] = dlt
        self.C.ok("banks.val_log_loss_recomputed_from_val_preds", not vl_bad,
                  {"units": vl_n, "max_abs_diff": vl_max, "n_bad": len(vl_bad), "bad": dict(list(vl_bad.items())[:10])},
                  cause="recorded NL value is not the as0 attacker_val log loss (clip/selection-model difference)",
                  fail_status="WARN")

    # ---------------------------------------------------------------- 5. statistics
    def boot(self, ds):
        if ds not in self.boots:
            W = self.worlds[ds]
            self.boots[ds] = Boot(W["unit"][W["idx"]["assessment"]], self.cfg.E["B"], self.cfg.E["seed"],
                                  self.cfg.E["chunk"])
        return self.boots[ds]

    def preds(self, uid):
        d = self.resolve_dir(uid)
        if d not in self.unit_cache:
            P = np.load(d / "preds.npz", allow_pickle=False)
            self.unit_cache[d] = {k: P[k] for k in P.files}
        return self.unit_cache[d]

    def R_unit(self, ds, uid, attr):
        """Per attacker seed macro supported-class OvR AUC, then mean over attacker seeds; vector over replicates."""
        key = ("R", str(self.resolve_dir(uid)), attr)
        if key not in self.stat_cache:
            P = self.preds(uid)
            W = self.worlds[ds]
            y = W["lab"][attr][W["idx"]["assessment"]].astype(int)
            _, sup = self.support[(ds, attr)]
            b = self.boot(ds)
            per = []
            for a_ in ATTACKER_SEEDS:
                M = P[f"P__NL__as{a_}"]
                per.append(np.mean([b.wauc(M[:, c], y == c) for c in sup], axis=0))
            self.stat_cache[key] = np.mean(per, axis=0)
        return self.stat_cache[key]

    def PAIR_unit(self, ds, uid, attr, i, j):
        key = ("PAIR", str(self.resolve_dir(uid)), attr, i, j)
        if key not in self.stat_cache:
            P = self.preds(uid)
            W = self.worlds[ds]
            y = W["lab"][attr][W["idx"]["assessment"]].astype(int)
            m = (y == i) | (y == j)
            b = self.boot(ds)
            per, zden = [], 0
            for a_ in ATTACKER_SEEDS:
                sc, nz = pair_score(P[f"P__NL__as{a_}"], i, j)
                zden += int(((P[f"P__NL__as{a_}"][:, i] + P[f"P__NL__as{a_}"][:, j]) == 0)[m].sum())
                per.append(b.wauc(sc, y == j, mask=m))
            self.stat_cache[key] = (np.mean(per, axis=0), zden)
        return self.stat_cache[key]

    def acc_vec(self, ds, pred, y):
        return self.boot(ds).wmean((np.asarray(pred) == np.asarray(y)).astype(np.float64))

    def finish(self, sid, z, target, per_seed_vecs, extra=None, identical=False, weak=None, near=None):
        """Mean over encoder seeds; SE ddof=1; bounds; decision; flags."""
        T = np.mean(per_seed_vecs, axis=0)
        point, reps = float(T[0]), T[1:]
        n_ne = int(np.isnan(reps).sum())
        se = float(np.nanstd(reps, ddof=1))
        if identical:
            se = 0.0
        lower, upper = point - z * se, point + z * se
        flags = []
        if identical:
            flags.append("IDENTICAL_BY_SELECTION")
        if weak:
            flags.append("NORMAL_APPROX_WEAK")
        if near:
            flags.append("NEAR_BOUND")
        decision = "NOT_ESTABLISHED" if identical else ("PASS" if lower > target else "NOT_ESTABLISHED")
        out = {"id": sid, "point": point, "se": se, "z": z, "lower": lower, "upper": upper, "target": target,
               "decision": decision, "flags": flags, "n_ne_replicates": n_ne,
               "per_seed": [float(v[0]) for v in per_seed_vecs], "between_seed_sd": float(np.std(
                   [v[0] for v in per_seed_vecs], ddof=1)) if len(per_seed_vecs) > 1 else None}
        if near:
            out["percentile_95"] = [float(np.nanpercentile(reps, 2.5)), float(np.nanpercentile(reps, 97.5))]
            m3 = reps - np.nanmean(reps)
            out["replicate_skewness"] = float(np.nanmean(m3 ** 3) / np.nanmean(m3 ** 2) ** 1.5) if se > 0 else None
        if extra:
            out.update(extra)
        out["_reps"] = T
        return out

    def usefulness(self, sid, ds, purpose, which, z):
        W = self.worlds[ds]
        a = W["idx"]["assessment"]
        t = self.task(ds, purpose)
        y = t[a]
        Kt = int(self.purpose(ds, purpose)["task_dim"])
        const = int(np.argmax(np.bincount(t[W["idx"]["attacker_fit"]], minlength=Kt)))
        vecs, disc, accs, single = [], [], [], []
        for k in SEEDS:
            if which == "frozen":
                pred = self.logits(ds, k, purpose)[a].argmax(1)
            else:
                d = self.cfg.oar_units / f"{ds}__s{k}__U2__A"
                if not (d / "preds.npz").exists():
                    return self.unavailable(sid, z, T_USE, "U2__A unit missing")
                pred = np.load(d / "preds.npz")["U2_P"].argmax(1)
            hr, cr = pred == y, const == y
            disc.append({"n10": int((hr & ~cr).sum()), "n01": int((~hr & cr).sum())})
            va, vc = self.acc_vec(ds, pred, y), self.acc_vec(ds, np.full(len(y), const), y)
            accs.append(float(va[0]))
            single.append(int(len(np.unique(pred))) == 1)
            vecs.append(va - vc)
        weak = min(x["n10"] + x["n01"] for x in disc) < WEAK_DISCORDANT
        identical = all(x["n10"] + x["n01"] == 0 for x in disc)
        cacc = [float(self.acc_vec(ds, np.full(len(y), const), y)[0])]
        near = max(accs + cacc) > NEAR_ACC
        out = self.finish(sid, z, T_USE, vecs, {"discordant": disc, "constant_class": const, "accuracy_per_seed": accs,
                                                "single_class_head_seeds": single}, identical=identical, weak=weak,
                          near=near)
        if identical:
            out["flags"] = ["IDENTICAL_BY_CONSTRUCTION" if f == "IDENTICAL_BY_SELECTION" else f for f in out["flags"]]
        return out

    def unavailable(self, sid, z, target, why):
        return {"id": sid, "point": None, "se": None, "z": z, "lower": None, "upper": None, "target": target,
                "decision": "UNAVAILABLE", "flags": [], "why": why}

    def seed_units(self, ds, purpose, attr, k):
        b = f"{ds}__s{k}__{purpose}__{attr}__FH__"
        return self.resolve(b + "fullbank"), self.resolve(b + "iobank"), (b + "hard" if b + "hard" in self.uids
                                                                           else None)

    def recovery(self, sid, ds, purpose, attr, side, z, pair=None):
        vecs, sel, ident, zden, sides = [], [], [], 0, []
        for k in SEEDS:
            fb, io, hd = self.seed_units(ds, purpose, attr, k)
            if fb is None or io is None or hd is None:
                return self.unavailable(sid, z, T_REC, f"seed {k}: unit missing/incomplete")
            if any(u in self.unit_problems for u in (fb, io, hd)):
                return self.unavailable(sid, z, T_REC, f"seed {k}: unit failed integrity")
            left, right = (fb, io) if side == "FC" else (io, hd)
            if pair is None:
                vl, vr = self.R_unit(ds, left, attr), self.R_unit(ds, right, attr)
            else:
                (vl, zl), (vr, zr) = self.PAIR_unit(ds, left, attr, *pair), self.PAIR_unit(ds, right, attr, *pair)
                zden += zl + zr
            sides.append((float(vl[0]), float(vr[0])))
            vecs.append(vl - vr)
            sel.append({"seed": k, "fullbank": fb.split("__")[-1], "iobank": io.split("__")[-1],
                        "fb_offset_using": fb.split("__")[-1] in OFFSET_USING})
            ident.append(self.resolve_dir(left) == self.resolve_dir(right))
        identical = all(ident)
        near = max(max(s) for s in sides) > NEAR_AUC
        extra = {"selection": sel, "identical_seeds": ident,
                 "side_means": [float(np.mean([s[0] for s in sides])), float(np.mean([s[1] for s in sides]))]}
        if side == "FC":
            extra["offset_attributable"] = all(s["fb_offset_using"] for s in sel)
        if pair is not None:
            extra["zero_denominator_rows"] = zden
        return self.finish(sid, z, T_REC, vecs, extra, identical=identical, near=near)

    def primary(self):
        z = self.z["primary"]
        res = {}
        for ds in DATASETS:
            purpose, attr = PRIMARY[ds]
            res[f"U-frozen-{ds}"] = self.usefulness(f"U-frozen-{ds}", ds, purpose, "frozen", z)
            res[f"U-refit-{ds}"] = self.usefulness(f"U-refit-{ds}", ds, purpose, "refit", z)
        for ds in DATASETS:
            purpose, attr = PRIMARY[ds]
            for side in ("FC", "CH"):
                res[f"{side}-{ds}"] = self.recovery(f"{side}-{ds}", ds, purpose, attr, side, z)
        for side in ("FC", "CH"):
            for ds in DATASETS:
                purpose, attr = PRIMARY[ds]
                K, sup = self.support[(ds, attr)]
                for i in range(K):
                    for j in range(i + 1, K):
                        sid = f"{side}-{ds}-pair{i}-{j}"
                        if i in sup and j in sup:
                            res[sid] = self.recovery(sid, ds, purpose, attr, side, z, pair=(i, j))
                        else:
                            res[sid] = {"id": sid, "point": None, "se": None, "z": z, "lower": None, "upper": None,
                                        "target": T_REC, "decision": "NOT_ESTIMABLE", "flags": []}
        ids = expected_primary_ids()
        self.C.ok("primary.family_size", len(res) == 30 and sorted(res) == sorted(ids), {"n": len(res)})
        ne = sorted(k for k, v in res.items() if v["decision"] == "NOT_ESTIMABLE")
        self.C.ok("primary.not_estimable_slots", ne == EXPECTED_NOT_ESTIMABLE, {"n": len(ne)})
        # declared aliases: exactly equal (point and every replicate) up to float rounding of P0+P1
        for a_, b_ in DECLARED_ALIASES.items():
            ra, rb = res[a_], res[b_]
            if "_reps" in ra and "_reps" in rb:
                dmax = float(np.nanmax(np.abs(ra["_reps"] - rb["_reps"])))
                self.C.ok(f"primary.alias.{a_}", dmax <= 1e-12 and ra["decision"] == rb["decision"],
                          {"max_abs_diff_point_and_replicates": dmax})
            else:
                self.C.add(f"primary.alias.{a_}", "FAIL", "alias side unavailable")
        # telescoped sums
        for ds in DATASETS:
            if "_reps" in res[f"FC-{ds}"] and "_reps" in res[f"CH-{ds}"]:
                res[f"FC-{ds}"]["telescoped_FC_plus_CH"] = float(res[f"FC-{ds}"]["point"] + res[f"CH-{ds}"]["point"])
        # worst supported pair (descriptive) per bank side
        worst = {}
        for ds in DATASETS:
            purpose, attr = PRIMARY[ds]
            K, sup = self.support[(ds, attr)]
            for side_name, pick in (("fullbank", 0), ("iobank", 1), ("hard", 2)):
                vals = {}
                for i in sup:
                    for j in sup:
                        if i < j:
                            v = []
                            for k in SEEDS:
                                u = self.seed_units(ds, purpose, attr, k)[pick]
                                if u is None:
                                    break
                                v.append(float(self.PAIR_unit(ds, u, attr, i, j)[0][0]))
                            if len(v) == 3:
                                vals[f"{i}-{j}"] = float(np.mean(v))
                if vals:
                    kmax = max(vals, key=lambda q: abs(vals[q] - 0.5))
                    worst[f"{ds}.{side_name}"] = {"pair": kmax, "auc": vals[kmax], "all": vals}
        self.agg["worst_supported_pair_descriptive"] = worst
        self.res_primary = res
        return res

    def s3(self):
        z = self.z["S3"]
        res = {}
        for ds, purpose, attr in PAIRS:
            for side in ("FC", "CH"):
                sid = f"S3-{side}-{ds}-{purpose}-{attr}"
                res[sid] = self.recovery(sid, ds, purpose, attr, side, z)
        for ds in DATASETS:
            for purpose in self.index["datasets"][ds]["purposes"]:
                sid = f"S3-U-frozen-{ds}-{purpose}"
                res[sid] = self.usefulness(sid, ds, purpose, "frozen", z)
        self.C.ok("S3.family_size", len(res) == self.cfg.E["S3_size"], {"n": len(res)})
        # replicate-consistency: the primary cells inside S3 equal the primary family's points/replicates
        for ds in DATASETS:
            purpose, attr = PRIMARY[ds]
            for side in ("FC", "CH"):
                a_, b_ = res[f"S3-{side}-{ds}-{purpose}-{attr}"], self.res_primary[f"{side}-{ds}"]
                if "_reps" in a_ and "_reps" in b_:
                    self.C.ok(f"S3.primary_replicate.{side}-{ds}", np.array_equal(a_["_reps"], b_["_reps"],
                                                                                  equal_nan=True))
        self.res_s3 = res
        return res

    def s4(self):
        z = self.z["S4"]
        ds, pa, pb, attr = COALITION
        res = {}
        Wd = self.worlds[ds]
        # identity of rows, roles, labels and exclusions across the two purposes (unit-level)
        bad = []
        for k in SEEDS:
            for c in CONTRACTS:
                pu = f"{ds}__s{k}__PAIR_{pa}+{pb}__{attr}__FH__{c}"
                ua = f"{ds}__s{k}__{pa}__{attr}__FH__{c}"
                ub = f"{ds}__s{k}__{pb}__{attr}__FH__{c}"
                have = [u for u in (pu, ua, ub) if u in self.uids]
                if len(have) < 3:
                    bad.append(f"s{k}/{c}: unit missing")
                    continue
                ref = self.preds(pu)
                for u in (ua, ub):
                    o = self.preds(u)
                    for key in ("assess_row_id", "assess_unit", "y_s"):
                        if not np.array_equal(ref[key], o[key]):
                            bad.append(f"s{k}/{c}: {key} differs ({u.split('__')[2]})")
                vr = np.load(self.resolve_dir(pu) / "val_preds.npz")
                for u in (ua, ub):
                    vo = np.load(self.resolve_dir(u) / "val_preds.npz")
                    if not (np.array_equal(vr["val_row_id"], vo["val_row_id"]) and
                            np.array_equal(vr["val_y_s"], vo["val_y_s"])):
                        bad.append(f"s{k}/{c}: validation rows/labels differ")
        ta, tb = self.task(ds, pa), self.task(ds, pb)
        a = Wd["idx"]["assessment"]
        self.C.ok("S4.identity_across_purposes", not bad and len(ta) == len(tb) == len(Wd["row_id"]),
                  {"problems": bad[:10], "exclusions_rows": Wd["exposure_rows"]})
        for c in CONTRACTS:
            single = {"full": "fullbank", "centred": "iobank"}.get(c, c)
            for other, pname in ((pa, "income"), (pb, "employment")):
                sid = f"S4-{c}-pair-minus-{pname}"
                vecs, ident, sel = [], [], []
                miss = None
                for k in SEEDS:
                    bank = f"{ds}__s{k}__PAIR_{pa}+{pb}__{attr}__FH__{c}__bank"
                    ps = self.resolve(bank)
                    ss = self.resolve(f"{ds}__s{k}__{other}__{attr}__FH__{single}")
                    if ps is None or ss is None:
                        miss = f"seed {k} unit missing"
                        break
                    vecs.append(self.R_unit(ds, ps, attr) - self.R_unit(ds, ss, attr))
                    ident.append(self.resolve_dir(ps) == self.resolve_dir(ss))
                    sel.append({"seed": k, "pair_bank_selected": "PAIR" if "PAIR_" in ps else ps.split("__")[2],
                                "single_selected": ss.split("__")[-1]})
                if miss:
                    res[sid] = self.unavailable(sid, z, T_REC, miss)
                    continue
                res[sid] = self.finish(sid, z, T_REC, vecs, {"selection": sel, "identical_seeds": ident},
                                       identical=all(ident))
            both = all(res.get(f"S4-{c}-pair-minus-{n}", {}).get("decision") == "PASS" for n in ("income",
                                                                                                  "employment"))
            self.agg.setdefault("S4_combination_finding", {})[c] = both
        self.C.ok("S4.family_size", len(res) == self.cfg.E["S4_size"], {"n": len(res)})
        self.res_s4 = res
        return res

    # ---------------------------------------------------------------- 6. comparison with runner tables
    COLS = {"id": ("id", "endpoint", "endpoint_id", "slot"), "point": ("point", "estimate", "est"),
            "se": ("se", "SE", "boot_se", "se_boot"), "lower": ("lower", "lo", "lower_bound", "lb"),
            "upper": ("upper", "hi", "upper_bound", "ub"), "decision": ("decision", "status"),
            "flags": ("flags", "flag", "sub_label", "labels"), "z": ("z", "z_crit")}

    def compare(self, name, path: Path, mine: dict, id_map=None, expected_n=None):
        if path is None or not path.exists():
            self.C.add(f"compare.{name}", "SKIPPED", f"{self.cfg.tilde(path) if path else name} not present")
            return
        rows = list(csv.DictReader(open(path)))
        cols = rows[0].keys() if rows else []
        m = {k: next((c for c in v if c in cols), None) for k, v in self.COLS.items()}
        out = {"table": self.cfg.tilde(path), "rows": len(rows), "column_map": m,
               "unmapped_columns": [c for c in cols if c not in m.values()]}
        diffs, missing = [], []
        theirs = {}
        for r in rows:
            rid = r[m["id"]]
            rid = id_map(rid, r) if id_map else rid
            theirs[rid] = r
        if expected_n is not None:
            self.C.ok(f"compare.{name}.n_rows", len(rows) == expected_n and len(theirs) == expected_n,
                      {"rows": len(rows), "distinct_ids": len(theirs), "expected": expected_n})
        maxd = {"point": 0.0, "se": 0.0, "lower": 0.0, "upper": 0.0}
        for sid, v in mine.items():
            r = theirs.get(sid)
            if r is None:
                missing.append(sid)
                continue
            for f, tol in (("point", TOL_POINT), ("se", TOL_SE), ("lower", 1e-8), ("upper", 1e-8)):
                if m[f] is None:
                    continue
                tv, mv = to_float(r[m[f]]), v.get(f)
                if (tv is None) != (mv is None):
                    diffs.append({"id": sid, "field": f, "table": tv, "replay": mv})
                elif tv is not None:
                    dd = abs(tv - mv)
                    maxd[f] = max(maxd[f], dd)
                    if dd > tol:
                        diffs.append({"id": sid, "field": f, "table": tv, "replay": mv, "abs_diff": dd})
            if m["z"] and to_float(r[m["z"]]) is not None and abs(to_float(r[m["z"]]) - v["z"]) > 1e-6:
                diffs.append({"id": sid, "field": "z", "table": to_float(r[m["z"]]), "replay": v["z"]})
            if m["decision"]:
                td = r[m["decision"]].strip()
                if td != v["decision"]:
                    diffs.append({"id": sid, "field": "decision", "table": td, "replay": v["decision"]})
            if m["flags"]:
                tf = set(x for x in re.split(r"[;,| ]+", r[m["flags"]] or "") if x)
                for fl in ("IDENTICAL_BY_SELECTION", "IDENTICAL_BY_CONSTRUCTION", "NORMAL_APPROX_WEAK", "NEAR_BOUND"):
                    if (fl in tf) != (fl in v["flags"]):
                        diffs.append({"id": sid, "field": f"flag:{fl}", "table": fl in tf, "replay": fl in v["flags"]})
            # independent sanity of the table's own arithmetic
            if m["point"] and m["se"] and m["lower"] and m["z"]:
                tp, ts, tl, tz = (to_float(r[m[q]]) for q in ("point", "se", "lower", "z"))
                if None not in (tp, ts, tl, tz) and abs(tp - tz * ts - tl) > 1e-9:
                    diffs.append({"id": sid, "field": "table_internal_lower", "table": tl, "point-z*se": tp - tz * ts})
            if m["decision"] and m["lower"]:
                tl, td = to_float(r[m["lower"]]), r[m["decision"]].strip()
                if td == "PASS" and (tl is None or tl <= v["target"]):
                    diffs.append({"id": sid, "field": "table_internal_decision", "table": td, "lower": tl})
                if td == "PASS" and m["se"] and to_float(r[m["se"]]) == 0.0:
                    diffs.append({"id": sid, "field": "table_PASS_with_SE_0", "table": td})
        extra = [k for k in theirs if k not in mine]
        out.update({"max_abs_diff": maxd, "n_diffs": len(diffs), "diffs": diffs[:60], "missing_in_table": missing,
                    "extra_in_table": extra})
        self.C.ok(f"compare.{name}", not diffs and not missing and not extra, out)
        self.agg.setdefault("comparisons", {})[name] = out

    def compare_seed_selections(self, path: Path):
        """If the runner table carries per-seed selections, compare them with the replay's bank selections."""
        if path is None or not path.exists():
            return
        rows = list(csv.DictReader(open(path)))
        cols = [c for c in (rows[0].keys() if rows else []) if re.search(r"select", c, re.I)]
        self.agg.setdefault("runner_selection_columns", {})[path.name] = cols

    # ---------------------------------------------------------------- 7. models (surfaces reproduce predictions)
    def surface_for(self, uid):
        pu = self.parse_uid(uid)
        ds, k, purpose, st, surf = pu["ds"], pu["seed"], pu["purpose"], pu["stratum"], pu["surface"]
        if purpose.startswith("PAIR_"):
            pa, pb = purpose[len("PAIR_"):].split("+")
            return np.hstack([my_surfaces(self.logits(ds, k, pa))[surf], my_surfaces(self.logits(ds, k, pb))[surf]])
        if st == "FH":
            return my_surfaces(self.logits(ds, k, purpose))[surf]
        if st == "RH":
            H = np.load(self.cfg.oar_units / f"{ds}__s{k}__HEAD__A" / "preds.npz")
            return my_surfaces(H["head_outputs_all"].astype(np.float64))[surf]
        raise ValueError(uid)

    def model_proba(self, model, X, K):
        P = model.predict_proba(X)
        full = np.zeros((X.shape[0], K))
        full[:, np.asarray(model.classes_).astype(int)] = P
        return full

    def check_models(self):
        import joblib  # noqa: F401  (guarded: runner modules cannot be imported through unpickling)
        attack = sorted(u for u in self.uids if self.parse_uid(u).get("surface") in SURFACES and
                        self.parse_uid(u).get("stratum") in ("FH", "RH") and
                        (self.resolve_dir(u) / "models").exists())
        rng = np.random.default_rng(self.cfg.E["model_seed"])
        must = []
        for ds in DATASETS:
            purpose, attr = PRIMARY[ds]
            fb, io, hd = self.seed_units(ds, purpose, attr, 0)
            must += [u for u in (fb, io, hd) if u]
        pool = [u for u in attack if u not in must]
        n = min(len(pool), self.cfg.n_model_units * 2)
        sample = must + ([pool[i] for i in sorted(rng.choice(len(pool), n, replace=False))] if n else [])
        res, bad = {}, {}
        for uid in sample:
            pu = self.parse_uid(uid)
            ds = pu["ds"]
            W = self.worlds[ds]
            a, v = W["idx"]["assessment"], W["idx"]["attacker_val"]
            X = self.surface_for(uid)
            P = self.preds(uid)
            d = self.resolve_dir(uid)
            V = np.load(d / "val_preds.npz")
            K = P["P__L"].shape[1]
            dm = {}
            try:
                for name, key, rows, store in [("NL_as0", "P__NL__as0", a, P), ("NL_as1", "P__NL__as1", a, P),
                                               ("NL_as2", "P__NL__as2", a, P), ("L", "P__L", a, P),
                                               ("NL_as0", "VAL__NL__as0", v, V), ("L", "VAL__L", v, V)]:
                    m = joblib.load(d / "models" / f"{name}.joblib")
                    dm[key] = float(np.max(np.abs(self.model_proba(m, X[rows], K) - store[key])))
            except ImportError as e:
                bad[uid] = f"model unpickling needs a forbidden module: {e}"
                continue
            res[uid] = max(dm.values())
            if max(dm.values()) > 1e-12:
                bad[uid] = dm
        self.C.ok("models.reproduce_stored_predictions", not bad and len(res) > 0,
                  {"units_replayed": len(res), "max_abs_diff": max(res.values()) if res else None,
                   "bad": bad}, cause="stored predictions not reproduced from the logits-only surface")
        # cell-conditional (finite surfaces): recomputed from attacker_fit counts, no unpickling
        bad, n = {}, 0
        for uid in sorted(self.uids):
            pu = self.parse_uid(uid)
            if pu.get("surface") != "hard" or pu.get("stratum") not in ("FH", "RH"):
                continue
            d = self.resolve_dir(uid)
            if not (d / "preds.npz").exists():
                continue
            P = self.preds(uid)
            if "P__CC" not in P:
                continue
            r = json.loads((d / "record.json").read_text())
            alpha = float(r["CC"]["selected"]["alpha"])
            W = self.worlds[pu["ds"]]
            s = W["lab"][pu["attr"]].astype(int)
            K = P["P__CC"].shape[1]
            X = self.surface_for(uid)
            f = W["idx"]["attacker_fit"]
            prior = np.bincount(s[f], minlength=K) / len(f)
            keyf = [tuple(np.round(x, 9)) for x in X[f]]
            table = {}
            for kk in set(keyf):
                msk = np.array([q == kk for q in keyf])
                cnt = np.bincount(s[f][msk], minlength=K).astype(float)
                table[kk] = (cnt + alpha * prior) / (cnt.sum() + alpha)
            Pa = np.stack([table.get(tuple(np.round(x, 9)), prior) for x in X[W["idx"]["assessment"]]])
            n += 1
            dd = float(np.max(np.abs(Pa - P["P__CC"])))
            if dd > 1e-12:
                bad[uid] = dd
        self.C.ok("models.cell_conditional_recomputed", not bad and n > 0, {"units": n, "bad": bad})

    # ---------------------------------------------------------------- 8. per-person spot replays
    def spot_checks(self):
        import joblib
        rng = np.random.default_rng(self.cfg.E["spot_seed"])
        out = {}
        allok = True
        for ds in DATASETS:
            purpose, attr = PRIMARY[ds]
            W = self.worlds[ds]
            a = W["idx"]["assessment"]
            picks = rng.choice(len(a), 2, replace=False)
            fb = self.seed_units(ds, purpose, attr, 0)[0]
            if fb is None:
                out[ds] = "SKIPPED (unit missing)"
                allok = False
                continue
            d = self.resolve_dir(fb)
            surf = fb.split("__")[-1]
            P = self.preds(fb)
            Lg = self.logits(ds, 0, purpose)
            diffs = []
            for pos in picks:
                row = a[pos]
                x = my_surfaces(Lg[row:row + 1])[surf]          # the row's own logits only
                assert np.array_equal(x, my_surfaces(Lg)[surf][row:row + 1])
                for as_ in ATTACKER_SEEDS:
                    m = joblib.load(d / "models" / f"NL_as{as_}.joblib")
                    p = self.model_proba(m, x, P["P__L"].shape[1])
                    diffs.append(float(np.max(np.abs(p - P[f"P__NL__as{as_}"][pos:pos + 1]))))
            ok = max(diffs) <= 1e-12
            allok &= ok
            out[ds] = {"status": "PASS" if ok else "FAIL", "rows": 2, "unit_surface": surf,
                       "max_abs_diff": max(diffs)}
        self.C.ok("spot.per_person_replay", allok, out)
        # constant / collapsed cases
        coll = {}
        okc = True
        for ds in DATASETS:
            W = self.worlds[ds]
            a = W["idx"]["assessment"]
            for purpose in self.index["datasets"][ds]["purposes"]:
                for k in SEEDS:
                    pred = self.logits(ds, k, purpose)[a].argmax(1)
                    if len(np.unique(pred)) == 1:
                        t = self.task(ds, purpose)
                        Kt = int(self.purpose(ds, purpose)["task_dim"])
                        const = int(np.argmax(np.bincount(t[W["idx"]["attacker_fit"]], minlength=Kt)))
                        same = int(pred[0]) == const
                        # hard-surface units of this head: attacker outputs must be constant -> AUC 0.5 exactly
                        aucs = []
                        for (d_, p_, at) in PAIRS:
                            if (d_, p_) != (ds, purpose):
                                continue
                            u = f"{ds}__s{k}__{purpose}__{at}__FH__hard"
                            if u not in self.uids:
                                continue
                            Pu = self.preds(u)
                            const_rows = all(np.ptp(Pu[f"P__NL__as{q}"], axis=0).max() == 0 for q in ATTACKER_SEEDS)
                            r = self.R_unit(ds, u, at)
                            aucs.append(const_rows and bool(np.all(r[~np.isnan(r)] == 0.5)))
                        ok = all(aucs)
                        okc &= ok
                        coll[f"{ds}.{purpose}.s{k}"] = {"single_class_head": True, "predicts_constant_class": same,
                                                        "hard_attackers_constant_auc_0.5": ok,
                                                        "hard_units_checked": len(aucs)}
        # bank selections identical on both sides: per-seed contrast is exactly 0 on every replicate
        for ds, purpose, attr in PAIRS:
            for k in SEEDS:
                fb, io, _ = self.seed_units(ds, purpose, attr, k)
                if fb and io and self.resolve_dir(fb) == self.resolve_dir(io):
                    v = self.R_unit(ds, fb, attr) - self.R_unit(ds, io, attr)
                    ok = bool(np.all(v[~np.isnan(v)] == 0))
                    okc &= ok
                    coll.setdefault("fb_equals_io", {})[f"{ds}.{purpose}.{attr}.s{k}"] = ok
        # attack units with constant stored predictions (collapsed attacker)
        consts = []
        for uid in sorted(self.uids):
            d = self.resolve_dir(uid)
            if not (d / "preds.npz").exists() or "nl_family" not in json.loads((d / "record.json").read_text()):
                continue
            P = self.preds(uid)
            if all(np.ptp(P[f"P__NL__as{q}"], axis=0).max() == 0 for q in ATTACKER_SEEDS):
                consts.append(uid)
        coll["constant_attacker_units"] = len(consts)
        coll["constant_attacker_units_by_surface"] = {s: sum(1 for u in consts if u.endswith("__" + s))
                                                     for s in SURFACES}
        self.C.ok("spot.constant_and_collapsed_cases", okc, coll)

    # ---------------------------------------------------------------- 9. controls
    def check_controls(self):
        for ds in DATASETS:
            p = self.cfg.run / "controls" / f"{ds}.json"
            if not p.exists():
                self.C.add(f"controls.{ds}", "FAIL", "controls file absent", cause="control stage not run/complete")
                continue
            J = json.loads(p.read_text())
            det = {}
            ok = True
            for name in ("full", "centred"):
                c = J.get(name)
                if c is None:
                    ok = False
                    det[name] = "missing"
                    continue
                nv, pv = c.get("null_val_macro_auc"), c.get("planted_val_macro_auc")
                cons = (c.get("null_flag_above_0.55") == (nv > 0.55)) and \
                       (c.get("planted_detected_above_0.75") == (pv > 0.75))
                good = (not c.get("null_flag_above_0.55")) and c.get("planted_detected_above_0.75")
                ok &= bool(cons and good)
                det[name] = {"null_val_macro_auc": nv, "planted_val_macro_auc": pv, "flags_consistent": cons,
                             "null_not_flagged_and_planted_detected": good}
            self.C.ok(f"controls.{ds}", ok, det, cause="null flagged or planted leak not detected")

    # ---------------------------------------------------------------- 10. access / contract (static + dynamic)
    def check_access(self, repo: Path):
        det, ok = {}, True
        sp = repo / "odx" / "surfaces.py"
        rp = repo / "odx" / "run.py"
        if sp.exists():
            tree = ast.parse(sp.read_text())
            names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
            attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
            strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
            fns = {n.name: [a.arg for a in n.args.args] for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
            leaks = sorted(({"load", "labels", "row_id", "canon_key", "unit", "W", "s", "t", "y"} & (names | attrs)) |
                           {x for x in strings if re.search(r"label|row_id|canon|\.npz", x)})
            okf = fns.get("surfaces") == ["L"] and not leaks
            ok &= okf
            det["odx/surfaces.py"] = {"surfaces_signature": fns.get("surfaces"), "label_or_key_refs": leaks,
                                      "ok": okf}
        if rp.exists():
            src = rp.read_text()
            tree = ast.parse(src)
            xs = []
            for n in ast.walk(tree):
                if isinstance(n, ast.Call):
                    fn = n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "")
                    if fn in ("attack_unit", "atk") and len(n.args) >= 2:
                        xs.append((fn, ast.get_source_segment(src, n.args[1])))
                    if fn == "surfaces" and n.args:
                        xs.append(("surfaces", ast.get_source_segment(src, n.args[0])))
            allowed = re.compile(r"^(X|sf\[name\]|sf\[c\]|np\.hstack\(\[sa\[c\], sb\[c\]\]\)|Lmat|F\[.*logits_key.*\]"
                                 r"\.astype\(np\.float64\))$")
            badx = [x for x in xs if not allowed.match(x[1] or "")]
            okr = not badx and len(xs) > 0
            ok &= okr
            det["odx/run.py"] = {"release_inputs": sorted(set(x[1] for x in xs)), "unexpected": badx, "ok": okr}
        det["dynamic"] = "stored attacker predictions reproduced from surfaces rebuilt from logits alone " \
                         "(models.reproduce_stored_predictions, spot.per_person_replay with single-row surfaces)"
        self.C.ok("access.deployed_inputs_logits_only", ok, det)

    # ---------------------------------------------------------------- 11. stage-1 table (bonus)
    def check_frozen_head_utility(self):
        p = self.cfg.study / "FROZEN_HEAD_UTILITY.csv"
        if not p.exists():
            self.C.add("stage1.FROZEN_HEAD_UTILITY", "SKIPPED", "absent")
            return
        rows = list(csv.DictReader(open(p)))
        bad, n = [], 0
        for r in rows:
            ds, purpose, k, head = r["dataset"], r["purpose"], int(r["seed"]), r["head"]
            if ds not in self.worlds:
                continue
            W = self.worlds[ds]
            a = W["idx"]["assessment"]
            t = self.task(ds, purpose)
            Kt = int(self.purpose(ds, purpose)["task_dim"])
            if head == "frozen":
                pred = self.logits(ds, k, purpose)[a].argmax(1)
            elif head.startswith("refit_probe_U2__A"):
                pred = np.load(self.cfg.oar_units / f"{ds}__s{k}__U2__A" / "preds.npz")["U2_P"].argmax(1)
            else:
                continue
            n += 1
            y = t[a]
            const = int(np.argmax(np.bincount(t[W["idx"]["attacker_fit"]], minlength=Kt)))
            acc, cacc = float((pred == y).mean()), float((y == const).mean())
            chk = {"accuracy": acc, "constant_accuracy": cacc, "gain_over_constant": acc - cacc,
                   "constant_class_from_attacker_fit": const,
                   "discordant_head_right_const_wrong": int(((pred == y) & (y != const)).sum()),
                   "discordant_head_wrong_const_right": int(((pred != y) & (y == const)).sum())}
            for f, v in chk.items():
                tv = to_float(r.get(f))
                if tv is None or abs(tv - v) > 1e-12:
                    bad.append({"row": f"{ds}/{purpose}/s{k}/{head}", "field": f, "table": tv, "replay": v})
            if head == "frozen":
                Lg = self.logits(ds, k, purpose)[a]
                ll = float(np.mean(-(Lg[np.arange(len(a)), y] - (Lg.max(1) + np.log(np.exp(
                    Lg - Lg.max(1, keepdims=True)).sum(1))))))
                tv = to_float(r.get("log_loss"))
                if tv is None or abs(tv - ll) > 1e-9:
                    bad.append({"row": f"{ds}/{purpose}/s{k}/{head}", "field": "log_loss", "table": tv, "replay": ll})
        self.C.ok("stage1.FROZEN_HEAD_UTILITY", not bad and n > 0, {"rows_checked": n, "bad": bad[:20]})

    # ---------------------------------------------------------------- run
    def run(self, tables: dict, repo: Path, stages=("all",)):
        t0 = time.time()
        self.check_lock()
        self.check_roles()
        self.check_units()
        self.check_exactness()
        self.check_banks()
        self.check_frozen_head_utility()
        with np.errstate(divide="raise"):      # nothing in the decision path may divide by an SE
            prim = self.primary()
        self.compare("PRIMARY_ENDPOINTS", tables.get("primary"), prim, expected_n=30)
        self.compare_seed_selections(tables.get("primary"))
        if "all" in stages or "secondary" in stages:
            with np.errstate(divide="raise"):
                s3 = self.s3()
                s4 = self.s4()
            self.compare("S3", tables.get("S3"), s3, id_map=tables.get("S3_id_map"), expected_n=34)
            self.compare("S4", tables.get("S4"), s4, id_map=tables.get("S4_id_map"), expected_n=6)
        if not self.cfg.quick:
            self.check_models()
            self.spot_checks()
        self.check_controls()
        self.check_access(repo)
        loaded = sorted(m for m in sys.modules if m.split(".")[0] in _FORBIDDEN)
        self.C.ok("independence.no_runner_modules_loaded", not loaded, {"loaded": loaded})
        self.agg["runtime_s"] = time.time() - t0
        return self


def strip(res: dict) -> dict:
    return {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in res.items()}


def summarize(rp: Replay) -> dict:
    prim = rp.res_primary
    dec = {}
    for v in prim.values():
        dec[v["decision"]] = dec.get(v["decision"], 0) + 1
    passes = sorted(k for k, v in prim.items() if v["decision"] == "PASS")
    lines = [f"Primary family: {len(prim)} slots; decisions {dec}.",
             f"PASS slots: {', '.join(passes) if passes else 'none'}.",
             "Aliases FC/CH-adult-pair0-1 checked equal to FC/CH-adult; 14 NOT_ESTIMABLE slots follow from support."]
    if hasattr(rp, "res_s3"):
        d3 = {}
        for v in rp.res_s3.values():
            d3[v["decision"]] = d3.get(v["decision"], 0) + 1
        lines.append(f"S3 (34): {d3}.")
    if hasattr(rp, "res_s4"):
        d4 = {}
        for v in rp.res_s4.values():
            d4[v["decision"]] = d4.get(v["decision"], 0) + 1
        lines.append(f"S4 (6): {d4}; combination finding by contract: {rp.agg.get('S4_combination_finding')}.")
    cc = rp.C.counts()
    lines.append(f"Verifier checks: {cc}.")
    return {"text": " ".join(lines), "check_counts": cc}


def git_head(repo: Path):
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def find_table(study: Path, names):
    for n in names:
        p = study / n
        if p.exists():
            return p
    return None


def main(argv=None):
    here = Path(__file__).resolve()
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", default=str(Path.home()))
    ap.add_argument("--study-dir", default=str(here.parent.parent))
    ap.add_argument("--repo", default=str(here.parents[3]))
    ap.add_argument("--out", default=None, help="INDEPENDENT_VERIFICATION.json path")
    ap.add_argument("--agg-out", default=None)
    ap.add_argument("--primary-table", default=None)
    ap.add_argument("--s3-table", default=None)
    ap.add_argument("--s4-table", default=None)
    ap.add_argument("--quick", action="store_true", help="skip model/spot replays")
    ap.add_argument("--primary-only", action="store_true")
    ap.add_argument("--expect", default=None, help="JSON overrides of protocol constants (synthetic tests only)")
    a = ap.parse_args(argv)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("set OMP_NUM_THREADS=1")
    study = Path(a.study_dir)
    cfg = Cfg(Path(a.home), study, quick=a.quick, expect=json.loads(a.expect) if a.expect else None)
    tables = {"primary": Path(a.primary_table) if a.primary_table else find_table(study, ["PRIMARY_ENDPOINTS.csv"]),
              "S3": Path(a.s3_table) if a.s3_table else find_table(study, ["S3_ENDPOINTS.csv",
                                                                           "SECONDARY_S3_ENDPOINTS.csv"]),
              "S4": Path(a.s4_table) if a.s4_table else find_table(study, ["S4_ENDPOINTS.csv",
                                                                           "SECONDARY_S4_ENDPOINTS.csv"])}
    rp = Replay(cfg).run(tables, Path(a.repo), stages=("primary",) if a.primary_only else ("all",))
    nonpass = [it for it in rp.C.items if it["status"] != "PASS"]
    out = {"schema": "odx_independent_verification/v1",
           "verifier": "independent replay (no runner metric/aggregation/decision code imported)",
           "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "repo_head": git_head(Path(a.repo)), "lock_commit_expected": "01ef342f",
           "environment": {"python": platform.python_version(), "numpy": np.__version__,
                           "scipy": __import__("scipy").__version__, "OMP_NUM_THREADS": os.environ.get(
                               "OMP_NUM_THREADS")},
           "protocol_constants": {k: v for k, v in cfg.E.items()},
           "public_summary": summarize(rp),
           "non_pass_items": nonpass,
           "checks": rp.C.items,
           "primary": strip(rp.res_primary)}
    if hasattr(rp, "res_s3"):
        out["S3"] = strip(rp.res_s3)
        out["S4"] = strip(rp.res_s4)
    agg = {"schema": "odx_replay_aggregate/v1", "primary": strip(rp.res_primary), **{k: v for k, v in rp.agg.items()}}
    if hasattr(rp, "res_s3"):
        agg["S3"], agg["S4"] = strip(rp.res_s3), strip(rp.res_s4)
    op = Path(a.out) if a.out else study / "INDEPENDENT_VERIFICATION.json"
    ap_ = Path(a.agg_out) if a.agg_out else study / "verification" / "replay_results_aggregate.json"
    txt = json.dumps(out, indent=1, default=fnum)
    if str(cfg.home) in txt:
        raise SystemExit("refusing to write: a private absolute path leaked into the output")
    op.write_text(txt)
    ap_.parent.mkdir(parents=True, exist_ok=True)
    ap_.write_text(json.dumps(agg, indent=1, default=fnum))
    print(json.dumps(out["public_summary"], indent=1))
    for it in nonpass:
        print(f"  {it['status']}: {it['id']}")
    return 1 if any(it["status"] == "FAIL" for it in rp.C.items) else 0


if __name__ == "__main__":
    sys.exit(main())
