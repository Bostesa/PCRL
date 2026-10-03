"""Independent replay of the output-aware removal study (oar v1) from saved per-row predictions.

Written by the independent verifier. It does NOT import oar.* or stored_model_eval.*; the runner code was read only
to learn file layouts. Scoring uses numpy only (no sklearn, no scipy). sklearn/joblib are imported only inside the
view-3 head contract check, to load the saved head and recompute its outputs (not a scoring path).

What it recomputes (see the verifier brief):
  1. roles      oar-roles-v1 from labels.npz (exposure removal, oar-cert-v1 carve-out, oar-head-v1 holdout); counts,
                group counts and row-id hashes vs ROLES_AND_SUPPORT.json; class vocabulary and support vs the lock.
  2. leakage    assessment / val / fit / cert disjointness; every unit's assess_row_id / val_row_id vs the roles;
                FARE fit rows (fit_rows_sha256 / fit_labels_sha256) vs defense_fit.
  3. primary    the 12 family rows: recovery (own Mann-Whitney AUC, ties 1/2, weighted in the bootstrap), U2 accuracy,
                plus-surface aliasing, validation-only FARE nominee, mirrored group bootstrap (B 20,000, seed
                20261011, chunk 500), simultaneous one-sided bound at 0.05/12 (numpy 'linear'), decision.
  4. membership identical assessment rows; seed sets; sigma* and release seeds; alias / duplicate counts.
  5. contracts  view-3 head = deterministic function of the protected features only; FARE certificates reported
                only with premises (UNAVAILABLE where premises fail; cell-presence premise recomputed).
  6. stage 1/2  C1 seed-paired contrasts (marital_status cell), C2 stable rho1^2 points (+ intervals of the
                corrected rows); exposure sensitivity of the 24 original benchmark endpoints on retained rows.

Usage (real run, after "run complete"):
  OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python results/combined_output_aware_removal_v1/verification/replay_oar.py

Outputs: <pkg>/INDEPENDENT_VERIFICATION.json and <pkg>/verification/replay_results_aggregate.json (aggregates only,
home-relative paths). Every disagreement is kept with both values; nothing is forced to match.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HOME = Path.home()
HERE = Path(__file__).resolve().parent
PKG_DEFAULT = HERE.parent
WT_DEFAULT = HERE.parents[2]

SCORED = ("attacker_fit", "attacker_val", "assessment")
SEEDS = (0, 1, 2)
ATT = (0, 1, 2)
REL = (0, 1, 2)
CHUNK = 500
CLIP = 1e-12
# exposure counts stated in PROTOCOL.md / EXPOSURE_RULE.md: test rows (all roles), assessment rows
EXPECT_EXPOSURE = {"adult": [17, 7], "hmda": [42, 14]}

# cell specification (from PROTOCOL.md / RELEASE_CONTRACTS.md; cross-checked against the lock's class vocabulary)
CELLS = {
    "adult": {"purpose": "income_prediction", "attr": "sex", "task": "task_income", "rep": "rep_p0",
              "logits": "logits_income_prediction", "K_s": 2, "K_t": 2},
    "hmda": {"purpose": "underwriting", "attr": "race", "task": "task_loan_decision", "rep": "rep_p0",
             "logits": "logits_underwriting", "K_s": 5, "K_t": 2},
}


def tilde(p) -> str:
    s = str(p)
    h = str(HOME)
    return "~" + s[len(h):] if s.startswith(h) else s


def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha_arrays(*arrays) -> str:
    """Hash convention used by the study for arrays: dtype string + shape string + raw bytes, concatenated."""
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def u01(salt_key: str) -> float:
    return int(hashlib.sha256(salt_key.encode()).hexdigest()[:8], 16) / 2 ** 32


# ================================================================================================ report
class Report:
    def __init__(self):
        self.items = []

    def add(self, scope, iid, status, detail=None, cause=None, **vals):
        assert status in ("PASS", "FAIL", "MC_BORDERLINE", "UNRESOLVED", "NOT_CHECKED", "FLAG"), status
        it = {"scope": scope, "id": iid, "status": status}
        if detail is not None:
            it["detail"] = detail
        if cause is not None:
            it["cause"] = cause
        it.update(vals)
        self.items.append(it)
        return it

    def check(self, scope, iid, ok, detail=None, cause=None, **vals):
        return self.add(scope, iid, "PASS" if ok else "FAIL", detail, None if ok else cause, **vals)


# ================================================================================================ statistics
def midrank_auc(score, pos) -> float:
    """Unweighted Mann-Whitney AUC from midranks (ties count 1/2). Independent of the weighted path."""
    score = np.asarray(score, np.float64)
    pos = np.asarray(pos, bool)
    n1 = int(pos.sum())
    n0 = len(pos) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    _, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    start = np.cumsum(cnt) - cnt
    ranks = (start + (cnt + 1) / 2.0)[inv]
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


class AUCBase:
    """Weighted Mann-Whitney AUC: sum_{i pos, j neg} w_i w_j [s_i > s_j] + 1/2 [s_i = s_j], over (sum w_pos)(sum w_neg).
    Per positive row, the negative weight strictly below / equal is read from a cumulative sum over negatives sorted by
    score (searchsorted left/right)."""

    def __init__(self, score, pos):
        score = np.asarray(score, np.float64)
        pos = np.asarray(pos, bool)
        if not np.all(np.isfinite(score)):
            raise ValueError("non-finite score")
        self.score, self.pos = score, pos
        self.pi = np.flatnonzero(pos)
        ni = np.flatnonzero(~pos)
        o = np.argsort(score[ni], kind="stable")
        self.ni = ni[o]
        ns = score[self.ni]
        ps = score[self.pi]
        self.left = np.searchsorted(ns, ps, "left")
        self.right = np.searchsorted(ns, ps, "right")

    def eval(self, WT):
        c = WT.shape[1]
        cum = np.zeros((len(self.ni) + 1, c))
        np.cumsum(WT[self.ni], axis=0, out=cum[1:])
        A = 0.5 * (cum[self.left] + cum[self.right])
        Wp = WT[self.pi]
        num = (Wp * A).sum(0)
        den = Wp.sum(0) * cum[-1]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, np.nan)

    def point(self):
        return midrank_auc(self.score, self.pos)


class MeanBase:
    """Weighted mean of a per-row vector (accuracy indicator, log loss)."""

    def __init__(self, vec):
        self.v = np.asarray(vec, np.float64)

    def eval(self, WT):
        d = WT.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(d > 0, (self.v @ WT) / d, np.nan)

    def point(self):
        return float(np.mean(self.v))


class SkillBase:
    """1 - sum w num / sum w den (Brier-type R^2 against a fixed prior)."""

    def __init__(self, num, den):
        self.n, self.d = np.asarray(num, np.float64), np.asarray(den, np.float64)

    def eval(self, WT):
        d = self.d @ WT
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(d > 0, 1.0 - (self.n @ WT) / d, np.nan)

    def point(self):
        d = self.d.sum()
        return float(1.0 - self.n.sum() / d) if d > 0 else float("nan")


class RhoBase:
    """Weighted squared Pearson correlation of (u, v), computed on centred float64 copies (shift-invariant)."""

    def __init__(self, u, v):
        u = np.asarray(u, np.float64)
        v = np.asarray(v, np.float64)
        self.u, self.v = u - u.mean(), v - v.mean()

    def eval(self, WT):
        u, v = self.u, self.v
        s0 = WT.sum(0)
        mu, mv = (u @ WT) / s0, (v @ WT) / s0
        cuv = ((u * v) @ WT) / s0 - mu * mv
        vu = ((u * u) @ WT) / s0 - mu * mu
        vv = ((v * v) @ WT) / s0 - mv * mv
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where((vu > 0) & (vv > 0), cuv ** 2 / (vu * vv), np.nan)

    def point(self):
        u, v = self.u - self.u.mean(), self.v - self.v.mean()
        vu, vv = float(np.mean(u * u)), float(np.mean(v * v))
        if vu <= 0 or vv <= 0:
            return float("nan")
        return float(np.mean(u * v) ** 2 / (vu * vv))


def draw_chunks(assess_unit, B, seed, chunk=CHUNK):
    """Frozen draw: rng = default_rng(seed); per replicate counts = rng.multinomial(n_units, full(n_units, 1/n_units));
    row weight = counts[index of the row's unit in numpy.unique order]; chunks of `chunk` in sequential RNG order."""
    _, inv = np.unique(np.asarray(assess_unit), return_inverse=True)
    n_units = int(inv.max()) + 1
    rng = np.random.default_rng(seed)
    p = np.full(n_units, 1.0 / n_units)
    done = 0
    while done < B:
        c = min(chunk, B - done)
        counts = np.empty((n_units, c))
        for j in range(c):
            counts[:, j] = rng.multinomial(n_units, p)
        yield counts[inv]
        done += c


class Engine:
    """Base statistics registered once by content key; replicate arrays computed in one pass over the frozen draw."""

    def __init__(self):
        self.bases = {}

    def reg(self, key, factory):
        if key not in self.bases:
            self.bases[key] = factory()
        return key

    def points(self):
        return {k: b.point() for k, b in self.bases.items()}

    def points_weighted_ones(self, n):
        ones = np.ones((n, 1))
        return {k: float(b.eval(ones)[0]) for k, b in self.bases.items()}

    def replicates(self, assess_unit, B, seed, keys=None):
        keys = list(self.bases) if keys is None else list(keys)
        out = {k: np.empty(B) for k in keys}
        done = 0
        for WT in draw_chunks(assess_unit, B, seed):
            c = WT.shape[1]
            for k in keys:
                out[k][done:done + c] = self.bases[k].eval(WT)
            done += c
        return out


def combine(expr, vals):
    """expr: list of (coef, base key). Returns sum coef * value (NaN propagates)."""
    tot = 0.0
    for c, k in expr:
        tot = tot + c * vals[k]
    return tot


def hier_mean(parts):
    """parts: list of sub-expressions (each a list of (coef, key)); equal-weight mean of the parts."""
    n = len(parts)
    return [(c / n, k) for p in parts for c, k in p]


def quant(r, q):
    return float(np.quantile(r, q, method="linear"))


def bound_pair(r, a_lo, a_hi, max_ne):
    ok = np.isfinite(r)
    n_ne = int((~ok).sum())
    if ok.sum() == 0 or n_ne / len(r) > max_ne:
        return None, None, n_ne
    v = r[ok]
    return quant(v, a_lo), quant(v, a_hi), n_ne


def mc_se(r, a):
    """Monte Carlo SE of the a-quantile: half the spread between the quantiles at a +/- sqrt(a(1-a)/B)."""
    v = r[np.isfinite(r)]
    if len(v) < 2:
        return float("nan")
    s = np.sqrt(a * (1 - a) / len(v))
    lo, hi = max(a - s, 0.0), min(a + s, 1.0)
    return (quant(v, hi) - quant(v, lo)) / 2.0


def hash_key(*arrays) -> str:
    return sha_arrays(*[np.asarray(a) for a in arrays])[:32]


def auc_expr(eng, y, P, classes, tag=""):
    """Macro one-vs-rest AUC over supported classes: one base per class (content-keyed)."""
    y = np.asarray(y).astype(int)
    P = np.asarray(P, np.float64)
    parts = []
    for c in classes:
        key = "auc|" + hash_key(P[:, c], y == c)
        eng.reg(key, lambda c=c: AUCBase(P[:, c], y == c))
        parts.append([(1.0, key)])
    return hier_mean(parts)


def acc_expr(eng, y, P):
    corr = (np.asarray(P).argmax(1) == np.asarray(y).astype(int)).astype(np.float64)
    key = "acc|" + hash_key(corr)
    eng.reg(key, lambda: MeanBase(corr))
    return [(1.0, key)]


# ================================================================================================ io helpers
class UnitStore:
    """Read-only access to unit directories; verifies COMPLETE.json file hashes once per unit."""

    def __init__(self, root: Path, rep: Report, scope: str):
        self.root, self.rep, self.scope = Path(root), rep, scope
        self.cache, self.verified = {}, {}

    def exists(self, uid) -> bool:
        return (self.root / uid / "COMPLETE.json").exists()

    def verify(self, uid) -> bool:
        if uid in self.verified:
            return self.verified[uid]
        d = self.root / uid
        ok, bad = True, []
        try:
            files = json.loads((d / "COMPLETE.json").read_text())["files"]
            for f, h in files.items():
                if not (d / f).exists() or sha_file(d / f) != h:
                    ok = False
                    bad.append(f)
        except Exception as e:  # noqa: BLE001
            ok, bad = False, [repr(e)]
        self.verified[uid] = ok
        if not ok:
            self.rep.add(self.scope, f"custody:{uid}", "FAIL", "unit files do not match COMPLETE.json", bad=bad[:5],
                         cause="unit modified or incomplete after COMPLETE.json was written")
        return ok

    def npz(self, uid, name="preds.npz"):
        k = (uid, name)
        if k not in self.cache:
            self.verify(uid)
            with np.load(self.root / uid / name, allow_pickle=False) as z:
                self.cache[k] = {kk: z[kk] for kk in z.files}
        return self.cache[k]

    def json(self, uid, name="record.json"):
        k = (uid, name)
        if k not in self.cache:
            self.verify(uid)
            self.cache[k] = json.loads((self.root / uid / name).read_text())
        return self.cache[k]


def read_csv(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def fnum(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x) if np.isfinite(x) else None
    x = str(x).strip()
    if x == "" or x.lower() in ("none", "nan", "null"):
        return None
    try:
        return float(x)
    except ValueError:
        return None


def pick(row, *names):
    for n in names:
        if n in row and row[n] not in (None, ""):
            return row[n]
    return None


# ================================================================================================ 1. roles
def build_roles(ds, inputs: Path, lock: dict):
    L = np.load(inputs / f"{ds}_labels.npz", allow_pickle=False)
    L = {k: L[k] for k in L.files}
    rr = lock["roles_rule"]
    split = L["split"].astype(str)
    ck = L["canon_key"].astype(str)
    role0 = L["role"].astype(str)
    rid, unit = L["row_id"].astype(np.int64), L["unit"]
    train_keys = set(ck[split == "train"].tolist())
    exposed = (split == "test") & np.array([k in train_keys for k in ck])
    scored0 = np.isin(role0, SCORED)
    excl = exposed & scored0
    csalt = rr["cert"]["salt"] + ds + "|"
    hsalt = rr["head_holdout"]["salt"] + ds + "|"
    cert_h = np.array([u01(csalt + k) < rr["cert"]["share"] for k in ck])
    head_h = np.array([u01(hsalt + k) < rr["head_holdout"]["share"] for k in ck])
    m = {}
    m["defense_fit"] = role0 == "defense_fit"
    afit = (role0 == "attacker_fit") & ~excl
    m["cert"] = afit & cert_h
    m["attacker_fit"] = afit & ~cert_h
    m["attacker_val"] = (role0 == "attacker_val") & ~excl
    m["assessment"] = (role0 == "assessment") & ~excl
    m["head_val"] = m["defense_fit"] & head_h
    m["head_fit"] = m["defense_fit"] & ~head_h
    idx = {r: np.flatnonzero(v) for r, v in m.items()}
    cv = rr["class_vocabulary"][ds]
    W = {"ds": ds, "row_id": rid, "unit": unit, "canon_key": ck, "split": split, "role0": role0, "idx": idx,
         "s": L[cv["attr"]].astype(int), "t": L[cv["task"]].astype(int), "exposed": exposed, "excl": excl,
         "labels": L, "cert_hash": cert_h}
    return W


def role_summary(W):
    out = {}
    for r, ix in W["idx"].items():
        out[r] = {"rows": int(len(ix)), "groups": int(len(np.unique(W["unit"][ix]))),
                  "row_ids_sha256": sha_arrays(np.sort(W["row_id"][ix]).astype(np.int64))}
    return out


def check_roles(rep, W, ras, lock, ds):
    sc = "roles"
    mine = role_summary(W)
    theirs = ras[ds]["roles"]
    for r in ("defense_fit", "cert", "attacker_fit", "attacker_val", "assessment", "head_fit", "head_val"):
        a, b = mine.get(r), theirs.get(r)
        ok = b is not None and a == b
        rep.check(sc, f"{ds}:{r}:rows_groups_hash", ok, cause="role membership differs from ROLES_AND_SUPPORT.json",
                  replay={"rows": a["rows"], "groups": a["groups"]},
                  reported={"rows": (b or {}).get("rows"), "groups": (b or {}).get("groups")},
                  hash_equal=bool(b is not None and a["row_ids_sha256"] == b["row_ids_sha256"]))
    n_excl = int(W["excl"].sum())
    n_excl_assess = int((W["excl"] & (W["role0"] == "assessment")).sum())
    by_role = {r: int((W["excl"] & (W["role0"] == r)).sum()) for r in SCORED}
    exp_total = EXPECT_EXPOSURE.get(ds, [None])[0]
    rep.check(sc, f"{ds}:exposure_rows_removed", exp_total is None or n_excl == exp_total,
              cause="exposure count differs from PROTOCOL.md (17 Adult / 42 HMDA)", replay=n_excl,
              protocol=exp_total, by_role=by_role)
    rep_ex = ras[ds]["roles"].get("excluded_exposure_rows")
    rep_gr = ras[ds].get("exposure_groups_removed")
    if rep_ex != n_excl or rep_gr != int(len(np.unique(W["unit"][W["excl"]]))):
        # diagnose: the runner stores roles in a fixed-width '<U16' array; 'excluded_exposure' has 17 characters
        a = np.array(["assessment"], dtype="<U16")
        a[0] = "excluded_exposure"
        trunc = bool(a[0] != "excluded_exposure")
        rep.add(sc, f"{ds}:excluded_exposure_rows_field", "FAIL",
                "ROLES_AND_SUPPORT.json reports excluded_exposure_rows / exposure_groups_removed that differ from the "
                "rows actually removed from scored roles",
                cause=("reporting-only defect: oar/study.py casts roles to '<U16' and assigns the 17-character label "
                       "'excluded_exposure', which numpy truncates to 'excluded_exposur'; the count of "
                       "role == 'excluded_exposure' is therefore 0. The rows ARE removed from every scored role "
                       "(the role-set counts and hashes above match). Truncation reproduced: %s" % trunc),
                reported={"excluded_exposure_rows": rep_ex, "exposure_groups_removed": rep_gr},
                replay={"excluded_exposure_rows": n_excl,
                        "exposure_groups_removed": int(len(np.unique(W["unit"][W["excl"]])))},
                assessment_rows_removed=n_excl_assess, true_removed_by_role=by_role,
                true_groups_removed_by_role={r: int(len(np.unique(W["unit"][W["excl"] & (W["role0"] == r)])))
                                             for r in SCORED})
    else:
        rep.add(sc, f"{ds}:excluded_exposure_rows_field", "PASS")
    # cert is a group-level carve-out: no unit may straddle cert / attacker_fit
    u_c, u_f = set(W["unit"][W["idx"]["cert"]].tolist()), set(W["unit"][W["idx"]["attacker_fit"]].tolist())
    rep.check(sc, f"{ds}:cert_group_consistent", not (u_c & u_f), cause="a record group is split between cert and "
              "attacker_fit", n_shared_units=len(u_c & u_f))
    # vocabulary and support
    cv = lock["roles_rule"]["class_vocabulary"][ds]
    K_s, K_t = cv["sensitive_K"], cv["task_K"]
    ok_vocab = (K_s == CELLS[ds]["K_s"] and K_t == CELLS[ds]["K_t"] and cv["attr"] == CELLS[ds]["attr"]
                and cv["task"] == CELLS[ds]["task"])
    scored_any = np.isin(W["role0"], SCORED) | (W["role0"] == "defense_fit")
    ok_range = bool(W["s"][scored_any].min() >= 0 and W["s"][scored_any].max() < K_s and
                    W["t"][scored_any].min() >= 0 and W["t"][scored_any].max() < K_t)
    rep.check(sc, f"{ds}:class_vocabulary", ok_vocab and ok_range, cause="class vocabulary differs from the lock",
              K_s=K_s, K_t=K_t)
    thr = lock["roles_rule"]["support"]["thresholds"]
    for what, y, K in (("sensitive", W["s"], K_s), ("task", W["t"], K_t)):
        counts = {r: np.bincount(y[W["idx"][r]], minlength=K).tolist() for r in thr}
        sup = [k for k in range(K) if all(counts[r][k] >= thr[r] for r in thr)]
        rs = ras[ds]["support"][what]
        locked = lock["fare"]["supported_classes"][ds] if what == "sensitive" else rs["supported_classes"]
        ok = sup == rs["supported_classes"] == locked and all(counts[r] == rs["counts"][r] for r in thr)
        rep.check(sc, f"{ds}:support_{what}", ok, cause="supported classes / counts differ (100/30/100 rule)",
                  replay_supported=sup, reported_supported=rs["supported_classes"], lock_supported=locked,
                  thresholds=thr)
    return mine


# ================================================================================================ 2./4. per-unit checks
ATTACK_KINDS = ("O_full", "O_prob", "O_hard", "O_head")


def list_units(run: Path):
    d = run / "units"
    return sorted(p.name for p in d.iterdir() if p.is_dir() and (p / "COMPLETE.json").exists()
                  and not p.name.startswith("_") and ".partial-" not in p.name)


def unit_kind(uid):
    rest = uid.split("__", 2)[2] if uid.count("__") >= 2 else ""
    if rest.startswith("FAREFIT_"):
        return "farefit"
    if rest.startswith("HEAD__"):
        return "head"
    if rest.startswith("U2__"):
        return "u2"
    if rest == "REF":
        return "ref"
    return "attack"


def check_units_rows(rep, US, W, uids, ds):
    """Every scored unit carries exactly the assessment rows (order, ids, units, labels) and, where it has val
    predictions, exactly the attacker_val rows; no assessment row reaches any val set."""
    sc = "leakage"
    a, v = W["idx"]["assessment"], W["idx"]["attacker_val"]
    rid_a, unit_a, rid_v = W["row_id"][a], W["unit"][a], W["row_id"][v]
    aset = set(rid_a.tolist())
    bad_a, bad_v, bad_y, leak, bad_role = [], [], [], [], []
    n = 0
    for u in uids:
        k = unit_kind(u)
        if k == "farefit":
            continue
        n += 1
        p = US.npz(u)
        if not (np.array_equal(p.get("assess_row_id"), rid_a) and np.array_equal(p.get("assess_unit"), unit_a)):
            bad_a.append(u)
        if "y_s" in p and not np.array_equal(p["y_s"], W["s"][a]):
            bad_y.append(u)
        if "y_t" in p and not np.array_equal(p["y_t"], W["t"][a]):
            bad_y.append(u)
        if k == "head" and not np.array_equal(p.get("row_id"), W["row_id"]):
            bad_a.append(u + "(row_id)")
        val = US.npz(u, "val_preds.npz")
        if "val_row_id" in val:
            if not np.array_equal(val["val_row_id"], rid_v):
                bad_v.append(u)
            if aset & set(val["val_row_id"].tolist()):
                leak.append(u)
            yv = val.get("val_y_s", val.get("val_y_t"))
            yt = W["s"][v] if "val_y_s" in val else W["t"][v]
            if yv is not None and not np.array_equal(yv, yt):
                bad_y.append(u + "(val)")
        if k == "attack":
            r = US.json(u)
            if r.get("selection_role") != "attacker_val" or r.get("fit_role") != "attacker_fit":
                bad_role.append(u)
    rep.check(sc, f"{ds}:assessment_rows_equal_roles", not bad_a, cause="unit assessment rows differ from oar-roles-v1",
              n_units=n, units=bad_a[:10])
    rep.check(sc, f"{ds}:val_rows_equal_attacker_val", not bad_v, cause="unit validation rows differ from attacker_val",
              units=bad_v[:10])
    rep.check(sc, f"{ds}:no_assessment_row_in_val", not leak, cause="assessment rows inside a selection/val set",
              units=leak[:10])
    rep.check(sc, f"{ds}:labels_match_inputs", not bad_y, cause="saved labels differ from labels.npz", units=bad_y[:10])
    rep.check(sc, f"{ds}:attack_records_fit_attacker_fit_select_attacker_val", not bad_role,
              cause="an attacker record names another fit/selection role", units=bad_role[:10])


def check_role_disjointness(rep, W, ds):
    sc = "leakage"
    idx, unit = W["idx"], W["unit"]
    base = ("defense_fit", "cert", "attacker_fit", "attacker_val", "assessment")
    bad_rows, bad_units = [], []
    for i, r1 in enumerate(base):
        for r2 in base[i + 1:]:
            if len(np.intersect1d(idx[r1], idx[r2])):
                bad_rows.append((r1, r2))
            if len(np.intersect1d(unit[idx[r1]], unit[idx[r2]])):
                bad_units.append((r1, r2))
    rep.check(sc, f"{ds}:roles_row_disjoint", not bad_rows, cause="a row belongs to two roles", pairs=bad_rows)
    rep.check(sc, f"{ds}:roles_group_disjoint", not bad_units, cause="a record group spans two roles", pairs=bad_units)
    hv, hf = idx["head_val"], idx["head_fit"]
    ok = (len(np.intersect1d(hv, hf)) == 0 and len(np.union1d(hv, hf)) == len(idx["defense_fit"]) and
          not len(np.intersect1d(hv, np.concatenate([idx[r] for r in base if r != "defense_fit"]))))
    rep.check(sc, f"{ds}:head_holdout_inside_defense_fit", ok, cause="head fit/holdout not a partition of defense_fit")
    ex = np.flatnonzero(W["excl"])
    scored_now = np.concatenate([idx[r] for r in ("cert", "attacker_fit", "attacker_val", "assessment")])
    rep.check(sc, f"{ds}:exposure_rows_in_no_scored_role", not len(np.intersect1d(ex, scored_now)),
              cause="an exposure row is still scored")


def check_farefit(rep, US, W, uids, ds, inputs):
    sc = "leakage"
    df = W["idx"]["defense_fit"]
    sh_lab = sha_arrays(W["t"][df].astype(np.int64), W["s"][df].astype(np.int64))
    bad = []
    nfit = 0
    Hs = {}
    for u in uids:
        if unit_kind(u) != "farefit":
            continue
        k = int(u.split("__")[1][1:])
        if k not in Hs:
            F = np.load(inputs / f"{ds}_s{k}_forward.npz", allow_pickle=False)
            if not np.array_equal(F["row_id"], W["row_id"]):
                bad.append((u, "forward row order differs from labels"))
            Hs[k] = sha_arrays(np.asarray(F[CELLS[ds]["rep"]], np.float64)[df])
        r = US.json(u, "rec.json")
        nfit += 1
        prob = []
        if r.get("n_fit") != len(df):
            prob.append("n_fit")
        if r.get("fit_rows_sha256") != Hs[k]:
            prob.append("fit_rows_sha256")
        if r.get("fit_labels_sha256") != sh_lab:
            prob.append("fit_labels_sha256")
        if r.get("n_all") != len(W["row_id"]):
            prob.append("n_all")
        cells = np.load(US.root / u / "cells.npy", allow_pickle=False)
        if cells.shape != (len(W["row_id"]),) or cells.min() < 0 or cells.max() >= int(r.get("n_cells", 0)):
            prob.append("cells.npy range/shape")
        if prob:
            bad.append((u, prob))
    rep.check(sc, f"{ds}:fare_fit_rows_are_defense_fit", not bad and nfit > 0,
              cause="a FARE fit used rows/labels other than defense_fit", n_farefit_units=nfit, problems=bad[:10])


# ================================================================================================ nominee
def recompute_nominee(rep, US, W, ds, k, lock, sel):
    """Frozen validation-only rule from val_preds only."""
    sc = "nominee"
    P = f"{ds}__s{k}"
    grid = lock["fare"]["grid"]
    classes = lock["fare"]["supported_classes"][ds]
    # aliases: identical cells to an earlier (non-alias) configuration
    cells, alias, seen = {}, {}, []
    for g in grid:
        i = g["id"]
        u = f"{P}__FAREFIT_c{i}"
        if not US.exists(u):
            rep.add(sc, f"{P}:farefit_c{i}_present", "FAIL", cause="FARE grid unit missing")
            return None
        US.verify(u)
        c = np.load(US.root / u / "cells.npy", allow_pickle=False)
        cells[i] = c
        same = [j for j, cj in seen if np.array_equal(cj, c)]
        if same:
            alias[i] = same[0]
        else:
            seen.append((i, c))
    rep_alias = {int(a): int(b) for a, b in (sel.get("aliases") or {}).items()}
    rep.check(sc, f"{P}:fare_aliases", rep_alias == alias, cause="alias map differs from identical-cells rule",
              replay=alias, reported=rep_alias)
    # aliased configs must not have been scored separately
    dup = [i for i in alias if US.exists(f"{P}__Fc{i}__rep") or US.exists(f"{P}__U2__Fc{i}")]
    rep.check(sc, f"{P}:aliases_not_refitted", not dup, cause="an alias configuration was scored again", configs=dup)

    def val_acc(uid):
        v = US.npz(uid, "val_preds.npz")
        return float(np.mean(v["VAL__U2_P"].argmax(1) == v["val_y_t"]))

    def val_auc(uid):
        v = US.npz(uid, "val_preds.npz")
        y, Pv = v["val_y_s"].astype(int), v["VAL__NL__as0"]
        return float(np.mean([midrank_auc(Pv[:, c], y == c) for c in classes]))

    accA = val_acc(f"{P}__U2__A")
    rows = []
    for g in grid:
        i = g["id"]
        src = alias.get(i, i)
        acc = val_acc(f"{P}__U2__Fc{src}")
        auc = val_auc(f"{P}__Fc{src}__rep")
        rows.append({"config": i, "alias_of": alias.get(i), "val_u2_accuracy": acc, "val_nl_macro_auc": auc,
                     "admissible": bool(acc >= accA - 0.01), "margin_to_cap": acc - (accA - 0.01)})
    adm = [r for r in rows if r["admissible"]]
    if adm:
        best = min(adm, key=lambda r: (r["val_nl_macro_auc"], r["config"]))
        ties = [r["config"] for r in adm if abs(r["val_nl_macro_auc"] - best["val_nl_macro_auc"]) <= 1e-12]
        nom, admissible = min(ties), True
    else:
        fb = max(rows, key=lambda r: (r["val_u2_accuracy"], -r["config"]))
        nom, admissible = fb["config"], False
    out = {"nominee": nom, "nominee_unit_source": alias.get(nom, nom), "admissible": admissible,
           "untreated_val_u2_accuracy": accA, "table": rows, "aliases": alias,
           "borderline_admissibility": [r["config"] for r in rows if abs(r["margin_to_cap"]) < 1e-9]}
    # compare with the runner's selection file
    ok = (sel.get("nominee") == nom and sel.get("nominee_unit_source") == alias.get(nom, nom)
          and bool(sel.get("admissible")) == admissible)
    rep.check(sc, f"{P}:nominee", ok, cause="validation-only nominee differs from selection file",
              replay={"nominee": nom, "source": alias.get(nom, nom), "admissible": admissible},
              reported={"nominee": sel.get("nominee"), "source": sel.get("nominee_unit_source"),
                        "admissible": sel.get("admissible")})
    tbl = {r["config"]: r for r in sel.get("table", [])}
    dif = []
    for r in rows:
        t = tbl.get(r["config"])
        if (t is None or abs(t["val_u2_accuracy"] - r["val_u2_accuracy"]) > 1e-12 or
                abs(t["val_nl_macro_auc"] - r["val_nl_macro_auc"]) > 1e-9 or bool(t["admissible"]) != r["admissible"]
                or abs(sel.get("untreated_val_u2_accuracy", np.nan) - accA) > 1e-12):
            dif.append(r["config"])
    rep.check(sc, f"{P}:selection_table_values", not dif, cause="selection-table validation values differ",
              configs=dif)
    if out["borderline_admissibility"]:
        rep.add(sc, f"{P}:admissibility_borderline", "FLAG", "a configuration sits within 1e-9 of the cap",
                configs=out["borderline_admissibility"])
    return out


# ================================================================================================ 3. primary
def resolve_plus(US, uid, rep, P, seen=None):
    """Plus-surface rule: a validation-selected ignore-candidate means the unit's recovery IS the aliased unit's."""
    r = US.json(uid)
    ps = r.get("plus_selection")
    if not ps or not ps.get("alias_source"):
        return uid
    src = ps["alias_source"]
    seen = set() if seen is None else seen
    if src in seen:
        raise ValueError(f"alias cycle at {uid}")
    seen.add(uid)
    return resolve_plus(US, src, rep, P, seen)


def check_plus_record(rep, US, uid, rep_unit, out_unit):
    r = US.json(uid)
    ps = r.get("plus_selection") or {}
    cand = ps.get("candidates_attacker_val_log_loss") or {}
    probs = []
    try:
        if abs(cand["GBT/MLP"] - r["val_log_loss"]["NL"]) > 0:
            probs.append("own candidate != record val_log_loss.NL")
        if abs(cand["ignore_rep"] - US.json(out_unit)["val_log_loss"]["NL"]) > 0:
            probs.append("ignore_rep candidate != outputs unit val loss")
        if abs(cand["ignore_out"] - US.json(rep_unit)["val_log_loss"]["NL"]) > 0:
            probs.append("ignore_out candidate != rep unit val loss")
        best = None
        for name in ("GBT/MLP", "ignore_rep", "ignore_out"):
            ll = cand.get(name)
            if ll is None or not np.isfinite(ll):
                continue
            if best is None or ll < cand[best]:
                best = name
        if ps.get("selected") != best:
            probs.append(f"selected {ps.get('selected')} != argmin {best}")
        exp_src = {"ignore_rep": out_unit, "ignore_out": rep_unit}.get(best)
        if ps.get("alias_source") != exp_src:
            probs.append(f"alias_source {ps.get('alias_source')} != {exp_src}")
    except Exception as e:  # noqa: BLE001
        probs.append(repr(e))
    return probs


def read_family(pkg: Path, lock: dict, rep: Report):
    rows = read_csv(pkg / "PRIMARY_FAMILY.csv")
    fam = [(r["id"], r["cell"], r["question"], r["delta"], float(r["margin"]), r["kind"]) for r in rows]
    lrows = [tuple(x) for x in lock["primary_family"]["rows"]]
    ok = (len(fam) == 12 == lock["primary_family"]["size"] == len(lrows) and
          [(a, b, c, d, float(e), f) for a, b, c, d, e, f in lrows] == fam and len({f[0] for f in fam}) == 12)
    rep.check("primary", "family_size_and_rows", ok, cause="PRIMARY_FAMILY.csv / lock family is not the frozen 12 rows",
              n_csv=len(fam), n_lock=len(lrows), size_lock=lock["primary_family"]["size"])
    return fam


def endpoint_sides(fid, ds, P_of, nominee_src, US, rep):
    """Return for one family row: (sideA, sideB, kind) where each side is {seed: [uids]} plus metric type."""
    q = fid.split("-")[0]

    def per_seed(f):
        return {k: f(k) for k in SEEDS}
    if q == "P1":
        return ("R", per_seed(lambda k: f"{P_of(k)}__O_full"), per_seed(lambda k: f"{P_of(k)}__O_hard"))
    if q == "P2":
        return ("R", per_seed(lambda k: f"{P_of(k)}__B__rep"),
                per_seed(lambda k: f"{P_of(k)}__Fc{nominee_src[k]}__rep" if nominee_src.get(k) else None))
    if q == "P3":
        return ("Acc", per_seed(lambda k: f"{P_of(k)}__U2__Fc{nominee_src[k]}" if nominee_src.get(k) else None),
                per_seed(lambda k: f"{P_of(k)}__U2__A"))
    if q == "P4":
        return ("Acc", per_seed(lambda k: f"{P_of(k)}__U2__Fc{nominee_src[k]}" if nominee_src.get(k) else None),
                per_seed(lambda k: f"{P_of(k)}__U2__B"))
    if q == "P5":
        return ("R", per_seed(lambda k: f"{P_of(k)}__F__rep+clean"), per_seed(lambda k: f"{P_of(k)}__F__rep+head"))
    if q == "P6":
        return ("R", per_seed(lambda k: f"{P_of(k)}__B__rep+head"), per_seed(lambda k: f"{P_of(k)}__F__rep+head"))
    raise ValueError(fid)


PLUS_COMPONENTS = {  # plus unit suffix -> (rep component suffix, outputs component suffix); F resolved per seed
    "A__rep+clean": ("A__rep", "O_full"), "B__rep+clean": ("B__rep", "O_full"), "C__rep+clean": ("C__rep", "O_full"),
    "A__rep+head": ("A__rep", "O_headA"), "B__rep+head": ("B__rep", "O_headB"),
    "FZ__rep+clean": ("FZ__rep", "O_full"), "FZ__rep+head": ("FZ__rep", "O_headFZ"),
}


def primary_dataset(rep, US, W, ds, lock, fam, nsrc, B, seed, agg, tag="", membership=True):
    """Replay the six family rows of one cell for a given per-seed FARE unit source (nominee)."""
    sc = "primary"
    alpha = lock["primary_family"]["alpha_family"] / len(fam)
    rows = [f for f in fam if f[1] == ds]
    classes = lock["fare"]["supported_classes"][ds]
    eng = Engine()
    P_of = lambda k: f"{ds}__s{k}"  # noqa: E731
    plans = {}
    a = W["idx"]["assessment"]
    for f in rows:
        fid, margin = f[0], f[4]
        kind, sideA, sideB = endpoint_sides(fid, ds, P_of, nsrc, US, rep)
        plan = {"kind": kind, "margin": margin, "sides": [], "members": [], "missing": []}
        for side in (sideA, sideB):
            parts, mem = [], {}
            for k in SEEDS:
                uid = side.get(k)
                if uid is None or not US.exists(uid):
                    plan["missing"].append(uid or f"{P_of(k)}:<no nominee>")
                    continue
                if kind == "R":
                    ru = resolve_plus(US, uid, rep, P_of(k))
                    p = US.npz(ru)
                    sub = hier_mean([auc_expr(eng, p["y_s"], p[f"P__NL__as{s}"], classes) for s in ATT])
                    mem[k] = {"unit": uid, "scored_as": ru}
                else:
                    p = US.npz(uid)
                    sub = acc_expr(eng, p["y_t"], p["U2_P"])
                    mem[k] = {"unit": uid}
                parts.append(sub)
            plan["members"].append(mem)
            plan["sides"].append(hier_mean(parts) if len(parts) == len(SEEDS) else None)
        if membership:
            seedsA, seedsB = set(plan["members"][0]), set(plan["members"][1])
            rep.check("membership", f"{fid}:encoder_seeds_paired", seedsA == seedsB == set(SEEDS),
                      cause="sides of a paired comparison do not use the same full encoder-seed set",
                      seeds_side1=sorted(seedsA), seeds_side2=sorted(seedsB), missing=plan["missing"])
        plans[fid] = plan
    pts = eng.points()
    if membership:
        ptw = eng.points_weighted_ones(len(a))
        maxdiff = max([abs(pts[k] - ptw[k]) for k in pts if np.isfinite(pts[k])] or [0.0])
        rep.check(sc, f"{ds}:auc_point_two_algorithms", maxdiff < 1e-12,
                  cause="midrank AUC and weighted AUC at unit weights disagree", max_abs_diff=maxdiff)
    assess_unit = W["unit"][a]
    t0 = time.process_time()
    reps = eng.replicates(assess_unit, B, seed) if eng.bases else {}
    agg.setdefault("timing", {})[f"primary_bootstrap_{ds}{tag}_cpu_s"] = time.process_time() - t0
    agg.setdefault("primary_bootstrap", {})[ds + tag] = {"n_bases": len(eng.bases),
                                                         "n_units": int(len(np.unique(assess_unit))),
                                                         "n_rows": int(len(a)), "B": B, "seed": seed}
    results = {}
    for fid, plan in plans.items():
        res = {"id": fid, "cell": ds, "margin": plan["margin"], "alpha_each": alpha, "B": B, "seed": seed,
               "members": plan["members"]}
        if plan["sides"][0] is None or plan["sides"][1] is None:
            res.update(status="UNRESOLVED", decision="UNRESOLVED", reason="missing units", missing=plan["missing"])
            results[fid] = res
            continue
        s1p, s2p = combine(plan["sides"][0], pts), combine(plan["sides"][1], pts)
        r = combine(plan["sides"][0], reps) - combine(plan["sides"][1], reps)
        lo, hi, n_ne = bound_pair(r, alpha, 1 - alpha, alpha)
        dec = "UNRESOLVED" if lo is None else ("PASS" if lo > plan["margin"] else "NOT_ESTABLISHED")
        res.update(point=s1p - s2p, side1_point=s1p, side2_point=s2p, lower=lo, upper=hi, n_ne_replicates=n_ne,
                   mc_se_lower=mc_se(r, alpha), mc_se_upper=mc_se(r, 1 - alpha), decision=dec,
                   status="DECIDED" if lo is not None else "UNRESOLVED",
                   n_replicates_above_margin=int((r > plan["margin"]).sum()))
        results[fid] = res
    return results


def run_primary(rep, US, roles, lock, fam, nominees, runner_rows, B, seed, agg):
    sc = "primary"
    alpha = lock["primary_family"]["alpha_family"] / len(fam)
    if abs(alpha - lock["primary_family"]["alpha_each"]) > 1e-15:
        rep.add(sc, "alpha_each", "FAIL", cause="alpha_each differs from 0.05/12")
    results, alt = {}, {}
    for ds in ("adult", "hmda"):
        if ds not in roles:
            for f in fam:
                if f[1] == ds:
                    results[f[0]] = {"id": f[0], "status": "UNRESOLVED", "decision": "UNRESOLVED",
                                     "reason": "dataset not replayed"}
            continue
        mine = {k: (nominees.get((ds, k)) or {}).get("nominee_unit_source") for k in SEEDS}
        theirs = {k: (nominees.get((ds, k)) or {}).get("runner_nominee_unit_source") for k in SEEDS}
        results.update(primary_dataset(rep, US, roles[ds], ds, lock, fam, mine, B, seed, agg))
        if mine != theirs:
            # keep both values until the cause is resolved
            alt.update(primary_dataset(rep, US, roles[ds], ds, lock, fam, theirs, B, seed, agg,
                                       tag="_runner_nominee", membership=False))
    agg["primary_endpoints"] = [results.get(f[0], {"id": f[0], "status": "UNRESOLVED"}) for f in fam]
    if alt:
        agg["primary_endpoints_with_runner_nominee"] = list(alt.values())
    compare_primary(rep, fam, results, runner_rows, B)
    for ds in ("adult", "hmda"):
        conj = lock["primary_family"]["competitive_conjunction"][ds]
        allpass = all(results.get(i, {}).get("decision") == "PASS" for i in conj)
        adm = all((nominees.get((ds, k)) or {}).get("admissible") for k in SEEDS)
        agg.setdefault("competitive", {})[ds] = {"conjunction_rows": conj, "all_pass": allpass,
                                                 "every_seed_admissible_nominee": bool(adm),
                                                 "competitive_method": bool(allpass and adm)}
    return results


def compare_primary(rep, fam, results, runner_rows, B):
    sc = "primary"
    if runner_rows is None:
        for f in fam:
            rep.add(sc, f"{f[0]}:vs_runner", "UNRESOLVED", "runner PRIMARY_ENDPOINTS.csv not found",
                    cause="runner table not yet written")
        return
    ids = [pick(r, "id", "endpoint", "row_id") for r in runner_rows]
    ok_fs = len(runner_rows) == 12 and sorted(ids) == sorted(f[0] for f in fam)
    rep.check(sc, "runner_family_size_12", ok_fs, cause="runner primary table does not have exactly the 12 family rows",
              n_rows=len(runner_rows), extra=sorted(set(ids) - {f[0] for f in fam}),
              missing=sorted({f[0] for f in fam} - set(ids)))
    byid = {}
    for r in runner_rows:
        byid.setdefault(pick(r, "id", "endpoint", "row_id"), r)
    for f in fam:
        fid = f[0]
        mine = results.get(fid, {})
        r = byid.get(fid)
        if r is None:
            rep.add(sc, f"{fid}:vs_runner", "FAIL", cause="row missing from runner table", replay=mine.get("decision"))
            continue
        rp = fnum(pick(r, "point", "delta_point", "estimate"))
        rl = fnum(pick(r, "lower", "lcb", "lower_bound", "LCB", "simultaneous_lower"))
        ru = fnum(pick(r, "upper", "ucb", "upper_bound", "UCB", "simultaneous_upper"))
        rd = (pick(r, "decision", "verdict") or "").strip()
        md = mine.get("decision")
        vals = {"replay": {"point": mine.get("point"), "lower": mine.get("lower"), "upper": mine.get("upper"),
                           "decision": md},
                "runner": {"point": rp, "lower": rl, "upper": ru, "decision": rd}}
        if mine.get("status") == "UNRESOLVED" or mine.get("point") is None:
            st = "PASS" if rd.upper().startswith("UNRESOLVED") else "FAIL"
            rep.add(sc, f"{fid}:vs_runner", st, "replay could not compute this row", None if st == "PASS" else
                    "runner reports a value for a row whose inputs are missing or unpaired in the replay", **vals)
            continue
        dp = None if rp is None else abs(rp - mine["point"])
        tol_l = 6.5 * mine["mc_se_lower"] + 5e-5
        tol_u = 6.5 * mine["mc_se_upper"] + 5e-5
        dl = None if (rl is None or mine["lower"] is None) else abs(rl - mine["lower"])
        du = None if (ru is None or mine["upper"] is None) else abs(ru - mine["upper"])
        rep.check(sc, f"{fid}:point", dp is not None and dp <= 1e-9, cause="deterministic point disagreement",
                  abs_diff=dp, **vals)
        bst = "PASS"
        if dl is None or du is None:
            bst = "FAIL"
        elif dl > tol_l or du > tol_u:
            bst = "FAIL"
        rep.add(sc, f"{fid}:bounds", bst, None, None if bst == "PASS" else "bound outside 6.5 MC-SE + 5e-5",
                abs_diff_lower=dl, abs_diff_upper=du, tol_lower=tol_l, tol_upper=tol_u, exact_match=(dl == 0 and du == 0))
        same = (rd.upper() == md) or (md == "NOT_ESTABLISHED" and rd.upper() in ("NOT_ESTABLISHED", "NOT ESTABLISHED"))
        near = mine["lower"] is not None and abs(mine["lower"] - mine["margin"]) <= tol_l
        if same and not near:
            rep.add(sc, f"{fid}:decision", "PASS", replay=md, runner=rd)
        elif near:
            rep.add(sc, f"{fid}:decision", "MC_BORDERLINE", "lower bound within the MC tolerance band of the margin",
                    cause=("Monte Carlo resolution: the bound is within 6.5 MC-SE + 5e-5 of the margin; " +
                           ("the decision is reproduced under the mirrored draw, but an independent draw could flip it"
                            if same else "the decisions differ inside the Monte Carlo band")),
                    replay=md, runner=rd, lower=mine["lower"], margin=mine["margin"], tol=tol_l, decision_agrees=same)
        else:
            rep.add(sc, f"{fid}:decision", "FAIL", cause="decision differs and the bound is not near the margin",
                    replay=md, runner=rd, lower=mine["lower"], margin=mine["margin"])


# ================================================================================================ 4. membership
def check_membership(rep, US, W, ds, lock, uids, agg, inputs):
    sc = "membership"
    sig = lock["noise"]["sigma_star"][ds]
    rs_lock = list(lock["noise"]["release_seeds"])
    bad_noise, plus_alias = [], Counter()
    for k in SEEDS:
        P = f"{ds}__s{k}"
        for rs in rs_lock:
            u = f"{P}__D_rs{rs}__rep"
            if not US.exists(u):
                bad_noise.append((u, "missing"))
                continue
            c = (US.json(u).get("contract") or {}).get("features", "")
            if c != f"noise sigma={sig} rs={rs}":
                bad_noise.append((u, c))
            for v in ("full", "prob", "hard"):
                if not US.exists(f"{P}__D_rs{rs}__rep+{v}"):
                    bad_noise.append((f"{P}__D_rs{rs}__rep+{v}", "missing"))
            if not US.exists(f"{P}__U2__D_rs{rs}"):
                bad_noise.append((f"{P}__U2__D_rs{rs}", "missing"))
        extra = [u for u in uids if u.startswith(f"{P}__D_rs") and int(u.split("__")[2][4:]) not in rs_lock]
        bad_noise += [(u, "unexpected release seed") for u in extra]
    rep.check(sc, f"{ds}:noise_sigma_and_release_seeds", not bad_noise,
              cause="noise units missing, with another sigma, or other release seeds", sigma_star=sig,
              release_seeds=rs_lock, problems=bad_noise[:10])
    # plus-surface records: candidates and alias consistency (validation-only selection)
    bad_plus = []
    for u in uids:
        if unit_kind(u) != "attack":
            continue
        r = US.json(u)
        if not r.get("plus_selection"):
            continue
        P, rest = u.split("__", 2)[0] + "__" + u.split("__", 2)[1], u.split("__", 2)[2]
        if rest in PLUS_COMPONENTS:
            rp, op = PLUS_COMPONENTS[rest]
            comp = (f"{P}__{rp}", f"{P}__{op}")
        elif rest.startswith("F__"):
            nm = agg.get("nominee", {}).get(P, {})
            src = nm.get("runner_nominee_unit_source")
            comp = (f"{P}__Fc{src}__rep", f"{P}__O_full" if rest.endswith("clean") else f"{P}__O_headF")
        elif rest.startswith("D_rs"):
            q, v = rest.split("__")
            comp = (f"{P}__{q}__rep", f"{P}__O_{v.split('+')[1]}")
        else:
            bad_plus.append((u, "unknown plus layout"))
            continue
        pr = check_plus_record(rep, US, u, *comp)
        if pr:
            bad_plus.append((u, pr))
        plus_alias[(r["plus_selection"].get("selected") or "none")] += 1
    rep.check(sc, f"{ds}:plus_surface_selection_and_alias", not bad_plus,
              cause="plus-surface selection or alias_source inconsistent with validation losses",
              counts=dict(plus_alias), problems=bad_plus[:10])
    # output alias report: recompute from the historical logits
    bad_al = []
    for k in SEEDS:
        p = US.root.parent / "aliases" / f"{ds}__s{k}.json"
        if not p.exists():
            bad_al.append((k, "missing"))
            continue
        A = json.loads(p.read_text())
        F = np.load(inputs / f"{ds}_s{k}_forward.npz", allow_pickle=False)
        O = np.asarray(F[CELLS[ds]["logits"]], np.float64)
        Z = O - O.max(1, keepdims=True)
        Pr = np.exp(Z) / np.exp(Z).sum(1, keepdims=True)
        sd = float(np.std(O.sum(1)))
        lp = np.log(np.clip(Pr, 1e-300, 1))
        rec_err = float(np.max(np.abs((lp - lp.mean(1, keepdims=True)) - (O - O.mean(1, keepdims=True)))))
        if (A.get("K") != O.shape[1] or bool(A.get("logit_sum_is_constant")) != (sd < 1e-9) or
                abs(A.get("logit_sum_sd", np.nan) - sd) > 1e-9 * max(1, sd) or
                abs(A.get("prob_determines_centred_logits_maxabs", np.nan) - rec_err) > 1e-9):
            bad_al.append((k, {"reported_sd": A.get("logit_sum_sd"), "replay_sd": sd}))
        agg.setdefault("output_alias", {})[f"{ds}__s{k}"] = {"logit_sum_sd": sd, "constant": sd < 1e-9,
                                                             "full_vs_prob_alias": sd < 1e-9}
    rep.check(sc, f"{ds}:output_alias_reports", not bad_al, cause="alias report differs from the logits",
              problems=bad_al)
    # duplicates: attack units whose NL predictions are byte-identical
    hs = defaultdict(list)
    for u in uids:
        if unit_kind(u) == "attack":
            p = US.npz(u)
            hs[sha_arrays(*[p[f"P__NL__as{s}"] for s in ATT])].append(u)
    dups = [v for v in hs.values() if len(v) > 1]
    agg.setdefault("duplicates", {})[ds] = {"n_attack_units": sum(len(v) for v in hs.values()),
                                            "n_identical_prediction_groups": len(dups),
                                            "groups": dups[:20]}
    n_alias = sum(len(agg.get("nominee", {}).get(f"{ds}__s{k}", {}).get("aliases", {})) for k in SEEDS)
    agg.setdefault("alias_counts", {})[ds] = {"fare_grid_aliases": n_alias, "plus_surface_selected": dict(plus_alias)}
    rep.add(sc, f"{ds}:alias_duplicate_counts", "PASS", "recorded in aggregate (fare aliases recomputed from cells)",
            fare_grid_aliases=n_alias, identical_prediction_groups=len(dups))


# ================================================================================================ 5. contracts
def leace_numpy(H, mapdir: Path):
    z = np.load(mapdir / "leace_map.npz", allow_pickle=False)
    return H - ((H - z["mean_x"]) @ z["proj_right"].T) @ z["proj_left"].T


def head_features(ds, k, tag, US, inputs, bench, nominee_src):
    F = np.load(inputs / f"{ds}_s{k}_forward.npz", allow_pickle=False)
    H = np.asarray(F[CELLS[ds]["rep"]], np.float64)
    P = f"{ds}__s{k}"
    if tag == "A":
        return H
    if tag == "B":
        c = CELLS[ds]
        return leace_numpy(H, bench / "defenses" / f"{ds}__s{k}__{c['purpose']}__B_{c['attr']}" / "map")
    if tag in ("F", "FZ"):
        u = f"{P}__FAREFIT_Z" if tag == "FZ" else f"{P}__FAREFIT_c{nominee_src}"
        cells = np.load(US.root / u / "cells.npy", allow_pickle=False).astype(int)
        X = np.zeros((len(cells), int(cells.max()) + 1))
        X[np.arange(len(cells)), cells] = 1.0
        return X
    raise ValueError(tag)


def check_heads(rep, US, W, ds, inputs, bench, nominees, agg, tol=1e-9):
    sc = "contracts"
    try:
        import joblib  # noqa: F401  (loading only; not a scoring path)
    except Exception as e:  # noqa: BLE001
        rep.add(sc, f"{ds}:heads", "NOT_CHECKED", cause=f"joblib unavailable: {e!r}")
        return
    import joblib
    a = W["idx"]["assessment"]
    for k in SEEDS:
        P = f"{ds}__s{k}"
        nsrc = (nominees.get((ds, k)) or {}).get("runner_nominee_unit_source")
        for tag in ("A", "B", "F", "FZ"):
            u = f"{P}__HEAD__{tag}"
            if not US.exists(u):
                rep.add(sc, f"{u}:deterministic_on_features", "UNRESOLVED", cause="head unit missing")
                continue
            r = US.json(u)
            p = US.npz(u)
            prob = []
            if r.get("inputs_at_runtime") != "protected features only":
                prob.append("inputs_at_runtime")
            if r.get("fit_role") != "defense_fit minus oar-head-v1 holdout" or "oar-head-v1" not in str(
                    r.get("select_role")):
                prob.append("fit/select role")
            try:
                X = head_features(ds, k, tag, US, inputs, bench, nsrc)
                m = joblib.load(US.root / u / "models" / "head.joblib")
                nin = getattr(m, "n_features_in_", None)
                if nin is not None and nin != X.shape[1]:
                    prob.append(f"head expects {nin} inputs, protected features have {X.shape[1]}")
                Pm = m.predict_proba(X)
                K_t = CELLS[ds]["K_t"]
                full = np.zeros((X.shape[0], K_t))
                full[:, np.asarray(m.classes_).astype(int)] = Pm
                Oh = np.log(np.clip(full, CLIP, 1.0))
                d = float(np.max(np.abs(Oh - p["head_outputs_all"])))
                if not d <= tol:
                    prob.append(f"recomputed outputs differ by {d:.3g}")
                Pa = np.exp(p["head_outputs_all"][a])
                acc = float(np.mean(Pa.argmax(1) == W["t"][a]))
                if abs(acc - r.get("assessment_accuracy", np.nan)) > 1e-12:
                    prob.append("assessment_accuracy")
                agg.setdefault("heads", {})[u] = {"max_abs_diff_recomputed": d, "n_inputs": int(X.shape[1]),
                                                  "assessment_accuracy": acc,
                                                  "assessment_log_loss": r.get("assessment_log_loss")}
            except Exception as e:  # noqa: BLE001
                prob.append(repr(e)[:300])
            rep.check(sc, f"{u}:deterministic_on_features", not prob,
                      cause="view-3 head is not reproduced from the protected features alone", problems=prob)


def cert_block_problems(cert, who, kind, US, fu, W, ccfg, sec):
    """A certificate is reported only with its premises: OK needs a bound, premises, every checked premise true and
    every pair OK; UNAVAILABLE needs a reason and no bound. The every-cell-present premise is recomputed."""
    prob = []
    st = cert.get("status")
    if st == "OK":
        prem = cert.get("premises") or []
        if cert.get("bound") is None or not prem:
            prob.append(f"{who}/{kind}: OK without bound/premises")
        if any(pp.get("how") == "checked" and pp.get("holds") is not True for pp in prem):
            prob.append(f"{who}/{kind}: OK but a checked premise fails")
        if any(pq.get("status") != "OK" for pq in cert.get("pairs", [])):
            prob.append(f"{who}/{kind}: OK with a non-OK pair")
    elif st == "UNAVAILABLE":
        if cert.get("bound") is not None or not cert.get("reason"):
            prob.append(f"{who}/{kind}: UNAVAILABLE with a bound or without a reason")
    else:
        prob.append(f"{who}/{kind}: status {st}")
    if st in ("OK", "UNAVAILABLE") and cert.get("pairs"):
        try:
            cr = W["idx"]["cert"]
            cells_all = np.load(US.root / fu / "cells.npy", allow_pickle=False).astype(int)
            model = json.loads((US.root / fu / "model" / "model.json").read_text())
            gc = np.asarray(model["cell_group_counts"])
            gcodes = [int(g) for g in model["group_codes"]]
            kcells = gc.shape[0]
            s = W["s"][cr]
            grp = sorted(int(g) for g in (kind == "secondary_groups" and sec or gcodes))
            keep = np.isin(s, grp)
            cc, ss = cells_all[cr][keep], s[keep]
            n = len(ss)
            perm = np.random.RandomState(int(ccfg["split_seed"])).permutation(n)
            nv = int(round(float(ccfg["val_fraction"]) * n))
            va, te = perm[:nv], perm[nv:]
            for pq in cert["pairs"]:
                gi, gj = pq["groups"]
                if gi not in gcodes or gj not in gcodes:
                    continue
                miss = {"base": int(kcells - np.count_nonzero(gc[:, gcodes.index(gi)] + gc[:, gcodes.index(gj)]))}
                for nm, ix in (("val", va), ("test", te)):
                    mm = (ss[ix] == gi) | (ss[ix] == gj)
                    miss[nm] = int(kcells - len(np.unique(cc[ix][mm])))
                if pq.get("cells_missing") is not None and pq["cells_missing"] != miss:
                    prob.append(f"{who}/{kind} pair {gi}-{gj}: cells_missing {pq['cells_missing']} != {miss}")
                if any(miss.values()) and pq.get("status") == "OK":
                    prob.append(f"{who}/{kind} pair {gi}-{gj}: OK although a cell is missing")
        except Exception as e:  # noqa: BLE001
            prob.append(f"{who}/{kind}: premise recompute error {e!r}"[:300])
    return prob


def check_certificates(rep, US, W, ds, lock, nominees, agg, inputs=None):
    sc = "contracts"
    cr = W["idx"]["cert"]
    ccfg = lock["fare"]["certificate"]
    sec = (lock["fare"].get("secondary_certificate_groups") or {}).get(ds)
    amend_p = US.root.parent / "certificates" / "AMENDMENT_A1.json"
    amend = json.loads(amend_p.read_text()) if amend_p.exists() else None
    for k in SEEDS:
        P = f"{ds}__s{k}"
        p = US.root.parent / "certificates" / f"{P}.json"
        if not p.exists():
            rep.add(sc, f"{P}:certificates", "UNRESOLVED", cause="certificate file missing")
            continue
        C = json.loads(p.read_text())
        prob = []
        if C.get("cert_rows") != len(cr):
            prob.append(f"cert_rows {C.get('cert_rows')} != {len(cr)}")
        nj = (nominees.get((ds, k)) or {}).get("runner_nominee")
        nsrc = (nominees.get((ds, k)) or {}).get("runner_nominee_unit_source")
        want = ["primary_all_groups"] + (["secondary_groups"] if sec else [])
        for who, fu in (("nominee", f"{P}__FAREFIT_c{nj}"), ("zero_fairness", f"{P}__FAREFIT_Z")):
            block = C.get(who) or {}
            for kind in want:
                cert = block.get(kind)
                if cert is None:
                    prob.append(f"{who}/{kind} missing")
                    continue
                prob += cert_block_problems(cert, who, kind, US, fu, W, ccfg, sec)
                agg.setdefault("certificates", {}).setdefault(P, {})[f"original:{who}/{kind}"] = {
                    "status": cert.get("status"), "bound": cert.get("bound"), "reason": cert.get("reason")}
        rep.check(sc, f"{P}:certificates_with_premises", not prob,
                  cause="certificate reported without its premises or despite a failed premise", problems=prob[:10])
        if amend is None or P not in amend:
            continue
        # dated amendment A1: row-identity guard instead of the feature-hash guard; originals kept beside
        A = amend[P]
        prob = []
        if A.get("original") != C:
            prob.append("amendment does not carry the original certificate record unchanged")
        if len(np.intersect1d(W["row_id"][cr], W["row_id"][W["idx"]["defense_fit"]])) or \
                len(np.intersect1d(W["unit"][cr], W["unit"][W["idx"]["defense_fit"]])):
            prob.append("cert rows / groups intersect the FARE fit rows (row-identity guard fails)")
        ndup = None
        if inputs is not None:
            F = np.load(inputs / f"{ds}_s{k}_forward.npz", allow_pickle=False)
            H = np.ascontiguousarray(np.asarray(F[CELLS[ds]["rep"]], np.float64))
            fit_rows = {H[i].tobytes() for i in W["idx"]["defense_fit"]}
            ndup = int(sum(H[i].tobytes() in fit_rows for i in cr))
        for who, fu in (("nominee", f"{P}__FAREFIT_c{nsrc}"), ("zero_fairness", f"{P}__FAREFIT_Z")):
            blk = (A.get("amended") or {}).get(who) or {}
            for kind in want:
                cert = blk.get(kind)
                if cert is None:
                    prob.append(f"amended {who}/{kind} missing")
                    continue
                prob += ["amended " + x for x in cert_block_problems(cert, who, kind, US, fu, W, ccfg, sec)]
                agg.setdefault("certificates", {}).setdefault(P, {})[f"amended_A1:{who}/{kind}"] = {
                    "status": cert.get("status"), "bound": cert.get("bound"), "reason": cert.get("reason")}
            rd = blk.get("cert_rows_with_feature_vector_equal_to_a_fit_row")
            if ndup is not None and rd != ndup:
                prob.append(f"{who}: feature-duplicate count {rd} != replay {ndup}")
        rep.check(sc, f"{P}:certificates_amendment_A1", not prob,
                  cause="amended certificate inconsistent with its premises or the row-identity guard",
                  problems=prob[:10], feature_duplicate_cert_rows=ndup)


def check_controls(rep, run: Path, ds, agg):
    p = run / "controls" / f"{ds}.json"
    if not p.exists():
        rep.add("controls", f"{ds}:controls", "UNRESOLVED", cause="controls file missing")
        return
    C = json.loads(p.read_text())
    bad, flagged, undetected = [], [], []
    for name, r in C.items():
        if (r.get("null_flag_above_0.55") != (r.get("null_val_macro_auc", 0) > 0.55) or
                r.get("planted_detected_above_0.75") != (r.get("planted_val_macro_auc", 0) > 0.75)):
            bad.append(name)
        if r.get("null_flag_above_0.55"):
            flagged.append(name)
        if not r.get("planted_detected_above_0.75"):
            undetected.append(name)
    agg.setdefault("controls", {})[ds] = {k: {"null": v.get("null_val_macro_auc"), "planted": v.get("planted_val_macro_auc")}
                                          for k, v in C.items()}
    rep.check("controls", f"{ds}:flags_consistent", not bad, cause="control flag inconsistent with its value",
              interfaces=bad)
    if flagged or undetected:
        rep.add("controls", f"{ds}:control_outcomes", "FLAG", "scientific flag (not a replay disagreement)",
                null_flagged=flagged, planted_undetected=undetected)


# ================================================================================================ bench (stage 1/2)
def parse_bench(u):
    ds, s, pur, attr, arm = u.split("__")
    out = {"dataset": ds, "seed": int(s[1:]), "purpose": pur, "attr": attr, "cell": f"{ds}__{pur}__{attr}",
           "armtag": arm}
    if arm.startswith("D_sigma"):
        sig, rs = arm[len("D_sigma"):].split("_rs")
        out.update(arm="D", sigma=float(sig), release_seed=int(rs), group=f"D_sigma{float(sig):g}")
    else:
        out.update(arm=arm, group=arm)
    return out


class BenchUnits:
    def __init__(self, root: Path, rep: Report, drop: dict | None = None):
        self.US = UnitStore(root, rep, "bench_custody")
        self.drop = drop or {}

    def preds(self, u):
        fr = self.US.json(u, "fit_records.json")
        src = fr.get("alias_of") or u
        p = self.US.npz(src)
        ds = u.split("__")[0]
        if ds in self.drop:
            keep = ~np.isin(p["assess_row_id"], self.drop[ds])
            n = len(keep)
            p = {k: (v[keep] if (getattr(v, "ndim", 0) >= 1 and v.shape[0] == n and not k.endswith("_prior")
                                 and k not in ("s_prior_fit", "t_prior_fit")) else v) for k, v in p.items()}
        return p, fr

    def supported(self, u):
        return self.US.json(u, "supported.json")


BENCH_RECIPES = {("rep", "NL"): "P__rep__NL__as{k}", ("rep", "L"): "P__rep__L__as{k}",
                 ("rep+outputs", "NL"): "P__repPLUSoutputs__NL__as{k}",
                 ("rep+outputs", "L"): "P__repPLUSoutputs__Lslate__as{k}",
                 ("outputs", "NL"): "P__outputs__NL__as{k}"}


def softmax(L):
    L = np.asarray(L, np.float64)
    Z = L - L.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def bench_unit_stats(eng, p, classes, clip=CLIP, want=None):
    """Unit-level expressions for the benchmark statistics needed by C1 / exposure (own definitions).
    want: optional set of (surface, recipe, metric) keys to register (others skipped)."""
    out = {}
    y = p["y_s"].astype(int)
    W_ = (lambda k: True) if want is None else (lambda k: k in want)
    for (surf, rec), pat in BENCH_RECIPES.items():
        if not W_((surf, rec, "macro_auc")):
            continue
        keys = [pat.format(k=k) for k in ATT if pat.format(k=k) in p]
        if keys:
            out[(surf, rec, "macro_auc")] = hier_mean([auc_expr(eng, y, p[kk], classes) for kk in keys])
    for nm in ("G1", "G2"):
        if f"{nm}_pred" in p and W_(("rep", nm, "r2")):
            pred = np.asarray(p[f"{nm}_pred"], np.float64)
            Y = np.eye(pred.shape[1])[y]
            num = ((Y - pred) ** 2).sum(1)
            den = ((Y - np.asarray(p[f"{nm}_prior"], np.float64)[None, :]) ** 2).sum(1)
            key = "skill|" + hash_key(num, den)
            eng.reg(key, lambda num=num, den=den: SkillBase(num, den))
            out[("rep", nm, "r2")] = [(1.0, key)]
    t = p["y_task"].astype(int)
    U2 = np.asarray(p["U2_P"], np.float64)
    if W_(("U2", "U2", "accuracy")):
        out[("U2", "U2", "accuracy")] = acc_expr(eng, t, U2)
    if W_(("U2", "U2", "log_loss")):
        Pc = np.clip(U2, clip, 1.0)
        Pc = Pc / Pc.sum(1, keepdims=True)
        ll = -np.log(Pc[np.arange(len(t)), t])
        key = "mean|" + hash_key(ll)
        eng.reg(key, lambda: MeanBase(ll))
        out[("U2", "U2", "log_loss")] = [(1.0, key)]
    if "U1_logits" in p and p["U1_logits"].size and W_(("U1", "U1", "accuracy")):
        out[("U1", "U1", "accuracy")] = acc_expr(eng, t, softmax(p["U1_logits"]))
    if "RHO_u" in p and W_(("rep", "RHO1", "rho1sq")):
        key = "rho|" + hash_key(p["RHO_u"], p["RHO_v"])
        eng.reg(key, lambda: RhoBase(p["RHO_u"], p["RHO_v"]))
        out[("rep", "RHO1", "rho1sq")] = [(1.0, key)]
    return out


def group_expr(stats_per_unit, key):
    have = [s[key] for s in stats_per_unit if key in s]
    if len(have) != len(stats_per_unit) or not have:
        return None
    return hier_mean(have)


def bench_units_list(bench: Path):
    d = bench / "units"
    return sorted(p.name for p in d.iterdir() if (p / "COMPLETE.json").exists())


EXPOSURE_KEYS = {("rep", "G1", "r2"), ("rep", "NL", "macro_auc"), ("rep+outputs", "NL", "macro_auc"),
                 ("U2", "U2", "accuracy")}
DIFF_KEYS = [("rep", "NL", "macro_auc"), ("rep", "L", "macro_auc"), ("rep+outputs", "NL", "macro_auc"),
             ("rep+outputs", "L", "macro_auc"), ("rep", "G1", "r2"), ("rep", "G2", "r2"), ("U2", "U2", "accuracy"),
             ("U2", "U2", "log_loss"), ("U1", "U1", "accuracy")]


def run_c1(rep, bench, pkg, agg, B=2000, seed=20261003, level=0.90):
    sc = "corrections_C1"
    BU = BenchUnits(bench / "units", rep)
    allu = bench_units_list(bench)
    by_cell = defaultdict(lambda: defaultdict(list))
    for u in allu:
        i = parse_bench(u)
        by_cell[i["cell"]][i["group"]].append(u)
    truncated = []
    for cell, groups in sorted(by_cell.items()):
        if "A" not in groups:
            continue
        sA = {parse_bench(u)["seed"] for u in groups["A"]}
        for g, us in groups.items():
            if g.startswith("D_") and {parse_bench(u)["seed"] for u in us} != sA:
                truncated.append((cell, g))
    rep.check(sc, "truncated_groups", sorted({c for c, _ in truncated}) == ["adult__employment_analysis__marital_status"],
              cause="truncated noise groups are not exactly the marital_status cell", truncated=truncated)
    c1 = json.loads((pkg / "corrections" / "C1_seed_pairing.json").read_text())
    rows_by_id = {r["id"]: r for r in c1 if "id" in r}
    corrected_csv = {r["id"]: r for r in read_csv(pkg / "MATCHED_COMPARISONS_CORRECTED.csv")
                     if r.get("correction_status") == "C1_seed_pairing"}
    a2 = (1 - level) / 2
    out = []
    for cell in sorted({c for c, _ in truncated}):
        groups = by_cell[cell]
        eng = Engine()
        any_u = groups["A"][0]
        classes = BU.supported(any_u)["sensitive"]["supported_classes"]
        stats = {}
        for us in groups.values():
            for u in us:
                stats[u] = bench_unit_stats(eng, BU.preds(u)[0], classes,
                                            want=set(DIFF_KEYS) | {("outputs", "NL", "macro_auc")})
        plans = {}
        for (c_, g) in truncated:
            if c_ != cell:
                continue
            Du = sorted(groups[g])
            seedsD = {parse_bench(u)["seed"] for u in Du}
            Ash = sorted(u for u in groups["A"] if parse_bench(u)["seed"] in seedsD)
            for kk in DIFF_KEYS:
                gd, ga = group_expr([stats[u] for u in Du], kk), group_expr([stats[u] for u in Ash], kk)
                if gd is None or ga is None:
                    continue
                plans[f"{cell}|{g}-A|{'|'.join(kk)}"] = (gd, ga)
                # unpaired version (original error) for diagnosis
                gaa = group_expr([stats[u] for u in sorted(groups["A"])], kk)
                plans[f"UNPAIRED::{cell}|{g}-A|{'|'.join(kk)}"] = (gd, gaa)
            out_cell = group_expr([stats[u] for u in Ash], ("outputs", "NL", "macro_auc"))
            for arm, us in ((g, Du), ("A", Ash)):
                gp = group_expr([stats[u] for u in us], ("rep+outputs", "NL", "macro_auc"))
                if gp is not None and out_cell is not None:
                    plans[f"{g}::{cell}|{arm}|plus_NL_minus_outputs_NL|macro_auc"] = (gp, out_cell)
        pts = eng.points()
        p0, _ = BU.preds(any_u)
        reps = eng.replicates(p0["assess_unit"], B, seed)
        for pid, (e1, e2) in plans.items():
            pt = combine(e1, pts) - combine(e2, pts)
            r = combine(e1, reps) - combine(e2, reps)
            lo, hi, _ = bound_pair(r, a2, 1 - a2, a2)
            sid = pid.split("::", 1)[1] if "::" in pid else pid
            if pid.startswith("UNPAIRED::"):
                o = next((x for x in c1 if x.get("id") == sid), None)
                out.append({"id": sid, "kind": "unpaired_reconstruction", "point": pt,
                            "original_point": o.get("original_point") if o else None})
                continue
            grp = pid.split("::")[0] if "::" in pid else None
            cands = [x for x in c1 if x.get("id") == sid and (grp is None or x.get("arm_group") == grp)]
            o = cands[0] if cands else None
            row = {"id": sid, "group": grp, "point": pt, "lower90": lo, "upper90": hi,
                   "runner_point": o.get("corrected_point") if o else None,
                   "runner_lower90": o.get("corrected_lower90") if o else None,
                   "runner_upper90": o.get("corrected_upper90") if o else None}
            out.append(row)
            if o is None:
                rep.add(sc, f"{sid}", "FAIL", cause="corrected row missing from C1_seed_pairing.json")
                continue
            tl = 6.5 * mc_se(r, a2) + 5e-5
            tu = 6.5 * mc_se(r, 1 - a2) + 5e-5
            okp = abs(pt - o["corrected_point"]) <= 1e-9
            okb = (lo is not None and o.get("corrected_lower90") is not None and
                   abs(lo - o["corrected_lower90"]) <= tl and abs(hi - o["corrected_upper90"]) <= tu)
            st = "PASS" if (okp and okb) else "FAIL"
            rep.add(sc, f"{sid}" + (f" [{grp}]" if grp else ""), st, None,
                    None if st == "PASS" else ("point" if not okp else "interval") + " disagreement",
                    point_abs_diff=abs(pt - o["corrected_point"]),
                    lower_abs_diff=None if lo is None or o.get("corrected_lower90") is None else abs(lo - o["corrected_lower90"]))
            cs = corrected_csv.get(sid)
            if cs is not None and (grp is None or "|A|" not in sid):
                ok = abs(float(cs["point"]) - pt) <= 1e-9
                rep.check(sc, f"{sid}:csv", ok, cause="MATCHED_COMPARISONS_CORRECTED.csv differs from replay",
                          abs_diff=abs(float(cs["point"]) - pt))
    n_corr = sum(1 for r in c1 if r.get("status") == "CORRECTED")
    n_rep = sum(1 for r in out if r.get("kind") != "unpaired_reconstruction" and r.get("group") is None) + \
        sum(1 for r in out if r.get("group") and "|A|" not in r["id"])
    rep.check(sc, "row_count", n_corr == n_rep == 50, cause="number of corrected C1 rows differs (expected 50)",
              runner=n_corr, replay=n_rep)
    unp = [r for r in out if r.get("kind") == "unpaired_reconstruction" and r.get("original_point") is not None]
    md = max([abs(r["point"] - r["original_point"]) for r in unp] or [float("nan")])
    rep.check(sc, "original_error_reconstructed", bool(unp) and md <= 1e-9,
              cause="pairing A over {0,1,2} does not reproduce the original (erroneous) values", max_abs_diff=md,
              n=len(unp))
    agg["C1"] = out


def run_c2(rep, bench, pkg, agg, B=2000, seed=20261003, level=0.90):
    sc = "corrections_C2"
    BU = BenchUnits(bench / "units", rep)
    c2 = json.loads((pkg / "corrections" / "C2_rho_stable.json").read_text())
    runner_units = {r["unit"]: r for r in c2["units"]}
    mine = {}
    worst = 0.0
    bad = []
    for u in bench_units_list(bench):
        p = BU.US.npz(u)
        if "RHO_u" not in p:
            continue
        v = RhoBase(p["RHO_u"], p["RHO_v"]).point()
        mine[u] = v
        r = runner_units.get(u)
        if r is None:
            bad.append((u, "missing"))
            continue
        d = abs(v - r["stable_point"])
        worst = max(worst, d)
        if not d <= 1e-9:
            bad.append((u, d))
    rep.check(sc, "unit_points", not bad and len(mine) == len(runner_units), cause="stable rho1^2 point disagreement",
              n_replay=len(mine), n_runner=len(runner_units), max_abs_diff=worst, problems=bad[:10])
    # corrected rows (group and unit level) with intervals
    a2 = (1 - level) / 2
    rec_csv = {r["id"]: r for r in read_csv(pkg / "RECOVERY_CORRECTED.csv")
               if r.get("correction_status") == "C2_stable_rho1sq"}
    cells = sorted({r["cell"] for r in c2["intervals"]})
    rows = []
    for cell in cells:
        us = [u for u in mine if parse_bench(u)["cell"] == cell]
        eng = Engine()
        groups = defaultdict(list)
        exprs = {}
        for u in us:
            p = BU.US.npz(u)
            key = "rho|" + hash_key(p["RHO_u"], p["RHO_v"])
            eng.reg(key, lambda p=p: RhoBase(p["RHO_u"], p["RHO_v"]))
            exprs[f"{u}|rep|RHO1|rho1sq"] = [(1.0, key)]
            groups[parse_bench(u)["group"]].append([(1.0, key)])
        for g, parts in groups.items():
            exprs[f"{cell}|{g}|rep|RHO1|rho1sq"] = hier_mean(parts)
        pts = eng.points()
        reps = eng.replicates(BU.US.npz(us[0])["assess_unit"], B, seed)
        for r in [x for x in c2["intervals"] if x["cell"] == cell]:
            e = exprs.get(r["id"])
            if e is None:
                rep.add(sc, r["id"], "FAIL", cause="id not reconstructible")
                continue
            pt = combine(e, pts)
            rr = combine(e, reps)
            lo, hi, _ = bound_pair(rr, a2, 1 - a2, a2)
            tl, tu = 6.5 * mc_se(rr, a2) + 5e-5, 6.5 * mc_se(rr, 1 - a2) + 5e-5
            okp = abs(pt - r["corrected_point"]) <= 1e-9
            okb = lo is not None and abs(lo - r["corrected_lower90"]) <= tl and abs(hi - r["corrected_upper90"]) <= tu
            rep.add(sc, r["id"], "PASS" if okp and okb else "FAIL", None,
                    None if okp and okb else ("point" if not okp else "interval") + " disagreement",
                    point_abs_diff=abs(pt - r["corrected_point"]), lower_abs_diff=None if lo is None else abs(lo - r["corrected_lower90"]))
            cs = rec_csv.get(r["id"])
            if cs is not None:
                rep.check(sc, f"{r['id']}:csv", abs(float(cs["point"]) - pt) <= 1e-9,
                          cause="RECOVERY_CORRECTED.csv differs from replay", abs_diff=abs(float(cs["point"]) - pt))
            rows.append({"id": r["id"], "point": pt, "lower90": lo, "upper90": hi})
    agg["C2"] = {"unit_points_max_abs_diff": worst, "n_units": len(mine), "corrected_rows": rows}


def bench_primary(rep, bench, bench_pkg, lock, agg, drop, B, seed, label, runner_rows, original_rows=None):
    """The 24 original benchmark endpoints (optionally on retained rows), own implementation."""
    sc = f"exposure_{label}"
    fam = json.loads((bench_pkg / "PRIMARY_FAMILY.json").read_text())
    eps = fam["endpoints"]
    if len(eps) != fam["size"] or fam["size"] != 24:
        rep.add(sc, "family_size_24", "FAIL", cause="benchmark family is not 24 rows")
    alpha = fam["alpha_family"] / fam["size"]
    BU = BenchUnits(bench / "units", rep, drop)
    allu = bench_units_list(bench)
    sig = {ds: lock["noise"]["sigma_star"][ds] for ds in ("adult", "hmda")}
    ss = json.loads((bench / "infer" / "SIGMA_STAR.json").read_text())
    for ds in sig:
        rep.check(sc, f"{ds}:sigma_star_lock_vs_benchmark", ss["datasets"][ds]["sigma_star"] == sig[ds],
                  cause="sigma* differs between lock and benchmark SIGMA_STAR.json")
    results = {}
    for cell in sorted({e["cell"] for e in eps}):
        ds = cell.split("__")[0]
        members = defaultdict(list)
        for u in allu:
            i = parse_bench(u)
            if i["cell"] == cell:
                if BU.US.json(u, "fit_records.json").get("status") == "NE":
                    rep.add(sc, f"{u}:NE_unit", "FLAG", "unit recorded as NE; excluded as in the benchmark")
                    continue
                members[i["group"]].append(u)
        armmap = {"A": "A", "B": "B", "C": "C", "Dstar": f"D_sigma{sig[ds]:g}"}
        eng = Engine()
        anyu = members["A"][0]
        sup = BU.supported(anyu)["sensitive"]
        classes = sup["supported_classes"]
        p0, _ = BU.preds(anyu)
        thr = sup["thresholds"]["assessment"]
        cnt = np.bincount(p0["y_s"].astype(int), minlength=sup["n_classes"])
        lost = [c for c in classes if cnt[c] < thr]
        tsup = BU.supported(anyu)["task"]
        tcnt = np.bincount(p0["y_task"].astype(int), minlength=tsup["n_classes"])
        tlost = [c for c in tsup["supported_classes"] if tcnt[c] < tsup["thresholds"]["assessment"]]
        rep.check(sc, f"{cell}:support_retained", not lost and not tlost,
                  cause="a supported class falls below the assessment threshold on retained rows",
                  sensitive_counts=cnt.tolist(), task_counts=tcnt.tolist(), lost=lost, task_lost=tlost)
        stats, rows_ref = {}, None
        for g, us in members.items():
            for u in us:
                p, _ = BU.preds(u)
                r = (p["assess_row_id"], p["assess_unit"])
                if rows_ref is None:
                    rows_ref = r
                elif not (np.array_equal(r[0], rows_ref[0]) and np.array_equal(r[1], rows_ref[1])):
                    rep.add(sc, f"{cell}:identical_rows", "FAIL", cause="assessment rows differ within cell")
                stats[u] = bench_unit_stats(eng, p, classes, want=EXPOSURE_KEYS)
        gexpr = {}
        for g, us in members.items():
            for kk in DIFF_KEYS + [("rep", "G1", "r2")]:
                e = group_expr([stats[u] for u in sorted(us)], kk)
                if e is not None:
                    gexpr[(g, kk)] = e
        plans = {}
        for e in [x for x in eps if x["cell"] == cell]:
            g = armmap[e["arm"]]
            st = e["statistic"]
            if st == "G1_r2":
                plans[e["id"]] = (gexpr.get((g, ("rep", "G1", "r2"))), None)
            elif st == "C_rep_NL_macro_auc":
                plans[e["id"]] = (gexpr.get((g, ("rep", "NL", "macro_auc"))), None)
            elif st == "C_rep_plus_clean_out_NL_macro_auc":
                plans[e["id"]] = (gexpr.get((g, ("rep+outputs", "NL", "macro_auc"))), None)
            elif st == "U2_accuracy_diff_vs_A":
                plans[e["id"]] = (gexpr.get((g, ("U2", "U2", "accuracy"))), gexpr.get(("A", ("U2", "U2", "accuracy"))))
            # membership of the endpoint group
            seeds = sorted({parse_bench(u)["seed"] for u in members.get(g, [])})
            rels = sorted({parse_bench(u).get("release_seed") for u in members.get(g, [])} - {None})
            ok = seeds == [0, 1, 2] and (e["arm"] != "Dstar" or (rels == [0, 1, 2] and len(members[g]) == 9))
            rep.check(sc, f"{e['id']}:membership", ok, cause="endpoint group lacks a seed", seeds=seeds,
                      release_seeds=rels, n_members=len(members.get(g, [])))
        pts = eng.points()
        t0 = time.process_time()
        reps = eng.replicates(rows_ref[1], B, seed)
        agg.setdefault("timing", {})[f"{label}_{cell}_cpu_s"] = time.process_time() - t0
        n_units = int(len(np.unique(rows_ref[1])))
        for e in [x for x in eps if x["cell"] == cell]:
            e1, e2 = plans[e["id"]]
            res = {"id": e["id"], "statistic": e["statistic"], "bar": e["bar"], "n_rows": int(len(rows_ref[0])),
                   "n_units": n_units, "B": B, "seed": seed, "alpha_each": alpha}
            if e1 is None or (e["statistic"] == "U2_accuracy_diff_vs_A" and e2 is None):
                res.update(decision="NE", point=None, lower=None, upper=None)
                results[e["id"]] = res
                continue
            pt = combine(e1, pts) - (combine(e2, pts) if e2 else 0.0)
            r = combine(e1, reps) - (combine(e2, reps) if e2 else 0.0)
            lo, hi, n_ne = bound_pair(r, alpha, 1 - alpha, alpha)
            if lo is None:
                dec = "NE"
            elif e["statistic"] == "U2_accuracy_diff_vs_A":
                dec = "NONINFERIOR" if lo >= e["bar"] else ("INFERIOR" if hi < e["bar"] else "UNRESOLVED")
            else:
                dec = "ESTABLISHED_ABOVE" if lo > e["bar"] else ("ESTABLISHED_BELOW" if hi < e["bar"] else "UNRESOLVED")
            res.update(point=pt, lower=lo, upper=hi, decision=dec, n_ne_replicates=n_ne,
                       mc_se_lower=mc_se(r, alpha), mc_se_upper=mc_se(r, 1 - alpha))
            results[e["id"]] = res
    # compare with the runner (and label STABLE / CHANGED against the original decisions)
    for e in eps:
        m = results.get(e["id"], {})
        if original_rows is not None:
            od = (original_rows.get(e["id"]) or {}).get("decision")
            m["original_decision"] = od
            m["label"] = ("UNRESOLVED" if m.get("decision") in (None, "NE", "UNRESOLVED") or od in (None, "UNRESOLVED")
                          else ("STABLE" if od == m.get("decision") else "CHANGED"))
        if runner_rows is None:
            rep.add(sc, f"{e['id']}:vs_runner", "UNRESOLVED", cause="runner values not found")
            continue
        r = runner_rows.get(e["id"])
        if r is None:
            rep.add(sc, f"{e['id']}:vs_runner", "FAIL", cause="row missing in runner output")
            continue
        rp, rl, ru, rd = fnum(r.get("point")), fnum(r.get("lower")), fnum(r.get("upper")), r.get("decision")
        if m.get("point") is None:
            rep.check(sc, f"{e['id']}:vs_runner", rp is None, cause="replay NE but runner has a value")
            continue
        tl, tu = 6.5 * m["mc_se_lower"] + 5e-5, 6.5 * m["mc_se_upper"] + 5e-5
        okp = rp is not None and abs(rp - m["point"]) <= 1e-9
        okb = (rl is not None and ru is not None and abs(rl - m["lower"]) <= tl and abs(ru - m["upper"]) <= tu)
        near = min(abs(m["lower"] - e["bar"]), abs(m["upper"] - e["bar"])) <= max(tl, tu)
        okd = rd == m["decision"]
        if okp and okb and okd:
            st, cause = ("MC_BORDERLINE", "a bound is within 6.5 MC-SE + 5e-5 of the bar: the decision is reproduced "
                         "under the mirrored draw, but an independent draw could flip it") if near else ("PASS", None)
        elif okp and okd is False and near:
            st, cause = "MC_BORDERLINE", "decision differs within the MC tolerance band"
        else:
            st, cause = "FAIL", ("point" if not okp else ("bounds" if not okb else "decision")) + " disagreement"
        rep.add(sc, f"{e['id']}:vs_runner", st, None, cause, decision_agrees=bool(okd),
                replay={k: m.get(k) for k in ("point", "lower", "upper", "decision")},
                runner={"point": rp, "lower": rl, "upper": ru, "decision": rd},
                abs_diff_point=None if rp is None else abs(rp - m["point"]),
                abs_diff_lower=None if rl is None else abs(rl - m["lower"]),
                abs_diff_upper=None if ru is None else abs(ru - m["upper"]))
    agg[f"exposure_{label}"] = [results.get(e["id"]) for e in eps]
    return results


def exposure_drop(bench: Path, rep: Report, agg):
    """Rule: drop every assessment row whose canon_key occurs in the train split, with its whole assess_unit group."""
    drop, info = {}, {}
    for ds in ("adult", "hmda"):
        L = np.load(bench / "inputs" / f"{ds}_labels.npz", allow_pickle=False)
        split, ck, role = L["split"].astype(str), L["canon_key"].astype(str), L["role"].astype(str)
        tk = set(ck[split == "train"].tolist())
        exposed = (split == "test") & np.array([k in tk for k in ck])
        ex_a = exposed & (role == "assessment")
        units = np.unique(L["unit"][ex_a])
        grp_rows = (role == "assessment") & np.isin(L["unit"], units)
        drop[ds] = L["row_id"][grp_rows].astype(np.int64)
        info[ds] = {"exposed_test_rows_all_roles": int(exposed.sum()), "exposed_assessment_rows": int(ex_a.sum()),
                    "groups": int(len(units)), "rows_dropped_with_groups": int(grp_rows.sum())}
        exp = EXPECT_EXPOSURE[ds][1]
        rep.check("exposure_rule", f"{ds}:exposed_test_rows_all_roles", int(exposed.sum()) == EXPECT_EXPOSURE[ds][0],
                  cause="exposed test rows differ from 17 Adult / 42 HMDA", replay=int(exposed.sum()))
        rep.check("exposure_rule", f"{ds}:dropped_rows", int(ex_a.sum()) == exp == int(grp_rows.sum()),
                  cause="dropped assessment rows differ from the rule (7 Adult / 14 HMDA) or group-drop adds rows",
                  **info[ds])
    agg["exposure_rule"] = info
    return drop


# ================================================================================================ main
def load_runner_primary(pkg: Path, path: str | None):
    for p in ([Path(path)] if path else [pkg / "PRIMARY_ENDPOINTS.csv", pkg / "tables" / "PRIMARY_ENDPOINTS.csv"]):
        if p.exists():
            return read_csv(p), p
    return None, None


def load_exposure_runner(exposure_dir: Path, pkg: Path):
    p = exposure_dir / "INFER_ALL_retained.json"
    if p.exists():
        d = json.loads(p.read_text())
        return {e["id"]: e for e in d["primary"]["endpoints"]}, p
    for q in (pkg / "exposure" / "EXPOSURE_PRIMARY.csv", pkg / "EXPOSURE_SENSITIVITY_PRIMARY.csv"):
        if q.exists():
            return {r["id"]: r for r in read_csv(q)}, q
    return None, None


def summarize(rep: Report):
    by_status = Counter(i["status"] for i in rep.items)
    by_scope = defaultdict(Counter)
    for i in rep.items:
        by_scope[i["scope"]][i["status"]] += 1
    nonpass = [{k: i.get(k) for k in ("scope", "id", "status", "cause", "detail")}
               for i in rep.items if i["status"] != "PASS"]
    return {"counts_by_status": dict(by_status), "counts_by_scope": {k: dict(v) for k, v in sorted(by_scope.items())},
            "non_pass": nonpass}


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, Path):
        return tilde(o)
    return o


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pkg", default=str(PKG_DEFAULT))
    ap.add_argument("--run", default=str(HOME / "PCRL_eval_cache_private" / "oar_v1" / "run"))
    ap.add_argument("--bench", default=str(HOME / "PCRL_eval_cache_private" / "bench_v1"))
    ap.add_argument("--bench-pkg", default=str(WT_DEFAULT / "results" / "combined_matched_removal_benchmark_v1"))
    ap.add_argument("--exposure-dir", default=str(HOME / "PCRL_eval_cache_private" / "oar_v1" / "exposure"))
    ap.add_argument("--runner-primary", default=None, help="runner PRIMARY_ENDPOINTS.csv (default <pkg>/...)")
    ap.add_argument("--datasets", default="adult,hmda")
    ap.add_argument("--stages", default="roles,leakage,nominee,primary,membership,contracts,controls,c1,c2,exposure")
    ap.add_argument("--exposure-original", action="store_true",
                    help="also recompute the 24 original endpoints on all rows and compare to the benchmark table")
    ap.add_argument("--B-primary", type=int, default=None)
    ap.add_argument("--B-exposure", type=int, default=None)
    ap.add_argument("--B-corrections", type=int, default=2000)
    ap.add_argument("--expect-exposure", default=None, help='JSON {"adult": [17, 7], ...} (synthetic tests)')
    ap.add_argument("--out", default=None, help="INDEPENDENT_VERIFICATION.json path (default <pkg>/...)")
    ap.add_argument("--agg", default=None, help="aggregate path (default <pkg>/verification/replay_results_aggregate.json)")
    a = ap.parse_args(argv)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        print("note: OMP_NUM_THREADS is not 1", file=sys.stderr)
    for mod in list(sys.modules):
        if mod.startswith(("oar", "stored_model_eval")) and mod.split(".")[0] in ("oar", "stored_model_eval"):
            raise SystemExit("replay must not import runner code")
    t_wall, t_cpu = time.time(), time.process_time()
    if a.expect_exposure:
        EXPECT_EXPOSURE.update(json.loads(a.expect_exposure))
    pkg, run, bench, bench_pkg = Path(a.pkg), Path(a.run), Path(a.bench), Path(a.bench_pkg)
    inputs = bench / "inputs"
    stages = set(a.stages.split(","))
    dss = [d for d in a.datasets.split(",") if d]
    lock = json.loads((pkg / "EXECUTION_LOCK.json").read_text())
    ras = json.loads((pkg / "ROLES_AND_SUPPORT.json").read_text())
    rep, agg = Report(), {}
    Bp = a.B_primary or lock["primary_family"]["B"]
    Sp = lock["primary_family"]["seed"]
    agg["settings"] = {"B_primary": Bp, "seed_primary": Sp, "chunk": CHUNK, "quantile": "linear",
                       "alpha_each": lock["primary_family"]["alpha_each"], "datasets": dss, "stages": sorted(stages)}
    rep.check("lock", "roles_and_support_hash",
              hashlib.sha256(json.dumps(ras, sort_keys=True).encode()).hexdigest() == lock["roles_and_support_sha256"],
              cause="ROLES_AND_SUPPORT.json changed after the lock")
    fam = read_family(pkg, lock, rep)
    for ds in dss:
        cv = lock["roles_rule"]["class_vocabulary"][ds]
        rep.check("lock", f"{ds}:sigma_star", lock["noise"]["sigma_star"][ds] == {"adult": 2.0, "hmda": 4.0}[ds],
                  cause="lock sigma* differs from protocol")
        rep.check("lock", f"{ds}:vocabulary_vs_protocol", cv["attr"] == CELLS[ds]["attr"], cause="attribute differs")
    roles, US = {}, UnitStore(run / "units", rep, "custody")
    uids_all = list_units(run) if (run / "units").exists() else []
    nominees = {}
    for ds in dss:
        W = build_roles(ds, inputs, lock)
        roles[ds] = W
        if "roles" in stages:
            agg.setdefault("roles", {})[ds] = check_roles(rep, W, ras, lock, ds)
        uids = [u for u in uids_all if u.startswith(ds + "__")]
        if "leakage" in stages:
            check_role_disjointness(rep, W, ds)
            check_units_rows(rep, US, W, uids, ds)
            check_farefit(rep, US, W, uids, ds, inputs)
        # nominee (needed by primary/membership/contracts)
        for k in SEEDS:
            P = f"{ds}__s{k}"
            sp = run / "selection" / f"{P}.json"
            if not sp.exists():
                rep.add("nominee", f"{P}:selection_file", "UNRESOLVED", cause="selection file missing (seed not run)")
                continue
            sel = json.loads(sp.read_text())
            mine = recompute_nominee(rep, US, W, ds, k, lock, sel) if "nominee" in stages else None
            if mine is None:
                mine = {"nominee_unit_source": sel.get("nominee_unit_source"), "admissible": sel.get("admissible")}
            mine["runner_nominee_unit_source"] = sel.get("nominee_unit_source")
            mine["runner_nominee"] = sel.get("nominee")
            nominees[(ds, k)] = mine
            agg.setdefault("nominee", {})[P] = {kk: v for kk, v in mine.items()}
        rep.check("nominee", f"{ds}:all_encoder_seeds_have_selection",
                  all((ds, k) in nominees for k in SEEDS), cause="an encoder seed has no FARE selection",
                  seeds=[k for k in SEEDS if (ds, k) in nominees])
    if "primary" in stages:
        rr, rpath = load_runner_primary(pkg, a.runner_primary)
        agg["runner_primary_table"] = tilde(rpath) if rpath else None
        run_primary(rep, US, roles, lock, fam, nominees, rr, Bp, Sp, agg)
    for ds in dss:
        uids = [u for u in uids_all if u.startswith(ds + "__")]
        if "membership" in stages:
            check_membership(rep, US, roles[ds], ds, lock, uids, agg, inputs)
        if "contracts" in stages:
            check_heads(rep, US, roles[ds], ds, inputs, bench, nominees, agg)
            check_certificates(rep, US, roles[ds], ds, lock, nominees, agg, inputs)
        if "controls" in stages:
            check_controls(rep, run, ds, agg)
    if "c1" in stages:
        run_c1(rep, bench, pkg, agg, B=a.B_corrections)
    if "c2" in stages:
        run_c2(rep, bench, pkg, agg, B=a.B_corrections)
    if "exposure" in stages:
        bfam = json.loads((bench_pkg / "PRIMARY_FAMILY.json").read_text())
        orig = {r["id"]: r for r in read_csv(bench_pkg / "PRIMARY_ENDPOINTS.csv")}
        Be = a.B_exposure or 20000
        if a.exposure_original:
            bench_primary(rep, bench, bench_pkg, lock, agg, {}, Be, 20261004, "original_custody", orig)
        drop = exposure_drop(bench, rep, agg)
        er, epath = load_exposure_runner(Path(a.exposure_dir), pkg)
        agg["exposure_runner_table"] = tilde(epath) if epath else None
        bench_primary(rep, bench, bench_pkg, lock, agg, drop, Be, 20261004, "retained", er, original_rows=orig)
        agg["exposure_family_size"] = bfam["size"]
    agg["timing"] = {**agg.get("timing", {}), "wall_s": time.time() - t_wall, "cpu_s": time.process_time() - t_cpu}
    code_sha = sha_file(Path(__file__))
    out = {"schema": "oar_independent_verification/v1",
           "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "replay_code": {"path": "results/combined_output_aware_removal_v1/verification/replay_oar.py",
                           "sha256": code_sha, "imports_runner_code": False, "numpy": np.__version__},
           "inputs": {"pkg": tilde(pkg), "run": tilde(run), "bench": tilde(bench), "bench_pkg": tilde(bench_pkg),
                      "exposure_dir": tilde(a.exposure_dir)},
           "settings": agg["settings"],
           "public_summary": summarize(rep),
           "primary_endpoints": [{k: v for k, v in (r or {}).items() if k != "members"} for r in agg.get("primary_endpoints", [])],
           "competitive": agg.get("competitive"),
           "items": rep.items}
    outp = Path(a.out) if a.out else pkg / "INDEPENDENT_VERIFICATION.json"
    aggp = Path(a.agg) if a.agg else pkg / "verification" / "replay_results_aggregate.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    aggp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(jsonable(out), indent=1))
    aggp.write_text(json.dumps(jsonable(agg), indent=1))
    s = out["public_summary"]
    print(json.dumps({"counts_by_status": s["counts_by_status"], "n_non_pass": len(s["non_pass"]),
                      "wall_s": round(time.time() - t_wall, 1)}, indent=1))
    return out


if __name__ == "__main__":
    main()
