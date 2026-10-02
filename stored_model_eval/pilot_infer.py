"""Pilot inference and report: consumes ONLY the saved per-unit outputs (preds.npz, supported.json, fit_records.json).

No model is refitted and nothing is selected here. Every statistic is a function of row weights W over the
assessment rows, so a cluster bootstrap over assessment units re-weights rows with the fitted predictors, the
attacker_fit priors and the frozen support held fixed.

Bootstrap draw (identical for exploratory and primary streams, different seeds):
    units = numpy.unique(assess_unit); idx = position of each row's unit in `units`
    rng = numpy.random.default_rng(seed)
    for b in range(B): counts = rng.multinomial(n_units, numpy.full(n_units, 1/n_units)); w_row = counts[idx]
All units share the same assessment rows (checked by ID), so one draw re-weights every unit, every surface and
every release seed in the same replicate (paired differences; within-replicate seed aggregation).

Exploratory: two-sided 90% percentile intervals, B=2000, seed 20261002, decisions vs bars / tau (unadjusted).
Primary family (16 endpoints, frozen in effective.PRIMARY_FAMILY): Bonferroni simultaneous one-sided percentile
bounds at alpha = 0.05/16, B = 20000, seed 20261003.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from .admission import sha256_file
from .effective import (EFFECTIVE_PROTOCOL, OUTPUTS_REUSE_SOURCE, PRIMARY_FAMILY, Tracked, consumption_report,
                        effective_hash, unit_info)
from .metrics import to_jsonable

SURF_FROM_KEY = {"rep": "rep", "outputs": "outputs", "repPLUSoutputs": "rep+outputs"}


# --------------------------------------------------------------------------------------------------
# bootstrap weights
# --------------------------------------------------------------------------------------------------


class UnitBootstrap:
    def __init__(self, assess_unit: np.ndarray, B: int, seed: int, chunk: int):
        self.units, self.idx = np.unique(np.asarray(assess_unit), return_inverse=True)
        self.n_units, self.n_rows = len(self.units), len(self.idx)
        self.B, self.seed, self.chunk = int(B), int(seed), int(chunk)

    def chunks(self):
        """Yield (n_rows x c) weight matrices; replicate order is the sequential RNG order."""
        rng = np.random.default_rng(self.seed)
        p = np.full(self.n_units, 1.0 / self.n_units)
        done = 0
        while done < self.B:
            c = min(self.chunk, self.B - done)
            counts = np.stack([rng.multinomial(self.n_units, p) for _ in range(c)], axis=1)  # n_units x c
            yield counts[self.idx].astype(np.float64)
            done += c


# --------------------------------------------------------------------------------------------------
# vectorised weighted statistics: each factory returns f(WT) -> (c,) with WT of shape (n_rows, c)
# --------------------------------------------------------------------------------------------------


def _ratio_skill(num_vec, den_vec):
    num_vec, den_vec = np.asarray(num_vec, float), np.asarray(den_vec, float)

    def f(WT):
        d = den_vec @ WT
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(d > 0, 1.0 - (num_vec @ WT) / d, np.nan)
    return f


def _weighted_mean(vec):
    vec = np.asarray(vec, float)

    def f(WT):
        d = WT.sum(0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(d > 0, (vec @ WT) / d, np.nan)
    return f


def _auc_prep(score, pos, rows, n):
    score, pos, rows = np.asarray(score, float), np.asarray(pos, bool), np.asarray(rows)
    s, p, r = score[rows], pos[rows], rows
    _, inv = np.unique(s, return_inverse=True)
    U = int(inv.max()) + 1 if len(inv) else 0
    Mp = csr_matrix((np.ones(int(p.sum())), (inv[p], r[p])), shape=(U, n))
    Mn = csr_matrix((np.ones(int((~p).sum())), (inv[~p], r[~p])), shape=(U, n))
    return Mp, Mn


def _auc_eval(prep, WT):
    Mp, Mn = prep
    gp, gn = np.asarray(Mp @ WT), np.asarray(Mn @ WT)
    below = np.cumsum(gn, 0) - gn
    num = (gp * (below + 0.5 * gn)).sum(0)
    den = gp.sum(0) * gn.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def auc_class_stats(y, P, classes, pairs):
    """Return dict of factories: macro (mean over supported classes), worst_class, worst_pair."""
    y = np.asarray(y).astype(int)
    P = np.asarray(P, float)
    n = len(y)
    allrows = np.arange(n)
    preps = [_auc_prep(P[:, k], y == k, allrows, n) for k in classes]
    pair_preps = []
    for i, j in pairs:
        rows = np.flatnonzero((y == i) | (y == j))
        den = P[rows, i] + P[rows, j]
        sc = np.full(n, 0.5)
        sc[rows] = np.where(den > 0, P[rows, j] / np.where(den > 0, den, 1.0), 0.5)
        pair_preps.append(_auc_prep(sc, y == j, rows, n))

    def per_class(WT):
        return np.stack([_auc_eval(p, WT) for p in preps])

    def macro(WT):
        return per_class(WT).mean(0)

    def worst_class(WT):
        return per_class(WT).max(0)

    def worst_pair(WT):
        return np.stack([_auc_eval(p, WT) for p in pair_preps]).max(0) if pair_preps else \
            np.full(WT.shape[1], np.nan)
    return {"macro_auc": macro, "worst_class_auc": worst_class, "worst_pair_auc": worst_pair}


def prob_skill_stats(y, P, prior, clip):
    y = np.asarray(y).astype(int)
    P = np.clip(np.asarray(P, float), clip, 1.0)
    P = P / P.sum(1, keepdims=True)
    prior = np.clip(np.asarray(prior, float), clip, 1.0)
    K = P.shape[1]
    Y = np.eye(K)[y]
    ll = -np.log(P[np.arange(len(y)), y])
    llp = -np.log(prior[y])
    br = ((P - Y) ** 2).sum(1)
    brp = ((prior[None, :] - Y) ** 2).sum(1)
    # Addendum D1 #2: both, named. LLR_nats = LL0 - LL (nats per row), LL_skill = 1 - LL/LL0; LL0 = attacker_fit prior
    return {"LLR_nats": _weighted_mean(llp - ll), "LL_skill": _ratio_skill(ll, llp),
            "brier_skill": _ratio_skill(br, brp)}


def r2_stat(y, pred, prior):
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred, float)
    Y = np.eye(pred.shape[1])[y]
    return _ratio_skill(((Y - pred) ** 2).sum(1), ((Y - np.asarray(prior, float)[None, :]) ** 2).sum(1))


def rho_stat(u, v):
    """Weighted squared correlation of the saved per-row RHO_u = H b and RHO_v = a[y] (directions fixed)."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    M = np.stack([np.ones_like(u), u, v, u * u, v * v, u * v])

    def f(WT):
        s0, su, sv, suu, svv, suv = M @ WT
        with np.errstate(divide="ignore", invalid="ignore"):
            cov = suv / s0 - (su / s0) * (sv / s0)
            vu = suu / s0 - (su / s0) ** 2
            vv = svv / s0 - (sv / s0) ** 2
            return np.where((vu > 0) & (vv > 0), cov ** 2 / (vu * vv), np.nan)
    return f


def softmax(L):
    L = np.asarray(L, float)
    L = L - L.max(1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(1, keepdims=True)


def macro_f1_stat(y, pred, classes):
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    TP = np.stack([((pred == k) & (y == k)) for k in classes]).astype(float)
    FP = np.stack([((pred == k) & (y != k)) for k in classes]).astype(float)
    FN = np.stack([((pred != k) & (y == k)) for k in classes]).astype(float)

    def f(WT):
        tp, fp, fn = TP @ WT, FP @ WT, FN @ WT
        den = 2 * tp + fp + fn
        with np.errstate(divide="ignore", invalid="ignore"):
            f1 = np.where(den > 0, 2 * tp / den, np.nan)
        return f1.mean(0)
    return f


def utility_stats(y_task, P, classes, clip, metrics_all, metrics_sup):
    """accuracy / log_loss over ALL assessment rows; macro_f1 / macro_auc over supported task classes (D1 9f)."""
    y = np.asarray(y_task).astype(int)
    P = np.asarray(P, float)
    pred = P.argmax(1)
    Pc = np.clip(P, clip, 1.0)
    Pc = Pc / Pc.sum(1, keepdims=True)
    ll = -np.log(Pc[np.arange(len(y)), y])
    allm = {"accuracy": _weighted_mean((pred == y).astype(float)), "log_loss": _weighted_mean(ll)}
    out = {m: allm[m] for m in metrics_all}
    if len(classes) >= 2:
        supm = {"macro_f1": macro_f1_stat(y, pred, classes),
                "macro_auc": auc_class_stats(y, P, classes, [])["macro_auc"]}
        out.update({m: supm[m] for m in metrics_sup})
    return out


# --------------------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------------------


def load_units(units_dir: Path, unit_ids) -> dict:
    from .pilot import verify_complete
    out = {}
    for u in unit_ids:
        d = Path(units_dir) / u
        if not d.exists():
            continue
        ok, errs = verify_complete(d)
        if not ok:
            raise ValueError(f"unit {u} outputs incomplete or modified: {errs}")
        with np.load(d / "preds.npz", allow_pickle=False) as z:
            preds = {k: z[k] for k in z.files}
        out[u] = {"preds": preds, "supported": json.loads((d / "supported.json").read_text()),
                  "fit_records": json.loads((d / "fit_records.json").read_text()),
                  "hashes": {f: sha256_file(d / f) for f in ("preds.npz", "supported.json", "fit_records.json")}}
    return out


# --------------------------------------------------------------------------------------------------
# statistic registry
# --------------------------------------------------------------------------------------------------


class Registry:
    def __init__(self):
        self.base = {}      # id -> (factory, meta)
        self.derived = {}   # id -> (combine(list of vectors) -> vector, [ids], meta)
        self.order = []

    def add(self, sid, fn, **meta):
        self.base[sid] = (fn, meta)
        self.order.append(sid)

    def add_derived(self, sid, parts, how, **meta):
        self.derived[sid] = (how, parts, meta)
        self.order.append(sid)


def _diff(v):
    return v[0] - v[1]


def _mean(v):
    return np.mean(np.stack(v), axis=0)


def _ratio(v):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(v[1] != 0, v[0] / v[1], np.nan)


def build_registry(U: dict, E) -> tuple[Registry, list]:
    reg = Registry()
    notes = []
    clip = E["recovery_metrics"]["prob_clip"]
    RM = E["recovery_metrics"]
    if RM["auc"] != "macro_ovr_supported":
        raise ValueError("unsupported AUC convention")
    # definitions implemented by auc_class_stats / prob_skill_stats; carried into the output verbatim
    notes.append({"metric_definitions": {k: RM[k] for k in ("auc", "LLR_nats", "LL_skill", "brier_skill",
                                                             "worst_class", "worst_pair")}})
    for u, d in U.items():
        info = unit_info(u)
        p, sup, fr = d["preds"], d["supported"]["sensitive"], d["fit_records"]
        y, t = p["y_s"], p["y_task"]
        cls, pairs = sup["supported_classes"], [tuple(x) for x in sup["supported_pairs"]]
        base_meta = {"unit": u, "kind": info["kind"], "sigma": info.get("sigma"),
                     "release_seed": info.get("release_seed")}
        if sup["status"] == "NE":
            notes.append({"unit": u, "status": "NE", "reason": sup["ne_reason"]})
        else:
            for name, pk, pr in (("G1", "G1_pred", "G1_prior"), ("G2", "G2_pred", "G2_prior")):
                reg.add(f"{u}|{name}|r2", r2_stat(y, p[pk], p[pr]), quantity=name, metric="r2", scale="R2",
                        surface="rep", recipe=name, **base_meta)
            if "RHO_u" in p:
                reg.add(f"{u}|RHO1|rho1sq", rho_stat(p["RHO_u"], p["RHO_v"]), quantity="RHO1",
                        metric="rho1sq", scale="R2", surface="rep", recipe="RHO1", **base_meta)
            pm = auc_class_stats(y, p["G1_pred"], cls, pairs)
            reg.add(f"{u}|G1|macro_auc", pm["macro_auc"], quantity="pure_metric_G1", metric="macro_auc",
                    scale="AUC", surface="rep", recipe="G1", **base_meta)
            for k in sorted(p):
                if not k.startswith("P__"):
                    continue
                _, sk, rec = k.split("__")
                surf = SURF_FROM_KEY[sk]
                if info["kind"] == "noise" and surf == "outputs":
                    note = {"unit": u, "surface": "outputs", "status": "REUSED",
                            "source": f"{OUTPUTS_REUSE_SOURCE}|outputs", "note": "not new evidence"}
                    if note not in notes:
                        notes.append(note)
                    continue
                for m, fn in auc_class_stats(y, p[k], cls, pairs).items():
                    reg.add(f"{u}|{surf}|{rec}|{m}", fn, quantity="recovery", metric=m, scale="AUC", surface=surf,
                            recipe=rec, **base_meta)
                for m, fn in prob_skill_stats(y, p[k], p["s_prior_fit"], clip).items():
                    reg.add(f"{u}|{surf}|{rec}|{m}", fn, quantity="recovery", metric=m, scale="skill",
                            surface=surf, recipe=rec, **base_meta)
            if info["kind"] == "untreated":
                lo = auc_class_stats(y, p["LO_P"], cls, pairs)
                reg.add(f"{u}|LO|macro_auc", lo["macro_auc"], quantity="label_only", metric="macro_auc", scale="AUC",
                        surface="label_only", recipe="LO", **base_meta)
                for m, fn in prob_skill_stats(y, p["LO_P"], p["s_prior_fit"], clip).items():
                    reg.add(f"{u}|LO|{m}", fn, quantity="label_only", metric=m, scale="skill",
                            surface="label_only", recipe="LO", **base_meta)
                if f"{u}|outputs|NL|macro_auc" in reg.base:
                    reg.add_derived(f"{u}|outputs_NL_minus_LO|macro_auc", [f"{u}|outputs|NL|macro_auc",
                                                                          f"{u}|LO|macro_auc"], _diff,
                                    quantity="output_leakage_beyond_label_only", metric="macro_auc_diff",
                                    scale="AUC_diff", **base_meta)
                # decomposition steps (one factor per step)
                steps = [("F3_minus_F2", f"{u}|rep|L|macro_auc", f"{u}|G1|macro_auc"),
                         ("F4_minus_F3", f"{u}|rep|NL|macro_auc", f"{u}|rep|L|macro_auc"),
                         ("F5_minus_F4", f"{u}|outputs|NL|macro_auc", f"{u}|rep|NL|macro_auc"),
                         ("F6_minus_F4", f"{u}|rep+outputs|NL|macro_auc", f"{u}|rep|NL|macro_auc")]
                for name, a, b in steps:
                    if a in reg.base and b in reg.base:
                        reg.add_derived(f"{u}|decomp|{name}", [a, b], _diff, quantity="decomposition_step",
                                        metric="macro_auc_diff", scale="AUC_diff", **base_meta)
            else:
                notes.append({"unit": u, "surface": "label_only", "status": "REUSED_EQUIVALENT",
                              "source": f"{OUTPUTS_REUSE_SOURCE}|LO",
                              "note": "LO depends only on (s, y_task); identical to the untreated pair"})
        # utility (always; independent of sensitive support)
        tcls = d["supported"]["task"]["supported_classes"]
        UT = E["utility"]
        ma, ms = list(UT["metrics_all_rows"]), list(UT["metrics_supported_task_classes"])
        ceiling = info["purpose"] in UT["near_ceiling_purposes"]
        umeta = dict(base_meta, purpose=info["purpose"], near_ceiling=ceiling)
        const = int(np.argmax(p["t_prior_fit"]))
        if UT["normalised_lift"]["constant_predictor"] != "attacker_fit_majority_task_class":
            raise ValueError("unsupported constant predictor")
        reg.add(f"{u}|Uconst|accuracy", _weighted_mean((t == const).astype(float)), quantity="Uconst",
                metric="accuracy", scale="utility", surface="none", recipe="constant", **umeta)
        Ps = {}
        if p["U1_logits"].size:
            Ps["U1"] = softmax(p["U1_logits"])
        else:
            notes.append({"unit": u, "quantity": "U1", "status": "NE", "reason": fr.get("U1", {}).get("reason")})
        Ps["U2"] = p["U2_P"]
        for q, PP in Ps.items():
            for m, fn in utility_stats(t, PP, tcls, clip, ma, ms).items():
                reg.add(f"{u}|{q}|{m}", fn, quantity=q, metric=m, scale="utility", surface="rep", recipe=q, **umeta)
            reg.add_derived(f"{u}|{q}|accuracy_lift", [f"{u}|{q}|accuracy", f"{u}|Uconst|accuracy"], _diff,
                            quantity=f"{q}_lift", metric="accuracy_lift_over_constant", scale="utility_diff", **umeta)
    # paired utility differences vs the untreated unit of the same purpose (noise units)
    for u in U:
        info = unit_info(u)
        if info["kind"] != "noise":
            continue
        ref = OUTPUTS_REUSE_SOURCE
        for q in ("U1", "U2"):
            for m in ("accuracy", "log_loss", "macro_f1", "macro_auc"):
                a, b = f"{u}|{q}|{m}", f"{ref}|{q}|{m}"
                if a in reg.base and b in reg.base:
                    reg.add_derived(f"{u}|{q}|{m}|diff_vs_untreated", [a, b], _diff, quantity=f"{q}_diff",
                                    metric=m, scale="utility_diff", unit=u, kind="noise", sigma=info["sigma"],
                                    release_seed=info["release_seed"], reference_unit=ref)
            a, b = f"{u}|{q}|accuracy_lift", f"{ref}|{q}|accuracy_lift"
            if a in reg.derived and b in reg.derived:
                reg.add_derived(f"{u}|{q}|normalised_lift", [a, b], _ratio, quantity=f"{q}_normalised_lift",
                                metric="normalised_accuracy_lift", scale="utility_ratio", unit=u, kind="noise",
                                sigma=info["sigma"], release_seed=info["release_seed"], reference_unit=ref,
                                clean_lift_id=b)
    # within-replicate seed aggregation for noise arms
    groups = defaultdict(list)
    sig_ok, seed_ok = set(E["units"]["noise_sigmas"]), set(E["units"]["noise_seeds"])
    unt, noi = set(E["units"]["untreated"]), set(E["units"]["noise"])
    for u in U:
        info = unit_info(u)
        if (info["kind"] == "noise") != (u in noi) or (info["kind"] == "untreated") != (u in unt):
            raise ValueError(f"unit {u} kind disagrees with the effective unit lists")
        if info["kind"] == "noise":
            if info["sigma"] not in sig_ok or info["release_seed"] not in seed_ok:
                raise ValueError(f"unit {u}: sigma/seed outside the frozen grid")
            groups[(u.rsplit("_seed", 1)[0], info["sigma"])].append(u)
    for (gname, sigma), members in sorted(groups.items(), key=lambda x: x[0][1]):
        members = sorted(members)
        suffixes = defaultdict(list)
        for sid in reg.order:
            uu, rest = sid.split("|", 1)
            if uu in members:
                suffixes[rest].append(sid)
        for rest, sids in suffixes.items():
            if len(sids) == len(members):
                meta = dict((reg.base.get(sids[0]) or reg.derived.get(sids[0]))[-1])
                meta.update(unit=f"{gname}__seedmean", kind="noise_seed_aggregate", release_seed=None,
                            members=members, n_seeds=len(members))
                reg.add_derived(f"{gname}__seedmean|{rest}", sids, _mean, **meta)
        if len(members) != 3:
            notes.append({"group": gname, "status": "INCOMPLETE_SEEDS", "members": members})
    return reg, notes


def run_registry(reg: Registry, boot: UnitBootstrap, ids=None) -> tuple[dict, dict]:
    """Evaluate base stats on all replicates (chunked) and derived stats from them. Returns (points, reps)."""
    need = set(ids) if ids is not None else set(reg.order)
    # closure over derived dependencies
    stack = list(need)
    while stack:
        s = stack.pop()
        if s in reg.derived:
            for p in reg.derived[s][1]:
                if p not in need:
                    need.add(p)
                    stack.append(p)
    base_ids = [s for s in reg.order if s in need and s in reg.base]
    ones = np.ones((boot.n_rows, 1))
    points = {s: float(reg.base[s][0](ones)[0]) for s in base_ids}
    reps = {s: [] for s in base_ids}
    for WT in boot.chunks():
        for s in base_ids:
            reps[s].append(reg.base[s][0](WT))
    reps = {s: np.concatenate(v) for s, v in reps.items()}
    for s in reg.order:
        if s in need and s in reg.derived:
            how, parts, _ = reg.derived[s]
            points[s] = float(how([np.array([points[p]]) for p in parts])[0])
            reps[s] = how([reps[p] for p in parts])
    return points, reps


def _interval(r: np.ndarray, lo_q: float, hi_q: float, method: str, max_ne_frac: float):
    ok = np.isfinite(r)
    n_ne = int((~ok).sum())
    if ok.sum() == 0 or n_ne / len(r) > max_ne_frac:
        return None, n_ne
    v = r[ok]
    return (float(np.quantile(v, lo_q, method=method)), float(np.quantile(v, hi_q, method=method))), n_ne


def _decide(lo, hi, bar, below_inclusive: bool):
    if lo is None:
        return "NE"
    if lo > bar:
        return "ESTABLISHED_ABOVE"
    if (hi <= bar) if below_inclusive else (hi < bar):
        return "ESTABLISHED_BELOW"
    return "UNRESOLVED"


# --------------------------------------------------------------------------------------------------
# main entry points
# --------------------------------------------------------------------------------------------------


def infer_pilot(units_dir: Path, eff: dict | None = None, unit_ids=None) -> dict:
    effp = EFFECTIVE_PROTOCOL if eff is None else eff
    E = Tracked(effp)
    expected = list(E["units"]["registered"]) if unit_ids is None else list(unit_ids)
    U = load_units(units_dir, expected)
    if not U:
        raise ValueError(f"no complete units under {units_dir}")
    for u, d in U.items():
        if d["fit_records"]["effective_protocol_sha256"] != effective_hash(effp):
            raise ValueError(f"unit {u} was produced under a different effective protocol")
    first = next(iter(U.values()))["preds"]
    for u, d in U.items():
        if not np.array_equal(d["preds"]["assess_row_id"], first["assess_row_id"]) or \
                not np.array_equal(d["preds"]["assess_unit"], first["assess_unit"]):
            raise ValueError(f"unit {u}: assessment IDs/units differ from the other units (pairing impossible)")
    I = E["inference"]
    if I["sampling_unit"] != "assess_unit":
        raise ValueError("sampling unit must be assess_unit")
    reg, notes = build_registry(U, E)
    ex = I["exploratory"]
    boot = UnitBootstrap(first["assess_unit"], ex["B"], ex["seed"], I["chunk"])
    level = ex["level"]
    a2 = (1 - level) / 2
    pts, reps = run_registry(reg, boot)
    bars, tau = list(I["bars"]), I["tau"]
    rows = []
    for sid in reg.order:
        meta = (reg.base.get(sid) or reg.derived.get(sid))[-1]
        iv, n_ne = _interval(reps[sid], a2, 1 - a2, I["quantile_method"], a2)
        pt = pts[sid]
        row = {"id": sid, **{k: v for k, v in meta.items()}, "point": pt if np.isfinite(pt) else None,
               "interval": list(iv) if iv else None, "level": level, "boot_B": ex["B"], "boot_seed": ex["seed"],
               "n_ne_replicates": n_ne, "status": "ESTIMATED" if (iv and np.isfinite(pt)) else "NE",
               "label": "exploratory (unadjusted)"}
        sc = meta.get("scale")
        lo, hi = (iv if iv else (None, None))
        if sc == "AUC":
            row["decisions"] = {f"{b:.2f}": _decide(lo, hi, b, False) for b in bars}
        elif sc == "R2":
            # tau = 0.05 is primary-tau; the grid is sensitivity from the same exploratory interval (D1 9e)
            row["decisions"] = {f"tau={tt}": _decide(lo, hi, tt, True) for tt in I["tau_sensitivity"]}
        elif sc in ("AUC_diff", "utility_diff"):
            row["decisions"] = {"vs_0": _decide(lo, hi, 0.0, False)}
        if meta.get("clean_lift_id"):
            ref_lift = pts.get(meta["clean_lift_id"])
            if ref_lift is None or not np.isfinite(ref_lift) or \
                    ref_lift < E["utility"]["normalised_lift"]["min_clean_lift"]:
                row.update(status="FLAGGED_LOW_CLEAN_LIFT", point=None, interval=None,
                           note=f"clean lift over constant {ref_lift} < min_clean_lift; normalised lift not reported")
        if meta.get("kind") == "noise_seed_aggregate":
            row["per_seed_points"] = [pts[p] for p in reg.derived[sid][1]]
            row["seed_sd"] = float(np.std(row["per_seed_points"], ddof=1)) if len(row["per_seed_points"]) > 1 else None
        rows.append(row)

    # primary family
    P = I["primary"]
    alpha_each = P["alpha_family"] / P["family_size"]
    fam = [dict(e) for e in PRIMARY_FAMILY]
    if len(fam) != P["family_size"]:
        raise ValueError("family size mismatch")
    pboot = UnitBootstrap(first["assess_unit"], P["B"], P["seed"], I["chunk"])
    ids = [f"{e['unit']}|G1|r2" if e["statistic"] == "G1_r2" else f"{e['unit']}|rep|NL|macro_auc" for e in fam]
    present = [i for i in ids if i in reg.base]
    ppts, preps = run_registry(reg, pboot, present) if present else ({}, {})
    endpoints = []
    for e, sid in zip(fam, ids):
        rec = {**e, "stat_id": sid, "alpha_each": alpha_each, "B": P["B"], "seed": P["seed"],
               "tail_count": alpha_each * P["B"], "resolution": 1.0 / P["B"]}
        if sid not in preps:
            reason = "unit outputs missing" if e["unit"] not in U else \
                U[e["unit"]]["supported"]["sensitive"].get("ne_reason") or "statistic not available"
            rec.update(point=None, lower=None, upper=None, decision="NE", status="NE", reason=reason,
                       n_ne_replicates=None)
        else:
            iv, n_ne = _interval(preps[sid], alpha_each, 1 - alpha_each, I["quantile_method"], alpha_each)
            lo, hi = iv if iv else (None, None)
            dec = _decide(lo, hi, e["bar"], below_inclusive=(e["statistic"] == "G1_r2"))
            rec.update(point=ppts[sid], lower=lo, upper=hi, decision=dec, n_ne_replicates=n_ne,
                       status="NE" if dec == "NE" else ("UNRESOLVED" if dec == "UNRESOLVED" else "DECIDED"),
                       n_replicates_below_bar=int((preps[sid] <= e["bar"]).sum()),
                       n_replicates_above_bar=int((preps[sid] > e["bar"]).sum()))
        endpoints.append(rec)

    native = {u: d["fit_records"]["native"] for u, d in U.items()}
    support = {u: d["supported"] for u, d in U.items()}
    out = {"schema": "stored_model_eval.pilot_infer/v1", "effective_protocol_sha256": effective_hash(effp),
           "units_dir": str(units_dir), "units_found": sorted(U), "units_missing": [u for u in expected if u not in U],
           "unit_input_hashes": {u: d["hashes"] for u, d in U.items()},
           "assessment": {"n_rows": int(boot.n_rows), "n_units": int(boot.n_units),
                          "row_ids_sha256": hashlib.sha256(np.ascontiguousarray(first["assess_row_id"]).tobytes()
                                                           ).hexdigest()},
           "exploratory_settings": {"B": ex["B"], "seed": ex["seed"], "level": level, "chunk": I["chunk"],
                                    "quantile_method": I["quantile_method"], "draw": I["draw"],
                                    "sampling_unit": I["sampling_unit"], "seed_aggregation": I["seed_aggregation"]},
           "exploratory": rows, "notes": notes,
           "primary": {"family_size": P["family_size"], "alpha_family": P["alpha_family"], "alpha_each": alpha_each,
                       "B": P["B"], "seed": P["seed"], "tail_count": alpha_each * P["B"], "resolution": 1.0 / P["B"],
                       "adjustment": P["adjustment"], "validity_note": P["validity_note"],
                       "quantile_method": I["quantile_method"], "draw": I["draw"],
                       "family": [{"id": e["id"], "unit": e["unit"], "statistic": e["statistic"], "bar": e["bar"]}
                                  for e in fam],
                       "endpoints": endpoints},
           "native": native, "support": support,
           "fit_records_summary": {u: {"recipes": [{k: r.get(k) for k in ("surface", "recipe", "status",
                                                                         "selected", "reused_from")}
                                                   for r in d["fit_records"]["recipes"]],
                                       "U1": d["fit_records"].get("U1"), "reuse": d["fit_records"].get("reuse"),
                                       "closed_form": d["fit_records"].get("closed_form"),
                                       "timing": d["fit_records"].get("timing")} for u, d in U.items()},
           "effective_keys_consumed_in_infer": sorted(E.log)}
    consumed = set(E.log)
    for d in U.values():
        consumed |= set(d["fit_records"].get("effective_keys_consumed_in_unit", []))
    out["consumption"] = consumption_report(consumed, effp)
    return to_jsonable(out)


# --------------------------------------------------------------------------------------------------
# report tables
# --------------------------------------------------------------------------------------------------


def _w(path: Path, rows: list[dict], cols: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else (json.dumps(r[c]) if isinstance(r.get(c), (dict, list))
                                                        else r[c])) for c in cols})


MEANING = {("G1_r2", "ESTABLISHED_ABOVE"): "fails to generalise (held-out native quantity above tau)",
           ("G1_r2", "ESTABLISHED_BELOW"): "generalises (held-out native quantity at or below tau)",
           ("rep__NL__macro_auc", "ESTABLISHED_ABOVE"): "recovery outside scope established",
           ("rep__NL__macro_auc", "ESTABLISHED_BELOW"): "below bar established"}


def _historical(hist: dict | None, unit: str):
    if not hist:
        return None
    purpose, attr = unit.split("__")[:2]
    for r in hist.get("per_seed", {}).get("0", {}).get("rows", []):
        if r.get("purpose") == purpose and r.get("attribute") == attr:
            return r.get("r2_onehot")
    return None


def report_pilot(inf: dict, tables_dir: Path, historical_native: dict | None = None) -> dict:
    tables_dir = Path(tables_dir)
    ex = {r["id"]: r for r in inf["exploratory"]}
    native = inf["native"]

    def iv(sid):
        r = ex.get(sid)
        if not r:
            return None, None, None, "NE"
        lo, hi = r["interval"] if r["interval"] else (None, None)
        return r["point"], lo, hi, r["status"]

    # PRIMARY_ENDPOINTS
    prim = []
    for e in inf["primary"]["endpoints"]:
        n0 = native.get(e["unit"], {}).get("N0", {})
        n0cat = n0.get("category")
        if e["decision"] in ("NE", "UNRESOLVED"):
            cat = "C5"
        elif n0cat == "C1":
            cat = "C1 (native check fails as historically defined)"
        elif e["decision"] == "ESTABLISHED_ABOVE":
            cat = "C2" if e["statistic"] == "G1_r2" else "C3"
        else:
            cat = "none (established below)"
        prim.append({**e, "meaning": MEANING.get((e["statistic"], e["decision"]), e["decision"]),
                     "native_N0_category": n0cat, "category": cat, "adjusted": "Bonferroni simultaneous (16)"})
    _w(tables_dir / "PRIMARY_ENDPOINTS.csv", prim,
       ["id", "unit", "statistic", "bar", "point", "lower", "upper", "decision", "meaning", "status", "category",
        "native_N0_category", "alpha_each", "B", "seed", "tail_count", "resolution", "n_ne_replicates",
        "n_replicates_below_bar", "n_replicates_above_bar", "reason", "adjusted"])

    # DECOMPOSITION (untreated units)
    dec = []
    for u in sorted(native):
        if unit_info(u)["kind"] != "untreated":
            continue
        n0 = native[u].get("N0", {})
        hist = _historical(historical_native, u)
        hm = n0.get("historical_mixed_precision", {})
        dec.append({"unit": u, "step": "F0",
                    "quantity": "N0 historical native check: in-sample on the PCRL test split (mixed precision, "
                                "clamped)",
                    "changes": "-", "scale": "R2", "point": hm.get("clamped"),
                    "point_float64": n0.get("float64", {}).get("clamped"),
                    "point_raw_mixed": hm.get("raw"), "historical_r2_onehot": hist,
                    "abs_diff_vs_historical": (abs(hm["clamped"] - hist) if hist is not None and hm else None),
                    "category": n0.get("category"), "interval_kind": "none (historical in-sample point)"})
        for step, sid, q, ch in (
                ("F1", f"{u}|G1|r2", "G1 held-out fixed-penalty R2", "rows: in-sample -> held-out"),
                ("F2", f"{u}|G1|macro_auc", "G1 predictor scored as macro AUC", "metric: R2 -> AUC (same predictor)"),
                ("F3", f"{u}|rep|L|macro_auc", "L (logistic) macro AUC on rep", "model: least squares -> logistic"),
                ("F4", f"{u}|rep|NL|macro_auc", "NL-selected macro AUC on rep", "attacker family: linear -> NL"),
                ("F5", f"{u}|outputs|NL|macro_auc", "NL-selected macro AUC on outputs", "surface: rep -> outputs"),
                ("F6", f"{u}|rep+outputs|NL|macro_auc", "NL-selected macro AUC on rep+outputs",
                 "surface: rep -> rep+outputs")):
            pt, lo, hi, st = iv(sid)
            dsid = {"F3": "F3_minus_F2", "F4": "F4_minus_F3", "F5": "F5_minus_F4", "F6": "F6_minus_F4"}.get(step)
            dpt, dlo, dhi, _ = iv(f"{u}|decomp|{dsid}") if dsid else (None, None, None, None)
            dec.append({"unit": u, "step": step, "quantity": q, "changes": ch,
                        "scale": "R2" if step == "F1" else "AUC", "point": pt, "lower90": lo, "upper90": hi,
                        "status": st, "delta_name": dsid, "delta_point": dpt, "delta_lower90": dlo,
                        "delta_upper90": dhi, "interval_kind": "exploratory 90% percentile (unadjusted)"})
    _w(tables_dir / "DECOMPOSITION.csv", dec,
       ["unit", "step", "quantity", "changes", "scale", "point", "lower90", "upper90", "status", "delta_name",
        "delta_point", "delta_lower90", "delta_upper90", "point_float64", "point_raw_mixed",
        "historical_r2_onehot", "abs_diff_vs_historical", "category", "interval_kind"])

    # NATIVE_CHECKS: N0 (historical check; noise units 'unverified') and N1 (descriptive within-assessment)
    nat = []
    for u in sorted(native):
        n0, n1 = native[u].get("N0", {}), native[u].get("N1", {})
        hm0, f0 = n0.get("historical_mixed_precision", {}), n0.get("float64", {})
        hm1, f1 = n1.get("historical_mixed_precision", {}), n1.get("float64", {})
        nat.append({"unit": u, "N0_label": n0.get("label"), "N0_status": n0.get("status", "REPRODUCED"),
                    "N0_mixed_clamped": hm0.get("clamped"), "N0_mixed_raw": hm0.get("raw"),
                    "N0_float64_clamped": f0.get("clamped"), "N0_rows": n0.get("n_rows"),
                    "N0_category": n0.get("category"), "N0_variants_agree": n0.get("variants_agree_on_category"),
                    "historical_r2_onehot": _historical(historical_native, u) if unit_info(u)["kind"] == "untreated"
                    else None, "N1_label": n1.get("label"), "N1_mixed_raw": hm1.get("raw"),
                    "N1_float64_raw": f1.get("raw"), "N1_rows": hm1.get("n_rows")})
    _w(tables_dir / "NATIVE_CHECKS.csv", nat, ["unit", "N0_label", "N0_status", "N0_mixed_clamped", "N0_mixed_raw",
                                               "N0_float64_clamped", "N0_rows", "N0_category", "N0_variants_agree",
                                               "historical_r2_onehot", "N1_label", "N1_mixed_raw", "N1_float64_raw",
                                               "N1_rows"])

    # UTILITY
    ut = []
    note_ceiling = ("near ceiling: task label is a deterministic recoding of inputs (Addendum D1 #10)")
    for r in inf["exploratory"]:
        r = dict(r)
        if r.get("near_ceiling"):
            r["note"] = (r.get("note") + "; " if r.get("note") else "") + note_ceiling
        if r.get("quantity") in ("U1", "U2", "Uconst", "U1_lift", "U2_lift", "U1_normalised_lift",
                                 "U2_normalised_lift") and r.get("kind") != "noise_seed_aggregate":
            d = ex.get(r["id"] + "|diff_vs_untreated")
            ut.append({"unit": r["unit"], "kind": r["quantity"], "metric": r["metric"], "point": r["point"],
                       "lower90": (r["interval"] or [None, None])[0], "upper90": (r["interval"] or [None, None])[1],
                       "status": r["status"], "sigma": r.get("sigma"), "release_seed": r.get("release_seed"),
                       "reference_unit": d.get("reference_unit") if d else None,
                       "diff_point": d["point"] if d else None,
                       "diff_lower90": (d["interval"] or [None, None])[0] if d else None,
                       "diff_upper90": (d["interval"] or [None, None])[1] if d else None,
                       "seed_sd": r.get("seed_sd"), "paired_on": "identical assessment IDs",
                       "note": r.get("note")})
        elif str(r.get("quantity", "")).startswith(("U1", "U2")) and r.get("kind") == "noise_seed_aggregate":
            ut.append({"unit": r["unit"], "kind": r["quantity"], "metric": r["metric"], "point": r["point"],
                       "lower90": (r["interval"] or [None, None])[0], "upper90": (r["interval"] or [None, None])[1],
                       "status": r["status"], "sigma": r.get("sigma"), "reference_unit": r.get("reference_unit"),
                       "seed_sd": r.get("seed_sd"), "paired_on": "identical assessment IDs (seed mean within replicate)",
                       "note": r.get("note")})
    _w(tables_dir / "UTILITY.csv", ut, ["unit", "kind", "metric", "point", "lower90", "upper90", "status", "sigma",
                                        "release_seed", "reference_unit", "diff_point", "diff_lower90", "diff_upper90",
                                        "seed_sd", "paired_on", "note"])

    # SUPPORT_COVERAGE
    sc = []
    for u, s in sorted(inf["support"].items()):
        for what in ("sensitive", "task"):
            rec = s[what]
            for k in range(rec["n_classes"]):
                ks = str(k)
                sc.append({"unit": u, "what": what, "class": k,
                           **{f"n_{r}": rec["counts_per_role"][r][k] for r in rec["counts_per_role"]},
                           "supported": k in rec["supported_classes"],
                           "reason": rec["unsupported_classes"].get(ks, {}).get("detail"),
                           "unit_status": rec["status"], "ne_reason": rec["ne_reason"],
                           "class_coverage": f"{rec['coverage']['classes'][0]}/{rec['coverage']['classes'][1]}",
                           "pair_coverage": f"{rec['coverage']['pairs'][0]}/{rec['coverage']['pairs'][1]}",
                           "assessment_row_coverage": rec["coverage"]["assessment_row_fraction_in_supported_classes"]})
    for u in inf["units_missing"]:
        sc.append({"unit": u, "what": "unit", "unit_status": "MISSING_OUTPUTS"})
    _w(tables_dir / "SUPPORT_COVERAGE.csv", sc,
       ["unit", "what", "class", "n_attacker_fit", "n_attacker_val", "n_assessment", "supported", "reason",
        "unit_status", "ne_reason", "class_coverage", "pair_coverage", "assessment_row_coverage"])

    # EXPLORATORY (every row)
    exr = [{**r, "lower90": (r["interval"] or [None, None])[0], "upper90": (r["interval"] or [None, None])[1]}
           for r in inf["exploratory"]]
    _w(tables_dir / "EXPLORATORY_ENDPOINTS.csv", exr,
       ["id", "unit", "kind", "sigma", "release_seed", "quantity", "surface", "recipe", "metric", "scale", "point",
        "lower90", "upper90", "decisions", "status", "n_ne_replicates", "seed_sd", "per_seed_points", "boot_B",
        "boot_seed", "label",
        "reference_unit", "note"])
    (tables_dir / "PRIMARY_FAMILY.json").write_text(json.dumps(
        {k: inf["primary"][k] for k in ("family", "family_size", "alpha_family", "alpha_each", "B", "seed",
                                         "tail_count", "resolution", "adjustment", "validity_note",
                                         "quantile_method", "draw")}, indent=1))
    return {"tables_dir": str(tables_dir), "n_primary": len(prim), "n_decomposition_rows": len(dec),
            "n_utility_rows": len(ut), "n_support_rows": len(sc), "n_exploratory_rows": len(exr),
            "files": ["PRIMARY_ENDPOINTS.csv", "DECOMPOSITION.csv", "UTILITY.csv", "SUPPORT_COVERAGE.csv",
                      "NATIVE_CHECKS.csv", "EXPLORATORY_ENDPOINTS.csv", "PRIMARY_FAMILY.json"]}
