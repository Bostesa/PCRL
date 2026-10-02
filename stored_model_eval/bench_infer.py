"""Benchmark inference: consumes ONLY saved unit outputs (preds.npz, supported.json, fit_records.json) and
SIGMA_STAR.json. Nothing is refitted or selected here.

Every statistic is a function of row weights over the assessment rows of one dataset. One cluster bootstrap draw per
dataset (sampling unit = assess_unit; numpy default_rng(seed).multinomial, pilot convention) re-weights every unit,
arm, encoder seed, release seed and attacker seed of that dataset (each cell regenerates the identical stream).

Replicate summary (frozen): an endpoint is the MEAN over encoder seeds x release seeds x attacker seeds of the
per-unit statistic, computed WITHIN each bootstrap replicate on the same resampled people. Seeds are never units.

Worst-class / worst-pair: simultaneous per-class/pair bounds at alpha/K, LCB(max) = max_k LCB_k(alpha/K),
UCB(max) = max_k UCB_k(alpha/K), point = max_k point_k (K = supported classes or pairs).

Exploratory: two-sided 90% percentile intervals (B = 2000, seed 20261003, numpy 'linear'). Primary family: 24
Bonferroni simultaneous one-sided bounds at 0.05/24 (B = 20000, seed 20261004).
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .admission import sha256_file
from .bench_effective import (BENCH_EFFECTIVE, Tracked, all_registered, bench_effective_hash, consumption_report,
                              parse_unit, primary_family, tier1_units, tier2_units)
from .metrics import to_jsonable
from .pilot_infer import (UnitBootstrap, _auc_eval, _auc_prep, prob_skill_stats, r2_stat, rho_stat, softmax,
                          utility_stats)

ATT_SEEDS = (0, 1, 2)
# (surface, recipe) -> saved key pattern(s); "{k}" = attacker seed
RECIPES = {
    ("rep", "NL"): ["P__rep__NL__as{k}"], ("rep", "L"): ["P__rep__L__as{k}"],
    ("rep", "GBT"): ["P__rep__GBT"], ("rep", "MLP"): ["P__rep__MLP"],
    ("rep+outputs", "NL"): ["P__repPLUSoutputs__NL__as{k}"], ("rep+outputs", "L"): ["P__repPLUSoutputs__Lslate__as{k}"],
    ("rep+outputs", "Lconcat"): ["P__repPLUSoutputs__L"], ("rep+outputs", "GBT"): ["P__repPLUSoutputs__GBT"],
    ("rep+outputs", "MLP"): ["P__repPLUSoutputs__MLP"], ("rep+outputs", "ignore_rep"): ["P__repPLUSoutputs__ignore_rep"],
    ("rep+outputs", "ignore_out"): ["P__repPLUSoutputs__ignore_out"],
    ("outputs", "NL"): ["P__outputs__NL__as{k}"], ("outputs", "L"): ["P__outputs__L__as{k}"],
    ("rep", "LRT_A2"): ["P__rep__LRT_A2"], ("rep", "LRT_A4"): ["P__rep__LRT_A4"],
    ("label_only", "LO"): ["LO_P"],
}
WORST = {("rep", "NL"), ("rep", "L"), ("rep+outputs", "NL"), ("rep+outputs", "L"), ("outputs", "NL"), ("outputs", "L"),
         ("rep", "LRT_A2"), ("rep", "LRT_A4"), ("label_only", "LO")}
CELL_LEVEL = {"outputs", "label_only", "clean_output", "constant", "Uconst"}   # identical across arms: reported once per cell
DIFF_STATS = [("rep", "NL", "macro_auc"), ("rep", "L", "macro_auc"), ("rep+outputs", "NL", "macro_auc"),
              ("rep+outputs", "L", "macro_auc"), ("rep", "G1", "r2"), ("rep", "G2", "r2"), ("U2", "U2", "accuracy"),
              ("U2", "U2", "log_loss"), ("U1", "U1", "accuracy")]


def _h(a) -> str:
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


# --------------------------------------------------------------------------------------------------
# statistic graph: bases (functions of weights) and aggregates (mean / diff of other ids)
# --------------------------------------------------------------------------------------------------


class Graph:
    def __init__(self):
        self.base, self.agg, self.order, self.meta = {}, {}, [], {}
        self._by_hash = {}

    def base_once(self, key: str, sid: str, fn) -> str:
        """Register a base statistic once per content key (identical arrays -> one computation)."""
        if key in self._by_hash:
            return self._by_hash[key]
        self.base[sid] = fn
        self._by_hash[key] = sid
        return sid

    def add(self, sid: str, op: str, parts: list, **meta):
        if sid in self.agg or sid in self.base:
            return sid
        self.agg[sid] = (op, list(parts))
        self.order.append(sid)
        if meta:
            self.meta[sid] = meta
        return sid

    def closure(self, ids) -> tuple[list, list]:
        need = set()
        stack = list(ids)
        while stack:
            s = stack.pop()
            if s in need:
                continue
            need.add(s)
            if s in self.agg:
                stack += self.agg[s][1]
        return [s for s in self.base if s in need], [s for s in self.order if s in need]

    def evaluate(self, WT, bases, aggs) -> dict:
        v = {s: np.asarray(self.base[s](WT), dtype=np.float64) for s in bases}
        for s in aggs:
            op, parts = self.agg[s]
            X = [v[p] for p in parts]
            if op == "mean":
                v[s] = np.mean(np.stack(X), axis=0)
            elif op == "diff":
                v[s] = X[0] - X[1]
            else:
                raise ValueError(op)
        return v


def run(graph: Graph, boot: UnitBootstrap, ids) -> tuple[dict, dict]:
    bases, aggs = graph.closure(ids)
    ones = np.ones((boot.n_rows, 1))
    pv = graph.evaluate(ones, bases, aggs)
    points = {s: float(pv[s][0]) for s in ids}
    reps = {s: [] for s in ids}
    for WT in boot.chunks():
        v = graph.evaluate(WT, bases, aggs)
        for s in ids:
            reps[s].append(v[s])
    return points, {s: np.concatenate(r) for s, r in reps.items()}


def points_only(graph: Graph, n_rows: int, ids) -> dict:
    bases, aggs = graph.closure(ids)
    pv = graph.evaluate(np.ones((n_rows, 1)), bases, aggs)
    return {s: float(pv[s][0]) for s in ids}


def _q(r, lo, hi, method, max_ne):
    ok = np.isfinite(r)
    n_ne = int((~ok).sum())
    if ok.sum() == 0 or n_ne / len(r) > max_ne:
        return None, n_ne
    v = r[ok]
    return (float(np.quantile(v, lo, method=method)), float(np.quantile(v, hi, method=method))), n_ne


def _decide(lo, hi, bar):
    if lo is None:
        return "NE"
    if lo > bar:
        return "ESTABLISHED_ABOVE"
    if hi < bar:
        return "ESTABLISHED_BELOW"
    return "UNRESOLVED"


def _ni(lo, hi, margin):
    if lo is None:
        return "NE"
    if lo >= margin:
        return "NONINFERIOR"
    if hi < margin:
        return "INFERIOR"
    return "UNRESOLVED"


# --------------------------------------------------------------------------------------------------
# weighted factories not in pilot_infer
# --------------------------------------------------------------------------------------------------


def class_auc(y, P, k):
    prep = _auc_prep(np.asarray(P, float)[:, k], np.asarray(y) == k, np.arange(len(y)), len(y))
    return lambda WT: _auc_eval(prep, WT)


def pair_auc(y, P, i, j):
    y = np.asarray(y).astype(int)
    P = np.asarray(P, float)
    n = len(y)
    rows = np.flatnonzero((y == i) | (y == j))
    den = P[rows, i] + P[rows, j]
    sc = np.full(n, 0.5)
    sc[rows] = np.where(den > 0, P[rows, j] / np.where(den > 0, den, 1.0), 0.5)
    prep = _auc_prep(sc, y == j, rows, n)
    return lambda WT: _auc_eval(prep, WT)


def crosscov_stat(HX, Z, what: str):
    """Weighted cross-covariance of released h with a concept one-hot on assessment rows: Frobenius or max-abs."""
    HX, Z = np.asarray(HX, float), np.asarray(Z, float)
    prods = [HX * Z[:, j:j + 1] for j in range(Z.shape[1])]

    def f(WT):
        s0 = WT.sum(0)
        mh = HX.T @ WT / s0                      # d x c
        mz = Z.T @ WT / s0                       # q x c
        C = np.stack([(Pj.T @ WT) / s0 - mh * mz[j] for j, Pj in enumerate(prods)], axis=1)  # d x q x c
        return np.sqrt((C ** 2).sum((0, 1))) if what == "fro" else np.abs(C).max((0, 1))
    return f


# --------------------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------------------


def load_unit(units_dir: Path, uid: str) -> dict | None:
    from .bench import verify_unit
    d = Path(units_dir) / uid
    if not d.exists():
        return None
    ok, errs = verify_unit(d)
    if not ok:
        raise ValueError(f"unit {uid} outputs incomplete or modified: {errs}")
    fr = json.loads((d / "fit_records.json").read_text())
    with np.load(d / "preds.npz", allow_pickle=False) as z:
        preds = {k: z[k] for k in z.files}
    return {"preds": preds, "fit_records": fr, "supported": json.loads((d / "supported.json").read_text()),
            "hashes": {f: sha256_file(d / f) for f in ("preds.npz", "supported.json", "fit_records.json")}}


# --------------------------------------------------------------------------------------------------
# per-cell graph
# --------------------------------------------------------------------------------------------------


def build_cell(cell: str, members: dict, U: dict, E, sigma_star: float | None) -> tuple[Graph, dict]:
    """members: arm group -> [unit ids] (arm group A, B, C, D_sigma<s>). Returns graph and registry metadata."""
    g = Graph()
    meta = {"rows": {}, "worst": {}, "unit_keys": defaultdict(dict), "groups": {}, "notes": []}
    clip = E["recovery_metrics"]["prob_clip"]
    RM = E["recovery_metrics"]
    if RM["auc"] != "macro_ovr_supported":
        raise ValueError("unsupported AUC convention")
    UT = E["utility"]
    ma, ms = list(UT["metrics_all_rows"]), list(UT["metrics_supported_task_classes"])
    any_u = next(iter(U[u] for us in members.values() for u in us))
    sup = any_u["supported"]["sensitive"]
    classes, pairs = sup["supported_classes"], [tuple(p) for p in sup["supported_pairs"]]
    tcls = any_u["supported"]["task"]["supported_classes"]
    estimable = sup["status"] != "NE"

    def reg_row(sid, **m):
        meta["rows"][sid] = m

    def unit_stats(u, p):
        """Register per-unit stats; returns {(surf, rec, metric): unit-level id} and per-class/pair ids."""
        y, t = p["y_s"], p["y_task"]
        out, pcl = {}, {}
        if estimable and "G1_pred" in p:
            for nm in ("G1", "G2"):
                out[("rep", nm, "r2")] = g.base_once(f"r2|{_h(p[nm + '_pred'])}", f"{u}|rep|{nm}|r2",
                                                     r2_stat(y, p[nm + "_pred"], p[nm + "_prior"]))
            if "RHO_u" in p:
                out[("rep", "RHO1", "rho1sq")] = g.base_once(f"rho|{_h(p['RHO_u'])}", f"{u}|rep|RHO1|rho1sq",
                                                             rho_stat(p["RHO_u"], p["RHO_v"]))
            if "HX" in p:
                Zt = np.eye(int(any_u["supported"]["sensitive"]["n_classes"]))[y.astype(int)]
                for cname, Z in (("target", Zt), ("policy", p["ZP"])):
                    for w in ("fro", "max_abs"):
                        out[("rep", f"crosscov_{cname}", w)] = g.base_once(
                            f"cc|{w}|{cname}|{_h(p['HX'])}|{_h(Z)}", f"{u}|rep|crosscov_{cname}|{w}",
                            crosscov_stat(p["HX"], Z, w))
            for (surf, rec), pats in RECIPES.items():
                keys = [pt.format(k=k) for pt in pats for k in (ATT_SEEDS if "{k}" in pt else [None])]
                keys = [k for k in keys if k in p]
                if not keys:
                    continue
                per_key_macro, per_key_cls, per_key_pair, per_key_sk = [], defaultdict(list), defaultdict(list), \
                    defaultdict(list)
                for key in keys:
                    P = p[key]
                    hP = _h(P) + "|" + _h(y)
                    cl = [g.base_once(f"cls{k}|{hP}", f"{u}|{key}|cls{k}", class_auc(y, P, k)) for k in classes]
                    mk = g.add(f"{u}|{key}|macro_auc", "mean", cl)
                    per_key_macro.append(mk)
                    meta["unit_keys"][u][key] = mk
                    for k, c in zip(classes, cl):
                        per_key_cls[k].append(c)
                    if (surf, rec) in WORST:
                        for i, j in pairs:
                            per_key_pair[(i, j)].append(g.base_once(f"pair{i}-{j}|{hP}", f"{u}|{key}|pair{i}-{j}",
                                                                    pair_auc(y, P, i, j)))
                    prior = p["s_prior_fit"]
                    for m, fn in prob_skill_stats(y, P, prior, clip).items():
                        per_key_sk[m].append(g.base_once(f"{m}|{hP}|{_h(prior)}", f"{u}|{key}|{m}", fn))
                out[(surf, rec, "macro_auc")] = g.add(f"{u}|{surf}|{rec}|macro_auc", "mean", per_key_macro,
                                                      keys=keys)
                for m, ids in per_key_sk.items():
                    out[(surf, rec, m)] = g.add(f"{u}|{surf}|{rec}|{m}", "mean", ids)
                if (surf, rec) in WORST:
                    for k, ids in per_key_cls.items():
                        pcl[(surf, rec, f"cls{k}")] = g.add(f"{u}|{surf}|{rec}|cls{k}", "mean", ids)
                    for (i, j), ids in per_key_pair.items():
                        pcl[(surf, rec, f"pair{i}-{j}")] = g.add(f"{u}|{surf}|{rec}|pair{i}-{j}", "mean", ids)
            Pc = np.tile(np.asarray(p["s_prior_fit"], float), (len(y), 1))
            hc = _h(Pc) + "|" + _h(y)
            ccl = [g.base_once(f"cls{k}|{hc}", f"{u}|constant|cls{k}", class_auc(y, Pc, k)) for k in classes]
            out[("constant", "prior", "macro_auc")] = g.add(f"{u}|constant|prior|macro_auc", "mean", ccl)
            for m, fn in prob_skill_stats(y, Pc, p["s_prior_fit"], clip).items():
                out[("constant", "prior", m)] = g.base_once(f"{m}|{hc}", f"{u}|constant|prior|{m}", fn)
            G1 = p["G1_pred"]
            cl = [g.base_once(f"cls{k}|{_h(G1)}|{_h(y)}", f"{u}|G1pred|cls{k}", class_auc(y, G1, k)) for k in classes]
            out[("rep", "G1_scores", "macro_auc")] = g.add(f"{u}|rep|G1_scores|macro_auc", "mean", cl)
        # utility (independent of sensitive support)
        Ps = {"U2": p["U2_P"], "U1": softmax(p["U1_logits"]) if p["U1_logits"].size else None,
              "clean_output": softmax(p["OUT_logits"])}
        for q, PP in Ps.items():
            if PP is None:
                continue
            for m, fn in utility_stats(t, PP, tcls, clip, ma, ms).items():
                out[(q, q, m)] = g.base_once(f"{q}|{m}|{_h(PP)}|{_h(t)}", f"{u}|{q}|{q}|{m}", fn)
        const = int(np.argmax(p["t_prior_fit"]))
        out[("Uconst", "Uconst", "accuracy")] = g.base_once(
            f"uconst|{_h(t)}|{const}", f"{u}|Uconst|Uconst|accuracy",
            (lambda tt, cc: (lambda WT: ((tt == cc).astype(float) @ WT) / WT.sum(0)))(t, const))
        return out, pcl

    per_unit = {}
    for arm, us in members.items():
        for u in us:
            per_unit[u] = unit_stats(u, U[u]["preds"])

    def group(arm, us, label):
        stats = defaultdict(list)
        for u in us:
            o, pcl = per_unit[u]
            for k, sid in list(o.items()) + list(pcl.items()):
                stats[k].append(sid)
        ids = {}
        for (surf, rec, m), sids in stats.items():
            if (surf in CELL_LEVEL) != (label == "CELL"):
                continue
            if len(sids) != len(us):
                continue
            ids[(surf, rec, m)] = g.add(f"{cell}|{label}|{surf}|{rec}|{m}", "mean", sids, n_members=len(us))
        meta["groups"][label] = {"members": list(us), "ids": {"|".join(k): v for k, v in ids.items()}}
        return ids

    gid = {}
    for arm, us in members.items():
        gid[arm] = group(arm, us, arm)
    ref = members.get("A") or next(iter(members.values()))
    gid["CELL"] = group("CELL", ref, "CELL")
    # paired differences vs A on identical people
    diffs = {}
    if "A" in gid:
        for arm in members:
            if arm == "A":
                continue
            for k in DIFF_STATS:
                if k in gid[arm] and k in gid["A"]:
                    diffs[(arm, k)] = g.add(f"{cell}|{arm}-A|{'|'.join(k)}", "diff", [gid[arm][k], gid["A"][k]],
                                            kind="paired_diff_vs_A")
    # output references: outputs NL - LO; rep+outputs NL - outputs NL per arm
    cellg = gid["CELL"]
    if ("outputs", "NL", "macro_auc") in cellg and ("label_only", "LO", "macro_auc") in cellg:
        g.add(f"{cell}|CELL|outputs_NL_minus_LO|macro_auc", "diff",
              [cellg[("outputs", "NL", "macro_auc")], cellg[("label_only", "LO", "macro_auc")]])
    for arm in members:
        if ("rep+outputs", "NL", "macro_auc") in gid[arm] and ("outputs", "NL", "macro_auc") in cellg:
            g.add(f"{cell}|{arm}|plus_NL_minus_outputs_NL|macro_auc", "diff",
                  [gid[arm][("rep+outputs", "NL", "macro_auc")], cellg[("outputs", "NL", "macro_auc")]])
    meta.update(gid=gid, diffs=diffs, per_unit=per_unit, classes=classes, pairs=pairs, estimable=estimable,
                task_classes=tcls)
    return g, meta


# --------------------------------------------------------------------------------------------------
# main entry
# --------------------------------------------------------------------------------------------------


def infer_bench(root: Path, eff: dict | None = None, tier: str | None = None, unit_ids=None) -> dict:
    effp = BENCH_EFFECTIVE if eff is None else eff
    E = Tracked(effp)
    root = Path(root)
    ss_path = root / "infer" / "SIGMA_STAR.json"
    if not ss_path.exists():
        raise ValueError(f"{ss_path} missing: sigma* is selected on attacker_val before assessment scoring")
    SS = json.loads(ss_path.read_text())
    if tier in (None, "1"):
        expected = tier1_units()
    elif tier == "all":
        expected = all_registered()
    elif tier == "2":
        t2 = tier2_units()
        expected = tier1_units() + t2["E1"] + t2["E2"] + t2["E3"]
    else:
        raise ValueError(tier)
    if unit_ids is not None:
        expected = [u for u in expected if u in set(unit_ids)]
    U, missing, aliases, ne_units = {}, [], {}, {}
    for u in expected:
        d = load_unit(root / "units", u)
        if d is None:
            missing.append(u)
            continue
        if d["fit_records"]["effective_protocol_sha256"] != bench_effective_hash(effp):
            raise ValueError(f"unit {u} was produced under a different effective protocol")
        if d["fit_records"].get("status") == "NE":
            ne_units[u] = d["fit_records"].get("ne_reason")
            continue
        U[u] = d
    for u, d in list(U.items()):
        src = d["fit_records"].get("alias_of")
        if src:
            if src not in U:
                raise ValueError(f"alias {u} -> {src}: source outputs missing")
            aliases[u] = src
            U[u] = {**U[src], "alias_of": src, "hashes": d["hashes"]}
    I = E["inference"]
    if I["sampling_unit"] != "assess_unit" or I["quantile_method"] != "linear":
        raise ValueError("sampling unit / quantile rule changed")
    ex, PR = I["exploratory"], I["primary"]
    level = ex["level"]
    a2 = (1 - level) / 2
    bars, tau, margin = list(I["bars"]), I["tau"], I["ni_margin"]
    cells = defaultdict(lambda: defaultdict(list))
    for u in U:
        i = parse_unit(u)
        grp = i["arm"] if i["arm"] != "D" else f"D_sigma{i['sigma']:g}"
        cells[i["cell"]][grp].append(u)
    by_ds = defaultdict(list)
    for c in cells:
        by_ds[c.split("__")[0]].append(c)
    for ds, cs in by_ds.items():
        first = None
        for c in cs:
            for us in cells[c].values():
                for u in us:
                    p = U[u]["preds"]
                    if first is None:
                        first = (p["assess_row_id"], p["assess_unit"])
                    elif not (np.array_equal(p["assess_row_id"], first[0]) and
                              np.array_equal(p.get("assess_unit", first[1]), first[1])):
                        raise ValueError(f"{u}: assessment ids differ within {ds} (pairing impossible)")
    exploratory, worst, seedvar, primary_ids = [], [], [], {}
    fam = [dict(e) for e in primary_family()]
    if len(fam) != PR["family_size"]:
        raise ValueError("family size mismatch")
    alpha_each = PR["alpha_family"] / PR["family_size"]
    sig_star = {d: v["sigma_star"] for d, v in SS["datasets"].items()}
    cell_graphs = {}
    for cell in sorted(cells):
        ds = cell.split("__")[0]
        members = {k: sorted(v) for k, v in cells[cell].items()}
        g, meta = build_cell(cell, members, U, E, sig_star.get(ds))
        anyu = next(iter(U[u] for us in members.values() for u in us))
        boot = UnitBootstrap(anyu["preds"]["assess_unit"], ex["B"], ex["seed"], I["chunk"])
        # ids reported with intervals: group-level, paired diffs, per-unit headline stats
        report_ids = [sid for grp in meta["gid"].values() for sid in grp.values()]
        report_ids += list(meta["diffs"].values())
        report_ids += [s for s in g.order if s.endswith(("outputs_NL_minus_LO|macro_auc",
                                                         "plus_NL_minus_outputs_NL|macro_auc"))]
        unit_ids_rep = []
        for u, (o, _) in meta["per_unit"].items():
            for k in (("rep", "NL", "macro_auc"), ("rep", "L", "macro_auc"), ("rep+outputs", "NL", "macro_auc"),
                      ("rep+outputs", "L", "macro_auc"), ("rep", "G1", "r2"), ("rep", "G2", "r2"),
                      ("rep", "RHO1", "rho1sq"), ("U2", "U2", "accuracy"), ("U1", "U1", "accuracy"),
                      ("outputs", "NL", "macro_auc"), ("rep", "LRT_A2", "macro_auc"), ("rep", "LRT_A4", "macro_auc"),
                      ("rep", "crosscov_target", "fro"), ("rep", "crosscov_policy", "fro")):
                if k in o:
                    unit_ids_rep.append((u, k, o[k]))
        report_ids += [sid for _, _, sid in unit_ids_rep]
        report_ids = list(dict.fromkeys(report_ids))
        pts, reps = run(g, boot, report_ids) if report_ids else ({}, {})
        inv = {}
        for grp_label, grp in meta["gid"].items():
            for k, sid in grp.items():
                inv[sid] = {"level": "group", "arm_group": grp_label, "surface": k[0], "recipe": k[1], "metric": k[2],
                            "members": meta["groups"][grp_label]["members"]}
        for (arm, k), sid in meta["diffs"].items():
            inv[sid] = {"level": "paired_diff_vs_A", "arm_group": f"{arm}-A", "surface": k[0], "recipe": k[1],
                        "metric": k[2]}
        for u, k, sid in unit_ids_rep:
            inv.setdefault(sid, {"level": "unit", "unit": u, "arm_group": parse_unit(u)["arm_tag"], "surface": k[0],
                                 "recipe": k[1], "metric": k[2]})
        for sid in report_ids:
            m = inv.get(sid, {"level": "reference", "metric": sid.split("|")[-1], "surface": sid.split("|")[-2],
                              "arm_group": sid.split("|")[1]})
            iv, n_ne = _q(reps[sid], a2, 1 - a2, I["quantile_method"], a2)
            lo, hi = iv if iv else (None, None)
            pt = pts[sid]
            met = m.get("metric", "")
            row = {"id": sid, "cell": cell, "dataset": ds, **{k: v for k, v in m.items() if k != "members"},
                   "n_members": len(m.get("members", [])) or None, "point": pt if np.isfinite(pt) else None,
                   "lower90": lo, "upper90": hi, "n_ne_replicates": n_ne, "boot_B": ex["B"], "boot_seed": ex["seed"],
                   "status": "ESTIMATED" if (iv and np.isfinite(pt)) else "NE", "label": "exploratory (unadjusted)"}
            if "auc" in met and "diff" not in m.get("level", "") and "minus" not in sid:
                row["decisions"] = {f"{b:.2f}": _decide(lo, hi, b) for b in bars}
            elif met == "r2" and m.get("level") != "paired_diff_vs_A":
                row["decisions"] = {f"tau={tau}": _decide(lo, hi, tau)}
            elif m.get("level") == "paired_diff_vs_A":
                row["decisions"] = {"vs_0": _decide(lo, hi, 0.0)}
                if met == "accuracy" and m.get("surface") == "U2":
                    row["decisions"]["NI_margin_-0.01_exploratory"] = _ni(lo, hi, margin)
            exploratory.append(row)
        # worst-class / worst-pair simultaneous bounds (group level, exploratory alpha = 1 - level)
        alpha = 1 - level
        wids = []
        for grp_label, grp in meta["gid"].items():
            for (surf, rec) in WORST:
                for kind in ("cls", "pair"):
                    ks = [(k, sid) for k, sid in grp.items() if k[0] == surf and k[1] == rec and k[2].startswith(kind)]
                    if ks:
                        wids.append((grp_label, surf, rec, kind, ks))
        allw = [sid for *_, ks in wids for _, sid in ks]
        if allw:
            wp, wr = run(g, boot, list(dict.fromkeys(allw)))
            for grp_label, surf, rec, kind, ks in wids:
                K = len(ks)
                lcb, ucb, ptk, ne = [], [], [], 0
                for k, sid in ks:
                    iv, n_ne = _q(wr[sid], alpha / (2 * K), 1 - alpha / (2 * K), I["quantile_method"], alpha / (2 * K))
                    ne += n_ne
                    if iv is None:
                        lcb = None
                        break
                    lcb.append(iv[0])
                    ucb.append(iv[1])
                    ptk.append(wp[sid])
                row = {"cell": cell, "dataset": ds, "arm_group": grp_label, "surface": surf, "recipe": rec,
                       "statistic": "worst_class_auc" if kind == "cls" else "worst_pair_auc", "K": K,
                       "components": [k[2] for k, _ in ks], "boot_B": ex["B"], "boot_seed": ex["seed"],
                       "rule": E["inference"]["worst_rule"], "label": "exploratory (simultaneous over K, level 0.90)"}
                if lcb is None:
                    row.update(point=None, lower=None, upper=None, status="NE")
                else:
                    am = int(np.argmax(ptk))
                    row.update(point=float(max(ptk)), argmax=ks[am][0][2], lower=float(max(lcb)),
                               upper=float(max(ucb)), status="ESTIMATED", per_component_point=ptk,
                               per_component_lower=lcb, per_component_upper=ucb,
                               decisions={f"{b:.2f}": _decide(max(lcb), max(ucb), b) for b in bars})
                worst.append(row)
        # seed variation from point estimates (no bootstrap)
        key_pts_ids = [mk for u in meta["unit_keys"] for mk in meta["unit_keys"][u].values()]
        kp = points_only(g, boot.n_rows, list(dict.fromkeys(key_pts_ids))) if key_pts_ids else {}
        for grp_label, us in members.items():
            for (surf, rec), pats in RECIPES.items():
                if surf in CELL_LEVEL:
                    continue
                per = {}
                for u in us:
                    i = parse_unit(u)
                    vals = [kp[meta["unit_keys"][u][pt.format(k=k)]] for pt in pats
                            for k in (ATT_SEEDS if "{k}" in pt else [None])
                            if pt.format(k=k) in meta["unit_keys"][u]]
                    if vals:
                        per[(i["seed"], i.get("release_seed"))] = vals
                if not per:
                    continue
                enc = defaultdict(list)
                for (k, rs), v in per.items():
                    enc[k].append(float(np.mean(v)))
                att_sd = [float(np.std(v, ddof=1)) for v in per.values() if len(v) > 1]
                rel_sd = [float(np.std(v, ddof=1)) for v in enc.values() if len(v) > 1]
                enc_means = [float(np.mean(v)) for v in enc.values()]
                seedvar.append({"cell": cell, "dataset": ds, "arm_group": grp_label, "surface": surf, "recipe": rec,
                                "metric": "macro_auc", "n_encoder_seeds": len(enc),
                                "n_release_seeds": max(len(v) for v in enc.values()),
                                "n_attacker_seeds": max(len(v) for v in per.values()),
                                "encoder_seed_sd": float(np.std(enc_means, ddof=1)) if len(enc_means) > 1 else None,
                                "release_seed_sd_mean": float(np.mean(rel_sd)) if rel_sd else None,
                                "attacker_seed_sd_mean": float(np.mean(att_sd)) if att_sd else None,
                                "attacker_seed_sd_max": float(np.max(att_sd)) if att_sd else None,
                                "per_encoder_seed_point": {str(k): float(np.mean(v)) for k, v in enc.items()},
                                "per_unit_attacker_seed_points": {f"s{k}_rs{rs}": v for (k, rs), v in per.items()},
                                "definition": "attacker-seed SD: SD over attacker seeds within (encoder, release) seed, "
                                              "averaged; release-seed SD: SD over release seeds of attacker-seed means "
                                              "within encoder seed, averaged; encoder-seed SD: SD over encoder seeds "
                                              "of the (release x attacker)-seed means. Point estimates; not sampling "
                                              "intervals."})
        # primary endpoints in this cell
        ss = sig_star.get(ds)
        armmap = {"A": "A", "B": "B", "C": "C", "Dstar": f"D_sigma{ss:g}" if ss is not None else None}
        for e in fam:
            if e["cell"] != cell:
                continue
            arm = armmap[e["arm"]]
            grp = meta["gid"].get(arm, {}) if arm else {}
            if e["statistic"] == "G1_r2":
                sid = grp.get(("rep", "G1", "r2"))
            elif e["statistic"] == "C_rep_NL_macro_auc":
                sid = grp.get(("rep", "NL", "macro_auc"))
            elif e["statistic"] == "C_rep_plus_clean_out_NL_macro_auc":
                sid = grp.get(("rep+outputs", "NL", "macro_auc"))
            else:
                sid = meta["diffs"].get((arm, ("U2", "U2", "accuracy")))
            primary_ids[e["id"]] = {"sid": sid, "cell": cell, "arm_group": arm,
                                    "alias_of": (aliases.get(meta["groups"].get(arm, {}).get("members", [None])[0])
                                                 if arm == "C" else None),
                                    "members": meta["groups"].get(arm, {}).get("members"),
                                    "estimable": meta["estimable"]}
        cell_graphs[cell] = (g, anyu["preds"]["assess_unit"])
    # primary bootstrap (B = 20000): one run per cell over that cell's endpoint statistics (dataset stream)
    per_cell = defaultdict(list)
    for e in fam:
        pi = primary_ids.get(e["id"])
        if pi and pi.get("sid"):
            per_cell[pi["cell"]].append(pi["sid"])
    pres = {}
    for cell, sids in per_cell.items():
        g, au = cell_graphs[cell]
        pboot = UnitBootstrap(au, PR["B"], PR["seed"], I["chunk"])
        p_, r_ = run(g, pboot, list(dict.fromkeys(sids)))
        for sid in sids:
            pres[sid] = (p_[sid], r_[sid])
    endpoints = []
    for e in fam:
        pi = primary_ids.get(e["id"], {"sid": None})
        rec = {**e, "alpha_each": alpha_each, "B": PR["B"], "seed": PR["seed"], "tail_count": alpha_each * PR["B"],
               "resolution": 1.0 / PR["B"], "arm_group": pi.get("arm_group"), "members": pi.get("members"),
               "alias_of": pi.get("alias_of")}
        if e["arm"] == "Dstar":
            rec["sigma_star"] = sig_star.get(e["dataset"])
            rec["sigma_star_flagged"] = SS["datasets"].get(e["dataset"], {}).get("flagged_fallback")
        if pi.get("sid") is None:
            reason = "cell not estimable (support)" if pi.get("estimable") is False else "unit outputs missing or NE"
            rec.update(point=None, lower=None, upper=None, decision="NE", status="NE", reason=reason)
            endpoints.append(rec)
            continue
        pt, r = pres[pi["sid"]]
        iv, n_ne = _q(r, alpha_each, 1 - alpha_each, I["quantile_method"], alpha_each)
        lo, hi = iv if iv else (None, None)
        dec = _ni(lo, hi, margin) if e["statistic"] == "U2_accuracy_diff_vs_A" else _decide(lo, hi, e["bar"])
        rec.update(stat_id=pi["sid"], point=pt if np.isfinite(pt) else None, lower=lo, upper=hi, decision=dec,
                   n_ne_replicates=n_ne,
                   status="NE" if dec == "NE" else ("UNRESOLVED" if dec == "UNRESOLVED" else "DECIDED"),
                   n_replicates_below_bar=int((r < e["bar"]).sum()), n_replicates_above_bar=int((r > e["bar"]).sum()))
        if pi.get("alias_of"):
            rec["note"] = f"alias row: C map equals B map (scored once as {pi['alias_of']})"
        endpoints.append(rec)
    native = {u: d["fit_records"].get("native") for u, d in U.items() if not d.get("alias_of")}
    support = {}
    for u, d in U.items():
        support.setdefault(parse_unit(u)["cell"], d["supported"])
    metric_definitions = {k: E["recovery_metrics"][k] for k in ("auc", "LLR_nats", "LL_skill", "brier_skill",
                                                                 "worst_class", "worst_pair", "prob_clip")}
    out = {"schema": "stored_model_eval.bench_infer/v1", "effective_protocol_sha256": bench_effective_hash(effp),
           "metric_definitions": metric_definitions,
           "sigma_star": {"path": str(ss_path), "sha256": sha256_file(ss_path), "values": sig_star,
                          "datasets": SS["datasets"]},
           "units_expected": len(expected), "units_found": sorted(U), "units_missing": missing,
           "aliases": aliases, "ne_units": ne_units,
           "unit_input_hashes": {u: d["hashes"] for u, d in U.items()},
           "exploratory_settings": {"B": ex["B"], "seed": ex["seed"], "level": level, "chunk": I["chunk"],
                                    "quantile_method": I["quantile_method"], "draw": I["draw"],
                                    "replicate_summary": I["replicate_summary"]},
           "exploratory": exploratory, "worst": worst, "seed_variation": seedvar,
           "primary": {"family_size": PR["family_size"], "alpha_family": PR["alpha_family"], "alpha_each": alpha_each,
                       "B": PR["B"], "seed": PR["seed"], "tail_count": alpha_each * PR["B"],
                       "adjustment": PR["adjustment"], "validity_note": PR["validity_note"],
                       "family": fam, "endpoints": endpoints},
           "native": native, "support": support,
           "fit_records_summary": {u: {"release": d["fit_records"].get("release"), "U1": d["fit_records"].get("U1"),
                                       "timing": d["fit_records"].get("timing"),
                                       "n_model_fits": d["fit_records"].get("n_model_fits"),
                                       "shared": d["fit_records"].get("shared"),
                                       "recipes": [{k: r.get(k) for k in ("surface", "recipe", "status", "selected",
                                                                          "plus_selection", "seed_retrain")}
                                                   for r in d["fit_records"].get("recipes", [])]}
                                   for u, d in U.items() if not d.get("alias_of")},
           "effective_keys_consumed_in_infer": sorted(E.log)}
    consumed = set(E.log)
    for d in U.values():
        consumed |= set(d["fit_records"].get("effective_keys_consumed_in_unit", []))
    out["consumption"] = consumption_report(consumed, effp)
    return to_jsonable(out)


__all__ = ["infer_bench", "build_cell", "Graph", "run", "crosscov_stat", "class_auc", "pair_auc"]
