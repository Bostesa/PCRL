"""Inference for the useful-head comparison, from saved per-row predictions only.

Point: mean over encoder seeds 0,1,2 of per-seed statistics (recovery: macro AUC of the base NL attacker, mean over
attacker seeds as0-as2 first). SE: sd (ddof 1) over B = 1999 paired multinomial bootstrap replicates of assessment
record groups (seed 20261041; the same draws for every unit). Interval: est +/- z * SE, z = two-sided Bonferroni normal
critical value of the family. PASS iff lower > target (the comparison is strict).

Unit resolution: bank -> validation-selected candidate; a features+output unit whose registered plus-surface rule
selected ignore_rep / ignore_out -> its component unit; an alias -> its output-aware source directory.

    OMP_NUM_THREADS=1 python -m cap.infer
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

import odx.infer as I
import oar.study as S
from cap import family as F
from cap.run import ARMS, DS, OAR, RUN, uid

HOME = Path.home()
PKG = Path(__file__).resolve().parents[1] / "results" / "combined_analysis_paper_v1"
CLASSES = (0, 1)


def unit_path(u):
    p = RUN / "units" / u
    return p if p.exists() else OAR / "units" / u


def resolve(u):
    seen = []
    while True:
        seen.append(u)
        d = unit_path(u)
        rec = json.loads((d / "record.json").read_text())
        if rec.get("kind") == "bank":
            u = rec["bank_selected"]
            continue
        src = (rec.get("plus_selection") or {}).get("alias_source")
        if src:
            u = src
            continue
        if (d / "ALIAS.json").exists():
            return HOME / json.loads((d / "ALIAS.json").read_text())["source"].removeprefix("~/")
        return d


def resolution_chain(u):
    """Human-readable chain for the report (which unit actually scores)."""
    out = [u]
    while True:
        rec = json.loads((unit_path(u) / "record.json").read_text())
        nxt = rec.get("bank_selected") if rec.get("kind") == "bank" else (rec.get("plus_selection") or {}).get("alias_source")
        if not nxt:
            return out
        u = nxt
        out.append(u)


I.resolve = resolve  # odx.infer.DS.recovery / load_preds call the module-level resolve


def world():
    W = S.load_world(DS)
    a, f = W["idx"]["assessment"], W["idx"]["attacker_fit"]
    maj = int(np.argmax(np.bincount(W["t"][f])))
    return W, a, maj


def head_probs(k, arm, W, a):
    z = np.load(OAR / "units" / f"{DS}__s{k}__HEAD__{ARMS[arm]}" / "preds.npz")
    assert np.array_equal(z["assess_row_id"], W["row_id"][a]) and np.array_equal(z["y_t"], W["t"][a])
    return np.exp(z["head_outputs_all"][a])


def u2_acc(k, arm, W, a):
    if arm == "F":
        j = json.loads((OAR / "selection" / f"{DS}__s{k}.json").read_text())["nominee_unit_source"]
        u = f"{DS}__s{k}__U2__Fc{j}"
    else:
        u = f"{DS}__s{k}__U2__{ARMS[arm]}"
    z = np.load(OAR / "units" / u / "preds.npz")
    assert np.array_equal(z["assess_row_id"], W["row_id"][a])
    return float((z["U2_P"].argmax(1) == z["y_t"]).mean()), u


def actual_head_utility(W, a, maj):
    t = W["t"][a]
    rows = []
    minority = int(np.argmin(np.bincount(W["t"][W["idx"]["attacker_fit"]])))
    const_acc = float((t == maj).mean())
    for k in S.SEEDS:
        for arm in ("A",) + F.ARMS:
            P = head_probs(k, arm, W, a)
            yhat = P.argmax(1)
            rec = [float((yhat[t == c] == c).mean()) for c in (0, 1)]
            u2, u2u = u2_acc(k, arm, W, a)
            rows.append({"seed": k, "arm": arm, "head_unit": f"{DS}__s{k}__HEAD__{ARMS[arm]}",
                         "accuracy": float((yhat == t).mean()), "balanced_accuracy": float(np.mean(rec)),
                         "minority_class": minority, "minority_recall": rec[minority],
                         "log_loss": float(-np.mean(np.log(np.clip(P[np.arange(len(t)), t], 1e-12, 1)))),
                         "constant_accuracy": const_acc, "gain_over_constant": float((yhat == t).mean()) - const_acc,
                         "share_predicted_positive": float((yhat == 1).mean()),
                         "historical_probe_U2_accuracy": u2, "historical_probe_unit": u2u})
    return rows


def build(W, a, maj):
    D = I.DS(W["row_id"][a], W["unit"][a])
    t = W["t"][a]
    cK = (t == maj).astype(float)
    correct = {(k, arm): (head_probs(k, arm, W, a).argmax(1) == t).astype(float) for k in S.SEEDS for arm in ("A",) + F.ARMS}
    ids = {}
    # ---- primary utility (9)
    for arm in F.ARMS:
        ids[f"U-acc-{arm}-vs-A"] = D.mean(f"m-acc-{arm}", [D.accuracy(f"acc-{arm}-A-s{k}", correct[k, arm] - correct[k, "A"]) for k in S.SEEDS])
        ids[f"U-gain-{arm}"] = D.mean(f"m-gain-{arm}", [D.accuracy(f"gain-{arm}-s{k}", correct[k, arm] - cK) for k in S.SEEDS])
        ids[f"U-retain-{arm}"] = D.mean(f"m-ret-{arm}", [D.accuracy(f"ret-{arm}-s{k}", correct[k, arm] - 0.8 * correct[k, "A"] - 0.2 * cK)
                                                       for k in S.SEEDS])
    # ---- recovery per (arm, view, surface)
    R = {}

    def rec(arm, view, surface):
        key = (arm, view, surface)
        if key not in R:
            R[key] = D.mean(f"R|{arm}|{view}|{surface}", [D.recovery(uid(k, arm, view, surface), CLASSES) for k in S.SEEDS])
        return R[key]

    for contract in ("centred", "hard"):
        for hi, lo in F.CONTRASTS:
            ids[f"R-out-{contract}-{hi}-minus-{lo}"] = D.diff(f"d-out-{contract}-{hi}-{lo}", rec(hi, "out", contract), rec(lo, "out", contract))
    # ---- secondary (33)
    for hi, lo in F.CONTRASTS:
        ids[f"S-out-full-{hi}-minus-{lo}"] = D.diff(f"d-out-full-{hi}-{lo}", rec(hi, "out", "full"), rec(lo, "out", "full"))
    for c in ("full", "centred", "hard"):
        for hi, lo in F.CONTRASTS:
            ids[f"S-feat+out-{c}-{hi}-minus-{lo}"] = D.diff(f"d-fo-{c}-{hi}-{lo}", rec(hi, "feat+out", c), rec(lo, "feat+out", c))
    for hi, lo in F.CONTRASTS:
        ids[f"S-feat-{hi}-minus-{lo}"] = D.diff(f"d-feat-{hi}-{lo}", rec(hi, "feat", "none"), rec(lo, "feat", "none"))
    for arm in ("A",) + F.ARMS:
        ids[f"S-offset-out-{arm}"] = D.diff(f"d-off-{arm}", rec(arm, "out", "fullbank"), rec(arm, "out", "iobank"))
        ids[f"S-bypass-{arm}"] = D.diff(f"d-byp-{arm}", rec(arm, "feat+clean", "full"), rec(arm, "feat+out", "full"))
    # ---- descriptive levels (not tested)
    levels = {f"R|{a}|{v}|{s}": i for (a, v, s), i in R.items()}
    for arm in ("A",) + F.ARMS:
        for view, surface in (("out", "dc"), ("out", "prob"), ("feat+out", "prob"), ("out", "iobank"), ("out", "fullbank"),
                              ("feat+out", "iobank"), ("feat+out", "fullbank")):
            levels[f"R|{arm}|{view}|{surface}"] = rec(arm, view, surface)
        levels[f"Acc|{arm}"] = D.mean(f"lvl-acc-{arm}", [D.accuracy(f"acc-{arm}-s{k}", correct[k, arm]) for k in S.SEEDS])
    levels["Acc|const"] = D.accuracy("acc-const", cK)
    per_seed = {}
    for (arm, view, surface) in list(R):
        for k in S.SEEDS:
            per_seed[f"R|{arm}|{view}|{surface}|s{k}"] = D.recovery(uid(k, arm, view, surface), CLASSES)
    for arm in ("A",) + F.ARMS:
        for k in S.SEEDS:
            per_seed[f"Acc|{arm}|s{k}"] = D.accuracy(f"acc-{arm}-s{k}", correct[k, arm])
    return D, ids, levels, per_seed


def decide(e, target):
    return "PASS" if e["lower"] > target else "NOT_ESTABLISHED"


def main():
    assert os.environ.get("OMP_NUM_THREADS") == "1"
    W, a, maj = world()
    util = actual_head_utility(W, a, maj)
    D, ids, levels, per_seed = build(W, a, maj)
    allids = list(ids.values()) + list(levels.values()) + list(per_seed.values())
    estP = I.estimate(D, [ids[e["id"]] for e in F.PRIMARY], F.B_SE, F.SEED_SE, F.Z_PRIMARY)
    estS = I.estimate(D, [ids[e["id"]] for e in F.SECONDARY], F.B_SE, F.SEED_SE, F.Z_SECONDARY)
    estL = I.estimate(D, list(levels.values()) + list(per_seed.values()), F.B_SE, F.SEED_SE, 1.6448536269514722)
    out = {"primary": [], "secondary": [], "levels": {}, "per_seed": {}, "utility": util,
           "resolution": {}, "n_assessment_rows": int(len(a)), "n_assessment_groups": int(len(np.unique(W["unit"][a])))}
    for fam, est, key in ((F.PRIMARY, estP, "primary"), (F.SECONDARY, estS, "secondary")):
        for e in fam:
            r = est[ids[e["id"]]]
            out[key].append({**{k: v for k, v in e.items()}, "point": r["point"], "se": r["se"], "lower": r["lower"],
                             "upper": r["upper"], "z": r["z"], "decision": decide(r, e["target"]),
                             "n_finite_replicates": r["n_finite_replicates"]})
    for name, i in levels.items():
        out["levels"][name] = {k: estL[i][k] for k in ("point", "se")}
    for name, i in per_seed.items():
        out["per_seed"][name] = estL[i]["point"]
    for k in S.SEEDS:
        for arm in ("A",) + F.ARMS:
            for view, surface in (("out", "centred"), ("out", "hard"), ("out", "full"), ("feat+out", "full"), ("feat+out", "centred"),
                                  ("feat+out", "hard"), ("feat+out", "fullbank"), ("out", "fullbank"), ("out", "iobank"),
                                  ("feat", "none"), ("feat+clean", "full")):
                u = uid(k, arm, view, surface)
                out["resolution"][u] = {"chain": resolution_chain(u), "scored_dir": "~/" + str(resolve(u).relative_to(HOME))}
    (RUN / "inference.json").write_text(json.dumps(out, indent=1, default=float))
    write_tables(out)
    return out


def _w(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})


def write_tables(out):
    _w(PKG / "ACTUAL_HEAD_UTILITY.csv", out["utility"],
       ["seed", "arm", "head_unit", "accuracy", "balanced_accuracy", "minority_class", "minority_recall", "log_loss",
        "constant_accuracy", "gain_over_constant", "share_predicted_positive", "historical_probe_U2_accuracy", "historical_probe_unit"])
    for key, name in (("primary", "PRIMARY_USEFUL_HEAD_ENDPOINTS.csv"), ("secondary", "SECONDARY_COMPLETE_RELEASE_ENDPOINTS.csv")):
        rows = []
        for e in out[key]:
            r = {"id": e["id"], "contract": e.get("contract", "deployed head utility"), "statistic": e.get("stat") or f"R({e['hi']}) - R({e['lo']})",
                 "target": e["target"], "point": e["point"], "se": e["se"], "lower": e["lower"], "upper": e["upper"], "z": e["z"],
                 "decision": e["decision"], "n_finite_replicates": e["n_finite_replicates"]}
            pts = []
            for k in S.SEEDS:
                pts.append(_seed_point(out, e, k))
            r.update({f"seed{k}": p for k, p in zip(S.SEEDS, pts)})
            rows.append(r)
        _w(PKG / name, rows, ["id", "contract", "statistic", "target", "point", "se", "lower", "upper", "z", "decision",
                              "seed0", "seed1", "seed2", "n_finite_replicates"])


def _seed_point(out, e, k):
    ps = out["per_seed"]
    if e.get("kind") == "utility":
        arm = e["arm"]
        u = {r["arm"]: r for r in out["utility"] if r["seed"] == k}
        if e["id"].startswith("U-acc"):
            return u[arm]["accuracy"] - u["A"]["accuracy"]
        if e["id"].startswith("U-gain"):
            return u[arm]["gain_over_constant"]
        return u[arm]["accuracy"] - 0.8 * u["A"]["accuracy"] - 0.2 * u[arm]["constant_accuracy"]
    view_surface = {"output-only/centred": ("out", "centred"), "output-only/hard": ("out", "hard"), "output-only/full": ("out", "full"),
                    "features+own-output/full": ("feat+out", "full"), "features+own-output/centred": ("feat+out", "centred"),
                    "features+own-output/hard": ("feat+out", "hard"), "features-only": ("feat", "none")}
    if "hi" in e:
        v, s = view_surface[e["contract"]]
        return ps[f"R|{e['hi']}|{v}|{s}|s{k}"] - ps[f"R|{e['lo']}|{v}|{s}|s{k}"]
    if e["id"].startswith("S-offset"):
        return ps[f"R|{e['arm']}|out|fullbank|s{k}"] - ps[f"R|{e['arm']}|out|iobank|s{k}"]
    return ps[f"R|{e['arm']}|feat+clean|full|s{k}"] - ps[f"R|{e['arm']}|feat+out|full|s{k}"]


if __name__ == "__main__":
    main()
