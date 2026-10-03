"""Single outer scoring (running ladder F). Refuses to run unless SELECTION_LOCK.json is committed and pushed.

Per seed and arm (U, E, L, J, JP, S12, S21, F, F0; nominee, or the INFEASIBLE descriptive configuration):
  primary   views v1, v2, coalition = [v1, v2] (feature vector + centred logits); FINAL slate fitted on attacker_fit,
            selected on attacker_val log loss (coalition bank adds the selected v1-only / v2-only attackers), refitted at
            attacker seeds 0,1,2, scored on assessment. Per-row assessment probabilities are saved (private).
  secondary output-only probabilities and hard decisions (each recipient and the pair) with the SECONDARY slate
            (inner slate + defense-aware; finite releases add the cell-conditional attacker); race stress audit on the
            primary views (secondary slate, supported classes only).
  utility   deployed heads on assessment (accuracy, balanced accuracy over supported classes, per-class recall,
            minority recall = smallest supported class, log loss, Brier, ECE); common refitted probe (LR, C on
            attacker_val) on each release, reported separately.
  native    LEACE native check on fitting rows (from finalisation) beside held-out linear diagnostics.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np

from jcv import audit as AU
from jcv import data as DA
from jcv import finalize as FN
from jcv import run as R

ARMS = ["U", "E", "L", "J", "JP", "S12", "S21", "F", "F0"]
SUPPORT_MIN = 30   # race / task classes with >= 30 rows in EVERY scored role are "supported"


def secondary_slate(finite):
    s = AU.slate("inner") + [("DA_canonical_MLP", lambda sd: AU._da(sd))]
    if finite:
        s += [(f"CC_alpha{a}", (lambda sd, a=a: AU.CellConditional(a))) for a in (0.1, 1.0, 10.0)]
    return s


def fit_select(Xf, yf, Xv, yv, slate, K):
    best, table = None, []
    for name, fac in slate:
        m = fac(0).fit(Xf, yf)
        Pv = AU.proba(m, Xv, K)
        ll = AU.logloss(yv, Pv)
        table.append({"attacker": name, "attacker_val_log_loss": ll})
        if best is None or ll < best[0] - 1e-12:
            best = (ll, name, fac, m)
    return {"selected": best[1], "val_log_loss": best[0], "factory": best[2], "model0": best[3], "table": table}


def score(sel, Xf, yf, Xa, K):
    Ps = []
    for s in AU.ATT_SEEDS:
        m = sel["model0"] if s == 0 else sel["factory"](s).fit(Xf, yf)
        Ps.append(AU.proba(m, Xa, K))
    return np.stack(Ps)


def audit_views(V, y, D, slate_fn, K, finite, coalition_key="pair"):
    """Fit/select/score each view; the coalition bank includes the local selections (ignore-other-view)."""
    f, v, a = D["idx"]["attacker_fit"], D["idx"]["attacker_val"], D["idx"]["assessment"]
    sels, out, P = {}, {}, {}
    for w, X in V.items():
        sels[w] = fit_select(X[f], y[f], X[v], y[v], slate_fn(finite), K)
    for w, X in V.items():
        sel = sels[w]
        src = w
        if w == coalition_key:
            cands = [(sels[w]["val_log_loss"], 0, w)] + [(sels[o]["val_log_loss"], j + 1, o)
                                                          for j, o in enumerate(x for x in V if x != coalition_key)]
            src = min(cands)[2]
            sel = sels[src]
        P[w] = score(sel, V[src][f], y[f], V[src][a], K)
        out[w] = {"selected": sel["selected"], "selected_view": src, "val_log_loss": sel["val_log_loss"],
                  "table": sel["table"]}
    return out, P


def supported(y, D, K):
    return [c for c in range(K) if all((y[D["idx"][r]] == c).sum() >= SUPPORT_MIN
                                       for r in ("attacker_fit", "attacker_val", "assessment"))]


def utility(P, hard, y, D, K, const):
    a = D["idx"]["assessment"]
    yy, Pa, ha = y[a], P[a], hard[a]
    sup = supported(y, D, K)
    rec = {c: float((ha[yy == c] == c).mean()) for c in range(K) if (yy == c).sum()}
    minority = min(sup, key=lambda c: (yy == c).sum())
    onehot = np.eye(K)[yy]
    probs = Pa.max(1)
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    rel = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (probs > lo) & (probs <= hi)
        if m.sum():
            conf, acc = float(probs[m].mean()), float((ha[m] == yy[m]).mean())
            ece += m.mean() * abs(conf - acc)
            rel.append({"bin": [float(lo), float(hi)], "n": int(m.sum()), "confidence": conf, "accuracy": acc})
    return {"accuracy": float((ha == yy).mean()), "balanced_accuracy_supported": float(np.mean([rec[c] for c in sup])),
            "per_class_recall": rec, "supported_classes": sup, "minority_class": int(minority),
            "minority_recall": rec[minority], "log_loss": AU.logloss(yy, Pa),
            "brier": float(np.mean(np.sum((Pa - onehot) ** 2, 1))), "ece_10bin": float(ece), "reliability": rel,
            "const_accuracy": float((yy == const).mean()), "useful_gain": float((ha == yy).mean() - (yy == const).mean())}


def probe(Rfeat, y, D, K):
    f, v, a = D["idx"]["attacker_fit"], D["idx"]["attacker_val"], D["idx"]["assessment"]
    sel = fit_select(Rfeat[f], y[f], Rfeat[v], y[v], [(f"LR_C{C}", (lambda s, C=C: AU._lr(C, s))) for C in
                                                       (0.01, 0.1, 1.0, 10.0, 100.0)], K)
    Pa = AU.proba(sel["model0"], Rfeat[a], K)
    return {"probe": sel["selected"], "accuracy": float((Pa.argmax(1) == y[a]).mean())}, Pa.argmax(1)


def held_out_linear(Rfeat, D):
    a = D["idx"]["assessment"]
    S = D["sex"][a].astype(float)
    Z = Rfeat[a]
    sd = Z.std(0)
    corr = np.where(sd > 1e-12, ((Z - Z.mean(0)) * (S - S.mean())[:, None]).mean(0) / (sd * S.std() + 1e-300), 0.0)
    return {"max_abs_corr_assessment": float(np.max(np.abs(corr)))}


def fare_certificates(arm_rec, D):
    FO = R.fare_env()
    cert = D["idx"].get("cert")
    if cert is None:
        return {"status": "UNAVAILABLE", "reason": "no cert role"}
    out = {}
    for i, nm in enumerate(arm_rec["units"]):
        uid = json.loads((R.U(nm) / "record.json").read_text())["fare_uid"]
        try:
            m = FO.FareModel.load(FO.units_root() / uid / "model")
            out[i] = FO.certificate(m, D["X"][cert].astype(np.float64), D["sex"][cert])
        except Exception as e:   # reported, never patched
            out[i] = {"status": "UNAVAILABLE", "reason": f"{type(e).__name__}: {e}"}
    return out


def release_for(arm_rec):
    if "units" in arm_rec:   # FARE pair
        n1, n2 = arm_rec["units"]
        V = R.fare_pair_views(n1, n2)
        r1, r2 = np.load(R.U(n1) / "release.npz")["r"], np.load(R.U(n2) / "release.npz")["r"]
        return V, [r1, r2], True, None
    V = R.release_views(arm_rec["unit"])
    z = np.load(R.U(arm_rec["unit"]) / "release.npz")
    meta = json.loads((R.U(arm_rec["unit"]) / "record.json").read_text()).get("finalize", {})
    return V, [z["r1"], z["r2"]], False, meta


def outer_unit(D, k, arm, arm_rec):
    name = f"outer__s{k}__{arm}"
    if R.done(name):
        return
    t0 = time.time()
    V, Rs, finite, meta = release_for(arm_rec)
    S, race = D["sex"], D["race"]
    a = D["idx"]["assessment"]
    prim, Pp = audit_views({w: V[w] for w in ("v1", "v2", "pair")}, S, D,
                           lambda fin: AU.slate("final", fin), 2, finite)
    o = V["out"]
    outv = {"p1": o["p1"], "p2": o["p2"], "ppair": np.hstack([o["p1"], o["p2"]]),
            "h1": np.eye(2)[o["hard1"]], "h2": np.eye(6)[o["hard2"]], "hpair": np.hstack([np.eye(2)[o["hard1"]], np.eye(6)[o["hard2"]]])}
    secP, Ps1 = audit_views({k_: outv[k_] for k_ in ("p1", "p2", "ppair")}, S, D, secondary_slate, 2, finite, "ppair")
    secH, Ps2 = audit_views({k_: outv[k_] for k_ in ("h1", "h2", "hpair")}, S, D, secondary_slate, 2, True, "hpair")
    Kr = int(race.max()) + 1
    rsup = supported(race, D, Kr)
    keep = np.isin(race, rsup)
    Dr = {"idx": {r: D["idx"][r][keep[D["idx"][r]]] for r in ("attacker_fit", "attacker_val", "assessment")}}
    raceA, Pr = audit_views({w: V[w] for w in ("v1", "v2", "pair")}, race, Dr, secondary_slate, Kr, finite)
    cst = R.constants(D)
    util, prb = {}, {}
    for i, t in enumerate(("income", "occupation_group")):
        y = D["y"][t]
        util[i] = utility(o[f"p{i + 1}"], o[f"hard{i + 1}"], y, D, R.KS[i], cst[i])
        prb[i], _ = probe(Rs[i], y, D, R.KS[i])
    native = {"leace_native_fit_rows": (meta or {}).get("leace"),
              "held_out_linear": {i: held_out_linear(Rs[i], D) for i in (0, 1)}}
    if finite:   # FARE / F0: the native certificate per purpose on the cert role (official procedure)
        native["fare_certificate"] = fare_certificates(arm_rec, D)
    rec = {"unit": name, "seed": k, "arm": arm, "status": arm_rec["status"], "source": arm_rec.get("unit", arm_rec.get("units")),
           "beta": arm_rec.get("beta"), "primary": prim, "secondary_prob": secP, "secondary_hard": secH,
           "race": {"supported_classes": rsup, "audit": raceA, "n_assessment": int(len(Dr["idx"]["assessment"]))},
           "utility_deployed": util, "utility_common_probe": prb, "native_vs_audit": native, "wall_s": time.time() - t0}
    preds = {"assess_row_id": D["row_id"][a], "assess_unit": D["unit"][a], "sex": S[a], "race_row_id":
             D["row_id"][Dr["idx"]["assessment"]], "race": race[Dr["idx"]["assessment"]],
             "y_income": D["y"]["income"][a], "y_occ": D["y"]["occupation_group"][a],
             "hard1": o["hard1"][a], "hard2": o["hard2"][a], "p1": o["p1"][a], "p2": o["p2"][a]}
    for w, P in {**Pp, **Ps1, **Ps2}.items():
        preds[f"P_{w}"] = P
    for w, P in Pr.items():
        preds[f"Prace_{w}"] = P
    R.FN.save_unit(R.U(name), {"preds.npz": lambda p: np.savez_compressed(p, **preds)}, rec)
    R.event("unit complete", unit=name)


def lock_is_pushed(path):
    rel = str(Path(path).resolve().relative_to(R.WT))
    head = subprocess.run(["git", "-C", str(R.WT), "log", "-1", "--format=%H", "--", rel], capture_output=True,
                          text=True).stdout.strip()
    if not head:
        return False
    rem = subprocess.run(["git", "-C", str(R.WT), "branch", "-r", "--contains", head], capture_output=True,
                         text=True).stdout
    return "origin/research/pcrl-joint-complete-view-method-v1" in rem


def main(argv=None):
    import argparse
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection-lock", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--arms", nargs="*", default=ARMS)
    a = ap.parse_args(argv)
    assert os.environ.get("OMP_NUM_THREADS") == "1"
    if not lock_is_pushed(a.selection_lock):
        raise SystemExit("REFUSED: SELECTION_LOCK.json is not committed and pushed")
    SL = json.loads(Path(a.selection_lock).read_text())
    D = DA.load()
    for k in a.seeds:
        for arm in a.arms:
            outer_unit(D, k, arm, SL["seeds"][str(k)]["arms"][arm])


if __name__ == "__main__":
    main()
