"""Single outer scoring for the focused no-erasure study. Refuses unless SELECTION_LOCK.json is committed and pushed.

Scored per seed (descriptive grid + every reference; the families use the frozen selections only):
  U, E, JP x {0.1, 1, 10} (erased), PN x {0.1, 1, 10}, LN x {0.1, 1, 10}, F pair, F0 pair.
Attack slates, roles, refits, utility and native diagnostics are the predecessor's (jcv.audit / jcv.outer helpers).
Provenance repair versus the predecessor: every coalition bank records the coalition's OWN slate table AND both
ignore-other-view slate tables, plus the bank decision, even when a local attacker wins.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np

from jcv import audit as AU
from jcv import outer as JO
from jcv import run as JR
from pnx import run as R


def audit_views_full(V, y, D, slate_fn, K, finite, coalition_key):
    f, v, a = D["idx"]["attacker_fit"], D["idx"]["attacker_val"], D["idx"]["assessment"]
    sels, out, P = {}, {}, {}
    for w, X in V.items():
        sels[w] = JO.fit_select(X[f], y[f], X[v], y[v], slate_fn(finite), K)
    locals_ = [x for x in V if x != coalition_key]
    for w, X in V.items():
        src = w
        rec = {"own_selected": sels[w]["selected"], "own_val_log_loss": sels[w]["val_log_loss"], "own_table": sels[w]["table"]}
        if w == coalition_key:
            cands = [(sels[w]["val_log_loss"], 0, w)] + [(sels[o]["val_log_loss"], j + 1, o) for j, o in enumerate(locals_)]
            src = min(cands)[2]
            rec["bank"] = [{"candidate": "coalition" if c[2] == w else f"ignore_other:{c[2]}", "view": c[2],
                            "attacker": sels[c[2]]["selected"], "val_log_loss": c[0]} for c in cands]
            rec["ignore_other_tables"] = {o: sels[o]["table"] for o in locals_}
        sel = sels[src]
        P[w] = JO.score(sel, V[src][f], y[f], V[src][a], K)
        out[w] = {**rec, "selected": sel["selected"], "selected_view": src, "val_log_loss": sel["val_log_loss"]}
    return out, P


def release_for(spec):
    if "units" in spec:
        n1, n2 = spec["units"]
        V = R.fare_pair_views(n1, n2)
        r1, r2 = np.load(R.udir(n1) / "release.npz")["r"], np.load(R.udir(n2) / "release.npz")["r"]
        return V, [r1, r2], True, None
    V = R.release_views(spec["unit"])
    z = np.load(R.udir(spec["unit"]) / "release.npz")
    meta = json.loads((R.udir(spec["unit"]) / "record.json").read_text()).get("finalize", {})
    return V, [z["r1"], z["r2"]], False, meta


def outer_unit(D, k, label, spec):
    name = f"outer__s{k}__{label}"
    if R.done(name):
        return
    t0 = time.time()
    V, Rs, finite, meta = release_for(spec)
    S, race = D["sex"], D["race"]
    a = D["idx"]["assessment"]
    prim, Pp = audit_views_full({w: V[w] for w in ("v1", "v2", "pair")}, S, D, lambda fin: AU.slate("final", fin), 2,
                                finite, "pair")
    o = V["out"]
    outv = {"p1": o["p1"], "p2": o["p2"], "ppair": np.hstack([o["p1"], o["p2"]]),
            "h1": np.eye(2)[o["hard1"]], "h2": np.eye(6)[o["hard2"]],
            "hpair": np.hstack([np.eye(2)[o["hard1"]], np.eye(6)[o["hard2"]]])}
    secP, Ps1 = audit_views_full({x: outv[x] for x in ("p1", "p2", "ppair")}, S, D, JO.secondary_slate, 2, finite, "ppair")
    secH, Ps2 = audit_views_full({x: outv[x] for x in ("h1", "h2", "hpair")}, S, D, JO.secondary_slate, 2, True, "hpair")
    Kr = int(race.max()) + 1
    rsup = JO.supported(race, D, Kr)
    keep = np.isin(race, rsup)
    Dr = {"idx": {r: D["idx"][r][keep[D["idx"][r]]] for r in ("attacker_fit", "attacker_val", "assessment")}}
    raceA, Pr = audit_views_full({w: V[w] for w in ("v1", "v2", "pair")}, race, Dr, JO.secondary_slate, Kr, finite, "pair")
    cst = JR.constants(D)
    util, prb = {}, {}
    for i, t in enumerate(("income", "occupation_group")):
        y = D["y"][t]
        util[i] = JO.utility(o[f"p{i + 1}"], o[f"hard{i + 1}"], y, D, JR.KS[i], cst[i])
        prb[i], _ = JO.probe(Rs[i], y, D, JR.KS[i])
    native = {"leace_native_fit_rows": (meta or {}).get("leace") or None,
              "held_out_linear": {i: JO.held_out_linear(Rs[i], D) for i in (0, 1)},
              "fit_rows_max_abs_rel_crosscov": {i: fit_crosscov(Rs[i], D) for i in (0, 1)},
              "ols_r2_sex": ols_r2(Rs, D)}
    rec = {"unit": name, "seed": k, "label": label, "source": spec.get("unit", spec.get("units")), "beta": spec.get("beta"),
           "primary": prim, "secondary_prob": secP, "secondary_hard": secH,
           "race": {"supported_classes": rsup, "audit": raceA, "n_assessment": int(len(Dr["idx"]["assessment"]))},
           "utility_deployed": util, "utility_common_probe": prb, "native_vs_audit": native, "wall_s": time.time() - t0}
    preds = {"assess_row_id": D["row_id"][a], "assess_unit": D["unit"][a], "sex": S[a],
             "race_row_id": D["row_id"][Dr["idx"]["assessment"]], "race": race[Dr["idx"]["assessment"]],
             "y_income": D["y"]["income"][a], "y_occ": D["y"]["occupation_group"][a],
             "hard1": o["hard1"][a], "hard2": o["hard2"][a], "p1": o["p1"][a], "p2": o["p2"][a]}
    for w, P in {**Pp, **Ps1, **Ps2}.items():
        preds[f"P_{w}"] = P
    for w, P in Pr.items():
        preds[f"Prace_{w}"] = P
    R.FN.save_unit(R.U(name), {"preds.npz": lambda p: np.savez_compressed(p, **preds)}, rec)
    R.event("unit complete", unit=name)


def fit_crosscov(Rf, D):
    """Fitted-moment diagnostic on defense_train rows: max |Cov(r_j, S)| / (sd(r_j) sd(S)). A measurement for PN/LN
    (no guarantee); for erased controls LEACE makes it ~0 on these rows by construction."""
    tr = D["idx"]["defense_train"]
    Z, S = Rf[tr], D["sex"][tr].astype(float)
    sd = Z.std(0)
    c = ((Z - Z.mean(0)) * (S - S.mean())[:, None]).mean(0)
    return float(np.max(np.where(sd > 1e-12, np.abs(c) / (sd * S.std()), 0.0)))


def ols_r2(Rs, D):
    """Rotation-invariant linear diagnostic (review A2): OLS R^2 of SEX on r1, r2 and [r1, r2], fitted and scored on the
    fitting rows, and fitted on attacker_fit / scored on assessment; plus a shuffled-SEX null on the fitting rows."""
    from sklearn.linear_model import LinearRegression
    tr, af, a = D["idx"]["defense_train"], D["idx"]["attacker_fit"], D["idx"]["assessment"]
    S = D["sex"].astype(float)
    rng = np.random.default_rng(20261016)
    out = {}
    for nm, Z in (("r1", Rs[0]), ("r2", Rs[1]), ("r1+r2", np.hstack(Rs))):
        m = LinearRegression().fit(Z[tr], S[tr])
        Sn = rng.permutation(S[tr])
        mn = LinearRegression().fit(Z[tr], Sn)
        m2 = LinearRegression().fit(Z[af], S[af])
        out[nm] = {"fit_rows_r2": float(m.score(Z[tr], S[tr])), "fit_rows_null_r2": float(mn.score(Z[tr], Sn)),
                   "heldout_r2_fit_attacker_fit_score_assessment": float(m2.score(Z[a], S[a]))}
    return out


def scored_units(k, SL):
    out = {"U": {"unit": f"nn__s{k}__U", "beta": 0.0}, "E": {"unit": f"nn__s{k}__E", "beta": 0.0}}
    for b in R.BETAS:
        out[f"JP_b{b:g}"] = {"unit": f"nn__s{k}__JP__b{b:g}", "beta": b}
        for arm in ("PN", "LN"):
            out[f"{arm}_b{b:g}"] = {"unit": R.unit_name(k, arm, b), "beta": b}
    for arm in ("F", "F0"):
        out[arm] = {"units": SL["seeds"][str(k)]["arms"][arm]["units"]}
    return out


def lock_is_pushed(path):
    rel = str(Path(path).resolve().relative_to(R.WT))
    head = subprocess.run(["git", "-C", str(R.WT), "log", "-1", "--format=%H", "--", rel], capture_output=True, text=True).stdout.strip()
    if not head:
        return False
    rem = subprocess.run(["git", "-C", str(R.WT), "branch", "-r", "--contains", head], capture_output=True, text=True).stdout
    return "origin/research/pcrl-penalty-no-erasure-v1" in rem


def main(argv=None):
    import argparse
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection-lock", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    a = ap.parse_args(argv)
    assert os.environ.get("OMP_NUM_THREADS") == "1"
    if not lock_is_pushed(a.selection_lock):
        raise SystemExit("REFUSED: SELECTION_LOCK.json is not committed and pushed")
    SL = json.loads(Path(a.selection_lock).read_text())
    D = R.load_D()
    for k in a.seeds:
        for label, spec in scored_units(k, SL).items():
            outer_unit(D, k, label, spec)


if __name__ == "__main__":
    main()
