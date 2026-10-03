"""Stage 5 (conditional): screen for one useful-task cell on validation only, then official FARE vs target LEACE vs the
zero-fairness twin vs untreated under the same complete contract (features + own head).

    OAR_RUN_UNITS=~/PCRL_eval_cache_private/odx_v1/run/units python -m odx.fare5 --lock <LOCK.json> [--screen-only]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from . import run as R
import oar.study as S

GRID_SOURCE = Path(__file__).resolve().parents[1] / "results/combined_output_aware_removal_v1/notes/fare/FARE_GRID_PROPOSAL.json"


def screen(E, auth, syn, led):
    """Validation-only eligibility (PROTOCOL stage 5). Fits only U2 probes on untreated features (attacker_fit / val)."""
    rows = []
    for (ds, purpose, attr) in R.PAIRS:
        if (ds, purpose, attr) in R.PRIMARY:
            continue
        p = R.purposes(ds)[purpose]
        W = R.world(ds, purpose, attr)
        t, f, v = W["t"], W["idx"]["attacker_fit"], W["idx"]["attacker_val"]
        Kt = int(p["task_dim"])
        maj = int(np.argmax(np.bincount(t[f], minlength=Kt)))
        cval = float((t[v] == maj).mean())
        fg, ug = [], []
        for k in S.SEEDS:
            Fw = np.load(S.BENCH / "inputs" / f"{ds}_s{k}_forward.npz")
            fg.append(float((Fw[p["logits_key"]][v].argmax(1) == t[v]).mean()) - cval)
            u = S.u2_unit(f"{ds}__s{k}__{purpose}__U2__A", Fw[p["rep_key"]].astype(np.float64), W, t, Kt, E, auth, syn,
                          cpu_ledger=led)
            ug.append(float(u["val_accuracy"]) - cval)
        K = R.K_of(ds, attr)
        dfc = np.bincount(W["s"][W["idx"]["defense_fit"]], minlength=K)
        elig = (np.mean(fg) >= 0.03 and sum(g > 0 for g in fg) >= 2 and np.mean(ug) >= 0.03 and all(g > 0 for g in ug)
                and bool((dfc >= 100).all()))
        rows.append({"dataset": ds, "purpose": purpose, "attribute": attr, "frozen_val_gain_by_seed": fg,
                     "frozen_val_gain_mean": float(np.mean(fg)), "u2_val_gain_by_seed": ug,
                     "u2_val_gain_mean": float(np.mean(ug)), "defense_fit_class_counts": dfc.tolist(),
                     "constant_val_accuracy": cval, "eligible_before_support": bool(elig)})
    return rows


def run(lock, screen_only=False, log=print):
    E, auth, syn = S.effective(), S.auth_real(), False
    led = R.ledger("s5")
    rows = screen(E, auth, syn, led)
    # support requirement (>= 2 supported classes on attacker roles, 100/30/100) from COVERAGE_AND_SUPPORT.csv
    import csv
    cov = list(csv.DictReader(open(Path(__file__).resolve().parents[1] / "results/combined_output_diagnosis_v1/COVERAGE_AND_SUPPORT.csv")))
    for r in rows:
        sup = [int(c["id"]) for c in cov if c["dataset"] == r["dataset"] and c["purpose"] == r["purpose"]
               and c["attribute"] == r["attribute"] and c["what"] == "class" and c["supported"] == "True"]
        r["supported_classes"] = sup
        r["eligible"] = r["eligible_before_support"] and len(sup) >= 2
    elig = [r for r in rows if r["eligible"]]
    chosen = None
    if elig:
        best = max(r["frozen_val_gain_mean"] for r in elig)
        tied = sorted([r for r in elig if abs(r["frozen_val_gain_mean"] - best) <= 1e-12],
                      key=lambda r: (r["dataset"], r["purpose"], r["attribute"]))
        chosen = tied[0]
    out = {"screen": rows, "chosen": chosen and {k: chosen[k] for k in ("dataset", "purpose", "attribute")},
           "rule": "highest mean frozen-head validation gain among eligible; ties lexicographic (dataset, purpose, attribute)"}
    (R.RUN / "s5").mkdir(parents=True, exist_ok=True)
    (R.RUN / "s5" / "SCREEN.json").write_text(json.dumps(out, indent=1, default=str))
    log(f"S5 screen: chosen={out['chosen']}")
    if screen_only or chosen is None:
        return out
    return run_cell(chosen, lock, E, auth, syn, led, log)


def _val_macro(uid, classes):
    with np.load(S.unit_dir(uid) / "val_preds.npz") as z:
        y, P = z["val_y_s"], z["VAL__NL__as0"]
    return float(np.mean([roc_auc_score(y == c, P[:, c]) for c in classes]))


def run_cell(ch, lock, E, auth, syn, led, log):
    from oar import fare_official as FO
    from stored_model_eval.defenses import LeaceMap
    ds, purpose, attr = ch["dataset"], ch["purpose"], ch["attribute"]
    p = R.purposes(ds)[purpose]
    W = R.world(ds, purpose, attr)
    s, t, K, Kt = W["s"], W["t"], R.K_of(ds, attr), int(p["task_dim"])
    cls = ch["supported_classes"]
    grid = [{k: g[k] for k in ("id", "max_leaf_nodes", "min_samples_leaf", "gamma", "criterion")}
            for g in json.loads(GRID_SOURCE.read_text())["grid"]]
    df, f, v = W["idx"]["defense_fit"], W["idx"]["attacker_fit"], W["idx"]["attacker_val"]
    maj = int(np.argmax(np.bincount(t[f], minlength=Kt)))
    cval = float((t[v] == maj).mean())
    res = {"cell": [ds, purpose, attr], "seeds": {}}

    def atk(uid, X, finite=False, plus=None):
        return S.attack_unit(uid, X, W, s, K, E, auth, syn, finite=finite, plus=plus, cpu_ledger=led)

    for k in S.SEEDS:
        P = f"S5__{ds}__s{k}__{purpose}__{attr}"
        H = np.load(S.BENCH / "inputs" / f"{ds}_s{k}_forward.npz")[p["rep_key"]].astype(np.float64)
        mid = f"{ds}__s{k}__{purpose}__B_{attr}"
        B = np.asarray(LeaceMap.load(S.BENCH / "defenses" / mid / "map", verify_package=True).transform(H), dtype=np.float64)
        rel = {"A": H, "B": B}
        uA = S.u2_unit(f"{ds}__s{k}__{purpose}__U2__A", H, W, t, Kt, E, auth, syn, cpu_ledger=led)
        accA = float(uA["val_accuracy"])
        table, cells_by = [], {}
        for cfg in grid:
            m, cells, rec = FO.fit_encode_cached(f"{P}__FAREFIT_c{cfg['id']}", H[df], t[df], s[df], H, cfg,
                                                 seed=k, auth=auth, synthetic=syn, ledger=led)
            alias = next((j for j, cj in cells_by.items() if np.array_equal(cj, cells)), None)
            cells_by[cfg["id"]] = cells
            src = alias or cfg["id"]
            if alias is None:
                X = S.onehot(cells, int(cells.max()) + 1)
                atk(f"{P}__Fc{cfg['id']}__rep", X, finite=True)
                S.u2_unit(f"{P}__U2__Fc{cfg['id']}", X, W, t, Kt, E, auth, syn, cpu_ledger=led)
            accF = float(json.loads((S.unit_dir(f"{P}__U2__Fc{src}") / "record.json").read_text())["val_accuracy"])
            ok = (accF >= accA - 0.01) and ((accF - cval) >= 0.8 * (accA - cval))
            table.append({"config": cfg["id"], "alias_of": alias, "n_cells": int(cells.max()) + 1, "val_u2_accuracy": accF,
                          "val_nl_macro_auc": _val_macro(f"{P}__Fc{src}__rep", cls), "feasible": bool(ok)})
        feas = [r for r in table if r["feasible"]]
        nom = min(feas, key=lambda r: (round(r["val_nl_macro_auc"], 12), r["config"])) if feas else None
        seed_rec = {"untreated_val_u2_accuracy": accA, "constant_val_accuracy": cval, "table": table,
                    "nominee": nom and nom["config"], "status": "OK" if nom else "NO_FEASIBLE_NOMINEE"}
        res["seeds"][k] = seed_rec
        if nom is None:
            continue
        src = nom["alias_of"] or nom["config"]
        cellsF = cells_by[src]
        mF = FO.FareModel.load(Path(os.environ["OAR_RUN_UNITS"]) / f"{P}__FAREFIT_c{src}" / "model")
        cfgZ = FO.zero_fairness(next(g for g in grid if g["id"] == nom["config"]))
        mZ, cellsZ, _ = FO.fit_encode_cached(f"{P}__FAREFIT_Z", H[df], t[df], s[df], H, cfgZ, seed=k, auth=auth,
                                             synthetic=syn, ledger=led)
        rel["F"], rel["FZ"] = S.onehot(cellsF, int(cellsF.max()) + 1), S.onehot(cellsZ, int(cellsZ.max()) + 1)
        for tag, X in rel.items():
            fin = tag in ("F", "FZ")
            rep_u = f"{P}__Fc{src}__rep" if tag == "F" else f"{P}__{tag}__rep"
            if tag != "F":
                atk(rep_u, X, finite=fin)
            if tag != "A":
                S.u2_unit(f"{P}__U2__{tag}", X, W, t, Kt, E, auth, syn, cpu_ledger=led)
            h = S.head_unit(f"{P}__HEAD__{tag}", X, W, t, Kt, E, auth, syn, cpu_ledger=led)
            atk(f"{P}__O_head{tag}", h["outputs"])
            atk(f"{P}__{tag}__rep+head", np.hstack([X, h["outputs"]]), plus={"rep_unit": rep_u, "out_unit": f"{P}__O_head{tag}"})
        cr = W["idx"]["cert"]
        def _c(m):
            try:
                return FO.certificate(m, H[cr], s[cr], {"delta": 0.05, "groups": None, "split_seed": 0, "val_fraction": 0.5,
                                                         "eps_b_fraction": 0.1, "eps_s_fraction": 0.1})
            except FO.CertificateRefused as e:
                return {"status": "UNAVAILABLE", "reason": f"CertificateRefused (feature-hash guard): {e}"}
        seed_rec["certificate"] = {"nominee": _c(mF), "zero_fairness": _c(mZ), "cert_rows": int(len(cr))}
        seed_rec["tree_own_task_accuracy_assessment"] = FO.own_task_accuracy(mF, H[W["idx"]["assessment"]], t[W["idx"]["assessment"]])
        log(f"S5 s{k}: nominee {nom['config']} ({nom['n_cells']} cells)")
    (R.RUN / "s5" / "CELL.json").write_text(json.dumps(res, indent=1, default=str))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", required=True)
    ap.add_argument("--screen-only", action="store_true")
    a = ap.parse_args(argv)
    if os.environ.get("OMP_NUM_THREADS") != "1" or "OAR_RUN_UNITS" not in os.environ:
        raise SystemExit("REFUSED: need OMP_NUM_THREADS=1 and OAR_RUN_UNITS pointing at the odx run units")
    from .lock import verify_lock
    v = verify_lock(Path(a.lock))
    if not v["ok"]:
        raise SystemExit("REFUSED: lock does not verify: " + "; ".join(v["mismatches"][:10]))
    R.event("start S5")
    run(json.loads(Path(a.lock).read_text()), screen_only=a.screen_only)
    R.event("end S5")


if __name__ == "__main__":
    main()
