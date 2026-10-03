"""FARE stage for one (dataset, encoder seed): frozen grid on defense_fit -> validation-only nominee -> three access
views + head, zero-fairness twin at the nominee's budget, native certificate on the reserved cert rows.

Unit names: grid configs  <P>__Fc<id>__rep / <P>__U2__Fc<id>   (id = grid id 1..6)
            nominee views <P>__F__rep+clean, <P>__F__rep+head, <P>__O_headF, <P>__HEAD__F   (F__rep := Fc<nominee>__rep)
            zero-fairness <P>__FZ__rep, __rep+clean, __rep+head, <P>__U2__FZ, <P>__HEAD__FZ, <P>__O_headFZ
A grid config whose encoder produces exactly the same cells as an earlier config is an ALIAS of it (not refitted)."""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import roc_auc_score

from . import study as S


def val_macro_auc(uid, classes):
    d = S.unit_dir(uid)
    with np.load(d / "val_preds.npz") as z:
        y, P = z["val_y_s"], z["VAL__NL__as0"]
    return float(np.mean([roc_auc_score(y == c, P[:, c]) for c in classes]))


def select_nominee(P, grid, classes, alias, cap_pts=0.01):
    """Frozen validation-only rule. Admissible: U2 attacker_val accuracy >= untreated U2 attacker_val accuracy - cap.
    Among admissible: lowest attacker_val macro AUC (base NL, attacker seed 0) on the FARE features; ties (<=1e-12)
    -> lower grid id. None admissible: no nominee; fallback (descriptive only) = highest validation accuracy, ties ->
    lower id."""
    accA = json.loads((S.unit_dir(f"{P}__U2__A") / "record.json").read_text())["val_accuracy"]
    rows = []
    for cfg in grid:
        i = cfg["id"]
        src = alias.get(i, i)
        acc = json.loads((S.unit_dir(f"{P}__U2__Fc{src}") / "record.json").read_text())["val_accuracy"]
        rows.append({"config": i, "alias_of": alias.get(i), "val_u2_accuracy": acc,
                     "val_nl_macro_auc": val_macro_auc(f"{P}__Fc{src}__rep", classes),
                     "admissible": acc >= accA - cap_pts})
    adm = [r for r in rows if r["admissible"]]
    if adm:
        best = min(adm, key=lambda r: (round(r["val_nl_macro_auc"], 12), r["config"]))
        return {"untreated_val_u2_accuracy": accA, "cap": cap_pts, "table": rows, "nominee": best["config"],
                "nominee_unit_source": alias.get(best["config"], best["config"]), "admissible": True}
    fb = max(rows, key=lambda r: (round(r["val_u2_accuracy"], 12), -r["config"]))
    return {"untreated_val_u2_accuracy": accA, "cap": cap_pts, "table": rows, "nominee": fb["config"],
            "nominee_unit_source": alias.get(fb["config"], fb["config"]), "admissible": False,
            "note": "no admissible nominee: task-inferior fallback, descriptive only; cannot pass the competitive conjunction"}


def _cert(FO, m, X, s, cfg):
    try:
        return FO.certificate(m, X, s, cfg)
    except FO.CertificateRefused as e:   # e.g. exact duplicate of a fit row: reported, never patched
        return {"status": "UNAVAILABLE", "reason": f"CertificateRefused: {e}"}


def run_fare_seed(ds, k, P, W, H, O, var, arms, heads, fare, E, auth, syn, atk, u2, led, log):
    from . import fare_official as FO
    c = S.CELLS[ds]
    df = W["idx"]["defense_fit"]
    grid = fare["grid"]
    fitted, alias, seen = {}, {}, []
    for cfg in grid:
        i = cfg["id"]
        m, cells, rec = FO.fit_encode_cached(f"{P}__FAREFIT_c{i}", H[df], W["t"][df], W["s"][df], H, cfg,
                                             seed=fare["seed_base"] + k, auth=auth, synthetic=syn, ledger=led)
        fitted[i] = (m, cells, rec)
        same = [j for j, cj in seen if np.array_equal(cj, cells)]
        if same:
            alias[i] = same[0]
            continue
        seen.append((i, cells))
        X = S.onehot(cells, int(cells.max()) + 1)
        atk(f"{P}__Fc{i}__rep", X, finite=True, contract={"features": f"FARE grid id {i}", "outputs": None})
        u2(f"{P}__U2__Fc{i}", X)
    sel = select_nominee(P, grid, fare["supported_classes"][ds], alias)
    sel["aliases"] = {str(a): b for a, b in alias.items()}
    (S.RUN / "selection").mkdir(parents=True, exist_ok=True)
    (S.RUN / "selection" / f"{P}.json").write_text(json.dumps(sel, indent=1))
    j = sel["nominee"]
    jsrc = sel["nominee_unit_source"]
    m, cells, rec = fitted[j]
    XF = S.onehot(cells, int(cells.max()) + 1)
    cfgZ = FO.zero_fairness(next(g for g in grid if g["id"] == j))
    mZ, cellsZ, recZ = FO.fit_encode_cached(f"{P}__FAREFIT_Z", H[df], W["t"][df], W["s"][df], H, cfgZ,
                                            seed=fare["seed_base"] + k, auth=auth, synthetic=syn, ledger=led)
    XZ = S.onehot(cellsZ, int(cellsZ.max()) + 1)
    atk(f"{P}__FZ__rep", XZ, finite=True, contract={"features": "FARE zero-fairness twin", "outputs": None})
    u2(f"{P}__U2__FZ", XZ)
    for tag, X, rep_unit in (("F", XF, f"{P}__Fc{jsrc}__rep"), ("FZ", XZ, f"{P}__FZ__rep")):
        atk(f"{P}__{tag}__rep+clean", np.hstack([X, O]), plus={"rep_unit": rep_unit, "out_unit": f"{P}__O_full"},
            contract={"features": tag, "outputs": "historical clean logits"})
        h = S.head_unit(f"{P}__HEAD__{tag}", X, W, W["t"], c["K_t"], E, auth, syn, cpu_ledger=led)
        atk(f"{P}__O_head{tag}", h["outputs"], contract={"outputs": f"head on {tag} features", "features": None})
        atk(f"{P}__{tag}__rep+head", np.hstack([X, h["outputs"]]),
            plus={"rep_unit": rep_unit, "out_unit": f"{P}__O_head{tag}"},
            contract={"features": tag, "outputs": f"head fitted on {tag} features only"})
    cr, a = W["idx"]["cert"], W["idx"]["assessment"]
    cert = {"cert_rows": int(len(cr)), "cert_role": "20% of exposure-cleaned attacker_fit groups (oar-cert-v1)",
            "nominee": {"primary_all_groups": _cert(FO, m, H[cr], W["s"][cr], fare["certificate"])},
            "zero_fairness": {"primary_all_groups": _cert(FO, mZ, H[cr], W["s"][cr], fare["certificate"])},
            "tree_own_task_accuracy_assessment": {"nominee": FO.own_task_accuracy(m, H[a], W["t"][a]),
                                                  "zero_fairness": FO.own_task_accuracy(mZ, H[a], W["t"][a])},
            "n_cells": {"nominee": int(cells.max()) + 1, "zero_fairness": int(cellsZ.max()) + 1}}
    sec = fare.get("secondary_certificate_groups", {}).get(ds)
    if sec:
        cfg2 = {**fare["certificate"], "groups": sec}
        cert["nominee"]["secondary_groups"] = _cert(FO, m, H[cr], W["s"][cr], cfg2)
        cert["zero_fairness"]["secondary_groups"] = _cert(FO, mZ, H[cr], W["s"][cr], cfg2)
    (S.RUN / "certificates").mkdir(parents=True, exist_ok=True)
    (S.RUN / "certificates" / f"{P}.json").write_text(json.dumps(cert, indent=1, default=str))
    log(f"[{ds}] s{k} FARE nominee {j} admissible={sel['admissible']} aliases={alias}")
    return sel
