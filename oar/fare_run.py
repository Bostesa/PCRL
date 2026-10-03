"""FARE stage for one (dataset, encoder seed): frozen grid on defense_fit -> validation-only nominee -> three access
views + head, zero-fairness control at the nominee's budget, native certificate on the reserved cert rows."""
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


def select_nominee(P, grid, classes, cap_pts=0.01):
    """Validation-only rule (frozen): admissible = U2 attacker_val accuracy >= untreated U2 attacker_val accuracy -
    cap; among admissible, lowest attacker_val macro AUC of the base NL (as0) on the FARE features; ties (<= 1e-12)
    -> lower grid index. None admissible -> no nominee; fallback (descriptive only) = highest validation accuracy,
    ties -> lower index."""
    accA = json.loads((S.unit_dir(f"{P}__U2__A") / "record.json").read_text())["val_accuracy"]
    rows = []
    for j, _ in enumerate(grid):
        acc = json.loads((S.unit_dir(f"{P}__U2__F{j}") / "record.json").read_text())["val_accuracy"]
        rows.append({"config": j, "val_u2_accuracy": acc, "val_nl_macro_auc": val_macro_auc(f"{P}__F{j}__rep", classes),
                     "admissible": acc >= accA - cap_pts})
    adm = [r for r in rows if r["admissible"]]
    if adm:
        best = min(adm, key=lambda r: (round(r["val_nl_macro_auc"], 12), r["config"]))
        return {"untreated_val_u2_accuracy": accA, "cap": cap_pts, "table": rows, "nominee": best["config"],
                "admissible": True}
    fb = max(rows, key=lambda r: (round(r["val_u2_accuracy"], 12), -r["config"]))
    return {"untreated_val_u2_accuracy": accA, "cap": cap_pts, "table": rows, "nominee": fb["config"],
            "admissible": False, "note": "no admissible nominee: task-inferior fallback, descriptive only; cannot pass "
                                         "the competitive conjunction"}


def run_fare_seed(ds, k, P, W, H, O, var, arms, heads, fare, E, auth, syn, atk, u2, led, log):
    from . import fare_official as FO
    c = S.CELLS[ds]
    df = W["idx"]["defense_fit"]
    s_fit = W["s"][df] if fare["sensitive_for_fit"] == "full_declared" else None
    grid = fare["grid"]
    models = {}
    for j, cfg in enumerate(grid):
        m, cells, rec = FO.fit_encode_cached(f"{P}__FAREFIT_F{j}", H[df], W["t"][df], W["s"][df], H, cfg,
                                             seed=fare["seed_base"] + k, auth=auth, synthetic=syn, ledger=led)
        models[j] = (m, cells, rec)
        X = S.onehot(cells, rec["n_cells"])
        atk(f"{P}__F{j}__rep", X, finite=True, contract={"features": f"FARE config {j}", "outputs": None})
        u2(f"{P}__U2__F{j}", X)
    sel = select_nominee(P, grid, fare["supported_classes"][ds])
    (S.RUN / "selection").mkdir(parents=True, exist_ok=True)
    (S.RUN / "selection" / f"{P}.json").write_text(json.dumps(sel, indent=1))
    j = sel["nominee"]
    m, cells, rec = models[j]
    XF = S.onehot(cells, rec["n_cells"])
    # zero-fairness control at the nominee's tree/output budget
    cfg0 = FO.zero_fairness(grid[j])
    m0, cells0, rec0 = FO.fit_encode_cached(f"{P}__FAREFIT_F0", H[df], W["t"][df], W["s"][df], H, cfg0,
                                            seed=fare["seed_base"] + k, auth=auth, synthetic=syn, ledger=led)
    X0 = S.onehot(cells0, rec0["n_cells"])
    for tag, X in (("F", XF), ("F0", X0)):
        atk(f"{P}__{tag}__rep", X, finite=True, contract={"features": tag, "outputs": None})
        atk(f"{P}__{tag}__rep+clean", np.hstack([X, O]), plus={"rep_unit": f"{P}__{tag}__rep", "out_unit": f"{P}__O_full"},
            contract={"features": tag, "outputs": "historical clean logits"})
        h = S.head_unit(f"{P}__HEAD__{tag}", X, W, W["t"], c["K_t"], E, auth, syn, cpu_ledger=led)
        atk(f"{P}__O_head{tag}", h["outputs"], contract={"outputs": f"head on {tag} features", "features": None})
        atk(f"{P}__{tag}__rep+head", np.hstack([X, h["outputs"]]),
            plus={"rep_unit": f"{P}__{tag}__rep", "out_unit": f"{P}__O_head{tag}"},
            contract={"features": tag, "outputs": f"head fitted on {tag} features only"})
        if tag == "F0":
            u2(f"{P}__U2__F0", X)
    # native certificate on the reserved cert rows (independent of the tree fit and of every attacker)
    cr = W["idx"]["cert"]
    cert = {"nominee": FO.certificate(m, H[cr], W["s"][cr], fare["certificate"]),
            "zero_fairness": FO.certificate(m0, H[cr], W["s"][cr], fare["certificate"]),
            "cert_rows": int(len(cr)), "cert_role": "20% of exposure-cleaned attacker_fit groups (oar-cert-v1)",
            "tree_own_task_accuracy_assessment": FO.own_task_accuracy(m, H[W["idx"]["assessment"]],
                                                                      W["t"][W["idx"]["assessment"]])}
    (S.RUN / "certificates").mkdir(parents=True, exist_ok=True)
    (S.RUN / "certificates" / f"{P}.json").write_text(json.dumps(cert, indent=1, default=str))
    log(f"[{ds}] s{k} FARE nominee {j} admissible={sel['admissible']}")
