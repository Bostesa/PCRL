"""Independent attacker slates for the refreshed guarded joint study (audit/baseline owner; never the training critics).

Roles. Every attacker is FITTED on AUDIT_FIT and SELECTED on INNER_SELECTION. Inner (selection) audits also report
their recovery on INNER_SELECTION (the selected attacker's AUC = the bank maximum; an optimistic selection statistic,
used identically for every candidate). Final audits refit the selected attackers at attacker seeds 0, 1, 2 and score
them on rows passed in by the caller (`score_idx`); only rgj.assess passes DEVELOPMENT_ASSESSMENT rows, and only after
it has verified a pushed EVALUATION_LOCK.json. This module never names or indexes that role itself.

Recovery metric. AUC of the score P(S = 1 | view) for binary SEX (fixed orientation: column 1 of predict_proba, never
flipped, an AUC below 0.5 stays below 0.5); macro one-vs-rest AUC over the supplied classes for the multiclass race
audit (score for class c = P(c | view)). Proper-loss recovery: cross-entropy (log loss, probabilities clipped at 1e-12).

Dual selection (final slate). The PRIMARY attacker of a view is the slate member with the highest INNER_SELECTION AUC;
a separate PROPER-LOSS attacker is the member with the lowest INNER_SELECTION cross-entropy (log-loss reporting only).
Ties go to the earlier candidate in bank order. Nothing is ever selected on, or re-oriented by, the scored rows.

Coalition bank. For the pair view [v1, v2] the candidates are, in this order:
  coalition          every slate member fitted on the pair view (its own joint attackers);
  ignore_recipient_2 every slate member fitted on v1 alone (the coalition may discard recipient 2's view);
  ignore_recipient_1 every slate member fitted on v2 alone.
Coalition recovery is the AUC (CE) of the bank's selection. Because the ignore candidates are in the bank, inner
coalition recovery is >= the better inner local recovery by construction; this is a property of the attack family,
not a clamp, and on scored rows the coalition estimate is never clamped to the local estimates. All tables (the
coalition's own and both ignore tables) are returned and archived even when an ignore candidate wins.

Slates (factories are the pinned predecessor helpers in jcv.audit, imported unchanged).
  INNER       LR(C=1) | MLP(64,64) | HGB(lr 0.1, 31 leaves, 200 iterations)          finite views add CC
  FINAL       the registered jcv final slate:
              LR      StandardScaler + LogisticRegression, C in {0.01, 0.1, 1, 10, 100}
              MLP     StandardScaler + MLP hidden in {(64,), (128,), (64,64), (128,128)}, adam, alpha 1e-4, early
                      stopping on a 10% split of AUDIT_FIT, max_iter 300
              HGB     HistGradientBoosting lr in {0.05, 0.1} x max_leaf_nodes in {15, 31}, 200 iterations
              DA      defense-aware canonical MLP (below)
              finite views (FARE cells, hard decisions) add CC = exact cell-conditional attacker, alpha in {0.1,1,10}
  SECONDARY   INNER + DA (+ CC on finite views): output-only views and the race stress audit.

What makes DA "defense-aware" (and why it is not white-box). DA is an ordinary supervised attacker trained on the
frozen released views with SEX labels on AUDIT_FIT. Its defense awareness is limited to knowledge of the release
STRUCTURE and the public deployed head: a recipient view is [r_i, centred logits_i] and the centred logits are an
exactly affine function of r_i through the public head (and, for the LEACE reference, one direction of r_i is
exactly collapsed by the public map). DA therefore canonicalises the view: centre, PCA fitted on AUDIT_FIT, drop
the structurally degenerate directions (explained variance <= 1e-9 x the largest; exact affine redundancy and
collapsed directions sit at float rounding level), and WHITEN the remaining directions to unit variance, which
restores low-variance (but present) directions to the scale of the others before an MLP(128,128). It uses no
encoder weights, gradients, internal states, raw permitted inputs, repeated queries or labels at inference. It
is therefore not a white-box or adaptive-query attacker. Limitation: a genuine signal whose variance ratio is below
1e-9 (sd ratio < 3.2e-5) is dropped by DA's canonicalisation; such a signal remains visible to the scale-invariant
members of the same slate (HGB quantile binning, StandardScaler + LR/MLP for coordinate-aligned signals), which the
tests check with an amplitude-1e-6 planted clue.

Controls (inner roles only: AUDIT_FIT -> INNER_SELECTION; `controls`). (a) Real-data shuffled-label null: SEX is
permuted within AUDIT_FIT and within INNER_SELECTION with a generator seeded by CONTROL_SEED; the permutation is drawn
(and hashed) before any attacker is fitted; the selected attacker's AUC must stay <= 0.55. (b) Planted leak: a copy of
the release view gets appended columns that are a declared transformation of SEX: PLANT_ONEHOT = one-hot of SEX with
20% of rows replaced by a uniformly random class (CONTROL_SEED stream); PLANT_TINY = 1e-6 x (that noisy SEX - 0.5),
an amplitude check. The selected attacker's AUC must exceed 0.75 for each. For the pair, the plant goes into the pair
copy only (v1/v2 stay clean), so detection has to come from the coalition's own joint attackers (the ignore candidates
see only clean views). Failure flags invalidate the corresponding audit; they are never dropped.
"""
from __future__ import annotations

import hashlib
import time
import warnings

import numpy as np
from sklearn.metrics import roc_auc_score

from jcv import audit as JA   # pinned predecessor factories (LR/MLP/HGB/DA/CellConditional), unchanged

warnings.filterwarnings("ignore")

ATT_SEEDS = (0, 1, 2)
CLIP = 1e-12
TIE = 1e-12
CONTROL_SEED = 20261014
NULL_MAX = 0.55
PLANT_MIN = 0.75
PLANT_NOISE = 0.20
PLANT_TINY_AMP = 1e-6
FIT_ROLE, SEL_ROLE = "AUDIT_FIT", "INNER_SELECTION"
PRIMARY_VIEWS = ("v1", "v2", "pair")
IGNORE_NAMES = ("ignore_recipient_2", "ignore_recipient_1")   # local_keys[0] keeps recipient 1 only, [1] recipient 2


# ------------------------------------------------------------------ slates
def _cc(finite):
    if not finite:
        return []
    return [(f"CC_alpha{a}", (lambda s, a=a: JA.CellConditional(a))) for a in (0.1, 1.0, 10.0)]


def inner_slate(finite=False):
    return JA.slate("inner") + _cc(finite)


def final_slate(finite=False):
    return JA.slate("final", finite)


def secondary_slate(finite=False):
    return JA.slate("inner") + [("DA_canonical_MLP", lambda s: JA._da(s))] + _cc(finite)


SLATES = {"inner": inner_slate, "final": final_slate, "secondary": secondary_slate}


# ------------------------------------------------------------------ metrics (fixed orientation)
def proba(m, X, K):
    return JA.proba(m, X, K)


def logloss(y, P):
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], CLIP, 1))))


def auc_fixed(y, P, classes=(0, 1)):
    """Binary: AUC of P[:, 1] for y == 1 (never flipped). Multiclass: macro OvR AUC of P[:, c] over `classes`."""
    classes = list(classes)
    if len(classes) == 2:
        return float(roc_auc_score(y == classes[1], P[:, classes[1]]))
    return float(np.mean([roc_auc_score(y == c, P[:, c]) for c in classes]))


# ------------------------------------------------------------------ core: fit a slate on one view, select a bank
def _finite_for(finite, w):
    return bool(finite.get(w, False)) if isinstance(finite, dict) else bool(finite)


def _fit_table(X, y, fit_idx, sel_idx, slate, K, classes):
    rows, models = [], {}
    for order, (name, fac) in enumerate(slate):
        t0 = time.time()
        m = fac(0).fit(X[fit_idx], y[fit_idx])
        P = proba(m, X[sel_idx], K)
        rows.append({"attacker": name, "order": order, "inner_auc": auc_fixed(y[sel_idx], P, classes),
                     "inner_ce": logloss(y[sel_idx], P), "fit_s": round(time.time() - t0, 3)})
        models[name] = (fac, m)
    return rows, models


def _pick(cands, key, maximize):
    best = None
    for j, c in enumerate(cands):
        v = c[key]
        if best is None or (v > cands[best][key] + TIE if maximize else v < cands[best][key] - TIE):
            best = j
    return best


def select_bank(V, y, fit_idx, sel_idx, slate_fn, finite=False, K=2, classes=(0, 1), coalition_key="pair",
                local_keys=("v1", "v2")):
    """Fit `slate_fn` on every view in V (fit_idx), score on sel_idx, and select per view by AUC (primary) and CE.

    Views other than `coalition_key` are local views. If coalition_key and both local_keys are in V, the coalition
    bank = own pair attackers + ignore-recipient candidates (see module docstring); local_keys must be ordered
    (recipient-1 view, recipient-2 view), e.g. ("v1", "v2"), ("p1", "p2"), ("h1", "h2"). Returns (record, models) where
    models[(view, attacker)] = (factory, seed-0 model) for refits.
    """
    tables, models = {}, {}
    for w, X in V.items():
        assert len(X) == len(y), f"view {w} has {len(X)} rows, labels have {len(y)}"
        rows, ms = _fit_table(np.asarray(X), y, fit_idx, sel_idx, slate_fn(_finite_for(finite, w)), K, classes)
        tables[w] = rows
        models.update({(w, nm): m for nm, m in ms.items()})
    sel = {}
    for w in V:
        if w == coalition_key and all(lk in V for lk in local_keys):
            bank = [{"candidate": "coalition", "view": w, **r} for r in tables[w]]
            for lk, cname in zip(local_keys, IGNORE_NAMES):
                bank += [{"candidate": cname, "view": lk, **r} for r in tables[lk]]
        else:
            bank = [{"candidate": "own", "view": w, **r} for r in tables[w]]
        ia, ic = _pick(bank, "inner_auc", True), _pick(bank, "inner_ce", False)
        sel[w] = {"auc": {**_brief(bank[ia]), "bank_index": ia}, "ce": {**_brief(bank[ic]), "bank_index": ic}}
        if w == coalition_key and len(bank) > len(tables[w]):
            sel[w]["bank"] = [_brief(b) for b in bank]
    rec = {"tables": tables, "selection": sel, "n_fit": int(len(fit_idx)), "n_select": int(len(sel_idx)),
           "classes": [int(c) for c in classes]}
    return rec, models


def _brief(b):
    return {"candidate": b["candidate"], "view": b["view"], "attacker": b["attacker"],
            "inner_auc": b["inner_auc"], "inner_ce": b["inner_ce"]}


def _label(sel):
    return f"{sel['candidate']}:{sel['view']}:{sel['attacker']}"


def _views(V, views):
    keys = views or [w for w in ("v1", "v2", "pair", "v") if w in V]
    return {w: np.asarray(V[w]) for w in keys}


# ------------------------------------------------------------------ INNER (selection) audit
def inner_audit(V, D, finite=False, views=None, y=None, slate="inner"):
    """Inner audit of one frozen release (AUDIT_FIT -> INNER_SELECTION; never assessment rows).

    V: dict of row-aligned views over all D rows, e.g. rgj.finalize.views_from_release(z) (keys v1, v2, pair; other
       keys such as "out" are ignored). Single-view mode for per-purpose FARE selection: V = {"v": X}.
    finite: bool or {view: bool}; finite views add the exact cell-conditional attacker.
    Returns a JSON-serialisable dict:
      auc[w]          AUC of the INNER_SELECTION-AUC-selected attacker (local views; coalition bank for "pair")
      ce[w]           cross-entropy of the separately CE-selected attacker
      selected[w]     "<candidate>:<view>:<attacker>" of the AUC selection; ce_selected[w] likewise
      worse_local / mean_local / coalition_minus_best_local   (when v1, v2 [and pair] are present)
      tables, selection (incl. the full coalition bank), n_fit, n_select, wall_s
    """
    t0 = time.time()
    VV = _views(V, views)
    yy = D["sex"] if y is None else y
    rec, _ = select_bank(VV, yy, D["idx"][FIT_ROLE], D["idx"][SEL_ROLE], SLATES[slate], finite)
    out = {"auc": {}, "ce": {}, "selected": {}, "ce_selected": {}}
    for w, s in rec["selection"].items():
        out["auc"][w], out["ce"][w] = s["auc"]["inner_auc"], s["ce"]["inner_ce"]
        out["selected"][w], out["ce_selected"][w] = _label(s["auc"]), _label(s["ce"])
    if "v1" in out["auc"] and "v2" in out["auc"]:
        out["worse_local"] = max(out["auc"]["v1"], out["auc"]["v2"])
        out["mean_local"] = (out["auc"]["v1"] + out["auc"]["v2"]) / 2
        if "pair" in out["auc"]:
            out["coalition_minus_best_local"] = out["auc"]["pair"] - out["worse_local"]
    out.update({"slate": slate, "finite": finite if isinstance(finite, bool) else dict(finite),
                "roles": {"fit": FIT_ROLE, "select_and_score": SEL_ROLE},
                "orientation": "P(S=1), fixed; never flipped", **rec, "wall_s": round(time.time() - t0, 2)})
    return out


def inner_local(X, D, finite=False, slate="inner"):
    """Single-view inner audit (e.g. one FARE purpose): returns (AUC of the AUC-selected attacker, full record)."""
    r = inner_audit({"v": X}, D, finite=finite, slate=slate)
    return r["auc"]["v"], r


# ------------------------------------------------------------------ FINAL slate (dual selection + seed refits)
def final_audit(V, y, fit_idx, sel_idx, score_idx, slate_fn=final_slate, finite=False, K=2, classes=(0, 1),
                coalition_key="pair", local_keys=("v1", "v2"), seeds=ATT_SEEDS):
    """Fit on fit_idx, dual-select on sel_idx, refit each selected attacker at `seeds`, score on score_idx.

    score_idx must be disjoint from fit_idx and sel_idx. Returns (record, probs) with probs[f"{crit}_{w}"] an array
    (len(seeds), len(score_idx), K) of per-row probabilities for crit in {"auc", "ce"} and each view w. The record
    holds every slate table and the coalition bank, the selections with their inner values, and the scored-row AUC/CE
    per seed (descriptive; the lead's inference recomputes endpoints from the saved probabilities).
    """
    fit_idx, sel_idx, score_idx = (np.asarray(a) for a in (fit_idx, sel_idx, score_idx))
    assert not np.intersect1d(score_idx, fit_idx).size and not np.intersect1d(score_idx, sel_idx).size, \
        "scored rows overlap the attacker fitting or selection rows"
    t0 = time.time()
    rec, models = select_bank(V, y, fit_idx, sel_idx, slate_fn, finite, K, classes, coalition_key, local_keys)
    cache, probs, scored = {}, {}, {}
    for w, s in rec["selection"].items():
        scored[w] = {}
        for crit in ("auc", "ce"):
            src, att = s[crit]["view"], s[crit]["attacker"]
            Ps = []
            for sd in seeds:
                key = (src, att, sd)
                if key not in cache:
                    fac, m0 = models[(src, att)]
                    m = m0 if sd == 0 else fac(sd).fit(np.asarray(V[src])[fit_idx], y[fit_idx])
                    cache[key] = proba(m, np.asarray(V[src])[score_idx], K)
                Ps.append(cache[key])
            P = np.stack(Ps)
            probs[f"{crit}_{w}"] = P
            ys = y[score_idx]
            scored[w][crit] = {"source_view": src, "attacker": att, "candidate": s[crit]["candidate"],
                               "seeds": list(seeds), "auc_per_seed": [auc_fixed(ys, p, classes) for p in P],
                               "ce_per_seed": [logloss(ys, p) for p in P]}
            scored[w][crit]["auc_mean"] = float(np.mean(scored[w][crit]["auc_per_seed"]))
            scored[w][crit]["ce_mean"] = float(np.mean(scored[w][crit]["ce_per_seed"]))
    rec.update({"scored": scored, "n_score": int(len(score_idx)), "refits": len(cache),
                "orientation": "P(class) columns, fixed; never flipped on scored rows",
                "wall_s": round(time.time() - t0, 2)})
    return rec, probs


# ------------------------------------------------------------------ controls (inner roles only)
def frozen_permutation(S, D, seed=CONTROL_SEED):
    """SEX permuted within AUDIT_FIT and within INNER_SELECTION (drawn before any fit). Returns (S_perm, sha256)."""
    rng = np.random.default_rng(seed)
    Sp = S.copy()
    for r in (FIT_ROLE, SEL_ROLE):
        ix = D["idx"][r]
        Sp[ix] = S[ix][rng.permutation(len(ix))]
    return Sp, hashlib.sha256(np.ascontiguousarray(Sp[np.concatenate([D["idx"][FIT_ROLE], D["idx"][SEL_ROLE]])])
                              .astype(np.int64).tobytes()).hexdigest()


def planted_columns(S, seed=CONTROL_SEED):
    """Declared transformations of SEX: (one-hot of noisy SEX, tiny-amplitude noisy SEX column), noisy share 20%."""
    rng = np.random.default_rng(seed + 1)
    noisy = S.copy()
    flip = rng.random(len(S)) < PLANT_NOISE
    noisy[flip] = rng.integers(0, 2, int(flip.sum()))
    return {"PLANT_ONEHOT": np.eye(2)[noisy], "PLANT_TINY": PLANT_TINY_AMP * (noisy[:, None] - 0.5)}


def controls(V, D, finite=False, slate="final", views=None):
    """Shuffled-label null and planted-leak controls on one release (inner roles only). JSON-serialisable."""
    t0 = time.time()
    VV = _views(V, views)
    S = D["sex"]
    Sp, perm_sha = frozen_permutation(S, D)       # frozen before fitting
    plants = planted_columns(S)
    f, v = D["idx"][FIT_ROLE], D["idx"][SEL_ROLE]
    out = {"slate": slate, "control_seed": CONTROL_SEED, "permutation_sha256": perm_sha,
           "plant_definition": {"PLANT_ONEHOT": "one-hot(noisy SEX), noisy = SEX with 20% rows replaced by a random class",
                                "PLANT_TINY": f"{PLANT_TINY_AMP} * (noisy SEX - 0.5), one column"},
           "thresholds": {"null_max": NULL_MAX, "plant_min": PLANT_MIN}, "views": {}}
    null_rec, _ = select_bank(VV, Sp, f, v, SLATES[slate], finite)
    for w in VV:
        out["views"][w] = {"null_auc": null_rec["selection"][w]["auc"]["inner_auc"],
                           "null_selected": _label(null_rec["selection"][w]["auc"])}
    for pname, col in plants.items():
        for w in VV:
            Vp = dict(VV)
            Vp[w] = np.hstack([VV[w], col])
            sub = {w: Vp[w]} if w != "pair" else Vp     # pair: bank with clean local copies
            r, _ = select_bank(sub, S, f, v, SLATES[slate], finite)
            out["views"][w][f"{pname}_auc"] = r["selection"][w]["auc"]["inner_auc"]
            out["views"][w][f"{pname}_selected"] = _label(r["selection"][w]["auc"])
    for w, d in out["views"].items():
        d["null_ok"] = d["null_auc"] <= NULL_MAX
        d["plant_ok"] = all(d[f"{p}_auc"] > PLANT_MIN for p in plants)
    out["all_ok"] = all(d["null_ok"] and d["plant_ok"] for d in out["views"].values())
    out["wall_s"] = round(time.time() - t0, 1)
    return out


# ------------------------------------------------------------------ helpers for predecessor/test releases
def align_release(z, D, keys=None):
    """Rows of a release npz (with row_id) re-ordered to D's rows by row_id; raises if any D row is missing."""
    rid = np.asarray(z["row_id"])
    pos = {int(r): j for j, r in enumerate(rid)}
    try:
        take = np.array([pos[int(r)] for r in D["row_id"]])
    except KeyError as e:
        raise ValueError(f"release lacks D row_id {e}") from None
    return {k: np.asarray(z[k])[take] for k in (keys or z.files) if k != "row_id"} | {"row_id": rid[take]}


def main(argv=None):
    """python -m rgj.audit controls --unit <name> [--units-dir DIR] [--slate final|inner] [--finite]"""
    import argparse
    import json
    from pathlib import Path

    from rgj import data as DA
    from rgj import finalize as FN
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("controls", "inner"))
    ap.add_argument("--unit", required=True)
    ap.add_argument("--units-dir", default=str(Path.home() / "PCRL_eval_cache_private" / "rgj_v1" / "run" / "units"))
    ap.add_argument("--slate", default="final")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    D = DA.load()
    d = Path(a.units_dir) / a.unit
    if not FN.unit_complete(d):
        raise SystemExit(f"REFUSED: {d.name} is not hash-complete")
    z = align_release(np.load(d / "release.npz"), D)
    V = FN.views_from_release(z)
    res = controls(V, D, slate=a.slate) if a.cmd == "controls" else inner_audit(V, D)
    txt = json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    if a.out:
        Path(a.out).write_text(txt)
    print(txt if a.cmd == "controls" else json.dumps({k: res[k] for k in ("auc", "selected", "wall_s")}, indent=1))


if __name__ == "__main__":
    main()
