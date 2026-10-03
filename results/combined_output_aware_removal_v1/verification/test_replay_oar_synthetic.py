"""Synthetic self-test of replay_oar.py.

Builds a complete synthetic study tree in the real layouts (benchmark inputs, oar run units, selection / certificate /
alias / control files, benchmark units for the corrections and the exposure sensitivity, runner tables), computes the
"runner" values with code written separately here (bincount-based weighted AUC, replicate-by-replicate bootstrap),
checks that the replay reproduces a clean tree with no failure, then plants defects one at a time and confirms each
one is caught by the intended check.

    OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python results/combined_output_aware_removal_v1/verification/test_replay_oar_synthetic.py \
        [--scratch DIR] [--out results/combined_output_aware_removal_v1/verification/synthetic_selftest_result.json]
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG_REAL = HERE.parent
sys.path.insert(0, str(HERE))
import replay_oar as R  # noqa: E402  (the replay under test)

B_PRI, B_EXP, B_COR = 1000, 600, 600
EXPECT_EXPOSURE = {"adult": [6, 3], "hmda": [6, 3]}   # exposed test rows (all roles), exposed assessment rows
SEED_PRI, SEED_EXP, SEED_COR = 20261011, 20261004, 20261003
GRID_IDS = (1, 2, 3, 4, 5, 6)
SIG = {"adult": 2.0, "hmda": 4.0}
DS = {"adult": {"K_s": 2, "attr": "sex", "task": "task_income", "logits": "logits_income_prediction",
                "purpose": "income_prediction", "sup": [0, 1]},
      "hmda": {"K_s": 5, "attr": "race", "task": "task_loan_decision", "logits": "logits_underwriting",
               "purpose": "underwriting", "sup": [0, 1, 2]}}


# ------------------------------------------------------------------------------------------------ small utilities
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def arr_sha(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def u01(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) / 2 ** 32


def complete(d: Path):
    d = Path(d)
    files = {str(p.relative_to(d)): sha(p) for p in sorted(d.rglob("*")) if p.is_file() and p.name != "COMPLETE.json"}
    (d / "COMPLETE.json").write_text(json.dumps({"files": files}, indent=1))


def prep(score):
    _, inv = np.unique(score, return_inverse=True)
    return inv, int(inv.max()) + 1


def wauc_one(score, pos, w, pre=None):
    """Weighted AUC (ties 1/2) via unique-score bincounts; separate algorithm from the replay."""
    inv, U = prep(score) if pre is None else pre
    gp = np.bincount(inv[pos], weights=w[pos], minlength=U)
    gn = np.bincount(inv[~pos], weights=w[~pos], minlength=U)
    below = np.cumsum(gn) - gn
    den = gp.sum() * gn.sum()
    return float((gp * (below + 0.5 * gn)).sum() / den) if den > 0 else float("nan")


def boot_weights(assess_unit, B, seed):
    _, inv = np.unique(assess_unit, return_inverse=True)
    n = inv.max() + 1
    rng = np.random.default_rng(seed)
    p = np.full(n, 1.0 / n)
    for _ in range(B):
        yield rng.multinomial(n, p)[inv].astype(float)


def softmax(Z):
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def make_P(rng, y, K, beta):
    Z = beta * np.eye(K)[y] + rng.normal(size=(len(y), K))
    return softmax(Z)


# ------------------------------------------------------------------------------------------------ world
def build_inputs(root: Path, rng):
    inp = root / "bench" / "inputs"
    inp.mkdir(parents=True)
    worlds = {}
    for ds, c in DS.items():
        n_df, n_af, n_av, n_as = 700, 260, 150, 420
        N = n_df + n_af + n_av + n_as
        role = np.array(["defense_fit"] * n_df + ["attacker_fit"] * n_af + ["attacker_val"] * n_av +
                        ["assessment"] * n_as, dtype="<U12")
        split = np.where(role == "defense_fit", "train", "test").astype("<U5")
        row_id = np.arange(1000, 1000 + N, dtype=np.int64)[rng.permutation(N)]
        ck = np.array([f"{ds}k{i:05d}" for i in range(N)], dtype="<U20")
        unit = np.arange(N, dtype=np.int64)
        # exposure: some test rows duplicate a train record key (2 afit, 1 aval, 3 assessment)
        tr = np.flatnonzero(role == "defense_fit")
        for j, r in enumerate([n_df + 1, n_df + 2, n_df + n_af + 3, n_df + n_af + n_av + 5, n_df + n_af + n_av + 9,
                               n_df + n_af + n_av + 30]):
            ck[r] = ck[tr[j]]
        # a few multi-row record groups inside assessment / attacker_fit (same key, same unit)
        for r in (n_df + n_af + n_av + 50, n_df + n_af + n_av + 80, n_df + 40):
            ck[r + 1] = ck[r]
            unit[r + 1] = unit[r]
        if c["K_s"] == 2:
            s = (rng.random(N) < 0.35).astype(np.int64)
        else:
            s = rng.choice(5, size=N, p=[0.55, 0.15, 0.27, 0.02, 0.01]).astype(np.int64)
        t = (rng.random(N) < 0.4).astype(np.int64)
        lab = {"row_id": row_id, "split": split, "unit": unit, "record_key": ck.copy(), "canon_key": ck, "role": role,
               c["attr"]: s, c["task"]: t}
        np.savez(inp / f"{ds}_labels.npz", **lab)
        for k in range(3):
            H = rng.normal(size=(N, 4)) + 0.8 * np.eye(4)[s % 4] + 0.6 * t[:, None]
            O = np.stack([-(H[:, 0] + t), H[:, 1] + t * 1.5], 1) + rng.normal(scale=0.3, size=(N, 2))
            np.savez(inp / f"{ds}_s{k}_forward.npz", row_id=row_id, split=split, rep_p0=H, **{c["logits"]: O})
        worlds[ds] = lab
    (inp / "INPUTS_INDEX.json").write_text("{}")
    return worlds


def roles_of(ds, lab, thr):
    """Generator-side roles (separate code from the replay)."""
    role0, ck = lab["role"].astype(str), lab["canon_key"].astype(str)
    tk = set(ck[lab["split"] == "train"])
    excl = np.array([(sp == "test" and k in tk) for sp, k in zip(lab["split"], ck)]) & (role0 != "defense_fit")
    rr = {}
    rr["defense_fit"] = role0 == "defense_fit"
    af = (role0 == "attacker_fit") & ~excl
    ch = np.array([u01("oar-cert-v1|" + ds + "|" + k) < 0.2 for k in ck])
    rr["cert"], rr["attacker_fit"] = af & ch, af & ~ch
    rr["attacker_val"] = (role0 == "attacker_val") & ~excl
    rr["assessment"] = (role0 == "assessment") & ~excl
    hh = np.array([u01("oar-head-v1|" + ds + "|" + k) < 0.2 for k in ck])
    rr["head_val"] = rr["defense_fit"] & hh
    rr["head_fit"] = rr["defense_fit"] & ~hh
    idx = {k: np.flatnonzero(v) for k, v in rr.items()}
    return idx, excl


def ras_entry(ds, lab, idx, excl, thr):
    c = DS[ds]
    out = {"roles": {}}
    for r, ix in idx.items():
        out["roles"][r] = {"rows": int(len(ix)), "groups": int(len(np.unique(lab["unit"][ix]))),
                           "row_ids_sha256": arr_sha(np.sort(lab["row_id"][ix]).astype(np.int64))}
    out["roles"]["excluded_exposure_rows"] = int(excl.sum())
    sup = {}
    for what, y, K in (("sensitive", lab[c["attr"]], c["K_s"]), ("task", lab[c["task"]], 2)):
        counts = {r: np.bincount(y[idx[r]], minlength=K).tolist() for r in ("defense_fit", "cert", *thr)}
        ok = [k for k in range(K) if all(counts[r][k] >= thr[r] for r in thr)]
        sup[what] = {"K": K, "counts": counts, "thresholds": thr, "supported_classes": ok,
                     "supported_pairs": [[i, j] for i in ok for j in ok if i < j],
                     "not_estimable_classes": [k for k in range(K) if k not in ok]}
    out["support"] = sup
    out["exposure_groups_removed"] = int(len(np.unique(lab["unit"][excl])))
    out["assessment_identical_for_all_methods_and_seeds"] = True
    return out


# ------------------------------------------------------------------------------------------------ oar run tree
class Gen:
    def __init__(self, root, rng, worlds, thr):
        self.root, self.rng, self.W, self.thr = root, rng, worlds, thr
        self.run = root / "run"
        self.units = self.run / "units"
        self.units.mkdir(parents=True)
        self.idx, self.excl = {}, {}
        for ds, lab in worlds.items():
            self.idx[ds], self.excl[ds] = roles_of(ds, lab, thr)

    def _base(self, ds):
        lab, idx = self.W[ds], self.idx[ds]
        a, v = idx["assessment"], idx["attacker_val"]
        return lab, a, v

    def attack(self, ds, uid, beta, contract, plus=None, vll=None):
        lab, a, v = self._base(ds)
        K = DS[ds]["K_s"]
        s = lab[DS[ds]["attr"]]
        d = self.units / uid
        (d / "models").mkdir(parents=True)
        preds = {"assess_row_id": lab["row_id"][a], "assess_unit": lab["unit"][a], "y_s": s[a]}
        for k in range(3):
            preds[f"P__NL__as{k}"] = make_P(self.rng, s[a], K, beta + 0.05 * k)
        preds["P__L"] = make_P(self.rng, s[a], K, beta * 0.8)
        val = {"val_row_id": lab["row_id"][v], "val_y_s": s[v], "VAL__NL__as0": make_P(self.rng, s[v], K, beta),
               "VAL__L": make_P(self.rng, s[v], K, beta * 0.8)}
        np.savez_compressed(d / "preds.npz", **preds)
        np.savez_compressed(d / "val_preds.npz", **val)
        (d / "models" / "NL_as0.joblib").write_bytes(b"synthetic")
        vll = float(self.rng.uniform(0.4, 0.7)) if vll is None else vll
        rec = {"id": uid, "nl_family": "GBT", "val_log_loss": {"NL": vll, "L": vll + 0.01},
               "contract": contract, "selection_role": "attacker_val", "fit_role": "attacker_fit"}
        if plus is not None:
            rp, op, pref = plus
            rv = json.loads((self.units / rp / "record.json").read_text())["val_log_loss"]["NL"]
            ov = json.loads((self.units / op / "record.json").read_text())["val_log_loss"]["NL"]
            own = (max(ov, rv) + 0.05) if pref else (min(ov, rv) - 0.05)
            rec["val_log_loss"]["NL"] = own
            cand = {"GBT/MLP": own, "ignore_rep": ov, "ignore_out": rv}
            best = None
            for nm in ("GBT/MLP", "ignore_rep", "ignore_out"):
                if best is None or cand[nm] < cand[best]:
                    best = nm
            rec["plus_selection"] = {"candidates_attacker_val_log_loss": cand, "selected": best,
                                     "alias_source": {"ignore_rep": op, "ignore_out": rp}.get(best)}
        (d / "record.json").write_text(json.dumps(rec, indent=1))
        complete(d)

    def u2(self, ds, uid, acc_extra, base_correct):
        lab, a, v = self._base(ds)
        t = lab[DS[ds]["task"]]
        d = self.units / uid
        (d / "models").mkdir(parents=True)

        def P_for(rows, bc):
            correct = bc.copy()
            flip = (~correct) & (self.rng.random(len(rows)) < acc_extra)
            correct |= flip
            pred = np.where(correct, t[rows], 1 - t[rows])
            P = np.full((len(rows), 2), 0.3)
            P[np.arange(len(rows)), pred] = 0.7
            return P
        Pa = P_for(a, base_correct["a"])
        Pv = P_for(v, base_correct["v"])
        np.savez_compressed(d / "preds.npz", assess_row_id=lab["row_id"][a], assess_unit=lab["unit"][a], y_t=t[a],
                            U2_P=Pa)
        np.savez_compressed(d / "val_preds.npz", val_row_id=lab["row_id"][v], val_y_t=t[v], VAL__U2_P=Pv)
        (d / "models" / "U2.joblib").write_bytes(b"synthetic")
        (d / "record.json").write_text(json.dumps({"id": uid, "val_accuracy": float((Pv.argmax(1) == t[v]).mean())}))
        complete(d)

    def head(self, ds, uid, X):
        import joblib
        from sklearn.linear_model import LogisticRegression
        lab, a, _ = self._base(ds)
        t = lab[DS[ds]["task"]]
        hf = self.idx[ds]["head_fit"]
        m = LogisticRegression(C=1.0, max_iter=500).fit(X[hf], t[hf])
        P = np.zeros((X.shape[0], 2))
        P[:, m.classes_.astype(int)] = m.predict_proba(X)
        Oh = np.log(np.clip(P, 1e-12, 1.0))
        d = self.units / uid
        (d / "models").mkdir(parents=True)
        joblib.dump(m, d / "models" / "head.joblib")
        Pa = np.exp(Oh[a])
        np.savez_compressed(d / "preds.npz", head_outputs_all=Oh, row_id=lab["row_id"], assess_row_id=lab["row_id"][a],
                            assess_unit=lab["unit"][a], y_t=t[a])
        np.savez_compressed(d / "val_preds.npz")
        rec = {"id": uid, "fit_role": "defense_fit minus oar-head-v1 holdout",
               "select_role": "defense_fit holdout (oar-head-v1)", "inputs_at_runtime": "protected features only",
               "assessment_accuracy": float((Pa.argmax(1) == t[a]).mean()),
               "assessment_log_loss": float(-np.mean(np.log(np.clip(Pa[np.arange(len(a)), t[a]], 1e-12, 1))))}
        (d / "record.json").write_text(json.dumps(rec, indent=1))
        complete(d)
        return Oh

    def farefit(self, ds, k, uid, cells, H):
        lab = self.W[ds]
        df = self.idx[ds]["defense_fit"]
        s, t = lab[DS[ds]["attr"]], lab[DS[ds]["task"]]
        d = self.units / uid
        (d / "model").mkdir(parents=True)
        kc = int(cells.max()) + 1
        gcodes = sorted(np.unique(s[df]).tolist())
        gc = np.zeros((kc, len(gcodes)), int)
        for i, g in enumerate(gcodes):
            gc[:, i] = np.bincount(cells[df][s[df] == g], minlength=kc)
        (d / "model" / "model.json").write_text(json.dumps({"cell_group_counts": gc.tolist(), "group_codes": gcodes}))
        np.save(d / "cells.npy", cells.astype(np.int32))
        rec = {"uid": uid, "n_cells": kc, "n_fit": int(len(df)), "n_all": int(len(cells)),
               "fit_rows_sha256": arr_sha(H[df]), "fit_labels_sha256": arr_sha(t[df].astype(np.int64),
                                                                                s[df].astype(np.int64))}
        (d / "rec.json").write_text(json.dumps(rec, indent=1))
        complete(d)


def cells_for(H, i, k):
    q = {1: 8, 2: 4, 3: 4, 4: 3, 5: 2, 6: 2}[i]
    col = {1: 0, 2: 1, 3: 1, 4: 2, 5: 3, 6: 0}[i]
    edges = np.quantile(H[:, col], np.linspace(0, 1, q + 1)[1:-1])
    return np.searchsorted(edges, H[:, col]).astype(np.int32)


def cert_record(gen, ds, k, fu, groups_sel, ccfg):
    lab = gen.W[ds]
    cr = gen.idx[ds]["cert"]
    s = lab[DS[ds]["attr"]][cr]
    model = json.loads((gen.units / fu / "model" / "model.json").read_text())
    gc, gcodes = np.asarray(model["cell_group_counts"]), model["group_codes"]
    kc = gc.shape[0]
    cells = np.load(gen.units / fu / "cells.npy")[cr]
    grp = sorted(groups_sel or gcodes)
    keep = np.isin(s, grp)
    cc, ss = cells[keep], s[keep]
    n = len(ss)
    perm = np.random.RandomState(ccfg["split_seed"]).permutation(n)
    nv = int(round(ccfg["val_fraction"] * n))
    va, te = perm[:nv], perm[nv:]
    pairs = []
    import itertools
    for gi, gj in itertools.combinations(grp, 2):
        miss = {"base": int(kc - np.count_nonzero(gc[:, gcodes.index(gi)] + gc[:, gcodes.index(gj)]))}
        for nm, ix in (("val", va), ("test", te)):
            m = (ss[ix] == gi) | (ss[ix] == gj)
            miss[nm] = int(kc - len(np.unique(cc[ix][m])))
        st = "UNAVAILABLE" if any(miss.values()) else "OK"
        pairs.append({"groups": [gi, gj], "cells_missing": miss, "status": st, "ub": 0.3 if st == "OK" else None})
    ok = all(p["status"] == "OK" for p in pairs)
    prem = [{"name": "cert_rows_disjoint_from_fit_rows", "holds": True, "how": "checked"},
            {"name": "every_cell_present_in_base_val_test_for_every_pair", "holds": ok, "how": "checked"},
            {"name": "bound_finite", "holds": ok, "how": "checked"},
            {"name": "rows_iid_from_target_distribution", "holds": None, "how": "stated"}]
    return {"status": "OK" if ok else "UNAVAILABLE", "reason": None if ok else "cell missing", "bound": 0.3 if ok else None,
            "pairs": pairs, "premises": prem}


def build_run(root, rng, worlds, thr, lock):
    g = Gen(root, rng, worlds, thr)
    inp = root / "bench" / "inputs"
    nominees = {}
    for ds, c in DS.items():
        lab = worlds[ds]
        idx = g.idx[ds]
        t = lab[c["task"]]
        a, v = idx["assessment"], idx["attacker_val"]
        for k in range(3):
            P = f"{ds}__s{k}"
            F = np.load(inp / f"{ds}_s{k}_forward.npz")
            H, O = F["rep_p0"], F[c["logits"]]
            # alias report
            Z = O - O.max(1, keepdims=True)
            Pr = np.exp(Z) / np.exp(Z).sum(1, keepdims=True)
            lp = np.log(np.clip(Pr, 1e-300, 1))
            (g.run / "aliases").mkdir(parents=True, exist_ok=True)
            (g.run / "aliases" / f"{P}.json").write_text(json.dumps({
                "K": 2, "prob_determines_centred_logits_maxabs": float(np.max(np.abs((lp - lp.mean(1, keepdims=True)) - (O - O.mean(1, keepdims=True))))),
                "logit_sum_sd": float(np.std(O.sum(1))), "logit_sum_is_constant": bool(np.std(O.sum(1)) < 1e-9),
                "full_vs_prob": "x", "hard_is_function_of_prob": True}))
            d = g.units / f"{P}__REF"
            (d / "models").mkdir(parents=True)
            np.savez_compressed(d / "preds.npz", assess_row_id=lab["row_id"][a], assess_unit=lab["unit"][a],
                                y_s=lab[c["attr"]][a])
            np.savez_compressed(d / "val_preds.npz")
            (d / "record.json").write_text("{}")
            complete(d)
            g.attack(ds, f"{P}__O_full", 1.8, {"outputs": "full", "features": None})
            g.attack(ds, f"{P}__O_prob", 1.7, {"outputs": "prob", "features": None})
            g.attack(ds, f"{P}__O_hard", 0.4, {"outputs": "hard", "features": None})
            base = {"a": rng.random(len(a)) < 0.75, "v": rng.random(len(v)) < 0.75}
            # B: numpy LEACE map written into the benchmark defenses tree
            mapd = root / "bench" / "defenses" / f"{ds}__s{k}__{c['purpose']}__B_{c['attr']}" / "map"
            mapd.mkdir(parents=True)
            pl, pr, mx = rng.normal(size=(4, 1)), rng.normal(size=(1, 4)) * 0.3, rng.normal(size=4)
            np.savez(mapd / "leace_map.npz", proj_left=pl, proj_right=pr, mean_x=mx)
            XB = H - ((H - mx) @ pr.T) @ pl.T
            feats = {"A": H, "B": XB, "C": XB}
            for arm, beta in (("A", 1.5), ("B", 1.2), ("C", 1.1)):
                g.attack(ds, f"{P}__{arm}__rep", beta, {"features": arm, "outputs": None})
                g.attack(ds, f"{P}__{arm}__rep+clean", beta + 0.3, {"features": arm, "outputs": "clean"},
                         plus=(f"{P}__{arm}__rep", f"{P}__O_full", None))
                g.u2(ds, f"{P}__U2__{arm}", 0.0, base)
                if arm in ("A", "B"):
                    g.head(ds, f"{P}__HEAD__{arm}", feats[arm])
                    g.attack(ds, f"{P}__O_head{arm}", 1.0, {"outputs": f"head on {arm}", "features": None})
                    pref = "ignore_out" if (arm == "B" and k == 1) else None
                    g.attack(ds, f"{P}__{arm}__rep+head", beta + 0.1, {"features": arm, "outputs": "head"},
                             plus=(f"{P}__{arm}__rep", f"{P}__O_head{arm}", pref))
            for rs in range(3):
                q = f"{P}__D_rs{rs}"
                g.attack(ds, f"{q}__rep", 0.3, {"features": f"noise sigma={SIG[ds]} rs={rs}", "outputs": None})
                for vv in ("full", "prob", "hard"):
                    g.attack(ds, f"{q}__rep+{vv}", 1.0, {"features": "noise", "outputs": vv},
                             plus=(f"{q}__rep", f"{P}__O_{vv}", "ignore_rep"))
                g.u2(ds, f"{P}__U2__D_rs{rs}", 0.0, {"a": base["a"] & (rng.random(len(a)) < 0.8),
                                                      "v": base["v"] & (rng.random(len(v)) < 0.8)})
            # FARE grid
            seen, alias, cells_of = [], {}, {}
            strengths = {1: 0.9, 2: 0.35, 3: 0.35, 4: 0.25, 5: 0.15, 6: 0.2}
            accx = {1: 0.35, 2: 0.30, 3: 0.30, 4: 0.25, 5: -1, 6: 0.28}
            for i in GRID_IDS:
                cells = cells_for(H, i, k)
                cells_of[i] = cells
                g.farefit(ds, k, f"{P}__FAREFIT_c{i}", cells, H)
                same = [j for j, cj in seen if np.array_equal(cj, cells)]
                if same:
                    alias[i] = same[0]
                    continue
                seen.append((i, cells))
                g.attack(ds, f"{P}__Fc{i}__rep", strengths[i] + 0.02 * k, {"features": f"FARE grid id {i}",
                                                                           "outputs": None})
                if accx[i] < 0:   # task-inferior configuration: drops 20 % of correct rows
                    bb = {"a": base["a"] & (rng.random(len(a)) < 0.8), "v": base["v"] & (rng.random(len(v)) < 0.8)}
                    g.u2(ds, f"{P}__U2__Fc{i}", 0.0, bb)
                else:
                    g.u2(ds, f"{P}__U2__Fc{i}", accx[i], base)
            # nominee by the frozen rule (generator-side code)
            accA = json.loads((g.units / f"{P}__U2__A" / "record.json").read_text())["val_accuracy"]
            rows = []
            for i in GRID_IDS:
                src = alias.get(i, i)
                acc = json.loads((g.units / f"{P}__U2__Fc{src}" / "record.json").read_text())["val_accuracy"]
                z = np.load(g.units / f"{P}__Fc{src}__rep" / "val_preds.npz")
                y, Pv = z["val_y_s"], z["VAL__NL__as0"]
                auc = float(np.mean([wauc_one(Pv[:, cc], y == cc, np.ones(len(y))) for cc in c["sup"]]))
                rows.append({"config": i, "alias_of": alias.get(i), "val_u2_accuracy": acc, "val_nl_macro_auc": auc,
                             "admissible": acc >= accA - 0.01})
            adm = [r for r in rows if r["admissible"]]
            best = min(adm, key=lambda r: (round(r["val_nl_macro_auc"], 12), r["config"]))
            nom, src = best["config"], alias.get(best["config"], best["config"])
            sel = {"untreated_val_u2_accuracy": accA, "cap": 0.01, "table": rows, "nominee": nom,
                   "nominee_unit_source": src, "admissible": True, "aliases": {str(x): y for x, y in alias.items()}}
            (g.run / "selection").mkdir(parents=True, exist_ok=True)
            (g.run / "selection" / f"{P}.json").write_text(json.dumps(sel, indent=1))
            nominees[(ds, k)] = (nom, src)
            # nominee views, zero-fairness twin
            def onehot(cl):
                X = np.zeros((len(cl), int(cl.max()) + 1))
                X[np.arange(len(cl)), cl] = 1
                return X
            cZ = cells_for(H, 1, k)  # zero-fairness twin: the finest partition
            g.farefit(ds, k, f"{P}__FAREFIT_Z", cZ, H)
            g.attack(ds, f"{P}__FZ__rep", 0.9, {"features": "FZ", "outputs": None})
            g.u2(ds, f"{P}__U2__FZ", 0.3, base)
            for tag, X, rep_unit in (("F", onehot(cells_of[src]), f"{P}__Fc{src}__rep"),
                                     ("FZ", onehot(cZ), f"{P}__FZ__rep")):
                g.attack(ds, f"{P}__{tag}__rep+clean", 1.6, {"features": tag, "outputs": "clean"},
                         plus=(rep_unit, f"{P}__O_full", "ignore_rep" if k != 2 else None))
                g.head(ds, f"{P}__HEAD__{tag}", X)
                g.attack(ds, f"{P}__O_head{tag}", 0.5, {"outputs": f"head on {tag}", "features": None})
                g.attack(ds, f"{P}__{tag}__rep+head", 0.5, {"features": tag, "outputs": "head"},
                         plus=(rep_unit, f"{P}__O_head{tag}", "ignore_out" if k == 0 else None))
            ccfg = lock["fare"]["certificate"]
            secg = (lock["fare"].get("secondary_certificate_groups") or {}).get(ds)
            cert = {"cert_rows": int(len(idx["cert"]))}
            for who, fu in (("nominee", f"{P}__FAREFIT_c{nom}"), ("zero_fairness", f"{P}__FAREFIT_Z")):
                cert[who] = {"primary_all_groups": cert_record(g, ds, k, fu, None, ccfg)}
                if secg:
                    cert[who]["secondary_groups"] = cert_record(g, ds, k, fu, secg, ccfg)
            (g.run / "certificates").mkdir(parents=True, exist_ok=True)
            (g.run / "certificates" / f"{P}.json").write_text(json.dumps(cert, indent=1))
        (g.run / "controls").mkdir(parents=True, exist_ok=True)
        (g.run / "controls" / f"{ds}.json").write_text(json.dumps({"O_prob": {
            "null_val_macro_auc": 0.51, "planted_val_macro_auc": 0.93, "null_flag_above_0.55": False,
            "planted_detected_above_0.75": True}}))
    return g, nominees


# ------------------------------------------------------------------------------------------------ generator-side inference
def unit_R(units, uid, classes):
    rec = json.loads((units / uid / "record.json").read_text())
    ps = rec.get("plus_selection")
    if ps and ps.get("alias_source"):
        return unit_R(units, ps["alias_source"], classes)
    z = np.load(units / uid / "preds.npz")
    return [(z[f"P__NL__as{s}"][:, c], z["y_s"] == c, prep(z[f"P__NL__as{s}"][:, c])) for s in range(3)
            for c in classes], len(classes) * 3


def runner_primary(g, nominees, fam_rows):
    out = []
    for ds in DS:
        classes = DS[ds]["sup"]
        a = g.idx[ds]["assessment"]
        au = g.W[ds]["unit"][a]

        def sideR(name):
            terms = []
            for k in range(3):
                P = f"{ds}__s{k}"
                nm, src = nominees[(ds, k)]
                uid = name.format(P=P, src=src)
                tt, _ = unit_R(g.units, uid, classes)
                terms.append(tt)
            return terms

        def sideA(name):
            terms = []
            for k in range(3):
                P = f"{ds}__s{k}"
                nm, src = nominees[(ds, k)]
                z = np.load(g.units / name.format(P=P, src=src) / "preds.npz")
                terms.append((z["U2_P"].argmax(1) == z["y_t"]).astype(float))
            return terms
        defs = {"P1": ("R", "{P}__O_full", "{P}__O_hard"), "P2": ("R", "{P}__B__rep", "{P}__Fc{src}__rep"),
                "P3": ("A", "{P}__U2__Fc{src}", "{P}__U2__A"), "P4": ("A", "{P}__U2__Fc{src}", "{P}__U2__B"),
                "P5": ("R", "{P}__F__rep+clean", "{P}__F__rep+head"), "P6": ("R", "{P}__B__rep+head", "{P}__F__rep+head")}
        sides = {}
        for q, (kind, s1, s2) in defs.items():
            sides[q] = (kind, (sideR if kind == "R" else sideA)(s1), (sideR if kind == "R" else sideA)(s2))

        def value(kind, side, w):
            if kind == "R":
                return np.mean([np.mean([wauc_one(sc, pos, w, pr) for sc, pos, pr in t]) for t in side])
            return np.mean([float((c * w).sum() / w.sum()) for c in side])
        pts = {q: value(k_, s1, np.ones(len(a))) - value(k_, s2, np.ones(len(a))) for q, (k_, s1, s2) in sides.items()}
        reps = {q: [] for q in sides}
        for w in boot_weights(au, B_PRI, SEED_PRI):
            for q, (k_, s1, s2) in sides.items():
                reps[q].append(value(k_, s1, w) - value(k_, s2, w))
        alpha = 0.05 / 12
        for f in fam_rows:
            if f["cell"] != ds:
                continue
            q = f["id"].split("-")[0]
            r = np.array(reps[q])
            lo, hi = np.quantile(r, alpha, method="linear"), np.quantile(r, 1 - alpha, method="linear")
            out.append({"id": f["id"], "cell": ds, "point": pts[q], "lower": lo, "upper": hi,
                        "decision": "PASS" if lo > float(f["margin"]) else "NOT_ESTABLISHED"})
    return out


# ------------------------------------------------------------------------------------------------ benchmark tree
BENCH_KEYS = ("P__rep__NL__as{k}", "P__rep__L__as{k}", "P__repPLUSoutputs__NL__as{k}", "P__repPLUSoutputs__Lslate__as{k}",
              "P__outputs__NL__as{k}")


def bench_unit(root, rng, ds, lab, uid, beta, alias_of=None, rho_offset=0.0):
    c = DS[ds]
    d = root / "bench" / "units" / uid
    d.mkdir(parents=True)
    a = np.flatnonzero(lab["role"] == "assessment")
    s, t = lab[c["attr"]][a], lab[c["task"]][a]
    K = c["K_s"]
    sup = {"sensitive": {"n_classes": K, "thresholds": {"assessment": 10}, "supported_classes": c["sup"],
                         "counts_per_role": {"assessment": np.bincount(s, minlength=K).tolist()}, "status": "OK"},
           "task": {"n_classes": 2, "thresholds": {"assessment": 10}, "supported_classes": [0, 1],
                    "counts_per_role": {"assessment": np.bincount(t, minlength=2).tolist()}, "status": "OK"}}
    (d / "supported.json").write_text(json.dumps(sup))
    (d / "fit_records.json").write_text(json.dumps({"alias_of": alias_of}))
    if alias_of is not None:
        shutil.copy(root / "bench" / "units" / alias_of / "preds.npz", d / "preds.npz")
    else:
        p = {"assess_row_id": lab["row_id"][a], "assess_unit": lab["unit"][a], "y_s": s, "y_task": t,
             "s_prior_fit": np.bincount(s, minlength=K) / len(s), "t_prior_fit": np.array([0.6, 0.4])}
        for pat in BENCH_KEYS:
            for k in range(3):
                p[pat.format(k=k)] = make_P(rng, s, K, beta + 0.1 * k)
        for nm in ("G1", "G2"):
            p[f"{nm}_pred"] = make_P(rng, s, K, 0.7)
            p[f"{nm}_prior"] = np.bincount(s, minlength=K) / len(s)
        p["U2_P"] = make_P(rng, t, 2, beta)
        p["U1_logits"] = rng.normal(size=(len(a), 2)) + 1.2 * np.eye(2)[t]
        u = rng.normal(size=len(a))
        p["RHO_u"] = u + rho_offset
        p["RHO_v"] = 0.1 * u + rng.normal(size=len(a)) + 3.0
        np.savez_compressed(d / "preds.npz", **p)
    complete(d)


def build_bench(root, rng, worlds):
    units = []
    for ds in DS:
        lab = worlds[ds]
        cell = f"{DS[ds]['purpose']}__{DS[ds]['attr']}"
        for k in range(3):
            bench_unit(root, rng, ds, lab, f"{ds}__s{k}__{cell}__A", 1.5)
            bench_unit(root, rng, ds, lab, f"{ds}__s{k}__{cell}__B", 1.3)
            if ds == "adult":
                bench_unit(root, rng, ds, lab, f"{ds}__s{k}__{cell}__C", 0, alias_of=f"{ds}__s{k}__{cell}__B")
            else:
                bench_unit(root, rng, ds, lab, f"{ds}__s{k}__{cell}__C", 1.25)
            for rs in range(3):
                bench_unit(root, rng, ds, lab, f"{ds}__s{k}__{cell}__D_sigma{SIG[ds]:g}_rs{rs}", 0.2)
                bench_unit(root, rng, ds, lab, f"{ds}__s{k}__{cell}__D_sigma0.25_rs{rs}", 0.9)
    lab = worlds["adult"]
    for k in range(3):
        bench_unit(root, rng, "adult", lab, f"adult__s{k}__employment_analysis__marital_status__A", 1.4)
        if k < 2:
            for sg in ("0.5", "1", "2", "4", "8"):
                for rs in range(3):
                    bench_unit(root, rng, "adult", lab,
                               f"adult__s{k}__employment_analysis__marital_status__D_sigma{sg}_rs{rs}", 0.3)
        for arm in ("A", "B", "C"):
            bench_unit(root, rng, "adult", lab, f"adult__s{k}__education_assessment__income__{arm}", 1.0,
                       rho_offset=6.5e6)
    (root / "bench" / "infer").mkdir(parents=True, exist_ok=True)
    (root / "bench" / "infer" / "SIGMA_STAR.json").write_text(json.dumps(
        {"datasets": {ds: {"sigma_star": SIG[ds]} for ds in DS}}))


def bench_preds(root, uid, drop=None):
    d = root / "bench" / "units" / uid
    fr = json.loads((d / "fit_records.json").read_text())
    src = fr.get("alias_of") or uid
    z = dict(np.load(root / "bench" / "units" / src / "preds.npz"))
    if drop is not None:
        keep = ~np.isin(z["assess_row_id"], drop)
        n = len(keep)
        z = {k: (v[keep] if v.ndim >= 1 and v.shape[0] == n and "prior" not in k else v) for k, v in z.items()}
    return z


def gen_unit_stat(z, key, classes):
    """Per-unit statistic as a function of weights w (generator-side)."""
    y = z["y_s"].astype(int)
    if key[2] == "macro_auc":
        pat = {("rep", "NL"): "P__rep__NL__as{k}", ("rep", "L"): "P__rep__L__as{k}",
               ("rep+outputs", "NL"): "P__repPLUSoutputs__NL__as{k}",
               ("rep+outputs", "L"): "P__repPLUSoutputs__Lslate__as{k}",
               ("outputs", "NL"): "P__outputs__NL__as{k}"}[key[:2]]
        terms = [(z[pat.format(k=k)][:, c], y == c, prep(z[pat.format(k=k)][:, c])) for k in range(3) for c in classes]
        return lambda w: float(np.mean([wauc_one(sc, pos, w, pr) for sc, pos, pr in terms]))
    if key[2] == "r2":
        pred, prior = z[f"{key[1]}_pred"], z[f"{key[1]}_prior"]
        Y = np.eye(pred.shape[1])[y]
        num, den = ((Y - pred) ** 2).sum(1), ((Y - prior[None]) ** 2).sum(1)
        return lambda w: float(1 - (num * w).sum() / (den * w).sum())
    t = z["y_task"].astype(int)
    if key == ("U2", "U2", "accuracy"):
        c = (z["U2_P"].argmax(1) == t).astype(float)
        return lambda w: float((c * w).sum() / w.sum())
    if key == ("U2", "U2", "log_loss"):
        P = np.clip(z["U2_P"], 1e-12, 1)
        P = P / P.sum(1, keepdims=True)
        ll = -np.log(P[np.arange(len(t)), t])
        return lambda w: float((ll * w).sum() / w.sum())
    if key == ("U1", "U1", "accuracy"):
        c = (softmax(z["U1_logits"]).argmax(1) == t).astype(float)
        return lambda w: float((c * w).sum() / w.sum())
    if key == ("rep", "RHO1", "rho1sq"):
        u, v = z["RHO_u"] - z["RHO_u"].mean(), z["RHO_v"] - z["RHO_v"].mean()

        def f(w):
            s0 = w.sum()
            mu, mv = (u * w).sum() / s0, (v * w).sum() / s0
            cuv = ((u - mu) * (v - mv) * w).sum() / s0
            return float(cuv ** 2 / ((((u - mu) ** 2) * w).sum() / s0 * (((v - mv) ** 2) * w).sum() / s0))
        return f
    raise KeyError(key)


def gen_bootstrap(fns, au, B, seed):
    reps = {k: [] for k in fns}
    for w in boot_weights(au, B, seed):
        for k, f in fns.items():
            reps[k].append(f(w))
    return {k: np.array(v) for k, v in reps.items()}


def build_bench_tables(root, worlds, bench_pkg, exp_dir, pkg):
    """Benchmark family (24), original endpoints, retained endpoints, C1 / C2 corrections (generator-side)."""
    eps = []
    for ds in DS:
        cell = f"{ds}__{DS[ds]['purpose']}__{DS[ds]['attr']}"
        eps.append({"id": f"P-{ds}-G1-A", "dataset": ds, "cell": cell, "arm": "A", "statistic": "G1_r2", "bar": 0.05})
        for arm in ("A", "B", "C", "Dstar"):
            eps.append({"id": f"P-{ds}-Rrep-{arm}", "dataset": ds, "cell": cell, "arm": arm,
                        "statistic": "C_rep_NL_macro_auc", "bar": 0.55})
        for arm in ("A", "B", "C", "Dstar"):
            eps.append({"id": f"P-{ds}-Rplus-{arm}", "dataset": ds, "cell": cell, "arm": arm,
                        "statistic": "C_rep_plus_clean_out_NL_macro_auc", "bar": 0.55})
        for arm in ("B", "C", "Dstar"):
            eps.append({"id": f"P-{ds}-U2NI-{arm}", "dataset": ds, "cell": cell, "arm": arm,
                        "statistic": "U2_accuracy_diff_vs_A", "bar": -0.01})
    bench_pkg.mkdir(parents=True, exist_ok=True)
    (bench_pkg / "PRIMARY_FAMILY.json").write_text(json.dumps({"alpha_family": 0.05, "size": 24, "endpoints": eps}))
    drop = {}
    for ds, lab in worlds.items():
        ck = lab["canon_key"].astype(str)
        tk = set(ck[lab["split"] == "train"])
        ex = np.array([(sp == "test" and k in tk) for sp, k in zip(lab["split"], ck)]) & (lab["role"] == "assessment")
        drop[ds] = lab["row_id"][ex]

    def endpoints(dr):
        out = {}
        alpha = 0.05 / 24
        for ds in DS:
            cell = f"{DS[ds]['purpose']}__{DS[ds]['attr']}"
            groups = {"A": [f"{ds}__s{k}__{cell}__A" for k in range(3)], "B": [f"{ds}__s{k}__{cell}__B" for k in range(3)],
                      "C": [f"{ds}__s{k}__{cell}__C" for k in range(3)],
                      "Dstar": [f"{ds}__s{k}__{cell}__D_sigma{SIG[ds]:g}_rs{r}" for k in range(3) for r in range(3)]}
            Z = {u: bench_preds(root, u, None if dr is None else dr[ds]) for us in groups.values() for u in us}
            fns = {}
            for e in [x for x in eps if x["dataset"] == ds]:
                arm = e["arm"]
                key = {"G1_r2": ("rep", "G1", "r2"), "C_rep_NL_macro_auc": ("rep", "NL", "macro_auc"),
                       "C_rep_plus_clean_out_NL_macro_auc": ("rep+outputs", "NL", "macro_auc"),
                       "U2_accuracy_diff_vs_A": ("U2", "U2", "accuracy")}[e["statistic"]]
                f1 = [gen_unit_stat(Z[u], key, DS[ds]["sup"]) for u in groups[arm]]
                if e["statistic"] == "U2_accuracy_diff_vs_A":
                    f0 = [gen_unit_stat(Z[u], key, DS[ds]["sup"]) for u in groups["A"]]
                    fns[e["id"]] = (lambda w, f1=f1, f0=f0: np.mean([f(w) for f in f1]) - np.mean([f(w) for f in f0]))
                else:
                    fns[e["id"]] = (lambda w, f1=f1: np.mean([f(w) for f in f1]))
            au = Z[groups["A"][0]]["assess_unit"]
            reps = gen_bootstrap(fns, au, B_EXP, SEED_EXP)
            for e in [x for x in eps if x["dataset"] == ds]:
                pt = fns[e["id"]](np.ones(len(au)))
                lo = np.quantile(reps[e["id"]], alpha, method="linear")
                hi = np.quantile(reps[e["id"]], 1 - alpha, method="linear")
                if e["statistic"] == "U2_accuracy_diff_vs_A":
                    dec = "NONINFERIOR" if lo >= e["bar"] else ("INFERIOR" if hi < e["bar"] else "UNRESOLVED")
                else:
                    dec = "ESTABLISHED_ABOVE" if lo > e["bar"] else ("ESTABLISHED_BELOW" if hi < e["bar"] else "UNRESOLVED")
                out[e["id"]] = {"id": e["id"], "point": pt, "lower": lo, "upper": hi, "decision": dec}
        return out
    orig = endpoints(None)
    with open(bench_pkg / "PRIMARY_ENDPOINTS.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "point", "lower", "upper", "decision"])
        w.writeheader()
        for e in eps:
            w.writerow(orig[e["id"]])
    ret = endpoints(drop)
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "INFER_ALL_retained.json").write_text(json.dumps({"primary": {"endpoints": list(ret.values())}}))
    # ---- C1
    cell = "adult__employment_analysis__marital_status"
    A = [f"adult__s{k}__employment_analysis__marital_status__A" for k in range(3)]
    keys = [("rep", "NL", "macro_auc"), ("rep", "L", "macro_auc"), ("rep+outputs", "NL", "macro_auc"),
            ("rep+outputs", "L", "macro_auc"), ("rep", "G1", "r2"), ("rep", "G2", "r2"), ("U2", "U2", "accuracy"),
            ("U2", "U2", "log_loss"), ("U1", "U1", "accuracy")]
    c1, fns, meta = [], {}, {}
    Z = {}
    for sg in ("0.5", "1", "2", "4", "8"):
        g = f"D_sigma{sg}"
        D = [f"adult__s{k}__employment_analysis__marital_status__{g}_rs{r}" for k in range(2) for r in range(3)]
        Ash = A[:2]
        for u in D + A:
            Z.setdefault(u, bench_preds(root, u))
        for key in keys:
            sid = f"{cell}|{g}-A|{'|'.join(key)}"
            fd = [gen_unit_stat(Z[u], key, [0, 1]) for u in D]
            fa = [gen_unit_stat(Z[u], key, [0, 1]) for u in Ash]
            faa = [gen_unit_stat(Z[u], key, [0, 1]) for u in A]
            fns[sid] = lambda w, fd=fd, fa=fa: np.mean([f(w) for f in fd]) - np.mean([f(w) for f in fa])
            meta[sid] = {"g": g, "kind": "paired_diff_vs_A", "status": "CORRECTED",
                         "orig": np.mean([f(np.ones(len(Z[A[0]]["y_s"]))) for f in fd]) -
                         np.mean([f(np.ones(len(Z[A[0]]["y_s"]))) for f in faa])}
        fo = [gen_unit_stat(Z[u], ("outputs", "NL", "macro_auc"), [0, 1]) for u in Ash]
        for arm, us, st in ((g, D, "CORRECTED"), ("A", Ash, "PAIRED REFERENCE (A on shared seeds)")):
            fp = [gen_unit_stat(Z[u], ("rep+outputs", "NL", "macro_auc"), [0, 1]) for u in us]
            sid = f"{g}::{cell}|{arm}|plus_NL_minus_outputs_NL|macro_auc"
            fns[sid] = lambda w, fp=fp, fo=fo: np.mean([f(w) for f in fp]) - np.mean([f(w) for f in fo])
            meta[sid] = {"g": g, "kind": "rep+outputs minus outputs (paired seeds)", "status": st, "orig": None}
    au = Z[A[0]]["assess_unit"]
    reps = gen_bootstrap(fns, au, B_COR, SEED_COR)
    for sid, f in fns.items():
        m = meta[sid]
        pt = f(np.ones(len(au)))
        lo, hi = np.quantile(reps[sid], 0.05, method="linear"), np.quantile(reps[sid], 0.95, method="linear")
        rid = sid.split("::", 1)[1] if "::" in sid else sid
        c1.append({"cell": cell, "arm_group": m["g"], "truncated": True, "id": rid, "kind": m["kind"],
                   "original_point": m["orig"], "corrected_point": pt, "corrected_lower90": lo,
                   "corrected_upper90": hi, "status": m["status"]})
    (pkg / "corrections").mkdir(parents=True, exist_ok=True)
    (pkg / "corrections" / "C1_seed_pairing.json").write_text(json.dumps(c1, indent=1))
    with open(pkg / "MATCHED_COMPARISONS_CORRECTED.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "point", "correction_status"])
        w.writeheader()
        for r in c1:
            if r["status"] == "CORRECTED":
                w.writerow({"id": r["id"], "point": r["corrected_point"], "correction_status": "C1_seed_pairing"})
    # ---- C2
    units = sorted(p.name for p in (root / "bench" / "units").iterdir())
    c2u = []
    for u in units:
        z = dict(np.load(root / "bench" / "units" / u / "preds.npz"))
        c2u.append({"unit": u, "stable_point": gen_unit_stat(z, ("rep", "RHO1", "rho1sq"), None)(np.ones(len(z["y_s"])))})
    cell2 = "adult__education_assessment__income"
    fns, ids = {}, []
    for arm in ("A", "B", "C"):
        us = [f"adult__s{k}__education_assessment__income__{arm}" for k in range(3)]
        fs = [gen_unit_stat(bench_preds(root, u), ("rep", "RHO1", "rho1sq"), None) for u in us]
        for u, f in zip(us, fs):
            fns[f"{u}|rep|RHO1|rho1sq"] = f
        fns[f"{cell2}|{arm}|rep|RHO1|rho1sq"] = lambda w, fs=fs: np.mean([f(w) for f in fs])
    au = bench_preds(root, f"adult__s0__education_assessment__income__A")["assess_unit"]
    reps = gen_bootstrap(fns, au, B_COR, SEED_COR)
    c2i = []
    for sid, f in fns.items():
        c2i.append({"id": sid, "cell": cell2, "corrected_point": f(np.ones(len(au))),
                    "corrected_lower90": np.quantile(reps[sid], 0.05, method="linear"),
                    "corrected_upper90": np.quantile(reps[sid], 0.95, method="linear")})
    (pkg / "corrections" / "C2_rho_stable.json").write_text(json.dumps({"units": c2u, "intervals": c2i}, indent=1))
    with open(pkg / "RECOVERY_CORRECTED.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "point", "correction_status"])
        w.writeheader()
        for r in c2i:
            w.writerow({"id": r["id"], "point": r["corrected_point"], "correction_status": "C2_stable_rho1sq"})


# ------------------------------------------------------------------------------------------------ tree assembly
def build_tree(root: Path, seed=7):
    rng = np.random.default_rng(seed)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    worlds = build_inputs(root, rng)
    lock = json.loads((PKG_REAL / "EXECUTION_LOCK.json").read_text())
    thr = {"attacker_fit": 10, "attacker_val": 5, "assessment": 10}
    lock["roles_rule"]["support"]["thresholds"] = thr
    pkg = root / "pkg"
    pkg.mkdir()
    ras = {}
    for ds, lab in worlds.items():
        idx, excl = roles_of(ds, lab, thr)
        ras[ds] = ras_entry(ds, lab, idx, excl, thr)
    (pkg / "ROLES_AND_SUPPORT.json").write_text(json.dumps(ras, indent=1))
    lock["roles_and_support_sha256"] = hashlib.sha256(json.dumps(ras, sort_keys=True).encode()).hexdigest()
    (pkg / "EXECUTION_LOCK.json").write_text(json.dumps(lock, indent=1))
    shutil.copy(PKG_REAL / "PRIMARY_FAMILY.csv", pkg / "PRIMARY_FAMILY.csv")
    g, nominees = build_run(root, rng, worlds, thr, lock)
    fam_rows = R.read_csv(pkg / "PRIMARY_FAMILY.csv")
    prim = runner_primary(g, nominees, fam_rows)
    with open(pkg / "PRIMARY_ENDPOINTS.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "cell", "point", "lower", "upper", "decision"])
        w.writeheader()
        for r in prim:
            w.writerow(r)
    build_bench(root, rng, worlds)
    build_bench_tables(root, worlds, root / "bench_pkg", root / "exposure", pkg)
    return {"nominees": {f"{d}__s{k}": v for (d, k), v in nominees.items()}, "primary": prim}


def run_replay(root: Path, stages=None, extra=()):
    args = ["--expect-exposure", json.dumps(EXPECT_EXPOSURE), "--pkg", str(root / "pkg"), "--run", str(root / "run"), "--bench", str(root / "bench"),
            "--bench-pkg", str(root / "bench_pkg"), "--exposure-dir", str(root / "exposure"),
            "--B-primary", str(B_PRI), "--B-exposure", str(B_EXP), "--B-corrections", str(B_COR),
            "--out", str(root / "out" / "IV.json"), "--agg", str(root / "out" / "agg.json"), *extra]
    if stages:
        args += ["--stages", stages]
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        return R.main(args)


# ------------------------------------------------------------------------------------------------ defects
def rewrite_unit(d: Path):
    complete(d)


def d_wrong_fare_alias(root):
    p = root / "run" / "selection" / "adult__s0.json"
    s = json.loads(p.read_text())
    s["aliases"] = {"3": 1}
    p.write_text(json.dumps(s))
    return "nominee", ["adult__s0:fare_aliases"]


def d_wrong_plus_alias(root):
    d = root / "run" / "units" / "adult__s1__B__rep+head"
    r = json.loads((d / "record.json").read_text())
    cur = r["plus_selection"]["alias_source"]
    r["plus_selection"]["alias_source"] = "adult__s1__O_headB" if cur == "adult__s1__B__rep" else "adult__s1__B__rep"
    (d / "record.json").write_text(json.dumps(r))
    rewrite_unit(d)
    return "plus", ["adult:plus_surface_selection_and_alias", "P6-adult:point"]


def d_assess_leak(root):
    d = root / "run" / "units" / "adult__s0__O_full"
    z = dict(np.load(d / "val_preds.npz"))
    a = np.load(root / "run" / "units" / "adult__s0__O_full" / "preds.npz")["assess_row_id"]
    z["val_row_id"] = z["val_row_id"].copy()
    z["val_row_id"][0] = a[3]
    np.savez_compressed(d / "val_preds.npz", **z)
    rewrite_unit(d)
    return "leakage", ["adult:val_rows_equal_attacker_val", "adult:no_assessment_row_in_val"]


def d_seed_set(root):
    shutil.rmtree(root / "run" / "units" / "hmda__s2__O_hard")
    return "seeds", ["P1-hmda:encoder_seeds_paired", "P1-hmda:vs_runner"]


def d_wrong_nominee(root):
    p = root / "run" / "selection" / "hmda__s1.json"
    s = json.loads(p.read_text())
    alt = [r["config"] for r in s["table"] if r["admissible"] and r["config"] != s["nominee"] and r["alias_of"] is None]
    s["nominee"] = alt[0]
    s["nominee_unit_source"] = alt[0]
    p.write_text(json.dumps(s))
    return "nominee", ["hmda__s1:nominee"]


def d_wrong_decision(root):
    p = root / "pkg" / "PRIMARY_ENDPOINTS.csv"
    rows = R.read_csv(p)
    for r in rows:
        if r["id"] == "P2-adult":
            r["decision"] = "NOT_ESTABLISHED" if r["decision"] == "PASS" else "PASS"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return "primary", ["P2-adult:decision"]


def d_family_13_runner(root):
    p = root / "pkg" / "PRIMARY_ENDPOINTS.csv"
    rows = R.read_csv(p)
    rows.append({**rows[0], "id": "P7-adult"})
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return "primary", ["runner_family_size_12"]


def d_family_11_csv(root):
    p = root / "pkg" / "PRIMARY_FAMILY.csv"
    rows = R.read_csv(p)[:-1]
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return "primary", ["family_size_and_rows"]


def d_point_error(root):
    p = root / "pkg" / "PRIMARY_ENDPOINTS.csv"
    rows = R.read_csv(p)
    for r in rows:
        if r["id"] == "P4-hmda":
            r["point"] = repr(float(r["point"]) + 1e-6)
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return "primary", ["P4-hmda:point"]


def d_head(root):
    d = root / "run" / "units" / "adult__s0__HEAD__A"
    z = dict(np.load(d / "preds.npz"))
    z["head_outputs_all"] = z["head_outputs_all"] + 1e-6
    np.savez_compressed(d / "preds.npz", **z)
    rewrite_unit(d)
    return "contracts", ["adult__s0__HEAD__A:deterministic_on_features"]


def d_sigma(root):
    d = root / "run" / "units" / "hmda__s0__D_rs1__rep"
    r = json.loads((d / "record.json").read_text())
    r["contract"]["features"] = "noise sigma=2.0 rs=1"
    (d / "record.json").write_text(json.dumps(r))
    rewrite_unit(d)
    return "membership", ["hmda:noise_sigma_and_release_seeds"]


def d_fit_rows(root):
    d = root / "run" / "units" / "adult__s1__FAREFIT_c4"
    r = json.loads((d / "rec.json").read_text())
    r["fit_rows_sha256"] = "0" * 64
    (d / "rec.json").write_text(json.dumps(r))
    rewrite_unit(d)
    return "leakage", ["adult:fare_fit_rows_are_defense_fit"]


def d_cert(root):
    p = root / "run" / "certificates" / "hmda__s0.json"
    c = json.loads(p.read_text())
    blk = c["nominee"]["primary_all_groups"]
    blk["status"] = "OK"
    blk["bound"] = 0.2
    blk["premises"] = []
    p.write_text(json.dumps(c))
    return "contracts", ["hmda__s0:certificates_with_premises"]


def d_tamper(root):
    d = root / "run" / "units" / "adult__s2__B__rep"
    z = dict(np.load(d / "preds.npz"))
    z["P__NL__as1"] = z["P__NL__as1"][::-1].copy()
    np.savez_compressed(d / "preds.npz", **z)   # COMPLETE.json NOT updated
    return "custody", ["custody:adult__s2__B__rep"]


def d_exposure_point(root):
    p = root / "exposure" / "INFER_ALL_retained.json"
    d = json.loads(p.read_text())
    for e in d["primary"]["endpoints"]:
        if e["id"] == "P-adult-Rrep-B":
            e["point"] += 1e-5
    p.write_text(json.dumps(d))
    return "exposure", ["P-adult-Rrep-B:vs_runner"]


def d_c1_point(root):
    p = root / "pkg" / "corrections" / "C1_seed_pairing.json"
    d = json.loads(p.read_text())
    tgt = d[0]["id"]
    d[0]["corrected_point"] += 1e-4
    p.write_text(json.dumps(d))
    return "c1", [tgt]


def d_c1_unpaired(root):
    """Re-introduce the original pairing bug: the runner's corrected value is the unpaired one."""
    p = root / "pkg" / "corrections" / "C1_seed_pairing.json"
    d = json.loads(p.read_text())
    r = next(x for x in d if x["kind"] == "paired_diff_vs_A" and x["id"].endswith("rep|NL|macro_auc"))
    r["corrected_point"] = r["original_point"]
    p.write_text(json.dumps(d))
    return "c1", [r["id"]]


DEFECTS = {
    "wrong_fare_alias": (d_wrong_fare_alias, "roles,leakage,nominee"),
    "wrong_plus_alias": (d_wrong_plus_alias, "roles,nominee,primary,membership"),
    "assessment_rows_leak_into_val": (d_assess_leak, "roles,leakage,nominee"),
    "mismatched_seed_set": (d_seed_set, "roles,nominee,primary"),
    "wrong_nominee": (d_wrong_nominee, "roles,nominee"),
    "wrong_decision": (d_wrong_decision, "roles,nominee,primary"),
    "family_size_13_runner_table": (d_family_13_runner, "roles,nominee,primary"),
    "family_size_11_family_csv": (d_family_11_csv, "roles,nominee,primary"),
    "point_arithmetic_error": (d_point_error, "roles,nominee,primary"),
    "head_not_function_of_features": (d_head, "roles,nominee,contracts"),
    "wrong_noise_sigma": (d_sigma, "roles,nominee,membership"),
    "fare_fit_rows_not_defense_fit": (d_fit_rows, "roles,leakage,nominee"),
    "certificate_without_premises": (d_cert, "roles,nominee,contracts"),
    "unit_tampered_after_complete": (d_tamper, "roles,nominee,primary"),
    "exposure_point_error": (d_exposure_point, "exposure"),
    "c1_point_error": (d_c1_point, "c1"),
    "c1_unpaired_seeds_regression": (d_c1_unpaired, "c1"),
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", default=None)
    ap.add_argument("--out", default=str(HERE / "synthetic_selftest_result.json"))
    a = ap.parse_args(argv)
    import tempfile
    scratch = Path(a.scratch) if a.scratch else Path(tempfile.mkdtemp(prefix="oar_replay_selftest_"))
    t0 = time.time()
    base = scratch / "baseline"
    truth = build_tree(base)
    t_build = time.time() - t0
    out = {"schema": "oar_replay_synthetic_selftest/v1", "replay_sha256": sha(HERE / "replay_oar.py"),
           "test_sha256": sha(Path(__file__)), "B": {"primary": B_PRI, "exposure": B_EXP, "corrections": B_COR},
           "build_s": t_build}
    res = run_replay(base, extra=("--exposure-original",))
    bad = [i for i in res["items"] if i["status"] not in ("PASS",)]
    out["baseline"] = {"n_items": len(res["items"]), "counts": res["public_summary"]["counts_by_status"],
                       "non_pass": [{k: i.get(k) for k in ("scope", "id", "status", "cause")} for i in bad]}
    # replay nominee agrees with the generator; primary decisions agree
    out["baseline"]["ok"] = not bad
    defects = {}
    for name, (fn, stages) in DEFECTS.items():
        d = scratch / name
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(base, d, ignore=shutil.ignore_patterns("out"))
        _, expect = fn(d)
        r = run_replay(d, stages=stages)
        st = {i["id"]: i["status"] for i in r["items"]}
        caught = {e: st.get(e) for e in expect}
        ok = all(v not in (None, "PASS") for v in caught.values())
        defects[name] = {"expected_non_pass_items": caught, "caught": ok,
                         "n_non_pass": sum(1 for i in r["items"] if i["status"] != "PASS")}
    out["defects"] = defects
    out["all_defects_caught"] = all(v["caught"] for v in defects.values())
    out["passed"] = bool(out["baseline"]["ok"] and out["all_defects_caught"])
    out["wall_s"] = time.time() - t0
    Path(a.out).write_text(json.dumps(R.jsonable(out), indent=1))
    print(json.dumps({"passed": out["passed"], "baseline_ok": out["baseline"]["ok"],
                      "baseline_non_pass": out["baseline"]["non_pass"][:10],
                      "defects": {k: v["caught"] for k, v in defects.items()}, "wall_s": round(out["wall_s"], 1)},
                     indent=1))
    return out


if __name__ == "__main__":
    main()
