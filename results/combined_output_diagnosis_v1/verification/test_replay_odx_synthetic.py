"""Synthetic self-test for replay_odx.py (independent verifier).

Builds a synthetic home directory with the same layout as the study (bench inputs, oar/odx unit trees with
preds/val_preds/models/record/COMPLETE, aliases, banks, S4 pair units, controls, LOCK and public tables), checks that
the replay passes on the clean tree and agrees with a separate reference implementation (rank-based Mann-Whitney on
replicate-expanded samples), then plants defects one at a time and confirms each is caught.

    OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python results/combined_output_diagnosis_v1/verification/test_replay_odx_synthetic.py
    (or: pytest -q <this file>)

Scratch: $REPLAY_SCRATCH or <tmp>/odx_replay_synth. Synthetic data only; nothing private is read.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import joblib  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import norm, rankdata  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
spec = importlib.util.spec_from_file_location("replay_odx", HERE / "replay_odx.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)

B_SYN = 199
SEED = 20261021
SCRATCH = Path(os.environ.get("REPLAY_SCRATCH", Path(tempfile.gettempdir()) / "odx_replay_synth"))

ATTRS = {"adult": {"sex": [.35, .65], "race": [.05, .15, .15, .05, .60], "age_group": [.2, .4, .3, .1],
                   "marital_status": [.5, .5], "income": [.75, .25]},
         "hmda": {"race": [.50, .20, .22, .05, .03], "ethnicity": [.7, .3], "sex": [.4, .6]}}
PURPOSES = {"adult": [("income_prediction", "income", 2, "task_income"),
                      ("employment_analysis", "occupation_group", 6, "task_occupation_group"),
                      ("education_assessment", "education_level", 4, "task_education_level")],
            "hmda": [("underwriting", "loan_decision", 2, "task_loan_decision"),
                     ("pricing_analysis", "loan_amount_band", 5, "task_loan_amount_band"),
                     ("fair_lending_audit", "tract_denial_high", 2, "task_tract_denial_high")]}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_complete(d: Path, uid, alias=False):
    files = {str(p.relative_to(d)): sha(p) for p in sorted(d.rglob("*")) if p.is_file() and p.name != "COMPLETE.json"}
    rec = {"id": uid, "files": files}
    if alias:
        rec["alias"] = True
    else:
        rec["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (d / "COMPLETE.json").write_text(json.dumps(rec, indent=1))


# ============================================================================== synthetic world ("runner stand-in")
class Synth:
    def __init__(self, root: Path):
        self.root = root
        self.home = root / "home"
        self.cache = self.home / "PCRL_eval_cache_private"
        self.inp = self.cache / "bench_v1" / "inputs"
        self.oar = self.cache / "oar_v1" / "run" / "units"
        self.odx = self.cache / "odx_v1" / "run" / "units"
        self.run = self.cache / "odx_v1" / "run"
        self.study = root / "study"
        self.rng = np.random.default_rng(7)

    # ------------------------------------------------------------------ inputs
    def build_inputs(self):
        self.inp.mkdir(parents=True, exist_ok=True)
        index = {"schema": "synthetic", "datasets": {}}
        self.L, self.roles, self.F = {}, {}, {}
        for ds in ("adult", "hmda"):
            rng = np.random.default_rng({"adult": 11, "hmda": 12}[ds])
            n_test, n_train = 2400, 800
            n = n_test + n_train
            split = np.array(["test"] * n_test + ["train"] * n_train)
            canon = np.array([f"{ds[0]}{i:06d}" for i in range(n)], dtype="<U20")
            # exposure: 9 test rows duplicate a train record; 3 within-test duplicate pairs (HMDA only)
            expo = rng.choice(n_test, 9, replace=False)
            for j, i in enumerate(expo):
                canon[i] = canon[n_test + j]
            if ds == "hmda":
                canon[100], canon[101] = canon[102], canon[102]
            uniq, unit = np.unique(canon, return_inverse=True)
            role = np.empty(n, dtype="<U12")
            u = rng.random(n_test)
            role[:n_test] = np.where(u < 0.5, "attacker_fit", np.where(u < 0.65, "attacker_val", "assessment"))
            # within-test duplicates share a role
            if ds == "hmda":
                role[100] = role[101] = role[102]
            role[n_test:] = "defense_fit"
            test_keys = set(canon[:n_test].tolist())
            for i in range(n_test, n):
                if canon[i] in test_keys:
                    role[i] = "excluded_dup"
            lab = {"row_id": np.arange(n), "split": split, "unit": unit.astype(np.int64), "record_key": canon,
                   "canon_key": canon, "role": role}
            for a, p in ATTRS[ds].items():
                lab[a] = rng.choice(len(p), n, p=p).astype(np.int64)
            for (pname, task, Kt, tkey) in PURPOSES[ds]:
                lab[tkey] = rng.integers(0, Kt, n).astype(np.int64)
            np.savez(self.inp / f"{ds}_labels.npz", **lab)
            rr = {}
            for r in ("defense_fit", "attacker_fit", "attacker_val", "assessment"):
                rr[f"{r}__row_id"] = lab["row_id"][role == r]
                rr[f"{r}__unit"] = lab["unit"][role == r]
            rr["excluded_dup__row_id"] = lab["row_id"][role == "excluded_dup"]
            np.savez(self.inp / f"{ds}_roles.npz", **rr)
            self.L[ds] = lab
            purposes = {}
            for idx_p, (pname, task, Kt, tkey) in enumerate(PURPOSES[ds]):
                purposes[pname] = {"index": idx_p, "task": task, "task_dim": Kt, "rep_key": f"rep_p{idx_p}",
                                   "logits_key": f"logits_{pname}", "labels_task_key": tkey}
            index["datasets"][ds] = {"purposes": purposes}
            for k in (0, 1, 2):
                fw = {"row_id": lab["row_id"], "split": split}
                for idx_p, (pname, task, Kt, tkey) in enumerate(PURPOSES[ds]):
                    t = lab[tkey]
                    Lg = rng.normal(0, 1, (n, Kt)) + 1.5 * np.eye(Kt)[t]
                    attr_main = {"adult": "sex", "hmda": "race"}[ds]
                    s = lab[attr_main]
                    if ds == "adult":
                        Lg += (0.9 * s)[:, None]                               # offset carries sex
                    else:
                        Lg[:, 1] += 0.8 * (s == 1) - 0.8 * (s == 2)            # margin carries race
                    if ds == "hmda" and pname == "fair_lending_audit" and k == 1:
                        Lg[:, 0] += 100.0                                       # single-class (collapsed) head
                    fw[f"logits_{pname}"] = Lg.astype(np.float32).astype(np.float64)
                    fw[f"rep_p{idx_p}"] = rng.normal(0, 1, (n, 4))
                np.savez(self.inp / f"{ds}_s{k}_forward.npz", **fw)
                self.F[(ds, k)] = fw
        (self.inp / "INPUTS_INDEX.json").write_text(json.dumps(index, indent=1))
        self.index = index

    # ------------------------------------------------------------------ roles (stand-in implementation)
    def world(self, ds):
        lab = self.L[ds]
        role = lab["role"].astype("<U32").copy()
        tk = set(lab["canon_key"][lab["split"] == "train"].tolist())
        ex = (lab["split"] == "test") & np.array([c in tk for c in lab["canon_key"]])
        sc = np.isin(role, ("attacker_fit", "attacker_val", "assessment"))
        role[ex & sc] = "excluded_exposure"
        cert = np.array([int(hashlib.sha256(("oar-cert-v1|" + ds + "|" + c).encode()).hexdigest()[:8], 16) / 2 ** 32
                         < 0.2 for c in lab["canon_key"]]) & (role == "attacker_fit")
        role[cert] = "cert"
        return {r: np.flatnonzero(role == r) for r in ("defense_fit", "cert", "attacker_fit", "attacker_val",
                                                       "assessment")}

    def logits(self, ds, k, purpose):
        return self.F[(ds, k)][f"logits_{purpose}"]

    @staticmethod
    def surfaces(L):
        K = L.shape[1]
        if K == 2:
            d = L[:, 1] - L[:, 0]
            c = (L[:, 0] + L[:, 1]) / 2
            cen = d[:, None]
        else:
            c = L.mean(1)
            cen = L - c[:, None]
        Z = L - L.max(1, keepdims=True)
        P = np.exp(Z) / np.exp(Z).sum(1, keepdims=True)
        H = np.zeros_like(L)
        H[np.arange(len(L)), L.argmax(1)] = 1
        return {"full": L, "dc": np.hstack([cen, c[:, None]]), "centred": cen, "prob": P, "offset": c[:, None],
                "hard": H}

    # ------------------------------------------------------------------ units
    def attack(self, unit_dir: Path, uid, X, ds, attr, finite=False, weak_nl=False):
        W = self.world(ds)
        s = self.L[ds][attr]
        K = int(s.max()) + 1
        f, v, a = W["attacker_fit"], W["attacker_val"], W["assessment"]
        Cs = [1e-6] * 3 if weak_nl else [1.0, 0.3, 3.0]
        models = {f"NL_as{i}": make_pipeline(StandardScaler(), LogisticRegression(C=Cs[i], max_iter=500)).fit(
            X[f], s[f]) for i in range(3)}
        models["L"] = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500)).fit(X[f], s[f])

        def pr(m, rows):
            P = np.zeros((len(rows), K))
            P[:, m.classes_.astype(int)] = m.predict_proba(X[rows])
            return P
        preds = {"assess_row_id": self.L[ds]["row_id"][a], "assess_unit": self.L[ds]["unit"][a], "y_s": s[a]}
        val = {"val_row_id": self.L[ds]["row_id"][v], "val_y_s": s[v]}
        for i in range(3):
            preds[f"P__NL__as{i}"] = pr(models[f"NL_as{i}"], a)
        preds["P__L"] = pr(models["L"], a)
        val["VAL__NL__as0"] = pr(models["NL_as0"], v)
        val["VAL__L"] = pr(models["L"], v)
        ll = lambda P, y: float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-15, 1))))  # noqa: E731
        rec = {"id": uid, "nl_family": "GBT", "nl_selected": {}, "val_log_loss": {"NL": ll(val["VAL__NL__as0"], s[v]),
                                                                                 "L": ll(val["VAL__L"], s[v])},
               "selection_tables": {}, "nl_seed_record": {}, "contract": None, "n_columns": int(X.shape[1]),
               "selection_role": "attacker_val", "fit_role": "attacker_fit"}
        if finite:
            keys = lambda M: [tuple(np.round(r, 9)) for r in M]  # noqa: E731
            prior = np.bincount(s[f], minlength=K) / len(f)
            best = None
            for alpha in (0.1, 1.0, 10.0, 100.0):
                kf = keys(X[f])
                tab = {}
                for key in set(kf):
                    m = np.array([q == key for q in kf])
                    c = np.bincount(s[f][m], minlength=K).astype(float)
                    tab[key] = (c + alpha * prior) / (c.sum() + alpha)
                Pv = np.stack([tab.get(q, prior) for q in keys(X[v])])
                lv = ll(Pv, s[v])
                if best is None or lv < best[0]:
                    best = (lv, alpha, tab)
            preds["P__CC"] = np.stack([best[2].get(q, prior) for q in keys(X[a])])
            val["VAL__CC"] = np.stack([best[2].get(q, prior) for q in keys(X[v])])
            rec["CC"] = {"selected": {"alpha": best[1]}, "selection_table": []}
            rec["NLDA_selected"] = "NL"
        (unit_dir / "models").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(unit_dir / "preds.npz", **preds)
        np.savez_compressed(unit_dir / "val_preds.npz", **val)
        for name, m in models.items():
            joblib.dump(m, unit_dir / "models" / f"{name}.joblib")
        (unit_dir / "record.json").write_text(json.dumps(rec, indent=1))
        write_complete(unit_dir, uid)

    def ref_unit(self, unit_dir, uid, ds, purpose, attr):
        W = self.world(ds)
        s = self.L[ds][attr]
        K = int(s.max()) + 1
        a, f = W["assessment"], W["attacker_fit"]
        prior = np.bincount(s[f], minlength=K) / len(f)
        unit_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(unit_dir / "preds.npz", assess_row_id=self.L[ds]["row_id"][a],
                            assess_unit=self.L[ds]["unit"][a], y_s=s[a], P__LO=np.tile(prior, (len(a), 1)),
                            P__const=np.tile(prior, (len(a), 1)))
        np.savez_compressed(unit_dir / "val_preds.npz")
        (unit_dir / "record.json").write_text(json.dumps({"id": uid, "LO_alpha": 1.0, "prior": prior.tolist()}))
        write_complete(unit_dir, uid)

    def alias(self, uid, src: Path):
        d = self.odx / uid
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(src / "record.json", d / "record.json")
        (d / "ALIAS.json").write_text(json.dumps({"source": "~/" + str(src.relative_to(self.home)),
                                                  "source_COMPLETE_sha256": sha(src / "COMPLETE.json"),
                                                  "why": "synthetic"}, indent=1))
        files = {f: sha(d / f) for f in ("record.json", "ALIAS.json")}
        (d / "COMPLETE.json").write_text(json.dumps({"id": uid, "files": files, "alias": True}, indent=1))

    def bank(self, uid, cands):
        vals, resolved = {}, {}
        for c in cands:
            r = json.loads((self.odx / c / "record.json").read_text())
            vals[c] = float(r["val_log_loss"]["NL"])
            resolved[c] = r.get("bank_selected") or c
        best = None
        for c in cands:
            if best is None or vals[c] < vals[best]:
                best = c
        d = self.odx / uid
        d.mkdir(parents=True, exist_ok=True)
        (d / "record.json").write_text(json.dumps({"id": uid, "kind": "bank", "candidates_attacker_val_log_loss": vals,
                                                   "bank_selected": resolved[best],
                                                   "val_log_loss": {"NL": vals[best]},
                                                   "selection_role": "attacker_val", "note": "synthetic"}, indent=1))
        write_complete(d, uid)

    def build_units(self):
        prim = {("adult", "income_prediction", "sex"), ("hmda", "underwriting", "race")}
        for (ds, purpose, attr) in R.PAIRS:
            for k in (0, 1, 2):
                sf = self.surfaces(self.logits(ds, k, purpose))
                base = f"{ds}__s{k}__{purpose}__{attr}__FH__"
                for name in R.SURFACES:
                    uid = base + name
                    weak = (ds, purpose, attr) == ("hmda", "underwriting", "race") and name in ("full", "dc")
                    if (ds, purpose, attr) in prim and name in ("full", "prob", "hard"):
                        src = self.oar / f"{ds}__s{k}__O_{name}"
                        if not src.exists():
                            self.attack(src, src.name, sf[name], ds, attr, finite=name == "hard", weak_nl=weak)
                        self.alias(uid, src)
                    else:
                        self.attack(self.odx / uid, uid, sf[name], ds, attr, finite=name == "hard", weak_nl=weak)
                ref = f"{ds}__s{k}__{purpose}__{attr}__REF__ref"
                if (ds, purpose, attr) in prim:
                    src = self.oar / f"{ds}__s{k}__REF"
                    self.ref_unit(src, src.name, ds, purpose, attr)
                    self.alias(ref, src)
                else:
                    self.ref_unit(self.odx / ref, ref, ds, purpose, attr)
                self.bank(base + "iobank", [base + c for c in R.IO])
                self.bank(base + "fullbank", [base + c for c in R.FB])
        # S4
        ds, pa, pb, attr = R.COALITION
        for k in (0, 1, 2):
            sa, sb = self.surfaces(self.logits(ds, k, pa)), self.surfaces(self.logits(ds, k, pb))
            for c in R.CONTRACTS:
                uid = f"{ds}__s{k}__PAIR_{pa}+{pb}__{attr}__FH__{c}"
                self.attack(self.odx / uid, uid, np.hstack([sa[c], sb[c]]), ds, attr, finite=c == "hard")
                single = lambda p_: f"{ds}__s{k}__{p_}__{attr}__FH__" + {"full": "fullbank", "centred": "iobank"}.get(
                    c, c)  # noqa: E731
                self.bank(uid + "__bank", [uid, single(pa), single(pb)])
        # U2__A (oar)
        for ds in ("adult", "hmda"):
            W = self.world(ds)
            purpose, _, Kt, tkey = PURPOSES[ds][0]
            t = self.L[ds][tkey]
            for k in (0, 1, 2):
                X = self.F[(ds, k)]["rep_p0"]
                m = LogisticRegression(max_iter=500).fit(X[W["attacker_fit"]], t[W["attacker_fit"]])
                d = self.oar / f"{ds}__s{k}__U2__A"
                (d / "models").mkdir(parents=True, exist_ok=True)
                a = W["assessment"]
                np.savez_compressed(d / "preds.npz", assess_row_id=self.L[ds]["row_id"][a],
                                    assess_unit=self.L[ds]["unit"][a], y_t=t[a], U2_P=m.predict_proba(X[a]))
                np.savez_compressed(d / "val_preds.npz")
                (d / "record.json").write_text(json.dumps({"id": d.name}))
                write_complete(d, d.name)
        # controls
        (self.run / "controls").mkdir(parents=True, exist_ok=True)
        for ds in ("adult", "hmda"):
            c = {n: {"id": f"{ds}__CTL__{n}", "null_val_macro_auc": 0.503, "planted_val_macro_auc": 0.91,
                     "null_flag_above_0.55": False, "planted_detected_above_0.75": True} for n in ("full", "centred")}
            (self.run / "controls" / f"{ds}.json").write_text(json.dumps(c, indent=1))

    # ------------------------------------------------------------------ public files
    def build_public(self):
        st = self.study
        st.mkdir(parents=True, exist_ok=True)
        (st / "PROTOCOL.md").write_text("synthetic protocol\n")
        # coverage
        rows = []
        for (ds, purpose, attr) in R.PAIRS:
            W = self.world(ds)
            s = self.L[ds][attr]
            K = int(s.max()) + 1
            sup = []
            for c in range(K):
                cnt = {r: int((s[W[r]] == c).sum()) for r in ("defense_fit", "cert", "attacker_fit", "attacker_val",
                                                             "assessment")}
                ok = cnt["attacker_fit"] >= 100 and cnt["attacker_val"] >= 30 and cnt["assessment"] >= 100
                if ok:
                    sup.append(c)
                rows.append({"dataset": ds, "purpose": purpose, "attribute": attr, "what": "class", "id": str(c),
                             **{f"n_{r}": cnt[r] for r in cnt}, "supported": str(ok),
                             "status": "ESTIMABLE" if ok else "NOT_ESTIMABLE", "seeds": "0;1;2", "exposure": ""})
            for i in range(K):
                for j in range(i + 1, K):
                    ok = i in sup and j in sup
                    rows.append({"dataset": ds, "purpose": purpose, "attribute": attr, "what": "pair",
                                 "id": f"{i}-{j}", "supported": str(ok),
                                 "status": "ESTIMABLE" if ok else "NOT_ESTIMABLE", "seeds": "0;1;2"})
            Kt = self.index["datasets"][ds]["purposes"][purpose]["task_dim"]
            tk = self.index["datasets"][ds]["purposes"][purpose]["labels_task_key"]
            t = self.L[ds][tk]
            rows.append({"dataset": ds, "purpose": purpose, "attribute": attr, "what": "task_classes",
                         "id": json.dumps({r: np.bincount(t[W[r]], minlength=Kt).tolist()
                                           for r in ("attacker_fit", "attacker_val", "assessment")}),
                         "supported": "True", "status": "INFO"})
        cols = ["dataset", "purpose", "attribute", "what", "id", "n_defense_fit", "n_cert", "n_attacker_fit",
                "n_attacker_val", "n_assessment", "supported", "status", "seeds", "exposure"]
        with open(st / "COVERAGE_AND_SUPPORT.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in cols})
        # exactness (stand-in implementation)
        ex = {}
        for ds in ("adult", "hmda"):
            for k in (0, 1, 2):
                for (purpose, _, Kt, _) in PURPOSES[ds]:
                    L = self.logits(ds, k, purpose)
                    c = L.mean(1)
                    cen = L - c[:, None]
                    Z = L - L.max(1, keepdims=True)
                    P = np.exp(Z) / np.exp(Z).sum(1, keepdims=True)
                    Zc = cen - cen.max(1, keepdims=True)
                    Pc = np.exp(Zc) / np.exp(Zc).sum(1, keepdims=True)
                    e = {"K": Kt, "n": len(L), "reconstruct_full_from_centred_plus_offset_maxabs":
                         float(np.max(np.abs(cen + c[:, None] - L))),
                         "softmax_of_centred_minus_softmax_full_maxabs": float(np.max(np.abs(Pc - P))),
                         "offset_sd": float(np.std(c))}
                    if Kt == 2:
                        d = L[:, 1] - L[:, 0]
                        with np.errstate(over="ignore"):
                            sig = 1 / (1 + np.exp(-d))
                        e.update({"sigmoid_margin_minus_softmax_p1_maxabs": float(np.max(np.abs(sig - P[:, 1]))),
                                  "argmax_equals_margin_positive": bool(np.array_equal(L.argmax(1), (d > 0) * 1)),
                                  "ties_d_equal_0": int((d == 0).sum()),
                                  "p1_saturated_exact_0_or_1": int(((P[:, 1] == 0) | (P[:, 1] == 1)).sum()),
                                  "rows_d_ge_36_7368_float64_p1_eq_1": int((d >= 36.7368005696771).sum()),
                                  "rows_abs_d_ge_16_6355_float32_saturation": int((np.abs(d) >= 16.635532333438686
                                                                                   ).sum()),
                                  "abs_margin_max": float(np.max(np.abs(d)))})
                    ex[f"{ds}__s{k}__{purpose}"] = e
        (st / "EXACTNESS.json").write_text(json.dumps(ex, indent=1))
        # lock
        real = json.loads((HERE.parent / "LOCK.json").read_text())
        fam = json.loads(json.dumps(real["families"]))
        fam["bootstrap"]["B"] = B_SYN
        lock = {"schema": "odx_lock/v1", "built_at": "2000-01-01T00:00:00Z", "pairs": [list(p) for p in R.PAIRS],
                "primary_cells": [["adult", "income_prediction", "sex"], ["hmda", "underwriting", "race"]],
                "families": fam, "file_sha256": {"PROTOCOL.md": sha(st / "PROTOCOL.md")},
                "admitted": {"inputs": {"~/" + str(p.relative_to(self.home)): sha(p) for p in sorted(
                    self.inp.iterdir()) if p.is_file()},
                    "reused_units": {"~/" + str((d / "COMPLETE.json").relative_to(self.home)): sha(
                        d / "COMPLETE.json") for d in sorted(self.oar.iterdir())}}}
        (st / "LOCK.json").write_text(json.dumps(lock, indent=1))
        (st / "verification").mkdir(exist_ok=True)

    def build(self):
        if self.root.exists():
            shutil.rmtree(self.root)
        self.build_inputs()
        self.build_units()
        self.build_public()
        ref = Reference(self)
        ref.write_tables(self.study)
        return self


# ============================================================================== reference statistics (separate code)
def draws(n_units, B, seed=SEED):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):                      # one call per replicate (equivalent stream to chunked size= draws)
        out.append(rng.multinomial(n_units, np.full(n_units, 1.0 / n_units)))
    return np.array(out)


def ref_auc(score, pos, w):
    """Rank-sum (midrank) Mann-Whitney on the replicate-expanded sample."""
    rep = np.repeat(np.arange(len(score)), w.astype(int))
    s, p = score[rep], pos[rep]
    n1, n0 = p.sum(), (~p).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[p].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


class Reference:
    def __init__(self, sy: Synth, seeds_used=(0, 1, 2), z_primary=None, flip_pairs=False):
        self.sy = sy
        self.seeds_used = seeds_used
        self.flip = flip_pairs
        self.zp = z_primary or float(norm.ppf(1 - 0.05 / 60))
        self.W = {}
        for ds in ("adult", "hmda"):
            W = sy.world(ds)
            unit = sy.L[ds]["unit"][W["assessment"]]
            uu, inv = np.unique(unit, return_inverse=True)
            C = draws(len(uu), B_SYN)
            self.W[ds] = np.vstack([np.ones(len(inv)), C[:, inv]])
        self.cache = {}

    def P(self, uid):
        d = self.sy.odx / uid
        if (d / "ALIAS.json").exists():
            d = self.sy.home / json.loads((d / "ALIAS.json").read_text())["source"][2:]
        return np.load(d / "preds.npz")

    def sel(self, uid):
        r = json.loads((self.sy.odx / uid / "record.json").read_text())
        return r["bank_selected"] if r.get("kind") == "bank" else uid

    def support(self, ds, attr):
        W = self.sy.world(ds)
        s = self.sy.L[ds][attr]
        return [c for c in range(int(s.max()) + 1) if (s[W["attacker_fit"]] == c).sum() >= 100 and
                (s[W["attacker_val"]] == c).sum() >= 30 and (s[W["assessment"]] == c).sum() >= 100]

    def R(self, ds, uid, attr, pair=None):
        key = (uid, attr, pair)
        if key in self.cache:
            return self.cache[key]
        P = self.P(uid)
        y = P["y_s"]
        Wm = self.W[ds]
        out = np.zeros(Wm.shape[0])
        for a in range(3):
            M = P[f"P__NL__as{a}"]
            for b in range(Wm.shape[0]):
                if pair is None:
                    sup = self.support(ds, attr)
                    out[b] += np.mean([ref_auc(M[:, c], y == c, Wm[b]) for c in sup]) / 3
                else:
                    i, j = pair
                    m = (y == i) | (y == j)
                    den = M[m, i] + M[m, j]
                    num = M[m, i] if self.flip else M[m, j]
                    sc = np.where(den > 0, num / np.where(den > 0, den, 1), 0.5)
                    out[b] += ref_auc(sc, y[m] == j, Wm[b][m]) / 3   # flip: P_i numerator, j still positive
        self.cache[key] = out
        return out

    def endpoint(self, sid, vecs, target, z, identical=False, weak=False, near=False):
        T = np.mean(vecs, axis=0)
        se = 0.0 if identical else float(np.std(T[1:], ddof=1))
        lo, up = T[0] - z * se, T[0] + z * se
        flags = [f for f, on in (("IDENTICAL_BY_SELECTION", identical), ("NORMAL_APPROX_WEAK", weak),
                                 ("NEAR_BOUND", near)) if on]
        dec = "NOT_ESTABLISHED" if identical else ("PASS" if lo > target else "NOT_ESTABLISHED")
        return {"id": sid, "point": T[0], "se": se, "z": z, "lower": lo, "upper": up, "decision": dec,
                "flags": ";".join(flags)}

    def rec(self, sid, ds, purpose, attr, side, z, pair=None):
        vecs, ident, sides = [], [], []
        for k in self.seeds_used:
            base = f"{ds}__s{k}__{purpose}__{attr}__FH__"
            fb, io, hd = self.sel(base + "fullbank"), self.sel(base + "iobank"), base + "hard"
            l_, r_ = (fb, io) if side == "FC" else (io, hd)
            vl, vr = self.R(ds, l_, attr, pair), self.R(ds, r_, attr, pair)
            vecs.append(vl - vr)
            ident.append(l_ == r_ or self._same(l_, r_))
            sides += [vl[0], vr[0]]
        return self.endpoint(sid, vecs, 0.02, z, identical=all(ident), near=max(sides) > 0.98)

    def _same(self, a, b):
        return False

    def use(self, sid, ds, purpose, which, z):
        W = self.sy.world(ds)
        a = W["assessment"]
        pinfo = self.sy.index["datasets"][ds]["purposes"][purpose]
        t = self.sy.L[ds][pinfo["labels_task_key"]]
        const = int(np.argmax(np.bincount(t[W["attacker_fit"]], minlength=pinfo["task_dim"])))
        vecs, disc, accs = [], [], []
        for k in self.seeds_used:
            if which == "frozen":
                pred = self.sy.logits(ds, k, purpose)[a].argmax(1)
            else:
                pred = np.load(self.sy.oar / f"{ds}__s{k}__U2__A" / "preds.npz")["U2_P"].argmax(1)
            hr, cr = (pred == t[a]), (const == t[a])
            disc.append(int((hr != cr).sum()))
            Wm = self.W[ds]
            vecs.append((Wm @ hr - Wm @ cr) / Wm.sum(1))
            accs += [hr.mean(), cr.mean()]
        ident = all(x == 0 for x in disc)
        e = self.endpoint(sid, vecs, 0.01, z, identical=ident, weak=min(disc) < 30, near=max(accs) > 0.99)
        if ident:
            e["flags"] = e["flags"].replace("IDENTICAL_BY_SELECTION", "IDENTICAL_BY_CONSTRUCTION")
        return e

    def primary(self):
        z = self.zp
        rows = []
        cells = {"adult": ("income_prediction", "sex"), "hmda": ("underwriting", "race")}
        for ds in ("adult", "hmda"):
            p, a = cells[ds]
            rows += [self.use(f"U-frozen-{ds}", ds, p, "frozen", z), self.use(f"U-refit-{ds}", ds, p, "refit", z)]
        for ds in ("adult", "hmda"):
            p, a = cells[ds]
            rows += [self.rec(f"FC-{ds}", ds, p, a, "FC", z), self.rec(f"CH-{ds}", ds, p, a, "CH", z)]
        for side in ("FC", "CH"):
            for ds in ("adult", "hmda"):
                p, a = cells[ds]
                K = int(self.sy.L[ds][a].max()) + 1
                sup = self.support(ds, a)
                for i in range(K):
                    for j in range(i + 1, K):
                        sid = f"{side}-{ds}-pair{i}-{j}"
                        if i in sup and j in sup:
                            rows.append(self.rec(sid, ds, p, a, side, z, pair=(i, j)))
                        else:
                            rows.append({"id": sid, "point": "", "se": "", "z": z, "lower": "", "upper": "",
                                         "decision": "NOT_ESTIMABLE", "flags": ""})
        return rows

    def s3(self):
        z = float(norm.ppf(1 - 0.05 / 68))
        rows = []
        for (ds, p, a) in R.PAIRS:
            rows += [self.rec(f"S3-{s}-{ds}-{p}-{a}", ds, p, a, s, z) for s in ("FC", "CH")]
        for ds in ("adult", "hmda"):
            for (p, _, _, _) in PURPOSES[ds]:
                rows.append(self.use(f"S3-U-{ds}-{p}", ds, p, "frozen", z))
        return rows

    def s4(self):
        z = float(norm.ppf(1 - 0.05 / 12))
        ds, pa, pb, attr = R.COALITION
        rows = []
        for c in R.CONTRACTS:
            single = {"full": "fullbank", "centred": "iobank"}.get(c, c)
            for other, nm in ((pa, pa), (pb, pb)):
                vecs, ident = [], []
                for k in self.seeds_used:
                    ps = self.sel(f"{ds}__s{k}__PAIR_{pa}+{pb}__{attr}__FH__{c}__bank")
                    ss = self.sel(f"{ds}__s{k}__{other}__{attr}__FH__{single}")
                    vecs.append(self.R(ds, ps, attr) - self.R(ds, ss, attr))
                    ident.append(self._resolved_dir(ps) == self._resolved_dir(ss))
                rows.append(self.endpoint(f"S4-{c}-pair-minus-{nm}", vecs, 0.02, z, identical=all(ident)))
        return rows

    def _resolved_dir(self, uid):
        d = self.sy.odx / uid
        if (d / "ALIAS.json").exists():
            return self.sy.home / json.loads((d / "ALIAS.json").read_text())["source"][2:]
        return d

    @staticmethod
    def write(rows, path):
        cols = ["id", "point", "se", "z", "lower", "upper", "decision", "flags"]
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: (repr(float(r[c])) if isinstance(r[c], (float, np.floating)) else r[c]) for c in cols})

    def write_tables(self, st):
        self.write(self.primary(), st / "PRIMARY_ENDPOINTS.csv")
        self.write(self.s3(), st / "S3_ENDPOINTS.csv")
        self.write(self.s4(), st / "S4_ENDPOINTS.csv")


# ============================================================================== harness
def synth_expect(root: Path):
    sy = rebuild_reader(root)
    ex, ag = {}, {}
    for ds in ("adult", "hmda"):
        lab = sy.L[ds]
        tk = set(lab["canon_key"][lab["split"] == "train"].tolist())
        e = (lab["split"] == "test") & np.array([c in tk for c in lab["canon_key"]])
        by = [int((e & (lab["role"] == r)).sum()) for r in ("attacker_fit", "attacker_val", "assessment")]
        ex[ds] = [sum(by), by]
        a = sy.world(ds)["assessment"]
        ag[ds] = [int(len(a)), int(len(np.unique(lab["unit"][a])))]
    return {"B": B_SYN, "exposure": ex, "assess_rows_groups": ag}


def run_replay(root: Path, quick=True, tag="run"):
    out = root / f"IV_{tag}.json"
    agg = root / f"AGG_{tag}.json"
    rc = R.main(["--home", str(root / "home"), "--study-dir", str(root / "study"), "--repo", str(REPO),
                 "--out", str(out), "--agg-out", str(agg), "--expect", json.dumps(synth_expect(root))] +
                (["--quick"] if quick else []))
    J = json.loads(out.read_text())
    return rc, J


def status(J, cid):
    return next((it["status"] for it in J["checks"] if it["id"] == cid), None)


def failing(J):
    return sorted(it["id"] for it in J["checks"] if it["status"] == "FAIL")


def clone(clean: Path, name: str) -> Path:
    dst = clean.parent / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(clean, dst)
    return dst


def rewrite_csv(path, fn):
    rows = list(csv.DictReader(open(path)))
    cols = list(rows[0].keys())
    rows = fn(rows)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def touch_complete(d: Path):
    rec = json.loads((d / "COMPLETE.json").read_text())
    write_complete(d, rec["id"], alias=rec.get("alias", False))


def unit_test_wauc():
    rng = np.random.default_rng(3)
    n = 60
    score = np.round(rng.normal(size=n), 1)            # many ties
    pos = rng.random(n) < 0.4
    w = rng.integers(0, 4, n).astype(float)
    b = R.Boot(np.arange(n), 3, 1)
    b.W = np.vstack([np.ones(n), w, np.zeros(n)])
    got = b.wauc(score, pos)
    num = sum(w[i] * w[j] * (1.0 if score[i] > score[j] else 0.5 if score[i] == score[j] else 0.0)
              for i in range(n) if pos[i] for j in range(n) if not pos[j])
    brute = num / (w[pos].sum() * w[~pos].sum())
    num1 = sum((1.0 if score[i] > score[j] else 0.5 if score[i] == score[j] else 0.0)
               for i in range(n) if pos[i] for j in range(n) if not pos[j])
    assert abs(got[0] - num1 / (pos.sum() * (~pos).sum())) < 1e-14
    assert abs(got[1] - brute) < 1e-14
    assert np.isnan(got[2])
    sc, nz = R.pair_score(np.array([[0.0, 0.0, 1.0], [0.2, 0.6, 0.2]]), 0, 1)
    assert nz == 1 and sc[0] == 0.5 and abs(sc[1] - 0.75) < 1e-15
    # chunked multinomial == per-replicate draws
    b2 = R.Boot(np.repeat(np.arange(50), 2), 1203, SEED)
    C = draws(50, 1203)
    assert np.array_equal(b2.W[1:, ::2], C.astype(float))
    return "PASS"


def main():
    t0 = time.time()
    results = {}
    results["unit.weighted_auc_and_draws"] = unit_test_wauc()
    clean = SCRATCH / "clean"
    Synth(clean).build()
    print(f"synthetic tree built in {time.time() - t0:.0f}s")

    # ---- clean tree: replay passes everything and agrees with the reference tables
    rc, J = run_replay(clean, quick=False, tag="clean")
    bad = [it for it in J["checks"] if it["status"] in ("FAIL", "WARN")]
    results["clean.all_pass"] = "PASS" if rc == 0 and not bad else f"FAIL {[(b['id'], b['status']) for b in bad]}"
    fc = J["primary"]["FC-hmda"]
    results["clean.SE0_identical_by_selection"] = "PASS" if (
        fc["se"] == 0.0 and fc["decision"] == "NOT_ESTABLISHED" and "IDENTICAL_BY_SELECTION" in fc["flags"]) \
        else f"FAIL {fc}"
    results["clean.compare_primary_matches_reference"] = status(J, "compare.PRIMARY_ENDPOINTS")
    results["clean.compare_S3_matches_reference"] = status(J, "compare.S3")
    results["clean.compare_S4_matches_reference"] = status(J, "compare.S4")
    coll = next(it for it in J["checks"] if it["id"] == "spot.constant_and_collapsed_cases")
    results["clean.collapsed_head_detected"] = "PASS" if (
        coll["status"] == "PASS" and "hmda.fair_lending_audit.s1" in coll["detail"]) else f"FAIL {coll}"

    def defect(name, mutate, expect_fail, quick=True):
        root = clone(clean, f"defect_{name}")
        mutate(root)
        rc_, J_ = run_replay(root, quick=quick, tag=name)
        fails = failing(J_)
        hit = [e for e in expect_fail if e in fails]
        results[f"defect.{name}"] = "PASS (caught: " + ", ".join(hit) + ")" if hit and rc_ != 0 else \
            f"FAIL (not caught; fails={fails})"

    odx = lambda root: root / "home" / "PCRL_eval_cache_private" / "odx_v1" / "run" / "units"  # noqa: E731
    st = lambda root: root / "study"  # noqa: E731

    # 1 wrong bank selection
    def m_bank(root):
        d = odx(root) / "adult__s1__income_prediction__sex__FH__iobank"
        r = json.loads((d / "record.json").read_text())
        r["bank_selected"] = [c for c in r["candidates_attacker_val_log_loss"] if c != r["bank_selected"]][0]
        (d / "record.json").write_text(json.dumps(r, indent=1))
        touch_complete(d)
    defect("wrong_bank_selection", m_bank, ["banks.selection"])

    # 2a alias pointing at another seed's historical unit
    def m_alias(root):
        d = odx(root) / "adult__s0__income_prediction__sex__FH__full"
        a = json.loads((d / "ALIAS.json").read_text())
        a["source"] = a["source"].replace("adult__s0__O_full", "adult__s1__O_full")
        src = root / "home" / a["source"][2:]
        a["source_COMPLETE_sha256"] = sha(src / "COMPLETE.json")
        (d / "ALIAS.json").write_text(json.dumps(a, indent=1))
        touch_complete(d)
    defect("alias_wrong_source", m_alias, ["units.integrity"])

    # 2b declared alias row not equal to its macro twin in the runner table
    def m_alias_tab(root):
        def f(rows):
            for r in rows:
                if r["id"] == "FC-adult-pair0-1":
                    r["point"] = repr(float(r["point"]) + 1e-6)
            return rows
        rewrite_csv(st(root) / "PRIMARY_ENDPOINTS.csv", f)
    defect("alias_mismatch_table", m_alias_tab, ["compare.PRIMARY_ENDPOINTS"])

    # 3 family size other than 30 (table and lock)
    defect("family_size_table_29", lambda root: rewrite_csv(st(root) / "PRIMARY_ENDPOINTS.csv", lambda rows: rows[:-1]),
           ["compare.PRIMARY_ENDPOINTS.n_rows"])

    def m_lock_size(root):
        p = st(root) / "LOCK.json"
        L = json.loads(p.read_text())
        L["families"]["primary_size"] = 31
        L["families"]["primary_ids"].append("FC-extra")
        p.write_text(json.dumps(L))
    defect("family_size_lock_31", m_lock_size, ["lock.primary_size"])

    # 4 wrong z (table bounds at the one-sided 0.05/30 value; lock z edited)
    def m_z(root):
        rows = list(csv.DictReader(open(st(root) / "PRIMARY_ENDPOINTS.csv")))
        zw = float(norm.ppf(1 - 0.05 / 30))
        for r in rows:
            if r["se"]:
                p_, s_ = float(r["point"]), float(r["se"])
                r["z"], r["lower"], r["upper"] = repr(zw), repr(p_ - zw * s_), repr(p_ + zw * s_)
        with open(st(root) / "PRIMARY_ENDPOINTS.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    defect("wrong_z_table", m_z, ["compare.PRIMARY_ENDPOINTS"])

    def m_lockz(root):
        p = st(root) / "LOCK.json"
        L = json.loads(p.read_text())
        L["families"]["z_primary"] = 2.935199468866699
        p.write_text(json.dumps(L))
    defect("wrong_z_lock", m_lockz, ["lock.z_primary"])

    # 5 wrong pair orientation (runner scored P_i/(P_i+P_j) against positive class j; note that flipping BOTH the
    #   score and the positive class leaves the AUC unchanged, so that is not a detectable defect)
    def m_orient(root):
        ref = Reference(rebuild_reader(root), flip_pairs=True)
        ref.write(ref.primary(), st(root) / "PRIMARY_ENDPOINTS.csv")
    defect("wrong_pair_orientation", m_orient, ["compare.PRIMARY_ENDPOINTS"])

    # 6 assessment rows leaking into validation
    def m_leak(root):
        d = odx(root) / "hmda__s2__underwriting__race__FH__centred"
        V = dict(np.load(d / "val_preds.npz"))
        P = np.load(d / "preds.npz")
        V["val_row_id"] = V["val_row_id"].copy()
        V["val_row_id"][:5] = P["assess_row_id"][:5]
        np.savez_compressed(d / "val_preds.npz", **V)
        touch_complete(d)
    defect("assessment_leak_into_validation", m_leak, ["units.row_identities"])

    # 7a mismatched seed set in the runner table (seed 2 dropped)
    def m_seeds(root):
        ref = Reference(rebuild_reader(root), seeds_used=(0, 1))
        ref.write(ref.primary(), st(root) / "PRIMARY_ENDPOINTS.csv")
    defect("mismatched_seed_set_table", m_seeds, ["compare.PRIMARY_ENDPOINTS"])

    # 7b a seed's unit missing on disk while the table reports the endpoint
    def m_seed_unit(root):
        shutil.rmtree(odx(root) / "hmda__s2__underwriting__race__FH__centred")
    defect("missing_seed_unit", m_seed_unit, ["compare.PRIMARY_ENDPOINTS"])

    # 8 wrong decision
    def m_dec(root):
        def f(rows):
            for r in rows:
                if r["decision"] == "NOT_ESTABLISHED" and r["id"].startswith("CH-hmda-pair"):
                    r["decision"] = "PASS"
                    break
            return rows
        rewrite_csv(st(root) / "PRIMARY_ENDPOINTS.csv", f)
    defect("wrong_decision", m_dec, ["compare.PRIMARY_ENDPOINTS"])

    # 9 SE = 0 case reported as PASS without the identical-by-selection flag
    def m_se0(root):
        def f(rows):
            for r in rows:
                if r["id"] == "FC-hmda":
                    r["decision"], r["flags"], r["lower"] = "PASS", "", "0.05"
            return rows
        rewrite_csv(st(root) / "PRIMARY_ENDPOINTS.csv", f)
    defect("se0_reported_as_pass", m_se0, ["compare.PRIMARY_ENDPOINTS"])

    # extra: coverage count, exactness claim, planted control, mislabeled y_s, stored preds not reproducible
    def m_cov(root):
        def f(rows):
            for r in rows:
                if r["what"] == "class" and r["dataset"] == "hmda" and r["attribute"] == "race" and r["id"] == "3":
                    r["n_attacker_fit"] = "100"
                    r["supported"], r["status"] = "True", "ESTIMABLE"
                    break
            return rows
        rewrite_csv(st(root) / "COVERAGE_AND_SUPPORT.csv", f)
    defect("coverage_count", m_cov, ["coverage.counts_and_support"])

    def m_exact(root):
        p = st(root) / "EXACTNESS.json"
        X = json.loads(p.read_text())
        X["adult__s0__income_prediction"]["argmax_equals_margin_positive"] = False
        p.write_text(json.dumps(X))
    defect("exactness_claim", m_exact, ["exactness.EXACTNESS_json"])

    def m_ctl(root):
        p = root / "home" / "PCRL_eval_cache_private" / "odx_v1" / "run" / "controls" / "hmda.json"
        C = json.loads(p.read_text())
        C["centred"]["planted_val_macro_auc"], C["centred"]["planted_detected_above_0.75"] = 0.6, False
        p.write_text(json.dumps(C))
    defect("planted_control_not_detected", m_ctl, ["controls.hmda"])

    def m_preds(root):
        d = odx(root) / "adult__s0__income_prediction__sex__FH__centred"
        P = dict(np.load(d / "preds.npz"))
        P["P__NL__as1"] = P["P__NL__as1"][::-1].copy()
        np.savez_compressed(d / "preds.npz", **P)
        touch_complete(d)
    defect("stored_preds_not_from_surface", m_preds, ["models.reproduce_stored_predictions",
                                                      "compare.PRIMARY_ENDPOINTS"], quick=False)

    print(json.dumps(results, indent=1))
    print(f"total {time.time() - t0:.0f}s")
    ok = all(str(v).startswith("PASS") for v in results.values())
    (SCRATCH / "SELF_TEST_RESULTS.json").write_text(json.dumps(results, indent=1))
    return 0 if ok else 1


def rebuild_reader(root: Path) -> Synth:
    """Re-open a cloned synthetic tree (inputs read back from disk) for the reference implementation."""
    sy = Synth(root)
    sy.index = json.loads((sy.inp / "INPUTS_INDEX.json").read_text())
    sy.L = {ds: dict(np.load(sy.inp / f"{ds}_labels.npz")) for ds in ("adult", "hmda")}
    sy.F = {(ds, k): dict(np.load(sy.inp / f"{ds}_s{k}_forward.npz")) for ds in ("adult", "hmda") for k in (0, 1, 2)}
    return sy


def test_synthetic_self_test():
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
