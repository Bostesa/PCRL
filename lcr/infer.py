"""Inference from saved OSF_DEVELOPMENT_ASSESSMENT predictions and EVALUATION_LOCK.json only (no refits, no selection).

Adapted from cbp/infer.py at 7f3ec67; the family (37 slots, z = 3.2048452050105634), clause classifier and label truth
table come from lcr.family. Per model seed: recovery = SEX AUC (score = P(SEX=1), fixed orientation) of the
inner-AUC-selected final attacker, averaged over attacker seeds 0-2; accuracy = released decisions; true-label log loss
= mean -log(clip(prob_y, 1e-12)); Brier = mean sum_k (prob_k - 1[y=k])^2; const = OSF_DEFENSE_FIT majority class;
U = the task-only teacher's continuous output (label "SRC|U"). Roles (P*, N*, J*, T*, C*, C_pair*, Q) are GLOBAL
configurations resolved from the lock; a registered fallback is scored but its rows are DESCRIPTIVE_ONLY and can never
pass. Endpoint = mean over seeds of the per-seed paired statistic. SE = sd (ddof 1) over B = 1999 paired multinomial
bootstrap replicates of exact-record groups (seed 20261009; the same draws for every statistic, arm and seed);
interval = point +- z SE. Every slot gets a clause outcome (PASS / NOT_ESTABLISHED_PRECISION / NOT_ESTABLISHED_POINT /
MEASURED_VIOLATION / INVALID); a slot with any nonfinite replicate is INVALID (never dropped).

Technical failures found after the lock (e.g. an independent-verification FAIL) are passed with --failures FILE:
{"global": [...], "per_claim": {"A": [...], ...}}; global entries block every favourable label, per-claim entries make
only that claim INCOMPLETE_OR_INVALID (LABEL_TRUTH_TABLE.json failure_scope).

    OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m lcr.infer --evaluation-lock results/pcrl_learned_decoder_constrained_release_v1/EVALUATION_LOCK.json
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from jcv.infer import G, acc_stat
from stored_model_eval.bench_infer import class_auc, run
from stored_model_eval.pilot_infer import UnitBootstrap

from lcr import family as FAM
from lcr import run as R

SEEDS = (0, 1, 2)
ROLES = ("P*", "N*", "J*", "T*", "C*", "C_pair*", "Q")
EPS = 1e-12


def safe(label):
    return str(label).replace("*", "star").replace("/", "_").replace("|", "_").replace(" ", "_")


def row_losses(prob, y):
    prob = np.asarray(prob, dtype=np.float64)
    ll = -np.log(np.clip(prob[np.arange(len(y)), y], EPS, 1.0))
    br = ((prob - np.eye(prob.shape[1])[y]) ** 2).sum(1)
    return ll, br


class Ctx:
    def __init__(self, EL, check_prior=True, units=None):
        self.EL, self.g, self.preds = EL, G(), {}
        self.labels = list(EL["seeds"]["0"]["score"])
        units = Path(units) if units else R.UNITS
        z0 = None
        for k in SEEDS:
            assert list(EL["seeds"][str(k)]["score"]) == self.labels
            for lab in self.labels:
                z = np.load(units / f"outer__s{k}__{safe(lab)}" / "preds.npz", allow_pickle=False)
                p = {x: z[x] for x in z.files}
                self.preds[(k, lab)] = p
                z0 = z0 if z0 is not None else p
                assert np.array_equal(p["assess_row_id"], z0["assess_row_id"]), "assessment rows differ"
                assert np.array_equal(p["assess_unit"], z0["assess_unit"]), "assessment groups differ"
                for key in ("sex", "y_income", "y_occ", "const_class"):
                    assert np.array_equal(p[key], z0[key]), f"{key} differs across arms"
        self.finiteness = {f"s{k}|{lab}": {x: int((~np.isfinite(np.asarray(p[x], dtype=np.float64))).sum())
                                            for x in p if x.startswith(("P_auc_", "P_ce_", "prob", "hard"))}
                           for (k, lab), p in self.preds.items()}
        self.units, self.sex, self.rows = z0["assess_unit"], z0["sex"], z0["assess_row_id"]
        self.y = {0: z0["y_income"], 1: z0["y_occ"]}
        self.const = {j: int(z0["const_class"][j]) for j in (0, 1)}
        if check_prior:
            from lcr import data as DA
            from lcr.eval_lock import prior_hash
            assert prior_hash(DA.load()) == EL["sex_prior_defense_fit_sha256"], "fitting prior differs from the lock"

    def lab(self, x):
        x = self.EL["resolved"].get(x) if x in ROLES else x
        return x if x in self.labels else None

    def rec(self, k, lab, view, fam=None):
        key = f"P_auc_{view}" if fam is None else f"P_auc_{fam}_{view}"
        P3 = np.asarray(self.preds[(k, lab)][key], dtype=np.float64)
        if not np.isfinite(P3[..., 1]).all():          # nonfinite scores are INVALID, never ranked as extreme values
            nan = lambda WT: np.full(WT.shape[1], np.nan)                       # noqa: E731
            ids = [self.g.base_once(f"auc#{k}#{lab}#{fam}#{view}#{s}", f"auc#{k}#{lab}#{fam}#{view}#{s}", nan)
                   for s in range(3)]
            return self.g.add(f"R#{k}#{lab}#{fam}#{view}", "mean", ids)
        ids = [self.g.base_once(f"auc#{k}#{lab}#{fam}#{view}#{s}", f"auc#{k}#{lab}#{fam}#{view}#{s}",
                                class_auc(self.sex, P3[s], 1)) for s in range(3)]
        return self.g.add(f"R#{k}#{lab}#{fam}#{view}", "mean", ids)

    def has(self, k, lab, view, fam):
        return (f"P_auc_{view}" if fam is None else f"P_auc_{fam}_{view}") in self.preds[(k, lab)]

    def acc(self, k, lab, j):
        p = self.preds[(k, lab)]
        return self.g.base_once(f"acc#{k}#{lab}#{j}", f"acc#{k}#{lab}#{j}", acc_stat(p[f"hard{j + 1}"] == self.y[j]))

    def loss(self, k, lab, j, kind):
        p = self.preds[(k, lab)]
        ll, br = row_losses(p[f"prob{j + 1}"], self.y[j])
        return self.g.base_once(f"{kind}#{k}#{lab}#{j}", f"{kind}#{k}#{lab}#{j}", acc_stat(ll if kind == "ll" else br))

    def constacc(self, j):
        return self.g.base_once(f"const#{j}", f"const#{j}", acc_stat(self.y[j] == self.const[j]))


def build(ctx):
    g, ids = ctx.g, {}
    U = "SRC|U"
    for e in FAM.PRIMARY:
        nom = ctx.lab(e["nominee"])
        ref = ctx.lab(e["ref"]) if "ref" in e else None
        if nom is None or ("ref" in e and ref is None):
            ids[e["id"]] = None
            continue
        per = []
        for k in SEEDS:
            j = e.get("task")
            if e["kind"] == "coalition":
                per.append(g.add(f"{e['id']}#{k}", "diff", [ctx.rec(k, ref, "pair"), ctx.rec(k, nom, "pair")]))
            elif e["kind"] == "local":
                per.append(g.add(f"{e['id']}#{k}", "diff", [ctx.rec(k, nom, e["view"]), ctx.rec(k, ref, e["view"])]))
            elif e["kind"] == "acc":
                per.append(g.add(f"{e['id']}#{k}", "diff", [ctx.acc(k, nom, j), ctx.acc(k, U, j)]))
            elif e["kind"] == "logloss":
                per.append(g.add(f"{e['id']}#{k}", "diff", [ctx.loss(k, nom, j, "ll"), ctx.loss(k, U, j, "ll")]))
            elif e["kind"] == "brier":
                per.append(g.add(f"{e['id']}#{k}", "diff", [ctx.loss(k, nom, j, "br"), ctx.loss(k, U, j, "br")]))
            else:
                per.append(g.add(f"{e['id']}#{k}", "lin", [ctx.acc(k, nom, j), ctx.acc(k, U, j), ctx.constacc(j)]))
        ids[e["id"]] = g.add(e["id"], "mean", per)
    levels = {}
    fams = [None] + sorted({k[len("P_auc_"):].rsplit("_", 1)[0] for p in ctx.preds.values() for k in p
                            if k.startswith("P_auc_") and k[len("P_auc_"):] not in ("v1", "v2", "pair")})
    for lab in ctx.labels:
        for k in SEEDS:
            for fam in fams:
                for v in ("v1", "v2", "pair"):
                    if ctx.has(k, lab, v, fam):
                        levels[f"R#{k}#{lab}#{fam or 'primary'}#{v}"] = ctx.rec(k, lab, v, fam)
            for j in (0, 1):
                levels[f"acc#{k}#{lab}#{j}"] = ctx.acc(k, lab, j)
                levels[f"ll#{k}#{lab}#{j}"] = ctx.loss(k, lab, j, "ll")
                levels[f"br#{k}#{lab}#{j}"] = ctx.loss(k, lab, j, "br")
                for kind, kk in (("ll", "llx"), ("br", "brx")):          # excess over U, per seed
                    if lab != U:
                        levels[f"{kk}#{k}#{lab}#{j}"] = g.add(f"{kk}#{k}#{lab}#{j}", "diff",
                                                              [ctx.loss(k, lab, j, kind), ctx.loss(k, U, j, kind)])
        for fam in fams:
            for v in ("v1", "v2", "pair"):
                if all(ctx.has(k, lab, v, fam) for k in SEEDS):
                    levels[f"Rmean#{lab}#{fam or 'primary'}#{v}"] = g.add(f"Rmean#{lab}#{fam}#{v}", "mean",
                                                                         [ctx.rec(k, lab, v, fam) for k in SEEDS])
        for j in (0, 1):
            for kind, fn in (("acc", lambda k: ctx.acc(k, lab, j)), ("ll", lambda k: ctx.loss(k, lab, j, "ll")),
                             ("br", lambda k: ctx.loss(k, lab, j, "br"))):
                levels[f"{kind}mean#{lab}#{j}"] = g.add(f"{kind}mean#{lab}#{j}", "mean", [fn(k) for k in SEEDS])
            if lab != U:
                for kk in ("llx", "brx"):
                    levels[f"{kk}mean#{lab}#{j}"] = g.add(f"{kk}mean#{lab}#{j}", "mean",
                                                          [levels[f"{kk}#{k}#{lab}#{j}"] for k in SEEDS])
    for j in (0, 1):
        levels[f"const#{j}"] = ctx.constacc(j)
    return ids, levels


def label_from(out, EL, failures=None):
    """Claim statuses, Q status and the overall label from the clause outcomes (lcr.family truth table)."""
    st = EL["statuses"]
    failures = failures or {}
    per_claim = failures.get("per_claim") or {}
    outc = {e["id"]: e["outcome"] for e in out["primary"]}
    claims = {}
    for claim, (nom, ref) in FAM.CLAIMS.items():
        o = {i: outc[i] for i in FAM.claim_ids(claim)}
        s, cause, failing = FAM.claim_status(FAM.role_state(st.get(nom)), FAM.role_state(st.get(ref)), o,
                                             required_control_ok=not per_claim.get(claim),
                                             nominee_reason=(st.get(nom) or {}).get("reason"), claim=claim)
        by, detail = FAM.cause_by_kind(o)
        claims[claim] = {"status": s, "root_cause": cause, "failing": failing, "by_kind": by,
                         "cause_detail": detail, "nominee": (st.get(nom) or {}).get("config") or
                         (st.get(nom) or {}).get("descriptive_config"),
                         "comparator": (st.get(ref) or {}).get("config") or (st.get(ref) or {}).get("descriptive_config"),
                         "control_failures": per_claim.get(claim) or []}
    qo = {i: outc[i] for i in FAM.claim_ids("Q")}
    qs, qc, qf = FAM.q_status("TECHNICAL_FAILURE" if per_claim.get("Q") else FAM.role_state(st.get("Q")), qo)
    q = {"status": qs, "root_cause": qc, "failing": qf, "by_kind": FAM.cause_by_kind(qo)[0]}
    tv = bool(EL.get("technical_validity", {}).get("ok", False)) and not failures.get("global")
    lab, shown = FAM.overall_label({c: v["status"] for c, v in claims.items()}, qs, tv,
                                   (st.get("P*") or {}).get("winning"), gate_met=True)
    return claims, q, lab, shown, tv


def main(argv=None, check_prior=True, units=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluation-lock", required=True)
    ap.add_argument("--failures", default=None)
    a = ap.parse_args(argv)
    EL = json.loads(Path(a.evaluation_lock).read_text())
    failures = json.loads(Path(a.failures).read_text()) if a.failures else None
    ctx = Ctx(EL, check_prior=check_prior, units=units)
    ids, levels = build(ctx)
    boot = UnitBootstrap(ctx.units, FAM.B, FAM.BOOT_SEED, 250)
    want = [i for i in ids.values() if i] + list(levels.values())
    pts, reps = run(ctx.g, boot, list(dict.fromkeys(want)))
    z = FAM.Z_PRIMARY
    alias = EL.get("alias_of_by_role") or {}
    out = {"schema": "lcr-inference-v1", "primary": [], "levels": {}, "B": FAM.B, "seed": FAM.BOOT_SEED, "z": z,
           "resampling_unit": "OSF_DEVELOPMENT_ASSESSMENT exact-record group", "n_assessment": int(len(ctx.units)),
           "n_groups": int(len(np.unique(ctx.units))), "resolved": EL["resolved"], "statuses": EL["statuses"],
           "role_aliases": EL.get("role_aliases") or {}}
    for e in FAM.PRIMARY:
        sid = ids[e["id"]]
        base = {**e, "z": z, "alias_of_by_role": alias.get(e["id"])}
        if sid is None:
            out["primary"].append({**base, "point": None, "outcome": "INVALID", "decision": "INVALID",
                                   "reason": "nominee or comparator configuration absent (technical)"})
            continue
        r = reps[sid]
        nonfinite = int((~np.isfinite(r)).sum())
        pt = float(pts[sid])
        if nonfinite or not np.isfinite(pt):
            out["primary"].append({**base, "point": pt, "outcome": "INVALID", "decision": "INVALID",
                                   "nonfinite_replicates": nonfinite})
            continue
        se = float(np.std(r, ddof=1))
        lo, hi = pt - z * se, pt + z * se
        oc = FAM.clause_outcome(e["side"], e["target"], pt, lo, hi)
        row = {**base, "point": pt, "se": se, "lower": lo, "upper": hi, "outcome": oc, "decision": oc,
               "n_finite_replicates": int(len(r))}
        roles = [e["nominee"]] + ([e["ref"]] if "ref" in e else [])
        if any((EL["statuses"].get(x) or {}).get("status") != "NOMINEE" for x in roles):
            row["decision"] = "DESCRIPTIVE_ONLY"
            row["fallback_rank_status"] = {x: (EL["statuses"].get(x) or {}).get("fallback_rank_status") for x in roles
                                           if (EL["statuses"].get(x) or {}).get("status") != "NOMINEE"}
        out["primary"].append(row)
    for nm, sid in levels.items():
        r = reps[sid]
        fin = r[np.isfinite(r)]
        out["levels"][nm] = {"point": float(pts[sid]), "se": float(np.std(fin, ddof=1)) if len(fin) > 1 else None,
                             "nonfinite": int((~np.isfinite(r)).sum())}
    claims, q, lab, shown, tv = label_from(out, EL, failures)
    out.update({"claim_status": claims, "q_status": q, "label": lab, "displayed_statuses": shown,
                "technical_valid": tv, "failures_input": failures,
                "winning": (EL["statuses"].get("P*") or {}).get("winning"),
                "clauses_passing_scored": {c: sum(e["decision"] == "PASS" for e in out["primary"] if e["claim"] == c)
                                           for c in "ABCQ"},
                "clauses_passing_numeric_including_descriptive": {
                    c: sum(e["outcome"] == "PASS" for e in out["primary"] if e["claim"] == c) for c in "ABCQ"},
                "finiteness_receipt": {"nonfinite_counts": ctx.finiteness,
                                       "all_finite": not any(v for d in ctx.finiteness.values() for v in d.values())}})
    out = R._finite(out)
    (R.RUN / "inference.json").write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")
    cols = ["id", "claim", "kind", "stat", "target", "side", "point", "se", "lower", "upper", "z", "outcome",
            "decision", "alias_of", "alias_of_by_role", "fallback_rank_status"]
    with open(R.PKG / "PRIMARY_ENDPOINTS.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for e in out["primary"]:
            w.writerow({c: (f"{e[c]:.6f}" if isinstance(e.get(c), float) else e.get(c)) for c in cols})
    with open(R.PKG / "ALL_LEVELS.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["quantity", "seed", "label", "detail", "point", "se"])
        for nm, v in sorted(out["levels"].items()):
            parts = nm.split("#")
            if parts[0] in ("R", "acc", "ll", "br", "llx", "brx"):
                seed, lab_, det = parts[1], parts[2], "|".join(parts[3:])
            elif parts[0] == "const":
                seed, lab_, det = "", "const", parts[1]
            else:
                seed, lab_, det = "mean", parts[1], "|".join(parts[2:])
            f6 = lambda x: "" if x is None else f"{x:.6f}"                        # noqa: E731
            w.writerow([parts[0], seed, lab_, det, f6(v["point"]), f6(v["se"])])
    return out


if __name__ == "__main__":
    main()
