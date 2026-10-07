"""Registered inner-only selection of the held-out calibration study (hcal; prompt section 9; SELECTION_RULES.json;
frozen in SCIENCE_LOCK). No assessment value is read.

Inputs (private units written by hcal.stages): util__s{k} (inner utility of every registered release: codes, U variants;
qpc.utility conventions on INNER_SELECTION; decision-preservation flags), the audit plan, com__s{k}__<partition> (the
common attack-bank record shared by every decoder variant of the partition) and com__s{k}__SRC_U (continuous U's
composed record, shared by its three decoder variants).

UTILITY GATES (each task, each source seed; inclusive; no tolerance; no 0.006 buffer):
  U0 gate      qpc.utility.gate_record(code, U0, preserved)["eligible"]: acc >= U0 - 0.01, LL <= U0 + 0.01,
               Brier <= U0 + 0.005, gain >= 0.8 U0 gain, gain >= 0.03, exact decision preservation
  Ucal gate    LL <= Ucal* + 0.01 and Brier <= Ucal* + 0.005, in the qpc margin form (Ucal* + allowance) - value >= 0
               (calibration-matched; new, separate from the original eleven clauses)
Ucal*: among U identity, H-GLOBAL-TEMP, H-CLASS-TEMP, the smallest mean over seeds of (income + occupation) INNER NLL
(float64 (s0 + s1 + s2) / 3); a candidate within 1e-12 of the minimum ties, and ties prefer identity, then global, then
class; finite vectors and verified decision preservation required. One family across seeds. "NLL" here and in the
"mean summed task NLL" tie key is the SOURCE SCORING log loss (qpc.utility per-row metrics: natural log of the true-class
probability clipped at 1e-12), the registered evaluation convention; the unclipped NLL is only the calibrators' fitting
objective (MATH_REVIEW finding 1, made explicit before SCIENCE_LOCK).
AUDIT PLAN (the only scientific pruning rule): a partition is audited iff some non-diagnostic variant passes BOTH gates
on every task and seed, plus the always-audited task-only (DIRECT-TASK, FINE-TASK, CLASS, C-TASK) and diagnostic
(DIRECT-TASK, FINE-TASK, CLASS, JOINT lambda 0.1) partitions; others are PREDECLARED_UTILITY_INELIGIBLE.
T*: lowest mean inner pair AUC among the eligible (both gates, every seed) task-only candidates (hcal.ids.
task_only_candidate); ties: lowest worst-seed normalised excess (max over tasks, seeds and both references of
max((LL - LL_ref) / 0.01, (B - B_ref) / 0.005)), then lowest mean summed task NLL, then registered ID order.
P*: among eligible nominee-capable releases (hcal.ids.nominee_capable) with every recipient's inner AUC <= T*'s + 0.005
on every seed and mean inner pair benefit AUC(T*) - AUC(P*) >= 0.02: lowest mean pair AUC, then the same tie keys.
Ordering keys (not the gates) are float64 seed means (s0 + s1 + s2) / 3 compared after round(x, 12); gates and guards use
the exact margin form. Selection runs only on technically valid inputs: if the real-data controls are not all_ok, every
role is recorded TECHNICAL_FAILURE (rankings kept descriptively). Every rejection and margin is recorded.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math

import numpy as np

from hcal import ids as I

TASKS = ("income", "occupation")
VIEWS = ("v1", "v2", "pair")
UCAL_TIE = 1e-12
GUARD = 0.005
BENEFIT = 0.02
ALLOW_LL, ALLOW_BRIER = 0.01, 0.005
UCAL_ORDER = ("identity", "H-GLOBAL-TEMP", "H-CLASS-TEMP")


def _fin(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def mean3(vals):
    vals = [float(v) for v in vals]
    return (vals[0] + vals[1] + vals[2]) / 3.0


def r12(x):
    return round(float(x), 12)


def controls_all_ok(ctl):
    """The REQUIRED controls verdict of hcal.controls.summarise_controls (verdict.all_ok; common-bank checks,
    null calibration, ROT and the realised-threshold receipt). Missing -> False (fail safe)."""
    return bool(((ctl or {}).get("verdict") or {}).get("all_ok") is True)


def apply_controls(sel, ctl):
    """Controls must pass before nomination (PROTOCOL section 5): otherwise every role is TECHNICAL_FAILURE and the
    would-be statuses are kept descriptively."""
    sel["controls_all_ok"] = controls_all_ok(ctl)
    if not sel["controls_all_ok"]:
        for role in ("T*", "P*"):
            sel[role]["status_if_controls_had_passed"] = sel[role]["status"]
            sel[role]["status"] = "TECHNICAL_FAILURE"
    sel["T*"]["is_continuous_U"] = bool(sel["T*"].get("release") and sel["T*"]["release"].startswith(I.U_ID))
    return sel


def u_rid(fam):
    return I.U_ID if fam == "identity" else f"{I.U_ID}|{fam}"


# ------------------------------------------------------------------ Ucal*
def ucal_star(util):
    """util[k][rid] = {"inner": {task: metrics}, "preserved": {...}, "finite": bool}. Returns the Ucal* record."""
    rows = []
    for fam in UCAL_ORDER:
        rid = u_rid(fam)
        per = []
        ok = True
        for k in I.SEEDS:
            u = util[k].get(rid)
            if u is None or not u.get("finite", False) or not all(bool(v) for v in u["preserved"].values()):
                ok = False
                per.append(None)
                continue
            per.append(float(u["inner"]["income"]["logloss"]) + float(u["inner"]["occupation"]["logloss"]))
        val = mean3(per) if ok and all(_fin(x) for x in per) else None
        rows.append({"family": fam, "release": rid, "per_seed_summed_nll": per, "mean": val, "valid": val is not None})
    valid = [r for r in rows if r["valid"]]
    if not valid:
        return {"status": "TECHNICAL_FAILURE", "rows": rows, "family": None}
    best = min(r["mean"] for r in valid)
    tied = [r for r in valid if r["mean"] <= best + UCAL_TIE]
    win = tied[0]                                  # registered order: identity, global, class
    return {"status": "SELECTED", "family": win["family"], "release": win["release"], "rows": rows,
            "tied_within_1e-12": [r["family"] for r in tied],
            "rule": "smallest mean over seeds of INNER income + occupation NLL; ties within 1e-12 -> identity, "
                    "H-GLOBAL-TEMP, H-CLASS-TEMP; one family across seeds"}


# ------------------------------------------------------------------ gates
def gate_seed(u, u0, ucal):
    """One seed. u, u0, ucal: util entries. Returns {u0: gate_record, ucal: per task {ll_ok, brier_ok, ll_excess,
    brier_excess}, eligible_u0, eligible_ucal, eligible}."""
    from qpc import utility as UT
    pres = {int(i): bool(v) for i, v in u["preserved"].items()}
    gr = UT.gate_record(u["inner"], u0["inner"], pres)
    uc = {}
    for t in TASKS:
        dll = float(u["inner"][t]["logloss"]) - float(ucal["inner"][t]["logloss"])
        dbr = float(u["inner"][t]["brier"]) - float(ucal["inner"][t]["brier"])
        # margin form exactly as qpc.utility.gate: (ref + allowance) - value >= 0
        m_ll = (float(ucal["inner"][t]["logloss"]) + ALLOW_LL) - float(u["inner"][t]["logloss"])
        m_br = (float(ucal["inner"][t]["brier"]) + ALLOW_BRIER) - float(u["inner"][t]["brier"])
        uc[t] = {"ll_excess": dll, "brier_excess": dbr, "ll_margin": m_ll, "brier_margin": m_br,
                 "ll_ok": m_ll >= 0, "brier_ok": m_br >= 0}
    e_ucal = all(uc[t]["ll_ok"] and uc[t]["brier_ok"] for t in TASKS)
    e_u0 = bool(gr["eligible"])
    return {"u0": {t: {k: gr[t][k] for k in ("acc_ok", "ll_ok", "brier_ok", "retention_ok", "gain_ok", "preserved",
                                              "ll_excess", "brier_excess", "norm_excess")} for t in TASKS},
            "ucal": uc, "eligible_u0": e_u0, "eligible_ucal": e_ucal, "eligible": e_u0 and e_ucal and u["finite"]}


def norm_excess(u, refs):
    """max over tasks and the given reference utilities of max((LL - LL_ref) / 0.01, (B - B_ref) / 0.005)."""
    out = -math.inf
    for ref in refs:
        for t in TASKS:
            out = max(out, (float(u["inner"][t]["logloss"]) - float(ref["inner"][t]["logloss"])) / ALLOW_LL,
                      (float(u["inner"][t]["brier"]) - float(ref["inner"][t]["brier"])) / ALLOW_BRIER)
    return out


def utility_table(util, ucal_fam):
    """Per release: per-seed gates, eligibility on every seed, worst-seed normalised excess, mean summed NLL."""
    urid = u_rid(ucal_fam)
    rids = list(util[I.SEEDS[0]])
    out = {}
    for rid in rids:
        seeds, ok = {}, True
        for k in I.SEEDS:
            u = util[k][rid]
            g = gate_seed(u, util[k][I.U_ID], util[k][urid])
            seeds[str(k)] = g
            ok = ok and g["eligible"]
        ne = max(norm_excess(util[k][rid], (util[k][I.U_ID], util[k][urid])) for k in I.SEEDS)
        snll = mean3([float(util[k][rid]["inner"]["income"]["logloss"]) + float(util[k][rid]["inner"]["occupation"]
                                                                                 ["logloss"]) for k in I.SEEDS])
        out[rid] = {"eligible_all_seeds": ok, "eligible_u0_all_seeds": all(seeds[str(k)]["eligible_u0"]
                                                                             for k in I.SEEDS),
                    "eligible_ucal_all_seeds": all(seeds[str(k)]["eligible_ucal"] for k in I.SEEDS),
                    "worst_norm_excess": ne, "mean_summed_nll": snll, "seeds": seeds}
    return out


def audit_plan(table):
    """The registered pruning rule (module docstring)."""
    always = list(dict.fromkeys(list(I.TASK_ONLY_PARTITIONS) + list(I.DIAGNOSTIC_PARTITIONS)))
    plan = {}
    for p in I.partitions():
        elig = [I.release_id(p, d) for d in I.decoders_of(p) if d != I.DIAG_DECODER
                and table[I.release_id(p, d)]["eligible_all_seeds"]]
        if elig:
            plan[p] = {"audit": True, "reason": "UTILITY_ELIGIBLE_VARIANT", "eligible_variants": elig,
                       "always_audited": p in always}
        elif p in always:
            plan[p] = {"audit": True, "reason": "ALWAYS_AUDITED_TASK_ONLY_OR_DIAGNOSTIC", "eligible_variants": []}
        else:
            plan[p] = {"audit": False, "reason": "PREDECLARED_UTILITY_INELIGIBLE", "eligible_variants": []}
    return {"rule": "audit iff some non-diagnostic variant passes the U0 and Ucal* inner gates on every task and seed; "
                    "plus the task-only and fixed diagnostic partitions",
            "always_audited": always, "partitions": plan,
            "audited": [p for p in I.partitions() if plan[p]["audit"]],
            "skipped": [p for p in I.partitions() if not plan[p]["audit"]]}


# ------------------------------------------------------------------ recovery
def recovery(com, rid):
    """{view: [auc per seed]} for a release: its partition's common record (codes) or U's composed record."""
    p, _ = I.parse_release(rid) if not rid.startswith(I.U_ID) else (None, None)
    key = "SRC|U" if p is None else p
    return {w: [float(com[k][key]["auc"][w]) for k in I.SEEDS] for w in VIEWS}


def _order_key(row):
    return (r12(row["mean_pair_auc"]), r12(row["worst_norm_excess"]), r12(row["mean_summed_nll"]), row["order"])


def choose(table, com, audited, pool, ref=None):
    """Rank the eligible members of `pool`. With ref (T*'s recovery), apply the guard and the benefit gate."""
    order = {rid: n for n, rid in enumerate(I.code_release_ids() + I.u_release_ids())}
    rows, rejected = [], []
    for rid in pool:
        p = None if rid.startswith(I.U_ID) else I.parse_release(rid)[0]
        t = table[rid]
        base = {"release": rid, "partition": p, "order": order[rid], "worst_norm_excess": t["worst_norm_excess"],
                "mean_summed_nll": t["mean_summed_nll"]}
        if not t["eligible_all_seeds"]:
            fails = sorted({f"seed{k}:{'U0' if not t['seeds'][str(k)]['eligible_u0'] else 'Ucal'}"
                            for k in I.SEEDS if not t["seeds"][str(k)]["eligible"]})
            rejected.append({**base, "reason": "UTILITY_INELIGIBLE", "failing": fails})
            continue
        if p is not None and p not in audited:
            rejected.append({**base, "reason": "NOT_AUDITED_TECHNICAL"})
            continue
        rec = recovery(com, rid)
        row = {**base, "auc_per_seed": rec, "mean_pair_auc": mean3(rec["pair"]),
               "mean_v1_auc": mean3(rec["v1"]), "mean_v2_auc": mean3(rec["v2"])}
        if ref is not None:
            guard = {w: [rec[w][k] - ref[w][k] for k in range(3)] for w in ("v1", "v2")}
            gmarg = {w: [(ref[w][k] + GUARD) - rec[w][k] for k in range(3)] for w in ("v1", "v2")}   # margin form
            g_ok = all(x >= 0 for w in gmarg for x in gmarg[w])
            benefit = mean3([ref["pair"][k] - rec["pair"][k] for k in range(3)])
            row.update({"guard_differences": guard, "guard_margins": gmarg, "guard_ok": g_ok, "pair_benefit": benefit,
                        "benefit_ok": benefit >= BENEFIT})
            if not g_ok:
                rejected.append({**row, "reason": "LOCAL_GUARD_FAILURE"})
                continue
            if not row["benefit_ok"]:
                rejected.append({**row, "reason": "PAIR_BENEFIT_BELOW_0.02"})
                continue
        rows.append(row)
    rows.sort(key=_order_key)
    return rows, rejected


def release_identity(k, rid, tables_fn, D):
    """sha256 of the deployed code release (tokens, released vectors, decisions) on the permitted rows (OSF_DEFENSE_FIT
    + AUDIT_FIT + INNER_SELECTION); None for continuous U."""
    if rid.startswith(I.U_ID):
        return None
    pos = np.concatenate([np.asarray(D["idx"][r], dtype=np.int64) for r in ("OSF_DEFENSE_FIT", "AUDIT_FIT",
                                                                            "INNER_SELECTION")])
    h = hashlib.sha256()
    tok, q, hard = tables_fn(k, rid)
    for i in (1, 2):
        for a in (tok[i][pos], q[i][pos], hard[i][pos]):
            a = np.ascontiguousarray(a)
            h.update(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    return h.hexdigest()


def select(util, com, audited, tables_fn=None, D=None):
    """Full registered selection (no assessment value). Returns the SELECTION record."""
    uc = ucal_star(util)
    if uc["status"] != "SELECTED":
        return {"status": "TECHNICAL_FAILURE", "ucal_star": uc}
    table = utility_table(util, uc["family"])
    ids = list(util[I.SEEDS[0]])
    t_pool = [r for r in ids if I.task_only_candidate(r)]
    t_rows, t_rej = choose(table, com, audited, t_pool)
    out = {"schema": "hcal-selection-v1", "ucal_star": uc,
           "T*": {"status": "NOMINEE" if t_rows else "NO_ELIGIBLE", "release": t_rows[0]["release"] if t_rows else None,
                  "ranking": t_rows, "rejected": t_rej}}
    if not t_rows:
        out["P*"] = {"status": "NOT_SELECTED_NO_COMPARATOR", "release": None, "ranking": [], "rejected": []}
        out["table"] = table
        return out
    tstar = t_rows[0]
    ref = tstar["auc_per_seed"]
    p_pool = [r for r in ids if I.nominee_capable(r)]
    p_rows, p_rej = choose(table, com, audited, p_pool, ref=ref)
    out["P*"] = {"status": "NOMINEE" if p_rows else "NO_ELIGIBLE_COMPETITIVE_NOMINEE",
                 "release": p_rows[0]["release"] if p_rows else None, "ranking": p_rows, "rejected": p_rej}
    if p_rows:
        win = p_rows[0]
        p, dec = I.parse_release(win["release"])
        disc = []
        if dec in ("D0", "D1"):
            disc.append("an ORIGINAL decoder won: the incumbent won; no calibration credit")
        fam = p.split("|")[1]
        if fam in ("LOCAL", "SEQ-12", "SEQ-21") or fam in ("W-LOCAL", "W-SEQ-12", "W-SEQ-21") or \
                fam in ("K-LOCAL", "K-SEQ-12", "K-SEQ-21"):
            disc.append("a LOCAL or sequential map won: no joint-design credit")
        if win["release"] == "U|JOINT|i8o64|l0.1":
            disc.append("same release ID as the lra P* (distinct study role and provenance recorded)")
        aliases = []
        if tables_fn is not None and D is not None:
            for k in I.SEEDS:
                hp = release_identity(k, win["release"], tables_fn, D)
                for r in t_pool:
                    if not r.startswith(I.U_ID) and release_identity(k, r, tables_fn, D) == hp:
                        aliases.append({"seed": k, "task_only_alias": r})
        if aliases:
            disc.append("exact deployed alias of a task-only map: no privacy-training credit")
        out["P*"].update({"disclosures": disc, "task_only_aliases": aliases,
                          "partition_privacy_trained_with_SEX": True,
                          "lra_source_P*": {"release": "U|JOINT|i8o64|l0.1", "study": "lra",
                                            "note": "distinct study role; same frozen map family may recur"}})
    out["table"] = table
    return out


def selection_rules():
    """SELECTION_RULES.json body (registered before SCIENCE_LOCK)."""
    return {"schema": "hcal-selection-rules-v1", "rows": "INNER_SELECTION (2,235 rows); utility via qpc.utility",
            "u0_gate": "qpc.utility.gate_record(...)['eligible'] each task each seed: acc >= U0 - 0.01; LL <= U0 + "
                       "0.01; Brier <= U0 + 0.005; gain >= 0.8 U0 gain; gain >= 0.03; exact decision preservation",
            "ucal_gate": "LL <= Ucal* + 0.01 and Brier <= Ucal* + 0.005 each task each seed (inclusive)",
            "ucal_star": "smallest mean over seeds of INNER income + occupation NLL among identity, H-GLOBAL-TEMP, "
                         "H-CLASS-TEMP; ties within 1e-12 -> identity, global, class; finite and decision-preserving; "
                         "one family across seeds; selected BEFORE any privacy selection (utility stage)",
            "audit_plan": "audit iff some non-diagnostic variant passes both gates every task and seed; plus "
                          "DIRECT-TASK, FINE-TASK, CLASS, C-TASK, JOINT lambda 0.1; others "
                          "PREDECLARED_UTILITY_INELIGIBLE",
            "recovery": "common attack-bank record of the release's partition (identical for every decoder variant); "
                        "continuous U: composed record; auc = attacker-seed 0-2 mean of the inner-AUC-selected reader",
            "T*": {"pool": "hcal.ids.task_only_candidate (DIRECT-TASK, FINE-TASK, CLASS, C-TASK with D0/D1/MEAN/"
                           "H-TOKEN32/H-GLOBAL-TEMP/H-CLASS-TEMP; SRC|U identity / H-GLOBAL-TEMP / H-CLASS-TEMP); "
                           "T-TOKEN32 excluded; no SEX-trained partition",
                   "order": "mean inner pair AUC, worst-seed normalised excess (both references), mean summed task "
                            "NLL, registered ID order (each value rounded to 12 decimals)"},
            "P*": {"pool": "hcal.ids.nominee_capable (privacy-trained partitions: 24 legacy, 24 W-, 5 K-; decoders "
                           "D0 (legacy), D1, H-TOKEN32, H-GLOBAL-TEMP, H-CLASS-TEMP; MEAN and T-TOKEN32 never)",
                   "guard": f"AUC_v(P*) <= AUC_v(T*) + {GUARD} for v in (v1, v2) on every seed (inclusive; margin form "
                            f"(AUC_v(T*) + {GUARD}) - AUC_v(P*) >= 0)",
                   "benefit": f"mean over seeds of AUC_pair(T*) - AUC_pair(P*) >= {BENEFIT} (inclusive; fixed before "
                              "fitting; never reduced)",
                   "order": "as T*", "absent": "NO_ELIGIBLE_COMPETITIVE_NOMINEE; no fallback nominee"},
            "means": "float64 (s0 + s1 + s2) / 3 over source seeds",
            "disclosures": ["original decoder wins -> incumbent won, no calibration credit",
                            "LOCAL / sequential map wins -> no joint-design credit",
                            "exact deployed alias of a task-only map -> no privacy-training credit",
                            "the lra P* and this study's P* carry distinct roles and provenance"]}


# ------------------------------------------------------------------ stage
def select_all(D, shard_spec=None):
    from hcal import run as R
    from hcal import stages as ST
    plan = json.loads((R.RUN / "audit_plan.json").read_text())
    util = {k: R.rec(f"util__s{k}")["releases"] for k in I.SEEDS}
    audited = plan["audited"]
    com = {k: {} for k in I.SEEDS}
    for k in I.SEEDS:
        for p in audited:
            com[k][p] = R.rec(ST.com_name(k, p))["recovery"]
        com[k]["SRC|U"] = R.rec(ST.com_name(k, "SRC|U"))["recovery"]
    ctl = json.loads((R.RUN / "controls.json").read_text())
    sel = select(util, com, audited, tables_fn=ST.release_arrays, D=D)
    if sel.get("ucal_star", {}).get("family") != plan["ucal_star"]["family"]:
        raise SystemExit("REFUSED: Ucal* differs from the one locked in the audit plan")
    apply_controls(sel, ctl)
    sel["audit_plan_sha256"] = hashlib.sha256((R.RUN / "audit_plan.json").read_bytes()).hexdigest()
    (R.RUN / "selection.json").write_text(json.dumps(R._finite(sel), indent=1, allow_nan=False) + "\n")
    R.event("selection", T=sel["T*"]["release"], P=sel["P*"]["release"], ucal=sel["ucal_star"]["family"])
