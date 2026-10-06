"""Registered inner-only selection (PROTOCOL.md section 11; frozen in AUDIT_AND_SELECTION_LOCK). No assessment value is
read anywhere. Inputs: inner__<unit> records (qpc.audit, attackers fitted on AUDIT_FIT and selected/scored on
INNER_SELECTION; utility on INNER_SELECTION via qpc.utility) for every candidate of qpc.run.scored_ids().

Required record fields (a missing/nonfinite field is a TECHNICAL FAILURE of that candidate, never silently skipped):
  recovery.auc.{v1,v2,pair}   inner SEX AUC of the AUC-selected attacker (mean over attacker seeds 0-2); for continuous
                              sources this is the COMPOSED bank (own slate + every fitted code's readers), assembled
                              in the inner_src stage before this selection
  utility.{income,occupation} qpc.utility.release_inner_utility on INNER_SELECTION
  preserved.{1,2}             decision preservation against the release's own teacher (all rows)
  token_states                alpha1 + alpha2 of the deployed policy (None for continuous releases -> +inf)
Eligibility per seed = qpc.utility.gate_record(code, U anchor of that seed, preserved)["eligible"]; a configuration is
ELIGIBLE iff eligible on every seed.
Ordering (C_rate, C_global, T*, J*, P*): mean inner pair AUC, mean summed true-label task log loss, mean actual total
states, configuration ID.
Fallback for an absent nominee (DESCRIPTIVE_ONLY): (0 if eligible else worst-seed normalized excess, guard shortfall,
ordering key).
Statuses: NOMINEE; NO_ELIGIBLE_NOMINEE (nominee roles) / NO_ELIGIBLE_COMPARATOR (comparator roles) when candidates were
computed and none is eligible; INVALID_COMPARATOR / INVALID_NOMINEE when any candidate of the role is a technical
failure, or when an otherwise eligible nominee is blocked only by a missing guard comparator.
"""
from __future__ import annotations

import csv
import json
import math

from qpc import family as FAM
from qpc import run as R
from qpc import utility as UT

BUFFER = 0.005
SEEDS = R.SEEDS
NONJOINT = ("DIRECT-TASK", "FINE-TASK", "LOCAL", "SEQ-12", "SEQ-21")
PRIVACY = ("LOCAL", "SEQ-12", "SEQ-21", "JOINT")
UNTRAINED = ("DIRECT-TASK", "FINE-TASK", "CLASS")
VIEWS = ("v1", "v2", "pair")
TASKS = ("income", "occupation")


def family(cid):
    if cid.startswith("SRC|"):
        return "SRC"
    if cid.startswith("REF|"):
        return "REF"
    return cid.split("|")[1]


def rate(cid):
    if family(cid) in ("SRC", "REF", "CLASS"):
        return None
    p = R.parse_id(cid)
    return (p["m1"], p["m2"])


def _finite(x):
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def candidate_rows(ids):
    """One row per configuration; technical failures recorded per row."""
    anchors, rows = {}, {}
    for k in SEEDS:
        try:
            anchors[k] = R.rec(f"inner__{R.unit_for(k, 'SRC|U')}")["utility"]
        except Exception as e:                                              # noqa: BLE001
            anchors[k] = None
    for cid in ids:
        seeds, fail = {}, []
        for k in SEEDS:
            n = f"inner__{R.unit_for(k, cid)}"
            if not R.done(n):
                fail.append(f"{n}: missing or not hash-complete")
                continue
            r = R.rec(n)
            try:
                a = {w: float(r["recovery"]["auc"][w]) for w in VIEWS}
                u = r["utility"]
                pres = {int(i): bool(v) for i, v in r["preserved"].items()}
                assert set(pres) == {1, 2}
                assert all(_finite(a[w]) for w in VIEWS)
                assert all(_finite(u[t][x]) for t in TASKS for x in ("acc", "logloss", "brier"))
                assert anchors[k] is not None
                gr = UT.gate_record(u, anchors[k], pres)
            except Exception as e:                                          # noqa: BLE001
                fail.append(f"{n}: {type(e).__name__}: {e}")
                continue
            ts = r.get("token_states")
            seeds[k] = {"unit": R.unit_for(k, cid), "auc": a, "utility": {t: {x: u[t][x] for x in
                        ("acc", "logloss", "brier", "const_acc", "gain") if x in u[t]} for t in TASKS},
                        "gate": {"eligible": bool(gr["eligible"]), "headroom": bool(gr["headroom"]),
                                 "norm_excess": max(float(gr[t]["norm_excess"]) for t in TASKS),
                                 **{t: {x: gr[t][x] for x in ("ll_excess", "brier_excess")} for t in TASKS}},
                        "token_states": float(ts) if ts is not None else math.inf,
                        "composed_winner": (r.get("composed") or {}).get("winner")}
        ok = not fail and len(seeds) == len(SEEDS)
        row = {"config": cid, "family": family(cid), "rate": rate(cid), "technical_failure": fail, "ok": ok,
               "seeds": seeds}
        if ok:
            mean = lambda f: sum(f(s) for s in seeds.values()) / len(SEEDS)    # noqa: E731
            row.update({"mean_pair": mean(lambda s: s["auc"]["pair"]), "mean_v1": mean(lambda s: s["auc"]["v1"]),
                        "mean_v2": mean(lambda s: s["auc"]["v2"]),
                        "mean_sum_logloss": mean(lambda s: s["utility"]["income"]["logloss"] +
                                                 s["utility"]["occupation"]["logloss"]),
                        "mean_states": mean(lambda s: s["token_states"]),
                        "eligible": all(s["gate"]["eligible"] for s in seeds.values()),
                        "worst_norm_excess": max(s["gate"]["norm_excess"] for s in seeds.values())})
        rows[cid] = row
    return rows


def key(r):
    return (round(r["mean_pair"], 12), round(r["mean_sum_logloss"], 12), r["mean_states"], r["config"])


def guard_excess(r, guards):
    """max over guards, seeds and recipients of local AUC - (guard AUC + BUFFER); <= 0 means the guards hold."""
    xs = [r["seeds"][k]["auc"][w] - (g["seeds"][k]["auc"][w] + BUFFER) for g in guards.values() for k in SEEDS
          for w in ("v1", "v2")]
    return max(xs) if xs else -math.inf


def pick(rows, guards=None, nominee=True):
    """rows: candidate rows of one role. guards: {name: row} (all must exist)."""
    none = "NO_ELIGIBLE_NOMINEE" if nominee else "NO_ELIGIBLE_COMPARATOR"
    bad = "INVALID_NOMINEE" if nominee else "INVALID_COMPARATOR"
    if not rows:
        return {"status": bad, "config": None, "reason": "empty candidate set"}
    tf = [r["config"] for r in rows if not r["ok"]]
    if tf:
        return {"status": bad, "config": None, "reason": "technical failure of candidates", "failed": tf}
    guards = guards or {}
    ev = []
    for r in rows:
        ge = guard_excess(r, guards)
        e = {"config": r["config"], "eligible": r["eligible"], "guard_excess": ge, "guard_ok": ge <= 0,
             "nominable": r["eligible"] and ge <= 0,
             "fallback_key": (0.0 if r["eligible"] else round(r["worst_norm_excess"], 12),
                              round(max(0.0, ge), 12)) + key(r)}
        ev.append(e)
    el = [rows_by(rows)[e["config"]] for e in ev if e["nominable"]]
    if el:
        b = min(el, key=key)
        return {"status": "NOMINEE", "config": b["config"], "evaluated": ev}
    fb = min(ev, key=lambda e: e["fallback_key"])
    return {"status": none, "config": None, "descriptive_config": fb["config"], "evaluated": ev}


def rows_by(rows):
    return {r["config"]: r for r in rows}


def select_all(D=None, shard_spec=None):
    ids = R.scored_ids()
    rows = candidate_rows(ids)
    allr = list(rows.values())
    st = {}
    sa = [r for r in allr if r["family"] == "DIRECT-TASK"]
    # Q*: smallest worst-seed normalized excess among eligible Stage A codes, then fewer states, then id
    if any(not r["ok"] for r in sa):
        st["Q*"] = {"status": "INVALID_NOMINEE", "config": None, "failed": [r["config"] for r in sa if not r["ok"]]}
    else:
        el = [r for r in sa if r["eligible"]]
        st["Q*"] = ({"status": "NOMINEE", "config": min(el, key=lambda r: (round(r["worst_norm_excess"], 12),
                                                                              r["mean_states"], r["config"]))["config"]}
                    if el else {"status": "NO_ELIGIBLE_NOMINEE", "config": None,
                                "descriptive_config": min(sa, key=lambda r: (round(r["worst_norm_excess"], 12),
                                                                            r["mean_states"], r["config"]))["config"]})
    nonjoint = [r for r in allr if r["family"] in NONJOINT]
    rates = sorted({r["rate"] for r in allr if r["family"] == "JOINT"})
    C_rate = {f"i{a}o{b}": pick([r for r in nonjoint if r["rate"] == (a, b)], nominee=False) for a, b in rates}
    st["C_global"] = pick(nonjoint + [r for r in allr if r["family"] in ("CLASS", "SRC", "REF")], nominee=False)
    st["T*"] = pick([r for r in allr if r["family"] in UNTRAINED + ("SRC",) or r["config"] == "REF|F0"],
                    nominee=False)
    # J*: each JOINT configuration is guarded by its own C_rate and by C_global
    jr = [r for r in allr if r["family"] == "JOINT"]
    blocked = []
    jev = []
    for r in jr:
        cr = C_rate[f"i{r['rate'][0]}o{r['rate'][1]}"]
        g, miss = {}, []
        for name, P in (("C_rate", cr), ("C_global", st["C_global"])):
            if P["status"] == "NOMINEE":
                g[name] = rows[P["config"]]
            else:
                miss.append(name)
        if not r["ok"]:
            jev.append({"config": r["config"], "technical_failure": r["technical_failure"]})
            continue
        e = pick([r], guards=g)
        e1 = (e.get("evaluated") or [{}])[0]
        e1 = {**e1, "missing_guards": miss, "nominable": bool(e1.get("nominable")) and not miss}
        if r["eligible"] and e1.get("guard_ok") and miss:
            blocked.append(r["config"])
        jev.append(e1)
    if any("technical_failure" in e for e in jev) or not jr:
        st["J*"] = {"status": "INVALID_NOMINEE", "config": None, "evaluated": jev}
    else:
        el = [rows[e["config"]] for e in jev if e["nominable"]]
        if el:
            st["J*"] = {"status": "NOMINEE", "config": min(el, key=key)["config"], "evaluated": jev}
        elif blocked:
            st["J*"] = {"status": "INVALID_NOMINEE", "config": None, "reason": "eligible JOINT blocked only by a "
                        "missing guard comparator (guards are never dropped)", "blocked": blocked, "evaluated": jev}
        else:
            fb = min(jev, key=lambda e: e["fallback_key"])
            st["J*"] = {"status": "NO_ELIGIBLE_NOMINEE", "config": None, "descriptive_config": fb["config"],
                        "evaluated": jev}
    jcell = st["J*"].get("config") or st["J*"].get("descriptive_config")
    st["C_rate"] = (dict(C_rate[f"i{rate(jcell)[0]}o{rate(jcell)[1]}"], cell=f"i{rate(jcell)[0]}o{rate(jcell)[1]}")
                    if jcell else {"status": "INVALID_COMPARATOR", "config": None, "reason": "no J* cell"})
    # P*: privacy-trained, guarded by T*
    if st["T*"]["status"] == "NOMINEE":
        st["P*"] = pick([r for r in allr if r["family"] in PRIVACY], guards={"T*": rows[st["T*"]["config"]]})
        if st["P*"]["status"] == "NOMINEE":
            st["P*"]["winning_family"] = family(st["P*"]["config"])
    else:
        st["P*"] = {"status": "INVALID_NOMINEE", "config": None, "reason": "T* missing (guard never dropped)"}
    for x in ("C_global", "T*", "C_rate"):                         # comparator role states for the truth table
        if st[x]["status"] == "NO_ELIGIBLE_NOMINEE":
            st[x]["status"] = "NO_ELIGIBLE_COMPARATOR"
    claims = {c: {"nominee": FAM.role_state(st[n]), "comparator": FAM.role_state(st[m])}
              for c, (n, m) in FAM.CLAIMS.items()}
    out = {"schema": "qpc-selection-v1", "rule": "PROTOCOL.md section 11 (AUDIT_AND_SELECTION_LOCK)",
           "candidates": ids, "statuses": st, "claim_role_states": claims,
           "resolved": {x: (s.get("config") or s.get("descriptive_config")) for x, s in st.items()},
           "technical_failures": {c: r["technical_failure"] for c, r in rows.items() if r["technical_failure"]},
           "U_eligible": rows.get("SRC|U", {}).get("eligible"),
           "deployable_eligible_privacy_code": st["P*"].get("config") or (st["J*"].get("config")),
           "rows": rows}
    (R.RUN / "selection.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    pub = {k: v for k, v in out.items() if k != "rows"}
    pub["rows"] = {c: {x: r.get(x) for x in ("family", "rate", "ok", "eligible", "mean_pair", "mean_v1", "mean_v2",
                                            "mean_sum_logloss", "mean_states", "worst_norm_excess")}
                   for c, r in rows.items()}
    (R.PKG / "SELECTION.json").write_text(json.dumps(pub, indent=1, default=float) + "\n")
    with open(R.PKG / "INNER_SELECTION_TABLE.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["config", "family", "seed", "auc_v1", "auc_v2", "auc_pair", "ll_income", "ll_occupation",
                    "brier_income", "brier_occupation", "ll_excess_occupation", "brier_excess_occupation",
                    "norm_excess", "seed_eligible", "token_states", "composed_winner_pair"])
        for c, r in rows.items():
            for k, s in sorted(r["seeds"].items()):
                w.writerow([c, r["family"], k] + [f"{s['auc'][v]:.6f}" for v in VIEWS] +
                           [f"{s['utility'][t]['logloss']:.6f}" for t in TASKS] +
                           [f"{s['utility'][t]['brier']:.6f}" for t in TASKS] +
                           [f"{s['gate']['occupation']['ll_excess']:.6f}",
                            f"{s['gate']['occupation']['brier_excess']:.6f}", f"{s['gate']['norm_excess']:.6f}",
                            s["gate"]["eligible"], s["token_states"],
                            (s["composed_winner"] or {}).get("pair", "") if s["composed_winner"] else ""])
    R.event("selection written", statuses={x: [s["status"], s.get("config") or s.get("descriptive_config")]
                                           for x, s in st.items()})
    return out
