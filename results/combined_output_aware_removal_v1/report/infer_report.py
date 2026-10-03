"""Post-run inference and tables for the output-aware removal study (saved predictions only; no fits)."""
import csv, json, os, sys, time
from pathlib import Path
import numpy as np

assert os.environ.get("OMP_NUM_THREADS") == "1"
WT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WT))
from oar import infer as I  # noqa: E402
from stored_model_eval.bench_infer import points_only  # noqa: E402
from oar.family import (FAMILY, FAMILY_SIZE, ALPHA_EACH, B_PRIMARY, SEED_PRIMARY, B_EXPLORATORY, SEED_EXPLORATORY,
                        LEVEL_EXPLORATORY, COMPETITIVE_CONJUNCTION)  # noqa: E402
from oar.study import RUN, SEEDS, RELEASE_SEEDS, unit_dir, unit_complete  # noqa: E402

PKG = WT / "results/combined_output_aware_removal_v1"
ROLES = json.loads((PKG / "ROLES_AND_SUPPORT.json").read_text())
t0, c0 = time.time(), time.process_time()
assert len(FAMILY) == FAMILY_SIZE == 12


def rj(p):
    return json.loads(Path(p).read_text())


prim_rows, expl_rows, worst_rows, frontier_rows, util_rows, da_rows, membership = [], [], [], [], [], {}, {}
for ds in ("adult", "hmda"):
    sup = ROLES[ds]["support"]["sensitive"]
    cg = I.CellGraph(sup["supported_classes"], sup["supported_pairs"])
    sel = {k: rj(RUN / "selection" / f"{ds}__s{k}.json") if (RUN / "selection" / f"{ds}__s{k}.json").exists() else None
           for k in SEEDS}
    G, members = {}, {}

    def grp(name, uids, recipe="NL"):
        uids = [u for u in uids]
        if not all(unit_complete(u) for u in uids):
            G[name] = None
            members[name] = {"members": uids, "complete": False}
            return None
        G[name] = cg.group(f"{ds}|{name}", [cg.recovery_of(u, recipe) for u in uids], uids)
        members[name] = {"members": uids, "complete": True, "recipe": recipe}
        return G[name]

    def grp_acc(name, uids):
        if not all(unit_complete(u) for u in uids):
            G[name] = None
            members[name] = {"members": uids, "complete": False}
            return None
        G[name] = cg.group(f"{ds}|{name}", [cg.accuracy_of(u) for u in uids], uids)
        members[name] = {"members": uids, "complete": True}
        return G[name]

    S_ = [f"{ds}__s{k}" for k in SEEDS]
    nom = {k: (sel[k]["nominee_unit_source"] if sel[k] else None) for k in SEEDS}
    for v in ("full", "prob", "hard"):
        grp(f"R(O_{v})", [f"{p}__O_{v}" for p in S_])
    grp("R(O_hard)|CC", [f"{p}__O_hard" for p in S_], "CC")
    grp("R(LO)", [f"{p}__REF" for p in S_], "LO")
    grp("R(const)", [f"{p}__REF" for p in S_], "const")
    for a in ("A", "B", "C"):
        grp(f"R({a},rep)", [f"{p}__{a}__rep" for p in S_])
        grp(f"R({a},rep+clean)", [f"{p}__{a}__rep+clean" for p in S_])
        grp_acc(f"Acc({a})", [f"{p}__U2__{a}" for p in S_])
    for a in ("A", "B"):
        grp(f"R({a},rep+head)", [f"{p}__{a}__rep+head" for p in S_])
        grp(f"R(O_head{a})", [f"{p}__O_head{a}" for p in S_])
    for v in ("rep", "rep+full", "rep+prob", "rep+hard"):
        uids = [f"{p}__D_rs{r}__{v}" for p in S_ for r in RELEASE_SEEDS]
        # mean over release seeds within encoder seed, then over encoder seeds (equal counts -> plain mean)
        grp(f"R(D*,{v})", uids)
    grp_acc("Acc(D*)", [f"{p}__U2__D_rs{r}" for p in S_ for r in RELEASE_SEEDS])
    if all(nom.values()):
        grp("R(F*,rep)", [f"{p}__Fc{nom[k]}__rep" for p, k in zip(S_, SEEDS)])
        grp("R(F*,rep)|CC", [f"{p}__Fc{nom[k]}__rep" for p, k in zip(S_, SEEDS)], "CC")
        grp_acc("Acc(F*)", [f"{p}__U2__Fc{nom[k]}" for p, k in zip(S_, SEEDS)])
    for tag in ("F", "FZ"):
        grp(f"R({tag if tag == 'FZ' else 'F*'},rep+clean)", [f"{p}__{tag}__rep+clean" for p in S_])
        grp(f"R({tag if tag == 'FZ' else 'F*'},rep+head)", [f"{p}__{tag}__rep+head" for p in S_])
        grp(f"R(O_head{tag})", [f"{p}__O_head{tag}" for p in S_])
    grp("R(FZ,rep)", [f"{p}__FZ__rep" for p in S_])
    grp("R(FZ,rep)|CC", [f"{p}__FZ__rep" for p in S_], "CC")
    grp_acc("Acc(FZ)", [f"{p}__U2__FZ" for p in S_])
    membership[ds] = members

    def term(expr):
        expr = expr.strip()
        return G.get(expr)

    # primary
    pids = {}
    for row in FAMILY:
        rid, cell, q, delta, margin, kind = row
        if cell != ds:
            continue
        a_, b_ = [x.strip() for x in delta.split(" - ")]
        ga, gb = term(a_), term(b_)
        pids[rid] = cg.diff(f"{ds}|{rid}", ga, gb) if (ga and gb) else None
    ids = [v for v in pids.values() if v]
    expl_ids = [v for v in G.values() if v]
    extra = {}
    for nm, (a_, b_) in {"R(O_full)-R(O_prob)": ("R(O_full)", "R(O_prob)"), "R(O_prob)-R(O_hard)": ("R(O_prob)", "R(O_hard)"),
                         "R(O_hard)-R(LO)": ("R(O_hard)", "R(LO)"), "R(O_full)-R(LO)": ("R(O_full)", "R(LO)"),
                         "R(D*,rep+full)-R(D*,rep+hard)": ("R(D*,rep+full)", "R(D*,rep+hard)"),
                         "R(D*,rep+full)-R(D*,rep)": ("R(D*,rep+full)", "R(D*,rep)"),
                         "R(A,rep)-R(F*,rep)": ("R(A,rep)", "R(F*,rep)"), "R(FZ,rep)-R(F*,rep)": ("R(FZ,rep)", "R(F*,rep)"),
                         "R(F*,rep+clean)-R(O_full)": ("R(F*,rep+clean)", "R(O_full)"),
                         "R(B,rep+clean)-R(B,rep+head)": ("R(B,rep+clean)", "R(B,rep+head)"),
                         "R(A,rep+head)-R(F*,rep+head)": ("R(A,rep+head)", "R(F*,rep+head)"),
                         "Acc(F*)-Acc(FZ)": ("Acc(F*)", "Acc(FZ)"), "Acc(D*)-Acc(A)": ("Acc(D*)", "Acc(A)"),
                         "Acc(B)-Acc(A)": ("Acc(B)", "Acc(A)"),
                         "R(F*,rep)|CC-R(F*,rep)": ("R(F*,rep)|CC", "R(F*,rep)"),
                         "R(O_hard)|CC-R(O_hard)": ("R(O_hard)|CC", "R(O_hard)")}.items():
        if G.get(a_) and G.get(b_):
            extra[nm] = cg.diff(f"{ds}|{nm}", G[a_], G[b_])
    expl_ids += list(extra.values())
    # primary bootstrap
    pts, reps, _ = I.bootstrap(cg, ids, B_PRIMARY, SEED_PRIMARY)
    for row in FAMILY:
        rid, cell, q, delta, margin, kind = row
        if cell != ds:
            continue
        sid = pids.get(rid)
        rec = {"id": rid, "cell": cell, "question": q, "delta": delta, "margin": margin, "kind": kind,
               "alpha_each": ALPHA_EACH, "B": B_PRIMARY, "seed": SEED_PRIMARY}
        if not sid:
            rec.update(point=None, lower=None, upper=None, decision="UNRESOLVED", reason="missing units")
        else:
            (lo, hi), n_ne = I.bound(reps[sid], ALPHA_EACH, 1 - ALPHA_EACH)
            rec.update(point=pts[sid], lower=lo, upper=hi, n_ne_replicates=n_ne,
                       decision=("PASS" if (lo is not None and lo > margin) else
                                 ("UNRESOLVED" if lo is None else "NOT_ESTABLISHED")))
        if rid.startswith(("P2", "P3", "P4", "P5", "P6")):
            rec["fare_nominee_admissible_all_seeds"] = all(sel[k] and sel[k]["admissible"] for k in SEEDS)
            rec["fare_nominees"] = {k: (sel[k]["nominee"] if sel[k] else None) for k in SEEDS}
        prim_rows.append(rec)
    # exploratory
    epts, ereps, _ = I.bootstrap(cg, list(dict.fromkeys(expl_ids)), B_EXPLORATORY, SEED_EXPLORATORY)
    a2 = (1 - LEVEL_EXPLORATORY) / 2
    names = {v: k for k, v in list(G.items()) + list(extra.items()) if v}
    for sid in dict.fromkeys(expl_ids):
        (lo, hi), _ = I.bound(ereps[sid], a2, 1 - a2)
        expl_rows.append({"cell": ds, "statistic": names[sid], "point": epts[sid], "lower90": lo, "upper90": hi,
                          "members": ";".join(members.get(names[sid], {}).get("members", []))})
    # worst-class / worst-pair (secondary; simultaneous over K components at level 0.90)
    wtargets = {k: v for k, v in members.items() if k.startswith("R(") and "|" not in k and v["complete"]}
    for name, m in wtargets.items():
        parts = [cg.worst_parts(u, m.get("recipe", "NL")) for u in m["members"]]
        keys = list(parts[0].keys())
        comp = {kk: cg.g.add(f"{ds}|{name}|{kk}", "mean", [p[kk] for p in parts]) for kk in keys}
        wp, wr, _ = I.bootstrap(cg, list(comp.values()), B_EXPLORATORY, SEED_EXPLORATORY)
        for kind in ("cls", "pair"):
            ks = [kk for kk in keys if kk.startswith(kind)]
            if not ks:
                continue
            K = len(ks)
            lcb, ucb = [], []
            for kk in ks:
                (lo, hi), _ = I.bound(wr[comp[kk]], (1 - LEVEL_EXPLORATORY) / (2 * K), 1 - (1 - LEVEL_EXPLORATORY) / (2 * K))
                lcb.append(lo), ucb.append(hi)
            pv = [wp[comp[kk]] for kk in ks]
            worst_rows.append({"cell": ds, "group": name, "statistic": "worst_class_auc" if kind == "cls" else "worst_pair_auc",
                               "K": K, "point": max(pv), "argmax": ks[int(np.argmax(pv))],
                               "lower": max(lcb) if None not in lcb else None, "upper": max(ucb) if None not in ucb else None})
    # FARE frontier (descriptive) and selection tables
    for k in SEEDS:
        if not sel[k]:
            continue
        for r in sel[k]["table"]:
            src = r["alias_of"] or r["config"]
            u, uu = f"{ds}__s{k}__Fc{src}__rep", f"{ds}__s{k}__U2__Fc{src}"
            sid = cg.recovery_of(u)
            aid = cg.accuracy_of(uu)
            pp = points_only(cg.g, len(cg.rows[0]), [sid, aid])
            cells = np.load(unit_dir(f"{ds}__s{k}__FAREFIT_c{src}") / "cells.npy")
            frontier_rows.append({"cell": ds, "seed": k, "config": r["config"], "alias_of": r["alias_of"],
                                  "n_cells": int(cells.max()) + 1, "val_u2_accuracy": r["val_u2_accuracy"],
                                  "val_nl_macro_auc": r["val_nl_macro_auc"], "admissible": r["admissible"],
                                  "selected": r["config"] == sel[k]["nominee"],
                                  "assessment_nl_macro_auc_descriptive": pp[sid], "assessment_u2_accuracy_descriptive": pp[aid]})
    # head (release) utility
    for k in SEEDS:
        for tag in ("A", "B", "F", "FZ"):
            u = f"{ds}__s{k}__HEAD__{tag}"
            if unit_complete(u):
                r = rj(unit_dir(u) / "record.json")
                util_rows.append({"cell": ds, "seed": k, "head_on": tag, "assessment_accuracy": r["assessment_accuracy"],
                                  "assessment_log_loss": r["assessment_log_loss"], "selected": json.dumps(r["selected"])})

for name, rows in (("PRIMARY_ENDPOINTS", prim_rows), ("EXPLORATORY", expl_rows), ("WORST_CLASS_PAIR", worst_rows),
                   ("FARE_FRONTIER", frontier_rows), ("HEAD_UTILITY", util_rows)):
    if rows:
        cols = list(dict.fromkeys(k for r in rows for k in r))
        with open(PKG / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
conj = {}
for ds, ids in COMPETITIVE_CONJUNCTION.items():
    rs = [r for r in prim_rows if r["id"] in ids]
    conj[ds] = {"rows": {r["id"]: r["decision"] for r in rs},
                "nominee_admissible_all_seeds": all(r.get("fare_nominee_admissible_all_seeds") for r in rs),
                "competitive": all(r["decision"] == "PASS" for r in rs) and all(r.get("fare_nominee_admissible_all_seeds") for r in rs)}
(PKG / "PRIMARY_SUMMARY.json").write_text(json.dumps({"family_size": FAMILY_SIZE, "n_rows": len(prim_rows),
                                                      "decisions": {r["id"]: r["decision"] for r in prim_rows},
                                                      "competitive_conjunction": conj, "membership": membership,
                                                      "wall_s": time.time() - t0, "cpu_s": time.process_time() - c0},
                                                     indent=1, default=str))
print(json.dumps({r["id"]: (round(r["point"], 4) if r["point"] is not None else None, r["decision"]) for r in prim_rows}, indent=1))
print(time.time() - t0, "s")
