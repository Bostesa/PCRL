"""Post-run inference for the output-leak diagnosis study (saved predictions only; no fits).
Writes PRIMARY_ENDPOINTS.csv (30 slots), S3_ENDPOINTS.csv (34), S4_ENDPOINTS.csv (6), OUTPUT_SURFACE_RESULTS.csv,
WORST_PAIR_SUMMARY.csv, BANK_SELECTIONS.csv."""
import csv, json, os, sys, time
from pathlib import Path
import numpy as np

assert os.environ.get("OMP_NUM_THREADS") == "1"
WT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WT))
from odx import family as F, run as R  # noqa: E402
from odx import infer as I  # noqa: E402
import oar.study as S  # noqa: E402

PKG = WT / "results/combined_output_diagnosis_v1"
OAR_UNITS = Path.home() / "PCRL_eval_cache_private/oar_v1/run/units"
COV = list(csv.DictReader(open(PKG / "COVERAGE_AND_SUPPORT.csv")))
t0, c0 = time.time(), time.process_time()


def supported(ds, purpose, attr):
    return [int(r["id"]) for r in COV if r["dataset"] == ds and r["purpose"] == purpose and r["attribute"] == attr
            and r["what"] == "class" and r["supported"] == "True"]


def rec(uid):
    return json.loads((I.ODX_UNITS / uid / "record.json").read_text())


def resolved_name(uid):
    r = rec(uid)
    return r["bank_selected"] if r.get("kind") == "bank" else uid


def surface_of(uid):
    return uid.split("__")[-1]


def decide(est, target, z, ident=False):
    if est is None:
        return {"decision": "UNAVAILABLE"}
    if ident:
        return {**est, "se": 0.0, "lower": est["point"], "upper": est["point"], "decision": "NOT_ESTABLISHED",
                "flag": "IDENTICAL_BY_SELECTION"}
    return {**est, "decision": "PASS" if est["lower"] > target else "NOT_ESTABLISHED"}


prim, s3, s4, surf_rows, worst_rows, bank_rows = [], [], [], [], [], []
for ds in ("adult", "hmda"):
    pairs = [p for p in R.PAIRS if p[0] == ds]
    W0 = R.world(*pairs[0])
    a = W0["idx"]["assessment"]
    D = I.DS(W0["row_id"][a], W0["unit"][a])
    ids, meta = {}, {}
    # ---- usefulness (frozen heads every purpose; U2__A refit probe on the primary purpose)
    for purpose, p in R.purposes(ds).items():
        W = R.world(ds, purpose, p["disallowed_attrs"][0])
        t = W["t"]
        maj = int(np.argmax(np.bincount(t[W["idx"]["attacker_fit"]], minlength=int(p["task_dim"]))))
        ta = t[a]
        cst = D.accuracy(f"{ds}|{purpose}|const", ta == maj)
        per = []
        disc = []
        for k in S.SEEDS:
            L = np.load(S.BENCH / "inputs" / f"{ds}_s{k}_forward.npz")[p["logits_key"]][a]
            yh = L.argmax(1)
            per.append(D.diff(f"{ds}|{purpose}|s{k}|gain", D.accuracy(f"{ds}|{purpose}|s{k}|frozen", yh == ta), cst))
            disc.append(int(((yh == ta) != (ta == maj)).sum()))
        ids[f"U-frozen|{purpose}"] = D.mean(f"{ds}|U-frozen|{purpose}", per)
        meta[f"U-frozen|{purpose}"] = {"discordant_rows_min": min(disc)}
        if purpose == S.CELLS[ds]["purpose"]:
            per = []
            disc = []
            for k in S.SEEDS:
                u = np.load(OAR_UNITS / f"{ds}__s{k}__U2__A/preds.npz")
                assert np.array_equal(u["assess_row_id"], W["row_id"][a])
                yh = u["U2_P"].argmax(1)
                per.append(D.diff(f"{ds}|U2|s{k}|gain", D.accuracy(f"{ds}|U2|s{k}", yh == ta), cst))
                disc.append(int(((yh == ta) != (ta == maj)).sum()))
            ids["U-refit"] = D.mean(f"{ds}|U-refit", per)
            meta["U-refit"] = {"discordant_rows_min": min(disc)}
    # ---- recovery per pair
    for (d_, purpose, attr) in pairs:
        cls = supported(ds, purpose, attr)
        strata = ["FH"] + (["RH"] if (ds, purpose, attr) in R.PRIMARY else [])
        for st in strata:
            per = {}
            for name in list(R.SURFACE_ORDER) + ["iobank", "fullbank"]:
                per[name] = [R.uid_of(ds, k, purpose, attr, st, name) for k in S.SEEDS]
            per["LO"] = per["const"] = [R.uid_of(ds, k, purpose, attr, "REF", "ref") for k in S.SEEDS]
            g = {}
            for name, us in per.items():
                recipe = name if name in ("LO", "const") else "NL"
                if len(cls) >= 2:
                    g[name] = [D.recovery(u, cls, recipe) for u in us]
                    ids[f"R|{purpose}|{attr}|{st}|{name}"] = D.mean(f"{ds}|R|{purpose}|{attr}|{st}|{name}", g[name])
            if len(cls) >= 2:
                fc = [D.diff(f"{ds}|FC|{purpose}|{attr}|{st}|s{k}", g["fullbank"][k], g["iobank"][k]) for k in S.SEEDS]
                ch = [D.diff(f"{ds}|CH|{purpose}|{attr}|{st}|s{k}", g["iobank"][k], g["hard"][k]) for k in S.SEEDS]
                ids[f"FC|{purpose}|{attr}|{st}"] = D.mean(f"{ds}|FC|{purpose}|{attr}|{st}", fc)
                ids[f"CH|{purpose}|{attr}|{st}"] = D.mean(f"{ds}|CH|{purpose}|{attr}|{st}", ch)
                ids[f"FB-H|{purpose}|{attr}|{st}"] = D.mean(f"{ds}|FBH|{purpose}|{attr}|{st}",
                                                            [D.diff(f"{ds}|FBH|{purpose}|{attr}|{st}|s{k}", g["fullbank"][k], g["hard"][k]) for k in S.SEEDS])
                sel_fb = [surface_of(resolved_name(u)) for u in per["fullbank"]]
                sel_io = [surface_of(resolved_name(u)) for u in per["iobank"]]
                meta[f"FC|{purpose}|{attr}|{st}"] = {"fullbank_selected": sel_fb, "iobank_selected": sel_io,
                                                     "identical_by_selection": all(resolved_name(f) == resolved_name(i) for f, i in zip(per["fullbank"], per["iobank"])),
                                                     "offset_attributable": all(s_ in R.OFFSET_USING for s_ in sel_fb)}
                for k in S.SEEDS:
                    bank_rows.append({"dataset": ds, "purpose": purpose, "attribute": attr, "stratum": st, "seed": k,
                                      "fullbank_selected": sel_fb[k], "iobank_selected": sel_io[k],
                                      "fullbank_candidates_val_ll": json.dumps(rec(per["fullbank"][k])["candidates_attacker_val_log_loss"])})
            # pairs (primary attribute pairs of the primary cells use declared pair slots)
            if (ds, purpose, attr) in R.PRIMARY and st == "FH":
                K = R.K_of(ds, attr)
                for i in range(K):
                    for j in range(i + 1, K):
                        if i in cls and j in cls:
                            pk = {nm: [D.pair(u, i, j) for u in per[nm]] for nm in ("fullbank", "iobank", "hard")}
                            ids[f"FCpair|{i}-{j}"] = D.mean(f"{ds}|FCpair{i}-{j}", [D.diff(f"{ds}|FCp{i}-{j}|s{k}", pk["fullbank"][k], pk["iobank"][k]) for k in S.SEEDS])
                            ids[f"CHpair|{i}-{j}"] = D.mean(f"{ds}|CHpair{i}-{j}", [D.diff(f"{ds}|CHp{i}-{j}|s{k}", pk["iobank"][k], pk["hard"][k]) for k in S.SEEDS])
                            for nm in ("fullbank", "iobank", "hard"):
                                ids[f"PAIR|{nm}|{i}-{j}"] = D.mean(f"{ds}|PAIR|{nm}|{i}-{j}", pk[nm])
    # ---- S4 coalition (Adult)
    if ds == R.COALITION["dataset"]:
        pa, pb = R.COALITION["purposes"]
        attr = R.COALITION["attribute"]
        cls = supported(ds, pa, attr)
        for c in R.COALITION["contracts"]:
            pair_u = [f"{ds}__s{k}__PAIR_{pa}+{pb}__{attr}__FH__{c}__bank" for k in S.SEEDS]
            sname = {"full": "fullbank", "centred": "iobank"}.get(c, c)
            if all((I.ODX_UNITS / u / "record.json").exists() for u in pair_u):
                gp = [D.recovery(u, cls) for u in pair_u]
                ids[f"S4|{c}|pair"] = D.mean(f"S4|{c}|pair", gp)
                for who in (pa, pb):
                    gs = [D.recovery(R.uid_of(ds, k, who, attr, "FH", sname), cls) for k in S.SEEDS]
                    ids[f"S4|{c}|{who}"] = D.mean(f"S4|{c}|{who}", gs)
                    ids[f"S4|{c}|pair-minus-{who}"] = D.mean(f"S4|{c}|pair-minus-{who}", [D.diff(f"S4|{c}|pm|{who}|s{k}", gp[k], gs[k]) for k in S.SEEDS])
                meta[f"S4|{c}"] = {"pair_bank_selected": [surface_of(resolved_name(u)) if "PAIR_" not in resolved_name(u) else "pair" for u in pair_u],
                                   "pair_bank_selected_uid": [resolved_name(u) for u in pair_u]}
    # ---- estimate everything (one paired bootstrap per dataset)
    est = I.estimate(D, list(ids.values()), F.B_SE, F.SEED_SE, 1.0)          # z applied per family below
    EST = {k: est[v] for k, v in ids.items()}
    json.dump({"ids": list(EST), "meta": meta}, open(Path.home() / f"PCRL_eval_cache_private/odx_v1/run/infer_{ds}_meta.json", "w"), default=str)

    def with_z(e, z):
        return None if e is None else {**e, "lower": e["point"] - z * e["se"], "upper": e["point"] + z * e["se"], "z": z}
    # primary
    zp = F.Z_PRIMARY
    pc = next(p for p in R.PRIMARY if p[0] == ds)
    for e in F.PRIMARY:
        if e["dataset"] != ds:
            continue
        row = {"id": e["id"], "block": e["block"], "dataset": ds, "stat": e["stat"], "target": e["target"], "z": zp,
               "B": F.B_SE, "seed": F.SEED_SE}
        if e["id"] in F.EXPECTED_NOT_ESTIMABLE:
            row.update(decision="NOT_ESTIMABLE", reason="pair involves an unsupported race group (declared before fits)")
        else:
            if e["block"] == "usefulness":
                key = f"U-frozen|{pc[1]}" if e["id"].startswith("U-frozen") else "U-refit"
                ident = False
            elif e["block"] == "macro":
                con = e["id"].split("-")[0]
                key, ident = f"{con}|{pc[1]}|{pc[2]}|FH", (con == "FC" and meta[f"FC|{pc[1]}|{pc[2]}|FH"]["identical_by_selection"])
            else:
                con = e["id"].split("-")[0]
                i, j = e["pair"]
                key, ident = f"{con}pair|{i}-{j}", (con == "FC" and meta[f"FC|{pc[1]}|{pc[2]}|FH"]["identical_by_selection"])
            r_ = decide(with_z(EST.get(key), zp), e["target"], zp, ident)
            row.update({k: v for k, v in r_.items() if k not in ("B", "seed")})
            m = meta.get(key) or meta.get(f"FC|{pc[1]}|{pc[2]}|FH") if "FC" in e["id"] else meta.get(key)
            if e["block"] == "usefulness":
                row["flag_normal_approx_weak"] = meta[key]["discordant_rows_min"] < 30
            if e["id"].startswith("FC"):
                row["offset_attributable"] = meta[f"FC|{pc[1]}|{pc[2]}|FH"]["offset_attributable"]
                row["fullbank_selected_by_seed"] = json.dumps(meta[f"FC|{pc[1]}|{pc[2]}|FH"]["fullbank_selected"])
            if e["id"] in F.DECLARED_ALIASES:
                row["declared_alias_of"] = F.DECLARED_ALIASES[e["id"]]
        prim.append(row)
    # S3
    z3 = F.z_two_sided(F.S3_SIZE)
    for (d_, purpose, attr) in pairs:
        for con in ("FC", "CH"):
            key = f"{con}|{purpose}|{attr}|FH"
            ident = con == "FC" and meta.get(key, {}).get("identical_by_selection", False)
            r_ = decide(with_z(EST.get(key), z3), 0.02, z3, ident) if key in EST else {"decision": "NOT_ESTIMABLE"}
            s3.append({"id": f"S3-{con}-{ds}-{purpose}-{attr}", "z": z3, **r_, **({"offset_attributable": meta[key]["offset_attributable"],
                       "fullbank_selected_by_seed": json.dumps(meta[key]["fullbank_selected"])} if con == "FC" and key in meta else {})})
    for purpose in R.purposes(ds):
        key = f"U-frozen|{purpose}"
        s3.append({"id": f"S3-U-{ds}-{purpose}", "z": z3, **decide(with_z(EST[key], z3), 0.01, z3),
                   "flag_normal_approx_weak": meta[key]["discordant_rows_min"] < 30})
    # S4
    if ds == R.COALITION["dataset"]:
        z4 = F.z_two_sided(F.S4_SIZE)
        for c in R.COALITION["contracts"]:
            for who in R.COALITION["purposes"]:
                key = f"S4|{c}|pair-minus-{who}"
                ident = False
                if f"S4|{c}" in meta:
                    ident = all(u == R.uid_of(ds, k, who, R.COALITION["attribute"], "FH", {"full": "fullbank", "centred": "iobank"}.get(c, c))
                                or resolved_name(R.uid_of(ds, k, who, R.COALITION["attribute"], "FH", {"full": "fullbank", "centred": "iobank"}.get(c, c))) == u
                                for k, u in enumerate(meta[f"S4|{c}"]["pair_bank_selected_uid"]))
                r_ = decide(with_z(EST.get(key), z4), 0.02, z4, ident) if key in EST else {"decision": "UNAVAILABLE"}
                s4.append({"id": f"S4-{c}-pair-minus-{who}", "z": z4, **r_,
                           "pair_bank_selected_by_seed": json.dumps(meta.get(f"S4|{c}", {}).get("pair_bank_selected_uid"))})
    # surface table and worst pairs
    for k_, e in EST.items():
        if k_.startswith(("R|", "PAIR|", "FB-H|")):
            surf_rows.append({"dataset": ds, "key": k_, "point": e["point"], "se": e["se"],
                              "lower90": e["point"] - 1.6449 * e["se"], "upper90": e["point"] + 1.6449 * e["se"]})
    for nm in ("fullbank", "iobank", "hard"):
        ps = {k_: e for k_, e in EST.items() if k_.startswith(f"PAIR|{nm}|")}
        if ps:
            kk = max(ps, key=lambda x: ps[x]["point"])
            worst_rows.append({"dataset": ds, "surface": nm, "worst_supported_pair": kk.split("|")[-1], "point": ps[kk]["point"],
                               "se": ps[kk]["se"], "n_supported_pairs": len(ps)})

for name, rows in (("PRIMARY_ENDPOINTS", prim), ("S3_ENDPOINTS", s3), ("S4_ENDPOINTS", s4),
                   ("OUTPUT_SURFACE_RESULTS", surf_rows), ("WORST_PAIR_SUMMARY", worst_rows), ("BANK_SELECTIONS", bank_rows)):
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(PKG / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
assert len(prim) == F.PRIMARY_SIZE, len(prim)
assert len(s3) == F.S3_SIZE, len(s3)
from collections import Counter
print("primary", Counter(r["decision"] for r in prim), "S3", Counter(r["decision"] for r in s3), "S4", Counter(r["decision"] for r in s4))
for r in prim:
    if r["decision"] != "NOT_ESTIMABLE":
        print(r["id"], round(r.get("point", float("nan")), 4), round(r.get("lower", float("nan")), 4), r["decision"], r.get("flag", ""), r.get("offset_attributable", ""))
print("wall", time.time() - t0, "cpu", time.process_time() - c0)
