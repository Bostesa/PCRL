"""Pre-fit: UNIT_MANIFEST.csv (every planned unit, reuse status) and COVERAGE_AND_SUPPORT.csv (class/pair support per
role, every pair; support uses labels and roles only, identical across encoder seeds). Also EXACTNESS.json on the
actual frozen-head logits (no labels used). No fits."""
import csv, json, sys
from pathlib import Path
import numpy as np
WT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WT))
from odx import run as R
from odx.surfaces import exactness
import oar.study as S
PKG = WT / "results/combined_output_diagnosis_v1"
THR = {"attacker_fit": 100, "attacker_val": 30, "assessment": 100}
units, cov, exact = [], [], {}
for (ds, purpose, attr) in R.PAIRS:
    W = R.world(ds, purpose, attr)
    K = R.K_of(ds, attr)
    counts = {r: np.bincount(W["s"][W["idx"][r]], minlength=K).tolist() for r in ("defense_fit", "cert", *THR)}
    sup = [c for c in range(K) if all(counts[r][c] >= THR[r] for r in THR)]
    for c in range(K):
        cov.append({"dataset": ds, "purpose": purpose, "attribute": attr, "what": "class", "id": c,
                    **{f"n_{r}": counts[r][c] for r in counts}, "supported": c in sup,
                    "status": "ESTIMABLE" if c in sup else "NOT_ESTIMABLE", "seeds": "0;1;2 (support is seed-independent)",
                    "exposure": "17 Adult / 42 HMDA training-overlapping records removed from all scored roles"})
    for i in range(K):
        for j in range(i + 1, K):
            ok = i in sup and j in sup
            cov.append({"dataset": ds, "purpose": purpose, "attribute": attr, "what": "pair", "id": f"{i}-{j}",
                        "supported": ok, "status": "ESTIMABLE" if ok else "NOT_ESTIMABLE", "seeds": "0;1;2"})
    tcounts = {r: np.bincount(W["t"][W["idx"][r]]).tolist() for r in ("attacker_fit", "attacker_val", "assessment")}
    cov.append({"dataset": ds, "purpose": purpose, "attribute": attr, "what": "task_classes", "id": json.dumps(tcounts),
                "supported": True, "status": "INFO"})
    for k in S.SEEDS:
        strata = ["FH"] + (["RH"] if (ds, purpose, attr) in R.PRIMARY else [])
        units.append({"uid": R.uid_of(ds, k, purpose, attr, "REF", "ref"), "stage": "S2/S3", "kind": "reference",
                      "status": "REUSE_ALIAS" if (ds, purpose, attr) in R.PRIMARY else "PLANNED"})
        for st in strata:
            for s in R.SURFACE_ORDER:
                reuse = st == "FH" and (ds, purpose, attr) in R.PRIMARY and s in R.REUSE_OAR
                units.append({"uid": R.uid_of(ds, k, purpose, attr, st, s), "stage": "S2" if (ds, purpose, attr) in R.PRIMARY else "S3",
                              "kind": "attack", "status": "REUSE_ALIAS" if reuse else "PLANNED"})
            for b in ("iobank", "fullbank"):
                units.append({"uid": R.uid_of(ds, k, purpose, attr, st, b), "stage": "S2/S3", "kind": "bank (no fit)", "status": "PLANNED"})
for (ds, purpose, attr) in R.PAIRS:
    for k in S.SEEDS:
        F = np.load(S.BENCH / "inputs" / f"{ds}_s{k}_forward.npz")
        exact[f"{ds}__s{k}__{purpose}"] = exactness(F[R.purposes(ds)[purpose]["logits_key"]].astype(np.float64))
c = R.COALITION
for k in S.SEEDS:
    for con in c["contracts"]:
        u = f"adult__s{k}__PAIR_{c['purposes'][0]}+{c['purposes'][1]}__{c['attribute']}__FH__{con}"
        units += [{"uid": u, "stage": "S4", "kind": "attack", "status": "PLANNED"}, {"uid": u + "__bank", "stage": "S4", "kind": "bank (no fit)", "status": "PLANNED"}]
for ds in ("adult", "hmda"):
    units += [{"uid": f"{ds}__CTL__{n}", "stage": "controls", "kind": "null+planted (fit/val only)", "status": "PLANNED"} for n in ("full", "centred")]
units.append({"uid": "S5 conditional FARE", "stage": "S5", "kind": "conditional", "status": "CONDITIONAL (budget + screen)"})
for name, rows in (("UNIT_MANIFEST.csv", units), ("COVERAGE_AND_SUPPORT.csv", cov)):
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(PKG / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
(PKG / "EXACTNESS.json").write_text(json.dumps(exact, indent=1))
from collections import Counter
print(len(units), Counter((u["stage"], u["status"]) for u in units))
print({k: (v["K"], round(v["offset_sd"], 3), v.get("p1_saturated_exact_0_or_1"), round(v.get("sigmoid_margin_minus_softmax_p1_maxabs", 0), 18)) for k, v in exact.items() if k.endswith(("income_prediction", "underwriting"))})
