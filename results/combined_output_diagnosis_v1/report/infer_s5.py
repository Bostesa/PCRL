"""Stage 5 inference (saved predictions only): S5 family (4 endpoints, z = 2.497705), FARE frontier, certificates."""
import csv, json, os, sys
from pathlib import Path
import numpy as np
assert os.environ.get("OMP_NUM_THREADS") == "1"
WT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WT))
from odx import family as F, infer as I, run as R
PKG = WT / "results/combined_output_diagnosis_v1"
RUN = Path.home() / "PCRL_eval_cache_private/odx_v1/run"
cell = json.loads((RUN / "s5/CELL.json").read_text())
ds, pur, att = cell["cell"]
cls = [int(r["id"]) for r in csv.DictReader(open(PKG / "COVERAGE_AND_SUPPORT.csv")) if r["dataset"] == ds and r["purpose"] == pur
       and r["attribute"] == att and r["what"] == "class" and r["supported"] == "True"]
W = R.world(ds, pur, att)
a, f = W["idx"]["assessment"], W["idx"]["attacker_fit"]
t = W["t"]
maj = int(np.argmax(np.bincount(t[f])))
D = I.DS(W["row_id"][a], W["unit"][a])
z = F.z_two_sided(F.S5_SIZE)
seeds = {int(k): v for k, v in cell["seeds"].items()}
feasible_all = all(v["status"] == "OK" for v in seeds.values())
rows, front = [], []
for k, v in seeds.items():
    for r in v["table"]:
        front.append({"seed": k, **r, "selected": r["config"] == v["nominee"], "untreated_val_u2_accuracy": v["untreated_val_u2_accuracy"],
                      "constant_val_accuracy": v["constant_val_accuracy"]})
ids = {}
if feasible_all:
    P = lambda k: f"S5__{ds}__s{k}__{pur}__{att}"  # noqa: E731
    cst = D.accuracy("const", t[a] == maj)
    def accid(u):
        p = np.load(I.ODX_UNITS / u / "preds.npz")
        assert np.array_equal(p["assess_row_id"], W["row_id"][a])
        return D.accuracy(u, p["U2_P"].argmax(1) == p["y_t"])
    rec = {tag: [D.recovery(f"{P(k)}__{tag}__rep+head", cls) for k in seeds] for tag in ("A", "B", "F", "FZ")}
    accA = [accid(f"{ds}__s{k}__{pur}__U2__A") for k in seeds]
    accF = [accid(f"{P(k)}__U2__F") for k in seeds]
    ids["S5-LEACE-minus-FARE-rep+head"] = D.mean("e1", [D.diff(f"e1{k}", rec["B"][i], rec["F"][i]) for i, k in enumerate(seeds)])
    ids["S5-FZ-minus-FARE-rep+head"] = D.mean("e2", [D.diff(f"e2{k}", rec["FZ"][i], rec["F"][i]) for i, k in enumerate(seeds)])
    ids["S5-Acc(FARE)-Acc(A)"] = D.mean("e3", [D.diff(f"e3{k}", accF[i], accA[i]) for i, k in enumerate(seeds)])
    # linear retention contrast A_F - 0.8 A_A - 0.2 const  (= A_F - const - 0.8 (A_A - const))
    def lin_stat(cF, cA, cK):
        cF, cA, cK = (np.asarray(v, float) for v in (cF, cA, cK))
        def fn(WT):
            n = WT.sum(0)
            return ((cF - 0.8 * cA - 0.2 * cK) @ WT) / n
        return fn
    lids = []
    for i, k in enumerate(seeds):
        pF = np.load(I.ODX_UNITS / f"{P(k)}__U2__F" / "preds.npz"); pA = np.load(I.ODX_UNITS / f"{ds}__s{k}__{pur}__U2__A" / "preds.npz")
        lids.append(D.g.base_once(f"lin{k}", f"lin{k}", lin_stat(pF["U2_P"].argmax(1) == pF["y_t"], pA["U2_P"].argmax(1) == pA["y_t"], t[a] == maj)))
    ids["S5-retained-gain-share"] = D.mean("e4", lids)
    for tag in ("A", "B", "F", "FZ"):
        ids[f"R({tag},rep+head)"] = D.mean(f"R{tag}", rec[tag])
    ids["Acc(A)"], ids["Acc(F)"] = D.mean("aA", accA), D.mean("aF", accF)
    ids["const"] = cst
    est = I.estimate(D, list(ids.values()), F.B_SE, F.SEED_SE, z)
    E = {k: est[v] for k, v in ids.items()}
    targets = {"S5-LEACE-minus-FARE-rep+head": 0.02, "S5-FZ-minus-FARE-rep+head": 0.02, "S5-Acc(FARE)-Acc(A)": -0.01, "S5-retained-gain-share": 0.0}
    for e in F.S5:
        r = E[e]
        rows.append({"id": e, "target": targets[e], "z": z, **r, "decision": "PASS" if r["lower"] > targets[e] else "NOT_ESTABLISHED"})
    desc = {k: E[k] for k in E if k not in targets}
else:
    for e in F.S5:
        rows.append({"id": e, "decision": "NO_FEASIBLE_NOMINEE", "reason": "at least one encoder seed had no feasible FARE nominee; S5 slots cannot pass"})
    desc = {}
for name, rr in (("S5_ENDPOINTS", rows), ("FARE_USEFUL_TASK_FRONTIER", front)):
    cols = list(dict.fromkeys(k for r in rr for k in r))
    with open(PKG / f"{name}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rr)
json.dump({"cell": cell["cell"], "supported_classes": cls, "feasible_all_seeds": feasible_all, "descriptive": desc,
           "nominees": {k: v["nominee"] for k, v in seeds.items()},
           "certificates": {k: v.get("certificate") for k, v in seeds.items()},
           "tree_own_task_accuracy_assessment": {k: v.get("tree_own_task_accuracy_assessment") for k, v in seeds.items()}},
          open(PKG / "S5_SUMMARY.json", "w"), indent=1, default=str)
for r in rows:
    print(r["id"], r.get("point"), r.get("lower"), r["decision"])
print({k: round(v["point"], 4) for k, v in desc.items()})
