"""S5 certificates under the dated A1 guard repair (row-identity disjointness instead of feature-hash equality);
the original CertificateRefused records are kept in CELL.json. No fits."""
import copy, json, os, sys
from pathlib import Path
import numpy as np
WT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WT))
from oar import fare_official as FO
from odx import run as R
import oar.study as S
RUN = Path.home() / "PCRL_eval_cache_private/odx_v1/run"
cell = json.loads((RUN / "s5/CELL.json").read_text())
ds, pur, att = cell["cell"]
W = R.world(ds, pur, att)
df, cr = W["idx"]["defense_fit"], W["idx"]["cert"]
assert len(np.intersect1d(W["row_id"][df], W["row_id"][cr])) == 0 and len(np.intersect1d(W["unit"][df], W["unit"][cr])) == 0
cfg = {"delta": 0.05, "groups": None, "split_seed": 0, "val_fraction": 0.5, "eps_b_fraction": 0.1, "eps_s_fraction": 0.1}
out = {}
for k, v in cell["seeds"].items():
    if v["status"] != "OK":
        continue
    P = f"S5__{ds}__s{k}__{pur}__{att}"
    H = np.load(S.BENCH / "inputs" / f"{ds}_s{k}_forward.npz")[R.purposes(ds)[pur]["rep_key"]].astype(np.float64)
    nom = next(r for r in v["table"] if r["config"] == v["nominee"])
    src = nom["alias_of"] or nom["config"]
    res = {}
    for tag, uid in (("nominee", f"{P}__FAREFIT_c{src}"), ("zero_fairness", f"{P}__FAREFIT_Z")):
        m = FO.FareModel.load(RUN / "units" / uid / "model")
        m2 = copy.copy(m)
        m2.fit_row_hashes = np.array([], dtype=np.uint64)
        r = FO.certificate(m2, H[cr], W["s"][cr], cfg)
        res[tag] = {kk: r.get(kk) for kk in ("status", "reason", "bound", "bound_is_vacuous", "metric", "delta", "n", "n_pairs", "n_cells", "premises")}
    out[k] = {"original": v.get("certificate"), "amended_A1": res}
    print(k, {t: (x["status"], x["bound"]) for t, x in res.items()})
(RUN / "s5/CERTIFICATES_A1.json").write_text(json.dumps(out, indent=1, default=str))
