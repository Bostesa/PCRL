"""Amendment A3 (descriptive native-criterion diagnostic; no fits): FARE certificates for the selected F and F0 trees
under the record-identity guard used by the earlier study's amendment A1 (cert rows are disjoint records from the
fitting rows by role construction; the wrapper's feature-hash guard refused because different people share identical
permitted-input vectors). The original refusals stay in the outer records.
    ~/PCRL/.venv/bin/python results/pcrl_joint_complete_view_method_v1/report/fare_certificates_a3.py"""
import copy, json, os, sys
from pathlib import Path
import numpy as np
WT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WT))
os.environ.setdefault("OAR_RUN_UNITS", str(Path.home() / "PCRL_eval_cache_private/jcv_v1/fare_cache"))
from oar import fare_official as FO
from jcv import data as DA, run as R
D = DA.load()
tr, cr = D["idx"]["defense_train"], D["idx"]["cert"]
assert len(np.intersect1d(D["unit"][tr], D["unit"][cr])) == 0, "cert and fit records overlap"
same_vec = int(np.isin(FO.row_hashes(D["X"][cr].astype(np.float64)), FO.row_hashes(D["X"][tr].astype(np.float64))).sum())
cfg = {"delta": 0.05, "groups": None, "split_seed": 0, "val_fraction": 0.5, "eps_b_fraction": 0.1, "eps_s_fraction": 0.1}
SL = json.loads((R.PKG / "SELECTION_LOCK.json").read_text())
out = {"guard": "record identity (unit) disjointness, asserted", "cert_rows": int(len(cr)),
       "cert_rows_with_feature_vector_identical_to_some_fit_row": same_vec, "seeds": {}}
for k in ("0", "1", "2"):
    res = {}
    for arm in ("F", "F0"):
        for i, nm in enumerate(SL["seeds"][k]["arms"][arm]["units"]):
            uid = json.loads((R.U(nm) / "record.json").read_text())["fare_uid"]
            m = FO.FareModel.load(FO.units_root() / uid / "model")
            m2 = copy.copy(m)
            m2.fit_row_hashes = np.array([], dtype=np.uint64)
            r = FO.certificate(m2, D["X"][cr].astype(np.float64), D["sex"][cr], cfg)
            res[f"{arm}/purpose{i}"] = {kk: r.get(kk) for kk in ("status", "reason", "bound", "bound_is_vacuous", "metric", "delta", "n", "n_cells")}
    out["seeds"][k] = res
    print(k, {t: (x["status"], None if x["bound"] is None else round(x["bound"], 3)) for t, x in res.items()})
(R.PKG / "FARE_CERTIFICATES_A3.json").write_text(json.dumps(out, indent=1, default=str))
