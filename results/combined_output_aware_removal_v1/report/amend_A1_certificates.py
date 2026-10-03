"""Amendment A1 (dated 2026-10-03, after the run; implementation guard repair, not a scientific redefinition).

The FARE wrapper refused every certificate because it detects "fit rows" by byte-identical FEATURE vectors. Distinct
records (different persons/records, disjoint roles by construction) can share a feature vector; under the
certificate's i.i.d. premise such coincidences are legitimate draws, and dropping them would condition the cert set on
the fit set. The repair replaces the feature-hash guard by a ROW-IDENTITY guard (cert row ids disjoint from the tree's
fit row ids, asserted here) and recomputes the official certificate unchanged otherwise. The original UNAVAILABLE
records are kept beside the amended ones. Certificates are native tests, outside the primary family."""
import copy, json, sys
from pathlib import Path
import numpy as np

WT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WT))
from oar import fare_official as FO  # noqa: E402
from oar import study as S  # noqa: E402

lock = json.loads((WT / "results/combined_output_aware_removal_v1/EXECUTION_LOCK.json").read_text())
fare = lock["fare"]
out = {}
for ds in ("adult", "hmda"):
    W = S.load_world(ds)
    df, cr = W["idx"]["defense_fit"], W["idx"]["cert"]
    assert len(np.intersect1d(W["row_id"][df], W["row_id"][cr])) == 0
    assert len(np.intersect1d(W["unit"][df], W["unit"][cr])) == 0
    for k in S.SEEDS:
        P = f"{ds}__s{k}"
        sp = S.RUN / "selection" / f"{P}.json"
        if not sp.exists():
            continue
        sel = json.loads(sp.read_text())
        H = S.load_seed(ds, k)["H"]
        orig = json.loads((S.RUN / "certificates" / f"{P}.json").read_text())
        dup = {}
        rec = {"original": orig, "amended": {}}
        for tag, uid in (("nominee", f"{P}__FAREFIT_c{sel['nominee_unit_source']}"), ("zero_fairness", f"{P}__FAREFIT_Z")):
            m = FO.load_model(S.unit_dir(uid)) if hasattr(FO, "load_model") else None
            if m is None:
                raise SystemExit("wrapper has no load_model(); cannot amend")
            n_feat_dup = int(np.isin(FO.row_hashes(H[cr]), m.fit_row_hashes).sum())
            m2 = copy.copy(m)
            m2.fit_row_hashes = np.array([], dtype=m.fit_row_hashes.dtype)   # feature-hash guard replaced by row-id guard (asserted above)
            res = {"primary_all_groups": FO.certificate(m2, H[cr], W["s"][cr], fare["certificate"])}
            sec = fare.get("secondary_certificate_groups", {}).get(ds)
            if sec:
                res["secondary_groups"] = FO.certificate(m2, H[cr], W["s"][cr], {**fare["certificate"], "groups": sec})
            res["cert_rows_with_feature_vector_equal_to_a_fit_row"] = n_feat_dup
            rec["amended"][tag] = res
        out[P] = rec
        print(P, {t: (r["primary_all_groups"].get("status"), r["primary_all_groups"].get("dp_ub", r["primary_all_groups"].get("bound"))) for t, r in rec["amended"].items()})
(S.RUN / "certificates" / "AMENDMENT_A1.json").write_text(json.dumps(out, indent=1, default=str))
