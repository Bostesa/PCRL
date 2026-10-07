"""Custody runner of the held-out calibration study (hcal; roles A + F; prompt section 14). Closeout code, separately
versioned; it never changes a locked result. Uses hcal.closeout (role F) for the copy, uncached re-reads and receipts.

  copy      versioned SAME_DEVICE_COPY of <PRIVATE_CACHE>/hcal_v1 (+ the pinned input), SHA256SUMS, uncached re-read
  restore   from the COPY ALONE (subprocesses with PCRL_HCAL_PRIVATE_CACHE pointing at the copied store, so every hcal
            path resolves inside the copy; outputs are written outside the copy):
              teachers U s0-2 (own forward pass, bitwise), representative frozen-bank tables (sha256),
              every calibrator family (H-TOKEN32, H-GLOBAL-TEMP, H-CLASS-TEMP, T-TOKEN32 and the U calibrations: refit
              from the copied bank / teacher and the calibration labels, bitwise vs the copied tables),
              the selected and control releases (python -m hcal.deploy on the copied teacher / policy / decoder,
              bitwise vs the registered release arrays), and the selected reader (the P* partition's pair AUC winner
              at seed 0 refit from the copy and scored on the assessment rows, exact vs the saved oatt__ predictions).
  The sealed D is loaded with the pinned loader on the pinned input, whose sha256 must equal the copy's bundled input
  (no loader rebinding).

    PYTHONPATH=. <python> -m hcal.custody copy
    PYTHONPATH=. <python> -m hcal.custody restore <copy root>
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from hcal import ids as I

REP_PARTITIONS = ("U|JOINT|i8o64|l0.1", "U|FINE-TASK|i8o64", "U|CLASS|i1o1", "U|C-TASK|i8o64")


def _sub(code, store, timeout=3600):
    env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONPATH=".", PCRL_HCAL_PRIVATE_CACHE=str(store))
    r = subprocess.run([sys.executable, "-c", code], cwd=str(I.WT), env=env, capture_output=True, text=True,
                       timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout)[-1500:])
    return json.loads(r.stdout.strip().splitlines()[-1])


CAL_CODE = r'''
import json, numpy as np
from hcal import ids as I, run as R, stages as ST, calib as C, data as HD
D = HD.load()
out = {}
h_rows, h_y = ST.cal_labels(D, "CALIBRATION_HELDOUT"); t_rows, t_y = ST.cal_labels(D, "CALIBRATION_TRAIN_MATCHED")
for k, p in json.loads(__ARGS__):
    b = R.bank(k, p); z = R.npz(ST.cal_name(k, p), "tables.npz")
    for fam in [f for f in I.decoders_of(p) if f in I.NEW_DECODERS + (I.DIAG_DECODER,)]:
        rows, ys = (t_rows, t_y) if fam == I.DIAG_DECODER else (h_rows, h_y)
        tabs = C.calibrate_partition(b, rows, ys, fam)
        out[f"{k}|{p}|{fam}"] = all(bool(np.array_equal(tabs[i]["q"], z[f"{fam}|q{i}"])) for i in (1, 2))
for k in (0, 1, 2):
    T = R.teacher(k); z = R.npz(ST.calu_name(k), "u.npz")
    for fam in C.TEMP_FAMILIES:
        ok = True
        for i in (1, 2):
            P = np.asarray(T[f"p{i}"], dtype=np.float64); d = np.asarray(T[f"d{i}"], dtype=np.int64)
            tab = C.fit_u_temp(P[h_rows], h_y[ST.TASKS[i - 1]], fam, d[h_rows])
            ok = ok and bool(np.array_equal(C.apply_u(P, tab, d), z[f"{fam}|p{i}"]))
        out[f"{k}|SRC|U|{fam}"] = ok
print(json.dumps(out))
'''

READER_CODE = r'''
import json, numpy as np
from hcal import ids as I, run as R, stages as ST, data as HD
D = HD.load()
a = np.asarray(D["idx"]["OSF_DEVELOPMENT_ASSESSMENT"], dtype=np.int64)
k, key, w, crit = json.loads(__ARGS__)
_, P = ST.refit_winner(k, key, w, crit, D, score_rows=a)
np.save(__OUT__, P)
det = R.rec(ST.com_name(k, key))["recovery"]["winner_detail"][w][crit]
print(json.dumps({"bank": det["bank"], "kind": det["kind"], "attacker": det.get("attacker"), "shape": list(P.shape)}))
'''


def copy():
    from hcal import closeout as CO
    rec = CO.same_device_copy()
    pub = CO.public(rec)
    (I.RUN / "same_device_copy.json").write_text(json.dumps(pub, indent=1) + "\n")
    print(json.dumps({k: pub.get(k) for k in ("label", "destination", "store_files", "verified")}, indent=1))
    return rec


def restore(copy_root):
    from hcal import closeout as CO
    from hcal import infer as INF
    root = Path(copy_root)
    store = CO.copy_store(root)
    checks = {}
    src, want = CO.pinned_input()
    bundled = root / "dependencies" / "jcv_v1" / "inputs" / "adult_jcv.npz"
    have = CO.sha(bundled, nocache=True) if bundled.exists() else None
    checks["pinned_input_equals_copy"] = {"status": "PASS" if have == want == CO.sha(src, nocache=True) else "FAIL",
                                          "sha256": have, "rule": "the sealed D used by the restores is loaded with "
                                          "the pinned loader on the pinned input, byte-identical to the copy's bundle"}
    t = CO.restore_teachers(root, seeds=I.SEEDS, teachers=("U",))
    checks["teacher"] = {"status": "PASS" if t and all(v.get("status") == "PASS" for v in t.values()) else "FAIL",
                         "per_seed": t}
    checks["frozen_bank"] = CO.restore_bank_tables(root)
    jobs = [(k, p) for k in I.SEEDS for p in REP_PARTITIONS]
    try:
        cal = _sub(CAL_CODE.replace("__ARGS__", repr(json.dumps(jobs))), store)
        fam = {}
        for key, ok in cal.items():
            f = key.split("|")[-1] if "SRC|U" not in key else "U|" + key.split("|")[-1]
            fam.setdefault(f, []).append(ok)
        checks["calibrator_families"] = {"status": "PASS" if all(all(v) for v in fam.values()) and
                                         set(I.NEW_DECODERS) <= set(fam) and I.DIAG_DECODER in fam else "FAIL",
                                         "families": {f: {"tables": len(v), "bitwise": all(v)} for f, v in fam.items()},
                                         "partitions": list(REP_PARTITIONS), "seeds": list(I.SEEDS)}
    except Exception as e:                                                            # noqa: BLE001
        checks["calibrator_families"] = {"status": "FAIL", "reason": str(e)[-600:]}
    sel = json.loads((store / "run" / "selection.json").read_text())
    rids = [sel["P*"]["release"], sel["T*"]["release"]]
    from hcal import deploy_checks as DCK
    rec = DCK.run(1, rids, root=str(store))
    checks["selected_and_control_release"] = {
        "status": "PASS" if rec["all_deployments_bitwise"] else "FAIL", "seed": 1,
        "releases": {r: {"rc": v["rc"], "bitwise": v.get("bitwise_equal_to_registered_release"),
                         "decoder_sha256": v.get("decoder_sha256")} for r, v in rec["deployments"].items()},
        "refusals_exit_2": rec.get("all_refusals_exit_2"), "from": "copy (teacher, policy, decoder, input)"}
    key = I.parse_release(sel["P*"]["release"])[0]
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "P.npy"
        try:
            info = _sub(READER_CODE.replace("__ARGS__", repr(json.dumps([0, key, "pair", "auc"])))
                        .replace("__OUT__", repr(str(out))), store)
            P = np.load(out)
            res = {"status": "FAIL", **info}
            for nm, base in (("copy", store), ("live", I.PRIV)):
                with np.load(base / "run" / "units" / INF.att_name(0, key) / "preds.npz", allow_pickle=False) as z:
                    s = np.asarray(z["P_auc_pair"])
                res[f"max_abs_diff_vs_{nm}"] = float(np.abs(P - s).max()) if P.shape == s.shape else None
            res["status"] = "PASS" if all(res[f"max_abs_diff_vs_{n}"] == 0.0 for n in ("copy", "live")) else "FAIL"
            res["note"] = "attacker seeds 0-2 refit from the copy; INNER refits bitwise first; compared on all " \
                          "assessment rows"
        except Exception as e:                                                        # noqa: BLE001
            res = {"status": "FAIL", "reason": str(e)[-600:]}
    checks["selected_reader"] = {**res, "partition": key, "view": "pair", "criterion": "auc", "seed": 0}
    summ = {k: v.get("status") for k, v in checks.items()}
    body = {"schema": "hcal-restore-v1", "copy": root.name, "checks": checks, "summary": summ,
            "all_pass": all(s == "PASS" for s in summ.values())}
    (I.RUN / "restore_checks.json").write_text(json.dumps(CO.public(body), indent=1) + "\n")
    print(json.dumps(summ, indent=1))
    return body


if __name__ == "__main__":
    if sys.argv[1] == "copy":
        copy()
    elif sys.argv[1] == "restore":
        restore(sys.argv[2])
