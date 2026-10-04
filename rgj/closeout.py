"""Private backup with uncached read-back, restore checks from the drive copy alone, and the model/restore manifests.

    OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python -m rgj.closeout --dest <drive folder>

Copies ~/PCRL_eval_cache_private/rgj_v1 to <dest>/private_rgj_v1_20261004/rgj_v1 (refuses to overwrite a differing file;
nothing is deleted anywhere), writes SHA256SUMS, re-reads every copied file with F_NOCACHE (uncached read; not a
physical cold-disk read), then from the drive copy alone: rebuilds the releases of U, J-G (descriptive closest
checkpoint) and L-G for seed 1 with the deployment graph and compares them bitwise with the local releases; refits the
recorded AUC-selected coalition attacker of outer__s1__J-G from the drive-copy release (AUDIT_FIT) and compares its
assessment probabilities with the saved ones. Writes BACKUP_VERIFICATION.json, RESTORE_INDEX.json, MODEL_MANIFEST.json.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from rgj import run as R

SRC = R.PRIV
NAME = "private_rgj_v1_20261004"


def sha(p, nocache=False):
    fd = os.open(p, os.O_RDONLY)
    try:
        if nocache:
            fcntl.fcntl(fd, 48, 1)    # F_NOCACHE
        h = hashlib.sha256()
        while True:
            b = os.read(fd, 1 << 22)
            if not b:
                break
            h.update(b)
        return h.hexdigest()
    finally:
        os.close(fd)


def backup(dest):
    root = Path(dest) / NAME
    dst = root / "rgj_v1"
    files = sorted(p for p in SRC.rglob("*") if p.is_file() and ".tmp" not in p.parts[-2])
    copied = same = 0
    for p in files:
        q = dst / p.relative_to(SRC)
        if q.exists():
            if sha(q) != sha(p):
                raise SystemExit(f"REFUSED: differing file at destination {q.relative_to(root)}")
            same += 1
            continue
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)
        copied += 1
    sums = {str(p.relative_to(SRC)): sha(p) for p in files}
    (root / "SHA256SUMS").write_text("".join(f"{h}  rgj_v1/{r}\n" for r, h in sums.items()))
    ok = sum(sha(dst / r, nocache=True) == h for r, h in sums.items())
    return root, {"files": len(files), "copied_now": copied, "already_identical": same, "uncached_readback_match": ok,
                  "bytes": int(sum(p.stat().st_size for p in files)),
                  "SHA256SUMS_sha256": sha(root / "SHA256SUMS")}


def restore_checks(root, D):
    from rgj import deploy as DP
    el = json.loads((R.PKG / "EVALUATION_LOCK.json").read_text())
    sc = el["seeds"]["1"]["score"]
    out = []
    for lab in ("U", "J-G", "L-G"):
        unit = sc[lab]["unit"]
        drv = root / "rgj_v1" / "run" / "units" / unit
        rel = DP.release(drv, D["X"])
        loc = np.load(R.U(unit) / "release.npz")
        out.append({"label": lab, "unit": unit, "source": "drive copy",
                    "bitwise_equal_to_local_release": all(np.array_equal(rel[k], loc[k]) for k in rel)})
    # attacker restore: refit the recorded AUC-selected coalition attacker from the drive-copy release
    from rgj import audit as AU
    from rgj import finalize as FN
    orec = json.loads((root / "rgj_v1" / "run" / "units" / "outer__s1__J-G" / "record.json").read_text())
    z = np.load(root / "rgj_v1" / "run" / "units" / sc["J-G"]["unit"] / "release.npz")
    V = FN.views_from_release(z)
    scored = orec["primary"]["scored"]["pair"]["auc"]
    name, src = scored.get("attacker"), scored.get("source_view", "pair")
    fac = dict(AU.final_slate(False))[name]
    idx = D["idx"]
    m = fac(0).fit(V[src][idx["AUDIT_FIT"]], D["sex"][idx["AUDIT_FIT"]])
    P = AU.proba(m, V[src][idx["DEVELOPMENT_ASSESSMENT"]], 2)
    saved = np.load(root / "rgj_v1" / "run" / "units" / "outer__s1__J-G" / "preds.npz")["P_auc_pair"][0]
    out.append({"label": "attacker (outer__s1__J-G, pair, AUC-selected)", "attacker": name, "source_view": src,
                "source": "drive copy release + AUDIT_FIT refit", "max_abs_diff_vs_saved": float(np.abs(P - saved).max())})
    return out


def manifests(root_rel, backup_rec, restore):
    el = json.loads((R.PKG / "EVALUATION_LOCK.json").read_text())
    mm = {"schema": "rgj-model-manifest-v1", "label": "EXPERIMENTAL_NO_ADVANTAGE",
          "note": "J-G had no feasible nominee on any seed; the packaged J-G checkpoints are the declared closest "
                  "(descriptive) checkpoints and are labelled EXPERIMENTAL. Baselines and controls are packaged beside it.",
          "deploy": "python -m rgj.deploy --unit <unit> --X <permitted_inputs.npy (83 columns)> --out release.npz",
          "seeds": {}}
    for s, v in el["seeds"].items():
        mm["seeds"][s] = {lab: {"spec": spec, "status": v["status"].get(lab),
                                "files_sha256": {u: v["unit_file_sha256"][u] for u in
                                                 ([spec["unit"]] if spec["kind"] == "neural" else spec["units"])}}
                          for lab, spec in v["score"].items()}
        mm["seeds"][s]["C*"] = v["comparator"]
    (R.PKG / "MODEL_MANIFEST.json").write_text(json.dumps(mm, indent=1) + "\n")
    ri = {"schema": "rgj-restore-index-v1", "private_local": "~/PCRL_eval_cache_private/rgj_v1",
          "drive_copy": f"<drive>/{root_rel}/rgj_v1", "checksums": f"<drive>/{root_rel}/SHA256SUMS",
          "layout": {"run/units/<unit>/": "atomic unit: files + record.json + COMPLETE.json (sha256 of every file)",
                     "ck__*, tl__*, tc__*, lc__*": "model.pt, head_0/1.joblib, release.npz (+ critics.pt / leace_i/)",
                     "fare__*": "re-headed official FARE releases (trees from the predecessor jcv_v1 store / fare_cache)",
                     "run__*": "training receipts: final.pt (theta_T-1, theta_T, critics), captures.pt",
                     "outer__*": "assessment records and per-row predictions (private)",
                     "inner__*, calib__*, track__*, whiten__diag, controls__s0, parity__*": "inner-role records"},
          "restore": ["copy <drive>/" + root_rel + "/rgj_v1 to ~/PCRL_eval_cache_private/rgj_v1",
                      "verify: shasum -a 256 -c SHA256SUMS (run inside the copy's parent)",
                      "predecessor inputs/warm starts/FARE trees: ~/PCRL_eval_cache_private/jcv_v1 (drive copy "
                      "<drive>/private_jcv_v1_20261003)", "then follow QUICKSTART.md"],
          "restore_checks": restore}
    (R.PKG / "RESTORE_INDEX.json").write_text(json.dumps(ri, indent=1) + "\n")
    bv = {"schema": "rgj-backup-verification-v1", "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
          "source": "~/PCRL_eval_cache_private/rgj_v1", "destination": f"<drive>/{root_rel}", **backup_rec,
          "read_back": "every copied file re-read with F_NOCACHE (uncached read; not a physical cold-disk read)",
          "restore_checks": restore, "deleted": "nothing"}
    (R.PKG / "BACKUP_VERIFICATION.json").write_text(json.dumps(bv, indent=1) + "\n")
    return bv


def main(argv=None):
    from rgj import data as DA
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    a = ap.parse_args(argv)
    root, rec = backup(a.dest)
    D = DA.load()
    restore = restore_checks(root, D)
    rel = f"{Path(a.dest).name}/{NAME}"
    bv = manifests(rel, rec, restore)
    (root / "BACKUP_RECORD.json").write_text(json.dumps(bv, indent=1) + "\n")
    print(json.dumps({k: bv[k] for k in ("files", "copied_now", "uncached_readback_match", "restore_checks")}, indent=1))


if __name__ == "__main__":
    main()
