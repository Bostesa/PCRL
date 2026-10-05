"""Private backup with uncached read-back, restore checks from the drive copy alone, and the model/restore manifests.

    OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python -m smf.closeout --dest <drive folder>

Copies <PRIVATE_CACHE>/smf_v1 to <dest>/private_smf_v1_20261005/smf_v1 (refuses to overwrite a differing file;
nothing is deleted anywhere), writes SHA256SUMS, re-reads every copied file with F_NOCACHE (uncached read; not a
physical cold-disk read), then from the drive copy alone: rebuilds the releases of U, J-F (candidate or descriptive
checkpoint) and L-F for seed 1 with the deployment graph and compares them bitwise with the local releases; refits the
recorded AUC-selected coalition attacker of outer__s1__J-F from the drive-copy release (AUDIT_FIT) and compares its
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

from smf import run as R

SRC = R.PRIV
NAME = "private_smf_v1_20261005"


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
    dst = root / "smf_v1"
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
    (root / "SHA256SUMS").write_text("".join(f"{h}  smf_v1/{r}\n" for r, h in sums.items()))
    ok = sum(sha(dst / r, nocache=True) == h for r, h in sums.items())
    return root, {"files": len(files), "copied_now": copied, "already_identical": same, "uncached_readback_match": ok,
                  "bytes": int(sum(p.stat().st_size for p in files)),
                  "SHA256SUMS_sha256": sha(root / "SHA256SUMS")}


def restore_checks(root, D):
    from smf import deploy as DP
    el = json.loads((R.PKG / "EVALUATION_LOCK.json").read_text())
    sc = el["seeds"]["1"]["score"]
    out = []
    for lab in ("U", "J-F", "L-F"):
        unit = sc[lab]["unit"]
        drv = root / "smf_v1" / "run" / "units" / unit
        rel = DP.release(drv, D["X"])
        loc = np.load(R.U(unit) / "release.npz")
        out.append({"label": lab, "unit": unit, "source": "drive copy",
                    "bitwise_equal_to_local_release": all(np.array_equal(rel[k], loc[k]) for k in rel)})
    # attacker restore: refit the recorded AUC-selected coalition attacker from the drive-copy release
    from smf import audit as AU
    from rgj import finalize as FN
    orec = json.loads((root / "smf_v1" / "run" / "units" / "outer__s1__J-F" / "record.json").read_text())
    z = np.load(root / "smf_v1" / "run" / "units" / sc["J-F"]["unit"] / "release.npz")
    V = FN.views_from_release(z)
    scored = orec["primary"]["scored"]["pair"]["auc"]
    name, src = scored.get("attacker"), scored.get("source_view", "pair")
    fac = dict(AU.final_slate(False))[name]
    idx = D["idx"]
    m = fac(0).fit(V[src][idx["AUDIT_FIT"]], D["sex"][idx["AUDIT_FIT"]])
    P = AU.proba(m, V[src][idx["DEVELOPMENT_ASSESSMENT"]], 2)
    saved = np.load(root / "smf_v1" / "run" / "units" / "outer__s1__J-F" / "preds.npz")["P_auc_pair"][0]
    out.append({"label": "attacker (outer__s1__J-F, pair, AUC-selected)", "attacker": name, "source_view": src,
                "source": "drive copy release + AUDIT_FIT refit", "max_abs_diff_vs_saved": float(np.abs(P - saved).max())})
    return out


def manifests(root_rel, backup_rec, restore):
    el = json.loads((R.PKG / "EVALUATION_LOCK.json").read_text())
    inf = json.loads((R.RUN / "inference.json").read_text())
    adv = inf["claimA"]["decision"] == "PASS" and inf["claimB"]["decision"] == "PASS"
    mm = {"schema": "smf-model-manifest-v1",
          "label": "DEVELOPMENT_ADVANTAGE_ESTABLISHED" if adv else "EXPERIMENTAL_NO_ADVANTAGE",
          "claims": {c: inf[f"claim{c}"]["decision"] for c in ("A", "B")},
          "note": "Per-seed statuses below: a NOMINEE J-F is the candidate release; a NO_FEASIBLE_NOMINEE J-F is the "
                  "declared descriptive fallback, labelled EXPERIMENTAL. C* (strongest task-feasible control) is packaged "
                  "as the strongest development baseline beside it.",
          "deploy": "python -m rgj.deploy --unit <unit> --X <permitted_inputs.npy (83 columns)> --out release.npz",
          "seeds": {}}
    for s, v in el["seeds"].items():
        mm["seeds"][s] = {lab: {"spec": spec, "status": v["status"].get(lab),
                                "files_sha256": {u: v["unit_file_sha256"][u] for u in
                                                 ([spec["unit"]] if spec["kind"] == "neural" else spec["units"])}}
                          for lab, spec in v["score"].items()}
        mm["seeds"][s]["C*"] = v["comparator"]
    (R.PKG / "MODEL_MANIFEST.json").write_text(json.dumps(mm, indent=1) + "\n")
    ri = {"schema": "smf-restore-index-v1", "private_local": "<PRIVATE_CACHE>/smf_v1",
          "drive_copy": f"<DRIVE_ROOT>/{root_rel}/smf_v1", "checksums": f"<DRIVE_ROOT>/{root_rel}/SHA256SUMS",
          "layout": {"run/units/<unit>/": "atomic unit: files + record.json + COMPLETE.json (sha256 of every file)",
                     "ck__*, tl__*, tc__*, lc__*": "model.pt, head_0/1.joblib, release.npz (+ critics.pt / leace_i/)",
                     "fare__*": "re-headed official FARE releases (trees from the predecessor jcv_v1 store / fare_cache)",
                     "run__*": "training receipts: final.pt (theta_T-1, theta_T, critics), captures.pt",
                     "outer__*": "assessment records and per-row predictions (private)",
                     "inner__*, calib__*, track__*, whiten__diag, controls__s0, parity__*": "inner-role records"},
          "restore": ["copy <DRIVE_ROOT>/" + root_rel + "/smf_v1 to <PRIVATE_CACHE>/smf_v1",
                      "verify: shasum -a 256 -c SHA256SUMS (run inside the copy's parent)",
                      "predecessor inputs/warm starts/FARE trees: <PRIVATE_CACHE>/jcv_v1 (drive copy "
                      "<DRIVE_ROOT>/private_jcv_v1_20261003)", "then follow QUICKSTART.md"],
          "restore_checks": restore}
    (R.PKG / "RESTORE_INDEX.json").write_text(json.dumps(ri, indent=1) + "\n")
    bv = {"schema": "smf-backup-verification-v1", "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
          "source": "<PRIVATE_CACHE>/smf_v1", "destination": f"<DRIVE_ROOT>/{root_rel}", **backup_rec,
          "read_back": "every copied file re-read with F_NOCACHE (uncached read; not a physical cold-disk read)",
          "restore_checks": restore, "deleted": "nothing"}
    (R.PKG / "BACKUP_VERIFICATION.json").write_text(json.dumps(bv, indent=1) + "\n")
    return bv


def main(argv=None):
    from smf import data as DA
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    a = ap.parse_args(argv)
    root, rec = backup(a.dest)
    D = DA.load()
    restore = restore_checks(root, D)
    rel = NAME   # public files name the drive only as <DRIVE_ROOT>; never the local folder
    bv = manifests(rel, rec, restore)
    (root / "BACKUP_RECORD.json").write_text(json.dumps(bv, indent=1) + "\n")
    print(json.dumps({k: bv[k] for k in ("files", "copied_now", "uncached_readback_match", "restore_checks")}, indent=1))


if __name__ == "__main__":
    main()
