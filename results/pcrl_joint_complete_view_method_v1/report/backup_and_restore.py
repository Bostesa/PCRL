"""Versioned private backup to the external drive with uncached read-back and restore replays from the copy alone.

    OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python results/pcrl_joint_complete_view_method_v1/report/backup_and_restore.py --dest <drive>

Copies ~/PCRL_eval_cache_private/jcv_v1 to <dest>/private_jcv_v1_20261003/jcv_v1 (refuses to overwrite a differing
file; nothing deleted), writes SHA256SUMS, re-reads every copied file with F_NOCACHE (uncached read; NOT a cold-disk
unmount test), then restores FROM THE COPY ONLY: the candidate J, the L control (seed 0 selected/descriptive units)
and the published control F (FARE trees re-encode the permitted inputs in the official environment) and reproduces
their released features, centred logits, probabilities and decisions.
"""
import argparse, fcntl, hashlib, json, os, shutil, sys, time
from pathlib import Path

import numpy as np

WT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WT))
SRC = Path.home() / "PCRL_eval_cache_private" / "jcv_v1"
NAME = "private_jcv_v1_20261003"


def sha(p, nocache=False):
    fd = os.open(p, os.O_RDONLY)
    try:
        if nocache:
            fcntl.fcntl(fd, 48, 1)
        h = hashlib.sha256()
        while True:
            b = os.read(fd, 1 << 22)
            if not b:
                break
            h.update(b)
        return h.hexdigest()
    finally:
        os.close(fd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    a = ap.parse_args()
    root = Path(a.dest) / NAME
    dst = root / "jcv_v1"
    files = sorted(p for p in SRC.rglob("*") if p.is_file())
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
    sums = root / "SHA256SUMS"
    with open(sums, "w") as f:
        for p in files:
            f.write(f"{sha(p)}  jcv_v1/{p.relative_to(SRC)}\n")
    ok = bad = 0
    for line in open(sums):
        h, rel = line.rstrip("\n").split("  ", 1)
        if sha(root / rel, nocache=True) == h:
            ok += 1
        else:
            bad += 1
    # ---- restore replay from the copy only
    import joblib
    import torch
    from jcv import finalize as FN, train as T
    from stored_model_eval.defenses import LeaceMap
    Z = np.load(dst / "inputs" / "adult_jcv.npz")
    X = Z["X"]
    SL = json.loads((WT / "results/pcrl_joint_complete_view_method_v1/SELECTION_LOCK.json").read_text())
    units = dst / "run" / "units"
    replays = []
    for arm in ("J", "L"):
        nm = SL["seeds"]["0"]["arms"][arm]["unit"]
        d = units / nm
        model = T.Model(X.shape[1], [2, 6], json.loads((d / "record.json").read_text())["seed"])
        model.load_state_dict(torch.load(d / "model.pt"))
        rel = np.load(d / "release.npz")
        diffs = {}
        with torch.no_grad():
            for i in (0, 1):
                H = model.encode(i, torch.from_numpy(X)).double().numpy()
                r = LeaceMap.load(d / f"leace_{i}", verify_package=True).transform(H)
                c, P, hard = FN.outputs(joblib.load(d / f"head_{i}.joblib"), r)
                diffs.update({f"r{i + 1}": float(np.abs(r - rel[f"r{i + 1}"]).max()),
                              f"c{i + 1}": float(np.abs(c - rel[f"c{i + 1}"]).max()),
                              f"p{i + 1}": float(np.abs(P - rel[f"p{i + 1}"]).max()),
                              f"hard{i + 1}_equal": bool(np.array_equal(hard, rel[f"hard{i + 1}"]))})
        replays.append({"arm": arm, "unit": nm, **diffs})
    os.environ["OAR_RUN_UNITS"] = str(dst / "fare_cache")
    from oar import fare_official as FO
    for nm in SL["seeds"]["0"]["arms"]["F"]["units"]:
        d = units / nm
        uid = json.loads((d / "record.json").read_text())["fare_uid"]
        m = FO.FareModel.load(dst / "fare_cache" / uid / "model")
        cells = FO.encode(m, X.astype(np.float64))
        rel = np.load(d / "release.npz")
        R = np.eye(rel["r"].shape[1])[cells]
        c, P, hard = FN.outputs(joblib.load(d / "head.joblib"), R)
        replays.append({"arm": "F", "unit": nm, "cells_equal": bool(np.array_equal(cells, rel["cells"])),
                        "c": float(np.abs(c - rel["c"]).max()), "p": float(np.abs(P - rel["p"]).max()),
                        "hard_equal": bool(np.array_equal(hard, rel["hard"]))})
    rec = {"schema": "jcv-private-backup-v1", "name": NAME, "dest": "<drive>/" + NAME, "source": "~/PCRL_eval_cache_private/jcv_v1",
           "files": len(files), "copied_now": copied, "already_identical": same,
           "uncached_readback": f"{ok}/{ok + bad} match (F_NOCACHE uncached read; no cold unmount)",
           "SHA256SUMS_sha256": sha(sums), "restore_replay_from_copy": replays,
           "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "deleted": "nothing"}
    (root / "BACKUP_RECORD.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
