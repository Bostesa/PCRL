"""Versioned private backup with uncached read-back and a restore replay (no fits).

    OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python results/combined_analysis_paper_v1/report/backup_and_restore.py --dest <dir>

Copies ~/PCRL_eval_cache_private/cap_v1 to <dest>/private_cap_v1_20261003/cap_v1 (refuses to overwrite a differing
file), writes SHA256SUMS, re-reads every copied file with F_NOCACHE, then, FROM THE COPY ONLY, reloads sampled saved
attackers and reproduces their saved assessment predictions on surfaces rebuilt from the reused heads. Writes
<dest>/private_cap_v1_20261003/BACKUP_RECORD.json. Nothing is deleted.
"""
import argparse, fcntl, hashlib, json, os, shutil, sys, time
from pathlib import Path
import numpy as np

HOME = Path.home()
SRC = HOME / "PCRL_eval_cache_private" / "cap_v1"
OAR = HOME / "PCRL_eval_cache_private" / "oar_v1" / "run" / "units"
NAME = "private_cap_v1_20261003"


def sha(p, nocache=False):
    fd = os.open(p, os.O_RDONLY)
    try:
        if nocache:
            fcntl.fcntl(fd, 48, 1)  # F_NOCACHE
        h = hashlib.sha256()
        while True:
            b = os.read(fd, 1 << 22)
            if not b:
                break
            h.update(b)
        return h.hexdigest()
    finally:
        os.close(fd)


def surfaces(L):
    d = (L[:, 1] - L[:, 0])[:, None]
    c = L.mean(1, keepdims=True)
    e = np.exp(L - L.max(1, keepdims=True))
    P = e / e.sum(1, keepdims=True)
    H = np.zeros_like(L)
    H[np.arange(len(L)), L.argmax(1)] = 1.0
    return {"full": L, "dc": np.hstack([d, c]), "centred": d, "prob": P, "hard": H}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    a = ap.parse_args()
    root = Path(a.dest).expanduser() / NAME
    dst = root / "cap_v1"
    files = sorted(p for p in SRC.rglob("*") if p.is_file())
    copied = same = 0
    for p in files:
        q = dst / p.relative_to(SRC)
        if q.exists():
            if sha(q) != sha(p):
                raise SystemExit(f"REFUSED: differing file already at destination: {q.relative_to(root)}")
            same += 1
            continue
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)
        copied += 1
    sums = root / "SHA256SUMS"
    with open(sums, "w") as f:
        for p in files:
            f.write(f"{sha(p)}  cap_v1/{p.relative_to(SRC)}\n")
    ok = bad = 0
    for line in open(sums):
        h, rel = line.rstrip("\n").split("  ", 1)
        if sha(root / rel, nocache=True) == h:
            ok += 1
        else:
            bad += 1
    # restore replay from the copy only
    import joblib
    lab = np.load(HOME / "PCRL_eval_cache_private" / "bench_v1" / "inputs" / "adult_labels.npz")
    row_pos = {int(r): i for i, r in enumerate(lab["row_id"])}
    replays = []
    for k, arm, tag, s in ((0, "A", "A", "centred"), (1, "F", "F", "hard"), (2, "F0", "FZ", "prob"), (2, "B", "B", "dc")):
        u = dst / "run" / "units" / f"cap__s{k}__{arm}__out__{s}"
        z = np.load(u / "preds.npz")
        L = np.load(OAR / f"adult__s{k}__HEAD__{tag}" / "preds.npz")["head_outputs_all"]
        idx = np.array([row_pos[int(r)] for r in z["assess_row_id"]])
        X = surfaces(L)[s][idx]
        m = joblib.load(u / "models" / "NL_as0.joblib")
        P = m.predict_proba(X)
        replays.append({"unit": u.name, "max_abs_diff_P_NL_as0": float(np.max(np.abs(P - z["P__NL__as0"])))})
    rec = {"schema": "cap-private-backup-v1", "name": NAME, "dest": "<dest>/" + NAME, "source": "~/PCRL_eval_cache_private/cap_v1",
           "files": len(files), "copied_now": copied, "already_identical": same,
           "uncached_readback": f"{ok}/{ok + bad} match", "SHA256SUMS_sha256": sha(sums),
           "restore_replay_from_copy": replays, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "deleted": "nothing"}
    (root / "BACKUP_RECORD.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))
    if bad or any(r["max_abs_diff_P_NL_as0"] > 1e-12 for r in replays):
        sys.exit(1)


if __name__ == "__main__":
    main()
