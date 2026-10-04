"""Private backup to the drive with uncached read-back and restore replays from the drive copies only.
    OMP_NUM_THREADS=1 ~/PCRL/.venv/bin/python results/pcrl_penalty_no_erasure_v1/report/backup_and_restore.py --dest <drive>
Copies ~/PCRL_eval_cache_private/pnx_v1 to <dest>/private_pnx_v1_20261003/pnx_v1 (refuses to overwrite a differing file;
nothing deleted), writes SHA256SUMS, re-reads every file with F_NOCACHE (uncached; not a cold unmount), then from the
drive copies alone restores PN and LN (selected seed-1 units, identity map) and the reference E (predecessor drive copy
private_jcv_v1_20261003, official LEACE map) and reproduces their releases with the deployment graph."""
import argparse, fcntl, hashlib, json, os, shutil, sys, time
from pathlib import Path
import numpy as np
WT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WT))
SRC = Path.home() / "PCRL_eval_cache_private" / "pnx_v1"
NAME = "private_pnx_v1_20261003"


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
    dst = root / "pnx_v1"
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
            f.write(f"{sha(p)}  pnx_v1/{p.relative_to(SRC)}\n")
    ok = bad = 0
    for line in open(sums):
        h, rel = line.rstrip("\n").split("  ", 1)
        if sha(root / rel, nocache=True) == h:
            ok += 1
        else:
            bad += 1
    import joblib, torch
    from jcv import finalize as FN, train as T
    from stored_model_eval.defenses import LeaceMap
    pred_drive = Path(a.dest) / "private_jcv_v1_20261003" / "jcv_v1"
    X = np.load(pred_drive / "inputs" / "adult_jcv.npz")["X"]
    replays = []
    for d in (dst / "run" / "units" / "pn__s1__PN__b0.1", dst / "run" / "units" / "pn__s1__LN__b0.1",
              pred_drive / "run" / "units" / "nn__s1__E"):
        rec = json.loads((d / "record.json").read_text())
        model = T.Model(X.shape[1], [2, 6], rec["seed"])
        model.load_state_dict(torch.load(d / "model.pt"))
        rel = np.load(d / "release.npz")
        diffs = {}
        with torch.no_grad():
            for i in (0, 1):
                H = model.encode(i, torch.from_numpy(X)).double().numpy()
                m = d / f"leace_{i}"
                r = LeaceMap.load(m, verify_package=True).transform(H) if m.exists() else H
                c, P, hard = FN.outputs(joblib.load(d / f"head_{i}.joblib"), r)
                diffs.update({f"r{i + 1}": float(np.abs(r - rel[f"r{i + 1}"]).max()), f"c{i + 1}": float(np.abs(c - rel[f"c{i + 1}"]).max()),
                              f"p{i + 1}": float(np.abs(P - rel[f"p{i + 1}"]).max()),
                              f"hard{i + 1}_equal": bool(np.array_equal(hard, rel[f"hard{i + 1}"]))})
        replays.append({"unit": d.name, "source": "drive copy", "map": "LEACE" if (d / "leace_0").exists() else "identity", **diffs})
    rec = {"schema": "pnx-private-backup-v1", "name": NAME, "dest": "<drive>/" + NAME, "source": "~/PCRL_eval_cache_private/pnx_v1",
           "files": len(files), "copied_now": copied, "already_identical": same,
           "uncached_readback": f"{ok}/{ok + bad} match (F_NOCACHE uncached read; no cold unmount)",
           "SHA256SUMS_sha256": sha(sums), "restore_replay_from_drive": replays,
           "reused_predecessor_data": "<drive>/private_jcv_v1_20261003 (verified 2085/2085 in the predecessor study)",
           "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "deleted": "nothing"}
    (root / "BACKUP_RECORD.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
