"""Custody of the ccm sprint (role A; closeout code written after FEASIBILITY_LOCK; it never changes a locked result).

  copy      versioned SAME_DEVICE_COPY of <PRIVATE_CACHE>/ccm_v1 plus the pinned source input (bundled under
            dependencies/), SHA256SUMS, and an uncached re-read of every copied file. Never deletes or moves an original.
  restore   from the COPY: a subprocess with PCRL_CCM_PRIVATE_CACHE pointing inside the copy (so every ccm path resolves
            there) reloads the cached Ucal reference vectors, recomputes the registered G geometry (F1-F5, go inputs)
            for every seed and recipient, and compares it with the copied private records; recomputes the finite-law
            oracles and compares them with ORACLE_RESULTS.json; and re-hashes the copy afterwards (the restore must not
            write into it). The bundled input must be byte-identical to the pinned input.

    PYTHONPATH=. <python> -m ccm.custody copy
    PYTHONPATH=. <python> -m ccm.custody restore <copy root>
Writes BACKUP_VERIFICATION.json and RESTORE_INDEX.json (public, scrubbed paths) and <PRIVATE_CACHE>/ccm_v1/run/
same_device_copy.json / restore_checks.json.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from ccm import ids as I

CACHE = I.PRIV.parent
PINNED_INPUT = CACHE / "jcv_v1" / "inputs" / "adult_jcv.npz"
PINNED_SHA = "e0d9e54af780f30788ee29cfe6795ec82cbdcadc127b1978c69a3891485d2f12"
F_NOCACHE = 48                                  # macOS fcntl: bypass the unified buffer cache on re-read


def sha(p, nocache=False):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        if nocache:
            try:
                fcntl.fcntl(f.fileno(), F_NOCACHE, 1)
            except OSError:
                pass
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def scrub(s):
    return str(s).replace(str(CACHE), "<PRIVATE_CACHE>").replace(str(I.WT), "<REPO>").replace(str(Path.home()), "~")


def _files(root):
    return sorted(p for p in Path(root).rglob("*") if p.is_file() and p.name != "SHA256SUMS")


def copy():
    stamp = time.strftime("%Y%m%d", time.gmtime())
    dest = CACHE / f"ccm_v1_local_copy_{stamp}"
    if dest.exists():
        raise SystemExit(f"REFUSED: {scrub(dest)} exists (versioned copies are never overwritten)")
    if sha(PINNED_INPUT) != PINNED_SHA:
        raise SystemExit("REFUSED: pinned input hash mismatch")
    free0 = shutil.disk_usage(CACHE).free
    src_files = _files(I.PRIV)
    before = {str(p.relative_to(I.PRIV)): sha(p) for p in src_files}
    shutil.copytree(I.PRIV, dest / "ccm_v1")
    dep = dest / "dependencies" / "jcv_v1" / "inputs"
    dep.mkdir(parents=True)
    shutil.copy2(PINNED_INPUT, dep / "adult_jcv.npz")
    after = {str(p.relative_to(I.PRIV)): sha(p) for p in src_files}
    changed = sorted(f for f in before if before[f] != after.get(f))
    sums = {str(p.relative_to(dest)): sha(p) for p in _files(dest)}
    (dest / "SHA256SUMS").write_text("".join(f"{h}  {f}\n" for f, h in sorted(sums.items())))
    reread = {f: sha(dest / f, nocache=True) for f in sums}
    match = sum(reread[f] == sums[f] for f in sums)
    src_match = sum(sums.get(f"ccm_v1/{f}") == after[f] for f in after)
    rec = {"label": "SAME_DEVICE_COPY", "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "source": scrub(I.PRIV), "destination": scrub(dest), "versioned_folder": dest.name,
           "store_files": len(src_files), "files": len(sums), "bytes": sum((dest / f).stat().st_size for f in sums),
           "dependencies": {"dependencies/jcv_v1/inputs/adult_jcv.npz": f"pinned source input (sha256 {PINNED_SHA})"},
           "source_files_equal_to_copy": src_match, "live_files_changed_during_copy": changed,
           "SHA256SUMS_sha256": sha(dest / "SHA256SUMS"),
           "uncached_readback": {"entries": len(sums), "match": match, "pass": match == len(sums)},
           "free_gib_before": round(free0 / 2**30, 2), "free_gib_after": round(shutil.disk_usage(CACHE).free / 2**30, 2),
           "originals_deleted": False}
    rec["verified"] = bool(rec["uncached_readback"]["pass"] and src_match == len(src_files) and not changed)
    (I.RUN / "same_device_copy.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: rec[k] for k in ("destination", "files", "verified")}, indent=1))
    return rec


RESTORE_CODE = r'''
import json, numpy as np
from ccm import ids as I, data as CD, geometry as GE, guard as GU, oracle as OR, run as R
out = {"geometry": {}, "store": str(I.PRIV)}
for k in I.SEEDS:
    fit = CD.reference(k, R.GEOMETRY_ROLES["fit"]); held = CD.reference(k, R.GEOMETRY_ROLES["held"])
    for i in I.RECIPIENTS:
        rec = GE.run_geometry(fit["Ucal"][i], held["Ucal"][i], GU.capacity_for(recipient=R.NAMES[i]),
                              dec_fit=fit["d"][i], dec_held=held["d"][i], contract="G")
        pub = R._finite(R._split_public(rec))
        out["geometry"][f"s{k}|r{i}"] = pub
laws = json.loads((I.PKG / "TOY_LAWS.json").read_text())
out["oracle"] = R._finite(OR.run_all(laws))
print(json.dumps(out, allow_nan=False))
'''


def _strip_timing(o):
    if isinstance(o, dict):
        return {k: _strip_timing(v) for k, v in o.items() if k not in ("cpu_s", "wall_s")}
    if isinstance(o, list):
        return [_strip_timing(v) for v in o]
    return o


def restore(copy_root):
    root = Path(copy_root).resolve()
    store = root / "ccm_v1"
    sums0 = {f: sha(root / f) for f in (l.split("  ", 1)[1].strip() for l in (root / "SHA256SUMS").read_text().splitlines())}
    checks = {}
    bundled = root / "dependencies" / "jcv_v1" / "inputs" / "adult_jcv.npz"
    checks["pinned_input_equals_copy"] = {"status": "PASS" if sha(bundled, True) == PINNED_SHA == sha(PINNED_INPUT, True)
                                          else "FAIL", "sha256": PINNED_SHA}
    env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONPATH=".", PCRL_CCM_PRIVATE_CACHE=str(store))
    t0 = time.time()
    r = subprocess.run([sys.executable, "-c", RESTORE_CODE], cwd=str(I.WT), env=env, capture_output=True, text=True,
                       timeout=3600)
    if r.returncode != 0:
        checks["restore_subprocess"] = {"status": "FAIL", "reason": scrub((r.stderr or r.stdout)[-1500:])}
    else:
        got = json.loads(r.stdout.strip().splitlines()[-1])
        checks["store_resolved_inside_copy"] = {"status": "PASS" if Path(got["store"]).resolve() == store else "FAIL"}
        per = {}
        for key, pub in got["geometry"].items():
            k, i = key.split("|")
            rec = json.loads((store / "run" / "geometry" / f"G__{k}__{i}.json").read_text())
            saved = _strip_timing(R_split(rec))
            per[key] = _strip_timing(pub) == saved
        checks["geometry_G_recomputed_from_copy"] = {"status": "PASS" if per and all(per.values()) else "FAIL",
                                                     "units": per, "compared": "every public F1-F5 field and the go "
                                                     "inputs, timing fields excluded"}
        orc = json.loads((I.PKG / "ORACLE_RESULTS.json").read_text())["results"]
        checks["oracle_recomputed"] = {"status": "PASS" if _strip_timing(got["oracle"]) == _strip_timing(orc) else "FAIL"}
        checks["restore_wall_s"] = {"status": "INFO", "wall_s": round(time.time() - t0, 1)}
    sums1 = {f: sha(root / f) for f in sums0}
    checks["copy_unchanged_by_restore"] = {"status": "PASS" if sums0 == sums1 and
                                           all(l.split("  ", 1)[0] == sums0[l.split("  ", 1)[1].strip()]
                                               for l in (root / "SHA256SUMS").read_text().splitlines()) else "FAIL"}
    summ = {k: v["status"] for k, v in checks.items() if v["status"] != "INFO"}
    body = {"schema": "ccm-restore-v1", "copy": root.name, "checks": checks, "summary": summ,
            "all_pass": all(s == "PASS" for s in summ.values())}
    (I.RUN / "restore_checks.json").write_text(json.dumps(body, indent=1) + "\n")
    print(json.dumps(summ, indent=1))
    return body


def R_split(rec):
    from ccm import run as R
    return R._finite(R._split_public(rec))


def write_public(copy_rec, restore_rec):
    drive = [p.name for p in Path("/Volumes").iterdir() if p.name not in ("Macintosh HD", "BackgroundSyncService Setup")]
    bv = {"schema": "ccm-backup-verification-v1", "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
          "copy": copy_rec, "restore": {k: restore_rec[k] for k in ("summary", "all_pass")},
          "off_device_copy": "PENDING: no external drive attached at closeout" if not drive else "drive present",
          "rule": "originals are never deleted or moved; the copy is versioned and never overwritten"}
    (I.PKG / "BACKUP_VERIFICATION.json").write_text(json.dumps(bv, indent=1) + "\n")
    ri = {"schema": "ccm-restore-index-v1", "written_at": bv["written_at"], "private_local": "<PRIVATE_CACHE>/ccm_v1",
          "copy": f"{copy_rec['destination']} (same device; off-device copy PENDING)",
          "checksums": f"{copy_rec['destination']}/SHA256SUMS",
          "layout": {"ccm_v1/ref/ref__s{k}.npz|json": "label-free Ucal/U0 reference vectors of the permitted roles "
                     "(cache tied to the sha256 of ccm/data.py)",
                     "ccm_v1/run/geometry/{G,G_exp}__s{k}__r{i}.json": "private geometry records (bins, "
                     "representatives, member indices)",
                     "ccm_v1/run/*.jsonl": "activity, compute and semaphore ledgers",
                     "ccm_v1/run_*.log": "stage logs with /usr/bin/time -l",
                     "dependencies/jcv_v1/inputs/adult_jcv.npz": "the pinned source input, bundled"},
          "restore": [f"verify: shasum -a 256 -c SHA256SUMS (inside {copy_rec['destination']})",
                      "copy ccm_v1 back to <PRIVATE_CACHE>/ccm_v1 only if the live store is lost",
                      f"PYTHONPATH=. <python> -m ccm.custody restore {copy_rec['destination']}"],
          "note": "the Ucal cache is rebuilt from the hcal admitted teacher (<PRIVATE_CACHE>/hcal_v1) if lost; that "
                  "store has its own same-device copy (hcal RESTORE_INDEX.json)",
          "required_restores": restore_rec["summary"], "restore_all_pass": restore_rec["all_pass"]}
    (I.PKG / "RESTORE_INDEX.json").write_text(json.dumps(ri, indent=1) + "\n")


def main(argv=None):
    a = (argv or sys.argv[1:])
    if a[:1] == ["copy"]:
        copy()
    elif a[:1] == ["restore"]:
        restore(a[1])
    elif a[:1] == ["publish"]:
        write_public(json.loads((I.RUN / "same_device_copy.json").read_text()),
                     json.loads((I.RUN / "restore_checks.json").read_text()))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
