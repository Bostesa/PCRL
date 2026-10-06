"""Custody closeout for the decision-preserving compression study (data/custody owner).

Two jobs; each has a --dry-run that writes nothing (backup --dry-run also rehearses the restore checks read-only against
the live private store):

1. New-study backup and independent restore (prompt section 17), called by the lead after the EVALUATION_LOCK:
       OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m dpc.closeout backup [--dest <DRIVE_ROOT>]
                                                         [--targets targets.json] [--seed 1] [--dry-run]
   - destination: --dest, else the external volume holding the verified prior copies, identified by CONTENT (a mounted
     volume qualifies iff <volume>/private_smf_v1_20261005/SHA256SUMS hashes to the value the closed smf study recorded
     in its BACKUP_VERIFICATION.json, as osf/closeout.py does); the boot volume and one unrelated read-only installer
     image are skipped by name before any filesystem call. Copy -> <DRIVE_ROOT>/private_dpc_v1_<UTC date>[_vN]/dpc_v1.
   - drive absent: a versioned SAME-DEVICE copy <PRIVATE_CACHE>/dpc_v1_local_copy_<UTC date>[_vN]/dpc_v1, status
     LOCAL_SAME_DEVICE_COPY_VERIFIED_OFF_DEVICE_BACKUP_PENDING. A same-device copy is never called a drive restore.
   - every copied file is listed in SHA256SUMS and re-read with F_NOCACHE (uncached read; not a physical cold-disk read).
   - restore FROM THE COPY ALONE (input file, admitted teachers, policy units and attack records inside the copy):
       teachers U and RAW-J beta 0.3 (seed --seed): own forward pass (dpc.admit.forward, torch functional ops on the
         copied model.pt, weights_only) + copied deployed heads on the copy's 83-column input -> features, centred
         logits, probabilities, decisions, compared bitwise with the copied release, the live release and the run's
         teacher unit tea__s{k}__<t> when present;
       policies named in targets.json (e.g. the best runnable compact policy and its strongest comparator): the copied
         policy.json is loaded with dpc.release and RE-ENCODED from the restored teacher outputs (dpc.release.encode
         per recipient); tok/q/hard are compared with the copied and the live release.npz; class preservation against
         the restored teacher decisions is checked pointwise. A comparator that is a teacher (tea__) or reference
         (ref__) unit is rebuilt from the copied admitted artifacts instead;
       one final attacker: targets.json names the audit owner's refit entry point ("module:function"), called as
         fn(units_root=<copy>/run/units, D=<D from the copy's input>, **kwargs) -> assessment predictions, compared with
         the saved predictions named in targets.json (read from the copy and from the live store).
   - writes BACKUP_VERIFICATION.json and RESTORE_INDEX.json (public, placeholders only) and BACKUP_RECORD.json in the copy.

   targets.json (written by the lead after the EVALUATION_LOCK):
       {"seed": 1,
        "policies": {"best runnable compact policy (J*)": "pol__s1__<config>",
                     "strongest comparator (C_global)": "pol__s1__<config>" | "tea__s1__U" | "ref__s1__F"},
        "attacker": {"fn": "dpc.audit:<refit function>", "kwargs": {...},
                     "saved": {"unit": "<assessment unit>", "file": "preds.npz", "key": "<array>"},
                     "tolerance": 0.0}}

2. Predecessor custody (prompt section 17): the documented osf routines
       python -m osf.closeout backup --dest <DRIVE_ROOT>        and        python -m osf.closeout predecessor --restore
   write their public receipts into results/pcrl_online_strength_frontier_v1/ (the CLOSED osf results) by default.
   This job runs them in-process from this worktree (osf code byte-identical to the pin, SOURCE_INDEX.json) with
   every public write REDIRECTED into results/pcrl_decision_preserving_compression_v1/provenance/predecessor_custody/
   (osf.closeout.write_public, OUT_REPAIR and REPLAY_OUT rebound; any other write into the closed osf directory is
   refused), checks that the closed osf and smf results trees are byte-identical before and after, and keeps osf's
   attacker restore sealed until this study's EVALUATION_LOCK is pushed (assessment labels are not loaded earlier).
       OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m dpc.closeout predecessor [--restore] [--dry-run]
   Drive absent: records PENDING with the exact commands (provenance/predecessor_custody/STATUS.json and
   ADMISSION.json["custody"]). Nothing in <PRIVATE_CACHE>/osf_v1, <PRIVATE_CACHE>/smf_v1, the custody supplement or any
   closed worktree is modified, moved or deleted.

       OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m dpc.closeout status      (drive probe; writes nothing)
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

HOME = Path.home()
WT = Path(__file__).resolve().parents[1]
REL = "results/pcrl_decision_preserving_compression_v1"
PKG = WT / REL
PRED_DIR = PKG / "provenance" / "predecessor_custody"
CACHE = HOME / "PCRL_eval_cache_private"
SRC = CACHE / "dpc_v1"
OSF_PKG = WT / "results" / "pcrl_online_strength_frontier_v1"
SMF_PKG = WT / "results" / "pcrl_strength_matched_feedback_v1"
SMF_DRIVE_NAME = "private_smf_v1_20261005"
VOLUMES_ROOT = Path("/Volumes")
SKIP_VOLUMES = ("Macintosh HD", "BackgroundSyncService Setup")
DRIVE_FOLDER = "private_dpc_v1_{date}"
LOCAL_FOLDER = "dpc_v1_local_copy_{date}"
TEACHERS = ("U", "RAW-J_b0.3")
INPUT_REL = "admitted/inputs/adult_jcv.npz"
CMD = "OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m dpc.closeout"
PENDING = {
    "new_study_backup_and_restore": f"{CMD} backup --dest <DRIVE_ROOT> --targets <targets.json>",
    "osf_backup_and_predecessor_restore": f"{CMD} predecessor --restore   (runs `python -m osf.closeout backup --dest "
                                          "<DRIVE_ROOT>` and `python -m osf.closeout predecessor --restore` in-process "
                                          "with receipts redirected into " + REL + "/provenance/predecessor_custody/; "
                                          "covers the osf drive copy, the smf drive restore and the two later smf logs)",
    "equivalent_on_the_osf_worktree_NOT_recommended": "cd <SOURCE_WORKTREE> && OMP_NUM_THREADS=1 PYTHONPATH=. <python> "
                                                      "-m osf.closeout backup --dest <DRIVE_ROOT> && ... predecessor "
                                                      "--restore  (writes into the closed osf results directory; use "
                                                      "the redirected form above)"}


# ------------------------------------------------------------------ helpers
def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha(p, nocache=False):
    fd = os.open(p, os.O_RDONLY)
    try:
        if nocache:
            fcntl.fcntl(fd, 48, 1)    # F_NOCACHE: uncached read (not a physical cold-disk read)
        h = hashlib.sha256()
        while True:
            b = os.read(fd, 1 << 22)
            if not b:
                break
            h.update(b)
        return h.hexdigest()
    finally:
        os.close(fd)


def jload(p):
    return json.loads(Path(p).read_text())


def scrub(text):
    """Refuse public text containing a home folder, a volume path or the local user name."""
    if re.search(r"/Users/|/Volumes/|/private/|" + re.escape(HOME.name), text):
        raise SystemExit("REFUSED: identifying path or user name in a public file")
    return text


def scrub_safe(s):
    s = re.sub(r"/(Users|Volumes|private)/[^\s'\"]+", "<PATH>", str(s))
    return s.replace(HOME.name, "<USER>")


def write_public(path, obj):
    txt = scrub(json.dumps(obj, indent=1, default=str) + "\n")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(txt)
    tmp.replace(path)


def versioned(parent: Path, name: str):
    d, n = parent / name, 1
    while d.exists():
        n += 1
        d = parent / f"{name}_v{n}"
    return d


def tree_state(root: Path):
    """(files, sha256 over sorted relative path + file hash) of a directory; None if absent."""
    if not root.exists():
        return None
    files = sorted(p for p in root.rglob("*") if p.is_file())
    h = hashlib.sha256()
    for p in files:
        h.update(str(p.relative_to(root)).encode() + b"\0" + sha(p).encode())
    return {"files": len(files), "tree_sha256": h.hexdigest()}


def lock_pushed():
    from dpc import data as DD
    return DD.evaluation_lock_pushed()


# ------------------------------------------------------------------ drive detection (by content)
def smf_recorded_sums_sha():
    return jload(SMF_PKG / "BACKUP_VERIFICATION.json")["SHA256SUMS_sha256"]


def locate_drive(volumes_root: Path = None, want: str = None):
    """(volume, evidence): the mounted volume holding the verified smf copy, or (None, evidence). Names never recorded."""
    root = Path(volumes_root or VOLUMES_ROOT)
    want = want or smf_recorded_sums_sha()
    ev = {"rule": f"<volume>/{SMF_DRIVE_NAME}/SHA256SUMS exists and has sha256 {want} (closed smf study "
                  "BACKUP_VERIFICATION.json)", "checked_at": now(), "skipped_by_name": 0, "candidate_volumes": 0,
          "matching_volumes": 0}
    names = sorted(os.listdir(root)) if root.exists() else []
    hit = None
    for n in names:
        if n in SKIP_VOLUMES:
            ev["skipped_by_name"] += 1
            continue
        ev["candidate_volumes"] += 1
        f = root / n / SMF_DRIVE_NAME / "SHA256SUMS"
        try:
            if f.is_file() and sha(f) == want:
                ev["matching_volumes"] += 1
                hit = hit or root / n
        except OSError:
            continue
    ev["mounted"] = hit is not None
    return hit, ev


def diskutil_probe():
    """Counts only (no disk or volume names) from `diskutil list external`."""
    try:
        r = subprocess.run(["diskutil", "list", "external"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"ran": False, "error": type(e).__name__}
    out = r.stdout
    disks = re.findall(r"^/dev/(disk\d+) \(([^)]*)\)", out, re.M)
    return {"ran": r.returncode == 0, "external_entries": len(disks),
            "physical_external_disks": sum(1 for _, kind in disks if "disk image" not in kind),
            "disk_images": sum(1 for _, kind in disks if "disk image" in kind),
            "note": "names withheld; an unrelated read-only installer disk image is expected and never touched"}


def status():
    vol, ev = locate_drive()
    return {"at": now(), "diskutil_list_external": diskutil_probe(), "drive_with_verified_prior_copies": ev,
            "mounted": vol is not None}


# ------------------------------------------------------------------ copying
def inventory(src: Path = None):
    src = Path(src or SRC)
    files = sorted(p for p in src.rglob("*") if p.is_file()
                   and not any(x.endswith(".tmp") or ".tmp" in x for x in p.relative_to(src).parts))
    return files, int(sum(p.stat().st_size for p in files))


def copy_tree(src: Path, rels, dst: Path):
    """Copy rels from src into a NEW dst (refuses an existing one); uncached re-read of every copy."""
    if dst.exists():
        raise SystemExit(f"REFUSED: {dst.name} exists (versioned folders are never reused)")
    sums = {}
    for r in rels:
        s, q = src / r, dst / r
        q.parent.mkdir(parents=True, exist_ok=True)
        sums[r] = sha(s)
        shutil.copy2(s, q)
    reread = {r: sha(dst / r, nocache=True) == h for r, h in sums.items()}
    return sums, reread


# ------------------------------------------------------------------ restore from a copy
@contextlib.contextmanager
def input_from(path: Path):
    """Load D through the pinned osf.data loader, reading the input file at `path` (hash-checked by osf.data)."""
    from osf import data as OD
    old = OD.SRC
    OD.SRC = Path(path)
    try:
        yield
    finally:
        OD.SRC = old


def load_D_from_copy(copy: Path):
    from dpc import data as DD
    from osf import data as OD
    p = copy / INPUT_REL
    if not p.exists():
        raise SystemExit("REFUSED: the copy holds no admitted input file")
    with input_from(p):
        D = OD.load(verify=True, unseal=False)         # sealed; refuses unless the copy's input hashes to the pin
    _, bad = DD.check_against_source(D)
    if bad:
        raise SystemExit(f"REFUSED: roles from the copied input differ from the pinned manifest: {bad}")
    return D


def unit_verify(copy_units: Path, live_units: Path, name: str):
    """COMPLETE.json of the copy equals the live one and every copied file re-hashes (uncached)."""
    dd, ld = copy_units / name, live_units / name
    if not (dd / "COMPLETE.json").exists():
        return {"unit": name, "present_in_copy": False}
    dc = jload(dd / "COMPLETE.json")
    return {"unit": name, "present_in_copy": True,
            "copy_complete_equals_live": (ld / "COMPLETE.json").exists() and dc == jload(ld / "COMPLETE.json"),
            "copy_files_hash_ok_uncached": all(sha(dd / f, nocache=True) == h for f, h in dc["files"].items())}


def cmp_arrays(mine: dict, other, keys):
    out = {}
    for k in keys:
        a, b = np.asarray(mine[k]), np.asarray(other[k])
        if a.shape != b.shape:
            out[k] = {"shape_mismatch": True, "bitwise": False}
        elif np.issubdtype(a.dtype, np.integer):
            out[k] = {"mismatches": int((a != b).sum()), "bitwise": bool(np.array_equal(a, b))}
        else:
            out[k] = {"max_abs_diff": float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()) if a.size else 0.0,
                      "bitwise": bool(a.dtype == b.dtype and np.array_equal(a, b))}
    return out


def restore_teacher(copy: Path, live: Path, name: str, k: int, D):
    """Own forward pass + copied heads on the copy's inputs; compare with copied/live releases and tea units."""
    import joblib
    from dpc import admit as AD
    import torch
    unit = f"rel__s{k}__{name}"
    info = unit_verify(copy / "admitted", live / "admitted", unit)
    if not info.get("present_in_copy") or not (info["copy_complete_equals_live"] and info["copy_files_hash_ok_uncached"]):
        return {**info, "status": "FAIL", "reason": "copy missing or does not verify; nothing unpickled"}, None
    d = copy / "admitted" / unit
    st = torch.load(d / "model.pt", map_location="cpu", weights_only=True)
    H = AD.forward(st, D["X"])
    out = {"row_id": D["row_id"]}
    for i in (0, 1):
        c, P, hard = AD.head_outputs(joblib.load(d / f"head_{i}.joblib"), H[i])
        out.update({f"r{i + 1}": H[i], f"c{i + 1}": c, f"p{i + 1}": P, f"d{i + 1}": hard, f"hard{i + 1}": hard})
    keys = ["r1", "c1", "p1", "hard1", "r2", "c2", "p2", "hard2"]
    z_copy = np.load(d / "release.npz", allow_pickle=False)
    z_live = np.load(live / "admitted" / unit / "release.npz", allow_pickle=False)
    info["rows"] = int(len(D["row_id"]))
    info["row_order_equals_copy_release"] = bool(np.array_equal(z_copy["row_id"], D["row_id"]))
    info["vs_copy_release"] = cmp_arrays(out, z_copy, keys)
    info["vs_live_release"] = cmp_arrays(out, z_live, keys)
    tk = ["row_id", "p1", "p2", "d1", "d2", "c1", "c2", "r1", "r2"]
    for where, root in (("copy", copy), ("live", live)):
        tu = root / "run" / "units" / f"tea__s{k}__{name}" / "teacher.npz"
        if tu.exists():
            info[f"vs_{where}_teacher_unit"] = cmp_arrays(out, np.load(tu, allow_pickle=False), tk)
    allc = [c for kk, cmp in info.items() if kk.startswith("vs_") for c in cmp.values()]
    info["status"] = "PASS" if info["row_order_equals_copy_release"] and all(c["bitwise"] for c in allc) else "FAIL"
    return info, out


def _policy_pair(path: Path, loader=None):
    from dpc import release as RL
    if loader:
        mod, fn = loader.split(":")
        return getattr(importlib.import_module(mod), fn)(path)
    z = json.loads(Path(path).read_text())
    if hasattr(RL, "policy_pair_from_dict"):
        pair = RL.policy_pair_from_dict(z)
    elif z.get("kind") == "dpc.PolicyPair":
        pair = RL.PolicyPair.from_dict(z)
    else:
        pair = RL.load_policy(path)
    if hasattr(pair, "p1"):
        return pair.p1, pair.p2
    return pair[0], pair[1]


def restore_policy(copy: Path, live: Path, name: str, teachers: dict, loader=None, encode=None):
    """Re-encode a copied policy unit from the restored teacher outputs and compare with its saved releases."""
    units, lunits = copy / "run" / "units", live / "run" / "units"
    info = unit_verify(units, lunits, name)
    if not info.get("present_in_copy") or not (info["copy_complete_equals_live"] and info["copy_files_hash_ok_uncached"]):
        return {**info, "status": "FAIL", "reason": "policy unit missing in the copy or does not verify"}
    rec = jload(units / name / "record.json")
    k = int(rec.get("seed"))
    t = (rec.get("cfg") or {}).get("teacher") or rec.get("teacher")
    info.update({"seed": k, "teacher": t, "config": rec.get("config")})
    T = teachers.get((t, k))
    if T is None:
        return {**info, "status": "FAIL", "reason": f"teacher {t} seed {k} was not restored from the copy"}
    p1, p2 = _policy_pair(units / name / "policy.json", loader)
    if encode is None:
        from dpc import release as RL
        encode = RL.encode
    mine = {}
    for i, pol in ((1, p1), (2, p2)):
        tok, q, dec = encode(pol, T[f"p{i}"], T[f"d{i}"])
        mine.update({f"tok{i}": np.asarray(tok), f"q{i}": np.asarray(q), f"hard{i}": np.asarray(dec)})
    keys = ["tok1", "q1", "hard1", "tok2", "q2", "hard2"]
    info["vs_copy_release"] = cmp_arrays(mine, np.load(units / name / "release.npz", allow_pickle=False), keys)
    info["vs_live_release"] = cmp_arrays(mine, np.load(lunits / name / "release.npz", allow_pickle=False), keys)
    info["class_preserved_vs_restored_teacher"] = {f"recipient_{i}": bool(np.array_equal(mine[f"hard{i}"], T[f"d{i}"]))
                                                   for i in (1, 2)}
    info["token_states"] = {f"recipient_{i}": int(len(np.unique(mine[f"tok{i}"]))) for i in (1, 2)}
    ok = all(v["bitwise"] for kk in ("vs_copy_release", "vs_live_release") for v in info[kk].values())
    info["status"] = "PASS" if ok and all(info["class_preserved_vs_restored_teacher"].values()) else "FAIL"
    return info


def restore_reference(copy: Path, live: Path, name: str, D):
    """ref__s{k}__<label>: rebuild from the copied admitted LEACE/FARE artifacts and compare with the reference unit."""
    from dpc import admit as AD
    m = re.match(r"ref__s(\d)__(E|F0|F)$", name)
    if not m:
        return {"unit": name, "status": "FAIL", "reason": "not a reference unit name"}
    k, lab = int(m.group(1)), m.group(2)
    units = AD.resolve()
    fit = D["idx"]["OSF_DEFENSE_FIT"]
    info = {"unit": name, "label": lab, "seed": k}
    adm = copy / "admitted"
    if lab == "E":
        chk = AD.admit_leace(units[("E", k)][0], D, fit, units[("U", k)][0], root=adm)
        info["admitted_rebuild"] = {"all_bitwise": chk["all_bitwise"], "pass": chk["pass"]}
    else:
        fp = AD.fit_row_fingerprint(np.ascontiguousarray(D["X"][fit].astype(np.float64)))
        res = [AD.admit_fare(e, D, fit, fp, root=adm, fare_root=adm / "fare_cache")[0] for e in units[(lab, k)]]
        info["admitted_rebuild"] = {"all_bitwise": all(r["all_bitwise"] for r in res), "pass": all(r["pass"] for r in res)}
    ru = unit_verify(copy / "run" / "units", live / "run" / "units", name)
    info.update(ru)
    same = True
    if ru.get("present_in_copy"):
        z = np.load(copy / "run" / "units" / name / "reference.npz", allow_pickle=False)
        mine = admitted_reference_arrays(adm, lab, k)
        keys = [x for x in z.files if x in mine]
        info["reference_unit_vs_rebuilt_admitted_release"] = cmp_arrays(mine, z, keys)
        same = bool(keys) and all(v["bitwise"] for v in info["reference_unit_vs_rebuilt_admitted_release"].values())
    info["status"] = "PASS" if info["admitted_rebuild"]["pass"] and same and (not ru.get("present_in_copy") or (
        ru["copy_complete_equals_live"] and ru["copy_files_hash_ok_uncached"])) else "FAIL"
    return info


def admitted_reference_arrays(adm: Path, lab: str, k: int):
    """dpc.admit.reference() layout, read from an admitted directory (here: the copy's)."""
    if lab == "E":
        z = np.load(adm / f"lc__s{k}__E" / "release.npz", allow_pickle=False)
        out = {"row_id": z["row_id"]}
        for i in (1, 2):
            out.update({f"c{i}": z[f"c{i}"], f"p{i}": z[f"p{i}"], f"d{i}": z[f"hard{i}"], f"r{i}": z[f"r{i}"]})
        return out
    tag = "c1" if lab == "F" else "Z1"
    out = {}
    for i in (0, 1):
        z = np.load(adm / f"fare__s{k}__p{i}__{tag}" / "release.npz", allow_pickle=False)
        j = i + 1
        out.update({"row_id": z["row_id"], f"c{j}": z["c"], f"p{j}": z["p"], f"d{j}": z["hard"], f"r{j}": z["r"],
                    f"cells{j}": z["cells"]})
    return out


def restore_attacker(copy: Path, live: Path, spec: dict, D):
    if not spec or not spec.get("fn"):
        return {"status": "PENDING", "reason": "no attacker refit entry point in targets.json (set after the "
                                               "EVALUATION_LOCK; see the module docstring)"}
    mod, fn = spec["fn"].split(":")
    f = getattr(importlib.import_module(mod), fn)
    P = np.asarray(f(units_root=copy / "run" / "units", D=D, **(spec.get("kwargs") or {})), dtype=np.float64)
    sv = spec["saved"]
    saved = {w: np.asarray(np.load(root / "run" / "units" / sv["unit"] / sv.get("file", "preds.npz"),
                                   allow_pickle=False)[sv["key"]], dtype=np.float64) for w, root in (("copy", copy),
                                                                                                   ("live", live))}
    tol = float(spec.get("tolerance", 0.0))
    d = {w: (float(np.abs(P - s).max()) if P.shape == s.shape else float("inf")) for w, s in saved.items()}
    return {"status": "PASS" if all(v <= tol for v in d.values()) else "FAIL", "entry_point": spec["fn"],
            "saved": {k: sv[k] for k in ("unit", "key") if k in sv}, "source": "refit from the copied release on "
            "AUDIT_FIT (attacker labels: SEX of AUDIT_FIT only); assessment predictions compared, no assessment label "
            "read", "max_abs_diff_vs_saved_copy": d["copy"], "max_abs_diff_vs_saved_live": d["live"], "tolerance": tol,
            "rows_compared": int(P.shape[0]) if P.ndim else 0}


def restore_all(copy: Path, live: Path, targets: dict, seed: int):
    t0, c0 = time.time(), time.process_time()
    D = load_D_from_copy(copy)
    checks, teachers = {}, {}
    seeds = {seed}
    for lab, u in (targets.get("policies") or {}).items():
        m = re.match(r"(?:pol|tea|ref)__s(\d)__", u)
        if m:
            seeds.add(int(m.group(1)))
    for k in sorted(seeds):
        for t in TEACHERS:
            info, out = restore_teacher(copy, live, t, k, D)
            checks[f"teacher {t} (seed {k})"] = info
            if out is not None and info["status"] == "PASS":
                teachers[(t, k)] = out
    for lab, u in (targets.get("policies") or {}).items():
        if u.startswith("pol__"):
            checks[lab] = restore_policy(copy, live, u, teachers, targets.get("policy_loader"))
        elif u.startswith("tea__"):
            m = re.match(r"tea__s(\d)__(.+)$", u)
            k, t = int(m.group(1)), m.group(2)
            checks[lab] = {"unit": u, "status": checks.get(f"teacher {t} (seed {k})", {}).get("status", "FAIL"),
                           "rebuild": f"teacher {t} seed {k} restored above (own forward pass from the copy)"}
        elif u.startswith("ref__"):
            checks[lab] = restore_reference(copy, live, u, D)
        else:
            checks[lab] = {"unit": u, "status": "FAIL", "reason": "unknown unit kind"}
    checks["final attacker"] = restore_attacker(copy, live, targets.get("attacker"), D)
    del D
    return checks, {"wall_s": round(time.time() - t0, 1), "cpu_s": round(time.process_time() - c0, 1)}


# ------------------------------------------------------------------ 1. new-study backup
def load_targets(targets_file):
    if not targets_file:
        return {"policies": {}, "note": "no targets.json: teachers only; policies and attacker PENDING"}
    t = jload(targets_file)
    for lab, u in (t.get("policies") or {}).items():
        if not re.match(r"^(pol|tea|ref)__s\d__[A-Za-z0-9_.\-]+$", u):
            raise SystemExit(f"REFUSED: target {lab!r} is not a dpc unit name")
    return t


def backup(dest=None, targets_file=None, seed=1, dry_run=False, src=None, cache=None, volumes_root=None, out_pkg=None):
    src, cache, out_pkg = Path(src or SRC), Path(cache or CACHE), Path(out_pkg or PKG)
    files, nbytes = inventory(src)
    targets = load_targets(targets_file)
    seed = int(targets.get("seed", seed))
    if dest:
        vol, how, ev = Path(dest), "--dest", {"mounted": True, "selected_by": "--dest"}
    else:
        vol, ev = locate_drive(volumes_root)
        how = "volume holding the verified smf copy (content match)"
    same_device = vol is None
    date = time.strftime("%Y%m%d", time.gmtime())
    parent = cache if same_device else vol
    root = versioned(parent, (LOCAL_FOLDER if same_device else DRIVE_FOLDER).format(date=date))
    place = "<PRIVATE_CACHE>" if same_device else "<DRIVE_ROOT>"
    plan = {"schema": "dpc-closeout-plan-v1", "at": now(), "source": "<PRIVATE_CACHE>/dpc_v1", "files": len(files),
            "bytes": nbytes, "drive_mounted": not same_device, "drive_selected_by": None if same_device else how,
            "destination": f"{place}/{root.name}/dpc_v1", "same_device": same_device, "restore_seed": seed,
            "targets": targets, "drive_detection": ev, "pending_if_absent": PENDING}
    if dry_run:
        plan["restore_rehearsal_read_only_on_live_store"] = rehearse(src, targets, seed)
        print(scrub(json.dumps(plan, indent=1, default=str)))
        return plan
    rels = [str(p.relative_to(src)) for p in files]
    copy = root / "dpc_v1"
    sums, reread = copy_tree(src, rels, copy)
    (root / "SHA256SUMS").write_text("".join(f"{h}  dpc_v1/{r}\n" for r, h in sums.items()))
    checks, cost = restore_all(copy, src, targets, seed)
    statuses = {k: v.get("status") for k, v in checks.items()}
    restored_ok = all(s == "PASS" for k, s in statuses.items() if k != "final attacker") and \
        statuses.get("final attacker") in ("PASS", "PENDING")
    bv = {"schema": "dpc-backup-verification-v1", "written_at": now(),
          "status": ("LOCAL_SAME_DEVICE_COPY_VERIFIED_OFF_DEVICE_BACKUP_PENDING" if same_device else
                     "DRIVE_COPY_VERIFIED"),
          "off_device_backup": "PENDING (external drive not mounted)" if same_device else "DONE",
          "source": "<PRIVATE_CACHE>/dpc_v1", "destination": f"{place}/{root.name}/dpc_v1", "files": len(files),
          "bytes": nbytes, "copied_now": len(sums), "uncached_readback_match": int(sum(reread.values())),
          "SHA256SUMS_sha256": sha(root / "SHA256SUMS"),
          "read_back": "every copied file re-read with F_NOCACHE (uncached read; not a physical cold-disk read)",
          "restore_kind": ("restore test from a SAME-DEVICE copy (not a drive restore; proves restorability, not "
                           "off-device custody)" if same_device else "restore from the drive copy alone"),
          "restore_seed": seed, "restore_targets": targets.get("policies"), "restore_checks": checks,
          "restore_statuses": statuses, "restore_all_pass": bool(restored_ok), "restore_cost": cost,
          "drive_detection": ev, "deleted": "nothing"}
    if same_device:
        bv["custody_gap"] = ("The external drive was not mounted: this copy is on the SAME device. Pending: " +
                             PENDING["new_study_backup_and_restore"] + " ; and predecessor custody: " +
                             PENDING["osf_backup_and_predecessor_restore"])
        bv["pending"] = PENDING
    ri = {"schema": "dpc-restore-index-v1", "private_local": "<PRIVATE_CACHE>/dpc_v1",
          "copy": f"{place}/{root.name}/dpc_v1" + (" (same device; off-device copy PENDING)" if same_device else ""),
          "checksums": f"{place}/{root.name}/SHA256SUMS",
          "layout": {"admitted/<unit>/": "admitted osf teacher/reference units (ADMISSION.json), hash-complete",
                     "admitted/fare_cache/<uid>/": "official FARE trees of the admitted F/F0 units",
                     "admitted/inputs/adult_jcv.npz": "the pinned 83-column input (sha256 e0d9e54a...5f12)",
                     "run/units/<unit>/": "atomic study units: files + record.json + COMPLETE.json"},
          "restore": [f"copy {place}/{root.name}/dpc_v1 to <PRIVATE_CACHE>/dpc_v1",
                      "verify: shasum -a 256 -c SHA256SUMS (inside the copy's parent folder)",
                      "verify admission: shasum -a 256 -c SHA256SUMS (inside dpc_v1/admitted)",
                      "then follow QUICKSTART.md"],
          "restore_statuses": statuses}
    write_public(out_pkg / "BACKUP_VERIFICATION.json", bv)
    write_public(out_pkg / "RESTORE_INDEX.json", ri)
    (root / "BACKUP_RECORD.json").write_text(json.dumps(bv, indent=1, default=str) + "\n")
    print(json.dumps({k: bv[k] for k in ("status", "files", "uncached_readback_match", "restore_statuses")}, indent=1))
    return bv


def rehearse(src: Path, targets: dict, seed: int):
    """Dry-run: run the restore checks read-only with the live store standing in for the copy (nothing written)."""
    try:
        checks, cost = restore_all(src, src, targets, seed)
    except SystemExit as e:
        return {"status": "NOT_RUN", "reason": scrub_safe(e)}
    return {"note": "live store used as the 'copy' (no copy exists in a dry run); this rehearses code paths only",
            "statuses": {k: v.get("status") for k, v in checks.items()}, "cost": cost}


# ------------------------------------------------------------------ 2. predecessor custody (redirected osf routines)
def redirected(OC, out_dir: Path):
    """Rebind osf.closeout's public outputs into out_dir; refuse any other write under the closed osf directory."""
    out_dir = Path(out_dir)
    pkg = Path(OC.PKG).resolve()
    remap = {pkg / "BACKUP_VERIFICATION.json": out_dir / "osf_BACKUP_VERIFICATION.json",
             pkg / "RESTORE_INDEX.json": out_dir / "osf_RESTORE_INDEX.json",
             pkg / "PREDECESSOR_CUSTODY_REPAIR.json": out_dir / "PREDECESSOR_CUSTODY_REPAIR.json"}
    orig = OC.write_public
    writes = []

    def write(path, obj):
        p = Path(path).resolve()
        tgt = remap.get(p)
        if tgt is None and (p == pkg or pkg in p.parents):
            raise SystemExit("REFUSED: unredirected write into the closed osf results directory")
        tgt = tgt or p
        writes.append(str(tgt.relative_to(WT)) if WT in tgt.parents else "<outside worktree>")
        orig(tgt, obj)

    saved = {"write_public": OC.write_public, "OUT_REPAIR": OC.OUT_REPAIR, "REPLAY_OUT": OC.REPLAY_OUT}
    OC.write_public = write
    OC.OUT_REPAIR = out_dir / "PREDECESSOR_CUSTODY_REPAIR.json"
    OC.REPLAY_OUT = out_dir / "INDEPENDENT_VERIFICATION_DRIVE_RESTORE.json"
    return saved, writes


def restore_bindings(OC, saved):
    for k, v in saved.items():
        setattr(OC, k, v)


@contextlib.contextmanager
def osf_assessment_sealed_until_dpc_lock():
    """osf.closeout.backup opens osf's assessment to refit its final attacker; keep it sealed (osf's own PENDING path)
    until this study's EVALUATION_LOCK is pushed, so no assessment label is loaded earlier by custody work."""
    from osf import assess as AS
    ok, why = lock_pushed()
    orig = AS.open_assessment
    if not ok:
        def refuse(*a, **k):
            raise SystemExit(f"sealed by dpc custody: {why}")
        AS.open_assessment = refuse
    try:
        yield ok
    finally:
        AS.open_assessment = orig


def closed_trees():
    return {"osf_results": tree_state(OSF_PKG), "smf_results": tree_state(SMF_PKG)}


def write_status(rec, out_dir=PRED_DIR):
    write_public(Path(out_dir) / "STATUS.json", rec)
    adm = PKG / "ADMISSION.json"
    if Path(out_dir) == PRED_DIR and adm.exists():                        # keep ADMISSION.json's custody section current (owned by this role)
        a = jload(adm)
        a["custody"] = {**a.get("custody", {}), "predecessor_custody": rec, "updated_at": now()}
        write_public(adm, a)


def predecessor(restore=False, dry_run=False, volumes_root=None, out_dir=PRED_DIR, OC=None):
    vol, ev = locate_drive(volumes_root)
    base = {"schema": "dpc-predecessor-custody-v1", "at": now(), "drive_detection": ev,
            "diskutil_list_external": diskutil_probe(),
            "routines": ["python -m osf.closeout backup --dest <DRIVE_ROOT>",
                         "python -m osf.closeout predecessor --restore"],
            "redirect": {"osf BACKUP_VERIFICATION.json": f"{REL}/provenance/predecessor_custody/osf_BACKUP_VERIFICATION"
                                                          ".json",
                         "osf RESTORE_INDEX.json": f"{REL}/provenance/predecessor_custody/osf_RESTORE_INDEX.json",
                         "osf PREDECESSOR_CUSTODY_REPAIR.json": f"{REL}/provenance/predecessor_custody/"
                                                                "PREDECESSOR_CUSTODY_REPAIR.json",
                         "smf replay report": f"{REL}/provenance/predecessor_custody/INDEPENDENT_VERIFICATION_DRIVE_"
                                             "RESTORE.json"},
            "closed_results_never_written": ["results/pcrl_online_strength_frontier_v1/",
                                             "results/pcrl_strength_matched_feedback_v1/"]}
    if vol is None:
        rec = {**base, "status": "PENDING", "reason": "external drive with the verified prior copies not mounted",
               "pending_commands": PENDING,
               "gaps": ["osf has no off-device copy (osf BACKUP_VERIFICATION.json: same-device copy only)",
                        "the smf drive restore (seed-1 U, J-F, L-F by the smf verifier) has not been executed",
                        "the two later smf logs are on the same internal disk only (local custody supplement)"]}
        if dry_run:
            print(scrub(json.dumps(rec, indent=1)))
            return rec
        write_status(rec, out_dir)
        print(json.dumps({"status": rec["status"]}, indent=1))
        return rec
    if dry_run:
        rec = {**base, "status": "DRY_RUN", "would_run": base["routines"] if restore else base["routines"][:1]}
        print(scrub(json.dumps(rec, indent=1)))
        return rec
    OC = OC or importlib.import_module("osf.closeout")
    before = closed_trees()
    saved, writes = redirected(OC, out_dir)
    runs = {}
    try:
        with osf_assessment_sealed_until_dpc_lock() as unsealed_ok:
            try:
                bv = OC.backup(dest=str(vol), seed=1)
                runs["osf_backup"] = {"status": bv.get("status"), "files": bv.get("files"),
                                      "uncached_readback_match": bv.get("uncached_readback_match"),
                                      "restore_statuses": {k: v.get("status") for k, v in
                                                           (bv.get("restore_checks") or {}).items()},
                                      "attacker_restore_unsealed": bool(unsealed_ok)}
            except SystemExit as e:
                runs["osf_backup"] = {"status": "FAILED", "reason": scrub_safe(e)}
            if restore:
                try:
                    pr = OC.predecessor(restore=True, dry_run=False)
                    runs["osf_predecessor_restore"] = {"status": pr.get("status"),
                                                       "unresolved_gap": pr.get("unresolved_gap")}
                except SystemExit as e:
                    runs["osf_predecessor_restore"] = {"status": "FAILED", "reason": scrub_safe(e)}
    finally:
        restore_bindings(OC, saved)
    after = closed_trees()
    rec = {**base, "status": "RUN", "runs": runs, "redirected_writes": writes,
           "closed_results_unchanged": before == after, "closed_results_state": after}
    if before != after:
        rec["status"] = "FAILED_CLOSED_RESULTS_CHANGED"
    write_status(rec, out_dir)
    print(json.dumps({"status": rec["status"], "runs": {k: v.get("status") for k, v in runs.items()}}, indent=1))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=("backup", "predecessor", "status"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--restore", action="store_true", help="predecessor: also run osf.closeout predecessor --restore")
    ap.add_argument("--dest", default=None)
    ap.add_argument("--targets", default=None)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("REFUSED: OMP_NUM_THREADS must be 1")
    if a.job == "status":
        print(json.dumps(status(), indent=1))
    elif a.job == "backup":
        backup(a.dest, a.targets, a.seed, a.dry_run)
    else:
        predecessor(a.restore, a.dry_run)


if __name__ == "__main__":
    main()
