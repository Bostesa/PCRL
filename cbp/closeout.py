"""Custody closeout for the confidence-budgeted privacy study (cbp; owner role F, independent verifier and custody).

Adapted from qpc/closeout.py at the source tip d0c8a45 (IMPORTED, not edited: the drive probe by content, the
versioned copy with uncached read-back, verify_sums, the teacher restore by own forward pass, the policy re-encode and
deployment from the copy, the attacker restore and the redirected dpc / osf routines are the pinned qpc and dpc code).
New here: the cbp store, the bundled pinned input (the cbp store holds no copy of adult_jcv.npz, so a restore from the
copy alone needs it beside the store), the cbp EVALUATION_LOCK seal, the PENDING qpc custody job (which itself carries the
pending dpc and osf / smf custody) run in-process with every public receipt redirected into THIS study's provenance/,
and the job sequencing.

The drive is identified by CONTENT, never by a volume name: a mounted volume qualifies iff
<volume>/private_smf_v1_20261005/SHA256SUMS hashes to the value the closed smf study recorded in its
BACKUP_VERIFICATION.json (read at the pinned dpc evidence commit). The boot volume and the unrelated read-only installer
image "BackgroundSyncService Setup" are skipped by name BEFORE any filesystem call, so nothing on that image is ever
touched. Volume names are never recorded.

Jobs (every job has --dry-run, which writes nothing; `--plan-only` with --dry-run also skips the read-only restore
rehearsal, i.e. loads no data). Every job that loads data or restores runs under the shared semaphore:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m cbp.sema
        --label F:closeout -- env OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m cbp.closeout <job> [...]

    status                                        drive probe, lock state, store self-containment (writes nothing)
    qpc-custody [--qpc-targets T] [--dry-run]     the source documented `python -m qpc.closeout all --targets
                                                  <PRIVATE_CACHE>/qpc_v1/run/closeout_targets.json`, in-process
    backup [--dest D] [--targets T] [--dry-run] [--plan-only]
    all [--targets T] [--qpc-targets T] [--dry-run] [--plan-only]

`all` with the drive present, in order:
  (1) the pending qpc custody (qpc.closeout.run_all from this worktree, whose qpc / dpc / osf / smf code is byte-identical
      to the source tip, verified before the call): the dpc off-device backup, the osf drive copy, the smf drive restore
      and the two later smf logs, and the qpc drive copy + restore from it. Every public receipt is REDIRECTED into
      provenance/qpc_custody/ (dpc_custody/, predecessor_custody/ below it); any other write into a closed results tree
      (qpc, dpc, osf, smf) is refused; the closed trees are hashed before and after and must be identical; osf's
      assessment stays sealed until THIS study's EVALUATION_LOCK is byte-identical on origin;
  (2) this study: <DRIVE_ROOT>/private_cbp_v1_<UTC date>[_vN]/{cbp_v1, dependencies/jcv_v1/inputs/adult_jcv.npz,
      SHA256SUMS} (never reuses a folder);
  (3) every file of every folder created by (1)-(2) and of every prior known private_<study>_v1_<date> folder re-read
      UNCACHED (F_NOCACHE; an uncached read, not a physical cold-disk read, which would need a purge or remount that
      this closeout never performs);
  (4) restore FROM THE cbp COPY ALONE: the U teacher (own forward pass on the copy's input vs the copied / live / pinned
      dpc teacher units), Q, P* or the registered fallback (policy re-encoded from the restored teacher and deployed
      from the copy, bitwise vs the copied and live releases; decision preservation; binding) and one selected attacker;
  (5) receipts: BACKUP_VERIFICATION.json, RESTORE_INDEX.json (this study), provenance/qpc_custody/*, and
      BACKUP_RECORD.json inside each new drive folder.
`all` with the drive absent: the qpc custody receipt records PENDING with the exact command (no redundant qpc
same-device copy: qpc already holds two verified same-device copies); a versioned SAME-DEVICE copy
<PRIVATE_CACHE>/cbp_v1_local_copy_<UTC date>[_vN] verified uncached and restored from, status
LOCAL_SAME_DEVICE_COPY_VERIFIED_OFF_DEVICE_BACKUP_PENDING, with the exact pending commands. A same-device copy is never
called off-device custody or a drive restore. Nothing anywhere is deleted or moved; other studies' stores are only read.

targets.json (lead, after the EVALUATION_LOCK; <PRIVATE_CACHE>/cbp_v1/run/closeout_targets.json):
    {"seed": 1,
     "policies": {"Q (DIRECT-TASK i8o64)": "pol__s1__U_DIRECT-TASK_i8o64",
                  "P* (or: fallback <config>, INELIGIBLE / DESCRIPTIVE_ONLY)": "pol__s1__<config>"},
     "attacker": {"fn": "cbp.<module>:<refit function>", "kwargs": {...},
                  "saved": {"unit": "<outer unit>", "file": "preds.npz", "key": "<array>"}, "tolerance": 0.0}}
Policies are re-encoded with qpc.release and deployed from the copy with qpc.deploy.release (the cbp codes are stored in
the qpc formats); optional "encode" / "policy_loader" entry points (cbp.* or qpc.*) override the first two.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from dpc import closeout as DC
from qpc import closeout as QC

HOME = Path.home()
WT = Path(__file__).resolve().parents[1]
REL = "results/pcrl_confidence_budgeted_privacy_v1"
PKG = WT / REL
PROV = PKG / "provenance"
QPC_CUSTODY_DIR = PROV / "qpc_custody"
CACHE = HOME / "PCRL_eval_cache_private"
SRC = CACHE / "cbp_v1"
QPC_TARGETS = CACHE / "qpc_v1" / "run" / "closeout_targets.json"
CBP_TARGETS = SRC / "run" / "closeout_targets.json"
INPUT = CACHE / "jcv_v1" / "inputs" / "adult_jcv.npz"
INPUT_SHA = "e0d9e54af780f30788ee29cfe6795ec82cbdcadc127b1978c69a3891485d2f12"
DEP_REL = "dependencies/jcv_v1/inputs/adult_jcv.npz"
SOURCE_TIP = "d0c8a45c879d01fb8b736ccc091ec3e2c3e9b351"
BRANCH = "research/pcrl-confidence-budgeted-privacy-v1"
VOLUMES_ROOT = DC.VOLUMES_ROOT
SKIP_VOLUMES = DC.SKIP_VOLUMES
DRIVE_FOLDER = "private_cbp_v1_{date}"
LOCAL_FOLDER = "cbp_v1_local_copy_{date}"
KNOWN_FOLDERS = re.compile(r"^private_(cbp|qpc|dpc|osf|smf)_v1(_custody_supplement)?_\d{8}(_v\d+)?$")
TEACHERS = ("U", "RAW-J_b0.3")
MIN_FREE_GIB = 5
CLOSED_RESULTS = {"qpc_results": WT / "results" / "pcrl_confidence_capacity_v1",
                  "dpc_results": WT / "results" / "pcrl_decision_preserving_compression_v1",
                  "osf_results": WT / "results" / "pcrl_online_strength_frontier_v1",
                  "smf_results": WT / "results" / "pcrl_strength_matched_feedback_v1"}
SOURCE_PACKAGES = ("qpc", "dpc", "osf", "smf")
STATUS_LOCAL = "LOCAL_SAME_DEVICE_COPY_VERIFIED_OFF_DEVICE_BACKUP_PENDING"
STATUS_DRIVE = "DRIVE_COPY_VERIFIED"
SEMA = ("OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. <python> -m cbp.sema --label "
        "F:closeout -- env OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m cbp.closeout")
PENDING = {
    "everything_when_the_drive_is_mounted": f"{SEMA} all --targets <PRIVATE_CACHE>/cbp_v1/run/closeout_targets.json   "
                                            "(finds <DRIVE_ROOT> by content; runs the qpc custody, the cbp drive copy, "
                                            "the uncached re-read and the restores, in order)",
    "cbp_off_device_backup_and_restore": f"{SEMA} backup --dest <DRIVE_ROOT> --targets "
                                         "<PRIVATE_CACHE>/cbp_v1/run/closeout_targets.json",
    "qpc_dpc_osf_smf_pending_custody": f"{SEMA} qpc-custody   (the source documented `python -m qpc.closeout all "
                                       "--targets <PRIVATE_CACHE>/qpc_v1/run/closeout_targets.json`, in-process from "
                                       "this worktree, receipts redirected into " + REL + "/provenance/qpc_custody/)",
    "equivalent_in_the_source_worktree_NOT_recommended": "cd <SOURCE_WORKTREE> && OMP_NUM_THREADS=1 PYTHONPATH=. "
                                                          "<python> -m qpc.closeout all --targets <PRIVATE_CACHE>/qpc_v1"
                                                          "/run/closeout_targets.json   (writes its receipts into the "
                                                          "closed qpc results; use the redirected form above)"}

now = DC.now
sha = DC.sha
jload = DC.jload
scrub = DC.scrub
scrub_safe = DC.scrub_safe
write_public = DC.write_public
versioned = DC.versioned
tree_state = DC.tree_state
unit_verify = DC.unit_verify
diskutil_probe = DC.diskutil_probe
verify_sums = QC.verify_sums
free_gib = QC.free_gib
top_folders = QC.top_folders


def _rel(p):
    try:
        return str(Path(p).relative_to(WT))
    except ValueError:
        return "<outside worktree>"


# ------------------------------------------------------------------ drive detection, lock seal, code identity
def smf_marker_sha():
    return QC.smf_marker_sha()


def locate_drive(volumes_root=None, want=None):
    """(volume, evidence) by content (skipped names checked before any filesystem call; names never recorded)."""
    return DC.locate_drive(Path(volumes_root or VOLUMES_ROOT), want or smf_marker_sha())


def cbp_lock_pushed():
    """(ok, reason): this study's EVALUATION_LOCK.json committed and byte-identical on origin (fetched now)."""
    lock = PKG / "EVALUATION_LOCK.json"
    rel = f"{REL}/EVALUATION_LOCK.json"
    if not lock.exists():
        return False, "EVALUATION_LOCK.json does not exist"
    subprocess.run(["git", "-C", str(WT), "fetch", "-q", "origin", BRANCH], capture_output=True)
    r = subprocess.run(["git", "-C", str(WT), "show", f"origin/{BRANCH}:{rel}"], capture_output=True)
    if r.returncode != 0:
        return False, "EVALUATION_LOCK.json is not on origin"
    if r.stdout != lock.read_bytes():
        return False, "local EVALUATION_LOCK.json differs from origin"
    return True, "EVALUATION_LOCK.json byte-identical on origin"


def source_code_identity():
    """Every module of the pinned packages in this worktree is byte-identical to its blob at the source tip, so the
    in-process run IS the source documented command."""
    diff = []
    n = 0
    for top in SOURCE_PACKAGES:
        for p in sorted((WT / top).rglob("*.py")):
            rel = str(p.relative_to(WT))
            r = subprocess.run(["git", "-C", str(WT), "show", f"{SOURCE_TIP}:{rel}"], capture_output=True)
            n += 1
            if r.returncode != 0 or r.stdout != p.read_bytes():
                diff.append(rel)
    return {"files": n, "differing_from_source_tip": diff, "identical": not diff and n > 0}


def uncached_read_capability(probe: Path):
    """Whether F_NOCACHE reads work on this filesystem (recorded); a physical cold-disk read is never claimed."""
    try:
        fd = os.open(probe, os.O_RDONLY)
        try:
            fcntl.fcntl(fd, 48, 1)
            os.read(fd, 1)
        finally:
            os.close(fd)
        return {"F_NOCACHE_uncached_read": "available", "physical_cold_read": "not performed (would need a cache purge "
                "or a remount of the volume; not authorised for custody work)"}
    except OSError as e:
        return {"F_NOCACHE_uncached_read": f"unavailable ({type(e).__name__})", "physical_cold_read": "not performed"}


def status(volumes_root=None):
    vol, ev = locate_drive(volumes_root)
    files, nbytes = DC.inventory(SRC) if SRC.exists() else ([], 0)
    return {"at": now(), "diskutil_list_external": diskutil_probe(), "drive_with_verified_prior_copies": ev,
            "mounted": vol is not None, "cbp_evaluation_lock_pushed": cbp_lock_pushed()[0],
            "store": {"files": len(files), "bytes": nbytes}, "self_contained": self_containment(SRC)}


def self_containment(src: Path):
    """What a restore from the copy alone needs: admitted teachers, deployment inputs, the pinned input (bundled)."""
    need = [f"admitted/rel__s{k}__U/{f}" for k in (0, 1, 2) for f in ("model.pt", "head_0.joblib", "head_1.joblib",
                                                                        "release.npz", "COMPLETE.json")]
    need += ["inputs/deploy_input.npz", "inputs/schema.json"] + [f"run/units/tea__s{k}__U/teacher.npz" for k in (0, 1, 2)]
    missing = [r for r in need if not (Path(src) / r).is_file()]
    return {"required_present": len(need) - len(missing), "required": len(need), "missing": missing,
            "pinned_input_bundled_from": "<PRIVATE_CACHE>/jcv_v1/inputs/adult_jcv.npz", "pinned_input_present":
            INPUT.is_file()}


@contextlib.contextmanager
def osf_assessment_sealed_until_cbp_lock():
    """osf.closeout.backup opens osf's assessment (the same rows as this study's assessment) to refit its final
    attacker: keep it sealed until THIS study's EVALUATION_LOCK is on origin."""
    from osf import assess as AS
    ok, why = cbp_lock_pushed()
    orig = AS.open_assessment
    if not ok:
        def refuse(*a, **k):
            raise SystemExit(f"sealed by cbp custody: {why}")
        AS.open_assessment = refuse
    try:
        yield ok
    finally:
        AS.open_assessment = orig


@contextlib.contextmanager
def redirected_qpc(out_dir: Path):
    """Rebind qpc.closeout's receipt folders into out_dir, its closed-tree set to include the qpc results, the osf seal
    to this study's lock, and refuse any public write (qpc or dpc closeout) into a closed results tree."""
    out_dir = Path(out_dir)
    closed = [Path(p).resolve() for p in CLOSED_RESULTS.values()]
    writes = []
    orig = {"QC": {k: getattr(QC, k) for k in ("DPC_DIR", "PRED_DIR", "PKG", "CLOSED_RESULTS", "write_public",
                                             "osf_assessment_sealed_until_qpc_lock")},
            "DC": {"write_public": DC.write_public}}
    base = orig["DC"]["write_public"]

    def guarded(path, obj):
        p = Path(path).resolve()
        if any(p == c or c in p.parents for c in closed):
            raise SystemExit("REFUSED: unredirected write into a closed results tree")
        writes.append(_rel(p))
        base(p, obj)
    QC.DPC_DIR = out_dir / "dpc_custody"
    QC.PRED_DIR = out_dir / "predecessor_custody"
    QC.PKG = out_dir
    QC.CLOSED_RESULTS = dict(CLOSED_RESULTS)
    QC.write_public = guarded
    DC.write_public = guarded
    QC.osf_assessment_sealed_until_qpc_lock = osf_assessment_sealed_until_cbp_lock
    try:
        yield writes
    finally:
        for k, v in orig["QC"].items():
            setattr(QC, k, v)
        DC.write_public = orig["DC"]["write_public"]


def closed_trees():
    return {k: tree_state(v) for k, v in CLOSED_RESULTS.items()}


# ------------------------------------------------------------------ (1) the pending qpc / dpc / osf / smf custody
def qpc_custody(dry_run=False, volumes_root=None, out_dir=None, qpc_targets=None, QCmod=None):
    out_dir = Path(out_dir or QPC_CUSTODY_DIR)
    vol, ev = locate_drive(volumes_root)
    qt = Path(qpc_targets or QPC_TARGETS)
    base = {"schema": "cbp-qpc-custody-v1", "at": now(), "drive_detection": ev,
            "documented_command": "OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m qpc.closeout all --targets "
                                  "<PRIVATE_CACHE>/qpc_v1/run/closeout_targets.json",
            "how": "qpc.closeout.run_all called in-process from this worktree (qpc / dpc / osf / smf modules byte-identical "
                   "to the source tip d0c8a45; checked before the call); receipts redirected",
            "covers": ["dpc off-device backup (python -m dpc.closeout backup --dest <DRIVE_ROOT> --targets "
                       "<PRIVATE_CACHE>/dpc_v1/run/closeout_targets.json)",
                       "osf drive copy + smf drive restore + the two later smf logs (python -m osf.closeout backup "
                       "--dest <DRIVE_ROOT>; python -m osf.closeout predecessor --restore)",
                       "qpc drive copy and restore from it (U teacher, Q*, privacy/control code, attacker)"],
            "redirect": {"qpc BACKUP_VERIFICATION.json / RESTORE_INDEX.json": f"{REL}/provenance/qpc_custody/",
                         "dpc custody receipts": f"{REL}/provenance/qpc_custody/dpc_custody/",
                         "osf / smf predecessor receipts": f"{REL}/provenance/qpc_custody/predecessor_custody/"},
            "closed_results_never_written": [_rel(p) + "/" for p in CLOSED_RESULTS.values()],
            "historical_status": "qpc, dpc and osf off-device custody PENDING at their closeouts (drive absent); the "
                                 "qpc closeout's drive steps were dry-run only and are not claimed as executed",
            "qpc_targets_present": qt.exists()}
    if vol is None:
        rec = {**base, "status": "PENDING", "reason": "external drive with the verified prior copies not mounted",
               "pending_command": PENDING["qpc_dpc_osf_smf_pending_custody"],
               "gaps": ["qpc: same-device copies only (qpc_v1_local_copy_20261006, _v2); off-device PENDING",
                        "dpc: same-device copy only; off-device PENDING",
                        "osf: no off-device copy; the smf drive restore and the two later smf logs PENDING"],
               "note": "no redundant qpc same-device copy is made: qpc already holds two verified same-device copies"}
        if dry_run:
            print(scrub(json.dumps(rec, indent=1)))
        else:
            write_public(out_dir / "STATUS.json", rec)
        return rec
    ident = source_code_identity()
    if not ident["identical"]:
        rec = {**base, "status": "REFUSED", "reason": "pinned package differs from the source tip",
               "code_identity": ident}
        if not dry_run:
            write_public(out_dir / "STATUS.json", rec)
        return rec
    if dry_run:
        rec = {**base, "status": "DRY_RUN", "would_run": base["documented_command"], "code_identity": ident}
        print(scrub(json.dumps(rec, indent=1)))
        return rec
    QCm = QCmod or QC
    before = closed_trees()
    runs = {}
    with redirected_qpc(out_dir) as writes:
        try:
            bv = QCm.run_all(targets_file=str(qt) if qt.exists() else None, volumes_root=volumes_root, out_pkg=out_dir)
            runs["qpc_run_all"] = {"status": bv.get("status"), "files": bv.get("files"),
                                   "uncached_readback_match": bv.get("uncached_readback_match"),
                                   "required_restores": bv.get("required_restores"),
                                   "restore_all_pass": bv.get("restore_all_pass"),
                                   "components": {k: (v or {}).get("status") for k, v in (bv.get("components") or {}).items()
                                                  if isinstance(v, dict)},
                                   "uncached_reread_all_pass": (bv.get("uncached_reread_of_drive_folders") or {}).get(
                                       "all_pass")}
        except SystemExit as e:
            runs["qpc_run_all"] = {"status": "FAILED", "reason": scrub_safe(e)}
    after = closed_trees()
    ok_run = runs["qpc_run_all"].get("status") in (STATUS_DRIVE,)
    rec = {**base, "status": ("RUN" if ok_run else "RUN_WITH_FAILURES") if before == after else
           "FAILED_CLOSED_RESULTS_CHANGED", "runs": runs, "redirected_writes": writes, "code_identity": ident,
           "closed_results_unchanged": before == after, "closed_results_state": after}
    write_public(out_dir / "STATUS.json", rec)
    return rec


# ------------------------------------------------------------------ (2) this study's copy
def copy_study(src: Path, root: Path, deps=None, retries=2):
    """Copy every file of src into root/<src.name> and each dependency into root/<rel> (a NEW folder; never reused),
    write SHA256SUMS of the copied bytes and re-read every copy uncached. A live append-only file that changes while it
    is copied is re-copied (at most `retries` times) and, if still changing, recorded (never silently accepted)."""
    files, nbytes = DC.inventory(src)
    if root.exists():
        raise SystemExit(f"REFUSED: {root.name} exists (versioned folders are never reused)")
    jobs = [(p, f"{src.name}/{p.relative_to(src)}") for p in files] + [(Path(s), r) for r, s in (deps or {}).items()]
    sums, stable, changed = {}, {}, []
    for s_, rel in jobs:
        q = root / rel
        q.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(retries + 1):
            h0 = sha(s_)
            shutil.copy2(s_, q)
            hq = sha(q, nocache=True)
            h1 = sha(s_)
            if h0 == h1 == hq:
                break
        else:
            changed.append(rel)
        sums[rel] = hq
        stable[rel] = hq == h0 == h1
    (root / "SHA256SUMS").write_text("".join(f"{h}  {r}\n" for r, h in sums.items()))
    final = {r: sha(root / r, nocache=True) == h for r, h in sums.items()}
    dep_bytes = sum(Path(s).stat().st_size for s in (deps or {}).values())
    return root / src.name, {"files": len(jobs), "store_files": len(files), "dependency_files": len(deps or {}),
                             "bytes": nbytes + dep_bytes, "copied_now": len(sums),
                             "uncached_readback_match": int(sum(final.values())),
                             "source_stable_during_copy": int(sum(stable.values())),
                             "live_files_changed_during_copy": changed,
                             "SHA256SUMS_sha256": sha(root / "SHA256SUMS")}


def load_D_from_input(path: Path):
    """Sealed D through the pinned loader reading ONLY `path` (the copy's bundled input); every qpc role check re-run."""
    from dpc import data as DD
    from osf import data as OD
    from qpc import data as QD
    p = Path(path)
    if not p.is_file():
        raise SystemExit("REFUSED: the copy holds no bundled input file")
    if sha(p, nocache=True) != INPUT_SHA:
        raise SystemExit("REFUSED: the copy's input file fails its pinned hash")
    with DC.input_from(p):
        D = OD.load(verify=True, unseal=False)
    _, b1 = DD.check_against_source(D)
    _, b2 = QD.check_against_dpc(D)
    _, b3 = QD.check_counts_and_isolation(D)
    if b1 or b2 or b3:
        raise SystemExit(f"REFUSED: roles from the copied input differ from the pinned manifests: {b1 + b2 + b3}")
    return D


def _entry(spec):
    if not spec:
        return None
    import importlib
    mod, fn = spec.split(":")
    return getattr(importlib.import_module(mod), fn)


def load_targets(targets_file):
    if not targets_file or not Path(targets_file).exists():
        return {"policies": {}, "note": "no targets.json: U teacher only; Q, P* / fallback and attacker PENDING"}
    t = jload(targets_file)
    for lab, u in (t.get("policies") or {}).items():
        if not re.match(r"^(pol|tea)__s\d__[A-Za-z0-9_.\-]+$", u):
            raise SystemExit(f"REFUSED: target {lab!r} is not a cbp unit name")
    for key in ("encode", "policy_loader"):
        if t.get(key) and not re.match(r"^(cbp|qpc)\.[a-z_]+:[A-Za-z_]+$", t[key]):
            raise SystemExit(f"REFUSED: {key} must be a cbp / qpc entry point")
    if (t.get("attacker") or {}).get("fn") and not re.match(r"^(cbp|qpc)\.[a-z_]+:[A-Za-z_]+$", t["attacker"]["fn"]):
        raise SystemExit("REFUSED: attacker.fn must be a cbp / qpc entry point")
    return t


def restore_all(copy: Path, live: Path, targets: dict, seed: int, input_path: Path):
    """Restore from the copy ALONE: `copy` is the copied cbp_v1 folder and `input_path` the copy's bundled input; the
    live store is only the comparison target."""
    t0, c0 = time.time(), time.process_time()
    copy = Path(copy)
    D = load_D_from_input(Path(input_path))
    checks, teachers = {}, {}
    seeds = {seed}
    for u in (targets.get("policies") or {}).values():
        m = re.match(r"(?:pol|tea)__s(\d)__", u)
        if m:
            seeds.add(int(m.group(1)))
    for k in sorted(seeds):
        for t in TEACHERS:
            if not (copy / "admitted" / f"rel__s{k}__{t}").exists() and t != "U":
                continue
            info, out = QC.restore_teacher(copy, live, t, k, D)
            checks[f"teacher {t} (seed {k})"] = info
            if out is not None and info["status"] == "PASS":
                teachers[(t, k)] = out
    enc, ldr = _entry(targets.get("encode")), _entry(targets.get("policy_loader"))
    for lab, u in (targets.get("policies") or {}).items():
        if u.startswith("pol__"):
            checks[lab] = QC.restore_policy(copy, live, u, teachers, D, encode=enc, loader=ldr)
        else:
            m = re.match(r"tea__s(\d)__(.+)$", u)
            kk, t = int(m.group(1)), m.group(2)
            checks[lab] = {"unit": u, "status": checks.get(f"teacher {t} (seed {kk})", {}).get("status", "FAIL"),
                           "rebuild": f"teacher {t} seed {kk} restored above (own forward pass from the copy)"}
    checks["attacker"] = DC.restore_attacker(copy, live, targets.get("attacker"), D)
    del D
    return checks, {"wall_s": round(time.time() - t0, 1), "cpu_s": round(time.process_time() - c0, 1)}


def required_targets_status(statuses):
    """The four required restore classes (prompt section 15): U teacher, Q, P* or the fallback, a selected attacker."""
    have_u = any(k.startswith("teacher U ") and v == "PASS" for k, v in statuses.items())
    pol = {k: v for k, v in statuses.items() if not k.startswith("teacher ") and k != "attacker"}
    return {"U teacher": "PASS" if have_u else "FAIL",
            "Q": next((v for k, v in pol.items() if k.startswith("Q")), "PENDING (no Q target)"),
            "P* or fallback": next((v for k, v in pol.items() if k.startswith(("P*", "fallback", "P"))),
                                   "PENDING (no P* / fallback target)"),
            "attacker": statuses.get("attacker", "PENDING")}


def rehearse(src: Path, targets: dict, seed: int):
    """Dry run: the restore checks read-only with the live store standing in for the copy and the pinned input read in
    place (nothing written)."""
    try:
        checks, cost = restore_all(src, src, targets, seed, INPUT)
    except SystemExit as e:
        return {"status": "NOT_RUN", "reason": scrub_safe(e)}
    st = {k: v.get("status") for k, v in checks.items()}
    return {"note": "live store used as the 'copy' (no copy exists in a dry run); rehearses code paths only",
            "statuses": st, "required": required_targets_status(st), "cost": cost}


def backup(dest=None, targets_file=None, seed=1, dry_run=False, src=None, cache=None, volumes_root=None, out_pkg=None,
           components=None, detection=None, plan_only=False, deps=None):
    src, cache, out_pkg = Path(src or SRC), Path(cache or CACHE), Path(out_pkg or PKG)
    deps = {DEP_REL: INPUT} if deps is None else deps
    files, nbytes = DC.inventory(src)
    targets = load_targets(targets_file if targets_file is not None else CBP_TARGETS)
    seed = int(targets.get("seed", seed))
    if dest:
        vol, ev = Path(dest), detection or {"mounted": True, "selected_by": "--dest"}
    else:
        vol, ev = locate_drive(volumes_root)
    same_device = vol is None
    date = time.strftime("%Y%m%d", time.gmtime())
    parent = cache if same_device else vol
    root = versioned(parent, (LOCAL_FOLDER if same_device else DRIVE_FOLDER).format(date=date))
    place = "<PRIVATE_CACHE>" if same_device else "<DRIVE_ROOT>"
    dep_bytes = sum(Path(s).stat().st_size for s in deps.values() if Path(s).exists())
    free = free_gib(parent)
    plan = {"schema": "cbp-closeout-plan-v1", "at": now(), "source": "<PRIVATE_CACHE>/cbp_v1", "files": len(files),
            "bytes": nbytes, "dependencies": {r: "<PRIVATE_CACHE>/jcv_v1/inputs/adult_jcv.npz (pinned sha256 " +
                                              INPUT_SHA[:12] + "...)" for r in deps},
            "drive_mounted": not same_device, "destination": f"{place}/{root.name}/cbp_v1", "same_device": same_device,
            "restore_seed": seed, "targets": targets, "drive_detection": ev, "self_contained": self_containment(src),
            "free_gib_before": round(free, 2), "free_gib_after_estimate": round(free - (nbytes + dep_bytes) / 2 ** 30, 2),
            "pending_if_absent": PENDING}
    if dry_run:
        plan["restore_rehearsal_read_only_on_live_store"] = (
            {"status": "SKIPPED (--plan-only: no data loaded, no forward pass)"} if plan_only else rehearse(src, targets, seed))
        print(scrub(json.dumps(plan, indent=1, default=str)))
        return plan
    if free - (nbytes + dep_bytes) / 2 ** 30 < MIN_FREE_GIB:
        raise SystemExit(f"REFUSED: the copy would leave less than {MIN_FREE_GIB} GiB free at the destination")
    for r, s_ in deps.items():
        if sha(s_) != INPUT_SHA:
            raise SystemExit(f"REFUSED: dependency {r} fails its pinned hash")
    copy, cp = copy_study(src, root, deps)
    checks, cost = restore_all(copy, src, targets, seed, root / next(iter(deps)) if deps else INPUT)
    statuses = {k: v.get("status") for k, v in checks.items()}
    req = required_targets_status(statuses)
    restored_ok = all(s == "PASS" for k, s in statuses.items() if k != "attacker") and \
        statuses.get("attacker") in ("PASS", "PENDING")
    bv = {"schema": "cbp-backup-verification-v1", "written_at": now(),
          "status": STATUS_LOCAL if same_device else STATUS_DRIVE,
          "off_device_backup": "PENDING (external drive not mounted)" if same_device else "DONE (this study)",
          "source": "<PRIVATE_CACHE>/cbp_v1", "destination": f"{place}/{root.name}/cbp_v1",
          "bundled_dependency": f"{place}/{root.name}/{DEP_REL}", **cp,
          "read_back": "every copied file re-read with F_NOCACHE (uncached read; not a physical cold-disk read)",
          "read_capability": uncached_read_capability(root / "SHA256SUMS"),
          "restore_kind": ("restore test from a SAME-DEVICE copy (not a drive restore; proves restorability, not "
                           "off-device custody)" if same_device else "restore from the drive copy alone"),
          "restore_seed": seed, "restore_targets": targets.get("policies"), "restore_checks": checks,
          "restore_statuses": statuses, "required_restores": req, "restore_all_pass": bool(restored_ok),
          "restore_cost": cost, "drive_detection": ev, "deleted": "nothing", "moved": "nothing"}
    if same_device:
        bv["custody_gap"] = ("The external drive was not mounted: this copy is on the SAME device and is not "
                             "off-device custody. Pending (exact commands): " + "; ".join(PENDING.values()))
        bv["pending"] = PENDING
    if components:
        bv["components"] = components
    write_public(out_pkg / "BACKUP_VERIFICATION.json", bv)
    write_public(out_pkg / "RESTORE_INDEX.json", restore_index(place, root, statuses, same_device))
    (root / "BACKUP_RECORD.json").write_text(json.dumps(bv, indent=1, default=str) + "\n")
    print(json.dumps({k: bv[k] for k in ("status", "files", "uncached_readback_match", "required_restores")}, indent=1))
    return bv


def restore_index(place, root, statuses, same_device):
    return {"schema": "cbp-restore-index-v1", "written_at": now(), "private_local": "<PRIVATE_CACHE>/cbp_v1",
            "copy": f"{place}/{root.name}/cbp_v1" + (" (same device; off-device copy PENDING)" if same_device else ""),
            "checksums": f"{place}/{root.name}/SHA256SUMS",
            "layout": {"cbp_v1/admitted/rel__s{k}__{t}/": "admitted teachers (model.pt, heads, release; verified copies "
                                                          "of the qpc admitted artifacts) + ADMISSION_RECEIPT.json",
                       "cbp_v1/inputs/": "the authorised 83-column deployment input and schema",
                       "cbp_v1/run/units/<unit>/": "atomic study units: files + record.json + COMPLETE.json",
                       DEP_REL: "the pinned source input (sha256 e0d9e54a...2f12), bundled so the copy restores alone"},
            "restore": [f"copy {place}/{root.name}/cbp_v1 to <PRIVATE_CACHE>/cbp_v1 (only if the live store is lost)",
                        "verify: shasum -a 256 -c SHA256SUMS (inside the copy's folder)",
                        f"re-run the restore checks: {SEMA} backup --dry-run --targets <targets.json>",
                        "then follow QUICKSTART.md"],
            "restore_statuses": statuses, "not_off_device": bool(same_device)}


# ------------------------------------------------------------------ everything, in order
def run_all(targets_file=None, seed=1, dry_run=False, volumes_root=None, src=None, cache=None, out_pkg=None,
            plan_only=False, qpc_targets=None, deps=None):
    vol, ev = locate_drive(volumes_root)
    before = closed_trees()
    folders0 = top_folders(vol)
    comp = {"drive_mounted": vol is not None, "drive_detection": ev}
    if vol is None or dry_run:
        q = qpc_custody(dry_run, volumes_root, qpc_targets=qpc_targets)
        comp["qpc_custody"] = {"status": q.get("status"), "receipt": "provenance/qpc_custody/STATUS.json",
                               "pending_command": q.get("pending_command")}
        return backup(None if vol is None else str(vol), targets_file, seed, dry_run, src, cache, volumes_root, out_pkg,
                      components=comp, detection=ev, plan_only=plan_only, deps=deps)
    q = qpc_custody(False, volumes_root, qpc_targets=qpc_targets)                       # (1)
    comp["qpc_custody"] = {"status": q.get("status"), "receipt": "provenance/qpc_custody/STATUS.json"}
    new_before_cbp = sorted(top_folders(vol) - folders0)
    bv = backup(str(vol), targets_file, seed, False, src, cache, volumes_root, out_pkg, components=comp, detection=ev,
                deps=deps)                                                              # (2) + (4)
    new = sorted(top_folders(vol) - folders0)                                           # (3)
    prior = sorted(n for n in folders0 if KNOWN_FOLDERS.match(n))
    pub = lambda n: n if KNOWN_FOLDERS.match(n) else "<other new folder>"              # noqa: E731
    reread = {pub(n): verify_sums(Path(vol) / n) for n in new}
    prior_reread = {n: verify_sums(Path(vol) / n) for n in prior}
    after = closed_trees()
    bv["uncached_reread_of_drive_folders"] = {
        "created_by_this_closeout": reread, "created_before_the_cbp_copy": len(new_before_cbp),
        "prior_known_backup_folders": prior_reread,
        "all_pass": all(v["pass"] for v in list(reread.values()) + list(prior_reread.values())),
        "rule": "every file listed in each folder's SHA256SUMS re-read with F_NOCACHE (uncached; not a physical cold-disk "
                "read); only folder names of the known private_<study>_v1_<date> form are published"}
    bv["closed_results_unchanged"] = before == after
    if before != after:
        bv["status"] = "FAILED_CLOSED_RESULTS_CHANGED"
    write_public(Path(out_pkg or PKG) / "BACKUP_VERIFICATION.json", bv)
    return bv


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m cbp.closeout")
    ap.add_argument("job", choices=("status", "all", "backup", "qpc-custody"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--plan-only", action="store_true", help="with --dry-run: skip the read-only restore rehearsal")
    ap.add_argument("--dest", default=None)
    ap.add_argument("--targets", default=None)
    ap.add_argument("--qpc-targets", default=None)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("REFUSED: OMP_NUM_THREADS must be 1")
    if a.plan_only and not a.dry_run:
        raise SystemExit("REFUSED: --plan-only requires --dry-run")
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    if a.job == "status":
        print(scrub(json.dumps(status(), indent=1)))
    elif a.job == "all":
        run_all(a.targets, a.seed, a.dry_run, plan_only=a.plan_only, qpc_targets=a.qpc_targets)
    elif a.job == "backup":
        backup(a.dest, a.targets, a.seed, a.dry_run, plan_only=a.plan_only)
    else:
        print(json.dumps({"status": qpc_custody(a.dry_run, qpc_targets=a.qpc_targets)["status"]}))


if __name__ == "__main__":
    main()
