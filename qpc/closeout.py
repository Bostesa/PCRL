"""Custody closeout for the confidence-capacity study (qpc; data/custody owner E). Adapted from dpc/closeout.py at the
source evidence commit 0a7b05a (imported, not edited: drive probe, copying, uncached re-read, teacher restore,
attacker restore and the redirected osf routines are dpc's code; this module adds the qpc policy restore, the qpc
EVALUATION_LOCK seal, the dpc/osf/smf redirection targets under THIS study's provenance/, and the job sequencing).

The drive is identified by CONTENT, never by a volume name: a mounted volume qualifies iff
<volume>/private_smf_v1_20261005/SHA256SUMS hashes to the value the closed smf study recorded in its
BACKUP_VERIFICATION.json (read at the pin). The boot volume and one unrelated read-only installer image are skipped by
name BEFORE any filesystem call, so nothing on that image is ever touched. Volume names are never recorded.

Jobs (every one has --dry-run, which writes nothing; `backup --dry-run` also rehearses the restore checks read-only
against the live store, a heavy real-data check to run under qpc.sema; add --plan-only to skip that rehearsal):

    OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m qpc.closeout status
    OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m qpc.closeout all [--targets <targets.json>] [--dry-run]
    OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m qpc.closeout backup [--dest <DRIVE_ROOT>] [--targets ..]
    OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m qpc.closeout dpc-backup [--dry-run]
    OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m qpc.closeout predecessor [--restore] [--dry-run]

`all` with the drive present, in order:
  (1) dpc off-device backup: the dpc documented command `python -m dpc.closeout backup --dest <DRIVE_ROOT> --targets
      <PRIVATE_CACHE>/dpc_v1/run/closeout_targets.json`, run in-process with its two public receipts REDIRECTED into
      provenance/dpc_custody/ (dpc.closeout.backup out_pkg), so the closed dpc results are never written;
  (2) predecessor repair: the source documented commands `python -m osf.closeout backup --dest <DRIVE_ROOT>` and
      `python -m osf.closeout predecessor --restore` (the osf off-device copy, the pending smf drive restore by the smf
      verifier and the two later smf log copies), in-process with every public write redirected into
      provenance/predecessor_custody/ (dpc.closeout.redirected); osf's attacker restore stays sealed until THIS study's
      EVALUATION_LOCK is pushed;
  (3) this study: <DRIVE_ROOT>/private_qpc_v1_<UTC date>[_vN]/qpc_v1 (never reuses a folder), SHA256SUMS;
  (4) every hash of every folder created by (1)-(3) re-read UNCACHED (F_NOCACHE; not a physical cold-disk read);
  (5) restore FROM THE qpc COPY ALONE: the U teacher (own forward pass on the copy's input vs the copied/live teacher
      units and the pinned dpc teacher unit), Q*, a privacy/control code (policy re-encoded from the restored teacher
      with qpc.release.encode and deployed through qpc.deploy.release from the copy; tokens/probabilities/decisions
      bitwise vs the copied and live releases; decision preservation; binding to the copied model.pt) and one attacker
      (the audit owner's refit entry point, as in dpc);
  (6) separate receipts: BACKUP_VERIFICATION.json and RESTORE_INDEX.json (this study), provenance/dpc_custody/*,
      provenance/predecessor_custody/*, BACKUP_RECORD.json inside each new drive folder. No historical artifact is
      modified: the closed dpc, osf and smf results trees are hashed before and after and must be identical.
`all` with the drive absent: a versioned SAME-DEVICE copy <PRIVATE_CACHE>/qpc_v1_local_copy_<UTC date>[_vN]/qpc_v1,
verified uncached and restored from, status LOCAL_SAME_DEVICE_COPY_VERIFIED_OFF_DEVICE_BACKUP_PENDING with the exact
pending commands. A same-device copy is never called off-device custody or a drive restore.

targets.json (lead, after the EVALUATION_LOCK; <PRIVATE_CACHE>/qpc_v1/run/closeout_targets.json):
    {"seed": 1,
     "policies": {"Q*": "pol__s1__<config>", "privacy/control code": "pol__s1__<config>"},
     "attacker": {"fn": "qpc.<module>:<refit function>", "kwargs": {...},
                  "saved": {"unit": "<unit>", "file": "preds.npz", "key": "<array>"}, "tolerance": 0.0}}
Nothing anywhere is deleted or moved; other studies' private stores are only read.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np

from dpc import closeout as DC
from qpc import data as QD

HOME = Path.home()
WT = QD.WT
REL = QD.REL
PKG = QD.PKG
PROV = PKG / "provenance"
PRED_DIR = PROV / "predecessor_custody"
DPC_DIR = PROV / "dpc_custody"
CACHE = HOME / "PCRL_eval_cache_private"
SRC = CACHE / "qpc_v1"
DPC_SRC = CACHE / "dpc_v1"
DPC_TARGETS = DPC_SRC / "run" / "closeout_targets.json"
DPC_UNITS = DPC_SRC / "run" / "units"
VOLUMES_ROOT = DC.VOLUMES_ROOT
SKIP_VOLUMES = DC.SKIP_VOLUMES
SMF_DRIVE_NAME = DC.SMF_DRIVE_NAME
DRIVE_FOLDER = "private_qpc_v1_{date}"
LOCAL_FOLDER = "qpc_v1_local_copy_{date}"
KNOWN_FOLDERS = re.compile(r"^private_(qpc|dpc|osf|smf)_v1(_custody_supplement)?_\d{8}(_v\d+)?$")
TEACHERS = ("U", "RAW-J_b0.3")
INPUT_REL = DC.INPUT_REL                       # admitted/inputs/adult_jcv.npz (qpc.admit layout)
MIN_FREE_GIB = 5
CLOSED_RESULTS = {"dpc_results": WT / QD.DPC_REL, "osf_results": WT / QD.OSF_REL,
                  "smf_results": WT / "results" / "pcrl_strength_matched_feedback_v1"}
SMF_BV_REL = "results/pcrl_strength_matched_feedback_v1/BACKUP_VERIFICATION.json"
STATUS_LOCAL = "LOCAL_SAME_DEVICE_COPY_VERIFIED_OFF_DEVICE_BACKUP_PENDING"
STATUS_DRIVE = "DRIVE_COPY_VERIFIED"
CMD = "OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m qpc.closeout"
PENDING = {
    "everything_when_the_drive_is_mounted": f"{CMD} all --targets <PRIVATE_CACHE>/qpc_v1/run/closeout_targets.json   "
                                            "(finds <DRIVE_ROOT> by content; runs steps 1-6 in order)",
    "qpc_off_device_backup_and_restore": f"{CMD} backup --dest <DRIVE_ROOT> --targets "
                                         "<PRIVATE_CACHE>/qpc_v1/run/closeout_targets.json",
    "dpc_off_device_backup": f"{CMD} dpc-backup   (runs the dpc documented `python -m dpc.closeout backup --dest "
                             "<DRIVE_ROOT> --targets <PRIVATE_CACHE>/dpc_v1/run/closeout_targets.json` in-process, "
                             f"receipts redirected into {REL}/provenance/dpc_custody/)",
    "osf_smf_predecessor_repair": f"{CMD} predecessor --restore   (runs `python -m osf.closeout backup --dest "
                                  "<DRIVE_ROOT>` and `python -m osf.closeout predecessor --restore` in-process, "
                                  f"receipts redirected into {REL}/provenance/predecessor_custody/; covers the osf "
                                  "drive copy, the smf drive restore and the two later smf logs)"}

now = DC.now
sha = DC.sha
jload = DC.jload
scrub = DC.scrub
scrub_safe = DC.scrub_safe
write_public = DC.write_public
versioned = DC.versioned
tree_state = DC.tree_state
cmp_arrays = DC.cmp_arrays
unit_verify = DC.unit_verify
diskutil_probe = DC.diskutil_probe


def _rel(p):
    try:
        return str(Path(p).relative_to(WT))
    except ValueError:
        return "<outside worktree>"


# ------------------------------------------------------------------ drive detection (by content)
def smf_marker_sha():
    """smf's recorded SHA256SUMS hash, read from the smf receipt as committed at the source evidence commit."""
    return json.loads(QD.pinned_bytes(QD.SOURCE_SHA, SMF_BV_REL))["SHA256SUMS_sha256"]


def locate_drive(volumes_root=None, want=None):
    """(volume, evidence) by content (dpc.closeout.locate_drive; skipped names checked before any filesystem call)."""
    return DC.locate_drive(Path(volumes_root or VOLUMES_ROOT), want or smf_marker_sha())


def status(volumes_root=None):
    vol, ev = locate_drive(volumes_root)
    return {"at": now(), "diskutil_list_external": diskutil_probe(), "drive_with_verified_prior_copies": ev,
            "mounted": vol is not None, "qpc_lock_pushed": QD.evaluation_lock_pushed()[0]}


def free_gib(path: Path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 2 ** 30


def top_folders(vol: Path):
    """Names of top-level folders on the drive (kept private; only KNOWN_FOLDERS names are ever published)."""
    return {p.name for p in Path(vol).iterdir() if p.is_dir()} if vol else set()


# ------------------------------------------------------------------ copies
def inventory(src=None):
    return DC.inventory(Path(src or SRC))


def copy_study(src: Path, root: Path, retries=2):
    """Copy every file of src into root/<src.name> (a NEW folder; never reused), write SHA256SUMS of the copied bytes and
    re-read every copy uncached. Live append-only logs may change while copying: each file is hashed before and after
    its copy; a file whose source changed is re-copied (at most `retries` times) and, if still changing, recorded in
    `live_files_changed_during_copy` with the snapshot hash that SHA256SUMS lists (never silently accepted)."""
    import shutil
    files, nbytes = inventory(src)
    rels = [str(p.relative_to(src)) for p in files]
    copy = root / src.name
    if copy.exists() or root.exists():
        raise SystemExit(f"REFUSED: {root.name} exists (versioned folders are never reused)")
    sums, reread, changed = {}, {}, []
    for r in rels:
        s_, q = src / r, copy / r
        q.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(retries + 1):
            h0 = sha(s_)
            shutil.copy2(s_, q)
            hq = sha(q, nocache=True)
            h1 = sha(s_)
            if h0 == h1 == hq:
                break
        else:
            changed.append(r)
        sums[r] = hq
        reread[r] = (hq == h0 == h1)
    (root / "SHA256SUMS").write_text("".join(f"{h}  {src.name}/{r}\n" for r, h in sums.items()))
    final = {r: sha(copy / r, nocache=True) == h for r, h in sums.items()}
    return copy, {"files": len(files), "bytes": nbytes, "copied_now": len(sums),
                  "uncached_readback_match": int(sum(final.values())),
                  "source_stable_during_copy": int(sum(reread.values())),
                  "live_files_changed_during_copy": changed,
                  "SHA256SUMS_sha256": sha(root / "SHA256SUMS")}


def verify_sums(folder: Path):
    """Re-read every file listed in <folder>/SHA256SUMS uncached; counts only."""
    sums = Path(folder) / "SHA256SUMS"
    if not sums.exists():
        return {"SHA256SUMS_present": False, "entries": 0, "match": 0, "pass": False}
    lines = [ln.split("  ", 1) for ln in sums.read_text().splitlines() if ln.strip()]
    ok = 0
    for h, rel in lines:
        p = Path(folder) / rel
        if p.is_file() and sha(p, nocache=True) == h:
            ok += 1
    return {"SHA256SUMS_present": True, "SHA256SUMS_sha256": sha(sums, nocache=True), "entries": len(lines),
            "match": ok, "pass": ok == len(lines) and len(lines) > 0}


# ------------------------------------------------------------------ restore from a copy
def load_D_from_copy(copy: Path):
    """Sealed D through the pinned loader, reading ONLY the copy's input file; all qpc role checks re-run."""
    from osf import data as OD
    from dpc import data as DD
    p = copy / INPUT_REL
    if not p.exists():
        raise SystemExit("REFUSED: the copy holds no admitted input file")
    if sha(p, nocache=True) != QD.INPUT_SHA:
        raise SystemExit("REFUSED: the copy's input file fails its pinned hash")
    with DC.input_from(p):
        D = OD.load(verify=True, unseal=False)
    _, b1 = DD.check_against_source(D)
    _, b2 = QD.check_against_dpc(D)
    _, b3 = QD.check_counts_and_isolation(D)
    if b1 or b2 or b3:
        raise SystemExit(f"REFUSED: roles from the copied input differ from the pinned manifests: {b1 + b2 + b3}")
    return D


def restore_teacher(copy: Path, live: Path, t: str, k: int, D):
    """dpc.closeout.restore_teacher on the qpc layout + bitwise comparison with the pinned dpc teacher unit."""
    info, out = DC.restore_teacher(copy, live, t, k, D)
    if out is not None:
        tu = DPC_UNITS / f"tea__s{k}__{t}" / "teacher.npz"
        if tu.exists():
            info["vs_pinned_dpc_teacher_unit"] = cmp_arrays(out, np.load(tu, allow_pickle=False),
                                                            ["row_id", "p1", "p2", "d1", "d2", "c1", "c2", "r1", "r2"])
            if not all(c["bitwise"] for c in info["vs_pinned_dpc_teacher_unit"].values()):
                info["status"] = "FAIL"
        d = copy / "admitted" / f"rel__s{k}__{t}"
        info["model_pt_sha256"] = sha(d / "model.pt", nocache=True)
    return info, out


def _unit_teacher_seed(units: Path, name: str):
    rec = jload(units / name / "record.json") if (units / name / "record.json").exists() else {}
    m = re.match(r"pol__s(\d)__(.+)$", name)
    k = rec.get("seed", int(m.group(1)) if m else None)
    t = rec.get("teacher") or (rec.get("cfg") or {}).get("teacher")
    if t is None and m:
        t = "RAW-J_b0.3" if m.group(2).startswith("RAW-J_b0.3_") else "U"
    return t, (int(k) if k is not None else None), rec


def restore_policy(copy: Path, live: Path, name: str, teachers: dict, D, encode=None, loader=None, deploy=True):
    """Re-encode a copied qpc policy unit from the restored teacher; deploy it from the copy; compare bitwise."""
    from qpc import release as RL
    encode = encode or RL.encode
    loader = loader or RL.load_policy
    units, lunits = copy / "run" / "units", live / "run" / "units"
    info = unit_verify(units, lunits, name)
    if not info.get("present_in_copy") or not (info["copy_complete_equals_live"] and info["copy_files_hash_ok_uncached"]):
        return {**info, "status": "FAIL", "reason": "policy unit missing in the copy or does not verify"}
    t, k, rec = _unit_teacher_seed(units, name)
    info.update({"seed": k, "teacher": t, "config": rec.get("config")})
    T = teachers.get((t, k))
    if T is None:
        return {**info, "status": "FAIL", "reason": f"teacher {t} seed {k} was not restored from the copy"}
    pair = loader(units / name / "policy.json")
    p1, p2 = (pair.p1, pair.p2) if hasattr(pair, "p1") else pair
    mine = {}
    for i, pol in ((1, p1), (2, p2)):
        tok, q, dec = encode(pol, T[f"p{i}"], T[f"d{i}"])
        mine.update({f"tok{i}": np.asarray(tok), f"q{i}": np.asarray(q), f"hard{i}": np.asarray(dec)})
    keys = ["tok1", "q1", "hard1", "tok2", "q2", "hard2"]
    info["vs_copy_release"] = cmp_arrays(mine, np.load(units / name / "release.npz", allow_pickle=False), keys)
    info["vs_live_release"] = cmp_arrays(mine, np.load(lunits / name / "release.npz", allow_pickle=False), keys)
    info["decision_preserved_vs_restored_teacher"] = {f"recipient_{i}": bool(np.array_equal(mine[f"hard{i}"],
                                                                                           T[f"d{i}"])) for i in (1, 2)}
    info["token_states_emitted"] = {f"recipient_{i}": int(len(np.unique(mine[f"tok{i}"]))) for i in (1, 2)}
    ok = all(v["bitwise"] for kk in ("vs_copy_release", "vs_live_release") for v in info[kk].values())
    if deploy and hasattr(pair, "p1"):
        info["deployment_from_copy"] = deploy_from_copy(copy, pair, t, k, D, mine)
        ok = ok and info["deployment_from_copy"]["status"] == "PASS"
    info["status"] = "PASS" if ok and all(info["decision_preserved_vs_restored_teacher"].values()) else "FAIL"
    return info


def deploy_from_copy(copy: Path, pair, t, k, D, mine):
    """qpc.deploy.release with the copy's teacher unit and the copy's 83-column input (binding checked)."""
    from qpc import deploy as QDP
    try:
        names = [str(x) for x in D["feature_names"]]
        out, b = QDP.release(copy / "admitted" / f"rel__s{k}__{t}", pair, D["X"], QDP.schema_sha256(names), k)
    except (QDP.Refused, SystemExit) as e:
        return {"status": "FAIL", "reason": scrub_safe(e)}
    except Exception as e:  # noqa: BLE001  (a corrupt or incompatible copied artifact is a restore FAIL, not a crash)
        return {"status": "FAIL", "reason": f"{type(e).__name__}: {scrub_safe(e)[:200]}"}
    eq = {f"recipient_{i}": bool(np.array_equal(out[f"tokens_{i}"], mine[f"tok{i}"])
                                 and np.array_equal(out[f"probs_{i}"], mine[f"q{i}"])
                                 and np.array_equal(out[f"decision_{i}"], mine[f"hard{i}"])) for i in (1, 2)}
    return {"status": "PASS" if all(eq.values()) and b.get("binding") == "BOUND" else "FAIL", "binding":
            b.get("binding"), "outputs": sorted(out), "bitwise_equal_to_re_encoded_release": eq}


def restore_all(copy: Path, live: Path, targets: dict, seed: int):
    t0, c0 = time.time(), time.process_time()
    D = load_D_from_copy(copy)
    checks, teachers = {}, {}
    seeds = {seed}
    for u in (targets.get("policies") or {}).values():
        m = re.match(r"(?:pol|tea|ref)__s(\d)__", u)
        if m:
            seeds.add(int(m.group(1)))
    for k in sorted(seeds):
        for t in TEACHERS:
            if not (copy / "admitted" / f"rel__s{k}__{t}").exists() and t != "U":
                continue
            info, out = restore_teacher(copy, live, t, k, D)
            checks[f"teacher {t} (seed {k})"] = info
            if out is not None and info["status"] == "PASS":
                teachers[(t, k)] = out
    enc = _entry(targets.get("encode"))
    ldr = _entry(targets.get("policy_loader"))
    for lab, u in (targets.get("policies") or {}).items():
        if u.startswith("pol__"):
            checks[lab] = restore_policy(copy, live, u, teachers, D, encode=enc, loader=ldr)
        elif u.startswith("tea__"):
            m = re.match(r"tea__s(\d)__(.+)$", u)
            kk, t = int(m.group(1)), m.group(2)
            checks[lab] = {"unit": u, "status": checks.get(f"teacher {t} (seed {kk})", {}).get("status", "FAIL"),
                           "rebuild": f"teacher {t} seed {kk} restored above (own forward pass from the copy)"}
        else:
            checks[lab] = {"unit": u, "status": "FAIL", "reason": "unsupported unit kind for a qpc restore target"}
    checks["attacker"] = DC.restore_attacker(copy, live, targets.get("attacker"), D)
    del D
    return checks, {"wall_s": round(time.time() - t0, 1), "cpu_s": round(time.process_time() - c0, 1)}


def _entry(spec):
    if not spec:
        return None
    mod, fn = spec.split(":")
    return getattr(importlib.import_module(mod), fn)


def load_targets(targets_file):
    if not targets_file:
        return {"policies": {}, "note": "no targets.json: U teacher only; Q*, privacy/control code and attacker PENDING"}
    t = jload(targets_file)
    for lab, u in (t.get("policies") or {}).items():
        if not re.match(r"^(pol|tea)__s\d__[A-Za-z0-9_.\-]+$", u):
            raise SystemExit(f"REFUSED: target {lab!r} is not a qpc unit name")
    for key in ("encode", "policy_loader"):
        if t.get(key) and not re.match(r"^qpc\.[a-z_]+:[A-Za-z_]+$", t[key]):
            raise SystemExit(f"REFUSED: {key} must be a qpc entry point")
    if (t.get("attacker") or {}).get("fn") and not re.match(r"^(qpc|dpc)\.[a-z_]+:[A-Za-z_]+$", t["attacker"]["fn"]):
        raise SystemExit("REFUSED: attacker.fn must be a qpc/dpc entry point")
    return t


def required_targets_status(statuses):
    """The four required restore classes: U teacher, Q*, a privacy/control code, an attacker."""
    have_u = any(k.startswith("teacher U ") and v == "PASS" for k, v in statuses.items())
    pol = {k: v for k, v in statuses.items() if not k.startswith("teacher ") and k != "attacker"}
    return {"U teacher": "PASS" if have_u else "FAIL",
            "Q*": next((v for k, v in pol.items() if k.startswith("Q*")), "PENDING (no Q* target)"),
            "privacy/control code": next((v for k, v in pol.items() if not k.startswith("Q*")),
                                         "PENDING (no privacy/control target)"),
            "attacker": statuses.get("attacker", "PENDING")}


def rehearse(src: Path, targets: dict, seed: int):
    """Dry run: the restore checks read-only with the live store standing in for the copy (nothing written)."""
    try:
        checks, cost = restore_all(src, src, targets, seed)
    except SystemExit as e:
        return {"status": "NOT_RUN", "reason": scrub_safe(e)}
    st = {k: v.get("status") for k, v in checks.items()}
    return {"note": "live store used as the 'copy' (no copy exists in a dry run); rehearses code paths only",
            "statuses": st, "required": required_targets_status(st), "cost": cost}


# ------------------------------------------------------------------ (3)+(5) this study's backup
def backup(dest=None, targets_file=None, seed=1, dry_run=False, src=None, cache=None, volumes_root=None,
           out_pkg=None, components=None, detection=None, plan_only=False):
    src, cache, out_pkg = Path(src or SRC), Path(cache or CACHE), Path(out_pkg or PKG)
    files, nbytes = inventory(src)
    targets = load_targets(targets_file)
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
    free = free_gib(parent)
    plan = {"schema": "qpc-closeout-plan-v1", "at": now(), "source": "<PRIVATE_CACHE>/qpc_v1", "files": len(files),
            "bytes": nbytes, "drive_mounted": not same_device, "destination": f"{place}/{root.name}/qpc_v1",
            "same_device": same_device, "restore_seed": seed, "targets": targets, "drive_detection": ev,
            "free_gib_before": round(free, 2), "free_gib_after_estimate": round(free - nbytes / 2 ** 30, 2),
            "pending_if_absent": PENDING}
    if dry_run:
        plan["restore_rehearsal_read_only_on_live_store"] = (
            {"status": "SKIPPED (--plan-only: no data loaded, no forward pass)"} if plan_only
            else rehearse(src, targets, seed))
        print(scrub(json.dumps(plan, indent=1, default=str)))
        return plan
    if free - nbytes / 2 ** 30 < MIN_FREE_GIB:
        raise SystemExit(f"REFUSED: the copy would leave less than {MIN_FREE_GIB} GiB free at the destination")
    copy, cp = copy_study(src, root)
    checks, cost = restore_all(copy, src, targets, seed)
    statuses = {k: v.get("status") for k, v in checks.items()}
    req = required_targets_status(statuses)
    restored_ok = all(s == "PASS" for k, s in statuses.items() if k != "attacker") and \
        statuses.get("attacker") in ("PASS", "PENDING")
    bv = {"schema": "qpc-backup-verification-v1", "written_at": now(),
          "status": STATUS_LOCAL if same_device else STATUS_DRIVE,
          "off_device_backup": "PENDING (external drive not mounted)" if same_device else "DONE (this study)",
          "source": "<PRIVATE_CACHE>/qpc_v1", "destination": f"{place}/{root.name}/qpc_v1", **cp,
          "read_back": "every copied file re-read with F_NOCACHE (uncached read; not a physical cold-disk read)",
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
    ri = restore_index(place, root, statuses, same_device)
    write_public(out_pkg / "BACKUP_VERIFICATION.json", bv)
    write_public(out_pkg / "RESTORE_INDEX.json", ri)
    (root / "BACKUP_RECORD.json").write_text(json.dumps(bv, indent=1, default=str) + "\n")
    print(json.dumps({k: bv[k] for k in ("status", "files", "uncached_readback_match", "required_restores")}, indent=1))
    return bv


def restore_index(place, root, statuses, same_device):
    return {"schema": "qpc-restore-index-v1", "written_at": now(), "private_local": "<PRIVATE_CACHE>/qpc_v1",
            "copy": f"{place}/{root.name}/qpc_v1" + (" (same device; off-device copy PENDING)" if same_device else ""),
            "checksums": f"{place}/{root.name}/SHA256SUMS",
            "layout": {"admitted/rel__s{k}__{t}/": "admitted teachers (model.pt, heads, release, COMPLETE.json; "
                                                   "provenance/ADMISSION_RESULT.json)",
                       "admitted/ref__s{k}__{E,F,F0}/": "admitted reference units (verified copies of dpc units)",
                       "admitted/inputs/adult_jcv.npz": "the pinned input (sha256 e0d9e54a...2f12)",
                       "admitted/SHA256SUMS": "admission checksums",
                       "inputs/": "reconstructed derived deploy input (qpc.admit.reconstruct_deploy_input)",
                       "run/units/<unit>/": "atomic study units: files + record.json + COMPLETE.json"},
            "restore": [f"copy {place}/{root.name}/qpc_v1 to <PRIVATE_CACHE>/qpc_v1 (only if the live store is lost)",
                        "verify: shasum -a 256 -c SHA256SUMS (inside the copy's parent folder)",
                        "verify admission: shasum -a 256 -c SHA256SUMS (inside qpc_v1/admitted)",
                        f"re-run the restore checks: {CMD} backup --dry-run --targets <targets.json>",
                        "then follow QUICKSTART.md"],
            "restore_statuses": statuses,
            "not_off_device": bool(same_device)}


# ------------------------------------------------------------------ (1) dpc off-device backup (redirected receipts)
def dpc_backup(dry_run=False, volumes_root=None, out_dir=None, DCmod=None):
    out_dir = Path(out_dir or DPC_DIR)
    DCm = DCmod or DC
    vol, ev = locate_drive(volumes_root)
    base = {"schema": "qpc-dpc-custody-v1", "at": now(), "drive_detection": ev,
            "documented_command": "OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m dpc.closeout backup --dest <DRIVE_ROOT> "
                                  "--targets <PRIVATE_CACHE>/dpc_v1/run/closeout_targets.json",
            "redirect": {"dpc BACKUP_VERIFICATION.json": f"{REL}/provenance/dpc_custody/BACKUP_VERIFICATION.json",
                         "dpc RESTORE_INDEX.json": f"{REL}/provenance/dpc_custody/RESTORE_INDEX.json"},
            "closed_results_never_written": [QD.DPC_REL + "/"],
            "historical_dpc_status": "LOCAL_SAME_DEVICE_COPY_VERIFIED_OFF_DEVICE_BACKUP_PENDING (dpc "
                                     "BACKUP_VERIFICATION.json at the source commit; unchanged)"}
    if vol is None:
        rec = {**base, "status": "PENDING", "reason": "external drive with the verified prior copies not mounted",
               "pending_command": PENDING["dpc_off_device_backup"]}
        if not dry_run:
            write_public(out_dir / "STATUS.json", rec)
        else:
            print(scrub(json.dumps(rec, indent=1)))
        return rec
    if dry_run:
        rec = {**base, "status": "DRY_RUN", "would_run": base["documented_command"]}
        print(scrub(json.dumps(rec, indent=1)))
        return rec
    before = closed_trees()
    try:
        bv = DCm.backup(dest=str(vol), targets_file=str(DPC_TARGETS) if DPC_TARGETS.exists() else None, seed=1,
                        out_pkg=out_dir)
        run = {"status": bv.get("status"), "destination": bv.get("destination"), "files": bv.get("files"),
               "uncached_readback_match": bv.get("uncached_readback_match"),
               "restore_statuses": bv.get("restore_statuses"), "restore_all_pass": bv.get("restore_all_pass")}
    except SystemExit as e:
        run = {"status": "FAILED", "reason": scrub_safe(e)}
    after = closed_trees()
    rec = {**base, "status": "RUN" if before == after else "FAILED_CLOSED_RESULTS_CHANGED", "run": run,
           "closed_results_unchanged": before == after, "closed_results_state": after}
    write_public(out_dir / "STATUS.json", rec)
    return rec


# ------------------------------------------------------------------ (2) predecessor repair (redirected osf routines)
def closed_trees():
    return {k: tree_state(v) for k, v in CLOSED_RESULTS.items()}


@contextlib.contextmanager
def osf_assessment_sealed_until_qpc_lock():
    """osf.closeout.backup opens osf's assessment (the same rows) to refit its final attacker: keep it sealed until
    THIS study's EVALUATION_LOCK is pushed, so custody work never loads an assessment label earlier."""
    from osf import assess as AS
    ok, why = QD.evaluation_lock_pushed()
    orig = AS.open_assessment
    if not ok:
        def refuse(*a, **k):
            raise SystemExit(f"sealed by qpc custody: {why}")
        AS.open_assessment = refuse
    try:
        yield ok
    finally:
        AS.open_assessment = orig


def predecessor(restore=False, dry_run=False, volumes_root=None, out_dir=None, OC=None):
    out_dir = Path(out_dir or PRED_DIR)
    vol, ev = locate_drive(volumes_root)
    rel_out = f"{REL}/provenance/predecessor_custody"
    base = {"schema": "qpc-predecessor-custody-v1", "at": now(), "drive_detection": ev,
            "diskutil_list_external": diskutil_probe(),
            "routines": ["python -m osf.closeout backup --dest <DRIVE_ROOT>",
                         "python -m osf.closeout predecessor --restore"],
            "redirect": {"osf BACKUP_VERIFICATION.json": f"{rel_out}/osf_BACKUP_VERIFICATION.json",
                         "osf RESTORE_INDEX.json": f"{rel_out}/osf_RESTORE_INDEX.json",
                         "osf PREDECESSOR_CUSTODY_REPAIR.json": f"{rel_out}/PREDECESSOR_CUSTODY_REPAIR.json",
                         "smf replay report": f"{rel_out}/INDEPENDENT_VERIFICATION_DRIVE_RESTORE.json"},
            "closed_results_never_written": [_rel(v) + "/" for v in CLOSED_RESULTS.values()],
            "historical_status": "PENDING in dpc provenance/predecessor_custody/STATUS.json at the source commit "
                                 "(unchanged; this file is the qpc receipt)"}
    if vol is None:
        rec = {**base, "status": "PENDING", "reason": "external drive with the verified prior copies not mounted",
               "pending_command": PENDING["osf_smf_predecessor_repair"],
               "gaps": ["osf has no off-device copy (osf BACKUP_VERIFICATION.json: same-device copy only)",
                        "the smf drive restore (seed-1 U, J-F, L-F by the smf verifier) has not been executed",
                        "the two later smf logs are on the same internal disk only (local custody supplement)"]}
        if dry_run:
            print(scrub(json.dumps(rec, indent=1)))
            return rec
        write_public(out_dir / "STATUS.json", rec)
        return rec
    if dry_run:
        rec = {**base, "status": "DRY_RUN", "would_run": base["routines"] if restore else base["routines"][:1]}
        print(scrub(json.dumps(rec, indent=1)))
        return rec
    OC = OC or importlib.import_module("osf.closeout")
    before = closed_trees()
    saved, writes = DC.redirected(OC, out_dir)
    runs = {}
    try:
        with osf_assessment_sealed_until_qpc_lock() as unsealed_ok:
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
        DC.restore_bindings(OC, saved)
    after = closed_trees()
    rec = {**base, "status": "RUN" if before == after else "FAILED_CLOSED_RESULTS_CHANGED", "runs": runs,
           "redirected_writes": writes, "closed_results_unchanged": before == after, "closed_results_state": after}
    write_public(out_dir / "STATUS.json", rec)
    return rec


# ------------------------------------------------------------------ everything, in order
def run_all(targets_file=None, seed=1, dry_run=False, volumes_root=None, src=None, cache=None, out_pkg=None,
            plan_only=False):
    vol, ev = locate_drive(volumes_root)
    before = closed_trees()
    folders0 = top_folders(vol)
    comp = {"drive_mounted": vol is not None, "drive_detection": ev}
    if vol is None:
        comp["dpc_off_device_backup"] = dpc_backup(dry_run, volumes_root)
        comp["predecessor_repair"] = predecessor(True, dry_run, volumes_root)
        comp = _summarise(comp)
        return backup(None, targets_file, seed, dry_run, src, cache, volumes_root, out_pkg, components=comp,
                      plan_only=plan_only)
    if dry_run:
        comp["dpc_off_device_backup"] = dpc_backup(True, volumes_root)
        comp["predecessor_repair"] = predecessor(True, True, volumes_root)
        return backup(str(vol), targets_file, seed, True, src, cache, volumes_root, out_pkg, detection=ev,
                      plan_only=plan_only)
    comp["dpc_off_device_backup"] = dpc_backup(False, volumes_root)                     # (1)
    comp["predecessor_repair"] = predecessor(True, False, volumes_root)                 # (2)
    new_before_qpc = sorted(top_folders(vol) - folders0)
    comp = _summarise(comp)
    bv = backup(str(vol), targets_file, seed, False, src, cache, volumes_root, out_pkg, components=comp,
                detection=ev)                                                           # (3) + (5)
    new = sorted(top_folders(vol) - folders0)                                           # (4)
    prior = sorted(n for n in folders0 if KNOWN_FOLDERS.match(n))
    pub = lambda n: n if KNOWN_FOLDERS.match(n) else "<other new folder>"              # noqa: E731
    reread = {pub(n): verify_sums(Path(vol) / n) for n in new}
    prior_reread = {n: verify_sums(Path(vol) / n) for n in prior}
    after = closed_trees()
    bv["uncached_reread_of_drive_folders"] = {
        "created_by_this_closeout": reread, "created_before_the_qpc_copy": len(new_before_qpc),
        "prior_known_backup_folders": prior_reread,
        "all_pass": all(v["pass"] for v in list(reread.values()) + list(prior_reread.values())),
        "rule": "every file listed in each folder's SHA256SUMS re-read with F_NOCACHE (uncached; not a physical "
                "cold-disk read); only folder names of the known private_<study>_v1_<date> form are published"}
    bv["closed_results_unchanged"] = before == after
    if before != after:
        bv["status"] = "FAILED_CLOSED_RESULTS_CHANGED"
    write_public(Path(out_pkg or PKG) / "BACKUP_VERIFICATION.json", bv)
    return bv


def _summarise(comp):
    out = dict(comp)
    for k in ("dpc_off_device_backup", "predecessor_repair"):
        r = comp.get(k) or {}
        out[k] = {"status": r.get("status"), "receipt": "provenance/" + ("dpc_custody" if k.startswith("dpc") else
                                                                         "predecessor_custody") + "/STATUS.json",
                  "pending_command": r.get("pending_command")}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m qpc.closeout")
    ap.add_argument("job", choices=("status", "all", "backup", "dpc-backup", "predecessor"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--plan-only", action="store_true", help="with --dry-run: skip the read-only restore rehearsal")
    ap.add_argument("--restore", action="store_true", help="predecessor: also run osf.closeout predecessor --restore")
    ap.add_argument("--dest", default=None)
    ap.add_argument("--targets", default=None)
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
        print(json.dumps(status(), indent=1))
    elif a.job == "all":
        run_all(a.targets, a.seed, a.dry_run, plan_only=a.plan_only)
    elif a.job == "backup":
        backup(a.dest, a.targets, a.seed, a.dry_run, plan_only=a.plan_only)
    elif a.job == "dpc-backup":
        print(json.dumps({"status": dpc_backup(a.dry_run)["status"]}))
    else:
        print(json.dumps({"status": predecessor(a.restore, a.dry_run)["status"]}))


if __name__ == "__main__":
    main()
