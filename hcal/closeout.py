"""Custody, resource accounting and reporting for the held-out calibration study (hcal; owner role F).

Ported from lra/closeout.py at 9762025 (the template). IMPORTED unchanged from the pinned predecessors, exactly as
lra.closeout used them: the drive probe by content (dpc.closeout.locate_drive with the smf marker read at the pinned qpc
source commit, qpc.closeout.smf_marker_sha), the versioned folder rule (dpc.closeout.versioned), the copy with uncached
read-back (cbp.closeout.copy_study), the SHA256SUMS re-read (qpc.closeout.verify_sums), the F_NOCACHE hash
(dpc.closeout.sha(nocache=True)), the scrubbers and the counts-only diskutil probe. NOT ported (lra-specific): the cbp /
qpc / dpc / osf / smf predecessor custody chain, the lra role-target vocabulary and the lra D1 decoder re-certification.
lra keeps its own custody (its verified same-device copy); hcal only reads it.

PRIVATE STORE. hcal.ids.PRIV (environment variable PCRL_HCAL_PRIVATE_CACHE; default <HOME>/PCRL_eval_cache_private/
hcal_v1). HOME is never repurposed. Tracked outputs name places only by placeholders: <PRIVATE_CACHE> (= the folder that
holds hcal_v1), <DRIVE_ROOT>, <python>, <WORKTREE>; never an absolute path, a volume name or the local user name
(write_public refuses them).

COPIES (nothing is ever deleted, moved or overwritten; versioned folders are never reused):
  same_device_copy  <PRIVATE_CACHE>/hcal_v1_local_copy_<UTC YYYYMMDD>[_vN]/{hcal_v1/, dependencies/jcv_v1/inputs/
                    adult_jcv.npz, SHA256SUMS, BACKUP_RECORD.json}. Label SAME_DEVICE_COPY. It is NOT off-device custody
                    and is never called an off-device backup or a drive restore.
  off_device_copy   only on the volume identified by CONTENT: <volume>/private_smf_v1_20261005/SHA256SUMS must hash to the
                    value the closed smf study recorded. The boot volume and the unrelated read-only installer image
                    "BackgroundSyncService Setup" are skipped by name BEFORE any filesystem call (never touched); volume
                    names are never recorded. Absent drive: status PENDING with the exact command.
  Every copied file is listed in SHA256SUMS and re-read with F_NOCACHE (an uncached read, NOT a physical cold-disk read:
  a cache purge or remount is not authorised). The pinned source input is bundled so the copy restores alone; the store
  is copied byte for byte (no file is parsed; outer__ units exist only after this study's EVALUATION_LOCK).

RESTORES FROM THE COPY ALONE (every function takes the copy root; the live store is at most an optional comparison):
  restore_teacher        own forward pass: dpc.deploy.load_teacher + teacher_probs on the copy's 83-column deploy input
                         vs the copy's teacher.npz, bitwise (p1, p2, d1 = argmax p1, d2 = argmax p2); model.pt and the
                         input hash-checked against the copy's ADMISSION_RECEIPT.json and the tracked SOURCE_ADMISSION.json.
  restore_bank_tables    representative frozen-bank tables: uncached sha256 equal to the admission receipt (copy) and the
                         tracked SOURCE_ADMISSION.json frozen_bank entry.
  restore_family_tables  hook: every calibrator family's table units rebuilt by a caller-supplied function, bitwise.
  restore_release        hook: the selected or a control release rebuilt by a caller-supplied function, bitwise.
  restore_reader_refit   hook: the selected reader refit by a caller-supplied function vs its saved predictions.
  restore_summary        required classes; a missing class is PENDING (never PASS), NOT_APPLICABLE needs a reason.
Hooks accept a Python callable or an entry point "hcal.<module>:<function>" (any other module is refused).

ACCOUNTING AND REPORTING (read-only): accounting() sums child CPU from <PRIVATE_CACHE>/hcal_v1/run/SEMA_LOG.jsonl by role
label prefix, the stage CPU from COMPUTE_LEDGER.jsonl, elapsed time from START.txt and the maximum concurrent semaphore
holds; identity_scan() scans files for identifying paths, the local user name, emails, credential-like strings and
per-person record patterns; owned_processes() lists processes whose command line contains hcal.run / hcal.sema /
hcal.assess / hcal.verify (never kills anything); status_report() is the 30-60 min status.

    OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m hcal.sema --label F:closeout -- env OMP_NUM_THREADS=1 PYTHONPATH=. \\
        <python> -m hcal.closeout {status | drive | accounting | processes | scan FILE... |
                                   copy [--dest-name N] [--dry-run] | offdevice [--dry-run]}
"""
from __future__ import annotations

import argparse
import calendar
import importlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np

from cbp import closeout as CB
from dpc import closeout as DC
from hcal import ids as I
from qpc import closeout as QC

PRIV = I.PRIV
RUN = I.RUN
PKG = I.PKG
REL = I.REL
STORE_NAME = PRIV.name                                   # "hcal_v1"
LOCAL_RE = re.compile(r"^hcal_v1_local_copy_\d{8}$")
LOCAL_FOLDER = "hcal_v1_local_copy_{date}"
DRIVE_FOLDER = "private_hcal_v1_{date}"
KNOWN_FOLDERS = re.compile(r"^private_(hcal|lra|lcr|cbp|qpc|dpc|osf|smf)_v1(_custody_supplement)?_\d{8}(_v\d+)?$")
DEP_REL = "dependencies/jcv_v1/inputs/adult_jcv.npz"
VOLUMES_ROOT = DC.VOLUMES_ROOT
SKIP_VOLUMES = DC.SKIP_VOLUMES
MIN_FREE_GIB = 5
SLOTS = 2
BUDGET_CPU_H = 20.0
BUDGET_WALL_H = 10.0
TEACHERS = ("U", "RAW-J_b0.3")
REPRESENTATIVE_PARTITIONS = ("U|DIRECT-TASK|i8o64", "U|JOINT|i8o64|l0.1", "U|C-TASK|i8o64", "U|K-JOINT-PAIR|i8o64")
LABEL_LOCAL = "SAME_DEVICE_COPY"
LABEL_DRIVE = "OFF_DEVICE_COPY"
REQUIRED = ("U teacher", "frozen bank", "calibrator families", "selected release", "control release",
            "selected reader refit")
ENTRY = re.compile(r"^hcal\.[a-z_]+:[A-Za-z_][A-Za-z0-9_]*$")
OWNED_PATTERNS = ("hcal.run", "hcal.sema", "hcal.assess", "hcal.verify", "hcal/sema.py")
HOME_NAME = Path.home().name
SEMA = ("OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m hcal.sema --label F:closeout -- env OMP_NUM_THREADS=1 "
        "PYTHONPATH=. <python> -m hcal.closeout")
PENDING = {"off_device_copy_and_restore": f"{SEMA} offdevice   (finds <DRIVE_ROOT> by content; copies "
                                          "<PRIVATE_CACHE>/hcal_v1 and the pinned input to <DRIVE_ROOT>/"
                                          "private_hcal_v1_<UTC date>[_vN], re-reads every file uncached, then the "
                                          "coordinator runs the restore hooks on that copy)"}

now = DC.now
sha = DC.sha
jload = DC.jload
scrub_safe = DC.scrub_safe
versioned = DC.versioned
cmp_arrays = DC.cmp_arrays
diskutil_probe = DC.diskutil_probe
verify_sums = QC.verify_sums
free_gib = QC.free_gib
top_folders = QC.top_folders
copy_study = CB.copy_study
uncached_read_capability = CB.uncached_read_capability


# ------------------------------------------------------------------ public writing
def _finite(o):
    if isinstance(o, dict):
        return {str(k): _finite(v) for k, v in o.items() if not str(k).startswith("_")}   # "_" keys stay private
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    if isinstance(o, Path):
        return "<PATH>"
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(float(o)) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _finite(o.tolist())
    return o


def scrub(text):
    """Refuse public text with a home, volume or private-tmp path or the local user name (dpc rule; HOME_NAME here)."""
    if re.search(r"/(?:Users|Volumes|private)/|/(?:home)/[A-Za-z0-9._-]", text) or (HOME_NAME and HOME_NAME in text):
        raise SystemExit("REFUSED: identifying path or user name in a public file")
    return text


def public(obj):
    """The publishable form of a record: '_'-prefixed keys dropped, Paths masked, nonfinite -> null, scrubbed."""
    return json.loads(scrub(json.dumps(_finite(obj), default=str, allow_nan=False)))


def write_public(path, obj):
    """Atomic, scrubbed, finite-or-null JSON for every public receipt (BACKUP_VERIFICATION.json, RESTORE_INDEX.json)."""
    txt = scrub(json.dumps(_finite(obj), indent=1, default=str, allow_nan=False) + "\n")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(txt)
    tmp.replace(path)


def _cache_parent(src=None):
    return Path(src or PRIV).parent


def _utc_date():
    return time.strftime("%Y%m%d", time.gmtime())


def pinned_input():
    """(path, sha256) of the pinned source input, from the pinned loader (rgj.data via osf.data); read lazily."""
    from rgj import data as RD
    return Path(RD.SRC), RD.SRC_SHA


# ------------------------------------------------------------------ drive detection (by content, lra rule)
def smf_marker_sha():
    return QC.smf_marker_sha()


def locate_drive(volumes_root=None, want=None):
    """(volume, evidence) by content (dpc.closeout.locate_drive; skipped names checked before any filesystem call;
    names never recorded)."""
    return DC.locate_drive(Path(volumes_root or VOLUMES_ROOT), want or smf_marker_sha())


def drive_status(volumes_root=None):
    vol, ev = locate_drive(volumes_root)
    return {"at": now(), "mounted": vol is not None, "drive_detection": ev, "diskutil_list_external": diskutil_probe(),
            "off_device_copy": "POSSIBLE (drive mounted)" if vol is not None else "PENDING (drive not mounted)",
            "pending_command": None if vol is not None else PENDING["off_device_copy_and_restore"]}


# ------------------------------------------------------------------ copies
def _deps(deps):
    if deps is not None:
        return dict(deps)
    p, _ = pinned_input()
    return {DEP_REL: p}


def _check_deps(deps, want_sha):
    for r, s_ in deps.items():
        if not Path(s_).is_file():
            raise SystemExit(f"REFUSED: dependency {r} is missing")
        if sha(s_) != want_sha:
            raise SystemExit(f"REFUSED: dependency {r} fails its pinned hash")


def _copy(parent: Path, name: str, place: str, label: str, src=None, deps=None, dep_sha=None, dry_run=False,
          min_free=MIN_FREE_GIB):
    src = Path(src or PRIV)
    if not src.is_dir():
        raise SystemExit("REFUSED: the private store does not exist")
    deps = _deps(deps)
    if dep_sha is None:
        dep_sha = pinned_input()[1]
    files, nbytes = DC.inventory(src)
    dep_bytes = sum(Path(s_).stat().st_size for s_ in deps.values() if Path(s_).exists())
    root = versioned(Path(parent), name)
    free = free_gib(parent)
    plan = {"label": label, "at": now(), "source": f"<PRIVATE_CACHE>/{src.name}", "store_files": len(files),
            "store_bytes": nbytes, "dependencies": {r: "pinned source input (sha256 " + str(dep_sha)[:12] + "...)"
                                                    for r in deps},
            "destination": f"{place}/{root.name}", "versioned_folder": root.name,
            "free_gib_before": round(free, 2), "free_gib_after_estimate": round(free - (nbytes + dep_bytes) / 2 ** 30, 2),
            "_root": root}
    if dry_run:
        return {**plan, "dry_run": True, "written": "nothing"}
    if free - (nbytes + dep_bytes) / 2 ** 30 < min_free:
        raise SystemExit(f"REFUSED: the copy would leave less than {min_free} GiB free at the destination")
    _check_deps(deps, dep_sha)
    copy, cp = copy_study(src, root, deps)
    vs = verify_sums(root)
    rec = {**plan, **cp, "independent_uncached_reread": vs,
           "verified": bool(vs["pass"] and cp["uncached_readback_match"] == cp["files"]),
           "live_files_note": "a live file that kept changing is copied as a hashed snapshot and listed in "
                              "live_files_changed_during_copy (never silently accepted)",
           "read_back": "every copied file re-read with F_NOCACHE (uncached read; not a physical cold-disk read)",
           "read_capability": uncached_read_capability(root / "SHA256SUMS"),
           "deleted": "nothing", "moved": "nothing", "_copy_store": copy}
    (root / "BACKUP_RECORD.json").write_text(json.dumps(public(rec), indent=1) + "\n")
    return rec


def same_device_copy(dest_name=None, src=None, parent=None, deps=None, dep_sha=None, dry_run=False):
    """A versioned SAME-DEVICE copy <PRIVATE_CACHE>/hcal_v1_local_copy_<UTC YYYYMMDD>[_vN] of the hcal_v1 store and the
    pinned input, with SHA256SUMS and an uncached re-read of every file. Label SAME_DEVICE_COPY (never off-device
    custody). Returns the record; '_root' / '_copy_store' are the private paths (dropped from public output)."""
    name = dest_name or LOCAL_FOLDER.format(date=_utc_date())
    if not LOCAL_RE.match(name):
        raise SystemExit("REFUSED: dest_name must be hcal_v1_local_copy_<YYYYMMDD> (a folder name, not a path)")
    rec = _copy(Path(parent or _cache_parent(src)), name, "<PRIVATE_CACHE>", LABEL_LOCAL, src, deps, dep_sha, dry_run)
    rec["off_device_copy"] = "PENDING (this copy is on the SAME device; it is not off-device custody)"
    rec["pending_command"] = PENDING["off_device_copy_and_restore"]
    return rec


def off_device_copy(volumes_root=None, src=None, deps=None, dep_sha=None, dry_run=False, want=None):
    """The off-device copy on the content-identified drive, or PENDING when it is absent (nothing written). With the
    drive: <DRIVE_ROOT>/private_hcal_v1_<UTC date>[_vN], uncached re-read of the new folder and of every prior known
    private_<study>_v1_<date> folder (only those names are ever published)."""
    vol, ev = locate_drive(volumes_root, want)
    if vol is None:
        return {"label": LABEL_DRIVE, "status": "PENDING", "at": now(), "reason": "external drive with the verified "
                "prior copies not mounted", "drive_detection": ev, "pending_command":
                PENDING["off_device_copy_and_restore"]}
    before = top_folders(vol)
    rec = _copy(vol, DRIVE_FOLDER.format(date=_utc_date()), "<DRIVE_ROOT>", LABEL_DRIVE, src, deps, dep_sha, dry_run)
    rec["drive_detection"] = ev
    if dry_run:
        rec["status"] = "DRY_RUN"
        return rec
    prior = sorted(n for n in before if KNOWN_FOLDERS.match(n))
    rec["prior_known_folders_uncached_reread"] = {n: verify_sums(Path(vol) / n) for n in prior}
    rec["other_folders_on_drive"] = len([n for n in before if not KNOWN_FOLDERS.match(n)])
    ok = rec["verified"] and all(v["pass"] for v in rec["prior_known_folders_uncached_reread"].values())
    rec["status"] = "OFF_DEVICE_COPY_VERIFIED" if ok else "OFF_DEVICE_COPY_FAILED_VERIFICATION"
    return rec


# ------------------------------------------------------------------ restore from the copy alone
def copy_store(copy_root):
    s = Path(copy_root) / STORE_NAME
    if not s.is_dir():
        raise SystemExit(f"REFUSED: the copy holds no {STORE_NAME} store")
    return s


def _receipt(store):
    p = Path(store) / "admitted" / "ADMISSION_RECEIPT.json"
    return jload(p) if p.is_file() else None


def _public_admission(path=None):
    p = Path(path) if path else PKG / "SOURCE_ADMISSION.json"
    try:
        return jload(p)
    except (OSError, ValueError):
        return None


def _pub(public_admission):
    """The tracked SOURCE_ADMISSION.json (default), a given dict, or None when public_admission is False."""
    if public_admission is False:
        return None
    return public_admission if public_admission is not None else _public_admission()


def _resolve(fn):
    if callable(fn):
        return fn
    if isinstance(fn, str) and ENTRY.match(fn):
        mod, name = fn.split(":")
        return getattr(importlib.import_module(mod), name)
    raise SystemExit(f"REFUSED: hook {fn!r} must be a callable or an hcal.<module>:<function> entry point")


def unit_ok(units: Path, name: str, live_units=None):
    """A copied unit: COMPLETE.json present and every listed file re-hashes uncached; optionally equal to the live one."""
    d = Path(units) / name
    if not (d / "COMPLETE.json").is_file():
        return {"unit": name, "present_in_copy": False, "ok": False}
    c = jload(d / "COMPLETE.json")
    files_ok = all((d / f).is_file() and sha(d / f, nocache=True) == h for f, h in c["files"].items())
    r = {"unit": name, "present_in_copy": True, "files": len(c["files"]), "copy_files_hash_ok_uncached": files_ok}
    if live_units is not None:
        lc = Path(live_units) / name / "COMPLETE.json"
        r["copy_complete_equals_live"] = lc.is_file() and jload(lc) == c
    r["ok"] = bool(files_ok and r.get("copy_complete_equals_live", True))
    return r


def restore_teacher(copy_root, k, t="U", loader=None, probs=None, public_admission=None):
    """Teacher t of seed k from the copy alone: own forward pass (dpc.deploy.load_teacher + teacher_probs) of the copied
    model.pt + heads on the copy's deploy input, compared bitwise with the copy's admitted teacher.npz."""
    from dpc import deploy as DD
    store = copy_store(copy_root)
    loader, probs = loader or DD.load_teacher, probs or DD.teacher_probs
    tdir, tea = store / "admitted" / f"rel__s{k}__{t}", store / "admitted" / "units" / f"tea__s{k}__{t}"
    inp = store / "admitted" / "inputs"
    res = {"teacher": t, "seed": int(k), "how": "dpc.deploy.load_teacher + teacher_probs on the copy's deploy input"}
    try:
        names = DD.schema_names(inp / "schema.json")
        X = DD.load_permitted_input(inp / "deploy_input.npz", names)
        model, heads, msha = loader(tdir, seed=k)
        P = probs(model, heads, X)
        with np.load(tea / "teacher.npz", allow_pickle=False) as zz:
            z = {x: zz[x] for x in ("row_id", "p1", "p2", "d1", "d2")}
    except (DD.Refused, OSError, KeyError, ValueError, RuntimeError) as e:
        return {**res, "status": "FAIL", "reason": f"{type(e).__name__}: {scrub_safe(e)[:200]}"}
    for i in (1, 2):
        res[f"p{i}_bitwise"] = bool(np.array_equal(P[i - 1], z[f"p{i}"]) and P[i - 1].dtype == z[f"p{i}"].dtype)
        res[f"d{i}_bitwise"] = bool(np.array_equal(P[i - 1].argmax(1), z[f"d{i}"]))
    res["rows_equal"] = bool(len(X) == len(z["row_id"]))
    res["teacher_unit"] = unit_ok(store / "admitted" / "units", f"tea__s{k}__{t}")
    rc = _receipt(store) or {}
    pub = _pub(public_admission)
    res["model_sha256_equals_copy_receipt"] = (rc.get("teachers") or {}).get(f"{t}|{k}", {}).get("model_sha256") == msha
    res["input_sha256_equals_copy_receipt"] = (rc.get("inputs") or {}).get("deploy_input.npz") == \
        sha(inp / "deploy_input.npz", nocache=True)
    if pub is not None:
        res["model_sha256_equals_tracked_admission"] = \
            (pub.get("teacher_parity") or {}).get(f"{t}|{k}", {}).get("model_sha256") == msha
    oks = [v for kk, v in res.items() if isinstance(v, bool)] + [res["teacher_unit"]["ok"]]
    res["status"] = "PASS" if all(oks) else "FAIL"
    return res


def restore_teachers(copy_root, seeds=I.SEEDS, teachers=TEACHERS, **kw):
    return {f"teacher {t} (seed {k})": restore_teacher(copy_root, k, t, **kw) for k in seeds for t in teachers}


def restore_bank_tables(copy_root, keys=None, public_admission=None):
    """Frozen-bank tables (default: the four REPRESENTATIVE_PARTITIONS x seeds 0-2; keys = ["<k>|<partition>", ...] or
    "all"): uncached sha256 of the copied table equal to the copy's ADMISSION_RECEIPT.json and to the tracked
    SOURCE_ADMISSION.json frozen_bank entry; the table must load without pickles."""
    from hcal import admit as AD
    store = copy_store(copy_root)
    rc = _receipt(store)
    if rc is None:
        return {"status": "FAIL", "reason": "the copy holds no admitted/ADMISSION_RECEIPT.json"}
    bank = rc.get("bank") or {}
    pub = _pub(public_admission)
    pbank = (pub or {}).get("frozen_bank") or {}
    if keys == "all":
        keys = sorted(bank)
    keys = list(keys or [f"{k}|{p}" for k in I.SEEDS for p in REPRESENTATIVE_PARTITIONS])
    out = {}
    for key in keys:
        k, p = key.split("|", 1)
        path = store / "admitted" / "bank" / AD.bank_path(int(k), p).name
        want = (bank.get(key) or {}).get("bank_sha256")
        r = {"table": path.name, "in_copy_receipt": want is not None}
        if not path.is_file() or want is None:
            out[key] = {**r, "status": "FAIL", "reason": "table or receipt entry missing in the copy"}
            continue
        got = sha(path, nocache=True)
        r["sha256_equals_copy_receipt"] = got == want
        if pub is not None:
            r["sha256_equals_tracked_admission"] = (pbank.get(key) or {}).get("bank_sha256") == got
        try:
            with np.load(path, allow_pickle=False) as z:
                r["arrays"] = len(z.files)
            r["loads_without_pickle"] = True
        except (OSError, ValueError) as e:
            r["loads_without_pickle"] = False
            r["reason"] = type(e).__name__
        r["status"] = "PASS" if all(v for kk, v in r.items() if isinstance(v, bool)) else "FAIL"
        out[key] = r
    return {"status": "PASS" if out and all(v["status"] == "PASS" for v in out.values()) else "FAIL",
            "tables": out, "checked": len(out), "tracked_admission_compared": pub is not None}


def restore_unit_tables(copy_root, units, rebuild, table_file, keys=None, units_dir="run/units", live_root=None,
                        kwargs=None):
    """Generic: for each unit, verify it in the copy, call rebuild(store=<copy store>, unit=<name>, **kwargs) -> {key:
    array}, and compare bitwise with <copy unit>/<table_file> (and the live unit when live_root is given)."""
    store = copy_store(copy_root)
    f = _resolve(rebuild)
    out = {}
    for u in units:
        info = unit_ok(store / units_dir, u, (Path(live_root) / units_dir) if live_root else None)
        if not info["ok"]:
            out[u] = {**info, "status": "FAIL", "reason": "unit missing in the copy or does not verify"}
            continue
        try:
            mine = f(store=store, unit=u, **(kwargs or {}))
            with np.load(store / units_dir / u / table_file, allow_pickle=False) as z:
                ks = list(keys or z.files)
                info["vs_copy"] = cmp_arrays(mine, z, ks)
            if live_root:
                with np.load(Path(live_root) / units_dir / u / table_file, allow_pickle=False) as z:
                    info["vs_live"] = cmp_arrays(mine, z, ks)
        except SystemExit as e:
            out[u] = {**info, "status": "FAIL", "reason": scrub_safe(e)[:200]}
            continue
        except Exception as e:                                                       # noqa: BLE001
            out[u] = {**info, "status": "FAIL", "reason": f"{type(e).__name__}: {scrub_safe(e)[:200]}"}
            continue
        ok = bool(ks) and all(c["bitwise"] for kk in ("vs_copy", "vs_live") for c in info.get(kk, {}).values())
        out[u] = {**info, "keys": ks, "status": "PASS" if ok else "FAIL"}
    return out


def restore_family_tables(copy_root, families: dict, rebuild, table_file="table.npz", required=None, **kw):
    """Each calibrator family table: families = {family: [unit, ...]} (e.g. H-TOKEN32, H-GLOBAL-TEMP, H-CLASS-TEMP,
    T-TOKEN32, U calibrations). A required family with no unit is MISSING (FAIL), never PASS."""
    required = tuple(required if required is not None else I.NEW_DECODERS)
    out = {}
    for fam in list(dict.fromkeys(list(required) + list(families))):
        us = list(families.get(fam) or [])
        if not us:
            out[fam] = {"status": "FAIL", "reason": "MISSING: no unit given for this family"}
            continue
        r = restore_unit_tables(copy_root, us, rebuild, table_file, **kw)
        out[fam] = {"status": "PASS" if all(v["status"] == "PASS" for v in r.values()) else "FAIL", "units": r}
    return {"status": "PASS" if out and all(v["status"] == "PASS" for v in out.values()) else "FAIL", "families": out}


def restore_release(copy_root, unit, rebuild, role="selected", release_file="release.npz", keys=None,
                    decision_teacher=None, units_dir="run/units", live_root=None, kwargs=None):
    """The selected or a control release: rebuilt from the copy by `rebuild`, bitwise vs the copied release; with
    decision_teacher=(k, t) the rebuilt hard1/hard2 must equal the copied teacher's d1/d2 (decision preservation)."""
    r = restore_unit_tables(copy_root, [unit], rebuild, release_file, keys, units_dir, live_root, kwargs)[unit]
    r["role"] = role
    if decision_teacher is not None and r["status"] == "PASS":
        k, t = decision_teacher
        store = copy_store(copy_root)
        with np.load(store / units_dir / unit / release_file, allow_pickle=False) as z, \
                np.load(store / "admitted" / "units" / f"tea__s{k}__{t}" / "teacher.npz", allow_pickle=False) as T:
            r["decisions_preserved"] = {f"recipient_{i}": bool(np.array_equal(z[f"hard{i}"], T[f"d{i}"]))
                                        for i in (1, 2)}
        if not all(r["decisions_preserved"].values()):
            r["status"] = "FAIL"
    return r


def restore_reader_refit(copy_root, unit, refit, saved_key, saved_file="preds.npz", tolerance=0.0, D=None,
                         units_dir="run/units", live_root=None, kwargs=None):
    """The selected reader: refit(store=<copy store>, unit=<name>, D=D, **kwargs) -> predictions, compared with the
    predictions saved in the copied unit (and the live unit) within `tolerance` (default exact)."""
    store = copy_store(copy_root)
    info = unit_ok(store / units_dir, unit, (Path(live_root) / units_dir) if live_root else None)
    if not info["ok"]:
        return {**info, "status": "FAIL", "reason": "unit missing in the copy or does not verify"}
    try:
        P = np.asarray(_resolve(refit)(store=store, unit=unit, D=D, **(kwargs or {})), dtype=np.float64)
        roots = [("copy", store)] + ([("live", Path(live_root))] if live_root else [])
        d = {}
        for w, root in roots:
            with np.load(Path(root) / units_dir / unit / saved_file, allow_pickle=False) as z:
                s_ = np.asarray(z[saved_key], dtype=np.float64)
            d[w] = float(np.abs(P - s_).max()) if P.shape == s_.shape and P.size else float("inf")
    except SystemExit as e:
        return {**info, "status": "FAIL", "reason": scrub_safe(e)[:200]}
    except Exception as e:                                                           # noqa: BLE001
        return {**info, "status": "FAIL", "reason": f"{type(e).__name__}: {scrub_safe(e)[:200]}"}
    tol = float(tolerance)
    return {**info, "status": "PASS" if all(v <= tol for v in d.values()) else "FAIL", "tolerance": tol,
            "max_abs_diff": d, "rows_compared": int(P.shape[0]) if P.ndim else 0, "saved_key": saved_key}


def load_D_from_copy(copy_root, want_sha=None):
    """Sealed D for a reader refit, read ONLY from the copy's bundled pinned input (hash-checked uncached here and by
    osf.data). The only rebinding is osf.data.SRC -> that byte-identical file (dpc.closeout.input_from, exactly as
    lra.closeout.load_D_from_input); unseal is never requested, and no gate or label allowlist is touched."""
    p = Path(copy_root) / DEP_REL
    if not p.is_file():
        raise SystemExit("REFUSED: the copy holds no bundled input file")
    if sha(p, nocache=True) != (want_sha or pinned_input()[1]):
        raise SystemExit("REFUSED: the copy's input file fails its pinned hash")
    from hcal import data as HD
    with DC.input_from(p):
        D = HD.load(verify=True, unseal=False)
    if not D.get("sealed"):
        raise SystemExit("REFUSED: assessment labels are not sealed in the restored data")
    return D


def _status_of(v):
    if isinstance(v, dict) and "status" in v:
        return v["status"]
    if isinstance(v, dict) and v:
        return "PASS" if all(_status_of(x) == "PASS" for x in v.values()) else "FAIL"
    return "FAIL"


def restore_summary(checks: dict, not_applicable=None):
    """checks = {required class: result (a status dict, or a dict of status dicts)}. Missing -> PENDING (never PASS);
    not_applicable = {class: reason} -> NOT_APPLICABLE (<reason>) (never for the U teacher or the frozen bank)."""
    na = dict(not_applicable or {})
    bad = sorted(c for c in na if c in ("U teacher", "frozen bank") or not (isinstance(na[c], str) and na[c].strip()))
    if bad:
        raise SystemExit(f"REFUSED: not_applicable {bad} (the U teacher and frozen bank restores are always required, "
                         "and every exclusion needs a reason)")
    req = {}
    for c in REQUIRED:
        if c in checks and checks[c] is not None:
            if c in na:
                raise SystemExit(f"REFUSED: {c!r} marked not applicable but a restore check for it is given")
            req[c] = _status_of(checks[c])
        elif c in na:
            req[c] = f"NOT_APPLICABLE ({na[c]})"
        else:
            req[c] = "PENDING"
    return {"required_restores": req,
            "restore_all_pass": all(v == "PASS" or v.startswith("NOT_APPLICABLE") for v in req.values()),
            "complete": not any(v == "PENDING" for v in req.values())}


def backup_verification(copy_rec, off_rec=None, checks=None, not_applicable=None):
    """BACKUP_VERIFICATION.json body (public form). The same-device label is never upgraded to off-device custody."""
    summ = restore_summary(checks or {}, not_applicable)
    off = off_rec or {"status": "PENDING", "pending_command": PENDING["off_device_copy_and_restore"]}
    label = copy_rec.get("label")
    bv = {"schema": "hcal-backup-verification-v1", "written_at": now(), "copy": copy_rec,
          "copy_label": label, "off_device": off,
          "custody_status": ("OFF_DEVICE_COPY_VERIFIED" if off.get("status") == "OFF_DEVICE_COPY_VERIFIED" else
                             "SAME_DEVICE_COPY_VERIFIED_OFF_DEVICE_PENDING" if label == LABEL_LOCAL and
                             copy_rec.get("verified") else "NOT_VERIFIED"),
          "restore_kind": ("restore test from a SAME-DEVICE copy (proves restorability, not off-device custody)"
                           if label == LABEL_LOCAL else "restore from the off-device copy alone"),
          "restore_checks": checks or {}, **summ, "deleted": "nothing", "moved": "nothing"}
    return public(bv)


def restore_index(copy_rec, summary=None):
    place = copy_rec.get("destination", "<PRIVATE_CACHE>/<copy>")
    return public({"schema": "hcal-restore-index-v1", "written_at": now(), "private_local": "<PRIVATE_CACHE>/hcal_v1",
                   "copy": place + (" (same device; off-device copy PENDING)" if copy_rec.get("label") == LABEL_LOCAL
                                    else ""), "checksums": f"{place}/SHA256SUMS",
                   "layout": {"hcal_v1/admitted/": "verified copies of the lra units, teachers, deploy input, frozen "
                                                   "bank tables and ADMISSION_RECEIPT.json (never written after "
                                                   "admission)",
                              "hcal_v1/run/units/<unit>/": "atomic study units: files + record.json + COMPLETE.json",
                              "hcal_v1/run/*.jsonl": "activity, compute and semaphore ledgers",
                              DEP_REL: "the pinned source input, bundled so the copy restores alone"},
                   "restore": [f"verify: shasum -a 256 -c SHA256SUMS (inside {place})",
                               "copy hcal_v1 back to <PRIVATE_CACHE>/hcal_v1 only if the live store is lost",
                               "re-run the restore hooks of hcal.closeout on the copy root"],
                   **(summary or {})})


# ------------------------------------------------------------------ accounting (read-only)
def _ts(s):
    return calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ"))


def _jsonl(p):
    out = []
    try:
        for ln in Path(p).read_text().splitlines():
            if ln.strip():
                try:
                    out.append(json.loads(ln))
                except ValueError:
                    out.append({"_unparsed": True})
    except OSError:
        pass
    return out


def accounting(run_dir=None, start_file=None, now_ts=None, budget_cpu_h=BUDGET_CPU_H, budget_wall_h=BUDGET_WALL_H):
    """CPU / wall accounting from the shared semaphore log (child CPU summed by role label prefix 'A:', 'F:', ...),
    the stage compute ledger, START.txt and the maximum number of concurrent semaphore holds. Work outside the
    semaphore (agents' own processes, light commands) is not in these logs and is reported as not captured."""
    run_dir = Path(run_dir or RUN)
    t_now = now_ts if now_ts is not None else time.time()
    sema = _jsonl(run_dir / "SEMA_LOG.jsonl")
    by_role, n_rel, intervals, open_ = {}, 0, [], {}
    for e in sema:
        key = (e.get("wrapper_pid"), e.get("slot"), e.get("label"))
        try:
            _ts(e.get("at", ""))
        except (TypeError, ValueError):
            e["_unparsed"] = True
            continue
        if e.get("event") == "acquire":
            open_[key] = _ts(e["at"])
        elif e.get("event") == "release":
            n_rel += 1
            role = str(e.get("label", "?")).split(":", 1)[0]
            r = by_role.setdefault(role, {"cpu_s": 0.0, "wall_s": 0.0, "holds": 0, "max_child_rss_bytes": 0})
            r["cpu_s"] += float(e.get("cpu_s") or 0.0)
            r["wall_s"] += float(e.get("wall_s") or 0.0)
            r["holds"] += 1
            r["max_child_rss_bytes"] = max(r["max_child_rss_bytes"], int(e.get("child_maxrss_bytes") or 0))
            t1 = _ts(e["at"])
            intervals.append((open_.pop(key, t1 - float(e.get("wall_s") or 0.0)), t1))
    active = [{"label": k[2], "slot": k[1], "held_s": round(t_now - t0, 1),
               "note": "no release logged yet (running, or its wrapper died unlogged)"} for k, t0 in open_.items()]
    intervals += [(t0, t_now) for t0 in open_.values()]
    ev = sorted([(a, 1) for a, _ in intervals] + [(b, -1) for _, b in intervals], key=lambda x: (x[0], x[1]))
    cur = peak = 0
    for _, d in ev:                                   # half-open holds: a release at t frees the slot before t's acquire
        cur += d
        peak = max(peak, cur)
    led = _jsonl(run_dir / "COMPUTE_LEDGER.jsonl")
    by_stage = {}
    for e in led:
        s_ = by_stage.setdefault(str(e.get("stage")), {"cpu_s": 0.0, "wall_s": 0.0, "shards": 0, "max_rss_bytes": 0})
        s_["cpu_s"] += float(e.get("cpu_s") or 0.0)
        s_["wall_s"] += float(e.get("wall_s") or 0.0)
        s_["shards"] += 1
        s_["max_rss_bytes"] = max(s_["max_rss_bytes"], int(e.get("maxrss_bytes") or 0))
    sf = Path(start_file) if start_file else PRIV / "START.txt"
    if not sf.is_file() and (run_dir / "START.txt").is_file():
        sf = run_dir / "START.txt"
    try:
        start = sf.read_text().strip()
        elapsed_h = (t_now - _ts(start)) / 3600
    except (OSError, ValueError):
        start, elapsed_h = None, None
    cpu_h = sum(r["cpu_s"] for r in by_role.values()) / 3600
    for d in list(by_role.values()) + list(by_stage.values()):
        for kk in ("cpu_s", "wall_s"):
            d[kk] = round(d[kk], 1)
    return {"at": now(), "start": start, "elapsed_h": None if elapsed_h is None else round(elapsed_h, 3),
            "semaphore_child_cpu_h": round(cpu_h, 3), "cpu_s_by_role": by_role, "released_holds": n_rel,
            "active_holds": active, "max_concurrent_holds": peak, "slot_limit": SLOTS,
            "slot_limit_respected": peak <= SLOTS, "unparsed_lines": sum(1 for e in sema + led if e.get("_unparsed")),
            "stage_ledger_cpu_s_by_stage": by_stage,
            "budget": {"cpu_h_limit": budget_cpu_h, "wall_h_limit": budget_wall_h,
                       "cpu_h_used_semaphore": round(cpu_h, 3), "cpu_h_remaining": round(budget_cpu_h - cpu_h, 3),
                       "wall_h_remaining": None if elapsed_h is None else round(budget_wall_h - elapsed_h, 3)},
            "not_captured": "agent processes and light commands outside the semaphore (not in SEMA_LOG.jsonl); the "
                            "stage ledger is a subset of the semaphore children (not added twice)"}


# ------------------------------------------------------------------ identity scan (read-only)
_HOME_PATH = re.compile(r"(/(?:Users|home)/[A-Za-z0-9._-]+|[A-Za-z]:\\\\?Users\\\\?[A-Za-z0-9._-]+)")
_VOL_PATH = re.compile(r"(/(?:Volumes)/[^\s'\"<>]+|/(?:private)/(?:var|tmp|etc)[^\s'\"<>]*)")
_HOME_REL = re.compile(r"(~/[A-Za-z0-9._-]+|\$HOME/[A-Za-z0-9._-]+)")
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_CREDENTIALS = (
    ("aws_access_key_id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("aws_secret_assignment", re.compile(r"aws_secret_access_key\s*[=:]", re.I)),
    ("private_key_block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{20,})")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}")),
    ("api_key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}")),
    ("bearer_token", re.compile(r"Authorization:\s*Bearer\s+\S{10,}", re.I)),
    ("secret_assignment", re.compile(r"(?:password|passwd|secret|api[_-]?key|access[_-]?token)\s*[=:]\s*['\"][^'\"\s]{8,}"
                                     r"['\"]", re.I)))
_RECORD = (
    ("adult_record_line", re.compile(r"(?=.*\b(?:Male|Female)\b)(?=.*(?:<=|>)50K)")),
    ("per_row_array", re.compile(r"\"(?:row_ids?|rows|sex|SEX|y|labels?|unit|group|record)s?\"\s*:\s*\[\s*-?\d+"
                                 r"(?:\s*,\s*-?\d+(?:\.\d+)?){49,}")),
    ("ssn_like", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")))
_IDENTIFYING = ("home_path", "volume_or_private_path", "user_name", "email", "credential", "per_person_record")


def identity_scan(files, extra_terms=(), allow_emails=("git@github.com",), max_per_file=50, user_name=None):
    """Scan text files (and their path names) for identifying directory names (home / volume / private paths, the
    local user name), home-relative paths (review), emails, credential-like strings and per-person record patterns.
    Returns {"findings": [...], "skipped_binary": [...], "files": n, "identifying": n}. Excerpts are masked so the
    findings themselves can be published; file names are reported relative to the worktree or as <outside>."""
    user = HOME_NAME if user_name is None else user_name
    files = list(files)
    findings, skipped = [], []

    def name_of(p):
        try:
            return str(Path(p).resolve().relative_to(I.WT.resolve()))
        except ValueError:
            return "<outside>/" + Path(p).name

    def mask(s):
        if any(rx.search(s) for _, rx in _RECORD):
            return "<per-person record pattern; masked>"
        for _, rx in _CREDENTIALS:
            s = rx.sub("<CREDENTIAL>", s)
        s = _HOME_PATH.sub("<HOMEPATH>", s)
        s = _VOL_PATH.sub("<PATH>", s)
        s = _EMAIL.sub("<EMAIL>", s)
        if user:
            s = s.replace(user, "<USER>")
        for t in extra_terms:
            s = s.replace(t, "<TERM>")
        return s[:160]

    def check(text, fname, line):
        out = []
        if _HOME_PATH.search(text):
            out.append("home_path")
        if _VOL_PATH.search(text):
            out.append("volume_or_private_path")
        if user and re.search(r"(?<![A-Za-z0-9])" + re.escape(user) + r"(?![A-Za-z0-9])", text):
            out.append("user_name")
        if any(t and t in text for t in extra_terms):
            out.append("extra_term")
        if any(m.group(0) not in allow_emails for m in _EMAIL.finditer(text)):
            out.append("email")
        out += ["credential:" + n for n, rx in _CREDENTIALS if rx.search(text)]
        out += ["per_person_record:" + n for n, rx in _RECORD if rx.search(text)]
        if _HOME_REL.search(text):
            out.append("home_relative_path")
        return [{"file": fname, "line": line, "kind": k,
                 "severity": "identifying" if k.split(":")[0] in _IDENTIFYING + ("extra_term",) else "review",
                 "excerpt": mask(text.strip())} for k in out]

    for f in files:
        p = Path(f)
        fname = name_of(p)
        hits = check(fname, fname, 0)                    # the file's own (worktree-relative) name
        try:
            b = p.read_bytes()
        except OSError as e:
            skipped.append({"file": fname, "reason": type(e).__name__})
            continue
        if b"\0" in b[:8192]:
            skipped.append({"file": fname, "reason": "binary"})
            findings += hits
            continue
        for n, ln in enumerate(b.decode("utf-8", "replace").splitlines(), 1):
            hits += check(ln, fname, n)
            if len(hits) >= max_per_file:
                hits.append({"file": fname, "line": n, "kind": "truncated", "severity": "review",
                             "excerpt": f"more than {max_per_file} findings"})
                break
        findings += hits
    return {"at": now(), "files": len(files),
            "findings": findings, "identifying": sum(1 for x in findings if x["severity"] == "identifying"),
            "review": sum(1 for x in findings if x["severity"] == "review"), "skipped_binary": skipped}


# ------------------------------------------------------------------ owned processes (never killed)
def _cputime_s(s):
    d, _, rest = s.rpartition("-")
    parts = [float(x) for x in rest.split(":")]
    sec = 0.0
    for x in parts:
        sec = sec * 60 + x
    return sec + (int(d) * 86400 if d else 0)


def owned_processes(patterns=OWNED_PATTERNS, ps_output=None):
    """Processes whose command line contains one of `patterns` (hcal.run / hcal.sema / hcal.assess / hcal.verify, and
    the by-path form hcal/sema.py). Listing only: this module has no kill path. Commands are scrubbed."""
    if ps_output is None:
        r = subprocess.run(["ps", "-axww", "-o", "pid=,ppid=,etime=,time=,rss=,pcpu=,command="], capture_output=True,
                           text=True, timeout=30)
        ps_output = r.stdout
    me = {os.getpid(), os.getppid()}
    out = []
    for ln in ps_output.splitlines():
        parts = ln.split(None, 6)
        if len(parts) < 7 or not any(p in parts[6] for p in patterns):
            continue
        if parts[6].lstrip().startswith(("ps ", "grep ")):
            continue
        pid = int(parts[0])
        try:
            cpu_s = _cputime_s(parts[3])
        except ValueError:
            cpu_s = None
        out.append({"pid": pid, "ppid": int(parts[1]), "elapsed": parts[2], "cpu_s": cpu_s,
                    "rss_mib": round(int(parts[4]) / 1024, 1), "pcpu": float(parts[5]),
                    "this_closeout_or_its_wrapper": pid in me,
                    "command": scrub_safe(parts[6]).replace(str(Path.home()), "<HOME>")[:240]})
    return {"at": now(), "patterns": list(patterns), "processes": out, "count": len(out),
            "rss_mib_total": round(sum(p["rss_mib"] for p in out), 1), "killed": "nothing (listing only)"}


# ------------------------------------------------------------------ status report (30-60 min cadence)
def _units_by_prefix(units):
    out = {}
    for d in (sorted(Path(units).iterdir()) if Path(units).is_dir() else []):
        if d.is_dir() and (d / "COMPLETE.json").is_file():
            pre = d.name.split("__")[0]
            out[pre] = out.get(pre, 0) + 1
    return out


def _last_stage(run_dir):
    st = None
    for e in _jsonl(Path(run_dir) / "ACTIVITY_LOG.jsonl"):
        m = re.match(r"^(start|end) (\S+)$", str(e.get("event", "")))
        if m:
            st = {"event": m.group(1), "stage": m.group(2), "at": e.get("at"), "shard": e.get("shard")}
    return st


def status_report(expected=None, findings=None, run_dir=None, priv=None):
    """The periodic status (read-only): completed / remaining units, active processes, CPU / wall / memory, free disk,
    stage and unresolved findings. expected = {unit prefix: registered count} (from the queue manifest)."""
    priv = Path(priv or PRIV)
    run_dir = Path(run_dir or priv / "run")
    done = _units_by_prefix(run_dir / "units")
    rem = {k: max(int(v) - done.get(k, 0), 0) for k, v in (expected or {}).items()}
    try:
        from hcal import sema as SM
        slots = SM.status() if Path(SM.RUN) == run_dir else "not the configured store"
    except Exception as e:                                                           # noqa: BLE001
        slots = f"unavailable ({type(e).__name__})"
    try:
        mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=10).stdout)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        mem = None
    procs = owned_processes()
    disk_at = priv if priv.exists() else priv.parent
    return public({"at": now(), "stage": _last_stage(run_dir), "units_completed": done, "units_remaining": rem or None,
                   "admitted_units": len(list((priv / "admitted" / "units").glob("*"))) if
                   (priv / "admitted" / "units").is_dir() else 0,
                   "semaphore_slots": slots, "processes": procs,
                   "accounting": accounting(run_dir, priv / "START.txt"),
                   "memory": {"owned_rss_mib": procs["rss_mib_total"], "limit_gib": 8,
                              "machine_gib": None if mem is None else round(mem / 2 ** 30, 1)},
                   "free_disk_gib": round(free_gib(disk_at), 2) if disk_at.exists() else None,
                   "free_disk_floor_gib": MIN_FREE_GIB, "drive": drive_status(),
                   "unresolved_findings": list(findings or [])})


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m hcal.closeout")
    ap.add_argument("job", choices=("status", "drive", "accounting", "processes", "scan", "copy", "offdevice"))
    ap.add_argument("files", nargs="*")
    ap.add_argument("--dest-name", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("REFUSED: OMP_NUM_THREADS must be 1")
    if a.job == "status":
        out = status_report()
    elif a.job == "drive":
        out = drive_status()
    elif a.job == "accounting":
        out = accounting()
    elif a.job == "processes":
        out = owned_processes()
    elif a.job == "scan":
        out = identity_scan(a.files)
    elif a.job == "copy":
        out = same_device_copy(a.dest_name, dry_run=a.dry_run)
    else:
        out = off_device_copy(dry_run=a.dry_run)
    print(json.dumps(public(out), indent=1))


if __name__ == "__main__":
    main()
