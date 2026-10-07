"""[lra port of lcr/closeout.py at 091afc2: lcr->lra renames; later edits are listed in PORT_LOG.md]
Custody closeout for the Adult learned-decoder constrained-release study (lra; owner role F, claims and custody).

Adapted from cbp/closeout.py at the cbp tip 7f3ec67 (IMPORTED, not edited: the drive probe by content, the versioned copy
with uncached read-back, verify_sums, the teacher restore by own forward pass, the D0 policy re-encode and deployment
from the copy and the attacker restore are the pinned dpc / qpc / cbp code). New here (lcr, kept):
  - the lra store and its bundled pinned input (the lra store holds no copy of adult_jcv.npz);
  - the D1 restore: the LEARNED DECODER re-certified from the copy alone (lra.decoder.load_decoder_pair with
    verify_solve: hash, binding to the copied policy pair, release invariants, every supervised token re-solved
    bitwise from its stored statistics; and those statistics recomputed independently from the copy's restored teacher,
    the bundled input's OSF_DEFENSE_FIT true task labels and the re-encoded tokens), the release re-encoded with
    lra.decoder.encode_d1 and compared bitwise with the copied and live releases, and an optional deployment from the
    copy;
  - the PENDING cbp custody: the source documented `python -m cbp.closeout all --targets
    <PRIVATE_CACHE>/cbp_v1/run/closeout_targets.json`, run IN-PROCESS with the cbp / qpc / dpc / osf / smf code of this
    worktree verified byte-identical to the cbp tip, every public receipt redirected into THIS study's
    provenance/cbp_custody/, the closed results trees (cbp, qpc, dpc, osf, smf) hashed before and after;
  - the job sequencing and the lra receipts.
New in lra (role F, 2026-10-07):
  - THIS study fits the learned decoder and its attackers on Adult. Once the store holds an Adult D1 unit (dec__ /
    new__, i.e. the science stages ran), the learned-decoder and attacker restores are REQUIRED: they can no longer be
    marked not applicable, and a missing or failing restore makes restore_all_pass false (science_state()).
  - ROLE TARGETS with truthful statuses (load_targets / check_targets_against_selection): Q, P* (or its registered
    fallback), the constrained candidate, the decoder-only (same-map D0) baseline and the sequential winner, each with
    a status from ROLE_STATUS; units are restored from the copy, the decoder-only baseline's tokens must equal its
    paired release's tokens bitwise (identical partition), and each nominated / fallback status must agree with the
    inner selection record (run/selection.json) when it exists.
  - PREDECESSOR CUSTODY RULE (prompt sec. 15). The cbp custody (and through it qpc / dpc / osf / smf) runs ONLY after
    THIS study's assessment opening: lra_assessment_opened() requires the EVALUATION_LOCK committed and byte-identical
    on origin AND at least one hash-complete assessment unit (outer__) whose record binds that lock's sha256. The
    alternative route ("proven from code that it cannot read assessment labels earlier") is NOT available: the chain
    reaches osf.closeout.backup, which calls osf.assess.open_assessment / load_unsealed on OSF_DEVELOPMENT_ASSESSMENT,
    this study's assessment role, to refit osf's final attacker (CODE_PROOF_OF_NO_EARLY_READ). Until the opening, the
    custody is recorded PENDING with the owner's command, and inside any custody run osf.assess.open_assessment and
    osf.data.load(unseal=True) are refused unless the opening is verified.
  - the study label in custody receipts comes from the new correctness gate: ENGINEERING_BLOCKED_NOT_RUN when
    ENGINEERING_GATE_RESULT.json says ENGINEERING_BLOCKED, otherwise the label of the locked inference output
    (run/inference.json) once it exists; the historical lcr MECHANISM_GATE_NOT_MET never becomes this study's label.
  - store namespaces covered by the copy, the inventory and the restore index: pol__, dec__, new__ (with trace.json),
    d0s__ (same-map D0 diagnostics), diag__, tea__, ref__, fine__, cor__ (correctness fixtures), aud__ (inner audits),
    outer__ (assessment units), admitted/, inputs/, run/attempts/ and the ledgers.

The drive is identified by CONTENT, never by a volume name: a mounted volume qualifies iff
<volume>/private_smf_v1_20261005/SHA256SUMS hashes to the value the closed smf study recorded (read at the pinned qpc
source commit). The boot volume and the unrelated read-only installer image "BackgroundSyncService Setup" are skipped by
name BEFORE any filesystem call, so nothing on that image is ever touched. Volume names are never recorded.

Jobs (each has --dry-run, which writes nothing; `--plan-only` with --dry-run also skips the read-only restore rehearsal,
i.e. loads no data). Every job that loads data or restores runs under the SHARED lra semaphore:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m lra.sema
        --label F:closeout -- env OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m lra.closeout <job> [...]

    status                                         drive probe, lock states, store self-containment (writes nothing)
    cbp-custody [--cbp-targets T] [--dry-run]      the pending source custody (see above)
    backup [--dest D] [--targets T] [--dry-run] [--plan-only]
    all [--targets T] [--cbp-targets T] [--dry-run] [--plan-only]
    refresh [--copy-root R] [--targets T] [--dry-run]
                                                   bring the newest same-device copy up to date (new files, grown
                                                   ledgers); deletes nothing, keeps the previous SHA256SUMS, re-reads
                                                   uncached; a changed restore input is recorded as STALE evidence

`all` with the drive present, in order:
  (1) the pending cbp custody (requires THIS study's assessment opening, lra_assessment_opened(); otherwise recorded
      PENDING with the seal reason and the cbp owner's command, never run): the cbp drive copy and restore from it, which
      itself runs the pending qpc / dpc / osf / smf custody. Receipts go to
      provenance/cbp_custody/ (qpc_custody/, qpc_custody/dpc_custody/, qpc_custody/predecessor_custody/);
  (2) this study: <DRIVE_ROOT>/private_lra_v1_<UTC date>[_vN]/{lra_v1, dependencies/jcv_v1/inputs/adult_jcv.npz,
      SHA256SUMS} (a new folder; never reused);
  (3) every file of every folder created by (1)-(2) and of every prior known private_<study>_v1_<date> folder (lra, lcr,
      cbp, qpc, dpc, osf, smf) re-read UNCACHED (F_NOCACHE; an uncached read, not a physical cold-disk read);
  (4) restore FROM THE lra COPY ALONE: the U teacher (own forward pass), the learned decoder (re-certified), every role
      target (Q, P* or its fallback, the constrained candidate, the decoder-only baseline, the sequential winner) and
      one selected attacker (refit on AUDIT_FIT; no assessment label is read);
  (5) receipts: BACKUP_VERIFICATION.json, RESTORE_INDEX.json, provenance/cbp_custody/*, BACKUP_RECORD.json in the folder.
`all` with the drive absent: the cbp custody receipt records PENDING with the exact command (no redundant cbp same-device
copy: cbp already holds a verified same-device copy); a versioned SAME-DEVICE copy
<PRIVATE_CACHE>/lra_v1_local_copy_<UTC date>[_vN], verified uncached and restored from, with status
LOCAL_SAME_DEVICE_COPY_VERIFIED_OFF_DEVICE_BACKUP_PENDING and the exact pending commands. A same-device copy is never
called off-device custody or a drive restore. Nothing anywhere is deleted or moved; other studies' stores are only read.

targets.json (written by the lead after the assessment; <PRIVATE_CACHE>/lra_v1/run/closeout_targets.json):
    {"seed": 1,
     "roles": {                                    # REQUIRED in a targets file: all five ROLES, each with a status
       "Q":                     {"status": "NOMINATED" | "DESCRIPTIVE_FALLBACK" | "FIXED" | "INVALID",
                                 "unit": "pol__s1__U_DIRECT-TASK_i8o64", "config": "U|DIRECT-TASK|i8o64"},
       "P*":                    {"status": ..., "unit": "dec__s1__<safe>" | "new__s1__<safe>" | "pol__s1__<safe>",
                                 "config": "<privacy-trained config id>", "family": "...", "construction": "...",
                                 "reason": "<selection reason; required unless NOMINATED / FIXED>"},
       "constrained candidate": {"status": ..., "unit": "new__s1__U_K-<ARM>_i8o64_D1", "config": "U|K-<ARM>|i8o64|D1",
                                 "selection_role": "N*" | "J*"},
       "decoder-only baseline": {"status": "FIXED" | ..., "unit": "pol__s1__<D0 id>" | "d0s__s1__<safe>_D0SAME",
                                 "config": "...", "paired_with": "P*" | "constrained candidate" | ...},
       "sequential winner":     {"status": ..., "unit": "new__s1__U_K-SEQ-12_i8o64_D1" | ..., "config": "..."}},
     "policies": {"<label>": "<unit>", ...},       # optional extra restore targets (e.g. "U baseline": "tea__s1__U")
     "deploy": "lra.deploy:<fn>",                                        # optional
     "not_applicable": {"protected map": "<reason>", "learned decoder": "<reason>", "attacker": "<reason>"},
                                                                         # optional; refused once the science ran for
                                                                         # the learned decoder and the attacker
     "attacker": {"fn": "lra.<module>:<refit function>", "kwargs": {...},
                  "saved": {"unit": "<outer unit>", "file": "preds.npz", "key": "<array>"}, "tolerance": 0.0}}
ROLE_STATUS (truthful status vocabulary): NOMINATED (an eligible inner nominee), DESCRIPTIVE_FALLBACK (no eligible
nominee; the registered descriptive fallback, ineligible), FIXED (fixed by the registration, never nominated: Q, the
same-map D0 decoder-only baseline, an inner-ranked sequential arm), INVALID (technical failure of the role), NOT_RUN (the
work that would produce it did not run), ABSENT (no such release exists). NOMINATED / DESCRIPTIVE_FALLBACK / FIXED need a
unit and a config; NOT_RUN / ABSENT forbid a unit; everything except NOMINATED / FIXED needs a reason. A role with no unit
is reported NOT_APPLICABLE (<status>: <reason>) in the required restores, never PASS and never silently PENDING.
Legacy labels in "policies" beginning "P*" or "fallback" are the protected map and "Q" is Q. A class listed in
not_applicable is reported as NOT_APPLICABLE (<reason>) instead of PENDING (never for the U teacher, refused if a target
for it is also given, and refused for the learned decoder and the attacker once the science ran). pol__ and d0s__ units
are D0 codes (qpc formats, mean-teacher decoder); dec__, new__ and diag__ units are D1 codes (policy.json token map +
decoder.json + release.npz). Entry points must be lra.*, cbp.* or qpc.*.
"""
from __future__ import annotations

import argparse
import contextlib
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
from lra.run import _finite
from qpc import closeout as QC

HOME = Path.home()
WT = Path(__file__).resolve().parents[1]
REL = "results/pcrl_adult_learned_decoder_release_v1"
PKG = WT / REL
PROV = PKG / "provenance"
CBP_CUSTODY_DIR = PROV / "cbp_custody"
CACHE = HOME / "PCRL_eval_cache_private"
SRC = CACHE / "lra_v1"
CBP_SRC = CACHE / "cbp_v1"
CBP_TARGETS = CBP_SRC / "run" / "closeout_targets.json"
LRA_TARGETS = SRC / "run" / "closeout_targets.json"
INPUT = CACHE / "jcv_v1" / "inputs" / "adult_jcv.npz"
INPUT_SHA = "e0d9e54af780f30788ee29cfe6795ec82cbdcadc127b1978c69a3891485d2f12"
DEP_REL = "dependencies/jcv_v1/inputs/adult_jcv.npz"
CBP_TIP = "7f3ec67b2ecd86d474e2ff27167091af9923f572"
CBP_BRANCH = "research/pcrl-confidence-budgeted-privacy-v1"
BRANCH = "research/pcrl-adult-learned-decoder-release-v1"
VOLUMES_ROOT = DC.VOLUMES_ROOT
SKIP_VOLUMES = DC.SKIP_VOLUMES
DRIVE_FOLDER = "private_lra_v1_{date}"
LOCAL_FOLDER = "lra_v1_local_copy_{date}"
KNOWN_FOLDERS = re.compile(r"^private_(lra|lcr|cbp|qpc|dpc|osf|smf)_v1(_custody_supplement)?_\d{8}(_v\d+)?$")
GATE_RESULT = "ENGINEERING_GATE_RESULT.json"
BLOCKED_VERDICT = "ENGINEERING_BLOCKED"
BLOCKED_LABEL = "ENGINEERING_BLOCKED_NOT_RUN"
LOCK_FILE = "EVALUATION_LOCK.json"
ROLES = ("Q", "P*", "constrained candidate", "decoder-only baseline", "sequential winner")
ROLE_STATUS = {
    "NOMINATED": "an eligible inner nominee (selection status NOMINEE)",
    "DESCRIPTIVE_FALLBACK": "no eligible nominee: the registered descriptive fallback (ineligible; descriptive only)",
    "FIXED": "fixed by the registration, never nominated (Q; the same-map D0 decoder-only baseline; an inner-ranked "
             "sequential arm)",
    "INVALID": "technical failure of the role (reason required)",
    "NOT_RUN": "the work that would produce it did not run (reason required)",
    "ABSENT": "no such release exists under the registration (reason required)"}
UNIT_STATUSES = ("NOMINATED", "DESCRIPTIVE_FALLBACK", "FIXED")
NO_UNIT_STATUSES = ("NOT_RUN", "ABSENT")
ROLE_ALLOWED = {"Q": ("NOMINATED", "DESCRIPTIVE_FALLBACK", "FIXED", "INVALID"),
                "P*": ("NOMINATED", "DESCRIPTIVE_FALLBACK", "INVALID", "NOT_RUN", "ABSENT"),
                "constrained candidate": ("NOMINATED", "DESCRIPTIVE_FALLBACK", "INVALID", "NOT_RUN", "ABSENT"),
                "decoder-only baseline": ("FIXED", "INVALID", "NOT_RUN", "ABSENT"),
                "sequential winner": ("NOMINATED", "DESCRIPTIVE_FALLBACK", "FIXED", "INVALID", "NOT_RUN", "ABSENT")}
ROLE_CLASS = {"Q": "Q", "P*": "protected map", "constrained candidate": "constrained candidate",
              "decoder-only baseline": "decoder-only baseline", "sequential winner": "sequential winner"}
SELECTION_STATUS = {"NOMINATED": ("NOMINEE",), "DESCRIPTIVE_FALLBACK": ("NO_ELIGIBLE_NOMINEE", "NO_ELIGIBLE_COMPARATOR"),
                    "INVALID": ("INVALID_NOMINEE", "INVALID_COMPARATOR")}
CODE_PROOF_OF_NO_EARLY_READ = {
    "status": "NOT_AVAILABLE",
    "finding": "the predecessor chain DOES read this study's assessment labels: cbp.closeout.run_all -> "
               "qpc.closeout -> dpc.closeout -> osf.closeout.backup, which calls osf.assess.open_assessment and "
               "osf.assess.load_unsealed (osf.data.load(unseal=True)) on OSF_DEVELOPMENT_ASSESSMENT, the same role and "
               "rows as this study's assessment, to refit osf's final attacker (osf/closeout.py backup)",
    "consequence": "only the opening route is wired: the predecessor custody runs only after this study's assessment "
                   "opening (lra_assessment_opened)"}
TEACHERS = ("U", "RAW-J_b0.3")
TASK_KEYS = ("income", "occupation_group")           # D["y"] keys (dpc.utility.TASKS); recipient i <-> TASK_KEYS[i-1]
FIT_ROLE = "OSF_DEFENSE_FIT"
MIN_FREE_GIB = 5
CLOSED_RESULTS = {"cbp_results": WT / "results" / "pcrl_confidence_budgeted_privacy_v1",
                  "qpc_results": WT / "results" / "pcrl_confidence_capacity_v1",
                  "dpc_results": WT / "results" / "pcrl_decision_preserving_compression_v1",
                  "osf_results": WT / "results" / "pcrl_online_strength_frontier_v1",
                  "smf_results": WT / "results" / "pcrl_strength_matched_feedback_v1"}
SOURCE_PACKAGES = ("cbp", "qpc", "dpc", "osf", "smf")
STATUS_LOCAL = "LOCAL_SAME_DEVICE_COPY_VERIFIED_OFF_DEVICE_BACKUP_PENDING"
STATUS_DRIVE = "DRIVE_COPY_VERIFIED"
SEMA = ("OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. <python> -m lra.sema --label "
        "F:closeout -- env OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m lra.closeout")
CBP_DOCUMENTED = "python -m cbp.closeout all --targets <PRIVATE_CACHE>/cbp_v1/run/closeout_targets.json"
CBP_OWNER_COMMAND = ("cd <CBP_WORKTREE> && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. "
                     "<python> -m cbp.sema --label F:closeout -- env OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m "
                     "cbp.closeout all --targets <PRIVATE_CACHE>/cbp_v1/run/closeout_targets.json")
SEAL_REASON = ("the cbp custody reaches osf.closeout.backup, which opens osf's assessment (OSF_DEVELOPMENT_ASSESSMENT, the "
               "SAME rows as this study's assessment) to refit osf's final attacker; this study has not yet opened those "
               "rows under its own lock and keeps them sealed: it runs the cbp custody only after its own assessment "
               "opening (EVALUATION_LOCK on origin and the locked assessment units bound to it)")
PENDING = {
    "everything_when_the_drive_is_mounted": f"{SEMA} all --targets <PRIVATE_CACHE>/lra_v1/run/closeout_targets.json   "
                                            "(finds <DRIVE_ROOT> by content; the lra drive copy, the uncached re-read "
                                            "and the restores, in order; the cbp custody runs only after the lra "
                                            "assessment opening, otherwise it is recorded PENDING)",
    "lra_off_device_backup_and_restore": f"{SEMA} backup --dest <DRIVE_ROOT> --targets "
                                         "<PRIVATE_CACHE>/lra_v1/run/closeout_targets.json",
    "cbp_qpc_dpc_osf_smf_pending_custody_for_its_owner": CBP_OWNER_COMMAND + "   (the cbp documented `"
                                                         f"{CBP_DOCUMENTED}`, run in the cbp worktree at {CBP_TIP[:7]} "
                                                         "under cbp's own semaphore; it covers the cbp drive copy and "
                                                         "restore and, through cbp, the pending qpc / dpc / osf / smf "
                                                         "custody. It OPENS osf's assessment rows (the same rows as "
                                                         "this study's assessment) inside osf.closeout.backup; run it "
                                                         "only after this study's assessment opening)",
    "in_process_variant_from_this_worktree": f"{SEMA} cbp-custody   (the same cbp code, verified byte-identical to "
                                             f"{CBP_TIP[:7]}, with receipts redirected into {REL}/provenance/"
                                             "cbp_custody/; before this study's assessment opening it only records "
                                             "PENDING and osf's assessment stays sealed)",
    "lcr_off_device_backup_for_its_owner": "cd <LCR_WORKTREE> && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
                                           "MKL_NUM_THREADS=1 PYTHONPATH=. "
                                           "<python> -m lcr.sema --label F:closeout -- env OMP_NUM_THREADS=1 "
                                           "PYTHONPATH=. <python> -m lcr.closeout backup --dest <DRIVE_ROOT> --targets "
                                           "<PRIVATE_CACHE>/lcr_v1/run/closeout_targets.json   (the lcr owner's "
                                           "recorded pending command, lcr BACKUP_VERIFICATION.json; lcr holds a "
                                           "verified same-device copy only; it reads no assessment label)"}

now = DC.now
sha = DC.sha
jload = DC.jload
scrub = DC.scrub
scrub_safe = DC.scrub_safe
versioned = DC.versioned
tree_state = DC.tree_state
unit_verify = DC.unit_verify
cmp_arrays = DC.cmp_arrays
diskutil_probe = DC.diskutil_probe
verify_sums = QC.verify_sums
free_gib = QC.free_gib
top_folders = QC.top_folders
copy_study = CB.copy_study
uncached_read_capability = CB.uncached_read_capability
_is_prefix = CB._is_prefix


def write_public(path, obj):
    """Atomic, scrubbed, finite-or-null JSON (nonfinite -> null, then allow_nan=False) for every NEW public receipt."""
    txt = scrub(json.dumps(_finite(obj), indent=1, default=str, allow_nan=False) + "\n")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(txt)
    tmp.replace(path)


def _rel(p):
    try:
        return str(Path(p).resolve().relative_to(WT.resolve()))
    except ValueError:
        return "<outside worktree>"


# ------------------------------------------------------------------ drive detection, locks, code identity
def smf_marker_sha():
    return QC.smf_marker_sha()


def locate_drive(volumes_root=None, want=None):
    """(volume, evidence) by content (skipped names checked before any filesystem call; names never recorded)."""
    return DC.locate_drive(Path(volumes_root or VOLUMES_ROOT), want or smf_marker_sha())


def lra_lock_pushed():
    """(ok, reason): this study's EVALUATION_LOCK.json committed and byte-identical on origin (fetched now)."""
    from lra import data as LD
    return LD.evaluation_lock_pushed()


def lra_assessment_opened(src=None):
    """(ok, reason, evidence): THIS study's assessment opening. ok iff the EVALUATION_LOCK is committed and byte-identical
    on origin AND at least one hash-complete assessment unit outer__s{k}__* in the store records that lock's sha256
    (lra.assess.outer_unit writes record["evaluation_lock"] = {"commit", "sha256"} only after open_assessment())."""
    from jcv.finalize import unit_complete
    ok, why = lra_lock_pushed()
    ev = {"evaluation_lock_pushed": bool(ok), "evaluation_lock_state": why, "bound_assessment_units": 0}
    if not ok:
        return False, f"EVALUATION_LOCK not verified on origin ({why})", ev
    if not (PKG / LOCK_FILE).is_file():
        return False, "EVALUATION_LOCK.json is not in the package", ev
    lock_sha = sha(PKG / LOCK_FILE)
    ev["evaluation_lock_sha256"] = lock_sha
    units = Path(src or SRC) / "run" / "units"
    bound = []
    for d in (sorted(units.glob("outer__s*__*")) if units.is_dir() else []):
        if not (d.is_dir() and unit_complete(d)):
            continue
        try:
            r = jload(d / "record.json")
        except (OSError, ValueError):
            continue
        if (r.get("evaluation_lock") or {}).get("sha256") == lock_sha:
            bound.append(d.name)
    ev["bound_assessment_units"] = len(bound)
    if not bound:
        return False, ("EVALUATION_LOCK is on origin but no hash-complete assessment unit is bound to it: the "
                       "assessment has not been opened"), ev
    return True, f"assessment opened under the pushed EVALUATION_LOCK ({len(bound)} bound assessment units)", ev


def engineering_gate_state():
    """The NEW correctness-only gate (prompt sec. 9): the verdict in ENGINEERING_GATE_RESULT.json and the full
    lra.run.engineering_ready() check (lock-bound, on origin). Never hard-coded."""
    verdict = None
    try:
        verdict = jload(PKG / GATE_RESULT).get("verdict")
    except (OSError, ValueError):
        pass
    try:
        ok, why = _engineering_ready()
    except (SystemExit, Exception) as e:                                             # noqa: BLE001
        ok, why = False, f"engineering_ready() did not verify: {type(e).__name__}: {scrub_safe(e)[:160]}"
    return {"verdict": verdict, "engineering_ready": bool(ok), "state": why}


def _engineering_ready():
    from lra import run as R
    return R.engineering_ready()


def study_label(gate=None, src=None):
    """(label or None, source): ENGINEERING_BLOCKED_NOT_RUN when the gate is blocked; otherwise the label of the locked
    inference output (run/inference.json) once it exists. The lcr MECHANISM_GATE_NOT_MET is never returned."""
    gate = gate or engineering_gate_state()
    if gate["verdict"] == BLOCKED_VERDICT:
        return BLOCKED_LABEL, f"{GATE_RESULT} verdict {BLOCKED_VERDICT}: every Adult claim NOT RUN"
    if not gate["engineering_ready"]:
        return None, f"no study label yet: the engineering gate is not verified ({gate['state']})"
    p = Path(src or SRC) / "run" / "inference.json"
    try:
        lab = jload(p).get("label")
    except (OSError, ValueError):
        return None, "no study label yet: the locked inference has not run (run/inference.json absent)"
    if not lab:
        return None, "run/inference.json carries no label"
    return lab, "label from the locked inference output (run/inference.json)"


def science_state(src=None):
    """Whether this study's Adult science ran: hash-complete D1 units (dec__ fixed-map decodes, new__ fits) exist in the
    store. Once true, the learned decoder and an attacker are REQUIRED custody restores."""
    units = Path(src or SRC) / "run" / "units"
    n = {"dec": 0, "new": 0}
    for d in (sorted(units.iterdir()) if units.is_dir() else []):
        pre = d.name.split("__")[0]
        if pre in n and d.is_dir() and (d / "COMPLETE.json").is_file():
            n[pre] += 1
    return {"adult_d1_fixed_map_units": n["dec"], "adult_new_fit_units": n["new"],
            "science_ran": bool(n["dec"] or n["new"])}


def lra_assessment_state(src=None):
    """This study's state for custody receipts: the engineering gate, the study label, the lock and the opening."""
    ok, why, ev = lra_assessment_opened(src)
    gate = engineering_gate_state()
    lab, lab_src = study_label(gate, src)
    return {"engineering_gate": gate, "study_label": lab, "study_label_source": lab_src,
            "evaluation_lock_pushed": ev["evaluation_lock_pushed"], "evaluation_lock_state": ev["evaluation_lock_state"],
            "assessment_opened_by_this_study": bool(ok), "assessment_opening_state": why,
            "bound_assessment_units": ev["bound_assessment_units"],
            "historical_source_label": "lcr MECHANISM_GATE_NOT_MET (SOURCE_FIXTURE_GATE.json; never this study's label)"}


def source_code_identity():
    """Every module of the cbp / qpc / dpc / osf / smf packages in this worktree is byte-identical to its blob at the cbp
    tip, so the in-process run IS the source documented command."""
    diff, n = [], 0
    for top in SOURCE_PACKAGES:
        for p in sorted((WT / top).rglob("*.py")):
            rel = str(p.relative_to(WT))
            r = subprocess.run(["git", "-C", str(WT), "show", f"{CBP_TIP}:{rel}"], capture_output=True)
            n += 1
            if r.returncode != 0 or r.stdout != p.read_bytes():
                diff.append(rel)
    return {"files": n, "differing_from_cbp_tip": diff, "identical": not diff and n > 0, "pin": CBP_TIP}


def self_containment(src: Path):
    """What a restore from the copy alone needs: admitted teachers, the deployment inputs, the teacher units and the
    pinned input (bundled beside the store)."""
    need = [f"admitted/rel__s{k}__U/{f}" for k in (0, 1, 2) for f in ("model.pt", "head_0.joblib", "head_1.joblib",
                                                                        "release.npz", "COMPLETE.json")]
    need += ["inputs/deploy_input.npz", "inputs/schema.json"] + [f"run/units/tea__s{k}__U/teacher.npz" for k in (0, 1, 2)]
    missing = [r for r in need if not (Path(src) / r).is_file()]
    return {"required_present": len(need) - len(missing), "required": len(need), "missing": missing,
            "pinned_input_bundled_from": "<PRIVATE_CACHE>/jcv_v1/inputs/adult_jcv.npz", "pinned_input_present":
            INPUT.is_file()}


def status(volumes_root=None):
    vol, ev = locate_drive(volumes_root)
    files, nbytes = DC.inventory(SRC) if SRC.exists() else ([], 0)
    return {"at": now(), "diskutil_list_external": diskutil_probe(), "drive_with_verified_prior_copies": ev,
            "mounted": vol is not None, "lra_state": lra_assessment_state(),
            "cbp_evaluation_lock_pushed": CB.cbp_lock_pushed()[0], "store": {"files": len(files), "bytes": nbytes},
            "science": science_state(), "self_contained": self_containment(SRC),
            "cbp_targets_present": CBP_TARGETS.is_file(), "lra_targets_present": LRA_TARGETS.is_file()}


def closed_trees():
    return {k: tree_state(v) for k, v in CLOSED_RESULTS.items()}


# ------------------------------------------------------------------ (1) the pending cbp / qpc / dpc / osf / smf custody
@contextlib.contextmanager
def osf_assessment_sealed_until_lra_opening():
    """osf.closeout.backup opens osf's assessment (the SAME rows as this study's assessment) to refit its final attacker:
    keep it sealed until THIS study's assessment opening (lra_assessment_opened) and the cbp lock are verified. While
    sealed, both osf.assess.open_assessment and every osf.data.load(unseal=True) (the root of every unsealing path of
    the osf / qpc / cbp / lra loaders) refuse."""
    from osf import assess as AS
    from osf import data as OD
    ok_l, why_l, _ = lra_assessment_opened()
    ok_c, why_c = CB.cbp_lock_pushed()
    ok = ok_l and ok_c
    orig, orig_load = AS.open_assessment, OD.load
    if not ok:
        why = why_l if not ok_l else why_c

        def refuse(*a, **k):
            raise SystemExit(f"sealed by lra custody: {why}")

        def sealed_load(*a, **k):
            if k.get("unseal") or (len(a) > 1 and a[1]):
                raise SystemExit(f"sealed by lra custody: {why}")
            return orig_load(*a, **k)
        AS.open_assessment = refuse
        OD.load = sealed_load
    try:
        yield ok
    finally:
        AS.open_assessment = orig
        OD.load = orig_load


@contextlib.contextmanager
def redirected_cbp(out_dir: Path):
    """Rebind cbp.closeout's receipt folders into out_dir (qpc_custody/ below it), its closed-tree set to include the cbp
    results, its osf seal to this study's lock, and refuse any public write (cbp, qpc or dpc closeout) into a closed
    results tree. Every binding is restored afterwards, also on failure."""
    CBm = CB
    out_dir = Path(out_dir)
    closed = [Path(p).resolve() for p in CLOSED_RESULTS.values()]
    writes = []
    keys = ("PKG", "PROV", "QPC_CUSTODY_DIR", "CLOSED_RESULTS", "write_public", "osf_assessment_sealed_until_cbp_lock")
    saved = {"CB": {k: getattr(CBm, k) for k in keys}, "QC": QC.write_public, "DC": DC.write_public}
    base = saved["DC"]

    def guarded(path, obj):
        p = Path(path).resolve()
        if any(p == c or c in p.parents for c in closed):
            raise SystemExit("REFUSED: unredirected write into a closed results tree")
        writes.append(_rel(p))
        base(p, _finite(obj))
    CBm.PKG = out_dir
    CBm.PROV = out_dir
    CBm.QPC_CUSTODY_DIR = out_dir / "qpc_custody"
    CBm.CLOSED_RESULTS = dict(CLOSED_RESULTS)
    CBm.write_public = guarded
    QC.write_public = guarded
    DC.write_public = guarded
    CBm.osf_assessment_sealed_until_cbp_lock = osf_assessment_sealed_until_lra_opening
    try:
        yield writes
    finally:
        for k, v in saved["CB"].items():
            setattr(CBm, k, v)
        QC.write_public = saved["QC"]
        DC.write_public = saved["DC"]


def cbp_custody(dry_run=False, volumes_root=None, out_dir=None, cbp_targets=None, CBmod=None):
    out_dir = Path(out_dir or CBP_CUSTODY_DIR)
    vol, ev = locate_drive(volumes_root)
    ct = Path(cbp_targets or CBP_TARGETS)
    base = {"schema": "lra-cbp-custody-v1", "at": now(), "drive_detection": ev,
            "documented_command": f"OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m cbp.closeout all --targets "
                                  "<PRIVATE_CACHE>/cbp_v1/run/closeout_targets.json",
            "how": f"cbp.closeout.run_all called in-process from this worktree under the shared lra semaphore (cbp / qpc "
                   f"/ dpc / osf / smf modules byte-identical to the cbp tip {CBP_TIP[:7]}; checked before the call); "
                   "receipts redirected; run only after this study's assessment opening (EVALUATION_LOCK on origin "
                   "and the locked assessment units bound to it)",
            "rule": "prompt sec. 15: predecessor restore/custody only after this study's evaluation opening, or once "
                    "it is proven from code that it cannot read assessment labels earlier",
            "code_proof_of_no_early_read": CODE_PROOF_OF_NO_EARLY_READ,
            "covers": ["cbp drive copy and restore from it (U teacher, Q, P*, the J* fallback, attacker)",
                       "through cbp: the qpc custody (qpc drive copy + restore), the dpc off-device backup, the osf "
                       "drive copy, the smf drive restore and the two later smf logs"],
            "redirect": {"cbp BACKUP_VERIFICATION.json / RESTORE_INDEX.json": f"{REL}/provenance/cbp_custody/",
                         "qpc custody receipts": f"{REL}/provenance/cbp_custody/qpc_custody/",
                         "dpc custody receipts": f"{REL}/provenance/cbp_custody/qpc_custody/dpc_custody/",
                         "osf / smf predecessor receipts":
                             f"{REL}/provenance/cbp_custody/qpc_custody/predecessor_custody/"},
            "closed_results_never_written": [_rel(p) + "/" for p in CLOSED_RESULTS.values()],
            "historical_status": "cbp, qpc, dpc and osf off-device custody PENDING at their closeouts (drive absent); "
                                 "cbp holds a verified same-device copy (cbp_v1_local_copy_20261006); lcr left its cbp "
                                 "custody PENDING (its EVALUATION_LOCK was never written) and holds a verified "
                                 "same-device copy only (lcr_v1_local_copy_20261007)",
            "cbp_targets_present": ct.exists(), "osf_assessment_opened": False}
    state = lra_assessment_state()
    base["lra_assessment_state"] = state
    opened = state["assessment_opened_by_this_study"]
    if vol is None:
        reasons = ["external drive with the verified prior copies not mounted"]
        if not opened:
            reasons.append(SEAL_REASON)
        rec = {**base, "status": "PENDING", "reason": "; and ".join(reasons),
               "pending_command": PENDING["cbp_qpc_dpc_osf_smf_pending_custody_for_its_owner"],
               "gaps": ["cbp: same-device copy only (cbp_v1_local_copy_20261006); off-device PENDING",
                        "qpc: same-device copies only; off-device PENDING",
                        "dpc: same-device copy only; off-device PENDING",
                        "osf: no off-device copy; the smf drive restore and the two later smf logs PENDING"],
               "note": "no redundant cbp same-device copy is made: cbp already holds a verified same-device copy"}
        if dry_run:
            print(scrub(json.dumps(rec, indent=1)))
        else:
            write_public(out_dir / "STATUS.json", rec)
        return rec
    why_lock = state["assessment_opening_state"]
    if not opened:
        rec = {**base, "status": "PENDING", "reason": f"{SEAL_REASON} (lra assessment opening: {why_lock})",
               "pending_command": PENDING["cbp_qpc_dpc_osf_smf_pending_custody_for_its_owner"]}
        if dry_run:
            print(scrub(json.dumps(rec, indent=1)))
        else:
            write_public(out_dir / "STATUS.json", rec)
        return rec
    ident = source_code_identity()
    if not ident["identical"]:
        rec = {**base, "status": "REFUSED", "reason": "a pinned package differs from the cbp tip", "code_identity": ident}
        if not dry_run:
            write_public(out_dir / "STATUS.json", rec)
        return rec
    if dry_run:
        rec = {**base, "status": "DRY_RUN", "would_run": base["documented_command"], "code_identity": ident}
        print(scrub(json.dumps(rec, indent=1)))
        return rec
    CBm = CBmod or CB
    before = closed_trees()
    runs = {}
    with redirected_cbp(out_dir) as writes:
        try:
            bv = CBm.run_all(targets_file=str(ct) if ct.exists() else None, volumes_root=volumes_root,
                             out_pkg=out_dir)
            runs["cbp_run_all"] = {"status": bv.get("status"), "files": bv.get("files"),
                                   "uncached_readback_match": bv.get("uncached_readback_match"),
                                   "required_restores": bv.get("required_restores"),
                                   "restore_all_pass": bv.get("restore_all_pass"),
                                   "components": {k: (v or {}).get("status") for k, v in
                                                  (bv.get("components") or {}).items() if isinstance(v, dict)},
                                   "uncached_reread_all_pass": (bv.get("uncached_reread_of_drive_folders") or {}).get(
                                       "all_pass"),
                                   "closed_results_unchanged_per_cbp": bv.get("closed_results_unchanged")}
        except SystemExit as e:
            runs["cbp_run_all"] = {"status": "FAILED", "reason": scrub_safe(e)}
    after = closed_trees()
    ok_run = runs["cbp_run_all"].get("status") == STATUS_DRIVE
    rec = {**base, "status": ("RUN" if ok_run else "RUN_WITH_FAILURES") if before == after else
           "FAILED_CLOSED_RESULTS_CHANGED", "runs": runs, "redirected_writes": writes, "code_identity": ident,
           "lra_assessment_opening": why_lock, "closed_results_unchanged": before == after,
           "closed_results_state": after}
    write_public(out_dir / "STATUS.json", rec)
    return rec


# ------------------------------------------------------------------ (2) restore from the copy alone
def load_D_from_input(path: Path):
    """Sealed D through the pinned loader (lra.data -> qpc.data -> osf.data) reading ONLY `path` (the copy's bundled
    input); every role check re-run; assessment labels stay -1."""
    from lra import data as LD
    p = Path(path)
    if not p.is_file():
        raise SystemExit("REFUSED: the copy holds no bundled input file")
    if sha(p, nocache=True) != INPUT_SHA:
        raise SystemExit("REFUSED: the copy's input file fails its pinned hash")
    with DC.input_from(p):
        D = LD.load(verify=True, unseal=False)
    if not D.get("sealed"):
        raise SystemExit("REFUSED: assessment labels are not sealed in the restored data")
    return D


def _entry(spec):
    if not spec:
        return None
    mod, fn = spec.split(":")
    return getattr(importlib.import_module(mod), fn)


ENTRY = re.compile(r"^(lra|cbp|qpc)\.[a-z_]+:[A-Za-z_][A-Za-z0-9_]*$")
RESTORE_CLASSES = ("U teacher", "learned decoder", "protected map", "Q", "constrained candidate",
                   "decoder-only baseline", "sequential winner", "attacker")
UNIT = re.compile(r"^(pol|dec|new|tea|d0s|diag)__s\d__[A-Za-z0-9_.\-]+$")
UNIT_SEED = re.compile(r"^(?:pol|dec|new|tea|d0s|diag)__s(\d)__")
D0_PREFIX = ("pol__", "d0s__")
D1_PREFIX = ("dec__", "new__", "diag__")
CONSTRAINED_UNIT = re.compile(r"^new__s\d__U_K-(LOCAL|SEQ-12|SEQ-21|JOINT-SINGLE|JOINT-PAIR)_i8o64_D1$")
Q_UNIT = re.compile(r"^pol__s\d__U_DIRECT-TASK_i8o64$")


def role_label(role, entry):
    """The restore-check label of a role target: '<role> [<status>] <config or unit>' (starts with the role name)."""
    return f"{role} [{entry['status']}] {entry.get('config') or entry.get('unit')}"


def _check_role(role, e, seed):
    """Structural truthfulness of one role entry (refuses: SystemExit)."""
    if not isinstance(e, dict):
        raise SystemExit(f"REFUSED: role {role!r} must be an object with a status")
    st = e.get("status")
    if st not in ROLE_STATUS:
        raise SystemExit(f"REFUSED: role {role!r} status {st!r} is not one of {sorted(ROLE_STATUS)}")
    if st not in ROLE_ALLOWED[role]:
        raise SystemExit(f"REFUSED: role {role!r} cannot have status {st} (allowed: {list(ROLE_ALLOWED[role])})")
    u, cid = e.get("unit"), e.get("config")
    if st not in ("NOMINATED", "FIXED") and not (isinstance(e.get("reason"), str) and e["reason"].strip()):
        raise SystemExit(f"REFUSED: role {role!r} status {st} needs a stated reason")
    if st in NO_UNIT_STATUSES and u:
        raise SystemExit(f"REFUSED: role {role!r} status {st} cannot name a unit")
    if st in UNIT_STATUSES and not (u and cid):
        raise SystemExit(f"REFUSED: role {role!r} status {st} needs a unit and a config")
    if not u:
        return
    if not UNIT.match(u):
        raise SystemExit(f"REFUSED: role {role!r} unit {u!r} is not an lra unit name")
    m = UNIT_SEED.match(u)
    if int(m.group(1)) != int(seed):
        raise SystemExit(f"REFUSED: role {role!r} unit {u!r} is not a seed-{seed} unit")
    if cid:
        tail = u.split("__", 2)[2]
        want = cid.replace("|", "_")
        if tail != want:
            raise SystemExit(f"REFUSED: role {role!r} unit {u!r} does not carry config {cid!r}")
    if role == "Q" and not Q_UNIT.match(u):
        raise SystemExit("REFUSED: Q is the fixed original D0 DIRECT-TASK i8o64 code (pol__s{k}__U_DIRECT-TASK_i8o64)")
    if role == "P*":
        if not u.startswith(("pol__", "dec__", "new__")):
            raise SystemExit("REFUSED: P* must be a registered code release (pol__ / dec__ / new__)")
        from lra import run as R
        try:
            p = R.parse_id(cid)
        except Exception:                                                            # noqa: BLE001
            raise SystemExit(f"REFUSED: P* config {cid!r} does not parse") from None
        if not (p.get("kind") == "policy" and p.get("privacy_trained")):
            raise SystemExit(f"REFUSED: P* config {cid!r} is not privacy-trained (P* is drawn from private arms only)")
    if role == "constrained candidate" and not CONSTRAINED_UNIT.match(u):
        raise SystemExit("REFUSED: the constrained candidate must be a new__ U_K-<ARM>_i8o64_D1 unit")
    if role == "constrained candidate" and st in ("NOMINATED", "DESCRIPTIVE_FALLBACK") and \
            e.get("selection_role") not in ("N*", "J*"):
        raise SystemExit("REFUSED: the constrained candidate must name its selection_role (N* or J*)")
    if role == "decoder-only baseline" and not u.startswith(D0_PREFIX):
        raise SystemExit("REFUSED: the decoder-only baseline must be a D0 (mean-teacher) unit: pol__ or d0s__")
    if role == "sequential winner" and not re.search(r"(^|[_|])(K-|W-)?SEQ-(12|21)([_|]|$)", cid or u):
        raise SystemExit("REFUSED: the sequential winner must be a SEQ-12 / SEQ-21 release")


def role_targets(t):
    """[(label, unit)] of every restore target: the role units (in ROLES order), then the extra policies."""
    out = [(role_label(r, e), e["unit"]) for r, e in ((r, (t.get("roles") or {}).get(r)) for r in ROLES)
           if isinstance(e, dict) and e.get("unit")]
    seen = {u for _, u in out}
    out += [(lab, u) for lab, u in (t.get("policies") or {}).items() if u not in seen]
    return out


def paired_unit(t):
    """(label, unit) of the release the decoder-only baseline is paired with (a role or an extra policy label)."""
    e = (t.get("roles") or {}).get("decoder-only baseline") or {}
    pw = e.get("paired_with")
    roles = t.get("roles") or {}
    if pw in roles and pw != "decoder-only baseline" and isinstance(roles[pw], dict) and roles[pw].get("unit"):
        return role_label(pw, roles[pw]), roles[pw]["unit"]
    u = (t.get("policies") or {}).get(pw) if pw else None
    return (pw, u) if u else (None, None)


def check_targets_against_selection(targets, selection):
    """Truthfulness against the inner selection record (run/selection.json; statuses[role] with status / config /
    descriptive_config): a NOMINATED role must be that role's NOMINEE config, a DESCRIPTIVE_FALLBACK its descriptive
    config with a non-nominee status, an INVALID role an INVALID_* status. Returns a list of mismatches."""
    if not selection:
        return []
    st = selection.get("statuses") or {}
    mm = []
    for role, e in (targets.get("roles") or {}).items():
        sel_role = {"Q": "Q", "P*": "P*"}.get(role) or (e.get("selection_role") if isinstance(e, dict) else None)
        if not sel_role or not isinstance(e, dict) or e.get("status") not in SELECTION_STATUS:
            continue
        s = st.get(sel_role)
        if s is None:
            mm.append(f"{role}: selection has no role {sel_role}")
            continue
        want = SELECTION_STATUS[e["status"]]
        if s.get("status") not in want:
            mm.append(f"{role}: status {e['status']} but selection {sel_role} is {s.get('status')}")
            continue
        cfg = s.get("config") if e["status"] == "NOMINATED" else s.get("descriptive_config")
        if e["status"] != "INVALID" and e.get("config") != cfg:
            mm.append(f"{role}: config {e.get('config')!r} but selection {sel_role} names {cfg!r}")
    return mm


def targets_truthfulness(targets, src=None):
    """The role statuses against the private inner selection record <store>/run/selection.json (when it exists)."""
    p = Path(src or SRC) / "run" / "selection.json"
    if not (targets.get("roles") or {}):
        return {"pass": True, "checked_against": None, "mismatches": [], "note": "no role targets"}
    if not p.is_file():
        return {"pass": True, "checked_against": None, "mismatches": [],
                "note": "run/selection.json absent (selection did not run): structural checks only"}
    try:
        sel = jload(p)
    except (OSError, ValueError) as e:
        return {"pass": False, "checked_against": "run/selection.json", "mismatches": [f"unreadable: {scrub_safe(e)}"]}
    mm = check_targets_against_selection(targets, sel)
    return {"pass": not mm, "checked_against": "run/selection.json", "mismatches": mm}


def load_targets(targets_file):
    if not targets_file or not Path(targets_file).exists():
        return {"policies": {}, "roles": {}, "note": "no targets.json: U teacher only; decoder, role targets and "
                                                     "attacker PENDING"}
    t = jload(targets_file)
    roles = t.get("roles")
    if not isinstance(roles, dict) or set(roles) != set(ROLES):
        raise SystemExit(f"REFUSED: targets.json must give exactly the roles {list(ROLES)} with truthful statuses")
    for r in ROLES:
        _check_role(r, roles[r], t.get("seed", 1))
    for lab, u in (t.get("policies") or {}).items():
        if not UNIT.match(u):
            raise SystemExit(f"REFUSED: target {lab!r} is not an lra unit name")
    if roles["decoder-only baseline"].get("unit"):
        _, pu = paired_unit(t)
        if not pu:
            raise SystemExit("REFUSED: the decoder-only baseline must be paired_with a role or policy label that names "
                             "a unit")
        if not pu.startswith(D1_PREFIX):
            raise SystemExit("REFUSED: the decoder-only baseline must pair with a D1 (learned-decoder) release")
    for spec in (t.get("deploy"), (t.get("attacker") or {}).get("fn")):
        if spec and not ENTRY.match(spec):
            raise SystemExit("REFUSED: entry points must be lra / cbp / qpc module functions")
    na = t.get("not_applicable") or {}
    if not isinstance(na, dict) or not all(isinstance(v, str) and v for v in na.values()):
        raise SystemExit("REFUSED: not_applicable must map restore classes to a stated reason")
    bad = sorted(set(na) - set(RESTORE_CLASSES) | ({"U teacher"} & set(na)))
    if bad:
        raise SystemExit(f"REFUSED: not_applicable classes {bad} (the U teacher restore is always required)")
    labels = list((t.get("policies") or {}))
    clash = [c for c, pre in (("protected map", ("P*", "fallback")), ("Q", ("Q",))) if c in na and
             any(lb.startswith(pre) for lb in labels)]
    clash += [ROLE_CLASS[r] for r in ROLES if ROLE_CLASS[r] in na and roles[r].get("unit")]
    if "attacker" in na and (t.get("attacker") or {}).get("fn"):
        clash.append("attacker")
    if "learned decoder" in na and any(u.startswith(D1_PREFIX) for _, u in role_targets(t)):
        clash.append("learned decoder")
    if clash:
        raise SystemExit(f"REFUSED: {sorted(set(clash))} marked not applicable but a target for it is given")
    return t


def _unit_seed_teacher(units: Path, name: str):
    rec = jload(units / name / "record.json") if (units / name / "record.json").exists() else {}
    m = re.match(r"(?:pol|dec|new|d0s|diag)__s(\d)__", name)
    k = rec.get("seed", int(m.group(1)) if m else None)
    t = rec.get("teacher") or (rec.get("cfg") or {}).get("teacher") or "U"
    return t, (int(k) if k is not None else None), rec


def fitting_labels(D, i):
    """Row indices of OSF_DEFENSE_FIT and recipient i's true task labels there, through the pinned allowlist
    (procedure 'fitting'; the supervised use registered for this study)."""
    from qpc import data as QD
    ix = np.asarray(QD.labels_for(D, "fitting", FIT_ROLE))
    return ix, np.asarray(D["y"][TASK_KEYS[i - 1]])[ix]


def recompute_stats(tok, P, D, i, K, T):
    """Independent per-token sufficient statistics on the fitting rows (row-level accumulation): n_t, y_t, s_t."""
    ix, y = fitting_labels(D, i)
    y = np.asarray(y, dtype=np.int64)
    if y.size and (y.min() < 0 or y.max() >= K):
        raise ValueError(f"recipient {i}: fitting labels outside 0..{K - 1}")
    tf = np.asarray(tok, dtype=np.int64)[ix]
    Pf = np.asarray(P, dtype=np.float64)[ix]
    T = max(int(T), int(tf.max()) + 1 if tf.size else 0)
    n = np.bincount(tf, minlength=T).astype(np.int64)
    Y = np.bincount(tf * K + y, minlength=T * K).reshape(T, K).astype(np.float64)
    S = np.stack([np.bincount(tf, weights=Pf[:, k], minlength=T) for k in range(K)], 1)
    return n, Y, S


def recertify_decoder(dec_path: Path, pair, toks: dict, T: dict, D):
    """The learned decoder restored from the copy alone.
    (a) lra.decoder.load_decoder_pair(verify_solve=True): the decoder hash, its binding to the copied policy pair
        (teacher model and feature schema), the release invariants (class-dominant strict argmax, normalisation, D0
        fallback for tokens without fitting rows) and the registered solver re-solving EVERY supervised token from its
        stored statistics, bitwise;
    (b) the stored statistics equal statistics recomputed independently from the COPY: the restored teacher, the
        bundled input's OSF_DEFENSE_FIT true task labels and the re-encoded tokens (counts and label counts exact;
        teacher sums within lra.decoder.STATS_ROW_TOL x max(n_t, 1), the registered summation-order tolerance)."""
    from lra import decoder as DEC
    try:
        cid, d1, d2, dsha = DEC.load_decoder_pair(dec_path, pair, verify_solve=True)
    except (ValueError, KeyError, TypeError, AssertionError) as e:
        return {"status": "FAIL", "reason": f"decoder.json does not load or verify: {type(e).__name__}: "
                                            f"{scrub_safe(e)[:200]}"}, None
    res = {"config": cid, "decoder_sha256": dsha, "loaded_with": "lra.decoder.load_decoder_pair(verify_solve=True)",
           "re_solved_bitwise_from_stored_statistics": True, "bound_to_copied_policy_pair": True}
    ok = True
    for i, dec in ((1, d1), (2, d2)):
        try:
            n, Y, S = recompute_stats(toks[i], T[f"p{i}"], D, i, dec.K, dec.T)
        except ValueError as e:
            res[f"recipient_{i}"] = {"status": "FAIL", "reason": scrub_safe(e)}
            ok = False
            continue
        shape_ok = n.shape == dec.n.shape and Y.shape == dec.y.shape and S.shape == dec.s.shape
        tol = DEC.STATS_ROW_TOL * np.maximum(n, 1)[:, None] if shape_ok else None
        r = {"tokens": int(dec.T), "supervised_tokens": int((~dec.fallback).sum()),
             "fallback_tokens": int(dec.fallback.sum()), "shapes_match": bool(shape_ok),
             "counts_equal": bool(shape_ok and np.array_equal(n, dec.n)),
             "label_counts_equal": bool(shape_ok and np.array_equal(Y, dec.y)),
             "teacher_sum_max_abs_dev": float(np.abs(S - dec.s).max()) if shape_ok and S.size else None,
             "teacher_sum_within_tolerance": bool(shape_ok and np.all(np.abs(S - dec.s) <= tol)),
             "teacher_sum_tolerance": f"lra.decoder.STATS_ROW_TOL ({DEC.STATS_ROW_TOL:g}) x max(n_t, 1)",
             "stats_hash": dec.stats_hash()}
        r["status"] = "PASS" if (r["counts_equal"] and r["label_counts_equal"] and
                                 r["teacher_sum_within_tolerance"]) else "FAIL"
        ok = ok and r["status"] == "PASS"
        res[f"recipient_{i}"] = r
    res["status"] = "PASS" if ok else "FAIL"
    return res, (d1, d2)


def restore_d1_policy(copy: Path, live: Path, name: str, teachers: dict, D, deploy=None):
    """A D1 code unit restored from the copy: the learned decoder re-certified (recertify_decoder), the release
    re-encoded from the restored teacher with lra.decoder.encode_d1 (tokens, D1 vectors, decisions) and compared
    bitwise with the copied and live releases; decision preservation; optionally deployed from the copy."""
    from lra import decoder as DEC
    from qpc import release as RL
    units, lunits = Path(copy) / "run" / "units", Path(live) / "run" / "units"
    info = unit_verify(units, lunits, name)
    if not info.get("present_in_copy") or not (info["copy_complete_equals_live"] and info["copy_files_hash_ok_uncached"]):
        return {**info, "status": "FAIL", "reason": "D1 unit missing in the copy or does not verify"}
    t, k, rec = _unit_seed_teacher(units, name)
    info.update({"seed": k, "teacher": t, "config": rec.get("config"), "decoder": "D1"})
    T = teachers.get((t, k))
    if T is None:
        return {**info, "status": "FAIL", "reason": f"teacher {t} seed {k} was not restored from the copy"}
    if not (units / name / "decoder.json").is_file():
        return {**info, "status": "FAIL", "reason": "no decoder.json in the copied unit"}
    pair = RL.load_policy(units / name / "policy.json")
    toks = {}
    for i, pol in ((1, pair.p1), (2, pair.p2)):
        tok, _, _ = RL.encode(pol, T[f"p{i}"], T[f"d{i}"])
        toks[i] = np.asarray(tok)
    cert, decs = recertify_decoder(units / name / "decoder.json", pair, toks, T, D)
    info["decoder_recertified_from_copy"] = cert
    if decs is None:
        return {**info, "status": "FAIL", "reason": "the learned decoder did not re-certify from the copy"}
    mine = {}
    try:
        for i, (pol, dec) in ((1, (pair.p1, decs[0])), (2, (pair.p2, decs[1]))):
            tok, q, hard = DEC.encode_d1(pol, dec, T[f"p{i}"], T[f"d{i}"])
            mine.update({f"tok{i}": np.asarray(tok), f"q{i}": np.asarray(q), f"hard{i}": np.asarray(hard)})
    except (ValueError, AssertionError) as e:
        return {**info, "status": "FAIL", "reason": f"D1 re-encode refused: {scrub_safe(e)[:200]}"}
    keys = ["tok1", "q1", "hard1", "tok2", "q2", "hard2"]
    info["vs_copy_release"] = cmp_arrays(mine, np.load(units / name / "release.npz", allow_pickle=False), keys)
    info["vs_live_release"] = cmp_arrays(mine, np.load(lunits / name / "release.npz", allow_pickle=False), keys)
    info["decision_preserved_vs_restored_teacher"] = {f"recipient_{i}": bool(np.array_equal(mine[f"hard{i}"], T[f"d{i}"]))
                                                      for i in (1, 2)}
    info["token_states_emitted"] = {f"recipient_{i}": int(len(np.unique(mine[f"tok{i}"]))) for i in (1, 2)}
    ok = all(v["bitwise"] for kk in ("vs_copy_release", "vs_live_release") for v in info[kk].values())
    ok = ok and all(info["decision_preserved_vs_restored_teacher"].values()) and cert["status"] == "PASS"
    if deploy is not None:
        info["deployment_from_copy"] = _deploy_d1(deploy, copy, units / name, t, k, D, mine)
        ok = ok and info["deployment_from_copy"]["status"] == "PASS"
    else:
        info["deployment_from_copy"] = {"status": "NOT_RUN", "reason": "no lra deploy entry point in targets.json"}
    info["status"] = "PASS" if ok else "FAIL"
    return info


def _deploy_d1(fn, copy, unit_dir, t, k, D, mine):
    """fn(teacher_dir, unit_dir, X, schema_sha256, seed) -> (outputs, binding record), the lra deployment entry point
    (output keys tokens_i, probs_i, decision_i only); compared bitwise with the restored release."""
    from qpc import deploy as QDP
    try:
        names = [str(x) for x in D["feature_names"]]
        out, b = fn(Path(copy) / "admitted" / f"rel__s{k}__{t}", unit_dir, D["X"], QDP.schema_sha256(names), k)
    except SystemExit as e:
        return {"status": "FAIL", "reason": scrub_safe(e)}
    except Exception as e:                                                            # noqa: BLE001
        return {"status": "FAIL", "reason": f"{type(e).__name__}: {scrub_safe(e)[:200]}"}
    eq = {f"recipient_{i}": bool(np.array_equal(out[f"tokens_{i}"], mine[f"tok{i}"])
                                 and np.array_equal(out[f"probs_{i}"], mine[f"q{i}"])
                                 and np.array_equal(out[f"decision_{i}"], mine[f"hard{i}"])) for i in (1, 2)}
    allowed = {f"{x}_{i}" for x in ("tokens", "probs", "decision") for i in (1, 2)}
    return {"status": "PASS" if all(eq.values()) and b.get("binding") == "BOUND" and set(out) <= allowed else "FAIL",
            "binding": b.get("binding"), "outputs": sorted(out), "bitwise_equal_to_restored_release": eq}


def restore_all(copy: Path, live: Path, targets: dict, seed: int, input_path: Path):
    """Restore from the copy ALONE: `copy` is the copied lra_v1 folder and `input_path` the copy's bundled input; the
    live store is only the comparison target."""
    t0, c0 = time.time(), time.process_time()
    copy = Path(copy)
    D = load_D_from_input(Path(input_path))
    checks, teachers = {}, {}
    seeds = {seed}
    tlist = role_targets(targets)
    for _, u in tlist:
        m = UNIT_SEED.match(u)
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
    dep = _entry(targets.get("deploy"))
    for lab, u in tlist:
        if u.startswith(D0_PREFIX):
            checks[lab] = QC.restore_policy(copy, live, u, teachers, D)
        elif u.startswith(D1_PREFIX):
            checks[lab] = restore_d1_policy(copy, live, u, teachers, D, dep)
        else:
            m = re.match(r"tea__s(\d)__(.+)$", u)
            kk, t = int(m.group(1)), m.group(2)
            checks[lab] = {"unit": u, "status": checks.get(f"teacher {t} (seed {kk})", {}).get("status", "FAIL"),
                           "rebuild": f"teacher {t} seed {kk} restored above (own forward pass from the copy)"}
    base = (targets.get("roles") or {}).get("decoder-only baseline") or {}
    if base.get("unit"):
        plab, pu = paired_unit(targets)
        checks[SAME_MAP_CHECK] = same_map_tokens(copy, base["unit"], pu, plab)
    na = targets.get("not_applicable") or {}
    if "attacker" in na:
        checks["attacker"] = {"status": "NOT_APPLICABLE", "reason": na["attacker"]}
    else:
        checks["attacker"] = DC.restore_attacker(copy, live, targets.get("attacker"), D)
    del D
    return checks, {"wall_s": round(time.time() - t0, 1), "cpu_s": round(time.process_time() - c0, 1)}


SAME_MAP_CHECK = "same-map tokens (decoder-only baseline)"


def same_map_tokens(copy: Path, d0_unit: str, d1_unit, d1_label=None):
    """The decoder-only baseline and its paired D1 release must carry IDENTICAL token partitions: row_id, tok1, tok2,
    hard1 and hard2 of the two copied releases bitwise equal (only the decoded probabilities may differ)."""
    units = Path(copy) / "run" / "units"
    out = {"d0_unit": d0_unit, "paired_unit": d1_unit, "paired_label": d1_label}
    if not d1_unit:
        return {**out, "status": "FAIL", "reason": "no paired release unit"}
    try:
        a = np.load(units / d0_unit / "release.npz", allow_pickle=False)
        b = np.load(units / d1_unit / "release.npz", allow_pickle=False)
        eq = {k: bool(np.array_equal(a[k], b[k])) for k in ("row_id", "tok1", "tok2", "hard1", "hard2")}
        qdiff = {k: bool(not np.array_equal(a[k], b[k])) for k in ("q1", "q2")}
    except (OSError, KeyError, ValueError) as e:
        return {**out, "status": "FAIL", "reason": f"{type(e).__name__}: {scrub_safe(e)[:160]}"}
    out.update({"identical": eq, "decoded_vectors_differ": qdiff})
    out["status"] = "PASS" if all(eq.values()) else "FAIL"
    return out


def required_targets_status(statuses, checks=None, not_applicable=None, roles=None, science_ran=False):
    """The required restore classes: U teacher, learned decoder, the five role targets (protected map = P* or its
    registered fallback, Q, constrained candidate, decoder-only baseline, sequential winner) and the attacker. A role
    whose truthful status names no unit is NOT_APPLICABLE (<status>: <reason>); once the science ran, a missing
    learned-decoder or attacker restore is FAIL, never PENDING."""
    checks = checks or {}
    roles = roles or {}
    have_u = any(k.startswith("teacher U ") and v == "PASS" for k, v in statuses.items())
    pol = {k: v for k, v in statuses.items() if not k.startswith("teacher ") and k not in ("attacker", SAME_MAP_CHECK)}
    d1 = {k: (checks.get(k) or {}).get("decoder_recertified_from_copy", {}).get("status") for k in pol
          if "decoder_recertified_from_copy" in (checks.get(k) or {})}
    dec = ("PASS" if d1 and all(v == "PASS" for v in d1.values()) else "FAIL" if d1 else
           "FAIL (the science ran but no learned-decoder target was restored)" if science_ran else
           "PENDING (no D1 target)")
    out = {"U teacher": "PASS" if have_u else "FAIL",
           "learned decoder": dec,
           "protected map": next((v for k, v in pol.items() if k.startswith(("P*", "fallback"))),
                                 "PENDING (no P* / fallback target)"),
           "Q": next((v for k, v in pol.items() if k.startswith("Q")), "PENDING (no Q target)")}
    for r in ("constrained candidate", "decoder-only baseline", "sequential winner"):
        out[r] = next((v for k, v in pol.items() if k.startswith(r)), f"PENDING (no {r} target)")
    if SAME_MAP_CHECK in statuses and out["decoder-only baseline"] == "PASS" and statuses[SAME_MAP_CHECK] != "PASS":
        out["decoder-only baseline"] = "FAIL (tokens differ from the paired D1 release)"
    att = statuses.get("attacker", "PENDING")
    out["attacker"] = ("FAIL (the science ran but no attacker was restored)" if science_ran and att != "PASS" and
                       not str(att).startswith("FAIL") else att)
    for r, e in roles.items():
        c = ROLE_CLASS.get(r)
        if c and isinstance(e, dict) and not e.get("unit"):
            out[c] = f"NOT_APPLICABLE ({e.get('status')}: {e.get('reason')})"
    for c, why in (not_applicable or {}).items():
        if c in out and c != "U teacher" and str(out[c]).startswith(("PENDING", "NOT_APPLICABLE")):
            out[c] = f"NOT_APPLICABLE ({why})"
    return out


def restore_passed(statuses, required, science_ran=False, truthful=True):
    """Every restore check PASS (the attacker may be PENDING / NOT_APPLICABLE only before the science ran), every
    required class PASS or NOT_APPLICABLE, and the role statuses truthful."""
    att_ok = ("PASS",) if science_ran else ("PASS", "PENDING", "NOT_APPLICABLE")
    ok = all(s == "PASS" for k, s in statuses.items() if k != "attacker") and statuses.get("attacker", "PENDING") in att_ok
    ok = ok and all(str(v) == "PASS" or str(v).startswith("NOT_APPLICABLE") or
                    (not science_ran and str(v).startswith("PENDING")) for v in required.values())
    return bool(ok and truthful)


def rehearse(src: Path, targets: dict, seed: int):
    """Dry run: the restore checks read-only with the live store standing in for the copy and the pinned input read in
    place (nothing written)."""
    try:
        checks, cost = restore_all(src, src, targets, seed, INPUT)
    except SystemExit as e:
        return {"status": "NOT_RUN", "reason": scrub_safe(e)}
    st = {k: v.get("status") for k, v in checks.items()}
    sci = science_state(src)["science_ran"]
    return {"note": "live store used as the 'copy' (no copy exists in a dry run); rehearses code paths only",
            "statuses": st, "required": required_targets_status(st, checks, targets.get("not_applicable"),
                                                                targets.get("roles"), sci),
            "cost": cost}


# ------------------------------------------------------------------ (3) this study's copy
def backup(dest=None, targets_file=None, seed=1, dry_run=False, src=None, cache=None, volumes_root=None, out_pkg=None,
           components=None, detection=None, plan_only=False, deps=None):
    src, cache, out_pkg = Path(src or SRC), Path(cache or CACHE), Path(out_pkg or PKG)
    deps = {DEP_REL: INPUT} if deps is None else deps
    files, nbytes = DC.inventory(src)
    targets = load_targets(targets_file if targets_file is not None else LRA_TARGETS)
    seed = int(targets.get("seed", seed))
    sci = science_state(src)
    na_sci = sorted({"learned decoder", "attacker"} & set(targets.get("not_applicable") or {}))
    if sci["science_ran"] and na_sci:
        raise SystemExit(f"REFUSED: {na_sci} cannot be marked not applicable: this study fitted Adult D1 units "
                         f"({sci['adult_d1_fixed_map_units']} dec__, {sci['adult_new_fit_units']} new__)")
    truth = targets_truthfulness(targets, src)
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
    plan = {"schema": "lra-closeout-plan-v1", "at": now(), "source": "<PRIVATE_CACHE>/lra_v1", "files": len(files),
            "bytes": nbytes, "dependencies": {r: "<PRIVATE_CACHE>/jcv_v1/inputs/adult_jcv.npz (pinned sha256 " +
                                              INPUT_SHA[:12] + "...)" for r in deps},
            "drive_mounted": not same_device, "destination": f"{place}/{root.name}/lra_v1", "same_device": same_device,
            "restore_seed": seed, "targets": targets, "targets_truthfulness": truth, "science": sci,
            "drive_detection": ev, "self_contained": self_containment(src),
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
    req = required_targets_status(statuses, checks, targets.get("not_applicable"), targets.get("roles"),
                                  sci["science_ran"])
    restored_ok = restore_passed(statuses, req, sci["science_ran"], truth["pass"])
    bv = {"schema": "lra-backup-verification-v1", "written_at": now(),
          "status": STATUS_LOCAL if same_device else STATUS_DRIVE,
          "off_device_backup": "PENDING (external drive not mounted)" if same_device else "DONE (this study)",
          "source": "<PRIVATE_CACHE>/lra_v1", "destination": f"{place}/{root.name}/lra_v1",
          "bundled_dependency": f"{place}/{root.name}/{DEP_REL}", **cp,
          "read_back": "every copied file re-read with F_NOCACHE (uncached read; not a physical cold-disk read)",
          "read_capability": uncached_read_capability(root / "SHA256SUMS"),
          "restore_kind": ("restore test from a SAME-DEVICE copy (not a drive restore; proves restorability, not "
                           "off-device custody)" if same_device else "restore from the drive copy alone"),
          "restore_seed": seed, "restore_targets": dict(role_targets(targets)), "role_statuses":
              {r: {x: e.get(x) for x in ("status", "config", "unit", "reason", "selection_role", "paired_with",
                                         "family", "construction") if e.get(x) is not None}
               for r, e in (targets.get("roles") or {}).items()},
          "targets_truthfulness": truth, "science": sci, "restore_checks": checks,
          "restore_statuses": statuses, "required_restores": req, "restore_all_pass": bool(restored_ok),
          "restore_cost": cost, "drive_detection": ev, "lra_assessment_state": lra_assessment_state(src),
          "unit_inventory": unit_inventory(copy), "deleted": "nothing", "moved": "nothing"}
    if same_device:
        bv["custody_gap"] = ("The external drive was not mounted: this copy is on the SAME device and is not "
                             "off-device custody. Pending (exact commands): " + "; ".join(PENDING.values()))
        bv["pending"] = PENDING
    if components:
        bv["components"] = components
    write_public(out_pkg / "BACKUP_VERIFICATION.json", bv)
    write_public(out_pkg / "RESTORE_INDEX.json", restore_index(place, root, statuses, same_device, req,
                                                               unit_inventory(copy)))
    (root / "BACKUP_RECORD.json").write_text(json.dumps(bv, indent=1, default=str) + "\n")
    print(json.dumps({k: bv[k] for k in ("status", "files", "uncached_readback_match", "required_restores")}, indent=1))
    return bv


def unit_inventory(store: Path):
    """Unit counts per name prefix in a (copied) store, e.g. {"pol": 81, "dec": 81, "d0s": 6} (names only)."""
    u = Path(store) / "run" / "units"
    out = {}
    for d in (sorted(u.iterdir()) if u.is_dir() else []):
        if d.is_dir():
            pre = d.name.split("__")[0]
            out[pre] = out.get(pre, 0) + 1
            if (d / "trace.json").is_file():
                out["units_with_trace.json"] = out.get("units_with_trace.json", 0) + 1
    att = Path(store) / "run" / "attempts"
    if att.is_dir():
        out["run/attempts"] = sorted(p.name for p in att.iterdir() if p.is_dir())
    return out


def restore_index(place, root, statuses, same_device, required=None, inventory=None):
    return {"schema": "lra-restore-index-v1", "written_at": now(), "private_local": "<PRIVATE_CACHE>/lra_v1",
            "copy": f"{place}/{root.name}/lra_v1" + (" (same device; off-device copy PENDING)" if same_device else ""),
            "checksums": f"{place}/{root.name}/SHA256SUMS",
            "layout": {"lra_v1/admitted/rel__s{k}__{t}/": "admitted teachers (model.pt, heads, release; verified copies "
                                                          "of the cbp admitted artifacts) + ADMISSION_RECEIPT.json",
                       "lra_v1/inputs/": "the authorised 83-column deployment input and schema",
                       "lra_v1/run/units/pol__*": "admitted D0 codes (qpc formats; mean-teacher decoder)",
                       "lra_v1/run/units/dec__*, new__*, diag__*": "D1 codes (dec__ fixed-map decodes incl. CLASS|D1, "
                                                                   "new__ fits, diag__ diagnostics): policy.json (token "
                                                                   "map; its token_proto are D0 means and are NOT "
                                                                   "released) + decoder.json (learned decoder: "
                                                                   "sufficient statistics, released vectors, "
                                                                   "certificates) + release.npz; new__ units also hold "
                                                                   "the mapper trace.json (starts, accepted moves, "
                                                                   "statistic hashes, termination)",
                       "lra_v1/run/units/d0s__*": "same-map D0 diagnostics: the mean-teacher decoder on a new map's "
                                                  "unchanged tokens (decoder-only baseline)",
                       "lra_v1/run/units/cor__*": "correctness-stage units (the pinned known laws under "
                                                  "CORRECTNESS_LOCK; no Adult data)",
                       "lra_v1/run/units/aud__*, outer__*": "inner audits (AUDIT_FIT / INNER_SELECTION) and the locked "
                                                            "assessment units (attacker predictions; private)",
                       "lra_v1/run/units/tea__*, ref__*, fine__*": "teacher outputs, references, label-blind fine "
                                                                   "partitions",
                       "lra_v1/run/attempts/": "retained stage attempts (never overwritten)",
                       "lra_v1/run/*.jsonl, *.log": "activity, compute and semaphore ledgers, stage logs",
                       "lra_v1/run/units/<unit>/": "atomic study units: files + record.json + COMPLETE.json",
                       DEP_REL: "the pinned source input (sha256 e0d9e54a...2f12), bundled so the copy restores alone"},
            "restore": [f"copy {place}/{root.name}/lra_v1 to <PRIVATE_CACHE>/lra_v1 (only if the live store is lost)",
                        "verify: shasum -a 256 -c SHA256SUMS (inside the copy's folder)",
                        f"re-run the restore checks: {SEMA} backup --dry-run --targets <targets.json>",
                        "then follow QUICKSTART.md"],
            "restore_statuses": statuses, "required_restores": required, "unit_inventory": inventory,
            "not_off_device": bool(same_device)}


# ------------------------------------------------------------------ incremental refresh of an existing copy
def refresh(copy_root=None, src=None, cache=None, out_pkg=None, dry_run=False, protect=()):
    """Bring an existing versioned copy up to date WITHOUT deleting anything: files new in the live store are copied;
    files that grew append-only (ledgers) are re-copied; a file whose old copied bytes are NOT a prefix of the live
    bytes is never overwritten (recorded as refused); files present only in the copy are kept. The previous SHA256SUMS
    and BACKUP_RECORD.json are preserved beside the new ones; every entry is re-read uncached. `protect` lists copied
    paths (the restore targets) whose change makes the earlier restore evidence stale (recorded, never hidden)."""
    src, cache, out_pkg = Path(src or SRC), Path(cache or CACHE), Path(out_pkg or PKG)
    roots = sorted(q for q in cache.glob(LOCAL_FOLDER.format(date="*")) if q.is_dir())
    root = Path(copy_root) if copy_root else (roots[-1] if roots else None)
    if root is None or not (root / "SHA256SUMS").exists():
        raise SystemExit("REFUSED: no existing copy with SHA256SUMS to refresh")
    old = {}
    for ln in (root / "SHA256SUMS").read_text().splitlines():
        if ln.strip():
            h, rel = ln.split("  ", 1)
            old[rel] = h
    files, _ = DC.inventory(src)
    plan = {"new": [], "appended": [], "unchanged": 0, "refused_non_append_change": [], "kept_only_in_copy": []}
    live = {}
    for p in files:
        rel = f"{src.name}/{p.relative_to(src)}"
        h = sha(p)
        live[rel] = (p, h)
        if rel not in old:
            plan["new"].append(rel)
        elif old[rel] == h:
            plan["unchanged"] += 1
        elif _is_prefix(root / rel, p):
            plan["appended"].append(rel)
        else:
            plan["refused_non_append_change"].append(rel)
    plan["kept_only_in_copy"] = sorted(r for r in old if r not in live and r.startswith(f"{src.name}/"))
    deps_kept = sorted(r for r in old if not r.startswith(f"{src.name}/"))       # bundled dependencies, never live
    stale = sorted(r for r in plan["appended"] + plan["refused_non_append_change"] if any(r.startswith(x) for x in protect))
    place = "<PRIVATE_CACHE>" if Path(root).parent == cache else "<DRIVE_ROOT>"
    summary = {"copy": f"{place}/{root.name}", "new": len(plan["new"]), "appended": len(plan["appended"]),
               "unchanged": plan["unchanged"], "refused_non_append_change": plan["refused_non_append_change"],
               "kept_only_in_copy": len(plan["kept_only_in_copy"]), "kept_only_in_copy_files": plan["kept_only_in_copy"],
               "bundled_dependencies_kept": len(deps_kept), "restore_evidence_stale_for": stale}
    if dry_run:
        print(scrub(json.dumps({"dry_run": True, **summary, "appended_files": plan["appended"]}, indent=1)))
        return summary
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    prev = f"SHA256SUMS.before_refresh_{stamp}"
    (root / prev).write_bytes((root / "SHA256SUMS").read_bytes())
    if (root / "BACKUP_RECORD.json").exists():
        (root / f"BACKUP_RECORD.before_refresh_{stamp}.json").write_bytes((root / "BACKUP_RECORD.json").read_bytes())
    sums = dict(old)
    changed_during = []
    import shutil
    for rel in plan["new"] + plan["appended"]:
        p, h0 = live[rel]
        q = root / rel
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)
        hq = sha(q, nocache=True)
        if hq != h0 or sha(p) != h0:
            changed_during.append(rel)
        sums[rel] = hq
    (root / "SHA256SUMS").write_text("".join(f"{h}  {r}\n" for r, h in sums.items()))
    vs = verify_sums(root)
    nbytes = sum((root / r).stat().st_size for r in sums)
    summary.update({"at": now(), "bytes": nbytes, "previous_SHA256SUMS_sha256": sha(root / prev),
                    "previous_SHA256SUMS_kept_as": prev, "SHA256SUMS_sha256": vs.get("SHA256SUMS_sha256"),
                    "entries": vs["entries"], "uncached_readback_match": vs["match"], "pass": vs["pass"],
                    "live_files_changed_during_refresh": changed_during, "appended_files": plan["appended"],
                    "deleted": "nothing", "overwritten": "only append-only files whose previous copy is a prefix"})
    bvp = out_pkg / "BACKUP_VERIFICATION.json"
    if bvp.exists():
        bv = jload(bvp)
        bv.setdefault("refreshes", []).append(summary)
        bv.update({"files": vs["entries"], "bytes": nbytes, "uncached_readback_match": vs["match"], "SHA256SUMS_sha256":
                   vs.get("SHA256SUMS_sha256"), "last_refresh_at": summary["at"],
                   "restore_evidence_note": "the restore checks were run on the copy at its first write; the refresh only "
                                            "added new files and grown append-only ledgers" +
                                            (f"; STALE for {stale}" if stale else " (no restore input changed)")})
        write_public(bvp, bv)
        (root / "BACKUP_RECORD.json").write_text(json.dumps(bv, indent=1, default=str) + "\n")
    rip = out_pkg / "RESTORE_INDEX.json"
    if rip.exists():
        ri = jload(rip)
        ri["last_refresh"] = {"at": summary["at"], "SHA256SUMS_sha256": summary["SHA256SUMS_sha256"],
                              "entries": summary["entries"], "previous_sums_kept_as": prev}
        write_public(rip, ri)
    print(json.dumps({k: summary[k] for k in ("new", "appended", "entries", "uncached_readback_match", "pass")}, indent=1))
    return summary


def protected_paths(targets: dict, seed: int):
    """Copied paths whose change makes the restore evidence stale: the target units, the teachers, the attacker units."""
    k = int(targets.get("seed", seed))
    out = [f"lra_v1/run/units/{u}/" for _, u in role_targets(targets)]
    out += [f"lra_v1/admitted/rel__s{k}__U/", f"lra_v1/run/units/tea__s{k}__U/"]
    att = (targets.get("attacker") or {})
    out += [f"lra_v1/run/units/{u}/" for u in ((att.get("kwargs") or {}).get("outer_unit"),
                                               (att.get("kwargs") or {}).get("release_unit"),
                                               (att.get("saved") or {}).get("unit")) if u]
    return sorted(set(out))


# ------------------------------------------------------------------ everything, in order
def run_all(targets_file=None, seed=1, dry_run=False, volumes_root=None, src=None, cache=None, out_pkg=None,
            plan_only=False, cbp_targets=None, deps=None):
    vol, ev = locate_drive(volumes_root)
    before = closed_trees()
    folders0 = top_folders(vol)
    comp = {"drive_mounted": vol is not None, "drive_detection": ev}
    if vol is None or dry_run:
        c = cbp_custody(dry_run, volumes_root, cbp_targets=cbp_targets)
        comp["cbp_custody"] = {"status": c.get("status"), "receipt": "provenance/cbp_custody/STATUS.json",
                               "pending_command": c.get("pending_command")}
        return backup(None if vol is None else str(vol), targets_file, seed, dry_run, src, cache, volumes_root, out_pkg,
                      components=comp, detection=ev, plan_only=plan_only, deps=deps)
    c = cbp_custody(False, volumes_root, cbp_targets=cbp_targets)                       # (1)
    comp["cbp_custody"] = {"status": c.get("status"), "receipt": "provenance/cbp_custody/STATUS.json",
                           "pending_command": c.get("pending_command")}
    new_before_lra = sorted(top_folders(vol) - folders0)
    bv = backup(str(vol), targets_file, seed, False, src, cache, volumes_root, out_pkg, components=comp, detection=ev,
                deps=deps)                                                              # (2) + (4)
    new = sorted(top_folders(vol) - folders0)                                           # (3)
    prior = sorted(n for n in folders0 if KNOWN_FOLDERS.match(n))
    pub = lambda n: n if KNOWN_FOLDERS.match(n) else "<other new folder>"              # noqa: E731
    reread = {pub(n): verify_sums(Path(vol) / n) for n in new}
    prior_reread = {n: verify_sums(Path(vol) / n) for n in prior}
    after = closed_trees()
    bv["uncached_reread_of_drive_folders"] = {
        "created_by_this_closeout": reread, "created_before_the_lra_copy": len(new_before_lra),
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
    ap = argparse.ArgumentParser(prog="python -m lra.closeout")
    ap.add_argument("job", choices=("status", "all", "backup", "cbp-custody", "refresh"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--plan-only", action="store_true", help="with --dry-run: skip the read-only restore rehearsal")
    ap.add_argument("--dest", default=None)
    ap.add_argument("--targets", default=None)
    ap.add_argument("--cbp-targets", default=None)
    ap.add_argument("--copy-root", default=None, help="refresh: the copy folder (default: newest same-device copy)")
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
        run_all(a.targets, a.seed, a.dry_run, plan_only=a.plan_only, cbp_targets=a.cbp_targets)
    elif a.job == "refresh":
        tg = load_targets(a.targets if a.targets is not None else LRA_TARGETS)
        refresh(copy_root=a.copy_root, dry_run=a.dry_run, protect=protected_paths(tg, a.seed))
    elif a.job == "backup":
        backup(a.dest, a.targets, a.seed, a.dry_run, plan_only=a.plan_only)
    else:
        print(json.dumps({"status": cbp_custody(a.dry_run, cbp_targets=a.cbp_targets)["status"]}))


if __name__ == "__main__":
    main()
