"""Custody closeout for the online-strength frontier study (data/custody owner; adapted from smf/closeout.py).

Two jobs, each with a dry-run that works without the external drive:

1. Predecessor custody repair (prompt section 5), writes results/.../PREDECESSOR_CUSTODY_REPAIR.json:
       OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m osf.closeout predecessor [--restore] [--dry-run]
   - locates the smf drive copy <DRIVE_ROOT>/private_smf_v1_20261005/smf_v1 by CONTENT, not by a guessed volume name:
     a mounted volume qualifies iff private_smf_v1_20261005/SHA256SUMS exists there and its sha256 equals the value the
     smf closeout recorded at backup (BACKUP_VERIFICATION.json SHA256SUMS_sha256). The private smf store itself records
     the folder name only (smf/closeout.py NAME; RESTORE_INDEX.json), never the volume.
   - copies the two later smf logs (run/closeout_backup.log, run/watchdog.log) into a NEW versioned local supplement
     <PRIVATE_CACHE>/smf_v1_custody_supplement_20261005/ (SHA256SUMS, uncached re-read) and, when the drive is
     mounted, into a new versioned drive folder <DRIVE_ROOT>/private_smf_v1_custody_supplement_20261005[_vN]/.
   - --restore with the drive mounted runs the pinned independent replay ONCE, single process, on the source worktree,
     with its report redirected by --out (replay_smf.py otherwise writes INDEPENDENT_VERIFICATION.json into the closed
     smf results directory); the closed worktree must be clean at the pin before and after.
   Nothing in <PRIVATE_CACHE>/smf_v1 or the closed worktree is written, moved or deleted.

2. Study backup and independent restore at closeout (prompt section 17), called by the lead:
       OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m osf.closeout backup --dry-run
       OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python -m osf.closeout backup [--dest <drive root>] [--seed 1]
                                                                                    [--targets targets.json]
   - drive: --dest, else the volume holding the verified smf copy (same external drive).
   - copies <PRIVATE_CACHE>/osf_v1 (incl. admitted/ copies) into a NEW versioned folder
     <DRIVE_ROOT>/private_osf_v1_<UTC date>[_vN]/osf_v1 (never overwrites; nothing deleted anywhere), writes SHA256SUMS,
     re-reads every copied file with F_NOCACHE (uncached read; NOT a physical cold-disk read - recorded as a limitation).
   - independent restore FROM THE DRIVE COPY ALONE for one seed: U, the incumbent RAW-J beta 0.3, N* (or its labelled
     descriptive fallback), R*, the local comparator L*, using this module's own forward pass (torch functional ops on
     the drive model.pt, weights_only), the drive heads and, for LEACE units, the drive maps: features, centred logits,
     probabilities and decisions are rebuilt from the 83 permitted inputs and compared with the local releases; then
     the recorded AUC-selected pair attacker of the deployable best model's assessment unit is refitted from the drive
     release on AUDIT_FIT and its assessment probabilities compared with the saved ones.
   - drive absent: an explicit LOCAL_ONLY record (private SHA256SUMS manifest of osf_v1 in a versioned local folder,
     no claim of off-device custody) and the exact pending backup/restore command.
   Writes (real run): BACKUP_VERIFICATION.json, RESTORE_INDEX.json (public, placeholders only), BACKUP_RECORD.json on
   the drive. --dry-run writes nothing and prints the plan.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
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
PKG = WT / "results" / "pcrl_online_strength_frontier_v1"
CACHE = HOME / "PCRL_eval_cache_private"
SRC = CACHE / "osf_v1"
SMF = CACHE / "smf_v1"
SMF_PKG = WT / "results" / "pcrl_strength_matched_feedback_v1"
SOURCE_PIN = "a9951ed2fed9943d445a208a8a7e456a56f39114"
SOURCE_BRANCH = "research/pcrl-strength-matched-feedback-v1"
SMF_DRIVE_NAME = "private_smf_v1_20261005"
SMF_SUPPLEMENT = "smf_v1_custody_supplement_20261005"
SMF_DRIVE_SUPPLEMENT = "private_smf_v1_custody_supplement_20261005"
LATER_LOGS = ("run/closeout_backup.log", "run/watchdog.log")
# boot volume and an unrelated read-only installer image: skipped by NAME before any filesystem call on them
SKIP_VOLUMES = ("Macintosh HD", "BackgroundSyncService Setup")
REPLAY_REL = "results/pcrl_strength_matched_feedback_v1/verification/replay_smf.py"
REPLAY_OUT = PKG / "predecessor_custody" / "INDEPENDENT_VERIFICATION_DRIVE_RESTORE.json"
PY = HOME / "PCRL" / ".venv" / "bin" / "python"
OUT_REPAIR = PKG / "PREDECESSOR_CUSTODY_REPAIR.json"


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


def git(wt, *a):
    r = subprocess.run(["git", "-C", str(wt), *a], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def scrub(text):
    """Refuse public text containing a home folder, a volume path or the local user name."""
    if re.search(r"/Users/|/Volumes/|" + re.escape(HOME.name), text):
        raise SystemExit("REFUSED: identifying path in a public file")
    return text


def write_public(path, obj):
    txt = scrub(json.dumps(obj, indent=1, default=str) + "\n")
    tmp = Path(path).with_suffix(".json.tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(txt)
    tmp.replace(path)


def volumes():
    """Mounted volumes other than the boot volume and the unrelated installer image (names only are listed)."""
    root = Path("/Volumes")
    names = sorted(os.listdir(root)) if root.exists() else []
    return [root / n for n in names if n not in SKIP_VOLUMES]


def smf_recorded_sums_sha():
    return jload(SMF_PKG / "BACKUP_VERIFICATION.json")["SHA256SUMS_sha256"]


def locate_smf_drive():
    """(volume, evidence) for the volume holding the verified smf copy, or (None, evidence)."""
    want = smf_recorded_sums_sha()
    ev = {"rule": f"<volume>/{SMF_DRIVE_NAME}/SHA256SUMS exists and sha256 = {want} (BACKUP_VERIFICATION.json of the "
                  "closed smf study)", "checked_at": now(), "candidate_volumes": 0, "matching_volumes": 0}
    hit = None
    for v in volumes():
        ev["candidate_volumes"] += 1
        f = v / SMF_DRIVE_NAME / "SHA256SUMS"
        try:
            if f.is_file() and sha(f) == want:
                ev["matching_volumes"] += 1
                hit = hit or v
        except OSError:
            continue
    ev["mounted"] = hit is not None
    return hit, ev


def versioned(parent: Path, name: str):
    d, n = parent / name, 1
    while d.exists():
        n += 1
        d = parent / f"{name}_v{n}"
    return d


def copy_files(src_root: Path, rels, dst_root: Path):
    """Copy rels (relative paths) from src_root into a NEW dst_root (refuses an existing one); re-reads every copy
    uncached. Returns ({rel: sha256 of the source}, {rel: uncached re-read matches})."""
    if dst_root.exists():
        raise SystemExit(f"REFUSED: {dst_root.name} exists (versioned folders are never reused)")
    sums = {}
    for r in rels:
        s, q = src_root / r, dst_root / r
        q.parent.mkdir(parents=True, exist_ok=True)
        sums[r] = sha(s)
        shutil.copy2(s, q)
    reread = {r: sha(dst_root / r, nocache=True) == h for r, h in sums.items()}
    return sums, reread


# ------------------------------------------------------------------ 1. predecessor custody repair
def closed_tree_state(src_wt):
    pkg = src_wt / "results" / "pcrl_strength_matched_feedback_v1"
    files = sorted(p for p in pkg.rglob("*") if p.is_file())
    h = hashlib.sha256()
    for p in files:
        h.update(str(p.relative_to(src_wt)).encode() + b"\0" + sha(p).encode())
    return {"head": git(src_wt, "rev-parse", "HEAD"), "porcelain_clean": git(src_wt, "status", "--porcelain") == "",
            "closed_results_files": len(files), "closed_results_tree_sha256": h.hexdigest()}


def source_worktree():
    out = git(WT, "worktree", "list", "--porcelain") or ""
    for block in out.split("\n\n"):
        lines = dict(ln.split(" ", 1) for ln in block.splitlines() if " " in ln)
        if lines.get("branch") == f"refs/heads/{SOURCE_BRANCH}":
            return Path(lines["worktree"])
    return None


def local_supplement(dry_run):
    dst = CACHE / SMF_SUPPLEMENT
    hashes = {r: sha(SMF / r) for r in LATER_LOGS}
    rec = {"folder": f"<PRIVATE_CACHE>/{SMF_SUPPLEMENT}", "layout": "smf_v1/run/<log> (mirrors the drive copy layout)",
           "files_sha256": {f"smf_v1/{r}": h for r, h in hashes.items()},
           "sizes": {f"smf_v1/{r}": (SMF / r).stat().st_size for r in LATER_LOGS},
           "source_mtime_utc": {f"smf_v1/{r}": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime((SMF / r).stat().st_mtime))
                                for r in LATER_LOGS}}
    if dry_run:
        rec["status"] = "DRY_RUN (not written)"
        return rec
    if dst.exists():          # idempotent re-run: verify the existing supplement instead of writing another
        sums = dict(reversed(ln.split("  ", 1)) for ln in (dst / "SHA256SUMS").read_text().splitlines())
        rec["status"] = "PRESENT (verified)" if all(
            sums.get(f"smf_v1/{r}") == h and sha(dst / "smf_v1" / r, nocache=True) == h for r, h in hashes.items()) \
            else "PRESENT_BUT_DIFFERS"
        rec["SHA256SUMS_sha256"] = sha(dst / "SHA256SUMS")
        rec["created_at"] = jload(dst / "SUPPLEMENT_RECORD.json")["created_at"]
        return rec
    (dst / "smf_v1" / "run").mkdir(parents=True)
    for r in LATER_LOGS:
        shutil.copy2(SMF / r, dst / "smf_v1" / r)
    (dst / "SHA256SUMS").write_text("".join(f"{h}  smf_v1/{r}\n" for r, h in hashes.items()))
    reread = {f"smf_v1/{r}": sha(dst / "smf_v1" / r, nocache=True) == h for r, h in hashes.items()}
    created = now()
    (dst / "SUPPLEMENT_RECORD.json").write_text(json.dumps(
        {"schema": "smf-custody-supplement-v1", "created_at": created, "source": "<PRIVATE_CACHE>/smf_v1",
         "files_sha256": rec["files_sha256"], "uncached_reread_match": reread,
         "why": "written to the smf private store after the 01:52Z drive backup; absent from the drive copy",
         "limitation": "a second copy on the same internal disk; not an off-device backup and not a restore"},
        indent=1) + "\n")
    rec.update({"status": "CREATED", "created_at": created, "uncached_reread_match": reread,
                "SHA256SUMS_sha256": sha(dst / "SHA256SUMS")})
    return rec


def drive_supplement(vol, dry_run):
    dst = versioned(vol, SMF_DRIVE_SUPPLEMENT)
    if dry_run:
        return {"status": "DRY_RUN", "folder": f"<DRIVE_ROOT>/{dst.name}"}
    (dst / "smf_v1" / "run").mkdir(parents=True)
    hashes = {}
    for r in LATER_LOGS:
        hashes[f"smf_v1/{r}"] = sha(SMF / r)
        shutil.copy2(SMF / r, dst / "smf_v1" / r)
    (dst / "SHA256SUMS").write_text("".join(f"{h}  {r}\n" for r, h in hashes.items()))
    reread = {r: sha(dst / r, nocache=True) == h for r, h in hashes.items()}
    return {"status": "CREATED", "folder": f"<DRIVE_ROOT>/{dst.name}", "files_sha256": hashes,
            "uncached_reread_match": reread, "SHA256SUMS_sha256": sha(dst / "SHA256SUMS"),
            "read_back": "F_NOCACHE uncached read; not a physical cold-disk read"}


def restore_command(public=True):
    return ("cd <SOURCE_WORKTREE> && OMP_NUM_THREADS=1 <python> " + REPLAY_REL +
            f" --drive-root <DRIVE_ROOT>/{SMF_DRIVE_NAME}/smf_v1 --out <WORKTREE>/results/"
            "pcrl_online_strength_frontier_v1/predecessor_custody/INDEPENDENT_VERIFICATION_DRIVE_RESTORE.json")


def run_replay(src_wt, vol):
    REPLAY_OUT.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["/usr/bin/time", "-l", str(PY), REPLAY_REL, "--drive-root", str(vol / SMF_DRIVE_NAME / "smf_v1"),
           "--out", str(REPLAY_OUT)]
    env = dict(os.environ, OMP_NUM_THREADS="1")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=src_wt, capture_output=True, text=True, env=env)
    wall = time.time() - t0
    m_user = re.search(r"([\d.]+) user", r.stderr)
    m_sys = re.search(r"([\d.]+) sys", r.stderr)
    m_rss = re.search(r"(\d+)\s+maximum resident set size", r.stderr)
    out = {"returncode": r.returncode, "wall_s": round(wall, 1),
           "cpu_s": round(float(m_user.group(1)) + float(m_sys.group(1)), 1) if m_user and m_sys else None,
           "peak_rss_bytes": int(m_rss.group(1)) if m_rss else None, "processes": 1}
    if REPLAY_OUT.exists():
        rep = jload(REPLAY_OUT)
        out["report"] = "results/pcrl_online_strength_frontier_v1/predecessor_custody/INDEPENDENT_VERIFICATION_DRIVE_" \
                        "RESTORE.json"
        out["report_sha256"] = sha(REPLAY_OUT)
        out["drive_restore"] = rep["checks"].get("drive_restore")
        out["overall"] = rep["summary"]["overall"]
        out["status_counts_top_level"] = rep["summary"]["status_counts_top_level"]
    else:
        out["stderr_tail"] = scrub_safe(r.stderr[-1500:])
    return out


def scrub_safe(s):
    return re.sub(r"/(Users|Volumes)/[^\s'\"]+", "<PATH>", s).replace(HOME.name, "<USER>")


def predecessor(restore=False, dry_run=False):
    src_wt = source_worktree()
    vol, drive_ev = locate_smf_drive()
    closed_before = closed_tree_state(src_wt) if src_wt else None
    supp = local_supplement(dry_run)
    smf_files_now = sum(1 for p in SMF.rglob("*") if p.is_file() and ".tmp" not in p.parts[-2])
    units = SMF / "run" / "units"
    unit_ok = 0
    n_units = 0
    for d in sorted(units.iterdir()):
        c = d / "COMPLETE.json"
        if not c.exists():
            continue
        n_units += 1
        files = jload(c)["files"]
        unit_ok += all(sha(d / f) == h for f, h in files.items())
    rec = {
        "schema": "osf-predecessor-custody-repair-v1",
        "written_at": now(),
        "predecessor": {"study": "pcrl_strength_matched_feedback_v1 (smf)", "branch": SOURCE_BRANCH,
                        "evidence_sha": SOURCE_PIN,
                        "source_worktree_head": closed_before["head"] if closed_before else None,
                        "remote_branch_head": git(WT, "rev-parse", f"origin/{SOURCE_BRANCH}")},
        "scientific_results_unchanged": {
            "verdict": jload(SMF_PKG / "MODEL_MANIFEST.json")["label"],
            "claims": jload(SMF_PKG / "MODEL_MANIFEST.json")["claims"],
            "assessment_seed_means_reported_in_the_prompt": {
                "U": {"pair_sex_auc": 0.882, "income_sex_auc": 0.860, "occupation_sex_auc": 0.880, "income_acc": 0.852,
                      "occupation_acc": 0.480},
                "RAW-J": {"pair_sex_auc": 0.810, "income_sex_auc": 0.768, "occupation_sex_auc": 0.796,
                          "income_acc": 0.853, "occupation_acc": 0.474},
                "J-F (descriptive fallback, never a valid nominee)": {"pair_sex_auc": 0.864, "income_sex_auc": 0.797,
                                                                      "occupation_sex_auc": 0.846, "income_acc": 0.851,
                                                                      "occupation_acc": 0.481},
                "L-F": {"pair_sex_auc": 0.870, "income_sex_auc": 0.823, "occupation_sex_auc": 0.848, "income_acc": 0.851,
                        "occupation_acc": 0.476}},
            "closed_files_at_evidence_sha_sha256": {
                f: sha(SMF_PKG / f) for f in ("RESEARCH_DECISION.md", "PRIMARY_ENDPOINTS.csv", "MODEL_MANIFEST.json",
                                              "INDEPENDENT_VERIFICATION.json", "BACKUP_VERIFICATION.json",
                                              "RESTORE_INDEX.json", "VALIDATION.md", "COST_AND_CLOSEOUT.md")},
            "closed_files_equal_evidence_sha_blobs": all(
                git(WT, "rev-parse", f"{SOURCE_PIN}:results/pcrl_strength_matched_feedback_v1/{f}") ==
                git(WT, "hash-object", str(SMF_PKG / f))
                for f in ("RESEARCH_DECISION.md", "PRIMARY_ENDPOINTS.csv", "MODEL_MANIFEST.json",
                          "INDEPENDENT_VERIFICATION.json", "BACKUP_VERIFICATION.json", "RESTORE_INDEX.json")),
            "note": "No closed predecessor file, verdict or receipt is modified by this record or by any osf code."},
        "drive_copy": {
            "root": f"<DRIVE_ROOT>/{SMF_DRIVE_NAME}/smf_v1",
            "located_from": ("the smf private records and closeout code: smf/closeout.py NAME = '" + SMF_DRIVE_NAME +
                             "', RESTORE_INDEX.json drive_copy, BACKUP_VERIFICATION.json (2,412 files; SHA256SUMS "
                             "sha256). The private smf_v1 store (run/closeout_backup.log: 'drive copy') records no "
                             "volume name; the volume is identified by content (the recorded SHA256SUMS hash), never "
                             "by a guessed name."),
            "recorded_backup": {k: jload(SMF_PKG / "BACKUP_VERIFICATION.json")[k]
                                for k in ("written_at", "files", "uncached_readback_match", "bytes", "SHA256SUMS_sha256")},
            "detection": {**drive_ev, "note": "the boot volume and one unrelated read-only installer image are skipped "
                                              "by name and never opened"}},
        "local_private_store_now": {"files": smf_files_now, "files_at_backup": 2412,
                                    "unit_directories_with_COMPLETE": n_units, "units_hash_verified": unit_ok,
                                    "changed_after_backup": list(LATER_LOGS)},
        "later_logs_supplement": {"local": supp},
        "restore": {"command": restore_command(),
                    "safety": ("replay_smf.py has exactly one write: its JSON report, by default "
                               "results/pcrl_strength_matched_feedback_v1/INDEPENDENT_VERIFICATION.json inside the CLOSED "
                               "results directory (OUT; tmp file beside it). --out redirects both the report and its "
                               "tmp file; its git calls are read-only (rev-parse, log, branch -r, reflog, hash-object "
                               "without -w). The closed worktree is checked clean at the evidence SHA before and after."),
                    "expected_cost": "single process; the smf verifier's full run took about 264 s wall",
                    "independence": "the smf verifier's own script (import guard against smf/rgj/jcv/pnx/oar/"
                                    "stored_model_eval), reading parameters and heads from the drive copy only"},
        "statements": [
            "A disconnected drive is not evidence that models were lost.",
            "A successful copy alone is not an independent restore.",
            "The historical drive-folder-name exposure in four earlier commits of the refreshed-study branch is "
            "unchanged; it is not claimed to be removed and no history was rewritten.",
            "The local supplement is a second copy on the same internal disk, not an off-device backup."],
    }
    if vol is None:
        rec["status"] = "PENDING"
        rec["restore"]["status"] = "PENDING (drive not mounted; restore not executed)"
        rec["later_logs_supplement"]["drive"] = {"status": "PENDING", "folder": f"<DRIVE_ROOT>/{SMF_DRIVE_SUPPLEMENT}"}
    else:
        rec["later_logs_supplement"]["drive"] = drive_supplement(vol, dry_run)
        if restore and not dry_run:
            if not (closed_before and closed_before["head"] == SOURCE_PIN and closed_before["porcelain_clean"]):
                raise SystemExit("REFUSED: source worktree is not clean at the evidence SHA")
            rec["restore"]["run"] = run_replay(src_wt, vol)
            after = closed_tree_state(src_wt)
            rec["restore"]["closed_worktree_unchanged"] = after == closed_before
            dr = (rec["restore"]["run"].get("drive_restore") or {}).get("status")
            rec["restore"]["status"] = dr or "FAILED_TO_RUN"
            rec["status"] = "RESTORED" if dr == "PASS" else f"RESTORE_{dr or 'FAILED'}"
        else:
            rec["restore"]["status"] = "PENDING (drive mounted; run with --restore)"
            rec["status"] = "PENDING_RESTORE"
    if vol is None:
        gaps = ["The smf verifier's independent restore of seed-1 U, J-F and L-F from the drive copy has not been "
                "executed: the drive was not mounted at any check.",
                "run/closeout_backup.log and run/watchdog.log are not yet on the drive (local versioned supplement on "
                "the same internal disk only)."]
    elif rec["status"] == "RESTORED":
        gaps = []
    elif rec["status"] == "PENDING_RESTORE":
        gaps = ["Drive mounted but the independent restore has not been run (python -m osf.closeout predecessor "
                "--restore)."]
    else:
        gaps = [f"Independent restore did not pass: {rec['status']}."]
    rec["unresolved_gap"] = gaps
    if closed_before is not None:
        rec["closed_worktree_check"] = {"head_equals_evidence_sha": closed_before["head"] == SOURCE_PIN,
                                        "clean": closed_before["porcelain_clean"],
                                        "closed_results_files": closed_before["closed_results_files"],
                                        "closed_results_tree_sha256": closed_before["closed_results_tree_sha256"]}
    if dry_run:
        print(scrub(json.dumps(rec, indent=1, default=str)))
        return rec
    write_public(OUT_REPAIR, rec)
    print(json.dumps({"status": rec["status"], "drive": drive_ev["mounted"], "supplement": supp["status"]}, indent=1))
    return rec


# ------------------------------------------------------------------ 2. study backup + independent restore
def inventory():
    files = sorted(p for p in SRC.rglob("*") if p.is_file() and not any(".tmp" in x for x in p.relative_to(SRC).parts))
    return files, int(sum(p.stat().st_size for p in files))


def resolve_targets(seed, targets_file=None):
    """label -> unit name. Defaults from the pushed EVALUATION_LOCK.json (resolved L*, N*, R*; score[cid]['unit'])."""
    from osf import run as R
    t = {"U": R.rel_name(seed, "U"), "incumbent RAW-J beta 0.3": R.rel_name(seed, "RAW-J|b0.3")}
    notes = {}
    el = PKG / "EVALUATION_LOCK.json"
    if el.exists():
        L = jload(el)
        score = L["seeds"][str(seed)]["score"]
        for lab, key in (("N*", "N*"), ("R*", "R*"), ("L* (local comparator)", "L*")):
            cfg = (L.get("resolved") or {}).get(key)
            st = (L.get("statuses") or {}).get(key, {}).get("status")
            if cfg and cfg in score and score[cfg].get("unit"):
                name = lab if st == "NOMINEE" else f"{lab} [descriptive fallback: {st}]"
                t[name] = score[cfg]["unit"]
            else:
                notes[lab] = f"not resolvable from EVALUATION_LOCK (config {cfg}, status {st})"
        best = L.get("deployable_best")
        notes["deployable_best"] = best
    else:
        notes["EVALUATION_LOCK"] = "absent: N*, R*, L* resolve after the lock is pushed (or pass --targets)"
    if targets_file:
        t.update(jload(targets_file))
    return t, notes


def forward(state, X):
    """Own forward pass (no osf.train/jcv.train model code): 83-64-64-16 ReLU per recipient, float32, as released."""
    import torch
    import torch.nn.functional as F
    Xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
    out = []
    with torch.no_grad():
        for i in (0, 1):
            h = F.relu(F.linear(Xt, state[f"enc.{i}.0.weight"], state[f"enc.{i}.0.bias"]))
            h = F.relu(F.linear(h, state[f"enc.{i}.2.weight"], state[f"enc.{i}.2.bias"]))
            h = F.linear(h, state[f"enc.{i}.4.weight"], state[f"enc.{i}.4.bias"])
            out.append(h.double().numpy())
    return out


def head_outputs(head, R):
    z = np.asarray(head.decision_function(R), dtype=np.float64)
    if z.ndim == 1:
        z = np.stack([np.zeros_like(z), z], 1)
    P = head.predict_proba(R)
    return z - z.mean(1, keepdims=True), P, P.argmax(1)


def restore_unit(drive_units: Path, name: str, D):
    import joblib
    import torch
    dd, ld = drive_units / name, SRC / "run" / "units" / name
    info = {"unit": name}
    if not (dd / "COMPLETE.json").exists():
        return {**info, "status": "FAIL", "reason": "unit missing on the drive copy"}
    dc = jload(dd / "COMPLETE.json")
    info["drive_complete_equals_local"] = dc == jload(ld / "COMPLETE.json")
    info["drive_files_hash_ok_uncached"] = all(sha(dd / f, nocache=True) == h for f, h in dc["files"].items())
    if not (info["drive_complete_equals_local"] and info["drive_files_hash_ok_uncached"]):
        return {**info, "status": "FAIL", "reason": "drive copy does not verify; nothing unpickled from it"}
    if not (dd / "model.pt").exists():
        info["status"] = "PASS" if info["drive_complete_equals_local"] and info["drive_files_hash_ok_uncached"] else "FAIL"
        info["rebuild"] = "not a neural release (hashes verified only)"
        return info
    st = torch.load(dd / "model.pt", weights_only=True)
    loc = np.load(ld / "release.npz", allow_pickle=False)
    pos = {int(r): j for j, r in enumerate(D["row_id"])}
    ix = np.array([pos[int(r)] for r in loc["row_id"]])
    H = forward(st, D["X"][ix])
    if (dd / "leace_0").exists():
        from stored_model_eval.defenses import LeaceMap
        H = [LeaceMap.load(dd / f"leace_{i}").transform(H[i]) for i in (0, 1)]
    diffs = {}
    for i in (0, 1):
        h = joblib.load(dd / f"head_{i}.joblib")
        c, P, hard = head_outputs(h, H[i])
        k = i + 1
        diffs[f"recipient_{k}"] = {"features_max_abs_diff": float(np.abs(H[i] - loc[f"r{k}"]).max()),
                                   "centred_logits_max_abs_diff": float(np.abs(c - loc[f"c{k}"]).max()),
                                   "prob_max_abs_diff": float(np.abs(P - loc[f"p{k}"]).max()),
                                   "hard_mismatches": int((hard != loc[f"hard{k}"]).sum())}
    info["rebuild_from_drive_vs_local_release"] = diffs
    ok = info["drive_complete_equals_local"] and info["drive_files_hash_ok_uncached"] and all(
        v["features_max_abs_diff"] == 0 and v["centred_logits_max_abs_diff"] <= 1e-12 and v["prob_max_abs_diff"] <= 1e-12
        and v["hard_mismatches"] == 0 for v in diffs.values())
    info["status"] = "PASS" if ok else "FAIL"
    return info


def restore_attacker(drive_units: Path, release_unit: str, seed: int, D):
    """Refit the recorded AUC-selected pair attacker of the assessment unit scoring release_unit from the drive copy."""
    units = sorted((SRC / "run" / "units").glob(f"outer__s{seed}__*"))
    rec_dir = next((u for u in units if release_unit in (u / "record.json").read_text()), None) if units else None
    if rec_dir is None:
        return {"status": "PENDING", "reason": f"no assessment unit for {release_unit} (assessment not run)"}
    try:
        from smf import audit as AU       # osf.assess scores with the pinned smf.audit final slate
        from rgj import finalize as FN
    except ImportError as e:              # pragma: no cover
        return {"status": "PENDING", "reason": f"attacker slate unavailable: {e}"}
    orec = jload(drive_units / rec_dir.name / "record.json")
    scored = orec["primary"]["scored"]["pair"]["auc"]
    name, src = scored.get("attacker"), scored.get("source_view", "pair")
    z = np.load(drive_units / release_unit / "release.npz")
    V = FN.views_from_release(z)
    pos = {int(r): j for j, r in enumerate(z["row_id"])}
    take = lambda ix: np.array([pos[int(r)] for r in D["row_id"][ix]])   # noqa: E731
    fit, ass = D["idx"]["AUDIT_FIT"], D["idx"]["OSF_DEVELOPMENT_ASSESSMENT"]
    fac = dict(AU.final_slate(False))[name]
    m = fac(0).fit(V[src][take(fit)], D["sex"][fit])
    P = AU.proba(m, V[src][take(ass)], 2)
    saved = np.load(drive_units / rec_dir.name / "preds.npz")["P_auc_pair"][0]
    d = float(np.abs(P - saved).max())
    return {"status": "PASS" if d == 0 else "FAIL", "assessment_unit": rec_dir.name, "attacker": name,
            "source_view": src, "source": "drive-copy release + AUDIT_FIT refit", "max_abs_diff_vs_saved": d}


def backup(dest=None, seed=1, targets_file=None, dry_run=False):
    files, nbytes = inventory()
    if dest:
        vol = Path(dest)
        how = "--dest"
    else:
        vol, _ = locate_smf_drive()
        how = "volume holding the verified smf copy"
    date = time.strftime("%Y%m%d", time.gmtime())
    targets, notes = resolve_targets(seed, targets_file)
    plan = {"schema": "osf-closeout-plan-v1", "at": now(), "source": "<PRIVATE_CACHE>/osf_v1", "files": len(files),
            "bytes": nbytes, "drive_mounted": vol is not None, "drive_selected_by": how if vol else None,
            "drive_folder": (f"<DRIVE_ROOT>/{versioned(vol, f'private_osf_v1_{date}').name}/osf_v1" if vol else
                             f"<DRIVE_ROOT>/private_osf_v1_{date}[_vN]/osf_v1 (PENDING: drive absent)"),
            "restore_seed": seed, "restore_targets": targets, "target_notes": notes,
            "target_units_present_locally": {k: (SRC / "run" / "units" / u / "COMPLETE.json").exists()
                                             for k, u in targets.items()},
            "attacker_restore": "refit the AUC-selected pair attacker of the deployable best model's assessment unit "
                                "(outer__s{seed}__*) from the drive release on AUDIT_FIT; compare with preds.npz "
                                "P_auc_pair[0]",
            "fallback_if_drive_absent": ("LOCAL_ONLY: private SHA256SUMS manifest of every osf_v1 file in "
                                         "<PRIVATE_CACHE>/osf_v1_closeout_manifest_<date>[_vN]/ ; BACKUP_VERIFICATION.json "
                                         "status LOCAL_ONLY_PENDING_DRIVE with the pending command; no off-device "
                                         "custody claimed"),
            "pending_command": "OMP_NUM_THREADS=1 PYTHONPATH=. <python> -m osf.closeout backup --dest <DRIVE_ROOT>",
            "predecessor_supplement": "also run: python -m osf.closeout predecessor --restore (drive mounted)",
            "limitations": ["F_NOCACHE re-read is an uncached read, not a physical cold-disk read",
                            "FARE references are hash-verified on the drive; their official encoder is not re-run here"]}
    if dry_run:
        print(scrub(json.dumps(plan, indent=1, default=str)))
        return plan
    from osf import data as DA
    if vol is None:                       # explicit local-only fallback
        mdir = versioned(CACHE, f"osf_v1_closeout_manifest_{date}")
        mdir.mkdir(parents=True)
        sums = {str(p.relative_to(SRC)): sha(p, nocache=True) for p in files}
        (mdir / "SHA256SUMS").write_text("".join(f"{h}  osf_v1/{r}\n" for r, h in sums.items()))
        bv = {"schema": "osf-backup-verification-v1", "written_at": now(), "status": "LOCAL_ONLY_PENDING_DRIVE",
              "source": "<PRIVATE_CACHE>/osf_v1", "files": len(files), "bytes": nbytes,
              "local_manifest": f"<PRIVATE_CACHE>/{mdir.name}/SHA256SUMS",
              "local_manifest_sha256": sha(mdir / "SHA256SUMS"),
              "pending": {"backup_and_restore": plan["pending_command"],
                          "predecessor": "python -m osf.closeout predecessor --restore"},
              "statement": "No off-device copy exists for this study yet; nothing was deleted; restore not executed.",
              "deleted": "nothing"}
        write_public(PKG / "BACKUP_VERIFICATION.json", bv)
        print(json.dumps({k: bv[k] for k in ("status", "files", "local_manifest_sha256")}, indent=1))
        return bv
    root = versioned(vol, f"private_osf_v1_{date}")
    dst = root / "osf_v1"
    rels = [str(p.relative_to(SRC)) for p in files]
    sums, reread = copy_files(SRC, rels, dst)
    (root / "SHA256SUMS").write_text("".join(f"{h}  osf_v1/{r}\n" for r, h in sums.items()))
    try:                                   # assessment labels only through osf.assess's verified pushed-lock path
        from osf import assess as AS
        AS.open_assessment(PKG / "EVALUATION_LOCK.json")
        D = AS.load_unsealed()
    except (SystemExit, FileNotFoundError, ImportError) as e:
        D = DA.load()
        unseal_note = f"sealed: {e}"
    else:
        unseal_note = "unsealed after osf.assess verified the pushed EVALUATION_LOCK"
    restores = {lab: restore_unit(dst / "run" / "units", u, D) for lab, u in targets.items()}
    best = notes.get("deployable_best")
    best_unit = targets.get("N*") or targets.get("R*") or targets["incumbent RAW-J beta 0.3"]
    restores["final attacker"] = restore_attacker(dst / "run" / "units", best_unit, seed, D) if D["sealed"] is False \
        else {"status": "PENDING", "reason": unseal_note}
    del D
    rel = root.name
    bv = {"schema": "osf-backup-verification-v1", "written_at": now(), "status": "DRIVE_COPY_VERIFIED",
          "source": "<PRIVATE_CACHE>/osf_v1", "destination": f"<DRIVE_ROOT>/{rel}", "files": len(files),
          "bytes": nbytes, "copied_now": len(sums), "uncached_readback_match": int(sum(reread.values())),
          "SHA256SUMS_sha256": sha(root / "SHA256SUMS"),
          "read_back": "every copied file re-read with F_NOCACHE (uncached read; not a physical cold-disk read)",
          "restore_seed": seed, "deployable_best": best, "restore_checks": restores, "deleted": "nothing"}
    ri = {"schema": "osf-restore-index-v1", "private_local": "<PRIVATE_CACHE>/osf_v1",
          "drive_copy": f"<DRIVE_ROOT>/{rel}/osf_v1", "checksums": f"<DRIVE_ROOT>/{rel}/SHA256SUMS",
          "layout": {"run/units/<unit>/": "atomic unit: files + record.json + COMPLETE.json (sha256 of every file)",
                     "admitted/<smf unit>/": "admitted smf checkpoints (ADMISSION.json)",
                     "admitted/fare_cache/<uid>/": "admitted official FARE trees"},
          "restore": [f"copy <DRIVE_ROOT>/{rel}/osf_v1 to <PRIVATE_CACHE>/osf_v1",
                      "verify: shasum -a 256 -c SHA256SUMS (run inside the copy's parent)",
                      "inputs: <PRIVATE_CACHE>/jcv_v1/inputs/adult_jcv.npz (sha256 e0d9e54a...5f12)",
                      "then follow QUICKSTART.md"],
          "restore_checks": restores}
    write_public(PKG / "BACKUP_VERIFICATION.json", bv)
    write_public(PKG / "RESTORE_INDEX.json", ri)
    (root / "BACKUP_RECORD.json").write_text(json.dumps(bv, indent=1, default=str) + "\n")
    print(json.dumps({k: bv[k] for k in ("status", "files", "uncached_readback_match")}, indent=1))
    return bv


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=("predecessor", "backup"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--restore", action="store_true", help="predecessor: run the pinned replay if the drive is mounted")
    ap.add_argument("--dest", default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--targets", default=None)
    a = ap.parse_args(argv)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("REFUSED: OMP_NUM_THREADS must be 1")
    if a.job == "predecessor":
        predecessor(restore=a.restore, dry_run=a.dry_run)
    else:
        backup(a.dest, a.seed, a.targets, a.dry_run)


if __name__ == "__main__":
    main()
