"""SOURCE_INDEX.json for the confidence-capacity study (qpc; data/custody owner E).

Records which source files qpc reuses and their blob hashes at the source evidence commit 0a7b05a (dpc):
  * reused code: every module of a pinned study package (dpc, osf, smf, rgj, jcv, stored_model_eval, oar, pcrl) that a
    qpc/*.py file imports, closed transitively over the imports of those source files (static scan, no import). The
    scan covers `import x`, `from x import y` (also inside functions) and literal `importlib.import_module("x")`; loads
    whose module name comes from DATA (an entry point in a targets file) are listed in DYNAMIC with their reason and
    seed the same transitive closure;
  * read evidence: the dpc result files the prompt requires (section 2) plus the custody files this study builds on;
  * remote check: origin/<source branch> still equals the pin (git ls-remote, read-only);
  * every reused working-tree file must be byte-identical to its blob at the pin.
Publishes hashes, sizes and booleans only. Re-run whenever a qpc module adds a source import (lead, before a lock).

    PYTHONPATH=. ~/PCRL/.venv/bin/python results/pcrl_confidence_capacity_v1/provenance/source_index.py
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
WT = PKG.parents[1]
OUT = PKG / "SOURCE_INDEX.json"
PIN = "0a7b05a52746544213742f50efd0a48167efffb1"
BRANCH = "research/pcrl-decision-preserving-compression-v1"
TEACHER_PIN = "925e0fddfcb666116c6179575339728a324ed78e"
TEACHER_BRANCH = "research/pcrl-online-strength-frontier-v1"
STUDY_BRANCH = "research/pcrl-confidence-capacity-v1"
DPC = "results/pcrl_decision_preserving_compression_v1"
PKGS = ("dpc", "osf", "smf", "rgj", "jcv", "stored_model_eval", "oar", "pcrl")
SECTION_2 = [f"{DPC}/{f}" for f in ("RESEARCH_DECISION.md", "ADVISOR_BRIEF.md", "PROTOCOL.md", "MATH_REVIEW.md",
                                    "VALIDATION.md", "INNER_SELECTION_TABLE.csv", "COALITION_AND_RATE_RESULTS.csv",
                                    "MODEL_MANIFEST.json", "CLASS_PRESERVATION.json", "QUICKSTART.md",
                                    "PRIOR_ART_AND_BASELINE_GAPS.md")]
ALSO_READ = [f"{DPC}/{f}" for f in ("ADMISSION.json", "ROLE_MANIFEST.json", "SOURCE_INDEX.json", "EXPOSURE_LEDGER.md",
                                    "BACKUP_VERIFICATION.json", "RESTORE_INDEX.json", "EVALUATION_LOCK.json",
                                    "COST_AND_CLOSEOUT.md", "METHOD_CARD.md",
                                    "provenance/predecessor_custody/STATUS.json")]
IMPORT_RE = re.compile(r"^\s*(?:from\s+([\w.]+)\s+import\s+([\w, ]+)|import\s+([\w.]+))", re.M)
IMPORTLIB_RE = re.compile(r"import_module\(\s*[\"']([\w.]+)[\"']\s*\)")
# module names that are not literals in any source file: they come from data at run time
DYNAMIC = {"dpc/report.py": ("qpc.closeout dpc-backup -> dpc.closeout.backup -> dpc.closeout.restore_attacker loads the "
                             "attacker entry point 'dpc.report:refit_selected_attacker' named in <PRIVATE_CACHE>/dpc_v1/"
                             "run/closeout_targets.json (importlib, module name from data)")}
SUBPROCESS_NOTE = ("osf.closeout predecessor --restore runs results/pcrl_strength_matched_feedback_v1/verification/"
                   "replay_smf.py as a SUBPROCESS on the closed smf source worktree at its own pin (not imported from this "
                   "tree; osf.closeout checks that worktree is clean at the pin before and after)")


def git(*a, text=False):
    r = subprocess.run(["git", "-C", str(WT), *a], capture_output=True, text=text)
    return r.stdout if r.returncode == 0 else None


def module_files(src: str):
    """Pinned-package source files imported by `src` (python text)."""
    out = set()
    for frm, names, imp in IMPORT_RE.findall(src):
        mod = frm or imp
        top = mod.split(".")[0]
        if top not in PKGS:
            continue
        cands = [mod.replace(".", "/") + ".py"]
        if frm:
            cands += [f"{mod.replace('.', '/')}/{n.split()[0]}.py" for n in names.split(",") if n.strip()]
        for c in cands:
            if (WT / c).is_file():
                out.add(c)
    for mod in IMPORTLIB_RE.findall(src):
        c = mod.replace(".", "/") + ".py"
        if mod.split(".")[0] in PKGS and (WT / c).is_file():
            out.add(c)
    return out


def reused_code():
    """{file: [qpc importers]} closed transitively over the pinned packages' own imports."""
    users = {}
    todo = []
    for p in sorted((WT / "qpc").glob("*.py")):
        for f in module_files(p.read_text()):
            users.setdefault(f, set()).add(str(p.relative_to(WT)))
            todo.append(f)
    for f, why in DYNAMIC.items():
        users.setdefault(f, set()).add(f"<dynamic: {why}>")
        todo.append(f)
    seen = set()
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        for g in module_files((WT / f).read_text()):
            users.setdefault(g, set()).add(f)
            todo.append(g)
    return {f: sorted(u) for f, u in sorted(users.items())}


def entry(rel, pin=PIN):
    blob = git("show", f"{pin}:{rel}")
    if blob is None:
        return {"present_at_pin": False}
    wt = WT / rel
    return {"present_at_pin": True, "sha256_at_pin": hashlib.sha256(blob).hexdigest(), "bytes": len(blob),
            "git_blob_id_at_pin": (git("rev-parse", f"{pin}:{rel}", text=True) or "").strip(),
            "working_tree_equals_pin": wt.exists() and wt.read_bytes() == blob}


def remote(branch, pin):
    out = git("ls-remote", "origin", f"refs/heads/{branch}", text=True) or ""
    head = out.split()[0] if out.strip() else None
    track = (git("rev-parse", f"origin/{branch}", text=True) or "").strip() or None
    return {"method": f"git ls-remote origin refs/heads/{branch} (read-only) and the local remote-tracking ref",
            "remote_head": head, "remote_equals_pin": head == pin, "remote_tracking_ref": track,
            "remote_tracking_equals_pin": track == pin, "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def main():
    code = reused_code()
    code_entries = {f: {**entry(f), "imported_by": u} for f, u in code.items()}
    for f in [x for x in ("osf/data.py",) if x in code_entries]:   # also pinned at the teacher-provenance commit
        code_entries[f]["equals_blob_at_teacher_pin"] = git("show", f"{TEACHER_PIN}:{f}") == (WT / f).read_bytes()
    obj = {"schema": "qpc-source-index-v1", "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "study": "pcrl_confidence_capacity_v1", "study_branch": STUDY_BRANCH,
           "source_study": "pcrl_decision_preserving_compression_v1 (dpc)", "source_branch": BRANCH,
           "source_evidence_commit": PIN, "teacher_provenance_commit": TEACHER_PIN,
           "remote_check": {"source": remote(BRANCH, PIN), "teacher_provenance": remote(TEACHER_BRANCH, TEACHER_PIN)},
           "study_branch_created_from_pin": subprocess.run(
               ["git", "-C", str(WT), "merge-base", "--is-ancestor", PIN, "HEAD"]).returncode == 0,
           "study_head_at_index": (git("rev-parse", "HEAD", text=True) or "").strip(),
           "reused_code": {"rule": "pinned-package modules imported by qpc/*.py, closed transitively (static scan of "
                                   "import statements and literal importlib.import_module calls, plus the DYNAMIC "
                                   "data-named loads); reused by import, unchanged; dpc/ is never edited",
                           "dynamic_loads": DYNAMIC, "subprocess_note": SUBPROCESS_NOTE,
                           "files": code_entries,
                           "all_equal_to_pin": all(v.get("working_tree_equals_pin") for v in code_entries.values())},
           "read_before_implementation (prompt section 2)": {f: entry(f) for f in SECTION_2},
           "also_read": {f: entry(f) for f in ALSO_READ},
           "privacy": "hashes, sizes and booleans only"}
    obj["all_read_files_equal_pin"] = all(v.get("working_tree_equals_pin") for d in
                                          (obj["read_before_implementation (prompt section 2)"], obj["also_read"])
                                          for v in d.values())
    txt = json.dumps(obj, indent=1) + "\n"
    if re.search(r"/Users/|/Volumes/|/private/|" + re.escape(Path.home().name), txt):
        raise SystemExit("REFUSED: identifying path in a public file")
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(txt)
    tmp.replace(OUT)
    print(json.dumps({"reused_code_files": len(code_entries), "all_code_equal_pin": obj["reused_code"]["all_equal_to_pin"],
                      "all_read_files_equal_pin": obj["all_read_files_equal_pin"],
                      "source_remote_equals_pin": obj["remote_check"]["source"]["remote_equals_pin"],
                      "teacher_remote_equals_pin": obj["remote_check"]["teacher_provenance"]["remote_equals_pin"]},
                     indent=1))


if __name__ == "__main__":
    main()
