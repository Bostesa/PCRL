"""SOURCE_INDEX.json for the decision-preserving compression study (data/custody owner).

Pins the source study's files (prompt section 1) by their blob sha256 at the exact evidence commit, checks that
origin/research/pcrl-online-strength-frontier-v1 still equals that commit (git ls-remote, read-only), that the
working-tree copies on this branch are byte-identical to the pinned blobs, and that the admitted input file hashes to
the pinned value. Publishes hashes, sizes and booleans only.

    OMP_NUM_THREADS=1 PYTHONPATH=. ~/PCRL/.venv/bin/python results/pcrl_decision_preserving_compression_v1/provenance/source_index.py
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
PIN = "925e0fddfcb666116c6179575339728a324ed78e"
BRANCH = "research/pcrl-online-strength-frontier-v1"
STUDY_BRANCH = "research/pcrl-decision-preserving-compression-v1"
OSF = "results/pcrl_online_strength_frontier_v1"
INPUT = Path.home() / "PCRL_eval_cache_private" / "jcv_v1" / "inputs" / "adult_jcv.npz"
INPUT_SHA = "e0d9e54af780f30788ee29cfe6795ec82cbdcadc127b1978c69a3891485d2f12"
SECTION_1 = [f"{OSF}/{f}" for f in (
    "RESEARCH_DECISION.md", "VALIDATION.md", "METHOD_CARD.md", "PROTOCOL.md", "SOURCE_INDEX.json",
    "MODEL_MANIFEST.json", "ROLE_MANIFEST.json", "ADMISSION.json",
    "ALL_LEVELS.csv", "ACTUAL_TASK_UTILITY.csv", "PRIMARY_ENDPOINTS.csv", "SECONDARY_ENDPOINTS.csv",
    "PREDECESSOR_CUSTODY_REPAIR.json", "COST_AND_CLOSEOUT.md", "QUICKSTART.md")] + [
    "osf/data.py", "osf/audit.py", "osf/assess.py", "osf/select.py", "rgj/finalize.py", "jcv/finalize.py"]
ALSO_USED = [f"{OSF}/{f}" for f in ("EXPOSURE_LEDGER.md", "BACKUP_VERIFICATION.json", "RESTORE_INDEX.json",
                                    "EVALUATION_LOCK.json", "provenance/role_check.py")] + [
    "osf/closeout.py", "osf/deploy.py", "osf/run.py", "smf/data.py", "rgj/data.py", "jcv/train.py",
    "stored_model_eval/defenses.py", "oar/fare_official.py"]


def git(*a):
    r = subprocess.run(["git", "-C", str(WT), *a], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def entry(rel):
    blob = git("show", f"{PIN}:{rel}")
    if blob is None:
        return {"present_at_pin": False}
    wt = WT / rel
    return {"sha256_at_pin": sha_bytes(blob), "bytes": len(blob),
            "git_blob_id_at_pin": (git("rev-parse", f"{PIN}:{rel}") or b"").decode().strip(),
            "working_tree_equals_pin": wt.exists() and wt.read_bytes() == blob}


def main():
    ls = git("ls-remote", "origin", f"refs/heads/{BRANCH}")
    remote = ls.decode().split()[0] if ls else None
    tracking = (git("rev-parse", f"origin/{BRANCH}") or b"").decode().strip() or None
    head = (git("rev-parse", "HEAD") or b"").decode().strip()
    pin_is_ancestor = subprocess.run(["git", "-C", str(WT), "merge-base", "--is-ancestor", PIN, "HEAD"]).returncode == 0
    idx = {"schema": "dpc-source-index-v1", "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "source_study": "pcrl_online_strength_frontier_v1 (osf)", "source_branch": BRANCH, "pinned_commit": PIN,
           "remote_check": {"method": "git ls-remote origin refs/heads/" + BRANCH + " (read-only) and the local "
                                      "remote-tracking ref",
                            "remote_head_of_source_branch": remote, "remote_equals_pin": remote == PIN,
                            "remote_tracking_ref": tracking, "remote_tracking_equals_pin": tracking == PIN,
                            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
           "study_branch": STUDY_BRANCH, "study_branch_created_from_pin": pin_is_ancestor,
           "study_head_at_index": head,
           "input": {"file": "<PRIVATE_CACHE>/jcv_v1/inputs/adult_jcv.npz", "sha256_pinned": INPUT_SHA,
                     "sha256_recomputed": sha_file(INPUT), "matches": sha_file(INPUT) == INPUT_SHA,
                     "bytes": INPUT.stat().st_size},
           "read_before_implementation (prompt section 1)": {r: entry(r) for r in SECTION_1},
           "also_used_by_data_custody": {r: entry(r) for r in ALSO_USED},
           "private_sources": {"osf_units": "<PRIVATE_CACHE>/osf_v1/run/units (read-only; admitted copies in "
                                            "<PRIVATE_CACHE>/dpc_v1/admitted, ADMISSION.json)",
                               "osf_fare_trees": "<PRIVATE_CACHE>/osf_v1/admitted/fare_cache (read-only)",
                               "smf_store": "<PRIVATE_CACHE>/smf_v1 (read-only; predecessor custody only)",
                               "smf_custody_supplement": "<PRIVATE_CACHE>/smf_v1_custody_supplement_20261005 "
                                                         "(read-only)"}}
    allf = {**idx["read_before_implementation (prompt section 1)"], **idx["also_used_by_data_custody"]}
    idx["all_present_at_pin"] = all(v.get("present_at_pin", True) for v in allf.values())
    idx["all_working_tree_equal_pin"] = all(v.get("working_tree_equals_pin") for v in allf.values())
    idx["files_differing_from_pin"] = sorted(r for r, v in allf.items() if not v.get("working_tree_equals_pin"))
    txt = json.dumps(idx, indent=1) + "\n"
    if re.search(r"/Users/|/Volumes/|" + re.escape(Path.home().name), txt):
        raise SystemExit("REFUSED: identifying path in a public file")
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(txt)
    tmp.replace(OUT)
    print(json.dumps({"remote_equals_pin": idx["remote_check"]["remote_equals_pin"],
                      "input_matches": idx["input"]["matches"], "all_working_tree_equal_pin":
                      idx["all_working_tree_equal_pin"], "differing": idx["files_differing_from_pin"]}, indent=1))


if __name__ == "__main__":
    main()
