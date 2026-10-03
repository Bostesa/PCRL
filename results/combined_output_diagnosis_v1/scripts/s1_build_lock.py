"""Build LOCK.json before any new scientific fit."""
import hashlib, json, sys, time
from pathlib import Path
WT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WT))
from odx import lock as LK, family as F, run as R
PKG = WT / "results/combined_output_diagnosis_v1"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
lock = {"schema": "odx_lock/v1", "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "LOCKED before any new attacker/head/defense fit (repeatedly used development data; not confirmation)",
        "source_pin": "f7425b15cb89bee841f5b1f9dd290d05a879ae1e", "worktree_head_at_build": LK.git_head(),
        "code_files": LK.code_files(), "dependencies": LK.deps(), "admitted": LK.inputs_state(),
        "pairs": [list(p) for p in R.PAIRS], "primary_cells": sorted([list(p) for p in R.PRIMARY]),
        "surfaces": list(R.SURFACE_ORDER), "bank_candidates": list(R.BANK_CANDIDATES), "reuse": R.REUSE_OAR,
        "coalition": R.COALITION,
        "families": {"primary_size": F.PRIMARY_SIZE, "primary_ids": [e["id"] for e in F.PRIMARY], "primary": F.PRIMARY,
                     "expected_not_estimable": F.EXPECTED_NOT_ESTIMABLE, "declared_aliases": F.DECLARED_ALIASES,
                     "banks": {"io": list(R.IO_BANK_CANDIDATES), "full": list(R.BANK_CANDIDATES), "offset_using": list(R.OFFSET_USING),
                               "selection": "NL attacker-seed-0 attacker_val log loss recorded in each unit; once per encoder seed; ties earlier listed; L not a candidate"},
                     "S3_size": F.S3_SIZE, "S4_size": F.S4_SIZE, "S5_size": F.S5_SIZE, "S5": F.S5,
                     "alpha": F.ALPHA, "z_primary": F.Z_PRIMARY, "z_S3": F.z_two_sided(F.S3_SIZE),
                     "z_S4": F.z_two_sided(F.S4_SIZE), "z_S5": F.z_two_sided(F.S5_SIZE),
                     "bootstrap": {"B": F.B_SE, "seed": F.SEED_SE, "unit": "assessment record group (canon_key unit)",
                                   "paired": "same multinomial draws for every unit of a dataset", "se": "sd ddof=1",
                                   "interval": "point +/- z*SE (normal approximation)"}},
        "file_sha256": {n: sha(PKG / n) for n in ("PROTOCOL.md", "UNIT_MANIFEST.csv", "COVERAGE_AND_SUPPORT.csv", "CUSTODY.json",
                                                   "ORIGINAL_VS_REPAIRED.csv", "AMENDMENT_R_2026-10-03.md", "EXACTNESS.json",
                                                   "notes/review/STATS_REVIEW.md", "notes/review/SCORE_MATH.md")},
        "runtime": {"elapsed_hours_ceiling": 10, "cpu_hours_ceiling": 12, "workers": 2, "memory_gib": 6,
                    "runner_cpu_s_budget": 6 * 3600, "reserve": "final 90 minutes for verification, backup, reporting"}}
(PKG / "LOCK.json").write_text(json.dumps(lock, indent=1))
print(LK.verify_lock(PKG / "LOCK.json"))
