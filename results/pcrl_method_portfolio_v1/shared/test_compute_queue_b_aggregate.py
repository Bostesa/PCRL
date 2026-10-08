"""Synthetic admission proof for the frozen Team B aggregate stage.

This test inspects hashes and queue specs. It does not run the stage or open
role exports, reader vectors, checkpoints, or person-level aggregates.
"""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from compute_queue import Queue, THREAD_ENV, sha256_file


CAMPAIGN = Path(__file__).resolve().parents[1]
COORD = CAMPAIGN.parents[1]
TEAM = COORD.parent / "pcrl-method-portfolio-b"
TEAM_CAMPAIGN = TEAM / "results/pcrl_method_portfolio_v1"
SCRIPT = TEAM_CAMPAIGN / "teams/b/stage_b_screen_comparison.py"
PINS = TEAM_CAMPAIGN / "teams/b/B_AGGREGATE_STAGE_PINS.json"
S4_INDEX = CAMPAIGN / "S4_INDEPENDENT_NUMERICAL_REPLAY_INDEX.json"
OUTDIR = TEAM_CAMPAIGN / "teams/b/.private/b_aggregate_stage_verified_001"
UNIT = "B-AGGREGATE-VERIFIED-STAGE-083"


def exact_spec() -> dict:
    assert SCRIPT.is_file() and PINS.is_file() and S4_INDEX.is_file()
    assert not OUTDIR.exists(), "the aggregate output is write-once"
    s4_sha = sha256_file(S4_INDEX)
    return {
        "unit_id": UNIT,
        "mode": "aggregate_metadata",
        "stage": "verified_aggregate_stage",
        "family": "B",
        "team": "B",
        "arm": "B_MATCHED_POOL",
        "arm_id": "B_MATCHED_POOL",
        "seed": None,
        "data_access": "FROZEN_B_AND_S4_AGGREGATES_ONLY",
        "cwd": str(TEAM),
        "argv": [
            str(COORD.parent.parent / ".venv/bin/python"), str(SCRIPT),
            "--coord-root", str(COORD),
            "--team-root", str(TEAM),
            "--pins", str(PINS),
            "--s4-index-sha256", s4_sha,
            "--output-dir", str(OUTDIR),
        ],
        "deps": [],
        "max_wall_s": 600,
        "cpu_reserve_s": 600,
        "max_rss_gib": 2.0,
        "env": {**THREAD_ENV, "PYTHONDONTWRITEBYTECODE": "1"},
        "outputs": [str(OUTDIR / name) for name in (
            "STAGED_18_ROWS.json", "STAGED_18_ROWS.csv", "CONTROL_SELECTION.json")],
        "runner_sha256": sha256_file(SCRIPT),
        "pins_sha256": sha256_file(PINS),
        "s4_index_sha256": s4_sha,
        "campaign_lock_sha256":
            "956061db72ee4cef6893e0e3f98d6ad4a85b57c4e6aa0a15856c084d5653c39e",
    }


class AggregateMetadataAdmissionTests(unittest.TestCase):
    def test_exact_frozen_aggregate_spec_enqueues_without_running(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            queue = Queue(Path(directory) / "queue")
            item = queue.enqueue(exact_spec())
            self.assertEqual(item["status"], "planned")
            self.assertFalse(OUTDIR.exists())

    def test_scope_identity_and_provenance_mutations_are_denied(self) -> None:
        base = exact_spec()
        mutations = [
            {"unit_id": "B-AGGREGATE-UNLISTED"},
            {"stage": "screen_score"},
            {"family": "C"},
            {"team": "C"},
            {"arm_id": "B-TASK"},
            {"data_access": "NONE"},
            {"seed": 0},
            {"runner_sha256": "f" * 64},
            {"pins_sha256": "f" * 64},
            {"s4_index_sha256": "f" * 64},
            {"campaign_lock_sha256": "f" * 64},
            {"deps": ["S4-B-TASK-S0"]},
            {"argv": [base["argv"][0], "-c", "pass"]},
            {"outputs": [str(OUTDIR / "wrong.json")]},
            {"max_wall_s": 601, "cpu_reserve_s": 601},
            {"cpu_reserve_s": 601},
            {"max_rss_gib": 2.1},
            {"env": {**base["env"], "PYTHONPATH": str(TEAM)}},
        ]
        for patch in mutations:
            with self.subTest(patch=patch), tempfile.TemporaryDirectory() as directory:
                queue = Queue(Path(directory) / "queue")
                with self.assertRaises(ValueError):
                    queue.enqueue({**copy.deepcopy(base), **patch})


if __name__ == "__main__":
    unittest.main()
