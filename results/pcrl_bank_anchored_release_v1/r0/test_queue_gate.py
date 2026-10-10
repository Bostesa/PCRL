"""Data-free stage admission checks for this campaign's two identity gates."""

import hashlib

from .compute_queue import Queue


def test_protection_requires_bank_and_channel_identity(tmp_path):
    queue = Queue(tmp_path / "queue")
    seal = tmp_path / "seal.json"
    seal.write_text("{}")
    state = queue.read()
    state["seal"] = {"path": str(seal),
                     "sha256": hashlib.sha256(seal.read_bytes()).hexdigest()}
    spec = {"stage": "protection", "max_wall_s": 1,
            "projected_output_bytes": 0, "sources": [],
            "cpu_reserve_s": 1, "max_rss_gib": 0.1,
            "deps": []}
    assert queue._admit_reason(state, spec) == "bank_identity_gate"
    state["milestones"]["witness_pass"] = {"path": str(seal),
                                             "sha256": state["seal"]["sha256"]}
    assert queue._admit_reason(state, spec) == "bank_channel_identity_gate"
