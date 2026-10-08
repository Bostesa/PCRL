"""Build sealed S3 specs for the ten fixed Team A source-bank preparation units.

This utility reads public manifests and lock metadata only. It never reads role
NPZ values, trains a model, enqueues a job, or runs a source-bank adapter.
"""

from __future__ import annotations

from pathlib import Path


PLAN_SHA256 = "06b8e3ae947d23659f7ba46f3e78d9c83a9ce9966656d96d5f766b5984b07565"


def build_specs(plan_path: Path, lock_path: Path, expected_lock_sha256: str,
                coordinator_root: Path, queue_root: Path) -> list[dict]:
    """Pure metadata validation and exact S3 spec construction."""
    raise NotImplementedError("test-first RED: exact A spec construction pending")
