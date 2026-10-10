"""Write an immutable-source R03 spec for R04's small independent fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[2]
FILES = (
    "independent_arithmetic.py", "independent_d_restore.py", "independent_d_view.py",
    "independent_release.py", "independent_replay.py", "test_independent_arithmetic.py",
    "test_independent_d_restore.py", "test_independent_d_view.py",
    "test_independent_release.py", "test_independent_replay.py",
)


def main() -> None:
    python = Path("/Users/nathansamson/PCRL/.venv/bin/python")
    if not python.exists():
        raise SystemExit("project .venv Python missing")
    source = [
        {"path": str(HERE / name),
         "sha256": hashlib.sha256((HERE / name).read_bytes()).hexdigest()}
        for name in FILES
    ]
    spec = {
        "unit_id": "R04-INDEPENDENT-SYNTHETIC-001",
        "stage": "verification",
        "argv": [str(python), "-m", "pytest", "-q", *[str(HERE / name) for name in FILES if name.startswith("test_")]],
        "cwd": str(WORKTREE),
        "max_wall_s": 90,
        "cpu_reserve_s": 90,
        "max_rss_gib": 1,
        "projected_output_bytes": 300000,
        "sources": source,
        "deps": [],
        "outputs": [],
    }
    target = HERE / "R04_SYNTHETIC_QUEUE_SPEC.json"
    target.write_text(json.dumps(spec, sort_keys=True, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
