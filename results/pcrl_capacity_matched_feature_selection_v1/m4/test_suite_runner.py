"""Queue-only data-free M4 synthetic test receipt.

All tests use generated arrays or temporary files. This runner has no role
loader call and executes pytest in-process under the shared S3 semaphore.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
TESTS = (
    "test_views.py", "test_endpoints.py", "test_raw_union.py",
    "test_release_bundle.py", "test_attack_runner.py", "test_worker.py",
    "test_endpoint_index.py", "test_endpoint_worker.py",
)
SOURCES = (
    "views.py", "endpoints.py", "raw_union.py", "release_bundle.py",
    "attack_runner.py", "v2_runner.py", "worker.py", "endpoint_index.py",
    "endpoint_worker.py",
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    expected = HERE.parent / ".private/m4/tests/M4_TEST_RECEIPT.json"
    if output != expected or output.exists():
        raise PermissionError("M4 data-free test receipt path is not fresh/fixed")
    result = pytest.main([str(HERE / name) for name in TESTS] + ["-q"])
    if int(result) != 0:
        raise RuntimeError(f"M4 data-free synthetic tests failed: {int(result)}")
    output.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    receipt = {
        "schema": "pcrl-focused-m4-data-free-test-receipt-v1",
        "status": "PASS", "adult_role_values_accessed": False,
        "tests": {name: _sha(HERE / name) for name in TESTS},
        "sources": {name: _sha(HERE / name) for name in SOURCES},
        "pytest_exit_code": 0,
    }
    encoded = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"schema": receipt["schema"], "status": "PASS",
                      "receipt_sha256": _sha(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
