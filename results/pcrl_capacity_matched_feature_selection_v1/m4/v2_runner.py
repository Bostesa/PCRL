"""Write-once real V2 null, shuffle and planted probes per complete view."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

from attack_controls_v2 import run_control_probes_v2, save_control_probe_v2_custody


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_real_view_controls(
    *, att_view: object, sex_att: Mapping[str, object],
    inner_view: object, sex_inner: Mapping[str, object],
    seed: int, pair_id: str, view_name: str,
    release_index_sha256: str, lock_sha256: str,
    manifest_sha256: str, private_output: str | Path,
) -> dict:
    """Run exactly two S2 LR fits on an engineering copy of one real wire."""
    if view_name not in ("income", "occupation_group", "coalition"):
        raise ValueError("real V2 probe needs one complete view")
    out = Path(private_output).resolve()
    if ".private" not in out.parts or out.exists():
        raise PermissionError("real V2 probe output must be a fresh private directory")
    report, artifacts = run_control_probes_v2(
        att_view, dict(sex_att), inner_view, dict(sex_inner),
        return_artifacts=True,
    )
    out.mkdir(parents=True, mode=0o700)
    custody = save_control_probe_v2_custody(artifacts, out / "probe_custody")
    index = {
        "schema": "pcrl-focused-real-v2-view-index-v1",
        "status": report["status"], "seed": seed,
        "pair_id": pair_id, "view_name": view_name,
        "release_index_sha256": release_index_sha256,
        "lock_sha256": lock_sha256, "manifest_sha256": manifest_sha256,
        "att_view_content_sha256": att_view.content_hashes(),
        "inner_view_content_sha256": inner_view.content_hashes(),
        "probe": report, "probe_custody": custody,
        "fresh_fit_count": 2,
    }
    path = out / "V2_INDEX.json"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(index, stream, sort_keys=True, separators=(",", ":"),
                  allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return index
