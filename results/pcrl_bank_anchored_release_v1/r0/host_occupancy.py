"""Write a short-lived private host-process receipt for queue admission.

The sandbox denies ps/pgrep. lsof exposes process names and PIDs without
reading any scientific arrays. R0 conservatively counts external numerical
process candidates against the two host heavy slots.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import subprocess

from .compute_queue import ROOT


def scientific_candidates(listing: str, current_pid: int) -> list[dict]:
    current = None
    processes = {}
    for line in listing.splitlines():
        if line.startswith("p") and line[1:].isdigit():
            current = int(line[1:])
        elif line.startswith("c") and current is not None:
            processes[current] = line[1:]
    return [{"pid": pid, "name": name} for pid, name in sorted(processes.items())
            if name.lower().startswith(("python", "rscript", "julia", "matlab"))
            and pid != current_pid]


def main() -> None:
    result = subprocess.run(["lsof", "-nP", "-Fpc"], capture_output=True,
                            text=True, timeout=20, check=False)
    if result.returncode not in (0, 1):
        raise RuntimeError("lsof process inventory unavailable; no capacity receipt")
    if not result.stdout.strip():
        raise RuntimeError("empty process inventory; no capacity receipt")
    candidates = scientific_candidates(result.stdout, os.getpid())
    external_heavy = min(2, len(candidates))
    root = ROOT
    root.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema": "pcrl-host-occupancy-v1",
        "checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "inventory_method": "lsof -nP -Fpc; ps and pgrep sandbox denied",
        "scientific_process_candidates": candidates,
        "conservative_external_heavy_reserved": external_heavy,
        "campaign_local_heavy_slots": min(1, max(0, 2 - external_heavy)),
        "scope": "process names and PIDs only; no source or private scientific file opened",
    }
    path = root / "host_occupancy.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    print(path)


if __name__ == "__main__":
    main()
