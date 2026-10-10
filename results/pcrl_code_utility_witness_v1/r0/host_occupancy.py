"""Write a short-lived private host-process receipt for queue admission.

The sandbox denies ps/pgrep. lsof exposes process names and PIDs without
reading any scientific arrays. R0 conservatively reserves one of two host
heavy slots for another session even if that process appears idle.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import subprocess

from .compute_queue import ROOT


def main() -> None:
    result = subprocess.run(["lsof", "-nP", "-Fpc"], capture_output=True,
                            text=True, timeout=20, check=False)
    if result.returncode not in (0, 1):
        raise RuntimeError("lsof process inventory unavailable; no capacity receipt")
    current = None
    processes = {}
    for line in result.stdout.splitlines():
        if line.startswith("p") and line[1:].isdigit():
            current = int(line[1:])
        elif line.startswith("c") and current is not None:
            processes[current] = line[1:]
    if not processes:
        raise RuntimeError("empty process inventory; no capacity receipt")
    candidates = [{"pid": pid, "name": name} for pid, name in sorted(processes.items())
                  if name.lower().startswith(("python", "rscript", "julia", "matlab"))
                  and pid != os.getpid()]
    root = ROOT
    root.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema": "pcrl-host-occupancy-v1",
        "checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "inventory_method": "lsof -nP -Fpc; ps and pgrep sandbox denied",
        "scientific_process_candidates": candidates,
        "conservative_external_heavy_reserved": 1,
        "campaign_local_heavy_slots": 1,
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
