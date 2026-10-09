"""Run presealed focused units one at a time through the shared queue.

This is an operator convenience wrapper. It cannot create or change a unit
spec; the private registry and effective lock determine every worker byte.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .queue import FocusedQueue, sha256


def run_unit(study: Path, unit_id: str, evidence: str) -> dict:
    study = study.resolve(strict=True)
    if not evidence.strip():
        raise ValueError("current external-job occupancy evidence is required")
    resource = json.loads((study / "RESOURCE_PLAN.json").read_text())
    prior = (study.parents[1] / "../pcrl-method-portfolio-v1/results/"
             "pcrl_method_portfolio_v1").resolve(strict=True)
    slot = (prior / "shared/.private/compute/slot0.lock").resolve(strict=True)
    queue = FocusedQueue(study / ".private/queue", slot,
                         epoch_start_utc=resource["epoch_start_utc"])
    seal = (study / "LOCAL_SEAL_RECEIPT.json").resolve(strict=True)
    queue.admit_science(seal_path=seal, seal_sha256=sha256(seal),
                        external_heavy=1, evidence=evidence)
    registry_path = study / ".private/UNIT_SPECS.json"
    lock = json.loads((study / "CAMPAIGN_LOCK.json").read_text())
    if sha256(registry_path) != lock["private_unit_registry_sha256"]:
        raise PermissionError("potential unit registry changed")
    registry = json.loads(registry_path.read_text())
    sealed_spec = {**registry["unit_specs"][unit_id],
                   "seal_sha256": sha256(seal)}
    state = queue.status()
    if unit_id not in state["units"]:
        queue.enqueue(sealed_spec)
    else:
        prior = state["units"][unit_id]
        prior_status = prior["status"]
        if prior_status == "DONE":
            raise FileExistsError("unit already DONE; preserve its identity")
        if prior_status != "QUEUED":
            raise RuntimeError("unit has terminal or running receipt: " + prior_status)
        spec_path = Path(prior["spec_path"]).resolve(strict=True)
        if (sha256(spec_path) != prior["spec_sha256"] or
                json.loads(spec_path.read_text()) != sealed_spec):
            raise PermissionError("preexisting queued unit differs from sealed registry")
    result = queue.run_once(expected_unit_id=unit_id)
    if result.get("unit_id") != unit_id or result.get("status") != "DONE":
        raise RuntimeError("requested unit did not finish DONE: " + json.dumps(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True)
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--external-evidence", required=True)
    args = parser.parse_args()
    print(json.dumps(run_unit(Path(args.study), args.unit_id,
                              args.external_evidence), sort_keys=True))


if __name__ == "__main__":
    main()
