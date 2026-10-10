"""Control-only R03 release entrypoint using R12's exact cached wire worker.

R12's release worker owns source/role/stream/parent FIT lineage verification and
the common paired cache. This wrapper refuses non-control arms and pins its own
source bytes before entering that worker. It opens no Adult role file itself.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from results.pcrl_latent_and_sampled_methods_v1.r03.compute_queue import assert_active_from_env
from ..impl.admitted_fit import _sha
from ..impl.run_release import run as run_common_release
from .control_dispatch import ARMS


RELATIVE_SOURCE = "sampled_channel_v1/controls/run_control_release.py"


def run(args: argparse.Namespace) -> dict:
    """Delegate only a source-pinned, queue-owned control release unit."""
    active = assert_active_from_env(unit_id=args.unit_id, stage="a")
    if args.arm_id not in ARMS:
        raise PermissionError("release entrypoint accepts only R13 control arms")
    campaign = Path(args.campaign_root).resolve(strict=True)
    lock = Path(args.original_lock).resolve(strict=True)
    amendment_file = Path(args.a_amendment).resolve(strict=True)
    if (campaign != lock.parent or _sha(lock) != active["seal_sha256"] or
            _sha(amendment_file) != args.a_amendment_sha256):
        raise PermissionError("control release effective seal differs")
    amendment = json.loads(amendment_file.read_text())
    source = campaign / RELATIVE_SOURCE
    expected = amendment.get("implementation_source_sha256", {}).get(RELATIVE_SOURCE)
    state = json.loads((Path(os.environ["PCRL_QUEUE_ROOT"]) / "state.json").read_text())
    spec = state.get("units", {}).get(args.unit_id, {}).get("spec", {})
    listed = {Path(row["path"]).resolve(): row["sha256"]
              for row in spec.get("sources", [])}
    if (type(expected) is not str or len(expected) != 64 or
            listed.get(source.resolve(strict=True)) != expected or
            _sha(source) != expected or
            spec.get("arm_id") != args.arm_id or
            spec.get("outputs") != [str(Path(args.private_output).resolve(strict=False) /
                                        "RELEASE_RECEIPT.json")]):
        raise PermissionError("control release entrypoint source/output differs")
    return run_common_release(args)


def main() -> None:
    parser = argparse.ArgumentParser()
    for key in ("unit_id", "arm_id", "original_lock", "a_amendment",
                "a_amendment_sha256", "private_output", "repair_manifest",
                "source_audit", "portfolio_package", "campaign_root",
                "policy_dir", "private_stream_roots"):
        parser.add_argument(f"--{key.replace('_', '-')}", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--realization", type=int, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
