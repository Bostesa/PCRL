"""Coordinator-dispatched common inner audit with a pinned alias index."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

from . import evaluate, release_specs


def _sha(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_or_verify(path: Path, record: dict) -> None:
    encoded = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise ValueError(f"existing {path.name} differs")
        return
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(encoded)
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def _verify_complete(output: Path, canonical: dict, anchor: int,
                     slate: str, input_sha: str) -> dict:
    """Recover a complete immutable panel without fitting a second time."""
    receipt = json.loads((output / "COMPLETE.json").read_text())
    report = json.loads((output / "INNER_AUDIT.json").read_text())
    ids = sorted(canonical)
    if (receipt.get("schema") != "pcrl-adaptive-inner-audit-complete-v1" or
            receipt.get("anchor") != anchor or
            receipt.get("release_ids") != ids or
            receipt.get("index_sha256") != input_sha or
            receipt.get("artifacts") != evaluate._inventory(output) or
            report.get("schema") != "pcrl-adaptive-inner-audit-v1" or
            report.get("anchor") != anchor or report.get("slate") != slate or
            report.get("index_sha256") != input_sha or
            report.get("outer_pool_opened") is not False or
            sorted(report.get("releases", {})) != ids):
        raise ValueError("completed inner panel identity or artifact inventory differs")
    for release_id, spec in canonical.items():
        descriptor = report["releases"][release_id].get("source", {})
        if "task_only_model_dir" in spec:
            observed = descriptor.get("task_only_receipt_artifact", {}).get("sha256")
            expected = spec["task_only_receipt_sha256"]
        else:
            observed = descriptor.get("channel_artifact", {}).get("sha256")
            expected = spec["channel_artifact_sha256"]
            if callable(spec.get("router")) and descriptor.get(
                    "router_artifact", {}).get("sha256") != spec["router_sha256"]:
                raise ValueError("completed inner panel router source differs")
        if observed != expected:
            raise ValueError("completed inner panel release source differs")
    return report


def dispatch_inner(anchor: int, delta: float, input_index: str | Path,
                   a_center_dir: str | Path, output_dir: str | Path,
                   spec_index: str | Path, *,
                   b_center_dir: str | Path | None = None,
                   b_parity_receipt: str | Path | None = None,
                   a_controls_dir: str | Path | None = None,
                   b_controls_dir: str | Path | None = None,
                   task_only_dir: str | Path | None = None,
                   slate: str = "catchup", resume: bool = False) -> dict:
    """Audit each distinct law once while preserving every declared name."""
    spec_path = Path(spec_index)
    output = Path(output_dir)
    if ("private" not in spec_path.parts or "private" not in output.parts or
            spec_path.is_relative_to(output)):
        raise ValueError("audit panel and separate spec index must stay private")
    bundle = release_specs.build_release_specs(
        anchor, delta, a_center_dir,
        b_center_dir=b_center_dir, b_parity_receipt=b_parity_receipt,
        a_controls_dir=a_controls_dir, b_controls_dir=b_controls_dir,
        task_only_dir=task_only_dir)
    manifest = {key: value for key, value in bundle.items() if key != "releases"}
    manifest.update({"schema": "pcrl-inner-audit-spec-index-v1",
                     "anchor": anchor, "delta": delta, "slate": slate,
                     "canonical_ids": sorted(bundle["releases"]),
                     "input_index_sha256": _sha(input_index),
                     "source_module_sha256": _sha(release_specs.__file__)})
    spec_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _write_or_verify(spec_path, manifest)
    if (output / "COMPLETE.json").is_file():
        report = _verify_complete(output, bundle["releases"], anchor,
                                  slate, manifest["input_index_sha256"])
    else:
        report = evaluate.audit_panel(anchor, bundle["releases"], input_index,
                                      output, resume=resume, slate=slate)
        report = _verify_complete(output, bundle["releases"], anchor,
                                  slate, manifest["input_index_sha256"])
    if sorted(report["releases"]) != manifest["canonical_ids"]:
        raise RuntimeError("completed audit differs from pinned release names")
    binding = {"schema": "pcrl-inner-panel-binding-v1", "anchor": anchor,
               "slate": slate, "canonical_ids": manifest["canonical_ids"],
               "source_receipts": manifest["source_receipts"],
               "spec_index_sha256": _sha(spec_path),
               "inner_complete_sha256": _sha(output / "COMPLETE.json"),
               "inner_audit_sha256": _sha(output / "INNER_AUDIT.json")}
    _write_or_verify(spec_path.with_suffix(".binding.json"), binding)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--delta", type=float, choices=(0., .001, .003), required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--a-center-dir", required=True)
    parser.add_argument("--b-center-dir")
    parser.add_argument("--b-parity-receipt")
    parser.add_argument("--a-controls-dir")
    parser.add_argument("--b-controls-dir")
    parser.add_argument("--task-only-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--spec-index", required=True)
    parser.add_argument("--slate", choices=("standard", "catchup"), default="catchup")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    report = dispatch_inner(
        args.anchor, args.delta, args.index, args.a_center_dir,
        args.output_dir, args.spec_index,
        b_center_dir=args.b_center_dir, b_parity_receipt=args.b_parity_receipt,
        a_controls_dir=args.a_controls_dir, b_controls_dir=args.b_controls_dir,
        task_only_dir=args.task_only_dir, slate=args.slate, resume=args.resume)
    print(json.dumps({"anchor": args.anchor,
                      "canonical_releases": sorted(report["releases"]),
                      "outer_pool_opened": report["outer_pool_opened"]},
                     sort_keys=True))


if __name__ == "__main__":
    main()
