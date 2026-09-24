"""Emit a locked archived T32 channel as one token; replay public aggregates.

The Linux x86 runtime receives only X_A and H_A. The wire contains exactly
unchanged H_A and one token. A private stable replay key enforces one draw per
immutable ID/input/channel; no repeated-release composition claim is made.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import stat

import numpy as np

from experiments.pcrl_task_aligned_cuts_v1 import release
from experiments.pcrl_task_directed_release_v1.data import RuntimeInputs
from . import evaluate, fit_a, fit_controls, privacy_first_fit, render_results


def _private(path: str | Path) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise ValueError("private artifact path cannot be a symlink")
    resolved = raw.resolve()
    if "private" not in resolved.parts:
        raise ValueError("fitted objects, person inputs, replay key and wire require private paths")
    return resolved


def _member(root: Path, relative: str) -> Path:
    piece = Path(relative)
    candidate = (root / piece).resolve()
    if (piece.is_absolute() or ".." in piece.parts
            or not candidate.is_relative_to(root)
            or candidate.is_symlink() or not candidate.is_file()):
        raise ValueError("unsafe or missing selected channel member")
    return candidate


def _input(path: Path) -> tuple[RuntimeInputs, list, np.ndarray]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("private input NPZ is missing or symbolic")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"x_a", "h_a", "release_ids"}:
            raise ValueError("private input archive must have only allowed x_a, h_a, release_ids fields")
        x, h, ids = (archive[key].copy() for key in ("x_a", "h_a", "release_ids"))
    if ids.ndim != 1 or ids.dtype.kind not in ("i", "u", "U"):
        raise ValueError("private release IDs must be a one-dimensional integer or Unicode array")
    inputs = RuntimeInputs(x, h)
    if len(ids) != len(h):
        raise ValueError("private release IDs and local inputs differ in length")
    return inputs, ids.tolist(), h


def _verified_channel(mode: str, root: Path, record: dict, anchor: int,
                      encoder_sha256: str) -> tuple[Path, np.ndarray]:
    """Verify a restored mode's completed unit before loading its selected Q."""
    if mode == "U":
        complete = json.loads((root / "COMPLETE.json").read_text())
        if (complete.get("schema") != "pcrl-adaptive-A-center-v1"
                or complete.get("status") != "COMPLETE"
                or complete.get("anchor") != anchor or complete.get("delta") != .001
                or complete.get("selected_round") != record.get("selected_round")
                or complete.get("encoder_sha256") != encoder_sha256
                or complete.get("artifact_sha256") != fit_a._inventory(root)):
            raise ValueError("restored A center completion or encoder inventory differs")
        relative = f"round_r{record['selected_round']:02d}/channel/Q.npz"
    elif mode == "P":
        privacy_first_fit.load_release_spec(root)  # complete inventory and T0 content replay
        relative = "channel/Q.npz"
    else:
        complete = json.loads((root / "COMPLETE.json").read_text())
        if (complete.get("status") != "COMPLETE"
                or complete.get("artifact_sha256") != fit_controls._inventory(root)):
            raise ValueError("restored D17 control inventory differs")
        controls_record = json.loads((root / "CONTROLS.json").read_text())
        d17 = controls_record.get("controls", {}).get("D17", {})
        relative = record.get("D17_channel_relative_path")
        if (relative != "channels/D17/Q.npz" or d17.get("relative_path") != relative
                or d17.get("Q_file_sha256") != record.get("D17_channel_file_sha256")):
            raise ValueError("completed D17 control selection differs")
    channel = _member(root, relative)
    expected = (record.get("D17_channel_file_sha256") if mode == "D17"
                else record.get("channel_file_sha256"))
    if release.sha256_file(channel) != expected:
        raise ValueError("selected channel SHA-256 differs from trained model manifest")
    with np.load(channel, allow_pickle=False) as archive:
        if list(archive.files) != ["Q"]:
            raise ValueError("selected channel archive must contain only Q")
        q = archive["Q"].copy()
    if q.shape != (32, 17):
        raise ValueError("locked T32 channel must have exactly 32 states and 17 tokens")
    return channel, q


def emit(model_manifest_path: str | Path, expected_manifest_sha256: str,
         selection_lock_path: str | Path, expected_lock_sha256: str,
         anchor: int, mode: str, a_center_dir: str | Path,
         artifact_dir: str | Path, encoder_path: str | Path,
         input_npz: str | Path, replay_key_path: str | Path,
         output_npz: str | Path) -> dict:
    """Hash-replay a restored locked U/P/D17 T32 channel and emit one keyed token."""
    if anchor not in (0, 1, 2):
        raise ValueError("registered anchor required")
    if mode not in ("U", "P", "D17"):
        raise ValueError("mode must be U, P or D17")
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        raise RuntimeError("frozen T0 release requires verified Linux x86 runtime")
    manifest_file = evaluate._pin(model_manifest_path, expected_manifest_sha256)
    manifest = json.loads(manifest_file.read_text())
    lock_file = evaluate._pin(selection_lock_path, expected_lock_sha256)
    lock = json.loads(lock_file.read_text())
    anchors = manifest.get("anchors", {})
    if (manifest.get("schema") != "pcrl-adaptive-trained-model-manifest-v1"
            or manifest.get("status") != "EXPERIMENTAL_NO_ADVANTAGE"
            or manifest.get("assessment_year") != 2018
            or manifest.get("selection_lock_sha256") != expected_lock_sha256
            or lock.get("schema") != "pcrl-adaptive-selection-lock-v1"
            or lock.get("status") != "LOCKED" or lock.get("assessment_year") != 2018
            or str(anchor) not in anchors):
        raise ValueError("pinned trained manifest or selection lock differs")
    records = anchors[str(anchor)]
    bundle_name = {"U": "A_center_d001", "P": "privacy_first_d001", "D17": "A_controls_d001"}[mode]
    record = records[bundle_name]
    if (record.get("readback_sha256_verified") is not True
            or record.get("restored_inventory_verified") is not True):
        raise ValueError("selected model archive lacks verified readback/restore")
    a_root, root = _private(a_center_dir), _private(artifact_dir)
    encoder_file = _private(encoder_path)
    input_file, key_file, output = map(_private, (input_npz, replay_key_path, output_npz))
    if (output.exists() or output.with_suffix(output.suffix + ".manifest.json").exists()):
        raise FileExistsError("one-release wire output or receipt already exists")
    encoder_sha = records.get("encoder_sha256")
    a_complete = json.loads((a_root / "COMPLETE.json").read_text())
    if (a_complete.get("schema") != "pcrl-adaptive-A-center-v1"
            or a_complete.get("status") != "COMPLETE"
            or a_complete.get("anchor") != anchor or a_complete.get("delta") != .001
            or a_complete.get("encoder_sha256") != encoder_sha
            or a_complete.get("artifact_sha256") != fit_a._inventory(a_root)):
        raise ValueError("frozen A center or encoder SHA differs")
    channel, q = _verified_channel(mode, root, record, anchor, encoder_sha)
    logical = {"U": "U_candidate", "P": "P_candidate", "D17": "D17"}[mode]
    canonical = lock["anchors"][str(anchor)]["logical_to_canonical"].get(logical)
    if canonical != {"U": "A_selected", "P": "PrivacyFirst_selected", "D17": "A_control_D17"}[mode]:
        raise ValueError("locked logical candidate resolves to a different archived model")
    source = lock["anchors"][str(anchor)]["releases"].get(canonical, {})
    if (source.get("router_kind") != "T0"
            or source.get("channel_artifact", {}).get("sha256") != release.sha256_file(channel)
            or source.get("channel_array_sha256") != evaluate._array_sha(q)):
        raise ValueError("locked T0 channel file or array differs")
    if not key_file.is_file() or key_file.is_symlink() or stat.S_IMODE(key_file.stat().st_mode) & 0o077:
        raise PermissionError("private replay key must be an owner-only regular file")
    key = key_file.read_bytes()
    if len(key) < 32:
        raise ValueError("private replay key must contain at least 32 bytes")
    inputs, person_ids, original_h = _input(input_file)
    status = "BASELINE_CONTROL" if mode == "D17" else "EXPERIMENTAL_NO_ADVANTAGE"
    artifact = release.ChannelArtifact(q, encoder_sha, canonical, status)
    session = release.OneReleaseSession.from_verified_encoder_path(
        artifact, encoder_file, replay_key=key)
    wire = session.emit(inputs, person_ids)
    if (set(wire) != {"h_a", "token"}
            or wire["h_a"].dtype != original_h.dtype
            or wire["h_a"].tobytes() != original_h.tobytes()
            or wire["token"].shape != (len(original_h),)
            or not np.issubdtype(wire["token"].dtype, np.integer)
            or np.any((wire["token"] < 0) | (wire["token"] >= 17))):
        raise AssertionError("one-token wire or byte-preserved H_A contract failed")
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    with temporary.open("xb") as stream:
        np.savez_compressed(stream, h_a=wire["h_a"], token=wire["token"])
    os.chmod(temporary, 0o600)
    with np.load(temporary, allow_pickle=False) as check:
        if (set(check.files) != {"h_a", "token"}
                or check["h_a"].dtype != original_h.dtype
                or check["h_a"].tobytes() != original_h.tobytes()):
            raise AssertionError("stored wire differs from release contract")
    os.replace(temporary, output)
    receipt = {"schema": "pcrl-adaptive-one-token-wire-v1",
               "created_utc": datetime.now(timezone.utc).isoformat(),
               "assessment_year": 2018, "development_only": True,
               "anchor": anchor, "mode": mode, "canonical_release_id": canonical,
               "model_status": status,
               "model_manifest_sha256": expected_manifest_sha256,
               "selection_lock_sha256": expected_lock_sha256,
               "channel_file_sha256": release.sha256_file(channel),
               "channel_array_sha256": evaluate._array_sha(q),
               "encoder_sha256": encoder_sha,
               "wire_sha256": release.sha256_file(output),
               "wire_fields": ["h_a", "token"],
               "service_byte_parity": True,
               "original_people": len(person_ids),
               "private_key_recorded": False,
               "one_release_scope": "stable keyed draw for immutable ID/input/channel; no cross-release composition claim"}
    receipt_path = output.with_suffix(output.suffix + ".manifest.json")
    with receipt_path.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.chmod(receipt_path, 0o600)
    return {"wire_path": str(output), "receipt_path": str(receipt_path),
            "wire_fields": receipt["wire_fields"],
            "model_status": receipt["model_status"],
            "service_byte_parity": True,
            "original_people": len(person_ids)}


def reproduce_aggregates(selection_lock_path: str | Path, lock_sha256: str,
                         inner_selection_path: str | Path, inner_sha256: str,
                         inference_path: str | Path, inference_sha256: str,
                         outer_reports, output_csv: str | Path,
                         output_table: str | Path, output_plot: str | Path) -> dict:
    """Delegate only to the tested pinned aggregate renderer; no private rows."""
    return render_results.render_results(
        selection_lock_path, lock_sha256, inner_selection_path, inner_sha256,
        inference_path, inference_sha256, outer_reports,
        output_csv, output_table, output_plot)


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    wire = sub.add_parser("emit")
    wire.add_argument("--model-manifest", required=True)
    wire.add_argument("--manifest-sha256", required=True)
    wire.add_argument("--anchor", type=int, choices=(0, 1, 2), required=True)
    wire.add_argument("--selection-lock", required=True)
    wire.add_argument("--lock-sha256", required=True)
    wire.add_argument("--mode", choices=("U", "P", "D17"), required=True)
    wire.add_argument("--a-center-dir", required=True)
    wire.add_argument("--artifact-dir", required=True)
    wire.add_argument("--encoder", required=True)
    wire.add_argument("--input-npz", required=True)
    wire.add_argument("--replay-key", required=True)
    wire.add_argument("--wire-out", required=True)
    aggregate = sub.add_parser("reproduce-aggregates")
    aggregate.add_argument("--selection-lock", required=True)
    aggregate.add_argument("--lock-sha256", required=True)
    aggregate.add_argument("--inner-selection", required=True)
    aggregate.add_argument("--inner-selection-sha256", required=True)
    aggregate.add_argument("--inference", required=True)
    aggregate.add_argument("--inference-sha256", required=True)
    aggregate.add_argument("--outer-report", action="append", default=[])
    aggregate.add_argument("--outer-sha256", action="append", default=[])
    aggregate.add_argument("--output-csv", required=True)
    aggregate.add_argument("--output-table", required=True)
    aggregate.add_argument("--output-plot", required=True)
    args = parser.parse_args(argv)
    if args.command == "emit":
        result = emit(args.model_manifest, args.manifest_sha256,
                      args.selection_lock, args.lock_sha256, args.anchor, args.mode,
                      args.a_center_dir, args.artifact_dir,
                      args.encoder, args.input_npz, args.replay_key, args.wire_out)
    else:
        reports = render_results._anchor_args(args.outer_report, "outer report")
        pins = render_results._anchor_args(args.outer_sha256, "outer SHA")
        if set(reports) != set(pins) or (reports and set(reports) != {0, 1, 2}):
            parser.error("provide all three outer report paths and SHA pins, or none")
        result = reproduce_aggregates(
            args.selection_lock, args.lock_sha256,
            args.inner_selection, args.inner_selection_sha256,
            args.inference, args.inference_sha256,
            {a: (reports[a], pins[a]) for a in (0, 1, 2)} if reports else None,
            args.output_csv, args.output_table, args.output_plot)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
