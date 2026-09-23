"""Test and operate the one-token release API without exposing hidden rows.

``synthetic`` is a public toy demonstration. ``emit`` consumes only private
PCA32 X_A, four H_A coordinates and caller-held IDs; its wire archive contains
only byte-preserved H_A and one token. A private keyed replay key makes the
same immutable person/channel produce one stable computational draw.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import stat

import joblib
import numpy as np

from experiments.pcrl_task_directed_release_v1.data import RuntimeInputs
from . import method, release
from .synthetic_fixture import SyntheticEncoder


def _unused_file(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    return path


def _private_path(path):
    path = Path(path)
    if "private" not in path.parts:
        raise ValueError("person inputs, wire output and replay key require private paths")
    return path


def generate_key(path):
    path = _unused_file(_private_path(path))
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(os.urandom(32))
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return {"key_file": str(path), "mode": "0600", "key_bytes": 32,
            "note": "retain privately; never include in the release artifact or wire"}


def emit(channel_dir, encoder_path, input_npz, key_file, output_npz):
    input_path = _private_path(input_npz)
    key_path = _private_path(key_file)
    output_path = _unused_file(_private_path(output_npz))
    if stat.S_IMODE(key_path.stat().st_mode) & 0o077:
        raise PermissionError("replay key must be readable only by its owner")
    artifact = release.ChannelArtifact.load(channel_dir)
    with np.load(input_path, allow_pickle=False) as archive:
        if set(archive.files) != {"x_a", "h_a", "release_ids"}:
            raise ValueError("private input archive must contain only x_a, h_a, release_ids")
        x = archive["x_a"].copy()
        h = archive["h_a"].copy()
        ids = archive["release_ids"].copy()
    if ids.ndim != 1 or ids.dtype.kind not in ("i", "u", "U"):
        raise ValueError("release IDs must be a one-dimensional string or integer array")
    inputs = RuntimeInputs(x, h)
    session = release.OneReleaseSession.from_verified_encoder_path(
        artifact, encoder_path, replay_key=key_path.read_bytes())
    wire = session.emit(inputs, ids.tolist())
    np.savez_compressed(output_path, h_a=wire["h_a"], token=wire["token"])
    with np.load(output_path, allow_pickle=False) as check:
        if set(check.files) != {"h_a", "token"} or not np.array_equal(check["h_a"], h):
            raise AssertionError("wire archive changed H_A or includes hidden fields")
        if check["h_a"].dtype != h.dtype or check["h_a"].tobytes() != h.tobytes():
            raise AssertionError("H_A is not byte-identical")
    receipt = {
        "schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "wire_file": output_path.name, "wire_sha256": release.sha256_file(output_path),
        "channel_sha256": release.sha256_file(Path(channel_dir) / "Q.npz"),
        "encoder_sha256": artifact.encoder_sha256,
        "original_people": len(ids),
        "wire_fields": ["h_a", "token"],
        "service_byte_parity": True,
        "one_release_policy": "private keyed replay; immutable input IDs and channel required",
        "key_recorded": False,
    }
    receipt_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return {"output": str(output_path), "manifest": str(receipt_path),
            "original_people": len(ids), "service_byte_parity": True}


def synthetic(output_dir):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    encoder = SyntheticEncoder()
    encoder_path = root / "synthetic_encoder.joblib"
    joblib.dump(encoder, encoder_path)
    q = np.full((32, 17), .2/17, dtype=np.float64)
    q[np.arange(32), np.arange(32) % 17] += .8
    artifact = release.ChannelArtifact(q, release.sha256_file(encoder_path),
                                       "public_synthetic_fixture", "SYNTHETIC_ONLY")
    artifact.save(root / "channel")
    x = np.zeros((3, 32), dtype=np.float32)
    x[:, 0] = [.1, 1.7, -.4]
    h = np.asarray([[.1, .2, .3, .4], [1.1, -.2, .7, 2.0],
                    [-.5, 1.0, .2, -.1]], dtype=np.float32)
    ids = np.asarray(["synthetic-0", "synthetic-1", "synthetic-2"])
    inputs = RuntimeInputs(x, h)
    public_fixture_key = bytes(range(32))  # a fixture constant, never a deployment secret
    session = release.OneReleaseSession.from_verified_encoder_path(
        artifact, encoder_path, replay_key=public_fixture_key)
    first = session.emit(inputs, ids.tolist())
    again = release.OneReleaseSession.from_verified_encoder_path(
        release.ChannelArtifact.load(root / "channel"), encoder_path,
        replay_key=public_fixture_key).emit(inputs, ids.tolist())
    if not np.array_equal(first["token"], again["token"]):
        raise AssertionError("synthetic keyed replay differs across fresh sessions")
    if first["h_a"].tobytes() != h.tobytes():
        raise AssertionError("synthetic service parity failed")
    np.savez_compressed(root / "synthetic_inputs.npz", x_a=x, h_a=h, release_ids=ids)
    np.savez_compressed(root / "synthetic_wire.npz", h_a=first["h_a"], token=first["token"])
    codes = encoder.encode(inputs)["codes"]["T0"]
    p1 = np.clip(.2 + .03*np.arange(17)[None, :] + .03*h[:, :1], .01, .99)
    predictions = method.binary_predictions(p1)
    expected = method.score_expected(q, codes, np.asarray([0, 1, 1]),
                                     predictions, np.asarray([1., 2., 1.]))
    report = {
        "schema": 1, "kind": "public_synthetic_only", "original_people": 3,
        "channel_sha256": release.sha256_file(root / "channel/Q.npz"),
        "encoder_sha256": release.sha256_file(encoder_path),
        "wire_sha256": release.sha256_file(root / "synthetic_wire.npz"),
        "wire_fields": ["h_a", "token"], "service_byte_parity": True,
        "keyed_replay_across_fresh_sessions": True,
        "exact_expected_task_loss": expected,
        "note": "Toy encoder and public fixture key are not the ACS mechanism or a production secret",
    }
    (root / "DEMO.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("synthetic")
    demo.add_argument("--output-dir", required=True)
    key = sub.add_parser("key-init")
    key.add_argument("--output", required=True)
    wire = sub.add_parser("emit")
    wire.add_argument("--channel-dir", required=True)
    wire.add_argument("--encoder", required=True)
    wire.add_argument("--input-npz", required=True)
    wire.add_argument("--key-file", required=True)
    wire.add_argument("--output-npz", required=True)
    args = parser.parse_args(argv)
    if args.command == "synthetic":
        result = synthetic(args.output_dir)
    elif args.command == "key-init":
        result = generate_key(args.output)
    else:
        result = emit(args.channel_dir, args.encoder, args.input_npz,
                      args.key_file, args.output_npz)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
