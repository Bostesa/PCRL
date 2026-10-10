"""Deploy a fitted hard-code service on a pinned, permitted 83-column matrix.

The exported per-person wire contains only each recipient's hard bits, the
probabilities reconstructed from those bits, and their predicted class.  The
source latent, teacher score, scaler state, and labels are never exported.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from dpc.deploy import load_permitted_input, schema_names, schema_sha256
from results.pcrl_latent_and_sampled_methods_v1.r01.frozen_u import FrozenUEncoder

from .service import load_service, sha256_file


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m code_witness.deploy",
                                     allow_abbrev=False)
    parser.add_argument("--service-manifest", required=True)
    parser.add_argument("--service-manifest-sha256", required=True)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--source-model-sha256", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--schema-sha256", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bits", type=int, choices=(10, 16), required=True)
    parser.add_argument("--purpose", type=int, choices=(0, 1), required=True)
    args = parser.parse_args(argv)

    manifest_path = Path(args.service_manifest)
    if sha256_file(manifest_path) != args.service_manifest_sha256:
        raise ValueError("service manifest pin differs")
    manifest = json.loads(manifest_path.read_text())
    names = tuple(schema_names(args.schema))
    actual_schema_sha = schema_sha256(names)
    if (actual_schema_sha != args.schema_sha256
            or actual_schema_sha != manifest.get("source_schema_sha256")):
        raise ValueError("source feature schema pin differs")
    release = load_service(manifest_path,
                           expected_manifest_sha256=args.service_manifest_sha256,
                           expected_source_sha256=args.source_model_sha256,
                           expected_bits=args.bits)
    encoder = FrozenUEncoder.load(args.source_model, args.source_model_sha256,
                                  names, actual_schema_sha,
                                  int(manifest["source_seed"]))
    raw = load_permitted_input(args.input, list(names))
    one = release.recipient_from_raw(args.purpose, raw, names, encoder)
    exported = {"code": one["code"], "prob": one["probabilities"],
                "decision": one["decision"]}
    output = Path(args.output)
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        np.savez_compressed(stream, **exported)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"schema": "code-utility-hardbits-deployment-v1",
                      "rows": len(raw), "bits": args.bits, "purpose": args.purpose,
                      "source_schema_sha256": actual_schema_sha,
                      "service_manifest_sha256": args.service_manifest_sha256,
                      "release_sha256": sha256_file(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
