"""Deploy a hash-bound incumbent-code channel from permitted raw83 input.

The private provider computes P0 codes; the policy sees only those codes and
their fixed public table values. Coin arrays are a separate private input.
Neither labels nor teacher scores are transmitted or passed to the policy.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from qpc import deploy as QD
from .r1_bank import sha_file
from .r1_channel import COIN_WIDTH, selected_wires_from_raw
from .r1_deploy import load_service
from .r2_package import load_policy


def private_uniforms(path, expected_sha256):
    if sha_file(path) != expected_sha256:
        raise ValueError("private uniform file differs")
    with ZipFile(path) as stream:
        if [member.filename for member in stream.infolist()] != ["uniform.npy"]:
            raise ValueError("uniform archive must have exactly one member")
    with np.load(path, allow_pickle=False) as stream:
        return np.array(stream["uniform"], copy=True)


def write_wires(path, wires):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        for wire in wires:
            stream.write(json.dumps(wire, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return sha_file(path)


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--policy-dir", required=True)
    parser.add_argument("--policy-receipt-sha256", required=True)
    parser.add_argument("--policy-lock-sha256", required=True)
    parser.add_argument("--arm-id", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--uniforms")
    parser.add_argument("--uniforms-sha256")
    parser.add_argument("--output-r1", required=True)
    parser.add_argument("--output-r2", required=True)
    args = parser.parse_args(argv)

    if args.output_r1 == args.output_r2:
        raise ValueError("recipient output paths must differ")
    if sha_file(args.input) != args.input_sha256:
        raise ValueError("raw83 input bytes differ")
    if Path(args.output_r1).exists() or Path(args.output_r2).exists():
        raise FileExistsError("write-once recipient output already exists")

    service = load_service(args.source_manifest, args.seed, args.source_manifest_sha256)
    receipt_path = Path(args.policy_dir) / "POLICY_RECEIPT.json"
    if sha_file(receipt_path) != args.policy_receipt_sha256:
        raise ValueError("selected policy receipt differs")
    receipt = json.loads(receipt_path.read_text())
    source_manifest = json.loads(Path(args.source_manifest).read_text())
    expected_bank_sha256 = source_manifest["seeds"][str(args.seed)]["public_bank"]["sha256"]
    if receipt["public_bank_sha256"] != expected_bank_sha256:
        raise ValueError("selected policy source or bank differs")
    if (receipt["source_binding_sha256"] != service.bank["source_binding_sha256"] or
            receipt["source_seed"] != args.seed):
        raise ValueError("selected policy source binding differs")
    if (receipt["source_lock_sha256"] != args.policy_lock_sha256 or
            receipt["arm_id"] != args.arm_id):
        raise ValueError("selected policy lock or arm differs")
    policy = load_policy(args.policy_dir, expected_receipt_sha256=args.policy_receipt_sha256)
    if policy.law_kind not in COIN_WIDTH:
        raise ValueError("selected policy has unregistered sampling contract")
    width = COIN_WIDTH[policy.law_kind]
    if width == 0:
        if args.uniforms is not None or args.uniforms_sha256 is not None:
            raise ValueError("identity/deterministic law cannot accept coins")
        uniforms = None
    else:
        if args.uniforms is None or args.uniforms_sha256 is None:
            raise ValueError("stochastic law requires pinned private coins")
        uniforms = private_uniforms(args.uniforms, args.uniforms_sha256)
    raw = QD.load_permitted_input(args.input, list(service.names))
    wires = selected_wires_from_raw(service, policy, raw, service.names, uniforms)
    digests = (write_wires(args.output_r1, wires[0]),
               write_wires(args.output_r2, wires[1]))
    print(json.dumps({"schema": "pcrl-incumbent-selected-raw-deployment-v1",
                      "seed": args.seed, "arm_id": receipt["arm_id"],
                      "policy_receipt_sha256": args.policy_receipt_sha256,
                      "rows": len(wires[0]), "draw_kind": policy.law_kind,
                      "release_r1_sha256": digests[0],
                      "release_r2_sha256": digests[1]}, sort_keys=True))


if __name__ == "__main__":
    main()
