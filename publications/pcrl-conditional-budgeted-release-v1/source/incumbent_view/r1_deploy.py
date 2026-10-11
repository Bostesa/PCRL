"""Exact frozen P0 raw83 router and original metadata-preserving JSONL wire."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from hcal.deploy import load_decoder
from qpc import deploy as QD
from qpc import release as RL
from .r1_bank import load_bank, sha_file, wires_for_codes


def load_pinned_teacher(paths, seed):
    """Reconstruct frozen dpc teacher without opening historical release.npz.

    The original COMPLETE sidecar includes an all-role person-level release.
    The admitted model/head/record hashes are checked against that sidecar and
    against the source manifest before this function is called. The weights
    and estimators are loaded with the same operations as dpc.deploy.
    """
    import joblib
    import torch
    from jcv.train import Model

    expected = {"model.pt": "model", "head_0.joblib": "head0",
                "head_1.joblib": "head1", "record.json": "record"}
    complete = json.loads(Path(paths["complete"]["path"]).read_text())
    listed = complete.get("files", {})
    if not isinstance(listed, dict):
        raise ValueError("frozen teacher COMPLETE sidecar differs")
    for filename, key in expected.items():
        item = paths[key]
        if Path(item["path"]).name != filename or listed.get(filename) != item["sha256"]:
            raise ValueError("frozen teacher hash/filename differs from source COMPLETE")
    record = json.loads(Path(paths["record"]["path"]).read_text())
    if int(record["seed"]) != int(seed):
        raise ValueError("frozen teacher seed differs")
    model = Model(83, [2, 6], int(seed))
    model.load_state_dict(torch.load(paths["model"]["path"]))
    model.eval()
    heads = [joblib.load(paths[key]["path"]) for key in ("head0", "head1")]
    return model, heads, paths["model"]["sha256"]


class P0Service:
    def __init__(self, paths, bank):
        self.bank = bank
        self.seed = bank["seed"]
        self.names = tuple(bank["feature_names"])
        self.schema_sha256 = bank["feature_schema_sha256"]
        self.pair = QD.load_pair(paths["policy"]["path"])
        self.model, self.heads, model_sha = load_pinned_teacher(paths, self.seed)
        QD.check_bindings(self.pair, model_sha, self.schema_sha256)
        family, tables, decoder_sha, d1 = load_decoder(paths["decoder"]["path"], self.pair,
                                                     bank["decoder_embedded_sha256"])
        if family != "H-CLASS-TEMP" or d1 is not None or decoder_sha != bank["decoder_embedded_sha256"]:
            raise ValueError("exact historical P0 decoder differs")
        self.tables = tables
        for i in (1, 2):
            p = np.asarray(bank["recipients"][str(i)]["probability"], dtype=np.float64)
            if not np.array_equal(tables[i-1], p):
                raise ValueError("P0 public table is not exact historical decoder")

    def route_raw(self, raw, names):
        if tuple(names) != self.names:
            raise ValueError("raw83 predictor order differs")
        x = np.asarray(raw)
        if x.ndim != 2 or x.shape[1] != 83 or not np.issubdtype(x.dtype, np.floating) or not np.isfinite(x).all():
            raise ValueError("finite raw83 matrix required")
        # Same input cast and frozen teacher/heads as historical hcal.deploy.
        scores = QD.teacher_probs(self.model, self.heads, x.astype(np.float32))
        outputs = []
        for i, policy in enumerate((self.pair.p1, self.pair.p2)):
            code, _, decision = RL.encode(policy, scores[i])
            probability = self.tables[i][code]
            if not np.array_equal(probability.argmax(1), decision):
                raise ValueError("P0 decoded decisions changed")
            outputs.append((code.astype(np.int64), probability, decision.astype(np.int64)))
        return outputs, scores


def load_service(source_paths, seed, expected_manifest_sha256):
    if sha_file(source_paths) != expected_manifest_sha256:
        raise ValueError("P0 source manifest differs")
    manifest = json.loads(Path(source_paths).read_text())
    record = manifest["seeds"][str(seed)]
    for item in record["pins"].values():
        if sha_file(item["path"]) != item["sha256"]:
            raise ValueError("P0 source bytes differ")
    bank = load_bank(record["public_bank"]["path"], record["public_bank"]["sha256"])
    return P0Service(record["pins"], bank)


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--recipient", type=int, choices=(1, 2), required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    service = load_service(args.source_manifest, args.seed, args.source_manifest_sha256)
    raw = QD.load_permitted_input(args.input, list(service.names))
    outputs, _ = service.route_raw(raw, service.names)
    wires = wires_for_codes(service.bank, args.recipient, outputs[args.recipient-1][0])
    fd = os.open(args.output, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        for wire in wires:
            stream.write(json.dumps(wire, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n")
    print(json.dumps({"schema":"pcrl-incumbent-raw-deployment-v1", "seed":args.seed,
                      "recipient":args.recipient,"rows":len(wires),
                      "release_sha256":sha_file(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
