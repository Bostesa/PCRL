"""Recipient-local deployment of frozen Q and FINE-TASK ten-bit score banks.

The sole per-person output is hard public code bits and the probability and
decision determined by their public decoder table.  The frozen U task score
is used to route a code locally and is never exported.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from dpc.deploy import load_permitted_input, schema_names, schema_sha256
from hcal.deploy import load_decoder
from results.pcrl_latent_and_sampled_methods_v1.r01.frozen_u import FrozenUEncoder

from .bank import admit_policy_pair, route_scores
from .service import sha256_file


@dataclass(frozen=True)
class BankService:
    bank: str
    seed: int
    source_sha256: str
    schema_sha256: str
    table_sha256: str
    encoder: FrozenUEncoder
    heads: tuple
    policy_pair: object
    tables: tuple[np.ndarray, np.ndarray]

    def recipient_from_raw(self, purpose: int, raw: np.ndarray,
                           columns: tuple[str, ...]) -> dict[str, np.ndarray]:
        if purpose not in (0, 1) or tuple(columns) != self.encoder.columns:
            raise ValueError("recipient purpose or ordered predictor schema differs")
        latent = self.encoder.encode_one(purpose, raw, columns, "DEPLOYMENT")
        score = np.asarray(self.heads[purpose].predict_proba(
            latent.astype(np.float64)), dtype=np.float64)
        if (score.shape != (len(raw), (2, 6)[purpose])
                or not np.isfinite(score).all()
                or not np.allclose(score.sum(axis=1), 1, rtol=0, atol=1e-8)):
            raise ValueError("frozen U task score shape or probabilities differ")
        one = route_scores(self.policy_pair, purpose, score, 10,
                           self.tables[purpose] if self.bank == "W_TSTAR" else None)
        return {"code": one["code"], "prob": one["probabilities"],
                "decision": one["decision"]}


def load_bank_service(*, source_root: Path, qpc_root: Path, table_path: Path,
                      expected_table_sha256: str, expected_source_sha256: str,
                      expected_schema_sha256: str, ordered_names: tuple[str, ...],
                      bank: str, seed: int) -> BankService:
    if bank not in ("W_Q", "W_TSTAR") or seed not in (0, 1):
        raise ValueError("only admitted bank and paired seed identities allowed")
    if (schema_sha256(ordered_names) != expected_schema_sha256
            or sha256_file(table_path) != expected_table_sha256):
        raise ValueError("ordered schema or public table byte pin differs")
    records = json.loads(Path(table_path).read_text())
    if (records.get("schema") != "code-utility-public-bank-tables-v1"
            or records.get("seed") != seed or records.get("bit_count") != 10
            or records.get("bit_order") != "c0_least_significant"
            or records.get("source_model_sha256") != expected_source_sha256
            or records.get("feature_schema_sha256") != expected_schema_sha256):
        raise ValueError("public bank table/source/schema identity differs")
    item = records["families"][bank]
    source = Path(source_root) / "admitted" / f"rel__s{seed}__U"
    complete = json.loads((source / "COMPLETE.json").read_text())
    if complete.get("id") != f"rel__s{seed}__U":
        raise ValueError("frozen U source unit identity differs")
    for name in ("model.pt", "head_0.joblib", "head_1.joblib", "record.json"):
        if sha256_file(source / name) != complete.get("files", {}).get(name):
            raise ValueError("frozen U source component hash differs")
    if complete["files"]["model.pt"] != expected_source_sha256:
        raise ValueError("frozen U checkpoint differs from table binding")
    encoder = FrozenUEncoder.load(source / "model.pt", expected_source_sha256,
                                  ordered_names, expected_schema_sha256, seed)
    heads = tuple(joblib.load(source / f"head_{i}.joblib") for i in (0, 1))
    if any(not np.array_equal(heads[i].classes_, np.arange((2, 6)[i]))
           for i in (0, 1)):
        raise ValueError("frozen U task class order differs")
    policy_path = (Path(qpc_root) / "run" / "units" /
                   f"pol__s{seed}__U_{'DIRECT-TASK' if bank == 'W_Q' else 'FINE-TASK'}_i8o64" /
                   "policy.json")
    pair, admission = admit_policy_pair(
        policy_path, expected_sha256=item["policy_sha256"],
        source_sha256=expected_source_sha256,
        schema_sha256=expected_schema_sha256)
    if admission["policy_fingerprint"] != item["policy_fingerprint"]:
        raise ValueError("public bank policy fingerprint differs")
    if bank == "W_TSTAR":
        decoder_path = (Path(source_root) / "package" /
                        f"decoder__s{seed}__U_FINE-TASK_i8o64_H-GLOBAL-TEMP.json")
        family, tables, embedded, _ = load_decoder(
            decoder_path, pair, item["decoder_sha256"])
        if family != "H-GLOBAL-TEMP" or embedded != item["decoder_sha256"]:
            raise ValueError("held-out decoder identity differs")
    else:
        if item["decoder_sha256"] is not None:
            raise ValueError("Q bank unexpectedly has a fitted decoder")
        tables = (pair.p1.token_proto, pair.p2.token_proto)
    for purpose in (0, 1):
        if (not np.array_equal(tables[purpose],
                               np.asarray(item[f"table{purpose}"], dtype=np.float64))
                or not np.array_equal((pair.p1, pair.p2)[purpose].token_class,
                                      np.asarray(item[f"token_class{purpose}"],
                                                 dtype=np.int64))):
            raise ValueError("public decoder table or token class differs")
    return BankService(bank, seed, expected_source_sha256,
                       expected_schema_sha256, expected_table_sha256,
                       encoder, heads, pair, tables)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m code_witness.bank_deploy",
                                     allow_abbrev=False)
    parser.add_argument("--bank", choices=("W_Q", "W_TSTAR"), required=True)
    parser.add_argument("--purpose", type=int, choices=(0, 1), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1), required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--qpc-root", required=True)
    parser.add_argument("--public-table", required=True)
    parser.add_argument("--public-table-sha256", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--schema-sha256", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    names = tuple(schema_names(args.schema))
    service = load_bank_service(
        source_root=Path(args.source_root), qpc_root=Path(args.qpc_root),
        table_path=Path(args.public_table),
        expected_table_sha256=args.public_table_sha256,
        expected_source_sha256=args.source_sha256,
        expected_schema_sha256=args.schema_sha256,
        ordered_names=names, bank=args.bank, seed=args.seed)
    raw = load_permitted_input(args.input, list(names))
    wire = service.recipient_from_raw(args.purpose, raw, names)
    output = Path(args.output)
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        np.savez_compressed(stream, **wire)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"schema": "code-utility-public-bank-deployment-v1",
                      "bank": args.bank, "purpose": args.purpose,
                      "seed": args.seed, "rows": len(raw),
                      "source_schema_sha256": args.schema_sha256,
                      "public_table_sha256": args.public_table_sha256,
                      "release_sha256": sha256_file(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
