"""Public decoder-table service for compatible inherited W_Q/W_TSTAR attacks.

Only the public table and ten transmitted bits determine a recipient output.
The score router and source probabilities are provider-private and never enter
the attack view. This module does not read any role or label file.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class BankService:
    arm_id: str
    seed: int
    source_model_sha256: str
    public_table_sha256: str
    policy_sha256: str
    decoder_sha256: str | None
    tables: tuple[np.ndarray, np.ndarray]

    @property
    def model(self):
        # The generic archive verifier requests an empty synthetic input only
        # to recover fixed metadata. No bank latent or source input is exposed.
        return SimpleNamespace(latent_dim=16)

    def probabilities_from_code(self, purpose: int, code: np.ndarray) -> np.ndarray:
        if purpose not in (0, 1):
            raise ValueError("bank purpose invalid")
        bits = np.asarray(code)
        if (bits.dtype != np.uint8 or bits.ndim != 2 or bits.shape[1] != 10
                or not np.isin(bits, (0, 1)).all()):
            raise ValueError("bank public hard code invalid")
        ids = np.sum(bits.astype(np.int64) *
                     (1 << np.arange(10, dtype=np.int64))[None, :], axis=1)
        table = self.tables[purpose]
        if np.any(ids >= len(table)):
            raise ValueError("public bank code outside decoder table")
        return table[ids].copy()

    def recipient(self, purpose: int, latent: np.ndarray) -> dict:
        if purpose not in (0, 1) or np.asarray(latent).shape != (0, 16):
            raise ValueError("bank metadata request must be empty")
        code = np.zeros((0, 10), dtype=np.uint8)
        probability = self.probabilities_from_code(purpose, code)
        metadata = {
            "schema": "code-utility-bank-public-wire-v1", "purpose": purpose,
            "bits": 10, "bit_order": "c0_least_significant",
            "version": f"{self.arm_id}_s{self.seed}_compiled10_v1",
            "source_sha256": self.source_model_sha256,
            "public_table_sha256": self.public_table_sha256,
            "policy_sha256": self.policy_sha256,
            "decoder_sha256": self.decoder_sha256,
        }
        return {"code": code, "probabilities": probability,
                "decision": np.empty(0, dtype=np.int64), "metadata": metadata}


def make_bank_service(table_doc: Mapping, *, arm_id: str, seed: int,
                      source_model_sha256: str, feature_schema_sha256: str,
                      public_table_sha256: str, policy_sha256: str,
                      decoder_sha256: str | None) -> BankService:
    """Validate public table/source/policy identity before any release read."""
    if (arm_id not in ("W_Q", "W_TSTAR") or seed not in (0, 1)
            or table_doc.get("schema") != "code-utility-public-bank-tables-v1"
            or table_doc.get("seed") != seed or table_doc.get("bit_count") != 10
            or table_doc.get("source_model_sha256") != source_model_sha256
            or table_doc.get("feature_schema_sha256") != feature_schema_sha256
            or any(not isinstance(x, str) or len(x) != 64 for x in (
                source_model_sha256, feature_schema_sha256, public_table_sha256,
                policy_sha256))):
        raise ValueError("public bank table/source/seed identity differs")
    family = table_doc.get("families", {}).get(arm_id)
    if (not isinstance(family, Mapping)
            or family.get("policy_sha256") != policy_sha256
            or family.get("decoder_sha256") != decoder_sha256
            or (arm_id == "W_Q" and decoder_sha256 is not None)
            or (arm_id == "W_TSTAR" and
                (not isinstance(decoder_sha256, str) or len(decoder_sha256) != 64))):
        raise ValueError("public bank policy/decoder identity differs")
    tables = []
    for purpose, classes in ((0, 2), (1, 6)):
        table = np.asarray(family[f"table{purpose}"], dtype=np.float64)
        labels = np.asarray(family[f"token_class{purpose}"])
        if (table.ndim != 2 or table.shape[1] != classes
                or not 1 <= len(table) <= 1024
                or labels.shape != (len(table),) or labels.dtype.kind not in "iu"
                or not np.isfinite(table).all() or np.any(table < 0)
                or not np.allclose(table.sum(axis=1), 1.0, rtol=0, atol=1e-10)
                or not np.array_equal(table.argmax(axis=1), labels)):
            raise ValueError("public bank decoder table/class/capacity differs")
        tables.append(table)
    return BankService(arm_id, seed, source_model_sha256, public_table_sha256,
                       policy_sha256, decoder_sha256, tuple(tables))
