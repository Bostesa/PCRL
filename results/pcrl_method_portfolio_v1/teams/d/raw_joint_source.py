"""Distinct, hash-admitted RAW-J frozen encoder adapter for a matched D control.

This only binds public source identity and the checkpoint's direct preprivacy
encoder computation. It does not itself admit RAW-J for Adult scoring: S3 must
queue a permitted role-only parity forward and S4 must independently approve
its architecture and replay before a science runner can enable this arm.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from encoder_adapter import FrozenUEncoder, hash_file


RAW_MODEL_SHA = {
    0: "56aaea46201de6be0a7b3ae59954f26986c33f47d8b4eb1c50b89429d196611d",
    1: "684fdfb735942de90391309b2d046e774d15b33ba56daced5a78d539da0c27e3",
    2: "5e803ef45adc61104921bd1ae05fdaf5ae86b1db3d6fee81b06b86a8bbdb90df",
}
SOURCE_KIND = "RAW-J_b0.3"


class FrozenRawJointEncoder:
    """Purpose-local direct `jcv.Model.encode(i,X)` from admitted RAW-J weights."""

    def __init__(self, inner: FrozenUEncoder, source_admission_sha256: str):
        if (inner.seed not in RAW_MODEL_SHA
                or inner.model_sha256 != RAW_MODEL_SHA[inner.seed]):
            raise ValueError("RAW-J wrapper requires a pinned RAW-J checkpoint")
        self._inner = inner
        self.seed = inner.seed
        self.model_sha256 = inner.model_sha256
        self.columns = inner.columns
        self.schema_sha256 = inner.schema_sha256
        self.source_admission_sha256 = source_admission_sha256
        self.source_kind = SOURCE_KIND

    @classmethod
    def load(cls, model_path: Path, source_admission_path: Path,
             expected_columns: tuple[str, ...], expected_schema_sha256: str,
             seed: int) -> "FrozenRawJointEncoder":
        if seed not in RAW_MODEL_SHA:
            raise PermissionError("unregistered RAW-J source seed")
        model_path = Path(model_path)
        if (model_path.name != "model.pt"
                or model_path.parent.name != f"rel__s{seed}__{SOURCE_KIND}"):
            raise PermissionError("RAW-J source package path differs from hcal admission")
        admitted = json.loads(Path(source_admission_path).read_text())
        package = admitted.get("admitted_teacher_dirs", {}).get(model_path.parent.name, {})
        if (admitted.get("schema") != "hcal-source-admission-v1"
                or admitted.get("verdict") != "ADMITTED"
                or admitted.get("source_tip") != "976202546a100f2b89a863cb3785c3465cf76c6a"
                or package.get("model.pt") != RAW_MODEL_SHA[seed]):
            raise PermissionError("hcal admission does not bind RAW-J checkpoint bytes")
        inner = FrozenUEncoder.load(model_path, RAW_MODEL_SHA[seed],
                                    expected_columns, expected_schema_sha256, seed)
        return cls(inner, hash_file(source_admission_path))

    def encode_one(self, purpose: int, x: np.ndarray,
                   columns: tuple[str, ...], role: str) -> np.ndarray:
        return self._inner.encode_one(purpose, x, columns, role)

    def encode_with_receipt(self, x: np.ndarray, columns: tuple[str, ...],
                            role: str) -> tuple[np.ndarray, np.ndarray, dict]:
        h0, h1, base = self._inner.encode_with_receipt(x, columns, role)
        receipt = dict(base)
        receipt.update({
            "schema": "team-d-frozen-rawjoint-encoder-parity-v1",
            "source_kind": SOURCE_KIND,
            "source_admission_sha256": self.source_admission_sha256,
            "base_adapter_parity_schema": base["schema"],
            "scope": "direct preprivacy Model.encode only; S4 Adult role admission is separate",
        })
        return h0, h1, receipt
