"""Provider-side frozen bank route and protected pair deployment.

The provider computes both task scores and fine cells internally. Recipients
receive only their emitted global bank code and its fixed decoded output.
"""

from __future__ import annotations

import numpy as np

from code_witness.bank_deploy import BankService
from dpc.partition import assign_fine

from .r1_release import release_pair_from_scores


def release_pair_from_raw(service: BankService, policy, raw: np.ndarray,
                          columns: tuple[str, ...],
                          uniforms: np.ndarray | None) -> tuple[dict, dict]:
    """Run an admitted policy on the actual frozen U -> W_TSTAR route."""
    if (not isinstance(service, BankService) or service.bank != "W_TSTAR" or
            tuple(columns) != service.encoder.columns):
        raise ValueError("selected bank or ordered 83-column schema differs")
    value = np.asarray(raw)
    if (value.ndim != 2 or value.shape[1] != 83 or
            not np.issubdtype(value.dtype, np.floating) or
            not np.isfinite(value).all()):
        raise ValueError("permitted preprocessed raw input differs")
    scores, cells = [], []
    for purpose in (0, 1):
        latent = service.encoder.encode_one(purpose, value, columns, "DEPLOYMENT")
        score = np.asarray(service.heads[purpose].predict_proba(
            latent.astype(np.float64)), dtype=np.float64)
        if score.shape != (len(value), (2, 6)[purpose]):
            raise ValueError("frozen task score class order or shape differs")
        policy_part = (service.policy_pair.p1, service.policy_pair.p2)[purpose]
        cell = assign_fine(score, score.argmax(axis=1).astype(np.int64),
                           policy_part.fine)
        scores.append(score)
        cells.append(cell.astype(np.int64))
    return release_pair_from_scores(policy, scores[0], scores[1],
                                    cells[0], cells[1], service.tables, uniforms)
