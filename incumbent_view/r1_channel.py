"""Raw-input selected-channel adapter preserving the actual incumbent wire.

The trained policy sees original P0 codes and their fixed decoded values.
Private raw predictors and frozen teacher scores stop at the P0 router.
"""
from __future__ import annotations

import numpy as np

from .r1_bank import wires_for_codes


COIN_WIDTH = {
    "incumbent_identity_v1": 0,
    "incumbent_deterministic_v1": 0,
    "incumbent_joint_v1": 1,
    "incumbent_product_v1": 2,
    "incumbent_local_v1": 2,
    "incumbent_seq12_v1": 2,
    "incumbent_seq21_v1": 2,
}


def selected_wires_from_raw(service, policy, raw, feature_names, uniforms=None):
    """Emit two original-schema JSON wire collections, one fixed draw/person."""
    kind = policy.law_kind
    if kind not in COIN_WIDTH:
        raise ValueError("unregistered incumbent channel sampling contract")
    for recipient in (1, 2):
        bank_item = service.bank["recipients"][str(recipient)]
        actual = np.asarray(bank_item["probability"], dtype=np.float64)
        if not np.array_equal(actual, getattr(policy.space, f"probability{recipient}")):
            raise ValueError("selected policy changes original ID decoder bank")
        actions = getattr(policy.space, f"actions{recipient}")
        valid = getattr(policy.space, f"valid{recipient}")
        if (actions.shape != (len(actual), 4) or valid.shape != actions.shape or
                any(tuple(actions[row, valid[row]]) != tuple(support)
                    for row, support in enumerate(bank_item["support"]))):
            raise ValueError("selected policy changes fixed original-ID support")
    source, _ = service.route_raw(raw, feature_names)
    n = len(source[0][0])
    width = COIN_WIDTH[kind]
    if width == 0:
        if uniforms is not None:
            raise ValueError("deterministic/P0 deployment must not accept random coins")
    else:
        u = np.asarray(uniforms)
        shape = (n,) if width == 1 else (n, width)
        if (u.dtype != np.float64 or u.shape != shape or not np.isfinite(u).all()
                or np.any(u < 0) or np.any(u >= 1)):
            raise ValueError("frozen private uniform shape/type/range differs")
        uniforms = u
    codes = policy.sample_codes(source[0][0], source[1][0], uniforms)
    if not isinstance(codes, tuple) or len(codes) != 2:
        raise ValueError("selected policy must emit exactly two public ID arrays")
    wires = []
    for recipient, code in enumerate(codes, 1):
        c = np.asarray(code)
        table = np.asarray(service.bank["recipients"][str(recipient)]["probability"], dtype=np.float64)
        if (c.dtype != np.int64 or c.shape != (n,) or np.any(c < 0) or np.any(c >= len(table))
                or not np.array_equal(table[c].argmax(1), source[recipient-1][2])):
            raise ValueError("channel ID output invalid or changes source decisions")
        wires.append(wires_for_codes(service.bank, recipient, c))
    return tuple(wires)
