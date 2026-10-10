"""Label-free exact P0-bank public wire encoding for final fresh attacks.

Global IDs are public categorical IDs, never refitted or renumbered. The
inherited reference codec preserves every public metadata field. Private row
and duplicate IDs serve only alignment and cached-release checks.
"""
from dataclasses import replace
import numpy as np

from results.pcrl_latent_and_sampled_methods_v1.r01.role_store import sha_array
from results.pcrl_latent_and_sampled_methods_v1.r02.complete_view import pair_batch, prepare_views
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.reference_materialize import reconstruct_reference_wires
from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.attacks.reference_wire import ReferenceRoleRelease, reference_recipient_batch
from .r4_public_law import row_log_mass

ARRAY_KEYS = frozenset({"row_id", "unit", "code_r1", "code_r2", "prob_r1", "prob_r2", "decision_r1", "decision_r2"})


def package_from_arrays(arrays, *, identity, role):
    """Pure wire reconstruction; caller must verify source/custody receipts.

    identity contains the actually emitted release_id/source binding and two
    full public probability banks. It is not a scientific arm rename.
    """
    if set(arrays) != ARRAY_KEYS or role not in ("ATTACK_FIT_NEW", "INNER_SELECTION"):
        raise ValueError("exact admitted role release array set required")
    row, unit = arrays["row_id"], arrays["unit"]
    if (row.dtype != np.int64 or row.ndim != 1 or not len(row) or
            unit.dtype != np.int64 or unit.shape != row.shape or
            len(np.unique(row)) != len(row)):
        raise ValueError("row/group alignment invalid")
    wires, bank_hashes = {}, {}
    for purpose, width in ((1, 2), (2, 6)):
        key = f"r{purpose}"
        bank = identity[f"bank_{key}"]
        codes, prob, decision = (arrays[f"code_{key}"], arrays[f"prob_{key}"], arrays[f"decision_{key}"])
        if (bank.dtype != np.float64 or bank.ndim != 2 or bank.shape[1] != width or
                codes.dtype != np.int64 or codes.shape != row.shape or
                np.any(codes < 0) or np.any(codes >= len(bank)) or
                prob.dtype != np.float64 or prob.shape != (len(row), width) or
                bank[codes].tobytes() != prob.tobytes()):
            raise ValueError("global ID/probability differs from P0 decoder")
        # One fixed release per duplicate group; no private row-ID variation.
        for group in np.unique(unit):
            selected = np.flatnonzero(unit == group)
            if not np.all(codes[selected] == codes[selected[0]]):
                raise ValueError("duplicate group has multiple cached releases")
        bank_hashes[key] = sha_array(bank)
        wires[key] = reconstruct_reference_wires(prob, decision, codes,
            recipient=purpose, interface_kind="finite", release_id=identity["release_id"],
            source_binding_sha256=identity["source_binding_sha256"],
            bank_sha256=bank_hashes[key], bank_size=len(bank))
    receipt = {"interface_kind": "finite", "role": role,
               "release_id": identity["release_id"],
               "source_binding_sha256": identity["source_binding_sha256"],
               "bank_sha256_by_recipient": bank_hashes}
    return ReferenceRoleRelease(receipt, "", row, unit, wires["r1"], wires["r2"])


def local_batch(package, recipient, masses):
    batch = reference_recipient_batch(package, recipient)
    probability = batch.fields["probability"].values
    code=np.asarray(batch.fields["code_id"].values,dtype=np.int64)
    return replace(batch, public_derived={
        "public_log_probability": np.log(np.maximum(probability, 1e-12)),
        "public_uniform_source_log_output_mass": row_log_mass(masses,f"r{recipient}",code)})


def coalition_batch(left, right, masses):
    batch = pair_batch(left, right)
    bits1 = np.asarray(left.fields["code_bits"].values, np.float64)
    bits2 = np.asarray(right.fields["code_bits"].values, np.float64)
    return replace(batch, public_derived={**batch.public_derived,
        "pair_code_bit_products": (bits1[:, :, None] * bits2[:, None, :]).reshape(
            len(bits1), bits1.shape[1] * bits2.shape[1]),
        "pair_public_uniform_source_log_output_mass": row_log_mass(masses,"pair",
            np.asarray(left.fields["code_id"].values,dtype=np.int64),
            np.asarray(right.fields["code_id"].values,dtype=np.int64))})


def prepare_three_views(att, inner, *, uniform_masses):
    for key in ("release_id", "source_binding_sha256", "bank_sha256_by_recipient"):
        if att.receipt[key] != inner.receipt[key]:
            raise ValueError("ATT/INNER public identities differ")
    if att.receipt["role"] != "ATTACK_FIT_NEW" or inner.receipt["role"] != "INNER_SELECTION":
        raise ValueError("ATT/INNER role identities differ")
    batches = {}
    for role, package in (("att", att), ("inner", inner)):
        left, right = local_batch(package, 1, uniform_masses), local_batch(package, 2, uniform_masses)
        batches[role] = {"r1": left, "r2": right, "pair": coalition_batch(left, right, uniform_masses)}
    prepared = {view: prepare_views(batches["att"][view], batches["inner"][view])
                for view in ("r1", "r2", "pair")}
    if any(getattr(p, side).x_defense_aware is None
           for p in prepared.values() for side in ("att", "inner")):
        raise ValueError("mandatory defense-aware view absent")
    return prepared
