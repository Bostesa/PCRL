"""Small data-free checks for the queued role-sized timing worker."""

from __future__ import annotations

import numpy as np

from . import smoke_attack, wire


def test_synthetic_bank_and_wires_match_complete_public_contract():
    p1, c1 = smoke_attack._scores(8, 2, 16)
    p2, c2 = smoke_attack._scores(12, 6, 321)
    assert p1.shape == (16, 2) and p2.shape == (321, 6)
    assert c1.shape == (8, 2) and c2.shape == (12, 6)
    banks = (smoke_attack._bank(1, p1), smoke_attack._bank(2, p2))
    package, labels = smoke_attack._role(
        "ATTACK_FIT_NEW", 100, 12, banks, np.random.default_rng(3))
    assert set(labels) == {"row_id", "unit", "sex"}
    assert len(package.wires1) == len(package.wires2) == 12
    assert set(package.wires1[0]) == wire.WIRE_KEYS
    left = wire.materialize_recipient_batch(
        package.wires1, package.row_id, package.unit, banks[0])
    right = wire.materialize_recipient_batch(
        package.wires2, package.row_id, package.unit, banks[1])
    assert left.fields["probability"].values.shape == (12, 2)
    assert right.fields["probability"].values.shape == (12, 6)


def test_role_sized_plan_counts_all_three_q_readers_and_v2_controls():
    plan = smoke_attack.protocol.reader_plan(family="A", code_bits=True)
    counts = smoke_attack.protocol.unit_fit_forecast(plan)
    assert counts == {
        "core_fits": 18, "probability_only_fits": 6,
        "v2_control_fits": 6, "ignore_recipient_fits": 0,
        "mandatory_fits": 30, "views": ("r1", "r2", "pair"),
        "fit_role": "ATTACK_FIT_NEW/ATTACK_TRAIN",
        "tune_role": "ATTACK_FIT_NEW/ATTACK_TUNE",
        "score_role": "INNER_SELECTION",
    }
    assert counts["mandatory_fits"] + 3 == smoke_attack.EXPECTED_FITS
