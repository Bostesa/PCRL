"""Tiny data-free checks for the queued full-grid Q timing worker."""

from __future__ import annotations

import argparse
from unittest import mock

import numpy as np
import pytest

from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.attacks import smoke_q_preprocess as worker


def test_no_active_slot_refuses_before_public_grid_work(tmp_path):
    args = argparse.Namespace(unit_id=worker.UNIT_ID,
                              original_lock=tmp_path / "LOCK.json",
                              amendment=tmp_path / "A.json",
                              amendment_sha256="a" * 64,
                              private_root=tmp_path / ".private" / "q")
    with mock.patch.object(worker, "assert_active_from_env",
                           side_effect=RuntimeError("active queue slot environment missing")), \
         mock.patch.object(worker.smoke_attack, "_scores",
                           side_effect=AssertionError("grid created before gate")):
        with pytest.raises(RuntimeError, match="active queue slot"):
            worker.run(args)


def test_selected_and_fixed_laws_preserve_public_decisions():
    p1, c1 = worker.smoke_attack._scores(4, 2, 4)
    p2, c2 = worker.smoke_attack._scores(12, 6, 12)
    banks = (worker.smoke_attack._bank(1, p1),
             worker.smoke_attack._bank(2, p2))
    for fixed in (False, True):
        law = worker._law(*banks, fixed=fixed)
        for i, left in enumerate(c1):
            for j, right in enumerate(c2):
                triples = law(left, right, i, j)
                assert len(triples) == (1 if fixed else 16)
                assert sum(mass for _, _, mass in triples) == pytest.approx(1.0)
                for a, b, mass in triples:
                    assert mass > 0
                    assert np.argmax(banks[0].probability[a]) == np.argmax(left)
                    assert np.argmax(banks[1].probability[b]) == np.argmax(right)


def test_exact_two_case_output_contract():
    assert worker.OUTPUT_NAMES == [
        "SMOKE_RECEIPT.json", "selected/Q_TABLE_RECEIPT.json",
        "selected/q_table.npz", "fixed/Q_TABLE_RECEIPT.json",
        "fixed/q_table.npz"]
