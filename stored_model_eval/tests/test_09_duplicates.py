"""Test 9: duplicated records do not inflate the number of independent units in the bootstrap."""
import numpy as np

from stored_model_eval.fixtures import make_synthetic
from stored_model_eval.inference import cluster_bootstrap, resolve_units
from stored_model_eval.metrics import auc_binary


def _sd(H, S, unit_index, n_boot=400):
    stat = lambda w: auc_binary(S == 1, H[:, 0], w)  # noqa: E731 (fixed oracle score, no fitting)
    return cluster_bootstrap(stat, unit_index, n_boot=n_boot, seed=9)


def test_duplicates_collapse_to_units(record_property):
    one = make_synthetic("direct", n_units=500, seed=19, signal=0.3)
    dup = make_synthetic("direct", n_units=500, seed=19, signal=0.3, dup_factor=5)
    assert len(dup["S"]) == 2500 and len(np.unique(dup["units"])) == 2500  # duplicates got fresh unit ids
    u = resolve_units(dup["units"], dup["record_keys"])
    assert u["n_units"] == 500 and u["merged_by_record_key"] == 2000
    r_one = _sd(one["H"], one["S"], resolve_units(one["units"])["unit_index"])
    r_dup = _sd(dup["H"], dup["S"], u["unit_index"])
    r_naive = _sd(dup["H"], dup["S"], np.arange(2500))
    assert r_dup["n_units"] == 500 and r_naive["n_units"] == 2500
    assert abs(r_dup["point"] - r_one["point"]) < 1e-12
    ratio = r_dup["boot_sd"] / r_one["boot_sd"]
    record_property("n_units", [r_one["n_units"], r_dup["n_units"], r_naive["n_units"]])
    record_property("boot_sd", [r_one["boot_sd"], r_dup["boot_sd"], r_naive["boot_sd"]])
    assert 0.8 < ratio < 1.25, ratio
    assert r_naive["boot_sd"] / r_one["boot_sd"] < 0.6  # what treating duplicates as independent would do


def test_seeds_are_refit_replicates_not_units():
    fx = make_synthetic("direct", n_units=400, seed=20, signal=0.3)
    ui = resolve_units(fx["units"])["unit_index"]
    fns = [lambda w, k=k: auc_binary(fx["S"] == 1, fx["H"][:, 0] + 0.1 * k, w) for k in range(3)]
    r = cluster_bootstrap(fns, ui, n_boot=100)
    assert r["n_units"] == 400 and r["n_refit_seeds"] == 3
    assert "between_seed_sd" in r["refit_replicates"]
