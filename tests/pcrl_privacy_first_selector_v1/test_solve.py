"""Synthetic tests for the privacy-first selector solver (no ACS data)."""
import itertools
import json

import numpy as np
import pytest

from experiments.pcrl_privacy_first_selector_v1 import solve
from experiments.pcrl_shared_context_release_v1 import channel, fit_nm, laws
from tests.pcrl_shared_context_release_v1.test_channel import synthetic_instance


def _calibrated(seed=0, K=4, M=5, n=900, n_attacks=8):
    inst = synthetic_instance(seed=seed, n=n, K=K, M=M, n_attacks=n_attacks)
    calibrated = channel.calibrate_nested(inst["d17"], inst["bank"], .001)
    return inst, inst["cost"], calibrated["cuts"], calibrated


def test_d17_witness_has_zero_t_and_is_feasible():
    inst, cost, cuts, _ = _calibrated()
    caps = solve.task_caps(cost, inst["d17"])
    rec = solve.evaluate(inst["d17"], np.zeros((4, 5)), cost, cuts, caps)
    assert rec["feasible"] and abs(rec["t"]) < 1e-12
    compact = solve.evaluate(*solve._assignment_params([0] * 4, 4, 5)[:2], cost, cuts, caps)
    assert compact["feasible"] and abs(compact["t"]) < 1e-10


def test_enumeration_is_exact_and_tie_break_registered():
    inst, cost, cuts, _ = _calibrated(seed=1)
    caps = solve.task_caps(cost, inst["d17"])
    rows = solve.enumerate_assignments(cost, cuts, caps, 4, 5)
    assert len(rows) == 625
    best, n_feasible = solve.best_privacy(rows)
    brute = []
    for index in itertools.product(range(5), repeat=4):
        B, A, _ = solve._assignment_params(index, 4, 5)
        rec = solve.evaluate(B, A, cost, cuts, caps)
        if rec["feasible"]:
            brute.append((rec["t"], list(index)))
    assert n_feasible == len(brute)
    assert best["t"] == pytest.approx(max(t for t, _ in brute), abs=1e-15)


def test_mixture_lp_dominates_enumeration_and_matches_grid_k1():
    inst, cost, cuts, _ = _calibrated(seed=2)
    caps = solve.task_caps(cost, inst["d17"])
    c1, k1 = solve.collapse_k1(cost, cuts)
    r1 = solve.solve_mixture(c1, k1, caps, 1, 5)
    d1, _ = solve.best_privacy(solve.enumerate_assignments(c1, k1, caps, 1, 5))
    assert r1["record"]["t"] >= d1["t"] - 1e-9
    # coarse grid over the 5-simplex: never better than the LP
    step = 10
    best_grid = -np.inf
    for parts in itertools.product(range(step + 1), repeat=4):
        if sum(parts) > step:
            continue
        a = np.array([*parts, step - sum(parts)], dtype=float)[None, :] / step
        rec = solve.evaluate(np.zeros((32, 17)), a, c1, k1, caps)
        if rec["feasible"]:
            best_grid = max(best_grid, rec["t"])
    assert r1["record"]["t"] >= best_grid - 1e-9
    assert abs(r1["record"]["dual_gap"]) < 1e-6
    r4 = solve.solve_mixture(cost, cuts, caps, 4, 5)
    assert r4["record"]["t"] >= r1["record"]["t"] - 1e-9
    assert np.allclose(r4["A"].sum(1), 1) and r4["A"].min() >= 0


def test_k1_collapse_equals_direct_k1_blocks():
    inst = synthetic_instance(seed=3, n=500, K=4, M=5)
    c1, _ = solve.collapse_k1(inst["cost"], [])
    direct = channel.coefficient_pair_blocks(inst["task"], inst["codes"], np.zeros_like(inst["ctx"]),
                                             inst["tokens"], inst["pwgtp"], n_contexts=1)
    for v in ("U", "W"):
        assert np.allclose(c1[v]["A"], direct[v]["A"], atol=1e-14, rtol=0)
        assert np.array_equal(c1[v]["B"], direct[v]["B"])


def _fake_units(tmp_path, seed=4):
    inst, cost, cuts, calibrated = _calibrated(seed=seed)
    bank = tmp_path / "prior" / "a0_bank"
    bank.mkdir(parents=True)
    (bank / "BANK_COMPLETE.json").write_text(json.dumps(
        {"policy_bank_sha256": "p" * 64, "contexts_sha256": "c" * 64}))
    nm = tmp_path / "prior" / "a0_NM4_U"
    (nm / "closing").mkdir(parents=True)
    fit_nm._save_npz(nm / "closing" / "TASK_BLOCKS.npz", U_B=cost["U"]["B"], U_A=cost["U"]["A"],
                     W_B=cost["W"]["B"], W_A=cost["W"]["A"])
    fit_nm._save_cut_blocks(nm / "closing" / "CALIBRATED_BLOCKS.npz", cuts)
    (nm / "closing" / "CALIBRATED_BANK.json").write_text(json.dumps(
        {"bank_sha256": calibrated["bank_sha256"], "rho": calibrated["rho"], "cuts": fit_nm._meta(cuts)},
        default=str))
    (nm / "INPUTS.json").write_text("{}")
    receipt = {"variant": "NM4_U", "K": 4, "smoke": False, "status": "COMPLETE", "anchor": 0,
               "policy_columns": list(solve.POLICY_COLUMNS),
               "bank_manifest_sha256": laws.sha256_file(bank / "BANK_COMPLETE.json"),
               "artifact_sha256": fit_nm._inventory(nm)}
    (nm / "COMPLETE.json").write_text(json.dumps(receipt))
    det = tmp_path / "prior" / "a0_DET_SEL4"
    det.mkdir()
    best, _ = fit_nm.deterministic_selector(cost, cuts, 4, list(solve.POLICY_COLUMNS))
    (det / "DET_SEL.json").write_text(json.dumps({"selected": best}))
    fit_nm._save_npz(det / "PARAMS.npz", B=np.zeros((32, 17)),
                     A=solve._assignment_params(best["index"], 4, 5)[1], eta=np.asarray(1.))
    return inst, nm, bank, det


def test_solve_anchor_end_to_end(tmp_path):
    inst, nm, bank, det = _fake_units(tmp_path)
    out = tmp_path / "private" / "units"
    report = solve.solve_anchor(nm, bank, det, inst["d17"], out)
    assert all(report["nesting_checks"].values())
    for release in solve.RELEASES:
        root = out / f"a0_{release}"
        descriptor = laws.read_descriptor(root)
        assert descriptor["kind"] == "nested"
        spec = json.loads((root / "RELEASE_SPEC.json").read_text())
        assert (root / spec["bank_dir_relative"]).resolve() == bank.resolve()
        with np.load(root / "PARAMS.npz") as p:
            channel.validate_params(p["B"], p["A"], float(p["eta"]))
    d4 = report["releases"]["D4"]
    assert d4["t"] >= report["releases"]["DET_SEL4"]["t"] - 1e-12 or not report["releases"]["DET_SEL4"]["feasible"]
    assert report["releases"]["TASK_SEL4"]["task_balanced"] <= d4["task_balanced"] + 1e-15
    with pytest.raises(FileExistsError):
        solve.solve_anchor(nm, bank, det, inst["d17"], out)


def test_det_sel4_mismatch_aborts(tmp_path):
    inst, nm, bank, det = _fake_units(tmp_path, seed=5)
    record = json.loads((det / "DET_SEL.json").read_text())
    record["selected"]["index"] = [(i + 1) % 5 for i in record["selected"]["index"]]
    (det / "DET_SEL.json").write_text(json.dumps(record))
    with pytest.raises(AssertionError):
        solve.solve_anchor(nm, bank, det, inst["d17"], tmp_path / "private" / "u")


def test_tampered_basis_refused(tmp_path):
    inst, nm, bank, det = _fake_units(tmp_path, seed=6)
    with np.load(nm / "closing" / "TASK_BLOCKS.npz") as t:
        arrays = {k: t[k].copy() for k in t.files}
    arrays["U_A"][0, 1] += 1e-3
    np.savez(nm / "closing" / "TASK_BLOCKS.npz", **arrays)
    with pytest.raises(ValueError):
        solve.load_basis(nm, bank)
