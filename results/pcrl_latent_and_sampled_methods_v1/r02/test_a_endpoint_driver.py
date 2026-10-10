import numpy as np
import pytest
import json

from results.pcrl_latent_and_sampled_methods_v1.r02.a_endpoint_driver import (
    expected_attack_cells, normalized_cell, point_rows, save_private_cell,
    sha, verify_reader_custody,
)
from results.pcrl_latent_and_sampled_methods_v1.r02.a_inference import (
    A_ARMS, A_REFS, STOCHASTIC_ARMS,
)


def _plan():
    plans = {}
    for arm in (*A_ARMS, *A_REFS):
        for seed in (0, 1):
            for draw in (range(3) if arm in STOCHASTIC_ARMS else (0,)):
                kind = ("reference_attack" if arm in A_REFS else
                        "diagnostic_attack" if arm in ("A_IDENTITY", "A_CLASS_ONLY")
                        else "attack")
                unit_id = f"unit-{arm}-s{seed}-d{draw}"
                plans[unit_id] = {"kind": kind, "output_id": unit_id,
                                  "seed": seed, "realization": draw,
                                  ("reference_id" if arm in A_REFS else "arm_id"): arm}
    return plans


def test_exact_58_instances_and_deterministic_alias_inventory():
    plan = _plan()
    cells = expected_attack_cells(plan)
    assert len(cells) == 58
    assert ("A_LOCAL", 0, 2) in cells
    assert ("D0_LOCAL", 1, 0) in cells
    assert ("D0_LOCAL", 1, 1) not in cells
    duplicate = plan[cells["A_LOCAL", 0, 0]].copy()
    plan["second"] = {**duplicate, "output_id": "second"}
    with pytest.raises(ValueError, match="duplicate"):
        expected_attack_cells(plan)


def test_normalized_cell_rejects_changed_probability_or_missing_view():
    row = np.array([1, 2], dtype=np.int64)
    q1 = np.array([[.8, .2], [.3, .7]], dtype=np.float64)
    q2 = np.eye(6, dtype=np.float64)[[0, 1]]
    pred = {view: {"LR_C1": np.array([.1, .9])} for view in ("r1", "r2", "pair")}
    cell = normalized_cell(row, row, q1, q2, pred)
    assert cell["prob_1"].shape == (2, 2)
    with pytest.raises(ValueError, match="simplex"):
        normalized_cell(row, row, q1 + .1, q2, pred)
    with pytest.raises(ValueError, match="three frozen"):
        normalized_cell(row, row, q1, q2, {"r1": pred["r1"]})


def test_private_normalized_cell_retains_exact_reader_arrays(tmp_path):
    path = tmp_path / ".private" / "cell.npz"
    path.parent.mkdir()
    row = np.array([1, 2], dtype=np.int64)
    q1 = np.array([[.8, .2], [.3, .7]], dtype=np.float64)
    q2 = np.eye(6, dtype=np.float64)[[0, 1]]
    pred = {view: {"LR_C1": np.array([.1, .9])} for view in ("r1", "r2", "pair")}
    cell = normalized_cell(row, row, q1, q2, pred)
    receipt = save_private_cell(path, cell)
    assert receipt["sha256"] == sha(path)
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == set(receipt["npz_keys"])
        assert np.array_equal(archive["reader__r1__LR_C1"], pred["r1"]["LR_C1"])
    with pytest.raises(PermissionError, match="new private"):
        save_private_cell(path, cell)


def test_point_rows_reports_three_draw_spread_and_single_draw_zero():
    row = np.array([0, 1, 2, 3], dtype=np.int64)
    truth = {"row_id": row, "unit": row, "income": np.array([0, 1, 0, 1]),
             "occupation": np.array([0, 1, 0, 1]), "sex": np.array([0, 1, 0, 1])}
    q1 = np.array([[.8, .2], [.2, .8], [.7, .3], [.3, .7]])
    q2 = np.eye(6)[[0, 1, 0, 1]] * .9 + .1 / 6

    def release(delta=0):
        p1 = q1.copy()
        p1[:, 1] += delta
        p1[:, 0] -= delta
        return {"row_id": row, "unit": row, "prob_1": p1, "prob_2": q2,
                "reader_predictions": {v: {"LR_C1": np.array([.2, .8, .3, .7])}
                                       for v in ("r1", "r2", "pair")}}

    seed = {"truth": truth, "releases": {"A_LOCAL": {0: release(0), 1: release(.05),
                                                        2: release(-.05)},
                                           "U0": {0: release(0)}}}
    rows = point_rows({0: seed, 1: seed})
    assert len(rows) == 2 * (4 * 9 + 2 * 9)
    assert any(r["arm"] == "A_LOCAL" and r["draw"] == "mean" and
               r["draw_sd"] > 0 for r in rows)
    assert all(r["draw_sd"] == 0 for r in rows if r["arm"] == "U0" and
               r["draw"] == "mean")


def test_saved_complete_view_and_reader_hashes_are_replayed(tmp_path):
    from results.pcrl_latent_and_sampled_methods_v1.r02.complete_view import PreparedViews
    from attack_eval import AttackView

    root = tmp_path / ".private" / "attack"
    common = root / "common"
    common.mkdir(parents=True)
    row = np.array([1, 2], dtype=np.int64)
    view = AttackView(row_id=row, unit=row, x=np.array([[0.], [1.]]),
                      components=("probability",), wire_components=("probability",),
                      component_columns=(("probability", 0, 1),))
    prepared = {name: PreparedViews(view, view, view.x, view.x, {})
                for name in ("r1", "r2", "pair")}
    reader_sha = {}
    for name in prepared:
        folder = common / name / "readers"
        folder.mkdir(parents=True)
        files = {}
        for role in ("att", "inner"):
            path = folder / f"view_{role}.npz"
            np.savez_compressed(path, row_id=row, unit=row, x=view.x)
            files[path.name] = sha(path)
        readers = {"LR_C1": np.array([.1, .9], dtype=np.float64)}
        if name == "pair":
            readers.update({"IGNORE_R2__LR_C1": readers["LR_C1"],
                            "IGNORE_R1__LR_C1": readers["LR_C1"]})
        for role in ("tune", "inner"):
            path = folder / f"predictions_{role}.npz"
            np.savez_compressed(path, **readers)
            files[path.name] = sha(path)
        receipt = {"status": "FROZEN", "files": files,
                   "att_view_content_sha256": view.content_hashes(),
                   "inner_view_content_sha256": view.content_hashes(),
                   "reader_order": list(readers)}
        path = folder / "READERS_RECEIPT.json"
        path.write_text(json.dumps(receipt))
        reader_sha[name] = sha(path)
    freeze = {"status": "FROZEN_COMPLETE", "effective_lock_sha256": "a" * 64,
              "reader_receipt_sha256": reader_sha,
              "att_view_sha256": {v: view.content_hashes() for v in prepared},
              "inner_view_sha256": {v: view.content_hashes() for v in prepared}}
    freeze_path = common / "FROZEN_COMPLETE.json"
    freeze_path.write_text(json.dumps(freeze))
    index = {"status": "COMPLETE", "original_lock_sha256": "a" * 64,
             "control_status": {v: "PASS" for v in prepared},
             "reader_freeze_sha256": sha(freeze_path),
             "view_content_sha256": {v: {"att": view.content_hashes(),
                                            "inner": view.content_hashes()}
                                     for v in prepared}}
    index_path = root / "ATTACK_ATTEMPT_INDEX.json"
    index_path.write_text(json.dumps(index))
    bank = verify_reader_custody(root, prepared, index, index_sha256=sha(index_path))
    assert set(bank["pair"]) == {"LR_C1", "IGNORE_R2__LR_C1", "IGNORE_R1__LR_C1"}
    saved = common / "r1/readers/view_inner.npz"
    np.savez_compressed(saved, row_id=row, unit=row, x=np.array([[.2], [1.]]))
    with pytest.raises(ValueError, match="artifact hash"):
        verify_reader_custody(root, prepared, index, index_sha256=sha(index_path))
