import numpy as np
import pytest
import json

from results.pcrl_latent_and_sampled_methods_v1.r02.a_endpoint_driver import (
    FINITE_READERS, expected_attack_cells, normalized_cell, point_rows, save_private_cell,
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
    assert path.stat().st_mode & 0o777 == 0o600
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
        readers = {reader: np.array([.1, .9], dtype=np.float64)
                   for reader in FINITE_READERS}
        alias_records = []
        if name == "pair":
            for source, prefix in (("r1", "IGNORE_R2__"), ("r2", "IGNORE_R1__")):
                for reader in FINITE_READERS:
                    alias = prefix + reader
                    readers[alias] = readers[reader]
                    alias_records.append({"alias": alias, "source_view": source,
                                          "source_reader": reader, "new_fit": False})
        for role in ("tune", "inner"):
            path = folder / f"predictions_{role}.npz"
            np.savez_compressed(path, **readers)
            files[path.name] = sha(path)
        for reader in FINITE_READERS:
            suffix = ".bin" if reader.startswith(("PROB_", "Q_POST_")) else ".joblib"
            path = folder / f"model__{reader}{suffix}"
            path.write_bytes(b"synthetic model")
            files[path.name] = sha(path)
        receipt = {"schema": "pcrl-portfolio-private-reader-custody-v1",
                   "status": "FROZEN", "files": files,
                   "att_view_content_sha256": view.content_hashes(),
                   "inner_view_content_sha256": view.content_hashes(),
                   "reader_order": list(readers),
                   "ignore_recipient_aliases": alias_records}
        path = folder / "READERS_RECEIPT.json"
        path.write_text(json.dumps(receipt))
        reader_sha[name] = sha(path)
        control_root = common / name / "controls"
        control_root.mkdir()
        control_files = {}
        for filename in ("shuffled_model.joblib", "planted_model.joblib",
                         "probe_replay.npz"):
            file = control_root / filename
            file.write_bytes(b"synthetic")
            control_files[filename] = sha(file)
        (control_root / "CUSTODY_RECEIPT.json").write_text(json.dumps({
            "schema": "pcrl-portfolio-private-control-v2-custody-v1",
            "status": "FROZEN", "files": control_files,
            "att_view_content_sha256": view.content_hashes(),
            "inner_view_content_sha256": view.content_hashes()}))
    freeze = {"status": "FROZEN_COMPLETE", "effective_lock_sha256": "a" * 64,
              "reader_receipt_sha256": reader_sha,
              "att_view_sha256": {v: view.content_hashes() for v in prepared},
              "inner_view_sha256": {v: view.content_hashes() for v in prepared}}
    freeze_path = common / "FROZEN_COMPLETE.json"
    freeze_path.write_text(json.dumps(freeze))
    index = {"schema": "SCV1-sampled-attack-attempt-v1", "status": "COMPLETE",
             "original_lock_sha256": "a" * 64,
             "control_status": {v: "PASS" for v in prepared},
             "reader_freeze_sha256": sha(freeze_path),
             "view_content_sha256": {v: {"att": view.content_hashes(),
                                            "inner": view.content_hashes()}
                                     for v in prepared}}
    index_path = root / "ATTACK_ATTEMPT_INDEX.json"
    index_path.write_text(json.dumps(index))
    bank = verify_reader_custody(root, prepared, index, index_sha256=sha(index_path))
    assert len(bank["pair"]) == 27
    saved = common / "r1/readers/view_inner.npz"
    original_bytes = saved.read_bytes()
    np.savez_compressed(saved, row_id=row, unit=row, x=np.array([[.2], [1.]]))
    with pytest.raises(ValueError, match="artifact hash"):
        verify_reader_custody(root, prepared, index, index_sha256=sha(index_path))
    saved.write_bytes(original_bytes)
    receipt_path = common / "r1/readers/READERS_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["reader_order"].remove("Q_POST_LR_C1")
    receipt_path.write_text(json.dumps(receipt))
    freeze["reader_receipt_sha256"]["r1"] = sha(receipt_path)
    freeze_path.write_text(json.dumps(freeze))
    index["reader_freeze_sha256"] = sha(freeze_path)
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="reader receipt view identity"):
        verify_reader_custody(root, prepared, index, index_sha256=sha(index_path))
    receipt["reader_order"].append("Q_POST_LR_C1")
    del receipt["files"]["model__Q_POST_LR_C1.bin"]
    receipt_path.write_text(json.dumps(receipt))
    freeze["reader_receipt_sha256"]["r1"] = sha(receipt_path)
    freeze_path.write_text(json.dumps(freeze))
    index["reader_freeze_sha256"] = sha(freeze_path)
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="required view/prediction/model inventory"):
        verify_reader_custody(root, prepared, index, index_sha256=sha(index_path))


def test_diagnostic_replay_uses_canonical_bank_order_for_saved_code_ids(tmp_path):
    from results.pcrl_latent_and_sampled_methods_v1.r01.role_store import sha_array
    from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.attacks.run_reference_attack import (
        _diagnostic_q_mechanism,
    )
    from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.attacks.reference_wire import (
        load_reference_role_package,
    )
    from results.pcrl_latent_and_sampled_methods_v1.sampled_channel_v1.impl.reference_materialize import (
        materialize_reference_role,
    )

    old1 = np.array([[.1, .9], [.9, .1]], dtype=np.float64)
    permutation = np.array([4, 1, 5, 0, 3, 2])
    old2 = (np.eye(6, dtype=np.float64) * .9 + .1 / 6)[permutation]
    centroids = (np.eye(2, dtype=np.float64)[np.arange(64) % 2],
                 np.eye(6, dtype=np.float64)[np.arange(641) % 6])
    fixed = {"bank_r1": old1, "bank_r2": old2,
             "fine_to_code_r1": np.array([1, 0], dtype=np.int64)[
                 np.argmax(centroids[0], axis=1)],
             "fine_to_code_r2": np.argsort(permutation)[
                 np.argmax(centroids[1], axis=1)].astype(np.int64)}
    binding = "a" * 64
    original_receipt = "b" * 64
    mechanism, descriptor, _, canonical1, canonical2 = _diagnostic_q_mechanism(
        fixed, {"source_binding_sha256": binding,
                "original_bank_receipt_sha256": original_receipt}, centroids,
        ref="A_IDENTITY", seed=0, release_id="A_IDENTITY-s0-v1")
    assert len(mechanism) == len(descriptor) == 64
    assert not np.array_equal(canonical1, old1)
    assert not np.array_equal(canonical2, old2)
    row = np.array([11, 12], dtype=np.int64)
    unit = np.array([21, 22], dtype=np.int64)
    code = np.array([0, 1], dtype=np.int64)
    root = tmp_path / ".private" / "diagnostic"
    root.parent.mkdir(mode=0o700)
    materialize_reference_role(
        row_id=row, unit=unit, prob1=canonical1[code], prob2=canonical2[code],
        decision1=np.argmax(canonical1[code], axis=1).astype(np.int64),
        decision2=np.argmax(canonical2[code], axis=1).astype(np.int64),
        code1=code, code2=code, arm_id="A_IDENTITY", seed=0,
        role="INNER_SELECTION", release_id="A_IDENTITY-s0-v1",
        source_binding_sha256=binding,
        source_identity={"role_source_file_sha256": "c" * 64,
                         "reference_file_sha256": "d" * 64,
                         "bank_receipt_sha256": original_receipt,
                         "source_audit_sha256": "e" * 64},
        interface_kind="finite",
        bank_sha256_by_recipient={"r1": sha_array(canonical1),
                                  "r2": sha_array(canonical2)},
        expected_row_unit_sha256=sha_array(np.column_stack((row, unit))),
        output_dir=root, bank1=canonical1, bank2=canonical2)
    digest = sha(root / "REFERENCE_RELEASE_RECEIPT.json")
    restored = load_reference_role_package(root, receipt_sha256=digest,
                                           full_bank1=canonical1,
                                           full_bank2=canonical2)
    assert restored.wires1[0]["probability"] == canonical1[0].tolist()
    with pytest.raises(ValueError, match="bank|code"):
        load_reference_role_package(root, receipt_sha256=digest,
                                    full_bank1=old1, full_bank2=old2)


def test_endpoint_replay_imports_originate_from_pinned_worktree():
    from results.pcrl_latent_and_sampled_methods_v1.r02.a_endpoint_worker import (
        _assert_module_origins,
    )
    _assert_module_origins()
