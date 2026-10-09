"""Synthetic containment and stage integration before Adult fits."""

import hashlib
import json

import numpy as np

from fit_worker import gate, refit, screen
from fit_splits import outer_split
from prototype import AdultGroupedSchema


class FakeStore:
    def __init__(self):
        rng = np.random.default_rng(381)
        n = 96
        row = np.arange(n, dtype=np.int64)
        group = row.copy()
        self.feature = {"row_id": row, "unit": group,
                        "X": rng.normal(size=(n, 83)).astype(np.float32),
                        "feature_names": np.asarray(AdultGroupedSchema.feature_names)}
        self.income = {"row_id": row, "unit": group,
                       "income": (row % 2).astype(np.int64)}
        self.occupation = {"row_id": row, "unit": group,
                           "occupation": (row % 6).astype(np.int64)}
        self.by_role = {"OSF_DEFENSE_FIT": (self.feature, self.income, self.occupation)}
        for role, offset, count in (("CALIBRATION_HELDOUT", 100, 36),
                                    ("INNER_SELECTION", 200, 24)):
            rid = np.arange(offset, offset + count, dtype=np.int64)
            groups = rid.copy()
            fx = {"row_id": rid, "unit": groups,
                  "X": rng.normal(size=(count, 83)).astype(np.float32),
                  "feature_names": self.feature["feature_names"]}
            iy = {"row_id": rid, "unit": groups,
                  "income": (rid % 2).astype(np.int64)}
            oy = {"row_id": rid, "unit": groups,
                  "occupation": (rid % 6).astype(np.int64)}
            self.by_role[role] = (fx, iy, oy)
        names = tuple(str(value) for value in self.feature["feature_names"])
        self.manifest = {"source_schema_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest()}
        self.calls = []

    def read(self, role, field):
        self.calls.append((role, field))
        if field.startswith("references_s"):
            fx = self.by_role[role][0]
            n = len(fx["row_id"])
            p1 = np.tile(np.array([0.35, 0.65]), (n, 1))
            p2 = np.tile(np.arange(1, 7) / 21.0, (n, 1))
            return {"row_id": fx["row_id"], "unit": fx["unit"],
                    "U0_1": p1, "Ucal_1": p1,
                    "U0_2": p2, "Ucal_2": p2,
                    "d_1": p1.argmax(1), "d_2": p2.argmax(1)}
        return dict(zip(("features", "income", "occupation"), self.by_role[role]))[field]


def test_screen_uses_fit_only_and_saves_raw_predictions(tmp_path):
    store = FakeStore()
    root = tmp_path / ".private"
    root.mkdir()
    output = root / "r1/screen/seed_0/SCREEN_RECEIPT.json"
    receipt = screen(store, 0, root, output)
    assert set(store.calls) == {("OSF_DEFENSE_FIT", name)
                                for name in ("features", "income", "occupation")}
    assert receipt["calibration_role_access"] == "NOT_OPENED_IN_RECIPE_SELECTION"
    assert receipt["outer_split"]["validation_rows"] > 0
    with np.load(root / receipt["predictions_relative_path"], allow_pickle=False) as z:
        for recipe in ("A", "B"):
            assert z[f"{recipe}_income"].shape == (receipt["outer_split"]["validation_rows"], 2)
            assert z[f"{recipe}_occupation_group"].shape == (receipt["outer_split"]["validation_rows"], 6)
    assert hashlib.sha256(output.read_bytes()).hexdigest()


def test_outer_split_uses_prior_frozen_group_geometry():
    unit = np.repeat(np.arange(40, dtype=np.int64), 2)
    yi = (np.arange(len(unit)) % 2).astype(np.int64)
    yo = (np.arange(len(unit)) % 6).astype(np.int64)
    fit, val, receipt = outer_split(unit, yi, yo, 0)
    assert receipt["split_seed_offset"] == 26000
    assert receipt["selection_fraction"] == 0.25
    assert len(np.intersect1d(unit[fit], unit[val])) == 0


def test_b_refit_and_gate_requires_both_frozen_f_packages(tmp_path):
    store = FakeStore()
    root = tmp_path / ".private"
    root.mkdir()
    screens = {}
    for seed in (0, 1):
        output = root / f"r1/screen/seed_{seed}/SCREEN_RECEIPT.json"
        screens[seed] = screen(store, seed, root, output)
    selection = tmp_path / "RECIPE_SELECTION.json"
    selection.write_text(json.dumps({
        "schema": "pcrl-utility-recipe-selection-v1", "status": "FROZEN",
        "selected_recipe": "B", "rule": "lowest_sum_outer_fit_validation_nll_A_tie",
        "screen_output_sha256_by_seed": {
            str(seed): hashlib.sha256((root / f"r1/screen/seed_{seed}/SCREEN_RECEIPT.json").read_bytes()).hexdigest()
            for seed in (0, 1)},
        "selected_C_by_seed_task": {
            str(seed): {task: screens[seed]["packages"]["B"]["C_by_task"][task]["selected_C"]
                        for task in ("income", "occupation_group")}
            for seed in (0, 1)},
    }))
    full = {}
    for seed in (0, 1):
        output = root / f"r1/full/seed_{seed}/F_PACKAGE_RECEIPT.json"
        full[seed] = refit(store, seed, root, output, selection)
    f_lock = tmp_path / "F_ARTIFACT_LOCK.json"
    f_lock.write_text(json.dumps({
        "schema": "pcrl-utility-f-artifact-lock-v1", "status": "FROZEN",
        "recipe_selection_sha256": hashlib.sha256(selection.read_bytes()).hexdigest(),
        "full_refit_output_sha256_by_seed": {
            str(seed): hashlib.sha256((root / f"r1/full/seed_{seed}/F_PACKAGE_RECEIPT.json").read_bytes()).hexdigest()
            for seed in (0, 1)},
        "f_package_sha256_by_seed_task": {
            str(seed): {task: full[seed]["task_packages"][task]["sha256"]
                        for task in ("income", "occupation_group")}
            for seed in (0, 1)},
    }))
    result = gate(store, 0, root, root / "r1/gate/seed_0/GATE_RECEIPT.json",
                  selection, f_lock)
    assert result["inner_row_unit_sha256"]
    assert len(result["margins"]) == 2
