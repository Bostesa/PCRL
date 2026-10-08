"""Synthetic S2 store spy proving task-only access is genuinely label-free."""

import json
from pathlib import Path

import numpy as np
import pytest

from bottleneck import model_fingerprint
from control_artifacts import critic_fingerprint
from control_pipeline import fit_arm
from fit_input_bridge import build_fit_input
from test_control_artifacts import fixtures
from test_control_pipeline import admitted, task_only_data


class SpyStore:
    def __init__(self, manifest_path: str, data, *, sex_must_not_be_read: bool):
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text())
        self.data = data
        self.sex_must_not_be_read = sex_must_not_be_read
        self.reads = []

    def read(self, role, field, procedure=None):
        assert role == "OSF_DEFENSE_FIT"
        self.reads.append((field, procedure))
        if field == "sex" and self.sex_must_not_be_read:
            raise AssertionError("Task-only builder accessed protected SEX")
        ids = np.asarray(self.data.row_ids, dtype=np.int64)
        groups = np.asarray(self.data.group_ids, dtype=np.int64)
        common = {"row_id": ids, "unit": groups}
        if field == "features":
            return common | {"X": self.data.x.numpy(),
                             "feature_names": np.asarray(self.data.feature_names)}
        if field in ("income", "occupation"):
            return common | {field: self.data.labels[field].numpy()}
        if field.startswith("references_s"):
            return common | {"Ucal_1": self.data.teacher_probabilities["income"].numpy(),
                             "Ucal_2": self.data.teacher_probabilities["occupation"].numpy()}
        if field == "sex":
            return common | {"sex": self.data.sex.numpy()}
        raise KeyError(field)


@pytest.mark.parametrize("arm", ["B-TASK", "B-RAW-TASK"])
def test_task_only_build_and_fit_never_requests_sex(tmp_path, arm):
    base, critics, full, hyper = fixtures()
    data = task_only_data(full)
    seal = admitted(data, tmp_path, include_sex=True)
    store = SpyStore(seal.admission_manifest_path, data, sex_must_not_be_read=True)
    built, built_seal = build_fit_input(store, arm=arm, seed=0)
    assert built.sex is None
    assert built_seal.data_sha256 == built.sha256
    assert "sex" not in [field for field, _ in store.reads]
    fit_arm(
        arm, 0, data=built, base_model=base, base_critics=critics,
        hyper=hyper, source_id=built_seal.source_id, role_id=built_seal.role_id,
        admitted_fit=built_seal, private_attempt_dir=tmp_path / ".private" / arm,
        expected_base_model_sha256=model_fingerprint(base),
        expected_base_critics_sha256=critic_fingerprint(critics),
    )
    assert "sex" not in [field for field, _ in store.reads]


def test_protected_builder_requests_complete_sex(tmp_path):
    _, _, data, _ = fixtures()
    seal = admitted(data, tmp_path, include_sex=True)
    store = SpyStore(seal.admission_manifest_path, data, sex_must_not_be_read=False)
    built, built_seal = build_fit_input(store, arm="B-JVIB", seed=0)
    assert built.sex is not None
    assert built_seal.data_sha256 == built.sha256
    assert ("sex", "defense_fit") in store.reads
