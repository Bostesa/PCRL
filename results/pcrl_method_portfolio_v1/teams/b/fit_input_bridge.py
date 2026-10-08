"""Build Team B fitting inputs only through S2's admitted role-only store.

Task-only arms never request the protected-label file. Production callers must
pass S2's SafeRoleStore, whose file allowlists and byte checks precede this
builder. The returned seal binds its manifest and exact fitting arrays before
the queued arm fit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bottleneck import schema_fingerprint
from control_artifacts import file_sha256
from control_pipeline import FitAdmissionSeal
from train_bottleneck import FitData


TASK_ONLY = frozenset({"B-TASK", "B-RAW-TASK"})
ROLE = "OSF_DEFENSE_FIT"


def build_fit_input(store, *, arm: str, seed: int) -> tuple[FitData, FitAdmissionSeal]:
    """Read only the exact fields allowed by this registered B arm."""
    from control_plan import ARMS

    if arm not in ARMS or seed not in (0, 1, 2):
        raise ValueError("Unknown Team B arm or source seed")
    if store.manifest.get("status") != "ROLE_FILES_ADMITTED_S4_VERIFIED":
        raise ValueError("S2 role-only files lack independent admission")
    features = store.read(ROLE, "features")
    income = store.read(ROLE, "income", "defense_fit")
    occupation = store.read(ROLE, "occupation", "defense_fit")
    reference = store.read(ROLE, f"references_s{seed}")
    # No call to store.read(..., 'sex') occurs for task-only controls.
    sex = None if arm in TASK_ONLY else store.read(ROLE, "sex", "defense_fit")
    records = [features, income, occupation, reference]
    if sex is not None:
        records.append(sex)
    rid = np.asarray(features["row_id"])
    unit = np.asarray(features["unit"])
    if rid.dtype != np.int64 or unit.dtype != np.int64:
        raise ValueError("S2 fitting IDs must remain int64")
    for record in records[1:]:
        if (not np.array_equal(np.asarray(record["row_id"]), rid)
                or not np.array_equal(np.asarray(record["unit"]), unit)):
            raise ValueError("S2 fitting fields differ in ordered people/groups")
    names = tuple(str(name) for name in features["feature_names"])
    schema_sha = schema_fingerprint(names)
    data = FitData(
        x=features["X"],
        labels={"income": income["income"], "occupation": occupation["occupation"]},
        teacher_probabilities={"income": reference["Ucal_1"],
                               "occupation": reference["Ucal_2"]},
        sex=None if sex is None else sex["sex"],
        feature_names=names, permitted_feature_names=names,
        expected_schema_sha256=schema_sha,
        row_ids=tuple(str(int(value)) for value in rid),
        group_ids=tuple(str(int(value)) for value in unit),
        teacher_identity={"income": "Ucal_1", "occupation": "Ucal_2"},
    )
    manifest = store.manifest
    spec = manifest["real_file_specs"]
    role = manifest["roles"][ROLE]
    feature_spec = spec[f"{ROLE}:features"]
    ref_spec = spec[f"{ROLE}:references_s{seed}"]
    if feature_spec["row_unit_pair_sha256"] != ref_spec["row_unit_pair_sha256"]:
        raise ValueError("S2 fitting feature/reference rows differ")
    manifest_path = Path(store.manifest_path).resolve()
    seal = FitAdmissionSeal(
        source_id=manifest["source"]["source_npz_sha256"],
        role_id=ROLE, reference_seed=seed,
        admission_manifest_path=str(manifest_path),
        role_only_export_sha256=feature_spec["sha256"],
        reference_export_sha256=ref_spec["sha256"],
        source_npz_sha256=manifest["source"]["source_npz_sha256"],
        admission_manifest_sha256=file_sha256(manifest_path),
        admitted_rows=role["rows"], admitted_groups=role["groups"],
        admitted_row_id_sha256=role["row_id_sha256"],
        admitted_group_id_sha256=role["group_id_sha256"],
        admitted_row_unit_pair_sha256=feature_spec["row_unit_pair_sha256"],
        data_sha256=data.sha256,
        row_order_sha256=data.row_order_sha256,
        group_order_sha256=data.group_order_sha256,
        feature_schema_sha256=data.feature_schema_sha256,
        teacher_identity=tuple(sorted(data.teacher_identity.items())),
    )
    return data, seal
