"""Synthetic contracts for the separate continuous-wire audit adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.pcrl_adaptive_release_v1 import external_audit, roles
from experiments.pcrl_task_aligned_cuts_v1 import audit as inherited_audit


POOLS = ("representation_fit", "downstream_fit", "downstream_validation", "attacker_fit")


def _prepared() -> dict:
    pools = {}
    for j, name in enumerate(POOLS):
        n = 6
        ids = np.asarray([f"{name}-{i}" for i in range(n)])
        houses = []
        candidate = 0
        for role in ("audit_fit", "inner_selection", "inner_check"):
            matches = []
            while len(matches) < 2:
                name_candidate = f"house-{j}-{candidate}"
                candidate += 1
                if roles.role_of(name_candidate) == role:
                    matches.append(name_candidate)
            houses.extend(matches)
        houses = np.asarray(houses)
        ha = np.column_stack((np.arange(n) + 10 * j, np.ones((n, 3)))).astype(np.float64)
        hb = np.column_stack((np.arange(n) + 20 * j, np.zeros(n))).astype(np.float64)
        pools[name] = {
            "x": np.zeros((n, 32)), "ha": ha, "hb": hb,
            "J": np.full((n, 16), j + 0.25),
            "weights": np.ones(n), "ids": ids, "households": houses,
            "labels": {"same_residence": np.zeros(n, dtype=int),
                       "SEX": np.zeros(n, dtype=int), "RAC1P": np.zeros(n, dtype=int)},
        }
    encoded = {name: {"codes": {"T0": np.zeros(6, dtype=int)},
                      "p": np.zeros(6), "r": np.zeros(6), "risk": np.zeros((6, 11))}
               for name in POOLS}
    return {"ctx": {"pools": pools}, "encoded": encoded}


def _external_file(path: Path, prepared: dict, *, corrupt_h: bool = False) -> str:
    arrays = {}
    for j, name in enumerate(POOLS):
        pool = prepared["ctx"]["pools"][name]
        a = np.column_stack((pool["ha"], np.full((6, 16), j + 0.75)))
        if corrupt_h and name == "attacker_fit":
            a[0, 0] += 1
        arrays[f"wire/A/{name}"] = a
        arrays[f"wire/B/{name}"] = pool["hb"]
        arrays[f"wire/AB/{name}"] = np.column_stack((a, pool["hb"]))
    np.savez_compressed(path, **arrays)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_j_rows_follow_global_household_roles_and_preserve_h() -> None:
    prepared = _prepared()
    for role in ("audit_fit", "inner_selection", "inner_check"):
        result = external_audit.continuous_role_rows(prepared, role, "J")
        expected = []
        for j, name in enumerate(POOLS):
            pool = prepared["ctx"]["pools"][name]
            mask = np.asarray([roles.role_of(h) == role for h in pool["households"]])
            expected.append(np.full((int(mask.sum()), 16), j + 0.25))
        assert np.array_equal(result["aux"], np.concatenate(expected))
        assert result["ha"].tobytes() == roles.pooled_role(prepared, role)["ha"].tobytes()
        assert result["hb"].tobytes() == roles.pooled_role(prepared, role)["hb"].tobytes()


def test_continuous_wire_routes_use_one_bookkeeping_state_and_actual_service() -> None:
    rows = external_audit.continuous_role_rows(_prepared(), "audit_fit", "J")
    law = external_audit._unit_law(rows)
    a = inherited_audit.role_arrays(rows, law, "utility:A/same_residence")
    ab = inherited_audit.role_arrays(rows, law, "attack:AB/SEX")
    h = inherited_audit.role_arrays(external_audit._no_aux(rows), law,
                                    "utility:A/same_residence")
    assert law.shape == (len(rows["ids"]), 1)
    assert a["h"].shape[1] == 20
    assert ab["h"].shape[1] == 22
    assert h["h"].shape[1] == 4
    assert np.array_equal(a["h"][:, :4], rows["ha"])
    assert np.array_equal(ab["h"][:, :6], np.column_stack((rows["ha"], rows["hb"])))
    assert np.array_equal(ab["h"][:, 6:], rows["aux"])


def test_external_file_rejects_changed_h_before_role_filter(tmp_path: Path) -> None:
    prepared = _prepared()
    source = tmp_path / "release.npz"
    digest = _external_file(source, prepared, corrupt_h=True)
    with pytest.raises(ValueError, match="service byte parity"):
        external_audit.verified_external_aux(source, digest, prepared)


def test_external_file_rejects_hash_change_and_preserves_archived_order(tmp_path: Path) -> None:
    prepared = _prepared()
    source = tmp_path / "release.npz"
    digest = _external_file(source, prepared)
    aux = external_audit.verified_external_aux(source, digest, prepared)
    assert set(aux) == set(POOLS)
    assert np.array_equal(aux["representation_fit"], np.full((6, 16), 0.75))
    assert np.array_equal(external_audit.archived_ab_wire(
        prepared["ctx"]["pools"]["representation_fit"], aux["representation_fit"]),
        np.column_stack((prepared["ctx"]["pools"]["representation_fit"]["ha"],
                         aux["representation_fit"],
                         prepared["ctx"]["pools"]["representation_fit"]["hb"])))
    with pytest.raises(ValueError, match="SHA-256"):
        external_audit.verified_external_aux(source, "0" * 64, prepared)


def test_external_index_pins_only_declared_release(tmp_path: Path) -> None:
    prepared = _prepared()
    source = tmp_path / "releases.npz"
    digest = _external_file(source, prepared)
    index = tmp_path / "external.private.json"
    index.write_text(json.dumps({"schema": "pcrl-external-release-inputs-private-v1",
        "historical_competitive_commit": "7f961d5c7f6f0562efcb25a27a77bb5221c279a7",
        "verified_release_file_count": 15, "service_view_mismatch_count": 0,
        "release_files": [
        {"anchor": 0, "method": "leace_A0", "sha256": digest,
         "archive_member_path": "private/seed_0/leace_A0/releases.npz"}]}))
    index_sha = hashlib.sha256(index.read_bytes()).hexdigest()
    _, record = external_audit.pinned_external_record(index, index_sha, 0, "leace_A0")
    assert record["sha256"] == digest
    with pytest.raises(ValueError, match="undeclared"):
        external_audit.pinned_external_record(index, index_sha, 0, "new_candidate")
