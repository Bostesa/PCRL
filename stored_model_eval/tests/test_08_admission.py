"""Test 8: admission rejects row/label mismatch, equal-length reordered arrays, duplicated ids, values
shuffled under intact ids (reference check), hash mismatch, missing id arrays and units crossing roles."""
import json

import numpy as np
import pytest

from stored_model_eval.admission import admit, sha256_file
from stored_model_eval.fixtures import make_synthetic, write_manifest


@pytest.fixture
def base(tmp_path):
    fx = make_synthetic("direct", n_units=300, seed=18)
    mp = write_manifest(fx, tmp_path)
    return fx, tmp_path, json.loads(mp.read_text())


def _with_labels(tmp_path, man, ids, y, name="labels_variant"):
    p = tmp_path / f"{name}.npz"
    np.savez(p, row_id=ids, S=y)
    m = json.loads(json.dumps(man))
    m["files"]["lab"] = {"path": p.name, "sha256": sha256_file(p)}
    m["arrays"]["labels"] = {"file": "lab", "key": "S", "ids": "row_id"}
    return m


def test_baseline_admitted(base):
    fx, d, man = base
    rec = admit(man, d)
    assert rec["status"] == "ADMITTED", rec["errors"]
    assert rec["role_summary"]["evaluation"]["n_rows"] > 0


def test_equal_length_reordered_rejected(base):
    fx, d, man = base
    perm = np.random.default_rng(0).permutation(len(fx["S"]))
    m = _with_labels(d, man, fx["row_ids"][perm], fx["S"][perm])  # consistent but reordered
    rec = admit(m, d)
    assert rec["status"] == "REJECTED" and any("different order" in e for e in rec["errors"])


def test_id_mismatch_rejected(base):
    fx, d, man = base
    ids = fx["row_ids"].copy()
    ids[0] = 99_999_999
    rec = admit(_with_labels(d, man, ids, fx["S"]), d)
    assert rec["status"] == "REJECTED" and any("ID sets differ" in e for e in rec["errors"])


def test_duplicated_ids_rejected(base):
    fx, d, man = base
    ids = fx["row_ids"].copy()
    ids[1] = ids[0]
    rec = admit(_with_labels(d, man, ids, fx["S"]), d)
    assert rec["status"] == "REJECTED" and any("duplicated ids" in e for e in rec["errors"])


def test_values_shuffled_under_intact_ids_caught_by_reference(base):
    fx, d, man = base
    y = fx["S"][np.random.default_rng(1).permutation(len(fx["S"]))]
    m = _with_labels(d, man, fx["row_ids"], y)
    assert admit(m, d)["status"] == "ADMITTED"  # undetectable without a reference ...
    m["reference_checks"] = [{"array": "labels", "file": "data", "key": "S", "ids": "row_id"}]
    rec = admit(m, d)
    assert rec["status"] == "REJECTED" and any("disagree with the reference" in e for e in rec["errors"])


def test_hash_missing_ids_and_role_crossing(base):
    fx, d, man = base
    m = json.loads(json.dumps(man))
    m["files"]["data"]["sha256"] = "0" * 64
    assert any("sha256 mismatch" in e for e in admit(m, d)["errors"])
    m = json.loads(json.dumps(man))
    del m["arrays"]["labels"]["ids"]
    assert any("no explicit ID array" in e for e in admit(m, d)["errors"])
    units = fx["units"].copy()
    ev = np.flatnonzero(fx["roles"] == "evaluation")[0]
    fit = np.flatnonzero(fx["roles"] == "attacker_fit")[0]
    units[ev] = units[fit]
    p = d / "units_variant.npz"
    np.savez(p, row_id=fx["row_ids"], unit=units)
    m = json.loads(json.dumps(man))
    m["files"]["u"] = {"path": p.name, "sha256": sha256_file(p)}
    m["arrays"]["units"] = {"file": "u", "key": "unit", "ids": "row_id"}
    rec = admit(m, d)
    assert rec["status"] == "REJECTED" and any("more than one role" in e for e in rec["errors"])
