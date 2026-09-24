"""Family generation, lock structure and outer labels (synthetic; no ACS data)."""
import copy

import pytest

from experiments.pcrl_privacy_first_selector_v1 import host, lockgate, outer

ROLES = ("utility:A/same_residence", "attack:A/SEX", "attack:A/RAC1P", "attack:AB/SEX", "attack:AB/RAC1P")


def _report(mapping):
    route = {"source_view": "A", "wire": "token", "model_sha256": "m" * 64}
    releases = {c: {"source": {"kind": "nested", "release_id": c},
                    "roles": {r: {"selected_route": route, "H_selected_route": {**route, "wire": "H"}}
                              for r in ROLES}}
                for c in sorted(set(mapping.values()))}
    return {"logical_to_canonical": mapping, "releases": releases}


def _pins():
    return {str(a): {"inner_panel_complete_sha256": "a" * 64, "inner_audit_sha256": "b" * 64,
                     "index_sha256": "c" * 64} for a in (0, 1, 2)}


PRIOR = {"positive_controls": {f"{a}|attack:{r}": {"detected": True} for a in (0, 1, 2)
                               for r in ("AB/SEX", "AB/RAC1P")}}


def test_family_counts_without_aliases():
    fam = lockgate.generate({})
    assert len(fam["primary"]) == 30 and len(fam["secondary"]) == 110 and len(fam["capability"]) == 4
    assert {e["clause"] for e in fam["primary"]} == {"task", "target", "guard"}
    target = [e for e in fam["primary"] if e["clause"] == "target"]
    assert all(e["role"] == "attack:AB/SEX" and e["threshold"] == -.002 for e in target)
    assert all(e["threshold"] == .001 for e in fam["primary"] if e["clause"] != "target")


def test_lock_roundtrip_and_alias_merging():
    distinct = {n: n for n in host.DECLARED}
    reports = {a: _report(distinct) for a in (0, 1, 2)}
    lock = lockgate.build_lock(reports, _pins(), code_commit="f" * 40, extra_pins={},
                               prior_lock=PRIOR, prior_lock_sha256="d" * 64)
    lockgate.verify_lock_structure(lock)
    assert lock["family_manifest"]["n_endpoints"] == 30
    # TASK_SEL4 == DET_SEL4 on every anchor: merged inside the D4 control slot only
    merged = dict(distinct, TASK_SEL4="DET_SEL4")
    lock2 = lockgate.build_lock({a: _report(merged) for a in (0, 1, 2)}, _pins(), code_commit="f" * 40,
                                extra_pins={}, prior_lock=PRIOR, prior_lock_sha256="d" * 64)
    assert lock2["alias_of"] == {"TASK_SEL4": "DET_SEL4"}
    assert lock2["family_manifest"]["n_endpoints"] == 30
    assert lock2["secondary_manifest"]["n_endpoints"] == 100
    tampered = copy.deepcopy(lock2)
    tampered["family_manifest"]["endpoints"][0]["threshold"] = 1.0
    with pytest.raises(PermissionError):
        lockgate.verify_lock_structure(tampered)


def test_exact_zero_when_candidate_is_comparator_everywhere():
    mapping = {n: n for n in host.DECLARED}
    mapping["D4"] = "D17"
    lock = lockgate.build_lock({a: _report(mapping) for a in (0, 1, 2)}, _pins(), code_commit="f" * 40,
                               extra_pins={}, prior_lock=PRIOR, prior_lock_sha256="d" * 64)
    d_rows = [e for e in lock["family_manifest"]["endpoints"] if e["candidate"] == "D"]
    assert all(e["exact_zero_same_route"] for e in d_rows)


def _rows(d_values):
    rows = []
    for slot in ("D", "R", "R_vs_D"):
        for role in ROLES:
            for w in ("U", "PWGTP"):
                rows.append({"candidate": slot, "role": role, "weighting": w,
                             "estimate": d_values.get((role, w), 0.) if slot == "D" else 0.,
                             "passed_upper_bound": False})
    return rows


def test_lead_reproduced_rule():
    good = {("attack:AB/SEX", "U"): -.0025, ("attack:AB/SEX", "PWGTP"): -.0021,
            ("utility:A/same_residence", "U"): .0009, ("utility:A/same_residence", "PWGTP"): -.001}
    assert outer.labels(_rows(good))["LEAD_REPRODUCED"]
    assert not outer.labels(_rows({**good, ("attack:AB/SEX", "PWGTP"): -.0019}))["LEAD_REPRODUCED"]
    assert not outer.labels(_rows({**good, ("attack:A/RAC1P", "U"): .0011}))["LEAD_REPRODUCED"]
    assert not outer.labels(_rows({**good, ("utility:A/same_residence", "U"): .0011}))["LEAD_REPRODUCED"]
