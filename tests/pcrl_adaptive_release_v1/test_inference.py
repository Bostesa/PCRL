"""Arm-specific, household-paired inference fixtures."""
from __future__ import annotations

import importlib

import numpy as np
import pytest


def inference():
    try:
        return importlib.import_module("experiments.pcrl_adaptive_release_v1.inference")
    except ModuleNotFoundError as error:
        pytest.fail(f"adaptive inference implementation missing: {error}")


def test_u_and_p_endpoint_families_have_frozen_signs_and_alias_collapse():
    i = inference()
    endpoints = i.enumerate_endpoints([
        {"id": "U-candidate", "arm": "U", "comparators": ["D17", "D17-alias"]},
        {"id": "P-candidate", "arm": "P", "comparators": ["D17"]},
    ], alias_of={"D17-alias": "D17"})
    assert len(endpoints) == 20
    assert {tuple(e["comparator_names"]) for e in endpoints if e["candidate"] == "U-candidate"} == {("D17", "D17-alias")}
    by = {(e["arm"], e["role"], e["weighting"]): e for e in endpoints}
    assert by["U", "utility:A/same_residence", "U"]["threshold"] == -.003
    assert by["P", "utility:A/same_residence", "PWGTP"]["threshold"] == .001
    target = by["P", "attack:AB/SEX", "U"]
    assert target["threshold"] == -.002
    assert target["plus"] == "D17" and target["minus"] == "P-candidate"
    assert by["U", "attack:AB/SEX", "U"]["threshold"] == .001
    assert by["P", "attack:A/RAC1P", "U"]["threshold"] == .001


def test_invalid_arm_or_duplicate_candidate_fails_before_assessment():
    i = inference()
    with pytest.raises(ValueError):
        i.enumerate_endpoints([{"id": "x", "arm": "unknown", "comparators": ["D17"]}])
    with pytest.raises(ValueError):
        i.enumerate_endpoints([{"id": "x", "arm": "U", "comparators": ["D17"]},
                               {"id": "x", "arm": "P", "comparators": ["D17"]}])


def test_paired_household_contrast_rejects_misaligned_people():
    i = inference()
    endpoint = i.enumerate_endpoints([{"id": "Q", "arm": "P", "comparators": ["D17"]}])[0]
    scores = {}
    for anchor in (0, 1, 2):
        for release in ("Q", "D17"):
            scores[(release, anchor, endpoint["role"])] = {
                "ids": np.array([1, 2]), "households": np.array(["h1", "h2"]),
                "weights": np.array([1., 3.]), "loss": np.array([.2, .4])}
    contrasts = i.contrasts_from_scores([endpoint], scores)
    assert len(contrasts) == 1 and len(contrasts[0]["anchors"]) == 3
    assert np.array_equal(contrasts[0]["anchors"][0]["household"], np.array(["h1", "h2"]))
    scores[("D17", 1, endpoint["role"])]["households"] = np.array(["h1", "wrong"])
    with pytest.raises(ValueError):
        i.contrasts_from_scores([endpoint], scores)


def test_family_manifest_counts_actual_endpoints_not_guessed_divisor():
    i = inference()
    manifest = i.family_manifest([{"id": "U", "arm": "U", "comparators": ["D17", "MILP"]},
                                  {"id": "P", "arm": "P", "comparators": ["D17"]}])
    assert manifest["n_endpoints"] == 30
    assert len({e["id"] for e in manifest["endpoints"]}) == 30
