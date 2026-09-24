from experiments.pcrl_adaptive_release_v1 import selection


def _scores():
    values = {}
    for anchor in (0, 1, 2):
        releases = {}
        for name in ("H", "D17", "strong_task", "private", "harmful"):
            rows = {}
            for role in selection.ROLES:
                rows[role] = {}
                for weighting in selection.WEIGHTINGS:
                    if role == selection.TASK_ROLE:
                        base = .55 if name == "H" else .50
                        offset = {"strong_task": -.004, "private": .0005,
                                  "harmful": -.02}.get(name, 0.)
                    else:
                        base = .60
                        offset = {"strong_task": -.0005, "private": .0005,
                                  "harmful": -.01}.get(name, 0.)
                        if role == selection.P_TARGET and name == "private":
                            offset = .003
                    rows[role][weighting] = base + offset
            releases[name] = rows
        values[anchor] = releases
    return values


def test_registered_signs_routes_and_family_fallback():
    scores = _scores()
    u = selection.aggregate_differences(scores, "strong_task")
    assert u["task"]["U"] < 0
    assert all(value > 0 for role in u["sensitive"].values()
               for value in role.values())
    assert selection.point_eligible(u, "U")
    p = selection.aggregate_differences(scores, "private")
    assert p["sensitive"][selection.P_TARGET]["U"] < -.002
    assert selection.point_eligible(p, "P")
    chosen = selection.select_family_representative(
        scores, "simple", ["harmful"], "U")
    assert chosen["selected"] == "D17"
    assert chosen["D17_fallback_included"]


def test_candidate_selection_uses_three_anchors_and_H_capability():
    scores = _scores()
    assert selection.select_route_candidate(scores, ["strong_task"], "U")["status"] == "NOMINATED_POINT_SCREEN"
    for anchor in (0, 1, 2):
        scores[anchor]["H"][selection.TASK_ROLE] = {"U": .501, "PWGTP": .501}
    selected = selection.select_route_candidate(scores, ["strong_task"], "U")
    assert selected["status"] == "DIAGNOSTIC_ONLY"
    assert not selected["points"][0]["capability_point_eligible"]


def test_recovery_guard_prevents_large_sensitive_harm():
    scores = _scores()
    for anchor in (0, 1, 2):
        scores[anchor]["strong_task"]["attack:A/SEX"] = {
            "U": .58, "PWGTP": .58}
    difference = selection.aggregate_differences(scores, "strong_task")
    assert difference["sensitive"]["attack:A/SEX"]["U"] > .001
    assert not selection.point_eligible(difference, "U")


def test_validation_extraction_never_uses_inner_check_outcome():
    roles = {}
    for role in selection.ROLES:
        roles[role] = {
            "selected_candidate": "own/logistic",
            "candidate_validation_scores": {
                "own/logistic": {"U": .4, "PWGTP": .41}},
            "H_selected_candidate": "H/logistic",
            "H_validation_scores": {
                "H/logistic": {"U": .5, "PWGTP": .51}},
            "candidate": {"U": -999., "PWGTP": -999.},
        }
    report = {"schema": "pcrl-adaptive-inner-audit-v1",
              "selection_role": "inner_selection", "score_role": "inner_check",
              "outer_pool_opened": False, "releases": {"candidate": {"roles": roles}}}
    extracted = selection.inner_validation_from_report(report)
    assert extracted["candidate"][selection.TASK_ROLE]["U"] == .4
    assert extracted["H"][selection.TASK_ROLE]["U"] == .5


def test_named_alias_expansion_restores_d17_and_preserves_exact_scores():
    canonical = {"A_selected": {"task": {"U": .4}},
                 "A_control_D17": {"task": {"U": .5}},
                 "H": {"task": {"U": .6}}}
    aliases = {"A_selected": "A_selected", "B_selected": "A_selected",
               "A_control_D17": "A_control_D17"}
    named = selection.expand_named_scores(canonical, aliases)
    assert named["D17"] == canonical["A_control_D17"]
    assert named["B_selected"] == canonical["A_selected"]
    assert named["H"] == canonical["H"]
    assert set(named) >= {"D17", "A_selected", "B_selected", "H"}


def test_named_alias_expansion_fails_closed_on_missing_or_conflicting_routes():
    import pytest

    canonical = {"A_selected": {"task": {"U": .4}},
                 "H": {"task": {"U": .6}}}
    with pytest.raises(ValueError, match="D17"):
        selection.expand_named_scores(canonical, {"A_selected": "A_selected"})
    with pytest.raises(ValueError, match="absent"):
        selection.expand_named_scores(canonical, {"A_control_D17": "missing"})
    with pytest.raises(ValueError, match="conflicting"):
        selection.expand_named_scores(
            {**canonical, "A_control_D17": {"task": {"U": .5}}},
            {"A_control_D17": "A_selected"})
