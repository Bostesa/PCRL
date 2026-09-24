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
