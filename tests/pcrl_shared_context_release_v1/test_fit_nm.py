"""Synthetic end-to-end build-bank -> NM/T32 units -> DET_SEL -> release (no ACS data)."""
import json

import numpy as np
import pytest

from experiments.pcrl_adaptive_release_v1.roles import role_of
from experiments.pcrl_shared_context_release_v1 import channel, fit_nm, policies, release

SIZES = {"nuisance_train": 160, "audit_fit": 90, "coefficient_split": 120,
         "inner_selection": 70, "inner_check": 50}


def _households(role, count, start=0):
    out, i = [], start
    while len(out) < count:
        h = f"synthetic-hh-{i}"
        if role_of(h) == role:
            out.append(h)
        i += 1
    return out


def synthetic_roles(seed=0):
    rng = np.random.default_rng(seed)
    d17 = np.zeros((32, 17)); d17[np.arange(32), np.arange(32) % 5] = 1.
    hist = 0.7 * d17 + 0.3 / 17
    role_dict = {}
    for r_index, (role, n) in enumerate(SIZES.items()):
        hh = np.repeat(_households(role, (n + 1) // 2), 2)[:n]
        x = rng.normal(size=(n, 32)).astype(np.float32)
        ha = rng.normal(size=(n, 4))
        sex = (x[:, 0] + 0.5 * rng.normal(size=n) > 0).astype(np.int64)
        residence = (ha[:, 0] + x[:, 1] + 0.3 * rng.normal(size=n) > 0).astype(np.int64)
        role_dict[role] = {
            "x": x, "ha": ha, "hb": np.column_stack((sex + 0.3 * rng.normal(size=n), rng.normal(size=n))),
            "weights": rng.uniform(5, 40, n), "ids": np.asarray([f"{role}-{i}" for i in range(n)]),
            "households": hh, "token_codes": rng.integers(0, 32, n).astype(np.int64),
            "teacher_p": 1 / (1 + np.exp(-ha[:, 0])), "residual": rng.normal(size=n),
            "risk": rng.dirichlet(np.ones(11), n),
            "labels": {"same_residence": residence, "SEX": sex,
                       "RAC1P": rng.integers(0, 9, n).astype(np.int64)}}
    return role_dict, d17, hist


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("sc") / "private"
    role_dict, d17, hist = synthetic_roles()
    bank = fit_nm.build_bank(0, role_dict, d17, hist, "0" * 64, root / "bank_a0",
                             target_roles=("A/SEX", "AB/SEX"))
    return {"root": root, "roles": role_dict, "d17": d17, "hist": hist, "bank": bank}


def test_bank_is_frozen_and_complete(built):
    bank = built["bank"]
    manifest = bank["manifest"]
    assert manifest["status"] == "COMPLETE" and manifest["outer_labels_accessed"] is False
    assert bank["retained_policies"][0] == "D17"
    pol = json.loads((bank["root"] / "policies" / "POLICIES.json").read_text())
    assert set(pol["fits"]) == set(policies.POLICY_NAMES[1:])
    assert pol["crossfit"]["fold_people"][0] + pol["crossfit"]["fold_people"][1] == SIZES["nuisance_train"]
    ctx = json.loads((bank["root"] / "contexts" / "CONTEXTS.json").read_text())
    assert set(ctx["census_coefficient_split"]) == {"1", "2", "4"}
    # resume verifies the inventory instead of refitting
    again = fit_nm.build_bank(0, built["roles"], built["d17"], built["hist"], "0" * 64,
                              bank["root"], target_roles=("A/SEX", "AB/SEX"))
    assert again["manifest_sha256"] == bank["manifest_sha256"]
    # the D17 column reproduces the historical map on every role
    for rows in built["roles"].values():
        tokens = bank["policy_bank"].predict(policies.legal_inputs(rows))
        assert np.array_equal(tokens[:, 0], np.argmax(built["d17"][rows["token_codes"]], 1))


def test_t32_control_reproduces_bank_round0_lp(built):
    out = built["root"] / "T32_U"
    receipt = fit_nm.run_unit("T32_U", 0, built["roles"], built["d17"], built["hist"],
                              built["bank"]["root"], out, rounds=0)
    record = json.loads((out / "round_r00" / "ROUND.json").read_text())
    assert record["t32_round0_max_abs_difference_vs_bank_lp"] == 0.
    assert receipt["rounds_summary"][0]["eta"] == 0.
    blocks = json.loads((out / "blocks_r00.json").read_text())
    assert blocks["ar_b_block_max_abs_difference"] <= 1e-12


def test_nm_unit_alternation_release_and_det_sel(built):
    out = built["root"] / "NM4_U"
    receipt = fit_nm.run_unit("NM4_U", 0, built["roles"], built["d17"], built["hist"],
                              built["bank"]["root"], out, rounds=1, nm4_k=4)
    assert receipt["status"] == "COMPLETE" and len(receipt["rounds_summary"]) == 2
    first = (out / "COMPLETE.json").read_bytes()
    again = fit_nm.run_unit("NM4_U", 0, built["roles"], built["d17"], built["hist"],
                            built["bank"]["root"], out, rounds=1, nm4_k=4)
    assert (out / "COMPLETE.json").read_bytes() == first and again["selected_round"] == receipt["selected_round"]
    closing = json.loads((out / "closing" / "CLOSING.json").read_text())
    assert closing["round_index"] == 2 and closing["new_attack_count"] > 0
    selection = json.loads((out / "FINAL_BANK_SELECTION.json").read_text())
    assert selection["closing_refit"]["bank_sha256"] == closing["bank_sha256"]
    assert receipt["closing_refit"]["cut_count"] == closing["cut_count"]
    rec = json.loads((out / "round_r01" / "ROUND.json").read_text())
    assert closing["cut_count"] > rec["cut_count"]
    assert rec["solution"]["replay"]["maximum_cut_violation"] <= 1e-7
    assert rec["cut_count"] > json.loads((out / "round_r00" / "ROUND.json").read_text())["cut_count"]
    gate = rec["nonalias"]["coefficient_split"]
    assert {"tv_to_d17", "within_t32", "deterministic_emission"} <= set(gate)
    assert "task" in rec["within_t32_projection_rescore"]
    # released law replays the stored parameters with legal inputs only
    law = release.load_law(out)
    from experiments.pcrl_shared_context_release_v1 import laws
    dispatched = laws.load(out)
    assert dispatched.kind == "nested"
    rows = built["roles"]["coefficient_split"]
    legal = policies.legal_inputs(rows)
    q = law(legal)
    channel.validate_law(q)
    assert np.array_equal(dispatched(legal), q)
    spec = json.loads((out / "RELEASE_SPEC.json").read_text())
    with np.load(out / spec["params_relative"]) as p:
        bank = built["bank"]
        direct = channel.person_law(p["B"], p["A"], float(p["eta"]), rows["token_codes"],
                                    bank["rules"][4].assign(legal), bank["policy_bank"].predict(legal))
    assert np.array_equal(q, direct)
    with pytest.raises(PermissionError):
        law({**legal, "hb": rows["hb"]})
    session = release.SharedContextRelease(law, replay_key=b"k" * 32, release_id="test")
    wire = session.emit(legal, rows["ids"])
    assert set(wire) == {"h_a", "token"} and wire["token"].shape == (len(q),)
    assert np.all(q[np.arange(len(q)), wire["token"]] > 0)
    assert np.array_equal(session.emit(legal, rows["ids"])["token"], wire["token"])
    fresh = release.SharedContextRelease(law, replay_key=b"k" * 32, release_id="test")
    assert np.array_equal(fresh.emit(legal, rows["ids"])["token"], wire["token"])
    # exhaustive deterministic selector from the NM4_U unit
    det = fit_nm.run_det_sel("DET_SEL4", out, built["root"] / "DET_SEL4",
                             role_dict=built["roles"], q_ref=built["d17"])
    record = json.loads((built["root"] / "DET_SEL4" / "DET_SEL.json").read_text())
    M = len(receipt["policy_columns"])
    assert record["bank_source"] == "closing"
    assert record["assignments_evaluated"] == M ** 4 and record["feasible_assignments"] >= 1
    assert record["nonalias_coefficient_split"]["deterministic_emission"]["all_rows_one_hot"]
    det_law = release.load_law(built["root"] / "DET_SEL4")
    assert laws.load(built["root"] / "DET_SEL4").kind == "nested"
    assert channel.deterministic_emission(det_law(legal))["all_rows_one_hot"]
    assert det["status"] == "COMPLETE"


def test_privacy_first_variant_runs(built):
    out = built["root"] / "NM1_P"
    receipt = fit_nm.run_unit("NM1_P", 0, built["roles"], built["d17"], built["hist"],
                              built["bank"]["root"], out, rounds=0)
    tau = receipt["rounds_summary"][0]["tau"]
    assert tau is not None and tau >= 0


def test_unit_refuses_nonprivate_or_undecided_k(built):
    from pathlib import Path
    with pytest.raises(ValueError):
        fit_nm.run_unit("NM4_U", 0, built["roles"], built["d17"], built["hist"],
                        built["bank"]["root"], built["root"] / "x", rounds=0)
    with pytest.raises(ValueError):
        fit_nm.run_unit("NM1_U", 0, built["roles"], built["d17"], built["hist"],
                        built["bank"]["root"], Path("public_unit_never_created"), rounds=0)
    assert not Path("public_unit_never_created").exists()
