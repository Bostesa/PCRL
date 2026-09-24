"""Synthetic tests for the RD_TASK / RD_PRIV competitors (no ACS rows)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.pcrl_adaptive_release_v1 import fit_a, reference
from experiments.pcrl_shared_context_release_v1 import policies, rd
from experiments.pcrl_task_aligned_cuts_v1 import solver

ROLE_NAMES = rd.SCIENCE_ROLES


# ---------------------------------------------------------------------------
# synthetic fixture (also loaded by test_adv.py)
# ---------------------------------------------------------------------------

def d17_map():
    q = np.zeros((32, 17))
    q[np.arange(32), np.arange(32) % 17] = 1.0
    return q


def make_rows(name, n, rng, y_signal=2.0):
    x = rng.normal(size=(n, 32))
    ha = rng.normal(size=(n, 4))
    hb = rng.normal(size=(n, 2))
    sex = (x[:, 1] + 0.5 * rng.normal(size=n) > 0).astype(np.int64)
    race = np.clip(((x[:, 2] + 2) * 2).astype(np.int64), 0, 8)
    y = (y_signal * x[:, 0] + 0.3 * ha[:, 0] + 0.5 * rng.normal(size=n) > 0).astype(np.int64)
    edges = np.quantile(x[:, 5], np.linspace(0, 1, 17)[1:-1])
    codes = (x[:, 4] > 0).astype(np.int64) * 16 + np.digitize(x[:, 5], edges)
    risk = np.abs(rng.normal(size=(n, 11)))
    risk[:, :2] /= risk[:, :2].sum(1, keepdims=True)
    risk[:, 2:] /= risk[:, 2:].sum(1, keepdims=True)
    return {"x": x.astype(np.float32), "ha": ha, "hb": hb,
            "weights": rng.uniform(0.5, 2.0, size=n),
            "ids": np.array([f"{name}_{i}" for i in range(n)]),
            "households": np.array([f"{name}_h{i // 2}" for i in range(n)]),
            "token_codes": codes.astype(np.int64),
            "teacher_p": 1 / (1 + np.exp(-x[:, 0])), "residual": 0.1 * x[:, 3], "risk": risk,
            "labels": {"same_residence": y, "SEX": sex, "RAC1P": race}}


def make_roles(n=500, seed=0):
    rng = np.random.default_rng(seed)
    return {name: make_rows(name, n, rng) for name in ROLE_NAMES}


class FakeDecoder:
    """Token-only frequency decoder (H ignored); predict_token_proba like TokenCandidate."""

    def __init__(self, p1):
        self.p1 = np.asarray(p1, dtype=float)

    @classmethod
    def fit(cls, law, y, w):
        law = np.asarray(law)
        num = (law * (w * y)[:, None]).sum(0) + 1
        den = (law * w[:, None]).sum(0) + 2
        return cls(num / den)

    def predict_token_proba(self, h, n_tokens=17):
        p = np.broadcast_to(self.p1, (len(h), n_tokens))
        return np.stack([1 - p, p], axis=2).copy()


def fake_fit_decoder(fit_rows, fit_law, sel_rows, sel_law, out_dir, seed):
    m = rd.valid_mask(fit_rows, "same_residence")
    dec = FakeDecoder.fit(np.asarray(fit_law)[m], fit_rows["labels"]["same_residence"][m],
                          np.asarray(fit_rows["weights"])[m])
    return dec, {"fake": True, "seed": seed}


def fake_attack_losses(spec, rows):
    s = np.asarray(rows["labels"][spec["target"]])
    valid = s >= 0
    s = s[valid]
    if spec.get("table") is None:
        prior = np.asarray(spec["prior"])
        return valid, np.repeat(-np.log(prior[s])[:, None], 17, 1)
    table = np.asarray(spec["table"])  # (17, k)
    return valid, -np.log(table[:, s].T)


def table_attack(law, s, k):
    num = np.stack([(law * (s == c)[:, None]).sum(0) for c in range(k)], 1) + 0.5
    return num / num.sum(1, keepdims=True)


def fake_best_response(role_dict, law_fn, out_dir, seed, tag):
    aud = role_dict["audit_fit"]
    law = law_fn(rd.legal_inputs(aud))
    specs, cuts = {}, []
    for role in rd.PROTECTED_ROLES:
        target = role.split("/")[1]
        aid = f"{role}/from_{role.split('/')[0]}/{tag}/table"
        specs[aid] = {"id": aid, "role": role, "target": target,
                      "table": table_attack(law, aud["labels"][target], rd.CLASS_COUNT[target])}
        cuts += [{"id": f"{aid}/{w}", "attack_id": aid, "role": role, "weighting": w} for w in "UW"]
    return specs, cuts


def make_bank(role_dict, d17, multipliers=None):
    aud = role_dict["audit_fit"]
    d17_law = rd.onehot(rd.d17_tokens(d17, aud["token_codes"]))
    specs, cuts = {}, []
    for role in rd.PROTECTED_ROLES:
        target = role.split("/")[1]
        k = rd.CLASS_COUNT[target]
        s = aud["labels"][target]
        prior = (np.bincount(s, minlength=k) + 0.5) / (len(s) + 0.5 * k)
        for name, spec in (("H", {"table": None, "prior": prior}),
                           ("D17", {"table": table_attack(d17_law, s, k)})):
            aid = f"{role}/from_A/{name}/fake"
            specs[aid] = {"id": aid, "role": role, "target": target, **spec}
            cuts += [{"id": f"{aid}/{w}", "attack_id": aid, "role": role, "weighting": w} for w in "UW"]
    ntr = role_dict["nuisance_train"]
    dec = FakeDecoder.fit(role_dict["_round0_law_ntr"], ntr["labels"]["same_residence"], ntr["weights"])
    return rd.Round0Bank(dec, {"fake": True}, specs, cuts, multipliers, [], {"layout": "synthetic"})


@pytest.fixture
def synthetic(monkeypatch):
    monkeypatch.setattr(rd, "attack_losses", fake_attack_losses)
    monkeypatch.setattr(rd, "fit_decoder_on_law", fake_fit_decoder)
    monkeypatch.setattr(rd, "fit_best_response", fake_best_response)
    role_dict = make_roles()
    d17 = d17_map()
    q_hist = 0.5 * d17 + 0.5 / 17
    rd.attach_round0_laws(role_dict, d17, q_hist)
    return role_dict, d17, make_bank(role_dict, d17)


# ---------------------------------------------------------------------------
# unit tests
# ---------------------------------------------------------------------------

def test_person_bank_matches_ar_t32_calibration(synthetic):
    role_dict, d17, bank = synthetic
    coef = role_dict["coefficient_split"]
    pb = rd.PersonBank(coef, d17)
    pb.add(bank.specs, bank.cut_meta, "round0")
    rho = pb.calibrate()
    ar_cuts = []
    for cut in pb.cuts:
        valid, losses = fake_attack_losses(bank.specs[cut["attack_id"]], coef)
        pair = fit_a.aggregate_loss_coefficients(coef["token_codes"][valid], losses, coef["weights"][valid], 32)
        ar_cuts.append({"id": cut["id"], "role": cut["role"], "weighting": cut["weighting"],
                        "coeff": pair[cut["weighting"]]})
    cal = reference.calibrate_reference(d17, ar_cuts, rd.DELTA, require_provenance=False)
    for group, value in cal["rho"].items():
        assert rho[group] == pytest.approx(value["value"], abs=1e-12)
    q = 0.6 * d17 + 0.4 / 17  # any T32 law: person replay == AR T32 replay
    ar = solver.replay_p1(q, np.zeros((32, 17)), cal["cuts"])
    ours = pb.check(q[coef["token_codes"]])
    assert max(0.0, ours["max_violation"]) == pytest.approx(ar["maximum_cut_violation"], abs=1e-10)
    values = pb.cut_values(q[coef["token_codes"]])
    by_id = {c["id"]: float(np.sum(c["coeff"] * q)) for c in cal["cuts"]}
    np.testing.assert_allclose(values, [by_id[c["id"]] for c in pb.cuts], atol=1e-12)
    assert pb.check(pb.d17_law)["max_violation"] == pytest.approx(-rd.DELTA, abs=1e-12)


def test_price_targets_match_policies_convention(synthetic):
    role_dict, d17, bank = synthetic
    ntr = role_dict["nuisance_train"]
    mult = {c["id"]: (0.3 if i % 3 == 0 else 0.0) for i, c in enumerate(bank.cut_meta)}
    lam, record = rd.price_direction(bank.cut_meta, mult)
    groups = policies.price_groups(bank.cut_meta, mult)["all_priced_x2"]["multipliers"]
    assert lam == pytest.approx({k: v / 2 for k, v in groups.items()})
    attacks = {c["attack_id"]: fake_attack_losses(bank.specs[c["attack_id"]], ntr) for c in bank.cut_meta}
    ours = rd.person_price_targets(len(ntr["weights"]), attacks, lam, bank.cut_meta)
    rows = [{"id": c["id"], "valid_mask": attacks[c["attack_id"]][0], "losses": attacks[c["attack_id"]][1]}
            for c in bank.cut_meta if c["id"] in lam]
    theirs = policies.priced_person_costs(np.zeros((len(ntr["weights"]), 17)), rows, lam,
                                          np.ones(len(ntr["weights"]), bool))
    np.testing.assert_allclose(ours, -theirs, atol=1e-12)
    # all-zero duals -> registered fallback 1/(#cuts)
    lam0, rec0 = rd.price_direction(bank.cut_meta, {c["id"]: 0.0 for c in bank.cut_meta})
    assert rec0["fallback_used"] and set(lam0.values()) == {1 / len(bank.cut_meta)}


def test_switched_policy_ties_and_tau_go_to_d17():
    base = np.array([0, 1, 2])
    delta = np.zeros((3, 17))       # paired: D17 column is 0
    delta[1, 5] = -0.001            # gain below tau -> D17
    delta[2, 5] = -0.01             # gain above tau -> token 5
    assert rd.switch_tokens(delta, base, 0.002).tolist() == [0, 1, 5]
    tie = np.zeros((1, 17))
    tie[0, 3] = tie[0, 9] = -0.5    # tie among non-D17 tokens -> lowest id
    assert rd.switch_tokens(tie, np.array([0]), 0.0).tolist() == [3]
    assert rd.switch_tokens(np.zeros((1, 17)), np.array([9]), 0.0).tolist() == [9]  # all-zero -> D17


def test_mu_paired_oracle_is_linear_and_d17_column_zero():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(200, 3))
    base = rng.integers(0, 17, size=200)
    u = rng.normal(size=(200, 17)) + z[:, :1]
    p = rng.normal(size=(200, 17)) - z[:, 1:2]
    hh = np.array([f"h{i}" for i in range(200)])
    ou, _ = rd.paired_oracle(z, u, base, hh, 0, np.ones(200))
    op, _ = rd.paired_oracle(z, p, base, hh, 1, np.ones(200))
    for mu in (0.0, 0.7, 3.0):
        o = rd.MuPairedOracle(ou.models, op.models, mu)
        d = o.predict_delta(z, base)
        np.testing.assert_allclose(d, ou.predict_delta(z, base) - mu * op.predict_delta(z, base))
        assert np.all(d[np.arange(200), base] == 0)
        assert isinstance(o, policies.PairedOracle)


def test_bisection_finds_smallest_feasible_mu():
    def feasible(law):
        return {"feasible": law >= 3.7, "max_violation": 3.7 - law}
    mu, trace, status = rd.bisect_mu(lambda m: m, feasible)
    assert status == "bisection" and 3.7 <= mu < 3.7 * 1.01
    assert len(trace) <= 1 + 3 + rd.BISECTION_STEPS
    mu0, _, s0 = rd.bisect_mu(lambda m: m, lambda law: {"feasible": True, "max_violation": -1})
    assert mu0 == 0.0 and s0 == "mu0_feasible"
    none, _, s1 = rd.bisect_mu(lambda m: m, lambda law: {"feasible": False, "max_violation": 1})
    assert none is None and s1 == "no_feasible_mu_in_bracket"


def test_crossfit_folds_are_household_grouped():
    hh = np.array([f"h{i // 3}" for i in range(300)])
    folds = rd.household_fold(hh, "sc_crossfit|", 2)
    for h in np.unique(hh):
        assert len(set(folds[hh == h])) == 1
    assert 0.3 < folds.mean() < 0.7


def test_rd_task_learns_task_signal_and_pins_law(synthetic, tmp_path):
    role_dict, d17, bank = synthetic
    out = tmp_path / "rd_task"
    sel = rd.run_rd_task(0, role_dict, d17, bank, tmp_path / "nobank", out, rounds=1, log=lambda *_: None)
    assert (out / "SELECTED.json").exists()
    best = next(c for c in sel["candidates"] if c["policy"] == sel["policy"])
    assert len(sel["candidates"]) == 2 * len(rd.TAU_GRID)
    assert best["inner_selection_task"]["balanced"] == min(c["inner_selection_task"]["balanced"] for c in sel["candidates"])
    # the task signal is in x, not in T32: the competitor must find it
    assert best["inner_selection_task_minus_D17_receiver"] < -0.05
    assert sel["preflight_M1_4"]["selected_minus_D17_balanced"] < -0.05
    law = rd.load_law(out)
    inputs = rd.legal_inputs(role_dict["inner_check"])
    q = law(inputs)
    assert q.shape == (len(inputs["x"]), 17) and np.array_equal(q.sum(1), np.ones(len(q)))
    assert set(np.unique(q)) <= {0.0, 1.0}
    with pytest.raises((ValueError, PermissionError)):
        law({**inputs, "hb": role_dict["inner_check"]["hb"]})
    policy_path = out / sel["policy"]["path"]
    policy_path.write_bytes(policy_path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="hash"):
        rd.load_law(out)


def test_rd_priv_selects_final_bank_feasible_round(synthetic, tmp_path):
    role_dict, d17, bank = synthetic
    out = tmp_path / "rd_priv"
    sel = rd.run_rd_priv(0, role_dict, d17, bank, out, rounds=1, log=lambda *_: None)
    assert sel["dual_source"].startswith("recomputed")
    assert sel["witness"]["final_bank_check"]["feasible"]
    if sel["selected_round"] is None:
        assert sel["witness_selected"] and sel["flag"] in ("WITNESS_FALLBACK", "WITNESS_SELECTED_BY_RULE")
        if sel["flag"] == "WITNESS_FALLBACK":
            assert all(not r["final_bank_check"]["feasible"] for r in sel["rounds"])
    else:
        chosen = sel["rounds"][sel["selected_round"]]
        assert chosen["final_bank_check"]["feasible"] and sel["flag"] is None
    for rec in sel["rounds"]:
        # bisection result is feasible on the bank it was bisected against
        if "infeasible" not in rec["bisection_status"]:
            assert rec["check_on_bank_used_for_bisection"]["feasible"]
    # final bank contains the closing best response on the last round
    assert sel["final_bank_cut_count"] == len(bank.cut_meta) + 2 * 8
    q = rd.load_law(out)(rd.legal_inputs(role_dict["inner_check"]))
    assert set(np.unique(q)) <= {0.0, 1.0}
    saved = json.loads((out / "SELECTED.json").read_text())
    assert saved["outer_labels_accessed"] is False


def test_rd_priv_tau_fallback_and_d17_alias(synthetic, tmp_path, monkeypatch):
    """If no mu is feasible, tau bisection shrinks toward D17; if even the round
    policies fail their own best responses, RD_PRIV is D17 (flagged)."""
    role_dict, d17, bank = synthetic
    real_check = rd.PersonBank.check

    def strict_check(self, law):
        out = real_check(self, law)
        changed = float((np.argmax(law, 1) != np.argmax(self.d17_law, 1)).mean())
        # pretend any deviation from D17 on > 1% of people violates a cut
        if changed > 0.01:
            out = {**out, "feasible": False, "max_violation": changed}
        return out
    monkeypatch.setattr(rd.PersonBank, "check", strict_check)
    sel = rd.run_rd_priv(0, role_dict, d17, bank, tmp_path / "p", rounds=0, log=lambda *_: None)
    rec = sel["rounds"][0]
    assert "fallback" in rec["bisection_status"] and rec["tau"] > rd.TAU_GRID[0]
    assert rec["check_on_bank_used_for_bisection"]["feasible"]
    q = rd.load_law(tmp_path / "p")(rd.legal_inputs(role_dict["inner_check"]))
    changed = (np.argmax(q, 1) != rd.d17_tokens(d17, role_dict["inner_check"]["token_codes"])).mean()
    if sel["selected_round"] is None:
        assert changed == 0.0
    else:
        assert changed <= 0.05


def test_rd_priv_witness_fallback_is_exact_d17(synthetic, tmp_path, monkeypatch):
    """M4: if no round survives its own refits, RD_PRIV is exactly D17 (WITNESS_FALLBACK)."""
    role_dict, d17, bank = synthetic
    real_check = rd.PersonBank.check

    def reject_non_d17(self, law):
        out = real_check(self, law)
        if np.any(np.argmax(law, 1) != np.argmax(self.d17_law, 1)) and any(
                c["origin"].startswith("BR_") for c in self.cuts):
            out = {**out, "feasible": False, "max_violation": 0.01}
        return out
    monkeypatch.setattr(rd.PersonBank, "check", reject_non_d17)
    sel = rd.run_rd_priv(0, role_dict, d17, bank, tmp_path / "w", rounds=1, log=lambda *_: None)
    any_feasible = any(r["final_bank_check"]["feasible"] for r in sel["rounds"])
    assert all(r["final_bank_check"]["feasible"] == (r["census_coefficient_split"]["fraction_changed_vs_D17"] == 0)
               for r in sel["rounds"])
    if not any_feasible:
        assert sel["flag"] == "WITNESS_FALLBACK" and sel["selected_round"] is None
    elif sel["witness_selected"]:
        assert sel["flag"] == "WITNESS_SELECTED_BY_RULE"
    q = rd.load_law(tmp_path / "w")(rd.legal_inputs(role_dict["inner_check"]))
    if sel["witness_selected"]:
        np.testing.assert_array_equal(q, rd.onehot(rd.d17_tokens(d17, role_dict["inner_check"]["token_codes"])))
