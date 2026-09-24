"""Synthetic tests for the PPAN-style ADV competitor (no ACS rows)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.pcrl_shared_context_release_v1 import adv, rd

_spec = importlib.util.spec_from_file_location("_sc_rd_fixture", Path(__file__).with_name("test_rd.py"))
fx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fx)


@pytest.fixture
def synthetic(monkeypatch):
    monkeypatch.setattr(rd, "attack_losses", fx.fake_attack_losses)
    monkeypatch.setattr(rd, "fit_decoder_on_law", fx.fake_fit_decoder)
    monkeypatch.setattr(rd, "fit_best_response", fx.fake_best_response)
    role_dict = fx.make_roles(n=300)
    d17 = fx.d17_map()
    rd.attach_round0_laws(role_dict, d17, 0.5 * d17 + 0.5 / 17)
    return role_dict, d17, fx.make_bank(role_dict, d17)


def test_expected_ce_is_exact_token_expectation():
    torch.manual_seed(0)
    head = torch.nn.Linear(3 + 17, 4)
    h = torch.randn(6, 3)
    q = torch.softmax(torch.randn(6, 17), 1)
    y = torch.tensor([0, 1, 2, 3, 0, 1])
    w = torch.rand(6) + 0.5
    mask = torch.tensor([True, True, False, True, True, True])
    got = float(adv.expected_ce(head, h, q, y, w, mask))
    total, wsum = 0.0, 0.0
    for i in range(6):
        if not mask[i]:
            continue
        loss = 0.0
        for z in range(17):
            logits = head(torch.cat([h[i], torch.nn.functional.one_hot(torch.tensor(z), 17).float()]))
            loss += float(q[i, z]) * float(-torch.log_softmax(logits, 0)[y[i]])
        total += float(w[i]) * loss
        wsum += float(w[i])
    assert got == pytest.approx(total / wsum, rel=1e-5)


def test_pruned_law_rows_are_simplex_with_bounded_support():
    p = np.array([[0.9, 0.0995, 0.0005] + [0.0] * 14, [1 / 17] * 17])
    q = adv.pruned_law(p)
    np.testing.assert_allclose(q.sum(1), 1.0, atol=1e-15)
    assert q[0, 2] == 0.0 and (q[1] > 0).all()


def test_adv_end_to_end_selection_and_pinned_law(synthetic, tmp_path):
    role_dict, d17, bank = synthetic
    out = tmp_path / "adv"
    sel = adv.run_adv(0, 2.0, role_dict, d17, bank, out, max_epochs=3, min_epochs=3, shortlist=2,
                      log=lambda *_: None)
    feasible = [s for s in sel["shortlist"] if s["final_bank"]["feasible"]] + [sel["witness"]]
    assert sel["witness"]["final_bank"]["feasible"]
    best = min(f["inner_selection_task"] for f in feasible)
    if sel["witness_selected"]:
        assert sel["selected_epoch"] is None and sel["witness"]["inner_selection_task"] == best
    else:
        chosen = next(s for s in sel["shortlist"] if s["epoch"] == sel["selected_epoch"])
        assert chosen["final_bank"]["feasible"] and sel["flag"] is None
        assert chosen["inner_selection_task"] == best
    # final bank = round-0 + one best-response set per shortlisted checkpoint
    assert sel["final_bank_cut_count"] == len(bank.cut_meta) + 2 * 8
    assert sel["timing_seconds"]["per_epoch_mean"] > 0
    law = adv.load_law(out)
    inputs = rd.legal_inputs(role_dict["inner_check"])
    q = law(inputs)
    assert q.shape == (len(inputs["x"]), 17)
    np.testing.assert_allclose(q.sum(1), 1.0, atol=1e-12)
    assert np.all((q == 0) | (q >= adv.PRUNE * 0.99))
    with pytest.raises((ValueError, PermissionError)):
        law({**inputs, "hb": role_dict["inner_check"]["hb"]})
    pinned = out / (sel["scaler"]["path"] if sel["witness_selected"] else sel["encoder"]["path"])
    pinned.write_bytes(pinned.read_bytes() + b"x")
    with pytest.raises(ValueError, match="hash"):
        adv.load_law(out)


def test_adv_is_deterministic_under_fixed_seed(synthetic, tmp_path):
    role_dict, d17, bank = synthetic
    laws = []
    for k in range(2):
        adv.run_adv(1, 0.5, role_dict, d17, bank, tmp_path / f"r{k}", max_epochs=2, min_epochs=2,
                    shortlist=1, log=lambda *_: None)
        laws.append(adv.load_law(tmp_path / f"r{k}")(rd.legal_inputs(role_dict["inner_check"])))
    np.testing.assert_array_equal(laws[0], laws[1])


def _plugin_ce(law, s, k):
    """In-sample plug-in CE of S given the released token (higher = less leakage)."""
    table = np.stack([(law * (s == c)[:, None]).sum(0) for c in range(k)], 1) + 1e-3
    table /= table.sum(1, keepdims=True)
    return float(-(law * np.log(table[:, s].T)).sum(1).mean())


def test_adversarial_pressure_reduces_sex_leakage(monkeypatch, tmp_path):
    """Task signal x0 also predicts SEX: the beta=2 shortlisted law leaks less SEX than beta=0's."""
    monkeypatch.setattr(rd, "attack_losses", fx.fake_attack_losses)
    monkeypatch.setattr(rd, "fit_best_response", fx.fake_best_response)
    role_dict = fx.make_roles(n=800, seed=3)
    rng = np.random.default_rng(4)
    for rows in role_dict.values():
        x0 = rows["x"][:, 0].astype(float)
        rows["labels"]["SEX"] = (x0 + 0.3 * rng.normal(size=len(x0)) > 0).astype(np.int64)
    d17 = fx.d17_map()
    rd.attach_round0_laws(role_dict, d17, 0.5 * d17 + 0.5 / 17)
    bank = fx.make_bank(role_dict, d17)
    leak = {}
    rows = role_dict["inner_check"]
    for beta in (0.0, 2.0):
        out = tmp_path / f"b{beta}"
        sel = adv.run_adv(2, beta, role_dict, d17, bank, out, max_epochs=15, min_epochs=15,
                          shortlist=1, init="random", lr=1e-3, weight_decay=1e-4, log=lambda *_: None)
        e = sel["shortlist"][0]["epoch"]
        assert e > 0, "shortlist must move off the random init for the test to mean anything"
        enc, _, _ = adv.build_modules(adv.N_ENCODER_FEATURES, 0)
        enc.load_state_dict(torch.load(out / "checkpoints" / f"encoder_e{e:03d}.pt", weights_only=True))
        with np.load(out / "feature_scaler.npz") as s:
            law = adv.AdvLaw(enc, rd.Scaler(s["mean"], s["scale"]), s["d17_token_map"].copy())
        leak[beta] = _plugin_ce(law(rd.legal_inputs(rows)), rows["labels"]["SEX"], 2)
    assert leak[2.0] > leak[0.0]


def test_privacy_units_m5_rule(synthetic, tmp_path):
    role_dict, d17, bank = synthetic
    sel = adv.run_adv(0, 2.0, role_dict, d17, bank, tmp_path / "p", max_epochs=4, min_epochs=4, shortlist=2,
                      select="privacy", log=lambda *_: None)
    task = adv.run_adv(0, 2.0, role_dict, d17, bank, tmp_path / "t", max_epochs=1, min_epochs=1, shortlist=1,
                       log=lambda *_: None)
    assert sel["training"]["seed"] == task["training"]["seed"] + 500          # distinct seeds (M5)
    beta = 2.0
    hist = {h["epoch"]: h for h in sel["history"]}
    for s_ in sel["shortlist"]:                                                 # privacy key + task cap
        h = hist[s_["epoch"]]
        assert h["task_balanced"] <= sel["witness"]["inner_selection_task"] + 0.001
        adv_ce = h["adv_ce_inner_selection"]
        assert s_["selection_key"][1] == pytest.approx(h["task_balanced"] - beta * sum(adv_ce.values()) / 4)
    feas = [s_ for s_ in sel["shortlist"] if s_["final_bank"]["feasible"]] + [sel["witness"]]
    top = max(f["final_bank_ab_sex_slack"] for f in feas)                       # refit-attacker slack key
    assert sel["witness"]["final_bank_ab_sex_slack"] == pytest.approx(0.0, abs=1e-12)
    if sel["witness_selected"]:
        assert sel["witness"]["final_bank_ab_sex_slack"] == top
    else:
        chosen = next(s_ for s_ in sel["shortlist"] if s_["epoch"] == sel["selected_epoch"])
        assert chosen["final_bank_ab_sex_slack"] == top and top > 0
    if not sel["shortlist"]:
        assert sel["privacy_note"].startswith("NO_ELIGIBLE_EPOCH") and sel["flag"] == "WITNESS_FALLBACK"
    with pytest.raises(ValueError):
        adv.run_adv(0, 2.0, role_dict, d17, bank, tmp_path / "q", max_epochs=1, min_epochs=1, shortlist=1,
                    select="bogus", log=lambda *_: None)


def test_adv_witness_fallback_returns_exact_d17(synthetic, tmp_path, monkeypatch):
    """M4: no feasible checkpoint -> exact one-hot D17 law, flag WITNESS_FALLBACK."""
    role_dict, d17, bank = synthetic
    real_check = rd.PersonBank.check

    def reject_non_d17(self, law):
        out = real_check(self, law)
        if np.any(law != self.d17_law):
            out = {**out, "feasible": False, "max_violation": 0.01}
        return out
    monkeypatch.setattr(rd.PersonBank, "check", reject_non_d17)
    out = tmp_path / "w"
    sel = adv.run_adv(0, 0.5, role_dict, d17, bank, out, max_epochs=2, min_epochs=2, shortlist=2,
                      log=lambda *_: None)
    assert sel["flag"] == "WITNESS_FALLBACK" and sel["witness_selected"] and sel["selected_epoch"] is None
    rows = role_dict["inner_check"]
    q = adv.load_law(out)(rd.legal_inputs(rows))
    np.testing.assert_array_equal(q, rd.onehot(rd.d17_tokens(d17, rows["token_codes"])))
