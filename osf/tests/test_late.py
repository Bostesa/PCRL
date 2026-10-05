"""Inference and selection end-to-end on synthetic records (nothing touches real units or the results folder)."""
import json

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from osf import data as DA
from osf import eval_lock as EL
from osf import family as FAM
from osf import infer as INF
from osf import run as R
from osf import select as SEL
from osf import train as T
from rgj import finalize as FN
from smf.tests.test_late import fake_preds


def test_family_sizes_and_z():
    assert FAM.PRIMARY_SIZE == 27 and round(FAM.Z_PRIMARY, 6) == 3.113017
    ids = [e["id"] for e in FAM.PRIMARY]
    assert ids == [f"P{i:02d}" for i in range(1, 28)]
    assert [e["alias_of"] for e in FAM.PRIMARY if e["claim"] == "B" and e.get("alias_of")] == \
        ["P04", "P05", "P06", "P07", "P08", "P09"]
    assert all(e.get("alias_of") is None for e in FAM.PRIMARY if e["claim"] == "C")


def test_conjunction_requires_valid_nominees():
    ok = {x: {"status": "NOMINEE", "config": c} for x, c in (("N*", "NORM-J|r3|a1"), ("L*", "RAW-L|b0.3"),
                                                              ("C*", "RAW-J|b0.3"), ("R*", "RAW-J|b0.3"))}
    for claim in "ABC":
        ids = [e["id"] for e in FAM.PRIMARY if e["claim"] == claim]
        assert FAM.claim_decision(claim, {i: "PASS" for i in ids}, ok)["decision"] == "PASS"
        assert FAM.claim_decision(claim, {**{i: "PASS" for i in ids}, ids[3]: "NOT_ESTABLISHED"}, ok)["decision"] == "NOT_ESTABLISHED"
    bad = {**ok, "N*": {"status": "NO_FEASIBLE_NOMINEE", "config": None, "descriptive_config": "NORM-J|r5|a1"}}
    ids = [e["id"] for e in FAM.PRIMARY if e["claim"] == "A"]
    assert FAM.claim_decision("A", {i: "PASS" for i in ids}, bad)["decision"] == "NOT_ESTABLISHED"
    assert FAM.overall_label({"A": {"decision": "PASS"}, "B": {"decision": "NOT_ESTABLISHED"},
                              "C": {"decision": "PASS"}}) == "RAW_JOINT_DEVELOPMENT_CRITERION_MET"


def _fake_preds(rng, n, sex, y_inc, y_occ, strength):
    p = fake_preds(rng, n, sex, y_inc, y_occ, strength)
    p["assess_unit"] = np.arange(n) // 2
    return p


def test_inference_end_to_end(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    n = 400
    sex = rng.integers(0, 2, n)
    y_inc, y_occ = rng.integers(0, 2, n), rng.integers(0, 6, n)
    monkeypatch.setattr(R, "UNITS", tmp_path / "units")
    monkeypatch.setattr(R, "RUN", tmp_path)
    monkeypatch.setattr(R, "PKG", tmp_path / "pkg")
    monkeypatch.setattr(FAM, "B", 60)
    labels = [T.config_id(c) for c in T.bank("full")] + ["E", "F", "F0"]
    strength = {lab: {"pair": (1.2 if lab == "RAW-L|b0.3" else 0.6 if lab == "RAW-J|b0.3" else 0.9), "v1": 0.5,
                      "v2": 0.5} for lab in labels}
    for k in (0, 1, 2):
        for lab in labels:
            p = _fake_preds(np.random.default_rng([k, labels.index(lab)]), n, sex, y_inc, y_occ, strength[lab])
            FN.save_unit(R.U(f"outer__s{k}__{INF.safe(lab)}"), {"preds.npz": lambda q, p=p: np.savez(q, **p)}, {"x": 1})
    D = DA.load()
    st = {"N*": {"status": "NO_FEASIBLE_NOMINEE", "config": None, "descriptive_config": "NORM-J|r3|a1"},
          "R*": {"status": "NOMINEE", "config": "RAW-J|b0.3"}, "L*": {"status": "NOMINEE", "config": "RAW-L|b0.3"},
          "C*": {"status": "NOMINEE", "config": "RAW-J|b0.3"}}
    lock = {"sex_prior_defense_fit_sha256": EL.prior_hash(D), "statuses": st,
            "resolved": {x: s.get("config") or s.get("descriptive_config") for x, s in st.items()},
            "seeds": {str(k): {"score": {lab: {} for lab in labels}} for k in (0, 1, 2)}}
    lp = tmp_path / "EL.json"
    lp.write_text(json.dumps(lock))
    out = INF.main(["--evaluation-lock", str(lp)])
    z = np.load(R.U("outer__s0__RAW-J_b0.3") / "preds.npz")
    a = np.mean([roc_auc_score(sex, z["P_auc_pair"][s][:, 1]) for s in range(3)])
    assert out["levels"]["R#0#RAW-J|b0.3#prim#pair"]["point"] == pytest.approx(a, abs=1e-12)
    p19 = next(e for e in out["primary"] if e["id"] == "P19")       # R* vs L*: L* pair stronger leak -> positive
    assert p19["point"] > 0.05
    p13 = next(e for e in out["primary"] if e["id"] == "P13")
    p04 = next(e for e in out["primary"] if e["id"] == "P04")
    assert p13["point"] == p04["point"]
    assert out["claimA"]["decision"] == "NOT_ESTABLISHED" and not out["claimA"]["status_requirements_met"]
    assert len(out["secondary"]) == FAM.SECONDARY_SIZE
    assert (tmp_path / "pkg" / "ALL_LEVELS.csv").exists()
    lp.write_text(json.dumps(dict(lock, sex_prior_defense_fit_sha256="0" * 64)))
    with pytest.raises(AssertionError):
        INF.main(["--evaluation-lock", str(lp)])


def _row(cid, aucs, accs, const=(0.75, 0.3)):
    seeds = {}
    for k in (0, 1, 2):
        u = {0: {"acc": accs[0], "const_acc": const[0]}, 1: {"acc": accs[1], "const_acc": const[1]}}
        seeds[k] = {"auc": dict(zip(("v1", "v2", "pair"), aucs)), "utility": u}
    return seeds


def test_selection_rules_on_synthetic_rows():
    uref = {k: {0: {"acc": 0.85, "const_acc": 0.75}, 1: {"acc": 0.48, "const_acc": 0.3}} for k in (0, 1, 2)}

    def row(cid, aucs, accs):
        s = _row(cid, aucs, accs)
        r = {"config": cid, "family": SEL.family_of(cid), "seeds": {}, "compute": 0}
        for k in s:
            gm = SEL.gate_margins(s[k]["utility"], uref[k])
            r["seeds"][k] = {**s[k], "gate_margins": gm, "gate_shortfall": SEL.shortfall(gm),
                             "task_feasible": SEL.shortfall(gm) == 0.0}
        r["mean_pair"] = aucs[2]
        r["task_feasible"] = all(x["task_feasible"] for x in r["seeds"].values())
        r["gate_shortfall"] = sum(x["gate_shortfall"] for x in r["seeds"].values())
        return r
    L1 = row("RAW-L|b0.3", (0.80, 0.82, 0.85), (0.85, 0.475))
    L2 = row("NORM-L|r5|a1", (0.70, 0.70, 0.75), (0.80, 0.40))      # lower recovery but fails the gates
    pL = SEL.pick([L1, L2], tie=lambda r: (r["compute"], r["config"]))
    assert pL["status"] == "NOMINEE" and pL["config"] == "RAW-L|b0.3"
    N1 = row("NORM-J|r3|a1", (0.79, 0.83, 0.80), (0.85, 0.475))     # v2 0.83 > 0.82 + 0.005 -> guard fails
    N2 = row("NORM-J|r3|a2", (0.79, 0.824, 0.81), (0.85, 0.475))
    pN = SEL.pick([N1, N2], guards={"L*": L1}, tie=lambda r: (T.parse_id(r["config"])["rho"], r["config"]))
    assert pN["config"] == "NORM-J|r3|a2"
    pN2 = SEL.pick([N1], guards={"L*": L1}, tie=lambda r: (r["config"],))
    assert pN2["status"] == "NO_FEASIBLE_NOMINEE" and pN2["descriptive_config"] == "NORM-J|r3|a1"
    assert pN2["row"]["nomination_shortfall"] == pytest.approx(3 * (0.83 - 0.825))
    assert "guard_ok" not in L1          # evaluation happens on private copies
