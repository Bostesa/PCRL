"""[lra port of lcr/tests/test_admit.py at 091afc2: lcr->lra renames; later edits are listed in PORT_LOG.md]
Lead: registered config-id scheme, unit naming and the admission plan (hash-only; no data load)."""
from lra import admit as AD
from lra import run as R


def test_config_scheme_and_bank_counts():
    assert len(R.d0_ids()) == 27 and len(R.d1_fixed_ids()) == 26 and len(R.new_fit_ids()) == 30
    assert len(R.code_ids()) == 83 == len(set(R.code_ids())) and len(R.scored_ids()) == 88
    assert len(R.weighted_ids()) == 24 and len(R.constrained_ids()) == 5
    p = R.parse_id("U|W-SEQ-12|i8o64|l0.025|D1")
    assert (p["arm"], p["base_family"], p["lam"], p["decoder"], p["privacy_trained"]) == ("weighted", "SEQ-12", 0.025, "D1", True)
    p = R.parse_id("U|SEQ-21|i8o64|l0.1|D1")
    assert p["arm"] == "d1_fixed" and p["privacy_trained"] is True          # fixed-map D1 privacy maps stay privacy-trained
    assert R.parse_id("U|C-TASK|i8o64|D1")["privacy_trained"] is False
    assert R.parse_id("U|FINE-TASK|i8o64|D1")["privacy_trained"] is False
    assert R.parse_id("U|K-JOINT-PAIR|i8o64|D1")["arm"] == "constrained"
    assert R.unit_for(2, "U|JOINT|i8o64|l0.06") == "pol__s2__U_JOINT_i8o64_l0.06"
    assert R.unit_for(2, "U|JOINT|i8o64|l0.06|D1") == "dec__s2__U_JOINT_i8o64_l0.06_D1"
    assert R.unit_for(1, "U|K-LOCAL|i8o64|D1") == "new__s1__U_K-LOCAL_i8o64_D1"


def test_admission_units_cover_the_d0_bank():
    for k in R.SEEDS:
        u = AD.units_for(k)
        pol = [x for x in u if x.startswith("pol__")]
        assert sorted(pol) == sorted(R.unit_for(k, c) for c in R.d0_ids())
        assert all(f"inner__{x}" in u for x in pol) and not [x for x in u if x.startswith("outer__")]
        assert f"fine__s{k}" in u and f"tea__s{k}__U" in u


def test_plan_hash_precheck_is_clean():
    pl = AD.plan()
    assert pl["bad"] == [] and len(pl["units"]) == 189 and len(pl["admitted"]) == 6
