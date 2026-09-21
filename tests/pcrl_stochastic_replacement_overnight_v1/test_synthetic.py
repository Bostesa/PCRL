"""The certified synthetic comparison (§9) and the 2016 containment assertion."""
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.pcrl_stochastic_channel_v1 import channel as ch
from experiments.pcrl_stochastic_replacement_overnight_v1 import synthetic as S

RES = Path(__file__).resolve().parents[2] / 'results' / 'pcrl_stochastic_replacement_overnight_v1'


def test_enumeration_covers_every_deterministic_map():
    assert sum(1 for _ in S.deterministic_maps(5, 2)) == 2 ** 5
    assert sum(1 for _ in S.deterministic_maps(5, 3)) == 3 ** 5
    seen = {tuple(a) for a, _ in S.deterministic_maps(3, 2)}
    assert len(seen) == 8


def test_every_enumerated_map_is_a_valid_channel():
    for _, q in S.deterministic_maps(4, 3):
        assert np.all(q >= 0) and np.allclose(q.sum(1), 1.0)
        assert set(np.unique(q)) <= {0.0, 1.0}


def test_registered_model_permits_a_locally_private_deterministic_map():
    """The model is deliberately NOT rigged against deterministic maps.

    Some proper subsets average to the overall rate, so exact privacy is reachable deterministically
    under the trivial context. This is why the local comparison shows no randomization advantage, and
    it is what makes the coalition advantage meaningful rather than an artifact of the construction.
    """
    import itertools
    m = S.registered_model()
    assert m['overall_p_s1'] == pytest.approx(0.5)
    p = m['p_s1']
    averaging = [sub for r in range(1, len(p))
                 for sub in itertools.combinations(range(len(p)), r)
                 if abs(np.mean(p[list(sub)]) - 0.5) < 1e-12]
    assert averaging, 'expected at least one locally private nonconstant grouping'
    assert (2,) in averaging and (0, 4) in averaging and (1, 3) in averaging


def test_coalition_role_is_strictly_stricter_than_the_local_role():
    """Conditioning on the public context must be able to expose a leak the marginal hides."""
    m = S.registered_model()
    q = np.zeros((5, 2))
    q[[1, 3], 1] = 1.0          # the map that matches the useful target exactly
    q[[0, 2, 4], 0] = 1.0
    local = ch.cmi(m['roles']['A/S'], q)
    coalition = ch.cmi(m['roles']['AB/S'], q)
    assert local == pytest.approx(0.0, abs=1e-12)     # marginally private by coincidence
    assert coalition > 1e-3                            # and not conditionally private


def test_constant_channel_is_always_feasible_at_zero_budget():
    """Failure to find a useful null direction is not infeasibility of the simplex."""
    m = S.registered_model()
    for k in (2, 3):
        q = np.zeros((5, k))
        q[:, 0] = 1.0
        for name, p in m['roles'].items():
            assert ch.cmi(p, q) == pytest.approx(0.0, abs=1e-12)


def test_cost_is_zero_exactly_when_the_token_matches_the_target():
    m = S.registered_model()
    cost = S.cost_from_target(m['y'], 2)
    for t, y in enumerate(m['y']):
        assert cost[t, y] == 0.0
        assert cost[t, 1 - y] == 1.0


def test_recorded_comparison_is_internally_consistent():
    """Re-check the published artifact: stochastic feasible, deterministic certified exhaustive."""
    path = RES / 'synthetic' / 'DETERMINISTIC_COMPARISON.json'
    if not path.exists():
        pytest.skip('synthetic comparison not yet run')
    rec = json.loads(path.read_text())
    for tag, r in rec.items():
        k = r['k']
        n_enum = r['deterministic']['n_enumerated']
        assert n_enum == k ** 5, (tag, n_enum)
        assert r['deterministic']['certified'] is True
        # The stochastic solution must actually satisfy its budgets, recomputed independently.
        for role, v in r['stochastic']['constraint_violation'].items():
            assert v <= 1e-6, (tag, role, v)
        if r['deterministic']['best'] is not None:
            assert r['advantage_of_randomization'] == pytest.approx(
                r['deterministic']['best']['objective'] - r['stochastic']['objective'], abs=1e-9)


def test_recorded_comparison_shows_no_local_advantage_and_a_coalition_advantage():
    """The honest shape of the result, pinned so it cannot be overstated later."""
    path = RES / 'synthetic' / 'DETERMINISTIC_COMPARISON.json'
    if not path.exists():
        pytest.skip('synthetic comparison not yet run')
    rec = json.loads(path.read_text())
    local = [r for tag, r in rec.items() if '|L|' in tag]
    coalition = [r for tag, r in rec.items() if '|C|' in tag]
    assert local and coalition
    assert all(r['strictly_better'] is False for r in local)
    assert all(r['strictly_better'] is True for r in coalition)


# ---------------------------------------------------------------- containment
def test_no_study_artifact_points_at_2016_data():
    """2016 stays sealed.

    The consequential risk is a *machine-read* artifact -- a manifest, ledger, run matrix or bundle
    list -- resolving to 2016 data, or a file named for it. Prose that documents the containment rule
    (`DATA_USE.md` names the forbidden tokens on purpose) is not a leak, so markdown is excluded from
    the path scan and checked separately for path-shaped references only.
    """
    path_tokens = ('/2016/', 'ss16/', 'psam_p06_2016', '2016/1-Year', 'psam_h06_2016')
    offenders = []
    for p in RES.rglob('*'):
        if not p.is_file():
            continue
        if '2016' in p.name or 'ss16' in p.name:
            offenders.append(f'filename:{p}')
            continue
        if p.suffix.lower() in {'.json', '.csv', '.txt'}:
            text = p.read_text(errors='ignore')
            offenders += [f'{p}:{tok}' for tok in path_tokens if tok in text]
        elif p.suffix.lower() == '.md':
            text = p.read_text(errors='ignore')
            offenders += [f'{p}:{tok}' for tok in ('/2016/', '2016/1-Year', 'psam_p06_2016')
                          if tok in text]
    assert offenders == [], offenders


def test_loader_inputs_resolve_only_to_2018():
    """The restored input tree this study reads must contain no 2016 extract."""
    from experiments.pcrl_nonlinear_rank_v1.inputs import FALLBACK_ROOTS
    for root in FALLBACK_ROOTS:
        folk = Path(root) / 'data' / 'folktables'
        if folk.exists():
            years = sorted(c.name for c in folk.iterdir() if c.is_dir())
            assert '2016' not in years, (str(folk), years)
