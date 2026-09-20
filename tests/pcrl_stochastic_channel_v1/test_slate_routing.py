"""Routing of untouched-J predictors into an extended release's candidate slate.

The pilot's METHOD section 1 promised that untouched-J predictor candidates are part of every
comparison, on the ground that "a larger input can make a finite learner worse". The pilot routed
only H-only ancestors (columns 0..H_A_WIDTH-1) and audited `ref_J` as a separate condition, which
does not serve that rationale: the two numbers being differenced come from independently selected
slates. These fixtures pin the corrected routing. No data is needed.
"""
import numpy as np
import pytest

from experiments.pcrl_direct_adversarial_v1 import inputs as dax
from experiments.pcrl_stochastic_channel_v1 import slate


def test_j_base_is_the_untouched_j_wire_width():
    """[H_A, Z_J] is columns 0..19; R starts at 20. Pins the constant against the wire layout."""
    assert slate.J_BASE == dax.H_A_WIDTH + dax.A0_WIDTH == 20


def test_local_route_is_h_a_and_z_j_and_excludes_r():
    """A-view J anchor must see H_A and Z_J and nothing else, whatever the extension width."""
    for r in (2, 4, 8):
        width_a = slate.J_BASE + r
        route = slate.route_j_anchor('A', width_a)
        assert route == tuple(range(20))
        assert max(route) < slate.J_BASE            # never reaches into R


def test_coalition_route_picks_up_h_b_after_the_extension_block():
    """AB = [H_A, Z_J, R, H_B]; H_B sits at width_a, width_a+1 (build_wires appends it last)."""
    r = 4
    width_a = slate.J_BASE + r
    route = slate.route_j_anchor('AB', width_a)
    assert route == tuple(range(20)) + (24, 25)
    assert len(route) == slate.J_BASE + dax.H_B_WIDTH == 22
    assert all(c < slate.J_BASE or c >= width_a for c in route)   # skips the whole R block


def test_route_respects_a_candidate_that_used_a_column_subset():
    """A J candidate fitted on a projection of its own wire keeps that projection after routing."""
    width_a = slate.J_BASE + 2
    assert slate.route_j_anchor('A', width_a, columns=(0, 5, 19)) == (0, 5, 19)
    assert slate.route_j_anchor('AB', width_a, columns=(3, 20, 21)) == (3, 22, 23)


def test_b_view_is_refused_because_canonical_b_candidates_are_shared():
    """The B wire is H_B in every system, so a J anchor there would duplicate the canonical slate."""
    with pytest.raises(ValueError, match='canonical'):
        slate.route_j_anchor('B', slate.J_BASE + 2)


def test_routed_columns_select_the_untouched_j_wire_out_of_an_extended_wire():
    """End-to-end on synthetic arrays: routing must reproduce the J wire bit-for-bit."""
    rng = np.random.default_rng(0)
    n, r = 37, 4
    ha = rng.normal(size=(n, dax.H_A_WIDTH))
    zj = rng.normal(size=(n, dax.A0_WIDTH))
    hb = rng.normal(size=(n, dax.H_B_WIDTH))
    ext_r = rng.normal(size=(n, r))

    j_a = np.column_stack((ha, zj))                       # untouched-J wire/A
    j_ab = np.column_stack((j_a, hb))                     # untouched-J wire/AB
    x_a = np.column_stack((ha, zj, ext_r))                # extended wire/A
    x_ab = np.column_stack((x_a, hb))                     # extended wire/AB
    width_a = x_a.shape[1]

    assert np.array_equal(x_a[:, slate.route_j_anchor('A', width_a)], j_a)
    assert np.array_equal(x_ab[:, slate.route_j_anchor('AB', width_a)], j_ab)


def test_anchor_metadata_marks_the_candidate_as_an_ignore_r_j_predictor():
    """The slate must be able to distinguish a separate J comparator from an in-slate J predictor."""
    meta = slate.j_anchor_metadata({'candidate_id': 'wire__mlp_0', 'view': 'A',
                                    'projection_columns': None, 'validation_scores': {'log_loss': 1.}},
                                   'j_anchor__wire__mlp_0', 'A', (0, 1, 2))
    assert meta['candidate_id'] == 'j_anchor__wire__mlp_0'
    assert meta['source_condition'] == 'ref_J'
    assert meta['j_anchor'] is True
    assert meta['ignores_extension'] is True
    assert meta['anchor_ancestor'] is False          # not an H ancestor; a different provenance
    assert meta['validation_scores'] == {'log_loss': 1.}   # carried for the parity re-check


class _Stub:
    def __init__(self, cid, columns=None, **meta):
        self.base = object()
        self.columns = columns
        self.metadata = dict({'candidate_id': cid, 'diagnostic_only': False,
                              'validation_scores': {'log_loss': 0.5}}, **meta)


def _fake_j_audits():
    def role_slate():
        return {'wire__mlp_0': _Stub('wire__mlp_0'),
                'wire__kernel__1.0': _Stub('wire__kernel__1.0'),
                'anchor__wire__mlp_0': _Stub('anchor__wire__mlp_0', (0, 1, 2, 3), anchor_ancestor=True),
                'inherited_A__wire__mlp_0': _Stub('inherited_A__wire__mlp_0', (0, 1))}
    return {'candidates': {120: {'A/SEX': role_slate(), 'AB/SEX': role_slate(), 'B/SEX': role_slate()},
                           360: {'A/SEX': role_slate(), 'AB/SEX': role_slate(), 'B/SEX': role_slate()}}}


def test_supplement_covers_both_budgets_and_skips_the_b_roles():
    out = slate.build_j_anchor_candidates(_fake_j_audits(), width_a=slate.J_BASE + 2)
    assert set(out) == {120, 360}
    for budget in (120, 360):
        assert set(out[budget]) == {'A/SEX', 'AB/SEX'}          # no B role supplemented


def test_supplement_takes_fresh_j_predictors_and_refuses_to_re_anchor():
    """H ancestors are routed by the extension itself and inherited singletons are regenerated."""
    out = slate.build_j_anchor_candidates(_fake_j_audits(), width_a=slate.J_BASE + 2)
    ids = set(out[360]['A/SEX'])
    assert ids == {'j_anchor__wire__mlp_0', 'j_anchor__wire__kernel__1.0'}
    assert not any('anchor__wire__mlp_0' == i.removeprefix('j_anchor__') and
                   out[360]['A/SEX'][i].metadata.get('anchor_ancestor') for i in ids)


def test_supplemented_ids_cannot_collide_with_the_extension_own_slate():
    """`j_anchor__` prefix keeps the namespace disjoint from fresh, kernel and `anchor__` ids."""
    out = slate.build_j_anchor_candidates(_fake_j_audits(), width_a=slate.J_BASE + 2)
    for cid in out[120]['AB/SEX']:
        assert cid.startswith('j_anchor__')
        assert not cid.startswith('anchor__') and not cid.startswith('inherited_')


def test_supplemented_coalition_candidate_carries_the_h_b_columns():
    out = slate.build_j_anchor_candidates(_fake_j_audits(), width_a=slate.J_BASE + 8)
    c = out[360]['AB/SEX']['j_anchor__wire__mlp_0']
    assert c.columns == tuple(range(20)) + (28, 29)
    assert c.metadata['projection_columns'] == list(c.columns)
    assert c.metadata['source_condition'] == 'ref_J'


def test_build_audits_accepts_the_supplement_without_changing_default_behaviour():
    """The hook is opt-in: the parameter exists and defaults to None."""
    import inspect
    from experiments import acs_spectral_audits as audit
    sig = inspect.signature(audit.build_audits)
    assert sig.parameters['extra_candidates'].default is None
    assert sig.parameters['extra_candidates'].kind is inspect.Parameter.KEYWORD_ONLY


def test_adding_a_candidate_cannot_raise_the_selected_validation_loss():
    """Selection is argmin validation log loss, so an added J anchor only lowers the attacker's
    loss -- i.e. raises measured recovery. This correction can never repair a failed screen."""
    base = {'wire__mlp_0': 0.70, 'wire__kernel__1.0': 0.66}
    with_anchor = dict(base, j_anchor__wire__mlp_0=0.61)
    pick = lambda d: min(sorted(d), key=lambda k: (d[k], k))
    assert d_min(with_anchor) <= d_min(base)
    assert pick(with_anchor) == 'j_anchor__wire__mlp_0'


def d_min(d):
    return min(d.values())
