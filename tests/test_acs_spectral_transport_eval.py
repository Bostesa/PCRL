import json
import numpy as np
import pytest
from experiments import acs_spectral_transport_eval as ev
from experiments.acs_transfer_heads import metrics
from scripts import report_acs_spectral_transport as rep


def test_final_partition_sealed_without_lock(tmp_path):
    with pytest.raises(PermissionError):
        ev.load_final(tmp_path)
    with pytest.raises(PermissionError):
        ev.load_fit_partition('final_evaluation')


def test_lock_detects_tampering(tmp_path):
    f = tmp_path/'object.txt'; f.write_text('frozen')
    (tmp_path/'TRANSPORT_LOCK.json').write_text(json.dumps({'files': {str(f): ev.sha_file(f)}, 'final_partition_sha256': 'x'}))
    ev.verify_lock(tmp_path)
    f.write_text('changed')
    with pytest.raises(PermissionError):
        ev.verify_lock(tmp_path)


def test_dry_run_cannot_target_study_or_final(monkeypatch):
    monkeypatch.setattr(ev, 'DRY_RUN_PARTITION', 'task_validation')
    with pytest.raises(PermissionError):
        ev.load_final(ev.OUT)
    monkeypatch.setattr(ev, 'DRY_RUN_PARTITION', 'final_evaluation')
    with pytest.raises(PermissionError):
        ev.load_final('/tmp/elsewhere')


def _rec(cid, origin, space='wire', diagnostic=False, cols=None):
    return {'candidate_id': cid, 'transport_origin': origin, 'space': space, 'diagnostic_only': diagnostic,
            'projection_columns': cols, 'inherited_singleton': False}


def test_scope_membership_and_selection():
    recs = {'wire__a': _rec('wire__a', 'fresh'), 'derived__a': _rec('derived__a', 'fresh', 'derived'),
            'catchup': _rec('catchup', 'catchup'), 'saved_adversary': _rec('saved_adversary', 'catchup', diagnostic=True),
            'frozen2018__x': _rec('frozen2018__x', 'frozen2018')}
    assert ev.scope_members(recs, 'common_fresh') == ['wire__a']
    assert ev.scope_members(recs, 'fresh_expanded') == ['derived__a', 'wire__a']
    assert 'saved_adversary' not in ev.scope_members(recs, 'transport_all')
    scores = {k: {'log_loss': v} for k, v in zip(recs, (.5, .4, .3, .0, .2))}
    sel = ev.select(recs, scores)
    assert sel == {'common_fresh': 'wire__a', 'fresh_expanded': 'derived__a', 'fresh_catchup': 'catchup', 'transport_all': 'frozen2018__x'}


def test_coalition_inherits_on_correct_block():
    roles = {'A/SEX': {'x': _rec('x', 'fresh', cols=[1, 3])}, 'B/SEX': {'y': _rec('y', 'fresh')},
             'AB/SEX': {}, 'A/RAC1P': {}, 'B/RAC1P': {}, 'AB/RAC1P': {}}
    ev.inherit(roles, {'wire': (20, 2)})
    assert roles['AB/SEX']['inherited_A__x']['projection_columns'] == [1, 3]
    assert roles['AB/SEX']['inherited_B__y']['projection_columns'] == [20, 21]
    assert roles['AB/SEX']['inherited_B__y']['source_view'] == 'B'


def test_person_loss_matches_scorer_and_bootstrap_ratio():
    rng = np.random.default_rng(0); y = rng.integers(0, 3, 50); p = rng.dirichlet(np.ones(3), 50); p[0] = [1, 0, 0]
    w = rng.integers(1, 100, 50).astype(float)
    assert np.isclose(rep.person_loss(y, p).mean(), metrics(y, p, 3)['log_loss'], rtol=0, atol=1e-12)
    serial = np.repeat(np.arange(25), 2).astype(str)
    b = rep.Bootstrap(serial, w, {'t': y}, replicates=5, seed=1)
    l = rep.person_loss(y, p)
    reps, point = b.means('t', 'person_weighted', l[:, None])
    assert np.isclose(point[0], w @ l/w.sum())
    c = b.rows[2]
    assert np.isclose(reps[2, 0], (c*w) @ l/((c*w).sum()))
    assert all(set(np.unique(b.rows[k].reshape(25, 2), axis=1).shape) for k in range(5))
    assert np.all(b.rows[:, 0::2] == b.rows[:, 1::2])  # both rows of a household share a count


def test_simultaneous_intervals_cover_and_handle_degenerate():
    rng = np.random.default_rng(3); R = rng.normal(size=(4000, 5)); R[:, 4] = 0.
    low, high, se, crit = rep.simultaneous(R, np.zeros(5))
    assert 2.0 < crit < 3.0 and low[4] == high[4] == 0.
    inside = np.all((R[:, :4] >= low[:4]) & (R[:, :4] <= high[:4]), axis=1).mean()
    assert inside >= .95


def test_withholding_mixture_is_linear():
    rng = np.random.default_rng(2); lh, la = rng.random(100), rng.random(100); w = rng.random(100)
    for p in rep.P_VALUES:
        assert np.isclose(w @ ((1-p)*lh+p*la)/w.sum(), (1-p)*(w @ lh)/w.sum()+p*(w @ la)/w.sum())
