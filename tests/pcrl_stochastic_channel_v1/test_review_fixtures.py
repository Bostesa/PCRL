"""The review's exact rational fixtures, vendored and pinned.

These establish mathematical properties of finite models only. They read no ACS data and imply
nothing about the ACS frontier. Provenance: `PCRL_Mathematical_Review_and_Fixtures.zip`,
20 September 2026; the script is vendored byte-for-byte and its sha256 is pinned below.
"""
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[2] / 'experiments' / 'pcrl_stochastic_channel_v1' / 'fixtures'
SCRIPT = FIXTURES / 'stochastic_channel_fixtures.py'
COMMITTED = FIXTURES / 'stochastic_channel_fixtures.json'
SCRIPT_SHA = '998573c81c4b5558ea524a09eaddbc893a19ddd9f7f065949685336c4092c085'


def test_vendored_script_is_byte_identical_to_the_reviewed_artifact():
    import hashlib
    assert hashlib.sha256(SCRIPT.read_bytes()).hexdigest() == SCRIPT_SHA


def test_script_self_reports_the_same_sha():
    """The committed JSON records the sha of the script that produced it."""
    assert json.loads(COMMITTED.read_text())['script_sha256'] == SCRIPT_SHA


def test_fixtures_regenerate_identical_output(tmp_path):
    """Standard library only, deterministic, and reproducing the committed artifact exactly."""
    work = tmp_path / 'stochastic_channel_fixtures.py'
    work.write_bytes(SCRIPT.read_bytes())
    proc = subprocess.run([sys.executable, str(work)], capture_output=True, text=True, cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    produced = json.loads((tmp_path / 'stochastic_channel_fixtures.json').read_text())
    committed = json.loads(COMMITTED.read_text())
    produced.pop('output', None)
    committed.pop('output', None)
    assert produced == committed


def test_randomization_strictly_beats_every_deterministic_perfectly_private_map():
    """The separation that motivates the whole candidate, read off the committed artifact."""
    for toy in json.loads(COMMITTED.read_text())['toys']:
        assert toy['random_release_sensitive_independence_exact'] is True
        assert toy['random_release_sensitive_information_nats'] == 0.0
        assert toy['largest_exact_private_deterministic_task_information_nats'] == 0
        assert toy['random_release_task_information_nats'] > 0.3
        # Every deterministic map is covered: all five set partitions of three states.
        assert len(toy['all_deterministic_partitions']) == 5
        private = [p for p in toy['all_deterministic_partitions'] if p['exact_private']]
        assert len(private) == 1 and private[0]['task_information_nats'] == 0.0


def test_independence_of_the_stochastic_release_holds_in_exact_arithmetic():
    """Recomputed here in `fractions`, independently of the script, for the interior-probability toy."""
    p_t = [Fraction(1, 3)] * 3
    p_s1_given_t = [Fraction(1, 10), Fraction(2, 5), Fraction(9, 10)]
    q = [Fraction(1), Fraction(0), Fraction(11, 13)]
    p_z1 = sum(pt * qt for pt, qt in zip(p_t, q))
    p_s1_z1 = sum(pt * ps * qt for pt, ps, qt in zip(p_t, p_s1_given_t, q))
    p_s1 = sum(pt * ps for pt, ps in zip(p_t, p_s1_given_t))
    assert p_s1_z1 == p_s1 * p_z1                      # exact, not a tolerance
    assert p_s1 == Fraction(7, 15)


def test_no_deterministic_map_can_hit_the_required_sensitive_rate():
    """Why the deterministic family fails: no proper subset of states averages to P(S=1) = 7/15."""
    p_s1_given_t = [Fraction(1, 10), Fraction(2, 5), Fraction(9, 10)]
    overall = sum(p_s1_given_t) / 3
    assert overall == Fraction(7, 15)
    subsets = [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2)]
    for sub in subsets:
        assert sum(p_s1_given_t[i] for i in sub) / len(sub) != overall


def test_coarse_conditioning_counterexample_is_recorded():
    """Binning the context can hide a leak entirely: S xor C is invisible given a constant bin."""
    ce = json.loads(COMMITTED.read_text())['view_and_coarsening_counterexamples']
    x = ce['individually_private_jointly_disclosing']
    assert x['A_increment_nats'] == 0.0
    assert x['increment_conditioned_on_constant_context_bin_nats'] == 0.0
    assert abs(x['coalition_increment_given_C_nats'] - 0.6931471805599453) < 1e-12
    y = ce['coalition_increment_zero_local_increment_positive']
    assert y['coalition_increment_given_B_nats'] == 0.0
    assert y['A_increment_nats'] > 0.69
    # Both directions occur, so both role constraints are necessary; neither implies the other.


def test_teacher_fidelity_counterexample_is_recorded():
    """A better teacher reconstruction can be strictly worse for the task and better for leakage."""
    u = json.loads(COMMITTED.read_text())['utility_proxy_counterexample']
    assert u['release_S']['teacher_total_squared_error'] < u['release_Y']['teacher_total_squared_error']
    assert u['release_S']['task_information_nats'] == 0
    assert u['release_S']['sensitive_information_nats'] > 0
    assert u['release_Y']['sensitive_information_nats'] == 0
