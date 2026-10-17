"""Synthetic checks of the frozen lookup reader (no real data): exact-byte keys, smoothing, unseen fallback, and that a
finite-valued release with a planted cell-level signal is read while a continuous release falls back to the base rate."""
import numpy as np

from structural_pilot.readers import fit_lookup, lookup_keys, predict_lookup


def test_lookup():
    rng = np.random.default_rng(0)
    cells = rng.integers(0, 20, 4000)
    p_cell = rng.dirichlet(np.ones(5), 20)
    sex_rate = np.linspace(0.2, 0.8, 20)
    y = (rng.random(4000) < sex_rate[cells]).astype(int)
    rel = p_cell[cells]
    rec = fit_lookup(lookup_keys(rel), y)
    assert len(rec['table']) == 20
    k0 = lookup_keys(p_cell[[3]])[0]
    n1, n = rec['table'][k0]
    pi = y.mean()
    assert abs(predict_lookup(rec, [k0])[0] - (n1 + 2 * pi) / (n + 2)) < 1e-15
    assert predict_lookup(rec, [lookup_keys(np.full((1, 5), 0.2))[0]])[0] == pi
    # one ulp away is a different key (exact bytes)
    tweaked = p_cell[[3]].copy(); tweaked[0, 0] = np.nextafter(tweaked[0, 0], 1.0)
    assert predict_lookup(rec, lookup_keys(tweaked))[0] == pi
    # pair keys concatenate both vectors
    a = rng.dirichlet(np.ones(2), 5); b = rng.dirichlet(np.ones(5), 5)
    assert lookup_keys(a, b)[2] == np.ascontiguousarray(np.column_stack((a, b))[2]).tobytes()
    # a continuous release: almost every key unseen, constant score
    cont = rng.random((1000, 5)); cont /= cont.sum(1, keepdims=True)
    rc = fit_lookup(lookup_keys(cont[:500]), y[:500])
    s = predict_lookup(rc, lookup_keys(cont[500:]))
    assert np.all(s == rc['pi'])


if __name__ == '__main__':
    test_lookup(); print('lookup synthetic checks: PASS')
