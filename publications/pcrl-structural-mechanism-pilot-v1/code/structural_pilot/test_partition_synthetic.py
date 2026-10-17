"""Brute-force checks of the shared-cell merge identities on invented data (no real data)."""
import numpy as np

from structural_pilot import partition as P


def pooled_metrics(assign, t1, t2, pi, sex, occ):
    """Plug-in metrics of the pooled-posterior release, teacher posteriors as truth; guard uses labels."""
    N = len(assign); out = {}
    for name, t in (('1', t1), ('2', t2)):
        rel = np.zeros_like(t)
        for c in np.unique(assign):
            m = assign == c; rel[m] = t[m].mean(0)
        out['brier' + name] = ((rel - t) ** 2).sum(1).mean()          # excess over the teacher = within-cell variance
        out['nll' + name] = P.kl_rows(t, rel).mean()
        out['acc' + name] = (t.max(1) - t[np.arange(N), rel.argmax(1)]).mean()
        if name == '2':
            rel2 = rel
    pc = np.zeros(N)
    for c in np.unique(assign):
        m = assign == c; pc[m] = pi[m].mean()
    out['sexvar'] = ((pc - pi.mean()) ** 2).mean()
    out['sexinfo'] = P.kl_bern(pc, np.full(N, pi.mean())).mean()
    y = np.eye(t2.shape[1])[occ]
    for s in (0, 1):
        m = sex == s
        out[f'guard{s}'] = ((rel2[m] - y[m]) ** 2).sum(1).mean()
    return out


def test_identities():
    rng = np.random.default_rng(1)
    N, k = 600, 40
    t1 = rng.dirichlet([2, 2], N); t2 = rng.dirichlet([1, 1, 1, 1, 1], N); pi = rng.beta(2, 2, N)
    sex = (rng.random(N) < pi).astype(int); occ = np.array([rng.choice(5, p=p) for p in t2])
    assign = rng.integers(0, k, N)
    C = P.Cells(assign, k, t1, t2, pi, {'sex': sex, 'occ': occ})
    base = pooled_metrics(assign, t1, t2, pi, sex, occ)
    worst = 0.0
    for a, b in [(0, 1), (3, 17), (5, 39), (12, 13)]:
        d = {kk: float(v[0]) for kk, v in P.merge_delta(C, a, b).items()}
        merged = assign.copy(); merged[merged == b] = a
        after = pooled_metrics(merged, t1, t2, pi, sex, occ)
        for key in ('brier1', 'brier2', 'nll1', 'nll2', 'acc1', 'acc2', 'guard0', 'guard1'):
            worst = max(worst, abs((after[key] - base[key]) - d[key]))
        # SEX criteria are reductions of between-cell spread: merged spread = base - delta
        worst = max(worst, abs((base['sexvar'] - after['sexvar']) - d['sexvar']))
        worst = max(worst, abs((base['sexinfo'] - after['sexinfo']) - d['sexinfo']))
    assert worst < 1e-12, worst
    # greedy with all-pairs edges reaches the stop count and returns a valid map
    C2 = P.Cells(assign, k, t1, t2, pi)
    edges = [(i, j) for i in range(k) for j in range(i + 1, k)]
    final, spent, steps, reason = P.greedy(C2, edges, lambda d: d['brier2'] + d['nll2'], lambda d, s: True, stop_count=10)
    assert len(np.unique(final)) == 10 and steps == k - 10 and reason == 'stop_count'
    # spent equals the brute-force total change
    m = final[assign]
    after = pooled_metrics(m, t1, t2, pi, sex, occ)
    assert abs((after['brier2'] - base['brier2']) - spent['brier2']) < 1e-12
    # k-means is deterministic and assigns to nearest centroids
    Q = np.column_stack((t1, t2))
    c1, a1 = P.kmeans(Q, 16, seed=3, iters=20); c2, a2 = P.kmeans(Q, 16, seed=3, iters=20)
    assert np.array_equal(a1, a2) and np.array_equal(P.nearest(Q, c1), a1)
    return worst


if __name__ == '__main__':
    print('max identity error', test_identities()); print('partition synthetic checks: PASS')
