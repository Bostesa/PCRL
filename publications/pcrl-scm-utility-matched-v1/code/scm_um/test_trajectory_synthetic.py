"""Synthetic checks of the trajectory recorder (invented data only): the recorded run equals scm_bench.engine.run_path for every
kind; replaying the recorded merges reproduces the terminal map and every intermediate prefix equals a fresh engine run with
the matching target; the checkpoint rules (NLL grid with slack, counts, terminal, companions, reached flags) behave as registered;
effective-release keys deduplicate identical prefixes."""
import json

import numpy as np

from scm_bench import engine as E
from scm_bench.test_engine_synthetic import invented
from scm_um import trajectory as TR


def stats(n=900, cells=40, seed=3):
    T1, T2, pi, sex, occ = invented(n, seed)
    _, _, fine, _ = E.kl_kmeans(T1, T2, cells, seed)
    k = int(fine.max()) + 1
    return E.Stats(fine, k, T1, T2, pi, sex, occ)


def main():
    res = {}
    st0 = stats()
    k = st0.k
    for kind, kw in (('SCM', dict(beta=1.0)), ('TWIN', dict(beta=1.0)), ('IBSI', dict(gamma=1.0, target=4)), ('SHUF', dict(beta=1.0))):
        base = st0
        if kind == 'SHUF':
            base = st0.copy(); perm = np.random.default_rng(7).permutation(k); base.SP = st0.n * (st0.SP / st0.n)[perm]
        p = TR.run_path_recorded(base, kind, **kw)
        ref = E.run_path(base, kind, **kw)
        same = bool(np.array_equal(p['final'], ref[0]) and p['steps'] == ref[4]['steps'] and p['stop_reason'] == ref[4]['stop_reason']
                    and all(p['spent_final'][key] == ref[4]['spent_plugin_vs_C0'][key] for key in E.BUDGET_KEYS))
        steps = list(range(p['steps'] + 1))
        snaps = TR.replay(base, p['merges'], steps)
        term_ok = bool(np.array_equal(snaps[p['steps']][0]['cell_of_initial'], p['final']))
        # every prefix equals a fresh engine run stopped at that count (same deterministic order up to the target)
        pref_ok = True
        for m in sorted({0, 1, p['steps'] // 2, max(p['steps'] - 1, 0)}):
            tgt = k - m
            r2 = E.run_path(base, kind, **{**kw, 'target': tgt})
            tab_ref = E.tables(r2[2], r2[3], r2[0], r2[1])
            pref_ok &= all(np.array_equal(snaps[m][0][x], tab_ref[x]) for x in ('v1', 'v2', 'cell_of_initial', 'group_of_initial'))
        F = p['spent'][:, list(E.BUDGET_KEYS).index('nll2')]
        cps = TR.checkpoints(F, p['steps'], k, {t: min(int(5 * i), p['steps']) for i, t in enumerate(TR.NLL_GRID)}, 3)
        rule_ok = True
        for t in TR.NLL_GRID:
            m, reached, _ = cps[TR.nll_label(t)]
            over = [j for j in range(len(F)) if F[j] > t + TR.SLACK]
            rule_ok &= bool(m == (over[0] - 1 if over else p['steps']))
            rule_ok &= reached == bool(len(over) > 0 or F[p['steps']] >= t - TR.SLACK)
        rule_ok &= cps['TERM'][0] == p['steps'] and cps['HIST'][0] == 3
        for c in TR.CNT_GRID:
            m = k - c
            rule_ok &= cps[f'C{c}'][0] == (m if 0 <= m <= p['steps'] else None)
        res[kind] = {'recorded_equals_engine': same, 'replay_terminal': term_ok, 'prefixes_equal_targeted_runs': bool(pref_ok),
                     'checkpoint_rules': bool(rule_ok), 'steps': p['steps'], 'stop': p['stop_reason']}
    # effective-release keys: identical prefixes collide, different prefixes do not
    p = TR.run_path_recorded(st0, 'IBSI', gamma=0.5, target=4)
    sn = TR.replay(st0, p['merges'], [0, 1, 2])
    keys = [TR.release_key(sn[m][0]) for m in (0, 1, 2)]
    sn2 = TR.replay(st0, p['merges'], [1])
    res['release_keys'] = {'distinct_prefixes_distinct_keys': len(set(keys)) == 3, 'same_prefix_same_key': TR.release_key(sn2[1][0]) == keys[1]}
    res['all_ok'] = all(v if isinstance(v, bool) else all(x for x in v.values() if isinstance(x, bool)) for v in res.values())
    print(json.dumps(res, indent=1))
    print('PASS' if res['all_ok'] else 'FAIL')
    return res


if __name__ == '__main__':
    main()
