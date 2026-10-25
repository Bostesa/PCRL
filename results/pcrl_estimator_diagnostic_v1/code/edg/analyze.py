"""Frozen analysis of EDG1 (PROTOCOL.md sections 4 and 6; written before any simulation outcome). Reads out/sim_*.jsonl.gz and
out/bank_*.jsonl.gz, rebuilds the known laws, and writes SUMMARY.json, TABLES.md and figures/ (two PNG figures).

Per cell x view x estimator: mean, replicate SD, Monte Carlo SE of the mean (SD / sqrt(R)), bias and RMSE against the relevant
known-law quantity (A for every estimator; for D_* also against the fitted reader's own population AUC), undefined count.
Per cell x comparison x estimator (comparisons are written finer-or-linked minus coarser-or-single where that applies):
the true population difference dA, mean / SD / MCSE of the estimated difference, P(d > 0), P(d < 0), P(d = 0), and the DECLARED
rule R2SE: 'X > Y' iff d - 2 se > 0, 'X < Y' iff d + 2 se < 0 (se: linearized paired SE with the scored rankings held fixed; the
household-clustered SE in the clustered cell). Classification: if dA = 0 (|dA| <= 1e-12) any resolved conclusion is a FALSE POSITIVE
(reported in total and by direction); if dA != 0 a resolved conclusion of the opposite sign is a WRONG DIRECTIONAL CONCLUSION, the
correct one is power; a DESCRIPTIVE SIGN ERROR is sign(d) != sign(dA) (point estimate only). Secondary margin rule M20:
|d| >= 0.020 (DRF1's D4 target scale) when dA = 0. Rates carry binomial MCSEs."""
from __future__ import annotations

import glob
import gzip
import json
import math
from collections import defaultdict

import numpy as np

from . import OUT, REPORT, RESULTS
from . import laws as L
from . import simulate as SIM

EST = SIM.ESTIMATORS
TOL0 = 1e-12


def _load(pattern):
    rows = []
    for p in sorted(glob.glob(str(OUT / pattern))):
        with gzip.open(p, 'rt') as fh:
            rows += [json.loads(x) for x in fh]
    return rows


def _stats(x, truth=None):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=np.float64)
    R = len(x)
    if R == 0:
        return {'R': 0}
    sd = float(x.std(ddof=1)) if R > 1 else float('nan')
    out = {'R': R, 'mean': float(x.mean()), 'sd': sd, 'mcse': sd / math.sqrt(R)}
    if truth is not None:
        out['bias'] = float(x.mean() - truth)
        out['rmse'] = float(np.sqrt(np.mean((x - truth) ** 2)))
    return out


def _rate(k, R):
    p = k / R if R else float('nan')
    return {'rate': p, 'mcse': math.sqrt(p * (1 - p) / R) if R else float('nan'), 'count': int(k)}


def summarize_sim(rows):
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[r['cell']].append(r)
    scs = {s['cell']: s for s in SIM.scenarios()}
    laws = {}
    out = {}
    for cell, rs in sorted(by_cell.items()):
        sc = scs[cell]
        k = (sc['family'], sc['pi'], sc['occ'])
        law = laws.get(k) or laws.setdefault(k, L.build(*k))
        cres = {'scenario': sc, 'reps': len(rs), 'undefined_values': sum(len(r['undefined']) for r in rs),
                'mean_n': float(np.mean([r['n'] for r in rs])), 'views': {}, 'comparisons': {}}
        for v, V in law['views'].items():
            A = V['A']
            vr = {'A': A, 'support': V['K'], 'occupied_mean': float(np.mean([r['views'][v]['occupied'] for r in rs]))}
            for e in EST:
                vr[e] = _stats([r['views'][v][e] for r in rs], A)
            for e, popk in (('D_nested', 'D_nested_pop_pooled'), ('D_nested', 'D_nested_pop_fold_mean'), ('D_bank', 'D_bank_pop')):
                pop = np.array([r['views'][v][popk] for r in rs], dtype=np.float64)
                val = np.array([r['views'][v][e] for r in rs], dtype=np.float64)
                vr[popk] = _stats(pop, A)
                vr[f'{popk}__estimation_loss_A_minus_pop'] = _stats(A - pop)
                vr[f'{popk}__test_variation_value_minus_pop'] = _stats(val - pop)
            vr['C_same_minus_B_same'] = _stats([r['views'][v]['C_same'] - r['views'][v]['B_same'] for r in rs])
            vr['C_nested_minus_D_nested'] = _stats([r['views'][v]['C_nested'] - r['views'][v]['D_nested'] for r in rs])
            vr['D_nested_unseen_test_share'] = _stats([r['views'][v]['D_nested_unseen_test_share'] for r in rs])
            vr['D_bank_unseen_score_share'] = _stats([r['views'][v]['D_bank_unseen_score_share'] for r in rs])
            vr['D_bank_val_auc'] = _stats([r['views'][v]['D_bank_val_auc'] for r in rs])
            cres['views'][v] = vr
        for x, y in law['comparisons']:
            name = f'{x}-{y}'
            dA = law['views'][x]['A'] - law['views'][y]['A']
            null = abs(dA) <= TOL0
            cr = {'dA': dA, 'null': null}
            for e in EST:
                trip = np.array([r['cmp'][name][e] for r in rs], dtype=np.float64)
                ok = np.isfinite(trip).all(1)
                d, se, se_p = trip[ok, 0], trip[ok, 1], trip[ok, 2]
                R = int(ok.sum())
                st = _stats(d, dA)
                st['undefined'] = int((~ok).sum())
                st['P_d_pos'], st['P_d_neg'], st['P_d_zero'] = (_rate(int((d > 0).sum()), R), _rate(int((d < 0).sum()), R),
                                                                _rate(int((d == 0).sum()), R))
                for tag, s_ in (('R2SE', se), ('R2SE_person_se', se_p)):
                    up, dn = (d - 2 * s_ > 0), (d + 2 * s_ < 0)
                    rr = {'resolved_pos': _rate(int(up.sum()), R), 'resolved_neg': _rate(int(dn.sum()), R)}
                    if null:
                        rr['false_positive'] = _rate(int((up | dn).sum()), R)
                    else:
                        right = up if dA > 0 else dn
                        wrong = dn if dA > 0 else up
                        rr['power'] = _rate(int(right.sum()), R)
                        rr['wrong_direction'] = _rate(int(wrong.sum()), R)
                    st[tag] = rr
                    if not sc['clustered'] and tag == 'R2SE_person_se':
                        st.pop(tag)
                st['mean_se'] = float(np.mean(se)) if R else float('nan')
                if null:
                    st['M20_spurious'] = _rate(int((np.abs(d) >= 0.020).sum()), R)
                else:
                    st['descriptive_sign_error'] = _rate(int((np.sign(d) != np.sign(dA)).sum()), R)
                cr[e] = st
            cres['comparisons'][name] = cr
        out[str(cell)] = cres
    return out


def summarize_bank(rows):
    by = defaultdict(list)
    for r in rows:
        by[r['bank_cell']].append(r)
    out = {}
    for key, rs in by.items():
        views = {}
        for v in rs[0]['views']:
            e = [r['views'][v] for r in rs]
            A = e[0]['A']
            win = [x['winner_pop_auc'] for x in e]
            best = [x['best_pop_auc_in_bank'] for x in e]
            vr = {'A': A, 'reps': len(e),
                  'winner_score_auc': _stats([x['winner_score_auc'] for x in e], A),
                  'winner_pop_auc': _stats(win, A), 'winner_val_auc': _stats([x['winner_val_auc'] for x in e], A),
                  'best_pop_auc_in_bank': _stats(best, A),
                  'decomposition': {
                      'finite_training_plus_class_loss_A_minus_best_pop': _stats([A - b for b in best]),
                      'selection_loss_best_pop_minus_winner_pop': _stats([b - w for b, w in zip(best, win)]),
                      'winners_curse_val_minus_winner_pop': _stats([x['winner_val_auc'] - x['winner_pop_auc'] for x in e]),
                      'test_variation_score_minus_winner_pop': _stats([x['winner_score_auc'] - x['winner_pop_auc'] for x in e])},
                  'winners': dict(sorted(((w, sum(x['winner'] == w for x in e)) for w in {x['winner'] for x in e}),
                                         key=lambda t: -t[1])),
                  'near_tie_rate': float(np.mean([x['near_tie'] for x in e])),
                  'recipe_pop_auc_mean': {n: float(np.mean([x['recipes'][n]['pop_auc'] for x in e])) for n in e[0]['recipes']},
                  'recipe_val_auc_mean': {n: float(np.mean([x['recipes'][n]['val_auc'] for x in e])) for n in e[0]['recipes']}}
            views[v] = vr
        out[key] = {'reps': len(rs), 'views': views}
    return out


# ---------------------------------------------------------------- tables and figures
def _f(x, d=4):
    return '' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{d}f}'


def tables(S):
    lines = ['# EDG1 tables (generated by edg.analyze; synthetic known laws only)', '']
    lines += ['## 1. Estimator bias against the population optimum A (mean over repetitions; MCSE in brackets)', '',
              '| cell | family | pi | occ | n | view | support | A | C_same bias | C_nested bias | D_nested bias | D_bank bias | B_indep bias |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for cell, c in S['sim'].items():
        sc = c['scenario']
        for v, vr in c['views'].items():
            cells = [f"{_f(vr[e].get('bias'))} ({_f(vr[e].get('mcse'))})" for e in ('C_same', 'C_nested', 'D_nested', 'D_bank', 'B_indep')]
            lines.append(f"| {cell}{'*' if sc['clustered'] else ''} | {sc['family']} | {sc['pi']} | {sc['occ']} | {sc['n']} | {v} | "
                         f"{vr['support']} | {_f(vr['A'])} | " + ' | '.join(cells) + ' |')
    lines += ['', '`*` clustered household cell. MCSE = replicate SD / sqrt(R).', '',
              '## 2. Paired comparisons: mean estimated difference, false-positive / wrong-direction rate under R2SE', '',
              '| cell | family | pi | occ | n | comparison | dA | estimator | mean d | SD | P(d>0) | R2SE FP or wrong-dir | power | M20 |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for cell, c in S['sim'].items():
        sc = c['scenario']
        for name, cr in c['comparisons'].items():
            for e in EST:
                st = cr[e]
                rr = st.get('R2SE', {})
                bad = rr.get('false_positive') if cr['null'] else rr.get('wrong_direction')
                lines.append(f"| {cell} | {sc['family']} | {sc['pi']} | {sc['occ']} | {sc['n']} | {name} | {_f(cr['dA'])} | {e} | "
                             f"{_f(st.get('mean'))} | {_f(st.get('sd'))} | {_f(st['P_d_pos']['rate'], 3)} | "
                             f"{_f(bad['rate'], 3) if bad else ''} | {_f(rr['power']['rate'], 3) if 'power' in rr else ''} | "
                             f"{_f(st['M20_spurious']['rate'], 3) if 'M20_spurious' in st else ''} |")
    if S.get('bank'):
        lines += ['', '## 3. Reader-bank subset (n = 8,000: FIT 4,000 / VALIDATION 1,600 / SCORE 2,400)', '',
                  '| bank cell | view | reps | A | winner SCORE AUC | winner pop AUC | best pop in bank | A - best pop | selection loss | '
                  "winner's curse | test variation (SD) |", '|---|---|---|---|---|---|---|---|---|---|---|']
        for key, b in S['bank'].items():
            for v, vr in b['views'].items():
                d = vr['decomposition']
                lines.append(f"| {key} | {v} | {vr['reps']} | {_f(vr['A'])} | {_f(vr['winner_score_auc']['mean'])} | "
                             f"{_f(vr['winner_pop_auc']['mean'])} | {_f(vr['best_pop_auc_in_bank']['mean'])} | "
                             f"{_f(d['finite_training_plus_class_loss_A_minus_best_pop']['mean'])} | "
                             f"{_f(d['selection_loss_best_pop_minus_winner_pop']['mean'])} | "
                             f"{_f(d['winners_curse_val_minus_winner_pop']['mean'])} | "
                             f"{_f(d['test_variation_score_minus_winner_pop']['mean'])} ({_f(d['test_variation_score_minus_winner_pop']['sd'])}) |")
    return '\n'.join(lines) + '\n'


def figures(S):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig_dir = REPORT / 'figures'
    fig_dir.mkdir(exist_ok=True)
    sim = S['sim']

    def find(fam, pi, occ, n, clustered=False):
        for c in sim.values():
            sc = c['scenario']
            if (sc['family'], sc['pi'], sc['occ'], sc['n'], sc['clustered']) == (fam, pi, occ, n, clustered):
                return c
    # Figure 1: bias of the same-sample and nested empirical optimum vs observable support (pi = 0.5), IND and REF
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, fam, vmap in ((axes[0], 'IND', {'income': 16, 'occupation': 64, 'pair': 1024}),
                          (axes[1], 'REF', {'L16': 16, 'L64': 64, 'L256': 256, 'L1024': 1024})):
        for occ, ls in (('skewed', '-'), ('uniform', ':')):
            for n, col in ((2000, 'tab:red'), (8000, 'tab:orange'), (20000, 'tab:blue')):
                c = find(fam, 0.5, occ, n)
                if c is None:
                    continue
                xs = sorted(vmap.values())
                inv = {k: v for v, k in vmap.items()}
                ax.plot(xs, [c['views'][inv[x]]['C_same']['bias'] for x in xs], ls, color=col, marker='o', ms=3,
                        label=f'C same-sample, n={n:,} ({occ})')
            c = find(fam, 0.5, occ, 20000)
            if c is not None:
                ax.plot(xs, [c['views'][inv[x]]['C_nested']['bias'] for x in xs], ls, color='black', marker='s', ms=3,
                        label=f'C nested (5 folds of 4,000) ({occ})')
                ax.plot(xs, [c['views'][inv[x]]['D_nested']['bias'] for x in xs], ls, color='tab:green', marker='^', ms=3,
                        label=f'D nested LOOKUP ({occ})')
        ax.axhline(0, color='grey', lw=0.8)
        ax.set_xscale('log', base=2)
        ax.set_xlabel('observable output support (distinct wires)')
        ax.set_title({'IND': 'Independence (A = 0.5 at every support)', 'REF': 'Irrelevant refinement (A identical at every level)'}[fam])
    axes[0].set_ylabel('mean estimate - population optimum A')
    axes[1].legend(fontsize=6.5, loc='upper left')
    fig.suptitle('EDG1: bias of empirical optimal AUC grows with support and shrinks with n; LOOKUP is biased low (pi = 0.5)', fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / 'fig1_bias_vs_support.png', dpi=150)
    plt.close(fig)
    # Figure 2: decomposition of the nested-exact-to-LOOKUP gap at DRF1-like sizes (n = 20,000, pi = 0.5, skewed)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    labels, opt, loss, tv = [], [], [], []
    for fam in L.FAMILIES:
        c = find(fam, 0.5, 'skewed', 20000)
        if c is None:
            continue
        for v, vr in c['views'].items():
            labels.append(f'{fam}\n{v} ({vr["support"]})')
            opt.append(vr['C_nested']['bias'])
            loss.append(-vr['D_nested_pop_pooled__estimation_loss_A_minus_pop']['mean'])
            tv.append(vr['D_nested_pop_pooled__test_variation_value_minus_pop']['mean'])
    x = np.arange(len(labels))
    ax.bar(x - 0.25, opt, 0.25, label='C nested - A (optimism of nested exact AUC*)', color='tab:red')
    ax.bar(x, loss, 0.25, label='pop(LOOKUP) - A (fitted LOOKUP estimation loss)', color='tab:green')
    ax.bar(x + 0.25, tv, 0.25, label='LOOKUP sample - pop (test variation + cross-fit pooling)', color='tab:grey')
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('AUC difference')
    ax.legend(fontsize=7)
    ax.set_title('EDG1: the nested-exact minus LOOKUP gap = optimism + reader loss (+ zero-mean test variation); n = 20,000, pi = 0.5, skewed', fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / 'fig2_gap_decomposition.png', dpi=150)
    plt.close(fig)


def main(argv=None):
    S = {'schema': 'EDG1-summary-v1', 'sim': summarize_sim(_load('sim_*.jsonl.gz'))}
    bank_rows = _load('bank_*.jsonl.gz')
    if bank_rows:
        S['bank'] = summarize_bank(bank_rows)
        runs = sorted(glob.glob(str(OUT / 'bank_*.run.json')))
        if runs:
            S['bank_class_limits'] = json.load(open(runs[-1]))['class_limits']
    with open(REPORT / 'SUMMARY.json', 'w') as fh:
        json.dump(S, fh, indent=1, allow_nan=True)
    with open(REPORT / 'TABLES.md', 'w') as fh:
        fh.write(tables(S))
    figures(S)
    print('cells', len(S['sim']), 'bank cells', len(S.get('bank', {})))
