"""Locked-score report for the 2017 transport study. Fits nothing, selects nothing.

Consumes the completed Mode A/B metrics and cached per-person predictions,
recomputes every selected loss from predictions and labels (score replay),
runs the prespecified household bootstrap, and writes compact aggregates,
decisions and figures. Person-level arrays stay local.
"""
from __future__ import annotations
import argparse, gzip, hashlib, io, itertools, json, statistics, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import acs_spectral_transport_eval as ev

WEIGHTS = ('unweighted', 'person_weighted')
SCORE = {'unweighted': 'test', 'person_weighted': 'test_person_weighted'}
AUDIT_ROLES = {'A': ('public_coverage', 'commute_over20', 'SEX', 'RAC1P'),
               'B': ('income_binary', 'civilian_at_work', 'same_residence', 'SEX', 'RAC1P'), 'AB': ('SEX', 'RAC1P')}
FORBIDDEN = tuple(v+'/'+t for v, ts in AUDIT_ROLES.items() for t in ts)
SENSITIVE = tuple(v+'/'+t for v in ('A', 'B', 'AB') for t in ('SEX', 'RAC1P'))
FAMILY_SENSITIVE = ('A/SEX', 'AB/SEX', 'A/RAC1P', 'AB/RAC1P')
UTILITY = ('income_binary', 'civilian_at_work', 'public_coverage', 'same_residence', 'commute_over20')
UTILITY_VIEW = {'income_binary': 'A', 'civilian_at_work': 'A', 'same_residence': 'A', 'public_coverage': 'B', 'commute_over20': 'B'}
SOURCE = ('income_binary', 'civilian_at_work', 'public_coverage')
MODE_SCOPES = {'A': ('standard_independent', 'expanded_independent', 'expanded_catchup', 'kernel_standard_independent',
                     'kernel_expanded_independent', 'kernel_expanded_catchup'),
               'B': tuple(ev.MODE_B_SCOPES)}
PRIMARY = {'A': ('kernel_expanded_catchup', 360), 'B': ('transport_all', 360)}
BOOT_KEYS = (('A', 'kernel_expanded_catchup', 360), ('B', 'transport_all', 360), ('B', 'common_fresh', 360))
P_VALUES = (0., .25, .5, .75, 1.)
LABEL = {c: c.replace('spectral_', 'S:') for c in ev.INTERFACES}


def read(p): return json.loads(Path(p).read_text())


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json(p, v):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(v, indent=2, allow_nan=False, default=float)+'\n')


def csv_gz(path, frame):
    buf = io.StringIO(); frame.to_csv(buf, index=False)
    with open(path, 'wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as z:
        z.write(buf.getvalue().encode())


def finite(x): return x is not None and np.isfinite(x)


def person_loss(y, p):
    p = np.clip(np.asarray(p, float), 1e-12, 1.); p /= p.sum(1, keepdims=True)
    return -np.log(p[np.arange(len(y)), y])


class Predictions:
    def __init__(self, path):
        with np.load(path) as z:
            ids = dict(zip(z['keys'].tolist(), z['ids'].tolist()))
            self.arrays = {k: z['p/'+h] for k, h in ids.items()}

    def __getitem__(self, k): return self.arrays[k]


# ------------------------------------------------------------------ loading
def load(out, seeds):
    out = Path(out)
    lab = np.load(out/'final_labels.npz')
    labels = {t: lab['y/'+t] for t in ev.TARGETS}; weights = lab['weights']; serial = lab['serialno']
    points = {}; losses = {}; allrows = []; replay = {'checked': 0, 'max_abs_difference': 0.}
    context = {}
    for s in seeds:
        cm = read(out/f'seed_{s}'/'context'/'metrics.json')['rows']; cp = Predictions(out/f'seed_{s}'/'context'/'predictions.npz')
        ref = read(out/f'seed_{s}'/'reference'/'selection.json')['reference_probes']
        ctx = {'prior': {}, 'reference': {}, 'service': {}}
        for r in cm:
            t = r['target']; valid = labels[t] >= 0
            if r['kind'] == 'prior':
                key = f"{r['mode']}/prior/{t}"; ctx['prior'][r['mode'], t] = r['scores']
                losses[s, r['mode'], 'prior', t] = person_loss(labels[t][valid], cp[key])
            elif r['kind'] == 'service':
                ctx['service'][t] = r['scores']
                losses[s, 'service', t] = person_loss(labels[t][valid], cp[f'service/{t}'])
            elif r['selected']:
                ctx['reference'][r['mode'], r['release'], t] = r['scores']
                key = f"B/reference/{r['release']}/{t}/{r['candidate_id']}" if r['mode'] == 'B' else f"A/reference/{r['release']}/{t}/{r['candidate_id']}"
                losses[s, r['mode'], 'reference', r['release'], t] = person_loss(labels[t][valid], cp[key])
        context[s] = ctx
        for c in ev.INTERFACES:
            for mode in ('A', 'B'):
                d = out/f'seed_{s}'/c/f'mode_{mode}'
                rows = read(d/'metrics.json')['raw_metrics']; pred = Predictions(d/'predictions.npz')
                util = {}
                for r in rows:
                    if r['role'] == 'utility' and r['selected']:
                        t = r['target']; valid = labels[t] >= 0
                        l = person_loss(labels[t][valid], pred[f"utility/{r['view']}/{t}/{r['candidate_id']}"])
                        losses[s, mode, c, 'utility', t] = l; util[t] = r['scores']
                        replay['checked'] += 1
                        replay['max_abs_difference'] = max(replay['max_abs_difference'], abs(l.mean()-r['scores']['test']['log_loss']))
                for r in rows:
                    base = {'seed': s, 'mode': mode, 'condition': c, 'role': r['role'], 'view': r['view'], 'target': r['target'],
                            'budget': r.get('audit_budget'), 'candidate_id': r['candidate_id'],
                            'origin': r.get('transport_origin', r.get('candidate_origin')), 'family': r.get('family'),
                            'selected_scopes': ';'.join(r.get('selected_scopes', ['utility'] if r.get('selected') else [])),
                            'validation_log_loss_2017': r.get('validation_log_loss')}
                    for w in WEIGHTS:
                        sc = r['scores'][SCORE[w]]
                        allrows.append({**base, 'weight': w, 'log_loss': sc['log_loss'], 'accuracy': sc.get('accuracy'),
                                        'macro_auroc': sc.get('macro_auroc'), 'coverage_complete': sc.get('coverage_complete')})
                for scope in MODE_SCOPES[mode]:
                    for b in (120, 360):
                        sel = {}
                        for r in rows:
                            if r['role'] == 'audit' and r['audit_budget'] == b and scope in r['selected_scopes']:
                                e = r['view']+'/'+r['target']
                                if e in sel: raise ValueError(f'duplicate selection {s} {c} {mode} {scope} {e}')
                                sel[e] = r
                        if set(sel) != set(FORBIDDEN): raise ValueError(f'incomplete selection {s} {c} {mode} {scope} {b}')
                        for e, r in sel.items():
                            t = r['target']; valid = labels[t] >= 0
                            key = f"audit/{b}/{e}/{r['candidate_id']}"
                            l = person_loss(labels[t][valid], pred[key])
                            if (mode, scope, b) in BOOT_KEYS: losses[s, mode, c, scope, b, e] = l
                            if True:
                                replay['checked'] += 1
                                replay['max_abs_difference'] = max(replay['max_abs_difference'], abs(l.mean()-r['scores']['test']['log_loss']))
                        for w in WEIGHTS:
                            points[s, mode, c, scope, b, w] = {
                                'utility': {t: util[t][SCORE[w]]['log_loss'] for t in UTILITY},
                                'losses': {e: sel[e]['scores'][SCORE[w]]['log_loss'] for e in FORBIDDEN},
                                'gains': {e: ctx['prior'][mode, e.split('/')[1]][SCORE[w]]['log_loss']-sel[e]['scores'][SCORE[w]]['log_loss'] for e in FORBIDDEN},
                                'selected': {e: sel[e]['candidate_id'] for e in FORBIDDEN},
                                'coverage': {e: sel[e]['scores'][SCORE[w]].get('coverage_complete') for e in FORBIDDEN}}
    return {'labels': labels, 'weights': weights, 'serial': serial, 'points': points, 'losses': losses,
            'rows': pd.DataFrame(allrows), 'context': context, 'replay': replay}


# ---------------------------------------------------------------- bootstrap
class Bootstrap:
    def __init__(self, serial, weights, labels, replicates=2000, seed=20260917):
        codes, inv = np.unique(serial, return_inverse=True)
        self.groups = len(codes); self.replicates = replicates
        draws = np.random.default_rng(seed).integers(0, self.groups, size=(replicates, self.groups))
        counts = np.zeros((replicates, self.groups), np.float64)
        for b in range(replicates): counts[b] = np.bincount(draws[b], minlength=self.groups)
        self.rows = counts[:, inv]; self.weights = weights; self.labels = labels; self.cache = {}

    def means(self, target, weight, vectors):
        """vectors: (n_valid, k) per-person losses -> (replicates, k) and point (k,)."""
        key = (target, weight)
        if key not in self.cache:
            valid = self.labels[target] >= 0
            r = self.rows[:, valid]; w = self.weights[valid] if weight == 'person_weighted' else np.ones(int(valid.sum()))
            rw = r*w; self.cache[key] = (rw, rw.sum(1), w)
        rw, denom, w = self.cache[key]
        return (rw @ vectors)/denom[:, None], (w @ vectors)/w.sum()


def effective_counts(data):
    out = {'final_households': int(len(np.unique(data['serial']))), 'final_rows': int(len(data['serial']))}
    for t, y in data['labels'].items():
        v = y >= 0; w = data['weights'][v]
        out[t] = {'valid_rows': int(v.sum()), 'households': int(len(np.unique(data['serial'][v]))),
                  'kish_effective_n_pwgtp': float(w.sum()**2/(w**2).sum()),
                  'class_support': np.bincount(y[v], minlength=ev.old.CLASSES[t]).tolist()}
    return out


def endpoint_vectors(data, s, mode, c, scope, b, endpoint):
    """Per-person loss for utility/<task> or recovery/<view>/<target> (prior minus attack)."""
    L = data['losses']
    kind, rest = endpoint.split('/', 1)
    if kind == 'utility': return rest, L[s, mode, c, 'utility', rest], 1.
    t = rest.split('/')[1]
    return t, L[s, mode, 'prior', t]-L[s, mode, c, scope, b, rest], 1.


def contrast_replicates(boot, data, seeds, mode, scope, b, left, right, endpoint, weight):
    """Seed-mean paired difference: bootstrap replicates and point estimate, plus per-seed points."""
    reps, pts, per = 0., 0., {}
    for s in seeds:
        t, a, _ = endpoint_vectors(data, s, mode, left, scope, b, endpoint)
        _, z, _ = endpoint_vectors(data, s, mode, right, scope, b, endpoint)
        r, p = boot.means(t, weight, np.column_stack((a, z)))
        reps = reps+(r[:, 0]-r[:, 1])/len(seeds); pts = pts+(p[0]-p[1])/len(seeds); per[s] = float(p[0]-p[1])
    return reps, float(pts), per


def adjusted_family(boot, data, seeds, spec_family):
    mode, scope, b = spec_family['mode'], spec_family['scope'], spec_family['budget']
    rows, reps = [], []
    for left, right in spec_family['contrasts']:
        for endpoint in spec_family['endpoints']:
            for w in WEIGHTS:
                r, p, per = contrast_replicates(boot, data, seeds, mode, scope, b, left, right, endpoint, w)
                rows.append({'left': left, 'right': right, 'endpoint': endpoint, 'weight': w, 'estimate': p,
                             **{f'seed_{s}': v for s, v in per.items()}, 'se': float(np.std(r, ddof=1)),
                             'unadjusted_low': float(np.quantile(r, .025)), 'unadjusted_high': float(np.quantile(r, .975))})
                reps.append(r)
    est = np.array([r['estimate'] for r in rows])
    low, high, se, crit = simultaneous(np.column_stack(reps), est)
    for r, lo, hi, s_ in zip(rows, low, high, se):
        r.update(degenerate=not s_ > 1e-15, adjusted_low=float(lo), adjusted_high=float(hi))
    return rows, crit


def simultaneous(R, est, level=.95):
    """Single-step studentized max-|t| intervals; zero-SE endpoints are degenerate and excluded."""
    se = np.std(R, axis=0, ddof=1); live = se > 1e-15
    if not live.any(): return est.copy(), est.copy(), se, float('nan')
    T = np.max(np.abs(R[:, live]-est[live])/se[live], axis=1)
    crit = float(np.quantile(T, level, method='higher'))
    return np.where(live, est-crit*se, est), np.where(live, est+crit*se, est), se, crit


def advantage(rows, left, right, weight):
    sub = [r for r in rows if r['left'] == left and r['right'] == right and r['weight'] == weight and r['endpoint'].startswith('recovery/')]
    better = [r['endpoint'] for r in sub if r['adjusted_high'] < 0]
    worse = [r['endpoint'] for r in sub if r['adjusted_low'] > 0]
    res = [r for r in rows if r['left'] == left and r['right'] == right and r['weight'] == weight and r['endpoint'] == 'utility/same_residence'][0]
    return {'advantage': bool(better) and not worse, 'significantly_better': better, 'significantly_worse': worse,
            'residence_difference': res['estimate'], 'residence_adjusted': [res['adjusted_low'], res['adjusted_high']],
            'local_race_significantly_worse': 'recovery/A/RAC1P' in worse}


# --------------------------------------------------------------- criteria
def directional(left, right, delta, leak=1e-12):
    du = {t: left['utility'][t]-right['utility'][t] for t in UTILITY}
    dg = {e: left['gains'][e]-right['gains'][e] for e in SENSITIVE}
    u = all(v <= delta+1e-12 for v in du.values()); g = all(v <= leak for v in dg.values())
    strict = any(v < -1e-12 for v in du.values()) or any(v < -1e-12 for v in dg.values())
    return {'utility_no_worse': u, 'sensitive_no_worse': g, 'strict': strict, 'dominates': u and g and strict,
            'utility_differences': du, 'sensitive_differences': dg}


def dev_gate(points, seeds, mode, scope, left, right):
    rows = [directional(points[s, mode, left, scope, 360, w], points[s, mode, right, scope, 360, w], .001, .0005+1e-12) | {'seed': s, 'weight': w}
            for s in seeds for w in WEIGHTS]
    vector = all(r['utility_no_worse'] and r['sensitive_no_worse'] for r in rows)
    residence = all(r['utility_differences']['same_residence'] < -.001 for r in rows)
    shared = [e for e in SENSITIVE if all(r['sensitive_differences'][e] < -.001 for r in rows)]
    return {'pass': vector and (residence or bool(shared)), 'vector_nonworsening': vector, 'strict_residence': residence, 'shared_strict_roles': shared}


def criteria(data, seeds):
    P, ctx = data['points'], data['context']; out = []
    for mode in ('A', 'B'):
        scope, b = PRIMARY[mode]
        for c, s, w in itertools.product(ev.INTERFACES, seeds, WEIGHTS):
            pt = P[s, mode, c, scope, b, w]; ref = lambda rel, t: ctx[s]['reference'][mode, rel, t][SCORE[w]]['log_loss']
            pca_res, bank = ref('E_pca', 'same_residence'), min(ref('B_rich_bank', 'same_residence'), ref('C_tree_bank', 'same_residence'))
            h = P[s, mode, 'H', scope, b, w]
            out.append({'mode': mode, 'condition': c, 'seed': s, 'weight': w,
                        'residence_gain_vs_H': h['utility']['same_residence']-pt['utility']['same_residence'],
                        'residence_loss': pt['utility']['same_residence'], 'pca32_residence_loss': pca_res, 'best_bank_residence_loss': bank,
                        'half_headroom_pass': bool(pt['utility']['same_residence'] <= (pca_res+bank)/2+1e-12),
                        **{f'source_excess_{t}': pt['utility'][t]-ref('E_pca', t) for t in SOURCE},
                        'source_allowance_pass': all(pt['utility'][t] <= ref('E_pca', t)+.01+1e-12 for t in SOURCE)})
    return pd.DataFrame(out)


# ------------------------------------------------------------- withholding
def branch_uniforms(serial, sporder, seed, condition):
    return np.array([int.from_bytes(hashlib.sha256(f'spectral-transport-withholding-v1|2017|{seed}|{condition}|{a}|{b}'.encode()).digest()[:8], 'big')/2**64
                     for a, b in zip(serial, sporder)])


def withholding(data, seeds, sporder):
    P, L, rows, checks = data['points'], data['losses'], [], []
    w_all = data['weights']
    for mode, scope, b in (('A', *PRIMARY['A']), ('B', *PRIMARY['B']), ('B', 'common_fresh', 360)):
        for src, s, p in itertools.product(('E', 'A0', 'L025', 'L20', 'J', 'spectral_S0'), seeds, P_VALUES):
            for w in WEIGHTS:
                h, a = P[s, mode, 'H', scope, b, w], P[s, mode, src, scope, b, w]
                mixed = {'utility': {t: (1-p)*h['utility'][t]+p*a['utility'][t] for t in UTILITY},
                         'gains': {e: (1-p)*h['gains'][e]+p*a['gains'][e] for e in FORBIDDEN}}
                P[s, mode, f'W[{src},{p}]', scope, b, w] = mixed
                rows.extend({'mode': mode, 'scope': scope, 'budget': b, 'source': src, 'p': p, 'seed': s, 'weight': w,
                             'kind': k, 'endpoint': e, 'value': v} for k, d in (('utility_loss', mixed['utility']), ('absolute_recovery', mixed['gains'])) for e, v in d.items())
        # per-person expected-loss arithmetic and one realized routing, per protocol
        for src in ('J', 'spectral_S0'):
            s = seeds[0]; y = data['labels']['SEX']; valid = y >= 0
            lh, la = L[s, mode, 'H', scope, b, 'AB/SEX'], L[s, mode, src, scope, b, 'AB/SEX']
            u = branch_uniforms(data['serial'][valid], sporder[valid], s, src)
            for p in P_VALUES:
                for w in WEIGHTS:
                    ww = w_all[valid] if w == 'person_weighted' else np.ones(int(valid.sum()))
                    expected = float(ww @ ((1-p)*lh+p*la)/ww.sum()); realized = float(ww @ np.where(u < p, la, lh)/ww.sum())
                    linear = (1-p)*float(ww @ lh/ww.sum())+p*float(ww @ la/ww.sum())
                    checks.append({'mode': mode, 'scope': scope, 'source': src, 'seed': s, 'p': p, 'weight': w, 'endpoint': 'AB/SEX attack loss',
                                   'expected_per_person_mixture': expected, 'linear_combination': linear, 'abs_difference': abs(expected-linear),
                                   'one_realized_routing': realized, 'realized_fraction_augmented': float(np.mean(u < p))})
    return pd.DataFrame(rows), checks


def withholding_dominance(P, seeds):
    out = []
    for mode, scope in (('A', PRIMARY['A'][0]), ('B', PRIMARY['B'][0]), ('B', 'common_fresh')):
        for arm, src, p in itertools.product(ev.SPECTRAL, ('E', 'A0', 'L025', 'L20', 'J', 'spectral_S0'), P_VALUES):
            rows = [(directional(P[s, mode, f'W[{src},{p}]', scope, 360, w], P[s, mode, arm, scope, 360, w], .001),
                     directional(P[s, mode, arm, scope, 360, w], P[s, mode, f'W[{src},{p}]', scope, 360, w], .001)) for s in seeds for w in WEIGHTS]
            out.append({'mode': mode, 'scope': scope, 'spectral_arm': arm, 'withholding_source': src, 'p': p,
                        'withholding_dominates_count': sum(a['dominates'] for a, _ in rows), 'spectral_dominates_count': sum(z['dominates'] for _, z in rows),
                        'comparisons': len(rows), 'withholding_dominates_all': all(a['dominates'] for a, _ in rows)})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ figures
COLORS = {'hist': '#2a78d6', 'spec': '#eb6834', 'with': '#1baf7a', 'H': '#8a8984'}


def figures(out, agg, withdf, fam, dev, seeds):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#52514e',
                         'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.grid': True,
                         'grid.color': '#e4e3df', 'grid.linewidth': .5})
    figdir = Path(out)/'figures'; figdir.mkdir(exist_ok=True); made = []
    def color(c): return COLORS['H'] if c == 'H' else COLORS['spec'] if c.startswith('spectral_') else COLORS['hist']
    def val(mode, scope, c, s, w, kind, e):
        r = agg[(agg['mode'] == mode) & (agg['scope'] == scope) & (agg['condition'] == c) & (agg['weight'] == w) & (agg['kind'] == kind) & (agg['endpoint'] == e)]
        return float(r['seed_mean'].iloc[0]) if s == 'mean' else float(r[f'seed_{s}'].iloc[0])
    for name, e in (('residence_vs_coalition_sex', 'AB/SEX'), ('residence_vs_coalition_race', 'AB/RAC1P'), ('residence_vs_individual_race', 'A/RAC1P')):
        for mode in ('B', 'A'):
            scope = PRIMARY[mode][0]
            fig, axes = plt.subplots(2, 4, figsize=(11, 5.4), sharex=False, sharey=False)
            for row, w in enumerate(WEIGHTS):
                for col, s in enumerate([*seeds, 'mean']):
                    ax = axes[row, col]
                    for src in ('J', 'spectral_S0'):
                        sub = withdf[(withdf['mode'] == mode) & (withdf['scope'] == scope) & (withdf['source'] == src) & (withdf['weight'] == w)]
                        xs, ys = [], []
                        for p in P_VALUES:
                            q = sub[sub['p'] == p]
                            if s == 'mean': q = q.groupby(['kind', 'endpoint'])['value'].mean().reset_index()
                            else: q = q[q['seed'] == s]
                            hq = withdf[(withdf['mode'] == mode) & (withdf['scope'] == scope) & (withdf['source'] == src) & (withdf['weight'] == w) & (withdf['p'] == 0)]
                            if s != 'mean': hq = hq[hq['seed'] == s]
                            hres = hq[(hq['kind'] == 'utility_loss') & (hq['endpoint'] == 'same_residence')]['value'].mean()
                            hgain = hq[(hq['kind'] == 'absolute_recovery') & (hq['endpoint'] == e)]['value'].mean()
                            xs.append(hres-q[(q['kind'] == 'utility_loss') & (q['endpoint'] == 'same_residence')]['value'].mean())
                            ys.append(q[(q['kind'] == 'absolute_recovery') & (q['endpoint'] == e)]['value'].mean()-hgain)
                        ax.plot(xs, ys, color=COLORS['with'], lw=1.2, ls='-' if src == 'J' else '--', zorder=1)
                        ax.annotate(f'withhold {LABEL[src]}', (xs[-1], ys[-1]), fontsize=6, color='#52514e', xytext=(2, -8), textcoords='offset points')
                    for c in ev.INTERFACES:
                        x = val(mode, scope, c, s, w, 'utility_gain_vs_H', 'same_residence'); y = val(mode, scope, c, s, w, 'additional_recovery', e)
                        ax.scatter(x, y, s=22, color=color(c), edgecolor='white', linewidth=.8, zorder=3)
                        if c in ('H', 'J', 'L20', 'spectral_C1', 'spectral_L1', 'spectral_L2', 'spectral_S0'):
                            ax.annotate(LABEL[c], (x, y), fontsize=6, xytext=(3, 3), textcoords='offset points', color='#0b0b0b')
                    ax.axhline(0, color='#8a8984', lw=.6); ax.axvline(.01, color='#8a8984', lw=.6, ls=':')
                    ax.set_title(f"{'seed '+str(s) if s != 'mean' else 'three-seed mean'} · {'PWGTP' if w == 'person_weighted' else 'unweighted'}", fontsize=7)
                    if col == 0: ax.set_ylabel(f'additional {e} recovery (nats)')
                    if row == 1: ax.set_xlabel('residence gain over H (nats)')
            fig.suptitle(f'Mode {mode} ({scope}, budget 360): residence gain vs additional {e} recovery. Orange spectral, blue neural, gray H, green withholding (p=0..1). Dotted line: .01 reference.', fontsize=8)
            fig.tight_layout(); p = figdir/f'{name}_mode{mode}'
            for ext in ('png', 'pdf'): fig.savefig(f'{p}.{ext}', dpi=200)
            plt.close(fig); made.append(str(p.relative_to(out))+'.png')
    # paired primary comparisons: one panel per family, each with its own labels
    fig, axes = plt.subplots(1, 4, figsize=(15, 9), gridspec_kw={'width_ratios': [1, 1, .6, 1.8]})
    for ax, fname in zip(axes, ('F1_primary', 'F2_neural_replication', 'F3_secondary', 'F4_frozen_transfer')):
        rows = fam[fname]['rows']; ticks = []
        keys = list(dict.fromkeys((r['left'], r['right'], r['endpoint']) for r in rows))
        for i, key in enumerate(keys):
            yv = len(keys)-i
            for r in rows:
                if (r['left'], r['right'], r['endpoint']) != key: continue
                off = .15 if r['weight'] == 'person_weighted' else -.15
                ax.plot([r['adjusted_low'], r['adjusted_high']], [yv+off]*2, color=COLORS['spec'] if r['left'].startswith('spectral') else COLORS['hist'], lw=1.4)
                ax.scatter(r['estimate'], yv+off, s=12, color='#0b0b0b', marker='o' if r['weight'] == 'unweighted' else 'D', zorder=3)
            ticks.append((yv, f"{LABEL[key[0]]} − {LABEL[key[1]]}: {key[2].replace('recovery/', 'rec. ').replace('utility/', 'loss ')}"))
        ax.set_yticks([y for y, _ in ticks]); ax.set_yticklabels([t for _, t in ticks], fontsize=6)
        ax.axvline(0, color='#8a8984', lw=.7); ax.set_title(fname.replace('_', ' ')+f" (c = {fam[fname]['critical']:.2f})", fontsize=8)
        ax.set_xlabel('paired difference (nats); < 0 favors left')
    fig.suptitle('Simultaneous 95% intervals (single-step max-|t| household bootstrap, 2,000 replicates). Circle: unweighted; diamond: PWGTP. Mode B except F4 (Mode A).', fontsize=8)
    fig.tight_layout(); p = figdir/'paired_primary_comparisons'
    for ext in ('png', 'pdf'): fig.savefig(f'{p}.{ext}', dpi=200)
    plt.close(fig); made.append(str(p.relative_to(out))+'.png')
    # development vs transport
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, (kind, e, title) in zip(axes, (('utility_gain_vs_H', 'same_residence', 'residence gain vs H'), ('additional_recovery', 'AB/SEX', 'additional AB/SEX recovery'), ('additional_recovery', 'AB/RAC1P', 'additional AB/RAC1P recovery'))):
        xs, ys = [], []
        for c in ev.INTERFACES:
            d = dev.get((c, kind, e));
            if d is None: continue
            y = val('B', PRIMARY['B'][0], c, 'mean', 'unweighted', kind, e)
            ax.scatter(d, y, s=22, color=color(c), edgecolor='white', linewidth=.8, zorder=3); xs.append(d); ys.append(y)
            if c in ('J', 'spectral_C1', 'spectral_L2', 'spectral_S0', 'L20'): ax.annotate(LABEL[c], (d, y), fontsize=6, xytext=(3, 3), textcoords='offset points')
        lo, hi = min(xs+ys), max(xs+ys); ax.plot([lo, hi], [lo, hi], color='#8a8984', lw=.7, ls='--')
        ax.set_xlabel('2018 development (three-seed mean)'); ax.set_ylabel('2017 transport Mode B (three-seed mean)'); ax.set_title(title+' · unweighted', fontsize=8)
    fig.tight_layout(); p = figdir/'development_vs_transport'
    for ext in ('png', 'pdf'): fig.savefig(f'{p}.{ext}', dpi=200)
    plt.close(fig); made.append(str(p.relative_to(out))+'.png')
    return made


# --------------------------------------------------------------------- main
def dev_means():
    path = ev.DEV/'PUBLICATION_EVIDENCE/PER_SEED.csv.gz'
    d = pd.read_csv(path)
    d = d[(d['split'] == 'test') & (d['scope'] == 'kernel_expanded_catchup') & (d['budget'] == 360)]
    means = d.groupby(['condition', 'weight', 'kind', 'endpoint'])['value'].mean()
    return {(c, k, e): v for (c, w, k, e), v in means.items() if w == 'unweighted'}, means, sha(path)


def report(out, seeds=(0, 1, 2)):
    out = Path(out); spec_ = read(out/'COMPARISONS.json')
    data = load(out, seeds)
    lab = np.load(out/'final_labels.npz'); sporder = lab['sporder']
    boot = Bootstrap(data['serial'], data['weights'], data['labels'])
    counts = effective_counts(data)
    # per-seed long table
    P = data['points']; flat = []
    for (s, mode, c, scope, b, w), pt in P.items():
        h = P[s, mode, 'H', scope, b, w]
        kinds = {'utility_loss': pt['utility'], 'attack_loss': pt['losses'], 'absolute_recovery': pt['gains'],
                 'additional_recovery': {e: v-h['gains'][e] for e, v in pt['gains'].items()},
                 'utility_gain_vs_H': {t: h['utility'][t]-v for t, v in pt['utility'].items()}}
        for k, dct in kinds.items():
            flat.extend({'seed': s, 'mode': mode, 'condition': c, 'scope': scope, 'budget': b, 'weight': w, 'kind': k, 'endpoint': e,
                         'value': float(v), 'selected_candidate': pt['selected'].get(e) if k in ('attack_loss', 'absolute_recovery', 'additional_recovery') else None,
                         'coverage_complete': pt['coverage'].get(e)} for e, v in dct.items())
    per_seed = pd.DataFrame(flat)
    wide = per_seed.pivot_table(index=['mode', 'condition', 'scope', 'budget', 'weight', 'kind', 'endpoint'], columns='seed', values='value').reset_index()
    wide.columns = [f'seed_{c}' if isinstance(c, (int, np.integer)) else c for c in wide.columns]
    sc = [f'seed_{s}' for s in seeds]
    wide['seed_mean'] = wide[sc].mean(1); wide['seed_sd'] = wide[sc].std(1, ddof=1)
    wide['n_negative_seeds'] = (wide[sc] < -1e-12).sum(1)
    # bootstrap intervals for seed-mean gain/additional recovery at the bootstrapped scopes
    ci = []
    for (mode, scope, b), c, w in itertools.product(BOOT_KEYS, ev.INTERFACES, WEIGHTS):
        if c == 'H': continue
        for e in ['utility/'+t for t in UTILITY]+['recovery/'+x for x in FORBIDDEN]:
            r, pnt, _ = contrast_replicates(boot, data, seeds, mode, scope, b, 'H', c, e, w) if e.startswith('utility') else contrast_replicates(boot, data, seeds, mode, scope, b, c, 'H', e, w)
            ci.append({'mode': mode, 'scope': scope, 'budget': b, 'condition': c, 'weight': w,
                       'kind': 'utility_gain_vs_H' if e.startswith('utility') else 'additional_recovery', 'endpoint': e.split('/', 1)[1],
                       'boot_estimate': pnt, 'boot_low': float(np.quantile(r, .025)), 'boot_high': float(np.quantile(r, .975))})
    cidf = pd.DataFrame(ci)
    agg = wide.merge(cidf, on=['mode', 'scope', 'budget', 'condition', 'weight', 'kind', 'endpoint'], how='left')
    agg = agg[agg['budget'] == 360].copy()
    # families
    fam = {}
    for name, f in spec_['families'].items():
        rows, crit = adjusted_family(boot, data, seeds, f)
        fam[name] = {'critical': crit, 'endpoints': len(rows), 'rows': rows,
                     'decisions': {f'{l} vs {r} / {w}': advantage(rows, l, r, w) for (l, r) in f['contrasts'] for w in WEIGHTS}}
    def coordination(name, left, rights):
        d = fam[name]['decisions']
        ok = all(d[f'{left} vs {r} / {w}']['advantage'] for r in rights for w in WEIGHTS)
        res = all(d[f'{left} vs {r} / {w}']['residence_difference'] <= .001+1e-12 for r in rights for w in WEIGHTS)
        race = any(d[f'{left} vs {r} / {w}']['local_race_significantly_worse'] for r in rights for w in WEIGHTS)
        return {'supported': ok and res and not race, 'advantage_all': ok, 'residence_within_001_all': res, 'local_race_significantly_worse_any': race}
    decisions = {'primary_coordination_F1': coordination('F1_primary', 'spectral_C1', ('spectral_L1', 'spectral_L2')),
                 'neural_replication_F2': coordination('F2_neural_replication', 'J', ('L025', 'L20')),
                 'secondary_C1_vs_J_F3': coordination('F3_secondary', 'spectral_C1', ('J',)),
                 'frozen_transfer_F4': {'C1': coordination('F4_frozen_transfer', 'spectral_C1', ('spectral_L1', 'spectral_L2')),
                                        'J': coordination('F4_frozen_transfer', 'J', ('L025', 'L20'))}}
    # context contrasts (unadjusted)
    ctx_rows = []
    contrasts = [tuple(x) for x in spec_['context_contrasts']]+[tuple(x) for f in spec_['families'].values() for x in f['contrasts']]
    for (left, right), (mode, scope, b), w in itertools.product(dict.fromkeys(contrasts), BOOT_KEYS, WEIGHTS):
        for e in ['utility/'+t for t in UTILITY]+['recovery/'+x for x in FORBIDDEN]:
            r, pnt, per = contrast_replicates(boot, data, seeds, mode, scope, b, left, right, e, w)
            ctx_rows.append({'left': left, 'right': right, 'mode': mode, 'scope': scope, 'budget': b, 'weight': w, 'endpoint': e,
                             'estimate': pnt, **{f'seed_{s}': v for s, v in per.items()},
                             'unadjusted_low': float(np.quantile(r, .025)), 'unadjusted_high': float(np.quantile(r, .975)),
                             'unadjusted_excludes_zero_worse': bool(np.quantile(r, .025) > 0), 'unadjusted_excludes_zero_better': bool(np.quantile(r, .975) < 0)})
    paired = pd.DataFrame(ctx_rows)
    costs = paired[(paired['mode'] == 'B') & (paired['scope'] == 'transport_all') & paired['unadjusted_excludes_zero_worse']
                   & paired[['left', 'right']].apply(tuple, axis=1).isin([('spectral_C1', 'spectral_L1'), ('spectral_C1', 'spectral_L2'), ('J', 'L025'), ('J', 'L20'), ('spectral_C1', 'J')])]
    crit = criteria(data, seeds)
    crit_summary = crit.groupby(['mode', 'condition', 'weight']).agg(
        residence_gain_mean=('residence_gain_vs_H', 'mean'), residence_gain_min=('residence_gain_vs_H', 'min'),
        half_headroom_pass_seeds=('half_headroom_pass', 'sum'), source_allowance_pass_seeds=('source_allowance_pass', 'sum')).reset_index()
    crit_summary['residence_01_all_seeds_and_mean'] = (crit_summary['residence_gain_mean'] >= .01-1e-12) & (crit_summary['residence_gain_min'] >= .01-1e-12)
    withdf, wchecks = withholding(data, seeds, sporder)
    wdom = withholding_dominance(P, seeds)
    gates = {mode_scope: {**{f'{l} vs {r}': dev_gate(P, seeds, m, sc_, l, r) for l, r in (('spectral_C1', 'spectral_L1'), ('spectral_C1', 'spectral_L2'), ('J', 'L025'), ('J', 'L20'))},
                          **{f'{a} vs J': dev_gate(P, seeds, m, sc_, a, 'J') for a in ev.SPECTRAL}}
             for mode_scope, (m, sc_) in {'B/transport_all': ('B', 'transport_all'), 'A/kernel_expanded_catchup': ('A', 'kernel_expanded_catchup')}.items()}
    # negative increments and selected-vs-routed-H
    neg = per_seed[(per_seed['kind'] == 'additional_recovery') & (per_seed['condition'] != 'H')]
    negc = neg.groupby(['mode', 'scope', 'budget', 'weight']).apply(lambda g: pd.Series({'negative': int((g['value'] < -1e-12).sum()), 'total': len(g)}), include_groups=False).reset_index()
    rows = data['rows']
    worse_than_anchor = []
    for (mode, scope, b) in (('B', 'transport_all', 360), ('B', 'common_fresh', 360)):
        for s, c in itertools.product(seeds, ev.INTERFACES):
            if c == 'H': continue
            hsel = P[s, mode, 'H', scope, b, 'unweighted']['selected']; sel = P[s, mode, c, scope, b, 'unweighted']['selected']
            sub = rows[(rows['seed'] == s) & (rows['mode'] == mode) & (rows['condition'] == c) & (rows['budget'] == b)]
            for e in ('A/SEX', 'A/RAC1P', 'AB/SEX', 'AB/RAC1P'):
                v, t = e.split('/')
                for w in WEIGHTS:
                    q = sub[(sub['view'] == v) & (sub['target'] == t) & (sub['weight'] == w)].set_index('candidate_id')['log_loss']
                    hs = hsel[e]
                    anc = ('inherited_A__anchor__'+hs[len('inherited_A__'):]) if hs.startswith('inherited_A__') else hs if hs.startswith('inherited_B__') else 'anchor__'+hs
                    if anc in q.index:
                        worse_than_anchor.append({'mode': mode, 'scope': scope, 'seed': s, 'condition': c, 'endpoint': e, 'weight': w,
                                                  'selected_minus_routed_H_selected': float(q[sel[e]]-q[anc])})
    wta = pd.DataFrame(worse_than_anchor)
    # race class support and per-class coverage
    y = data['labels']['RAC1P']
    race_support = {'final_class_counts': np.bincount(y[y >= 0], minlength=9).tolist(),
                    'insufficient_classes_lt10': [k for k, n in enumerate(np.bincount(y[y >= 0], minlength=9)) if n < 10]}
    fit_support = {}
    for s in seeds:
        rec = read(out/f'seed_{s}'/'spectral_C1'/'audit_selection.json')
        base = rec['candidates']['360']['A/RAC1P']['wire__logistic']['base_candidate_directory']
        fit_support[f'mode_B_seed_{s}'] = read(Path(base)/'metadata.json')['fit_support']
        drec = read(ev.DEV/f'seed_{s}'/'spectral_C1'/'audits'/'audit_selection.json')['candidates']['360']['A/RAC1P']['wire__logistic']
        fit_support[f'mode_A_seed_{s}'] = drec.get('fit_support')
    race_support['attacker_fit_class_support'] = fit_support
    # comparable-category race diagnostic: rows whose race class had fitting support in every mode and seed
    supported = [k for k in range(9) if all(v is not None and v[k] > 0 for v in fit_support.values())]
    race_support['comparable_classes'] = supported
    comparable = []
    yr = data['labels']['RAC1P']; keep = np.isin(yr[yr >= 0], supported)
    for name, f in spec_['families'].items():
        for (left, right), e, w in itertools.product(f['contrasts'], ('A/RAC1P', 'AB/RAC1P'), WEIGHTS):
            ww = data['weights'][yr >= 0][keep] if w == 'person_weighted' else np.ones(int(keep.sum()))
            diffs = [float(ww @ (data['losses'][s, f['mode'], left, f['scope'], f['budget'], e][keep]-data['losses'][s, f['mode'], right, f['scope'], f['budget'], e][keep])/ww.sum())
                     for s in seeds]
            comparable.append({'family': name, 'left': left, 'right': right, 'endpoint': e, 'weight': w, 'rows': int(keep.sum()),
                               'recovery_difference_seed_mean': -float(np.mean(diffs)), 'label': 'comparable-category diagnostic; unadjusted point estimate'})
    race_support['comparable_category_diagnostic'] = comparable
    # development comparison
    devm, devfull, devsha = dev_means()
    consistency = []
    devpaired = pd.read_csv(ev.DEV/'PUBLICATION_EVIDENCE/PAIRED.csv.gz')
    devpaired = devpaired[(devpaired['split'] == 'test') & (devpaired['scope'] == 'kernel_expanded_catchup') & (devpaired['budget'] == 360)]
    for name, f in fam.items():
        for r in f['rows']:
            kind, e = r['endpoint'].split('/', 1)
            q = devpaired[(devpaired['left'] == r['left']) & (devpaired['right'] == r['right']) & (devpaired['weight'] == r['weight'])
                          & (devpaired['kind'] == ('utility' if kind == 'utility' else 'gains')) & (devpaired['endpoint'] == e)]
            d = float(q['difference'].mean()) if len(q) == 3 else None
            status = None
            if d is not None:
                if np.sign(r['estimate']) == np.sign(d) and d != 0: status = 'consistent'
                elif (d > 0 and r['adjusted_high'] < 0) or (d < 0 and r['adjusted_low'] > 0): status = 'contradicted'
                else: status = 'unresolved'
            consistency.append({'family': name, 'left': r['left'], 'right': r['right'], 'endpoint': r['endpoint'], 'weight': r['weight'],
                                'development_seed_mean': d, 'transport_estimate': r['estimate'], 'status': status})
    # service quality
    service = []
    for s in seeds:
        dev_anchor = {x['task']: x['checks'] for x in read(ev.FIXED/'ANCHOR_PARITY.json') if x['seed'] == s}
        for t in SOURCE:
            sc_ = data['context'][s]['service'][t]
            service.append({'seed': s, 'task': t, 'transport_unweighted': sc_['test']['log_loss'], 'transport_pwgtp': sc_['test_person_weighted']['log_loss'],
                            'transport_accuracy': sc_['test']['accuracy'], 'transport_auroc': sc_['test']['auroc'],
                            'development_unweighted': dev_anchor[t]['test']['unweighted']['log_loss'], 'development_pwgtp': dev_anchor[t]['test']['PWGTP']['log_loss'],
                            'development_accuracy': dev_anchor[t]['test']['unweighted']['accuracy']})
    # unique fit accounting
    fits = {'five_candidate_roles': 0, 'kernel_feature_fits': 0, 'catchup_trajectories': 0, 'utility_probe_pairs': 0, 'reference_probe_pairs': 0}
    for s in seeds:
        base = out/f'seed_{s}'
        fits['five_candidate_roles'] += len([p for p in base.rglob('fresh/unit_complete.json')])
        fits['kernel_feature_fits'] += len([p for p in base.rglob('kernel/complete.json')])
        fits['catchup_trajectories'] += len([p for p in base.rglob('saved_start/unit_complete.json')])
        fits['utility_probe_pairs'] += len([p for p in base.glob('*/fitted/utility/*/*/mlp/metadata.json')])
        fits['reference_probe_pairs'] += len([p for p in base.glob('reference/fitted/*/*/mlp/metadata.json')])
    # write
    ev_dir = out/'evidence'; ev_dir.mkdir(exist_ok=True)
    csv_gz(out/'PER_SEED.csv.gz', per_seed)
    csv_gz(out/'AGGREGATE.csv.gz', agg)
    csv_gz(ev_dir/'PAIRED_UNADJUSTED.csv.gz', paired)
    pd.DataFrame([{**r, 'family': n, 'critical_value': f['critical']} for n, f in fam.items() for r in f['rows']]).to_csv(out/'FAMILIES.csv', index=False)
    crit.to_csv(ev_dir/'CRITERIA_PER_SEED.csv', index=False); crit_summary.to_csv(ev_dir/'CRITERIA_SUMMARY.csv', index=False)
    csv_gz(ev_dir/'WITHHOLDING.csv.gz', withdf); wdom.to_csv(ev_dir/'WITHHOLDING_DOMINANCE.csv', index=False)
    negc.to_csv(ev_dir/'NEGATIVE_INCREMENTS.csv', index=False); wta.to_csv(ev_dir/'SELECTED_VS_ROUTED_H.csv', index=False)
    pd.DataFrame(consistency).to_csv(ev_dir/'DEVELOPMENT_CONSISTENCY.csv', index=False)
    pd.DataFrame(service).to_csv(ev_dir/'SERVICE_QUALITY.csv', index=False)
    costs.to_csv(ev_dir/'CANDIDATE_COSTS.csv', index=False)
    csv_gz(ev_dir/'ALL_CANDIDATES.csv.gz', rows.drop(columns=[]))
    made = figures(out, agg, withdf, fam, devm, seeds)
    decision = {'created_utc': ev.now(), 'lock_sha256': sha(out/'TRANSPORT_LOCK.json'), 'score_replay': data['replay'],
                'bootstrap': {'replicates': boot.replicates, 'households': boot.groups}, 'effective_counts': counts,
                'decisions': decisions, 'families': {n: {'critical': f['critical'], 'endpoints': f['endpoints'], 'decisions': f['decisions']} for n, f in fam.items()},
                'development_gate_replay': gates, 'race_support': race_support, 'withholding_arithmetic_checks': wchecks,
                'withholding_dominates_C1_all_seeds_weights': wdom[(wdom['spectral_arm'] == 'spectral_C1') & wdom['withholding_dominates_all']].to_dict('records'),
                'negative_increment_summary': negc.to_dict('records'),
                'selected_worse_than_routed_H': {f'{m}/{sc_}': {'count': int((g['selected_minus_routed_H_selected'] > 1e-12).sum()), 'total': len(g)} for (m, sc_), g in wta.groupby(['mode', 'scope'])},
                'unique_fits': fits, 'figures': made, 'development_source_sha256': devsha,
                'scope_note': 'Household-resampling uncertainty for fixed fitted systems on the 2017 final partition; not survey-design variance or retraining variability.'}
    write_json(out/'TRANSPORT_DECISION.json', decision)
    return decision


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, default=ev.OUT); p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    a = p.parse_args(); d = report(a.out, tuple(a.seeds))
    print(json.dumps(d['decisions'], indent=1)); print('replay', d['score_replay'])


if __name__ == '__main__':
    main()
