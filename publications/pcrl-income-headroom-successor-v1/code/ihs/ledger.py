"""The single experiment ledger (results/<campaign>/EXPERIMENT_LEDGER.csv), appended by every role through `append`.

One row per seed-specific configuration evaluation (or per reference / diagnostic). `counts_toward_36` marks new seed-specific
release or task-model configurations (including newly fitted controls); evaluations of frozen historical releases are recorded
with 0. `n_model_fits` counts the individual model fits behind the row (task models, calibrators, critics, mechanism fits);
reader fits are counted separately in `reader_fits`. Failed or stopped attempts are rows too (decision FAILED / STOPPED).
"""
from __future__ import annotations
import csv
import fcntl
import hashlib
from pathlib import Path

from . import common as C

LEDGER = C.RESULTS / 'EXPERIMENT_LEDGER.csv'
INCOME_KEYS = [(r, m) for r in ('U0', 'Ucal') for m in ('accuracy_loss', 'nll_excess', 'brier_excess')]
COLUMNS = (['entry', 'utc', 'family', 'version', 'config_id', 'seed', 'kind', 'counts_toward_36', 'n_model_fits', 'code_sha256',
            'unit_id', 'hypothesis', 'change_from_previous', 'roles_read', 'training_effort', 'selection_effort',
            'ms_point_pass_12', 'ms_guard_pass', 'ms_worst_clause', 'ms_worst_normalized']
           + [f'inc_{m}_{r}{suf}' for r, m in INCOME_KEYS for suf in ('', '_se')]
           + ['inc_min_margin_se_units', 'occ_worst_clause', 'occ_worst_normalized', 'occ_min_margin_se_units',
              'guard_men', 'guard_men_se', 'guard_women', 'guard_women_se', 'reserve_status', 'reserve_failed',
              'proxy_pair_mean', 'proxy_pair_min', 'proxy_pair_max', 'proxy_income_mean', 'proxy_occupation_mean',
              'benefit_vs_NEURAL_RAW_mean', 'benefit_vs_NEURAL_RAW_min', 'benefit_vs_H_LJ_mean', 'benefit_vs_H_LJ_min',
              'inc_recipient_vs_NEURAL_RAW_max', 'inc_recipient_vs_H_LJ_max', 'occ_recipient_vs_NEURAL_RAW_max',
              'occ_recipient_vs_H_LJ_max', 'distinct_income', 'distinct_pair', 'cpu_s', 'reader_fits', 'decision', 'notes'])


def code_sha(*paths):
    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).read_bytes())
    return h.hexdigest()[:16]


def _r(x, k=6):
    return '' if x is None else (round(float(x), k) if isinstance(x, (int, float)) else x)


def summarize(m, cmp=None, reserve=None):
    """Ledger fields from ihs.evaluate.measure (+ reference comparisons and the reserve screen when available)."""
    u = m['utility']; cl = u['clauses']
    worst = max(cl, key=lambda c: c['normalized'])
    by = {(c['task'], c['reference'], c['metric']): c for c in cl}
    g = {x['group']: x for x in u['guards']}
    occ = [c for c in cl if c['task'] == 'occupation']
    occ_worst = max(occ, key=lambda c: c['normalized'])
    out = {'ms_point_pass_12': u['point_pass_12'], 'ms_guard_pass': u['guard_pass'],
           'ms_worst_clause': f"{worst['task']}|{worst['reference']}|{worst['metric']}", 'ms_worst_normalized': _r(worst['normalized'], 4),
           'inc_min_margin_se_units': _r(min(c['margin_se_units'] for c in cl if c['task'] == 'income'), 3),
           'occ_worst_clause': f"{occ_worst['reference']}|{occ_worst['metric']}", 'occ_worst_normalized': _r(occ_worst['normalized'], 4),
           'occ_min_margin_se_units': _r(min(c['margin_se_units'] for c in occ), 3),
           'guard_men': _r(g['sex0']['value']), 'guard_men_se': _r(g['sex0']['se']), 'guard_women': _r(g['sex1']['value']),
           'guard_women_se': _r(g['sex1']['se']), 'distinct_income': m['distinct_outputs']['income'], 'distinct_pair': m['distinct_outputs']['pair']}
    for r, mm in INCOME_KEYS:
        c = by[('income', r, mm)]
        out[f'inc_{mm}_{r}'] = _r(c['value']); out[f'inc_{mm}_{r}_se'] = _r(c['se'])
    if 'proxy' in m:
        s = m['proxy']['summary']
        out.update({'proxy_pair_mean': _r(s['pair']['mean'], 4), 'proxy_pair_min': _r(s['pair']['min'], 4), 'proxy_pair_max': _r(s['pair']['max'], 4),
                    'proxy_income_mean': _r(s['income']['mean'], 4), 'proxy_occupation_mean': _r(s['occupation']['mean'], 4),
                    'reader_fits': m['proxy']['n_reader_fits_requested']})
    if cmp:
        for ref in ('NEURAL_RAW', 'H_LJ'):
            if ref in cmp:
                b = cmp[ref]
                out[f'benefit_vs_{ref}_mean'] = _r(b['pair']['mean'], 4); out[f'benefit_vs_{ref}_min'] = _r(b['pair']['min'], 4)
                out[f'inc_recipient_vs_{ref}_max'] = _r(b['income']['max'], 4); out[f'occ_recipient_vs_{ref}_max'] = _r(b['occupation']['max'], 4)
    if reserve:
        out['reserve_status'] = reserve['status']; out['reserve_failed'] = ';'.join(reserve.get('failed', []))[:400]
    return out


def append(row):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    unknown = set(row) - set(COLUMNS)
    if unknown:
        raise KeyError(f'unknown ledger columns {sorted(unknown)}')
    lock = C.PRIV / 'compute' / 'ledger_csv.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    with open(lock, 'a+') as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        new = not LEDGER.exists()
        n = 0 if new else sum(1 for _ in open(LEDGER)) - 1
        with open(LEDGER, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            if new:
                w.writeheader()
            w.writerow({**{k: '' for k in COLUMNS}, **row, 'entry': n + 1, 'utc': C.now()})
        fcntl.flock(lf, fcntl.LOCK_UN)
    return n + 1


def rows():
    if not LEDGER.exists():
        return []
    with open(LEDGER, newline='') as f:
        return list(csv.DictReader(f))


if __name__ == '__main__':
    import argparse
    import os
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('decision')
    d.add_argument('--family', required=True); d.add_argument('--config', required=True); d.add_argument('--decision', required=True)
    d.add_argument('--notes', default='')
    a = ap.parse_args()
    n = append({'family': a.family, 'version': 'decision', 'config_id': a.config, 'kind': 'decision', 'counts_toward_36': 0,
                'n_model_fits': 0, 'unit_id': os.environ.get(C.UNIT_ENV, ''), 'decision': a.decision, 'notes': a.notes})
    print(n)
