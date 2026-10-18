"""The single experiment ledger of the overnight occupation-headroom campaign (results/<campaign>/EXPERIMENT_LEDGER.csv).

One row per seed-specific configuration evaluation, per reference evaluation, per reader-grid evaluation, per failed or abandoned
attempt and per recorded decision. `budget_class` marks which ceiling a NEW seed-specific fitted configuration counts toward
(OCC36 occupation task model, PRIV24 privacy stage, INC8 income branch, CTRL12 matched control / stability; NONE for frozen
references, copies, combinations and repeated evaluations of an already counted configuration); `counts` is 1 only on the first
row of a new configuration. Individual fits are counted separately (model, calibrator, cross-fitting, reader) with cache hits.

    python -m ohn.ledger add --family F --config ID --seed s --kind failed|abandoned|decision --decision TEXT [--notes ...]
"""
from __future__ import annotations
import argparse
import csv
import fcntl
import hashlib
import os
from pathlib import Path

from . import common as C

LEDGER = C.RESULTS / 'EXPERIMENT_LEDGER.csv'
OCC_KEYS = [(r, m) for r in ('U0', 'Ucal') for m in ('accuracy_loss', 'nll_excess', 'brier_excess')]
COLUMNS = (['entry', 'utc', 'family', 'version', 'config_id', 'seed', 'anchor', 'kind', 'budget_class', 'counts',
            'n_model_fits', 'n_calibrator_fits', 'n_crossfit_fits', 'reader_fits', 'cache_hits', 'code_sha256', 'unit_id',
            'interpreter', 'hypothesis', 'change_from_previous', 'roles_read', 'training_effort', 'selection_effort',
            'point_pass_12', 'guard_pass', 'worst_clause', 'worst_normalized', 'inc_min_margin_se_units', 'inc_worst_clause']
           + [f'occ_{m}_{r}{suf}' for r, m in OCC_KEYS for suf in ('', '_se')]
           + ['occ_min_margin_se_units', 'guard_men', 'guard_men_se', 'guard_women', 'guard_women_se',
              'occ_nll_vs_T_TREE', 'occ_nll_vs_T_TREE_se', 'occ_brier_vs_T_TREE', 'occ_nll_vs_TASK_REFIT', 'occ_nll_vs_TASK_REFIT_se',
              'oof_nll_vs_U0_in_sample', 'deployed_minus_foldavg_nll', 'deployed_minus_foldavg_nll_se',
              'reserve_status', 'reserve_min_ratio', 'reserve_binding', 'witness_status',
              'grid_pair_benefit_summary', 'grid_recipient_summary', 'stability_status',
              'cpu_s', 'decision', 'notes'])


def code_sha(*paths):
    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).read_bytes())
    return h.hexdigest()[:16]


def _r(x, k=6):
    return '' if x is None else (round(float(x), k) if isinstance(x, (int, float)) else x)


def summarize(m):
    """Ledger fields from ohn.evaluate.complete_eval (+ reserve when present)."""
    u = m['utility']; cl = u['clauses']
    worst = max(cl, key=lambda c: c['normalized'])
    by = {(c['task'], c['reference'], c['metric']): c for c in cl}
    g = {x['group']: x for x in u['guards']}
    inc = [c for c in cl if c['task'] == 'income']
    inc_worst = min(inc, key=lambda c: c['margin_se_units'])
    out = {'point_pass_12': u['point_pass_12'], 'guard_pass': u['guard_pass'],
           'worst_clause': f"{worst['task']}|{worst['reference']}|{worst['metric']}", 'worst_normalized': _r(worst['normalized'], 4),
           'inc_min_margin_se_units': _r(inc_worst['margin_se_units'], 3),
           'inc_worst_clause': f"{inc_worst['reference']}|{inc_worst['metric']}",
           'occ_min_margin_se_units': _r(min(c['margin_se_units'] for c in cl if c['task'] == 'occupation'), 3),
           'guard_men': _r(g['sex0']['value']), 'guard_men_se': _r(g['sex0']['se']), 'guard_women': _r(g['sex1']['value']),
           'guard_women_se': _r(g['sex1']['se'])}
    for r, mm in OCC_KEYS:
        c = by[('occupation', r, mm)]
        out[f'occ_{mm}_{r}'] = _r(c['value']); out[f'occ_{mm}_{r}_se'] = _r(c['se'])
    v = m.get('occupation_versus_task_only_references', {})
    if v:
        out.update({'occ_nll_vs_T_TREE': _r(v['T_TREE|nll_diff']['value']), 'occ_nll_vs_T_TREE_se': _r(v['T_TREE|nll_diff']['se']),
                    'occ_brier_vs_T_TREE': _r(v['T_TREE|brier_diff']['value']),
                    'occ_nll_vs_TASK_REFIT': _r(v['TASK_REFIT|nll_diff']['value']), 'occ_nll_vs_TASK_REFIT_se': _r(v['TASK_REFIT|nll_diff']['se'])})
    cf = m.get('crossfit')
    if cf:
        out['oof_nll_vs_U0_in_sample'] = _r(cf['oof_vs_U0_in_sample']['nll_excess']['value'])
        d = cf['deployed_minus_foldavg_on_MODEL_SELECT']['nll_diff']
        out['deployed_minus_foldavg_nll'] = _r(d['value']); out['deployed_minus_foldavg_nll_se'] = _r(d['se'])
    rs = m.get('reserve')
    if isinstance(rs, dict):
        out['reserve_status'] = rs.get('status', ''); out['reserve_min_ratio'] = _r(rs.get('min_ratio'), 3)
        out['reserve_binding'] = str(rs.get('binding_condition', ''))[:300]
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
        n = 0 if new else sum(1 for _ in open(LEDGER, newline='')) - 1
        with open(LEDGER, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            if new:
                w.writeheader()
            w.writerow({**{k: '' for k in COLUMNS}, **{k: _r(v) if isinstance(v, float) else v for k, v in row.items()},
                        'entry': n + 1, 'utc': C.now()})
        fcntl.flock(lf, fcntl.LOCK_UN)
    return n + 1


def rows():
    if not LEDGER.exists():
        return []
    with open(LEDGER, newline='') as f:
        return list(csv.DictReader(f))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('add')
    d.add_argument('--family', required=True); d.add_argument('--config', required=True); d.add_argument('--seed', default='')
    d.add_argument('--kind', required=True, choices=('failed', 'abandoned', 'decision', 'stopped'))
    d.add_argument('--decision', required=True); d.add_argument('--notes', default='')
    d.add_argument('--budget-class', default='NONE', choices=('OCC36', 'PRIV24', 'INC8', 'CTRL12', 'NONE'))
    d.add_argument('--counts', type=int, default=0); d.add_argument('--n-model-fits', default='')
    a = ap.parse_args(argv)
    n = append({'family': a.family, 'version': a.kind, 'config_id': a.config, 'seed': a.seed, 'kind': a.kind,
                'budget_class': a.budget_class, 'counts': a.counts, 'n_model_fits': a.n_model_fits,
                'unit_id': os.environ.get(C.UNIT_ENV, ''), 'decision': a.decision, 'notes': a.notes})
    print(n)


if __name__ == '__main__':
    main()
