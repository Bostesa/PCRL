"""The single experiment ledger (results/<campaign>/EXPERIMENT_LEDGER.csv), appended by every agent through `append`.

One row per seed-specific configuration evaluation (or per diagnostic). `counts_toward_36` marks new seed-specific
defense/task-model configurations (fits); evaluations of frozen historical releases are recorded with 0.
"""
from __future__ import annotations
import csv
import fcntl
import hashlib
from pathlib import Path

from . import common as C

LEDGER = C.RESULTS / 'EXPERIMENT_LEDGER.csv'
COLUMNS = ('entry', 'utc', 'family', 'version', 'config_id', 'seed', 'kind', 'counts_toward_36', 'code_sha256', 'unit_id',
           'hypothesis', 'change_from_previous', 'roles_read', 'training_effort', 'selection_effort',
           'ms_point_pass_12', 'ms_guard_pass', 'ms_worst_clause', 'ms_worst_normalized', 'ms_guard_men', 'ms_guard_women',
           'ms_inc_nll_excess_U0', 'ms_inc_brier_excess_U0', 'ms_inc_acc_loss_U0', 'ms_occ_nll_excess_U0',
           'ms_occ_brier_excess_U0', 'ms_occ_acc_loss_U0', 'proxy_pair_max', 'proxy_pair_source', 'proxy_income_max',
           'proxy_occupation_max', 'distinct_pair', 'cpu_s', 'decision', 'notes')


def code_sha(*paths):
    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).read_bytes())
    return h.hexdigest()[:16]


def summarize(m):
    """Ledger fields from an amd.evaluate.measure result."""
    cl = {(c['task'], c['reference'], c['metric']): c['value'] for c in m['clauses']}
    worst = max(m['clauses'], key=lambda c: c['normalized'])
    g = m['guard_occupation_brier_excess_vs_U0']
    out = {'ms_point_pass_12': m['point_pass_12'], 'ms_guard_pass': m['guard_pass'],
           'ms_worst_clause': f"{worst['task']}|{worst['reference']}|{worst['metric']}", 'ms_worst_normalized': round(worst['normalized'], 4),
           'ms_guard_men': round(g['sex0'], 5), 'ms_guard_women': round(g['sex1'], 5),
           'ms_inc_nll_excess_U0': round(cl[('income', 'U0', 'nll_excess')], 5), 'ms_inc_brier_excess_U0': round(cl[('income', 'U0', 'brier_excess')], 5),
           'ms_inc_acc_loss_U0': round(cl[('income', 'U0', 'accuracy_loss')], 5), 'ms_occ_nll_excess_U0': round(cl[('occupation', 'U0', 'nll_excess')], 5),
           'ms_occ_brier_excess_U0': round(cl[('occupation', 'U0', 'brier_excess')], 5),
           'ms_occ_acc_loss_U0': round(cl[('occupation', 'U0', 'accuracy_loss')], 5), 'distinct_pair': m['distinct_outputs']['pair']}
    if 'proxy' in m:
        out.update({'proxy_pair_max': round(m['proxy']['pair']['max'], 4), 'proxy_pair_source': round(m['proxy']['pair']['source_proxy'], 4),
                    'proxy_income_max': round(m['proxy']['income']['max'], 4), 'proxy_occupation_max': round(m['proxy']['occupation']['max'], 4)})
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
            w.writerow({**{c: '' for c in COLUMNS}, **row, 'entry': n + 1, 'utc': C.now()})
        fcntl.flock(lf, fcntl.LOCK_UN)
    return n + 1


def configs_used():
    if not LEDGER.exists():
        return 0
    with open(LEDGER) as f:
        return sum(int(r['counts_toward_36'] or 0) for r in csv.DictReader(f))
