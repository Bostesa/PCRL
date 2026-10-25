"""EXPERIMENT_LEDGER.csv of the PCRL direct-recovery frontier campaign (label-free; built from the saved records and the queue ledger;
adapted from nrm.ledger).

    python3 -m drf.ledger

One row per queue unit (every attempt, failed ones included) with its stage, status, CPU, wall time, code digest at launch, git
head, interpreter, the unit's own fit counters, the label operations it performed (from the access log: operation x role x
card, counts). CONFIG_LEDGER.csv lists every saved release configuration (the 48-configuration budget).
"""
from __future__ import annotations
import csv
import json
import sys
from collections import Counter, defaultdict

from . import common as C
from . import jobqueue as Q

HEAD = ['entry', 'unit_id', 'stage', 'status', 'returncode', 'start_utc', 'end_utc', 'cpu_s', 'wall_s', 'code_digest_at_launch', 'git_head',
        'interpreter', 'entry_point', 'argv_tail', 'label_reads', 'model_fits', 'crossfit_fits', 'routing_fits', 'calibrator_fits',
        'reader_fits', 'cache_hits_own', 'cache_hits_source', 'unit_stats_json']


def main():
    es = Q.entries()
    starts = {e['unit_id']: e for e in es if e.get('event') == 'unit_start'}
    ends = {e['unit_id']: e for e in es if e.get('event') == 'unit_end'}
    acc = defaultdict(Counter)
    for e in C.access_events():
        acc[e.get('unit_id')][f"{e.get('operation')}:{e.get('role')}:{(e.get('card') or '').split('/')[-1]}"] += 1
    rows = []
    for i, (u, s) in enumerate(starts.items(), start=1):
        e = ends.get(u, {})
        argv = s.get('argv') or []
        ep = argv[argv.index('-m') + 1] if '-m' in argv else next((a.split('/')[-1] for a in argv if a.endswith('.py')), '')
        st = e.get('unit_stats') or {}
        rows.append([i, u, s.get('stage'), e.get('status', 'running_or_lost'), e.get('returncode'), s.get('utc'), e.get('utc'),
                     e.get('charged_cpu_s'), e.get('wall_s'), (s.get('code_digest_at_launch') or '')[:16], (s.get('git_head') or '')[:10],
                     (s.get('interpreter') or {}).get('python'), ep, ' '.join(a for a in argv[argv.index('-m') + 2:] if '/' not in a)
                     if '-m' in argv else '', json.dumps(dict(acc.get(u, {})), sort_keys=True), st.get('n_model_fits'),
                     st.get('n_crossfit_fits'), st.get('n_routing_fits'), st.get('n_calibrator_fits'), st.get('reader_fits'),
                     st.get('cache_hits_own'), st.get('cache_hits_source'), json.dumps(st, sort_keys=True)])
    with open(C.RESULTS / 'EXPERIMENT_LEDGER.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(HEAD); w.writerows(rows)
    from . import release as RL
    cfg = RL.configurations()
    keys = ['configuration_number', 'utc', 'family', 'config_id', 'version', 'kind', 'randomized', 'card', 'card_sha256', 'release_sha256',
            'unit_id', 'n_model_fits', 'n_calibrator_fits', 'n_crossfit_fits', 'n_critic_fits']
    with open(C.RESULTS / 'CONFIG_LEDGER.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore'); w.writeheader(); w.writerows(cfg)
    charges = [e for e in es if e.get('event') == 'charge']
    print(json.dumps({'rows': len(rows), 'failed': [u for u, e in ends.items() if e['status'] != 'done'], 'charges': len(charges)}))


if __name__ == '__main__':
    main()
    sys.exit(0)
