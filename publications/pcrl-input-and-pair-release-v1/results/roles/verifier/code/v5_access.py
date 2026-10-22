"""Verifier check 5 (label-free): the label-access log against the registration receipt, the cards, the queue and the policy.

- every read in access/LABEL_ACCESS.jsonl is after the verified registration receipt (02:14:15Z) and inside a queue unit that was
  running at that time (unit_start before the read; unit_end, if any, after it);
- every fitting read ('fit' / 'calibrate') carries a card under results/<campaign>/cards/ whose blob at the recorded commit has the
  recorded SHA-256, committed before the read;
- no MODEL_SELECT read came from a fitting entry point (entry point or module chain in ipr.ia / ipr.ib / ipr.ctl / ipr.fs);
- the first MODEL_SELECT read follows the commit of the IA and IB cards (9b70c5d77) and its commit time is read from git;
- no ATTACK_FIT / ATTACK_VALIDATION / DEV_REPORT label read; every read's role / fields / operation allowed by the policy.

    python3 -m ipr.jobqueue run --unit IPR_VER_ACCESS_V1 --stage verify --rss-gib 1 -- python3 <this file>
"""
from __future__ import annotations
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(HERE))

import vlib as V  # noqa: E402

FIT_PREFIX = ('ipr.ia', 'ipr.ib', 'ipr.ctl', 'ipr.fs')
POLICY = {'fit': {'roles': ('MODEL_FIT',), 'fields': ('occupation', 'income', 'sex')},
          'calibrate': {'roles': ('CALIBRATION',), 'fields': ('occupation', 'income')},
          'benchmark': {'roles': ('MODEL_SELECT',), 'fields': ('income', 'occupation', 'sex')},
          'readers': {'roles': ('MODEL_SELECT',), 'fields': ('sex',)},
          'verify': {'roles': ('MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT'), 'fields': ('income', 'occupation', 'sex')}}
ENTRY = {'fit': FIT_PREFIX, 'calibrate': FIT_PREFIX, 'benchmark': ('ipr.evaluate',),
         'readers': ('ipr.devreaders', 'ipr.readercustody', 'ipr.richreaders'),
         'verify': (f'results/{V.CAMPAIGN}/roles/verifier/code/',)}


def git(*a):
    r = subprocess.run(['git', '-C', str(V.ROOT), *a], capture_output=True)
    return r.returncode, r.stdout


def is_fit_module(m):
    return any(str(m) == p or str(m).startswith(p + '.') for p in FIT_PREFIX)


def main():
    t0 = time.process_time()
    rc_ = json.loads(V.RECEIPT.read_text())
    receipt_utc = V.parse_utc(rc_['verified_utc'])
    rc, ts = git('show', '-s', '--format=%cI', '9b70c5d77')
    cards_commit_utc = V.parse_utc(ts.decode().strip()) if rc == 0 else None
    rc2, files = git('show', '--name-only', '--format=', '9b70c5d77')
    cards_in_commit = [f for f in files.decode().split('\n') if '/cards/' in f]
    ev = [json.loads(l) for l in V.ACCESS_LOG.read_text().splitlines() if l.strip()]
    q = [json.loads(l) for l in V.QUEUE_LEDGER.read_text().splitlines() if l.strip()]
    starts = {e['unit_id']: V.parse_utc(e['utc']) for e in q if e.get('event') == 'unit_start'}
    ends = {e['unit_id']: V.parse_utc(e['utc']) for e in q if e.get('event') == 'unit_end'}
    blob = {}
    rows, problems = [], []
    for i, e in enumerate(ev):
        t = V.parse_utc(e['utc'])
        u = e.get('unit_id')
        op = e.get('operation')
        r = {'i': i, 'utc': e['utc'], 'unit': u, 'role': e.get('role'), 'fields': e.get('fields'), 'operation': op,
             'entry_point': e.get('entry_point')}
        r['after_receipt'] = t > receipt_utc
        r['in_running_unit'] = bool(u in starts and starts[u] <= t and (u not in ends or ends[u] >= t))
        pol = POLICY.get(op)
        r['policy_ok'] = bool(pol and e.get('role') in pol['roles'] and set(e.get('fields') or []) <= set(pol['fields']))
        ep = str(e.get('entry_point', '')).split(':')[0]
        r['entry_ok'] = any(ep == x or ep.startswith(x if x.endswith('/') else x + '.') or (x.endswith('/') and ep.startswith(x))
                            for x in ENTRY.get(op, ()))
        r['closed_role'] = e.get('role') in ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT', 'SEALED_ASSESSMENT')
        if op in ('fit', 'calibrate'):
            card, csha, ccom = e.get('card'), e.get('card_sha256'), e.get('card_commit')
            ok = bool(card and csha and ccom and str(card).startswith(f'results/{V.CAMPAIGN}/cards/'))
            if ok:
                key = (card, ccom)
                if key not in blob:
                    rc3, b = git('show', f'{ccom}:{card}')
                    blob[key] = hashlib.sha256(b).hexdigest() if rc3 == 0 else None
                ok = blob[key] == csha
                r['card_commit_before_read'] = V.parse_utc(e['card_commit_utc']) <= t
                ok = ok and r['card_commit_before_read']
            r['card_ok'] = ok
        if e.get('role') == 'MODEL_SELECT':
            chain = e.get('module_chain') or []
            r['ms_from_fit_entry'] = is_fit_module(ep) or any(is_fit_module(m) for m in chain)
        bad = [k for k in ('after_receipt', 'in_running_unit', 'policy_ok', 'entry_ok') if not r[k]]
        if r['closed_role']:
            bad.append('closed_role')
        if r.get('card_ok') is False:
            bad.append('card')
        if r.get('ms_from_fit_entry'):
            bad.append('ms_from_fit_entry')
        r['problems'] = bad
        if bad:
            problems.append(r)
        rows.append(r)
    ms_reads = [r for r in rows if r['role'] == 'MODEL_SELECT']
    first_ms = min((V.parse_utc(r['utc']) for r in ms_reads), default=None)
    out = {'unit_id': V.unit_id(), 'utc': V.utc(), 'receipt_verified_utc': str(receipt_utc),
           'cards_commit_9b70c5d77_utc': str(cards_commit_utc), 'cards_in_commit': cards_in_commit,
           'n_reads': len(rows), 'by_operation_role': {}, 'first_label_read_utc': min((r['utc'] for r in rows), default=None),
           'first_model_select_read_utc': str(first_ms) if first_ms else None,
           'first_ms_after_cards_commit': bool(first_ms is None or (cards_commit_utc and first_ms > cards_commit_utc)),
           'problems': problems, 'rows': rows}
    for r in rows:
        k = f"{r['operation']}|{r['role']}"
        out['by_operation_role'][k] = out['by_operation_role'].get(k, 0) + 1
    out['all_ok'] = (not problems) and out['first_ms_after_cards_commit']
    out['cpu_s'] = round(time.process_time() - t0, 3)
    p = V.write_record(f'V5_ACCESS__{V.unit_id() or "NOQUEUE"}.json', out)
    print(json.dumps({k: out[k] for k in ('n_reads', 'by_operation_role', 'first_label_read_utc', 'first_model_select_read_utc',
                                          'cards_commit_9b70c5d77_utc', 'first_ms_after_cards_commit', 'all_ok')}
                     | {'n_problems': len(problems), 'record': str(p)}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
