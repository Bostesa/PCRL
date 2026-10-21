"""Verifier check 5 (label-free): the label-access audit.

Reads .private/<campaign>/access/LABEL_ACCESS.jsonl, the registration receipt, the queue ledger and git, and checks:
(a) every label read happened at or after the verified registration receipt (17:42:42Z) and inside a queue unit that was running
    (unit_start <= read utc, and no unit_end before the read);
(b) every fitting read ('fit' / 'calibrate') carried a committed card whose SHA-256 equals the card blob at the recorded commit,
    committed before the read, with the family prefix of its entry point;
(c) no MODEL_SELECT read came from a fitting entry point (entry point and module chain free of ddr.da / ddr.db / ddr.ctl / ddr.fs);
(d) the first MODEL_SELECT read followed the commit of both registered cards (5f2fd02a9);
(e) every read is allowed by the registered policy (role x field x operation; entry points incl. the dated amendments);
(f) no closed-role label read; (g) reported: the first MODEL_SELECT read, the commit times of later cards (registered / adaptive tag
    under RULES.md 7 / 10.1).

    python3 -m ddr.jobqueue run --unit DDR_VER_ACCESS_V1 --stage verify --rss-gib 2 -- python3 <this file>
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402

FIT_PREFIXES = ('ddr.da', 'ddr.db', 'ddr.ctl', 'ddr.fs')
POLICY = {   # typed from ddr/common.py POLICY and ACCESS_POLICY.json (registered)
    'fit': {'roles': {'MODEL_FIT'}, 'fields': {'occupation', 'income', 'sex'}},
    'calibrate': {'roles': {'CALIBRATION'}, 'fields': {'occupation', 'income'}},
    'benchmark': {'roles': {'MODEL_SELECT'}, 'fields': {'income', 'occupation', 'sex'}},
    'readers': {'roles': {'MODEL_SELECT'}, 'fields': {'sex'}},
    'verify': {'roles': {'MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT'}, 'fields': {'income', 'occupation', 'sex'}},
}
ENTRY = {'fit': FIT_PREFIXES, 'calibrate': FIT_PREFIXES, 'benchmark': ('ddr.evaluate',), 'readers': ('ddr.devreaders', 'ddr.readercustody'),
         'verify': (f'results/{V.CAMPAIGN}/roles/verifier/code/',)}
CARD_COMMIT = '5f2fd02a98b5f527b089067ca9f964a97030f30d'


def git(*a):
    r = subprocess.run(['git', '-C', str(V.ROOT), *a], capture_output=True)
    return r.returncode, r.stdout


def is_family(mod):
    m = str(mod)
    return any(m == p or m.startswith(p + '.') or m.startswith(p + ':') for p in FIT_PREFIXES)


def main():
    t0 = time.process_time()
    reads = [json.loads(l) for l in V.ACCESS_LOG.read_text().splitlines() if l.strip()]
    receipt = json.loads(V.RECEIPT.read_text())
    t_receipt = V.parse_utc(receipt['verified_utc'])
    ledger = [json.loads(l) for l in V.QUEUE_LEDGER.read_text().splitlines() if l.strip()]
    starts = {e['unit_id']: V.parse_utc(e['utc']) for e in ledger if e.get('event') == 'unit_start'}
    ends = {e['unit_id']: V.parse_utc(e['utc']) for e in ledger if e.get('event') == 'unit_end'}
    amend = json.loads((V.RESULTS / 'ACCESS_AMENDMENTS.json').read_text())
    entry = {k: list(v) for k, v in ENTRY.items()}
    for a in amend.get('amendments', []):
        entry.setdefault(a['operation'], []).append(a['entry_point'])
    rc, out = git('log', '-1', '--format=%cI', CARD_COMMIT)
    t_cards = V.parse_utc(out.decode().strip())
    # card commit check: both registered cards in that commit
    rc2, names = git('show', '--name-only', '--format=', CARD_COMMIT)
    cards_in_commit = sorted(n for n in names.decode().split('\n') if n.startswith(f'results/{V.CAMPAIGN}/cards/'))
    blob = {}
    rows, problems = [], []
    for i, r in enumerate(reads):
        t = V.parse_utc(r['utc'])
        op, role, fields = r['operation'], r['role'], set(r['fields'])
        ep = r['entry_point'].split(':')[0]
        chk = {'i': i, 'utc': r['utc'], 'unit': r.get('unit_id'), 'role': role, 'op': op, 'entry': r['entry_point'],
               'after_receipt': t >= t_receipt}
        pol = POLICY.get(op)
        chk['policy_ok'] = bool(pol and role in pol['roles'] and fields <= pol['fields'])
        chk['entry_ok'] = any(ep == e or ep.startswith(e if e.endswith('/') else e + '.') or (e.endswith('/') and ep.startswith(e))
                              for e in entry.get(op, []))
        u = r.get('unit_id')
        chk['inside_running_unit'] = bool(u and u in starts and starts[u] <= t and (u not in ends or ends[u] >= t))
        chk['closed_role'] = role in ('ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT', 'SEALED_ASSESSMENT')
        if op in ('fit', 'calibrate'):
            card = r.get('card')
            key = (card, r.get('card_commit'))
            if key not in blob:
                rc, b = git('show', f"{r.get('card_commit')}:{card}") if card and r.get('card_commit') else (1, b'')
                blob[key] = V.hashlib.sha256(b).hexdigest() if rc == 0 else None
            fam = ep.split('.')[1].upper() if ep.count('.') >= 1 else None
            chk['card_ok'] = bool(card and blob[key] == r.get('card_sha256') and V.parse_utc(r['card_commit_utc']) <= t
                                  and Path(card).name.startswith(f'{fam}_'))
            chk['card_file_unchanged_now'] = bool(card and V.file_sha(V.ROOT / card) == r.get('card_sha256'))
        if role == 'MODEL_SELECT':
            chk['ms_not_from_fitting'] = not is_family(ep) and not any(is_family(m) for m in r.get('module_chain', []))
            chk['ms_after_cards'] = t >= t_cards
        for k in ('after_receipt', 'policy_ok', 'entry_ok', 'inside_running_unit', 'card_ok', 'ms_not_from_fitting', 'ms_after_cards'):
            if k in chk and not chk[k]:
                problems.append({'i': i, 'check': k, 'entry': r['entry_point'], 'utc': r['utc']})
        if chk['closed_role']:
            problems.append({'i': i, 'check': 'closed_role', 'entry': r['entry_point'], 'utc': r['utc']})
        rows.append(chk)
    ms_reads = [r for r in reads if r['role'] == 'MODEL_SELECT']
    first_ms = min((V.parse_utc(r['utc']) for r in ms_reads), default=None)
    # later cards: commit time vs first MODEL_SELECT read (RULES.md 7 / 10.1 tag)
    cards = sorted(p.name for p in (V.RESULTS / 'cards').glob('*.md') if p.name != 'README.md')
    card_times = {}
    for c in cards:
        rc, o = git('log', '--diff-filter=A', '--format=%H %cI', '--', f'results/{V.CAMPAIGN}/cards/{c}')
        line = o.decode().strip().split('\n')[-1] if o.strip() else ''
        if line:
            h, when = line.split()
            card_times[c] = {'first_commit': h, 'utc': V.parse_utc(when).astimezone(V.dt.timezone.utc).isoformat(),
                             'before_first_model_select_read': (first_ms is None) or V.parse_utc(when) < first_ms}
    rec = {'unit_id': V.unit_id(), 'utc': V.utc(), 'reads': len(reads), 'receipt_verified_utc': receipt['verified_utc'],
           'cards_commit': CARD_COMMIT, 'cards_commit_utc': t_cards.astimezone(V.dt.timezone.utc).isoformat(),
           'cards_in_commit': cards_in_commit,
           'first_read_utc': min(r['utc'] for r in reads) if reads else None,
           'first_model_select_read_utc': first_ms.isoformat() if first_ms else None,
           'model_select_reads': len(ms_reads), 'fit_reads': sum(r['operation'] == 'fit' for r in reads),
           'by_operation': {op: sum(r['operation'] == op for r in reads) for op in sorted({r['operation'] for r in reads})},
           'entry_points': sorted({r['entry_point'] for r in reads}), 'card_first_commits': card_times,
           'rows': rows, 'problems': problems, 'cpu_s': round(time.process_time() - t0, 3)}
    p = V.write_record(f'V5_ACCESS__{V.unit_id() or "NOQUEUE"}.json', rec)
    print(json.dumps({'record': str(p), 'reads': len(reads), 'problems': problems, 'first_ms': rec['first_model_select_read_utc'],
                      'cards_commit_utc': rec['cards_commit_utc'], 'cards_in_commit': cards_in_commit,
                      'card_first_commits': card_times}, default=V._default))
    return 0 if not problems else 1


if __name__ == '__main__':
    sys.exit(main())
