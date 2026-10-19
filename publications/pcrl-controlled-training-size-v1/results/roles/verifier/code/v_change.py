"""Record one correction of a verifier check in CHECK_CHANGES.json (the earlier check stays in CHECKS.json unchanged).

    python3 .../v_change.py --before ID --after ID --cause TEXT --before-result TEXT --after-result TEXT [--criterion-unchanged TEXT]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402


def adjudicate(n, text):
    """Record the coordinator's adjudication of change n (the change record itself is not edited; the previous file version is
    snapshotted to records/history first)."""
    import datetime as dt
    import os
    import shutil
    doc = json.loads(V.CHECK_CHANGES.read_text())
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    shutil.copyfile(V.CHECK_CHANGES, V.HISTORY / f'CHECK_CHANGES__before_adjudication_{n}_{stamp}.json')
    hit = [c for c in doc['changes'] if c['n'] == n]
    if len(hit) != 1:
        raise SystemExit(f'change {n} not found')
    hit[0].setdefault('adjudication_history', []).append({'from': hit[0].get('adjudication'), 'to': text, 'utc': V.utc()})
    hit[0]['adjudication'] = text
    tmp = V.CHECK_CHANGES.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(doc, indent=1) + '\n')
    os.replace(tmp, V.CHECK_CHANGES)
    print(json.dumps({'change': n, 'adjudication': text}))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--adjudicate':
        adjudicate(int(sys.argv[2]), sys.argv[3])
        return
    ap = argparse.ArgumentParser()
    ap.add_argument('--before', required=True); ap.add_argument('--after', required=True); ap.add_argument('--cause', required=True)
    ap.add_argument('--before-result', required=True); ap.add_argument('--after-result', required=True)
    ap.add_argument('--criterion-unchanged', required=True)
    ap.add_argument('--kind', default='VERIFIER_BUG', choices=('VERIFIER_BUG', 'INPUT_CHANGED', 'SCOPE'))
    a = ap.parse_args()
    doc = json.loads(V.CHECKS.read_text())
    ids = {c['id']: c for c in doc['checks']}
    if a.before not in ids or a.after not in ids:
        raise SystemExit('both check ids must exist in CHECKS.json')
    n = V.add_check_change({'check_id_before': a.before, 'check_id_after': a.after, 'kind': a.kind,
                            'before': {'status': ids[a.before]['status'], 'result': a.before_result},
                            'after': {'status': ids[a.after]['status'], 'result': a.after_result},
                            'cause': a.cause, 'acceptance_rule': a.criterion_unchanged,
                            'adjudication': 'pending (coordinator)'})
    print(json.dumps({'change': n}))


if __name__ == '__main__':
    main()
