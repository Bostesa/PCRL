"""Provenance timing (light; git objects and the queue ledger only): every mechanism-card revision commit against the start of its
first fit unit (cards written before their fits), the registration / receipt order, and the code-lock time against the first
reader-grid unit."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vcommon as V  # noqa: E402

REVISIONS = [  # (label, commit, regex of the revision's fit units)
    ('A initial design (registration b1fd591d0)', 'b1fd591d0', r'OHN_A_A0_'),
    ('B initial design (registration b1fd591d0)', 'b1fd591d0', r'OHN_B_(BASES_(TREE|ADD)_V1|FIT_B1)'),
    ('C initial design (registration b1fd591d0)', 'b1fd591d0', r'OHN_C_(C1_|CC_CONT|CC_VSCAL)(?!.*DRYRUN)'),
    ('A revision 1', 'cb89a8b51', r'OHN_A_A1_'),
    ('B revision B2', '2f785ec18', r'OHN_B_(BASES_(TREE|ADD)_V2|FIT_B2)'),
    ('C revision 1', '8f244b3d2', r'OHN_C_(C2_|CC_XF_CONT)'),
    ('C revision 2', '7eaf3abdf', r'OHN_C_(C3_|CC_XC)'),
    ('B revision B3', '5c9c6cef5', r'OHN_B_(FIT_)?B3'),
    ('Protection card (fab5254ac) -> P1 fits', 'fab5254ac', r'OHN_PROT_P1_'),
    # extension 07:30Z (scope added after the first PASS run; earlier result kept in records/history): later revisions
    ('Income branch card (721a64c15) -> INC fits', '721a64c15', r'OHN_INC_(STEP1|INC_)'),
    ('Protection revision P2 card (651d57b7f) -> P2 fits', '651d57b7f', r'OHN_PROT_P2_'),
    ('Protection revision P2 card (651d57b7f) -> P2A fits', '651d57b7f', r'OHN_PROT_P2A_'),
    ('Protection revision P3 card (06b0ca47a) -> P3 / P3A fits', '06b0ca47a', r'OHN_PROT_P3'),
]


def commit_time(c):
    r = subprocess.run(['git', '-C', str(V.ROOT), 'show', '-s', '--format=%cI', c], capture_output=True, text=True)
    return datetime.fromisoformat(r.stdout.strip()).astimezone(timezone.utc) if r.returncode == 0 else None


def main():
    L = [json.loads(l) for l in open(V.PRIV / 'compute' / 'ledger.jsonl') if l.strip()]
    starts = [(datetime.fromisoformat(e['utc']), e['unit_id'], e['stage']) for e in L if e.get('event') == 'unit_start']
    rows, issues = [], []
    for label, c, pat in REVISIONS:
        t = commit_time(c)
        fits = [s for s in starts if re.match(pat, s[1]) and s[2] in ('fit',)]
        first = fits[0] if fits else None
        ok = None if first is None else (t is not None and t <= first[0])
        rows.append({'revision': label, 'commit': c, 'commit_utc': t.isoformat() if t else None,
                     'first_fit_unit': first[1] if first else None, 'first_fit_start_utc': first[0].isoformat() if first else None,
                     'n_fit_units': len(fits), 'committed_before_first_fit': ok})
        if ok is False:
            issues.append(f'{label}: first fit {first[1]} at {first[0].isoformat()} precedes commit {c}')
    rec = json.loads((V.RES / 'RECEIPTS.json').read_text())['receipts'][0]
    reg_t, rcp_t = commit_time(rec['research_commit']), commit_time(rec['receipt_commit'])
    lock_t = datetime.fromisoformat(json.loads((V.RES / 'DEVGRID_CODE_LOCK.json').read_text())['utc'])
    grid_units = [s for s in starts if s[2] == 'readers' or s[1].startswith('OHN_GRID_')]
    out = {'check': 'VER_PROVENANCE', 'utc': V.utc(), 'revisions': rows, 'issues': issues,
           'registration_commit_utc': reg_t.isoformat(), 'receipt_commit_utc': rcp_t.isoformat(), 'receipt_verified_utc': rec['verified_utc'],
           'devgrid_code_lock_utc': lock_t.isoformat(), 'first_reader_grid_unit': (grid_units[0][1], grid_units[0][0].isoformat()) if grid_units else None,
           'code_lock_before_first_grid_unit': (not grid_units) or lock_t <= grid_units[0][0],
           'status': 'PASS' if not issues else 'ATTENTION',
           'summary': f"{sum(1 for r in rows if r['committed_before_first_fit'])} revisions committed before their first fit; issues {len(issues)}"}
    V.write_record('VER_PROVENANCE', out)
    print(json.dumps(out, indent=1, default=str))


if __name__ == '__main__':
    main()
