"""Summary of the verifier's CHECKS.json (no data read): counts by numerical and procedural status, every non-PASS check with its
statuses, and the integrity of the append-only entry hashes.

    python3 .../v_summary.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlib as V  # noqa: E402


def main():
    doc = json.loads(V.CHECKS.read_text())
    ok_hashes = [V._canon(c) for c in doc['checks']] == doc['entry_sha256']
    rows = []
    for c in doc['checks']:
        rows.append((c['id'], c['numerical_status'], c['procedural_status'], c['utc'], c.get('unit_id')))
    print(json.dumps({'n_checks': len(rows), 'append_only_hashes_intact': ok_hashes, 'summary': doc['summary']}, indent=1))
    for r in rows:
        flag = '' if r[1] in ('PASS', 'NOT_APPLICABLE') and r[2] in ('PASS', 'NOT_APPLICABLE') else '  <--'
        print(f'{r[0]:<60} num={r[1]:<15} proc={r[2]:<15} {r[3]} {r[4]}{flag}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
