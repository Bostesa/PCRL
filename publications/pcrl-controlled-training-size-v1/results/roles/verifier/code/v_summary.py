"""Print a summary of CHECKS.json (counts by kind and status; every FAIL / ATTENTION / NOT_RUN with its id) and CHECK_CHANGES.json.
No data is read. PCRL controlled training-size campaign (TSZ1), independent verifier."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vlib as V  # noqa: E402


def main():
    doc = json.loads(V.CHECKS.read_text())
    print(json.dumps(doc['summary'], indent=1))
    for st in ('FAIL', 'ATTENTION', 'NOT_RUN'):
        for c in doc['checks']:
            if c['status'] == st:
                print(st, c['kind'], c['id'], c['utc'])
    if V.CHECK_CHANGES.is_file():
        for ch in json.loads(V.CHECK_CHANGES.read_text())['changes']:
            print('CHANGE', ch['n'], ch['check_id_before'], '->', ch['check_id_after'], '|', ch.get('adjudication'))
    print('ids:', len(doc['checks']))


if __name__ == '__main__':
    main()
