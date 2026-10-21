"""Run a list of queue units in order (one 'lane'); stop at the first failed unit (later units depend on earlier ones).

    python3 -m ddr.lane LANE.json

LANE.json: [{"unit": "DDR_..._V1", "stage": "...", "rss_gib": 1.5, "argv": ["python3", "-m", "ddr.fa.route", ...]}, ...]
Each unit goes through ddr.jobqueue.run (slots, threads, budgets, job-start code hashes, ledger). Two lanes at most run at once, so
the two-heavy-job limit is kept by the queue's slot locks. Interpreter aliases: "PY" = python3, "TORCH" = the torch venv,
"CATBOOST" = the CatBoost venv.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

from . import common as C
from . import jobqueue as Q

ALIASES = {'PY': 'python3', 'TORCH': str(C.TORCH_PY), 'CATBOOST': str(C.CATBOOST_PY)}


def run_lane(units):
    done = []
    for u in units:
        argv = [ALIASES.get(a, a) for a in u['argv']]
        if any(e.get('event') == 'unit_end' and e['unit_id'] == u['unit'] and e['status'] == 'done' for e in Q.entries()):
            done.append({'unit': u['unit'], 'status': 'already_done'}); continue
        rec = Q.run(u['unit'], argv, u['stage'], float(u.get('rss_gib', 1.5)), max_wall_s=int(u.get('max_wall_s', 3 * 3600)))
        done.append({'unit': u['unit'], 'status': rec['status'], 'cpu_s': rec['charged_cpu_s'], 'wall_s': rec['wall_s']})
        print(json.dumps(done[-1]), flush=True)
        if rec['status'] != 'done':
            print(json.dumps({'lane_stopped_at': u['unit']}), flush=True)
            return 1
    return 0


def main(argv=None):
    argv = argv or sys.argv[1:]
    units = json.loads(Path(argv[0]).read_text())
    return run_lane(units)


if __name__ == '__main__':
    sys.exit(main())
