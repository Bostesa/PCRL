"""Verifier unit AMD_VERIFY_ACCESS_V1 (label-free): check 1 evidence from the access logs, the queue ledger and static code
inspection of every load_role call in amd/*.py."""
from __future__ import annotations
import json
import re
import sys

sys.path.insert(0, '<repository>/.claude/worktrees/pcrl-autonomous-method-development-v1/.private/pcrl_autonomous_method_development_v1/verifier/independent')
import vcommon as V
from amd import common as C

out = {'unit': 'AMD_VERIFY_ACCESS_V1', 'utc': C.now()}
dev = [json.loads(l) for l in (V.PRIV / 'access' / 'DEV_REPORT_REREAD.jsonl').read_text().splitlines() if l.strip()]
ms = [json.loads(l) for l in (V.PRIV / 'access' / 'MODEL_SELECT_USE.jsonl').read_text().splitlines() if l.strip()]
out['dev_events'] = [{k: e.get(k) for k in ('event', 'utc', 'unit_id', 'reporting_lock_sha256')} for e in dev]
out['ms_events'] = [{k: e.get(k) for k in ('utc', 'unit_id', 'purpose')} for e in ms]
src = C.SRC_ACCESS_LOG
srcev = [json.loads(l) for l in src.read_text().splitlines() if l.strip()] if src.is_file() else []
out['source_access_log_events_this_campaign'] = [{k: e.get(k) for k in ('event', 'utc', 'unit_id')} for e in srcev if e.get('campaign') == C.CAMPAIGN]
out['source_access_log_sealed_mentions'] = [e for e in srcev if 'SEALED' in json.dumps(e).upper() and 'UNTOUCHED' not in json.dumps(e).upper()]
L = [json.loads(l) for l in (V.PRIV / 'compute' / 'ledger.jsonl').read_text().splitlines() if l.strip()]
starts = {e['unit_id']: e for e in L if e.get('event') == 'unit_start'}
ends = {e['unit_id']: e for e in L if e.get('event') == 'unit_end'}
units = []
for u, e in starts.items():
    units.append({'unit': u, 'start': e['utc'], 'stage': e.get('stage'), 'end': ends.get(u, {}).get('utc'), 'status': ends.get(u, {}).get('status'),
                  'argv': ' '.join(e['argv'])[-160:]})
out['units'] = units
out['n_units'] = len(units)
out['failed_units'] = [x['unit'] for x in units if x['status'] not in ('done',)]
out['sealed_in_any_argv'] = [x['unit'] for x in units if 'SEALED' in x['argv'].upper()]
out['charged_cpu_s_total'] = sum(e.get('charged_cpu_s', 0) for e in L if e.get('event') == 'unit_end') + sum(e.get('cpu_s', 0) for e in L if e.get('event') == 'charge')
# MODEL_SELECT label reads by unit stage
st_of = {u: starts[u].get('stage') for u in starts}
out['ms_reads_by_stage'] = {}
for e in ms:
    st = st_of.get(e.get('unit_id'), 'UNKNOWN_UNIT')
    out['ms_reads_by_stage'].setdefault(st, []).append(e.get('unit_id'))
# static inspection of load_role / label loaders in amd/*.py
calls = []
for p in sorted((V.WT / 'amd').glob('*.py')):
    for i, line in enumerate(p.read_text().splitlines(), 1):
        if re.search(r"load_role\(|dev_report_truth\(|ATTACK_FIT|ATTACK_VALIDATION|SEALED", line):
            calls.append({'file': p.name, 'line': i, 'text': line.strip()[:220]})
out['static_role_lines'] = calls
out['attack_sex_reads'] = [c for c in calls if 'load_role' in c['text'] and 'ATTACK' in c['text'] and 'sex' in c['text']]
V.dump('V_ACCESS.json', out)
print(json.dumps({'dev_events': out['dev_events'], 'n_ms': len(ms), 'ms_by_stage': {k: len(v) for k, v in out['ms_reads_by_stage'].items()},
                  'attack_sex_reads': out['attack_sex_reads'], 'failed_units': out['failed_units'], 'sealed': out['sealed_in_any_argv'],
                  'src_events': out['source_access_log_events_this_campaign'], 'sealed_mentions': len(out['source_access_log_sealed_mentions']),
                  'cpu_s': out['charged_cpu_s_total']}, indent=1))
