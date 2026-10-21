"""Coordinator's view of the campaign queue: running units, the last finished units, waiting tickets, CPU used.

    python3 results/pcrl_dual_design_release_v1/roles/coordinator/queue_status.py
"""
import datetime as dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
Q = ROOT / '.private' / 'pcrl_dual_design_release_v1' / 'compute'
es = [json.loads(l) for l in (Q / 'ledger.jsonl').read_text().splitlines() if l.strip()]
st = {e['unit_id']: e for e in es if e.get('event') == 'unit_start'}
en = {e['unit_id']: e for e in es if e.get('event') == 'unit_end'}
print('utc', dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds'))
print('running:', [(u, st[u]['utc'][11:19]) for u in st if u not in en])
for u in list(en)[-8:]:
    print('  done', u, en[u]['status'], en[u]['charged_cpu_s'], en[u]['utc'][11:19])
print('waiting:', sorted(p.name.split('_', 2)[-1] for p in (Q / 'tickets').iterdir()))
cpu = sum(float(e.get('charged_cpu_s', 0)) for e in es if e.get('event') in ('unit_end', 'charge'))
print(f'cpu charged {cpu:.0f} s = {cpu / 3600:.3f} h; failed', [u for u, e in en.items() if e['status'] != 'done'])
