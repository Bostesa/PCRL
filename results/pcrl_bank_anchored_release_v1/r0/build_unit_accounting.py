"""Publish path-free aggregate accounting from the private durable queue."""
from __future__ import annotations
import collections
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
state=json.loads((ROOT/'r0/.private/compute/state.json').read_text())
units=state['units']
status=collections.Counter(u['status'] for u in units.values())
by_stage=collections.Counter((u.get('stage',u.get('spec',{}).get('stage','unknown')),u['status']) for u in units.values())
known_scientific={'R2_FIT_UNITS':sum(u['status']=='done' and uid.startswith('R2_') and '_FIT_' in uid for uid,u in units.items()),
                  'R4_EFFECTIVE_ATTACK_UNITS':sum(u['status']=='done' and uid.startswith('R4_ATTACK_') for uid,u in units.items()),
                  'R5_NUMERICAL_UNITS_DONE':sum(u['status']=='done' and uid.startswith('R5_') for uid,u in units.items())}
obj={'schema':'BAR1-global-queue-unit-accounting-v1',
     'status':'FINAL_CLOSEOUT_IF_QUEUE_CLOSED' if state.get('closed') else 'INTERIM',
     'queue_closed':bool(state.get('closed')),
     'queue_close_reason':state.get('close_reason'),
     'unit_count_registered':len(units),
     'unit_status_counts':dict(sorted(status.items())),
     'unit_stage_status_counts':[{ 'stage':stage,'status':st,'count':n} for (stage,st),n in sorted(by_stage.items())],
     'queued_not_run_policy':'Superseded earlier versions or conditional units remain queued in the immutable queue record; queue closure prevents their execution. These are not completed science.',
     'failed_unit_ids':sorted(uid for uid,u in units.items() if u['status']=='failed'),
     'queued_unit_ids':sorted(uid for uid,u in units.items() if u['status']=='queued'),
     'actual_scientific_counts':known_scientific,
     'queue_charged_cpu_s':state['used_cpu_s'],
     'queue_charged_cpu_h':state['used_cpu_s']/3600,
     'queue_charged_by_stage_s':state['used_by_stage_s'],
     'amendment_count':len(state.get('amendments',[])),
     'external_heavy_reserved_last_receipt':state.get('occupancy',{}).get('external_heavy'),
     'heavy_process_limit_machine_wide':2,
     'local_heavy_process_limit_with_external_slot':1,
     'source_scope':'Private queue state summarized without private paths or person-level data. Lightweight engineering commands outside the queue are not included in the measured queue CPU.'}
p=ROOT/'UNIT_ACCOUNTING.json';p.write_text(json.dumps(obj,sort_keys=True,indent=2)+'\n')
print(obj['unit_count_registered'],obj['unit_status_counts'],round(obj['queue_charged_cpu_h'],6))
