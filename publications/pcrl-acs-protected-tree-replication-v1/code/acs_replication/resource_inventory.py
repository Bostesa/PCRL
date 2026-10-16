"""Build RESOURCE_INVENTORY.json from the campaign queue ledger (closeout; no data access).

Usage: PYTHONPATH=. python -m acs_replication.resource_inventory --out results/.../RESOURCE_INVENTORY.json [--close]

Everything is derived from .private/acs_replication/compute/ledger.jsonl:
  * per-unit wall time, charged CPU (child user+sys) and sampled peak RSS;
  * the maximum number of simultaneously running queue units (from start/end intervals);
  * outside-queue charges (agents, provenance searches, light scripts), which are upper estimates.
``child_maxrss_bytes_platform`` on macOS is the running maximum over all reaped children of the queue
process, so it is reported only as an upper bound; ``sampled_peak_rss_bytes`` is the per-unit figure.
"""
import argparse
import datetime as dt
import json
from pathlib import Path

from . import jobqueue as Q


def _t(s):
    return dt.datetime.fromisoformat(s)


def build(close=False):
    es = Q.entries()
    starts = {e['unit_id']: e for e in es if e.get('event') == 'unit_start'}
    ends = [e for e in es if e.get('event') == 'unit_end']
    units = []
    for e in ends:
        s = starts.get(e['unit_id'], {})
        units.append({'unit_id': e['unit_id'], 'stage': e['stage'], 'status': e['status'],
                      'start_utc': s.get('utc'), 'end_utc': e['utc'], 'wall_s': e['wall_s'],
                      'charged_cpu_s': e['charged_cpu_s'], 'sampled_peak_rss_gib': round(e['sampled_peak_rss_bytes'] / 2**30, 3),
                      'rss_reservation_gib': e['rss_reservation_gib']})
    events = []
    for u in units:
        if u['start_utc']:
            events.append((_t(u['start_utc']), 1)); events.append((_t(u['end_utc']), -1))
    events.sort(key=lambda x: (x[0], x[1]))
    cur = peak = 0
    for _, d in events:
        cur += d; peak = max(peak, cur)
    charges = [e for e in es if e.get('event') == 'charge']
    start = Q.start_utc()
    now = Q.now()
    q_cpu = sum(u['charged_cpu_s'] for u in units)
    c_cpu = sum(float(c['charged_cpu_s']) for c in charges)
    out = {
        'campaign': 'pcrl_acs_protected_tree_replication_v1',
        'start_utc': start.isoformat(timespec='seconds'),
        'inventory_utc': now.isoformat(timespec='seconds'),
        'elapsed_h': round((now - start).total_seconds() / 3600, 3),
        'limits': {'cpu_h': Q.CPU_LIMIT_S / 3600, 'elapsed_h': Q.ELAPSED_LIMIT_H, 'rss_gib_per_job': Q.RSS_LIMIT_GIB,
                   'heavy_job_slots': Q.SLOTS, 'threads_per_job': 1, 'paid_cloud_usd': 0},
        'queue_units': {'done': sum(u['status'] == 'done' for u in units), 'failed': sum(u['status'] != 'done' for u in units),
                        'failed_unit_ids': [u['unit_id'] for u in units if u['status'] != 'done'],
                        'cpu_s': round(q_cpu, 1), 'max_concurrent_units': peak,
                        'max_sampled_peak_rss_gib': max((u['sampled_peak_rss_gib'] for u in units), default=0.0),
                        'max_platform_child_maxrss_gib_upper_bound': round(max((e['child_maxrss_bytes_platform'] for e in ends), default=0) / 2**30, 3)},
        'outside_queue_charges': [{'label': c['label'], 'cpu_s': c['charged_cpu_s'], 'note': c['note'], 'utc': c['utc']} for c in charges],
        'charged_total': {'cpu_s': round(q_cpu + c_cpu, 1), 'cpu_h': round((q_cpu + c_cpu) / 3600, 3),
                          'failed_attempts_included': True, 'agent_work_included_as_upper_estimates': True},
        'paid_cloud_usd': 0,
        'external_messages_or_submissions': 0,
        'units': units,
    }
    out['within_limits'] = {'cpu': out['charged_total']['cpu_h'] <= out['limits']['cpu_h'],
                            'elapsed': out['elapsed_h'] <= out['limits']['elapsed_h'],
                            'concurrency': peak <= Q.SLOTS,
                            'rss': out['queue_units']['max_sampled_peak_rss_gib'] <= Q.RSS_LIMIT_GIB}
    if close:
        out['queue_closed_utc'] = now.isoformat(timespec='seconds')
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--close', action='store_true', help='record the queue as closed (no further units are run)')
    a = ap.parse_args(argv)
    rec = build(close=a.close)
    if a.close:
        Q._append({'event': 'close', 'utc': rec['queue_closed_utc'], 'charged_cpu_s': 0.0})
    Path(a.out).write_text(json.dumps(rec, indent=2) + '\n')
    print(json.dumps({k: rec[k] for k in ('elapsed_h', 'charged_total', 'within_limits')}))


if __name__ == '__main__':
    main()
