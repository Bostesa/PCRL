"""Write the Terminal-1 handoff status atomically. Never touches Terminal 2's files."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

WT = Path('/Users/nathansamson/PCRL-terminal-1-replacement')
sys.path.insert(0, str(WT))
from experiments.pcrl_stochastic_replacement_overnight_v1 import ledger as L  # noqa: E402

RES = WT / 'results' / 'pcrl_stochastic_replacement_overnight_v1'
PROTOCOL_FILES = ('PROTOCOL.md', 'METHOD.md', 'STATISTICAL_PLAN.md', 'DATA_USE.md',
                  'RUN_MATRIX.json', 'RELEASE_CONTRACT.md', 'PRECURSOR_CORRECTIONS.md')


def git(*args):
    return subprocess.run(['git', '-C', str(WT), *args], capture_output=True,
                          text=True).stdout.strip()


def protocol_hash() -> dict:
    parts = {}
    h = hashlib.sha256()
    for name in PROTOCOL_FILES:
        p = RES / name
        if p.exists():
            d = L.sha_file(p)
            parts[name] = d
            h.update(name.encode())
            h.update(bytes.fromhex(d))
    return {'combined': h.hexdigest(), 'files': parts}


def write(stage, next_action, blockers=None, extra=None):
    gcd = Path(git('rev-parse', '--git-common-dir')).resolve()
    led = L.Ledger(RES / 'ledger')
    payload = {
        'terminal': 1,
        'role': 'experiments',
        'branch': git('rev-parse', '--abbrev-ref', 'HEAD'),
        'commit': git('rev-parse', 'HEAD'),
        'commit_short': git('rev-parse', '--short', 'HEAD'),
        'protocol_hash': protocol_hash(),
        'stage': stage,
        'next_action': next_action,
        'blockers': blockers or [],
        'unit_counts': led.counts(),
        'active_owned_cloud_resources': (extra or {}).get('cloud', []),
        'run_lock': {
            'path': str(RES / 'run.lock'),
            'holder': L.read_json(RES / 'run.lock', None),
            'holder_alive': L.lock_holder_alive(RES / 'run.lock'),
        },
        'terminal_2_seen': L.read_terminal_2(gcd),
        'ceilings': {'wall_clock_hours': 10, 'aws_usd': 50, 'start_utc': '2026-09-21T06:39:48Z',
                     'absolute_compute_deadline_utc': '2026-09-21T16:00:00Z',
                     'closeout_begins_utc': '2026-09-21T15:40:00Z'},
        'sealed': {'2016': 'sealed', '2017': 'not opened'},
    }
    if extra:
        payload.update({k: v for k, v in extra.items() if k != 'cloud'})
    p = L.write_status(gcd, payload)
    print('status ->', p)
    print('commit', payload['commit_short'], '| protocol', payload['protocol_hash']['combined'][:16])
    print('units', payload['unit_counts'])
    return payload


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'unknown'
    nxt = sys.argv[2] if len(sys.argv) > 2 else 'unspecified'
    blockers = sys.argv[3:] or None
    write(stage, nxt, blockers)
