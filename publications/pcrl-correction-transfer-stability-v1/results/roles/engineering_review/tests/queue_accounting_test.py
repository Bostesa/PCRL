"""Engineering review (CTS1): invented-data tests of cts.jobqueue accounting. No person data, no label, no real unit.

    python3 results/pcrl_correction_transfer_stability_v1/roles/engineering_review/tests/queue_accounting_test.py

Each scenario loads a COPY of cts/jobqueue.py (the committed file, or the committed file with PATCHES/20-22 applied) from its own
temporary directory, so the copy's ledger, slots and code manifest live in that directory and never touch the campaign queue.
Every scenario runs in a fresh Python process (a killed-but-unreaped child of one scenario must not be reaped by the next one).
Scenarios:
  timeout   a unit that burns CPU past max_wall_s=1 is killed: is its CPU charged? (committed: kill without wait -> not reaped)
  nan       charge(nan) after charges above the 14 CPU-h limit: is the next start still refused?
  negative  a negative charge offsetting earlier charges: is the next start still refused?
  drift     a code file changes while the unit runs (after launch): does the ledger record it?
"""
from __future__ import annotations
import datetime as dt
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

WT = Path(__file__).resolve().parents[5]
PATCHES = Path(__file__).resolve().parents[1] / 'PATCHES'
BURN = 'import time\nt = time.process_time()\nwhile time.process_time() - t < 3.0:\n    pass\n'


def load(copy_dir):
    spec = importlib.util.spec_from_file_location('jq_copy', copy_dir / 'cts' / 'jobqueue.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    m.CAMPAIGN_START_UTC = dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')
    return m


def scenario(name, copy_dir):
    Q = load(Path(copy_dir))
    out = {}
    if name == 'timeout':
        rec = Q.run('T_TIMEOUT', [sys.executable, '-c', BURN], 'test', 0.1, max_wall_s=1)
        out = {'status': rec['status'], 'returncode': rec['returncode'], 'wall_s': rec['wall_s'], 'charged_cpu_s': rec['charged_cpu_s']}
    elif name in ('nan', 'negative'):
        Q.charge('BIG', Q.CPU_LIMIT_S + 10, 'invented: budget used')
        try:
            Q.charge('BAD', float('nan') if name == 'nan' else -(Q.CPU_LIMIT_S + 10), 'invented: corrupt charge')
            out['bad_charge_accepted'] = True
        except ValueError:
            out['bad_charge_accepted'] = False
        out['charged_cpu_s_seen'] = str(Q.charged_cpu_s())
        try:
            Q.run('T_AFTER_BUDGET', [sys.executable, '-c', 'pass'], 'test', 0.1, max_wall_s=30)
            out['start_refused'] = False
        except PermissionError:
            out['start_refused'] = True
    elif name == 'drift':
        mod = Path(copy_dir) / 'cts' / 'dummy_lazy.py'
        mod.write_text('X = 1\n')

        def edit():
            time.sleep(1.5); mod.write_text('X = 2  # edited while the unit runs\n')
        th = threading.Thread(target=edit); th.start()
        rec = Q.run('T_DRIFT', [sys.executable, '-c', 'import time; time.sleep(3)'], 'test', 0.1, max_wall_s=30)
        th.join()
        out = {'recorded_fields': sorted(k for k in rec if 'code' in k), 'code_changed_during_run': rec.get('code_changed_during_run')}
    print(json.dumps(out))


def make_copy(root, label, patches):
    d = Path(root) / label
    (d / 'cts').mkdir(parents=True)
    shutil.copy(WT / 'cts' / 'jobqueue.py', d / 'cts' / 'jobqueue.py')
    for p in patches:
        r = subprocess.run(['patch', '-p1', '-s', '-i', str(PATCHES / p)], cwd=d, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f'{p}: {r.stdout} {r.stderr}')
    return d


def main():
    res, cpu0 = {}, time.process_time()
    with tempfile.TemporaryDirectory() as td:
        for label, patches in (('committed', ()), ('patched', ('20.diff', '21.diff', '22.diff'))):
            res[label] = {}
            for sc in ('timeout', 'nan', 'negative', 'drift'):
                d = make_copy(td, f'{label}_{sc}', patches)
                r = subprocess.run([sys.executable, __file__, '--scenario', sc, str(d)], capture_output=True, text=True, timeout=120)
                try:
                    res[label][sc] = json.loads(r.stdout.strip().splitlines()[-1])
                except Exception:
                    res[label][sc] = {'error': (r.stderr or r.stdout)[-600:]}
    c, p = res['committed'], res['patched']
    checks = {
        'committed_timeout_cpu_lost': c['timeout'].get('charged_cpu_s', 1) < 0.2,
        'patched_timeout_cpu_charged': p['timeout'].get('charged_cpu_s', 0) >= 0.5,
        'committed_nan_charge_disables_budget': c['nan'].get('start_refused') is False,
        'patched_nan_charge_refused_and_budget_holds': p['nan'].get('bad_charge_accepted') is False and p['nan'].get('start_refused') is True,
        'committed_negative_charge_reopens_budget': c['negative'].get('start_refused') is False,
        'patched_negative_charge_refused_and_budget_holds': p['negative'].get('bad_charge_accepted') is False and p['negative'].get('start_refused') is True,
        'committed_drift_not_recorded': not c['drift'].get('code_changed_during_run'),
        'patched_drift_recorded': p['drift'].get('code_changed_during_run') == ['cts/dummy_lazy.py'],
    }
    print(json.dumps({'checks': checks, 'all_expected': all(checks.values()), 'results': res,
                      'driver_cpu_s': round(time.process_time() - cpu0, 2)}, indent=1))
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--scenario':
        scenario(sys.argv[2], sys.argv[3])
    else:
        sys.exit(main())
