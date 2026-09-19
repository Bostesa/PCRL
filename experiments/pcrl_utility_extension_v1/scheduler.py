"""Detached tier scheduler (systemd service on the cloud host). Machine-readable gates drive it.

Each tier writes gates/T<k>.json with PASS / FAIL / UNASSESSABLE, inputs and hashes, exact
counts, elapsed time and the next action. A FAIL or UNASSESSABLE gate, the wall-clock deadline
or the estimated-cost guard takes the closeout path (final archive + verify + stop). A tier whose
complete comparison will not fit the remaining time is not started; the closeout records the
exact resume command. Completed units are reused on restart; nothing is refitted.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

CFG = json.loads(Path(os.environ['PCRL_UX_RUNCFG']).read_text())
OUT = Path(os.environ['PCRL_UX_OUT'])
SCHED = OUT / '_scheduler'
GATES = SCHED / 'gates'
WORKERS = int(CFG.get('workers', 6))


def utc():
    return datetime.now(timezone.utc).isoformat()


def log(*msg):
    line = f'{utc()} ' + ' '.join(str(m) for m in msg)
    print(line, flush=True)
    with open(SCHED / 'scheduler.log', 'a') as fh:
        fh.write(line + '\n')


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, indent=1, default=str))
    os.replace(tmp, path)


def elapsed_h():
    return (time.time() - CFG['remote_start_epoch']) / 3600


def cost_estimate():
    return elapsed_h() * (CFG['hourly_usd'] + CFG['ebs_hourly_usd'])


def remaining_h():
    return CFG['deadline_epoch'] / 3600 - time.time() / 3600 - CFG['closing_reserve_h']


def guard(next_tier_h: float) -> str | None:
    if cost_estimate() + next_tier_h * (CFG['hourly_usd'] + CFG['ebs_hourly_usd']) > CFG['compute_ceiling_usd']:
        return 'cost_guard'
    if remaining_h() < next_tier_h:
        return 'time_guard'
    return None


def sync(label):
    r = subprocess.run(['aws', 's3', 'sync', str(OUT), f's3://{CFG["bucket"]}/{CFG["prefix"]}/run/results',
                        '--only-show-errors', '--sse', 'AES256'], capture_output=True, text=True)
    log('SYNC', label, 'rc', r.returncode, r.stderr[-300:])
    return r.returncode == 0


def gate(tier, status, **info):
    rec = {'tier': tier, 'status': status, 'utc': utc(), 'elapsed_h': round(elapsed_h(), 3),
           'cost_estimate_usd': round(cost_estimate(), 2), **info}
    write(GATES / f'{tier}.json', rec)
    log('GATE', tier, status, info.get('next_action'))
    sync(f'gate_{tier}')
    return rec


def pull_code():
    r = subprocess.run(['git', '-C', CFG['repo'], 'pull', '--ff-only', '-q'], capture_output=True, text=True)
    head = subprocess.run(['git', '-C', CFG['repo'], 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    log('CODE', head, r.returncode, r.stderr[-200:])
    return head


# ------------------------------------------------------------------ worker entry points
def _init_worker():
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, CFG['repo'])


def _fit(args):
    seed, name = args
    from experiments.pcrl_utility_extension_v1 import program as pg, extension as ext
    try:
        r = pg.fit_job(seed, name, ext.Config())
        return {'seed': seed, 'unit': name, 'status': r.get('status'), 'step': r.get('selected_step')}
    except Exception as exc:
        return {'seed': seed, 'unit': name, 'status': 'ERROR', 'error': repr(exc), 'tb': traceback.format_exc()[-2000:]}


def _audit(args):
    seed, name = args
    from experiments.pcrl_utility_extension_v1 import program as pg
    try:
        r = pg.audit_job(seed, name)
        return {'seed': seed, 'unit': name, 'duplicate_of': r.get('duplicate_of'), 'failed': r.get('failed', False)}
    except Exception as exc:
        return {'seed': seed, 'unit': name, 'failed': True, 'error': repr(exc), 'tb': traceback.format_exc()[-2000:]}


def _calib(args):
    seed, recipe = args
    import numpy as np
    from experiments.pcrl_utility_extension_v1 import calibrate as cal, program as pg
    dest = OUT / 'calibration' / f'seed_{seed}' / f'{recipe}.json'
    if dest.exists():
        return {'seed': seed, 'recipe': recipe, 'reused': True}
    rel = {k: dict(np.load(pg.OUT / f'seed_{seed}' / 'releases' / f'ref_{k}' / 'releases.npz'))
           for k in ('A0', 'J', 'leace_A0')}
    try:
        res = cal.calibrate_seed(seed, rel, recipes={recipe: cal.RECIPES[recipe]})
    except Exception as exc:
        return {'seed': seed, 'recipe': recipe, 'error': repr(exc), 'tb': traceback.format_exc()[-2000:]}
    write(dest, res)
    return {'seed': seed, 'recipe': recipe, 'seconds': res['_seconds']}


def pool(fn, jobs, label):
    tick = time.time()
    results = []
    with cf.ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_worker) as ex:
        for r in ex.map(fn, jobs):
            results.append(r)
            if r.get('status') == 'ERROR' or r.get('failed'):
                log('JOB_FAIL', label, r)
    log('POOL', label, len(jobs), 'jobs', round(time.time() - tick, 1), 's')
    write(SCHED / f'pool_{label}.json', {'jobs': len(jobs), 'seconds': time.time() - tick, 'results': results})
    return results


# ------------------------------------------------------------------ tiers
def tier0():
    from experiments.pcrl_utility_extension_v1 import tier0 as t0
    res = t0.run()
    write(SCHED / 'tier0_smoke.json', res)
    vdir = Path(CFG['verify_dir'])
    exec_ok = all(json.loads(p.read_text()).get('verified') for p in vdir.glob('exec_*.verify.json')) and \
        len(list(vdir.glob('exec_*.verify.json'))) == CFG['exec_chunks']
    forecast = cost_estimate() + CFG['forecast_hours'] * (CFG['hourly_usd'] + CFG['ebs_hourly_usd'])
    ok = res['pass'] and exec_ok and forecast <= CFG['compute_ceiling_usd'] and CFG['independent_stop_installed']
    return gate('T0', 'PASS' if ok else 'FAIL', smoke_pass=res['pass'], exec_bundle_verified=exec_ok,
                forecast_compute_usd=round(forecast, 2), independent_stop=CFG['independent_stop_installed'],
                next_action='T1' if ok else 'closeout')


def tier1():
    from experiments.pcrl_utility_extension_v1 import program as pg, extension as ext, calibrate as cal
    from experiments.pcrl_utility_extension_v1 import tier1_reanalysis as ra
    import numpy as np
    comp = Path(CFG['repo']) / 'results/pcrl_competitive_method_v1'
    rea = ra.run(comp / 'INTERVALS_X.csv', OUT / 'TIER1_REANALYSIS.json', OUT / 'TIER1_REANALYSIS.md')
    for s in pg.SEEDS:
        pg.references(s, ext.Config())
    refs = [(s, n) for s in pg.SEEDS for n in ('ref_J', 'ref_A0', 'ref_leace_A0')]
    audits = pool(_audit, refs, 'T1_ref_audits')
    calib = pool(_calib, [(s, r) for s in pg.SEEDS for r in cal.RECIPES], 'T1_calibration')
    bad = [a for a in audits if a.get('failed')] + [c for c in calib if 'error' in c]
    if bad:
        return gate('T1', 'FAIL', reason='reference audit or calibration failed', failures=bad, next_action='closeout')
    # standard-slate validation losses, positive control and portability
    weights = pg.WEIGHTS
    standard, control, portab = {}, {}, {}
    from experiments.pcrl_direct_adversarial_v1 import report as rep
    for s in pg.SEEDS:
        standard[s] = {}
        for iname, unit in (('A0', 'ref_A0'), ('J', 'ref_J'), ('leace_A0', 'ref_leace_A0'), ('H', 'H')):
            standard[s][iname] = {f'{e}|{w}': pg.point(s, unit, 'validation', w)['losses'][e]
                                  for e in cal.ENDPOINTS for w in weights}
        for w in weights:
            a0, h = pg.point(s, 'ref_A0', 'validation', w), pg.point(s, 'H', 'validation', w)
            control[(s, w)] = {e: a0['gains'][e] - h['gains'][e] for e in cal.ENDPOINTS}
            for unit, hist in (('ref_A0', 'A0'), ('ref_J', 'J')):
                cloud = pg.point(s, unit, 'test', w)
                old = rep.point_from_records(json.loads(rep.condition_file(Path('/x'), s, hist, 'metrics.json').read_text())['raw_metrics'],
                                             pg.prior(s), 'test', w, pg.MAIN_BUDGET, pg.MAIN_SCOPE)
                d = [abs(cloud['gains'][e] - old['gains'][e]) for e in cal.ENDPOINTS] + \
                    [abs(cloud['utility'][t] - old['utility'][t]) for t in cloud['utility']]
                portab[f'{unit}|{s}|{w}'] = max(d)
    calib_all = {s: {r: json.loads((OUT / 'calibration' / f'seed_{s}' / f'{r}.json').read_text())[r]
                     for r in cal.RECIPES} for s in pg.SEEDS}
    decision = cal.decide(calib_all, standard, pg.SEEDS)
    pos = {e: {w: float(np.mean([control[(s, w)][e] for s in pg.SEEDS])) for w in weights} for e in cal.ENDPOINTS}
    competent = all(pos[e][w] > 0.01 for e in ('A/SEX', 'A/RAC1P') for w in weights)
    port_ok = max(portab.values()) <= CFG['audit_portability_tol']
    write(OUT / 'CALIBRATION.json', {'decision': decision, 'positive_control_A0_minus_H_validation': pos,
                                     'audit_portability_max_abs': portab, 'audit_portability_tol': CFG['audit_portability_tol'],
                                     'audit_portability_ok': port_ok})
    status = 'PASS' if competent else 'FAIL'
    return gate('T1', status, reanalysis={k: rea[k] for k in ('new_family_size', 'n_pass_vs_J', 'n_pass_vs_leace_A0', 'n_pass_both')},
                positive_control=pos, competent=competent, identity_checks='asserted at reference build',
                adopted_recipe=decision['adopted'], stress_strength_established=decision['stress_strength_established'],
                audit_portability_ok=port_ok, next_action='T2' if competent else 'closeout')


def fit_and_audit(names, label):
    from experiments.pcrl_utility_extension_v1 import program as pg
    fits = pool(_fit, [(s, n) for s in pg.SEEDS for n in names if not n.startswith('LEACE')], f'{label}_fits')
    leace = [(s, n) for s in pg.SEEDS for n in names if n.startswith('LEACE')]
    if leace:
        fits += pool(_fit, leace, f'{label}_leace_fits')
    ok = [(f['seed'], f['unit']) for f in fits if f.get('status') == 'FITTED']
    audits = pool(_audit, ok, f'{label}_audits')
    return fits, audits


def tier2():
    from experiments.pcrl_utility_extension_v1 import program as pg
    names = pg.units(pg.PILOT_R)
    fits, audits = fit_and_audit(names, 'T2')
    rows = pg.screen(names)
    write(OUT / 'PILOT_SCREEN.json', rows)
    x = {n: r for n, r in rows.items() if n.startswith('X_')}
    if any(r['pass'] is None for r in x.values()):
        status = 'UNASSESSABLE'
    else:
        status = 'PASS' if any(r['pass'] for r in x.values()) else 'FAIL'
    return gate('T2', status, passing=[n for n, r in x.items() if r.get('pass')],
                fitted=sum(f.get('status') == 'FITTED' for f in fits),
                fit_errors=[f for f in fits if f.get('status') == 'ERROR'],
                audited=len(audits), duplicates=sum(1 for a in audits if a.get('duplicate_of')),
                audit_failures=[a for a in audits if a.get('failed')],
                next_action='T3' if status == 'PASS' else 'pilot_closeout_report')


def tier3():
    from experiments.pcrl_utility_extension_v1 import program as pg
    names = pg.units(pg.FULL_R, leace=True)
    fits, audits = fit_and_audit(names, 'T3')
    rows = pg.screen(names)
    selected = pg.select_tier3(rows)
    sel = {'utc': utc(), 'selected': selected, 'rule': pg.select_tier3.__doc__,
           'read_test_split': False, 'screen': rows}
    write(OUT / 'SELECTION.json', sel)
    sync('selection_frozen')
    return gate('T3', 'PASS' if selected else 'FAIL', selected=selected,
                fitted=sum(f.get('status') == 'FITTED' for f in fits), audited=len(audits),
                next_action='T4' if selected else 'closeout_no_nominee')


def closeout(reason):
    log('CLOSEOUT', reason)
    write(SCHED / 'CLOSEOUT.json', {'reason': reason, 'utc': utc(), 'elapsed_h': elapsed_h(),
                                    'cost_estimate_usd': cost_estimate(),
                                    'resume': f'sudo systemctl start pcrl-ux (resumes from completed units; gates in {GATES})'})
    synced = sync('closeout')
    arch = subprocess.run([sys.executable, str(Path(CFG['repo']) / 'infra/pcrl_utility_extension_v1/closeout.py')],
                          capture_output=True, text=True)
    log('ARCHIVE', arch.returncode, arch.stdout[-1500:], arch.stderr[-800:])
    write(SCHED / 'CLOSEOUT.json', {'reason': reason, 'utc': utc(), 'elapsed_h': elapsed_h(),
                                    'cost_estimate_usd': cost_estimate(), 'sync_ok': synced,
                                    'archive_rc': arch.returncode})
    sync('closeout_final')
    if CFG.get('stop_on_closeout', True):
        subprocess.run(['sudo', 'shutdown', '-h', '+2'])


def wait_for_start():
    """The scheduler begins only after launch.py has rehearsed the independent stop and written
    run/START. If START never appears before the deadline window, close out without fitting."""
    while True:
        r = subprocess.run(['aws', 's3', 'ls', f's3://{CFG["bucket"]}/{CFG["prefix"]}/run/START'],
                           capture_output=True, text=True)
        if r.returncode == 0 and 'START' in r.stdout:
            log('START flag present')
            return True
        if remaining_h() < 2.0:
            return False
        time.sleep(60)


def main():
    SCHED.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, CFG['repo'])
    if not wait_for_start():
        return closeout('START flag never written')
    plan = [('T0', tier0, 0.2), ('T1', tier1, 0.8), ('T2', tier2, 1.0), ('T3', tier3, 1.5)]
    for tier, fn, need_h in plan:
        done = GATES / f'{tier}.json'
        if done.exists():
            rec = json.loads(done.read_text())
            if rec['status'] == 'PASS':
                log('REUSE_GATE', tier)
                continue
            return closeout(f'{tier} {rec["status"]} (previous run)')
        why = guard(need_h)
        if why:
            write(GATES / f'{tier}.json', {'tier': tier, 'status': 'NOT_STARTED', 'reason': why,
                                           'resume': 'restart pcrl-ux after raising the budget'})
            return closeout(f'{why} before {tier}')
        pull_code()
        try:
            rec = fn()
        except Exception as exc:
            gate(tier, 'FAIL', reason='exception', error=repr(exc), tb=traceback.format_exc()[-3000:], next_action='closeout')
            return closeout(f'{tier} exception')
        if rec['status'] != 'PASS':
            return closeout(f'{tier} {rec["status"]}')
    # T4 is loaded from the pulled branch if present (implemented only if T3 selects a nominee).
    pull_code()
    try:
        from experiments.pcrl_utility_extension_v1 import tier4
        if remaining_h() < 1.0:
            return closeout('time_guard before T4')
        rec = tier4.run(OUT, gate)
        return closeout(f'T4 {rec["status"]}')
    except ImportError:
        gate('T4', 'NOT_STARTED', reason='tier4 module not present on the branch', next_action='closeout')
        return closeout('T4 module absent')


if __name__ == '__main__':
    main()
