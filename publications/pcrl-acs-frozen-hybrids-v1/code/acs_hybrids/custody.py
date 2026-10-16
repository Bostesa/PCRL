"""Custody: a versioned private snapshot of this campaign and a restore-and-use test from the backup alone.

snapshot: copies the campaign's private tree (restored source artifacts, bank, readers, deployment packages, admission detail,
access log, queue ledger) plus the results folder and the campaign code into ~/PCRL_eval_cache_private/acsh_v1/backups/<NAME>,
with a sha256 manifest. restore-check: copies a snapshot into a fresh directory, verifies every hash, then USES it without the
live tree: the restored H_LJ packages re-deploy the restored DEV_REPORT raw inputs and must equal the restored bank outputs, and
restored winning pair readers re-score the restored bank's DEV_REPORT wires and must equal the restored stored scores.
No labels are read.
"""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from . import common as C

HOME_BACKUPS = Path.home() / 'PCRL_eval_cache_private' / 'acsh_v1' / 'backups'
INCLUDE_PRIV = ('source', 'bank', 'runs', 'deploy', 'admission', 'access', 'RESTORE_RECEIPT.json', 'compute/ledger.jsonl', 'compute/START_UTC.json')


def snapshot(name):
    dest = HOME_BACKUPS / name
    if dest.exists():
        raise PermissionError('snapshot exists (versioned, never overwritten)')
    dest.mkdir(parents=True, mode=0o700)
    man = []
    def add(src, rel):
        d = dest / rel; d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, d); man.append({'path': str(rel), 'sha256': C.sha(d), 'bytes': d.stat().st_size, 'source_sha256': C.sha(src)})
    for item in INCLUDE_PRIV:
        p = C.PRIV / item
        if p.is_file():
            add(p, Path('private') / item)
        elif p.is_dir():
            for f in sorted(p.rglob('*')):
                if f.is_file():
                    add(f, Path('private') / f.relative_to(C.PRIV))
    for f in sorted(C.RESULTS.rglob('*')):
        if f.is_file():
            add(f, Path('results') / f.relative_to(C.RESULTS))
    for f in sorted((C.ROOT / 'acs_hybrids').glob('*.py')):
        add(f, Path('code') / 'acs_hybrids' / f.name)
    if any(m['sha256'] != m['source_sha256'] for m in man):
        raise PermissionError('snapshot copy differs from source')
    (dest / 'SNAPSHOT_MANIFEST.json').write_text(json.dumps(man, indent=1))
    return {'snapshot': str(dest).replace(str(Path.home()), '~'), 'files': len(man), 'bytes': sum(m['bytes'] for m in man),
            'manifest_sha256': C.sha(dest / 'SNAPSHOT_MANIFEST.json')}


def restore_check(snapshot_dir, restore_dir):
    from task_conditioned_representation.attacks import predict_reader
    import joblib
    from acs_replication import deploy as base_deploy
    from . import hybrid_deploy, readers
    snap, rest = Path(snapshot_dir), Path(restore_dir)
    if rest.exists():
        raise PermissionError('restore directory must be fresh')
    shutil.copytree(snap, rest)
    man = json.loads((rest / 'SNAPSHOT_MANIFEST.json').read_text())
    bad = [m['path'] for m in man if C.sha(rest / m['path']) != m['sha256']]
    out = {'files': len(man), 'hash_mismatches': len(bad)}
    rr = json.loads((rest / 'private' / 'source' / 'extraction' / 'EXTRACTION_RECEIPT.json').read_text())
    raw_item = rr['files']['DEV_REPORT']['raw_inputs']
    raw_path = rest / 'private' / 'source' / Path(raw_item['path']).relative_to('.private/acs_replication')
    with np.load(raw_path) as z:
        raw = {k: z[k].copy() for k in z.files}
    rows = [{c: raw[c][i] for c in hybrid_deploy.PERMITTED} for i in range(len(raw['row_id']))]
    rlock = json.loads((rest / 'results' / 'REPORTING_LOCK.json').read_text()) if (rest / 'results' / 'REPORTING_LOCK.json').exists() else None
    for s in C.SEEDS:
        with np.load(rest / 'private' / 'bank' / f'FROZEN_BANK_S{s}.npz') as z:
            bank = {k: z[k].copy() for k in z.files if k.startswith(('H_LJ__DEV_REPORT', 'H_LL__', 'H_JJ__', 'H_JL__', 'NEURAL_RAW__', 'TASK_REFIT__', 'T_TREE__'))
                    and '__DEV_REPORT__' in k}
        res = hybrid_deploy.release(rest / 'private' / 'deploy' / f'H_LJ_S{s}', rows)
        out[f'S{s}_package_parity'] = {t: {'max_abs_prob_diff': float(np.max(np.abs(res[t]['probability'] - bank[f'H_LJ__DEV_REPORT__{t}']))),
                                           'decisions_identical': bool(np.array_equal(res[t]['decision'], bank[f'H_LJ__DEV_REPORT__{t}'].argmax(1)))} for t in C.TASKS}
        if rlock:
            rd = rest / 'private' / 'runs' / f'READERS_{s}'
            with np.load(rd / 'DEV_REPORT_SCORES.npz') as z:
                stored = {k: z[k].copy() for k in z.files}
            rec = json.loads((rd / 'READERS_SEED.json').read_text())
            checks = {}
            for svc in ('H_LJ', 'H_LL', 'NEURAL_RAW', 'TASK_REFIT'):
                w = rlock['readers'][f'S{s}']['winners'][svc]['pair']
                if not w['reader_id'].startswith('PAIR__'):
                    continue
                name = w['reader_id'].split('__')[-1]; kind = dict(readers.SLATE)[name]
                a, b = readers.kinds(bank[f'{svc}__DEV_REPORT__income'], 2), readers.kinds(bank[f'{svc}__DEV_REPORT__occupation'], 5)
                x = np.column_stack((a[kind], b[kind]))
                m = joblib.load(rd / f"MODEL__{w['reader_id']}.joblib")
                sign = rec['services'][svc]['pair_readers'][name]['orientation']
                sc = predict_reader(m, x); sc = sc if sign == 1 else 1 - sc
                checks[f'{svc}|pair|{name}'] = bool(np.array_equal(sc, stored[w['reader_id']]))
            out[f'S{s}_reader_rescoring_from_backup'] = checks
    out['pass'] = (out['hash_mismatches'] == 0 and all(v['max_abs_prob_diff'] <= 1e-12 and v['decisions_identical']
                                                       for s in C.SEEDS for v in out[f'S{s}_package_parity'].values())
                   and all(all(out.get(f'S{s}_reader_rescoring_from_backup', {'none': True}).values()) for s in C.SEEDS))
    return out


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    a1 = sub.add_parser('snapshot'); a1.add_argument('--name', required=True)
    a2 = sub.add_parser('restore-check'); a2.add_argument('--snapshot', required=True); a2.add_argument('--restore', required=True)
    a = ap.parse_args()
    print(json.dumps(snapshot(a.name) if a.cmd == 'snapshot' else restore_check(a.snapshot, a.restore), indent=1))


if __name__ == '__main__':
    main()
