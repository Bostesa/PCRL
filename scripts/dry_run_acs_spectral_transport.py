"""Exercise the sealed scoring and report code before the lock.

A 2017 *validation* partition stands in for the final partition, outputs go to a
separate directory, and nothing is fitted. Resulting numbers are code-test
output on data already used for selection, never transport results.
"""
from __future__ import annotations
import argparse, shutil
from pathlib import Path
from experiments import acs_spectral_transport_eval as ev
from experiments import run_acs_spectral_transport as run


def main():
    p = argparse.ArgumentParser(); p.add_argument('--dest', type=Path, required=True)
    p.add_argument('--partition', default='task_validation'); p.add_argument('--workers', type=int, default=8)
    a = p.parse_args(); dest = a.dest.resolve()
    assert dest != ev.OUT.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(ev.OUT/'COMPARISONS.json', dest/'COMPARISONS.json')
    (dest/'TRANSPORT_LOCK.json').write_text('{"dry_run": true}\n')
    for s in (0, 1, 2):
        for c in (*ev.INTERFACES, 'reference'):
            src = ev.OUT/f'seed_{s}'/c
            for name in ('audit_selection.json', 'utility_selection.json', 'selection.json'):
                if (src/name).exists():
                    (dest/f'seed_{s}'/c).mkdir(parents=True, exist_ok=True); shutil.copy(src/name, dest/f'seed_{s}'/c/name)
        shutil.copytree(ev.OUT/f'seed_{s}'/'releases_2017', dest/f'seed_{s}'/'releases_2017', dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('final_evaluation.npz'))
    import os; os.environ['PCRL_TRANSPORT_DRY_RUN'] = a.partition; ev.DRY_RUN_PARTITION = a.partition
    run._init(); run.phase_score(dest, (0, 1, 2), a.workers)
    from scripts import report_acs_spectral_transport as rep
    d = rep.report(dest)
    print('DRY RUN (validation partition, not results):', d['score_replay'])


if __name__ == '__main__':
    main()
