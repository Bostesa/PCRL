"""After the 2018 audit queue: panel -> commit -> stronger attack -> 2017 transport.

Single worker, in the registered order. The panel is committed and pushed BEFORE any
panel test-split table is produced. The 2017 stage stops *launching* units at
`DEADLINE_2017` so the final hour stays free for analysis; unlaunched units are recorded
as pending with their resume command. Candidates are transported before controls.
"""
from __future__ import annotations

import subprocess
from datetime import datetime, timezone

from .common import OUT, ROOT, OrchestratorLock, limit_threads, read_json, utcnow, write_json_atomic

# RUN_STATUS amendment 3: the user directed that the schedule cutoffs not stop the run.
DEADLINE_2017 = None
SEEDS = (0, 1, 2)
COMPARATORS = ('ref_A0', 'ref_J', 'leace_A0', 'splince_A0', 'optnet16_C1')


def git(*args):
    return subprocess.run(['git', '-C', str(ROOT), '-c', 'user.name=Bostesa',
                           '-c', 'user.email=natestesa@gmail.com', *args],
                          capture_output=True, text=True)


def main():
    limit_threads()
    with OrchestratorLock('orchestrator').acquire('post_2018'):
        from . import aggregate, panel
        aggregate.track_n()
        aggregate.track_e()
        aggregate.counts()
        result = panel.run()
        git('add', 'results/pcrl_competitive_method_v1/PANEL.json',
            'results/pcrl_competitive_method_v1/TRACK_N_FITS.json',
            'results/pcrl_competitive_method_v1/TRACK_E_FITS.json',
            'results/pcrl_competitive_method_v1/COUNTS.json')
        commit = git('commit', '-m', 'Freeze the candidate panel from validation-split evidence\n\n'
                     'Nominated by the PROTOCOL section 4 lexicographic rule before any panel '
                     'test-split table was produced. Adds fit ledgers and counts.')
        git('push', '-q')
        head = git('rev-parse', 'HEAD').stdout.strip()
        write_json_atomic(OUT / 'logs' / 'panel_commit.json',
                          {'utc': utcnow(), 'head': head, 'commit_stdout': commit.stdout[-500:]})
        print('PANEL_FROZEN', head, [p['unit'] for p in result['panel']], flush=True)

        candidates = [p['unit'] for p in result['panel']]
        controls = [c for p in result['panel'] for c in p['controls']]
        stressed = list(dict.fromkeys(['H'] + candidates + controls + list(COMPARATORS)))
        from . import stress
        cache = {}
        for seed in SEEDS:
            for condition in stressed:
                try:
                    stress.run_condition(seed, condition, cache)
                except Exception as exc:                      # scoped stop, continue others
                    print('STRESS_FAILED', seed, condition, repr(exc)[:200], flush=True)
        print('STRESS_COMPLETE', flush=True)

        from . import transport_2017
        for tag, units in (('candidates', candidates), ('controls', controls)):
            for seed in SEEDS:
                try:
                    transport_2017.run(seed, units, DEADLINE_2017, tag)
                except Exception as exc:
                    print('X2017_FAILED', seed, tag, repr(exc)[:300], flush=True)
        print('POST_2018_DONE', utcnow(), flush=True)


if __name__ == '__main__':
    main()
