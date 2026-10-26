"""Family FAM, revision R1 (card results/pcrl_reader_fit_aware_release_v1/cards/RF_RFIT_R1.md; adaptive: designed after the v1
MODEL_SELECT reading RFA_v1). EXPLORATORY DEVELOPMENT; not confirmation.

The fit stage of `rfa.fam.fit` unchanged, with the revision's objective module `rfa.fam.rfit_r1` (the inner reader family set adds the
table-like family; everything else is v1's: START, episodes, sizes, surrogate, lambda rule, path, budgets, brackets), the revision's
configuration ids (RF1_*), its own fit record, and an alias hook: a CONT (TASK) or ALT (RAW-J) point whose weights are bitwise those
of an already saved v1 configuration could be reused (`alias_check_for`, both arms do not depend on the objective); it is NOT used:
the versions run in parallel and a per-version alias could leave a member id missing in one version (bitwise identities with the v1
configurations are recorded by the plan instead).

    <repository>/.venv/bin/python -m rfa.fam.fit_r1 fit --version V --card results/pcrl_reader_fit_aware_release_v1/cards/RF_RFIT_R1.md
"""
from __future__ import annotations

import argparse
import json
import sys

from .. import common as C
from .. import famkit as FK
from . import fit as F
from . import net as NET

REVISION = 'R1'
IDS = {'RF_FULL': 'RF1_FULL', 'RF_TASK': 'RF1_TASK', 'RF_STOP_LO': 'RF1_STOP_LO', 'RF_STOP_HI': 'RF1_STOP_HI',
       'RF_ALTC_HI': 'RF1_ALTC_HI', 'RF_RAWJ_HI': 'RF1_RAWJ_HI'}
CONFIGS_R1 = {k: {**v, 'id': IDS[k], 'what': 'R1 (table-like inner reader family added): ' + v['what']} for k, v in F.CONFIGS.items()}
V1_IDS = {'TASK': 'RF_TASK', 'RAWJ': 'RF_RAWJ_HI'}
PURPOSE = ('RF revision R1: train the occupation network with the reader-fit-aware episode objective with the table-like inner family '
           '(FULL), its stopped-fit and alternating-critic controls, its zero-protection continuation and the RAW-J ALT path on MODEL_FIT '
           '(SEX as a training label of the protection objectives and of the within-SEX guard components of the strength rule; the '
           'TASK arm never reads SEX)')


def alias_check_for(version):
    """CONT and ALT do not depend on the objective: reuse the saved v1 configuration when its checkpoint and weights are identical."""
    def check(base_cfg, arm, step, weights):
        cfg = V1_IDS.get(arm)
        if cfg is None:
            return None
        try:
            P = FK.load_model(NET.FAMILY, cfg, version)
        except FileNotFoundError:
            return None
        if P.get('checkpoint') != f'S{int(step)}':
            return None
        same = all(len(P['weights'][t]) == len(weights[t]) and
                   all((a[0] == b[0]).all() and (a[1] == b[1]).all() for a, b in zip(P['weights'][t], weights[t])) for t in NET.TASKS)
        return cfg if same else None
    return check


def main(argv=None):
    ap = argparse.ArgumentParser(prog='rfa.fam.fit_r1')
    sub = ap.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('fit')
    f.add_argument('--version', type=int, required=True)
    f.add_argument('--card', required=True)
    a = ap.parse_args(argv)
    from .. import release as RL
    from . import rfit_r1 as R1
    data = C.load('MODEL_FIT', ('features', 'income', 'occupation', 'sex'), 'fit', PURPOSE, card=a.card)
    out = F.stage_fit(a.version, data, a.card, RL.save, recipe_rf=R1.RECIPE_RF_R1, rf_module=R1, configs=CONFIGS_R1,
                      record_name=f'FIT_RF_R1_V{int(a.version)}.json', revision=REVISION)
    # no alias hook: the two versions run in parallel, so a per-version alias could leave a member id missing in one version;
    # R1 saves at most 6 configurations per version, which the remaining 12 of the campaign's 24 cover
    print(json.dumps({'k_star': (out.get('selection') or {}).get('k_star'), 'status': (out.get('cand') or {}).get('status'),
                      'gate': (out.get('mechanism_gate') or {}).get('label'),
                      'cand_spend_over_start': (out.get('cand') or {}).get('spend_over_start'),
                      'brackets': {k: {'lo': v['lo'], 'hi': v['hi'], 'flags': v['flags']} for k, v in (out.get('paths') or {}).items()},
                      'saved': sorted(out.get('saved') or {}), 'aliases': out.get('aliases'), 'checks': out.get('checks'),
                      'cpu_s': round(out['cpu_s'], 1)}, default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
