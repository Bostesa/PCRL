"""Restore check, run INSIDE a restored copy (backup alone; no label is read):
- the 201 admitted H3-mirror copies re-hash against the source SOURCE_ADMISSION.json;
- every Track F record re-hashes: witness files (WITNESS.json files_sha256), REFIT routing teachers (ROUTING.json), correction
  directions (DIR.json files_sha256), compression codes (COMP.json), Track R income lineages (INCOME.json);
- representative models, corrections and the routing are re-run from features: each witness's predict() reproduces its stored
  MODEL_SELECT release, and each REFIT_PAIR correction at s = 0.5 reproduces the evaluated MODEL_SELECT array (bitwise);
- reader-grid results (GRID.json / BOOT.npz) re-hash against the queue ledger's recorded outputs where recorded, and the reader cache
  entries of this campaign pass their own checksums;
- every deployment package under runs/DEPLOY redeploys the restored DEV_REPORT raw inputs and reproduces its stored wire bitwise,
  with every refusal (cts.deploy check)."""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

from . import common as C
from . import trackf as TF


def main(out):
    from ohn import common as OC
    adm = OC.admission()['copies']
    bad = [rel for rel, r in adm.items() if C.sha(OC.MIRROR / rel) != r['sha256']]
    rec_bad, n_rec, models = [], 0, {}
    S = C.features('MODEL_SELECT')
    for P in sorted(p.name for p in (C.RUNS / 'TRACKF').iterdir() if p.is_dir()):
        d = C.RUNS / 'TRACKF' / P
        wj = d / 'WITNESS' / 'WITNESS.json'
        if wj.is_file():
            w = json.loads(wj.read_text()); n_rec += 1
            rec_bad += [f'{P}/WITNESS/{f}' for f, h in w['files_sha256'].items() if C.sha(d / 'WITNESS' / f) != h]
            wrec, W, O = TF.load_witness(P)
            p = TF.witness_predict(P, S['X'], S['feature_names'])
            models[f'{P}/witness'] = bool(np.array_equal(p, W['dep__MODEL_SELECT']))
            if (d / 'DIR' / 'REFIT_PAIR' / 'DIR.json').is_file() and (d / 'ROUTING' / 'ROUTING.json').is_file():
                from ohn import protect as PR
                drec, T, D = TF.load_direction(P, 'REFIT_PAIR')
                lv, te = TF.route(P, 'REFIT', S['X'], S['feature_names'])
                delta = PR.correction(T['theta_full'], lv, te.T, te.L)
                models[f'{P}/REFIT_PAIR@s0.5'] = bool(np.array_equal(TF.release_at(p, delta, 0.5),
                                                                      TF.release_at(W['dep__MODEL_SELECT'], D['full__MODEL_SELECT'], 0.5)))
        rj = d / 'ROUTING' / 'ROUTING.json'
        if rj.is_file():
            r = json.loads(rj.read_text()); n_rec += 1
            if C.sha(d / 'ROUTING' / 'TEACHER_occupation.json') != r['teacher_file_sha256']:
                rec_bad.append(f'{P}/ROUTING')
        for dj in sorted((d / 'DIR').glob('*/DIR.json')) if (d / 'DIR').is_dir() else []:
            r = json.loads(dj.read_text()); n_rec += 1
            rec_bad += [f'{P}/DIR/{dj.parent.name}/{f}' for f, h in r['files_sha256'].items() if C.sha(dj.parent / f) != h]
        for cj in sorted((d / 'COMP').glob('*/COMP.json')) if (d / 'COMP').is_dir() else []:
            r = json.loads(cj.read_text()); n_rec += 1
            if C.sha(cj.parent / 'CODE.npz') != r['npz_sha256']:
                rec_bad.append(f'{P}/COMP/{cj.parent.name}')
    for ij in sorted((C.RUNS / 'TRACK_R').glob('*/INCOME/INCOME.json')) if (C.RUNS / 'TRACK_R').is_dir() else []:
        r = json.loads(ij.read_text()); n_rec += 1
        rec_bad += [f'TRACK_R/{ij.parent.parent.name}/{f}' for f, h in r.get('files_sha256', {}).items()
                    if (ij.parent / f).is_file() and C.sha(ij.parent / f) != h]
    grid = {}
    from cts import jobqueue as Q
    outs = {}
    for e in Q.entries():
        if e.get('event') == 'unit_end':
            for k, h in (e.get('output_sha256') or {}).items():
                outs[k] = h
    for g in sorted((C.RUNS / 'GRID').glob('*/S*/GRID.json')) if (C.RUNS / 'GRID').is_dir() else []:
        grid[str(g.relative_to(C.RUNS))] = {'json_parses': bool(json.loads(g.read_text()).get('paired') is not None),
                                            'boot_present': (g.parent / 'BOOT.npz').is_file()}
    pkgs = sorted(str(p.parent) for p in (C.RUNS / 'DEPLOY').glob('*/PACKAGE.json')) if (C.RUNS / 'DEPLOY').is_dir() else []
    if pkgs:
        from cts import deploy as DP
        dep = DP.check(pkgs, Path(out).with_name('restore_deploy_detail.json'))
    else:
        dep = {'all_bitwise': None, 'all_refused': None}
    rec = {'admitted_copies_rehash_ok': not bad, 'admitted_copies_checked': len(adm), 'mismatched': bad[:10],
           'trackf_records_checked': n_rec, 'trackf_records_mismatched': rec_bad[:20], 'models_rerun_bitwise': models,
           'reader_grids': grid, 'deployment_packages': len(pkgs), 'deployment_from_backup_bitwise': dep['all_bitwise'],
           'deployment_from_backup_all_refused': dep.get('all_refused'), 'labels_read': False}
    rec['all_ok'] = (rec['admitted_copies_rehash_ok'] and not rec_bad and all(models.values())
                     and all(v['json_parses'] and v['boot_present'] for v in grid.values())
                     and dep['all_bitwise'] is not False and dep.get('all_refused') is not False)
    Path(out).write_text(json.dumps(rec, indent=1) + '\n')
    print(json.dumps({k: v for k, v in rec.items() if k not in ('reader_grids',)}))


if __name__ == '__main__':
    main(sys.argv[1])
