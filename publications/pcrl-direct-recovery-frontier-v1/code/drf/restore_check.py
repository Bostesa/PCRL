"""Restore check, run INSIDE a restored copy (backup alone; no label is read):
- the admitted source mirror copies re-hash against the source SOURCE_ADMISSION.json;
- every saved release (runs/<FAMILY>/<config>/V<v>/RELEASE.json) re-hashes its RELEASE.npz;
- representative fitted models re-run from features: each family's predict() reproduces its saved MODEL_SELECT release bitwise
  (the packaged releases, or every release when nothing is packaged yet; draw 0 for randomized releases);
- representative readers (runs/READER_CUSTODY/*.pkl, fitted reader models saved with their SCORE outputs) re-load from the backup and
  reproduce their saved scores bitwise on the restored released wires;
- reader-grid results (GRID.json / BOOT.npz) are present and parse; the reader cache entries pass their own checksums;
- every deployment package (under the campaign's private deploy/ folder) redeploys the restored DEV_REPORT raw inputs and reproduces
  its stored wire bitwise, with every refusal (drf.deploy check). Derived from nrm.restore_check (relative imports resolve to drf).
Adapted for this campaign (2026-10-24): the cached task teachers (nrm fa_cache TASK.pkl) and SUCC's frozen files re-hash against
SOURCE_ADMISSION.json (drf.fa.tfr.preflight); the external-input snapshot (FB's fit-time STAGE_A / NEURAL_RAW files) against the
admission pins; every FA / FB / FS / CTL MODEL.pkl against its MODEL.json; every bank member of every reader grid re-loads from the
restored files and reproduces the wire hashes the grid recorded (references included); and the check fails if a reader grid exists but
no custody reader was restored."""
from __future__ import annotations
import json
import pickle
import sys
from pathlib import Path

import numpy as np

from . import common as C
from . import release as RL


def main(out):
    from ohn import common as OC
    adm = OC.admission()['copies']
    bad_adm = [rel for rel, r in adm.items() if C.sha(OC.MIRROR / rel) != r['sha256']]
    rel_ok, rel_bad, reran = 0, [], {}
    pkgs = sorted(str(p.parent) for p in (C.PRIV / 'deploy').glob('*/PACKAGE.json')) if (C.PRIV / 'deploy').is_dir() else []
    packaged = set()
    for p in pkgs:
        meta = json.loads((Path(p) / 'PACKAGE.json').read_text())
        packaged.add((meta['name'], int(meta['version'])))
    S = C.features('MODEL_SELECT')
    import sys as _sys
    here = _sys.version.split()[0]
    skipped = {}
    for rj in sorted(C.RUNS.glob('*/*/V*/RELEASE.json')):
        rec = json.loads(rj.read_text())
        if C.sha(rj.parent / 'RELEASE.npz') != rec['release_sha256']:
            rel_bad.append(str(rj.parent.relative_to(C.RUNS))); continue
        rel_ok += 1
        name, v = f"{rec['family']}/{rec['config_id']}", int(rec['version'])
        if packaged and (name, v) not in packaged:
            continue
        if rec.get('python') and rec['python'] != here:      # ENG1-04: re-run under the family's own fit interpreter only
            skipped[f'{name}/V{v}'] = f"fitted under python {rec['python']}; this pass runs {here} (checked in the other pass)"
            continue
        from .deploy import from_features, _keys_for
        try:
            saved, _ = RL.load(rec['family'], rec['config_id'], v, roles=('MODEL_SELECT',))
            new = from_features(name, v, S['X'], S['feature_names'], draw=0, keys=_keys_for(rec, 'MODEL_SELECT', S['row_id']))
            reran[f'{name}/V{v}'] = all(bool(np.array_equal(saved['MODEL_SELECT'][t][0] if saved['MODEL_SELECT'][t].ndim == 3
                                                             else saved['MODEL_SELECT'][t], new[t])) for t in C.TASKS)
        except Exception as ex:                                # recorded per release, never an abort (ENG1-04)
            reran[f'{name}/V{v}'] = False
            skipped[f'{name}/V{v}__error'] = f'{type(ex).__name__}: {str(ex)[:200]}'
    readers = {}
    readers_other = {}
    for pk in sorted((C.RUNS / 'READER_CUSTODY').glob('*.pkl')) if (C.RUNS / 'READER_CUSTODY').is_dir() else []:
        fit_py = None
        try:                                                   # recorded per reader, never an abort (as for releases)
            with open(pk, 'rb') as f:
                obj = pickle.load(f)
            fit_py = obj.get('python')
            X = obj['design_fn'](obj['wire']) if callable(obj.get('design_fn')) else obj['X_score']
            ok, err = bool(np.array_equal(obj['predict'](obj['model'], X), obj['score_raw'])), None
        except Exception as ex:
            ok, err = False, f'{type(ex).__name__}: {str(ex)[:200]}'
        if not ok and fit_py and fit_py != here:               # ENG2-06: judged in its fit interpreter's pass, not failed here
            readers_other[pk.name] = f'fitted under python {fit_py}; this pass runs {here}; not bitwise here' + (f' ({err})' if err else '')
            continue
        readers[pk.name] = ok
        if err:
            skipped[f'{pk.name}__error'] = err
    grid = {}
    for g in sorted((C.RUNS / 'GRID').glob('*/S*/GRID.json')) if (C.RUNS / 'GRID').is_dir() else []:
        grid[str(g.relative_to(C.RUNS))] = {'json_parses': bool(json.loads(g.read_text()).get('paired') is not None),
                                            'boot_present': (g.parent / 'BOOT.npz').is_file()}
    cache_ok, cache_bad = 0, []
    if (C.RUNS / 'devgrid_cache').is_dir():
        from .devreaders import _read_entry, RECIPES
        for e in sorted((C.RUNS / 'devgrid_cache').glob('*.npz')):
            try:
                with np.load(e, allow_pickle=False) as z:
                    n = len(z[f'score__{RECIPES[0]}'])
                _read_entry(e, n); cache_ok += 1
            except Exception as ex:                                   # recorded, never hidden
                cache_bad.append(f'{e.name}: {type(ex).__name__}')
    mine = []                                                  # packages whose release was fitted under this interpreter
    for p in pkgs:
        meta = json.loads((Path(p) / 'PACKAGE.json').read_text())
        fam, cfg = meta['name'].split('/')
        if RL.load(fam, cfg, int(meta['version']), roles=('DEV_REPORT',))[1].get('python') == here:
            mine.append(p)
    if mine:
        from . import deploy as DP
        dep = DP.check(mine, Path(out).with_name(f'restore_deploy_detail_py{here}.json'))
    else:
        dep = {'all_bitwise': None, 'all_refused': None}
    # source artefacts this campaign reads: TASK.pkl and SUCC's frozen files (FA's preflight), FB's fit-time external inputs, and the
    # wire of every bank member the reader grids read (references included), against the hashes the grids recorded
    src_bad, src_n = [], 0
    from .fa import tfr as TF
    for v in C.VERSIONS:
        pf = TF.preflight(v)
        src_n += 1 + len(pf['frozen'])
        src_bad += [f'{x} V{v}' for x in pf['missing'] + pf['mismatched']]
    adm2 = json.loads((C.RESULTS / 'SOURCE_ADMISSION.json').read_text())
    ext = C.PRIV / 'external_inputs' / 'acs_replication_runs'
    for v, rec_v in adm2['acs_replication_inputs']['files'].items():
        sv = v[1:]
        u0 = ext / f'STAGE_A_{sv}' / 'U0_MODEL.pt'
        src_n += 1
        if not u0.is_file() or C.sha(u0) != rec_v.get('U0_MODEL.pt'):
            src_bad.append(f'external U0_MODEL.pt {v}')
        for ck, r in (rec_v.get('NEURAL_RAW_checkpoints') or {}).items():
            rho, e = ck.split('/')
            f = ext / f'NEURAL_RAW_{sv}' / f'NEURAL_RAW__{rho}__{e}.pt'
            src_n += 1
            if not f.is_file() or C.sha(f) != r['sha256']:
                src_bad.append(f'external NEURAL_RAW {v} {ck}')
    from . import devreaders as DR
    members_ok, members_bad = 0, []
    for g in sorted((C.RUNS / 'GRID').glob('*/S*/GRID.json')) if (C.RUNS / 'GRID').is_dir() else []:
        G = json.loads(g.read_text())
        spec = json.loads((C.RESULTS / 'grid_specs' / f"{G['bank_id']}.json").read_text())
        for m, prov in G['members'].items():
            try:
                _, p2 = DR.load_member(spec['members'][m], int(G['seed']), row_id=S['row_id'])
                ok = p2['wire_sha256'] == prov['wire_sha256']
            except Exception as ex:                                     # recorded, never an abort
                ok = False
                skipped[f"{G['bank_id']}/S{G['seed']}/{m}__error"] = f'{type(ex).__name__}: {str(ex)[:160]}'
            if ok:
                members_ok += 1
            else:
                members_bad.append(f"{G['bank_id']}/S{G['seed']}/{m}")
    cache_bad2 = []
    model_bad, model_n = [], 0
    for mj in [m for fam in ('FA', 'FB', 'FS', 'CTL') for m in sorted(C.RUNS.glob(f'{fam}/*/V*/MODEL.json'))]:
        model_n += 1
        if C.sha(mj.parent / 'MODEL.pkl') != json.loads(mj.read_text())['model_sha256']:
            model_bad.append(str(mj.parent.relative_to(C.RUNS)))
    rec = {'admitted_copies_rehash_ok': not bad_adm, 'admitted_copies_checked': len(adm), 'mismatched': bad_adm[:10],
           'source_artefacts_checked': src_n, 'source_artefacts_mismatched': src_bad,
           'bank_member_wires_reloaded_ok': members_ok, 'bank_member_wires_mismatched': members_bad,
           'private_cache_problems': cache_bad2, 'family_models_rehashed': model_n, 'family_models_mismatched': model_bad,
           'python': here, 'models_not_rerun_in_this_pass': skipped,
           'releases_rehashed_ok': rel_ok, 'releases_mismatched': rel_bad[:20], 'models_rerun_bitwise': reran,
           'representative_readers_bitwise': readers, 'readers_judged_in_other_pass': readers_other, 'reader_grids': grid,
           'reader_cache_entries_ok': cache_ok,
           'reader_cache_entries_bad': cache_bad[:20], 'deployment_packages': len(pkgs),
           'deployment_from_backup_bitwise': dep['all_bitwise'], 'deployment_from_backup_all_refused': dep.get('all_refused'),
           'labels_read': False}
    rec['custody_readers_required'] = bool(grid)
    rec['all_ok'] = (rec['admitted_copies_rehash_ok'] and not rel_bad and all(reran.values()) and all(readers.values())
                     and not src_bad and not cache_bad2 and not model_bad and not members_bad and members_ok > 0 and (bool(readers) or bool(readers_other) or not grid)
                     and all(v['json_parses'] and v['boot_present'] for v in grid.values()) and not cache_bad
                     and dep['all_bitwise'] is not False and dep.get('all_refused') is not False)
    Path(out).write_text(json.dumps(rec, indent=1) + '\n')
    print(json.dumps({k: v for k, v in rec.items() if k not in ('reader_grids', 'models_rerun_bitwise')}))


if __name__ == '__main__':
    main(sys.argv[1])
