"""Restore check, run INSIDE a restored copy (backup alone; no label is read):
- the admitted source mirror copies re-hash against the source SOURCE_ADMISSION.json;
- every saved release (runs/<FAMILY>/<config>/V<v>/RELEASE.json) re-hashes its RELEASE.npz;
- representative fitted models re-run from features: each family's predict() reproduces its saved MODEL_SELECT release bitwise
  (the packaged releases, or every release when nothing is packaged yet; draw 0 for randomized releases);
- representative readers (runs/READER_CUSTODY/*.pkl, fitted reader models saved with their SCORE outputs) re-load from the backup and
  reproduce their saved scores bitwise on the restored released wires;
- reader-grid results (GRID.json / BOOT.npz) are present and parse; the reader cache entries pass their own checksums;
- every deployment package (under the campaign's private deploy/ folder) redeploys the restored DEV_REPORT raw inputs and reproduces
  its stored wire bitwise, with every refusal (frc.deploy check). Derived from nrm.restore_check (relative imports resolve to frc).
Added for this campaign (engineering review finding 12): the read-only source artefacts re-hash against SOURCE_ADMISSION.json; the
private fa_cache (SEXSIDE.pkl against its sidecar and the selection records, the nested per-person files against the NEST records);
every FA MODEL.pkl against its MODEL.json; and the check fails if a reader grid exists but no custody reader was restored."""
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
    # source artefacts, private caches and every model file (engineering review finding 12)
    src_bad, src_n = [], 0
    adm2 = json.loads((C.RESULTS / 'SOURCE_ADMISSION.json').read_text())['source_private_artefacts']
    for key, pin in adm2.items():
        if key.startswith('fa_cache/'):
            f = C.NRM_FA_CACHE / key[len('fa_cache/'):]
            pairs = [(f, pin['sha256'])]
        elif key.startswith('runs/FA/'):
            d = C.NRM_RUNS / key[len('runs/'):]
            pairs = [(d / 'MODEL.pkl', pin['MODEL.pkl']), (d / 'RELEASE.npz', pin['RELEASE.npz'])]
        elif key.startswith('results/designer_a/'):
            pairs = [(C.NRM_RESULTS / 'roles' / 'designer_a' / key.split('/')[-1], pin['sha256'])]
        else:
            continue
        for f, want in pairs:
            src_n += 1
            if not f.is_file() or C.sha(f) != want:
                src_bad.append(key)
    cache_bad2 = []
    for v in C.VERSIONS:
        sx = C.PRIV / 'fa_cache' / f'V{v}' / 'SEXSIDE.pkl'
        if sx.is_file():
            if C.sha(sx) != (sx.with_suffix('.sha256')).read_text().strip():
                cache_bad2.append(f'SEXSIDE V{v} sidecar')
            selp = C.RESULTS / 'fa' / f'SELECTION_V{v}.json'
            if selp.is_file() and json.loads(selp.read_text())['sexside_cache_sha256'] != C.sha(sx):
                cache_bad2.append(f'SEXSIDE V{v} selection record')
    for nj in sorted((C.RESULTS / 'fa').glob('NEST_*.json')) if (C.RESULTS / 'fa').is_dir() else []:
        npz = C.PRIV / 'fa_cache' / 'nested' / (nj.stem + '.npz')
        if not npz.is_file() or C.sha(npz) != json.loads(nj.read_text())['per_person_private_file_sha256']:
            cache_bad2.append(nj.name)
    model_bad, model_n = [], 0
    for mj in sorted(C.RUNS.glob('FA/*/V*/MODEL.json')):
        model_n += 1
        if C.sha(mj.parent / 'MODEL.pkl') != json.loads(mj.read_text())['model_sha256']:
            model_bad.append(str(mj.parent.relative_to(C.RUNS)))
    rec = {'admitted_copies_rehash_ok': not bad_adm, 'admitted_copies_checked': len(adm), 'mismatched': bad_adm[:10],
           'source_artefacts_checked': src_n, 'source_artefacts_mismatched': src_bad,
           'private_cache_problems': cache_bad2, 'fa_models_rehashed': model_n, 'fa_models_mismatched': model_bad,
           'python': here, 'models_not_rerun_in_this_pass': skipped,
           'releases_rehashed_ok': rel_ok, 'releases_mismatched': rel_bad[:20], 'models_rerun_bitwise': reran,
           'representative_readers_bitwise': readers, 'readers_judged_in_other_pass': readers_other, 'reader_grids': grid,
           'reader_cache_entries_ok': cache_ok,
           'reader_cache_entries_bad': cache_bad[:20], 'deployment_packages': len(pkgs),
           'deployment_from_backup_bitwise': dep['all_bitwise'], 'deployment_from_backup_all_refused': dep.get('all_refused'),
           'labels_read': False}
    rec['custody_readers_required'] = bool(grid)
    rec['all_ok'] = (rec['admitted_copies_rehash_ok'] and not rel_bad and all(reran.values()) and all(readers.values())
                     and not src_bad and not cache_bad2 and not model_bad and (bool(readers) or bool(readers_other) or not grid)
                     and all(v['json_parses'] and v['boot_present'] for v in grid.values()) and not cache_bad
                     and dep['all_bitwise'] is not False and dep.get('all_refused') is not False)
    Path(out).write_text(json.dumps(rec, indent=1) + '\n')
    print(json.dumps({k: v for k, v in rec.items() if k not in ('reader_grids', 'models_rerun_bitwise')}))


if __name__ == '__main__':
    main(sys.argv[1])
