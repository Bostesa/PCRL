"""Restore check, run INSIDE a restored copy (backup alone; no label is read):
- the admitted source mirror copies re-hash against the source SOURCE_ADMISSION.json;
- every saved release (runs/<FAMILY>/<config>/V<v>/RELEASE.json) re-hashes its RELEASE.npz;
- representative fitted models re-run from features: each family's predict() reproduces its saved MODEL_SELECT release bitwise
  (the packaged releases, or every release when nothing is packaged yet; draw 0 for randomized releases);
- representative readers (runs/READER_CUSTODY/*.pkl, fitted reader models saved with their SCORE outputs) re-load from the backup and
  reproduce their saved scores bitwise on the restored released wires;
- reader-grid results (GRID.json / BOOT.npz) are present and parse; the reader cache entries pass their own checksums;
- every deployment package (under the campaign's private deploy/ folder) redeploys the restored DEV_REPORT raw inputs and reproduces
  its stored wire bitwise, with every refusal (nrm.deploy check)."""
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
    for rj in sorted(C.RUNS.glob('*/*/V*/RELEASE.json')):
        rec = json.loads(rj.read_text())
        if C.sha(rj.parent / 'RELEASE.npz') != rec['release_sha256']:
            rel_bad.append(str(rj.parent.relative_to(C.RUNS))); continue
        rel_ok += 1
        name, v = f"{rec['family']}/{rec['config_id']}", int(rec['version'])
        if packaged and (name, v) not in packaged:
            continue
        from .deploy import from_features
        saved, _ = RL.load(rec['family'], rec['config_id'], v, roles=('MODEL_SELECT',))
        new = from_features(name, v, S['X'], S['feature_names'], draw=0)
        reran[f'{name}/V{v}'] = all(bool(np.array_equal(saved['MODEL_SELECT'][t][0] if saved['MODEL_SELECT'][t].ndim == 3
                                                         else saved['MODEL_SELECT'][t], new[t])) for t in C.TASKS)
    readers = {}
    for pk in sorted((C.RUNS / 'READER_CUSTODY').glob('*.pkl')) if (C.RUNS / 'READER_CUSTODY').is_dir() else []:
        with open(pk, 'rb') as f:
            obj = pickle.load(f)
        X = obj['design_fn'](obj['wire']) if callable(obj.get('design_fn')) else obj['X_score']
        readers[pk.name] = bool(np.array_equal(obj['predict'](obj['model'], X), obj['score_raw']))
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
    if pkgs:
        from . import deploy as DP
        dep = DP.check(pkgs, Path(out).with_name('restore_deploy_detail.json'))
    else:
        dep = {'all_bitwise': None, 'all_refused': None}
    rec = {'admitted_copies_rehash_ok': not bad_adm, 'admitted_copies_checked': len(adm), 'mismatched': bad_adm[:10],
           'releases_rehashed_ok': rel_ok, 'releases_mismatched': rel_bad[:20], 'models_rerun_bitwise': reran,
           'representative_readers_bitwise': readers, 'reader_grids': grid, 'reader_cache_entries_ok': cache_ok,
           'reader_cache_entries_bad': cache_bad[:20], 'deployment_packages': len(pkgs),
           'deployment_from_backup_bitwise': dep['all_bitwise'], 'deployment_from_backup_all_refused': dep.get('all_refused'),
           'labels_read': False}
    rec['all_ok'] = (rec['admitted_copies_rehash_ok'] and not rel_bad and all(reran.values()) and all(readers.values())
                     and all(v['json_parses'] and v['boot_present'] for v in grid.values()) and not cache_bad
                     and dep['all_bitwise'] is not False and dep.get('all_refused') is not False)
    Path(out).write_text(json.dumps(rec, indent=1) + '\n')
    print(json.dumps({k: v for k, v in rec.items() if k not in ('reader_grids', 'models_rerun_bitwise')}))


if __name__ == '__main__':
    main(sys.argv[1])
