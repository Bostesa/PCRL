"""Restore check, run INSIDE a restored copy (backup alone; no label is read):
- the admitted H3-mirror copies re-hash against the source SOURCE_ADMISSION.json;
- every saved record under runs/ that lists `files_sha256` re-hashes (size fits, controls, correction bases and directions, nested
  estimates, selections, stored wires);
- every A1_PO_16 size fit is re-run from features: predict() reproduces its saved MODEL_SELECT release bitwise;
- reader-grid results (GRID.json / BOOT.npz) are present and parse;
- every deployment package (under the campaign's private deploy/ folder) redeploys the restored DEV_REPORT raw inputs and reproduces
  its stored wire bitwise, with every refusal (tsz.deploy check)."""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

from . import common as C


def _rehash(rec_path):
    rec = json.loads(rec_path.read_text())
    files = rec.get('files_sha256') or {}
    bad = [f for f, h in files.items() if not (rec_path.parent / f).is_file() or C.sha(rec_path.parent / f) != h]
    return len(files), bad


def main(out):
    from ohn import common as OC
    adm = OC.admission()['copies']
    bad_adm = [rel for rel, r in adm.items() if C.sha(OC.MIRROR / rel) != r['sha256']]
    n_rec, n_files, rec_bad = 0, 0, []
    for p in sorted(C.RUNS.rglob('*.json')):
        if 'devgrid_cache' in p.parts:
            continue
        try:
            nf, bad = _rehash(p)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if nf:
            n_rec += 1; n_files += nf
            rec_bad += [f'{p.parent.relative_to(C.RUNS)}/{b}' for b in bad]
    models = {}
    S = C.features('MODEL_SELECT')
    from . import sizefit
    for d in sorted((C.RUNS / 'SIZE').iterdir()) if (C.RUNS / 'SIZE').is_dir() else []:
        if (d / 'FIT.json').is_file():
            rel, _ = sizefit.release(d.name)
            models[d.name] = bool(np.array_equal(sizefit.predict(d.name, S['X'], S['feature_names']), rel['MODEL_SELECT']))
    grid = {}
    for g in sorted((C.RUNS / 'GRID').glob('*/S*/GRID.json')) if (C.RUNS / 'GRID').is_dir() else []:
        grid[str(g.relative_to(C.RUNS))] = {'json_parses': bool(json.loads(g.read_text()).get('paired') is not None),
                                            'boot_present': (g.parent / 'BOOT.npz').is_file()}
    pkgs = sorted(str(p.parent) for p in (C.PRIV / 'deploy').glob('*/PACKAGE.json')) if (C.PRIV / 'deploy').is_dir() else []
    if pkgs:
        from . import deploy as DP
        dep = DP.check(pkgs, Path(out).with_name('restore_deploy_detail.json'))
    else:
        dep = {'all_bitwise': None, 'all_refused': None}
    rec = {'admitted_copies_rehash_ok': not bad_adm, 'admitted_copies_checked': len(adm), 'mismatched': bad_adm[:10],
           'records_checked': n_rec, 'files_rehashed': n_files, 'records_mismatched': rec_bad[:20],
           'size_fits_rerun_bitwise': models, 'reader_grids': grid, 'deployment_packages': len(pkgs),
           'deployment_from_backup_bitwise': dep['all_bitwise'], 'deployment_from_backup_all_refused': dep.get('all_refused'),
           'labels_read': False}
    rec['all_ok'] = (rec['admitted_copies_rehash_ok'] and not rec_bad and all(models.values())
                     and all(v['json_parses'] and v['boot_present'] for v in grid.values())
                     and dep['all_bitwise'] is not False and dep.get('all_refused') is not False)
    Path(out).write_text(json.dumps(rec, indent=1) + '\n')
    print(json.dumps({k: v for k, v in rec.items() if k not in ('reader_grids', 'size_fits_rerun_bitwise')}))


if __name__ == '__main__':
    main(sys.argv[1])
