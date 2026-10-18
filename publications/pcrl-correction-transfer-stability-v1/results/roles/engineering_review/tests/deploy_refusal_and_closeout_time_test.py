"""Engineering review (CTS1): invented-data tests of the deployment refusals and of the closeout freeze-time comparison.
No person data, no label, no stored model: a fake package directory with an empty dependency list and invented rows / matrices.

    python3 results/pcrl_correction_transfer_stability_v1/roles/engineering_review/tests/deploy_refusal_and_closeout_time_test.py
"""
from __future__ import annotations
import csv
import datetime as dt
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

WT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(WT))
from cts import deploy as D  # noqa: E402

NAMES = ['AGEP', 'WKHP', 'COW=1', 'COW=__OTHER__', 'SCHL=21', 'SCHL=__OTHER__', 'MAR=1', 'MAR=__OTHER__', 'POBP=12', 'POBP=__OTHER__']
ROW = {'AGEP': 40, 'WKHP': 40, 'COW': 1, 'SCHL': 21, 'MAR': 1, 'POBP': 12}


def refused(fn, *a, **k):
    try:
        fn(*a, **k)
        return 'NOT_REFUSED'
    except PermissionError:
        return 'PermissionError'
    except ValueError:
        return 'ValueError'
    except Exception as e:                          # any other failure: not a clean refusal, reported as such
        return f'OTHER:{type(e).__name__}'


def main():
    out = {}
    with tempfile.TemporaryDirectory() as td:
        pkg = Path(td) / 'pkg'; pkg.mkdir()
        (pkg / 'PACKAGE.json').write_text(json.dumps({'depends_sha256': {}, 'seed': 0, 'name': 'W', 'partition': 'D0',
                                                      'encoder': {'feature_names': NAMES}}))
        rows = {
            'SEX_field': [{**ROW, 'SEX': 1}], 'sex_lowercase_field': [{**ROW, 'sex': 2}], 'target_OCCP': [{**ROW, 'OCCP': '0010'}],
            'target_PINCP': [{**ROW, 'PINCP': 1}], 'extra_nonforbidden_FOO': [{**ROW, 'FOO': 1}],
            'reordered': [dict(reversed(list(ROW.items())))], 'missing_POBP': [{k: v for k, v in ROW.items() if k != 'POBP'}],
            'csv_restkey_None': [{**ROW, None: ['7']}]}
        out['wrapper'] = {k: refused(D.release, str(pkg), v) for k, v in rows.items()}
        out['hidden_output'] = {'raw_score': refused(D.release, str(pkg), [ROW], requested=('probability', 'raw_score')),
                                'string_request': refused(D.release, str(pkg), [ROW], requested='probability')}
        csvs = {'csv_extra_FOO': list(D.PERMITTED) + ['FOO'], 'csv_Sex_mixed_case': list(D.PERMITTED) + ['Sex'],
                'csv_reordered': list(D.PERMITTED)[::-1], 'csv_missing': list(D.PERMITTED)[:-1]}
        out['csv'] = {}
        for k, hdr in csvs.items():
            q = Path(td) / f'{k}.csv'
            with open(q, 'w', newline='') as f:
                w = csv.writer(f); w.writerow(hdr); w.writerow([ROW.get(h, 1) for h in hdr])
            out['csv'][k] = refused(D.release_csv, str(pkg), q)
        X = np.zeros((3, len(NAMES)))
        mats = {'SEX_column': (np.column_stack((X, np.ones(3))), NAMES + ['SEX']), 'SEX_onehot_column': (np.column_stack((X, np.ones(3))), NAMES + ['SEX=2']),
                'target_column': (np.column_stack((X, np.ones(3))), NAMES + ['OCCP']), 'extra_column': (np.column_stack((X, np.ones(3))), NAMES + ['EXTRA']),
                'missing_column': (X[:, 1:], NAMES[1:]), 'reordered_columns': (X[:, ::-1], NAMES[::-1]),
                'nonfinite': (np.where(np.eye(3, len(NAMES)) > 0, np.nan, X), NAMES), 'wrong_width': (X[:, :-1], NAMES)}
        out['underlying'] = {k: refused(D.underlying, str(pkg), Xb, nb) for k, (Xb, nb) in mats.items()}
        out['empty_dependency_manifest_loads'] = refused(D._load, str(pkg)) == 'NOT_REFUSED'
    clean = [v for grp in ('wrapper', 'hidden_output', 'csv', 'underlying') for v in out[grp].values()]
    out['all_refused_cleanly'] = all(v in ('PermissionError', 'ValueError') for v in clean)
    out['SEX_and_targets_PermissionError'] = all(out['wrapper'][k] == 'PermissionError' for k in
                                                 ('SEX_field', 'sex_lowercase_field', 'target_OCCP', 'target_PINCP')) and \
        out['underlying']['SEX_column'] == 'PermissionError' and out['underlying']['SEX_onehot_column'] == 'PermissionError'

    # closeout: the committed comparison `e['utc'] < freeze_c` with freeze_c from git's %cI (local -04:00 offset here)
    freeze_local = '2026-10-18T19:00:00-04:00'                 # = 23:00Z
    read_before = {'utc': '2026-10-18T22:00:00+00:00', 'operation': 'check_repair'}   # one hour BEFORE the freeze
    read_after = {'utc': '2026-10-18T23:30:00+00:00', 'operation': 'check_repair'}    # after
    out['closeout_committed_misses_pre_freeze_read'] = not (read_before['utc'] < freeze_local)
    out['closeout_committed_no_freeze_hides_reads'] = len([e for e in [read_before] if e['operation'] not in ('source_replay',)
                                                           and None and e['utc'] < None]) == 0
    import shutil
    import subprocess
    with tempfile.TemporaryDirectory() as td2:
        (Path(td2) / 'cts').mkdir()
        shutil.copy(WT / 'cts' / 'closeout.py', Path(td2) / 'cts' / 'closeout.py')
        r = subprocess.run(['patch', '-p1', '-s', '-i', str(Path(__file__).resolve().parents[1] / 'PATCHES' / '23.diff')], cwd=td2,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        spec = importlib.util.spec_from_file_location('cts.closeout_patched', Path(td2) / 'cts' / 'closeout.py')
        m = importlib.util.module_from_spec(spec); m.__package__ = 'cts'; spec.loader.exec_module(m)
        fz = dt.datetime.fromisoformat(freeze_local).astimezone(dt.timezone.utc)
        got = m.panel_reads_before_freeze([read_before, read_after, {'utc': '2026-10-18T20:00:00+00:00', 'operation': 'source_replay'}], fz)
        out['closeout_patched_flags_only_pre_freeze_new_release_read'] = got == [read_before]
        out['closeout_patched_no_freeze_flags_all_new_release_reads'] = m.panel_reads_before_freeze([read_before, read_after], None) == [read_before, read_after]
        out['closeout_patched_freeze_commit_utc_now'] = str(m.freeze_commit_utc())
    print(json.dumps(out, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
