"""V1 source admission replay (label-free): own git and hash calls against SOURCE_ADMISSION.json.

- the pinned new-release-mechanisms publication on the GitHub remote (git ls-remote) and the branch base;
- the private source artefacts reused read-only, re-hashed against their OWN records: TASK.pkl against TASK.sha256 and against the
  twins' RELEASE.json task_cache_sha256; MODEL.pkl against MODEL.json model_sha256; RELEASE.npz against RELEASE.json
  release_sha256; the GATE records against the published blobs of the pinned publication (git hash-object vs ls-tree);
- every one of these hashes against the value SOURCE_ADMISSION.json recorded;
- no file under the source's pinned private directories modified after this campaign's start (22:49:23Z), read-only rule;
- the MODEL_SELECT wires of the admitted references (own wire hash of the arrays from the admitted loaders
  ohn.devrun.reference_release and tsz.evaluate source_a1 / h3_income; no metric code) against the admission record;
- the python3 library versions against the admission record.

    python3 .../v1_admission.py --tag V1
    python3 .../v1_admission.py --selftest      (invented files in a temporary directory)
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vlib as V  # noqa: E402

NRM_PIN = '682068d52147bc1bdfc75d9e49199115b1deaaed'
NRM_TIP = 'fd933a824'
NRM_PUB = 'publications/pcrl-new-release-mechanisms-v1'
TWINS = ('PRC_TWIN', 'PRC_TWIN_C128')
REFERENCES = ('U0', 'Ucal', 'T_TREE', 'TASK_REFIT', 'NEURAL_RAW', 'H_LJ', 'B2TWIN', 'RETIRED_B3CTL', 'SUCC', 'TWJ', 'H3+TASK_REFIT',
              'H3+T_TREE')
START = V.parse_utc('2026-10-20T22:49:23+00:00')


def rehash(src_runs, src_cache, versions=(0, 1), twins=TWINS):
    """Own re-hash of the pinned private artefacts against their own records. Returns (artefacts, problems)."""
    arts, probs = {}, []
    for v in versions:
        d = Path(src_cache) / f'V{v}'
        t = V.file_sha(d / 'TASK.pkl')
        recorded = (d / 'TASK.sha256').read_text().strip()
        arts[f'fa_cache/V{v}/TASK.pkl'] = {'sha256': t, 'equals_TASK.sha256': t == recorded}
        if t != recorded:
            probs.append(f'TASK V{v} differs from TASK.sha256')
        for tw in twins:
            md = Path(src_runs) / 'FA' / tw / f'V{v}'
            meta = json.loads((md / 'MODEL.json').read_text())
            relrec = json.loads((md / 'RELEASE.json').read_text())
            m, r = V.file_sha(md / 'MODEL.pkl'), V.file_sha(md / 'RELEASE.npz')
            row = {'MODEL.pkl': m, 'RELEASE.npz': r, 'model_json_agrees': m == meta.get('model_sha256'),
                   'release_json_agrees': r == relrec.get('release_sha256'),
                   'record_task_cache_sha256_agrees': relrec.get('task_cache_sha256') == t,
                   'model_json_identity': meta.get('config_id') == tw and int(meta.get('version', -1)) == v}
            arts[f'runs/FA/{tw}/V{v}'] = row
            for k in ('model_json_agrees', 'release_json_agrees', 'record_task_cache_sha256_agrees', 'model_json_identity'):
                if not row[k]:
                    probs.append(f'{tw} V{v}: {k} false')
    return arts, probs


def modified_after(dirs, start):
    out = []
    for d in dirs:
        d = Path(d)
        if not d.exists():
            continue
        for p in sorted(d.rglob('*')):
            if p.is_file() and dt.datetime.fromtimestamp(p.stat().st_mtime, dt.timezone.utc) > start:
                out.append(V.rel(p))
    return out


def compare_admission(arts, adm_arts):
    diffs = []
    for k, row in arts.items():
        a = adm_arts.get(k)
        if a is None:
            diffs.append(f'{k} absent from SOURCE_ADMISSION.json')
            continue
        for f in ('sha256', 'MODEL.pkl', 'RELEASE.npz'):
            if f in row and row[f] != a.get(f):
                diffs.append(f'{k} {f}: mine {row[f]} recorded {a.get(f)}')
    return diffs


def run(tag):
    adm_path = V.RESULTS / 'SOURCE_ADMISSION.json'
    adm = json.loads(adm_path.read_text())
    res, probs = {}, []
    rc, ls, err = V.git('ls-remote', V.REMOTE_URL, 'refs/heads/publication/pcrl-new-release-mechanisms-v1')
    res['nrm_remote_tip'] = ls.split()[0] if rc == 0 and ls else f'unavailable: {err[-200:]}'
    if res['nrm_remote_tip'] != NRM_PIN:
        probs.append('pinned source publication is not the remote branch tip')
    res['branch_base_ancestor_of_HEAD'] = V.is_ancestor(NRM_TIP)
    if not res['branch_base_ancestor_of_HEAD']:
        probs.append('source research tip fd933a824 is not an ancestor of HEAD')
    arts, p2 = rehash(V.SRC_RUNS, V.SRC_FA_CACHE)
    probs += p2
    # GATE records against the pinned publication's blobs
    rc, tree, _ = V.git('ls-tree', '-r', NRM_PIN, '--', f'{NRM_PUB}/results/roles/designer_a/')
    pub_blobs = {line.split('\t')[1].rsplit('/', 1)[1]: line.split()[2] for line in tree.splitlines() if '\t' in line}
    for g in sorted((V.SRC_RESULTS / 'roles' / 'designer_a').glob('GATE_V*.json')):
        _, local_blob, _ = V.git('hash-object', str(g))
        row = {'sha256': V.file_sha(g), 'published_blob_equal': pub_blobs.get(g.name) == local_blob}
        arts[f'results/designer_a/{g.name}'] = row
        if not row['published_blob_equal']:
            probs.append(f'{g.name} differs from the pinned publication')
    res['artefacts'] = arts
    diffs = compare_admission(arts, adm.get('source_private_artefacts', {}))
    res['differences_from_SOURCE_ADMISSION'] = diffs
    probs += diffs
    mod = modified_after([V.SRC_RUNS / 'FA' / tw for tw in TWINS] + [V.SRC_FA_CACHE / f'V{v}' for v in (0, 1)]
                         + [V.SRC_RESULTS / 'roles' / 'designer_a'], START)
    res['source_files_modified_after_campaign_start'] = mod
    if mod:
        probs.append('source files modified after the campaign start (read-only rule)')
    # reference wires (label-free arrays through the admitted loaders; own wire hash)
    if str(V.ROOT) not in sys.path:
        sys.path.insert(0, str(V.ROOT))
    from ohn.devrun import reference_release
    from tsz.evaluate import source_a1, h3_income
    rec_w = adm.get('reference_releases_MODEL_SELECT_wire_sha256', {})
    wires, wbad = {}, []
    for v in (0, 1):
        for name in REFERENCES:
            r = reference_release(name, v, 'MODEL_SELECT')
            wires[f'{name}/S{v}'] = {t: V.wire_sha(r[t]) for t in V.TASKS}
        wires[f'W/S{v}'] = {'income': V.wire_sha(h3_income(v)), 'occupation': V.wire_sha(source_a1(v))}
    for k, w in wires.items():
        rw = rec_w.get(k) or {}
        if any(w[t] != rw.get(t) for t in V.TASKS):
            wbad.append(k)
    res['reference_wires_mine'] = wires
    res['reference_wires_differ'] = wbad
    if wbad:
        probs.append(f'reference wires differ from the admission record: {wbad}')
    import numpy, scipy, sklearn  # noqa: E401
    res['python3_versions'] = {'numpy': numpy.__version__, 'scipy': scipy.__version__, 'scikit-learn': sklearn.__version__}
    ip = adm['interpreters']['python3']
    if (ip.get('numpy'), ip.get('scipy'), ip.get('scikit-learn')) != (numpy.__version__, scipy.__version__, sklearn.__version__):
        probs.append('python3 library versions differ from the admission record')
    res['admission_ok_flag'] = adm.get('ok')
    if adm.get('ok') is not True:
        probs.append('SOURCE_ADMISSION.json ok flag is not true')
    chk = {'id': f'V1.SOURCE_ADMISSION__{tag}',
           'what': 'source admission replayed label-free with own calls: remote pin of the NRM publication, branch base, pinned '
                   'private artefacts re-hashed against their own records (TASK.sha256, MODEL.json, RELEASE.json, published GATE '
                   'blobs) and against SOURCE_ADMISSION.json, source files unmodified since the campaign start, 26 reference '
                   'MODEL_SELECT wires, interpreter versions',
           'inputs': V.inputs_sha([adm_path]), 'result': {**res, 'problems': probs},
           'numerical_status': 'PASS' if not (wbad or diffs or p2) else 'FAIL',
           'procedural_status': 'PASS' if not probs else 'FAIL'}
    print(json.dumps(V.append_checks([chk]), indent=1))
    print(json.dumps({k: x for k, x in res.items() if k not in ('reference_wires_mine', 'artefacts')}, indent=1,
                     default=V._jd)[:4000])
    print(json.dumps({'problems': probs}, indent=1))
    return 0


def selftest():
    """Invented artefacts in a temporary directory: a consistent set passes; each planted change is flagged."""
    import numpy as np
    ok = {}
    with tempfile.TemporaryDirectory() as tmp:
        runs, cache = Path(tmp) / 'runs', Path(tmp) / 'cache'
        (cache / 'V0').mkdir(parents=True)
        (cache / 'V0' / 'TASK.pkl').write_bytes(b'task-bytes')
        (cache / 'V0' / 'TASK.sha256').write_text(V.bytes_sha(b'task-bytes'))
        md = runs / 'FA' / 'TW' / 'V0'
        md.mkdir(parents=True)
        (md / 'MODEL.pkl').write_bytes(b'model')
        (md / 'MODEL.json').write_text(json.dumps({'model_sha256': V.bytes_sha(b'model'), 'config_id': 'TW', 'version': 0}))
        np.savez(md / 'RELEASE.npz', a=np.arange(3))
        (md / 'RELEASE.json').write_text(json.dumps({'release_sha256': V.file_sha(md / 'RELEASE.npz'),
                                                     'task_cache_sha256': V.bytes_sha(b'task-bytes')}))
        arts, probs = rehash(runs, cache, versions=(0,), twins=('TW',))
        ok['consistent_set_passes'] = probs == []
        adm = {k: dict(v) for k, v in arts.items()}
        ok['admission_compare_equal'] = compare_admission(arts, adm) == []
        adm['runs/FA/TW/V0']['MODEL.pkl'] = 'x'
        ok['admission_compare_flags_change'] = bool(compare_admission(arts, adm))
        (md / 'MODEL.pkl').write_bytes(b'model2')
        ok['flags_changed_model'] = any('model_json_agrees' in p for p in rehash(runs, cache, (0,), ('TW',))[1])
        (md / 'MODEL.pkl').write_bytes(b'model')
        (cache / 'V0' / 'TASK.pkl').write_bytes(b'task-bytes2')
        pr = rehash(runs, cache, (0,), ('TW',))[1]
        ok['flags_changed_task_cache'] = any('TASK V0' in p for p in pr) and any('task_cache' in p for p in pr)
        ok['modified_after_detects'] = bool(modified_after([cache], dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc)))
        ok['modified_after_quiet_for_future_start'] = modified_after([cache], dt.datetime(2100, 1, 1, tzinfo=dt.timezone.utc)) == []
    ok['all_ok'] = all(ok.values())
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        ok = selftest()
        print(json.dumps(ok, indent=1))
        return 0 if ok['all_ok'] else 1
    return run(a.tag)


if __name__ == '__main__':
    sys.exit(main())
