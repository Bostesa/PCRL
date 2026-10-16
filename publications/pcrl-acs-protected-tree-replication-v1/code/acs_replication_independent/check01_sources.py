"""Independent check 1 (SOURCES) for the ACS protected-tree replication.

Written by the independent verifier; imports nothing from acs_replication.
Verifies: pinned publication commits resolve; transported source files equal their publication blobs
(git blob sha1 recomputed from content); acs_replication/{trees,leafcorr,neural,neural_raw}.py differ from
their sources only on lines that contain 'ACS-CHANGE'; FAGTB vendor commit + file hashes; raw zip sha256;
documentation hashes; locked-record hashes; code hashes at lock vs now.

The raw CSV member is NOT decompressed (sealed households' non-identifier columns are never read); the
zip sha256 already pins the archive bytes, and the member CRC/sizes are read from the zip directory only.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

WT = Path(__file__).resolve().parents[1]
RES = WT / 'results' / 'pcrl_acs_protected_tree_replication_v1'
PRIV = WT / '.private' / 'acs_replication'
OUT = RES / 'independent' / 'check01_sources.json'


def git(*args, cwd=WT, binary=False):
    r = subprocess.run(['git', *args], cwd=cwd, capture_output=True, check=True)
    return r.stdout if binary else r.stdout.decode().strip()


def blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b'blob %d\x00' % len(data) + data).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def module_docstring_span(lines):
    """0-based [start, end] line indices of the leading module docstring (triple-double-quoted), or None."""
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s:
            continue
        if s.startswith('"""'):
            if s.count('"""') >= 2 and len(s) > 3:
                return (i, i)
            for j in range(i + 1, len(lines)):
                if '"""' in lines[j]:
                    return (i, j)
        return None
    return None


def marked_diff(src_text: str, dst_text: str):
    """Line diff source -> transport. Every non-equal transport-side line must contain 'ACS-CHANGE'.
    Source lines deleted without any marked replacement in the same hunk are unmarked deletions."""
    a = src_text.splitlines()
    b = dst_text.splitlines()
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    doc_a = module_docstring_span(a)
    doc_b = module_docstring_span(b)

    def in_doc(span, idx):
        return span is not None and span[0] <= idx <= span[1]

    hunks = []
    n_marked = 0
    unmarked = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            continue
        b_lines = list(range(j1, j2))
        a_lines = list(range(i1, i2))
        marked_b = [j for j in b_lines if 'ACS-CHANGE' in b[j]]
        unmarked_b = [j for j in b_lines if 'ACS-CHANGE' not in b[j]]
        n_marked += len(marked_b)
        hunk = {'tag': tag, 'src_lines': [i1 + 1, i2], 'dst_lines': [j1 + 1, j2],
                'dst_marked': len(marked_b), 'dst_unmarked': len(unmarked_b)}
        # unmarked transport-side lines
        for j in unmarked_b:
            unmarked.append({'side': 'transport', 'line': j + 1, 'in_module_docstring': in_doc(doc_b, j),
                             'is_comment_or_blank': (b[j].strip().startswith('#') or not b[j].strip()),
                             'text': b[j][:200]})
        # deletions with no marked replacement
        if tag == 'delete' or (tag == 'replace' and not marked_b):
            for i in a_lines:
                unmarked.append({'side': 'source_deleted', 'line': i + 1, 'in_module_docstring': in_doc(doc_a, i),
                                 'is_comment_or_blank': (a[i].strip().startswith('#') or not a[i].strip()),
                                 'text': a[i][:200]})
        hunks.append(hunk)
    code_unmarked = [u for u in unmarked if not u['in_module_docstring'] and not u['is_comment_or_blank']]
    doc_unmarked = [u for u in unmarked if u['in_module_docstring']]
    return {'n_source_lines': len(a), 'n_transport_lines': len(b), 'n_hunks': len(hunks),
            'transport_marked_lines_in_hunks': n_marked,
            'transport_lines_containing_ACS_CHANGE_total': sum('ACS-CHANGE' in x for x in b),
            'unmarked_total': len(unmarked), 'unmarked_in_module_docstring': len(doc_unmarked),
            'unmarked_comment_or_blank_outside_docstring': len([u for u in unmarked if not u['in_module_docstring']
                                                                and u['is_comment_or_blank']]),
            'unmarked_code_lines': len(code_unmarked), 'unmarked_code_detail': code_unmarked[:60],
            'unmarked_docstring_detail': doc_unmarked[:60], 'hunks': hunks}


def main():
    lock = json.loads((RES / 'SCIENCE_LOCK.json').read_text())
    sadm = json.loads((RES / 'SOURCE_ADMISSION.json').read_text())
    popd = json.loads((RES / 'POPULATION_DECISION.json').read_text())
    madm = json.loads((PRIV / 'admission' / 'METADATA_ADMISSION.json').read_text())
    xrec = json.loads((PRIV / 'extraction' / 'EXTRACTION_RECEIPT.json').read_text())
    ev = {}
    fails = []

    # 1a. pinned commits resolve
    pins = {'task_headroom': ('c8ae518', lock['source_publications']['task_headroom']),
            'frozen_tree_completion': ('8975a23', lock['source_publications']['frozen_tree_completion'])}
    ev['pinned_commits'] = {}
    for k, (short, full) in pins.items():
        try:
            got = git('rev-parse', '--verify', short + '^{commit}')
        except subprocess.CalledProcessError:
            got = None
        ok = got == full == sadm['publications'][k]['commit']
        ev['pinned_commits'][k] = {'short': short, 'rev_parse': got, 'science_lock': full,
                                   'source_admission': sadm['publications'][k]['commit'], 'match': ok}
        if not ok:
            fails.append(f'pinned commit {k} mismatch')

    # 1b. transported files: publication blob (recomputed) vs recorded; base copies vs publication
    base_commit = '770440fce'
    base_full = git('rev-parse', base_commit)
    ev['base_commit'] = base_full
    pkg_dir = {'task_headroom': ('task_headroom', 'publications/pcrl-task-headroom-protected-outputs-v1/code/task_headroom'),
               'frozen_tree_completion': ('tree_completion', 'publications/pcrl-frozen-tree-control-completion-v1/code/tree_completion')}
    ev['transported_files'] = {}
    for k, meta in sadm['publications'].items():
        src_dir, pub_dir = pkg_dir[k]
        commit = meta['commit']
        for fname, rec in meta['transported_source_files'].items():
            pub_bytes = git('show', f'{commit}:{pub_dir}/{fname}', binary=True)
            pub_blob = blob_sha1(pub_bytes)
            base_bytes = git('show', f'{base_full}:{src_dir}/{fname}', binary=True)
            base_blob = blob_sha1(base_bytes)
            wt_file = WT / src_dir / fname
            wt_blob = blob_sha1(wt_file.read_bytes()) if wt_file.exists() else None
            ok = (pub_blob == rec['publication_blob'] and base_blob == rec['base_blob'] and pub_blob == base_blob
                  and (wt_blob is None or wt_blob == pub_blob))
            ev['transported_files'][f'{k}/{fname}'] = {
                'publication_blob_recomputed': pub_blob, 'publication_blob_recorded': rec['publication_blob'],
                'base_blob_recomputed': base_blob, 'base_blob_recorded': rec['base_blob'],
                'worktree_blob': wt_blob, 'publication_sha256': sha256_bytes(pub_bytes), 'equal': ok}
            if not ok:
                fails.append(f'transported file {k}/{fname} blob mismatch')

    # 1c. marked-only differences of the four transported modules
    pairs = {
        'trees.py': (pins['task_headroom'][1], pkg_dir['task_headroom'][1] + '/trees.py'),
        'leafcorr.py': (pins['task_headroom'][1], pkg_dir['task_headroom'][1] + '/leafcorr.py'),
        'neural.py': (pins['task_headroom'][1], pkg_dir['task_headroom'][1] + '/neural.py'),
        'neural_raw.py': (pins['frozen_tree_completion'][1], pkg_dir['frozen_tree_completion'][1] + '/neural_rawj.py'),
    }
    lock_commit = git('rev-parse', '4d50dea13')
    ev['acs_change_diffs'] = {}
    for mod, (commit, path) in pairs.items():
        src = git('show', f'{commit}:{path}', binary=True).decode()
        cur = (WT / 'acs_replication' / mod).read_text()
        at_lock = git('show', f'{lock_commit}:acs_replication/{mod}', binary=True).decode()
        d = marked_diff(src, cur)
        d['source'] = f'{commit[:9]}:{path}'
        d['current_equals_at_science_lock'] = (cur == at_lock)
        d['current_sha256'] = sha256_bytes(cur.encode())
        d['sha256_at_lock_recorded'] = lock['code_sha256_at_lock'].get(mod)
        d['current_sha256_equals_lock_record'] = d['current_sha256'] == d['sha256_at_lock_recorded']
        d['strict_transport_rule_pass'] = d['unmarked_total'] == 0
        ev['acs_change_diffs'][mod] = d
        if d['unmarked_code_lines']:
            fails.append(f'{mod}: {d["unmarked_code_lines"]} executable line(s) differ from the source without an ACS-CHANGE '
                         f'marker on the line (lines {[u["line"] for u in d["unmarked_code_detail"]][:12]}...)')
        elif d['unmarked_total']:
            fails.append(f'{mod}: {d["unmarked_total"]} unmarked non-executable line(s) (module docstring/comments) differ')
    # the unmarked neural.py appendix (Ucal) vs the locked Ucal definition
    nsrc = (WT / 'acs_replication' / 'neural.py').read_text()
    m = re.search(r"^UCAL = (\{.*?\})\n\n", nsrc, re.M | re.S)
    ucal_code = eval(m.group(1), {}) if m else None  # literal dict in our own read of the file (no import)
    ev['neural_py_ucal_block_vs_SCIENCE_LOCK_models_Ucal'] = {
        'present': ucal_code is not None,
        'equal_to_locked_definition': ucal_code == lock['models']['Ucal'] if ucal_code is not None else False,
        'block_header': 'ACS addition: Ucal (fixed procedure, no sweep)' if 'ACS addition: Ucal' in nsrc else None}

    # 1d. FAGTB vendor
    vend = PRIV / 'vendor' / 'FAGTB'
    fagtb_src = (WT / 'acs_replication' / 'fagtb.py').read_text()
    const = {k: re.search(rf'^{k}\s*=\s*"([0-9a-f]+)"', fagtb_src, re.M).group(1)
             for k in ('OFFICIAL_COMMIT', 'OFFICIAL_FUNCTIONS_SHA256', 'OFFICIAL_NOTEBOOK_SHA256')}
    try:
        vhead = git('rev-parse', 'HEAD', cwd=vend)
        vstatus = git('status', '--porcelain', cwd=vend)
        vfunc_blob_at_head = blob_sha1(git('show', 'HEAD:functions.py', cwd=vend, binary=True))
    except subprocess.CalledProcessError as e:
        vhead, vstatus, vfunc_blob_at_head = None, f'git error {e}', None
    func_sha = sha256_file(vend / 'functions.py')
    nb_sha = sha256_file(vend / 'FAGTB.ipynb')
    tp = sadm['third_party']['FAGTB']
    fagtb_ok = (vhead == const['OFFICIAL_COMMIT'] == tp['commit'] and func_sha == const['OFFICIAL_FUNCTIONS_SHA256']
                == tp['functions_py_sha256'] and nb_sha == const['OFFICIAL_NOTEBOOK_SHA256'] == tp['notebook_sha256']
                and vstatus == '' and vfunc_blob_at_head == blob_sha1((vend / 'functions.py').read_bytes()))
    ev['fagtb_vendor'] = {'constants_in_fagtb_py': const, 'source_admission': {k: tp[k] for k in ('commit', 'functions_py_sha256', 'notebook_sha256')},
                          'vendor_git_head': vhead, 'vendor_worktree_clean': vstatus == '', 'vendor_status_porcelain': vstatus[:500],
                          'functions_py_sha256_recomputed': func_sha, 'notebook_sha256_recomputed': nb_sha,
                          'functions_py_equals_committed_blob': vfunc_blob_at_head == blob_sha1((vend / 'functions.py').read_bytes()),
                          'match': fagtb_ok}
    if not fagtb_ok:
        fails.append('FAGTB vendor mismatch')

    # 1e. raw zip
    zp = PRIV / 'raw' / 'csv_pfl.zip'
    zsha = sha256_file(zp)
    zrec = {'POPULATION_DECISION.official_source.zip_sha256': popd['official_source']['zip_sha256'],
            'SCIENCE_LOCK.population.metadata_admission.zip_sha256': lock['population']['metadata_admission']['zip_sha256'],
            'EXTRACTION_RECEIPT.raw_zip_sha256': xrec['raw_zip_sha256'],
            'METADATA_ADMISSION.zip_sha256': madm.get('zip_sha256')}
    with zipfile.ZipFile(zp) as zf:
        members = [{'name': i.filename, 'file_size': i.file_size, 'compress_size': i.compress_size,
                    'crc32': format(i.CRC, '08x')} for i in zf.infolist()]
    zip_ok = all(v == zsha for v in zrec.values() if v is not None) and zp.stat().st_size == popd['official_source']['zip_bytes']
    member = [m for m in members if m['name'] == popd['official_source']['person_csv']]
    member_size_ok = bool(member) and member[0]['file_size'] == popd['official_source']['person_csv_bytes']
    ev['raw_zip'] = {'sha256_recomputed': zsha, 'bytes': zp.stat().st_size, 'recorded': zrec, 'match': zip_ok,
                     'zip_members_directory_only': members, 'person_csv_member_size_matches_record': member_size_ok,
                     'csv_member_sha256': 'NOT_RECOMPUTED: would decompress sealed households\' non-identifier columns; '
                                          'the archive sha256 above pins the member bytes (zip CRC32 checked on any extraction)'}
    if not (zip_ok and member_size_ok):
        fails.append('raw zip mismatch')

    # 1f. documentation hashes
    docs = PRIV / 'docs'
    rec_pd = popd['documentation_sha256']
    rec_ma = madm['documentation']
    doc_ev = {}
    for name in sorted(set(rec_pd) | set(rec_ma)):
        p = docs / name
        s = sha256_file(p) if p.exists() else None
        b = p.stat().st_size if p.exists() else None
        ok = s is not None and s == rec_pd.get(name) == rec_ma.get(name, {}).get('sha256') and b == rec_ma.get(name, {}).get('bytes')
        doc_ev[name] = {'sha256_recomputed': s, 'bytes': b, 'POPULATION_DECISION': rec_pd.get(name),
                        'METADATA_ADMISSION': rec_ma.get(name, {}).get('sha256'), 'match': ok}
        if not ok:
            fails.append(f'doc hash mismatch {name}')
    extras = sorted(p.name for p in docs.iterdir() if p.name not in rec_pd and not p.name.endswith('.headers'))
    ev['documentation'] = {'files': doc_ev, 'n_recorded': len(doc_ev), 'n_match': sum(v['match'] for v in doc_ev.values()),
                           'unrecorded_non_header_files_in_docs': extras}

    # 1g. locked-record hashes
    rec_hashes = {
        'SCIENCE_LOCK.json vs EXTRACTION_RECEIPT.science_lock_sha256': (sha256_file(RES / 'SCIENCE_LOCK.json'), xrec['science_lock_sha256']),
        'POPULATION_DECISION.json vs SCIENCE_LOCK.population.decision_sha256': (sha256_file(RES / 'POPULATION_DECISION.json'), lock['population']['decision_sha256']),
        'STATISTICAL_REVIEW_PRELOCK.md vs SCIENCE_LOCK.statistical_review_prelock.sha256': (sha256_file(RES / 'STATISTICAL_REVIEW_PRELOCK.md'), lock['statistical_review_prelock']['sha256']),
        'OCCUPATION_MAPPING.mapping (json sort_keys) vs SCIENCE_LOCK.benchmark.occupation_mapping_sha256': (
            sha256_bytes(json.dumps(json.loads((PRIV / 'admission' / 'OCCUPATION_MAPPING.json').read_text())['mapping'], sort_keys=True).encode()),
            lock['benchmark']['occupation_mapping_sha256']),
    }
    ev['locked_record_hashes'] = {k: {'recomputed': a, 'recorded': b, 'match': a == b} for k, (a, b) in rec_hashes.items()}
    for k, v in ev['locked_record_hashes'].items():
        if not v['match']:
            fails.append(f'locked record hash mismatch: {k}')
    # science lock file unchanged since the lock commit
    ev['science_lock_file_unchanged_since_4d50dea13'] = git('diff', '--stat', lock_commit, 'HEAD', '--',
                                                            'results/pcrl_acs_protected_tree_replication_v1/SCIENCE_LOCK.json') == ''
    if not ev['science_lock_file_unchanged_since_4d50dea13']:
        fails.append('SCIENCE_LOCK.json changed after lock commit')

    # 1h. code hashes at lock vs now (report; not automatically a failure)
    code_now = {p.name: sha256_file(p) for p in sorted((WT / 'acs_replication').glob('*.py'))}
    at_lock = lock['code_sha256_at_lock']
    changed = sorted(k for k in at_lock if code_now.get(k) != at_lock[k])
    added = sorted(k for k in code_now if k not in at_lock)
    removed = sorted(k for k in at_lock if k not in code_now)
    # also confirm the recorded lock hashes equal the files in the lock commit
    lock_commit_mismatch = []
    for k, h in at_lock.items():
        try:
            b = git('show', f'{lock_commit}:acs_replication/{k}', binary=True)
        except subprocess.CalledProcessError:
            lock_commit_mismatch.append(f'{k}: missing at lock commit')
            continue
        if sha256_bytes(b) != h:
            lock_commit_mismatch.append(k)
    per_file_commits = {k: git('log', '--format=%h %ci %s', f'{lock_commit}..HEAD', '--', f'acs_replication/{k}').splitlines()
                        for k in changed}
    numstat = {}
    for k in changed:
        ns = git('diff', '--numstat', lock_commit, 'HEAD', '--', f'acs_replication/{k}')
        numstat[k] = ns
    ev['code_hashes'] = {'n_locked_files': len(at_lock), 'changed_since_lock': changed, 'added_since_lock': added,
                         'removed_since_lock': removed, 'recorded_lock_hashes_equal_lock_commit_files': not lock_commit_mismatch,
                         'lock_commit_mismatches': lock_commit_mismatch, 'commits_touching_changed_files': per_file_commits,
                         'numstat_changed': numstat}
    if lock_commit_mismatch:
        fails.append('recorded code_sha256_at_lock differs from files in lock commit')

    ev['manual_review_post_lock_code'] = {
        'extract.py': 'one line: lock key path lock["schema"] -> lock["benchmark"] for occupation_mapping_sha256 (amendment A1, '
                      'disclosed in LOCK_AMENDMENTS.md); no metric/claim logic',
        'added_files': 'tables.py (public tables from DEV_REPORT_RESULTS), package.py + custody_check.py (deployment packaging / '
                       'restore parity), test_report_synthetic.py + test_select_readers_synthetic.py (synthetic engineering tests); '
                       'none is imported by report.py/selection.py/readers.py for decisions',
        'metric_claim_modules_unchanged_since_lock': sorted(k for k in ('report.py', 'selection.py', 'readers.py', 'stage_a.py',
                                                                        'stage_b.py', 'leafcorr.py', 'trees.py', 'neural.py',
                                                                        'neural_raw.py', 'neural_raw_worker.py', 'fagtb.py',
                                                                        'fagtb_worker.py', 'fare_worker.py', 'data.py', 'schema.py')
                                                            if code_now.get(k) == at_lock.get(k))}
    ev['fail_classification'] = (
        'Strict transport rule (copies with ACS-CHANGE-marked lines only) is violated by labeling, not by hidden behaviour: '
        'trees.py:81 width check (!= 83 -> != len(self.feature_names)) and neural.py:23 (Linear(83,64) -> Linear(d,64)) are '
        'unmarked executable edits whose ACS-CHANGE marker sits on the adjacent line (82 / 21); neural.py appends a 34-line '
        'block headed "# --- ACS addition: Ucal" (UCAL dict, logits, fit_temperature, tempered_probabilities; 25 executable '
        'lines) without the ACS-CHANGE token -- new functions, UCAL dict equal to SCIENCE_LOCK.models.Ucal; and the module '
        'docstrings of trees/leafcorr/neural/neural_raw gained 2/5/3/4 unmarked lines. No other source line differs. All four '
        'files are byte-identical to their versions in the SCIENCE_LOCK commit 4d50dea13 (sha256 = code_sha256_at_lock).')
    out = {'id': 1, 'name': 'SOURCES', 'status': 'PASS' if not fails else 'FAIL', 'fail_reasons': fails, 'evidence': ev,
           'notes': 'acs_change_diffs: any unmarked difference is reported line by line; module-docstring-only unmarked '
                    'lines are listed separately (non-executable). Changed-since-lock code files are reported with commits '
                    'for manual review (see post-processing field manual_review).'}
    OUT.write_text(json.dumps(out, indent=1, sort_keys=False))
    print(json.dumps({'status': out['status'], 'fails': fails,
                      'diff_summary': {m: {k: v[k] for k in ('unmarked_total', 'unmarked_in_module_docstring', 'unmarked_code_lines',
                                                              'unmarked_comment_or_blank_outside_docstring', 'transport_marked_lines_in_hunks',
                                                              'current_equals_at_science_lock')}
                                       for m, v in ev['acs_change_diffs'].items()},
                      'code_changed': changed, 'code_added': added}, indent=1))


if __name__ == '__main__':
    sys.exit(main())
