"""Independent check 5: closed SEALED_ASSESSMENT, single DEV_REPORT opening, lock chronology and frozen-file integrity.

Reads: file names and npz identifier/integer arrays under .private/acs_replication (no label values are used),
the raw SERIALNO column (to form sealed household/person identifiers only), git metadata (commit times, reflog of
the remote-tracking publication ref, optional read-only `git ls-remote github`), the compute ledger, unit logs,
queue files and lock JSONs. Does not import acs_replication. Aggregates only.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import os
import re
import subprocess
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

WT = Path(__file__).resolve().parent.parent
RES = WT / 'results' / 'pcrl_acs_protected_tree_replication_v1'
PRIV = WT / '.private' / 'acs_replication'
OUT = RES / 'independent' / 'check05_access.json'
DEV_ROLES = ['MODEL_FIT', 'CALIBRATION', 'MODEL_SELECT', 'ATTACK_FIT', 'ATTACK_VALIDATION', 'DEV_REPORT']
COMMITS = {'SCIENCE_LOCK': '4d50dea13', 'TRAINING_COMPLETE_LOCK': '40f6df5b2', 'SELECTION_LOCK': '239ea95ac',
           'REPORTING_LOCK': '62fc5c606', 'DEV_REPORT_RESULTS': '40872a196', 'PUBLIC_RECEIPT': '7d23e0e'}
PUB_BRANCH = 'publication/pcrl-acs-protected-tree-replication-v1'

problems = []
ev = {}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def git(*a, check=True, timeout=60):
    return subprocess.run(['git', '-C', str(WT), *a], capture_output=True, text=True, check=check, timeout=timeout).stdout


def git_bytes(*a):
    return subprocess.run(['git', '-C', str(WT), *a], capture_output=True, check=True).stdout


def ts(s):
    return dt.datetime.fromisoformat(s).astimezone(dt.timezone.utc)


def iso(t):
    return t.astimezone(dt.timezone.utc).isoformat(timespec='seconds')


# ------------------------------------------------------------------ 1. no SEALED export anywhere
def walk_names(base):
    hits, n = [], 0
    if not base.exists():
        return None, 0
    for root, dirs, files in os.walk(base):
        for name in dirs + files:
            n += 1
            if 'sealed' in name.lower():
                hits.append(str(Path(root, name).relative_to(base)))
    return hits, n


name_scan = {}
for label, base in (('.private/acs_replication', PRIV), ('results/pcrl_acs_protected_tree_replication_v1', RES),
                    ('~/PCRL_eval_cache_private/acsr_v1 (custody cache, names only)', Path.home() / 'PCRL_eval_cache_private' / 'acsr_v1')):
    hits, n = walk_names(base)
    docs = [h for h in (hits or []) if h.lower().endswith('.md')]
    data_like = [h for h in (hits or []) if not h.lower().endswith('.md')]
    name_scan[label] = {'entries_scanned': n, 'names_containing_SEALED': hits if hits is not None else 'path absent',
                        'documentation_files_(.md,_not_exports)': docs, 'data_like_paths': data_like}
    if data_like:
        problems.append(f'SEALED-named data path under {label}: {data_like[:5]}')
export_dirs = sorted(p.name for p in (PRIV / 'exports').iterdir() if p.is_dir())
expected_dirs = sorted(DEV_ROLES + ['DEV_REPORT_TRUTH'])
ev['sealed_file_search'] = {'name_scan': name_scan, 'export_dirs': export_dirs, 'export_dirs_equal_expected_7': export_dirs == expected_dirs}
if export_dirs != expected_dirs:
    problems.append(f'unexpected export directories: {export_dirs}')

# role sizes from exports (identifier arrays only)
role_size = {}
for r in DEV_ROLES:
    with np.load(PRIV / 'exports' / r / 'features.npz', allow_pickle=False) as z:
        role_size[r] = int(z['row_id'].shape[0])

# sealed identifiers from SERIALNO only
bm = json.loads((RES / 'SCIENCE_LOCK.json').read_text())['benchmark']
ROLE_SALT = bm['role_salt']
HU = re.compile(r'^2019HU[0-9]{7}$')


def hkey(s):
    return f"{bm['survey']}|{bm['year']}|{bm['horizon']}|{bm['state'][1]}|{s}"


def iid(t):
    return int(hashlib.sha256(t.encode()).hexdigest()[:15], 16)


with zipfile.ZipFile(PRIV / 'raw' / 'csv_pfl.zip') as z, z.open(bm['person_csv']) as fb:
    rows_per = Counter(pd.read_csv(io.TextIOWrapper(fb, encoding='latin-1', newline=''), usecols=['SERIALNO'], dtype=str)['SERIALNO'].tolist())
# SEALED iff u < 0.2 with u = u_int / 2^64, evaluated exactly as 5 * u_int < 2^64
sealed_serials = [s for s in rows_per if HU.match(s) and 5 * int(hashlib.sha256((ROLE_SALT + hkey(s)).encode()).hexdigest()[:16], 16) < 2 ** 64]
sealed_units = np.asarray([iid(hkey(s)) for s in sealed_serials], dtype=np.int64)
sealed_rows = []
for s in sealed_serials:
    for k in range(1, rows_per[s] + 6):
        sealed_rows.append(iid(f'{hkey(s)}|{k:02d}' if k < 10 else f'{hkey(s)}|{k}'))
sealed_ids = np.unique(np.concatenate([sealed_units, np.asarray(sealed_rows, dtype=np.int64)]))

npz_scan = {'files': 0, 'arrays': 0, 'integer_arrays_scanned_for_sealed_ids': 0, 'sealed_id_hits': 0, 'keys_containing_SEALED': [],
            'output_roles_found': Counter(), 'output_shape_mismatches': [], 'reader_score_shape_mismatches': []}
for p in sorted(PRIV.rglob('*.npz')):
    rel = str(p.relative_to(PRIV))
    if rel.startswith(('pylib/', 'vendor/')):
        continue
    npz_scan['files'] += 1
    with np.load(p, allow_pickle=False) as zf:
        for k in zf.files:
            npz_scan['arrays'] += 1
            if 'sealed' in k.lower():
                npz_scan['keys_containing_SEALED'].append(f'{rel}:{k}')
            a = zf[k]
            if a.dtype.kind in 'iu' and a.size:
                npz_scan['integer_arrays_scanned_for_sealed_ids'] += 1
                hits = int(np.isin(a.astype(np.int64, copy=False).ravel(), sealed_ids).sum())
                if hits:
                    npz_scan['sealed_id_hits'] += hits
                    problems.append(f'sealed identifier found in {rel}:{k}')
            parts = k.split('__')
            if rel.startswith('runs/') and len(parts) == 3 and parts[1] in DEV_ROLES + ['SEALED_ASSESSMENT']:
                npz_scan['output_roles_found'][parts[1]] += 1
                if parts[1] not in DEV_ROLES or a.shape[0] != role_size[parts[1]]:
                    npz_scan['output_shape_mismatches'].append(f'{rel}:{k}:{a.shape}')
            elif rel.endswith('DEV_REPORT_SCORES.npz'):
                if a.shape[0] != role_size['DEV_REPORT']:
                    npz_scan['reader_score_shape_mismatches'].append(f'{rel}:{k}:{a.shape}')
npz_scan['output_roles_found'] = dict(npz_scan['output_roles_found'])
npz_scan['sealed_identifier_universe'] = {'sealed_HU_households_all': len(sealed_serials), 'candidate_ids_household_plus_person': int(sealed_ids.size),
                                          'note': 'all housing-unit households hashed to SEALED_ASSESSMENT (superset of the eligible sealed set); person ids enumerated from SERIALNO with 2-digit SPORDER (format verified in check 4)'}
ev['sealed_npz_scan'] = npz_scan
if npz_scan['keys_containing_SEALED'] or npz_scan['output_shape_mismatches'] or npz_scan['reader_score_shape_mismatches'] or 'SEALED_ASSESSMENT' in npz_scan['output_roles_found']:
    problems.append('SEALED key or role-size mismatch in stored arrays')

# ------------------------------------------------------------------ 2. single opening + chronology
lines = [l for l in (PRIV / 'access' / 'DEV_REPORT_ACCESS.jsonl').read_text().splitlines() if l.strip()]
events = [json.loads(l) for l in lines]
openings = [e for e in events if e.get('event') == 'DEV_REPORT_TRUTH_OPENED']
commit_time = {k: ts(git('show', '-s', '--format=%cI', c).strip()) for k, c in COMMITS.items()}
full = {k: git('rev-parse', c).strip() for k, c in COMMITS.items()}
reflog = git('reflog', 'show', '--date=iso-strict', f'refs/remotes/github/{PUB_BRANCH}', check=False)
push_time = None
for l in reflog.splitlines():
    m = re.match(r'^([0-9a-f]+) .*@\{([^}]+)\}: update by push', l)
    if m and full['PUBLIC_RECEIPT'].startswith(m.group(1)):
        push_time = ts(m.group(2))
try:
    lsr = git('ls-remote', 'github', f'refs/heads/{PUB_BRANCH}', timeout=30).split()
    remote_tip = lsr[0] if lsr else None
    remote_contains = (subprocess.run(['git', '-C', str(WT), 'merge-base', '--is-ancestor', full['PUBLIC_RECEIPT'], remote_tip]).returncode == 0) if remote_tip else False
    lsremote = {'performed': True, 'remote_tip': remote_tip, 'remote_tip_contains_7d23e0e': remote_contains}
except Exception as exc:  # network optional
    lsremote = {'performed': False, 'error': type(exc).__name__}
anc = lambda a, b: subprocess.run(['git', '-C', str(WT), 'merge-base', '--is-ancestor', a, b]).returncode == 0
rl_sha = sha(RES / 'REPORTING_LOCK.json')
open_t = ts(openings[0]['utc']) if openings else None
ev['access_log'] = {
    'lines': len(lines), 'openings': len(openings), 'opening_utc': openings[0]['utc'] if openings else None,
    'opening_records_reporting_lock_sha256': openings[0].get('reporting_lock_sha256') if openings else None,
    'current_REPORTING_LOCK_sha256': rl_sha,
    'opening_lock_hash_matches_current_lock': bool(openings) and openings[0].get('reporting_lock_sha256') == rl_sha,
    'commit_times_utc': {k: iso(v) for k, v in commit_time.items()},
    'public_receipt_branches': git('branch', '-a', '--contains', full['PUBLIC_RECEIPT']).split(),
    'public_receipt_push_time_utc_from_local_remote_tracking_reflog': iso(push_time) if push_time else None,
    'git_ls_remote_github': lsremote,
    'opening_after_REPORTING_LOCK_commit': bool(open_t and open_t > commit_time['REPORTING_LOCK']),
    'opening_after_public_receipt_commit': bool(open_t and open_t > commit_time['PUBLIC_RECEIPT']),
    'opening_after_public_receipt_push': bool(open_t and push_time and open_t > push_time),
    'seconds_REPORTING_LOCK_commit_to_opening': (open_t - commit_time['REPORTING_LOCK']).total_seconds() if open_t else None,
    'seconds_receipt_push_to_opening': (open_t - push_time).total_seconds() if (open_t and push_time) else None,
    'ancestry_REPORTING_LOCK_before_results_commit': anc(full['REPORTING_LOCK'], full['DEV_REPORT_RESULTS']),
    'ancestry_lock_chain': anc(full['SCIENCE_LOCK'], full['TRAINING_COMPLETE_LOCK']) and anc(full['TRAINING_COMPLETE_LOCK'], full['SELECTION_LOCK']) and anc(full['SELECTION_LOCK'], full['REPORTING_LOCK']),
    'DEV_REPORT_RESULTS_utc': json.loads((RES / 'DEV_REPORT_RESULTS.json').read_text()).get('utc'),
}
a = ev['access_log']
if not (a['lines'] == 1 and a['openings'] == 1 and a['opening_lock_hash_matches_current_lock'] and a['opening_after_REPORTING_LOCK_commit']
        and a['opening_after_public_receipt_commit'] and a['opening_after_public_receipt_push'] and a['ancestry_REPORTING_LOCK_before_results_commit'] and a['ancestry_lock_chain']):
    problems.append('access log / opening chronology failure')
if lsremote.get('performed') and not lsremote.get('remote_tip_contains_7d23e0e'):
    problems.append('GitHub publication branch does not contain 7d23e0e')

# lock files: current == blob at their commit; publication copies equal research copies
lock_files = {}
pub_tree = git('ls-tree', '-r', '--name-only', full['PUBLIC_RECEIPT']).splitlines()
for name, key in (('SCIENCE_LOCK.json', 'SCIENCE_LOCK'), ('TRAINING_COMPLETE_LOCK.json', 'TRAINING_COMPLETE_LOCK'),
                  ('SELECTION_LOCK.json', 'SELECTION_LOCK'), ('REPORTING_LOCK.json', 'REPORTING_LOCK')):
    cur = sha(RES / name)
    blob = hashlib.sha256(git_bytes('show', f"{full[key]}:results/pcrl_acs_protected_tree_replication_v1/{name}")).hexdigest()
    pub = [p for p in pub_tree if p.endswith('/' + name)]
    pub_sha = hashlib.sha256(git_bytes('show', f"{full['PUBLIC_RECEIPT']}:{pub[0]}")).hexdigest() if pub else None
    lock_files[name] = {'current_sha256': cur, 'blob_at_lock_commit_sha256': blob, 'current_equals_commit_blob': cur == blob,
                        'publication_copy': pub[0] if pub else None, 'publication_copy_equal': (pub_sha == cur) if pub_sha else None}
    if cur != blob or (pub_sha is not None and pub_sha != cur):
        problems.append(f'{name} differs from its lock commit or publication copy')
ev['lock_files'] = lock_files
ev['science_lock_sha_passed_to_extraction'] = {'expected': '1bd11de2cf09577632e8938e98fe728d686a838bb13016a0be43601d97384d23',
                                               'current': lock_files['SCIENCE_LOCK.json']['current_sha256'],
                                               'match': lock_files['SCIENCE_LOCK.json']['current_sha256'] == '1bd11de2cf09577632e8938e98fe728d686a838bb13016a0be43601d97384d23'}
if not ev['science_lock_sha_passed_to_extraction']['match']:
    problems.append('SCIENCE_LOCK hash differs from the one passed to extraction')

# ------------------------------------------------------------------ 3. truth / export files unmodified since extraction
receipt = json.loads((PRIV / 'extraction' / 'EXTRACTION_RECEIPT.json').read_text())
summary = json.loads((RES / 'EXTRACTION_SUMMARY.json').read_text())
exp_files, bad = 0, []
for role, files in receipt['files'].items():
    for kind, rec in files.items():
        exp_files += 1
        cur = sha(WT / rec['path'])
        if cur != rec['sha256'] or cur != summary['export_file_sha256'][role][kind]:
            bad.append(rec['path'])
truth = {k: {'path': v['path'], 'receipt_sha256': v['sha256'], 'current_sha256': sha(WT / v['path'])}
         for k, v in receipt['files']['DEV_REPORT'].items() if k in ('income', 'occupation', 'sex', 'weights')}
ledger = [json.loads(l) for l in (PRIV / 'compute' / 'ledger.jsonl').read_text().splitlines() if l.strip()]
extract_end = max(ts(e['utc']) for e in ledger if e['event'] == 'unit_end' and e['unit_id'].startswith('EXTRACT_') and e.get('status') == 'done')
truth_mtimes = {k: iso(dt.datetime.fromtimestamp(os.stat(WT / v['path']).st_mtime, dt.timezone.utc)) for k, v in truth.items()}
ev['export_integrity'] = {'export_files_checked': exp_files, 'mismatching_files': bad,
                          'DEV_REPORT_TRUTH': {k: dict(v, match=v['receipt_sha256'] == v['current_sha256']) for k, v in truth.items()},
                          'DEV_REPORT_TRUTH_mtime_utc': truth_mtimes, 'extraction_unit_end_utc': iso(extract_end),
                          'truth_mtime_not_after_extraction_end': all(ts(m) <= extract_end + dt.timedelta(seconds=1) for m in truth_mtimes.values())}
if bad or not all(v['receipt_sha256'] == v['current_sha256'] for v in truth.values()):
    problems.append('export/truth file differs from the extraction receipt')

# frozen artifacts: hashes recorded in the stage locks still match
def check_hashes(mapping, base):
    miss, diff = [], []
    for rel, h in mapping.items():
        p = base / rel
        if not p.exists():
            miss.append(rel)
        elif sha(p) != h:
            diff.append(rel)
    return {'files': len(mapping), 'missing': miss, 'differ': diff}


rl = json.loads((RES / 'REPORTING_LOCK.json').read_text())
tc = json.loads((RES / 'TRAINING_COMPLETE_LOCK.json').read_text())
sl = json.loads((RES / 'SELECTION_LOCK.json').read_text())
frozen = {
    'TRAINING_COMPLETE_LOCK.frozen_files_sha256': check_hashes(tc['frozen_files_sha256'], PRIV / 'runs'),
    'SELECTION_LOCK.output_file_sha256': check_hashes(sl['output_file_sha256'], PRIV),
    'REPORTING_LOCK.reader_files_sha256': check_hashes(rl['reader_files_sha256'], PRIV / 'runs'),
    'REPORTING_LOCK.dev_report_scores': {s: sha(PRIV / 'runs' / f'READERS_{s}' / 'DEV_REPORT_SCORES.npz') == rl[f'dev_report_scores_sha256_s{s}'] for s in (0, 1)},
    'REPORTING_LOCK.selection_lock_sha256_matches': rl['selection_lock_sha256'] == lock_files['SELECTION_LOCK.json']['current_sha256'],
    'lock_flags': {'TRAINING_COMPLETE.DEV_REPORT_labels_opened': tc.get('DEV_REPORT_labels_opened'),
                   'SELECTION.DEV_REPORT_labels_opened': sl.get('DEV_REPORT_labels_opened'), 'SELECTION.sealed_assessment_touched': sl.get('sealed_assessment_touched'),
                   'REPORTING.DEV_REPORT_labels_opened': rl.get('DEV_REPORT_labels_opened'), 'REPORTING.sealed_assessment_touched': rl.get('sealed_assessment_touched')},
}
ev['frozen_artifacts'] = frozen
for k in ('TRAINING_COMPLETE_LOCK.frozen_files_sha256', 'SELECTION_LOCK.output_file_sha256', 'REPORTING_LOCK.reader_files_sha256'):
    if frozen[k]['missing'] or frozen[k]['differ']:
        problems.append(f'{k}: frozen files missing/changed')
if not all(frozen['REPORTING_LOCK.dev_report_scores'].values()) or not frozen['REPORTING_LOCK.selection_lock_sha256_matches']:
    problems.append('reader scores or selection lock hash differ from REPORTING_LOCK')
if any(v not in (False,) for v in frozen['lock_flags'].values()):
    problems.append('a stage lock records opened DEV_REPORT labels or touched sealed assessment')

# ------------------------------------------------------------------ 4. ledger ordering vs lock commits
starts = {e['unit_id']: ts(e['utc']) for e in ledger if e['event'] == 'unit_start'}
ends = {e['unit_id']: (ts(e['utc']), e.get('status')) for e in ledger if e['event'] == 'unit_end'}
fit_prefix = ('STAGE_A_', 'STAGE_B_', 'NEURAL_RAW_', 'FARE_', 'FAGTB_')
order = []


def rule(name, ok, detail):
    order.append({'rule': name, 'ok': bool(ok), **detail})
    if not ok:
        problems.append(f'ledger ordering failed: {name}')


sci, trn, sel, rep = commit_time['SCIENCE_LOCK'], commit_time['TRAINING_COMPLETE_LOCK'], commit_time['SELECTION_LOCK'], commit_time['REPORTING_LOCK']
ext = {u: t for u, t in starts.items() if u.startswith('EXTRACT_')}
fits = {u: t for u, t in starts.items() if u.startswith(fit_prefix)}
rule('science lock commit before every extraction unit start', all(t > sci for t in ext.values()),
     {'lock_commit_utc': iso(sci), 'units': {u: iso(t) for u, t in ext.items()}})
rule('science lock commit before every fit unit start', all(t > sci for t in fits.values()),
     {'lock_commit_utc': iso(sci), 'earliest_fit_start': iso(min(fits.values())), 'n_fit_units': len(fits)})
rule('every fit unit ended (done) before the TRAINING_COMPLETE_LOCK commit', all(ends[u][0] < trn and ends[u][1] == 'done' for u in fits),
     {'lock_commit_utc': iso(trn), 'latest_fit_end': iso(max(ends[u][0] for u in fits))})
selu = {u: t for u, t in starts.items() if u.startswith('SELECTION')}
rule('TRAINING_COMPLETE_LOCK commit before SELECTION start; selection ended before SELECTION_LOCK commit',
     all(t > trn for t in selu.values()) and all(ends[u][0] < sel for u in selu),
     {'training_complete_commit_utc': iso(trn), 'selection_units': {u: [iso(t), iso(ends[u][0])] for u, t in selu.items()}, 'selection_lock_commit_utc': iso(sel)})
rdu = {u: t for u, t in starts.items() if u.startswith('READERS_')}
rule('SELECTION_LOCK commit before every READERS start; readers ended before REPORTING_LOCK commit',
     all(t > sel for t in rdu.values()) and all(ends[u][0] < rep for u in rdu),
     {'selection_lock_commit_utc': iso(sel), 'reader_units': {u: [iso(t), iso(ends[u][0])] for u, t in rdu.items()}, 'reporting_lock_commit_utc': iso(rep)})
opu = {u: t for u, t in starts.items() if u.startswith('DEV_REPORT_OPEN')}
rule('exactly one DEV_REPORT opening unit; it starts after the REPORTING_LOCK commit and the receipt push; access event inside the unit window',
     len(opu) == 1 and all(t > rep and (push_time is None or t > push_time) for t in opu.values())
     and all(t <= open_t <= ends[u][0] for u, t in opu.items()),
     {'reporting_lock_commit_utc': iso(rep), 'receipt_push_utc': iso(push_time) if push_time else None,
      'opening_units': {u: [iso(t), iso(ends[u][0])] for u, t in opu.items()}, 'access_event_utc': iso(open_t) if open_t else None})
report_like = [e['unit_id'] for e in ledger if e['event'] == 'unit_start' and any('acs_replication.report' == x for x in e.get('argv', []))]
rule('the report module was started exactly once', len(report_like) == 1, {'units_running_acs_replication.report': report_like})
ev['ledger_ordering'] = order

# ------------------------------------------------------------------ 5. truth references before the opening
mentions = {}
for p in sorted((PRIV / 'compute').rglob('*')):
    if p.is_file() and p.suffix in ('.log', '.json', '.jsonl'):
        txt = p.read_text(errors='replace')
        if 'DEV_REPORT_TRUTH' in txt:
            mentions[str(p.relative_to(PRIV))] = txt.count('DEV_REPORT_TRUTH')
modules_before_opening = sorted({x.split('.')[-1] for e in ledger if e['event'] == 'unit_start' and starts[e['unit_id']] < (open_t or starts[e['unit_id']])
                                 for x in e.get('argv', []) if x.startswith('acs_replication.')})
truth_refs = {}
for f in sorted((WT / 'acs_replication').glob('*.py')):
    t = f.read_text()
    n = t.count('DEV_REPORT_TRUTH') + len(re.findall(r"load_role\('DEV_REPORT',[^)]*(?:income|occupation|sex|weights)", t))
    if n:
        truth_refs[f.name] = n
gate = (WT / 'acs_replication' / 'data.py').read_text()
ev['truth_references'] = {
    'compute_files_mentioning_DEV_REPORT_TRUTH': mentions,
    'modules_started_before_opening': modules_before_opening,
    'code_modules_referencing_DEV_REPORT_truth': truth_refs,
    'pre_opening_modules_referencing_truth': sorted(set(truth_refs) & {m + '.py' for m in modules_before_opening} - {'extract.py'}),
    'loader_gate_present': "DEV_REPORT truth opens only after the committed REPORTING_LOCK" in gate and "the sealed assessment is closed" in gate,
    'synthetic_report_test_redirects_private_paths_and_access_log_to_tmp': all(s in (WT / 'acs_replication' / 'test_report_synthetic.py').read_text()
        for s in ("tempfile.mkdtemp", "D.PRIV, D.RESULTS, registry.RUNS, registry.PRIV = priv, results, runs, priv", "report.ACCESS_LOG = priv / 'access' / 'DEV_REPORT_ACCESS.jsonl'")),
}
if ev['truth_references']['pre_opening_modules_referencing_truth'] or mentions:
    problems.append('a pre-opening unit references DEV_REPORT truth')

out = {'id': 5, 'name': 'CLOSED ASSESSMENT + ACCESS', 'status': 'PASS' if not problems else 'FAIL', 'evidence': ev, 'problems': problems,
       'notes': ('commit/ledger/access timestamps are self-reported by the local machine (second resolution); ordering is additionally '
                 'supported by git ancestry and by the push reflog of the remote-tracking publication ref. DEV_REPORT_TRUTH files were '
                 'hashed (bytes) only; no label value was read in this check.'),
       'sealed_assessment_touched': False}
OUT.write_text(json.dumps(out, indent=2, sort_keys=True, default=str))
print(json.dumps({'status': out['status'], 'problems': problems}, indent=1))
